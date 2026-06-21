"""Gradient guard and trust-clip helpers for B-spline/SU2 runs."""

import math

import numpy as np

SAFE_EVALUATION_STATUSES = {
    "ok",
    "ok_clipped_benign",
    "ok_clipped_weak",
    "benign_clipped_legacy",
}

LAST_EVAL_CACHE_BLOCKED_TRUST_CLIP_CLASSES = (
    "weak_clipped_progress",
    "accepted_clipped_restart",
    "rejected_toxic_clip",
)

ALLOWED_TRUST_CLIP_POLICIES = ("OFF", "ACCEPT_RESTART")

def _trust_clip_options(options=None):
    defaults = {
        "policy": "ACCEPT_RESTART",
        "beta_tol": 1.0e-12,
        "legacy_beta_min": 0.50,
        "severe_beta": 0.50,
        "worsening_tol": 0.05,
        "soft_gnorm_factor": 20.0,
        "bad_patience": 2,
        "bad_window": 5,
        "stag_tol": 1.0e-6,
        "gnorm_floor": 1.0e-14,
        "objective_floor": 1.0e-12,
    }
    defaults.update(dict(options or {}))
    defaults["policy"] = str(defaults["policy"]).strip().upper()
    return defaults

def classify_clipped_trial(
    entry,
    recent_safe_raw_gnorms,
    best_safe_entry,
    anchor_entry,
    recent_level_clip_events,
    options=None,
):
    """Classify one physical evaluation without stopping the SLSQP block."""

    opts = _trust_clip_options(options)
    beta = float(entry.get("beta_eff", 1.0))
    objective = float(entry.get("objective", math.nan))
    gnorm_raw = float(entry.get("gnorm_raw", math.nan))
    beta_tol = float(opts["beta_tol"])
    clipped = bool(entry.get("was_clipped", False)) or beta < 1.0 - beta_tol
    best_objective = (
        None if best_safe_entry is None else float(best_safe_entry["objective"])
    )
    anchor_objective = (
        None if anchor_entry is None else float(anchor_entry["objective"])
    )
    if anchor_objective is None:
        anchor_objective = best_objective
    gnorm_floor = max(float(opts.get("gnorm_floor", 1.0e-14)), 1.0e-30)
    obj_floor = float(opts.get("objective_floor", 1.0e-12))
    if not np.isfinite(obj_floor) or obj_floor <= 0.0:
        obj_floor = 1.0e-12
    improvement_rel = (
        None
        if anchor_objective is None or not np.isfinite(anchor_objective)
        else (anchor_objective - objective) / max(abs(anchor_objective), obj_floor)
    )
    relative_worsening = (
        None
        if best_objective is None or not np.isfinite(best_objective)
        else (objective - best_objective) / max(abs(best_objective), obj_floor)
    )
    best_improvement_rel = (
        None
        if best_objective is None or not np.isfinite(best_objective)
        else (best_objective - objective) / max(abs(best_objective), obj_floor)
    )
    stag_tol = float(opts["stag_tol"])
    significant_improvement = bool(
        improvement_rel is not None and improvement_rel >= stag_tol
    )
    significant_best_improvement = bool(
        best_improvement_rel is not None and best_improvement_rel >= stag_tol
    )
    weak_improvement = bool(
        improvement_rel is not None and abs(improvement_rel) < stag_tol
    )

    good_gnorms = [
        float(value)
        for value in recent_safe_raw_gnorms
        if np.isfinite(value) and float(value) > 0.0
    ]
    gnorm_reference = (
        max(float(np.median(good_gnorms[-5:])), gnorm_floor)
        if good_gnorms
        else None
    )
    gnorm_ratio = (
        gnorm_raw / gnorm_reference
        if gnorm_reference is not None and np.isfinite(gnorm_raw)
        else None
    )
    objective_finite = bool(np.isfinite(objective))
    gradient_finite = bool(np.isfinite(gnorm_raw))

    recent = list(recent_level_clip_events or [])
    current_weak_event = bool(clipped and weak_improvement)
    plateau_events = recent[-max(0, int(opts["bad_window"]) - 1) :] + [
        {"clipped": clipped, "weak_improvement": current_weak_event}
    ]
    clipped_stagnation_plateau = (
        sum(
            1
            for event in plateau_events
            if event.get("clipped") and event.get("weak_improvement")
        )
        >= int(opts["bad_patience"])
    )

    toxic_reasons = []
    accepted_reasons = []
    if clipped:
        if not objective_finite:
            toxic_reasons.append("nonfinite_objective")
        if not gradient_finite:
            toxic_reasons.append("nonfinite_gradient")
        if (
            relative_worsening is not None
            and relative_worsening > float(opts["worsening_tol"])
        ):
            toxic_reasons.append("worsening_vs_best_safe")
        if improvement_rel is not None and improvement_rel < -stag_tol:
            toxic_reasons.append("worsening_vs_anchor")
        if (
            gnorm_ratio is not None
            and gnorm_ratio > float(opts["soft_gnorm_factor"])
        ):
            toxic_reasons.append("soft_raw_gradient_ratio")
        if beta < float(opts["severe_beta"]) and not (
            significant_improvement or significant_best_improvement
        ):
            toxic_reasons.append("severe_clip_without_strong_improvement")
        if clipped_stagnation_plateau:
            toxic_reasons.append("clipped_stagnation_plateau")

    base_toxic = bool(toxic_reasons)
    toxic_events = recent[-max(0, int(opts["bad_window"]) - 1) :] + [
        {"toxic": bool(clipped and base_toxic)}
    ]
    toxic_repeated = (
        bool(clipped and base_toxic)
        and sum(1 for event in toxic_events if event.get("toxic"))
        >= int(opts["bad_patience"])
    )
    if toxic_repeated:
        toxic_reasons.append("toxic_clipped_repeated")

    if not clipped:
        classification = "not_clipped"
        accepted_reasons.append("unclipped")
    elif toxic_reasons:
        classification = "rejected_toxic_clip"
    elif beta >= float(opts["legacy_beta_min"]) and significant_improvement:
        classification = "benign_clipped_legacy"
        accepted_reasons.extend(["moderate_beta", "significant_improvement", "sane_gradient"])
    elif beta < float(opts["legacy_beta_min"]) and (
        significant_improvement or significant_best_improvement
    ):
        classification = "accepted_clipped_restart"
        accepted_reasons.extend(["useful_severe_clip", "significant_improvement", "sane_gradient"])
    else:
        # A single weak clipped step is observed but not rejected until the
        # sliding-window plateau condition becomes active.
        classification = "weak_clipped_progress"
        accepted_reasons.append("weak_clipped_progress")

    diagnostics = {
        "beta_eff": beta,
        "objective": objective,
        "best_objective": best_objective,
        "anchor_objective": anchor_objective,
        "improvement_rel": improvement_rel,
        "relative_worsening": relative_worsening,
        "gnorm_raw": gnorm_raw,
        "objective_floor": obj_floor,
        "gnorm_floor": gnorm_floor,
        "gnorm_reference": gnorm_reference,
        "gnorm_ratio": gnorm_ratio,
        "clipped": clipped,
        "significant_improvement": significant_improvement,
        "weak_improvement": weak_improvement,
        "clipped_stagnation_plateau": clipped_stagnation_plateau,
        "toxic_repeated": toxic_repeated,
        "toxic_reasons": toxic_reasons,
        "accepted_reasons": accepted_reasons,
    }
    return classification, diagnostics

def gradient_guard_triggered(
    entry,
    recent_safe_raw_gnorms,
    *,
    factor=100.0,
    window=5,
    min_history=3,
    floor=1.0e-14,
):
    """Detect a pathological pre-beta, pre-optimizer-scaled gradient."""

    gnorm_raw = float(entry["gnorm_raw"])
    if not np.isfinite(gnorm_raw):
        return True, {
            "reason": "nonfinite_raw_gradient",
            "gnorm_raw": gnorm_raw,
            "reference": None,
            "ratio": np.inf,
        }

    good = [
        float(value)
        for value in recent_safe_raw_gnorms
        if np.isfinite(value) and float(value) > 0.0
    ]
    if len(good) < int(min_history):
        return False, {
            "reason": "insufficient_history",
            "gnorm_raw": gnorm_raw,
            "reference": None,
            "ratio": None,
        }

    window = max(1, int(window))
    reference = max(float(np.median(good[-window:])), float(floor))
    ratio = gnorm_raw / reference
    if ratio > float(factor):
        return True, {
            "reason": "raw_gradient_explosion",
            "gnorm_raw": gnorm_raw,
            "reference": reference,
            "ratio": ratio,
        }
    return False, {
        "reason": "ok",
        "gnorm_raw": gnorm_raw,
        "reference": reference,
        "ratio": ratio,
    }
