#!/usr/bin/env python

"""Shared online progressive-refinement trigger engine."""

import csv
import math
import os
import sys


class RefinementTriggered(Exception):
    pass


ALLOWED_TRIGGERS = (
    "MAX_ITER",
    "SLOPE_EFFICIENCY_TRIGGER",
    "SLOPE_EFFICIENCY_FILTERED",
    "SLOPE_EFFICIENCY_BEST_LOG",
    "STAGNATION_TRIGGER",
    "ECONOMIC_TRIGGER",
    "BATCH_STABILITY",
    "TRAJECTORY_READY",
)


# Internal calibration of TRAJECTORY_READY.  These are intentionally not
# exposed as user knobs while the criterion is being validated across cases.
_TRAJECTORY_EMA_BETA = 0.5
_TRAJECTORY_SATURATION_RATIO = 0.2
_TRAJECTORY_MIN_RATES = 4
_TRAJECTORY_ENRICHMENT_GAIN = 0.2
_TRAJECTORY_FRONTIER_MARGIN = 0.02
_TRAJECTORY_COMPATIBLE_SAMPLES = 2
_TRAJECTORY_EPS = 1.0e-14


def trigger_prefix(project):
    return getattr(project, "progressive_label", "PROGRESSIVE_HH")


def _write(project, message):
    sys.stdout.write(f"[{trigger_prefix(project)}] {message}\n")


def _init_trigger_state(project):
    if not hasattr(project, "trigger_state") or project.trigger_state is None:
        project.trigger_state = {
            "accepted_history": [],
            "best_obj": None,
            "sat_counter": 0,
        }


def _update_filtered_history(accepted_history, value, filter_tol):
    eps = 1.0e-14

    if not accepted_history:
        accepted_history.append(value)
        return True

    ref = accepted_history[-1]

    if value <= ref:
        accepted_history.append(value)
        return True

    rel_wors = (value - ref) / max(abs(ref), eps)

    if rel_wors <= filter_tol:
        accepted_history.append(value)
        return True

    return False


def _compute_smoothed_history(history, window):
    if window <= 1:
        return list(history)

    smooth = []
    for i in range(window - 1, len(history)):
        avg = sum(history[i - window + 1 : i + 1]) / float(window)
        smooth.append(avg)
    return smooth


def _trigger_can_fire(project, opts):
    warmup_iter = int(opts.get("warmup_iter", 0))
    return len(getattr(project, "trigger_history", [])) > warmup_iter


def _log_trigger_warmup(name, project, opts):
    warmup_iter = int(opts.get("warmup_iter", 0))
    _write(
        project,
        f"{name} | "
        f"warmup guard active ({len(project.trigger_history)}/{warmup_iter}); "
        "state updated, trigger suppressed",
    )


def _check_slope_trigger(project, obj_value, opts):
    _init_trigger_state(project)

    can_fire = _trigger_can_fire(project, opts)

    w = max(1, int(opts.get("window", 1)))
    r = float(opts.get("tol", 0.2))
    filter_tol = float(opts.get("filter_tol", 0.02))

    accepted_history = project.trigger_state["accepted_history"]
    accepted_now = _update_filtered_history(accepted_history, obj_value, filter_tol)

    if not accepted_now:
        _write(
            project,
            "SLOPE_EFFICIENCY ONLINE | large worsening ignored in filtered history",
        )
        return

    smooth = _compute_smoothed_history(accepted_history, w)

    if len(smooth) < 2:
        return

    slopes = []
    for i in range(1, len(smooth)):
        dj = smooth[i - 1] - smooth[i]
        slopes.append(dj)

    if not slopes:
        return

    current_slope = slopes[-1]

    if current_slope <= 0.0:
        _write(
            project,
            "SLOPE_EFFICIENCY ONLINE | last accepted step not improving, skip trigger check",
        )
        return

    positive_slopes = [s for s in slopes if s > 0.0]

    if not positive_slopes:
        return

    max_slope = max(positive_slopes)

    if max_slope <= 1.0e-16:
        _write(
            project,
            "SLOPE_EFFICIENCY ONLINE | flat positive history, skip trigger check",
        )
        return

    ratio = current_slope / max_slope

    if not can_fire:
        _log_trigger_warmup("SLOPE_EFFICIENCY ONLINE", project, opts)
        return

    _write(
        project,
        f"SLOPE_EFFICIENCY ONLINE | ratio={ratio:.6e} threshold={r:.6e}",
    )

    if ratio < r:
        project.refinement_triggered = True
        _write(project, "Efficiency trigger -> STOP")
        raise RefinementTriggered()


def _check_slope_best_log_trigger(project, obj_value, opts):
    _init_trigger_state(project)

    state = project.trigger_state
    state.setdefault("last_log_best", None)
    state.setdefault("improvements", [])
    state.setdefault("max_slope_seen", 0.0)
    state.setdefault("bad_count", 0)

    window = max(1, int(opts.get("window", 1)))
    tol = float(opts.get("tol", 0.2))
    eps = float(opts.get("eps", 1.0e-300))
    patience = max(1, int(opts.get("patience", 1)))

    best_obj = state.get("best_obj", None)
    if best_obj is None or obj_value < best_obj:
        best_obj = obj_value
        state["best_obj"] = best_obj

    y_k = math.log(max(best_obj, eps))

    if state["last_log_best"] is None:
        state["last_log_best"] = y_k
        return

    delta_k = state["last_log_best"] - y_k
    if delta_k < 0.0:
        delta_k = 0.0
    state["improvements"].append(delta_k)

    if len(state["improvements"]) < window:
        state["last_log_best"] = y_k
        return

    recent = state["improvements"][-window:]
    recent_slope = sum(recent) / float(window)

    if recent_slope > 0.0:
        state["max_slope_seen"] = max(state["max_slope_seen"], recent_slope)

    can_fire = _trigger_can_fire(project, opts)

    if not can_fire:
        state["last_log_best"] = y_k
        _log_trigger_warmup("SLOPE_EFFICIENCY_BEST_LOG", project, opts)
        return

    ratio = recent_slope / max(state["max_slope_seen"], eps)

    _write(
        project,
        "SLOPE_EFFICIENCY_BEST_LOG | "
        f"ratio={ratio:.6e} threshold={tol:.6e} "
        f"bad_count={state['bad_count']}/{patience}",
    )

    if ratio < tol:
        state["bad_count"] += 1
    else:
        state["bad_count"] = 0

    state["last_log_best"] = y_k

    if state["bad_count"] >= patience:
        project.refinement_triggered = True
        _write(project, "SLOPE_EFFICIENCY_BEST_LOG -> STOP")
        raise RefinementTriggered()


def _check_economic_trigger(project, obj_value, opts):
    """Robust-saturation condition of the economic trigger.

    Fires when the recent log-rate of the best-so-far objective falls below
    alpha times a transient-free reference rate:

        r_now < alpha * r_ref

    r_now  = mean drop of log(best obj) per evaluation over the last k;
    r_ref  = median of the rolling rates whose windows do not touch the
             first 2 evaluations of the level (the initial transient can
             never enter the reference — the structural defect of the
             slope criterion);
    r_ref is usable only after min_ref clean rates exist.

    Forced exits: level length >= n_max evaluations, or r_now below an
    absolute floor once the reference is usable.

    Calibrated offline on the 8 W2/4/6/8 runs (replay 26/07/2026):
    alpha=0.2, k=4, dwell=4, min_ref=4, patience=2.
    """
    _init_trigger_state(project)

    state = project.trigger_state
    state.setdefault("econ_log_best", [])
    state.setdefault("econ_clean_rates", [])
    state.setdefault("econ_bad_count", 0)

    alpha = float(opts.get("tol", 0.2))
    k = max(1, int(opts.get("window", 4)))
    dwell = max(1, int(opts.get("dwell", 4)))
    min_ref = max(1, int(opts.get("min_ref", 4)))
    patience = max(1, int(opts.get("patience", 2)))
    n_max = int(opts.get("n_max", 40))
    rate_floor = float(opts.get("rate_floor", 1.0e-4))
    eps = float(opts.get("eps", 1.0e-300))

    best_obj = state.get("best_obj", None)
    if best_obj is None or obj_value < best_obj:
        best_obj = obj_value
        state["best_obj"] = best_obj

    log_best = state["econ_log_best"]
    log_best.append(math.log(max(best_obj, eps)))
    i = len(log_best)

    if i <= k:
        return

    r_now = (log_best[i - k - 1] - log_best[i - 1]) / float(k)

    # the window [i-k, i] is transient-free if it starts at evaluation 3+
    if i - k >= 3:
        state["econ_clean_rates"].append(r_now)

    clean = state["econ_clean_rates"]

    if i < dwell:
        return

    if not _trigger_can_fire(project, opts):
        _log_trigger_warmup("ECONOMIC_TRIGGER", project, opts)
        return

    if i >= n_max:
        project.refinement_triggered = True
        _write(project, f"ECONOMIC_TRIGGER | forced exit: n_max={n_max} reached -> STOP")
        raise RefinementTriggered()

    if len(clean) < min_ref:
        _write(
            project,
            f"ECONOMIC_TRIGGER | reference not ready ({len(clean)}/{min_ref} clean rates)",
        )
        return

    if r_now < rate_floor:
        project.refinement_triggered = True
        _write(
            project,
            f"ECONOMIC_TRIGGER | forced exit: rate {r_now:.6e} < floor {rate_floor:.6e} -> STOP",
        )
        raise RefinementTriggered()

    sorted_rates = sorted(clean)
    n = len(sorted_rates)
    if n % 2 == 1:
        r_ref = sorted_rates[n // 2]
    else:
        r_ref = 0.5 * (sorted_rates[n // 2 - 1] + sorted_rates[n // 2])

    if r_ref <= 0.0:
        _write(project, "ECONOMIC_TRIGGER | non-positive reference rate, skip check")
        return

    saturated = r_now < alpha * r_ref

    if saturated:
        state["econ_bad_count"] += 1
    else:
        state["econ_bad_count"] = 0

    _write(
        project,
        "ECONOMIC_TRIGGER | "
        f"r_now={r_now:.6e} r_ref={r_ref:.6e} alpha={alpha:.3f} "
        f"ratio={r_now / r_ref:.6e} "
        f"bad_count={state['econ_bad_count']}/{patience}",
    )

    if state["econ_bad_count"] >= patience:
        project.refinement_triggered = True
        _write(project, "ECONOMIC_TRIGGER | robust saturation -> STOP")
        raise RefinementTriggered()


def _check_stagnation_trigger(project, obj_value, opts):
    _init_trigger_state(project)
    can_fire = _trigger_can_fire(project, opts)

    eps = 1.0e-14
    stag_tol = float(opts.get("stag_tol", 1.0e-3))
    stag_band = float(opts.get("stag_band", 0.02))
    stag_window = int(opts.get("stag_window", 3))

    best_obj = project.trigger_state["best_obj"]
    sat_counter = project.trigger_state["sat_counter"]

    if best_obj is None:
        project.trigger_state["best_obj"] = obj_value
        project.trigger_state["sat_counter"] = 0
        if not can_fire:
            _log_trigger_warmup("STAGNATION ONLINE", project, opts)
        return

    if obj_value < best_obj:
        improvement = (best_obj - obj_value) / max(abs(best_obj), eps)
        project.trigger_state["best_obj"] = obj_value

        if improvement > stag_tol:
            project.trigger_state["sat_counter"] = 0
            _write(
                project,
                f"STAGNATION ONLINE | significant new best, reset counter (impr={improvement:.6e})",
            )
        else:
            project.trigger_state["sat_counter"] = sat_counter + 1
            _write(
                project,
                "STAGNATION ONLINE | "
                f"small new best, counter={project.trigger_state['sat_counter']} "
                f"(impr={improvement:.6e}, tol={stag_tol:.6e})",
            )
    else:
        gap = (obj_value - best_obj) / max(abs(best_obj), eps)

        if gap < stag_band:
            project.trigger_state["sat_counter"] = sat_counter + 1
            _write(
                project,
                "STAGNATION ONLINE | "
                f"near best, counter={project.trigger_state['sat_counter']} "
                f"(gap={gap:.6e}, band={stag_band:.6e})",
            )
        else:
            project.trigger_state["sat_counter"] = sat_counter
            _write(
                project,
                "STAGNATION ONLINE | "
                f"outside band, ignored large oscillation "
                f"(counter={project.trigger_state['sat_counter']}, "
                f"gap={gap:.6e}, band={stag_band:.6e})",
            )

    if not can_fire:
        _log_trigger_warmup("STAGNATION ONLINE", project, opts)
        return

    if project.trigger_state["sat_counter"] >= stag_window:
        project.refinement_triggered = True
        _write(project, "Stagnation trigger -> STOP")
        raise RefinementTriggered()


def check_project_trigger(project, obj_value, opts=None):
    opts = getattr(project, "trigger_opts", None) if opts is None else opts
    if not opts:
        return

    trigger = str(opts.get("trigger", "")).upper()

    if trigger in ("SLOPE_EFFICIENCY_TRIGGER", "SLOPE_EFFICIENCY_FILTERED"):
        _check_slope_trigger(project, obj_value, opts)
    elif trigger == "SLOPE_EFFICIENCY_BEST_LOG":
        _check_slope_best_log_trigger(project, obj_value, opts)
    elif trigger == "STAGNATION_TRIGGER":
        _check_stagnation_trigger(project, obj_value, opts)
    elif trigger == "ECONOMIC_TRIGGER":
        _check_economic_trigger(project, obj_value, opts)
    # BATCH_STABILITY and TRAJECTORY_READY are checked only from
    # record_gradient_and_check (post-adjoint).


def record_objective_and_check(project, obj_value, opts=None):
    if not hasattr(project, "trigger_history") or project.trigger_history is None:
        project.trigger_history = []
    project.trigger_history.append(obj_value)
    check_project_trigger(project, obj_value, opts=opts)


# ---------------------------------------------------------------------------
#  Post-adjoint adaptive-ranking triggers
# ---------------------------------------------------------------------------


def _batch_key(scoring_result):
    """Return an ordered tuple of (side, x) from selected_candidates.

    Raises ValueError if 'selected_candidates' is absent (contract violation).
    Returns an empty tuple for an empty batch (caller decides whether to skip).
    Rounds x to 12 decimal places; preserves scorer order (sequential greedy).
    """
    if "selected_candidates" not in scoring_result:
        raise ValueError(
            "BATCH_STABILITY scorer contract violation: "
            "'selected_candidates' key missing from result"
        )
    selected = scoring_result["selected_candidates"]
    return tuple(
        (str(c.get("side", "")).upper(), round(float(c.get("x", 0.0)), 12))
        for c in selected
    )


def _write_batch_stability_csv_row(project, row):
    """Append one row to the batch-stability CSV log if a log path is set."""
    log_path = getattr(project, "batch_stability_log_path", None)
    if not log_path:
        return
    fieldnames = ["call_idx", "obj_value", "armed", "history_len", "batch"]
    write_header = not os.path.exists(log_path)
    try:
        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except Exception as exc:
        _write(project, f"BATCH_STABILITY | CSV log write failed: {exc!r}")


def _check_batch_stability_trigger(project, obj_value, opts):
    """Fire when k consecutive adjoint-best re-scorings yield the same candidate batch.

    State machine (all state updates are transactional — no side-effects on error):
      1. Skip if not a new best.
      2. Call the scorer. On exception: log and skip WITHOUT updating best_obj.
      3. Validate the 'selected_candidates' contract; raise on violation.
      4. Skip (with log) if the batch is empty.
      5. Arm on the first change in batch.
      6. Fire when the last k history entries are identical.

    The batch that caused the fire is stored in state['trigger_batch'] for
    consistency verification at refine time.
    """
    _init_trigger_state(project)
    state = project.trigger_state
    state.setdefault("batch_history", [])
    state.setdefault("batch_armed", False)
    state.setdefault("batch_call_idx", 0)
    state.setdefault("trigger_batch", None)

    k = max(2, int(opts.get("k", 3)))

    if obj_value is None:
        _write(project, "BATCH_STABILITY | no objective recorded yet, skip")
        return

    best_obj = state.get("best_obj", None)
    is_new_best = best_obj is None or float(obj_value) < float(best_obj)

    if not is_new_best:
        _write(
            project,
            f"BATCH_STABILITY | not a new best "
            f"({obj_value:.6e} >= {best_obj:.6e}), skip",
        )
        return

    scorer_fn = getattr(project, "batch_stability_scorer_fn", None)
    if scorer_fn is None:
        _write(project, "BATCH_STABILITY | no scorer_fn attached, trigger inactive")
        state["best_obj"] = float(obj_value)
        return

    # Run the scorer; hold off updating best_obj until we have a valid result.
    try:
        result = scorer_fn()
    except Exception as exc:
        _write(project, f"BATCH_STABILITY | scorer raised {exc!r}, sample skipped")
        return

    # Contract: selected_candidates must be present (fail-fast on violation).
    batch_key = _batch_key(result)

    # Empty batch: skip with explicit log (not an error, but not a valid sample).
    if not batch_key:
        _write(project, "BATCH_STABILITY | scorer returned empty batch, sample skipped")
        return

    # All checks passed — now it is safe to commit the state update.
    state["best_obj"] = float(obj_value)
    state["batch_call_idx"] += 1
    history = state["batch_history"]

    if history and batch_key != history[-1]:
        state["batch_armed"] = True

    history.append(batch_key)

    _write(
        project,
        f"BATCH_STABILITY | best={state['best_obj']:.6e} armed={state['batch_armed']} "
        f"history_len={len(history)} k={k} batch={batch_key}",
    )

    _write_batch_stability_csv_row(
        project,
        {
            "call_idx": state["batch_call_idx"],
            "obj_value": f"{state['best_obj']:.10e}",
            "armed": int(state["batch_armed"]),
            "history_len": len(history),
            "batch": str(list(batch_key)),
        },
    )

    if not state["batch_armed"]:
        _write(project, "BATCH_STABILITY | waiting for first batch change to arm")
        return

    if len(history) < k:
        _write(
            project,
            f"BATCH_STABILITY | armed, waiting for {k} observations "
            f"({len(history)} so far)",
        )
        return

    last_k = history[-k:]
    if len(set(last_k)) == 1:
        state["trigger_batch"] = batch_key
        # Pin the level transition to the exact major iterate whose adjoint
        # produced the stable batch.  This must never be replaced by a later
        # objective-only line-search trial.
        trigger_dv_values = getattr(project, "last_obj_grad_x_full", None)
        if trigger_dv_values is None:
            trigger_dv_values = getattr(project, "last_obj_grad_x", None)
        state["trigger_dv_values"] = (
            None
            if trigger_dv_values is None
            else [float(value) for value in trigger_dv_values]
        )
        trigger_reduced = getattr(project, "last_obj_grad_x", None)
        state["trigger_reduced_dv_values"] = (
            None
            if trigger_reduced is None
            else [float(value) for value in trigger_reduced]
        )
        state["trigger_design_folder"] = getattr(
            project,
            "last_obj_grad_design_folder",
            None,
        )
        project.refinement_dv_values = state["trigger_dv_values"]
        project.refinement_triggered = True
        _write(project, f"BATCH_STABILITY | {k} consecutive identical batches -> STOP")
        raise RefinementTriggered()


def _trajectory_candidate_key(candidate):
    """Return the stable identity of one sequential refinement candidate."""

    if not isinstance(candidate, dict) or "side" not in candidate or "x" not in candidate:
        raise ValueError(
            "TRAJECTORY_READY scorer contract violation: every selected "
            "candidate requires 'side' and 'x'"
        )
    return (
        str(candidate["side"]).upper(),
        round(float(candidate["x"]), 12),
    )


def _trajectory_candidate_score(candidate):
    for field in ("indicator", "score_net"):
        if field not in candidate or candidate[field] in (None, ""):
            continue
        try:
            value = float(candidate[field])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    raise ValueError(
        "TRAJECTORY_READY scorer contract violation: candidate requires a "
        "finite 'indicator' or 'score_net'"
    )


def _trajectory_candidate_is_admissible(candidate):
    value = candidate.get("admissible", True)
    if isinstance(value, str):
        return value.strip().upper() not in ("NO", "FALSE", "0", "OFF", "")
    return bool(value)


def _trajectory_candidate_rank(candidate):
    value = candidate.get("rank", None)
    if value in (None, ""):
        return None
    try:
        rank = int(value)
    except (TypeError, ValueError):
        return None
    return rank if rank > 0 else None


def _trajectory_candidate_step(candidate, fallback=None):
    value = candidate.get("insertion_step", fallback)
    if value in (None, ""):
        raise ValueError(
            "TRAJECTORY_READY scorer contract violation: candidate requires "
            "'insertion_step'"
        )
    try:
        step = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "TRAJECTORY_READY scorer contract violation: invalid "
            f"insertion_step={value!r}"
        ) from exc
    if step < 1:
        raise ValueError(
            "TRAJECTORY_READY scorer contract violation: insertion_step must "
            f"be positive, got {step}"
        )
    return step


def _trajectory_selected_candidates(scoring_result):
    if not isinstance(scoring_result, dict):
        raise ValueError(
            "TRAJECTORY_READY scorer contract violation: scorer result must be a dict"
        )
    if "selected_candidates" not in scoring_result:
        raise ValueError(
            "TRAJECTORY_READY scorer contract violation: "
            "'selected_candidates' key missing from result"
        )
    selected = scoring_result["selected_candidates"]
    if not isinstance(selected, (list, tuple)):
        raise ValueError(
            "TRAJECTORY_READY scorer contract violation: "
            "'selected_candidates' must be a sequence"
        )
    return list(selected)


def _trajectory_enrichment_gain(selected):
    """Measure the relative residual-energy captured by the proposed batch."""

    first = selected[0]
    last = selected[-1]
    try:
        energy_current = float(first["energy_current"])
        energy_candidate = float(last["energy_candidate"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "TRAJECTORY_READY scorer contract violation: selected batch "
            "requires finite energy_current/energy_candidate values from "
            "VIRTUAL_TANGENT scoring"
        ) from exc
    if not math.isfinite(energy_current) or not math.isfinite(energy_candidate):
        raise ValueError(
            "TRAJECTORY_READY scorer contract violation: non-finite "
            "energy_current/energy_candidate"
        )
    if energy_current <= _TRAJECTORY_EPS:
        raise ValueError(
            "TRAJECTORY_READY cannot normalize enrichment: "
            f"energy_current={energy_current:.6e}"
        )
    return (energy_candidate - energy_current) / max(
        abs(energy_current),
        _TRAJECTORY_EPS,
    )


def _trajectory_frontier_snapshot(scoring_result, selected, batch_key):
    """Build the ranked frontier used for margins and crossover detection."""

    if "raw_candidates" not in scoring_result:
        raise ValueError(
            "TRAJECTORY_READY scorer contract violation: "
            "'raw_candidates' key missing from result"
        )
    raw_candidates = scoring_result["raw_candidates"]
    if not isinstance(raw_candidates, (list, tuple)):
        raise ValueError(
            "TRAJECTORY_READY scorer contract violation: "
            "'raw_candidates' must be a sequence"
        )

    selected_by_step = {}
    for candidate in selected:
        if not _trajectory_candidate_is_admissible(candidate):
            raise ValueError(
                "TRAJECTORY_READY scorer contract violation: selected "
                "candidate is not admissible"
            )
        step = _trajectory_candidate_step(candidate)
        if step in selected_by_step:
            raise ValueError(
                "TRAJECTORY_READY scorer contract violation: duplicate selected "
                f"insertion_step={step}"
            )
        _trajectory_candidate_score(candidate)
        selected_by_step[step] = candidate

    raw_by_step = {}
    for candidate in raw_candidates:
        if not isinstance(candidate, dict):
            raise ValueError(
                "TRAJECTORY_READY scorer contract violation: every raw "
                "candidate must be a dict"
            )
        if not _trajectory_candidate_is_admissible(candidate):
            continue
        _trajectory_candidate_score(candidate)
        step = _trajectory_candidate_step(candidate)
        _trajectory_candidate_key(candidate)
        raw_by_step.setdefault(step, []).append(candidate)

    steps = {}
    margins = []
    all_margins_available = True

    for step in sorted(selected_by_step):
        chosen = selected_by_step[step]
        chosen_key = _trajectory_candidate_key(chosen)
        chosen_score = _trajectory_candidate_score(chosen)
        rows = list(raw_by_step.get(step, []))
        if chosen_key not in {
            _trajectory_candidate_key(candidate) for candidate in rows
        }:
            raise ValueError(
                "TRAJECTORY_READY scorer contract violation: selected "
                f"candidate {chosen_key} is absent from raw insertion step {step}"
            )

        # Both production sequential scorers assign rank only after the same
        # interval reduction used for the actual choice.  Prefer precisely
        # that ranked set so a within-interval near-duplicate cannot create a
        # spurious near tie.
        ranked_rows = [
            candidate
            for candidate in rows
            if _trajectory_candidate_rank(candidate) is not None
        ]
        if ranked_rows:
            rows = ranked_rows
        else:
            interval_winners = [
                candidate
                for candidate in rows
                if bool(candidate.get("interval_winner", False))
            ]
            if interval_winners:
                rows = interval_winners

        by_key = {}
        for candidate in rows + [chosen]:
            key = _trajectory_candidate_key(candidate)
            score = _trajectory_candidate_score(candidate)
            existing = by_key.get(key)
            if existing is None or score > existing["score"]:
                by_key[key] = {
                    "score": score,
                    "rank": _trajectory_candidate_rank(candidate),
                }

        ordered = sorted(
            by_key.items(),
            key=lambda item: (
                -item[1]["score"],
                item[0][0],
                item[0][1],
            ),
        )
        scores = {}
        ranks = {}
        for derived_rank, (key, values) in enumerate(ordered, start=1):
            scores[key] = values["score"]
            ranks[key] = (
                values["rank"]
                if values["rank"] is not None
                else derived_rank
            )

        runner_scores = [
            score for key, score in scores.items() if key != chosen_key
        ]
        if runner_scores:
            runner_score = max(runner_scores)
            margin = (chosen_score - runner_score) / max(
                abs(chosen_score),
                _TRAJECTORY_EPS,
            )
            margins.append(margin)
        else:
            margin = None
            all_margins_available = False

        steps[step] = {
            "selected": chosen_key,
            "scores": scores,
            "ranks": ranks,
            "margin": margin,
        }

    return {
        "batch": batch_key,
        "steps": steps,
        "min_margin": min(margins) if margins else None,
        "all_margins_available": all_margins_available,
    }


def _trajectory_resolved_crossover(frontier_history, current):
    """Detect a challenger that closes its gap and then becomes the winner.

    Only the first changed insertion is tested.  Later sequential insertions
    are allowed to change as a consequence of that upstream crossover.
    """

    if len(frontier_history) < 2:
        return False

    older = frontier_history[-2]
    previous = frontier_history[-1]
    current_steps = current["steps"]
    previous_steps = previous["steps"]
    older_steps = older["steps"]
    all_steps = sorted(set(current_steps) | set(previous_steps))
    changed_steps = [
        step
        for step in all_steps
        if current_steps.get(step, {}).get("selected")
        != previous_steps.get(step, {}).get("selected")
    ]
    if not changed_steps:
        return False

    step = changed_steps[0]
    if step not in current_steps or step not in previous_steps or step not in older_steps:
        return False

    # The sequential basis entering this insertion must be identical in all
    # three observations; otherwise the score-gap trend is not comparable.
    for prefix_step in sorted(s for s in all_steps if s < step):
        prefix = [
            snapshot["steps"].get(prefix_step, {}).get("selected")
            for snapshot in (older, previous, current)
        ]
        if prefix[0] is None or len(set(prefix)) != 1:
            return False

    new_winner = current_steps[step]["selected"]
    old_winner = previous_steps[step]["selected"]
    if old_winner is None or new_winner is None or new_winner == old_winner:
        return False
    if older_steps[step]["selected"] != old_winner:
        return False

    # Require the new winner to have been the actual runner-up in the two
    # preceding comparable frontiers.
    if previous_steps[step]["ranks"].get(new_winner) != 2:
        return False
    if older_steps[step]["ranks"].get(new_winner) != 2:
        return False
    if current_steps[step]["ranks"].get(new_winner) != 1:
        return False

    try:
        older_old = float(older_steps[step]["scores"][old_winner])
        older_new = float(older_steps[step]["scores"][new_winner])
        previous_old = float(previous_steps[step]["scores"][old_winner])
        previous_new = float(previous_steps[step]["scores"][new_winner])
        current_old = float(current_steps[step]["scores"][old_winner])
        current_new = float(current_steps[step]["scores"][new_winner])
    except (KeyError, TypeError, ValueError):
        return False

    older_gap = (older_old - older_new) / max(abs(older_old), _TRAJECTORY_EPS)
    previous_gap = (previous_old - previous_new) / max(
        abs(previous_old),
        _TRAJECTORY_EPS,
    )
    if older_gap < 0.0 or previous_gap < 0.0:
        return False
    if not previous_gap < older_gap - 1.0e-12:
        return False

    crossover_tol = 1.0e-12 * max(
        1.0,
        abs(current_old),
        abs(current_new),
    )
    return current_new >= current_old - crossover_tol


def _trajectory_saturation_preview(state, obj_value):
    previous_best = state.get("trajectory_best_obj", None)
    if previous_best is None:
        return {
            "best_obj": float(obj_value),
            "rate": None,
            "rate_ema": None,
            "rate_ref": None,
            "rate_count": 0,
            "ratio": None,
            "saturated": False,
        }

    if previous_best <= 0.0 or float(obj_value) <= 0.0:
        raise ValueError(
            "TRAJECTORY_READY log-rate requires a positive objective; "
            f"previous_best={previous_best!r}, obj_value={obj_value!r}"
        )

    rate = max(0.0, math.log(float(previous_best) / float(obj_value)))
    previous_ema = state.get("trajectory_rate_ema", None)
    if previous_ema is None:
        rate_ema = rate
    else:
        rate_ema = (
            _TRAJECTORY_EMA_BETA * rate
            + (1.0 - _TRAJECTORY_EMA_BETA) * float(previous_ema)
        )
    previous_ref = state.get("trajectory_rate_ref", None)
    rate_ref = max(
        rate_ema,
        0.0 if previous_ref is None else float(previous_ref),
    )
    rate_count = int(state.get("trajectory_rate_count", 0)) + 1
    ratio = (
        rate_ema / rate_ref
        if rate_ref > _TRAJECTORY_EPS
        else None
    )
    saturated = bool(
        rate_count >= _TRAJECTORY_MIN_RATES
        and ratio is not None
        and ratio < _TRAJECTORY_SATURATION_RATIO
    )
    return {
        "best_obj": float(obj_value),
        "rate": rate,
        "rate_ema": rate_ema,
        "rate_ref": rate_ref,
        "rate_count": rate_count,
        "ratio": ratio,
        "saturated": saturated,
    }


def _format_trajectory_csv_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12e}"
    return value


def _write_trajectory_ready_csv_row(project, row):
    log_path = getattr(project, "trajectory_ready_log_path", None)
    if not log_path:
        return
    fieldnames = [
        "call_idx",
        "design_folder",
        "obj_value",
        "best_obj",
        "rate",
        "rate_ema",
        "rate_ref",
        "rate_count",
        "saturation_ratio",
        "saturated",
        "enrichment_gain",
        "enrichment_useful",
        "min_margin",
        "unambiguous",
        "batch_changed",
        "compatible_count",
        "resolved_crossover",
        "frontier_ready",
        "triggered",
        "batch",
        "reason",
    ]
    write_header = not os.path.exists(log_path)
    try:
        with open(log_path, "a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    field: _format_trajectory_csv_value(row.get(field))
                    for field in fieldnames
                }
            )
    except Exception as exc:
        _write(project, f"TRAJECTORY_READY | CSV log write failed: {exc!r}")


def _pin_trajectory_ready_sample(project, state, batch_key):
    """Pin refinement to the exact accepted major iterate just scored."""

    trigger_dv_values = getattr(project, "last_obj_grad_x_full", None)
    if trigger_dv_values is None:
        trigger_dv_values = getattr(project, "last_obj_grad_x", None)
    state["trigger_dv_values"] = (
        None
        if trigger_dv_values is None
        else [float(value) for value in trigger_dv_values]
    )
    trigger_reduced = getattr(project, "last_obj_grad_x", None)
    state["trigger_reduced_dv_values"] = (
        None
        if trigger_reduced is None
        else [float(value) for value in trigger_reduced]
    )
    state["trigger_design_folder"] = getattr(
        project,
        "last_obj_grad_design_folder",
        None,
    )
    state["trajectory_trigger_batch"] = batch_key
    project.refinement_dv_values = state["trigger_dv_values"]
    project.refinement_triggered = True


def _check_trajectory_ready_trigger(project, obj_value, opts):
    """Refine when the objective, enrichment, and ranking trajectory agree.

    The three gates are evaluated only on accepted designs with an available
    objective adjoint:

      1. the EMA of the best-objective log rate has saturated relative to its
         best rate observed in the level;
      2. the proposed refined space captures enough additional signal energy;
      3. the ordered sequential frontier is either confirmed and unambiguous,
         or has just completed a monotone runner-up/winner crossover.

    When all gates agree, the current DSN and DV vector are pinned before
    raising RefinementTriggered.
    """

    del opts  # calibration is deliberately internal for the first validation
    _init_trigger_state(project)
    state = project.trigger_state
    state.setdefault("trajectory_best_obj", None)
    state.setdefault("trajectory_rate_ema", None)
    state.setdefault("trajectory_rate_ref", None)
    state.setdefault("trajectory_rate_count", 0)
    state.setdefault("trajectory_frontier_history", [])
    state.setdefault("trajectory_compatible_count", 0)
    state.setdefault("trajectory_call_idx", 0)
    state.setdefault("trajectory_trigger_batch", None)

    if obj_value is None:
        _write(project, "TRAJECTORY_READY | no objective recorded yet, skip")
        return
    try:
        obj_value = float(obj_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"TRAJECTORY_READY received invalid objective {obj_value!r}"
        ) from exc
    if not math.isfinite(obj_value):
        raise ValueError(
            f"TRAJECTORY_READY received non-finite objective {obj_value!r}"
        )

    best_obj = state.get("trajectory_best_obj", None)
    if best_obj is not None and obj_value >= float(best_obj):
        _write(
            project,
            "TRAJECTORY_READY | not a new adjoint best "
            f"({obj_value:.6e} >= {float(best_obj):.6e}), skip",
        )
        return

    scorer_fn = getattr(project, "trajectory_ready_scorer_fn", None)
    if scorer_fn is None:
        raise RuntimeError(
            "TRAJECTORY_READY is active but no post-adjoint scorer is attached"
        )

    # Scoring failures do not consume the objective best: a later accepted
    # design can retry against the last successfully sampled state.
    try:
        scoring_result = scorer_fn()
    except Exception as exc:
        _write(project, f"TRAJECTORY_READY | scorer raised {exc!r}, sample skipped")
        return

    selected = _trajectory_selected_candidates(scoring_result)
    if not selected:
        _write(project, "TRAJECTORY_READY | scorer returned empty batch, sample skipped")
        return
    batch_key = tuple(_trajectory_candidate_key(item) for item in selected)
    enrichment_gain = _trajectory_enrichment_gain(selected)
    frontier = _trajectory_frontier_snapshot(
        scoring_result,
        selected,
        batch_key,
    )
    saturation = _trajectory_saturation_preview(state, obj_value)

    history = state["trajectory_frontier_history"]
    batch_changed = bool(history and batch_key != history[-1]["batch"])
    if history and batch_key == history[-1]["batch"]:
        compatible_count = int(state["trajectory_compatible_count"]) + 1
    else:
        compatible_count = 1
    resolved_crossover = bool(
        batch_changed and _trajectory_resolved_crossover(history, frontier)
    )
    min_margin = frontier["min_margin"]
    unambiguous = bool(
        frontier["all_margins_available"]
        and min_margin is not None
        and min_margin >= _TRAJECTORY_FRONTIER_MARGIN
    )
    frontier_ready = bool(
        (
            compatible_count >= _TRAJECTORY_COMPATIBLE_SAMPLES
            and unambiguous
        )
        or resolved_crossover
    )
    enrichment_useful = bool(
        enrichment_gain >= _TRAJECTORY_ENRICHMENT_GAIN
    )
    triggered = bool(
        saturation["saturated"]
        and enrichment_useful
        and frontier_ready
    )

    reasons = []
    if not saturation["saturated"]:
        reasons.append("OBJECTIVE_NOT_SATURATED")
    if not enrichment_useful:
        reasons.append("ENRICHMENT_TOO_SMALL")
    if not frontier_ready:
        if not frontier["all_margins_available"]:
            reasons.append("FRONTIER_MARGIN_UNAVAILABLE")
        elif not unambiguous:
            reasons.append("FRONTIER_AMBIGUOUS")
        elif compatible_count < _TRAJECTORY_COMPATIBLE_SAMPLES:
            reasons.append("FRONTIER_NOT_CONFIRMED")
        else:
            reasons.append("FRONTIER_NOT_READY")
    reason = "READY" if triggered else ";".join(reasons)

    # Commit only after the full scorer contract and all metrics are valid.
    state["trajectory_best_obj"] = saturation["best_obj"]
    state["trajectory_rate_ema"] = saturation["rate_ema"]
    state["trajectory_rate_ref"] = saturation["rate_ref"]
    state["trajectory_rate_count"] = saturation["rate_count"]
    state["trajectory_compatible_count"] = compatible_count
    state["trajectory_call_idx"] += 1
    state["trajectory_ready"] = triggered
    history.append(frontier)

    row = {
        "call_idx": state["trajectory_call_idx"],
        "design_folder": getattr(
            project,
            "last_obj_grad_design_folder",
            "",
        ),
        "obj_value": obj_value,
        "best_obj": saturation["best_obj"],
        "rate": saturation["rate"],
        "rate_ema": saturation["rate_ema"],
        "rate_ref": saturation["rate_ref"],
        "rate_count": saturation["rate_count"],
        "saturation_ratio": saturation["ratio"],
        "saturated": int(saturation["saturated"]),
        "enrichment_gain": enrichment_gain,
        "enrichment_useful": int(enrichment_useful),
        "min_margin": min_margin,
        "unambiguous": int(unambiguous),
        "batch_changed": int(batch_changed),
        "compatible_count": compatible_count,
        "resolved_crossover": int(resolved_crossover),
        "frontier_ready": int(frontier_ready),
        "triggered": int(triggered),
        "batch": str(list(batch_key)),
        "reason": reason,
    }
    _write_trajectory_ready_csv_row(project, row)

    ratio_text = (
        "NA"
        if saturation["ratio"] is None
        else f"{saturation['ratio']:.6e}"
    )
    margin_text = "NA" if min_margin is None else f"{min_margin:.6e}"
    _write(
        project,
        "TRAJECTORY_READY | "
        f"best={obj_value:.6e} sat={int(saturation['saturated'])} "
        f"ratio={ratio_text} enrich={enrichment_gain:.6e} "
        f"margin={margin_text} compatible={compatible_count} "
        f"crossover={int(resolved_crossover)} "
        f"frontier={int(frontier_ready)} trigger={int(triggered)} "
        f"batch={batch_key}",
    )

    if triggered:
        _pin_trajectory_ready_sample(project, state, batch_key)
        _write(
            project,
            "TRAJECTORY_READY | objective saturated, enrichment useful, "
            "frontier ready -> STOP",
        )
        raise RefinementTriggered()


def record_gradient_and_check(project, obj_value, opts=None):
    """Hook called after each adjoint (gradient) evaluation.

    Adaptive-ranking triggers use this hook so their decision belongs to the
    exact accepted design whose adjoint was just computed.  Scalar-only
    triggers are checked post-objective via record_objective_and_check.
    """
    opts = getattr(project, "trigger_opts", None) if opts is None else opts
    if not opts:
        return
    trigger = str(opts.get("trigger", "")).upper()
    if trigger == "BATCH_STABILITY":
        _check_batch_stability_trigger(project, obj_value, opts)
    elif trigger == "TRAJECTORY_READY":
        _check_trajectory_ready_trigger(project, obj_value, opts)


def _is_final_level(current_level, current_ndv=None, final_ndv=None, nlevels=None):
    if final_ndv is not None and current_ndv is not None:
        return int(current_ndv) >= int(final_ndv)
    if final_ndv is None and nlevels is not None:
        return int(current_level) >= int(nlevels) - 1
    return False


def build_online_trigger_opts(
    trigger,
    current_level=0,
    current_ndv=None,
    final_ndv=None,
    nlevels=None,
    window=1,
    tolerance=0.2,
    filter_tolerance=0.02,
    warmup=0,
    eps=1.0e-300,
    patience=1,
    stagnation_tolerance=1.0e-3,
    stagnation_band=0.02,
    stagnation_window=3,
    dwell=4,
    min_ref=4,
    n_max=40,
    rate_floor=1.0e-4,
    **_unused,
):
    if _is_final_level(
        current_level,
        current_ndv=current_ndv,
        final_ndv=final_ndv,
        nlevels=nlevels,
    ):
        return None

    trigger = str(trigger or "MAX_ITER").upper()
    if trigger not in ALLOWED_TRIGGERS:
        raise ValueError(
            f"Invalid progressive trigger {trigger!r}; allowed values are {ALLOWED_TRIGGERS}"
        )
    if trigger == "MAX_ITER":
        return None

    warmup_iter = int(warmup)

    if trigger in ("SLOPE_EFFICIENCY_TRIGGER", "SLOPE_EFFICIENCY_FILTERED"):
        return {
            "trigger": trigger,
            "window": int(window),
            "tol": float(tolerance),
            "filter_tol": float(filter_tolerance),
            "warmup_iter": warmup_iter,
        }

    if trigger == "SLOPE_EFFICIENCY_BEST_LOG":
        return {
            "trigger": "SLOPE_EFFICIENCY_BEST_LOG",
            "window": int(window),
            "tol": float(tolerance),
            "warmup_iter": warmup_iter,
            "eps": float(eps),
            "patience": int(patience),
        }

    if trigger == "STAGNATION_TRIGGER":
        return {
            "trigger": trigger,
            "stag_tol": float(stagnation_tolerance),
            "stag_band": float(stagnation_band),
            "stag_window": int(stagnation_window),
            "warmup_iter": warmup_iter,
        }

    if trigger == "ECONOMIC_TRIGGER":
        return {
            "trigger": trigger,
            "window": int(window),
            "tol": float(tolerance),
            "dwell": int(dwell),
            "min_ref": int(min_ref),
            "patience": int(patience),
            "n_max": int(n_max),
            "rate_floor": float(rate_floor),
            "warmup_iter": warmup_iter,
            "eps": float(eps),
        }

    if trigger == "BATCH_STABILITY":
        # k is an internal constant (validated on the dense replay); no user knob
        return {
            "trigger": trigger,
            "k": 3,
        }

    if trigger == "TRAJECTORY_READY":
        return {
            "trigger": trigger,
        }

    return None
