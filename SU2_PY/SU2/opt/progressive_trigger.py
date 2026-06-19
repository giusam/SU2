#!/usr/bin/env python

"""Shared online progressive-refinement trigger engine."""

import math
import sys


class RefinementTriggered(Exception):
    pass


ALLOWED_TRIGGERS = (
    "MAX_ITER",
    "SLOPE_EFFICIENCY_TRIGGER",
    "SLOPE_EFFICIENCY_FILTERED",
    "SLOPE_EFFICIENCY_BEST_LOG",
    "STAGNATION_TRIGGER",
)


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


def record_objective_and_check(project, obj_value, opts=None):
    if not hasattr(project, "trigger_history") or project.trigger_history is None:
        project.trigger_history = []
    project.trigger_history.append(obj_value)
    check_project_trigger(project, obj_value, opts=opts)


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

    return None
