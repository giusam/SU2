"""Knot-depth and batch-penalty helpers for adaptive B-splines."""

import math



from .errors import (
    BSplineAdaptiveError,
    _as_bool,
    _as_float,
)
from .knot_space import knot_insertion_spans

ALLOWED_KNOT_BATCH_PENALTY_MODES = ("NONE", "STREUBER_DEPTH", "POWER")

ALLOWED_KNOT_DEPTH_PENALTY_MODES = ALLOWED_KNOT_BATCH_PENALTY_MODES

KNOT_DEPTH_PENALTY_MODE_CHOICES = (
    *ALLOWED_KNOT_DEPTH_PENALTY_MODES,
    "STREUBER",
)

def _span_key(left, right, digits=14):
    return (
        f"{round(float(left), int(digits)):.{int(digits)}f}|"
        f"{round(float(right), int(digits)):.{int(digits)}f}"
    )

def _as_knot_depth(value, name):
    try:
        depth = int(value)
        value_float = float(value)
    except Exception:
        raise BSplineAdaptiveError(f"{name} must be an integer >= 1")
    if not math.isfinite(value_float) or value_float != float(depth) or depth < 1:
        raise BSplineAdaptiveError(f"{name} must be an integer >= 1")
    return depth

def _knot_initial_span_depth(settings):
    depth = _as_knot_depth(
        settings.get("knot_initial_span_depth", 1),
        "BSPLINE_KNOT_INITIAL_SPAN_DEPTH",
    )
    if depth < 1:
        raise BSplineAdaptiveError("BSPLINE_KNOT_INITIAL_SPAN_DEPTH must be >= 1")
    return depth

def initialize_knot_span_depths(knot_vector, min_width=1.0e-8, initial_depth=1):
    initial_depth = _as_knot_depth(initial_depth, "initial knot span depth")
    return {
        _span_key(left, right): initial_depth
        for left, right, _inserted in knot_insertion_spans(
            knot_vector,
            min_width=min_width,
        )
    }

def _validated_knot_span_depths(raw_depths):
    if not isinstance(raw_depths, dict):
        raise BSplineAdaptiveError("knot_span_depths must be a JSON object")
    span_depths = {}
    for key, value in raw_depths.items():
        depth = _as_knot_depth(value, f"knot_span_depths[{key!r}]")
        span_depths[str(key)] = depth
    return span_depths

def _normalize_knot_depth_penalty_mode(value):
    mode = str(value).strip().upper()
    if mode == "STREUBER":
        mode = "STREUBER_DEPTH"
    if mode not in ALLOWED_KNOT_DEPTH_PENALTY_MODES:
        raise BSplineAdaptiveError(
            f"unsupported knot depth penalty mode {mode!r}; "
            f"allowed values are {ALLOWED_KNOT_DEPTH_PENALTY_MODES}"
        )
    return mode

def _knot_depth_penalty_mode(settings):
    mode = _normalize_knot_depth_penalty_mode(
        settings.get(
            "knot_depth_penalty_mode",
            settings.get("knot_batch_penalty_mode", "NONE"),
        )
    )
    enabled = _as_bool(
        settings.get(
            "knot_depth_penalty",
            settings.get("knot_batch_diversity", mode != "NONE"),
        ),
        default=False,
    )
    if enabled and mode == "NONE":
        raise BSplineAdaptiveError(
            "BSPLINE_KNOT_DEPTH_PENALTY=YES requires "
            "BSPLINE_KNOT_DEPTH_PENALTY_MODE=STREUBER_DEPTH or POWER"
        )
    return mode if enabled else "NONE"

def _knot_depth_power_gamma(settings):
    gamma = _as_float(
        settings.get(
            "knot_depth_power_gamma",
            settings.get("knot_batch_power_gamma", 0.25),
        ),
        "BSPLINE_KNOT_DEPTH_POWER_GAMMA",
    )
    if not (0.0 < gamma <= 1.0):
        raise BSplineAdaptiveError(
            "BSPLINE_KNOT_DEPTH_POWER_GAMMA must satisfy 0 < gamma <= 1"
        )
    return gamma

def get_or_initialize_knot_span_depths(spec, knot_vector, settings):
    settings = dict(settings or {})
    min_width = settings.get("knot_min_span_width", 1.0e-8)
    current_keys = [
        _span_key(left, right)
        for left, right, _inserted in knot_insertion_spans(
            knot_vector,
            min_width=min_width,
        )
    ]
    if spec.get("knot_span_depths") is None:
        return initialize_knot_span_depths(
            knot_vector,
            min_width=min_width,
            initial_depth=_knot_initial_span_depth(settings),
        )

    span_depths = _validated_knot_span_depths(spec["knot_span_depths"])
    missing = [key for key in current_keys if key not in span_depths]
    if missing:
        if _knot_depth_penalty_mode(settings) != "NONE":
            raise BSplineAdaptiveError(
                "knot_span_depths is missing current spans required by the active "
                f"knot depth penalty: {missing[:5]}"
            )
        initial_depth = _knot_initial_span_depth(settings)
        for key in missing:
            span_depths[key] = initial_depth
    return {
        key: span_depths[key]
        for key in current_keys
    }

def _span_contains(parent_left, parent_right, child_left, child_right, tol=1.0e-12):
    return (
        float(child_left) >= float(parent_left) - float(tol)
        and float(child_right) <= float(parent_right) + float(tol)
    )

def _knot_batch_depth(row, selected_insertions, tol=1.0e-12):
    child_left = float(row["span_left"])
    child_right = float(row["span_right"])
    depth = 1
    for selected in selected_insertions:
        if _span_contains(
            selected["span_left"],
            selected["span_right"],
            child_left,
            child_right,
            tol=tol,
        ):
            depth += 1
    return depth

def _streuber_depth_penalty(depth):
    value = 1.0 - 0.5 * (
        math.tanh(3.0) + math.tanh((float(depth) - 1.0) - 3.0)
    )
    return min(1.0, max(0.0, float(value)))

def _knot_batch_penalty(depth, settings):
    mode = _knot_depth_penalty_mode(settings)
    if mode == "NONE":
        return 1.0
    if mode == "STREUBER_DEPTH":
        return _streuber_depth_penalty(depth)
    if mode == "POWER":
        gamma = _knot_depth_power_gamma(settings)
        return float(gamma) ** (int(depth) - 1)
    raise BSplineAdaptiveError(
        f"unsupported knot depth penalty mode {mode!r}; "
        f"allowed values are {ALLOWED_KNOT_DEPTH_PENALTY_MODES}"
    )

def apply_knot_depth_penalty(step_rows, span_depths, settings):
    mode = _knot_depth_penalty_mode(settings)

    for row in step_rows:
        key = _span_key(row["span_left"], row["span_right"])
        if key not in span_depths:
            raise BSplineAdaptiveError(
                "knot_span_depths is missing current span "
                f"[{float(row['span_left']):.14g}, {float(row['span_right']):.14g}]"
            )
        parent_depth = int(span_depths[key])
        child_depth = parent_depth + 1
        score_unpenalized = float(row.get("score_raw", row.get("score", 0.0)))
        penalty = 1.0 if mode == "NONE" else _knot_batch_penalty(child_depth, settings)
        score_effective = score_unpenalized * penalty
        row["span_key"] = key
        row["parent_depth"] = int(parent_depth)
        row["child_depth"] = int(child_depth)
        row["depth_penalty"] = float(penalty)
        row["depth_penalty_mode"] = mode
        row["score_effective"] = float(score_effective)
        row["selection_score"] = float(score_effective)
        row["batch_depth"] = int(child_depth)
        row["batch_penalty"] = float(penalty)
        row["batch_penalty_mode"] = mode

    if mode != "NONE":
        step_rows.sort(
            key=lambda row: (
                -float(row["selection_score"]),
                float(row["span_left"]),
                float(row["span_right"]),
            )
        )
        for rank, row in enumerate(step_rows, start=1):
            row["rank"] = rank
    return step_rows

def _knot_intra_batch_penalty(depth, settings, has_selected_insertions=False):
    mode = _knot_depth_penalty_mode(settings)
    depth = int(depth)
    child_depth = int(depth) + 1

    if mode == "NONE":
        return 1.0

    # In the legacy intra-batch wrapper, a fresh span before any batch
    # insertion starts from parent_depth=1 and child_depth=2, matching
    # the persistent depth-penalty convention. Once the batch already
    # contains selected insertions, however, a span on a fresh branch
    # must remain unpenalized so it can compete with nested descendants.
    if bool(has_selected_insertions) and depth <= 1:
        return 1.0

    if mode == "STREUBER_DEPTH":
        return _streuber_depth_penalty(child_depth)

    if mode == "POWER":
        gamma = _knot_depth_power_gamma(settings)
        return float(gamma) ** (child_depth - 1)

    raise BSplineAdaptiveError(
        f"unsupported knot depth penalty mode {mode!r}; "
        f"allowed values are {ALLOWED_KNOT_DEPTH_PENALTY_MODES}"
    )

def apply_knot_batch_penalty(step_rows, selected_insertions, settings):
    mode = _knot_depth_penalty_mode(settings)
    has_selected_insertions = bool(selected_insertions)

    for row in step_rows:
        depth = _knot_batch_depth(row, selected_insertions)
        score_unpenalized = float(row.get("score_raw", row.get("score", 0.0)))
        penalty = _knot_intra_batch_penalty(
            depth,
            settings,
            has_selected_insertions=has_selected_insertions,
        )
        score_effective = score_unpenalized * penalty

        row["span_key"] = row.get(
            "span_key",
            _span_key(row["span_left"], row["span_right"]),
        )
        row["parent_depth"] = int(depth)
        row["child_depth"] = int(depth) + 1
        row["depth_penalty"] = float(penalty)
        row["depth_penalty_mode"] = mode
        row["batch_depth"] = int(depth)
        row["batch_penalty"] = float(penalty)
        row["batch_penalty_mode"] = mode
        row["score_effective"] = float(score_effective)
        row["selection_score"] = float(score_effective)

    if mode != "NONE":
        step_rows.sort(
            key=lambda row: (
                -float(row["selection_score"]),
                float(row["span_left"]),
                float(row["span_right"]),
            )
        )
        for rank, row in enumerate(step_rows, start=1):
            row["rank"] = rank

    return step_rows
