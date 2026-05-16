#!/usr/bin/env python

import math

from SU2.opt.hh_spring import (
    apply_hh_spring_after_selection,
    spring_redistribute_centers,
)
from SU2.opt.progressive_hh_projection import _compute_dot_candidate_scores


class HHLevel:
    def __init__(
        self,
        level_id,
        upper,
        lower,
        workdir,
        config_filename,
        project_filename,
        mesh_source=None,
        selection_metadata=None,
        post_opt_spring_pending=False,
        spring_reallocated=False,
    ):
        self.level_id = level_id
        self.upper = list(upper)
        self.lower = list(lower)
        self.workdir = workdir
        self.config_filename = config_filename
        self.project_filename = project_filename
        self.mesh_source = mesh_source
        self.selection_metadata = selection_metadata
        self.post_opt_spring_pending = post_opt_spring_pending
        self.spring_reallocated = spring_reallocated

    @property
    def ndv(self):
        return len(self.upper) + len(self.lower)


def _as_bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().upper() in ("YES", "TRUE", "1", "ON")


def _parse_none_or_int(value):
    if value is None:
        return None

    raw = str(value).strip()
    if not raw or raw.upper() == "NONE":
        return None

    return int(raw)


def _as_optional_int(value):
    return _parse_none_or_int(value)


def _count_initial_points(value):
    if value is None:
        return None

    raw = str(value).strip()
    if not raw:
        return None

    raw = raw.strip("()[]")
    if not raw:
        return 0

    return len([x for x in raw.split(",") if x.strip()])


def get_progressive_hh_options(config):
    scale = 1.0
    if "DEFINITION_DV" in config:
        def_dv = config["DEFINITION_DV"]
        if "SCALE" in def_dv and def_dv["SCALE"]:
            try:
                scale = float(def_dv["SCALE"][0])
            except Exception:
                pass

    surface_mode = str(config.get("PROGRESSIVE_HH_SURFACE", "BOTH")).upper()
    trigger = str(config.get("PROGRESSIVE_HH_TRIGGER", "MAX_ITER")).upper()
    n0 = int(config.get("PROGRESSIVE_HH_N0", 3))
    nfinal = _as_optional_int(config.get("PROGRESSIVE_HH_NFINAL", None))
    nadd_mode = str(config.get("PROGRESSIVE_HH_NADD_MODE", "GROWTH_RATIO")).upper()
    fixed_nadd = int(config.get("PROGRESSIVE_HH_FIXED_NADD", 1))
    batch_size_max = int(config.get("PROGRESSIVE_HH_BATCH_SIZE_MAX", 1))
    batch_score_rel_tol = float(
        config.get("PROGRESSIVE_HH_BATCH_SCORE_REL_TOL", 0.85)
    )
    batch_min_separation = float(
        config.get("PROGRESSIVE_HH_BATCH_MIN_SEPARATION", 0.04)
    )
    batch_max_per_side = _parse_none_or_int(
        config.get("PROGRESSIVE_HH_BATCH_MAX_PER_SIDE", None)
    )
    candidate_samples = int(config.get("PROGRESSIVE_HH_CANDIDATE_SAMPLES", 1))
    spring_timing = str(
        config.get("PROGRESSIVE_HH_SPRING_TIMING", "POST_OPT")
    ).upper()
    spring_score_mode = str(
        config.get("PROGRESSIVE_HH_SPRING_SCORE_MODE", "COEFFICIENT")
    ).upper()

    allowed_triggers = (
        "MAX_ITER",
        "SLOPE_EFFICIENCY_TRIGGER",
        "SLOPE_EFFICIENCY_FILTERED",
        "SLOPE_EFFICIENCY_BEST_LOG",
        "STAGNATION_TRIGGER",
    )
    if trigger not in allowed_triggers:
        raise ValueError(
            "Invalid PROGRESSIVE_HH_TRIGGER "
            f"{trigger!r}; allowed values are {allowed_triggers}"
        )

    allowed_nadd_modes = ("GROWTH_RATIO", "FIXED", "SCORE_BATCH")
    if nadd_mode not in allowed_nadd_modes:
        raise ValueError(
            "Invalid PROGRESSIVE_HH_NADD_MODE "
            f"{nadd_mode!r}; allowed values are {allowed_nadd_modes}"
        )

    allowed_spring_timings = ("POST_OPT", "PRE_REFINE")
    if spring_timing not in allowed_spring_timings:
        raise ValueError(
            "Invalid PROGRESSIVE_HH_SPRING_TIMING "
            f"{spring_timing!r}; allowed values are {allowed_spring_timings}"
        )

    allowed_spring_score_modes = ("COEFFICIENT", "INDICATOR")
    if spring_score_mode not in allowed_spring_score_modes:
        raise ValueError(
            "Invalid PROGRESSIVE_HH_SPRING_SCORE_MODE "
            f"{spring_score_mode!r}; allowed values are {allowed_spring_score_modes}"
        )

    if fixed_nadd < 1:
        raise ValueError("PROGRESSIVE_HH_FIXED_NADD must be >= 1")
    if batch_size_max < 1:
        raise ValueError("PROGRESSIVE_HH_BATCH_SIZE_MAX must be >= 1")
    if not (0.0 <= batch_score_rel_tol <= 1.0):
        raise ValueError("PROGRESSIVE_HH_BATCH_SCORE_REL_TOL must be in [0, 1]")
    if batch_min_separation < 0.0:
        raise ValueError("PROGRESSIVE_HH_BATCH_MIN_SEPARATION must be >= 0")
    if batch_max_per_side is not None and batch_max_per_side < 1:
        raise ValueError("PROGRESSIVE_HH_BATCH_MAX_PER_SIDE must be NONE or >= 1")
    if candidate_samples < 1:
        raise ValueError("PROGRESSIVE_HH_CANDIDATE_SAMPLES must be >= 1")

    initial_ndv = 0
    if surface_mode in ("UPPER", "BOTH"):
        upper_count = _count_initial_points(
            config.get("PROGRESSIVE_HH_INITIAL_UPPER", None)
        )
        initial_ndv += n0 if upper_count is None else upper_count
    if surface_mode in ("LOWER", "BOTH"):
        lower_count = _count_initial_points(
            config.get("PROGRESSIVE_HH_INITIAL_LOWER", None)
        )
        initial_ndv += n0 if lower_count is None else lower_count

    if nfinal is not None:
        if nfinal < 1:
            raise ValueError("PROGRESSIVE_HH_NFINAL must be NONE or >= 1")
        if nfinal < initial_ndv:
            raise ValueError(
                "PROGRESSIVE_HH_NFINAL must be >= the initial HH NDV "
                f"({initial_ndv})"
            )

    return {
        "enabled": _as_bool(config.get("PROGRESSIVE_HH", "NO")),
        "nlevels": int(config.get("PROGRESSIVE_HH_NLEVELS", 1)),
        "n0": n0,
        "nfinal": nfinal,
        "surface_mode": surface_mode,
        "trigger": trigger,
        "window": int(config.get("PROGRESSIVE_HH_WINDOW", 1)),
        "tol": float(config.get("PROGRESSIVE_HH_TOL", 0.2)),
        "slope_filter_tol": float(config.get("PROGRESSIVE_HH_SLOPE_FILTER_TOL", 0.02)),
        "trigger_eps": float(config.get("PROGRESSIVE_HH_TRIGGER_EPS", 1.0e-300)),
        "slope_patience": int(config.get("PROGRESSIVE_HH_SLOPE_PATIENCE", 1)),
        "stag_tol": float(config.get("PROGRESSIVE_HH_STAG_TOL", 1.0e-3)),
        "stag_band": float(config.get("PROGRESSIVE_HH_STAG_BAND", 0.02)),
        "stag_window": int(config.get("PROGRESSIVE_HH_STAG_WINDOW", 3)),
        "warmup_iter": int(config.get("PROGRESSIVE_HH_WARMUP_ITER", 0)),
        "max_iter_per_level": int(
            config.get("PROGRESSIVE_HH_MAX_ITER_PER_LEVEL", config.OPT_ITERATIONS)
        ),
        "refinement": str(config.get("PROGRESSIVE_HH_REFINEMENT", "UNIFORM")).upper(),
        "growth_ratio": float(config.get("PROGRESSIVE_HH_GROWTH_RATIO", 2.0)),
        "nadd_mode": nadd_mode,
        "fixed_nadd": fixed_nadd,
        "batch_size_max": batch_size_max,
        "batch_score_rel_tol": batch_score_rel_tol,
        "batch_min_separation": batch_min_separation,
        "batch_max_per_side": batch_max_per_side,
        "candidate_samples": candidate_samples,
        "adaptive_indicator": str(
            config.get("PROGRESSIVE_HH_ADAPTIVE_INDICATOR", "ABS_GRAD")
        ).upper(),
        "marker": str(config.get("DV_MARKER", "Airfoil")),
        "scale": scale,
        "spring_enabled": _as_bool(config.get("PROGRESSIVE_HH_SPRING", "NO")),
        "spring_A": float(config.get("PROGRESSIVE_HH_SPRING_A", 20.0)),
        "spring_timing": spring_timing,
        "spring_score_mode": spring_score_mode,
    }


def initial_centers(n0):
    if n0 <= 0:
        return []
    return [(i + 1) / float(n0 + 1) for i in range(n0)]


def refine_uniform(centers):
    if not centers:
        return []

    centers = sorted(centers)
    extended = [0.0] + centers + [1.0]

    new_points = []
    for i in range(len(extended) - 1):
        xm = 0.5 * (extended[i] + extended[i + 1])
        if 0.0 < xm < 1.0:
            new_points.append(xm)

    return sorted(set(centers + new_points))


def _compute_adaptive_nadd(current_ndv, ncandidates, growth_ratio):
    if ncandidates <= 0:
        return 0

    growth_ratio = float(growth_ratio)
    if growth_ratio <= 1.0:
        target_ndv = current_ndv + 1
    else:
        target_ndv = int(math.ceil(growth_ratio * current_ndv))

    nadd = max(1, target_ndv - current_ndv)
    nadd = min(nadd, ncandidates)
    return nadd


def _select_top_candidates(candidates, nadd):
    if nadd <= 0 or not candidates:
        return []

    ranked = sorted(candidates, key=lambda c: (-c["indicator"], c["x"]))
    return ranked[:nadd]


def select_adaptive_candidate_batch(
    candidates,
    max_batch_size,
    score_rel_tol,
    min_separation,
    max_per_side=None,
    n_remaining=None,
):
    if not candidates:
        return []

    max_batch_size = max(1, int(max_batch_size))
    score_rel_tol = float(score_rel_tol)
    min_separation = float(min_separation)

    if max_per_side is not None:
        max_per_side = int(max_per_side)

    if n_remaining is not None:
        n_remaining = int(n_remaining)
        if n_remaining <= 0:
            return []
        max_batch_size = min(max_batch_size, n_remaining)

    ranked = sorted(
        candidates,
        key=lambda c: (-float(c["indicator"]), str(c["side"]), float(c["x"])),
    )

    best_indicator = float(ranked[0]["indicator"])
    threshold = score_rel_tol * best_indicator
    selected = []
    per_side = {}

    for c in ranked:
        if len(selected) >= max_batch_size:
            break

        indicator = float(c["indicator"])
        side = str(c["side"])
        x = float(c["x"])

        if selected and indicator < threshold:
            continue

        if max_per_side is not None and per_side.get(side, 0) >= max_per_side:
            continue

        too_close = False
        for s in selected:
            if str(s["side"]) == side and abs(float(s["x"]) - x) < min_separation:
                too_close = True
                break
        if too_close:
            continue

        selected.append(c)
        per_side[side] = per_side.get(side, 0) + 1

    return selected


def select_candidates_by_nadd_mode(candidates, current_ndv, opts):
    if not candidates:
        return []

    nfinal = opts.get("nfinal", None)
    n_remaining = None
    if nfinal is not None:
        n_remaining = int(nfinal) - int(current_ndv)
        if n_remaining <= 0:
            return []

    mode = str(opts.get("nadd_mode", "GROWTH_RATIO")).upper()

    if mode == "GROWTH_RATIO":
        nadd = _compute_adaptive_nadd(
            current_ndv,
            len(candidates),
            opts.get("growth_ratio", 2.0),
        )
        if n_remaining is not None:
            nadd = min(nadd, n_remaining)
        return _select_top_candidates(candidates, nadd)

    if mode == "FIXED":
        nadd = int(opts.get("fixed_nadd", 1))
        if n_remaining is not None:
            nadd = min(nadd, n_remaining)
        return _select_top_candidates(candidates, nadd)

    if mode == "SCORE_BATCH":
        return select_adaptive_candidate_batch(
            candidates,
            opts.get("batch_size_max", 1),
            opts.get("batch_score_rel_tol", 0.85),
            opts.get("batch_min_separation", 0.04),
            max_per_side=opts.get("batch_max_per_side", None),
            n_remaining=n_remaining,
        )

    raise ValueError(f"Unknown progressive HH candidate addition mode: {mode}")


def refine_adaptive(prev_level, result, opts):
    current_ndv = prev_level.ndv
    opts["_last_selection_metadata"] = None

    try:
        scoring = _compute_dot_candidate_scores(prev_level, opts)
        candidates = scoring.get("candidates", [])
        active_upper_scores = scoring.get("active_upper_scores", [])
        active_lower_scores = scoring.get("active_lower_scores", [])
    except Exception as err:
        print(
            "[PROGRESSIVE_HH] WARNING: ADAPTIVE refine failed -> fallback to UNIFORM | "
            f"{err}"
        )
        return refine_uniform(prev_level.upper), refine_uniform(prev_level.lower)

    if not candidates:
        print("[PROGRESSIVE_HH] ADAPTIVE refine | no candidates -> fallback to UNIFORM")
        return refine_uniform(prev_level.upper), refine_uniform(prev_level.lower)

    chosen = select_candidates_by_nadd_mode(candidates, current_ndv, opts)
    nadd = len(chosen)

    if not chosen:
        nfinal = opts.get("nfinal", None)
        if nfinal is not None and int(nfinal) - current_ndv <= 0:
            print(
                "[PROGRESSIVE_HH] ADAPTIVE refine | "
                f"NFINAL reached (ndv={current_ndv}, nfinal={nfinal}) -> no refinement"
            )
        else:
            print("[PROGRESSIVE_HH] ADAPTIVE refine | no candidates selected")
        return sorted(prev_level.upper), sorted(prev_level.lower)

    spring_debug = None

    spring_enabled = bool(opts.get("spring_enabled", False))
    spring_timing = str(opts.get("spring_timing", "POST_OPT")).upper()
    spring_score_mode = str(opts.get("spring_score_mode", "COEFFICIENT")).upper()

    if spring_enabled and spring_timing == "PRE_REFINE":
        if spring_score_mode != "INDICATOR":
            print(
                "[PROGRESSIVE_HH][SPRING] PRE_REFINE requested without "
                "indicator score mode -> using candidate indicators"
            )
        old_upper = sorted(prev_level.upper)
        old_lower = sorted(prev_level.lower)

        new_upper, new_lower = apply_hh_spring_after_selection(
            prev_level,
            chosen,
            active_upper_scores,
            active_lower_scores,
            opts,
        )

        def _spacing_stats(xs):
            xs = sorted(xs)
            if len(xs) < 2:
                return None

            dx = []
            for i in range(len(xs) - 1):
                dx.append(xs[i + 1] - xs[i])

            return {
                "min": min(dx),
                "max": max(dx),
                "ratio": max(dx) / min(dx),
            }

        spring_debug = {
            "old_upper": old_upper,
            "old_lower": old_lower,
            "selected_upper": [c["x"] for c in chosen if c["side"] == "UPPER"],
            "selected_lower": [c["x"] for c in chosen if c["side"] == "LOWER"],
            "new_upper": new_upper,
            "new_lower": new_lower,
            "upper_spacing": _spacing_stats(new_upper),
            "lower_spacing": _spacing_stats(new_lower),
        }
    else:
        new_upper = sorted(prev_level.upper)
        new_lower = sorted(prev_level.lower)

        for c in chosen:
            if c["side"] == "UPPER":
                new_upper.append(c["x"])
            else:
                new_lower.append(c["x"])

        new_upper = sorted(set(new_upper))
        new_lower = sorted(set(new_lower))

    ndv_after = len(set(new_upper)) + len(set(new_lower))
    best_indicator = max(float(c["indicator"]) for c in chosen)
    ratios = [
        float(c["indicator"]) / best_indicator if best_indicator != 0.0 else 0.0
        for c in chosen
    ]
    centers = [float(c["x"]) for c in chosen]
    indicators = [float(c["indicator"]) for c in chosen]

    print(
        "[PROGRESSIVE_HH] ADAPTIVE batch summary | "
        f"mode={opts.get('nadd_mode', 'GROWTH_RATIO')} "
        f"ndv_before={current_ndv} "
        f"ndv_after={ndv_after} "
        f"n_added={nadd} "
        f"centers={[round(x, 6) for x in centers]} "
        f"indicators={[float(f'{i:.6e}') for i in indicators]} "
        f"ratios={[float(f'{r:.6e}') for r in ratios]}"
    )

    print(
        f"[PROGRESSIVE_HH] ADAPTIVE refine | add={nadd} "
        f"spring={spring_enabled} spring_timing={spring_timing}"
    )

    for c in chosen:
        print(
            "[PROGRESSIVE_HH] ADAPTIVE selected | "
            f"side={c['side']} x={c['x']:.6f} I={c['indicator']:.6e}"
        )

    if spring_debug is not None:
        if spring_debug["upper_spacing"] is not None:
            print(
                "[PROGRESSIVE_HH][SPRING] UPPER spacing | "
                f"min={spring_debug['upper_spacing']['min']:.6f} "
                f"max={spring_debug['upper_spacing']['max']:.6f} "
                f"ratio={spring_debug['upper_spacing']['ratio']:.2f}"
            )

        if spring_debug["lower_spacing"] is not None:
            print(
                "[PROGRESSIVE_HH][SPRING] LOWER spacing | "
                f"min={spring_debug['lower_spacing']['min']:.6f} "
                f"max={spring_debug['lower_spacing']['max']:.6f} "
                f"ratio={spring_debug['lower_spacing']['ratio']:.2f}"
            )

        print("[PROGRESSIVE_HH][SPRING] Upper before:", spring_debug["old_upper"])
        print("[PROGRESSIVE_HH][SPRING] Upper selected:", spring_debug["selected_upper"])
        print("[PROGRESSIVE_HH][SPRING] Upper after :", spring_debug["new_upper"])

        print("[PROGRESSIVE_HH][SPRING] Lower before:", spring_debug["old_lower"])
        print("[PROGRESSIVE_HH][SPRING] Lower selected:", spring_debug["selected_lower"])
        print("[PROGRESSIVE_HH][SPRING] Lower after :", spring_debug["new_lower"])

    opts["_last_selection_metadata"] = {
        "level_id": prev_level.level_id,
        "ndv_before": current_ndv,
        "ndv_after": ndv_after,
        "n_added": nadd,
        "nadd_mode": opts.get("nadd_mode", "GROWTH_RATIO"),
        "trigger_mode": opts.get("trigger", "MAX_ITER"),
        "refinement": opts.get("refinement", "UNIFORM"),
        "spring_enabled": spring_enabled,
        "spring_timing": spring_timing,
        "spring_score_mode": spring_score_mode,
        "post_opt_spring_pending": spring_enabled and spring_timing == "POST_OPT",
        "upper_before": sorted(prev_level.upper),
        "lower_before": sorted(prev_level.lower),
        "upper_after": sorted(set(new_upper)),
        "lower_after": sorted(set(new_lower)),
        "selected": [
            {
                "side": c["side"],
                "x": float(c["x"]),
                "indicator": float(c["indicator"]),
                "indicator_ratio_to_best": ratios[i],
                "interval_id": c.get("interval_id"),
                "interval_left": c.get("interval_left"),
                "interval_right": c.get("interval_right"),
                "sample_index": c.get("sample_index"),
                "sample_fraction": c.get("sample_fraction"),
            }
            for i, c in enumerate(chosen)
        ],
    }

    return sorted(set(new_upper)), sorted(set(new_lower))


def apply_post_opt_coefficient_spring(level, result, opts):
    dv_values = result.get("dv_values", None)
    if dv_values is None:
        print(
            "[PROGRESSIVE_HH][SPRING] WARNING: missing optimized DV values; "
            "post-opt spring skipped"
        )
        return None

    expected_ndv = level.ndv
    if len(dv_values) != expected_ndv:
        print(
            "[PROGRESSIVE_HH][SPRING] WARNING: optimized DV size mismatch; "
            f"got {len(dv_values)}, expected {expected_ndv}; post-opt spring skipped"
        )
        return None

    try:
        coeff_abs = [abs(float(v)) for v in dv_values]
    except Exception as err:
        print(
            "[PROGRESSIVE_HH][SPRING] WARNING: invalid optimized DV values; "
            f"{err}; post-opt spring skipped"
        )
        return None

    n_upper = len(level.upper)
    n_lower = len(level.lower)
    upper_scores = coeff_abs[:n_upper]
    lower_scores = coeff_abs[n_upper : n_upper + n_lower]

    A = float(opts.get("spring_A", 20.0))

    try:
        new_upper = spring_redistribute_centers(level.upper, upper_scores, A=A)
        new_lower = spring_redistribute_centers(level.lower, lower_scores, A=A)
    except Exception as err:
        print(
            "[PROGRESSIVE_HH][SPRING] WARNING: post-opt coefficient spring failed; "
            f"{err}"
        )
        return None

    print("[PROGRESSIVE_HH][SPRING] POST_OPT coefficient spring")
    print("[PROGRESSIVE_HH][SPRING] upper |a| =", upper_scores)
    print("[PROGRESSIVE_HH][SPRING] lower |a| =", lower_scores)
    print("[PROGRESSIVE_HH][SPRING] Upper before:", sorted(level.upper))
    print("[PROGRESSIVE_HH][SPRING] Upper after :", new_upper)
    print("[PROGRESSIVE_HH][SPRING] Lower before:", sorted(level.lower))
    print("[PROGRESSIVE_HH][SPRING] Lower after :", new_lower)

    result["spring_upper_coeff_abs"] = upper_scores
    result["spring_lower_coeff_abs"] = lower_scores

    return sorted(new_upper), sorted(new_lower)


def should_refine(history, opts, level_id):
    if opts.get("nfinal", None) is not None:
        return True

    if level_id >= opts["nlevels"] - 1:
        return False

    if opts["trigger"] == "MAX_ITER":
        return True

    return False
