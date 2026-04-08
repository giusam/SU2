#!/usr/bin/env python

import math

from SU2.opt.hh_spring import apply_hh_spring_after_selection
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
    ):
        self.level_id = level_id
        self.upper = list(upper)
        self.lower = list(lower)
        self.workdir = workdir
        self.config_filename = config_filename
        self.project_filename = project_filename
        self.mesh_source = mesh_source

    @property
    def ndv(self):
        return len(self.upper) + len(self.lower)


def _as_bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().upper() in ("YES", "TRUE", "1", "ON")


def get_progressive_hh_options(config):
    scale = 1.0
    if "DEFINITION_DV" in config:
        def_dv = config["DEFINITION_DV"]
        if "SCALE" in def_dv and def_dv["SCALE"]:
            try:
                scale = float(def_dv["SCALE"][0])
            except Exception:
                pass

    return {
        "enabled": _as_bool(config.get("PROGRESSIVE_HH", "NO")),
        "nlevels": int(config.get("PROGRESSIVE_HH_NLEVELS", 1)),
        "n0": int(config.get("PROGRESSIVE_HH_N0", 3)),
        "surface_mode": str(config.get("PROGRESSIVE_HH_SURFACE", "BOTH")).upper(),
        "trigger": str(config.get("PROGRESSIVE_HH_TRIGGER", "MAX_ITER")).upper(),
        "window": int(config.get("PROGRESSIVE_HH_WINDOW", 1)),
        "tol": float(config.get("PROGRESSIVE_HH_TOL", 0.2)),
        "slope_filter_tol": float(config.get("PROGRESSIVE_HH_SLOPE_FILTER_TOL", 0.02)),
        "stag_tol": float(config.get("PROGRESSIVE_HH_STAG_TOL", 1.0e-3)),
        "stag_band": float(config.get("PROGRESSIVE_HH_STAG_BAND", 0.02)),
        "stag_window": int(config.get("PROGRESSIVE_HH_STAG_WINDOW", 3)),
        "warmup_iter": int(config.get("PROGRESSIVE_HH_WARMUP_ITER", 0)),
        "max_iter_per_level": int(
            config.get("PROGRESSIVE_HH_MAX_ITER_PER_LEVEL", config.OPT_ITERATIONS)
        ),
        "refinement": str(config.get("PROGRESSIVE_HH_REFINEMENT", "UNIFORM")).upper(),
        "growth_ratio": float(config.get("PROGRESSIVE_HH_GROWTH_RATIO", 2.0)),
        "adaptive_indicator": str(
            config.get("PROGRESSIVE_HH_ADAPTIVE_INDICATOR", "ABS_GRAD")
        ).upper(),
        "marker": str(config.get("DV_MARKER", "Airfoil")),
        "scale": scale,
        "spring_enabled": _as_bool(config.get("PROGRESSIVE_HH_SPRING", "NO")),
        "spring_A": float(config.get("PROGRESSIVE_HH_SPRING_A", 20.0)),
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


def refine_adaptive(prev_level, result, opts):
    current_ndv = prev_level.ndv

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

    nadd = _compute_adaptive_nadd(
        current_ndv, len(candidates), opts["growth_ratio"]
    )

    chosen = _select_top_candidates(candidates, nadd)

    spring_debug = None

    if opts.get("spring_enabled", False):
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

    print(
        f"[PROGRESSIVE_HH] ADAPTIVE refine | add={nadd} "
        f"spring={opts.get('spring_enabled')}"
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

    return sorted(set(new_upper)), sorted(set(new_lower))


def should_refine(history, opts, level_id):
    if level_id >= opts["nlevels"] - 1:
        return False

    if opts["trigger"] == "MAX_ITER":
        return True

    return False