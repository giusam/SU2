"""Settings, config parsing, and startup helpers for adaptive B-splines."""

from pathlib import Path


from SU2.opt.bspline_dot import normalize_sensitivity_weighting
from SU2.opt.bspline_modes import (
    BSplineModeError,
    LE_SAFE_DEFAULT_POWER,
    LE_SAFE_DEFAULT_X0,
    LE_SAFE_DEFAULT_X1,
    active_sides_from_surface_mode,
    load_mode_spec,
    normalize_deformation_direction_mode,
    normalize_surface_mode,
    validate_le_safe_direction_options,
    validate_mode_spec,
)
from SU2.opt.bspline_driver.constants import ALLOWED_SYMMETRY_COUPLINGS
from SU2.opt.bspline_driver.config_apply import (
    fixed_driver_options_from_config,
    resolve_thickness_domain_mode,
)
from SU2.opt.bspline_driver.errors import BSplineSU2DriverError
from SU2.opt.bspline_driver.reduction import (
    active_mode_ids,
    write_mode_spec,
)

from .errors import (
    BSplineAdaptiveError,
    _as_bool,
    _as_float,
)

ALLOWED_TRIGGERS = (
    "MAX_ITER",
    "SLOPE_EFFICIENCY_TRIGGER",
    "SLOPE_EFFICIENCY_FILTERED",
    "SLOPE_EFFICIENCY_BEST_LOG",
    "STAGNATION_TRIGGER",
)

ALLOWED_NADD_MODES = ("GROWTH_RATIO", "FIXED")

ALLOWED_REFINE_MODES = ("KNOT_INSERTION",)

ALLOWED_KNOT_SCORE_MODES = ("VIRTUAL_INSERTION", "RESIDUAL_ENERGY")

REMOVED_CANDIDATE_CONFIG_KEYS = (
    "BSPLINE_SCORE_MODE",
    "BSPLINE_CANDIDATE_SOURCE",
    "BSPLINE_GENERATED_PEAKS_PER_SIDE",
    "BSPLINE_GENERATED_WIDTHS",
    "BSPLINE_GENERATED_MIN_SEPARATION",
    "BSPLINE_GENERATED_XMIN",
    "BSPLINE_GENERATED_XMAX",
    "BSPLINE_EDGE_XLE",
    "BSPLINE_EDGE_XTE",
    "BSPLINE_ROUGH_LAMBDA",
    "BSPLINE_ROUGH_POWER",
    "BSPLINE_BATCH_SCORE_REL_TOL",
)

GLOBAL_MODE_KEYS = (
    "version",
    "dimension",
    "marker",
    "chord",
    "normal_displacement",
    "class_shape",
    "class_shape_exponent",
    "normalize_basis",
    "normalization_mode",
    "surface_mode",
    "knot_span_depths",
)

KNOT_SCORE_FIELDNAMES = [
    "batch_step",
    "rank",
    "span_left",
    "span_right",
    "span_width",
    "inserted_knot",
    "side",
    "score_mode",
    "score",
    "score_raw",
    "score_effective",
    "selection_score",
    "span_key",
    "parent_depth",
    "child_depth",
    "depth_penalty",
    "depth_penalty_mode",
    "batch_depth",
    "batch_penalty",
    "batch_penalty_mode",
    "residual_energy",
    "incremental_rank",
    "incremental_columns",
    "condition_number",
    "selected",
    "status",
]

def _mode_display(mode):
    left, right, _center = _mode_support(mode)
    return (
        f"{mode['id']} | side={mode.get('side')} | "
        f"support=[{left:.6f},{right:.6f}] | "
        f"coeff={float(mode.get('coefficient', 0.0)):.6e}"
    )

def _print_level_start(
    level,
    refine_state,
    kept=0,
    added=0,
    log_active_modes=False,
):
    from .mode_utils import _active_modes

    print(f"[PROGRESSIVE_BSPLINE] Level {level.level_id} | NDV = {level.ndv}")
    print(f"[PROGRESSIVE_BSPLINE] Refinement state: {refine_state}")
    if level.level_id > 0:
        print(f"[PROGRESSIVE_BSPLINE] Kept optimized coefficients from previous level: {kept}")
        print(f"[PROGRESSIVE_BSPLINE] Added/refined active modes: {added}")
    if log_active_modes:
        print("[PROGRESSIVE_BSPLINE] Active modes:")
        for mode in _active_modes(level.active_modes):
            print("[PROGRESSIVE_BSPLINE]   " + _mode_display(mode))
    else:
        sides = sorted(
            {
                str(mode.get("side", "")).strip().lower()
                for mode in _active_modes(level.active_modes)
            }
        )
        print(
            "[PROGRESSIVE_BSPLINE] Active mode summary: n={} sides={}".format(
                level.ndv,
                ",".join(side for side in sides if side) or "unknown",
            )
        )

def validate_adaptive_options(opts):
    opts = dict(opts)
    opts["refinement"] = str(opts.get("refinement", "ADAPTIVE")).upper()
    if opts["refinement"] != "ADAPTIVE":
        raise BSplineAdaptiveError(
            "B-spline progressive optimization supports only --refinement ADAPTIVE"
        )

    opts["refine_state"] = str(opts.get("refine_state", REFINE_STATE)).upper()
    if opts["refine_state"] != REFINE_STATE:
        raise BSplineAdaptiveError(
            "BSPLINE_REFINE_STATE is fixed internally to INITIAL_MESH_KEEP_DV."
        )

    if opts.get("score_mode") is not None:
        raise BSplineAdaptiveError(
            "This cfg contains removed candidate/generated B-spline options: "
            "BSPLINE_SCORE_MODE. Remove them. The adaptive optimizer now "
            "supports only internal KNOT_INSERTION."
        )
    opts["refine_mode"] = str(opts.get("refine_mode", REFINE_MODE)).upper()
    if opts["refine_mode"] != REFINE_MODE:
        raise BSplineAdaptiveError(
            "BSPLINE_REFINE_MODE is no longer user-configurable. "
            "Candidate/generated refinement has been removed; KNOT_INSERTION "
            "is fixed internally."
        )
    opts["knot_score_mode"] = str(opts.get("knot_score_mode", "VIRTUAL_INSERTION")).upper()
    if opts["knot_score_mode"] not in ALLOWED_KNOT_SCORE_MODES:
        raise BSplineAdaptiveError(
            f"unsupported knot score mode {opts['knot_score_mode']!r}; allowed values are {ALLOWED_KNOT_SCORE_MODES}"
        )
    mode = _normalize_knot_depth_penalty_mode(
        opts.get(
            "knot_depth_penalty_mode",
            opts.get("knot_batch_penalty_mode", "NONE"),
        )
    )
    depth_penalty = _as_bool(
        opts.get(
            "knot_depth_penalty",
            opts.get("knot_batch_diversity", mode != "NONE"),
        ),
        default=False,
    )
    if depth_penalty and mode == "NONE":
        raise BSplineAdaptiveError(
            "BSPLINE_KNOT_DEPTH_PENALTY=YES requires "
            "BSPLINE_KNOT_DEPTH_PENALTY_MODE=STREUBER_DEPTH or POWER"
        )
    if not depth_penalty:
        mode = "NONE"
    opts["knot_depth_penalty"] = depth_penalty
    opts["knot_depth_penalty_mode"] = mode
    opts["knot_depth_power_gamma"] = _as_float(
        opts.get(
            "knot_depth_power_gamma",
            opts.get("knot_batch_power_gamma", 0.25),
        ),
        "BSPLINE_KNOT_DEPTH_POWER_GAMMA",
    )
    if not (0.0 < opts["knot_depth_power_gamma"] <= 1.0):
        raise BSplineAdaptiveError(
            "BSPLINE_KNOT_DEPTH_POWER_GAMMA must satisfy 0 < gamma <= 1"
        )
    opts["knot_initial_span_depth"] = _as_knot_depth(
        opts.get("knot_initial_span_depth", 1),
        "BSPLINE_KNOT_INITIAL_SPAN_DEPTH",
    )
    # Legacy aliases remain populated for old cfg/CLI consumers and logs.
    opts["knot_batch_diversity"] = depth_penalty
    opts["knot_batch_penalty_mode"] = mode
    opts["knot_batch_power_gamma"] = opts["knot_depth_power_gamma"]
    knot_insertions = opts.get("knot_insertions_per_refine", 1)
    if str(knot_insertions).strip().upper() == "AUTO":
        opts["knot_insertions_per_refine"] = "AUTO"
    else:
        opts["knot_insertions_per_refine"] = int(knot_insertions)
        if opts["knot_insertions_per_refine"] < 1:
            raise BSplineAdaptiveError("--knot-insertions-per-refine must be AUTO or >= 1")
    opts["knot_min_span_width"] = _as_float(
        opts.get("knot_min_span_width", 1.0e-8),
        "BSPLINE_KNOT_MIN_SPAN_WIDTH",
    )
    if opts["knot_min_span_width"] <= 0.0:
        raise BSplineAdaptiveError("--knot-min-span-width must be positive")
    opts["transfer_method"] = str(
        opts.get("transfer_method", "BOEHM")
    ).strip().upper()
    if opts["transfer_method"] != "BOEHM":
        raise BSplineAdaptiveError(
            "Only BSPLINE_TRANSFER_METHOD=BOEHM is currently implemented."
        )
    opts["transfer_bound_policy"] = str(
        opts.get("transfer_bound_policy", "ERROR")
    ).strip().upper()
    if opts["transfer_bound_policy"] != "ERROR":
        raise BSplineAdaptiveError(
            "Only BSPLINE_TRANSFER_BOUND_POLICY=ERROR is currently implemented."
        )
    opts["transfer_geometry_abs_tol"] = _as_float(
        opts.get("transfer_geometry_abs_tol", 1.0e-10),
        "BSPLINE_TRANSFER_GEOMETRY_ABS_TOL",
    )
    opts["transfer_geometry_rel_tol"] = _as_float(
        opts.get("transfer_geometry_rel_tol", 1.0e-8),
        "BSPLINE_TRANSFER_GEOMETRY_REL_TOL",
    )
    if opts["transfer_geometry_abs_tol"] < 0.0:
        raise BSplineAdaptiveError(
            "BSPLINE_TRANSFER_GEOMETRY_ABS_TOL must be non-negative"
        )
    if opts["transfer_geometry_rel_tol"] < 0.0:
        raise BSplineAdaptiveError(
            "BSPLINE_TRANSFER_GEOMETRY_REL_TOL must be non-negative"
        )

    opts["trigger"] = str(opts.get("trigger", "MAX_ITER")).upper()
    if opts["trigger"] not in ALLOWED_TRIGGERS:
        raise BSplineAdaptiveError(
            f"unsupported trigger {opts['trigger']!r}; allowed values are {ALLOWED_TRIGGERS}"
        )

    opts["nadd_mode"] = str(opts.get("nadd_mode", "GROWTH_RATIO")).upper()
    if opts["nadd_mode"] == "SCORE_BATCH":
        raise BSplineAdaptiveError(
            "BSPLINE_NADD_MODE=SCORE_BATCH was part of the removed "
            "candidate/generated strategy. Use GROWTH_RATIO or FIXED."
        )
    if opts["nadd_mode"] not in ALLOWED_NADD_MODES:
        raise BSplineAdaptiveError(
            f"unsupported nadd mode {opts['nadd_mode']!r}; allowed values are {ALLOWED_NADD_MODES}"
        )

    removed_runtime = []
    for key in (
        "candidate_source",
        "generated_peaks_per_side",
        "generated_widths",
        "generated_min_separation",
        "generated_xmin",
        "generated_xmax",
        "edge_xle",
        "edge_xte",
        "rough_lambda",
        "rough_power",
        "batch_score_rel_tol",
    ):
        if opts.get(key) is not None:
            removed_runtime.append(key)
    if removed_runtime:
        raise BSplineAdaptiveError(
            "This cfg contains removed candidate/generated B-spline options: "
            + ", ".join(sorted(removed_runtime))
            + ". Remove them. The adaptive optimizer now supports only "
            "internal KNOT_INSERTION."
        )

    sensitivity_weighting = normalize_sensitivity_weighting(
        opts.get("sensitivity_weighting", "NODAL")
    )
    if sensitivity_weighting != "NODAL":
        raise BSplineAdaptiveError(
            "BSPLINE_SENSITIVITY_WEIGHTING=DENSITY is not supported by "
            "adaptive KNOT_INSERTION. NODAL is fixed internally."
        )
    opts["sensitivity_weighting"] = "NODAL"

    opts["eval_layout"] = str(opts.get("eval_layout", "DSN")).upper()
    if opts["eval_layout"] != "DSN":
        raise BSplineAdaptiveError(
            "BSPLINE_EVAL_LAYOUT=FLAT has been removed. DSN is the only supported layout."
        )
    opts["eval_layout"] = "DSN"
    opts["symmetry_coupling"] = str(opts.get("symmetry_coupling", "NONE")).upper()
    if opts["symmetry_coupling"] not in ALLOWED_SYMMETRY_COUPLINGS:
        raise BSplineAdaptiveError(
            f"unsupported symmetry coupling {opts['symmetry_coupling']!r}; allowed values are {ALLOWED_SYMMETRY_COUPLINGS}"
        )
    try:
        opts["surface_mode"] = normalize_surface_mode(
            opts.get("surface_mode", "BOTH")
        )
    except BSplineModeError as exc:
        raise BSplineAdaptiveError(str(exc))
    if opts["surface_mode"] != "BOTH" and opts["symmetry_coupling"] != "NONE":
        raise BSplineAdaptiveError(
            "BSPLINE_SYMMETRY_COUPLING is only valid with BSPLINE_SURFACE_MODE=BOTH"
        )
    opts["objective_adjoint"] = str(opts.get("objective_adjoint", "drag")).strip() or "drag"

    opts["auto_scale_bounds_to_geometry"] = bool(opts.get("auto_scale_bounds_to_geometry", False))
    opts["max_normal_displacement"] = (
        float(opts["max_normal_displacement"])
        if opts.get("max_normal_displacement") is not None
        else None
    )
    opts["max_rms_normal_displacement"] = (
        float(opts["max_rms_normal_displacement"])
        if opts.get("max_rms_normal_displacement") is not None
        else None
    )
    opts["min_bound_scale"] = float(opts.get("min_bound_scale", 0.0))
    if opts["min_bound_scale"] < 0.0:
        raise BSplineAdaptiveError("--min-bound-scale must be non-negative")
    if opts["auto_scale_bounds_to_geometry"] and (
        opts["max_normal_displacement"] is None
        and opts["max_rms_normal_displacement"] is None
    ):
        raise BSplineAdaptiveError(
            "at least one of --max-normal-displacement or --max-rms-normal-displacement must be provided when --auto-scale-bounds-to-geometry is enabled"
        )

    opts["nlevels"] = max(1, int(opts.get("nlevels", 1)))
    opts["nfinal"] = int(opts["nfinal"]) if opts.get("nfinal") is not None else None
    opts["max_iter_per_level"] = max(1, int(opts.get("max_iter_per_level", 5)))
    opts["window"] = max(1, int(opts.get("window", 1)))
    opts["tol"] = float(opts.get("tol", 0.2))
    opts["eps"] = float(opts.get("eps", 1.0e-300))
    if opts["eps"] <= 0.0:
        raise BSplineAdaptiveError("--trigger-eps must be positive")
    opts["slope_filter_tol"] = float(opts.get("slope_filter_tol", 0.02))
    opts["slope_patience"] = max(1, int(opts.get("slope_patience", 1)))
    opts["stag_window"] = max(1, int(opts.get("stag_window", opts["window"])))
    opts["stag_tol"] = float(opts.get("stag_tol", 1.0e-3))
    opts["stag_band"] = float(opts.get("stag_band", 0.02))
    opts["stag_patience"] = max(1, int(opts.get("stag_patience", 1)))
    opts["warmup_iter"] = max(0, int(opts.get("warmup_iter", 0)))
    opts["growth_ratio"] = float(opts.get("growth_ratio", 2.0))
    opts["fixed_nadd"] = max(1, int(opts.get("fixed_nadd", 1)))
    opts["batch_size_max"] = max(1, int(opts.get("batch_size_max", 1)))
    opts["opt_accuracy"] = (
        float(opts["opt_accuracy"])
        if opts.get("opt_accuracy") is not None
        else None
    )
    opts["opt_bound_upper"] = (
        float(opts["opt_bound_upper"])
        if opts.get("opt_bound_upper") is not None
        else None
    )
    opts["opt_bound_lower"] = (
        float(opts["opt_bound_lower"])
        if opts.get("opt_bound_lower") is not None
        else None
    )
    if (opts["opt_bound_lower"] is None) != (opts["opt_bound_upper"] is None):
        raise BSplineAdaptiveError(
            "--opt-bound-lower and --opt-bound-upper must be provided together"
        )
    opts["opt_relax_factor"] = float(
        1.0 if opts.get("opt_relax_factor") is None else opts.get("opt_relax_factor")
    )
    if opts["opt_relax_factor"] <= 0.0:
        raise BSplineAdaptiveError("--opt-relax-factor must be positive")
    opts["opt_gradient_factor"] = float(
        1.0
        if opts.get("opt_gradient_factor") is None
        else opts.get("opt_gradient_factor")
    )
    if opts["opt_gradient_factor"] <= 0.0:
        raise BSplineAdaptiveError("--opt-gradient-factor must be positive")
    opts["gradient_guard"] = _as_bool(
        opts.get("gradient_guard", True),
        default=True,
    )
    opts["gradient_guard_factor"] = _as_float(
        opts.get("gradient_guard_factor", 100.0),
        "BSPLINE_GRADIENT_GUARD_FACTOR",
    )
    opts["gradient_guard_window"] = int(opts.get("gradient_guard_window", 5))
    opts["gradient_guard_min_history"] = int(
        opts.get("gradient_guard_min_history", 3)
    )
    opts["gradient_guard_floor"] = _as_float(
        opts.get("gradient_guard_floor", 1.0e-14),
        "BSPLINE_GRADIENT_GUARD_FLOOR",
    )
    opts["gradient_guard_restart_limit"] = int(
        opts.get("gradient_guard_restart_limit", 2)
    )
    if opts["gradient_guard_factor"] <= 0.0:
        raise BSplineAdaptiveError("BSPLINE_GRADIENT_GUARD_FACTOR must be positive")
    if opts["gradient_guard_window"] < 1:
        raise BSplineAdaptiveError("BSPLINE_GRADIENT_GUARD_WINDOW must be >= 1")
    if opts["gradient_guard_min_history"] < 1:
        raise BSplineAdaptiveError("BSPLINE_GRADIENT_GUARD_MIN_HISTORY must be >= 1")
    if opts["gradient_guard_floor"] <= 0.0:
        raise BSplineAdaptiveError("BSPLINE_GRADIENT_GUARD_FLOOR must be positive")
    if opts["gradient_guard_restart_limit"] < 0:
        raise BSplineAdaptiveError("BSPLINE_GRADIENT_GUARD_RESTART_LIMIT must be >= 0")
    opts["trust_clip_policy"] = str(
        opts.get("trust_clip_policy", "OFF")
    ).strip().upper()
    if opts["trust_clip_policy"] not in ("OFF", "ACCEPT_RESTART"):
        raise BSplineAdaptiveError(
            "BSPLINE_TRUST_CLIP_POLICY must be OFF or ACCEPT_RESTART"
        )
    for key, default in (
        ("trust_clip_beta_tol", 1.0e-12),
        ("trust_clip_legacy_beta_min", 0.50),
        ("trust_clip_severe_beta", 0.50),
        ("trust_clip_worsening_tol", 0.05),
        ("trust_clip_soft_gnorm_factor", 20.0),
        ("trust_clip_stag_tol", 1.0e-6),
    ):
        opts[key] = float(opts.get(key, default))
    opts["trust_clip_bad_patience"] = int(
        opts.get("trust_clip_bad_patience", 2)
    )
    opts["trust_clip_bad_window"] = int(opts.get("trust_clip_bad_window", 5))
    opts["trust_clip_restart_limit"] = int(
        opts.get("trust_clip_restart_limit", 1)
    )
    if opts["trust_clip_beta_tol"] < 0.0:
        raise BSplineAdaptiveError("BSPLINE_TRUST_CLIP_BETA_TOL must be non-negative")
    if not 0.0 <= opts["trust_clip_legacy_beta_min"] <= 1.0:
        raise BSplineAdaptiveError("BSPLINE_TRUST_CLIP_LEGACY_BETA_MIN must be in [0, 1]")
    if not 0.0 <= opts["trust_clip_severe_beta"] <= 1.0:
        raise BSplineAdaptiveError("BSPLINE_TRUST_CLIP_SEVERE_BETA must be in [0, 1]")
    if opts["trust_clip_worsening_tol"] < 0.0 or opts["trust_clip_stag_tol"] < 0.0:
        raise BSplineAdaptiveError("trust-clip tolerances must be non-negative")
    if opts["trust_clip_soft_gnorm_factor"] <= 0.0:
        raise BSplineAdaptiveError("BSPLINE_TRUST_CLIP_SOFT_GNORM_FACTOR must be positive")
    if opts["trust_clip_bad_patience"] < 1 or opts["trust_clip_bad_window"] < 1:
        raise BSplineAdaptiveError("trust-clip bad patience/window must be >= 1")
    if opts["trust_clip_restart_limit"] < 0:
        raise BSplineAdaptiveError("BSPLINE_TRUST_CLIP_RESTART_LIMIT must be >= 0")
    opts["opt_line_search_bound"] = (
        float(opts["opt_line_search_bound"])
        if opts.get("opt_line_search_bound") is not None
        else None
    )
    if opts["opt_line_search_bound"] is not None and opts["opt_line_search_bound"] <= 0.0:
        raise BSplineAdaptiveError("--opt-line-search-bound must be positive")
    opts["local_step_limit"] = _as_bool(opts.get("local_step_limit", False), default=False)
    opts["log_active_modes"] = _as_bool(opts.get("log_active_modes", False), default=False)
    opts["local_step_limit_ratio"] = _as_float(
        opts.get("local_step_limit_ratio", 200.0),
        "BSPLINE_LOCAL_STEP_LIMIT_RATIO",
    )
    if opts["local_step_limit_ratio"] <= 0.0:
        raise BSplineAdaptiveError("--local-step-limit-ratio must be positive")
    opts["thickness_options"] = dict(opts.get("thickness_options") or {})
    if _as_bool(
        opts["thickness_options"].get("PROGRESSIVE_THICKNESS_CONSTRAINT", False),
        default=False,
    ):
        try:
            domain_mode = resolve_thickness_domain_mode(
                opts["surface_mode"],
                opts["thickness_options"].get(
                    "PROGRESSIVE_THICKNESS_DOMAIN_MODE",
                    "AUTO",
                ),
            )
        except BSplineSU2DriverError as exc:
            raise BSplineAdaptiveError(str(exc))
        opts["thickness_options"]["PROGRESSIVE_THICKNESS_DOMAIN_MODE"] = domain_mode

    try:
        direction_mode = normalize_deformation_direction_mode(
            opts.get("deformation_direction_mode"),
            le_safe_direction=opts.get("le_safe_direction", False),
        )
        le_safe_opts = validate_le_safe_direction_options(
            le_safe_direction=direction_mode == "LE_SAFE",
            le_safe_x0=(
                opts.get("le_safe_x0", LE_SAFE_DEFAULT_X0)
                if direction_mode == "LE_SAFE"
                else LE_SAFE_DEFAULT_X0
            ),
            le_safe_x1=(
                opts.get("le_safe_x1", LE_SAFE_DEFAULT_X1)
                if direction_mode == "LE_SAFE"
                else LE_SAFE_DEFAULT_X1
            ),
            le_safe_power=(
                opts.get("le_safe_power", LE_SAFE_DEFAULT_POWER)
                if direction_mode == "LE_SAFE"
                else LE_SAFE_DEFAULT_POWER
            ),
        )
    except BSplineModeError as exc:
        raise BSplineAdaptiveError(str(exc))
    opts["deformation_direction_mode"] = direction_mode
    opts["le_safe_direction"] = le_safe_opts["le_safe_direction"]
    opts["le_safe_x0"] = le_safe_opts["le_safe_x0"]
    opts["le_safe_x1"] = le_safe_opts["le_safe_x1"]
    opts["le_safe_power"] = le_safe_opts["le_safe_power"]
    return opts

def adaptive_options_from_config(config_values):
    removed_keys = [key for key in REMOVED_CANDIDATE_CONFIG_KEYS if key in config_values]
    if removed_keys:
        raise BSplineAdaptiveError(
            "This cfg contains removed candidate/generated B-spline options: "
            + ", ".join(sorted(removed_keys))
            + ". Remove them. The adaptive optimizer now supports only "
            "internal KNOT_INSERTION."
        )
    options = fixed_driver_options_from_config(config_values)
    if "maxiter" in options:
        options["max_iter_per_level"] = options.pop("maxiter")
    case_mapping = {
        "BSPLINE_BASE_MESH": "base_mesh",
        "BSPLINE_MARKER": "marker",
        "BSPLINE_WORKDIR": "workdir",
        "BSPLINE_MPI": "mpi",
        "BSPLINE_MODES": "modes",
        "BSPLINE_DEF_TEMPLATE": "def_template",
        "BSPLINE_PRIMAL_TEMPLATE": "primal_template",
        "BSPLINE_ADJOINT_TEMPLATE": "adjoint_template",
        "BSPLINE_GENERATE_INITIAL_MODES": "generate_initial_modes",
        "BSPLINE_INITIAL_NPER_SIDE": "initial_nper_side",
        "BSPLINE_INITIAL_DEGREE": "initial_degree",
        "BSPLINE_INITIAL_COEFFICIENT": "initial_coefficient",
        "BSPLINE_INITIAL_BOUND_LOWER": "initial_bound_lower",
        "BSPLINE_INITIAL_BOUND_UPPER": "initial_bound_upper",
        "BSPLINE_INITIAL_NORMALIZE_BASIS": "initial_normalize_basis",
        "BSPLINE_INITIAL_NORMALIZATION_MODE": "initial_normalization_mode",
        "BSPLINE_INITIAL_CLASS_SHAPE_EXPONENT": "initial_class_shape_exponent",
    }
    for key, dest in case_mapping.items():
        if key in config_values:
            options[dest] = config_values[key]
    if "BSPLINE_INITIAL_CLASS_SHAPE" in config_values:
        options["initial_class_shape"] = config_values["BSPLINE_INITIAL_CLASS_SHAPE"]
    elif "BSPLINE_USE_CLASS_SHAPE" in config_values:
        options["initial_class_shape"] = (
            "sqrt_x_one_minus_x"
            if _as_bool(config_values["BSPLINE_USE_CLASS_SHAPE"], default=True)
            else "none"
        )

    if "base_mesh" not in options and config_values.get("MESH_FILENAME"):
        options["base_mesh"] = config_values["MESH_FILENAME"]
    marker = _infer_marker_from_config(config_values)
    if marker is not None:
        options["marker"] = marker

    objective = str(
        config_values.get(
            "OPT_OBJECTIVE",
            config_values.get("OBJECTIVE_FUNCTION", ""),
        )
    ).strip().strip('"').strip("'").upper()
    if objective:
        options["opt_objective"] = objective
        if "objective_column" not in options and objective == "DRAG":
            options["objective_column"] = "CD"
        if "objective_adjoint" not in options and objective == "DRAG":
            options["objective_adjoint"] = "drag"

    for dest in (
        "base_mesh",
        "modes",
        "def_template",
        "primal_template",
        "adjoint_template",
        "workdir",
    ):
        if dest in options and options[dest]:
            options[dest] = _resolve_cfg_path(config_values, options[dest])
    options["_case_config"] = config_values.get("_optimizer_config_filename")

    mapping = {
        "BSPLINE_NLEVELS": "nlevels",
        "BSPLINE_NFINAL": "nfinal",
        "BSPLINE_REFINE_MODE": "refine_mode",
        "BSPLINE_REFINE_STATE": "refine_state",
        "BSPLINE_SENSITIVITY_WEIGHTING": "sensitivity_weighting",
        "BSPLINE_KNOT_SCORE_MODE": "knot_score_mode",
        "BSPLINE_KNOT_INSERTIONS_PER_REFINE": "knot_insertions_per_refine",
        "BSPLINE_KNOT_MIN_SPAN_WIDTH": "knot_min_span_width",
        "BSPLINE_KNOT_DEPTH_PENALTY": "knot_depth_penalty",
        "BSPLINE_KNOT_DEPTH_PENALTY_MODE": "knot_depth_penalty_mode",
        "BSPLINE_KNOT_DEPTH_POWER_GAMMA": "knot_depth_power_gamma",
        "BSPLINE_KNOT_INITIAL_SPAN_DEPTH": "knot_initial_span_depth",
        "BSPLINE_KNOT_BATCH_DIVERSITY": "knot_batch_diversity",
        "BSPLINE_KNOT_BATCH_PENALTY_MODE": "knot_batch_penalty_mode",
        "BSPLINE_KNOT_BATCH_POWER_GAMMA": "knot_batch_power_gamma",
        "BSPLINE_TRANSFER_METHOD": "transfer_method",
        "BSPLINE_TRANSFER_BOUND_POLICY": "transfer_bound_policy",
        "BSPLINE_TRANSFER_GEOMETRY_ABS_TOL": "transfer_geometry_abs_tol",
        "BSPLINE_TRANSFER_GEOMETRY_REL_TOL": "transfer_geometry_rel_tol",
        "BSPLINE_TRIGGER": "trigger",
        "BSPLINE_TRIGGER_WINDOW": "window",
        "BSPLINE_TRIGGER_RATIO": "tol",
        "BSPLINE_TRIGGER_EPS": "eps",
        "BSPLINE_TRIGGER_WARMUP_ITER": "warmup_iter",
        "BSPLINE_SLOPE_FILTER_TOL": "slope_filter_tol",
        "BSPLINE_SLOPE_PATIENCE": "slope_patience",
        "BSPLINE_STAGNATION_WINDOW": "stag_window",
        "BSPLINE_STAGNATION_REL_TOL": "stag_tol",
        "BSPLINE_STAGNATION_BAND": "stag_band",
        "BSPLINE_STAGNATION_PATIENCE": "stag_patience",
        "BSPLINE_NADD_MODE": "nadd_mode",
        "BSPLINE_FIXED_NADD": "fixed_nadd",
        "BSPLINE_GROWTH_RATIO": "growth_ratio",
        "BSPLINE_BATCH_SIZE_MAX": "batch_size_max",
        "BSPLINE_LOG_ACTIVE_MODES": "log_active_modes",
    }
    for key, dest in mapping.items():
        if key in config_values:
            options[dest] = config_values[key]
    if (
        "BSPLINE_KNOT_DEPTH_PENALTY" not in config_values
        and "BSPLINE_KNOT_BATCH_DIVERSITY" in config_values
    ):
        options["knot_depth_penalty"] = config_values[
            "BSPLINE_KNOT_BATCH_DIVERSITY"
        ]
    if (
        "BSPLINE_KNOT_DEPTH_PENALTY_MODE" not in config_values
        and "BSPLINE_KNOT_BATCH_PENALTY_MODE" in config_values
    ):
        options["knot_depth_penalty_mode"] = config_values[
            "BSPLINE_KNOT_BATCH_PENALTY_MODE"
        ]
    if (
        "BSPLINE_KNOT_DEPTH_POWER_GAMMA" not in config_values
        and "BSPLINE_KNOT_BATCH_POWER_GAMMA" in config_values
    ):
        options["knot_depth_power_gamma"] = config_values[
            "BSPLINE_KNOT_BATCH_POWER_GAMMA"
        ]
    return options

def _strip_cfg_atom(value):
    return str(value).strip().strip('"').strip("'")

def _cfg_value_list(value):
    if isinstance(value, (list, tuple)):
        tokens = list(value)
    else:
        text = _strip_cfg_atom(value)
        if (text.startswith("(") and text.endswith(")")) or (
            text.startswith("[") and text.endswith("]")
        ):
            text = text[1:-1]
        tokens = [token for token in text.replace(",", " ").split() if token]
    return [_strip_cfg_atom(token) for token in tokens if _strip_cfg_atom(token)]

def _infer_marker_from_config(config_values):
    for key in ("BSPLINE_MARKER", "DV_MARKER", "MARKER_MONITORING", "MARKER_PLOTTING"):
        if key not in config_values or config_values[key] in (None, ""):
            continue
        markers = _cfg_value_list(config_values[key])
        if len(markers) == 1:
            return markers[0]
        clear = [
            marker
            for marker in markers
            if str(marker).strip().upper() in ("AIRFOIL", "WALL", "WING", "BODY")
            or "AIRFOIL" in str(marker).strip().upper()
        ]
        if len(clear) == 1:
            return clear[0]
        raise BSplineAdaptiveError(
            "Could not uniquely infer BSPLINE_MARKER. Please set BSPLINE_MARKER= ..."
        )
    return None

def _resolve_cfg_path(config_values, value):
    path = Path(_strip_cfg_atom(value)).expanduser()
    if path.is_absolute():
        return str(path)
    cfg_filename = config_values.get("_optimizer_config_filename")
    if not cfg_filename:
        return str(path)
    return str((Path(cfg_filename).resolve().parent / path).resolve())

def _format_cfg_value(value):
    if isinstance(value, (list, tuple)):
        return "( " + ", ".join(_format_cfg_value(item) for item in value) + " )"
    if isinstance(value, bool):
        return "YES" if value else "NO"
    return str(value)

def _write_cfg(filename, values):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w") as fp:
        for key, value in values:
            fp.write(f"{key}= {_format_cfg_value(value)}\n")

def generate_initial_bspline_modes(
    filename,
    marker,
    nper_side=7,
    degree=3,
    coefficient=0.0,
    bound_lower=-0.01,
    bound_upper=0.01,
    class_shape="sqrt_x_one_minus_x",
    class_shape_exponent=0.5,
    normalize_basis=True,
    normalization_mode="max",
    surface_mode="BOTH",
):
    nper_side = int(nper_side)
    degree = int(degree)
    if degree != 3:
        raise BSplineAdaptiveError("BSPLINE_INITIAL_DEGREE must currently be 3 for KNOT_INSERTION")
    if nper_side < degree + 1:
        raise BSplineAdaptiveError("BSPLINE_INITIAL_NPER_SIDE must be >= degree + 1")
    bound_lower = float(bound_lower)
    bound_upper = float(bound_upper)
    if bound_upper < bound_lower:
        raise BSplineAdaptiveError("BSPLINE_INITIAL_BOUND_UPPER must be >= lower")
    class_shape_exponent = _as_float(
        class_shape_exponent,
        "BSPLINE_INITIAL_CLASS_SHAPE_EXPONENT",
    )
    if class_shape_exponent < 0.0:
        raise BSplineAdaptiveError(
            "BSPLINE_INITIAL_CLASS_SHAPE_EXPONENT must be finite and >= 0"
        )
    marker = str(marker or "").strip()
    if not marker:
        raise BSplineAdaptiveError("BSPLINE_MARKER is required to generate initial B-spline modes")
    try:
        surface_mode = normalize_surface_mode(surface_mode)
    except BSplineModeError as exc:
        raise BSplineAdaptiveError(str(exc))

    n_internal = nper_side - degree - 1
    internal = [
        (i + 1) / float(n_internal + 1)
        for i in range(n_internal)
    ]
    knot_vector = [0.0] * (degree + 1) + internal + [1.0] * (degree + 1)
    modes = []
    for side in active_sides_from_surface_mode(surface_mode):
        for basis_index in range(nper_side):
            modes.append(
                {
                    "id": f"{side}_clamped_i{basis_index:03d}",
                    "side": side,
                    "basis_type": "clamped",
                    "degree": degree,
                    "knot_vector": knot_vector,
                    "basis_index": basis_index,
                    "coefficient": float(coefficient),
                    "bounds": [bound_lower, bound_upper],
                    "active": True,
                }
            )
    spec = {
        "version": 1,
        "dimension": 2,
        "marker": marker,
        "chord": {"mode": "auto"},
        "normal_displacement": True,
        "class_shape": str(class_shape),
        "class_shape_exponent": float(class_shape_exponent),
        "normalize_basis": _as_bool(normalize_basis, default=True),
        "normalization_mode": str(normalization_mode),
        "surface_mode": surface_mode,
        "modes": modes,
    }
    write_mode_spec(validate_mode_spec(spec), filename)
    return spec

def _forced_template_lines(case_config, forced):
    skip_prefixes = ("BSPLINE_", "PROGRESSIVE_")
    skip_keys = {
        "MATH_PROBLEM",
        "MESH_FILENAME",
        "MESH_OUT_FILENAME",
        "OBJECTIVE_FUNCTION",
        "TABULAR_FORMAT",
        "CONV_FILENAME",
        "SOLUTION_FILENAME",
        "RESTART_FILENAME",
        "SOLUTION_ADJ_FILENAME",
        "RESTART_ADJ_FILENAME",
        "SURFACE_ADJ_FILENAME",
        "VOLUME_ADJ_FILENAME",
        "HISTORY_OUTPUT",
        "SCREEN_OUTPUT",
        "DV_KIND",
        "DV_MARKER",
        "DV_FILENAME",
        "DEFINITION_DV",
        "OPT_OBJECTIVE",
        "OPT_ITERATIONS",
        "OPT_ACCURACY",
        "OPT_BOUND_LOWER",
        "OPT_BOUND_UPPER",
        "OPT_RELAX_FACTOR",
        "OPT_GRADIENT_FACTOR",
    }
    lines = []
    if case_config:
        with open(case_config, "r") as fp:
            for raw_line in fp:
                line = raw_line.rstrip("\n")
                stripped = line.strip()
                key = ""
                if stripped and not stripped.startswith(("%", "#")) and "=" in stripped:
                    key = stripped.split("=", 1)[0].strip().upper()
                if key and (key in skip_keys or key.startswith(skip_prefixes)):
                    continue
                lines.append(line)
    if lines and lines[-1].strip():
        lines.append("")
    for key, value in forced:
        lines.append(f"{key}= {_format_cfg_value(value)}")
    return lines

def _write_case_template(case_config, filename, forced):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    lines = _forced_template_lines(case_config, forced)
    with open(filename, "w") as fp:
        fp.write("\n".join(lines).rstrip())
        fp.write("\n")

def generate_missing_templates(settings):
    workdir = Path(settings["workdir"])
    template_dir = workdir / "templates"
    objective = str(settings.get("opt_objective", "DRAG")).upper()
    base_mesh = settings["base_mesh"]
    marker = settings["marker"]
    case_config = settings.get("case_config") or settings.get("optimizer_config")

    if not settings.get("def_template"):
        if not case_config:
            raise BSplineAdaptiveError(
                "missing BSPLINE_DEF_TEMPLATE and no case cfg is available to generate one"
            )

        path = template_dir / "def_template_auto.cfg"

        # SU2_DEF requires DV_MARKER to also exist in the BC marker lists.
        # Therefore the deformation template must preserve MARKER_* entries
        # from the case cfg, e.g. MARKER_EULER=(AIRFOIL), MARKER_FAR=(FARFIELD).
        #
        # SURFACE_FILE deformation also requires the standard DV_PARAM/DV_VALUE
        # entries. Without these, SU2_DEF may run but leave the mesh unchanged.
        _write_case_template(
            case_config,
            path,
            [
                ("MESH_FILENAME", base_mesh),
                ("MESH_OUT_FILENAME", "deformed_mesh.su2"),
                ("MESH_FORMAT", "SU2"),
                ("DV_KIND", "SURFACE_FILE"),
                ("DV_MARKER", [marker]),
                ("DV_PARAM", [1, 0.5]),
                ("DV_VALUE", 0.0),
                ("DV_FILENAME", "surface_positions.dat"),
                ("DEFORM_LINEAR_SOLVER", "FGMRES"),
                ("DEFORM_LINEAR_SOLVER_ERROR", 1e-14),
                ("DEFORM_LINEAR_SOLVER_ITER", 500),
            ],
        )

        settings["def_template"] = str(path.resolve())

    if not settings.get("primal_template"):
        if not case_config:
            raise BSplineAdaptiveError("missing BSPLINE_PRIMAL_TEMPLATE and no case cfg is available to generate one")
        path = template_dir / "primal_template_auto.cfg"
        _write_case_template(
            case_config,
            path,
            [
                ("MATH_PROBLEM", "DIRECT"),
                ("OBJECTIVE_FUNCTION", objective),
                ("MESH_FILENAME", "mesh.su2"),
                ("MESH_OUT_FILENAME", "primal_mesh_out.su2"),
                ("TABULAR_FORMAT", "CSV"),
                ("CONV_FILENAME", "history_primal"),
                ("SOLUTION_FILENAME", "solution_flow.dat"),
                ("RESTART_FILENAME", "restart_flow.dat"),
                ("HISTORY_OUTPUT", ["INNER_ITER", "RMS_RES", "AERO_COEFF"]),
                ("SCREEN_OUTPUT", ["INNER_ITER", "RMS_RES", "LIFT", "DRAG"]),
            ],
        )
        settings["primal_template"] = str(path.resolve())

    if not settings.get("adjoint_template"):
        if not case_config:
            raise BSplineAdaptiveError("missing BSPLINE_ADJOINT_TEMPLATE and no case cfg is available to generate one")
        path = template_dir / "adjoint_template_auto.cfg"
        _write_case_template(
            case_config,
            path,
            [
                ("MATH_PROBLEM", "DISCRETE_ADJOINT"),
                ("OBJECTIVE_FUNCTION", objective),
                ("MESH_FILENAME", "mesh.su2"),
                ("SOLUTION_FILENAME", "solution_flow.dat"),
                ("RESTART_FILENAME", "restart_flow.dat"),
                ("SOLUTION_ADJ_FILENAME", "solution_adj.dat"),
                ("RESTART_ADJ_FILENAME", "solution_adj.dat"),
                ("SURFACE_ADJ_FILENAME", "surface_adjoint"),
                ("VOLUME_ADJ_FILENAME", "volume_adjoint"),
                ("TABULAR_FORMAT", "CSV"),
                ("CONV_FILENAME", "history_adjoint"),
                ("HISTORY_OUTPUT", ["INNER_ITER", "RMS_RES", "AERO_COEFF"]),
                ("SCREEN_OUTPUT", ["INNER_ITER", "RMS_RES"]),
            ],
        )
        settings["adjoint_template"] = str(path.resolve())

def prepare_bspline_launch_settings(settings):
    settings = dict(settings)
    if settings.get("_prepared"):
        return settings

    if settings.get("candidate_bank"):
        print(
            "[PROGRESSIVE_BSPLINE] WARNING: --candidate-bank is ignored. "
            "Candidate/generated refinement has been removed."
        )
    settings["candidate_bank"] = None
    try:
        settings["surface_mode"] = normalize_surface_mode(
            settings.get("surface_mode", "BOTH")
        )
    except BSplineModeError as exc:
        raise BSplineAdaptiveError(str(exc))

    if settings.get("nproc") is not None and not settings.get("_mpi_cli_provided", False):
        settings["mpi"] = f"mpirun -n {int(settings['nproc'])}"
    settings.setdefault("mpi", "")

    for key in ("base_mesh", "marker", "workdir"):
        if not settings.get(key):
            raise BSplineAdaptiveError(f"missing required B-spline setting: {key}")

    workdir = Path(settings["workdir"]).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    settings["workdir"] = str(workdir)

    generated_initial = False
    if not settings.get("modes"):
        if _as_bool(settings.get("generate_initial_modes", False), default=False):
            modes_path = workdir / "generated" / "initial_modes.json"
            generate_initial_bspline_modes(
                modes_path,
                settings["marker"],
                nper_side=settings.get("initial_nper_side", 7),
                degree=settings.get("initial_degree", 3),
                coefficient=settings.get("initial_coefficient", 0.0),
                bound_lower=settings.get("initial_bound_lower", -0.01),
                bound_upper=settings.get("initial_bound_upper", 0.01),
                class_shape=settings.get(
                    "initial_class_shape",
                    "sqrt_x_one_minus_x",
                ),
                class_shape_exponent=settings.get(
                    "initial_class_shape_exponent",
                    0.5,
                ),
                normalize_basis=settings.get("initial_normalize_basis", True),
                normalization_mode=settings.get("initial_normalization_mode", "max"),
                surface_mode=settings["surface_mode"],
            )
            settings["modes"] = str(modes_path.resolve())
            generated_initial = True
        else:
            raise BSplineAdaptiveError(
                "missing --modes or BSPLINE_MODES or BSPLINE_GENERATE_INITIAL_MODES=YES"
            )
    settings["_initial_modes_generated"] = generated_initial

    generate_missing_templates(settings)
    for key in (
        "modes",
        "base_mesh",
        "def_template",
        "primal_template",
        "adjoint_template",
    ):
        if settings.get(key):
            settings[key] = str(Path(settings[key]).resolve())
    settings["_prepared"] = True
    return settings

def print_startup_summary(settings):
    case_config = settings.get("case_config") or settings.get("optimizer_config") or ""
    case_config = Path(case_config).name if case_config else ""
    print(f"[PROGRESSIVE_BSPLINE] Case config: {case_config}")
    print(f"[PROGRESSIVE_BSPLINE] modes: {settings.get('modes')}")
    print(
        "[PROGRESSIVE_BSPLINE] initial modes generated: "
        f"{'YES' if settings.get('_initial_modes_generated') else 'NO'}"
    )
    print(f"[PROGRESSIVE_BSPLINE] initial n_per_side: {settings.get('initial_nper_side', '')}")
    print(f"[PROGRESSIVE_BSPLINE] base mesh: {settings.get('base_mesh')}")
    print(f"[PROGRESSIVE_BSPLINE] marker: {settings.get('marker')}")
    print(f"[PROGRESSIVE_BSPLINE] def template: {settings.get('def_template')}")
    print(f"[PROGRESSIVE_BSPLINE] primal template: {settings.get('primal_template')}")
    print(f"[PROGRESSIVE_BSPLINE] adjoint template: {settings.get('adjoint_template')}")
    print(f"[PROGRESSIVE_BSPLINE] workdir: {settings.get('workdir')}")
    print(f"[PROGRESSIVE_BSPLINE] mpi: {settings.get('mpi', '')}")
    print("[PROGRESSIVE_BSPLINE] eval layout: DSN")
    print("[PROGRESSIVE_BSPLINE] sensitivity weighting: NODAL")
    print("[PROGRESSIVE_BSPLINE] refinement: KNOT_INSERTION")
    print(
        "[PROGRESSIVE_BSPLINE] KNOT_DEPTH penalty={} mode={} gamma={} initial_depth={}".format(
            "YES" if settings.get("knot_depth_penalty", False) else "NO",
            settings.get("knot_depth_penalty_mode", "NONE"),
            settings.get("knot_depth_power_gamma", 0.25),
            settings.get("knot_initial_span_depth", 1),
        )
    )
    print("[PROGRESSIVE_BSPLINE] refinement state: INITIAL_MESH_KEEP_DV")
    print(
        "[PROGRESSIVE_BSPLINE] deformation direction: "
        f"{settings.get('deformation_direction_mode', 'NORMAL')}"
    )
    surface_mode = settings.get("surface_mode", "BOTH")
    active_sides = active_sides_from_surface_mode(surface_mode)
    try:
        ndv = len(active_mode_ids(load_mode_spec(settings["modes"])))
    except Exception:
        ndv = ""
    print(f"[PROGRESSIVE_BSPLINE][SURFACE] mode = {surface_mode}")
    print(f"[PROGRESSIVE_BSPLINE][SURFACE] active sides = {active_sides}")
    print(f"[PROGRESSIVE_BSPLINE][SURFACE] ndv = {ndv}")
    print(
        "[PROGRESSIVE_BSPLINE][SURFACE] deformation direction = "
        f"{settings.get('deformation_direction_mode', 'NORMAL')}"
    )
    thickness_options = settings.get("thickness_options") or {}
    if _as_bool(
        thickness_options.get("PROGRESSIVE_THICKNESS_CONSTRAINT", False),
        default=False,
    ):
        print(
            "[PROGRESSIVE_BSPLINE][THICKNESS] domain = "
            f"{thickness_options.get('PROGRESSIVE_THICKNESS_DOMAIN_MODE')}"
        )
        print(
            "[PROGRESSIVE_BSPLINE][THICKNESS] symmetry_y = "
            f"{float(thickness_options.get('PROGRESSIVE_THICKNESS_SYMMETRY_Y', 0.0))}"
        )
    print(f"[PROGRESSIVE_BSPLINE] online trigger: {settings.get('trigger', 'MAX_ITER')}")
    print(
        "[PROGRESSIVE_BSPLINE] raw-gradient guard: {} factor={} window={} "
        "min_history={} floor={} restart_limit={}".format(
            "ON" if settings.get("gradient_guard", True) else "OFF",
            settings.get("gradient_guard_factor", 100.0),
            settings.get("gradient_guard_window", 5),
            settings.get("gradient_guard_min_history", 3),
            settings.get("gradient_guard_floor", 1.0e-14),
            settings.get("gradient_guard_restart_limit", 2),
        )
    )
    print(
        "[PROGRESSIVE_BSPLINE] trust-clip policy: {} legacy_beta_min={} "
        "severe_beta={} bad_patience={}/{} restart_limit={}".format(
            settings.get("trust_clip_policy", "OFF"),
            settings.get("trust_clip_legacy_beta_min", 0.50),
            settings.get("trust_clip_severe_beta", 0.50),
            settings.get("trust_clip_bad_patience", 2),
            settings.get("trust_clip_bad_window", 5),
            settings.get("trust_clip_restart_limit", 1),
        )
    )

def _settings_from_args(args):
    return {
        "case_config": getattr(args, "_case_config", getattr(args, "case_config", None)),
        "modes": args.modes,
        "candidate_bank": args.candidate_bank,
        "base_mesh": args.base_mesh,
        "marker": args.marker,
        "def_template": args.def_template,
        "primal_template": args.primal_template,
        "adjoint_template": args.adjoint_template,
        "workdir": args.workdir,
        "objective_column": args.objective_column,
        "optimizer_config": args.optimizer_config,
        "mpi": args.mpi,
        "nproc": getattr(args, "nproc", None),
        "_mpi_cli_provided": getattr(args, "_mpi_cli_provided", False),
        "nlevels": args.nlevels,
        "nfinal": args.nfinal,
        "max_iter_per_level": args.max_iter_per_level,
        "refinement": args.refinement,
        "refine_mode": args.refine_mode,
        "refine_state": args.refine_state,
        "knot_score_mode": args.knot_score_mode,
        "knot_insertions_per_refine": args.knot_insertions_per_refine,
        "knot_min_span_width": args.knot_min_span_width,
        "knot_depth_penalty": getattr(args, "knot_depth_penalty", False),
        "knot_depth_penalty_mode": getattr(args, "knot_depth_penalty_mode", "NONE"),
        "knot_depth_power_gamma": getattr(args, "knot_depth_power_gamma", 0.25),
        "knot_initial_span_depth": getattr(args, "knot_initial_span_depth", 1),
        "transfer_method": args.transfer_method,
        "transfer_bound_policy": args.transfer_bound_policy,
        "transfer_geometry_abs_tol": args.transfer_geometry_abs_tol,
        "transfer_geometry_rel_tol": args.transfer_geometry_rel_tol,
        "nadd_mode": args.nadd_mode,
        "batch_size_max": args.batch_size_max,
        "growth_ratio": args.growth_ratio,
        "fixed_nadd": args.fixed_nadd,
        "sensitivity_weighting": getattr(args, "sensitivity_weighting", "NODAL"),
        "eval_layout": args.eval_layout,
        "objective_adjoint": args.objective_adjoint,
        "symmetry_coupling": args.symmetry_coupling,
        "surface_mode": getattr(args, "surface_mode", "BOTH"),
        "deformation_direction_mode": getattr(
            args,
            "deformation_direction_mode",
            None,
        ),
        "trigger": args.trigger,
        "window": args.window,
        "tol": args.tol,
        "eps": args.eps,
        "slope_filter_tol": args.slope_filter_tol,
        "slope_patience": args.slope_patience,
        "stag_window": args.stag_window,
        "stag_tol": args.stag_tol,
        "stag_band": args.stag_band,
        "stag_patience": args.stag_patience,
        "warmup_iter": args.warmup_iter,
        "generate_initial_modes": getattr(args, "generate_initial_modes", None),
        "initial_nper_side": getattr(args, "initial_nper_side", 7),
        "initial_degree": getattr(args, "initial_degree", 3),
        "initial_coefficient": getattr(args, "initial_coefficient", 0.0),
        "initial_bound_lower": getattr(args, "initial_bound_lower", -0.01),
        "initial_bound_upper": getattr(args, "initial_bound_upper", 0.01),
        "initial_class_shape": getattr(
            args,
            "initial_class_shape",
            "sqrt_x_one_minus_x",
        ),
        "initial_class_shape_exponent": getattr(
            args,
            "initial_class_shape_exponent",
            0.5,
        ),
        "initial_normalize_basis": getattr(args, "initial_normalize_basis", True),
        "initial_normalization_mode": getattr(args, "initial_normalization_mode", "max"),
        "opt_objective": getattr(args, "opt_objective", None),
        "dry_run": args.dry_run,
        "show_commands": args.show_commands,
        "stream_solver_output": args.stream_solver_output,
        "print_optimizer_table": args.print_optimizer_table,
        "log_active_modes": args.log_active_modes,
        "opt_accuracy": getattr(args, "opt_accuracy", None),
        "opt_bound_upper": args.opt_bound_upper,
        "opt_bound_lower": args.opt_bound_lower,
        "opt_relax_factor": args.opt_relax_factor,
        "opt_gradient_factor": args.opt_gradient_factor,
        "gradient_guard": getattr(args, "gradient_guard", True),
        "gradient_guard_factor": getattr(args, "gradient_guard_factor", 100.0),
        "gradient_guard_window": getattr(args, "gradient_guard_window", 5),
        "gradient_guard_min_history": getattr(args, "gradient_guard_min_history", 3),
        "gradient_guard_floor": getattr(args, "gradient_guard_floor", 1.0e-14),
        "gradient_guard_restart_limit": getattr(
            args,
            "gradient_guard_restart_limit",
            2,
        ),
        "trust_clip_policy": getattr(args, "trust_clip_policy", "OFF"),
        "trust_clip_beta_tol": getattr(args, "trust_clip_beta_tol", 1.0e-12),
        "trust_clip_legacy_beta_min": getattr(
            args, "trust_clip_legacy_beta_min", 0.50
        ),
        "trust_clip_severe_beta": getattr(args, "trust_clip_severe_beta", 0.50),
        "trust_clip_worsening_tol": getattr(
            args, "trust_clip_worsening_tol", 0.05
        ),
        "trust_clip_soft_gnorm_factor": getattr(
            args, "trust_clip_soft_gnorm_factor", 20.0
        ),
        "trust_clip_bad_patience": getattr(args, "trust_clip_bad_patience", 2),
        "trust_clip_bad_window": getattr(args, "trust_clip_bad_window", 5),
        "trust_clip_stag_tol": getattr(args, "trust_clip_stag_tol", 1.0e-6),
        "trust_clip_restart_limit": getattr(args, "trust_clip_restart_limit", 1),
        "opt_line_search_bound": args.opt_line_search_bound,
        "local_step_limit": args.local_step_limit,
        "local_step_limit_ratio": args.local_step_limit_ratio,
        "thickness_options": getattr(args, "thickness_options", None),
        "auto_scale_bounds_to_geometry": args.auto_scale_bounds_to_geometry,
        "max_normal_displacement": args.max_normal_displacement,
        "max_rms_normal_displacement": args.max_rms_normal_displacement,
        "min_bound_scale": args.min_bound_scale,
        "le_safe_direction": getattr(args, "le_safe_direction", False),
        "le_safe_x0": getattr(args, "le_safe_x0", LE_SAFE_DEFAULT_X0),
        "le_safe_x1": getattr(args, "le_safe_x1", LE_SAFE_DEFAULT_X1),
        "le_safe_power": getattr(args, "le_safe_power", LE_SAFE_DEFAULT_POWER),
    }

# Imported after function definitions to keep settings as a configuration layer
# without creating import cycles through mode_utils/knot_space/penalties.
from .knot_space import REFINE_MODE, REFINE_STATE
from .penalties import (
    _as_knot_depth,
    _normalize_knot_depth_penalty_mode,
)
