"""Helpers that apply optimizer config values to driver options/CLI args."""

import sys
from pathlib import Path

from SU2.opt.bspline_modes import (
    BSplineModeError,
    normalize_surface_mode,
)
from SU2.opt.thickness_constraint import (
    THICKNESS_PROGRESSIVE_KEYS,
    _resolve_from_cfg_dir,
)
from .config_parse import (
    _ConfigDict,
    parse_optimizer_config,
)
from .errors import BSplineSU2DriverError
from .native_constraints import normalize_native_constraints
from .tables import history_column_for_function

def _objective_column_from_config(config_values):
    if "OBJECTIVE_COLUMN" in config_values:
        return str(config_values["OBJECTIVE_COLUMN"])
    objective = str(config_values.get("OPT_OBJECTIVE", "")).strip().upper()
    if objective:
        return history_column_for_function(objective)
    return None

def _objective_adjoint_from_config(config_values):
    objective = str(config_values.get("OPT_OBJECTIVE", "")).strip().upper()
    if objective == "DRAG":
        return "drag"
    if objective:
        return _normalized_name(objective).lower()
    return None

def fixed_driver_options_from_config(config_values):
    options = {}
    objective_column = _objective_column_from_config(config_values)
    if objective_column is not None:
        options["objective_column"] = objective_column
    objective_adjoint = _objective_adjoint_from_config(config_values)
    if objective_adjoint is not None:
        options["objective_adjoint"] = objective_adjoint
    if "OPT_CONSTRAINT" in config_values:
        options["native_constraints"] = normalize_native_constraints(
            config_values["OPT_CONSTRAINT"]
        )
    mapping = {
        "OPT_ITERATIONS": "maxiter",
        "OPT_ACCURACY": "opt_accuracy",
        "OPT_BOUND_UPPER": "opt_bound_upper",
        "OPT_BOUND_LOWER": "opt_bound_lower",
        "OPT_RELAX_FACTOR": "opt_relax_factor",
        "OPT_GRADIENT_FACTOR": "opt_gradient_factor",
        "OPT_LINE_SEARCH_BOUND": "opt_line_search_bound",
        "BSPLINE_LOCAL_STEP_LIMIT": "local_step_limit",
        "BSPLINE_LOCAL_STEP_LIMIT_RATIO": "local_step_limit_ratio",
        "BSPLINE_AUTO_SCALE_BOUNDS_TO_GEOMETRY": "auto_scale_bounds_to_geometry",
        "BSPLINE_MAX_NORMAL_DISPLACEMENT": "max_normal_displacement",
        "BSPLINE_MAX_RMS_NORMAL_DISPLACEMENT": "max_rms_normal_displacement",
        "BSPLINE_MIN_BOUND_SCALE": "min_bound_scale",
        "BSPLINE_SHOW_COMMANDS": "show_commands",
        "BSPLINE_STREAM_SOLVER_OUTPUT": "stream_solver_output",
        "BSPLINE_PRINT_OPTIMIZER_TABLE": "print_optimizer_table",
        "BSPLINE_EVAL_LAYOUT": "eval_layout",
        "BSPLINE_SENSITIVITY_SOURCE": "sensitivity_source",
        "BSPLINE_GEOMETRY_FD_EPS": "geometry_fd_eps",
        "BSPLINE_GEOMETRY_CONSTRAINT_GRADIENT": "geometry_constraint_gradient",
        "BSPLINE_SYMMETRY_COUPLING": "symmetry_coupling",
        "BSPLINE_SURFACE_MODE": "surface_mode",
        "BSPLINE_DEFORMATION_DIRECTION": "deformation_direction_mode",
        "BSPLINE_GRADIENT_GUARD": "gradient_guard",
        "BSPLINE_GRADIENT_GUARD_FACTOR": "gradient_guard_factor",
        "BSPLINE_GRADIENT_GUARD_WINDOW": "gradient_guard_window",
        "BSPLINE_GRADIENT_GUARD_MIN_HISTORY": "gradient_guard_min_history",
        "BSPLINE_GRADIENT_GUARD_FLOOR": "gradient_guard_floor",
        "BSPLINE_GRADIENT_GUARD_RESTART_LIMIT": "gradient_guard_restart_limit",
        "BSPLINE_TRUST_CLIP_POLICY": "trust_clip_policy",
        "BSPLINE_TRUST_CLIP_BETA_TOL": "trust_clip_beta_tol",
        "BSPLINE_TRUST_CLIP_LEGACY_BETA_MIN": "trust_clip_legacy_beta_min",
        "BSPLINE_TRUST_CLIP_SEVERE_BETA": "trust_clip_severe_beta",
        "BSPLINE_TRUST_CLIP_WORSENING_TOL": "trust_clip_worsening_tol",
        "BSPLINE_TRUST_CLIP_SOFT_GNORM_FACTOR": "trust_clip_soft_gnorm_factor",
        "BSPLINE_TRUST_CLIP_BAD_PATIENCE": "trust_clip_bad_patience",
        "BSPLINE_TRUST_CLIP_BAD_WINDOW": "trust_clip_bad_window",
        "BSPLINE_TRUST_CLIP_STAG_TOL": "trust_clip_stag_tol",
        "BSPLINE_TRUST_CLIP_RESTART_LIMIT": "trust_clip_restart_limit",
        "BSPLINE_LE_SAFE_DIRECTION": "le_safe_direction",
        "BSPLINE_LE_SAFE_X0": "le_safe_x0",
        "BSPLINE_LE_SAFE_X1": "le_safe_x1",
        "BSPLINE_LE_SAFE_POWER": "le_safe_power",
        "BSPLINE_KNOT_DEPTH_PENALTY": "knot_depth_penalty",
        "BSPLINE_KNOT_DEPTH_PENALTY_MODE": "knot_depth_penalty_mode",
        "BSPLINE_KNOT_DEPTH_POWER_GAMMA": "knot_depth_power_gamma",
        "BSPLINE_KNOT_INITIAL_SPAN_DEPTH": "knot_initial_span_depth",
        "BSPLINE_KNOT_BATCH_DIVERSITY": "knot_batch_diversity",
        "BSPLINE_KNOT_BATCH_PENALTY_MODE": "knot_batch_penalty_mode",
        "BSPLINE_KNOT_BATCH_POWER_GAMMA": "knot_batch_power_gamma",
    }
    for key, dest in mapping.items():
        if key in config_values:
            options[dest] = config_values[key]
    thickness_options = thickness_options_from_config(config_values)
    if thickness_options:
        options["thickness_options"] = thickness_options
    return options

def thickness_options_from_config(config_values):
    options = {}
    for key in THICKNESS_PROGRESSIVE_KEYS:
        if key in config_values:
            options[key] = config_values[key]

    if not options:
        return {}

    config_filename = config_values.get("_optimizer_config_filename")
    if config_filename:
        cfg = _ConfigDict(options)
        cfg._filename = config_filename
        for key in (
            "PROGRESSIVE_THICKNESS_REF_MESH",
            "PROGRESSIVE_THICKNESS_CACHE_FILE",
        ):
            if key in options and options[key]:
                options[key] = _resolve_from_cfg_dir(cfg, options[key])
    return options

def resolve_thickness_domain_mode(surface_mode, value="AUTO"):
    """Resolve AUTO and enforce a physically compatible thickness domain."""

    try:
        surface_mode = normalize_surface_mode(surface_mode)
    except BSplineModeError as exc:
        raise BSplineSU2DriverError(str(exc))
    domain_mode = str(value or "AUTO").strip().upper()
    allowed = ("AUTO", "FULL", "HALF_UPPER", "HALF_LOWER")
    if domain_mode not in allowed:
        raise BSplineSU2DriverError(
            "PROGRESSIVE_THICKNESS_DOMAIN_MODE must be AUTO, FULL, "
            f"HALF_UPPER, or HALF_LOWER; got {domain_mode!r}"
        )

    natural = {
        "BOTH": "FULL",
        "UPPER": "HALF_UPPER",
        "LOWER": "HALF_LOWER",
    }[surface_mode]
    if domain_mode == "AUTO":
        return natural
    if domain_mode == natural:
        return domain_mode
    if domain_mode == "FULL":
        raise BSplineSU2DriverError(
            "FULL thickness requires a complete upper/lower surface; "
            f"use {natural} with BSPLINE_SURFACE_MODE={surface_mode}."
        )
    raise BSplineSU2DriverError(
        f"{domain_mode} thickness is incompatible with "
        f"BSPLINE_SURFACE_MODE={surface_mode}; use {natural}."
    )

def _explicit_cli_dests(parser, argv):
    argv = list(sys.argv[1:] if argv is None else argv)
    option_to_dest = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_to_dest[option] = action.dest

    explicit = set()
    for item in argv:
        if not str(item).startswith("--"):
            continue
        option = str(item).split("=", 1)[0]
        dest = option_to_dest.get(option)
        if dest:
            explicit.add(dest)
    return explicit

def apply_optimizer_config_to_args(
    args,
    parser,
    argv,
    config_to_options,
    warning_prefix="[BSPLINE_SU2_DRIVER]",
):
    if not getattr(args, "optimizer_config", None):
        return args
    config_values = parse_optimizer_config(args.optimizer_config, warning_prefix=warning_prefix)
    config_values["_optimizer_config_filename"] = str(Path(args.optimizer_config).resolve())
    explicit = _explicit_cli_dests(parser, argv)
    config_options = config_to_options(config_values)
    for dest, value in config_options.items():
        if dest not in explicit:
            setattr(args, dest, value)
    setattr(args, "_optimizer_config_values", config_values)
    return args
