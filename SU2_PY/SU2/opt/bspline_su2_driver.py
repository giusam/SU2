#!/usr/bin/env python

"""External fixed-mode B-spline optimization driver for SU2."""

import argparse
import copy
import csv
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace
from pathlib import Path

import numpy as np

from SU2.opt.bspline_def import extract_marker_nodes, read_su2_mesh
from SU2.opt.bspline_dot import (
    BSplineDotError,
    normalize_sensitivity_weighting,
    read_metadata,
)
from SU2.opt.bspline_modes import (
    ALLOWED_DEFORMATION_DIRECTION_MODES,
    ALLOWED_SURFACE_MODES,
    BSplineModeError,
    LE_SAFE_DEFAULT_POWER,
    LE_SAFE_DEFAULT_X0,
    LE_SAFE_DEFAULT_X1,
    active_sides_from_surface_mode,
    evaluate_all_modes,
    load_mode_spec,
    normalize_deformation_direction_mode,
    normalize_surface_mode,
    validate_le_safe_direction_options,
    validate_mode_spec,
    validate_surface_mode_against_modes,
)
from SU2.opt.progressive_trigger import (
    RefinementTriggered,
    record_objective_and_check,
    trigger_prefix,
)
from SU2.opt.thickness_constraint import (
    THICKNESS_PROGRESSIVE_KEYS,
    _as_bool as _thickness_as_bool,
    _load_or_build_reference,
    _normalize_gradient_mode,
    _parse_x_stations,
    _resolve_from_cfg_dir,
    _x_stations_value_is_empty,
)


class BSplineSU2DriverError(RuntimeError):
    pass


class GradientGuardStop(RuntimeError):
    def __init__(self, last_safe_entry, bad_entry, guard_info=None):
        self.last_safe_entry = last_safe_entry
        self.bad_entry = bad_entry
        self.guard_info = guard_info or {}
        super().__init__("Raw-gradient guard stop: rollback to best safe evaluation")


class TrustClipStop(RuntimeError):
    def __init__(self, classification, entry, rollback_entry, diagnostics=None, action=None):
        self.classification = str(classification)
        self.entry = entry
        self.rollback_entry = rollback_entry
        self.diagnostics = diagnostics or {}
        self.action = str(action or "restart_same_level")
        super().__init__(
            f"Trust-clip stop: {self.classification}; action={self.action}"
        )


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


DEFAULT_BOUNDS = (-0.01, 0.01)
ALLOWED_EVAL_LAYOUTS = ("DSN",)
ALLOWED_SYMMETRY_COUPLINGS = ("NONE", "NORMAL_EQUAL", "NORMAL_OPPOSITE")
UNSUPPORTED_OPT_CONFIG_KEYS = (
    "OPT_CONSTRAINT",
)
SUPPORTED_OPT_CONFIG_KEYS = {
    "OPT_OBJECTIVE",
    "OBJECTIVE_COLUMN",
    "OPT_ITERATIONS",
    "OPT_ACCURACY",
    "OPT_BOUND_UPPER",
    "OPT_BOUND_LOWER",
    "OPT_RELAX_FACTOR",
    "OPT_GRADIENT_FACTOR",
    "OPT_LINE_SEARCH_BOUND",
    "BSPLINE_LOCAL_STEP_LIMIT",
    "BSPLINE_LOCAL_STEP_LIMIT_RATIO",
    "BSPLINE_EVAL_LAYOUT",
    "BSPLINE_SYMMETRY_COUPLING",
    "BSPLINE_SURFACE_MODE",
    "BSPLINE_DEFORMATION_DIRECTION",
    "BSPLINE_GRADIENT_GUARD",
    "BSPLINE_GRADIENT_GUARD_FACTOR",
    "BSPLINE_GRADIENT_GUARD_WINDOW",
    "BSPLINE_GRADIENT_GUARD_MIN_HISTORY",
    "BSPLINE_GRADIENT_GUARD_FLOOR",
    "BSPLINE_GRADIENT_GUARD_RESTART_LIMIT",
    "BSPLINE_TRUST_CLIP_POLICY",
    "BSPLINE_TRUST_CLIP_BETA_TOL",
    "BSPLINE_TRUST_CLIP_LEGACY_BETA_MIN",
    "BSPLINE_TRUST_CLIP_SEVERE_BETA",
    "BSPLINE_TRUST_CLIP_WORSENING_TOL",
    "BSPLINE_TRUST_CLIP_SOFT_GNORM_FACTOR",
    "BSPLINE_TRUST_CLIP_BAD_PATIENCE",
    "BSPLINE_TRUST_CLIP_BAD_WINDOW",
    "BSPLINE_TRUST_CLIP_STAG_TOL",
    "BSPLINE_TRUST_CLIP_RESTART_LIMIT",
    *THICKNESS_PROGRESSIVE_KEYS,
    "BSPLINE_NLEVELS",
    "BSPLINE_NFINAL",
    "BSPLINE_KNOT_SCORE_MODE",
    "BSPLINE_KNOT_INSERTIONS_PER_REFINE",
    "BSPLINE_KNOT_MIN_SPAN_WIDTH",
    "BSPLINE_KNOT_DEPTH_PENALTY",
    "BSPLINE_KNOT_DEPTH_PENALTY_MODE",
    "BSPLINE_KNOT_DEPTH_POWER_GAMMA",
    "BSPLINE_KNOT_INITIAL_SPAN_DEPTH",
    "BSPLINE_KNOT_BATCH_DIVERSITY",
    "BSPLINE_KNOT_BATCH_PENALTY_MODE",
    "BSPLINE_KNOT_BATCH_POWER_GAMMA",
    "BSPLINE_TRANSFER_METHOD",
    "BSPLINE_TRANSFER_BOUND_POLICY",
    "BSPLINE_TRANSFER_GEOMETRY_ABS_TOL",
    "BSPLINE_TRANSFER_GEOMETRY_REL_TOL",
    "BSPLINE_USE_CLASS_SHAPE",
    "BSPLINE_INITIAL_CLASS_SHAPE",
    "BSPLINE_INITIAL_CLASS_SHAPE_EXPONENT",
    "BSPLINE_TRIGGER",
    "BSPLINE_TRIGGER_WINDOW",
    "BSPLINE_TRIGGER_RATIO",
    "BSPLINE_TRIGGER_EPS",
    "BSPLINE_SLOPE_FILTER_TOL",
    "BSPLINE_SLOPE_PATIENCE",
    "BSPLINE_STAGNATION_WINDOW",
    "BSPLINE_STAGNATION_REL_TOL",
    "BSPLINE_STAGNATION_PATIENCE",
    "BSPLINE_NADD_MODE",
    "BSPLINE_FIXED_NADD",
    "BSPLINE_GROWTH_RATIO",
    "BSPLINE_BATCH_SIZE_MAX",
    "BSPLINE_AUTO_SCALE_BOUNDS_TO_GEOMETRY",
    "BSPLINE_MAX_NORMAL_DISPLACEMENT",
    "BSPLINE_MAX_RMS_NORMAL_DISPLACEMENT",
    "BSPLINE_MIN_BOUND_SCALE",
    "BSPLINE_SHOW_COMMANDS",
    "BSPLINE_STREAM_SOLVER_OUTPUT",
    "BSPLINE_PRINT_OPTIMIZER_TABLE",
    "BSPLINE_LOG_ACTIVE_MODES",
    *UNSUPPORTED_OPT_CONFIG_KEYS,
}


@dataclass(frozen=True)
class EvalPaths:
    eval_layout: str
    objective_adjoint: str
    eval_dir: Path
    deform_dir: Path
    direct_dir: Path
    adjoint_dir: Path
    modes_current: Path
    surface_positions: Path
    metadata: Path
    def_cfg: Path
    primal_cfg: Path
    adjoint_cfg: Path
    deformed_mesh: Path
    primal_mesh_out: Path
    adjoint_mesh_out: Path
    primal_history: Path
    adjoint_history: Path
    primal_solution: Path
    primal_restart: Path
    adjoint_solution: Path
    adjoint_restart: Path
    volume_adjoint: Path
    surface_adjoint: Path
    gradients: Path
    summary: Path
    commands_log: Path
    bspline_def_log: Path
    su2_def_log: Path
    su2_cfd_log: Path
    su2_cfd_ad_log: Path
    bspline_dot_log: Path


class _ConfigDict(dict):
    pass


def _parse_optimizer_config_value(value):
    value = str(value).strip().strip('"').strip("'")
    upper = value.upper()
    if upper == "YES":
        return True
    if upper == "NO":
        return False
    try:
        if any(char in value for char in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_optimizer_config(filename, warning_prefix="[BSPLINE_SU2_DRIVER]"):
    values = {}
    with open(filename, "r") as fp:
        for line_number, raw_line in enumerate(fp, start=1):
            line = raw_line.strip()
            if not line or line.startswith("%") or line.startswith("#"):
                continue
            if "=" not in line:
                raise BSplineSU2DriverError(
                    f"{filename}:{line_number} expected KEY= VALUE"
                )
            key, value = line.split("=", 1)
            key = key.strip().upper()
            if not key:
                raise BSplineSU2DriverError(f"{filename}:{line_number} has an empty key")
            parsed_value = _parse_optimizer_config_value(value)
            values[key] = parsed_value

    values["_optimizer_config_filename"] = str(Path(filename).resolve())
    for key in UNSUPPORTED_OPT_CONFIG_KEYS:
        if key in values:
            print(
                f"{warning_prefix} WARNING: {key} is parsed but not implemented yet; ignoring."
            )
    return values


def _objective_column_from_config(config_values):
    if "OBJECTIVE_COLUMN" in config_values:
        return str(config_values["OBJECTIVE_COLUMN"])
    objective = str(config_values.get("OPT_OBJECTIVE", "")).strip().upper()
    if objective == "DRAG":
        return "CD"
    if objective:
        return objective
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


def _as_float(value, name):
    try:
        result = float(value)
    except Exception:
        raise BSplineSU2DriverError(f"{name} must be numeric")
    if not math.isfinite(result):
        raise BSplineSU2DriverError(f"{name} must be finite")
    return result


def _normalized_name(value):
    return "".join(
        char.lower()
        for char in str(value).strip().strip('"').strip("'")
        if char.isalnum()
    )


def _active_modes(mode_spec):
    validate_mode_spec(mode_spec)
    return [
        mode
        for mode in mode_spec.get("modes", [])
        if mode.get("active", True) is not False
    ]


def active_mode_ids(mode_spec):
    return [str(mode["id"]) for mode in _active_modes(mode_spec)]


def active_coefficient_vector(mode_spec):
    return [
        _as_float(mode.get("coefficient", 0.0), f"mode {mode['id']} coefficient")
        for mode in _active_modes(mode_spec)
    ]


def active_bounds(mode_spec, default_bounds=DEFAULT_BOUNDS):
    lower_default, upper_default = _validated_bounds(default_bounds, "default_bounds")
    bounds = []
    for mode in _active_modes(mode_spec):
        mode_bounds = mode.get("bounds")
        if mode_bounds is None:
            bounds.append((lower_default, upper_default))
        else:
            bounds.append(_validated_bounds(mode_bounds, f"mode {mode['id']} bounds"))
    return bounds


@dataclass(frozen=True)
class ReducedVariable:
    id: str
    mode_indices: tuple
    mode_ids: tuple
    signs: tuple


def _safe_identifier(value):
    text = str(value)
    chars = []
    for char in text:
        chars.append(char if char.isalnum() else "_")
    return "_".join(part for part in "".join(chars).split("_") if part)


def _mode_support_key(mode):
    basis_type = str(mode.get("basis_type", "")).strip().lower()
    if basis_type != "clamped":
        # Knot-insertion only. Reject any non-clamped basis at the support-key
        # boundary; callers should never observe legacy basis types here because
        # validate_mode_spec rejects them upstream.
        raise BSplineSU2DriverError(
            "Only basis_type='clamped' is supported by the knot-insertion B-spline workflow."
        )
    degree = int(mode.get("degree", 3))
    knots = mode.get("knot_vector", mode.get("knots"))
    if knots is None or "basis_index" not in mode:
        left, right = 0.0, 1.0
    else:
        knots = [float(value) for value in knots]
        index = int(mode["basis_index"])
        right_index = min(len(knots) - 1, index + degree + 1)
        left, right = knots[index], knots[right_index]
    return basis_type, degree, round(float(left), 12), round(float(right), 12)


def _mode_pairing_key(mode):
    basis_type = str(mode.get("basis_type", "")).strip().lower()
    if basis_type != "clamped":
        raise BSplineSU2DriverError(
            "Only basis_type='clamped' is supported by the knot-insertion B-spline workflow."
        )
    degree = int(mode.get("degree", 3))
    knots = mode.get("knot_vector", mode.get("knots"))
    if knots is None or "basis_index" not in mode:
        return basis_type, degree, (), None
    knots = tuple(round(float(value), 12) for value in knots)
    return basis_type, degree, knots, int(mode["basis_index"])


def _symmetry_group_id(key):
    basis_type, degree, third, fourth = key
    if basis_type == "clamped":
        return _safe_identifier(
            "group_{}_d{}_i{}".format(
                basis_type,
                degree,
                "missing" if fourth is None else int(fourth),
            )
        )
    left, right = third, fourth
    return _safe_identifier(
        "group_{}_d{}_s{:.12g}_{:.12g}".format(
            basis_type,
            degree,
            float(left),
            float(right),
        )
    )


def build_reduced_variables(mode_spec, coupling="NONE"):
    coupling = str(coupling or "NONE").strip().upper()
    if coupling not in ALLOWED_SYMMETRY_COUPLINGS:
        raise BSplineSU2DriverError(
            f"BSPLINE_SYMMETRY_COUPLING must be one of {ALLOWED_SYMMETRY_COUPLINGS}; got {coupling!r}"
        )

    active_modes = _active_modes(mode_spec)
    if coupling == "NONE":
        return [
            ReducedVariable(
                id=_safe_identifier(mode["id"]),
                mode_indices=(index,),
                mode_ids=(str(mode["id"]),),
                signs=(1.0,),
            )
            for index, mode in enumerate(active_modes)
        ], []

    grouped = {}
    for index, mode in enumerate(active_modes):
        grouped.setdefault(_mode_pairing_key(mode), []).append((index, mode))

    reduced = []
    warnings = []
    for key in sorted(grouped, key=lambda item: (item[0], item[1], item[2], item[3])):
        entries = grouped[key]
        by_side = {}
        for index, mode in entries:
            by_side.setdefault(str(mode.get("side", "")).strip().lower(), []).append((index, mode))

        if len(entries) == 2 and len(by_side.get("upper", [])) == 1 and len(by_side.get("lower", [])) == 1:
            upper_index, upper_mode = by_side["upper"][0]
            lower_index, lower_mode = by_side["lower"][0]
            lower_sign = 1.0 if coupling == "NORMAL_EQUAL" else -1.0
            reduced.append(
                ReducedVariable(
                    id=_symmetry_group_id(key),
                    mode_indices=(upper_index, lower_index),
                    mode_ids=(str(upper_mode["id"]), str(lower_mode["id"])),
                    signs=(1.0, lower_sign),
                )
            )
            continue

        mode_ids = ", ".join(str(mode["id"]) for _index, mode in entries)
        warnings.append(
            "{} symmetry coupling left mode(s) unpaired for support {}: {}".format(
                coupling,
                key,
                mode_ids,
            )
        )
        for index, mode in entries:
            reduced.append(
                ReducedVariable(
                    id=_safe_identifier(mode["id"]),
                    mode_indices=(index,),
                    mode_ids=(str(mode["id"]),),
                    signs=(1.0,),
                )
            )

    reduced.sort(key=lambda variable: min(variable.mode_indices))
    return reduced, warnings


def expand_reduced_coefficients(reduced_coefficients, reduced_variables, n_modes):
    values = [_as_float(value, "reduced coefficient") for value in reduced_coefficients]
    if len(values) != len(reduced_variables):
        raise BSplineSU2DriverError(
            f"expected {len(reduced_variables)} reduced coefficient(s), got {len(values)}"
        )
    full = [0.0] * int(n_modes)
    for value, variable in zip(values, reduced_variables):
        for index, sign in zip(variable.mode_indices, variable.signs):
            full[int(index)] = float(sign) * float(value)
    return full


def compress_full_coefficients(
    coefficients,
    reduced_variables,
    tolerance=1.0e-8,
    warn=None,
    coupling="NONE",
):
    """Collapse full coefficients to the reduced space.

    With ``NORMAL_OPPOSITE`` the reduced variable ``r`` represents the
    antisymmetric component: ``a_upper = +r``, ``a_lower = -r``. A
    unilateral bump (e.g. ``a_upper = 0, a_lower = +0.01``) cannot be
    represented in that subspace; the silent transformation
    ``r = (0 - 0.01)/2`` would map it to ``a_upper = -0.005`` and
    ``a_lower = +0.005`` and the original ``warn`` mismatch test
    (which only checks the round-trip through the antisymmetric
    formula) would never fire. We therefore detect this case explicitly
    when ``coupling == 'NORMAL_OPPOSITE'`` and a 2-element pair is
    supplied, and raise so the user can either:
      * set the modes to zero and let the optimizer introduce them
        via the adaptive driver, or
      * switch to ``NORMAL_EQUAL`` if a pure mirror is acceptable.
    """
    coupling = str(coupling or "NONE").strip().upper()
    values = [_as_float(value, "coefficient") for value in coefficients]
    reduced = []
    for variable in reduced_variables:
        paired_values = [values[int(index)] for index in variable.mode_indices]
        signed_values = [
            float(sign) * float(value)
            for sign, value in zip(variable.signs, paired_values)
        ]
        reduced_value = sum(signed_values) / float(len(signed_values))
        mismatch = max(
            abs(float(value) - float(sign) * reduced_value)
            for sign, value in zip(variable.signs, paired_values)
        )
        # For NORMAL_OPPOSITE a 2-element pair is representable iff it is
        # antisymmetric, i.e. a_upper + a_lower == 0. ANY other input
        # (a unilateral bump like [0, 0.01], or a same-sign pair like
        # [0.01, 0.01]) is silently mapped to an inverted/mirrored
        # deformation by the reduced-variable round-trip, so we refuse it
        # outright instead of merely warning. This is checked BEFORE the
        # generic mismatch warning below so the error message is specific.
        if (
            coupling == "NORMAL_OPPOSITE"
            and len(variable.mode_indices) == 2
            and len(variable.signs) == 2
            and list(variable.signs) == [1.0, -1.0]
        ):
            a_upper, a_lower = paired_values[0], paired_values[1]
            if abs(float(a_upper) + float(a_lower)) > float(tolerance):
                raise BSplineSU2DriverError(
                    "NORMAL_OPPOSITE coupling requires antisymmetric initial "
                    "coefficients (a_upper + a_lower == 0) for modes {} but got "
                    "a_upper={:.6e}, a_lower={:.6e}; set both to zero, make them "
                    "antisymmetric, or use NORMAL_EQUAL.".format(
                        ", ".join(variable.mode_ids),
                        float(a_upper),
                        float(a_lower),
                    )
                )
        if len(variable.mode_indices) > 1 and mismatch > float(tolerance) and warn is not None:
            warn(
                "initial coefficients for coupled modes {} are inconsistent; using reduced value {:.15g}".format(
                    ", ".join(variable.mode_ids),
                    reduced_value,
                )
            )
        reduced.append(float(reduced_value))
    return reduced


def reduced_bounds_from_full_bounds(bounds, reduced_variables):
    reduced_bounds = []
    for variable in reduced_variables:
        lower = -math.inf
        upper = math.inf
        for index, sign in zip(variable.mode_indices, variable.signs):
            mode_lower, mode_upper = _validated_bounds(
                bounds[int(index)],
                f"bounds for reduced variable {variable.id}",
            )
            if float(sign) >= 0.0:
                candidate_lower, candidate_upper = mode_lower, mode_upper
            else:
                candidate_lower, candidate_upper = -mode_upper, -mode_lower
            lower = max(lower, candidate_lower)
            upper = min(upper, candidate_upper)
        if upper < lower:
            raise BSplineSU2DriverError(
                "empty bound intersection for coupled reduced variable {} ({})".format(
                    variable.id,
                    ", ".join(variable.mode_ids),
                )
            )
        reduced_bounds.append((float(lower), float(upper)))
    return reduced_bounds


def collapse_full_gradient(gradient, reduced_variables):
    values = [_as_float(value, "gradient") for value in gradient]
    collapsed = []
    for variable in reduced_variables:
        collapsed.append(
            sum(
                float(sign) * values[int(index)]
                for index, sign in zip(variable.mode_indices, variable.signs)
            )
        )
    return collapsed


def collapse_full_jacobian(jacobian, reduced_variables):
    jacobian = np.asarray(jacobian, dtype=float)
    if jacobian.ndim != 2:
        raise BSplineSU2DriverError("constraint Jacobian must be two-dimensional")
    collapsed = np.zeros((jacobian.shape[0], len(reduced_variables)), dtype=float)
    for column, variable in enumerate(reduced_variables):
        for index, sign in zip(variable.mode_indices, variable.signs):
            collapsed[:, column] += float(sign) * jacobian[:, int(index)]
    return collapsed


def mode_support_length(mode):
    key = _mode_support_key(mode)
    return max(0.0, float(key[3]) - float(key[2]))


def reduced_step_limits_from_modes(mode_spec, reduced_variables, ratio):
    ratio = _as_float(ratio, "BSPLINE_LOCAL_STEP_LIMIT_RATIO")
    if ratio <= 0.0:
        raise BSplineSU2DriverError("BSPLINE_LOCAL_STEP_LIMIT_RATIO must be positive")
    active_modes = _active_modes(mode_spec)
    limits = []
    for variable in reduced_variables:
        lengths = [mode_support_length(active_modes[int(index)]) for index in variable.mode_indices]
        if not lengths or min(lengths) <= 0.0:
            limits.append(math.inf)
        else:
            limits.append(min(lengths) / ratio)
    return limits


def _validated_bounds(bounds, name):
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise BSplineSU2DriverError(f"{name} must contain [lower, upper]")
    lower = _as_float(bounds[0], f"{name} lower")
    upper = _as_float(bounds[1], f"{name} upper")
    if upper < lower:
        raise BSplineSU2DriverError(f"{name} upper bound is below lower bound")
    return lower, upper


def update_mode_coefficients(mode_spec, coefficients):
    """Return a copy of mode_spec with active-mode coefficients replaced."""

    spec = copy.deepcopy(validate_mode_spec(mode_spec))
    values = [_as_float(value, "coefficient") for value in coefficients]
    active_count = len(_active_modes(spec))
    if len(values) != active_count:
        raise BSplineSU2DriverError(
            f"expected {active_count} active coefficient(s), got {len(values)}"
        )

    index = 0
    for mode in spec.get("modes", []):
        if mode.get("active", True) is False:
            continue
        mode["coefficient"] = values[index]
        index += 1

    return validate_mode_spec(spec)


def write_mode_spec(mode_spec, filename):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w") as fp:
        json.dump(mode_spec, fp, indent=2)
        fp.write("\n")


def cache_key(coefficients, tol=1.0e-12):
    tol = _as_float(tol, "cache tolerance")
    if tol <= 0.0:
        raise BSplineSU2DriverError("cache tolerance must be positive")
    digits = max(0, int(math.ceil(-math.log10(tol))))
    return tuple(round(_as_float(value, "coefficient"), digits) for value in coefficients)


def _format_config_atom(value):
    if isinstance(value, float):
        return "{:.15g}".format(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, separators=(",", ":"))
    if value is None:
        return ""
    return str(value)


def _format_config_value(value):
    if isinstance(value, (list, tuple)):
        return "( " + ", ".join(_format_config_atom(item) for item in value) + " )"
    return _format_config_atom(value)


def _vector_summary(values):
    values = np.asarray([float(value) for value in values], dtype=float)
    if values.size == 0:
        return "n=0"
    return "n={} min={:.6g} max={:.6g} norm2={:.6g}".format(
        values.size,
        float(np.min(values)),
        float(np.max(values)),
        float(np.linalg.norm(values)),
    )


def _bounds_summary(bounds):
    bounds = [(float(lower), float(upper)) for lower, upper in bounds]
    if not bounds:
        return "n=0"
    first = bounds[0]
    if all(
        abs(lower - first[0]) <= 1.0e-15 and abs(upper - first[1]) <= 1.0e-15
        for lower, upper in bounds
    ):
        return "uniform [{:.6g}, {:.6g}]".format(first[0], first[1])
    lowers = np.asarray([lower for lower, _upper in bounds], dtype=float)
    uppers = np.asarray([upper for _lower, upper in bounds], dtype=float)
    return (
        "nonuniform n={} lower[min={:.6g}, max={:.6g}] "
        "upper[min={:.6g}, max={:.6g}]"
    ).format(
        len(bounds),
        float(np.min(lowers)),
        float(np.max(lowers)),
        float(np.min(uppers)),
        float(np.max(uppers)),
    )


def _bounds_are_uniform(bounds):
    bounds = [(float(lower), float(upper)) for lower, upper in bounds]
    if not bounds:
        return True
    first = bounds[0]
    return all(
        abs(lower - first[0]) <= 1.0e-15 and abs(upper - first[1]) <= 1.0e-15
        for lower, upper in bounds
    )


def _line_config_key(line):
    stripped = line.lstrip()
    if not stripped or stripped.startswith("%") or stripped.startswith("#"):
        return None
    if "=" not in line:
        return None
    return line.split("=", 1)[0].strip()


def patch_config_template(template_filename, output_filename, updates):
    """Copy a SU2 config template and patch selected key/value assignments."""

    updates = {str(key).strip().upper(): value for key, value in updates.items()}
    seen = set()
    output_lines = []

    with open(template_filename, "r") as fp:
        for raw_line in fp:
            line = raw_line.rstrip("\n")
            key = _line_config_key(line)
            if key is not None and key.upper() in updates:
                update_key = key.upper()
                output_lines.append(
                    f"{key}= {_format_config_value(updates[update_key])}"
                )
                seen.add(update_key)
            else:
                output_lines.append(line)

    missing = [key for key in updates if key not in seen]
    if missing and output_lines and output_lines[-1].strip():
        output_lines.append("")
    for key in missing:
        output_lines.append(f"{key}= {_format_config_value(updates[key])}")

    Path(output_filename).parent.mkdir(parents=True, exist_ok=True)
    with open(output_filename, "w") as fp:
        fp.write("\n".join(output_lines))
        fp.write("\n")


def _normalize_eval_layout(eval_layout):
    if eval_layout is None:
        return "DSN"
    layout = str(eval_layout).strip().upper()
    if layout == "FLAT":
        raise BSplineSU2DriverError(
            "BSPLINE_EVAL_LAYOUT=FLAT has been removed. DSN is the only supported layout."
        )
    if layout not in ALLOWED_EVAL_LAYOUTS:
        raise BSplineSU2DriverError(
            f"BSPLINE_EVAL_LAYOUT must be one of {ALLOWED_EVAL_LAYOUTS}; got {layout!r}"
        )
    return layout


def _normalize_objective_adjoint(objective_adjoint):
    value = str(objective_adjoint or "drag").strip()
    normalized = _normalized_name(value).lower()
    if not normalized:
        raise BSplineSU2DriverError("objective adjoint name must be non-empty")
    return normalized


def _relative_path(target, start_dir):
    return os.path.relpath(Path(target), Path(start_dir))


def build_eval_paths(eval_dir, eval_layout="DSN", objective_adjoint="drag"):
    eval_dir = Path(eval_dir)
    eval_layout = _normalize_eval_layout(eval_layout)
    objective_adjoint = _normalize_objective_adjoint(objective_adjoint)
    deform_dir = eval_dir / "deform"
    direct_dir = eval_dir / "direct"
    adjoint_dir = eval_dir / f"adjoint_{objective_adjoint}"
    return EvalPaths(
        eval_layout=eval_layout,
        objective_adjoint=objective_adjoint,
        eval_dir=eval_dir,
        deform_dir=deform_dir,
        direct_dir=direct_dir,
        adjoint_dir=adjoint_dir,
        modes_current=eval_dir / "modes_current.json",
        surface_positions=deform_dir / "surface_positions.dat",
        metadata=deform_dir / "bspline_surface_metadata.csv",
        def_cfg=deform_dir / "def.cfg",
        primal_cfg=direct_dir / "primal.cfg",
        adjoint_cfg=adjoint_dir / "adjoint.cfg",
        deformed_mesh=deform_dir / "deformed_mesh.su2",
        primal_mesh_out=direct_dir / "primal_mesh_out.su2",
        adjoint_mesh_out=adjoint_dir / "adjoint_mesh_out.su2",
        primal_history=direct_dir / "history_primal.csv",
        adjoint_history=adjoint_dir / "history_adjoint.csv",
        primal_solution=direct_dir / "solution_flow.dat",
        primal_restart=direct_dir / "restart_flow.dat",
        adjoint_solution=adjoint_dir / "solution_adj.dat",
        adjoint_restart=adjoint_dir / "solution_adj.dat",
        volume_adjoint=adjoint_dir / "volume_adjoint",
        surface_adjoint=adjoint_dir / "surface_adjoint.csv",
        gradients=eval_dir / "bspline_gradients.csv",
        summary=eval_dir / "summary.json",
        commands_log=eval_dir / "commands.log",
        bspline_def_log=deform_dir / "bspline_def.log",
        su2_def_log=deform_dir / "su2_def.log",
        su2_cfd_log=direct_dir / "su2_cfd.log",
        su2_cfd_ad_log=adjoint_dir / "su2_cfd_ad.log",
        bspline_dot_log=adjoint_dir / "bspline_dot.log",
    )


def _command_list(command):
    if isinstance(command, str):
        return shlex.split(command)
    return [str(part) for part in command]


def _with_mpi(command, mpi_prefix=None):
    command = _command_list(command)
    if mpi_prefix:
        return shlex.split(str(mpi_prefix)) + command
    return command


def build_eval_commands(
    paths,
    base_mesh,
    marker,
    mpi_prefix=None,
    python_executable=None,
    sensitivity_weighting="NODAL",
    surface_mode="BOTH",
    deformation_direction_mode=None,
    le_safe_direction=False,
    le_safe_x0=None,
    le_safe_x1=None,
    le_safe_power=None,
):
    """Build subprocess commands for one evaluation directory."""

    python_executable = python_executable or sys.executable or "python3"
    try:
        sensitivity_weighting = normalize_sensitivity_weighting(sensitivity_weighting)
    except BSplineDotError as exc:
        raise BSplineSU2DriverError(str(exc))
    try:
        surface_mode = normalize_surface_mode(surface_mode)
        direction_mode = normalize_deformation_direction_mode(
            deformation_direction_mode,
            le_safe_direction=le_safe_direction,
        )
    except BSplineModeError as exc:
        raise BSplineSU2DriverError(str(exc))
    base_mesh = str(base_mesh)

    bspline_def = [
        python_executable,
        "-m",
        "SU2.opt.bspline_def",
        "--mesh",
        base_mesh,
        "--modes",
        _relative_path(paths.modes_current, paths.deform_dir),
        "--output",
        paths.surface_positions.name,
        "--metadata",
        paths.metadata.name,
        "--surface-mode",
        surface_mode,
        "--deformation-direction",
        direction_mode,
    ]
    if marker:
        bspline_def.extend(["--marker", str(marker)])
    if direction_mode == "LE_SAFE":
        bspline_def.append("--le-safe-direction")
        if le_safe_x0 is not None:
            bspline_def.extend(["--le-safe-x0", str(float(le_safe_x0))])
        if le_safe_x1 is not None:
            bspline_def.extend(["--le-safe-x1", str(float(le_safe_x1))])
        if le_safe_power is not None:
            bspline_def.extend(["--le-safe-power", str(float(le_safe_power))])

    return {
        "bspline_def": bspline_def,
        "def": _with_mpi(["SU2_DEF", paths.def_cfg.name], mpi_prefix),
        "primal": _with_mpi(["SU2_CFD", paths.primal_cfg.name], mpi_prefix),
        "adjoint": _with_mpi(["SU2_CFD_AD", paths.adjoint_cfg.name], mpi_prefix),
        "bspline_dot": [
            python_executable,
            "-m",
            "SU2.opt.bspline_dot",
            "--modes",
            paths.modes_current.name,
            "--metadata",
            _relative_path(paths.metadata, paths.eval_dir),
            "--sens",
            _relative_path(paths.surface_adjoint, paths.eval_dir),
            "--output",
            paths.gradients.name,
            "--summary",
            paths.summary.name,
            "--prefer-vector",
            "--sensitivity-weighting",
            sensitivity_weighting,
        ],
    }


def command_to_string(command):
    if isinstance(command, str):
        return command
    return " ".join(shlex.quote(str(part)) for part in command)


def _subprocess_env():
    env = os.environ.copy()
    root = str(Path(__file__).resolve().parents[2])
    pythonpath = env.get("PYTHONPATH", "")
    paths = [path for path in pythonpath.split(os.pathsep) if path]
    if root not in paths:
        paths.insert(0, root)
        env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def _tail_file(filename, max_lines=20, include_command=True):
    try:
        with open(filename, "r", errors="replace") as fp:
            lines = fp.readlines()
    except OSError:
        return ""
    if not include_command:
        lines = [line for line in lines if not line.startswith("$ ")]
    return "".join(lines[-max_lines:])


def run_command(command, cwd, log_file, show_command=False, stream_output=False, stage=None):
    """Run a command in cwd, writing stdout/stderr to log_file."""

    cwd = Path(cwd)
    log_file = Path(log_file)
    cwd.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    command_text = command_to_string(command)
    if show_command:
        print(f"[BSPLINE_SU2_DRIVER] {command_text}")
    with open(log_file, "w") as log:
        log.write("$ " + command_text + "\n")
        log.flush()
        if stream_output:
            process = subprocess.Popen(
                command if isinstance(command, str) else _command_list(command),
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=isinstance(command, str),
                env=_subprocess_env(),
                text=True,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                sys.stdout.write(line)
            return_code = process.wait()
        else:
            completed = subprocess.run(
                command if isinstance(command, str) else _command_list(command),
                cwd=str(cwd),
                stdout=log,
                stderr=subprocess.STDOUT,
                shell=isinstance(command, str),
                env=_subprocess_env(),
                text=True,
            )
            return_code = completed.returncode

    if return_code != 0:
        tail = _tail_file(log_file, max_lines=20, include_command=show_command).rstrip()
        message = [
            "failed stage: {}".format(stage or log_file.stem),
            "exit code: {}".format(return_code),
            "eval directory: {}".format(cwd),
            "log file: {}".format(log_file),
        ]
        if show_command:
            message.append("command: {}".format(command_text))
        message.append("last 20 log lines:")
        message.append(tail if tail else "(log file is empty)")
        raise BSplineSU2DriverError("\n".join(message))


def _append_command_log(log_file, stage, cwd, command):
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a") as fp:
        fp.write("[{}] cwd={}\n".format(stage or "command", Path(cwd)))
        fp.write("$ {}\n\n".format(command_to_string(command)))


def _symlink_or_copy(src, dst):
    src = Path(src)
    dst = Path(dst)
    if not src.exists() or src.resolve() == dst.resolve():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(os.path.relpath(src, dst.parent), dst)
    except OSError:
        shutil.copy2(src, dst)


def create_eval_aliases(paths):
    aliases = [
        (paths.metadata, paths.eval_dir / "bspline_surface_metadata.csv"),
        (paths.surface_adjoint, paths.eval_dir / "surface_adjoint.csv"),
        (paths.primal_history, paths.eval_dir / "history_primal.csv"),
        (paths.adjoint_history, paths.eval_dir / "history_adjoint.csv"),
    ]
    for src, dst in aliases:
        _symlink_or_copy(src, dst)


def _read_table(filename):
    lines = []
    with open(filename, "r") as fp:
        for raw_line in fp:
            line = raw_line.strip()
            if not line or line.startswith("%") or line.startswith("#"):
                continue
            upper = line.upper()
            if upper.startswith("TITLE"):
                continue
            if upper.startswith("VARIABLES") and "=" in line:
                line = line.split("=", 1)[1].strip()
            elif upper.startswith("VARIABLES"):
                continue
            lines.append(line)

    if not lines:
        raise BSplineSU2DriverError(f"{filename} has no readable table rows")

    if "," in lines[0]:
        rows = [
            [field.strip().strip('"').strip("'") for field in row]
            for row in csv.reader(lines, skipinitialspace=True)
            if row
        ]
    else:
        rows = []
        for line in lines:
            try:
                fields = shlex.split(line)
            except ValueError:
                fields = line.split()
            rows.append([field.strip().strip('"').strip("'") for field in fields])

    if not rows or len(rows) < 2:
        raise BSplineSU2DriverError(f"{filename} has no data rows")
    return rows[0], rows[1:]


def _find_column_index(headers, requested_column):
    requested = _normalized_name(requested_column)
    for index, header in enumerate(headers):
        if _normalized_name(header) == requested:
            return index
    raise BSplineSU2DriverError(
        "column {!r} was not found. Available columns: {}".format(
            requested_column,
            ", ".join(headers),
        )
    )


def read_objective_from_history(history_filename, objective_column):
    headers, rows = _read_table(history_filename)
    column_index = _find_column_index(headers, objective_column)

    value = None
    for row in rows:
        if column_index >= len(row):
            continue
        raw = row[column_index].strip()
        if raw:
            value = raw
    if value is None:
        raise BSplineSU2DriverError(
            f"{history_filename} has no values for column {objective_column!r}"
        )
    return _as_float(value, f"{history_filename} {objective_column}")


def read_bspline_gradients(gradients_filename, allow_nonfinite=False):
    with open(gradients_filename, "r", newline="") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames:
            raise BSplineSU2DriverError(f"{gradients_filename} is missing a header")
        mode_field = _find_field(reader.fieldnames, "mode_id")
        gradient_field = _find_field(reader.fieldnames, "gradient")

        gradients = {}
        for row_number, row in enumerate(reader, start=2):
            mode_id = str(row.get(mode_field, "")).strip()
            if not mode_id:
                raise BSplineSU2DriverError(
                    f"{gradients_filename}:{row_number} has an empty mode_id"
                )
            if mode_id in gradients:
                raise BSplineSU2DriverError(
                    f"{gradients_filename} has duplicate mode_id {mode_id!r}"
                )
            value = row.get(gradient_field, "")
            if allow_nonfinite:
                try:
                    gradients[mode_id] = float(value)
                except Exception:
                    raise BSplineSU2DriverError(
                        f"{gradients_filename}:{row_number} gradient must be numeric"
                    )
            else:
                gradients[mode_id] = _as_float(
                    value,
                    f"{gradients_filename}:{row_number} gradient",
                )
    return gradients


def _find_field(fieldnames, requested_field):
    requested = _normalized_name(requested_field)
    for fieldname in fieldnames:
        if _normalized_name(fieldname) == requested:
            return fieldname
    raise BSplineSU2DriverError(
        "field {!r} was not found. Available fields: {}".format(
            requested_field,
            ", ".join(fieldnames),
        )
    )


def read_gradient_vector(gradients_filename, mode_ids, allow_nonfinite=False):
    gradients = read_bspline_gradients(
        gradients_filename,
        allow_nonfinite=allow_nonfinite,
    )
    missing = [mode_id for mode_id in mode_ids if mode_id not in gradients]
    if missing:
        raise BSplineSU2DriverError(
            "bspline_gradients.csv is missing mode_id(s): "
            + ", ".join(str(mode_id) for mode_id in missing)
        )
    return [gradients[mode_id] for mode_id in mode_ids]


def ensure_adjoint_solution_input(primal_restart, primal_solution):
    primal_restart = Path(primal_restart)
    primal_solution = Path(primal_solution)
    if primal_solution.exists():
        return primal_solution
    if primal_restart.exists():
        shutil.copy2(primal_restart, primal_solution)
        return primal_solution
    raise BSplineSU2DriverError(
        "neither primal solution nor restart file exists: {}, {}".format(
            primal_solution,
            primal_restart,
        )
    )


def _project_to_bounds(coefficients, bounds):
    clipped = []
    for value, (lower, upper) in zip(coefficients, bounds):
        clipped.append(min(max(float(value), lower), upper))
    return clipped


def _bounds_to_list(bounds):
    return [[float(lower), float(upper)] for lower, upper in bounds]


def _max_radius_from_bounds(initial_coefficients, original_bounds):
    radii = []
    for coefficient, (lower, upper) in zip(initial_coefficients, original_bounds):
        radii.append(max(abs(float(lower) - float(coefficient)), abs(float(upper) - float(coefficient))))
    return radii


def compute_geometry_aware_bound_scaling(
    basis_matrix,
    initial_coefficients,
    original_bounds,
    max_normal_displacement=None,
    max_rms_normal_displacement=None,
    min_bound_scale=0.0,
):
    basis_matrix = np.asarray(basis_matrix, dtype=float)
    initial_coefficients = [float(value) for value in initial_coefficients]
    original_bounds = [
        _validated_bounds(bound, f"mode {index} bounds")
        for index, bound in enumerate(original_bounds)
    ]
    min_bound_scale = float(min_bound_scale)
    if min_bound_scale < 0.0:
        raise BSplineSU2DriverError("--min-bound-scale must be non-negative")

    if basis_matrix.ndim != 2:
        raise BSplineSU2DriverError("basis matrix for geometry-aware scaling must be two-dimensional")
    if basis_matrix.shape[1] != len(initial_coefficients):
        raise BSplineSU2DriverError("basis matrix column count must match the number of active coefficients")
    if basis_matrix.shape[1] != len(original_bounds):
        raise BSplineSU2DriverError("basis matrix column count must match the number of active bounds")

    abs_d0 = np.abs(basis_matrix.dot(np.asarray(initial_coefficients, dtype=float)))
    coefficient_radius = np.asarray(
        _max_radius_from_bounds(initial_coefficients, original_bounds),
        dtype=float,
    )
    row_radius = np.abs(basis_matrix).dot(coefficient_radius)
    max_abs_dn_current = float(np.max(abs_d0)) if len(abs_d0) else 0.0
    rms_dn_current = float(np.sqrt(np.mean(abs_d0 * abs_d0))) if len(abs_d0) else 0.0
    row_radius_max = float(np.max(row_radius)) if len(row_radius) else 0.0

    def _beta_from_max_limit(limit):
        limit = float(limit)
        if max_abs_dn_current > limit:
            raise BSplineSU2DriverError(
                "Initial geometry already violates max-normal-displacement; cannot make bounds safe by shrinking around it."
            )
        candidates = [
            (limit - float(abs_value)) / float(radius)
            for abs_value, radius in zip(abs_d0, row_radius)
            if float(radius) > 0.0
        ]
        beta = min([1.0] + candidates) if candidates else 1.0
        return min(1.0, max(0.0, float(beta)))

    def _rms_bound(beta):
        values = abs_d0 + float(beta) * row_radius
        return float(np.sqrt(np.mean(values * values))) if len(values) else 0.0

    def _beta_from_rms_limit(limit):
        limit = float(limit)
        if rms_dn_current > limit:
            raise BSplineSU2DriverError(
                "Initial geometry already violates max-rms-normal-displacement; cannot make bounds safe by shrinking around it."
            )
        if _rms_bound(1.0) <= limit:
            return 1.0
        lo = 0.0
        hi = 1.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if _rms_bound(mid) <= limit:
                lo = mid
            else:
                hi = mid
        return min(1.0, max(0.0, lo))

    betas = [1.0]
    if max_normal_displacement is not None:
        betas.append(_beta_from_max_limit(max_normal_displacement))
    if max_rms_normal_displacement is not None:
        betas.append(_beta_from_rms_limit(max_rms_normal_displacement))

    beta_safe = min(betas)
    if beta_safe < min_bound_scale:
        raise BSplineSU2DriverError(
            "Required geometry-safe bound scale beta={:.15g} is below --min-bound-scale={:.15g}".format(
                beta_safe,
                min_bound_scale,
            )
        )

    scaled_bounds = []
    for coefficient, (lower, upper) in zip(initial_coefficients, original_bounds):
        coeff = float(coefficient)
        scaled_bounds.append(
            (
                coeff + beta_safe * (float(lower) - coeff),
                coeff + beta_safe * (float(upper) - coeff),
            )
        )

    summary = {
        "enabled": True,
        "beta_safe": beta_safe,
        "max_abs_dn_current": max_abs_dn_current,
        "rms_dn_current": rms_dn_current,
        "max_normal_displacement": None if max_normal_displacement is None else float(max_normal_displacement),
        "max_rms_normal_displacement": None if max_rms_normal_displacement is None else float(max_rms_normal_displacement),
        "original_bounds": _bounds_to_list(original_bounds),
        "scaled_bounds": _bounds_to_list(scaled_bounds),
        "initial_coefficients": [float(value) for value in initial_coefficients],
        "row_radius_max": row_radius_max,
    }
    return summary


class BSplineThicknessConstraint:
    def __init__(
        self,
        metadata,
        basis_matrix,
        mode_ids,
        reference_measure,
        x_stations,
        margin=0.0,
        domain_mode="FULL",
        symmetry_y=0.0,
        fd_eps=1.0e-6,
        gradient_mode="AUTO",
        closed=False,
        marker=None,
    ):
        self.metadata = list(metadata)
        self.basis_matrix = np.asarray(basis_matrix, dtype=float)
        self.mode_ids = [str(mode_id) for mode_id in mode_ids]
        self.reference_measure = np.asarray(reference_measure, dtype=float)
        self.reference_thickness = self.reference_measure
        self.x_stations = np.asarray(x_stations, dtype=float)
        self.margin = float(margin)
        self.domain_mode = str(domain_mode).upper()
        self.symmetry_y = float(symmetry_y)
        self.fd_eps = float(fd_eps)
        self.gradient_mode = _normalize_gradient_mode(gradient_mode)
        self.closed = bool(closed)
        self.marker = None if marker is None else str(marker)
        self._fallback_warned = False
        self._switch_warned = False

        if self.domain_mode not in ("FULL", "HALF_UPPER", "HALF_LOWER"):
            raise BSplineSU2DriverError(
                "PROGRESSIVE_THICKNESS_DOMAIN_MODE must be FULL, HALF_UPPER, or HALF_LOWER"
            )
        if self.basis_matrix.ndim != 2:
            raise BSplineSU2DriverError("thickness basis matrix must be two-dimensional")
        if self.basis_matrix.shape[0] != len(self.metadata):
            raise BSplineSU2DriverError(
                "thickness basis matrix row count must match metadata rows"
            )
        if self.basis_matrix.shape[1] != len(self.mode_ids):
            raise BSplineSU2DriverError(
                "thickness basis matrix column count must match active modes"
            )
        if len(self.reference_measure) != len(self.x_stations):
            raise BSplineSU2DriverError(
                "reference thickness count must match x station count"
            )

        self.x_base = np.asarray([float(row["x"]) for row in self.metadata], dtype=float)
        self.y_base = np.asarray([float(row["y"]) for row in self.metadata], dtype=float)
        # Reconstruct the geometry with the effective NORMAL, LE_SAFE, or
        # VERTICAL direction recorded by bspline_def.
        self.deform_dir_x = np.asarray(
            [float(row.get("deform_dir_x", row["normal_x"])) for row in self.metadata],
            dtype=float,
        )
        self.deform_dir_y = np.asarray(
            [float(row.get("deform_dir_y", row["normal_y"])) for row in self.metadata],
            dtype=float,
        )
        self._segments = [(index, index + 1) for index in range(len(self.metadata) - 1)]
        if self.closed and len(self.metadata) > 2:
            self._segments.append((len(self.metadata) - 1, 0))

    def _deformed_arrays(self, coefficients):
        coefficients = np.asarray(coefficients, dtype=float)
        if coefficients.shape != (len(self.mode_ids),):
            raise BSplineSU2DriverError(
                f"expected {len(self.mode_ids)} thickness coefficient(s)"
            )
        dn = self.basis_matrix.dot(coefficients)
        x_def = self.x_base + self.deform_dir_x * dn
        y_def = self.y_base + self.deform_dir_y * dn
        dx_da = self.deform_dir_x[:, None] * self.basis_matrix
        dy_da = self.deform_dir_y[:, None] * self.basis_matrix
        return x_def, y_def, dx_da, dy_da

    def _station_hits(self, x_station, x_def, y_def, dx_da=None, dy_da=None):
        hits = []
        tol = 1.0e-12
        x_station = float(x_station)

        for i0, i1 in self._segments:
            x0, x1 = float(x_def[i0]), float(x_def[i1])
            y0, y1 = float(y_def[i0]), float(y_def[i1])
            if x_station < min(x0, x1) - tol or x_station > max(x0, x1) + tol:
                continue

            if abs(x1 - x0) <= tol:
                if abs(x_station - x0) <= tol:
                    if dy_da is None:
                        hits.append((y0, None))
                        hits.append((y1, None))
                    else:
                        hits.append((y0, np.asarray(dy_da[i0], dtype=float)))
                        hits.append((y1, np.asarray(dy_da[i1], dtype=float)))
                continue

            t = (x_station - x0) / (x1 - x0)
            if -tol <= t <= 1.0 + tol:
                t = max(0.0, min(1.0, float(t)))
                y_hit = y0 + t * (y1 - y0)
                if dx_da is None or dy_da is None:
                    hits.append((y_hit, None))
                    continue

                dx0 = np.asarray(dx_da[i0], dtype=float)
                dx1 = np.asarray(dx_da[i1], dtype=float)
                dy0 = np.asarray(dy_da[i0], dtype=float)
                dy1 = np.asarray(dy_da[i1], dtype=float)
                dt_da = (((t - 1.0) * dx0) - t * dx1) / (x1 - x0)
                dy_hit_da = (1.0 - t) * dy0 + t * dy1 + (y1 - y0) * dt_da
                hits.append((y_hit, dy_hit_da))

        return hits

    def section_measure(self, coefficients):
        x_def, y_def, _dx_da, _dy_da = self._deformed_arrays(coefficients)
        values = []
        for x_station in self.x_stations:
            hits = self._station_hits(x_station, x_def, y_def)
            if self.domain_mode == "FULL":
                if len(hits) < 2:
                    raise BSplineSU2DriverError(
                        f"Could not compute B-spline thickness at x={float(x_station):.12g}"
                    )
                y_values = [hit[0] for hit in hits]
                values.append(max(y_values) - min(y_values))
            elif self.domain_mode == "HALF_UPPER":
                if not hits:
                    raise BSplineSU2DriverError(
                        f"Could not compute B-spline upper half-thickness at x={float(x_station):.12g}"
                    )
                values.append(max(hit[0] for hit in hits) - self.symmetry_y)
            else:
                if not hits:
                    raise BSplineSU2DriverError(
                        f"Could not compute B-spline lower half-thickness at x={float(x_station):.12g}"
                    )
                values.append(self.symmetry_y - min(hit[0] for hit in hits))
        return np.asarray(values, dtype=float)

    def values(self, coefficients):
        current = self.section_measure(coefficients)
        return current - self.reference_measure - self.margin

    def jacobian_analytic(self, coefficients):
        x_def, y_def, dx_da, dy_da = self._deformed_arrays(coefficients)
        jac = np.zeros((len(self.x_stations), len(self.mode_ids)), dtype=float)
        # Switch margin: if the upper/lower pair at a station are closer than
        # this in y, the (upper, lower) identity is unstable under small
        # perturbations of the coefficients and the analytic gradient below
        # is only a subgradient. Warn once per driver so the user can decide
        # whether to use the FD fallback or to add safety margin.
        switch_margin = 1.0e-6

        for i_x, x_station in enumerate(self.x_stations):
            hits = self._station_hits(x_station, x_def, y_def, dx_da=dx_da, dy_da=dy_da)
            if self.domain_mode == "FULL":
                if len(hits) < 2:
                    raise BSplineSU2DriverError(
                        f"Could not compute B-spline thickness gradient at x={float(x_station):.12g}"
                    )
                upper = max(hits, key=lambda item: item[0])
                lower = min(hits, key=lambda item: item[0])
                if (
                    not self._switch_warned
                    and abs(upper[0] - lower[0]) < switch_margin
                ):
                    print(
                        "[BSPLINE_SU2_DRIVER] WARNING: analytic B-spline thickness gradient "
                        f"is near a max/min switch at x={float(x_station):.12g} "
                        f"(|y_u - y_l| = {abs(upper[0] - lower[0]):.3e}); the gradient "
                        "is only a subgradient. Consider enabling PROGRESSIVE_THICKNESS_GRADIENT=FINITE_DIFFERENCE "
                        "or increasing the safety margin."
                    )
                    self._switch_warned = True
                jac[i_x, :] = upper[1] - lower[1]
            elif self.domain_mode == "HALF_UPPER":
                if not hits:
                    raise BSplineSU2DriverError(
                        f"Could not compute B-spline upper half-thickness gradient at x={float(x_station):.12g}"
                    )
                upper = max(hits, key=lambda item: item[0])
                jac[i_x, :] = upper[1]
            else:
                if not hits:
                    raise BSplineSU2DriverError(
                        f"Could not compute B-spline lower half-thickness gradient at x={float(x_station):.12g}"
                    )
                lower = min(hits, key=lambda item: item[0])
                jac[i_x, :] = -lower[1]
        return jac

    def jacobian_fd_physical(self, coefficients):
        coefficients = np.asarray(coefficients, dtype=float)
        g0 = np.asarray(self.values(coefficients), dtype=float)
        jac = np.zeros((len(g0), len(coefficients)), dtype=float)
        for j in range(len(coefficients)):
            trial = coefficients.copy()
            trial[j] += self.fd_eps
            jac[:, j] = (np.asarray(self.values(trial), dtype=float) - g0) / self.fd_eps
        return jac

    def jacobian_physical(self, coefficients):
        if self.gradient_mode == "FINITE_DIFFERENCE":
            return self.jacobian_fd_physical(coefficients)
        try:
            return self.jacobian_analytic(coefficients)
        except Exception as exc:
            if self.gradient_mode == "ANALYTIC":
                raise
            if not self._fallback_warned:
                print(
                    "[BSPLINE_SU2_DRIVER] WARNING: analytic B-spline thickness gradient unavailable; "
                    f"falling back to finite differences ({exc})"
                )
                self._fallback_warned = True
            return self.jacobian_fd_physical(coefficients)


class BSplineSU2Driver:
    def __init__(
        self,
        modes_filename,
        base_mesh,
        marker,
        def_template,
        primal_template,
        adjoint_template,
        workdir,
        objective_column="CD",
        mpi_prefix="",
        default_bounds=DEFAULT_BOUNDS,
        cache_tol=1.0e-12,
        python_executable=None,
        show_commands=False,
        stream_solver_output=False,
        print_optimizer_table=True,
        auto_scale_bounds_to_geometry=False,
        max_normal_displacement=None,
        max_rms_normal_displacement=None,
        min_bound_scale=0.0,
        opt_accuracy=None,
        opt_bound_upper=None,
        opt_bound_lower=None,
        opt_relax_factor=1.0,
        opt_gradient_factor=1.0,
        gradient_guard=True,
        gradient_guard_factor=100.0,
        gradient_guard_window=5,
        gradient_guard_min_history=3,
        gradient_guard_floor=1.0e-14,
        gradient_guard_next_action="restart_same_level",
        refinement_available=None,
        trust_clip_policy="OFF",
        trust_clip_beta_tol=1.0e-12,
        trust_clip_legacy_beta_min=0.50,
        trust_clip_severe_beta=0.50,
        trust_clip_worsening_tol=0.05,
        trust_clip_soft_gnorm_factor=20.0,
        trust_clip_bad_patience=2,
        trust_clip_bad_window=5,
        trust_clip_stag_tol=1.0e-6,
        opt_line_search_bound=None,
        thickness_options=None,
        eval_layout="DSN",
        objective_adjoint="drag",
        symmetry_coupling="NONE",
        surface_mode="BOTH",
        sensitivity_weighting="NODAL",
        local_step_limit=False,
        local_step_limit_ratio=200.0,
        trigger_opts=None,
        progressive_label="PROGRESSIVE_BSPLINE",
        deformation_direction_mode=None,
        le_safe_direction=False,
        le_safe_x0=None,
        le_safe_x1=None,
        le_safe_power=None,
    ):
        self.modes_filename = Path(modes_filename).resolve()
        self.base_mesh = Path(base_mesh).resolve()
        self.marker = marker
        self.def_template = Path(def_template).resolve()
        self.primal_template = Path(primal_template).resolve()
        self.adjoint_template = Path(adjoint_template).resolve()
        self.workdir = Path(workdir).resolve()
        self.objective_column = objective_column
        self.eval_layout = _normalize_eval_layout(eval_layout)
        self.objective_adjoint = _normalize_objective_adjoint(objective_adjoint)
        try:
            self.surface_mode = normalize_surface_mode(surface_mode)
        except BSplineModeError as exc:
            raise BSplineSU2DriverError(str(exc))
        self.symmetry_coupling = str(symmetry_coupling or "NONE").strip().upper()
        if self.symmetry_coupling not in ALLOWED_SYMMETRY_COUPLINGS:
            raise BSplineSU2DriverError(
                f"BSPLINE_SYMMETRY_COUPLING must be one of {ALLOWED_SYMMETRY_COUPLINGS}; got {self.symmetry_coupling!r}"
            )
        if self.surface_mode != "BOTH" and self.symmetry_coupling != "NONE":
            raise BSplineSU2DriverError(
                "BSPLINE_SYMMETRY_COUPLING is only valid with BSPLINE_SURFACE_MODE=BOTH"
            )
        try:
            self.deformation_direction_mode = normalize_deformation_direction_mode(
                deformation_direction_mode,
                le_safe_direction=le_safe_direction,
            )
            self.le_safe_direction_options = validate_le_safe_direction_options(
                le_safe_direction=self.deformation_direction_mode == "LE_SAFE",
                le_safe_x0=le_safe_x0
                if self.deformation_direction_mode == "LE_SAFE"
                and le_safe_x0 is not None
                else LE_SAFE_DEFAULT_X0,
                le_safe_x1=le_safe_x1
                if self.deformation_direction_mode == "LE_SAFE"
                and le_safe_x1 is not None
                else LE_SAFE_DEFAULT_X1,
                le_safe_power=le_safe_power
                if self.deformation_direction_mode == "LE_SAFE"
                and le_safe_power is not None
                else LE_SAFE_DEFAULT_POWER,
            )
        except BSplineModeError as exc:
            raise BSplineSU2DriverError(str(exc))
        try:
            self.sensitivity_weighting = normalize_sensitivity_weighting(sensitivity_weighting)
        except BSplineDotError as exc:
            raise BSplineSU2DriverError(str(exc))
        self.mpi_prefix = mpi_prefix or ""
        self.default_bounds = _validated_bounds(default_bounds, "default_bounds")
        self.cache_tol = cache_tol
        self.python_executable = python_executable or sys.executable or "python3"
        self.show_commands = bool(show_commands)
        self.stream_solver_output = bool(stream_solver_output)
        self.print_optimizer_table = bool(print_optimizer_table)
        self.auto_scale_bounds_to_geometry = bool(auto_scale_bounds_to_geometry)
        self.max_normal_displacement = max_normal_displacement
        self.max_rms_normal_displacement = max_rms_normal_displacement
        self.min_bound_scale = 0.0 if min_bound_scale is None else float(min_bound_scale)
        self.opt_accuracy = None if opt_accuracy is None else float(opt_accuracy)
        self.opt_bound_upper = (
            None if opt_bound_upper is None else _as_float(opt_bound_upper, "OPT_BOUND_UPPER")
        )
        self.opt_bound_lower = (
            None if opt_bound_lower is None else _as_float(opt_bound_lower, "OPT_BOUND_LOWER")
        )
        self.opt_relax_factor = _as_float(
            1.0 if opt_relax_factor is None else opt_relax_factor,
            "OPT_RELAX_FACTOR",
        )
        if self.opt_relax_factor <= 0.0:
            raise BSplineSU2DriverError("OPT_RELAX_FACTOR must be positive")
        self.opt_gradient_factor = _as_float(
            1.0 if opt_gradient_factor is None else opt_gradient_factor,
            "OPT_GRADIENT_FACTOR",
        )
        if self.opt_gradient_factor <= 0.0:
            raise BSplineSU2DriverError("OPT_GRADIENT_FACTOR must be positive")
        self.gradient_guard_enabled = _thickness_as_bool(
            gradient_guard,
            default=True,
        )
        self.gradient_guard_factor = _as_float(
            gradient_guard_factor,
            "BSPLINE_GRADIENT_GUARD_FACTOR",
        )
        self.gradient_guard_window = int(gradient_guard_window)
        self.gradient_guard_min_history = int(gradient_guard_min_history)
        self.gradient_guard_floor = _as_float(
            gradient_guard_floor,
            "BSPLINE_GRADIENT_GUARD_FLOOR",
        )
        self.gradient_guard_next_action = str(
            gradient_guard_next_action or "restart_same_level"
        ).strip().lower()
        if self.gradient_guard_factor <= 0.0:
            raise BSplineSU2DriverError("BSPLINE_GRADIENT_GUARD_FACTOR must be positive")
        if self.gradient_guard_window < 1:
            raise BSplineSU2DriverError("BSPLINE_GRADIENT_GUARD_WINDOW must be >= 1")
        if self.gradient_guard_min_history < 1:
            raise BSplineSU2DriverError("BSPLINE_GRADIENT_GUARD_MIN_HISTORY must be >= 1")
        if self.gradient_guard_floor <= 0.0:
            raise BSplineSU2DriverError("BSPLINE_GRADIENT_GUARD_FLOOR must be positive")
        if self.gradient_guard_next_action not in (
            "refine",
            "restart_same_level",
            "terminate_last_safe",
        ):
            raise BSplineSU2DriverError(
                "gradient guard next action must be refine, restart_same_level, "
                "or terminate_last_safe"
            )
        self.refinement_available = (
            self.gradient_guard_next_action == "refine"
            if refinement_available is None
            else bool(refinement_available)
        )
        self.trust_clip_options = _trust_clip_options(
            {
                "policy": trust_clip_policy,
                "beta_tol": _as_float(trust_clip_beta_tol, "BSPLINE_TRUST_CLIP_BETA_TOL"),
                "legacy_beta_min": _as_float(
                    trust_clip_legacy_beta_min,
                    "BSPLINE_TRUST_CLIP_LEGACY_BETA_MIN",
                ),
                "severe_beta": _as_float(
                    trust_clip_severe_beta,
                    "BSPLINE_TRUST_CLIP_SEVERE_BETA",
                ),
                "worsening_tol": _as_float(
                    trust_clip_worsening_tol,
                    "BSPLINE_TRUST_CLIP_WORSENING_TOL",
                ),
                "soft_gnorm_factor": _as_float(
                    trust_clip_soft_gnorm_factor,
                    "BSPLINE_TRUST_CLIP_SOFT_GNORM_FACTOR",
                ),
                "bad_patience": int(trust_clip_bad_patience),
                "bad_window": int(trust_clip_bad_window),
                "stag_tol": _as_float(
                    trust_clip_stag_tol,
                    "BSPLINE_TRUST_CLIP_STAG_TOL",
                ),
                "gnorm_floor": self.gradient_guard_floor,
                "objective_floor": 1.0e-12,
            }
        )
        if self.trust_clip_options["policy"] not in ALLOWED_TRUST_CLIP_POLICIES:
            raise BSplineSU2DriverError(
                "BSPLINE_TRUST_CLIP_POLICY must be OFF or ACCEPT_RESTART"
            )
        for key in ("beta_tol", "worsening_tol", "stag_tol"):
            if float(self.trust_clip_options[key]) < 0.0:
                raise BSplineSU2DriverError(f"trust-clip {key} must be non-negative")
        for key in ("legacy_beta_min", "severe_beta"):
            if not 0.0 <= float(self.trust_clip_options[key]) <= 1.0:
                raise BSplineSU2DriverError(f"trust-clip {key} must be in [0, 1]")
        if float(self.trust_clip_options["soft_gnorm_factor"]) <= 0.0:
            raise BSplineSU2DriverError("trust-clip soft_gnorm_factor must be positive")
        if int(self.trust_clip_options["bad_patience"]) < 1:
            raise BSplineSU2DriverError("trust-clip bad_patience must be >= 1")
        if int(self.trust_clip_options["bad_window"]) < 1:
            raise BSplineSU2DriverError("trust-clip bad_window must be >= 1")
        self.opt_line_search_bound = (
            None
            if opt_line_search_bound is None
            else _as_float(opt_line_search_bound, "OPT_LINE_SEARCH_BOUND")
        )
        if self.opt_line_search_bound is not None and self.opt_line_search_bound <= 0.0:
            raise BSplineSU2DriverError("OPT_LINE_SEARCH_BOUND must be positive")
        self.local_step_limit = _thickness_as_bool(local_step_limit, default=False)
        self.local_step_limit_ratio = _as_float(
            local_step_limit_ratio,
            "BSPLINE_LOCAL_STEP_LIMIT_RATIO",
        )
        if self.local_step_limit_ratio <= 0.0:
            raise BSplineSU2DriverError("BSPLINE_LOCAL_STEP_LIMIT_RATIO must be positive")
        if (self.opt_bound_lower is None) != (self.opt_bound_upper is None):
            raise BSplineSU2DriverError(
                "OPT_BOUND_LOWER and OPT_BOUND_UPPER must be provided together"
            )
        self.thickness_options = dict(thickness_options or {})
        self.thickness_constraint = None
        self._thickness_constraint_configured = False
        self._thickness_fallback_warned = False
        self._printed_iteration_header = False

        for filename in (
            self.modes_filename,
            self.base_mesh,
            self.def_template,
            self.primal_template,
            self.adjoint_template,
        ):
            if not filename.exists():
                raise BSplineSU2DriverError(f"required file was not found: {filename}")

        self.mode_spec = load_mode_spec(str(self.modes_filename))
        try:
            validate_surface_mode_against_modes(self.mode_spec, self.surface_mode)
        except BSplineModeError as exc:
            raise BSplineSU2DriverError(str(exc))
        if self.surface_mode != "BOTH" and "surface_mode" not in self.mode_spec:
            self.mode_spec["surface_mode"] = self.surface_mode
        self.mode_ids = active_mode_ids(self.mode_spec)
        if not self.mode_ids:
            raise BSplineSU2DriverError("bspline_modes.json has no active modes")
        original_initial_coefficients = active_coefficient_vector(self.mode_spec)
        if self.opt_bound_lower is not None and self.opt_bound_upper is not None:
            override_bounds = _validated_bounds(
                (self.opt_bound_lower, self.opt_bound_upper),
                "OPT_BOUND_LOWER/OPT_BOUND_UPPER",
            )
            self.bounds = [override_bounds for _mode_id in self.mode_ids]
        else:
            self.bounds = active_bounds(self.mode_spec, self.default_bounds)
        self.original_bounds = list(self.bounds)
        self.reduced_variables, symmetry_warnings = build_reduced_variables(
            self.mode_spec,
            self.symmetry_coupling,
        )
        for warning in symmetry_warnings:
            print(f"[BSPLINE_SU2_DRIVER] WARNING: {warning}")
        self.reduced_variable_ids = [variable.id for variable in self.reduced_variables]
        self.initial_reduced_coefficients = compress_full_coefficients(
            original_initial_coefficients,
            self.reduced_variables,
            warn=lambda message: print(f"[BSPLINE_SU2_DRIVER] WARNING: {message}"),
            coupling=self.symmetry_coupling,
        )
        self.initial_coefficients = expand_reduced_coefficients(
            self.initial_reduced_coefficients,
            self.reduced_variables,
            len(self.mode_ids),
        )
        self.reduced_bounds = reduced_bounds_from_full_bounds(
            self.bounds,
            self.reduced_variables,
        )
        self.reduced_local_step_limits = reduced_step_limits_from_modes(
            self.mode_spec,
            self.reduced_variables,
            self.local_step_limit_ratio,
        )
        self.geometry_bounds_scaling = None
        self._geometry_bounds_configured = False
        self._geometry_probe = None
        self._line_search_basis_matrix = None
        self._line_search_bound_configured = False
        self._line_search_anchor_physical = list(self.initial_coefficients)
        self._local_step_anchor_reduced = list(self.initial_reduced_coefficients)
        self._cache = {}
        self._last_eval_physical_key = None
        self._last_eval_result = None
        self._last_eval_info = None
        self._history_records = []
        self.last_safe_entry = None
        self.best_safe_entry = None
        self.best_physical_entry = None
        self.recent_safe_raw_gnorms = deque(maxlen=self.gradient_guard_window)
        self.last_gradient_guard_stop = None
        self.anchor_entry = None
        self.recent_level_clip_events = deque(
            maxlen=int(self.trust_clip_options["bad_window"])
        )
        self._trust_clip_by_requested_key = {}
        self.last_trust_clip_stop = None
        self._slsqp_major_iter = 0
        self._run_eval_count = 0
        self._printed_commands_log_path = False
        self.trigger_project = SimpleNamespace(
            trigger_opts=dict(trigger_opts) if trigger_opts else None,
            trigger_history=[],
            trigger_state=None,
            refinement_triggered=False,
            progressive_label=str(progressive_label or "PROGRESSIVE_BSPLINE"),
        )
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._next_eval_id = self._initial_eval_id()

    def _probe_geometry_aware_bounds(self):
        if self._geometry_probe is not None:
            return self._geometry_probe

        probe_dir = self.workdir / "bounds_scaling_probe"
        probe_paths = build_eval_paths(
            probe_dir,
            eval_layout=self.eval_layout,
            objective_adjoint=self.objective_adjoint,
        )
        probe_dir.mkdir(parents=True, exist_ok=True)
        write_mode_spec(
            update_mode_coefficients(self.mode_spec, self.initial_coefficients),
            probe_paths.modes_current,
        )

        commands = build_eval_commands(
            probe_paths,
            self.base_mesh,
            self.marker,
            mpi_prefix=self.mpi_prefix,
            python_executable=self.python_executable,
            sensitivity_weighting=self.sensitivity_weighting,
            surface_mode=self.surface_mode,
            deformation_direction_mode=self.deformation_direction_mode,
            le_safe_direction=self.le_safe_direction_options["le_safe_direction"],
            le_safe_x0=self.le_safe_direction_options["le_safe_x0"],
            le_safe_x1=self.le_safe_direction_options["le_safe_x1"],
            le_safe_power=self.le_safe_direction_options["le_safe_power"],
        )
        _append_command_log(
            probe_paths.commands_log,
            "bounds_scaling_bspline_def",
            probe_paths.deform_dir,
            commands["bspline_def"],
        )
        _append_command_log(
            self.workdir / "commands.log",
            "bounds_scaling_bspline_def",
            probe_paths.deform_dir,
            commands["bspline_def"],
        )
        if self.show_commands and not self._printed_commands_log_path:
            print(
                "[BSPLINE_SU2_DRIVER] Commands are logged in {}".format(
                    self.workdir / "commands.log"
                )
            )
            self._printed_commands_log_path = True
        run_command(
            commands["bspline_def"],
            probe_paths.deform_dir,
            probe_paths.bspline_def_log,
            show_command=False,
            stream_output=self.stream_solver_output,
            stage="bounds_scaling_bspline_def",
        )

        metadata = read_metadata(probe_paths.metadata)
        x_over_c = [float(row["x_over_c"]) for row in metadata]
        sides = [str(row["side"]).strip().lower() for row in metadata]
        values_by_id = evaluate_all_modes(self.mode_spec, x_over_c, sides=sides)
        active_modes = _active_modes(self.mode_spec)
        columns = [
            np.asarray(values_by_id[str(mode["id"])], dtype=float)
            for mode in active_modes
        ]
        basis_matrix = (
            np.column_stack(columns) if columns else np.zeros((len(metadata), 0), dtype=float)
        )
        self._geometry_probe = (metadata, basis_matrix)
        return self._geometry_probe

    def configure_geometry_aware_bounds(self):
        if self._geometry_bounds_configured:
            return self.geometry_bounds_scaling

        self._geometry_bounds_configured = True
        if not self.auto_scale_bounds_to_geometry:
            return None
        if (
            self.max_normal_displacement is None
            and self.max_rms_normal_displacement is None
        ):
            raise BSplineSU2DriverError(
                "at least one of --max-normal-displacement or --max-rms-normal-displacement must be provided when --auto-scale-bounds-to-geometry is enabled"
            )

        _metadata, basis_matrix = self._probe_geometry_aware_bounds()
        summary = compute_geometry_aware_bound_scaling(
            basis_matrix,
            self.initial_coefficients,
            self.original_bounds,
            max_normal_displacement=self.max_normal_displacement,
            max_rms_normal_displacement=self.max_rms_normal_displacement,
            min_bound_scale=self.min_bound_scale,
        )
        self.bounds = [tuple(bounds) for bounds in summary["scaled_bounds"]]
        self.reduced_bounds = reduced_bounds_from_full_bounds(
            self.bounds,
            self.reduced_variables,
        )
        self.geometry_bounds_scaling = summary

        print("[BSPLINE_SU2_DRIVER] Geometry-aware bound scaling active")
        print(
            "[BSPLINE_SU2_DRIVER] max|dn| current = {:.15g}".format(
                summary["max_abs_dn_current"]
            )
        )
        print(
            "[BSPLINE_SU2_DRIVER] rms(dn) current = {:.15g}".format(
                summary["rms_dn_current"]
            )
        )
        print(
            "[BSPLINE_SU2_DRIVER] beta_safe = {:.15g}".format(
                summary["beta_safe"]
            )
        )
        print("[BSPLINE_SU2_DRIVER] coefficient bounds rescaled around current design")

        bounds_scaling_file = self.workdir / "bounds_scaling.json"
        with open(bounds_scaling_file, "w") as fp:
            json.dump(summary, fp, indent=2, sort_keys=True)
            fp.write("\n")
        return summary

    def physical_to_optimizer(self, coefficients):
        return [float(value) / self.opt_relax_factor for value in coefficients]

    def optimizer_to_physical(self, variables):
        return [float(value) * self.opt_relax_factor for value in variables]

    def expand_reduced_physical(self, reduced_coefficients):
        return expand_reduced_coefficients(
            reduced_coefficients,
            self.reduced_variables,
            len(self.mode_ids),
        )

    def compress_full_physical(self, coefficients, warn=False):
        # NOTE: coupling is intentionally NOT forwarded here. This wrapper is
        # called on coefficients that already came out of
        # expand_reduced_physical during optimization, so they are
        # antisymmetric by construction and the C1 unilateral-bump check
        # would only risk a spurious mid-run abort. The check is enforced
        # once, on the user-supplied initial coefficients, in __init__.
        return compress_full_coefficients(
            coefficients,
            self.reduced_variables,
            warn=(
                (lambda message: print(f"[BSPLINE_SU2_DRIVER] WARNING: {message}"))
                if warn
                else None
            ),
        )

    def collapse_gradient_to_reduced(self, gradient):
        return collapse_full_gradient(gradient, self.reduced_variables)

    def collapse_jacobian_to_reduced(self, jacobian):
        return collapse_full_jacobian(jacobian, self.reduced_variables)

    def optimizer_bounds(self):
        return [
            (
                float(lower) / self.opt_relax_factor,
                float(upper) / self.opt_relax_factor,
            )
            for lower, upper in self.reduced_bounds
        ]

    def _line_search_default_info(self):
        return {
            "line_search_beta": 1.0,
            "line_search_maxdiff": 0.0,
            "line_search_limited": 0,
            "line_search_beta_geometry": 1.0,
            "local_step_beta": 1.0,
            "local_step_limited": 0,
            "local_step_limiting_mode": "",
            "local_step_da": 0.0,
            "local_step_limit": 0.0,
        }

    def _configure_line_search_bound(self):
        if self._line_search_bound_configured:
            return self._line_search_basis_matrix

        self._line_search_bound_configured = True
        self._line_search_anchor_physical = list(self.initial_coefficients)
        self._local_step_anchor_reduced = list(self.initial_reduced_coefficients)
        if self.opt_line_search_bound is None:
            return None

        _metadata, basis_matrix = self._probe_geometry_aware_bounds()
        basis_matrix = np.asarray(basis_matrix, dtype=float)
        if basis_matrix.ndim != 2:
            raise BSplineSU2DriverError("basis matrix for OPT_LINE_SEARCH_BOUND must be two-dimensional")
        if basis_matrix.shape[1] != len(self.mode_ids):
            raise BSplineSU2DriverError(
                "basis matrix column count must match the number of active modes"
            )
        self._line_search_basis_matrix = basis_matrix
        print(
            "[BSPLINE_SU2_DRIVER] OPT_LINE_SEARCH_BOUND active: max accepted normal jump = {:.15g}".format(
                float(self.opt_line_search_bound)
            )
        )
        return self._line_search_basis_matrix

    def _marker_closed(self, marker):
        mesh = read_su2_mesh(str(self.base_mesh))
        _tag, _node_ids, closed = extract_marker_nodes(mesh, marker)
        return bool(closed)

    def _build_bspline_thickness_constraint(self):
        options = dict(self.thickness_options or {})
        enabled = _thickness_as_bool(
            options.get("PROGRESSIVE_THICKNESS_CONSTRAINT", "NO"),
            default=False,
        )
        if not enabled:
            return None

        ref_mesh_value = options.get("PROGRESSIVE_THICKNESS_REF_MESH")
        if not ref_mesh_value:
            raise BSplineSU2DriverError(
                "PROGRESSIVE_THICKNESS_REF_MESH is required when "
                "PROGRESSIVE_THICKNESS_CONSTRAINT=YES"
            )

        cfg = _ConfigDict(options)
        if options.get("_optimizer_config_filename"):
            cfg._filename = str(options["_optimizer_config_filename"])

        ref_mesh = _resolve_from_cfg_dir(cfg, ref_mesh_value)
        marker = str(options.get("PROGRESSIVE_THICKNESS_MARKER", self.marker))
        if marker.strip().lower() != str(self.marker).strip().lower():
            raise BSplineSU2DriverError(
                "B-spline thickness marker must match the optimizer marker "
                f"({marker!r} != {self.marker!r})"
            )

        domain_mode = resolve_thickness_domain_mode(
            self.surface_mode,
            options.get("PROGRESSIVE_THICKNESS_DOMAIN_MODE", "AUTO"),
        )
        symmetry_y = float(options.get("PROGRESSIVE_THICKNESS_SYMMETRY_Y", 0.0))
        margin = float(options.get("PROGRESSIVE_THICKNESS_MARGIN", 0.0))
        fd_eps = float(options.get("PROGRESSIVE_THICKNESS_FD_EPS", 1.0e-6))
        gradient_mode = _normalize_gradient_mode(
            options.get("PROGRESSIVE_THICKNESS_GRADIENT", "AUTO")
        )

        x_stations_value = options.get("PROGRESSIVE_THICKNESS_X_STATIONS")
        explicit_x_stations = not _x_stations_value_is_empty(x_stations_value)
        if explicit_x_stations:
            x_stations = _parse_x_stations(x_stations_value)
        else:
            npoints = int(options.get("PROGRESSIVE_THICKNESS_NPOINTS", 101))
            xmin = float(options.get("PROGRESSIVE_THICKNESS_XMIN", 0.001))
            xmax = float(options.get("PROGRESSIVE_THICKNESS_XMAX", 0.999))
            if npoints < 2:
                raise BSplineSU2DriverError("PROGRESSIVE_THICKNESS_NPOINTS must be >= 2")
            if not xmin < xmax:
                raise BSplineSU2DriverError("PROGRESSIVE_THICKNESS_XMIN must be < XMAX")
            x_stations = np.linspace(xmin, xmax, npoints)

        cache_value = options.get(
            "PROGRESSIVE_THICKNESS_CACHE_FILE",
            "thickness_reference.npz",
        )
        cache_file = _resolve_from_cfg_dir(cfg, cache_value) if cache_value else None
        reference = _load_or_build_reference(
            ref_mesh,
            marker,
            x_stations,
            cache_file,
            domain_mode,
            symmetry_y,
        )

        metadata, basis_matrix = self._probe_geometry_aware_bounds()
        closed = self._marker_closed(marker)
        constraint = BSplineThicknessConstraint(
            metadata,
            basis_matrix,
            self.mode_ids,
            reference,
            x_stations,
            margin=margin,
            domain_mode=domain_mode,
            symmetry_y=symmetry_y,
            fd_eps=fd_eps,
            gradient_mode=gradient_mode,
            closed=closed,
            marker=marker,
        )

        print("[BSPLINE_SU2_DRIVER] Thickness constraint active")
        print(f"[BSPLINE_SU2_DRIVER] thickness marker = {marker}")
        print(f"[BSPLINE_SU2_DRIVER] thickness domain mode = {domain_mode}")
        print(f"[PROGRESSIVE_BSPLINE][THICKNESS] domain = {domain_mode}")
        print(f"[PROGRESSIVE_BSPLINE][THICKNESS] symmetry_y = {symmetry_y}")
        print(f"[BSPLINE_SU2_DRIVER] thickness gradient mode = {gradient_mode}")
        print(f"[BSPLINE_SU2_DRIVER] thickness stations = {len(x_stations)}")
        print(
            "[BSPLINE_SU2_DRIVER] min reference thickness = {:.6e}".format(
                float(np.min(reference))
            )
        )
        return constraint

    def configure_thickness_constraint(self):
        if self._thickness_constraint_configured:
            return self.thickness_constraint
        self._thickness_constraint_configured = True
        self.thickness_constraint = self._build_bspline_thickness_constraint()
        return self.thickness_constraint

    def _thickness_values_for_physical(self, coefficients):
        if self.thickness_constraint is None:
            return None
        return np.asarray(self.thickness_constraint.values(coefficients), dtype=float)

    def _thickness_history_info(self, coefficients):
        values = self._thickness_values_for_physical(coefficients)
        if values is None:
            return {}
        min_value = float(np.min(values)) if len(values) else 0.0
        return {
            "min_thickness_constraint": min_value,
            "thickness_constraint_active": 1 if min_value <= 1.0e-10 else 0,
        }

    def _thickness_constraint_functions(self):
        if self.thickness_constraint is None:
            return []

        def thickness_fun(variables):
            reduced_trial = self.optimizer_to_physical(variables)
            physical_trial = self.expand_reduced_physical(reduced_trial)
            physical_eval, _info = self._apply_line_search_bound(physical_trial)
            return self._thickness_values_for_physical(physical_eval)

        def thickness_jac(variables):
            variables = np.asarray(variables, dtype=float)
            if self.thickness_constraint.gradient_mode == "FINITE_DIFFERENCE":
                return self._thickness_jacobian_fd_optimizer(variables, thickness_fun)

            reduced_trial = self.optimizer_to_physical(variables)
            physical_trial = self.expand_reduced_physical(reduced_trial)
            physical_eval, info = self._apply_line_search_bound(physical_trial)
            beta = float(info.get("line_search_beta", 1.0))
            try:
                jac_a = self.thickness_constraint.jacobian_analytic(physical_eval)
            except Exception as exc:
                if self.thickness_constraint.gradient_mode == "ANALYTIC":
                    raise
                if not self._thickness_fallback_warned:
                    print(
                        "[BSPLINE_SU2_DRIVER] WARNING: analytic B-spline thickness gradient unavailable; "
                        f"falling back to finite differences ({exc})"
                    )
                    self._thickness_fallback_warned = True
                return self._thickness_jacobian_fd_optimizer(variables, thickness_fun)
            jac_p = self.collapse_jacobian_to_reduced(jac_a)
            return np.asarray(jac_p, dtype=float) * self.opt_relax_factor * beta

        return [{"type": "ineq", "fun": thickness_fun, "jac": thickness_jac}]

    def _thickness_jacobian_fd_optimizer(self, variables, thickness_fun):
        variables = np.asarray(variables, dtype=float)
        g0 = np.asarray(thickness_fun(variables), dtype=float)
        jac = np.zeros((len(g0), len(variables)), dtype=float)
        eps = float(self.thickness_constraint.fd_eps)
        for j in range(len(variables)):
            trial = variables.copy()
            trial[j] += eps
            jac[:, j] = (np.asarray(thickness_fun(trial), dtype=float) - g0) / eps
        return jac

    def _local_step_limit_info(self, physical_trial):
        info = {
            "local_step_beta": 1.0,
            "local_step_limited": 0,
            "local_step_limiting_mode": "",
            "local_step_da": 0.0,
            "local_step_limit": 0.0,
        }
        if not self.local_step_limit:
            return info
        reduced_trial = self.compress_full_physical(physical_trial)
        anchor = np.asarray(self._local_step_anchor_reduced, dtype=float)
        trial = np.asarray(reduced_trial, dtype=float)
        delta = trial - anchor
        beta = 1.0
        limiting_index = None
        limiting_da = 0.0
        limiting_limit = 0.0
        eps = 1.0e-30
        for index, (da, limit) in enumerate(zip(delta, self.reduced_local_step_limits)):
            limit = float(limit)
            if not math.isfinite(limit) or limit <= 0.0:
                continue
            abs_da = abs(float(da))
            if abs_da > limit and abs_da > eps:
                candidate_beta = limit / abs_da
                if candidate_beta < beta:
                    beta = candidate_beta
                    limiting_index = index
                    limiting_da = float(da)
                    limiting_limit = limit
        if limiting_index is not None:
            variable = self.reduced_variables[limiting_index]
            info.update(
                {
                    "local_step_beta": float(beta),
                    "local_step_limited": 1,
                    "local_step_limiting_mode": ",".join(variable.mode_ids),
                    "local_step_da": limiting_da,
                    "local_step_limit": limiting_limit,
                }
            )
        return info

    def _apply_line_search_bound(self, physical_trial):
        physical_trial = [_as_float(value, "coefficient") for value in physical_trial]
        if len(physical_trial) != len(self.mode_ids):
            raise BSplineSU2DriverError(
                f"expected {len(self.mode_ids)} coefficients, got {len(physical_trial)}"
            )
        info = self._line_search_default_info()

        local_info = self._local_step_limit_info(physical_trial)
        info.update(local_info)

        if self.opt_line_search_bound is not None and self._line_search_basis_matrix is None:
            self._configure_line_search_bound()

        anchor = np.asarray(self._line_search_anchor_physical, dtype=float)
        trial = np.asarray(physical_trial, dtype=float)
        delta = trial - anchor

        geometry_beta = 1.0
        maxdiff = 0.0
        geometry_limited = 0
        if self.opt_line_search_bound is not None:
            basis_matrix = self._line_search_basis_matrix
            delta_dn = basis_matrix.dot(delta)
            maxdiff = float(np.max(np.abs(delta_dn))) if len(delta_dn) else 0.0
            if maxdiff > float(self.opt_line_search_bound) and maxdiff > 0.0:
                geometry_beta = float(self.opt_line_search_bound) / maxdiff
                geometry_limited = 1

        beta = min(float(geometry_beta), float(info["local_step_beta"]))
        limited = int(geometry_limited or info["local_step_limited"])
        if beta < 1.0:
            trial = anchor + beta * delta
        info.update(
            {
                "line_search_beta": float(beta),
                "line_search_beta_geometry": float(geometry_beta),
                "line_search_maxdiff": float(maxdiff),
                "line_search_limited": int(limited),
            }
        )
        return [float(value) for value in trial], info

    def _update_line_search_anchor_from_optimizer_variables(self, variables):
        reduced_trial = self.optimizer_to_physical(variables)
        physical_trial = self.expand_reduced_physical(reduced_trial)
        physical_eval, _info = self._apply_line_search_bound(physical_trial)
        self._line_search_anchor_physical = list(physical_eval)
        self._local_step_anchor_reduced = self.compress_full_physical(physical_eval)

    def _optimizer_gradient_for_logging(self, raw_gradient, line_search_info=None):
        info = line_search_info or {}
        beta = float(info.get("line_search_beta", 1.0))
        values = [float(value) for value in raw_gradient]
        reduced = []
        for variable in self.reduced_variables:
            reduced.append(
                sum(
                    float(sign) * values[int(index)]
                    for index, sign in zip(variable.mode_indices, variable.signs)
                )
            )
        return np.asarray(reduced, dtype=float) * (
            self.opt_relax_factor * self.opt_gradient_factor * beta
        )

    def _gradient_entry(self, result, paths, line_search_info=None):
        info = {
            **self._line_search_default_info(),
            **(line_search_info or {}),
        }
        raw_gradient = np.asarray(result.get("gradient") or [], dtype=float)
        optimizer_gradient = self._optimizer_gradient_for_logging(
            raw_gradient,
            info,
        )
        evaluated = [float(value) for value in result.get("coefficients", [])]
        requested = [
            float(value)
            for value in info.get("requested_x", evaluated)
        ]
        beta_eff = float(info.get("line_search_beta", 1.0))
        return {
            "eval_id": int(result.get("eval_id", -1)),
            "objective": float(result["objective"]),
            "requested_x": requested,
            "evaluated_x": evaluated,
            "beta_eff": beta_eff,
            "was_clipped": bool(
                int(info.get("line_search_limited", 0)) or beta_eff < 1.0
            ),
            "gnorm_raw": float(np.linalg.norm(raw_gradient)),
            "gnorm_opt": float(np.linalg.norm(optimizer_gradient)),
            "eval_dir": Path(paths.eval_dir),
            "modes_file": Path(paths.modes_current),
        }

    def restore_modes_from_entry(self, entry):
        self.optimized_modes_filename.parent.mkdir(parents=True, exist_ok=True)
        source = None if entry is None else entry.get("modes_file")
        if source is not None and Path(source).exists():
            source = Path(source)
            if source.resolve() != self.optimized_modes_filename.resolve():
                shutil.copy2(source, self.optimized_modes_filename)
        else:
            self.write_optimized_modes(self.initial_coefficients)
        return self.optimized_modes_filename

    def _log_gradient_guard_stop(self, bad_entry, guard_info):
        rollback = self.best_safe_entry or self.last_safe_entry
        safe_eval = (
            None
            if rollback is None
            else rollback.get("eval_id")
        )
        print("GRADIENT_GUARD_STOP")
        print(f"  reason          = {guard_info.get('reason')}")
        print(f"  bad_eval        = {bad_entry.get('eval_id')}")
        print(f"  restore_eval    = {safe_eval}")
        print(f"  gnorm_raw_bad   = {bad_entry.get('gnorm_raw')}")
        print(f"  gnorm_raw_ref   = {guard_info.get('reference')}")
        print(f"  raw_ratio       = {guard_info.get('ratio')}")
        print(f"  gnorm_opt_bad   = {bad_entry.get('gnorm_opt')}")
        print(f"  beta_eff_bad    = {bad_entry.get('beta_eff')}")
        print("  action          = rollback_to_best_safe")
        print(f"  next_action     = {self.gradient_guard_next_action}")

    def _promote_safe_entry(self, entry, update_recent_raw_gnorm=True):
        self.last_safe_entry = entry
        if update_recent_raw_gnorm:
            self.recent_safe_raw_gnorms.append(float(entry["gnorm_raw"]))
        objective = float(entry["objective"])
        if np.isfinite(objective) and (
            self.best_safe_entry is None
            or objective < float(self.best_safe_entry["objective"])
        ):
            self.best_safe_entry = entry
            self.best_physical_entry = entry
        if self.anchor_entry is None:
            self.anchor_entry = entry

    def register_gradient_entry(self, entry, promote=True):
        triggered = False
        guard_info = {
            "reason": "disabled",
            "gnorm_raw": float(entry["gnorm_raw"]),
            "reference": None,
            "ratio": None,
        }
        if self.gradient_guard_enabled:
            triggered, guard_info = gradient_guard_triggered(
                entry,
                self.recent_safe_raw_gnorms,
                factor=self.gradient_guard_factor,
                window=self.gradient_guard_window,
                min_history=self.gradient_guard_min_history,
                floor=self.gradient_guard_floor,
            )
        if triggered:
            rollback_entry = self.best_safe_entry or self.last_safe_entry
            self.restore_modes_from_entry(rollback_entry)
            self._log_gradient_guard_stop(entry, guard_info)
            stop = GradientGuardStop(
                rollback_entry,
                bad_entry=entry,
                guard_info=guard_info,
            )
            self.last_gradient_guard_stop = stop
            raise stop

        if promote:
            self._promote_safe_entry(entry)
        return guard_info

    def _trust_clip_enabled(self):
        return self.trust_clip_options["policy"] == "ACCEPT_RESTART"

    def _requested_optimizer_key(self, variables):
        return cache_key(variables, self.cache_tol)

    def _line_search_anchor_key(self):
        return cache_key(list(self._line_search_anchor_physical), self.cache_tol)

    def _pending_trust_clip_key(self, optimizer_variables):
        return (
            self._requested_optimizer_key(optimizer_variables),
            self._line_search_anchor_key(),
        )

    def _prune_stale_trust_clip_pending(self):
        current_anchor_key = self._line_search_anchor_key()
        for key in list(self._trust_clip_by_requested_key):
            _request_key, anchor_key = key
            if anchor_key != current_anchor_key:
                del self._trust_clip_by_requested_key[key]

    def _classify_trust_clip_entry(self, entry):
        classification, diagnostics = classify_clipped_trial(
            entry,
            self.recent_safe_raw_gnorms,
            self.best_safe_entry,
            self.anchor_entry,
            self.recent_level_clip_events,
            self.trust_clip_options,
        )
        event = {
            "classification": classification,
            "clipped": bool(diagnostics["clipped"]),
            "weak_improvement": bool(diagnostics["weak_improvement"]),
            "toxic": classification == "rejected_toxic_clip",
            "clipped_stagnation_plateau": bool(
                diagnostics["clipped_stagnation_plateau"]
            ),
        }
        self.recent_level_clip_events.append(event)
        return classification, diagnostics

    def _trust_clip_status(self, classification):
        return {
            "not_clipped": "ok",
            "benign_clipped_legacy": "ok_clipped_benign",
            "weak_clipped_progress": "ok_clipped_weak",
            "accepted_clipped_restart": "accepted_clipped_restart",
            "rejected_toxic_clip": "rejected_toxic_clip",
        }[classification]

    def _log_trust_clip_stop(self, stop):
        diagnostics = stop.diagnostics
        print("TRUST_CLIP_STOP")
        print(f"  class           = {stop.classification}")
        print(f"  eval             = {stop.entry.get('eval_id')}")
        print(
            "  restore_eval     = "
            f"{None if stop.rollback_entry is None else stop.rollback_entry.get('eval_id')}"
        )
        print(f"  beta_eff         = {diagnostics.get('beta_eff')}")
        print(f"  improvement_rel  = {diagnostics.get('improvement_rel')}")
        print(f"  relative_worsen  = {diagnostics.get('relative_worsening')}")
        print(f"  gnorm_ratio      = {diagnostics.get('gnorm_ratio')}")
        print(f"  reasons          = {','.join(diagnostics.get('toxic_reasons', []))}")
        print(f"  action           = {stop.action}")

    def _trust_clip_callback(self, optimizer_variables):
        if not self._trust_clip_enabled():
            self._update_line_search_anchor_from_optimizer_variables(optimizer_variables)
            return
        key = self._pending_trust_clip_key(optimizer_variables)
        pending = self._trust_clip_by_requested_key.pop(key, None)
        if pending is None:
            self._update_line_search_anchor_from_optimizer_variables(optimizer_variables)
            self._prune_stale_trust_clip_pending()
            return
        classification = pending["classification"]
        entry = pending["entry"]
        diagnostics = pending["diagnostics"]
        if classification in (
            "not_clipped",
            "benign_clipped_legacy",
            "weak_clipped_progress",
        ):
            self._update_line_search_anchor_from_optimizer_variables(optimizer_variables)
            self.anchor_entry = entry
            self._prune_stale_trust_clip_pending()
            return
        if classification == "accepted_clipped_restart":
            self._promote_safe_entry(entry)
            self._update_line_search_anchor_from_optimizer_variables(optimizer_variables)
            self.anchor_entry = entry
            rollback_entry = entry
            action = "restart_from_evaluated"
        else:
            rollback_entry = self.best_safe_entry or self.last_safe_entry
            toxic_reasons = set(diagnostics.get("toxic_reasons", []))
            force_refine = bool(
                self.refinement_available
                and toxic_reasons.intersection(
                    {"toxic_clipped_repeated", "clipped_stagnation_plateau"}
                )
            )
            action = "refine" if force_refine else "rollback_best_safe_restart"
        self._prune_stale_trust_clip_pending()
        self.restore_modes_from_entry(rollback_entry)
        stop = TrustClipStop(
            classification,
            entry,
            rollback_entry,
            diagnostics=diagnostics,
            action=action,
        )
        self.last_trust_clip_stop = stop
        self._log_trust_clip_stop(stop)
        raise stop

    def _evaluate_optimizer_variables(self, variables):
        reduced_trial = self.optimizer_to_physical(variables)
        physical_trial = self.expand_reduced_physical(reduced_trial)
        physical_eval, info = self._apply_line_search_bound(physical_trial)
        info["requested_x"] = list(physical_trial)
        info["evaluated_x"] = list(physical_eval)
        info["requested_optimizer_x"] = [float(value) for value in variables]
        physical_key = cache_key(physical_eval, self.cache_tol)
        beta_tol = float(self.trust_clip_options.get("beta_tol", 1.0e-12))
        beta_now = float(info.get("beta_eff", info.get("line_search_beta", 1.0)))
        if (
            self._last_eval_physical_key == physical_key
            and self._last_eval_result is not None
            and self._last_eval_info is not None
        ):
            beta_prev = float(
                self._last_eval_info.get(
                    "beta_eff",
                    self._last_eval_info.get("line_search_beta", 1.0),
                )
            )
            prev_class = str(self._last_eval_result.get("trust_clip_class", ""))
            if (
                beta_prev >= 1.0 - beta_tol
                and beta_now >= 1.0 - beta_tol
                and prev_class not in LAST_EVAL_CACHE_BLOCKED_TRUST_CLIP_CLASSES
            ):
                cached_info = dict(self._last_eval_info)
                cached_info["cache_hit"] = True
                cached_info["last_eval_cache_hit"] = True
                cached_info["requested_optimizer_x"] = [float(value) for value in variables]
                cached_info["requested_x"] = list(physical_trial)
                cached_info["evaluated_x"] = list(physical_eval)
                return self._last_eval_result, cached_info
        if self._trust_clip_enabled():
            pending = self._trust_clip_by_requested_key.get(
                self._pending_trust_clip_key(variables)
            )
            if pending is not None:
                return pending["result"], info
        result = self.evaluate(physical_eval, line_search_info=info)
        self._last_eval_physical_key = physical_key
        self._last_eval_result = result
        self._last_eval_info = dict(info)
        return result, info

    def _evaluate_reduced_physical(self, reduced_coefficients):
        physical_trial = self.expand_reduced_physical(reduced_coefficients)
        physical_eval, info = self._apply_line_search_bound(physical_trial)
        info["requested_x"] = list(physical_trial)
        info["evaluated_x"] = list(physical_eval)
        result = self.evaluate(physical_eval, line_search_info=info)
        reduced_eval = self.compress_full_physical(physical_eval)
        return result, info, reduced_eval

    def _initial_eval_id(self):
        ids = []
        for path in self.workdir.glob("eval_[0-9][0-9][0-9][0-9]"):
            try:
                ids.append(int(path.name.split("_", 1)[1]))
            except Exception:
                pass
        return max(ids) + 1 if ids else 0

    @property
    def optimization_history_filename(self):
        return self.workdir / "optimization_history.csv"

    @property
    def optimized_modes_filename(self):
        return self.workdir / "optimized_modes.json"

    def _next_paths(self):
        while True:
            eval_id = self._next_eval_id
            self._next_eval_id += 1
            paths = build_eval_paths(
                self.workdir / f"eval_{eval_id:04d}",
                eval_layout=self.eval_layout,
                objective_adjoint=self.objective_adjoint,
            )
            if not paths.eval_dir.exists():
                return eval_id, paths

    def _prepare_eval_files(self, coefficients, paths):
        paths.eval_dir.mkdir(parents=True, exist_ok=False)
        for directory in {paths.deform_dir, paths.direct_dir, paths.adjoint_dir}:
            directory.mkdir(parents=True, exist_ok=True)
        current_spec = update_mode_coefficients(self.mode_spec, coefficients)
        write_mode_spec(current_spec, paths.modes_current)

        patch_config_template(
            self.def_template,
            paths.def_cfg,
            {
                "MESH_FILENAME": str(self.base_mesh),
                "MESH_OUT_FILENAME": paths.deformed_mesh.name,
                "DV_KIND": "SURFACE_FILE",
                "DV_MARKER": [self.marker],
                "DV_FILENAME": paths.surface_positions.name,
            },
        )

        primal_mesh_filename = _relative_path(paths.deformed_mesh, paths.direct_dir)
        adjoint_mesh_filename = _relative_path(paths.deformed_mesh, paths.adjoint_dir)
        adjoint_flow_solution = _relative_path(paths.primal_solution, paths.adjoint_dir)
        adjoint_flow_restart = _relative_path(paths.primal_restart, paths.adjoint_dir)

        shared_primal_updates = {
            "SOLUTION_FILENAME": paths.primal_solution.name,
            "RESTART_FILENAME": paths.primal_restart.name,
            "SOLUTION_ADJ_FILENAME": paths.adjoint_solution.name,
            "RESTART_ADJ_FILENAME": paths.adjoint_restart.name,
            "SURFACE_ADJ_FILENAME": paths.surface_adjoint.stem,
            "VOLUME_ADJ_FILENAME": paths.volume_adjoint.name,
            "TABULAR_FORMAT": "CSV",
        }
        patch_config_template(
            self.primal_template,
            paths.primal_cfg,
            dict(
                shared_primal_updates,
                MESH_FILENAME=primal_mesh_filename,
                MESH_OUT_FILENAME=paths.primal_mesh_out.name,
                CONV_FILENAME=paths.primal_history.stem,
                HISTORY_OUTPUT=["ITER", "RMS_RES", "AERO_COEFF"],
                SCREEN_OUTPUT=["INNER_ITER", "RMS_RES", "LIFT", "DRAG"],
            ),
        )
        patch_config_template(
            self.adjoint_template,
            paths.adjoint_cfg,
            dict(
                shared_primal_updates,
                MESH_FILENAME=adjoint_mesh_filename,
                SOLUTION_FILENAME=adjoint_flow_solution,
                RESTART_FILENAME=adjoint_flow_restart,
                SURFACE_ADJ_FILENAME=paths.surface_adjoint.stem,
                VOLUME_ADJ_FILENAME=paths.volume_adjoint.name,
                MESH_OUT_FILENAME=paths.adjoint_mesh_out.name,
                CONV_FILENAME=paths.adjoint_history.stem,
            ),
        )

    def _write_eval_summary_metadata(self, paths, line_search_info=None):
        data = {}
        if paths.summary.exists():
            try:
                with open(paths.summary, "r") as fp:
                    data = json.load(fp)
            except Exception:
                data = {}
        line_search_info = {
            **self._line_search_default_info(),
            **(line_search_info or {}),
        }
        data.update(
            {
                "eval_layout": paths.eval_layout,
                "deform_dir": str(paths.deform_dir),
                "direct_dir": str(paths.direct_dir),
                "adjoint_dir": str(paths.adjoint_dir),
                "objective_adjoint": paths.objective_adjoint,
                "symmetry_coupling": self.symmetry_coupling,
                "surface_mode": self.surface_mode,
                "sensitivity_weighting": self.sensitivity_weighting,
                "n_active_modes": len(self.mode_ids),
                "n_design_variables": len(self.reduced_variable_ids),
                "local_step_limit_enabled": bool(self.local_step_limit),
                "local_step_limit_ratio": float(self.local_step_limit_ratio),
                "local_step_beta": line_search_info.get("local_step_beta", 1.0),
                "local_step_limited": line_search_info.get("local_step_limited", 0),
                "line_search_beta": line_search_info.get("line_search_beta", 1.0),
            }
        )
        with open(paths.summary, "w") as fp:
            json.dump(data, fp, indent=2, sort_keys=True)
            fp.write("\n")

    def _write_gradient_guard_summary(self, paths, entry, guard_info, status):
        data = {}
        if paths.summary.exists():
            try:
                with open(paths.summary, "r") as fp:
                    data = json.load(fp)
            except Exception:
                data = {}
        data["gradient_guard"] = {
            "status": str(status),
            "reason": guard_info.get("reason"),
            "gnorm_raw": entry.get("gnorm_raw"),
            "gnorm_opt": entry.get("gnorm_opt"),
            "reference": guard_info.get("reference"),
            "ratio": guard_info.get("ratio"),
            "beta_eff": entry.get("beta_eff"),
            "was_clipped": bool(entry.get("was_clipped", False)),
        }
        with open(paths.summary, "w") as fp:
            json.dump(data, fp, indent=2, sort_keys=True)
            fp.write("\n")

    def _write_trust_clip_summary(self, paths, classification, diagnostics, action):
        data = {}
        if paths.summary.exists():
            try:
                with open(paths.summary, "r") as fp:
                    data = json.load(fp)
            except Exception:
                data = {}
        data["trust_clip"] = {
            "policy": self.trust_clip_options["policy"],
            "classification": classification,
            "action": action,
            **dict(diagnostics),
        }
        with open(paths.summary, "w") as fp:
            json.dump(data, fp, indent=2, sort_keys=True)
            fp.write("\n")

    def _append_history_record(
        self,
        eval_id,
        objective,
        coefficients,
        gradient,
        status,
        line_search_info=None,
        eval_dir=None,
        eval_index=None,
        gradient_guard_info=None,
        gradient_entry=None,
        trust_clip_classification="",
        trust_clip_diagnostics=None,
        trust_clip_action="",
    ):
        line_search_info = {
            **self._line_search_default_info(),
            **(line_search_info or {}),
        }
        reduced_coefficients = (
            self.compress_full_physical(coefficients)
            if self.symmetry_coupling != "NONE"
            else []
        )
        gradient_values = list(gradient or [])
        gradient_is_finite = all(np.isfinite(float(value)) for value in gradient_values)
        reduced_gradient = (
            self.collapse_gradient_to_reduced(gradient_values)
            if self.symmetry_coupling != "NONE"
            and gradient_values
            and gradient_is_finite
            else []
        )
        gradient_guard_info = dict(gradient_guard_info or {})
        gradient_entry = dict(gradient_entry or {})
        trust_clip_diagnostics = dict(trust_clip_diagnostics or {})
        record = {
            "eval_index": eval_index,
            "slsqp_iter": self._slsqp_major_iter,
            "eval_id": eval_id,
            "eval_dir": str(eval_dir) if eval_dir is not None else "",
            "objective": objective,
            "coefficients": list(coefficients),
            "reduced_coefficients": list(reduced_coefficients),
            "gradients": gradient_values,
            "gnorm_raw": gradient_entry.get("gnorm_raw", ""),
            "gnorm_opt": gradient_entry.get("gnorm_opt", ""),
            "gradient_guard_reason": gradient_guard_info.get("reason", ""),
            "gradient_guard_reference": gradient_guard_info.get("reference", ""),
            "gradient_guard_ratio": gradient_guard_info.get("ratio", ""),
            "trust_clip_class": trust_clip_classification,
            "trust_clip_action": trust_clip_action,
            "trust_clip_beta": trust_clip_diagnostics.get("beta_eff", ""),
            "trust_clip_improvement_rel": trust_clip_diagnostics.get(
                "improvement_rel", ""
            ),
            "trust_clip_relative_worsening": trust_clip_diagnostics.get(
                "relative_worsening", ""
            ),
            "trust_clip_gnorm_ratio": trust_clip_diagnostics.get("gnorm_ratio", ""),
            "trust_clip_reasons": ",".join(
                trust_clip_diagnostics.get("toxic_reasons", [])
                or trust_clip_diagnostics.get("accepted_reasons", [])
            ),
            "status": status,
            "line_search_beta": line_search_info["line_search_beta"],
            "line_search_maxdiff": line_search_info["line_search_maxdiff"],
            "line_search_limited": line_search_info["line_search_limited"],
            "local_step_beta": line_search_info["local_step_beta"],
            "local_step_limited": line_search_info["local_step_limited"],
            "local_step_limiting_mode": line_search_info["local_step_limiting_mode"],
            "local_step_da": line_search_info["local_step_da"],
            "local_step_limit": line_search_info["local_step_limit"],
        }
        if self.thickness_constraint is not None:
            try:
                record.update(self._thickness_history_info(coefficients))
            except Exception:
                record["min_thickness_constraint"] = ""
                record["thickness_constraint_active"] = ""
        for mode_id, coefficient in zip(self.mode_ids, coefficients):
            record[f"coeff__{mode_id}"] = coefficient
        if self.symmetry_coupling != "NONE":
            for reduced_id, coefficient in zip(self.reduced_variable_ids, reduced_coefficients):
                record[f"reduced_coeff__{reduced_id}"] = coefficient
        for mode_id, value in zip(self.mode_ids, gradient_values):
            record[f"grad__{mode_id}"] = value
        if self.symmetry_coupling != "NONE":
            for reduced_id, value in zip(self.reduced_variable_ids, reduced_gradient):
                record[f"reduced_grad__{reduced_id}"] = value
        self._history_records.append(record)
        self.write_optimization_history()

    def _history_fieldnames(self):
        fieldnames = (
            [
                "eval_index",
                "slsqp_iter",
                "eval_id",
                "eval_dir",
                "objective",
                "coefficients",
                "reduced_coefficients",
                "gradients",
                "gnorm_raw",
                "gnorm_opt",
                "gradient_guard_reason",
                "gradient_guard_reference",
                "gradient_guard_ratio",
                "trust_clip_class",
                "trust_clip_action",
                "trust_clip_beta",
                "trust_clip_improvement_rel",
                "trust_clip_relative_worsening",
                "trust_clip_gnorm_ratio",
                "trust_clip_reasons",
            ]
            + [f"coeff__{mode_id}" for mode_id in self.mode_ids]
            + (
                [f"reduced_coeff__{reduced_id}" for reduced_id in self.reduced_variable_ids]
                if self.symmetry_coupling != "NONE"
                else []
            )
            + [f"grad__{mode_id}" for mode_id in self.mode_ids]
            + (
                [f"reduced_grad__{reduced_id}" for reduced_id in self.reduced_variable_ids]
                if self.symmetry_coupling != "NONE"
                else []
            )
            + [
                "line_search_beta",
                "line_search_maxdiff",
                "line_search_limited",
                "local_step_beta",
                "local_step_limited",
                "local_step_limiting_mode",
                "local_step_da",
                "local_step_limit",
            ]
        )
        if self.thickness_constraint is not None:
            fieldnames += [
                "min_thickness_constraint",
                "thickness_constraint_active",
            ]
        fieldnames += ["status"]
        return fieldnames

    def _best_ok_history_record(self):
        ok_records = [
            record
            for record in self._history_records
            if str(record.get("status", "")).strip().lower()
            in SAFE_EVALUATION_STATUSES
            and record.get("objective") is not None
            and np.isfinite(float(record.get("objective")))
        ]
        if not ok_records:
            return None
        return min(ok_records, key=lambda record: float(record["objective"]))

    def write_optimization_history(self):
        fieldnames = self._history_fieldnames()
        with open(self.optimization_history_filename, "w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for record in self._history_records:
                writer.writerow(
                    {
                        field: _format_config_atom(record.get(field, ""))
                        for field in fieldnames
                    }
                )

    def evaluate(self, coefficients, line_search_info=None):
        coefficients = [_as_float(value, "coefficient") for value in coefficients]
        if len(coefficients) != len(self.mode_ids):
            raise BSplineSU2DriverError(
                f"expected {len(self.mode_ids)} coefficients, got {len(coefficients)}"
            )

        key = cache_key(coefficients, self.cache_tol)
        if key in self._cache and not self._trust_clip_enabled():
            return self._cache[key]

        eval_id, paths = self._next_paths()
        self._run_eval_count += 1
        eval_index = self._run_eval_count
        objective = None
        gradient = None
        try:
            self._prepare_eval_files(coefficients, paths)
            commands = build_eval_commands(
                paths,
                self.base_mesh,
                self.marker,
                mpi_prefix=self.mpi_prefix,
                python_executable=self.python_executable,
                sensitivity_weighting=self.sensitivity_weighting,
                surface_mode=self.surface_mode,
                deformation_direction_mode=self.deformation_direction_mode,
                le_safe_direction=self.le_safe_direction_options["le_safe_direction"],
                le_safe_x0=self.le_safe_direction_options["le_safe_x0"],
                le_safe_x1=self.le_safe_direction_options["le_safe_x1"],
                le_safe_power=self.le_safe_direction_options["le_safe_power"],
            )
            for stage, command in commands.items():
                _append_command_log(paths.commands_log, stage, paths.eval_dir, command)
                _append_command_log(self.workdir / "commands.log", stage, paths.eval_dir, command)
            if self.show_commands and not self._printed_commands_log_path:
                print(
                    "[BSPLINE_SU2_DRIVER] Commands are logged in {}".format(
                        self.workdir / "commands.log"
                    )
                )
                self._printed_commands_log_path = True

            run_command(
                commands["bspline_def"],
                paths.deform_dir,
                paths.bspline_def_log,
                show_command=False,
                stream_output=self.stream_solver_output,
                stage="bspline_def",
            )
            run_command(
                commands["def"],
                paths.deform_dir,
                paths.su2_def_log,
                show_command=False,
                stream_output=self.stream_solver_output,
                stage="su2_def",
            )
            run_command(
                commands["primal"],
                paths.direct_dir,
                paths.su2_cfd_log,
                show_command=False,
                stream_output=self.stream_solver_output,
                stage="su2_cfd",
            )
            ensure_adjoint_solution_input(paths.primal_restart, paths.primal_solution)
            objective = read_objective_from_history(
                paths.primal_history,
                self.objective_column,
            )
            run_command(
                commands["adjoint"],
                paths.adjoint_dir,
                paths.su2_cfd_ad_log,
                show_command=False,
                stream_output=self.stream_solver_output,
                stage="su2_cfd_ad",
            )
            run_command(
                commands["bspline_dot"],
                paths.eval_dir,
                paths.bspline_dot_log,
                show_command=False,
                stream_output=self.stream_solver_output,
                stage="bspline_dot",
            )
            create_eval_aliases(paths)
            self._write_eval_summary_metadata(paths, line_search_info=line_search_info)
            gradient = read_gradient_vector(
                paths.gradients,
                self.mode_ids,
                allow_nonfinite=(
                    self.gradient_guard_enabled or self._trust_clip_enabled()
                ),
            )

            result = {
                "eval_index": eval_index,
                "eval_id": eval_id,
                "eval_dir": str(paths.eval_dir),
                "objective": objective,
                "gradient": gradient,
                "coefficients": coefficients,
                "status": "ok",
            }
            gradient_entry = self._gradient_entry(
                result,
                paths,
                line_search_info=line_search_info,
            )
            try:
                guard_info = self.register_gradient_entry(
                    gradient_entry,
                    promote=not self._trust_clip_enabled(),
                )
            except GradientGuardStop as stop:
                result.update(
                    {
                        "status": "rejected_gradient_guard",
                        "gnorm_raw": gradient_entry["gnorm_raw"],
                        "gnorm_opt": gradient_entry["gnorm_opt"],
                        "gradient_guard_info": stop.guard_info,
                    }
                )
                self._write_gradient_guard_summary(
                    paths,
                    gradient_entry,
                    stop.guard_info,
                    "rejected",
                )
                self._append_history_record(
                    eval_id,
                    objective,
                    coefficients,
                    gradient,
                    "rejected_gradient_guard",
                    line_search_info=line_search_info,
                    eval_dir=paths.eval_dir,
                    eval_index=eval_index,
                    gradient_guard_info=stop.guard_info,
                    gradient_entry=gradient_entry,
                )
                raise

            trust_clip_classification = ""
            trust_clip_diagnostics = {}
            trust_clip_action = ""
            status = "ok"
            cache_as_safe = True
            if self._trust_clip_enabled():
                (
                    trust_clip_classification,
                    trust_clip_diagnostics,
                ) = self._classify_trust_clip_entry(gradient_entry)
                status = self._trust_clip_status(trust_clip_classification)
                trust_clip_action = {
                    "not_clipped": "continue",
                    "benign_clipped_legacy": "legacy_continue",
                    "weak_clipped_progress": "weak_continue",
                    "accepted_clipped_restart": "defer_restart_to_callback",
                    "rejected_toxic_clip": "defer_rollback_to_callback",
                }[trust_clip_classification]
                finite_safe_candidate = bool(
                    np.isfinite(float(gradient_entry["objective"]))
                    and np.isfinite(float(gradient_entry["gnorm_raw"]))
                )
                if trust_clip_classification in (
                    "not_clipped",
                    "benign_clipped_legacy",
                    "weak_clipped_progress",
                ) and finite_safe_candidate:
                    self._promote_safe_entry(
                        gradient_entry,
                        update_recent_raw_gnorm=(
                            trust_clip_classification != "weak_clipped_progress"
                        ),
                    )
                cache_as_safe = (
                    finite_safe_candidate
                    and trust_clip_classification
                    in (
                        "not_clipped",
                        "benign_clipped_legacy",
                        "weak_clipped_progress",
                    )
                )
                requested_optimizer_x = (line_search_info or {}).get(
                    "requested_optimizer_x"
                )
                if requested_optimizer_x is not None:
                    request_key = self._pending_trust_clip_key(requested_optimizer_x)
                    self._trust_clip_by_requested_key[request_key] = {
                        "classification": trust_clip_classification,
                        "diagnostics": trust_clip_diagnostics,
                        "entry": gradient_entry,
                        "result": result,
                    }
                self._write_trust_clip_summary(
                    paths,
                    trust_clip_classification,
                    trust_clip_diagnostics,
                    trust_clip_action,
                )

            result.update(
                {
                    "status": status,
                    "gnorm_raw": gradient_entry["gnorm_raw"],
                    "gnorm_opt": gradient_entry["gnorm_opt"],
                    "gradient_guard_info": guard_info,
                    "trust_clip_class": trust_clip_classification,
                    "trust_clip_diagnostics": trust_clip_diagnostics,
                    "trust_clip_action": trust_clip_action,
                }
            )
            self._write_gradient_guard_summary(
                paths,
                gradient_entry,
                guard_info,
                "safe",
            )
            if cache_as_safe:
                self._cache[key] = result
            self._print_iteration_row(result, line_search_info=line_search_info)
            self._append_history_record(
                eval_id,
                objective,
                coefficients,
                gradient,
                status,
                line_search_info=line_search_info,
                eval_dir=paths.eval_dir,
                eval_index=eval_index,
                gradient_guard_info=guard_info,
                gradient_entry=gradient_entry,
                trust_clip_classification=trust_clip_classification,
                trust_clip_diagnostics=trust_clip_diagnostics,
                trust_clip_action=trust_clip_action,
            )
            return result
        except GradientGuardStop:
            raise
        except Exception as exc:
            self._append_history_record(
                eval_id,
                objective,
                coefficients,
                gradient,
                "failed",
                line_search_info=line_search_info,
                eval_dir=paths.eval_dir,
                eval_index=eval_index,
            )
            if isinstance(exc, BSplineSU2DriverError):
                raise
            raise BSplineSU2DriverError(
                f"evaluation {eval_id} failed in {paths.eval_dir}: {exc}"
            ) from exc

    def write_optimized_modes(self, coefficients):
        optimized_spec = update_mode_coefficients(self.mode_spec, coefficients)
        write_mode_spec(optimized_spec, self.optimized_modes_filename)
        return optimized_spec

    def _print_slsqp_parameters(self, maxiter, optimizer_bounds=None):
        if not self.print_optimizer_table:
            return
        optimizer_bounds = list(self.optimizer_bounds() if optimizer_bounds is None else optimizer_bounds)
        initial_optimizer_variables = self.physical_to_optimizer(
            self.initial_reduced_coefficients
        )
        print("Sequential Least SQuares Programming (SLSQP) parameters:")
        print(f"Number of active modes: {len(self.mode_ids)}")
        print(f"Number of design variables: {len(self.reduced_variable_ids)}")
        print(f"Symmetry coupling: {self.symmetry_coupling}")
        print(f"[PROGRESSIVE_BSPLINE][SURFACE] mode = {self.surface_mode}")
        print(
            "[PROGRESSIVE_BSPLINE][SURFACE] active sides = "
            f"{active_sides_from_surface_mode(self.surface_mode)}"
        )
        print(f"[PROGRESSIVE_BSPLINE][SURFACE] ndv = {len(self.reduced_variable_ids)}")
        print(
            "[PROGRESSIVE_BSPLINE][SURFACE] deformation direction = "
            f"{self.deformation_direction_mode}"
        )
        print(f"Eval layout: {self.eval_layout}")
        print(f"Sensitivity weighting: {self.sensitivity_weighting}")
        print(
            "Raw-gradient guard: {} factor={} window={} min_history={} floor={}".format(
                "ON" if self.gradient_guard_enabled else "OFF",
                self.gradient_guard_factor,
                self.gradient_guard_window,
                self.gradient_guard_min_history,
                self.gradient_guard_floor,
            )
        )
        print(
            "Trust-clip policy: {} legacy_beta_min={} severe_beta={} "
            "bad_patience={}/{}".format(
                self.trust_clip_options["policy"],
                self.trust_clip_options["legacy_beta_min"],
                self.trust_clip_options["severe_beta"],
                self.trust_clip_options["bad_patience"],
                self.trust_clip_options["bad_window"],
            )
        )
        print(
            "Objective function scaling factor: [{:.15g}]".format(
                float(self.opt_gradient_factor)
            )
        )
        print(
            "Variable scaling: physical coefficient = optimizer variable * {:.15g}".format(
                float(self.opt_relax_factor)
            )
        )
        print(f"Maximum number of iterations: {int(maxiter)}")
        accuracy = 1.0e-10 if self.opt_accuracy is None else self.opt_accuracy
        print("Requested accuracy: {:.15g}".format(float(accuracy)))
        print(
            "Initial physical coefficients: "
            + _vector_summary(self.initial_coefficients)
        )
        print("Physical coefficient bounds: " + _bounds_summary(self.bounds))
        print(f"Local step limiter: {'ON' if self.local_step_limit else 'OFF'}")
        if self.local_step_limit:
            print("Local step limit ratio: {:.15g}".format(float(self.local_step_limit_ratio)))
        print(
            "Initial SLSQP variables: "
            + _vector_summary(initial_optimizer_variables)
        )
        print("SLSQP variable bounds: " + _bounds_summary(optimizer_bounds))
        if not _bounds_are_uniform(self.bounds) or not _bounds_are_uniform(optimizer_bounds):
            arrays_file = self.workdir / "slsqp_parameter_arrays.json"
            with open(arrays_file, "w") as fp:
                json.dump(
                    {
                        "initial_physical_coefficients": list(self.initial_coefficients),
                        "physical_bounds": _bounds_to_list(self.bounds),
                        "initial_slsqp_variables": list(initial_optimizer_variables),
                        "slsqp_bounds": _bounds_to_list(optimizer_bounds),
                    },
                    fp,
                    indent=2,
                    sort_keys=True,
                )
                fp.write("\n")
            print(f"Full nonuniform SLSQP parameter arrays: {arrays_file}")
        print(
            "Note: EVAL_ID is the eval_XXXX directory suffix and may not start from zero if the workdir is reused. "
            "FC is the current-run CFD evaluation counter."
        )
        print(
            "Note: SLSQP_IT is the latest accepted SLSQP major iteration known at evaluation time; "
            "evaluations printed before a callback use the previous accepted iteration."
        )
        print("")

    def _print_iteration_row(self, result, line_search_info=None):
        if not self.print_optimizer_table:
            return

        if not self._printed_iteration_header:
            print(
                "SLSQP_IT   FC   EVAL_ID      OBJFUN_PHYS     OBJFUN_SLSQP      "
                "GNORM_RAW        GNORM_OPT          LS_BETA LS_BETA_LOCAL"
            )
            self._printed_iteration_header = True

        gradient = result.get("gradient") or []
        gnorm_raw = math.sqrt(
            sum(float(value) * float(value) for value in gradient)
        )
        reduced_gradient = self.collapse_gradient_to_reduced(gradient) if gradient else []

        info = line_search_info or {}
        beta = float(info.get("line_search_beta", 1.0))
        local_beta = float(info.get("local_step_beta", 1.0))
        if int(info.get("local_step_limited", 0)):
            print(
                "[BSPLINE_SU2_DRIVER] Local step limiter: beta={:.6e}, limiting_mode={}, da={:.6e}, limit={:.6e}".format(
                    local_beta,
                    info.get("local_step_limiting_mode", ""),
                    float(info.get("local_step_da", 0.0)),
                    float(info.get("local_step_limit", 0.0)),
                )
            )
        if beta < 1.0:
            print(
                "[BSPLINE_SU2_DRIVER] WARNING: line_search_beta/local_step_beta < 1; "
                "the gradient passed to SLSQP is approximate because the evaluated design is clipped."
            )

        obj_phys = float(result["objective"])
        obj_slsqp = obj_phys * float(self.opt_gradient_factor)

        gnorm_opt = (
            math.sqrt(sum(float(value) * float(value) for value in reduced_gradient))
            * float(self.opt_relax_factor)
            * float(self.opt_gradient_factor)
            * beta
        )

        slsqp_iter = int(self._slsqp_major_iter)
        fc = int(result.get("eval_index", self._run_eval_count))
        eval_id = int(result.get("eval_id", -1))

        print(
            "{:8d} {:4d} {:9d} {:16.6E} {:16.6E} {:16.6E} {:16.6E} {:12.4E} {:13.4E}".format(
                slsqp_iter,
                fc,
                eval_id,
                obj_phys,
                obj_slsqp,
                gnorm_raw,
                gnorm_opt,
                beta,
                local_beta,
            )
        )

    def _apply_trigger_resume_state(self, trigger_resume_state=None):
        state = dict(trigger_resume_state or {})
        self.trigger_project.trigger_history = list(state.get("trigger_history", []))
        self.trigger_project.trigger_state = state.get("trigger_state", None)
        self.trigger_project.refinement_triggered = bool(
            state.get("refinement_triggered", False)
        )

    def _trigger_resume_state(self):
        return {
            "trigger_history": list(self.trigger_project.trigger_history),
            "trigger_state": self.trigger_project.trigger_state,
            "refinement_triggered": bool(self.trigger_project.refinement_triggered),
        }

    def _attach_trigger_state(self, result):
        result["trigger_history"] = list(self.trigger_project.trigger_history)
        result["trigger_state"] = self.trigger_project.trigger_state
        result["trigger_history_len"] = len(self.trigger_project.trigger_history)
        result["refinement_triggered"] = bool(self.trigger_project.refinement_triggered)
        return result

    def _controlled_gradient_guard_result(self, stop, optimizer="SLSQP"):
        safe = stop.last_safe_entry
        self.restore_modes_from_entry(safe)
        if safe is None:
            coefficients = list(self.initial_coefficients)
            objective = math.inf
            safe_eval_id = None
        else:
            coefficients = [float(value) for value in safe["evaluated_x"]]
            objective = float(safe["objective"])
            safe_eval_id = int(safe["eval_id"])
        refine = self.gradient_guard_next_action == "refine"
        if refine:
            self.trigger_project.refinement_triggered = True
        if self.print_optimizer_table:
            print("Raw-gradient guard stop    (controlled rollback)")
            print(f"            Restored evaluation: {safe_eval_id}")
            print(f"            Current function value: {objective:.12g}")
        return self._attach_trigger_state({
            "optimizer": optimizer,
            "success": True,
            "message": "Raw-gradient guard stop: restored last safe evaluation",
            "objective": objective,
            "coefficients": coefficients,
            "status": "gradient_guard_stop",
            "gradient_guard_triggered": True,
            "gradient_guard_info": dict(stop.guard_info),
            "gradient_guard_bad_eval_id": stop.bad_entry.get("eval_id"),
            "gradient_guard_restore_eval_id": safe_eval_id,
            "gradient_guard_next_action": self.gradient_guard_next_action,
            "early_refine_triggered": refine,
            "refinement_triggered": refine,
        })

    def _controlled_trust_clip_result(self, stop, optimizer="SLSQP"):
        rollback = stop.rollback_entry
        self.restore_modes_from_entry(rollback)
        if rollback is None:
            coefficients = list(self.initial_coefficients)
            objective = math.inf
            restore_eval_id = None
        else:
            coefficients = [float(value) for value in rollback["evaluated_x"]]
            objective = float(rollback["objective"])
            restore_eval_id = int(rollback["eval_id"])
        next_action = (
            "refine" if stop.action == "refine" else "restart_same_level"
        )
        refine = next_action == "refine"
        if refine:
            self.trigger_project.refinement_triggered = True
        return self._attach_trigger_state({
            "optimizer": optimizer,
            "success": True,
            "message": f"Trust-clip controlled stop: {stop.classification}",
            "objective": objective,
            "coefficients": coefficients,
            "status": "trust_clip_stop",
            "trust_clip_triggered": True,
            "trust_clip_class": stop.classification,
            "trust_clip_diagnostics": dict(stop.diagnostics),
            "trust_clip_action": stop.action,
            "trust_clip_bad_eval_id": stop.entry.get("eval_id"),
            "trust_clip_restore_eval_id": restore_eval_id,
            "trust_clip_next_action": next_action,
            "early_refine_triggered": refine,
            "refinement_triggered": refine,
        })

    def optimize(
        self,
        maxiter=5,
        fallback_step=0.1,
        gradient_tol=1.0e-8,
        trigger_resume_state=None,
    ):
        self._apply_trigger_resume_state(trigger_resume_state)
        self.configure_geometry_aware_bounds()
        self._configure_line_search_bound()
        self.configure_thickness_constraint()
        try:
            from scipy.optimize import minimize
        except Exception:
            if self.thickness_constraint is not None:
                raise BSplineSU2DriverError(
                    "SciPy is required when PROGRESSIVE_THICKNESS_CONSTRAINT=YES"
                )
            try:
                return self._attach_trigger_state(self._optimize_projected_gradient_descent(
                    maxiter=maxiter,
                    step_size=fallback_step,
                    gradient_tol=gradient_tol,
                ))
            except GradientGuardStop as stop:
                return self._controlled_gradient_guard_result(
                    stop,
                    optimizer="projected_gradient_descent",
                )

        x0 = self.physical_to_optimizer(self.initial_reduced_coefficients)
        bounds_u = self.optimizer_bounds()
        constraints = self._thickness_constraint_functions()
        self._print_slsqp_parameters(maxiter, optimizer_bounds=bounds_u)

        def fun(x):
            result, info = self._evaluate_optimizer_variables(list(x))
            if result.get("trust_clip_class") not in (
                "weak_clipped_progress",
                "accepted_clipped_restart",
                "rejected_toxic_clip",
            ) and not info.get("cache_hit", False):
                record_objective_and_check(
                    self.trigger_project,
                    float(result["objective"]),
                )
            return float(result["objective"]) * self.opt_gradient_factor

        def jac(x):
            result, info = self._evaluate_optimizer_variables(list(x))
            beta = float(info.get("line_search_beta", 1.0))
            reduced_gradient = self.collapse_gradient_to_reduced(result["gradient"])
            return [
                float(value) * self.opt_relax_factor * self.opt_gradient_factor * beta
                for value in reduced_gradient
            ]

        def callback(x):
            self._slsqp_major_iter += 1
            self._trust_clip_callback(list(x))

        options = {
            "maxiter": int(maxiter),
            "disp": False,
        }
        if self.opt_accuracy is not None:
            options["ftol"] = float(self.opt_accuracy) * self.opt_gradient_factor

        early_refine_triggered = False
        result = None
        try:
            result = minimize(
                fun,
                x0,
                jac=jac,
                bounds=bounds_u,
                constraints=constraints,
                method="SLSQP",
                callback=callback,
                options=options,
            )
        except TrustClipStop as stop:
            return self._controlled_trust_clip_result(stop, optimizer="SLSQP")
        except GradientGuardStop as stop:
            return self._controlled_gradient_guard_result(stop, optimizer="SLSQP")
        except RefinementTriggered:
            early_refine_triggered = True
            print(
                f"[{trigger_prefix(self.trigger_project)}] "
                "Optimization stopped early due to refinement trigger"
            )
        best_record = self._best_ok_history_record()
        if best_record is not None:
            final_coefficients = [
                float(best_record.get(f"coeff__{mode_id}", value))
                for mode_id, value in zip(
                    self.mode_ids,
                    self.initial_coefficients
                    if result is None
                    else self.expand_reduced_physical(self.optimizer_to_physical(result.x)),
                )
            ]
            final_objective = float(best_record["objective"])
        else:
            if result is not None:
                final_coefficients = self.expand_reduced_physical(
                    self.optimizer_to_physical(result.x)
                )
                final_objective = float(result.fun) / self.opt_gradient_factor
            else:
                final_coefficients = list(self.initial_coefficients)
                final_objective = math.inf
        self.write_optimized_modes(final_coefficients)
        if self.print_optimizer_table:
            if early_refine_triggered:
                print("Early refinement trigger    (Exit mode early_refine_trigger)")
            else:
                print("{}    (Exit mode {})".format(str(result.message), int(result.status)))
            print("            Current function value: {:.12g}".format(final_objective))
            print("            Iterations: {}".format(int(getattr(result, "nit", self._slsqp_major_iter))))
            print("            Function evaluations: {}".format(int(getattr(result, "nfev", len(self._history_records)))))
            print("            Gradient evaluations: {}".format(int(getattr(result, "njev", len(self._history_records)))))
        return self._attach_trigger_state({
            "optimizer": "SLSQP",
            "success": True if early_refine_triggered else bool(result.success),
            "message": (
                "Early refinement trigger"
                if early_refine_triggered
                else str(result.message)
            ),
            "objective": final_objective,
            "coefficients": final_coefficients,
            "status": (
                "early_refine_trigger"
                if early_refine_triggered
                else "ok"
            ),
            "early_refine_triggered": bool(early_refine_triggered),
            "refinement_triggered": bool(
                getattr(self.trigger_project, "refinement_triggered", False)
            ),
        })

    def _optimize_projected_gradient_descent(self, maxiter, step_size, gradient_tol):
        self.configure_geometry_aware_bounds()
        self._configure_line_search_bound()
        x = _project_to_bounds(self.initial_reduced_coefficients, self.reduced_bounds)
        self._print_slsqp_parameters(maxiter)
        current, _line_search_info, x = self._evaluate_reduced_physical(x)
        self._line_search_anchor_physical = list(current["coefficients"])
        self._local_step_anchor_reduced = list(x)
        step = float(step_size)

        for _ in range(int(maxiter)):
            gradient = self.collapse_gradient_to_reduced(current["gradient"])
            gradient_norm = math.sqrt(sum(value * value for value in gradient))
            if gradient_norm <= gradient_tol:
                break

            accepted = False
            trial_step = step
            for _inner in range(12):
                trial = _project_to_bounds(
                    [value - trial_step * grad for value, grad in zip(x, gradient)],
                    self.reduced_bounds,
                )
                if trial == x:
                    trial_step *= 0.5
                    continue
                trial_result, _line_search_info, trial_reduced_eval = self._evaluate_reduced_physical(trial)
                if trial_result["objective"] <= current["objective"]:
                    x = trial_reduced_eval
                    current = trial_result
                    self._line_search_anchor_physical = list(trial_result["coefficients"])
                    self._local_step_anchor_reduced = list(x)
                    step = min(trial_step * 1.25, 1.0)
                    accepted = True
                    break
                trial_step *= 0.5

            if not accepted:
                break

        best_record = self._best_ok_history_record()
        if best_record is not None:
            final_coefficients = [
                float(best_record.get(f"coeff__{mode_id}", value))
                for mode_id, value in zip(self.mode_ids, self.expand_reduced_physical(x))
            ]
            current = {
                "objective": float(best_record["objective"]),
                "gradient": best_record.get("gradient", current.get("gradient")),
            }
        else:
            final_coefficients = self.expand_reduced_physical(x)
        self.write_optimized_modes(final_coefficients)
        if self.print_optimizer_table:
            print("Optimization terminated successfully    (projected gradient fallback)")
            print("            Current function value: {:.12g}".format(float(current["objective"])))
            print("            Function evaluations: {}".format(len(self._history_records)))
            print("            Gradient evaluations: {}".format(len(self._history_records)))
        return {
            "optimizer": "projected_gradient_descent",
            "success": True,
            "message": "SciPy unavailable; used projected gradient descent fallback",
            "objective": current["objective"],
            "coefficients": final_coefficients,
        }


def run_bspline_su2_optimization(
    modes_filename,
    base_mesh,
    marker,
    def_template,
    primal_template,
    adjoint_template,
    workdir,
    objective_column="CD",
    maxiter=5,
    mpi_prefix="",
    default_bounds=DEFAULT_BOUNDS,
    cache_tol=1.0e-12,
    fallback_step=0.1,
    show_commands=False,
    stream_solver_output=False,
    print_optimizer_table=True,
    auto_scale_bounds_to_geometry=False,
    max_normal_displacement=None,
    max_rms_normal_displacement=None,
    min_bound_scale=0.0,
    opt_accuracy=None,
    opt_bound_upper=None,
    opt_bound_lower=None,
    opt_relax_factor=1.0,
    opt_gradient_factor=1.0,
    gradient_guard=True,
    gradient_guard_factor=100.0,
    gradient_guard_window=5,
    gradient_guard_min_history=3,
    gradient_guard_floor=1.0e-14,
    gradient_guard_next_action="restart_same_level",
    refinement_available=None,
    trust_clip_policy="OFF",
    trust_clip_beta_tol=1.0e-12,
    trust_clip_legacy_beta_min=0.50,
    trust_clip_severe_beta=0.50,
    trust_clip_worsening_tol=0.05,
    trust_clip_soft_gnorm_factor=20.0,
    trust_clip_bad_patience=2,
    trust_clip_bad_window=5,
    trust_clip_stag_tol=1.0e-6,
    opt_line_search_bound=None,
    thickness_options=None,
    eval_layout="DSN",
    objective_adjoint="drag",
    symmetry_coupling="NONE",
    surface_mode="BOTH",
    sensitivity_weighting="NODAL",
    local_step_limit=False,
    local_step_limit_ratio=200.0,
    trigger_opts=None,
    trigger_resume_state=None,
    progressive_label="PROGRESSIVE_BSPLINE",
    deformation_direction_mode=None,
    le_safe_direction=False,
    le_safe_x0=None,
    le_safe_x1=None,
    le_safe_power=None,
):
    """Run the fixed active-mode B-spline optimization and return its result."""

    driver = BSplineSU2Driver(
        modes_filename=modes_filename,
        base_mesh=base_mesh,
        marker=marker,
        def_template=def_template,
        primal_template=primal_template,
        adjoint_template=adjoint_template,
        workdir=workdir,
        objective_column=objective_column,
        mpi_prefix=mpi_prefix,
        default_bounds=default_bounds,
        cache_tol=cache_tol,
        show_commands=show_commands,
        stream_solver_output=stream_solver_output,
        print_optimizer_table=print_optimizer_table,
        auto_scale_bounds_to_geometry=auto_scale_bounds_to_geometry,
        max_normal_displacement=max_normal_displacement,
        max_rms_normal_displacement=max_rms_normal_displacement,
        min_bound_scale=min_bound_scale,
        opt_accuracy=opt_accuracy,
        opt_bound_upper=opt_bound_upper,
        opt_bound_lower=opt_bound_lower,
        opt_relax_factor=opt_relax_factor,
        opt_gradient_factor=opt_gradient_factor,
        gradient_guard=gradient_guard,
        gradient_guard_factor=gradient_guard_factor,
        gradient_guard_window=gradient_guard_window,
        gradient_guard_min_history=gradient_guard_min_history,
        gradient_guard_floor=gradient_guard_floor,
        gradient_guard_next_action=gradient_guard_next_action,
        refinement_available=refinement_available,
        trust_clip_policy=trust_clip_policy,
        trust_clip_beta_tol=trust_clip_beta_tol,
        trust_clip_legacy_beta_min=trust_clip_legacy_beta_min,
        trust_clip_severe_beta=trust_clip_severe_beta,
        trust_clip_worsening_tol=trust_clip_worsening_tol,
        trust_clip_soft_gnorm_factor=trust_clip_soft_gnorm_factor,
        trust_clip_bad_patience=trust_clip_bad_patience,
        trust_clip_bad_window=trust_clip_bad_window,
        trust_clip_stag_tol=trust_clip_stag_tol,
        opt_line_search_bound=opt_line_search_bound,
        thickness_options=thickness_options,
        eval_layout=eval_layout,
        objective_adjoint=objective_adjoint,
        symmetry_coupling=symmetry_coupling,
        surface_mode=surface_mode,
        sensitivity_weighting=sensitivity_weighting,
        local_step_limit=local_step_limit,
        local_step_limit_ratio=local_step_limit_ratio,
        trigger_opts=trigger_opts,
        progressive_label=progressive_label,
        deformation_direction_mode=deformation_direction_mode,
        le_safe_direction=le_safe_direction,
        le_safe_x0=le_safe_x0,
        le_safe_x1=le_safe_x1,
        le_safe_power=le_safe_power,
    )
    result = driver.optimize(
        maxiter=maxiter,
        fallback_step=fallback_step,
        trigger_resume_state=trigger_resume_state,
    )
    result["optimization_history"] = str(driver.optimization_history_filename)
    result["optimized_modes"] = str(driver.optimized_modes_filename)
    result["workdir"] = str(driver.workdir)
    return result


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Optimize fixed active external B-spline modes with SU2."
    )
    parser.add_argument("--modes", required=True, help="Input bspline_modes.json")
    parser.add_argument("--base-mesh", required=True, help="Undeformed SU2 mesh")
    parser.add_argument("--marker", required=True, help="Surface marker name")
    parser.add_argument("--def-template", required=True, help="SU2_DEF config template")
    parser.add_argument("--primal-template", required=True, help="SU2_CFD config template")
    parser.add_argument("--adjoint-template", required=True, help="SU2_CFD_AD config template")
    parser.add_argument("--workdir", required=True, help="Optimization work directory")
    parser.add_argument(
        "--optimizer-config",
        default=None,
        help="Optional SU2-style external optimizer config file",
    )
    parser.add_argument(
        "--objective-column",
        default="CD",
        help="Primal history CSV column to minimize",
    )
    parser.add_argument("--maxiter", type=int, default=5, help="Optimizer iteration limit")
    parser.add_argument("--mpi", default="", help="Optional MPI prefix, e.g. 'mpirun -n 6'")
    parser.add_argument(
        "--default-lower-bound",
        type=float,
        default=DEFAULT_BOUNDS[0],
        help="Lower bound for modes missing a bounds field",
    )
    parser.add_argument(
        "--default-upper-bound",
        type=float,
        default=DEFAULT_BOUNDS[1],
        help="Upper bound for modes missing a bounds field",
    )
    parser.add_argument(
        "--cache-tol",
        type=float,
        default=1.0e-12,
        help="Coefficient rounding tolerance for evaluation caching",
    )
    parser.add_argument(
        "--fallback-step",
        type=float,
        default=0.1,
        help="Projected-gradient fallback step size when SciPy is unavailable",
    )
    parser.add_argument(
        "--show-commands",
        action="store_true",
        default=False,
        help="Print low-level SU2 commands before running them",
    )
    parser.add_argument(
        "--quiet-driver",
        dest="show_commands",
        action="store_false",
        help="Do not print low-level SU2 commands; logs are still written",
    )
    parser.add_argument(
        "--stream-solver-output",
        action="store_true",
        help="Stream solver stdout/stderr to console as well as log files",
    )
    parser.add_argument(
        "--no-optimizer-table",
        dest="print_optimizer_table",
        action="store_false",
        help="Suppress the compact optimizer table",
    )
    parser.set_defaults(print_optimizer_table=True)
    parser.add_argument(
        "--auto-scale-bounds-to-geometry",
        action="store_true",
        default=False,
        help="Automatically rescale coefficient bounds to satisfy geometry limits",
    )
    parser.add_argument(
        "--max-normal-displacement",
        type=float,
        default=None,
        help="Maximum allowed normal displacement for automatic bound scaling",
    )
    parser.add_argument(
        "--max-rms-normal-displacement",
        type=float,
        default=None,
        help="Maximum allowed RMS normal displacement for automatic bound scaling",
    )
    parser.add_argument(
        "--min-bound-scale",
        type=float,
        default=0.0,
        help="Minimum admissible geometry-safe bound scale",
    )
    parser.add_argument(
        "--opt-bound-upper",
        type=float,
        default=None,
        help="SU2-style physical upper bound applied to every active B-spline coefficient",
    )
    parser.add_argument(
        "--opt-bound-lower",
        type=float,
        default=None,
        help="SU2-style physical lower bound applied to every active B-spline coefficient",
    )
    parser.add_argument(
        "--opt-relax-factor",
        type=float,
        default=1.0,
        help="SU2-style variable scaling: physical coefficient = SLSQP variable * factor",
    )
    parser.add_argument(
        "--opt-gradient-factor",
        type=float,
        default=1.0,
        help="SU2-style objective/gradient scaling factor for SLSQP",
    )
    parser.add_argument(
        "--gradient-guard",
        dest="gradient_guard",
        action="store_true",
        help="Enable the raw-gradient explosion guard (default)",
    )
    parser.add_argument(
        "--no-gradient-guard",
        dest="gradient_guard",
        action="store_false",
        help="Disable the raw-gradient explosion guard",
    )
    parser.set_defaults(gradient_guard=True)
    parser.add_argument("--gradient-guard-factor", type=float, default=100.0)
    parser.add_argument("--gradient-guard-window", type=int, default=5)
    parser.add_argument("--gradient-guard-min-history", type=int, default=3)
    parser.add_argument("--gradient-guard-floor", type=float, default=1.0e-14)
    parser.add_argument(
        "--trust-clip-policy",
        default="OFF",
        choices=ALLOWED_TRUST_CLIP_POLICIES,
    )
    parser.add_argument("--trust-clip-beta-tol", type=float, default=1.0e-12)
    parser.add_argument("--trust-clip-legacy-beta-min", type=float, default=0.50)
    parser.add_argument("--trust-clip-severe-beta", type=float, default=0.50)
    parser.add_argument("--trust-clip-worsening-tol", type=float, default=0.05)
    parser.add_argument("--trust-clip-soft-gnorm-factor", type=float, default=20.0)
    parser.add_argument("--trust-clip-bad-patience", type=int, default=2)
    parser.add_argument("--trust-clip-bad-window", type=int, default=5)
    parser.add_argument("--trust-clip-stag-tol", type=float, default=1.0e-6)
    parser.add_argument(
        "--opt-line-search-bound",
        type=float,
        default=None,
        help="Maximum accepted physical normal-displacement jump per SLSQP iteration",
    )
    parser.add_argument(
        "--eval-layout",
        default="DSN",
        choices=ALLOWED_EVAL_LAYOUTS,
        help="Evaluation directory layout",
    )
    parser.add_argument(
        "--objective-adjoint",
        default="drag",
        help="Objective adjoint folder suffix for DSN layout",
    )
    parser.add_argument(
        "--symmetry-coupling",
        default="NONE",
        choices=ALLOWED_SYMMETRY_COUPLINGS,
        help="Optional upper/lower B-spline coefficient coupling",
    )
    parser.add_argument(
        "--surface-mode",
        default="BOTH",
        choices=ALLOWED_SURFACE_MODES,
        help="Optimize both airfoil surfaces or one half-domain surface",
    )
    parser.add_argument(
        "--sensitivity-weighting",
        default="NODAL",
        choices=("NODAL", "DENSITY"),
        help="Treat SU2 surface sensitivities as nodal values or densities requiring arc-length weights",
    )
    parser.add_argument(
        "--deformation-direction",
        dest="deformation_direction_mode",
        default=None,
        choices=ALLOWED_DEFORMATION_DIRECTION_MODES,
        help="Direction used to apply the scalar B-spline deformation",
    )
    parser.add_argument(
        "--le-safe-direction",
        action="store_true",
        default=False,
        help="Legacy alias selecting LE_SAFE when --deformation-direction is omitted",
    )
    parser.add_argument("--le-safe-x0", type=float, default=LE_SAFE_DEFAULT_X0)
    parser.add_argument("--le-safe-x1", type=float, default=LE_SAFE_DEFAULT_X1)
    parser.add_argument("--le-safe-power", type=float, default=LE_SAFE_DEFAULT_POWER)
    parser.add_argument(
        "--local-step-limit",
        action="store_true",
        default=False,
        help="Limit accepted physical coefficient jumps by support length / ratio",
    )
    parser.add_argument(
        "--local-step-limit-ratio",
        type=float,
        default=200.0,
        help="Denominator for local coefficient step limits",
    )
    return parser


def main(argv=None):
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    args = apply_optimizer_config_to_args(
        args,
        parser,
        argv,
        fixed_driver_options_from_config,
        warning_prefix="[BSPLINE_SU2_DRIVER]",
    )

    try:
        result = run_bspline_su2_optimization(
            modes_filename=args.modes,
            base_mesh=args.base_mesh,
            marker=args.marker,
            def_template=args.def_template,
            primal_template=args.primal_template,
            adjoint_template=args.adjoint_template,
            workdir=args.workdir,
            objective_column=args.objective_column,
            mpi_prefix=args.mpi,
            default_bounds=(args.default_lower_bound, args.default_upper_bound),
            cache_tol=args.cache_tol,
            maxiter=args.maxiter,
            fallback_step=args.fallback_step,
            show_commands=args.show_commands,
            stream_solver_output=args.stream_solver_output,
            print_optimizer_table=args.print_optimizer_table,
            auto_scale_bounds_to_geometry=args.auto_scale_bounds_to_geometry,
            max_normal_displacement=args.max_normal_displacement,
            max_rms_normal_displacement=args.max_rms_normal_displacement,
            min_bound_scale=args.min_bound_scale,
            opt_accuracy=getattr(args, "opt_accuracy", None),
            opt_bound_upper=args.opt_bound_upper,
            opt_bound_lower=args.opt_bound_lower,
            opt_relax_factor=args.opt_relax_factor,
            opt_gradient_factor=args.opt_gradient_factor,
            gradient_guard=args.gradient_guard,
            gradient_guard_factor=args.gradient_guard_factor,
            gradient_guard_window=args.gradient_guard_window,
            gradient_guard_min_history=args.gradient_guard_min_history,
            gradient_guard_floor=args.gradient_guard_floor,
            trust_clip_policy=args.trust_clip_policy,
            trust_clip_beta_tol=args.trust_clip_beta_tol,
            trust_clip_legacy_beta_min=args.trust_clip_legacy_beta_min,
            trust_clip_severe_beta=args.trust_clip_severe_beta,
            trust_clip_worsening_tol=args.trust_clip_worsening_tol,
            trust_clip_soft_gnorm_factor=args.trust_clip_soft_gnorm_factor,
            trust_clip_bad_patience=args.trust_clip_bad_patience,
            trust_clip_bad_window=args.trust_clip_bad_window,
            trust_clip_stag_tol=args.trust_clip_stag_tol,
            opt_line_search_bound=args.opt_line_search_bound,
            thickness_options=getattr(args, "thickness_options", None),
            eval_layout=args.eval_layout,
            objective_adjoint=getattr(args, "objective_adjoint", "drag"),
            symmetry_coupling=args.symmetry_coupling,
            surface_mode=args.surface_mode,
            sensitivity_weighting=args.sensitivity_weighting,
            deformation_direction_mode=args.deformation_direction_mode,
            le_safe_direction=args.le_safe_direction,
            le_safe_x0=args.le_safe_x0,
            le_safe_x1=args.le_safe_x1,
            le_safe_power=args.le_safe_power,
            local_step_limit=args.local_step_limit,
            local_step_limit_ratio=args.local_step_limit_ratio,
        )
    except (BSplineSU2DriverError, BSplineModeError, OSError, ValueError) as exc:
        parser.error(str(exc))

    print("Optimizer: {}".format(result["optimizer"]))
    print("Success: {}".format(result["success"]))
    print("Objective: {:.15g}".format(float(result["objective"])))
    print("Wrote {}".format(result["optimization_history"]))
    print("Wrote {}".format(result["optimized_modes"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
