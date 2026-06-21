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



from SU2.opt.bspline_driver.commands import (
    EvalPaths,
    _append_command_log,
    _command_list,
    _normalize_eval_layout,
    _normalize_objective_adjoint,
    _relative_path,
    _subprocess_env,
    _symlink_or_copy,
    _tail_file,
    _with_mpi,
    build_eval_commands,
    build_eval_paths,
    command_to_string,
    create_eval_aliases,
    ensure_adjoint_solution_input,
    run_command,
)
from SU2.opt.bspline_driver.config_apply import (
    _explicit_cli_dests,
    _objective_adjoint_from_config,
    _objective_column_from_config,
    apply_optimizer_config_to_args,
    fixed_driver_options_from_config,
    resolve_thickness_domain_mode,
    thickness_options_from_config,
)
from SU2.opt.bspline_driver.config_keys import (
    SUPPORTED_OPT_CONFIG_KEYS,
    UNSUPPORTED_OPT_CONFIG_KEYS,
)
from SU2.opt.bspline_driver.config_parse import (
    _ConfigDict,
    _format_config_atom,
    _format_config_value,
    _line_config_key,
    _parse_optimizer_config_value,
    parse_optimizer_config,
    patch_config_template,
)
from SU2.opt.bspline_driver.constants import (
    ALLOWED_EVAL_LAYOUTS,
    ALLOWED_SYMMETRY_COUPLINGS,
    DEFAULT_BOUNDS,
)
from SU2.opt.bspline_driver.errors import (
    BSplineSU2DriverError,
    GradientGuardStop,
    TrustClipStop,
    _as_float,
    _normalized_name,
)
from SU2.opt.bspline_driver.geometry_bounds import (
    _bounds_are_uniform,
    _bounds_summary,
    _bounds_to_list,
    _max_radius_from_bounds,
    _vector_summary,
    compute_geometry_aware_bound_scaling,
)
from SU2.opt.bspline_driver.guards import (
    ALLOWED_TRUST_CLIP_POLICIES,
    LAST_EVAL_CACHE_BLOCKED_TRUST_CLIP_CLASSES,
    SAFE_EVALUATION_STATUSES,
    _trust_clip_options,
    classify_clipped_trial,
    gradient_guard_triggered,
)
from SU2.opt.bspline_driver.reduction import (
    ReducedVariable,
    _active_modes,
    _mode_pairing_key,
    _mode_support_key,
    _safe_identifier,
    _symmetry_group_id,
    _validated_bounds,
    active_bounds,
    active_coefficient_vector,
    active_mode_ids,
    build_reduced_variables,
    cache_key,
    collapse_full_gradient,
    collapse_full_jacobian,
    compress_full_coefficients,
    expand_reduced_coefficients,
    mode_support_length,
    reduced_bounds_from_full_bounds,
    reduced_step_limits_from_modes,
    update_mode_coefficients,
    write_mode_spec,
)
from SU2.opt.bspline_driver.tables import (
    _find_column_index,
    _find_field,
    _read_table,
    read_bspline_gradients,
    read_gradient_vector,
    read_objective_from_history,
)


























































































from SU2.opt.bspline_driver.thickness import BSplineThicknessConstraint
from SU2.opt.bspline_driver.driver import (
    BSplineSU2Driver,
    _project_to_bounds,
    run_bspline_su2_optimization,
)


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
