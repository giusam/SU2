#!/usr/bin/env python

"""Progressive/adaptive external B-spline optimization driver for SU2."""

import argparse
import csv
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from SU2.opt.bspline_dot import (
    match_sensitivities_to_metadata,
    normalize_sensitivity_weighting,
    read_metadata,
    read_sensitivity_file,
)
from SU2.opt.bspline_modes import (
    ALLOWED_DEFORMATION_DIRECTION_MODES,
    ALLOWED_SURFACE_MODES,
    BSplineModeError,
    LE_SAFE_DEFAULT_POWER,
    LE_SAFE_DEFAULT_X0,
    LE_SAFE_DEFAULT_X1,
    active_sides_from_surface_mode,
    clamped_basis_count,
    evaluate_all_modes,
    load_mode_spec,
    mode_normalization_factor,
    normalize_deformation_direction_mode,
    normalize_surface_mode,
    validate_le_safe_direction_options,
    validate_mode_spec,
    validate_surface_mode_against_modes,
)
from SU2.opt.bspline_driver.config_apply import (
    apply_optimizer_config_to_args,
    fixed_driver_options_from_config,
    resolve_thickness_domain_mode,
)
from SU2.opt.bspline_driver.constants import (
    ALLOWED_EVAL_LAYOUTS,
    ALLOWED_SYMMETRY_COUPLINGS,
)
from SU2.opt.bspline_driver.commands import ALLOWED_SENSITIVITY_SOURCES
from SU2.opt.bspline_driver.errors import BSplineSU2DriverError
from SU2.opt.bspline_driver.reduction import (
    active_coefficient_vector,
    active_mode_ids,
    cache_key,
    write_mode_spec,
)
from SU2.opt.bspline_driver.tables import read_objective_from_history
from SU2.opt.bspline_su2_driver import (
    run_bspline_su2_optimization,
)
from SU2.opt.progressive_trigger import build_online_trigger_opts



from SU2.opt.bspline_adaptive.boehm import (
    _boehm_insert_once,
    _boehm_insert_to_target_knots,
    _boehm_transfer_side,
    _check_transferred_coefficients_within_bounds,
    _find_knot_span_for_insertion,
    _knot_multiplicity,
    _mode_normalization_factors,
    transfer_shape_to_inserted_space,
)
from SU2.opt.bspline_adaptive.errors import (
    BSplineAdaptiveError,
    _as_bool,
    _as_float,
    _as_float_list,
)
from SU2.opt.bspline_adaptive.history import (
    SAFE_OPTIMIZATION_STATUSES,
    _append_restart_history_rows,
    _compact_number,
    _csv_value,
    _cumulative_best,
    _level_summary_row,
    _log_trigger_decision,
    _read_optimization_history,
    _tail_count,
    _trigger_refine_from_history,
    _write_adaptive_history,
    _write_restart_history_rows,
    _write_selection_history,
    find_best_eval_dir,
    find_eval_dir_for_mode_coefficients,
    write_knot_span_scores_csv,
    write_selected_knot_refinement_json,
)
from SU2.opt.bspline_adaptive.knot_space import (
    REFINE_MODE,
    REFINE_STATE,
    _basis_for_spec_modes,
    _bounds_signature,
    _expanded_coefficients_from_reduced,
    _mode_template_for_side,
    _representative_bounds,
    _requested_knot_insertions,
    _rounded_knots,
    _side_order,
    coefficient_vector_for_space,
    extract_clamped_knot_space,
    insert_knot_midpoint,
    knot_insertion_spans,
    physical_ndv_for_knot_space,
    reduced_basis_matrix_for_space,
    reduced_ndv_for_knot_space,
    refinement_limit_ndv,
    regenerate_clamped_modes,
)
from SU2.opt.bspline_adaptive.mode_utils import (
    _active_modes,
    _copy_global_metadata,
    _mode_support,
    _mode_with_zero_coefficient,
    _spec_with_modes,
    build_level,
    build_next_active_modes,
    evaluate_basis_matrix,
    mode_sort_key,
    write_level_start,
)
from SU2.opt.bspline_adaptive.models import (
    BsplineLevel,
    ClampedKnotSpace,
    ClampedSideGroup,
    TriggerDecision,
)
from SU2.opt.bspline_adaptive.penalties import (
    ALLOWED_KNOT_BATCH_PENALTY_MODES,
    ALLOWED_KNOT_DEPTH_PENALTY_MODES,
    KNOT_DEPTH_PENALTY_MODE_CHOICES,
    _as_knot_depth,
    _knot_batch_depth,
    _knot_batch_penalty,
    _knot_depth_penalty_mode,
    _knot_depth_power_gamma,
    _knot_initial_span_depth,
    _knot_intra_batch_penalty,
    _normalize_knot_depth_penalty_mode,
    _span_contains,
    _span_key,
    _streuber_depth_penalty,
    _validated_knot_span_depths,
    apply_knot_batch_penalty,
    apply_knot_depth_penalty,
    get_or_initialize_knot_span_depths,
    initialize_knot_span_depths,
)
from SU2.opt.bspline_adaptive.refinement import build_next_knot_inserted_modes
from SU2.opt.bspline_adaptive.scoring import (
    _coeffs_are_well_conditioned,
    _rank_incremental_columns,
    _score_virtual_insertion,
    _tikhonov_projection,
    build_scalar_deformation_sensitivity,
    build_scalar_normal_sensitivity,
    load_adjoint_signal,
    project_onto_basis,
    residualize_candidate,
    score_knot_spans,
)
from SU2.opt.bspline_adaptive.settings import (
    ALLOWED_KNOT_SCORE_MODES,
    ALLOWED_NADD_MODES,
    ALLOWED_REALLOCATION_COUNT_MODES,
    ALLOWED_REALLOCATION_FREEZE_METRICS,
    ALLOWED_REFINE_MODES,
    ALLOWED_TRIGGERS,
    GLOBAL_MODE_KEYS,
    KNOT_SCORE_FIELDNAMES,
    REMOVED_CANDIDATE_CONFIG_KEYS,
    _cfg_value_list,
    _forced_template_lines,
    _format_cfg_value,
    _infer_marker_from_config,
    _mode_display,
    _print_level_start,
    _resolve_cfg_path,
    _settings_from_args,
    _strip_cfg_atom,
    _write_case_template,
    _write_cfg,
    adaptive_options_from_config,
    generate_initial_bspline_modes,
    generate_missing_templates,
    prepare_bspline_launch_settings,
    print_startup_summary,
    validate_adaptive_options,
)


# The adaptive B-spline optimizer is knot-insertion-only. These are fixed
# internally and are not user-configurable; candidate/generated refinement
# has been removed.
# NOTE: SLOPE_EFFICIENCY_FILTERED is accepted only for backward compatibility
# and is mapped internally to SLOPE_EFFICIENCY_TRIGGER.










































































































































































































from SU2.opt.bspline_adaptive.orchestration import (
    _trigger_resume_state_from_result,
    gradient_guard_next_action_for_level,
    progressive_bspline_su2_shape_optimization,
    run_with_gradient_guard_restarts,
)


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Progressive/adaptive external B-spline SU2 optimization."
    )
    parser.add_argument("-f", "--case-config", default=None)
    parser.add_argument("-n", "--nproc", type=int, default=None)
    parser.add_argument("--modes", default=None)
    parser.add_argument(
        "--candidate-bank",
        default=None,
        help="Deprecated and ignored; candidate/generated refinement has been removed.",
    )
    parser.add_argument("--base-mesh", default=None)
    parser.add_argument("--marker", default=None)
    parser.add_argument("--def-template", default=None)
    parser.add_argument("--primal-template", default=None)
    parser.add_argument("--adjoint-template", default=None)
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--optimizer-config", default=None)
    parser.add_argument("--objective-column", default="CD")
    parser.add_argument("--mpi", default="")
    parser.add_argument("--nlevels", type=int, default=1)
    parser.add_argument("--nfinal", type=int, default=None)
    parser.add_argument("--max-iter-per-level", type=int, default=5)
    parser.add_argument("--refinement", default="ADAPTIVE")
    parser.add_argument("--refine-mode", default="KNOT_INSERTION")
    parser.add_argument("--refine-state", default=REFINE_STATE)
    parser.add_argument("--knot-score-mode", default="VIRTUAL_INSERTION", choices=ALLOWED_KNOT_SCORE_MODES)
    parser.add_argument("--knot-insertions-per-refine", default="1")
    parser.add_argument("--knot-min-span-width", type=float, default=1.0e-8)
    parser.add_argument("--knot-depth-penalty", action="store_true", default=False)
    parser.add_argument("--no-knot-depth-penalty", dest="knot_depth_penalty", action="store_false")
    parser.add_argument("--knot-depth-penalty-mode", default="NONE", choices=KNOT_DEPTH_PENALTY_MODE_CHOICES)
    parser.add_argument("--knot-depth-power-gamma", type=float, default=0.25)
    parser.add_argument("--knot-initial-span-depth", type=int, default=1)
    parser.add_argument("--knot-batch-diversity", dest="knot_depth_penalty", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--knot-batch-penalty-mode", dest="knot_depth_penalty_mode", choices=KNOT_DEPTH_PENALTY_MODE_CHOICES, help=argparse.SUPPRESS)
    parser.add_argument("--knot-batch-power-gamma", dest="knot_depth_power_gamma", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--transfer-method", default="BOEHM")
    parser.add_argument("--transfer-bound-policy", default="ERROR")
    parser.add_argument("--transfer-geometry-abs-tol", type=float, default=1.0e-10)
    parser.add_argument("--transfer-geometry-rel-tol", type=float, default=1.0e-8)
    parser.add_argument("--nadd-mode", default="GROWTH_RATIO")
    parser.add_argument("--batch-size-max", type=int, default=1)
    parser.add_argument("--growth-ratio", type=float, default=2.0)
    parser.add_argument("--fixed-nadd", type=int, default=1)
    parser.add_argument("--active-budget-reallocation", action="store_true", default=False)
    parser.add_argument("--no-active-budget-reallocation", dest="active_budget_reallocation", action="store_false")
    parser.add_argument("--reallocation-improvement-rel-tol", type=float, default=0.10)
    parser.add_argument("--reallocation-count-mode", default="LAST_ADDED", choices=ALLOWED_REALLOCATION_COUNT_MODES)
    parser.add_argument("--reallocation-freeze-metric", default="COEFF_DELTA", choices=ALLOWED_REALLOCATION_FREEZE_METRICS)
    parser.add_argument("--eval-layout", default="DSN", choices=ALLOWED_EVAL_LAYOUTS)
    parser.add_argument("--objective-adjoint", default="drag")
    parser.add_argument(
        "--sensitivity-source",
        default="DOT_AD_TRANSFER",
        choices=ALLOWED_SENSITIVITY_SOURCES,
    )
    parser.add_argument("--geometry-fd-eps", type=float, default=1.0e-6)
    parser.add_argument(
        "--geometry-constraint-gradient",
        default="AUTO",
        choices=("AUTO", "ANALYTIC", "SU2_GEO"),
    )
    parser.add_argument("--symmetry-coupling", default="NONE", choices=ALLOWED_SYMMETRY_COUPLINGS)
    parser.add_argument("--surface-mode", default="BOTH", choices=ALLOWED_SURFACE_MODES)
    parser.add_argument(
        "--refine-side-coupling",
        default="COUPLED",
        choices=("COUPLED", "INDEPENDENT"),
    )
    parser.add_argument(
        "--deformation-direction",
        dest="deformation_direction_mode",
        default=None,
        choices=ALLOWED_DEFORMATION_DIRECTION_MODES,
    )
    parser.add_argument("--trigger", default="MAX_ITER")
    parser.add_argument("--trigger-window", dest="window", type=int, default=1)
    parser.add_argument("--trigger-ratio", dest="tol", type=float, default=0.2)
    parser.add_argument("--trigger-eps", dest="eps", type=float, default=1.0e-300)
    parser.add_argument("--slope-filter-tol", type=float, default=0.02)
    parser.add_argument("--slope-patience", type=int, default=1)
    parser.add_argument("--stagnation-window", dest="stag_window", type=int, default=3)
    parser.add_argument("--stagnation-rel-tol", dest="stag_tol", type=float, default=1.0e-3)
    parser.add_argument("--stagnation-band", dest="stag_band", type=float, default=0.02)
    parser.add_argument("--stagnation-patience", dest="stag_patience", type=int, default=1)
    parser.add_argument("--warmup-iter", type=int, default=0)
    parser.add_argument("--generate-initial-modes", default=None)
    parser.add_argument("--initial-nper-side", type=int, default=7)
    parser.add_argument("--initial-degree", type=int, default=3)
    parser.add_argument("--initial-coefficient", type=float, default=0.0)
    parser.add_argument("--initial-bound-lower", type=float, default=-0.01)
    parser.add_argument("--initial-bound-upper", type=float, default=0.01)
    parser.add_argument("--initial-class-shape", default="sqrt_x_one_minus_x")
    parser.add_argument("--initial-class-shape-exponent", type=float, default=0.5)
    parser.add_argument("--initial-normalize-basis", default=True)
    parser.add_argument("--initial-normalization-mode", default="max")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--show-commands", action="store_true", default=False)
    parser.add_argument("--stream-solver-output", action="store_true", default=False)
    parser.add_argument("--quiet-driver", dest="show_commands", action="store_false")
    parser.add_argument("--no-optimizer-table", dest="print_optimizer_table", action="store_false")
    parser.set_defaults(print_optimizer_table=True)
    parser.add_argument("--log-active-modes", action="store_true", default=False)
    parser.add_argument("--auto-scale-bounds-to-geometry", action="store_true", default=False)
    parser.add_argument("--max-normal-displacement", type=float, default=None)
    parser.add_argument("--max-rms-normal-displacement", type=float, default=None)
    parser.add_argument("--min-bound-scale", type=float, default=0.0)
    parser.add_argument("--opt-bound-upper", type=float, default=None)
    parser.add_argument("--opt-bound-lower", type=float, default=None)
    parser.add_argument("--opt-relax-factor", type=float, default=1.0)
    parser.add_argument("--opt-gradient-factor", type=float, default=1.0)
    parser.add_argument("--gradient-guard", dest="gradient_guard", action="store_true")
    parser.add_argument("--no-gradient-guard", dest="gradient_guard", action="store_false")
    parser.set_defaults(gradient_guard=True)
    parser.add_argument("--gradient-guard-factor", type=float, default=100.0)
    parser.add_argument("--gradient-guard-window", type=int, default=5)
    parser.add_argument("--gradient-guard-min-history", type=int, default=3)
    parser.add_argument("--gradient-guard-floor", type=float, default=1.0e-14)
    parser.add_argument("--gradient-guard-restart-limit", type=int, default=2)
    parser.add_argument(
        "--trust-clip-policy",
        default="OFF",
        choices=("OFF", "ACCEPT_RESTART"),
    )
    parser.add_argument("--trust-clip-beta-tol", type=float, default=1.0e-12)
    parser.add_argument("--trust-clip-legacy-beta-min", type=float, default=0.50)
    parser.add_argument("--trust-clip-severe-beta", type=float, default=0.50)
    parser.add_argument("--trust-clip-worsening-tol", type=float, default=0.05)
    parser.add_argument("--trust-clip-soft-gnorm-factor", type=float, default=20.0)
    parser.add_argument("--trust-clip-bad-patience", type=int, default=2)
    parser.add_argument("--trust-clip-bad-window", type=int, default=5)
    parser.add_argument("--trust-clip-stag-tol", type=float, default=1.0e-6)
    parser.add_argument("--trust-clip-restart-limit", type=int, default=1)
    parser.add_argument("--opt-line-search-bound", type=float, default=None)
    parser.add_argument("--local-step-limit", action="store_true", default=False)
    parser.add_argument("--local-step-limit-ratio", type=float, default=200.0)
    parser.add_argument(
        "--le-safe-direction",
        action="store_true",
        default=False,
        help="Use the fixed-leading-edge-safe deformation direction near x/c=0",
    )
    parser.add_argument("--le-safe-x0", type=float, default=LE_SAFE_DEFAULT_X0)
    parser.add_argument("--le-safe-x1", type=float, default=LE_SAFE_DEFAULT_X1)
    parser.add_argument("--le-safe-power", type=float, default=LE_SAFE_DEFAULT_POWER)
    return parser


def _option_provided(argv, option):
    for item in list(argv or []):
        text = str(item)
        if text == option or text.startswith(option + "="):
            return True
    return False


def parse_adaptive_options(argv=None):
    parser = _build_arg_parser()
    argv_list = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)
    args._mpi_cli_provided = _option_provided(argv_list, "--mpi")
    if args.case_config and not args.optimizer_config:
        args.optimizer_config = args.case_config
    args = apply_optimizer_config_to_args(
        args,
        parser,
        argv,
        adaptive_options_from_config,
        warning_prefix="[PROGRESSIVE_BSPLINE]",
    )
    settings = prepare_bspline_launch_settings(_settings_from_args(args))
    return validate_adaptive_options(settings)


def main(argv=None):
    parser = _build_arg_parser()
    try:
        argv_list = list(sys.argv[1:] if argv is None else argv)
        args = parser.parse_args(argv)
        args._mpi_cli_provided = _option_provided(argv_list, "--mpi")
        if args.case_config and not args.optimizer_config:
            args.optimizer_config = args.case_config
        args = apply_optimizer_config_to_args(
            args,
            parser,
            argv,
            adaptive_options_from_config,
            warning_prefix="[PROGRESSIVE_BSPLINE]",
        )
        result = progressive_bspline_su2_shape_optimization(_settings_from_args(args))
    except (BSplineAdaptiveError, BSplineModeError, BSplineSU2DriverError, OSError, ValueError, NotImplementedError) as exc:
        parser.error(str(exc))

    print("[PROGRESSIVE_BSPLINE] Status: {}".format(result["status"]))
    if result.get("active_modes_final"):
        print("[PROGRESSIVE_BSPLINE] Wrote {}".format(result["active_modes_final"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
