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









































































































































































































def gradient_guard_next_action_for_level(
    level_id,
    nlevels,
    current_ndv,
    nfinal=None,
):
    refinement_available = int(level_id) < int(nlevels) - 1
    if nfinal is not None and int(current_ndv) >= int(nfinal):
        refinement_available = False
    return "refine" if refinement_available else "restart_same_level"


def _trigger_resume_state_from_result(result):
    return {
        "trigger_history": list(result.get("trigger_history", [])),
        "trigger_state": result.get("trigger_state", None),
        "refinement_triggered": bool(result.get("refinement_triggered", False)),
    }






def run_with_gradient_guard_restarts(
    run_optimizer,
    optimizer_kwargs,
    *,
    active_modes_start_filename,
    optimized_modes_filename,
    next_action,
    refinement_available=None,
    restart_limit=2,
    trust_clip_restart_limit=1,
):
    """Run a level, resetting SLSQP state after controlled safety stops."""

    optimizer_kwargs = dict(optimizer_kwargs)
    restart_count = 0
    trust_clip_restart_count = 0
    attempt_id = 0
    attempt_restart_reason = ""
    trigger_resume_state = None
    full_level_history_rows = []
    full_level_history_fieldnames = []
    final_history_filename = None
    restart_limit = int(restart_limit)
    trust_clip_restart_limit = int(trust_clip_restart_limit)
    if refinement_available is None:
        refinement_available = str(next_action) == "refine"
    refinement_available = bool(refinement_available)
    while True:
        if trigger_resume_state is None:
            optimizer_kwargs.pop("trigger_resume_state", None)
        else:
            optimizer_kwargs["trigger_resume_state"] = trigger_resume_state
        result = run_optimizer(**optimizer_kwargs)
        final_history_filename = result.get("optimization_history", final_history_filename)
        if final_history_filename:
            _append_restart_history_rows(
                full_level_history_rows,
                full_level_history_fieldnames,
                final_history_filename,
                attempt_id,
                attempt_restart_reason,
            )
        current_trigger_resume_state = _trigger_resume_state_from_result(result)
        gradient_triggered = result.get("gradient_guard_triggered", False)
        trust_clip_triggered = result.get("trust_clip_triggered", False)
        if not gradient_triggered and not trust_clip_triggered:
            break
        if gradient_triggered:
            if str(next_action) != "restart_same_level":
                break
            if restart_count >= restart_limit:
                result["gradient_guard_restart_limit_reached"] = True
                print(
                    "[PROGRESSIVE_BSPLINE] GRADIENT_GUARD restart limit reached; "
                    "terminating successfully with the restored best-safe design"
                )
                break
            restart_count += 1
            label = "GRADIENT_GUARD"
            count = restart_count
            limit = restart_limit
        else:
            requested_action = str(
                result.get("trust_clip_next_action", "restart_same_level")
            )
            if requested_action == "refine" and refinement_available:
                result["trust_clip_force_refine"] = True
                break
            if trust_clip_restart_count >= trust_clip_restart_limit:
                result["trust_clip_restart_limit_reached"] = True
                if refinement_available:
                    result["trust_clip_force_refine"] = True
                    result["trust_clip_next_action"] = "refine"
                print(
                    "[PROGRESSIVE_BSPLINE] TRUST_CLIP restart limit reached; "
                    + (
                        "forcing refinement from best-safe"
                        if refinement_available
                        else "terminating successfully with best-safe"
                    )
                )
                break
            trust_clip_restart_count += 1
            label = "TRUST_CLIP"
            count = trust_clip_restart_count
            limit = trust_clip_restart_limit
        shutil.copy2(optimized_modes_filename, active_modes_start_filename)
        trigger_resume_state = current_trigger_resume_state
        print(
            "[PROGRESSIVE_BSPLINE][TRIGGER] preserve on restart_same_level "
            f"| history_len={len(trigger_resume_state['trigger_history'])}"
        )
        print(
            f"[PROGRESSIVE_BSPLINE] {label} restart_same_level "
            f"attempt={count}/{limit} from restored physical design"
        )
        attempt_id += 1
        attempt_restart_reason = label
    if final_history_filename:
        _write_restart_history_rows(
            final_history_filename,
            full_level_history_rows,
            full_level_history_fieldnames,
        )
    result["gradient_guard_restart_count"] = restart_count
    result["trust_clip_restart_count"] = trust_clip_restart_count
    return result


def progressive_bspline_su2_shape_optimization(settings):
    settings = prepare_bspline_launch_settings(settings)
    settings = validate_adaptive_options(settings)
    print_startup_summary(settings)
    workdir = Path(settings["workdir"]).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    with open(workdir / "adaptive_settings.json", "w") as fp:
        json.dump(settings, fp, indent=2, sort_keys=True)
        fp.write("\n")

    initial_modes = load_mode_spec(settings["modes"])
    try:
        validate_surface_mode_against_modes(initial_modes, settings["surface_mode"])
    except BSplineModeError as exc:
        raise BSplineAdaptiveError(str(exc))
    if settings["surface_mode"] != "BOTH" and "surface_mode" not in initial_modes:
        initial_modes["surface_mode"] = settings["surface_mode"]
    current_modes = initial_modes
    adaptive_rows = []

    for level_id in range(settings["nlevels"]):
        level = build_level(
            level_id,
            current_modes,
            workdir,
            settings["modes"],
        )
        kept = len(active_mode_ids(current_modes)) - len(settings.get("_last_added_ids", []))
        added = len(settings.get("_last_added_ids", []))
        write_level_start(level)
        _print_level_start(
            level,
            settings["refine_state"],
            kept=kept,
            added=added,
            log_active_modes=settings.get("log_active_modes", False),
        )
        print(
            "[PROGRESSIVE_BSPLINE][TRIGGER] reset on new refinement level "
            f"| level={level_id}"
        )

        if settings.get("dry_run", False):
            print("[PROGRESSIVE_BSPLINE] Dry run requested; not running SU2.")
            return {
                "status": "dry_run",
                "workdir": str(workdir),
                "level": level,
            }

        current_reduced_ndv = refinement_limit_ndv(current_modes, settings)
        trigger_opts = build_online_trigger_opts(
            settings["trigger"],
            current_level=level_id,
            current_ndv=current_reduced_ndv,
            final_ndv=settings.get("nfinal"),
            nlevels=settings.get("nlevels"),
            window=settings["window"],
            tolerance=settings["tol"],
            filter_tolerance=settings["slope_filter_tol"],
            warmup=settings["warmup_iter"],
            eps=settings["eps"],
            patience=settings["slope_patience"],
            stagnation_tolerance=settings["stag_tol"],
            stagnation_band=settings["stag_band"],
            stagnation_window=settings["stag_window"],
        )

        gradient_guard_action = gradient_guard_next_action_for_level(
            level_id,
            settings["nlevels"],
            current_reduced_ndv,
            settings.get("nfinal"),
        )
        refinement_available = (
            int(level_id) < int(settings["nlevels"]) - 1
            and (
                settings.get("nfinal") is None
                or int(current_reduced_ndv) < int(settings["nfinal"])
            )
        )
        optimizer_kwargs = {
            "modes_filename": str(level.active_modes_start_filename),
            "base_mesh": settings["base_mesh"],
            "marker": settings["marker"],
            "def_template": settings["def_template"],
            "primal_template": settings["primal_template"],
            "adjoint_template": settings["adjoint_template"],
            "workdir": str(level.opt_workdir),
            "objective_column": settings["objective_column"],
            "maxiter": settings["max_iter_per_level"],
            "mpi_prefix": settings.get("mpi", ""),
            "show_commands": bool(settings.get("show_commands", False)),
            "stream_solver_output": bool(settings.get("stream_solver_output", False)),
            "print_optimizer_table": bool(settings.get("print_optimizer_table", True)),
            "auto_scale_bounds_to_geometry": bool(settings.get("auto_scale_bounds_to_geometry", False)),
            "max_normal_displacement": settings.get("max_normal_displacement"),
            "max_rms_normal_displacement": settings.get("max_rms_normal_displacement"),
            "min_bound_scale": settings.get("min_bound_scale", 0.0),
            "opt_accuracy": settings.get("opt_accuracy"),
            "opt_bound_upper": settings.get("opt_bound_upper"),
            "opt_bound_lower": settings.get("opt_bound_lower"),
            "opt_relax_factor": settings.get("opt_relax_factor", 1.0),
            "opt_gradient_factor": settings.get("opt_gradient_factor", 1.0),
            "gradient_guard": settings.get("gradient_guard", True),
            "gradient_guard_factor": settings.get("gradient_guard_factor", 100.0),
            "gradient_guard_window": settings.get("gradient_guard_window", 5),
            "gradient_guard_min_history": settings.get("gradient_guard_min_history", 3),
            "gradient_guard_floor": settings.get("gradient_guard_floor", 1.0e-14),
            "gradient_guard_next_action": gradient_guard_action,
            "refinement_available": refinement_available,
            "trust_clip_policy": settings.get("trust_clip_policy", "OFF"),
            "trust_clip_beta_tol": settings.get("trust_clip_beta_tol", 1.0e-12),
            "trust_clip_legacy_beta_min": settings.get(
                "trust_clip_legacy_beta_min", 0.50
            ),
            "trust_clip_severe_beta": settings.get("trust_clip_severe_beta", 0.50),
            "trust_clip_worsening_tol": settings.get(
                "trust_clip_worsening_tol", 0.05
            ),
            "trust_clip_soft_gnorm_factor": settings.get(
                "trust_clip_soft_gnorm_factor", 20.0
            ),
            "trust_clip_bad_patience": settings.get("trust_clip_bad_patience", 2),
            "trust_clip_bad_window": settings.get("trust_clip_bad_window", 5),
            "trust_clip_stag_tol": settings.get("trust_clip_stag_tol", 1.0e-6),
            "opt_line_search_bound": settings.get("opt_line_search_bound"),
            "local_step_limit": settings.get("local_step_limit", False),
            "local_step_limit_ratio": settings.get("local_step_limit_ratio", 200.0),
            "sensitivity_weighting": settings.get("sensitivity_weighting", "NODAL"),
            "thickness_options": settings.get("thickness_options"),
            "eval_layout": settings.get("eval_layout", "DSN"),
            "objective_adjoint": settings.get("objective_adjoint", "drag"),
            "symmetry_coupling": settings.get("symmetry_coupling", "NONE"),
            "surface_mode": settings.get("surface_mode", "BOTH"),
            "trigger_opts": trigger_opts,
            "progressive_label": "PROGRESSIVE_BSPLINE",
            "deformation_direction_mode": settings.get("deformation_direction_mode"),
            "le_safe_direction": bool(settings.get("le_safe_direction", False)),
            "le_safe_x0": settings.get("le_safe_x0", LE_SAFE_DEFAULT_X0),
            "le_safe_x1": settings.get("le_safe_x1", LE_SAFE_DEFAULT_X1),
            "le_safe_power": settings.get("le_safe_power", LE_SAFE_DEFAULT_POWER),
        }
        result = run_with_gradient_guard_restarts(
            run_bspline_su2_optimization,
            optimizer_kwargs,
            active_modes_start_filename=level.active_modes_start_filename,
            optimized_modes_filename=level.optimized_modes_filename,
            next_action=gradient_guard_action,
            refinement_available=refinement_available,
            restart_limit=settings.get("gradient_guard_restart_limit", 2),
            trust_clip_restart_limit=settings.get("trust_clip_restart_limit", 1),
        )

        opt_rows = _read_optimization_history(level.opt_workdir)
        optimized_modes = load_mode_spec(str(level.optimized_modes_filename))
        safe_opt_rows = [
            row
            for row in opt_rows
            if str(row.get("status", "ok")).strip().lower()
            in SAFE_OPTIMIZATION_STATUSES
            and math.isfinite(float(row["_objective"]))
        ]
        adjoint_eval_dir = None
        if safe_opt_rows:
            adjoint_eval_dir = find_eval_dir_for_mode_coefficients(
                level.opt_workdir,
                optimized_modes,
            )
            best_objective = min(row["_objective"] for row in safe_opt_rows)
        else:
            best_objective = float(result.get("objective", math.inf))
        print(
            "[PROGRESSIVE_BSPLINE] Level {} optimization complete | best {} = {:.6e} | adjoint eval = {}".format(
                level_id,
                settings["objective_column"],
                best_objective,
                adjoint_eval_dir.name if adjoint_eval_dir is not None else "NONE",
            )
        )

        reached_limits = (
            level_id >= settings["nlevels"] - 1
            or (
                settings["nfinal"] is not None
                and refinement_limit_ndv(optimized_modes, settings) >= settings["nfinal"]
            )
        )
        trigger_mode = str(settings["trigger"]).upper()
        if result.get("trust_clip_triggered", False):
            diagnostics = result.get("trust_clip_diagnostics") or {}
            toxic_reasons = diagnostics.get("toxic_reasons") or []
            refine_requested = bool(result.get("trust_clip_force_refine", False))
            if "clipped_stagnation_plateau" in toxic_reasons:
                trigger_reason = "clipped_stagnation_plateau"
            elif "toxic_clipped_repeated" in toxic_reasons:
                trigger_reason = "toxic_clipped_plateau"
            else:
                trigger_reason = "trust_clip_restart_limit"
            trigger_counter = 1
        elif result.get("gradient_guard_triggered", False):
            refine_requested = gradient_guard_action == "refine"
            trigger_reason = "raw_gradient_guard"
            trigger_counter = 1
        elif trigger_mode == "MAX_ITER":
            refine_requested = True
            trigger_reason = "level_complete"
            trigger_counter = 1
        elif result.get("early_refine_triggered", False):
            refine_requested = True
            trigger_reason = "early_refine_trigger"
            trigger_counter = 1
        else:
            refine_requested = False
            trigger_reason = "online_trigger_not_fired"
            trigger_counter = 0
        trigger_decision = TriggerDecision(
            trigger_mode=trigger_mode,
            threshold=settings["tol"] if trigger_mode != "MAX_ITER" else "",
            window=settings["window"] if trigger_mode != "MAX_ITER" else "",
            patience=settings["slope_patience"] if trigger_mode != "MAX_ITER" else "",
            counter=trigger_counter,
            refine_now=refine_requested,
            reason=trigger_reason,
        )
        refine_now = bool(
            not reached_limits
            and trigger_decision.refine_now
            and adjoint_eval_dir is not None
        )
        if trigger_decision.refine_now and not refine_now:
            if reached_limits:
                trigger_decision.reason = "limits_reached"
            elif adjoint_eval_dir is None:
                trigger_decision.reason = "no_safe_adjoint_for_refinement"
            trigger_decision.refine_now = False
            _log_trigger_decision(trigger_decision)

        selected_modes = []
        knot_score_rows = []
        knot_selected_data = {}
        knot_refine_used = False
        if refine_now:
            metadata, signal = load_adjoint_signal(
                adjoint_eval_dir / "surface_adjoint.csv",
                adjoint_eval_dir / "bspline_surface_metadata.csv",
            )
            next_modes, knot_score_rows, knot_selected_data = build_next_knot_inserted_modes(
                optimized_modes,
                metadata,
                signal,
                settings,
            )
            knot_score_file = level.workdir / f"knot_span_scores_level_{level_id:03d}.csv"
            knot_selected_file = level.workdir / f"selected_knot_refinement_level_{level_id:03d}.json"
            write_knot_span_scores_csv(knot_score_rows, knot_score_file)
            write_selected_knot_refinement_json(
                level_id,
                knot_selected_data,
                knot_score_rows,
                knot_selected_file,
            )
            if next_modes is not None:
                next_file = level.workdir / "active_modes_next.json"
                write_mode_spec(next_modes, next_file)
                knot_refine_used = True
                current_modes = next_modes
                settings["_last_added_ids"] = [
                    mode_id
                    for mode_id in active_mode_ids(next_modes)
                    if mode_id not in set(active_mode_ids(optimized_modes))
                ]
                print(
                    "[PROGRESSIVE_BSPLINE] Level {} | NDV = {} | refined span=[{:.6f},{:.6f}] knot={:.6f} | added modes={}".format(
                        level_id + 1,
                        len(active_mode_ids(next_modes)),
                        float(knot_selected_data["span_left"]),
                        float(knot_selected_data["span_right"]),
                        float(knot_selected_data["inserted_knot"]),
                        int(knot_selected_data["ndv_after"]) - int(knot_selected_data["ndv_before"]),
                    )
                )
                print(
                    "[PROGRESSIVE_BSPLINE] refinement | ndv_before={} ndv_after={} "
                    "selected side={} x={:.6f}".format(
                        int(knot_selected_data["ndv_before"]),
                        int(knot_selected_data["ndv_after"]),
                        str(knot_selected_data.get("side", "BOTH")).upper(),
                        float(knot_selected_data["inserted_knot"]),
                    )
                )
                print(
                    "[PROGRESSIVE_BSPLINE] KNOT_INSERTION transfer | "
                    "rms={:.6e} max={:.6e} rel={:.6e}".format(
                        float(knot_selected_data["transfer_rms_error"]),
                        float(knot_selected_data["transfer_max_error"]),
                        float(knot_selected_data["transfer_relative_error"]),
                    )
                )
                print(
                    "[PROGRESSIVE_BSPLINE] Coefficients transferred from previous level: kept={} added={} transfer_max={:.6e}".format(
                        int(knot_selected_data["ndv_before"]),
                        int(knot_selected_data["ndv_after"]) - int(knot_selected_data["ndv_before"]),
                        float(knot_selected_data["transfer_max_error"]),
                    )
                )
            else:
                print("[PROGRESSIVE_BSPLINE] KNOT_INSERTION refine | no positive knot span selected")
                refine_now = False
                current_modes = optimized_modes
        else:
            current_modes = optimized_modes

        level_summary = _level_summary_row(
            level,
            opt_rows,
            selected_modes,
            trigger_decision,
            refine_now,
            "ok",
        )
        if knot_refine_used and knot_selected_data:
            level_summary["n_added"] = int(knot_selected_data["ndv_after"]) - int(
                knot_selected_data["ndv_before"]
            )
            level_summary["selected_ids"] = "knot@{:.12g}".format(
                float(knot_selected_data["inserted_knot"])
            )
        with open(level.workdir / "level_summary.json", "w") as fp:
            json.dump(
                level_summary
                | {
                    "refine_mode": settings["refine_mode"],
                    "knot_score_mode": settings["knot_score_mode"],
                    "knot_refine_used": knot_refine_used,
                    "knot_refinement": knot_selected_data,
                    "optimizer_result": result,
                    "sensitivity_weighting": settings["sensitivity_weighting"],
                },
                fp,
                indent=2,
                sort_keys=True,
            )
            fp.write("\n")
        adaptive_rows.append(level_summary)
        _write_adaptive_history(adaptive_rows, workdir / "adaptive_history.csv")

        if not refine_now:
            break

    write_mode_spec(current_modes, workdir / "active_modes_final.json")
    return {
        "status": "ok",
        "workdir": str(workdir),
        "active_modes_final": str(workdir / "active_modes_final.json"),
        "levels": len(adaptive_rows),
    }


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
    parser.add_argument("--eval-layout", default="DSN", choices=ALLOWED_EVAL_LAYOUTS)
    parser.add_argument("--objective-adjoint", default="drag")
    parser.add_argument("--symmetry-coupling", default="NONE", choices=ALLOWED_SYMMETRY_COUPLINGS)
    parser.add_argument("--surface-mode", default="BOTH", choices=ALLOWED_SURFACE_MODES)
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
