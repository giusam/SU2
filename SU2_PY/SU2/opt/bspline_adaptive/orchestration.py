"""Adaptive optimization orchestration (progressive driver + gradient-guard restarts)."""

import json
import math
import shutil
from pathlib import Path
from SU2.opt.bspline_modes import (
    BSplineModeError,
    LE_SAFE_DEFAULT_POWER,
    LE_SAFE_DEFAULT_X0,
    LE_SAFE_DEFAULT_X1,
    load_mode_spec,
    validate_surface_mode_against_modes,
)
from SU2.opt.bspline_driver.reduction import (
    active_mode_ids,
    write_mode_spec,
)
from SU2.opt.bspline_driver.driver import run_bspline_su2_optimization
from SU2.opt.progressive_trigger import build_online_trigger_opts
from .errors import BSplineAdaptiveError
from .history import (
    SAFE_OPTIMIZATION_STATUSES,
    _append_restart_history_rows,
    _level_summary_row,
    _log_trigger_decision,
    _read_optimization_history,
    _write_adaptive_history,
    _write_restart_history_rows,
    find_eval_dir_for_mode_coefficients,
    write_knot_span_scores_csv,
    write_selected_knot_refinement_json,
)
from .diagnostics import (
    finalize_level_diagnostics,
    initialize_level_diagnostics,
)
from .ikkt import build_ikkt_score_signal, write_ikkt_diagnostics
from .knot_space import refinement_limit_ndv
from .mode_utils import (
    build_level,
    write_level_start,
)
from .models import TriggerDecision
from .refinement import (
    build_next_knot_inserted_modes,
    compute_freeze_eligible_count,
)
from .scoring import load_adjoint_signal
from .settings import (
    _print_level_start,
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

def _relative_level_improvement(safe_opt_rows, eps):
    if not safe_opt_rows:
        return None
    start = float(safe_opt_rows[0]["_objective"])
    best = min(float(row["_objective"]) for row in safe_opt_rows)
    return (start - best) / (abs(start) + float(eps))


def _eval_id_from_eval_dir(eval_dir):
    if eval_dir is None:
        return None
    name = Path(eval_dir).name
    if name.startswith("eval_"):
        try:
            return int(name.split("_", 1)[1])
        except Exception:
            return None
    return None


def _history_row_for_eval_id(rows, eval_id):
    if eval_id is None:
        return {}
    for row in rows:
        try:
            if int(row.get("_eval_id", row.get("eval_id", -1))) == int(eval_id):
                return row
        except Exception:
            continue
    return {}


def _scoring_diagnostic_context(
    settings,
    level,
    level_id,
    adjoint_eval_dir,
    opt_rows,
    optimized_modes_filename,
):
    eval_id = _eval_id_from_eval_dir(adjoint_eval_dir)
    history_row = _history_row_for_eval_id(opt_rows, eval_id)
    score_mode = str(settings.get("knot_score_mode", "VIRTUAL_INSERTION")).upper()
    return {
        "diagnostic_version": 1,
        "level": int(level_id),
        "batch_step": 1,
        "scoring_pass_id": 1,
        "score_mode": score_mode,
        "primary_signal_name": (
            "ikkt_residual"
            if score_mode == "IKKT_VIRTUAL_INSERTION"
            else "objective"
        ),
        "workdir": str(settings.get("workdir")),
        "case_name": Path(str(settings.get("workdir", ""))).name,
        "config_file": settings.get("case_config") or settings.get("optimizer_config"),
        "modes_file": str(optimized_modes_filename),
        "eval_id": eval_id,
        "slsqp_it": history_row.get("slsqp_iter", history_row.get("slsqp_it")),
        "sensitivity_source": settings.get("sensitivity_source", "DOT_AD_TRANSFER"),
        "sensitivity_weighting": "NODAL",
        "deformation_direction": settings.get("deformation_direction_mode"),
        "surface_mode": settings.get("surface_mode", "BOTH"),
        "symmetry_coupling": settings.get("symmetry_coupling", "NONE"),
        "refine_side_coupling": settings.get("refine_side_coupling", "COUPLED"),
        "ikkt_scaling_mode": settings.get("ikkt_scaling_mode"),
        "ikkt_sign_convention": settings.get("ikkt_sign_convention"),
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
        final_history_filename = result.get(
            "optimization_information",
            result.get("optimization_history", final_history_filename),
        )
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
        last_added_ids = settings.get("_last_added_ids", [])
        added = len(last_added_ids) if last_added_ids else int(settings.get("_last_added_design_count", 0))
        kept = max(0, len(active_mode_ids(current_modes)) - int(added))
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
        ikkt_aero_refinement = (
            str(settings.get("knot_score_mode", "VIRTUAL_INSERTION")).strip().upper()
            == "IKKT_VIRTUAL_INSERTION"
            and bool(settings.get("ikkt_include_aero_constraints", False))
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
            "moving_bounds": settings.get("moving_bounds", False),
            "local_step_limit": settings.get("local_step_limit", False),
            "local_step_limit_ratio": settings.get("local_step_limit_ratio", 200.0),
            "sensitivity_weighting": settings.get("sensitivity_weighting", "NODAL"),
            "sensitivity_source": settings.get("sensitivity_source", "DOT_AD_TRANSFER"),
            "prepare_ikkt_aero_adjoints": ikkt_aero_refinement,
            "force_prepare_ikkt_aero_adjoints": (
                ikkt_aero_refinement
                and str(settings.get("trigger", "MAX_ITER")).strip().upper() == "MAX_ITER"
            ),
            "ikkt_active_tol": settings.get("ikkt_active_tol", 1.0e-6),
            "thickness_options": settings.get("thickness_options"),
            "native_constraints": settings.get("native_constraints"),
            "geometry_fd_eps": settings.get("geometry_fd_eps", 1.0e-6),
            "geometry_constraint_gradient": settings.get(
                "geometry_constraint_gradient",
                "AUTO",
            ),
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
            sensitivity_filename = (
                adjoint_eval_dir / "surface_sens.csv"
                if settings.get("sensitivity_source", "DOT_AD_TRANSFER") == "DOT_AD_TRANSFER"
                else adjoint_eval_dir / "surface_adjoint.csv"
            )
            metadata, signal = load_adjoint_signal(
                sensitivity_filename,
                adjoint_eval_dir / "bspline_surface_metadata.csv",
            )
            refinement_settings = settings
            ikkt_diagnostics = None
            objective_signal_for_diagnostics = signal
            if str(settings.get("knot_score_mode", "VIRTUAL_INSERTION")).upper() == "IKKT_VIRTUAL_INSERTION":
                objective_signal = signal
                ikkt_settings = dict(settings)
                ikkt_settings["_ikkt_eval_dir"] = str(adjoint_eval_dir)
                signal, ikkt_diagnostics = build_ikkt_score_signal(
                    optimized_modes,
                    metadata,
                    objective_signal,
                    ikkt_settings,
                )
                ikkt_file = level.workdir / f"ikkt_score_signal_level_{level_id:03d}.json"
                write_ikkt_diagnostics(ikkt_file, ikkt_diagnostics)
                refinement_settings = dict(ikkt_settings)
                refinement_settings["_ikkt_objective_signal"] = objective_signal
                objective_signal_for_diagnostics = objective_signal
                print(
                    "[PROGRESSIVE_BSPLINE] IKKT_VIRTUAL_INSERTION signal | "
                    "constraints={} residual_norm={:.6e} rel={:.6e} diagnostics={}".format(
                        len(ikkt_diagnostics.get("included_constraints", [])),
                        float(ikkt_diagnostics.get("lagrangian_residual_norm", 0.0)),
                        float(ikkt_diagnostics.get("relative_lagrangian_residual_norm", 0.0)),
                        ikkt_file.name,
                    )
                )
            initialize_level_diagnostics(
                refinement_settings,
                _scoring_diagnostic_context(
                    settings,
                    level,
                    level_id,
                    adjoint_eval_dir,
                    opt_rows,
                    level.optimized_modes_filename,
                ),
                metadata,
                signal,
                objective_signal=objective_signal_for_diagnostics,
                ikkt_diagnostics=ikkt_diagnostics,
            )
            rel_improvement = None
            if settings.get("active_budget_reallocation", False):
                rel_improvement = _relative_level_improvement(
                    safe_opt_rows,
                    settings.get("eps", 1.0e-300),
                )
                tol = float(settings.get("reallocation_improvement_rel_tol", 0.10))
                if rel_improvement is None:
                    print(
                        "[PROGRESSIVE_BSPLINE] ACTIVE_BUDGET_REALLOCATION normal refinement | "
                        "no safe objective history available"
                    )
                elif rel_improvement >= tol:
                    print(
                        "[PROGRESSIVE_BSPLINE] ACTIVE_BUDGET_REALLOCATION normal refinement | "
                        "rel_improvement={:.6e} tol={:.6e}".format(
                            float(rel_improvement),
                            tol,
                        )
                    )
                else:
                    last_added = int(settings.get("_last_added_design_count", 0))
                    if last_added <= 0:
                        print(
                            "[PROGRESSIVE_BSPLINE] ACTIVE_BUDGET_REALLOCATION normal refinement | "
                            "last_added_design_count <= 0"
                        )
                    else:
                        eligible_count = compute_freeze_eligible_count(optimized_modes)
                        if eligible_count == 0:
                            print(
                                "[PROGRESSIVE_BSPLINE] ACTIVE_BUDGET_REALLOCATION normal refinement | "
                                "no freeze-eligible design modes available"
                            )
                        else:
                            k_eff = min(last_added, int(eligible_count))
                            if k_eff < last_added:
                                print(
                                    "[PROGRESSIVE_BSPLINE] ACTIVE_BUDGET_REALLOCATION warning | "
                                    "eligible_count={} < requested_k={}; using k_eff={}".format(
                                        int(eligible_count),
                                        int(last_added),
                                        int(k_eff),
                                    )
                                )
                            refinement_settings = dict(refinement_settings)
                            refinement_settings["_active_budget_reallocation_current"] = True
                            refinement_settings["_reallocation_forced_design_add_count"] = int(k_eff)
                            refinement_settings["_reallocation_freeze_count"] = int(k_eff)
                            refinement_settings["_reallocation_rel_improvement"] = float(rel_improvement)
                            print(
                                "[PROGRESSIVE_BSPLINE] ACTIVE_BUDGET_REALLOCATION selected | "
                                "rel_improvement={:.6e} tol={:.6e} k={} k_eff={} eligible={}".format(
                                    float(rel_improvement),
                                    tol,
                                    int(last_added),
                                    int(k_eff),
                                    int(eligible_count),
                                )
                            )
            next_modes, knot_score_rows, knot_selected_data = build_next_knot_inserted_modes(
                optimized_modes,
                metadata,
                signal,
                refinement_settings,
                level_start_modes=level.active_modes,
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
            finalize_level_diagnostics(
                refinement_settings,
                knot_score_rows,
                knot_selected_data,
            )
            if next_modes is not None:
                if knot_selected_data.get("active_budget_reallocation", False):
                    n_design_before = int(knot_selected_data["n_design_before"])
                    n_design_after = int(knot_selected_data["n_design_after"])
                    if n_design_after != n_design_before:
                        raise BSplineAdaptiveError(
                            "active-budget reallocation invariant failed: "
                            f"n_design_before={n_design_before} n_design_after={n_design_after}"
                        )
                next_file = level.workdir / "active_modes_next.json"
                write_mode_spec(next_modes, next_file)
                knot_refine_used = True
                current_modes = next_modes
                n_added_design = int(
                    knot_selected_data.get(
                        "n_added_design",
                        max(
                            0,
                            int(knot_selected_data.get("n_design_after", 0))
                            - int(knot_selected_data.get("n_design_before", 0)),
                        ),
                    )
                )
                settings["_last_added_design_count"] = n_added_design
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
                        int(n_added_design),
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
                        int(knot_selected_data.get("n_design_before", knot_selected_data["ndv_before"])),
                        int(n_added_design),
                        float(knot_selected_data["transfer_max_error"]),
                    )
                )
                if knot_selected_data.get("active_budget_reallocation", False):
                    frozen_mode_ids = knot_selected_data.get("reallocated_frozen_mode_ids", [])
                    print(
                        "[PROGRESSIVE_BSPLINE] ACTIVE_BUDGET_REALLOCATION applied | "
                        "n_design={} n_frozen={} n_geometric_total={} froze={}".format(
                            int(knot_selected_data["n_design_after"]),
                            int(knot_selected_data["n_frozen"]),
                            int(knot_selected_data["n_geometric_total"]),
                            int(knot_selected_data["reallocated_freeze_count"]),
                        )
                    )
                    print(
                        "[PROGRESSIVE_BSPLINE] ACTIVE_BUDGET_REALLOCATION frozen modes: {}".format(
                            ", ".join(str(mode_id) for mode_id in frozen_mode_ids)
                            if frozen_mode_ids
                            else "NONE"
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
            level_summary["n_added"] = int(
                knot_selected_data.get(
                    "n_added_design",
                    int(knot_selected_data["ndv_after"]) - int(knot_selected_data["ndv_before"]),
                )
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
                    "sensitivity_source": settings["sensitivity_source"],
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
