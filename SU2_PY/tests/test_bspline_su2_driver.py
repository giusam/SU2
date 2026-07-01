import copy
import csv
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import SU2.opt.bspline_driver.driver as bspline_driver_module
import SU2.opt.bspline_su2_driver as bspline_su2_driver
from SU2.opt.bspline_modes import evaluate_normal_displacement
from SU2.opt.bspline_driver.geometry_constraints import BSplineAirfoilAreaMetric
from SU2.opt.bspline_su2_driver import (
    BSplineThicknessConstraint,
    BSplineSU2Driver,
    BSplineSU2DriverError,
    GradientGuardStop,
    TrustClipStop,
    _build_arg_parser,
    active_bounds,
    active_coefficient_vector,
    active_mode_ids,
    apply_optimizer_config_to_args,
    build_eval_commands,
    build_eval_paths,
    build_reduced_variables,
    cache_key,
    classify_clipped_trial,
    collapse_full_gradient,
    collapse_full_jacobian,
    compute_geometry_aware_bound_scaling,
    expand_reduced_coefficients,
    fixed_driver_options_from_config,
    gradient_guard_triggered,
    patch_config_template,
    parse_optimizer_config,
    read_bspline_gradients,
    read_gradient_vector,
    read_objective_from_history,
    resolve_thickness_domain_mode,
    run_command,
    run_bspline_su2_optimization,
    thickness_options_from_config,
    update_mode_coefficients,
)


def _base_spec():
    knots = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    return {
        "version": 1,
        "dimension": 2,
        "marker": "airfoil",
        "chord": {"mode": "auto", "x_le": None, "x_te": None},
        "normal_displacement": True,
        "class_shape": "none",
        "normalize_basis": False,
        "normalization_mode": "max",
        "modes": [
            {
                "id": "upper_a",
                "side": "upper",
                "basis_type": "clamped",
                "degree": 3,
                "knot_vector": knots,
                "basis_index": 1,
                "coefficient": 0.001,
                "bounds": [-0.02, 0.03],
                "label": "preserve-me",
            },
            {
                "id": "lower_b",
                "side": "lower",
                "basis_type": "clamped",
                "degree": 3,
                "knot_vector": knots,
                "basis_index": 1,
                "coefficient": -0.002,
            },
            {
                "id": "inactive_c",
                "side": "upper",
                "basis_type": "clamped",
                "degree": 3,
                "knot_vector": knots,
                "basis_index": 1,
                "coefficient": 0.5,
                "bounds": [-1.0, 1.0],
                "active": False,
            },
        ],
    }


def _single_mode_spec(coefficient=0.0):
    spec = copy.deepcopy(_base_spec())
    spec["modes"] = [
        {
            "id": "lower_b",
            "side": "lower",
            "basis_type": "clamped",
            "degree": 3,
            "knot_vector": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            "basis_index": 1,
            "coefficient": coefficient,
            "bounds": [-1.0, 1.0],
        }
    ]
    return spec


def _paired_spec(npairs=4, coupling_consistent=True):
    # Each pair shares the same basis_index so they reduce to 1 variable.
    knots = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    modes = []
    for index in range(int(npairs)):
        upper_coeff = 0.001 * (index + 1)
        lower_coeff = upper_coeff if coupling_consistent else -upper_coeff
        modes.extend(
            [
                {
                    "id": f"upper_{index}",
                    "side": "upper",
                    "basis_type": "clamped",
                    "degree": 3,
                    "knot_vector": knots,
                    "basis_index": index,
                    "coefficient": upper_coeff,
                    "bounds": [-0.02, 0.03],
                },
                {
                    "id": f"lower_{index}",
                    "side": "lower",
                    "basis_type": "clamped",
                    "degree": 3,
                    "knot_vector": knots,
                    "basis_index": index,
                    "coefficient": lower_coeff,
                    "bounds": [-0.01, 0.02],
                },
            ]
        )
    spec = copy.deepcopy(_base_spec())
    spec["modes"] = modes
    return spec


def _paired_clamped_spec():
    knots = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    modes = []
    for index in range(4):
        for side in ("upper", "lower"):
            modes.append(
                {
                    "id": f"{side}_clamped_i{index:03d}",
                    "side": side,
                    "basis_type": "clamped",
                    "degree": 3,
                    "knot_vector": knots,
                    "basis_index": index,
                    "coefficient": 0.0,
                    "bounds": [-0.02, 0.02],
                }
            )
    spec = copy.deepcopy(_base_spec())
    spec["modes"] = modes
    return spec


def _write_driver_inputs(tmp_path, spec=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    modes_file = tmp_path / "modes.json"
    modes_file.write_text(json.dumps(_base_spec() if spec is None else spec, indent=2))
    base = tmp_path / "base.su2"
    base.write_text("mesh")
    def_template = tmp_path / "def.cfg"
    primal_template = tmp_path / "primal.cfg"
    adjoint_template = tmp_path / "adj.cfg"
    for path in (def_template, primal_template, adjoint_template):
        path.write_text("MESH_FILENAME= old.su2\n")
    return modes_file, base, def_template, primal_template, adjoint_template


def _write_simple_airfoil_mesh(path):
    path.write_text(
        "NDIME= 2\n"
        "NPOIN= 6\n"
        "0.0 0.0 0\n"
        "0.25 0.1 1\n"
        "0.75 0.1 2\n"
        "1.0 0.0 3\n"
        "0.75 -0.1 4\n"
        "0.25 -0.1 5\n"
        "NMARK= 1\n"
        "MARKER_TAG= airfoil\n"
        "MARKER_ELEMS= 6\n"
        "3 0 1\n"
        "3 1 2\n"
        "3 2 3\n"
        "3 3 4\n"
        "3 4 5\n"
        "3 5 0\n"
    )
    return path


def _synthetic_thickness_constraint(gradient_mode="ANALYTIC"):
    metadata = [
        {"node_id": 0, "x": 0.0, "y": 0.0, "normal_x": 0.0, "normal_y": 0.0},
        {"node_id": 1, "x": 0.25, "y": 0.1, "normal_x": 0.0, "normal_y": 1.0},
        {"node_id": 2, "x": 0.75, "y": 0.1, "normal_x": 0.0, "normal_y": 1.0},
        {"node_id": 3, "x": 1.0, "y": 0.0, "normal_x": 0.0, "normal_y": 0.0},
        {"node_id": 4, "x": 0.75, "y": -0.1, "normal_x": 0.0, "normal_y": -1.0},
        {"node_id": 5, "x": 0.25, "y": -0.1, "normal_x": 0.0, "normal_y": -1.0},
    ]
    basis = np.asarray([[0.0], [0.0], [0.0], [0.0], [1.0], [1.0]], dtype=float)
    return BSplineThicknessConstraint(
        metadata,
        basis,
        ["lower_b"],
        reference_measure=[0.2],
        x_stations=[0.5],
        margin=0.0,
        domain_mode="FULL",
        gradient_mode=gradient_mode,
        closed=True,
        marker="airfoil",
    )


def _make_driver(tmp_path, spec=None, **kwargs):
    modes_file, base, def_template, primal_template, adjoint_template = _write_driver_inputs(
        tmp_path,
        spec=spec,
    )
    return BSplineSU2Driver(
        modes_filename=str(modes_file),
        base_mesh=str(base),
        marker="airfoil",
        def_template=str(def_template),
        primal_template=str(primal_template),
        adjoint_template=str(adjoint_template),
        workdir=str(tmp_path / "run"),
        print_optimizer_table=False,
        **kwargs,
    )


def _install_fake_scipy_minimize(monkeypatch, minimize):
    scipy_module = types.ModuleType("scipy")
    optimize_module = types.ModuleType("scipy.optimize")
    optimize_module.minimize = minimize
    monkeypatch.setitem(sys.modules, "scipy", scipy_module)
    monkeypatch.setitem(sys.modules, "scipy.optimize", optimize_module)


@pytest.mark.parametrize(
    "gnorm_raw,history,expected,reason",
    [
        (math.nan, [], True, "nonfinite_raw_gradient"),
        (1.0, [], False, "insufficient_history"),
        (50.0, [1.0, 1.0], False, "insufficient_history"),
        (10.0, [1.0, 1.0, 1.0], False, "ok"),
        (50.0, [1.0, 1.0, 1.0], False, "ok"),
        (100.01, [1.0, 1.0, 1.0], True, "raw_gradient_explosion"),
    ],
)
def test_raw_gradient_guard_policy(gnorm_raw, history, expected, reason):
    triggered, info = gradient_guard_triggered(
        {"gnorm_raw": gnorm_raw},
        history,
    )

    assert triggered is expected
    assert info["reason"] == reason


def test_small_beta_does_not_mask_raw_gradient_explosion():
    entry = {
        "gnorm_raw": 101.0,
        "gnorm_opt": 0.0303,
        "beta_eff": 0.0003,
    }

    triggered, info = gradient_guard_triggered(entry, [1.0, 1.0, 1.0])

    assert triggered is True
    assert info["ratio"] == pytest.approx(101.0)


def _clip_entry(beta=1.0, objective=0.9, gnorm_raw=1.0, eval_id=1):
    return {
        "eval_id": eval_id,
        "objective": objective,
        "beta_eff": beta,
        "was_clipped": beta < 1.0,
        "gnorm_raw": gnorm_raw,
        "gnorm_opt": beta * gnorm_raw,
        "evaluated_x": [0.0, 0.0],
        "modes_file": None,
    }


def _safe_clip_reference(objective=1.0, eval_id=0):
    return _clip_entry(beta=1.0, objective=objective, gnorm_raw=1.0, eval_id=eval_id)


def test_trust_clip_classifies_normal_unclipped_evaluation():
    classification, diagnostics = classify_clipped_trial(
        _clip_entry(beta=1.0),
        [1.0],
        _safe_clip_reference(),
        _safe_clip_reference(),
        [],
    )

    assert classification == "not_clipped"
    assert diagnostics["clipped"] is False


def test_trust_clip_unclipped_status_remains_ok(tmp_path):
    driver = _make_driver(tmp_path, trust_clip_policy="ACCEPT_RESTART")

    assert driver._trust_clip_status("not_clipped") == "ok"


def test_trust_clip_classifies_benign_moderate_improving_clip():
    classification, diagnostics = classify_clipped_trial(
        _clip_entry(beta=0.636, objective=0.9),
        [1.0, 1.0, 1.0],
        _safe_clip_reference(),
        _safe_clip_reference(),
        [],
    )

    assert classification == "benign_clipped_legacy"
    assert diagnostics["significant_improvement"] is True


def test_trust_clip_moderate_beta_alone_does_not_hide_worsening():
    classification, diagnostics = classify_clipped_trial(
        _clip_entry(beta=0.8, objective=1.1),
        [1.0, 1.0, 1.0],
        _safe_clip_reference(),
        _safe_clip_reference(),
        [],
    )

    assert classification == "rejected_toxic_clip"
    assert "worsening_vs_best_safe" in diagnostics["toxic_reasons"]


def test_trust_clip_classifies_useful_severe_clip_for_restart():
    classification, diagnostics = classify_clipped_trial(
        _clip_entry(beta=0.4, objective=0.9),
        [1.0, 1.0, 1.0],
        _safe_clip_reference(),
        _safe_clip_reference(),
        [],
    )

    assert classification == "accepted_clipped_restart"
    assert diagnostics["toxic_reasons"] == []


def test_trust_clip_rejects_soft_raw_gradient_explosion():
    classification, diagnostics = classify_clipped_trial(
        _clip_entry(beta=0.8, objective=0.9, gnorm_raw=21.0),
        [1.0, 1.0, 1.0],
        _safe_clip_reference(),
        _safe_clip_reference(),
        [],
    )

    assert classification == "rejected_toxic_clip"
    assert "soft_raw_gradient_ratio" in diagnostics["toxic_reasons"]


def test_trust_clip_repeated_toxic_detection_uses_sliding_window():
    events = [
        {"toxic": True, "clipped": True},
        {"toxic": False, "clipped": False},
    ]
    classification, diagnostics = classify_clipped_trial(
        _clip_entry(beta=0.8, objective=1.1),
        [1.0, 1.0, 1.0],
        _safe_clip_reference(),
        _safe_clip_reference(),
        events,
    )

    assert classification == "rejected_toxic_clip"
    assert "toxic_clipped_repeated" in diagnostics["toxic_reasons"]


def test_trust_clip_repeated_weak_improvements_form_stagnation_plateau():
    events = [
        {"clipped": True, "weak_improvement": True, "toxic": False},
    ]
    classification, diagnostics = classify_clipped_trial(
        _clip_entry(beta=0.8, objective=1.0 - 1.0e-7),
        [1.0, 1.0, 1.0],
        _safe_clip_reference(),
        _safe_clip_reference(),
        events,
    )

    assert classification == "rejected_toxic_clip"
    assert "clipped_stagnation_plateau" in diagnostics["toxic_reasons"]


def test_trust_clip_single_weak_improvement_is_observed_without_rejection():
    classification, diagnostics = classify_clipped_trial(
        _clip_entry(beta=0.8, objective=1.0 - 1.0e-7),
        [1.0, 1.0, 1.0],
        _safe_clip_reference(),
        _safe_clip_reference(),
        [],
    )

    assert classification == "weak_clipped_progress"
    assert diagnostics["weak_improvement"] is True
    assert diagnostics["clipped_stagnation_plateau"] is False


def test_trust_clip_objective_ratios_use_objective_floor_not_gnorm_floor():
    entry = _clip_entry(beta=0.8, objective=-1.0e-13)
    best = _safe_clip_reference(objective=0.0)
    anchor = _safe_clip_reference(objective=0.0)

    _classification, diagnostics = classify_clipped_trial(
        entry,
        [1.0e-8, 1.0e-8, 1.0e-8],
        best,
        anchor,
        [],
        options={"objective_floor": 1.0e-6, "gnorm_floor": 1.0e-2},
    )
    _classification_changed, diagnostics_changed = classify_clipped_trial(
        entry,
        [1.0e-8, 1.0e-8, 1.0e-8],
        best,
        anchor,
        [],
        options={"objective_floor": 1.0e-6, "gnorm_floor": 1.0e-20},
    )

    assert diagnostics["improvement_rel"] == pytest.approx(1.0e-7)
    assert diagnostics["relative_worsening"] == pytest.approx(-1.0e-7)
    assert diagnostics["improvement_rel"] == pytest.approx(
        diagnostics_changed["improvement_rel"]
    )
    assert diagnostics["relative_worsening"] == pytest.approx(
        diagnostics_changed["relative_worsening"]
    )


def test_trust_clip_gnorm_reference_uses_gnorm_floor_not_objective_floor():
    entry = _clip_entry(beta=0.8, objective=0.9, gnorm_raw=1.0e-8)

    _classification, diagnostics = classify_clipped_trial(
        entry,
        [1.0e-20, 1.0e-20, 1.0e-20],
        _safe_clip_reference(),
        _safe_clip_reference(),
        [],
        options={"objective_floor": 1.0e-1, "gnorm_floor": 1.0e-6},
    )
    _classification_changed, diagnostics_changed = classify_clipped_trial(
        entry,
        [1.0e-20, 1.0e-20, 1.0e-20],
        _safe_clip_reference(),
        _safe_clip_reference(),
        [],
        options={"objective_floor": 1.0e-20, "gnorm_floor": 1.0e-6},
    )

    assert diagnostics["gnorm_reference"] == pytest.approx(1.0e-6)
    assert diagnostics["gnorm_ratio"] == pytest.approx(1.0e-2)
    assert diagnostics["gnorm_reference"] == pytest.approx(
        diagnostics_changed["gnorm_reference"]
    )
    assert diagnostics["gnorm_ratio"] == pytest.approx(
        diagnostics_changed["gnorm_ratio"]
    )


def test_trust_clip_cd_scale_objective_ratios_ignore_floor():
    entry = _clip_entry(beta=0.8, objective=0.0095)
    best = _safe_clip_reference(objective=0.0100)
    anchor = _safe_clip_reference(objective=0.0100)

    _classification_a, diagnostics_a = classify_clipped_trial(
        entry,
        [1.0, 1.0, 1.0],
        best,
        anchor,
        [],
        options={"objective_floor": 1.0e-12, "gnorm_floor": 1.0e-14},
    )
    _classification_b, diagnostics_b = classify_clipped_trial(
        entry,
        [1.0, 1.0, 1.0],
        best,
        anchor,
        [],
        options={"objective_floor": 1.0e-20, "gnorm_floor": 1.0e-14},
    )

    assert diagnostics_a["improvement_rel"] == pytest.approx(0.05)
    assert diagnostics_a["relative_worsening"] == pytest.approx(-0.05)
    assert diagnostics_a["improvement_rel"] == pytest.approx(
        diagnostics_b["improvement_rel"]
    )
    assert diagnostics_a["relative_worsening"] == pytest.approx(
        diagnostics_b["relative_worsening"]
    )


def test_guard_rollback_restores_last_safe_modes_and_rejects_bad_state(tmp_path):
    driver = _make_driver(
        tmp_path,
        gradient_guard=True,
        gradient_guard_factor=100.0,
        gradient_guard_min_history=3,
    )
    safe_modes = tmp_path / "safe_modes.json"
    bad_modes = tmp_path / "bad_modes.json"
    safe_spec = update_mode_coefficients(driver.mode_spec, [0.003, -0.004])
    bad_spec = update_mode_coefficients(driver.mode_spec, [0.02, 0.02])
    safe_modes.write_text(json.dumps(safe_spec, indent=2))
    bad_modes.write_text(json.dumps(bad_spec, indent=2))
    driver.optimized_modes_filename.parent.mkdir(parents=True, exist_ok=True)
    driver.optimized_modes_filename.write_text(bad_modes.read_text())

    safe_entry = {
        "eval_id": 4,
        "objective": 0.8,
        "requested_x": [0.003, -0.004],
        "evaluated_x": [0.003, -0.004],
        "beta_eff": 1.0,
        "was_clipped": False,
        "gnorm_raw": 1.0,
        "gnorm_opt": 1.0,
        "eval_dir": tmp_path / "eval_0004",
        "modes_file": safe_modes,
    }
    bad_entry = {
        "eval_id": 5,
        "objective": 0.1,
        "requested_x": [0.02, 0.02],
        "evaluated_x": [0.02, 0.02],
        "beta_eff": 0.001,
        "was_clipped": True,
        "gnorm_raw": 101.0,
        "gnorm_opt": 0.101,
        "eval_dir": tmp_path / "eval_0005",
        "modes_file": bad_modes,
    }
    driver.last_safe_entry = safe_entry
    driver.best_physical_entry = safe_entry
    driver.recent_safe_raw_gnorms.extend([1.0, 1.0, 1.0])

    with pytest.raises(GradientGuardStop) as exc_info:
        driver.register_gradient_entry(bad_entry)

    assert exc_info.value.last_safe_entry is safe_entry
    assert driver.last_safe_entry is safe_entry
    assert driver.best_physical_entry is safe_entry
    assert list(driver.recent_safe_raw_gnorms) == [1.0, 1.0, 1.0]
    assert json.loads(driver.optimized_modes_filename.read_text()) == json.loads(
        safe_modes.read_text()
    )


def test_gradient_entry_logs_raw_and_actual_optimizer_norm_separately(tmp_path):
    driver = _make_driver(
        tmp_path,
        opt_relax_factor=2.0,
        opt_gradient_factor=3.0,
    )
    paths = build_eval_paths(tmp_path / "eval_0000")
    entry = driver._gradient_entry(
        {
            "eval_id": 0,
            "objective": 1.0,
            "gradient": [3.0, 4.0],
            "coefficients": [0.0, 0.0],
        },
        paths,
        line_search_info={"line_search_beta": 0.01},
    )

    assert entry["gnorm_raw"] == pytest.approx(5.0)
    assert entry["gnorm_opt"] == pytest.approx(0.3)


def test_benign_clip_keeps_frozen_beta_optimizer_gradient_scaling(tmp_path):
    driver = _make_driver(
        tmp_path,
        opt_relax_factor=2.0,
        opt_gradient_factor=3.0,
        trust_clip_policy="ACCEPT_RESTART",
    )

    gradient = driver._optimizer_gradient_for_logging(
        [3.0, 4.0],
        {"line_search_beta": 0.6},
    )

    assert gradient == pytest.approx([10.8, 14.4])


def test_accepted_clipped_restart_is_not_best_safe_during_evaluation(tmp_path):
    driver = _make_driver(tmp_path, trust_clip_policy="ACCEPT_RESTART")
    old_safe = _safe_clip_reference(objective=1.0)
    accepted = _clip_entry(beta=0.4, objective=0.8, eval_id=2)
    driver._promote_safe_entry(old_safe)

    classification, _diagnostics = driver._classify_trust_clip_entry(accepted)

    assert classification == "accepted_clipped_restart"
    assert driver.best_safe_entry is old_safe
    assert driver.last_safe_entry is old_safe


def test_accepted_clipped_restart_becomes_best_safe_in_matching_callback(tmp_path):
    driver = _make_driver(tmp_path, trust_clip_policy="ACCEPT_RESTART")
    modes_file = tmp_path / "accepted_modes.json"
    modes_file.write_text(json.dumps(update_mode_coefficients(driver.mode_spec, [0.0, 0.0])))
    old_safe = _safe_clip_reference(objective=1.0)
    accepted = dict(
        _clip_entry(beta=0.4, objective=0.8, eval_id=2),
        modes_file=modes_file,
    )
    driver._promote_safe_entry(old_safe)
    classification, diagnostics = driver._classify_trust_clip_entry(accepted)
    requested = [0.0, 0.0]
    driver._trust_clip_by_requested_key[driver._pending_trust_clip_key(requested)] = {
        "classification": classification,
        "diagnostics": diagnostics,
        "entry": accepted,
        "result": {},
    }

    with pytest.raises(TrustClipStop) as exc_info:
        driver._trust_clip_callback(requested)

    assert classification == "accepted_clipped_restart"
    assert driver.best_safe_entry is accepted
    assert driver.last_safe_entry is accepted
    assert exc_info.value.rollback_entry is accepted


def test_internal_accepted_clipped_trial_never_promotes_without_callback(tmp_path):
    driver = _make_driver(tmp_path, trust_clip_policy="ACCEPT_RESTART")
    old_safe = _safe_clip_reference(objective=1.0)
    accepted = _clip_entry(beta=0.4, objective=0.8, eval_id=2)
    driver._promote_safe_entry(old_safe)
    classification, diagnostics = driver._classify_trust_clip_entry(accepted)
    requested = [0.0, 0.0]
    driver._trust_clip_by_requested_key[driver._pending_trust_clip_key(requested)] = {
        "classification": classification,
        "diagnostics": diagnostics,
        "entry": accepted,
        "result": {},
    }

    driver._trust_clip_callback([0.1, 0.1])

    assert driver.best_safe_entry is old_safe
    assert driver.last_safe_entry is old_safe
    assert driver._trust_clip_by_requested_key == {}


def test_rejected_toxic_clip_is_not_promoted_as_best_safe(tmp_path):
    driver = _make_driver(tmp_path, trust_clip_policy="ACCEPT_RESTART")
    old_safe = _safe_clip_reference(objective=1.0)
    toxic = _clip_entry(beta=0.8, objective=1.2, eval_id=2)
    driver._promote_safe_entry(old_safe)

    classification, _diagnostics = driver._classify_trust_clip_entry(toxic)

    assert classification == "rejected_toxic_clip"
    assert driver.best_safe_entry is old_safe
    assert driver.last_safe_entry is old_safe


def test_hard_gradient_guard_rollback_prefers_best_safe_over_last_safe(tmp_path):
    driver = _make_driver(tmp_path, gradient_guard_factor=100.0)
    best_file = tmp_path / "best.json"
    last_file = tmp_path / "last.json"
    best_file.write_text(json.dumps(update_mode_coefficients(driver.mode_spec, [0.001, 0.001])))
    last_file.write_text(json.dumps(update_mode_coefficients(driver.mode_spec, [0.002, 0.002])))
    best = dict(_safe_clip_reference(objective=0.5, eval_id=1), modes_file=best_file)
    last = dict(_safe_clip_reference(objective=0.8, eval_id=2), modes_file=last_file)
    driver.best_safe_entry = best
    driver.last_safe_entry = last
    driver.recent_safe_raw_gnorms.extend([1.0, 1.0, 1.0])

    with pytest.raises(GradientGuardStop) as exc_info:
        driver.register_gradient_entry(_clip_entry(gnorm_raw=101.0, eval_id=3))

    assert exc_info.value.last_safe_entry is best
    assert json.loads(driver.optimized_modes_filename.read_text()) == json.loads(
        best_file.read_text()
    )


def test_trust_clip_stop_requires_matching_requested_x_and_anchor(tmp_path):
    driver = _make_driver(tmp_path, trust_clip_policy="ACCEPT_RESTART")
    modes_file = tmp_path / "accepted_modes.json"
    modes_file.write_text(json.dumps(update_mode_coefficients(driver.mode_spec, [0.0, 0.0])))
    entry = dict(
        _clip_entry(beta=0.4, objective=0.8, eval_id=2),
        modes_file=modes_file,
    )
    driver._promote_safe_entry(_safe_clip_reference(objective=1.0))
    classification, diagnostics = driver._classify_trust_clip_entry(entry)
    assert classification == "accepted_clipped_restart"  # Trial classification itself does not stop.
    requested = [0.0, 0.0]
    driver._trust_clip_by_requested_key[driver._pending_trust_clip_key(requested)] = {
        "classification": classification,
        "diagnostics": diagnostics,
        "entry": entry,
        "result": {},
    }

    driver._trust_clip_callback([0.1, 0.1])  # unrelated accepted iterate
    driver._trust_clip_callback(requested)  # same requested x, different anchor

    assert driver._trust_clip_by_requested_key == {}
    assert driver.best_safe_entry["objective"] == pytest.approx(1.0)


def test_trust_clip_policy_off_preserves_legacy_callback_behavior(tmp_path):
    driver = _make_driver(tmp_path, trust_clip_policy="OFF")

    assert driver._trust_clip_enabled() is False
    driver._trust_clip_callback([0.0, 0.0])


def test_toxic_trust_clip_stop_rolls_callback_back_to_best_safe(tmp_path):
    driver = _make_driver(tmp_path, trust_clip_policy="ACCEPT_RESTART")
    best_file = tmp_path / "best_modes.json"
    best_file.write_text(json.dumps(update_mode_coefficients(driver.mode_spec, [0.001, -0.001])))
    best = dict(_safe_clip_reference(objective=0.5, eval_id=1), modes_file=best_file)
    driver.best_safe_entry = best
    driver.last_safe_entry = _safe_clip_reference(objective=0.8, eval_id=2)
    toxic = _clip_entry(beta=0.8, objective=1.2, eval_id=3)
    classification, diagnostics = driver._classify_trust_clip_entry(toxic)
    requested = [0.0, 0.0]
    driver._trust_clip_by_requested_key[driver._pending_trust_clip_key(requested)] = {
        "classification": classification,
        "diagnostics": diagnostics,
        "entry": toxic,
        "result": {},
    }

    with pytest.raises(TrustClipStop) as exc_info:
        driver._trust_clip_callback(requested)

    assert classification == "rejected_toxic_clip"
    assert exc_info.value.rollback_entry is best


def test_pending_trust_clip_key_includes_line_search_anchor(tmp_path):
    driver = _make_driver(tmp_path, trust_clip_policy="ACCEPT_RESTART")
    requested = [0.0, 0.0]

    key_at_initial_anchor = driver._pending_trust_clip_key(requested)
    driver._line_search_anchor_physical = [0.1, 0.0]
    key_at_new_anchor = driver._pending_trust_clip_key(requested)

    assert key_at_initial_anchor[0] == key_at_new_anchor[0]
    assert key_at_initial_anchor[1] != key_at_new_anchor[1]


def test_pending_trust_clip_evaluation_get_preserves_func_fprime_sharing(tmp_path, monkeypatch):
    driver = _make_driver(tmp_path, trust_clip_policy="ACCEPT_RESTART")
    calls = []

    def fake_evaluate(coefficients, line_search_info=None):
        calls.append(list(coefficients))
        return {
            "eval_id": len(calls),
            "objective": 1.0,
            "gradient": [1.0, 0.0],
            "coefficients": list(coefficients),
            "status": "ok",
        }

    monkeypatch.setattr(driver, "evaluate", fake_evaluate)

    first, _info = driver._evaluate_optimizer_variables([0.0, 0.0])
    pending_key = driver._pending_trust_clip_key([0.0, 0.0])
    driver._trust_clip_by_requested_key[pending_key] = {
        "classification": "weak_clipped_progress",
        "diagnostics": {},
        "entry": _clip_entry(beta=0.8, objective=1.0, eval_id=1),
        "result": first,
    }

    second, _info = driver._evaluate_optimizer_variables([0.0, 0.0])
    third, _info = driver._evaluate_optimizer_variables([0.0, 0.0])

    assert first is second is third
    assert calls == [[0.0, 0.0]]
    assert pending_key in driver._trust_clip_by_requested_key


def test_pending_trust_clip_anchor_mismatch_does_not_reuse_old_result(tmp_path, monkeypatch):
    driver = _make_driver(tmp_path, trust_clip_policy="ACCEPT_RESTART")
    calls = []

    def fake_evaluate(coefficients, line_search_info=None):
        calls.append(list(coefficients))
        return {
            "eval_id": len(calls),
            "objective": float(len(calls)),
            "gradient": [1.0, 0.0],
            "coefficients": list(coefficients),
            "status": "ok",
        }

    monkeypatch.setattr(driver, "evaluate", fake_evaluate)
    requested = [0.0, 0.0]
    old_key = driver._pending_trust_clip_key(requested)
    driver._trust_clip_by_requested_key[old_key] = {
        "classification": "accepted_clipped_restart",
        "diagnostics": {},
        "entry": _clip_entry(beta=0.4, objective=0.8, eval_id=1),
        "result": {"eval_id": 99, "objective": 99.0, "gradient": [0.0, 0.0]},
    }

    driver._line_search_anchor_physical = [0.1, 0.0]
    result, _info = driver._evaluate_optimizer_variables(requested)

    assert result["eval_id"] == 1
    assert calls == [[0.0, 0.0]]


def test_stale_pending_entry_cannot_trigger_callback_stop_after_anchor_change(tmp_path):
    driver = _make_driver(tmp_path, trust_clip_policy="ACCEPT_RESTART")
    old_safe = _safe_clip_reference(objective=1.0)
    driver._promote_safe_entry(old_safe)
    accepted = _clip_entry(beta=0.4, objective=0.8, eval_id=2)
    requested = [0.0, 0.0]
    driver._trust_clip_by_requested_key[driver._pending_trust_clip_key(requested)] = {
        "classification": "accepted_clipped_restart",
        "diagnostics": {},
        "entry": accepted,
        "result": {},
    }
    driver._line_search_anchor_physical = [0.1, 0.0]

    driver._trust_clip_callback(requested)

    assert driver.best_safe_entry is old_safe
    assert driver._trust_clip_by_requested_key == {}


def test_stale_toxic_pending_entry_cannot_trigger_callback_stop_after_anchor_change(tmp_path):
    driver = _make_driver(tmp_path, trust_clip_policy="ACCEPT_RESTART")
    old_safe = _safe_clip_reference(objective=1.0)
    driver._promote_safe_entry(old_safe)
    requested = [0.0, 0.0]
    driver._trust_clip_by_requested_key[driver._pending_trust_clip_key(requested)] = {
        "classification": "rejected_toxic_clip",
        "diagnostics": {"toxic_reasons": ["worsening_vs_best_safe"]},
        "entry": _clip_entry(beta=0.8, objective=1.2, eval_id=2),
        "result": {},
    }
    driver._line_search_anchor_physical = [0.1, 0.0]

    driver._trust_clip_callback(requested)

    assert driver.best_safe_entry is old_safe
    assert driver._trust_clip_by_requested_key == {}


def test_pending_map_prunes_stale_entries_without_clearing_current_anchor(tmp_path):
    driver = _make_driver(tmp_path, trust_clip_policy="ACCEPT_RESTART")
    requested = [0.0, 0.0]
    old_key = driver._pending_trust_clip_key(requested)
    driver._line_search_anchor_physical = [0.1, 0.0]
    current_key = driver._pending_trust_clip_key(requested)
    driver._trust_clip_by_requested_key[old_key] = {"result": "old"}
    driver._trust_clip_by_requested_key[current_key] = {"result": "current"}

    driver._prune_stale_trust_clip_pending()

    assert old_key not in driver._trust_clip_by_requested_key
    assert driver._trust_clip_by_requested_key[current_key]["result"] == "current"


def test_weak_clipped_progress_is_best_safe_by_objective_but_not_recent_safe(tmp_path):
    driver = _make_driver(tmp_path, trust_clip_policy="ACCEPT_RESTART")
    old_safe = _safe_clip_reference(objective=1.0)
    weak = _clip_entry(beta=0.8, objective=1.0 - 1.0e-7, eval_id=2)
    driver._promote_safe_entry(old_safe)
    classification, diagnostics = driver._classify_trust_clip_entry(weak)

    driver._promote_safe_entry(weak, update_recent_raw_gnorm=False)

    assert classification == "weak_clipped_progress"
    assert "weak_clipped_progress" in diagnostics["accepted_reasons"]
    assert driver.best_safe_entry is weak
    assert list(driver.recent_safe_raw_gnorms) == [1.0]


def test_weak_clipped_progress_not_best_safe_when_objective_is_not_lower(tmp_path):
    driver = _make_driver(tmp_path, trust_clip_policy="ACCEPT_RESTART")
    old_safe = _safe_clip_reference(objective=1.0)
    weak = _clip_entry(beta=0.8, objective=1.0, eval_id=2)
    driver._promote_safe_entry(old_safe)

    driver._promote_safe_entry(weak, update_recent_raw_gnorm=False)

    assert driver.best_safe_entry is old_safe
    assert driver.last_safe_entry is weak


def test_rejected_statuses_are_not_best_safe_or_cache_ok(tmp_path):
    driver = _make_driver(tmp_path, trust_clip_policy="ACCEPT_RESTART")
    old_safe = _safe_clip_reference(objective=1.0)
    driver._promote_safe_entry(old_safe)
    toxic = _clip_entry(beta=0.8, objective=1.2, eval_id=2)
    toxic_class, _diagnostics = driver._classify_trust_clip_entry(toxic)
    guard = _clip_entry(beta=1.0, objective=0.4, gnorm_raw=101.0, eval_id=3)
    driver.recent_safe_raw_gnorms.extend([1.0, 1.0, 1.0])

    with pytest.raises(GradientGuardStop):
        driver.register_gradient_entry(guard)

    assert toxic_class == "rejected_toxic_clip"
    assert driver.best_safe_entry is old_safe
    assert driver._best_ok_history_record() is None


def test_best_ok_history_allows_weak_but_not_rejected_or_failed_rows(tmp_path):
    driver = _make_driver(tmp_path, trust_clip_policy="ACCEPT_RESTART")
    for eval_id, objective, status in (
        (1, 0.1, "rejected_toxic_clip"),
        (2, 0.2, "rejected_gradient_guard"),
        (3, 0.3, "failed"),
        (4, 0.8, "ok"),
        (5, 0.5, "ok_clipped_weak"),
    ):
        driver._append_history_record(
            eval_id,
            objective,
            driver.initial_coefficients,
            [0.0] * len(driver.mode_ids),
            status,
        )

    assert driver._best_ok_history_record()["eval_id"] == 5


def test_slsqp_guard_stop_returns_success_with_restored_safe_design(
    tmp_path,
    monkeypatch,
):
    driver = _make_driver(
        tmp_path,
        gradient_guard_next_action="refine",
    )
    safe_modes = tmp_path / "safe_modes.json"
    safe_spec = update_mode_coefficients(driver.mode_spec, [0.003, -0.004])
    safe_modes.write_text(json.dumps(safe_spec, indent=2))
    safe_entry = {
        "eval_id": 4,
        "objective": 0.8,
        "evaluated_x": [0.003, -0.004],
        "modes_file": safe_modes,
    }
    bad_entry = {"eval_id": 5}
    stop = GradientGuardStop(
        safe_entry,
        bad_entry,
        {"reason": "raw_gradient_explosion", "ratio": 101.0},
    )

    def fake_minimize(*args, **kwargs):
        raise stop

    _install_fake_scipy_minimize(monkeypatch, fake_minimize)

    result = driver.optimize(maxiter=2)

    assert result["success"] is True
    assert result["status"] == "gradient_guard_stop"
    assert result["coefficients"] == pytest.approx([0.003, -0.004])
    assert result["refinement_triggered"] is True
    restored = json.loads(driver.optimized_modes_filename.read_text())
    assert [
        mode["coefficient"]
        for mode in restored["modes"]
        if mode.get("active", True) is not False
    ] == pytest.approx([0.003, -0.004])


def test_update_mode_coefficients_preserves_non_coefficient_fields():
    spec = _base_spec()
    original = copy.deepcopy(spec)

    updated = update_mode_coefficients(spec, [0.011, -0.012])

    assert spec == original
    assert updated["modes"][0]["coefficient"] == 0.011
    assert updated["modes"][1]["coefficient"] == -0.012
    assert updated["modes"][2]["coefficient"] == 0.5

    for before, after in zip(original["modes"], updated["modes"]):
        before_other = {key: value for key, value in before.items() if key != "coefficient"}
        after_other = {key: value for key, value in after.items() if key != "coefficient"}
        assert before_other == after_other


def test_active_coefficient_vector_extraction_skips_inactive_modes():
    spec = _base_spec()

    assert active_mode_ids(spec) == ["upper_a", "lower_b"]
    assert active_coefficient_vector(spec) == [0.001, -0.002]


def test_frozen_modes_contribute_to_geometry_but_not_design_vectors():
    spec = _base_spec()
    spec["modes"][1]["frozen"] = True

    assert active_mode_ids(spec) == ["upper_a"]
    assert active_coefficient_vector(spec) == [0.001]
    assert active_bounds(spec, default_bounds=(-0.01, 0.01)) == [(-0.02, 0.03)]

    displacement, values_by_id = evaluate_normal_displacement(
        spec,
        [0.5],
        ["lower"],
    )
    assert "lower_b" in values_by_id
    assert displacement[0] == pytest.approx(-0.002 * values_by_id["lower_b"][0])

    updated = update_mode_coefficients(spec, [0.011])
    assert updated["modes"][0]["coefficient"] == 0.011
    assert updated["modes"][1]["coefficient"] == -0.002
    assert updated["modes"][1]["frozen"] is True
    assert updated["modes"][2]["coefficient"] == 0.5


def test_geometry_probe_uses_design_modes_when_frozen_modes_exist(tmp_path, monkeypatch):
    spec = _base_spec()
    spec["modes"][1]["frozen"] = True

    def fake_run_command(command, cwd, log_file, show_command=False, stream_output=False, stage=None):
        del command, log_file, show_command, stream_output, stage
        cwd = Path(cwd)
        cwd.mkdir(parents=True, exist_ok=True)
        with (cwd / "bspline_surface_metadata.csv").open("w", newline="") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=[
                    "node_id",
                    "x",
                    "y",
                    "x_over_c",
                    "side",
                    "normal_x",
                    "normal_y",
                    "weight",
                    "deformed_x",
                    "deformed_y",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "node_id": 0,
                    "x": 0.5,
                    "y": 0.1,
                    "x_over_c": 0.5,
                    "side": "upper",
                    "normal_x": 0.0,
                    "normal_y": 1.0,
                    "weight": 1.0,
                    "deformed_x": 0.5,
                    "deformed_y": 0.1,
                }
            )
            writer.writerow(
                {
                    "node_id": 1,
                    "x": 0.5,
                    "y": -0.1,
                    "x_over_c": 0.5,
                    "side": "lower",
                    "normal_x": 0.0,
                    "normal_y": -1.0,
                    "weight": 1.0,
                    "deformed_x": 0.5,
                    "deformed_y": -0.1,
                }
            )

    monkeypatch.setattr(bspline_driver_module, "run_command", fake_run_command)
    driver = _make_driver(tmp_path, spec=spec, opt_line_search_bound=0.1)

    metadata, basis_matrix = driver._probe_geometry_aware_bounds()
    driver._configure_line_search_bound()

    assert len(metadata) == 2
    assert basis_matrix.shape == (2, 1)
    assert driver._line_search_basis_matrix.shape == (2, 1)
    assert driver.mode_ids == ["upper_a"]


def test_bounds_extraction_uses_mode_bounds_and_default():
    spec = _base_spec()

    assert active_bounds(spec, default_bounds=(-0.01, 0.01)) == [
        (-0.02, 0.03),
        (-0.01, 0.01),
    ]


def test_opt_bound_upper_lower_override_mode_json_bounds(tmp_path):
    driver = _make_driver(
        tmp_path,
        opt_bound_lower=-0.01,
        opt_bound_upper=0.01,
    )

    assert driver.bounds == [(-0.01, 0.01), (-0.01, 0.01)]
    assert driver.original_bounds == [(-0.01, 0.01), (-0.01, 0.01)]


def test_symmetry_mapping_counts_design_variables_for_paired_modes():
    spec = _paired_spec(npairs=4)

    none_vars, none_warnings = build_reduced_variables(spec, "NONE")
    equal_vars, equal_warnings = build_reduced_variables(spec, "NORMAL_EQUAL")
    opposite_vars, opposite_warnings = build_reduced_variables(spec, "NORMAL_OPPOSITE")

    assert len(none_vars) == 8
    assert len(equal_vars) == 4
    assert len(opposite_vars) == 4
    assert none_warnings == []
    assert equal_warnings == []
    assert opposite_warnings == []


def test_symmetry_mapping_pairs_clamped_modes_by_basis_index_not_support():
    spec = _paired_clamped_spec()

    equal_vars, equal_warnings = build_reduced_variables(spec, "NORMAL_EQUAL")

    assert len(equal_vars) == 4
    assert equal_warnings == []
    assert [variable.mode_ids for variable in equal_vars] == [
        ("upper_clamped_i000", "lower_clamped_i000"),
        ("upper_clamped_i001", "lower_clamped_i001"),
        ("upper_clamped_i002", "lower_clamped_i002"),
        ("upper_clamped_i003", "lower_clamped_i003"),
    ]


def test_symmetry_expansion_and_gradient_collapse():
    spec = _paired_spec(npairs=1)
    equal_vars, _warnings = build_reduced_variables(spec, "NORMAL_EQUAL")
    opposite_vars, _warnings = build_reduced_variables(spec, "NORMAL_OPPOSITE")

    assert expand_reduced_coefficients([0.01], equal_vars, 2) == pytest.approx([0.01, 0.01])
    assert expand_reduced_coefficients([0.01], opposite_vars, 2) == pytest.approx([0.01, -0.01])
    assert collapse_full_gradient([2.0, 3.0], equal_vars) == pytest.approx([5.0])
    assert collapse_full_gradient([2.0, 3.0], opposite_vars) == pytest.approx([-1.0])

    jac = np.asarray([[2.0, 3.0], [5.0, 7.0]], dtype=float)
    assert np.asarray(collapse_full_jacobian(jac, equal_vars)) == pytest.approx(
        np.asarray([[5.0], [12.0]])
    )
    assert np.asarray(collapse_full_jacobian(jac, opposite_vars)) == pytest.approx(
        np.asarray([[-1.0], [-2.0]])
    )


def test_symmetry_driver_uses_reduced_bounds_and_initial_coefficients(tmp_path):
    driver_equal = _make_driver(
        tmp_path / "equal",
        spec=_paired_spec(npairs=4),
        symmetry_coupling="NORMAL_EQUAL",
    )
    driver_opposite = _make_driver(
        tmp_path / "opposite",
        spec=_paired_spec(npairs=4, coupling_consistent=False),
        symmetry_coupling="NORMAL_OPPOSITE",
    )

    assert len(driver_equal.mode_ids) == 8
    assert len(driver_equal.reduced_variable_ids) == 4
    assert driver_equal.initial_reduced_coefficients == pytest.approx([0.001, 0.002, 0.003, 0.004])
    assert driver_equal.reduced_bounds[0] == pytest.approx((-0.01, 0.02))
    assert len(driver_opposite.reduced_variable_ids) == 4
    assert driver_opposite.initial_reduced_coefficients == pytest.approx([0.001, 0.002, 0.003, 0.004])
    assert driver_opposite.reduced_bounds[0] == pytest.approx((-0.02, 0.01))


def test_slsqp_uses_reduced_variables_for_symmetry_coupling(tmp_path, monkeypatch):
    captured = {}

    def fake_evaluate(self, coefficients, line_search_info=None):
        captured["coefficients"] = list(coefficients)
        return {
            "eval_id": 0,
            "objective": 1.0,
            "gradient": [1.0] * 8,
            "coefficients": list(coefficients),
            "status": "ok",
        }

    def fake_minimize(fun, x0, jac, bounds, constraints, method, callback, options):
        captured["x0"] = list(x0)
        captured["bounds"] = list(bounds)
        captured["jac"] = jac(list(x0))
        return types.SimpleNamespace(
            x=list(x0),
            fun=fun(list(x0)),
            success=True,
            message="ok",
            status=0,
            nit=0,
            nfev=1,
            njev=1,
        )

    _install_fake_scipy_minimize(monkeypatch, fake_minimize)
    monkeypatch.setattr(BSplineSU2Driver, "evaluate", fake_evaluate)
    driver = _make_driver(
        tmp_path,
        spec=_paired_spec(npairs=4),
        symmetry_coupling="NORMAL_EQUAL",
    )

    result = driver.optimize(maxiter=1)

    assert len(captured["x0"]) == 4
    assert len(captured["bounds"]) == 4
    assert captured["jac"] == pytest.approx([2.0, 2.0, 2.0, 2.0])
    assert len(result["coefficients"]) == 8
    assert result["coefficients"][0] == pytest.approx(result["coefficients"][1])


def test_history_keeps_full_coefficients_and_adds_reduced_coefficients_when_coupled(tmp_path):
    driver = _make_driver(
        tmp_path,
        spec=_paired_spec(npairs=1),
        symmetry_coupling="NORMAL_EQUAL",
    )

    driver._append_history_record(
        0,
        1.0,
        [0.02, 0.02],
        [2.0, 3.0],
        "ok",
    )

    with open(driver.optimization_history_filename, "r", newline="") as fp:
        row = next(csv.DictReader(fp))
    assert row["coeff__upper_0"] == "0.02"
    assert row["coeff__lower_0"] == "0.02"
    reduced_fields = [field for field in row if field.startswith("reduced_coeff__")]
    assert len(reduced_fields) == 1
    assert float(row[reduced_fields[0]]) == pytest.approx(0.02)
    reduced_grad_fields = [field for field in row if field.startswith("reduced_grad__")]
    assert len(reduced_grad_fields) == 1
    assert float(row[reduced_grad_fields[0]]) == pytest.approx(5.0)


def test_relax_factor_scales_slsqp_variables_bounds_and_ftol(tmp_path, monkeypatch):
    captured = {}

    def fake_minimize(fun, x0, jac, bounds, constraints, method, callback, options):
        captured["x0"] = list(x0)
        captured["bounds"] = list(bounds)
        captured["method"] = method
        captured["options"] = dict(options)
        return types.SimpleNamespace(
            x=list(x0),
            fun=2.0,
            success=True,
            message="ok",
            status=0,
            nit=0,
            nfev=0,
            njev=0,
        )

    _install_fake_scipy_minimize(monkeypatch, fake_minimize)
    driver = _make_driver(
        tmp_path,
        opt_relax_factor=1000.0,
        opt_gradient_factor=2.0,
        opt_accuracy=1.0e-8,
    )

    result = driver.optimize(maxiter=4)

    assert captured["method"] == "SLSQP"
    assert captured["x0"] == pytest.approx([1.0e-6, -2.0e-6])
    assert [lower for lower, _upper in captured["bounds"]] == pytest.approx(
        [-2.0e-5, -1.0e-5]
    )
    assert [upper for _lower, upper in captured["bounds"]] == pytest.approx(
        [3.0e-5, 1.0e-5]
    )
    assert captured["options"]["ftol"] == pytest.approx(2.0e-8)
    assert result["coefficients"] == pytest.approx([0.001, -0.002])


def test_optimizer_variables_convert_to_physical_and_scale_gradient(tmp_path, monkeypatch):
    captured = {}

    def fake_evaluate(self, coefficients, line_search_info=None):
        captured["coefficients"] = list(coefficients)
        return {
            "eval_id": 0,
            "objective": 4.0,
            "gradient": [2.0, -4.0],
            "coefficients": list(coefficients),
            "status": "ok",
        }

    def fake_minimize(fun, x0, jac, bounds, constraints, method, callback, options):
        u_trial = [0.002, -0.003]
        captured["objective_seen_by_slsqp"] = fun(u_trial)
        captured["gradient_seen_by_slsqp"] = jac(u_trial)
        return types.SimpleNamespace(
            x=u_trial,
            fun=captured["objective_seen_by_slsqp"],
            success=True,
            message="ok",
            status=0,
            nit=1,
            nfev=1,
            njev=1,
        )

    _install_fake_scipy_minimize(monkeypatch, fake_minimize)
    monkeypatch.setattr(BSplineSU2Driver, "evaluate", fake_evaluate)
    driver = _make_driver(
        tmp_path,
        opt_relax_factor=100.0,
        opt_gradient_factor=3.0,
    )

    result = driver.optimize(maxiter=1)

    assert captured["coefficients"] == pytest.approx([0.2, -0.3])
    assert captured["objective_seen_by_slsqp"] == pytest.approx(12.0)
    assert captured["gradient_seen_by_slsqp"] == pytest.approx([600.0, -1200.0])
    assert result["coefficients"] == pytest.approx([0.2, -0.3])


def test_last_eval_cache_reuses_immediate_nonclipped_physical_design(tmp_path, monkeypatch):
    driver = _make_driver(tmp_path)
    calls = []

    def fake_evaluate(coefficients, line_search_info=None):
        calls.append(list(coefficients))
        return {
            "eval_id": len(calls),
            "objective": 1.0,
            "gradient": [1.0, 0.0],
            "coefficients": list(coefficients),
            "status": "ok",
        }

    monkeypatch.setattr(driver, "evaluate", fake_evaluate)

    first, first_info = driver._evaluate_optimizer_variables([0.0, 0.0])
    second, second_info = driver._evaluate_optimizer_variables([0.0, 0.0])

    assert first is second
    assert calls == [[0.0, 0.0]]
    assert first_info.get("cache_hit", False) is False
    assert second_info["cache_hit"] is True
    assert second_info["last_eval_cache_hit"] is True


def test_last_eval_cache_ignores_clipped_physical_design(tmp_path, monkeypatch):
    spec = _base_spec()
    spec["modes"][0]["coefficient"] = 0.0
    spec["modes"][1]["coefficient"] = 0.0
    monkeypatch.setattr(
        BSplineSU2Driver,
        "_probe_geometry_aware_bounds",
        lambda self: ([], [[1.0, 0.0], [0.0, 1.0]]),
    )
    driver = _make_driver(tmp_path, spec=spec, opt_line_search_bound=0.25)
    driver._configure_line_search_bound()
    calls = []

    def fake_evaluate(coefficients, line_search_info=None):
        calls.append(list(coefficients))
        return {
            "eval_id": len(calls),
            "objective": 1.0,
            "gradient": [1.0, 0.0],
            "coefficients": list(coefficients),
            "status": "ok",
        }

    monkeypatch.setattr(driver, "evaluate", fake_evaluate)

    _first, first_info = driver._evaluate_optimizer_variables([1.0, 0.0])
    _second, second_info = driver._evaluate_optimizer_variables([1.0, 0.0])

    assert first_info["line_search_beta"] == pytest.approx(0.25)
    assert second_info["line_search_beta"] == pytest.approx(0.25)
    assert calls == [[0.25, 0.0], [0.25, 0.0]]
    assert second_info.get("last_eval_cache_hit", False) is False


def test_last_eval_cache_ignores_special_trust_clip_classes(tmp_path, monkeypatch):
    driver = _make_driver(tmp_path)
    driver._last_eval_physical_key = cache_key([0.0, 0.0], driver.cache_tol)
    driver._last_eval_result = {
        "trust_clip_class": "accepted_clipped_restart",
        "objective": 1.0,
        "gradient": [1.0, 0.0],
    }
    driver._last_eval_info = {"line_search_beta": 1.0}
    calls = []

    def fake_evaluate(coefficients, line_search_info=None):
        calls.append(list(coefficients))
        return {
            "eval_id": len(calls),
            "objective": 0.9,
            "gradient": [1.0, 0.0],
            "coefficients": list(coefficients),
            "status": "ok",
        }

    monkeypatch.setattr(driver, "evaluate", fake_evaluate)

    result, info = driver._evaluate_optimizer_variables([0.0, 0.0])

    assert result["objective"] == pytest.approx(0.9)
    assert calls == [[0.0, 0.0]]
    assert info.get("last_eval_cache_hit", False) is False


def test_slsqp_fun_does_not_record_trigger_on_last_eval_cache_hit(tmp_path, monkeypatch):
    driver = _make_driver(tmp_path)
    recorded = []
    calls = []

    def fake_record(project, objective):
        recorded.append(float(objective))

    def fake_evaluate(coefficients, line_search_info=None):
        calls.append(list(coefficients))
        return {
            "eval_id": len(calls),
            "objective": 2.0,
            "gradient": [1.0, 0.0],
            "coefficients": list(coefficients),
            "status": "ok",
        }

    def fake_minimize(fun, x0, jac, bounds, constraints, method, callback, options):
        first = fun(list(x0))
        second = fun(list(x0))
        jac(list(x0))
        return types.SimpleNamespace(
            x=list(x0),
            fun=first,
            success=True,
            message=f"second={second}",
            status=0,
            nit=0,
            nfev=2,
            njev=1,
        )

    _install_fake_scipy_minimize(monkeypatch, fake_minimize)
    monkeypatch.setattr(driver, "evaluate", fake_evaluate)
    from SU2.opt.bspline_driver import driver as _bspline_driver_module
    monkeypatch.setattr(
        _bspline_driver_module, "record_objective_and_check", fake_record
    )

    driver.optimize(maxiter=1)

    assert calls == [[0.001, -0.002]]
    assert recorded == [2.0]


def test_line_search_bound_limits_physical_normal_jump(tmp_path, monkeypatch):
    spec = _base_spec()
    spec["modes"][0]["coefficient"] = 0.0
    spec["modes"][1]["coefficient"] = 0.0
    monkeypatch.setattr(
        BSplineSU2Driver,
        "_probe_geometry_aware_bounds",
        lambda self: ([], [[2.0, 0.0], [0.0, 1.0]]),
    )
    driver = _make_driver(tmp_path, spec=spec, opt_line_search_bound=0.1)
    driver._configure_line_search_bound()

    applied, info = driver._apply_line_search_bound([1.0, 0.0])

    assert applied == pytest.approx([0.05, 0.0])
    assert info["line_search_beta"] == pytest.approx(0.05)
    assert info["line_search_maxdiff"] == pytest.approx(2.0)
    assert info["line_search_limited"] == 1


def test_history_stores_applied_physical_coefficients_for_scaled_limited_eval(
    tmp_path,
    monkeypatch,
):
    spec = _base_spec()
    spec["modes"][0]["coefficient"] = 0.0
    spec["modes"][1]["coefficient"] = 0.0
    monkeypatch.setattr(
        BSplineSU2Driver,
        "_probe_geometry_aware_bounds",
        lambda self: ([], [[1.0, 0.0], [0.0, 1.0]]),
    )
    driver = _make_driver(
        tmp_path,
        spec=spec,
        opt_relax_factor=10.0,
        opt_line_search_bound=0.25,
    )
    driver._configure_line_search_bound()

    def fake_evaluate(coefficients, line_search_info=None):
        driver._append_history_record(
            0,
            9.0,
            coefficients,
            [0.0, 0.0],
            "ok",
            line_search_info=line_search_info,
        )
        return {
            "eval_id": 0,
            "objective": 9.0,
            "gradient": [0.0, 0.0],
            "coefficients": list(coefficients),
            "status": "ok",
        }

    monkeypatch.setattr(driver, "evaluate", fake_evaluate)

    result, info = driver._evaluate_optimizer_variables([0.1, 0.0])

    assert result["coefficients"] == pytest.approx([0.25, 0.0])
    assert info["line_search_beta"] == pytest.approx(0.25)
    with open(driver.optimization_history_filename, "r", newline="") as fp:
        rows = list(csv.DictReader(fp))
    assert len(rows) == 1
    assert float(rows[0]["coeff__upper_a"]) == pytest.approx(0.25)
    assert float(rows[0]["coeff__upper_a"]) != pytest.approx(0.1)
    assert float(rows[0]["line_search_beta"]) == pytest.approx(0.25)
    assert float(rows[0]["line_search_maxdiff"]) == pytest.approx(1.0)
    assert rows[0]["line_search_limited"] == "1"


def test_thickness_config_parser_accepts_progressive_keys(tmp_path):
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    config = config_dir / "bspline_opt.cfg"
    config.write_text(
        "PROGRESSIVE_THICKNESS_CONSTRAINT= YES\n"
        "PROGRESSIVE_THICKNESS_REF_MESH= ref.su2\n"
        "PROGRESSIVE_THICKNESS_MARKER= airfoil\n"
        "PROGRESSIVE_THICKNESS_NPOINTS= 11\n"
        "PROGRESSIVE_THICKNESS_XMIN= 0.1\n"
        "PROGRESSIVE_THICKNESS_XMAX= 0.9\n"
        "PROGRESSIVE_THICKNESS_X_STATIONS= (0.25, 0.5, 0.75)\n"
        "PROGRESSIVE_THICKNESS_MARGIN= 0.001\n"
        "PROGRESSIVE_THICKNESS_FD_EPS= 1e-7\n"
        "PROGRESSIVE_THICKNESS_GRADIENT= ANALYTIC\n"
        "PROGRESSIVE_THICKNESS_CACHE_FILE= ref_cache.npz\n"
        "PROGRESSIVE_THICKNESS_DOMAIN_MODE= FULL\n"
        "PROGRESSIVE_THICKNESS_SYMMETRY_Y= 0.0\n"
    )

    values = parse_optimizer_config(config)
    values["_optimizer_config_filename"] = str(config.resolve())
    options = thickness_options_from_config(values)

    assert options["PROGRESSIVE_THICKNESS_CONSTRAINT"] is True
    assert options["PROGRESSIVE_THICKNESS_MARKER"] == "airfoil"
    assert options["PROGRESSIVE_THICKNESS_GRADIENT"] == "ANALYTIC"
    assert options["PROGRESSIVE_THICKNESS_REF_MESH"] == str(config_dir / "ref.su2")
    assert options["PROGRESSIVE_THICKNESS_CACHE_FILE"] == str(config_dir / "ref_cache.npz")


def test_disabled_thickness_constraint_passes_no_slsqp_constraints(tmp_path, monkeypatch):
    captured = {}

    def fake_minimize(fun, x0, jac, bounds, constraints, method, callback, options):
        captured["constraints"] = list(constraints)
        return types.SimpleNamespace(
            x=list(x0),
            fun=1.0,
            success=True,
            message="ok",
            status=0,
            nit=0,
            nfev=0,
            njev=0,
        )

    _install_fake_scipy_minimize(monkeypatch, fake_minimize)
    driver = _make_driver(
        tmp_path,
        spec=_single_mode_spec(),
        thickness_options={"PROGRESSIVE_THICKNESS_CONSTRAINT": "NO"},
    )

    driver.optimize(maxiter=1)

    assert captured["constraints"] == []


def test_airfoil_area_metric_value_and_gradient_for_vertical_bspline_mode():
    metadata = [
        {
            "x": 0.0,
            "y": 0.0,
            "normal_x": 0.0,
            "normal_y": 1.0,
            "deform_dir_x": 0.0,
            "deform_dir_y": 1.0,
        },
        {
            "x": 1.0,
            "y": 0.0,
            "normal_x": 0.0,
            "normal_y": 1.0,
            "deform_dir_x": 0.0,
            "deform_dir_y": 1.0,
        },
        {
            "x": 1.0,
            "y": 1.0,
            "normal_x": 0.0,
            "normal_y": 1.0,
            "deform_dir_x": 0.0,
            "deform_dir_y": 1.0,
        },
        {
            "x": 0.0,
            "y": 1.0,
            "normal_x": 0.0,
            "normal_y": 1.0,
            "deform_dir_x": 0.0,
            "deform_dir_y": 1.0,
        },
    ]
    basis = np.asarray([[0.0], [0.0], [1.0], [1.0]], dtype=float)
    metric = BSplineAirfoilAreaMetric(
        metadata,
        basis,
        ["top"],
        closed=True,
    )

    value, gradient = metric.value_and_gradient([0.2])

    assert value == pytest.approx(1.2)
    assert gradient.tolist() == pytest.approx([1.0])


def _area_probe():
    metadata = [
        {
            "x": 0.0,
            "y": 0.0,
            "normal_x": 0.0,
            "normal_y": 1.0,
            "deform_dir_x": 0.0,
            "deform_dir_y": 1.0,
        },
        {
            "x": 1.0,
            "y": 0.0,
            "normal_x": 0.0,
            "normal_y": 1.0,
            "deform_dir_x": 0.0,
            "deform_dir_y": 1.0,
        },
        {
            "x": 1.0,
            "y": 1.0,
            "normal_x": 0.0,
            "normal_y": 1.0,
            "deform_dir_x": 0.0,
            "deform_dir_y": 1.0,
        },
        {
            "x": 0.0,
            "y": 1.0,
            "normal_x": 0.0,
            "normal_y": 1.0,
            "deform_dir_x": 0.0,
            "deform_dir_y": 1.0,
        },
    ]
    basis = np.asarray([[0.0], [0.0], [1.0], [1.0]], dtype=float)
    return metadata, basis


def test_geometry_constraint_gradient_config_is_parsed():
    options = fixed_driver_options_from_config(
        {"BSPLINE_GEOMETRY_CONSTRAINT_GRADIENT": "SU2_GEO"}
    )

    assert options["geometry_constraint_gradient"] == "SU2_GEO"


def test_invalid_geometry_constraint_gradient_mode_fails(tmp_path):
    with pytest.raises(
        BSplineSU2DriverError,
        match="BSPLINE_GEOMETRY_CONSTRAINT_GRADIENT",
    ):
        _make_driver(tmp_path, geometry_constraint_gradient="BAD")


def test_airfoil_area_constraint_requires_both_surface_mode(tmp_path):
    spec = _single_mode_spec()
    spec["modes"][0]["side"] = "upper"

    with pytest.raises(
        BSplineSU2DriverError,
        match="AIRFOIL_AREA requires BSPLINE_SURFACE_MODE=BOTH",
    ):
        _make_driver(
            tmp_path,
            spec=spec,
            surface_mode="UPPER",
            native_constraints="(AIRFOIL_AREA>0.1)*1.0",
        )


def test_geometry_metric_builder_caches_area_and_ignores_unsupported(tmp_path, monkeypatch):
    driver = _make_driver(tmp_path, spec=_single_mode_spec())
    monkeypatch.setattr(driver, "_probe_geometry_aware_bounds", _area_probe)
    monkeypatch.setattr(driver, "_marker_closed", lambda marker: True)

    metric = driver._geometry_metric_for("AIRFOIL_AREA")
    cached = driver._geometry_metric_for("AIRFOIL_AREA")
    value, gradient = metric.value_and_gradient([0.2])

    assert metric is cached
    assert value == pytest.approx(1.2)
    assert gradient.tolist() == pytest.approx([1.0])
    assert driver._geometry_metric_for("AIRFOIL_CHORD") is None


def test_airfoil_area_auto_vertical_both_uses_analytic_backend(
    tmp_path,
    monkeypatch,
):
    calls = []
    driver = _make_driver(
        tmp_path,
        spec=_single_mode_spec(),
        deformation_direction_mode="VERTICAL",
    )
    monkeypatch.setattr(driver, "_probe_geometry_aware_bounds", _area_probe)
    monkeypatch.setattr(driver, "_marker_closed", lambda marker: True)
    monkeypatch.setattr(
        bspline_driver_module,
        "run_command",
        lambda *args, **kwargs: calls.append(kwargs.get("stage")),
    )

    result = driver.evaluate_geometry_constraint_function([0.2], "AIRFOIL_AREA")

    assert result["source"] == "ANALYTIC"
    assert result["eval_id"] is None
    assert result["eval_dir"] is None
    assert result["value"] == pytest.approx(1.2)
    assert result["gradient"] == pytest.approx([1.0])
    assert "geometry_su2_geo" not in calls
    with open(driver.geometry_constraint_history_filename, "r", newline="") as fp:
        rows = list(csv.DictReader(fp))
    assert rows[-1]["function"] == "AIRFOIL_AREA"
    assert rows[-1]["source"] == "ANALYTIC"
    assert float(rows[-1]["value"]) == pytest.approx(1.2)


def test_geometry_constraint_uses_su2_geo_fd_backend(tmp_path, monkeypatch):
    calls = []

    def fake_run_command(command, cwd, log_file, show_command=False, stream_output=False, stage=None):
        cwd = Path(cwd)
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text("$ fake\n")
        calls.append((stage, list(command), cwd))
        eval_dir = cwd
        while eval_dir.name and not eval_dir.name.startswith("eval_"):
            eval_dir = eval_dir.parent
        if stage == "geometry_su2_def":
            (eval_dir / "deform" / "deformed_mesh.su2").write_text("mesh")
        if stage == "geometry_su2_geo":
            modes = json.loads((eval_dir / "modes_current.json").read_text())
            coefficient = float(modes["modes"][0]["coefficient"])
            value = 1.0 + 2.0 * coefficient
            (cwd / "of_func.csv").write_text(
                '"AIRFOIL_AREA"\n{:.16g}\n'.format(value)
            )

    monkeypatch.setattr(bspline_driver_module, "run_command", fake_run_command)
    driver = _make_driver(
        tmp_path,
        spec=_single_mode_spec(coefficient=0.25),
        geometry_fd_eps=1.0e-5,
        geometry_constraint_gradient="SU2_GEO",
    )

    result = driver.evaluate_geometry_constraint_function([0.25], "AIRFOIL_AREA")

    assert result["source"] == "SU2_GEO"
    assert result["value"] == pytest.approx(1.5)
    assert result["gradient"] == pytest.approx([2.0])
    assert [call[0] for call in calls].count("geometry_su2_geo") == 2
    geo_cfgs = list((Path(result["eval_dir"]) / "geometry").glob("geo.cfg"))
    assert geo_cfgs
    assert "GEO_PARAM= AIRFOIL_AREA" in geo_cfgs[0].read_text()
    assert "GEO_MODE= FUNCTION" in geo_cfgs[0].read_text()
    # SU2_GEO solves no adjoint: the geometry eval must not spawn a stray adjoint dir.
    assert not list(Path(result["eval_dir"]).glob("adjoint_*"))


def test_airfoil_area_auto_nonvertical_falls_back_to_su2_geo(tmp_path, monkeypatch):
    calls = []

    def fake_run_command(command, cwd, log_file, show_command=False, stream_output=False, stage=None):
        cwd = Path(cwd)
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        Path(log_file).write_text("$ fake\n")
        calls.append(stage)
        eval_dir = cwd
        while eval_dir.name and not eval_dir.name.startswith("eval_"):
            eval_dir = eval_dir.parent
        if stage == "geometry_su2_def":
            (eval_dir / "deform" / "deformed_mesh.su2").write_text("mesh")
        if stage == "geometry_su2_geo":
            (cwd / "of_func.csv").write_text('"AIRFOIL_AREA"\n1.0\n')

    monkeypatch.setattr(bspline_driver_module, "run_command", fake_run_command)
    driver = _make_driver(
        tmp_path,
        spec=_single_mode_spec(),
        deformation_direction_mode="NORMAL",
    )

    result = driver.evaluate_geometry_constraint_function([0.0], "AIRFOIL_AREA")

    assert result["source"] == "SU2_GEO"
    assert calls.count("geometry_su2_geo") == 2


def test_unsupported_geometry_constraint_uses_su2_geo(tmp_path, monkeypatch):
    calls = []

    def fake_run_command(command, cwd, log_file, show_command=False, stream_output=False, stage=None):
        cwd = Path(cwd)
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        Path(log_file).write_text("$ fake\n")
        calls.append(stage)
        eval_dir = cwd
        while eval_dir.name and not eval_dir.name.startswith("eval_"):
            eval_dir = eval_dir.parent
        if stage == "geometry_su2_def":
            (eval_dir / "deform" / "deformed_mesh.su2").write_text("mesh")
        if stage == "geometry_su2_geo":
            (cwd / "of_func.csv").write_text('"AIRFOIL_CHORD"\n1.0\n')

    monkeypatch.setattr(bspline_driver_module, "run_command", fake_run_command)
    driver = _make_driver(
        tmp_path,
        spec=_single_mode_spec(),
        deformation_direction_mode="VERTICAL",
    )

    result = driver.evaluate_geometry_constraint_function([0.0], "AIRFOIL_CHORD")

    assert result["source"] == "SU2_GEO"
    assert calls.count("geometry_su2_geo") == 2


def test_su2_geo_forced_overrides_available_analytic_backend(tmp_path, monkeypatch):
    calls = []

    def fake_run_command(command, cwd, log_file, show_command=False, stream_output=False, stage=None):
        cwd = Path(cwd)
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        Path(log_file).write_text("$ fake\n")
        calls.append(stage)
        eval_dir = cwd
        while eval_dir.name and not eval_dir.name.startswith("eval_"):
            eval_dir = eval_dir.parent
        if stage == "geometry_su2_def":
            (eval_dir / "deform" / "deformed_mesh.su2").write_text("mesh")
        if stage == "geometry_su2_geo":
            (cwd / "of_func.csv").write_text('"AIRFOIL_AREA"\n1.0\n')

    monkeypatch.setattr(bspline_driver_module, "run_command", fake_run_command)
    driver = _make_driver(
        tmp_path,
        spec=_single_mode_spec(),
        deformation_direction_mode="VERTICAL",
        geometry_constraint_gradient="SU2_GEO",
    )

    result = driver.evaluate_geometry_constraint_function([0.0], "AIRFOIL_AREA")

    assert result["source"] == "SU2_GEO"
    assert calls.count("geometry_su2_geo") == 2


def test_forced_analytic_rejects_unsupported_geometry_constraint(tmp_path):
    driver = _make_driver(
        tmp_path,
        spec=_single_mode_spec(),
        geometry_constraint_gradient="ANALYTIC",
    )

    with pytest.raises(
        BSplineSU2DriverError,
        match="unsupported for AIRFOIL_CHORD",
    ):
        driver.evaluate_geometry_constraint_function([0.0], "AIRFOIL_CHORD")


def test_forced_analytic_area_rejects_nonvertical_direction(tmp_path):
    driver = _make_driver(
        tmp_path,
        spec=_single_mode_spec(),
        deformation_direction_mode="NORMAL",
        geometry_constraint_gradient="ANALYTIC",
    )

    with pytest.raises(
        BSplineSU2DriverError,
        match="BSPLINE_DEFORMATION_DIRECTION=VERTICAL",
    ):
        driver.evaluate_geometry_constraint_function([0.0], "AIRFOIL_AREA")


def test_forced_analytic_thickness_rejects_half_domain(tmp_path):
    spec = _single_mode_spec()
    spec["modes"][0]["side"] = "upper"
    driver = _make_driver(
        tmp_path,
        spec=spec,
        surface_mode="UPPER",
        deformation_direction_mode="VERTICAL",
        geometry_constraint_gradient="ANALYTIC",
    )

    with pytest.raises(
        BSplineSU2DriverError,
        match="analytic AIRFOIL_THICKNESS requires BSPLINE_SURFACE_MODE=BOTH",
    ):
        driver.evaluate_geometry_constraint_function([0.0], "AIRFOIL_THICKNESS")


def test_native_su2_constraints_use_slsqp_sign_convention(tmp_path, monkeypatch):
    captured = {}

    def fake_minimize(fun, x0, jac, bounds, constraints, method, callback, options):
        constraints = list(constraints)
        captured["types"] = [constraint["type"] for constraint in constraints]
        captured["values"] = [constraint["fun"](x0) for constraint in constraints]
        captured["jacs"] = [constraint["jac"](x0).tolist() for constraint in constraints]
        return types.SimpleNamespace(
            x=list(x0),
            fun=1.0,
            success=True,
            message="ok",
            status=0,
            nit=0,
            nfev=0,
            njev=0,
        )

    _install_fake_scipy_minimize(monkeypatch, fake_minimize)
    driver = _make_driver(
        tmp_path,
        spec=_single_mode_spec(),
        native_constraints=(
            "(LIFT>0.5)*2.0; (DRAG<0.02)*3.0; "
            "(MOMENT_Z=0.0)*4.0; (AIRFOIL_AREA>0.1)*5.0"
        ),
        opt_gradient_factor=10.0,
        opt_relax_factor=0.5,
    )

    def fake_constraint_eval(coefficients, function_name):
        data = {
            "LIFT": (0.6, [1.0]),
            "DRAG": (0.03, [2.0]),
            "MOMENT_Z": (-0.1, [3.0]),
        }
        value, gradient = data[str(function_name).upper()]
        return {"value": value, "gradient": gradient}

    def fake_geometry_constraint_eval(coefficients, function_name):
        assert str(function_name).upper() == "AIRFOIL_AREA"
        return {"value": 0.2, "gradient": [4.0]}

    monkeypatch.setattr(
        driver,
        "evaluate_constraint_function",
        fake_constraint_eval,
    )
    monkeypatch.setattr(
        driver,
        "evaluate_geometry_constraint_function",
        fake_geometry_constraint_eval,
    )

    driver.optimize(maxiter=1)

    assert captured["types"] == ["ineq", "ineq", "eq", "ineq"]
    assert captured["values"] == pytest.approx([2.0, -0.3, -4.0, 5.0])
    assert np.asarray(captured["jacs"], dtype=float) == pytest.approx(
        np.asarray([[5.0], [-10.0], [15.0], [20.0]], dtype=float)
    )


def test_enabled_bspline_thickness_constraint_builds_from_closed_airfoil_mesh(
    tmp_path,
    monkeypatch,
):
    modes_file, base, def_template, primal_template, adjoint_template = _write_driver_inputs(
        tmp_path,
        spec=_single_mode_spec(),
    )
    _write_simple_airfoil_mesh(base)
    constraint = _synthetic_thickness_constraint()
    monkeypatch.setattr(
        BSplineSU2Driver,
        "_probe_geometry_aware_bounds",
        lambda self: (constraint.metadata, constraint.basis_matrix),
    )
    driver = BSplineSU2Driver(
        modes_filename=str(modes_file),
        base_mesh=str(base),
        marker="airfoil",
        def_template=str(def_template),
        primal_template=str(primal_template),
        adjoint_template=str(adjoint_template),
        workdir=str(tmp_path / "run"),
        print_optimizer_table=False,
        thickness_options={
            "PROGRESSIVE_THICKNESS_CONSTRAINT": "YES",
            "PROGRESSIVE_THICKNESS_REF_MESH": str(base),
            "PROGRESSIVE_THICKNESS_MARKER": "airfoil",
            "PROGRESSIVE_THICKNESS_X_STATIONS": "0.5",
            "PROGRESSIVE_THICKNESS_CACHE_FILE": str(tmp_path / "cache.npz"),
        },
    )

    built = driver.configure_thickness_constraint()

    assert built is not None
    assert built.values([0.0]) == pytest.approx([0.0], abs=1.0e-12)


def test_bspline_thickness_values_detect_thinning():
    constraint = _synthetic_thickness_constraint()

    assert constraint.values([0.0]) == pytest.approx([0.0], abs=1.0e-12)
    assert constraint.values([-0.05])[0] < 0.0


def test_bspline_thickness_analytic_jacobian_matches_finite_difference():
    constraint = _synthetic_thickness_constraint()
    coeffs = np.asarray([-0.02], dtype=float)

    analytic = constraint.jacobian_analytic(coeffs)
    finite_difference = constraint.jacobian_fd_physical(coeffs)

    assert analytic == pytest.approx(finite_difference, rel=1.0e-6, abs=1.0e-8)


@pytest.mark.parametrize(
    "domain_mode,y_base,deform_y,expected_value,expected_gradient",
    [
        ("HALF_UPPER", 0.1, 1.0, 0.1, 1.0),
        ("HALF_LOWER", -0.1, 1.0, 0.1, -1.0),
    ],
)
def test_bspline_half_thickness_value_and_gradient_sign(
    domain_mode,
    y_base,
    deform_y,
    expected_value,
    expected_gradient,
):
    metadata = [
        {"node_id": 0, "x": 0.0, "y": y_base, "normal_x": 0.0, "normal_y": deform_y},
        {"node_id": 1, "x": 1.0, "y": y_base, "normal_x": 0.0, "normal_y": deform_y},
    ]
    constraint = BSplineThicknessConstraint(
        metadata,
        np.ones((2, 1), dtype=float),
        ["mode"],
        reference_measure=[expected_value],
        x_stations=[0.5],
        domain_mode=domain_mode,
        symmetry_y=0.0,
        closed=False,
    )

    assert constraint.section_measure([0.0]) == pytest.approx([expected_value])
    assert constraint.jacobian_analytic([0.0])[0, 0] == pytest.approx(expected_gradient)


@pytest.mark.parametrize(
    "surface_mode,configured,expected",
    [
        ("BOTH", "AUTO", "FULL"),
        ("UPPER", "AUTO", "HALF_UPPER"),
        ("LOWER", "AUTO", "HALF_LOWER"),
        ("HALF_LOWER", "HALF_LOWER", "HALF_LOWER"),
    ],
)
def test_thickness_domain_auto_follows_surface_mode(surface_mode, configured, expected):
    assert resolve_thickness_domain_mode(surface_mode, configured) == expected


@pytest.mark.parametrize(
    "surface_mode,configured,message",
    [
        ("UPPER", "FULL", "FULL thickness requires a complete upper/lower surface"),
        ("LOWER", "HALF_UPPER", "HALF_UPPER thickness is incompatible"),
    ],
)
def test_thickness_domain_rejects_incompatible_surface_mode(
    surface_mode,
    configured,
    message,
):
    with pytest.raises(BSplineSU2DriverError, match=message):
        resolve_thickness_domain_mode(surface_mode, configured)


@pytest.mark.parametrize("surface_mode,side", [("UPPER", "upper"), ("LOWER", "lower")])
def test_driver_uses_only_single_surface_modes(tmp_path, surface_mode, side):
    spec = _single_mode_spec()
    spec["modes"][0]["side"] = side
    spec["modes"][0]["id"] = f"{side}_b"
    spec["surface_mode"] = surface_mode
    driver = _make_driver(
        tmp_path,
        spec=spec,
        surface_mode=surface_mode,
        symmetry_coupling="NONE",
    )

    assert driver.mode_ids == [f"{side}_b"]
    assert len(driver.reduced_variable_ids) == 1
    driver.write_optimized_modes([0.004])
    optimized = json.loads(driver.optimized_modes_filename.read_text())
    assert optimized["surface_mode"] == surface_mode
    assert {mode["side"] for mode in optimized["modes"]} == {side}


@pytest.mark.parametrize("surface_mode", ["UPPER", "LOWER"])
def test_half_domain_rejects_symmetry_coupling(tmp_path, surface_mode):
    spec = _single_mode_spec()
    side = surface_mode.lower()
    spec["modes"][0]["side"] = side
    spec["surface_mode"] = surface_mode
    with pytest.raises(
        BSplineSU2DriverError,
        match="BSPLINE_SYMMETRY_COUPLING is only valid with BSPLINE_SURFACE_MODE=BOTH",
    ):
        _make_driver(
            tmp_path,
            spec=spec,
            surface_mode=surface_mode,
            symmetry_coupling="NORMAL_EQUAL",
        )


def test_bspline_thickness_slsqp_jacobian_scales_by_relax_and_beta(tmp_path):
    driver = _make_driver(
        tmp_path,
        spec=_single_mode_spec(),
        opt_relax_factor=1000.0,
    )
    driver.thickness_constraint = _synthetic_thickness_constraint()
    constraint = driver._thickness_constraint_functions()[0]

    jac_u = constraint["jac"]([0.0])

    assert jac_u == pytest.approx(
        driver.thickness_constraint.jacobian_analytic([0.0]) * 1000.0
    )


def test_bspline_thickness_logging_writes_station_values(tmp_path):
    driver = _make_driver(tmp_path, spec=_single_mode_spec())
    driver.thickness_constraint = _synthetic_thickness_constraint()
    eval_dir = tmp_path / "run" / "eval_0000"

    driver._append_history_record(
        0,
        1.0,
        [-0.004],
        [0.0],
        "ok",
        eval_dir=eval_dir,
        eval_index=1,
    )

    values_file = eval_dir / "thickness_constraint" / "values.csv"
    metadata_file = eval_dir / "thickness_constraint" / "metadata.json"
    with values_file.open("r", newline="") as fp:
        rows = list(csv.DictReader(fp))
    metadata = json.loads(metadata_file.read_text())

    assert len(rows) == 1
    assert float(rows[0]["x"]) == pytest.approx(0.5)
    assert float(rows[0]["current_measure"]) == pytest.approx(0.196)
    assert float(rows[0]["reference_measure"]) == pytest.approx(0.2)
    assert float(rows[0]["constraint_value"]) == pytest.approx(-0.004)
    assert rows[0]["active"] == "1"
    assert metadata["values_written"] is True
    assert metadata["gradient_mode_used"] == "NOT_EVALUATED"


def test_bspline_thickness_logging_writes_slsqp_jacobian(tmp_path):
    driver = _make_driver(
        tmp_path,
        spec=_single_mode_spec(),
        opt_relax_factor=1000.0,
    )
    driver.thickness_constraint = _synthetic_thickness_constraint()
    constraint = driver._thickness_constraint_functions()[0]
    variables = driver.physical_to_optimizer([-0.004])

    jacobian = constraint["jac"](variables)
    eval_dir = tmp_path / "run" / "eval_0001"
    driver._append_history_record(
        1,
        1.0,
        [-0.004],
        [0.0],
        "ok",
        eval_dir=eval_dir,
        eval_index=1,
    )

    jacobian_file = eval_dir / "thickness_constraint" / "jacobian.csv"
    metadata_file = eval_dir / "thickness_constraint" / "metadata.json"
    with jacobian_file.open("r", newline="") as fp:
        rows = list(csv.DictReader(fp))
    metadata = json.loads(metadata_file.read_text())
    field = f"dg_d_{driver.reduced_variable_ids[0]}"

    assert len(rows) == 1
    assert float(rows[0][field]) == pytest.approx(float(jacobian[0, 0]))
    assert metadata["jacobian_written"] is True
    assert metadata["gradient_mode_used"] == "ANALYTIC"
    assert metadata["variable_space"] == "slsqp_reduced"


def test_bspline_thickness_constraint_uses_line_search_limited_coefficients(
    tmp_path,
):
    driver = _make_driver(
        tmp_path,
        spec=_single_mode_spec(),
        opt_line_search_bound=0.05,
    )
    driver.thickness_constraint = _synthetic_thickness_constraint()
    driver._line_search_bound_configured = True
    driver._line_search_basis_matrix = np.asarray([[1.0]], dtype=float)
    driver._line_search_anchor_physical = [0.0]
    constraint = driver._thickness_constraint_functions()[0]

    g_limited = constraint["fun"]([-0.1])

    assert g_limited == pytest.approx(driver.thickness_constraint.values([-0.05]))
    assert g_limited != pytest.approx(driver.thickness_constraint.values([-0.1]))


def test_config_patching_replaces_existing_keys_and_appends_missing(tmp_path):
    template = tmp_path / "template.cfg"
    output = tmp_path / "patched.cfg"
    template.write_text(
        "% preserved comment\n"
        "MESH_FILENAME= old.su2\n"
        "SCREEN_OUTPUT= ( INNER_ITER )\n"
    )

    patch_config_template(
        template,
        output,
        {
            "MESH_FILENAME": "new.su2",
            "SCREEN_OUTPUT": ["INNER_ITER", "RMS_RES", "LIFT", "DRAG"],
            "SOLUTION_FILENAME": "solution_flow.dat",
        },
    )

    text = output.read_text()
    assert "% preserved comment" in text
    assert "MESH_FILENAME= new.su2" in text
    assert "SCREEN_OUTPUT= ( INNER_ITER, RMS_RES, LIFT, DRAG )" in text
    assert "SOLUTION_FILENAME= solution_flow.dat" in text


def test_objective_parsing_reads_final_history_csv_value(tmp_path):
    history = tmp_path / "history.csv"
    history.write_text(
        '"Iter",       "CD"      , "CL"\n'
        "0, 0.20, 0.10\n"
        "1, 0.13, 0.11\n"
    )

    assert read_objective_from_history(history, "CD") == 0.13


def test_gradient_parsing_preserves_requested_mode_order(tmp_path):
    gradients = tmp_path / "bspline_gradients.csv"
    with gradients.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["mode_id", "gradient"])
        writer.writeheader()
        writer.writerow({"mode_id": "upper_a", "gradient": "1.25"})
        writer.writerow({"mode_id": "lower_b", "gradient": "-2.5"})

    assert read_bspline_gradients(gradients) == {
        "upper_a": 1.25,
        "lower_b": -2.5,
    }
    assert read_gradient_vector(gradients, ["lower_b", "upper_a"]) == [-2.5, 1.25]


def test_cache_key_rounds_coefficients_to_tolerance():
    assert cache_key([0.123456789, -0.2], tol=1.0e-6) == cache_key(
        [0.123456781, -0.2],
        tol=1.0e-6,
    )
    assert cache_key([0.123456789], tol=1.0e-6) != cache_key(
        [0.123459],
        tol=1.0e-6,
    )


def test_command_builder_produces_expected_commands(tmp_path):
    paths = build_eval_paths(tmp_path / "eval_0000")

    commands = build_eval_commands(
        paths,
        base_mesh="/abs/base_mesh.su2",
        marker="airfoil",
        mpi_prefix="mpirun -n 6",
        python_executable="python3",
    )

    assert commands["bspline_def"][:4] == [
        "python3",
        "-m",
        "SU2.opt.bspline_def",
        "--mesh",
    ]
    assert "/abs/base_mesh.su2" in commands["bspline_def"]
    direction_index = commands["bspline_def"].index("--deformation-direction")
    assert commands["bspline_def"][direction_index + 1] == "NORMAL"
    assert commands["bspline_def"][-2:] == ["--marker", "airfoil"]
    assert commands["def"] == ["mpirun", "-n", "6", "SU2_DEF", "def.cfg"]
    assert commands["primal"] == ["mpirun", "-n", "6", "SU2_CFD", "primal.cfg"]
    assert commands["adjoint"] == [
        "mpirun",
        "-n",
        "6",
        "SU2_CFD_AD",
        "adjoint.cfg",
    ]
    assert commands["dot_ad"] == [
        "mpirun",
        "-n",
        "6",
        "SU2_DOT_AD",
        "dot_ad.cfg",
    ]
    assert paths.deform_dir == tmp_path / "eval_0000" / "deform"
    assert paths.direct_dir == tmp_path / "eval_0000" / "direct"
    assert paths.adjoint_dir == tmp_path / "eval_0000" / "adjoint_drag"
    assert "--prefer-vector" in commands["bspline_dot"]
    assert "--sensitivity-weighting" in commands["bspline_dot"]
    assert commands["bspline_dot"][commands["bspline_dot"].index("--sensitivity-weighting") + 1] == "NODAL"
    assert "adjoint_drag/surface_sens.csv" in commands["bspline_dot"]


def test_command_builder_legacy_surface_adjoint_source(tmp_path):
    paths = build_eval_paths(tmp_path / "eval_0000")

    commands = build_eval_commands(
        paths,
        base_mesh="/abs/base_mesh.su2",
        marker="airfoil",
        sensitivity_source="CFD_ADJOINT_SURFACE",
    )

    assert "dot_ad" not in commands
    assert "adjoint_drag/surface_adjoint.csv" in commands["bspline_dot"]


def test_command_builder_uses_vertical_direction_without_le_safe_options(tmp_path):
    paths = build_eval_paths(tmp_path / "eval_0000")

    commands = build_eval_commands(
        paths,
        base_mesh="/abs/base_mesh.su2",
        marker="airfoil",
        python_executable="python3",
        deformation_direction_mode="VERTICAL",
        le_safe_direction=True,
        le_safe_x0=0.0,
        le_safe_x1=0.05,
        le_safe_power=2.0,
    )
    command = commands["bspline_def"]

    mode_index = command.index("--deformation-direction")
    assert command[mode_index + 1] == "VERTICAL"
    assert "--le-safe-direction" not in command
    assert "--le-safe-x0" not in command
    assert "--le-safe-x1" not in command
    assert "--le-safe-power" not in command


def test_command_builder_passes_half_domain_surface_mode(tmp_path):
    paths = build_eval_paths(tmp_path / "eval_0000")
    command = build_eval_commands(
        paths,
        base_mesh="/abs/mesh_half_lower.su2",
        marker="airfoil",
        surface_mode="LOWER",
    )["bspline_def"]

    index = command.index("--surface-mode")
    assert command[index + 1] == "LOWER"


def test_driver_config_maps_vertical_deformation_direction():
    options = fixed_driver_options_from_config(
        {
            "BSPLINE_DEFORMATION_DIRECTION": "VERTICAL",
            "BSPLINE_LE_SAFE_DIRECTION": True,
        }
    )
    assert options["deformation_direction_mode"] == "VERTICAL"
    assert options["le_safe_direction"] is True


def test_driver_config_maps_surface_mode_alias():
    options = fixed_driver_options_from_config(
        {"BSPLINE_SURFACE_MODE": "HALF_UPPER"}
    )
    assert options["surface_mode"] == "HALF_UPPER"


def test_driver_config_maps_gradient_guard_options():
    options = fixed_driver_options_from_config(
        {
            "BSPLINE_GRADIENT_GUARD": False,
            "BSPLINE_GRADIENT_GUARD_FACTOR": 125.0,
            "BSPLINE_GRADIENT_GUARD_WINDOW": 7,
            "BSPLINE_GRADIENT_GUARD_MIN_HISTORY": 4,
            "BSPLINE_GRADIENT_GUARD_FLOOR": 1.0e-12,
            "BSPLINE_GRADIENT_GUARD_RESTART_LIMIT": 3,
        }
    )

    assert options == {
        "gradient_guard": False,
        "gradient_guard_factor": 125.0,
        "gradient_guard_window": 7,
        "gradient_guard_min_history": 4,
        "gradient_guard_floor": 1.0e-12,
        "gradient_guard_restart_limit": 3,
    }


def test_driver_config_maps_knot_batch_penalty_options():
    options = fixed_driver_options_from_config(
        {
            "BSPLINE_KNOT_BATCH_DIVERSITY": "YES",
            "BSPLINE_KNOT_BATCH_PENALTY_MODE": "POWER",
            "BSPLINE_KNOT_BATCH_POWER_GAMMA": 0.25,
        }
    )

    assert options == {
        "knot_batch_diversity": "YES",
        "knot_batch_penalty_mode": "POWER",
        "knot_batch_power_gamma": 0.25,
    }


def test_driver_config_maps_knot_depth_penalty_options():
    options = fixed_driver_options_from_config(
        {
            "BSPLINE_KNOT_DEPTH_PENALTY": "YES",
            "BSPLINE_KNOT_DEPTH_PENALTY_MODE": "POWER",
            "BSPLINE_KNOT_DEPTH_POWER_GAMMA": 0.25,
            "BSPLINE_KNOT_INITIAL_SPAN_DEPTH": 1,
        }
    )

    assert options == {
        "knot_depth_penalty": "YES",
        "knot_depth_penalty_mode": "POWER",
        "knot_depth_power_gamma": 0.25,
        "knot_initial_span_depth": 1,
    }


def test_driver_config_maps_experimental_trust_clip_options():
    options = fixed_driver_options_from_config(
        {
            "BSPLINE_TRUST_CLIP_POLICY": "ACCEPT_RESTART",
            "BSPLINE_TRUST_CLIP_BETA_TOL": 1.0e-10,
            "BSPLINE_TRUST_CLIP_LEGACY_BETA_MIN": 0.6,
            "BSPLINE_TRUST_CLIP_SEVERE_BETA": 0.4,
            "BSPLINE_TRUST_CLIP_WORSENING_TOL": 0.08,
            "BSPLINE_TRUST_CLIP_SOFT_GNORM_FACTOR": 25.0,
            "BSPLINE_TRUST_CLIP_BAD_PATIENCE": 3,
            "BSPLINE_TRUST_CLIP_BAD_WINDOW": 7,
            "BSPLINE_TRUST_CLIP_STAG_TOL": 2.0e-6,
            "BSPLINE_TRUST_CLIP_RESTART_LIMIT": 2,
        }
    )

    assert options["trust_clip_policy"] == "ACCEPT_RESTART"
    assert options["trust_clip_legacy_beta_min"] == pytest.approx(0.6)
    assert options["trust_clip_bad_window"] == 7
    assert options["trust_clip_restart_limit"] == 2


def test_command_builder_keeps_legacy_le_safe_activation(tmp_path):
    paths = build_eval_paths(tmp_path / "eval_0000")
    command = build_eval_commands(
        paths,
        base_mesh="/abs/base_mesh.su2",
        marker="airfoil",
        python_executable="python3",
        le_safe_direction=True,
        le_safe_x0=0.0,
        le_safe_x1=0.05,
        le_safe_power=2.0,
    )["bspline_def"]

    mode_index = command.index("--deformation-direction")
    assert command[mode_index + 1] == "LE_SAFE"
    assert "--le-safe-direction" in command
    assert "--le-safe-x0" in command
    assert "--le-safe-x1" in command
    assert "--le-safe-power" in command


def test_flat_eval_layout_is_removed(tmp_path):
    with pytest.raises(BSplineSU2DriverError, match="BSPLINE_EVAL_LAYOUT=FLAT has been removed"):
        build_eval_paths(tmp_path / "eval_0000", eval_layout="FLAT")


def test_default_eval_layout_is_dsn(tmp_path):
    paths = build_eval_paths(tmp_path / "eval_0000")
    assert paths.eval_layout == "DSN"
    assert paths.deform_dir == tmp_path / "eval_0000" / "deform"
    assert paths.direct_dir == tmp_path / "eval_0000" / "direct"
    assert paths.adjoint_dir == tmp_path / "eval_0000" / "adjoint_drag"


def test_dsn_eval_layout_patches_relative_solver_paths(tmp_path):
    driver = _make_driver(tmp_path, eval_layout="DSN")
    eval_id, paths = driver._next_paths()

    driver._prepare_eval_files(driver.initial_coefficients, paths)

    assert eval_id == 0
    assert paths.def_cfg.exists()
    assert paths.primal_cfg.exists()
    assert paths.adjoint_cfg.exists()
    assert paths.dot_ad_cfg.exists()
    assert "MESH_FILENAME= ../deform/deformed_mesh.su2" in paths.primal_cfg.read_text()
    adjoint_text = paths.adjoint_cfg.read_text()
    assert "MESH_FILENAME= ../deform/deformed_mesh.su2" in adjoint_text
    assert "SOLUTION_FILENAME= ../direct/solution_flow.dat" in adjoint_text
    assert "RESTART_FILENAME= ../direct/restart_flow.dat" in adjoint_text
    assert "SURFACE_ADJ_FILENAME= surface_adjoint" in adjoint_text
    assert "VOLUME_ADJ_FILENAME= volume_adjoint" in adjoint_text
    dot_text = paths.dot_ad_cfg.read_text()
    assert "DV_KIND= SURFACE_FILE" in dot_text
    assert "DV_MARKER= ( airfoil )" in dot_text
    assert "DV_FILENAME= ../deform/surface_positions.dat" in dot_text
    assert "SURFACE_SENS_FILENAME= surface_sens" in dot_text
    assert "VOLUME_SENS_FILENAME= volume_sens" in dot_text
    assert "OUTPUT_FILES= ( SURFACE_CSV )" in dot_text
    assert "OUTPUT_PRECISION= 15" in dot_text


def test_reused_workdir_eval_id_is_independent_from_run_eval_index(tmp_path):
    driver = _make_driver(tmp_path)
    driver.workdir.mkdir(parents=True, exist_ok=True)
    for eval_id in range(9):
        (driver.workdir / f"eval_{eval_id:04d}").mkdir()
    driver._next_eval_id = driver._initial_eval_id()

    eval_id, paths = driver._next_paths()
    assert eval_id == 9
    driver._run_eval_count += 1
    driver._append_history_record(
        eval_id,
        1.0,
        driver.initial_coefficients,
        [0.0] * len(driver.mode_ids),
        "ok",
        eval_dir=paths.eval_dir,
        eval_index=driver._run_eval_count,
    )

    with open(driver.optimization_history_filename, "r", newline="") as fp:
        row = next(csv.DictReader(fp))
    assert row["eval_id"] == "9"
    assert row["eval_index"] == "1"
    assert row["eval_dir"].endswith("eval_0009")


def test_optimization_history_includes_counter_and_limiter_columns(tmp_path):
    driver = _make_driver(tmp_path)
    driver._append_history_record(
        0,
        1.0,
        driver.initial_coefficients,
        [0.0] * len(driver.mode_ids),
        "ok",
        line_search_info={
            "line_search_beta": 0.5,
            "line_search_maxdiff": 2.0,
            "line_search_limited": 1,
            "local_step_beta": 0.75,
            "local_step_limited": 1,
            "local_step_limiting_mode": "upper_a",
            "local_step_da": 0.1,
            "local_step_limit": 0.05,
        },
        eval_dir=driver.workdir / "eval_0000",
        eval_index=1,
    )

    with open(driver.optimization_history_filename, "r", newline="") as fp:
        reader = csv.DictReader(fp)
        row = next(reader)
    for field in (
        "eval_index",
        "slsqp_iter",
        "eval_id",
        "eval_dir",
        "objective",
        "coefficients",
        "reduced_coefficients",
        "gradients",
        "line_search_beta",
        "line_search_maxdiff",
        "line_search_limited",
        "local_step_beta",
        "local_step_limited",
        "local_step_limiting_mode",
        "local_step_da",
        "local_step_limit",
        "trust_clip_class",
        "trust_clip_action",
        "trust_clip_beta",
        "trust_clip_improvement_rel",
        "trust_clip_relative_worsening",
        "trust_clip_gnorm_ratio",
        "trust_clip_reasons",
        "status",
    ):
        assert field in reader.fieldnames
    assert row["eval_index"] == "1"
    assert row["line_search_beta"] == "0.5"
    assert row["local_step_beta"] == "0.75"


def test_dsn_eval_aliases_symlink_or_copy_root_compatibility_files(tmp_path):
    from SU2.opt.bspline_su2_driver import create_eval_aliases

    paths = build_eval_paths(tmp_path / "eval_0000", eval_layout="DSN")
    paths.metadata.parent.mkdir(parents=True)
    paths.primal_history.parent.mkdir(parents=True)
    paths.surface_adjoint.parent.mkdir(parents=True)
    paths.metadata.write_text("metadata")
    paths.surface_adjoint.write_text("adjoint")
    paths.surface_sens.write_text("sens")
    paths.primal_history.write_text("primal")
    paths.adjoint_history.write_text("adjoint history")

    create_eval_aliases(paths)

    assert (paths.eval_dir / "bspline_surface_metadata.csv").read_text() == "metadata"
    assert (paths.eval_dir / "surface_adjoint.csv").read_text() == "adjoint"
    assert (paths.eval_dir / "surface_sens.csv").read_text() == "sens"
    assert (paths.eval_dir / "history_primal.csv").read_text() == "primal"
    assert (paths.eval_dir / "history_adjoint.csv").read_text() == "adjoint history"


def _install_fake_aero_run(monkeypatch, default_gradient=1.0):
    calls = []

    def fake_run_command(command, cwd, log_file, show_command=False, stream_output=False, stage=None):
        cwd = Path(cwd)
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text("$ fake\n")
        calls.append((stage, cwd, list(command)))
        eval_dir = cwd
        while eval_dir.name and not eval_dir.name.startswith("eval_"):
            eval_dir = eval_dir.parent

        if stage == "bspline_def":
            (eval_dir / "deform").mkdir(parents=True, exist_ok=True)
            (eval_dir / "deform" / "surface_positions.dat").write_text("surface")
            (eval_dir / "deform" / "bspline_surface_metadata.csv").write_text(
                "node_id,x,y,normal_x,normal_y\n"
            )
        elif stage == "su2_def":
            (eval_dir / "deform" / "deformed_mesh.su2").write_text("mesh")
        elif stage == "su2_cfd":
            (eval_dir / "direct").mkdir(parents=True, exist_ok=True)
            (eval_dir / "direct" / "restart_flow.dat").write_text("restart")
            (eval_dir / "direct" / "history_primal.csv").write_text(
                '"CD","CL","CMz"\n0.11,0.22,0.33\n'
            )
        elif stage == "su2_cfd_ad":
            cwd.mkdir(parents=True, exist_ok=True)
            (cwd / "surface_adjoint.csv").write_text("adjoint")
            (cwd / "history_adjoint.csv").write_text('"RMS_ADJ"\n0.0\n')
        elif stage == "su2_dot_ad":
            cwd.mkdir(parents=True, exist_ok=True)
            # Tag the sensitivity with the adjoint dir name so root aliases can be
            # traced back to the function that produced them.
            (cwd / "surface_sens.csv").write_text(cwd.name)
        elif stage == "bspline_dot":
            command_text = " ".join(str(part) for part in command)
            if "adjoint_lift" in command_text:
                gradient = 2.0
            elif "adjoint_momentz" in command_text:
                gradient = 3.0
            else:
                gradient = default_gradient
            (eval_dir / "bspline_gradients.csv").write_text(
                "mode_id,gradient\nlower_b,{:.1f}\n".format(gradient)
            )

    monkeypatch.setattr(bspline_driver_module, "run_command", fake_run_command)
    return calls


def test_aero_constraints_and_objective_share_primal_with_lazy_adjoints(tmp_path, monkeypatch):
    calls = _install_fake_aero_run(monkeypatch)
    driver = _make_driver(
        tmp_path,
        spec=_single_mode_spec(),
        objective_adjoint="DRAG",
        gradient_guard=False,
    )

    lift = driver.evaluate_constraint_function([0.0], "LIFT")
    moment = driver.evaluate_constraint_function([0.0], "MOMENT_Z")
    objective = driver.evaluate([0.0])

    assert lift["eval_id"] == moment["eval_id"] == objective["eval_id"] == 0
    assert lift["eval_dir"] == moment["eval_dir"] == objective["eval_dir"]
    eval_dir = Path(objective["eval_dir"])
    assert (eval_dir / "direct").is_dir()
    assert (eval_dir / "adjoint_lift").is_dir()
    assert (eval_dir / "adjoint_momentz").is_dir()
    assert (eval_dir / "adjoint_drag").is_dir()
    assert [call[0] for call in calls].count("su2_cfd") == 1
    assert [call[0] for call in calls].count("su2_cfd_ad") == 3
    assert lift["value"] == pytest.approx(0.22)
    assert moment["value"] == pytest.approx(0.33)
    assert objective["objective"] == pytest.approx(0.11)
    assert lift["gradient"] == pytest.approx([2.0])
    assert moment["gradient"] == pytest.approx([3.0])
    assert objective["gradient"] == pytest.approx([1.0])
    assert read_gradient_vector(
        eval_dir / "adjoint_lift" / "bspline_gradients.csv",
        ["lower_b"],
    ) == pytest.approx([2.0])
    assert read_gradient_vector(
        eval_dir / "adjoint_momentz" / "bspline_gradients.csv",
        ["lower_b"],
    ) == pytest.approx([3.0])
    assert read_gradient_vector(
        eval_dir / "adjoint_drag" / "bspline_gradients.csv",
        ["lower_b"],
    ) == pytest.approx([1.0])
    assert read_gradient_vector(
        eval_dir / "bspline_gradients.csv",
        ["lower_b"],
    ) == pytest.approx([1.0])
    assert "OBJECTIVE_FUNCTION= LIFT" in (eval_dir / "adjoint_lift" / "adjoint.cfg").read_text()
    assert "OBJECTIVE_FUNCTION= MOMENT_Z" in (
        eval_dir / "adjoint_momentz" / "adjoint.cfg"
    ).read_text()
    assert "OBJECTIVE_FUNCTION= DRAG" in (eval_dir / "adjoint_drag" / "adjoint.cfg").read_text()


def test_objective_without_constraints_keeps_single_drag_adjoint(tmp_path, monkeypatch):
    calls = _install_fake_aero_run(monkeypatch)
    driver = _make_driver(
        tmp_path,
        spec=_single_mode_spec(),
        objective_adjoint="DRAG",
        gradient_guard=False,
    )

    result = driver.evaluate([0.0])

    eval_dir = Path(result["eval_dir"])
    assert sorted(path.name for path in eval_dir.glob("adjoint_*")) == ["adjoint_drag"]
    assert [call[0] for call in calls].count("su2_cfd") == 1
    assert [call[0] for call in calls].count("su2_cfd_ad") == 1


def test_objective_guard_stop_does_not_invalidate_shared_constraint_results(
    tmp_path, monkeypatch
):
    # Pathological objective gradient (drag) so the guard fires; constraints fine.
    calls = _install_fake_aero_run(monkeypatch, default_gradient=500.0)
    driver = _make_driver(
        tmp_path,
        spec=_single_mode_spec(),
        objective_adjoint="DRAG",
        gradient_guard=True,
        gradient_guard_factor=100.0,
        gradient_guard_min_history=3,
    )
    driver.recent_safe_raw_gnorms.extend([1.0, 1.0, 1.0])

    lift = driver.evaluate_constraint_function([0.0], "LIFT")
    moment = driver.evaluate_constraint_function([0.0], "MOMENT_Z")

    with pytest.raises(GradientGuardStop):
        driver.evaluate([0.0])

    # One shared primal even though the objective then tripped the guard,
    # and all three adjoints had run before the guard fired in _finalize_objective.
    assert [call[0] for call in calls].count("su2_cfd") == 1
    assert [call[0] for call in calls].count("su2_cfd_ad") == 3

    # The constraint results cached in the shared eval survive the objective rollback:
    # re-reading them does not trigger any new adjoint solve.
    assert lift["gradient"] == pytest.approx([2.0])
    assert moment["gradient"] == pytest.approx([3.0])
    assert driver.evaluate_constraint_function([0.0], "LIFT")["gradient"] == pytest.approx(
        [2.0]
    )
    assert driver.evaluate_constraint_function([0.0], "MOMENT_Z")[
        "gradient"
    ] == pytest.approx([3.0])
    assert [call[0] for call in calls].count("su2_cfd_ad") == 3


def test_trust_clip_reclassifies_same_design_point_without_resolving(tmp_path, monkeypatch):
    calls = _install_fake_aero_run(monkeypatch)
    driver = _make_driver(
        tmp_path,
        spec=_single_mode_spec(),
        objective_adjoint="DRAG",
        gradient_guard=False,
        trust_clip_policy="ACCEPT_RESTART",
    )

    driver.evaluate([0.0])
    driver.evaluate([0.0])

    # Same coefficients => same design point reused: primal/adjoint solved once,
    # but trust-clip re-classifies on every objective request.
    assert [call[0] for call in calls].count("su2_cfd") == 1
    assert [call[0] for call in calls].count("su2_cfd_ad") == 1
    assert len(driver.recent_level_clip_events) == 2


def test_shared_eval_promotes_objective_once_despite_constraints(tmp_path, monkeypatch):
    _install_fake_aero_run(monkeypatch)
    driver = _make_driver(
        tmp_path,
        spec=_single_mode_spec(),
        objective_adjoint="DRAG",
        gradient_guard=True,
        gradient_guard_min_history=3,
    )

    driver.evaluate_constraint_function([0.0], "LIFT")
    driver.evaluate_constraint_function([0.0], "MOMENT_Z")
    # Constraints never touch the safe-gradient history.
    assert list(driver.recent_safe_raw_gnorms) == []

    driver.evaluate([0.0])
    # The objective is promoted exactly once for the shared design point.
    assert list(driver.recent_safe_raw_gnorms) == [pytest.approx(1.0)]


def test_shared_eval_root_sensitivity_alias_tracks_objective_not_last_func(
    tmp_path, monkeypatch
):
    _install_fake_aero_run(monkeypatch)
    driver = _make_driver(
        tmp_path,
        spec=_single_mode_spec(),
        objective_adjoint="DRAG",
        gradient_guard=False,
    )

    # Objective first, constraint last: the eval-root surface_sens alias consumed by
    # adaptive scoring must stay the objective's, independent of evaluation order.
    objective = driver.evaluate([0.0])
    driver.evaluate_constraint_function([0.0], "LIFT")

    eval_dir = Path(objective["eval_dir"])
    assert (eval_dir / "surface_sens.csv").read_text() == "adjoint_drag"


def test_cli_default_hides_commands_unless_show_commands_is_used():
    parser = _build_arg_parser()
    required = [
        "--modes",
        "modes.json",
        "--base-mesh",
        "mesh.su2",
        "--marker",
        "airfoil",
        "--def-template",
        "def.cfg",
        "--primal-template",
        "primal.cfg",
        "--adjoint-template",
        "adj.cfg",
        "--workdir",
        "run",
    ]

    assert parser.parse_args(required).show_commands is False
    assert parser.parse_args(required + ["--show-commands"]).show_commands is True
    assert parser.parse_args(required + ["--show-commands", "--quiet-driver"]).show_commands is False


def test_run_command_failure_reports_stage_log_tail_without_command_by_default(tmp_path):
    log_file = tmp_path / "eval_0000" / "stage.log"

    with pytest.raises(BSplineSU2DriverError) as exc_info:
        run_command(
            ["python3", "-c", "print('tail marker'); raise SystemExit(7)"],
            tmp_path / "eval_0000",
            log_file,
            show_command=False,
            stage="synthetic_stage",
        )

    message = str(exc_info.value)
    assert "failed stage: synthetic_stage" in message
    assert "eval directory:" in message
    assert f"log file: {log_file}" in message
    assert "last 20 log lines:" in message
    assert "tail marker" in message
    assert "python3 -c" not in message


def test_eval_paths_include_commands_log(tmp_path):
    paths = build_eval_paths(tmp_path / "eval_0000")

    assert paths.commands_log == tmp_path / "eval_0000" / "commands.log"


def test_optimizer_table_uses_slsqp_iteration_fc_and_eval_id(tmp_path, capsys):
    driver = _make_driver(tmp_path)
    driver.print_optimizer_table = True
    driver._slsqp_major_iter = 3
    driver._run_eval_count = 1

    driver._print_iteration_row(
        {
            "eval_index": 1,
            "eval_id": 17,
            "objective": 2.0,
            "gradient": [1.0, 2.0],
        },
        line_search_info={"line_search_beta": 1.0, "local_step_beta": 1.0},
    )

    output = capsys.readouterr().out
    assert "SLSQP_IT   FC   EVAL_ID" in output
    assert "NIT" not in output
    assert "       3    1        17" in output


def test_geometry_bound_scaling_math_and_errors():
    basis = [[1.0, 0.0], [0.0, 1.0]]
    coeffs = [0.2, -0.1]
    bounds = [(-1.0, 1.0), (-0.5, 0.5)]

    unchanged = compute_geometry_aware_bound_scaling(
        basis,
        coeffs,
        bounds,
        max_normal_displacement=2.0,
    )
    assert unchanged["beta_safe"] == pytest.approx(1.0)
    assert unchanged["scaled_bounds"] == [[-1.0, 1.0], [-0.5, 0.5]]

    max_limited = compute_geometry_aware_bound_scaling(
        basis,
        coeffs,
        bounds,
        max_normal_displacement=0.4,
    )
    assert max_limited["beta_safe"] == pytest.approx(1.0 / 6.0)
    assert max_limited["scaled_bounds"][0][0] == pytest.approx(0.0)
    assert max_limited["scaled_bounds"][0][1] == pytest.approx(1.0 / 3.0)

    rms_limited = compute_geometry_aware_bound_scaling(
        basis,
        coeffs,
        bounds,
        max_rms_normal_displacement=0.25,
    )
    assert 0.0 <= rms_limited["beta_safe"] <= 1.0
    assert rms_limited["beta_safe"] < 1.0

    both_limited = compute_geometry_aware_bound_scaling(
        basis,
        coeffs,
        bounds,
        max_normal_displacement=0.4,
        max_rms_normal_displacement=0.25,
    )
    assert both_limited["beta_safe"] == pytest.approx(min(max_limited["beta_safe"], rms_limited["beta_safe"]))

    with pytest.raises(BSplineSU2DriverError, match="Initial geometry already violates max-normal-displacement"):
        compute_geometry_aware_bound_scaling(
            basis,
            [2.0, 0.0],
            bounds,
            max_normal_displacement=1.0,
        )
    with pytest.raises(BSplineSU2DriverError, match="Initial geometry already violates max-rms-normal-displacement"):
        compute_geometry_aware_bound_scaling(
            basis,
            [0.9, 0.0],
            bounds,
            max_rms_normal_displacement=0.5,
        )
    with pytest.raises(BSplineSU2DriverError, match="below --min-bound-scale"):
        compute_geometry_aware_bound_scaling(
            basis,
            coeffs,
            bounds,
            max_normal_displacement=0.4,
            min_bound_scale=0.5,
        )


def test_geometry_bound_scaling_cli_and_history_integration(tmp_path, monkeypatch):
    modes = _base_spec()
    modes["modes"][0]["coefficient"] = 0.2
    modes["modes"][0]["bounds"] = [-1.0, 1.0]
    modes["modes"][1]["coefficient"] = -0.1
    modes["modes"][1]["bounds"] = [-0.5, 0.5]
    base = tmp_path / "base.su2"
    base.write_text("mesh")
    def_template = tmp_path / "def.cfg"
    primal_template = tmp_path / "primal.cfg"
    adjoint_template = tmp_path / "adj.cfg"
    for path in (def_template, primal_template, adjoint_template):
        path.write_text("MESH_FILENAME= old.su2\n")
    modes_file = tmp_path / "modes.json"
    modes_file.write_text(json.dumps(modes, indent=2))

    parser = _build_arg_parser()
    required = [
        "--modes",
        str(modes_file),
        "--base-mesh",
        str(base),
        "--marker",
        "airfoil",
        "--def-template",
        str(def_template),
        "--primal-template",
        str(primal_template),
        "--adjoint-template",
        str(adjoint_template),
        "--workdir",
        str(tmp_path / "run"),
    ]
    assert parser.parse_args(required).auto_scale_bounds_to_geometry is False
    parsed = parser.parse_args(
        required
        + [
            "--auto-scale-bounds-to-geometry",
            "--max-normal-displacement",
            "0.4",
            "--max-rms-normal-displacement",
            "0.25",
            "--min-bound-scale",
            "0.09",
        ]
    )
    assert parsed.auto_scale_bounds_to_geometry is True
    assert parsed.max_normal_displacement == pytest.approx(0.4)
    assert parsed.max_rms_normal_displacement == pytest.approx(0.25)
    assert parsed.min_bound_scale == pytest.approx(0.09)

    beta = compute_geometry_aware_bound_scaling(
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=float),
        [0.2, -0.1],
        [(-1.0, 1.0), (-0.5, 0.5)],
        max_normal_displacement=0.4,
        max_rms_normal_displacement=0.25,
        min_bound_scale=0.09,
    )["beta_safe"]
    expected_bounds = [
        [0.2 + beta * (-1.2), 0.2 + beta * 0.8],
        [-0.1 + beta * (-0.4), -0.1 + beta * 0.6],
    ]
    monkeypatch.setattr(
        "SU2.opt.bspline_su2_driver.BSplineSU2Driver._probe_geometry_aware_bounds",
        lambda self: ([], [[1.0, 0.0], [0.0, 1.0]]),
    )

    def fake_optimize(self, *args, **kwargs):
        self.configure_geometry_aware_bounds()
        assert self.bounds == [tuple(bounds) for bounds in expected_bounds]
        self._history_records = [
            {
                "eval_id": 0,
                "objective": 1.23,
                "status": "ok",
                "coeff__upper_a": 0.2,
                "coeff__lower_b": -0.1,
                "grad__upper_a": 0.0,
                "grad__lower_b": 0.0,
            }
        ]
        self.write_optimization_history()
        self.write_optimized_modes([0.2, -0.1])
        return {
            "optimizer": "SLSQP",
            "success": True,
            "message": "ok",
            "objective": 1.23,
            "coefficients": [0.2, -0.1],
        }

    monkeypatch.setattr("SU2.opt.bspline_su2_driver.BSplineSU2Driver.optimize", fake_optimize)

    result = run_bspline_su2_optimization(
        modes_filename=str(modes_file),
        base_mesh=str(base),
        marker="airfoil",
        def_template=str(def_template),
        primal_template=str(primal_template),
        adjoint_template=str(adjoint_template),
        workdir=str(tmp_path / "run"),
        auto_scale_bounds_to_geometry=True,
        max_normal_displacement=0.4,
        max_rms_normal_displacement=0.25,
            min_bound_scale=0.09,
    )

    assert result["coefficients"] == [0.2, -0.1]
    assert (tmp_path / "run" / "bounds_scaling.json").exists()
    history_text = (tmp_path / "run" / "optimization_history.csv").read_text()
    assert "coeff__upper_a" in history_text
    assert "0.2" in history_text


def test_optimizer_config_parsing_and_cli_precedence(tmp_path, capsys):
    config = tmp_path / "bspline_opt.cfg"
    config.write_text(
        "% External B-spline adaptive SU2 optimizer config\n"
        "\n"
        "OPT_OBJECTIVE= DRAG\n"
        "OPT_ITERATIONS= 3\n"
        "OPT_ACCURACY= 1E-10\n"
        "BSPLINE_AUTO_SCALE_BOUNDS_TO_GEOMETRY= YES\n"
        "BSPLINE_MAX_NORMAL_DISPLACEMENT= 0.03\n"
        "BSPLINE_MAX_RMS_NORMAL_DISPLACEMENT= 0.015\n"
        "BSPLINE_MIN_BOUND_SCALE= 0.05\n"
        "BSPLINE_SHOW_COMMANDS= NO\n"
        "BSPLINE_STREAM_SOLVER_OUTPUT= NO\n"
        "BSPLINE_PRINT_OPTIMIZER_TABLE= YES\n"
        "OPT_RELAX_FACTOR= 0.5\n"
        "OPT_GRADIENT_FACTOR= 2.0\n"
        "OPT_BOUND_LOWER= -0.02\n"
        "OPT_BOUND_UPPER= 0.03\n"
        "OPT_LINE_SEARCH_BOUND= 0.004\n"
        "BSPLINE_EVAL_LAYOUT= DSN\n"
        "BSPLINE_SENSITIVITY_SOURCE= CFD_ADJOINT_SURFACE\n"
        "BSPLINE_GEOMETRY_FD_EPS= 2e-6\n"
        "BSPLINE_SYMMETRY_COUPLING= NORMAL_EQUAL\n"
        "BSPLINE_DEFORMATION_DIRECTION= VERTICAL\n"
        "BSPLINE_LOCAL_STEP_LIMIT= YES\n"
        "BSPLINE_LOCAL_STEP_LIMIT_RATIO= 150.0\n"
        "OPT_CONSTRAINT= (LIFT>0.5)*2.0; (DRAG<0.02)*0.5\n"
    )

    values = parse_optimizer_config(config)
    out = capsys.readouterr().out
    assert values["BSPLINE_AUTO_SCALE_BOUNDS_TO_GEOMETRY"] is True
    assert values["BSPLINE_SHOW_COMMANDS"] is False
    assert values["OPT_ITERATIONS"] == 3
    assert values["OPT_ACCURACY"] == pytest.approx(1.0e-10)
    assert values["OPT_RELAX_FACTOR"] == pytest.approx(0.5)
    assert values["OPT_GRADIENT_FACTOR"] == pytest.approx(2.0)
    assert values["OPT_BOUND_LOWER"] == pytest.approx(-0.02)
    assert values["OPT_BOUND_UPPER"] == pytest.approx(0.03)
    assert values["OPT_LINE_SEARCH_BOUND"] == pytest.approx(0.004)
    assert values["BSPLINE_EVAL_LAYOUT"] == "DSN"
    assert values["BSPLINE_SENSITIVITY_SOURCE"] == "CFD_ADJOINT_SURFACE"
    assert values["BSPLINE_GEOMETRY_FD_EPS"] == pytest.approx(2.0e-6)
    assert values["BSPLINE_SYMMETRY_COUPLING"] == "NORMAL_EQUAL"
    assert values["BSPLINE_DEFORMATION_DIRECTION"] == "VERTICAL"
    assert values["BSPLINE_LOCAL_STEP_LIMIT"] is True
    assert values["BSPLINE_LOCAL_STEP_LIMIT_RATIO"] == pytest.approx(150.0)
    assert values["BSPLINE_MAX_NORMAL_DISPLACEMENT"] == pytest.approx(0.03)
    assert "OPT_RELAX_FACTOR is parsed but not implemented yet" not in out
    assert "OPT_GRADIENT_FACTOR is parsed but not implemented yet" not in out
    assert "OPT_CONSTRAINT is parsed but not implemented yet; ignoring" not in out

    options = fixed_driver_options_from_config(values)
    assert options["objective_column"] == "CD"
    assert options["maxiter"] == 3
    assert options["opt_relax_factor"] == pytest.approx(0.5)
    assert options["opt_gradient_factor"] == pytest.approx(2.0)
    assert options["opt_bound_lower"] == pytest.approx(-0.02)
    assert options["opt_bound_upper"] == pytest.approx(0.03)
    assert options["opt_line_search_bound"] == pytest.approx(0.004)
    assert options["eval_layout"] == "DSN"
    assert options["sensitivity_source"] == "CFD_ADJOINT_SURFACE"
    assert options["geometry_fd_eps"] == pytest.approx(2.0e-6)
    assert options["symmetry_coupling"] == "NORMAL_EQUAL"
    assert options["deformation_direction_mode"] == "VERTICAL"
    assert options["objective_adjoint"] == "drag"
    assert options["local_step_limit"] is True
    assert options["local_step_limit_ratio"] == pytest.approx(150.0)
    assert options["auto_scale_bounds_to_geometry"] is True
    assert options["max_normal_displacement"] == pytest.approx(0.03)
    constraints = options["native_constraints"]
    assert [constraint.name for constraint in constraints] == ["LIFT", "DRAG"]
    assert [constraint.sign for constraint in constraints] == [">", "<"]
    assert [constraint.target for constraint in constraints] == pytest.approx(
        [0.5, 0.02]
    )
    assert [constraint.scale for constraint in constraints] == pytest.approx(
        [2.0, 0.5]
    )

    override_config = tmp_path / "override.cfg"
    override_config.write_text("OPT_OBJECTIVE= DRAG\nOBJECTIVE_COLUMN= MY_OBJ\n")
    assert fixed_driver_options_from_config(parse_optimizer_config(override_config))["objective_column"] == "MY_OBJ"

    parser = _build_arg_parser()
    required = [
        "--modes",
        "modes.json",
        "--base-mesh",
        "mesh.su2",
        "--marker",
        "airfoil",
        "--def-template",
        "def.cfg",
        "--primal-template",
        "primal.cfg",
        "--adjoint-template",
        "adj.cfg",
        "--workdir",
        "run",
        "--optimizer-config",
        str(config),
        "--objective-column",
        "CLI_CD",
        "--eval-layout",
        "DSN",
        "--sensitivity-weighting",
        "NODAL",
        "--local-step-limit-ratio",
        "125.0",
    ]
    args = parser.parse_args(required)
    args = apply_optimizer_config_to_args(args, parser, required, fixed_driver_options_from_config)
    assert args.objective_column == "CLI_CD"
    assert args.maxiter == 3
    assert args.opt_relax_factor == pytest.approx(0.5)
    assert args.opt_gradient_factor == pytest.approx(2.0)
    assert args.opt_bound_lower == pytest.approx(-0.02)
    assert args.opt_bound_upper == pytest.approx(0.03)
    assert args.opt_line_search_bound == pytest.approx(0.004)
    assert args.sensitivity_weighting == "NODAL"
    assert args.sensitivity_source == "CFD_ADJOINT_SURFACE"
    assert args.geometry_fd_eps == pytest.approx(2.0e-6)
    assert args.eval_layout == "DSN"
    assert args.symmetry_coupling == "NORMAL_EQUAL"
    assert args.deformation_direction_mode == "VERTICAL"
    assert args.local_step_limit is True
    assert args.local_step_limit_ratio == pytest.approx(125.0)
    assert args.auto_scale_bounds_to_geometry is True


def test_local_step_limiter_limits_step_not_absolute_coefficient(tmp_path):
    spec = _single_mode_spec(coefficient=0.0)
    # Clamped cubic basis 0 spans knots[0]..knots[4] = [0, 0.125].
    spec["modes"][0]["knot_vector"] = [0, 0, 0, 0, 0.125, 0.25, 0.5, 1, 1, 1, 1]
    spec["modes"][0]["basis_index"] = 0
    driver = _make_driver(
        tmp_path,
        spec=spec,
        local_step_limit=True,
        local_step_limit_ratio=200.0,
    )

    applied, info = driver._apply_line_search_bound([0.01])

    assert info["local_step_limited"] == 1
    assert info["local_step_beta"] == pytest.approx(0.0625)
    assert info["local_step_limit"] == pytest.approx(0.000625)
    assert applied == pytest.approx([0.000625])

    driver._line_search_anchor_physical = [0.005]
    driver._local_step_anchor_reduced = [0.005]
    applied, info = driver._apply_line_search_bound([0.015])

    assert info["local_step_limited"] == 1
    assert info["local_step_beta"] == pytest.approx(0.0625)
    assert applied == pytest.approx([0.005625])


def test_local_step_limiter_is_disabled_by_default(tmp_path):
    spec = _single_mode_spec(coefficient=0.0)
    # Clamped cubic basis 0 spans knots[0]..knots[4] = [0, 0.125].
    spec["modes"][0]["knot_vector"] = [0, 0, 0, 0, 0.125, 0.25, 0.5, 1, 1, 1, 1]
    spec["modes"][0]["basis_index"] = 0
    driver = _make_driver(tmp_path, spec=spec, local_step_limit=False)

    applied, info = driver._apply_line_search_bound([0.01])

    assert info["local_step_limited"] == 0
    assert info["local_step_beta"] == pytest.approx(1.0)
    assert applied == pytest.approx([0.01])


def test_local_step_limiter_uses_reduced_min_support_for_symmetry_coupling(tmp_path):
    spec = _paired_spec(npairs=1)
    # Clamped cubic basis 0 spans knots[0]..knots[4] = [0, 0.125].
    knots = [0, 0, 0, 0, 0.125, 0.25, 0.5, 1, 1, 1, 1]
    spec["modes"][0]["coefficient"] = 0.0
    spec["modes"][0]["knot_vector"] = knots
    spec["modes"][0]["basis_index"] = 0
    spec["modes"][1]["coefficient"] = 0.0
    spec["modes"][1]["knot_vector"] = knots
    spec["modes"][1]["basis_index"] = 0
    driver = _make_driver(
        tmp_path,
        spec=spec,
        symmetry_coupling="NORMAL_EQUAL",
        local_step_limit=True,
        local_step_limit_ratio=200.0,
    )

    applied, info = driver._apply_line_search_bound([0.01, 0.01])

    assert len(driver.reduced_variables) == 1
    assert driver.reduced_local_step_limits == pytest.approx([0.000625])
    assert info["local_step_limited"] == 1
    assert info["local_step_beta"] == pytest.approx(0.0625)
    assert applied == pytest.approx([0.000625, 0.000625])
