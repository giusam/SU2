import json
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from SU2.opt.bspline_modes import clamped_basis_value, evaluate_all_modes
from SU2.opt.bspline_su2_adaptive import (
    BSplineAdaptiveError,
    _boehm_insert_once,
    _boehm_insert_to_target_knots,
    _build_arg_parser,
    _check_transferred_coefficients_within_bounds,
    build_next_knot_inserted_modes,
    build_scalar_deformation_sensitivity,
    extract_clamped_knot_space,
    generate_initial_bspline_modes,
    parse_adaptive_options,
    progressive_bspline_su2_shape_optimization,
    regenerate_clamped_modes,
    transfer_shape_to_inserted_space,
    validate_adaptive_options,
)
from SU2.opt.bspline_su2_driver import BSplineSU2Driver
from SU2.opt.progressive_trigger import (
    RefinementTriggered,
    build_online_trigger_opts,
    record_objective_and_check,
)


def _trigger_fires(history, label="PROGRESSIVE_HH", **kwargs):
    opts = build_online_trigger_opts(current_level=0, current_ndv=4, final_ndv=9, **kwargs)
    if opts is None:
        return False
    project = SimpleNamespace(
        trigger_opts=opts,
        trigger_history=[],
        trigger_state=None,
        refinement_triggered=False,
        progressive_label=label,
    )
    fired = False
    for value in history:
        try:
            record_objective_and_check(project, float(value))
        except RefinementTriggered:
            fired = True
            break
    return fired


@pytest.mark.parametrize(
    "trigger,history,extra,expected",
    [
        (
            "SLOPE_EFFICIENCY_TRIGGER",
            [1.0, 0.9, 0.89],
            {"warmup": 5, "window": 1, "tolerance": 0.2},
            False,
        ),
        (
            "SLOPE_EFFICIENCY_TRIGGER",
            [1.0, 0.9, 1.5, 0.89],
            {"warmup": 0, "window": 1, "tolerance": 0.2, "filter_tolerance": 0.02},
            True,
        ),
        (
            "SLOPE_EFFICIENCY_FILTERED",
            [1.0, 0.9, 0.89],
            {"warmup": 0, "window": 1, "tolerance": 0.2},
            True,
        ),
        (
            "SLOPE_EFFICIENCY_BEST_LOG",
            [1.0, 0.8, 0.79, 0.789],
            {"warmup": 0, "window": 1, "tolerance": 0.2, "patience": 2},
            True,
        ),
        (
            "STAGNATION_TRIGGER",
            [1.0, 0.9995, 0.9994, 0.9993],
            {"warmup": 0, "stagnation_window": 3, "stagnation_tolerance": 1.0e-3},
            True,
        ),
    ],
)
def test_shared_trigger_decisions_match_hh_ffd_bspline(trigger, history, extra, expected):
    decisions = [
        _trigger_fires(history, label=label, trigger=trigger, **extra)
        for label in ("PROGRESSIVE_HH", "PROGRESSIVE_FFD", "PROGRESSIVE_BSPLINE")
    ]
    assert decisions == [expected, expected, expected]


def test_max_iter_builds_no_online_trigger():
    assert build_online_trigger_opts("MAX_ITER", current_level=0, current_ndv=4, final_ndv=9) is None


def _minimal_settings(**overrides):
    settings = {
        "refinement": "ADAPTIVE",
        "refine_state": "INITIAL_MESH_KEEP_DV",
        "refine_mode": "KNOT_INSERTION",
        "knot_score_mode": "VIRTUAL_INSERTION",
        "knot_insertions_per_refine": "AUTO",
        "knot_min_span_width": 1.0e-8,
        "trigger": "MAX_ITER",
        "nadd_mode": "GROWTH_RATIO",
        "sensitivity_weighting": "NODAL",
        "eval_layout": "DSN",
        "symmetry_coupling": "NORMAL_EQUAL",
        "nlevels": 1,
        "nfinal": 15,
        "max_iter_per_level": 1,
        "window": 1,
        "tol": 0.2,
        "eps": 1.0e-300,
        "slope_filter_tol": 0.02,
        "slope_patience": 1,
        "stag_window": 3,
        "stag_tol": 1.0e-3,
        "stag_band": 0.02,
        "stag_patience": 1,
        "warmup_iter": 0,
        "growth_ratio": 1.25,
        "fixed_nadd": 1,
        "batch_size_max": 3,
        "objective_column": "CD",
        "objective_adjoint": "drag",
        "workdir": "/tmp/bspline-test",
        "modes": "/tmp/modes.json",
        "base_mesh": "/tmp/mesh.su2",
        "marker": "AIRFOIL",
        "def_template": "/tmp/def.cfg",
        "primal_template": "/tmp/primal.cfg",
        "adjoint_template": "/tmp/adjoint.cfg",
    }
    settings.update(overrides)
    return settings


def _canonical_curve_values(knots, coefficients, degree=3, count=101):
    knots = [float(value) for value in knots]
    x_values = np.linspace(0.0, 1.0, count)
    values = np.asarray(
        [
            sum(
                float(coefficient)
                * clamped_basis_value(x_value, degree, knots, basis_index)
                for basis_index, coefficient in enumerate(coefficients)
            )
            for x_value in x_values
        ],
        dtype=float,
    )
    return x_values, values


def _mode_spec_deformation(spec, metadata):
    values = evaluate_all_modes(
        spec,
        [row["x_over_c"] for row in metadata],
        sides=[row["side"] for row in metadata],
    )
    deformation = np.zeros(len(metadata), dtype=float)
    for mode in spec["modes"]:
        if mode.get("active", True) is False:
            continue
        deformation += float(mode["coefficient"]) * np.asarray(
            values[mode["id"]],
            dtype=float,
        )
    return deformation


def test_boehm_single_cubic_insertion_preserves_canonical_curve():
    old_knots = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    old_coefficients = [0.0, 1.0, 2.0, 3.0]
    new_knots, new_coefficients = _boehm_insert_once(
        old_knots,
        old_coefficients,
        3,
        0.5,
    )

    x_values, old_values = _canonical_curve_values(old_knots, old_coefficients)
    _, new_values = _canonical_curve_values(new_knots, new_coefficients)

    assert x_values[0] == 0.0
    assert np.max(np.abs(new_values - old_values)) <= 1.0e-12


def test_boehm_multiple_insertions_preserve_canonical_curve():
    old_knots = [0.0, 0.0, 0.0, 0.0, 0.4, 1.0, 1.0, 1.0, 1.0]
    old_coefficients = [0.2, -0.1, 0.4, 0.7, -0.3]
    target_knots = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.2,
        0.4,
        0.75,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    new_coefficients = _boehm_insert_to_target_knots(
        old_knots,
        old_coefficients,
        3,
        target_knots,
    )

    _, old_values = _canonical_curve_values(old_knots, old_coefficients)
    _, new_values = _canonical_curve_values(target_knots, new_coefficients)

    assert np.max(np.abs(new_values - old_values)) <= 1.0e-12


def test_boehm_transfer_is_normalization_and_class_shape_aware(tmp_path):
    modes_path = tmp_path / "modes.json"
    spec = generate_initial_bspline_modes(
        modes_path,
        "AIRFOIL",
        nper_side=4,
        class_shape="sqrt_x_one_minus_x",
        normalize_basis=True,
    )
    coefficients = [0.002, -0.004, 0.003, -0.001]
    for mode in spec["modes"]:
        side_sign = 1.0 if mode["side"] == "upper" else -0.6
        mode["coefficient"] = side_sign * coefficients[int(mode["basis_index"])]

    metadata = [
        {"x_over_c": float(x_value), "side": side}
        for side in ("upper", "lower")
        for x_value in np.linspace(0.0, 1.0, 81)
    ]
    settings = validate_adaptive_options(
        _minimal_settings(symmetry_coupling="NONE")
    )
    space = extract_clamped_knot_space(spec, settings)
    target_knots = [0.0, 0.0, 0.0, 0.0, 0.25, 0.7, 1.0, 1.0, 1.0, 1.0]

    coefficients_by_side, diagnostics = transfer_shape_to_inserted_space(
        space,
        metadata,
        target_knots,
        settings=settings,
    )
    transferred = regenerate_clamped_modes(
        space,
        target_knots,
        coefficients_by_side,
    )

    assert diagnostics["transfer_method"] == "BOEHM"
    assert np.max(
        np.abs(
            _mode_spec_deformation(transferred, metadata)
            - _mode_spec_deformation(spec, metadata)
        )
    ) <= 1.0e-12


@pytest.mark.parametrize("surface_mode,side", [("UPPER", "upper"), ("LOWER", "lower")])
def test_boehm_transfer_preserves_single_surface_without_creating_other_side(
    tmp_path,
    surface_mode,
    side,
):
    modes_path = tmp_path / "modes.json"
    spec = generate_initial_bspline_modes(
        modes_path,
        "AIRFOIL",
        nper_side=4,
        surface_mode=surface_mode,
        class_shape="none",
    )
    for mode in spec["modes"]:
        mode["coefficient"] = 0.001 * (int(mode["basis_index"]) + 1)
    metadata = [
        {"x_over_c": float(x_value), "side": side}
        for x_value in np.linspace(0.0, 1.0, 41)
    ]
    settings = validate_adaptive_options(
        _minimal_settings(
            symmetry_coupling="NONE",
            surface_mode=surface_mode,
            nfinal=5,
        )
    )
    space = extract_clamped_knot_space(spec, settings)
    coefficients_by_side, _diagnostics = transfer_shape_to_inserted_space(
        space,
        metadata,
        [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0],
        settings=settings,
    )
    transferred = regenerate_clamped_modes(
        space,
        [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0],
        coefficients_by_side,
    )

    assert set(coefficients_by_side) == {side}
    assert len(transferred["modes"]) == 5
    assert {mode["side"] for mode in transferred["modes"]} == {side}


def test_boehm_transfer_bounds_failure_does_not_clip(tmp_path):
    modes_path = tmp_path / "modes.json"
    spec = generate_initial_bspline_modes(
        modes_path,
        "AIRFOIL",
        nper_side=4,
        class_shape="none",
        normalize_basis=True,
    )
    for mode in spec["modes"]:
        mode["coefficient"] = (
            0.01
            if mode["side"] == "upper" and int(mode["basis_index"]) == 1
            else 0.0
        )
        mode["normalization_factor"] = 0.1

    metadata = [
        {"x_over_c": float(x_value), "side": side}
        for side in ("upper", "lower")
        for x_value in np.linspace(0.0, 1.0, 81)
    ]
    settings = validate_adaptive_options(
        _minimal_settings(
            symmetry_coupling="NONE",
            opt_bound_lower=-0.01,
            opt_bound_upper=0.01,
        )
    )
    space = extract_clamped_knot_space(spec, settings)
    target_knots = [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0]
    coefficients_by_side, _diagnostics = transfer_shape_to_inserted_space(
        space,
        metadata,
        target_knots,
        settings=settings,
    )
    transferred = regenerate_clamped_modes(
        space,
        target_knots,
        coefficients_by_side,
    )
    coefficients_before_check = [
        float(mode["coefficient"]) for mode in transferred["modes"]
    ]

    with pytest.raises(
        BSplineAdaptiveError,
        match="Boehm-transferred coefficient violates optimization bounds",
    ):
        _check_transferred_coefficients_within_bounds(transferred, settings)

    assert [float(mode["coefficient"]) for mode in transferred["modes"]] == (
        coefficients_before_check
    )
    assert max(coefficients_before_check) > 0.01
    assert all(
        "normalization_factor" not in mode for mode in transferred["modes"]
    )


@pytest.mark.parametrize(
    "key,value,message",
    [
        ("eval_layout", "FLAT", "BSPLINE_EVAL_LAYOUT=FLAT has been removed"),
        ("sensitivity_weighting", "DENSITY", "NODAL is fixed internally"),
        ("refine_state", "DEFORMED_MESH_ZERO_DV", "BSPLINE_REFINE_STATE is fixed"),
        ("refine_mode", "CANDIDATE", "BSPLINE_REFINE_MODE is no longer user-configurable"),
        ("nadd_mode", "SCORE_BATCH", "BSPLINE_NADD_MODE=SCORE_BATCH"),
        ("candidate_source", "GENERATED", "removed candidate/generated"),
        ("transfer_method", "LS", "Only BSPLINE_TRANSFER_METHOD=BOEHM"),
        (
            "transfer_bound_policy",
            "CLIP",
            "Only BSPLINE_TRANSFER_BOUND_POLICY=ERROR",
        ),
    ],
)
def test_removed_and_fixed_options_fail_clearly(key, value, message):
    with pytest.raises(BSplineAdaptiveError, match=message):
        validate_adaptive_options(_minimal_settings(**{key: value}))


def test_le_safe_options_are_accepted_and_forwarded():
    settings = validate_adaptive_options(
        _minimal_settings(
            le_safe_direction=True,
            le_safe_x0=0.0,
            le_safe_x1=0.05,
            le_safe_power=2.0,
        )
    )
    assert settings["le_safe_direction"] is True
    assert settings["le_safe_x0"] == 0.0
    assert settings["le_safe_x1"] == 0.05
    assert settings["le_safe_power"] == 2.0


def test_le_safe_options_default_to_disabled():
    settings = validate_adaptive_options(_minimal_settings())
    assert settings["le_safe_direction"] is False
    assert settings["le_safe_x0"] > 0.0
    assert settings["le_safe_x1"] > settings["le_safe_x0"]


def test_vertical_direction_overrides_legacy_le_safe_flag():
    settings = validate_adaptive_options(
        _minimal_settings(
            deformation_direction_mode="VERTICAL",
            le_safe_direction=True,
        )
    )
    assert settings["deformation_direction_mode"] == "VERTICAL"
    assert settings["le_safe_direction"] is False


def test_normal_direction_overrides_legacy_le_safe_flag():
    settings = validate_adaptive_options(
        _minimal_settings(
            deformation_direction_mode="NORMAL",
            le_safe_direction=True,
        )
    )
    assert settings["deformation_direction_mode"] == "NORMAL"
    assert settings["le_safe_direction"] is False


def test_legacy_le_safe_flag_selects_le_safe_mode():
    settings = validate_adaptive_options(
        _minimal_settings(le_safe_direction=True)
    )
    assert settings["deformation_direction_mode"] == "LE_SAFE"
    assert settings["le_safe_direction"] is True


def test_le_safe_invalid_band_rejected():
    with pytest.raises(BSplineAdaptiveError, match="0 <= x0 < x1 <= 1"):
        validate_adaptive_options(
            _minimal_settings(le_safe_direction=True, le_safe_x0=0.05, le_safe_x1=0.01)
        )


def test_le_safe_invalid_power_rejected():
    with pytest.raises(BSplineAdaptiveError, match="positive"):
        validate_adaptive_options(
            _minimal_settings(le_safe_direction=True, le_safe_power=0.0)
        )


def test_ranking_signal_uses_deform_dir_not_pure_normal():
    # The adaptive ranking signal must project the adjoint sensitivity onto
    # the same deformation direction used during deformation (deform_dir_*),
    # not the raw surface normal.
    metadata = [
        {
            "node_id": 1,
            "x_over_c": 0.0,
            "side": "upper",
            "normal_x": 1.0,
            "normal_y": 0.0,
            "deform_dir_x": 0.0,
            "deform_dir_y": 1.0,
        },
        {
            "node_id": 2,
            "x_over_c": 0.5,
            "side": "upper",
            "normal_x": 0.0,
            "normal_y": 1.0,
            "deform_dir_x": 0.0,
            "deform_dir_y": 1.0,
        },
    ]
    sensitivities = [
        {"node_id": 1, "sensitivity_x": 3.0, "sensitivity_y": 7.0},
        {"node_id": 2, "sensitivity_x": 5.0, "sensitivity_y": 11.0},
    ]
    signal = build_scalar_deformation_sensitivity(metadata, sensitivities)
    # node 1: deform_dir=(0,1) -> 7.0 (NOT 3.0 which would be pure normal +x)
    # node 2: deform_dir=(0,1) -> 11.0
    assert list(signal) == pytest.approx([7.0, 11.0])


def test_ranking_signal_falls_back_to_normal_without_deform_dir():
    metadata = [
        {
            "node_id": 1,
            "x_over_c": 0.5,
            "side": "upper",
            "normal_x": 0.0,
            "normal_y": 1.0,
        }
    ]
    sensitivities = [{"node_id": 1, "sensitivity_x": 5.0, "sensitivity_y": 11.0}]
    signal = build_scalar_deformation_sensitivity(metadata, sensitivities)
    assert list(signal) == pytest.approx([11.0])


def test_single_cfg_launch_generates_modes_and_templates_relative_to_cfg(tmp_path):
    cfg = tmp_path / "Config_BSpline_Knot.cfg"
    cfg.write_text(
        "\n".join(
            [
                "MESH_FILENAME= mesh.su2",
                "MARKER_MONITORING= ( AIRFOIL )",
                "MARKER_PLOTTING= ( AIRFOIL )",
                "OBJECTIVE_FUNCTION= DRAG",
                "BSPLINE_WORKDIR= run",
                "BSPLINE_GENERATE_INITIAL_MODES= YES",
                "BSPLINE_INITIAL_NPER_SIDE= 7",
                "OPT_ITERATIONS= 2",
                "OPT_ACCURACY= 1E-6",
                "OPT_BOUND_LOWER= -0.01",
                "OPT_BOUND_UPPER= 0.01",
                "OPT_RELAX_FACTOR= 1000",
                "OPT_GRADIENT_FACTOR= 1E-9",
                "BSPLINE_NLEVELS= 2",
                "BSPLINE_NFINAL= 15",
                "BSPLINE_TRIGGER= MAX_ITER",
                "BSPLINE_NADD_MODE= GROWTH_RATIO",
                "BSPLINE_GROWTH_RATIO= 1.25",
                "BSPLINE_BATCH_SIZE_MAX= 3",
                "BSPLINE_SYMMETRY_COUPLING= NORMAL_EQUAL",
            ]
        )
        + "\n"
    )

    settings = parse_adaptive_options(["-f", str(cfg), "-n", "8"])

    workdir = tmp_path / "run"
    modes = Path(settings["modes"])
    assert modes == workdir / "generated" / "initial_modes.json"
    assert Path(settings["def_template"]) == workdir / "templates" / "def_template_auto.cfg"
    assert Path(settings["primal_template"]) == workdir / "templates" / "primal_template_auto.cfg"
    assert Path(settings["adjoint_template"]) == workdir / "templates" / "adjoint_template_auto.cfg"
    assert settings["base_mesh"] == str((tmp_path / "mesh.su2").resolve())
    assert settings["marker"] == "AIRFOIL"
    assert settings["mpi"] == "mpirun -n 8"

    spec = json.loads(modes.read_text())
    assert len(spec["modes"]) == 14
    assert spec["modes"][0]["knot_vector"] == [0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0]


def test_class_shape_cfg_alias_disables_global_class_shape(tmp_path):
    cfg = tmp_path / "case.cfg"
    cfg.write_text(
        _single_cfg_text(
            tmp_path,
            [
                "BSPLINE_WORKDIR= run_alias",
                "BSPLINE_USE_CLASS_SHAPE= NO",
            ],
        )
    )

    settings = parse_adaptive_options(["-f", str(cfg)])
    spec = json.loads(Path(settings["modes"]).read_text())

    assert spec["class_shape"] == "none"


def test_explicit_initial_class_shape_precedes_cfg_alias(tmp_path):
    cfg = tmp_path / "case.cfg"
    cfg.write_text(
        _single_cfg_text(
            tmp_path,
            [
                "BSPLINE_WORKDIR= run_explicit",
                "BSPLINE_USE_CLASS_SHAPE= NO",
                "BSPLINE_INITIAL_CLASS_SHAPE= sqrt_x_one_minus_x",
            ],
        )
    )

    settings = parse_adaptive_options(["-f", str(cfg)])
    spec = json.loads(Path(settings["modes"]).read_text())

    assert spec["class_shape"] == "sqrt_x_one_minus_x"


def test_cli_paths_override_cfg_paths(tmp_path):
    cfg_modes = tmp_path / "cfg_modes.json"
    cli_modes = tmp_path / "cli_modes.json"
    generate_initial_bspline_modes(cfg_modes, "AIRFOIL")
    generate_initial_bspline_modes(cli_modes, "AIRFOIL")
    for name in ("mesh.su2", "def.cfg", "primal.cfg", "adjoint.cfg"):
        (tmp_path / name).write_text("")
    cfg = tmp_path / "case.cfg"
    cfg.write_text(
        "MESH_FILENAME= cfg_mesh.su2\n"
        "MARKER_MONITORING= ( AIRFOIL )\n"
        "OBJECTIVE_FUNCTION= DRAG\n"
        "BSPLINE_WORKDIR= cfg_run\n"
        f"BSPLINE_MODES= {cfg_modes.name}\n"
        "OPT_ITERATIONS= 1\n"
    )

    settings = parse_adaptive_options(
        [
            "-f",
            str(cfg),
            "--modes",
            str(cli_modes),
            "--base-mesh",
            str(tmp_path / "mesh.su2"),
            "--def-template",
            str(tmp_path / "def.cfg"),
            "--primal-template",
            str(tmp_path / "primal.cfg"),
            "--adjoint-template",
            str(tmp_path / "adjoint.cfg"),
            "--workdir",
            str(tmp_path / "cli_run"),
        ]
    )
    assert settings["modes"] == str(cli_modes.resolve())
    assert settings["base_mesh"] == str((tmp_path / "mesh.su2").resolve())
    assert settings["workdir"] == str((tmp_path / "cli_run").resolve())


def _driver_mode_spec(path):
    generate_initial_bspline_modes(path, "AIRFOIL", nper_side=4)


def test_bspline_driver_online_trigger_stops_cleanly_and_writes_best(tmp_path, monkeypatch):
    modes = tmp_path / "modes.json"
    _driver_mode_spec(modes)
    mesh = tmp_path / "mesh.su2"
    mesh.write_text("mesh")
    for name in ("def.cfg", "primal.cfg", "adjoint.cfg"):
        (tmp_path / name).write_text("")

    optimize_module = types.ModuleType("scipy.optimize")

    def fake_minimize(fun, x0, **_kwargs):
        fun(list(x0))
        fun([value + 1.0e-4 for value in x0])
        fun([value + 2.0e-4 for value in x0])
        raise AssertionError("trigger should stop before fake_minimize returns")

    optimize_module.minimize = fake_minimize
    scipy_module = types.ModuleType("scipy")
    scipy_module.optimize = optimize_module
    monkeypatch.setitem(sys.modules, "scipy", scipy_module)
    monkeypatch.setitem(sys.modules, "scipy.optimize", optimize_module)

    driver = BSplineSU2Driver(
        modes,
        mesh,
        "AIRFOIL",
        tmp_path / "def.cfg",
        tmp_path / "primal.cfg",
        tmp_path / "adjoint.cfg",
        tmp_path / "run",
        trigger_opts=build_online_trigger_opts(
            "SLOPE_EFFICIENCY_TRIGGER",
            current_level=0,
            current_ndv=4,
            final_ndv=5,
            window=1,
            tolerance=0.2,
            warmup=0,
        ),
    )
    objectives = [1.0, 0.9, 0.89]

    def fake_evaluate(coefficients, line_search_info=None):
        index = len(driver._history_records)
        objective = objectives[index]
        gradient = [0.0] * len(driver.mode_ids)
        result = {
            "eval_index": index + 1,
            "eval_id": index,
            "eval_dir": str(driver.workdir / f"eval_{index:04d}"),
            "objective": objective,
            "gradient": gradient,
            "coefficients": list(coefficients),
            "status": "ok",
        }
        driver._append_history_record(
            index,
            objective,
            coefficients,
            gradient,
            "ok",
            line_search_info=line_search_info,
            eval_dir=driver.workdir / f"eval_{index:04d}",
            eval_index=index + 1,
        )
        return result

    monkeypatch.setattr(driver, "evaluate", fake_evaluate)

    result = driver.optimize(maxiter=10)

    assert result["status"] == "early_refine_trigger"
    assert result["success"] is True
    assert result["objective"] == pytest.approx(0.89)
    assert (driver.workdir / "optimization_history.csv").exists()
    assert (driver.workdir / "optimized_modes.json").exists()


def test_knot_insertion_growth_ratio_and_fixed_remain_supported(tmp_path):
    modes = tmp_path / "modes.json"
    generate_initial_bspline_modes(modes, "AIRFOIL", nper_side=4)
    spec = json.loads(modes.read_text())
    metadata = [
        {"x_over_c": 0.2, "side": "upper"},
        {"x_over_c": 0.4, "side": "upper"},
        {"x_over_c": 0.6, "side": "upper"},
        {"x_over_c": 0.8, "side": "upper"},
        {"x_over_c": 0.2, "side": "lower"},
        {"x_over_c": 0.4, "side": "lower"},
        {"x_over_c": 0.6, "side": "lower"},
        {"x_over_c": 0.8, "side": "lower"},
    ]
    signal = [math.sin(5.0 * math.pi * row["x_over_c"]) for row in metadata]
    settings = validate_adaptive_options(
        _minimal_settings(
            nadd_mode="FIXED",
            fixed_nadd=1,
            nfinal=5,
            knot_score_mode="RESIDUAL_ENERGY",
            symmetry_coupling="NORMAL_EQUAL",
        )
    )

    next_modes, rows, selected = build_next_knot_inserted_modes(spec, metadata, signal, settings)

    assert next_modes is not None
    assert rows
    assert selected["refine_mode"] == "KNOT_INSERTION"
    assert selected["reduced_ndv_after"] == selected["reduced_ndv_before"] + 1


@pytest.mark.parametrize("surface_mode,side", [("UPPER", "upper"), ("LOWER", "lower")])
def test_knot_insertion_refines_only_active_half_domain(
    tmp_path,
    surface_mode,
    side,
):
    modes = tmp_path / "modes.json"
    spec = generate_initial_bspline_modes(
        modes,
        "AIRFOIL",
        nper_side=4,
        surface_mode=surface_mode,
        class_shape="none",
    )
    metadata = [
        {"x_over_c": float(x_value), "side": side}
        for x_value in np.linspace(0.05, 0.95, 19)
    ]
    signal = [math.sin(5.0 * math.pi * row["x_over_c"]) for row in metadata]
    settings = validate_adaptive_options(
        _minimal_settings(
            symmetry_coupling="NONE",
            surface_mode=surface_mode,
            nfinal=5,
            nadd_mode="FIXED",
            fixed_nadd=1,
            knot_insertions_per_refine=1,
            knot_score_mode="RESIDUAL_ENERGY",
        )
    )

    next_modes, rows, selected = build_next_knot_inserted_modes(
        spec,
        metadata,
        signal,
        settings,
    )

    assert next_modes is not None
    assert len(next_modes["modes"]) == 5
    assert {mode["side"] for mode in next_modes["modes"]} == {side}
    assert {row["side"] for row in rows} == {surface_mode}
    assert selected["side"] == surface_mode
    assert selected["ndv_before"] == 4
    assert selected["ndv_after"] == 5


@pytest.mark.parametrize("surface_mode,side", [("UPPER", "upper"), ("LOWER", "lower")])
def test_generate_initial_modes_contains_only_requested_surface(
    tmp_path,
    surface_mode,
    side,
):
    path = tmp_path / f"{side}.json"
    spec = generate_initial_bspline_modes(
        path,
        "AIRFOIL",
        nper_side=7,
        surface_mode=surface_mode,
    )

    assert spec["surface_mode"] == surface_mode
    assert len(spec["modes"]) == 7
    assert {mode["side"] for mode in spec["modes"]} == {side}


def test_default_initial_mode_generation_remains_both_surfaces(tmp_path):
    spec = generate_initial_bspline_modes(
        tmp_path / "both.json",
        "AIRFOIL",
        nper_side=7,
    )

    assert spec["surface_mode"] == "BOTH"
    assert len(spec["modes"]) == 14
    assert {mode["side"] for mode in spec["modes"]} == {"upper", "lower"}


def _single_cfg_text(tmp_path, extra_lines=()):
    lines = [
        "MESH_FILENAME= mesh.su2",
        "MARKER_MONITORING= ( AIRFOIL )",
        "MARKER_PLOTTING= ( AIRFOIL )",
        "OBJECTIVE_FUNCTION= DRAG",
        "BSPLINE_WORKDIR= run",
        "BSPLINE_GENERATE_INITIAL_MODES= YES",
        "BSPLINE_INITIAL_NPER_SIDE= 7",
        "OPT_ITERATIONS= 2",
    ]
    lines.extend(extra_lines)
    return "\n".join(lines) + "\n"


def test_transfer_cfg_options_are_parsed(tmp_path):
    cfg = tmp_path / "case.cfg"
    cfg.write_text(
        _single_cfg_text(
            tmp_path,
            [
                "BSPLINE_TRANSFER_METHOD= BOEHM",
                "BSPLINE_TRANSFER_BOUND_POLICY= ERROR",
                "BSPLINE_TRANSFER_GEOMETRY_ABS_TOL= 2E-11",
                "BSPLINE_TRANSFER_GEOMETRY_REL_TOL= 3E-9",
            ],
        )
    )

    settings = parse_adaptive_options(["-f", str(cfg)])

    assert settings["transfer_method"] == "BOEHM"
    assert settings["transfer_bound_policy"] == "ERROR"
    assert settings["transfer_geometry_abs_tol"] == pytest.approx(2.0e-11)
    assert settings["transfer_geometry_rel_tol"] == pytest.approx(3.0e-9)


def test_vertical_deformation_direction_is_read_from_cfg(tmp_path):
    cfg = tmp_path / "case.cfg"
    cfg.write_text(
        _single_cfg_text(
            tmp_path,
            [
                "BSPLINE_DEFORMATION_DIRECTION= VERTICAL",
                "BSPLINE_LE_SAFE_DIRECTION= YES",
            ],
        )
    )

    settings = parse_adaptive_options(["-f", str(cfg)])

    assert settings["deformation_direction_mode"] == "VERTICAL"
    assert settings["le_safe_direction"] is False


@pytest.mark.parametrize(
    "surface_mode,expected_side,expected_domain",
    [
        ("UPPER", "upper", "HALF_UPPER"),
        ("LOWER", "lower", "HALF_LOWER"),
    ],
)
def test_surface_mode_cfg_generates_half_domain_and_resolves_thickness_auto(
    tmp_path,
    surface_mode,
    expected_side,
    expected_domain,
):
    cfg = tmp_path / "case.cfg"
    cfg.write_text(
        _single_cfg_text(
            tmp_path,
            [
                f"BSPLINE_SURFACE_MODE= {surface_mode}",
                "BSPLINE_SYMMETRY_COUPLING= NONE",
                "PROGRESSIVE_THICKNESS_CONSTRAINT= YES",
                "PROGRESSIVE_THICKNESS_REF_MESH= mesh.su2",
            ],
        )
    )

    settings = parse_adaptive_options(["-f", str(cfg)])
    spec = json.loads(Path(settings["modes"]).read_text())

    assert settings["surface_mode"] == surface_mode
    assert settings["thickness_options"]["PROGRESSIVE_THICKNESS_DOMAIN_MODE"] == expected_domain
    assert len(spec["modes"]) == 7
    assert {mode["side"] for mode in spec["modes"]} == {expected_side}


@pytest.mark.parametrize("surface_mode", ["UPPER", "LOWER"])
def test_adaptive_half_domain_rejects_symmetry_coupling(surface_mode):
    with pytest.raises(
        BSplineAdaptiveError,
        match="BSPLINE_SYMMETRY_COUPLING is only valid with BSPLINE_SURFACE_MODE=BOTH",
    ):
        validate_adaptive_options(
            _minimal_settings(
                surface_mode=surface_mode,
                symmetry_coupling="NORMAL_EQUAL",
            )
        )


def test_candidate_bank_is_not_required():
    parser = _build_arg_parser()
    args = parser.parse_args([])
    assert args.candidate_bank is None


def test_candidate_bank_flag_is_ignored_with_warning(tmp_path, capsys):
    cfg = tmp_path / "case.cfg"
    cfg.write_text(_single_cfg_text(tmp_path))
    settings = parse_adaptive_options(["-f", str(cfg), "--candidate-bank", str(tmp_path / "bank.json")])
    progressive_bspline_su2_shape_optimization(dict(settings, dry_run=True))
    out = capsys.readouterr().out
    assert "--candidate-bank is ignored" in out


@pytest.mark.parametrize(
    "option",
    [
        "--candidate-source",
        "--generated-peaks-per-side",
        "--generated-widths",
        "--generated-min-separation",
        "--generated-xmin",
        "--generated-xmax",
        "--edge-xle",
        "--edge-xte",
        "--rough-lambda",
        "--rough-power",
        "--batch-score-rel-tol",
        "--score-mode",
    ],
)
def test_legacy_candidate_cli_options_are_removed(option):
    parser = _build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([option, "1"])


@pytest.mark.parametrize(
    "cfg_key,cfg_value",
    [
        ("BSPLINE_SCORE_MODE", "ADJOINT_RESIDUALIZED_BSPLINE"),
        ("BSPLINE_CANDIDATE_SOURCE", "GENERATED"),
        ("BSPLINE_GENERATED_PEAKS_PER_SIDE", "3"),
        ("BSPLINE_GENERATED_WIDTHS", "( 0.02, 0.04 )"),
        ("BSPLINE_GENERATED_MIN_SEPARATION", "0.05"),
        ("BSPLINE_GENERATED_XMIN", "0.01"),
        ("BSPLINE_GENERATED_XMAX", "0.97"),
        ("BSPLINE_EDGE_XLE", "0.01"),
        ("BSPLINE_EDGE_XTE", "0.97"),
        ("BSPLINE_ROUGH_LAMBDA", "1.0"),
        ("BSPLINE_ROUGH_POWER", "2.0"),
        ("BSPLINE_BATCH_SCORE_REL_TOL", "0.85"),
    ],
)
def test_legacy_candidate_cfg_keys_fail_clearly(tmp_path, cfg_key, cfg_value):
    cfg = tmp_path / "case.cfg"
    cfg.write_text(_single_cfg_text(tmp_path, [f"{cfg_key}= {cfg_value}"]))
    with pytest.raises(BSplineAdaptiveError, match="removed candidate/generated"):
        parse_adaptive_options(["-f", str(cfg)])


def test_default_adaptive_sensitivity_weighting_is_nodal():
    settings = validate_adaptive_options(_minimal_settings())
    assert settings["sensitivity_weighting"] == "NODAL"


@pytest.mark.parametrize(
    "key,value",
    [
        ("refine_mode", "KNOT_INSERTION"),
        ("refine_state", "INITIAL_MESH_KEEP_DV"),
        ("sensitivity_weighting", "NODAL"),
    ],
)
def test_canonical_fixed_options_accepted(key, value):
    settings = validate_adaptive_options(_minimal_settings(**{key: value}))
    assert settings[key] == value


def test_nproc_builds_mpirun_and_mpi_flag_wins(tmp_path):
    cfg = tmp_path / "case.cfg"
    cfg.write_text(_single_cfg_text(tmp_path))
    settings = parse_adaptive_options(["-f", str(cfg), "-n", "8"])
    assert settings["mpi"] == "mpirun -n 8"

    settings = parse_adaptive_options(["-f", str(cfg), "-n", "8", "--mpi", "srun -n 4"])
    assert settings["mpi"] == "srun -n 4"


def test_generate_initial_modes_writes_expected_knot_vector(tmp_path):
    cfg = tmp_path / "case.cfg"
    cfg.write_text(_single_cfg_text(tmp_path))
    settings = parse_adaptive_options(["-f", str(cfg)])

    workdir = tmp_path / "run"
    modes = Path(settings["modes"])
    assert modes == workdir / "generated" / "initial_modes.json"
    assert settings["_initial_modes_generated"] is True
    spec = json.loads(modes.read_text())
    assert len(spec["modes"]) == 14
    assert spec["modes"][0]["knot_vector"] == [0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0]


def test_auto_templates_are_generated_under_workdir(tmp_path):
    cfg = tmp_path / "case.cfg"
    cfg.write_text(_single_cfg_text(tmp_path))
    settings = parse_adaptive_options(["-f", str(cfg)])

    templates = tmp_path / "run" / "templates"
    assert Path(settings["def_template"]) == templates / "def_template_auto.cfg"
    assert Path(settings["primal_template"]) == templates / "primal_template_auto.cfg"
    assert Path(settings["adjoint_template"]) == templates / "adjoint_template_auto.cfg"

    def_text = (templates / "def_template_auto.cfg").read_text()
    assert "DV_KIND= SURFACE_FILE" in def_text
    assert "DV_FILENAME= surface_positions.dat" in def_text
    assert "DV_MARKER= ( AIRFOIL )" in def_text

    primal_text = (templates / "primal_template_auto.cfg").read_text()
    assert "MATH_PROBLEM= DIRECT" in primal_text
    adjoint_text = (templates / "adjoint_template_auto.cfg").read_text()
    assert "MATH_PROBLEM= DISCRETE_ADJOINT" in adjoint_text


def test_adaptive_settings_json_records_resolved_paths(tmp_path):
    cfg = tmp_path / "case.cfg"
    cfg.write_text(_single_cfg_text(tmp_path))
    settings = parse_adaptive_options(["-f", str(cfg)])
    progressive_bspline_su2_shape_optimization(dict(settings, dry_run=True))

    recorded = json.loads((tmp_path / "run" / "adaptive_settings.json").read_text())
    assert recorded["base_mesh"] == str((tmp_path / "mesh.su2").resolve())
    assert recorded["modes"] == str((tmp_path / "run" / "generated" / "initial_modes.json").resolve())
    assert recorded["eval_layout"] == "DSN"
    assert recorded["sensitivity_weighting"] == "NODAL"
    assert recorded["refine_mode"] == "KNOT_INSERTION"
    assert recorded["refine_state"] == "INITIAL_MESH_KEEP_DV"
