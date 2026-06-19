import copy
import csv
import json
import sys
import types

import numpy as np
import pytest

from SU2.opt.bspline_su2_driver import (
    BSplineThicknessConstraint,
    BSplineSU2Driver,
    BSplineSU2DriverError,
    _build_arg_parser,
    active_bounds,
    active_coefficient_vector,
    active_mode_ids,
    apply_optimizer_config_to_args,
    build_eval_commands,
    build_eval_paths,
    build_reduced_variables,
    cache_key,
    collapse_full_gradient,
    collapse_full_jacobian,
    compute_geometry_aware_bound_scaling,
    expand_reduced_coefficients,
    fixed_driver_options_from_config,
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
    assert paths.deform_dir == tmp_path / "eval_0000" / "deform"
    assert paths.direct_dir == tmp_path / "eval_0000" / "direct"
    assert paths.adjoint_dir == tmp_path / "eval_0000" / "adjoint_drag"
    assert "--prefer-vector" in commands["bspline_dot"]
    assert "--sensitivity-weighting" in commands["bspline_dot"]
    assert commands["bspline_dot"][commands["bspline_dot"].index("--sensitivity-weighting") + 1] == "NODAL"


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
    assert "MESH_FILENAME= ../deform/deformed_mesh.su2" in paths.primal_cfg.read_text()
    adjoint_text = paths.adjoint_cfg.read_text()
    assert "MESH_FILENAME= ../deform/deformed_mesh.su2" in adjoint_text
    assert "SOLUTION_FILENAME= ../direct/solution_flow.dat" in adjoint_text
    assert "RESTART_FILENAME= ../direct/restart_flow.dat" in adjoint_text
    assert "SURFACE_ADJ_FILENAME= surface_adjoint" in adjoint_text
    assert "VOLUME_ADJ_FILENAME= volume_adjoint" in adjoint_text


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
    paths.primal_history.write_text("primal")
    paths.adjoint_history.write_text("adjoint history")

    create_eval_aliases(paths)

    assert (paths.eval_dir / "bspline_surface_metadata.csv").read_text() == "metadata"
    assert (paths.eval_dir / "surface_adjoint.csv").read_text() == "adjoint"
    assert (paths.eval_dir / "history_primal.csv").read_text() == "primal"
    assert (paths.eval_dir / "history_adjoint.csv").read_text() == "adjoint history"


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
        "BSPLINE_SYMMETRY_COUPLING= NORMAL_EQUAL\n"
        "BSPLINE_DEFORMATION_DIRECTION= VERTICAL\n"
        "BSPLINE_LOCAL_STEP_LIMIT= YES\n"
        "BSPLINE_LOCAL_STEP_LIMIT_RATIO= 150.0\n"
        "OPT_CONSTRAINT= THICKNESS\n"
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
    assert values["BSPLINE_SYMMETRY_COUPLING"] == "NORMAL_EQUAL"
    assert values["BSPLINE_DEFORMATION_DIRECTION"] == "VERTICAL"
    assert values["BSPLINE_LOCAL_STEP_LIMIT"] is True
    assert values["BSPLINE_LOCAL_STEP_LIMIT_RATIO"] == pytest.approx(150.0)
    assert values["BSPLINE_MAX_NORMAL_DISPLACEMENT"] == pytest.approx(0.03)
    assert "OPT_RELAX_FACTOR is parsed but not implemented yet" not in out
    assert "OPT_GRADIENT_FACTOR is parsed but not implemented yet" not in out
    assert "OPT_CONSTRAINT is parsed but not implemented yet; ignoring" in out

    options = fixed_driver_options_from_config(values)
    assert options["objective_column"] == "CD"
    assert options["maxiter"] == 3
    assert options["opt_relax_factor"] == pytest.approx(0.5)
    assert options["opt_gradient_factor"] == pytest.approx(2.0)
    assert options["opt_bound_lower"] == pytest.approx(-0.02)
    assert options["opt_bound_upper"] == pytest.approx(0.03)
    assert options["opt_line_search_bound"] == pytest.approx(0.004)
    assert options["eval_layout"] == "DSN"
    assert options["symmetry_coupling"] == "NORMAL_EQUAL"
    assert options["deformation_direction_mode"] == "VERTICAL"
    assert options["objective_adjoint"] == "drag"
    assert options["local_step_limit"] is True
    assert options["local_step_limit_ratio"] == pytest.approx(150.0)
    assert options["auto_scale_bounds_to_geometry"] is True
    assert options["max_normal_displacement"] == pytest.approx(0.03)

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
