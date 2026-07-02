import csv
import json
import subprocess
import sys

import numpy as np
import pytest

from SU2.opt.bspline_adaptive.geometry_sensitivity_fields import thickness_station_field
from SU2.opt.bspline_adaptive.ikkt import (
    build_ikkt_residual_field,
    build_ikkt_score_signal,
    estimate_ikkt_multipliers,
    lambda_bounds_for_sign,
    scale_constraint_field_for_ikkt,
    scale_objective_field_for_ikkt,
)
from SU2.opt.bspline_adaptive.knot_space import extract_clamped_knot_space
from SU2.opt.bspline_adaptive.scoring import score_knot_spans
from SU2.opt.bspline_su2_adaptive import generate_initial_bspline_modes


def _meta(x, y, side="upper", x_over_c=None):
    return {
        "x": float(x),
        "y": float(y),
        "deformed_x": float(x),
        "deformed_y": float(y),
        "x_over_c": float(x if x_over_c is None else x_over_c),
        "side": side,
        "normal_x": 0.0,
        "normal_y": 1.0,
        "deform_dir_x": 0.0,
        "deform_dir_y": 1.0,
        "deformation_direction_mode": "VERTICAL",
        "weight": 1.0,
    }


def _write_metadata_csv(path, rows):
    with open(path, "w", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "global_index",
                "x",
                "y",
                "x_over_c",
                "side",
                "normal_x",
                "normal_y",
                "deform_dir_x",
                "deform_dir_y",
                "deformation_direction_mode",
                "deformed_x",
                "deformed_y",
                "weight",
            ],
        )
        writer.writeheader()
        for index, row in enumerate(rows):
            data = dict(row)
            data["global_index"] = index
            writer.writerow(data)


def _write_sens_csv(path, nrows):
    with open(path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["global_index", "Sensitivity_x", "Sensitivity_y"])
        writer.writeheader()
        for index in range(nrows):
            writer.writerow(
                {
                    "global_index": index,
                    "Sensitivity_x": 0.0,
                    "Sensitivity_y": 1.0 + index,
                }
            )


def _write_sens_csv_values(path, values):
    with open(path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["global_index", "Sensitivity_x", "Sensitivity_y"])
        writer.writeheader()
        for index, value in enumerate(values):
            writer.writerow(
                {
                    "global_index": index,
                    "Sensitivity_x": 0.0,
                    "Sensitivity_y": float(value),
                }
            )


def _write_history(path, **values):
    with open(path, "w", newline="") as fp:
        fieldnames = ["ITER", *values.keys()]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        row = {"ITER": 1}
        row.update(values)
        writer.writerow(row)


def _closed_airfoil_metadata():
    return [
        _meta(0.0, 0.0, side="upper", x_over_c=0.0),
        _meta(0.5, 0.1, side="upper", x_over_c=0.5),
        _meta(1.0, 0.0, side="upper", x_over_c=1.0),
        _meta(0.5, -0.1, side="lower", x_over_c=0.5),
    ]


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
        "3 5 0\n",
        encoding="utf-8",
    )


def test_ikkt_bounds_and_residual_for_slsqp_ge_raw():
    assert lambda_bounds_for_sign(">", "SLSQP_GE_RAW") == (0.0, np.inf)
    assert lambda_bounds_for_sign("<", "SLSQP_GE_RAW") == (-np.inf, 0.0)
    assert lambda_bounds_for_sign(">", "HH_RAW") == (-np.inf, 0.0)

    residual = build_ikkt_residual_field(
        np.array([2.0, 4.0]),
        [np.array([0.5, 1.0])],
        [3.0],
    )
    assert residual == pytest.approx([0.5, 1.0])


def test_slsqp_ge_raw_allows_positive_thickness_multiplier():
    phi = np.eye(2)
    g_constraint = np.array([1.0, 2.0])
    g_objective = 3.0 * g_constraint

    hh_lambdas, _ = estimate_ikkt_multipliers(
        g_objective,
        [g_constraint],
        phi,
        [lambda_bounds_for_sign(">", "HH_RAW")],
    )
    slsqp_lambdas, _ = estimate_ikkt_multipliers(
        g_objective,
        [g_constraint],
        phi,
        [lambda_bounds_for_sign(">", "SLSQP_GE_RAW")],
    )

    assert hh_lambdas[0] == pytest.approx(0.0)
    assert slsqp_lambdas[0] == pytest.approx(3.0)


def test_physical_scaling_does_not_apply_opt_gradient_factor():
    field = np.array([1.0, 2.0, 3.0])
    settings = {"ikkt_scaling_mode": "PHYSICAL", "opt_gradient_factor": 10.0}

    objective, obj_scale = scale_objective_field_for_ikkt(field, settings)
    constraint, con_scale = scale_constraint_field_for_ikkt(
        field,
        settings,
        source="PROGRESSIVE_THICKNESS",
    )

    assert obj_scale == pytest.approx(1.0)
    assert con_scale == pytest.approx(1.0)
    assert objective == pytest.approx(field)
    assert constraint == pytest.approx(field)


def test_half_thickness_fields_have_no_factor_two():
    metadata = [_meta(0.0, 0.1), _meta(1.0, 0.3)]

    upper = thickness_station_field(
        metadata,
        0.25,
        domain_mode="HALF_UPPER",
        symmetry_y=0.0,
    )
    lower = thickness_station_field(
        metadata,
        0.25,
        domain_mode="HALF_LOWER",
        symmetry_y=0.5,
    )

    assert upper["current_measure"] == pytest.approx(0.15)
    assert upper["field"] == pytest.approx([0.75, 0.25])
    assert lower["current_measure"] == pytest.approx(0.35)
    assert lower["field"] == pytest.approx([-0.75, -0.25])


def test_ikkt_no_active_constraints_matches_virtual_insertion(tmp_path):
    spec = generate_initial_bspline_modes(
        tmp_path / "modes.json",
        "AIRFOIL",
        nper_side=4,
        surface_mode="UPPER",
    )
    metadata = [_meta(x, 0.1 * x) for x in [0.0, 0.25, 0.5, 0.75, 1.0]]
    signal = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    base_settings = {
        "surface_mode": "UPPER",
        "symmetry_coupling": "NONE",
        "sensitivity_weighting": "NODAL",
        "deformation_direction_mode": "VERTICAL",
        "knot_min_span_width": 1.0e-8,
    }
    space = extract_clamped_knot_space(spec, base_settings)
    objective_rows = score_knot_spans(
        space,
        metadata,
        signal,
        {**base_settings, "knot_score_mode": "VIRTUAL_INSERTION"},
    )
    ikkt_signal, diagnostics = build_ikkt_score_signal(
        spec,
        metadata,
        signal,
        {
            **base_settings,
            "knot_score_mode": "IKKT_VIRTUAL_INSERTION",
            "ikkt_include_geometry_constraints": False,
            "ikkt_include_aero_constraints": False,
            "ikkt_require_available_fields": True,
            "ikkt_scaling_mode": "PHYSICAL",
            "ikkt_sign_convention": "SLSQP_GE_RAW",
        },
    )
    ikkt_rows = score_knot_spans(
        space,
        metadata,
        ikkt_signal,
        {
            **base_settings,
            "knot_score_mode": "IKKT_VIRTUAL_INSERTION",
            "_ikkt_objective_signal": signal,
        },
    )

    assert diagnostics["status"] == "objective_only_no_active_constraints"
    assert [row["rank"] for row in ikkt_rows] == [row["rank"] for row in objective_rows]
    assert [row["score"] for row in ikkt_rows] == pytest.approx(
        [row["score"] for row in objective_rows]
    )


def test_offline_probe_config_includes_active_progressive_thickness(tmp_path):
    modes_file = tmp_path / "modes.json"
    generate_initial_bspline_modes(
        modes_file,
        "airfoil",
        nper_side=4,
        surface_mode="UPPER",
    )
    mesh_file = tmp_path / "mesh.su2"
    _write_simple_airfoil_mesh(mesh_file)
    metadata_file = tmp_path / "metadata.csv"
    sens_file = tmp_path / "surface_sens.csv"
    output_file = tmp_path / "ikkt_probe.csv"
    config_file = tmp_path / "Config_BSpline.cfg"
    config_file.write_text(
        "\n".join(
            [
                f"BSPLINE_BASE_MESH= {mesh_file}",
                "BSPLINE_MARKER= airfoil",
                f"BSPLINE_WORKDIR= {tmp_path / 'work'}",
                f"BSPLINE_MODES= {modes_file}",
                "BSPLINE_SURFACE_MODE= UPPER",
                "BSPLINE_DEFORMATION_DIRECTION= VERTICAL",
                "BSPLINE_SYMMETRY_COUPLING= NONE",
                "BSPLINE_KNOT_SCORE_MODE= IKKT_VIRTUAL_INSERTION",
                "BSPLINE_IKKT_INCLUDE_GEOMETRY_CONSTRAINTS= YES",
                "BSPLINE_IKKT_INCLUDE_AERO_CONSTRAINTS= NO",
                "BSPLINE_IKKT_REQUIRE_AVAILABLE_FIELDS= YES",
                "BSPLINE_IKKT_SCALING_MODE= PHYSICAL",
                "BSPLINE_IKKT_SIGN_CONVENTION= SLSQP_GE_RAW",
                "PROGRESSIVE_THICKNESS_CONSTRAINT= YES",
                f"PROGRESSIVE_THICKNESS_REF_MESH= {mesh_file}",
                "PROGRESSIVE_THICKNESS_DOMAIN_MODE= HALF_UPPER",
                "PROGRESSIVE_THICKNESS_X_STATIONS= 0.5",
                "PROGRESSIVE_THICKNESS_MARGIN= 0.0",
                "PROGRESSIVE_THICKNESS_SYMMETRY_Y= 0.0",
                "PROGRESSIVE_THICKNESS_MARKER= airfoil",
                f"PROGRESSIVE_THICKNESS_CACHE_FILE= {tmp_path / 'thickness_reference.npz'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    metadata = [
        _meta(0.0, 0.0),
        _meta(0.25, 0.1),
        _meta(0.5, 0.1),
        _meta(0.75, 0.1),
        _meta(1.0, 0.0),
    ]
    _write_metadata_csv(metadata_file, metadata)
    _write_sens_csv(sens_file, len(metadata))

    subprocess.run(
        [
            sys.executable,
            "tools/probe_bspline_ikkt_score.py",
            "--config",
            str(config_file),
            "--surface-sens",
            str(sens_file),
            "--metadata",
            str(metadata_file),
            "--output",
            str(output_file),
        ],
        check=True,
    )

    diagnostics = json.loads(output_file.with_suffix(".diagnostics.json").read_text())
    names = [item["name"] for item in diagnostics["included_constraints"]]
    assert names == ["PROGRESSIVE_THICKNESS[0]"]
    constraint = diagnostics["included_constraints"][0]
    assert constraint["lambda_bounds"] == [0.0, "inf"]
    assert constraint["lambda"] >= 0.0
    assert constraint["gap"] == pytest.approx(0.0)
    assert diagnostics["scaling_policy"]["scaling_mode"] == "PHYSICAL"
    assert diagnostics["sign_convention"] == "SLSQP_GE_RAW"

    rows = list(csv.DictReader(open(output_file)))
    assert rows
    assert "score_objective" in rows[0]
    assert "score_ikkt" in rows[0]


def test_ikkt_aero_less_than_constraint_uses_internal_c_ge_zero(tmp_path):
    spec = generate_initial_bspline_modes(
        tmp_path / "modes.json",
        "AIRFOIL",
        nper_side=4,
        surface_mode="UPPER",
    )
    metadata = [_meta(x, 0.1 * x) for x in [0.0, 0.25, 0.5, 0.75, 1.0]]
    eval_dir = tmp_path / "eval_0000"
    adjoint_dir = eval_dir / "adjoint_momentz"
    adjoint_dir.mkdir(parents=True)
    _write_sens_csv_values(adjoint_dir / "surface_sens.csv", [-1.0, -2.0, -3.0, -4.0, -5.0])
    _write_history(eval_dir / "history_primal.csv", CMz=0.094)

    signal, diagnostics = build_ikkt_score_signal(
        spec,
        metadata,
        np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float),
        {
            "surface_mode": "UPPER",
            "symmetry_coupling": "NONE",
            "deformation_direction_mode": "VERTICAL",
            "knot_score_mode": "IKKT_VIRTUAL_INSERTION",
            "ikkt_include_geometry_constraints": False,
            "ikkt_include_aero_constraints": True,
            "ikkt_require_available_fields": True,
            "ikkt_scaling_mode": "PHYSICAL",
            "ikkt_sign_convention": "SLSQP_GE_RAW",
            "native_constraints": "(MOMENT_Z<0.093)*0.001",
            "_ikkt_eval_dir": str(eval_dir),
        },
    )

    constraint = diagnostics["included_constraints"][0]
    assert constraint["name"] == "AERO[MOMENT_Z]"
    assert constraint["original_operator"] == "<"
    assert constraint["c_value"] == pytest.approx(-0.001)
    assert constraint["field_sign"] == pytest.approx(-1.0)
    assert constraint["lambda_bounds"] == [0.0, "inf"]
    assert constraint["config_scale"] == pytest.approx(0.001)
    assert constraint["used_in_ikkt_gradient"] is False
    assert diagnostics["least_squares"]["n_constraints"] == 1
    assert "grad_constraint_rank" in diagnostics["least_squares"]
    assert np.linalg.norm(signal) < np.linalg.norm([1.0, 2.0, 3.0, 4.0, 5.0])


def test_ikkt_aero_inactive_missing_field_is_skipped_with_require_yes(tmp_path):
    spec = generate_initial_bspline_modes(
        tmp_path / "modes.json",
        "AIRFOIL",
        nper_side=4,
        surface_mode="UPPER",
    )
    metadata = [_meta(x, 0.1 * x) for x in [0.0, 0.25, 0.5, 0.75, 1.0]]
    eval_dir = tmp_path / "eval_0000"
    eval_dir.mkdir()
    _write_history(eval_dir / "history_primal.csv", CMz=0.091)
    objective = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)

    signal, diagnostics = build_ikkt_score_signal(
        spec,
        metadata,
        objective,
        {
            "surface_mode": "UPPER",
            "symmetry_coupling": "NONE",
            "deformation_direction_mode": "VERTICAL",
            "knot_score_mode": "IKKT_VIRTUAL_INSERTION",
            "ikkt_include_geometry_constraints": False,
            "ikkt_include_aero_constraints": True,
            "ikkt_require_available_fields": True,
            "ikkt_scaling_mode": "PHYSICAL",
            "native_constraints": "(MOMENT_Z<0.093)*0.001",
            "_ikkt_eval_dir": str(eval_dir),
        },
    )

    assert signal == pytest.approx(objective)
    assert diagnostics["status"] == "objective_only_no_active_constraints"
    assert diagnostics["skipped_constraints"][0]["reason"] == "inactive"
    assert diagnostics["skipped_constraints"][0]["c_value"] == pytest.approx(0.002)


def test_ikkt_aero_active_missing_field_policy(tmp_path):
    spec = generate_initial_bspline_modes(
        tmp_path / "modes.json",
        "AIRFOIL",
        nper_side=4,
        surface_mode="UPPER",
    )
    metadata = [_meta(x, 0.1 * x) for x in [0.0, 0.25, 0.5, 0.75, 1.0]]
    eval_dir = tmp_path / "eval_0000"
    eval_dir.mkdir()
    _write_history(eval_dir / "history_primal.csv", CMz=0.094)
    settings = {
        "surface_mode": "UPPER",
        "symmetry_coupling": "NONE",
        "deformation_direction_mode": "VERTICAL",
        "knot_score_mode": "IKKT_VIRTUAL_INSERTION",
        "ikkt_include_geometry_constraints": False,
        "ikkt_include_aero_constraints": True,
        "ikkt_scaling_mode": "PHYSICAL",
        "native_constraints": "(MOMENT_Z<0.093)",
        "_ikkt_eval_dir": str(eval_dir),
    }

    with pytest.raises(Exception, match="AERO\\[MOMENT_Z\\]:active_field_unavailable"):
        build_ikkt_score_signal(
            spec,
            metadata,
            np.ones(len(metadata)),
            {**settings, "ikkt_require_available_fields": True},
        )

    signal, diagnostics = build_ikkt_score_signal(
        spec,
        metadata,
        np.ones(len(metadata)),
        {**settings, "ikkt_require_available_fields": False},
    )
    assert signal == pytest.approx(np.ones(len(metadata)))
    assert diagnostics["unsupported_constraints"][0]["reason"] == "active_field_unavailable"


def test_offline_probe_includes_aero_constraint_from_eval_dir_and_values_json(tmp_path):
    modes_file = tmp_path / "modes.json"
    generate_initial_bspline_modes(
        modes_file,
        "airfoil",
        nper_side=4,
        surface_mode="UPPER",
    )
    eval_dir = tmp_path / "eval_0000"
    objective_dir = eval_dir / "adjoint_drag"
    moment_dir = eval_dir / "adjoint_momentz"
    objective_dir.mkdir(parents=True)
    moment_dir.mkdir(parents=True)
    metadata_file = eval_dir / "bspline_surface_metadata.csv"
    metadata = [
        _meta(0.0, 0.0),
        _meta(0.25, 0.1),
        _meta(0.5, 0.1),
        _meta(0.75, 0.1),
        _meta(1.0, 0.0),
    ]
    _write_metadata_csv(metadata_file, metadata)
    _write_sens_csv_values(objective_dir / "surface_sens.csv", [1, 2, 3, 4, 5])
    _write_sens_csv_values(moment_dir / "surface_sens.csv", [-1, -2, -3, -4, -5])
    values_file = tmp_path / "values.json"
    values_file.write_text(json.dumps({"MOMENT_Z": 0.094}), encoding="utf-8")
    output_file = tmp_path / "ikkt_aero_probe.csv"
    config_file = tmp_path / "Config_BSpline.cfg"
    config_file.write_text(
        "\n".join(
            [
                f"BSPLINE_BASE_MESH= {tmp_path / 'mesh.su2'}",
                "BSPLINE_MARKER= airfoil",
                f"BSPLINE_WORKDIR= {tmp_path / 'work'}",
                f"BSPLINE_MODES= {modes_file}",
                "BSPLINE_SURFACE_MODE= UPPER",
                "BSPLINE_DEFORMATION_DIRECTION= VERTICAL",
                "BSPLINE_SYMMETRY_COUPLING= NONE",
                "BSPLINE_KNOT_SCORE_MODE= IKKT_VIRTUAL_INSERTION",
                "BSPLINE_IKKT_INCLUDE_GEOMETRY_CONSTRAINTS= NO",
                "BSPLINE_IKKT_INCLUDE_AERO_CONSTRAINTS= YES",
                "BSPLINE_IKKT_REQUIRE_AVAILABLE_FIELDS= YES",
                "BSPLINE_IKKT_SCALING_MODE= PHYSICAL",
                "OPT_CONSTRAINT= (MOMENT_Z<0.093)*0.001",
                "OBJECTIVE_FUNCTION= DRAG",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "tools/probe_bspline_ikkt_score.py",
            "--config",
            str(config_file),
            "--eval-dir",
            str(eval_dir),
            "--constraint-values-json",
            str(values_file),
            "--output",
            str(output_file),
        ],
        check=True,
    )

    diagnostics = json.loads(output_file.with_suffix(".diagnostics.json").read_text())
    constraint = diagnostics["included_constraints"][0]
    assert constraint["name"] == "AERO[MOMENT_Z]"
    assert constraint["field_file"].endswith("adjoint_momentz/surface_sens.csv")
    assert constraint["c_value"] == pytest.approx(-0.001)
    assert constraint["field_sign"] == pytest.approx(-1.0)
    rows = list(csv.DictReader(open(output_file)))
    assert rows
    assert "score_objective" in rows[0]
    assert "score_ikkt" in rows[0]


@pytest.mark.parametrize(
    "constraint_text, expected_c_value, expected_field_sign, expected_bounds",
    [
        ("(AIRFOIL_AREA>0.11)*0.001", -0.01, 1.0, [0.0, "inf"]),
        ("(AIRFOIL_AREA<0.09)*0.001", -0.01, -1.0, [0.0, "inf"]),
        ("(AIRFOIL_AREA=0.11)*0.001", -0.01, 1.0, ["-inf", "inf"]),
    ],
)
def test_ikkt_native_airfoil_area_constraint_is_analytic(
    tmp_path,
    constraint_text,
    expected_c_value,
    expected_field_sign,
    expected_bounds,
):
    spec = generate_initial_bspline_modes(
        tmp_path / "modes.json",
        "AIRFOIL",
        nper_side=4,
        surface_mode="BOTH",
    )
    metadata = _closed_airfoil_metadata()
    objective = np.asarray([0.0, 0.5, 0.0, -0.5], dtype=float)

    _signal, diagnostics = build_ikkt_score_signal(
        spec,
        metadata,
        objective,
        {
            "surface_mode": "BOTH",
            "symmetry_coupling": "NONE",
            "deformation_direction_mode": "VERTICAL",
            "knot_score_mode": "IKKT_VIRTUAL_INSERTION",
            "ikkt_include_geometry_constraints": True,
            "ikkt_include_aero_constraints": False,
            "ikkt_require_available_fields": True,
            "ikkt_scaling_mode": "PHYSICAL",
            "native_constraints": constraint_text,
        },
    )

    constraint = diagnostics["included_constraints"][0]
    assert constraint["name"] == "GEOMETRY[AIRFOIL_AREA]"
    assert constraint["source"] == "GEOMETRY"
    assert constraint["function_name"] == "AIRFOIL_AREA"
    assert constraint["current_value"] == pytest.approx(0.1)
    assert constraint["c_value"] == pytest.approx(expected_c_value)
    assert constraint["field_sign"] == pytest.approx(expected_field_sign)
    assert constraint["lambda_bounds"] == expected_bounds
    assert constraint["provider"] == "analytic_airfoil_area"
    assert constraint["config_scale"] == pytest.approx(0.001)
    assert constraint["used_in_ikkt_gradient"] is False
    assert constraint["gradient_norm"] > 0.0


def test_ikkt_native_constraints_are_classified_when_disabled_or_unsupported(tmp_path):
    spec = generate_initial_bspline_modes(
        tmp_path / "modes.json",
        "AIRFOIL",
        nper_side=4,
        surface_mode="BOTH",
    )
    metadata = _closed_airfoil_metadata()
    settings = {
        "surface_mode": "BOTH",
        "symmetry_coupling": "NONE",
        "deformation_direction_mode": "VERTICAL",
        "knot_score_mode": "IKKT_VIRTUAL_INSERTION",
        "ikkt_include_geometry_constraints": True,
        "ikkt_include_aero_constraints": False,
        "ikkt_require_available_fields": False,
        "ikkt_scaling_mode": "PHYSICAL",
        "native_constraints": "(AIRFOIL_THICKNESS>0.1); (LIFT>0.5)",
    }

    _signal, diagnostics = build_ikkt_score_signal(
        spec,
        metadata,
        np.ones(len(metadata)),
        settings,
    )

    assert diagnostics["unsupported_constraints"][0]["name"] == "GEOMETRY[AIRFOIL_THICKNESS]"
    assert diagnostics["unsupported_constraints"][0]["reason"] == "unsupported_geometry_constraint"
    assert diagnostics["skipped_constraints"][0]["name"] == "AERO[LIFT]"
    assert diagnostics["skipped_constraints"][0]["reason"] == "aero_constraints_disabled"

    with pytest.raises(
        Exception,
        match="GEOMETRY\\[AIRFOIL_THICKNESS\\]:unsupported_geometry_constraint",
    ):
        build_ikkt_score_signal(
            spec,
            metadata,
            np.ones(len(metadata)),
            {**settings, "ikkt_require_available_fields": True},
        )
