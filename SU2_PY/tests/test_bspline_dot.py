import csv
import json

import pytest

from SU2.opt.bspline_dot import (
    BSplineDotError,
    match_sensitivities_to_metadata,
    project_bspline_gradients,
    read_metadata,
    read_sensitivity_file,
    write_gradients_csv,
    write_summary_json,
)
from SU2.opt.bspline_modes import evaluate_all_modes


def _base_spec(include_inactive=False):
    modes = [
        {
            "id": "upper_B1",
            "side": "upper",
            "basis_type": "clamped",
            "degree": 3,
            "knot_vector": [0, 0, 0, 0, 1, 1, 1, 1],
            "basis_index": 1,
            "coefficient": 0.0,
            "bounds": [-0.01, 0.01],
        },
        {
            "id": "lower_B1",
            "side": "lower",
            "basis_type": "clamped",
            "degree": 3,
            "knot_vector": [0, 0, 0, 0, 1, 1, 1, 1],
            "basis_index": 1,
            "coefficient": 0.0,
            "bounds": [-0.01, 0.01],
        },
    ]
    if include_inactive:
        modes.append(
            {
                "id": "inactive_upper",
                "side": "upper",
                "basis_type": "clamped",
                "degree": 3,
                "knot_vector": [0, 0, 0, 0, 1, 1, 1, 1],
                "basis_index": 2,
                "coefficient": 0.0,
                "active": False,
            }
        )
    return {
        "version": 1,
        "dimension": 2,
        "marker": "airfoil",
        "chord": {"mode": "auto", "x_le": None, "x_te": None},
        "normal_displacement": True,
        "class_shape": "none",
        "normalize_basis": False,
        "normalization_mode": "max",
        "modes": modes,
    }


def _metadata_records():
    points = [
        (10, 0.0, 0.0, 0.0, "upper", 0.5),
        (11, 0.25, 0.04, 0.25, "upper", 1.0),
        (12, 0.50, 0.06, 0.50, "upper", 1.0),
        (13, 0.25, -0.04, 0.25, "lower", 1.0),
        (14, 0.50, -0.06, 0.50, "lower", 1.0),
        (15, 1.0, 0.0, 1.0, "lower", 0.5),
    ]
    return [
        {
            "node_id": node_id,
            "x": x,
            "y": y,
            "x_over_c": x_over_c,
            "side": side,
            "normal_x": 0.0,
            "normal_y": 1.0,
            "weight": weight,
            "deformed_x": x,
            "deformed_y": y,
        }
        for node_id, x, y, x_over_c, side, weight in points
    ]


def _write_metadata(path, records):
    fieldnames = [
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
    ]
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)
    return path


def _expected_gradient(spec, metadata, mode_id, scale=1.0, weighted=False):
    values = evaluate_all_modes(
        spec,
        [record["x_over_c"] for record in metadata],
        sides=[record["side"] for record in metadata],
    )[mode_id]
    if weighted:
        return scale * sum(
            phi * record["weight"]
            for phi, record in zip(values, metadata)
        )
    return scale * sum(values)


def _by_mode(gradients):
    return {record["mode_id"]: record for record in gradients}


def test_vector_sensitivity_projection_matches_discrete_mode_sum(tmp_path):
    metadata_file = _write_metadata(tmp_path / "metadata.csv", _metadata_records())
    sens_file = tmp_path / "sens.csv"
    sens_file.write_text(
        "node_id,Sensitivity_x,Sensitivity_y\n"
        + "".join(
            f"{record['node_id']},0.0,1.0\n"
            for record in _metadata_records()
        )
    )

    spec = _base_spec()
    metadata = read_metadata(str(metadata_file))
    sensitivities = read_sensitivity_file(str(sens_file))
    gradients = project_bspline_gradients(spec, metadata, sensitivities)

    by_mode = _by_mode(gradients)
    assert by_mode["upper_B1"]["gradient"] == pytest.approx(
        _expected_gradient(spec, metadata, "upper_B1")
    )
    assert by_mode["lower_B1"]["gradient"] == pytest.approx(
        _expected_gradient(spec, metadata, "lower_B1")
    )
    assert by_mode["upper_B1"]["projection_mode"] == "vector"


def test_lower_side_masking_keeps_upper_and_lower_modes_separate():
    spec = _base_spec()
    upper_metadata = [
        dict(record, side="upper")
        for record in _metadata_records()
        if record["side"] == "upper"
    ]
    lower_metadata = [
        dict(record, side="lower")
        for record in _metadata_records()
        if record["side"] == "lower"
    ]
    upper_sensitivities = [
        {"node_id": record["node_id"], "sensitivity_x": 0.0, "sensitivity_y": 1.0}
        for record in upper_metadata
    ]
    lower_sensitivities = [
        {"node_id": record["node_id"], "sensitivity_x": 0.0, "sensitivity_y": 1.0}
        for record in lower_metadata
    ]

    upper_gradients = _by_mode(
        project_bspline_gradients(spec, upper_metadata, upper_sensitivities)
    )
    lower_gradients = _by_mode(
        project_bspline_gradients(spec, lower_metadata, lower_sensitivities)
    )

    assert upper_gradients["upper_B1"]["gradient"] == pytest.approx(
        _expected_gradient(spec, upper_metadata, "upper_B1")
    )
    assert upper_gradients["lower_B1"]["gradient"] == pytest.approx(0.0)
    assert lower_gradients["upper_B1"]["gradient"] == pytest.approx(0.0)
    assert lower_gradients["lower_B1"]["gradient"] == pytest.approx(
        _expected_gradient(spec, lower_metadata, "lower_B1")
    )


def test_scalar_sensitivity_fallback_is_used_when_vector_columns_are_absent(tmp_path):
    metadata = _metadata_records()
    sens_file = tmp_path / "scalar_sens.csv"
    sens_file.write_text(
        "node_id,Surface_Sensitivity\n"
        + "".join(f"{record['node_id']},2.0\n" for record in metadata)
    )

    spec = _base_spec()
    sensitivities = read_sensitivity_file(str(sens_file))
    gradients = _by_mode(project_bspline_gradients(spec, metadata, sensitivities))

    assert gradients["upper_B1"]["projection_mode"] == "scalar"
    assert gradients["upper_B1"]["gradient"] == pytest.approx(
        _expected_gradient(spec, metadata, "upper_B1", scale=2.0, weighted=False)
    )


def test_sensitivity_weighting_controls_vector_and_scalar_weights():
    spec = {
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
                "id": "constant",
                "side": "upper",
                "basis_type": "clamped",
                "degree": 3,
                "knot_vector": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                "basis_index": 0,
                "coefficient": 0.0,
            }
        ],
    }
    metadata = [
        {
            "node_id": 1,
            "x": 0.5,
            "y": 0.0,
            "x_over_c": 0.5,
            "side": "upper",
            "normal_x": 0.0,
            "normal_y": 1.0,
            "weight": 10.0,
            "deformed_x": 0.5,
            "deformed_y": 0.0,
        }
    ]
    vector = [{"node_id": 1, "sensitivity_x": 0.0, "sensitivity_y": 1.0}]
    scalar = [{"node_id": 1, "surface_sensitivity": 1.0}]

    vector_nodal = project_bspline_gradients(
        spec,
        metadata,
        vector,
        sensitivity_weighting="NODAL",
    )[0]
    vector_density = project_bspline_gradients(
        spec,
        metadata,
        vector,
        sensitivity_weighting="DENSITY",
    )[0]
    scalar_nodal = project_bspline_gradients(
        spec,
        metadata,
        scalar,
        sensitivity_weighting="NODAL",
    )[0]
    scalar_density = project_bspline_gradients(
        spec,
        metadata,
        scalar,
        sensitivity_weighting="DENSITY",
    )[0]

    # basis index 0 of clamped cubic on [0,0,0,0,1,1,1,1] evaluates to 0.125 at x=0.5,
    # so the projected gradient equals 0.125 (nodal) or 10*0.125 (density).
    assert vector_nodal["gradient"] == pytest.approx(0.125)
    assert vector_density["gradient"] == pytest.approx(1.25)
    assert scalar_nodal["gradient"] == pytest.approx(0.125)
    assert scalar_density["gradient"] == pytest.approx(1.25)
    assert vector_nodal["sensitivity_weighting"] == "NODAL"
    assert vector_density["sensitivity_weighting"] == "DENSITY"


def test_projection_uses_deform_dir_columns_when_present():
    # When the metadata carries deform_dir_x/deform_dir_y (LE-safe direction),
    # the vector projection must use those, not the raw normal_x/normal_y.
    spec = {
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
                "id": "constant",
                "side": "upper",
                "basis_type": "clamped",
                "degree": 3,
                "knot_vector": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                "basis_index": 0,
                "coefficient": 0.0,
            }
        ],
    }
    # normal points purely +x, but the LE-safe deform direction is purely +y.
    metadata = [
        {
            "node_id": 1,
            "x": 0.5,
            "y": 0.0,
            "x_over_c": 0.5,
            "side": "upper",
            "normal_x": 1.0,
            "normal_y": 0.0,
            "deform_dir_x": 0.0,
            "deform_dir_y": 1.0,
            "weight": 1.0,
            "deformed_x": 0.5,
            "deformed_y": 0.0,
        }
    ]
    # sensitivity points purely +y, so projecting onto the normal (+x) gives 0,
    # projecting onto the deform direction (+y) gives the basis value 0.125.
    sensitivity = [{"node_id": 1, "sensitivity_x": 0.0, "sensitivity_y": 1.0}]
    gradient = project_bspline_gradients(spec, metadata, sensitivity)[0]
    assert gradient["gradient"] == pytest.approx(0.125)


def test_vertical_vector_projection_uses_only_signed_sensitivity_y():
    spec = _base_spec()
    metadata = [
        {
            "node_id": 1,
            "x": 0.5,
            "y": 0.05,
            "x_over_c": 0.5,
            "side": "upper",
            "normal_x": 0.8,
            "normal_y": 0.6,
            "deform_dir_x": 0.0,
            "deform_dir_y": 1.0,
            "weight": 1.0,
            "deformed_x": 0.5,
            "deformed_y": 0.05,
        },
        {
            "node_id": 2,
            "x": 0.5,
            "y": -0.05,
            "x_over_c": 0.5,
            "side": "lower",
            "normal_x": -0.8,
            "normal_y": -0.6,
            "deform_dir_x": 0.0,
            "deform_dir_y": -1.0,
            "weight": 1.0,
            "deformed_x": 0.5,
            "deformed_y": -0.05,
        },
    ]
    sensitivities = [
        {"node_id": 1, "sensitivity_x": 1000.0, "sensitivity_y": 2.0},
        {"node_id": 2, "sensitivity_x": -2000.0, "sensitivity_y": 3.0},
    ]

    gradients = _by_mode(project_bspline_gradients(spec, metadata, sensitivities))

    assert gradients["upper_B1"]["gradient"] == pytest.approx(0.75)
    assert gradients["lower_B1"]["gradient"] == pytest.approx(-1.125)


@pytest.mark.parametrize(
    "side,direction_y,sensitivity_y,expected",
    [
        ("upper", 1.0, 2.0, 0.75),
        ("lower", -1.0, 2.0, -0.75),
    ],
)
def test_vertical_projection_supports_single_surface_metadata(
    side,
    direction_y,
    sensitivity_y,
    expected,
):
    spec = _base_spec()
    spec["modes"] = [mode for mode in spec["modes"] if mode["side"] == side]
    spec["surface_mode"] = side.upper()
    metadata = [
        {
            "node_id": 1,
            "x": 0.5,
            "y": 0.05 if side == "upper" else -0.05,
            "x_over_c": 0.5,
            "side": side,
            "normal_x": 0.0,
            "normal_y": direction_y,
            "deform_dir_x": 0.0,
            "deform_dir_y": direction_y,
            "weight": 1.0,
            "deformed_x": 0.5,
            "deformed_y": 0.05 if side == "upper" else -0.05,
        }
    ]
    sensitivities = [
        {"node_id": 1, "sensitivity_x": 1000.0, "sensitivity_y": sensitivity_y}
    ]

    gradient = project_bspline_gradients(spec, metadata, sensitivities)[0]

    assert gradient["side"] == side
    assert gradient["gradient"] == pytest.approx(expected)


def test_scalar_normal_sensitivity_is_converted_to_effective_direction():
    spec = _base_spec()
    metadata = [
        {
            "node_id": 1,
            "x": 0.5,
            "y": 0.05,
            "x_over_c": 0.5,
            "side": "upper",
            "normal_x": 0.6,
            "normal_y": 0.8,
            "deform_dir_x": 0.0,
            "deform_dir_y": 1.0,
            "weight": 1.0,
            "deformed_x": 0.5,
            "deformed_y": 0.05,
        }
    ]
    sensitivities = [{"node_id": 1, "surface_sensitivity": 2.0}]

    gradient = project_bspline_gradients(
        spec,
        metadata,
        sensitivities,
        prefer_vector=False,
    )[0]

    assert gradient["projection_mode"] == "scalar"
    assert gradient["gradient"] == pytest.approx(0.6)


def test_read_metadata_defaults_deform_dir_to_normal_when_absent(tmp_path):
    # Backward compatibility: metadata without deform_dir_* falls back to normals.
    records = _metadata_records()
    metadata_file = _write_metadata(tmp_path / "metadata.csv", records)
    parsed = read_metadata(str(metadata_file))
    for row in parsed:
        assert row["deform_dir_x"] == pytest.approx(row["normal_x"])
        assert row["deform_dir_y"] == pytest.approx(row["normal_y"])


def test_matching_by_node_id_supports_shuffled_whitespace_rows(tmp_path):
    metadata = _metadata_records()
    sens_file = tmp_path / "sens.dat"
    rows = [
        f"{record['node_id']} 0.0 1.0\n"
        for record in reversed(metadata)
    ]
    sens_file.write_text("Point Sens_X Sens_Y\n" + "".join(rows))

    spec = _base_spec()
    sensitivities = read_sensitivity_file(str(sens_file))
    matched = match_sensitivities_to_metadata(metadata, sensitivities)
    gradients = _by_mode(project_bspline_gradients(spec, metadata, matched))

    assert [record["node_id"] for record in matched] == [
        record["node_id"] for record in metadata
    ]
    assert gradients["upper_B1"]["gradient"] == pytest.approx(
        _expected_gradient(spec, metadata, "upper_B1")
    )


def test_matching_by_node_id_supports_su2_point_index_header(tmp_path):
    metadata = _metadata_records()
    sens_file = tmp_path / "surface_adjoint.csv"
    rows = [
        f"{record['node_id']},0.0,{index + 1}.0\n"
        for index, record in enumerate(reversed(metadata))
    ]
    sens_file.write_text("Point_Index,Sensitivity_x,Sensitivity_y\n" + "".join(rows))

    sensitivities = read_sensitivity_file(str(sens_file))
    matched = match_sensitivities_to_metadata(metadata, sensitivities)

    assert [record["node_id"] for record in matched] == [
        record["node_id"] for record in metadata
    ]


def test_row_order_matching_is_allowed_without_node_ids(tmp_path):
    metadata = _metadata_records()
    sens_file = tmp_path / "sens.csv"
    sens_file.write_text(
        "Sensitivity_x,Sensitivity_y\n"
        + "".join("0.0,1.0\n" for _ in metadata)
    )

    sensitivities = read_sensitivity_file(str(sens_file))
    with pytest.warns(RuntimeWarning, match="positional sensitivity/metadata matching"):
        matched = match_sensitivities_to_metadata(metadata, sensitivities)

    assert len(matched) == len(metadata)
    assert all(record["sensitivity_y"] == pytest.approx(1.0) for record in matched)


def test_matching_error_is_clear_when_row_order_is_impossible():
    metadata = _metadata_records()
    sensitivities = [
        {"sensitivity_x": 0.0, "sensitivity_y": 1.0}
        for _ in metadata[:-1]
    ]

    with pytest.raises(BSplineDotError, match="row counts differ"):
        match_sensitivities_to_metadata(metadata, sensitivities)


def test_zero_sensitivity_gives_zero_gradient():
    metadata = _metadata_records()
    sensitivities = [
        {"node_id": record["node_id"], "sensitivity_x": 0.0, "sensitivity_y": 0.0}
        for record in metadata
    ]

    gradients = project_bspline_gradients(_base_spec(), metadata, sensitivities)

    assert all(record["gradient"] == pytest.approx(0.0) for record in gradients)


def test_inactive_modes_are_skipped():
    metadata = _metadata_records()
    sensitivities = [
        {"node_id": record["node_id"], "sensitivity_x": 0.0, "sensitivity_y": 1.0}
        for record in metadata
    ]

    gradients = project_bspline_gradients(
        _base_spec(include_inactive=True),
        metadata,
        sensitivities,
    )

    assert "inactive_upper" not in _by_mode(gradients)


def test_gradient_csv_and_summary_are_written(tmp_path):
    metadata = _metadata_records()
    sensitivities = [
        {"node_id": record["node_id"], "sensitivity_x": 0.0, "sensitivity_y": 1.0}
        for record in metadata
    ]
    gradients = project_bspline_gradients(_base_spec(), metadata, sensitivities)
    output_file = tmp_path / "bspline_gradients.csv"
    summary_file = tmp_path / "summary.json"

    write_gradients_csv(gradients, str(output_file))
    write_summary_json(metadata, sensitivities, gradients, str(summary_file))

    rows = list(csv.DictReader(output_file.open()))
    assert rows[0]["mode_id"] == "upper_B1"
    assert rows[0]["projection_mode"] == "vector"
    assert rows[0]["sensitivity_weighting"] == "NODAL"
    summary = json.loads(summary_file.read_text())
    assert summary["number_of_metadata_nodes"] == 6
    assert summary["sensitivity_weighting"] == "NODAL"
    assert summary["modes"][0]["sensitivity_weighting"] == "NODAL"
