import pytest

from tools.compare_progressive_ffd_blending import (
    _comparison_signature,
    _correlation_columns,
    _correlation_metrics,
    _parse_order,
    _prepared_mesh_path,
    _validate_zero_dv_manifest,
    _write_heatmap_svg,
    response_metrics,
)


def test_response_metrics_capture_support_and_isolation():
    records = [
        {
            "surface_side": "UPPER",
            "edge": index in (0, 4),
            "le": index == 0,
            "te": index == 4,
            "x_over_c": 0.25 * index,
            "dy": value,
            "displacement": abs(value),
        }
        for index, value in enumerate((0.0, 0.1, 1.0, 0.1, 0.0))
    ]
    records.append(
        {
            "surface_side": "LOWER",
            "edge": False,
            "le": False,
            "te": False,
            "x_over_c": 0.5,
            "dy": 1.0e-14,
            "displacement": 1.0e-14,
        }
    )
    metrics = response_metrics(records, "UPPER")
    assert metrics["target_max_displacement"] == pytest.approx(1.0)
    assert metrics["x_over_c_at_peak"] == pytest.approx(0.5)
    assert metrics["support_width_50pct"] == pytest.approx(0.0)
    assert metrics["support_width_10pct"] == pytest.approx(0.5)
    assert metrics["opposite_max_displacement"] == pytest.approx(1.0e-14)
    assert metrics["le_max_displacement"] == pytest.approx(0.0)
    assert metrics["te_max_displacement"] == pytest.approx(0.0)


def test_correlation_and_svg_are_written_without_plot_dependencies(tmp_path):
    matrix = [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]
    correlation = _correlation_columns(matrix)
    assert correlation[0][0] == pytest.approx(1.0)
    assert correlation[1][1] == pytest.approx(1.0)
    metrics = _correlation_metrics(correlation)
    assert metrics[0]["most_correlated_dv_index"] == 1
    assert metrics[0]["max_abs_correlation_other_dv"] == pytest.approx(1.0)
    svg = tmp_path / "heatmap.svg"
    _write_heatmap_svg(svg, matrix, "test")
    assert "<svg" in svg.read_text()


def test_order_parser_requires_three_positive_orders():
    assert _parse_order("4,2,2") == (4, 2, 2)
    with pytest.raises(Exception):
        _parse_order("4,2")
    with pytest.raises(Exception):
        _parse_order("4,1,2")


def test_comparison_helpers_enforce_names_signatures_and_zero_dv(tmp_path):
    assert _prepared_mesh_path(tmp_path, "BEZIER", (2, 2, 2)).endswith(
        "rae2822_dual_bezier.su2"
    )
    assert _prepared_mesh_path(
        tmp_path,
        "BSPLINE_UNIFORM",
        (5, 2, 2),
    ).endswith("rae2822_dual_bspline_o5.su2")

    request = {
        "raw_mesh": "/tmp/raw.su2",
        "raw_mesh_sha256": "abc",
        "marker": "AIRFOIL",
        "initial_columns": [0.25, 0.5, 0.75],
        "bootstrap_tag": "BOOTSTRAP",
        "bootstrap_y_padding_chord": 0.04,
        "upper_tag": "UPPER_BOX",
        "lower_tag": "LOWER_BOX",
        "upper_offset_chord": 0.05,
        "lower_offset_chord": 0.05,
    }
    manifest = {
        "request": request,
        "result": {"chord": 1.0, "smoke_coordinate_error": 1.0e-14},
    }
    assert _comparison_signature(manifest) == request
    check = _validate_zero_dv_manifest(manifest, "BEZIER")
    assert check["tolerance"] == pytest.approx(1.0e-10)

    manifest["result"]["smoke_coordinate_error"] = float("nan")
    with pytest.raises(RuntimeError, match="finite"):
        _validate_zero_dv_manifest(manifest, "BEZIER")


def test_correlation_rejects_nonfinite_response_values():
    with pytest.raises(RuntimeError, match="finite"):
        _correlation_columns([[1.0, float("nan")], [0.0, 1.0]])
