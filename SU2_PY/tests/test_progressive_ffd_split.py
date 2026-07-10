import csv
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from SU2.opt.bspline_def import read_su2_mesh
from SU2.opt.progressive_ffd_mesh import (
    _bezier_1d,
    _find_key_line,
    _find_tagged_ffd_block,
    _invert_monotone_bezier,
    _line_key,
    _parse_control_points,
    _parse_count_block,
    _parse_degree,
    _parse_blending_spec,
    _parse_int_value,
    _split_tokens,
)
from SU2.opt.progressive_ffd_blending import evaluate_curve
from SU2.opt.progressive_ffd_split import (
    FFDBoxSplitError,
    rewrite_dual_ffd_boxes_with_columns_and_reembed,
    split_bootstrap_ffd_box,
)


SYMMETRIC_POINTS = [
    (1.0, 0.0),
    (0.75, 0.06),
    (0.5, 0.08),
    (0.25, 0.05),
    (0.0, 0.0),
    (0.25, -0.05),
    (0.5, -0.08),
    (0.75, -0.06),
]

CAMBERED_POINTS = [
    (1.0, 0.0),
    (0.75, 0.09),
    (0.5, 0.13),
    (0.25, 0.11),
    (0.0, 0.0),
    (0.25, 0.02),
    (0.5, 0.015),
    (0.75, 0.005),
]

DEFAULT_COLUMNS = [-0.1, 0.25, 0.5, 0.75, 1.1]
OUTER_POINTS = [
    (2.0, 0.0),
    (1.5, 1.5),
    (0.5, 2.0),
    (-1.0, 1.5),
    (-2.0, 0.0),
    (-1.0, -1.5),
    (0.5, -2.0),
    (1.5, -1.5),
]


def _write_bootstrap_mesh(
    path,
    points=SYMMETRIC_POINTS,
    columns=DEFAULT_COLUMNS,
    *,
    blending="BEZIER",
    surface_ids=None,
    disconnected_marker=False,
):
    points = list(points)
    columns = list(columns)
    if len(points) != len(OUTER_POINTS):
        raise ValueError("Synthetic annulus expects eight airfoil points")
    if surface_ids is None:
        surface_ids = list(range(len(points)))

    all_points = points + OUTER_POINTS
    lines = ["NDIME= 2\n", f"NELEM= {len(points)}\n"]
    for index in range(len(points)):
        next_index = (index + 1) % len(points)
        lines.append(
            "9 "
            f"{index} {len(points) + index} "
            f"{len(points) + next_index} {next_index} {index}\n"
        )

    lines.append(f"NPOIN= {len(all_points)}\n")
    for point_id, (x, y) in enumerate(all_points):
        lines.append(f"{x:.16g} {y:.16g} {point_id}\n")

    marker_segments = [
        (index, (index + 1) % len(points)) for index in range(len(points))
    ]
    if disconnected_marker:
        marker_segments[3] = marker_segments[0]
    lines.extend(
        [
            "NMARK= 2\n",
            "MARKER_TAG= AIRFOIL\n",
            f"MARKER_ELEMS= {len(marker_segments)}\n",
        ]
    )
    for start, end in marker_segments:
        lines.append(f"3 {start} {end}\n")
    lines.extend(
        [
            "MARKER_TAG= FARFIELD\n",
            f"MARKER_ELEMS= {len(OUTER_POINTS)}\n",
        ]
    )
    for index in range(len(OUTER_POINTS)):
        next_index = (index + 1) % len(OUTER_POINTS)
        lines.append(
            f"3 {len(points) + index} {len(points) + next_index}\n"
        )

    y_rows = [-0.3, 0.3]
    z_planes = [-0.5, 0.5]
    lines.extend(
        [
            "FFD_NBOX= 1\n",
            "FFD_NLEVEL= 1\n",
            "FFD_TAG= BOOTSTRAP_BOX\n",
            "FFD_LEVEL= 0\n",
            f"FFD_DEGREE_I= {len(columns) - 1}\n",
            "FFD_DEGREE_J= 1\n",
            f"FFD_BLENDING= {blending}\n",
            "FFD_PARENTS= 0\n",
            "FFD_CHILDREN= 0\n",
            "FFD_CORNER_POINTS= 4\n",
            f"{columns[0]:.16g} {y_rows[0]:.16g}\n",
            f"{columns[-1]:.16g} {y_rows[0]:.16g}\n",
            f"{columns[0]:.16g} {y_rows[1]:.16g}\n",
            f"{columns[-1]:.16g} {y_rows[1]:.16g}\n",
            f"FFD_CONTROL_POINTS= {len(columns) * 4}\n",
        ]
    )
    for i, x in enumerate(columns):
        for j, y in enumerate(y_rows):
            for k, z in enumerate(z_planes):
                lines.append(f"{i} {j} {k} {x:.16g} {y:.16g} {z:.16g}\n")

    lines.append(f"FFD_SURFACE_POINTS= {len(surface_ids)}\n")
    for point_id in surface_ids:
        x, y = points[point_id]
        u = _invert_monotone_bezier(columns, x)
        v = (y - y_rows[0]) / (y_rows[1] - y_rows[0])
        lines.append(f"AIRFOIL {point_id} {u:.16g} {v:.16g} 0.5\n")
    lines.append("% NON_FFD_SENTINEL: preserve this trailing mesh content\n")

    path.write_text("".join(lines))
    return path


def _read_box(path, tag):
    lines = path.read_text().splitlines(keepends=True)
    start, end = _find_tagged_ffd_block(lines, tag)
    degree = _parse_degree(lines, start, end)
    corner_block = _parse_count_block(lines, start, end, "FFD_CORNER_POINTS")
    control_block = _parse_count_block(lines, start, end, "FFD_CONTROL_POINTS")
    surface_block = _parse_count_block(lines, start, end, "FFD_SURFACE_POINTS")
    controls, coord_dim, control_format = _parse_control_points(control_block)
    control_by_index = {
        (point["i"], point["j"], point["k"]): point["coords"]
        for point in controls
    }
    surface_rows = []
    for line in surface_block["data"]:
        tokens = _split_tokens(line)
        surface_rows.append(
            {
                "marker": tokens[0],
                "point_id": int(tokens[1]),
                "u": float(tokens[2]),
                "v": float(tokens[3]),
                "w": float(tokens[4]),
            }
        )
    return {
        "lines": lines,
        "degree": degree,
        "corners": corner_block,
        "controls": controls,
        "control_by_index": control_by_index,
        "coord_dim": coord_dim,
        "control_format": control_format,
        "surface": surface_rows,
        "surface_count": surface_block["count"],
        "blending_spec": _parse_blending_spec(
            lines,
            start,
            end,
            control_counts=(degree["i"] + 1, degree["j"] + 1, 2),
            dual_2d=True,
        ),
    }


def _split(mesh_in, mesh_out, **overrides):
    options = {
        "bootstrap_tag": "BOOTSTRAP_BOX",
        "marker": "AIRFOIL",
        "upper_tag": "UPPER_BOX",
        "lower_tag": "LOWER_BOX",
        "upper_offset_chord": 0.04,
        "lower_offset_chord": 0.06,
    }
    options.update(overrides)
    return split_bootstrap_ffd_box(mesh_in, mesh_out, **options)


def _assert_surface_reconstruction(box, mesh_points, tol=1.0e-10):
    degree_i = box["degree"]["i"]
    columns = [box["control_by_index"][(i, 0, 0)][0] for i in range(degree_i + 1)]
    row0 = [box["control_by_index"][(i, 0, 0)][1] for i in range(degree_i + 1)]
    row1 = [box["control_by_index"][(i, 1, 0)][1] for i in range(degree_i + 1)]
    for record in box["surface"]:
        u = record["u"]
        v = record["v"]
        spec = box["blending_spec"]
        x = evaluate_curve(columns, u, spec, axis=0)
        y0 = evaluate_curve(row0, u, spec, axis=0)
        y1 = evaluate_curve(row1, u, spec, axis=0)
        y = evaluate_curve([y0, y1], v, spec, axis=1)
        expected = mesh_points[record["point_id"]]
        assert math.hypot(x - expected[0], y - expected[1]) <= tol


def test_symmetric_split_builds_disjoint_boxes_and_reembeds_exactly(tmp_path):
    mesh_in = _write_bootstrap_mesh(tmp_path / "bootstrap.su2")
    original_text = mesh_in.read_text()
    mesh_out = tmp_path / "dual.su2"

    summary = _split(mesh_in, mesh_out)

    assert mesh_in.read_text() == original_text
    assert summary["upper_surface_points"] == 3
    assert summary["lower_surface_points"] == 3
    assert summary["fixed_edge_points"] == 2
    assert summary["columns"] == DEFAULT_COLUMNS

    output_text = mesh_out.read_text()
    assert "FFD_NBOX= 2\n" in output_text


    assert "FFD_TAG= UPPER_BOX\n" in output_text
    assert "FFD_TAG= LOWER_BOX\n" in output_text
    assert "% NON_FFD_SENTINEL: preserve this trailing mesh content\n" in output_text
    assert output_text.count("FFD_DEGREE_K") == 0

    upper = _read_box(mesh_out, "UPPER_BOX")
    lower = _read_box(mesh_out, "LOWER_BOX")
    assert upper["degree"]["i"] == 4
    assert upper["degree"]["j"] == 1
    assert upper["corners"]["count"] == 4
    assert len(upper["controls"]) == 20
    assert len(lower["controls"]) == 20
    assert upper["control_format"] == "INDEXED_3D"
    assert {point["point_id"] for point in upper["surface"]} == {1, 2, 3}
    assert {point["point_id"] for point in lower["surface"]} == {5, 6, 7}
    assert not (
        {point["point_id"] for point in upper["surface"]}
        & {point["point_id"] for point in lower["surface"]}
    )

    for k in (0, 1):
        assert upper["control_by_index"][(2, 0, k)][:2] == pytest.approx([0.5, 0.0])
        assert upper["control_by_index"][(2, 1, k)][:2] == pytest.approx([0.5, 0.12])
        assert lower["control_by_index"][(2, 0, k)][:2] == pytest.approx([0.5, -0.14])
        assert lower["control_by_index"][(2, 1, k)][:2] == pytest.approx([0.5, 0.0])
        assert upper["control_by_index"][(0, 1, k)][1] == pytest.approx(0.04)
        assert upper["control_by_index"][(4, 1, k)][1] == pytest.approx(0.04)
        assert lower["control_by_index"][(0, 0, k)][1] == pytest.approx(-0.06)
        assert lower["control_by_index"][(4, 0, k)][1] == pytest.approx(-0.06)

    mesh = read_su2_mesh(mesh_out)
    _assert_surface_reconstruction(upper, mesh["points"])
    _assert_surface_reconstruction(lower, mesh["points"])

    diagnostics_path = tmp_path / "dual_dual_ffd_diagnostics.csv"
    assert str(diagnostics_path) == summary["diagnostics_csv"]
    with diagnostics_path.open(newline="") as fp:
        diagnostics = list(csv.DictReader(fp))
    assert len(diagnostics) == len(SYMMETRIC_POINTS)
    assert set(diagnostics[0]) == {
        "point_id",
        "side",
        "box",
        "x",
        "y",
        "u",
        "v",
        "w",
        "reconstruction_error",
        "status",
    }
    fixed_ids = {
        int(row["point_id"]) for row in diagnostics if row["status"] == "FIXED_EDGE"
    }
    assert fixed_ids == {0, 4}
    assert all(
        float(row["reconstruction_error"]) <= 1.0e-10
        for row in diagnostics
        if row["status"] == "EMBEDDED"
    )


def test_bspline_split_and_rewrite_preserve_surface_and_fixed_order(tmp_path):
    bootstrap = _write_bootstrap_mesh(tmp_path / "bootstrap.su2")
    dual = tmp_path / "dual_bspline.su2"
    summary = _split(
        bootstrap,
        dual,
        output_blending="BSPLINE_UNIFORM",
        bspline_orders=(4, 2, 2),
    )
    assert summary["blending"] == "BSPLINE_UNIFORM"
    assert summary["bspline_orders"] == [4, 2, 2]
    text = dual.read_text()
    assert text.count("FFD_BLENDING= BSPLINE_UNIFORM") == 2
    assert text.count("BSPLINE_ORDER_I= 4") == 2

    mesh_points = read_su2_mesh(dual)["points"]
    _assert_surface_reconstruction(_read_box(dual, "UPPER_BOX"), mesh_points)
    _assert_surface_reconstruction(_read_box(dual, "LOWER_BOX"), mesh_points)

    rewritten = tmp_path / "rewritten.su2"
    rewrite_summary = rewrite_dual_ffd_boxes_with_columns_and_reembed(
        dual,
        rewritten,
        marker="AIRFOIL",
        upper_tag="UPPER_BOX",
        lower_tag="LOWER_BOX",
        upper_columns=[-0.1, 0.15, 0.35, 0.55, 0.75, 0.9, 1.1],
        lower_columns=[-0.1, 0.2, 0.5, 0.8, 1.1],
        upper_offset_chord=0.04,
        lower_offset_chord=0.06,
        diagnostics_csv=False,
        overwrite=False,
    )
    assert rewrite_summary["blending"] == "BSPLINE_UNIFORM"
    upper = _read_box(rewritten, "UPPER_BOX")
    lower = _read_box(rewritten, "LOWER_BOX")
    assert upper["degree"]["i"] == 6
    assert lower["degree"]["i"] == 4
    assert upper["blending_spec"].orders == (4, 2, 2)
    assert lower["blending_spec"].orders == (4, 2, 2)
    rewritten_points = read_su2_mesh(rewritten)["points"]
    _assert_surface_reconstruction(upper, rewritten_points)
    _assert_surface_reconstruction(lower, rewritten_points)


def test_cambered_lower_branch_above_chord_is_still_classified_topologically(tmp_path):
    mesh_in = _write_bootstrap_mesh(
        tmp_path / "rae_like_bootstrap.su2", points=CAMBERED_POINTS
    )
    mesh_out = tmp_path / "rae_like_dual.su2"

    _split(
        mesh_in,
        mesh_out,
        upper_offset_chord=0.05,
        lower_offset_chord=0.05,
    )

    upper = _read_box(mesh_out, "UPPER_BOX")
    lower = _read_box(mesh_out, "LOWER_BOX")
    assert {point["point_id"] for point in upper["surface"]} == {1, 2, 3}
    assert {point["point_id"] for point in lower["surface"]} == {5, 6, 7}
    assert all(CAMBERED_POINTS[point["point_id"]][1] > 0.0 for point in lower["surface"])

    for k in (0, 1):
        assert upper["control_by_index"][(1, 0, k)][1] == pytest.approx(0.065)
        assert upper["control_by_index"][(1, 1, k)][1] == pytest.approx(0.16)
        assert lower["control_by_index"][(1, 0, k)][1] == pytest.approx(-0.03)
        assert lower["control_by_index"][(1, 1, k)][1] == pytest.approx(0.065)

    mesh = read_su2_mesh(mesh_out)
    _assert_surface_reconstruction(upper, mesh["points"])
    _assert_surface_reconstruction(lower, mesh["points"])


def test_round_trip_preserves_non_ffd_mesh_content_and_external_columns(tmp_path):
    mesh_in = _write_bootstrap_mesh(tmp_path / "bootstrap.su2")
    mesh_out = tmp_path / "dual.su2"
    input_mesh = read_su2_mesh(mesh_in)

    _split(mesh_in, mesh_out, x_le=0.0, x_te=1.0)

    output_mesh = read_su2_mesh(mesh_out)
    assert output_mesh["ndime"] == input_mesh["ndime"]
    assert output_mesh["points"] == input_mesh["points"]
    assert output_mesh["markers"] == input_mesh["markers"]
    upper = _read_box(mesh_out, "UPPER_BOX")
    columns = [
        upper["control_by_index"][(i, 0, 0)][0]
        for i in range(upper["degree"]["i"] + 1)
    ]
    assert columns == DEFAULT_COLUMNS

    lines = mesh_out.read_text().splitlines(keepends=True)
    nbox_line = _find_key_line(lines, 0, len(lines), "FFD_NBOX")
    assert _parse_int_value(lines[nbox_line]) == 2
    assert sum(_line_key(line) == "FFD_TAG" for line in lines) == 2


def test_existing_output_is_not_overwritten_without_explicit_permission(tmp_path):
    mesh_in = _write_bootstrap_mesh(tmp_path / "bootstrap.su2")
    mesh_out = tmp_path / "dual.su2"
    mesh_out.write_text("keep me\n")

    with pytest.raises(FFDBoxSplitError, match="already exists"):
        _split(mesh_in, mesh_out)
    assert mesh_out.read_text() == "keep me\n"

    _split(mesh_in, mesh_out, overwrite=True)
    assert "FFD_NBOX= 2" in mesh_out.read_text()


def test_cli_writes_dual_mesh_and_default_diagnostics(tmp_path):
    mesh_in = _write_bootstrap_mesh(tmp_path / "bootstrap.su2")
    mesh_out = tmp_path / "cli_dual.su2"
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "split_progressive_ffd_box.py"),
            "--mesh-in",
            str(mesh_in),
            "--mesh-out",
            str(mesh_out),
            "--bootstrap-tag",
            "BOOTSTRAP_BOX",
            "--marker",
            "AIRFOIL",
            "--upper-tag",
            "UPPER_BOX",
            "--lower-tag",
            "LOWER_BOX",
            "--upper-offset-chord",
            "0.04",
            "--lower-offset-chord",
            "0.06",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert mesh_out.exists()
    assert (tmp_path / "cli_dual_dual_ffd_diagnostics.csv").exists()
    assert "FFD_TAG= UPPER_BOX" in mesh_out.read_text()
    assert "FFD_TAG= LOWER_BOX" in mesh_out.read_text()


@pytest.mark.parametrize("offset_name", ["upper_offset_chord", "lower_offset_chord"])
@pytest.mark.parametrize("value", [0.0, -0.01])
def test_nonpositive_offsets_are_rejected(tmp_path, offset_name, value):
    mesh_in = _write_bootstrap_mesh(tmp_path / "bootstrap.su2")
    options = {offset_name: value}
    with pytest.raises(FFDBoxSplitError, match="must be positive"):
        _split(mesh_in, tmp_path / "dual.su2", **options)


def test_missing_box_non_bezier_and_disconnected_marker_are_rejected(tmp_path):
    mesh_in = _write_bootstrap_mesh(tmp_path / "bootstrap.su2")
    with pytest.raises(FFDBoxSplitError, match="was not found"):
        _split(mesh_in, tmp_path / "missing.su2", bootstrap_tag="DOES_NOT_EXIST")

    non_bezier = _write_bootstrap_mesh(
        tmp_path / "non_bezier.su2", blending="BSPLINE_UNIFORM"
    )
    with pytest.raises(FFDBoxSplitError, match="only FFD_BLENDING=BEZIER"):
        _split(non_bezier, tmp_path / "non_bezier_out.su2")

    disconnected = _write_bootstrap_mesh(
        tmp_path / "disconnected.su2", disconnected_marker=True
    )
    with pytest.raises(FFDBoxSplitError, match="disconnected|branched|closed"):
        _split(disconnected, tmp_path / "disconnected_out.su2")


def test_incomplete_surface_embedding_is_rejected(tmp_path):
    mesh_in = _write_bootstrap_mesh(
        tmp_path / "incomplete.su2", surface_ids=list(range(7))
    )
    with pytest.raises(FFDBoxSplitError, match="must match the requested marker exactly"):
        _split(mesh_in, tmp_path / "dual.su2")


def test_native_endpoint_roundoff_at_embedding_tolerance_is_accepted(tmp_path):
    mesh_in = _write_bootstrap_mesh(
        tmp_path / "native_roundoff.su2",
        columns=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    lines = mesh_in.read_text().splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.startswith("AIRFOIL 0 "):
            tokens = line.split()
            tokens[2] = "0.9999999999"
            lines[index] = " ".join(tokens) + "\n"
            break
    else:
        raise AssertionError("Synthetic TE surface record was not found")
    mesh_in.write_text("".join(lines))

    summary = _split(mesh_in, tmp_path / "dual.su2")

    assert summary["fixed_edge_points"] == 2


def test_surface_outside_direct_bezier_rows_is_reported_with_point_and_box(tmp_path):
    sharp_points = [
        (1.0, 0.0),
        (0.75, 0.15),
        (0.5, 0.2),
        (0.25, 0.15),
        (0.0, 0.0),
        (0.25, -0.03),
        (0.5, -0.04),
        (0.75, -0.03),
    ]
    mesh_in = _write_bootstrap_mesh(
        tmp_path / "under_resolved.su2",
        points=sharp_points,
        columns=[-0.1, 1.1],
    )
    with pytest.raises(FFDBoxSplitError, match=r"Point \d+ has v=.*UPPER_BOX"):
        _split(
            mesh_in,
            tmp_path / "dual.su2",
            upper_offset_chord=0.01,
            lower_offset_chord=0.01,
        )


def test_degenerate_direct_rows_are_rejected(tmp_path):
    flat_points = [
        (1.0, 0.0),
        (0.75, 0.0),
        (0.5, 0.0),
        (0.25, 0.0),
        (0.0, 0.0),
        (0.25, 0.0),
        (0.5, 0.0),
        (0.75, 0.0),
    ]
    mesh_in = _write_bootstrap_mesh(tmp_path / "flat.su2", points=flat_points)
    with pytest.raises(FFDBoxSplitError, match="Degenerate or crossed rows"):
        _split(
            mesh_in,
            tmp_path / "dual.su2",
            upper_offset_chord=1.0e-16,
            lower_offset_chord=1.0e-16,
        )


def test_dual_rewrite_supports_independent_upper_lower_columns(tmp_path):
    bootstrap = _write_bootstrap_mesh(tmp_path / "bootstrap.su2")
    dual = tmp_path / "dual.su2"
    _split(bootstrap, dual)
    original_points = read_su2_mesh(dual)["points"]

    rewritten = tmp_path / "dual_rewritten.su2"
    summary = rewrite_dual_ffd_boxes_with_columns_and_reembed(
        dual,
        rewritten,
        marker="AIRFOIL",
        upper_tag="UPPER_BOX",
        lower_tag="LOWER_BOX",
        upper_columns=[-0.1, 0.2, 0.5, 0.8, 1.1],
        lower_columns=[-0.1, 0.15, 0.35, 0.6, 0.85, 1.1],
        upper_offset_chord=0.04,
        lower_offset_chord=0.06,
        diagnostics_csv=False,
    )

    upper = _read_box(rewritten, "UPPER_BOX")
    lower = _read_box(rewritten, "LOWER_BOX")
    assert upper["degree"]["i"] == 4
    assert lower["degree"]["i"] == 5
    assert len(upper["controls"]) == 20
    assert len(lower["controls"]) == 24
    assert summary["upper_columns"] == pytest.approx(
        [-0.1, 0.2, 0.5, 0.8, 1.1]
    )
    assert summary["lower_columns"] == pytest.approx(
        [-0.1, 0.15, 0.35, 0.6, 0.85, 1.1]
    )
    assert set(summary["upper_column_index_by_x"].values()) == set(range(5))
    assert set(summary["lower_column_index_by_x"].values()) == set(range(6))

    rewritten_points = read_su2_mesh(rewritten)["points"]
    assert rewritten_points == original_points
    _assert_surface_reconstruction(upper, rewritten_points)
    _assert_surface_reconstruction(lower, rewritten_points)


@pytest.mark.skipif(
    shutil.which("SU2_DEF") is None
    or os.environ.get("RUN_SU2_DEF_SMOKE", "NO").upper() != "YES",
    reason="set RUN_SU2_DEF_SMOKE=YES to enable the native SU2_DEF smoke test",
)
def test_su2_def_reads_both_boxes_at_zero_deformation(tmp_path):
    mesh_in = _write_bootstrap_mesh(tmp_path / "bootstrap.su2")
    mesh_out = tmp_path / "dual.su2"
    _split(mesh_in, mesh_out)

    config = tmp_path / "zero_deformation.cfg"
    config.write_text(
        "\n".join(
            [
                "SOLVER= EULER",
                "MATH_PROBLEM= DIRECT",
                "MESH_FILENAME= dual.su2",
                "MESH_FORMAT= SU2",
                "MESH_OUT_FILENAME= zero_deformation_out",
                "MARKER_EULER= ( AIRFOIL )",
                "MARKER_FAR= ( FARFIELD )",
                "MARKER_PLOTTING= ( AIRFOIL )",
                "MARKER_MONITORING= ( AIRFOIL )",
                "DV_KIND= FFD_CONTROL_POINT_2D",
                "DV_MARKER= ( AIRFOIL )",
                "DV_PARAM= ( UPPER_BOX, 2, 1, 0.0, 1.0 )",
                "DV_VALUE= 0.0",
                "DEFORM_LINEAR_SOLVER= FGMRES",
                "DEFORM_LINEAR_SOLVER_PREC= LU_SGS",
                "DEFORM_LINEAR_SOLVER_ITER= 100",
                "DEFORM_NONLINEAR_ITER= 1",
                "DEFORM_LINEAR_SOLVER_ERROR= 1E-14",
                "DEFORM_STIFFNESS_TYPE= INVERSE_VOLUME",
                "FFD_TOLERANCE= 1E-12",
                "FFD_ITERATIONS= 200",
                "OUTPUT_FILES= ( PARAVIEW_ASCII )",
                "",
            ]
        )
    )
    env = dict(os.environ)
    env.setdefault("OMPI_MCA_osc", "pt2pt")
    result = subprocess.run(
        [shutil.which("SU2_DEF"), config.name],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    combined_output = result.stdout + result.stderr
    assert result.returncode == 0, combined_output
    assert "2 Free Form Deformation boxes" in combined_output
    assert "FFD box tag: UPPER_BOX" in combined_output
    assert "FFD box tag: LOWER_BOX" in combined_output

    deformed_mesh = tmp_path / "zero_deformation_out.su2"
    assert deformed_mesh.exists()
    original_points = read_su2_mesh(mesh_out)["points"]
    deformed_points = read_su2_mesh(deformed_mesh)["points"]
    assert deformed_points.keys() == original_points.keys()
    for point_id in original_points:
        assert deformed_points[point_id] == pytest.approx(
            original_points[point_id], abs=1.0e-12
        )
