#!/usr/bin/env python3

"""Prepare and compare dual progressive FFD Bezier/B-spline responses."""

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import SU2

from SU2.opt.bspline_def import (
    classify_sides,
    extract_marker_nodes,
    infer_chord,
    read_su2_mesh,
)
from SU2.opt.progressive_ffd_core import get_progressive_ffd_options
from SU2.opt.progressive_ffd_levels import build_initial_ffd_level, write_ffd_level_config
from SU2.opt.progressive_ffd_prepare import _run_su2_def
from SU2.opt.progressive_hh_core import get_progressive_hh_options


EXPECTED_NDV = 14
PROBE_ARTIFACTS = (
    "mesh_deformed.su2",
    "ffd_boxes_0.vtk",
    "ffd_boxes_1.vtk",
    "ffd_boxes_def_0.vtk",
    "ffd_boxes_def_1.vtk",
    "surface_deformed.vtu",
)
COMPARISON_REQUEST_KEYS = (
    "raw_mesh",
    "raw_mesh_sha256",
    "marker",
    "initial_columns",
    "bootstrap_tag",
    "bootstrap_y_padding_chord",
    "upper_tag",
    "lower_tag",
    "upper_offset_chord",
    "lower_offset_chord",
)


def _parse_order(value):
    tokens = [token.strip() for token in str(value).split(",")]
    if len(tokens) != 3:
        raise argparse.ArgumentTypeError("B-spline order must be i,j,k")
    try:
        result = tuple(int(token) for token in tokens)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("B-spline orders must be integers") from exc
    if any(order < 2 for order in result):
        raise argparse.ArgumentTypeError("B-spline orders must be >= 2")
    return result


def _config_directory(config_path):
    return os.path.dirname(os.path.abspath(config_path))


def _resolve_cfg_file(config_path, value):
    value = str(value)
    if os.path.isabs(value):
        return os.path.abspath(value)
    return os.path.abspath(os.path.join(_config_directory(config_path), value))


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_finite(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be a finite number, got {value!r}") from exc
    if not math.isfinite(value):
        raise RuntimeError(f"{name} must be finite, got {value!r}")
    return value


def _prepared_mesh_path(output, blending, bspline_order):
    if str(blending).upper() == "BEZIER":
        filename = "rae2822_dual_bezier.su2"
    else:
        filename = f"rae2822_dual_bspline_o{int(bspline_order[0])}.su2"
    return os.path.abspath(os.path.join(output, filename))


def _comparison_signature(manifest):
    request = manifest.get("request", {})
    missing = [key for key in COMPARISON_REQUEST_KEYS if key not in request]
    if missing:
        raise RuntimeError(f"Preparation manifest is missing comparison keys: {missing}")
    return {key: request[key] for key in COMPARISON_REQUEST_KEYS}


def _validate_zero_dv_manifest(manifest, label):
    result = manifest.get("result", {})
    chord = _require_finite(result.get("chord"), f"{label} chord")
    error = _require_finite(
        result.get("smoke_coordinate_error"),
        f"{label} zero-DV coordinate error",
    )
    tolerance = 1.0e-10 * max(1.0, chord)
    if error > tolerance:
        raise RuntimeError(
            f"{label} zero-DV error {error:.6e} exceeds {tolerance:.6e}"
        )
    return {"chord": chord, "error": error, "tolerance": tolerance}


def _write_rows(path, fieldnames, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _surface_metadata(mesh_path, marker):
    mesh = read_su2_mesh(mesh_path)
    marker_tag, node_ids, closed = extract_marker_nodes(mesh, marker)
    x_le, x_te, chord = infer_chord(
        mesh["points"],
        node_ids,
        {"mode": "auto", "x_le": None, "x_te": None},
    )
    chord = _require_finite(chord, "airfoil chord")
    if chord <= 0.0:
        raise RuntimeError(f"Airfoil chord must be positive, got {chord:.16g}")
    x_over_c = []
    for point_id in node_ids:
        point = mesh["points"][point_id]
        for axis, value in enumerate(point):
            _require_finite(value, f"mesh point {point_id} coordinate {axis}")
        x_over_c.append((float(point[0]) - x_le) / chord)
    sides = classify_sides(
        node_ids,
        x_over_c,
        [float(mesh["points"][point_id][1]) for point_id in node_ids],
        closed=closed,
    )
    return {
        "mesh": mesh,
        "marker": marker_tag,
        "node_ids": node_ids,
        "x_over_c": x_over_c,
        "sides": [str(side).upper() for side in sides],
        "chord": float(chord),
    }


def _trapz(xs, ys):
    pairs = sorted(
        zip(
            (_require_finite(x, "integration coordinate") for x in xs),
            (_require_finite(y, "integration value") for y in ys),
        )
    )
    total = 0.0
    for (x0, y0), (x1, y1) in zip(pairs[:-1], pairs[1:]):
        total += 0.5 * (y0 + y1) * max(0.0, x1 - x0)
    return total


def response_metrics(records, target_side):
    target_side = str(target_side).upper()
    for row in records:
        for key in ("x_over_c", "dy", "displacement"):
            _require_finite(row[key], f"response {key}")
    target = [
        row
        for row in records
        if row["surface_side"] == target_side and not row["edge"]
    ]
    opposite = [
        row
        for row in records
        if row["surface_side"] != target_side and not row["edge"]
    ]
    edges = [row for row in records if row["edge"]]
    leading_edge = [row for row in records if row.get("le", False)]
    trailing_edge = [row for row in records if row.get("te", False)]
    if not target:
        raise ValueError(f"No target-side records for {target_side}")

    peak = max(target, key=lambda row: abs(float(row["dy"])))
    maximum = abs(float(peak["dy"]))
    widths = {}
    for fraction in (0.01, 0.10, 0.50):
        active = [
            float(row["x_over_c"])
            for row in target
            if abs(float(row["dy"])) >= fraction * maximum
        ]
        widths[fraction] = max(active) - min(active) if active else 0.0

    xs = [row["x_over_c"] for row in target]
    abs_dy = [abs(float(row["dy"])) for row in target]
    return {
        "target_max_displacement": maximum,
        "opposite_max_displacement": max(
            [abs(float(row["dy"])) for row in opposite] or [0.0]
        ),
        "edge_max_displacement": max(
            [float(row["displacement"]) for row in edges] or [0.0]
        ),
        "le_max_displacement": max(
            [float(row["displacement"]) for row in leading_edge] or [0.0]
        ),
        "te_max_displacement": max(
            [float(row["displacement"]) for row in trailing_edge] or [0.0]
        ),
        "x_over_c_at_peak": float(peak["x_over_c"]),
        "support_width_1pct": widths[0.01],
        "support_width_10pct": widths[0.10],
        "support_width_50pct": widths[0.50],
        "l1_dy": _trapz(xs, abs_dy),
        "l2_dy": math.sqrt(_trapz(xs, [value * value for value in abs_dy])),
    }


def _correlation_columns(matrix):
    if not matrix or not matrix[0]:
        return []
    nrow = len(matrix)
    ncol = len(matrix[0])
    columns = [
        [
            _require_finite(matrix[row][col], f"response matrix [{row},{col}]")
            for row in range(nrow)
        ]
        for col in range(ncol)
    ]
    result = [[0.0] * ncol for _ in range(ncol)]
    for i in range(ncol):
        mean_i = sum(columns[i]) / nrow
        centered_i = [value - mean_i for value in columns[i]]
        norm_i = math.sqrt(sum(value * value for value in centered_i))
        for j in range(ncol):
            mean_j = sum(columns[j]) / nrow
            centered_j = [value - mean_j for value in columns[j]]
            norm_j = math.sqrt(sum(value * value for value in centered_j))
            denominator = norm_i * norm_j
            result[i][j] = (
                sum(a * b for a, b in zip(centered_i, centered_j)) / denominator
                if denominator > 0.0
                else (1.0 if i == j else 0.0)
            )
    return result


def _correlation_metrics(correlation):
    metrics = []
    for dv_index, row in enumerate(correlation):
        candidates = [
            (abs(float(value)), other_index, float(value))
            for other_index, value in enumerate(row)
            if other_index != dv_index
        ]
        if not candidates:
            metrics.append(
                {
                    "most_correlated_dv_index": "",
                    "correlation_with_most_correlated_dv": 0.0,
                    "max_abs_correlation_other_dv": 0.0,
                }
            )
            continue
        absolute_value, other_index, signed_value = max(
            candidates,
            key=lambda item: (item[0], -item[1]),
        )
        metrics.append(
            {
                "most_correlated_dv_index": other_index,
                "correlation_with_most_correlated_dv": signed_value,
                "max_abs_correlation_other_dv": absolute_value,
            }
        )
    return metrics


def _write_matrix(path, metadata, matrix, prefix="dv"):
    fieldnames = ["point_id", "surface_side", "x_over_c"] + [
        f"{prefix}_{index:02d}" for index in range(len(matrix[0]) if matrix else 0)
    ]
    rows = []
    for row_index, item in enumerate(metadata):
        row = dict(item)
        for column_index, value in enumerate(matrix[row_index] if matrix else []):
            row[f"{prefix}_{column_index:02d}"] = value
        rows.append(row)
    _write_rows(path, fieldnames, rows)


def _color(value):
    value = max(-1.0, min(1.0, float(value)))
    if value >= 0.0:
        red, green, blue = 255, int(255 * (1.0 - value)), int(255 * (1.0 - value))
    else:
        red, green, blue = int(255 * (1.0 + value)), int(255 * (1.0 + value)), 255
    return f"rgb({red},{green},{blue})"


def _write_heatmap_svg(path, matrix, title):
    rows = len(matrix)
    columns = len(matrix[0]) if matrix else 0
    cell_w = 32
    cell_h = max(1.0, 480.0 / max(1, rows))
    margin_x = 70
    margin_y = 45
    width = margin_x + columns * cell_w + 20
    height = margin_y + rows * cell_h + 40
    maximum = max([abs(float(value)) for row in matrix for value in row] or [1.0])
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="10" y="22" font-family="sans-serif" font-size="16">{title}</text>',
    ]
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            normalized = float(value) / maximum if maximum else 0.0
            lines.append(
                f'<rect x="{margin_x + column_index * cell_w}" '
                f'y="{margin_y + row_index * cell_h:.3f}" width="{cell_w}" '
                f'height="{cell_h + 0.2:.3f}" fill="{_color(normalized)}"/>'
            )
    for column_index in range(columns):
        lines.append(
            f'<text x="{margin_x + column_index * cell_w + 4}" y="{margin_y - 7}" '
            f'font-family="sans-serif" font-size="10">{column_index}</text>'
        )
    lines.append("</svg>")
    Path(path).write_text("\n".join(lines) + "\n")


def _write_profiles_svg(path, profiles, title):
    width, height = 1000, 620
    left, right, top, bottom = 70, 20, 45, 60
    plot_w = width - left - right
    plot_h = height - top - bottom
    maximum = max([abs(float(row["dy"])) for row in profiles] or [1.0])
    grouped = {}
    for row in profiles:
        grouped.setdefault((row["blending"], int(row["dv_index"])), []).append(row)
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="10" y="24" font-family="sans-serif" font-size="17">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_h / 2}" x2="{left + plot_w}" y2="{top + plot_h / 2}" stroke="#888"/>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="black"/>',
    ]
    for index, ((blending, dv_index), rows) in enumerate(sorted(grouped.items())):
        rows = sorted(rows, key=lambda row: float(row["x_over_c"]))
        points = []
        for row in rows:
            x = left + plot_w * float(row["x_over_c"])
            y = top + 0.5 * plot_h * (1.0 - float(row["dy"]) / maximum)
            points.append(f"{x:.2f},{y:.2f}")
        dash = "" if blending == "BEZIER" else ' stroke-dasharray="5,3"'
        color = palette[dv_index % len(palette)]
        lines.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" '
            f'stroke-width="1.2" opacity="0.7"{dash}/>'
        )
    lines.extend(
        [
            f'<text x="{left + plot_w / 2}" y="{height - 15}" font-family="sans-serif">x/c</text>',
            f'<text x="8" y="{top + plot_h / 2}" font-family="sans-serif">dy</text>',
            '<text x="740" y="24" font-family="sans-serif" font-size="11">solid=BEZIER dashed=BSPLINE_UNIFORM</text>',
            "</svg>",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n")


def _run_prepare(config_path, workdir, output, blending, bspline_order, nproc):
    source = SU2.io.Config(config_path)
    raw_mesh = _resolve_cfg_file(config_path, source["MESH_FILENAME"])
    config = SU2.io.Config(copy.deepcopy(dict(source)))
    config["MESH_FILENAME"] = raw_mesh
    config["FFD_BLENDING"] = blending
    config["FFD_BSPLINE_ORDER"] = ", ".join(str(value) for value in bspline_order)
    config["PROGRESSIVE_FFD_AUTO_PREPARE"] = "YES"
    config["PROGRESSIVE_FFD_PREPARE_ONLY"] = "YES"
    config["PROGRESSIVE_FFD_PREPARE_OVERWRITE"] = "YES"
    config["PROGRESSIVE_FFD_PREPARE_SMOKE_TEST"] = "YES"
    prepared_mesh = _prepared_mesh_path(output, blending, bspline_order)
    config["PROGRESSIVE_FFD_PREPARED_MESH"] = prepared_mesh
    variant_cfg = os.path.join(workdir, "Config_FFD.cfg")
    config.dump(variant_cfg)

    command = [
        sys.executable,
        str(REPO_ROOT / "shape_optimization.py"),
        "-f",
        os.path.basename(variant_cfg),
        "-n",
        str(int(nproc)),
    ]
    environment = dict(os.environ)
    environment.setdefault("OMPI_MCA_osc", "pt2pt")
    with open(os.path.join(workdir, "prepare.log"), "w") as log:
        result = subprocess.run(
            command,
            cwd=workdir,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"Preparation failed for {blending}; see {workdir}/prepare.log")
    manifest_path = os.path.join(workdir, "FFD_PREP", "prepare_manifest.json")
    with open(manifest_path) as fp:
        manifest = json.load(fp)
    manifest_mesh = os.path.abspath(manifest["artifacts"]["prepared_mesh"])
    if manifest_mesh != prepared_mesh or not os.path.isfile(prepared_mesh):
        raise RuntimeError(
            f"Preparation produced an unexpected mesh for {blending}: {manifest_mesh}"
        )
    return variant_cfg, prepared_mesh, manifest


def _build_level_zero(config_path, prepared_mesh, workdir):
    config = SU2.io.Config(config_path)
    config["MESH_FILENAME"] = os.path.abspath(prepared_mesh)
    hh_opts = get_progressive_hh_options(config)
    opts = get_progressive_ffd_options(config, hh_opts)
    manifest_path = os.path.join(workdir, "FFD_PREP", "prepare_manifest.json")
    with open(manifest_path) as fp:
        result = json.load(fp)["result"]
    opts["ffd_active_xmin"] = float(result["x_le"])
    opts["ffd_active_xmax"] = float(result["x_te"])

    previous = os.getcwd()
    os.chdir(workdir)
    try:
        level = build_initial_ffd_level(config, opts)
        level_cfg = os.path.abspath(write_ffd_level_config(config, level, opts))
    finally:
        os.chdir(previous)
    return level_cfg, os.path.join(workdir, "LEVEL_0", "ffd_level0.su2")


def _probe_variant(
    label,
    config_path,
    mesh_path,
    marker,
    dv_value,
    nproc,
    workdir,
    expected_ndv=EXPECTED_NDV,
):
    base_config = SU2.io.Config(config_path)
    n_dv = sum(base_config["DEFINITION_DV"]["SIZE"])
    if n_dv != int(expected_ndv):
        raise RuntimeError(
            f"{label} comparison requires exactly {expected_ndv} DVs, got {n_dv}"
        )
    dv_value = _require_finite(dv_value, "DV probe value")
    if dv_value == 0.0:
        raise RuntimeError("DV probe value must be non-zero")
    reference = _surface_metadata(mesh_path, marker)
    profiles = []
    summary = []
    response_matrix = [[0.0] * n_dv for _ in reference["node_ids"]]

    for dv_index in range(n_dv):
        probe_dir = os.path.join(workdir, "PROBES", f"dv_{dv_index:02d}")
        os.makedirs(probe_dir, exist_ok=True)
        config = SU2.io.Config(copy.deepcopy(dict(base_config)))
        values = [0.0] * n_dv
        values[dv_index] = float(dv_value)
        config.unpack_dvs(values, [0.0] * n_dv)
        config["MESH_FILENAME"] = os.path.abspath(mesh_path)
        config["MESH_OUT_FILENAME"] = "mesh_deformed"
        probe_cfg = os.path.join(probe_dir, "probe.cfg")
        probe_log = os.path.join(probe_dir, "probe.log")
        config.dump(probe_cfg)
        _run_su2_def(probe_cfg, nproc, probe_log)
        missing_artifacts = [
            filename
            for filename in PROBE_ARTIFACTS
            if not os.path.isfile(os.path.join(probe_dir, filename))
        ]
        if missing_artifacts:
            raise RuntimeError(
                f"{label} DV {dv_index} is missing probe artifacts: "
                f"{missing_artifacts}"
            )

        deformed = read_su2_mesh(os.path.join(probe_dir, "mesh_deformed.su2"))
        if set(deformed["points"]) != set(reference["mesh"]["points"]):
            raise RuntimeError(f"{label} DV {dv_index} changed the mesh point set")

        target_side = "UPPER" if dv_index < n_dv // 2 else "LOWER"
        records = []
        for row_index, (point_id, x_over_c, surface_side) in enumerate(
            zip(reference["node_ids"], reference["x_over_c"], reference["sides"])
        ):
            initial = reference["mesh"]["points"][point_id]
            final = deformed["points"][point_id]
            for axis, value in enumerate(initial):
                _require_finite(
                    value,
                    f"{label} DV {dv_index} initial point {point_id} axis {axis}",
                )
            for axis, value in enumerate(final):
                _require_finite(
                    value,
                    f"{label} DV {dv_index} deformed point {point_id} axis {axis}",
                )
            dx = _require_finite(
                float(final[0]) - float(initial[0]),
                f"{label} DV {dv_index} dx at point {point_id}",
            )
            dy = _require_finite(
                float(final[1]) - float(initial[1]),
                f"{label} DV {dv_index} dy at point {point_id}",
            )
            displacement = math.hypot(dx, dy)
            x_over_c = _require_finite(x_over_c, "surface x/c")
            le = abs(x_over_c) <= 1.0e-12
            te = abs(x_over_c - 1.0) <= 1.0e-12
            edge = le or te
            record = {
                "blending": label,
                "dv_index": dv_index,
                "dv_side": target_side,
                "point_id": point_id,
                "surface_side": surface_side,
                "edge": edge,
                "le": le,
                "te": te,
                "x_over_c": x_over_c,
                "dx": dx,
                "dy": dy,
                "displacement": displacement,
            }
            records.append(record)
            profiles.append(record)
            response_matrix[row_index][dv_index] = _require_finite(
                dy / dv_value,
                f"{label} normalized response [{row_index},{dv_index}]",
            )

        metrics = response_metrics(records, target_side)
        summary.append(
            {
                "blending": label,
                "dv_index": dv_index,
                "side": target_side,
                "dv_value": dv_value,
                "chord": reference["chord"],
                **metrics,
            }
        )
        isolation_tolerance = 1.0e-12 * max(1.0, reference["chord"])
        if metrics["target_max_displacement"] <= isolation_tolerance:
            raise RuntimeError(f"{label} DV {dv_index} is inactive")
        if metrics["opposite_max_displacement"] > isolation_tolerance:
            raise RuntimeError(f"{label} DV {dv_index} moves the opposite side")
        if metrics["le_max_displacement"] > isolation_tolerance:
            raise RuntimeError(f"{label} DV {dv_index} moves the leading edge")
        if metrics["te_max_displacement"] > isolation_tolerance:
            raise RuntimeError(f"{label} DV {dv_index} moves the trailing edge")

    metadata = [
        {
            "point_id": point_id,
            "surface_side": side,
            "x_over_c": x,
        }
        for point_id, side, x in zip(
            reference["node_ids"], reference["sides"], reference["x_over_c"]
        )
    ]
    return summary, profiles, metadata, response_matrix


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--bspline-order", type=_parse_order, default=(4, 2, 2))
    parser.add_argument("--dv-value", type=float, default=0.01)
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--output", default="FFD_COMPARISON")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _path_is_within(path, directory):
    path = os.path.abspath(path)
    directory = os.path.abspath(directory)
    try:
        return os.path.commonpath((path, directory)) == directory
    except ValueError:
        return False


def main(argv=None):
    args = build_parser().parse_args(argv)
    config_path = os.path.abspath(args.config)
    output = os.path.abspath(args.output)
    if not os.path.isfile(config_path):
        raise RuntimeError(f"Source config does not exist: {config_path}")
    if int(args.nproc) <= 0:
        raise RuntimeError("--nproc must be positive")
    dv_value = _require_finite(args.dv_value, "DV probe value")
    if dv_value == 0.0:
        raise RuntimeError("--dv-value must be non-zero")

    source_config = SU2.io.Config(config_path)
    raw_mesh = _resolve_cfg_file(config_path, source_config["MESH_FILENAME"])
    for protected in (config_path, raw_mesh):
        if _path_is_within(protected, output):
            raise RuntimeError(
                f"Output directory must not contain source input {protected}"
            )
    source_hash_before = _sha256_file(config_path)

    if os.path.exists(output):
        if not args.overwrite:
            raise RuntimeError(f"Output directory exists: {output}; pass --overwrite")
        shutil.rmtree(output)
    os.makedirs(output)

    marker = str(source_config.get("PROGRESSIVE_FFD_MARKER", "AIRFOIL"))
    all_summary = []
    all_profiles = []
    metadata = None
    prepared_meshes = {}
    preparation_manifests = {}
    zero_dv_checks = {}
    comparison_signature = None

    variants = (
        ("BEZIER", (2, 2, 2)),
        ("BSPLINE_UNIFORM", args.bspline_order),
    )
    for label, order in variants:
        workdir = os.path.join(output, label)
        os.makedirs(workdir)
        variant_cfg, prepared_mesh, preparation_manifest = _run_prepare(
            config_path,
            workdir,
            output,
            label,
            order,
            args.nproc,
        )
        signature = _comparison_signature(preparation_manifest)
        if comparison_signature is None:
            comparison_signature = signature
        elif signature != comparison_signature:
            raise RuntimeError(
                f"{label} preparation does not match the reference stations/offsets"
            )
        zero_dv_checks[label] = _validate_zero_dv_manifest(
            preparation_manifest,
            label,
        )
        prepared_meshes[label] = prepared_mesh
        preparation_manifests[label] = os.path.join(
            workdir,
            "FFD_PREP",
            "prepare_manifest.json",
        )

        level_cfg, level_mesh = _build_level_zero(variant_cfg, prepared_mesh, workdir)
        summary, profiles, this_metadata, matrix = _probe_variant(
            label,
            level_cfg,
            level_mesh,
            marker,
            dv_value,
            args.nproc,
            workdir,
            expected_ndv=EXPECTED_NDV,
        )
        if metadata is not None and this_metadata != metadata:
            raise RuntimeError(
                f"{label} surface metadata does not match the reference variant"
            )

        correlation = _correlation_columns(matrix)
        correlation_by_dv = _correlation_metrics(correlation)
        for row in summary:
            row.update(correlation_by_dv[int(row["dv_index"])])

        all_summary.extend(summary)
        all_profiles.extend(profiles)
        metadata = this_metadata

        _write_matrix(
            os.path.join(output, f"response_matrix_{label.lower()}.csv"),
            metadata,
            matrix,
        )
        correlation_metadata = [
            {"point_id": index, "surface_side": "DV", "x_over_c": index}
            for index in range(len(correlation))
        ]
        _write_matrix(
            os.path.join(output, f"response_correlation_{label.lower()}.csv"),
            correlation_metadata,
            correlation,
            prefix="dv",
        )
        _write_heatmap_svg(
            os.path.join(output, f"response_heatmap_{label.lower()}.svg"),
            matrix,
            f"FFD response matrix: {label}",
        )
        _write_heatmap_svg(
            os.path.join(output, f"correlation_heatmap_{label.lower()}.svg"),
            correlation,
            f"FFD response correlation: {label}",
        )

    source_hash_after = _sha256_file(config_path)
    if source_hash_after != source_hash_before:
        raise RuntimeError("The source configuration changed during comparison")

    _write_rows(
        os.path.join(output, "dv_response_summary.csv"),
        list(all_summary[0]),
        all_summary,
    )
    _write_rows(
        os.path.join(output, "dv_response_profiles.csv"),
        list(all_profiles[0]),
        all_profiles,
    )
    _write_profiles_svg(
        os.path.join(output, "dv_response_profiles.svg"),
        all_profiles,
        "Dual FFD control-point responses",
    )
    aggregate_rows = []
    metric_names = (
        "target_max_displacement",
        "opposite_max_displacement",
        "edge_max_displacement",
        "le_max_displacement",
        "te_max_displacement",
        "support_width_1pct",
        "support_width_10pct",
        "support_width_50pct",
        "l1_dy",
        "l2_dy",
        "max_abs_correlation_other_dv",
    )
    for label, _ in variants:
        selected = [row for row in all_summary if row["blending"] == label]
        aggregate = {"blending": label, "ndv": len(selected)}
        for metric in metric_names:
            values = [float(row[metric]) for row in selected]
            aggregate[f"mean_{metric}"] = sum(values) / len(values)
            aggregate[f"max_{metric}"] = max(values)
        aggregate_rows.append(aggregate)
    _write_rows(
        os.path.join(output, "comparison_aggregate.csv"),
        list(aggregate_rows[0]),
        aggregate_rows,
    )

    aggregate_by_label = {row["blending"]: row for row in aggregate_rows}
    locality_rows = []
    for metric in (
        "support_width_1pct",
        "support_width_10pct",
        "support_width_50pct",
        "l1_dy",
        "l2_dy",
    ):
        bezier_mean = _require_finite(
            aggregate_by_label["BEZIER"][f"mean_{metric}"],
            f"BEZIER mean {metric}",
        )
        bspline_mean = _require_finite(
            aggregate_by_label["BSPLINE_UNIFORM"][f"mean_{metric}"],
            f"BSPLINE_UNIFORM mean {metric}",
        )
        ratio = bspline_mean / bezier_mean if bezier_mean != 0.0 else 0.0
        locality_rows.append(
            {
                "metric": metric,
                "bezier_mean": bezier_mean,
                "bspline_mean": bspline_mean,
                "bspline_to_bezier_ratio": ratio,
                "fractional_reduction": 1.0 - ratio,
            }
        )
    _write_rows(
        os.path.join(output, "locality_comparison.csv"),
        list(locality_rows[0]),
        locality_rows,
    )

    with open(os.path.join(output, "comparison_manifest.json"), "w") as fp:
        json.dump(
            {
                "source_config": config_path,
                "source_config_sha256": source_hash_before,
                "bspline_order": list(args.bspline_order),
                "dv_value": dv_value,
                "nproc": int(args.nproc),
                "variants": [label for label, _ in variants],
                "ndv_per_variant": EXPECTED_NDV,
                "prepared_meshes": prepared_meshes,
                "preparation_manifests": preparation_manifests,
                "comparison_signature": comparison_signature,
                "zero_dv_checks": zero_dv_checks,
                "isolation_tolerance_scale": 1.0e-12,
                "zero_dv_tolerance_scale": 1.0e-10,
                "locality_comparison": locality_rows,
            },
            fp,
            indent=2,
            sort_keys=True,
        )
        fp.write("\n")
    print(f"[FFD_BLEND_COMPARE] Completed: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
