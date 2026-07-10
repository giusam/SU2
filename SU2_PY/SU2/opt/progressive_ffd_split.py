#!/usr/bin/env python

"""Split one 2D bootstrap FFD box into independent upper/lower boxes."""

import csv
import math
import os
import tempfile

from SU2.opt.bspline_def import (
    BSplineDefError,
    classify_sides,
    extract_marker_nodes,
    infer_chord,
    read_su2_mesh,
)
from SU2.opt.progressive_ffd_mesh import (
    COORDS_ONLY_2D,
    COORDS_ONLY_3D,
    INDEXED_2D,
    INDEXED_2D_WITH_Z,
    INDEXED_3D,
    FFDMeshError,
    _as_int_if_possible,
    _bezier_1d,
    _find_key_line,
    _find_tagged_ffd_block,
    _format_float,
    _format_key_value,
    _format_surface_line,
    _infer_axes_from_control_points,
    _infer_surface_param_location,
    _invert_monotone_bezier,
    _line_key,
    _line_value,
    _normalize_tag,
    _parse_control_points,
    _parse_count_block,
    _parse_degree,
    _parse_int_value,
    _parse_blending_spec,
    _split_tokens,
)
from SU2.opt.progressive_ffd_blending import (
    BEZIER,
    BSPLINE_UNIFORM,
    FFDBlendingSpec,
    evaluate_curve,
    invert_monotone_curve,
    make_blending_spec,
    validate_blending_spec,
)


class FFDBoxSplitError(FFDMeshError):
    """Raised when a bootstrap FFD box cannot be split safely."""


def _atomic_write_lines(path, lines):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".dual_ffd_", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w") as fp:
            fp.writelines(lines)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _atomic_write_csv(path, fieldnames, rows):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".dual_ffd_diag_", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _replace_key_line(lines, key, value):
    line_id = _find_key_line(lines, 0, len(lines), key)
    if line_id is None:
        raise FFDBoxSplitError(f"Required mesh key {key} was not found")
    lines[line_id] = _format_key_value(key, value)


def _parse_zero_relation_count(lines, start, end, key):
    line_id = _find_key_line(lines, start, end, key)
    if line_id is None:
        raise FFDBoxSplitError(f"Bootstrap box is missing {key}")
    count = _parse_int_value(lines[line_id])
    if count != 0:
        raise FFDBoxSplitError(
            f"Bootstrap box must be independent: {key}={count}, expected 0"
        )


def _validate_output_path(mesh_in, mesh_out, overwrite):
    mesh_in_abs = os.path.abspath(mesh_in)
    mesh_out_abs = os.path.abspath(mesh_out)
    if not os.path.exists(mesh_in_abs):
        raise FFDBoxSplitError(f"Input mesh does not exist: {mesh_in}")
    if os.path.exists(mesh_out_abs) and not overwrite:
        raise FFDBoxSplitError(
            f"Output mesh already exists: {mesh_out}; pass overwrite=True to replace it"
        )
    if mesh_in_abs == mesh_out_abs and not overwrite:
        raise FFDBoxSplitError("Refusing to overwrite the input mesh without overwrite=True")


def _group_surface_samples(samples, tol):
    if not samples:
        raise FFDBoxSplitError("Cannot build an interpolant from an empty surface")

    grouped = []
    for x, y in sorted((float(x), float(y)) for x, y in samples):
        if not grouped or abs(x - grouped[-1][0]) > tol:
            grouped.append([x, [y]])
        else:
            grouped[-1][1].append(y)

    return [(x, sum(values) / float(len(values))) for x, values in grouped]


def _set_side_endpoint(samples, edge_points, x_edge, side, tol):
    candidates = [
        float(y)
        for x, y in edge_points
        if abs(float(x) - float(x_edge)) <= tol
    ]
    if not candidates:
        raise FFDBoxSplitError(
            f"Could not infer the {side} surface ordinate at x={x_edge:.16g}"
        )
    value = max(candidates) if side == "upper" else min(candidates)
    filtered = [
        (float(x), float(y))
        for x, y in samples
        if abs(float(x) - float(x_edge)) > tol
    ]
    return filtered + [(float(x_edge), value)]


def _make_linear_interpolant(samples, tol):
    samples = _group_surface_samples(samples, tol)
    if len(samples) < 2:
        raise FFDBoxSplitError("Surface interpolant requires at least two x stations")

    xs = [item[0] for item in samples]
    ys = [item[1] for item in samples]

    def interpolate(x):
        x = float(x)
        if x <= xs[0] + tol:
            return ys[0]
        if x >= xs[-1] - tol:
            return ys[-1]

        lo = 0
        hi = len(xs) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if xs[mid] <= x:
                lo = mid
            else:
                hi = mid

        span = xs[hi] - xs[lo]
        if span <= tol:
            return 0.5 * (ys[lo] + ys[hi])
        alpha = (x - xs[lo]) / span
        return ys[lo] + alpha * (ys[hi] - ys[lo])

    return interpolate


def _build_curved_control_point_lines(
    columns,
    control_y,
    z_planes,
    coord_dim,
    control_format,
):
    lines = []
    indexed = control_format in (INDEXED_3D, INDEXED_2D_WITH_Z, INDEXED_2D)
    if indexed:
        indices = (
            (i, j, k)
            for i in range(len(columns))
            for j in range(len(control_y))
            for k in range(len(z_planes))
        )
    else:
        # This is the tensor-product order assumed by the shared parser for
        # coordinate-only blocks: i varies fastest, followed by j and k.
        indices = (
            (i, j, k)
            for k in range(len(z_planes))
            for j in range(len(control_y))
            for i in range(len(columns))
        )

    for i, j, k in indices:
        x = float(columns[i])
        y = float(control_y[j][i])
        z = float(z_planes[k])
        coords = [x, y, z]
        if control_format == INDEXED_3D:
            values = [str(i), str(j), str(k)] + [
                _format_float(value) for value in coords[:3]
            ]
        elif control_format == INDEXED_2D_WITH_Z:
            values = [str(i), str(j)] + [
                _format_float(value) for value in coords[:3]
            ]
        elif control_format == INDEXED_2D:
            values = [str(i), str(j)] + [
                _format_float(value) for value in coords[:2]
            ]
        elif control_format == COORDS_ONLY_3D:
            values = [_format_float(value) for value in coords[:3]]
        elif control_format == COORDS_ONLY_2D:
            values = [_format_float(value) for value in coords[:2]]
        else:
            raise FFDBoxSplitError(
                f"Unsupported FFD control point format: {control_format}"
            )
        lines.append(" ".join(values) + "\n")
    return lines


def _build_curved_corner_lines(columns, control_y):
    return [
        f"{_format_float(columns[0])} {_format_float(control_y[0][0])}\n",
        f"{_format_float(columns[-1])} {_format_float(control_y[0][-1])}\n",
        f"{_format_float(columns[0])} {_format_float(control_y[1][0])}\n",
        f"{_format_float(columns[-1])} {_format_float(control_y[1][-1])}\n",
    ]


def _axis_index(value, values, tol):
    matches = [
        index
        for index, candidate in enumerate(values)
        if abs(float(value) - candidate) <= tol
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def _validate_bootstrap_control_lattice(
    control_points,
    degree,
    columns,
    y_rows,
    z_planes,
    control_format,
):
    ni = int(degree["i"]) + 1
    nj = int(degree["j"]) + 1
    nk = int(degree["k"]) + 1 if degree["k"] is not None else len(z_planes)
    expected_count = ni * nj * nk
    if len(control_points) != expected_count:
        raise FFDBoxSplitError(
            "Bootstrap FFD_CONTROL_POINTS are incomplete: "
            f"got {len(control_points)}, expected {expected_count}"
        )
    if len(columns) != ni or len(y_rows) != nj or len(z_planes) != nk:
        raise FFDBoxSplitError(
            "Bootstrap FFD_CONTROL_POINTS do not define the degree-sized "
            "tensor-product lattice"
        )
    if control_format in (INDEXED_2D, COORDS_ONLY_2D) and nk != 1:
        raise FFDBoxSplitError(
            f"Control point format {control_format} cannot preserve {nk} k-planes"
        )

    scale = max(
        1.0,
        max(abs(value) for value in columns + y_rows + z_planes),
    )
    tol = 1.0e-10 * scale
    occupied = set()
    for point in control_points:
        x, y, z = point["coords"]
        if control_format == INDEXED_3D:
            i, j, k = point["i"], point["j"], point["k"]
        elif control_format in (INDEXED_2D_WITH_Z, INDEXED_2D):
            i, j = point["i"], point["j"]
            k = _axis_index(z, z_planes, tol)
        else:
            i = _axis_index(x, columns, tol)
            j = _axis_index(y, y_rows, tol)
            k = _axis_index(z, z_planes, tol)

        if i is None or j is None or k is None:
            raise FFDBoxSplitError(
                "Bootstrap FFD_CONTROL_POINTS contain a point outside the "
                "inferred tensor-product axes"
            )
        if not (0 <= i < ni and 0 <= j < nj and 0 <= k < nk):
            raise FFDBoxSplitError(
                f"Bootstrap control point index {(i, j, k)} is outside the FFD degree"
            )
        if (
            abs(x - columns[i]) > tol
            or abs(y - y_rows[j]) > tol
            or abs(z - z_planes[k]) > tol
        ):
            raise FFDBoxSplitError(
                f"Bootstrap control point {(i, j, k)} is not on a rectangular lattice"
            )
        if (i, j, k) in occupied:
            raise FFDBoxSplitError(
                f"Bootstrap FFD_CONTROL_POINTS duplicate lattice index {(i, j, k)}"
            )
        occupied.add((i, j, k))

    if len(occupied) != expected_count:
        raise FFDBoxSplitError("Bootstrap FFD_CONTROL_POINTS lattice is incomplete")


def _validate_corner_block_2d(corner_block):
    for line in corner_block["data"]:
        tokens = _split_tokens(line)
        if len(tokens) < 2:
            raise FFDBoxSplitError(
                f"Invalid 2D FFD corner point line: {line.rstrip()}"
            )
        try:
            float(tokens[0])
            float(tokens[1])
        except ValueError as exc:
            raise FFDBoxSplitError(
                f"Invalid 2D FFD corner point line: {line.rstrip()}"
            ) from exc


def _parse_bootstrap_surface_lines(
    surface_block,
    points,
    old_axes,
    marker_tag,
    reconstruction_tol,
):
    parsed = {}
    for line in surface_block["data"]:
        tokens = _split_tokens(line)
        if not tokens:
            raise FFDBoxSplitError("Bootstrap FFD_SURFACE_POINTS contains an empty line")
        if _as_int_if_possible(tokens[0]) is None:
            if _normalize_tag(tokens[0]).lower() != _normalize_tag(marker_tag).lower():
                raise FFDBoxSplitError(
                    "Bootstrap FFD_SURFACE_POINTS contains a marker other than "
                    f"{marker_tag!r}: {tokens[0]!r}"
                )
        best = None
        for token in tokens[:4]:
            point_id = _as_int_if_possible(token)
            if point_id is None or point_id not in points:
                continue
            candidate = _infer_surface_param_location(
                tokens,
                points[point_id],
                old_axes,
                2,
            )
            if candidate is None or candidate["point_id"] != point_id:
                continue
            if best is None or candidate["score"] < best["score"]:
                best = candidate

        if best is None:
            raise FFDBoxSplitError(
                "Could not parse bootstrap FFD_SURFACE_POINTS line: "
                f"{line.rstrip()}"
            )
        point_id = int(best["point_id"])
        if point_id in parsed:
            raise FFDBoxSplitError(
                f"Bootstrap FFD_SURFACE_POINTS contains duplicate point ID {point_id}"
            )
        old_error = float(best["score"][0])
        point_scale = max(
            1.0,
            abs(float(points[point_id][0])),
            abs(float(points[point_id][1])),
        )
        effective_tol = reconstruction_tol + 64.0 * math.ulp(point_scale)
        if old_error > effective_tol:
            raise FFDBoxSplitError(
                f"Bootstrap FFD_SURFACE_POINTS does not embed point {point_id}: "
                f"error={old_error:.6e}, tolerance={effective_tol:.6e}"
            )

        param_start = int(best["param_start"])
        param_count = int(best["param_count"])
        old_uvw = [
            float(tokens[param_start]),
            float(tokens[param_start + 1]),
            float(tokens[param_start + 2]) if param_count >= 3 else 0.0,
        ]
        parsed[point_id] = {
            "tokens": tokens,
            "param_start": param_start,
            "param_count": param_count,
            "old_uvw": old_uvw,
        }
    return parsed


def _parse_curved_control_lattice(control_points, degree, control_format):
    """Recover a structured two-row lattice without assuming constant y rows."""

    degree_i = degree.get("i")
    degree_j = degree.get("j")
    degree_k = degree.get("k")
    if degree_i is None or degree_j != 1:
        raise FFDBoxSplitError(
            "A progressive dual FFD box requires FFD_DEGREE_J=1"
        )

    ni = int(degree_i) + 1
    nj = 2
    ncontrol = len(control_points)
    if degree_k is None:
        if ncontrol % (ni * nj) != 0:
            raise FFDBoxSplitError(
                "FFD_CONTROL_POINTS count is inconsistent with FFD_DEGREE_I/J"
            )
        nk = ncontrol // (ni * nj)
    else:
        nk = int(degree_k) + 1
    if nk < 1 or ni * nj * nk != ncontrol:
        raise FFDBoxSplitError(
            "FFD_CONTROL_POINTS do not define a complete two-row lattice"
        )

    indexed = control_format in (
        INDEXED_3D,
        INDEXED_2D_WITH_Z,
        INDEXED_2D,
    )
    lattice = {}

    if indexed:
        z_to_k = None
        if any(point["k"] is None for point in control_points) and nk > 1:
            unique_z = []
            for point in sorted(control_points, key=lambda item: item["coords"][2]):
                value = float(point["coords"][2])
                if not unique_z or abs(value - unique_z[-1]) > 1.0e-12:
                    unique_z.append(value)
            if len(unique_z) != nk:
                raise FFDBoxSplitError(
                    "Could not infer k planes from indexed FFD control points"
                )
            z_to_k = unique_z

        for point in control_points:
            i = point["i"]
            j = point["j"]
            if i is None or j is None:
                raise FFDBoxSplitError(
                    "Indexed FFD control points require i and j indices"
                )
            if point["k"] is not None:
                k = int(point["k"])
            elif nk == 1:
                k = 0
            else:
                z = float(point["coords"][2])
                matches = [
                    index
                    for index, candidate in enumerate(z_to_k)
                    if abs(z - candidate) <= 1.0e-12
                ]
                if len(matches) != 1:
                    raise FFDBoxSplitError(
                        "Could not assign an indexed FFD control point to a k plane"
                    )
                k = matches[0]
            key = (int(i), int(j), int(k))
            if key in lattice:
                raise FFDBoxSplitError(
                    f"Duplicate FFD control point index {key}"
                )
            lattice[key] = point["coords"]
    else:
        # Coordinate-only blocks written by this module use i-fastest order,
        # followed by j and k.
        for flat_index, point in enumerate(control_points):
            i = flat_index % ni
            j = (flat_index // ni) % nj
            k = flat_index // (ni * nj)
            lattice[(i, j, k)] = point["coords"]

    expected = {
        (i, j, k)
        for i in range(ni)
        for j in range(nj)
        for k in range(nk)
    }
    if set(lattice) != expected:
        missing = sorted(expected - set(lattice))
        extra = sorted(set(lattice) - expected)
        raise FFDBoxSplitError(
            "FFD control lattice indices are incomplete: "
            f"missing={missing[:8]}, extra={extra[:8]}"
        )

    def _consistent_average(values, label):
        values = [float(value) for value in values]
        mean = sum(values) / float(len(values))
        scale = max(1.0, abs(mean), *(abs(value) for value in values))
        if max(abs(value - mean) for value in values) > 1.0e-10 * scale:
            raise FFDBoxSplitError(
                f"FFD control lattice is not structured along {label}"
            )
        return mean

    columns = []
    for i in range(ni):
        columns.append(
            _consistent_average(
                [lattice[(i, j, k)][0] for j in range(nj) for k in range(nk)],
                f"column i={i}",
            )
        )

    control_y = [[], []]
    for j in range(nj):
        for i in range(ni):
            control_y[j].append(
                _consistent_average(
                    [lattice[(i, j, k)][1] for k in range(nk)],
                    f"row j={j}, column i={i}",
                )
            )

    z_planes = []
    for k in range(nk):
        z_planes.append(
            _consistent_average(
                [lattice[(i, j, k)][2] for i in range(ni) for j in range(nj)],
                f"plane k={k}",
            )
        )

    for left, right in zip(columns[:-1], columns[1:]):
        if right - left <= 1.0e-12:
            raise FFDBoxSplitError(
                "Progressive dual FFD x-columns must be strictly increasing"
            )

    return columns, control_y, z_planes


def _infer_curved_surface_param_location(
    tokens,
    point_coords,
    columns,
    control_y,
    z_planes,
    blending_spec,
):
    best = None
    candidate_point_indices = []
    for index, token in enumerate(tokens[:4]):
        point_id = _as_int_if_possible(token)
        if point_id is not None:
            candidate_point_indices.append((index, point_id))

    for point_token_index, point_id in candidate_point_indices:
        for start in range(point_token_index + 1, len(tokens)):
            for count in (3, 2):
                if start + count > len(tokens):
                    continue
                try:
                    params = [float(value) for value in tokens[start:start + count]]
                except Exception:
                    continue
                if any(value < 0.0 or value > 1.0 for value in params):
                    continue
                uvw = [params[0], params[1], params[2] if count >= 3 else 0.0]
                mapped = _eval_curved_two_row_ffd(
                    columns,
                    control_y,
                    z_planes,
                    uvw,
                    blending_spec,
                )
                error = math.hypot(
                    mapped[0] - float(point_coords[0]),
                    mapped[1] - float(point_coords[1]),
                )
                score = (error, -count, start)
                if best is None or score < best["score"]:
                    best = {
                        "point_id": point_id,
                        "point_token_index": point_token_index,
                        "param_start": start,
                        "param_count": count,
                        "score": score,
                    }
    return best


def _parse_curved_surface_lines(
    surface_block,
    points,
    columns,
    control_y,
    z_planes,
    marker_tag,
    reconstruction_tol,
    blending_spec,
):
    parsed = {}
    for line in surface_block["data"]:
        tokens = _split_tokens(line)
        if not tokens:
            raise FFDBoxSplitError("FFD_SURFACE_POINTS contains an empty line")
        if _as_int_if_possible(tokens[0]) is None:
            if _normalize_tag(tokens[0]).lower() != _normalize_tag(marker_tag).lower():
                raise FFDBoxSplitError(
                    "FFD_SURFACE_POINTS contains a marker other than "
                    f"{marker_tag!r}: {tokens[0]!r}"
                )

        best = None
        for token in tokens[:4]:
            point_id = _as_int_if_possible(token)
            if point_id is None or point_id not in points:
                continue
            candidate = _infer_curved_surface_param_location(
                tokens,
                points[point_id],
                columns,
                control_y,
                z_planes,
                blending_spec,
            )
            if candidate is None or candidate["point_id"] != point_id:
                continue
            if best is None or candidate["score"] < best["score"]:
                best = candidate

        if best is None:
            raise FFDBoxSplitError(
                "Could not parse curved FFD_SURFACE_POINTS line: "
                f"{line.rstrip()}"
            )
        point_id = int(best["point_id"])
        if point_id in parsed:
            raise FFDBoxSplitError(
                f"FFD_SURFACE_POINTS contains duplicate point ID {point_id}"
            )
        old_error = float(best["score"][0])
        point_scale = max(
            1.0,
            abs(float(points[point_id][0])),
            abs(float(points[point_id][1])),
        )
        effective_tol = reconstruction_tol + 64.0 * math.ulp(point_scale)
        if old_error > effective_tol:
            raise FFDBoxSplitError(
                f"Existing curved FFD box does not embed point {point_id}: "
                f"error={old_error:.6e}, tolerance={effective_tol:.6e}"
            )

        param_start = int(best["param_start"])
        param_count = int(best["param_count"])
        parsed[point_id] = {
            "tokens": tokens,
            "param_start": param_start,
            "param_count": param_count,
            "old_uvw": [
                float(tokens[param_start]),
                float(tokens[param_start + 1]),
                float(tokens[param_start + 2]) if param_count >= 3 else 0.0,
            ],
        }
    return parsed


def _checked_unit_parameter(value, point_id, box_tag, name):
    value = float(value)
    if not math.isfinite(value):
        raise FFDBoxSplitError(
            f"Point {point_id} has non-finite {name} in {box_tag}"
        )
    if value < 0.0 or value > 1.0:
        raise FFDBoxSplitError(
            f"Point {point_id} has {name}={value:.16g} outside {box_tag} "
            "parameter range [0,1]"
        )
    return value


def _eval_curved_two_row_ffd(columns, control_y, z_planes, uvw, blending_spec=None):
    u, v, w = [float(value) for value in uvw]
    spec = blending_spec or FFDBlendingSpec(BEZIER)
    x = evaluate_curve(columns, u, spec, axis=0)
    y0 = evaluate_curve(control_y[0], u, spec, axis=0)
    y1 = evaluate_curve(control_y[1], u, spec, axis=0)
    y = evaluate_curve([y0, y1], v, spec, axis=1)
    z = evaluate_curve(z_planes, w, spec, axis=2) if z_planes else 0.0
    return [x, y, z]


def _reembed_side(
    side,
    box_tag,
    ordered_node_ids,
    side_by_node,
    edge_node_ids,
    points,
    surface_templates,
    columns,
    control_y,
    z_planes,
    chord,
    blending_spec=None,
):
    reconstruction_tol = 1.0e-10 * max(1.0, float(chord))
    lines = []
    diagnostics = []
    max_error = 0.0

    x_min = min(float(columns[0]), float(columns[-1]))
    x_max = max(float(columns[0]), float(columns[-1]))
    for point_id in ordered_node_ids:
        if point_id in edge_node_ids or side_by_node[point_id] != side:
            continue

        template = surface_templates[point_id]
        point = points[point_id]
        x = float(point[0])
        y = float(point[1])
        if x < x_min or x > x_max:
            raise FFDBoxSplitError(
                f"Point {point_id} lies outside {box_tag} in x: "
                f"x={x:.16g}, box=[{x_min:.16g},{x_max:.16g}]"
            )

        spec = blending_spec or FFDBlendingSpec(BEZIER)
        try:
            u_raw = invert_monotone_curve(columns, x, spec, axis=0)
        except ValueError as exc:
            raise FFDBoxSplitError(
                f"Could not invert x for point {point_id} in {box_tag}: {exc}"
            ) from exc
        u = _checked_unit_parameter(u_raw, point_id, box_tag, "u")
        y0 = evaluate_curve(control_y[0], u, spec, axis=0)
        y1 = evaluate_curve(control_y[1], u, spec, axis=0)
        denominator = y1 - y0
        if denominator <= 1.0e-14 * max(1.0, float(chord)):
            raise FFDBoxSplitError(
                f"Degenerate or crossed rows in {box_tag} at point {point_id}: "
                f"Y0={y0:.16g}, Y1={y1:.16g}"
            )

        v = (y - y0) / denominator
        v = _checked_unit_parameter(v, point_id, box_tag, "v")

        if template["param_count"] >= 3:
            w = _checked_unit_parameter(
                template["old_uvw"][2], point_id, box_tag, "w"
            )
        else:
            w = 0.0
        uvw = [u, v, w]
        mapped = _eval_curved_two_row_ffd(
            columns, control_y, z_planes, uvw, spec
        )
        error = math.hypot(mapped[0] - x, mapped[1] - y)
        if error > reconstruction_tol:
            raise FFDBoxSplitError(
                f"Re-embedding error for point {point_id} in {box_tag}: "
                f"error={error:.6e}, tolerance={reconstruction_tol:.6e}"
            )
        max_error = max(max_error, error)

        lines.append(
            _format_surface_line(
                template["tokens"],
                template["param_start"],
                template["param_count"],
                uvw,
            )
        )
        diagnostics.append(
            {
                "point_id": point_id,
                "side": side,
                "box": box_tag,
                "x": x,
                "y": y,
                "u": u,
                "v": v,
                "w": w,
                "reconstruction_error": error,
                "status": "EMBEDDED",
            }
        )

    return lines, diagnostics, max_error


def _build_box_block(
    tag,
    degree_i,
    degree_k,
    columns,
    control_y,
    z_planes,
    coord_dim,
    control_format,
    surface_lines,
    blending_spec=None,
):
    spec = blending_spec or FFDBlendingSpec(BEZIER)
    control_lines = _build_curved_control_point_lines(
        columns,
        control_y,
        z_planes,
        coord_dim,
        control_format,
    )
    corner_lines = _build_curved_corner_lines(columns, control_y)

    lines = [
        _format_key_value("FFD_TAG", tag),
        _format_key_value("FFD_LEVEL", 0),
        _format_key_value("FFD_DEGREE_I", degree_i),
        _format_key_value("FFD_DEGREE_J", 1),
    ]
    if degree_k is not None:
        lines.append(_format_key_value("FFD_DEGREE_K", degree_k))
    lines.extend(
        [
            _format_key_value("FFD_BLENDING", spec.kind),
            _format_key_value("FFD_PARENTS", 0),
            _format_key_value("FFD_CHILDREN", 0),
            _format_key_value("FFD_CORNER_POINTS", len(corner_lines)),
        ]
    )
    if spec.kind != BEZIER:
        insert_at = 6 if degree_k is not None else 5
        order_lines = [
            _format_key_value("BSPLINE_ORDER_I", spec.orders[0]),
            _format_key_value("BSPLINE_ORDER_J", spec.orders[1]),
        ]
        if degree_k is not None:
            order_lines.append(_format_key_value("BSPLINE_ORDER_K", spec.orders[2]))
        lines[insert_at:insert_at] = order_lines
    lines.extend(corner_lines)
    lines.append(_format_key_value("FFD_CONTROL_POINTS", len(control_lines)))
    lines.extend(control_lines)
    lines.append(_format_key_value("FFD_SURFACE_POINTS", len(surface_lines)))
    lines.extend(surface_lines)
    return lines


def _parse_existing_dual_box(lines, tag):
    try:
        block_start, block_end = _find_tagged_ffd_block(lines, tag)
    except FFDMeshError as exc:
        raise FFDBoxSplitError(str(exc)) from exc

    level_line = _find_key_line(lines, block_start, block_end, "FFD_LEVEL")
    if level_line is None or _parse_int_value(lines[level_line]) != 0:
        raise FFDBoxSplitError(f"FFD box {tag!r} must have FFD_LEVEL=0")
    _parse_zero_relation_count(lines, block_start, block_end, "FFD_PARENTS")
    _parse_zero_relation_count(lines, block_start, block_end, "FFD_CHILDREN")

    degree = _parse_degree(lines, block_start, block_end)
    if degree["i"] is None or degree["j"] != 1:
        raise FFDBoxSplitError(
            f"FFD box {tag!r} must define FFD_DEGREE_I and FFD_DEGREE_J=1"
        )

    try:
        corner_block = _parse_count_block(
            lines, block_start, block_end, "FFD_CORNER_POINTS"
        )
        control_block = _parse_count_block(
            lines, block_start, block_end, "FFD_CONTROL_POINTS"
        )
        surface_block = _parse_count_block(
            lines, block_start, block_end, "FFD_SURFACE_POINTS"
        )
    except FFDMeshError as exc:
        raise FFDBoxSplitError(str(exc)) from exc

    if corner_block is None or corner_block["count"] != 4:
        raise FFDBoxSplitError(f"FFD box {tag!r} requires four corner points")
    _validate_corner_block_2d(corner_block)
    if control_block is None or control_block["count"] <= 0:
        raise FFDBoxSplitError(f"FFD box {tag!r} has no control points")
    if surface_block is None or surface_block["count"] <= 0:
        raise FFDBoxSplitError(f"FFD box {tag!r} has no surface points")

    try:
        control_points, coord_dim, control_format = _parse_control_points(
            control_block
        )
    except FFDMeshError as exc:
        raise FFDBoxSplitError(str(exc)) from exc
    columns, control_y, z_planes = _parse_curved_control_lattice(
        control_points,
        degree,
        control_format,
    )
    try:
        blending_spec = _parse_blending_spec(
            lines,
            block_start,
            block_end,
            control_counts=(len(columns), len(control_y), len(z_planes)),
            dual_2d=True,
        )
    except ValueError as exc:
        raise FFDBoxSplitError(str(exc)) from exc
    return {
        "tag": _normalize_tag(tag),
        "block_start": block_start,
        "block_end": block_end,
        "degree": degree,
        "corner_block": corner_block,
        "control_block": control_block,
        "surface_block": surface_block,
        "columns": columns,
        "control_y": control_y,
        "z_planes": z_planes,
        "coord_dim": coord_dim,
        "control_format": control_format,
        "blending_spec": blending_spec,
    }


def read_dual_ffd_box_specs(mesh_in, upper_tag, lower_tag):
    """Read and validate the blending metadata of an existing dual FFD mesh."""

    with open(mesh_in, "r") as fp:
        lines = fp.readlines()

    nbox_line = _find_key_line(lines, 0, len(lines), "FFD_NBOX")
    if nbox_line is None or _parse_int_value(lines[nbox_line]) != 2:
        raise FFDBoxSplitError("Progressive dual mesh requires FFD_NBOX=2")
    nlevel_line = _find_key_line(lines, 0, len(lines), "FFD_NLEVEL")
    if nlevel_line is None or _parse_int_value(lines[nlevel_line]) != 1:
        raise FFDBoxSplitError("Progressive dual mesh requires FFD_NLEVEL=1")

    tag_lines = [
        index for index, line in enumerate(lines) if _line_key(line) == "FFD_TAG"
    ]
    if len(tag_lines) != 2:
        raise FFDBoxSplitError(
            f"Progressive dual mesh requires exactly two boxes, found {len(tag_lines)}"
        )

    upper_box = _parse_existing_dual_box(lines, upper_tag)
    lower_box = _parse_existing_dual_box(lines, lower_tag)
    if {upper_box["block_start"], lower_box["block_start"]} != set(tag_lines):
        raise FFDBoxSplitError("Unexpected FFD box tag found in dual mesh")
    upper_spec = upper_box["blending_spec"]
    lower_spec = lower_box["blending_spec"]
    if upper_spec.kind != lower_spec.kind:
        raise FFDBoxSplitError(
            "Upper and lower FFD boxes must use the same blending"
        )
    if upper_spec.kind == BSPLINE_UNIFORM and upper_spec.orders != lower_spec.orders:
        raise FFDBoxSplitError(
            "Upper and lower B-spline FFD boxes must use the same order"
        )

    spec = upper_spec
    return {
        "upper_columns": list(upper_box["columns"]),
        "lower_columns": list(lower_box["columns"]),
        "blending": spec.kind,
        "bspline_orders": list(spec.orders),
    }


def rewrite_dual_ffd_boxes_with_columns_and_reembed(
    mesh_in,
    mesh_out,
    *,
    marker,
    upper_tag,
    lower_tag,
    upper_columns,
    lower_columns,
    upper_offset_chord,
    lower_offset_chord,
    diagnostics_csv=None,
    overwrite=False,
):
    """Rewrite two curved independent boxes on the current physical surface."""

    _validate_output_path(mesh_in, mesh_out, overwrite)
    upper_offset_chord = float(upper_offset_chord)
    lower_offset_chord = float(lower_offset_chord)
    if (
        not math.isfinite(upper_offset_chord)
        or not math.isfinite(lower_offset_chord)
        or upper_offset_chord <= 0.0
        or lower_offset_chord <= 0.0
    ):
        raise FFDBoxSplitError("Upper and lower chord offsets must be positive")

    upper_tag = _normalize_tag(upper_tag)
    lower_tag = _normalize_tag(lower_tag)
    if not upper_tag or not lower_tag or upper_tag == lower_tag:
        raise FFDBoxSplitError("Upper and lower FFD tags must be non-empty and distinct")

    upper_columns = [float(value) for value in upper_columns]
    lower_columns = [float(value) for value in lower_columns]
    for side, columns in (("upper", upper_columns), ("lower", lower_columns)):
        if len(columns) < 3:
            raise FFDBoxSplitError(
                f"The {side} dual FFD box requires two boundaries and one active column"
            )
        if not all(math.isfinite(value) for value in columns):
            raise FFDBoxSplitError(f"The {side} FFD columns must be finite")
        for left, right in zip(columns[:-1], columns[1:]):
            if right - left <= 1.0e-12:
                raise FFDBoxSplitError(
                    f"The {side} FFD columns must be strictly increasing"
                )

    with open(mesh_in, "r") as fp:
        lines = fp.readlines()

    nbox_line = _find_key_line(lines, 0, len(lines), "FFD_NBOX")
    if nbox_line is None or _parse_int_value(lines[nbox_line]) != 2:
        raise FFDBoxSplitError("Progressive dual rewrite requires FFD_NBOX=2")
    nlevel_line = _find_key_line(lines, 0, len(lines), "FFD_NLEVEL")
    if nlevel_line is None or _parse_int_value(lines[nlevel_line]) != 1:
        raise FFDBoxSplitError("Progressive dual rewrite requires FFD_NLEVEL=1")
    tag_lines = [index for index, line in enumerate(lines) if _line_key(line) == "FFD_TAG"]
    if len(tag_lines) != 2:
        raise FFDBoxSplitError(
            f"Progressive dual rewrite requires exactly two boxes, found {len(tag_lines)}"
        )

    upper_box = _parse_existing_dual_box(lines, upper_tag)
    lower_box = _parse_existing_dual_box(lines, lower_tag)
    if {upper_box["block_start"], lower_box["block_start"]} != set(tag_lines):
        raise FFDBoxSplitError("Unexpected FFD box tag found in dual mesh")
    upper_spec = upper_box["blending_spec"]
    lower_spec = lower_box["blending_spec"]
    if upper_spec.kind != lower_spec.kind:
        raise FFDBoxSplitError("Upper and lower FFD boxes must use the same blending")
    if upper_spec.kind == BSPLINE_UNIFORM and upper_spec.orders != lower_spec.orders:
        raise FFDBoxSplitError(
            "Upper and lower B-spline FFD boxes must use the same order"
        )
    blending_spec = upper_spec
    try:
        validate_blending_spec(
            blending_spec,
            control_counts=(len(upper_columns), 2, len(upper_box["z_planes"])),
            dual_2d=True,
        )
        validate_blending_spec(
            blending_spec,
            control_counts=(len(lower_columns), 2, len(lower_box["z_planes"])),
            dual_2d=True,
        )
    except ValueError as exc:
        raise FFDBoxSplitError(str(exc)) from exc

    try:
        mesh = read_su2_mesh(mesh_in)
        marker_tag, ordered_node_ids, closed = extract_marker_nodes(mesh, marker)
    except BSplineDefError as exc:
        raise FFDBoxSplitError(str(exc)) from exc
    if mesh["ndime"] != 2:
        raise FFDBoxSplitError("Progressive dual FFD supports only NDIME=2")
    if not closed:
        raise FFDBoxSplitError("Progressive dual FFD requires a closed marker")

    try:
        x_le, x_te, chord = infer_chord(
            mesh["points"],
            ordered_node_ids,
            {"mode": "auto", "x_le": None, "x_te": None},
        )
    except BSplineDefError as exc:
        raise FFDBoxSplitError(str(exc)) from exc
    edge_tol = 1.0e-10 * max(1.0, float(chord))

    for side, columns in (("upper", upper_columns), ("lower", lower_columns)):
        if columns[0] > x_le + edge_tol or columns[-1] < x_te - edge_tol:
            raise FFDBoxSplitError(
                f"The {side} FFD columns do not enclose the chord: "
                f"columns=[{columns[0]:.16g},{columns[-1]:.16g}], "
                f"chord=[{x_le:.16g},{x_te:.16g}]"
            )

    x_over_c = [
        (float(mesh["points"][node_id][0]) - x_le) / chord
        for node_id in ordered_node_ids
    ]
    y_values = [float(mesh["points"][node_id][1]) for node_id in ordered_node_ids]
    try:
        sides = classify_sides(
            ordered_node_ids,
            x_over_c,
            y_values,
            closed=True,
        )
    except BSplineDefError as exc:
        raise FFDBoxSplitError(str(exc)) from exc
    side_by_node = dict(zip(ordered_node_ids, sides))

    edge_node_ids = {
        node_id
        for node_id in ordered_node_ids
        if (
            abs(float(mesh["points"][node_id][0]) - x_le) <= edge_tol
            or abs(float(mesh["points"][node_id][0]) - x_te) <= edge_tol
        )
    }
    if not any(
        abs(float(mesh["points"][node_id][0]) - x_le) <= edge_tol
        for node_id in edge_node_ids
    ) or not any(
        abs(float(mesh["points"][node_id][0]) - x_te) <= edge_tol
        for node_id in edge_node_ids
    ):
        raise FFDBoxSplitError("Could not identify both LE and TE marker nodes")

    upper_samples = [
        (mesh["points"][node_id][0], mesh["points"][node_id][1])
        for node_id in ordered_node_ids
        if side_by_node[node_id] == "upper"
    ]
    lower_samples = [
        (mesh["points"][node_id][0], mesh["points"][node_id][1])
        for node_id in ordered_node_ids
        if side_by_node[node_id] == "lower"
    ]
    edge_points = [
        (mesh["points"][node_id][0], mesh["points"][node_id][1])
        for node_id in edge_node_ids
    ]
    for edge_x in (x_le, x_te):
        upper_samples = _set_side_endpoint(
            upper_samples, edge_points, edge_x, "upper", edge_tol
        )
        lower_samples = _set_side_endpoint(
            lower_samples, edge_points, edge_x, "lower", edge_tol
        )
    upper_y = _make_linear_interpolant(upper_samples, edge_tol)
    lower_y = _make_linear_interpolant(lower_samples, edge_tol)

    def _control_rows(columns):
        camber_values = []
        upper_outer_values = []
        lower_outer_values = []
        for x in columns:
            yu = float(upper_y(x))
            yl = float(lower_y(x))
            if yu < yl - edge_tol:
                raise FFDBoxSplitError(
                    f"Upper/lower interpolants cross at x={x:.16g}: "
                    f"yu={yu:.16g}, yl={yl:.16g}"
                )
            camber = 0.5 * (yu + yl)
            camber_values.append(camber)
            upper_outer_values.append(yu + upper_offset_chord * chord)
            lower_outer_values.append(yl - lower_offset_chord * chord)
        return camber_values, upper_outer_values, lower_outer_values

    upper_camber, upper_outer, _ = _control_rows(upper_columns)
    lower_camber, _, lower_outer = _control_rows(lower_columns)
    upper_control_y = [upper_camber, upper_outer]
    lower_control_y = [lower_outer, lower_camber]

    reconstruction_tol = 1.0e-10 * max(1.0, float(chord))
    upper_templates = _parse_curved_surface_lines(
        upper_box["surface_block"],
        mesh["points"],
        upper_box["columns"],
        upper_box["control_y"],
        upper_box["z_planes"],
        marker_tag,
        reconstruction_tol,
        upper_box["blending_spec"],
    )
    lower_templates = _parse_curved_surface_lines(
        lower_box["surface_block"],
        mesh["points"],
        lower_box["columns"],
        lower_box["control_y"],
        lower_box["z_planes"],
        marker_tag,
        reconstruction_tol,
        lower_box["blending_spec"],
    )

    expected_upper = {
        node_id
        for node_id in ordered_node_ids
        if node_id not in edge_node_ids and side_by_node[node_id] == "upper"
    }
    expected_lower = {
        node_id
        for node_id in ordered_node_ids
        if node_id not in edge_node_ids and side_by_node[node_id] == "lower"
    }
    if set(upper_templates) != expected_upper:
        raise FFDBoxSplitError(
            "UPPER_BOX surface-point membership does not match the upper branch"
        )
    if set(lower_templates) != expected_lower:
        raise FFDBoxSplitError(
            "LOWER_BOX surface-point membership does not match the lower branch"
        )

    upper_surface_lines, upper_diagnostics, upper_max_error = _reembed_side(
        "upper",
        upper_tag,
        ordered_node_ids,
        side_by_node,
        edge_node_ids,
        mesh["points"],
        upper_templates,
        upper_columns,
        upper_control_y,
        upper_box["z_planes"],
        chord,
        upper_box["blending_spec"],
    )
    lower_surface_lines, lower_diagnostics, lower_max_error = _reembed_side(
        "lower",
        lower_tag,
        ordered_node_ids,
        side_by_node,
        edge_node_ids,
        mesh["points"],
        lower_templates,
        lower_columns,
        lower_control_y,
        lower_box["z_planes"],
        chord,
        lower_box["blending_spec"],
    )

    upper_block = _build_box_block(
        upper_tag,
        len(upper_columns) - 1,
        upper_box["degree"]["k"],
        upper_columns,
        upper_control_y,
        upper_box["z_planes"],
        upper_box["coord_dim"],
        upper_box["control_format"],
        upper_surface_lines,
        blending_spec,
    )
    lower_block = _build_box_block(
        lower_tag,
        len(lower_columns) - 1,
        lower_box["degree"]["k"],
        lower_columns,
        lower_control_y,
        lower_box["z_planes"],
        lower_box["coord_dim"],
        lower_box["control_format"],
        lower_surface_lines,
        blending_spec,
    )

    first_start = min(upper_box["block_start"], lower_box["block_start"])
    last_end = max(
        upper_box["surface_block"]["data_end"],
        lower_box["surface_block"]["data_end"],
    )
    output_lines = list(lines[:first_start])
    _replace_key_line(output_lines, "FFD_NBOX", 2)
    output_lines.extend(upper_block)
    output_lines.extend(lower_block)
    output_lines.extend(lines[last_end:])

    diagnostics = upper_diagnostics + lower_diagnostics
    for point_id in edge_node_ids:
        point = mesh["points"][point_id]
        diagnostics.append(
            {
                "point_id": point_id,
                "side": side_by_node[point_id],
                "box": "",
                "x": float(point[0]),
                "y": float(point[1]),
                "u": "",
                "v": "",
                "w": "",
                "reconstruction_error": 0.0,
                "status": "FIXED_EDGE",
            }
        )
    diagnostics.sort(key=lambda row: int(row["point_id"]))

    diagnostics_abs = None
    if diagnostics_csv is not False:
        if diagnostics_csv is None:
            stem, _ = os.path.splitext(mesh_out)
            diagnostics_csv = stem + "_dual_ffd_diagnostics.csv"
        diagnostics_abs = os.path.abspath(diagnostics_csv)
        if diagnostics_abs in (os.path.abspath(mesh_in), os.path.abspath(mesh_out)):
            raise FFDBoxSplitError(
                "Diagnostics CSV path must differ from mesh input/output paths"
            )
        if os.path.exists(diagnostics_csv) and not overwrite:
            raise FFDBoxSplitError(
                f"Diagnostics CSV already exists: {diagnostics_csv}"
            )

    _atomic_write_lines(mesh_out, output_lines)
    if diagnostics_csv is not False:
        _atomic_write_csv(
            diagnostics_csv,
            [
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
            ],
            diagnostics,
        )

    summary = {
        "mesh_in": os.path.abspath(mesh_in),
        "mesh_out": os.path.abspath(mesh_out),
        "diagnostics_csv": diagnostics_abs,
        "marker": marker_tag,
        "upper_tag": upper_tag,
        "lower_tag": lower_tag,
        "x_le": float(x_le),
        "x_te": float(x_te),
        "chord": float(chord),
        "upper_columns": list(upper_columns),
        "lower_columns": list(lower_columns),
        "upper_column_index_by_x": {
            float(x): index for index, x in enumerate(upper_columns)
        },
        "lower_column_index_by_x": {
            float(x): index for index, x in enumerate(lower_columns)
        },
        "upper_surface_points": len(upper_surface_lines),
        "lower_surface_points": len(lower_surface_lines),
        "fixed_edge_points": len(edge_node_ids),
        "upper_max_reembedding_error": upper_max_error,
        "lower_max_reembedding_error": lower_max_error,
        "blending": blending_spec.kind,
        "bspline_orders": list(blending_spec.orders),
    }
    print(
        "[PROGRESSIVE_FFD_DUAL] Rewritten | "
        f"upper_columns={len(upper_columns)} "
        f"lower_columns={len(lower_columns)} "
        f"max_error={max(upper_max_error, lower_max_error):.6e}"
    )
    return summary


def split_bootstrap_ffd_box(
    mesh_in,
    mesh_out,
    *,
    bootstrap_tag,
    marker,
    upper_offset_chord,
    lower_offset_chord,
    upper_tag="UPPER_BOX",
    lower_tag="LOWER_BOX",
    x_le=None,
    x_te=None,
    diagnostics_csv=None,
    overwrite=False,
    output_blending=BEZIER,
    bspline_orders=(2, 2, 2),
):
    """Replace one bootstrap FFD box with independent curved upper/lower boxes."""

    _validate_output_path(mesh_in, mesh_out, overwrite)
    upper_offset_chord = float(upper_offset_chord)
    lower_offset_chord = float(lower_offset_chord)
    if (
        not math.isfinite(upper_offset_chord)
        or not math.isfinite(lower_offset_chord)
        or upper_offset_chord <= 0.0
        or lower_offset_chord <= 0.0
    ):
        raise FFDBoxSplitError("Upper and lower chord offsets must be positive")
    upper_tag = _normalize_tag(upper_tag)
    lower_tag = _normalize_tag(lower_tag)
    if not upper_tag or not lower_tag:
        raise FFDBoxSplitError("Upper and lower FFD tags must be non-empty")
    if upper_tag == lower_tag:
        raise FFDBoxSplitError("Upper and lower FFD tags must be different")
    if (x_le is None) != (x_te is None):
        raise FFDBoxSplitError("x_le and x_te must be provided together")
    try:
        output_spec = make_blending_spec(output_blending, bspline_orders)
    except ValueError as exc:
        raise FFDBoxSplitError(str(exc)) from exc

    with open(mesh_in, "r") as fp:
        lines = fp.readlines()

    ffd_tag_lines = [i for i, line in enumerate(lines) if _line_key(line) == "FFD_TAG"]
    if len(ffd_tag_lines) != 1:
        raise FFDBoxSplitError(
            f"V1 requires exactly one bootstrap FFD box, found {len(ffd_tag_lines)}"
        )
    nbox_line = _find_key_line(lines, 0, len(lines), "FFD_NBOX")
    if nbox_line is None or _parse_int_value(lines[nbox_line]) != 1:
        raise FFDBoxSplitError("V1 requires FFD_NBOX=1")
    nlevel_line = _find_key_line(lines, 0, len(lines), "FFD_NLEVEL")
    if nlevel_line is None or _parse_int_value(lines[nlevel_line]) != 1:
        raise FFDBoxSplitError("V1 requires FFD_NLEVEL=1")

    try:
        block_start, block_end = _find_tagged_ffd_block(lines, bootstrap_tag)
    except FFDMeshError as exc:
        raise FFDBoxSplitError(str(exc)) from exc
    if block_start != ffd_tag_lines[0]:
        raise FFDBoxSplitError(f"FFD_TAG={bootstrap_tag!r} is not the only input box")

    level_line = _find_key_line(lines, block_start, block_end, "FFD_LEVEL")
    if level_line is None or _parse_int_value(lines[level_line]) != 0:
        raise FFDBoxSplitError("Bootstrap box must have FFD_LEVEL=0")
    _parse_zero_relation_count(lines, block_start, block_end, "FFD_PARENTS")
    _parse_zero_relation_count(lines, block_start, block_end, "FFD_CHILDREN")

    blending_line = _find_key_line(lines, block_start, block_end, "FFD_BLENDING")
    if blending_line is None:
        raise FFDBoxSplitError("Bootstrap box is missing FFD_BLENDING")
    blending = _normalize_tag(_line_value(lines[blending_line])).upper()
    if blending != "BEZIER":
        raise FFDBoxSplitError(
            f"V1 supports only FFD_BLENDING=BEZIER, got {blending!r}"
        )

    degree = _parse_degree(lines, block_start, block_end)
    if degree["i"] is None:
        raise FFDBoxSplitError("Bootstrap box is missing FFD_DEGREE_I")
    if degree["j"] != 1:
        raise FFDBoxSplitError(
            f"Bootstrap box must have FFD_DEGREE_J=1, got {degree['j']}"
        )

    try:
        corner_block = _parse_count_block(
            lines, block_start, block_end, "FFD_CORNER_POINTS"
        )
        control_block = _parse_count_block(
            lines, block_start, block_end, "FFD_CONTROL_POINTS"
        )
        surface_block = _parse_count_block(
            lines, block_start, block_end, "FFD_SURFACE_POINTS"
        )
    except FFDMeshError as exc:
        raise FFDBoxSplitError(str(exc)) from exc
    if corner_block is None or corner_block["count"] != 4:
        raise FFDBoxSplitError("V1 requires four 2D FFD_CORNER_POINTS")
    _validate_corner_block_2d(corner_block)
    if control_block is None or control_block["count"] <= 0:
        raise FFDBoxSplitError("Bootstrap FFD_CONTROL_POINTS are required")
    if surface_block is None or surface_block["count"] <= 0:
        raise FFDBoxSplitError("Bootstrap FFD_SURFACE_POINTS are required")

    try:
        control_points, coord_dim, control_format = _parse_control_points(control_block)
        columns, old_y_rows, z_planes = _infer_axes_from_control_points(
            control_points,
            degree,
        )
    except FFDMeshError as exc:
        raise FFDBoxSplitError(str(exc)) from exc
    columns = [float(value) for value in columns]
    old_y_rows = [float(value) for value in old_y_rows]
    z_planes = [float(value) for value in (z_planes or [0.0])]
    if not all(
        math.isfinite(value) for value in columns + old_y_rows + z_planes
    ):
        raise FFDBoxSplitError("Bootstrap FFD_CONTROL_POINTS must be finite")
    if len(columns) != int(degree["i"]) + 1:
        raise FFDBoxSplitError(
            "Bootstrap FFD column count does not match FFD_DEGREE_I"
        )
    for left, right in zip(columns[:-1], columns[1:]):
        if right - left <= 1.0e-12:
            raise FFDBoxSplitError("Bootstrap FFD x-columns must be strictly increasing")
    if len(old_y_rows) != 2:
        raise FFDBoxSplitError("Bootstrap FFD box must contain exactly two y rows")
    try:
        validate_blending_spec(
            output_spec,
            control_counts=(len(columns), 2, len(z_planes)),
            dual_2d=True,
        )
    except ValueError as exc:
        raise FFDBoxSplitError(str(exc)) from exc
    _validate_bootstrap_control_lattice(
        control_points,
        degree,
        columns,
        old_y_rows,
        z_planes,
        control_format,
    )

    try:
        mesh = read_su2_mesh(mesh_in)
    except BSplineDefError as exc:
        raise FFDBoxSplitError(str(exc)) from exc
    if mesh["ndime"] != 2:
        raise FFDBoxSplitError("V1 supports only NDIME=2 meshes")
    try:
        marker_tag, ordered_node_ids, closed = extract_marker_nodes(mesh, marker)
    except BSplineDefError as exc:
        raise FFDBoxSplitError(str(exc)) from exc
    if not closed:
        raise FFDBoxSplitError("V1 requires a closed airfoil marker")

    chord_spec = {
        "mode": "auto" if x_le is None else "explicit",
        "x_le": x_le,
        "x_te": x_te,
    }
    try:
        x_le, x_te, chord = infer_chord(
            mesh["points"],
            ordered_node_ids,
            chord_spec,
        )
    except BSplineDefError as exc:
        raise FFDBoxSplitError(str(exc)) from exc
    if not all(math.isfinite(value) for value in (x_le, x_te, chord)):
        raise FFDBoxSplitError("LE, TE, and chord must be finite")
    edge_tol = 1.0e-10 * max(1.0, float(chord))

    marker_x = [float(mesh["points"][node_id][0]) for node_id in ordered_node_ids]
    if min(marker_x) < x_le - edge_tol or max(marker_x) > x_te + edge_tol:
        raise FFDBoxSplitError(
            "Requested LE/TE do not enclose the complete airfoil marker: "
            f"marker=[{min(marker_x):.16g},{max(marker_x):.16g}], "
            f"requested=[{x_le:.16g},{x_te:.16g}]"
        )
    if columns[0] > x_le + edge_tol or columns[-1] < x_te - edge_tol:
        raise FFDBoxSplitError(
            "Bootstrap FFD x-columns do not enclose the complete chord: "
            f"columns=[{columns[0]:.16g},{columns[-1]:.16g}], "
            f"chord=[{x_le:.16g},{x_te:.16g}]"
        )

    x_over_c = [
        (float(mesh["points"][node_id][0]) - x_le) / chord
        for node_id in ordered_node_ids
    ]
    y_values = [float(mesh["points"][node_id][1]) for node_id in ordered_node_ids]
    try:
        sides = classify_sides(
            ordered_node_ids,
            x_over_c,
            y_values,
            closed=True,
        )
    except BSplineDefError as exc:
        raise FFDBoxSplitError(str(exc)) from exc
    side_by_node = dict(zip(ordered_node_ids, sides))

    le_node_ids = {
        node_id
        for node_id in ordered_node_ids
        if abs(float(mesh["points"][node_id][0]) - x_le) <= edge_tol
    }
    te_node_ids = {
        node_id
        for node_id in ordered_node_ids
        if abs(float(mesh["points"][node_id][0]) - x_te) <= edge_tol
    }
    if not le_node_ids or not te_node_ids:
        raise FFDBoxSplitError("Could not identify both LE and TE marker nodes")
    edge_node_ids = le_node_ids | te_node_ids

    upper_samples = [
        (mesh["points"][node_id][0], mesh["points"][node_id][1])
        for node_id in ordered_node_ids
        if side_by_node[node_id] == "upper"
    ]
    lower_samples = [
        (mesh["points"][node_id][0], mesh["points"][node_id][1])
        for node_id in ordered_node_ids
        if side_by_node[node_id] == "lower"
    ]
    edge_points = [
        (mesh["points"][node_id][0], mesh["points"][node_id][1])
        for node_id in edge_node_ids
    ]
    for edge_x in (x_le, x_te):
        upper_samples = _set_side_endpoint(
            upper_samples, edge_points, edge_x, "upper", edge_tol
        )
        lower_samples = _set_side_endpoint(
            lower_samples, edge_points, edge_x, "lower", edge_tol
        )

    upper_y = _make_linear_interpolant(upper_samples, edge_tol)
    lower_y = _make_linear_interpolant(lower_samples, edge_tol)
    delta_upper = upper_offset_chord * chord
    delta_lower = lower_offset_chord * chord

    camber_values = []
    upper_outer_values = []
    lower_outer_values = []
    for x in columns:
        yu = float(upper_y(x))
        yl = float(lower_y(x))
        if yu < yl - edge_tol:
            raise FFDBoxSplitError(
                f"Upper/lower interpolants cross at x={x:.16g}: "
                f"yu={yu:.16g}, yl={yl:.16g}"
            )
        camber = 0.5 * (yu + yl)
        camber_values.append(camber)
        upper_outer_values.append(yu + delta_upper)
        lower_outer_values.append(yl - delta_lower)

    upper_control_y = [camber_values, upper_outer_values]
    lower_control_y = [lower_outer_values, camber_values]

    old_axes = {
        "columns": columns,
        "y_rows": old_y_rows,
        "z_planes": z_planes,
    }
    surface_templates = _parse_bootstrap_surface_lines(
        surface_block,
        mesh["points"],
        old_axes,
        marker_tag,
        1.0e-10 * max(1.0, float(chord)),
    )
    marker_node_set = set(ordered_node_ids)
    bootstrap_node_set = set(surface_templates)
    if marker_node_set != bootstrap_node_set:
        missing = sorted(marker_node_set - bootstrap_node_set)
        extra = sorted(bootstrap_node_set - marker_node_set)
        raise FFDBoxSplitError(
            "Bootstrap FFD_SURFACE_POINTS must match the requested marker exactly: "
            f"missing={missing[:8]}, extra={extra[:8]}"
        )

    upper_surface_lines, upper_diagnostics, upper_max_error = _reembed_side(
        "upper",
        upper_tag,
        ordered_node_ids,
        side_by_node,
        edge_node_ids,
        mesh["points"],
        surface_templates,
        columns,
        upper_control_y,
        z_planes,
        chord,
        output_spec,
    )
    lower_surface_lines, lower_diagnostics, lower_max_error = _reembed_side(
        "lower",
        lower_tag,
        ordered_node_ids,
        side_by_node,
        edge_node_ids,
        mesh["points"],
        surface_templates,
        columns,
        lower_control_y,
        z_planes,
        chord,
        output_spec,
    )
    if not upper_surface_lines or not lower_surface_lines:
        raise FFDBoxSplitError("Both output boxes must contain surface points")

    upper_block = _build_box_block(
        upper_tag,
        int(degree["i"]),
        degree["k"],
        columns,
        upper_control_y,
        z_planes,
        coord_dim,
        control_format,
        upper_surface_lines,
        output_spec,
    )
    lower_block = _build_box_block(
        lower_tag,
        int(degree["i"]),
        degree["k"],
        columns,
        lower_control_y,
        z_planes,
        coord_dim,
        control_format,
        lower_surface_lines,
        output_spec,
    )

    output_lines = list(lines[:block_start])
    _replace_key_line(output_lines, "FFD_NBOX", 2)
    output_lines.extend(upper_block)
    output_lines.extend(lower_block)
    output_lines.extend(lines[surface_block["data_end"]:])

    diagnostics = upper_diagnostics + lower_diagnostics
    for point_id in ordered_node_ids:
        if point_id not in edge_node_ids:
            continue
        point = mesh["points"][point_id]
        diagnostics.append(
            {
                "point_id": point_id,
                "side": side_by_node[point_id],
                "box": "",
                "x": float(point[0]),
                "y": float(point[1]),
                "u": "",
                "v": "",
                "w": "",
                "reconstruction_error": 0.0,
                "status": "FIXED_EDGE",
            }
        )
    diagnostics.sort(key=lambda row: int(row["point_id"]))

    if diagnostics_csv is None:
        stem, _ = os.path.splitext(mesh_out)
        diagnostics_csv = stem + "_dual_ffd_diagnostics.csv"
    diagnostics_abs = os.path.abspath(diagnostics_csv)
    if diagnostics_abs in (os.path.abspath(mesh_in), os.path.abspath(mesh_out)):
        raise FFDBoxSplitError(
            "Diagnostics CSV path must differ from both input and output mesh paths"
        )
    if os.path.exists(diagnostics_csv) and not overwrite:
        raise FFDBoxSplitError(
            f"Diagnostics CSV already exists: {diagnostics_csv}; "
            "pass overwrite=True to replace it"
        )

    _atomic_write_lines(mesh_out, output_lines)
    _atomic_write_csv(
        diagnostics_csv,
        [
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
        ],
        diagnostics,
    )

    summary = {
        "mesh_in": os.path.abspath(mesh_in),
        "mesh_out": os.path.abspath(mesh_out),
        "diagnostics_csv": os.path.abspath(diagnostics_csv),
        "marker": marker_tag,
        "bootstrap_tag": _normalize_tag(bootstrap_tag),
        "upper_tag": _normalize_tag(upper_tag),
        "lower_tag": _normalize_tag(lower_tag),
        "x_le": float(x_le),
        "x_te": float(x_te),
        "chord": float(chord),
        "columns": columns,
        "upper_surface_points": len(upper_surface_lines),
        "lower_surface_points": len(lower_surface_lines),
        "fixed_edge_points": len(edge_node_ids),
        "upper_max_reembedding_error": upper_max_error,
        "lower_max_reembedding_error": lower_max_error,
        "blending": output_spec.kind,
        "bspline_orders": list(output_spec.orders),
    }
    print(
        "[PROGRESSIVE_FFD_SPLIT] Completed | "
        f"upper_points={summary['upper_surface_points']} "
        f"lower_points={summary['lower_surface_points']} "
        f"fixed_edges={summary['fixed_edge_points']} "
        f"max_error={max(upper_max_error, lower_max_error):.6e}"
    )
    print(f"[PROGRESSIVE_FFD_SPLIT] Mesh: {summary['mesh_out']}")
    print(f"[PROGRESSIVE_FFD_SPLIT] Diagnostics: {summary['diagnostics_csv']}")
    return summary
