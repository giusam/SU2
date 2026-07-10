#!/usr/bin/env python

import math
import os
import re

from SU2.opt.progressive_ffd_blending import (
    BEZIER,
    FFDBlendingSpec,
    evaluate_curve,
    invert_monotone_curve,
    make_blending_spec,
    validate_blending_spec,
)


_KEY_VALUE_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*=\s*(.*?)\s*(?:%.*)?$")

COORDS_ONLY_2D = "COORDS_ONLY_2D"
COORDS_ONLY_3D = "COORDS_ONLY_3D"
INDEXED_2D = "INDEXED_2D"
INDEXED_2D_WITH_Z = "INDEXED_2D_WITH_Z"
INDEXED_3D = "INDEXED_3D"

_INDEXED_FORMATS = (INDEXED_2D, INDEXED_2D_WITH_Z, INDEXED_3D)


class FFDMeshError(RuntimeError):
    pass


def _strip_comment(value):
    return value.split("%", 1)[0].strip()


def _line_key(line):
    match = _KEY_VALUE_RE.match(line)
    if not match:
        return None
    return match.group(1).strip().upper()


def _line_value(line):
    match = _KEY_VALUE_RE.match(line)
    if not match:
        return ""
    return _strip_comment(match.group(2))


def _normalize_tag(value):
    return str(value).strip().strip("()[]").strip()


def _split_numbers(line):
    raw = _strip_comment(line).replace(",", " ")
    values = []
    for token in raw.split():
        try:
            values.append(float(token))
        except Exception:
            pass
    return values


def _split_tokens(line):
    raw = _strip_comment(line).replace(",", " ")
    return raw.split()


def _as_int_if_possible(token, tol=1.0e-12):
    try:
        value = float(token)
    except Exception:
        return None
    nearest = int(round(value))
    if abs(value - nearest) <= tol:
        return nearest
    return None


def _format_float(value):
    return f"{float(value):.16g}"


def _format_key_value(key, value):
    return f"{key}= {value}\n"


def _unique_sorted(values, tol=1.0e-10):
    unique = []
    for value in sorted(float(v) for v in values):
        if not unique or abs(value - unique[-1]) > tol:
            unique.append(value)
        else:
            unique[-1] = 0.5 * (unique[-1] + value)
    return unique


def _parse_int_value(line, default=None):
    numbers = _split_numbers(_line_value(line))
    if not numbers:
        return default
    return int(round(numbers[0]))


def _parse_mesh_points(lines):
    ndime = 3
    npoint_line = None
    npoint = None

    for i, line in enumerate(lines):
        key = _line_key(line)
        if key == "NDIME":
            ndime = int(round(_split_numbers(_line_value(line))[0]))
        elif key == "NPOIN":
            npoint_line = i
            npoint = int(round(_split_numbers(_line_value(line))[0]))
            break

    if npoint_line is None or npoint is None:
        raise FFDMeshError("Could not find NPOIN in SU2 mesh")

    points = {}
    for local_id in range(npoint):
        idx = npoint_line + 1 + local_id
        if idx >= len(lines):
            raise FFDMeshError("NPOIN block ended before all points were read")
        values = _split_numbers(lines[idx])
        if len(values) < ndime:
            raise FFDMeshError(f"Invalid mesh point line: {lines[idx].rstrip()}")
        coords = list(values[:ndime])
        if ndime == 2:
            coords.append(0.0)
        points[local_id] = coords[:3]

    return ndime, points


def _find_tagged_ffd_block(lines, box_tag):
    box_tag = _normalize_tag(box_tag)
    tag_lines = []

    for i, line in enumerate(lines):
        if _line_key(line) == "FFD_TAG":
            tag_lines.append(i)

    for pos, tag_line in enumerate(tag_lines):
        tag = _normalize_tag(_line_value(lines[tag_line]))
        if tag != box_tag:
            continue
        block_end = tag_lines[pos + 1] if pos + 1 < len(tag_lines) else len(lines)
        return tag_line, block_end

    raise FFDMeshError(f"FFD_TAG={box_tag!r} was not found in the mesh")


def _find_key_line(lines, start, end, key):
    key = key.upper()
    for i in range(start, end):
        if _line_key(lines[i]) == key:
            return i
    return None


def _parse_count_block(lines, start, end, key):
    line_id = _find_key_line(lines, start, end, key)
    if line_id is None:
        return None
    count = _parse_int_value(lines[line_id])
    if count is None:
        raise FFDMeshError(f"Could not parse {key} count")
    data_start = line_id + 1
    data_end = data_start + count
    if data_end > end:
        raise FFDMeshError(f"{key} block extends beyond the FFD box")
    return {
        "line": line_id,
        "count": count,
        "data_start": data_start,
        "data_end": data_end,
        "data": list(lines[data_start:data_end]),
    }


def _parse_degree(lines, start, end):
    degree_i = None
    degree_j = None
    degree_k = None

    line_i = _find_key_line(lines, start, end, "FFD_DEGREE_I")
    line_j = _find_key_line(lines, start, end, "FFD_DEGREE_J")
    line_k = _find_key_line(lines, start, end, "FFD_DEGREE_K")

    if line_i is not None:
        degree_i = _parse_int_value(lines[line_i])
    if line_j is not None:
        degree_j = _parse_int_value(lines[line_j])
    if line_k is not None:
        degree_k = _parse_int_value(lines[line_k])

    line_legacy = _find_key_line(lines, start, end, "FFD_DEGREE")
    if line_legacy is not None:
        values = _split_numbers(_line_value(lines[line_legacy]))
        if len(values) >= 1 and degree_i is None:
            degree_i = int(round(values[0]))
        if len(values) >= 2 and degree_j is None:
            degree_j = int(round(values[1]))
        if len(values) >= 3 and degree_k is None:
            degree_k = int(round(values[2]))

    return {
        "i": degree_i,
        "j": degree_j,
        "k": degree_k,
        "line_i": line_i,
        "line_j": line_j,
        "line_k": line_k,
        "line_legacy": line_legacy,
    }


def _parse_control_points(control_block):
    points = []
    coord_dim = None
    control_format = None

    for line in control_block["data"]:
        values = _split_numbers(line)
        if len(values) < 2:
            raise FFDMeshError(f"Invalid FFD control point line: {line.rstrip()}")

        i = None
        j = None
        k = None

        if (
            len(values) >= 6
            and _as_int_if_possible(values[0]) is not None
            and _as_int_if_possible(values[1]) is not None
            and _as_int_if_possible(values[2]) is not None
        ):
            i = _as_int_if_possible(values[0])
            j = _as_int_if_possible(values[1])
            k = _as_int_if_possible(values[2])
            coords = values[3:6]
            this_format = INDEXED_3D
            this_coord_dim = 3
        elif (
            len(values) == 5
            and _as_int_if_possible(values[0]) is not None
            and _as_int_if_possible(values[1]) is not None
        ):
            i = _as_int_if_possible(values[0])
            j = _as_int_if_possible(values[1])
            coords = values[2:5]
            this_format = INDEXED_2D_WITH_Z
            this_coord_dim = 3
        elif (
            len(values) == 4
            and _as_int_if_possible(values[0]) is not None
            and _as_int_if_possible(values[1]) is not None
        ):
            i = _as_int_if_possible(values[0])
            j = _as_int_if_possible(values[1])
            coords = [values[2], values[3], 0.0]
            this_format = INDEXED_2D
            this_coord_dim = 2
        elif len(values) >= 3:
            coords = values[:3]
            this_format = COORDS_ONLY_3D
            this_coord_dim = 3
        else:
            coords = [values[0], values[1], 0.0]
            this_format = COORDS_ONLY_2D
            this_coord_dim = 2

        if control_format is None:
            control_format = this_format
            coord_dim = this_coord_dim
        elif control_format != this_format:
            raise FFDMeshError(
                "Mixed FFD_CONTROL_POINTS formats are not supported: "
                f"{control_format} and {this_format}"
            )

        points.append(
            {
                "i": i,
                "j": j,
                "k": k,
                "coords": [float(coords[0]), float(coords[1]), float(coords[2])],
            }
        )

    return points, coord_dim or 2, control_format or COORDS_ONLY_2D


def _average_axis_values(values_by_index, axis_name, expected_count=None):
    if expected_count is not None and len(values_by_index) != expected_count:
        raise FFDMeshError(
            f"FFD_CONTROL_POINTS {axis_name}-index count mismatch: "
            f"got {len(values_by_index)}, expected {expected_count}"
        )

    values = []
    for index in sorted(values_by_index.keys()):
        axis_values = values_by_index[index]
        if not axis_values:
            raise FFDMeshError(f"Empty FFD {axis_name}-index bucket {index}")
        values.append(sum(axis_values) / float(len(axis_values)))
    return values


def _infer_axes_from_control_points(control_points, degree):
    ncontrol = len(control_points)
    degree_i = degree.get("i")
    degree_j = degree.get("j")
    degree_k = degree.get("k")

    ni = degree_i + 1 if degree_i is not None else None
    nj = degree_j + 1 if degree_j is not None else None
    nk = degree_k + 1 if degree_k is not None else None

    indexed = bool(control_points) and control_points[0]["i"] is not None

    if indexed:
        x_by_i = {}
        y_by_j = {}
        z_by_k = {}
        z_values = []

        for point in control_points:
            coords = point["coords"]
            if point["i"] is None or point["j"] is None:
                raise FFDMeshError("Indexed FFD_CONTROL_POINTS require i and j")
            x_by_i.setdefault(point["i"], []).append(coords[0])
            y_by_j.setdefault(point["j"], []).append(coords[1])
            z_values.append(coords[2])
            if point["k"] is not None:
                z_by_k.setdefault(point["k"], []).append(coords[2])

        x_columns = _average_axis_values(x_by_i, "i", ni)
        y_rows = _average_axis_values(y_by_j, "j", nj)

        if z_by_k:
            z_planes = _average_axis_values(z_by_k, "k", nk)
        else:
            z_planes = _unique_sorted(z_values)
            if nk is not None and len(z_planes) != nk:
                raise FFDMeshError(
                    "FFD_CONTROL_POINTS z-plane count mismatch: "
                    f"got {len(z_planes)}, expected {nk}"
                )

        return x_columns, y_rows, z_planes

    if ni is not None and nj is not None:
        if nk is None:
            if ncontrol % (ni * nj) != 0:
                raise FFDMeshError(
                    "FFD_CONTROL_POINTS count is inconsistent with FFD_DEGREE_I/J"
                )
            nk = ncontrol // (ni * nj)
        if ni * nj * nk != ncontrol:
            raise FFDMeshError(
                "FFD_CONTROL_POINTS count is inconsistent with FFD degrees"
            )

        x_columns = []
        y_rows = []
        z_planes = []

        for i in range(ni):
            vals = []
            for k in range(nk):
                for j in range(nj):
                    vals.append(control_points[(k * nj + j) * ni + i]["coords"][0])
            x_columns.append(sum(vals) / float(len(vals)))

        for j in range(nj):
            vals = []
            for k in range(nk):
                for i in range(ni):
                    vals.append(control_points[(k * nj + j) * ni + i]["coords"][1])
            y_rows.append(sum(vals) / float(len(vals)))

        for k in range(nk):
            vals = []
            for j in range(nj):
                for i in range(ni):
                    vals.append(control_points[(k * nj + j) * ni + i]["coords"][2])
            z_planes.append(sum(vals) / float(len(vals)))

        return sorted(x_columns), sorted(y_rows), sorted(z_planes)

    x_columns = _unique_sorted(p["coords"][0] for p in control_points)
    y_rows = _unique_sorted(p["coords"][1] for p in control_points)
    z_planes = _unique_sorted(p["coords"][2] for p in control_points)

    if len(x_columns) * len(y_rows) * len(z_planes) != ncontrol:
        if len(z_planes) == 1 and len(x_columns) * len(y_rows) == ncontrol:
            return x_columns, y_rows, z_planes
        raise FFDMeshError(
            "Could not infer a tensor-product FFD grid from FFD_CONTROL_POINTS"
        )

    return x_columns, y_rows, z_planes


def _parse_blending_spec(lines, start, end, control_counts=None, dual_2d=False):
    line_id = _find_key_line(lines, start, end, "FFD_BLENDING")
    kind = BEZIER if line_id is None else _normalize_tag(_line_value(lines[line_id]))
    orders = [2, 2, 2]
    for axis, key in enumerate(
        ("BSPLINE_ORDER_I", "BSPLINE_ORDER_J", "BSPLINE_ORDER_K")
    ):
        order_line = _find_key_line(lines, start, end, key)
        if order_line is not None:
            orders[axis] = _parse_int_value(lines[order_line])
    spec = make_blending_spec(kind, orders)
    return validate_blending_spec(
        spec,
        control_counts=control_counts,
        dual_2d=dual_2d,
    )


def _binomial(n, i):
    return math.factorial(n) / float(math.factorial(i) * math.factorial(n - i))


def _bezier_1d(values, t):
    return evaluate_curve(values, t, FFDBlendingSpec(BEZIER), axis=0)


def _invert_monotone_bezier(values, target):
    return invert_monotone_curve(values, target, FFDBlendingSpec(BEZIER), axis=0)


def _eval_rectangular_ffd(axes, uvw):
    u, v, w = uvw
    spec = axes.get("blending_spec", FFDBlendingSpec(BEZIER))
    x = evaluate_curve(axes["columns"], u, spec, axis=0)
    y = evaluate_curve(axes["y_rows"], v, spec, axis=1)
    z = evaluate_curve(axes["z_planes"], w, spec, axis=2) if axes["z_planes"] else 0.0
    return [x, y, z]


def _infer_surface_param_location(tokens, point_coords, old_axes, ndime):
    best = None
    candidate_point_indices = []

    for idx, token in enumerate(tokens[:4]):
        point_id = _as_int_if_possible(token)
        if point_id is not None:
            candidate_point_indices.append((idx, point_id))

    for point_token_index, point_id in candidate_point_indices:
        for start in range(point_token_index + 1, len(tokens)):
            for count in (3, 2):
                if start + count > len(tokens):
                    continue
                try:
                    params = [float(v) for v in tokens[start:start + count]]
                except Exception:
                    continue
                if any(v < -1.0e-8 or v > 1.0 + 1.0e-8 for v in params):
                    continue
                uvw = [params[0], params[1], params[2] if count >= 3 else 0.0]
                mapped = _eval_rectangular_ffd(old_axes, uvw)
                err = math.sqrt(
                    sum(
                        (mapped[i] - point_coords[i]) ** 2
                        for i in range(ndime)
                    )
                )
                score = (err, -count, start)
                if best is None or score < best["score"]:
                    best = {
                        "point_id": point_id,
                        "point_token_index": point_token_index,
                        "param_start": start,
                        "param_count": count,
                        "score": score,
                    }

    return best


def _format_surface_line(tokens, param_start, param_count, uvw):
    tokens = list(tokens)
    replacements = [_format_float(uvw[0]), _format_float(uvw[1])]
    if param_count >= 3:
        replacements.append(_format_float(uvw[2]))

    for i, value in enumerate(replacements):
        tokens[param_start + i] = value

    return " ".join(tokens) + "\n"


def _reembed_surface_lines(surface_lines, mesh_points, old_axes, new_axes, ndime, marker_name):
    rewritten = []
    max_error = 0.0
    n_reembedded = 0

    for line in surface_lines:
        tokens = _split_tokens(line)
        if not tokens:
            rewritten.append(line)
            continue

        if marker_name and not _as_int_if_possible(tokens[0]):
            if tokens[0] != marker_name:
                rewritten.append(line)
                continue

        best = None
        for candidate_idx, token in enumerate(tokens[:4]):
            point_id = _as_int_if_possible(token)
            if point_id is None or point_id not in mesh_points:
                continue
            local_best = _infer_surface_param_location(
                tokens,
                mesh_points[point_id],
                old_axes,
                ndime,
            )
            if local_best is None:
                continue
            if local_best["point_id"] != point_id:
                continue
            if best is None or local_best["score"] < best["score"]:
                best = local_best

        if best is None:
            raise FFDMeshError(
                "Could not identify point id and parametric coordinates in "
                f"FFD_SURFACE_POINTS line: {line.rstrip()}"
            )

        point = mesh_points[best["point_id"]]
        spec = new_axes.get("blending_spec", FFDBlendingSpec(BEZIER))
        u = invert_monotone_curve(new_axes["columns"], point[0], spec, axis=0)
        v = invert_monotone_curve(new_axes["y_rows"], point[1], spec, axis=1)
        if best["param_count"] >= 3 and len(new_axes["z_planes"]) > 1:
            w = invert_monotone_curve(new_axes["z_planes"], point[2], spec, axis=2)
        elif best["param_count"] >= 3:
            try:
                w = float(tokens[best["param_start"] + 2])
            except Exception:
                w = 0.0
        else:
            w = 0.0

        uvw = [u, v, w]
        mapped = _eval_rectangular_ffd(new_axes, uvw)
        err = math.sqrt(sum((mapped[i] - point[i]) ** 2 for i in range(ndime)))
        max_error = max(max_error, err)
        n_reembedded += 1

        rewritten.append(
            _format_surface_line(
                tokens,
                best["param_start"],
                best["param_count"],
                uvw,
            )
        )

    return rewritten, max_error, n_reembedded


def _build_control_point_lines(columns, y_rows, z_planes, coord_dim, control_format):
    lines = []
    for i, x in enumerate(columns):
        for j, y in enumerate(y_rows):
            for k, z in enumerate(z_planes):
                coords = [x, y, z]
                if control_format == INDEXED_3D:
                    values = [str(i), str(j), str(k)] + [
                        _format_float(c) for c in coords[:3]
                    ]
                elif control_format == INDEXED_2D_WITH_Z:
                    values = [str(i), str(j)] + [
                        _format_float(c) for c in coords[:3]
                    ]
                elif control_format == INDEXED_2D:
                    values = [str(i), str(j)] + [
                        _format_float(c) for c in coords[:2]
                    ]
                elif control_format == COORDS_ONLY_3D:
                    values = [_format_float(c) for c in coords[:3]]
                elif control_format == COORDS_ONLY_2D:
                    values = [_format_float(c) for c in coords[:2]]
                else:
                    values = [_format_float(c) for c in coords[:coord_dim]]
                lines.append(" ".join(values) + "\n")
    return lines


def _build_corner_lines(columns, y_rows, z_planes, coord_dim, old_count):
    if old_count not in (4, 8):
        return None

    x_values = [columns[0], columns[-1]]
    y_values = [y_rows[0], y_rows[-1]]
    if old_count == 8:
        z_values = [z_planes[0], z_planes[-1]]
    else:
        z_values = [z_planes[0] if z_planes else 0.0]

    corners = []
    if old_count == 4:
        for y in y_values:
            for x in x_values:
                corners.append([x, y, z_values[0]])
    else:
        for z in z_values:
            for y in y_values:
                for x in x_values:
                    corners.append([x, y, z])

    return [" ".join(_format_float(c) for c in p[:coord_dim]) + "\n" for p in corners]


def _ranges_to_skip(*blocks):
    skip = set()
    for block in blocks:
        if not block:
            continue
        for idx in range(block["data_start"], block["data_end"]):
            skip.add(idx)
    return skip


def validate_mesh_ffd_columns(columns):
    columns = sorted(float(x) for x in columns)
    if len(columns) < 2:
        raise ValueError("Progressive FFD requires at least two chordwise columns")
    for a, b in zip(columns[:-1], columns[1:]):
        if abs(a - b) <= 1.0e-10:
            raise ValueError("Progressive FFD columns must be unique")
    return columns


def _corner_boundary_columns(lines, block_start, block_end):
    corner_block = _parse_count_block(
        lines, block_start, block_end, "FFD_CORNER_POINTS"
    )
    if corner_block is None:
        raise FFDMeshError("FFD_CORNER_POINTS block was not found")

    x_values = []
    for line in corner_block["data"]:
        values = _split_numbers(line)
        if values:
            x_values.append(float(values[0]))

    if not x_values:
        raise FFDMeshError("FFD_CORNER_POINTS contains no x-coordinates")

    columns = validate_mesh_ffd_columns([min(x_values), max(x_values)])
    print(
        "[PROGRESSIVE_FFD] WARNING: using FFD_CORNER_POINTS for boundary columns"
    )
    return columns


def read_ffd_box_columns(mesh_in, box_tag):
    with open(mesh_in, "r") as fp:
        lines = fp.readlines()

    block_start, block_end = _find_tagged_ffd_block(lines, box_tag)
    degree = _parse_degree(lines, block_start, block_end)
    control_block = _parse_count_block(
        lines, block_start, block_end, "FFD_CONTROL_POINTS"
    )
    if control_block is None:
        return _corner_boundary_columns(lines, block_start, block_end)

    try:
        old_control_points, _, control_format = _parse_control_points(control_block)
        old_columns, y_rows, z_planes = _infer_axes_from_control_points(
            old_control_points,
            degree,
        )
        old_columns = validate_mesh_ffd_columns(old_columns)
    except Exception:
        return _corner_boundary_columns(lines, block_start, block_end)

    print(f"[PROGRESSIVE_FFD] FFD control point format = {control_format}")
    print(f"[PROGRESSIVE_FFD] Existing FFD columns = {old_columns}")
    print(f"[PROGRESSIVE_FFD] Existing FFD y rows = {y_rows}")
    print(f"[PROGRESSIVE_FFD] Existing FFD z planes = {z_planes}")

    return old_columns


def rewrite_ffd_box_with_columns_and_reembed(
    mesh_in,
    mesh_out,
    box_tag,
    new_columns,
    marker_name,
    domain_mode,
):
    """
    SU2 FFD boxes are structured tensor-product grids.  A progressive FFD
    backend cannot add a sparse isolated control point; changing chordwise
    resolution means rewriting the mesh FFD block with a complete new grid.
    After the control grid changes, FFD_SURFACE_POINTS must be re-embedded so
    that DV=0 maps the current surface coordinates back to themselves.
    """
    if str(domain_mode).upper() not in ("FULL", "HALF_UPPER"):
        raise ValueError(
            "PROGRESSIVE_FFD_DOMAIN_MODE must be FULL or HALF_UPPER, "
            f"got {domain_mode!r}"
        )

    new_columns = validate_mesh_ffd_columns(new_columns)

    with open(mesh_in, "r") as fp:
        lines = fp.readlines()

    ndime, mesh_points = _parse_mesh_points(lines)
    block_start, block_end = _find_tagged_ffd_block(lines, box_tag)
    degree = _parse_degree(lines, block_start, block_end)
    control_block = _parse_count_block(
        lines, block_start, block_end, "FFD_CONTROL_POINTS"
    )
    surface_block = _parse_count_block(
        lines, block_start, block_end, "FFD_SURFACE_POINTS"
    )
    corner_block = _parse_count_block(
        lines, block_start, block_end, "FFD_CORNER_POINTS"
    )

    if control_block is None:
        raise FFDMeshError("FFD_CONTROL_POINTS block is required for progressive FFD")
    if surface_block is None:
        raise FFDMeshError("FFD_SURFACE_POINTS block is required for re-embedding")

    old_control_points, coord_dim, control_format = _parse_control_points(control_block)
    old_columns, y_rows, z_planes = _infer_axes_from_control_points(
        old_control_points,
        degree,
    )
    old_columns = validate_mesh_ffd_columns(old_columns)

    if not y_rows:
        raise FFDMeshError("Could not infer FFD vertical rows")
    if not z_planes:
        z_planes = [0.0]

    coord_dim = max(coord_dim, 3 if len(z_planes) > 1 else coord_dim)
    blending_spec = _parse_blending_spec(
        lines,
        block_start,
        block_end,
        control_counts=(len(old_columns), len(y_rows), len(z_planes)),
    )
    validate_blending_spec(
        blending_spec,
        control_counts=(len(new_columns), len(y_rows), len(z_planes)),
    )

    old_axes = {
        "columns": old_columns,
        "y_rows": y_rows,
        "z_planes": z_planes,
        "blending_spec": blending_spec,
    }
    new_axes = {
        "columns": new_columns,
        "y_rows": y_rows,
        "z_planes": z_planes,
        "blending_spec": blending_spec,
    }

    new_control_lines = _build_control_point_lines(
        new_columns,
        y_rows,
        z_planes,
        coord_dim,
        control_format,
    )
    new_surface_lines, max_error, n_reembedded = _reembed_surface_lines(
        surface_block["data"],
        mesh_points,
        old_axes,
        new_axes,
        ndime,
        marker_name if str(domain_mode).upper() == "HALF_UPPER" else None,
    )

    new_corner_lines = None
    if corner_block is not None:
        new_corner_lines = _build_corner_lines(
            new_columns,
            y_rows,
            z_planes,
            coord_dim,
            corner_block["count"],
        )

    degree_i = len(new_columns) - 1
    degree_j = len(y_rows) - 1
    degree_k = len(z_planes) - 1
    control_count = len(new_control_lines)

    print(f"[PROGRESSIVE_FFD] FFD control point format = {control_format}")
    print(f"[PROGRESSIVE_FFD] Existing FFD columns = {old_columns}")
    print(f"[PROGRESSIVE_FFD] Existing FFD y rows = {y_rows}")
    print(f"[PROGRESSIVE_FFD] Existing FFD z planes = {z_planes}")

    skip = _ranges_to_skip(control_block, surface_block, corner_block)
    out_lines = list(lines[:block_start])
    inserted_degrees = False

    for idx in range(block_start, block_end):
        if idx in skip:
            continue

        key = _line_key(lines[idx])

        if key == "FFD_DEGREE_I":
            out_lines.append(_format_key_value("FFD_DEGREE_I", degree_i))
            inserted_degrees = True
            continue
        if key == "FFD_DEGREE_J":
            out_lines.append(_format_key_value("FFD_DEGREE_J", degree_j))
            inserted_degrees = True
            continue
        if key == "FFD_DEGREE_K":
            out_lines.append(_format_key_value("FFD_DEGREE_K", degree_k))
            inserted_degrees = True
            continue
        if key == "FFD_DEGREE":
            out_lines.append(
                _format_key_value(
                    "FFD_DEGREE",
                    f"{degree_i}, {degree_j}, {degree_k}",
                )
            )
            inserted_degrees = True
            continue

        if key == "FFD_CONTROL_POINTS":
            if not inserted_degrees and degree["line_legacy"] is None:
                out_lines.append(_format_key_value("FFD_DEGREE_I", degree_i))
                out_lines.append(_format_key_value("FFD_DEGREE_J", degree_j))
                out_lines.append(_format_key_value("FFD_DEGREE_K", degree_k))
                inserted_degrees = True
            out_lines.append(_format_key_value("FFD_CONTROL_POINTS", control_count))
            out_lines.extend(new_control_lines)
            continue

        if key == "FFD_SURFACE_POINTS":
            out_lines.append(
                _format_key_value("FFD_SURFACE_POINTS", surface_block["count"])
            )
            out_lines.extend(new_surface_lines)
            continue

        if key == "FFD_CORNER_POINTS" and new_corner_lines is not None:
            out_lines.append(_format_key_value("FFD_CORNER_POINTS", len(new_corner_lines)))
            out_lines.extend(new_corner_lines)
            continue

        out_lines.append(lines[idx])

    out_lines.extend(lines[block_end:])

    tmp_out = mesh_out + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(mesh_out)), exist_ok=True)
    with open(tmp_out, "w") as fp:
        fp.writelines(out_lines)
    os.replace(tmp_out, mesh_out)

    column_index_by_x = {float(x): i for i, x in enumerate(new_columns)}

    print(
        "[PROGRESSIVE_FFD] Re-embedding max error = "
        f"{max_error:.6e} | points={n_reembedded}"
    )

    return {
        "mesh_out": mesh_out,
        "box_tag": box_tag,
        "columns": list(new_columns),
        "column_index_by_x": column_index_by_x,
        "y_rows": list(y_rows),
        "z_planes": list(z_planes),
        "degree_i": degree_i,
        "degree_j": degree_j,
        "degree_k": degree_k,
        "control_points": control_count,
        "surface_points": surface_block["count"],
        "reembedding_max_error": max_error,
        "blending": blending_spec.kind,
        "bspline_orders": list(blending_spec.orders),
    }
