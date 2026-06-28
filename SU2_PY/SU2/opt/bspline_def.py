#!/usr/bin/env python

"""External BSPLINE_DEF v1 writer for SU2 SURFACE_FILE deformation."""

import argparse
import csv
import math
from collections import defaultdict

from SU2.opt.bspline_modes import (
    ALLOWED_DEFORMATION_DIRECTION_MODES,
    ALLOWED_SURFACE_MODES,
    BSplineModeError,
    LE_SAFE_DEFAULT_POWER,
    LE_SAFE_DEFAULT_X0,
    LE_SAFE_DEFAULT_X1,
    deformation_direction,
    evaluate_normal_displacement,
    load_mode_spec,
    normalize_deformation_direction_mode,
    normalize_surface_mode,
    validate_le_safe_direction_options,
    validate_mode_spec,
    validate_surface_mode_against_modes,
)


class BSplineDefError(RuntimeError):
    pass


def _strip_comment(line):
    return line.split("%", 1)[0].strip()


def _parse_key_value(line):
    if "=" not in line:
        return None, None
    key, value = line.split("=", 1)
    return key.strip().upper(), _strip_comment(value)


def _numbers(line):
    values = []
    raw = _strip_comment(line).replace(",", " ")
    for token in raw.split():
        try:
            values.append(float(token))
        except Exception:
            pass
    return values


def _ints(line):
    values = []
    raw = _strip_comment(line).replace(",", " ")
    for token in raw.split():
        try:
            values.append(int(float(token)))
        except Exception:
            pass
    return values


def _normalize_marker_name(value):
    return str(value).strip().strip("()[]").strip()


def _marker_element_nodes(values):
    if not values:
        return []
    elem_type = values[0]
    if elem_type == 3 and len(values) >= 3:
        return values[1:3]
    if elem_type == 5 and len(values) >= 4:
        return values[1:4]
    if elem_type == 9 and len(values) >= 5:
        return values[1:5]
    return values[1:]


def read_su2_mesh(mesh_filename):
    """Read enough of an ASCII SU2 mesh for 2D marker deformation."""

    with open(mesh_filename, "r") as fp:
        lines = fp.readlines()

    ndime = 2
    points = {}
    markers = {}
    i = 0

    while i < len(lines):
        key, value = _parse_key_value(lines[i])

        if key == "NDIME":
            vals = _numbers(value)
            if vals:
                ndime = int(round(vals[0]))

        elif key == "NPOIN":
            vals = _numbers(value)
            if not vals:
                raise BSplineDefError(f"Invalid NPOIN line in {mesh_filename}")
            npoint = int(round(vals[0]))
            for local_id in range(npoint):
                i += 1
                if i >= len(lines):
                    raise BSplineDefError("NPOIN block ended before all points were read")
                vals_point = _numbers(lines[i])
                if len(vals_point) < ndime:
                    raise BSplineDefError(
                        f"Invalid point line in {mesh_filename}: {lines[i].rstrip()}"
                    )
                coords = list(vals_point[:ndime])
                if ndime == 2:
                    coords.append(0.0)
                point_id = local_id
                if len(vals_point) > ndime:
                    candidate_id = int(round(vals_point[-1]))
                    if candidate_id >= 0:
                        point_id = candidate_id
                points[point_id] = coords[:3]

        elif key == "MARKER_TAG":
            tag = _normalize_marker_name(value)
            if i + 1 >= len(lines):
                raise BSplineDefError(f"MARKER_TAG {tag!r} is missing MARKER_ELEMS")
            elem_key, elem_value = _parse_key_value(lines[i + 1])
            if elem_key != "MARKER_ELEMS":
                raise BSplineDefError(
                    f"Expected MARKER_ELEMS after MARKER_TAG {tag!r}"
                )
            vals = _numbers(elem_value)
            nelem = int(round(vals[0])) if vals else 0

            elements = []
            segments = []
            for j in range(nelem):
                idx = i + 2 + j
                if idx >= len(lines):
                    raise BSplineDefError(
                        f"MARKER_ELEMS block for {tag!r} ended early"
                    )
                nodes = _marker_element_nodes(_ints(lines[idx]))
                if len(nodes) < 2:
                    continue
                elements.append(nodes)
                for a, b in zip(nodes[:-1], nodes[1:]):
                    segments.append((a, b))

            markers[tag] = {"elements": elements, "segments": segments}
            i += nelem + 1

        i += 1

    if not points:
        raise BSplineDefError(f"No mesh points were found in {mesh_filename}")

    return {"ndime": ndime, "points": points, "markers": markers}


def _find_marker(markers, marker_name):
    requested = _normalize_marker_name(marker_name)
    if requested in markers:
        return requested, markers[requested]
    requested_lower = requested.lower()
    for tag, marker in markers.items():
        if tag.lower() == requested_lower:
            return tag, marker
    raise BSplineDefError(f"Marker {marker_name!r} was not found in the mesh")


def _order_marker_nodes(segments):
    adjacency = defaultdict(list)
    for a, b in segments:
        if a == b:
            continue
        if b not in adjacency[a]:
            adjacency[a].append(b)
        if a not in adjacency[b]:
            adjacency[b].append(a)

    if not adjacency:
        raise BSplineDefError("Marker has no usable line segments")

    all_nodes = set(adjacency)
    endpoints = [node for node in all_nodes if len(adjacency[node]) == 1]
    if endpoints:
        start = endpoints[0]
        preferred_next = adjacency[start][0]
    else:
        start = segments[0][0]
        preferred_next = segments[0][1]

    ordered = [start]
    visited = {start}
    previous = None
    current = start

    while True:
        candidates = [node for node in adjacency[current] if node != previous]
        next_node = None

        if preferred_next in candidates and preferred_next not in visited:
            next_node = preferred_next
        else:
            for candidate in candidates:
                if candidate not in visited:
                    next_node = candidate
                    break

        if next_node is None:
            break

        ordered.append(next_node)
        visited.add(next_node)
        previous = current
        current = next_node
        preferred_next = None

    if len(ordered) != len(all_nodes):
        raise BSplineDefError("Marker boundary is disconnected or branched")

    closed = bool(ordered and ordered[0] in adjacency[ordered[-1]])
    return ordered, closed


def extract_marker_nodes(mesh, marker_name):
    tag, marker = _find_marker(mesh["markers"], marker_name)
    ordered_ids, closed = _order_marker_nodes(marker["segments"])

    missing = [node_id for node_id in ordered_ids if node_id not in mesh["points"]]
    if missing:
        raise BSplineDefError(
            f"Marker {tag!r} references nodes missing from NPOIN: {missing[:5]}"
        )

    return tag, ordered_ids, closed


def _distance_2d(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def infer_chord(points, node_ids, chord_spec):
    mode = str(chord_spec.get("mode", "auto")).strip().lower()
    auto = mode == "auto" or chord_spec.get("x_le") is None or chord_spec.get("x_te") is None

    if auto:
        xs = [float(points[node_id][0]) for node_id in node_ids]
        x_le = min(xs)
        x_te = max(xs)
    else:
        x_le = float(chord_spec.get("x_le"))
        x_te = float(chord_spec.get("x_te"))

    chord = x_te - x_le
    if chord <= 0.0:
        raise BSplineDefError(
            f"Invalid chord definition: x_le={x_le}, x_te={x_te}"
        )
    return x_le, x_te, chord


def _normalized_x(x, x_le, chord):
    value = (float(x) - x_le) / chord
    tol = 1.0e-12
    if abs(value) <= tol:
        return 0.0
    if abs(value - 1.0) <= tol:
        return 1.0
    if value < -1.0e-8 or value > 1.0 + 1.0e-8:
        raise BSplineDefError(
            f"Marker point x/c={value} lies outside the chord interval"
        )
    return value


def _cyclic_arc_indices(start, stop, n):
    indices = [start]
    index = start
    while index != stop:
        index = (index + 1) % n
        indices.append(index)
    return indices


def _mean_y(indices, y_values, split_indices):
    usable = [index for index in indices if index not in split_indices]
    if not usable:
        usable = list(indices)
    if not usable:
        raise BSplineDefError("Unable to classify marker sides: empty side arc")
    return sum(float(y_values[index]) for index in usable) / len(usable)


def _assign_split_node_sides(sides, split_indices, closed):
    n = len(sides)
    for index in sorted(split_indices):
        candidates = []
        if closed:
            candidates.extend(((index + 1) % n, (index - 1) % n))
        else:
            if index + 1 < n:
                candidates.append(index + 1)
            if index - 1 >= 0:
                candidates.append(index - 1)

        for candidate in candidates:
            if candidate != index and sides[candidate] in ("upper", "lower"):
                sides[index] = sides[candidate]
                break

        if sides[index] not in ("upper", "lower"):
            sides[index] = "upper"


def _classify_from_arcs(n, arc_a, arc_b, y_values, split_indices, closed):
    mean_a = _mean_y(arc_a, y_values, split_indices)
    mean_b = _mean_y(arc_b, y_values, split_indices)
    side_a, side_b = ("upper", "lower") if mean_a >= mean_b else ("lower", "upper")

    sides = [None] * n
    for index in arc_a:
        if index not in split_indices:
            sides[index] = side_a
    for index in arc_b:
        if index not in split_indices:
            sides[index] = side_b

    _assign_split_node_sides(sides, split_indices, closed)
    return sides


def _smooth_isolated_side_flips(sides, split_indices, closed):
    sides = list(sides)
    n = len(sides)
    if n < 3:
        return sides

    for _ in range(n):
        changed = False
        indices = range(n) if closed else range(1, n - 1)
        for index in indices:
            if index in split_indices:
                continue
            previous = (index - 1) % n
            next_index = (index + 1) % n
            if sides[previous] == sides[next_index] and sides[index] != sides[previous]:
                sides[index] = sides[previous]
                changed = True
        if not changed:
            break
    return sides


def _fallback_y_based_sides(y_values, split_indices, closed):
    if not y_values:
        return []
    y_min = min(float(value) for value in y_values)
    y_max = max(float(value) for value in y_values)
    threshold = 0.5 * (y_min + y_max)
    sides = [
        "upper" if float(value) >= threshold - 1.0e-12 else "lower"
        for value in y_values
    ]
    return _smooth_isolated_side_flips(sides, split_indices, closed)


def _classify_open_sides(n, i_le, i_te, y_values):
    split_indices = {i_le, i_te}
    if i_le == i_te or not ({i_le, i_te} & {0, n - 1}):
        return None, split_indices

    lo = min(i_le, i_te)
    hi = max(i_le, i_te)
    arc_a = list(range(lo, hi + 1))
    arc_b = list(range(0, lo + 1)) + list(range(hi, n))
    usable_a = [index for index in arc_a if index not in split_indices]
    usable_b = [index for index in arc_b if index not in split_indices]
    if not usable_a or not usable_b:
        return None, split_indices

    return (
        _classify_from_arcs(n, arc_a, arc_b, y_values, split_indices, closed=False),
        split_indices,
    )


def _raise_on_isolated_side_islands(
    node_ids,
    x_over_c,
    y_values,
    sides,
    split_indices,
    closed,
):
    n = len(sides)
    if n < 3:
        return

    indices = range(n) if closed else range(1, n - 1)
    for index in indices:
        if index in split_indices:
            continue
        previous = (index - 1) % n
        next_index = (index + 1) % n
        if sides[previous] == sides[next_index] and sides[index] != sides[previous]:
            raise BSplineDefError(
                "Isolated side classification island at node "
                f"{node_ids[index]}: x_over_c={float(x_over_c[index]):.16g}, "
                f"y={float(y_values[index]):.16g}, "
                f"previous side={sides[previous]!r}, "
                f"current side={sides[index]!r}, "
                f"next side={sides[next_index]!r}"
            )


def _apply_side_overrides(node_ids, sides, side_overrides):
    overrides = side_overrides or {}
    if not overrides:
        return list(sides)

    result = list(sides)
    for index, node_id in enumerate(node_ids):
        override = None
        if node_id in overrides:
            override = overrides[node_id]
        elif str(node_id) in overrides:
            override = overrides[str(node_id)]
        if override is None:
            continue

        override = str(override).strip().lower()
        if override not in ("upper", "lower"):
            raise BSplineDefError(
                f"Invalid side override for node {node_id}: {override!r}"
            )
        result[index] = override
    return result


def classify_sides(
    node_ids,
    x_over_c,
    y_values,
    side_overrides=None,
    nbins=None,
    *,
    closed=True,
):
    del nbins

    n = len(node_ids)
    if len(x_over_c) != n or len(y_values) != n:
        raise BSplineDefError(
            "node_ids, x_over_c, and y_values must have the same length"
        )
    if n == 0:
        return []

    i_le = min(range(n), key=lambda index: float(x_over_c[index]))
    i_te = max(range(n), key=lambda index: float(x_over_c[index]))
    split_indices = {i_le, i_te}

    if closed and i_le != i_te:
        arc_a = _cyclic_arc_indices(i_le, i_te, n)
        arc_b = _cyclic_arc_indices(i_te, i_le, n)
        sides = _classify_from_arcs(
            n,
            arc_a,
            arc_b,
            y_values,
            split_indices,
            closed=True,
        )
    else:
        sides, split_indices = _classify_open_sides(n, i_le, i_te, y_values)
        if sides is None:
            sides = _fallback_y_based_sides(y_values, split_indices, closed=False)

    _raise_on_isolated_side_islands(
        node_ids,
        x_over_c,
        y_values,
        sides,
        split_indices,
        closed,
    )
    return _apply_side_overrides(node_ids, sides, side_overrides)


def _signed_polygon_area(points, node_ids, closed):
    if not closed or len(node_ids) < 3:
        return 0.0
    area2 = 0.0
    for i, node_id in enumerate(node_ids):
        next_id = node_ids[(i + 1) % len(node_ids)]
        x0, y0 = points[node_id][0], points[node_id][1]
        x1, y1 = points[next_id][0], points[next_id][1]
        area2 += x0 * y1 - x1 * y0
    return 0.5 * area2


def _unit(dx, dy):
    length = math.hypot(dx, dy)
    if length <= 0.0:
        return 0.0, 0.0
    return dx / length, dy / length


def compute_surface_normals(points, node_ids, sides, closed):
    area = _signed_polygon_area(points, node_ids, closed)
    normals = []

    for i, node_id in enumerate(node_ids):
        if closed:
            prev_id = node_ids[(i - 1) % len(node_ids)]
            next_id = node_ids[(i + 1) % len(node_ids)]
            dx = points[next_id][0] - points[prev_id][0]
            dy = points[next_id][1] - points[prev_id][1]
        elif i == 0:
            next_id = node_ids[i + 1]
            dx = points[next_id][0] - points[node_id][0]
            dy = points[next_id][1] - points[node_id][1]
        elif i == len(node_ids) - 1:
            prev_id = node_ids[i - 1]
            dx = points[node_id][0] - points[prev_id][0]
            dy = points[node_id][1] - points[prev_id][1]
        else:
            prev_id = node_ids[i - 1]
            next_id = node_ids[i + 1]
            dx = points[next_id][0] - points[prev_id][0]
            dy = points[next_id][1] - points[prev_id][1]

        tx, ty = _unit(dx, dy)
        if tx == 0.0 and ty == 0.0:
            normals.append((0.0, 0.0))
            continue

        right = (ty, -tx)
        left = (-ty, tx)

        if closed and abs(area) > 1.0e-14:
            normal = right if area > 0.0 else left
        elif sides[i] == "upper":
            normal = right if right[1] >= left[1] else left
        else:
            normal = right if right[1] <= left[1] else left

        normals.append(normal)

    return normals


def compute_arc_length_weights(points, node_ids, closed):
    weights = []
    n = len(node_ids)
    for i, node_id in enumerate(node_ids):
        if closed:
            prev_id = node_ids[(i - 1) % n]
            next_id = node_ids[(i + 1) % n]
            weight = 0.5 * (
                _distance_2d(points[prev_id], points[node_id])
                + _distance_2d(points[node_id], points[next_id])
            )
        elif i == 0:
            weight = 0.5 * _distance_2d(points[node_id], points[node_ids[i + 1]])
        elif i == n - 1:
            weight = 0.5 * _distance_2d(points[node_ids[i - 1]], points[node_id])
        else:
            weight = 0.5 * (
                _distance_2d(points[node_ids[i - 1]], points[node_id])
                + _distance_2d(points[node_id], points[node_ids[i + 1]])
            )
        weights.append(weight)
    return weights


def compute_deformed_surface(
    mesh_filename,
    mode_spec,
    marker_name=None,
    side_overrides=None,
    le_safe_direction=False,
    le_safe_x0=LE_SAFE_DEFAULT_X0,
    le_safe_x1=LE_SAFE_DEFAULT_X1,
    le_safe_power=LE_SAFE_DEFAULT_POWER,
    deformation_direction_mode=None,
    surface_mode="BOTH",
):
    spec = validate_mode_spec(mode_spec)
    try:
        surface_mode = normalize_surface_mode(surface_mode)
        validate_surface_mode_against_modes(spec, surface_mode)
    except BSplineModeError as exc:
        raise BSplineDefError(str(exc))
    direction_mode = normalize_deformation_direction_mode(
        deformation_direction_mode,
        le_safe_direction=le_safe_direction,
    )
    direction_options = validate_le_safe_direction_options(
        direction_mode == "LE_SAFE",
        le_safe_x0 if direction_mode == "LE_SAFE" else LE_SAFE_DEFAULT_X0,
        le_safe_x1 if direction_mode == "LE_SAFE" else LE_SAFE_DEFAULT_X1,
        le_safe_power if direction_mode == "LE_SAFE" else LE_SAFE_DEFAULT_POWER,
    )
    if not spec.get("normal_displacement", True):
        raise BSplineDefError("BSPLINE_DEF v1 only supports normal_displacement=true")

    mesh = read_su2_mesh(mesh_filename)
    if mesh["ndime"] != 2:
        raise BSplineDefError("BSPLINE_DEF v1 only supports 2D SU2 meshes")

    requested_marker = marker_name or spec["marker"]
    marker_tag, node_ids, closed = extract_marker_nodes(mesh, requested_marker)
    points = mesh["points"]
    x_le, x_te, chord = infer_chord(points, node_ids, spec.get("chord", {}))

    x_over_c = [
        _normalized_x(points[node_id][0], x_le, chord)
        for node_id in node_ids
    ]
    y_values = [points[node_id][1] for node_id in node_ids]
    if surface_mode == "BOTH":
        sides = classify_sides(
            node_ids,
            x_over_c,
            y_values,
            side_overrides=side_overrides,
            closed=closed,
        )
    else:
        sides = [surface_mode.lower()] * len(node_ids)
    normals = compute_surface_normals(points, node_ids, sides, closed)
    weights = compute_arc_length_weights(points, node_ids, closed)
    normal_displacement, mode_values = evaluate_normal_displacement(
        spec,
        x_over_c,
        sides,
    )

    records = []
    for node_id, xoc, side, normal, weight, displacement in zip(
        node_ids,
        x_over_c,
        sides,
        normals,
        weights,
        normal_displacement,
    ):
        x = float(points[node_id][0])
        y = float(points[node_id][1])
        nx, ny = normal
        dir_x, dir_y = deformation_direction(
            xoc,
            side,
            nx,
            ny,
            use_le_safe_direction=direction_options["le_safe_direction"],
            le_safe_x0=direction_options["le_safe_x0"],
            le_safe_x1=direction_options["le_safe_x1"],
            le_safe_power=direction_options["le_safe_power"],
            direction_mode=direction_mode,
        )
        deformed_x = x + displacement * dir_x
        deformed_y = y + displacement * dir_y
        records.append(
            {
                "node_id": node_id,
                "x": x,
                "y": y,
                "x_over_c": xoc,
                "side": side,
                "normal_x": nx,
                "normal_y": ny,
                "deform_dir_x": dir_x,
                "deform_dir_y": dir_y,
                "deformation_direction_mode": direction_mode,
                "surface_mode": surface_mode,
                "weight": weight,
                "normal_displacement": displacement,
                "deformed_x": deformed_x,
                "deformed_y": deformed_y,
            }
        )

    return {
        "marker": marker_tag,
        "closed": closed,
        "x_le": x_le,
        "x_te": x_te,
        "chord": chord,
        "mode_values": mode_values,
        "records": records,
    }


def write_surface_positions(records, filename):
    with open(filename, "w") as fp:
        for record in records:
            fp.write(
                "{:d}\t{:.15g}\t{:.15g}\n".format(
                    int(record["node_id"]),
                    float(record["deformed_x"]),
                    float(record["deformed_y"]),
                )
            )


def write_metadata(records, filename):
    fieldnames = [
        "node_id",
        "x",
        "y",
        "x_over_c",
        "side",
        "normal_x",
        "normal_y",
        "deform_dir_x",
        "deform_dir_y",
        "deformation_direction_mode",
        "surface_mode",
        "weight",
        "deformed_x",
        "deformed_y",
    ]
    with open(filename, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({name: record[name] for name in fieldnames})


def write_bspline_surface_files(
    mesh_filename,
    modes_filename,
    output_filename="surface_positions.dat",
    metadata_filename="bspline_surface_metadata.csv",
    marker_name=None,
    side_overrides=None,
    le_safe_direction=False,
    le_safe_x0=LE_SAFE_DEFAULT_X0,
    le_safe_x1=LE_SAFE_DEFAULT_X1,
    le_safe_power=LE_SAFE_DEFAULT_POWER,
    deformation_direction_mode=None,
    surface_mode="BOTH",
):
    spec = load_mode_spec(modes_filename)
    result = compute_deformed_surface(
        mesh_filename,
        spec,
        marker_name=marker_name,
        side_overrides=side_overrides,
        le_safe_direction=le_safe_direction,
        le_safe_x0=le_safe_x0,
        le_safe_x1=le_safe_x1,
        le_safe_power=le_safe_power,
        deformation_direction_mode=deformation_direction_mode,
        surface_mode=surface_mode,
    )
    write_surface_positions(result["records"], output_filename)
    write_metadata(result["records"], metadata_filename)
    return result


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Write SU2 SURFACE_FILE positions from active B-spline modes."
    )
    parser.add_argument("--mesh", required=True, help="Input SU2 mesh file")
    parser.add_argument("--modes", required=True, help="B-spline mode JSON file")
    parser.add_argument(
        "--marker",
        default=None,
        help="Surface marker to deform; defaults to the JSON marker",
    )
    parser.add_argument(
        "--output",
        default="surface_positions.dat",
        help="Output SU2 surface positions file",
    )
    parser.add_argument(
        "--metadata",
        default="bspline_surface_metadata.csv",
        help="Output metadata CSV file",
    )
    parser.add_argument(
        "--surface-mode",
        default="BOTH",
        choices=ALLOWED_SURFACE_MODES,
        help="Treat the marker as a full airfoil or one half-domain surface",
    )
    parser.add_argument(
        "--deformation-direction",
        default=None,
        choices=ALLOWED_DEFORMATION_DIRECTION_MODES,
        help="Direction used to apply the scalar B-spline deformation",
    )
    parser.add_argument(
        "--le-safe-direction",
        action="store_true",
        default=False,
        help="Use the fixed-leading-edge-safe deformation direction near x/c=0",
    )
    parser.add_argument("--le-safe-x0", type=float, default=LE_SAFE_DEFAULT_X0)
    parser.add_argument("--le-safe-x1", type=float, default=LE_SAFE_DEFAULT_X1)
    parser.add_argument("--le-safe-power", type=float, default=LE_SAFE_DEFAULT_POWER)
    return parser


def main(argv=None):
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        result = write_bspline_surface_files(
            args.mesh,
            args.modes,
            output_filename=args.output,
            metadata_filename=args.metadata,
            marker_name=args.marker,
            le_safe_direction=args.le_safe_direction,
            le_safe_x0=args.le_safe_x0,
            le_safe_x1=args.le_safe_x1,
            le_safe_power=args.le_safe_power,
            deformation_direction_mode=args.deformation_direction,
            surface_mode=args.surface_mode,
        )
    except (BSplineDefError, BSplineModeError, OSError) as exc:
        parser.error(str(exc))

    print(
        "Wrote {} for marker {} with {} nodes".format(
            args.output,
            result["marker"],
            len(result["records"]),
        )
    )
    print("Wrote {}".format(args.metadata))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
