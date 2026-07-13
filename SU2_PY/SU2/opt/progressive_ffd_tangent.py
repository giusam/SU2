#!/usr/bin/env python

"""Physical tangent-space scoring for progressive FFD refinement."""

import csv
import math
import os

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import lsq_linear

from SU2.opt.bspline_def import extract_marker_nodes, infer_chord, read_su2_mesh
from SU2.opt.bspline_dot import (
    match_sensitivities_to_metadata,
    read_sensitivity_file,
)
from SU2.opt.progressive_ffd_blending import basis_values
from SU2.opt.progressive_ffd_core import ordered_ffd_records
from SU2.opt.progressive_ffd_split import (
    _parse_curved_surface_lines,
    _parse_existing_dual_box,
)


COMPONENT = "COMPONENT"
VIRTUAL_TANGENT = "VIRTUAL_TANGENT"
SUPPORTED_FFD_SCORING_MODES = (COMPONENT, VIRTUAL_TANGENT)

RANK_RTOL = 1.0e-10
RANK_ATOL = 1.0e-12
LOCALITY_RADIUS = 0.10


class FFDTangentError(RuntimeError):
    pass


def normalize_ffd_scoring_mode(value):
    mode = str(value or COMPONENT).strip().upper()
    if mode not in SUPPORTED_FFD_SCORING_MODES:
        raise ValueError(
            "PROGRESSIVE_FFD_SCORING_MODE must be one of "
            f"{SUPPORTED_FFD_SCORING_MODES}, got {value!r}"
        )
    return mode


def _lookup_column_index(columns, x, tol=1.0e-10):
    x = float(x)
    for index, value in enumerate(columns):
        if abs(float(value) - x) <= tol:
            return int(index)
    raise FFDTangentError(
        f"Active FFD column {x:.16g} is absent from geometric columns {columns}"
    )


def _validate_state_matrix(matrix, label):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise FFDTangentError(f"{label} tangent matrix must be two-dimensional")
    if not np.all(np.isfinite(matrix)):
        raise FFDTangentError(f"{label} tangent matrix contains non-finite values")
    return matrix


def build_ffd_tangent_state(mesh_path, marker, active_by_side, opts):
    """Build the complete Cartesian marker Jacobian for the active FFD DVs."""

    mesh_path = os.path.abspath(str(mesh_path))
    mesh = read_su2_mesh(mesh_path)
    if int(mesh.get("ndime", 2)) != 2:
        raise FFDTangentError("Virtual FFD tangent scoring currently requires a 2D mesh")

    marker_tag, node_ids, closed = extract_marker_nodes(mesh, marker)
    if not node_ids:
        raise FFDTangentError(f"Marker {marker!r} contains no nodes")
    points = mesh["points"]
    x_le, x_te, chord = infer_chord(points, node_ids, {"mode": "auto"})
    x_over_c = np.asarray(
        [(float(points[node_id][0]) - x_le) / chord for node_id in node_ids],
        dtype=float,
    )
    coordinates = np.asarray(
        [[float(points[node_id][0]), float(points[node_id][1])] for node_id in node_ids],
        dtype=float,
    )
    row_by_node = {int(node_id): index for index, node_id in enumerate(node_ids)}

    active_sides = tuple(
        side for side in ("UPPER", "LOWER") if side in active_by_side
    )
    if not active_sides:
        raise FFDTangentError("Virtual FFD tangent scoring requires an active side")
    normalized_active = {
        side: sorted(float(value) for value in active_by_side[side])
        for side in active_sides
    }
    records = ordered_ffd_records(normalized_active, active_sides)
    if not records:
        raise FFDTangentError("Virtual FFD tangent state has no active DVs")

    with open(mesh_path, "r") as stream:
        lines = stream.readlines()

    boxes = {}
    embedded_nodes = {}
    for side in active_sides:
        box_tag = (
            str(opts.get("ffd_upper_box_tag", "UPPER_BOX"))
            if side == "UPPER"
            else str(opts.get("ffd_lower_box_tag", "LOWER_BOX"))
        )
        box = _parse_existing_dual_box(lines, box_tag)
        templates = _parse_curved_surface_lines(
            box["surface_block"],
            points,
            box["columns"],
            box["control_y"],
            box["z_planes"],
            marker_tag,
            1.0e-10 * max(1.0, float(chord)),
            box["blending_spec"],
        )
        boxes[side] = box
        embedded_nodes[side] = templates

    matrix = np.zeros((2 * len(node_ids), len(records)), dtype=float)
    scale = float(opts.get("scale", 1.0))
    for dv_index, (side, x) in enumerate(records):
        box = boxes[side]
        control_index = _lookup_column_index(box["columns"], x)
        control_row = 1 if side == "UPPER" else 0
        direction_y = 1.0 if side == "UPPER" else -1.0
        for node_id, template in embedded_nodes[side].items():
            if int(node_id) not in row_by_node:
                continue
            u, v, w = [float(value) for value in template["old_uvw"]]
            along = basis_values(
                len(box["columns"]),
                u,
                box["blending_spec"],
                axis=0,
            )[control_index]
            across = basis_values(
                2,
                v,
                box["blending_spec"],
                axis=1,
            )[control_row]
            # FFD_CONTROL_POINT_2D moves the same point on every k plane, so
            # the spanwise partition of unity reduces to one.
            _ = w
            marker_index = row_by_node[int(node_id)]
            matrix[2 * marker_index + 1, dv_index] = (
                scale * direction_y * float(along) * float(across)
            )

    matrix = _validate_state_matrix(matrix, "FFD")
    return {
        "mesh_path": mesh_path,
        "marker": marker_tag,
        "node_ids": [int(value) for value in node_ids],
        "closed": bool(closed),
        "coordinates": coordinates,
        "x_over_c": x_over_c,
        "x_le": float(x_le),
        "x_te": float(x_te),
        "chord": float(chord),
        "records": list(records),
        "active_by_side": normalized_active,
        "matrix": matrix,
        "boxes": boxes,
        "embedded_nodes": embedded_nodes,
    }


def load_surface_sensitivity_vector(filename, tangent_state):
    """Return the raw Cartesian SU2 surface sensitivity in marker order."""

    metadata = [{"node_id": node_id} for node_id in tangent_state["node_ids"]]
    sensitivities = read_sensitivity_file(str(filename))
    aligned = match_sensitivities_to_metadata(metadata, sensitivities)
    values = np.zeros(2 * len(aligned), dtype=float)
    for index, record in enumerate(aligned):
        sx = record.get("sensitivity_x")
        sy = record.get("sensitivity_y")
        if sx is None or sy is None:
            raise FFDTangentError(
                f"{filename} must contain complete Sensitivity_x/Sensitivity_y columns"
            )
        values[2 * index] = float(sx)
        values[2 * index + 1] = float(sy)
    if not np.all(np.isfinite(values)):
        raise FFDTangentError(f"{filename} contains non-finite sensitivities")
    return values


def _orthonormal_space(matrix, rank_rtol=RANK_RTOL, rank_atol=RANK_ATOL):
    matrix = _validate_state_matrix(matrix, "SVD input")
    if matrix.shape[1] == 0:
        return np.zeros((matrix.shape[0], 0), dtype=float), np.asarray([], dtype=float)
    norms = np.linalg.norm(matrix, axis=0)
    keep = norms > float(rank_atol)
    if not np.any(keep):
        return np.zeros((matrix.shape[0], 0), dtype=float), np.asarray([], dtype=float)
    balanced = matrix[:, keep] / norms[keep]
    u, singular, _vh = np.linalg.svd(balanced, full_matrices=False)
    tolerance = max(float(rank_atol), float(rank_rtol) * float(singular[0]))
    rank = int(np.sum(singular > tolerance))
    return u[:, :rank], singular[:rank]


def _nullspace(matrix, rank_rtol=RANK_RTOL, rank_atol=RANK_ATOL):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise FFDTangentError("Nullspace input must be two-dimensional")
    if matrix.shape[1] == 0:
        return np.zeros((0, 0), dtype=float), 0
    _u, singular, vh = np.linalg.svd(matrix, full_matrices=True)
    if singular.size:
        tolerance = max(float(rank_atol), float(rank_rtol) * float(singular[0]))
        rank = int(np.sum(singular > tolerance))
    else:
        rank = 0
    return vh[rank:, :].T, rank


def compare_tangent_spaces(
    baseline_state,
    candidate_state,
    signal,
    candidate_x,
    locality_radius=LOCALITY_RADIUS,
):
    """Compare two full FFD tangent spaces using a raw discrete covector."""

    if baseline_state["node_ids"] != candidate_state["node_ids"]:
        raise FFDTangentError("Baseline/candidate marker node ordering differs")
    d0 = _validate_state_matrix(baseline_state["matrix"], "baseline")
    dt = _validate_state_matrix(candidate_state["matrix"], "candidate")
    signal = np.asarray(signal, dtype=float).reshape(-1)
    if signal.size != d0.shape[0] or signal.size != dt.shape[0]:
        raise FFDTangentError(
            "Surface signal dimension does not match baseline/candidate tangent rows"
        )
    if not np.all(np.isfinite(signal)):
        raise FFDTangentError("Surface scoring signal contains non-finite values")

    u0, sigma0 = _orthonormal_space(d0)
    ut, sigmat = _orthonormal_space(dt)
    energy0 = float(np.linalg.norm(u0.T @ signal) ** 2)
    energyt = float(np.linalg.norm(ut.T @ signal) ** 2)
    signal_energy = float(signal @ signal)
    score_net = energyt - energy0

    cross = u0.T @ ut
    if cross.size:
        _uc, principal_cosines, _vhc = np.linalg.svd(cross, full_matrices=False)
    else:
        principal_cosines = np.asarray([], dtype=float)
    rank_current = int(u0.shape[1])
    rank_candidate = int(ut.shape[1])
    if rank_current:
        overlap_sq = float(np.linalg.norm(cross, "fro") ** 2)
        nesting_rms = math.sqrt(max(0.0, 1.0 - overlap_sq / rank_current))
        sigma_min = (
            float(principal_cosines[-1])
            if principal_cosines.size >= rank_current
            else 0.0
        )
        nesting_max = math.sqrt(max(0.0, 1.0 - sigma_min * sigma_min))
    else:
        nesting_rms = 0.0
        nesting_max = 0.0

    vplus, overlap_rank = _nullspace(cross)
    uplus = ut @ vplus
    score_pure = float(np.linalg.norm(uplus.T @ signal) ** 2)
    leverage_rows = (
        np.sum(uplus * uplus, axis=1)
        if uplus.shape[1]
        else np.zeros(signal.size, dtype=float)
    )
    leverage_nodes = leverage_rows.reshape((-1, 2)).sum(axis=1)
    leverage_sum = float(np.sum(leverage_nodes))
    if leverage_sum > 0.0:
        xoc = np.asarray(candidate_state["x_over_c"], dtype=float)
        locality = float(
            np.sum(leverage_nodes[np.abs(xoc - float(candidate_x)) <= locality_radius])
            / leverage_sum
        )
        innovation_center = float(np.sum(xoc * leverage_nodes) / leverage_sum)
    else:
        locality = 0.0
        innovation_center = math.nan

    normalizer = signal_energy + 1.0e-300
    return {
        "energy_current": energy0,
        "energy_candidate": energyt,
        "signal_energy": signal_energy,
        "score_net": float(score_net),
        "score_net_normalized": float(score_net / normalizer),
        "score_pure": float(score_pure),
        "score_pure_normalized": float(score_pure / normalizer),
        "rank_current": rank_current,
        "rank_candidate": rank_candidate,
        "rank_gain": rank_candidate - rank_current,
        "overlap_rank": int(overlap_rank),
        "pure_rank": int(uplus.shape[1]),
        "nesting_rms": float(nesting_rms),
        "nesting_max": float(nesting_max),
        "principal_cosines": principal_cosines.tolist(),
        "sigma_current": sigma0.tolist(),
        "sigma_candidate": sigmat.tolist(),
        "locality_radius": float(locality_radius),
        "locality": locality,
        "innovation_center_x": innovation_center,
    }


def project_surface_field(tangent_state, field):
    field = np.asarray(field, dtype=float).reshape(-1)
    matrix = tangent_state["matrix"]
    if field.size != matrix.shape[0]:
        raise FFDTangentError("Surface field size does not match FFD tangent state")
    return matrix.T @ field


def airfoil_area_value_and_field(tangent_state):
    if len(tangent_state["node_ids"]) < 3:
        raise FFDTangentError("AIRFOIL_AREA requires at least three marker nodes")
    xy = np.asarray(tangent_state["coordinates"], dtype=float)
    x = xy[:, 0]
    y = xy[:, 1]
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    x_prev = np.roll(x, 1)
    y_prev = np.roll(y, 1)
    signed = 0.5 * float(np.sum(x * y_next - x_next * y))
    orientation = -1.0 if signed < 0.0 else 1.0
    darea_dx = 0.5 * (y_next - y_prev) * orientation
    darea_dy = 0.5 * (x_prev - x_next) * orientation
    field = np.empty(2 * len(x), dtype=float)
    field[0::2] = darea_dx
    field[1::2] = darea_dy
    return abs(float(signed)), field


def _rotate_closed_section_from_trailing_edge(coordinates):
    """Match SU2_GEO's trailing-edge-to-trailing-edge section ordering."""

    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise FFDTangentError("AIRFOIL_THICKNESS requires planar coordinates")
    if len(coordinates) < 5:
        raise FFDTangentError("AIRFOIL_THICKNESS requires at least five points")

    x = coordinates[:, 0]
    xmax = float(np.max(x))
    candidates = np.where(
        np.abs(x - xmax) <= 1.0e-12 * max(1.0, abs(xmax))
    )[0]
    # A blunt trailing edge can provide two candidates.  SU2's section starts
    # from one trailing-edge endpoint; choose deterministically by |y| then id.
    start = int(
        min(candidates.tolist(), key=lambda index: (abs(coordinates[index, 1]), index))
    )
    order = np.concatenate(
        (np.arange(start, len(coordinates)), np.arange(0, start))
    )
    return coordinates[order], order


def _su2_airfoil_max_thickness(coordinates):
    """Port the 2-D criterion used by CPhysicalGeometry::Compute_MaxThickness."""

    section, order = _rotate_closed_section_from_trailing_edge(coordinates)
    trailing = section[0]
    distances = np.linalg.norm(section - trailing, axis=1)
    leading_index = int(np.argmax(distances))
    max_distance = float(distances[leading_index])
    if max_distance <= 0.0:
        raise FFDTangentError("AIRFOIL_THICKNESS found a zero-length chord")

    translated = section - trailing
    denominator = float(trailing[0] - section[leading_index, 0])
    numerator = float(section[leading_index, 1] - trailing[1])
    angle = math.atan2(numerator, denominator)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotated = np.empty_like(translated)
    rotated[:, 0] = translated[:, 0] * cosine - translated[:, 1] * sine
    rotated[:, 1] = translated[:, 1] * cosine + translated[:, 0] * sine

    segment = np.diff(rotated, axis=0)
    segment_length = np.linalg.norm(segment, axis=1)
    if np.any(segment_length <= 0.0):
        raise FFDTangentError(
            "AIRFOIL_THICKNESS found duplicate consecutive section points"
        )
    # For a section in the x-y plane and a positive spanwise binormal, the
    # normal component used by SU2 is the chordwise tangent component.
    normal_vertical = segment[:, 0] / segment_length
    point_indices = np.arange(1, len(rotated), dtype=int)
    spline_mask = (normal_vertical >= 0.0) & (
        np.abs(rotated[1:, 0]) > max_distance * 0.01
    )
    spline_indices = point_indices[spline_mask]
    evaluation_indices = point_indices[normal_vertical < 0.0]
    if len(spline_indices) < 2 or len(evaluation_indices) == 0:
        raise FFDTangentError(
            "AIRFOIL_THICKNESS could not separate interpolation/evaluation sides"
        )

    spline_indices = spline_indices[
        np.argsort(rotated[spline_indices, 0], kind="stable")
    ]
    spline_x = rotated[spline_indices, 0]
    spline_y = rotated[spline_indices, 1]
    if np.any(np.diff(spline_x) <= 0.0):
        raise FFDTangentError(
            "AIRFOIL_THICKNESS interpolation side has repeated abscissae"
        )
    slope_first = float(
        (spline_y[1] - spline_y[0]) / (spline_x[1] - spline_x[0])
    )
    slope_last = float(
        (spline_y[-1] - spline_y[-2]) / (spline_x[-1] - spline_x[-2])
    )
    spline = CubicSpline(
        spline_x,
        spline_y,
        bc_type=((1, slope_first), (1, slope_last)),
        extrapolate=True,
    )

    best_value = -1.0
    best_index = None
    for index in evaluation_indices:
        value = abs(float(rotated[index, 1] - spline(rotated[index, 0])))
        # C++ uses a strict comparison, therefore the first exact tie wins.
        if value > best_value:
            best_value = value
            best_index = int(index)
    if best_index is None or not math.isfinite(best_value):
        raise FFDTangentError("AIRFOIL_THICKNESS produced no finite value")

    original_index = int(order[best_index])
    return float(best_value), {
        "section_order": order.tolist(),
        "trailing_edge_node_index": int(order[0]),
        "leading_edge_node_index": int(order[leading_index]),
        "selected_node_index": original_index,
        "selected_section_index": best_index,
        "selected_x": float(coordinates[original_index, 0]),
        "spline_node_indices": [int(order[index]) for index in spline_indices],
        "evaluation_node_indices": [
            int(order[index]) for index in evaluation_indices
        ],
        "rotation_angle_rad": float(angle),
        "chord": max_distance,
    }


def airfoil_thickness_value_and_field(tangent_state, fd_relative_step=1.0e-7):
    """Return SU2-compatible maximum thickness and its Cartesian nodal field.

    The value follows ``CPhysicalGeometry::Compute_MaxThickness``: chord
    alignment, normal-based side selection, one-percent trailing-edge
    exclusion, clamped cubic interpolation and strict first-wins tie handling.
    The raw nodal field is differentiated centrally in Python.  This happens
    once per level; candidate scoring then only projects the stored field.
    """

    if not tangent_state["closed"]:
        raise FFDTangentError("AIRFOIL_THICKNESS requires a closed marker")
    coordinates = np.asarray(tangent_state["coordinates"], dtype=float)
    value, diagnostics = _su2_airfoil_max_thickness(coordinates)
    step = float(fd_relative_step) * max(
        abs(float(tangent_state.get("chord", diagnostics["chord"]))),
        1.0e-12,
    )
    if not math.isfinite(step) or step <= 0.0:
        raise FFDTangentError("AIRFOIL_THICKNESS finite-difference step is invalid")

    field = np.zeros(2 * len(coordinates), dtype=float)
    for point_index in range(len(coordinates)):
        # FFD_CONTROL_POINT_2D has no x component.  Retaining an explicit zero
        # x field avoids unnecessary work without changing any projection.
        plus = coordinates.copy()
        minus = coordinates.copy()
        plus[point_index, 1] += step
        minus[point_index, 1] -= step
        plus_value, _ = _su2_airfoil_max_thickness(plus)
        minus_value, _ = _su2_airfoil_max_thickness(minus)
        field[2 * point_index + 1] = (plus_value - minus_value) / (2.0 * step)

    selected_x = float(diagnostics["selected_x"])
    diagnostics.update(
        {
            "x": selected_x,
            "x_over_c": float(
                (selected_x - float(tangent_state["x_le"]))
                / float(tangent_state["chord"])
            ),
            "finite_difference_step": step,
            "field_x_components": "ZERO_FOR_FFD_CONTROL_POINT_2D",
            "criterion": "SU2_COMPUTE_MAX_THICKNESS",
        }
    )
    return value, field, diagnostics


def load_design_gradient_csv(filename):
    """Load the GRADIENT column written by SU2_DOT/SU2_GEO."""

    with open(filename, newline="") as stream:
        rows = list(csv.reader(stream))
    if not rows:
        raise FFDTangentError(f"Empty gradient CSV: {filename}")
    header = [str(value).strip().strip('"').upper() for value in rows[0]]
    if "GRADIENT" not in header:
        raise FFDTangentError(f"Gradient CSV has no GRADIENT column: {filename}")
    gradient_index = header.index("GRADIENT")
    values = []
    for row in rows[1:]:
        if len(row) <= gradient_index or not str(row[gradient_index]).strip():
            continue
        values.append(float(str(row[gradient_index]).strip()))
    result = np.asarray(values, dtype=float)
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise FFDTangentError(f"Gradient CSV has no finite values: {filename}")
    return result


def validate_surface_projection(
    tangent_state,
    field,
    reference_gradient,
    absolute_tolerance=5.0e-6,
):
    """Validate a reconstructed design gradient against a saved SU2 gradient."""

    reconstructed = np.asarray(
        project_surface_field(tangent_state, field),
        dtype=float,
    )
    reference = np.asarray(reference_gradient, dtype=float).reshape(-1)
    if reference.shape != reconstructed.shape:
        raise FFDTangentError(
            "Surface reconstruction/reference gradient size mismatch: "
            f"{reconstructed.size} != {reference.size}"
        )
    error = np.abs(reconstructed - reference)
    maximum = float(np.max(error)) if error.size else 0.0
    passed = bool(maximum <= float(absolute_tolerance))
    return {
        "reconstructed": reconstructed.tolist(),
        "reference": reference.tolist(),
        "absolute_error": error.tolist(),
        "max_absolute_error": maximum,
        "absolute_tolerance": float(absolute_tolerance),
        "passed": passed,
    }


def internal_constraint_field(spec, current_value, raw_function_field):
    sign = str(spec.get("sign", "")).strip()
    target = spec.get("target")
    if target is None:
        raise FFDTangentError(
            f"Constraint {spec.get('name')} has no finite target value"
        )
    target = float(target)
    current_value = float(current_value)
    if sign == ">":
        field_sign = 1.0
        c_value = current_value - target
        bounds = (0.0, np.inf)
        representation = "F-target>=0"
    elif sign == "<":
        field_sign = -1.0
        c_value = target - current_value
        bounds = (0.0, np.inf)
        representation = "target-F>=0"
    elif sign == "=":
        field_sign = 1.0
        c_value = current_value - target
        bounds = (-np.inf, np.inf)
        representation = "F-target=0"
    else:
        raise FFDTangentError(
            f"Constraint {spec.get('name')} has unsupported sign {sign!r}"
        )
    raw_function_field = np.asarray(raw_function_field, dtype=float).reshape(-1)
    return {
        "field": field_sign * raw_function_field,
        "field_sign": field_sign,
        "c_value": float(c_value),
        "lambda_lower": float(bounds[0]),
        "lambda_upper": float(bounds[1]),
        "representation": representation,
    }


def fit_surface_ikkt_signal(objective_field, constraint_records, tangent_state):
    objective_field = np.asarray(objective_field, dtype=float).reshape(-1)
    q_obj = project_surface_field(tangent_state, objective_field)
    if not constraint_records:
        return objective_field.copy(), np.zeros(0, dtype=float), {
            "status": "no_active_constraints",
            "objective_gradient": q_obj.tolist(),
            "objective_gradient_norm": float(np.linalg.norm(q_obj)),
            "constraint_gradients": {},
            "lambdas": [],
            "residual_gradient": q_obj.tolist(),
            "residual_gradient_norm": float(np.linalg.norm(q_obj)),
        }

    fields = [np.asarray(record["field"], dtype=float) for record in constraint_records]
    q_constraints = [project_surface_field(tangent_state, field) for field in fields]
    matrix = np.column_stack(q_constraints)
    lower = np.asarray(
        [float(record["lambda_lower"]) for record in constraint_records],
        dtype=float,
    )
    upper = np.asarray(
        [float(record["lambda_upper"]) for record in constraint_records],
        dtype=float,
    )
    result = lsq_linear(matrix, q_obj, bounds=(lower, upper), lsmr_tol="auto")
    if not bool(result.success):
        raise FFDTangentError(
            f"Surface IKKT multiplier fit failed: {result.message}"
        )
    lambdas = np.asarray(result.x, dtype=float)
    surface_matrix = np.column_stack(fields)
    residual_field = objective_field - surface_matrix @ lambdas
    residual_gradient = q_obj - matrix @ lambdas
    diagnostics = {
        "status": "ok",
        "objective_gradient": q_obj.tolist(),
        "objective_gradient_norm": float(np.linalg.norm(q_obj)),
        "constraint_gradients": {
            str(record["name"]): gradient.tolist()
            for record, gradient in zip(constraint_records, q_constraints)
        },
        "constraint_matrix_rank": int(np.linalg.matrix_rank(matrix)),
        "constraint_matrix_condition": float(np.linalg.cond(matrix)),
        "lambda_lower": lower.tolist(),
        "lambda_upper": upper.tolist(),
        "lambdas": lambdas.tolist(),
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "active_mask": [int(value) for value in result.active_mask.tolist()],
        "residual_gradient": residual_gradient.tolist(),
        "residual_gradient_norm": float(np.linalg.norm(residual_gradient)),
        "relative_residual_gradient_norm": float(np.linalg.norm(residual_gradient))
        / max(float(np.linalg.norm(q_obj)), 1.0e-16),
        "surface_residual_norm": float(np.linalg.norm(residual_field)),
    }
    return residual_field, lambdas, diagnostics
