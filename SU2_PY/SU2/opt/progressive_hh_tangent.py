#!/usr/bin/env python

"""Physical tangent-space utilities for progressive Hicks--Henne scoring."""

import math
import os

import numpy as np

from SU2.opt.bspline_def import (
    classify_sides,
    extract_marker_nodes,
    infer_chord,
    read_su2_mesh,
)
from SU2.opt.progressive_ffd_tangent import (
    compare_tangent_spaces,
    load_design_gradient_csv,
    load_surface_sensitivity_vector,
    project_surface_field,
    validate_surface_projection,
)


COMPONENT = "COMPONENT"
VIRTUAL_TANGENT = "VIRTUAL_TANGENT"
SUPPORTED_HH_SCORING_MODES = (COMPONENT, VIRTUAL_TANGENT)


class HHTangentError(RuntimeError):
    pass


def _config_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().upper()
    if text in ("YES", "TRUE", "1", "ON"):
        return True
    if text in ("NO", "FALSE", "0", "OFF"):
        return False
    raise ValueError(
        "HICKS_HENNE_T2_BY_CENTER must be a YES/NO boolean value; "
        f"got {value!r}"
    )


def normalize_hh_scoring_mode(value):
    mode = str(value or COMPONENT).strip().upper()
    if mode not in SUPPORTED_HH_SCORING_MODES:
        raise ValueError(
            "PROGRESSIVE_HH_SCORING_MODE must be one of "
            f"{SUPPORTED_HH_SCORING_MODES}, got {value!r}"
        )
    return mode


def hicks_henne_t2_policy(config=None):
    """Return the validated native HH ``t2`` policy from a config-like object."""

    config = config or {}
    by_center = _config_bool(config.get("HICKS_HENNE_T2_BY_CENTER", "NO"))
    if not by_center:
        uniform = float(config.get("HICKS_HENNE_T2", 1.0))
        if not math.isfinite(uniform) or uniform <= 0.0:
            raise ValueError(
                "HICKS_HENNE_T2 must be finite and greater than zero"
            )
        return {
            "uniform": uniform,
            "by_center": False,
            "forward": None,
            "aft": None,
            "switch_x": None,
        }

    forward = float(config.get("HICKS_HENNE_T2_FORWARD", 3.0))
    aft = float(config.get("HICKS_HENNE_T2_AFT", 1.0))
    switch_x = float(config.get("HICKS_HENNE_T2_SWITCH_X", 0.5))
    for key, value in (
        ("HICKS_HENNE_T2_FORWARD", forward),
        ("HICKS_HENNE_T2_AFT", aft),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{key} must be finite and greater than zero")
    if not math.isfinite(switch_x) or not 0.0 < switch_x < 1.0:
        raise ValueError(
            "HICKS_HENNE_T2_SWITCH_X must be finite and strictly between zero and one"
        )

    return {
        "uniform": None,
        "by_center": True,
        "forward": forward,
        "aft": aft,
        "switch_x": switch_x,
    }


def hicks_henne_t2_for_center(center, policy):
    center = float(center)
    if not math.isfinite(center) or not 0.0 < center < 1.0:
        raise ValueError("Hicks--Henne center must be finite and strictly in (0, 1)")
    if not bool(policy.get("by_center", False)):
        return float(policy["uniform"])
    if center <= float(policy["switch_x"]):
        return float(policy["forward"])
    return float(policy["aft"])


def hicks_henne_kernel(x, center, t2):
    """Evaluate the kernel used by ``CSurfaceMovement::SetHicksHenne``."""

    x = float(x)
    center = float(center)
    t2 = float(t2)
    if x <= 10.0 * np.finfo(float).eps:
        return 0.0
    if not 0.0 < center < 1.0:
        raise HHTangentError(f"Invalid Hicks--Henne center {center!r}")
    if not math.isfinite(t2) or t2 <= 0.0:
        raise HHTangentError(f"Invalid Hicks--Henne t2 {t2!r}")
    exponent = math.log10(0.5) / math.log10(center)
    sine = math.sin(math.pi * math.pow(x, exponent))
    # Native HH usage assumes chordwise coordinates in [0, 1].  A small
    # roundoff excursion is harmless; a genuine excursion would make a
    # non-integer exponent of a negative sine undefined in the C++ path too.
    if sine < 0.0 and sine > -1.0e-14:
        sine = 0.0
    value = math.pow(sine, t2)
    if not math.isfinite(value):
        raise HHTangentError(
            f"Non-finite Hicks--Henne kernel at x={x}, center={center}, t2={t2}"
        )
    return value


def _geometrically_closed(points, node_ids, parsed_closed, chord):
    if parsed_closed or len(node_ids) < 2:
        return bool(parsed_closed)
    first = points[node_ids[0]]
    last = points[node_ids[-1]]
    distance = math.hypot(
        float(first[0]) - float(last[0]),
        float(first[1]) - float(last[1]),
    )
    return bool(distance <= 1.0e-10 * max(1.0, float(chord)))


def _ordered_records(active_by_side):
    records = []
    for side in ("UPPER", "LOWER"):
        for center in sorted(float(value) for value in active_by_side.get(side, [])):
            records.append((side, center))
    return records


def build_hh_tangent_state(
    mesh_path,
    marker,
    active_by_side,
    config=None,
    *,
    scale=1.0,
    symmetry_mode="NONE",
    symmetry_sign=-1.0,
):
    """Build the Cartesian marker Jacobian of the active native HH DVs."""

    mesh_path = os.path.abspath(str(mesh_path))
    mesh = read_su2_mesh(mesh_path)
    if int(mesh.get("ndime", 2)) != 2:
        raise HHTangentError("Virtual HH tangent scoring currently requires a 2D mesh")

    marker_tag, node_ids, parsed_closed = extract_marker_nodes(mesh, marker)
    if not node_ids:
        raise HHTangentError(f"Marker {marker!r} contains no nodes")
    points = mesh["points"]
    x_le, x_te, chord = infer_chord(points, node_ids, {"mode": "auto"})
    coordinates = np.asarray(
        [[float(points[node_id][0]), float(points[node_id][1])] for node_id in node_ids],
        dtype=float,
    )
    x_over_c = np.asarray(
        [(float(points[node_id][0]) - x_le) / chord for node_id in node_ids],
        dtype=float,
    )
    geometrically_closed = _geometrically_closed(
        points, node_ids, parsed_closed, chord
    )
    normalized_active = {
        side: sorted(float(value) for value in active_by_side.get(side, []))
        for side in ("UPPER", "LOWER")
        if active_by_side.get(side, [])
    }
    active_side_set = set(normalized_active)
    if not geometrically_closed and active_side_set == {"UPPER"}:
        # Half-domain airfoils commonly expose one open TE-to-LE marker.  The
        # native code classifies every vertex through its outward normal; the
        # progressive surface contract tells us which single side it is.
        sides = ["upper"] * len(node_ids)
    elif not geometrically_closed and active_side_set == {"LOWER"}:
        sides = ["lower"] * len(node_ids)
    else:
        sides = classify_sides(
            node_ids,
            x_over_c.tolist(),
            coordinates[:, 1].tolist(),
            closed=geometrically_closed,
        )

    full_records = _ordered_records(normalized_active)
    if not full_records:
        raise HHTangentError("Virtual HH tangent state has no active DVs")

    policy = hicks_henne_t2_policy(config)
    scale = float(scale)
    if not math.isfinite(scale) or scale == 0.0:
        raise HHTangentError("HH tangent scale must be finite and non-zero")
    full_matrix = np.zeros((2 * len(node_ids), len(full_records)), dtype=float)
    t2_by_record = []
    for column, (side, center) in enumerate(full_records):
        t2 = hicks_henne_t2_for_center(center, policy)
        t2_by_record.append(t2)
        direction = 1.0 if side == "UPPER" else -1.0
        expected_side = "upper" if side == "UPPER" else "lower"
        for row, node_id in enumerate(node_ids):
            if sides[row] != expected_side:
                continue
            x_native = float(points[node_id][0])
            full_matrix[2 * row + 1, column] = (
                scale * direction * hicks_henne_kernel(x_native, center, t2)
            )

    symmetry_mode = str(symmetry_mode or "NONE").strip().upper()
    symmetry_sign = float(symmetry_sign)
    if symmetry_mode == "REDUCED":
        upper = normalized_active.get("UPPER", [])
        lower = normalized_active.get("LOWER", [])
        if len(upper) != len(lower) or any(
            abs(xu - xl) > 1.0e-12 for xu, xl in zip(upper, lower)
        ):
            raise HHTangentError(
                "Reduced symmetric HH tangent requires identical upper/lower centers"
            )
        n_pair = len(upper)
        matrix = full_matrix[:, :n_pair] + symmetry_sign * full_matrix[:, n_pair:]
        records = [("PAIR", float(center)) for center in upper]
        t2_records = [hicks_henne_t2_for_center(center, policy) for center in upper]
    elif symmetry_mode == "NONE":
        matrix = full_matrix
        records = full_records
        t2_records = t2_by_record
    else:
        raise HHTangentError(f"Unsupported HH symmetry mode {symmetry_mode!r}")

    if not np.all(np.isfinite(matrix)):
        raise HHTangentError("HH tangent matrix contains non-finite values")
    return {
        "mesh_path": mesh_path,
        "marker": marker_tag,
        "node_ids": [int(value) for value in node_ids],
        "closed": bool(geometrically_closed),
        "coordinates": coordinates,
        "x_over_c": x_over_c,
        "x_le": float(x_le),
        "x_te": float(x_te),
        "chord": float(chord),
        "sides": list(sides),
        "records": list(records),
        "t2_by_record": list(t2_records),
        "active_by_side": normalized_active,
        "matrix": matrix,
        "t2_policy": dict(policy),
        "symmetry_mode": symmetry_mode,
        "symmetry_sign": symmetry_sign,
    }


__all__ = [
    "COMPONENT",
    "VIRTUAL_TANGENT",
    "SUPPORTED_HH_SCORING_MODES",
    "HHTangentError",
    "normalize_hh_scoring_mode",
    "hicks_henne_t2_policy",
    "hicks_henne_t2_for_center",
    "hicks_henne_kernel",
    "build_hh_tangent_state",
    "compare_tangent_spaces",
    "load_design_gradient_csv",
    "load_surface_sensitivity_vector",
    "project_surface_field",
    "validate_surface_projection",
]
