#!/usr/bin/env python

"""Shared node filtering for surface-based progressive scoring.

The helpers in this module build a reduced *scoring view* of a tangent state.
They never mutate the complete tangent state or the full surface fields used
outside candidate ranking.
"""

import math

import numpy as np


class SurfaceScoringMaskError(RuntimeError):
    pass


def parse_te_closure_node_eps(value, key):
    """Return a validated trailing-edge exclusion width in chord units."""

    try:
        eps = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a finite number in [0, 1)") from exc
    if not math.isfinite(eps) or eps < 0.0 or eps >= 1.0:
        raise ValueError(f"{key} must be a finite number in [0, 1)")
    return eps


def build_surface_scoring_view(tangent_state, te_closure_node_eps=0.0):
    """Filter TE nodes and their Cartesian rows from a tangent state copy.

    A positive ``te_closure_node_eps`` keeps nodes satisfying
    ``x/c < 1 - eps``.  Zero is an exact compatibility mode and keeps every
    node, including the geometric trailing-edge node.
    """

    eps = parse_te_closure_node_eps(
        te_closure_node_eps,
        "surface scoring TE closure-node epsilon",
    )
    node_ids = [int(value) for value in tangent_state.get("node_ids", [])]
    x_over_c = np.asarray(tangent_state.get("x_over_c", []), dtype=float).reshape(-1)
    matrix = np.asarray(tangent_state.get("matrix"), dtype=float)
    node_count = len(node_ids)
    if node_count == 0:
        raise SurfaceScoringMaskError("Surface scoring state contains no nodes")
    if x_over_c.size != node_count or not np.all(np.isfinite(x_over_c)):
        raise SurfaceScoringMaskError(
            "Surface scoring x/c values do not match the marker nodes"
        )
    if matrix.ndim != 2 or matrix.shape[0] != 2 * node_count:
        raise SurfaceScoringMaskError(
            "Surface scoring tangent rows must contain one x/y pair per node"
        )
    if not np.all(np.isfinite(matrix)):
        raise SurfaceScoringMaskError(
            "Surface scoring tangent matrix contains non-finite values"
        )

    cutoff = 1.0 - eps
    node_mask = (
        np.asarray(x_over_c < cutoff, dtype=bool)
        if eps > 0.0
        else np.ones(node_count, dtype=bool)
    )
    if not np.any(node_mask):
        raise SurfaceScoringMaskError(
            "Trailing-edge surface scoring mask removed every marker node"
        )
    row_mask = np.repeat(node_mask, 2)
    kept_indices = np.flatnonzero(node_mask)
    removed_x = x_over_c[~node_mask]

    view = dict(tangent_state)
    view["node_ids"] = [node_ids[index] for index in kept_indices]
    view["x_over_c"] = x_over_c[node_mask].copy()
    view["matrix"] = matrix[row_mask, :].copy()

    if "coordinates" in tangent_state:
        coordinates = np.asarray(tangent_state["coordinates"], dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[0] != node_count:
            raise SurfaceScoringMaskError(
                "Surface scoring coordinates do not match the marker nodes"
            )
        view["coordinates"] = coordinates[node_mask, :].copy()
    if "sides" in tangent_state:
        sides = list(tangent_state["sides"])
        if len(sides) != node_count:
            raise SurfaceScoringMaskError(
                "Surface scoring side labels do not match the marker nodes"
            )
        view["sides"] = [sides[index] for index in kept_indices]

    diagnostics = {
        "enabled": bool(eps > 0.0),
        "te_closure_node_eps": float(eps),
        "kept_rule": (
            f"x_over_c < {cutoff:.16g}" if eps > 0.0 else "all marker nodes"
        ),
        "cutoff_x_over_c": float(cutoff) if eps > 0.0 else None,
        "node_count_total": int(node_count),
        "node_count_kept": int(np.count_nonzero(node_mask)),
        "node_count_removed": int(np.count_nonzero(~node_mask)),
        "cartesian_row_count_total": int(2 * node_count),
        "cartesian_row_count_kept": int(np.count_nonzero(row_mask)),
        "removed_x_over_c_min": (
            float(np.min(removed_x)) if removed_x.size else None
        ),
        "removed_x_over_c_max": (
            float(np.max(removed_x)) if removed_x.size else None
        ),
    }
    return view, node_mask, diagnostics


def mask_surface_field(field, node_mask, label="surface field"):
    """Apply a per-node mask to an interleaved Cartesian surface field."""

    node_mask = np.asarray(node_mask, dtype=bool).reshape(-1)
    values = np.asarray(field, dtype=float).reshape(-1)
    expected = 2 * node_mask.size
    if values.size != expected:
        raise SurfaceScoringMaskError(
            f"{label} has {values.size} entries; expected {expected}"
        )
    if not np.all(np.isfinite(values)):
        raise SurfaceScoringMaskError(f"{label} contains non-finite values")
    return values[np.repeat(node_mask, 2)].copy()


def mask_surface_constraint_records(records, node_mask):
    """Copy constraint records while filtering only their surface fields."""

    masked = []
    for record in records:
        item = dict(record)
        item["field"] = mask_surface_field(
            record["field"],
            node_mask,
            label=f"constraint {record.get('name', '<unnamed>')} field",
        )
        masked.append(item)
    return masked


__all__ = [
    "SurfaceScoringMaskError",
    "parse_te_closure_node_eps",
    "build_surface_scoring_view",
    "mask_surface_field",
    "mask_surface_constraint_records",
]
