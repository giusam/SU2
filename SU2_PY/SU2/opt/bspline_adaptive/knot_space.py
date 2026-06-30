"""Clamped knot-space helpers for adaptive B-spline refinement."""

import math

import numpy as np

from SU2.opt.bspline_dot import normalize_sensitivity_weighting
from SU2.opt.bspline_modes import (
    BSplineModeError,
    active_sides_from_surface_mode,
    clamped_basis_count,
    normalize_surface_mode,
    validate_mode_spec,
    validate_surface_mode_against_modes,
)

from .errors import BSplineAdaptiveError
from .models import ClampedKnotSpace, ClampedSideGroup
from .mode_utils import (
    _active_modes,
    _copy_global_metadata,
    evaluate_basis_matrix,
)

REFINE_MODE = "KNOT_INSERTION"

REFINE_STATE = "INITIAL_MESH_KEEP_DV"

def _rounded_knots(knots):
    return tuple(round(float(value), 14) for value in knots)

def _side_order(side):
    order = {"upper": 0, "lower": 1}
    return order.get(str(side).strip().lower(), 2)

def _bounds_signature(mode):
    bounds = mode.get("bounds")
    if bounds is None:
        return None
    if len(bounds) != 2:
        return tuple(bounds)
    return tuple(round(float(value), 14) for value in bounds)

def extract_clamped_knot_space(mode_spec, settings=None):
    settings = dict(settings or {})
    coupling = str(settings.get("symmetry_coupling", "NONE")).upper()
    try:
        spec = validate_mode_spec(mode_spec)
        surface_mode = normalize_surface_mode(settings.get("surface_mode", "BOTH"))
        validate_surface_mode_against_modes(spec, surface_mode)
    except BSplineModeError as exc:
        raise BSplineAdaptiveError(str(exc))

    if int(spec.get("dimension", 2)) != 2:
        raise BSplineAdaptiveError("KNOT_INSERTION v1 supports only dimension=2")
    if spec.get("normal_displacement", True) is not True:
        raise BSplineAdaptiveError("KNOT_INSERTION v1 requires normal_displacement=true")
    if normalize_sensitivity_weighting(settings.get("sensitivity_weighting", "NODAL")) != "NODAL":
        raise BSplineAdaptiveError("KNOT_INSERTION v1 supports only NODAL sensitivity weighting")

    by_side = {}
    for mode in _active_modes(spec):
        basis_type = str(mode.get("basis_type", "")).strip().lower()
        if basis_type != "clamped":
            raise BSplineAdaptiveError(
                "Only basis_type='clamped' is supported by the knot-insertion B-spline workflow."
            )
        degree = int(mode.get("degree", 3))
        if degree != 3:
            raise BSplineAdaptiveError("KNOT_INSERTION v1 supports only degree 3")
        side = str(mode.get("side", "")).strip().lower()
        if side not in ("upper", "lower"):
            raise BSplineAdaptiveError("KNOT_INSERTION requires active modes on side upper/lower")
        by_side.setdefault(side, []).append(dict(mode, active=True))

    if not by_side:
        raise BSplineAdaptiveError("KNOT_INSERTION requires at least one active clamped mode")
    if coupling in ("NORMAL_EQUAL", "NORMAL_OPPOSITE") and set(by_side) != {"upper", "lower"}:
        raise BSplineAdaptiveError(
            f"{coupling} KNOT_INSERTION requires paired upper and lower clamped bases"
        )
    expected_sides = set(active_sides_from_surface_mode(surface_mode))
    if surface_mode != "BOTH" and set(by_side) != expected_sides:
        raise BSplineAdaptiveError(
            f"BSPLINE_SURFACE_MODE={surface_mode} requires active side "
            f"{surface_mode.lower()!r}"
        )

    groups = {}
    for side, modes in by_side.items():
        modes = sorted(modes, key=lambda mode: int(mode["basis_index"]))
        degree = int(modes[0].get("degree", 3))
        knot_vector = _rounded_knots(modes[0].get("knot_vector", modes[0].get("knots")))
        n_basis = clamped_basis_count(degree, knot_vector)
        indices = [int(mode["basis_index"]) for mode in modes]
        expected = list(range(n_basis))
        if indices != expected:
            raise BSplineAdaptiveError(
                f"KNOT_INSERTION side {side!r} must contain complete basis indices {expected}; got {indices}"
            )

        for mode in modes:
            if int(mode.get("degree", 3)) != degree:
                raise BSplineAdaptiveError(f"KNOT_INSERTION side {side!r} mixes degrees")
            if _rounded_knots(mode.get("knot_vector", mode.get("knots"))) != knot_vector:
                raise BSplineAdaptiveError(f"KNOT_INSERTION side {side!r} mixes knot vectors")

        groups[side] = ClampedSideGroup(
            side=side,
            degree=degree,
            knot_vector=knot_vector,
            modes=modes,
        )

    sides = tuple(sorted(groups, key=_side_order))
    degree = groups[sides[0]].degree
    knot_vector = groups[sides[0]].knot_vector
    for side in sides:
        group = groups[side]
        if group.degree != degree:
            raise BSplineAdaptiveError("KNOT_INSERTION active clamped groups mix degrees")
        if group.knot_vector != knot_vector:
            raise BSplineAdaptiveError(
                "KNOT_INSERTION v1 requires all active clamped side groups to share the same knot vector"
            )

    return ClampedKnotSpace(
        spec=spec,
        groups=groups,
        sides=sides,
        degree=degree,
        knot_vector=knot_vector,
        coupling=coupling,
    )

def knot_insertion_spans(knot_vector, min_width=1.0e-8):
    knots = [float(value) for value in knot_vector]
    spans = []
    for left, right in zip(knots[:-1], knots[1:]):
        if float(right) - float(left) > float(min_width):
            spans.append((float(left), float(right), 0.5 * (float(left) + float(right))))
    return spans

def insert_knot_midpoint(knot_vector, span):
    inserted = float(span[2])
    new_knots = [float(value) for value in knot_vector]
    new_knots.append(inserted)
    new_knots.sort()
    return tuple(new_knots)

def _representative_bounds(group):
    counts = {}
    by_key = {}
    for mode in group.modes:
        key = _bounds_signature(mode)
        counts[key] = counts.get(key, 0) + 1
        by_key[key] = mode.get("bounds")
    best_key = max(counts, key=lambda key: (counts[key], key is not None))
    bounds = by_key[best_key]
    return list(bounds) if bounds is not None else None

def _mode_template_for_side(group, basis_index, knot_vector, coefficient=0.0):
    old_by_index = {int(mode["basis_index"]): mode for mode in group.modes}
    source = old_by_index.get(int(basis_index), group.modes[min(int(basis_index), len(group.modes) - 1)])
    mode = {
        key: value
        for key, value in source.items()
        if key
        not in (
            "id",
            "knot_vector",
            "knots",
            "basis_index",
            "coefficient",
            "active",
            "normalization_factor",
        )
    }
    mode.update(
        {
            "id": f"{group.side}_clamped_i{int(basis_index):03d}",
            "side": group.side,
            "basis_type": "clamped",
            "degree": group.degree,
            "knot_vector": [float(value) for value in knot_vector],
            "basis_index": int(basis_index),
            "coefficient": float(coefficient),
            "active": True,
        }
    )
    bounds = source.get("bounds")
    if bounds is None:
        bounds = _representative_bounds(group)
    if bounds is not None:
        mode["bounds"] = list(bounds)
    return mode

def regenerate_clamped_modes(space, knot_vector, coefficients_by_side=None):
    coefficients_by_side = coefficients_by_side or {}
    modes = []
    for side in space.sides:
        old_group = space.groups[side]
        n_basis = clamped_basis_count(space.degree, knot_vector)
        coeffs = coefficients_by_side.get(side, np.zeros(n_basis, dtype=float))
        if len(coeffs) != n_basis:
            raise BSplineAdaptiveError(
                f"expected {n_basis} transferred coefficients for side {side}, got {len(coeffs)}"
            )
        for basis_index in range(n_basis):
            modes.append(
                _mode_template_for_side(
                    old_group,
                    basis_index,
                    knot_vector,
                    coefficient=float(coeffs[basis_index]),
                )
            )
    return validate_mode_spec(_copy_global_metadata(space.spec, modes))

def _basis_for_spec_modes(spec, metadata):
    modes = _active_modes(spec)
    matrix = evaluate_basis_matrix(spec, modes, metadata)
    return modes, matrix

def reduced_basis_matrix_for_space(space, spec, metadata):
    modes, full_matrix = _basis_for_spec_modes(spec, metadata)
    if space.coupling == "NONE":
        return modes, full_matrix

    n_basis = clamped_basis_count(space.degree, _rounded_knots(modes[0]["knot_vector"]))
    by_side_index = {}
    for column, mode in enumerate(modes):
        side = str(mode.get("side", "")).strip().lower()
        by_side_index[(side, int(mode["basis_index"]))] = column

    columns = []
    reduced_modes = []
    lower_sign = 1.0 if space.coupling == "NORMAL_EQUAL" else -1.0
    for basis_index in range(n_basis):
        upper_column = by_side_index.get(("upper", basis_index))
        lower_column = by_side_index.get(("lower", basis_index))
        if upper_column is None or lower_column is None:
            raise BSplineAdaptiveError(
                f"{space.coupling} KNOT_INSERTION missing paired basis_index {basis_index}"
            )
        columns.append(full_matrix[:, upper_column] + lower_sign * full_matrix[:, lower_column])
        reduced_modes.append(
            {
                "id": f"paired_clamped_i{basis_index:03d}",
                "side": "paired",
                "basis_type": "clamped",
                "basis_index": basis_index,
                "degree": space.degree,
            }
        )
    return reduced_modes, np.column_stack(columns) if columns else np.zeros((len(metadata), 0))

def _expanded_coefficients_from_reduced(space, knot_vector, reduced_coeffs):
    lower_sign = 1.0 if space.coupling == "NORMAL_EQUAL" else -1.0
    reduced_coeffs = np.asarray(reduced_coeffs, dtype=float)
    return {
        "upper": reduced_coeffs.copy(),
        "lower": lower_sign * reduced_coeffs,
    }

def coefficient_vector_for_space(space, spec):
    modes = _active_modes(spec)
    if space.coupling == "NONE":
        return np.asarray(
            [float(mode.get("coefficient", 0.0)) for mode in modes],
            dtype=float,
        )

    n_basis = clamped_basis_count(space.degree, space.knot_vector)
    by_side_index = {}
    for mode in modes:
        by_side_index[(str(mode.get("side", "")).strip().lower(), int(mode["basis_index"]))] = float(
            mode.get("coefficient", 0.0)
        )
    lower_sign = 1.0 if space.coupling == "NORMAL_EQUAL" else -1.0
    values = []
    for basis_index in range(n_basis):
        upper = by_side_index[("upper", basis_index)]
        lower = by_side_index[("lower", basis_index)]
        values.append(0.5 * (upper + lower_sign * lower))
    return np.asarray(values, dtype=float)

def reduced_ndv_for_knot_space(space, knot_vector=None):
    knot_vector = space.knot_vector if knot_vector is None else knot_vector
    n_basis = clamped_basis_count(space.degree, knot_vector)
    if space.coupling in ("NORMAL_EQUAL", "NORMAL_OPPOSITE"):
        return int(n_basis)
    return int(n_basis) * len(space.sides)

def physical_ndv_for_knot_space(space, knot_vector=None):
    knot_vector = space.knot_vector if knot_vector is None else knot_vector
    return int(clamped_basis_count(space.degree, knot_vector)) * len(space.sides)

def _independent_side_specs(mode_spec):
    """Split a mode spec into one self-consistent sub-spec per active side.

    Each side keeps its own knot vector, so the per-side specs remain valid for
    ``extract_clamped_knot_space`` even after upper and lower have diverged.
    """
    spec = validate_mode_spec(mode_spec)
    by_side = {}
    for mode in _active_modes(spec):
        side = str(mode.get("side", "")).strip().lower()
        by_side.setdefault(side, []).append(dict(mode))
    return {
        side: validate_mode_spec(_copy_global_metadata(spec, modes))
        for side, modes in by_side.items()
    }

def extract_independent_side_spaces(mode_spec, settings):
    """Per-side clamped knot spaces for INDEPENDENT refinement (coupling=NONE)."""
    side_settings = dict(settings or {})
    side_settings["symmetry_coupling"] = "NONE"
    return {
        side: extract_clamped_knot_space(side_spec, side_settings)
        for side, side_spec in _independent_side_specs(mode_spec).items()
    }

def refinement_limit_ndv(mode_spec, settings):
    from SU2.opt.bspline_driver.reduction import build_reduced_variables

    reduced_variables, _warnings = build_reduced_variables(
        mode_spec,
        coupling=(settings or {}).get("symmetry_coupling", "NONE"),
    )
    return len(reduced_variables)

def _requested_knot_insertions(space, settings, available_spans):
    reduced_per_insertion = (
        1 if space.coupling in ("NORMAL_EQUAL", "NORMAL_OPPOSITE") else len(space.sides)
    )
    return _insertion_budget(
        reduced_ndv_for_knot_space(space),
        available_spans,
        reduced_per_insertion,
        settings,
    )

def _insertion_budget(current_reduced, available_spans, reduced_per_insertion, settings):
    """Resolve how many midpoint knot insertions to request for one refinement.

    Pure arithmetic shared by the coupled path (one ``space``) and the
    independent path (combined upper+lower budget with one knot per side and
    ``reduced_per_insertion == 1``).
    """
    available_spans = max(0, int(available_spans))
    reduced_per_insertion = max(1, int(reduced_per_insertion))
    current_reduced = int(current_reduced)
    if available_spans <= 0:
        return 0, {
            "mode": str(settings.get("nadd_mode", "GROWTH_RATIO")).upper(),
            "current_reduced_ndv": current_reduced,
            "target_reduced_ndv": current_reduced,
            "reduced_ndv_per_insertion": reduced_per_insertion,
            "requested_insertions": 0,
            "clamped": True,
            "clamp_reason": "no_valid_spans",
        }

    mode = str(settings.get("nadd_mode", "GROWTH_RATIO")).upper()
    explicit = settings.get("knot_insertions_per_refine", 1)
    auto = str(explicit).strip().upper() == "AUTO"
    target_reduced = current_reduced

    if not auto:
        requested = int(explicit)
        mode_for_log = "EXPLICIT"
        target_reduced = current_reduced + requested
    elif mode == "GROWTH_RATIO":
        growth_ratio = float(settings.get("growth_ratio", 2.0))
        if growth_ratio <= 1.0:
            target_reduced = current_reduced + 1
        else:
            target_reduced = int(math.ceil(growth_ratio * current_reduced))
        requested = max(
            1,
            int(math.ceil(max(1, target_reduced - current_reduced) / float(reduced_per_insertion))),
        )
        mode_for_log = "GROWTH_RATIO"
    elif mode == "FIXED":
        requested = max(1, int(settings.get("fixed_nadd", 1)))
        target_reduced = current_reduced + requested
        mode_for_log = "FIXED"
    else:
        raise BSplineAdaptiveError(f"unsupported nadd mode {mode!r}")

    limits = [available_spans]
    nfinal = settings.get("nfinal")
    if nfinal is not None:
        remaining_reduced = max(0, int(nfinal) - int(current_reduced))
        limits.append(remaining_reduced // max(1, reduced_per_insertion))
    if settings.get("batch_size_max") is not None:
        limits.append(max(1, int(settings.get("batch_size_max", 1))))

    limited = min([int(requested)] + [int(limit) for limit in limits])
    limited = max(0, limited)
    clamp_reasons = []
    if limited < int(requested):
        if limited == available_spans:
            clamp_reasons.append("valid_spans")
        if settings.get("batch_size_max") is not None and limited == int(settings.get("batch_size_max", 1)):
            clamp_reasons.append("batch_size_max")
        if nfinal is not None:
            clamp_reasons.append("nfinal")
    return limited, {
        "mode": mode_for_log,
        "current_reduced_ndv": current_reduced,
        "target_reduced_ndv": target_reduced,
        "reduced_ndv_per_insertion": int(reduced_per_insertion),
        "requested_insertions": int(requested),
        "available_spans": available_spans,
        "insertions": int(limited),
        "clamped": limited < int(requested),
        "clamp_reason": ",".join(sorted(set(clamp_reasons))),
    }
