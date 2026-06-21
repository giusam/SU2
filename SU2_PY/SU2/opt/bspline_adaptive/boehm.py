"""Boehm knot insertion and shape-transfer helpers."""

import math

import numpy as np

from SU2.opt.bspline_modes import mode_normalization_factor

from .errors import (
    BSplineAdaptiveError,
    _as_float,
)
from .knot_space import (
    _basis_for_spec_modes,
    regenerate_clamped_modes,
)
from .mode_utils import _active_modes

def _knot_multiplicity(knots, u, tol=1.0e-12):
    return sum(
        1
        for knot in knots
        if math.isclose(float(knot), float(u), rel_tol=0.0, abs_tol=float(tol))
    )

def _find_knot_span_for_insertion(knots, degree, coeff_count, u, tol=1.0e-12):
    knots = np.asarray(knots, dtype=float)
    degree = int(degree)
    coeff_count = int(coeff_count)
    n = coeff_count - 1
    if len(knots) != coeff_count + degree + 1:
        raise BSplineAdaptiveError(
            "invalid knot vector length for Boehm insertion: "
            f"knots={len(knots)} coeffs={coeff_count} degree={degree}"
        )

    lower = float(knots[degree])
    upper = float(knots[n + 1])
    u = float(u)
    if u < lower - float(tol) or u > upper + float(tol):
        raise BSplineAdaptiveError(
            f"cannot insert knot {u:.16g} outside [{lower:.16g}, {upper:.16g}]"
        )
    if math.isclose(u, upper, rel_tol=0.0, abs_tol=float(tol)):
        return n

    for k in range(degree, n + 1):
        if (
            u >= float(knots[k]) - float(tol)
            and u < float(knots[k + 1]) - float(tol)
        ):
            return k
    raise BSplineAdaptiveError(f"could not find a knot span for insertion at u={u:.16g}")

def _boehm_insert_once(knots, coeffs, degree, u, tol=1.0e-12):
    knots = np.asarray(knots, dtype=float)
    coeffs = np.asarray(coeffs, dtype=float)
    degree = int(degree)
    if coeffs.ndim != 1:
        raise BSplineAdaptiveError("Boehm insertion coefficients must be one-dimensional")
    if not np.all(np.isfinite(knots)) or not np.all(np.isfinite(coeffs)):
        raise BSplineAdaptiveError("Boehm insertion requires finite knots and coefficients")

    matching = [
        float(knot)
        for knot in knots
        if math.isclose(float(knot), float(u), rel_tol=0.0, abs_tol=float(tol))
    ]
    inserted = matching[0] if matching else float(u)
    multiplicity = _knot_multiplicity(knots, inserted, tol=tol)
    if multiplicity > degree:
        raise BSplineAdaptiveError(
            f"cannot insert knot {inserted:.16g}: current multiplicity "
            f"{multiplicity} exceeds degree {degree}"
        )

    n = len(coeffs) - 1
    k = _find_knot_span_for_insertion(
        knots,
        degree,
        len(coeffs),
        inserted,
        tol=tol,
    )
    new_coeffs = np.empty(len(coeffs) + 1, dtype=float)

    for index in range(0, k - degree + 1):
        new_coeffs[index] = coeffs[index]
    for index in range(k - multiplicity, n + 1):
        new_coeffs[index + 1] = coeffs[index]
    for index in range(k - degree + 1, k - multiplicity + 1):
        denominator = float(knots[index + degree] - knots[index])
        if abs(denominator) <= float(tol):
            raise BSplineAdaptiveError(
                "degenerate denominator during Boehm insertion at "
                f"u={inserted:.16g}, basis_index={index}"
            )
        alpha = (inserted - float(knots[index])) / denominator
        new_coeffs[index] = (
            alpha * coeffs[index] + (1.0 - alpha) * coeffs[index - 1]
        )

    new_knots = np.insert(knots, int(np.searchsorted(knots, inserted, side="right")), inserted)
    return new_knots, new_coeffs

def _boehm_insert_to_target_knots(
    old_knots,
    old_coeffs,
    degree,
    target_knots,
    tol=1.0e-12,
):
    current_knots = np.asarray(old_knots, dtype=float)
    current_coeffs = np.asarray(old_coeffs, dtype=float)
    target_knots = np.asarray(target_knots, dtype=float)
    if np.any(np.diff(current_knots) < -float(tol)):
        raise BSplineAdaptiveError("old knot vector must be nondecreasing")
    if np.any(np.diff(target_knots) < -float(tol)):
        raise BSplineAdaptiveError("target knot vector must be nondecreasing")
    if len(target_knots) < len(current_knots):
        raise BSplineAdaptiveError("target knot vector cannot remove knots")

    inserted_knots = []
    old_index = 0
    for target in target_knots:
        if old_index < len(current_knots) and math.isclose(
            float(current_knots[old_index]),
            float(target),
            rel_tol=0.0,
            abs_tol=float(tol),
        ):
            old_index += 1
            continue
        if (
            old_index < len(current_knots)
            and float(current_knots[old_index]) < float(target) - float(tol)
        ):
            raise BSplineAdaptiveError(
                "target knot vector is not an insertion-only refinement of the old vector"
            )
        inserted_knots.append(float(target))
    if old_index != len(current_knots):
        raise BSplineAdaptiveError(
            "target knot vector is not an insertion-only refinement of the old vector"
        )

    for inserted in inserted_knots:
        current_knots, current_coeffs = _boehm_insert_once(
            current_knots,
            current_coeffs,
            degree,
            inserted,
            tol=tol,
        )

    if len(current_knots) != len(target_knots) or not np.allclose(
        current_knots,
        target_knots,
        rtol=0.0,
        atol=float(tol),
    ):
        raise BSplineAdaptiveError(
            "Boehm insertion did not reproduce the requested target knot vector"
        )
    return current_coeffs

def _mode_normalization_factors(modes, spec):
    modes = list(modes)
    if not bool(spec.get("normalize_basis", True)):
        return np.ones(len(modes), dtype=float)
    class_shape = spec.get("class_shape", "sqrt_x_one_minus_x")
    class_shape_exponent = spec.get("class_shape_exponent", 0.5)
    factors = np.asarray(
        [
            mode_normalization_factor(
                mode,
                class_shape=class_shape,
                class_shape_exponent=class_shape_exponent,
            )
            for mode in modes
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(factors)) or np.any(factors <= 0.0):
        raise BSplineAdaptiveError("B-spline normalization factors must be finite and positive")
    return factors

def _boehm_transfer_side(group, new_modes, spec, new_knots):
    old_modes = sorted(group.modes, key=lambda mode: int(mode["basis_index"]))
    new_modes = sorted(new_modes, key=lambda mode: int(mode["basis_index"]))
    old_factors = _mode_normalization_factors(old_modes, spec)
    new_factors = _mode_normalization_factors(new_modes, spec)
    canonical_old = group.coefficients / old_factors
    canonical_new = _boehm_insert_to_target_knots(
        group.knot_vector,
        canonical_old,
        group.degree,
        new_knots,
    )
    if len(canonical_new) != len(new_factors):
        raise BSplineAdaptiveError(
            f"Boehm transfer for side {group.side!r} produced {len(canonical_new)} "
            f"coefficients for {len(new_factors)} modes"
        )
    return canonical_new * new_factors

def _check_transferred_coefficients_within_bounds(next_modes, settings, tol=1.0e-12):
    policy = str(settings.get("transfer_bound_policy", "ERROR")).strip().upper()
    if policy != "ERROR":
        raise BSplineAdaptiveError(
            "Only BSPLINE_TRANSFER_BOUND_POLICY=ERROR is currently implemented."
        )
    global_lower = settings.get("opt_bound_lower")
    global_upper = settings.get("opt_bound_upper")
    use_global_bounds = global_lower is not None and global_upper is not None

    for mode in _active_modes(next_modes):
        coefficient = float(mode.get("coefficient", 0.0))
        if use_global_bounds:
            lower = float(global_lower)
            upper = float(global_upper)
        else:
            bounds = mode.get("bounds")
            if bounds is None:
                continue
            lower, upper = (float(bounds[0]), float(bounds[1]))

        if coefficient < lower - float(tol):
            violation = lower - coefficient
        elif coefficient > upper + float(tol):
            violation = coefficient - upper
        else:
            continue
        raise BSplineAdaptiveError(
            "Boehm-transferred coefficient violates optimization bounds: "
            f"mode_id={mode.get('id')} side={mode.get('side')} "
            f"basis_index={mode.get('basis_index')} coefficient={coefficient:.16e} "
            f"lower_bound={lower:.16e} upper_bound={upper:.16e} "
            f"violation={violation:.16e}"
        )
    print("[PROGRESSIVE_BSPLINE] KNOT_TRANSFER coefficient bounds OK")

def transfer_shape_to_inserted_space(
    space,
    metadata,
    new_knots,
    regularization=1.0e-12,
    settings=None,
):
    del regularization
    settings = dict(settings or {})
    method = str(settings.get("transfer_method", "BOEHM")).strip().upper()
    if method != "BOEHM":
        raise BSplineAdaptiveError(
            "Only BSPLINE_TRANSFER_METHOD=BOEHM is currently implemented."
        )

    old_modes, old_matrix = _basis_for_spec_modes(space.spec, metadata)
    old_coeffs = np.asarray(
        [float(mode.get("coefficient", 0.0)) for mode in old_modes],
        dtype=float,
    )
    old_deformation = old_matrix.dot(old_coeffs)
    virtual_spec = regenerate_clamped_modes(space, new_knots)
    new_modes_by_side = {}
    for mode in _active_modes(virtual_spec):
        side = str(mode.get("side", "")).strip().lower()
        new_modes_by_side.setdefault(side, []).append(mode)

    coefficients_by_side = {}
    for side in space.sides:
        coefficients_by_side[side] = _boehm_transfer_side(
            space.groups[side],
            new_modes_by_side.get(side, []),
            space.spec,
            new_knots,
        )

    transferred_spec = regenerate_clamped_modes(
        space,
        new_knots,
        coefficients_by_side,
    )
    new_modes, new_matrix = _basis_for_spec_modes(transferred_spec, metadata)
    new_coeffs = np.asarray(
        [float(mode.get("coefficient", 0.0)) for mode in new_modes],
        dtype=float,
    )
    transferred = new_matrix.dot(new_coeffs)
    error = transferred - old_deformation
    rms = float(math.sqrt(float(np.mean(error * error)))) if len(error) else 0.0
    max_error = float(np.max(np.abs(error))) if len(error) else 0.0
    scale = float(np.max(np.abs(old_deformation))) if len(old_deformation) else 0.0
    relative = max_error / max(scale, 1.0e-30)
    abs_tol = _as_float(
        settings.get("transfer_geometry_abs_tol", 1.0e-10),
        "BSPLINE_TRANSFER_GEOMETRY_ABS_TOL",
    )
    rel_tol = _as_float(
        settings.get("transfer_geometry_rel_tol", 1.0e-8),
        "BSPLINE_TRANSFER_GEOMETRY_REL_TOL",
    )
    if abs_tol < 0.0 or rel_tol < 0.0:
        raise BSplineAdaptiveError(
            "B-spline transfer geometry tolerances must be non-negative"
        )
    tolerance = abs_tol + rel_tol * scale

    coefficient_arrays = list(coefficients_by_side.values())
    all_coefficients = (
        np.concatenate(coefficient_arrays)
        if coefficient_arrays
        else np.zeros(0, dtype=float)
    )
    coeff_min = float(np.min(all_coefficients)) if len(all_coefficients) else 0.0
    coeff_max = float(np.max(all_coefficients)) if len(all_coefficients) else 0.0
    print("[PROGRESSIVE_BSPLINE] KNOT_TRANSFER method=BOEHM")
    print(
        "[PROGRESSIVE_BSPLINE] KNOT_TRANSFER geometry "
        f"rms={rms:.6e} max={max_error:.6e} rel={relative:.6e}"
    )
    print(
        "[PROGRESSIVE_BSPLINE] KNOT_TRANSFER coefficients "
        f"min={coeff_min:.6e} max={coeff_max:.6e}"
    )
    if max_error > tolerance:
        raise BSplineAdaptiveError(
            "Boehm knot transfer failed to preserve geometry: "
            f"rms={rms:.16e} max_error={max_error:.16e} "
            f"relative_error={relative:.16e} tolerance={tolerance:.16e}"
        )

    diagnostics = {
        "transfer_method": "BOEHM",
        "transfer_rms_error": rms,
        "transfer_max_error": max_error,
        "transfer_relative_error": relative,
        "transfer_tolerance": tolerance,
        "transfer_geometry_abs_tol": abs_tol,
        "transfer_geometry_rel_tol": rel_tol,
        "transfer_coefficient_min": coeff_min,
        "transfer_coefficient_max": coeff_max,
    }
    return coefficients_by_side, diagnostics
