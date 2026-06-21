"""Geometry-aware bound scaling helpers for B-spline/SU2 optimization."""


import numpy as np

from .errors import BSplineSU2DriverError
from .reduction import _validated_bounds

def _vector_summary(values):
    values = np.asarray([float(value) for value in values], dtype=float)
    if values.size == 0:
        return "n=0"
    return "n={} min={:.6g} max={:.6g} norm2={:.6g}".format(
        values.size,
        float(np.min(values)),
        float(np.max(values)),
        float(np.linalg.norm(values)),
    )

def _bounds_summary(bounds):
    bounds = [(float(lower), float(upper)) for lower, upper in bounds]
    if not bounds:
        return "n=0"
    first = bounds[0]
    if all(
        abs(lower - first[0]) <= 1.0e-15 and abs(upper - first[1]) <= 1.0e-15
        for lower, upper in bounds
    ):
        return "uniform [{:.6g}, {:.6g}]".format(first[0], first[1])
    lowers = np.asarray([lower for lower, _upper in bounds], dtype=float)
    uppers = np.asarray([upper for _lower, upper in bounds], dtype=float)
    return (
        "nonuniform n={} lower[min={:.6g}, max={:.6g}] "
        "upper[min={:.6g}, max={:.6g}]"
    ).format(
        len(bounds),
        float(np.min(lowers)),
        float(np.max(lowers)),
        float(np.min(uppers)),
        float(np.max(uppers)),
    )

def _bounds_are_uniform(bounds):
    bounds = [(float(lower), float(upper)) for lower, upper in bounds]
    if not bounds:
        return True
    first = bounds[0]
    return all(
        abs(lower - first[0]) <= 1.0e-15 and abs(upper - first[1]) <= 1.0e-15
        for lower, upper in bounds
    )

def _bounds_to_list(bounds):
    return [[float(lower), float(upper)] for lower, upper in bounds]

def _max_radius_from_bounds(initial_coefficients, original_bounds):
    radii = []
    for coefficient, (lower, upper) in zip(initial_coefficients, original_bounds):
        radii.append(max(abs(float(lower) - float(coefficient)), abs(float(upper) - float(coefficient))))
    return radii

def compute_geometry_aware_bound_scaling(
    basis_matrix,
    initial_coefficients,
    original_bounds,
    max_normal_displacement=None,
    max_rms_normal_displacement=None,
    min_bound_scale=0.0,
):
    basis_matrix = np.asarray(basis_matrix, dtype=float)
    initial_coefficients = [float(value) for value in initial_coefficients]
    original_bounds = [
        _validated_bounds(bound, f"mode {index} bounds")
        for index, bound in enumerate(original_bounds)
    ]
    min_bound_scale = float(min_bound_scale)
    if min_bound_scale < 0.0:
        raise BSplineSU2DriverError("--min-bound-scale must be non-negative")

    if basis_matrix.ndim != 2:
        raise BSplineSU2DriverError("basis matrix for geometry-aware scaling must be two-dimensional")
    if basis_matrix.shape[1] != len(initial_coefficients):
        raise BSplineSU2DriverError("basis matrix column count must match the number of active coefficients")
    if basis_matrix.shape[1] != len(original_bounds):
        raise BSplineSU2DriverError("basis matrix column count must match the number of active bounds")

    abs_d0 = np.abs(basis_matrix.dot(np.asarray(initial_coefficients, dtype=float)))
    coefficient_radius = np.asarray(
        _max_radius_from_bounds(initial_coefficients, original_bounds),
        dtype=float,
    )
    row_radius = np.abs(basis_matrix).dot(coefficient_radius)
    max_abs_dn_current = float(np.max(abs_d0)) if len(abs_d0) else 0.0
    rms_dn_current = float(np.sqrt(np.mean(abs_d0 * abs_d0))) if len(abs_d0) else 0.0
    row_radius_max = float(np.max(row_radius)) if len(row_radius) else 0.0

    def _beta_from_max_limit(limit):
        limit = float(limit)
        if max_abs_dn_current > limit:
            raise BSplineSU2DriverError(
                "Initial geometry already violates max-normal-displacement; cannot make bounds safe by shrinking around it."
            )
        candidates = [
            (limit - float(abs_value)) / float(radius)
            for abs_value, radius in zip(abs_d0, row_radius)
            if float(radius) > 0.0
        ]
        beta = min([1.0] + candidates) if candidates else 1.0
        return min(1.0, max(0.0, float(beta)))

    def _rms_bound(beta):
        values = abs_d0 + float(beta) * row_radius
        return float(np.sqrt(np.mean(values * values))) if len(values) else 0.0

    def _beta_from_rms_limit(limit):
        limit = float(limit)
        if rms_dn_current > limit:
            raise BSplineSU2DriverError(
                "Initial geometry already violates max-rms-normal-displacement; cannot make bounds safe by shrinking around it."
            )
        if _rms_bound(1.0) <= limit:
            return 1.0
        lo = 0.0
        hi = 1.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if _rms_bound(mid) <= limit:
                lo = mid
            else:
                hi = mid
        return min(1.0, max(0.0, lo))

    betas = [1.0]
    if max_normal_displacement is not None:
        betas.append(_beta_from_max_limit(max_normal_displacement))
    if max_rms_normal_displacement is not None:
        betas.append(_beta_from_rms_limit(max_rms_normal_displacement))

    beta_safe = min(betas)
    if beta_safe < min_bound_scale:
        raise BSplineSU2DriverError(
            "Required geometry-safe bound scale beta={:.15g} is below --min-bound-scale={:.15g}".format(
                beta_safe,
                min_bound_scale,
            )
        )

    scaled_bounds = []
    for coefficient, (lower, upper) in zip(initial_coefficients, original_bounds):
        coeff = float(coefficient)
        scaled_bounds.append(
            (
                coeff + beta_safe * (float(lower) - coeff),
                coeff + beta_safe * (float(upper) - coeff),
            )
        )

    summary = {
        "enabled": True,
        "beta_safe": beta_safe,
        "max_abs_dn_current": max_abs_dn_current,
        "rms_dn_current": rms_dn_current,
        "max_normal_displacement": None if max_normal_displacement is None else float(max_normal_displacement),
        "max_rms_normal_displacement": None if max_rms_normal_displacement is None else float(max_rms_normal_displacement),
        "original_bounds": _bounds_to_list(original_bounds),
        "scaled_bounds": _bounds_to_list(scaled_bounds),
        "initial_coefficients": [float(value) for value in initial_coefficients],
        "row_radius_max": row_radius_max,
    }
    return summary
