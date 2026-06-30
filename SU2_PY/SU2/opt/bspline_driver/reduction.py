"""Reduced-variable and symmetry helpers for B-spline/SU2 optimization."""

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from SU2.opt.bspline_modes import validate_mode_spec
from .constants import (
    ALLOWED_SYMMETRY_COUPLINGS,
    DEFAULT_BOUNDS,
)
from .errors import (
    BSplineSU2DriverError,
    _as_float,
)

def _active_modes(mode_spec):
    validate_mode_spec(mode_spec)
    return [
        mode
        for mode in mode_spec.get("modes", [])
        if mode.get("active", True) is not False
    ]

def _design_modes(mode_spec):
    return [
        mode
        for mode in _active_modes(mode_spec)
        if mode.get("frozen", False) is not True
    ]

def active_mode_ids(mode_spec):
    return [str(mode["id"]) for mode in _design_modes(mode_spec)]

def active_coefficient_vector(mode_spec):
    return [
        _as_float(mode.get("coefficient", 0.0), f"mode {mode['id']} coefficient")
        for mode in _design_modes(mode_spec)
    ]

def active_bounds(mode_spec, default_bounds=DEFAULT_BOUNDS):
    lower_default, upper_default = _validated_bounds(default_bounds, "default_bounds")
    bounds = []
    for mode in _design_modes(mode_spec):
        mode_bounds = mode.get("bounds")
        if mode_bounds is None:
            bounds.append((lower_default, upper_default))
        else:
            bounds.append(_validated_bounds(mode_bounds, f"mode {mode['id']} bounds"))
    return bounds

@dataclass(frozen=True)
class ReducedVariable:
    id: str
    mode_indices: tuple
    mode_ids: tuple
    signs: tuple

def _safe_identifier(value):
    text = str(value)
    chars = []
    for char in text:
        chars.append(char if char.isalnum() else "_")
    return "_".join(part for part in "".join(chars).split("_") if part)

def _mode_support_key(mode):
    basis_type = str(mode.get("basis_type", "")).strip().lower()
    if basis_type != "clamped":
        # Knot-insertion only. Reject any non-clamped basis at the support-key
        # boundary; callers should never observe legacy basis types here because
        # validate_mode_spec rejects them upstream.
        raise BSplineSU2DriverError(
            "Only basis_type='clamped' is supported by the knot-insertion B-spline workflow."
        )
    degree = int(mode.get("degree", 3))
    knots = mode.get("knot_vector", mode.get("knots"))
    if knots is None or "basis_index" not in mode:
        left, right = 0.0, 1.0
    else:
        knots = [float(value) for value in knots]
        index = int(mode["basis_index"])
        right_index = min(len(knots) - 1, index + degree + 1)
        left, right = knots[index], knots[right_index]
    return basis_type, degree, round(float(left), 12), round(float(right), 12)

def _mode_pairing_key(mode):
    basis_type = str(mode.get("basis_type", "")).strip().lower()
    if basis_type != "clamped":
        raise BSplineSU2DriverError(
            "Only basis_type='clamped' is supported by the knot-insertion B-spline workflow."
        )
    degree = int(mode.get("degree", 3))
    knots = mode.get("knot_vector", mode.get("knots"))
    if knots is None or "basis_index" not in mode:
        return basis_type, degree, (), None
    knots = tuple(round(float(value), 12) for value in knots)
    return basis_type, degree, knots, int(mode["basis_index"])

def _symmetry_group_id(key):
    basis_type, degree, third, fourth = key
    if basis_type == "clamped":
        return _safe_identifier(
            "group_{}_d{}_i{}".format(
                basis_type,
                degree,
                "missing" if fourth is None else int(fourth),
            )
        )
    left, right = third, fourth
    return _safe_identifier(
        "group_{}_d{}_s{:.12g}_{:.12g}".format(
            basis_type,
            degree,
            float(left),
            float(right),
        )
    )

def build_reduced_variables(mode_spec, coupling="NONE"):
    coupling = str(coupling or "NONE").strip().upper()
    if coupling not in ALLOWED_SYMMETRY_COUPLINGS:
        raise BSplineSU2DriverError(
            f"BSPLINE_SYMMETRY_COUPLING must be one of {ALLOWED_SYMMETRY_COUPLINGS}; got {coupling!r}"
        )

    active_modes = _design_modes(mode_spec)
    if coupling == "NONE":
        return [
            ReducedVariable(
                id=_safe_identifier(mode["id"]),
                mode_indices=(index,),
                mode_ids=(str(mode["id"]),),
                signs=(1.0,),
            )
            for index, mode in enumerate(active_modes)
        ], []

    grouped = {}
    for index, mode in enumerate(active_modes):
        grouped.setdefault(_mode_pairing_key(mode), []).append((index, mode))

    reduced = []
    warnings = []
    for key in sorted(grouped, key=lambda item: (item[0], item[1], item[2], item[3])):
        entries = grouped[key]
        by_side = {}
        for index, mode in entries:
            by_side.setdefault(str(mode.get("side", "")).strip().lower(), []).append((index, mode))

        if len(entries) == 2 and len(by_side.get("upper", [])) == 1 and len(by_side.get("lower", [])) == 1:
            upper_index, upper_mode = by_side["upper"][0]
            lower_index, lower_mode = by_side["lower"][0]
            lower_sign = 1.0 if coupling == "NORMAL_EQUAL" else -1.0
            reduced.append(
                ReducedVariable(
                    id=_symmetry_group_id(key),
                    mode_indices=(upper_index, lower_index),
                    mode_ids=(str(upper_mode["id"]), str(lower_mode["id"])),
                    signs=(1.0, lower_sign),
                )
            )
            continue

        mode_ids = ", ".join(str(mode["id"]) for _index, mode in entries)
        warnings.append(
            "{} symmetry coupling left mode(s) unpaired for support {}: {}".format(
                coupling,
                key,
                mode_ids,
            )
        )
        for index, mode in entries:
            reduced.append(
                ReducedVariable(
                    id=_safe_identifier(mode["id"]),
                    mode_indices=(index,),
                    mode_ids=(str(mode["id"]),),
                    signs=(1.0,),
                )
            )

    reduced.sort(key=lambda variable: min(variable.mode_indices))
    return reduced, warnings

def expand_reduced_coefficients(reduced_coefficients, reduced_variables, n_modes):
    values = [_as_float(value, "reduced coefficient") for value in reduced_coefficients]
    if len(values) != len(reduced_variables):
        raise BSplineSU2DriverError(
            f"expected {len(reduced_variables)} reduced coefficient(s), got {len(values)}"
        )
    full = [0.0] * int(n_modes)
    for value, variable in zip(values, reduced_variables):
        for index, sign in zip(variable.mode_indices, variable.signs):
            full[int(index)] = float(sign) * float(value)
    return full

def compress_full_coefficients(
    coefficients,
    reduced_variables,
    tolerance=1.0e-8,
    warn=None,
    coupling="NONE",
):
    """Collapse full coefficients to the reduced space.

    With ``NORMAL_OPPOSITE`` the reduced variable ``r`` represents the
    antisymmetric component: ``a_upper = +r``, ``a_lower = -r``. A
    unilateral bump (e.g. ``a_upper = 0, a_lower = +0.01``) cannot be
    represented in that subspace; the silent transformation
    ``r = (0 - 0.01)/2`` would map it to ``a_upper = -0.005`` and
    ``a_lower = +0.005`` and the original ``warn`` mismatch test
    (which only checks the round-trip through the antisymmetric
    formula) would never fire. We therefore detect this case explicitly
    when ``coupling == 'NORMAL_OPPOSITE'`` and a 2-element pair is
    supplied, and raise so the user can either:
      * set the modes to zero and let the optimizer introduce them
        via the adaptive driver, or
      * switch to ``NORMAL_EQUAL`` if a pure mirror is acceptable.
    """
    coupling = str(coupling or "NONE").strip().upper()
    values = [_as_float(value, "coefficient") for value in coefficients]
    reduced = []
    for variable in reduced_variables:
        paired_values = [values[int(index)] for index in variable.mode_indices]
        signed_values = [
            float(sign) * float(value)
            for sign, value in zip(variable.signs, paired_values)
        ]
        reduced_value = sum(signed_values) / float(len(signed_values))
        mismatch = max(
            abs(float(value) - float(sign) * reduced_value)
            for sign, value in zip(variable.signs, paired_values)
        )
        # For NORMAL_OPPOSITE a 2-element pair is representable iff it is
        # antisymmetric, i.e. a_upper + a_lower == 0. ANY other input
        # (a unilateral bump like [0, 0.01], or a same-sign pair like
        # [0.01, 0.01]) is silently mapped to an inverted/mirrored
        # deformation by the reduced-variable round-trip, so we refuse it
        # outright instead of merely warning. This is checked BEFORE the
        # generic mismatch warning below so the error message is specific.
        if (
            coupling == "NORMAL_OPPOSITE"
            and len(variable.mode_indices) == 2
            and len(variable.signs) == 2
            and list(variable.signs) == [1.0, -1.0]
        ):
            a_upper, a_lower = paired_values[0], paired_values[1]
            if abs(float(a_upper) + float(a_lower)) > float(tolerance):
                raise BSplineSU2DriverError(
                    "NORMAL_OPPOSITE coupling requires antisymmetric initial "
                    "coefficients (a_upper + a_lower == 0) for modes {} but got "
                    "a_upper={:.6e}, a_lower={:.6e}; set both to zero, make them "
                    "antisymmetric, or use NORMAL_EQUAL.".format(
                        ", ".join(variable.mode_ids),
                        float(a_upper),
                        float(a_lower),
                    )
                )
        if len(variable.mode_indices) > 1 and mismatch > float(tolerance) and warn is not None:
            warn(
                "initial coefficients for coupled modes {} are inconsistent; using reduced value {:.15g}".format(
                    ", ".join(variable.mode_ids),
                    reduced_value,
                )
            )
        reduced.append(float(reduced_value))
    return reduced

def reduced_bounds_from_full_bounds(bounds, reduced_variables):
    reduced_bounds = []
    for variable in reduced_variables:
        lower = -math.inf
        upper = math.inf
        for index, sign in zip(variable.mode_indices, variable.signs):
            mode_lower, mode_upper = _validated_bounds(
                bounds[int(index)],
                f"bounds for reduced variable {variable.id}",
            )
            if float(sign) >= 0.0:
                candidate_lower, candidate_upper = mode_lower, mode_upper
            else:
                candidate_lower, candidate_upper = -mode_upper, -mode_lower
            lower = max(lower, candidate_lower)
            upper = min(upper, candidate_upper)
        if upper < lower:
            raise BSplineSU2DriverError(
                "empty bound intersection for coupled reduced variable {} ({})".format(
                    variable.id,
                    ", ".join(variable.mode_ids),
                )
            )
        reduced_bounds.append((float(lower), float(upper)))
    return reduced_bounds

def collapse_full_gradient(gradient, reduced_variables):
    values = [_as_float(value, "gradient") for value in gradient]
    collapsed = []
    for variable in reduced_variables:
        collapsed.append(
            sum(
                float(sign) * values[int(index)]
                for index, sign in zip(variable.mode_indices, variable.signs)
            )
        )
    return collapsed

def collapse_full_jacobian(jacobian, reduced_variables):
    jacobian = np.asarray(jacobian, dtype=float)
    if jacobian.ndim != 2:
        raise BSplineSU2DriverError("constraint Jacobian must be two-dimensional")
    collapsed = np.zeros((jacobian.shape[0], len(reduced_variables)), dtype=float)
    for column, variable in enumerate(reduced_variables):
        for index, sign in zip(variable.mode_indices, variable.signs):
            collapsed[:, column] += float(sign) * jacobian[:, int(index)]
    return collapsed

def mode_support_length(mode):
    key = _mode_support_key(mode)
    return max(0.0, float(key[3]) - float(key[2]))

def reduced_step_limits_from_modes(mode_spec, reduced_variables, ratio):
    ratio = _as_float(ratio, "BSPLINE_LOCAL_STEP_LIMIT_RATIO")
    if ratio <= 0.0:
        raise BSplineSU2DriverError("BSPLINE_LOCAL_STEP_LIMIT_RATIO must be positive")
    active_modes = _design_modes(mode_spec)
    limits = []
    for variable in reduced_variables:
        lengths = [mode_support_length(active_modes[int(index)]) for index in variable.mode_indices]
        if not lengths or min(lengths) <= 0.0:
            limits.append(math.inf)
        else:
            limits.append(min(lengths) / ratio)
    return limits

def _validated_bounds(bounds, name):
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise BSplineSU2DriverError(f"{name} must contain [lower, upper]")
    lower = _as_float(bounds[0], f"{name} lower")
    upper = _as_float(bounds[1], f"{name} upper")
    if upper < lower:
        raise BSplineSU2DriverError(f"{name} upper bound is below lower bound")
    return lower, upper

def update_mode_coefficients(mode_spec, coefficients):
    """Return a copy of mode_spec with design-mode coefficients replaced."""

    spec = copy.deepcopy(validate_mode_spec(mode_spec))
    values = [_as_float(value, "coefficient") for value in coefficients]
    design_count = len(_design_modes(spec))
    if len(values) != design_count:
        raise BSplineSU2DriverError(
            f"expected {design_count} design coefficient(s), got {len(values)}"
        )

    index = 0
    for mode in spec.get("modes", []):
        if mode.get("active", True) is False or mode.get("frozen", False) is True:
            continue
        mode["coefficient"] = values[index]
        index += 1

    return validate_mode_spec(spec)

def write_mode_spec(mode_spec, filename):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w") as fp:
        json.dump(mode_spec, fp, indent=2)
        fp.write("\n")

def cache_key(coefficients, tol=1.0e-12):
    tol = _as_float(tol, "cache tolerance")
    if tol <= 0.0:
        raise BSplineSU2DriverError("cache tolerance must be positive")
    digits = max(0, int(math.ceil(-math.log10(tol))))
    return tuple(round(_as_float(value, "coefficient"), digits) for value in coefficients)
