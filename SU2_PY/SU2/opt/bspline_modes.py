#!/usr/bin/env python

"""B-spline deformation-mode utilities for external SU2 surface files."""

import json
import math


class BSplineModeError(ValueError):
    pass


UNSUPPORTED_BASIS_TYPE_MESSAGE = (
    "Only basis_type='clamped' is supported by the knot-insertion B-spline workflow."
)
DEFAULT_NORMALIZATION_SAMPLES = 5001
LE_SAFE_DEFAULT_X0 = 0.005
LE_SAFE_DEFAULT_X1 = 0.025
LE_SAFE_DEFAULT_POWER = 1.5
ALLOWED_DEFORMATION_DIRECTION_MODES = ("NORMAL", "LE_SAFE", "VERTICAL")
ALLOWED_SURFACE_MODES = ("BOTH", "UPPER", "LOWER")


def _as_float(value, name):
    try:
        result = float(value)
    except Exception:
        raise BSplineModeError(f"{name} must be numeric")
    if not math.isfinite(result):
        raise BSplineModeError(f"{name} must be finite")
    return result


def _as_bool(value, name):
    if isinstance(value, bool):
        return value
    text = str(value).strip().upper()
    if text in ("YES", "TRUE", "1", "ON"):
        return True
    if text in ("NO", "FALSE", "0", "OFF"):
        return False
    raise BSplineModeError(f"{name} must be YES or NO")


def _as_float_list(values, name):
    if not isinstance(values, (list, tuple)):
        raise BSplineModeError(f"{name} must be a list")
    return [_as_float(value, name) for value in values]


def normalize_surface_mode(value=None):
    """Return the canonical B-spline surface mode."""

    normalized = str("BOTH" if value is None else value).strip().upper().replace("-", "_")
    aliases = {
        "BOTH": "BOTH",
        "FULL": "BOTH",
        "UPPER": "UPPER",
        "HALF_UPPER": "UPPER",
        "LOWER": "LOWER",
        "HALF_LOWER": "LOWER",
    }
    try:
        return aliases[normalized]
    except KeyError:
        raise BSplineModeError(
            "BSPLINE_SURFACE_MODE must be BOTH, UPPER, or LOWER; "
            f"got {value!r}"
        )


def active_sides_from_surface_mode(surface_mode):
    mode = normalize_surface_mode(surface_mode)
    if mode == "UPPER":
        return ["upper"]
    if mode == "LOWER":
        return ["lower"]
    return ["upper", "lower"]


def validate_surface_mode_against_modes(spec, surface_mode):
    """Reject active modes belonging to a side absent from a half-domain."""

    mode = normalize_surface_mode(surface_mode)
    required_side = None if mode == "BOTH" else mode.lower()
    if required_side is None:
        return mode

    for item in spec.get("modes", []):
        if item.get("active", True) is False:
            continue
        if str(item.get("side", "")).strip().lower() != required_side:
            raise BSplineModeError(
                f"BSPLINE_SURFACE_MODE={mode} requires all active modes "
                f"to have side='{required_side}'"
            )
    return mode


def _validate_degree(value, name):
    try:
        degree = int(value)
    except Exception:
        raise BSplineModeError(f"{name} must be an integer")
    if degree < 0:
        raise BSplineModeError(f"{name} must be non-negative")
    return degree


def _mode_knot_vector(mode):
    if "knot_vector" in mode:
        return mode["knot_vector"]
    if "knots" in mode:
        return mode["knots"]
    raise BSplineModeError(f"Mode {mode.get('id', '<unnamed>')!r} is missing knot_vector")


def clamped_basis_count(degree, knot_vector):
    """Return the number of scalar B-spline basis functions."""

    degree = _validate_degree(degree, "degree")
    knots = _as_float_list(knot_vector, "knot_vector")
    count = len(knots) - degree - 1
    if count <= 0:
        raise BSplineModeError("knot_vector is too short for the requested degree")
    return count


def _validate_knot_vector(degree, knot_vector, mode_id):
    knots = _as_float_list(knot_vector, f"mode {mode_id!r} knot_vector")
    if len(knots) < 2 * (degree + 1):
        raise BSplineModeError(
            f"Mode {mode_id!r} knot_vector is too short for a clamped degree {degree} basis"
        )
    for left, right in zip(knots[:-1], knots[1:]):
        if right < left:
            raise BSplineModeError(f"Mode {mode_id!r} knot_vector must be nondecreasing")

    tol = 1.0e-14
    left = knots[0]
    right = knots[-1]
    if not left < right:
        raise BSplineModeError(f"Mode {mode_id!r} knot_vector must span a positive interval")
    if not all(abs(knots[i] - left) <= tol for i in range(degree + 1)):
        raise BSplineModeError(
            f"Mode {mode_id!r} must use a clamped knot vector at the left endpoint"
        )
    if not all(abs(knots[-1 - i] - right) <= tol for i in range(degree + 1)):
        raise BSplineModeError(
            f"Mode {mode_id!r} must use a clamped knot vector at the right endpoint"
        )
    return knots


def _validate_mode(mode, seen_ids):
    if not isinstance(mode, dict):
        raise BSplineModeError("Each mode entry must be an object")

    mode_id = str(mode.get("id", "")).strip()
    if not mode_id:
        raise BSplineModeError("Each mode requires a non-empty id")
    if mode_id in seen_ids:
        raise BSplineModeError(f"Duplicate mode id {mode_id!r}")
    seen_ids.add(mode_id)

    side = str(mode.get("side", "")).strip().lower()
    if side not in ("upper", "lower"):
        raise BSplineModeError(f"Mode {mode_id!r} side must be 'upper' or 'lower'")

    basis_type = str(mode.get("basis_type", "")).strip().lower()
    if basis_type != "clamped":
        raise BSplineModeError(UNSUPPORTED_BASIS_TYPE_MESSAGE)

    degree = _validate_degree(mode.get("degree", 3), f"mode {mode_id!r} degree")
    if degree != 3:
        raise BSplineModeError(
            f"Mode {mode_id!r} degree must be 3 for the knot-insertion B-spline workflow"
        )

    knots = _validate_knot_vector(degree, _mode_knot_vector(mode), mode_id)
    n_basis = clamped_basis_count(degree, knots)
    try:
        basis_index = int(mode.get("basis_index"))
    except Exception:
        raise BSplineModeError(f"Mode {mode_id!r} requires an integer basis_index")
    if basis_index < 0 or basis_index >= n_basis:
        raise BSplineModeError(
            f"Mode {mode_id!r} basis_index {basis_index} outside [0, {n_basis - 1}]"
        )

    if "coefficient" not in mode:
        raise BSplineModeError(f"Mode {mode_id!r} requires a coefficient")
    _as_float(mode.get("coefficient"), f"mode {mode_id!r} coefficient")

    if "apply_class_shape" in mode and not isinstance(mode["apply_class_shape"], bool):
        # Guard against bool("false") == True: a non-empty string would
        # silently enable the class-shape factor for a basis that the user
        # meant to disable.
        raise BSplineModeError(
            f"Mode {mode_id!r} apply_class_shape must be a JSON boolean"
        )

    if "bounds" in mode and mode["bounds"] is not None:
        bounds = _as_float_list(mode["bounds"], f"mode {mode_id!r} bounds")
        if len(bounds) != 2 or bounds[1] < bounds[0]:
            raise BSplineModeError(
                f"Mode {mode_id!r} bounds must contain [lower, upper]"
            )


def validate_mode_spec(spec):
    """Validate a BSPLINE_DEF v1 mode JSON object and return it."""

    if not isinstance(spec, dict):
        raise BSplineModeError("Mode specification must be a JSON object")
    if int(spec.get("version", -1)) != 1:
        raise BSplineModeError("Only mode specification version 1 is supported")
    if int(spec.get("dimension", -1)) != 2:
        raise BSplineModeError("Only dimension=2 is supported in BSPLINE_DEF v1")
    if not str(spec.get("marker", "")).strip():
        raise BSplineModeError("Mode specification requires a marker")

    chord = spec.get("chord")
    if not isinstance(chord, dict):
        raise BSplineModeError("Mode specification requires a chord object")
    chord_mode = str(chord.get("mode", "auto")).strip().lower()
    if chord_mode != "auto":
        _as_float(chord.get("x_le"), "chord.x_le")
        _as_float(chord.get("x_te"), "chord.x_te")

    if not isinstance(spec.get("normal_displacement", True), bool):
        raise BSplineModeError("normal_displacement must be a JSON boolean")
    class_shape = str(spec.get("class_shape", "sqrt_x_one_minus_x")).strip().lower()
    if class_shape not in (
        "sqrt_x_one_minus_x",
        "none",
    ):
        raise BSplineModeError(
            "class_shape must be 'sqrt_x_one_minus_x' or 'none'"
        )
    if class_shape != "none":
        alpha = _as_float(spec.get("class_shape_exponent", 0.5), "class_shape_exponent")
        if alpha < 0.0:
            raise BSplineModeError("class_shape_exponent must be >= 0")
    if not isinstance(spec.get("normalize_basis", True), bool):
        raise BSplineModeError("normalize_basis must be a JSON boolean")
    if str(spec.get("normalization_mode", "max")).strip().lower() != "max":
        raise BSplineModeError("Only normalization_mode='max' is supported")

    modes = spec.get("modes")
    if not isinstance(modes, list):
        raise BSplineModeError("Mode specification requires a modes list")
    seen_ids = set()
    for mode in modes:
        _validate_mode(mode, seen_ids)

    if "surface_mode" in spec:
        surface_mode = normalize_surface_mode(spec["surface_mode"])
        validate_surface_mode_against_modes(spec, surface_mode)

    return spec


def load_mode_spec(filename):
    with open(filename, "r") as fp:
        spec = json.load(fp)
    return validate_mode_spec(spec)


def cox_de_boor_basis(x, degree, knot_vector, basis_index):
    """Evaluate one B-spline basis function with the Cox-de Boor recursion."""

    x = float(x)
    degree = _validate_degree(degree, "degree")
    knots = _as_float_list(knot_vector, "knot_vector")
    n_basis = clamped_basis_count(degree, knots)
    basis_index = int(basis_index)
    if basis_index < 0 or basis_index >= n_basis:
        raise BSplineModeError(
            f"basis_index {basis_index} outside [0, {n_basis - 1}]"
        )

    tol = 1.0e-14
    if x < knots[0] - tol or x > knots[-1] + tol:
        return 0.0
    if abs(x - knots[-1]) <= tol:
        return 1.0 if basis_index == n_basis - 1 else 0.0

    def recurse(i, p):
        if p == 0:
            return 1.0 if knots[i] <= x < knots[i + 1] else 0.0

        value = 0.0
        left_den = knots[i + p] - knots[i]
        if left_den > 0.0:
            value += ((x - knots[i]) / left_den) * recurse(i, p - 1)

        right_den = knots[i + p + 1] - knots[i + 1]
        if right_den > 0.0:
            value += ((knots[i + p + 1] - x) / right_den) * recurse(i + 1, p - 1)

        return value

    return recurse(basis_index, degree)


def clamped_basis_value(x, degree, knot_vector, basis_index):
    return cox_de_boor_basis(x, degree, knot_vector, basis_index)


def class_shape_factor(x, class_shape="sqrt_x_one_minus_x", class_shape_exponent=0.5):
    class_shape = str(class_shape or "sqrt_x_one_minus_x").strip().lower()
    if class_shape == "none":
        return 1.0
    if class_shape != "sqrt_x_one_minus_x":
        raise BSplineModeError(f"Unsupported class_shape {class_shape!r}")
    alpha = _as_float(class_shape_exponent, "class_shape_exponent")
    if not math.isfinite(alpha) or alpha < 0.0:
        raise BSplineModeError("class_shape_exponent must be finite and >= 0")

    x = float(x)
    if x <= 0.0 or x >= 1.0:
        return 0.0

    return (x ** alpha) * (1.0 - x)


def normalize_deformation_direction_mode(value=None, le_safe_direction=False):
    if value is None or str(value).strip() == "":
        legacy_le_safe = _as_bool(
            le_safe_direction,
            "BSPLINE_LE_SAFE_DIRECTION",
        )
        return "LE_SAFE" if legacy_le_safe else "NORMAL"

    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "normal": "NORMAL",
        "normals": "NORMAL",
        "le_safe": "LE_SAFE",
        "lesafe": "LE_SAFE",
        "vertical": "VERTICAL",
        "y": "VERTICAL",
    }
    try:
        return aliases[normalized]
    except KeyError:
        raise BSplineModeError(
            "BSPLINE_DEFORMATION_DIRECTION must be one of "
            f"{ALLOWED_DEFORMATION_DIRECTION_MODES}; got {value!r}"
        )


def vertical_direction(side):
    side = str(side).strip().lower()
    if side == "upper":
        return 0.0, 1.0
    if side == "lower":
        return 0.0, -1.0
    raise BSplineModeError(f"invalid side for vertical direction: {side!r}")


def le_safe_direction(
    x_over_c,
    side,
    normal_x,
    normal_y,
    x0=LE_SAFE_DEFAULT_X0,
    x1=LE_SAFE_DEFAULT_X1,
    power=LE_SAFE_DEFAULT_POWER,
):
    if x1 <= x0:
        raise ValueError("BSPLINE_LE_SAFE_X1 must be greater than BSPLINE_LE_SAFE_X0")

    t = (float(x_over_c) - float(x0)) / (float(x1) - float(x0))
    t = max(0.0, min(1.0, t))

    s = 3.0 * t * t - 2.0 * t * t * t
    w = s ** float(power)

    side = str(side).strip().lower()
    if side == "upper":
        vy0 = 1.0
    elif side == "lower":
        vy0 = -1.0
    else:
        raise ValueError(f"invalid side for LE-safe direction: {side!r}")

    vx = w * float(normal_x)
    vy = (1.0 - w) * vy0 + w * float(normal_y)

    return vx, vy


def validate_le_safe_direction_options(
    le_safe_direction=False,
    le_safe_x0=LE_SAFE_DEFAULT_X0,
    le_safe_x1=LE_SAFE_DEFAULT_X1,
    le_safe_power=LE_SAFE_DEFAULT_POWER,
):
    enabled = _as_bool(le_safe_direction, "BSPLINE_LE_SAFE_DIRECTION")
    x0 = _as_float(le_safe_x0, "BSPLINE_LE_SAFE_X0")
    x1 = _as_float(le_safe_x1, "BSPLINE_LE_SAFE_X1")
    power = _as_float(le_safe_power, "BSPLINE_LE_SAFE_POWER")

    if enabled:
        if not (0.0 <= x0 < x1 <= 1.0):
            raise BSplineModeError(
                "BSPLINE_LE_SAFE_X0 and BSPLINE_LE_SAFE_X1 must satisfy 0 <= x0 < x1 <= 1"
            )
        if power <= 0.0:
            raise BSplineModeError("BSPLINE_LE_SAFE_POWER must be positive")

    return {
        "le_safe_direction": enabled,
        "le_safe_x0": x0,
        "le_safe_x1": x1,
        "le_safe_power": power,
    }


def deformation_direction(
    x_over_c,
    side,
    normal_x,
    normal_y,
    use_le_safe_direction=False,
    le_safe_x0=LE_SAFE_DEFAULT_X0,
    le_safe_x1=LE_SAFE_DEFAULT_X1,
    le_safe_power=LE_SAFE_DEFAULT_POWER,
    direction_mode=None,
):
    mode = normalize_deformation_direction_mode(
        direction_mode,
        le_safe_direction=use_le_safe_direction,
    )
    if mode == "VERTICAL":
        return vertical_direction(side)
    if mode == "NORMAL":
        return float(normal_x), float(normal_y)
    return le_safe_direction(
        x_over_c,
        side,
        normal_x,
        normal_y,
        x0=le_safe_x0,
        x1=le_safe_x1,
        power=le_safe_power,
    )


def mode_apply_class_shape(mode, global_class_shape):
    # In addition to the validation in _validate_mode, this helper itself
    # refuses non-bool ``apply_class_shape`` so that any call path (not
    # only the one going through ``validate_mode_spec``) cannot fall
    # into the ``bool("false") == True`` pitfall. Strings, ints and
    # other non-bool types all raise BSplineModeError.
    if "apply_class_shape" in mode:
        flag = mode["apply_class_shape"]
        if not isinstance(flag, bool):
            mode_id = str(mode.get("id", "<unnamed>"))
            raise BSplineModeError(
                f"Mode {mode_id!r} apply_class_shape must be a JSON boolean, "
                f"got {type(flag).__name__}: {flag!r}"
            )
        return flag
    return str(global_class_shape or "sqrt_x_one_minus_x").strip().lower() != "none"


def _mode_normalization_support(mode):
    basis_type = str(mode.get("basis_type", "")).strip().lower()
    if basis_type != "clamped":
        raise BSplineModeError(UNSUPPORTED_BASIS_TYPE_MESSAGE)
    degree = int(mode.get("degree", 3))
    knots = _as_float_list(_mode_knot_vector(mode), "knot_vector")
    basis_index = int(mode.get("basis_index"))
    left_index = max(0, min(len(knots) - 1, basis_index))
    right_index = max(0, min(len(knots) - 1, basis_index + degree + 1))
    return knots[left_index], knots[right_index]


def _normalization_samples(left, right, count):
    count = max(3, int(count))
    left = float(left)
    right = float(right)
    if not left < right:
        return [left]
    step = (right - left) / float(count - 1)
    samples = [left + step * index for index in range(count)]
    eps = max((right - left) * 1.0e-12, 1.0e-15)
    if count > 2:
        samples[0] = left + eps
        samples[-1] = right - eps
    samples.append(0.5 * (left + right))
    return samples


def _mode_characteristic_samples(mode, left, right):
    basis_type = str(mode.get("basis_type", "")).strip().lower()
    samples = []
    if basis_type == "clamped":
        degree = int(mode.get("degree", 3))
        if degree > 0:
            knots = _as_float_list(_mode_knot_vector(mode), "knot_vector")
            basis_index = int(mode.get("basis_index"))
            start = basis_index + 1
            stop = basis_index + degree + 1
            if 0 <= start < stop <= len(knots):
                samples.append(sum(knots[start:stop]) / float(degree))
    else:
        raise BSplineModeError(UNSUPPORTED_BASIS_TYPE_MESSAGE)
    return [
        float(value)
        for value in samples
        if float(left) <= float(value) <= float(right)
    ]


def mode_normalization_factor(
    mode,
    class_shape="sqrt_x_one_minus_x",
    class_shape_exponent=0.5,
    normalization_samples=DEFAULT_NORMALIZATION_SAMPLES,
):
    if "normalization_factor" in mode:
        factor = _as_float(mode.get("normalization_factor"), "normalization_factor")
        if factor <= 0.0:
            raise BSplineModeError("normalization_factor must be positive")
        return factor

    left, right = _mode_normalization_support(mode)
    use_class_shape = mode_apply_class_shape(mode, class_shape)
    max_abs = 0.0
    samples = _normalization_samples(left, right, normalization_samples)
    samples.extend(_mode_characteristic_samples(mode, left, right))
    for x in samples:
        value = mode_basis_value(mode, x)
        if use_class_shape:
            value *= class_shape_factor(
                x,
                class_shape,
                class_shape_exponent=class_shape_exponent,
            )
        max_abs = max(max_abs, abs(value))

    return max_abs if max_abs > 0.0 else 1.0


def mode_basis_value(mode, x):
    basis_type = str(mode.get("basis_type", "")).strip().lower()
    degree = int(mode.get("degree", 3))
    if basis_type == "clamped":
        return clamped_basis_value(
            x,
            degree,
            _mode_knot_vector(mode),
            int(mode.get("basis_index")),
        )
    raise BSplineModeError(UNSUPPORTED_BASIS_TYPE_MESSAGE)


def evaluate_mode_values(
    mode,
    x_over_c,
    class_shape="sqrt_x_one_minus_x",
    class_shape_exponent=0.5,
    normalize=True,
    normalization_mode="max",
    normalization_samples=DEFAULT_NORMALIZATION_SAMPLES,
):
    """Evaluate C(x)B_j(x) for one mode on a sequence of x/c coordinates."""

    use_class_shape = mode_apply_class_shape(mode, class_shape)
    values = []
    for x in x_over_c:
        value = mode_basis_value(mode, x)
        if use_class_shape:
            value *= class_shape_factor(
                x,
                class_shape,
                class_shape_exponent=class_shape_exponent,
            )
        values.append(value)

    if normalize:
        if str(normalization_mode).strip().lower() != "max":
            raise BSplineModeError("Only max normalization is supported")
        factor = mode_normalization_factor(
            mode,
            class_shape=class_shape,
            class_shape_exponent=class_shape_exponent,
            normalization_samples=normalization_samples,
        )
        if factor > 0.0:
            values = [value / factor for value in values]

    return values


def evaluate_all_modes(spec, x_over_c, sides=None):
    """Evaluate every active mode and optionally mask by upper/lower side."""

    validate_mode_spec(spec)
    x_values = [float(x) for x in x_over_c]
    side_values = None
    if sides is not None:
        side_values = [str(side).strip().lower() for side in sides]
        if len(side_values) != len(x_values):
            raise BSplineModeError("sides must have the same length as x_over_c")

    class_shape = spec.get("class_shape", "sqrt_x_one_minus_x")
    class_shape_exponent = spec.get("class_shape_exponent", 0.5)
    normalize = bool(spec.get("normalize_basis", True))
    normalization_mode = spec.get("normalization_mode", "max")

    values_by_id = {}
    for mode in spec.get("modes", []):
        if mode.get("active", True) is False:
            continue
        values = evaluate_mode_values(
            mode,
            x_values,
            class_shape=class_shape,
            class_shape_exponent=class_shape_exponent,
            normalize=normalize,
            normalization_mode=normalization_mode,
        )
        if side_values is not None:
            mode_side = str(mode["side"]).strip().lower()
            values = [
                value if side == mode_side else 0.0
                for value, side in zip(values, side_values)
            ]
        values_by_id[str(mode["id"])] = values

    return values_by_id


def evaluate_normal_displacement(spec, x_over_c, sides):
    """Return scalar normal displacement values and per-mode shape values."""

    values_by_id = evaluate_all_modes(spec, x_over_c, sides=sides)
    total = [0.0 for _ in x_over_c]

    for mode in spec.get("modes", []):
        if mode.get("active", True) is False:
            continue
        coefficient = float(mode.get("coefficient", 0.0))
        values = values_by_id[str(mode["id"])]
        total = [
            current + coefficient * value
            for current, value in zip(total, values)
        ]

    return total, values_by_id
