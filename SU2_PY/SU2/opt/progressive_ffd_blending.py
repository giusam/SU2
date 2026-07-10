#!/usr/bin/env python

"""Blending utilities shared by progressive FFD mesh editors."""

from dataclasses import dataclass
import math
import re


BEZIER = "BEZIER"
BSPLINE_UNIFORM = "BSPLINE_UNIFORM"
SUPPORTED_FFD_BLENDINGS = (BEZIER, BSPLINE_UNIFORM)


@dataclass(frozen=True)
class FFDBlendingSpec:
    kind: str = BEZIER
    orders: tuple = (2, 2, 2)

    def order_for_axis(self, axis):
        return int(self.orders[int(axis)])


def _validated_unit_parameter(value, name="parameter"):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must lie in [0,1], got {value:.16g}")
    return value


def normalize_ffd_blending(value):
    kind = str(value or BEZIER).strip().upper()
    if kind not in SUPPORTED_FFD_BLENDINGS:
        raise ValueError(
            f"FFD_BLENDING must be one of {SUPPORTED_FFD_BLENDINGS}, got {value!r}"
        )
    return kind


def parse_bspline_orders(value, default=(2, 2, 2)):
    if value is None or value == "":
        values = list(default)
    elif isinstance(value, str):
        tokens = re.findall(r"[-+]?\d+(?:\.\d+)?", value)
        values = [float(token) for token in tokens]
    elif isinstance(value, (tuple, list)):
        values = list(value)
    else:
        try:
            values = list(value)
        except TypeError as exc:
            raise ValueError("FFD_BSPLINE_ORDER must contain three values") from exc

    if len(values) != 3:
        raise ValueError(
            f"FFD_BSPLINE_ORDER must contain three values, got {values!r}"
        )
    orders = []
    for value in values:
        numeric = float(value)
        integer = int(round(numeric))
        if not math.isfinite(numeric) or abs(numeric - integer) > 1.0e-12:
            raise ValueError("FFD_BSPLINE_ORDER values must be finite integers")
        if integer < 2:
            raise ValueError("FFD_BSPLINE_ORDER values must be >= 2")
        orders.append(integer)
    return tuple(orders)


def make_blending_spec(kind=BEZIER, orders=(2, 2, 2)):
    return FFDBlendingSpec(
        normalize_ffd_blending(kind),
        parse_bspline_orders(orders),
    )


def validate_blending_spec(spec, control_counts=None, dual_2d=False):
    if not isinstance(spec, FFDBlendingSpec):
        spec = make_blending_spec(spec)
    if spec.kind == BEZIER:
        return spec

    if dual_2d and (spec.orders[1] != 2 or spec.orders[2] != 2):
        raise ValueError(
            "Dual 2D BSPLINE_UNIFORM requires FFD_BSPLINE_ORDER=(order_i,2,2)"
        )
    if control_counts is not None:
        counts = list(control_counts)
        if len(counts) != 3:
            raise ValueError("control_counts must contain i, j, and k counts")
        for axis, (order, count) in enumerate(zip(spec.orders, counts)):
            if int(order) > int(count):
                raise ValueError(
                    f"B-spline order {order} exceeds control-point count {count} "
                    f"on axis {axis}"
                )
    return spec


def open_uniform_knot_vector(ncontrol, order):
    ncontrol = int(ncontrol)
    order = int(order)
    if ncontrol < 1:
        raise ValueError("A B-spline requires at least one control point")
    if order < 2 or order > ncontrol:
        raise ValueError(
            f"B-spline order must satisfy 2 <= order <= ncontrol, got "
            f"order={order}, ncontrol={ncontrol}"
        )
    knots = [0.0] * (order + ncontrol)
    for index in range(ncontrol - order):
        knots[order + index] = float(index + 1) / float(ncontrol - order + 1)
    for index in range(ncontrol - order, ncontrol):
        knots[order + index] = 1.0
    return knots


def _binomial(n, i):
    return math.factorial(n) / float(math.factorial(i) * math.factorial(n - i))


def bezier_basis_values(ncontrol, t):
    ncontrol = int(ncontrol)
    if ncontrol <= 0:
        return []
    degree = ncontrol - 1
    t = _validated_unit_parameter(t, "Bezier parameter")
    omt = 1.0 - t
    return [
        _binomial(degree, index)
        * (t ** index)
        * (omt ** (degree - index))
        for index in range(ncontrol)
    ]


def bspline_basis_values(ncontrol, order, t):
    ncontrol = int(ncontrol)
    order = int(order)
    knots = open_uniform_knot_vector(ncontrol, order)
    t = _validated_unit_parameter(t, "B-spline parameter")
    if t >= 1.0:
        values = [0.0] * ncontrol
        values[-1] = 1.0
        return values

    degree_zero = [0.0] * ncontrol
    for index in range(ncontrol):
        if knots[index] <= t < knots[index + 1]:
            degree_zero[index] = 1.0

    previous = degree_zero
    for degree in range(1, order):
        current = [0.0] * ncontrol
        for index in range(ncontrol):
            left = 0.0
            left_den = knots[index + degree] - knots[index]
            if left_den > 0.0:
                left = (t - knots[index]) / left_den * previous[index]

            right = 0.0
            if index + 1 < ncontrol:
                right_den = knots[index + degree + 1] - knots[index + 1]
                if right_den > 0.0:
                    right = (
                        (knots[index + degree + 1] - t)
                        / right_den
                        * previous[index + 1]
                    )
            current[index] = left + right
        previous = current
    return previous


def basis_values(ncontrol, t, spec, axis=0):
    if not isinstance(spec, FFDBlendingSpec):
        spec = make_blending_spec(spec)
    if spec.kind == BEZIER:
        return bezier_basis_values(ncontrol, t)
    return bspline_basis_values(ncontrol, spec.order_for_axis(axis), t)


def evaluate_curve(values, t, spec, axis=0):
    values = [float(value) for value in values]
    if not values:
        return 0.0
    weights = basis_values(len(values), t, spec, axis=axis)
    return sum(weight * value for weight, value in zip(weights, values))


def invert_monotone_curve(values, target, spec, axis=0, iterations=80):
    values = [float(value) for value in values]
    if not values:
        raise ValueError("Cannot invert an empty curve")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Curve control values must be finite")

    target = float(target)
    if not math.isfinite(target):
        raise ValueError("Curve inversion target must be finite")

    nondecreasing = all(right >= left for left, right in zip(values[:-1], values[1:]))
    nonincreasing = all(right <= left for left, right in zip(values[:-1], values[1:]))
    if not nondecreasing and not nonincreasing:
        raise ValueError("Curve control values must be monotone")

    lower = min(values[0], values[-1])
    upper = max(values[0], values[-1])
    if target < lower or target > upper:
        raise ValueError(
            "Curve inversion target lies outside the endpoint range: "
            f"target={target:.16g}, range=[{lower:.16g},{upper:.16g}]"
        )

    if len(values) == 1 or values[0] == values[-1]:
        if target != values[0]:
            raise ValueError("A constant curve can only be inverted at its value")
        return 0.0

    if target == values[0]:
        return 0.0
    if target == values[-1]:
        return 1.0

    iterations = int(iterations)
    if iterations <= 0:
        raise ValueError("Curve inversion iterations must be positive")

    increasing = values[-1] >= values[0]
    lo = 0.0
    hi = 1.0
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        value = evaluate_curve(values, mid, spec, axis=axis)
        if (increasing and value < target) or (not increasing and value > target):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
