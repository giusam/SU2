"""Adaptive B-spline errors and parsing wrappers."""

from SU2.opt.bspline_common import parse_bool, parse_float, parse_float_list

class BSplineAdaptiveError(RuntimeError):
    pass

def _as_float(value, name):
    try:
        return parse_float(value, name, finite=True)
    except ValueError as exc:
        raise BSplineAdaptiveError(str(exc))

def _as_bool(value, default=False):
    try:
        return parse_bool(value, default=default)
    except ValueError as exc:
        raise BSplineAdaptiveError(str(exc))

def _as_float_list(value, name):
    try:
        return parse_float_list(value, name, finite=True)
    except ValueError as exc:
        raise BSplineAdaptiveError(str(exc))
