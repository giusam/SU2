"""SU2-style native objective constraints for the B-spline driver."""

import math
import re
from dataclasses import dataclass

from .errors import BSplineSU2DriverError, _as_float, _normalized_name


_CONSTRAINT_RE = re.compile(
    r"^\(?([A-Za-z0-9_]+)([<>=])"
    r"([+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[Ee][+-]?\d+)?)\)?"
    r"(?:\*([+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[Ee][+-]?\d+)?))?$"
)


@dataclass(frozen=True)
class NativeConstraint:
    name: str
    sign: str
    target: float
    scale: float = 1.0

    @property
    def kind(self):
        return "EQUALITY" if self.sign == "=" else "INEQUALITY"

    @property
    def normalized_name(self):
        return _normalized_name(self.name).upper()

    def as_dict(self):
        return {
            "name": self.name,
            "sign": self.sign,
            "target": float(self.target),
            "scale": float(self.scale),
            "kind": self.kind,
        }


def native_constraint_internal_form(spec, current_value):
    sign = str(spec.sign).strip()
    target = float(spec.target)
    current_value = float(current_value)
    if sign == "<":
        return {
            "c_value": target - current_value,
            "field_sign": -1.0,
            "representation": "c = target - F >= 0",
            "lambda_bounds": (0.0, math.inf),
            "bounds_reason": "internal c>=0: F < target -> c=target-F, lambda>=0",
        }
    if sign == ">":
        return {
            "c_value": current_value - target,
            "field_sign": 1.0,
            "representation": "c = F - target >= 0",
            "lambda_bounds": (0.0, math.inf),
            "bounds_reason": "internal c>=0: F > target -> c=F-target, lambda>=0",
        }
    if sign == "=":
        return {
            "c_value": current_value - target,
            "field_sign": 1.0,
            "representation": "c = F - target = 0",
            "lambda_bounds": (-math.inf, math.inf),
            "bounds_reason": "internal c=0: equality uses free lambda",
        }
    raise BSplineSU2DriverError(
        f"unsupported OPT_CONSTRAINT sign {sign!r} for native constraint {spec.name}"
    )


def native_constraint_active_status(sign, c_value, active_tol):
    if str(sign).strip() == "=":
        return "equality"
    if float(c_value) < 0.0:
        return "violated"
    if float(c_value) <= float(active_tol):
        return "near_active"
    return "inactive"


def native_constraint_status(spec, current_value, active_tol):
    internal = native_constraint_internal_form(spec, current_value)
    return (
        native_constraint_active_status(
            spec.sign,
            internal["c_value"],
            active_tol,
        ),
        internal,
    )


def _constraint_from_parts(name, sign, target, scale=1.0):
    name = str(name or "").strip().strip('"').strip("'").upper()
    sign = str(sign or "").strip()
    if not name:
        raise BSplineSU2DriverError("OPT_CONSTRAINT has an empty function name")
    if sign not in ("<", ">", "="):
        raise BSplineSU2DriverError(
            "OPT_CONSTRAINT sign must be one of '<', '>', or '='; got {!r}".format(
                sign
            )
        )
    target = _as_float(target, f"OPT_CONSTRAINT {name} target")
    scale = _as_float(scale, f"OPT_CONSTRAINT {name} scale")
    if scale == 0.0:
        raise BSplineSU2DriverError(
            f"OPT_CONSTRAINT scale for {name} must be non-zero"
        )
    return NativeConstraint(name=name, sign=sign, target=target, scale=scale)


def _parse_constraint_text(value):
    text = str(value or "").strip().strip('"').strip("'")
    if not text or text.upper() == "NONE":
        return []
    constraints = []
    for raw_item in text.split(";"):
        item = "".join(str(raw_item).split())
        if not item:
            continue
        match = _CONSTRAINT_RE.match(item)
        if not match:
            raise BSplineSU2DriverError(
                "OPT_CONSTRAINT entry {!r} must look like "
                "(LIFT>0.5)*1.0, (DRAG<0.02), or (MOMENT_Z=0.0)".format(
                    raw_item.strip()
                )
            )
        name, sign, target, scale = match.groups()
        constraints.append(
            _constraint_from_parts(
                name,
                sign,
                target,
                1.0 if scale is None else scale,
            )
        )
    return constraints


def _constraint_from_mapping(item):
    data = {str(key).strip().upper(): value for key, value in dict(item).items()}
    name = (
        data.get("NAME")
        or data.get("FUNCTION")
        or data.get("OBJECTIVE")
        or data.get("FIELD")
    )
    target = data.get("TARGET", data.get("VALUE"))
    sign = data.get("SIGN", "=" if data.get("KIND", "").upper() == "EQUALITY" else None)
    scale = data.get("SCALE", 1.0)
    return _constraint_from_parts(name, sign, target, scale)


def _constraints_from_su2_dict(value):
    constraints = []
    data_by_key = {str(key).strip().upper(): val for key, val in dict(value).items()}
    equality = dict(data_by_key.get("EQUALITY", {}) or {})
    inequality = dict(data_by_key.get("INEQUALITY", {}) or {})
    for name, data in equality.items():
        data = {str(key).strip().upper(): val for key, val in dict(data or {}).items()}
        constraints.append(
            _constraint_from_parts(
                name,
                data.get("SIGN", "="),
                data.get("VALUE"),
                data.get("SCALE", 1.0),
            )
        )
    for name, data in inequality.items():
        data = {str(key).strip().upper(): val for key, val in dict(data or {}).items()}
        constraints.append(
            _constraint_from_parts(
                name,
                data.get("SIGN"),
                data.get("VALUE"),
                data.get("SCALE", 1.0),
            )
        )
    return constraints


def normalize_native_constraints(value):
    """Return a list of NativeConstraint objects from SU2-style input."""

    if value is None:
        return []
    if isinstance(value, str):
        return _parse_constraint_text(value)
    if isinstance(value, NativeConstraint):
        return [value]
    if isinstance(value, dict):
        upper_keys = {str(key).strip().upper() for key in value}
        if upper_keys & {"EQUALITY", "INEQUALITY"}:
            return _constraints_from_su2_dict(value)
        return [_constraint_from_mapping(value)]
    if isinstance(value, (list, tuple)):
        constraints = []
        for item in value:
            constraints.extend(normalize_native_constraints(item))
        return constraints
    raise BSplineSU2DriverError(
        "OPT_CONSTRAINT must be a SU2-style string, mapping, or list; got {}".format(
            type(value).__name__
        )
    )
