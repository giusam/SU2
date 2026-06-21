"""Shared parsing helpers for B-spline/SU2 optimization modules."""

import math
from pathlib import Path


def normalized_name(value):
    return "".join(
        char.lower()
        for char in str(value).strip().strip('"').strip("'")
        if char.isalnum()
    )


def parse_float(value, name="value", finite=False):
    try:
        result = float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if finite and not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def parse_int(value, name="value"):
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(f"{name} must be an integer") from exc


def parse_optional_float(value, name="value"):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return parse_float(value, name)


def parse_optional_int(value, name="value"):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return parse_int(value, name)


def parse_bool(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().upper()
    if text in ("YES", "TRUE", "1", "ON"):
        return True
    if text in ("NO", "FALSE", "0", "OFF"):
        return False
    raise ValueError(f"expected YES/NO boolean value, got {value!r}")


def parse_float_list(value, name="value", finite=False):
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if (text.startswith("(") and text.endswith(")")) or (
            text.startswith("[") and text.endswith("]")
        ):
            text = text[1:-1]
        tokens = [token for token in text.replace(",", " ").split() if token]
    elif isinstance(value, (list, tuple)):
        tokens = list(value)
    else:
        tokens = [value]
    return [parse_float(token, name, finite=finite) for token in tokens]


def relative_path(path, base):
    try:
        return str(Path(path).resolve().relative_to(Path(base).resolve()))
    except Exception:
        return str(path)
