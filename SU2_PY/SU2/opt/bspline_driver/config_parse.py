"""Optimizer config parsing and config-template patching helpers."""

import json
from pathlib import Path

from .config_keys import UNSUPPORTED_OPT_CONFIG_KEYS
from .errors import BSplineSU2DriverError

class _ConfigDict(dict):
    pass

def _parse_optimizer_config_value(value):
    value = str(value).strip().strip('"').strip("'")
    upper = value.upper()
    if upper == "YES":
        return True
    if upper == "NO":
        return False
    try:
        if any(char in value for char in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value

def parse_optimizer_config(filename, warning_prefix="[BSPLINE_SU2_DRIVER]"):
    values = {}
    with open(filename, "r") as fp:
        for line_number, raw_line in enumerate(fp, start=1):
            line = raw_line.strip()
            if not line or line.startswith("%") or line.startswith("#"):
                continue
            if "=" not in line:
                raise BSplineSU2DriverError(
                    f"{filename}:{line_number} expected KEY= VALUE"
                )
            key, value = line.split("=", 1)
            key = key.strip().upper()
            if not key:
                raise BSplineSU2DriverError(f"{filename}:{line_number} has an empty key")
            parsed_value = _parse_optimizer_config_value(value)
            values[key] = parsed_value

    values["_optimizer_config_filename"] = str(Path(filename).resolve())
    for key in UNSUPPORTED_OPT_CONFIG_KEYS:
        if key in values:
            print(
                f"{warning_prefix} WARNING: {key} is parsed but not implemented yet; ignoring."
            )
    return values

def _format_config_atom(value):
    if isinstance(value, float):
        return "{:.15g}".format(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, separators=(",", ":"))
    if value is None:
        return ""
    return str(value)

def _format_config_value(value):
    if isinstance(value, (list, tuple)):
        return "( " + ", ".join(_format_config_atom(item) for item in value) + " )"
    return _format_config_atom(value)

def _line_config_key(line):
    stripped = line.lstrip()
    if not stripped or stripped.startswith("%") or stripped.startswith("#"):
        return None
    if "=" not in line:
        return None
    return line.split("=", 1)[0].strip()

def patch_config_template(template_filename, output_filename, updates):
    """Copy a SU2 config template and patch selected key/value assignments."""

    updates = {str(key).strip().upper(): value for key, value in updates.items()}
    seen = set()
    output_lines = []

    with open(template_filename, "r") as fp:
        for raw_line in fp:
            line = raw_line.rstrip("\n")
            key = _line_config_key(line)
            if key is not None and key.upper() in updates:
                update_key = key.upper()
                output_lines.append(
                    f"{key}= {_format_config_value(updates[update_key])}"
                )
                seen.add(update_key)
            else:
                output_lines.append(line)

    missing = [key for key in updates if key not in seen]
    if missing and output_lines and output_lines[-1].strip():
        output_lines.append("")
    for key in missing:
        output_lines.append(f"{key}= {_format_config_value(updates[key])}")

    Path(output_filename).parent.mkdir(parents=True, exist_ok=True)
    with open(output_filename, "w") as fp:
        fp.write("\n".join(output_lines))
        fp.write("\n")
