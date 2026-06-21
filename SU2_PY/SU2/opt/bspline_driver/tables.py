"""Table readers used by the B-spline/SU2 driver."""

import csv
import shlex

from .errors import (
    BSplineSU2DriverError,
    _as_float,
    _normalized_name,
)

def _read_table(filename):
    lines = []
    with open(filename, "r") as fp:
        for raw_line in fp:
            line = raw_line.strip()
            if not line or line.startswith("%") or line.startswith("#"):
                continue
            upper = line.upper()
            if upper.startswith("TITLE"):
                continue
            if upper.startswith("VARIABLES") and "=" in line:
                line = line.split("=", 1)[1].strip()
            elif upper.startswith("VARIABLES"):
                continue
            lines.append(line)

    if not lines:
        raise BSplineSU2DriverError(f"{filename} has no readable table rows")

    if "," in lines[0]:
        rows = [
            [field.strip().strip('"').strip("'") for field in row]
            for row in csv.reader(lines, skipinitialspace=True)
            if row
        ]
    else:
        rows = []
        for line in lines:
            try:
                fields = shlex.split(line)
            except ValueError:
                fields = line.split()
            rows.append([field.strip().strip('"').strip("'") for field in fields])

    if not rows or len(rows) < 2:
        raise BSplineSU2DriverError(f"{filename} has no data rows")
    return rows[0], rows[1:]

def _find_column_index(headers, requested_column):
    requested = _normalized_name(requested_column)
    for index, header in enumerate(headers):
        if _normalized_name(header) == requested:
            return index
    raise BSplineSU2DriverError(
        "column {!r} was not found. Available columns: {}".format(
            requested_column,
            ", ".join(headers),
        )
    )

def read_objective_from_history(history_filename, objective_column):
    headers, rows = _read_table(history_filename)
    column_index = _find_column_index(headers, objective_column)

    value = None
    for row in rows:
        if column_index >= len(row):
            continue
        raw = row[column_index].strip()
        if raw:
            value = raw
    if value is None:
        raise BSplineSU2DriverError(
            f"{history_filename} has no values for column {objective_column!r}"
        )
    return _as_float(value, f"{history_filename} {objective_column}")

def read_bspline_gradients(gradients_filename, allow_nonfinite=False):
    with open(gradients_filename, "r", newline="") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames:
            raise BSplineSU2DriverError(f"{gradients_filename} is missing a header")
        mode_field = _find_field(reader.fieldnames, "mode_id")
        gradient_field = _find_field(reader.fieldnames, "gradient")

        gradients = {}
        for row_number, row in enumerate(reader, start=2):
            mode_id = str(row.get(mode_field, "")).strip()
            if not mode_id:
                raise BSplineSU2DriverError(
                    f"{gradients_filename}:{row_number} has an empty mode_id"
                )
            if mode_id in gradients:
                raise BSplineSU2DriverError(
                    f"{gradients_filename} has duplicate mode_id {mode_id!r}"
                )
            value = row.get(gradient_field, "")
            if allow_nonfinite:
                try:
                    gradients[mode_id] = float(value)
                except Exception:
                    raise BSplineSU2DriverError(
                        f"{gradients_filename}:{row_number} gradient must be numeric"
                    )
            else:
                gradients[mode_id] = _as_float(
                    value,
                    f"{gradients_filename}:{row_number} gradient",
                )
    return gradients

def _find_field(fieldnames, requested_field):
    requested = _normalized_name(requested_field)
    for fieldname in fieldnames:
        if _normalized_name(fieldname) == requested:
            return fieldname
    raise BSplineSU2DriverError(
        "field {!r} was not found. Available fields: {}".format(
            requested_field,
            ", ".join(fieldnames),
        )
    )

def read_gradient_vector(gradients_filename, mode_ids, allow_nonfinite=False):
    gradients = read_bspline_gradients(
        gradients_filename,
        allow_nonfinite=allow_nonfinite,
    )
    missing = [mode_id for mode_id in mode_ids if mode_id not in gradients]
    if missing:
        raise BSplineSU2DriverError(
            "bspline_gradients.csv is missing mode_id(s): "
            + ", ".join(str(mode_id) for mode_id in missing)
        )
    return [gradients[mode_id] for mode_id in mode_ids]
