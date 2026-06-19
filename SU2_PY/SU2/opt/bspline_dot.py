#!/usr/bin/env python

"""Project SU2 surface sensitivities onto external B-spline modes."""

import argparse
import csv
import json
import math
import shlex
import warnings

from SU2.opt.bspline_modes import (
    BSplineModeError,
    evaluate_all_modes,
    load_mode_spec,
    validate_mode_spec,
)


class BSplineDotError(RuntimeError):
    pass


_NODE_ID_ALIASES = {
    "nodeid",
    "point",
    "pointid",
    "pointindex",
    "globalindex",
    "index",
}

_SENSITIVITY_X_ALIASES = {
    "sensitivityx",
    "sensx",
    "djdx",
    "dfdx",
}

_SENSITIVITY_Y_ALIASES = {
    "sensitivityy",
    "sensy",
    "djdy",
    "dfdy",
}

_SURFACE_SENSITIVITY_ALIASES = {
    "surfacesensitivity",
    "sensitivity",
    "normalsensitivity",
}

_METADATA_ALIASES = {
    "node_id": _NODE_ID_ALIASES,
    "x": {"x"},
    "y": {"y"},
    "x_over_c": {"xoverc", "xoc"},
    "side": {"side"},
    "normal_x": {"normalx", "nx"},
    "normal_y": {"normaly", "ny"},
    "deform_dir_x": {"deformdirx", "deformationdirectionx", "dirx"},
    "deform_dir_y": {"deformdiry", "deformationdirectiony", "diry"},
    "weight": {"weight", "w"},
    "deformed_x": {"deformedx"},
    "deformed_y": {"deformedy"},
}

_GRADIENT_FIELDNAMES = [
    "mode_id",
    "side",
    "basis_type",
    "coefficient",
    "lower_bound",
    "upper_bound",
    "gradient",
    "projection_mode",
    "sensitivity_weighting",
    "raw_norm",
    "weighted_phi_norm",
]


def _normalized_name(value):
    return "".join(
        char.lower()
        for char in str(value).strip().strip('"').strip("'")
        if char.isalnum()
    )


def _as_float(value, name):
    try:
        result = float(str(value).strip())
    except Exception:
        raise BSplineDotError(f"{name} must be numeric")
    if not math.isfinite(result):
        raise BSplineDotError(f"{name} must be finite")
    return result


def _as_optional_float(value, name):
    if value is None or str(value).strip() == "":
        return None
    return _as_float(value, name)


def _as_int(value, name):
    try:
        raw = float(str(value).strip())
    except Exception:
        raise BSplineDotError(f"{name} must be an integer")
    result = int(round(raw))
    if abs(raw - result) > 1.0e-8:
        raise BSplineDotError(f"{name} must be an integer")
    return result


def _as_optional_int(value, name):
    if value is None or str(value).strip() == "":
        return None
    return _as_int(value, name)


def normalize_sensitivity_weighting(value):
    value = str(value or "NODAL").strip().upper()
    if value not in ("NODAL", "DENSITY"):
        raise BSplineDotError(
            "sensitivity weighting must be 'NODAL' or 'DENSITY'; got {!r}".format(value)
        )
    return value


def _is_number(value):
    try:
        float(str(value).strip())
    except Exception:
        return False
    return True


def _is_int_like(value):
    try:
        raw = float(str(value).strip())
    except Exception:
        return False
    return math.isfinite(raw) and abs(raw - int(round(raw))) <= 1.0e-8


def _find_columns(fieldnames, aliases_by_key):
    columns = {}
    for key, aliases in aliases_by_key.items():
        for fieldname in fieldnames:
            if _normalized_name(fieldname) in aliases:
                columns[key] = fieldname
                break
    return columns


def read_metadata(metadata_filename):
    """Read BSPLINE_DEF surface metadata."""

    required = [
        "node_id",
        "x",
        "y",
        "x_over_c",
        "side",
        "normal_x",
        "normal_y",
        "weight",
        "deformed_x",
        "deformed_y",
    ]

    with open(metadata_filename, "r", newline="") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames:
            raise BSplineDotError(f"{metadata_filename} is missing a header row")
        columns = _find_columns(reader.fieldnames, _METADATA_ALIASES)
        missing = [name for name in required if name not in columns]
        if missing:
            raise BSplineDotError(
                "{} is missing required metadata column(s): {}".format(
                    metadata_filename,
                    ", ".join(missing),
                )
            )

        records = []
        for row_number, row in enumerate(reader, start=2):
            side = str(row[columns["side"]]).strip().lower()
            if side not in ("upper", "lower"):
                raise BSplineDotError(
                    f"{metadata_filename}:{row_number} side must be 'upper' or 'lower'"
                )

            weight = _as_float(
                row[columns["weight"]],
                f"{metadata_filename}:{row_number} weight",
            )
            if weight < 0.0:
                raise BSplineDotError(
                    f"{metadata_filename}:{row_number} weight must be non-negative"
                )

            normal_x = _as_float(
                row[columns["normal_x"]],
                f"{metadata_filename}:{row_number} normal_x",
            )
            normal_y = _as_float(
                row[columns["normal_y"]],
                f"{metadata_filename}:{row_number} normal_y",
            )
            if "deform_dir_x" in columns:
                deform_dir_x = _as_float(
                    row[columns["deform_dir_x"]],
                    f"{metadata_filename}:{row_number} deform_dir_x",
                )
            else:
                deform_dir_x = normal_x
            if "deform_dir_y" in columns:
                deform_dir_y = _as_float(
                    row[columns["deform_dir_y"]],
                    f"{metadata_filename}:{row_number} deform_dir_y",
                )
            else:
                deform_dir_y = normal_y

            records.append(
                {
                    "node_id": _as_int(
                        row[columns["node_id"]],
                        f"{metadata_filename}:{row_number} node_id",
                    ),
                    "x": _as_float(row[columns["x"]], f"{metadata_filename}:{row_number} x"),
                    "y": _as_float(row[columns["y"]], f"{metadata_filename}:{row_number} y"),
                    "x_over_c": _as_float(
                        row[columns["x_over_c"]],
                        f"{metadata_filename}:{row_number} x_over_c",
                    ),
                    "side": side,
                    "normal_x": normal_x,
                    "normal_y": normal_y,
                    "deform_dir_x": deform_dir_x,
                    "deform_dir_y": deform_dir_y,
                    "weight": weight,
                    "deformed_x": _as_float(
                        row[columns["deformed_x"]],
                        f"{metadata_filename}:{row_number} deformed_x",
                    ),
                    "deformed_y": _as_float(
                        row[columns["deformed_y"]],
                        f"{metadata_filename}:{row_number} deformed_y",
                    ),
                }
            )

    if not records:
        raise BSplineDotError(f"{metadata_filename} has no metadata rows")
    return records


def _sensitivity_column_key(fieldname):
    name = _normalized_name(fieldname)
    if name in _NODE_ID_ALIASES:
        return "node_id"
    if name in _SENSITIVITY_X_ALIASES:
        return "sensitivity_x"
    if name in _SENSITIVITY_Y_ALIASES:
        return "sensitivity_y"
    if name in _SURFACE_SENSITIVITY_ALIASES:
        return "surface_sensitivity"
    return None


def _split_fields(line):
    if "," in line:
        return [field.strip() for field in next(csv.reader([line], skipinitialspace=True))]
    if ";" in line:
        return [
            field.strip()
            for field in next(csv.reader([line], delimiter=";", skipinitialspace=True))
        ]
    try:
        return [field.strip() for field in shlex.split(line)]
    except ValueError:
        return [field.strip() for field in line.split()]


def _read_sensitivity_lines(sensitivity_filename):
    lines = []
    with open(sensitivity_filename, "r") as fp:
        for raw_line in fp:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("%"):
                continue
            if line.startswith("#"):
                candidate = line[1:].strip()
                columns = _mapped_sensitivity_columns(_split_fields(candidate))
                if not candidate or not _has_sensitivity_column(columns):
                    continue
                line = candidate
            if "%" in line:
                line = line.split("%", 1)[0].strip()
                if not line:
                    continue
            upper = line.upper()
            if upper.startswith("TITLE") or upper.startswith("ZONE"):
                continue
            if upper.startswith("VARIABLES"):
                if "=" not in line:
                    continue
                line = line.split("=", 1)[1].strip()
                if not line:
                    continue
            lines.append(line)
    return lines


def _looks_like_header(fields):
    return any(not _is_number(field) for field in fields)


def _mapped_sensitivity_columns(headers):
    columns = {}
    for index, header in enumerate(headers):
        key = _sensitivity_column_key(header)
        if key is not None and key not in columns:
            columns[key] = index
    return columns


def _has_sensitivity_column(columns):
    return (
        "surface_sensitivity" in columns
        or "sensitivity_x" in columns
        or "sensitivity_y" in columns
    )


def _record_from_mapped_fields(fields, columns, row_label):
    record = {}
    if "node_id" in columns and columns["node_id"] < len(fields):
        record["node_id"] = _as_optional_int(
            fields[columns["node_id"]],
            f"{row_label} node_id",
        )
    if "sensitivity_x" in columns and columns["sensitivity_x"] < len(fields):
        record["sensitivity_x"] = _as_optional_float(
            fields[columns["sensitivity_x"]],
            f"{row_label} sensitivity_x",
        )
    if "sensitivity_y" in columns and columns["sensitivity_y"] < len(fields):
        record["sensitivity_y"] = _as_optional_float(
            fields[columns["sensitivity_y"]],
            f"{row_label} sensitivity_y",
        )
    if "surface_sensitivity" in columns and columns["surface_sensitivity"] < len(fields):
        record["surface_sensitivity"] = _as_optional_float(
            fields[columns["surface_sensitivity"]],
            f"{row_label} surface_sensitivity",
        )
    return record


def _first_column_looks_like_node_ids(parsed_rows):
    ids = []
    for fields in parsed_rows:
        if not fields:
            continue
        if not _is_int_like(fields[0]):
            return False
        ids.append(_as_int(fields[0], "inferred node_id"))
    return bool(ids) and len(ids) == len(set(ids))


def _infer_headerless_sensitivities(parsed_rows, sensitivity_filename):
    first_column_is_node_id = _first_column_looks_like_node_ids(parsed_rows)
    records = []

    for row_number, fields in enumerate(parsed_rows, start=1):
        if not fields:
            continue
        row_label = f"{sensitivity_filename}:{row_number}"
        record = {}

        if first_column_is_node_id:
            record["node_id"] = _as_int(fields[0], f"{row_label} node_id")
            if len(fields) == 2:
                record["surface_sensitivity"] = _as_float(
                    fields[1],
                    f"{row_label} surface_sensitivity",
                )
            elif len(fields) >= 3:
                record["sensitivity_x"] = _as_float(
                    fields[1],
                    f"{row_label} sensitivity_x",
                )
                record["sensitivity_y"] = _as_float(
                    fields[2],
                    f"{row_label} sensitivity_y",
                )
                if len(fields) >= 4:
                    record["surface_sensitivity"] = _as_float(
                        fields[3],
                        f"{row_label} surface_sensitivity",
                    )
            else:
                raise BSplineDotError(
                    f"{row_label} does not contain sensitivity values"
                )
        else:
            if len(fields) == 1:
                record["surface_sensitivity"] = _as_float(
                    fields[0],
                    f"{row_label} surface_sensitivity",
                )
            elif len(fields) >= 2:
                record["sensitivity_x"] = _as_float(
                    fields[0],
                    f"{row_label} sensitivity_x",
                )
                record["sensitivity_y"] = _as_float(
                    fields[1],
                    f"{row_label} sensitivity_y",
                )
                if len(fields) >= 3:
                    record["surface_sensitivity"] = _as_float(
                        fields[2],
                        f"{row_label} surface_sensitivity",
                    )
            else:
                raise BSplineDotError(
                    f"{row_label} does not contain sensitivity values"
                )

        records.append(record)

    return records


def read_sensitivity_file(sensitivity_filename):
    """Read a named or simple headerless SU2 sensitivity file."""

    lines = _read_sensitivity_lines(sensitivity_filename)
    if not lines:
        raise BSplineDotError(f"{sensitivity_filename} has no sensitivity rows")

    first_fields = _split_fields(lines[0])
    if _looks_like_header(first_fields):
        columns = _mapped_sensitivity_columns(first_fields)
        if not _has_sensitivity_column(columns):
            raise BSplineDotError(
                "{} header does not contain a recognized sensitivity column".format(
                    sensitivity_filename
                )
            )

        records = []
        for row_number, line in enumerate(lines[1:], start=2):
            fields = _split_fields(line)
            if not fields:
                continue
            records.append(
                _record_from_mapped_fields(
                    fields,
                    columns,
                    f"{sensitivity_filename}:{row_number}",
                )
            )
    else:
        parsed_rows = [_split_fields(line) for line in lines]
        records = _infer_headerless_sensitivities(parsed_rows, sensitivity_filename)

    if not records:
        raise BSplineDotError(f"{sensitivity_filename} has no sensitivity rows")
    return records


def _record_node_id(record, label):
    if "node_id" not in record or record.get("node_id") is None:
        return None
    return _as_int(record.get("node_id"), f"{label} node_id")


def match_sensitivities_to_metadata(metadata, sensitivities):
    """Return sensitivity records aligned to the metadata row order."""

    if not metadata:
        raise BSplineDotError("metadata has no rows")
    if not sensitivities:
        raise BSplineDotError("sensitivity file has no rows")

    metadata_node_ids = [
        _record_node_id(record, f"metadata row {index + 1}")
        for index, record in enumerate(metadata)
    ]
    sensitivity_node_ids = [
        _record_node_id(record, f"sensitivity row {index + 1}")
        for index, record in enumerate(sensitivities)
    ]

    metadata_has_ids = all(node_id is not None for node_id in metadata_node_ids)
    sensitivity_has_any_ids = any(node_id is not None for node_id in sensitivity_node_ids)

    if metadata_has_ids and sensitivity_has_any_ids:
        if not all(node_id is not None for node_id in sensitivity_node_ids):
            raise BSplineDotError(
                "sensitivity rows mix node_id values with missing node_id values"
            )

        by_node_id = {}
        for node_id, record in zip(sensitivity_node_ids, sensitivities):
            if node_id in by_node_id:
                raise BSplineDotError(f"duplicate sensitivity node_id {node_id}")
            by_node_id[node_id] = record

        missing = [node_id for node_id in metadata_node_ids if node_id not in by_node_id]
        if missing:
            preview = ", ".join(str(node_id) for node_id in missing[:8])
            if len(missing) > 8:
                preview += ", ..."
            raise BSplineDotError(
                "sensitivity file is missing rows for metadata node_id(s): "
                + preview
            )

        return [by_node_id[node_id] for node_id in metadata_node_ids]

    if len(metadata) != len(sensitivities):
        raise BSplineDotError(
            "cannot match sensitivities to metadata: sensitivity rows have no usable "
            "node_id and row counts differ (metadata={}, sensitivities={})".format(
                len(metadata),
                len(sensitivities),
            )
        )

    warnings.warn(
        "falling back to positional sensitivity/metadata matching because "
        "sensitivity rows have no usable node_id; verify that sensitivity file "
        "row order matches the B-spline metadata row order",
        RuntimeWarning,
        stacklevel=2,
    )
    return list(sensitivities)


def _all_have_vector(sensitivities):
    return all(
        record.get("sensitivity_x") is not None and record.get("sensitivity_y") is not None
        for record in sensitivities
    )


def _all_have_scalar(sensitivities):
    return all(record.get("surface_sensitivity") is not None for record in sensitivities)


def _choose_projection_mode(sensitivities, prefer_vector):
    has_vector = _all_have_vector(sensitivities)
    has_scalar = _all_have_scalar(sensitivities)

    if prefer_vector and has_vector:
        return "vector"
    if not prefer_vector and has_scalar:
        return "scalar"
    if has_vector:
        return "vector"
    if has_scalar:
        return "scalar"

    raise BSplineDotError(
        "sensitivity data must contain either complete vector columns "
        "(Sensitivity_x and Sensitivity_y) or a scalar Surface_Sensitivity column"
    )


def _metadata_values(metadata):
    x_over_c = []
    sides = []
    directions_x = []
    directions_y = []
    weights = []

    for index, record in enumerate(metadata, start=1):
        side = str(record.get("side")).strip().lower()
        if side not in ("upper", "lower"):
            raise BSplineDotError(f"metadata row {index} side must be 'upper' or 'lower'")

        weight = _as_float(record.get("weight"), f"metadata row {index} weight")
        if weight < 0.0:
            raise BSplineDotError(f"metadata row {index} weight must be non-negative")

        x_over_c.append(_as_float(record.get("x_over_c"), f"metadata row {index} x_over_c"))
        sides.append(side)
        directions_x.append(
            _as_float(
                record.get("deform_dir_x", record.get("normal_x")),
                f"metadata row {index} deform_dir_x",
            )
        )
        directions_y.append(
            _as_float(
                record.get("deform_dir_y", record.get("normal_y")),
                f"metadata row {index} deform_dir_y",
            )
        )
        weights.append(weight)

    return x_over_c, sides, directions_x, directions_y, weights


def project_bspline_gradients(
    mode_spec,
    metadata,
    sensitivities,
    prefer_vector=True,
    sensitivity_weighting="NODAL",
):
    """Project surface sensitivities onto active B-spline coefficients."""

    spec = validate_mode_spec(mode_spec)
    sensitivity_weighting = normalize_sensitivity_weighting(sensitivity_weighting)
    aligned_sensitivities = match_sensitivities_to_metadata(metadata, sensitivities)
    projection_mode = _choose_projection_mode(aligned_sensitivities, prefer_vector)
    x_over_c, sides, directions_x, directions_y, weights = _metadata_values(metadata)
    mode_values = evaluate_all_modes(spec, x_over_c, sides=sides)

    gradients = []
    for mode in spec.get("modes", []):
        if mode.get("active", True) is False:
            continue

        mode_id = str(mode["id"])
        phi_values = mode_values[mode_id]

        gradient = 0.0
        raw_norm_square = 0.0
        weighted_phi_norm_square = 0.0

        for phi, weight, dir_x, dir_y, sensitivity in zip(
            phi_values,
            weights,
            directions_x,
            directions_y,
            aligned_sensitivities,
        ):
            if projection_mode == "vector":
                projected_sensitivity = (
                    _as_float(sensitivity.get("sensitivity_x"), "sensitivity_x") * dir_x
                    + _as_float(sensitivity.get("sensitivity_y"), "sensitivity_y") * dir_y
                )
            else:
                projected_sensitivity = _as_float(
                    sensitivity.get("surface_sensitivity"),
                    "surface_sensitivity",
                )
            factor = weight if sensitivity_weighting == "DENSITY" else 1.0
            gradient += projected_sensitivity * phi * factor

            raw_norm_square += phi * phi
            weighted_phi_norm_square += phi * phi * weight

        bounds = mode.get("bounds")
        lower_bound = None
        upper_bound = None
        if bounds is not None:
            lower_bound = _as_float(bounds[0], f"mode {mode_id} lower_bound")
            upper_bound = _as_float(bounds[1], f"mode {mode_id} upper_bound")

        gradients.append(
            {
                "mode_id": mode_id,
                "side": str(mode.get("side", "")).strip().lower(),
                "basis_type": str(mode.get("basis_type", "")).strip().lower(),
                "coefficient": _as_float(
                    mode.get("coefficient", 0.0),
                    f"mode {mode_id} coefficient",
                ),
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "gradient": gradient,
                "projection_mode": projection_mode,
                "sensitivity_weighting": sensitivity_weighting,
                "raw_norm": math.sqrt(raw_norm_square),
                "weighted_phi_norm": math.sqrt(weighted_phi_norm_square),
            }
        )

    return gradients


def _format_csv_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return "{:.15g}".format(value)
    return value


def write_gradients_csv(gradients, output_filename):
    with open(output_filename, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=_GRADIENT_FIELDNAMES)
        writer.writeheader()
        for gradient in gradients:
            writer.writerow(
                {
                    fieldname: _format_csv_value(gradient.get(fieldname))
                    for fieldname in _GRADIENT_FIELDNAMES
                }
            )


def _summary(metadata, sensitivities, gradients):
    gradient_values = [record["gradient"] for record in gradients]
    return {
        "number_of_metadata_nodes": len(metadata),
        "number_of_sensitivity_nodes": len(sensitivities),
        "projection_mode": gradients[0]["projection_mode"] if gradients else None,
        "sensitivity_weighting": gradients[0]["sensitivity_weighting"] if gradients else None,
        "gradient_min": min(gradient_values) if gradient_values else None,
        "gradient_max": max(gradient_values) if gradient_values else None,
        "modes": [
            {
                "mode_id": record["mode_id"],
                "side": record["side"],
                "basis_type": record["basis_type"],
                "gradient": record["gradient"],
                "sensitivity_weighting": record["sensitivity_weighting"],
                "raw_norm": record["raw_norm"],
                "weighted_phi_norm": record["weighted_phi_norm"],
            }
            for record in gradients
        ],
    }


def write_summary_json(metadata, sensitivities, gradients, output_filename):
    with open(output_filename, "w") as fp:
        json.dump(_summary(metadata, sensitivities, gradients), fp, indent=2, sort_keys=True)
        fp.write("\n")


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Project SU2 surface sensitivities onto active B-spline modes."
    )
    parser.add_argument("--modes", required=True, help="B-spline mode JSON file")
    parser.add_argument(
        "--metadata",
        required=True,
        help="BSPLINE_DEF metadata CSV file",
    )
    parser.add_argument("--sens", required=True, help="SU2 sensitivity file")
    parser.add_argument(
        "--output",
        default="bspline_gradients.csv",
        help="Output projected-gradient CSV file",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Optional output JSON summary file",
    )
    parser.add_argument(
        "--sensitivity-weighting",
        default="NODAL",
        choices=("NODAL", "DENSITY"),
        help="Treat sensitivities as nodal/integrated values or densities requiring metadata weights",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--prefer-vector",
        dest="prefer_vector",
        action="store_true",
        help="Prefer vector Sensitivity_x/Sensitivity_y columns when available",
    )
    mode_group.add_argument(
        "--prefer-scalar",
        dest="prefer_vector",
        action="store_false",
        help="Prefer scalar Surface_Sensitivity when available",
    )
    parser.set_defaults(prefer_vector=True)
    return parser


def main(argv=None):
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        mode_spec = load_mode_spec(args.modes)
        metadata = read_metadata(args.metadata)
        sensitivities = read_sensitivity_file(args.sens)
        gradients = project_bspline_gradients(
            mode_spec,
            metadata,
            sensitivities,
            prefer_vector=args.prefer_vector,
            sensitivity_weighting=args.sensitivity_weighting,
        )
        write_gradients_csv(gradients, args.output)
        if args.summary:
            write_summary_json(metadata, sensitivities, gradients, args.summary)
    except (BSplineDotError, BSplineModeError, OSError) as exc:
        parser.error(str(exc))

    projection_mode = gradients[0]["projection_mode"] if gradients else "none"
    print(
        "Wrote {} projected B-spline gradient(s) to {} using {} sensitivities".format(
            len(gradients),
            args.output,
            projection_mode,
        )
    )
    if args.summary:
        print("Wrote {}".format(args.summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
