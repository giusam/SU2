#!/usr/bin/env python

"""Accepted-design utilities shared by progressive HH and FFD workflows."""

import csv
import glob
import json
import math
import os


RANKING_DESIGN_MANIFEST = "progressive_ranking_design.json"


def _float_vector(values):
    if values is None:
        return None
    try:
        result = [float(value) for value in values]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in result):
        return None
    return result


def find_project_design(project, dv_values, tolerance=0.0):
    """Return the project Design whose vector matches ``dv_values`` exactly.

    Project.closest_design is intentionally not used here: a merely close DSN
    is not a valid geometry/adjoint anchor for a level transition.
    """

    target = _float_vector(dv_values)
    if target is None:
        raise RuntimeError("Cannot resolve a DSN without finite accepted DV values")

    closest_gap = None
    for design in getattr(project, "designs", []) or []:
        try:
            candidate = _float_vector(design.state.design_vector())
        except Exception:
            candidate = None
        if candidate is None or len(candidate) != len(target):
            continue

        gap = max(
            (abs(actual - expected) for actual, expected in zip(candidate, target)),
            default=0.0,
        )
        scale = max(
            [1.0]
            + [abs(value) for value in candidate]
            + [abs(value) for value in target]
        )
        if closest_gap is None or gap < closest_gap:
            closest_gap = gap
        if gap <= float(tolerance) * scale:
            return design

    detail = "no compatible DSN"
    if closest_gap is not None:
        detail = f"closest max-DV gap={closest_gap:.6e}"
    raise RuntimeError(
        "No DSN exactly matches the accepted SLSQP design vector " + detail
    )


def resolve_design_directory(level_dir, design_or_folder):
    """Resolve a Project Design (or its folder string) inside ``level_dir``."""

    folder = getattr(design_or_folder, "folder", design_or_folder)
    if not folder:
        raise RuntimeError("Accepted SLSQP design has no DSN folder")
    folder = str(folder)
    if os.path.isabs(folder):
        path = os.path.abspath(folder)
    else:
        path = os.path.abspath(os.path.join(level_dir, folder))
    if not os.path.isdir(path):
        raise RuntimeError(f"Accepted SLSQP DSN directory does not exist: {path}")
    return path


def find_design_mesh(design_dir):
    """Return the mesh evaluated by one exact DSN.

    ``Design`` links the level input mesh into every DSN.  From the second
    progressive HH level onward that input already ends in ``_deform.su2``;
    a nonzero design then creates ``*_deform_deform.su2`` beside it.  A glob
    alone therefore cannot distinguish the linked input from the mesh
    generated for this design.

    ``config_DSN.cfg`` retains the level input in ``MESH_FILENAME``.  SU2's
    deformation driver deterministically appends ``_deform`` to that name, so
    prefer that exact regular output.  If it does not exist and no other
    generated mesh is present, the design is the zero-DV case and its linked
    input mesh is authoritative.
    """

    design_dir = os.path.abspath(design_dir)
    meshes = sorted(glob.glob(os.path.join(design_dir, "*.su2")))
    config_path = os.path.join(design_dir, "config_DSN.cfg")
    raw_input = _read_cfg_scalar(config_path, "MESH_FILENAME")
    if raw_input:
        input_name = str(raw_input).strip().strip("\"'")
        input_name = os.path.basename(input_name)
        input_path = os.path.join(design_dir, input_name)
        stem, extension = os.path.splitext(input_name)
        generated_path = os.path.join(
            design_dir,
            f"{stem}_deform{extension}",
        )

        if (
            generated_path != input_path
            and os.path.isfile(generated_path)
            and not os.path.islink(generated_path)
        ):
            return generated_path

        other_generated = [
            path
            for path in meshes
            if os.path.abspath(path) != os.path.abspath(input_path)
            and not os.path.islink(path)
        ]
        if other_generated:
            raise RuntimeError(
                "Accepted DSN lacks its expected generated mesh "
                f"{generated_path}; unexpected meshes: {other_generated}"
            )
        if os.path.exists(input_path):
            return input_path
        raise RuntimeError(
            f"Accepted DSN input mesh from config_DSN.cfg does not exist: "
            f"{input_path}"
        )

    # Legacy/test fallback for DSNs without config_DSN.cfg.  Prefer a unique
    # regular deformation output over linked inputs, but remain fail-fast on
    # genuine ambiguity.
    deformed = sorted(
        path
        for path in meshes
        if path.endswith("_deform.su2") and not os.path.islink(path)
    )
    if len(deformed) == 1:
        return deformed[0]
    if len(deformed) > 1:
        raise RuntimeError(
            f"Accepted DSN has ambiguous deformed meshes: {deformed}"
        )

    if len(meshes) == 1:
        return meshes[0]
    if not meshes:
        raise RuntimeError(f"Accepted DSN has no mesh: {design_dir}")
    raise RuntimeError(f"Accepted DSN has ambiguous meshes: {meshes}")


def _latest_direct_log(design_dir):
    logs = glob.glob(os.path.join(design_dir, "DIRECT", "log_Direct*.out"))
    logs = [path for path in logs if os.path.isfile(path)]
    if not logs:
        return None
    return max(logs, key=lambda path: (os.path.getmtime(path), path))


def _explicit_log_convergence(design_dir):
    log_path = _latest_direct_log(design_dir)
    if log_path is None:
        return None, "DIRECT log unavailable"
    try:
        with open(log_path, "r", errors="replace") as stream:
            text = stream.read()
    except OSError as exc:
        return None, f"DIRECT log unreadable: {exc}"

    success_pos = text.rfind("All convergence criteria satisfied.")
    failure_pos = max(
        text.rfind("before convergence."),
        text.rfind("before convergence\n"),
    )
    if success_pos < 0 and failure_pos < 0:
        return None, f"DIRECT log has no explicit convergence verdict: {log_path}"
    if success_pos > failure_pos:
        return True, f"explicit SU2 convergence verdict in {log_path}"
    return False, f"explicit SU2 non-convergence verdict in {log_path}"


def _normalize_csv_key(value):
    return str(value).strip().strip('"').strip().upper()


def _read_direct_last_row(design_dir):
    history_path = os.path.join(design_dir, "DIRECT", "history_direct.csv")
    if not os.path.isfile(history_path):
        return None
    try:
        with open(history_path, "r", newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader, None)
            rows = list(reader)
    except (OSError, csv.Error):
        return None
    if not header or not rows:
        return None
    keys = [_normalize_csv_key(key) for key in header]
    values = {}
    for key, raw in zip(keys, rows[-1]):
        try:
            values[key] = float(raw)
        except (TypeError, ValueError):
            continue
    return values


def _read_cfg_scalar(path, key):
    if not os.path.isfile(path):
        return None
    key = str(key).upper()
    try:
        with open(path, "r", errors="replace") as stream:
            for line in stream:
                content = line.split("%", 1)[0].strip()
                if "=" not in content:
                    continue
                lhs, rhs = content.split("=", 1)
                if lhs.strip().upper() != key:
                    continue
                return rhs.strip()
    except OSError:
        return None
    return None


def _history_convergence_fallback(design_dir):
    """Fallback for fixtures/legacy logs lacking SU2's explicit verdict."""

    cfg_path = os.path.join(design_dir, "DIRECT", "config_CFD.cfg")
    raw_threshold = _read_cfg_scalar(cfg_path, "CONV_RESIDUAL_MINVAL")
    try:
        threshold = float(raw_threshold)
    except (TypeError, ValueError):
        return None, "CONV_RESIDUAL_MINVAL unavailable"

    row = _read_direct_last_row(design_dir)
    if not row:
        return None, "DIRECT history unavailable"

    raw_fields = _read_cfg_scalar(cfg_path, "CONV_FIELD")
    if raw_fields:
        fields = [
            _normalize_csv_key(value)
            for value in raw_fields.strip("()[]").replace(",", " ").split()
        ]
    else:
        fields = ["RMS_DENSITY"]

    missing = [field for field in fields if field not in row]
    if missing:
        return None, f"DIRECT convergence fields unavailable: {missing}"
    converged = all(
        math.isfinite(row[field]) and row[field] <= threshold for field in fields
    )
    values = ", ".join(f"{field}={row[field]:.6e}" for field in fields)
    return converged, f"history fallback ({values}, threshold={threshold:.6e})"


def direct_convergence_status(design_dir):
    """Return ``(converged, detail)`` for the DIRECT solve in one DSN."""

    converged, detail = _explicit_log_convergence(design_dir)
    if converged is not None:
        return converged, detail
    fallback, fallback_detail = _history_convergence_fallback(design_dir)
    if fallback is not None:
        return fallback, fallback_detail
    return False, f"convergence could not be established ({detail}; {fallback_detail})"


def require_direct_convergence(design_dir):
    converged, detail = direct_convergence_status(design_dir)
    if not converged:
        raise RuntimeError(
            f"Refusing progressive level anchor with non-converged DIRECT solve: "
            f"{design_dir} ({detail})"
        )
    return detail


def require_objective_adjoint(design_dir, objective_name):
    objective_name = str(objective_name).strip().upper()
    adjoint_dir = os.path.join(design_dir, f"ADJOINT_{objective_name}")
    if not os.path.isdir(adjoint_dir):
        raise RuntimeError(
            "Accepted converged DSN lacks the objective adjoint required for "
            f"adaptive ranking: {adjoint_dir}"
        )
    dot_configs = [
        os.path.join(adjoint_dir, "config_DOT.cfg"),
        os.path.join(adjoint_dir, "config_DOT_AD.cfg"),
    ]
    if not any(os.path.isfile(path) for path in dot_configs):
        raise RuntimeError(
            "Accepted converged DSN lacks a DOT configuration required for "
            f"adaptive ranking: {adjoint_dir}"
        )
    return adjoint_dir


def write_ranking_design_manifest(
    level_dir,
    design_dir,
    mesh_path,
    dv_values,
    objective_name,
):
    """Persist the one DSN that every ranking asset must come from."""

    level_dir = os.path.abspath(level_dir)
    design_dir = os.path.abspath(design_dir)
    mesh_path = os.path.abspath(mesh_path)
    relative_design = os.path.relpath(design_dir, level_dir)
    if relative_design == os.pardir or relative_design.startswith(os.pardir + os.sep):
        raise RuntimeError(
            f"Ranking DSN must live inside its level directory: {design_dir}"
        )
    payload = {
        "design_dir": relative_design,
        "mesh": os.path.relpath(mesh_path, level_dir),
        "dv_values": _float_vector(dv_values),
        "objective": str(objective_name).strip().upper(),
        "direct_converged": True,
    }
    path = os.path.join(level_dir, RANKING_DESIGN_MANIFEST)
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)
    return path


def read_ranking_design_directory(level_dir):
    """Return the anchored DSN, or ``None`` when no transition anchor exists."""

    level_dir = os.path.abspath(level_dir)
    path = os.path.join(level_dir, RANKING_DESIGN_MANIFEST)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as stream:
            payload = json.load(stream)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Invalid progressive ranking-design manifest: {path}") from exc

    raw = payload.get("design_dir")
    if not raw:
        raise RuntimeError(f"Ranking-design manifest lacks design_dir: {path}")
    design_dir = (
        os.path.abspath(raw)
        if os.path.isabs(str(raw))
        else os.path.abspath(os.path.join(level_dir, str(raw)))
    )
    relative = os.path.relpath(design_dir, level_dir)
    if relative == os.pardir or relative.startswith(os.pardir + os.sep):
        raise RuntimeError(f"Ranking-design manifest escapes its level: {path}")
    if not os.path.isdir(design_dir):
        raise RuntimeError(f"Anchored ranking DSN does not exist: {design_dir}")
    return design_dir
