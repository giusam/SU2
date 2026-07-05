"""Diagnostic-only writers for adaptive B-spline knot scoring."""

import csv
import json
import math
import os
import traceback
from pathlib import Path

import numpy as np


DIAGNOSTIC_VERSION = 1
FILES_PER_LEVEL = [
    "00_context.json",
    "01_objective_signal.json",
    "02_ikkt_constraints.json",
    "03_ikkt_fit.json",
    "04_ikkt_residual.json",
    "05_nodal_fields.csv",
    "06_old_basis.json",
    "07_knot_span_scores_extended.csv",
    "08_ranking_summary.json",
    "09_selected_candidates.csv",
]
README_TEXT = """This directory contains diagnostic-only outputs for the adaptive B-spline knot-insertion scoring pipeline.

Diagnostics do not affect optimization behavior.

Each level_XXX directory is a snapshot of the data used to rank candidate knot insertions at that refinement level.

Important files:
- 05_nodal_fields.csv: nodal objective, constraint, lambda-weighted fields and IKKT residual.
- 07_knot_span_scores_extended.csv: per-candidate score/SVD/ranking diagnostics.
- 08_ranking_summary.json: compact ranking summary and warning flags.
- 09_selected_candidates.csv: candidates actually selected after penalties/batch logic.
"""


def _as_bool(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().upper()
    if text in ("YES", "Y", "TRUE", "T", "1", "ON"):
        return True
    if text in ("NO", "N", "FALSE", "F", "0", "OFF"):
        return False
    return bool(default)


def diagnostics_enabled(settings):
    return _as_bool((settings or {}).get("scoring_diagnostics", False), default=False)


def diagnostic_root(settings, workdir):
    root_name = str((settings or {}).get("scoring_diagnostic_dir", "DIAGNOSTIC") or "DIAGNOSTIC")
    return Path(workdir) / root_name


def level_diagnostic_dir(settings, workdir, level):
    return diagnostic_root(settings, workdir) / f"level_{int(level):03d}"


def ensure_level_diagnostic_dir(settings, workdir, level):
    directory = level_diagnostic_dir(settings, workdir, level)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def safe_float(value):
    try:
        value = float(value)
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    return value


def safe_json(obj):
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return safe_json(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(key): safe_json(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [safe_json(value) for value in obj]
    return str(obj)


def _csv_cell(value):
    value = safe_json(value)
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return "{:.15g}".format(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return value


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fp:
        json.dump(safe_json(payload), fp, indent=2, sort_keys=True, allow_nan=False)
        fp.write("\n")
    os.replace(tmp, path)


def write_csv_atomic(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(fieldnames)
    with open(tmp, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_cell(row.get(field, "")) for field in fieldnames})
    os.replace(tmp, path)


def append_or_rewrite_summary_levels(path, row, key="level"):
    path = Path(path)
    existing = []
    fieldnames = []
    if path.exists():
        with open(path, "r", newline="") as fp:
            reader = csv.DictReader(fp)
            fieldnames = list(reader.fieldnames or [])
            existing = [dict(item) for item in reader]
    key_value = str(row.get(key, ""))
    replaced = False
    for index, item in enumerate(existing):
        if str(item.get(key, "")) == key_value:
            existing[index] = dict(row)
            replaced = True
            break
    if not replaced:
        existing.append(dict(row))
    for item in [*existing, row]:
        for name in item:
            if name not in fieldnames:
                fieldnames.append(name)
    try:
        existing.sort(key=lambda item: int(float(item.get(key, 0))))
    except Exception:
        existing.sort(key=lambda item: str(item.get(key, "")))
    write_csv_atomic(path, existing, fieldnames)


def write_readme_once(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    readme = root / "README.md"
    if not readme.exists():
        tmp = readme.with_suffix(readme.suffix + ".tmp")
        with open(tmp, "w") as fp:
            fp.write(README_TEXT)
        os.replace(tmp, readme)


def update_manifest(root, level, case_name=None, diagnostic_dir="DIAGNOSTIC"):
    root = Path(root)
    manifest = root / "manifest.json"
    payload = {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "case": case_name,
        "created_by": "bspline_su2_adaptive",
        "diagnostic_dir": diagnostic_dir,
        "levels": [],
        "files_per_level": list(FILES_PER_LEVEL),
    }
    if manifest.exists():
        try:
            with open(manifest, "r") as fp:
                current = json.load(fp)
            if isinstance(current, dict):
                payload.update(current)
        except Exception:
            pass
    levels = {int(value) for value in payload.get("levels", [])}
    levels.add(int(level))
    payload["levels"] = sorted(levels)
    payload["diagnostic_version"] = DIAGNOSTIC_VERSION
    payload["case"] = case_name
    payload["diagnostic_dir"] = diagnostic_dir
    payload["files_per_level"] = list(FILES_PER_LEVEL)
    write_json_atomic(manifest, payload)


def warning_payload(message, exc=None):
    payload = {"status": "error", "message": str(message)}
    if exc is not None:
        payload["traceback"] = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
    return payload


def warn_diagnostic(message):
    print(f"[BSPLINE_SCORING_DIAGNOSTICS] warning | {message}")


def normalized_side(value):
    text = str(value or "").strip().lower()
    if text == "upper":
        return "upper"
    if text == "lower":
        return "lower"
    return text or "unknown"


def scoring_pass_active(settings):
    return diagnostics_enabled(settings) and isinstance(
        (settings or {}).get("_scoring_diagnostics_state"),
        dict,
    )


def initialize_level_diagnostics(
    settings,
    context,
    metadata,
    primary_signal,
    objective_signal=None,
    ikkt_diagnostics=None,
):
    if not diagnostics_enabled(settings):
        return
    settings["_scoring_diagnostics_state"] = {
        "context": dict(context or {}),
        "metadata": list(metadata or []),
        "primary_signal": np.asarray(primary_signal, dtype=float),
        "objective_signal": (
            np.asarray(objective_signal, dtype=float)
            if objective_signal is not None
            else np.asarray(primary_signal, dtype=float)
        ),
        "ikkt_diagnostics": dict(ikkt_diagnostics or {}),
        "passes": [],
        "next_scoring_pass_id": 1,
    }


def begin_scoring_pass(settings, batch_step, side=None):
    if not scoring_pass_active(settings):
        return
    state = settings["_scoring_diagnostics_state"]
    pass_id = int(state.get("next_scoring_pass_id", 1))
    state["next_scoring_pass_id"] = pass_id + 1
    settings["_diagnostic_scoring_pass_id"] = pass_id
    settings["_diagnostic_batch_step"] = int(batch_step)
    if side is not None:
        settings["_diagnostic_side"] = str(side).upper()


def record_scoring_pass(settings, payload):
    if not scoring_pass_active(settings):
        return
    try:
        settings["_scoring_diagnostics_state"].setdefault("passes", []).append(
            dict(payload or {})
        )
    except Exception as exc:
        warn_diagnostic(f"could not record scoring pass: {exc}")


def candidate_id(level, scoring_pass_id, batch_step, side, left, right, inserted):
    return (
        f"L{int(level):03d}:P{int(scoring_pass_id):03d}:B{int(batch_step):03d}:"
        f"{str(side).strip().upper()}:"
        f"{round(float(left), 14):.14f}:"
        f"{round(float(right), 14):.14f}:"
        f"{round(float(inserted), 14):.14f}"
    )


def finalize_candidate_ids(rows, level):
    for row in rows:
        batch_step = int(row.get("batch_step") or row.get("scoring_pass_id") or 1)
        scoring_pass_id = int(row.get("scoring_pass_id") or batch_step)
        row["scoring_pass_id"] = scoring_pass_id
        row["batch_step"] = batch_step
        row["candidate_id"] = candidate_id(
            level,
            scoring_pass_id,
            batch_step,
            row.get("side", "BOTH"),
            row["span_left"],
            row["span_right"],
            row["inserted_knot"],
        )


def _numeric_array(values):
    return np.asarray(values, dtype=float).reshape(-1)


def _side_masks(metadata, mask=None):
    if mask is None:
        mask = np.ones(len(metadata), dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    upper = np.asarray(
        [normalized_side(row.get("side")) == "upper" for row in metadata],
        dtype=bool,
    )
    lower = np.asarray(
        [normalized_side(row.get("side")) == "lower" for row in metadata],
        dtype=bool,
    )
    return mask, upper & mask, lower & mask


def field_summary(metadata, values, mask=None, prefix=""):
    values = _numeric_array(values)
    if mask is None:
        mask = np.ones(len(values), dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    _, upper_mask, lower_mask = _side_masks(metadata, mask)
    scoped = values[mask]
    energy = float(np.dot(scoped, scoped)) if scoped.size else 0.0
    eps = 1.0e-300

    def _norm(local_mask):
        local = values[local_mask]
        return float(np.linalg.norm(local)) if local.size else 0.0

    def _energy(local_mask):
        local = values[local_mask]
        return float(np.dot(local, local)) if local.size else 0.0

    max_abs = 0.0
    max_x = None
    max_side = None
    if scoped.size:
        scoped_indices = np.where(mask)[0]
        local_index = int(np.argmax(np.abs(scoped)))
        index = int(scoped_indices[local_index])
        max_abs = float(abs(values[index]))
        try:
            max_x = float(metadata[index].get("x_over_c"))
        except Exception:
            max_x = None
        max_side = normalized_side(metadata[index].get("side"))

    squared = np.sort(scoped * scoped)[::-1]
    top1 = int(max(1, math.ceil(0.01 * len(squared)))) if squared.size else 0
    top5 = int(max(1, math.ceil(0.05 * len(squared)))) if squared.size else 0
    upper_energy = _energy(upper_mask)
    lower_energy = _energy(lower_mask)
    data = {
        f"{prefix}norm": float(np.linalg.norm(scoped)) if scoped.size else 0.0,
        f"{prefix}norm_upper": _norm(upper_mask),
        f"{prefix}norm_lower": _norm(lower_mask),
        f"{prefix}energy": energy,
        f"{prefix}energy_upper": upper_energy,
        f"{prefix}energy_lower": lower_energy,
        f"{prefix}energy_fraction_upper": upper_energy / (energy + eps),
        f"{prefix}energy_fraction_lower": lower_energy / (energy + eps),
        f"{prefix}max_abs": max_abs,
        f"{prefix}max_abs_x_over_c": max_x,
        f"{prefix}max_abs_side": max_side,
    }
    if not prefix:
        data["top_1pct_energy_fraction"] = (
            float(np.sum(squared[:top1])) / (energy + eps) if top1 else 0.0
        )
        data["top_5pct_energy_fraction"] = (
            float(np.sum(squared[:top5])) / (energy + eps) if top5 else 0.0
        )
    return data


def objective_signal_payload(metadata, objective_signal, scoring_mask):
    payload = {
        "status": "ok",
        "n_nodes_total": int(len(metadata)),
        "n_nodes_scoring": int(np.sum(scoring_mask)),
        "n_closure_dropped": int(len(metadata) - np.sum(scoring_mask)),
    }
    payload.update(field_summary(metadata, objective_signal, scoring_mask))
    return payload


def _finite_dot_cosine(a, b):
    a = _numeric_array(a)
    b = _numeric_array(b)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1.0e-300
    return float(np.dot(a, b) / denom)


def ikkt_constraint_payload(metadata, objective_signal, ikkt_diagnostics, scoring_mask):
    if not ikkt_diagnostics:
        return {"status": "not_applicable", "reason": "score_mode_is_virtual_insertion"}
    nodal = ikkt_diagnostics.get("_nodal_fields_payload") or {}
    constraints_by_name = {
        item.get("name"): item for item in nodal.get("constraints", [])
        if isinstance(item, dict)
    }
    included = []
    for item in ikkt_diagnostics.get("included_constraints", []):
        row = dict(item)
        field_payload = constraints_by_name.get(item.get("name"), {})
        field = field_payload.get("fit_field")
        if field is not None:
            summary = field_summary(metadata, field, scoring_mask, prefix="field_")
            row.update(summary)
            row["cosine_with_objective"] = _finite_dot_cosine(
                _numeric_array(objective_signal)[scoring_mask],
                _numeric_array(field)[scoring_mask],
            )
        row["status"] = row.get("active_status") or row.get("sign") or "active"
        row["internal_c"] = row.get("c_value", row.get("gap"))
        row["used_in_ikkt_gradient"] = bool(row.get("used_in_ikkt_gradient", True))
        included.append(row)
    skipped = []
    for item in (
        list(ikkt_diagnostics.get("skipped_constraints", []))
        + list(ikkt_diagnostics.get("unsupported_constraints", []))
    ):
        row = dict(item)
        row["status"] = row.get("active_status") or row.get("reason", "skipped")
        row["used_in_ikkt_gradient"] = False
        skipped.append(row)
    return {"status": "ok", "included": included, "skipped": skipped}


def ikkt_fit_payload(ikkt_diagnostics):
    if not ikkt_diagnostics:
        return {"status": "not_applicable", "reason": "score_mode_is_virtual_insertion"}
    lsq = dict(ikkt_diagnostics.get("least_squares") or {})
    included = list(ikkt_diagnostics.get("included_constraints", []))
    zero = safe_float(lsq.get("zero_lambda_cost"))
    cost = safe_float(lsq.get("cost"))
    ratio = None
    if zero is not None and zero > 1.0e-300 and cost is not None:
        ratio = cost / zero
    lambdas = {}
    bounds = {}
    at_bound = {}
    for item in included:
        name = str(item.get("function_name") or item.get("name"))
        lam = safe_float(item.get("lambda"))
        lambdas[name] = lam
        bounds[name] = item.get("lambda_bounds")
        lower = None
        upper = None
        raw_bounds = item.get("lambda_bounds") or [None, None]
        if len(raw_bounds) >= 2:
            lower = safe_float(raw_bounds[0])
            upper = safe_float(raw_bounds[1])
        near = False
        if lam is not None:
            if lower is not None and abs(lam - lower) <= 1.0e-10:
                near = True
            if upper is not None and abs(lam - upper) <= 1.0e-10:
                near = True
        at_bound[name] = near
    lambda_values = [value for value in lambdas.values() if value is not None]
    skipped = [
        str(item.get("function_name") or item.get("name"))
        for item in (
            list(ikkt_diagnostics.get("skipped_constraints", []))
            + list(ikkt_diagnostics.get("unsupported_constraints", []))
        )
    ]
    return {
        "status": ikkt_diagnostics.get("status", "ok"),
        "zero_lambda_cost": zero,
        "cost": cost,
        "cost_ratio": ratio,
        "cost_improvement_ratio": None if ratio is None else 1.0 - ratio,
        "grad_constraint_rank": lsq.get("grad_constraint_rank"),
        "grad_constraint_condition": lsq.get("grad_constraint_condition"),
        "lambda_norm": float(np.linalg.norm(lambda_values)) if lambda_values else 0.0,
        "lambdas": lambdas,
        "lambda_bounds": bounds,
        "lambda_at_bound": at_bound,
        "active_constraints": list(lambdas.keys()),
        "skipped_constraints": skipped,
    }


def ikkt_residual_payload(metadata, objective_signal, primary_signal, ikkt_diagnostics, scoring_mask):
    if not ikkt_diagnostics:
        return {"status": "not_applicable", "reason": "score_mode_is_virtual_insertion"}
    objective = _numeric_array(objective_signal)
    residual = _numeric_array(primary_signal)
    combo = objective - residual
    objective_norm = float(np.linalg.norm(objective[scoring_mask]))
    combo_norm = float(np.linalg.norm(combo[scoring_mask]))
    residual_norm = float(np.linalg.norm(residual[scoring_mask]))
    eps = 1.0e-300
    payload = {
        "status": ikkt_diagnostics.get("status", "ok"),
        "objective_norm": objective_norm,
        "constraint_combo_norm": combo_norm,
        "residual_norm": residual_norm,
        "residual_to_objective_norm": residual_norm / (objective_norm + eps),
        "cancellation_ratio": combo_norm / (objective_norm + eps),
        "cosine_residual_objective": _finite_dot_cosine(
            residual[scoring_mask],
            objective[scoring_mask],
        ),
    }
    payload.update(field_summary(metadata, residual, scoring_mask, prefix="residual_"))
    return payload


def _constraint_column_name(name):
    key = str(name or "").upper()
    if key.endswith("[LIFT]") or key == "LIFT":
        return "lift"
    if key.endswith("[AIRFOIL_AREA]") or key == "AIRFOIL_AREA":
        return "area"
    if key.endswith("[MOMENT_Z]") or key == "MOMENT_Z":
        return "moment"
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in key).strip("_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe or "unknown"


def nodal_field_rows(metadata, objective_signal, primary_signal, ikkt_diagnostics, scoring_mask, scoring_pass_id):
    objective = _numeric_array(objective_signal)
    residual = _numeric_array(primary_signal)
    combo = objective - residual
    is_alias = not bool(ikkt_diagnostics)
    constraints = []
    if ikkt_diagnostics:
        constraints = list((ikkt_diagnostics.get("_nodal_fields_payload") or {}).get("constraints", []))
    rows = []
    dynamic_columns = set()
    for constraint in constraints:
        name = constraint.get("function_name") or constraint.get("name")
        stem = _constraint_column_name(name)
        dynamic_columns.add(f"constraint_{stem}_field")
        dynamic_columns.add(f"lambda_{stem}_times_field")
    for index, row in enumerate(metadata):
        base = {
            "scoring_pass_id": int(scoring_pass_id),
            "level": "",
            "node_id": row.get("node_id", index),
            "x": row.get("x"),
            "y": row.get("y"),
            "x_over_c": row.get("x_over_c"),
            "side": normalized_side(row.get("side")),
            "mask_scoring": bool(scoring_mask[index]),
            "objective_signal": objective[index] if index < len(objective) else "",
            "lift_field": "",
            "area_field": "",
            "moment_field": "",
            "lambda_lift_times_field": "",
            "lambda_area_times_field": "",
            "lambda_moment_times_field": "",
            "constraint_combo": 0.0 if is_alias else combo[index],
            "ikkt_residual": residual[index] if index < len(residual) else "",
            "ikkt_residual_is_alias_of_objective": is_alias,
        }
        for column in dynamic_columns:
            base[column] = ""
        for constraint in constraints:
            name = constraint.get("function_name") or constraint.get("name")
            stem = _constraint_column_name(name)
            field = constraint.get("fit_field")
            lam = safe_float(constraint.get("lambda"))
            if field is None:
                continue
            values = _numeric_array(field)
            value = values[index] if index < len(values) else ""
            if stem in ("lift", "area", "moment"):
                base[f"{stem}_field"] = value
                base[f"lambda_{stem}_times_field"] = "" if lam is None else lam * value
            else:
                base[f"constraint_{stem}_field"] = value
                base[f"lambda_{stem}_times_field"] = "" if lam is None else lam * value
        rows.append(base)
    return rows


def old_basis_payload(old_matrix, primary_signal, objective_signal, primary_signal_name, regularization):
    matrix = np.asarray(old_matrix, dtype=float)
    primary = _numeric_array(primary_signal)
    objective = _numeric_array(objective_signal)
    singular_values = []
    rank = 0
    condition = None
    if matrix.size and matrix.shape[1] > 0:
        try:
            singular_values = [float(value) for value in np.linalg.svd(matrix, compute_uv=False)]
            tol = max(
                float(regularization) ** 0.5,
                np.finfo(float).eps * max(matrix.shape) * float(singular_values[0] if singular_values else 0.0),
            )
            rank = int(sum(value > tol for value in singular_values))
            if rank > 0:
                condition = float(singular_values[0] / singular_values[rank - 1])
        except Exception:
            tol = float(regularization) ** 0.5
    else:
        tol = float(regularization) ** 0.5

    def _projection_stats(values, prefix):
        if matrix.size == 0 or matrix.shape[1] == 0:
            projected = np.zeros_like(values)
        else:
            try:
                coeffs, *_ = np.linalg.lstsq(matrix, values, rcond=None)
                projected = matrix.dot(coeffs)
            except Exception:
                projected = np.zeros_like(values)
        orthogonal = values - projected
        energy = float(np.dot(values, values))
        return {
            f"projected_{prefix}_norm": float(np.linalg.norm(projected)),
            f"orthogonal_{prefix}_norm": float(np.linalg.norm(orthogonal)),
            f"orthogonal_{prefix}_fraction": float(np.dot(orthogonal, orthogonal)) / (energy + 1.0e-300),
        }

    payload = {
        "status": "ok",
        "n_rows": int(matrix.shape[0]) if matrix.ndim == 2 else 0,
        "n_columns": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
        "rank": int(rank),
        "rank_tolerance": float(tol),
        "condition": condition,
        "singular_values": singular_values,
        "primary_signal_name": primary_signal_name,
        "primary_signal_norm": float(np.linalg.norm(primary)),
    }
    payload.update(_projection_stats(objective, "objective"))
    payload.update(_projection_stats(primary, "score_signal"))
    return payload


def ranking_summary_for_rows(rows, primary_signal_name, primary_signal_norm, warnings=None):
    warnings = list(warnings or [])
    rows = [row for row in rows if row.get("status", "ok") != "nonpositive_score"]
    scores = []
    for row in rows:
        value = safe_float(row.get("score"))
        scores.append(0.0 if value is None else max(0.0, value))
    total = float(sum(scores))
    eps = 1.0e-300
    sorted_pairs = sorted(
        zip(rows, scores),
        key=lambda item: (
            -item[1],
            float(item[0].get("span_left", 0.0)),
            float(item[0].get("span_right", 0.0)),
        ),
    )
    top_scores = [score for _row, score in sorted_pairs]
    p = np.asarray(top_scores, dtype=float) / (total + eps) if top_scores else np.asarray([])
    entropy = float(-np.sum(p * np.log(p + eps))) if p.size else 0.0
    entropy_norm = entropy / math.log(len(p)) if len(p) > 1 else 0.0
    upper_score = sum(
        score for row, score in zip(rows, scores)
        if normalized_side(row.get("side")) == "upper"
    )
    lower_score = sum(
        score for row, score in zip(rows, scores)
        if normalized_side(row.get("side")) == "lower"
    )
    top5 = []
    for index, (row, score) in enumerate(sorted_pairs[:5], start=1):
        top5.append(
            {
                "rank": index,
                "candidate_id": row.get("candidate_id"),
                "side": row.get("side"),
                "span_left": row.get("span_left"),
                "span_right": row.get("span_right"),
                "inserted_knot": row.get("inserted_knot"),
                "score": score,
                "score_effective": row.get("score_effective"),
            }
        )
    return {
        "score_mode": rows[0].get("score_mode") if rows else "",
        "primary_signal_name": primary_signal_name,
        "primary_signal_norm": primary_signal_norm,
        "n_candidates": int(len(rows)),
        "top1_score": top_scores[0] if len(top_scores) >= 1 else None,
        "top2_score": top_scores[1] if len(top_scores) >= 2 else None,
        "top1_top2_ratio": (
            top_scores[0] / (top_scores[1] + eps) if len(top_scores) >= 2 else None
        ),
        "top1_score_fraction": (
            sum(top_scores[:1]) / (total + eps) if top_scores else None
        ),
        "top3_score_fraction": (
            sum(top_scores[:3]) / (total + eps) if top_scores else None
        ),
        "top5_score_fraction": (
            sum(top_scores[:5]) / (total + eps) if top_scores else None
        ),
        "ranking_entropy": entropy,
        "ranking_entropy_normalized": entropy_norm,
        "upper_score_fraction": upper_score / (total + eps),
        "lower_score_fraction": lower_score / (total + eps),
        "top5": top5,
        "ikkt_vs_objective": ikkt_vs_objective_summary(rows),
        "warnings": warnings,
    }


def _rank_map(rows, rank_key, score_key):
    valid = [
        row for row in rows
        if row.get(score_key, "") != "" and safe_float(row.get(score_key)) is not None
    ]
    valid.sort(
        key=lambda row: (
            -float(row[score_key]),
            float(row.get("span_left", 0.0)),
            float(row.get("span_right", 0.0)),
        )
    )
    return {row.get("candidate_id"): index for index, row in enumerate(valid, start=1)}


def _spearman_from_ranks(a, b):
    common = [key for key in a if key in b]
    n = len(common)
    if n < 2:
        return None
    diffs = [(float(a[key]) - float(b[key])) ** 2 for key in common]
    return 1.0 - (6.0 * sum(diffs)) / (n * (n * n - 1.0))


def ikkt_vs_objective_summary(rows):
    if not any(row.get("score_ikkt", "") != "" for row in rows):
        return {"available": False}
    ikkt_ranks = _rank_map(rows, "rank_ikkt", "score_ikkt")
    objective_ranks = _rank_map(rows, "rank_objective", "score_objective")
    if not ikkt_ranks or not objective_ranks:
        return {"available": False}
    top_ikkt = {key for key, _rank in sorted(ikkt_ranks.items(), key=lambda item: item[1])[:5]}
    top_obj = {key for key, _rank in sorted(objective_ranks.items(), key=lambda item: item[1])[:5]}
    union = top_ikkt | top_obj
    jaccard = len(top_ikkt & top_obj) / len(union) if union else None
    return {
        "available": True,
        "spearman": _spearman_from_ranks(ikkt_ranks, objective_ranks),
        "kendall": None,
        "kendall_reason": "not_computed_without_optional_stats_dependency",
        "top5_jaccard": jaccard,
    }


def _add_warning_if(condition, warnings, name):
    if condition and name not in warnings:
        warnings.append(name)


def diagnostics_warnings(final_summary, ikkt_fit=None, ikkt_residual=None, pass_count=1):
    warnings = []
    _add_warning_if(
        final_summary.get("top1_top2_ratio") is not None
        and final_summary.get("top1_top2_ratio") < 1.1,
        warnings,
        "TOP1_TOP2_RATIO_LOW",
    )
    comparison = final_summary.get("ikkt_vs_objective") or {}
    if comparison.get("available"):
        jaccard = comparison.get("top5_jaccard")
        _add_warning_if(jaccard is not None and jaccard < 0.2, warnings, "IKKT_OBJECTIVE_TOP5_OVERLAP_LOW")
        _add_warning_if(jaccard == 0.0, warnings, "IKKT_OBJECTIVE_TOP5_OVERLAP_ZERO")
    if ikkt_fit:
        cost_ratio = ikkt_fit.get("cost_ratio")
        _add_warning_if(cost_ratio is not None and cost_ratio > 0.6, warnings, "LAMBDA_FIT_COST_RATIO_MODERATE")
        _add_warning_if(cost_ratio is not None and cost_ratio > 0.8, warnings, "LAMBDA_FIT_COST_RATIO_HIGH")
    _add_warning_if(int(pass_count) > 1, warnings, "RANKING_RECOMPUTED_AFTER_INSERTION")
    return warnings


def finalize_level_diagnostics(settings, score_rows, selected_data):
    if not diagnostics_enabled(settings):
        return
    state = settings.get("_scoring_diagnostics_state")
    if not isinstance(state, dict):
        return
    try:
        context = dict(state.get("context") or {})
        level = int(context.get("level", 0))
        workdir = Path(context.get("workdir") or settings.get("workdir") or ".")
        root = diagnostic_root(settings, workdir)
        level_dir = ensure_level_diagnostic_dir(settings, workdir, level)
        write_readme_once(root)
        update_manifest(
            root,
            level,
            case_name=context.get("case_name"),
            diagnostic_dir=str(settings.get("scoring_diagnostic_dir", "DIAGNOSTIC")),
        )

        metadata = state.get("metadata") or []
        primary_signal = _numeric_array(state.get("primary_signal", []))
        objective_signal = _numeric_array(state.get("objective_signal", primary_signal))
        ikkt_diagnostics = dict(state.get("ikkt_diagnostics") or {})
        ikkt_diagnostics_enabled = _as_bool(
            settings.get("scoring_diagnostic_ikkt", True),
            default=True,
        )
        scoring_mask = np.asarray(
            [
                not (
                    float(row.get("x_over_c", 0.0)) <= 1.0e-6
                    or float(row.get("x_over_c", 0.0)) >= 1.0 - 1.0e-6
                )
                for row in metadata
            ],
            dtype=bool,
        )
        primary_signal_name = context.get("primary_signal_name") or (
            "ikkt_residual" if ikkt_diagnostics else "objective"
        )
        pass_ids = sorted(
            {
                int(row.get("scoring_pass_id") or 1)
                for row in score_rows
            }
        )
        pass_count = len(pass_ids) or 1
        context["diagnostic_version"] = DIAGNOSTIC_VERSION
        context.setdefault("batch_step", 1)
        context.setdefault("scoring_pass_id", pass_ids[0] if pass_ids else 1)
        write_json_atomic(level_dir / "00_context.json", context)
        write_json_atomic(
            level_dir / "01_objective_signal.json",
            objective_signal_payload(metadata, objective_signal, scoring_mask),
        )
        if ikkt_diagnostics and ikkt_diagnostics_enabled:
            constraints_payload = ikkt_constraint_payload(
                metadata,
                objective_signal,
                ikkt_diagnostics,
                scoring_mask,
            )
            fit_payload = ikkt_fit_payload(ikkt_diagnostics)
            residual_payload = ikkt_residual_payload(
                metadata,
                objective_signal,
                primary_signal,
                ikkt_diagnostics,
                scoring_mask,
            )
        else:
            if ikkt_diagnostics and not ikkt_diagnostics_enabled:
                constraints_payload = {
                    "status": "disabled",
                    "reason": "BSPLINE_SCORING_DIAGNOSTIC_IKKT=NO",
                }
            else:
                constraints_payload = {
                    "status": "not_applicable",
                    "reason": "score_mode_is_virtual_insertion",
                }
            fit_payload = dict(constraints_payload)
            residual_payload = dict(constraints_payload)
        write_json_atomic(level_dir / "02_ikkt_constraints.json", constraints_payload)
        write_json_atomic(level_dir / "03_ikkt_fit.json", fit_payload)
        write_json_atomic(level_dir / "04_ikkt_residual.json", residual_payload)

        if _as_bool(settings.get("scoring_diagnostic_nodal_fields", True), default=True):
            nodal_rows = nodal_field_rows(
                metadata,
                objective_signal,
                primary_signal,
                ikkt_diagnostics if ikkt_diagnostics_enabled else {},
                scoring_mask,
                pass_ids[0] if pass_ids else 1,
            )
            for row in nodal_rows:
                row["level"] = level
            nodal_fields = [
                "level",
                "scoring_pass_id",
                "node_id",
                "x",
                "y",
                "x_over_c",
                "side",
                "mask_scoring",
                "objective_signal",
                "lift_field",
                "area_field",
                "moment_field",
                "lambda_lift_times_field",
                "lambda_area_times_field",
                "lambda_moment_times_field",
                "constraint_combo",
                "ikkt_residual",
                "ikkt_residual_is_alias_of_objective",
            ]
            for row in nodal_rows:
                for key in row:
                    if key not in nodal_fields:
                        nodal_fields.append(key)
            write_csv_atomic(level_dir / "05_nodal_fields.csv", nodal_rows, nodal_fields)

        passes = list(state.get("passes") or [])
        write_json_atomic(
            level_dir / "06_old_basis.json",
            {
                "diagnostic_version": DIAGNOSTIC_VERSION,
                "level": level,
                "passes": [
                    {
                        "batch_step": item.get("batch_step"),
                        "scoring_pass_id": item.get("scoring_pass_id"),
                        "side": item.get("side"),
                        **dict(item.get("old_basis", {})),
                    }
                    for item in passes
                ],
            },
        )

        _finalize_rank_columns(score_rows, selected_data)
        extended_fields = _extended_score_fieldnames(score_rows)
        write_csv_atomic(
            level_dir / "07_knot_span_scores_extended.csv",
            score_rows,
            extended_fields,
        )

        primary_norm = float(np.linalg.norm(primary_signal[scoring_mask])) if primary_signal.size else 0.0
        pass_summaries = []
        for pass_id in pass_ids:
            pass_rows = [
                row for row in score_rows
                if int(row.get("scoring_pass_id") or 1) == int(pass_id)
            ]
            pass_summary = ranking_summary_for_rows(
                pass_rows,
                primary_signal_name,
                primary_norm,
                warnings=[],
            )
            matching_pass = next(
                (
                    item for item in passes
                    if int(item.get("scoring_pass_id") or -1) == int(pass_id)
                ),
                {},
            )
            pass_summary.update(
                {
                    "level": level,
                    "batch_step": matching_pass.get("batch_step", pass_rows[0].get("batch_step") if pass_rows else ""),
                    "scoring_pass_id": pass_id,
                    "side": matching_pass.get("side", pass_rows[0].get("side") if pass_rows else ""),
                }
            )
            pass_summaries.append(pass_summary)
        final_rows = _final_pass_rows(score_rows, selected_data)
        final_summary = ranking_summary_for_rows(
            final_rows or score_rows,
            primary_signal_name,
            primary_norm,
            warnings=[],
        )
        warnings = diagnostics_warnings(
            final_summary,
            ikkt_fit=fit_payload if ikkt_diagnostics else None,
            ikkt_residual=residual_payload if ikkt_diagnostics else None,
            pass_count=pass_count,
        )
        final_summary["warnings"] = warnings
        ranking_payload = {
            "diagnostic_version": DIAGNOSTIC_VERSION,
            "level": level,
            "score_mode": context.get("score_mode"),
            "primary_signal_name": primary_signal_name,
            "primary_signal_norm": primary_norm,
            "passes": pass_summaries,
            "final": final_summary,
            "warnings": warnings,
        }
        write_json_atomic(level_dir / "08_ranking_summary.json", ranking_payload)

        selected_rows = _selected_candidate_rows(score_rows, selected_data)
        selected_fields = _selected_candidate_fieldnames(selected_rows)
        write_csv_atomic(level_dir / "09_selected_candidates.csv", selected_rows, selected_fields)
        append_or_rewrite_summary_levels(
            root / "summary_levels.csv",
            _summary_level_row(
                level,
                context,
                final_summary,
                fit_payload if ikkt_diagnostics else {},
                residual_payload if ikkt_diagnostics else {},
                selected_rows,
                warnings,
            ),
        )
    except Exception as exc:
        warn_diagnostic(f"failed to write diagnostics: {exc}")
        try:
            context = dict(state.get("context") or {})
            level = int(context.get("level", 0))
            workdir = Path(context.get("workdir") or settings.get("workdir") or ".")
            level_dir = ensure_level_diagnostic_dir(settings, workdir, level)
            write_json_atomic(level_dir / "diagnostic_error.json", warning_payload("failed to write diagnostics", exc))
        except Exception:
            pass


def _finalize_rank_columns(rows, selected_data):
    selected_order = {}
    for index, item in enumerate(selected_data.get("selected_insertions", []) or [], start=1):
        cid = item.get("candidate_id")
        if cid:
            selected_order[cid] = index
    by_pass = {}
    for row in rows:
        by_pass.setdefault(int(row.get("scoring_pass_id") or 1), []).append(row)
    for pass_rows in by_pass.values():
        raw_sorted = sorted(
            pass_rows,
            key=lambda row: (
                -float(row.get("score", 0.0)),
                float(row.get("span_left", 0.0)),
                float(row.get("span_right", 0.0)),
            ),
        )
        effective_sorted = sorted(
            pass_rows,
            key=lambda row: (
                -float(row.get("selection_score", row.get("score_effective", row.get("score", 0.0)))),
                float(row.get("span_left", 0.0)),
                float(row.get("span_right", 0.0)),
            ),
        )
        raw_ranks = {id(row): index for index, row in enumerate(raw_sorted, start=1)}
        effective_ranks = {id(row): index for index, row in enumerate(effective_sorted, start=1)}
        for row in pass_rows:
            raw_rank = raw_ranks.get(id(row), row.get("rank", ""))
            effective_rank = effective_ranks.get(id(row), raw_rank)
            row["raw_rank"] = raw_rank
            row["effective_rank"] = effective_rank
            row["selected_rank"] = selected_order.get(row.get("candidate_id"), "")
            row["rank_shift_due_to_penalty"] = (
                int(effective_rank) - int(raw_rank)
                if raw_rank != "" and effective_rank != ""
                else ""
            )


def _extended_score_fieldnames(rows):
    preferred = [
        "level",
        "batch_step",
        "scoring_pass_id",
        "candidate_id",
        "raw_rank",
        "effective_rank",
        "selected_rank",
        "rank_shift_due_to_penalty",
        "rank",
        "span_left",
        "span_right",
        "span_width",
        "inserted_knot",
        "side",
        "score_mode",
        "primary_signal_name",
        "score",
        "score_raw",
        "score_effective",
        "selection_score",
        "score_normalized",
        "score_objective",
        "score_ikkt",
        "score_objective_normalized",
        "score_ikkt_normalized",
        "ikkt_objective_score_ratio",
        "candidate_projection_cosine_ikkt_objective",
        "rank_objective",
        "rank_ikkt",
        "objective_projection",
        "lagrangian_projection",
        "residual_energy",
        "span_node_count",
        "span_node_count_upper",
        "span_node_count_lower",
        "incremental_support_node_count",
        "incremental_support_node_count_upper",
        "incremental_support_node_count_lower",
        "pre_svd_columns",
        "rejected_columns",
        "residual_column_norm_min",
        "residual_column_norm_max",
        "residual_column_norms",
        "svd_tol",
        "svd_singular_values_all",
        "svd_singular_values_kept",
        "rank_gap",
        "raw_incremental_condition",
        "gram_condition_after_svd",
        "u_orthogonality_error",
        "old_space_orthogonality",
        "score_signal_norm",
        "projection_norm",
        "incremental_rank",
        "incremental_columns",
        "condition_number",
        "span_key",
        "parent_depth",
        "child_depth",
        "depth_penalty",
        "depth_penalty_mode",
        "batch_depth",
        "batch_penalty",
        "batch_penalty_mode",
        "selected",
        "status",
    ]
    fields = list(preferred)
    for row in rows:
        for key in row:
            if key not in fields and not str(key).startswith("_"):
                fields.append(key)
    return fields


def _selected_candidate_rows(score_rows, selected_data):
    by_id = {row.get("candidate_id"): row for row in score_rows if row.get("candidate_id")}
    rows = []
    for index, item in enumerate(selected_data.get("selected_insertions", []) or [], start=1):
        row = dict(by_id.get(item.get("candidate_id"), {}))
        row.update(item)
        row["selected_rank"] = index
        row["selected"] = True
        row["score_raw"] = row.get("score_raw", item.get("score_raw"))
        row["score_effective"] = row.get("score_effective", item.get("score_effective"))
        row["selection_score"] = row.get("selection_score", item.get("selection_score"))
        rows.append(row)
    if not rows and selected_data.get("selected"):
        row = dict(next((item for item in score_rows if item.get("selected")), {}))
        if row:
            row["selected_rank"] = 1
            rows.append(row)
    return rows


def _selected_candidate_fieldnames(rows):
    preferred = [
        "level",
        "batch_step",
        "scoring_pass_id",
        "candidate_id",
        "selected_rank",
        "side",
        "span_left",
        "span_right",
        "inserted_knot",
        "score_raw",
        "score",
        "score_effective",
        "selection_score",
        "parent_depth",
        "child_depth",
        "depth_penalty",
        "batch_penalty",
        "raw_rank",
        "effective_rank",
        "rank_shift_due_to_penalty",
        "score_objective",
        "score_ikkt",
        "rank_objective",
        "rank_ikkt",
        "incremental_rank",
        "rank_gap",
        "raw_incremental_condition",
        "condition_number",
        "selected",
    ]
    fields = list(preferred)
    for row in rows:
        for key in row:
            if key not in fields and not str(key).startswith("_"):
                fields.append(key)
    return fields


def _final_pass_rows(score_rows, selected_data):
    selected = selected_data.get("selected_insertions", []) or []
    if selected:
        final_pass_id = selected[-1].get("scoring_pass_id")
    else:
        final_pass_id = max((int(row.get("scoring_pass_id") or 1) for row in score_rows), default=1)
    return [
        row for row in score_rows
        if int(row.get("scoring_pass_id") or 1) == int(final_pass_id or 1)
    ]


def _summary_level_row(level, context, final_summary, fit_payload, residual_payload, selected_rows, warnings):
    top1 = (final_summary.get("top5") or [{}])[0]
    comparison = final_summary.get("ikkt_vs_objective") or {}
    return {
        "level": int(level),
        "score_mode": context.get("score_mode"),
        "primary_signal_name": final_summary.get("primary_signal_name"),
        "primary_signal_norm": final_summary.get("primary_signal_norm"),
        "n_candidates": final_summary.get("n_candidates"),
        "top1_side": top1.get("side"),
        "top1_span_left": top1.get("span_left"),
        "top1_span_right": top1.get("span_right"),
        "top1_knot": top1.get("inserted_knot"),
        "top1_top2_ratio": final_summary.get("top1_top2_ratio"),
        "top1_score_fraction": final_summary.get("top1_score_fraction"),
        "top3_score_fraction": final_summary.get("top3_score_fraction"),
        "upper_score_fraction": final_summary.get("upper_score_fraction"),
        "lower_score_fraction": final_summary.get("lower_score_fraction"),
        "spearman_ikkt_objective": comparison.get("spearman"),
        "top5_jaccard_ikkt_objective": comparison.get("top5_jaccard"),
        "cost_ratio": fit_payload.get("cost_ratio"),
        "residual_to_objective_norm": residual_payload.get("residual_to_objective_norm"),
        "grad_constraint_condition": fit_payload.get("grad_constraint_condition"),
        "n_selected": len(selected_rows),
        "warnings": ",".join(warnings or []),
    }
