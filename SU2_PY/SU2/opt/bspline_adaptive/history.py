"""History, selection, restart, and trigger CSV helpers."""

import csv
import json
import math
from pathlib import Path


from SU2.opt.bspline_driver.reduction import (
    active_coefficient_vector,
    active_mode_ids,
)

from .errors import BSplineAdaptiveError
from .models import TriggerDecision
from .settings import KNOT_SCORE_FIELDNAMES

SAFE_OPTIMIZATION_STATUSES = {
    "ok",
    "ok_clipped_benign",
    "benign_clipped_legacy",
    "accepted_clipped_restart",
}

def _csv_value(value):
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return "{:.15g}".format(value)
    return value

def write_knot_span_scores_csv(rows, filename):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=KNOT_SCORE_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: _csv_value(row.get(field, "")) for field in KNOT_SCORE_FIELDNAMES}
            )

def write_selected_knot_refinement_json(level_id, selected_data, score_rows, filename):
    data = {
        "level_id": int(level_id),
        "selected": bool(selected_data.get("selected", False)),
        "refinement": selected_data,
        "selected_span_score": next(
            (row for row in score_rows if row.get("selected")),
            None,
        ),
    }
    with open(filename, "w") as fp:
        json.dump(data, fp, indent=2, sort_keys=True)
        fp.write("\n")
    return data

def find_best_eval_dir(opt_run_dir, objective_column="objective"):
    opt_run_dir = Path(opt_run_dir)
    history_file = opt_run_dir / "optimization_history.csv"
    valid_rows = []
    with open(history_file, "r", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            try:
                objective = float(row.get(objective_column, ""))
                eval_id = int(float(row.get("eval_id", "")))
            except Exception:
                continue
            status = str(row.get("status", "ok")).strip().lower()
            # NaN/inf objectives must never win a minimization; replace
            # them with +inf so they sort to the bottom and are skipped.
            if not math.isfinite(objective):
                objective = math.inf
            valid_rows.append((objective, eval_id, status))

    if not valid_rows:
        raise BSplineAdaptiveError(f"{history_file} has no valid evaluation rows")

    finite_rows = [row for row in valid_rows if math.isfinite(row[0])]
    if not finite_rows:
        raise BSplineAdaptiveError(
            f"{history_file} has no evaluation with a finite objective"
        )

    ok_rows = [
        row
        for row in valid_rows
        if row[2] in SAFE_OPTIMIZATION_STATUSES and math.isfinite(row[0])
    ]
    if not ok_rows:
        # Do not fall back to rejected/failed rows: their adjoint is not a
        # valid adaptive-scoring source.
        raise BSplineAdaptiveError(
            f"{history_file} has no safe evaluation with a finite objective"
        )
    best = min(ok_rows, key=lambda item: item[0])
    return opt_run_dir / f"eval_{best[1]:04d}"

def find_eval_dir_for_mode_coefficients(
    opt_run_dir,
    optimized_modes,
    tolerance=1.0e-10,
):
    opt_run_dir = Path(opt_run_dir)
    history_file = opt_run_dir / "optimization_history.csv"
    mode_ids = active_mode_ids(optimized_modes)
    coefficients = active_coefficient_vector(optimized_modes)
    matched_rows = []

    with open(history_file, "r", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            try:
                eval_id = int(float(row.get("eval_id", "")))
            except Exception:
                continue

            status = str(row.get("status", "ok")).strip().lower()
            if status and status not in SAFE_OPTIMIZATION_STATUSES:
                continue

            matched = True
            for mode_id, coefficient in zip(mode_ids, coefficients):
                field = f"coeff__{mode_id}"
                if field not in row:
                    matched = False
                    break
                try:
                    history_value = float(row[field])
                except Exception:
                    matched = False
                    break
                if abs(history_value - coefficient) > float(tolerance):
                    matched = False
                    break

            if matched:
                try:
                    objective = float(row.get("objective", "inf"))
                except Exception:
                    objective = math.inf
                # NaN/inf objectives must never win a minimization: skip
                # them entirely here too, so the matched-eval selection
                # cannot pick a broken run.
                if math.isfinite(objective):
                    matched_rows.append((objective, eval_id))

    if matched_rows:
        _objective, matched_eval_id = min(matched_rows, key=lambda item: item[0])
        return opt_run_dir / f"eval_{matched_eval_id:04d}"

    print(
        "[PROGRESSIVE_BSPLINE] WARNING: no eval row with finite objective matches optimized_modes.json coefficients; falling back to best objective eval"
    )
    return find_best_eval_dir(opt_run_dir)

def _read_optimization_history(opt_run_dir):
    history_file = Path(opt_run_dir) / "optimization_history.csv"
    rows = []
    with open(history_file, "r", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            try:
                row["_objective"] = float(row.get("objective", ""))
                row["_eval_id"] = int(float(row.get("eval_id", "")))
            except Exception:
                continue
            rows.append(row)
    return rows

def _cumulative_best(values):
    best_values = []
    best = None
    for value in values:
        value = float(value)
        best = value if best is None else min(best, value)
        best_values.append(float(best))
    return best_values

def _compact_number(value):
    if value == "":
        return ""
    try:
        return "{:.6e}".format(float(value))
    except Exception:
        return str(value)

def _log_trigger_decision(decision):
    print(
        "[PROGRESSIVE_BSPLINE] Trigger {} | window={} metric={} threshold={} "
        "patience={}/{} refine_now={} reason={}".format(
            decision.trigger_mode,
            decision.window,
            _compact_number(decision.metric),
            _compact_number(decision.threshold),
            int(decision.counter),
            decision.patience,
            bool(decision.refine_now),
            decision.reason,
        )
    )

def _tail_count(values, predicate):
    count = 0
    for value in reversed(values):
        if predicate(value):
            count += 1
        else:
            break
    return count

def _trigger_refine_from_history(objectives, opts):
    trigger = str(opts.get("trigger", "MAX_ITER")).upper()
    if trigger == "SLOPE_EFFICIENCY_FILTERED":
        print(
            "[PROGRESSIVE_BSPLINE] SLOPE_EFFICIENCY_FILTERED is deprecated and is mapped internally to SLOPE_EFFICIENCY_TRIGGER."
        )
        trigger = "SLOPE_EFFICIENCY_TRIGGER"

    window = max(1, int(opts.get("window", 1)))
    threshold = float(opts.get("tol", 0.2))
    patience = max(1, int(opts.get("slope_patience", 1)))
    decision = TriggerDecision(
        trigger_mode=trigger,
        threshold=threshold,
        window=window,
        patience=patience,
        refine_now=False,
    )
    if trigger == "MAX_ITER":
        decision.metric = ""
        decision.threshold = ""
        decision.window = ""
        decision.patience = ""
        decision.counter = 1
        decision.refine_now = True
        decision.reason = "level_complete"
        _log_trigger_decision(decision)
        return decision
    if len(objectives) < 2:
        decision.reason = "insufficient_history"
        _log_trigger_decision(decision)
        return decision

    warmup = int(opts.get("warmup_iter", 0))
    if len(objectives) <= warmup:
        decision.reason = f"warmup {len(objectives)}/{warmup}"
        _log_trigger_decision(decision)
        return decision

    best_values = _cumulative_best(objectives)

    if trigger == "SLOPE_EFFICIENCY_TRIGGER":
        if len(objectives) < window + 1:
            decision.reason = "insufficient_window"
            _log_trigger_decision(decision)
            return decision
        improvements = [
            max(0.0, float(prev) - float(current))
            for prev, current in zip(best_values[:-1], best_values[1:])
        ]
        windowed = [
            sum(improvements[i - window + 1 : i + 1]) / float(window)
            for i in range(window - 1, len(improvements))
        ]
        best_efficiency = max(windowed) if windowed else 0.0
        denominator = max(best_efficiency, float(opts.get("eps", 1.0e-300)))
        ratios = [float(value) / denominator for value in windowed]
        ratio = ratios[-1] if ratios else 0.0
        decision.metric = ratio
        decision.counter = _tail_count(ratios, lambda value: float(value) < threshold)
        decision.refine_now = decision.counter >= patience
        decision.reason = "efficiency_below_threshold" if decision.refine_now else "efficiency_ok"
        _log_trigger_decision(decision)
        return decision

    if trigger == "SLOPE_EFFICIENCY_BEST_LOG":
        eps = float(opts.get("eps", 1.0e-300))
        if len(objectives) < window + 1:
            decision.reason = "insufficient_window"
            _log_trigger_decision(decision)
            return decision
        improvements = []
        for prev, current in zip(best_values[:-1], best_values[1:]):
            prev_log = math.log(max(float(prev) + eps, eps))
            current_log = math.log(max(float(current) + eps, eps))
            improvements.append(max(0.0, prev_log - current_log))
        if not improvements:
            decision.reason = "no_improvements"
            _log_trigger_decision(decision)
            return decision
        windowed = [
            sum(improvements[i - window + 1 : i + 1]) / float(window)
            for i in range(window - 1, len(improvements))
        ]
        max_slope = max(max(windowed) if windowed else 0.0, eps)
        ratios = [float(value) / max_slope for value in windowed]
        ratio = ratios[-1] if ratios else 0.0
        decision.metric = ratio
        decision.counter = _tail_count(ratios, lambda value: float(value) < threshold)
        decision.refine_now = decision.counter >= patience
        decision.reason = "log_efficiency_below_threshold" if decision.refine_now else "log_efficiency_ok"
        _log_trigger_decision(decision)
        return decision

    if trigger == "STAGNATION_TRIGGER":
        stag_tol = float(opts.get("stag_tol", 1.0e-3))
        stag_window = max(1, int(opts.get("stag_window", window)))
        stag_patience = max(1, int(opts.get("stag_patience", patience)))
        decision.threshold = stag_tol
        decision.window = stag_window
        decision.patience = stag_patience
        if len(best_values) < stag_window + 1:
            decision.reason = "insufficient_window"
            _log_trigger_decision(decision)
            return decision
        improvements = []
        for end_index in range(stag_window, len(best_values)):
            start = float(best_values[end_index - stag_window])
            end = float(best_values[end_index])
            improvements.append(
                max(0.0, start - end)
                / max(abs(start), float(opts.get("eps", 1.0e-300)))
            )
        improvement = improvements[-1] if improvements else 0.0
        decision.metric = improvement
        decision.counter = _tail_count(improvements, lambda value: float(value) < stag_tol)
        decision.refine_now = decision.counter >= stag_patience
        decision.reason = "stagnated" if decision.refine_now else "improving"
        _log_trigger_decision(decision)
        return decision

    raise BSplineAdaptiveError(f"unsupported trigger {trigger!r}")

def _level_summary_row(level, rows, selected_modes, trigger_decision, refine_now, status):
    safe_rows = [
        row
        for row in rows
        if str(row.get("status", "ok")).strip().lower()
        in SAFE_OPTIMIZATION_STATUSES
        and math.isfinite(float(row["_objective"]))
    ]
    objectives = [row["_objective"] for row in safe_rows]
    best_row = min(safe_rows, key=lambda row: row["_objective"]) if safe_rows else None
    if isinstance(trigger_decision, TriggerDecision):
        trigger_mode = trigger_decision.trigger_mode
        trigger_metric = trigger_decision.metric
        trigger_threshold = trigger_decision.threshold
        trigger_window = trigger_decision.window
        trigger_patience = trigger_decision.patience
        trigger_counter = trigger_decision.counter
        trigger_reason = trigger_decision.reason
    else:
        trigger_mode = str(trigger_decision)
        trigger_metric = ""
        trigger_threshold = ""
        trigger_window = ""
        trigger_patience = ""
        trigger_counter = ""
        trigger_reason = ""
    return {
        "level_id": level.level_id,
        "ndv": level.ndv,
        "workdir": str(level.workdir),
        "opt_workdir": str(level.opt_workdir),
        "objective_start": objectives[0] if objectives else "",
        "objective_final": objectives[-1] if objectives else "",
        "best_objective": best_row["_objective"] if best_row else "",
        "best_eval_id": best_row["_eval_id"] if best_row else "",
        "n_function_evals": len(rows),
        "n_gradient_evals": len(rows),
        "n_added": len(selected_modes),
        "selected_ids": ";".join(str(mode["id"]) for mode in selected_modes),
        "trigger_mode": trigger_mode,
        "trigger_metric": trigger_metric,
        "trigger_threshold": trigger_threshold,
        "trigger_window": trigger_window,
        "trigger_patience": trigger_patience,
        "trigger_counter": trigger_counter,
        "refine_now": refine_now,
        "trigger_reason": trigger_reason,
        "status": status,
    }

def _write_adaptive_history(rows, filename):
    fieldnames = [
        "level_id",
        "ndv",
        "workdir",
        "opt_workdir",
        "objective_start",
        "objective_final",
        "best_objective",
        "best_eval_id",
        "n_function_evals",
        "n_gradient_evals",
        "n_added",
        "selected_ids",
        "trigger_mode",
        "trigger_metric",
        "trigger_threshold",
        "trigger_window",
        "trigger_patience",
        "trigger_counter",
        "refine_now",
        "trigger_reason",
        "status",
    ]
    with open(filename, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fieldnames})

def _write_selection_history(rows, filename):
    fieldnames = ["level_id", "selection_order", "mode_id", "side", "score", "raw_grad"]
    with open(filename, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fieldnames})

def _append_restart_history_rows(
    accumulated_rows,
    accumulated_fieldnames,
    history_filename,
    restart_id,
    restart_reason,
):
    history_filename = Path(history_filename)
    if not history_filename.exists():
        return
    with open(history_filename, "r", newline="") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames:
            return
        for field in list(reader.fieldnames) + ["restart_id", "restart_reason"]:
            if field not in accumulated_fieldnames:
                accumulated_fieldnames.append(field)
        for row in reader:
            row["restart_id"] = int(restart_id)
            row["restart_reason"] = str(restart_reason)
            accumulated_rows.append(row)

def _write_restart_history_rows(history_filename, rows, fieldnames):
    if not rows:
        return
    history_filename = Path(history_filename)
    history_filename.parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with open(history_filename, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
