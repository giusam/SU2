#!/usr/bin/env python3

"""Compare classic and virtual-tangent HH selections and level histories."""

import argparse
import csv
import json
import os


def _clean_key(value):
    return str(value).strip().strip('"').upper()


def _read_history(path, objective="DRAG"):
    objective = str(objective).upper()
    with open(path, newline="") as stream:
        reader = csv.reader(stream)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"Empty history: {path}")
    header = [_clean_key(value) for value in rows[0]]
    if objective not in header:
        raise RuntimeError(f"{objective} absent from {path}: {header}")
    objective_index = header.index(objective)
    evaluation_index = header.index("EVALUATION") if "EVALUATION" in header else None
    records = []
    for row_number, row in enumerate(rows[1:], start=1):
        if len(row) <= objective_index or not str(row[objective_index]).strip():
            continue
        evaluation = (
            int(float(row[evaluation_index]))
            if evaluation_index is not None
            else row_number
        )
        records.append(
            {
                "evaluation": evaluation,
                "objective": float(row[objective_index]),
            }
        )
    if not records:
        raise RuntimeError(f"No objective records in {path}")
    values = [record["objective"] for record in records]
    initial = values[0]
    final = values[-1]
    best = min(values)
    return {
        "path": os.path.abspath(path),
        "evaluations": len(records),
        "first_evaluation": records[0]["evaluation"],
        "last_evaluation": records[-1]["evaluation"],
        "initial_objective": initial,
        "final_objective": final,
        "best_objective": best,
        "absolute_improvement": initial - best,
        "relative_improvement": (initial - best) / abs(initial) if initial else 0.0,
        "records": records,
    }


def _read_classic_selection(path, level_id=0):
    with open(path, newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = []
    for row in rows:
        if int(row.get("level_id", -1)) != int(level_id):
            continue
        selected.append(
            {
                "side": str(row.get("side", "")).upper(),
                "x": float(row["x"]),
                "indicator": float(row["indicator"]),
                "indicator_ratio_to_best": float(row["indicator_ratio_to_best"]),
                "scoring_mode": "COMPONENT",
            }
        )
    if not selected:
        raise RuntimeError(f"No classic selection for level {level_id} in {path}")
    return selected


def _read_virtual_selection(path):
    with open(path) as stream:
        payload = json.load(stream)
    summary = payload.get("modes", {}).get("VIRTUAL_TANGENT", payload)
    selected = list(summary.get("selected", summary.get("selected_candidates", [])))
    if not selected:
        raise RuntimeError(f"No virtual-tangent selection in {path}")
    return selected


def _selection_sequence(selected):
    return [
        {"side": str(item["side"]).upper(), "x": float(item["x"])}
        for item in selected
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classic-history", required=True)
    parser.add_argument("--virtual-history", required=True)
    parser.add_argument("--classic-selection-history", required=True)
    parser.add_argument("--virtual-score-json", required=True)
    parser.add_argument("--objective", default="DRAG")
    parser.add_argument("--selection-level", type=int, default=0)
    parser.add_argument("--output", default="hh_level_score_comparison.json")
    parser.add_argument("--plot", help="Optional convergence plot output path.")
    args = parser.parse_args(argv)

    classic_selected = _read_classic_selection(
        args.classic_selection_history,
        args.selection_level,
    )
    virtual_selected = _read_virtual_selection(args.virtual_score_json)
    classic = _read_history(args.classic_history, args.objective)
    virtual = _read_history(args.virtual_history, args.objective)
    payload = {
        "objective": str(args.objective).upper(),
        "selection_level": int(args.selection_level),
        "classic": {
            "scoring_mode": "COMPONENT",
            "selected": classic_selected,
            "history": classic,
        },
        "virtual_tangent": {
            "scoring_mode": "VIRTUAL_TANGENT",
            "selected": virtual_selected,
            "history": virtual,
        },
        "comparison": {
            "same_selection": (
                _selection_sequence(classic_selected)
                == _selection_sequence(virtual_selected)
            ),
            "evaluation_delta_virtual_minus_classic": (
                virtual["evaluations"] - classic["evaluations"]
            ),
            "best_objective_delta_virtual_minus_classic": (
                virtual["best_objective"] - classic["best_objective"]
            ),
            "final_objective_delta_virtual_minus_classic": (
                virtual["final_objective"] - classic["final_objective"]
            ),
        },
    }
    output = os.path.abspath(args.output)
    with open(output, "w") as stream:
        json.dump(payload, stream, allow_nan=False, indent=2, sort_keys=True)

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(figsize=(7.0, 4.2))
        axes.plot(
            [item["evaluation"] for item in classic["records"]],
            [item["objective"] for item in classic["records"]],
            marker="o",
            label="COMPONENT",
        )
        axes.plot(
            [item["evaluation"] for item in virtual["records"]],
            [item["objective"] for item in virtual["records"]],
            marker="s",
            label="VIRTUAL_TANGENT",
        )
        axes.set_xlabel("Evaluation")
        axes.set_ylabel(str(args.objective).upper())
        axes.grid(True, alpha=0.3)
        axes.legend()
        figure.tight_layout()
        figure.savefig(os.path.abspath(args.plot), dpi=180)
        plt.close(figure)

    print("CLASSIC selected:", _selection_sequence(classic_selected))
    print("VIRTUAL selected:", _selection_sequence(virtual_selected))
    print(
        "CLASSIC level: "
        f"evals={classic['evaluations']} best={classic['best_objective']:.12g}"
    )
    print(
        "VIRTUAL level: "
        f"evals={virtual['evaluations']} best={virtual['best_objective']:.12g}"
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
