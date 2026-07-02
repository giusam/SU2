#!/usr/bin/env python3
"""Offline probe for IKKT B-spline virtual insertion scores."""

import argparse
import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from SU2.opt.bspline_adaptive.ikkt import (
    aero_adjoint_field_candidates,
    build_ikkt_score_signal,
    write_ikkt_diagnostics,
)
from SU2.opt.bspline_adaptive.knot_space import extract_clamped_knot_space
from SU2.opt.bspline_adaptive.scoring import load_adjoint_signal, score_knot_spans
from SU2.opt.bspline_modes import load_mode_spec
from SU2.opt.bspline_su2_adaptive import parse_adaptive_options


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Probe IKKT virtual insertion B-spline scores from saved sensitivities."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--modes", default=None)
    parser.add_argument("--eval-dir", default=None)
    parser.add_argument("--surface-sens", default=None)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--diagnostics", default=None)
    parser.add_argument("--constraint-values-json", default=None)
    parser.add_argument("--surface-mode", default=None)
    parser.add_argument("--symmetry-coupling", default=None)
    parser.add_argument("--deformation-direction", default=None)
    parser.add_argument("--knot-min-span-width", type=float, default=None)
    parser.add_argument("--ikkt-scaling-mode", default=None, choices=("PHYSICAL", "DRIVER"))
    parser.add_argument("--ikkt-sign-convention", default=None, choices=("SLSQP_GE_RAW", "HH_RAW"))
    parser.add_argument("--ikkt-active-tol", type=float, default=None)
    parser.add_argument("--ikkt-geom-thickness-active-tol", type=float, default=None)
    return parser.parse_args(argv)


def _fallback_settings():
    return {
        "knot_score_mode": "IKKT_VIRTUAL_INSERTION",
        "knot_min_span_width": 1.0e-8,
        "surface_mode": "BOTH",
        "symmetry_coupling": "NONE",
        "deformation_direction_mode": "VERTICAL",
        "sensitivity_weighting": "NODAL",
        "ikkt_include_geometry_constraints": True,
        "ikkt_include_aero_constraints": False,
        "ikkt_require_available_fields": True,
        "ikkt_scaling_mode": "PHYSICAL",
        "ikkt_sign_convention": "SLSQP_GE_RAW",
        "ikkt_active_tol": 1.0e-6,
        "ikkt_geom_thickness_active_tol": 1.0e-4,
    }


def _settings(args):
    settings = (
        parse_adaptive_options(["--case-config", args.config])
    if args.config
        else _fallback_settings()
    )
    settings = dict(settings)
    if args.eval_dir is not None:
        settings["_ikkt_eval_dir"] = str(Path(args.eval_dir))
    if args.constraint_values_json is not None:
        with open(args.constraint_values_json, "r") as fp:
            settings["_ikkt_aero_constraint_values"] = json.load(fp)
    overrides = {
        "surface_mode": args.surface_mode,
        "symmetry_coupling": args.symmetry_coupling,
        "deformation_direction_mode": args.deformation_direction,
        "knot_min_span_width": args.knot_min_span_width,
        "ikkt_scaling_mode": args.ikkt_scaling_mode,
        "ikkt_sign_convention": args.ikkt_sign_convention,
        "ikkt_active_tol": args.ikkt_active_tol,
        "ikkt_geom_thickness_active_tol": args.ikkt_geom_thickness_active_tol,
    }
    for key, value in overrides.items():
        if value is not None:
            settings[key] = value
    return settings


def _resolve_metadata(args):
    if args.metadata:
        return Path(args.metadata)
    if not args.eval_dir:
        raise SystemExit("--metadata is required unless --eval-dir is provided")
    eval_dir = Path(args.eval_dir)
    candidates = [
        eval_dir / "bspline_surface_metadata.csv",
        eval_dir / "deform" / "bspline_surface_metadata.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit(
        "--metadata was not provided and no metadata file was found in {}".format(
            ", ".join(str(path) for path in candidates)
        )
    )


def _resolve_surface_sens(args, settings):
    if args.surface_sens:
        return Path(args.surface_sens)
    if not args.eval_dir:
        raise SystemExit("--surface-sens is required unless --eval-dir is provided")
    eval_dir = Path(args.eval_dir)
    source = str(settings.get("sensitivity_source", "DOT_AD_TRANSFER")).strip().upper()
    root_candidates = (
        [eval_dir / "surface_sens.csv", eval_dir / "surface_adjoint.csv"]
        if source == "DOT_AD_TRANSFER"
        else [eval_dir / "surface_adjoint.csv", eval_dir / "surface_sens.csv"]
    )
    objective = settings.get("objective_adjoint") or settings.get("opt_objective") or "drag"
    candidates = root_candidates + aero_adjoint_field_candidates(
        objective,
        eval_dir,
        settings,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit(
        "--surface-sens was not provided and no objective sensitivity file was found in {}".format(
            ", ".join(str(path) for path in candidates)
        )
    )


def main(argv=None):
    args = _parse_args(argv)
    settings = _settings(args)
    if str(settings.get("knot_score_mode", "")).upper() != "IKKT_VIRTUAL_INSERTION":
        raise SystemExit(
            "--config must set BSPLINE_KNOT_SCORE_MODE=IKKT_VIRTUAL_INSERTION "
            "for the IKKT probe"
        )

    modes_filename = args.modes or settings.get("modes")
    if not modes_filename:
        raise SystemExit("--modes is required unless --config supplies BSPLINE_MODES")
    mode_spec = load_mode_spec(modes_filename)
    metadata_file = _resolve_metadata(args)
    surface_sens_file = _resolve_surface_sens(args, settings)
    metadata, objective_signal = load_adjoint_signal(surface_sens_file, metadata_file)
    ikkt_signal, diagnostics = build_ikkt_score_signal(
        mode_spec,
        metadata,
        objective_signal,
        settings,
    )

    score_settings = dict(settings)
    score_settings["_ikkt_objective_signal"] = objective_signal
    objective_settings = dict(settings)
    objective_settings["knot_score_mode"] = "VIRTUAL_INSERTION"
    space = extract_clamped_knot_space(mode_spec, objective_settings)
    rows = score_knot_spans(space, metadata, ikkt_signal, score_settings)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_file = (
        Path(args.diagnostics)
        if args.diagnostics
        else output.with_suffix(".diagnostics.json")
    )
    write_ikkt_diagnostics(diagnostics_file, diagnostics)

    lambda_cols = [
        f"lambda_{index}_{item['name']}"
        for index, item in enumerate(diagnostics.get("included_constraints", []))
    ]
    fieldnames = [
        "candidate_knot",
        "span_left",
        "span_right",
        "side",
        "score_objective",
        "score_ikkt",
        "rank_objective",
        "rank_ikkt",
        "objective_projection",
        "lagrangian_projection",
        *lambda_cols,
    ]
    with open(output, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for source in rows:
            row = {
                "candidate_knot": source.get("inserted_knot", ""),
                "span_left": source.get("span_left", ""),
                "span_right": source.get("span_right", ""),
                "side": source.get("side", ""),
                "score_objective": source.get("score_objective", ""),
                "score_ikkt": source.get("score_ikkt", ""),
                "rank_objective": source.get("rank_objective", ""),
                "rank_ikkt": source.get("rank_ikkt", ""),
                "objective_projection": source.get("objective_projection", ""),
                "lagrangian_projection": source.get("lagrangian_projection", ""),
            }
            for column, item in zip(lambda_cols, diagnostics.get("included_constraints", [])):
                row[column] = item.get("lambda", "")
            writer.writerow(row)


if __name__ == "__main__":
    main()
