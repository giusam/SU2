#!/usr/bin/env python3

"""Compare COMPONENT and VIRTUAL_TANGENT on an existing FFD level.

The tool stages the level through symlinks in an isolated work directory and
calls the production candidate scorer.  Original run artifacts are never
modified.  COMPONENT can invoke SU2_DOT; VIRTUAL_TANGENT only reuses saved
surface sensitivities and the production FFD re-embedding code.
"""

import argparse
import copy
import glob
import json
import os
from pathlib import Path
import re
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import SU2

from SU2.opt.progressive_ffd_core import (
    FFDLevel,
    ffd_active_include_bounds_from_opts,
    ffd_active_range_from_opts,
    get_progressive_ffd_options,
)
from SU2.opt.progressive_ffd_mesh import read_ffd_box_columns
from SU2.opt.progressive_ffd_projection import _compute_ffd_dot_candidate_scores
from SU2.opt.progressive_ffd_tangent import COMPONENT, VIRTUAL_TANGENT
from SU2.opt.progressive_hh_core import get_progressive_hh_options


MODES = (COMPONENT, VIRTUAL_TANGENT)
_GENERATED_NAMES = {
    "FFD_SELECTED_CANDIDATE",
}


def _parse_override(value):
    if "=" not in str(value):
        raise argparse.ArgumentTypeError("--set requires KEY=VALUE")
    key, raw = str(value).split("=", 1)
    key = key.strip().upper()
    if not key or not raw.strip():
        raise argparse.ArgumentTypeError("--set requires non-empty KEY=VALUE")
    return key, raw.strip()


def _resolve_level_config(level_dir, requested=None):
    level_dir = os.path.abspath(level_dir)
    if requested:
        path = requested
        if not os.path.isabs(path):
            path = os.path.join(level_dir, path)
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        return path
    candidates = sorted(glob.glob(os.path.join(level_dir, "config_level*.cfg")))
    if len(candidates) != 1:
        raise RuntimeError(
            "Could not identify one level config; use --level-config. "
            f"Candidates: {candidates}"
        )
    return os.path.abspath(candidates[0])


def _infer_level_id(level_dir, level_config):
    for value in (os.path.basename(level_dir), os.path.basename(level_config)):
        match = re.search(r"level[_-]?(\d+)", value, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return 0


def _resolve_level_mesh(level_dir, level_config):
    mesh_name = str(level_config.get("MESH_FILENAME", "")).strip()
    if not mesh_name:
        raise RuntimeError("Level config has no MESH_FILENAME")
    if os.path.isabs(mesh_name):
        path = mesh_name
    else:
        path = os.path.join(level_dir, mesh_name)
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def _clean_control_point_params(kind, params):
    kind = str(kind).upper()
    params = list(params)
    if kind != "FFD_CONTROL_POINT_2D":
        raise RuntimeError(
            "Score comparison supports only FFD_CONTROL_POINT_2D, got "
            f"{kind!r}"
        )
    if len(params) == 5:
        params = params[1:]
    if len(params) != 4:
        raise RuntimeError(
            f"Unexpected FFD_CONTROL_POINT_2D parameters: {params}"
        )
    return params


def infer_active_columns(level_config, level_mesh, opts):
    """Recover physical active stations from DEFINITION_DV and FFD boxes."""

    definition = level_config.get("DEFINITION_DV")
    if not isinstance(definition, dict):
        raise RuntimeError("Level config has no parsed DEFINITION_DV")
    required = ("KIND", "FFDTAG", "PARAM")
    if any(key not in definition for key in required):
        raise RuntimeError("DEFINITION_DV is incomplete")

    upper_tag = str(opts.get("ffd_upper_box_tag", "UPPER_BOX"))
    lower_tag = str(opts.get("ffd_lower_box_tag", "LOWER_BOX"))
    tag_side = {upper_tag: "UPPER", lower_tag: "LOWER"}
    columns_by_tag = {}
    active = {}
    for kind, tag, raw_params in zip(
        definition["KIND"],
        definition["FFDTAG"],
        definition["PARAM"],
    ):
        tag = str(tag)
        if tag not in tag_side:
            raise RuntimeError(
                f"FFD tag {tag!r} is neither {upper_tag!r} nor {lower_tag!r}"
            )
        params = _clean_control_point_params(kind, raw_params)
        control_i = int(round(float(params[0])))
        if tag not in columns_by_tag:
            columns_by_tag[tag] = read_ffd_box_columns(level_mesh, tag)
        columns = columns_by_tag[tag]
        if control_i < 0 or control_i >= len(columns):
            raise RuntimeError(
                f"Control index {control_i} is outside FFD tag {tag!r}"
            )
        active.setdefault(tag_side[tag], []).append(float(columns[control_i]))

    normalized = {
        side: sorted(set(values))
        for side, values in active.items()
        if values
    }
    if not normalized or not set(normalized).issubset({"UPPER", "LOWER"}):
        raise RuntimeError(f"Could not infer active FFD sides: {normalized}")
    return normalized


def _make_level(level_dir, level_config_path, level_config, opts, workdir):
    level_mesh = _resolve_level_mesh(level_dir, level_config)
    active = infer_active_columns(level_config, level_mesh, opts)
    level_id = _infer_level_id(level_dir, level_config_path)
    active_xmin, active_xmax = ffd_active_range_from_opts(opts)
    common = {
        "level_id": level_id,
        "workdir": os.path.abspath(workdir),
        "config_filename": os.path.basename(level_config_path),
        "project_filename": f"project_level{level_id}.pkl",
        "mesh_source": level_mesh,
        "initial_mesh_source": level_mesh,
        "ffd_dv_kind": opts["ffd_dv_kind"],
        "marker": opts["ffd_marker"],
        "domain_mode": opts["ffd_domain_mode"],
        "active_xmin": active_xmin,
        "active_xmax": active_xmax,
        "active_include_bounds": ffd_active_include_bounds_from_opts(opts),
    }
    if set(active) == {"UPPER", "LOWER"}:
        return FFDLevel(
            columns=active["UPPER"],
            upper_columns=active["UPPER"],
            lower_columns=active["LOWER"],
            dual_box=True,
            upper_box_tag=opts["ffd_upper_box_tag"],
            lower_box_tag=opts["ffd_lower_box_tag"],
            direction="OUTWARD",
            **common,
        )
    side = next(iter(active))
    return FFDLevel(
        columns=active[side],
        dual_box=False,
        ffd_box_tag=(
            opts["ffd_upper_box_tag"]
            if side == "UPPER"
            else opts["ffd_lower_box_tag"]
        ),
        control_row=1 if side == "UPPER" else 0,
        direction="OUTWARD",
        side=side,
        **common,
    )


def _stage_level(level_dir, destination):
    os.makedirs(destination, exist_ok=False)
    for source in Path(level_dir).iterdir():
        name = source.name
        if name in _GENERATED_NAMES:
            continue
        if re.fullmatch(r"ffd_candidate_scores_level\d+\.csv", name):
            continue
        if name.startswith("ffd_projection_level"):
            continue
        os.symlink(str(source.resolve()), os.path.join(destination, name))


def _candidate_summary(candidate):
    keys = (
        "insertion_step",
        "rank",
        "side",
        "x",
        "indicator",
        "score_net",
        "score_net_normalized",
        "score_pure",
        "energy_current",
        "energy_candidate",
        "rank_current",
        "rank_candidate",
        "rank_gain",
        "pure_rank",
        "nesting_rms",
        "nesting_max",
        "locality",
        "innovation_center_x",
        "signal_source",
        "admissible",
        "rejected_reason",
    )
    return {key: candidate.get(key) for key in keys if key in candidate}


def summarize_scoring(mode, result, workdir):
    selected = [
        _candidate_summary(candidate)
        for candidate in result.get("selected_candidates", [])
    ]
    candidates = [
        _candidate_summary(candidate)
        for candidate in result.get("raw_candidates", [])
    ]
    dot_only = sorted(
        os.path.basename(path)
        for path in glob.glob(os.path.join(workdir, "DOT_ONLY_*"))
    )
    if mode == VIRTUAL_TANGENT and dot_only:
        raise RuntimeError(
            "VIRTUAL_TANGENT unexpectedly generated DOT_ONLY artifacts: "
            f"{dot_only}"
        )
    return {
        "mode": mode,
        "scoring_basis": result.get("scoring_basis"),
        "ffd_blending": result.get("ffd_blending"),
        "insertion_target": result.get("insertion_target"),
        "insertions_completed": result.get("insertions_completed"),
        "selected": selected,
        "candidates": candidates,
        "dot_only_artifacts": dot_only,
    }


def _run_one_mode(
    mode,
    base_config,
    level_dir,
    level_config_path,
    accepted_mesh,
    workdir,
):
    config = copy.deepcopy(base_config)
    config["PROGRESSIVE_FFD_SCORING_MODE"] = mode
    hh_opts = get_progressive_hh_options(config)
    opts = get_progressive_ffd_options(config, hh_opts)
    level_config = SU2.io.Config(level_config_path)
    level = _make_level(
        level_dir,
        level_config_path,
        level_config,
        opts,
        workdir,
    )
    result = _compute_ffd_dot_candidate_scores(
        level,
        opts,
        mesh_source=accepted_mesh,
    )
    return summarize_scoring(mode, result, workdir)


def _print_summary(summary):
    for mode, result in summary["modes"].items():
        print(f"\n[{mode}] {result.get('scoring_basis')}")
        for selected in result.get("selected", []):
            score = selected.get("score_net", selected.get("indicator"))
            print(
                "  insertion={step} side={side} x={x:.8f} score={score:.8e}".format(
                    step=int(selected.get("insertion_step", 1)),
                    side=selected.get("side", ""),
                    x=float(selected["x"]),
                    score=float(score),
                )
            )
    comparison = summary.get("comparison")
    if comparison:
        print(
            "\n[COMPARISON] same selected sequence = "
            f"{'YES' if comparison['same_selected_sequence'] else 'NO'}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--level-dir", required=True)
    parser.add_argument("--level-config")
    parser.add_argument("--accepted-mesh")
    parser.add_argument(
        "--mode",
        choices=("BOTH",) + MODES,
        default="BOTH",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        type=_parse_override,
        metavar="KEY=VALUE",
        help="Override one base-config value before both comparisons.",
    )
    parser.add_argument(
        "--unset",
        action="append",
        default=[],
        metavar="KEY",
        help="Remove one base-config key before both comparisons.",
    )
    parser.add_argument(
        "--keep-workdirs",
        help="Keep isolated COMPONENT/VIRTUAL_TANGENT artifacts below this path.",
    )
    parser.add_argument(
        "--output",
        default="ffd_score_comparison.json",
    )
    args = parser.parse_args(argv)

    base_config_path = os.path.abspath(args.base_config)
    level_dir = os.path.abspath(args.level_dir)
    if not os.path.isfile(base_config_path):
        raise FileNotFoundError(base_config_path)
    if not os.path.isdir(level_dir):
        raise NotADirectoryError(level_dir)
    level_config_path = _resolve_level_config(level_dir, args.level_config)
    accepted_mesh = (
        None if args.accepted_mesh is None else os.path.abspath(args.accepted_mesh)
    )
    if accepted_mesh is not None and not os.path.isfile(accepted_mesh):
        raise FileNotFoundError(accepted_mesh)

    base_config = SU2.io.Config(base_config_path)
    unset_keys = [str(key).strip().upper() for key in args.unset]
    for key in unset_keys:
        if key in base_config:
            del base_config[key]
    for key, value in args.set:
        base_config[key] = value
    modes = MODES if args.mode == "BOTH" else (args.mode,)
    summary = {
        "base_config": base_config_path,
        "level_dir": level_dir,
        "level_config": level_config_path,
        "accepted_mesh": accepted_mesh,
        "overrides": {key: value for key, value in args.set},
        "unset_keys": unset_keys,
        "modes": {},
    }

    temporary_roots = []
    try:
        for mode in modes:
            if args.keep_workdirs:
                root = os.path.abspath(args.keep_workdirs)
                os.makedirs(root, exist_ok=True)
                workdir = os.path.join(root, mode.lower())
                _stage_level(level_dir, workdir)
            else:
                holder = tempfile.TemporaryDirectory(
                    prefix=f"ffd_score_{mode.lower()}_"
                )
                temporary_roots.append(holder)
                workdir = os.path.join(holder.name, "LEVEL")
                _stage_level(level_dir, workdir)
            summary["modes"][mode] = _run_one_mode(
                mode,
                base_config,
                level_dir,
                level_config_path,
                accepted_mesh,
                workdir,
            )

        if set(summary["modes"]) == set(MODES):
            component_sequence = [
                (item.get("side"), item.get("x"))
                for item in summary["modes"][COMPONENT]["selected"]
            ]
            virtual_sequence = [
                (item.get("side"), item.get("x"))
                for item in summary["modes"][VIRTUAL_TANGENT]["selected"]
            ]
            summary["comparison"] = {
                "component_selected_sequence": component_sequence,
                "virtual_selected_sequence": virtual_sequence,
                "same_selected_sequence": component_sequence == virtual_sequence,
            }

        output = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, "w") as stream:
            json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
        _print_summary(summary)
        print(f"\nWrote {output}")
        return 0
    finally:
        for holder in temporary_roots:
            holder.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
