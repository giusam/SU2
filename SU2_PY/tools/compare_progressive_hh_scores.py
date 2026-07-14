#!/usr/bin/env python3

"""Compare legacy COMPONENT and SVD-energy HH scores on an existing level."""

import argparse
import copy
import glob
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import SU2

from SU2.opt.progressive_hh_core import (
    HHLevel,
    apply_post_opt_coefficient_spring,
    get_progressive_hh_options,
    is_symmetric_reduced,
    select_candidates_by_nadd_mode,
)
from SU2.opt.progressive_hh_projection import _compute_dot_candidate_scores
from SU2.opt.progressive_hh_tangent import COMPONENT, VIRTUAL_TANGENT


MODES = (COMPONENT, VIRTUAL_TANGENT)


def _parse_override(value):
    if "=" not in str(value):
        raise argparse.ArgumentTypeError("--set requires KEY=VALUE")
    key, raw = str(value).split("=", 1)
    key = key.strip().upper()
    if not key or not raw.strip():
        raise argparse.ArgumentTypeError("--set requires non-empty KEY=VALUE")
    return key, raw.strip()


def _parse_centers(value):
    if value is None:
        return None
    raw = str(value).strip().strip("()[]")
    if not raw or raw.upper() == "NONE":
        return []
    return [float(token.strip()) for token in raw.split(",") if token.strip()]


def _resolve_level_config(level_dir, requested=None):
    if requested:
        path = requested if os.path.isabs(requested) else os.path.join(level_dir, requested)
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


def _infer_level_id(level_dir, config_path):
    for value in (os.path.basename(level_dir), os.path.basename(config_path)):
        match = re.search(r"level[_-]?(\d+)", value, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return 0


def infer_hh_centers(level_config):
    definition = level_config.get("DEFINITION_DV")
    if not isinstance(definition, dict):
        raise RuntimeError("Level config has no parsed DEFINITION_DV")
    kinds = definition.get("KIND", [])
    params = definition.get("PARAM", [])
    if len(kinds) != len(params):
        raise RuntimeError("DEFINITION_DV KIND/PARAM sizes differ")
    upper = []
    lower = []
    for kind, raw in zip(kinds, params):
        if str(kind).strip().upper() != "HICKS_HENNE":
            raise RuntimeError(f"Expected HICKS_HENNE, got {kind!r}")
        values = [float(value) for value in raw]
        if len(values) < 2:
            raise RuntimeError(f"Invalid Hicks--Henne parameters {raw!r}")
        (upper if bool(round(values[-2])) else lower).append(values[-1])
    return sorted(upper), sorted(lower)


def _resolve_level_mesh(level_dir, level_config):
    mesh_name = str(level_config.get("MESH_FILENAME", "")).strip()
    if not mesh_name:
        raise RuntimeError("Level config has no MESH_FILENAME")
    path = mesh_name if os.path.isabs(mesh_name) else os.path.join(level_dir, mesh_name)
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def _stage_level(level_dir, destination):
    os.makedirs(destination, exist_ok=False)
    generated = re.compile(
        r"(?:hh_candidate_scores_level\d+\.csv|hh_virtual_tangent_level\d+\.json)"
    )
    for source in Path(level_dir).iterdir():
        if generated.fullmatch(source.name):
            continue
        os.symlink(str(source.resolve()), os.path.join(destination, source.name))


def _make_level(
    level_dir,
    config_path,
    level_config,
    workdir,
    upper_override=None,
    lower_override=None,
):
    upper, lower = infer_hh_centers(level_config)
    if upper_override is not None:
        upper = sorted(float(value) for value in upper_override)
    if lower_override is not None:
        lower = sorted(float(value) for value in lower_override)
    level_id = _infer_level_id(level_dir, config_path)
    mesh = _resolve_level_mesh(level_dir, level_config)
    return HHLevel(
        level_id=level_id,
        upper=upper,
        lower=lower,
        workdir=os.path.abspath(workdir),
        config_filename=os.path.basename(config_path),
        project_filename=f"project_level{level_id}.pkl",
        mesh_source=mesh,
        initial_mesh_source=mesh,
    )


def _load_project_dv_values(project_path):
    with tempfile.TemporaryDirectory(prefix="hh_project_read_") as tmp:
        local = os.path.join(tmp, "project.pkl")
        shutil.copy2(project_path, local)
        project = SU2.io.load_data(local)
    values = getattr(project, "opt_dv_values", None)
    if values is None:
        values = getattr(project, "last_dv_values", None)
    if values is None:
        raise RuntimeError(f"No optimized HH values found in {project_path}")
    return [float(value) for value in values]


def _selected_from_result(result, level, opts):
    if result.get("sequential_selection", False):
        return list(result.get("selected_candidates", []))
    active = {
        "UPPER": list(level.upper),
        "LOWER": list(level.lower),
    }
    if is_symmetric_reduced(opts):
        active = {"PAIR": list(level.upper)}
    return select_candidates_by_nadd_mode(
        result.get("candidates", []),
        level.ndv,
        opts,
        active_centers_by_side=active,
    )


def _candidate_summary(candidate):
    fields = (
        "insertion_step",
        "rank",
        "side",
        "x",
        "t2",
        "indicator",
        "grad",
        "score_net",
        "score_net_normalized",
        "score_pure",
        "score_pure_normalized",
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
    return {key: candidate.get(key) for key in fields if key in candidate}


def _run_mode(mode, base_config, level_dir, config_path, workdir):
    config = copy.deepcopy(base_config)
    config["PROGRESSIVE_HH_SCORING_MODE"] = mode
    opts = get_progressive_hh_options(config)
    level_config = SU2.io.Config(config_path)
    level = _make_level(
        level_dir,
        config_path,
        level_config,
        workdir,
        upper_override=base_config.get("_HH_SCORE_UPPER_OVERRIDE"),
        lower_override=base_config.get("_HH_SCORE_LOWER_OVERRIDE"),
    )
    post_spring_project = base_config.get("_HH_SCORE_POST_SPRING_PROJECT")
    if post_spring_project:
        result_state = {
            "dv_values": _load_project_dv_values(post_spring_project),
        }
        reallocated = apply_post_opt_coefficient_spring(
            level,
            result_state,
            opts,
        )
        if reallocated is None:
            raise RuntimeError("Could not apply post-opt HH spring for score comparison")
        level.upper, level.lower = reallocated
    result = _compute_dot_candidate_scores(level, opts)
    selected = _selected_from_result(result, level, opts)
    return {
        "mode": mode,
        "scoring_basis": result.get("scoring_basis", "COMPONENT"),
        "signal_source": result.get("signal_source"),
        "selected": [_candidate_summary(candidate) for candidate in selected],
        "candidates": [
            _candidate_summary(candidate)
            for candidate in result.get("raw_candidates", [])
        ],
        "candidate_scores_csv": result.get("candidate_scores_csv"),
        "metadata_json": result.get("metadata_json"),
        "active_upper": list(level.upper),
        "active_lower": list(level.lower),
    }


def _sequence(summary):
    return [
        (str(item.get("side", "")), round(float(item["x"]), 12))
        for item in summary.get("selected", [])
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--level-dir", required=True)
    parser.add_argument("--level-config")
    parser.add_argument("--mode", choices=("BOTH",) + MODES, default="BOTH")
    parser.add_argument("--set", action="append", default=[], type=_parse_override)
    parser.add_argument(
        "--upper-centers",
        type=_parse_centers,
        help="Override active upper centers (comma-separated).",
    )
    parser.add_argument(
        "--lower-centers",
        type=_parse_centers,
        help="Override active lower centers; use NONE for an empty side.",
    )
    parser.add_argument(
        "--post-spring-project",
        help="Apply the configured POST_OPT coefficient spring using this project pickle.",
    )
    parser.add_argument("--keep-workdirs")
    parser.add_argument("--output", default="hh_score_comparison.json")
    args = parser.parse_args(argv)

    base_config_path = os.path.abspath(args.base_config)
    level_dir = os.path.abspath(args.level_dir)
    config_path = _resolve_level_config(level_dir, args.level_config)
    base_config = SU2.io.Config(base_config_path)
    for key, value in args.set:
        base_config[key] = value
    if args.post_spring_project and (
        args.upper_centers is not None or args.lower_centers is not None
    ):
        raise ValueError(
            "--post-spring-project cannot be combined with explicit center overrides"
        )
    if args.upper_centers is not None:
        base_config["_HH_SCORE_UPPER_OVERRIDE"] = args.upper_centers
    if args.lower_centers is not None:
        base_config["_HH_SCORE_LOWER_OVERRIDE"] = args.lower_centers
    if args.post_spring_project:
        project_path = os.path.abspath(args.post_spring_project)
        if not os.path.isfile(project_path):
            raise FileNotFoundError(project_path)
        base_config["_HH_SCORE_POST_SPRING_PROJECT"] = project_path

    modes = MODES if args.mode == "BOTH" else (args.mode,)
    temporary = None
    if args.keep_workdirs:
        root = os.path.abspath(args.keep_workdirs)
        os.makedirs(root, exist_ok=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="hh_score_compare_")
        root = temporary.name

    summaries = {}
    try:
        for mode in modes:
            workdir = os.path.join(root, mode.lower())
            _stage_level(level_dir, workdir)
            summaries[mode] = _run_mode(
                mode,
                base_config,
                level_dir,
                config_path,
                workdir,
            )
        output = {
            "base_config": base_config_path,
            "level_dir": level_dir,
            "level_config": config_path,
            "modes": summaries,
        }
        if len(summaries) == 2:
            output["comparison"] = {
                "component_sequence": _sequence(summaries[COMPONENT]),
                "virtual_tangent_sequence": _sequence(summaries[VIRTUAL_TANGENT]),
                "same_selected_sequence": (
                    _sequence(summaries[COMPONENT])
                    == _sequence(summaries[VIRTUAL_TANGENT])
                ),
            }
        output_path = os.path.abspath(args.output)
        with open(output_path, "w") as stream:
            json.dump(output, stream, allow_nan=False, indent=2, sort_keys=True)
        for mode, summary in summaries.items():
            print(f"\n[{mode}] {summary['scoring_basis']}")
            for candidate in summary["selected"]:
                score = candidate.get("score_pure", candidate.get("indicator", 0.0))
                print(
                    f"  step={candidate.get('insertion_step', 1)} "
                    f"side={candidate['side']} x={float(candidate['x']):.8f} "
                    f"score={float(score):.8e}"
                )
        print(f"\nWrote {output_path}")
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    main()
