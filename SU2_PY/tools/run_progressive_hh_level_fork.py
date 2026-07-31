#!/usr/bin/env python3

"""Run one isolated progressive HH level from a saved score selection."""

import argparse
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

from shape_optimization import _build_online_trigger_opts, run_single_level
from SU2.opt.progressive_hh_core import HHLevel, get_progressive_hh_options
from SU2.opt.progressive_hh_levels import (
    build_spring_reallocated_level,
    collect_level_result,
    write_level_config,
)
from SU2.opt.thickness_constraint import build_thickness_constraint_from_config


def _resolve_level_config(level_dir, requested=None):
    if requested:
        path = requested if os.path.isabs(requested) else os.path.join(level_dir, requested)
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        return path
    candidates = sorted(Path(level_dir).glob("config_level*.cfg"))
    if len(candidates) != 1:
        raise RuntimeError(
            "Could not identify one source level config; use --source-level-config"
        )
    return str(candidates[0].resolve())


def _infer_level_id(level_dir, config_path):
    for value in (os.path.basename(level_dir), os.path.basename(config_path)):
        match = re.search(r"level[_-]?(\d+)", value, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return 0


def _infer_hh_centers(level_config):
    definition = level_config.get("DEFINITION_DV")
    if not isinstance(definition, dict):
        raise RuntimeError("Source level config has no parsed DEFINITION_DV")
    upper = []
    lower = []
    for kind, raw in zip(definition.get("KIND", []), definition.get("PARAM", [])):
        if str(kind).strip().upper() != "HICKS_HENNE":
            raise RuntimeError(f"Expected HICKS_HENNE, got {kind!r}")
        values = [float(value) for value in raw]
        if len(values) < 2:
            raise RuntimeError(f"Invalid Hicks--Henne parameters {raw!r}")
        (upper if bool(round(values[-2])) else lower).append(values[-1])
    return sorted(upper), sorted(lower)


def _load_score_summary(score_json):
    with open(score_json) as stream:
        payload = json.load(stream)
    modes = payload.get("modes", {})
    summary = modes.get("VIRTUAL_TANGENT", payload)
    selected = list(summary.get("selected", summary.get("selected_candidates", [])))
    if not selected:
        raise RuntimeError(f"No VIRTUAL_TANGENT selection found in {score_json}")
    return selected, summary


def _add_selected(upper, lower, selected):
    upper = list(upper)
    lower = list(lower)
    for candidate in selected:
        side = str(candidate.get("side", "")).upper()
        x = float(candidate["x"])
        if side == "UPPER":
            upper.append(x)
        elif side == "LOWER":
            lower.append(x)
        elif side == "PAIR":
            upper.append(x)
            lower.append(x)
        else:
            raise RuntimeError(f"Unsupported selected HH side {side!r}")
    return sorted(set(upper)), sorted(set(lower))


def _source_result(level_dir, config_path, upper, lower):
    source_level = HHLevel(
        level_id=_infer_level_id(level_dir, config_path),
        upper=upper,
        lower=lower,
        workdir=os.path.abspath(level_dir),
        config_filename=os.path.basename(config_path),
        project_filename="project.pkl",
    )
    projects = sorted(Path(level_dir).glob("project*.pkl"))
    if len(projects) != 1:
        raise RuntimeError(
            "Could not identify one source-level project pickle required to "
            "resolve an accepted SLSQP design"
        )
    source_project = SU2.io.load_data(str(projects[0]))
    return collect_level_result(source_level, project=source_project)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--source-level-dir", required=True)
    parser.add_argument("--source-level-config")
    parser.add_argument("--score-json", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--partitions", type=int, default=1)
    parser.add_argument("--gradient", default="DISCRETE_ADJOINT")
    parser.add_argument("--optimizer", default="SLSQP")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)

    base_config_path = os.path.abspath(args.base_config)
    source_level_dir = os.path.abspath(args.source_level_dir)
    source_config_path = _resolve_level_config(
        source_level_dir,
        args.source_level_config,
    )
    score_json = os.path.abspath(args.score_json)
    output_root = os.path.abspath(args.output_root)
    if os.path.exists(output_root) and os.listdir(output_root):
        raise RuntimeError(f"Output root is not empty: {output_root}")
    os.makedirs(output_root, exist_ok=True)
    score_snapshot = os.path.join(output_root, "level0_virtual_tangent_score.json")
    shutil.copy2(score_json, score_snapshot)

    base_config = SU2.io.Config(base_config_path)
    base_config["PROGRESSIVE_HH_SCORING_MODE"] = "VIRTUAL_TANGENT"
    original_restart_filename = str(base_config.get("RESTART_FILENAME", ""))
    # SU2_SOL 8.4 uses a fixed-size internal buffer for the solution base
    # name.  A long isolated output path can abort in libc string handling,
    # so keep this unique workspace deliberately short.
    restart_workspace = tempfile.mkdtemp(prefix="hhvt_", dir="/tmp")
    base_config["RESTART_FILENAME"] = os.path.join(
        restart_workspace,
        "restart_flow",
    )
    opts = get_progressive_hh_options(base_config)
    source_config = SU2.io.Config(source_config_path)
    source_upper, source_lower = _infer_hh_centers(source_config)
    selected, score_summary = _load_score_summary(score_json)
    copied_score_artifacts = {}
    for key, destination_name in (
        ("candidate_scores_csv", "level0_hh_candidate_scores.csv"),
        ("metadata_json", "level0_hh_virtual_tangent_metadata.json"),
    ):
        source = score_summary.get(key)
        if source and os.path.isfile(source):
            destination = os.path.join(output_root, destination_name)
            shutil.copy2(source, destination)
            copied_score_artifacts[key] = destination
    upper, lower = _add_selected(source_upper, source_lower, selected)
    source_result = _source_result(
        source_level_dir,
        source_config_path,
        source_upper,
        source_lower,
    )
    mesh_source = source_result.get("final_mesh")
    if not mesh_source or not os.path.isfile(mesh_source):
        raise FileNotFoundError(f"Source accepted mesh unavailable: {mesh_source}")

    level_id = _infer_level_id(source_level_dir, source_config_path) + 1
    level_dir = os.path.join(output_root, f"LEVEL_{level_id}")
    level = HHLevel(
        level_id=level_id,
        upper=upper,
        lower=lower,
        workdir=level_dir,
        config_filename=f"config_level{level_id}.cfg",
        project_filename=f"project_level{level_id}.pkl",
        mesh_source=mesh_source,
        initial_mesh_source=mesh_source,
        dv_values=[0.0] * (len(upper) + len(lower)),
    )
    cfg_path = write_level_config(base_config, level, opts)
    manifest = {
        "base_config": base_config_path,
        "original_restart_filename": original_restart_filename,
        "isolated_restart_filename": base_config["RESTART_FILENAME"],
        "restart_workspace": restart_workspace,
        "source_level_dir": source_level_dir,
        "source_level_config": source_config_path,
        "source_final_mesh": mesh_source,
        "score_json": score_json,
        "score_json_snapshot": score_snapshot,
        "score_artifacts": copied_score_artifacts,
        "selected": selected,
        "upper_before": source_upper,
        "lower_before": source_lower,
        "upper_level": upper,
        "lower_level": lower,
        "level_id": level_id,
        "level_config": cfg_path,
        "trigger": opts["trigger"],
        "trigger_options": _build_online_trigger_opts(
            opts,
            level_id,
            current_ndv=level.ndv,
        ),
    }
    manifest_path = os.path.join(output_root, "fork_manifest.json")
    with open(manifest_path, "w") as stream:
        json.dump(manifest, stream, allow_nan=False, indent=2, sort_keys=True)
    print(f"Prepared {cfg_path}")
    print(f"Centers UPPER: {upper}")
    print(f"Centers LOWER: {lower}")
    print(f"Trigger: {opts['trigger']}")
    if args.prepare_only:
        print(f"Wrote {manifest_path}")
        return

    thickness_constraint = build_thickness_constraint_from_config(base_config)
    trigger_opts = _build_online_trigger_opts(
        opts,
        level_id,
        current_ndv=level.ndv,
    )
    cwd = os.getcwd()
    try:
        os.chdir(level.workdir)
        project = run_single_level(
            os.path.basename(cfg_path),
            os.path.basename(level.project_filename),
            args.partitions,
            args.gradient,
            args.optimizer,
            False,
            1,
            trigger_opts=trigger_opts,
            progressive_hh_opts=opts,
            thickness_constraint=thickness_constraint,
            progressive_label="PROGRESSIVE_HH_VIRTUAL_TANGENT",
        )
    finally:
        os.chdir(cwd)
    result = collect_level_result(level, project=project)
    dv_values = result["dv_values"]
    spring_level = None
    if (
        bool(opts.get("spring_enabled", False))
        and str(opts.get("spring_timing", "POST_OPT")).upper() == "POST_OPT"
    ):
        spring_level = build_spring_reallocated_level(
            level,
            result,
            opts,
            reoptimize=False,
        )
    manifest.update(
        {
            "history_file": result.get("history_file"),
            "final_mesh": result.get("final_mesh"),
            "optimized_dv_values": dv_values,
            "refinement_triggered": bool(
                getattr(project, "refinement_triggered", False)
            ),
            "post_spring_upper": (
                None if spring_level is None else list(spring_level.upper)
            ),
            "post_spring_lower": (
                None if spring_level is None else list(spring_level.lower)
            ),
        }
    )
    with open(manifest_path, "w") as stream:
        json.dump(manifest, stream, allow_nan=False, indent=2, sort_keys=True)
    print(f"Completed level {level_id}; wrote {manifest_path}")


if __name__ == "__main__":
    main()
