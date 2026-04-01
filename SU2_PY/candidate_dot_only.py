#!/usr/bin/env python3

import os
import sys
import copy
import glob
import shutil
import argparse
import contextlib

sys.path.append(os.environ["SU2_RUN"])
import SU2


def extract_level_from_config(cfg_path):
    cfg = SU2.io.Config(cfg_path)
    def_dv = cfg["DEFINITION_DV"]

    upper = []
    lower = []

    for param in def_dv["PARAM"]:
        side_flag = float(param[0])
        x_loc = float(param[1])
        if side_flag == 1.0:
            upper.append(x_loc)
        else:
            lower.append(x_loc)

    return cfg, sorted(upper), sorted(lower)


def find_real_adjoint_dir(level_dir):
    candidates = sorted(
        glob.glob(os.path.join(level_dir, "DESIGNS", "DSN_*", "ADJOINT_DRAG"))
    )
    if not candidates:
        raise FileNotFoundError(f"No ADJOINT_DRAG found in {level_dir}")
    return candidates[-1]


def build_extended_dot_config(project, cfg_level, all_upper, all_lower):
    cfg_dot = SU2.io.Config(copy.deepcopy(dict(cfg_level)))

    if "NUMBER_PART" in project.config:
        cfg_dot["NUMBER_PART"] = int(project.config["NUMBER_PART"])
    elif "NUMBER_PART" in cfg_level:
        cfg_dot["NUMBER_PART"] = int(cfg_level["NUMBER_PART"])
    else:
        cfg_dot["NUMBER_PART"] = 1

    if "NZONES" in project.config:
        cfg_dot["NZONES"] = int(project.config["NZONES"])
    elif "NZONES" in cfg_level:
        cfg_dot["NZONES"] = int(cfg_level["NZONES"])
    else:
        cfg_dot["NZONES"] = 1

    cfg_dot["MATH_PROBLEM"] = "DISCRETE_ADJOINT"
    cfg_dot["GRADIENT_METHOD"] = "DISCRETE_ADJOINT"
    cfg_dot["OBJECTIVE_FUNCTION"] = "DRAG"
    cfg_dot["RESTART_SOL"] = "NO"
    cfg_dot["CONSOLE"] = "NONE"

    # Important: inside DOT_ONLY_TEST these names already exist
    cfg_dot["MESH_FILENAME"] = "mesh_RAE2822_turb.su2"
    if "MULTIPOINT_MESH_FILENAME" in cfg_dot and cfg_dot["MULTIPOINT_MESH_FILENAME"]:
        cfg_dot["MULTIPOINT_MESH_FILENAME"] = "(mesh_RAE2822_turb.su2)"

    # Build extended DEFINITION_DV by mirroring the real active config structure
    old_def = copy.deepcopy(cfg_level["DEFINITION_DV"])

    marker_template = old_def["MARKER"][0]
    ffd_template = old_def["FFDTAG"][0]

    kinds = []
    scales = []
    markers = []
    ffdtags = []
    params = []
    sizes = []

    # use first active entry scale as template
    scale_template = old_def["SCALE"][0]

    for x in all_upper:
        kinds.append("HICKS_HENNE")
        scales.append(scale_template)
        markers.append(copy.deepcopy(marker_template))
        ffdtags.append(copy.deepcopy(ffd_template))
        params.append([1.0, float(x)])
        sizes.append(1)

    for x in all_lower:
        kinds.append("HICKS_HENNE")
        scales.append(scale_template)
        markers.append(copy.deepcopy(marker_template))
        ffdtags.append(copy.deepcopy(ffd_template))
        params.append([0.0, float(x)])
        sizes.append(1)

    cfg_dot["DEFINITION_DV"] = {
        "KIND": kinds,
        "SCALE": scales,
        "MARKER": markers,
        "FFDTAG": ffdtags,
        "PARAM": params,
        "SIZE": sizes,
    }

    ndv = len(all_upper) + len(all_lower)
    cfg_dot["DV_VALUE_NEW"] = [0.0] * ndv
    cfg_dot["DV_VALUE_OLD"] = [0.0] * ndv

    return cfg_dot


def make_projection_state():
    state = SU2.io.State()
    state.FUNCTIONS = {}
    state.GRADIENTS = {}

    state.FILES["MESH"] = "mesh_RAE2822_turb.su2"
    state.FILES["DIRECT"] = "solution_flow.dat"
    state.FILES["FLOW_META"] = "flow.meta"

    return state


def compute_dot_only_gradient(project_path, cfg_path):
    project_path = os.path.abspath(project_path)
    cfg_path = os.path.abspath(cfg_path)

    if not os.path.isfile(project_path):
        raise FileNotFoundError(f"Project file not found: {project_path}")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    project = SU2.io.load_data(project_path)

    cfg_level, active_upper, active_lower = extract_level_from_config(cfg_path)
    level_dir = os.path.dirname(cfg_path)

    # === candidate test set ===
    # For now: one midpoint candidate on upper and one on lower
    cand_upper = [0.375]
    cand_lower = [0.375]

    all_upper = list(active_upper) + list(cand_upper)
    all_lower = list(active_lower) + list(cand_lower)

    real_adj_dir = find_real_adjoint_dir(level_dir)
    dot_test_dir = os.path.join(level_dir, "DOT_ONLY_TEST")

    if os.path.isdir(dot_test_dir):
        shutil.rmtree(dot_test_dir)

    shutil.copytree(real_adj_dir, dot_test_dir, symlinks=False)

    # Ensure the adjoint restart file exists in DOT_ONLY_TEST
    real_design_dir = os.path.dirname(real_adj_dir)
    adj_restart_src = os.path.join(real_design_dir, "solution_adj_cd.dat")
    adj_restart_dst = os.path.join(dot_test_dir, "solution_adj_cd.dat")

    if not os.path.exists(adj_restart_src):
        raise FileNotFoundError(f"Missing adjoint restart file: {adj_restart_src}")

    shutil.copy2(adj_restart_src, adj_restart_dst)

    cfg_dot = build_extended_dot_config(project, cfg_level, all_upper, all_lower)
    print("[DOT-DEBUG] ext DEFINITION_DV PARAM =", cfg_dot["DEFINITION_DV"]["PARAM"])
    print("[DOT-DEBUG] ext NDV =", len(cfg_dot["DEFINITION_DV"]["PARAM"]))
    state = make_projection_state()

    print("[DOT-DEBUG] real_adj_dir =", real_adj_dir)
    print("[DOT-DEBUG] dot_test_dir =", dot_test_dir)
    print("[DOT-DEBUG] state FILES =", state.FILES)
    print("[DOT-DEBUG] cfg MESH_FILENAME =", cfg_dot["MESH_FILENAME"])
    print("[DOT-DEBUG] cfg NUMBER_PART =", cfg_dot["NUMBER_PART"])
    print("[DOT-DEBUG] cfg NZONES =", cfg_dot["NZONES"])
    print("[DOT-DEBUG] active_upper =", active_upper)
    print("[DOT-DEBUG] active_lower =", active_lower)
    print("[DOT-DEBUG] cand_upper =", cand_upper)
    print("[DOT-DEBUG] cand_lower =", cand_lower)
    print("[DOT-DEBUG] all_upper =", all_upper)
    print("[DOT-DEBUG] all_lower =", all_lower)


    cwd = os.getcwd()
    try:
        os.chdir(dot_test_dir)
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull),contextlib.redirect_stderr(devnull):
                info = SU2.run.projection(cfg_dot, state)
    finally:
        os.chdir(cwd)

    gradients = info.get("GRADIENTS", {})
    grad_drag = gradients.get("DRAG", None)

    if grad_drag is None:
        raise RuntimeError("DRAG gradient not found in projection output")

    n_active = len(active_upper) + len(active_lower)
    grad_active = grad_drag[:n_active]
    grad_candidate = grad_drag[n_active:]

    return {
        "real_adj_dir": real_adj_dir,
        "dot_test_dir": dot_test_dir,
        "info_keys": list(info.keys()),
        "grad_drag": grad_drag,
        "grad_active": grad_active,
        "grad_candidate": grad_candidate,
        "active_upper": active_upper,
        "active_lower": active_lower,
        "cand_upper": cand_upper,
        "cand_lower": cand_lower,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project",
        required=True,
        help="Path to LEVEL_X/project_levelX.pkl",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to LEVEL_X/config_levelX.cfg",
    )
    args = parser.parse_args()

    out = compute_dot_only_gradient(args.project, args.config)

    print("[DOT] real_adj_dir =", out["real_adj_dir"])
    print("[DOT] dot_test_dir =", out["dot_test_dir"])
    print("[DOT] info_keys =", out["info_keys"])
    print("[DOT] active_upper =", out["active_upper"])
    print("[DOT] active_lower =", out["active_lower"])
    print("[DOT] cand_upper =", out["cand_upper"])
    print("[DOT] cand_lower =", out["cand_lower"])
    print("[DOT] grad_drag =", out["grad_drag"])
    print("[DOT] grad_active =", out["grad_active"])
    print("[DOT] grad_candidate =", out["grad_candidate"])


if __name__ == "__main__":
    main()
