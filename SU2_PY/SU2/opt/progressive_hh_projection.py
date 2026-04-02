#!/usr/bin/env python

import os
import copy
import glob
import shutil
import contextlib

import numpy as np
import SU2


def get_midpoint_candidates(centers):
    centers = sorted(list(centers))
    if not centers:
        return []

    extended = [0.0] + centers + [1.0]
    candidates = []

    for i in range(len(extended) - 1):
        xm = 0.5 * (extended[i] + extended[i + 1])
        if 0.0 < xm < 1.0:
            candidates.append(
                {
                    "x": xm,
                    "interval_id": i,
                }
            )

    return candidates


def _find_real_adjoint_assets(level_dir, func_name):
    func_name = str(func_name).upper()

    folder_map = {
        "DRAG": "ADJOINT_DRAG",
        "MOMENT_Z": "ADJOINT_MOMENT_Z",
    }

    restart_map = {
        "DRAG": "solution_adj_cd.dat",
        "MOMENT_Z": "solution_adj_cmz.dat",
    }

    if func_name not in folder_map or func_name not in restart_map:
        raise FileNotFoundError(f"No adjoint asset mapping defined for {func_name}")

    adjoint_folder_name = folder_map[func_name]
    restart_name = restart_map[func_name]

    candidates = sorted(
        glob.glob(os.path.join(level_dir, "DESIGNS", "DSN_*", adjoint_folder_name))
    )
    if not candidates:
        raise FileNotFoundError(
            f"No {adjoint_folder_name} found in {os.path.join(level_dir, 'DESIGNS')}"
        )

    adjoint_dir = candidates[-1]
    design_dir = os.path.dirname(adjoint_dir)

    restart_path = os.path.join(design_dir, restart_name)
    if not os.path.exists(restart_path):
        raise FileNotFoundError(f"Missing adjoint restart file: {restart_path}")

    return adjoint_dir, restart_path


def _build_extended_dot_config(cfg_level, real_dot_cfg, mesh_name, all_upper, all_lower):
    cfg_dot = SU2.io.Config(copy.deepcopy(dict(cfg_level)))

    if "NUMBER_PART" in real_dot_cfg:
        cfg_dot["NUMBER_PART"] = int(real_dot_cfg["NUMBER_PART"])
    elif "NUMBER_PART" not in cfg_dot:
        cfg_dot["NUMBER_PART"] = 1

    if "NZONES" in real_dot_cfg:
        cfg_dot["NZONES"] = int(real_dot_cfg["NZONES"])
    elif "NZONES" not in cfg_dot:
        cfg_dot["NZONES"] = 1

    cfg_dot["MATH_PROBLEM"] = "DISCRETE_ADJOINT"
    cfg_dot["GRADIENT_METHOD"] = "DISCRETE_ADJOINT"
    cfg_dot["RESTART_SOL"] = "NO"
    cfg_dot["CONSOLE"] = "NONE"

    cfg_dot["MESH_FILENAME"] = mesh_name
    if "MULTIPOINT_MESH_FILENAME" in cfg_dot and cfg_dot["MULTIPOINT_MESH_FILENAME"]:
        cfg_dot["MULTIPOINT_MESH_FILENAME"] = f"({mesh_name})"

    old_def = copy.deepcopy(cfg_level["DEFINITION_DV"])

    marker_template = old_def["MARKER"][0]
    ffd_template = old_def["FFDTAG"][0]
    scale_template = old_def["SCALE"][0]

    kinds = []
    scales = []
    markers = []
    ffdtags = []
    params = []
    sizes = []

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


def _make_projection_state(mesh_name):
    state = SU2.io.State()
    state.FUNCTIONS = {}
    state.GRADIENTS = {}

    state.FILES["MESH"] = mesh_name
    state.FILES["DIRECT"] = "solution_flow.dat"
    state.FILES["FLOW_META"] = "flow.meta"

    return state


def _extract_constraint_names(cfg):
    names = []

    if "OPT_CONSTRAINT" not in cfg or not cfg["OPT_CONSTRAINT"]:
        return names

    opt_con = cfg["OPT_CONSTRAINT"]

    if isinstance(opt_con, dict):
        for group_name in ("EQUALITY", "INEQUALITY"):
            group = opt_con.get(group_name, {})
            if isinstance(group, dict):
                for cname in group.keys():
                    cname = str(cname).strip().upper()
                    if cname and cname not in names:
                        names.append(cname)
        return names

    if isinstance(opt_con, (list, tuple)):
        for entry in opt_con:
            if isinstance(entry, (list, tuple)) and len(entry) > 0:
                cname = str(entry[0]).strip().upper()
                if cname and cname not in names:
                    names.append(cname)
        return names

    raw = str(opt_con).strip()
    if not raw:
        return names

    parts = [p.strip() for p in raw.split(";") if p.strip()]

    for part in parts:
        if "(" in part and ")" in part:
            inside = part[part.find("(") + 1 : part.find(")")]
            tokens = (
                inside.replace("<", " ")
                .replace(">", " ")
                .replace("=", " ")
                .replace(",", " ")
                .split()
            )
            if tokens:
                cname = tokens[0].strip().upper()
                if cname and cname not in names:
                    names.append(cname)
                continue

        tokens = (
            part.replace("<", " ")
            .replace(">", " ")
            .replace("=", " ")
            .replace(",", " ")
            .split()
        )
        if tokens:
            cname = tokens[0].strip().upper()
            if cname and cname not in names:
                names.append(cname)

    return names


def _run_dot_for_function(level_dir, cfg_dot, state, func_name):
    func_name = str(func_name).upper()

    adjoint_dir, adjoint_restart = _find_real_adjoint_assets(level_dir, func_name)

    dot_test_dir = os.path.join(level_dir, f"DOT_ONLY_{func_name}")

    if os.path.isdir(dot_test_dir):
        shutil.rmtree(dot_test_dir)

    shutil.copytree(adjoint_dir, dot_test_dir, symlinks=False)

    restart_dst = os.path.join(dot_test_dir, os.path.basename(adjoint_restart))
    shutil.copy2(adjoint_restart, restart_dst)

    cfg_fun = SU2.io.Config(copy.deepcopy(dict(cfg_dot)))
    cfg_fun["OBJECTIVE_FUNCTION"] = func_name

    cwd = os.getcwd()
    try:
        os.chdir(dot_test_dir)
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                info = SU2.run.projection(cfg_fun, state)
    finally:
        os.chdir(cwd)

    return info["GRADIENTS"][func_name]


def _run_geo_gradient_for_function(level_dir, cfg_dot, func_name):
    func_name = str(func_name).upper()

    geo_test_dir = os.path.join(level_dir, f"GEO_ONLY_{func_name}")

    if os.path.isdir(geo_test_dir):
        shutil.rmtree(geo_test_dir)

    os.makedirs(geo_test_dir, exist_ok=True)

    cfg_geo = SU2.io.Config(copy.deepcopy(dict(cfg_dot)))
    cfg_geo["GEO_PARAM"] = func_name
    cfg_geo["GEO_MODE"] = "GRADIENT"
    cfg_geo["CONSOLE"] = "NONE"

    mesh_name = str(cfg_geo["MESH_FILENAME"])
    mesh_src = os.path.join(level_dir, mesh_name)
    mesh_dst = os.path.join(geo_test_dir, mesh_name)

    if not os.path.exists(mesh_src):
        raise FileNotFoundError(f"Missing mesh for geometry run: {mesh_src}")

    if os.path.abspath(mesh_src) != os.path.abspath(mesh_dst):
        shutil.copy2(mesh_src, mesh_dst)

    cwd = os.getcwd()
    try:
        os.chdir(geo_test_dir)
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                info = SU2.run.geometry(cfg_geo)
    finally:
        os.chdir(cwd)

    return info["GRADIENTS"][func_name]


def _compute_ikkt_residual_vector(g_obj, constraint_grads):
    g = np.asarray(g_obj, dtype=float)

    if not constraint_grads:
        return g.copy()

    A = np.column_stack([np.asarray(cg, dtype=float) for cg in constraint_grads])

    try:
        lam, _, _, _ = np.linalg.lstsq(A, g, rcond=None)
        r = g - A @ lam
    except Exception:
        r = g.copy()

    return r


def _compute_dot_candidate_scores(level, opts):
    cfg_path = os.path.join(level.workdir, level.config_filename)
    cfg_level = SU2.io.Config(cfg_path)

    active_upper = list(level.upper)
    active_lower = list(level.lower)

    cand_upper_raw = get_midpoint_candidates(active_upper)
    cand_lower_raw = get_midpoint_candidates(active_lower)

    cand_upper = [c["x"] for c in cand_upper_raw]
    cand_lower = [c["x"] for c in cand_lower_raw]

    if not cand_upper and not cand_lower:
        return {
            "candidates": [],
            "active_upper_scores": [],
            "active_lower_scores": [],
        }

    all_upper = active_upper + cand_upper
    all_lower = active_lower + cand_lower

    obj_name = str(cfg_level.get("OBJECTIVE_FUNCTION", "DRAG")).upper()
    obj_adj_dir, _ = _find_real_adjoint_assets(level.workdir, obj_name)
    real_dot_cfg = SU2.io.Config(os.path.join(obj_adj_dir, "config_DOT_AD.cfg"))
    mesh_name = str(real_dot_cfg["MESH_FILENAME"])

    cfg_dot = _build_extended_dot_config(
        cfg_level, real_dot_cfg, mesh_name, all_upper, all_lower
    )
    state = _make_projection_state(mesh_name)

    grad_obj = _run_dot_for_function(level.workdir, cfg_dot, state, obj_name)

    n_active = len(active_upper) + len(active_lower)
    n_upper_active = len(active_upper)

    grad_active = grad_obj[:n_active]
    grad_candidate = grad_obj[n_active:]

    if len(grad_candidate) == 0:
        raise RuntimeError("DOT returned empty candidate gradient")

    expected_ncand = len(cand_upper) + len(cand_lower)
    if len(grad_candidate) != expected_ncand:
        raise RuntimeError(
            f"DOT candidate gradient size mismatch: got {len(grad_candidate)}, expected {expected_ncand}"
        )

    indicator_mode = opts.get("adaptive_indicator", "ABS_GRAD").upper()

    if indicator_mode == "IKKT":
        constraint_names = _extract_constraint_names(cfg_level)
        constraint_grads_full = []

        for cname in constraint_names:
            cname = cname.upper()

            try:
                grad_c = _run_dot_for_function(level.workdir, cfg_dot, state, cname)
                print(f"[PROGRESSIVE_HH] IKKT | {cname} via DOT")
            except Exception:
                try:
                    grad_c = _run_geo_gradient_for_function(level.workdir, cfg_dot, cname)
                    print(f"[PROGRESSIVE_HH] IKKT | {cname} via GEOMETRY")
                except Exception:
                    print(
                        "[PROGRESSIVE_HH] IKKT warning | "
                        f"{cname} gradient not available -> skipped"
                    )
                    continue

            if len(grad_c) != len(grad_obj):
                raise RuntimeError(
                    f"{cname} full gradient size mismatch: "
                    f"got {len(grad_c)}, expected {len(grad_obj)}"
                )

            constraint_grads_full.append(grad_c)

        residual_full = _compute_ikkt_residual_vector(grad_obj, constraint_grads_full)
        active_indicator = np.abs(np.asarray(residual_full[:n_active], dtype=float)).tolist()
        candidate_indicator = np.abs(
            np.asarray(residual_full[n_active:], dtype=float)
        ).tolist()
    else:
        active_indicator = np.abs(np.asarray(grad_active, dtype=float)).tolist()
        candidate_indicator = np.abs(np.asarray(grad_candidate, dtype=float)).tolist()

    active_upper_scores = active_indicator[:n_upper_active]
    active_lower_scores = active_indicator[n_upper_active:]

    candidates = []
    k = 0

    for c in cand_upper_raw:
        candidates.append(
            {
                "side": "UPPER",
                "x": float(c["x"]),
                "grad": float(grad_candidate[k]),
                "indicator": float(candidate_indicator[k]),
                "interval_id": c["interval_id"],
            }
        )
        k += 1

    for c in cand_lower_raw:
        candidates.append(
            {
                "side": "LOWER",
                "x": float(c["x"]),
                "grad": float(grad_candidate[k]),
                "indicator": float(candidate_indicator[k]),
                "interval_id": c["interval_id"],
            }
        )
        k += 1

    print(
        f"[PROGRESSIVE_HH] Candidate scoring | mode={indicator_mode} "
        f"active_ndv={n_active} candidate_ndv={expected_ncand}"
    )
    for c in candidates:
        print(
            "[PROGRESSIVE_HH] Candidate | "
            f"side={c['side']} x={c['x']:.6f} I={c['indicator']:.6e}"
        )

    return {
        "candidates": candidates,
        "active_upper_scores": active_upper_scores,
        "active_lower_scores": active_lower_scores,
    }