#!/usr/bin/env python

import os
import shutil
import copy
import glob
import contextlib

import numpy as np
import SU2

from SU2.opt.progressive_hh_projection import (
    _build_extended_dot_config,
    _make_projection_state,
    _run_geo_gradient_for_function,
    _extract_constraint_names,
    _extract_constraint_signs,
    _compute_ikkt_residual_vector,
)

# ============================================================
# USER SETTINGS
# ============================================================

NCAND = 15
N_UPPER = 3
N_LOWER = 3
DX_MIN = 0.15

INDICATOR_MODE = "IKKT"   # "ABS_GRAD" or "IKKT"

N_PART = 6
N_ZONES = 1

TMP_DIR = "INIT_PICK_RUN"

GEOMETRY_FUNCTIONS = {
    "AIRFOIL_AREA",
    "AIRFOIL_THICKNESS",
}

# ============================================================


def generate_uniform_candidates(ncand):
    return [(i + 1) / float(ncand + 1) for i in range(ncand)]


def select_with_spacing(xs, scores, n_select, dx_min):
    pairs = sorted(zip(xs, scores), key=lambda p: -p[1])

    selected = []
    for x, s in pairs:
        if all(abs(x - xj) > dx_min for xj in selected):
            selected.append(x)
        if len(selected) == n_select:
            break

    return sorted(selected)


def _remove_progressive_keys(cfg):
    progressive_keys = [
        "PROGRESSIVE_HH",
        "PROGRESSIVE_HH_NLEVELS",
        "PROGRESSIVE_HH_N0",
        "PROGRESSIVE_HH_INITIAL_UPPER",
        "PROGRESSIVE_HH_INITIAL_LOWER",
        "PROGRESSIVE_HH_SURFACE",
        "PROGRESSIVE_HH_TRIGGER",
        "PROGRESSIVE_HH_MAX_ITER_PER_LEVEL",
        "PROGRESSIVE_HH_WINDOW",
        "PROGRESSIVE_HH_TOL",
        "PROGRESSIVE_HH_SLOPE_FILTER_TOL",
        "PROGRESSIVE_HH_STAG_TOL",
        "PROGRESSIVE_HH_STAG_BAND",
        "PROGRESSIVE_HH_STAG_WINDOW",
        "PROGRESSIVE_HH_REFINEMENT",
        "PROGRESSIVE_HH_GROWTH_RATIO",
        "PROGRESSIVE_HH_ADAPTIVE_INDICATOR",
        "PROGRESSIVE_HH_SPRING",
        "PROGRESSIVE_HH_SPRING_A",
    ]

    for key in progressive_keys:
        if key in cfg:
            del cfg[key]


def _copy_mesh_into_workdir(cfg, original_cwd):
    mesh_file = str(cfg["MESH_FILENAME"])

    if not os.path.isabs(mesh_file):
        mesh_file = os.path.abspath(os.path.join(original_cwd, mesh_file))

    if not os.path.exists(mesh_file):
        raise FileNotFoundError(f"Mesh file not found: {mesh_file}")

    mesh_basename = os.path.basename(mesh_file)
    shutil.copy2(mesh_file, mesh_basename)

    cfg["MESH_FILENAME"] = mesh_basename

    if "MULTIPOINT_MESH_FILENAME" in cfg and cfg["MULTIPOINT_MESH_FILENAME"]:
        cfg["MULTIPOINT_MESH_FILENAME"] = f"({mesh_basename})"

    return mesh_basename


def _prepare_base_cfg(cfg):
    cfg["NUMBER_PART"] = int(N_PART)
    cfg["NZONES"] = int(N_ZONES)
    cfg["GRADIENT_METHOD"] = "DISCRETE_ADJOINT"
    cfg["RESTART_SOL"] = "NO"
    cfg["CONSOLE"] = "NONE"

    _remove_progressive_keys(cfg)

    return cfg


def _prepare_direct_cfg(cfg):
    cfg = SU2.io.Config(copy.deepcopy(dict(cfg)))

    cfg = _prepare_base_cfg(cfg)
    cfg["MATH_PROBLEM"] = "DIRECT"
    cfg["SOLUTION_FILENAME"] = "restart_flow"
    cfg["RESTART_FILENAME"] = "restart_flow"

    return cfg


def _prepare_adjoint_cfg(cfg, func_name):
    cfg = SU2.io.Config(copy.deepcopy(dict(cfg)))

    cfg = _prepare_base_cfg(cfg)
    cfg["MATH_PROBLEM"] = "DISCRETE_ADJOINT"
    cfg["OBJECTIVE_FUNCTION"] = str(func_name).upper()
    cfg["RESTART_SOL"] = "NO"

    # Make adjoint read the direct restart we copy from DIRECT/
    cfg["SOLUTION_FILENAME"] = "restart_flow"
    cfg["RESTART_FILENAME"] = "restart_flow"

    return cfg


def _tmp_root():
    return os.path.abspath(TMP_DIR)


def _direct_dir():
    return os.path.join(_tmp_root(), "DIRECT")


def _adjoint_dir(func_name):
    return os.path.join(_tmp_root(), f"ADJOINT_{str(func_name).upper()}")


def run_baseline_direct(cfg_file):
    direct_dir = _direct_dir()

    if os.path.isdir(TMP_DIR):
        shutil.rmtree(TMP_DIR)

    os.makedirs(direct_dir, exist_ok=True)

    cwd = os.getcwd()

    try:
        cfg = SU2.io.Config(cfg_file)

        os.chdir(direct_dir)

        cfg = _prepare_direct_cfg(cfg)
        mesh_basename = _copy_mesh_into_workdir(cfg, cwd)

        cfg.dump("config.cfg")

        print("[INIT_PICK] Running baseline direct...")

        state = SU2.io.State()
        state.FUNCTIONS = {}
        state.GRADIENTS = {}

        state.FILES["MESH"] = mesh_basename
        state.FILES["DIRECT"] = "restart_flow.dat"
        state.FILES["FLOW_META"] = "flow.meta"

        SU2.run.direct(cfg)

    finally:
        os.chdir(cwd)


def run_adjoint_for_function(cfg_file, func_name):
    func_name = str(func_name).upper()

    direct_dir = _direct_dir()
    adj_dir = _adjoint_dir(func_name)

    os.makedirs(adj_dir, exist_ok=True)

    cwd = os.getcwd()

    try:
        cfg = SU2.io.Config(cfg_file)

        os.chdir(adj_dir)

        cfg = _prepare_adjoint_cfg(cfg, func_name)
        mesh_basename = _copy_mesh_into_workdir(cfg, cwd)

        # Copy direct restart into this adjoint folder
        direct_restart = os.path.join(direct_dir, "restart_flow.dat")
        if not os.path.exists(direct_restart):
            raise FileNotFoundError(f"Missing direct restart: {direct_restart}")

        shutil.copy2(direct_restart, "restart_flow.dat")

        # Copy optional flow meta if present
        direct_meta = os.path.join(direct_dir, "flow.meta")
        if os.path.exists(direct_meta):
            shutil.copy2(direct_meta, "flow.meta")

        cfg.dump("config.cfg")

        print(f"[INIT_PICK] Running adjoint for {func_name}...")

        state = SU2.io.State()
        state.FUNCTIONS = {}
        state.GRADIENTS = {}

        state.FILES["MESH"] = mesh_basename
        state.FILES["DIRECT"] = "restart_flow.dat"
        state.FILES["FLOW_META"] = "flow.meta"

        SU2.run.adjoint(cfg)

    finally:
        os.chdir(cwd)


def _find_init_adjoint_assets(func_name):
    func_name = str(func_name).upper()

    adj_dir = _adjoint_dir(func_name)
    if not os.path.isdir(adj_dir):
        raise FileNotFoundError(f"Missing adjoint directory: {adj_dir}")

    restart_candidates = []

    restart_candidates.extend(
        sorted(glob.glob(os.path.join(adj_dir, "solution_adj*.dat")))
    )
    restart_candidates.extend(
        sorted(glob.glob(os.path.join(adj_dir, "restart_adj*.dat")))
    )

    if not restart_candidates:
        raise FileNotFoundError(f"No adjoint restart files found in {adj_dir}")

    return adj_dir, restart_candidates


def _run_dot_for_function_init(cfg_dot, state, func_name):
    func_name = str(func_name).upper()

    adjoint_dir, restart_candidates = _find_init_adjoint_assets(func_name)

    dot_test_dir = os.path.join(TMP_DIR, f"DOT_ONLY_{func_name}")

    if os.path.isdir(dot_test_dir):
        shutil.rmtree(dot_test_dir)

    shutil.copytree(adjoint_dir, dot_test_dir, symlinks=False)

    # Copy all adjoint restart files found in that adjoint directory
    for restart_src in restart_candidates:
        base = os.path.basename(restart_src)

        if base.startswith("restart_adj"):
            base = base.replace("restart_adj", "solution_adj", 1)

        restart_dst = os.path.join(dot_test_dir, base)
        shutil.copy2(restart_src, restart_dst)

    # Copy direct restart too, in case projection needs it
    direct_restart = os.path.join(_direct_dir(), "restart_flow.dat")
    if os.path.exists(direct_restart):
        shutil.copy2(direct_restart, os.path.join(dot_test_dir, "restart_flow.dat"))

    direct_meta = os.path.join(_direct_dir(), "flow.meta")
    if os.path.exists(direct_meta):
        shutil.copy2(direct_meta, os.path.join(dot_test_dir, "flow.meta"))

    cfg_fun = SU2.io.Config(copy.deepcopy(dict(cfg_dot)))
    cfg_fun["OBJECTIVE_FUNCTION"] = func_name

    cwd = os.getcwd()
    try:
        os.chdir(dot_test_dir)

        # -----------------------------
        # REMOVE PROGRESSIVE_HH OPTIONS
        # -----------------------------
        keys_to_remove = [
            k for k in cfg_fun.keys()
            if str(k).startswith("PROGRESSIVE_HH")
        ]

        for k in keys_to_remove:
            del cfg_fun[k]

        info = SU2.run.projection(cfg_fun, state)

    finally:
        os.chdir(cwd)

    return info["GRADIENTS"][func_name]

def _run_geo_gradient_for_function_init(cfg_dot, func_name):
    func_name = str(func_name).upper()

    geo_test_dir = os.path.join(_tmp_root(), f"GEO_ONLY_{func_name}")

    if os.path.isdir(geo_test_dir):
        shutil.rmtree(geo_test_dir)

    os.makedirs(geo_test_dir, exist_ok=True)

    cfg_geo = SU2.io.Config(copy.deepcopy(dict(cfg_dot)))
    cfg_geo["GEO_PARAM"] = func_name
    cfg_geo["GEO_MODE"] = "GRADIENT"
    cfg_geo["CONSOLE"] = "NONE"

    # remove progressive keys
    _remove_progressive_keys(cfg_geo)

    # copy mesh from DIRECT
    direct_mesh = os.path.join(_direct_dir(), str(cfg_geo["MESH_FILENAME"]))
    if not os.path.exists(direct_mesh):
        raise FileNotFoundError(f"Missing mesh for geometry run: {direct_mesh}")

    mesh_basename = os.path.basename(direct_mesh)
    shutil.copy2(direct_mesh, os.path.join(geo_test_dir, mesh_basename))
    cfg_geo["MESH_FILENAME"] = mesh_basename

    cwd = os.getcwd()
    try:
        os.chdir(geo_test_dir)
        info = SU2.run.geometry(cfg_geo)
    finally:
        os.chdir(cwd)

    return info["GRADIENTS"][func_name]

def run_required_adjoint_set(cfg_file, indicator_mode):
    cfg = SU2.io.Config(cfg_file)

    obj_name = str(cfg.get("OBJECTIVE_FUNCTION", "DRAG")).upper()
    needed = [obj_name]

    if indicator_mode.upper() == "IKKT":
        for cname in _extract_constraint_names(cfg):
            cname = str(cname).upper()
            if cname not in GEOMETRY_FUNCTIONS and cname not in needed:
                needed.append(cname)

    print(f"[INIT_PICK] Required adjoints = {needed}")

    for func_name in needed:
        run_adjoint_for_function(cfg_file, func_name)


def _find_any_adjoint_dir():
    adj_dirs = sorted(
        d for d in os.listdir(TMP_DIR)
        if d.startswith("ADJOINT_") and os.path.isdir(os.path.join(TMP_DIR, d))
    )

    if not adj_dirs:
        raise RuntimeError("No ADJOINT_* folders found after baseline run")

    return os.path.join(TMP_DIR, adj_dirs[0])


def compute_initial_scores(cfg_file, indicator_mode):
    cfg = SU2.io.Config(cfg_file)

    cand_upper = generate_uniform_candidates(NCAND)
    cand_lower = generate_uniform_candidates(NCAND)

    all_upper = cand_upper
    all_lower = cand_lower

    obj_name = str(cfg.get("OBJECTIVE_FUNCTION", "DRAG")).upper()

    # Recover a reference config from the DIRECT folder
    direct_cfg_path = os.path.join(_direct_dir(), "config.cfg")
    real_dot_cfg = SU2.io.Config(direct_cfg_path)
    mesh_name = str(real_dot_cfg["MESH_FILENAME"])

    cfg_dot = _build_extended_dot_config(
        cfg,
        real_dot_cfg,
        mesh_name,
        all_upper,
        all_lower,
    )
    _remove_progressive_keys(cfg_dot) 
    state = _make_projection_state(mesh_name)
    state.FILES["DIRECT"] = "restart_flow.dat"

    grad_obj = _run_dot_for_function_init(cfg_dot, state, obj_name)
    grad_obj = np.asarray(grad_obj, dtype=float)

    n_upper = len(cand_upper)

    if indicator_mode.upper() == "ABS_GRAD":
        indicator = np.abs(grad_obj)

    elif indicator_mode.upper() == "IKKT":
        constraint_names = _extract_constraint_names(cfg)
        lambda_bounds = _extract_constraint_signs(cfg, constraint_names)

        print(f"[INIT_PICK] IKKT constraint order = {constraint_names}")

        constraint_grads_full = []

        for cname in constraint_names:
            cname = str(cname).upper()

            try:
                grad_c = _run_dot_for_function_init(cfg_dot, state, cname)
                print(f"[INIT_PICK] IKKT | {cname} via DOT")
            except Exception:
                if cname in GEOMETRY_FUNCTIONS:
                    try:
                        grad_c = _run_geo_gradient_for_function_init(cfg_dot, cname)
                        print(f"[INIT_PICK] IKKT | {cname} via GEOMETRY")
                    except Exception:
                        print(
                            f"[INIT_PICK] IKKT warning | {cname} gradient not available -> skipped"
                        )
                        continue
                else:
                    print(
                        f"[INIT_PICK] IKKT warning | {cname} gradient not available -> skipped"
                    )
                    continue

            grad_c = np.asarray(grad_c, dtype=float)

            if len(grad_c) != len(grad_obj):
                raise RuntimeError(
                    f"{cname} gradient size mismatch: got {len(grad_c)}, expected {len(grad_obj)}"
                )

            constraint_grads_full.append(grad_c)

        residual_full, lam = _compute_ikkt_residual_vector(
            grad_obj,
            constraint_grads_full,
            lambda_bounds=lambda_bounds,
        )

        indicator = np.abs(np.asarray(residual_full, dtype=float))

    else:
        raise ValueError(f"Unsupported indicator mode: {indicator_mode}")

    score_upper = indicator[:n_upper]
    score_lower = indicator[n_upper:]

    return cand_upper, cand_lower, score_upper, score_lower


def main(cfg_file):
    run_baseline_direct(cfg_file)
    run_required_adjoint_set(cfg_file, INDICATOR_MODE)

    cand_upper, cand_lower, score_upper, score_lower = compute_initial_scores(
        cfg_file,
        INDICATOR_MODE,
    )

    selected_upper = select_with_spacing(
        cand_upper, score_upper, N_UPPER, DX_MIN
    )
    selected_lower = select_with_spacing(
        cand_lower, score_lower, N_LOWER, DX_MIN
    )

    print("\n====================================")
    print(" AUTO INITIALIZATION RESULT")
    print("====================================\n")
    print(f"[INIT_PICK] indicator = {INDICATOR_MODE}")
    print(f"[INIT_PICK] ncand     = {NCAND}")
    print(f"[INIT_PICK] dx_min    = {DX_MIN}")
    print()

    print(
        "PROGRESSIVE_HH_INITIAL_UPPER = ("
        + ", ".join(f"{x:.4f}" for x in selected_upper)
        + ")"
    )
    print(
        "PROGRESSIVE_HH_INITIAL_LOWER = ("
        + ", ".join(f"{x:.4f}" for x in selected_lower)
        + ")"
    )

    print(f"\n[INIT_PICK] Temporary run stored in: {TMP_DIR}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python auto_pick_initial_hh.py config.cfg")
        raise SystemExit(1)

    main(sys.argv[1])
