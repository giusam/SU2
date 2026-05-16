#!/usr/bin/env python

import os
import copy
import glob
import shutil
import contextlib

import numpy as np
import SU2
from scipy.optimize import lsq_linear


def get_midpoint_candidates(centers, nsamples=1):
    centers = sorted(list(centers))
    if not centers:
        return []

    extended = [0.0] + centers + [1.0]
    candidates = []

    X_MAX = 0.97
    nsamples = int(nsamples)
    if nsamples < 1:
        raise ValueError("candidate sample count must be >= 1")

    for i in range(len(extended) - 1):
        x_left = float(extended[i])
        x_right = float(extended[i + 1])

        for j in range(1, nsamples + 1):
            frac = float(j) / float(nsamples + 1)
            x = x_left + frac * (x_right - x_left)

            if 0.0 < x < X_MAX:
                candidates.append(
                    {
                        "x": x,
                        "interval_id": i,
                        "interval_left": x_left,
                        "interval_right": x_right,
                        "sample_index": j,
                        "sample_fraction": frac,
                    }
                )

    return candidates


def _reduce_candidates_to_interval_best(candidates):
    best_by_interval = {}

    for c in candidates:
        key = (c["side"], c["interval_id"])
        current = best_by_interval.get(key)
        candidate_key = (
            float(c["indicator"]),
            -float(c["x"]),
        )

        if current is None:
            best_by_interval[key] = c
            continue

        current_key = (
            float(current["indicator"]),
            -float(current["x"]),
        )
        if candidate_key > current_key:
            best_by_interval[key] = c

    reduced = sorted(
        best_by_interval.values(),
        key=lambda c: (str(c["side"]), int(c["interval_id"]), float(c["x"])),
    )

    for c in reduced:
        print(
            "[PROGRESSIVE_HH] Candidate interval best | "
            f"side={c['side']} "
            f"interval=[{c['interval_left']:.6f},{c['interval_right']:.6f}] "
            f"selected_x={c['x']:.6f} I={c['indicator']:.6e}"
        )

    return reduced


def _find_real_adjoint_assets(level_dir, func_name):
    func_name = str(func_name).upper()

    adjoint_folder_name = f"ADJOINT_{func_name}"

    candidates = sorted(
        glob.glob(os.path.join(level_dir, "DESIGNS", "DSN_*", adjoint_folder_name))
    )
    if not candidates:
        raise FileNotFoundError(
            f"No {adjoint_folder_name} found in {os.path.join(level_dir, 'DESIGNS')}"
        )

    adjoint_dir = candidates[-1]
    design_dir = os.path.dirname(adjoint_dir)

    return adjoint_dir, design_dir


def _find_latest_design_with_geometry(level_dir, func_name=None):
    """
    Return the most recent DSN_* directory that contains usable geometry data.
    """
    designs = sorted(glob.glob(os.path.join(level_dir, "DESIGNS", "DSN_*")))
    if not designs:
        raise FileNotFoundError(
            f"No DSN_* folders found in {os.path.join(level_dir, 'DESIGNS')}"
        )

    func_name = None if func_name is None else str(func_name).upper()

    for dsn_dir in reversed(designs):
        geo_dirs = [
            os.path.join(dsn_dir, "GEOMETRY"),
            os.path.join(dsn_dir, "geometry"),
        ]

        has_geometry_dir = any(os.path.isdir(g) for g in geo_dirs)

        if func_name is None:
            if has_geometry_dir:
                return dsn_dir
            continue

        matched_files = []
        for pattern in [
            f"*{func_name}*",
            f"*{func_name.lower()}*",
            "*of_grad*",
            "*grad*",
            "*history*",
            "*values*",
        ]:
            matched_files.extend(glob.glob(os.path.join(dsn_dir, pattern)))
            for gdir in geo_dirs:
                matched_files.extend(glob.glob(os.path.join(gdir, pattern)))

        if has_geometry_dir or matched_files:
            return dsn_dir

    raise FileNotFoundError(
        f"No DSN_* folder with usable geometry data found in {os.path.join(level_dir, 'DESIGNS')}"
    )


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


def _extract_constraint_signs(cfg, constraint_names):
    """
    Build lambda bounds consistent with the paper:

      - maximum-bound inequality  -> lambda >= 0
      - minimum-bound inequality  -> lambda <= 0
      - equality constraints      -> lambda free

    Assumed SU2 convention:
      SIGN "<"  -> upper / maximum bound
      SIGN ">"  -> lower / minimum bound
      SIGN "="  -> equality
    """
    lb = []
    ub = []

    opt_con = cfg.get("OPT_CONSTRAINT", {})

    # -------------------------------------------------
    # Case 1: structured dict
    # -------------------------------------------------
    if isinstance(opt_con, dict):
        equality = opt_con.get("EQUALITY", {}) or {}
        inequality = opt_con.get("INEQUALITY", {}) or {}

        for cname in constraint_names:
            cname = str(cname).upper()

            if cname in equality:
                lb.append(-np.inf)
                ub.append(np.inf)
                continue

            entry = inequality.get(cname, None)

            if not isinstance(entry, dict):
                lb.append(-np.inf)
                ub.append(np.inf)
                continue

            sign = str(entry.get("SIGN", "")).strip()

            if sign == "<":
                lb.append(0.0)
                ub.append(np.inf)
            elif sign == ">":
                lb.append(-np.inf)
                ub.append(0.0)
            elif sign == "=":
                lb.append(-np.inf)
                ub.append(np.inf)
            else:
                lb.append(-np.inf)
                ub.append(np.inf)

        return np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)

    # -------------------------------------------------
    # Case 2: raw string, e.g.
    # ( MOMENT_Z < 0.092 )*0.01; (AIRFOIL_AREA>0.0778)*0.01; (LIFT=0.824)*0.01
    # -------------------------------------------------
    sign_map = {}

    raw = str(opt_con).strip()
    if raw:
        parts = [p.strip() for p in raw.split(";") if p.strip()]

        for part in parts:
            # take content inside first parentheses if present
            if "(" in part and ")" in part:
                inside = part[part.find("(") + 1 : part.find(")")]
            else:
                inside = part

            inside = inside.strip()

            # detect sign
            sign = None
            if "<" in inside:
                sign = "<"
                lhs = inside.split("<", 1)[0].strip()
            elif ">" in inside:
                sign = ">"
                lhs = inside.split(">", 1)[0].strip()
            elif "=" in inside:
                sign = "="
                lhs = inside.split("=", 1)[0].strip()
            else:
                continue

            cname = lhs.replace(",", " ").split()[0].strip().upper()
            if cname:
                sign_map[cname] = sign

    for cname in constraint_names:
        cname = str(cname).upper()
        sign = sign_map.get(cname, None)

        if sign == "<":
            lb.append(0.0)
            ub.append(np.inf)
        elif sign == ">":
            lb.append(-np.inf)
            ub.append(0.0)
        elif sign == "=":
            lb.append(-np.inf)
            ub.append(np.inf)
        else:
            lb.append(-np.inf)
            ub.append(np.inf)

    return np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)


def _run_dot_for_function(level_dir, cfg_dot, state, func_name):
    func_name = str(func_name).upper()

    adjoint_dir, design_dir = _find_real_adjoint_assets(level_dir, func_name)

    dot_test_dir = os.path.join(level_dir, f"DOT_ONLY_{func_name}")

    if os.path.isdir(dot_test_dir):
        shutil.rmtree(dot_test_dir)

    shutil.copytree(adjoint_dir, dot_test_dir, symlinks=False)

    restart_candidates = glob.glob(os.path.join(design_dir, "solution_adj_*.dat"))
    if not restart_candidates:
        raise FileNotFoundError(f"No adjoint restart files found in {design_dir}")

    for restart_src in restart_candidates:
        restart_dst = os.path.join(dot_test_dir, os.path.basename(restart_src))
        shutil.copy2(restart_src, restart_dst)

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

    geo_src_design = _find_latest_design_with_geometry(level_dir, func_name)
    print(f"[PROGRESSIVE_HH] GEOMETRY source design for {func_name}: {geo_src_design}")

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

    if not os.path.exists(mesh_src):
        alt_meshes = glob.glob(os.path.join(geo_src_design, "*.su2"))
        if not alt_meshes:
            raise FileNotFoundError(
                f"Missing mesh for geometry run: {mesh_src}, and no fallback mesh in {geo_src_design}"
            )
        mesh_src = sorted(alt_meshes)[-1]

    mesh_dst = os.path.join(geo_test_dir, os.path.basename(mesh_src))

    if os.path.abspath(mesh_src) != os.path.abspath(mesh_dst):
        shutil.copy2(mesh_src, mesh_dst)

    cfg_geo["MESH_FILENAME"] = os.path.basename(mesh_dst)

    cwd = os.getcwd()
    try:
        os.chdir(geo_test_dir)
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                info = SU2.run.geometry(cfg_geo)
    finally:
        os.chdir(cwd)

    return info["GRADIENTS"][func_name]


def _compute_ikkt_residual_vector(g_obj, constraint_grads, lambda_bounds=None):
    g = np.asarray(g_obj, dtype=float)

    if not constraint_grads:
        return g.copy(), np.zeros(0)

    A = np.column_stack([np.asarray(cg, dtype=float) for cg in constraint_grads])

    try:
        if lambda_bounds is None:
            lb = np.full(A.shape[1], -np.inf)
            ub = np.full(A.shape[1], np.inf)
            res = lsq_linear(A, g, bounds=(lb, ub), lsmr_tol="auto")
        else:
            lb, ub = lambda_bounds
            res = lsq_linear(A, g, bounds=(lb, ub), lsmr_tol="auto")

        lam = res.x
        residual = g - A @ lam

        # =========================
        # IKKT DEBUG BLOCK
        # =========================
        g_norm = np.linalg.norm(g)
        r_norm = np.linalg.norm(residual)
        rel_res = r_norm / max(g_norm, 1e-16)

        print("\n[IKKT DEBUG]")
        print(f"||g||           = {g_norm:.6e}")
        print(f"||r||           = {r_norm:.6e}")
        print(f"relative resid  = {rel_res:.6e}")
        print(f"lambdas         = {lam.tolist()}")
        print(f"lambda lower    = {lb.tolist()}")
        print(f"lambda upper    = {ub.tolist()}")

        for j, cg in enumerate(constraint_grads):
            contrib = abs(lam[j]) * np.linalg.norm(cg)
            print(f"||lambda[{j}] * gradC[{j}]|| = {contrib:.6e}")

        g_reconstructed = A @ lam
        print(f"||A lambda||    = {np.linalg.norm(g_reconstructed):.6e}")
        # =========================

    except Exception:
        residual = g.copy()
        lam = np.zeros(A.shape[1])

    return residual, lam


def _compute_dot_candidate_scores(level, opts):
    cfg_path = os.path.join(level.workdir, level.config_filename)
    cfg_level = SU2.io.Config(cfg_path)

    active_upper = list(level.upper)
    active_lower = list(level.lower)
    nsamples = int(opts.get("candidate_samples", 1))

    cand_upper_raw = get_midpoint_candidates(active_upper, nsamples=nsamples)
    cand_lower_raw = get_midpoint_candidates(active_lower, nsamples=nsamples)
    for c in cand_upper_raw:
        c["side"] = "UPPER"
    for c in cand_lower_raw:
        c["side"] = "LOWER"

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

    n_upper_active = len(active_upper)
    n_lower_active = len(active_lower)
    n_upper_candidate = len(cand_upper)
    n_lower_candidate = len(cand_lower)
    n_active = n_upper_active + n_lower_active

    i_upper_active_0 = 0
    i_upper_candidate_0 = i_upper_active_0 + n_upper_active
    i_lower_active_0 = i_upper_candidate_0 + n_upper_candidate
    i_lower_candidate_0 = i_lower_active_0 + n_lower_active

    grad_upper_active = grad_obj[i_upper_active_0:i_upper_candidate_0]
    grad_upper_candidate = grad_obj[i_upper_candidate_0:i_lower_active_0]
    grad_lower_active = grad_obj[i_lower_active_0:i_lower_candidate_0]
    grad_lower_candidate = grad_obj[i_lower_candidate_0:]

    grad_active = list(grad_upper_active) + list(grad_lower_active)
    grad_candidate = list(grad_upper_candidate) + list(grad_lower_candidate)

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
        lambda_bounds = _extract_constraint_signs(cfg_level, constraint_names)
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

        residual_full, lam = _compute_ikkt_residual_vector(
            grad_obj,
            constraint_grads_full,
            lambda_bounds=lambda_bounds,
        )

        residual_upper_active = residual_full[i_upper_active_0:i_upper_candidate_0]
        residual_upper_candidate = residual_full[i_upper_candidate_0:i_lower_active_0]
        residual_lower_active = residual_full[i_lower_active_0:i_lower_candidate_0]
        residual_lower_candidate = residual_full[i_lower_candidate_0:]

        active_indicator = np.abs(
            np.asarray(
                list(residual_upper_active) + list(residual_lower_active),
                dtype=float,
            )
        ).tolist()
        candidate_indicator = np.abs(
            np.asarray(
                list(residual_upper_candidate) + list(residual_lower_candidate),
                dtype=float,
            )
        ).tolist()
    else:
        active_indicator = np.abs(np.asarray(grad_active, dtype=float)).tolist()
        candidate_indicator = np.abs(np.asarray(grad_candidate, dtype=float)).tolist()

    active_upper_scores = active_indicator[:n_upper_active]
    active_lower_scores = active_indicator[n_upper_active:]
    candidate_upper_scores = candidate_indicator[:n_upper_candidate]
    candidate_lower_scores = candidate_indicator[n_upper_candidate:]

    candidates = []

    for k, c in enumerate(cand_upper_raw):
        candidates.append(
            {
                "side": c["side"],
                "x": float(c["x"]),
                "grad": float(grad_upper_candidate[k]),
                "indicator": float(candidate_upper_scores[k]),
                "interval_id": c["interval_id"],
                "interval_left": float(c["interval_left"]),
                "interval_right": float(c["interval_right"]),
                "sample_index": int(c["sample_index"]),
                "sample_fraction": float(c["sample_fraction"]),
            }
        )

    for k, c in enumerate(cand_lower_raw):
        candidates.append(
            {
                "side": c["side"],
                "x": float(c["x"]),
                "grad": float(grad_lower_candidate[k]),
                "indicator": float(candidate_lower_scores[k]),
                "interval_id": c["interval_id"],
                "interval_left": float(c["interval_left"]),
                "interval_right": float(c["interval_right"]),
                "sample_index": int(c["sample_index"]),
                "sample_fraction": float(c["sample_fraction"]),
            }
        )

    print(
        f"[PROGRESSIVE_HH] Candidate scoring | mode={indicator_mode} "
        f"active_ndv={n_active} candidate_ndv={expected_ncand} samples={nsamples}"
    )
    for c in candidates:
        print(
            "[PROGRESSIVE_HH] Candidate | "
            f"side={c['side']} interval_id={c['interval_id']} "
            f"interval=[{c['interval_left']:.6f},{c['interval_right']:.6f}] "
            f"sample_index={c['sample_index']} "
            f"sample_fraction={c['sample_fraction']:.6f} "
            f"x={c['x']:.6f} I={c['indicator']:.6e}"
        )

    reduced_candidates = _reduce_candidates_to_interval_best(candidates)

    return {
        "candidates": reduced_candidates,
        "raw_candidates": candidates,
        "active_upper_scores": active_upper_scores,
        "active_lower_scores": active_lower_scores,
    }
