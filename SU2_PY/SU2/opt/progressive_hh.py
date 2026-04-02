#!/usr/bin/env python

import os
import math
import copy
import csv
import glob
import shutil
import contextlib

import numpy as np
import SU2


class HHLevel:
    def __init__(
        self,
        level_id,
        upper,
        lower,
        workdir,
        config_filename,
        project_filename,
        mesh_source=None,
    ):
        self.level_id = level_id
        self.upper = list(upper)
        self.lower = list(lower)
        self.workdir = workdir
        self.config_filename = config_filename
        self.project_filename = project_filename
        self.mesh_source = mesh_source

    @property
    def ndv(self):
        return len(self.upper) + len(self.lower)


def _as_bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().upper() in ("YES", "TRUE", "1", "ON")


def get_progressive_hh_options(config):
    scale = 1.0
    if "DEFINITION_DV" in config:
        def_dv = config["DEFINITION_DV"]
        if "SCALE" in def_dv and def_dv["SCALE"]:
            try:
                scale = float(def_dv["SCALE"][0])
            except Exception:
                pass

    return {
        "enabled": _as_bool(config.get("PROGRESSIVE_HH", "NO")),
        "nlevels": int(config.get("PROGRESSIVE_HH_NLEVELS", 1)),
        "n0": int(config.get("PROGRESSIVE_HH_N0", 3)),
        "surface_mode": str(config.get("PROGRESSIVE_HH_SURFACE", "BOTH")).upper(),
        "trigger": str(config.get("PROGRESSIVE_HH_TRIGGER", "MAX_ITER")).upper(),
        "window": int(config.get("PROGRESSIVE_HH_WINDOW", 1)),
        "tol": float(config.get("PROGRESSIVE_HH_TOL", 0.2)),
        "slope_filter_tol": float(config.get("PROGRESSIVE_HH_SLOPE_FILTER_TOL", 0.02)),
        "stag_tol": float(config.get("PROGRESSIVE_HH_STAG_TOL", 1.0e-3)),
        "stag_band": float(config.get("PROGRESSIVE_HH_STAG_BAND", 0.02)),
        "stag_window": int(config.get("PROGRESSIVE_HH_STAG_WINDOW", 3)),
        "max_iter_per_level": int(
            config.get("PROGRESSIVE_HH_MAX_ITER_PER_LEVEL", config.OPT_ITERATIONS)
        ),
        "refinement": str(config.get("PROGRESSIVE_HH_REFINEMENT", "UNIFORM")).upper(),
        "growth_ratio": float(config.get("PROGRESSIVE_HH_GROWTH_RATIO", 2.0)),
        "adaptive_no_adjacent": _as_bool(
            config.get("PROGRESSIVE_HH_ADAPTIVE_NO_ADJACENT", "NO")
        ),
        "adaptive_indicator": str(
            config.get("PROGRESSIVE_HH_ADAPTIVE_INDICATOR", "ABS_GRAD")
        ).upper(),
        "marker": str(config.get("DV_MARKER", "Airfoil")),
        "scale": scale,
    }


def initial_centers(n0):
    if n0 <= 0:
        return []
    return [(i + 1) / float(n0 + 1) for i in range(n0)]


def refine_uniform(centers):
    if not centers:
        return []

    centers = sorted(centers)
    extended = [0.0] + centers + [1.0]

    new_points = []
    for i in range(len(extended) - 1):
        xm = 0.5 * (extended[i] + extended[i + 1])
        if 0.0 < xm < 1.0:
            new_points.append(xm)

    return sorted(set(centers + new_points))


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
        return []

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
        constraint_grads = []

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

            grad_c_candidate = grad_c[n_active:]

            if len(grad_c_candidate) != expected_ncand:
                raise RuntimeError(
                    f"{cname} candidate gradient size mismatch: "
                    f"got {len(grad_c_candidate)}, expected {expected_ncand}"
                )

            constraint_grads.append(grad_c_candidate)

        residual = _compute_ikkt_residual_vector(grad_candidate, constraint_grads)
        candidate_indicator = np.abs(residual).tolist()
    else:
        candidate_indicator = np.abs(np.asarray(grad_candidate, dtype=float)).tolist()

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

    return candidates


def _compute_adaptive_nadd(current_ndv, ncandidates, growth_ratio):
    if ncandidates <= 0:
        return 0

    growth_ratio = float(growth_ratio)
    if growth_ratio <= 1.0:
        target_ndv = current_ndv + 1
    else:
        target_ndv = int(math.ceil(growth_ratio * current_ndv))

    nadd = max(1, target_ndv - current_ndv)
    nadd = min(nadd, ncandidates)
    return nadd


def _select_top_candidates(candidates, nadd, no_adjacent=False):
    if nadd <= 0 or not candidates:
        return []

    ranked = sorted(candidates, key=lambda c: (-c["indicator"], c["x"]))

    if not no_adjacent:
        return ranked[:nadd]

    selected = []
    used = {"UPPER": set(), "LOWER": set()}

    for c in ranked:
        side = c["side"]
        i = c["interval_id"]

        if i in used[side] or (i - 1) in used[side] or (i + 1) in used[side]:
            continue

        selected.append(c)
        used[side].add(i)

        if len(selected) == nadd:
            return selected

    for c in ranked:
        if c not in selected:
            selected.append(c)
        if len(selected) == nadd:
            break

    return selected


def refine_adaptive(prev_level, result, opts):
    current_ndv = prev_level.ndv

    try:
        candidates = _compute_dot_candidate_scores(prev_level, opts)
    except Exception as err:
        print(
            "[PROGRESSIVE_HH] WARNING: ADAPTIVE refine failed -> fallback to UNIFORM | "
            f"{err}"
        )
        return refine_uniform(prev_level.upper), refine_uniform(prev_level.lower)

    if not candidates:
        print("[PROGRESSIVE_HH] ADAPTIVE refine | no candidates -> fallback to UNIFORM")
        return refine_uniform(prev_level.upper), refine_uniform(prev_level.lower)

    nadd = _compute_adaptive_nadd(
        current_ndv, len(candidates), opts["growth_ratio"]
    )

    chosen = _select_top_candidates(
        candidates, nadd, opts.get("adaptive_no_adjacent", False)
    )

    new_upper = sorted(prev_level.upper)
    new_lower = sorted(prev_level.lower)

    for c in chosen:
        if c["side"] == "UPPER":
            new_upper.append(c["x"])
        else:
            new_lower.append(c["x"])

    print(
        f"[PROGRESSIVE_HH] ADAPTIVE refine | add={nadd} "
        f"no_adj={opts.get('adaptive_no_adjacent')}"
    )

    for c in chosen:
        print(
            "[PROGRESSIVE_HH] ADAPTIVE selected | "
            f"side={c['side']} x={c['x']:.6f} I={c['indicator']:.6e}"
        )

    return sorted(set(new_upper)), sorted(set(new_lower))


def _resolve_from_cfg_dir(base_config, filename):
    if not filename:
        return filename

    filename = str(filename)

    if os.path.isabs(filename):
        return filename

    cfg_filename = getattr(base_config, "_filename", None)
    if cfg_filename:
        cfg_dir = os.path.dirname(os.path.abspath(cfg_filename))
    else:
        cfg_dir = os.getcwd()

    return os.path.abspath(os.path.join(cfg_dir, filename))


def build_initial_level(base_config, opts):
    upper = []
    lower = []

    if opts["surface_mode"] in ("UPPER", "BOTH"):
        upper = initial_centers(opts["n0"])

    if opts["surface_mode"] in ("LOWER", "BOTH"):
        lower = initial_centers(opts["n0"])

    initial_mesh = None
    if "MESH_FILENAME" in base_config and base_config["MESH_FILENAME"]:
        initial_mesh = _resolve_from_cfg_dir(base_config, base_config["MESH_FILENAME"])

    return HHLevel(
        level_id=0,
        upper=upper,
        lower=lower,
        workdir="LEVEL_0",
        config_filename="config_level0.cfg",
        project_filename="project_level0.pkl",
        mesh_source=initial_mesh,
    )


def build_next_level(prev_level, result, opts):
    next_id = prev_level.level_id + 1

    next_mesh = result.get("final_mesh", None)
    if next_mesh is None:
        next_mesh = prev_level.mesh_source

    if opts["refinement"] == "ADAPTIVE":
        upper, lower = refine_adaptive(prev_level, result, opts)
    else:
        upper = refine_uniform(prev_level.upper)
        lower = refine_uniform(prev_level.lower)

    return HHLevel(
        level_id=next_id,
        upper=upper,
        lower=lower,
        workdir=f"LEVEL_{next_id}",
        config_filename=f"config_level{next_id}.cfg",
        project_filename=f"project_level{next_id}.pkl",
        mesh_source=next_mesh,
    )


def make_hh_definition(level, scale, marker_name):
    kinds = []
    scales = []
    markers = []
    ffdtags = []
    params = []
    sizes = []

    for xc in level.upper:
        kinds.append("HICKS_HENNE")
        scales.append(scale)
        markers.append([str(marker_name)])
        ffdtags.append([""])
        params.append([1.0, float(xc)])
        sizes.append(1)

    for xc in level.lower:
        kinds.append("HICKS_HENNE")
        scales.append(scale)
        markers.append([str(marker_name)])
        ffdtags.append([""])
        params.append([0.0, float(xc)])
        sizes.append(1)

    return {
        "KIND": kinds,
        "SCALE": scales,
        "MARKER": markers,
        "FFDTAG": ffdtags,
        "PARAM": params,
        "SIZE": sizes,
    }


def _remove_progressive_keys(cfg):
    progressive_keys = [
        "PROGRESSIVE_HH",
        "PROGRESSIVE_HH_NLEVELS",
        "PROGRESSIVE_HH_N0",
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
        "PROGRESSIVE_HH_ADAPTIVE_NO_ADJACENT",
        "PROGRESSIVE_HH_ADAPTIVE_INDICATOR",
    ]

    for key in progressive_keys:
        if key in cfg:
            del cfg[key]


def _prepare_local_mesh(cfg, level):
    if not level.mesh_source:
        return

    src_mesh = os.path.abspath(level.mesh_source)
    mesh_basename = os.path.basename(src_mesh)
    dst_mesh = os.path.join(level.workdir, mesh_basename)

    os.makedirs(level.workdir, exist_ok=True)

    if src_mesh != dst_mesh:
        shutil.copy2(src_mesh, dst_mesh)

    cfg["MESH_FILENAME"] = mesh_basename

    if "MULTIPOINT_MESH_FILENAME" in cfg and cfg["MULTIPOINT_MESH_FILENAME"]:
        cfg["MULTIPOINT_MESH_FILENAME"] = f"({mesh_basename})"


def write_level_config(base_config, level, opts):
    cfg = SU2.io.Config(copy.deepcopy(dict(base_config)))

    _remove_progressive_keys(cfg)

    if level.level_id == opts["nlevels"] - 1:
        cfg["OPT_ITERATIONS"] = int(base_config["OPT_ITERATIONS"])
    else:
        cfg["OPT_ITERATIONS"] = opts["max_iter_per_level"]

    cfg["DEFINITION_DV"] = make_hh_definition(
        level,
        scale=opts["scale"],
        marker_name=opts["marker"],
    )

    cfg["DV_MARKER"] = str(opts["marker"])
    cfg["DV_KIND"] = "HICKS_HENNE"

    cfg["DV_VALUE_NEW"] = [0.0] * level.ndv
    cfg["DV_VALUE_OLD"] = [0.0] * level.ndv

    _prepare_local_mesh(cfg, level)

    if "RESTART_FILENAME" in cfg and cfg["RESTART_FILENAME"]:
        cfg["RESTART_FILENAME"] = _resolve_from_cfg_dir(
            base_config, cfg["RESTART_FILENAME"]
        )

    mesh_out_base = "mesh_out"
    if "MESH_OUT_FILENAME" in cfg and cfg["MESH_OUT_FILENAME"]:
        mesh_out_base = str(cfg["MESH_OUT_FILENAME"])
        if mesh_out_base.endswith(".su2"):
            mesh_out_base = mesh_out_base[:-4]
        mesh_out_base = os.path.basename(mesh_out_base)

    cfg["MESH_OUT_FILENAME"] = mesh_out_base

    os.makedirs(level.workdir, exist_ok=True)
    out_cfg = os.path.join(level.workdir, level.config_filename)
    cfg.dump(out_cfg)
    return out_cfg


def _find_history_file(level):
    candidates = []
    candidates.extend(glob.glob(os.path.join(level.workdir, "*history*.csv")))
    candidates.extend(glob.glob(os.path.join(level.workdir, "*HISTORY*.csv")))
    return candidates[0] if candidates else None


def _read_history_values(history_file):
    if not history_file or not os.path.isfile(history_file):
        return []

    with open(history_file, "r", newline="") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)

    if not rows:
        return []

    key_candidates = []
    for key in rows[0].keys():
        ku = key.strip().upper()
        if "DRAG" in ku or "OBJECTIVE" in ku or "OBJFUN" in ku:
            key_candidates.append(key)

    if not key_candidates:
        return []

    key = key_candidates[0]
    history = []

    for row in rows:
        try:
            history.append(float(row[key]))
        except Exception:
            pass

    return history


def _find_final_mesh(level):
    design_deform = glob.glob(
        os.path.join(level.workdir, "DESIGNS", "**", "*_deform.su2"),
        recursive=True,
    )
    if design_deform:
        return sorted(design_deform)[-1]

    design_all = glob.glob(
        os.path.join(level.workdir, "DESIGNS", "**", "*.su2"),
        recursive=True,
    )
    if design_all:
        return sorted(design_all)[-1]

    root_deform = glob.glob(os.path.join(level.workdir, "*_deform.su2"))
    if root_deform:
        return sorted(root_deform)[-1]

    root_all = glob.glob(os.path.join(level.workdir, "*.su2"))
    if root_all:
        return sorted(root_all)[-1]

    return None


def collect_level_result(level):
    history_file = _find_history_file(level)
    history = _read_history_values(history_file)
    final_mesh = _find_final_mesh(level)

    return {
        "history": history,
        "history_file": history_file,
        "final_mesh": final_mesh,
    }


def should_refine(history, opts, level_id):
    if level_id >= opts["nlevels"] - 1:
        return False

    if opts["trigger"] == "MAX_ITER":
        return True

    return False