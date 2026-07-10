#!/usr/bin/env python

import copy
import glob
import os
import shutil
import contextlib

import numpy as np
import SU2

from SU2.opt.progressive_ffd_core import (
    build_ffd_mesh_columns,
    ffd_active_range_from_opts,
    make_ffd_config_dump_compatible,
    make_dual_ffd_definition,
    make_ffd_definition,
    ordered_dual_ffd_records,
    validate_ffd_mesh_blending,
)
from SU2.opt.progressive_ffd_mesh import rewrite_ffd_box_with_columns_and_reembed
from SU2.opt.progressive_ffd_split import (
    rewrite_dual_ffd_boxes_with_columns_and_reembed,
)
from SU2.opt.progressive_hh_projection import (
    _compute_ikkt_residual_vector,
    _dot_problem_kind,
    _extract_constraint_names,
    _extract_constraint_signs,
    _find_latest_design_with_geometry,
    _find_real_adjoint_assets,
    _make_projection_state,
    _run_geo_gradient_for_function,
    _select_dot_config_path,
)


def _flatten_indicator_numeric_values(value):
    if value is None:
        return []
    if isinstance(value, str):
        raw = value.strip().strip("()[]").replace(",", " ")
        values = []
        for token in raw.split():
            try:
                values.append(float(token))
            except Exception:
                pass
        return values
    if isinstance(value, (list, tuple)):
        values = []
        for item in value:
            values.extend(_flatten_indicator_numeric_values(item))
        return values
    try:
        return [float(value)]
    except Exception:
        return []


def _scalar_config_value(cfg, key, default):
    values = _flatten_indicator_numeric_values(cfg.get(key, default))
    if not values:
        return float(default)
    return float(values[0])


def _gradient_descent_indicator(values, cfg):
    g = np.asarray(values, dtype=float)
    lower = _scalar_config_value(cfg, "OPT_BOUND_LOWER", -np.inf)
    upper = _scalar_config_value(cfg, "OPT_BOUND_UPPER", np.inf)

    tol = 1.0e-14
    if lower < -tol and upper > tol:
        rule = "TWO_SIDED"
        indicator = np.abs(g)
    elif lower >= -tol and upper > tol:
        rule = "POSITIVE_ONLY"
        indicator = np.maximum(0.0, -g)
    elif upper <= tol and lower < -tol:
        rule = "NEGATIVE_ONLY"
        indicator = np.maximum(0.0, g)
    else:
        rule = "FIXED_OR_INVALID"
        indicator = np.zeros_like(g)

    print(
        "[PROGRESSIVE_FFD] DESCENT_GRAD indicator | "
        f"bounds=[{lower:.6e},{upper:.6e}] rule={rule}"
    )
    return indicator.tolist()


def _indicator_from_gradient(values, cfg, indicator_mode):
    mode = str(indicator_mode or "ABS_GRAD").upper()
    if mode in (
        "DESCENT_GRAD",
        "BOUNDED_GRAD",
        "ABS_GRAD_DESCENT",
        "ONE_SIDED_ABS_GRAD",
    ):
        return _gradient_descent_indicator(values, cfg)
    return np.abs(np.asarray(values, dtype=float)).tolist()


def get_ffd_interval_candidates(active_columns, xmin, xmax, nsamples):
    active_columns = sorted(float(x) for x in active_columns)
    xmin = float(xmin)
    xmax = float(xmax)
    if not xmin < xmax:
        raise ValueError("FFD candidate range must satisfy xmin < xmax")

    nsamples = int(nsamples)
    if nsamples < 1:
        raise ValueError("candidate sample count must be >= 1")

    extended = []
    for value in [xmin] + active_columns + [xmax]:
        value = float(value)
        if not extended or abs(value - extended[-1]) > 1.0e-10:
            extended.append(value)

    candidates = []
    for i in range(len(extended) - 1):
        x_left = float(extended[i])
        x_right = float(extended[i + 1])
        if x_right <= x_left:
            continue

        for j in range(1, nsamples + 1):
            frac = float(j) / float(nsamples + 1)
            x = x_left + frac * (x_right - x_left)

            if xmin < x < xmax:
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


def _check_ffd_min_spacing(
    candidate,
    active_columns,
    accepted_candidates,
    min_spacing,
    xmin,
    xmax,
):
    min_spacing = float(min_spacing)
    if min_spacing <= 0.0:
        return True, None, None

    x = float(candidate["x"])
    reference = [float(xmin), float(xmax)]
    reference.extend(float(v) for v in active_columns)
    reference.extend(float(v["x"]) for v in accepted_candidates)

    nearest = None
    nearest_distance = None
    for value in reference:
        dist = abs(x - value)
        if nearest_distance is None or dist < nearest_distance:
            nearest = value
            nearest_distance = dist

    if nearest_distance is not None and nearest_distance < min_spacing:
        return False, nearest, nearest_distance
    return True, nearest, nearest_distance


def _filter_candidates_by_min_spacing(
    candidates,
    active_columns,
    min_spacing,
    xmin,
    xmax,
    side="FFD",
):
    if float(min_spacing) <= 0.0:
        return candidates

    filtered = []
    for candidate in candidates:
        accepted, nearest, nearest_distance = _check_ffd_min_spacing(
            candidate,
            active_columns,
            filtered,
            min_spacing,
            xmin,
            xmax,
        )
        if accepted:
            filtered.append(candidate)
            continue

        candidate["rejected_reason"] = "MIN_CENTER_SPACING"
        candidate["nearest_center_or_boundary"] = nearest
        candidate["nearest_distance"] = nearest_distance
        candidate["required_spacing"] = min_spacing
        print(
            "[PROGRESSIVE_FFD] Candidate rejected by min spacing | "
            f"side={side} x={float(candidate['x']):.6f} "
            f"nearest={float(nearest):.6f} dist={float(nearest_distance):.6f} "
            f"required={float(min_spacing):.6f}"
        )

    return filtered


def _reduce_candidates_to_interval_best(candidates):
    best_by_interval = {}
    for candidate in candidates:
        key = (candidate["side"], candidate["interval_id"])
        current = best_by_interval.get(key)
        candidate_key = (float(candidate["indicator"]), -float(candidate["x"]))
        if current is None:
            best_by_interval[key] = candidate
            continue
        current_key = (float(current["indicator"]), -float(current["x"]))
        if candidate_key > current_key:
            best_by_interval[key] = candidate

    reduced = sorted(
        best_by_interval.values(),
        key=lambda c: (str(c["side"]), int(c["interval_id"]), float(c["x"])),
    )
    for candidate in reduced:
        print(
            "[PROGRESSIVE_FFD] Candidate interval best | "
            f"side={candidate['side']} "
            f"interval=[{candidate['interval_left']:.6f},"
            f"{candidate['interval_right']:.6f}] "
            f"selected_x={candidate['x']:.6f} "
            f"I={candidate['indicator']:.6e}"
        )
    return reduced


def _find_projection_mesh_source(mesh_name, adjoint_dir, design_dir, level_dir):
    mesh_name = str(mesh_name)
    if os.path.isabs(mesh_name):
        if os.path.exists(mesh_name):
            return mesh_name
        raise FileNotFoundError(f"MESH_FILENAME absolute path not found: {mesh_name}")

    candidates = [
        os.path.join(adjoint_dir, mesh_name),
        os.path.join(design_dir, mesh_name),
        os.path.join(level_dir, mesh_name),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    basename = os.path.basename(mesh_name)
    for root in (design_dir, adjoint_dir, level_dir):
        matches = glob.glob(os.path.join(root, "**", basename), recursive=True)
        matches = [m for m in matches if os.path.exists(m)]
        if matches:
            return matches[0]

    raise FileNotFoundError(
        "Could not locate mesh for FFD DOT projection. "
        f"MESH_FILENAME={mesh_name}, searched: {candidates}"
    )


def _build_extended_ffd_dot_config(
    cfg_level,
    real_dot_cfg,
    mesh_name,
    ordered_columns,
    opts,
    column_index_by_x,
):
    cfg_dot = SU2.io.Config(copy.deepcopy(dict(real_dot_cfg)))

    if "NUMBER_PART" in real_dot_cfg:
        cfg_dot["NUMBER_PART"] = int(real_dot_cfg["NUMBER_PART"])
    elif "NUMBER_PART" not in cfg_dot:
        cfg_dot["NUMBER_PART"] = int(cfg_level.get("NUMBER_PART", 1))

    if "NZONES" in real_dot_cfg:
        cfg_dot["NZONES"] = int(real_dot_cfg["NZONES"])
    elif "NZONES" not in cfg_dot:
        cfg_dot["NZONES"] = int(cfg_level.get("NZONES", 1))

    dot_kind = _dot_problem_kind(real_dot_cfg)
    cfg_dot["MATH_PROBLEM"] = dot_kind
    cfg_dot["GRADIENT_METHOD"] = dot_kind

    if "RESTART_SOL" not in cfg_dot:
        cfg_dot["RESTART_SOL"] = "NO"
    cfg_dot["CONSOLE"] = "NONE"
    cfg_dot["MESH_FILENAME"] = mesh_name
    if "MULTIPOINT_MESH_FILENAME" in cfg_dot and cfg_dot["MULTIPOINT_MESH_FILENAME"]:
        cfg_dot["MULTIPOINT_MESH_FILENAME"] = f"({mesh_name})"

    if opts.get("ffd_dual_box", False):
        cfg_dot["DEFINITION_DV"] = make_dual_ffd_definition(
            ordered_columns,
            opts,
            column_index_by_x,
        )
        cfg_dot["FFD_CONTINUITY"] = "USER_INPUT"
        for key in ("FFD_FIX_I", "FFD_FIX_J", "FFD_FIX_K"):
            if key in cfg_dot:
                del cfg_dot[key]
    else:
        cfg_dot["DEFINITION_DV"] = make_ffd_definition(
            ordered_columns,
            opts,
            column_index_by_x,
        )
    cfg_dot["DV_MARKER"] = str(opts["ffd_marker"])
    cfg_dot["DV_KIND"] = str(opts["ffd_dv_kind"])
    cfg_dot["FFD_BLENDING"] = str(opts.get("ffd_blending", "BEZIER"))
    cfg_dot["FFD_BSPLINE_ORDER"] = ", ".join(
        str(int(value)) for value in opts.get("ffd_bspline_orders", (2, 2, 2))
    )
    cfg_dot["DV_VALUE_NEW"] = [0.0] * len(ordered_columns)
    cfg_dot["DV_VALUE_OLD"] = [0.0] * len(ordered_columns)

    return cfg_dot


def _ensure_dot_mesh_available(dot_test_dir, adjoint_dir, design_dir, level_dir, cfg_dot):
    mesh_name = str(cfg_dot.get("MESH_FILENAME", "")).strip()
    if not mesh_name:
        return

    if os.path.isabs(mesh_name):
        if not os.path.exists(mesh_name):
            raise FileNotFoundError(f"MESH_FILENAME absolute path not found: {mesh_name}")
        return

    mesh_dst = os.path.join(dot_test_dir, mesh_name)
    if os.path.exists(mesh_dst) and not os.path.islink(mesh_dst):
        return
    if os.path.lexists(mesh_dst):
        os.remove(mesh_dst)

    mesh_src = _find_projection_mesh_source(mesh_name, adjoint_dir, design_dir, level_dir)
    os.makedirs(os.path.dirname(mesh_dst), exist_ok=True)
    shutil.copy2(mesh_src, mesh_dst, follow_symlinks=True)
    print(
        "[PROGRESSIVE_FFD] DOT projection | copied mesh into DOT_ONLY: "
        f"{mesh_src} -> {mesh_dst}"
    )


def _run_ffd_dot_for_function(level_dir, cfg_dot, state, func_name):
    func_name = str(func_name).upper()
    dot_kind = _dot_problem_kind(cfg_dot)
    adjoint_dir, design_dir = _find_real_adjoint_assets(level_dir, func_name)
    dot_test_dir = os.path.join(level_dir, f"DOT_ONLY_{func_name}")

    if os.path.isdir(dot_test_dir):
        shutil.rmtree(dot_test_dir)
    shutil.copytree(adjoint_dir, dot_test_dir, symlinks=False)

    _ensure_dot_mesh_available(dot_test_dir, adjoint_dir, design_dir, level_dir, cfg_dot)

    restart_candidates = glob.glob(os.path.join(design_dir, "solution_adj_*.dat"))
    if restart_candidates:
        for restart_src in restart_candidates:
            restart_dst = os.path.join(dot_test_dir, os.path.basename(restart_src))
            shutil.copy2(restart_src, restart_dst)
    elif dot_kind == "DISCRETE_ADJOINT":
        raise FileNotFoundError(f"No adjoint restart files found in {design_dir}")
    else:
        print(
            "[PROGRESSIVE_FFD] DOT projection | "
            "continuous adjoint: no solution_adj_*.dat restart required"
        )

    cfg_fun = SU2.io.Config(copy.deepcopy(dict(cfg_dot)))
    cfg_fun["OBJECTIVE_FUNCTION"] = func_name
    cfg_fun_for_su2 = make_ffd_config_dump_compatible(cfg_fun)

    cwd = os.getcwd()
    try:
        os.chdir(dot_test_dir)
        try:
            cfg_fun_for_su2.dump("config_DOT_PROGRESSIVE_FFD.cfg")
        except Exception:
            pass
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                info = SU2.run.projection(cfg_fun_for_su2, state)
    finally:
        os.chdir(cwd)

    gradients = info.get("GRADIENTS", {})
    if func_name in gradients:
        grad = gradients[func_name]
    elif func_name.lower() in gradients:
        grad = gradients[func_name.lower()]
    else:
        raise KeyError(
            f"SU2_DOT did not return gradients for {func_name}. "
            f"Available keys: {list(gradients.keys())}"
        )

    grad_arr = np.asarray(grad, dtype=float)
    print(
        "[PROGRESSIVE_FFD] DOT projection | "
        f"kind={dot_kind} function={func_name} "
        f"ndv={grad_arr.size} norm={np.linalg.norm(grad_arr):.6e}"
    )
    return grad


def _run_ffd_geo_gradient_for_function(level_dir, cfg_dot, func_name):
    func_name = str(func_name).upper()
    geo_src_design = _find_latest_design_with_geometry(level_dir, func_name)
    print(f"[PROGRESSIVE_FFD] GEOMETRY source design for {func_name}: {geo_src_design}")

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
                f"Missing mesh for geometry run: {mesh_src}, "
                f"and no fallback mesh in {geo_src_design}"
            )
        mesh_src = sorted(alt_meshes)[-1]

    mesh_dst = os.path.join(geo_test_dir, os.path.basename(mesh_src))
    if os.path.abspath(mesh_src) != os.path.abspath(mesh_dst):
        shutil.copy2(mesh_src, mesh_dst)

    cfg_geo["MESH_FILENAME"] = os.path.basename(mesh_dst)
    cfg_geo = make_ffd_config_dump_compatible(cfg_geo)
    cwd = os.getcwd()
    try:
        os.chdir(geo_test_dir)
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                info = SU2.run.geometry(cfg_geo)
    finally:
        os.chdir(cwd)

    return info["GRADIENTS"][func_name]


def _compute_ffd_dot_candidate_scores(level, opts):
    if getattr(level, "dual_box", False):
        return _compute_dual_ffd_dot_candidate_scores(level, opts)

    cfg_path = os.path.join(level.workdir, level.config_filename)
    cfg_level = SU2.io.Config(cfg_path)

    active_columns = sorted(float(x) for x in level.columns)
    nsamples = int(opts.get("candidate_samples", 1))
    min_spacing = float(opts.get("min_center_spacing", 0.0))
    candidate_xmin, candidate_xmax = ffd_active_range_from_opts(opts)

    print(
        "[PROGRESSIVE_FFD] Candidate sampling | "
        f"samples={nsamples} min_spacing={min_spacing:.6f} "
        f"xrange=[{candidate_xmin:.6f},{candidate_xmax:.6f}]"
    )

    raw_candidates = get_ffd_interval_candidates(
        active_columns,
        xmin=candidate_xmin,
        xmax=candidate_xmax,
        nsamples=nsamples,
    )
    for candidate in raw_candidates:
        candidate["side"] = "FFD"

    raw_candidates = _filter_candidates_by_min_spacing(
        raw_candidates,
        active_columns,
        min_spacing,
        candidate_xmin,
        candidate_xmax,
    )
    candidate_columns = sorted(float(c["x"]) for c in raw_candidates)

    if not candidate_columns:
        if min_spacing > 0.0:
            print(
                "[PROGRESSIVE_FFD] No valid candidates remain after "
                "min-spacing filtering."
            )
        return {
            "candidates": [],
            "spacing_filtered_empty": min_spacing > 0.0,
            "active_scores": [],
        }

    dv_columns = active_columns + candidate_columns

    obj_name = str(cfg_level.get("OBJECTIVE_FUNCTION", "DRAG")).upper()
    obj_adj_dir, design_dir = _find_real_adjoint_assets(level.workdir, obj_name)
    real_dot_cfg_path, dot_kind = _select_dot_config_path(obj_adj_dir, cfg_level)

    print(
        "[PROGRESSIVE_FFD] Using DOT config: "
        f"{real_dot_cfg_path} | kind={dot_kind}"
    )

    real_dot_cfg = SU2.io.Config(real_dot_cfg_path)
    real_mesh_name = str(real_dot_cfg["MESH_FILENAME"])
    mesh_src = _find_projection_mesh_source(
        real_mesh_name,
        obj_adj_dir,
        design_dir,
        level.workdir,
    )
    mesh_columns, _ = build_ffd_mesh_columns(
        mesh_src,
        opts["ffd_box_tag"],
        active_columns=dv_columns,
        opts=opts,
    )

    extended_mesh_basename = f"ffd_projection_level{level.level_id}.su2"
    extended_mesh_path = os.path.join(level.workdir, extended_mesh_basename)
    mesh_info = rewrite_ffd_box_with_columns_and_reembed(
        mesh_src,
        extended_mesh_path,
        box_tag=opts["ffd_box_tag"],
        new_columns=mesh_columns,
        marker_name=opts["ffd_marker"],
        domain_mode=opts["ffd_domain_mode"],
    )
    validate_ffd_mesh_blending(
        mesh_info,
        opts,
        context=f"FFD DOT candidate mesh for level {level.level_id}",
    )

    cfg_dot = _build_extended_ffd_dot_config(
        cfg_level,
        real_dot_cfg,
        extended_mesh_basename,
        dv_columns,
        opts,
        mesh_info["column_index_by_x"],
    )
    state = _make_projection_state(extended_mesh_basename)

    grad_obj = _run_ffd_dot_for_function(level.workdir, cfg_dot, state, obj_name)

    n_active = len(active_columns)
    n_candidate = len(candidate_columns)
    grad_active = list(grad_obj[:n_active])
    grad_candidate = list(grad_obj[n_active:])

    if len(grad_candidate) == 0:
        raise RuntimeError("DOT returned empty FFD candidate gradient")
    if len(grad_candidate) != n_candidate:
        raise RuntimeError(
            "DOT FFD candidate gradient size mismatch: "
            f"got {len(grad_candidate)}, expected {n_candidate}"
        )

    indicator_mode = str(opts.get("adaptive_indicator", "ABS_GRAD")).upper()

    if indicator_mode == "IKKT":
        constraint_names = _extract_constraint_names(cfg_level)
        lambda_bounds = _extract_constraint_signs(cfg_level, constraint_names)
        constraint_grads_full = []

        for cname in constraint_names:
            cname = cname.upper()
            try:
                grad_c = _run_ffd_dot_for_function(level.workdir, cfg_dot, state, cname)
                print(f"[PROGRESSIVE_FFD] IKKT | {cname} via DOT")
            except Exception:
                try:
                    grad_c = _run_ffd_geo_gradient_for_function(
                        level.workdir,
                        cfg_dot,
                        cname,
                    )
                    print(f"[PROGRESSIVE_FFD] IKKT | {cname} via GEOMETRY")
                except Exception:
                    print(
                        "[PROGRESSIVE_FFD] IKKT warning | "
                        f"{cname} gradient not available -> skipped"
                    )
                    continue

            if len(grad_c) != len(grad_obj):
                raise RuntimeError(
                    f"{cname} full gradient size mismatch: "
                    f"got {len(grad_c)}, expected {len(grad_obj)}"
                )
            constraint_grads_full.append(grad_c)

        residual_full, _ = _compute_ikkt_residual_vector(
            grad_obj,
            constraint_grads_full,
            lambda_bounds=lambda_bounds,
        )
        active_indicator = np.abs(np.asarray(residual_full[:n_active], dtype=float)).tolist()
        candidate_indicator = np.abs(
            np.asarray(residual_full[n_active:], dtype=float)
        ).tolist()
    else:
        active_indicator = _indicator_from_gradient(
            grad_active,
            cfg_level,
            indicator_mode,
        )
        candidate_indicator = _indicator_from_gradient(
            grad_candidate,
            cfg_level,
            indicator_mode,
        )

    candidates = []
    for k, raw in enumerate(raw_candidates):
        candidates.append(
            {
                "side": "FFD",
                "x": float(raw["x"]),
                "grad": float(grad_candidate[k]),
                "indicator": float(candidate_indicator[k]),
                "interval_id": raw["interval_id"],
                "interval_left": float(raw["interval_left"]),
                "interval_right": float(raw["interval_right"]),
                "sample_index": int(raw["sample_index"]),
                "sample_fraction": float(raw["sample_fraction"]),
                "rejected_reason": raw.get("rejected_reason", ""),
                "nearest_center_or_boundary": raw.get(
                    "nearest_center_or_boundary", ""
                ),
                "nearest_distance": raw.get("nearest_distance", ""),
                "required_spacing": raw.get("required_spacing", ""),
            }
        )

    print(
        f"[PROGRESSIVE_FFD] Candidate scoring | mode={indicator_mode} "
        f"active_ndv={n_active} candidate_ndv={n_candidate} "
        f"samples={nsamples}"
    )
    for candidate in candidates:
        print(
            "[PROGRESSIVE_FFD] Candidate | "
            f"side={candidate['side']} interval_id={candidate['interval_id']} "
            f"interval=[{candidate['interval_left']:.6f},"
            f"{candidate['interval_right']:.6f}] "
            f"sample_index={candidate['sample_index']} "
            f"sample_fraction={candidate['sample_fraction']:.6f} "
            f"x={candidate['x']:.6f} I={candidate['indicator']:.6e}"
        )

    reduced_candidates = _reduce_candidates_to_interval_best(candidates)

    return {
        "candidates": reduced_candidates,
        "raw_candidates": candidates,
        "active_scores": active_indicator,
    }


def _compute_dual_ffd_dot_candidate_scores(level, opts):
    cfg_path = os.path.join(level.workdir, level.config_filename)
    cfg_level = SU2.io.Config(cfg_path)

    nsamples = int(opts.get("candidate_samples", 1))
    min_spacing = float(opts.get("min_center_spacing", 0.0))
    candidate_xmin, candidate_xmax = ffd_active_range_from_opts(opts)
    active_by_side = {
        "UPPER": sorted(float(x) for x in level.upper_columns),
        "LOWER": sorted(float(x) for x in level.lower_columns),
    }

    raw_candidates = []
    for side in ("UPPER", "LOWER"):
        side_candidates = get_ffd_interval_candidates(
            active_by_side[side],
            xmin=candidate_xmin,
            xmax=candidate_xmax,
            nsamples=nsamples,
        )
        for candidate in side_candidates:
            candidate["side"] = side
        side_candidates = _filter_candidates_by_min_spacing(
            side_candidates,
            active_by_side[side],
            min_spacing,
            candidate_xmin,
            candidate_xmax,
            side=side,
        )
        raw_candidates.extend(side_candidates)

    if not raw_candidates:
        return {
            "candidates": [],
            "spacing_filtered_empty": min_spacing > 0.0,
            "active_scores": [],
            "active_upper_scores": [],
            "active_lower_scores": [],
        }

    active_records = ordered_dual_ffd_records(
        level.upper_columns,
        level.lower_columns,
    )
    candidate_records = [
        (str(candidate["side"]).upper(), float(candidate["x"]))
        for candidate in raw_candidates
    ]
    ordered_records = active_records + candidate_records

    obj_name = str(cfg_level.get("OBJECTIVE_FUNCTION", "DRAG")).upper()
    obj_adj_dir, design_dir = _find_real_adjoint_assets(level.workdir, obj_name)
    real_dot_cfg_path, dot_kind = _select_dot_config_path(obj_adj_dir, cfg_level)
    print(
        "[PROGRESSIVE_FFD_DUAL] Using DOT config: "
        f"{real_dot_cfg_path} | kind={dot_kind}"
    )

    real_dot_cfg = SU2.io.Config(real_dot_cfg_path)
    real_mesh_name = str(real_dot_cfg["MESH_FILENAME"])
    mesh_src = _find_projection_mesh_source(
        real_mesh_name,
        obj_adj_dir,
        design_dir,
        level.workdir,
    )

    candidate_by_side = {"UPPER": [], "LOWER": []}
    for side, x in candidate_records:
        candidate_by_side[side].append(float(x))
    upper_extended_active = sorted(
        set(active_by_side["UPPER"] + candidate_by_side["UPPER"])
    )
    lower_extended_active = sorted(
        set(active_by_side["LOWER"] + candidate_by_side["LOWER"])
    )
    upper_mesh_columns, _ = build_ffd_mesh_columns(
        mesh_src,
        opts["ffd_upper_box_tag"],
        active_columns=upper_extended_active,
        opts=opts,
    )
    lower_mesh_columns, _ = build_ffd_mesh_columns(
        mesh_src,
        opts["ffd_lower_box_tag"],
        active_columns=lower_extended_active,
        opts=opts,
    )

    extended_mesh_basename = f"ffd_projection_level{level.level_id}.su2"
    extended_mesh_path = os.path.join(level.workdir, extended_mesh_basename)
    mesh_info = rewrite_dual_ffd_boxes_with_columns_and_reembed(
        mesh_src,
        extended_mesh_path,
        marker=opts["ffd_marker"],
        upper_tag=opts["ffd_upper_box_tag"],
        lower_tag=opts["ffd_lower_box_tag"],
        upper_columns=upper_mesh_columns,
        lower_columns=lower_mesh_columns,
        upper_offset_chord=opts["ffd_upper_offset_chord"],
        lower_offset_chord=opts["ffd_lower_offset_chord"],
        diagnostics_csv=False,
        overwrite=True,
    )
    validate_ffd_mesh_blending(
        mesh_info,
        opts,
        context=f"Dual FFD DOT candidate mesh for level {level.level_id}",
    )
    column_index_by_side = {
        "UPPER": mesh_info["upper_column_index_by_x"],
        "LOWER": mesh_info["lower_column_index_by_x"],
    }
    cfg_dot = _build_extended_ffd_dot_config(
        cfg_level,
        real_dot_cfg,
        extended_mesh_basename,
        ordered_records,
        opts,
        column_index_by_side,
    )
    state = _make_projection_state(extended_mesh_basename)
    grad_obj = _run_ffd_dot_for_function(
        level.workdir,
        cfg_dot,
        state,
        obj_name,
    )

    n_active = len(active_records)
    n_candidate = len(candidate_records)
    if len(grad_obj) != n_active + n_candidate:
        raise RuntimeError(
            "DOT dual FFD gradient size mismatch: "
            f"got {len(grad_obj)}, expected {n_active + n_candidate}"
        )
    grad_active = list(grad_obj[:n_active])
    grad_candidate = list(grad_obj[n_active:])
    indicator_mode = str(opts.get("adaptive_indicator", "ABS_GRAD")).upper()

    if indicator_mode == "IKKT":
        constraint_names = _extract_constraint_names(cfg_level)
        lambda_bounds = _extract_constraint_signs(cfg_level, constraint_names)
        constraint_grads_full = []
        for constraint_name in constraint_names:
            constraint_name = constraint_name.upper()
            try:
                grad_constraint = _run_ffd_dot_for_function(
                    level.workdir,
                    cfg_dot,
                    state,
                    constraint_name,
                )
                print(
                    f"[PROGRESSIVE_FFD_DUAL] IKKT | {constraint_name} via DOT"
                )
            except Exception:
                try:
                    grad_constraint = _run_ffd_geo_gradient_for_function(
                        level.workdir,
                        cfg_dot,
                        constraint_name,
                    )
                    print(
                        "[PROGRESSIVE_FFD_DUAL] IKKT | "
                        f"{constraint_name} via GEOMETRY"
                    )
                except Exception:
                    print(
                        "[PROGRESSIVE_FFD_DUAL] IKKT warning | "
                        f"{constraint_name} gradient unavailable -> skipped"
                    )
                    continue
            if len(grad_constraint) != len(grad_obj):
                raise RuntimeError(
                    f"{constraint_name} full gradient size mismatch: "
                    f"got {len(grad_constraint)}, expected {len(grad_obj)}"
                )
            constraint_grads_full.append(grad_constraint)

        residual_full, _ = _compute_ikkt_residual_vector(
            grad_obj,
            constraint_grads_full,
            lambda_bounds=lambda_bounds,
        )
        active_indicator = np.abs(
            np.asarray(residual_full[:n_active], dtype=float)
        ).tolist()
        candidate_indicator = np.abs(
            np.asarray(residual_full[n_active:], dtype=float)
        ).tolist()
    else:
        active_indicator = _indicator_from_gradient(
            grad_active,
            cfg_level,
            indicator_mode,
        )
        candidate_indicator = _indicator_from_gradient(
            grad_candidate,
            cfg_level,
            indicator_mode,
        )

    candidates = []
    for index, raw in enumerate(raw_candidates):
        candidates.append(
            {
                "side": str(raw["side"]).upper(),
                "x": float(raw["x"]),
                "grad": float(grad_candidate[index]),
                "indicator": float(candidate_indicator[index]),
                "interval_id": raw["interval_id"],
                "interval_left": float(raw["interval_left"]),
                "interval_right": float(raw["interval_right"]),
                "sample_index": int(raw["sample_index"]),
                "sample_fraction": float(raw["sample_fraction"]),
                "rejected_reason": raw.get("rejected_reason", ""),
                "nearest_center_or_boundary": raw.get(
                    "nearest_center_or_boundary", ""
                ),
                "nearest_distance": raw.get("nearest_distance", ""),
                "required_spacing": raw.get("required_spacing", ""),
            }
        )

    print(
        "[PROGRESSIVE_FFD_DUAL] Candidate scoring | "
        f"mode={indicator_mode} active_ndv={n_active} "
        f"candidate_ndv={n_candidate}"
    )
    reduced_candidates = _reduce_candidates_to_interval_best(candidates)
    n_upper_active = len(level.upper_columns)
    return {
        "candidates": reduced_candidates,
        "raw_candidates": candidates,
        "active_scores": active_indicator,
        "active_upper_scores": active_indicator[:n_upper_active],
        "active_lower_scores": active_indicator[n_upper_active:],
    }
