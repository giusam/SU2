#!/usr/bin/env python

import copy
import csv
import glob
import io
import json
import math
import os
import shutil
import contextlib

import numpy as np
import SU2

from SU2.opt.bspline_driver.geometry_constraints import (
    is_geometry_constraint_name,
    normalize_geometry_constraint_name,
)

from SU2.opt.progressive_ffd_core import (
    build_ffd_mesh_columns,
    ffd_active_range_from_opts,
    make_active_side_ffd_definition,
    make_ffd_config_dump_compatible,
    make_dual_ffd_definition,
    make_ffd_definition,
    ordered_ffd_records,
    ordered_dual_ffd_records,
    validate_ffd_mesh_blending,
)
from SU2.opt.progressive_ffd_blending import (
    BEZIER,
    BSPLINE_UNIFORM,
    normalize_ffd_blending,
)
from SU2.opt.progressive_ffd_tangent import (
    COMPONENT,
    VIRTUAL_TANGENT,
    FFDTangentError,
    airfoil_area_value_and_field,
    airfoil_thickness_value_and_field,
    build_ffd_tangent_state,
    compare_tangent_spaces,
    fit_surface_ikkt_signal,
    internal_constraint_field,
    load_design_gradient_csv,
    load_surface_sensitivity_vector,
    normalize_ffd_scoring_mode,
    project_surface_field,
    validate_surface_projection,
)
from SU2.opt.progressive_ffd_mesh import rewrite_ffd_box_with_columns_and_reembed
from SU2.opt.progressive_ffd_envelope import FFDEnvelopeError
from SU2.opt.progressive_ffd_split import (
    FFDBoxSplitError,
    rewrite_dual_ffd_boxes_with_columns_and_reembed,
    rewrite_single_ffd_box_with_columns_and_reembed,
)
from SU2.opt.progressive_hh_projection import (
    _compute_ikkt_residual_vector,
    _dot_problem_kind,
    _extract_constraint_names,
    _extract_constraint_signs,
    _extract_constraint_specs,
    _find_latest_design_with_geometry,
    _find_real_adjoint_assets,
    _make_projection_state,
    _run_geo_gradient_for_function,
    _select_active_ikkt_constraints,
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


def _gradient_descent_indicator(values, cfg, verbose=True):
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

    if verbose:
        print(
            "[PROGRESSIVE_FFD] DESCENT_GRAD indicator | "
            f"bounds=[{lower:.6e},{upper:.6e}] rule={rule}"
        )
    return indicator.tolist()


def _indicator_from_gradient(values, cfg, indicator_mode, verbose=True):
    mode = str(indicator_mode or "ABS_GRAD").upper()
    if mode in (
        "DESCENT_GRAD",
        "BOUNDED_GRAD",
        "ABS_GRAD_DESCENT",
        "ONE_SIDED_ABS_GRAD",
    ):
        return _gradient_descent_indicator(values, cfg, verbose=verbose)
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
    compare_candidates=True,
    verbose=True,
):
    if float(min_spacing) <= 0.0:
        return candidates

    filtered = []
    for candidate in candidates:
        accepted, nearest, nearest_distance = _check_ffd_min_spacing(
            candidate,
            active_columns,
            filtered if compare_candidates else [],
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
        if verbose:
            print(
                "[PROGRESSIVE_FFD] Candidate rejected by min spacing | "
                f"side={side} x={float(candidate['x']):.6f} "
                f"nearest={float(nearest):.6f} "
                f"dist={float(nearest_distance):.6f} "
                f"required={float(min_spacing):.6f}"
            )

    return filtered


def _finite_gradient(values, expected_size, context):
    gradient = np.asarray(values, dtype=float).reshape(-1)
    if gradient.size != int(expected_size):
        raise RuntimeError(
            f"{context} gradient size mismatch: got {gradient.size}, "
            f"expected {expected_size}"
        )
    if not np.all(np.isfinite(gradient)):
        raise RuntimeError(f"{context} gradient contains non-finite values")
    return gradient


def _dual_record_index(records, side, x, tol=1.0e-10):
    side = str(side).upper()
    x = float(x)
    matches = [
        index
        for index, (record_side, record_x) in enumerate(records)
        if str(record_side).upper() == side and abs(float(record_x) - x) <= tol
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "Could not identify a unique dual FFD candidate in the temporary "
            f"DV definition: side={side}, x={x:.16g}, matches={matches}"
        )
    return matches[0]


def _ikkt_indicators_with_fixed_lambdas(
    objective_gradient,
    constraint_gradients,
    lambdas,
):
    objective = np.asarray(objective_gradient, dtype=float).reshape(-1)
    multipliers = np.asarray(lambdas, dtype=float).reshape(-1)
    if len(constraint_gradients) != multipliers.size:
        raise RuntimeError(
            "IKKT constraint-gradient/multiplier count mismatch: "
            f"gradients={len(constraint_gradients)}, lambdas={multipliers.size}"
        )
    if not constraint_gradients:
        indicators = np.abs(objective)
    else:
        matrix = np.column_stack(
            [
                _finite_gradient(values, objective.size, "IKKT constraint")
                for values in constraint_gradients
            ]
        )
        indicators = np.abs(objective - matrix @ multipliers)
    if not np.all(np.isfinite(indicators)):
        raise RuntimeError("IKKT candidate indicators contain non-finite values")
    return indicators


def _reduce_candidates_to_interval_best(candidates, verbose=True):
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
    if verbose:
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


def _resolve_accepted_projection_mesh(
    accepted_mesh,
    dot_mesh,
    adjoint_dir,
    design_dir,
    level_dir,
    marker,
    domain_mode="FULL",
    verbose=True,
):
    if not accepted_mesh:
        return dot_mesh

    accepted_mesh = str(accepted_mesh)
    if os.path.exists(accepted_mesh):
        accepted_path = os.path.abspath(accepted_mesh)
    else:
        accepted_path = _find_projection_mesh_source(
            accepted_mesh,
            adjoint_dir,
            design_dir,
            level_dir,
        )

    from SU2.opt.progressive_ffd_prepare import (
        _max_mesh_coordinate_difference,
        _mesh_geometry,
    )

    geometry = _mesh_geometry(accepted_path, marker, domain_mode=domain_mode)
    coordinate_error, point_id = _max_mesh_coordinate_difference(
        accepted_path,
        dot_mesh,
    )
    tolerance = 1.0e-12 * max(1.0, float(geometry["chord"]))
    if not math.isfinite(coordinate_error) or coordinate_error > tolerance:
        raise RuntimeError(
            "The accepted final mesh and the objective-adjoint DOT mesh do not "
            "represent the same physical baseline: "
            f"max_coordinate_error={coordinate_error:.6e}, "
            f"tolerance={tolerance:.6e}, point_id={point_id}, "
            f"accepted={accepted_path}, dot={dot_mesh}"
        )
    if verbose:
        print(
            "[PROGRESSIVE_FFD_DUAL] Accepted/DOT mesh consistency | "
            f"max_error={coordinate_error:.6e} tolerance={tolerance:.6e}"
        )
    return accepted_path


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

    side_records = bool(ordered_columns) and isinstance(
        ordered_columns[0], (list, tuple)
    )
    if side_records:
        cfg_dot["DEFINITION_DV"] = make_active_side_ffd_definition(
            ordered_columns,
            opts,
            column_index_by_x,
        )
        cfg_dot["FFD_CONTINUITY"] = "USER_INPUT"
        for key in ("FFD_FIX_I", "FFD_FIX_J", "FFD_FIX_K"):
            if key in cfg_dot:
                del cfg_dot[key]
    elif opts.get("ffd_dual_box", False):
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


def _projection_mesh_stem(cfg_dot):
    return os.path.splitext(
        os.path.basename(str(cfg_dot.get("MESH_FILENAME", "mesh")))
    )[0]


def _ffd_dot_artifact_directory(level_dir, cfg_dot, func_name):
    return os.path.join(
        level_dir,
        f"DOT_ONLY_{str(func_name).upper()}_{_projection_mesh_stem(cfg_dot)}",
    )


def _ffd_geo_artifact_directory(level_dir, cfg_dot, func_name):
    return os.path.join(
        level_dir,
        f"GEO_ONLY_{str(func_name).upper()}_{_projection_mesh_stem(cfg_dot)}",
    )


def _remove_artifact_path(path):
    if not path:
        return
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    elif os.path.lexists(path):
        os.remove(path)


def _remove_candidate_artifacts(candidate):
    _remove_artifact_path(candidate.get("temporary_mesh"))
    for artifact in candidate.get("projection_artifacts", []):
        _remove_artifact_path(artifact.get("path"))


def _exact_candidate_rank_key(candidate):
    side_rank = {"UPPER": 0, "LOWER": 1, "FFD": 2}
    return (
        -float(candidate["indicator"]),
        side_rank.get(str(candidate.get("side", "FFD")).upper(), 99),
        float(candidate["x"]),
    )


def _json_safe_value(value):
    if isinstance(value, np.ndarray):
        return [_json_safe_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe_value(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return value


def _ensure_dot_mesh_available(
    dot_test_dir,
    adjoint_dir,
    design_dir,
    level_dir,
    cfg_dot,
    verbose=True,
):
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
    if verbose:
        print(
            "[PROGRESSIVE_FFD] DOT projection | copied mesh into DOT_ONLY: "
            f"{mesh_src} -> {mesh_dst}"
        )


def _run_ffd_dot_for_function(
    level_dir,
    cfg_dot,
    state,
    func_name,
    verbose=True,
):
    func_name = str(func_name).upper()
    dot_kind = _dot_problem_kind(cfg_dot)
    adjoint_dir, design_dir = _find_real_adjoint_assets(level_dir, func_name)
    dot_test_dir = _ffd_dot_artifact_directory(level_dir, cfg_dot, func_name)

    if os.path.isdir(dot_test_dir):
        shutil.rmtree(dot_test_dir)
    shutil.copytree(adjoint_dir, dot_test_dir, symlinks=False)

    _ensure_dot_mesh_available(
        dot_test_dir,
        adjoint_dir,
        design_dir,
        level_dir,
        cfg_dot,
        verbose=verbose,
    )

    restart_candidates = glob.glob(os.path.join(design_dir, "solution_adj_*.dat"))
    if restart_candidates:
        for restart_src in restart_candidates:
            restart_dst = os.path.join(dot_test_dir, os.path.basename(restart_src))
            shutil.copy2(restart_src, restart_dst)
    elif dot_kind == "DISCRETE_ADJOINT":
        raise FileNotFoundError(f"No adjoint restart files found in {design_dir}")
    else:
        if verbose:
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
    if verbose:
        print(
            "[PROGRESSIVE_FFD] DOT projection | "
            f"kind={dot_kind} function={func_name} "
            f"ndv={grad_arr.size} norm={np.linalg.norm(grad_arr):.6e}"
        )
    return grad


def _run_ffd_geo_gradient_for_function(
    level_dir,
    cfg_dot,
    func_name,
    verbose=True,
):
    func_name = str(func_name).upper()
    geo_src_design = _find_latest_design_with_geometry(level_dir, func_name)
    if verbose:
        print(
            f"[PROGRESSIVE_FFD] GEOMETRY source design for "
            f"{func_name}: {geo_src_design}"
        )

    geo_test_dir = _ffd_geo_artifact_directory(level_dir, cfg_dot, func_name)
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


def _compute_ffd_dot_candidate_scores(level, opts, mesh_source=None):
    if set(level.columns_by_side).issubset({"UPPER", "LOWER"}):
        return _compute_dual_ffd_dot_candidate_scores(
            level,
            opts,
            mesh_source=mesh_source,
        )

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
        requested_names = _extract_constraint_names(cfg_level)
        constraint_names, _ = _select_active_ikkt_constraints(
            cfg_level,
            requested_names,
            design_dir=design_dir,
            active_tol=opts.get("ikkt_active_tol", 1.0e-6),
        )
        included_names = []
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
            included_names.append(cname)

        lambda_bounds = _extract_constraint_signs(cfg_level, included_names)
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


def _prepare_projection_variant(
    level,
    opts,
    cfg_level,
    real_dot_cfg,
    mesh_src,
    active_by_side,
    suffix,
    verbose=True,
):
    active_sides = tuple(
        side
        for side in ("UPPER", "LOWER")
        if side in active_by_side
    )
    if not active_sides:
        raise ValueError("A projection variant requires at least one active FFD side")
    active_by_side = {
        side: sorted(float(x) for x in active_by_side[side])
        for side in active_sides
    }
    output_context = (
        contextlib.nullcontext()
        if verbose
        else contextlib.redirect_stdout(io.StringIO())
    )
    with output_context:
        mesh_basename = f"ffd_projection_level{level.level_id}_{suffix}.su2"
        mesh_path = os.path.join(level.workdir, mesh_basename)
        if len(active_sides) == 2:
            upper_mesh_columns, _ = build_ffd_mesh_columns(
                mesh_src,
                opts["ffd_upper_box_tag"],
                active_columns=active_by_side["UPPER"],
                opts=opts,
            )
            lower_mesh_columns, _ = build_ffd_mesh_columns(
                mesh_src,
                opts["ffd_lower_box_tag"],
                active_columns=active_by_side["LOWER"],
                opts=opts,
            )
            mesh_info = rewrite_dual_ffd_boxes_with_columns_and_reembed(
                mesh_src,
                mesh_path,
                marker=opts["ffd_marker"],
                upper_tag=opts["ffd_upper_box_tag"],
                lower_tag=opts["ffd_lower_box_tag"],
                upper_columns=upper_mesh_columns,
                lower_columns=lower_mesh_columns,
                upper_offset_chord=opts["ffd_upper_offset_chord"],
                lower_offset_chord=opts["ffd_lower_offset_chord"],
                envelope_spec=opts.get("ffd_envelope_spec"),
                diagnostics_csv=False,
                overwrite=True,
            )
            column_index_by_side = {
                "UPPER": mesh_info["upper_column_index_by_x"],
                "LOWER": mesh_info["lower_column_index_by_x"],
            }
        else:
            side = active_sides[0]
            box_tag = (
                opts["ffd_upper_box_tag"]
                if side == "UPPER"
                else opts["ffd_lower_box_tag"]
            )
            offset = (
                opts["ffd_upper_offset_chord"]
                if side == "UPPER"
                else opts["ffd_lower_offset_chord"]
            )
            mesh_columns, _ = build_ffd_mesh_columns(
                mesh_src,
                box_tag,
                active_columns=active_by_side[side],
                opts=opts,
            )
            mesh_info = rewrite_single_ffd_box_with_columns_and_reembed(
                mesh_src,
                mesh_path,
                marker=opts["ffd_marker"],
                side=side,
                box_tag=box_tag,
                columns=mesh_columns,
                offset_chord=offset,
                envelope_spec=opts.get("ffd_envelope_spec"),
                diagnostics_csv=False,
                overwrite=True,
            )
            column_index_by_side = {
                side: mesh_info["column_index_by_x"],
            }
        validate_ffd_mesh_blending(
            mesh_info,
            opts,
            context=f"FFD DOT projection variant {suffix}",
        )
    ordered_records = ordered_ffd_records(active_by_side, active_sides)
    cfg_dot = _build_extended_ffd_dot_config(
        cfg_level,
        real_dot_cfg,
        mesh_basename,
        ordered_records,
        opts,
        column_index_by_side,
    )
    return {
        "cfg_dot": cfg_dot,
        "state": _make_projection_state(mesh_basename),
        "records": ordered_records,
        "column_index_by_side": column_index_by_side,
        "mesh": mesh_path,
    }


def _prepare_dual_projection_variant(
    level,
    opts,
    cfg_level,
    real_dot_cfg,
    mesh_src,
    upper_active,
    lower_active,
    suffix,
    verbose=True,
):
    """Compatibility wrapper for callers that still pass two side arrays."""

    return _prepare_projection_variant(
        level,
        opts,
        cfg_level,
        real_dot_cfg,
        mesh_src,
        {"UPPER": upper_active, "LOWER": lower_active},
        suffix,
        verbose=verbose,
    )


def _prepare_active_projection_variant(
    level,
    opts,
    cfg_level,
    real_dot_cfg,
    mesh_src,
    active_by_side,
    suffix,
    verbose=True,
):
    """Route dual callers through the stable compatibility entry point."""

    if set(active_by_side) == {"UPPER", "LOWER"}:
        return _prepare_dual_projection_variant(
            level,
            opts,
            cfg_level,
            real_dot_cfg,
            mesh_src,
            active_by_side["UPPER"],
            active_by_side["LOWER"],
            suffix,
            verbose=verbose,
        )
    return _prepare_projection_variant(
        level,
        opts,
        cfg_level,
        real_dot_cfg,
        mesh_src,
        active_by_side,
        suffix,
        verbose=verbose,
    )


def _run_dual_projection_constraint_gradients(
    level,
    cfg_dot,
    state,
    constraint_names,
    required,
    artifact_records=None,
    verbose=True,
):
    names = []
    gradients = []
    expected_size = len(cfg_dot["DV_VALUE_NEW"])
    for constraint_name in constraint_names:
        constraint_name = str(constraint_name).upper()
        dot_artifact = _ffd_dot_artifact_directory(
            level.workdir,
            cfg_dot,
            constraint_name,
        )
        try:
            values = _run_ffd_dot_for_function(
                level.workdir,
                cfg_dot,
                state,
                constraint_name,
                verbose=verbose,
            )
            source = "DOT"
            artifact_path = dot_artifact
        except Exception as dot_error:
            _remove_artifact_path(dot_artifact)
            try:
                values = _run_ffd_geo_gradient_for_function(
                    level.workdir,
                    cfg_dot,
                    constraint_name,
                    verbose=verbose,
                )
                source = "GEOMETRY"
                artifact_path = _ffd_geo_artifact_directory(
                    level.workdir,
                    cfg_dot,
                    constraint_name,
                )
            except Exception as geo_error:
                if required:
                    raise RuntimeError(
                        f"{constraint_name} gradient is unavailable for an exact "
                        "FFD candidate"
                    ) from geo_error
                if verbose:
                    print(
                        "[PROGRESSIVE_FFD_DUAL] IKKT warning | "
                        f"{constraint_name} gradient unavailable -> skipped "
                        f"(DOT: {dot_error}; GEOMETRY: {geo_error})"
                    )
                continue

        gradient = _finite_gradient(
            values,
            expected_size,
            f"{constraint_name} {source}",
        )
        if verbose:
            print(
                f"[PROGRESSIVE_FFD_DUAL] IKKT | "
                f"{constraint_name} via {source}"
            )
        if artifact_records is not None:
            artifact_records.append(
                {
                    "kind": source,
                    "function": constraint_name,
                    "path": artifact_path,
                }
            )
        names.append(constraint_name)
        gradients.append(gradient)
    return names, gradients


def _exact_blending_slug(opts):
    blending = normalize_ffd_blending(opts.get("ffd_blending", BEZIER))
    if blending == BEZIER:
        return "bezier"
    if blending == BSPLINE_UNIFORM:
        return "bspline_uniform"
    raise ValueError(f"Unsupported exact progressive FFD blending {blending!r}")


def _cleanup_exact_scoring_artifacts(level, opts=None):
    """Clean temporary and persisted exact-sequential scoring artifacts."""

    mesh_prefix = (
        f"ffd_projection_level{level.level_id}_{_exact_blending_slug(opts or {})}_"
    )
    for pattern in (
        os.path.join(level.workdir, mesh_prefix + "*.su2"),
        os.path.join(level.workdir, "DOT_ONLY_*_" + mesh_prefix + "*"),
        os.path.join(level.workdir, "GEO_ONLY_*_" + mesh_prefix + "*"),
    ):
        for path in glob.glob(pattern):
            _remove_artifact_path(path)
    _remove_artifact_path(
        os.path.join(level.workdir, "FFD_SELECTED_CANDIDATE")
    )
    _remove_artifact_path(
        os.path.join(
            level.workdir,
            f"ffd_candidate_scores_level{level.level_id}.csv",
        )
    )


def _cleanup_exact_bezier_scoring_artifacts(level, opts=None):
    """Compatibility wrapper for the former Bezier-only workflow."""

    return _cleanup_exact_scoring_artifacts(level, opts)


def _generate_dual_exact_candidates(active_by_side, opts):
    nsamples = int(opts.get("candidate_samples", 1))
    min_spacing = float(opts.get("min_center_spacing", 0.0))
    candidate_xmin, candidate_xmax = ffd_active_range_from_opts(opts)
    candidates = []
    for side in ("UPPER", "LOWER"):
        if side not in active_by_side:
            continue
        side_candidates = get_ffd_interval_candidates(
            active_by_side[side],
            xmin=candidate_xmin,
            xmax=candidate_xmax,
            nsamples=nsamples,
        )
        for candidate in side_candidates:
            candidate["side"] = side
        candidates.extend(
            _filter_candidates_by_min_spacing(
                side_candidates,
                active_by_side[side],
                min_spacing,
                candidate_xmin,
                candidate_xmax,
                side=side,
                compare_candidates=False,
                verbose=False,
            )
        )
    return candidates


def _exact_growth_ratio_insertion_target(level, raw_candidates, opts):
    interval_count = len(
        {
            (str(candidate["side"]).upper(), int(candidate["interval_id"]))
            for candidate in raw_candidates
        }
    )
    growth_ratio = float(opts.get("growth_ratio", 2.0))
    if not math.isfinite(growth_ratio) or growth_ratio <= 0.0:
        raise ValueError("Sequential exact FFD growth_ratio must be finite and > 0")
    if growth_ratio <= 1.0:
        target = 1
    else:
        target = max(1, int(math.ceil(growth_ratio * level.ndv)) - level.ndv)
    nfinal = opts.get("nfinal", None)
    if nfinal is not None:
        target = min(target, max(0, int(nfinal) - int(level.ndv)))
    return int(target), int(interval_count)


def _write_exact_candidate_scores_csv(level, candidates):
    path = os.path.join(
        level.workdir,
        f"ffd_candidate_scores_level{level.level_id}.csv",
    )
    fieldnames = [
        "ffd_scoring_mode",
        "scoring_basis",
        "signal_source",
        "insertion_step",
        "insertion_target",
        "ndv_before_insertion",
        "ndv_after_insertion",
        "candidate_number",
        "side",
        "interval_id",
        "interval_left",
        "interval_right",
        "sample_index",
        "sample_fraction",
        "x",
        "candidate_dv_index",
        "control_point_i",
        "objective_function",
        "objective_gradient",
        "surface_residual_component",
        "constraint_gradients",
        "indicator",
        "energy_current",
        "energy_candidate",
        "signal_energy",
        "score_net",
        "score_net_normalized",
        "score_pure",
        "score_pure_normalized",
        "rank_current",
        "rank_candidate",
        "rank_gain",
        "pure_rank",
        "nesting_rms",
        "nesting_max",
        "locality",
        "innovation_center_x",
        "admissible",
        "rejected_reason",
        "interval_winner",
        "global_rank",
    ]
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {
                    "ffd_scoring_mode": candidate.get(
                        "ffd_scoring_mode", COMPONENT
                    ),
                    "scoring_basis": candidate.get("scoring_basis", ""),
                    "signal_source": candidate.get("signal_source", ""),
                    "insertion_step": candidate.get("insertion_step", 1),
                    "insertion_target": candidate.get("insertion_target", 1),
                    "ndv_before_insertion": candidate.get(
                        "ndv_before_insertion", ""
                    ),
                    "ndv_after_insertion": candidate.get(
                        "ndv_after_insertion", ""
                    ),
                    "candidate_number": candidate["candidate_number"],
                    "side": candidate["side"],
                    "interval_id": candidate["interval_id"],
                    "interval_left": f"{candidate['interval_left']:.16g}",
                    "interval_right": f"{candidate['interval_right']:.16g}",
                    "sample_index": candidate["sample_index"],
                    "sample_fraction": f"{candidate['sample_fraction']:.16g}",
                    "x": f"{candidate['x']:.16g}",
                    "candidate_dv_index": candidate["candidate_dv_index"],
                    "control_point_i": candidate["control_point_i"],
                    "objective_function": candidate["objective_function"],
                    "objective_gradient": (
                        f"{candidate['objective_gradient_component']:.16e}"
                    ),
                    "surface_residual_component": (
                        ""
                        if candidate.get("surface_residual_component") is None
                        else f"{float(candidate['surface_residual_component']):.16e}"
                    ),
                    "constraint_gradients": json.dumps(
                        candidate.get("constraint_gradient_components", {}),
                        sort_keys=True,
                    ),
                    "indicator": f"{candidate['indicator']:.16e}",
                    "energy_current": candidate.get("energy_current", ""),
                    "energy_candidate": candidate.get("energy_candidate", ""),
                    "signal_energy": candidate.get("signal_energy", ""),
                    "score_net": candidate.get("score_net", ""),
                    "score_net_normalized": candidate.get(
                        "score_net_normalized", ""
                    ),
                    "score_pure": candidate.get("score_pure", ""),
                    "score_pure_normalized": candidate.get(
                        "score_pure_normalized", ""
                    ),
                    "rank_current": candidate.get("rank_current", ""),
                    "rank_candidate": candidate.get("rank_candidate", ""),
                    "rank_gain": candidate.get("rank_gain", ""),
                    "pure_rank": candidate.get("pure_rank", ""),
                    "nesting_rms": candidate.get("nesting_rms", ""),
                    "nesting_max": candidate.get("nesting_max", ""),
                    "locality": candidate.get("locality", ""),
                    "innovation_center_x": candidate.get(
                        "innovation_center_x", ""
                    ),
                    "admissible": (
                        "YES" if candidate.get("admissible", True) else "NO"
                    ),
                    "rejected_reason": candidate.get("rejected_reason", ""),
                    "interval_winner": (
                        "YES" if candidate.get("interval_winner") else "NO"
                    ),
                    "global_rank": candidate.get("rank", ""),
                }
            )
    return path


def _print_exact_candidate_ranking(
    ranked_candidates,
    insertion_step=1,
    insertion_target=1,
):
    print(
        "[PROGRESSIVE_FFD_DUAL] Candidate ranking | "
        f"insertion={int(insertion_step)}/{int(insertion_target)}"
    )
    print(
        "[PROGRESSIVE_FFD_DUAL] "
        "rank side  interval                  x          indicator"
    )
    for candidate in ranked_candidates:
        print(
            "[PROGRESSIVE_FFD_DUAL] "
            f"{candidate['rank']:>4d} "
            f"{candidate['side']:<5s} "
            f"[{candidate['interval_left']:.6f},"
            f"{candidate['interval_right']:.6f}] "
            f"{candidate['x']:.6f} "
            f"{candidate['indicator']:.6e}"
        )


def _persist_selected_candidate_artifacts(
    level,
    selected,
    indicator_mode,
    ikkt_metadata,
    selected_dir=None,
):
    if selected_dir is None:
        selected_dir = os.path.join(level.workdir, "FFD_SELECTED_CANDIDATE")
    _remove_artifact_path(selected_dir)
    os.makedirs(selected_dir)

    selected_mesh = os.path.join(selected_dir, "candidate.su2")
    shutil.move(selected["temporary_mesh"], selected_mesh)
    selected["temporary_mesh"] = selected_mesh

    persisted_artifacts = []
    for artifact in selected.get("projection_artifacts", []):
        prefix = "DOT" if artifact["kind"] == "DOT" else "GEO"
        destination = os.path.join(
            selected_dir,
            f"{prefix}_{artifact['function']}",
        )
        _remove_artifact_path(destination)
        shutil.move(artifact["path"], destination)
        persisted = dict(artifact)
        persisted["path"] = destination
        persisted_artifacts.append(persisted)
    selected["projection_artifacts"] = persisted_artifacts
    selected["artifact_directory"] = selected_dir

    constraint_components = selected.get("constraint_gradient_components", {})
    gradients_csv = os.path.join(selected_dir, "candidate_gradients.csv")
    with open(gradients_csv, "w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["kind", "function", "gradient_component"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "kind": "OBJECTIVE",
                "function": selected["objective_function"],
                "gradient_component": (
                    f"{selected['objective_gradient_component']:.16e}"
                ),
            }
        )
        for name, value in constraint_components.items():
            writer.writerow(
                {
                    "kind": "CONSTRAINT",
                    "function": name,
                    "gradient_component": f"{float(value):.16e}",
                }
            )

    metadata = {
        key: value
        for key, value in selected.items()
        if key not in ("projection_artifacts",)
    }
    metadata.update(
        {
            "indicator_mode": indicator_mode,
            "projection_artifacts": persisted_artifacts,
            "candidate_gradients_csv": gradients_csv,
        }
    )
    metadata_path = os.path.join(selected_dir, "candidate_metadata.json")
    with open(metadata_path, "w") as stream:
        json.dump(
            _json_safe_value(metadata),
            stream,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )

    if indicator_mode == "IKKT":
        with open(os.path.join(selected_dir, "ikkt_baseline.json"), "w") as stream:
            json.dump(
                _json_safe_value(ikkt_metadata),
                stream,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
    return selected_dir


def _resolve_surface_sensitivity_file(adjoint_dir, function_name):
    direct = os.path.join(adjoint_dir, "surface_sens.csv")
    if os.path.isfile(direct):
        return direct
    candidates = sorted(
        path
        for path in glob.glob(os.path.join(adjoint_dir, "*surface*sens*.csv"))
        if os.path.isfile(path)
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No surface_sens.csv found for {function_name} in {adjoint_dir}"
        )
    raise RuntimeError(
        f"Ambiguous surface sensitivity files for {function_name}: {candidates}"
    )


def _resolve_saved_gradient_file(adjoint_dir):
    candidates = sorted(
        path
        for path in glob.glob(os.path.join(adjoint_dir, "*of_grad*.csv"))
        if os.path.isfile(path)
    )
    if not candidates:
        return None
    if len(candidates) > 1:
        raise RuntimeError(
            f"Ambiguous saved SU2 gradient files in {adjoint_dir}: {candidates}"
        )
    return candidates[0]


def _validate_saved_surface_projection(
    tangent_state,
    field,
    adjoint_dir,
    function_name,
):
    try:
        reference_file = _resolve_saved_gradient_file(adjoint_dir)
    except Exception as exc:
        return {
            "status": "reference_unavailable",
            "function": str(function_name).upper(),
            "reference_file": None,
            "detail": str(exc),
        }
    if reference_file is None:
        return {
            "status": "reference_unavailable",
            "function": str(function_name).upper(),
            "reference_file": None,
        }
    try:
        validation = validate_surface_projection(
            tangent_state,
            field,
            load_design_gradient_csv(reference_file),
            absolute_tolerance=5.0e-6,
        )
    except Exception as exc:
        return {
            "status": "reference_parameterization_differs",
            "function": str(function_name).upper(),
            "reference_file": reference_file,
            "detail": str(exc),
        }
    validation.update(
        {
            "status": (
                "passed"
                if validation["passed"]
                else "reference_parameterization_differs"
            ),
            "function": str(function_name).upper(),
            "reference_file": reference_file,
            "note": (
                "The saved gradient belongs to the optimized level box; the "
                "virtual baseline can use a newly re-embedded box. A strict "
                "validation requires SU2_DOT on that same virtual baseline."
            ),
        }
    )
    return validation


def _prepare_virtual_tangent_context(level, opts, cfg_level, obj_name, design_dir):
    obj_adj_dir, objective_design_dir = _find_real_adjoint_assets(
        level.workdir,
        obj_name,
    )
    if os.path.abspath(objective_design_dir) != os.path.abspath(design_dir):
        raise RuntimeError(
            "Objective adjoint and accepted-design directories disagree during "
            "virtual FFD scoring"
        )
    requested_names = _extract_constraint_names(cfg_level)
    active_names, constraint_status = _select_active_ikkt_constraints(
        cfg_level,
        requested_names,
        design_dir=design_dir,
        active_tol=opts.get("ikkt_active_tol", 1.0e-6),
        verbose=False,
    )
    specs = {
        spec["name"]: spec
        for spec in _extract_constraint_specs(cfg_level, requested_names)
    }
    return {
        "objective_function": str(obj_name).upper(),
        "objective_adjoint_dir": obj_adj_dir,
        "objective_field_file": _resolve_surface_sensitivity_file(
            obj_adj_dir,
            obj_name,
        ),
        "design_dir": design_dir,
        "active_constraint_names": list(active_names),
        "constraint_status": constraint_status,
        "constraint_specs": specs,
        "active_tol": float(opts.get("ikkt_active_tol", 1.0e-6)),
        "initialized_node_ids": None,
    }


def _constraint_active_from_value(spec, current_value, active_tol):
    sign = str(spec.get("sign", "")).strip()
    target = spec.get("target")
    if target is None:
        return True, None, "active_status_unknown"
    if sign == ">":
        c_value = float(current_value) - float(target)
    elif sign == "<":
        c_value = float(target) - float(current_value)
    elif sign == "=":
        c_value = float(current_value) - float(target)
        return True, c_value, "active_equality"
    else:
        return True, None, "active_status_unknown"
    active = bool(c_value <= float(active_tol))
    return active, c_value, "active_inequality" if active else "inactive"


def _initialize_virtual_tangent_context(
    level,
    opts,
    baseline_state,
    virtual_context,
):
    node_ids = list(baseline_state["node_ids"])
    initialized = virtual_context.get("initialized_node_ids")
    if initialized is not None:
        if list(initialized) != node_ids:
            raise RuntimeError(
                "Virtual FFD marker node order changed between sequential passes"
            )
        return virtual_context

    objective_field = load_surface_sensitivity_vector(
        virtual_context["objective_field_file"],
        baseline_state,
    )
    objective_projection_validation = _validate_saved_surface_projection(
        baseline_state,
        objective_field,
        virtual_context.get("objective_adjoint_dir", ""),
        virtual_context["objective_function"],
    ) if virtual_context.get("objective_adjoint_dir") else {
        "status": "reference_unavailable",
        "function": virtual_context["objective_function"],
        "reference_file": None,
    }
    status_by_name = {
        str(record["name"]).upper(): dict(record)
        for record in virtual_context["constraint_status"]
    }
    constraint_records = []
    inactive_records = [
        dict(record)
        for record in virtual_context["constraint_status"]
        if not bool(record.get("included", False))
    ]
    unsupported_records = []
    indicator_mode = str(opts.get("adaptive_indicator", "ABS_GRAD")).upper()

    if indicator_mode == "IKKT":
        for name in virtual_context["active_constraint_names"]:
            name = str(name).upper()
            spec = dict(virtual_context["constraint_specs"].get(name, {}))
            spec.setdefault("name", name)
            status = status_by_name.get(name, {})
            provider = None
            provider_metadata = {}
            raw_field = None
            provider_value = status.get("current_value")
            field_file = None

            if is_geometry_constraint_name(name):
                canonical = normalize_geometry_constraint_name(name)
                if canonical == "AIRFOIL_AREA":
                    provider_value, raw_field = airfoil_area_value_and_field(
                        baseline_state
                    )
                    provider = "analytic_airfoil_area"
                    provider_metadata = {
                        "marker_closed": bool(baseline_state["closed"]),
                        "implicit_chord_closure": not bool(
                            baseline_state["closed"]
                        ),
                    }
                elif canonical == "AIRFOIL_THICKNESS":
                    provider_value, raw_field, provider_metadata = (
                        airfoil_thickness_value_and_field(baseline_state)
                    )
                    provider = "analytic_airfoil_thickness"
                else:
                    unsupported_records.append(
                        {
                            "name": name,
                            "source": "GEOMETRY",
                            "reason": "unsupported_geometry_constraint",
                        }
                    )
                    continue

                active, c_value, active_status = _constraint_active_from_value(
                    spec,
                    provider_value,
                    virtual_context["active_tol"],
                )
                if not active:
                    inactive_records.append(
                        {
                            **status,
                            "name": name,
                            "provider": provider,
                            "provider_current_value": float(provider_value),
                            "c_value": c_value,
                            "status": active_status,
                        }
                    )
                    continue
            else:
                try:
                    constraint_adj_dir, constraint_design_dir = (
                        _find_real_adjoint_assets(level.workdir, name)
                    )
                except Exception as exc:
                    unsupported_records.append(
                        {
                            "name": name,
                            "source": "AERO",
                            "reason": "active_field_unavailable",
                            "detail": str(exc),
                        }
                    )
                    continue
                if os.path.abspath(constraint_design_dir) != os.path.abspath(
                    virtual_context["design_dir"]
                ):
                    unsupported_records.append(
                        {
                            "name": name,
                            "source": "AERO",
                            "reason": "adjoint_baseline_mismatch",
                            "field_design_dir": constraint_design_dir,
                            "objective_design_dir": virtual_context["design_dir"],
                        }
                    )
                    continue
                field_file = _resolve_surface_sensitivity_file(
                    constraint_adj_dir,
                    name,
                )
                raw_field = load_surface_sensitivity_vector(
                    field_file,
                    baseline_state,
                )
                provider = "adjoint_surface_sensitivity"
                if provider_value is None:
                    unsupported_records.append(
                        {
                            "name": name,
                            "source": "AERO",
                            "reason": "current_value_unavailable",
                            "field_file": field_file,
                        }
                    )
                    continue
                provider_metadata["projection_validation"] = (
                    _validate_saved_surface_projection(
                        baseline_state,
                        raw_field,
                        constraint_adj_dir,
                        name,
                    )
                )

            internal = internal_constraint_field(
                spec,
                provider_value,
                raw_field,
            )
            constraint_records.append(
                {
                    "name": name,
                    "source": (
                        "GEOMETRY" if is_geometry_constraint_name(name) else "AERO"
                    ),
                    "function_name": name,
                    "original_operator": str(spec.get("sign", "")),
                    "target": spec.get("target"),
                    "config_scale": float(spec.get("scale", 1.0)),
                    "current_value": float(provider_value),
                    "optimizer_current_value": status.get("current_value"),
                    "provider": provider,
                    "field_file": field_file,
                    "provider_metadata": provider_metadata,
                    **internal,
                }
            )

    if unsupported_records:
        details = "; ".join(
            f"{record['name']}:{record['reason']}"
            for record in unsupported_records
        )
        raise RuntimeError(
            "Virtual FFD IKKT requires every active constraint field; " + details
        )

    virtual_context.update(
        {
            "initialized_node_ids": node_ids,
            "objective_field": objective_field,
            "objective_projection_validation": objective_projection_validation,
            "constraint_records": constraint_records,
            "inactive_constraints": inactive_records,
            "unsupported_constraints": unsupported_records,
            "progressive_thickness": {
                "enabled": bool(opts.get("ffd_thickness_enabled", False)),
                "included": False,
                "reason": opts.get(
                    "ffd_thickness_ikkt_exclusion_reason",
                    "optimizer-only progressive constraint; excluded from FFD IKKT",
                ),
            },
        }
    )
    return virtual_context


def _virtual_ikkt_metadata(
    virtual_context,
    signal_source,
    fit_diagnostics,
    lambdas,
    insertion_step,
    insertion_target,
):
    included = []
    for index, record in enumerate(virtual_context.get("constraint_records", [])):
        item = {
            key: value
            for key, value in record.items()
            if key != "field"
        }
        item["lambda"] = float(lambdas[index])
        item["used_in_ikkt_gradient"] = True
        included.append(item)
    return {
        "scoring_mode": VIRTUAL_TANGENT,
        "signal_source": signal_source,
        "insertion_step": int(insertion_step),
        "insertion_target": int(insertion_target),
        "objective_function": virtual_context["objective_function"],
        "objective_field_file": virtual_context["objective_field_file"],
        "objective_projection_validation": virtual_context.get(
            "objective_projection_validation",
            {"status": "reference_unavailable"},
        ),
        "sensitivity_metric": "DISCRETE_EUCLIDEAN",
        "sensitivity_weighting": "NONE",
        "sign_convention": "C_GE_ZERO_LAGRANGIAN_MINUS_LAMBDA_C",
        "included_constraints": included,
        "inactive_constraints": virtual_context.get("inactive_constraints", []),
        "unsupported_constraints": virtual_context.get(
            "unsupported_constraints", []
        ),
        "fit_diagnostics": fit_diagnostics,
        "lambdas": np.asarray(lambdas, dtype=float).tolist(),
        "progressive_thickness": virtual_context["progressive_thickness"],
    }


def _virtual_rejected_candidate(
    raw,
    candidate_number,
    active_by_side,
    opts,
    obj_name,
    signal_source,
    insertion_step,
    insertion_target,
    reason,
):
    return {
        "insertion_step": int(insertion_step),
        "insertion_target": int(insertion_target),
        "ndv_before_insertion": sum(
            len(values) for values in active_by_side.values()
        ),
        "ndv_after_insertion": sum(
            len(values) for values in active_by_side.values()
        )
        + 1,
        "candidate_number": int(candidate_number),
        "side": str(raw["side"]).upper(),
        "x": float(raw["x"]),
        "grad": 0.0,
        "indicator": 0.0,
        "interval_id": raw["interval_id"],
        "interval_left": float(raw["interval_left"]),
        "interval_right": float(raw["interval_right"]),
        "sample_index": int(raw["sample_index"]),
        "sample_fraction": float(raw["sample_fraction"]),
        "candidate_dv_index": "",
        "control_point_i": "",
        "temporary_mesh": None,
        "projection_artifacts": [],
        "objective_function": obj_name,
        "objective_gradient_component": 0.0,
        "surface_residual_component": None,
        "constraint_gradient_components": {},
        "scoring_basis": "VIRTUAL_TANGENT_SPACE",
        "ffd_scoring_mode": VIRTUAL_TANGENT,
        "signal_source": signal_source,
        "ffd_blending": normalize_ffd_blending(
            opts.get("ffd_blending", BEZIER)
        ),
        "progressive_thickness_ikkt_included": False,
        "admissible": False,
        "rejected_reason": str(reason),
        "nearest_center_or_boundary": raw.get(
            "nearest_center_or_boundary", ""
        ),
        "nearest_distance": raw.get("nearest_distance", ""),
        "required_spacing": raw.get("required_spacing", ""),
    }


def _compute_dual_virtual_tangent_candidate_scores_impl(
    level,
    opts,
    cfg_level,
    raw_candidates,
    active_by_side,
    virtual_context,
    accepted_mesh=None,
    insertion_step=1,
    insertion_target=1,
):
    obj_name = str(cfg_level.get("OBJECTIVE_FUNCTION", "DRAG")).upper()
    obj_adj_dir, design_dir = _find_real_adjoint_assets(level.workdir, obj_name)
    real_dot_cfg_path, _ = _select_dot_config_path(obj_adj_dir, cfg_level)
    real_dot_cfg = SU2.io.Config(real_dot_cfg_path)
    dot_mesh_src = _find_projection_mesh_source(
        str(real_dot_cfg["MESH_FILENAME"]),
        obj_adj_dir,
        design_dir,
        level.workdir,
    )
    mesh_src = _resolve_accepted_projection_mesh(
        accepted_mesh,
        dot_mesh_src,
        obj_adj_dir,
        design_dir,
        level.workdir,
        opts.get("ffd_marker", "AIRFOIL"),
        domain_mode=opts.get("ffd_domain_mode", "FULL"),
        verbose=False,
    )

    baseline = _prepare_active_projection_variant(
        level,
        opts,
        cfg_level,
        real_dot_cfg,
        mesh_src,
        active_by_side,
        (
            f"{_exact_blending_slug(opts)}_virtual_step_"
            f"{int(insertion_step):03d}_baseline"
        ),
        verbose=False,
    )
    try:
        baseline_state = build_ffd_tangent_state(
            baseline["mesh"],
            opts["ffd_marker"],
            active_by_side,
            opts,
        )
        virtual_context = _initialize_virtual_tangent_context(
            level,
            opts,
            baseline_state,
            virtual_context,
        )
        indicator_mode = str(opts.get("adaptive_indicator", "ABS_GRAD")).upper()
        if indicator_mode == "IKKT":
            signal, lambdas, fit_diagnostics = fit_surface_ikkt_signal(
                virtual_context["objective_field"],
                virtual_context["constraint_records"],
                baseline_state,
            )
            signal_source = (
                "IKKT_SURFACE_RESIDUAL"
                if len(virtual_context["constraint_records"])
                else "OBJECTIVE_ONLY_NO_ACTIVE_CONSTRAINTS"
            )
        else:
            signal = np.asarray(virtual_context["objective_field"], dtype=float)
            lambdas = np.zeros(0, dtype=float)
            fit_diagnostics = {
                "status": "objective_only",
                "objective_gradient": project_surface_field(
                    baseline_state,
                    signal,
                ).tolist(),
            }
            signal_source = "OBJECTIVE_SURFACE_SENSITIVITY"
        ikkt_metadata = _virtual_ikkt_metadata(
            virtual_context,
            signal_source,
            fit_diagnostics,
            lambdas,
            insertion_step,
            insertion_target,
        )

        candidates = []
        best_candidate = None
        for candidate_number, raw in enumerate(raw_candidates):
            side = str(raw["side"]).upper()
            x = float(raw["x"])
            candidate_active = {
                active_side: list(values)
                for active_side, values in active_by_side.items()
            }
            candidate_active[side].append(x)
            variant = None
            candidate_suffix = (
                f"{_exact_blending_slug(opts)}_virtual_step_"
                f"{int(insertion_step):03d}_candidate_"
                f"{candidate_number:04d}_{side.lower()}"
            )
            candidate_mesh_path = os.path.join(
                level.workdir,
                f"ffd_projection_level{level.level_id}_{candidate_suffix}.su2",
            )
            try:
                variant = _prepare_active_projection_variant(
                    level,
                    opts,
                    cfg_level,
                    real_dot_cfg,
                    mesh_src,
                    candidate_active,
                    candidate_suffix,
                    verbose=False,
                )
                candidate_state = build_ffd_tangent_state(
                    variant["mesh"],
                    opts["ffd_marker"],
                    candidate_active,
                    opts,
                )
                metrics = compare_tangent_spaces(
                    baseline_state,
                    candidate_state,
                    signal,
                    x,
                )
            except (FFDBoxSplitError, FFDEnvelopeError, FFDTangentError) as exc:
                _remove_artifact_path(
                    variant.get("mesh") if variant is not None else candidate_mesh_path
                )
                candidates.append(
                    _virtual_rejected_candidate(
                        raw,
                        candidate_number,
                        active_by_side,
                        opts,
                        obj_name,
                        signal_source,
                        insertion_step,
                        insertion_target,
                        (
                            "VIRTUAL_TANGENT_REEMBED_INVALID:"
                            f"{type(exc).__name__}:{exc}"
                        ),
                    )
                )
                continue
            candidate_dv_index = _dual_record_index(variant["records"], side, x)
            objective_gradient = project_surface_field(
                candidate_state,
                virtual_context["objective_field"],
            )
            residual_gradient = project_surface_field(candidate_state, signal)
            constraint_components = {}
            for record in virtual_context["constraint_records"]:
                projected = project_surface_field(candidate_state, record["field"])
                constraint_components[record["name"]] = float(
                    projected[candidate_dv_index]
                )
            control_mapping = variant["column_index_by_side"][side]
            control_index = next(
                int(value)
                for station, value in control_mapping.items()
                if abs(float(station) - x) <= 1.0e-10
            )
            admissible = bool(
                int(metrics["rank_gain"]) == 1
                and int(metrics["pure_rank"]) == 1
            )
            rejected_reason = raw.get("rejected_reason", "")
            if not admissible:
                rejected_reason = (
                    "VIRTUAL_TANGENT_RANK_MISMATCH:"
                    f"rank_gain={metrics['rank_gain']},pure_rank={metrics['pure_rank']}"
                )
            candidate = {
                "insertion_step": int(insertion_step),
                "insertion_target": int(insertion_target),
                "ndv_before_insertion": sum(
                    len(values) for values in active_by_side.values()
                ),
                "ndv_after_insertion": sum(
                    len(values) for values in active_by_side.values()
                )
                + 1,
                "candidate_number": int(candidate_number),
                "side": side,
                "x": x,
                "grad": float(objective_gradient[candidate_dv_index]),
                "indicator": float(metrics["score_net"]),
                "interval_id": raw["interval_id"],
                "interval_left": float(raw["interval_left"]),
                "interval_right": float(raw["interval_right"]),
                "sample_index": int(raw["sample_index"]),
                "sample_fraction": float(raw["sample_fraction"]),
                "candidate_dv_index": int(candidate_dv_index),
                "control_point_i": control_index,
                "temporary_mesh": variant["mesh"],
                "projection_artifacts": [],
                "objective_function": obj_name,
                "objective_gradient_component": float(
                    objective_gradient[candidate_dv_index]
                ),
                "surface_residual_component": float(
                    residual_gradient[candidate_dv_index]
                ),
                "constraint_gradient_components": constraint_components,
                "scoring_basis": "VIRTUAL_TANGENT_SPACE",
                "ffd_scoring_mode": VIRTUAL_TANGENT,
                "signal_source": signal_source,
                "ffd_blending": normalize_ffd_blending(
                    opts.get("ffd_blending", BEZIER)
                ),
                "progressive_thickness_ikkt_included": False,
                "progressive_thickness_ikkt_exclusion_reason": opts.get(
                    "ffd_thickness_ikkt_exclusion_reason",
                    "excluded from FFD IKKT",
                ),
                "admissible": admissible,
                "rejected_reason": rejected_reason,
                "nearest_center_or_boundary": raw.get(
                    "nearest_center_or_boundary", ""
                ),
                "nearest_distance": raw.get("nearest_distance", ""),
                "required_spacing": raw.get("required_spacing", ""),
                **metrics,
            }
            candidates.append(candidate)
            if admissible and (
                best_candidate is None
                or _exact_candidate_rank_key(candidate)
                < _exact_candidate_rank_key(best_candidate)
            ):
                if best_candidate is not None:
                    _remove_candidate_artifacts(best_candidate)
                best_candidate = candidate
            else:
                _remove_candidate_artifacts(candidate)

        admissible_candidates = [
            candidate for candidate in candidates if candidate.get("admissible", False)
        ]
        reduced_candidates = _reduce_candidates_to_interval_best(
            admissible_candidates,
            verbose=False,
        )
        ranked_candidates = sorted(
            reduced_candidates,
            key=_exact_candidate_rank_key,
        )
        for rank, candidate in enumerate(ranked_candidates, start=1):
            candidate["rank"] = rank
            candidate["interval_winner"] = True
        for candidate in candidates:
            candidate.setdefault("interval_winner", False)
        if best_candidate is None or not ranked_candidates:
            raise RuntimeError(
                "Virtual FFD tangent scoring found no rank-one admissible candidate"
            )
        if ranked_candidates[0] is not best_candidate:
            raise RuntimeError(
                "Virtual FFD artifact retention disagrees with candidate ranking"
            )
        if float(best_candidate["indicator"]) <= 0.0:
            print(
                "[PROGRESSIVE_FFD_DUAL] VIRTUAL_TANGENT warning | all "
                "admissible candidate scores are non-positive; selecting maximum "
                f"score={float(best_candidate['indicator']):.6e}"
            )
        _print_exact_candidate_ranking(
            ranked_candidates,
            insertion_step=insertion_step,
            insertion_target=insertion_target,
        )
        return {
            "candidates": ranked_candidates,
            "raw_candidates": candidates,
            "scoring_basis": "VIRTUAL_TANGENT_SPACE",
            "ffd_blending": normalize_ffd_blending(
                opts.get("ffd_blending", BEZIER)
            ),
            "ikkt_constraint_names": [
                record["name"] for record in virtual_context["constraint_records"]
            ],
            "ikkt_lambdas": np.asarray(lambdas, dtype=float).tolist(),
            "ikkt_metadata": ikkt_metadata,
            "selected_candidate": best_candidate,
        }
    finally:
        _remove_artifact_path(baseline["mesh"])


def _compute_dual_bezier_exact_candidate_scores_impl(
    level,
    opts,
    cfg_level,
    raw_candidates,
    active_by_side,
    accepted_mesh=None,
    insertion_step=1,
    insertion_target=1,
):
    """Score one exact FFD insertion in its actual next-level basis.

    The historical function name is retained for compatibility with callers
    and tests; both BEZIER and BSPLINE_UNIFORM use this implementation.
    """

    obj_name = str(cfg_level.get("OBJECTIVE_FUNCTION", "DRAG")).upper()
    blending = normalize_ffd_blending(opts.get("ffd_blending", BEZIER))
    blending_slug = _exact_blending_slug(opts)
    obj_adj_dir, design_dir = _find_real_adjoint_assets(level.workdir, obj_name)
    real_dot_cfg_path, _ = _select_dot_config_path(obj_adj_dir, cfg_level)
    indicator_mode = str(opts.get("adaptive_indicator", "ABS_GRAD")).upper()
    real_dot_cfg = SU2.io.Config(real_dot_cfg_path)
    dot_mesh_src = _find_projection_mesh_source(
        str(real_dot_cfg["MESH_FILENAME"]),
        obj_adj_dir,
        design_dir,
        level.workdir,
    )
    mesh_src = _resolve_accepted_projection_mesh(
        accepted_mesh,
        dot_mesh_src,
        obj_adj_dir,
        design_dir,
        level.workdir,
        opts.get("ffd_marker", "AIRFOIL"),
        domain_mode=opts.get("ffd_domain_mode", "FULL"),
        verbose=False,
    )

    fixed_lambda = np.zeros(0, dtype=float)
    ikkt_constraint_names = []
    ikkt_metadata = {
        "constraint_names": [],
        "constraint_status": [],
        "lambdas": [],
        "lambda_lower": [],
        "lambda_upper": [],
        "sign_convention": "SLSQP_GE_RAW",
        "ffd_blending": blending,
        "progressive_thickness": {
            "enabled": bool(opts.get("ffd_thickness_enabled", False)),
            "included": False,
            "reason": opts.get(
                "ffd_thickness_ikkt_exclusion_reason",
                "excluded from FFD IKKT",
            ),
        },
    }
    if indicator_mode == "IKKT":
        baseline_artifacts = []
        baseline = _prepare_active_projection_variant(
            level,
            opts,
            cfg_level,
            real_dot_cfg,
            mesh_src,
            active_by_side,
            f"{blending_slug}_step_{int(insertion_step):03d}_baseline",
            verbose=False,
        )
        baseline_size = len(baseline["records"])
        baseline_obj = _finite_gradient(
            _run_ffd_dot_for_function(
                level.workdir,
                baseline["cfg_dot"],
                baseline["state"],
                obj_name,
                verbose=False,
            ),
            baseline_size,
            f"{obj_name} {blending} baseline",
        )
        baseline_artifacts.append(
            {
                "kind": "DOT",
                "function": obj_name,
                "path": _ffd_dot_artifact_directory(
                    level.workdir,
                    baseline["cfg_dot"],
                    obj_name,
                ),
            }
        )
        requested_names = _extract_constraint_names(cfg_level)
        active_names, constraint_status = _select_active_ikkt_constraints(
            cfg_level,
            requested_names,
            design_dir=design_dir,
            active_tol=opts.get("ikkt_active_tol", 1.0e-6),
            verbose=False,
        )
        ikkt_constraint_names, baseline_constraints = (
            _run_dual_projection_constraint_gradients(
                level,
                baseline["cfg_dot"],
                baseline["state"],
                active_names,
                required=True,
                artifact_records=baseline_artifacts,
                verbose=False,
            )
        )
        lambda_bounds = _extract_constraint_signs(
            cfg_level,
            ikkt_constraint_names,
        )
        fit_diagnostics = {}
        _, fixed_lambda = _compute_ikkt_residual_vector(
            baseline_obj,
            baseline_constraints,
            lambda_bounds=lambda_bounds,
            strict=True,
            verbose=False,
            diagnostics=fit_diagnostics,
        )
        fixed_lambda = np.asarray(fixed_lambda, dtype=float).reshape(-1)
        if not np.all(np.isfinite(fixed_lambda)):
            raise RuntimeError("Exact FFD baseline IKKT multipliers are non-finite")
        ikkt_metadata = {
            "insertion_step": int(insertion_step),
            "insertion_target": int(insertion_target),
            "constraint_names": list(ikkt_constraint_names),
            "constraint_status": constraint_status,
            "sign_convention": "SLSQP_GE_RAW",
            "ffd_blending": blending,
            "progressive_thickness": {
                "enabled": bool(opts.get("ffd_thickness_enabled", False)),
                "included": False,
                "reason": opts.get(
                    "ffd_thickness_ikkt_exclusion_reason",
                    "excluded from FFD IKKT",
                ),
            },
            "fit_diagnostics": fit_diagnostics,
            "lambdas": fixed_lambda.tolist(),
            "lambda_lower": np.asarray(lambda_bounds[0], dtype=float).tolist(),
            "lambda_upper": np.asarray(lambda_bounds[1], dtype=float).tolist(),
            "objective_gradient": baseline_obj.tolist(),
            "constraint_gradients": {
                name: np.asarray(values, dtype=float).tolist()
                for name, values in zip(
                    ikkt_constraint_names,
                    baseline_constraints,
                )
            },
        }
        _remove_artifact_path(baseline["mesh"])
        for artifact in baseline_artifacts:
            _remove_artifact_path(artifact["path"])

    candidates = []
    best_candidate = None
    for candidate_number, raw in enumerate(raw_candidates):
        side = str(raw["side"]).upper()
        x = float(raw["x"])
        if side not in active_by_side:
            raise ValueError(f"Unexpected inactive FFD candidate side {side!r}")
        candidate_active = {
            active_side: list(values)
            for active_side, values in active_by_side.items()
        }
        candidate_active[side].append(x)

        variant = _prepare_active_projection_variant(
            level,
            opts,
            cfg_level,
            real_dot_cfg,
            mesh_src,
            candidate_active,
            (
                f"{blending_slug}_step_{int(insertion_step):03d}_candidate_"
                f"{candidate_number:04d}_{side.lower()}"
            ),
            verbose=False,
        )
        records = variant["records"]
        candidate_dv_index = _dual_record_index(records, side, x)
        projection_artifacts = []
        obj_gradient = _finite_gradient(
            _run_ffd_dot_for_function(
                level.workdir,
                variant["cfg_dot"],
                variant["state"],
                obj_name,
                verbose=False,
            ),
            len(records),
            f"{obj_name} {blending} candidate {candidate_number}",
        )
        projection_artifacts.append(
            {
                "kind": "DOT",
                "function": obj_name,
                "path": _ffd_dot_artifact_directory(
                    level.workdir,
                    variant["cfg_dot"],
                    obj_name,
                ),
            }
        )

        constraint_gradients = []
        if indicator_mode == "IKKT":
            if ikkt_constraint_names:
                names, constraint_gradients = (
                    _run_dual_projection_constraint_gradients(
                        level,
                        variant["cfg_dot"],
                        variant["state"],
                        ikkt_constraint_names,
                        required=True,
                        artifact_records=projection_artifacts,
                        verbose=False,
                    )
                )
                if names != ikkt_constraint_names:
                    raise RuntimeError(
                        "Exact FFD candidate constraint order changed during "
                        "scoring"
                    )
            indicators = _ikkt_indicators_with_fixed_lambdas(
                obj_gradient,
                constraint_gradients,
                fixed_lambda,
            )
        else:
            indicators = np.asarray(
                _indicator_from_gradient(
                    obj_gradient,
                    cfg_level,
                    indicator_mode,
                    verbose=False,
                ),
                dtype=float,
            )

        indicator = float(indicators[candidate_dv_index])
        if not np.isfinite(indicator):
            raise RuntimeError(
                f"Exact FFD candidate {candidate_number} indicator is non-finite"
            )
        control_index = variant["column_index_by_side"][side]
        control_index = next(
            int(value)
            for station, value in control_index.items()
            if abs(float(station) - x) <= 1.0e-10
        )
        candidate = {
            "insertion_step": int(insertion_step),
            "insertion_target": int(insertion_target),
            "ndv_before_insertion": (
                sum(len(values) for values in active_by_side.values())
            ),
            "ndv_after_insertion": (
                sum(len(values) for values in active_by_side.values()) + 1
            ),
            "candidate_number": int(candidate_number),
            "side": side,
            "x": x,
            "grad": float(obj_gradient[candidate_dv_index]),
            "indicator": indicator,
            "interval_id": raw["interval_id"],
            "interval_left": float(raw["interval_left"]),
            "interval_right": float(raw["interval_right"]),
            "sample_index": int(raw["sample_index"]),
            "sample_fraction": float(raw["sample_fraction"]),
            "candidate_dv_index": int(candidate_dv_index),
            "control_point_i": control_index,
            "temporary_mesh": variant["mesh"],
            "projection_artifacts": projection_artifacts,
            "objective_function": obj_name,
            "objective_gradient_component": float(
                obj_gradient[candidate_dv_index]
            ),
            "constraint_gradient_components": {
                name: float(values[candidate_dv_index])
                for name, values in zip(
                    ikkt_constraint_names,
                    constraint_gradients,
                )
            },
            "scoring_basis": "EXACT_SEQUENTIAL_INSERTION",
            "ffd_scoring_mode": COMPONENT,
            "signal_source": (
                "IKKT_COMPONENT_RESIDUAL"
                if indicator_mode == "IKKT"
                else "OBJECTIVE_COMPONENT_GRADIENT"
            ),
            "ffd_blending": blending,
            "progressive_thickness_ikkt_included": False,
            "progressive_thickness_ikkt_exclusion_reason": opts.get(
                "ffd_thickness_ikkt_exclusion_reason",
                "excluded from FFD IKKT",
            ),
            "rejected_reason": raw.get("rejected_reason", ""),
            "nearest_center_or_boundary": raw.get(
                "nearest_center_or_boundary", ""
            ),
            "nearest_distance": raw.get("nearest_distance", ""),
            "required_spacing": raw.get("required_spacing", ""),
        }
        candidates.append(candidate)
        if (
            best_candidate is None
            or _exact_candidate_rank_key(candidate)
            < _exact_candidate_rank_key(best_candidate)
        ):
            if best_candidate is not None:
                _remove_candidate_artifacts(best_candidate)
            best_candidate = candidate
        else:
            _remove_candidate_artifacts(candidate)

    reduced_candidates = _reduce_candidates_to_interval_best(
        candidates,
        verbose=False,
    )
    ranked_candidates = sorted(
        reduced_candidates,
        key=_exact_candidate_rank_key,
    )
    for rank, candidate in enumerate(ranked_candidates, start=1):
        candidate["rank"] = rank
        candidate["interval_winner"] = True
    if best_candidate is None or ranked_candidates[0] is not best_candidate:
        raise RuntimeError(
            "Exact FFD artifact retention disagrees with candidate ranking"
        )
    for candidate in candidates:
        candidate.setdefault("interval_winner", False)

    _print_exact_candidate_ranking(
        ranked_candidates,
        insertion_step=insertion_step,
        insertion_target=insertion_target,
    )
    return {
        "candidates": ranked_candidates,
        "raw_candidates": candidates,
        "scoring_basis": "EXACT_SEQUENTIAL_INSERTION",
        "ffd_blending": blending,
        "ikkt_constraint_names": ikkt_constraint_names,
        "ikkt_lambdas": fixed_lambda.tolist(),
        "ikkt_metadata": ikkt_metadata,
        "selected_candidate": best_candidate,
    }


def _compute_dual_bezier_exact_candidate_scores(
    level,
    opts,
    cfg_level,
    raw_candidates,
    active_by_side,
    accepted_mesh=None,
):
    """Run exact sequential scoring for BEZIER or BSPLINE_UNIFORM FFD."""

    blending = normalize_ffd_blending(opts.get("ffd_blending", BEZIER))
    scoring_mode = normalize_ffd_scoring_mode(
        opts.get("ffd_scoring_mode", COMPONENT)
    )
    _cleanup_exact_scoring_artifacts(level, opts)
    try:
        insertion_target, interval_count = _exact_growth_ratio_insertion_target(
            level,
            raw_candidates,
            opts,
        )
        if insertion_target <= 0:
            return {
                "candidates": [],
                "raw_candidates": [],
                "active_scores": [],
                "active_upper_scores": [],
                "active_lower_scores": [],
                "sequential_selection": True,
                "scoring_basis": "EXACT_SEQUENTIAL_INSERTION",
                "ffd_scoring_mode": scoring_mode,
                "ffd_blending": blending,
                "selected_candidates": [],
                "selected_artifact_directory": None,
                "candidate_scores_csv": None,
                "insertion_target": 0,
                "insertions_completed": 0,
            }

        indicator_mode = str(opts.get("adaptive_indicator", "ABS_GRAD")).upper()
        virtual_context = None
        if scoring_mode == VIRTUAL_TANGENT:
            obj_name = str(cfg_level.get("OBJECTIVE_FUNCTION", "DRAG")).upper()
            _obj_adj_dir, design_dir = _find_real_adjoint_assets(
                level.workdir,
                obj_name,
            )
            virtual_context = _prepare_virtual_tangent_context(
                level,
                opts,
                cfg_level,
                obj_name,
                design_dir,
            )
        configured_nadd_mode = str(
            opts.get("nadd_mode", "GROWTH_RATIO")
        ).upper()
        if configured_nadd_mode != "GROWTH_RATIO":
            print(
                "[PROGRESSIVE_FFD_DUAL] Sequential exact FFD selection | "
                f"blending={blending} "
                f"using=GROWTH_RATIO configured={configured_nadd_mode}"
            )
        print(
            "[PROGRESSIVE_FFD_DUAL] Entering adaptive scoring phase | "
            f"level={level.level_id} indicator={indicator_mode} "
            f"scoring_mode={scoring_mode} "
            f"blending={blending} "
            f"intervals={interval_count} "
            f"samples_per_interval={int(opts.get('candidate_samples', 1))} "
            f"growth_ratio={float(opts.get('growth_ratio', 2.0)):.6g} "
            f"target_insertions={insertion_target}"
        )

        current_active = {
            side: sorted(float(x) for x in active_by_side[side])
            for side in ("UPPER", "LOWER")
            if side in active_by_side
        }
        selected_candidates = []
        all_candidates = []
        selected_root = os.path.join(
            level.workdir,
            "FFD_SELECTED_CANDIDATE",
        )
        if insertion_target > 1:
            os.makedirs(selected_root)

        current_raw_candidates = list(raw_candidates)
        for insertion_step in range(1, insertion_target + 1):
            if insertion_step > 1:
                current_raw_candidates = _generate_dual_exact_candidates(
                    current_active,
                    opts,
                )
            if not current_raw_candidates:
                print(
                    "[PROGRESSIVE_FFD_DUAL] Sequential scoring stopped | "
                    f"completed={len(selected_candidates)} "
                    f"target={insertion_target} reason=NO_VALID_CANDIDATES"
                )
                break

            if scoring_mode == VIRTUAL_TANGENT:
                pass_result = _compute_dual_virtual_tangent_candidate_scores_impl(
                    level,
                    opts,
                    cfg_level,
                    current_raw_candidates,
                    current_active,
                    virtual_context,
                    accepted_mesh=accepted_mesh,
                    insertion_step=insertion_step,
                    insertion_target=insertion_target,
                )
            else:
                pass_result = _compute_dual_bezier_exact_candidate_scores_impl(
                    level,
                    opts,
                    cfg_level,
                    current_raw_candidates,
                    current_active,
                    accepted_mesh=accepted_mesh,
                    insertion_step=insertion_step,
                    insertion_target=insertion_target,
                )
            selected = pass_result["selected_candidate"]
            side = str(selected["side"]).upper()
            selected_x = float(selected["x"])
            if any(
                abs(selected_x - float(active_x)) <= 1.0e-10
                for active_x in current_active[side]
            ):
                raise RuntimeError(
                    "Sequential exact FFD scoring selected an already-active "
                    f"station: side={side}, x={selected_x:.16g}"
                )
            if insertion_target == 1:
                selected_dir = None
            else:
                selected_dir = os.path.join(
                    selected_root,
                    f"insertion_{insertion_step:03d}",
                )
            selected_dir = _persist_selected_candidate_artifacts(
                level,
                selected,
                indicator_mode,
                pass_result["ikkt_metadata"],
                selected_dir=selected_dir,
            )
            selected["artifact_directory"] = selected_dir
            selected_candidates.append(selected)
            all_candidates.extend(pass_result["raw_candidates"])

            current_active[side] = sorted(
                current_active[side] + [selected_x]
            )
            print(
                "[PROGRESSIVE_FFD_DUAL] Selected candidate | "
                f"insertion={insertion_step}/{insertion_target} "
                f"rank={int(selected.get('rank', 1))} "
                f"side={side} "
                f"interval=[{float(selected['interval_left']):.6f},"
                f"{float(selected['interval_right']):.6f}] "
                f"x={float(selected['x']):.6f} "
                f"indicator={float(selected['indicator']):.6e} "
                f"ndv_before={int(selected['ndv_before_insertion'])} "
                f"ndv_after={int(selected['ndv_after_insertion'])}"
            )

        if not selected_candidates:
            raise RuntimeError(
                "Sequential exact FFD scoring did not select any candidate"
            )

        scores_csv = _write_exact_candidate_scores_csv(level, all_candidates)
        return {
            "candidates": selected_candidates,
            "raw_candidates": all_candidates,
            "active_scores": [],
            "active_upper_scores": [],
            "active_lower_scores": [],
            "sequential_selection": True,
            "scoring_basis": (
                "VIRTUAL_TANGENT_SPACE"
                if scoring_mode == VIRTUAL_TANGENT
                else "EXACT_SEQUENTIAL_INSERTION"
            ),
            "ffd_scoring_mode": scoring_mode,
            "ffd_blending": blending,
            "selected_candidate": selected_candidates[0],
            "selected_candidates": selected_candidates,
            "selected_artifact_directory": selected_root,
            "candidate_scores_csv": scores_csv,
            "insertion_target": insertion_target,
            "insertions_completed": len(selected_candidates),
            "active_by_side_after": current_active,
        }
    except Exception:
        _cleanup_exact_scoring_artifacts(level, opts)
        raise


def _compute_dual_exact_candidate_scores(*args, **kwargs):
    """Generic entry point for exact BEZIER/BSPLINE_UNIFORM scoring."""

    return _compute_dual_bezier_exact_candidate_scores(*args, **kwargs)


def _compute_dual_ffd_dot_candidate_scores(level, opts, mesh_source=None):
    cfg_path = os.path.join(level.workdir, level.config_filename)
    cfg_level = SU2.io.Config(cfg_path)

    nsamples = int(opts.get("candidate_samples", 1))
    min_spacing = float(opts.get("min_center_spacing", 0.0))
    candidate_xmin, candidate_xmax = ffd_active_range_from_opts(opts)
    exact_sequential_scoring = normalize_ffd_blending(
        opts.get("ffd_blending", BEZIER)
    ) in (BEZIER, BSPLINE_UNIFORM)
    active_by_side = {
        side: sorted(float(x) for x in columns)
        for side, columns in level.columns_by_side.items()
        if side in ("UPPER", "LOWER")
    }
    if not active_by_side:
        raise RuntimeError("Exact FFD scoring requires an active UPPER/LOWER side")

    raw_candidates = []
    for side in ("UPPER", "LOWER"):
        if side not in active_by_side:
            continue
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
            compare_candidates=not exact_sequential_scoring,
            verbose=not exact_sequential_scoring,
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

    if exact_sequential_scoring:
        return _compute_dual_exact_candidate_scores(
            level,
            opts,
            cfg_level,
            raw_candidates,
            active_by_side,
            accepted_mesh=mesh_source,
        )

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
        envelope_spec=opts.get("ffd_envelope_spec"),
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
        requested_names = _extract_constraint_names(cfg_level)
        constraint_names, _ = _select_active_ikkt_constraints(
            cfg_level,
            requested_names,
            design_dir=design_dir,
            active_tol=opts.get("ikkt_active_tol", 1.0e-6),
        )
        included_names = []
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
            included_names.append(constraint_name)

        lambda_bounds = _extract_constraint_signs(cfg_level, included_names)
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
