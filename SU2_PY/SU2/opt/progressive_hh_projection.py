#!/usr/bin/env python

import os
import copy
import csv
import glob
import json
import math
import shutil
import contextlib

import numpy as np
import SU2
from scipy.optimize import lsq_linear

from SU2.opt.bspline_driver.geometry_constraints import (
    is_geometry_constraint_name,
    normalize_geometry_constraint_name,
)
from SU2.opt.progressive_ffd_tangent import (
    airfoil_area_value_and_field,
    airfoil_thickness_value_and_field,
    fit_surface_ikkt_signal,
    internal_constraint_field,
)
from SU2.opt.progressive_hh_tangent import (
    COMPONENT,
    VIRTUAL_TANGENT,
    HHTangentError,
    build_hh_tangent_state,
    compare_tangent_spaces,
    load_design_gradient_csv,
    load_surface_sensitivity_vector,
    normalize_hh_scoring_mode,
    project_surface_field,
    validate_surface_projection,
)
from SU2.opt.progressive_surface_scoring import (
    build_surface_scoring_view,
    mask_surface_constraint_records,
    mask_surface_field,
)
from SU2.opt.progressive_design import read_ranking_design_directory


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


def _check_min_center_spacing(
    side,
    x,
    active_centers_by_side,
    selected_candidates_by_side,
    min_spacing,
):
    min_spacing = float(min_spacing)
    if min_spacing <= 0.0:
        return {
            "accepted": True,
            "nearest": None,
            "nearest_distance": None,
            "rejected_reason": "",
        }

    side = str(side)
    x = float(x)
    reference_values = []
    reference_values.extend(
        float(v) for v in active_centers_by_side.get(side, [])
    )
    reference_values.extend(
        float(v) for v in selected_candidates_by_side.get(side, [])
    )
    reference_values.extend([0.0, 1.0])

    nearest = None
    nearest_distance = None
    for value in reference_values:
        dist = abs(x - value)
        if nearest_distance is None or dist < nearest_distance:
            nearest = value
            nearest_distance = dist

    if nearest_distance is not None and nearest_distance < min_spacing:
        return {
            "accepted": False,
            "nearest": nearest,
            "nearest_distance": nearest_distance,
            "rejected_reason": "MIN_CENTER_SPACING",
        }

    return {
        "accepted": True,
        "nearest": nearest,
        "nearest_distance": nearest_distance,
        "rejected_reason": "",
    }


def _filter_candidates_by_min_spacing(
    candidates,
    active_centers_by_side,
    min_spacing,
):
    min_spacing = float(min_spacing)
    if min_spacing <= 0.0:
        return candidates

    selected_candidates_by_side = {}
    filtered = []

    for c in candidates:
        check = _check_min_center_spacing(
            c["side"],
            c["x"],
            active_centers_by_side,
            selected_candidates_by_side,
            min_spacing,
        )
        if check["accepted"]:
            filtered.append(c)
            continue

        c["rejected_reason"] = check["rejected_reason"]
        c["nearest_center_or_boundary"] = check["nearest"]
        c["nearest_distance"] = check["nearest_distance"]
        c["required_spacing"] = min_spacing
        print(
            "[PROGRESSIVE_HH] Candidate rejected by min spacing | "
            f"side={c['side']} x={float(c['x']):.6f} "
            f"nearest={float(check['nearest']):.6f} "
            f"dist={float(check['nearest_distance']):.6f} "
            f"required={min_spacing:.6f}"
        )

    return filtered


def _is_symmetric_reduced(opts):
    return str(opts.get("symmetry_mode", "NONE")).upper() == "REDUCED"


def _assert_symmetric_centers(upper, lower, tol=1.0e-12):
    if len(upper) != len(lower):
        raise ValueError(
            "Symmetric reduced HH requires upper/lower center lists with "
            f"the same length: {len(upper)} vs {len(lower)}"
        )
    for i, (xu, xl) in enumerate(zip(upper, lower)):
        if abs(float(xu) - float(xl)) > tol:
            raise ValueError(
                "Symmetric reduced HH requires identical upper/lower centers; "
                f"index {i}: upper={xu}, lower={xl}, tol={tol}"
            )


def _reduce_pair_values(upper_values, lower_values, sign):
    if len(upper_values) != len(lower_values):
        raise ValueError(
            "Symmetric reduced value size mismatch: "
            f"{len(upper_values)} vs {len(lower_values)}"
        )
    sign = float(sign)
    return [
        float(gu) + sign * float(gl)
        for gu, gl in zip(upper_values, lower_values)
    ]


def _find_real_adjoint_assets(level_dir, func_name):
    func_name = str(func_name).upper()

    adjoint_folder_name = f"ADJOINT_{func_name}"

    ranking_design = read_ranking_design_directory(level_dir)
    if ranking_design is not None:
        adjoint_dir = os.path.join(ranking_design, adjoint_folder_name)
        if not os.path.isdir(adjoint_dir):
            raise FileNotFoundError(
                "The accepted/converged ranking DSN lacks a required adjoint; "
                "refusing to borrow one from another design: "
                f"{adjoint_dir}"
            )
        return adjoint_dir, ranking_design

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
    ranking_design = read_ranking_design_directory(level_dir)
    if ranking_design is not None:
        designs = [ranking_design]
    else:
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



def _dot_problem_kind(cfg):
    math_problem = str(cfg.get("MATH_PROBLEM", "")).upper()
    gradient_method = str(cfg.get("GRADIENT_METHOD", "")).upper()

    if "CONTINUOUS" in math_problem or "CONTINUOUS" in gradient_method:
        return "CONTINUOUS_ADJOINT"

    return "DISCRETE_ADJOINT"


def _normalize_adjoint_kind(value):
    value = str(value or "").upper()
    if "CONTINUOUS" in value:
        return "CONTINUOUS_ADJOINT"
    if "DISCRETE" in value:
        return "DISCRETE_ADJOINT"
    return None


def _requested_dot_kind_from_level_config(cfg_level):
    """
    Automatic DOT adjoint selection.

    No progressive-specific flag is required: the DOT projection uses the
    adjoint family requested by the level config itself.  The primary source
    is GRADIENT_METHOD; MATH_PROBLEM is used as a fallback.
    """
    for key in ("GRADIENT_METHOD", "MATH_PROBLEM"):
        if key in cfg_level:
            kind = _normalize_adjoint_kind(cfg_level.get(key, ""))
            if kind is not None:
                return kind
    return None


def _select_dot_config_path(obj_adj_dir, cfg_level):
    """
    Select the DOT config automatically from the adjoint method.

    DISCRETE_ADJOINT   -> config_DOT_AD.cfg
    CONTINUOUS_ADJOINT -> config_DOT.cfg
    """
    requested = _requested_dot_kind_from_level_config(cfg_level)

    path_discrete = os.path.join(obj_adj_dir, "config_DOT_AD.cfg")
    path_continuous = os.path.join(obj_adj_dir, "config_DOT.cfg")

    if requested == "DISCRETE_ADJOINT":
        if not os.path.isfile(path_discrete):
            raise FileNotFoundError(
                "GRADIENT_METHOD requests DISCRETE_ADJOINT, but "
                f"config_DOT_AD.cfg was not found in {obj_adj_dir}"
            )
        return path_discrete, requested

    if requested == "CONTINUOUS_ADJOINT":
        if not os.path.isfile(path_continuous):
            raise FileNotFoundError(
                "GRADIENT_METHOD requests CONTINUOUS_ADJOINT, but "
                f"config_DOT.cfg was not found in {obj_adj_dir}"
            )
        return path_continuous, requested

    available = []
    if os.path.isfile(path_discrete):
        available.append((path_discrete, "DISCRETE_ADJOINT"))
    if os.path.isfile(path_continuous):
        available.append((path_continuous, "CONTINUOUS_ADJOINT"))

    if len(available) == 1:
        return available[0]

    if not available:
        raise FileNotFoundError(
            "No DOT config found in "
            f"{obj_adj_dir}. Tried: {path_discrete}, {path_continuous}"
        )

    raise RuntimeError(
        "Both config_DOT_AD.cfg and config_DOT.cfg are present, but the level "
        "config does not specify GRADIENT_METHOD/MATH_PROBLEM clearly. "
        "Set GRADIENT_METHOD to DISCRETE_ADJOINT or CONTINUOUS_ADJOINT."
    )


def _build_extended_dot_config(cfg_level, real_dot_cfg, mesh_name, all_upper, all_lower):
    # Start from the real DOT config written by SU2, not from the level config.
    # This preserves the correct continuous/discrete adjoint settings and the
    # file names that SU2_DOT expects in the adjoint design directory.
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


def _extract_constraint_specs(cfg, constraint_names=None):
    """Return normalized SU2 constraint metadata in the requested order."""

    opt_con = cfg.get("OPT_CONSTRAINT", {})
    specs_by_name = {}
    if isinstance(opt_con, dict):
        for group_name in ("EQUALITY", "INEQUALITY"):
            group = opt_con.get(group_name, {}) or {}
            if not isinstance(group, dict):
                continue
            for raw_name, raw_spec in group.items():
                name = str(raw_name).strip().upper()
                if not name or not isinstance(raw_spec, dict):
                    continue
                try:
                    target = float(raw_spec.get("VALUE"))
                except (TypeError, ValueError):
                    target = None
                try:
                    scale = float(raw_spec.get("SCALE", 1.0))
                except (TypeError, ValueError):
                    scale = 1.0
                specs_by_name[name] = {
                    "name": name,
                    "sign": str(raw_spec.get("SIGN", "")).strip(),
                    "target": target,
                    "scale": scale,
                    "group": group_name,
                }
    elif isinstance(opt_con, str):
        for raw_part in opt_con.split(";"):
            part = raw_part.strip()
            if not part:
                continue
            expression, separator, raw_scale = part.partition("*")
            try:
                scale = float(raw_scale) if separator else 1.0
            except ValueError:
                scale = 1.0
            expression = expression.strip().strip("() ")
            sign = next((item for item in ("<", ">", "=") if item in expression), "")
            if not sign:
                continue
            lhs, rhs = expression.split(sign, 1)
            name = lhs.strip().upper()
            if not name:
                continue
            try:
                target = float(rhs.strip())
            except ValueError:
                target = None
            specs_by_name[name] = {
                "name": name,
                "sign": sign,
                "target": target,
                "scale": scale,
                "group": "EQUALITY" if sign == "=" else "INEQUALITY",
            }

    names = (
        _extract_constraint_names(cfg)
        if constraint_names is None
        else [str(name).strip().upper() for name in constraint_names]
    )
    return [
        specs_by_name.get(
            name,
            {
                "name": name,
                "sign": "",
                "target": None,
                "scale": 1.0,
                "group": "UNKNOWN",
            },
        )
        for name in names
    ]


def _latest_design_function_values(design_dir):
    """Load the function values associated with the adjoint baseline design."""

    design_file = os.path.join(str(design_dir), "design.pkl")
    if not os.path.exists(design_file):
        return {}, None
    try:
        design = SU2.io.load_data(design_file)
        values = getattr(getattr(design, "state", None), "FUNCTIONS", {})
        return {
            str(name).strip().upper(): float(value)
            for name, value in values.items()
            if np.isfinite(float(value))
        }, design_file
    except Exception as exc:
        return {}, f"{design_file}: {exc}"


def _select_active_ikkt_constraints(
    cfg,
    constraint_names,
    design_dir=None,
    current_values=None,
    active_tol=1.0e-6,
    verbose=True,
):
    """Classify configured constraints for the FFD/HH IKKT active set."""

    active_tol = float(active_tol)
    if active_tol < 0.0:
        raise ValueError("PROGRESSIVE_HH_IKKT_ACTIVE_TOL must be non-negative")

    value_source = "provided"
    if current_values is None:
        current_values, value_source = _latest_design_function_values(design_dir)
    normalized_values = {
        str(name).strip().upper(): float(value)
        for name, value in (current_values or {}).items()
    }

    active_names = []
    records = []
    for spec in _extract_constraint_specs(cfg, constraint_names):
        name = spec["name"]
        sign = spec["sign"]
        target = spec["target"]
        value = normalized_values.get(name)
        c_value = None
        if value is not None and target is not None:
            if sign == ">":
                c_value = float(value) - float(target)
            elif sign == "<":
                c_value = float(target) - float(value)
            elif sign == "=":
                c_value = float(value) - float(target)

        if sign == "=":
            status = "active_equality"
            included = True
        elif sign in (">", "<") and c_value is not None:
            included = bool(c_value <= active_tol)
            status = "active_inequality" if included else "inactive"
        else:
            # Preserve compatibility for incomplete/mocked configs, while making
            # the missing active-set information explicit in diagnostics.
            included = True
            status = "active_status_unknown"

        if included:
            active_names.append(name)
        record = {
            **spec,
            "current_value": value,
            "c_value": c_value,
            "active_tol": active_tol,
            "status": status,
            "included": included,
            "value_source": value_source,
        }
        records.append(record)
        if verbose:
            gap_text = "unavailable" if c_value is None else f"{c_value:.6e}"
            print(
                "[PROGRESSIVE_IKKT] active set | "
                f"constraint={name} sign={sign or 'UNKNOWN'} "
                f"c={gap_text} status={status}"
            )

    return active_names, records


def _extract_constraint_signs(cfg, constraint_names):
    """
    Build multiplier bounds for ``residual = grad(J) - A @ lambda``.

    The columns of ``A`` are raw SU2 function gradients. Therefore a lower
    bound ``F > target`` uses lambda >= 0, while an upper bound
    ``F < target`` uses lambda <= 0. Equalities use a free multiplier.
    """
    lb = []
    ub = []
    for spec in _extract_constraint_specs(cfg, constraint_names):
        sign = spec["sign"]
        if sign == "<":
            lb.append(-np.inf)
            ub.append(0.0)
        elif sign == ">":
            lb.append(0.0)
            ub.append(np.inf)
        elif sign == "=":
            lb.append(-np.inf)
            ub.append(np.inf)
        else:
            lb.append(-np.inf)
            ub.append(np.inf)

    return np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)


def _flatten_indicator_numeric_values(value):
    """Flatten SU2 Config scalar/list/string numeric values into a flat list."""
    if value is None:
        return []

    if isinstance(value, str):
        raw = value.strip().strip("()[]")
        if not raw:
            return []
        raw = raw.replace(",", " ")
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
    """
    Indicator for one-sided DV bounds.

    ABS_GRAD ranks by sensitivity magnitude, assuming the optimizer can move a
    candidate coefficient in both directions. DESCENT_GRAD keeps only the
    gradient component compatible with the admissible coefficient direction:

      lower < 0 < upper  -> |g|
      lower >= 0         -> max(0, -g)
      upper <= 0         -> max(0,  g)
    """
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
        "[PROGRESSIVE_HH] DESCENT_GRAD indicator | "
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


def _ensure_dot_mesh_available(dot_test_dir, adjoint_dir, design_dir, level_dir, cfg_dot):
    """
    Make sure MESH_FILENAME is readable inside DOT_ONLY_<func>.

    Design folders often contain relative symlinks such as
      mesh.su2 -> ../mesh.su2
    After copying ADJOINT_<func> to DOT_ONLY_<func>, those links can point to
    the wrong directory. Copy the real mesh file into DOT_ONLY instead.
    """
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

    candidates = [
        os.path.join(adjoint_dir, mesh_name),
        os.path.join(design_dir, mesh_name),
        os.path.join(level_dir, mesh_name),
    ]

    mesh_src = None
    for candidate in candidates:
        if os.path.exists(candidate):
            mesh_src = candidate
            break

    if mesh_src is None:
        basename = os.path.basename(mesh_name)
        for root in (design_dir, adjoint_dir, level_dir):
            matches = glob.glob(os.path.join(root, "**", basename), recursive=True)
            matches = [m for m in matches if os.path.exists(m)]
            if matches:
                mesh_src = matches[0]
                break

    if mesh_src is None:
        raise FileNotFoundError(
            "Could not locate mesh for DOT projection. "
            f"MESH_FILENAME={mesh_name}, searched: {candidates}"
        )

    os.makedirs(os.path.dirname(mesh_dst), exist_ok=True)
    shutil.copy2(mesh_src, mesh_dst, follow_symlinks=True)
    print(
        "[PROGRESSIVE_HH] DOT projection | copied mesh into DOT_ONLY: "
        f"{mesh_src} -> {mesh_dst}"
    )


def _run_dot_for_function(level_dir, cfg_dot, state, func_name):
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
            "[PROGRESSIVE_HH] DOT projection | "
            "continuous adjoint: no solution_adj_*.dat restart required"
        )

    cfg_fun = SU2.io.Config(copy.deepcopy(dict(cfg_dot)))
    cfg_fun["OBJECTIVE_FUNCTION"] = func_name

    cwd = os.getcwd()
    try:
        os.chdir(dot_test_dir)

        try:
            cfg_fun.dump("config_DOT_PROGRESSIVE.cfg")
        except Exception:
            pass

        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                info = SU2.run.projection(cfg_fun, state)
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
        "[PROGRESSIVE_HH] DOT projection | "
        f"kind={dot_kind} function={func_name} "
        f"ndv={grad_arr.size} norm={np.linalg.norm(grad_arr):.6e}"
    )

    return grad


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


def _compute_ikkt_residual_vector(
    g_obj,
    constraint_grads,
    lambda_bounds=None,
    strict=False,
    verbose=True,
    diagnostics=None,
):
    g = np.asarray(g_obj, dtype=float)
    diagnostic_payload = diagnostics if diagnostics is not None else {}

    if not constraint_grads:
        diagnostic_payload.update(
            {
                "status": "no_active_constraints",
                "grad_objective_norm": float(np.linalg.norm(g)),
                "n_constraints": 0,
            }
        )
        return g.copy(), np.zeros(0)

    A = np.column_stack([np.asarray(cg, dtype=float) for cg in constraint_grads])
    diagnostic_payload.update(
        {
            "grad_objective_norm": float(np.linalg.norm(g)),
            "grad_constraint_norms": [
                float(np.linalg.norm(A[:, index])) for index in range(A.shape[1])
            ],
            "constraint_matrix_shape": [int(A.shape[0]), int(A.shape[1])],
            "constraint_matrix_rank": int(np.linalg.matrix_rank(A)),
            "constraint_matrix_condition": float(np.linalg.cond(A)),
            "n_constraints": int(A.shape[1]),
        }
    )

    try:
        if lambda_bounds is None:
            lb = np.full(A.shape[1], -np.inf)
            ub = np.full(A.shape[1], np.inf)
            res = lsq_linear(A, g, bounds=(lb, ub), lsmr_tol="auto")
        else:
            lb, ub = lambda_bounds
            res = lsq_linear(A, g, bounds=(lb, ub), lsmr_tol="auto")

        if not bool(res.success):
            raise RuntimeError(
                f"IKKT multiplier least-squares solve failed: {res.message}"
            )

        lam = res.x
        residual = g - A @ lam
        diagnostic_payload.update(
            {
                "status": "ok",
                "cost": float(res.cost),
                "optimality": float(res.optimality),
                "active_mask": [int(value) for value in res.active_mask.tolist()],
                "residual_norm": float(np.linalg.norm(residual)),
                "relative_residual_norm": float(np.linalg.norm(residual))
                / max(float(np.linalg.norm(g)), 1.0e-16),
            }
        )

        if verbose:
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

    except Exception as exc:
        diagnostic_payload.update({"status": "failed", "error": str(exc)})
        if strict:
            raise
        residual = g.copy()
        lam = np.zeros(A.shape[1])

    return residual, lam


def _compute_component_dot_candidate_scores_symmetric(level, opts):
    cfg_path = os.path.join(level.workdir, level.config_filename)
    cfg_level = SU2.io.Config(cfg_path)

    active_upper = list(level.upper)
    active_lower = list(level.lower)
    _assert_symmetric_centers(active_upper, active_lower)
    pair_centers = active_upper

    nsamples = int(opts.get("candidate_samples", 1))
    min_spacing = float(opts.get("min_center_spacing", 0.0))
    sign = float(opts.get("symmetry_sign", -1.0))

    print(
        "[PROGRESSIVE_HH] Candidate sampling | "
        f"samples={nsamples} min_spacing={min_spacing:.6f} symmetry=REDUCED"
    )

    cand_pair_raw = get_midpoint_candidates(pair_centers, nsamples=nsamples)
    for c in cand_pair_raw:
        c["side"] = "PAIR"

    active_centers_by_side = {"PAIR": pair_centers}
    cand_pair_raw = _filter_candidates_by_min_spacing(
        cand_pair_raw,
        active_centers_by_side,
        min_spacing,
    )

    cand_pair = [c["x"] for c in cand_pair_raw]

    if not cand_pair:
        if min_spacing > 0.0:
            print(
                "[PROGRESSIVE_HH] No valid pair candidates remain after "
                "min-spacing filtering."
            )
        return {
            "candidates": [],
            "spacing_filtered_empty": min_spacing > 0.0,
            "active_upper_scores": [],
            "active_lower_scores": [],
            "active_pair_scores": [],
        }

    cand_upper = list(cand_pair)
    cand_lower = list(cand_pair)
    all_upper = active_upper + cand_upper
    all_lower = active_lower + cand_lower

    obj_name = str(cfg_level.get("OBJECTIVE_FUNCTION", "DRAG")).upper()
    obj_adj_dir, design_dir = _find_real_adjoint_assets(level.workdir, obj_name)

    real_dot_cfg_path, dot_kind = _select_dot_config_path(obj_adj_dir, cfg_level)

    print(
        "[PROGRESSIVE_HH] Using DOT config: "
        f"{real_dot_cfg_path} | kind={dot_kind}"
    )

    real_dot_cfg = SU2.io.Config(real_dot_cfg_path)
    mesh_name = str(real_dot_cfg["MESH_FILENAME"])

    cfg_dot = _build_extended_dot_config(
        cfg_level, real_dot_cfg, mesh_name, all_upper, all_lower
    )
    state = _make_projection_state(mesh_name)

    grad_obj = _run_dot_for_function(level.workdir, cfg_dot, state, obj_name)

    n_pair_active = len(pair_centers)
    n_pair_candidate = len(cand_pair)
    n_active = 2 * n_pair_active
    expected_ncand = 2 * n_pair_candidate

    i_upper_active_0 = 0
    i_upper_candidate_0 = i_upper_active_0 + n_pair_active
    i_lower_active_0 = i_upper_candidate_0 + n_pair_candidate
    i_lower_candidate_0 = i_lower_active_0 + n_pair_active

    grad_upper_active = grad_obj[i_upper_active_0:i_upper_candidate_0]
    grad_upper_candidate = grad_obj[i_upper_candidate_0:i_lower_active_0]
    grad_lower_active = grad_obj[i_lower_active_0:i_lower_candidate_0]
    grad_lower_candidate = grad_obj[i_lower_candidate_0:]

    if len(grad_upper_candidate) + len(grad_lower_candidate) == 0:
        raise RuntimeError("DOT returned empty candidate gradient")

    if len(grad_upper_candidate) + len(grad_lower_candidate) != expected_ncand:
        raise RuntimeError(
            "DOT candidate gradient size mismatch: "
            f"got {len(grad_upper_candidate) + len(grad_lower_candidate)}, "
            f"expected {expected_ncand}"
        )

    active_pair_grad = _reduce_pair_values(
        grad_upper_active,
        grad_lower_active,
        sign,
    )
    candidate_pair_grad = _reduce_pair_values(
        grad_upper_candidate,
        grad_lower_candidate,
        sign,
    )

    indicator_mode = opts.get("adaptive_indicator", "ABS_GRAD").upper()

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
            included_names.append(cname)

        lambda_bounds = _extract_constraint_signs(cfg_level, included_names)
        residual_full, lam = _compute_ikkt_residual_vector(
            grad_obj,
            constraint_grads_full,
            lambda_bounds=lambda_bounds,
        )

        residual_upper_active = residual_full[i_upper_active_0:i_upper_candidate_0]
        residual_upper_candidate = residual_full[i_upper_candidate_0:i_lower_active_0]
        residual_lower_active = residual_full[i_lower_active_0:i_lower_candidate_0]
        residual_lower_candidate = residual_full[i_lower_candidate_0:]

        active_pair_indicator = np.abs(
            np.asarray(
                _reduce_pair_values(
                    residual_upper_active,
                    residual_lower_active,
                    sign,
                ),
                dtype=float,
            )
        ).tolist()
        candidate_pair_indicator = np.abs(
            np.asarray(
                _reduce_pair_values(
                    residual_upper_candidate,
                    residual_lower_candidate,
                    sign,
                ),
                dtype=float,
            )
        ).tolist()
    else:
        active_pair_indicator = _indicator_from_gradient(
            active_pair_grad,
            cfg_level,
            indicator_mode,
        )
        candidate_pair_indicator = _indicator_from_gradient(
            candidate_pair_grad,
            cfg_level,
            indicator_mode,
        )

    candidates = []
    for k, c in enumerate(cand_pair_raw):
        upper_indicator = abs(float(grad_upper_candidate[k]))
        lower_indicator = abs(float(grad_lower_candidate[k]))
        candidates.append(
            {
                "side": "PAIR",
                "x": float(c["x"]),
                "grad": float(candidate_pair_grad[k]),
                "indicator": float(candidate_pair_indicator[k]),
                "upper_grad": float(grad_upper_candidate[k]),
                "lower_grad": float(grad_lower_candidate[k]),
                "upper_indicator": float(upper_indicator),
                "lower_indicator": float(lower_indicator),
                "interval_id": c["interval_id"],
                "interval_left": float(c["interval_left"]),
                "interval_right": float(c["interval_right"]),
                "sample_index": int(c["sample_index"]),
                "sample_fraction": float(c["sample_fraction"]),
                "symmetry_mode": "REDUCED",
                "upper_center": float(c["x"]),
                "lower_center": float(c["x"]),
            }
        )

    print(
        f"[PROGRESSIVE_HH] Candidate scoring | mode={indicator_mode} "
        f"active_ndv={n_active} candidate_ndv={expected_ncand} "
        f"pair_candidates={n_pair_candidate} samples={nsamples}"
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
        "active_upper_scores": active_pair_indicator,
        "active_lower_scores": active_pair_indicator,
        "active_pair_scores": active_pair_indicator,
    }


def _compute_component_dot_candidate_scores(level, opts):
    if _is_symmetric_reduced(opts):
        return _compute_component_dot_candidate_scores_symmetric(level, opts)

    cfg_path = os.path.join(level.workdir, level.config_filename)
    cfg_level = SU2.io.Config(cfg_path)

    active_upper = list(level.upper)
    active_lower = list(level.lower)
    nsamples = int(opts.get("candidate_samples", 1))
    min_spacing = float(opts.get("min_center_spacing", 0.0))

    print(
        "[PROGRESSIVE_HH] Candidate sampling | "
        f"samples={nsamples} min_spacing={min_spacing:.6f}"
    )

    cand_upper_raw = get_midpoint_candidates(active_upper, nsamples=nsamples)
    cand_lower_raw = get_midpoint_candidates(active_lower, nsamples=nsamples)
    for c in cand_upper_raw:
        c["side"] = "UPPER"
    for c in cand_lower_raw:
        c["side"] = "LOWER"

    active_centers_by_side = {
        "UPPER": active_upper,
        "LOWER": active_lower,
    }
    raw_candidates = cand_upper_raw + cand_lower_raw
    raw_candidates = _filter_candidates_by_min_spacing(
        raw_candidates,
        active_centers_by_side,
        min_spacing,
    )
    cand_upper_raw = [c for c in raw_candidates if c["side"] == "UPPER"]
    cand_lower_raw = [c for c in raw_candidates if c["side"] == "LOWER"]

    cand_upper = [c["x"] for c in cand_upper_raw]
    cand_lower = [c["x"] for c in cand_lower_raw]

    if not cand_upper and not cand_lower:
        if min_spacing > 0.0:
            print(
                "[PROGRESSIVE_HH] No valid candidates remain after "
                "min-spacing filtering."
            )
        return {
            "candidates": [],
            "spacing_filtered_empty": min_spacing > 0.0,
            "active_upper_scores": [],
            "active_lower_scores": [],
        }

    all_upper = active_upper + cand_upper
    all_lower = active_lower + cand_lower

    obj_name = str(cfg_level.get("OBJECTIVE_FUNCTION", "DRAG")).upper()
    obj_adj_dir, design_dir = _find_real_adjoint_assets(level.workdir, obj_name)

    real_dot_cfg_path, dot_kind = _select_dot_config_path(obj_adj_dir, cfg_level)

    print(
        "[PROGRESSIVE_HH] Using DOT config: "
        f"{real_dot_cfg_path} | kind={dot_kind}"
    )

    real_dot_cfg = SU2.io.Config(real_dot_cfg_path)
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
            included_names.append(cname)

        lambda_bounds = _extract_constraint_signs(cfg_level, included_names)
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


def _find_hh_projection_mesh_source(mesh_name, adjoint_dir, design_dir, level_dir):
    mesh_name = str(mesh_name).strip()
    candidates = []
    if os.path.isabs(mesh_name):
        candidates.append(mesh_name)
    else:
        candidates.extend(
            os.path.join(root, mesh_name)
            for root in (adjoint_dir, design_dir, level_dir)
        )
    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)
    raise FileNotFoundError(
        f"Could not resolve HH projection mesh {mesh_name!r}; tried {candidates}"
    )


def _resolve_hh_surface_sensitivity_file(adjoint_dir, function_name):
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


def _resolve_hh_saved_gradient_file(adjoint_dir):
    candidates = sorted(
        path
        for path in glob.glob(os.path.join(adjoint_dir, "*of_grad*.csv"))
        if os.path.isfile(path)
    )
    if len(candidates) > 1:
        raise RuntimeError(
            f"Ambiguous saved SU2 gradient files in {adjoint_dir}: {candidates}"
        )
    return candidates[0] if candidates else None


def _validate_hh_saved_surface_projection(
    tangent_state,
    field,
    adjoint_dir,
    function_name,
):
    reference_file = _resolve_hh_saved_gradient_file(adjoint_dir)
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
            "status": "passed" if validation["passed"] else "failed",
            "function": str(function_name).upper(),
            "reference_file": reference_file,
        }
    )
    return validation


def _hh_config_tangent_records(cfg_level, symmetry_mode="NONE", symmetry_sign=-1.0):
    definition = cfg_level.get("DEFINITION_DV")
    if not isinstance(definition, dict):
        return None
    upper = []
    lower = []
    for kind, raw_params in zip(
        definition.get("KIND", []),
        definition.get("PARAM", []),
    ):
        if str(kind).strip().upper() != "HICKS_HENNE":
            return None
        values = [float(value) for value in raw_params]
        if len(values) < 2:
            return None
        (upper if bool(round(values[-2])) else lower).append(values[-1])
    upper = sorted(upper)
    lower = sorted(lower)
    if str(symmetry_mode).upper() == "REDUCED":
        if len(upper) != len(lower) or any(
            abs(xu - xl) > 1.0e-12 for xu, xl in zip(upper, lower)
        ):
            return None
        _ = symmetry_sign
        return [("PAIR", float(x)) for x in upper]
    return [
        *(("UPPER", float(x)) for x in upper),
        *(("LOWER", float(x)) for x in lower),
    ]


def _hh_tangent_records_match_config(tangent_state, cfg_level):
    configured = _hh_config_tangent_records(
        cfg_level,
        tangent_state.get("symmetry_mode", "NONE"),
        tangent_state.get("symmetry_sign", -1.0),
    )
    actual = list(tangent_state.get("records", []))
    if configured is None or len(configured) != len(actual):
        return False
    return all(
        str(side_a).upper() == str(side_b).upper()
        and abs(float(x_a) - float(x_b)) <= 1.0e-12
        for (side_a, x_a), (side_b, x_b) in zip(configured, actual)
    )


def _constraint_active_from_hh_value(spec, current_value, active_tol):
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


def _prepare_hh_virtual_signal(
    level,
    opts,
    cfg_level,
    baseline_state,
    obj_name,
    obj_adj_dir,
    design_dir,
):
    objective_file = _resolve_hh_surface_sensitivity_file(obj_adj_dir, obj_name)
    objective_field = load_surface_sensitivity_vector(objective_file, baseline_state)
    objective_validation = _validate_hh_saved_surface_projection(
        baseline_state,
        objective_field,
        obj_adj_dir,
        obj_name,
    )
    strict_projection_validation = _hh_tangent_records_match_config(
        baseline_state,
        cfg_level,
    )
    if objective_validation.get("status") not in ("passed", "reference_unavailable"):
        if strict_projection_validation:
            raise HHTangentError(
                "HH surface sensitivity does not reproduce the saved SU2_DOT "
                f"gradient for {obj_name}: {objective_validation}"
            )
        objective_validation["status"] = "parameterization_differs_expected"
        objective_validation["note"] = (
            "The active HH centers were redistributed after the saved adjoint "
            "gradient was written; the surface field remains the scoring source."
        )
    indicator_mode = str(opts.get("adaptive_indicator", "ABS_GRAD")).upper()
    requested_names = _extract_constraint_names(cfg_level)
    active_names, constraint_status = _select_active_ikkt_constraints(
        cfg_level,
        requested_names,
        design_dir=design_dir,
        active_tol=opts.get("ikkt_active_tol", 1.0e-6),
        verbose=False,
    )
    status_by_name = {
        str(record["name"]).upper(): dict(record)
        for record in constraint_status
    }
    specs = {
        spec["name"]: spec
        for spec in _extract_constraint_specs(cfg_level, requested_names)
    }
    constraint_records = []
    inactive_records = [
        dict(record)
        for record in constraint_status
        if not bool(record.get("included", False))
    ]
    unsupported = []

    if indicator_mode == "IKKT":
        for name in active_names:
            name = str(name).upper()
            spec = dict(specs.get(name, {"name": name}))
            spec.setdefault("name", name)
            status = status_by_name.get(name, {})
            provider = None
            provider_metadata = {}
            provider_value = status.get("current_value")
            field_file = None

            if is_geometry_constraint_name(name):
                canonical = normalize_geometry_constraint_name(name)
                if canonical == "AIRFOIL_AREA":
                    provider_value, raw_field = airfoil_area_value_and_field(
                        baseline_state
                    )
                    provider = "analytic_airfoil_area"
                elif canonical == "AIRFOIL_THICKNESS":
                    provider_value, raw_field, provider_metadata = (
                        airfoil_thickness_value_and_field(baseline_state)
                    )
                    provider = "analytic_airfoil_thickness"
                else:
                    unsupported.append(
                        {
                            "name": name,
                            "source": "GEOMETRY",
                            "reason": "unsupported_geometry_constraint",
                        }
                    )
                    continue
                active, c_value, active_status = _constraint_active_from_hh_value(
                    spec,
                    provider_value,
                    opts.get("ikkt_active_tol", 1.0e-6),
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
                    unsupported.append(
                        {
                            "name": name,
                            "source": "AERO",
                            "reason": "active_field_unavailable",
                            "detail": str(exc),
                        }
                    )
                    continue
                if os.path.abspath(constraint_design_dir) != os.path.abspath(design_dir):
                    unsupported.append(
                        {
                            "name": name,
                            "source": "AERO",
                            "reason": "adjoint_baseline_mismatch",
                        }
                    )
                    continue
                field_file = _resolve_hh_surface_sensitivity_file(
                    constraint_adj_dir,
                    name,
                )
                raw_field = load_surface_sensitivity_vector(
                    field_file,
                    baseline_state,
                )
                provider = "adjoint_surface_sensitivity"
                if provider_value is None:
                    unsupported.append(
                        {
                            "name": name,
                            "source": "AERO",
                            "reason": "current_value_unavailable",
                        }
                    )
                    continue
                provider_metadata["projection_validation"] = (
                    _validate_hh_saved_surface_projection(
                        baseline_state,
                        raw_field,
                        constraint_adj_dir,
                        name,
                    )
                )
                if provider_metadata["projection_validation"].get("status") not in (
                    "passed",
                    "reference_unavailable",
                ):
                    if strict_projection_validation:
                        unsupported.append(
                            {
                                "name": name,
                                "source": "AERO",
                                "reason": "surface_projection_validation_failed",
                                "detail": provider_metadata[
                                    "projection_validation"
                                ],
                            }
                        )
                        continue
                    provider_metadata["projection_validation"]["status"] = (
                        "parameterization_differs_expected"
                    )

            internal = internal_constraint_field(spec, provider_value, raw_field)
            constraint_records.append(
                {
                    "name": name,
                    "source": (
                        "GEOMETRY" if is_geometry_constraint_name(name) else "AERO"
                    ),
                    "provider": provider,
                    "provider_metadata": provider_metadata,
                    "field_file": field_file,
                    "current_value": float(provider_value),
                    "target": spec.get("target"),
                    "original_operator": str(spec.get("sign", "")),
                    **internal,
                }
            )

    if unsupported:
        detail = "; ".join(
            f"{record['name']}:{record['reason']}" for record in unsupported
        )
        raise RuntimeError(
            "Virtual HH IKKT requires every active constraint field; " + detail
        )

    baseline_scoring_state, scoring_node_mask, scoring_mask_metadata = (
        build_surface_scoring_view(
            baseline_state,
            opts.get("scoring_te_closure_node_eps", 0.0),
        )
    )
    scoring_objective_field = mask_surface_field(
        objective_field,
        scoring_node_mask,
        label="objective surface field",
    )
    scoring_constraint_records = mask_surface_constraint_records(
        constraint_records,
        scoring_node_mask,
    )
    if indicator_mode == "IKKT":
        signal, lambdas, fit_diagnostics = fit_surface_ikkt_signal(
            scoring_objective_field,
            scoring_constraint_records,
            baseline_scoring_state,
        )
        signal_source = (
            "IKKT_SURFACE_RESIDUAL"
            if constraint_records
            else "OBJECTIVE_ONLY_NO_ACTIVE_CONSTRAINTS"
        )
    else:
        signal = np.asarray(scoring_objective_field, dtype=float)
        lambdas = np.zeros(0, dtype=float)
        fit_diagnostics = {
            "status": "objective_only",
            "objective_gradient": project_surface_field(
                baseline_scoring_state,
                scoring_objective_field,
            ).tolist(),
        }
        signal_source = "OBJECTIVE_SURFACE_SENSITIVITY"

    return {
        "objective_function": str(obj_name).upper(),
        "objective_field_file": objective_file,
        "objective_field": objective_field,
        "objective_projection_validation": objective_validation,
        "constraint_records": constraint_records,
        "baseline_scoring_state": baseline_scoring_state,
        "surface_scoring_mask": scoring_mask_metadata,
        "inactive_constraints": inactive_records,
        "unsupported_constraints": unsupported,
        "signal": np.asarray(signal, dtype=float),
        "signal_source": signal_source,
        "lambdas": np.asarray(lambdas, dtype=float),
        "fit_diagnostics": fit_diagnostics,
    }


def _hh_active_map(level):
    return {
        side: sorted(float(value) for value in values)
        for side, values in (
            ("UPPER", level.upper),
            ("LOWER", level.lower),
        )
        if values
    }


def _generate_hh_virtual_candidates(active_by_side, opts):
    nsamples = int(opts.get("candidate_samples", 1))
    min_spacing = float(opts.get("min_center_spacing", 0.0))
    if _is_symmetric_reduced(opts):
        centers = active_by_side.get("UPPER", [])
        raw = get_midpoint_candidates(centers, nsamples=nsamples)
        for candidate in raw:
            candidate["side"] = "PAIR"
        return _filter_candidates_by_min_spacing(
            raw,
            {"PAIR": centers},
            min_spacing,
        )

    raw = []
    for side in ("UPPER", "LOWER"):
        if side not in active_by_side:
            continue
        side_candidates = get_midpoint_candidates(
            active_by_side[side],
            nsamples=nsamples,
        )
        for candidate in side_candidates:
            candidate["side"] = side
        raw.extend(side_candidates)
    return _filter_candidates_by_min_spacing(
        raw,
        active_by_side,
        min_spacing,
    )


def _hh_virtual_insertion_target(level, opts, candidate_count):
    mode = str(opts.get("nadd_mode", "GROWTH_RATIO")).upper()
    symmetric = _is_symmetric_reduced(opts)
    current = len(level.upper) if symmetric else int(level.ndv)
    if mode == "GROWTH_RATIO":
        ratio = float(opts.get("growth_ratio", 2.0))
        target_count = current + 1 if ratio <= 1.0 else int(math.ceil(ratio * current))
        target = max(1, target_count - current)
    elif mode == "FIXED":
        target = int(opts.get("fixed_nadd", 1))
    elif mode == "SCORE_BATCH":
        target = int(opts.get("batch_size_max", 1))
    else:
        raise ValueError(f"Unknown progressive HH candidate addition mode: {mode}")

    nfinal = opts.get("nfinal", None)
    if nfinal is not None:
        remaining_full = max(0, int(nfinal) - int(level.ndv))
        remaining = remaining_full // 2 if symmetric else remaining_full
        target = min(target, remaining)
    return max(0, min(int(target), int(candidate_count)))


def _hh_virtual_rank_key(candidate):
    side_rank = {"UPPER": 0, "LOWER": 1, "PAIR": 2}
    return (
        -float(candidate["indicator"]),
        side_rank.get(str(candidate.get("side", "")).upper(), 99),
        float(candidate["x"]),
    )


def _hh_candidate_active_map(active_by_side, side, x):
    updated = {key: list(values) for key, values in active_by_side.items()}
    if str(side).upper() == "PAIR":
        updated.setdefault("UPPER", []).append(float(x))
        updated.setdefault("LOWER", []).append(float(x))
    else:
        updated.setdefault(str(side).upper(), []).append(float(x))
    for key in updated:
        updated[key] = sorted(updated[key])
    return updated


def _hh_record_index(state, side, x):
    for index, (record_side, record_x) in enumerate(state["records"]):
        if str(record_side).upper() == str(side).upper() and abs(
            float(record_x) - float(x)
        ) <= 1.0e-12:
            return index
    raise HHTangentError(f"Candidate record {side}@{x} absent from HH tangent state")


def _json_safe_hh(value):
    if isinstance(value, np.ndarray):
        return [_json_safe_hh(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe_hh(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe_hh(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_hh(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_hh_virtual_scores(level, candidates, metadata):
    csv_path = os.path.join(
        level.workdir,
        f"hh_candidate_scores_level{level.level_id}.csv",
    )
    fields = [
        "scoring_mode",
        "scoring_basis",
        "signal_source",
        "insertion_step",
        "insertion_target",
        "rank",
        "selected",
        "side",
        "x",
        "t2",
        "interval_id",
        "interval_left",
        "interval_right",
        "sample_index",
        "sample_fraction",
        "objective_gradient",
        "surface_residual_component",
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
    ]
    with open(csv_path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {key: candidate.get(key, "") for key in fields}
            )

    json_path = os.path.join(
        level.workdir,
        f"hh_virtual_tangent_level{level.level_id}.json",
    )
    with open(json_path, "w") as stream:
        json.dump(
            _json_safe_hh(metadata),
            stream,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    return csv_path, json_path


def _compute_virtual_tangent_candidate_scores(level, opts):
    cfg_path = os.path.join(level.workdir, level.config_filename)
    cfg_level = SU2.io.Config(cfg_path)
    obj_name = str(cfg_level.get("OBJECTIVE_FUNCTION", "DRAG")).upper()
    obj_adj_dir, design_dir = _find_real_adjoint_assets(level.workdir, obj_name)
    real_dot_cfg_path, dot_kind = _select_dot_config_path(obj_adj_dir, cfg_level)
    real_dot_cfg = SU2.io.Config(real_dot_cfg_path)
    mesh_path = _find_hh_projection_mesh_source(
        real_dot_cfg["MESH_FILENAME"],
        obj_adj_dir,
        design_dir,
        level.workdir,
    )
    active = _hh_active_map(level)
    symmetry_mode = "REDUCED" if _is_symmetric_reduced(opts) else "NONE"
    symmetry_sign = float(opts.get("symmetry_sign", -1.0))
    baseline = build_hh_tangent_state(
        mesh_path,
        opts.get("marker", "AIRFOIL"),
        active,
        cfg_level,
        scale=opts.get("scale", 1.0),
        symmetry_mode=symmetry_mode,
        symmetry_sign=symmetry_sign,
    )
    virtual = _prepare_hh_virtual_signal(
        level,
        opts,
        cfg_level,
        baseline,
        obj_name,
        obj_adj_dir,
        design_dir,
    )
    initial_candidates = _generate_hh_virtual_candidates(active, opts)
    target = _hh_virtual_insertion_target(level, opts, len(initial_candidates))
    signal = virtual["signal"]
    objective_field = virtual["objective_field"]
    initial_projection = np.abs(
        project_surface_field(virtual["baseline_scoring_state"], signal)
    ).tolist()
    initial_records = list(baseline["records"])
    initial_scores = {
        (str(side).upper(), float(x)): float(initial_projection[index])
        for index, (side, x) in enumerate(initial_records)
    }

    print(
        "[PROGRESSIVE_HH] Candidate scoring | "
        f"indicator={str(opts.get('adaptive_indicator', 'ABS_GRAD')).upper()} "
        f"scoring_mode={VIRTUAL_TANGENT} kind={dot_kind} "
        f"signal={virtual['signal_source']} target_insertions={target} "
        "te_closure_node_eps="
        f"{virtual['surface_scoring_mask']['te_closure_node_eps']:.16g}"
    )

    current_active = {key: list(values) for key, values in active.items()}
    selected = []
    all_candidates = []
    first_best = None
    mode = str(opts.get("nadd_mode", "GROWTH_RATIO")).upper()
    for insertion_step in range(1, target + 1):
        raw_candidates = (
            initial_candidates
            if insertion_step == 1
            else _generate_hh_virtual_candidates(current_active, opts)
        )
        if not raw_candidates:
            break
        baseline_step = build_hh_tangent_state(
            mesh_path,
            opts.get("marker", "AIRFOIL"),
            current_active,
            cfg_level,
            scale=opts.get("scale", 1.0),
            symmetry_mode=symmetry_mode,
            symmetry_sign=symmetry_sign,
        )
        baseline_scoring_step, _baseline_mask, _baseline_mask_metadata = (
            build_surface_scoring_view(
                baseline_step,
                opts.get("scoring_te_closure_node_eps", 0.0),
            )
        )
        step_candidates = []
        for candidate_number, raw in enumerate(raw_candidates):
            side = str(raw["side"]).upper()
            x = float(raw["x"])
            candidate_active = _hh_candidate_active_map(
                current_active,
                side,
                x,
            )
            candidate_state = build_hh_tangent_state(
                mesh_path,
                opts.get("marker", "AIRFOIL"),
                candidate_active,
                cfg_level,
                scale=opts.get("scale", 1.0),
                symmetry_mode=symmetry_mode,
                symmetry_sign=symmetry_sign,
            )
            candidate_scoring_state, _candidate_mask, _candidate_mask_metadata = (
                build_surface_scoring_view(
                    candidate_state,
                    opts.get("scoring_te_closure_node_eps", 0.0),
                )
            )
            metrics = compare_tangent_spaces(
                baseline_scoring_step,
                candidate_scoring_state,
                signal,
                x,
            )
            candidate_index = _hh_record_index(candidate_state, side, x)
            objective_gradient = project_surface_field(
                candidate_state,
                objective_field,
            )
            residual_gradient = project_surface_field(
                candidate_scoring_state,
                signal,
            )
            admissible = bool(
                int(metrics["rank_gain"]) == 1
                and int(metrics["pure_rank"]) == 1
            )
            rejected_reason = str(raw.get("rejected_reason", ""))
            if not admissible:
                rejected_reason = (
                    "VIRTUAL_TANGENT_RANK_MISMATCH:"
                    f"rank_gain={metrics['rank_gain']},pure_rank={metrics['pure_rank']}"
                )
            record = {
                "scoring_mode": VIRTUAL_TANGENT,
                "scoring_basis": "HH_VIRTUAL_TANGENT_SPACE",
                "signal_source": virtual["signal_source"],
                "insertion_step": int(insertion_step),
                "insertion_target": int(target),
                "candidate_number": int(candidate_number),
                "side": side,
                "x": x,
                "t2": float(candidate_state["t2_by_record"][candidate_index]),
                "grad": float(objective_gradient[candidate_index]),
                "objective_gradient": float(objective_gradient[candidate_index]),
                "surface_residual_component": float(
                    residual_gradient[candidate_index]
                ),
                # For nested HH spaces score_pure and score_net agree up to
                # roundoff.  The residualized pure energy is the stable rank key.
                "indicator": float(metrics["score_pure"]),
                "interval_id": raw["interval_id"],
                "interval_left": float(raw["interval_left"]),
                "interval_right": float(raw["interval_right"]),
                "sample_index": int(raw["sample_index"]),
                "sample_fraction": float(raw["sample_fraction"]),
                "admissible": admissible,
                "rejected_reason": rejected_reason,
                "selected": "NO",
                **metrics,
            }
            step_candidates.append(record)

        admissible = [
            candidate
            for candidate in step_candidates
            if candidate.get("admissible", False)
        ]
        reduced = _reduce_candidates_to_interval_best(admissible)
        ranked = sorted(reduced, key=_hh_virtual_rank_key)
        for rank, candidate in enumerate(ranked, start=1):
            candidate["rank"] = rank
        all_candidates.extend(step_candidates)
        if not ranked:
            raise RuntimeError(
                "Virtual HH tangent scoring found no rank-one admissible candidate"
            )

        chosen = None
        for candidate in ranked:
            if mode == "SCORE_BATCH":
                score = float(candidate["indicator"])
                if first_best is not None and score < float(
                    opts.get("batch_score_rel_tol", 0.85)
                ) * first_best:
                    continue
                side = str(candidate["side"])
                if opts.get("batch_max_per_side", None) is not None:
                    count = sum(str(item["side"]) == side for item in selected)
                    if count >= int(opts["batch_max_per_side"]):
                        continue
                separation = float(opts.get("batch_min_separation", 0.04))
                if any(
                    str(item["side"]) == side
                    and abs(float(item["x"]) - float(candidate["x"])) < separation
                    for item in selected
                ):
                    continue
            chosen = candidate
            break
        if chosen is None:
            break
        if first_best is None:
            first_best = float(chosen["indicator"])
        chosen["selected"] = "YES"
        selected.append(chosen)
        current_active = _hh_candidate_active_map(
            current_active,
            chosen["side"],
            chosen["x"],
        )
        print(
            "[PROGRESSIVE_HH] VIRTUAL_TANGENT selected | "
            f"insertion={insertion_step}/{target} side={chosen['side']} "
            f"x={chosen['x']:.8f} t2={chosen['t2']:.6g} "
            f"pure_energy={chosen['score_pure']:.8e} "
            f"net_energy={chosen['score_net']:.8e}"
        )

    if not selected and target > 0:
        raise RuntimeError("Virtual HH tangent scoring selected no candidates")

    active_upper_scores = [
        initial_scores.get(("UPPER", float(x)), 0.0) for x in level.upper
    ]
    active_lower_scores = [
        initial_scores.get(("LOWER", float(x)), 0.0) for x in level.lower
    ]
    active_pair_scores = [
        initial_scores.get(("PAIR", float(x)), 0.0) for x in level.upper
    ] if _is_symmetric_reduced(opts) else []
    metadata = {
        "scoring_mode": VIRTUAL_TANGENT,
        "scoring_basis": "HH_VIRTUAL_TANGENT_SPACE",
        "level_id": int(level.level_id),
        "mesh": mesh_path,
        "marker": baseline["marker"],
        "t2_policy": baseline["t2_policy"],
        "symmetry_mode": symmetry_mode,
        "symmetry_sign": symmetry_sign,
        "signal_source": virtual["signal_source"],
        "objective_function": virtual["objective_function"],
        "objective_field_file": virtual["objective_field_file"],
        "objective_projection_validation": virtual[
            "objective_projection_validation"
        ],
        "fit_diagnostics": virtual["fit_diagnostics"],
        "lambdas": virtual["lambdas"],
        "surface_scoring_mask": virtual["surface_scoring_mask"],
        "included_constraints": [
            {key: value for key, value in record.items() if key != "field"}
            for record in virtual["constraint_records"]
        ],
        "inactive_constraints": virtual["inactive_constraints"],
        "insertion_target": int(target),
        "insertions_completed": len(selected),
        "selected_candidates": selected,
    }
    csv_path, json_path = _write_hh_virtual_scores(
        level,
        all_candidates,
        metadata,
    )
    return {
        "candidates": list(selected),
        "raw_candidates": all_candidates,
        "selected_candidates": selected,
        "sequential_selection": True,
        "scoring_basis": "HH_VIRTUAL_TANGENT_SPACE",
        "scoring_mode": VIRTUAL_TANGENT,
        "signal_source": virtual["signal_source"],
        "surface_scoring_mask": virtual["surface_scoring_mask"],
        "active_upper_scores": active_upper_scores,
        "active_lower_scores": active_lower_scores,
        "active_pair_scores": active_pair_scores,
        "candidate_scores_csv": csv_path,
        "metadata_json": json_path,
        "insertion_target": int(target),
        "insertions_completed": len(selected),
    }


def _compute_dot_candidate_scores(level, opts):
    """Dispatch the legacy component score or the opt-in HH tangent score."""

    mode = normalize_hh_scoring_mode(opts.get("scoring_mode", COMPONENT))
    if mode == VIRTUAL_TANGENT:
        return _compute_virtual_tangent_candidate_scores(level, opts)
    return _compute_component_dot_candidate_scores(level, opts)
