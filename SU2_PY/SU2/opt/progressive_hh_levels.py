#!/usr/bin/env python

import os
import copy
import csv
import glob
import shutil

import SU2

from SU2.opt.progressive_hh_core import (
    HHLevel,
    assert_symmetric_centers,
    initial_centers,
    is_symmetric_reduced,
    refine_uniform,
    refine_adaptive,
    apply_post_opt_coefficient_spring,
)


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

def _parse_initial_points(value):
    if value is None:
        return None

    raw = str(value).strip()
    if not raw:
        return None

    raw = raw.strip("()[]")
    if not raw:
        return []

    pts = [float(x.strip()) for x in raw.split(",") if x.strip()]
    pts = sorted(pts)

    for x in pts:
        if not (0.0 < x < 1.0):
            raise ValueError(
                f"Invalid HH initial point {x} (must satisfy 0 < x < 1)"
            )

    return pts

def build_initial_level(base_config, opts):
    upper = []
    lower = []

    upper_manual = _parse_initial_points(
        base_config.get("PROGRESSIVE_HH_INITIAL_UPPER", None)
    )
    lower_manual = _parse_initial_points(
        base_config.get("PROGRESSIVE_HH_INITIAL_LOWER", None)
    )

    if is_symmetric_reduced(opts):
        if opts["surface_mode"] != "BOTH":
            raise ValueError(
                "PROGRESSIVE_HH_SYMMETRY_MODE=REDUCED requires "
                "PROGRESSIVE_HH_SURFACE=BOTH"
            )
        if upper_manual is not None and lower_manual is not None:
            assert_symmetric_centers(upper_manual, lower_manual)
            pair_centers = upper_manual
        elif upper_manual is not None:
            print(
                "[PROGRESSIVE_HH][SYMMETRY] WARNING: INITIAL_LOWER missing; "
                "copying INITIAL_UPPER for reduced symmetry"
            )
            pair_centers = upper_manual
        elif lower_manual is not None:
            print(
                "[PROGRESSIVE_HH][SYMMETRY] WARNING: INITIAL_UPPER missing; "
                "copying INITIAL_LOWER for reduced symmetry"
            )
            pair_centers = lower_manual
        else:
            pair_centers = initial_centers(opts["n0"])

        upper = list(pair_centers)
        lower = list(pair_centers)

        initial_mesh = None
        if "MESH_FILENAME" in base_config and base_config["MESH_FILENAME"]:
            initial_mesh = _resolve_from_cfg_dir(base_config, base_config["MESH_FILENAME"])

        print(f"[PROGRESSIVE_HH] Initial upper centers = {upper}")
        print(f"[PROGRESSIVE_HH] Initial lower centers = {lower}")
        print(f"[PROGRESSIVE_HH][SYMMETRY] pair count = {len(upper)}")
        print(f"[PROGRESSIVE_HH][SYMMETRY] full SU2 HH = {len(upper) + len(lower)}")

        return HHLevel(
            level_id=0,
            upper=upper,
            lower=lower,
            workdir="LEVEL_0",
            config_filename="config_level0.cfg",
            project_filename="project_level0.pkl",
            mesh_source=initial_mesh,
        )

    if opts["surface_mode"] in ("UPPER", "BOTH"):
        if upper_manual is not None:
            upper = upper_manual
        else:
            upper = initial_centers(opts["n0"])

    if opts["surface_mode"] in ("LOWER", "BOTH"):
        if lower_manual is not None:
            lower = lower_manual
        else:
            lower = initial_centers(opts["n0"])

    initial_mesh = None
    if "MESH_FILENAME" in base_config and base_config["MESH_FILENAME"]:
        initial_mesh = _resolve_from_cfg_dir(base_config, base_config["MESH_FILENAME"])

    print(f"[PROGRESSIVE_HH] Initial upper centers = {upper}")
    print(f"[PROGRESSIVE_HH] Initial lower centers = {lower}")

    return HHLevel(
        level_id=0,
        upper=upper,
        lower=lower,
        workdir="LEVEL_0",
        config_filename="config_level0.cfg",
        project_filename="project_level0.pkl",
        mesh_source=initial_mesh,
    )


def _cap_uniform_refinement(prev_level, upper, lower, opts):
    nfinal = opts.get("nfinal", None)
    if nfinal is None:
        if is_symmetric_reduced(opts):
            assert_symmetric_centers(upper, lower)
            pair = sorted(upper)
            return pair, list(pair)
        return upper, lower

    if is_symmetric_reduced(opts):
        assert_symmetric_centers(prev_level.upper, prev_level.lower)
        assert_symmetric_centers(upper, lower)

        n_remaining_pairs = (int(nfinal) - prev_level.ndv) // 2
        if n_remaining_pairs <= 0:
            pair = sorted(prev_level.upper)
            return pair, list(pair)

        old_pair = set(prev_level.upper)
        add_pair = sorted(x for x in upper if x not in old_pair)
        keep = add_pair[:n_remaining_pairs]
        pair = sorted(set(list(prev_level.upper) + keep))
        return pair, list(pair)

    n_remaining = int(nfinal) - prev_level.ndv
    if n_remaining <= 0:
        return sorted(prev_level.upper), sorted(prev_level.lower)

    old_upper = set(prev_level.upper)
    old_lower = set(prev_level.lower)
    add_upper = sorted(x for x in upper if x not in old_upper)
    add_lower = sorted(x for x in lower if x not in old_lower)

    additions = []
    additions.extend(("UPPER", x) for x in add_upper)
    additions.extend(("LOWER", x) for x in add_lower)
    additions = sorted(additions, key=lambda item: (item[0], item[1]))

    keep = additions[:n_remaining]
    new_upper = sorted(prev_level.upper)
    new_lower = sorted(prev_level.lower)

    for side, x in keep:
        if side == "UPPER":
            new_upper.append(x)
        else:
            new_lower.append(x)

    return sorted(set(new_upper)), sorted(set(new_lower))


def build_next_level(prev_level, result, opts):
    next_id = prev_level.level_id + 1

    next_mesh = result.get("final_mesh", None)
    if next_mesh is None:
        next_mesh = prev_level.mesh_source

    selection_metadata = None

    if opts["refinement"] == "ADAPTIVE":
        upper, lower = refine_adaptive(prev_level, result, opts)
        selection_metadata = opts.get("_last_selection_metadata", None)
        if selection_metadata is None:
            upper, lower = _cap_uniform_refinement(prev_level, upper, lower, opts)
    else:
        if is_symmetric_reduced(opts):
            assert_symmetric_centers(prev_level.upper, prev_level.lower)
            upper = refine_uniform(prev_level.upper)
            lower = list(upper)
        else:
            upper = refine_uniform(prev_level.upper)
            lower = refine_uniform(prev_level.lower)
        upper, lower = _cap_uniform_refinement(prev_level, upper, lower, opts)

    if is_symmetric_reduced(opts):
        assert_symmetric_centers(upper, lower)

    return HHLevel(
        level_id=next_id,
        upper=upper,
        lower=lower,
        workdir=f"LEVEL_{next_id}",
        config_filename=f"config_level{next_id}.cfg",
        project_filename=f"project_level{next_id}.pkl",
        mesh_source=next_mesh,
        selection_metadata=selection_metadata,
        post_opt_spring_pending=bool(
            selection_metadata
            and selection_metadata.get("post_opt_spring_pending", False)
        ),
        spring_reallocated=False,
    )


def build_spring_reallocated_level(prev_level, result, opts, reoptimize=True):
    reallocated = apply_post_opt_coefficient_spring(prev_level, result, opts)
    if reallocated is None:
        print(
            "[PROGRESSIVE_HH][SPRING] WARNING: could not build spring-reallocated level"
        )
        return None

    upper, lower = reallocated
    next_id = prev_level.level_id + 1

    next_mesh = result.get("final_mesh", None)
    if next_mesh is None:
        next_mesh = prev_level.mesh_source

    spring_metadata = {
        "level_id": prev_level.level_id,
        "ndv": prev_level.ndv,
        "spring_timing": opts.get("spring_timing", "POST_OPT"),
        "spring_score_mode": opts.get("spring_score_mode", "COEFFICIENT"),
        "spring_post_action": opts.get("spring_post_action", "REOPTIMIZE"),
        "post_spring_optimization_skipped": not reoptimize,
        "upper_before": sorted(prev_level.upper),
        "lower_before": sorted(prev_level.lower),
        "upper_after": sorted(upper),
        "lower_after": sorted(lower),
        "upper_coeff_abs": result.get("spring_upper_coeff_abs", []),
        "lower_coeff_abs": result.get("spring_lower_coeff_abs", []),
        "history_file": result.get("history_file"),
        "final_mesh": result.get("final_mesh"),
    }

    if is_symmetric_reduced(opts):
        assert_symmetric_centers(upper, lower)
        spring_metadata["symmetry_mode"] = "REDUCED"
        spring_metadata["symmetry_sign"] = opts.get("symmetry_sign", -1.0)

    if not reoptimize:
        return HHLevel(
            level_id=prev_level.level_id,
            upper=upper,
            lower=lower,
            workdir=prev_level.workdir,
            config_filename=prev_level.config_filename,
            project_filename=prev_level.project_filename,
            mesh_source=next_mesh,
            selection_metadata=spring_metadata,
            post_opt_spring_pending=False,
            spring_reallocated=True,
        )

    return HHLevel(
        level_id=next_id,
        upper=upper,
        lower=lower,
        workdir=f"LEVEL_{next_id}",
        config_filename=f"config_level{next_id}.cfg",
        project_filename=f"project_level{next_id}.pkl",
        mesh_source=next_mesh,
        selection_metadata=spring_metadata,
        post_opt_spring_pending=False,
        spring_reallocated=True,
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
        "PROGRESSIVE_HH_NFINAL",
        "PROGRESSIVE_HH_INITIAL_UPPER",
        "PROGRESSIVE_HH_INITIAL_LOWER",
        "PROGRESSIVE_HH_SURFACE",
        "PROGRESSIVE_HH_SYMMETRY_MODE",
        "PROGRESSIVE_HH_SYMMETRY_SIGN",
        "PROGRESSIVE_HH_TRIGGER",
        "PROGRESSIVE_HH_MAX_ITER_PER_LEVEL",
        "PROGRESSIVE_HH_WINDOW",
        "PROGRESSIVE_HH_TOL",
        "PROGRESSIVE_HH_SLOPE_FILTER_TOL",
        "PROGRESSIVE_HH_TRIGGER_EPS",
        "PROGRESSIVE_HH_SLOPE_PATIENCE",
        "PROGRESSIVE_HH_STAG_TOL",
        "PROGRESSIVE_HH_STAG_BAND",
        "PROGRESSIVE_HH_STAG_WINDOW",
        "PROGRESSIVE_HH_WARMUP_ITER",
        "PROGRESSIVE_HH_REFINEMENT",
        "PROGRESSIVE_HH_GROWTH_RATIO",
        "PROGRESSIVE_HH_NADD_MODE",
        "PROGRESSIVE_HH_FIXED_NADD",
        "PROGRESSIVE_HH_BATCH_SIZE_MAX",
        "PROGRESSIVE_HH_BATCH_SCORE_REL_TOL",
        "PROGRESSIVE_HH_BATCH_MIN_SEPARATION",
        "PROGRESSIVE_HH_BATCH_MAX_PER_SIDE",
        "PROGRESSIVE_HH_CANDIDATE_SAMPLES",
        "PROGRESSIVE_HH_MIN_CENTER_SPACING",
        "PROGRESSIVE_HH_ADAPTIVE_INDICATOR",
        "PROGRESSIVE_HH_SPRING",
        "PROGRESSIVE_HH_SPRING_A",
        "PROGRESSIVE_HH_SPRING_TIMING",
        "PROGRESSIVE_HH_SPRING_SCORE_MODE",
        "PROGRESSIVE_HH_SPRING_POST_ACTION",
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

    nfinal = opts.get("nfinal", None)
    is_final_by_nlevels = (
        nfinal is None and level.level_id >= opts["nlevels"] - 1
    )
    is_final_by_nfinal = nfinal is not None and level.ndv >= int(nfinal)

    if is_final_by_nlevels or is_final_by_nfinal:
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


def _format_centers(values):
    return ";".join(f"{float(x):.12g}" for x in values)


def append_selection_history_csv(
    csv_path,
    selection_metadata,
    result,
):
    if not selection_metadata:
        return

    selected = selection_metadata.get("selected", [])
    if not selected:
        return

    columns = [
        "level_id",
        "ndv_before",
        "ndv_after",
        "n_added",
        "nadd_mode",
        "trigger_mode",
        "refinement",
        "spring_enabled",
        "side",
        "x",
        "indicator",
        "indicator_ratio_to_best",
        "interval_id",
        "interval_left",
        "interval_right",
        "sample_index",
        "sample_fraction",
        "rejected_reason",
        "nearest_center_or_boundary",
        "nearest_distance",
        "required_spacing",
        "upper_before",
        "lower_before",
        "upper_after",
        "lower_after",
        "history_file",
        "final_mesh",
    ]

    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        if write_header:
            writer.writeheader()

        for c in selected:
            writer.writerow(
                {
                    "level_id": selection_metadata.get("level_id"),
                    "ndv_before": selection_metadata.get("ndv_before"),
                    "ndv_after": selection_metadata.get("ndv_after"),
                    "n_added": selection_metadata.get("n_added"),
                    "nadd_mode": selection_metadata.get("nadd_mode"),
                    "trigger_mode": selection_metadata.get("trigger_mode"),
                    "refinement": selection_metadata.get("refinement"),
                    "spring_enabled": "YES"
                    if selection_metadata.get("spring_enabled")
                    else "NO",
                    "side": c.get("side"),
                    "x": f"{float(c.get('x')):.12g}",
                    "indicator": f"{float(c.get('indicator')):.12e}",
                    "indicator_ratio_to_best": (
                        f"{float(c.get('indicator_ratio_to_best')):.12e}"
                    ),
                    "interval_id": c.get("interval_id", ""),
                    "interval_left": ""
                    if c.get("interval_left") is None
                    else f"{float(c.get('interval_left')):.12g}",
                    "interval_right": ""
                    if c.get("interval_right") is None
                    else f"{float(c.get('interval_right')):.12g}",
                    "sample_index": c.get("sample_index", ""),
                    "sample_fraction": ""
                    if c.get("sample_fraction") is None
                    else f"{float(c.get('sample_fraction')):.12g}",
                    "rejected_reason": c.get("rejected_reason", ""),
                    "nearest_center_or_boundary": ""
                    if c.get("nearest_center_or_boundary") in (None, "")
                    else f"{float(c.get('nearest_center_or_boundary')):.12g}",
                    "nearest_distance": ""
                    if c.get("nearest_distance") in (None, "")
                    else f"{float(c.get('nearest_distance')):.12g}",
                    "required_spacing": ""
                    if c.get("required_spacing") in (None, "")
                    else f"{float(c.get('required_spacing')):.12g}",
                    "upper_before": _format_centers(
                        selection_metadata.get("upper_before", [])
                    ),
                    "lower_before": _format_centers(
                        selection_metadata.get("lower_before", [])
                    ),
                    "upper_after": _format_centers(
                        selection_metadata.get("upper_after", [])
                    ),
                    "lower_after": _format_centers(
                        selection_metadata.get("lower_after", [])
                    ),
                    "history_file": result.get("history_file"),
                    "final_mesh": result.get("final_mesh"),
                }
            )
