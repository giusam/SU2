#!/usr/bin/env python

import os
import copy
import csv
import glob
import shutil

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
        "marker": str(config.get("DV_MARKER", "Airfoil")),
        "scale": scale,
    }


def initial_centers(n0):
    """
    Uniform interior points in (0,1):
    x_i = (i+1)/(n0+1)
    """
    if n0 <= 0:
        return []

    return [(i + 1) / float(n0 + 1) for i in range(n0)]


def refine_uniform(centers):
    """
    Refinement using virtual boundaries at 0 and 1.
    Add midpoints of:
    [0, x1], [x1,x2], ..., [xn,1]
    without ever adding 0 or 1 as centers.
    """
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


def build_next_level(prev_level, result):
    next_id = prev_level.level_id + 1

    next_mesh = result.get("final_mesh", None)
    if next_mesh is None:
        next_mesh = prev_level.mesh_source

    return HHLevel(
        level_id=next_id,
        upper=refine_uniform(prev_level.upper),
        lower=refine_uniform(prev_level.lower),
        workdir=f"LEVEL_{next_id}",
        config_filename=f"config_level{next_id}.cfg",
        project_filename=f"project_level{next_id}.pkl",
        mesh_source=next_mesh,
    )


def make_hh_definition(level, scale, marker_name):
    """
    DEFINITION_DV come dict, compatibile con cfg.dump().
    Per Hicks-Henne:
      PARAM = [surface_flag, x_location]
      1.0 = upper
      0.0 = lower
    """
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
    ]

    for key in progressive_keys:
        if key in cfg:
            del cfg[key]


def _prepare_local_mesh(cfg, level):
    """
    Copy the mesh source into the level folder and use a local basename.
    """
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

    # Intermediate levels use the reduced iteration budget,
    # final level uses the original OPT_ITERATIONS from the base config.
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
        cfg["RESTART_FILENAME"] = _resolve_from_cfg_dir(base_config, cfg["RESTART_FILENAME"])

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
    """
    Find the final deformed mesh produced by the level.
    Priority:
    1) deformed mesh in DESIGNS
    2) deformed mesh in root
    3) any .su2 mesh as fallback
    """
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
    """
    Offline refinement logic is only kept for MAX_ITER.
    All other triggers are now handled online in scipy_tools.py.
    """
    if level_id >= opts["nlevels"] - 1:
        return False

    if opts["trigger"] == "MAX_ITER":
        return True

    return False