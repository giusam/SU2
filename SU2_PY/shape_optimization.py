#!/usr/bin/env python
## \file shape_optimization.py
## \brief Python script for performing the shape optimization.

import contextlib
import copy
import os
import sys
import shutil
from optparse import OptionParser

sys.path.append(os.environ["SU2_RUN"])
import SU2

from SU2.opt.thickness_constraint import build_thickness_constraint_from_config
from SU2.opt.progressive_trigger import build_online_trigger_opts
from SU2.opt.progressive_hh import (
    get_progressive_hh_options,
    is_symmetric_reduced,
    build_initial_level,
    build_next_level,
    build_spring_reallocated_level,
    write_level_config,
    collect_level_result,
    should_refine,
    append_selection_history_csv,
)
from SU2.opt.progressive_ffd import (
    get_progressive_ffd_options,
    build_initial_ffd_level,
    build_next_ffd_level,
    build_ffd_spring_reallocated_level,
    refresh_adaptive_scoring_baseline,
    prepare_progressive_ffd_input,
    write_ffd_level_config,
)
from SU2.opt.progressive_hh_projection import (
    _compute_dot_candidate_scores as _hh_scorer,
)
from SU2.opt.progressive_ffd_projection import (
    _compute_ffd_dot_candidate_scores as _ffd_scorer,
)
from SU2.opt.progressive_design import (
    find_design_mesh,
    require_direct_convergence,
    resolve_design_directory,
    write_ranking_design_manifest,
)


def _verify_trigger_batch_consistency(project, new_level, label):
    """Compare the post-adjoint trigger batch with the real refine batch.

    Logs a warning on mismatch; does not abort (the refine result is
    authoritative — the probe ran on an earlier adjoint snapshot).
    """
    trigger_state = getattr(project, "trigger_state", None) or {}
    if trigger_state.get("trajectory_trigger_batch"):
        trigger_name = "TRAJECTORY_READY"
        trigger_batch = trigger_state["trajectory_trigger_batch"]
    else:
        trigger_name = "BATCH_STABILITY"
        trigger_batch = trigger_state.get("trigger_batch")
    if not trigger_batch:
        return

    metadata = getattr(new_level, "selection_metadata", None) or {}
    selected = metadata.get("selected", [])
    refine_batch = tuple(
        (str(c.get("side", "")).upper(), round(float(c.get("x", 0.0)), 12))
        for c in selected
    )

    if refine_batch == tuple(trigger_batch):
        sys.stdout.write(
            f"[{label}] {trigger_name} | refine batch matches trigger batch\n"
        )
    else:
        sys.stdout.write(
            f"[{label}] {trigger_name} | WARNING: refine batch differs from "
            f"trigger batch\n"
            f"[{label}]   trigger: {list(trigger_batch)}\n"
            f"[{label}]   refine : {list(refine_batch)}\n"
        )


def _make_post_adjoint_probe(level, opts, scorer, trigger_name):
    """Build an isolated scorer closure for a post-adjoint trigger.

    The scorer normally writes artifacts (candidate meshes, DOT_ONLY_* dirs,
    FFD_SELECTED_CANDIDATE, score CSV/JSON) into level.workdir.  To keep probe
    calls from touching the real level directory, each call runs on a shallow
    clone of the level whose workdir is a scratch directory containing:
      - a symlink to the real DESIGNS tree (adjoint assets, read-only usage)
      - a copy of the level config
      - symlinks to any root-level mesh files the scorer may resolve
    All scorer outputs then land inside the scratch directory, which is
    recreated on every call.
    """
    trigger_name = str(trigger_name).upper()
    probe_directories = {
        "BATCH_STABILITY": "BATCH_STABILITY_PROBE",
        "TRAJECTORY_READY": "TRAJECTORY_READY_PROBE",
    }
    if trigger_name not in probe_directories:
        raise ValueError(f"Unsupported post-adjoint trigger {trigger_name!r}")

    real_workdir = os.path.abspath(level.workdir)
    probe_root = os.path.join(real_workdir, probe_directories[trigger_name])

    def _probe(design_folder=None):
        if not design_folder:
            raise RuntimeError(
                f"{trigger_name} cannot identify the DSN of the current adjoint"
            )
        real_design_dir = resolve_design_directory(real_workdir, design_folder)
        require_direct_convergence(real_design_dir)
        real_design_mesh = find_design_mesh(real_design_dir)

        if os.path.isdir(probe_root):
            shutil.rmtree(probe_root)
        os.makedirs(probe_root)

        designs_src = os.path.join(real_workdir, "DESIGNS")
        if os.path.isdir(designs_src):
            os.symlink(designs_src, os.path.join(probe_root, "DESIGNS"))

        cfg_src = os.path.join(real_workdir, level.config_filename)
        if os.path.isfile(cfg_src):
            shutil.copy2(cfg_src, os.path.join(probe_root, level.config_filename))

        for entry in os.listdir(real_workdir):
            if entry.endswith(".su2") and os.path.isfile(
                os.path.join(real_workdir, entry)
            ):
                os.symlink(
                    os.path.join(real_workdir, entry),
                    os.path.join(probe_root, entry),
                )

        probe_level = copy.copy(level)
        probe_level.workdir = probe_root
        probe_design_dir = os.path.join(
            probe_root,
            "DESIGNS",
            os.path.basename(real_design_dir),
        )
        probe_design_mesh = os.path.join(
            probe_design_dir,
            os.path.basename(real_design_mesh),
        )
        cfg_probe = SU2.io.Config(os.path.join(probe_root, level.config_filename))
        objective_name = str(
            cfg_probe.get("OBJECTIVE_FUNCTION", "DRAG")
        ).split(",", 1)[0].strip().upper()
        write_ranking_design_manifest(
            probe_root,
            probe_design_dir,
            probe_design_mesh,
            [],
            objective_name,
        )
        # The scorer prints its full per-candidate trace; during a trigger
        # probe only the compact trigger recap line is of interest.
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                return scorer(probe_level, opts)

    return _probe


HH_TE_BOUND_SCALE_KEYS = (
    "OPT_HH_TE_BOUND_SCALE",
    "PROGRESSIVE_HH_TE_BOUND_SCALE",
)
HH_TE_BOUND_X_MIN_KEYS = (
    "OPT_HH_TE_BOUND_X_MIN",
    "PROGRESSIVE_HH_TE_BOUND_X_MIN",
)
HH_TE_BOUND_KEYS = HH_TE_BOUND_SCALE_KEYS + HH_TE_BOUND_X_MIN_KEYS


def _is_final_progressive_hh_level(hh_opts, ilevel, current_ndv=None):
    nfinal = hh_opts.get("nfinal", None)
    if nfinal is not None and current_ndv is not None:
        return current_ndv >= int(nfinal)

    if nfinal is None:
        return ilevel >= hh_opts["nlevels"] - 1

    return False


def _build_online_trigger_opts(hh_opts, ilevel, current_ndv=None):
    """
    Build the trigger options passed to scipy_tools.py for online triggering.
    The last level is never refined online.
    """
    return build_online_trigger_opts(
        hh_opts["trigger"],
        current_level=ilevel,
        current_ndv=current_ndv,
        final_ndv=hh_opts.get("nfinal", None),
        nlevels=hh_opts.get("nlevels", None),
        window=hh_opts["window"],
        tolerance=hh_opts["tol"],
        filter_tolerance=hh_opts["slope_filter_tol"],
        warmup=hh_opts.get("warmup_iter", 0),
        eps=hh_opts.get("trigger_eps", 1.0e-300),
        patience=hh_opts.get("slope_patience", 1),
        stagnation_tolerance=hh_opts["stag_tol"],
        stagnation_band=hh_opts["stag_band"],
        stagnation_window=hh_opts["stag_window"],
        dwell=hh_opts.get("econ_dwell", 4),
        min_ref=hh_opts.get("econ_min_ref", 4),
        n_max=hh_opts.get("econ_n_max", 40),
        rate_floor=hh_opts.get("econ_rate_floor", 1.0e-4),
    )


def _config_option(config, keys, default):
    for key in keys:
        if key in config:
            return config[key], True
    return default, False


def _remove_hh_te_bound_config_options(config):
    for key in HH_TE_BOUND_KEYS:
        if key in config:
            del config[key]


def _is_sequence(value):
    return isinstance(value, (list, tuple))


def _is_param_entry(value):
    return isinstance(value, str) or _is_sequence(value)


def _definition_blocks(value, n_blocks):
    if n_blocks <= 0:
        return []
    if not _is_sequence(value):
        return [value] * n_blocks

    values = list(value)
    if len(values) == n_blocks:
        return values
    if len(values) == 1:
        return values * n_blocks
    if len(values) < n_blocks:
        return values + [None] * (n_blocks - len(values))
    return values[:n_blocks]


def _definition_param_blocks(value, n_blocks):
    if n_blocks <= 0:
        return []
    if value is None:
        return [None] * n_blocks
    if isinstance(value, str) or not _is_sequence(value):
        return [value] * n_blocks

    values = list(value)
    if len(values) == n_blocks and all(_is_param_entry(v) for v in values):
        return values
    if len(values) == 1:
        return values * n_blocks
    if not all(_is_param_entry(v) for v in values):
        return [values] * n_blocks
    if len(values) < n_blocks:
        return values + [None] * (n_blocks - len(values))
    return values[:n_blocks]


def _definition_param_values(param, size):
    if size <= 0:
        return []
    if isinstance(param, str) or not _is_sequence(param):
        return [param] * size

    values = list(param)
    if size > 1 and len(values) == size and all(_is_param_entry(v) for v in values):
        return values
    return [param] * size


def _expand_definition_dv_metadata(def_dv, n_dv):
    sizes = [int(size) for size in def_dv["SIZE"]]
    n_blocks = len(sizes)
    kinds = _definition_blocks(def_dv.get("KIND", None), n_blocks)
    params = _definition_param_blocks(def_dv.get("PARAM", None), n_blocks)

    metadata = []
    for i_block, size in enumerate(sizes):
        kind = kinds[i_block] if i_block < len(kinds) else None
        param = params[i_block] if i_block < len(params) else None
        for param_i in _definition_param_values(param, size):
            metadata.append((kind, param_i))

    if len(metadata) != n_dv:
        raise ValueError(
            "Expanded DEFINITION_DV metadata length "
            f"{len(metadata)} does not match n_dv={n_dv}"
        )
    return metadata


def _safe_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:
        return None
    return result


def _parse_hicks_henne_xc(param):
    if param is None:
        return None

    if isinstance(param, str):
        tokens = param.strip().strip("()").replace(",", " ").split()
        if len(tokens) < 2:
            return None
        return _safe_float(tokens[1])

    if not _is_sequence(param) or len(param) < 2:
        return None

    return _safe_float(param[1])


def _build_optimizer_bounds(config, def_dv, n_dv, bound_lower, bound_upper, relax_factor):
    base_lower = float(bound_lower) / float(relax_factor)
    base_upper = float(bound_upper) / float(relax_factor)

    scale_value, scale_explicit = _config_option(
        config,
        HH_TE_BOUND_SCALE_KEYS,
        1.0,
    )
    x_min_value, _ = _config_option(
        config,
        HH_TE_BOUND_X_MIN_KEYS,
        0.85,
    )

    try:
        scale = float(scale_value)
    except (TypeError, ValueError):
        raise ValueError(
            "OPT_HH_TE_BOUND_SCALE/PROGRESSIVE_HH_TE_BOUND_SCALE "
            "must be a positive number"
        )
    try:
        x_min = float(x_min_value)
    except (TypeError, ValueError):
        raise ValueError(
            "OPT_HH_TE_BOUND_X_MIN/PROGRESSIVE_HH_TE_BOUND_X_MIN "
            "must be in the range [0, 1]"
        )

    if not scale > 0.0:
        raise ValueError(
            "OPT_HH_TE_BOUND_SCALE/PROGRESSIVE_HH_TE_BOUND_SCALE must be > 0"
        )
    if not 0.0 <= x_min <= 1.0:
        raise ValueError(
            "OPT_HH_TE_BOUND_X_MIN/PROGRESSIVE_HH_TE_BOUND_X_MIN "
            "must be in the range [0, 1]"
        )

    bounds = list(zip([base_lower] * n_dv, [base_upper] * n_dv))
    scaled_indices = []

    if scale_explicit or scale != 1.0:
        metadata = _expand_definition_dv_metadata(def_dv, n_dv)
        for i_dv, (kind, param) in enumerate(metadata):
            if str(kind).upper() != "HICKS_HENNE":
                continue
            xc = _parse_hicks_henne_xc(param)
            if xc is None or xc <= x_min:
                continue
            bounds[i_dv] = (scale * base_lower, scale * base_upper)
            scaled_indices.append(i_dv)

        sys.stdout.write(
            "[OPT_BOUNDS] Hicks-Henne TE scaling | "
            f"x/c>{x_min:.6f} scale={scale:.6e} "
            f"scaled_dv={len(scaled_indices)}/{n_dv} indices={scaled_indices}\n"
        )

    if len(bounds) != n_dv:
        raise ValueError(
            f"Generated optimizer bounds length {len(bounds)} does not match n_dv={n_dv}"
        )

    return bounds


def main():
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="filename", help="read config from FILE", metavar="FILE")
    parser.add_option("-r", "--name", dest="projectname", default="", help="try to restart from project file NAME", metavar="NAME")
    parser.add_option("-n", "--partitions", dest="partitions", default=1, help="number of PARTITIONS", metavar="PARTITIONS")
    parser.add_option("-g", "--gradient", dest="gradient", default=None, help="Method for computing the GRADIENT", metavar="GRADIENT")
    parser.add_option("-o", "--optimization", dest="optimization", default="SLSQP", help="OPTIMIZATION technique", metavar="OPTIMIZATION")
    parser.add_option("-q", "--quiet", dest="quiet", default="True", help="True/False Quiet all SU2 output", metavar="QUIET")
    parser.add_option("-z", "--zones", dest="nzones", default="1", help="Number of Zones", metavar="ZONES")

    (options, args) = parser.parse_args()

    options.partitions = int(options.partitions)
    options.quiet = options.quiet.upper() == "TRUE"
    options.nzones = int(options.nzones)

    base_config = SU2.io.Config(options.filename)
    if options.gradient is None or str(options.gradient).strip() == "":
        options.gradient = str(
            base_config.get("GRADIENT_METHOD", "DISCRETE_ADJOINT")
        ).upper()
    else:
        options.gradient = options.gradient.upper()

    sys.stdout.write("\n-------------------------------------------------------------------------\n")
    sys.stdout.write("|  ___ _   _ ___                                                      |\n")
    sys.stdout.write('| / __| | | |_  )   Release 8.4.0 "Harrier"                           |\n')
    sys.stdout.write("| \\__ \\ |_| |/ /                                                      |\n")
    sys.stdout.write("| |___/\\___//___|   Aerodynamic Shape Optimization Script             |\n")
    sys.stdout.write("|                                                                     |\n")
    sys.stdout.write("-------------------------------------------------------------------------\n")

    hh_opts = get_progressive_hh_options(base_config)
    is_progressive_ffd = (
        hh_opts["enabled"]
        and str(hh_opts.get("param_kind", "HICKS_HENNE")).upper() == "FFD"
    )
    thickness_constraint = None
    if not is_progressive_ffd:
        thickness_constraint = build_thickness_constraint_from_config(base_config)

    if not hh_opts["enabled"]:
        run_single_level(
            options.filename,
            options.projectname,
            options.partitions,
            options.gradient,
            options.optimization,
            options.quiet,
            options.nzones,
            thickness_constraint=thickness_constraint,
        )
        return

    progressive_hh_shape_optimization(
        options.filename,
        options.projectname,
        options.partitions,
        options.gradient,
        options.optimization,
        options.quiet,
        options.nzones,
        thickness_constraint=thickness_constraint,
    )


def run_single_level(
    filename,
    projectname="",
    partitions=0,
    gradient="CONTINUOUS_ADJOINT",
    optimization="SLSQP",
    quiet=False,
    nzones=1,
    trigger_opts=None,
    progressive_hh_opts=None,
    thickness_constraint=None,
    progressive_label=None,
    batch_stability_scorer_fn=None,
    trajectory_ready_scorer_fn=None,
):
    config = SU2.io.Config(filename)
    if thickness_constraint is None:
        thickness_constraint = build_thickness_constraint_from_config(config)

    if "DV_MARKER" in config:
        dv_marker = config["DV_MARKER"]
        while isinstance(dv_marker, (list, tuple)) and len(dv_marker) == 1:
            dv_marker = dv_marker[0]
        config["DV_MARKER"] = dv_marker

    if "DV_KIND" in config:
        dv_kind = config["DV_KIND"]
        if isinstance(dv_kind, (list, tuple)):
            unique_kinds = []
            for k in dv_kind:
                if k not in unique_kinds:
                    unique_kinds.append(k)
            if len(unique_kinds) == 1:
                config["DV_KIND"] = unique_kinds[0]

    config.NUMBER_PART = partitions
    config.NZONES = int(nzones)

    if quiet:
        config.CONSOLE = "CONCISE"

    config.GRADIENT_METHOD = gradient

    its = int(config.OPT_ITERATIONS)
    bound_upper = float(config.OPT_BOUND_UPPER)
    bound_lower = float(config.OPT_BOUND_LOWER)
    relax_factor = float(config.OPT_RELAX_FACTOR)
    gradient_factor = float(config.OPT_GRADIENT_FACTOR)

    def_dv = config.DEFINITION_DV
    n_dv = sum(def_dv["SIZE"])
    accu = float(config.OPT_ACCURACY) * gradient_factor

    x0 = [0.0] * n_dv
    xb = _build_optimizer_bounds(
        config,
        def_dv,
        n_dv,
        bound_lower,
        bound_upper,
        relax_factor,
    )
    _remove_hh_te_bound_config_options(config)

    state = SU2.io.State()
    state.find_files(config)

    if projectname and os.path.exists(projectname):
        project = SU2.io.load_data(projectname)
        project.config = config
    else:
        project = SU2.opt.Project(config, state)

    if trigger_opts is not None:
        project.trigger_opts = dict(trigger_opts)
    else:
        project.trigger_opts = None

    if progressive_hh_opts is not None:
        project.progressive_hh_symmetry = {
            "mode": progressive_hh_opts.get("symmetry_mode", "NONE"),
            "sign": progressive_hh_opts.get("symmetry_sign", -1.0),
        }
    if progressive_label is not None:
        project.progressive_label = str(progressive_label)

    project.refinement_triggered = False

    def _bind_current_adjoint_scorer(scorer_fn, trigger_name):
        if scorer_fn is None:
            return None

        def _score_current_adjoint_design():
            design_folder = getattr(
                project,
                "last_obj_grad_design_folder",
                None,
            )
            if not design_folder:
                detail = getattr(project, "last_obj_grad_design_error", "unknown")
                raise RuntimeError(
                    f"{trigger_name} cannot resolve the current adjoint "
                    "to an exact DSN: "
                    f"{detail}"
                )
            return scorer_fn(design_folder)

        return _score_current_adjoint_design

    project.batch_stability_scorer_fn = _bind_current_adjoint_scorer(
        batch_stability_scorer_fn,
        "BATCH_STABILITY",
    )
    project.trajectory_ready_scorer_fn = _bind_current_adjoint_scorer(
        trajectory_ready_scorer_fn,
        "TRAJECTORY_READY",
    )
    # Trigger CSV logs live in the level workdir (cwd at this point).
    project.batch_stability_log_path = (
        os.path.abspath("batch_stability_log.csv")
        if batch_stability_scorer_fn is not None
        else None
    )
    project.trajectory_ready_log_path = (
        os.path.abspath("trajectory_ready_log.csv")
        if trajectory_ready_scorer_fn is not None
        else None
    )

    if (
        progressive_hh_opts is not None
        and is_symmetric_reduced(progressive_hh_opts)
        and optimization != "SLSQP"
    ):
        raise ValueError(
            "PROGRESSIVE_HH_SYMMETRY_MODE=REDUCED currently supports only SLSQP"
        )

    if thickness_constraint is not None and optimization != "SLSQP":
        raise NotImplementedError(
            "PROGRESSIVE_THICKNESS_CONSTRAINT currently supports only SLSQP"
        )

    project.thickness_constraint = thickness_constraint

    if optimization == "SLSQP":
        SU2.opt.SLSQP(project, x0, xb, its, accu)
    if optimization == "CG":
        SU2.opt.CG(project, x0, xb, its, accu)
    if optimization == "BFGS":
        SU2.opt.BFGS(project, x0, xb, its, accu)
    if optimization == "POWELL":
        SU2.opt.POWELL(project, x0, xb, its, accu)

    # Project._eval saves before SLSQP's callback/final return.  Save once more
    # so the accepted-iterate history and canonical optimizer result used by
    # the level transition are also present in project.pkl.
    project.save()

    if projectname:
        shutil.move("project.pkl", projectname)

    return project


def progressive_hh_shape_optimization(
    filename,
    projectname="",
    partitions=0,
    gradient="CONTINUOUS_ADJOINT",
    optimization="SLSQP",
    quiet=False,
    nzones=1,
    thickness_constraint=None,
):
    base_config = SU2.io.Config(filename)
    hh_opts = get_progressive_hh_options(base_config)

    if str(hh_opts.get("param_kind", "HICKS_HENNE")).upper() == "FFD":
        return progressive_ffd_shape_optimization(
            filename,
            projectname,
            partitions,
            gradient,
            optimization,
            quiet,
            nzones,
            thickness_constraint=thickness_constraint,
        )

    if thickness_constraint is None:
        thickness_constraint = build_thickness_constraint_from_config(base_config)

    old_levels = [
        d for d in os.listdir(".")
        if os.path.isdir(d) and d.startswith("LEVEL_")
    ]

    if old_levels:
        sys.stdout.write("\n[PROGRESSIVE_HH] Cleaning previous LEVEL_* folders\n")
        for d in old_levels:
            sys.stdout.write(f"[PROGRESSIVE_HH] Removing {d}\n")
            shutil.rmtree(d)

    selection_history_csv = "progressive_hh_selection_history.csv"
    if os.path.exists(selection_history_csv):
        os.remove(selection_history_csv)

    level = build_initial_level(base_config, hh_opts)
    final_project = None

    ilevel = 0
    while True:
        cfg_path = write_level_config(base_config, level, hh_opts)
        level_project = os.path.join(level.workdir, level.project_filename)

        sys.stdout.write(f"\n[PROGRESSIVE_HH] Level {ilevel} | NDV = {level.ndv}\n")
        sys.stdout.write(f"[PROGRESSIVE_HH] Upper centers: {level.upper}\n")
        sys.stdout.write(f"[PROGRESSIVE_HH] Lower centers: {level.lower}\n")
        sys.stdout.write(f"[PROGRESSIVE_HH] Mesh source: {level.mesh_source}\n")
        sys.stdout.write(
            "[PROGRESSIVE_HH][SYMMETRY] mode = "
            f"{hh_opts.get('symmetry_mode', 'NONE')}\n"
        )
        sys.stdout.write(
            "[PROGRESSIVE_HH][SYMMETRY] sign = "
            f"{hh_opts.get('symmetry_sign', -1.0)}\n"
        )
        if is_symmetric_reduced(hh_opts):
            sys.stdout.write(
                "[PROGRESSIVE_HH][SYMMETRY] reduced NDV = "
                f"{len(level.upper)}\n"
            )
            sys.stdout.write(
                "[PROGRESSIVE_HH][SYMMETRY] full SU2 HH = "
                f"{level.ndv}\n"
            )

        trigger_opts = _build_online_trigger_opts(
            hh_opts,
            ilevel,
            current_ndv=level.ndv,
        )

        _hh_trigger_name = str(hh_opts.get("trigger", "")).upper()
        _hh_batch_scorer_fn = (
            _make_post_adjoint_probe(
                level,
                hh_opts,
                _hh_scorer,
                "BATCH_STABILITY",
            )
            if _hh_trigger_name == "BATCH_STABILITY"
            else None
        )
        _hh_trajectory_scorer_fn = (
            _make_post_adjoint_probe(
                level,
                hh_opts,
                _hh_scorer,
                "TRAJECTORY_READY",
            )
            if _hh_trigger_name == "TRAJECTORY_READY"
            else None
        )

        cwd = os.getcwd()
        try:
            os.chdir(level.workdir)
            project = run_single_level(
                os.path.basename(cfg_path),
                os.path.basename(level_project),
                partitions,
                gradient,
                optimization,
                quiet,
                nzones,
                trigger_opts=trigger_opts,
                progressive_hh_opts=hh_opts,
                thickness_constraint=thickness_constraint,
                progressive_label="PROGRESSIVE_HH",
                batch_stability_scorer_fn=_hh_batch_scorer_fn,
                trajectory_ready_scorer_fn=_hh_trajectory_scorer_fn,
            )
        finally:
            os.chdir(cwd)

        final_project = level_project
        result = collect_level_result(level, project=project)

        force_refine_after_spring = False

        if getattr(level, "post_opt_spring_pending", False):
            spring_post_action = str(
                hh_opts.get("spring_post_action", "REOPTIMIZE")
            ).upper()
            spring_level = build_spring_reallocated_level(
                level,
                result,
                hh_opts,
                reoptimize=(spring_post_action == "REOPTIMIZE"),
            )
            if spring_level is not None:
                sys.stdout.write(
                    "[PROGRESSIVE_HH] POST_OPT spring applied | "
                    f"post_action={spring_post_action}\n"
                )
                level = spring_level
                if spring_post_action == "REOPTIMIZE":
                    ilevel += 1
                    continue
                sys.stdout.write(
                    "[PROGRESSIVE_HH] Skipping post-spring same-NDV optimization; "
                    "proceeding directly to refinement.\n"
                )
                force_refine_after_spring = True
            else:
                sys.stdout.write(
                    "[PROGRESSIVE_HH][SPRING] WARNING: post-opt spring skipped; "
                    "continuing normal progressive logic\n"
                )

        if _is_final_progressive_hh_level(
            hh_opts,
            ilevel,
            current_ndv=level.ndv,
        ):
            sys.stdout.write(f"[PROGRESSIVE_HH] Stop after final level {ilevel}\n")
            break

        if force_refine_after_spring:
            refine_now = True
        elif hh_opts["trigger"] == "MAX_ITER":
            refine_now = should_refine(result["history"], hh_opts, ilevel)
        else:
            refine_now = bool(getattr(project, "refinement_triggered", False))

        if not refine_now:
            if (
                hh_opts.get("nfinal", None) is not None
                and level.ndv < int(hh_opts["nfinal"])
            ):
                raise RuntimeError(
                    "Progressive HH stopped before reaching "
                    f"NFINAL={hh_opts['nfinal']} (current NDV={level.ndv})"
                )
            sys.stdout.write(f"[PROGRESSIVE_HH] Stop after level {ilevel}\n")
            break

        if hh_opts.get("nfinal", None) is None and ilevel == hh_opts["nlevels"] - 1:
            sys.stdout.write(f"[PROGRESSIVE_HH] Reached maximum level {ilevel}\n")
            break

        if str(hh_opts.get("refinement", "UNIFORM")).upper() == "ADAPTIVE":
            refresh_adaptive_scoring_baseline(
                project,
                level,
                result.get("dv_values"),
                hh_opts,
                label="PROGRESSIVE_HH",
            )
            result = collect_level_result(
                level,
                project=project,
                require_adjoint=True,
            )
            manifest = write_ranking_design_manifest(
                level.workdir,
                result["accepted_design_dir"],
                result["final_mesh"],
                result["dv_values"],
                result["ranking_objective"],
            )
            sys.stdout.write(
                "[PROGRESSIVE_HH] Ranking anchor locked | "
                f"design={result['accepted_design_folder']} "
                f"manifest={manifest}\n"
            )

        ndv_before_refine = level.ndv
        level = build_next_level(level, result, hh_opts)
        _verify_trigger_batch_consistency(project, level, "PROGRESSIVE_HH")
        append_selection_history_csv(
            selection_history_csv,
            getattr(level, "selection_metadata", None),
            result,
        )
        if hh_opts.get("nfinal", None) is not None and level.ndv <= ndv_before_refine:
            sys.stdout.write(
                "[PROGRESSIVE_HH] Stop: refinement did not increase NDV "
                f"before reaching NFINAL={hh_opts['nfinal']}\n"
            )
            break
        ilevel += 1

    if projectname and final_project and os.path.exists(final_project):
        shutil.copy(final_project, projectname)


def progressive_ffd_shape_optimization(
    filename,
    projectname="",
    partitions=0,
    gradient="CONTINUOUS_ADJOINT",
    optimization="SLSQP",
    quiet=False,
    nzones=1,
    thickness_constraint=None,
):
    base_config = SU2.io.Config(filename)
    hh_opts = get_progressive_hh_options(base_config)
    ffd_opts = get_progressive_ffd_options(base_config, hh_opts)

    preparation = prepare_progressive_ffd_input(
        base_config,
        ffd_opts,
        partitions=partitions,
    )
    if ffd_opts.get("ffd_prepare_only", False):
        sys.stdout.write(
            "\n[PROGRESSIVE_FFD_PREP] PREPARE_ONLY completed successfully\n"
        )
        if preparation is not None:
            sys.stdout.write(
                "[PROGRESSIVE_FFD_PREP] Mesh: "
                f"{preparation.get('prepared_mesh')}\n"
            )
            if preparation.get("diagnostics_csv"):
                sys.stdout.write(
                    "[PROGRESSIVE_FFD_PREP] Diagnostics: "
                    f"{preparation.get('diagnostics_csv')}\n"
                )
            if preparation.get("manifest"):
                sys.stdout.write(
                    "[PROGRESSIVE_FFD_PREP] Manifest: "
                    f"{preparation.get('manifest')}\n"
                )
        return preparation

    if thickness_constraint is None:
        thickness_constraint = build_thickness_constraint_from_config(base_config)

    old_levels = [
        d for d in os.listdir(".")
        if os.path.isdir(d) and d.startswith("LEVEL_")
    ]

    if old_levels:
        sys.stdout.write("\n[PROGRESSIVE_FFD] Cleaning previous LEVEL_* folders\n")
        for d in old_levels:
            sys.stdout.write(f"[PROGRESSIVE_FFD] Removing {d}\n")
            shutil.rmtree(d)

    selection_history_csv = "progressive_ffd_selection_history.csv"
    if os.path.exists(selection_history_csv):
        os.remove(selection_history_csv)

    level = build_initial_ffd_level(base_config, ffd_opts)
    final_project = None

    ilevel = 0
    while True:
        cfg_path = write_ffd_level_config(base_config, level, ffd_opts)
        level_project = os.path.join(level.workdir, level.project_filename)

        sys.stdout.write(f"\n[PROGRESSIVE_FFD] Level {ilevel} | NDV = {level.ndv}\n")
        sys.stdout.write(f"[PROGRESSIVE_FFD] FFD DV kind: {level.ffd_dv_kind}\n")
        sys.stdout.write(f"[PROGRESSIVE_FFD] Domain mode: {level.domain_mode}\n")
        if getattr(level, "dual_box", False):
            sys.stdout.write(
                "[PROGRESSIVE_FFD_DUAL] Box tags: "
                f"{level.upper_box_tag}, {level.lower_box_tag}\n"
            )
            sys.stdout.write(
                "[PROGRESSIVE_FFD_DUAL] Upper columns "
                f"({len(level.upper_columns)}): {level.upper_columns}\n"
            )
            sys.stdout.write(
                "[PROGRESSIVE_FFD_DUAL] Lower columns "
                f"({len(level.lower_columns)}): {level.lower_columns}\n"
            )
        else:
            sys.stdout.write(f"[PROGRESSIVE_FFD] Box tag: {level.ffd_box_tag}\n")
            sys.stdout.write(f"[PROGRESSIVE_FFD] Active columns: {level.columns}\n")
        sys.stdout.write(f"[PROGRESSIVE_FFD] Mesh source: {level.mesh_source}\n")

        trigger_opts = _build_online_trigger_opts(
            ffd_opts,
            ilevel,
            current_ndv=level.ndv,
        )

        _ffd_trigger_name = str(ffd_opts.get("trigger", "")).upper()
        _ffd_batch_scorer_fn = (
            _make_post_adjoint_probe(
                level,
                ffd_opts,
                _ffd_scorer,
                "BATCH_STABILITY",
            )
            if _ffd_trigger_name == "BATCH_STABILITY"
            else None
        )
        _ffd_trajectory_scorer_fn = (
            _make_post_adjoint_probe(
                level,
                ffd_opts,
                _ffd_scorer,
                "TRAJECTORY_READY",
            )
            if _ffd_trigger_name == "TRAJECTORY_READY"
            else None
        )

        cwd = os.getcwd()
        try:
            os.chdir(level.workdir)
            project = run_single_level(
                os.path.basename(cfg_path),
                os.path.basename(level_project),
                partitions,
                gradient,
                optimization,
                quiet,
                nzones,
                trigger_opts=trigger_opts,
                progressive_hh_opts=None,
                thickness_constraint=thickness_constraint,
                progressive_label="PROGRESSIVE_FFD",
                batch_stability_scorer_fn=_ffd_batch_scorer_fn,
                trajectory_ready_scorer_fn=_ffd_trajectory_scorer_fn,
            )
        finally:
            os.chdir(cwd)

        final_project = level_project
        result = collect_level_result(level, project=project)

        force_refine_after_spring = False

        if getattr(level, "post_opt_spring_pending", False):
            spring_post_action = str(
                ffd_opts.get("spring_post_action", "REOPTIMIZE")
            ).upper()
            spring_level = build_ffd_spring_reallocated_level(
                level,
                result,
                ffd_opts,
                reoptimize=(spring_post_action == "REOPTIMIZE"),
            )
            if spring_level is not None:
                sys.stdout.write(
                    "[PROGRESSIVE_FFD] POST_OPT spring applied | "
                    f"post_action={spring_post_action}\n"
                )
                level = spring_level
                if spring_post_action == "REOPTIMIZE":
                    ilevel += 1
                    continue
                sys.stdout.write(
                    "[PROGRESSIVE_FFD] Skipping post-spring same-NDV optimization; "
                    "proceeding directly to refinement.\n"
                )
                force_refine_after_spring = True
            else:
                sys.stdout.write(
                    "[PROGRESSIVE_FFD][SPRING] WARNING: post-opt spring skipped; "
                    "continuing normal progressive logic\n"
                )

        if _is_final_progressive_hh_level(
            ffd_opts,
            ilevel,
            current_ndv=level.ndv,
        ):
            sys.stdout.write(f"[PROGRESSIVE_FFD] Stop after final level {ilevel}\n")
            break

        if force_refine_after_spring:
            refine_now = True
        elif ffd_opts["trigger"] == "MAX_ITER":
            refine_now = should_refine(result["history"], ffd_opts, ilevel)
        else:
            refine_now = bool(getattr(project, "refinement_triggered", False))

        if not refine_now:
            if (
                ffd_opts.get("nfinal", None) is not None
                and level.ndv < int(ffd_opts["nfinal"])
            ):
                raise RuntimeError(
                    "Progressive FFD stopped before reaching "
                    f"NFINAL={ffd_opts['nfinal']} (current NDV={level.ndv})"
                )
            sys.stdout.write(f"[PROGRESSIVE_FFD] Stop after level {ilevel}\n")
            break

        if ffd_opts.get("nfinal", None) is None and ilevel == ffd_opts["nlevels"] - 1:
            sys.stdout.write(f"[PROGRESSIVE_FFD] Reached maximum level {ilevel}\n")
            break

        if str(ffd_opts.get("refinement", "UNIFORM")).upper() == "ADAPTIVE":
            refresh_adaptive_scoring_baseline(
                project,
                level,
                result.get("dv_values"),
                ffd_opts,
                label="PROGRESSIVE_FFD",
            )
            result = collect_level_result(
                level,
                project=project,
                require_adjoint=True,
            )
            manifest = write_ranking_design_manifest(
                level.workdir,
                result["accepted_design_dir"],
                result["final_mesh"],
                result["dv_values"],
                result["ranking_objective"],
            )
            sys.stdout.write(
                "[PROGRESSIVE_FFD] Ranking anchor locked | "
                f"design={result['accepted_design_folder']} "
                f"manifest={manifest}\n"
            )

        ndv_before_refine = level.ndv
        level = build_next_ffd_level(level, result, ffd_opts)
        _verify_trigger_batch_consistency(project, level, "PROGRESSIVE_FFD")
        append_selection_history_csv(
            selection_history_csv,
            getattr(level, "selection_metadata", None),
            result,
        )
        if ffd_opts.get("nfinal", None) is not None and level.ndv <= ndv_before_refine:
            message = (
                "Progressive FFD refinement did not increase NDV before "
                f"reaching NFINAL={ffd_opts['nfinal']}"
            )
            raise RuntimeError(message)
        ilevel += 1

    if projectname and final_project and os.path.exists(final_project):
        shutil.copy(final_project, projectname)


if __name__ == "__main__":
    main()
