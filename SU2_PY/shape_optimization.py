#!/usr/bin/env python
## \file shape_optimization.py
## \brief Python script for performing the shape optimization.

import os
import sys
import shutil
from optparse import OptionParser

sys.path.append(os.environ["SU2_RUN"])
import SU2

from SU2.opt.thickness_constraint import build_thickness_constraint_from_config
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
    write_ffd_level_config,
)


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
    if _is_final_progressive_hh_level(hh_opts, ilevel, current_ndv=current_ndv):
        return None

    trigger = hh_opts["trigger"]
    warmup_iter = int(hh_opts.get("warmup_iter", 0))

    if trigger == "MAX_ITER":
        return None

    if trigger in ("SLOPE_EFFICIENCY_TRIGGER", "SLOPE_EFFICIENCY_FILTERED"):
        return {
            "trigger": trigger,
            "window": hh_opts["window"],
            "tol": hh_opts["tol"],
            "filter_tol": hh_opts["slope_filter_tol"],
            "warmup_iter": warmup_iter,
        }

    if trigger == "SLOPE_EFFICIENCY_BEST_LOG":
        return {
            "trigger": "SLOPE_EFFICIENCY_BEST_LOG",
            "window": hh_opts["window"],
            "tol": hh_opts["tol"],
            "warmup_iter": warmup_iter,
            "eps": hh_opts.get("trigger_eps", 1.0e-300),
            "patience": hh_opts.get("slope_patience", 1),
        }

    if trigger == "STAGNATION_TRIGGER":
        return {
            "trigger": trigger,
            "stag_tol": hh_opts["stag_tol"],
            "stag_band": hh_opts["stag_band"],
            "stag_window": hh_opts["stag_window"],
            "warmup_iter": warmup_iter,
        }

    return None


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
    xb_low = [float(bound_lower) / float(relax_factor)] * n_dv
    xb_up = [float(bound_upper) / float(relax_factor)] * n_dv
    xb = list(zip(xb_low, xb_up))

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

    project.refinement_triggered = False

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
    if thickness_constraint is None:
        thickness_constraint = build_thickness_constraint_from_config(base_config)

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
            )
        finally:
            os.chdir(cwd)

        final_project = level_project
        result = collect_level_result(level)
        result["dv_values"] = getattr(
            project,
            "opt_dv_values",
            getattr(project, "last_dv_values", None),
        )

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
            sys.stdout.write(f"[PROGRESSIVE_HH] Stop after level {ilevel}\n")
            break

        if hh_opts.get("nfinal", None) is None and ilevel == hh_opts["nlevels"] - 1:
            sys.stdout.write(f"[PROGRESSIVE_HH] Reached maximum level {ilevel}\n")
            break

        ndv_before_refine = level.ndv
        level = build_next_level(level, result, hh_opts)
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
        sys.stdout.write(f"[PROGRESSIVE_FFD] Box tag: {level.ffd_box_tag}\n")
        sys.stdout.write(f"[PROGRESSIVE_FFD] Domain mode: {level.domain_mode}\n")
        sys.stdout.write(f"[PROGRESSIVE_FFD] Active columns: {level.columns}\n")
        sys.stdout.write(f"[PROGRESSIVE_FFD] Mesh source: {level.mesh_source}\n")

        trigger_opts = _build_online_trigger_opts(
            ffd_opts,
            ilevel,
            current_ndv=level.ndv,
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
            )
        finally:
            os.chdir(cwd)

        final_project = level_project
        result = collect_level_result(level)
        result["dv_values"] = getattr(
            project,
            "opt_dv_values",
            getattr(project, "last_dv_values", None),
        )

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
            sys.stdout.write(f"[PROGRESSIVE_FFD] Stop after level {ilevel}\n")
            break

        if ffd_opts.get("nfinal", None) is None and ilevel == ffd_opts["nlevels"] - 1:
            sys.stdout.write(f"[PROGRESSIVE_FFD] Reached maximum level {ilevel}\n")
            break

        ndv_before_refine = level.ndv
        level = build_next_ffd_level(level, result, ffd_opts)
        append_selection_history_csv(
            selection_history_csv,
            getattr(level, "selection_metadata", None),
            result,
        )
        if ffd_opts.get("nfinal", None) is not None and level.ndv <= ndv_before_refine:
            sys.stdout.write(
                "[PROGRESSIVE_FFD] Stop: refinement did not increase NDV "
                f"before reaching NFINAL={ffd_opts['nfinal']}\n"
            )
            break
        ilevel += 1

    if projectname and final_project and os.path.exists(final_project):
        shutil.copy(final_project, projectname)


if __name__ == "__main__":
    main()
