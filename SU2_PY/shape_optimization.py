#!/usr/bin/env python
## \file shape_optimization.py
## \brief Python script for performing the shape optimization.

import os
import sys
import shutil
from optparse import OptionParser

sys.path.append(os.environ["SU2_RUN"])
import SU2

from SU2.opt.progressive_hh import (
    get_progressive_hh_options,
    build_initial_level,
    build_next_level,
    write_level_config,
    collect_level_result,
    should_refine,
)


def _build_online_trigger_opts(hh_opts, ilevel):
    """
    Build the trigger options passed to scipy_tools.py for online triggering.
    The last level is never refined online.
    """
    if ilevel == hh_opts["nlevels"] - 1:
        return None

    trigger = hh_opts["trigger"]

    if trigger == "MAX_ITER":
        return None

    if trigger == "SLOPE_EFFICIENCY_TRIGGER":
        return {
            "trigger": trigger,
            "window": hh_opts["window"],
            "tol": hh_opts["tol"],
            "filter_tol": hh_opts["slope_filter_tol"],
        }

    if trigger == "STAGNATION_TRIGGER":
        return {
            "trigger": trigger,
            "stag_tol": hh_opts["stag_tol"],
            "stag_band": hh_opts["stag_band"],
            "stag_window": hh_opts["stag_window"],
        }

    return None


def main():
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="filename", help="read config from FILE", metavar="FILE")
    parser.add_option("-r", "--name", dest="projectname", default="", help="try to restart from project file NAME", metavar="NAME")
    parser.add_option("-n", "--partitions", dest="partitions", default=1, help="number of PARTITIONS", metavar="PARTITIONS")
    parser.add_option("-g", "--gradient", dest="gradient", default="DISCRETE_ADJOINT", help="Method for computing the GRADIENT", metavar="GRADIENT")
    parser.add_option("-o", "--optimization", dest="optimization", default="SLSQP", help="OPTIMIZATION technique", metavar="OPTIMIZATION")
    parser.add_option("-q", "--quiet", dest="quiet", default="True", help="True/False Quiet all SU2 output", metavar="QUIET")
    parser.add_option("-z", "--zones", dest="nzones", default="1", help="Number of Zones", metavar="ZONES")

    (options, args) = parser.parse_args()

    options.partitions = int(options.partitions)
    options.quiet = options.quiet.upper() == "TRUE"
    options.gradient = options.gradient.upper()
    options.nzones = int(options.nzones)

    sys.stdout.write("\n-------------------------------------------------------------------------\n")
    sys.stdout.write("|  ___ _   _ ___                                                      |\n")
    sys.stdout.write('| / __| | | |_  )   Release 8.4.0 "Harrier"                           |\n')
    sys.stdout.write("| \\__ \\ |_| |/ /                                                      |\n")
    sys.stdout.write("| |___/\\___//___|   Aerodynamic Shape Optimization Script             |\n")
    sys.stdout.write("|                                                                     |\n")
    sys.stdout.write("-------------------------------------------------------------------------\n")

    base_config = SU2.io.Config(options.filename)
    hh_opts = get_progressive_hh_options(base_config)

    if not hh_opts["enabled"]:
        run_single_level(
            options.filename,
            options.projectname,
            options.partitions,
            options.gradient,
            options.optimization,
            options.quiet,
            options.nzones,
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
):
    config = SU2.io.Config(filename)

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

    project.refinement_triggered = False

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
):
    base_config = SU2.io.Config(filename)
    hh_opts = get_progressive_hh_options(base_config)

    old_levels = [
        d for d in os.listdir(".")
        if os.path.isdir(d) and d.startswith("LEVEL_")
    ]

    if old_levels:
        sys.stdout.write("\n[PROGRESSIVE_HH] Cleaning previous LEVEL_* folders\n")
        for d in old_levels:
            sys.stdout.write(f"[PROGRESSIVE_HH] Removing {d}\n")
            shutil.rmtree(d)

    level = build_initial_level(base_config, hh_opts)
    final_project = None

    for ilevel in range(hh_opts["nlevels"]):
        cfg_path = write_level_config(base_config, level, hh_opts)
        level_project = os.path.join(level.workdir, level.project_filename)

        sys.stdout.write(f"\n[PROGRESSIVE_HH] Level {ilevel} | NDV = {level.ndv}\n")
        sys.stdout.write(f"[PROGRESSIVE_HH] Upper centers: {level.upper}\n")
        sys.stdout.write(f"[PROGRESSIVE_HH] Lower centers: {level.lower}\n")
        sys.stdout.write(f"[PROGRESSIVE_HH] Mesh source: {level.mesh_source}\n")

        trigger_opts = _build_online_trigger_opts(hh_opts, ilevel)

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
            )
        finally:
            os.chdir(cwd)

        final_project = level_project
        result = collect_level_result(level)
        result["final_grad"] = getattr(project, "last_obj_grad", None)
        result["final_grad_x"] = getattr(project, "last_obj_grad_x", None)

        if result["final_grad"] is None:
            sys.stdout.write("[PROGRESSIVE_HH] final_grad not available\n")
        else:
            sys.stdout.write(
                f"[PROGRESSIVE_HH] final_grad captured | size = {len(result['final_grad'])}\n"
            )
            sys.stdout.write(
                f"[PROGRESSIVE_HH] final_grad entries = {result['final_grad']}\n"
            )

        # Online trigger logic for SLOPE/STAGNATION.
        # Offline logic only remains for MAX_ITER.
        if hh_opts["trigger"] == "MAX_ITER":
            refine_now = should_refine(result["history"], hh_opts, ilevel)
        else:
            refine_now = bool(getattr(project, "refinement_triggered", False))

        if not refine_now:
            sys.stdout.write(f"[PROGRESSIVE_HH] Stop after level {ilevel}\n")
            break

        if ilevel == hh_opts["nlevels"] - 1:
            sys.stdout.write(f"[PROGRESSIVE_HH] Reached maximum level {ilevel}\n")
            break

        level = build_next_level(level, result, hh_opts)

    if projectname and final_project and os.path.exists(final_project):
        shutil.copy(final_project, projectname)


if __name__ == "__main__":
    main()