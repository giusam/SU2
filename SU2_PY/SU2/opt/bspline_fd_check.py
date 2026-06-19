#!/usr/bin/env python

"""Finite-difference diagnostic for B-spline sensitivity weighting."""

import argparse
import csv
import math
import os
import sys
from pathlib import Path

from SU2.opt.bspline_modes import BSplineModeError, load_mode_spec
from SU2.opt.bspline_su2_driver import (
    BSplineSU2DriverError,
    active_coefficient_vector,
    active_mode_ids,
    build_eval_commands,
    build_eval_paths,
    command_to_string,
    ensure_adjoint_solution_input,
    patch_config_template,
    read_gradient_vector,
    read_objective_from_history,
    run_command,
    update_mode_coefficients,
    write_mode_spec,
)


FIELDNAMES = [
    "mode_id",
    "eps",
    "fd",
    "grad_NODAL",
    "grad_DENSITY",
    "relerr_NODAL",
    "relerr_DENSITY",
    "ratio_NODAL",
    "ratio_DENSITY",
]


def _relative_path(target, start_dir):
    return os.path.relpath(Path(target), Path(start_dir))


def _eps_values(values):
    result = []
    for value in values or [1.0e-5]:
        eps = float(value)
        if eps <= 0.0 or not math.isfinite(eps):
            raise BSplineSU2DriverError("--eps values must be positive and finite")
        result.append(eps)
    return result


def _mode_ids(requested, available):
    available = [str(mode_id) for mode_id in available]
    if requested:
        missing = [mode_id for mode_id in requested if mode_id not in available]
        if missing:
            raise BSplineSU2DriverError(
                "requested mode_id(s) not active: {}".format(", ".join(missing))
            )
        return [str(mode_id) for mode_id in requested]
    return available[:2]


def _path_token(value):
    return "".join(char if char.isalnum() or char in ("_", "-") else "_" for char in str(value))


def _patch_eval_files(args, mode_spec, coefficients, paths):
    paths.eval_dir.mkdir(parents=True, exist_ok=False)
    for directory in {paths.deform_dir, paths.direct_dir, paths.adjoint_dir}:
        directory.mkdir(parents=True, exist_ok=True)

    write_mode_spec(update_mode_coefficients(mode_spec, coefficients), paths.modes_current)
    patch_config_template(
        args.def_template,
        paths.def_cfg,
        {
            "MESH_FILENAME": str(Path(args.base_mesh).resolve()),
            "MESH_OUT_FILENAME": paths.deformed_mesh.name,
            "DV_KIND": "SURFACE_FILE",
            "DV_MARKER": [args.marker],
            "DV_FILENAME": paths.surface_positions.name,
        },
    )

    primal_mesh_filename = (
        paths.deformed_mesh.name
        if paths.eval_layout == "FLAT"
        else _relative_path(paths.deformed_mesh, paths.direct_dir)
    )
    adjoint_mesh_filename = (
        paths.deformed_mesh.name
        if paths.eval_layout == "FLAT"
        else _relative_path(paths.deformed_mesh, paths.adjoint_dir)
    )
    adjoint_flow_solution = (
        paths.primal_solution.name
        if paths.eval_layout == "FLAT"
        else _relative_path(paths.primal_solution, paths.adjoint_dir)
    )
    adjoint_flow_restart = (
        paths.primal_restart.name
        if paths.eval_layout == "FLAT"
        else _relative_path(paths.primal_restart, paths.adjoint_dir)
    )

    shared_primal_updates = {
        "SOLUTION_FILENAME": paths.primal_solution.name,
        "RESTART_FILENAME": paths.primal_restart.name,
        "SOLUTION_ADJ_FILENAME": paths.adjoint_solution.name,
        "RESTART_ADJ_FILENAME": paths.adjoint_restart.name,
        "SURFACE_ADJ_FILENAME": paths.surface_adjoint.stem,
        "VOLUME_ADJ_FILENAME": paths.volume_adjoint.name,
        "TABULAR_FORMAT": "CSV",
    }
    patch_config_template(
        args.primal_template,
        paths.primal_cfg,
        dict(
            shared_primal_updates,
            MESH_FILENAME=primal_mesh_filename,
            MESH_OUT_FILENAME=paths.primal_mesh_out.name,
            CONV_FILENAME=paths.primal_history.stem,
            HISTORY_OUTPUT=["ITER", "RMS_RES", "AERO_COEFF"],
            SCREEN_OUTPUT=["INNER_ITER", "RMS_RES", "LIFT", "DRAG"],
        ),
    )
    patch_config_template(
        args.adjoint_template,
        paths.adjoint_cfg,
        dict(
            shared_primal_updates,
            MESH_FILENAME=adjoint_mesh_filename,
            MESH_OUT_FILENAME=paths.adjoint_mesh_out.name,
            CONV_FILENAME=paths.adjoint_history.stem,
            OBJECTIVE_FUNCTION=args.objective_adjoint,
            MATH_PROBLEM="DISCRETE_ADJOINT",
            SOLUTION_FILENAME=adjoint_flow_solution,
            RESTART_FILENAME=adjoint_flow_restart,
            SURFACE_ADJ_FILENAME=paths.surface_adjoint.stem,
            VOLUME_ADJ_FILENAME=paths.volume_adjoint.name,
            HISTORY_OUTPUT=["ITER", "RMS_RES", "AERO_COEFF"],
            SCREEN_OUTPUT=["INNER_ITER", "RMS_RES"],
        ),
    )


def _run_eval(
    args,
    mode_spec,
    coefficients,
    eval_name,
    run_adjoint=False,
    sensitivity_weighting="NODAL",
):
    paths = build_eval_paths(
        Path(args.workdir) / eval_name,
        eval_layout=args.eval_layout,
        objective_adjoint=args.objective_adjoint,
    )
    _patch_eval_files(args, mode_spec, coefficients, paths)
    commands = build_eval_commands(
        paths,
        Path(args.base_mesh).resolve(),
        args.marker,
        mpi_prefix=args.mpi,
        python_executable=args.python_executable,
        sensitivity_weighting=sensitivity_weighting,
    )
    if args.show_commands:
        for name in ("bspline_def", "def", "primal"):
            print("[BSPLINE_FD_CHECK] {}: {}".format(name, command_to_string(commands[name])))
        if run_adjoint:
            print("[BSPLINE_FD_CHECK] adjoint: {}".format(command_to_string(commands["adjoint"])))
            print("[BSPLINE_FD_CHECK] bspline_dot: {}".format(command_to_string(commands["bspline_dot"])))

    run_command(
        commands["bspline_def"],
        paths.deform_dir,
        paths.bspline_def_log,
        show_command=False,
        stream_output=args.stream_solver_output,
        stage="fd_check_bspline_def",
    )
    run_command(
        commands["def"],
        paths.deform_dir,
        paths.su2_def_log,
        show_command=False,
        stream_output=args.stream_solver_output,
        stage="fd_check_su2_def",
    )
    run_command(
        commands["primal"],
        paths.direct_dir,
        paths.su2_cfd_log,
        show_command=False,
        stream_output=args.stream_solver_output,
        stage="fd_check_su2_cfd",
    )
    objective = read_objective_from_history(paths.primal_history, args.objective_column)

    if run_adjoint:
        ensure_adjoint_solution_input(paths.primal_restart, paths.primal_solution)
        run_command(
            commands["adjoint"],
            paths.adjoint_dir,
            paths.su2_cfd_ad_log,
            show_command=False,
            stream_output=args.stream_solver_output,
            stage="fd_check_su2_cfd_ad",
        )
        run_command(
            commands["bspline_dot"],
            paths.eval_dir,
            paths.bspline_dot_log,
            show_command=False,
            stream_output=args.stream_solver_output,
            stage="fd_check_bspline_dot",
        )
    return paths, objective


def _relative_error(fd_value, gradient):
    denom = max(abs(float(fd_value)), abs(float(gradient)), 1.0e-30)
    return abs(float(gradient) - float(fd_value)) / denom


def _ratio(fd_value, gradient):
    if abs(float(fd_value)) <= 1.0e-30:
        return math.inf
    return float(gradient) / float(fd_value)


def run_fd_check(args):
    args.workdir = Path(args.workdir).resolve()
    args.workdir.mkdir(parents=True, exist_ok=True)
    args.eval_layout = str(args.eval_layout).upper()
    args.objective_adjoint = str(args.objective_adjoint or "drag").strip() or "drag"
    args.eps = _eps_values(args.eps)
    args.python_executable = args.python_executable or sys.executable or "python3"

    mode_spec = load_mode_spec(args.modes)
    mode_ids = active_mode_ids(mode_spec)
    coefficients = active_coefficient_vector(mode_spec)
    selected_mode_ids = _mode_ids(args.mode_id, mode_ids)

    gradient_by_weighting = {}
    _base_paths, base_objective = _run_eval(
        args,
        mode_spec,
        coefficients,
        "base_NODAL",
        run_adjoint=True,
        sensitivity_weighting="NODAL",
    )
    gradient_by_weighting["NODAL"] = dict(
        zip(mode_ids, read_gradient_vector(_base_paths.gradients, mode_ids))
    )

    density_paths, _density_objective = _run_eval(
        args,
        mode_spec,
        coefficients,
        "base_DENSITY",
        run_adjoint=True,
        sensitivity_weighting="DENSITY",
    )
    gradient_by_weighting["DENSITY"] = dict(
        zip(mode_ids, read_gradient_vector(density_paths.gradients, mode_ids))
    )

    print("[BSPLINE_FD_CHECK] Base {} = {:.15g}".format(args.objective_column, base_objective))
    print(
        "mode_id eps FD grad_NODAL grad_DENSITY relerr_NODAL relerr_DENSITY ratio_NODAL ratio_DENSITY"
    )

    rows = []
    for mode_id in selected_mode_ids:
        index = mode_ids.index(mode_id)
        for eps in args.eps:
            plus = list(coefficients)
            minus = list(coefficients)
            plus[index] += eps
            minus[index] -= eps
            _plus_paths, j_plus = _run_eval(
                args,
                mode_spec,
                plus,
                "fd_{}_eps{}_plus".format(_path_token(mode_id), _path_token(eps)),
                run_adjoint=False,
            )
            _minus_paths, j_minus = _run_eval(
                args,
                mode_spec,
                minus,
                "fd_{}_eps{}_minus".format(_path_token(mode_id), _path_token(eps)),
                run_adjoint=False,
            )
            fd_value = (float(j_plus) - float(j_minus)) / (2.0 * float(eps))
            grad_nodal = gradient_by_weighting["NODAL"][mode_id]
            grad_density = gradient_by_weighting["DENSITY"][mode_id]
            row = {
                "mode_id": mode_id,
                "eps": eps,
                "fd": fd_value,
                "grad_NODAL": grad_nodal,
                "grad_DENSITY": grad_density,
                "relerr_NODAL": _relative_error(fd_value, grad_nodal),
                "relerr_DENSITY": _relative_error(fd_value, grad_density),
                "ratio_NODAL": _ratio(fd_value, grad_nodal),
                "ratio_DENSITY": _ratio(fd_value, grad_density),
            }
            rows.append(row)
            print(
                "{mode_id} {eps:.6e} {fd:.6e} {grad_NODAL:.6e} {grad_DENSITY:.6e} "
                "{relerr_NODAL:.6e} {relerr_DENSITY:.6e} {ratio_NODAL:.6e} {ratio_DENSITY:.6e}".format(
                    **row
                )
            )

    output = args.workdir / "fd_check_results.csv"
    with output.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print("[BSPLINE_FD_CHECK] Wrote {}".format(output))
    return rows


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Finite-difference check for B-spline sensitivity weighting."
    )
    parser.add_argument("--modes", required=True)
    parser.add_argument("--base-mesh", required=True)
    parser.add_argument("--marker", required=True)
    parser.add_argument("--def-template", required=True)
    parser.add_argument("--primal-template", required=True)
    parser.add_argument("--adjoint-template", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--objective-column", default="CD")
    parser.add_argument("--objective-adjoint", default="drag")
    parser.add_argument("--eval-layout", default="DSN", choices=("DSN", "FLAT"))
    parser.add_argument("--mode-id", action="append", default=None)
    parser.add_argument("--eps", action="append", type=float, default=None)
    parser.add_argument("--mpi", default="")
    parser.add_argument("--python-executable", default=None)
    parser.add_argument("--show-commands", action="store_true", default=False)
    parser.add_argument("--stream-solver-output", action="store_true", default=False)
    return parser


def main(argv=None):
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        run_fd_check(args)
    except (BSplineSU2DriverError, BSplineModeError, OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
