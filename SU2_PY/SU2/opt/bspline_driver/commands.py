"""Subprocess command and evaluation-directory helpers for the driver."""

import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from SU2.opt.bspline_dot import (
    BSplineDotError,
    normalize_sensitivity_weighting,
)
from SU2.opt.bspline_modes import (
    BSplineModeError,
    normalize_deformation_direction_mode,
    normalize_surface_mode,
)
from .constants import ALLOWED_EVAL_LAYOUTS
from .errors import (
    BSplineSU2DriverError,
    _normalized_name,
)

@dataclass(frozen=True)
class EvalPaths:
    eval_layout: str
    objective_adjoint: str
    eval_dir: Path
    deform_dir: Path
    direct_dir: Path
    adjoint_dir: Path
    modes_current: Path
    surface_positions: Path
    metadata: Path
    def_cfg: Path
    primal_cfg: Path
    adjoint_cfg: Path
    deformed_mesh: Path
    primal_mesh_out: Path
    adjoint_mesh_out: Path
    primal_history: Path
    adjoint_history: Path
    primal_solution: Path
    primal_restart: Path
    adjoint_solution: Path
    adjoint_restart: Path
    volume_adjoint: Path
    surface_adjoint: Path
    gradients: Path
    summary: Path
    commands_log: Path
    bspline_def_log: Path
    su2_def_log: Path
    su2_cfd_log: Path
    su2_cfd_ad_log: Path
    bspline_dot_log: Path

def _normalize_eval_layout(eval_layout):
    if eval_layout is None:
        return "DSN"
    layout = str(eval_layout).strip().upper()
    if layout == "FLAT":
        raise BSplineSU2DriverError(
            "BSPLINE_EVAL_LAYOUT=FLAT has been removed. DSN is the only supported layout."
        )
    if layout not in ALLOWED_EVAL_LAYOUTS:
        raise BSplineSU2DriverError(
            f"BSPLINE_EVAL_LAYOUT must be one of {ALLOWED_EVAL_LAYOUTS}; got {layout!r}"
        )
    return layout

def _normalize_objective_adjoint(objective_adjoint):
    value = str(objective_adjoint or "drag").strip()
    normalized = _normalized_name(value).lower()
    if not normalized:
        raise BSplineSU2DriverError("objective adjoint name must be non-empty")
    return normalized

def _relative_path(target, start_dir):
    return os.path.relpath(Path(target), Path(start_dir))

def build_eval_paths(eval_dir, eval_layout="DSN", objective_adjoint="drag"):
    eval_dir = Path(eval_dir)
    eval_layout = _normalize_eval_layout(eval_layout)
    objective_adjoint = _normalize_objective_adjoint(objective_adjoint)
    deform_dir = eval_dir / "deform"
    direct_dir = eval_dir / "direct"
    adjoint_dir = eval_dir / f"adjoint_{objective_adjoint}"
    return EvalPaths(
        eval_layout=eval_layout,
        objective_adjoint=objective_adjoint,
        eval_dir=eval_dir,
        deform_dir=deform_dir,
        direct_dir=direct_dir,
        adjoint_dir=adjoint_dir,
        modes_current=eval_dir / "modes_current.json",
        surface_positions=deform_dir / "surface_positions.dat",
        metadata=deform_dir / "bspline_surface_metadata.csv",
        def_cfg=deform_dir / "def.cfg",
        primal_cfg=direct_dir / "primal.cfg",
        adjoint_cfg=adjoint_dir / "adjoint.cfg",
        deformed_mesh=deform_dir / "deformed_mesh.su2",
        primal_mesh_out=direct_dir / "primal_mesh_out.su2",
        adjoint_mesh_out=adjoint_dir / "adjoint_mesh_out.su2",
        primal_history=direct_dir / "history_primal.csv",
        adjoint_history=adjoint_dir / "history_adjoint.csv",
        primal_solution=direct_dir / "solution_flow.dat",
        primal_restart=direct_dir / "restart_flow.dat",
        adjoint_solution=adjoint_dir / "solution_adj.dat",
        adjoint_restart=adjoint_dir / "solution_adj.dat",
        volume_adjoint=adjoint_dir / "volume_adjoint",
        surface_adjoint=adjoint_dir / "surface_adjoint.csv",
        gradients=eval_dir / "bspline_gradients.csv",
        summary=eval_dir / "summary.json",
        commands_log=eval_dir / "commands.log",
        bspline_def_log=deform_dir / "bspline_def.log",
        su2_def_log=deform_dir / "su2_def.log",
        su2_cfd_log=direct_dir / "su2_cfd.log",
        su2_cfd_ad_log=adjoint_dir / "su2_cfd_ad.log",
        bspline_dot_log=adjoint_dir / "bspline_dot.log",
    )

def _command_list(command):
    if isinstance(command, str):
        return shlex.split(command)
    return [str(part) for part in command]

def _with_mpi(command, mpi_prefix=None):
    command = _command_list(command)
    if mpi_prefix:
        return shlex.split(str(mpi_prefix)) + command
    return command

def build_eval_commands(
    paths,
    base_mesh,
    marker,
    mpi_prefix=None,
    python_executable=None,
    sensitivity_weighting="NODAL",
    surface_mode="BOTH",
    deformation_direction_mode=None,
    le_safe_direction=False,
    le_safe_x0=None,
    le_safe_x1=None,
    le_safe_power=None,
):
    """Build subprocess commands for one evaluation directory."""

    python_executable = python_executable or sys.executable or "python3"
    try:
        sensitivity_weighting = normalize_sensitivity_weighting(sensitivity_weighting)
    except BSplineDotError as exc:
        raise BSplineSU2DriverError(str(exc))
    try:
        surface_mode = normalize_surface_mode(surface_mode)
        direction_mode = normalize_deformation_direction_mode(
            deformation_direction_mode,
            le_safe_direction=le_safe_direction,
        )
    except BSplineModeError as exc:
        raise BSplineSU2DriverError(str(exc))
    base_mesh = str(base_mesh)

    bspline_def = [
        python_executable,
        "-m",
        "SU2.opt.bspline_def",
        "--mesh",
        base_mesh,
        "--modes",
        _relative_path(paths.modes_current, paths.deform_dir),
        "--output",
        paths.surface_positions.name,
        "--metadata",
        paths.metadata.name,
        "--surface-mode",
        surface_mode,
        "--deformation-direction",
        direction_mode,
    ]
    if marker:
        bspline_def.extend(["--marker", str(marker)])
    if direction_mode == "LE_SAFE":
        bspline_def.append("--le-safe-direction")
        if le_safe_x0 is not None:
            bspline_def.extend(["--le-safe-x0", str(float(le_safe_x0))])
        if le_safe_x1 is not None:
            bspline_def.extend(["--le-safe-x1", str(float(le_safe_x1))])
        if le_safe_power is not None:
            bspline_def.extend(["--le-safe-power", str(float(le_safe_power))])

    return {
        "bspline_def": bspline_def,
        "def": _with_mpi(["SU2_DEF", paths.def_cfg.name], mpi_prefix),
        "primal": _with_mpi(["SU2_CFD", paths.primal_cfg.name], mpi_prefix),
        "adjoint": _with_mpi(["SU2_CFD_AD", paths.adjoint_cfg.name], mpi_prefix),
        "bspline_dot": [
            python_executable,
            "-m",
            "SU2.opt.bspline_dot",
            "--modes",
            paths.modes_current.name,
            "--metadata",
            _relative_path(paths.metadata, paths.eval_dir),
            "--sens",
            _relative_path(paths.surface_adjoint, paths.eval_dir),
            "--output",
            paths.gradients.name,
            "--summary",
            paths.summary.name,
            "--prefer-vector",
            "--sensitivity-weighting",
            sensitivity_weighting,
        ],
    }

def command_to_string(command):
    if isinstance(command, str):
        return command
    return " ".join(shlex.quote(str(part)) for part in command)

def _subprocess_env():
    env = os.environ.copy()
    root = str(Path(__file__).resolve().parents[3])
    pythonpath = env.get("PYTHONPATH", "")
    paths = [path for path in pythonpath.split(os.pathsep) if path]
    if root not in paths:
        paths.insert(0, root)
        env["PYTHONPATH"] = os.pathsep.join(paths)
    return env

def _tail_file(filename, max_lines=20, include_command=True):
    try:
        with open(filename, "r", errors="replace") as fp:
            lines = fp.readlines()
    except OSError:
        return ""
    if not include_command:
        lines = [line for line in lines if not line.startswith("$ ")]
    return "".join(lines[-max_lines:])

def run_command(command, cwd, log_file, show_command=False, stream_output=False, stage=None):
    """Run a command in cwd, writing stdout/stderr to log_file."""

    cwd = Path(cwd)
    log_file = Path(log_file)
    cwd.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    command_text = command_to_string(command)
    if show_command:
        print(f"[BSPLINE_SU2_DRIVER] {command_text}")
    with open(log_file, "w") as log:
        log.write("$ " + command_text + "\n")
        log.flush()
        if stream_output:
            process = subprocess.Popen(
                command if isinstance(command, str) else _command_list(command),
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=isinstance(command, str),
                env=_subprocess_env(),
                text=True,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                sys.stdout.write(line)
            return_code = process.wait()
        else:
            completed = subprocess.run(
                command if isinstance(command, str) else _command_list(command),
                cwd=str(cwd),
                stdout=log,
                stderr=subprocess.STDOUT,
                shell=isinstance(command, str),
                env=_subprocess_env(),
                text=True,
            )
            return_code = completed.returncode

    if return_code != 0:
        tail = _tail_file(log_file, max_lines=20, include_command=show_command).rstrip()
        message = [
            "failed stage: {}".format(stage or log_file.stem),
            "exit code: {}".format(return_code),
            "eval directory: {}".format(cwd),
            "log file: {}".format(log_file),
        ]
        if show_command:
            message.append("command: {}".format(command_text))
        message.append("last 20 log lines:")
        message.append(tail if tail else "(log file is empty)")
        raise BSplineSU2DriverError("\n".join(message))

def _append_command_log(log_file, stage, cwd, command):
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a") as fp:
        fp.write("[{}] cwd={}\n".format(stage or "command", Path(cwd)))
        fp.write("$ {}\n\n".format(command_to_string(command)))

def _symlink_or_copy(src, dst):
    src = Path(src)
    dst = Path(dst)
    if not src.exists() or src.resolve() == dst.resolve():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(os.path.relpath(src, dst.parent), dst)
    except OSError:
        shutil.copy2(src, dst)

def create_eval_aliases(paths):
    aliases = [
        (paths.metadata, paths.eval_dir / "bspline_surface_metadata.csv"),
        (paths.surface_adjoint, paths.eval_dir / "surface_adjoint.csv"),
        (paths.primal_history, paths.eval_dir / "history_primal.csv"),
        (paths.adjoint_history, paths.eval_dir / "history_adjoint.csv"),
    ]
    for src, dst in aliases:
        _symlink_or_copy(src, dst)

def ensure_adjoint_solution_input(primal_restart, primal_solution):
    primal_restart = Path(primal_restart)
    primal_solution = Path(primal_solution)
    if primal_solution.exists():
        return primal_solution
    if primal_restart.exists():
        shutil.copy2(primal_restart, primal_solution)
        return primal_solution
    raise BSplineSU2DriverError(
        "neither primal solution nor restart file exists: {}, {}".format(
            primal_solution,
            primal_restart,
        )
    )
