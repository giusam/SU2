from pathlib import Path
import pickle
from types import SimpleNamespace

import pytest

from SU2.opt import scipy_tools
from SU2.opt.project import Project
from SU2.opt.progressive_design import (
    find_design_mesh,
    write_ranking_design_manifest,
)
from SU2.opt.progressive_hh_levels import collect_level_result
from SU2.opt.progressive_hh_projection import _find_real_adjoint_assets


class _FakeState:
    def __init__(self, values):
        self._values = list(values)

    def design_vector(self):
        return list(self._values)


def _fake_design(folder, values):
    return SimpleNamespace(folder=folder, state=_FakeState(values))


def _write_design(level_dir, number, values, converged, adjoint=False):
    relative = f"DESIGNS/DSN_{number:03d}"
    design_dir = level_dir / relative
    direct = design_dir / "DIRECT"
    direct.mkdir(parents=True)
    (design_dir / "mesh_deform.su2").write_text("mesh\n")
    verdict = (
        "All convergence criteria satisfied.\n"
        if converged
        else "Maximum number of iterations reached (ITER = 99) before convergence.\n"
    )
    (direct / "log_Direct.out").write_text(verdict)
    if adjoint:
        adjoint_dir = design_dir / "ADJOINT_DRAG"
        adjoint_dir.mkdir()
        (adjoint_dir / "config_DOT.cfg").write_text("MATH_PROBLEM= CONTINUOUS_ADJOINT\n")
    return _fake_design(relative, values), design_dir


def _project(designs, accepted_history, refinement_values=None):
    return SimpleNamespace(
        config={"OBJECTIVE_FUNCTION": "DRAG"},
        designs=list(designs),
        accepted_dv_history=[list(values) for values in accepted_history],
        refinement_dv_values=refinement_values,
        opt_dv_values=(list(accepted_history[-1]) if accepted_history else None),
    )


def test_result_ignores_rejected_low_drag_dsn_and_rolls_back_nonconverged_accept(
    tmp_path,
):
    level_dir = tmp_path / "LEVEL_0"
    accepted_old, accepted_old_dir = _write_design(
        level_dir, 1, [0.0], converged=True
    )
    rejected_trial, _ = _write_design(level_dir, 2, [9.0], converged=True)
    accepted_bad, accepted_bad_dir = _write_design(
        level_dir, 3, [0.5], converged=False
    )

    # DSN_002 represents an objective-only line-search trial.  Its drag could
    # be arbitrarily low; absence from accepted_dv_history is authoritative.
    project = _project(
        [accepted_old, rejected_trial, accepted_bad],
        accepted_history=[[0.0], [0.5]],
    )
    level = SimpleNamespace(workdir=str(level_dir))

    result = collect_level_result(level, project=project)

    assert Path(result["accepted_design_dir"]) == accepted_old_dir
    assert Path(result["final_mesh"]).parent == accepted_old_dir
    assert result["dv_values"] == [0.0]
    assert len(result["rejected_accepted_designs"]) == 1
    assert Path(result["rejected_accepted_designs"][0]["design_dir"]) == accepted_bad_dir


def test_design_mesh_distinguishes_multilevel_hh_input_from_generated_mesh(
    tmp_path,
):
    level_dir = tmp_path / "LEVEL_1"
    level_dir.mkdir()
    input_mesh = level_dir / "naca0012_deform.su2"
    input_mesh.write_text("level input\n")

    design_dir = level_dir / "DESIGNS" / "DSN_002"
    design_dir.mkdir(parents=True)
    (design_dir / "config_DSN.cfg").write_text(
        "MESH_FILENAME= naca0012_deform.su2\n"
    )
    (design_dir / input_mesh.name).symlink_to(input_mesh)
    generated = design_dir / "naca0012_deform_deform.su2"
    generated.write_text("design output\n")

    assert Path(find_design_mesh(str(design_dir))) == generated


def test_design_mesh_uses_linked_input_for_zero_dv_multilevel_hh_design(tmp_path):
    level_dir = tmp_path / "LEVEL_1"
    level_dir.mkdir()
    input_mesh = level_dir / "naca0012_deform.su2"
    input_mesh.write_text("level input\n")

    design_dir = level_dir / "DESIGNS" / "DSN_001"
    design_dir.mkdir(parents=True)
    (design_dir / "config_DSN.cfg").write_text(
        "MESH_FILENAME= naca0012_deform.su2\n"
    )
    linked_input = design_dir / input_mesh.name
    linked_input.symlink_to(input_mesh)

    assert Path(find_design_mesh(str(design_dir))) == linked_input


def test_batch_pinned_nonconverged_design_fails_instead_of_changing_batch_baseline(
    tmp_path,
):
    level_dir = tmp_path / "LEVEL_0"
    accepted_old, _ = _write_design(level_dir, 1, [0.0], converged=True)
    trigger_bad, _ = _write_design(level_dir, 2, [0.5], converged=False)
    project = _project(
        [accepted_old, trigger_bad],
        accepted_history=[[0.0], [0.5]],
        refinement_values=[0.5],
    )

    with pytest.raises(RuntimeError, match="No accepted SLSQP design"):
        collect_level_result(SimpleNamespace(workdir=str(level_dir)), project=project)


def test_required_adjoint_must_belong_to_selected_accepted_design(tmp_path):
    level_dir = tmp_path / "LEVEL_0"
    accepted, accepted_dir = _write_design(level_dir, 1, [0.0], converged=True)
    other, _ = _write_design(level_dir, 99, [9.0], converged=True, adjoint=True)
    project = _project([accepted, other], accepted_history=[[0.0]])
    level = SimpleNamespace(workdir=str(level_dir))

    with pytest.raises(RuntimeError, match="lacks the objective adjoint"):
        collect_level_result(level, project=project, require_adjoint=True)

    accepted_adjoint = accepted_dir / "ADJOINT_DRAG"
    accepted_adjoint.mkdir()
    (accepted_adjoint / "config_DOT.cfg").write_text(
        "MATH_PROBLEM= CONTINUOUS_ADJOINT\n"
    )
    result = collect_level_result(level, project=project, require_adjoint=True)
    write_ranking_design_manifest(
        str(level_dir),
        result["accepted_design_dir"],
        result["final_mesh"],
        result["dv_values"],
        result["ranking_objective"],
    )

    adjoint_dir, design_dir = _find_real_adjoint_assets(str(level_dir), "DRAG")
    assert Path(design_dir) == accepted_dir
    assert Path(adjoint_dir) == accepted_adjoint


def test_ranking_anchor_never_borrows_missing_adjoint_from_later_dsn(tmp_path):
    level_dir = tmp_path / "LEVEL_0"
    accepted, accepted_dir = _write_design(level_dir, 1, [0.0], converged=True)
    _, _ = _write_design(level_dir, 99, [9.0], converged=True, adjoint=True)
    mesh = accepted_dir / "mesh_deform.su2"
    write_ranking_design_manifest(
        str(level_dir),
        str(accepted_dir),
        str(mesh),
        [0.0],
        "DRAG",
    )

    with pytest.raises(FileNotFoundError, match="refusing to borrow"):
        _find_real_adjoint_assets(str(level_dir), "DRAG")


def test_slsqp_early_trigger_uses_last_accepted_not_last_objective_trial(monkeypatch):
    class FakeProject:
        config = {
            "DEFINITION_DV": {"SIZE": [1], "SCALE": [1.0]},
            "OPT_OBJECTIVE": {"DRAG": {"SCALE": 1.0}},
            "GRADIENT_METHOD": "CONTINUOUS_ADJOINT",
        }
        progressive_hh_symmetry = None
        trigger_opts = {"trigger": "STAGNATION_TRIGGER"}

    project = FakeProject()

    def fake_fmin_slsqp(*, args, callback, **_kwargs):
        active_project = args[0]
        callback([2.0])  # accepted major iterate
        active_project.last_dv_values = [9.0]  # rejected objective-only trial
        raise scipy_tools.RefinementTriggered()

    monkeypatch.setattr("scipy.optimize.fmin_slsqp", fake_fmin_slsqp)

    result = scipy_tools.scipy_slsqp(
        project,
        x0=[0.0],
        xb=[(-10.0, 10.0)],
        its=5,
        accu=1.0e-6,
    )

    assert result is None
    assert project.accepted_dv_history == [[0.0], [2.0]]
    assert project.last_dv_values == [9.0]
    assert project.opt_dv_values == [2.0]


def test_runtime_batch_scorer_is_excluded_from_project_pickle():
    project = Project.__new__(Project)
    project.marker = "kept"
    project.batch_stability_scorer_fn = lambda: None
    project.trajectory_ready_scorer_fn = lambda: None

    restored = pickle.loads(pickle.dumps(project))

    assert restored.marker == "kept"
    assert restored.batch_stability_scorer_fn is None
    assert restored.trajectory_ready_scorer_fn is None
