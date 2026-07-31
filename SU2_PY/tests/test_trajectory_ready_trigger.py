"""Unit tests for the post-adjoint TRAJECTORY_READY trigger."""

import csv

import pytest

from SU2.opt.progressive_hh_core import get_progressive_hh_options
from SU2.opt.progressive_trigger import (
    RefinementTriggered,
    build_online_trigger_opts,
    record_gradient_and_check,
)


OBJECTIVES_THAT_SATURATE_ON_LAST = [1.0, 0.8, 0.7, 0.69, 0.689, 0.6889]


class _FakeProject:
    def __init__(self):
        self.trigger_opts = {"trigger": "TRAJECTORY_READY"}
        self.trigger_history = []
        self.trigger_state = None
        self.refinement_triggered = False
        self.progressive_label = "TEST"
        self.trajectory_ready_scorer_fn = None
        self.trajectory_ready_log_path = None
        self.last_obj_grad_x = [0.25]
        self.last_obj_grad_x_full = [0.25, -0.25]
        self.last_obj_grad_design_folder = "DESIGNS/DSN_001"


def _candidate(side, x, score, rank, *, energy_current=None, energy_candidate=None):
    candidate = {
        "insertion_step": 1,
        "side": side,
        "x": x,
        "indicator": score,
        "score_net": score,
        "rank": rank,
        "admissible": True,
    }
    if energy_current is not None:
        candidate["energy_current"] = energy_current
    if energy_candidate is not None:
        candidate["energy_candidate"] = energy_candidate
    return candidate


def _frontier(
    selected_side="UPPER",
    selected_x=0.25,
    selected_score=10.0,
    runner_side="UPPER",
    runner_x=0.375,
    runner_score=5.0,
    enrichment_gain=0.5,
):
    selected = _candidate(
        selected_side,
        selected_x,
        selected_score,
        1,
        energy_current=10.0,
        energy_candidate=10.0 * (1.0 + enrichment_gain),
    )
    runner = _candidate(runner_side, runner_x, runner_score, 2)
    return {
        "selected_candidates": [selected],
        "raw_candidates": [selected, runner],
    }


def _run_step(project, obj_value):
    project.trigger_history.append(obj_value)
    record_gradient_and_check(project, obj_value)


def test_parser_and_online_opts_accept_trajectory_ready():
    config = {
        "PROGRESSIVE_HH_TRIGGER": "TRAJECTORY_READY",
        "PROGRESSIVE_HH_REFINEMENT": "ADAPTIVE",
        "PROGRESSIVE_HH_SCORING_MODE": "VIRTUAL_TANGENT",
        "OPT_BOUND_LOWER": -1.0,
        "OPT_BOUND_UPPER": 1.0,
    }
    parsed = get_progressive_hh_options(config)
    assert parsed["trigger"] == "TRAJECTORY_READY"

    opts = build_online_trigger_opts(
        "TRAJECTORY_READY",
        current_level=0,
        current_ndv=4,
        final_ndv=10,
    )
    assert opts == {"trigger": "TRAJECTORY_READY"}


def test_trajectory_ready_requires_virtual_tangent_scoring():
    with pytest.raises(ValueError, match="VIRTUAL_TANGENT"):
        get_progressive_hh_options(
            {
                "PROGRESSIVE_HH_TRIGGER": "TRAJECTORY_READY",
                "PROGRESSIVE_HH_REFINEMENT": "ADAPTIVE",
                "PROGRESSIVE_HH_SCORING_MODE": "COMPONENT",
            }
        )


def test_stable_useful_frontier_fires_only_after_objective_saturates():
    project = _FakeProject()
    project.trajectory_ready_scorer_fn = lambda: _frontier()

    for obj_value in OBJECTIVES_THAT_SATURATE_ON_LAST[:-1]:
        _run_step(project, obj_value)
        assert not project.refinement_triggered

    with pytest.raises(RefinementTriggered):
        _run_step(project, OBJECTIVES_THAT_SATURATE_ON_LAST[-1])

    state = project.trigger_state
    assert project.refinement_triggered
    assert state["trajectory_call_idx"] == 6
    assert state["trajectory_rate_count"] == 5
    assert state["trajectory_ready"] is True
    assert state["trajectory_trigger_batch"] == (("UPPER", 0.25),)
    assert state["trigger_dv_values"] == [0.25, -0.25]
    assert state["trigger_reduced_dv_values"] == [0.25]
    assert state["trigger_design_folder"] == "DESIGNS/DSN_001"


def test_small_enrichment_blocks_trigger_after_saturation():
    project = _FakeProject()
    project.trajectory_ready_scorer_fn = lambda: _frontier(
        enrichment_gain=0.1
    )

    for obj_value in OBJECTIVES_THAT_SATURATE_ON_LAST:
        _run_step(project, obj_value)

    assert not project.refinement_triggered
    assert project.trigger_state["trajectory_ready"] is False


def test_ambiguous_frontier_blocks_trigger_after_saturation():
    project = _FakeProject()
    project.trajectory_ready_scorer_fn = lambda: _frontier(
        selected_score=10.0,
        runner_score=9.9,
    )

    for obj_value in OBJECTIVES_THAT_SATURATE_ON_LAST:
        _run_step(project, obj_value)

    assert not project.refinement_triggered
    assert project.trigger_state["trajectory_ready"] is False


def test_resolved_crossover_can_fire_despite_current_near_tie():
    frontiers = [
        _frontier(runner_score=5.0),
        _frontier(runner_score=6.0),
        _frontier(runner_score=7.0),
        _frontier(runner_score=9.0),
        _frontier(runner_score=9.6),
        _frontier(
            selected_side="UPPER",
            selected_x=0.375,
            selected_score=10.1,
            runner_side="UPPER",
            runner_x=0.25,
            runner_score=10.0,
        ),
    ]
    calls = [0]

    def scorer():
        result = frontiers[calls[0]]
        calls[0] += 1
        return result

    project = _FakeProject()
    project.trajectory_ready_scorer_fn = scorer

    for obj_value in OBJECTIVES_THAT_SATURATE_ON_LAST[:-1]:
        _run_step(project, obj_value)

    with pytest.raises(RefinementTriggered):
        _run_step(project, OBJECTIVES_THAT_SATURATE_ON_LAST[-1])

    state = project.trigger_state
    assert state["trajectory_frontier_history"][-1]["min_margin"] < 0.02
    assert state["trajectory_trigger_batch"] == (("UPPER", 0.375),)
    assert state["trajectory_ready"] is True


def test_failed_scorer_does_not_consume_best():
    calls = [0]

    def scorer():
        calls[0] += 1
        if calls[0] == 2:
            raise RuntimeError("transient")
        return _frontier()

    project = _FakeProject()
    project.trajectory_ready_scorer_fn = scorer

    _run_step(project, 1.0)
    _run_step(project, 0.8)
    assert project.trigger_state["trajectory_best_obj"] == 1.0

    # Still a new best relative to the last successfully scored design.
    _run_step(project, 0.9)
    assert project.trigger_state["trajectory_best_obj"] == 0.9
    assert calls[0] == 3


def test_missing_energy_contract_fails_before_state_commit():
    selected = _candidate("UPPER", 0.25, 10.0, 1)
    runner = _candidate("UPPER", 0.375, 5.0, 2)
    project = _FakeProject()
    project.trajectory_ready_scorer_fn = lambda: {
        "selected_candidates": [selected],
        "raw_candidates": [selected, runner],
    }

    with pytest.raises(ValueError, match="energy_current/energy_candidate"):
        _run_step(project, 1.0)

    assert project.trigger_state["trajectory_best_obj"] is None
    assert project.trigger_state["trajectory_call_idx"] == 0


def test_csv_records_all_three_gates(tmp_path):
    project = _FakeProject()
    project.trajectory_ready_scorer_fn = lambda: _frontier()
    project.trajectory_ready_log_path = str(
        tmp_path / "trajectory_ready_log.csv"
    )

    _run_step(project, 1.0)
    _run_step(project, 0.8)

    with open(project.trajectory_ready_log_path, newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == 2
    assert rows[-1]["saturated"] == "0"
    assert rows[-1]["enrichment_useful"] == "1"
    assert rows[-1]["unambiguous"] == "1"
    assert rows[-1]["frontier_ready"] == "1"
    assert rows[-1]["triggered"] == "0"
    assert rows[-1]["design_folder"] == "DESIGNS/DSN_001"
