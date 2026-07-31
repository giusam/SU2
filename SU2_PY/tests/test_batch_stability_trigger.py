"""Unit tests for the BATCH_STABILITY trigger in progressive_trigger.py."""

import pytest
from SU2.opt.progressive_trigger import (
    RefinementTriggered,
    build_online_trigger_opts,
    record_gradient_and_check,
    _batch_key,
)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


class _FakeProject:
    def __init__(self):
        self.trigger_opts = None
        self.trigger_history = []
        self.trigger_state = None
        self.refinement_triggered = False
        self.progressive_label = "TEST"
        self.batch_stability_scorer_fn = None
        self.batch_stability_log_path = None


def _scorer_fn(batch):
    """Return a callable that always returns the given batch."""
    def _fn():
        return {"selected_candidates": batch}
    return _fn


def _candidate(side, x):
    return {"side": side, "x": x}


def _run_gradient_step(project, obj_value):
    """Simulate one adjoint evaluation — mirrors what scipy_tools.obj_df does."""
    project.trigger_history.append(obj_value)
    record_gradient_and_check(project, obj_value)


# ---------------------------------------------------------------------------
#  _batch_key
# ---------------------------------------------------------------------------


def test_batch_key_empty():
    assert _batch_key({"selected_candidates": []}) == tuple()


def test_batch_key_ordered_tuple():
    c = [_candidate("UPPER", 0.25), _candidate("LOWER", 0.5)]
    assert _batch_key({"selected_candidates": c}) == (
        ("UPPER", 0.25),
        ("LOWER", 0.5),
    )


def test_batch_key_order_sensitive():
    c1 = [_candidate("UPPER", 0.25), _candidate("LOWER", 0.5)]
    c2 = [_candidate("LOWER", 0.5), _candidate("UPPER", 0.25)]
    assert _batch_key({"selected_candidates": c1}) != _batch_key(
        {"selected_candidates": c2}
    )


def test_batch_key_rounds_to_12_decimals():
    a = _batch_key({"selected_candidates": [_candidate("UPPER", 0.1 + 1e-14)]})
    b = _batch_key({"selected_candidates": [_candidate("UPPER", 0.1)]})
    assert a == b


def test_batch_key_missing_selected_candidates_raises():
    with pytest.raises(ValueError, match="contract violation"):
        _batch_key({"candidates": [_candidate("UPPER", 0.25)]})


# ---------------------------------------------------------------------------
#  build_online_trigger_opts
# ---------------------------------------------------------------------------


def test_build_opts_batch_stability():
    opts = build_online_trigger_opts(
        "BATCH_STABILITY",
        current_level=0,
        final_ndv=10,
        current_ndv=4,
    )
    assert opts is not None
    assert opts["trigger"] == "BATCH_STABILITY"
    assert opts["k"] == 3


def test_build_opts_final_level_returns_none():
    opts = build_online_trigger_opts(
        "BATCH_STABILITY",
        current_level=0,
        final_ndv=4,
        current_ndv=4,
    )
    assert opts is None


# ---------------------------------------------------------------------------
#  No scorer_fn attached → trigger stays silent
# ---------------------------------------------------------------------------


def test_no_scorer_fn_never_fires():
    project = _FakeProject()
    project.trigger_opts = {"trigger": "BATCH_STABILITY", "k": 3}
    project.batch_stability_scorer_fn = None

    for obj in [0.05, 0.04, 0.03, 0.02, 0.01]:
        _run_gradient_step(project, obj)

    assert not project.refinement_triggered


# ---------------------------------------------------------------------------
#  No first batch change → never arms, never fires
# ---------------------------------------------------------------------------


def test_stable_batch_never_fires():
    batch = [_candidate("UPPER", 0.25), _candidate("LOWER", 0.5)]
    project = _FakeProject()
    project.trigger_opts = {"trigger": "BATCH_STABILITY", "k": 3}
    project.batch_stability_scorer_fn = _scorer_fn(batch)

    for obj in [0.05, 0.04, 0.03, 0.02, 0.01]:
        _run_gradient_step(project, obj)

    assert not project.refinement_triggered


# ---------------------------------------------------------------------------
#  Batch changes once, then becomes stable for k rounds → fires
# ---------------------------------------------------------------------------


def test_fires_after_k_stable_batches():
    batch_a = [_candidate("UPPER", 0.25)]
    batch_b = [_candidate("UPPER", 0.35)]

    call_count = [0]
    objectives = [0.05, 0.04, 0.03, 0.025, 0.020, 0.015]
    batches = [batch_a, batch_a, batch_b, batch_b, batch_b, batch_b]

    def scorer():
        idx = call_count[0]
        call_count[0] += 1
        return {"selected_candidates": batches[idx]}

    project = _FakeProject()
    project.trigger_opts = {"trigger": "BATCH_STABILITY", "k": 3}
    project.batch_stability_scorer_fn = scorer
    project.last_obj_grad_x = [0.2]
    project.last_obj_grad_x_full = [0.2, -0.2]
    project.last_obj_grad_design_folder = "DESIGNS/DSN_005"

    fired = False
    for obj in objectives:
        try:
            _run_gradient_step(project, obj)
        except RefinementTriggered:
            fired = True
            break

    assert fired
    assert project.refinement_triggered
    # Sequence: [A,A,B,B,B,B], armed on step 3 (first B), last 3=[B,B,B] at step 5
    assert call_count[0] == 5
    # The firing batch is stored for refine-time consistency verification
    assert project.trigger_state["trigger_batch"] == (("UPPER", 0.35),)
    assert project.trigger_state["trigger_dv_values"] == [0.2, -0.2]
    assert project.trigger_state["trigger_design_folder"] == "DESIGNS/DSN_005"


# ---------------------------------------------------------------------------
#  Order changes count as batch changes (sequential scorer)
# ---------------------------------------------------------------------------


def test_order_change_arms_trigger():
    batch_ab = [_candidate("UPPER", 0.25), _candidate("LOWER", 0.5)]
    batch_ba = [_candidate("LOWER", 0.5), _candidate("UPPER", 0.25)]

    call_count = [0]
    batches = [batch_ab, batch_ba, batch_ba, batch_ba]

    def scorer():
        idx = call_count[0]
        call_count[0] += 1
        return {"selected_candidates": batches[idx]}

    project = _FakeProject()
    project.trigger_opts = {"trigger": "BATCH_STABILITY", "k": 3}
    project.batch_stability_scorer_fn = scorer

    fired = False
    for obj in [0.05, 0.04, 0.03, 0.02]:
        try:
            _run_gradient_step(project, obj)
        except RefinementTriggered:
            fired = True
            break

    # Armed at step 2 (order flip); [BA,BA,BA] at step 4 → fires
    assert fired


# ---------------------------------------------------------------------------
#  Non-new-best observations are skipped
# ---------------------------------------------------------------------------


def test_non_new_best_skipped():
    batch_a = [_candidate("UPPER", 0.25)]
    batch_b = [_candidate("UPPER", 0.35)]

    call_count = [0]
    batches_on_new_best = [batch_a, batch_b, batch_b, batch_b]

    def scorer():
        idx = call_count[0]
        call_count[0] += 1
        return {"selected_candidates": batches_on_new_best[idx]}

    project = _FakeProject()
    project.trigger_opts = {"trigger": "BATCH_STABILITY", "k": 3}
    project.batch_stability_scorer_fn = scorer

    # Decreasing sequence (all new bests) plus one non-improving spike
    objectives_with_spike = [0.05, 0.04, 0.09, 0.03, 0.025, 0.020, 0.015]

    fired = False
    for obj in objectives_with_spike:
        try:
            _run_gradient_step(project, obj)
        except RefinementTriggered:
            fired = True
            break

    assert fired
    # Scorer should only have been called for new bests
    assert call_count[0] == 4  # 0.05, 0.04, 0.03, 0.025 are new bests; 0.020 fires


# ---------------------------------------------------------------------------
#  Transactional state: failed scorer does NOT consume the best
# ---------------------------------------------------------------------------


def test_failed_scorer_preserves_best_for_retry():
    batch = [_candidate("UPPER", 0.25)]
    calls = [0]

    def flaky_scorer():
        calls[0] += 1
        if calls[0] == 2:
            raise RuntimeError("transient failure")
        return {"selected_candidates": batch}

    project = _FakeProject()
    project.trigger_opts = {"trigger": "BATCH_STABILITY", "k": 3}
    project.batch_stability_scorer_fn = flaky_scorer

    _run_gradient_step(project, 0.05)   # call 1 OK, best=0.05
    _run_gradient_step(project, 0.04)   # call 2 raises → best stays 0.05
    assert project.trigger_state["best_obj"] == 0.05

    # A later 0.045 IS a new best w.r.t. the last successfully scored best
    _run_gradient_step(project, 0.045)  # call 3 OK, best=0.045
    assert project.trigger_state["best_obj"] == 0.045
    assert calls[0] == 3


# ---------------------------------------------------------------------------
#  Empty batch is skipped and does not update state
# ---------------------------------------------------------------------------


def test_empty_batch_skipped():
    calls = [0]

    def empty_scorer():
        calls[0] += 1
        return {"selected_candidates": []}

    project = _FakeProject()
    project.trigger_opts = {"trigger": "BATCH_STABILITY", "k": 3}
    project.batch_stability_scorer_fn = empty_scorer

    for obj in [0.05, 0.04, 0.03, 0.02, 0.01]:
        _run_gradient_step(project, obj)

    assert not project.refinement_triggered
    # best_obj never committed → every design is a new best → scorer called 5x
    assert calls[0] == 5
    assert project.trigger_state["batch_history"] == []


# ---------------------------------------------------------------------------
#  Contract violation (no selected_candidates) fails fast
# ---------------------------------------------------------------------------


def test_contract_violation_fails_fast():
    def bad_scorer():
        return {"candidates": [_candidate("UPPER", 0.25)]}

    project = _FakeProject()
    project.trigger_opts = {"trigger": "BATCH_STABILITY", "k": 3}
    project.batch_stability_scorer_fn = bad_scorer

    with pytest.raises(ValueError, match="contract violation"):
        _run_gradient_step(project, 0.05)


# ---------------------------------------------------------------------------
#  CSV log is written when a path is set
# ---------------------------------------------------------------------------


def test_csv_log_written(tmp_path):
    batch = [_candidate("UPPER", 0.25)]
    project = _FakeProject()
    project.trigger_opts = {"trigger": "BATCH_STABILITY", "k": 3}
    project.batch_stability_scorer_fn = _scorer_fn(batch)
    log_path = tmp_path / "batch_stability_log.csv"
    project.batch_stability_log_path = str(log_path)

    for obj in [0.05, 0.04]:
        _run_gradient_step(project, obj)

    content = log_path.read_text().strip().splitlines()
    assert content[0] == "call_idx,obj_value,armed,history_len,batch"
    assert len(content) == 3  # header + 2 samples


# ---------------------------------------------------------------------------
#  Other triggers are unaffected by record_gradient_and_check
# ---------------------------------------------------------------------------


def test_economic_trigger_not_called_from_gradient_hook():
    project = _FakeProject()
    project.trigger_opts = {"trigger": "ECONOMIC_TRIGGER", "k": 3}

    for obj in [0.05, 0.04, 0.03, 0.02]:
        record_gradient_and_check(project, obj)

    # ECONOMIC_TRIGGER is not wired in record_gradient_and_check → no state change
    assert project.trigger_state is None
    assert not project.refinement_triggered
