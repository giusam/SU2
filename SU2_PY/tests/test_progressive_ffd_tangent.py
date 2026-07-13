import numpy as np
import pytest
from types import SimpleNamespace

from SU2.opt.progressive_ffd_core import get_progressive_ffd_options
from SU2.opt.progressive_ffd_tangent import (
    COMPONENT,
    VIRTUAL_TANGENT,
    _su2_airfoil_max_thickness,
    airfoil_area_value_and_field,
    airfoil_thickness_value_and_field,
    build_ffd_tangent_state,
    compare_tangent_spaces,
    fit_surface_ikkt_signal,
    internal_constraint_field,
    load_design_gradient_csv,
    load_surface_sensitivity_vector,
    project_surface_field,
    validate_surface_projection,
)
from SU2.opt.progressive_hh_core import get_progressive_hh_options
from SU2.opt.progressive_ffd_levels import build_next_ffd_level
from SU2.opt.progressive_ffd_projection import (
    _compute_dual_bezier_exact_candidate_scores,
    _compute_dual_virtual_tangent_candidate_scores_impl,
    _initialize_virtual_tangent_context,
)
from tests.test_progressive_ffd_dual import _dual_config
from tests.test_progressive_ffd_split import _split, _write_bootstrap_mesh


def _virtual_opts(**overrides):
    config = _dual_config(
        PROGRESSIVE_HH_REFINEMENT="ADAPTIVE",
        PROGRESSIVE_HH_NADD_MODE="GROWTH_RATIO",
        PROGRESSIVE_HH_ADAPTIVE_INDICATOR="IKKT",
        PROGRESSIVE_FFD_SCORING_MODE="VIRTUAL_TANGENT",
        OPT_BOUND_LOWER=-1.0,
        OPT_BOUND_UPPER=1.0,
        **overrides,
    )
    return get_progressive_ffd_options(
        config,
        get_progressive_hh_options(config),
    )


def _dual_state(tmp_path, blending="BEZIER"):
    bootstrap = _write_bootstrap_mesh(
        tmp_path / f"bootstrap_{blending.lower()}.su2",
    )
    mesh = tmp_path / f"dual_{blending.lower()}.su2"
    _split(bootstrap, mesh, output_blending=blending)
    opts = _virtual_opts(FFD_BLENDING=blending)
    active = {
        "UPPER": [0.25, 0.5, 0.75],
        "LOWER": [0.25, 0.5, 0.75],
    }
    return build_ffd_tangent_state(mesh, "AIRFOIL", active, opts), opts


def test_virtual_scoring_mode_is_explicit_and_component_remains_default():
    default_cfg = _dual_config()
    default = get_progressive_ffd_options(
        default_cfg,
        get_progressive_hh_options(default_cfg),
    )
    assert default["ffd_scoring_mode"] == COMPONENT
    assert _virtual_opts()["ffd_scoring_mode"] == VIRTUAL_TANGENT


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"PROGRESSIVE_HH_REFINEMENT": "UNIFORM"}, "REFINEMENT=ADAPTIVE"),
        ({"PROGRESSIVE_HH_NADD_MODE": "FIXED"}, "NADD_MODE=GROWTH_RATIO"),
        ({"OPT_BOUND_LOWER": 0.0}, "bilateral DV bounds"),
        ({"OPT_BOUND_UPPER": (1.0, 0.0)}, "bilateral DV bounds"),
        (
            {"PROGRESSIVE_HH_ADAPTIVE_INDICATOR": "DESCENT_GRAD"},
            "ABS_GRAD or IKKT",
        ),
    ],
)
def test_virtual_scoring_configuration_rejects_unsupported_contracts(
    overrides,
    message,
):
    values = {
        "PROGRESSIVE_HH_REFINEMENT": "ADAPTIVE",
        "PROGRESSIVE_HH_NADD_MODE": "GROWTH_RATIO",
        "PROGRESSIVE_HH_ADAPTIVE_INDICATOR": "IKKT",
        "PROGRESSIVE_FFD_SCORING_MODE": "VIRTUAL_TANGENT",
        "OPT_BOUND_LOWER": -1.0,
        "OPT_BOUND_UPPER": 1.0,
    }
    values.update(overrides)
    config = _dual_config(**values)
    with pytest.raises(ValueError, match=message):
        get_progressive_ffd_options(config, get_progressive_hh_options(config))


@pytest.mark.parametrize("blending", ["BEZIER", "BSPLINE_UNIFORM"])
def test_tangent_state_builds_upper_and_lower_cartesian_modes(tmp_path, blending):
    state, _opts = _dual_state(tmp_path, blending=blending)
    matrix = state["matrix"]
    assert matrix.shape == (16, 6)
    assert state["records"] == [
        ("UPPER", 0.25),
        ("UPPER", 0.5),
        ("UPPER", 0.75),
        ("LOWER", 0.25),
        ("LOWER", 0.5),
        ("LOWER", 0.75),
    ]
    assert np.allclose(matrix[0::2, :], 0.0)
    assert np.max(matrix[:, :3]) > 0.0
    assert np.min(matrix[:, 3:]) < 0.0


@pytest.mark.parametrize(
    "side, expected_sign",
    [("UPPER", 1.0), ("LOWER", -1.0)],
)
def test_tangent_state_supports_each_half_domain(tmp_path, side, expected_sign):
    state, opts = _dual_state(tmp_path)
    half = build_ffd_tangent_state(
        state["mesh_path"],
        "AIRFOIL",
        {side: [0.25, 0.5, 0.75]},
        opts,
    )
    assert half["matrix"].shape == (16, 3)
    assert {record[0] for record in half["records"]} == {side}
    nonzero = half["matrix"][np.abs(half["matrix"]) > 1.0e-14]
    assert np.all(expected_sign * nonzero > 0.0)


def test_tangent_state_includes_active_offset_endpoints(tmp_path):
    state, opts = _dual_state(tmp_path)
    columns = state["boxes"]["UPPER"]["columns"]
    endpoints = [columns[0], 0.5, columns[-1]]
    endpoint_state = build_ffd_tangent_state(
        state["mesh_path"],
        "AIRFOIL",
        {"UPPER": endpoints},
        opts,
    )
    assert endpoint_state["records"] == [
        ("UPPER", columns[0]),
        ("UPPER", 0.5),
        ("UPPER", columns[-1]),
    ]
    assert np.linalg.norm(endpoint_state["matrix"][:, 0]) > 0.0
    assert np.linalg.norm(endpoint_state["matrix"][:, -1]) > 0.0


def test_surface_sensitivity_is_matched_by_point_id(tmp_path):
    state, _opts = _dual_state(tmp_path)
    path = tmp_path / "surface_sens.csv"
    rows = [
        f"{node_id},{10.0 + node_id},{20.0 + node_id}\n"
        for node_id in reversed(state["node_ids"])
    ]
    path.write_text(
        "PointID,Sensitivity_x,Sensitivity_y\n" + "".join(rows)
    )
    signal = load_surface_sensitivity_vector(path, state)
    assert signal[0::2] == pytest.approx(
        [10.0 + node_id for node_id in state["node_ids"]]
    )
    assert signal[1::2] == pytest.approx(
        [20.0 + node_id for node_id in state["node_ids"]]
    )


def _space_state(matrix):
    matrix = np.asarray(matrix, dtype=float)
    nnode = matrix.shape[0] // 2
    return {
        "node_ids": list(range(nnode)),
        "matrix": matrix,
        "x_over_c": np.linspace(0.0, 1.0, nnode),
    }


def test_tangent_score_is_invariant_to_basis_scaling_and_change_of_coordinates():
    rng = np.random.default_rng(12)
    baseline = rng.normal(size=(12, 3))
    candidate = np.column_stack((baseline, rng.normal(size=12)))
    signal = rng.normal(size=12)
    reference = compare_tangent_spaces(
        _space_state(baseline),
        _space_state(candidate),
        signal,
        0.4,
    )
    transform0 = np.asarray([[2.0, 1.0, 0.0], [0.0, 0.5, 1.0], [1.0, 0.0, 1.0]])
    transformt = np.asarray(
        [[2.0, 1.0, 0.0, 0.0], [0.0, 0.5, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 2.0]]
    )
    changed = compare_tangent_spaces(
        _space_state(baseline @ transform0),
        _space_state(candidate @ transformt),
        signal,
        0.4,
    )
    assert changed["score_net"] == pytest.approx(reference["score_net"], abs=1e-12)
    assert changed["score_pure"] == pytest.approx(reference["score_pure"], abs=1e-12)
    assert changed["rank_gain"] == reference["rank_gain"] == 1
    assert changed["pure_rank"] == reference["pure_rank"] == 1


def test_tangent_score_can_be_negative_for_a_replacement_space():
    baseline = np.asarray([[1.0], [0.0], [0.0], [0.0]])
    candidate = np.asarray([[0.0], [1.0], [0.0], [0.0]])
    signal = np.asarray([1.0, 0.0, 0.0, 0.0])
    result = compare_tangent_spaces(
        _space_state(baseline),
        _space_state(candidate),
        signal,
        0.5,
    )
    assert result["score_net"] == pytest.approx(-1.0)


def test_nested_tangent_space_reports_pure_rank_one_and_positive_energy_gain():
    baseline = np.asarray([[1.0], [0.0], [0.0], [0.0]])
    candidate = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]
    )
    result = compare_tangent_spaces(
        _space_state(baseline),
        _space_state(candidate),
        np.asarray([0.0, 3.0, 0.0, 0.0]),
        0.5,
    )
    assert result["rank_gain"] == 1
    assert result["pure_rank"] == 1
    assert result["nesting_max"] == pytest.approx(0.0, abs=1e-12)
    assert result["score_net"] == pytest.approx(9.0)
    assert result["score_pure"] == pytest.approx(9.0)


def test_tangent_score_is_zero_when_signal_is_orthogonal_to_both_spaces():
    baseline = np.asarray([[1.0], [0.0], [0.0], [0.0]])
    candidate = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]
    )
    result = compare_tangent_spaces(
        _space_state(baseline),
        _space_state(candidate),
        np.asarray([0.0, 0.0, 1.0, 0.0]),
        0.5,
    )
    assert result["score_net"] == pytest.approx(0.0, abs=1e-14)
    assert result["score_pure"] == pytest.approx(0.0, abs=1e-14)


def test_dependent_candidate_column_is_rank_truncated():
    baseline = np.asarray([[1.0], [0.0], [0.0], [0.0]])
    candidate = np.asarray([[1.0, 2.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    result = compare_tangent_spaces(
        _space_state(baseline),
        _space_state(candidate),
        np.ones(4),
        0.5,
    )
    assert result["rank_gain"] == 0
    assert result["pure_rank"] == 0


def test_tangent_score_is_invariant_to_marker_row_permutation():
    rng = np.random.default_rng(21)
    baseline = rng.normal(size=(10, 3))
    candidate = np.column_stack((baseline, rng.normal(size=10)))
    signal = rng.normal(size=10)
    reference = compare_tangent_spaces(
        _space_state(baseline),
        _space_state(candidate),
        signal,
        0.4,
    )
    node_permutation = np.asarray([3, 1, 4, 0, 2])
    row_permutation = np.ravel(
        np.column_stack((2 * node_permutation, 2 * node_permutation + 1))
    )
    changed = compare_tangent_spaces(
        _space_state(baseline[row_permutation]),
        _space_state(candidate[row_permutation]),
        signal[row_permutation],
        0.4,
    )
    assert changed["score_net"] == pytest.approx(reference["score_net"], abs=1e-12)
    assert changed["score_pure"] == pytest.approx(reference["score_pure"], abs=1e-12)


def test_airfoil_area_surface_field_matches_central_difference(tmp_path):
    state, _opts = _dual_state(tmp_path)
    value, field = airfoil_area_value_and_field(state)
    rng = np.random.default_rng(3)
    displacement = rng.normal(size=field.size)
    epsilon = 1.0e-7

    def perturbed(sign):
        result = dict(state)
        result["coordinates"] = state["coordinates"] + (
            sign * epsilon * displacement.reshape((-1, 2))
        )
        return airfoil_area_value_and_field(result)[0]

    fd_value = (perturbed(1.0) - perturbed(-1.0)) / (2.0 * epsilon)
    assert value > 0.0
    assert fd_value == pytest.approx(float(field @ displacement), rel=1e-7, abs=1e-9)


def test_airfoil_area_supports_open_half_profile_with_implicit_chord_closure():
    state = {
        "node_ids": [0, 1, 2],
        "closed": False,
        "coordinates": np.asarray([[1.0, 0.0], [0.5, 0.1], [0.0, 0.0]]),
    }
    value, field = airfoil_area_value_and_field(state)
    assert value == pytest.approx(0.05)
    assert field.shape == (6,)


def test_airfoil_thickness_surface_field_matches_vertical_difference(tmp_path):
    state, _opts = _dual_state(tmp_path)
    value, field, diagnostics = airfoil_thickness_value_and_field(state)
    rng = np.random.default_rng(7)
    displacement = np.zeros(field.size)
    displacement[1::2] = rng.normal(size=len(state["node_ids"]))
    epsilon = 1.0e-7

    def perturbed(sign):
        result = dict(state)
        result["coordinates"] = state["coordinates"] + (
            sign * epsilon * displacement.reshape((-1, 2))
        )
        return airfoil_thickness_value_and_field(result)[0]

    fd_value = (perturbed(1.0) - perturbed(-1.0)) / (2.0 * epsilon)
    assert value > 0.0
    assert 0.0 <= diagnostics["x_over_c"] <= 1.0
    assert fd_value == pytest.approx(float(field @ displacement), rel=2e-5, abs=1e-8)


def test_su2_thickness_tie_uses_first_evaluation_station():
    coordinates = np.asarray(
        [
            [1.0, 0.0],
            [0.75, 0.2],
            [0.25, 0.2],
            [0.0, 0.0],
            [0.25, -0.2],
            [0.75, -0.2],
        ]
    )
    value, diagnostics = _su2_airfoil_max_thickness(coordinates)
    assert value == pytest.approx(0.4)
    assert diagnostics["selected_node_index"] == 1


def test_saved_gradient_loader_and_projection_validation(tmp_path):
    state = _space_state(
        np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5]]
        )
    )
    field = np.asarray([2.0, 3.0, 4.0, 5.0])
    reference = project_surface_field(state, field)
    path = tmp_path / "of_grad_cd.csv"
    path.write_text(
        '"VARIABLE", "GRADIENT", "FINDIFF_STEP"\n'
        + "".join(
            f"{index}, {value:.16e}, 1e-3\n"
            for index, value in enumerate(reference)
        )
    )
    loaded = load_design_gradient_csv(path)
    validation = validate_surface_projection(state, field, loaded)
    assert validation["passed"] is True
    assert validation["max_absolute_error"] == pytest.approx(0.0)


def test_surface_ikkt_fit_allows_negative_equality_multiplier():
    state = _space_state(np.asarray([[1.0], [0.0], [0.0], [0.0]]))
    equality = internal_constraint_field(
        {"name": "EQ", "sign": "=", "target": 0.0},
        0.0,
        np.asarray([1.0, 0.0, 0.0, 0.0]),
    )
    residual, lambdas, _ = fit_surface_ikkt_signal(
        np.asarray([-2.0, 0.0, 0.0, 0.0]),
        [{"name": "EQ", **equality}],
        state,
    )
    assert lambdas == pytest.approx([-2.0])
    assert residual == pytest.approx(np.zeros(4), abs=1e-12)


def test_virtual_pass_fits_ikkt_once_and_reuses_signal_for_all_candidates(
    tmp_path,
    monkeypatch,
):
    baseline_matrix = np.asarray([[1.0], [0.0], [0.0], [0.0]])
    candidate_matrices = {
        0.25: np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]
        ),
        0.75: np.asarray(
            [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
        ),
    }
    states = {}

    def fake_prepare(
        _level,
        _opts,
        _cfg_level,
        _real_dot_cfg,
        _mesh_src,
        active_by_side,
        suffix,
        verbose=False,
    ):
        del verbose
        active = list(active_by_side["UPPER"])
        if "baseline" in suffix:
            matrix = baseline_matrix
        else:
            inserted = next(value for value in active if value != 0.5)
            matrix = candidate_matrices[inserted]
        mesh = tmp_path / f"{suffix}.su2"
        mesh.write_text("temporary")
        states[str(mesh)] = {
            "node_ids": [0, 1],
            "matrix": matrix,
            "x_over_c": np.asarray([0.25, 0.75]),
        }
        return {
            "mesh": str(mesh),
            "records": [("UPPER", value) for value in sorted(active)],
            "column_index_by_side": {
                "UPPER": {value: index for index, value in enumerate(sorted(active))}
            },
        }

    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._prepare_active_projection_variant",
        fake_prepare,
    )
    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection.build_ffd_tangent_state",
        lambda mesh, *_args, **_kwargs: states[str(mesh)],
    )
    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._find_real_adjoint_assets",
        lambda *_args: (str(tmp_path), str(tmp_path)),
    )
    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._select_dot_config_path",
        lambda *_args: (str(tmp_path / "dot.cfg"), "DISCRETE_ADJOINT"),
    )
    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection.SU2.io.Config",
        lambda *_args: {"MESH_FILENAME": "mesh.su2"},
    )
    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._find_projection_mesh_source",
        lambda *_args: "mesh.su2",
    )
    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._resolve_accepted_projection_mesh",
        lambda *_args, **_kwargs: "mesh.su2",
    )
    fit_calls = []

    def fake_fit(objective, constraints, baseline):
        fit_calls.append((objective.copy(), list(constraints), baseline["matrix"].copy()))
        return np.asarray([0.0, 2.0, 1.0, 0.0]), np.asarray([0.5]), {
            "status": "ok",
            "residual_gradient_norm": 1.0,
        }

    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection.fit_surface_ikkt_signal",
        fake_fit,
    )
    context = {
        "initialized_node_ids": [0, 1],
        "objective_function": "DRAG",
        "objective_field_file": "surface_sens.csv",
        "objective_field": np.asarray([1.0, 2.0, 1.0, 0.0]),
        "constraint_records": [
            {
                "name": "LIFT",
                "field": np.asarray([1.0, 0.0, 0.0, 0.0]),
                "lambda_lower": 0.0,
                "lambda_upper": np.inf,
            }
        ],
        "inactive_constraints": [],
        "unsupported_constraints": [],
        "progressive_thickness": {"enabled": True, "included": False},
    }
    raw = [
        {
            "side": "UPPER",
            "x": x,
            "interval_id": index,
            "interval_left": 0.0 if index == 0 else 0.5,
            "interval_right": 0.5 if index == 0 else 1.0,
            "sample_index": 1,
            "sample_fraction": 0.5,
        }
        for index, x in enumerate((0.25, 0.75))
    ]
    result = _compute_dual_virtual_tangent_candidate_scores_impl(
        SimpleNamespace(workdir=str(tmp_path), level_id=0),
        {
            "adaptive_indicator": "IKKT",
            "ffd_marker": "AIRFOIL",
            "ffd_blending": "BEZIER",
        },
        {"OBJECTIVE_FUNCTION": "DRAG"},
        raw,
        {"UPPER": [0.5]},
        context,
    )
    assert len(fit_calls) == 1
    assert len(result["raw_candidates"]) == 2
    assert result["selected_candidate"]["x"] == pytest.approx(0.25)
    assert result["ikkt_lambdas"] == pytest.approx([0.5])


def test_sequential_virtual_selection_reinvokes_pass_with_updated_baseline(
    tmp_path,
    monkeypatch,
):
    level = SimpleNamespace(level_id=0, ndv=7, workdir=str(tmp_path))
    opts = {
        "ffd_scoring_mode": "VIRTUAL_TANGENT",
        "ffd_blending": "BEZIER",
        "adaptive_indicator": "IKKT",
        "growth_ratio": 1.25,
        "candidate_samples": 1,
        "nadd_mode": "GROWTH_RATIO",
    }
    initial = {
        "UPPER": [0.0, 0.125, 0.25, 0.5, 0.75, 0.875, 1.0]
    }
    raw_first = [
        {
            "side": "UPPER",
            "x": 0.0625,
            "interval_id": 0,
            "interval_left": 0.0,
            "interval_right": 0.125,
            "sample_index": 1,
            "sample_fraction": 0.5,
        }
    ]
    raw_second = [
        {
            "side": "UPPER",
            "x": 0.09375,
            "interval_id": 1,
            "interval_left": 0.0625,
            "interval_right": 0.125,
            "sample_index": 1,
            "sample_fraction": 0.5,
        }
    ]
    shared_context = {"token": object()}
    calls = []

    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._find_real_adjoint_assets",
        lambda *_args: (str(tmp_path), str(tmp_path)),
    )
    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._prepare_virtual_tangent_context",
        lambda *_args: shared_context,
    )
    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._generate_dual_exact_candidates",
        lambda *_args: raw_second,
    )

    def fake_pass(
        _level,
        _opts,
        _cfg,
        candidates,
        active,
        context,
        **kwargs,
    ):
        calls.append((context, {key: list(value) for key, value in active.items()}))
        chosen = dict(candidates[0])
        chosen.update(
            {
                "indicator": 1.0,
                "rank": 1,
                "ndv_before_insertion": sum(len(value) for value in active.values()),
                "ndv_after_insertion": sum(len(value) for value in active.values()) + 1,
                "objective_function": "DRAG",
                "objective_gradient_component": 1.0,
                "constraint_gradient_components": {},
                "temporary_mesh": str(tmp_path / f"candidate_{len(calls)}.su2"),
                "projection_artifacts": [],
            }
        )
        return {
            "selected_candidate": chosen,
            "raw_candidates": [chosen],
            "ikkt_metadata": {"step": kwargs["insertion_step"]},
        }

    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._compute_dual_virtual_tangent_candidate_scores_impl",
        fake_pass,
    )
    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._persist_selected_candidate_artifacts",
        lambda _level, _selected, _mode, _metadata, selected_dir=None: (
            selected_dir or str(tmp_path / "FFD_SELECTED_CANDIDATE")
        ),
    )
    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._write_exact_candidate_scores_csv",
        lambda *_args: str(tmp_path / "scores.csv"),
    )
    result = _compute_dual_bezier_exact_candidate_scores(
        level,
        opts,
        {"OBJECTIVE_FUNCTION": "DRAG"},
        raw_first,
        initial,
    )
    assert result["insertions_completed"] == 2
    assert len(calls) == 2
    assert calls[0][0] is shared_context and calls[1][0] is shared_context
    assert 0.0625 not in calls[0][1]["UPPER"]
    assert 0.0625 in calls[1][1]["UPPER"]


def test_next_virtual_level_reuses_last_winning_candidate_mesh(
    tmp_path,
    monkeypatch,
):
    winner = tmp_path / "candidate.su2"
    winner.write_text("winner")
    level = SimpleNamespace(
        level_id=0,
        mesh_source="initial.su2",
        initial_mesh_source="initial.su2",
        dual_box=False,
        side="UPPER",
    )
    selection = {
        "ffd_scoring_mode": "VIRTUAL_TANGENT",
        "selected": [{"temporary_mesh": str(winner)}],
    }
    opts = {
        "_last_selection_metadata": selection,
        "ffd_dual_box": False,
        "ffd_box_tag": "UPPER_BOX",
        "ffd_dv_kind": "FFD_CONTROL_POINT_2D",
        "ffd_marker": "AIRFOIL",
        "ffd_domain_mode": "HALF_UPPER",
        "ffd_control_row": 1,
        "ffd_direction": "OUTWARD",
        "ffd_active_xmin": 0.0,
        "ffd_active_xmax": 1.0,
        "ffd_active_include_bounds": False,
        "ffd_side": "UPPER",
    }
    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_levels.refine_ffd_columns",
        lambda *_args: [0.25, 0.5, 0.75],
    )
    next_level = build_next_ffd_level(
        level,
        {"final_mesh": "accepted.su2"},
        opts,
    )
    assert next_level.mesh_source == str(winner)


def test_surface_ikkt_fit_uses_c_ge_zero_sign_convention():
    state = _space_state(np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]))
    objective = np.asarray([2.0, -3.0, 0.0, 0.0])
    lower = internal_constraint_field(
        {"name": "LOWER", "sign": ">", "target": 0.0},
        0.0,
        np.asarray([1.0, 0.0, 0.0, 0.0]),
    )
    upper = internal_constraint_field(
        {"name": "UPPER", "sign": "<", "target": 0.0},
        0.0,
        np.asarray([0.0, 1.0, 0.0, 0.0]),
    )
    constraints = [
        {"name": "LOWER", **lower},
        {"name": "UPPER", **upper},
    ]
    residual, lambdas, diagnostics = fit_surface_ikkt_signal(
        objective,
        constraints,
        state,
    )
    assert lambdas == pytest.approx([2.0, 3.0])
    assert residual == pytest.approx(np.zeros(4), abs=1e-12)
    assert diagnostics["residual_gradient_norm"] == pytest.approx(0.0, abs=1e-12)


def _write_vector_sensitivity(path, node_ids, scale):
    path.write_text(
        "PointID,Sensitivity_x,Sensitivity_y\n"
        + "".join(
            f"{node_id},{0.25 * scale * (index + 1)},{scale * (index + 1)}\n"
            for index, node_id in enumerate(node_ids)
        )
    )


def test_virtual_ikkt_context_includes_aero_area_and_thickness_fields(
    tmp_path,
    monkeypatch,
):
    state, opts = _dual_state(tmp_path)
    opts["ffd_thickness_enabled"] = True
    objective_file = tmp_path / "objective_surface_sens.csv"
    _write_vector_sensitivity(objective_file, state["node_ids"], 1.0)
    design_dir = tmp_path / "DESIGNS" / "DSN_001"
    lift_dir = design_dir / "ADJOINT_LIFT"
    lift_dir.mkdir(parents=True)
    _write_vector_sensitivity(
        lift_dir / "surface_sens.csv",
        state["node_ids"],
        0.2,
    )
    area_value, _area_field = airfoil_area_value_and_field(state)
    thickness_value, _thickness_field, _ = airfoil_thickness_value_and_field(state)

    def fake_assets(_level_dir, function_name):
        assert str(function_name).upper() == "LIFT"
        return str(lift_dir), str(design_dir)

    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._find_real_adjoint_assets",
        fake_assets,
    )
    context = {
        "objective_function": "DRAG",
        "objective_field_file": str(objective_file),
        "design_dir": str(design_dir),
        "active_constraint_names": [
            "LIFT",
            "AIRFOIL_AREA",
            "AIRFOIL_THICKNESS",
        ],
        "constraint_status": [
            {
                "name": "LIFT",
                "current_value": 0.5,
                "included": True,
                "status": "active_equality",
            },
            {
                "name": "AIRFOIL_AREA",
                "current_value": area_value,
                "included": True,
                "status": "active_inequality",
            },
            {
                "name": "AIRFOIL_THICKNESS",
                "current_value": thickness_value,
                "included": True,
                "status": "active_inequality",
            },
            {
                "name": "MOMENT_Z",
                "current_value": 0.0,
                "included": False,
                "status": "inactive",
            },
        ],
        "constraint_specs": {
            "LIFT": {
                "name": "LIFT",
                "sign": "=",
                "target": 0.5,
                "scale": 1.0,
            },
            "AIRFOIL_AREA": {
                "name": "AIRFOIL_AREA",
                "sign": ">",
                "target": area_value,
                "scale": 1.0,
            },
            "AIRFOIL_THICKNESS": {
                "name": "AIRFOIL_THICKNESS",
                "sign": ">",
                "target": thickness_value,
                "scale": 1.0,
            },
        },
        "active_tol": 1.0e-6,
        "initialized_node_ids": None,
    }
    initialized = _initialize_virtual_tangent_context(
        SimpleNamespace(workdir=str(tmp_path)),
        opts,
        state,
        context,
    )
    by_name = {
        record["name"]: record
        for record in initialized["constraint_records"]
    }
    assert set(by_name) == {"LIFT", "AIRFOIL_AREA", "AIRFOIL_THICKNESS"}
    assert by_name["LIFT"]["provider"] == "adjoint_surface_sensitivity"
    assert by_name["AIRFOIL_AREA"]["provider"] == "analytic_airfoil_area"
    assert (
        by_name["AIRFOIL_THICKNESS"]["provider"]
        == "analytic_airfoil_thickness"
    )
    assert initialized["progressive_thickness"]["enabled"] is True
    assert initialized["progressive_thickness"]["included"] is False
    assert {
        record["name"] for record in initialized["inactive_constraints"]
    } == {"MOMENT_Z"}
    residual, lambdas, diagnostics = fit_surface_ikkt_signal(
        initialized["objective_field"],
        initialized["constraint_records"],
        state,
    )
    assert len(lambdas) == 3
    assert np.all(np.isfinite(residual))
    assert diagnostics["status"] == "ok"


def test_virtual_ikkt_rejects_unsupported_active_geometry_constraint(tmp_path):
    state, opts = _dual_state(tmp_path)
    objective_file = tmp_path / "objective_surface_sens.csv"
    _write_vector_sensitivity(objective_file, state["node_ids"], 1.0)
    context = {
        "objective_function": "DRAG",
        "objective_field_file": str(objective_file),
        "design_dir": str(tmp_path),
        "active_constraint_names": ["AIRFOIL_CHORD"],
        "constraint_status": [
            {
                "name": "AIRFOIL_CHORD",
                "current_value": 1.0,
                "included": True,
                "status": "active_inequality",
            }
        ],
        "constraint_specs": {
            "AIRFOIL_CHORD": {
                "name": "AIRFOIL_CHORD",
                "sign": ">",
                "target": 1.0,
                "scale": 1.0,
            }
        },
        "active_tol": 1.0e-6,
        "initialized_node_ids": None,
    }
    with pytest.raises(RuntimeError, match="unsupported_geometry_constraint"):
        _initialize_virtual_tangent_context(
            SimpleNamespace(workdir=str(tmp_path)),
            opts,
            state,
            context,
        )
