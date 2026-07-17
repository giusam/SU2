import numpy as np
import pytest

from SU2.opt.progressive_hh_core import HHLevel, get_progressive_hh_options
from SU2.opt.progressive_hh_levels import _remove_progressive_keys
from SU2.opt.progressive_hh_projection import (
    _compute_dot_candidate_scores,
    _hh_tangent_records_match_config,
)
from SU2.opt.progressive_hh_tangent import (
    COMPONENT,
    VIRTUAL_TANGENT,
    build_hh_tangent_state,
    compare_tangent_spaces,
    hicks_henne_kernel,
    hicks_henne_t2_for_center,
    hicks_henne_t2_policy,
    normalize_hh_scoring_mode,
)


def _write_airfoil_mesh(path):
    points = [
        (1.0, 0.0),
        (0.75, 0.06),
        (0.5, 0.08),
        (0.25, 0.05),
        (0.0, 0.0),
        (0.25, -0.05),
        (0.5, -0.08),
        (0.75, -0.06),
    ]
    lines = ["NDIME= 2\n", f"NPOIN= {len(points)}\n"]
    for index, (x, y) in enumerate(points):
        lines.append(f"{x:.16g} {y:.16g} {index}\n")
    lines.extend(
        [
            "NMARK= 1\n",
            "MARKER_TAG= AIRFOIL\n",
            f"MARKER_ELEMS= {len(points)}\n",
        ]
    )
    for index in range(len(points)):
        lines.append(f"3 {index} {(index + 1) % len(points)}\n")
    path.write_text("".join(lines))
    return points


def _write_half_upper_mesh(path):
    points = [(1.0, 0.0), (0.75, 0.06), (0.5, 0.08), (0.25, 0.05), (0.0, 0.0)]
    lines = ["NDIME= 2\n", f"NPOIN= {len(points)}\n"]
    for index, (x, y) in enumerate(points):
        lines.append(f"{x:.16g} {y:.16g} {index}\n")
    lines.extend(
        [
            "NMARK= 1\n",
            "MARKER_TAG= AIRFOIL\n",
            f"MARKER_ELEMS= {len(points) - 1}\n",
        ]
    )
    for index in range(len(points) - 1):
        lines.append(f"3 {index} {index + 1}\n")
    path.write_text("".join(lines))
    return points


def _base_config(**overrides):
    values = {
        "PROGRESSIVE_HH": "YES",
        "PROGRESSIVE_PARAM_KIND": "HICKS_HENNE",
        "PROGRESSIVE_HH_REFINEMENT": "ADAPTIVE",
        "PROGRESSIVE_HH_ADAPTIVE_INDICATOR": "IKKT",
        "PROGRESSIVE_HH_N0": 2,
        "PROGRESSIVE_HH_SURFACE": "UPPER",
        "PROGRESSIVE_HH_NADD_MODE": "GROWTH_RATIO",
        "PROGRESSIVE_HH_GROWTH_RATIO": 1.25,
        "OPT_ITERATIONS": 10,
        "OPT_BOUND_LOWER": -0.02,
        "OPT_BOUND_UPPER": 0.02,
    }
    values.update(overrides)
    return values


def test_component_remains_default_and_virtual_mode_is_explicit():
    default = get_progressive_hh_options(_base_config())
    assert default["scoring_mode"] == COMPONENT
    assert default["scoring_te_closure_node_eps"] == pytest.approx(0.0)
    opts = get_progressive_hh_options(
        _base_config(
            PROGRESSIVE_HH_SCORING_MODE=VIRTUAL_TANGENT,
            PROGRESSIVE_HH_SCORING_TE_CLOSURE_NODE_EPS=0.01,
        )
    )
    assert opts["scoring_mode"] == VIRTUAL_TANGENT
    assert opts["scoring_te_closure_node_eps"] == pytest.approx(0.01)
    with pytest.raises(ValueError, match="PROGRESSIVE_HH_SCORING_MODE"):
        normalize_hh_scoring_mode("UNKNOWN")


def test_hh_scoring_mask_requires_virtual_tangent_and_valid_width():
    with pytest.raises(ValueError, match="requires.*VIRTUAL_TANGENT"):
        get_progressive_hh_options(
            _base_config(PROGRESSIVE_HH_SCORING_TE_CLOSURE_NODE_EPS=0.01)
        )

    for value in (-0.01, 1.0, "NOT_A_NUMBER"):
        with pytest.raises(ValueError, match="finite number in"):
            get_progressive_hh_options(
                _base_config(
                    PROGRESSIVE_HH_SCORING_MODE=VIRTUAL_TANGENT,
                    PROGRESSIVE_HH_SCORING_TE_CLOSURE_NODE_EPS=value,
                )
            )


def test_native_level_config_does_not_receive_python_scoring_key():
    config = {
        "PROGRESSIVE_HH_SCORING_MODE": VIRTUAL_TANGENT,
        "PROGRESSIVE_HH_SCORING_TE_CLOSURE_NODE_EPS": 0.01,
        "HICKS_HENNE_T2_BY_CENTER": "YES",
    }
    _remove_progressive_keys(config)
    assert "PROGRESSIVE_HH_SCORING_MODE" not in config
    assert "PROGRESSIVE_HH_SCORING_TE_CLOSURE_NODE_EPS" not in config
    assert config["HICKS_HENNE_T2_BY_CENTER"] == "YES"


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"PROGRESSIVE_HH_REFINEMENT": "UNIFORM"}, "REFINEMENT=ADAPTIVE"),
        ({"PROGRESSIVE_HH_ADAPTIVE_INDICATOR": "DESCENT_GRAD"}, "ABS_GRAD or IKKT"),
        ({"OPT_BOUND_LOWER": 0.0}, "bilateral DV bounds"),
    ],
)
def test_virtual_mode_rejects_unsupported_contracts(overrides, message):
    values = {"PROGRESSIVE_HH_SCORING_MODE": VIRTUAL_TANGENT}
    values.update(overrides)
    with pytest.raises(ValueError, match=message):
        get_progressive_hh_options(_base_config(**values))


def test_t2_policy_uses_forward_branch_at_switch():
    policy = hicks_henne_t2_policy(
        {
            "HICKS_HENNE_T2_BY_CENTER": "YES",
            "HICKS_HENNE_T2_FORWARD": 3.0,
            "HICKS_HENNE_T2_AFT": 1.0,
            "HICKS_HENNE_T2_SWITCH_X": 0.5,
        }
    )
    assert [
        hicks_henne_t2_for_center(center, policy)
        for center in (0.25, 0.5, 0.75)
    ] == [3.0, 3.0, 1.0]
    assert hicks_henne_kernel(0.25, 0.25, 3.0) == pytest.approx(1.0)
    assert hicks_henne_kernel(0.75, 0.75, 1.0) == pytest.approx(1.0)


def test_uniform_t2_policy_ignores_inactive_center_parameters():
    policy = hicks_henne_t2_policy(
        {
            "HICKS_HENNE_T2": 2.0,
            "HICKS_HENNE_T2_BY_CENTER": "NO",
            "HICKS_HENNE_T2_FORWARD": -3.0,
            "HICKS_HENNE_T2_SWITCH_X": 2.0,
        }
    )
    assert hicks_henne_t2_for_center(0.75, policy) == 2.0
    with pytest.raises(ValueError, match="YES/NO boolean"):
        hicks_henne_t2_policy({"HICKS_HENNE_T2_BY_CENTER": "MAYBE"})


def test_hh_tangent_state_matches_native_side_sign_and_t2(tmp_path):
    mesh = tmp_path / "airfoil.su2"
    _write_airfoil_mesh(mesh)
    state = build_hh_tangent_state(
        mesh,
        "AIRFOIL",
        {"UPPER": [0.25, 0.5], "LOWER": [0.75]},
        {
            "HICKS_HENNE_T2_BY_CENTER": "YES",
            "HICKS_HENNE_T2_FORWARD": 3.0,
            "HICKS_HENNE_T2_AFT": 1.0,
            "HICKS_HENNE_T2_SWITCH_X": 0.5,
        },
    )
    assert state["records"] == [
        ("UPPER", 0.25),
        ("UPPER", 0.5),
        ("LOWER", 0.75),
    ]
    assert state["t2_by_record"] == [3.0, 3.0, 1.0]
    assert state["matrix"].shape == (16, 3)
    assert np.allclose(state["matrix"][0::2], 0.0)
    upper_rows = [index for index, side in enumerate(state["sides"]) if side == "upper"]
    lower_rows = [index for index, side in enumerate(state["sides"]) if side == "lower"]
    assert np.all(state["matrix"][2 * np.asarray(upper_rows) + 1, 0] >= 0.0)
    assert np.all(state["matrix"][2 * np.asarray(lower_rows) + 1, 2] <= 0.0)


def test_open_half_upper_marker_uses_the_declared_active_side(tmp_path):
    mesh = tmp_path / "half_upper.su2"
    _write_half_upper_mesh(mesh)
    state = build_hh_tangent_state(
        mesh,
        "AIRFOIL",
        {"UPPER": [0.25, 0.5, 0.75]},
    )
    assert state["closed"] is False
    assert set(state["sides"]) == {"upper"}
    assert np.all(state["matrix"][1::2] >= 0.0)


def test_svd_energy_is_the_residualized_candidate_energy(tmp_path):
    mesh = tmp_path / "airfoil.su2"
    _write_airfoil_mesh(mesh)
    baseline = build_hh_tangent_state(
        mesh,
        "AIRFOIL",
        {"UPPER": [0.25, 0.75]},
    )
    candidate = build_hh_tangent_state(
        mesh,
        "AIRFOIL",
        {"UPPER": [0.25, 0.5, 0.75]},
    )
    direction = candidate["matrix"][:, 1]
    q, _ = np.linalg.qr(baseline["matrix"])
    signal = direction - q @ (q.T @ direction)
    metrics = compare_tangent_spaces(baseline, candidate, signal, 0.5)
    assert metrics["rank_gain"] == 1
    assert metrics["pure_rank"] == 1
    assert metrics["score_pure"] > 0.0
    assert metrics["score_net"] == pytest.approx(metrics["score_pure"], rel=1e-11)
    assert metrics["energy_candidate"] == pytest.approx(
        metrics["energy_current"] + metrics["score_pure"], rel=1e-11
    )


def test_saved_dot_validation_is_strict_only_for_the_same_hh_centers():
    state = {
        "records": [("UPPER", 0.25), ("UPPER", 0.75)],
        "symmetry_mode": "NONE",
        "symmetry_sign": -1.0,
    }
    matching = {
        "DEFINITION_DV": {
            "KIND": ["HICKS_HENNE", "HICKS_HENNE"],
            "PARAM": [[1.0, 0.25], [1.0, 0.75]],
        }
    }
    redistributed = {
        "DEFINITION_DV": {
            "KIND": ["HICKS_HENNE", "HICKS_HENNE"],
            "PARAM": [[1.0, 0.2], [1.0, 0.8]],
        }
    }
    assert _hh_tangent_records_match_config(state, matching) is True
    assert _hh_tangent_records_match_config(state, redistributed) is False


def test_production_virtual_score_selects_the_residualized_center(tmp_path):
    mesh = tmp_path / "airfoil.su2"
    points = _write_airfoil_mesh(mesh)
    level_cfg = tmp_path / "config_level0.cfg"
    level_cfg.write_text(
        "\n".join(
            [
                "OBJECTIVE_FUNCTION= DRAG",
                "GRADIENT_METHOD= DISCRETE_ADJOINT",
                "MATH_PROBLEM= DIRECT",
                "MESH_FILENAME= airfoil.su2",
                "DV_MARKER= ( AIRFOIL )",
                "OPT_CONSTRAINT= NONE",
                "HICKS_HENNE_T2= 1.0",
                "HICKS_HENNE_T2_BY_CENTER= NO",
            ]
        )
        + "\n"
    )
    adjoint = tmp_path / "DESIGNS" / "DSN_000" / "ADJOINT_DRAG"
    adjoint.mkdir(parents=True)
    (adjoint / "config_DOT_AD.cfg").write_text(
        "MATH_PROBLEM= DISCRETE_ADJOINT\n"
        "GRADIENT_METHOD= DISCRETE_ADJOINT\n"
        "MESH_FILENAME= airfoil.su2\n"
    )

    baseline = build_hh_tangent_state(
        mesh,
        "AIRFOIL",
        {"UPPER": [0.25, 0.75]},
    )
    target = build_hh_tangent_state(
        mesh,
        "AIRFOIL",
        {"UPPER": [0.25, 0.5, 0.75]},
    )
    direction = target["matrix"][:, 1]
    q, _ = np.linalg.qr(baseline["matrix"])
    signal = direction - q @ (q.T @ direction)
    rows = [
        '"PointID","x","y","Sensitivity_x","Sensitivity_y","Surface_Sensitivity"\n'
    ]
    for index, (x, y) in enumerate(points):
        rows.append(
            f"{index},{x:.16e},{y:.16e},{signal[2*index]:.16e},"
            f"{signal[2*index+1]:.16e},0.0\n"
        )
    (adjoint / "surface_sens.csv").write_text("".join(rows))

    level = HHLevel(
        level_id=0,
        upper=[0.25, 0.75],
        lower=[],
        workdir=str(tmp_path),
        config_filename=level_cfg.name,
        project_filename="project_level0.pkl",
        mesh_source=str(mesh),
    )
    opts = {
        "scoring_mode": VIRTUAL_TANGENT,
        "adaptive_indicator": "IKKT",
        "candidate_samples": 1,
        "min_center_spacing": 0.0,
        "ikkt_active_tol": 1.0e-6,
        "marker": "AIRFOIL",
        "scale": 1.0,
        "symmetry_mode": "NONE",
        "symmetry_sign": -1.0,
        "nadd_mode": "GROWTH_RATIO",
        "growth_ratio": 1.25,
        "nfinal": None,
        "scoring_te_closure_node_eps": 0.01,
    }
    result = _compute_dot_candidate_scores(level, opts)
    assert result["sequential_selection"] is True
    assert len(result["selected_candidates"]) == 1
    assert result["selected_candidates"][0]["x"] == pytest.approx(0.5)
    assert result["selected_candidates"][0]["side"] == "UPPER"
    assert result["surface_scoring_mask"]["node_count_removed"] == 1
    assert (tmp_path / "hh_candidate_scores_level0.csv").is_file()
    assert (tmp_path / "hh_virtual_tangent_level0.json").is_file()
