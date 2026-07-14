import math
from types import SimpleNamespace

import numpy as np
import pytest

from SU2.opt.thickness_constraint import (
    ThicknessConstraint,
    _hicks_henne_bump,
    _hicks_henne_t2_for_center,
    _hicks_henne_t2_policy,
)


def test_hicks_henne_bump_matches_native_su2_kernel():
    x = 0.37
    center = 0.61
    exponent = math.log(0.5) / math.log(center)
    expected = math.sin(math.pi * (x**exponent))

    assert _hicks_henne_bump(x, center) == pytest.approx(expected)


def test_hicks_henne_bump_supports_uniform_t2_exponent():
    x = 0.37
    center = 0.61
    exponent = math.log(0.5) / math.log(center)
    expected = math.sin(math.pi * (x**exponent)) ** 3

    assert _hicks_henne_bump(x, center, t2=3.0) == pytest.approx(expected)


def test_hicks_henne_t2_is_selected_once_from_the_bump_center():
    config = {
        "HICKS_HENNE_T2_BY_CENTER": "YES",
        "HICKS_HENNE_T2_FORWARD": 3.0,
        "HICKS_HENNE_T2_AFT": 1.0,
        "HICKS_HENNE_T2_SWITCH_X": 0.5,
    }
    policy = _hicks_henne_t2_policy(config)

    assert _hicks_henne_t2_for_center(0.25, policy=policy) == 3.0
    assert _hicks_henne_t2_for_center(0.5, policy=policy) == 3.0
    assert _hicks_henne_t2_for_center(0.75, policy=policy) == 1.0


def test_hicks_henne_analytic_thickness_jacobian_includes_relax_factor():
    constraint = ThicknessConstraint(
        ref_mesh="unused.su2",
        marker="AIRFOIL",
        x_stations=[0.25, 0.5, 0.75],
        reference_measure=[0.0, 0.0, 0.0],
        gradient_mode="ANALYTIC",
        domain_mode="FULL",
    )
    definition = {
        "KIND": ["HICKS_HENNE", "HICKS_HENNE"],
        "SCALE": [2.0, 3.0],
        "MARKER": [["AIRFOIL"], ["AIRFOIL"]],
        "SIZE": [1, 1],
        "PARAM": [[1.0, 0.5], [0.0, 0.25]],
        "FFDTAG": [[], []],
    }
    project = SimpleNamespace(
        config={
            "DEFINITION_DV": definition,
            "OPT_RELAX_FACTOR": 37.0,
        }
    )

    jacobian = constraint.jacobian_analytic([0.0, 0.0], project)
    expected = np.asarray(
        [
            [
                2.0 * 37.0 * _hicks_henne_bump(x, 0.5),
                3.0 * 37.0 * _hicks_henne_bump(x, 0.25),
            ]
            for x in constraint.x_stations
        ]
    )

    assert jacobian == pytest.approx(expected)


def test_hicks_henne_analytic_jacobian_uses_center_selected_t2():
    constraint = ThicknessConstraint(
        ref_mesh="unused.su2",
        marker="AIRFOIL",
        x_stations=[0.2, 0.6, 0.9],
        reference_measure=[0.0, 0.0, 0.0],
        gradient_mode="ANALYTIC",
        domain_mode="FULL",
    )
    definition = {
        "KIND": ["HICKS_HENNE"] * 3,
        "SCALE": [1.0, 2.0, 3.0],
        "MARKER": [["AIRFOIL"]] * 3,
        "SIZE": [1, 1, 1],
        "PARAM": [[1.0, 0.25], [1.0, 0.5], [1.0, 0.75]],
        "FFDTAG": [[], [], []],
    }
    project = SimpleNamespace(
        config={
            "DEFINITION_DV": definition,
            "OPT_RELAX_FACTOR": 4.0,
            "HICKS_HENNE_T2_BY_CENTER": "YES",
            "HICKS_HENNE_T2_FORWARD": 3.0,
            "HICKS_HENNE_T2_AFT": 1.0,
            "HICKS_HENNE_T2_SWITCH_X": 0.5,
        }
    )

    jacobian = constraint.jacobian_analytic([0.0, 0.0, 0.0], project)
    expected = np.asarray(
        [
            [
                4.0 * _hicks_henne_bump(x, 0.25, t2=3.0),
                8.0 * _hicks_henne_bump(x, 0.5, t2=3.0),
                12.0 * _hicks_henne_bump(x, 0.75, t2=1.0),
            ]
            for x in constraint.x_stations
        ]
    )

    assert jacobian == pytest.approx(expected)


def test_thickness_value_cache_distinguishes_relax_factor():
    constraint = ThicknessConstraint(
        ref_mesh="unused.su2",
        marker="AIRFOIL",
        x_stations=[0.5],
        reference_measure=[0.0],
    )
    cfg = {
        "MESH_FILENAME": "mesh.su2",
        "DV_KIND": "HICKS_HENNE",
        "DV_MARKER": "AIRFOIL",
        "DEFINITION_DV": {"KIND": ["HICKS_HENNE"]},
        "DV_VALUE_OLD": [0.0],
        "OPT_RELAX_FACTOR": 1.0,
    }

    key_unit = constraint._cache_key([1.0e-6], cfg)
    cfg["OPT_RELAX_FACTOR"] = 1000.0
    key_relaxed = constraint._cache_key([1.0e-6], cfg)

    assert key_unit != key_relaxed


def test_thickness_value_cache_uses_effective_hicks_henne_t2_policy():
    constraint = ThicknessConstraint(
        ref_mesh="unused.su2",
        marker="AIRFOIL",
        x_stations=[0.5],
        reference_measure=[0.0],
    )
    cfg = {
        "MESH_FILENAME": "mesh.su2",
        "DV_KIND": "HICKS_HENNE",
        "DV_MARKER": "AIRFOIL",
        "DEFINITION_DV": {"KIND": ["HICKS_HENNE"]},
        "DV_VALUE_OLD": [0.0],
    }

    legacy_key = constraint._cache_key([1.0e-6], cfg)
    cfg.update({"HICKS_HENNE_T2_BY_CENTER": "NO", "HICKS_HENNE_T2": 1.0})
    explicit_legacy_key = constraint._cache_key([1.0e-6], cfg)
    assert explicit_legacy_key == legacy_key

    cfg["HICKS_HENNE_T2"] = 3.0
    uniform_three_key = constraint._cache_key([1.0e-6], cfg)
    assert uniform_three_key != legacy_key

    cfg.update(
        {
            "HICKS_HENNE_T2_BY_CENTER": "YES",
            "HICKS_HENNE_T2_FORWARD": 3.0,
            "HICKS_HENNE_T2_AFT": 1.0,
            "HICKS_HENNE_T2_SWITCH_X": 0.5,
        }
    )
    by_center_key = constraint._cache_key([1.0e-6], cfg)
    assert by_center_key not in (legacy_key, uniform_three_key)


@pytest.mark.parametrize(
    "config,match",
    [
        ({"HICKS_HENNE_T2": 0.0}, "HICKS_HENNE_T2"),
        (
            {
                "HICKS_HENNE_T2_BY_CENTER": "YES",
                "HICKS_HENNE_T2_FORWARD": float("nan"),
            },
            "HICKS_HENNE_T2_FORWARD",
        ),
        (
            {
                "HICKS_HENNE_T2_BY_CENTER": "YES",
                "HICKS_HENNE_T2_SWITCH_X": 1.0,
            },
            "HICKS_HENNE_T2_SWITCH_X",
        ),
        ({"HICKS_HENNE_T2_BY_CENTER": "MAYBE"}, "HICKS_HENNE_T2_BY_CENTER"),
    ],
)
def test_invalid_hicks_henne_t2_policy_is_rejected(config, match):
    with pytest.raises(ValueError, match=match):
        _hicks_henne_t2_policy(config)
