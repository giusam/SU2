import math
from types import SimpleNamespace

import numpy as np
import pytest

from SU2.opt.thickness_constraint import ThicknessConstraint, _hicks_henne_bump


def test_hicks_henne_bump_matches_native_su2_kernel():
    x = 0.37
    center = 0.61
    exponent = math.log(0.5) / math.log(center)
    expected = math.sin(math.pi * (x**exponent))

    assert _hicks_henne_bump(x, center) == pytest.approx(expected)


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
