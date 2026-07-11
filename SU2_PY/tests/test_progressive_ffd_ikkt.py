import numpy as np
import pytest

import SU2
from SU2.opt.progressive_hh_core import get_progressive_hh_options
from SU2.opt.progressive_hh_projection import (
    _compute_ikkt_residual_vector,
    _extract_constraint_names,
    _extract_constraint_signs,
    _select_active_ikkt_constraints,
)


def _ikkt_config(tmp_path):
    filename = tmp_path / "ikkt.cfg"
    filename.write_text(
        "MESH_FILENAME= mesh.su2\n"
        "OPT_ITERATIONS= 5\n"
        "OPT_CONSTRAINT= ( MOMENT_Z < 0.092 ); ( LIFT = 0.824 ); "
        "( AIRFOIL_AREA > 0.0778 )\n"
    )
    return SU2.io.Config(str(filename))


def test_ffd_ikkt_raw_gradient_multiplier_bounds_follow_slsqp_convention(tmp_path):
    cfg = _ikkt_config(tmp_path)
    names = _extract_constraint_names(cfg)
    lower, upper = _extract_constraint_signs(cfg, names)
    bounds = dict(zip(names, zip(lower, upper)))

    assert bounds["LIFT"] == (-np.inf, np.inf)
    assert bounds["MOMENT_Z"] == (-np.inf, 0.0)
    assert bounds["AIRFOIL_AREA"] == (0.0, np.inf)


def test_ffd_ikkt_active_set_keeps_violated_area_and_skips_satisfied_moment(
    tmp_path,
):
    cfg = _ikkt_config(tmp_path)
    names = _extract_constraint_names(cfg)
    active, records = _select_active_ikkt_constraints(
        cfg,
        names,
        current_values={
            "LIFT": 0.8063530125,
            "MOMENT_Z": 0.08417684449,
            "AIRFOIL_AREA": 0.0777347,
        },
        active_tol=1.0e-6,
        verbose=False,
    )
    by_name = {record["name"]: record for record in records}

    assert active == ["LIFT", "AIRFOIL_AREA"]
    assert by_name["LIFT"]["status"] == "active_equality"
    assert by_name["MOMENT_Z"]["status"] == "inactive"
    assert by_name["MOMENT_Z"]["c_value"] == pytest.approx(0.00782315551)
    assert by_name["AIRFOIL_AREA"]["status"] == "active_inequality"
    assert by_name["AIRFOIL_AREA"]["c_value"] == pytest.approx(-6.53e-5)


def test_ffd_ikkt_area_multiplier_is_not_clipped_to_zero(tmp_path):
    cfg = _ikkt_config(tmp_path)
    names = ["LIFT", "AIRFOIL_AREA"]
    bounds = _extract_constraint_signs(cfg, names)
    diagnostics = {}

    residual, multipliers = _compute_ikkt_residual_vector(
        g_obj=[0.5, 2.0],
        constraint_grads=[[1.0, 0.0], [0.0, 1.0]],
        lambda_bounds=bounds,
        strict=True,
        verbose=False,
        diagnostics=diagnostics,
    )

    assert multipliers == pytest.approx([0.5, 2.0])
    assert residual == pytest.approx([0.0, 0.0], abs=1.0e-12)
    assert diagnostics["status"] == "ok"
    assert diagnostics["constraint_matrix_rank"] == 2


def test_ffd_ikkt_active_tolerance_is_configurable_and_validated():
    cfg = SU2.io.Config(
        {
            "OPT_ITERATIONS": 5,
            "PROGRESSIVE_HH_IKKT_ACTIVE_TOL": 2.5e-5,
        }
    )
    assert get_progressive_hh_options(cfg)["ikkt_active_tol"] == pytest.approx(2.5e-5)

    cfg["PROGRESSIVE_HH_IKKT_ACTIVE_TOL"] = -1.0
    with pytest.raises(ValueError, match="IKKT_ACTIVE_TOL"):
        get_progressive_hh_options(cfg)
