import math

import pytest

import SU2
from SU2.opt.progressive_ffd_blending import (
    BEZIER,
    BSPLINE_UNIFORM,
    make_blending_spec,
)
from SU2.opt.progressive_ffd_core import get_progressive_ffd_options
from SU2.opt.progressive_ffd_envelope import (
    ADAPTIVE_CLEARANCE,
    FIXED_OFFSET,
    FFDClearanceSpec,
    FFDEnvelopeError,
    build_adaptive_outer_row,
)
from SU2.opt.progressive_ffd_prepare import (
    _build_prepare_request,
    _mesh_geometry,
    _write_adaptive_envelope_vtk,
)
from SU2.opt.progressive_ffd_split import (
    FFDBoxSplitError,
    build_single_surface_ffd_box,
    rewrite_dual_ffd_boxes_with_columns_and_reembed,
    rewrite_single_ffd_box_with_columns_and_reembed,
    split_bootstrap_ffd_box,
)
from SU2.opt.progressive_hh_core import get_progressive_hh_options
from tests.test_progressive_ffd_split import (
    HALF_LOWER_POINTS,
    HALF_UPPER_POINTS,
    SYMMETRIC_POINTS,
    _write_bootstrap_mesh,
)


COLUMNS = [0.0, 0.125, 0.25, 0.5, 0.75, 0.875, 1.0]


def _config(**overrides):
    values = {
        "PROGRESSIVE_HH": "YES",
        "PROGRESSIVE_PARAM_KIND": "FFD",
        "PROGRESSIVE_HH_NFINAL": 9,
        "PROGRESSIVE_HH_NLEVELS": 3,
        "PROGRESSIVE_HH_MAX_ITER_PER_LEVEL": 2,
        "PROGRESSIVE_HH_REFINEMENT": "UNIFORM",
        "PROGRESSIVE_HH_SPRING": "NO",
        "PROGRESSIVE_FFD_DOMAIN_MODE": "HALF_UPPER",
        "PROGRESSIVE_FFD_DV_KIND": "FFD_CONTROL_POINT_2D",
        "PROGRESSIVE_FFD_MARKER": "AIRFOIL",
        "PROGRESSIVE_FFD_AUTO_PREPARE": "YES",
        "PROGRESSIVE_FFD_PREPARE_ONLY": "NO",
        "PROGRESSIVE_FFD_BOOTSTRAP_TAG": "BOOTSTRAP_BOX",
        "PROGRESSIVE_FFD_UPPER_BOX_TAG": "UPPER_BOX",
        "PROGRESSIVE_FFD_LOWER_BOX_TAG": "LOWER_BOX",
        "PROGRESSIVE_FFD_INITIAL_COLUMNS": "( 0.125, 0.25, 0.5, 0.75, 0.875 )",
        "OPT_ITERATIONS": 4,
    }
    values.update(overrides)
    return SU2.io.Config(values)


def _opts(config):
    return get_progressive_ffd_options(config, get_progressive_hh_options(config))


def test_clearance_profile_values_and_validation():
    spec = FFDClearanceSpec()
    assert spec.clearance_chord(0.0) == pytest.approx(0.005)
    assert spec.clearance_chord(0.10) == pytest.approx(0.005)
    assert spec.clearance_chord(0.15) == pytest.approx(0.0075)
    assert spec.clearance_chord(0.20) == pytest.approx(0.01)
    assert spec.clearance_chord(1.0) == pytest.approx(0.01)

    with pytest.raises(FFDEnvelopeError, match="start < end"):
        FFDClearanceSpec(transition_start=0.2, transition_end=0.1)
    with pytest.raises(FFDEnvelopeError, match="positive"):
        FFDClearanceSpec(leading_chord=0.0)


@pytest.mark.parametrize(
    "blending,orders",
    [(BEZIER, (2, 2, 2)), (BSPLINE_UNIFORM, (4, 2, 2))],
)
@pytest.mark.parametrize("side", ["UPPER", "LOWER"])
def test_adaptive_envelope_contains_curved_surface(blending, orders, side):
    spec = make_blending_spec(blending, orders)
    clearance = FFDClearanceSpec()
    sign = 1.0 if side == "UPPER" else -1.0

    def surface_y(x):
        return sign * 0.08 * math.sqrt(max(0.0, float(x))) * (1.0 - float(x))

    result = build_adaptive_outer_row(
        columns=COLUMNS,
        inner_controls=[0.0] * len(COLUMNS),
        surface_y=surface_y,
        surface_x=[index / 256.0 for index in range(257)],
        x_le=0.0,
        x_te=1.0,
        chord=1.0,
        side=side,
        blending_spec=spec,
        clearance_spec=clearance,
    )

    assert result["min_clearance_margin"] >= -5.0e-12
    assert result["min_v"] >= -5.0e-12
    assert result["max_v"] <= 1.0 + 5.0e-12
    assert len(result["outer_controls"]) == len(COLUMNS)
    assert all(value > 0.0 for value in result["control_offsets"])


def test_envelope_config_is_opt_in_and_modes_are_mutually_exclusive():
    fixed = _opts(
        _config(
            PROGRESSIVE_FFD_UPPER_OFFSET_CHORD=0.02,
            PROGRESSIVE_FFD_LOWER_OFFSET_CHORD=0.03,
        )
    )
    assert fixed["ffd_envelope_mode"] == FIXED_OFFSET
    assert fixed["ffd_envelope_spec"] is None
    assert fixed["ffd_upper_offset_chord"] == pytest.approx(0.02)

    adaptive = _opts(
        _config(
            PROGRESSIVE_FFD_ENVELOPE_MODE=ADAPTIVE_CLEARANCE,
            PROGRESSIVE_FFD_CLEARANCE_LE_CHORD=0.005,
            PROGRESSIVE_FFD_CLEARANCE_TRANSITION_START=0.10,
            PROGRESSIVE_FFD_CLEARANCE_TRANSITION_END=0.20,
            PROGRESSIVE_FFD_CLEARANCE_TE_CHORD=0.01,
        )
    )
    assert adaptive["ffd_envelope_mode"] == ADAPTIVE_CLEARANCE
    assert adaptive["ffd_upper_offset_chord"] is None
    assert adaptive["ffd_envelope_spec"].clearance_chord(0.15) == pytest.approx(
        0.0075
    )

    with pytest.raises(ValueError, match="cannot be combined"):
        _opts(
            _config(
                PROGRESSIVE_FFD_ENVELOPE_MODE=ADAPTIVE_CLEARANCE,
                PROGRESSIVE_FFD_UPPER_OFFSET_CHORD=0.02,
            )
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        _opts(
            _config(
                PROGRESSIVE_FFD_CLEARANCE_LE_CHORD=0.005,
            )
        )


def test_prepare_request_preserves_fixed_schema_and_versions_adaptive(tmp_path):
    mesh_path = _write_bootstrap_mesh(
        tmp_path / "raw.su2",
        points=HALF_UPPER_POINTS,
        columns=COLUMNS,
        marker_closed=False,
    )
    geometry = _mesh_geometry(mesh_path, "AIRFOIL", domain_mode="HALF_UPPER")
    prepared = tmp_path / "prepared.su2"

    fixed_opts = _opts(
        _config(
            PROGRESSIVE_FFD_UPPER_OFFSET_CHORD=0.02,
            PROGRESSIVE_FFD_LOWER_OFFSET_CHORD=0.02,
        )
    )
    fixed_request = _build_prepare_request(mesh_path, prepared, geometry, fixed_opts)
    assert fixed_request["schema_version"] == 3
    assert fixed_request["upper_offset_chord"] == pytest.approx(0.02)
    assert "envelope_mode" not in fixed_request

    adaptive_opts = _opts(
        _config(PROGRESSIVE_FFD_ENVELOPE_MODE=ADAPTIVE_CLEARANCE)
    )
    adaptive_request = _build_prepare_request(
        mesh_path, prepared, geometry, adaptive_opts
    )
    assert adaptive_request["schema_version"] == 4
    assert adaptive_request["envelope_mode"] == ADAPTIVE_CLEARANCE
    assert adaptive_request["clearance_profile"] == FFDClearanceSpec().as_dict()
    assert "upper_offset_chord" not in adaptive_request


@pytest.mark.parametrize(
    "blending,orders",
    [(BEZIER, (2, 2, 2)), (BSPLINE_UNIFORM, (4, 2, 2))],
)
@pytest.mark.parametrize(
    "side,points,box_tag",
    [
        ("UPPER", HALF_UPPER_POINTS, "UPPER_BOX"),
        ("LOWER", HALF_LOWER_POINTS, "LOWER_BOX"),
    ],
)
def test_single_prepare_and_candidate_rewrite_use_adaptive_envelope(
    tmp_path, blending, orders, side, points, box_tag
):
    bootstrap = _write_bootstrap_mesh(
        tmp_path / "bootstrap.su2",
        points=points,
        columns=COLUMNS,
        marker_closed=False,
    )
    prepared = tmp_path / "prepared.su2"
    clearance = FFDClearanceSpec()
    summary = build_single_surface_ffd_box(
        bootstrap,
        prepared,
        bootstrap_tag="BOOTSTRAP_BOX",
        marker="AIRFOIL",
        side=side,
        envelope_spec=clearance,
        box_tag=box_tag,
        output_blending=blending,
        bspline_orders=orders,
    )
    assert summary["envelope_mode"] == ADAPTIVE_CLEARANCE
    assert summary["envelope"]["min_clearance_margin"] >= -5.0e-12

    candidate = tmp_path / "candidate.su2"
    candidate_columns = sorted(COLUMNS + [0.05])
    rewritten = rewrite_single_ffd_box_with_columns_and_reembed(
        prepared,
        candidate,
        marker="AIRFOIL",
        side=side,
        box_tag=box_tag,
        columns=candidate_columns,
        envelope_spec=clearance,
    )
    assert rewritten["envelope_mode"] == ADAPTIVE_CLEARANCE
    assert rewritten["envelope"]["min_clearance_margin"] >= -5.0e-12
    assert rewritten["envelope"]["max_v"] <= 1.0 + 5.0e-12


@pytest.mark.parametrize(
    "blending,orders",
    [(BEZIER, (2, 2, 2)), (BSPLINE_UNIFORM, (4, 2, 2))],
)
def test_dual_prepare_uses_mirrored_adaptive_envelopes(tmp_path, blending, orders):
    bootstrap = _write_bootstrap_mesh(
        tmp_path / "bootstrap.su2",
        points=SYMMETRIC_POINTS,
        columns=COLUMNS,
    )
    prepared = tmp_path / "prepared.su2"
    summary = split_bootstrap_ffd_box(
        bootstrap,
        prepared,
        bootstrap_tag="BOOTSTRAP_BOX",
        marker="AIRFOIL",
        envelope_spec=FFDClearanceSpec(),
        output_blending=blending,
        bspline_orders=orders,
    )
    assert summary["envelope_mode"] == ADAPTIVE_CLEARANCE
    assert summary["upper_envelope"]["min_clearance_margin"] >= -5.0e-12
    assert summary["lower_envelope"]["min_clearance_margin"] >= -5.0e-12
    assert summary["upper_envelope"]["max_v"] <= 1.0 + 5.0e-12
    assert summary["lower_envelope"]["max_v"] <= 1.0 + 5.0e-12


def test_prepare_writes_sampled_adaptive_envelope_vtk(tmp_path):
    bootstrap = _write_bootstrap_mesh(
        tmp_path / "bootstrap.su2",
        points=HALF_UPPER_POINTS,
        columns=COLUMNS,
        marker_closed=False,
    )
    geometry = _mesh_geometry(bootstrap, "AIRFOIL", domain_mode="HALF_UPPER")
    prepared = tmp_path / "prepared.su2"
    opts = _opts(
        _config(PROGRESSIVE_FFD_ENVELOPE_MODE=ADAPTIVE_CLEARANCE)
    )
    summary = build_single_surface_ffd_box(
        bootstrap,
        prepared,
        bootstrap_tag="BOOTSTRAP_BOX",
        marker="AIRFOIL",
        side="UPPER",
        envelope_spec=opts["ffd_envelope_spec"],
        box_tag="UPPER_BOX",
        output_blending=opts["ffd_blending"],
        bspline_orders=opts["ffd_bspline_orders"],
    )

    vtk_path = tmp_path / "ffd_envelope_curves.vtk"
    _write_adaptive_envelope_vtk(vtk_path, geometry, summary, opts)
    contents = vtk_path.read_text()
    assert "DATASET POLYDATA" in contents
    assert "LINES 5 " in contents
    assert "SCALARS component int 1" in contents


@pytest.mark.parametrize(
    "blending,orders",
    [(BEZIER, (2, 2, 2)), (BSPLINE_UNIFORM, (4, 2, 2))],
)
def test_dual_candidate_rewrite_uses_adaptive_envelope(
    tmp_path, blending, orders
):
    bootstrap = _write_bootstrap_mesh(
        tmp_path / "bootstrap.su2",
        points=SYMMETRIC_POINTS,
        columns=COLUMNS,
    )
    prepared = tmp_path / "prepared.su2"
    clearance = FFDClearanceSpec()
    split_bootstrap_ffd_box(
        bootstrap,
        prepared,
        bootstrap_tag="BOOTSTRAP_BOX",
        marker="AIRFOIL",
        envelope_spec=clearance,
        output_blending=blending,
        bspline_orders=orders,
    )

    candidate = tmp_path / "candidate.su2"
    candidate_columns = sorted(COLUMNS + [0.05])
    summary = rewrite_dual_ffd_boxes_with_columns_and_reembed(
        prepared,
        candidate,
        marker="AIRFOIL",
        upper_tag="UPPER_BOX",
        lower_tag="LOWER_BOX",
        upper_columns=candidate_columns,
        lower_columns=candidate_columns,
        envelope_spec=clearance,
    )
    assert summary["upper_envelope"]["min_clearance_margin"] >= -5.0e-12
    assert summary["lower_envelope"]["min_clearance_margin"] >= -5.0e-12
    assert summary["upper_envelope"]["max_v"] <= 1.0 + 5.0e-12
    assert summary["lower_envelope"]["max_v"] <= 1.0 + 5.0e-12


def test_split_api_rejects_mixed_fixed_and_adaptive_envelopes(tmp_path):
    bootstrap = _write_bootstrap_mesh(
        tmp_path / "bootstrap.su2",
        points=HALF_UPPER_POINTS,
        columns=COLUMNS,
        marker_closed=False,
    )
    with pytest.raises(FFDBoxSplitError, match="cannot combine"):
        build_single_surface_ffd_box(
            bootstrap,
            tmp_path / "prepared.su2",
            bootstrap_tag="BOOTSTRAP_BOX",
            marker="AIRFOIL",
            side="UPPER",
            offset_chord=0.02,
            envelope_spec=FFDClearanceSpec(),
            box_tag="UPPER_BOX",
        )
