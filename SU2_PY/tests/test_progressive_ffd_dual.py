import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import SU2
from SU2.opt.progressive_ffd_core import (
    FFDLevel,
    apply_post_opt_ffd_spring,
    build_ffd_mesh_columns,
    get_progressive_ffd_options,
    make_dual_ffd_definition,
    ordered_ffd_records,
    ordered_dual_ffd_records,
    refine_ffd_columns,
    select_ffd_candidates_by_nadd_mode,
    validate_ffd_mesh_blending,
)
from SU2.opt.progressive_ffd_levels import (
    build_ffd_spring_reallocated_level,
    build_next_ffd_level,
    build_initial_ffd_level,
    refresh_ffd_scoring_baseline,
    write_ffd_level_config,
)
from SU2.opt.progressive_ffd_prepare import (
    FFDPreparationError,
    _SMOKE_VISUALIZATION_FILENAMES,
    _build_prepare_request,
    _max_mesh_coordinate_difference,
    _mesh_geometry,
    _persist_smoke_artifacts,
    _validate_dual_mesh,
    _write_bootstrap_config,
)
from SU2.opt.progressive_ffd_split import (
    build_single_surface_ffd_box,
    read_single_ffd_box_spec,
)
from SU2.opt.progressive_ffd_projection import (
    _build_extended_ffd_dot_config,
    _compute_dual_bezier_exact_candidate_scores,
    _compute_dual_ffd_dot_candidate_scores,
    _dual_record_index,
    _exact_growth_ratio_insertion_target,
    _ffd_dot_artifact_directory,
    _filter_candidates_by_min_spacing,
    _ikkt_indicators_with_fixed_lambdas,
    _persist_selected_candidate_artifacts,
    _prepare_dual_projection_variant,
    _resolve_accepted_projection_mesh,
    _run_dual_projection_constraint_gradients,
)
from SU2.opt.progressive_hh_core import get_progressive_hh_options
from SU2.opt.progressive_hh_levels import append_selection_history_csv
from SU2.opt.progressive_hh_projection import _compute_ikkt_residual_vector
from SU2.opt.thickness_constraint import (
    ThicknessConstraint,
    _section_measure_from_segments,
)
from tests.test_progressive_ffd_split import (
    HALF_UPPER_POINTS,
    SYMMETRIC_POINTS,
    _split,
    _write_bootstrap_mesh,
)


def _dual_config(**overrides):
    values = {
        "PROGRESSIVE_HH": "YES",
        "PROGRESSIVE_PARAM_KIND": "FFD",
        "PROGRESSIVE_HH_NFINAL": 8,
        "PROGRESSIVE_HH_NLEVELS": 3,
        "PROGRESSIVE_HH_MAX_ITER_PER_LEVEL": 2,
        "PROGRESSIVE_HH_REFINEMENT": "UNIFORM",
        "PROGRESSIVE_HH_SPRING": "NO",
        "PROGRESSIVE_FFD_DUAL_BOX": "YES",
        "PROGRESSIVE_FFD_AUTO_PREPARE": "NO",
        "PROGRESSIVE_FFD_PREPARE_ONLY": "NO",
        "PROGRESSIVE_FFD_DV_KIND": "FFD_CONTROL_POINT_2D",
        "PROGRESSIVE_FFD_MARKER": "AIRFOIL",
        "PROGRESSIVE_FFD_DOMAIN_MODE": "FULL",
        "PROGRESSIVE_FFD_UPPER_BOX_TAG": "UPPER_BOX",
        "PROGRESSIVE_FFD_LOWER_BOX_TAG": "LOWER_BOX",
        "PROGRESSIVE_FFD_BOOTSTRAP_TAG": "BOOTSTRAP_BOX",
        "PROGRESSIVE_FFD_BOOTSTRAP_Y_PADDING_CHORD": 0.04,
        "PROGRESSIVE_FFD_UPPER_OFFSET_CHORD": 0.04,
        "PROGRESSIVE_FFD_LOWER_OFFSET_CHORD": 0.06,
        "PROGRESSIVE_FFD_REFINEMENT_COUPLING": "INDEPENDENT",
        "PROGRESSIVE_FFD_INITIAL_COLUMNS": "( 0.25, 0.5, 0.75 )",
        "OPT_ITERATIONS": 4,
    }
    values.update(overrides)
    return SU2.io.Config(values)


def _dual_opts(config):
    hh_opts = get_progressive_hh_options(config)
    return get_progressive_ffd_options(config, hh_opts)


def _half_upper_config(**overrides):
    values = dict(_dual_config())
    for key in (
        "PROGRESSIVE_FFD_DUAL_BOX",
        "PROGRESSIVE_FFD_AUTO_PREPARE",
        "PROGRESSIVE_FFD_PREPARE_ONLY",
    ):
        values.pop(key, None)
    values.update(
        {
            "PROGRESSIVE_HH_NFINAL": 4,
            "PROGRESSIVE_FFD_DOMAIN_MODE": "HALF_UPPER",
        }
    )
    values.update(overrides)
    return SU2.io.Config(values)


def test_bootstrap_config_writes_all_eight_ffd_corner_points(tmp_path):
    config_path = tmp_path / "bootstrap.cfg"
    _write_bootstrap_config(
        str(config_path),
        mesh_in="mesh.su2",
        mesh_out_base="mesh_out",
        marker="AIRFOIL",
        other_markers=[],
        symmetry_markers=[],
        bootstrap_tag="AIRFOIL_BOX",
        x_le=0.0,
        x_te=1.0,
        y_bottom=-0.1,
        y_top=0.1,
    )

    definition = next(
        line.strip()
        for line in config_path.read_text().splitlines()
        if line.startswith("FFD_DEFINITION=")
    )
    fields = [
        token.strip()
        for token in definition.split("=", 1)[1].strip().strip("()").split(",")
    ]

    assert fields[0] == "AIRFOIL_BOX"
    assert len(fields) == 25
    assert [float(value) for value in fields[13:]] == pytest.approx([0.0] * 12)


def test_dual_nfinal_counts_upper_and_lower_dvs():
    config = _dual_config(PROGRESSIVE_HH_NFINAL=5)
    with pytest.raises(ValueError, match="initial HH NDV.*6"):
        get_progressive_hh_options(config)

    config = _dual_config(PROGRESSIVE_HH_NFINAL=8)
    opts = _dual_opts(config)
    assert opts["nfinal"] == 8
    assert opts["ffd_initial_columns"] == pytest.approx([0.25, 0.5, 0.75])


def test_offset_endpoint_n0_counts_total_columns_per_side():
    config = _dual_config(
        PROGRESSIVE_FFD_OPTIMIZE_OFFSET_ENDPOINTS="YES",
        PROGRESSIVE_HH_N0=7,
        PROGRESSIVE_HH_NFINAL=14,
    )
    config.pop("PROGRESSIVE_FFD_INITIAL_COLUMNS")

    opts = _dual_opts(config)
    expected = [0.0] + [index / 6.0 for index in range(1, 6)] + [1.0]
    assert opts["ffd_optimize_offset_endpoints"] is True
    assert opts["ffd_active_include_bounds"] is True
    assert opts["ffd_initial_interior_columns"] == pytest.approx(expected[1:-1])
    assert opts["ffd_initial_columns"] == pytest.approx(expected)

    level = build_initial_ffd_level(
        SU2.io.Config(dict(config, MESH_FILENAME="prepared.su2")),
        opts,
    )
    assert level.ndv == 14
    assert level.upper_columns == pytest.approx(expected)
    assert level.lower_columns == pytest.approx(expected)


def test_te_only_offset_endpoint_n0_still_counts_total_columns_per_side():
    config = _dual_config(
        PROGRESSIVE_FFD_OPTIMIZE_OFFSET_ENDPOINTS="YES",
        PROGRESSIVE_FFD_OPTIMIZE_LE_OFFSET_ENDPOINTS="NO",
        PROGRESSIVE_FFD_OPTIMIZE_TE_OFFSET_ENDPOINTS="YES",
        PROGRESSIVE_HH_N0=7,
        PROGRESSIVE_HH_NFINAL=14,
    )
    config.pop("PROGRESSIVE_FFD_INITIAL_COLUMNS")

    opts = _dual_opts(config)
    expected_interior = [index / 7.0 for index in range(1, 7)]
    expected = expected_interior + [1.0]
    assert opts["ffd_optimize_offset_endpoints"] is False
    assert opts["ffd_optimize_le_offset_endpoints"] is False
    assert opts["ffd_optimize_te_offset_endpoints"] is True
    assert opts["ffd_active_include_bounds"] is True
    assert opts["ffd_initial_interior_columns"] == pytest.approx(
        expected_interior
    )
    assert opts["ffd_initial_columns"] == pytest.approx(expected)

    level = build_initial_ffd_level(
        SU2.io.Config(dict(config, MESH_FILENAME="prepared.su2")),
        opts,
    )
    assert level.ndv == 14
    assert level.upper_columns == pytest.approx(expected)
    assert level.lower_columns == pytest.approx(expected)


def test_te_only_offset_endpoint_explicit_list_adds_only_te():
    interior = [0.1, 0.25, 0.5, 0.75, 0.9]
    config = _dual_config(
        PROGRESSIVE_FFD_OPTIMIZE_OFFSET_ENDPOINTS="YES",
        PROGRESSIVE_FFD_OPTIMIZE_LE_OFFSET_ENDPOINTS="NO",
        PROGRESSIVE_FFD_OPTIMIZE_TE_OFFSET_ENDPOINTS="YES",
        PROGRESSIVE_FFD_INITIAL_COLUMNS="( 0.1, 0.25, 0.5, 0.75, 0.9 )",
        PROGRESSIVE_HH_NFINAL=12,
    )
    opts = _dual_opts(config)
    assert opts["ffd_initial_interior_columns"] == pytest.approx(interior)
    assert opts["ffd_initial_columns"] == pytest.approx(interior + [1.0])

    level = build_initial_ffd_level(
        SU2.io.Config(dict(config, MESH_FILENAME="prepared.su2")),
        opts,
    )
    assert level.ndv == 12
    assert level.upper_columns == pytest.approx(interior + [1.0])
    assert level.lower_columns == pytest.approx(interior + [1.0])


def test_offset_endpoint_explicit_list_contains_only_interior_columns():
    interior = [0.1, 0.25, 0.5, 0.75, 0.9]
    config = _dual_config(
        PROGRESSIVE_FFD_OPTIMIZE_OFFSET_ENDPOINTS="YES",
        PROGRESSIVE_FFD_INITIAL_COLUMNS="( 0.1, 0.25, 0.5, 0.75, 0.9 )",
        PROGRESSIVE_HH_NFINAL=14,
    )
    opts = _dual_opts(config)
    assert opts["ffd_initial_interior_columns"] == pytest.approx(interior)
    assert opts["ffd_initial_columns"] == pytest.approx(
        [0.0] + interior + [1.0]
    )

    with pytest.raises(ValueError, match=r"must satisfy 0.0 < x < 1.0"):
        _dual_opts(
            _dual_config(
                PROGRESSIVE_FFD_OPTIMIZE_OFFSET_ENDPOINTS="YES",
                PROGRESSIVE_FFD_INITIAL_COLUMNS="( 0.0, 0.25, 0.5 )",
                PROGRESSIVE_HH_NFINAL=10,
            )
        )


def test_offset_endpoint_initial_count_validates_nfinal_and_half_topology():
    full = _dual_config(
        PROGRESSIVE_FFD_OPTIMIZE_OFFSET_ENDPOINTS="YES",
        PROGRESSIVE_HH_N0=7,
        PROGRESSIVE_HH_NFINAL=13,
    )
    full.pop("PROGRESSIVE_FFD_INITIAL_COLUMNS")
    with pytest.raises(ValueError, match="initial HH NDV.*14"):
        get_progressive_hh_options(full)

    half = _half_upper_config(
        PROGRESSIVE_FFD_OPTIMIZE_OFFSET_ENDPOINTS="YES",
        PROGRESSIVE_HH_N0=7,
        PROGRESSIVE_HH_NFINAL=7,
    )
    half.pop("PROGRESSIVE_FFD_INITIAL_COLUMNS")
    opts = _dual_opts(half)
    level = build_initial_ffd_level(
        SU2.io.Config(dict(half, MESH_FILENAME="prepared.su2")),
        opts,
    )
    assert level.ndv == 7
    assert level.columns[0] == pytest.approx(0.0)
    assert level.columns[-1] == pytest.approx(1.0)


def test_offset_endpoints_keep_growth_ratio_based_on_total_ndv():
    level = SimpleNamespace(ndv=14)
    candidates = [
        {"side": "UPPER", "interval_id": index} for index in range(20)
    ]
    target, interval_count = _exact_growth_ratio_insertion_target(
        level,
        candidates,
        {"growth_ratio": 2.0, "nfinal": 40},
    )
    assert target == 14
    assert interval_count == 20


def test_offset_endpoint_definition_targets_only_outer_rows():
    columns = [0.0, 0.2, 0.5, 0.8, 1.0]
    records = ordered_dual_ffd_records(columns, columns)
    mapping = {value: index for index, value in enumerate(columns)}
    definition = make_dual_ffd_definition(
        records,
        {
            "ffd_dual_box": True,
            "ffd_active_sides": ("UPPER", "LOWER"),
            "ffd_marker": "AIRFOIL",
            "ffd_upper_box_tag": "UPPER_BOX",
            "ffd_lower_box_tag": "LOWER_BOX",
            "scale": 1.0,
        },
        {"UPPER": mapping, "LOWER": mapping},
    )
    assert definition["PARAM"][0] == [0, 1, 0.0, 1.0]
    assert definition["PARAM"][4] == [4, 1, 0.0, 1.0]
    assert definition["PARAM"][5] == [0, 0, 0.0, -1.0]
    assert definition["PARAM"][-1] == [4, 0, 0.0, -1.0]


def test_offset_endpoint_active_columns_reuse_mesh_boundaries(tmp_path):
    columns = [0.0, 0.2, 0.5, 0.8, 1.0]
    bootstrap = _write_bootstrap_mesh(
        tmp_path / "bootstrap.su2",
        columns=columns,
    )
    prepared = tmp_path / "prepared.su2"
    _split(bootstrap, prepared)

    mesh_columns, active_columns = build_ffd_mesh_columns(
        str(prepared),
        "UPPER_BOX",
        columns,
        {
            "ffd_domain_mode": "FULL",
            "ffd_active_xmin": 0.0,
            "ffd_active_xmax": 1.0,
            "ffd_optimize_offset_endpoints": True,
        },
    )
    assert mesh_columns == pytest.approx(columns)
    assert active_columns == pytest.approx(columns)


def test_te_only_active_columns_reuse_te_and_reject_disabled_le(tmp_path):
    geometric_columns = [0.0, 0.2, 0.5, 0.8, 1.0]
    active_columns = [0.2, 0.5, 0.8, 1.0]
    bootstrap = _write_bootstrap_mesh(
        tmp_path / "bootstrap.su2",
        columns=geometric_columns,
    )
    prepared = tmp_path / "prepared.su2"
    _split(bootstrap, prepared)
    opts = {
        "ffd_domain_mode": "FULL",
        "ffd_active_xmin": 0.0,
        "ffd_active_xmax": 1.0,
        "ffd_optimize_offset_endpoints": False,
        "ffd_optimize_le_offset_endpoints": False,
        "ffd_optimize_te_offset_endpoints": True,
    }

    mesh_columns, validated = build_ffd_mesh_columns(
        str(prepared),
        "UPPER_BOX",
        active_columns,
        opts,
    )
    assert mesh_columns == pytest.approx(geometric_columns)
    assert validated == pytest.approx(active_columns)

    with pytest.raises(
        ValueError,
        match="PROGRESSIVE_FFD_OPTIMIZE_LE_OFFSET_ENDPOINTS=YES",
    ):
        build_ffd_mesh_columns(
            str(prepared),
            "UPPER_BOX",
            geometric_columns,
            opts,
        )


def test_offset_endpoint_level_config_contains_fourteen_outer_row_dvs(
    tmp_path,
    monkeypatch,
):
    columns = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    bootstrap = _write_bootstrap_mesh(
        tmp_path / "bootstrap.su2",
        columns=columns,
    )
    prepared = tmp_path / "prepared.su2"
    _split(bootstrap, prepared)
    config = _dual_config(
        MESH_FILENAME=str(prepared),
        MESH_OUT_FILENAME="mesh_out",
        PROGRESSIVE_FFD_OPTIMIZE_OFFSET_ENDPOINTS="YES",
        PROGRESSIVE_FFD_INITIAL_COLUMNS="( 0.1, 0.25, 0.5, 0.75, 0.9 )",
        PROGRESSIVE_HH_NFINAL=14,
        FFD_CONTINUITY="2ND_DERIVATIVE",
        FFD_FIX_I="( 0, 1 )",
    )
    config._filename = str(tmp_path / "Config_FFD.cfg")
    opts = _dual_opts(config)
    level = build_initial_ffd_level(config, opts)

    monkeypatch.chdir(tmp_path)
    cfg_path = Path(write_ffd_level_config(config, level, opts))
    cfg_text = cfg_path.read_text()
    assert cfg_text.count("( 19") == 14
    assert cfg_text.count("UPPER_BOX") == 7
    assert cfg_text.count("LOWER_BOX") == 7
    assert "FFD_CONTINUITY= USER_INPUT" in cfg_text
    assert "FFD_FIX_I" not in cfg_text
    assert "PROGRESSIVE_FFD_OPTIMIZE_OFFSET_ENDPOINTS" not in cfg_text

    mesh_text = (tmp_path / "LEVEL_0" / "ffd_level0.su2").read_text()
    assert mesh_text.count("FFD_DEGREE_I= 6") == 2


def test_te_only_endpoint_level_config_contains_twelve_outer_row_dvs(
    tmp_path,
    monkeypatch,
):
    geometric_columns = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    bootstrap = _write_bootstrap_mesh(
        tmp_path / "bootstrap.su2",
        columns=geometric_columns,
    )
    prepared = tmp_path / "prepared.su2"
    _split(bootstrap, prepared)
    config = _dual_config(
        MESH_FILENAME=str(prepared),
        MESH_OUT_FILENAME="mesh_out",
        PROGRESSIVE_FFD_OPTIMIZE_OFFSET_ENDPOINTS="YES",
        PROGRESSIVE_FFD_OPTIMIZE_LE_OFFSET_ENDPOINTS="NO",
        PROGRESSIVE_FFD_OPTIMIZE_TE_OFFSET_ENDPOINTS="YES",
        PROGRESSIVE_FFD_INITIAL_COLUMNS="( 0.1, 0.25, 0.5, 0.75, 0.9 )",
        PROGRESSIVE_HH_NFINAL=12,
        FFD_CONTINUITY="2ND_DERIVATIVE",
        FFD_FIX_I="( 0, 1 )",
    )
    config._filename = str(tmp_path / "Config_FFD.cfg")
    opts = _dual_opts(config)
    level = build_initial_ffd_level(config, opts)
    assert level.upper_columns == pytest.approx(geometric_columns[1:])
    assert level.lower_columns == pytest.approx(geometric_columns[1:])

    monkeypatch.chdir(tmp_path)
    cfg_path = Path(write_ffd_level_config(config, level, opts))
    cfg_text = cfg_path.read_text()
    assert cfg_text.count("( 19") == 12
    assert cfg_text.count("UPPER_BOX") == 6
    assert cfg_text.count("LOWER_BOX") == 6
    assert "FFD_CONTINUITY= USER_INPUT" in cfg_text
    assert "FFD_FIX_I" not in cfg_text
    assert "PROGRESSIVE_FFD_OPTIMIZE_OFFSET_ENDPOINTS" not in cfg_text
    assert "PROGRESSIVE_FFD_OPTIMIZE_LE_OFFSET_ENDPOINTS" not in cfg_text
    assert "PROGRESSIVE_FFD_OPTIMIZE_TE_OFFSET_ENDPOINTS" not in cfg_text


def test_dual_bspline_options_are_native_and_validated():
    config = _dual_config(
        FFD_BLENDING="BSPLINE_UNIFORM",
        FFD_BSPLINE_ORDER="( 4, 2, 2 )",
    )
    opts = _dual_opts(config)
    assert opts["ffd_blending"] == "BSPLINE_UNIFORM"
    assert opts["ffd_bspline_orders"] == (4, 2, 2)

    with pytest.raises(ValueError, match="exceeds control-point count"):
        _dual_opts(
            _dual_config(
                FFD_BLENDING="BSPLINE_UNIFORM",
                FFD_BSPLINE_ORDER="( 6, 2, 2 )",
            )
        )
    half_opts = _dual_opts(
        _half_upper_config(
            FFD_BLENDING="BSPLINE_UNIFORM",
            FFD_BSPLINE_ORDER="( 3, 2, 2 )",
        )
    )
    assert half_opts["ffd_dual_box"] is False
    assert half_opts["ffd_active_sides"] == ("UPPER",)
    assert half_opts["ffd_bspline_orders"] == (3, 2, 2)


def test_mesh_blending_metadata_must_match_requested_options():
    bspline_opts = _dual_opts(
        _dual_config(
            FFD_BLENDING="BSPLINE_UNIFORM",
            FFD_BSPLINE_ORDER="( 4, 2, 2 )",
        )
    )
    validate_ffd_mesh_blending(
        {"blending": "BSPLINE_UNIFORM", "bspline_orders": [4, 2, 2]},
        bspline_opts,
    )
    with pytest.raises(RuntimeError, match="blending mismatch"):
        validate_ffd_mesh_blending(
            {"blending": "BEZIER", "bspline_orders": [2, 2, 2]},
            bspline_opts,
        )
    with pytest.raises(RuntimeError, match="order mismatch"):
        validate_ffd_mesh_blending(
            {"blending": "BSPLINE_UNIFORM", "bspline_orders": [3, 2, 2]},
            bspline_opts,
        )

    bezier_opts = _dual_opts(
        _dual_config(
            FFD_BLENDING="BEZIER",
            FFD_BSPLINE_ORDER="( 4, 2, 2 )",
        )
    )
    validate_ffd_mesh_blending(
        {"blending": "BEZIER", "bspline_orders": [2, 2, 2]},
        bezier_opts,
    )


@pytest.mark.parametrize(
    "real_values",
    [
        {"MATH_PROBLEM": "DISCRETE_ADJOINT"},
        {
            "MATH_PROBLEM": "DISCRETE_ADJOINT",
            "FFD_BLENDING": "BEZIER",
            "FFD_BSPLINE_ORDER": "2, 2, 2",
        },
    ],
)
def test_extended_dot_config_forces_requested_blending(real_values):
    opts = _dual_opts(
        _dual_config(
            FFD_BLENDING="BSPLINE_UNIFORM",
            FFD_BSPLINE_ORDER="( 4, 2, 2 )",
        )
    )
    cfg_dot = _build_extended_ffd_dot_config(
        SU2.io.Config({"NUMBER_PART": 1}),
        SU2.io.Config(real_values),
        "candidate.su2",
        [("UPPER", 0.25), ("LOWER", 0.25)],
        opts,
        {"UPPER": {0.25: 1}, "LOWER": {0.25: 1}},
    )
    assert cfg_dot["FFD_BLENDING"] == "BSPLINE_UNIFORM"
    assert cfg_dot["FFD_BSPLINE_ORDER"] == "4, 2, 2"


def test_half_lower_dot_definition_is_outward_negative():
    opts = _dual_opts(
        _half_upper_config(PROGRESSIVE_FFD_DOMAIN_MODE="HALF_LOWER")
    )
    cfg_dot = _build_extended_ffd_dot_config(
        SU2.io.Config({"NUMBER_PART": 1}),
        SU2.io.Config({"MATH_PROBLEM": "DISCRETE_ADJOINT"}),
        "candidate.su2",
        [("LOWER", 0.5)],
        opts,
        {"LOWER": {0.5: 3}},
    )
    assert cfg_dot["DEFINITION_DV"]["FFDTAG"] == ["LOWER_BOX"]
    assert cfg_dot["DEFINITION_DV"]["PARAM"] == [[3, 0, 0.0, -1.0]]
    assert cfg_dot["FFD_CONTINUITY"] == "USER_INPUT"


def test_prepared_dual_mesh_rejects_cfg_blending_mismatch(tmp_path):
    bootstrap = _write_bootstrap_mesh(tmp_path / "bootstrap.su2")
    dual = tmp_path / "dual_bezier.su2"
    _split(bootstrap, dual)
    config = _dual_config(
        MESH_FILENAME=str(dual),
        FFD_BLENDING="BSPLINE_UNIFORM",
        FFD_BSPLINE_ORDER="( 4, 2, 2 )",
    )
    opts = _dual_opts(config)
    geometry = _mesh_geometry(str(dual), "AIRFOIL")
    with pytest.raises(FFDPreparationError, match="blending mismatch"):
        _validate_dual_mesh(str(dual), geometry, opts)


def test_prepare_cache_request_changes_with_bspline_order(tmp_path):
    raw_mesh = _write_bootstrap_mesh(tmp_path / "raw.su2")
    geometry = _mesh_geometry(str(raw_mesh), "AIRFOIL")
    opts_o4 = _dual_opts(
        _dual_config(
            FFD_BLENDING="BSPLINE_UNIFORM",
            FFD_BSPLINE_ORDER="( 4, 2, 2 )",
        )
    )
    opts_o3 = _dual_opts(
        _dual_config(
            FFD_BLENDING="BSPLINE_UNIFORM",
            FFD_BSPLINE_ORDER="( 3, 2, 2 )",
        )
    )
    request_o4 = _build_prepare_request(
        str(raw_mesh),
        str(tmp_path / "prepared.su2"),
        geometry,
        opts_o4,
    )
    request_o3 = _build_prepare_request(
        str(raw_mesh),
        str(tmp_path / "prepared.su2"),
        geometry,
        opts_o3,
    )
    assert request_o4["bspline_orders"] == [4, 2, 2]
    assert request_o3["bspline_orders"] == [3, 2, 2]
    assert request_o4 != request_o3


def test_prepare_cache_request_changes_with_offset_endpoint_mode(tmp_path):
    raw_mesh = _write_bootstrap_mesh(tmp_path / "raw.su2")
    geometry = _mesh_geometry(str(raw_mesh), "AIRFOIL")
    disabled = _dual_opts(_dual_config())
    enabled = _dual_opts(
        _dual_config(
            PROGRESSIVE_FFD_OPTIMIZE_OFFSET_ENDPOINTS="YES",
            PROGRESSIVE_FFD_INITIAL_COLUMNS="( 0.25, 0.5, 0.75 )",
            PROGRESSIVE_HH_NFINAL=10,
        )
    )
    te_only = _dual_opts(
        _dual_config(
            PROGRESSIVE_FFD_OPTIMIZE_OFFSET_ENDPOINTS="YES",
            PROGRESSIVE_FFD_OPTIMIZE_LE_OFFSET_ENDPOINTS="NO",
            PROGRESSIVE_FFD_OPTIMIZE_TE_OFFSET_ENDPOINTS="YES",
            PROGRESSIVE_FFD_INITIAL_COLUMNS="( 0.25, 0.5, 0.75 )",
            PROGRESSIVE_HH_NFINAL=8,
        )
    )
    disabled_request = _build_prepare_request(
        str(raw_mesh), str(tmp_path / "prepared.su2"), geometry, disabled
    )
    enabled_request = _build_prepare_request(
        str(raw_mesh), str(tmp_path / "prepared.su2"), geometry, enabled
    )
    te_only_request = _build_prepare_request(
        str(raw_mesh), str(tmp_path / "prepared.su2"), geometry, te_only
    )
    assert disabled_request["schema_version"] == 3
    assert disabled_request["optimize_offset_endpoints"] is False
    assert disabled_request["optimize_le_offset_endpoints"] is False
    assert disabled_request["optimize_te_offset_endpoints"] is False
    assert enabled_request["optimize_offset_endpoints"] is True
    assert enabled_request["optimize_le_offset_endpoints"] is True
    assert enabled_request["optimize_te_offset_endpoints"] is True
    assert te_only_request["optimize_offset_endpoints"] is False
    assert te_only_request["optimize_le_offset_endpoints"] is False
    assert te_only_request["optimize_te_offset_endpoints"] is True
    assert enabled_request["initial_interior_columns"] == pytest.approx(
        [0.25, 0.5, 0.75]
    )
    assert enabled_request != disabled_request
    assert te_only_request != enabled_request
    assert te_only_request != disabled_request


def test_domain_mode_rejects_conflicting_legacy_topology_options():
    config = _dual_config(PROGRESSIVE_FFD_CONTROL_ROW=1)
    hh_opts = get_progressive_hh_options(config)
    with pytest.raises(ValueError, match="CONTROL_ROW conflicts"):
        get_progressive_ffd_options(config, hh_opts)

    config = _dual_config(PROGRESSIVE_FFD_DUAL_BOX="NO")
    hh_opts = get_progressive_hh_options(config)
    with pytest.raises(ValueError, match="DUAL_BOX conflicts"):
        get_progressive_ffd_options(config, hh_opts)

    config = _half_upper_config(PROGRESSIVE_FFD_DUAL_BOX="NO")
    hh_opts = get_progressive_hh_options(config)
    with pytest.warns(FutureWarning, match="DUAL_BOX is deprecated"):
        opts = get_progressive_ffd_options(config, hh_opts)
    assert opts["ffd_active_sides"] == ("UPPER",)


@pytest.mark.parametrize(
    "overrides,match",
    [
        (
            {"PROGRESSIVE_HH_SPRING_TIMING": "PRE_REFINE"},
            "PRE_REFINE is disabled",
        ),
        (
            {"PROGRESSIVE_HH_SPRING_SCORE_MODE": "INDICATOR"},
            "indicator-based spring is disabled",
        ),
    ],
)
def test_unified_ffd_spring_rejects_legacy_indicator_modes(overrides, match):
    config = _half_upper_config(PROGRESSIVE_HH_SPRING="YES", **overrides)
    hh_opts = get_progressive_hh_options(config)
    with pytest.raises(NotImplementedError, match=match):
        get_progressive_ffd_options(config, hh_opts)


def test_dual_level_and_definition_use_stable_outward_ordering():
    level = FFDLevel(
        level_id=0,
        columns=[0.2, 0.6],
        upper_columns=[0.2, 0.6],
        lower_columns=[0.3, 0.5, 0.8],
        dual_box=True,
        workdir="LEVEL_0",
        config_filename="config.cfg",
        project_filename="project.pkl",
        active_xmin=0.0,
        active_xmax=1.0,
    )
    assert level.ndv == 5
    assert level.dv_records == [
        ("UPPER", 0.2),
        ("UPPER", 0.6),
        ("LOWER", 0.3),
        ("LOWER", 0.5),
        ("LOWER", 0.8),
    ]

    config = _dual_config()
    opts = _dual_opts(config)
    definition = make_dual_ffd_definition(
        level.dv_records,
        opts,
        {
            "UPPER": {0.0: 0, 0.2: 1, 0.6: 2, 1.0: 3},
            "LOWER": {0.0: 0, 0.3: 1, 0.5: 2, 0.8: 3, 1.0: 4},
        },
    )
    assert definition["FFDTAG"] == [
        "UPPER_BOX",
        "UPPER_BOX",
        "LOWER_BOX",
        "LOWER_BOX",
        "LOWER_BOX",
    ]
    assert definition["PARAM"][:2] == [[1, 1, 0.0, 1.0], [2, 1, 0.0, 1.0]]
    assert definition["PARAM"][2:] == [
        [1, 0, 0.0, -1.0],
        [2, 0, 0.0, -1.0],
        [3, 0, 0.0, -1.0],
    ]


def test_score_batch_spacing_is_side_local_and_nfinal_is_exact():
    candidates = [
        {"side": "UPPER", "x": 0.4, "indicator": 10.0},
        {"side": "LOWER", "x": 0.41, "indicator": 9.0},
        {"side": "UPPER", "x": 0.45, "indicator": 8.0},
    ]
    opts = {
        "nfinal": 6,
        "nadd_mode": "SCORE_BATCH",
        "batch_size_max": 4,
        "batch_score_rel_tol": 0.0,
        "batch_min_separation": 0.1,
        "batch_max_per_side": None,
        "min_center_spacing": 0.0,
        "ffd_dual_box": True,
        "ffd_active_xmin": 0.0,
        "ffd_active_xmax": 1.0,
    }
    selected = select_ffd_candidates_by_nadd_mode(
        candidates,
        current_ndv=4,
        opts=opts,
        active_centers_by_side={"UPPER": [0.2], "LOWER": [0.2]},
    )
    assert [(item["side"], item["x"]) for item in selected] == [
        ("UPPER", 0.4),
        ("LOWER", 0.41),
    ]


def test_exact_growth_ratio_target_is_fixed_once_and_capped_by_nfinal():
    level = FFDLevel(
        level_id=0,
        columns=[0.25, 0.75],
        upper_columns=[0.25, 0.75],
        lower_columns=[0.25, 0.75],
        dual_box=True,
        workdir="LEVEL_0",
        config_filename="config.cfg",
        project_filename="project.pkl",
    )
    raw_candidates = [
        {"side": side, "interval_id": interval_id}
        for side in ("UPPER", "LOWER")
        for interval_id in range(3)
    ]

    target, interval_count = _exact_growth_ratio_insertion_target(
        level,
        raw_candidates,
        {"growth_ratio": 1.5},
    )
    assert interval_count == 6
    assert target == 2

    uncapped_by_initial_intervals, _ = _exact_growth_ratio_insertion_target(
        level,
        raw_candidates,
        {"growth_ratio": 3.0},
    )
    assert uncapped_by_initial_intervals == 8

    capped, _ = _exact_growth_ratio_insertion_target(
        level,
        raw_candidates,
        {"growth_ratio": 2.0, "nfinal": 5},
    )
    assert capped == 1


def test_uniform_dual_refinement_uses_global_widest_interval_upper_tie_first():
    level = FFDLevel(
        level_id=0,
        columns=[0.25, 0.75],
        upper_columns=[0.25, 0.75],
        lower_columns=[0.25, 0.75],
        dual_box=True,
        workdir="LEVEL_0",
        config_filename="config.cfg",
        project_filename="project.pkl",
        active_xmin=0.0,
        active_xmax=1.0,
    )
    opts = {
        "refinement": "UNIFORM",
        "nfinal": 5,
        "trigger": "MAX_ITER",
        "ffd_dual_box": True,
    }
    upper, lower = refine_ffd_columns(level, {}, opts)
    assert upper == pytest.approx([0.25, 0.5, 0.75])
    assert lower == pytest.approx([0.25, 0.75])
    assert len(upper) + len(lower) == 5


def test_post_opt_spring_redistributes_full_sides_independently():
    level = FFDLevel(
        level_id=1,
        columns=[0.2, 0.5, 0.8],
        upper_columns=[0.2, 0.5, 0.8],
        lower_columns=[0.2, 0.5, 0.8],
        dual_box=True,
        workdir="LEVEL_1",
        config_filename="config.cfg",
        project_filename="project.pkl",
        active_xmin=0.0,
        active_xmax=1.0,
    )
    result = {"dv_values": [4.0, 1.0, 0.2, 0.2, 1.0, 4.0]}
    redistributed = apply_post_opt_ffd_spring(
        level,
        result,
        {
            "ffd_domain_mode": "FULL",
            "ffd_active_xmin": 0.0,
            "ffd_active_xmax": 1.0,
            "min_center_spacing": 0.1,
            "spring_A": 20.0,
        },
    )

    assert set(redistributed) == {"UPPER", "LOWER"}
    assert len(redistributed["UPPER"]) == len(level.upper_columns)
    assert len(redistributed["LOWER"]) == len(level.lower_columns)
    assert redistributed["UPPER"] != pytest.approx(redistributed["LOWER"])
    for columns in redistributed.values():
        extended = [0.0] + columns + [1.0]
        assert min(b - a for a, b in zip(extended[:-1], extended[1:])) >= 0.1 - 1e-10
    assert result["spring_ffd_coeff_abs_by_side"] == {
        "UPPER": [4.0, 1.0, 0.2],
        "LOWER": [0.2, 1.0, 4.0],
    }


def test_offset_endpoint_spring_keeps_exact_boundary_anchors():
    columns = [0.0, 0.2, 0.5, 0.8, 1.0]
    level = FFDLevel(
        level_id=1,
        columns=columns,
        upper_columns=columns,
        lower_columns=columns,
        dual_box=True,
        workdir="LEVEL_1",
        config_filename="config.cfg",
        project_filename="project.pkl",
        active_xmin=0.0,
        active_xmax=1.0,
        active_include_bounds=True,
    )
    result = {
        "dv_values": [4.0, 2.0, 1.0, 0.5, 0.1, 0.1, 0.5, 1.0, 2.0, 4.0]
    }
    redistributed = apply_post_opt_ffd_spring(
        level,
        result,
        {
            "ffd_domain_mode": "FULL",
            "ffd_active_xmin": 0.0,
            "ffd_active_xmax": 1.0,
            "ffd_optimize_offset_endpoints": True,
            "min_center_spacing": 0.05,
            "spring_A": 20.0,
        },
    )

    for side in ("UPPER", "LOWER"):
        assert redistributed[side][0] == pytest.approx(0.0)
        assert redistributed[side][-1] == pytest.approx(1.0)
        assert len(redistributed[side]) == len(columns)
        assert all(
            right - left >= 0.05 - 1.0e-12
            for left, right in zip(
                redistributed[side][:-1], redistributed[side][1:]
            )
        )


def test_te_only_offset_endpoint_spring_keeps_only_te_anchor():
    columns = [0.2, 0.5, 0.8, 1.0]
    level = FFDLevel(
        level_id=1,
        columns=columns,
        upper_columns=columns,
        lower_columns=columns,
        dual_box=True,
        workdir="LEVEL_1",
        config_filename="config.cfg",
        project_filename="project.pkl",
        active_xmin=0.0,
        active_xmax=1.0,
        active_include_bounds=True,
    )
    result = {
        "dv_values": [4.0, 2.0, 0.5, 0.1, 0.1, 0.5, 2.0, 4.0]
    }
    redistributed = apply_post_opt_ffd_spring(
        level,
        result,
        {
            "ffd_domain_mode": "FULL",
            "ffd_active_xmin": 0.0,
            "ffd_active_xmax": 1.0,
            "ffd_optimize_offset_endpoints": False,
            "ffd_optimize_le_offset_endpoints": False,
            "ffd_optimize_te_offset_endpoints": True,
            "min_center_spacing": 0.05,
            "spring_A": 20.0,
        },
    )

    for side in ("UPPER", "LOWER"):
        assert redistributed[side][0] > 0.0
        assert redistributed[side][-1] == pytest.approx(1.0)
        assert len(redistributed[side]) == len(columns)


@pytest.mark.parametrize("reoptimize", [True, False])
def test_half_spring_supports_reoptimize_and_refine_post_actions(reoptimize):
    level = FFDLevel(
        level_id=2,
        columns=[0.2, 0.5, 0.8],
        workdir="LEVEL_2",
        config_filename="config_level2.cfg",
        project_filename="project_level2.pkl",
        mesh_source="accepted.su2",
        domain_mode="HALF_UPPER",
        side="UPPER",
        post_opt_spring_pending=True,
        active_xmin=0.0,
        active_xmax=1.0,
    )
    opts = {
        "ffd_domain_mode": "HALF_UPPER",
        "ffd_side": "UPPER",
        "ffd_box_tag": "UPPER_BOX",
        "ffd_upper_box_tag": "UPPER_BOX",
        "ffd_lower_box_tag": "LOWER_BOX",
        "ffd_dv_kind": "FFD_CONTROL_POINT_2D",
        "ffd_marker": "AIRFOIL",
        "ffd_direction": "OUTWARD",
        "ffd_control_row": 1,
        "ffd_active_xmin": 0.0,
        "ffd_active_xmax": 1.0,
        "ffd_active_include_bounds": False,
        "ffd_dual_box": False,
        "min_center_spacing": 0.05,
        "spring_A": 20.0,
        "spring_timing": "POST_OPT",
        "spring_score_mode": "COEFFICIENT",
        "spring_post_action": "REOPTIMIZE" if reoptimize else "REFINE",
    }
    spring_level = build_ffd_spring_reallocated_level(
        level,
        {
            "dv_values": [3.0, 1.0, 0.1],
            "final_mesh": "accepted_deformed.su2",
        },
        opts,
        reoptimize=reoptimize,
    )

    assert spring_level.ndv == level.ndv
    assert spring_level.side == "UPPER"
    assert spring_level.post_opt_spring_pending is False
    assert spring_level.spring_reallocated is True
    assert spring_level.mesh_source == "accepted_deformed.su2"
    assert spring_level.level_id == (3 if reoptimize else 2)
    assert spring_level.workdir == ("LEVEL_3" if reoptimize else "LEVEL_2")


@pytest.mark.parametrize(
    "indicator,expected_calls",
    [
        ("ABS_GRAD", ["OBJECTIVE"]),
        ("IKKT", ["OBJECTIVE", "EQUALITY", "INEQUALITY"]),
    ],
)
def test_exact_scoring_baseline_is_refreshed_at_the_accepted_design(
    tmp_path,
    indicator,
    expected_calls,
):
    workdir = tmp_path / "LEVEL_0"
    workdir.mkdir()
    level = FFDLevel(
        level_id=0,
        columns=[0.25, 0.75],
        workdir=str(workdir),
        config_filename="config.cfg",
        project_filename="project.pkl",
        domain_mode="HALF_UPPER",
        side="UPPER",
        active_xmin=0.0,
        active_xmax=1.0,
    )

    class FakeProject:
        def __init__(self):
            self.calls = []
            self.last_obj_grad_x = [0.1, 0.2]

        def _record(self, name, values):
            assert Path.cwd() == workdir
            self.calls.append(name)
            assert values == pytest.approx([0.3, 0.4])
            return []

        def obj_df(self, values):
            return self._record("OBJECTIVE", values)

        def con_dceq(self, values):
            return self._record("EQUALITY", values)

        def con_dcieq(self, values):
            return self._record("INEQUALITY", values)

    project = FakeProject()
    refreshed = refresh_ffd_scoring_baseline(
        project,
        level,
        [0.3, 0.4],
        {
            "refinement": "ADAPTIVE",
            "adaptive_indicator": indicator,
        },
    )
    assert refreshed is True
    assert project.calls == expected_calls
    assert project.last_obj_grad_x == pytest.approx([0.3, 0.4])


def test_exact_scoring_baseline_refresh_rejects_missing_accepted_dvs(tmp_path):
    level = FFDLevel(
        level_id=0,
        columns=[0.25, 0.75],
        workdir=str(tmp_path),
        config_filename="config.cfg",
        project_filename="project.pkl",
        domain_mode="HALF_UPPER",
        side="UPPER",
        active_xmin=0.0,
        active_xmax=1.0,
    )
    with pytest.raises(RuntimeError, match="accepted DV values"):
        refresh_ffd_scoring_baseline(
            object(),
            level,
            [0.1],
            {"refinement": "ADAPTIVE"},
        )


def test_adaptive_dual_refinement_can_add_only_to_lower(monkeypatch):
    level = FFDLevel(
        level_id=0,
        columns=[0.25, 0.75],
        upper_columns=[0.25, 0.75],
        lower_columns=[0.25, 0.75],
        dual_box=True,
        workdir="LEVEL_0",
        config_filename="config.cfg",
        project_filename="project.pkl",
        active_xmin=0.0,
        active_xmax=1.0,
    )

    def fake_scores(_level, _opts, mesh_source=None):
        return {
            "candidates": [
                {
                    "side": "LOWER",
                    "x": 0.5,
                    "indicator": 2.0,
                    "interval_id": 1,
                    "interval_left": 0.25,
                    "interval_right": 0.75,
                    "sample_index": 1,
                    "sample_fraction": 0.5,
                }
            ]
        }

    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._compute_ffd_dot_candidate_scores",
        fake_scores,
    )
    opts = {
        "refinement": "ADAPTIVE",
        "nfinal": 5,
        "nadd_mode": "FIXED",
        "fixed_nadd": 1,
        "min_center_spacing": 0.0,
        "trigger": "MAX_ITER",
        "ffd_dual_box": True,
        "ffd_active_xmin": 0.0,
        "ffd_active_xmax": 1.0,
    }
    upper, lower = refine_ffd_columns(level, {}, opts)
    assert upper == pytest.approx([0.25, 0.75])
    assert lower == pytest.approx([0.25, 0.5, 0.75])


def test_exact_candidate_scoring_uses_sequential_growth_ratio_selection(
    monkeypatch,
    capsys,
):
    level = FFDLevel(
        level_id=0,
        columns=[0.25, 0.75],
        upper_columns=[0.25, 0.75],
        lower_columns=[0.25, 0.75],
        dual_box=True,
        workdir="LEVEL_0",
        config_filename="config.cfg",
        project_filename="project.pkl",
        active_xmin=0.0,
        active_xmax=1.0,
    )

    scored_mesh_sources = []

    selected_upper = {
        "side": "UPPER",
        "x": 0.5,
        "indicator": 3.0,
        "rank": 1,
        "insertion_step": 1,
        "insertion_target": 2,
        "interval_id": 1,
        "interval_left": 0.25,
        "interval_right": 0.75,
        "sample_index": 1,
        "sample_fraction": 0.5,
        "candidate_dv_index": 1,
        "control_point_i": 2,
        "temporary_mesh": "candidate_upper.su2",
        "scoring_basis": "EXACT_SEQUENTIAL_INSERTION",
    }
    selected_lower = {
        "side": "LOWER",
        "x": 0.4,
        "indicator": 2.0,
        "rank": 1,
        "insertion_step": 2,
        "insertion_target": 2,
        "interval_id": 1,
        "interval_left": 0.25,
        "interval_right": 0.75,
        "sample_index": 1,
        "sample_fraction": 0.5,
        "candidate_dv_index": 4,
        "control_point_i": 2,
        "temporary_mesh": "candidate_lower.su2",
        "scoring_basis": "EXACT_SEQUENTIAL_INSERTION",
    }

    def fake_scores(_level, _opts, mesh_source=None):
        scored_mesh_sources.append(mesh_source)
        return {
            "sequential_selection": True,
            "scoring_basis": "EXACT_SEQUENTIAL_INSERTION",
            "selected_candidate": selected_upper,
            "selected_candidates": [selected_upper, selected_lower],
            "insertion_target": 2,
            "insertions_completed": 2,
            "selected_artifact_directory": "LEVEL_0/FFD_SELECTED_CANDIDATE",
            "candidate_scores_csv": "LEVEL_0/ffd_candidate_scores_level0.csv",
            "candidates": [selected_upper, selected_lower],
        }

    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._compute_ffd_dot_candidate_scores",
        fake_scores,
    )
    opts = {
        "refinement": "ADAPTIVE",
        "nfinal": 8,
        "nadd_mode": "GROWTH_RATIO",
        "growth_ratio": 2.0,
        "min_center_spacing": 0.0,
        "trigger": "MAX_ITER",
        "ffd_dual_box": True,
        "ffd_active_xmin": 0.0,
        "ffd_active_xmax": 1.0,
    }
    upper, lower = refine_ffd_columns(
        level,
        {"final_mesh": "accepted_deformed_mesh.su2"},
        opts,
    )
    assert upper == pytest.approx([0.25, 0.5, 0.75])
    assert lower == pytest.approx([0.25, 0.4, 0.75])
    assert opts["_last_selection_metadata"]["n_added"] == 2
    assert (
        opts["_last_selection_metadata"]["nadd_mode"]
        == "SEQUENTIAL_GROWTH_RATIO"
    )
    selected = opts["_last_selection_metadata"]["selected"]
    assert [item["insertion_step"] for item in selected] == [1, 2]
    assert selected[0]["candidate_dv_index"] == 1
    assert selected[1]["candidate_dv_index"] == 4
    assert all(
        item["scoring_basis"] == "EXACT_SEQUENTIAL_INSERTION"
        for item in selected
    )
    assert [item["indicator_ratio_to_best"] for item in selected] == [1.0, 1.0]
    assert scored_mesh_sources == ["accepted_deformed_mesh.su2"]
    output = capsys.readouterr().out
    assert "Leaving adaptive scoring phase | selected=2 target=2" in output
    assert "ADAPTIVE batch" not in output


@pytest.mark.parametrize("blending", ["BEZIER", "BSPLINE_UNIFORM"])
def test_exact_scoring_failure_is_not_hidden_by_uniform_fallback(
    monkeypatch,
    blending,
):
    level = FFDLevel(
        level_id=0,
        columns=[0.25, 0.75],
        upper_columns=[0.25, 0.75],
        lower_columns=[0.25, 0.75],
        dual_box=True,
        workdir="LEVEL_0",
        config_filename="config.cfg",
        project_filename="project.pkl",
        active_xmin=0.0,
        active_xmax=1.0,
    )

    def fail_scoring(_level, _opts, mesh_source=None):
        raise RuntimeError("SU2_DOT candidate failure")

    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._compute_ffd_dot_candidate_scores",
        fail_scoring,
    )
    opts = {
        "refinement": "ADAPTIVE",
        "ffd_blending": blending,
        "nfinal": 5,
        "ffd_dual_box": True,
    }
    with pytest.raises(RuntimeError, match="refusing.*uniform refinement"):
        refine_ffd_columns(level, {}, opts)


def test_bspline_uniform_dispatches_to_exact_sequential_scoring(
    tmp_path,
    monkeypatch,
):
    config_path = tmp_path / "config.cfg"
    config_path.write_text(
        "MESH_FILENAME= mesh.su2\n"
        "MARKER_MONITORING= ( AIRFOIL )\n"
        "OPT_OBJECTIVE= DRAG\n"
        "OPT_CONSTRAINT= NONE\n"
    )
    level = FFDLevel(
        level_id=0,
        columns=[0.25, 0.75],
        upper_columns=[0.25, 0.75],
        lower_columns=[0.25, 0.75],
        dual_box=True,
        workdir=str(tmp_path),
        config_filename=config_path.name,
        project_filename="project.pkl",
        active_xmin=0.0,
        active_xmax=1.0,
    )
    captured = {}

    def fake_exact(
        _level,
        opts,
        _cfg_level,
        raw_candidates,
        active_by_side,
        accepted_mesh=None,
    ):
        captured["blending"] = opts["ffd_blending"]
        captured["raw_candidates"] = list(raw_candidates)
        captured["active_by_side"] = active_by_side
        captured["accepted_mesh"] = accepted_mesh
        return {"scoring_basis": "EXACT_SEQUENTIAL_INSERTION"}

    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._compute_dual_bezier_exact_candidate_scores",
        fake_exact,
    )
    result = _compute_dual_ffd_dot_candidate_scores(
        level,
        {
            "ffd_blending": "BSPLINE_UNIFORM",
            "candidate_samples": 1,
            "min_center_spacing": 0.0,
            "ffd_active_xmin": 0.0,
            "ffd_active_xmax": 1.0,
        },
        mesh_source="accepted.su2",
    )

    assert result["scoring_basis"] == "EXACT_SEQUENTIAL_INSERTION"
    assert captured["blending"] == "BSPLINE_UNIFORM"
    assert captured["raw_candidates"]
    assert captured["active_by_side"] == {
        "UPPER": [0.25, 0.75],
        "LOWER": [0.25, 0.75],
    }
    assert captured["accepted_mesh"] == "accepted.su2"


def test_exact_sequential_zero_growth_target_is_a_noop(monkeypatch, capsys):
    level = FFDLevel(
        level_id=0,
        columns=[0.25, 0.75],
        upper_columns=[0.25, 0.75],
        lower_columns=[0.25, 0.75],
        dual_box=True,
        workdir="LEVEL_0",
        config_filename="config.cfg",
        project_filename="project.pkl",
        active_xmin=0.0,
        active_xmax=1.0,
    )
    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._compute_ffd_dot_candidate_scores",
        lambda *_args, **_kwargs: {
            "candidates": [],
            "sequential_selection": True,
            "scoring_basis": "EXACT_SEQUENTIAL_INSERTION",
            "insertion_target": 0,
            "selected_candidates": [],
        },
    )

    upper, lower = refine_ffd_columns(
        level,
        {},
        {
            "refinement": "ADAPTIVE",
            "ffd_blending": "BEZIER",
            "ffd_dual_box": True,
            "nfinal": 4,
        },
    )

    assert upper == pytest.approx([0.25, 0.75])
    assert lower == pytest.approx([0.25, 0.75])
    assert "Sequential growth target is zero" in capsys.readouterr().out


@pytest.mark.parametrize("blending", ["BEZIER", "BSPLINE_UNIFORM"])
def test_exact_candidates_are_scored_in_individual_next_level_bases(
    tmp_path,
    monkeypatch,
    blending,
):
    dot_cfg = tmp_path / "config_DOT.cfg"
    dot_cfg.write_text("MESH_FILENAME= source.su2\nMATH_PROBLEM= DISCRETE_ADJOINT\n")
    level = FFDLevel(
        level_id=2,
        columns=[0.2, 0.6],
        upper_columns=[0.2, 0.6],
        lower_columns=[0.3, 0.8],
        dual_box=True,
        workdir=str(tmp_path),
        config_filename="config.cfg",
        project_filename="project.pkl",
        active_xmin=0.0,
        active_xmax=1.0,
    )
    active_by_side = {"UPPER": [0.2, 0.6], "LOWER": [0.3, 0.8]}
    raw_candidates = [
        {
            "side": "UPPER",
            "x": 0.4,
            "interval_id": 1,
            "interval_left": 0.2,
            "interval_right": 0.6,
            "sample_index": 1,
            "sample_fraction": 0.5,
        },
        {
            "side": "LOWER",
            "x": 0.55,
            "interval_id": 1,
            "interval_left": 0.3,
            "interval_right": 0.8,
            "sample_index": 1,
            "sample_fraction": 0.5,
        },
    ]
    prepared_records = []

    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._find_real_adjoint_assets",
        lambda *_args: (str(tmp_path), str(tmp_path)),
    )
    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._select_dot_config_path",
        lambda *_args: (str(dot_cfg), "DISCRETE_ADJOINT"),
    )
    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._find_projection_mesh_source",
        lambda *_args: str(tmp_path / "source.su2"),
    )

    def fake_prepare(
        _level,
        _opts,
        _cfg_level,
        _real_dot_cfg,
        _mesh_src,
        upper_active,
        lower_active,
        suffix,
        verbose=True,
    ):
        records = ordered_dual_ffd_records(upper_active, lower_active)
        prepared_records.append(records)
        mappings = {
            "UPPER": {
                float(x): index + 1
                for index, x in enumerate(sorted(float(v) for v in upper_active))
            },
            "LOWER": {
                float(x): index + 1
                for index, x in enumerate(sorted(float(v) for v in lower_active))
            },
        }
        return {
            "cfg_dot": SU2.io.Config({"DV_VALUE_NEW": [0.0] * len(records)}),
            "state": {"records": records},
            "records": records,
            "column_index_by_side": mappings,
            "mesh": str(tmp_path / f"{suffix}.su2"),
        }

    def fake_dot(_workdir, _cfg_dot, state, _function, verbose=True):
        records = state["records"]
        gradient = [0.0] * len(records)
        if ("UPPER", 0.4) in records:
            gradient[records.index(("UPPER", 0.4))] = -2.0
        if ("LOWER", 0.55) in records:
            gradient[records.index(("LOWER", 0.55))] = 5.0
        return gradient

    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._prepare_dual_projection_variant",
        fake_prepare,
    )
    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._run_ffd_dot_for_function",
        fake_dot,
    )
    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._persist_selected_candidate_artifacts",
        lambda *_args, **_kwargs: str(tmp_path / "FFD_SELECTED_CANDIDATE"),
    )
    result = _compute_dual_bezier_exact_candidate_scores(
        level,
        {
            "adaptive_indicator": "ABS_GRAD",
            "ffd_blending": blending,
            "growth_ratio": 1.0,
        },
        SU2.io.Config({"OBJECTIVE_FUNCTION": "DRAG"}),
        raw_candidates,
        active_by_side,
    )

    assert len(prepared_records) == 2
    assert all(len(records) == level.ndv + 1 for records in prepared_records)
    assert not any(
        ("UPPER", 0.4) in records and ("LOWER", 0.55) in records
        for records in prepared_records
    )
    by_side = {item["side"]: item for item in result["raw_candidates"]}
    assert by_side["UPPER"]["candidate_dv_index"] == 1
    assert by_side["UPPER"]["indicator"] == pytest.approx(2.0)
    assert by_side["LOWER"]["candidate_dv_index"] == 3
    assert by_side["LOWER"]["indicator"] == pytest.approx(5.0)
    assert result["sequential_selection"] is True
    assert result["ffd_blending"] == blending
    assert result["insertion_target"] == 1
    assert result["insertions_completed"] == 1


@pytest.mark.parametrize("blending", ["BEZIER", "BSPLINE_UNIFORM"])
def test_half_upper_uses_the_same_exact_sequential_scoring(
    tmp_path,
    monkeypatch,
    blending,
):
    dot_cfg = tmp_path / "config_DOT.cfg"
    dot_cfg.write_text(
        "MESH_FILENAME= source.su2\nMATH_PROBLEM= DISCRETE_ADJOINT\n"
    )
    level = FFDLevel(
        level_id=1,
        columns=[0.25, 0.75],
        workdir=str(tmp_path),
        config_filename="config.cfg",
        project_filename="project.pkl",
        domain_mode="HALF_UPPER",
        side="UPPER",
        active_xmin=0.0,
        active_xmax=1.0,
    )
    active_by_side = {"UPPER": [0.25, 0.75]}
    raw_candidates = [
        {
            "side": "UPPER",
            "x": x,
            "interval_id": 1,
            "interval_left": 0.25,
            "interval_right": 0.75,
            "sample_index": index,
            "sample_fraction": index / 3.0,
        }
        for index, x in enumerate((0.4, 0.6), start=1)
    ]
    prepared = []

    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._find_real_adjoint_assets",
        lambda *_args: (str(tmp_path), str(tmp_path)),
    )
    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._select_dot_config_path",
        lambda *_args: (str(dot_cfg), "DISCRETE_ADJOINT"),
    )
    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._find_projection_mesh_source",
        lambda *_args: str(tmp_path / "source.su2"),
    )

    def fake_prepare(
        _level,
        _opts,
        _cfg_level,
        _real_dot_cfg,
        _mesh_src,
        current_by_side,
        suffix,
        verbose=True,
    ):
        records = ordered_ffd_records(current_by_side, ("UPPER",))
        prepared.append(records)
        return {
            "cfg_dot": SU2.io.Config({"DV_VALUE_NEW": [0.0] * len(records)}),
            "state": {"records": records},
            "records": records,
            "column_index_by_side": {
                "UPPER": {
                    float(x): index + 1
                    for index, x in enumerate(current_by_side["UPPER"])
                }
            },
            "mesh": str(tmp_path / f"{suffix}.su2"),
        }

    def fake_dot(_workdir, _cfg_dot, state, _function, verbose=True):
        scores = {("UPPER", 0.4): 2.0, ("UPPER", 0.6): 5.0}
        return [scores.get(record, 0.0) for record in state["records"]]

    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._prepare_projection_variant",
        fake_prepare,
    )
    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._run_ffd_dot_for_function",
        fake_dot,
    )
    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._persist_selected_candidate_artifacts",
        lambda *_args, **_kwargs: str(tmp_path / "FFD_SELECTED_CANDIDATE"),
    )

    result = _compute_dual_bezier_exact_candidate_scores(
        level,
        {
            "adaptive_indicator": "ABS_GRAD",
            "ffd_blending": blending,
            "growth_ratio": 1.0,
        },
        SU2.io.Config({"OBJECTIVE_FUNCTION": "DRAG"}),
        raw_candidates,
        active_by_side,
    )

    assert len(prepared) == 2
    assert all(all(side == "UPPER" for side, _ in records) for records in prepared)
    assert result["selected_candidates"][0]["side"] == "UPPER"
    assert result["selected_candidates"][0]["x"] == pytest.approx(0.6)
    assert result["scoring_basis"] == "EXACT_SEQUENTIAL_INSERTION"


def test_exact_bezier_scoring_keeps_only_selected_artifacts_and_compact_ranking(
    tmp_path,
    monkeypatch,
    capsys,
):
    dot_cfg = tmp_path / "config_DOT.cfg"
    dot_cfg.write_text(
        "MESH_FILENAME= source.su2\nMATH_PROBLEM= DISCRETE_ADJOINT\n"
    )
    level = FFDLevel(
        level_id=2,
        columns=[0.2, 0.6],
        upper_columns=[0.2, 0.6],
        lower_columns=[0.3, 0.8],
        dual_box=True,
        workdir=str(tmp_path),
        config_filename="config.cfg",
        project_filename="project.pkl",
        active_xmin=0.0,
        active_xmax=1.0,
    )
    active_by_side = {"UPPER": [0.2, 0.6], "LOWER": [0.3, 0.8]}
    raw_candidates = [
        {
            "side": "UPPER",
            "x": 0.3,
            "interval_id": 1,
            "interval_left": 0.2,
            "interval_right": 0.6,
            "sample_index": 1,
            "sample_fraction": 1.0 / 3.0,
        },
        {
            "side": "UPPER",
            "x": 0.45,
            "interval_id": 1,
            "interval_left": 0.2,
            "interval_right": 0.6,
            "sample_index": 2,
            "sample_fraction": 2.0 / 3.0,
        },
        {
            "side": "LOWER",
            "x": 0.5,
            "interval_id": 1,
            "interval_left": 0.3,
            "interval_right": 0.8,
            "sample_index": 1,
            "sample_fraction": 1.0 / 3.0,
        },
        {
            "side": "LOWER",
            "x": 0.65,
            "interval_id": 1,
            "interval_left": 0.3,
            "interval_right": 0.8,
            "sample_index": 2,
            "sample_fraction": 2.0 / 3.0,
        },
    ]
    score_by_record = {
        ("UPPER", 0.3): 2.0,
        ("UPPER", 0.45): 3.0,
        ("LOWER", 0.5): 5.0,
        ("LOWER", 0.65): 4.0,
        ("UPPER", 1.0 / 3.0): 7.0,
    }
    prepared_records = []

    stale_selected = tmp_path / "FFD_SELECTED_CANDIDATE"
    stale_selected.mkdir()
    (stale_selected / "stale.txt").write_text("old")
    stale_mesh = tmp_path / "ffd_projection_level2_bezier_stale.su2"
    stale_mesh.write_text("old")
    stale_dot = tmp_path / "DOT_ONLY_DRAG_ffd_projection_level2_bezier_stale"
    stale_dot.mkdir()

    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._find_real_adjoint_assets",
        lambda *_args: (str(tmp_path), str(tmp_path)),
    )
    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._select_dot_config_path",
        lambda *_args: (str(dot_cfg), "DISCRETE_ADJOINT"),
    )
    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._find_projection_mesh_source",
        lambda *_args: str(tmp_path / "source.su2"),
    )

    def fake_prepare(
        _level,
        _opts,
        _cfg_level,
        _real_dot_cfg,
        _mesh_src,
        upper_active,
        lower_active,
        suffix,
        verbose=True,
    ):
        records = ordered_dual_ffd_records(upper_active, lower_active)
        prepared_records.append((suffix, records))
        mesh = tmp_path / f"ffd_projection_level2_{suffix}.su2"
        mesh.write_text(suffix)
        mappings = {
            "UPPER": {
                float(x): index + 1
                for index, x in enumerate(sorted(float(v) for v in upper_active))
            },
            "LOWER": {
                float(x): index + 1
                for index, x in enumerate(sorted(float(v) for v in lower_active))
            },
        }
        cfg_dot = SU2.io.Config(
            {
                "DV_VALUE_NEW": [0.0] * len(records),
                "MESH_FILENAME": mesh.name,
            }
        )
        return {
            "cfg_dot": cfg_dot,
            "state": {"records": records},
            "records": records,
            "column_index_by_side": mappings,
            "mesh": str(mesh),
        }

    def fake_dot(workdir, cfg_dot, state, function, verbose=True):
        artifact = Path(
            _ffd_dot_artifact_directory(workdir, cfg_dot, function)
        )
        artifact.mkdir()
        (artifact / "gradient.dat").write_text("mock")
        gradient = [0.0] * len(state["records"])
        for index, record in enumerate(state["records"]):
            for (score_side, score_x), score in score_by_record.items():
                if record[0] == score_side and abs(record[1] - score_x) <= 1.0e-10:
                    gradient[index] = score
                    break
        return gradient

    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._prepare_dual_projection_variant",
        fake_prepare,
    )
    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._run_ffd_dot_for_function",
        fake_dot,
    )

    result = _compute_dual_bezier_exact_candidate_scores(
        level,
        {
            "adaptive_indicator": "ABS_GRAD",
            "candidate_samples": 2,
            "ffd_blending": "BEZIER",
            "growth_ratio": 1.5,
        },
        SU2.io.Config({"OBJECTIVE_FUNCTION": "DRAG"}),
        raw_candidates,
        active_by_side,
    )

    selected_dir = tmp_path / "FFD_SELECTED_CANDIDATE"
    assert result["selected_candidate"]["side"] == "LOWER"
    assert result["selected_candidate"]["x"] == pytest.approx(0.5)
    assert result["insertion_target"] == 2
    assert result["insertions_completed"] == 2
    assert [item["side"] for item in result["selected_candidates"]] == [
        "LOWER",
        "UPPER",
    ]
    assert result["selected_candidates"][1]["x"] == pytest.approx(1.0 / 3.0)
    assert sorted(path.name for path in selected_dir.iterdir()) == [
        "insertion_001",
        "insertion_002",
    ]
    for insertion in ("insertion_001", "insertion_002"):
        insertion_dir = selected_dir / insertion
        assert sorted(path.name for path in insertion_dir.iterdir()) == [
            "DOT_DRAG",
            "candidate.su2",
            "candidate_gradients.csv",
            "candidate_metadata.json",
        ]
        assert (insertion_dir / "DOT_DRAG" / "gradient.dat").is_file()
    assert not list(tmp_path.glob("DOT_ONLY_*"))
    assert not list(tmp_path.glob("GEO_ONLY_*"))
    assert not list(tmp_path.glob("ffd_projection_level2_bezier_*.su2"))
    assert not (selected_dir / "stale.txt").exists()

    with Path(result["candidate_scores_csv"]).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows_by_step = {
        step: [row for row in rows if row["insertion_step"] == step]
        for step in ("1", "2")
    }
    assert len(rows_by_step["1"]) == 4
    assert len(rows_by_step["2"]) == 14
    assert sum(row["interval_winner"] == "YES" for row in rows_by_step["1"]) == 2
    assert sum(row["interval_winner"] == "YES" for row in rows_by_step["2"]) == 7

    step_two_variants = [
        records
        for suffix, records in prepared_records
        if "step_002_candidate" in suffix
    ]
    assert step_two_variants
    assert all(("LOWER", 0.5) in records for records in step_two_variants)
    assert any(
        any(
            side == "UPPER" and abs(x - 1.0 / 3.0) <= 1.0e-10
            for side, x in records
        )
        for records in step_two_variants
    )

    output = capsys.readouterr().out
    assert "Entering adaptive scoring phase" in output
    assert "Candidate ranking" in output
    ranking_rows = [
        line
        for line in output.splitlines()
        if len(line.split()) > 1 and line.split()[1].isdigit()
    ]
    assert len(ranking_rows) == 9
    assert "LOWER" in ranking_rows[0] and "0.500000" in ranking_rows[0]
    assert any("UPPER" in line and "0.333333" in line for line in ranking_rows)
    assert "Candidate scoring |" not in output
    assert "Exact Bezier candidate |" not in output
    assert "DOT projection |" not in output
    assert "FFD control point format" not in output


def test_selected_ikkt_artifacts_are_grouped_and_json_is_strict(tmp_path):
    mesh = tmp_path / "candidate_tmp.su2"
    mesh.write_text("mesh")
    source_artifacts = []
    for kind, function in (
        ("DOT", "DRAG"),
        ("DOT", "LIFT"),
        ("GEOMETRY", "AIRFOIL_AREA"),
    ):
        path = tmp_path / f"{kind}_{function}_temporary"
        path.mkdir()
        (path / "result.dat").write_text(function)
        source_artifacts.append(
            {"kind": kind, "function": function, "path": str(path)}
        )
    selected = {
        "candidate_number": 3,
        "side": "UPPER",
        "x": 0.4,
        "indicator": 1.25,
        "interval_id": 1,
        "interval_left": 0.2,
        "interval_right": 0.6,
        "sample_index": 1,
        "sample_fraction": 0.5,
        "candidate_dv_index": 2,
        "control_point_i": 3,
        "temporary_mesh": str(mesh),
        "projection_artifacts": source_artifacts,
        "objective_function": "DRAG",
        "objective_gradient_component": -1.0,
        "constraint_gradient_components": {
            "LIFT": 0.5,
            "AIRFOIL_AREA": -0.25,
        },
    }
    level = FFDLevel(
        level_id=0,
        columns=[0.2, 0.6],
        upper_columns=[0.2, 0.6],
        lower_columns=[0.3, 0.8],
        dual_box=True,
        workdir=str(tmp_path),
        config_filename="config.cfg",
        project_filename="project.pkl",
    )

    selected_dir = Path(
        _persist_selected_candidate_artifacts(
            level,
            selected,
            "IKKT",
            {
                "constraint_names": ["LIFT", "AIRFOIL_AREA"],
                "lambdas": [0.0, 1.0],
                "lambda_lower": [-float("inf"), 0.0],
                "lambda_upper": [float("inf"), 1.0],
            },
        )
    )

    assert sorted(path.name for path in selected_dir.iterdir()) == [
        "DOT_DRAG",
        "DOT_LIFT",
        "GEO_AIRFOIL_AREA",
        "candidate.su2",
        "candidate_gradients.csv",
        "candidate_metadata.json",
        "ikkt_baseline.json",
    ]
    assert all(not Path(item["path"]).exists() for item in source_artifacts)
    raw_json = (selected_dir / "ikkt_baseline.json").read_text()
    assert "Infinity" not in raw_json
    baseline = json.loads(raw_json)
    assert baseline["lambda_lower"] == [None, 0.0]
    assert baseline["lambda_upper"] == [None, 1.0]


@pytest.mark.parametrize("blending", ["BEZIER", "BSPLINE_UNIFORM"])
def test_sequential_ikkt_recomputes_baseline_multipliers_after_each_insertion(
    tmp_path,
    monkeypatch,
    blending,
):
    dot_cfg = tmp_path / "config_DOT.cfg"
    dot_cfg.write_text(
        "MESH_FILENAME= source.su2\nMATH_PROBLEM= DISCRETE_ADJOINT\n"
    )
    level = FFDLevel(
        level_id=0,
        columns=[0.25, 0.75],
        upper_columns=[0.25, 0.75],
        lower_columns=[0.25, 0.75],
        dual_box=True,
        workdir=str(tmp_path),
        config_filename="config.cfg",
        project_filename="project.pkl",
        active_xmin=0.0,
        active_xmax=1.0,
    )
    raw_candidates = [
        {
            "side": side,
            "x": 0.5,
            "interval_id": 1,
            "interval_left": 0.25,
            "interval_right": 0.75,
            "sample_index": 1,
            "sample_fraction": 0.5,
        }
        for side in ("UPPER", "LOWER")
    ]
    baseline_sizes = []

    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._find_real_adjoint_assets",
        lambda *_args: (str(tmp_path), str(tmp_path)),
    )
    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._select_dot_config_path",
        lambda *_args: (str(dot_cfg), "DISCRETE_ADJOINT"),
    )
    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._find_projection_mesh_source",
        lambda *_args: str(tmp_path / "source.su2"),
    )
    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._extract_constraint_names",
        lambda *_args: ["LIFT"],
    )
    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._extract_constraint_signs",
        lambda *_args: ([-float("inf")], [float("inf")]),
    )

    def fake_prepare(
        _level,
        _opts,
        _cfg_level,
        _real_dot_cfg,
        _mesh_src,
        upper_active,
        lower_active,
        suffix,
        verbose=True,
    ):
        records = ordered_dual_ffd_records(upper_active, lower_active)
        mesh = tmp_path / f"ffd_projection_level0_{suffix}.su2"
        mesh.write_text(suffix)
        return {
            "cfg_dot": SU2.io.Config(
                {
                    "DV_VALUE_NEW": [0.0] * len(records),
                    "MESH_FILENAME": mesh.name,
                }
            ),
            "state": {"records": records},
            "records": records,
            "column_index_by_side": {
                "UPPER": {
                    float(x): index + 1
                    for index, x in enumerate(sorted(float(v) for v in upper_active))
                },
                "LOWER": {
                    float(x): index + 1
                    for index, x in enumerate(sorted(float(v) for v in lower_active))
                },
            },
            "mesh": str(mesh),
        }

    def fake_dot(workdir, cfg_dot, state, function, verbose=True):
        artifact = Path(_ffd_dot_artifact_directory(workdir, cfg_dot, function))
        artifact.mkdir()
        gradient = [0.0] * len(state["records"])
        for index, (side, x) in enumerate(state["records"]):
            if side == "UPPER" and abs(x - 0.5) <= 1.0e-10:
                gradient[index] = 5.0
            elif side == "LOWER" and abs(x - 0.5) <= 1.0e-10:
                gradient[index] = 4.0
        return gradient

    def fake_constraints(
        _level,
        cfg_dot,
        state,
        constraint_names,
        required,
        artifact_records=None,
        verbose=True,
    ):
        assert constraint_names == ["LIFT"]
        artifact = Path(
            _ffd_dot_artifact_directory(str(tmp_path), cfg_dot, "LIFT")
        )
        artifact.mkdir()
        if artifact_records is not None:
            artifact_records.append(
                {"kind": "DOT", "function": "LIFT", "path": str(artifact)}
            )
        return ["LIFT"], [[0.0] * len(state["records"])]

    def fake_ikkt(g_obj, constraint_grads, lambda_bounds=None, **_kwargs):
        baseline_sizes.append(len(g_obj))
        return list(g_obj), [float(len(g_obj))]

    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._prepare_dual_projection_variant",
        fake_prepare,
    )
    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._run_ffd_dot_for_function",
        fake_dot,
    )
    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._run_dual_projection_constraint_gradients",
        fake_constraints,
    )
    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._compute_ikkt_residual_vector",
        fake_ikkt,
    )

    result = _compute_dual_bezier_exact_candidate_scores(
        level,
        {
            "adaptive_indicator": "IKKT",
            "ffd_blending": blending,
            "growth_ratio": 1.5,
            "candidate_samples": 1,
            "min_center_spacing": 0.0,
            "ffd_active_xmin": 0.0,
            "ffd_active_xmax": 1.0,
            "ffd_marker": "AIRFOIL",
            "ffd_thickness_enabled": True,
            "ffd_thickness_ikkt_included": False,
            "ffd_thickness_ikkt_exclusion_reason": (
                "optimizer-only progressive constraint; excluded from FFD IKKT"
            ),
        },
        SU2.io.Config({"OBJECTIVE_FUNCTION": "DRAG"}),
        raw_candidates,
        {"UPPER": [0.25, 0.75], "LOWER": [0.25, 0.75]},
    )

    assert baseline_sizes == [4, 5]
    assert [item["side"] for item in result["selected_candidates"]] == [
        "UPPER",
        "LOWER",
    ]
    for insertion, expected_lambda in (("insertion_001", 4.0), ("insertion_002", 5.0)):
        metadata = json.loads(
            (
                tmp_path
                / "FFD_SELECTED_CANDIDATE"
                / insertion
                / "ikkt_baseline.json"
            ).read_text()
        )
        assert metadata["lambdas"] == [expected_lambda]
        assert metadata["ffd_blending"] == blending
        assert metadata["progressive_thickness"] == {
            "enabled": True,
            "included": False,
            "reason": (
                "optimizer-only progressive constraint; excluded from FFD IKKT"
            ),
        }


@pytest.mark.parametrize(
    "blending,slug",
    [("BEZIER", "bezier"), ("BSPLINE_UNIFORM", "bspline_uniform")],
)
def test_sequential_exact_scoring_cleans_all_artifacts_if_later_pass_fails(
    tmp_path,
    monkeypatch,
    blending,
    slug,
):
    level = FFDLevel(
        level_id=0,
        columns=[0.25, 0.75],
        upper_columns=[0.25, 0.75],
        lower_columns=[0.25, 0.75],
        dual_box=True,
        workdir=str(tmp_path),
        config_filename="config.cfg",
        project_filename="project.pkl",
    )
    raw_candidates = [
        {
            "side": "UPPER",
            "x": 0.5,
            "interval_id": 1,
            "interval_left": 0.25,
            "interval_right": 0.75,
            "sample_index": 1,
            "sample_fraction": 0.5,
        },
        {
            "side": "LOWER",
            "x": 0.5,
            "interval_id": 1,
            "interval_left": 0.25,
            "interval_right": 0.75,
            "sample_index": 1,
            "sample_fraction": 0.5,
        },
    ]

    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._generate_dual_exact_candidates",
        lambda *_args: list(raw_candidates),
    )

    def fake_pass(
        _level,
        _opts,
        _cfg_level,
        _raw_candidates,
        active_by_side,
        accepted_mesh=None,
        insertion_step=1,
        insertion_target=1,
    ):
        mesh = tmp_path / (
            f"ffd_projection_level0_{slug}_step_"
            f"{insertion_step:03d}_candidate.su2"
        )
        mesh.write_text("mesh")
        dot_dir = tmp_path / (
            f"DOT_ONLY_DRAG_ffd_projection_level0_{slug}_"
            f"step_{insertion_step:03d}_candidate"
        )
        dot_dir.mkdir()
        selected = {
            "insertion_step": insertion_step,
            "insertion_target": insertion_target,
            "ndv_before_insertion": (
                len(active_by_side["UPPER"]) + len(active_by_side["LOWER"])
            ),
            "ndv_after_insertion": (
                len(active_by_side["UPPER"])
                + len(active_by_side["LOWER"])
                + 1
            ),
            "candidate_number": 0,
            "side": "UPPER",
            "x": 0.5,
            "grad": -2.0,
            "indicator": 2.0,
            "rank": 1,
            "interval_winner": True,
            "interval_id": 1,
            "interval_left": 0.25,
            "interval_right": 0.75,
            "sample_index": 1,
            "sample_fraction": 0.5,
            "candidate_dv_index": 1,
            "control_point_i": 2,
            "temporary_mesh": str(mesh),
            "projection_artifacts": [
                {"kind": "DOT", "function": "DRAG", "path": str(dot_dir)}
            ],
            "objective_function": "DRAG",
            "objective_gradient_component": -2.0,
            "constraint_gradient_components": {},
            "scoring_basis": "EXACT_SEQUENTIAL_INSERTION",
        }
        return {
            "selected_candidate": selected,
            "raw_candidates": [selected],
            "ikkt_metadata": {},
        }

    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._compute_dual_bezier_exact_candidate_scores_impl",
        fake_pass,
    )

    with pytest.raises(RuntimeError, match="already-active station"):
        _compute_dual_bezier_exact_candidate_scores(
            level,
            {
                "adaptive_indicator": "ABS_GRAD",
                "ffd_blending": blending,
                "growth_ratio": 1.5,
                "candidate_samples": 1,
                "ffd_active_xmin": 0.0,
                "ffd_active_xmax": 1.0,
            },
            SU2.io.Config({"OBJECTIVE_FUNCTION": "DRAG"}),
            raw_candidates,
            {"UPPER": [0.25, 0.75], "LOWER": [0.25, 0.75]},
        )

    assert not (tmp_path / "FFD_SELECTED_CANDIDATE").exists()
    assert not list(tmp_path.glob(f"ffd_projection_level0_{slug}_*.su2"))
    assert not list(tmp_path.glob(f"DOT_ONLY_*_ffd_projection_level0_{slug}_*"))
    assert not (tmp_path / "ffd_candidate_scores_level0.csv").exists()


@pytest.mark.parametrize(
    "blending,orders",
    [("BEZIER", (2, 2, 2)), ("BSPLINE_UNIFORM", (4, 2, 2))],
)
def test_exact_variant_uses_native_upper_then_lower_dv_order(
    tmp_path,
    blending,
    orders,
):
    bootstrap = _write_bootstrap_mesh(tmp_path / "bootstrap.su2")
    dual = tmp_path / "dual.su2"
    _split(
        bootstrap,
        dual,
        output_blending=blending,
        bspline_orders=orders,
    )
    config = _dual_config(
        MESH_FILENAME=str(dual),
        FFD_BLENDING=blending,
        FFD_BSPLINE_ORDER=orders,
    )
    opts = _dual_opts(config)
    opts["ffd_active_xmin"] = 0.0
    opts["ffd_active_xmax"] = 1.0
    level = FFDLevel(
        level_id=1,
        columns=[0.25, 0.75],
        upper_columns=[0.25, 0.75],
        lower_columns=[0.25, 0.5, 0.75],
        dual_box=True,
        workdir=str(tmp_path),
        config_filename="config.cfg",
        project_filename="project.pkl",
        active_xmin=0.0,
        active_xmax=1.0,
    )
    variant = _prepare_dual_projection_variant(
        level,
        opts,
        SU2.io.Config({"NUMBER_PART": 1}),
        SU2.io.Config(
            {
                "MATH_PROBLEM": "DISCRETE_ADJOINT",
                "MESH_FILENAME": str(dual),
            }
        ),
        str(dual),
        upper_active=[0.25, 0.4, 0.75],
        lower_active=[0.25, 0.5, 0.75],
        suffix="ordering_probe",
    )
    assert variant["records"] == [
        ("UPPER", 0.25),
        ("UPPER", 0.4),
        ("UPPER", 0.75),
        ("LOWER", 0.25),
        ("LOWER", 0.5),
        ("LOWER", 0.75),
    ]
    assert _dual_record_index(variant["records"], "UPPER", 0.4) == 1
    assert variant["column_index_by_side"]["UPPER"][0.4] == 2
    definition = variant["cfg_dot"]["DEFINITION_DV"]
    assert definition["FFDTAG"] == [
        "UPPER_BOX",
        "UPPER_BOX",
        "UPPER_BOX",
        "LOWER_BOX",
        "LOWER_BOX",
        "LOWER_BOX",
    ]
    assert definition["PARAM"][1][0] == 2
    assert variant["cfg_dot"]["FFD_BLENDING"] == blending


def test_exact_candidate_spacing_does_not_compare_unselected_candidates():
    candidates = [{"x": 0.4}, {"x": 0.45}]
    default = _filter_candidates_by_min_spacing(
        [dict(item) for item in candidates],
        active_columns=[0.2, 0.8],
        min_spacing=0.1,
        xmin=0.0,
        xmax=1.0,
    )
    exact = _filter_candidates_by_min_spacing(
        [dict(item) for item in candidates],
        active_columns=[0.2, 0.8],
        min_spacing=0.1,
        xmin=0.0,
        xmax=1.0,
        compare_candidates=False,
    )
    assert [item["x"] for item in default] == pytest.approx([0.4])
    assert [item["x"] for item in exact] == pytest.approx([0.4, 0.45])


def test_exact_bezier_ikkt_uses_fixed_baseline_multipliers():
    indicators = _ikkt_indicators_with_fixed_lambdas(
        objective_gradient=[2.0, 10.0, -4.0],
        constraint_gradients=[[1.0, 3.0, 2.0]],
        lambdas=[2.0],
    )
    assert indicators == pytest.approx([0.0, 4.0, 8.0])

    with pytest.raises(RuntimeError, match="count mismatch"):
        _ikkt_indicators_with_fixed_lambdas(
            objective_gradient=[1.0],
            constraint_gradients=[],
            lambdas=[1.0],
        )

    with pytest.raises(Exception):
        _compute_ikkt_residual_vector(
            [1.0, 2.0],
            [[1.0, 1.0]],
            lambda_bounds=([0.0, 0.0], [1.0, 1.0]),
            strict=True,
        )


def test_exact_bezier_requires_dot_mesh_to_match_accepted_baseline(tmp_path):
    accepted = _write_bootstrap_mesh(tmp_path / "accepted.su2")
    matching = _write_bootstrap_mesh(tmp_path / "matching.su2")
    resolved = _resolve_accepted_projection_mesh(
        str(accepted),
        str(matching),
        str(tmp_path),
        str(tmp_path),
        str(tmp_path),
        "AIRFOIL",
    )
    assert resolved == str(accepted.resolve())

    shifted_points = list(SYMMETRIC_POINTS)
    shifted_points[1] = (shifted_points[1][0], shifted_points[1][1] + 1.0e-4)
    shifted = _write_bootstrap_mesh(
        tmp_path / "shifted.su2",
        points=shifted_points,
    )
    with pytest.raises(RuntimeError, match="same physical baseline"):
        _resolve_accepted_projection_mesh(
            str(accepted),
            str(shifted),
            str(tmp_path),
            str(tmp_path),
            str(tmp_path),
            "AIRFOIL",
        )


def test_exact_bezier_ikkt_requires_every_configured_constraint(monkeypatch):
    level = FFDLevel(
        level_id=0,
        columns=[0.25, 0.75],
        upper_columns=[0.25, 0.75],
        lower_columns=[0.25, 0.75],
        dual_box=True,
        workdir="LEVEL_0",
        config_filename="config.cfg",
        project_filename="project.pkl",
        active_xmin=0.0,
        active_xmax=1.0,
    )

    def unavailable(*_args, **_kwargs):
        raise FileNotFoundError("missing constraint assets")

    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._run_ffd_dot_for_function",
        unavailable,
    )
    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_projection._run_ffd_geo_gradient_for_function",
        unavailable,
    )
    with pytest.raises(RuntimeError, match="gradient is unavailable"):
        _run_dual_projection_constraint_gradients(
            level,
            SU2.io.Config({"DV_VALUE_NEW": [0.0, 0.0]}),
            SU2.io.State(),
            ["LIFT"],
            required=True,
        )


def test_next_dual_level_absorbs_mesh_and_zeros_old_and_new_dvs(monkeypatch):
    level = FFDLevel(
        level_id=0,
        columns=[0.25, 0.75],
        upper_columns=[0.25, 0.75],
        lower_columns=[0.25, 0.75],
        dual_box=True,
        workdir="LEVEL_0",
        config_filename="config.cfg",
        project_filename="project.pkl",
        mesh_source="initial.su2",
        active_xmin=0.0,
        active_xmax=1.0,
    )
    monkeypatch.setattr(
        "SU2.opt.progressive_ffd_levels.refine_ffd_columns",
        lambda *_args: ([0.25, 0.5, 0.75], [0.25, 0.75]),
    )
    opts = _dual_opts(_dual_config())
    next_level = build_next_ffd_level(
        level,
        {"final_mesh": "accepted_deformed_mesh.su2"},
        opts,
    )
    assert next_level.mesh_source == "accepted_deformed_mesh.su2"
    assert next_level.dv_records == [
        ("UPPER", 0.25),
        ("UPPER", 0.5),
        ("UPPER", 0.75),
        ("LOWER", 0.25),
        ("LOWER", 0.75),
    ]
    assert next_level.dv_values == pytest.approx([0.0] * 5)


def test_exact_candidate_identity_is_persisted_in_selection_history(tmp_path):
    csv_path = tmp_path / "selection_history.csv"
    append_selection_history_csv(
        str(csv_path),
        {
            "level_id": 0,
            "ndv_before": 4,
            "ndv_after": 5,
            "n_added": 1,
            "nadd_mode": "SEQUENTIAL_GROWTH_RATIO",
            "trigger_mode": "MAX_ITER",
            "refinement": "ADAPTIVE",
            "spring_enabled": False,
            "upper_before": [0.25, 0.75],
            "lower_before": [0.25, 0.75],
            "upper_after": [0.25, 0.5, 0.75],
            "lower_after": [0.25, 0.75],
            "selected": [
                {
                    "side": "UPPER",
                    "x": 0.5,
                    "indicator": 3.0,
                    "indicator_ratio_to_best": 1.0,
                    "candidate_dv_index": 1,
                    "control_point_i": 2,
                    "scoring_basis": "EXACT_SEQUENTIAL_INSERTION",
                    "temporary_mesh": "candidate_upper.su2",
                    "insertion_step": 1,
                    "insertion_target": 2,
                    "artifact_directory": "FFD_SELECTED_CANDIDATE/insertion_001",
                }
            ],
        },
        {"history_file": "history.csv", "final_mesh": "mesh_out.su2"},
    )
    with csv_path.open(newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["side"] == "UPPER"
    assert row["x"] == "0.5"
    assert row["candidate_dv_index"] == "1"
    assert row["control_point_i"] == "2"
    assert row["scoring_basis"] == "EXACT_SEQUENTIAL_INSERTION"
    assert row["temporary_mesh"] == "candidate_upper.su2"
    assert row["insertion_step"] == "1"
    assert row["insertion_target"] == "2"
    assert row["artifact_directory"].endswith("insertion_001")


def test_level_config_contains_two_independent_boxes_and_user_continuity(
    tmp_path,
    monkeypatch,
):
    bootstrap = _write_bootstrap_mesh(tmp_path / "bootstrap.su2")
    dual = tmp_path / "dual.su2"
    _split(bootstrap, dual)

    config = _dual_config(
        MESH_FILENAME=str(dual),
        MESH_OUT_FILENAME="mesh_out",
        FFD_CONTINUITY="2ND_DERIVATIVE",
        FFD_FIX_I="( 0, 1 )",
    )
    config._filename = str(tmp_path / "Config_FFD.cfg")
    opts = _dual_opts(config)
    opts["ffd_active_xmin"] = 0.0
    opts["ffd_active_xmax"] = 1.0
    level = build_initial_ffd_level(config, opts)

    monkeypatch.chdir(tmp_path)
    cfg_path = Path(write_ffd_level_config(config, level, opts))
    text = cfg_path.read_text()
    assert "FFD_CONTINUITY= USER_INPUT" in text
    assert "FFD_FIX_I" not in text
    assert text.count("( 19") == 6
    assert text.count("UPPER_BOX") == 3
    assert text.count("LOWER_BOX") == 3

    level_mesh = tmp_path / "LEVEL_0" / "ffd_level0.su2"
    mesh_text = level_mesh.read_text()
    assert "FFD_NBOX= 2" in mesh_text
    assert "FFD_TAG= UPPER_BOX" in mesh_text
    assert "FFD_TAG= LOWER_BOX" in mesh_text
    assert ordered_dual_ffd_records(level.upper_columns, level.lower_columns) == (
        level.dv_records
    )


def test_bspline_level_config_preserves_native_blending_and_order(tmp_path, monkeypatch):
    bootstrap = _write_bootstrap_mesh(tmp_path / "bootstrap.su2")
    dual = tmp_path / "dual_bspline.su2"
    _split(
        bootstrap,
        dual,
        output_blending="BSPLINE_UNIFORM",
        bspline_orders=(4, 2, 2),
    )
    config = _dual_config(
        MESH_FILENAME=str(dual),
        MESH_OUT_FILENAME="mesh_out",
        FFD_BLENDING="BSPLINE_UNIFORM",
        FFD_BSPLINE_ORDER="( 4, 2, 2 )",
    )
    config._filename = str(tmp_path / "Config_FFD.cfg")
    opts = _dual_opts(config)
    opts["ffd_active_xmin"] = 0.0
    opts["ffd_active_xmax"] = 1.0
    level = build_initial_ffd_level(config, opts)

    monkeypatch.chdir(tmp_path)
    cfg_path = Path(write_ffd_level_config(config, level, opts))
    cfg_text = cfg_path.read_text()
    assert "FFD_BLENDING= BSPLINE_UNIFORM" in cfg_text
    assert "FFD_BSPLINE_ORDER= 4, 2, 2" in cfg_text
    mesh_text = (tmp_path / "LEVEL_0" / "ffd_level0.su2").read_text()
    assert mesh_text.count("FFD_BLENDING= BSPLINE_UNIFORM") == 2
    assert mesh_text.count("BSPLINE_ORDER_I= 4") == 2


def test_half_upper_single_box_level_uses_canonical_outward_definition(
    tmp_path, monkeypatch
):
    bootstrap = _write_bootstrap_mesh(
        tmp_path / "bootstrap.su2",
        points=HALF_UPPER_POINTS,
        marker_closed=False,
    )
    prepared = tmp_path / "half_upper.su2"
    build_single_surface_ffd_box(
        bootstrap,
        prepared,
        bootstrap_tag="BOOTSTRAP_BOX",
        marker="AIRFOIL",
        side="upper",
        offset_chord=0.04,
        box_tag="UPPER_BOX",
        overwrite=True,
    )
    config = _half_upper_config(
        MESH_FILENAME=str(prepared),
        MESH_OUT_FILENAME="mesh_out",
    )
    config._filename = str(tmp_path / "Config_FFD.cfg")
    opts = _dual_opts(config)
    assert opts["ffd_dual_box"] is False
    level = build_initial_ffd_level(config, opts)
    assert level.dual_box is False
    assert level.ndv == 3
    assert level.dv_records == [
        ("UPPER", 0.25),
        ("UPPER", 0.5),
        ("UPPER", 0.75),
    ]

    monkeypatch.chdir(tmp_path)
    cfg_path = Path(write_ffd_level_config(config, level, opts))
    cfg_text = cfg_path.read_text()
    assert cfg_text.count("( 19") == 3
    assert "UPPER_BOX" in cfg_text
    assert "FFD_CONTINUITY= USER_INPUT" in cfg_text

    mesh_text = (tmp_path / "LEVEL_0" / "ffd_level0.su2").read_text()
    assert "FFD_NBOX= 1" in mesh_text
    assert "FFD_TAG= UPPER_BOX" in mesh_text
    assert "FFD_TAG= LOWER_BOX" not in mesh_text


@pytest.mark.parametrize("blending", ["BEZIER", "BSPLINE_UNIFORM"])
def test_half_upper_level_transition_preserves_curved_box_and_physical_mesh(
    tmp_path,
    monkeypatch,
    blending,
):
    bootstrap = _write_bootstrap_mesh(
        tmp_path / "bootstrap.su2",
        points=HALF_UPPER_POINTS,
        marker_closed=False,
    )
    prepared = tmp_path / "half_upper.su2"
    build_single_surface_ffd_box(
        bootstrap,
        prepared,
        bootstrap_tag="BOOTSTRAP_BOX",
        marker="AIRFOIL",
        side="UPPER",
        offset_chord=0.04,
        box_tag="UPPER_BOX",
        overwrite=True,
        output_blending=blending,
        bspline_orders=(4, 2, 2),
    )
    config = _half_upper_config(
        MESH_FILENAME=str(prepared),
        MESH_OUT_FILENAME="mesh_out",
        FFD_BLENDING=blending,
        FFD_BSPLINE_ORDER="( 4, 2, 2 )",
    )
    config._filename = str(tmp_path / "Config_FFD.cfg")
    opts = _dual_opts(config)

    monkeypatch.chdir(tmp_path)
    level0 = build_initial_ffd_level(config, opts)
    write_ffd_level_config(config, level0, opts)
    level0_mesh = tmp_path / "LEVEL_0" / "ffd_level0.su2"

    level1 = FFDLevel(
        level_id=1,
        columns=[0.25, 0.4, 0.5, 0.75],
        workdir="LEVEL_1",
        config_filename="config_level1.cfg",
        project_filename="project_level1.pkl",
        mesh_source=str(level0_mesh),
        initial_mesh_source=str(prepared),
        dv_values=[0.0] * 4,
        ffd_box_tag="UPPER_BOX",
        upper_box_tag="UPPER_BOX",
        lower_box_tag="LOWER_BOX",
        ffd_dv_kind="FFD_CONTROL_POINT_2D",
        marker="AIRFOIL",
        domain_mode="HALF_UPPER",
        control_row=1,
        direction="OUTWARD",
        side="UPPER",
        active_xmin=0.0,
        active_xmax=1.0,
    )
    cfg_path = Path(write_ffd_level_config(config, level1, opts))
    level1_mesh = tmp_path / "LEVEL_1" / "ffd_level1.su2"

    spec = read_single_ffd_box_spec(str(level1_mesh), "UPPER_BOX")
    assert spec["columns"] == pytest.approx(
        [-0.1, 0.25, 0.4, 0.5, 0.75, 1.1]
    )
    assert spec["blending"] == blending
    assert spec["bspline_orders"] == list(
        (4, 2, 2) if blending == "BSPLINE_UNIFORM" else (2, 2, 2)
    )
    coordinate_error, _ = _max_mesh_coordinate_difference(
        str(level0_mesh),
        str(level1_mesh),
    )
    assert coordinate_error <= 1.0e-12
    cfg_level1 = SU2.io.Config(str(cfg_path))
    assert sum(cfg_level1["DEFINITION_DV"]["SIZE"]) == 4
    assert cfg_level1["DEFINITION_DV"]["FFDTAG"] == ["UPPER_BOX"] * 4


@pytest.mark.parametrize("blending", ["BEZIER", "BSPLINE_UNIFORM"])
@pytest.mark.parametrize("domain_mode", ["FULL", "HALF_UPPER"])
def test_progressive_ffd_thickness_value_and_analytic_gradient_are_compatible(
    tmp_path,
    monkeypatch,
    blending,
    domain_mode,
):
    bspline_orders = (4, 2, 2)
    if domain_mode == "FULL":
        bootstrap = _write_bootstrap_mesh(tmp_path / "bootstrap_full.su2")
        prepared = tmp_path / "prepared_full.su2"
        _split(
            bootstrap,
            prepared,
            output_blending=blending,
            bspline_orders=bspline_orders,
        )
        config = _dual_config(
            MESH_FILENAME=str(prepared),
            FFD_BLENDING=blending,
            FFD_BSPLINE_ORDER="( 4, 2, 2 )",
            PROGRESSIVE_THICKNESS_CONSTRAINT="YES",
            PROGRESSIVE_THICKNESS_DOMAIN_MODE="FULL",
        )
    else:
        bootstrap = _write_bootstrap_mesh(
            tmp_path / "bootstrap_half.su2",
            points=HALF_UPPER_POINTS,
            marker_closed=False,
        )
        prepared = tmp_path / "prepared_half.su2"
        build_single_surface_ffd_box(
            bootstrap,
            prepared,
            bootstrap_tag="BOOTSTRAP_BOX",
            marker="AIRFOIL",
            side="UPPER",
            offset_chord=0.04,
            box_tag="UPPER_BOX",
            overwrite=True,
            output_blending=blending,
            bspline_orders=bspline_orders,
        )
        config = _half_upper_config(
            MESH_FILENAME=str(prepared),
            FFD_BLENDING=blending,
            FFD_BSPLINE_ORDER="( 4, 2, 2 )",
            PROGRESSIVE_THICKNESS_CONSTRAINT="YES",
            PROGRESSIVE_THICKNESS_DOMAIN_MODE="HALF_UPPER",
            PROGRESSIVE_THICKNESS_SYMMETRY_Y=0.0,
        )

    config._filename = str(tmp_path / "Config_FFD.cfg")
    opts = _dual_opts(config)
    assert opts["ffd_thickness_ikkt_included"] is False
    level = build_initial_ffd_level(config, opts)

    monkeypatch.chdir(tmp_path)
    cfg_path = Path(write_ffd_level_config(config, level, opts))
    cfg_level = SU2.io.Config(str(cfg_path))
    level_mesh = (tmp_path / "LEVEL_0" / "ffd_level0.su2").resolve()
    cfg_level["MESH_FILENAME"] = str(level_mesh)
    project = SimpleNamespace(config=cfg_level)

    x_stations = [0.5]
    reference = _section_measure_from_segments(
        str(level_mesh),
        "AIRFOIL",
        x_stations,
        domain_mode=domain_mode,
        symmetry_y=0.0,
    )
    constraint = ThicknessConstraint(
        ref_mesh=str(prepared),
        marker="AIRFOIL",
        x_stations=x_stations,
        reference_measure=reference,
        gradient_mode="ANALYTIC",
        domain_mode=domain_mode,
        symmetry_y=0.0,
    )

    values = constraint.values([0.0] * level.ndv, project)
    cfg_level["OPT_RELAX_FACTOR"] = 1.0
    jacobian_unit_relax = constraint.jacobian_analytic(
        [0.0] * level.ndv,
        project,
    )
    cfg_level["OPT_RELAX_FACTOR"] = 37.0
    jacobian = constraint.jacobian_analytic([0.0] * level.ndv, project)
    assert values == pytest.approx([0.0], abs=1.0e-13)
    assert jacobian.shape == (1, level.ndv)
    assert jacobian == pytest.approx(jacobian_unit_relax * 37.0)
    assert all(float(value) >= -1.0e-13 for value in jacobian[0])
    assert max(float(value) for value in jacobian[0]) > 0.0


def test_smoke_visualizations_are_persisted_in_ffd_prep(tmp_path):
    run_dir = tmp_path / "stage"
    prep_dir = tmp_path / "FFD_PREP"
    run_dir.mkdir()
    smoke_log = run_dir / "ffd_zero_smoke.log"
    smoke_log.write_text("Exit Success\n")

    visualizations = {}
    for key, filename in _SMOKE_VISUALIZATION_FILENAMES.items():
        source = run_dir / filename
        source.write_text(f"artifact={key}\n")
        visualizations[key] = str(source)

    prepared_mesh = tmp_path / "prepared_dual.su2"
    artifacts = _persist_smoke_artifacts(
        {
            "log": str(smoke_log),
            "visualizations": visualizations,
        },
        str(prep_dir),
        prepared_mesh=str(prepared_mesh),
        marker="AIRFOIL",
        other_markers=["FARFIELD"],
        symmetry_markers=[],
        box_tag="UPPER_BOX",
        control_row=1,
        direction_y=1.0,
        box_count=2,
    )

    for key, filename in _SMOKE_VISUALIZATION_FILENAMES.items():
        destination = prep_dir / filename
        assert destination.read_text() == f"artifact={key}\n"
        assert artifacts[key] == str(destination.resolve())

    smoke_cfg = (prep_dir / "ffd_zero_smoke.cfg").read_text()
    assert f"MESH_FILENAME= {prepared_mesh.resolve()}" in smoke_cfg
    assert "DV_PARAM= ( UPPER_BOX, 1, 1, 0.0, 1.0 )" in smoke_cfg
    assert "FFD_CONTINUITY= USER_INPUT" in smoke_cfg
