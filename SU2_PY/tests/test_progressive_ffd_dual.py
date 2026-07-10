from pathlib import Path

import pytest

import SU2
from SU2.opt.progressive_ffd_core import (
    FFDLevel,
    get_progressive_ffd_options,
    make_dual_ffd_definition,
    ordered_dual_ffd_records,
    refine_ffd_columns,
    select_ffd_candidates_by_nadd_mode,
)
from SU2.opt.progressive_ffd_levels import (
    build_initial_ffd_level,
    write_ffd_level_config,
)
from SU2.opt.progressive_ffd_prepare import (
    _SMOKE_VISUALIZATION_FILENAMES,
    _persist_smoke_artifacts,
)
from SU2.opt.progressive_hh_core import get_progressive_hh_options
from tests.test_progressive_ffd_split import _split, _write_bootstrap_mesh


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


def _legacy_config(**overrides):
    values = dict(_dual_config())
    for key in (
        "PROGRESSIVE_FFD_DUAL_BOX",
        "PROGRESSIVE_FFD_AUTO_PREPARE",
        "PROGRESSIVE_FFD_PREPARE_ONLY",
        "PROGRESSIVE_FFD_UPPER_BOX_TAG",
        "PROGRESSIVE_FFD_LOWER_BOX_TAG",
        "PROGRESSIVE_FFD_BOOTSTRAP_TAG",
        "PROGRESSIVE_FFD_BOOTSTRAP_Y_PADDING_CHORD",
        "PROGRESSIVE_FFD_UPPER_OFFSET_CHORD",
        "PROGRESSIVE_FFD_LOWER_OFFSET_CHORD",
        "PROGRESSIVE_FFD_REFINEMENT_COUPLING",
    ):
        values.pop(key, None)
    values.update(
        {
            "PROGRESSIVE_HH_NFINAL": 4,
            "PROGRESSIVE_FFD_BOX_TAG": "BOOTSTRAP_BOX",
            "PROGRESSIVE_FFD_CONTROL_ROW": 1,
            "PROGRESSIVE_FFD_DIRECTION": "Y",
        }
    )
    values.update(overrides)
    return SU2.io.Config(values)


def test_dual_nfinal_counts_upper_and_lower_dvs():
    config = _dual_config(PROGRESSIVE_HH_NFINAL=5)
    with pytest.raises(ValueError, match="initial HH NDV.*6"):
        get_progressive_hh_options(config)

    config = _dual_config(PROGRESSIVE_HH_NFINAL=8)
    opts = _dual_opts(config)
    assert opts["nfinal"] == 8
    assert opts["ffd_initial_columns"] == pytest.approx([0.25, 0.5, 0.75])


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

    with pytest.raises(NotImplementedError, match="only in dual-box mode"):
        _dual_opts(
            _legacy_config(
                FFD_BLENDING="BSPLINE_UNIFORM",
                FFD_BSPLINE_ORDER="( 3, 2, 2 )",
            )
        )


def test_dual_mode_rejects_legacy_single_box_options_and_spring():
    config = _dual_config(PROGRESSIVE_FFD_CONTROL_ROW=1)
    hh_opts = get_progressive_hh_options(config)
    with pytest.raises(ValueError, match="Legacy single-box options"):
        get_progressive_ffd_options(config, hh_opts)

    config = _dual_config(PROGRESSIVE_HH_SPRING="YES")
    hh_opts = get_progressive_hh_options(config)
    with pytest.raises(NotImplementedError, match="SPRING"):
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

    def fake_scores(_level, _opts):
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


def test_legacy_single_box_level_remains_unchanged(tmp_path, monkeypatch):
    bootstrap = _write_bootstrap_mesh(tmp_path / "bootstrap.su2")
    config = _legacy_config(
        MESH_FILENAME=str(bootstrap),
        MESH_OUT_FILENAME="mesh_out",
    )
    config._filename = str(tmp_path / "Config_FFD.cfg")
    opts = _dual_opts(config)
    assert opts["ffd_dual_box"] is False
    level = build_initial_ffd_level(config, opts)
    assert level.dual_box is False
    assert level.ndv == 3

    monkeypatch.chdir(tmp_path)
    cfg_path = Path(write_ffd_level_config(config, level, opts))
    cfg_text = cfg_path.read_text()
    assert cfg_text.count("( 19") == 3
    assert "BOOTSTRAP_BOX" in cfg_text
    assert "FFD_CONTINUITY= USER_INPUT" not in cfg_text

    mesh_text = (tmp_path / "LEVEL_0" / "ffd_level0.su2").read_text()
    assert "FFD_NBOX= 1" in mesh_text
    assert "FFD_TAG= BOOTSTRAP_BOX" in mesh_text
    assert "FFD_TAG= UPPER_BOX" not in mesh_text


def test_smoke_visualizations_are_persisted_in_ffd_prep(tmp_path):
    run_dir = tmp_path / "stage"
    prep_dir = tmp_path / "FFD_PREP"
    run_dir.mkdir()
    smoke_log = run_dir / "dual_zero_smoke.log"
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
        upper_tag="UPPER_BOX",
    )

    for key, filename in _SMOKE_VISUALIZATION_FILENAMES.items():
        destination = prep_dir / filename
        assert destination.read_text() == f"artifact={key}\n"
        assert artifacts[key] == str(destination.resolve())

    smoke_cfg = (prep_dir / "dual_zero_smoke.cfg").read_text()
    assert f"MESH_FILENAME= {prepared_mesh.resolve()}" in smoke_cfg
    assert "DV_PARAM= ( UPPER_BOX, 1, 1, 0.0, 1.0 )" in smoke_cfg
    assert "FFD_CONTINUITY= USER_INPUT" in smoke_cfg
