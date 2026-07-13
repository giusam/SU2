import csv
import os

import pytest

from tools.compare_progressive_ffd_scores import (
    _parse_override,
    infer_active_columns,
    summarize_scoring,
)
from SU2.opt.progressive_hh_levels import append_selection_history_csv
from tests.test_progressive_ffd_split import _split, _write_bootstrap_mesh


def test_parse_override_requires_key_value():
    assert _parse_override("PROGRESSIVE_FFD_SCORING_MODE=VIRTUAL_TANGENT") == (
        "PROGRESSIVE_FFD_SCORING_MODE",
        "VIRTUAL_TANGENT",
    )
    with pytest.raises(Exception, match="KEY=VALUE"):
        _parse_override("INVALID")


def test_infer_active_columns_uses_real_ffd_control_indices(tmp_path):
    bootstrap = _write_bootstrap_mesh(tmp_path / "bootstrap.su2")
    mesh = tmp_path / "dual.su2"
    _split(bootstrap, mesh)
    config = {
        "DEFINITION_DV": {
            "KIND": ["FFD_CONTROL_POINT_2D"] * 6,
            "FFDTAG": ["UPPER_BOX"] * 3 + ["LOWER_BOX"] * 3,
            "PARAM": [
                [0.0, 1.0, 1.0, 0.0, 1.0],
                [0.0, 2.0, 1.0, 0.0, 1.0],
                [0.0, 3.0, 1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.0, -1.0],
                [0.0, 2.0, 0.0, 0.0, -1.0],
                [0.0, 3.0, 0.0, 0.0, -1.0],
            ],
        }
    }
    active = infer_active_columns(
        config,
        mesh,
        {
            "ffd_upper_box_tag": "UPPER_BOX",
            "ffd_lower_box_tag": "LOWER_BOX",
        },
    )
    assert active == {
        "UPPER": [0.25, 0.5, 0.75],
        "LOWER": [0.25, 0.5, 0.75],
    }


def test_virtual_summary_rejects_dot_only_artifacts(tmp_path):
    result = {
        "scoring_basis": "VIRTUAL_TANGENT_SPACE",
        "ffd_blending": "BEZIER",
        "insertion_target": 1,
        "insertions_completed": 1,
        "selected_candidates": [
            {
                "insertion_step": 1,
                "side": "UPPER",
                "x": 0.25,
                "indicator": 1.0,
                "score_net": 1.0,
            }
        ],
        "raw_candidates": [],
    }
    summary = summarize_scoring("VIRTUAL_TANGENT", result, tmp_path)
    assert summary["dot_only_artifacts"] == []
    os.mkdir(tmp_path / "DOT_ONLY_DRAG_candidate")
    with pytest.raises(RuntimeError, match="unexpectedly generated"):
        summarize_scoring("VIRTUAL_TANGENT", result, tmp_path)


def test_selection_history_upgrades_legacy_header_before_append(tmp_path):
    path = tmp_path / "progressive_ffd_selection_history.csv"
    path.write_text("level_id,side,x,indicator\n0,UPPER,0.25,1.0\n")
    append_selection_history_csv(
        str(path),
        {
            "level_id": 1,
            "ndv_before": 3,
            "ndv_after": 4,
            "n_added": 1,
            "nadd_mode": "SEQUENTIAL_GROWTH_RATIO",
            "trigger_mode": "MAX_ITER",
            "refinement": "ADAPTIVE",
            "ffd_scoring_mode": "VIRTUAL_TANGENT",
            "selected": [
                {
                    "side": "UPPER",
                    "x": 0.5,
                    "indicator": 2.0,
                    "indicator_ratio_to_best": 1.0,
                }
            ],
        },
        {"history_file": "history.csv", "final_mesh": "mesh.su2"},
    )
    with open(path, newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    assert "ffd_scoring_mode" in fieldnames
    assert "score_net" in fieldnames
    assert rows[0]["level_id"] == "0"
    assert rows[0]["x"] == "0.25"
    assert rows[1]["ffd_scoring_mode"] == "VIRTUAL_TANGENT"
