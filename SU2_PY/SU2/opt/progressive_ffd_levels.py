#!/usr/bin/env python

import copy
import os

import SU2

from SU2.opt.thickness_constraint import clean_progressive_thickness_keys
from SU2.opt.progressive_ffd_core import (
    FFDLevel,
    apply_post_opt_ffd_spring,
    build_ffd_mesh_columns,
    ffd_active_range_from_opts,
    initial_ffd_columns_from_config,
    make_ffd_config_dump_compatible,
    make_dual_ffd_definition,
    make_ffd_definition,
    ordered_dual_ffd_records,
    refine_ffd_columns,
)
from SU2.opt.progressive_ffd_mesh import rewrite_ffd_box_with_columns_and_reembed
from SU2.opt.progressive_ffd_split import (
    rewrite_dual_ffd_boxes_with_columns_and_reembed,
)
from SU2.opt.progressive_hh_levels import (
    _remove_progressive_keys as _remove_hh_progressive_keys,
    _resolve_from_cfg_dir,
)


def _remove_ffd_progressive_keys(cfg):
    _remove_hh_progressive_keys(cfg)
    progressive_keys = [
        "PROGRESSIVE_PARAM_KIND",
        "PROGRESSIVE_FFD_DV_KIND",
        "PROGRESSIVE_FFD_BOX_TAG",
        "PROGRESSIVE_FFD_MARKER",
        "PROGRESSIVE_FFD_DOMAIN_MODE",
        "PROGRESSIVE_FFD_CONTROL_ROW",
        "PROGRESSIVE_FFD_DIRECTION",
        "PROGRESSIVE_FFD_INITIAL_COLUMNS",
        "PROGRESSIVE_FFD_ALLOW_EXTERNAL_COLUMNS",
        "PROGRESSIVE_FFD_ACTIVE_XMIN",
        "PROGRESSIVE_FFD_ACTIVE_XMAX",
        "PROGRESSIVE_FFD_DUAL_BOX",
        "PROGRESSIVE_FFD_AUTO_PREPARE",
        "PROGRESSIVE_FFD_PREPARE_ONLY",
        "PROGRESSIVE_FFD_PREPARED_MESH",
        "PROGRESSIVE_FFD_PREPARE_OVERWRITE",
        "PROGRESSIVE_FFD_PREPARE_SMOKE_TEST",
        "PROGRESSIVE_FFD_BOOTSTRAP_TAG",
        "PROGRESSIVE_FFD_BOOTSTRAP_Y_PADDING_CHORD",
        "PROGRESSIVE_FFD_UPPER_BOX_TAG",
        "PROGRESSIVE_FFD_LOWER_BOX_TAG",
        "PROGRESSIVE_FFD_UPPER_OFFSET_CHORD",
        "PROGRESSIVE_FFD_LOWER_OFFSET_CHORD",
        "PROGRESSIVE_FFD_REFINEMENT_COUPLING",
    ]
    for key in progressive_keys:
        if key in cfg:
            del cfg[key]
    clean_progressive_thickness_keys(cfg)


def build_initial_ffd_level(base_config, opts):
    if "MESH_FILENAME" not in base_config or not base_config["MESH_FILENAME"]:
        raise ValueError("PROGRESSIVE_PARAM_KIND=FFD requires MESH_FILENAME")

    initial_mesh = _resolve_from_cfg_dir(base_config, base_config["MESH_FILENAME"])
    columns = initial_ffd_columns_from_config(base_config, opts)

    print(f"[PROGRESSIVE_FFD] Initial columns = {columns}")
    active_xmin, active_xmax = ffd_active_range_from_opts(opts)

    if opts.get("ffd_dual_box", False):
        print(
            "[PROGRESSIVE_FFD_DUAL] Initial DV count | "
            f"upper={len(columns)} lower={len(columns)} total={2 * len(columns)}"
        )
        return FFDLevel(
            level_id=0,
            columns=columns,
            upper_columns=columns,
            lower_columns=columns,
            dual_box=True,
            workdir="LEVEL_0",
            config_filename="config_level0.cfg",
            project_filename="project_level0.pkl",
            mesh_source=initial_mesh,
            initial_mesh_source=initial_mesh,
            dv_values=[0.0] * (2 * len(columns)),
            ffd_box_tag="",
            upper_box_tag=opts["ffd_upper_box_tag"],
            lower_box_tag=opts["ffd_lower_box_tag"],
            ffd_dv_kind=opts["ffd_dv_kind"],
            marker=opts["ffd_marker"],
            domain_mode=opts["ffd_domain_mode"],
            control_row=None,
            direction="OUTWARD",
            active_xmin=active_xmin,
            active_xmax=active_xmax,
            active_include_bounds=False,
        )

    return FFDLevel(
        level_id=0,
        columns=columns,
        workdir="LEVEL_0",
        config_filename="config_level0.cfg",
        project_filename="project_level0.pkl",
        mesh_source=initial_mesh,
        initial_mesh_source=initial_mesh,
        dv_values=[0.0] * len(columns),
        ffd_box_tag=opts["ffd_box_tag"],
        ffd_dv_kind=opts["ffd_dv_kind"],
        marker=opts["ffd_marker"],
        domain_mode=opts["ffd_domain_mode"],
        control_row=opts.get("ffd_control_row", None),
        direction=opts["ffd_direction"],
        active_xmin=active_xmin,
        active_xmax=active_xmax,
        active_include_bounds=opts.get("ffd_active_include_bounds", False),
    )


def _ffdtype_level_kwargs(opts):
    active_xmin, active_xmax = ffd_active_range_from_opts(opts)
    if opts.get("ffd_dual_box", False):
        return {
            "dual_box": True,
            "ffd_box_tag": "",
            "upper_box_tag": opts["ffd_upper_box_tag"],
            "lower_box_tag": opts["ffd_lower_box_tag"],
            "ffd_dv_kind": opts["ffd_dv_kind"],
            "marker": opts["ffd_marker"],
            "domain_mode": opts["ffd_domain_mode"],
            "control_row": None,
            "direction": "OUTWARD",
            "active_xmin": active_xmin,
            "active_xmax": active_xmax,
            "active_include_bounds": False,
        }
    return {
        "ffd_box_tag": opts["ffd_box_tag"],
        "ffd_dv_kind": opts["ffd_dv_kind"],
        "marker": opts["ffd_marker"],
        "domain_mode": opts["ffd_domain_mode"],
        "control_row": opts.get("ffd_control_row", None),
        "direction": opts["ffd_direction"],
        "active_xmin": active_xmin,
        "active_xmax": active_xmax,
        "active_include_bounds": opts.get("ffd_active_include_bounds", False),
    }


def build_next_ffd_level(prev_level, result, opts):
    next_id = prev_level.level_id + 1
    columns = refine_ffd_columns(prev_level, result, opts)
    selection_metadata = opts.get("_last_selection_metadata", None)

    next_mesh = result.get("final_mesh", None)
    if next_mesh is None:
        next_mesh = prev_level.mesh_source

    if getattr(prev_level, "dual_box", False):
        upper_columns, lower_columns = columns
        return FFDLevel(
            level_id=next_id,
            columns=upper_columns,
            upper_columns=upper_columns,
            lower_columns=lower_columns,
            workdir=f"LEVEL_{next_id}",
            config_filename=f"config_level{next_id}.cfg",
            project_filename=f"project_level{next_id}.pkl",
            mesh_source=next_mesh,
            initial_mesh_source=getattr(prev_level, "initial_mesh_source", None),
            dv_values=[0.0] * (len(upper_columns) + len(lower_columns)),
            selection_metadata=selection_metadata,
            post_opt_spring_pending=False,
            spring_reallocated=False,
            **_ffdtype_level_kwargs(opts),
        )

    return FFDLevel(
        level_id=next_id,
        columns=columns,
        workdir=f"LEVEL_{next_id}",
        config_filename=f"config_level{next_id}.cfg",
        project_filename=f"project_level{next_id}.pkl",
        mesh_source=next_mesh,
        initial_mesh_source=getattr(prev_level, "initial_mesh_source", None),
        dv_values=[0.0] * len(columns),
        selection_metadata=selection_metadata,
        post_opt_spring_pending=bool(
            selection_metadata
            and selection_metadata.get("post_opt_spring_pending", False)
        ),
        spring_reallocated=False,
        **_ffdtype_level_kwargs(opts),
    )


def build_ffd_spring_reallocated_level(prev_level, result, opts, reoptimize=True):
    if getattr(prev_level, "dual_box", False):
        raise NotImplementedError("Spring redistribution is unavailable in dual FFD mode")
    columns = apply_post_opt_ffd_spring(prev_level, result, opts)
    if columns is None:
        print(
            "[PROGRESSIVE_FFD][SPRING] WARNING: could not build "
            "spring-reallocated level"
        )
        return None

    next_mesh = result.get("final_mesh", None)
    if next_mesh is None:
        next_mesh = prev_level.mesh_source

    spring_metadata = {
        "level_id": prev_level.level_id,
        "ndv": prev_level.ndv,
        "spring_timing": opts.get("spring_timing", "POST_OPT"),
        "spring_score_mode": opts.get("spring_score_mode", "COEFFICIENT"),
        "spring_post_action": opts.get("spring_post_action", "REOPTIMIZE"),
        "post_spring_optimization_skipped": not reoptimize,
        "columns_before": sorted(prev_level.columns),
        "columns_after": sorted(columns),
        "upper_before": sorted(prev_level.columns),
        "lower_before": [],
        "upper_after": sorted(columns),
        "lower_after": [],
        "column_coeff_abs": result.get("spring_ffd_coeff_abs", []),
        "history_file": result.get("history_file"),
        "final_mesh": result.get("final_mesh"),
    }

    if not reoptimize:
        return FFDLevel(
            level_id=prev_level.level_id,
            columns=columns,
            workdir=prev_level.workdir,
            config_filename=prev_level.config_filename,
            project_filename=prev_level.project_filename,
            mesh_source=next_mesh,
            initial_mesh_source=getattr(prev_level, "initial_mesh_source", None),
            dv_values=[0.0] * len(columns),
            selection_metadata=spring_metadata,
            post_opt_spring_pending=False,
            spring_reallocated=True,
            **_ffdtype_level_kwargs(opts),
        )

    next_id = prev_level.level_id + 1
    return FFDLevel(
        level_id=next_id,
        columns=columns,
        workdir=f"LEVEL_{next_id}",
        config_filename=f"config_level{next_id}.cfg",
        project_filename=f"project_level{next_id}.pkl",
        mesh_source=next_mesh,
        initial_mesh_source=getattr(prev_level, "initial_mesh_source", None),
        dv_values=[0.0] * len(columns),
        selection_metadata=spring_metadata,
        post_opt_spring_pending=False,
        spring_reallocated=True,
        **_ffdtype_level_kwargs(opts),
    )


def _ffd_mesh_basename(level):
    suffix = "_spring" if getattr(level, "spring_reallocated", False) else ""
    return f"ffd_level{level.level_id}{suffix}.su2"


def _prepare_ffd_mesh(cfg, level, opts):
    if not level.mesh_source:
        raise ValueError("Progressive FFD level has no mesh_source")

    src_mesh = os.path.abspath(level.mesh_source)
    os.makedirs(level.workdir, exist_ok=True)
    mesh_basename = _ffd_mesh_basename(level)
    dst_mesh = os.path.join(level.workdir, mesh_basename)

    print(
        "[PROGRESSIVE_FFD] Rewriting FFD mesh | "
        f"source={src_mesh} output={dst_mesh}"
    )
    if getattr(level, "dual_box", False):
        upper_mesh_columns, upper_active = build_ffd_mesh_columns(
            src_mesh,
            opts["ffd_upper_box_tag"],
            active_columns=level.upper_columns,
            opts=opts,
        )
        lower_mesh_columns, lower_active = build_ffd_mesh_columns(
            src_mesh,
            opts["ffd_lower_box_tag"],
            active_columns=level.lower_columns,
            opts=opts,
        )
        mesh_info = rewrite_dual_ffd_boxes_with_columns_and_reembed(
            src_mesh,
            dst_mesh,
            marker=opts["ffd_marker"],
            upper_tag=opts["ffd_upper_box_tag"],
            lower_tag=opts["ffd_lower_box_tag"],
            upper_columns=upper_mesh_columns,
            lower_columns=lower_mesh_columns,
            upper_offset_chord=opts["ffd_upper_offset_chord"],
            lower_offset_chord=opts["ffd_lower_offset_chord"],
            diagnostics_csv=False,
            overwrite=True,
        )
        mesh_info["active_columns_by_side"] = {
            "UPPER": upper_active,
            "LOWER": lower_active,
        }
        mesh_info["column_index_by_side"] = {
            "UPPER": mesh_info["upper_column_index_by_x"],
            "LOWER": mesh_info["lower_column_index_by_x"],
        }
        cfg["MESH_FILENAME"] = mesh_basename
        if "MULTIPOINT_MESH_FILENAME" in cfg and cfg["MULTIPOINT_MESH_FILENAME"]:
            cfg["MULTIPOINT_MESH_FILENAME"] = f"({mesh_basename})"
        return mesh_info

    mesh_columns, active_columns = build_ffd_mesh_columns(
        src_mesh,
        opts["ffd_box_tag"],
        active_columns=level.columns,
        opts=opts,
    )
    mesh_info = rewrite_ffd_box_with_columns_and_reembed(
        src_mesh,
        dst_mesh,
        box_tag=opts["ffd_box_tag"],
        new_columns=mesh_columns,
        marker_name=opts["ffd_marker"],
        domain_mode=opts["ffd_domain_mode"],
    )
    mesh_info["active_columns"] = active_columns

    cfg["MESH_FILENAME"] = mesh_basename
    if "MULTIPOINT_MESH_FILENAME" in cfg and cfg["MULTIPOINT_MESH_FILENAME"]:
        cfg["MULTIPOINT_MESH_FILENAME"] = f"({mesh_basename})"

    return mesh_info


def _validate_ffd_definition_request(level, opts, mesh_info):
    if getattr(level, "dual_box", False):
        expected = len(level.upper_columns) + len(level.lower_columns)
        if level.ndv != expected:
            raise RuntimeError("Unexpected dual FFD NDV/column mismatch")
        for side, columns in level.columns_by_side.items():
            mapping = mesh_info["column_index_by_side"][side]
            for x in columns:
                if not any(abs(float(x) - float(key)) <= 1.0e-10 for key in mapping):
                    raise RuntimeError(
                        f"Dual FFD {side} column {x} is missing from rewritten mesh"
                    )
        return

    ffd_dv_kind = str(opts.get("ffd_dv_kind", "FFD_CONTROL_POINT_2D")).upper()
    if ffd_dv_kind == "FFD_CONTROL_POINT_2D":
        control_row = int(opts.get("ffd_control_row"))
        n_rows = len(mesh_info.get("y_rows", []))
        if control_row >= n_rows:
            raise ValueError(
                "PROGRESSIVE_FFD_CONTROL_ROW is outside the rewritten FFD grid: "
                f"row={control_row}, available rows=0..{max(0, n_rows - 1)}"
            )

    if level.ndv != len(level.columns):
        raise RuntimeError("Unexpected FFD NDV/column mismatch")


def write_ffd_level_config(base_config, level, opts):
    cfg = SU2.io.Config(copy.deepcopy(dict(base_config)))
    _remove_ffd_progressive_keys(cfg)

    nfinal = opts.get("nfinal", None)
    is_final_by_nlevels = (
        nfinal is None and level.level_id >= opts["nlevels"] - 1
    )
    is_final_by_nfinal = nfinal is not None and level.ndv >= int(nfinal)

    if is_final_by_nlevels or is_final_by_nfinal:
        cfg["OPT_ITERATIONS"] = int(base_config["OPT_ITERATIONS"])
    else:
        cfg["OPT_ITERATIONS"] = opts["max_iter_per_level"]

    mesh_info = _prepare_ffd_mesh(cfg, level, opts)
    _validate_ffd_definition_request(level, opts, mesh_info)

    if getattr(level, "dual_box", False):
        cfg["DEFINITION_DV"] = make_dual_ffd_definition(
            ordered_dual_ffd_records(
                level.upper_columns,
                level.lower_columns,
            ),
            opts,
            mesh_info["column_index_by_side"],
        )
        cfg["FFD_CONTINUITY"] = "USER_INPUT"
        for key in ("FFD_FIX_I", "FFD_FIX_J", "FFD_FIX_K"):
            if key in cfg:
                del cfg[key]
    else:
        cfg["DEFINITION_DV"] = make_ffd_definition(
            level.columns,
            opts,
            mesh_info["column_index_by_x"],
        )
    cfg["DV_MARKER"] = str(opts["ffd_marker"])
    cfg["DV_KIND"] = str(opts["ffd_dv_kind"])
    cfg["FFD_BLENDING"] = str(opts.get("ffd_blending", "BEZIER"))
    cfg["FFD_BSPLINE_ORDER"] = ", ".join(
        str(int(value)) for value in opts.get("ffd_bspline_orders", (2, 2, 2))
    )

    dv_values = getattr(level, "dv_values", None)
    if dv_values is None or len(dv_values) != level.ndv:
        if dv_values is not None:
            print(
                "[PROGRESSIVE_FFD] WARNING: invalid level.dv_values length; "
                "falling back to zero initial FFD values"
            )
        dv_values = [0.0] * level.ndv
    else:
        dv_values = [float(v) for v in dv_values]

    cfg["DV_VALUE_NEW"] = dv_values
    cfg["DV_VALUE_OLD"] = [0.0] * level.ndv

    if "RESTART_FILENAME" in cfg and cfg["RESTART_FILENAME"]:
        cfg["RESTART_FILENAME"] = _resolve_from_cfg_dir(
            base_config, cfg["RESTART_FILENAME"]
        )

    mesh_out_base = "mesh_out"
    if "MESH_OUT_FILENAME" in cfg and cfg["MESH_OUT_FILENAME"]:
        mesh_out_base = str(cfg["MESH_OUT_FILENAME"])
        if mesh_out_base.endswith(".su2"):
            mesh_out_base = mesh_out_base[:-4]
        mesh_out_base = os.path.basename(mesh_out_base)
    cfg["MESH_OUT_FILENAME"] = mesh_out_base

    if getattr(level, "dual_box", False):
        print(
            "[PROGRESSIVE_FFD_DUAL] Active DV definition | "
            f"ndv={level.ndv} upper={len(level.upper_columns)} "
            f"lower={len(level.lower_columns)}"
        )
        print(
            "[PROGRESSIVE_FFD_DUAL] Upper columns: "
            f"{[round(float(x), 6) for x in level.upper_columns]}"
        )
        print(
            "[PROGRESSIVE_FFD_DUAL] Lower columns: "
            f"{[round(float(x), 6) for x in level.lower_columns]}"
        )
    else:
        print(
            "[PROGRESSIVE_FFD] Active DV definition | "
            f"kind={opts['ffd_dv_kind']} ndv={level.ndv} "
            f"columns={[round(float(x), 6) for x in level.columns]}"
        )

    os.makedirs(level.workdir, exist_ok=True)
    out_cfg = os.path.join(level.workdir, level.config_filename)
    make_ffd_config_dump_compatible(cfg).dump(out_cfg)
    return out_cfg
