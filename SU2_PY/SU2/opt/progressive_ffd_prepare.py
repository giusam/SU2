#!/usr/bin/env python

"""Automatic raw-mesh preparation for progressive dual FFD."""

import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import tempfile

from SU2.opt.bspline_def import (
    BSplineDefError,
    extract_marker_nodes,
    infer_chord,
    read_su2_mesh,
)
from SU2.opt.progressive_ffd_core import validate_active_ffd_columns
from SU2.opt.progressive_ffd_mesh import (
    read_ffd_box_columns,
    rewrite_ffd_box_with_columns_and_reembed,
)
from SU2.opt.progressive_ffd_split import split_bootstrap_ffd_box
from SU2.opt.progressive_ffd_blending import BEZIER


class FFDPreparationError(RuntimeError):
    """Raised when automatic dual-FFD preparation cannot be completed safely."""


_SMOKE_VISUALIZATION_FILENAMES = {
    "upper_box_vtk": "ffd_boxes_0.vtk",
    "lower_box_vtk": "ffd_boxes_1.vtk",
    "upper_box_deformed_vtk": "ffd_boxes_def_0.vtk",
    "lower_box_deformed_vtk": "ffd_boxes_def_1.vtk",
    "surface_vtu": "surface_deformed.vtu",
}


def _config_directory(base_config):
    filename = getattr(base_config, "_filename", None)
    if filename:
        return os.path.dirname(os.path.abspath(filename))
    return os.getcwd()


def _resolve_from_config(base_config, filename):
    filename = str(filename)
    if os.path.isabs(filename):
        return os.path.abspath(filename)
    return os.path.abspath(os.path.join(_config_directory(base_config), filename))


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path, text):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".ffd_prepare_", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w") as fp:
            fp.write(text)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _atomic_write_json(path, payload):
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _format_marker_list(markers):
    return "( " + ", ".join(str(marker) for marker in markers) + " )"


def _mesh_geometry(mesh_path, marker):
    try:
        mesh = read_su2_mesh(mesh_path)
        marker_tag, node_ids, closed = extract_marker_nodes(mesh, marker)
        x_le, x_te, chord = infer_chord(
            mesh["points"],
            node_ids,
            {"mode": "auto", "x_le": None, "x_te": None},
        )
    except BSplineDefError as exc:
        raise FFDPreparationError(str(exc)) from exc
    if mesh["ndime"] != 2:
        raise FFDPreparationError("Progressive dual FFD preparation requires NDIME=2")
    if not closed:
        raise FFDPreparationError("Progressive dual FFD requires a closed marker")
    if not math.isfinite(chord) or chord <= 0.0:
        raise FFDPreparationError("The airfoil chord must be finite and positive")

    y_values = [float(mesh["points"][node_id][1]) for node_id in node_ids]
    if not y_values or not all(math.isfinite(value) for value in y_values):
        raise FFDPreparationError("The airfoil marker contains invalid y coordinates")
    return {
        "mesh": mesh,
        "marker_tag": marker_tag,
        "node_ids": node_ids,
        "x_le": float(x_le),
        "x_te": float(x_te),
        "chord": float(chord),
        "y_min": min(y_values),
        "y_max": max(y_values),
    }


def _max_mesh_coordinate_difference(reference_path, candidate_path):
    reference = read_su2_mesh(reference_path)
    candidate = read_su2_mesh(candidate_path)
    if reference["ndime"] != candidate["ndime"]:
        raise FFDPreparationError("Prepared mesh changed NDIME")
    if set(reference["points"]) != set(candidate["points"]):
        raise FFDPreparationError("Prepared mesh changed the physical point IDs")
    max_error = 0.0
    max_point_id = None
    for point_id in reference["points"]:
        error = math.sqrt(
            sum(
                (
                    float(reference["points"][point_id][index])
                    - float(candidate["points"][point_id][index])
                ) ** 2
                for index in range(reference["ndime"])
            )
        )
        if error > max_error:
            max_error = error
            max_point_id = point_id
    return max_error, max_point_id


def _write_bootstrap_config(
    path,
    *,
    mesh_in,
    mesh_out_base,
    marker,
    other_markers,
    bootstrap_tag,
    x_le,
    x_te,
    y_bottom,
    y_top,
):
    lines = [
        f"MESH_FILENAME= {mesh_in}",
        f"MESH_OUT_FILENAME= {mesh_out_base}",
        "MESH_FORMAT= SU2",
        f"MARKER_EULER= {_format_marker_list([marker])}",
    ]
    if other_markers:
        lines.append(f"MARKER_FAR= {_format_marker_list(other_markers)}")
    lines.extend(
        [
            "DV_KIND= FFD_SETTING",
            f"DV_MARKER= {_format_marker_list([marker])}",
            "DV_PARAM= ( 1.0 )",
            "DV_VALUE= 0.0",
            (
                f"FFD_DEFINITION= ( {bootstrap_tag}, "
                f"{x_le:.16g}, {y_bottom:.16g}, 0.0, "
                f"{x_te:.16g}, {y_bottom:.16g}, 0.0, "
                f"{x_te:.16g}, {y_top:.16g}, 0.0, "
                f"{x_le:.16g}, {y_top:.16g}, 0.0 )"
            ),
            "FFD_DEGREE= ( 1, 1, 0 )",
            "FFD_BLENDING= BEZIER",
            "FFD_TOLERANCE= 1E-10",
            "FFD_ITERATIONS= 500",
            "",
        ]
    )
    _atomic_write_text(path, "\n".join(lines))


def _write_zero_smoke_config(
    path,
    *,
    mesh_in,
    mesh_out_base,
    marker,
    other_markers,
    upper_tag,
    blending=BEZIER,
    bspline_orders=(2, 2, 2),
):
    lines = [
        "SOLVER= EULER",
        "MATH_PROBLEM= DIRECT",
        f"MESH_FILENAME= {mesh_in}",
        "MESH_FORMAT= SU2",
        f"MESH_OUT_FILENAME= {mesh_out_base}",
        f"MARKER_EULER= {_format_marker_list([marker])}",
    ]
    if other_markers:
        lines.append(f"MARKER_FAR= {_format_marker_list(other_markers)}")
    lines.extend(
        [
            f"MARKER_PLOTTING= {_format_marker_list([marker])}",
            f"MARKER_MONITORING= {_format_marker_list([marker])}",
            "DV_KIND= FFD_CONTROL_POINT_2D",
            f"DV_MARKER= {_format_marker_list([marker])}",
            f"DV_PARAM= ( {upper_tag}, 1, 1, 0.0, 1.0 )",
            "DV_VALUE= 0.0",
            "DEFORM_LINEAR_SOLVER= FGMRES",
            "DEFORM_LINEAR_SOLVER_PREC= ILU",
            "DEFORM_LINEAR_SOLVER_ITER= 1000",
            "DEFORM_NONLINEAR_ITER= 1",
            "DEFORM_LINEAR_SOLVER_ERROR= 1E-14",
            "DEFORM_CONSOLE_OUTPUT= YES",
            "DEFORM_COEFF= 1E6",
            "DEFORM_STIFFNESS_TYPE= WALL_DISTANCE",
            "FFD_TOLERANCE= 1E-10",
            "FFD_ITERATIONS= 500",
            "FFD_CONTINUITY= USER_INPUT",
            f"FFD_BLENDING= {blending}",
            (
                "FFD_BSPLINE_ORDER= "
                + ", ".join(str(int(value)) for value in bspline_orders)
            ),
            "OUTPUT_FILES= ( PARAVIEW, SURFACE_PARAVIEW )",
            "",
        ]
    )
    _atomic_write_text(path, "\n".join(lines))


def _resolve_su2_def():
    executable = shutil.which("SU2_DEF")
    if executable:
        return executable
    su2_run = os.environ.get("SU2_RUN", "")
    if su2_run:
        candidate = os.path.join(su2_run, "SU2_DEF")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    raise FFDPreparationError(
        "SU2_DEF was not found in PATH or in the SU2_RUN directory"
    )


def _su2_def_command(config_path, partitions):
    executable = _resolve_su2_def()
    partitions = max(1, int(partitions or 1))
    if partitions <= 1:
        return [executable, os.path.basename(config_path)]

    custom_mpi = os.environ.get("SU2_MPI_COMMAND", "").strip()
    if custom_mpi:
        try:
            rendered = custom_mpi % (partitions, executable)
        except Exception as exc:
            raise FFDPreparationError(
                "SU2_MPI_COMMAND must accept the SU2-style '%i' and '%s' placeholders"
            ) from exc
        return shlex.split(rendered) + [os.path.basename(config_path)]

    launcher = shutil.which("mpirun") or shutil.which("mpiexec")
    if launcher is None:
        raise FFDPreparationError(
            f"An MPI launcher is required for NUMBER_PART={partitions}"
        )
    return [
        launcher,
        "-n",
        str(partitions),
        executable,
        os.path.basename(config_path),
    ]


def _run_su2_def(config_path, partitions, log_path):
    command = _su2_def_command(config_path, partitions)
    with open(log_path, "w") as log:
        log.write("COMMAND: " + " ".join(shlex.quote(token) for token in command) + "\n\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=os.path.dirname(os.path.abspath(config_path)),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        try:
            with open(log_path, "r") as fp:
                tail = fp.readlines()[-40:]
        except OSError:
            tail = []
        raise FFDPreparationError(
            "SU2_DEF failed while preparing the dual FFD mesh. "
            f"See {log_path}.\n" + "".join(tail)
        )


def _expected_smoke_visualizations(prep_dir):
    return {
        key: os.path.abspath(os.path.join(prep_dir, filename))
        for key, filename in _SMOKE_VISUALIZATION_FILENAMES.items()
    }


def _run_zero_smoke_test(
    run_dir,
    *,
    mesh_in,
    source_mesh,
    marker,
    other_markers,
    upper_tag,
    partitions,
    tolerance,
    blending=BEZIER,
    bspline_orders=(2, 2, 2),
):
    smoke_base = os.path.join(run_dir, "dual_zero_smoke")
    smoke_mesh = smoke_base + ".su2"
    smoke_cfg = os.path.join(run_dir, "dual_zero_smoke.cfg")
    smoke_log = os.path.join(run_dir, "dual_zero_smoke.log")

    # A preceding FFD_SETTING call in the same staging directory may have
    # emitted a single bootstrap-box VTK. Remove all expected visualization
    # names so every retained artifact is known to come from the dual-box run.
    for filename in _SMOKE_VISUALIZATION_FILENAMES.values():
        path = os.path.join(run_dir, filename)
        if os.path.isfile(path):
            os.remove(path)

    _write_zero_smoke_config(
        smoke_cfg,
        mesh_in=mesh_in,
        mesh_out_base=smoke_base,
        marker=marker,
        other_markers=other_markers,
        upper_tag=upper_tag,
        blending=blending,
        bspline_orders=bspline_orders,
    )
    _run_su2_def(smoke_cfg, partitions, smoke_log)
    if not os.path.isfile(smoke_mesh):
        raise FFDPreparationError(f"SU2_DEF smoke test did not create {smoke_mesh}")

    smoke_error, smoke_point = _max_mesh_coordinate_difference(
        source_mesh,
        smoke_mesh,
    )
    if smoke_error > tolerance:
        raise FFDPreparationError(
            "Zero-DV SU2_DEF smoke test changed the physical mesh: "
            f"point={smoke_point}, error={smoke_error:.6e}, "
            f"tolerance={tolerance:.6e}"
        )

    visualizations = {}
    missing = []
    for key, filename in _SMOKE_VISUALIZATION_FILENAMES.items():
        path = os.path.join(run_dir, filename)
        if not os.path.isfile(path):
            missing.append(filename)
        else:
            visualizations[key] = path
    if missing:
        raise FFDPreparationError(
            "SU2_DEF smoke test did not create the expected FFD visualization "
            f"files: {missing}"
        )

    return {
        "config": smoke_cfg,
        "log": smoke_log,
        "mesh": smoke_mesh,
        "coordinate_error": smoke_error,
        "visualizations": visualizations,
    }


def _persist_smoke_artifacts(
    smoke_run,
    prep_dir,
    *,
    prepared_mesh,
    marker,
    other_markers,
    upper_tag,
    blending=BEZIER,
    bspline_orders=(2, 2, 2),
):
    os.makedirs(prep_dir, exist_ok=True)
    persistent_cfg = os.path.abspath(os.path.join(prep_dir, "dual_zero_smoke.cfg"))
    persistent_log = os.path.abspath(os.path.join(prep_dir, "dual_zero_smoke.log"))
    _write_zero_smoke_config(
        persistent_cfg,
        mesh_in=os.path.abspath(prepared_mesh),
        mesh_out_base=os.path.abspath(os.path.join(prep_dir, "dual_zero_smoke")),
        marker=marker,
        other_markers=other_markers,
        upper_tag=upper_tag,
        blending=blending,
        bspline_orders=bspline_orders,
    )
    shutil.copy2(smoke_run["log"], persistent_log)

    artifacts = {
        "smoke_config": persistent_cfg,
        "smoke_log": persistent_log,
    }
    expected = _expected_smoke_visualizations(prep_dir)
    for key, destination in expected.items():
        shutil.copy2(smoke_run["visualizations"][key], destination)
        artifacts[key] = destination
    return artifacts


def _validate_dual_mesh(mesh_path, geometry, opts):
    try:
        upper_columns = read_ffd_box_columns(
            mesh_path,
            opts["ffd_upper_box_tag"],
        )
        lower_columns = read_ffd_box_columns(
            mesh_path,
            opts["ffd_lower_box_tag"],
        )
    except Exception as exc:
        raise FFDPreparationError(
            f"Prepared mesh does not contain valid dual FFD boxes: {exc}"
        ) from exc

    tolerance = 1.0e-10 * max(1.0, geometry["chord"])
    for side, columns in (("upper", upper_columns), ("lower", lower_columns)):
        if (
            float(columns[0]) > geometry["x_le"] + tolerance
            or float(columns[-1]) < geometry["x_te"] - tolerance
        ):
            raise FFDPreparationError(
                f"The {side} prepared FFD box does not enclose LE/TE"
            )
    return upper_columns, lower_columns


def _update_runtime_options(base_config, opts, prepared_mesh, geometry):
    opts["ffd_active_xmin"] = float(geometry["x_le"])
    opts["ffd_active_xmax"] = float(geometry["x_te"])
    opts["ffd_active_include_bounds"] = False
    opts["ffd_prepared_mesh_resolved"] = os.path.abspath(prepared_mesh)
    opts["ffd_initial_columns"] = validate_active_ffd_columns(
        opts["ffd_initial_columns"],
        xmin=geometry["x_le"],
        xmax=geometry["x_te"],
        include_bounds=False,
    )
    base_config["MESH_FILENAME"] = os.path.abspath(prepared_mesh)


def prepare_progressive_ffd_input(base_config, opts, partitions=1):
    """Prepare or validate the input mesh and update runtime config/options."""

    if not opts.get("ffd_dual_box", False):
        return None
    if "MESH_FILENAME" not in base_config or not base_config["MESH_FILENAME"]:
        raise FFDPreparationError("Progressive dual FFD requires MESH_FILENAME")

    source_mesh = _resolve_from_config(base_config, base_config["MESH_FILENAME"])
    if not os.path.isfile(source_mesh):
        raise FFDPreparationError(f"Input mesh does not exist: {source_mesh}")
    geometry = _mesh_geometry(source_mesh, opts["ffd_marker"])
    opts["ffd_initial_columns"] = validate_active_ffd_columns(
        opts["ffd_initial_columns"],
        xmin=geometry["x_le"],
        xmax=geometry["x_te"],
        include_bounds=False,
    )

    if not opts.get("ffd_auto_prepare", False):
        _validate_dual_mesh(source_mesh, geometry, opts)
        _update_runtime_options(base_config, opts, source_mesh, geometry)
        return {
            "prepared_mesh": source_mesh,
            "cache_reused": True,
            "auto_prepare": False,
            "x_le": geometry["x_le"],
            "x_te": geometry["x_te"],
            "chord": geometry["chord"],
        }

    configured_output = str(opts.get("ffd_prepared_mesh", "") or "").strip()
    if configured_output:
        prepared_mesh = _resolve_from_config(base_config, configured_output)
    else:
        stem = os.path.splitext(os.path.basename(source_mesh))[0]
        prepared_mesh = os.path.join(
            _config_directory(base_config),
            stem + "_dual_ffd.su2",
        )
    prepared_mesh = os.path.abspath(prepared_mesh)
    if prepared_mesh == os.path.abspath(source_mesh):
        raise FFDPreparationError("Prepared mesh path must differ from the raw mesh")

    diagnostics_mesh_stem, _ = os.path.splitext(prepared_mesh)
    diagnostics_path = diagnostics_mesh_stem + "_dual_ffd_diagnostics.csv"
    prep_dir = os.path.join(_config_directory(base_config), "FFD_PREP")
    manifest_path = os.path.join(prep_dir, "prepare_manifest.json")
    os.makedirs(prep_dir, exist_ok=True)

    request = {
        "schema_version": 1,
        "raw_mesh": os.path.abspath(source_mesh),
        "raw_mesh_sha256": _sha256_file(source_mesh),
        "marker": geometry["marker_tag"],
        "initial_columns": [float(x) for x in opts["ffd_initial_columns"]],
        "bootstrap_tag": opts["ffd_bootstrap_tag"],
        "bootstrap_y_padding_chord": float(
            opts["ffd_bootstrap_y_padding_chord"]
        ),
        "upper_tag": opts["ffd_upper_box_tag"],
        "lower_tag": opts["ffd_lower_box_tag"],
        "upper_offset_chord": float(opts["ffd_upper_offset_chord"]),
        "lower_offset_chord": float(opts["ffd_lower_offset_chord"]),
        "prepared_mesh": prepared_mesh,
        "smoke_test": bool(opts.get("ffd_prepare_smoke_test", True)),
        "ffd_blending": opts.get("ffd_blending", BEZIER),
        "bspline_orders": [int(value) for value in opts.get("ffd_bspline_orders", (2, 2, 2))],
    }

    manifest = None
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r") as fp:
                manifest = json.load(fp)
        except Exception:
            manifest = None
    cache_matches = (
        manifest is not None
        and manifest.get("request") == request
        and os.path.isfile(prepared_mesh)
        and os.path.isfile(diagnostics_path)
    )
    if cache_matches:
        if opts.get("ffd_prepare_smoke_test", True):
            expected_visualizations = _expected_smoke_visualizations(prep_dir)
            missing_visualizations = [
                path
                for path in expected_visualizations.values()
                if not os.path.isfile(path)
            ]
            if missing_visualizations:
                marker_names = list(geometry["mesh"]["markers"].keys())
                other_markers = [
                    name
                    for name in marker_names
                    if str(name).lower()
                    != str(geometry["marker_tag"]).lower()
                ]
                stage_dir = tempfile.mkdtemp(
                    prefix=".dual_ffd_visualization_",
                    dir=prep_dir,
                )
                try:
                    smoke_run = _run_zero_smoke_test(
                        stage_dir,
                        mesh_in=prepared_mesh,
                        source_mesh=source_mesh,
                        marker=geometry["marker_tag"],
                        other_markers=other_markers,
                        upper_tag=opts["ffd_upper_box_tag"],
                        partitions=partitions,
                        tolerance=1.0e-10 * max(1.0, geometry["chord"]),
                        blending=opts.get("ffd_blending", BEZIER),
                        bspline_orders=opts.get("ffd_bspline_orders", (2, 2, 2)),
                    )
                    smoke_artifacts = _persist_smoke_artifacts(
                        smoke_run,
                        prep_dir,
                        prepared_mesh=prepared_mesh,
                        marker=geometry["marker_tag"],
                        other_markers=other_markers,
                        upper_tag=opts["ffd_upper_box_tag"],
                        blending=opts.get("ffd_blending", BEZIER),
                        bspline_orders=opts.get("ffd_bspline_orders", (2, 2, 2)),
                    )
                    manifest.setdefault("artifacts", {}).update(smoke_artifacts)
                    manifest.setdefault("result", {})[
                        "smoke_coordinate_error"
                    ] = smoke_run["coordinate_error"]
                    _atomic_write_json(manifest_path, manifest)
                    print(
                        "[PROGRESSIVE_FFD_PREP] Added FFD visualizations: "
                        f"{prep_dir}"
                    )
                finally:
                    shutil.rmtree(stage_dir, ignore_errors=True)
            else:
                manifest.setdefault("artifacts", {}).update(
                    expected_visualizations
                )
                _atomic_write_json(manifest_path, manifest)
        _validate_dual_mesh(prepared_mesh, geometry, opts)
        _update_runtime_options(base_config, opts, prepared_mesh, geometry)
        print(f"[PROGRESSIVE_FFD_PREP] Reusing prepared mesh: {prepared_mesh}")
        return {
            "prepared_mesh": prepared_mesh,
            "diagnostics_csv": diagnostics_path,
            "manifest": manifest_path,
            "cache_reused": True,
            "auto_prepare": True,
            "x_le": geometry["x_le"],
            "x_te": geometry["x_te"],
            "chord": geometry["chord"],
        }

    existing_artifacts = [
        path
        for path in (prepared_mesh, diagnostics_path, manifest_path)
        if os.path.exists(path)
    ]
    if existing_artifacts and not opts.get("ffd_prepare_overwrite", False):
        raise FFDPreparationError(
            "Prepared FFD cache is missing or does not match the requested inputs. "
            "Set PROGRESSIVE_FFD_PREPARE_OVERWRITE=YES to regenerate it. "
            f"Existing artifacts: {existing_artifacts}"
        )

    stage_dir = tempfile.mkdtemp(prefix=".dual_ffd_prepare_", dir=prep_dir)
    try:
        marker_names = list(geometry["mesh"]["markers"].keys())
        other_markers = [
            name
            for name in marker_names
            if str(name).lower() != str(geometry["marker_tag"]).lower()
        ]
        padding = float(opts["ffd_bootstrap_y_padding_chord"]) * geometry["chord"]
        y_bottom = geometry["y_min"] - padding
        y_top = geometry["y_max"] + padding

        bootstrap_base = os.path.join(stage_dir, "bootstrap_raw")
        bootstrap_mesh = bootstrap_base + ".su2"
        bootstrap_cfg = os.path.join(stage_dir, "bootstrap_ffd_setting.cfg")
        bootstrap_log = os.path.join(stage_dir, "bootstrap_su2_def.log")
        _write_bootstrap_config(
            bootstrap_cfg,
            mesh_in=source_mesh,
            mesh_out_base=bootstrap_base,
            marker=geometry["marker_tag"],
            other_markers=other_markers,
            bootstrap_tag=opts["ffd_bootstrap_tag"],
            x_le=geometry["x_le"],
            x_te=geometry["x_te"],
            y_bottom=y_bottom,
            y_top=y_top,
        )
        print(
            "[PROGRESSIVE_FFD_PREP] Generating bootstrap box | "
            f"x=[{geometry['x_le']:.16g},{geometry['x_te']:.16g}] "
            f"y=[{y_bottom:.16g},{y_top:.16g}]"
        )
        _run_su2_def(bootstrap_cfg, partitions, bootstrap_log)
        if not os.path.isfile(bootstrap_mesh):
            raise FFDPreparationError(
                f"SU2_DEF did not create the bootstrap mesh: {bootstrap_mesh}"
            )

        exact_bootstrap = os.path.join(stage_dir, "bootstrap_exact_columns.su2")
        geometric_columns = [geometry["x_le"]] + list(
            opts["ffd_initial_columns"]
        ) + [geometry["x_te"]]
        rewrite_summary = rewrite_ffd_box_with_columns_and_reembed(
            bootstrap_mesh,
            exact_bootstrap,
            box_tag=opts["ffd_bootstrap_tag"],
            new_columns=geometric_columns,
            marker_name=geometry["marker_tag"],
            domain_mode="FULL",
        )

        staged_dual = os.path.join(stage_dir, "prepared_dual_ffd.su2")
        staged_diagnostics = os.path.join(stage_dir, "dual_ffd_diagnostics.csv")
        split_summary = split_bootstrap_ffd_box(
            exact_bootstrap,
            staged_dual,
            bootstrap_tag=opts["ffd_bootstrap_tag"],
            marker=geometry["marker_tag"],
            upper_tag=opts["ffd_upper_box_tag"],
            lower_tag=opts["ffd_lower_box_tag"],
            upper_offset_chord=opts["ffd_upper_offset_chord"],
            lower_offset_chord=opts["ffd_lower_offset_chord"],
            x_le=geometry["x_le"],
            x_te=geometry["x_te"],
            diagnostics_csv=staged_diagnostics,
            overwrite=False,
            output_blending=opts.get("ffd_blending", BEZIER),
            bspline_orders=opts.get("ffd_bspline_orders", (2, 2, 2)),
        )

        coordinate_error, coordinate_point = _max_mesh_coordinate_difference(
            source_mesh,
            staged_dual,
        )
        tolerance = 1.0e-10 * max(1.0, geometry["chord"])
        if coordinate_error > tolerance:
            raise FFDPreparationError(
                "Dual FFD preparation changed the physical mesh: "
                f"point={coordinate_point}, error={coordinate_error:.6e}, "
                f"tolerance={tolerance:.6e}"
            )
        _validate_dual_mesh(staged_dual, geometry, opts)

        smoke_error = None
        smoke_run = None
        if opts.get("ffd_prepare_smoke_test", True):
            smoke_run = _run_zero_smoke_test(
                stage_dir,
                mesh_in=staged_dual,
                source_mesh=source_mesh,
                marker=geometry["marker_tag"],
                other_markers=other_markers,
                upper_tag=opts["ffd_upper_box_tag"],
                partitions=partitions,
                tolerance=tolerance,
                blending=opts.get("ffd_blending", BEZIER),
                bspline_orders=opts.get("ffd_bspline_orders", (2, 2, 2)),
            )
            smoke_error = smoke_run["coordinate_error"]

        os.makedirs(os.path.dirname(prepared_mesh), exist_ok=True)
        shutil.copy2(
            bootstrap_cfg,
            os.path.join(prep_dir, "bootstrap_ffd_setting.cfg"),
        )
        shutil.copy2(
            bootstrap_log,
            os.path.join(prep_dir, "bootstrap_su2_def.log"),
        )
        smoke_artifacts = {}
        if smoke_run is not None:
            smoke_artifacts = _persist_smoke_artifacts(
                smoke_run,
                prep_dir,
                prepared_mesh=prepared_mesh,
                marker=geometry["marker_tag"],
                other_markers=other_markers,
                upper_tag=opts["ffd_upper_box_tag"],
                blending=opts.get("ffd_blending", BEZIER),
                bspline_orders=opts.get("ffd_bspline_orders", (2, 2, 2)),
            )

        os.replace(staged_dual, prepared_mesh)
        os.makedirs(os.path.dirname(diagnostics_path), exist_ok=True)
        os.replace(staged_diagnostics, diagnostics_path)

        result_payload = {
            "x_le": geometry["x_le"],
            "x_te": geometry["x_te"],
            "chord": geometry["chord"],
            "y_bottom": y_bottom,
            "y_top": y_top,
            "geometric_columns": geometric_columns,
            "degree_i": len(geometric_columns) - 1,
            "initial_total_ndv": 2 * len(opts["ffd_initial_columns"]),
            "upper_surface_points": split_summary["upper_surface_points"],
            "lower_surface_points": split_summary["lower_surface_points"],
            "fixed_edge_points": split_summary["fixed_edge_points"],
            "rewrite_max_error": rewrite_summary["reembedding_max_error"],
            "upper_max_error": split_summary["upper_max_reembedding_error"],
            "lower_max_error": split_summary["lower_max_reembedding_error"],
            "physical_coordinate_error": coordinate_error,
            "smoke_coordinate_error": smoke_error,
            "ffd_blending": opts.get("ffd_blending", BEZIER),
            "bspline_orders": [int(value) for value in opts.get("ffd_bspline_orders", (2, 2, 2))],
        }
        manifest_payload = {
            "request": request,
            "result": result_payload,
            "artifacts": {
                "prepared_mesh": prepared_mesh,
                "diagnostics_csv": diagnostics_path,
                "bootstrap_config": os.path.join(
                    prep_dir, "bootstrap_ffd_setting.cfg"
                ),
                "bootstrap_log": os.path.join(prep_dir, "bootstrap_su2_def.log"),
                "smoke_config": smoke_artifacts.get("smoke_config"),
                "smoke_log": smoke_artifacts.get("smoke_log"),
                **{
                    key: value
                    for key, value in smoke_artifacts.items()
                    if key not in ("smoke_config", "smoke_log")
                },
            },
        }
        _atomic_write_json(manifest_path, manifest_payload)
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)

    _update_runtime_options(base_config, opts, prepared_mesh, geometry)
    print(
        "[PROGRESSIVE_FFD_PREP] Completed | "
        f"mesh={prepared_mesh} diagnostics={diagnostics_path}"
    )
    return {
        "prepared_mesh": prepared_mesh,
        "diagnostics_csv": diagnostics_path,
        "manifest": manifest_path,
        "cache_reused": False,
        "auto_prepare": True,
        "x_le": geometry["x_le"],
        "x_te": geometry["x_te"],
        "chord": geometry["chord"],
    }
