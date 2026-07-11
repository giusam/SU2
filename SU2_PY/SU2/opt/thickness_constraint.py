#!/usr/bin/env python

import contextlib
import copy
import hashlib
import math
import os
import shutil
import sys
import numpy as np

import SU2


THICKNESS_PROGRESSIVE_KEYS = [
    "PROGRESSIVE_THICKNESS_CONSTRAINT",
    "PROGRESSIVE_THICKNESS_REF_MESH",
    "PROGRESSIVE_THICKNESS_MARKER",
    "PROGRESSIVE_THICKNESS_NPOINTS",
    "PROGRESSIVE_THICKNESS_XMIN",
    "PROGRESSIVE_THICKNESS_XMAX",
    "PROGRESSIVE_THICKNESS_X_STATIONS",
    "PROGRESSIVE_THICKNESS_MARGIN",
    "PROGRESSIVE_THICKNESS_FD_EPS",
    "PROGRESSIVE_THICKNESS_GRADIENT",
    "PROGRESSIVE_THICKNESS_CACHE_FILE",
    "PROGRESSIVE_THICKNESS_DOMAIN_MODE",
    "PROGRESSIVE_THICKNESS_SYMMETRY_Y",
]


def _as_bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().upper() in ("YES", "TRUE", "1", "ON")


def _normalize_gradient_mode(value):
    mode = str(value or "AUTO").strip().upper()
    aliases = {
        "FD": "FINITE_DIFFERENCE",
        "FINDIFF": "FINITE_DIFFERENCE",
        "FINITE_DIFF": "FINITE_DIFFERENCE",
        "FINITE_DIFFERENCES": "FINITE_DIFFERENCE",
        "FINITE_DIFFERENCE": "FINITE_DIFFERENCE",
        "ANALYTIC": "ANALYTIC",
        "ANALYTICAL": "ANALYTIC",
        "ACCELERATED": "ANALYTIC",
        "FAST": "ANALYTIC",
        "AUTO": "AUTO",
    }
    if mode not in aliases:
        raise ValueError(
            "PROGRESSIVE_THICKNESS_GRADIENT must be AUTO, ANALYTIC, "
            f"or FINITE_DIFFERENCE; got {mode!r}"
        )
    return aliases[mode]


def _resolve_domain_mode(value, surface_mode=None):
    domain_mode = str(value or "AUTO").strip().upper()
    allowed = ("AUTO", "FULL", "HALF_UPPER", "HALF_LOWER")
    if domain_mode not in allowed:
        raise ValueError(
            "PROGRESSIVE_THICKNESS_DOMAIN_MODE must be AUTO, FULL, "
            f"HALF_UPPER, or HALF_LOWER; got {domain_mode!r}"
        )

    if surface_mode is None:
        return "FULL" if domain_mode == "AUTO" else domain_mode

    surface_aliases = {
        "BOTH": "BOTH",
        "FULL": "BOTH",
        "UPPER": "UPPER",
        "HALF_UPPER": "UPPER",
        "LOWER": "LOWER",
        "HALF_LOWER": "LOWER",
    }
    surface_key = str(surface_mode).strip().upper().replace("-", "_")
    if surface_key not in surface_aliases:
        raise ValueError(
            "BSPLINE_SURFACE_MODE must be BOTH, UPPER, or LOWER; "
            f"got {surface_mode!r}"
        )
    surface_mode = surface_aliases[surface_key]
    natural = {
        "BOTH": "FULL",
        "UPPER": "HALF_UPPER",
        "LOWER": "HALF_LOWER",
    }[surface_mode]
    if domain_mode == "AUTO":
        return natural
    if domain_mode == natural:
        return domain_mode
    if domain_mode == "FULL":
        raise ValueError(
            "FULL thickness requires a complete upper/lower surface; "
            f"use {natural} with BSPLINE_SURFACE_MODE={surface_mode}."
        )
    raise ValueError(
        f"{domain_mode} thickness is incompatible with "
        f"BSPLINE_SURFACE_MODE={surface_mode}; use {natural}."
    )


def _resolve_from_cfg_dir(base_config, filename):
    if not filename:
        return filename

    filename = str(filename)
    if os.path.isabs(filename):
        return filename

    cfg_filename = getattr(base_config, "_filename", None)
    if cfg_filename:
        cfg_dir = os.path.dirname(os.path.abspath(cfg_filename))
    else:
        cfg_dir = os.getcwd()

    return os.path.abspath(os.path.join(cfg_dir, filename))


def _strip_comment(line):
    return line.split("%", 1)[0].strip()


def _parse_key_value(line):
    if "=" not in line:
        return None, None
    key, value = line.split("=", 1)
    return key.strip().upper(), _strip_comment(value)


def _numbers(line):
    values = []
    raw = _strip_comment(line).replace(",", " ")
    for token in raw.split():
        try:
            values.append(float(token))
        except Exception:
            pass
    return values


def _ints(line):
    values = []
    raw = _strip_comment(line).replace(",", " ")
    for token in raw.split():
        try:
            values.append(int(float(token)))
        except Exception:
            pass
    return values


def _x_stations_value_is_empty(value):
    if value is None:
        return True
    if isinstance(value, str):
        raw = _strip_comment(value).strip()
        for char in "(),[]":
            raw = raw.replace(char, " ")
        return not raw.split()
    try:
        return len(value) == 0
    except Exception:
        return False


def _flatten_station_tokens(value):
    if isinstance(value, str):
        raw = _strip_comment(value).strip()
        for char in "(),[]":
            raw = raw.replace(char, " ")
        return raw.split()

    if isinstance(value, np.ndarray):
        value = value.tolist()

    try:
        tokens = []
        for item in value:
            if isinstance(item, (list, tuple, np.ndarray)):
                tokens.extend(_flatten_station_tokens(item))
            else:
                tokens.append(item)
        return tokens
    except TypeError:
        return [value]


def _parse_x_stations(value):
    tokens = _flatten_station_tokens(value)
    if not tokens:
        raise ValueError(
            "PROGRESSIVE_THICKNESS_X_STATIONS requires at least one station"
        )

    stations = []
    for token in tokens:
        try:
            x = float(token)
        except Exception as exc:
            raise ValueError(
                "PROGRESSIVE_THICKNESS_X_STATIONS contains a non-numeric "
                f"station: {token!r}"
            ) from exc
        if not 0.0 < x < 1.0:
            raise ValueError(
                "PROGRESSIVE_THICKNESS_X_STATIONS values must satisfy "
                f"0 < x < 1; got {x:.12g}"
            )
        stations.append(x)

    stations.sort()
    unique = []
    tol = 1.0e-12
    for x in stations:
        if not unique or abs(x - unique[-1]) > tol:
            unique.append(x)

    return np.asarray(unique, dtype=float)


def _read_su2_points_and_marker_segments(mesh_filename, marker_name):
    points, segments = _read_su2_points_and_marker_segments_with_ids(
        mesh_filename,
        marker_name,
    )
    return points, [(points[a], points[b]) for a, b in segments]


def _read_su2_points_and_marker_segments_with_ids(mesh_filename, marker_name):
    marker_name = str(marker_name).strip()
    with open(mesh_filename, "r") as fp:
        lines = fp.readlines()

    ndime = 2
    points = {}
    i = 0
    while i < len(lines):
        key, value = _parse_key_value(lines[i])
        if key == "NDIME":
            vals = _numbers(value)
            if vals:
                ndime = int(round(vals[0]))
        elif key == "NPOIN":
            vals = _numbers(value)
            if not vals:
                raise ValueError(f"Invalid NPOIN line in {mesh_filename}")
            npoint = int(round(vals[0]))
            for local_id in range(npoint):
                i += 1
                vals_point = _numbers(lines[i])
                if len(vals_point) < ndime:
                    raise ValueError(
                        f"Invalid point line in {mesh_filename}: {lines[i].rstrip()}"
                    )
                coords = list(vals_point[:ndime])
                if ndime == 2:
                    coords.append(0.0)
                point_id = local_id
                if len(vals_point) > ndime:
                    candidate_id = int(round(vals_point[-1]))
                    if candidate_id >= 0:
                        point_id = candidate_id
                points[point_id] = coords[:3]
        elif key == "MARKER_TAG":
            tag = str(value).strip().strip("()")
            elem_key, elem_value = _parse_key_value(lines[i + 1])
            if elem_key != "MARKER_ELEMS":
                raise ValueError(
                    f"Expected MARKER_ELEMS after MARKER_TAG in {mesh_filename}"
                )
            nelem_vals = _numbers(elem_value)
            nelem = int(round(nelem_vals[0])) if nelem_vals else 0
            if tag == marker_name:
                segments = []
                for j in range(nelem):
                    elem_values = _ints(lines[i + 2 + j])
                    if len(elem_values) < 3:
                        continue
                    node_ids = elem_values[1:]
                    for a, b in zip(node_ids[:-1], node_ids[1:]):
                        if a in points and b in points:
                            segments.append((a, b))
                if not segments:
                    raise ValueError(
                        f"Marker {marker_name!r} has no usable boundary segments"
                    )
                return points, segments
            i += nelem + 1
        i += 1

    raise ValueError(f"Marker {marker_name!r} was not found in {mesh_filename}")


def _hicks_henne_bump(x, center):
    x = float(x)
    center = float(center)
    if not 0.0 < center < 1.0:
        return 0.0
    if not 0.0 < x < 1.0:
        return 0.0

    exponent = math.log(0.5) / math.log(center)
    value = math.sin(math.pi * (x ** exponent))
    return value ** 6


def _bernstein(n, i, t):
    n = int(n)
    i = int(i)
    if i < 0 or i > n:
        return 0.0
    t = max(0.0, min(1.0, float(t)))
    return math.comb(n, i) * (t ** i) * ((1.0 - t) ** (n - i))


def _definition_dv_size(def_dv):
    return int(sum(int(v) for v in def_dv.get("SIZE", [])))


def _as_scalar_param_list(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(value)]


def _ffd_control_point_2d_params(params):
    params = _as_scalar_param_list(params)
    if len(params) >= 5:
        return int(round(params[1])), int(round(params[2])), float(params[3]), float(params[4])
    if len(params) >= 4:
        return int(round(params[0])), int(round(params[1])), float(params[2]), float(params[3])
    raise ValueError(f"Invalid FFD_CONTROL_POINT_2D params: {params}")


def _mesh_filename_from_project(project):
    cfg = project.config
    mesh_in = str(cfg["MESH_FILENAME"])
    if not os.path.isabs(mesh_in):
        mesh_in = os.path.abspath(mesh_in)
    return mesh_in


def _read_ffd_surface_param_map(mesh_filename, box_tag, marker_name):
    from SU2.opt.progressive_ffd_mesh import (
        _find_tagged_ffd_block,
        _infer_axes_from_control_points,
        _infer_surface_param_location,
        _parse_control_points,
        _parse_count_block,
        _parse_degree,
        _parse_mesh_points,
        _split_tokens,
    )

    points, marker_segments = _read_su2_points_and_marker_segments_with_ids(
        mesh_filename,
        marker_name,
    )
    marker_point_ids = set()
    for a, b in marker_segments:
        marker_point_ids.add(a)
        marker_point_ids.add(b)

    with open(mesh_filename, "r") as fp:
        lines = fp.readlines()

    ndime, _ = _parse_mesh_points(lines)
    block_start, block_end = _find_tagged_ffd_block(lines, box_tag)
    degree = _parse_degree(lines, block_start, block_end)
    control_block = _parse_count_block(
        lines,
        block_start,
        block_end,
        "FFD_CONTROL_POINTS",
    )
    surface_block = _parse_count_block(
        lines,
        block_start,
        block_end,
        "FFD_SURFACE_POINTS",
    )
    if control_block is None or surface_block is None:
        raise ValueError("FFD_CONTROL_POINTS and FFD_SURFACE_POINTS are required")

    control_points, _, _ = _parse_control_points(control_block)
    x_columns, y_rows, z_planes = _infer_axes_from_control_points(
        control_points,
        degree,
    )
    if not z_planes:
        z_planes = [0.0]

    axes = {
        "columns": list(x_columns),
        "y_rows": list(y_rows),
        "z_planes": list(z_planes),
    }

    param_by_point = {}
    for line in surface_block["data"]:
        tokens = _split_tokens(line)
        if not tokens:
            continue
        best = None
        for token in tokens[:4]:
            try:
                point_id = int(round(float(token)))
            except Exception:
                continue
            if point_id not in marker_point_ids or point_id not in points:
                continue
            local_best = _infer_surface_param_location(
                tokens,
                points[point_id],
                axes,
                ndime,
            )
            if local_best is None or local_best["point_id"] != point_id:
                continue
            if best is None or local_best["score"] < best["score"]:
                best = local_best

        if best is None:
            continue

        point_id = best["point_id"]
        start = best["param_start"]
        count = best["param_count"]
        uvw = [float(tokens[start]), float(tokens[start + 1]), 0.0]
        if count >= 3:
            uvw[2] = float(tokens[start + 2])
        param_by_point[point_id] = uvw

    missing = sorted(pid for pid in marker_point_ids if pid not in param_by_point)
    if missing:
        raise ValueError(
            "Could not recover FFD parametric coordinates for "
            f"{len(missing)} marker points"
        )

    return points, marker_segments, param_by_point, axes


def _read_curved_ffd_surface_param_maps(mesh_filename, box_tags, marker_name):
    """Read per-box surface parameters for curved Bezier/B-spline FFD boxes."""

    from SU2.opt.progressive_ffd_split import (
        _parse_curved_surface_lines,
        _parse_existing_dual_box,
    )

    points, marker_segments = _read_su2_points_and_marker_segments_with_ids(
        mesh_filename,
        marker_name,
    )
    with open(mesh_filename, "r") as fp:
        lines = fp.readlines()

    coordinate_scale = max(
        [1.0]
        + [
            max(abs(float(point[0])), abs(float(point[1])))
            for point in points.values()
        ]
    )
    reconstruction_tol = 1.0e-10 * coordinate_scale
    box_data = {}
    for box_tag in box_tags:
        box = _parse_existing_dual_box(lines, box_tag)
        parsed = _parse_curved_surface_lines(
            box["surface_block"],
            points,
            box["columns"],
            box["control_y"],
            box["z_planes"],
            marker_name,
            reconstruction_tol,
            box["blending_spec"],
        )
        box_data[str(box_tag)] = {
            "params": {
                int(point_id): list(values["old_uvw"])
                for point_id, values in parsed.items()
            },
            "columns": list(box["columns"]),
            "control_y": [list(row) for row in box["control_y"]],
            "blending_spec": box["blending_spec"],
        }
    return points, marker_segments, box_data


def _section_measure_from_segments(
    mesh_filename,
    marker_name,
    x_stations,
    domain_mode="FULL",
    symmetry_y=0.0,
):
    _, segments = _read_su2_points_and_marker_segments(mesh_filename, marker_name)
    section_measure = []
    tol = 1.0e-12
    domain_mode = str(domain_mode).upper()
    symmetry_y = float(symmetry_y)

    for x in x_stations:
        y_hits = []
        x = float(x)
        for p0, p1 in segments:
            x0, y0 = float(p0[0]), float(p0[1])
            x1, y1 = float(p1[0]), float(p1[1])
            xmin = min(x0, x1)
            xmax = max(x0, x1)
            if x < xmin - tol or x > xmax + tol:
                continue
            if abs(x1 - x0) <= tol:
                if abs(x - x0) <= tol:
                    y_hits.extend([y0, y1])
                continue
            t = (x - x0) / (x1 - x0)
            if -tol <= t <= 1.0 + tol:
                y_hits.append(y0 + t * (y1 - y0))

        if domain_mode == "FULL":
            if len(y_hits) < 2:
                raise ValueError(
                    f"Could not compute airfoil thickness at x={x:.12g} "
                    f"from marker {marker_name!r} in {mesh_filename}"
                )
            section_measure.append(max(y_hits) - min(y_hits))
        elif domain_mode == "HALF_UPPER":
            if len(y_hits) < 1:
                raise ValueError(
                    f"Could not compute upper half-thickness at x={x:.12g} "
                    f"from marker {marker_name!r} in {mesh_filename}"
                )
            section_measure.append(max(y_hits) - symmetry_y)
        elif domain_mode == "HALF_LOWER":
            if len(y_hits) < 1:
                raise ValueError(
                    f"Could not compute lower half-thickness at x={x:.12g} "
                    f"from marker {marker_name!r} in {mesh_filename}"
                )
            section_measure.append(symmetry_y - min(y_hits))
        else:
            raise ValueError(
                "PROGRESSIVE_THICKNESS_DOMAIN_MODE must be FULL, HALF_UPPER, or HALF_LOWER, "
                f"got {domain_mode!r}"
            )

    return np.asarray(section_measure, dtype=float)


def _metadata_value(data, key, default=""):
    if key not in data:
        return default
    value = data[key]
    try:
        return str(value.item())
    except Exception:
        return str(value)


def _load_or_build_reference(
    ref_mesh,
    marker,
    x_stations,
    cache_file,
    domain_mode,
    symmetry_y,
):
    domain_mode = str(domain_mode).upper()
    symmetry_y = float(symmetry_y)
    ref_mesh_abs = os.path.abspath(ref_mesh)

    if cache_file and os.path.exists(cache_file):
        data = np.load(cache_file)
        if "x" in data and "section_measure" in data:
            x_cached = np.asarray(data["x"], dtype=float)
            measure_cached = np.asarray(data["section_measure"], dtype=float)
            cached_domain = _metadata_value(data, "domain_mode").upper()
            cached_marker = _metadata_value(data, "marker")
            cached_ref_mesh = _metadata_value(data, "ref_mesh")
            cached_symmetry_y = float(_metadata_value(data, "symmetry_y", "nan"))
            if (
                len(x_cached) == len(x_stations)
                and np.allclose(x_cached, x_stations)
                and cached_domain == domain_mode
                and cached_marker == str(marker)
                and os.path.abspath(cached_ref_mesh) == ref_mesh_abs
                and abs(cached_symmetry_y - symmetry_y) <= 1.0e-14
            ):
                return measure_cached

    section_measure = _section_measure_from_segments(
        ref_mesh,
        marker,
        x_stations,
        domain_mode=domain_mode,
        symmetry_y=symmetry_y,
    )

    if cache_file:
        os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
        np.savez(
            cache_file,
            x=x_stations,
            section_measure=section_measure,
            domain_mode=domain_mode,
            symmetry_y=symmetry_y,
            marker=str(marker),
            ref_mesh=ref_mesh_abs,
        )

    return section_measure


def clean_progressive_thickness_keys(cfg):
    for key in THICKNESS_PROGRESSIVE_KEYS:
        if key in cfg:
            del cfg[key]


@contextlib.contextmanager
def _redirect_stdout_stderr_to_file(log_path):
    log_dir = os.path.dirname(os.path.abspath(log_path))
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    sys.stdout.flush()
    sys.stderr.flush()
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    try:
        with open(log_path, "a") as log_file:
            os.dup2(log_file.fileno(), 1)
            os.dup2(log_file.fileno(), 2)
            try:
                yield
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
    finally:
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)


class ThicknessConstraint:
    def __init__(
        self,
        ref_mesh,
        marker,
        x_stations,
        reference_measure,
        margin=0.0,
        fd_eps=1.0e-6,
        gradient_mode="AUTO",
        domain_mode="FULL",
        symmetry_y=0.0,
    ):
        self.ref_mesh = os.path.abspath(ref_mesh)
        self.marker = str(marker)
        self.x_stations = np.asarray(x_stations, dtype=float)
        self.reference_measure = np.asarray(reference_measure, dtype=float)
        self.reference_thickness = self.reference_measure
        self.margin = float(margin)
        self.fd_eps = float(fd_eps)
        self.gradient_mode = _normalize_gradient_mode(gradient_mode)
        self.domain_mode = str(domain_mode).upper()
        self.symmetry_y = float(symmetry_y)
        self.eval_dir = "THICKNESS_CONSTRAINT_EVAL"
        self._cache = {}
        self._fallback_warned = False

    def _cache_key(self, x_eval, cfg):
        payload = repr(
            {
                "mesh": os.path.abspath(str(cfg.get("MESH_FILENAME", ""))),
                "dv_kind": cfg.get("DV_KIND", ""),
                "dv_marker": cfg.get("DV_MARKER", ""),
                "definition_dv": cfg.get("DEFINITION_DV", ""),
                "dv_value_old": cfg.get("DV_VALUE_OLD", ""),
                "x": np.asarray(x_eval, dtype=float).tolist(),
            }
        ).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()

    def _current_measure_for_x(self, x_eval, project):
        cfg = SU2.io.Config(copy.deepcopy(dict(project.config)))
        clean_progressive_thickness_keys(cfg)

        mesh_in = str(cfg["MESH_FILENAME"])
        if not os.path.isabs(mesh_in):
            mesh_in = os.path.abspath(mesh_in)
        cfg["MESH_FILENAME"] = mesh_in

        key = self._cache_key(x_eval, cfg)
        if key in self._cache:
            return self._cache[key]

        current = self._run_def_and_read_measure(cfg, x_eval, key)
        self._cache[key] = np.asarray(current, dtype=float)
        return self._cache[key]

    def _run_def_and_read_measure(self, cfg, x_eval, key):
        key12 = key[:12]
        eval_dir = os.path.abspath(os.path.join(self.eval_dir, f"eval_{key12}"))
        log_path = os.path.join(eval_dir, f"su2_def_{key12}.log")
        mesh_in = str(cfg["MESH_FILENAME"])

        cfg["MESH_OUT_FILENAME"] = f"thickness_constraint_{key[:12]}.su2"

        os.makedirs(eval_dir, exist_ok=True)
        cwd = os.getcwd()
        success = False
        try:
            os.chdir(eval_dir)
            cfg.unpack_dvs(list(np.asarray(x_eval, dtype=float)))
            if cfg["DV_VALUE_NEW"] == cfg["DV_VALUE_OLD"]:
                mesh_out = mesh_in
            else:
                with _redirect_stdout_stderr_to_file(log_path):
                    SU2.run.DEF(cfg)
                mesh_out = os.path.abspath(str(cfg["MESH_OUT_FILENAME"]))

            current = _section_measure_from_segments(
                mesh_out,
                self.marker,
                self.x_stations,
                self.domain_mode,
                self.symmetry_y,
            )
            success = True
            return np.asarray(current, dtype=float)
        except Exception as exc:
            raise RuntimeError(
                "Thickness constraint evaluation failed. "
                f"Temporary directory kept for debugging: {eval_dir}. "
                f"SU2_DEF log: {log_path}"
            ) from exc
        finally:
            os.chdir(cwd)
            if success:
                shutil.rmtree(eval_dir, ignore_errors=True)

    def values(self, x_eval, project):
        current = self._current_measure_for_x(x_eval, project)
        g = current - self.reference_measure - self.margin
        min_g = float(np.min(g))
        if min_g < -1.0e-10:
            print(f"[THICKNESS_CONSTRAINT] WARNING: min g = {min_g:.6e}")
        return g.tolist()

    def jacobian_fd(self, x_eval, project):
        x_eval = np.asarray(x_eval, dtype=float)
        g0 = np.asarray(self.values(x_eval, project), dtype=float)
        jac = np.zeros((len(g0), len(x_eval)))

        for j in range(len(x_eval)):
            xp = x_eval.copy()
            xp[j] += self.fd_eps
            gp = np.asarray(self.values(xp, project), dtype=float)
            jac[:, j] = (gp - g0) / self.fd_eps

        return jac

    def jacobian_analytic(self, x_eval, project):
        def_dv = project.config["DEFINITION_DV"]
        kinds = [str(k).upper() for k in def_dv.get("KIND", [])]
        if not kinds:
            raise ValueError("DEFINITION_DV is empty")

        if all(kind == "HICKS_HENNE" for kind in kinds):
            return self._jacobian_hicks_henne(def_dv)

        if all(kind == "FFD_CONTROL_POINT_2D" for kind in kinds):
            return self._jacobian_ffd_control_point_2d(def_dv, project)

        raise ValueError(
            "Analytic thickness gradient supports only pure HICKS_HENNE or "
            "pure FFD_CONTROL_POINT_2D definitions"
        )

    def jacobian(self, x_eval, project):
        if self.gradient_mode == "FINITE_DIFFERENCE":
            return self.jacobian_fd(x_eval, project)

        def_dv = project.config["DEFINITION_DV"]
        kinds = [str(k).upper() for k in def_dv.get("KIND", [])]
        if self.gradient_mode == "AUTO" and all(kind == "HICKS_HENNE" for kind in kinds):
            if not self._fallback_warned:
                print(
                    "[THICKNESS_CONSTRAINT] AUTO gradient keeps finite differences "
                    "for HICKS_HENNE because SU2 applies HH through surface-normal "
                    "classification; set PROGRESSIVE_THICKNESS_GRADIENT=ANALYTIC "
                    "to use the fast approximation."
                )
                self._fallback_warned = True
            return self.jacobian_fd(x_eval, project)

        try:
            return self.jacobian_analytic(x_eval, project)
        except Exception as exc:
            if self.gradient_mode == "ANALYTIC":
                raise RuntimeError(
                    "Analytic thickness constraint gradient failed"
                ) from exc

            if not self._fallback_warned:
                print(
                    "[THICKNESS_CONSTRAINT] WARNING: analytic gradient unavailable; "
                    f"falling back to finite differences ({exc})"
                )
                self._fallback_warned = True
            return self.jacobian_fd(x_eval, project)

    def _jacobian_hicks_henne(self, def_dv):
        n_dv = _definition_dv_size(def_dv)
        jac = np.zeros((len(self.x_stations), n_dv), dtype=float)

        k = 0
        for i_dv, kind in enumerate(def_dv["KIND"]):
            if str(kind).upper() != "HICKS_HENNE":
                raise ValueError("Mixed DV kinds are not supported")
            if int(def_dv["SIZE"][i_dv]) != 1:
                raise ValueError("HICKS_HENNE analytic thickness gradient requires SIZE=1")

            params = _as_scalar_param_list(def_dv["PARAM"][i_dv])
            if len(params) < 2:
                raise ValueError(f"Invalid HICKS_HENNE params: {params}")
            side = "UPPER" if float(params[0]) >= 0.5 else "LOWER"
            center = float(params[1])
            scale = float(def_dv["SCALE"][i_dv])

            for i_x, x in enumerate(self.x_stations):
                bump = scale * _hicks_henne_bump(x, center)
                if self.domain_mode == "FULL":
                    jac[i_x, k] = bump
                elif self.domain_mode == "HALF_UPPER":
                    jac[i_x, k] = bump if side == "UPPER" else 0.0
                elif self.domain_mode == "HALF_LOWER":
                    jac[i_x, k] = bump if side == "LOWER" else 0.0
                else:
                    raise ValueError(
                        "PROGRESSIVE_THICKNESS_DOMAIN_MODE must be FULL, HALF_UPPER, or HALF_LOWER"
                    )
            k += 1

        return jac

    def _jacobian_ffd_control_point_2d(self, def_dv, project):
        from SU2.opt.progressive_ffd_blending import basis_values

        n_dv = _definition_dv_size(def_dv)
        if any(int(size) != 1 for size in def_dv.get("SIZE", [])):
            raise ValueError("FFD_CONTROL_POINT_2D analytic gradient requires SIZE=1")

        mesh_filename = _mesh_filename_from_project(project)
        box_tags = [str(tag) for tag in def_dv.get("FFDTAG", []) if str(tag)]
        if not box_tags:
            raise ValueError("FFD_CONTROL_POINT_2D definitions require FFDTAG")
        unique_box_tags = list(dict.fromkeys(box_tags))
        points, segments, box_data = _read_curved_ffd_surface_param_maps(
            mesh_filename,
            unique_box_tags,
            self.marker,
        )

        marker_point_ids = {
            point_id
            for segment in segments
            for point_id in segment
        }
        point_dy = {
            point_id: np.zeros(n_dv, dtype=float)
            for point_id in marker_point_ids
        }
        embedded_point_ids = set()
        for point_map in box_data.values():
            embedded_point_ids.update(point_map["params"])

        marker_x = [float(points[point_id][0]) for point_id in marker_point_ids]
        x_min = min(marker_x)
        x_max = max(marker_x)
        edge_tol = 1.0e-10 * max(1.0, abs(x_min), abs(x_max))
        unexpected_missing = [
            point_id
            for point_id in marker_point_ids - embedded_point_ids
            if (
                abs(float(points[point_id][0]) - x_min) > edge_tol
                and abs(float(points[point_id][0]) - x_max) > edge_tol
            )
        ]
        if unexpected_missing:
            raise ValueError(
                "FFD surface parameter data are missing for non-edge marker "
                f"points: {sorted(unexpected_missing)[:8]}"
            )

        k = 0
        for i_dv, params in enumerate(def_dv["PARAM"]):
            box_tag = str(def_dv["FFDTAG"][i_dv])
            if box_tag not in box_data:
                raise ValueError(f"FFD box {box_tag!r} was not found in the mesh")
            i_idx, j_idx, dx, dy = _ffd_control_point_2d_params(params)
            if abs(dx) > 1.0e-14:
                raise ValueError(
                    "Analytic FFD thickness gradient supports only Y-direction "
                    "FFD_CONTROL_POINT_2D variables"
                )
            data = box_data[box_tag]
            n_i = len(data["columns"])
            n_j = len(data["control_y"])
            if not 0 <= i_idx < n_i or not 0 <= j_idx < n_j:
                raise ValueError(
                    f"FFD control index ({i_idx},{j_idx}) is outside box "
                    f"{box_tag!r} with shape ({n_i},{n_j})"
                )
            scale = float(def_dv["SCALE"][i_dv])
            for point_id, uvw in data["params"].items():
                u, v = float(uvw[0]), float(uvw[1])
                basis_i = basis_values(
                    n_i,
                    u,
                    data["blending_spec"],
                    axis=0,
                )[i_idx]
                basis_j = basis_values(
                    n_j,
                    v,
                    data["blending_spec"],
                    axis=1,
                )[j_idx]
                point_dy[point_id][k] = scale * dy * basis_i * basis_j
            k += 1

        jac = np.zeros((len(self.x_stations), n_dv), dtype=float)
        tol = 1.0e-12

        for i_x, x_station in enumerate(self.x_stations):
            hits = []
            x_station = float(x_station)
            for a, b in segments:
                p0 = points[a]
                p1 = points[b]
                x0, y0 = float(p0[0]), float(p0[1])
                x1, y1 = float(p1[0]), float(p1[1])
                xmin = min(x0, x1)
                xmax = max(x0, x1)
                if x_station < xmin - tol or x_station > xmax + tol:
                    continue

                if abs(x1 - x0) <= tol:
                    if abs(x_station - x0) <= tol:
                        hits.append((y0, point_dy[a]))
                        hits.append((y1, point_dy[b]))
                    continue

                t = (x_station - x0) / (x1 - x0)
                if -tol <= t <= 1.0 + tol:
                    t = max(0.0, min(1.0, t))
                    y_hit = y0 + t * (y1 - y0)
                    dy_hit = (1.0 - t) * point_dy[a] + t * point_dy[b]
                    hits.append((y_hit, dy_hit))

            if self.domain_mode == "FULL":
                if len(hits) < 2:
                    raise ValueError(
                        f"Could not compute FFD thickness gradient at x={x_station:.12g}"
                    )
                upper = max(hits, key=lambda item: item[0])
                lower = min(hits, key=lambda item: item[0])
                jac[i_x, :] = upper[1] - lower[1]
            elif self.domain_mode == "HALF_UPPER":
                if not hits:
                    raise ValueError(
                        f"Could not compute FFD half-thickness gradient at x={x_station:.12g}"
                    )
                upper = max(hits, key=lambda item: item[0])
                jac[i_x, :] = upper[1]
            elif self.domain_mode == "HALF_LOWER":
                if not hits:
                    raise ValueError(
                        f"Could not compute FFD lower half-thickness gradient at x={x_station:.12g}"
                    )
                lower = min(hits, key=lambda item: item[0])
                jac[i_x, :] = -lower[1]
            else:
                raise ValueError(
                    "PROGRESSIVE_THICKNESS_DOMAIN_MODE must be FULL, HALF_UPPER, or HALF_LOWER"
                )

        return jac


def build_thickness_constraint_from_config(base_config):
    enabled = _as_bool(
        base_config.get("PROGRESSIVE_THICKNESS_CONSTRAINT", "NO"),
        default=False,
    )
    if not enabled:
        return None

    ref_mesh_value = base_config.get("PROGRESSIVE_THICKNESS_REF_MESH", None)
    if not ref_mesh_value:
        raise ValueError(
            "PROGRESSIVE_THICKNESS_REF_MESH is required when "
            "PROGRESSIVE_THICKNESS_CONSTRAINT=YES"
        )

    ref_mesh = _resolve_from_cfg_dir(base_config, ref_mesh_value)
    marker = str(base_config.get("PROGRESSIVE_THICKNESS_MARKER", "AIRFOIL"))
    x_stations_value = base_config.get("PROGRESSIVE_THICKNESS_X_STATIONS", None)
    margin = float(base_config.get("PROGRESSIVE_THICKNESS_MARGIN", 0.0))
    fd_eps = float(base_config.get("PROGRESSIVE_THICKNESS_FD_EPS", 1.0e-6))
    gradient_mode = _normalize_gradient_mode(
        base_config.get("PROGRESSIVE_THICKNESS_GRADIENT", "AUTO")
    )
    if (
        str(base_config.get("PROGRESSIVE_PARAM_KIND", "")).strip().upper()
        == "FFD"
    ):
        default_domain = str(
            base_config.get("PROGRESSIVE_FFD_DOMAIN_MODE", "FULL")
        ).strip().upper()
    else:
        default_domain = (
            "AUTO" if "BSPLINE_SURFACE_MODE" in base_config else "FULL"
        )
    domain_mode = _resolve_domain_mode(
        base_config.get("PROGRESSIVE_THICKNESS_DOMAIN_MODE", default_domain),
        base_config.get("BSPLINE_SURFACE_MODE")
        if "BSPLINE_SURFACE_MODE" in base_config
        else None,
    )
    symmetry_y = float(base_config.get("PROGRESSIVE_THICKNESS_SYMMETRY_Y", 0.0))
    cache_value = str(
        base_config.get(
            "PROGRESSIVE_THICKNESS_CACHE_FILE",
            "thickness_reference.npz",
        )
    )
    cache_file = _resolve_from_cfg_dir(base_config, cache_value)

    explicit_x_stations = not _x_stations_value_is_empty(x_stations_value)
    if explicit_x_stations:
        x_stations = _parse_x_stations(x_stations_value)
    else:
        npoints = int(base_config.get("PROGRESSIVE_THICKNESS_NPOINTS", 101))
        xmin = float(base_config.get("PROGRESSIVE_THICKNESS_XMIN", 0.001))
        xmax = float(base_config.get("PROGRESSIVE_THICKNESS_XMAX", 0.999))
        if npoints < 2:
            raise ValueError("PROGRESSIVE_THICKNESS_NPOINTS must be >= 2")
        if not xmin < xmax:
            raise ValueError("PROGRESSIVE_THICKNESS_XMIN must be < XMAX")
        x_stations = np.linspace(xmin, xmax, npoints)

    reference = _load_or_build_reference(
        ref_mesh,
        marker,
        x_stations,
        cache_file,
        domain_mode,
        symmetry_y,
    )

    print("[THICKNESS_CONSTRAINT] enabled = YES")
    print(f"[THICKNESS_CONSTRAINT] reference mesh = {ref_mesh}")
    print(f"[THICKNESS_CONSTRAINT] marker = {marker}")
    print(f"[THICKNESS_CONSTRAINT] domain mode = {domain_mode}")
    print(f"[THICKNESS_CONSTRAINT] gradient mode = {gradient_mode}")
    if domain_mode in ("HALF_UPPER", "HALF_LOWER"):
        print(f"[THICKNESS_CONSTRAINT] symmetry y = {symmetry_y}")
    if explicit_x_stations:
        print("[THICKNESS_CONSTRAINT] x stations = explicit")
        print(f"[THICKNESS_CONSTRAINT] n points = {len(x_stations)}")
        print(
            "[THICKNESS_CONSTRAINT] x min/max = "
            f"[{float(np.min(x_stations))}, {float(np.max(x_stations))}]"
        )
    else:
        print(f"[THICKNESS_CONSTRAINT] x range = [{xmin}, {xmax}]")
        print(f"[THICKNESS_CONSTRAINT] n points = {npoints}")
    print(
        "[THICKNESS_CONSTRAINT] min original section measure = "
        f"{float(np.min(reference)):.6e}"
    )

    return ThicknessConstraint(
        ref_mesh=ref_mesh,
        marker=marker,
        x_stations=x_stations,
        reference_measure=reference,
        margin=margin,
        fd_eps=fd_eps,
        gradient_mode=gradient_mode,
        domain_mode=domain_mode,
        symmetry_y=symmetry_y,
    )
