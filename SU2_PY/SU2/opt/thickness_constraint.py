#!/usr/bin/env python

import contextlib
import copy
import hashlib
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
    "PROGRESSIVE_THICKNESS_CACHE_FILE",
    "PROGRESSIVE_THICKNESS_DOMAIN_MODE",
    "PROGRESSIVE_THICKNESS_SYMMETRY_Y",
]


def _as_bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().upper() in ("YES", "TRUE", "1", "ON")


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
                            segments.append((points[a], points[b]))
                if not segments:
                    raise ValueError(
                        f"Marker {marker_name!r} has no usable boundary segments"
                    )
                return points, segments
            i += nelem + 1
        i += 1

    raise ValueError(f"Marker {marker_name!r} was not found in {mesh_filename}")


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
        else:
            raise ValueError(
                "PROGRESSIVE_THICKNESS_DOMAIN_MODE must be FULL or HALF_UPPER, "
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
        self.domain_mode = str(domain_mode).upper()
        self.symmetry_y = float(symmetry_y)
        self.eval_dir = "THICKNESS_CONSTRAINT_EVAL"
        self._cache = {}

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
    domain_mode = str(
        base_config.get("PROGRESSIVE_THICKNESS_DOMAIN_MODE", "FULL")
    ).upper()
    symmetry_y = float(base_config.get("PROGRESSIVE_THICKNESS_SYMMETRY_Y", 0.0))
    cache_value = str(
        base_config.get(
            "PROGRESSIVE_THICKNESS_CACHE_FILE",
            "thickness_reference.npz",
        )
    )
    cache_file = _resolve_from_cfg_dir(base_config, cache_value)

    if domain_mode not in ("FULL", "HALF_UPPER"):
        raise ValueError(
            "PROGRESSIVE_THICKNESS_DOMAIN_MODE must be FULL or HALF_UPPER, "
            f"got {domain_mode!r}"
        )

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
    if domain_mode == "HALF_UPPER":
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
        domain_mode=domain_mode,
        symmetry_y=symmetry_y,
    )
