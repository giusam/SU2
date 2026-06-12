#!/usr/bin/env python

import copy
import math

from SU2.opt.hh_spring import spring_redistribute_centers
from SU2.opt.progressive_hh_core import _as_bool, initial_centers


class FFDLevel:
    def __init__(
        self,
        level_id,
        columns,
        workdir,
        config_filename,
        project_filename,
        mesh_source=None,
        initial_mesh_source=None,
        dv_values=None,
        selection_metadata=None,
        post_opt_spring_pending=False,
        spring_reallocated=False,
        ffd_box_tag="AIRFOIL_BOX",
        ffd_dv_kind="FFD_CONTROL_POINT_2D",
        marker="AIRFOIL",
        domain_mode="FULL",
        control_row=None,
        direction="Y",
        active_xmin=0.0,
        active_xmax=1.0,
        active_include_bounds=False,
    ):
        self.level_id = int(level_id)
        self.active_xmin = float(active_xmin)
        self.active_xmax = float(active_xmax)
        self.active_include_bounds = bool(active_include_bounds)
        self.columns = validate_active_ffd_columns(
            columns,
            xmin=self.active_xmin,
            xmax=self.active_xmax,
            include_bounds=self.active_include_bounds,
        )
        self.workdir = workdir
        self.config_filename = config_filename
        self.project_filename = project_filename
        self.mesh_source = mesh_source
        self.initial_mesh_source = (
            initial_mesh_source if initial_mesh_source is not None else mesh_source
        )
        self.dv_values = None if dv_values is None else [float(v) for v in dv_values]
        self.selection_metadata = selection_metadata
        self.post_opt_spring_pending = bool(post_opt_spring_pending)
        self.spring_reallocated = bool(spring_reallocated)
        self.ffd_box_tag = str(ffd_box_tag)
        self.ffd_dv_kind = str(ffd_dv_kind).upper()
        self.marker = str(marker)
        self.domain_mode = str(domain_mode).upper()
        self.control_row = None if control_row is None else int(control_row)
        self.direction = str(direction).upper()

    @property
    def ndv(self):
        return len(self.columns)


def _parse_ffd_initial_columns(
    value,
    xmin=0.0,
    xmax=1.0,
    include_bounds=False,
):
    if value is None:
        return None

    raw = str(value).strip()
    if not raw:
        return None

    raw = raw.strip("()[]")
    if not raw:
        return []

    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    return validate_active_ffd_columns(
        values,
        xmin=xmin,
        xmax=xmax,
        include_bounds=include_bounds,
    )


def _count_ffd_initial_columns(
    value,
    xmin=0.0,
    xmax=1.0,
    include_bounds=False,
):
    columns = _parse_ffd_initial_columns(
        value,
        xmin=xmin,
        xmax=xmax,
        include_bounds=include_bounds,
    )
    if columns is None:
        return None
    return len(columns)


def validate_active_ffd_columns(
    columns,
    xmin=0.0,
    xmax=1.0,
    include_bounds=False,
):
    columns = [float(x) for x in columns]
    xmin = float(xmin)
    xmax = float(xmax)
    if not xmin < xmax:
        raise ValueError(
            "Progressive FFD active column bounds must satisfy xmin < xmax "
            f"(got xmin={xmin}, xmax={xmax})"
        )
    if len(columns) < 2:
        raise ValueError("Progressive FFD requires at least two initial columns")

    tol = 1.0e-10
    for a, b in zip(columns[:-1], columns[1:]):
        if float(b) + tol < float(a):
            raise ValueError("Progressive FFD columns must be sorted")

    if include_bounds:
        bounds_text = f"{xmin} <= x <= {xmax}"
    else:
        bounds_text = f"{xmin} < x < {xmax}"

    for x in columns:
        if include_bounds:
            valid = xmin <= x <= xmax
        else:
            valid = xmin < x < xmax
        if not valid:
            raise ValueError(
                f"Invalid progressive FFD column {x} "
                f"(must satisfy {bounds_text})"
            )
    for a, b in zip(columns[:-1], columns[1:]):
        if abs(float(a) - float(b)) <= tol:
            raise ValueError("Progressive FFD columns must be unique")
    return sorted(columns)


def _ffd_allow_external_columns_from_opts(opts):
    if opts is None:
        return False
    return _as_bool(opts.get("ffd_allow_external_columns", False))


def ffd_active_range_from_opts(opts):
    if _ffd_allow_external_columns_from_opts(opts):
        return (
            float(opts.get("ffd_active_xmin", 0.0)),
            float(opts.get("ffd_active_xmax", 1.0)),
        )
    return 0.0, 1.0


def ffd_active_include_bounds_from_opts(opts):
    return _ffd_allow_external_columns_from_opts(opts)


def _sorted_unique_with_tolerance(values, tol=1.0e-10):
    unique = []
    for value in sorted(float(v) for v in values):
        if not unique or abs(value - unique[-1]) > tol:
            unique.append(value)
    return unique


def build_ffd_mesh_columns(mesh_in, box_tag, active_columns, opts=None):
    from SU2.opt.progressive_ffd_mesh import (
        read_ffd_box_columns,
        validate_mesh_ffd_columns,
    )

    raw_active_columns = [float(x) for x in active_columns]
    old_columns = read_ffd_box_columns(mesh_in, box_tag)
    if len(old_columns) < 2:
        raise ValueError(
            f"FFD box {box_tag!r} must contain at least two existing x-columns"
        )

    boundary_columns = [float(old_columns[0]), float(old_columns[-1])]
    box_x_min = float(boundary_columns[0])
    box_x_max = float(boundary_columns[-1])
    allow_external_columns = _ffd_allow_external_columns_from_opts(opts)
    active_xmin, active_xmax = ffd_active_range_from_opts(opts)
    include_bounds = ffd_active_include_bounds_from_opts(opts)

    for boundary in boundary_columns:
        for active in raw_active_columns:
            if abs(float(active) - float(boundary)) <= 1.0e-10:
                raise ValueError(
                    f"Active FFD column {active} coincides with the inactive "
                    "FFD box boundary."
                )

    if allow_external_columns:
        if not box_x_min < active_xmin:
            raise ValueError(
                "PROGRESSIVE_FFD_ACTIVE_XMIN must be strictly inside the FFD "
                f"box: box_x_min={box_x_min}, active_xmin={active_xmin}"
            )
        if not active_xmax < box_x_max:
            raise ValueError(
                "PROGRESSIVE_FFD_ACTIVE_XMAX must be strictly inside the FFD "
                f"box: active_xmax={active_xmax}, box_x_max={box_x_max}"
            )

    active_columns = validate_active_ffd_columns(
        sorted(raw_active_columns),
        xmin=active_xmin,
        xmax=active_xmax,
        include_bounds=include_bounds,
    )
    mesh_columns = _sorted_unique_with_tolerance(
        boundary_columns + list(active_columns)
    )
    mesh_columns = validate_mesh_ffd_columns(mesh_columns)

    inactive_boundary_columns = []
    for boundary in boundary_columns:
        if not any(abs(boundary - active) <= 1.0e-10 for active in active_columns):
            inactive_boundary_columns.append(boundary)

    print(
        "[PROGRESSIVE_FFD] external active columns = "
        f"{'YES' if allow_external_columns else 'NO'}"
    )
    print(f"[PROGRESSIVE_FFD] FFD box x-range = [{box_x_min}, {box_x_max}]")
    print(
        "[PROGRESSIVE_FFD] FFD active candidate x-range = "
        f"[{active_xmin}, {active_xmax}]"
    )
    print(f"[PROGRESSIVE_FFD] Active columns = {active_columns}")
    print(f"[PROGRESSIVE_FFD] Mesh columns = {mesh_columns}")
    print(f"[PROGRESSIVE_FFD] Inactive boundary columns = {inactive_boundary_columns}")

    return mesh_columns, active_columns


def get_progressive_ffd_options(config, hh_opts):
    opts = dict(hh_opts)
    opts["param_kind"] = "FFD"

    ffd_dv_kind = str(
        config.get("PROGRESSIVE_FFD_DV_KIND", "FFD_CONTROL_POINT_2D")
    ).strip().upper()
    box_tag = str(config.get("PROGRESSIVE_FFD_BOX_TAG", "AIRFOIL_BOX")).strip()
    marker = str(config.get("PROGRESSIVE_FFD_MARKER", opts.get("marker", "AIRFOIL"))).strip()
    domain_mode = str(config.get("PROGRESSIVE_FFD_DOMAIN_MODE", "FULL")).strip().upper()
    direction = str(config.get("PROGRESSIVE_FFD_DIRECTION", "Y")).strip().upper()
    allow_external_columns = _as_bool(
        config.get("PROGRESSIVE_FFD_ALLOW_EXTERNAL_COLUMNS", "NO")
    )
    active_xmin = float(config.get("PROGRESSIVE_FFD_ACTIVE_XMIN", 0.0))
    active_xmax = float(config.get("PROGRESSIVE_FFD_ACTIVE_XMAX", 1.0))
    if not active_xmin < active_xmax:
        raise ValueError(
            "PROGRESSIVE_FFD_ACTIVE_XMIN must be less than "
            "PROGRESSIVE_FFD_ACTIVE_XMAX"
        )
    initial_include_bounds = bool(allow_external_columns)

    control_row_value = config.get("PROGRESSIVE_FFD_CONTROL_ROW", None)
    control_row = None
    if control_row_value is not None and str(control_row_value).strip() != "":
        control_row = int(control_row_value)

    supported_2d = ("FFD_CONTROL_POINT_2D", "FFD_THICKNESS_2D")
    if ffd_dv_kind not in supported_2d:
        raise NotImplementedError(
            "Progressive FFD currently supports only FFD_CONTROL_POINT_2D "
            f"and FFD_THICKNESS_2D; got {ffd_dv_kind}"
        )

    if domain_mode not in ("FULL", "HALF_UPPER"):
        raise ValueError(
            "PROGRESSIVE_FFD_DOMAIN_MODE must be FULL or HALF_UPPER, "
            f"got {domain_mode!r}"
        )

    if ffd_dv_kind == "FFD_THICKNESS_2D" and domain_mode == "HALF_UPPER":
        raise NotImplementedError(
            "PROGRESSIVE_FFD_DOMAIN_MODE=HALF_UPPER is not validated for "
            "FFD_THICKNESS_2D"
        )

    if ffd_dv_kind == "FFD_CONTROL_POINT_2D" and control_row is None:
        raise ValueError(
            "PROGRESSIVE_FFD_CONTROL_ROW is required for FFD_CONTROL_POINT_2D"
        )

    if control_row is not None and control_row < 0:
        raise ValueError("PROGRESSIVE_FFD_CONTROL_ROW must be >= 0")

    if direction not in ("X", "Y"):
        raise ValueError("PROGRESSIVE_FFD_DIRECTION must be X or Y")

    refine_state_mode = str(
        opts.get("refine_state_mode", "DEFORMED_MESH_ZERO_DV")
    ).upper()
    if refine_state_mode == "INITIAL_MESH_KEEP_DV":
        raise NotImplementedError(
            "PROGRESSIVE_HH_REFINE_STATE=INITIAL_MESH_KEEP_DV is not supported "
            "for PROGRESSIVE_PARAM_KIND=FFD"
        )

    if str(opts.get("symmetry_mode", "NONE")).upper() == "REDUCED":
        raise NotImplementedError(
            "PROGRESSIVE_HH_SYMMETRY_MODE=REDUCED is HH-specific and is not "
            "supported for PROGRESSIVE_PARAM_KIND=FFD"
        )

    initial_columns_count = _count_ffd_initial_columns(
        config.get("PROGRESSIVE_FFD_INITIAL_COLUMNS", None),
        xmin=active_xmin if allow_external_columns else 0.0,
        xmax=active_xmax if allow_external_columns else 1.0,
        include_bounds=initial_include_bounds,
    )
    if initial_columns_count is None:
        initial_columns_count = _count_ffd_initial_columns(
            config.get("PROGRESSIVE_HH_INITIAL_UPPER", None),
            xmin=active_xmin if allow_external_columns else 0.0,
            xmax=active_xmax if allow_external_columns else 1.0,
            include_bounds=initial_include_bounds,
        )
    if initial_columns_count is None:
        initial_columns_count = int(opts.get("n0", 3))

    nfinal = opts.get("nfinal", None)
    if nfinal is not None and int(nfinal) < int(initial_columns_count):
        raise ValueError(
            "PROGRESSIVE_HH_NFINAL must be >= the initial FFD NDV "
            f"({initial_columns_count})"
        )

    opts.update(
        {
            "ffd_dv_kind": ffd_dv_kind,
            "ffd_box_tag": box_tag,
            "ffd_marker": marker,
            "ffd_domain_mode": domain_mode,
            "ffd_control_row": control_row,
            "ffd_direction": direction,
            "ffd_allow_external_columns": allow_external_columns,
            "ffd_active_xmin": active_xmin,
            "ffd_active_xmax": active_xmax,
            "ffd_active_include_bounds": initial_include_bounds,
            "marker": marker,
        }
    )

    if opts.get("enabled", False):
        print(f"[PROGRESSIVE_FFD] FFD DV kind = {ffd_dv_kind}")
        print(f"[PROGRESSIVE_FFD] box tag = {box_tag}")
        print(f"[PROGRESSIVE_FFD] marker = {marker}")
        print(f"[PROGRESSIVE_FFD] domain mode = {domain_mode}")
        if control_row is not None:
            print(f"[PROGRESSIVE_FFD] control row = {control_row}")
        print(f"[PROGRESSIVE_FFD] direction = {direction}")
        print(
            "[PROGRESSIVE_FFD] external active columns = "
            f"{'YES' if allow_external_columns else 'NO'}"
        )
        print(
            "[PROGRESSIVE_FFD] configured active candidate x-range = "
            f"[{active_xmin}, {active_xmax}]"
        )

    return opts


def make_ffd_definition(columns, opts, column_index_by_x):
    ffd_dv_kind = str(opts.get("ffd_dv_kind", "FFD_CONTROL_POINT_2D")).upper()
    marker = str(opts.get("ffd_marker", opts.get("marker", "AIRFOIL")))
    box_tag = str(opts.get("ffd_box_tag", "AIRFOIL_BOX"))
    scale = float(opts.get("scale", 1.0))

    kinds = []
    scales = []
    markers = []
    ffdtags = []
    params = []
    sizes = []

    for x in [float(v) for v in columns]:
        i_index = _lookup_column_index(x, column_index_by_x)
        kinds.append(ffd_dv_kind)
        scales.append(scale)
        markers.append([marker])
        ffdtags.append(box_tag)
        sizes.append(1)

        if ffd_dv_kind == "FFD_CONTROL_POINT_2D":
            control_row = int(opts.get("ffd_control_row"))
            direction = str(opts.get("ffd_direction", "Y")).upper()
            dx, dy = (1.0, 0.0) if direction == "X" else (0.0, 1.0)
            params.append([int(i_index), control_row, dx, dy])
        elif ffd_dv_kind == "FFD_THICKNESS_2D":
            params.append([int(i_index)])
        else:
            raise NotImplementedError(
                f"Progressive FFD does not support DV kind {ffd_dv_kind}"
            )

    return {
        "KIND": kinds,
        "SCALE": scales,
        "MARKER": markers,
        "FFDTAG": ffdtags,
        "PARAM": params,
        "SIZE": sizes,
    }


def _ffd_dump_padded_params(kind, params):
    kind = str(kind).upper()
    params = list(params)

    expected_clean_sizes = {
        "FFD_CONTROL_POINT_2D": 4,
        "FFD_THICKNESS_2D": 1,
    }
    if kind not in expected_clean_sizes:
        return params

    if len(params) == expected_clean_sizes[kind]:
        return [0.0] + params

    return params


def make_ffd_config_dump_compatible(config):
    """
    SU2.io.Config.dump has a legacy FFD convention: it writes FFDTAG and then
    skips PARAM[0].  The progressive FFD backend stores clean PARAM values, so
    use a padded copy only when handing a config to that dumper.
    """
    cfg = copy.deepcopy(config)
    if "DEFINITION_DV" not in cfg:
        return cfg

    def_dv = cfg["DEFINITION_DV"]
    if not isinstance(def_dv, dict) or "KIND" not in def_dv or "PARAM" not in def_dv:
        return cfg

    params = []
    for kind, param in zip(def_dv["KIND"], def_dv["PARAM"]):
        params.append(_ffd_dump_padded_params(kind, param))
    def_dv["PARAM"] = params

    return cfg


def _lookup_column_index(x, column_index_by_x, tol=1.0e-10):
    x = float(x)
    for key, value in column_index_by_x.items():
        if abs(x - float(key)) <= tol:
            return int(value)
    raise KeyError(f"FFD column {x} is not present in the rewritten mesh")


def _compute_adaptive_nadd(current_ndv, ncandidates, growth_ratio):
    if ncandidates <= 0:
        return 0
    growth_ratio = float(growth_ratio)
    if growth_ratio <= 1.0:
        target_ndv = current_ndv + 1
    else:
        target_ndv = int(math.ceil(growth_ratio * current_ndv))
    return min(max(1, target_ndv - current_ndv), ncandidates)


def _selected_by_side(selected):
    selected_by_side = {}
    for item in selected:
        selected_by_side.setdefault(str(item["side"]), []).append(float(item["x"]))
    return selected_by_side


def _passes_ffd_min_spacing(candidate, selected, active_centers_by_side, opts):
    min_spacing = float(opts.get("min_center_spacing", 0.0))
    if min_spacing <= 0.0:
        return True

    side = str(candidate["side"])
    x = float(candidate["x"])
    reference = []
    reference.extend(float(v) for v in active_centers_by_side.get(side, []))
    reference.extend(float(v) for v in _selected_by_side(selected).get(side, []))
    reference.extend(ffd_active_range_from_opts(opts))

    nearest = None
    nearest_distance = None
    for value in reference:
        dist = abs(x - value)
        if nearest_distance is None or dist < nearest_distance:
            nearest = value
            nearest_distance = dist

    if nearest_distance is not None and nearest_distance < min_spacing:
        print(
            "[PROGRESSIVE_FFD] Candidate rejected by min spacing | "
            f"side={side} x={x:.6f} nearest={nearest:.6f} "
            f"dist={nearest_distance:.6f} required={min_spacing:.6f}"
        )
        return False

    return True


def _ranked_with_min_spacing(ranked, nadd, active_centers_by_side, opts):
    selected = []
    for candidate in ranked:
        if len(selected) >= nadd:
            break
        if not _passes_ffd_min_spacing(
            candidate,
            selected,
            active_centers_by_side,
            opts,
        ):
            continue
        selected.append(candidate)
    return selected


def select_ffd_candidates_by_nadd_mode(
    candidates,
    current_ndv,
    opts,
    active_centers_by_side,
):
    if not candidates:
        return []

    nfinal = opts.get("nfinal", None)
    n_remaining = None
    if nfinal is not None:
        n_remaining = int(nfinal) - int(current_ndv)
        if n_remaining <= 0:
            return []

    mode = str(opts.get("nadd_mode", "GROWTH_RATIO")).upper()
    ranked = sorted(candidates, key=lambda c: (-float(c["indicator"]), float(c["x"])))

    if mode == "GROWTH_RATIO":
        nadd = _compute_adaptive_nadd(
            current_ndv,
            len(candidates),
            opts.get("growth_ratio", 2.0),
        )
        if n_remaining is not None:
            nadd = min(nadd, n_remaining)
        return _ranked_with_min_spacing(ranked, nadd, active_centers_by_side, opts)

    if mode == "FIXED":
        nadd = int(opts.get("fixed_nadd", 1))
        if n_remaining is not None:
            nadd = min(nadd, n_remaining)
        return _ranked_with_min_spacing(ranked, nadd, active_centers_by_side, opts)

    if mode == "SCORE_BATCH":
        max_batch_size = int(opts.get("batch_size_max", 1))
        if n_remaining is not None:
            max_batch_size = min(max_batch_size, n_remaining)
        score_rel_tol = float(opts.get("batch_score_rel_tol", 0.85))
        min_separation = float(opts.get("batch_min_separation", 0.04))
        selected = []
        best_indicator = float(ranked[0]["indicator"])
        threshold = score_rel_tol * best_indicator
        for candidate in ranked:
            if len(selected) >= max_batch_size:
                break
            if float(candidate["indicator"]) < threshold:
                continue
            if not _passes_ffd_min_spacing(
                candidate,
                selected,
                active_centers_by_side,
                opts,
            ):
                continue
            if any(
                abs(float(candidate["x"]) - float(prev["x"])) < min_separation
                for prev in selected
            ):
                continue
            selected.append(candidate)
        return selected

    raise ValueError(f"Unknown progressive candidate addition mode: {mode}")


def _spacing_stats(xs):
    xs = sorted(float(x) for x in xs)
    if len(xs) < 2:
        return None
    dx = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    return {"min": min(dx), "max": max(dx), "ratio": max(dx) / min(dx)}


def _cap_ffd_refinement(prev_level, columns, opts):
    columns = validate_active_ffd_columns(
        sorted(set(float(x) for x in columns)),
        xmin=float(getattr(prev_level, "active_xmin", ffd_active_range_from_opts(opts)[0])),
        xmax=float(getattr(prev_level, "active_xmax", ffd_active_range_from_opts(opts)[1])),
        include_bounds=bool(
            getattr(
                prev_level,
                "active_include_bounds",
                ffd_active_include_bounds_from_opts(opts),
            )
        ),
    )
    nfinal = opts.get("nfinal", None)
    if nfinal is None:
        return columns

    n_remaining = int(nfinal) - int(prev_level.ndv)
    if n_remaining <= 0:
        return sorted(prev_level.columns)

    old_columns = sorted(float(x) for x in prev_level.columns)
    old_set = set(old_columns)
    additions = sorted(x for x in columns if x not in old_set)
    keep = additions[:n_remaining]
    return sorted(set(old_columns + keep))


def _uniform_ffd_refinement(prev_level, opts):
    active_xmin = float(getattr(prev_level, "active_xmin", ffd_active_range_from_opts(opts)[0]))
    active_xmax = float(getattr(prev_level, "active_xmax", ffd_active_range_from_opts(opts)[1]))
    extended = _sorted_unique_with_tolerance(
        [active_xmin] + list(prev_level.columns) + [active_xmax]
    )
    new_points = []
    for i in range(len(extended) - 1):
        xm = 0.5 * (extended[i] + extended[i + 1])
        if active_xmin < xm < active_xmax:
            new_points.append(xm)
    columns = sorted(set(list(prev_level.columns) + new_points))
    return _cap_ffd_refinement(prev_level, columns, opts)


def _spring_redistribute_ffd_columns(columns, scores, opts):
    xmin, xmax = ffd_active_range_from_opts(opts)
    span = float(xmax) - float(xmin)
    if span <= 0.0:
        raise ValueError("Invalid FFD active range for spring redistribution")

    normalized = [(float(x) - float(xmin)) / span for x in columns]
    redistributed = spring_redistribute_centers(
        normalized,
        scores,
        A=float(opts.get("spring_A", 20.0)),
    )
    include_bounds = ffd_active_include_bounds_from_opts(opts)
    return validate_active_ffd_columns(
        [float(xmin) + span * float(x) for x in redistributed],
        xmin=xmin,
        xmax=xmax,
        include_bounds=include_bounds,
    )


def refine_ffd_columns(prev_level, result, opts):
    if str(opts.get("refinement", "UNIFORM")).upper() != "ADAPTIVE":
        opts["_last_selection_metadata"] = None
        return _uniform_ffd_refinement(prev_level, opts)

    return _refine_ffd_adaptive(prev_level, result, opts)


def _refine_ffd_adaptive(prev_level, result, opts):
    from SU2.opt.progressive_ffd_projection import _compute_ffd_dot_candidate_scores

    current_ndv = prev_level.ndv
    opts["_last_selection_metadata"] = None

    try:
        scoring = _compute_ffd_dot_candidate_scores(prev_level, opts)
        candidates = scoring.get("candidates", [])
        active_scores = scoring.get("active_scores", [])
    except Exception as err:
        print(
            "[PROGRESSIVE_FFD] WARNING: ADAPTIVE refine failed -> "
            f"fallback to UNIFORM | {err}"
        )
        return _uniform_ffd_refinement(prev_level, opts)

    if not candidates:
        if scoring.get("spacing_filtered_empty", False):
            print(
                "[PROGRESSIVE_FFD] ADAPTIVE refine | "
                "no valid candidates after min-spacing filtering -> no refinement"
            )
            return sorted(prev_level.columns)
        print("[PROGRESSIVE_FFD] ADAPTIVE refine | no candidates -> fallback to UNIFORM")
        return _uniform_ffd_refinement(prev_level, opts)

    active_centers_by_side = {"FFD": sorted(prev_level.columns)}
    chosen = select_ffd_candidates_by_nadd_mode(
        candidates,
        current_ndv,
        opts,
        active_centers_by_side=active_centers_by_side,
    )

    if not chosen:
        nfinal = opts.get("nfinal", None)
        if nfinal is not None and int(nfinal) - current_ndv <= 0:
            print(
                "[PROGRESSIVE_FFD] ADAPTIVE refine | "
                f"NFINAL reached (ndv={current_ndv}, nfinal={nfinal}) -> no refinement"
            )
        else:
            print("[PROGRESSIVE_FFD] ADAPTIVE refine | no candidates selected")
        return sorted(prev_level.columns)

    spring_enabled = bool(opts.get("spring_enabled", False))
    spring_timing = str(opts.get("spring_timing", "POST_OPT")).upper()
    spring_score_mode = str(opts.get("spring_score_mode", "COEFFICIENT")).upper()
    spring_debug = None

    if spring_enabled and spring_timing == "PRE_REFINE":
        if spring_score_mode != "INDICATOR":
            print(
                "[PROGRESSIVE_FFD][SPRING] PRE_REFINE requested without "
                "indicator score mode -> using candidate indicators"
            )
        columns_before = sorted(prev_level.columns)
        combined_columns = columns_before + [float(c["x"]) for c in chosen]
        combined_scores = list(active_scores) + [float(c["indicator"]) for c in chosen]
        new_columns = _spring_redistribute_ffd_columns(
            combined_columns,
            combined_scores,
            opts,
        )
        spring_debug = {
            "columns_before": columns_before,
            "selected": [float(c["x"]) for c in chosen],
            "columns_after": new_columns,
            "spacing": _spacing_stats(new_columns),
        }
    else:
        new_columns = sorted(
            set(list(prev_level.columns) + [float(c["x"]) for c in chosen])
        )

    new_columns = _cap_ffd_refinement(prev_level, new_columns, opts)
    ndv_after = len(new_columns)
    best_indicator = max(float(c["indicator"]) for c in chosen)
    ratios = [
        float(c["indicator"]) / best_indicator if best_indicator != 0.0 else 0.0
        for c in chosen
    ]

    print(
        "[PROGRESSIVE_FFD] ADAPTIVE batch summary | "
        f"mode={opts.get('nadd_mode', 'GROWTH_RATIO')} "
        f"ndv_before={current_ndv} ndv_after={ndv_after} "
        f"n_added={max(0, ndv_after - current_ndv)} "
        f"columns={[round(float(x), 6) for x in new_columns]}"
    )

    for c in chosen:
        print(
            "[PROGRESSIVE_FFD] ADAPTIVE selected | "
            f"side={c['side']} x={float(c['x']):.6f} "
            f"I={float(c['indicator']):.6e}"
        )

    if spring_debug is not None:
        if spring_debug["spacing"] is not None:
            print(
                "[PROGRESSIVE_FFD][SPRING] spacing | "
                f"min={spring_debug['spacing']['min']:.6f} "
                f"max={spring_debug['spacing']['max']:.6f} "
                f"ratio={spring_debug['spacing']['ratio']:.2f}"
            )
        print("[PROGRESSIVE_FFD][SPRING] Columns before:", spring_debug["columns_before"])
        print("[PROGRESSIVE_FFD][SPRING] Columns selected:", spring_debug["selected"])
        print("[PROGRESSIVE_FFD][SPRING] Columns after :", spring_debug["columns_after"])

    opts["_last_selection_metadata"] = {
        "level_id": prev_level.level_id,
        "ndv_before": current_ndv,
        "ndv_after": ndv_after,
        "n_added": max(0, ndv_after - current_ndv),
        "nadd_mode": opts.get("nadd_mode", "GROWTH_RATIO"),
        "trigger_mode": opts.get("trigger", "MAX_ITER"),
        "refinement": opts.get("refinement", "UNIFORM"),
        "spring_enabled": spring_enabled,
        "spring_timing": spring_timing,
        "spring_score_mode": spring_score_mode,
        "post_opt_spring_pending": spring_enabled and spring_timing == "POST_OPT",
        "columns_before": sorted(prev_level.columns),
        "columns_after": sorted(new_columns),
        "upper_before": sorted(prev_level.columns),
        "lower_before": [],
        "upper_after": sorted(new_columns),
        "lower_after": [],
        "selected": [
            {
                "side": c["side"],
                "x": float(c["x"]),
                "indicator": float(c["indicator"]),
                "indicator_ratio_to_best": ratios[i],
                "interval_id": c.get("interval_id"),
                "interval_left": c.get("interval_left"),
                "interval_right": c.get("interval_right"),
                "sample_index": c.get("sample_index"),
                "sample_fraction": c.get("sample_fraction"),
                "rejected_reason": c.get("rejected_reason", ""),
                "nearest_center_or_boundary": c.get(
                    "nearest_center_or_boundary", ""
                ),
                "nearest_distance": c.get("nearest_distance", ""),
                "required_spacing": c.get("required_spacing", ""),
            }
            for i, c in enumerate(chosen)
        ],
    }

    return sorted(new_columns)


def apply_post_opt_ffd_spring(level, result, opts):
    dv_values = result.get("dv_values", None)
    if dv_values is None:
        print(
            "[PROGRESSIVE_FFD][SPRING] WARNING: missing optimized DV values; "
            "post-opt spring skipped"
        )
        return None

    if len(dv_values) != level.ndv:
        print(
            "[PROGRESSIVE_FFD][SPRING] WARNING: optimized DV size mismatch; "
            f"got {len(dv_values)}, expected {level.ndv}; post-opt spring skipped"
        )
        return None

    coeff_abs = [abs(float(v)) for v in dv_values]
    try:
        new_columns = _spring_redistribute_ffd_columns(
            level.columns,
            coeff_abs,
            opts,
        )
    except Exception as err:
        print(
            "[PROGRESSIVE_FFD][SPRING] WARNING: post-opt coefficient spring failed; "
            f"{err}"
        )
        return None

    print("[PROGRESSIVE_FFD][SPRING] POST_OPT coefficient spring")
    print("[PROGRESSIVE_FFD][SPRING] column |dv| =", coeff_abs)
    print("[PROGRESSIVE_FFD][SPRING] Columns before:", sorted(level.columns))
    print("[PROGRESSIVE_FFD][SPRING] Columns after :", new_columns)
    result["spring_ffd_coeff_abs"] = coeff_abs
    return sorted(new_columns)


def initial_ffd_columns_from_config(base_config, opts):
    active_xmin, active_xmax = ffd_active_range_from_opts(opts)
    include_bounds = ffd_active_include_bounds_from_opts(opts)
    columns = _parse_ffd_initial_columns(
        base_config.get("PROGRESSIVE_FFD_INITIAL_COLUMNS", None),
        xmin=active_xmin,
        xmax=active_xmax,
        include_bounds=include_bounds,
    )
    if columns is not None:
        return columns

    columns = _parse_ffd_initial_columns(
        base_config.get("PROGRESSIVE_HH_INITIAL_UPPER", None),
        xmin=active_xmin,
        xmax=active_xmax,
        include_bounds=include_bounds,
    )
    if columns is not None:
        return columns

    if _ffd_allow_external_columns_from_opts(opts):
        n0 = int(opts.get("n0", 3))
        columns = [
            active_xmin + (active_xmax - active_xmin) * x
            for x in initial_centers(n0)
        ]
        return validate_active_ffd_columns(
            columns,
            xmin=active_xmin,
            xmax=active_xmax,
            include_bounds=include_bounds,
        )

    return validate_active_ffd_columns(initial_centers(int(opts.get("n0", 3))))
