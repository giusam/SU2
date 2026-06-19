#!/usr/bin/env python

"""Progressive/adaptive external B-spline optimization driver for SU2."""

import argparse
import csv
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from SU2.opt.bspline_dot import (
    match_sensitivities_to_metadata,
    normalize_sensitivity_weighting,
    read_metadata,
    read_sensitivity_file,
)
from SU2.opt.bspline_modes import (
    ALLOWED_DEFORMATION_DIRECTION_MODES,
    ALLOWED_SURFACE_MODES,
    BSplineModeError,
    LE_SAFE_DEFAULT_POWER,
    LE_SAFE_DEFAULT_X0,
    LE_SAFE_DEFAULT_X1,
    active_sides_from_surface_mode,
    clamped_basis_count,
    evaluate_all_modes,
    load_mode_spec,
    mode_normalization_factor,
    normalize_deformation_direction_mode,
    normalize_surface_mode,
    validate_le_safe_direction_options,
    validate_mode_spec,
    validate_surface_mode_against_modes,
)
from SU2.opt.bspline_su2_driver import (
    ALLOWED_EVAL_LAYOUTS,
    ALLOWED_SYMMETRY_COUPLINGS,
    BSplineSU2DriverError,
    active_coefficient_vector,
    active_mode_ids,
    apply_optimizer_config_to_args,
    cache_key,
    fixed_driver_options_from_config,
    read_objective_from_history,
    resolve_thickness_domain_mode,
    run_bspline_su2_optimization,
    write_mode_spec,
)
from SU2.opt.progressive_trigger import build_online_trigger_opts


class BSplineAdaptiveError(RuntimeError):
    pass


# The adaptive B-spline optimizer is knot-insertion-only. These are fixed
# internally and are not user-configurable; candidate/generated refinement
# has been removed.
REFINE_MODE = "KNOT_INSERTION"
REFINE_STATE = "INITIAL_MESH_KEEP_DV"
# NOTE: SLOPE_EFFICIENCY_FILTERED is accepted only for backward compatibility
# and is mapped internally to SLOPE_EFFICIENCY_TRIGGER.
ALLOWED_TRIGGERS = (
    "MAX_ITER",
    "SLOPE_EFFICIENCY_TRIGGER",
    "SLOPE_EFFICIENCY_FILTERED",
    "SLOPE_EFFICIENCY_BEST_LOG",
    "STAGNATION_TRIGGER",
)
ALLOWED_NADD_MODES = ("GROWTH_RATIO", "FIXED")
ALLOWED_REFINE_MODES = ("KNOT_INSERTION",)
ALLOWED_KNOT_SCORE_MODES = ("VIRTUAL_INSERTION", "RESIDUAL_ENERGY")
REMOVED_CANDIDATE_CONFIG_KEYS = (
    "BSPLINE_SCORE_MODE",
    "BSPLINE_CANDIDATE_SOURCE",
    "BSPLINE_GENERATED_PEAKS_PER_SIDE",
    "BSPLINE_GENERATED_WIDTHS",
    "BSPLINE_GENERATED_MIN_SEPARATION",
    "BSPLINE_GENERATED_XMIN",
    "BSPLINE_GENERATED_XMAX",
    "BSPLINE_EDGE_XLE",
    "BSPLINE_EDGE_XTE",
    "BSPLINE_ROUGH_LAMBDA",
    "BSPLINE_ROUGH_POWER",
    "BSPLINE_BATCH_SCORE_REL_TOL",
)
GLOBAL_MODE_KEYS = (
    "version",
    "dimension",
    "marker",
    "chord",
    "normal_displacement",
    "class_shape",
    "normalize_basis",
    "normalization_mode",
    "surface_mode",
)
KNOT_SCORE_FIELDNAMES = [
    "batch_step",
    "rank",
    "span_left",
    "span_right",
    "span_width",
    "inserted_knot",
    "side",
    "score_mode",
    "score",
    "score_raw",
    "residual_energy",
    "incremental_rank",
    "incremental_columns",
    "condition_number",
    "selected",
    "status",
]


@dataclass
class BsplineLevel:
    level_id: int
    workdir: Path
    opt_workdir: Path
    active_modes: dict
    active_modes_start_filename: Path
    optimized_modes_filename: Path
    selection_metadata: dict
    initial_modes_source: Path

    @property
    def ndv(self):
        return len(self.active_mode_ids)

    @property
    def active_mode_ids(self):
        return active_mode_ids(self.active_modes)


@dataclass
class TriggerDecision:
    trigger_mode: str
    metric: object = ""
    threshold: object = ""
    window: object = ""
    patience: object = ""
    counter: int = 0
    refine_now: bool = False
    reason: str = ""


def _as_float(value, name):
    try:
        result = float(value)
    except Exception:
        raise BSplineAdaptiveError(f"{name} must be numeric")
    if not math.isfinite(result):
        raise BSplineAdaptiveError(f"{name} must be finite")
    return result


def _as_bool(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().upper()
    if text in ("YES", "TRUE", "1", "ON"):
        return True
    if text in ("NO", "FALSE", "0", "OFF"):
        return False
    raise BSplineAdaptiveError(f"expected YES/NO boolean value, got {value!r}")


def _as_float_list(value, name):
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if (text.startswith("(") and text.endswith(")")) or (
            text.startswith("[") and text.endswith("]")
        ):
            text = text[1:-1]
        tokens = [token for token in text.replace(",", " ").split() if token]
    elif isinstance(value, (list, tuple)):
        tokens = list(value)
    else:
        tokens = [value]
    return [_as_float(token, name) for token in tokens]


def _active_modes(mode_spec):
    validate_mode_spec(mode_spec)
    return [
        mode
        for mode in mode_spec.get("modes", [])
        if mode.get("active", True) is not False
    ]


def _mode_support(mode):
    basis_type = str(mode.get("basis_type", "")).strip().lower()
    if basis_type == "clamped":
        knots = mode.get("knot_vector", mode.get("knots"))
        if knots is not None and "basis_index" in mode:
            degree = int(mode.get("degree", 3))
            index = int(mode["basis_index"])
            knots = [float(value) for value in knots]
            right_index = min(len(knots) - 1, index + degree + 1)
            left = knots[index]
            right = knots[right_index]
            return left, right, 0.5 * (left + right)

    return 0.0, 1.0, 0.5


def mode_sort_key(mode):
    side_order = {"upper": 0, "lower": 1}
    left, _right, center = _mode_support(mode)
    side = str(mode.get("side", "")).strip().lower()
    return (
        side_order.get(side, 2),
        float(center),
        float(left),
        str(mode.get("id", "")),
    )


def _copy_global_metadata(source_spec, modes):
    return {
        key: source_spec[key]
        for key in GLOBAL_MODE_KEYS
        if key in source_spec
    } | {"modes": list(modes)}


def _mode_with_zero_coefficient(mode):
    new_mode = dict(mode)
    new_mode["coefficient"] = 0.0
    new_mode["active"] = True
    return new_mode


def build_level(level_id, active_modes, workdir, initial_modes_source):
    level_dir = Path(workdir) / f"LEVEL_{int(level_id):03d}"
    return BsplineLevel(
        level_id=int(level_id),
        workdir=level_dir,
        opt_workdir=level_dir / "opt_run",
        active_modes=active_modes,
        active_modes_start_filename=level_dir / "active_modes_start.json",
        optimized_modes_filename=level_dir / "opt_run" / "optimized_modes.json",
        selection_metadata={},
        initial_modes_source=Path(initial_modes_source),
    )


def write_level_start(level):
    level.workdir.mkdir(parents=True, exist_ok=True)
    write_mode_spec(level.active_modes, level.active_modes_start_filename)


def build_next_active_modes(optimized_modes, selected_modes):
    optimized = validate_mode_spec(optimized_modes)
    existing_ids = set()
    modes = []
    for mode in _active_modes(optimized):
        copied = dict(mode)
        copied["active"] = True
        existing_ids.add(str(copied["id"]))
        modes.append(copied)

    for mode in selected_modes:
        mode_id = str(mode.get("id", ""))
        if mode_id in existing_ids:
            continue
        modes.append(_mode_with_zero_coefficient(mode))
        existing_ids.add(mode_id)

    ordered = sorted(modes, key=mode_sort_key)
    return validate_mode_spec(_copy_global_metadata(optimized, ordered))


def _spec_with_modes(template_spec, modes):
    prepared = []
    for mode in modes:
        copied = dict(mode)
        copied["active"] = True
        if "coefficient" not in copied:
            copied["coefficient"] = 0.0
        prepared.append(copied)
    return validate_mode_spec(_copy_global_metadata(template_spec, prepared))


def evaluate_basis_matrix(template_spec, modes, metadata):
    if not modes:
        return np.zeros((len(metadata), 0), dtype=float)
    spec = _spec_with_modes(template_spec, modes)
    x_over_c = [record["x_over_c"] for record in metadata]
    sides = [record["side"] for record in metadata]
    values = evaluate_all_modes(spec, x_over_c, sides=sides)
    columns = [np.asarray(values[str(mode["id"])], dtype=float) for mode in modes]
    return np.column_stack(columns) if columns else np.zeros((len(metadata), 0), dtype=float)


@dataclass
class ClampedSideGroup:
    side: str
    degree: int
    knot_vector: tuple
    modes: list

    @property
    def n_basis(self):
        return len(self.modes)

    @property
    def coefficients(self):
        return np.asarray(
            [float(mode.get("coefficient", 0.0)) for mode in self.modes],
            dtype=float,
        )


@dataclass
class ClampedKnotSpace:
    spec: dict
    groups: dict
    sides: tuple
    degree: int
    knot_vector: tuple
    coupling: str

    @property
    def modes(self):
        modes = []
        for side in self.sides:
            modes.extend(self.groups[side].modes)
        return modes


def _rounded_knots(knots):
    return tuple(round(float(value), 14) for value in knots)


def _side_order(side):
    order = {"upper": 0, "lower": 1}
    return order.get(str(side).strip().lower(), 2)


def _bounds_signature(mode):
    bounds = mode.get("bounds")
    if bounds is None:
        return None
    if len(bounds) != 2:
        return tuple(bounds)
    return tuple(round(float(value), 14) for value in bounds)


def extract_clamped_knot_space(mode_spec, settings=None):
    settings = dict(settings or {})
    coupling = str(settings.get("symmetry_coupling", "NONE")).upper()
    try:
        spec = validate_mode_spec(mode_spec)
        surface_mode = normalize_surface_mode(settings.get("surface_mode", "BOTH"))
        validate_surface_mode_against_modes(spec, surface_mode)
    except BSplineModeError as exc:
        raise BSplineAdaptiveError(str(exc))

    if int(spec.get("dimension", 2)) != 2:
        raise BSplineAdaptiveError("KNOT_INSERTION v1 supports only dimension=2")
    if spec.get("normal_displacement", True) is not True:
        raise BSplineAdaptiveError("KNOT_INSERTION v1 requires normal_displacement=true")
    if normalize_sensitivity_weighting(settings.get("sensitivity_weighting", "NODAL")) != "NODAL":
        raise BSplineAdaptiveError("KNOT_INSERTION v1 supports only NODAL sensitivity weighting")

    by_side = {}
    for mode in _active_modes(spec):
        basis_type = str(mode.get("basis_type", "")).strip().lower()
        if basis_type != "clamped":
            raise BSplineAdaptiveError(
                "Only basis_type='clamped' is supported by the knot-insertion B-spline workflow."
            )
        degree = int(mode.get("degree", 3))
        if degree != 3:
            raise BSplineAdaptiveError("KNOT_INSERTION v1 supports only degree 3")
        side = str(mode.get("side", "")).strip().lower()
        if side not in ("upper", "lower"):
            raise BSplineAdaptiveError("KNOT_INSERTION requires active modes on side upper/lower")
        by_side.setdefault(side, []).append(dict(mode, active=True))

    if not by_side:
        raise BSplineAdaptiveError("KNOT_INSERTION requires at least one active clamped mode")
    if coupling in ("NORMAL_EQUAL", "NORMAL_OPPOSITE") and set(by_side) != {"upper", "lower"}:
        raise BSplineAdaptiveError(
            f"{coupling} KNOT_INSERTION requires paired upper and lower clamped bases"
        )
    expected_sides = set(active_sides_from_surface_mode(surface_mode))
    if surface_mode != "BOTH" and set(by_side) != expected_sides:
        raise BSplineAdaptiveError(
            f"BSPLINE_SURFACE_MODE={surface_mode} requires active side "
            f"{surface_mode.lower()!r}"
        )

    groups = {}
    for side, modes in by_side.items():
        modes = sorted(modes, key=lambda mode: int(mode["basis_index"]))
        degree = int(modes[0].get("degree", 3))
        knot_vector = _rounded_knots(modes[0].get("knot_vector", modes[0].get("knots")))
        n_basis = clamped_basis_count(degree, knot_vector)
        indices = [int(mode["basis_index"]) for mode in modes]
        expected = list(range(n_basis))
        if indices != expected:
            raise BSplineAdaptiveError(
                f"KNOT_INSERTION side {side!r} must contain complete basis indices {expected}; got {indices}"
            )

        for mode in modes:
            if int(mode.get("degree", 3)) != degree:
                raise BSplineAdaptiveError(f"KNOT_INSERTION side {side!r} mixes degrees")
            if _rounded_knots(mode.get("knot_vector", mode.get("knots"))) != knot_vector:
                raise BSplineAdaptiveError(f"KNOT_INSERTION side {side!r} mixes knot vectors")

        groups[side] = ClampedSideGroup(
            side=side,
            degree=degree,
            knot_vector=knot_vector,
            modes=modes,
        )

    sides = tuple(sorted(groups, key=_side_order))
    degree = groups[sides[0]].degree
    knot_vector = groups[sides[0]].knot_vector
    for side in sides:
        group = groups[side]
        if group.degree != degree:
            raise BSplineAdaptiveError("KNOT_INSERTION active clamped groups mix degrees")
        if group.knot_vector != knot_vector:
            raise BSplineAdaptiveError(
                "KNOT_INSERTION v1 requires all active clamped side groups to share the same knot vector"
            )

    return ClampedKnotSpace(
        spec=spec,
        groups=groups,
        sides=sides,
        degree=degree,
        knot_vector=knot_vector,
        coupling=coupling,
    )


def knot_insertion_spans(knot_vector, min_width=1.0e-8):
    knots = [float(value) for value in knot_vector]
    spans = []
    for left, right in zip(knots[:-1], knots[1:]):
        if float(right) - float(left) > float(min_width):
            spans.append((float(left), float(right), 0.5 * (float(left) + float(right))))
    return spans


def insert_knot_midpoint(knot_vector, span):
    inserted = float(span[2])
    new_knots = [float(value) for value in knot_vector]
    new_knots.append(inserted)
    new_knots.sort()
    return tuple(new_knots)


def _representative_bounds(group):
    counts = {}
    by_key = {}
    for mode in group.modes:
        key = _bounds_signature(mode)
        counts[key] = counts.get(key, 0) + 1
        by_key[key] = mode.get("bounds")
    best_key = max(counts, key=lambda key: (counts[key], key is not None))
    bounds = by_key[best_key]
    return list(bounds) if bounds is not None else None


def _mode_template_for_side(group, basis_index, knot_vector, coefficient=0.0):
    old_by_index = {int(mode["basis_index"]): mode for mode in group.modes}
    source = old_by_index.get(int(basis_index), group.modes[min(int(basis_index), len(group.modes) - 1)])
    mode = {
        key: value
        for key, value in source.items()
        if key
        not in (
            "id",
            "knot_vector",
            "knots",
            "basis_index",
            "coefficient",
            "active",
            "normalization_factor",
        )
    }
    mode.update(
        {
            "id": f"{group.side}_clamped_i{int(basis_index):03d}",
            "side": group.side,
            "basis_type": "clamped",
            "degree": group.degree,
            "knot_vector": [float(value) for value in knot_vector],
            "basis_index": int(basis_index),
            "coefficient": float(coefficient),
            "active": True,
        }
    )
    bounds = source.get("bounds")
    if bounds is None:
        bounds = _representative_bounds(group)
    if bounds is not None:
        mode["bounds"] = list(bounds)
    return mode


def regenerate_clamped_modes(space, knot_vector, coefficients_by_side=None):
    coefficients_by_side = coefficients_by_side or {}
    modes = []
    for side in space.sides:
        old_group = space.groups[side]
        n_basis = clamped_basis_count(space.degree, knot_vector)
        coeffs = coefficients_by_side.get(side, np.zeros(n_basis, dtype=float))
        if len(coeffs) != n_basis:
            raise BSplineAdaptiveError(
                f"expected {n_basis} transferred coefficients for side {side}, got {len(coeffs)}"
            )
        for basis_index in range(n_basis):
            modes.append(
                _mode_template_for_side(
                    old_group,
                    basis_index,
                    knot_vector,
                    coefficient=float(coeffs[basis_index]),
                )
            )
    return validate_mode_spec(_copy_global_metadata(space.spec, modes))


def _basis_for_spec_modes(spec, metadata):
    modes = _active_modes(spec)
    matrix = evaluate_basis_matrix(spec, modes, metadata)
    return modes, matrix


def reduced_basis_matrix_for_space(space, spec, metadata):
    modes, full_matrix = _basis_for_spec_modes(spec, metadata)
    if space.coupling == "NONE":
        return modes, full_matrix

    n_basis = clamped_basis_count(space.degree, _rounded_knots(modes[0]["knot_vector"]))
    by_side_index = {}
    for column, mode in enumerate(modes):
        side = str(mode.get("side", "")).strip().lower()
        by_side_index[(side, int(mode["basis_index"]))] = column

    columns = []
    reduced_modes = []
    lower_sign = 1.0 if space.coupling == "NORMAL_EQUAL" else -1.0
    for basis_index in range(n_basis):
        upper_column = by_side_index.get(("upper", basis_index))
        lower_column = by_side_index.get(("lower", basis_index))
        if upper_column is None or lower_column is None:
            raise BSplineAdaptiveError(
                f"{space.coupling} KNOT_INSERTION missing paired basis_index {basis_index}"
            )
        columns.append(full_matrix[:, upper_column] + lower_sign * full_matrix[:, lower_column])
        reduced_modes.append(
            {
                "id": f"paired_clamped_i{basis_index:03d}",
                "side": "paired",
                "basis_type": "clamped",
                "basis_index": basis_index,
                "degree": space.degree,
            }
        )
    return reduced_modes, np.column_stack(columns) if columns else np.zeros((len(metadata), 0))


def _expanded_coefficients_from_reduced(space, knot_vector, reduced_coeffs):
    lower_sign = 1.0 if space.coupling == "NORMAL_EQUAL" else -1.0
    reduced_coeffs = np.asarray(reduced_coeffs, dtype=float)
    return {
        "upper": reduced_coeffs.copy(),
        "lower": lower_sign * reduced_coeffs,
    }


def coefficient_vector_for_space(space, spec):
    modes = _active_modes(spec)
    if space.coupling == "NONE":
        return np.asarray(
            [float(mode.get("coefficient", 0.0)) for mode in modes],
            dtype=float,
        )

    n_basis = clamped_basis_count(space.degree, space.knot_vector)
    by_side_index = {}
    for mode in modes:
        by_side_index[(str(mode.get("side", "")).strip().lower(), int(mode["basis_index"]))] = float(
            mode.get("coefficient", 0.0)
        )
    lower_sign = 1.0 if space.coupling == "NORMAL_EQUAL" else -1.0
    values = []
    for basis_index in range(n_basis):
        upper = by_side_index[("upper", basis_index)]
        lower = by_side_index[("lower", basis_index)]
        values.append(0.5 * (upper + lower_sign * lower))
    return np.asarray(values, dtype=float)


def project_onto_basis(matrix, values, regularization=1.0e-12):
    matrix = np.asarray(matrix, dtype=float)
    values = np.asarray(values, dtype=float)
    if matrix.size == 0 or matrix.shape[1] == 0:
        return np.zeros_like(values)
    try:
        coeffs, *_ = np.linalg.lstsq(matrix, values, rcond=None)
        if not np.all(np.isfinite(coeffs)):
            coeffs = _tikhonov_projection(matrix, values, regularization)
    except np.linalg.LinAlgError:
        coeffs = _tikhonov_projection(matrix, values, regularization)
    return matrix.dot(coeffs)


def _rank_incremental_columns(active_matrix, candidate_matrix, regularization):
    # Return an orthonormal basis for the truly new incremental subspace.
    #
    # Knot insertion can add several candidate columns, but after projection
    # onto the complement of the active basis many of them can be nearly
    # linearly dependent. Returning the raw residualized columns makes
    # Z.T @ Z nearly singular. We therefore residualize the block and compress
    # it with SVD to its numerical rank.
    residual_columns = []

    for column in range(candidate_matrix.shape[1]):
        z = residualize_candidate(
            active_matrix,
            candidate_matrix[:, column],
            regularization=regularization,
        )
        h = float(np.dot(z, z))
        if h > float(regularization) and math.isfinite(h):
            residual_columns.append(z)

    if not residual_columns:
        return np.zeros((candidate_matrix.shape[0], 0), dtype=float)

    Z = np.column_stack(residual_columns)

    try:
        U, S, _Vt = np.linalg.svd(Z, full_matrices=False)
    except np.linalg.LinAlgError:
        Q, R = np.linalg.qr(Z, mode="reduced")
        diag = np.abs(np.diag(R)) if R.ndim == 2 else np.array([])
        if diag.size == 0:
            return np.zeros((candidate_matrix.shape[0], 0), dtype=float)

        tol = max(
            float(regularization) ** 0.5,
            np.finfo(float).eps * max(Z.shape) * float(diag[0]),
        )
        rank = int(np.sum(diag > tol))

        if rank <= 0:
            return np.zeros((candidate_matrix.shape[0], 0), dtype=float)

        return Q[:, :rank]

    if S.size == 0:
        return np.zeros((candidate_matrix.shape[0], 0), dtype=float)

    tol = max(
        float(regularization) ** 0.5,
        np.finfo(float).eps * max(Z.shape) * float(S[0]),
    )
    rank = int(np.sum(S > tol))

    if rank <= 0:
        return np.zeros((candidate_matrix.shape[0], 0), dtype=float)

    return U[:, :rank]

def _score_virtual_insertion(active_matrix, new_matrix, signal, regularization):
    """Score a virtual knot insertion via the incremental-energy metric.

    Computes ``score = b^T G^{-1} b`` where ``G = Z^T Z`` is the Gram
    matrix of the residualised new columns and ``b = Z^T signal`` is the
    signal projected onto that incremental subspace. This is the
    energy-optimal projection of the adjoint residual onto the new
    subspace — not the same ``q^2 / h`` metric used for local
    candidate modes.  The difference matters: ``b^T G^{-1} b`` can
    favour ill-conditioned incremental bases (``G`` nearly singular)
    because the inverse amplifies the projection.  The
    ``condition_number`` returned by this function captures exactly
    that risk and should be consulted before accepting a span.
    """
    z_matrix = _rank_incremental_columns(active_matrix, new_matrix, regularization)
    if z_matrix.shape[1] == 0:
        return 0.0, 0.0, 0, 0, 0.0

    b = z_matrix.T.dot(signal)
    gram = z_matrix.T.dot(z_matrix)
    try:
        condition = float(np.linalg.cond(gram))
    except Exception:
        condition = math.inf
    try:
        solve = np.linalg.solve(gram, b)
    except np.linalg.LinAlgError:
        gram = gram + float(regularization) * np.eye(gram.shape[0])
        try:
            solve = np.linalg.solve(gram, b)
        except np.linalg.LinAlgError:
            solve = np.linalg.lstsq(gram, b, rcond=None)[0]
    score = float(b.dot(solve))
    if not math.isfinite(score) or score < 0.0:
        score = 0.0
    return score, score, int(np.linalg.matrix_rank(z_matrix)), int(z_matrix.shape[1]), condition


def score_knot_spans(space, metadata, signal, settings, regularization=1.0e-12):
    knot_score_mode = str(settings.get("knot_score_mode", "VIRTUAL_INSERTION")).upper()
    spans = knot_insertion_spans(
        space.knot_vector,
        min_width=settings.get("knot_min_span_width", 1.0e-8),
    )
    if not spans:
        raise BSplineAdaptiveError("KNOT_INSERTION found no non-degenerate knot spans")

    old_modes, old_matrix = reduced_basis_matrix_for_space(space, space.spec, metadata)
    signal = np.asarray(signal, dtype=float)
    residual = signal - project_onto_basis(old_matrix, signal, regularization=regularization)
    candidate_side = space.sides[0].upper() if len(space.sides) == 1 else "BOTH"
    rows = []
    for left, right, inserted in spans:
        new_knots = insert_knot_midpoint(space.knot_vector, (left, right, inserted))
        virtual_spec = regenerate_clamped_modes(space, new_knots)
        _new_modes, new_matrix = reduced_basis_matrix_for_space(space, virtual_spec, metadata)
        residual_energy = 0.0
        last_span = right == spans[-1][1]
        for value, row in zip(residual, metadata):
            x_over_c = float(row["x_over_c"])
            in_span = left <= x_over_c <= right if last_span else left <= x_over_c < right
            if in_span:
                residual_energy += float(value) * float(value)

        if knot_score_mode == "RESIDUAL_ENERGY":
            score_raw = float(residual_energy)
            score = score_raw
            rank = 0
            columns = 0
            condition = 0.0
        elif knot_score_mode == "VIRTUAL_INSERTION":
            score, score_raw, rank, columns, condition = _score_virtual_insertion(
                old_matrix,
                new_matrix,
                signal,
                regularization,
            )
            # Guard: if the incremental basis is numerically singular
            # (condition number infinite or NaN), the score from
            # b^T G^{-1} b is meaningless — skip this span.
            if not math.isfinite(condition) or condition > 1.0e14:
                score = 0.0
                score_raw = 0.0
        else:
            raise BSplineAdaptiveError(f"unsupported knot score mode {knot_score_mode!r}")

        rows.append(
            {
                "batch_step": "",
                "span_left": float(left),
                "span_right": float(right),
                "span_width": float(right) - float(left),
                "inserted_knot": float(inserted),
                "side": candidate_side,
                "score_mode": knot_score_mode,
                "score": float(score),
                "score_raw": float(score_raw),
                "residual_energy": float(residual_energy),
                "incremental_rank": rank,
                "incremental_columns": columns,
                "condition_number": condition,
                "selected": False,
                "status": "ok",
            }
        )

    rows.sort(key=lambda row: (-float(row["score"]), float(row["span_left"]), float(row["span_right"])))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    if rows and float(rows[0]["score"]) > 0.0 and math.isfinite(float(rows[0]["score"])):
        rows[0]["selected"] = True
    return rows


def _knot_multiplicity(knots, u, tol=1.0e-12):
    return sum(
        1
        for knot in knots
        if math.isclose(float(knot), float(u), rel_tol=0.0, abs_tol=float(tol))
    )


def _find_knot_span_for_insertion(knots, degree, coeff_count, u, tol=1.0e-12):
    knots = np.asarray(knots, dtype=float)
    degree = int(degree)
    coeff_count = int(coeff_count)
    n = coeff_count - 1
    if len(knots) != coeff_count + degree + 1:
        raise BSplineAdaptiveError(
            "invalid knot vector length for Boehm insertion: "
            f"knots={len(knots)} coeffs={coeff_count} degree={degree}"
        )

    lower = float(knots[degree])
    upper = float(knots[n + 1])
    u = float(u)
    if u < lower - float(tol) or u > upper + float(tol):
        raise BSplineAdaptiveError(
            f"cannot insert knot {u:.16g} outside [{lower:.16g}, {upper:.16g}]"
        )
    if math.isclose(u, upper, rel_tol=0.0, abs_tol=float(tol)):
        return n

    for k in range(degree, n + 1):
        if (
            u >= float(knots[k]) - float(tol)
            and u < float(knots[k + 1]) - float(tol)
        ):
            return k
    raise BSplineAdaptiveError(f"could not find a knot span for insertion at u={u:.16g}")


def _boehm_insert_once(knots, coeffs, degree, u, tol=1.0e-12):
    knots = np.asarray(knots, dtype=float)
    coeffs = np.asarray(coeffs, dtype=float)
    degree = int(degree)
    if coeffs.ndim != 1:
        raise BSplineAdaptiveError("Boehm insertion coefficients must be one-dimensional")
    if not np.all(np.isfinite(knots)) or not np.all(np.isfinite(coeffs)):
        raise BSplineAdaptiveError("Boehm insertion requires finite knots and coefficients")

    matching = [
        float(knot)
        for knot in knots
        if math.isclose(float(knot), float(u), rel_tol=0.0, abs_tol=float(tol))
    ]
    inserted = matching[0] if matching else float(u)
    multiplicity = _knot_multiplicity(knots, inserted, tol=tol)
    if multiplicity > degree:
        raise BSplineAdaptiveError(
            f"cannot insert knot {inserted:.16g}: current multiplicity "
            f"{multiplicity} exceeds degree {degree}"
        )

    n = len(coeffs) - 1
    k = _find_knot_span_for_insertion(
        knots,
        degree,
        len(coeffs),
        inserted,
        tol=tol,
    )
    new_coeffs = np.empty(len(coeffs) + 1, dtype=float)

    for index in range(0, k - degree + 1):
        new_coeffs[index] = coeffs[index]
    for index in range(k - multiplicity, n + 1):
        new_coeffs[index + 1] = coeffs[index]
    for index in range(k - degree + 1, k - multiplicity + 1):
        denominator = float(knots[index + degree] - knots[index])
        if abs(denominator) <= float(tol):
            raise BSplineAdaptiveError(
                "degenerate denominator during Boehm insertion at "
                f"u={inserted:.16g}, basis_index={index}"
            )
        alpha = (inserted - float(knots[index])) / denominator
        new_coeffs[index] = (
            alpha * coeffs[index] + (1.0 - alpha) * coeffs[index - 1]
        )

    new_knots = np.insert(knots, int(np.searchsorted(knots, inserted, side="right")), inserted)
    return new_knots, new_coeffs


def _boehm_insert_to_target_knots(
    old_knots,
    old_coeffs,
    degree,
    target_knots,
    tol=1.0e-12,
):
    current_knots = np.asarray(old_knots, dtype=float)
    current_coeffs = np.asarray(old_coeffs, dtype=float)
    target_knots = np.asarray(target_knots, dtype=float)
    if np.any(np.diff(current_knots) < -float(tol)):
        raise BSplineAdaptiveError("old knot vector must be nondecreasing")
    if np.any(np.diff(target_knots) < -float(tol)):
        raise BSplineAdaptiveError("target knot vector must be nondecreasing")
    if len(target_knots) < len(current_knots):
        raise BSplineAdaptiveError("target knot vector cannot remove knots")

    inserted_knots = []
    old_index = 0
    for target in target_knots:
        if old_index < len(current_knots) and math.isclose(
            float(current_knots[old_index]),
            float(target),
            rel_tol=0.0,
            abs_tol=float(tol),
        ):
            old_index += 1
            continue
        if (
            old_index < len(current_knots)
            and float(current_knots[old_index]) < float(target) - float(tol)
        ):
            raise BSplineAdaptiveError(
                "target knot vector is not an insertion-only refinement of the old vector"
            )
        inserted_knots.append(float(target))
    if old_index != len(current_knots):
        raise BSplineAdaptiveError(
            "target knot vector is not an insertion-only refinement of the old vector"
        )

    for inserted in inserted_knots:
        current_knots, current_coeffs = _boehm_insert_once(
            current_knots,
            current_coeffs,
            degree,
            inserted,
            tol=tol,
        )

    if len(current_knots) != len(target_knots) or not np.allclose(
        current_knots,
        target_knots,
        rtol=0.0,
        atol=float(tol),
    ):
        raise BSplineAdaptiveError(
            "Boehm insertion did not reproduce the requested target knot vector"
        )
    return current_coeffs


def _mode_normalization_factors(modes, spec):
    modes = list(modes)
    if not bool(spec.get("normalize_basis", True)):
        return np.ones(len(modes), dtype=float)
    class_shape = spec.get("class_shape", "sqrt_x_one_minus_x")
    factors = np.asarray(
        [
            mode_normalization_factor(mode, class_shape=class_shape)
            for mode in modes
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(factors)) or np.any(factors <= 0.0):
        raise BSplineAdaptiveError("B-spline normalization factors must be finite and positive")
    return factors


def _boehm_transfer_side(group, new_modes, spec, new_knots):
    old_modes = sorted(group.modes, key=lambda mode: int(mode["basis_index"]))
    new_modes = sorted(new_modes, key=lambda mode: int(mode["basis_index"]))
    old_factors = _mode_normalization_factors(old_modes, spec)
    new_factors = _mode_normalization_factors(new_modes, spec)
    canonical_old = group.coefficients / old_factors
    canonical_new = _boehm_insert_to_target_knots(
        group.knot_vector,
        canonical_old,
        group.degree,
        new_knots,
    )
    if len(canonical_new) != len(new_factors):
        raise BSplineAdaptiveError(
            f"Boehm transfer for side {group.side!r} produced {len(canonical_new)} "
            f"coefficients for {len(new_factors)} modes"
        )
    return canonical_new * new_factors


def _check_transferred_coefficients_within_bounds(next_modes, settings, tol=1.0e-12):
    policy = str(settings.get("transfer_bound_policy", "ERROR")).strip().upper()
    if policy != "ERROR":
        raise BSplineAdaptiveError(
            "Only BSPLINE_TRANSFER_BOUND_POLICY=ERROR is currently implemented."
        )
    global_lower = settings.get("opt_bound_lower")
    global_upper = settings.get("opt_bound_upper")
    use_global_bounds = global_lower is not None and global_upper is not None

    for mode in _active_modes(next_modes):
        coefficient = float(mode.get("coefficient", 0.0))
        if use_global_bounds:
            lower = float(global_lower)
            upper = float(global_upper)
        else:
            bounds = mode.get("bounds")
            if bounds is None:
                continue
            lower, upper = (float(bounds[0]), float(bounds[1]))

        if coefficient < lower - float(tol):
            violation = lower - coefficient
        elif coefficient > upper + float(tol):
            violation = coefficient - upper
        else:
            continue
        raise BSplineAdaptiveError(
            "Boehm-transferred coefficient violates optimization bounds: "
            f"mode_id={mode.get('id')} side={mode.get('side')} "
            f"basis_index={mode.get('basis_index')} coefficient={coefficient:.16e} "
            f"lower_bound={lower:.16e} upper_bound={upper:.16e} "
            f"violation={violation:.16e}"
        )
    print("[PROGRESSIVE_BSPLINE] KNOT_TRANSFER coefficient bounds OK")


def transfer_shape_to_inserted_space(
    space,
    metadata,
    new_knots,
    regularization=1.0e-12,
    settings=None,
):
    del regularization
    settings = dict(settings or {})
    method = str(settings.get("transfer_method", "BOEHM")).strip().upper()
    if method != "BOEHM":
        raise BSplineAdaptiveError(
            "Only BSPLINE_TRANSFER_METHOD=BOEHM is currently implemented."
        )

    old_modes, old_matrix = _basis_for_spec_modes(space.spec, metadata)
    old_coeffs = np.asarray(
        [float(mode.get("coefficient", 0.0)) for mode in old_modes],
        dtype=float,
    )
    old_deformation = old_matrix.dot(old_coeffs)
    virtual_spec = regenerate_clamped_modes(space, new_knots)
    new_modes_by_side = {}
    for mode in _active_modes(virtual_spec):
        side = str(mode.get("side", "")).strip().lower()
        new_modes_by_side.setdefault(side, []).append(mode)

    coefficients_by_side = {}
    for side in space.sides:
        coefficients_by_side[side] = _boehm_transfer_side(
            space.groups[side],
            new_modes_by_side.get(side, []),
            space.spec,
            new_knots,
        )

    transferred_spec = regenerate_clamped_modes(
        space,
        new_knots,
        coefficients_by_side,
    )
    new_modes, new_matrix = _basis_for_spec_modes(transferred_spec, metadata)
    new_coeffs = np.asarray(
        [float(mode.get("coefficient", 0.0)) for mode in new_modes],
        dtype=float,
    )
    transferred = new_matrix.dot(new_coeffs)
    error = transferred - old_deformation
    rms = float(math.sqrt(float(np.mean(error * error)))) if len(error) else 0.0
    max_error = float(np.max(np.abs(error))) if len(error) else 0.0
    scale = float(np.max(np.abs(old_deformation))) if len(old_deformation) else 0.0
    relative = max_error / max(scale, 1.0e-30)
    abs_tol = _as_float(
        settings.get("transfer_geometry_abs_tol", 1.0e-10),
        "BSPLINE_TRANSFER_GEOMETRY_ABS_TOL",
    )
    rel_tol = _as_float(
        settings.get("transfer_geometry_rel_tol", 1.0e-8),
        "BSPLINE_TRANSFER_GEOMETRY_REL_TOL",
    )
    if abs_tol < 0.0 or rel_tol < 0.0:
        raise BSplineAdaptiveError(
            "B-spline transfer geometry tolerances must be non-negative"
        )
    tolerance = abs_tol + rel_tol * scale

    coefficient_arrays = list(coefficients_by_side.values())
    all_coefficients = (
        np.concatenate(coefficient_arrays)
        if coefficient_arrays
        else np.zeros(0, dtype=float)
    )
    coeff_min = float(np.min(all_coefficients)) if len(all_coefficients) else 0.0
    coeff_max = float(np.max(all_coefficients)) if len(all_coefficients) else 0.0
    print("[PROGRESSIVE_BSPLINE] KNOT_TRANSFER method=BOEHM")
    print(
        "[PROGRESSIVE_BSPLINE] KNOT_TRANSFER geometry "
        f"rms={rms:.6e} max={max_error:.6e} rel={relative:.6e}"
    )
    print(
        "[PROGRESSIVE_BSPLINE] KNOT_TRANSFER coefficients "
        f"min={coeff_min:.6e} max={coeff_max:.6e}"
    )
    if max_error > tolerance:
        raise BSplineAdaptiveError(
            "Boehm knot transfer failed to preserve geometry: "
            f"rms={rms:.16e} max_error={max_error:.16e} "
            f"relative_error={relative:.16e} tolerance={tolerance:.16e}"
        )

    diagnostics = {
        "transfer_method": "BOEHM",
        "transfer_rms_error": rms,
        "transfer_max_error": max_error,
        "transfer_relative_error": relative,
        "transfer_tolerance": tolerance,
        "transfer_geometry_abs_tol": abs_tol,
        "transfer_geometry_rel_tol": rel_tol,
        "transfer_coefficient_min": coeff_min,
        "transfer_coefficient_max": coeff_max,
    }
    return coefficients_by_side, diagnostics


def build_next_knot_inserted_modes(optimized_modes, metadata, signal, settings):
    space = extract_clamped_knot_space(optimized_modes, settings)
    available_spans = len(
        knot_insertion_spans(
            space.knot_vector,
            min_width=settings.get("knot_min_span_width", 1.0e-8),
        )
    )
    n_insertions, batch_info = _requested_knot_insertions(
        space,
        settings,
        available_spans,
    )
    if n_insertions <= 0:
        return None, [], {
            "status": "no_valid_knot_span",
            "selected": False,
            "batch": batch_info,
        }

    print(
        "[PROGRESSIVE_BSPLINE] KNOT_INSERTION batch | "
        "mode={} current_reduced_ndv={} target_reduced_ndv={} insertions={}".format(
            batch_info["mode"],
            int(batch_info["current_reduced_ndv"]),
            int(batch_info["target_reduced_ndv"]),
            int(n_insertions),
        )
    )
    if batch_info.get("clamped"):
        print(
            "[PROGRESSIVE_BSPLINE] KNOT_INSERTION batch clamped | "
            "requested={} selected={} reason={}".format(
                int(batch_info.get("requested_insertions", n_insertions)),
                int(n_insertions),
                batch_info.get("clamp_reason", ""),
            )
        )

    score_rows = []
    selected_insertions = []
    current_spec = optimized_modes
    current_space = space
    current_knots = tuple(space.knot_vector)

    for step in range(1, int(n_insertions) + 1):
        try:
            step_rows = score_knot_spans(current_space, metadata, signal, settings)
        except BSplineAdaptiveError:
            if selected_insertions:
                break
            raise
        for row in step_rows:
            row["batch_step"] = step
            row["selected"] = False
        if not step_rows or float(step_rows[0].get("score", 0.0)) <= 0.0:
            for row in step_rows:
                row["status"] = "nonpositive_score"
            score_rows.extend(step_rows)
            break

        selected = step_rows[0]
        selected["selected"] = True
        selected["status"] = "selected"
        old_knots = tuple(current_knots)
        new_knots = insert_knot_midpoint(
            current_space.knot_vector,
            (selected["span_left"], selected["span_right"], selected["inserted_knot"]),
        )
        selected_insertions.append(
            {
                "step": step,
                "span_left": float(selected["span_left"]),
                "span_right": float(selected["span_right"]),
                "inserted_knot": float(selected["inserted_knot"]),
                "side": str(selected.get("side", "BOTH")).upper(),
                "score": float(selected["score"]),
                "score_raw": float(selected["score_raw"]),
                "residual_energy": float(selected["residual_energy"]),
                "old_knot_vector": [float(value) for value in old_knots],
                "new_knot_vector": [float(value) for value in new_knots],
                "incremental_rank": int(selected["incremental_rank"]),
                "incremental_columns": int(selected["incremental_columns"]),
                "condition_number": float(selected["condition_number"]),
            }
        )
        score_rows.extend(step_rows)
        print(
            "[PROGRESSIVE_BSPLINE] KNOT_INSERTION selected | "
            "step={} side={} span=[{:.6f},{:.6f}] knot={:.6f} score={:.6e}".format(
                step,
                str(selected.get("side", "BOTH")).upper(),
                float(selected["span_left"]),
                float(selected["span_right"]),
                float(selected["inserted_knot"]),
                float(selected["score"]),
            )
        )
        current_spec = regenerate_clamped_modes(current_space, new_knots)
        current_space = extract_clamped_knot_space(current_spec, settings)
        current_knots = tuple(new_knots)

    if not selected_insertions:
        return None, score_rows, {
            "status": "no_positive_knot_span",
            "selected": False,
            "batch": batch_info,
        }

    new_knots = current_knots
    coefficients_by_side, diagnostics = transfer_shape_to_inserted_space(
        space,
        metadata,
        new_knots,
        settings=settings,
    )
    next_modes = regenerate_clamped_modes(space, new_knots, coefficients_by_side)
    _check_transferred_coefficients_within_bounds(next_modes, settings)
    if settings.get("nfinal") is not None and refinement_limit_ndv(next_modes, settings) > int(settings["nfinal"]):
        return None, score_rows, {
            "status": "nfinal_limit",
            "selected": False,
            "ndv_before": len(active_mode_ids(optimized_modes)),
            "ndv_after": len(active_mode_ids(next_modes)),
            "batch": batch_info,
            "selected_insertions": selected_insertions,
        }
    first = selected_insertions[0]
    selected_data = {
        "status": "ok",
        "selected": True,
        "refine_mode": "KNOT_INSERTION",
        "knot_score_mode": str(settings.get("knot_score_mode", "VIRTUAL_INSERTION")).upper(),
        "span_left": float(first["span_left"]),
        "span_right": float(first["span_right"]),
        "inserted_knot": float(first["inserted_knot"]),
        "side": str(first.get("side", "BOTH")).upper(),
        "score": float(first["score"]),
        "score_raw": float(first["score_raw"]),
        "residual_energy": float(first["residual_energy"]),
        "ndv_before": len(active_mode_ids(optimized_modes)),
        "ndv_after": len(active_mode_ids(next_modes)),
        "reduced_ndv_before": reduced_ndv_for_knot_space(space),
        "reduced_ndv_after": refinement_limit_ndv(next_modes, settings),
        "old_knot_vector": [float(value) for value in space.knot_vector],
        "new_knot_vector": [float(value) for value in new_knots],
        "selected_insertions": selected_insertions,
        "batch": batch_info | {"insertions": len(selected_insertions)},
    } | diagnostics
    return next_modes, score_rows, selected_data


def build_scalar_deformation_sensitivity(metadata, sensitivities):
    aligned = match_sensitivities_to_metadata(metadata, sensitivities)
    values = []
    for index, (meta, sens) in enumerate(zip(metadata, aligned), start=1):
        if sens.get("sensitivity_x") is None or sens.get("sensitivity_y") is None:
            raise BSplineAdaptiveError(
                f"sensitivity row {index} is missing vector sensitivity columns"
            )
        dir_x = float(meta.get("deform_dir_x", meta["normal_x"]))
        dir_y = float(meta.get("deform_dir_y", meta["normal_y"]))
        values.append(
            float(sens["sensitivity_x"]) * dir_x
            + float(sens["sensitivity_y"]) * dir_y
        )
    return np.asarray(values, dtype=float)


def build_scalar_normal_sensitivity(metadata, sensitivities):
    return build_scalar_deformation_sensitivity(metadata, sensitivities)


def load_adjoint_signal(surface_adjoint_filename, metadata_filename):
    metadata = read_metadata(str(metadata_filename))
    sensitivities = read_sensitivity_file(str(surface_adjoint_filename))
    return metadata, build_scalar_deformation_sensitivity(metadata, sensitivities)


def _coeffs_are_well_conditioned(solve_matrix, coeffs, candidate, regularization):
    """Heuristic check: reject lstsq solutions with anomalously large coefficients.

    The active basis may become near-collinear as more modes are added during
    refinement, which would make lstsq produce coefficients that fit the noise
    in ``candidate`` rather than the true projection. We compare the coefficient
    norm against the magnitude of the problem itself.
    """
    del solve_matrix  # available if we later want to look at singular values
    if not np.all(np.isfinite(coeffs)):
        return False
    coeff_norm = float(np.linalg.norm(coeffs))
    candidate_norm = float(np.linalg.norm(candidate))
    if candidate_norm <= 0.0:
        return coeff_norm < 1.0e6
    threshold = 1.0e6 * candidate_norm
    return coeff_norm <= threshold or coeff_norm < 1.0e6 * max(
        float(regularization), 1.0e-30
    )


def _tikhonov_projection(solve_matrix, solve_rhs, regularization):
    lhs = solve_matrix.T.dot(solve_matrix)
    lhs += float(regularization) * np.eye(lhs.shape[0])
    rhs = solve_matrix.T.dot(solve_rhs)
    coeffs = np.linalg.solve(lhs, rhs)
    return coeffs


def residualize_candidate(active_matrix, candidate_values, weights=None, regularization=1.0e-12):
    """Project ``candidate`` onto the orthogonal complement of the active basis.

    The projection is performed in the weighted inner product
    ``<u, v>_W = u^T W v`` (identity if ``weights`` is None). The residual
    ``z = candidate - A c*`` therefore satisfies ``A^T W z = 0`` and the
    caller can safely use ``q = z^T W s`` and ``h = z^T W z`` as a
    Believer/Expected-Improvement-like score in the same inner product.
    """
    candidate = np.asarray(candidate_values, dtype=float)
    if active_matrix.size == 0 or active_matrix.shape[1] == 0:
        return candidate.copy()

    active_matrix = np.asarray(active_matrix, dtype=float)
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.shape[0] != candidate.shape[0]:
            raise BSplineAdaptiveError("weights must have the same length as candidate values")
        sqrt_w = np.sqrt(np.maximum(weights, 0.0))
        solve_matrix = active_matrix * sqrt_w[:, None]
        solve_rhs = candidate * sqrt_w
    else:
        solve_matrix = active_matrix
        solve_rhs = candidate

    try:
        coeffs, *_ = np.linalg.lstsq(solve_matrix, solve_rhs, rcond=None)
        if not _coeffs_are_well_conditioned(solve_matrix, coeffs, candidate, regularization):
            coeffs = _tikhonov_projection(solve_matrix, solve_rhs, regularization)
        projection = active_matrix.dot(coeffs)
    except np.linalg.LinAlgError:
        coeffs = _tikhonov_projection(solve_matrix, solve_rhs, regularization)
        projection = active_matrix.dot(coeffs)

    return candidate - projection


def reduced_ndv_for_knot_space(space, knot_vector=None):
    knot_vector = space.knot_vector if knot_vector is None else knot_vector
    n_basis = clamped_basis_count(space.degree, knot_vector)
    if space.coupling in ("NORMAL_EQUAL", "NORMAL_OPPOSITE"):
        return int(n_basis)
    return int(n_basis) * len(space.sides)


def physical_ndv_for_knot_space(space, knot_vector=None):
    knot_vector = space.knot_vector if knot_vector is None else knot_vector
    return int(clamped_basis_count(space.degree, knot_vector)) * len(space.sides)


def refinement_limit_ndv(mode_spec, settings):
    return reduced_ndv_for_knot_space(extract_clamped_knot_space(mode_spec, settings))


def _requested_knot_insertions(space, settings, available_spans):
    available_spans = max(0, int(available_spans))
    if available_spans <= 0:
        return 0, {
            "mode": str(settings.get("nadd_mode", "GROWTH_RATIO")).upper(),
            "current_reduced_ndv": reduced_ndv_for_knot_space(space),
            "target_reduced_ndv": reduced_ndv_for_knot_space(space),
            "reduced_ndv_per_insertion": 1 if space.coupling in ("NORMAL_EQUAL", "NORMAL_OPPOSITE") else len(space.sides),
            "requested_insertions": 0,
            "clamped": True,
            "clamp_reason": "no_valid_spans",
        }

    current_reduced = reduced_ndv_for_knot_space(space)
    reduced_per_insertion = (
        1 if space.coupling in ("NORMAL_EQUAL", "NORMAL_OPPOSITE") else len(space.sides)
    )
    mode = str(settings.get("nadd_mode", "GROWTH_RATIO")).upper()
    explicit = settings.get("knot_insertions_per_refine", 1)
    auto = str(explicit).strip().upper() == "AUTO"
    target_reduced = current_reduced

    if not auto:
        requested = int(explicit)
        mode_for_log = "EXPLICIT"
        target_reduced = current_reduced + requested
    elif mode == "GROWTH_RATIO":
        growth_ratio = float(settings.get("growth_ratio", 2.0))
        if growth_ratio <= 1.0:
            target_reduced = current_reduced + 1
        else:
            target_reduced = int(math.ceil(growth_ratio * current_reduced))
        requested = max(
            1,
            int(math.ceil(max(1, target_reduced - current_reduced) / float(reduced_per_insertion))),
        )
        mode_for_log = "GROWTH_RATIO"
    elif mode == "FIXED":
        requested = max(1, int(settings.get("fixed_nadd", 1)))
        target_reduced = current_reduced + requested
        mode_for_log = "FIXED"
    else:
        raise BSplineAdaptiveError(f"unsupported nadd mode {mode!r}")

    limits = [available_spans]
    nfinal = settings.get("nfinal")
    if nfinal is not None:
        remaining_reduced = max(0, int(nfinal) - int(current_reduced))
        limits.append(remaining_reduced // max(1, reduced_per_insertion))
    if settings.get("batch_size_max") is not None:
        limits.append(max(1, int(settings.get("batch_size_max", 1))))

    limited = min([int(requested)] + [int(limit) for limit in limits])
    limited = max(0, limited)
    clamp_reasons = []
    if limited < int(requested):
        if limited == available_spans:
            clamp_reasons.append("valid_spans")
        if settings.get("batch_size_max") is not None and limited == int(settings.get("batch_size_max", 1)):
            clamp_reasons.append("batch_size_max")
        if nfinal is not None:
            clamp_reasons.append("nfinal")
    return limited, {
        "mode": mode_for_log,
        "current_reduced_ndv": current_reduced,
        "target_reduced_ndv": target_reduced,
        "reduced_ndv_per_insertion": int(reduced_per_insertion),
        "requested_insertions": int(requested),
        "available_spans": available_spans,
        "insertions": int(limited),
        "clamped": limited < int(requested),
        "clamp_reason": ",".join(sorted(set(clamp_reasons))),
    }


def _csv_value(value):
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return "{:.15g}".format(value)
    return value


def write_knot_span_scores_csv(rows, filename):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=KNOT_SCORE_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: _csv_value(row.get(field, "")) for field in KNOT_SCORE_FIELDNAMES}
            )


def write_selected_knot_refinement_json(level_id, selected_data, score_rows, filename):
    data = {
        "level_id": int(level_id),
        "selected": bool(selected_data.get("selected", False)),
        "refinement": selected_data,
        "selected_span_score": next(
            (row for row in score_rows if row.get("selected")),
            None,
        ),
    }
    with open(filename, "w") as fp:
        json.dump(data, fp, indent=2, sort_keys=True)
        fp.write("\n")
    return data


def find_best_eval_dir(opt_run_dir, objective_column="objective"):
    opt_run_dir = Path(opt_run_dir)
    history_file = opt_run_dir / "optimization_history.csv"
    valid_rows = []
    with open(history_file, "r", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            try:
                objective = float(row.get(objective_column, ""))
                eval_id = int(float(row.get("eval_id", "")))
            except Exception:
                continue
            status = str(row.get("status", "ok")).strip().lower()
            # NaN/inf objectives must never win a minimization; replace
            # them with +inf so they sort to the bottom and are skipped.
            if not math.isfinite(objective):
                objective = math.inf
            valid_rows.append((objective, eval_id, status))

    if not valid_rows:
        raise BSplineAdaptiveError(f"{history_file} has no valid evaluation rows")

    finite_rows = [row for row in valid_rows if math.isfinite(row[0])]
    if not finite_rows:
        raise BSplineAdaptiveError(
            f"{history_file} has no evaluation with a finite objective"
        )

    ok_rows = [row for row in valid_rows if row[2] == "ok" and math.isfinite(row[0])]
    if not ok_rows:
        # Do NOT fall back to status != "ok" rows: a failed evaluation may
        # have written a partial/garbage objective, and its adjoint (used
        # downstream for residual projection) is unreliable. Refuse instead.
        raise BSplineAdaptiveError(
            f"{history_file} has no successful (status='ok') evaluation with a finite objective"
        )
    best = min(ok_rows, key=lambda item: item[0])
    return opt_run_dir / f"eval_{best[1]:04d}"


def find_eval_dir_for_mode_coefficients(
    opt_run_dir,
    optimized_modes,
    tolerance=1.0e-10,
):
    opt_run_dir = Path(opt_run_dir)
    history_file = opt_run_dir / "optimization_history.csv"
    mode_ids = active_mode_ids(optimized_modes)
    coefficients = active_coefficient_vector(optimized_modes)
    matched_rows = []

    with open(history_file, "r", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            try:
                eval_id = int(float(row.get("eval_id", "")))
            except Exception:
                continue

            status = str(row.get("status", "ok")).strip().lower()
            if status and status != "ok":
                continue

            matched = True
            for mode_id, coefficient in zip(mode_ids, coefficients):
                field = f"coeff__{mode_id}"
                if field not in row:
                    matched = False
                    break
                try:
                    history_value = float(row[field])
                except Exception:
                    matched = False
                    break
                if abs(history_value - coefficient) > float(tolerance):
                    matched = False
                    break

            if matched:
                try:
                    objective = float(row.get("objective", "inf"))
                except Exception:
                    objective = math.inf
                # NaN/inf objectives must never win a minimization: skip
                # them entirely here too, so the matched-eval selection
                # cannot pick a broken run.
                if math.isfinite(objective):
                    matched_rows.append((objective, eval_id))

    if matched_rows:
        _objective, matched_eval_id = min(matched_rows, key=lambda item: item[0])
        return opt_run_dir / f"eval_{matched_eval_id:04d}"

    print(
        "[PROGRESSIVE_BSPLINE] WARNING: no eval row with finite objective matches optimized_modes.json coefficients; falling back to best objective eval"
    )
    return find_best_eval_dir(opt_run_dir)


def _read_optimization_history(opt_run_dir):
    history_file = Path(opt_run_dir) / "optimization_history.csv"
    rows = []
    with open(history_file, "r", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            try:
                row["_objective"] = float(row.get("objective", ""))
                row["_eval_id"] = int(float(row.get("eval_id", "")))
            except Exception:
                continue
            rows.append(row)
    return rows


def _cumulative_best(values):
    best_values = []
    best = None
    for value in values:
        value = float(value)
        best = value if best is None else min(best, value)
        best_values.append(float(best))
    return best_values


def _compact_number(value):
    if value == "":
        return ""
    try:
        return "{:.6e}".format(float(value))
    except Exception:
        return str(value)


def _log_trigger_decision(decision):
    print(
        "[PROGRESSIVE_BSPLINE] Trigger {} | window={} metric={} threshold={} "
        "patience={}/{} refine_now={} reason={}".format(
            decision.trigger_mode,
            decision.window,
            _compact_number(decision.metric),
            _compact_number(decision.threshold),
            int(decision.counter),
            decision.patience,
            bool(decision.refine_now),
            decision.reason,
        )
    )


def _tail_count(values, predicate):
    count = 0
    for value in reversed(values):
        if predicate(value):
            count += 1
        else:
            break
    return count


def _trigger_refine_from_history(objectives, opts):
    trigger = str(opts.get("trigger", "MAX_ITER")).upper()
    if trigger == "SLOPE_EFFICIENCY_FILTERED":
        print(
            "[PROGRESSIVE_BSPLINE] SLOPE_EFFICIENCY_FILTERED is deprecated and is mapped internally to SLOPE_EFFICIENCY_TRIGGER."
        )
        trigger = "SLOPE_EFFICIENCY_TRIGGER"

    window = max(1, int(opts.get("window", 1)))
    threshold = float(opts.get("tol", 0.2))
    patience = max(1, int(opts.get("slope_patience", 1)))
    decision = TriggerDecision(
        trigger_mode=trigger,
        threshold=threshold,
        window=window,
        patience=patience,
        refine_now=False,
    )
    if trigger == "MAX_ITER":
        decision.metric = ""
        decision.threshold = ""
        decision.window = ""
        decision.patience = ""
        decision.counter = 1
        decision.refine_now = True
        decision.reason = "level_complete"
        _log_trigger_decision(decision)
        return decision
    if len(objectives) < 2:
        decision.reason = "insufficient_history"
        _log_trigger_decision(decision)
        return decision

    warmup = int(opts.get("warmup_iter", 0))
    if len(objectives) <= warmup:
        decision.reason = f"warmup {len(objectives)}/{warmup}"
        _log_trigger_decision(decision)
        return decision

    best_values = _cumulative_best(objectives)

    if trigger == "SLOPE_EFFICIENCY_TRIGGER":
        if len(objectives) < window + 1:
            decision.reason = "insufficient_window"
            _log_trigger_decision(decision)
            return decision
        improvements = [
            max(0.0, float(prev) - float(current))
            for prev, current in zip(best_values[:-1], best_values[1:])
        ]
        windowed = [
            sum(improvements[i - window + 1 : i + 1]) / float(window)
            for i in range(window - 1, len(improvements))
        ]
        best_efficiency = max(windowed) if windowed else 0.0
        denominator = max(best_efficiency, float(opts.get("eps", 1.0e-300)))
        ratios = [float(value) / denominator for value in windowed]
        ratio = ratios[-1] if ratios else 0.0
        decision.metric = ratio
        decision.counter = _tail_count(ratios, lambda value: float(value) < threshold)
        decision.refine_now = decision.counter >= patience
        decision.reason = "efficiency_below_threshold" if decision.refine_now else "efficiency_ok"
        _log_trigger_decision(decision)
        return decision

    if trigger == "SLOPE_EFFICIENCY_BEST_LOG":
        eps = float(opts.get("eps", 1.0e-300))
        if len(objectives) < window + 1:
            decision.reason = "insufficient_window"
            _log_trigger_decision(decision)
            return decision
        improvements = []
        for prev, current in zip(best_values[:-1], best_values[1:]):
            prev_log = math.log(max(float(prev) + eps, eps))
            current_log = math.log(max(float(current) + eps, eps))
            improvements.append(max(0.0, prev_log - current_log))
        if not improvements:
            decision.reason = "no_improvements"
            _log_trigger_decision(decision)
            return decision
        windowed = [
            sum(improvements[i - window + 1 : i + 1]) / float(window)
            for i in range(window - 1, len(improvements))
        ]
        max_slope = max(max(windowed) if windowed else 0.0, eps)
        ratios = [float(value) / max_slope for value in windowed]
        ratio = ratios[-1] if ratios else 0.0
        decision.metric = ratio
        decision.counter = _tail_count(ratios, lambda value: float(value) < threshold)
        decision.refine_now = decision.counter >= patience
        decision.reason = "log_efficiency_below_threshold" if decision.refine_now else "log_efficiency_ok"
        _log_trigger_decision(decision)
        return decision

    if trigger == "STAGNATION_TRIGGER":
        stag_tol = float(opts.get("stag_tol", 1.0e-3))
        stag_window = max(1, int(opts.get("stag_window", window)))
        stag_patience = max(1, int(opts.get("stag_patience", patience)))
        decision.threshold = stag_tol
        decision.window = stag_window
        decision.patience = stag_patience
        if len(best_values) < stag_window + 1:
            decision.reason = "insufficient_window"
            _log_trigger_decision(decision)
            return decision
        improvements = []
        for end_index in range(stag_window, len(best_values)):
            start = float(best_values[end_index - stag_window])
            end = float(best_values[end_index])
            improvements.append(
                max(0.0, start - end)
                / max(abs(start), float(opts.get("eps", 1.0e-300)))
            )
        improvement = improvements[-1] if improvements else 0.0
        decision.metric = improvement
        decision.counter = _tail_count(improvements, lambda value: float(value) < stag_tol)
        decision.refine_now = decision.counter >= stag_patience
        decision.reason = "stagnated" if decision.refine_now else "improving"
        _log_trigger_decision(decision)
        return decision

    raise BSplineAdaptiveError(f"unsupported trigger {trigger!r}")


def _level_summary_row(level, rows, selected_modes, trigger_decision, refine_now, status):
    objectives = [row["_objective"] for row in rows]
    best_row = min(rows, key=lambda row: row["_objective"]) if rows else None
    if isinstance(trigger_decision, TriggerDecision):
        trigger_mode = trigger_decision.trigger_mode
        trigger_metric = trigger_decision.metric
        trigger_threshold = trigger_decision.threshold
        trigger_window = trigger_decision.window
        trigger_patience = trigger_decision.patience
        trigger_counter = trigger_decision.counter
        trigger_reason = trigger_decision.reason
    else:
        trigger_mode = str(trigger_decision)
        trigger_metric = ""
        trigger_threshold = ""
        trigger_window = ""
        trigger_patience = ""
        trigger_counter = ""
        trigger_reason = ""
    return {
        "level_id": level.level_id,
        "ndv": level.ndv,
        "workdir": str(level.workdir),
        "opt_workdir": str(level.opt_workdir),
        "objective_start": objectives[0] if objectives else "",
        "objective_final": objectives[-1] if objectives else "",
        "best_objective": best_row["_objective"] if best_row else "",
        "best_eval_id": best_row["_eval_id"] if best_row else "",
        "n_function_evals": len(rows),
        "n_gradient_evals": len(rows),
        "n_added": len(selected_modes),
        "selected_ids": ";".join(str(mode["id"]) for mode in selected_modes),
        "trigger_mode": trigger_mode,
        "trigger_metric": trigger_metric,
        "trigger_threshold": trigger_threshold,
        "trigger_window": trigger_window,
        "trigger_patience": trigger_patience,
        "trigger_counter": trigger_counter,
        "refine_now": refine_now,
        "trigger_reason": trigger_reason,
        "status": status,
    }


def _write_adaptive_history(rows, filename):
    fieldnames = [
        "level_id",
        "ndv",
        "workdir",
        "opt_workdir",
        "objective_start",
        "objective_final",
        "best_objective",
        "best_eval_id",
        "n_function_evals",
        "n_gradient_evals",
        "n_added",
        "selected_ids",
        "trigger_mode",
        "trigger_metric",
        "trigger_threshold",
        "trigger_window",
        "trigger_patience",
        "trigger_counter",
        "refine_now",
        "trigger_reason",
        "status",
    ]
    with open(filename, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fieldnames})


def _write_selection_history(rows, filename):
    fieldnames = ["level_id", "selection_order", "mode_id", "side", "score", "raw_grad"]
    with open(filename, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fieldnames})


def _mode_display(mode):
    left, right, _center = _mode_support(mode)
    return (
        f"{mode['id']} | side={mode.get('side')} | "
        f"support=[{left:.6f},{right:.6f}] | "
        f"coeff={float(mode.get('coefficient', 0.0)):.6e}"
    )


def _print_level_start(
    level,
    refine_state,
    kept=0,
    added=0,
    log_active_modes=False,
):
    print(f"[PROGRESSIVE_BSPLINE] Level {level.level_id} | NDV = {level.ndv}")
    print(f"[PROGRESSIVE_BSPLINE] Refinement state: {refine_state}")
    if level.level_id > 0:
        print(f"[PROGRESSIVE_BSPLINE] Kept optimized coefficients from previous level: {kept}")
        print(f"[PROGRESSIVE_BSPLINE] Added/refined active modes: {added}")
    if log_active_modes:
        print("[PROGRESSIVE_BSPLINE] Active modes:")
        for mode in _active_modes(level.active_modes):
            print("[PROGRESSIVE_BSPLINE]   " + _mode_display(mode))
    else:
        sides = sorted(
            {
                str(mode.get("side", "")).strip().lower()
                for mode in _active_modes(level.active_modes)
            }
        )
        print(
            "[PROGRESSIVE_BSPLINE] Active mode summary: n={} sides={}".format(
                level.ndv,
                ",".join(side for side in sides if side) or "unknown",
            )
        )


def validate_adaptive_options(opts):
    opts = dict(opts)
    opts["refinement"] = str(opts.get("refinement", "ADAPTIVE")).upper()
    if opts["refinement"] != "ADAPTIVE":
        raise BSplineAdaptiveError(
            "B-spline progressive optimization supports only --refinement ADAPTIVE"
        )

    opts["refine_state"] = str(opts.get("refine_state", REFINE_STATE)).upper()
    if opts["refine_state"] != REFINE_STATE:
        raise BSplineAdaptiveError(
            "BSPLINE_REFINE_STATE is fixed internally to INITIAL_MESH_KEEP_DV."
        )

    if opts.get("score_mode") is not None:
        raise BSplineAdaptiveError(
            "This cfg contains removed candidate/generated B-spline options: "
            "BSPLINE_SCORE_MODE. Remove them. The adaptive optimizer now "
            "supports only internal KNOT_INSERTION."
        )
    opts["refine_mode"] = str(opts.get("refine_mode", REFINE_MODE)).upper()
    if opts["refine_mode"] != REFINE_MODE:
        raise BSplineAdaptiveError(
            "BSPLINE_REFINE_MODE is no longer user-configurable. "
            "Candidate/generated refinement has been removed; KNOT_INSERTION "
            "is fixed internally."
        )
    opts["knot_score_mode"] = str(opts.get("knot_score_mode", "VIRTUAL_INSERTION")).upper()
    if opts["knot_score_mode"] not in ALLOWED_KNOT_SCORE_MODES:
        raise BSplineAdaptiveError(
            f"unsupported knot score mode {opts['knot_score_mode']!r}; allowed values are {ALLOWED_KNOT_SCORE_MODES}"
        )
    knot_insertions = opts.get("knot_insertions_per_refine", 1)
    if str(knot_insertions).strip().upper() == "AUTO":
        opts["knot_insertions_per_refine"] = "AUTO"
    else:
        opts["knot_insertions_per_refine"] = int(knot_insertions)
        if opts["knot_insertions_per_refine"] < 1:
            raise BSplineAdaptiveError("--knot-insertions-per-refine must be AUTO or >= 1")
    opts["knot_min_span_width"] = _as_float(
        opts.get("knot_min_span_width", 1.0e-8),
        "BSPLINE_KNOT_MIN_SPAN_WIDTH",
    )
    if opts["knot_min_span_width"] <= 0.0:
        raise BSplineAdaptiveError("--knot-min-span-width must be positive")
    opts["transfer_method"] = str(
        opts.get("transfer_method", "BOEHM")
    ).strip().upper()
    if opts["transfer_method"] != "BOEHM":
        raise BSplineAdaptiveError(
            "Only BSPLINE_TRANSFER_METHOD=BOEHM is currently implemented."
        )
    opts["transfer_bound_policy"] = str(
        opts.get("transfer_bound_policy", "ERROR")
    ).strip().upper()
    if opts["transfer_bound_policy"] != "ERROR":
        raise BSplineAdaptiveError(
            "Only BSPLINE_TRANSFER_BOUND_POLICY=ERROR is currently implemented."
        )
    opts["transfer_geometry_abs_tol"] = _as_float(
        opts.get("transfer_geometry_abs_tol", 1.0e-10),
        "BSPLINE_TRANSFER_GEOMETRY_ABS_TOL",
    )
    opts["transfer_geometry_rel_tol"] = _as_float(
        opts.get("transfer_geometry_rel_tol", 1.0e-8),
        "BSPLINE_TRANSFER_GEOMETRY_REL_TOL",
    )
    if opts["transfer_geometry_abs_tol"] < 0.0:
        raise BSplineAdaptiveError(
            "BSPLINE_TRANSFER_GEOMETRY_ABS_TOL must be non-negative"
        )
    if opts["transfer_geometry_rel_tol"] < 0.0:
        raise BSplineAdaptiveError(
            "BSPLINE_TRANSFER_GEOMETRY_REL_TOL must be non-negative"
        )

    opts["trigger"] = str(opts.get("trigger", "MAX_ITER")).upper()
    if opts["trigger"] not in ALLOWED_TRIGGERS:
        raise BSplineAdaptiveError(
            f"unsupported trigger {opts['trigger']!r}; allowed values are {ALLOWED_TRIGGERS}"
        )

    opts["nadd_mode"] = str(opts.get("nadd_mode", "GROWTH_RATIO")).upper()
    if opts["nadd_mode"] == "SCORE_BATCH":
        raise BSplineAdaptiveError(
            "BSPLINE_NADD_MODE=SCORE_BATCH was part of the removed "
            "candidate/generated strategy. Use GROWTH_RATIO or FIXED."
        )
    if opts["nadd_mode"] not in ALLOWED_NADD_MODES:
        raise BSplineAdaptiveError(
            f"unsupported nadd mode {opts['nadd_mode']!r}; allowed values are {ALLOWED_NADD_MODES}"
        )

    removed_runtime = []
    for key in (
        "candidate_source",
        "generated_peaks_per_side",
        "generated_widths",
        "generated_min_separation",
        "generated_xmin",
        "generated_xmax",
        "edge_xle",
        "edge_xte",
        "rough_lambda",
        "rough_power",
        "batch_score_rel_tol",
    ):
        if opts.get(key) is not None:
            removed_runtime.append(key)
    if removed_runtime:
        raise BSplineAdaptiveError(
            "This cfg contains removed candidate/generated B-spline options: "
            + ", ".join(sorted(removed_runtime))
            + ". Remove them. The adaptive optimizer now supports only "
            "internal KNOT_INSERTION."
        )

    sensitivity_weighting = normalize_sensitivity_weighting(
        opts.get("sensitivity_weighting", "NODAL")
    )
    if sensitivity_weighting != "NODAL":
        raise BSplineAdaptiveError(
            "BSPLINE_SENSITIVITY_WEIGHTING=DENSITY is not supported by "
            "adaptive KNOT_INSERTION. NODAL is fixed internally."
        )
    opts["sensitivity_weighting"] = "NODAL"

    opts["eval_layout"] = str(opts.get("eval_layout", "DSN")).upper()
    if opts["eval_layout"] != "DSN":
        raise BSplineAdaptiveError(
            "BSPLINE_EVAL_LAYOUT=FLAT has been removed. DSN is the only supported layout."
        )
    opts["eval_layout"] = "DSN"
    opts["symmetry_coupling"] = str(opts.get("symmetry_coupling", "NONE")).upper()
    if opts["symmetry_coupling"] not in ALLOWED_SYMMETRY_COUPLINGS:
        raise BSplineAdaptiveError(
            f"unsupported symmetry coupling {opts['symmetry_coupling']!r}; allowed values are {ALLOWED_SYMMETRY_COUPLINGS}"
        )
    try:
        opts["surface_mode"] = normalize_surface_mode(
            opts.get("surface_mode", "BOTH")
        )
    except BSplineModeError as exc:
        raise BSplineAdaptiveError(str(exc))
    if opts["surface_mode"] != "BOTH" and opts["symmetry_coupling"] != "NONE":
        raise BSplineAdaptiveError(
            "BSPLINE_SYMMETRY_COUPLING is only valid with BSPLINE_SURFACE_MODE=BOTH"
        )
    opts["objective_adjoint"] = str(opts.get("objective_adjoint", "drag")).strip() or "drag"

    opts["auto_scale_bounds_to_geometry"] = bool(opts.get("auto_scale_bounds_to_geometry", False))
    opts["max_normal_displacement"] = (
        float(opts["max_normal_displacement"])
        if opts.get("max_normal_displacement") is not None
        else None
    )
    opts["max_rms_normal_displacement"] = (
        float(opts["max_rms_normal_displacement"])
        if opts.get("max_rms_normal_displacement") is not None
        else None
    )
    opts["min_bound_scale"] = float(opts.get("min_bound_scale", 0.0))
    if opts["min_bound_scale"] < 0.0:
        raise BSplineAdaptiveError("--min-bound-scale must be non-negative")
    if opts["auto_scale_bounds_to_geometry"] and (
        opts["max_normal_displacement"] is None
        and opts["max_rms_normal_displacement"] is None
    ):
        raise BSplineAdaptiveError(
            "at least one of --max-normal-displacement or --max-rms-normal-displacement must be provided when --auto-scale-bounds-to-geometry is enabled"
        )

    opts["nlevels"] = max(1, int(opts.get("nlevels", 1)))
    opts["nfinal"] = int(opts["nfinal"]) if opts.get("nfinal") is not None else None
    opts["max_iter_per_level"] = max(1, int(opts.get("max_iter_per_level", 5)))
    opts["window"] = max(1, int(opts.get("window", 1)))
    opts["tol"] = float(opts.get("tol", 0.2))
    opts["eps"] = float(opts.get("eps", 1.0e-300))
    if opts["eps"] <= 0.0:
        raise BSplineAdaptiveError("--trigger-eps must be positive")
    opts["slope_filter_tol"] = float(opts.get("slope_filter_tol", 0.02))
    opts["slope_patience"] = max(1, int(opts.get("slope_patience", 1)))
    opts["stag_window"] = max(1, int(opts.get("stag_window", opts["window"])))
    opts["stag_tol"] = float(opts.get("stag_tol", 1.0e-3))
    opts["stag_band"] = float(opts.get("stag_band", 0.02))
    opts["stag_patience"] = max(1, int(opts.get("stag_patience", 1)))
    opts["warmup_iter"] = max(0, int(opts.get("warmup_iter", 0)))
    opts["growth_ratio"] = float(opts.get("growth_ratio", 2.0))
    opts["fixed_nadd"] = max(1, int(opts.get("fixed_nadd", 1)))
    opts["batch_size_max"] = max(1, int(opts.get("batch_size_max", 1)))
    opts["opt_accuracy"] = (
        float(opts["opt_accuracy"])
        if opts.get("opt_accuracy") is not None
        else None
    )
    opts["opt_bound_upper"] = (
        float(opts["opt_bound_upper"])
        if opts.get("opt_bound_upper") is not None
        else None
    )
    opts["opt_bound_lower"] = (
        float(opts["opt_bound_lower"])
        if opts.get("opt_bound_lower") is not None
        else None
    )
    if (opts["opt_bound_lower"] is None) != (opts["opt_bound_upper"] is None):
        raise BSplineAdaptiveError(
            "--opt-bound-lower and --opt-bound-upper must be provided together"
        )
    opts["opt_relax_factor"] = float(
        1.0 if opts.get("opt_relax_factor") is None else opts.get("opt_relax_factor")
    )
    if opts["opt_relax_factor"] <= 0.0:
        raise BSplineAdaptiveError("--opt-relax-factor must be positive")
    opts["opt_gradient_factor"] = float(
        1.0
        if opts.get("opt_gradient_factor") is None
        else opts.get("opt_gradient_factor")
    )
    if opts["opt_gradient_factor"] <= 0.0:
        raise BSplineAdaptiveError("--opt-gradient-factor must be positive")
    opts["opt_line_search_bound"] = (
        float(opts["opt_line_search_bound"])
        if opts.get("opt_line_search_bound") is not None
        else None
    )
    if opts["opt_line_search_bound"] is not None and opts["opt_line_search_bound"] <= 0.0:
        raise BSplineAdaptiveError("--opt-line-search-bound must be positive")
    opts["local_step_limit"] = _as_bool(opts.get("local_step_limit", False), default=False)
    opts["log_active_modes"] = _as_bool(opts.get("log_active_modes", False), default=False)
    opts["local_step_limit_ratio"] = _as_float(
        opts.get("local_step_limit_ratio", 200.0),
        "BSPLINE_LOCAL_STEP_LIMIT_RATIO",
    )
    if opts["local_step_limit_ratio"] <= 0.0:
        raise BSplineAdaptiveError("--local-step-limit-ratio must be positive")
    opts["thickness_options"] = dict(opts.get("thickness_options") or {})
    if _as_bool(
        opts["thickness_options"].get("PROGRESSIVE_THICKNESS_CONSTRAINT", False),
        default=False,
    ):
        try:
            domain_mode = resolve_thickness_domain_mode(
                opts["surface_mode"],
                opts["thickness_options"].get(
                    "PROGRESSIVE_THICKNESS_DOMAIN_MODE",
                    "AUTO",
                ),
            )
        except BSplineSU2DriverError as exc:
            raise BSplineAdaptiveError(str(exc))
        opts["thickness_options"]["PROGRESSIVE_THICKNESS_DOMAIN_MODE"] = domain_mode

    try:
        direction_mode = normalize_deformation_direction_mode(
            opts.get("deformation_direction_mode"),
            le_safe_direction=opts.get("le_safe_direction", False),
        )
        le_safe_opts = validate_le_safe_direction_options(
            le_safe_direction=direction_mode == "LE_SAFE",
            le_safe_x0=(
                opts.get("le_safe_x0", LE_SAFE_DEFAULT_X0)
                if direction_mode == "LE_SAFE"
                else LE_SAFE_DEFAULT_X0
            ),
            le_safe_x1=(
                opts.get("le_safe_x1", LE_SAFE_DEFAULT_X1)
                if direction_mode == "LE_SAFE"
                else LE_SAFE_DEFAULT_X1
            ),
            le_safe_power=(
                opts.get("le_safe_power", LE_SAFE_DEFAULT_POWER)
                if direction_mode == "LE_SAFE"
                else LE_SAFE_DEFAULT_POWER
            ),
        )
    except BSplineModeError as exc:
        raise BSplineAdaptiveError(str(exc))
    opts["deformation_direction_mode"] = direction_mode
    opts["le_safe_direction"] = le_safe_opts["le_safe_direction"]
    opts["le_safe_x0"] = le_safe_opts["le_safe_x0"]
    opts["le_safe_x1"] = le_safe_opts["le_safe_x1"]
    opts["le_safe_power"] = le_safe_opts["le_safe_power"]
    return opts


def adaptive_options_from_config(config_values):
    removed_keys = [key for key in REMOVED_CANDIDATE_CONFIG_KEYS if key in config_values]
    if removed_keys:
        raise BSplineAdaptiveError(
            "This cfg contains removed candidate/generated B-spline options: "
            + ", ".join(sorted(removed_keys))
            + ". Remove them. The adaptive optimizer now supports only "
            "internal KNOT_INSERTION."
        )
    options = fixed_driver_options_from_config(config_values)
    if "maxiter" in options:
        options["max_iter_per_level"] = options.pop("maxiter")
    case_mapping = {
        "BSPLINE_BASE_MESH": "base_mesh",
        "BSPLINE_MARKER": "marker",
        "BSPLINE_WORKDIR": "workdir",
        "BSPLINE_MPI": "mpi",
        "BSPLINE_MODES": "modes",
        "BSPLINE_DEF_TEMPLATE": "def_template",
        "BSPLINE_PRIMAL_TEMPLATE": "primal_template",
        "BSPLINE_ADJOINT_TEMPLATE": "adjoint_template",
        "BSPLINE_GENERATE_INITIAL_MODES": "generate_initial_modes",
        "BSPLINE_INITIAL_NPER_SIDE": "initial_nper_side",
        "BSPLINE_INITIAL_DEGREE": "initial_degree",
        "BSPLINE_INITIAL_COEFFICIENT": "initial_coefficient",
        "BSPLINE_INITIAL_BOUND_LOWER": "initial_bound_lower",
        "BSPLINE_INITIAL_BOUND_UPPER": "initial_bound_upper",
        "BSPLINE_INITIAL_NORMALIZE_BASIS": "initial_normalize_basis",
        "BSPLINE_INITIAL_NORMALIZATION_MODE": "initial_normalization_mode",
    }
    for key, dest in case_mapping.items():
        if key in config_values:
            options[dest] = config_values[key]
    if "BSPLINE_INITIAL_CLASS_SHAPE" in config_values:
        options["initial_class_shape"] = config_values["BSPLINE_INITIAL_CLASS_SHAPE"]
    elif "BSPLINE_USE_CLASS_SHAPE" in config_values:
        options["initial_class_shape"] = (
            "sqrt_x_one_minus_x"
            if _as_bool(config_values["BSPLINE_USE_CLASS_SHAPE"], default=True)
            else "none"
        )

    if "base_mesh" not in options and config_values.get("MESH_FILENAME"):
        options["base_mesh"] = config_values["MESH_FILENAME"]
    marker = _infer_marker_from_config(config_values)
    if marker is not None:
        options["marker"] = marker

    objective = str(
        config_values.get(
            "OPT_OBJECTIVE",
            config_values.get("OBJECTIVE_FUNCTION", ""),
        )
    ).strip().strip('"').strip("'").upper()
    if objective:
        options["opt_objective"] = objective
        if "objective_column" not in options and objective == "DRAG":
            options["objective_column"] = "CD"
        if "objective_adjoint" not in options and objective == "DRAG":
            options["objective_adjoint"] = "drag"

    for dest in (
        "base_mesh",
        "modes",
        "def_template",
        "primal_template",
        "adjoint_template",
        "workdir",
    ):
        if dest in options and options[dest]:
            options[dest] = _resolve_cfg_path(config_values, options[dest])
    options["_case_config"] = config_values.get("_optimizer_config_filename")

    mapping = {
        "BSPLINE_NLEVELS": "nlevels",
        "BSPLINE_NFINAL": "nfinal",
        "BSPLINE_REFINE_MODE": "refine_mode",
        "BSPLINE_REFINE_STATE": "refine_state",
        "BSPLINE_SENSITIVITY_WEIGHTING": "sensitivity_weighting",
        "BSPLINE_KNOT_SCORE_MODE": "knot_score_mode",
        "BSPLINE_KNOT_INSERTIONS_PER_REFINE": "knot_insertions_per_refine",
        "BSPLINE_KNOT_MIN_SPAN_WIDTH": "knot_min_span_width",
        "BSPLINE_TRANSFER_METHOD": "transfer_method",
        "BSPLINE_TRANSFER_BOUND_POLICY": "transfer_bound_policy",
        "BSPLINE_TRANSFER_GEOMETRY_ABS_TOL": "transfer_geometry_abs_tol",
        "BSPLINE_TRANSFER_GEOMETRY_REL_TOL": "transfer_geometry_rel_tol",
        "BSPLINE_TRIGGER": "trigger",
        "BSPLINE_TRIGGER_WINDOW": "window",
        "BSPLINE_TRIGGER_RATIO": "tol",
        "BSPLINE_TRIGGER_EPS": "eps",
        "BSPLINE_TRIGGER_WARMUP_ITER": "warmup_iter",
        "BSPLINE_SLOPE_FILTER_TOL": "slope_filter_tol",
        "BSPLINE_SLOPE_PATIENCE": "slope_patience",
        "BSPLINE_STAGNATION_WINDOW": "stag_window",
        "BSPLINE_STAGNATION_REL_TOL": "stag_tol",
        "BSPLINE_STAGNATION_BAND": "stag_band",
        "BSPLINE_STAGNATION_PATIENCE": "stag_patience",
        "BSPLINE_NADD_MODE": "nadd_mode",
        "BSPLINE_FIXED_NADD": "fixed_nadd",
        "BSPLINE_GROWTH_RATIO": "growth_ratio",
        "BSPLINE_BATCH_SIZE_MAX": "batch_size_max",
        "BSPLINE_LOG_ACTIVE_MODES": "log_active_modes",
    }
    for key, dest in mapping.items():
        if key in config_values:
            options[dest] = config_values[key]
    return options


def _strip_cfg_atom(value):
    return str(value).strip().strip('"').strip("'")


def _cfg_value_list(value):
    if isinstance(value, (list, tuple)):
        tokens = list(value)
    else:
        text = _strip_cfg_atom(value)
        if (text.startswith("(") and text.endswith(")")) or (
            text.startswith("[") and text.endswith("]")
        ):
            text = text[1:-1]
        tokens = [token for token in text.replace(",", " ").split() if token]
    return [_strip_cfg_atom(token) for token in tokens if _strip_cfg_atom(token)]


def _infer_marker_from_config(config_values):
    for key in ("BSPLINE_MARKER", "DV_MARKER", "MARKER_MONITORING", "MARKER_PLOTTING"):
        if key not in config_values or config_values[key] in (None, ""):
            continue
        markers = _cfg_value_list(config_values[key])
        if len(markers) == 1:
            return markers[0]
        clear = [
            marker
            for marker in markers
            if str(marker).strip().upper() in ("AIRFOIL", "WALL", "WING", "BODY")
            or "AIRFOIL" in str(marker).strip().upper()
        ]
        if len(clear) == 1:
            return clear[0]
        raise BSplineAdaptiveError(
            "Could not uniquely infer BSPLINE_MARKER. Please set BSPLINE_MARKER= ..."
        )
    return None


def _resolve_cfg_path(config_values, value):
    path = Path(_strip_cfg_atom(value)).expanduser()
    if path.is_absolute():
        return str(path)
    cfg_filename = config_values.get("_optimizer_config_filename")
    if not cfg_filename:
        return str(path)
    return str((Path(cfg_filename).resolve().parent / path).resolve())


def _format_cfg_value(value):
    if isinstance(value, (list, tuple)):
        return "( " + ", ".join(_format_cfg_value(item) for item in value) + " )"
    if isinstance(value, bool):
        return "YES" if value else "NO"
    return str(value)


def _write_cfg(filename, values):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w") as fp:
        for key, value in values:
            fp.write(f"{key}= {_format_cfg_value(value)}\n")


def generate_initial_bspline_modes(
    filename,
    marker,
    nper_side=7,
    degree=3,
    coefficient=0.0,
    bound_lower=-0.01,
    bound_upper=0.01,
    class_shape="sqrt_x_one_minus_x",
    normalize_basis=True,
    normalization_mode="max",
    surface_mode="BOTH",
):
    nper_side = int(nper_side)
    degree = int(degree)
    if degree != 3:
        raise BSplineAdaptiveError("BSPLINE_INITIAL_DEGREE must currently be 3 for KNOT_INSERTION")
    if nper_side < degree + 1:
        raise BSplineAdaptiveError("BSPLINE_INITIAL_NPER_SIDE must be >= degree + 1")
    bound_lower = float(bound_lower)
    bound_upper = float(bound_upper)
    if bound_upper < bound_lower:
        raise BSplineAdaptiveError("BSPLINE_INITIAL_BOUND_UPPER must be >= lower")
    marker = str(marker or "").strip()
    if not marker:
        raise BSplineAdaptiveError("BSPLINE_MARKER is required to generate initial B-spline modes")
    try:
        surface_mode = normalize_surface_mode(surface_mode)
    except BSplineModeError as exc:
        raise BSplineAdaptiveError(str(exc))

    n_internal = nper_side - degree - 1
    internal = [
        (i + 1) / float(n_internal + 1)
        for i in range(n_internal)
    ]
    knot_vector = [0.0] * (degree + 1) + internal + [1.0] * (degree + 1)
    modes = []
    for side in active_sides_from_surface_mode(surface_mode):
        for basis_index in range(nper_side):
            modes.append(
                {
                    "id": f"{side}_clamped_i{basis_index:03d}",
                    "side": side,
                    "basis_type": "clamped",
                    "degree": degree,
                    "knot_vector": knot_vector,
                    "basis_index": basis_index,
                    "coefficient": float(coefficient),
                    "bounds": [bound_lower, bound_upper],
                    "active": True,
                }
            )
    spec = {
        "version": 1,
        "dimension": 2,
        "marker": marker,
        "chord": {"mode": "auto"},
        "normal_displacement": True,
        "class_shape": str(class_shape),
        "normalize_basis": _as_bool(normalize_basis, default=True),
        "normalization_mode": str(normalization_mode),
        "surface_mode": surface_mode,
        "modes": modes,
    }
    write_mode_spec(validate_mode_spec(spec), filename)
    return spec


def _forced_template_lines(case_config, forced):
    skip_prefixes = ("BSPLINE_", "PROGRESSIVE_")
    skip_keys = {
        "MATH_PROBLEM",
        "MESH_FILENAME",
        "MESH_OUT_FILENAME",
        "OBJECTIVE_FUNCTION",
        "TABULAR_FORMAT",
        "CONV_FILENAME",
        "SOLUTION_FILENAME",
        "RESTART_FILENAME",
        "SOLUTION_ADJ_FILENAME",
        "RESTART_ADJ_FILENAME",
        "SURFACE_ADJ_FILENAME",
        "VOLUME_ADJ_FILENAME",
        "HISTORY_OUTPUT",
        "SCREEN_OUTPUT",
        "DV_KIND",
        "DV_MARKER",
        "DV_FILENAME",
        "DEFINITION_DV",
        "OPT_OBJECTIVE",
        "OPT_ITERATIONS",
        "OPT_ACCURACY",
        "OPT_BOUND_LOWER",
        "OPT_BOUND_UPPER",
        "OPT_RELAX_FACTOR",
        "OPT_GRADIENT_FACTOR",
    }
    lines = []
    if case_config:
        with open(case_config, "r") as fp:
            for raw_line in fp:
                line = raw_line.rstrip("\n")
                stripped = line.strip()
                key = ""
                if stripped and not stripped.startswith(("%", "#")) and "=" in stripped:
                    key = stripped.split("=", 1)[0].strip().upper()
                if key and (key in skip_keys or key.startswith(skip_prefixes)):
                    continue
                lines.append(line)
    if lines and lines[-1].strip():
        lines.append("")
    for key, value in forced:
        lines.append(f"{key}= {_format_cfg_value(value)}")
    return lines


def _write_case_template(case_config, filename, forced):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    lines = _forced_template_lines(case_config, forced)
    with open(filename, "w") as fp:
        fp.write("\n".join(lines).rstrip())
        fp.write("\n")


def generate_missing_templates(settings):
    workdir = Path(settings["workdir"])
    template_dir = workdir / "templates"
    objective = str(settings.get("opt_objective", "DRAG")).upper()
    base_mesh = settings["base_mesh"]
    marker = settings["marker"]
    case_config = settings.get("case_config") or settings.get("optimizer_config")

    if not settings.get("def_template"):
        if not case_config:
            raise BSplineAdaptiveError(
                "missing BSPLINE_DEF_TEMPLATE and no case cfg is available to generate one"
            )

        path = template_dir / "def_template_auto.cfg"

        # SU2_DEF requires DV_MARKER to also exist in the BC marker lists.
        # Therefore the deformation template must preserve MARKER_* entries
        # from the case cfg, e.g. MARKER_EULER=(AIRFOIL), MARKER_FAR=(FARFIELD).
        #
        # SURFACE_FILE deformation also requires the standard DV_PARAM/DV_VALUE
        # entries. Without these, SU2_DEF may run but leave the mesh unchanged.
        _write_case_template(
            case_config,
            path,
            [
                ("MESH_FILENAME", base_mesh),
                ("MESH_OUT_FILENAME", "deformed_mesh.su2"),
                ("MESH_FORMAT", "SU2"),
                ("DV_KIND", "SURFACE_FILE"),
                ("DV_MARKER", [marker]),
                ("DV_PARAM", [1, 0.5]),
                ("DV_VALUE", 0.0),
                ("DV_FILENAME", "surface_positions.dat"),
                ("DEFORM_LINEAR_SOLVER", "FGMRES"),
                ("DEFORM_LINEAR_SOLVER_ERROR", 1e-14),
                ("DEFORM_LINEAR_SOLVER_ITER", 500),
            ],
        )

        settings["def_template"] = str(path.resolve())

    if not settings.get("primal_template"):
        if not case_config:
            raise BSplineAdaptiveError("missing BSPLINE_PRIMAL_TEMPLATE and no case cfg is available to generate one")
        path = template_dir / "primal_template_auto.cfg"
        _write_case_template(
            case_config,
            path,
            [
                ("MATH_PROBLEM", "DIRECT"),
                ("OBJECTIVE_FUNCTION", objective),
                ("MESH_FILENAME", "mesh.su2"),
                ("MESH_OUT_FILENAME", "primal_mesh_out.su2"),
                ("TABULAR_FORMAT", "CSV"),
                ("CONV_FILENAME", "history_primal"),
                ("SOLUTION_FILENAME", "solution_flow.dat"),
                ("RESTART_FILENAME", "restart_flow.dat"),
                ("HISTORY_OUTPUT", ["INNER_ITER", "RMS_RES", "AERO_COEFF"]),
                ("SCREEN_OUTPUT", ["INNER_ITER", "RMS_RES", "LIFT", "DRAG"]),
            ],
        )
        settings["primal_template"] = str(path.resolve())

    if not settings.get("adjoint_template"):
        if not case_config:
            raise BSplineAdaptiveError("missing BSPLINE_ADJOINT_TEMPLATE and no case cfg is available to generate one")
        path = template_dir / "adjoint_template_auto.cfg"
        _write_case_template(
            case_config,
            path,
            [
                ("MATH_PROBLEM", "DISCRETE_ADJOINT"),
                ("OBJECTIVE_FUNCTION", objective),
                ("MESH_FILENAME", "mesh.su2"),
                ("SOLUTION_FILENAME", "solution_flow.dat"),
                ("RESTART_FILENAME", "restart_flow.dat"),
                ("SOLUTION_ADJ_FILENAME", "solution_adj.dat"),
                ("RESTART_ADJ_FILENAME", "solution_adj.dat"),
                ("SURFACE_ADJ_FILENAME", "surface_adjoint"),
                ("VOLUME_ADJ_FILENAME", "volume_adjoint"),
                ("TABULAR_FORMAT", "CSV"),
                ("CONV_FILENAME", "history_adjoint"),
                ("HISTORY_OUTPUT", ["INNER_ITER", "RMS_RES", "AERO_COEFF"]),
                ("SCREEN_OUTPUT", ["INNER_ITER", "RMS_RES"]),
            ],
        )
        settings["adjoint_template"] = str(path.resolve())


def prepare_bspline_launch_settings(settings):
    settings = dict(settings)
    if settings.get("_prepared"):
        return settings

    if settings.get("candidate_bank"):
        print(
            "[PROGRESSIVE_BSPLINE] WARNING: --candidate-bank is ignored. "
            "Candidate/generated refinement has been removed."
        )
    settings["candidate_bank"] = None
    try:
        settings["surface_mode"] = normalize_surface_mode(
            settings.get("surface_mode", "BOTH")
        )
    except BSplineModeError as exc:
        raise BSplineAdaptiveError(str(exc))

    if settings.get("nproc") is not None and not settings.get("_mpi_cli_provided", False):
        settings["mpi"] = f"mpirun -n {int(settings['nproc'])}"
    settings.setdefault("mpi", "")

    for key in ("base_mesh", "marker", "workdir"):
        if not settings.get(key):
            raise BSplineAdaptiveError(f"missing required B-spline setting: {key}")

    workdir = Path(settings["workdir"]).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    settings["workdir"] = str(workdir)

    generated_initial = False
    if not settings.get("modes"):
        if _as_bool(settings.get("generate_initial_modes", False), default=False):
            modes_path = workdir / "generated" / "initial_modes.json"
            generate_initial_bspline_modes(
                modes_path,
                settings["marker"],
                nper_side=settings.get("initial_nper_side", 7),
                degree=settings.get("initial_degree", 3),
                coefficient=settings.get("initial_coefficient", 0.0),
                bound_lower=settings.get("initial_bound_lower", -0.01),
                bound_upper=settings.get("initial_bound_upper", 0.01),
                class_shape=settings.get(
                    "initial_class_shape",
                    "sqrt_x_one_minus_x",
                ),
                normalize_basis=settings.get("initial_normalize_basis", True),
                normalization_mode=settings.get("initial_normalization_mode", "max"),
                surface_mode=settings["surface_mode"],
            )
            settings["modes"] = str(modes_path.resolve())
            generated_initial = True
        else:
            raise BSplineAdaptiveError(
                "missing --modes or BSPLINE_MODES or BSPLINE_GENERATE_INITIAL_MODES=YES"
            )
    settings["_initial_modes_generated"] = generated_initial

    generate_missing_templates(settings)
    for key in (
        "modes",
        "base_mesh",
        "def_template",
        "primal_template",
        "adjoint_template",
    ):
        if settings.get(key):
            settings[key] = str(Path(settings[key]).resolve())
    settings["_prepared"] = True
    return settings


def print_startup_summary(settings):
    case_config = settings.get("case_config") or settings.get("optimizer_config") or ""
    case_config = Path(case_config).name if case_config else ""
    print(f"[PROGRESSIVE_BSPLINE] Case config: {case_config}")
    print(f"[PROGRESSIVE_BSPLINE] modes: {settings.get('modes')}")
    print(
        "[PROGRESSIVE_BSPLINE] initial modes generated: "
        f"{'YES' if settings.get('_initial_modes_generated') else 'NO'}"
    )
    print(f"[PROGRESSIVE_BSPLINE] initial n_per_side: {settings.get('initial_nper_side', '')}")
    print(f"[PROGRESSIVE_BSPLINE] base mesh: {settings.get('base_mesh')}")
    print(f"[PROGRESSIVE_BSPLINE] marker: {settings.get('marker')}")
    print(f"[PROGRESSIVE_BSPLINE] def template: {settings.get('def_template')}")
    print(f"[PROGRESSIVE_BSPLINE] primal template: {settings.get('primal_template')}")
    print(f"[PROGRESSIVE_BSPLINE] adjoint template: {settings.get('adjoint_template')}")
    print(f"[PROGRESSIVE_BSPLINE] workdir: {settings.get('workdir')}")
    print(f"[PROGRESSIVE_BSPLINE] mpi: {settings.get('mpi', '')}")
    print("[PROGRESSIVE_BSPLINE] eval layout: DSN")
    print("[PROGRESSIVE_BSPLINE] sensitivity weighting: NODAL")
    print("[PROGRESSIVE_BSPLINE] refinement: KNOT_INSERTION")
    print("[PROGRESSIVE_BSPLINE] refinement state: INITIAL_MESH_KEEP_DV")
    print(
        "[PROGRESSIVE_BSPLINE] deformation direction: "
        f"{settings.get('deformation_direction_mode', 'NORMAL')}"
    )
    surface_mode = settings.get("surface_mode", "BOTH")
    active_sides = active_sides_from_surface_mode(surface_mode)
    try:
        ndv = len(active_mode_ids(load_mode_spec(settings["modes"])))
    except Exception:
        ndv = ""
    print(f"[PROGRESSIVE_BSPLINE][SURFACE] mode = {surface_mode}")
    print(f"[PROGRESSIVE_BSPLINE][SURFACE] active sides = {active_sides}")
    print(f"[PROGRESSIVE_BSPLINE][SURFACE] ndv = {ndv}")
    print(
        "[PROGRESSIVE_BSPLINE][SURFACE] deformation direction = "
        f"{settings.get('deformation_direction_mode', 'NORMAL')}"
    )
    thickness_options = settings.get("thickness_options") or {}
    if _as_bool(
        thickness_options.get("PROGRESSIVE_THICKNESS_CONSTRAINT", False),
        default=False,
    ):
        print(
            "[PROGRESSIVE_BSPLINE][THICKNESS] domain = "
            f"{thickness_options.get('PROGRESSIVE_THICKNESS_DOMAIN_MODE')}"
        )
        print(
            "[PROGRESSIVE_BSPLINE][THICKNESS] symmetry_y = "
            f"{float(thickness_options.get('PROGRESSIVE_THICKNESS_SYMMETRY_Y', 0.0))}"
        )
    print(f"[PROGRESSIVE_BSPLINE] online trigger: {settings.get('trigger', 'MAX_ITER')}")


def _settings_from_args(args):
    return {
        "case_config": getattr(args, "_case_config", getattr(args, "case_config", None)),
        "modes": args.modes,
        "candidate_bank": args.candidate_bank,
        "base_mesh": args.base_mesh,
        "marker": args.marker,
        "def_template": args.def_template,
        "primal_template": args.primal_template,
        "adjoint_template": args.adjoint_template,
        "workdir": args.workdir,
        "objective_column": args.objective_column,
        "optimizer_config": args.optimizer_config,
        "mpi": args.mpi,
        "nproc": getattr(args, "nproc", None),
        "_mpi_cli_provided": getattr(args, "_mpi_cli_provided", False),
        "nlevels": args.nlevels,
        "nfinal": args.nfinal,
        "max_iter_per_level": args.max_iter_per_level,
        "refinement": args.refinement,
        "refine_mode": args.refine_mode,
        "refine_state": args.refine_state,
        "knot_score_mode": args.knot_score_mode,
        "knot_insertions_per_refine": args.knot_insertions_per_refine,
        "knot_min_span_width": args.knot_min_span_width,
        "transfer_method": args.transfer_method,
        "transfer_bound_policy": args.transfer_bound_policy,
        "transfer_geometry_abs_tol": args.transfer_geometry_abs_tol,
        "transfer_geometry_rel_tol": args.transfer_geometry_rel_tol,
        "nadd_mode": args.nadd_mode,
        "batch_size_max": args.batch_size_max,
        "growth_ratio": args.growth_ratio,
        "fixed_nadd": args.fixed_nadd,
        "sensitivity_weighting": getattr(args, "sensitivity_weighting", "NODAL"),
        "eval_layout": args.eval_layout,
        "objective_adjoint": args.objective_adjoint,
        "symmetry_coupling": args.symmetry_coupling,
        "surface_mode": getattr(args, "surface_mode", "BOTH"),
        "deformation_direction_mode": getattr(
            args,
            "deformation_direction_mode",
            None,
        ),
        "trigger": args.trigger,
        "window": args.window,
        "tol": args.tol,
        "eps": args.eps,
        "slope_filter_tol": args.slope_filter_tol,
        "slope_patience": args.slope_patience,
        "stag_window": args.stag_window,
        "stag_tol": args.stag_tol,
        "stag_band": args.stag_band,
        "stag_patience": args.stag_patience,
        "warmup_iter": args.warmup_iter,
        "generate_initial_modes": getattr(args, "generate_initial_modes", None),
        "initial_nper_side": getattr(args, "initial_nper_side", 7),
        "initial_degree": getattr(args, "initial_degree", 3),
        "initial_coefficient": getattr(args, "initial_coefficient", 0.0),
        "initial_bound_lower": getattr(args, "initial_bound_lower", -0.01),
        "initial_bound_upper": getattr(args, "initial_bound_upper", 0.01),
        "initial_class_shape": getattr(args, "initial_class_shape", "sqrt_x_one_minus_x"),
        "initial_normalize_basis": getattr(args, "initial_normalize_basis", True),
        "initial_normalization_mode": getattr(args, "initial_normalization_mode", "max"),
        "opt_objective": getattr(args, "opt_objective", None),
        "dry_run": args.dry_run,
        "show_commands": args.show_commands,
        "stream_solver_output": args.stream_solver_output,
        "print_optimizer_table": args.print_optimizer_table,
        "log_active_modes": args.log_active_modes,
        "opt_accuracy": getattr(args, "opt_accuracy", None),
        "opt_bound_upper": args.opt_bound_upper,
        "opt_bound_lower": args.opt_bound_lower,
        "opt_relax_factor": args.opt_relax_factor,
        "opt_gradient_factor": args.opt_gradient_factor,
        "opt_line_search_bound": args.opt_line_search_bound,
        "local_step_limit": args.local_step_limit,
        "local_step_limit_ratio": args.local_step_limit_ratio,
        "thickness_options": getattr(args, "thickness_options", None),
        "auto_scale_bounds_to_geometry": args.auto_scale_bounds_to_geometry,
        "max_normal_displacement": args.max_normal_displacement,
        "max_rms_normal_displacement": args.max_rms_normal_displacement,
        "min_bound_scale": args.min_bound_scale,
        "le_safe_direction": getattr(args, "le_safe_direction", False),
        "le_safe_x0": getattr(args, "le_safe_x0", LE_SAFE_DEFAULT_X0),
        "le_safe_x1": getattr(args, "le_safe_x1", LE_SAFE_DEFAULT_X1),
        "le_safe_power": getattr(args, "le_safe_power", LE_SAFE_DEFAULT_POWER),
    }


def progressive_bspline_su2_shape_optimization(settings):
    settings = prepare_bspline_launch_settings(settings)
    settings = validate_adaptive_options(settings)
    print_startup_summary(settings)
    workdir = Path(settings["workdir"]).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    with open(workdir / "adaptive_settings.json", "w") as fp:
        json.dump(settings, fp, indent=2, sort_keys=True)
        fp.write("\n")

    initial_modes = load_mode_spec(settings["modes"])
    try:
        validate_surface_mode_against_modes(initial_modes, settings["surface_mode"])
    except BSplineModeError as exc:
        raise BSplineAdaptiveError(str(exc))
    if settings["surface_mode"] != "BOTH" and "surface_mode" not in initial_modes:
        initial_modes["surface_mode"] = settings["surface_mode"]
    current_modes = initial_modes
    adaptive_rows = []

    for level_id in range(settings["nlevels"]):
        level = build_level(
            level_id,
            current_modes,
            workdir,
            settings["modes"],
        )
        kept = len(active_mode_ids(current_modes)) - len(settings.get("_last_added_ids", []))
        added = len(settings.get("_last_added_ids", []))
        write_level_start(level)
        _print_level_start(
            level,
            settings["refine_state"],
            kept=kept,
            added=added,
            log_active_modes=settings.get("log_active_modes", False),
        )

        if settings.get("dry_run", False):
            print("[PROGRESSIVE_BSPLINE] Dry run requested; not running SU2.")
            return {
                "status": "dry_run",
                "workdir": str(workdir),
                "level": level,
            }

        current_reduced_ndv = refinement_limit_ndv(current_modes, settings)
        trigger_opts = build_online_trigger_opts(
            settings["trigger"],
            current_level=level_id,
            current_ndv=current_reduced_ndv,
            final_ndv=settings.get("nfinal"),
            nlevels=settings.get("nlevels"),
            window=settings["window"],
            tolerance=settings["tol"],
            filter_tolerance=settings["slope_filter_tol"],
            warmup=settings["warmup_iter"],
            eps=settings["eps"],
            patience=settings["slope_patience"],
            stagnation_tolerance=settings["stag_tol"],
            stagnation_band=settings["stag_band"],
            stagnation_window=settings["stag_window"],
        )

        result = run_bspline_su2_optimization(
            modes_filename=str(level.active_modes_start_filename),
            base_mesh=settings["base_mesh"],
            marker=settings["marker"],
            def_template=settings["def_template"],
            primal_template=settings["primal_template"],
            adjoint_template=settings["adjoint_template"],
            workdir=str(level.opt_workdir),
            objective_column=settings["objective_column"],
            maxiter=settings["max_iter_per_level"],
            mpi_prefix=settings.get("mpi", ""),
            show_commands=bool(settings.get("show_commands", False)),
            stream_solver_output=bool(settings.get("stream_solver_output", False)),
            print_optimizer_table=bool(settings.get("print_optimizer_table", True)),
            auto_scale_bounds_to_geometry=bool(settings.get("auto_scale_bounds_to_geometry", False)),
            max_normal_displacement=settings.get("max_normal_displacement"),
            max_rms_normal_displacement=settings.get("max_rms_normal_displacement"),
            min_bound_scale=settings.get("min_bound_scale", 0.0),
            opt_accuracy=settings.get("opt_accuracy"),
            opt_bound_upper=settings.get("opt_bound_upper"),
            opt_bound_lower=settings.get("opt_bound_lower"),
            opt_relax_factor=settings.get("opt_relax_factor", 1.0),
            opt_gradient_factor=settings.get("opt_gradient_factor", 1.0),
            opt_line_search_bound=settings.get("opt_line_search_bound"),
            local_step_limit=settings.get("local_step_limit", False),
            local_step_limit_ratio=settings.get("local_step_limit_ratio", 200.0),
            sensitivity_weighting=settings.get("sensitivity_weighting", "NODAL"),
            thickness_options=settings.get("thickness_options"),
            eval_layout=settings.get("eval_layout", "DSN"),
            objective_adjoint=settings.get("objective_adjoint", "drag"),
            symmetry_coupling=settings.get("symmetry_coupling", "NONE"),
            surface_mode=settings.get("surface_mode", "BOTH"),
            trigger_opts=trigger_opts,
            progressive_label="PROGRESSIVE_BSPLINE",
            deformation_direction_mode=settings.get("deformation_direction_mode"),
            le_safe_direction=bool(settings.get("le_safe_direction", False)),
            le_safe_x0=settings.get("le_safe_x0", LE_SAFE_DEFAULT_X0),
            le_safe_x1=settings.get("le_safe_x1", LE_SAFE_DEFAULT_X1),
            le_safe_power=settings.get("le_safe_power", LE_SAFE_DEFAULT_POWER),
        )

        opt_rows = _read_optimization_history(level.opt_workdir)
        optimized_modes = load_mode_spec(str(level.optimized_modes_filename))
        adjoint_eval_dir = find_eval_dir_for_mode_coefficients(
            level.opt_workdir,
            optimized_modes,
        )
        best_objective = min(row["_objective"] for row in opt_rows)
        print(
            "[PROGRESSIVE_BSPLINE] Level {} optimization complete | best {} = {:.6e} | adjoint eval = {}".format(
                level_id,
                settings["objective_column"],
                best_objective,
                adjoint_eval_dir.name,
            )
        )

        reached_limits = (
            level_id >= settings["nlevels"] - 1
            or (
                settings["nfinal"] is not None
                and refinement_limit_ndv(optimized_modes, settings) >= settings["nfinal"]
            )
        )
        trigger_mode = str(settings["trigger"]).upper()
        if trigger_mode == "MAX_ITER":
            refine_requested = True
            trigger_reason = "level_complete"
            trigger_counter = 1
        elif result.get("early_refine_triggered", False):
            refine_requested = True
            trigger_reason = "early_refine_trigger"
            trigger_counter = 1
        else:
            refine_requested = False
            trigger_reason = "online_trigger_not_fired"
            trigger_counter = 0
        trigger_decision = TriggerDecision(
            trigger_mode=trigger_mode,
            threshold=settings["tol"] if trigger_mode != "MAX_ITER" else "",
            window=settings["window"] if trigger_mode != "MAX_ITER" else "",
            patience=settings["slope_patience"] if trigger_mode != "MAX_ITER" else "",
            counter=trigger_counter,
            refine_now=refine_requested,
            reason=trigger_reason,
        )
        refine_now = bool(not reached_limits and trigger_decision.refine_now)
        if trigger_decision.refine_now and not refine_now:
            if reached_limits:
                trigger_decision.reason = "limits_reached"
            trigger_decision.refine_now = False
            _log_trigger_decision(trigger_decision)

        selected_modes = []
        knot_score_rows = []
        knot_selected_data = {}
        knot_refine_used = False
        if refine_now:
            metadata, signal = load_adjoint_signal(
                adjoint_eval_dir / "surface_adjoint.csv",
                adjoint_eval_dir / "bspline_surface_metadata.csv",
            )
            next_modes, knot_score_rows, knot_selected_data = build_next_knot_inserted_modes(
                optimized_modes,
                metadata,
                signal,
                settings,
            )
            knot_score_file = level.workdir / f"knot_span_scores_level_{level_id:03d}.csv"
            knot_selected_file = level.workdir / f"selected_knot_refinement_level_{level_id:03d}.json"
            write_knot_span_scores_csv(knot_score_rows, knot_score_file)
            write_selected_knot_refinement_json(
                level_id,
                knot_selected_data,
                knot_score_rows,
                knot_selected_file,
            )
            if next_modes is not None:
                next_file = level.workdir / "active_modes_next.json"
                write_mode_spec(next_modes, next_file)
                knot_refine_used = True
                current_modes = next_modes
                settings["_last_added_ids"] = [
                    mode_id
                    for mode_id in active_mode_ids(next_modes)
                    if mode_id not in set(active_mode_ids(optimized_modes))
                ]
                print(
                    "[PROGRESSIVE_BSPLINE] Level {} | NDV = {} | refined span=[{:.6f},{:.6f}] knot={:.6f} | added modes={}".format(
                        level_id + 1,
                        len(active_mode_ids(next_modes)),
                        float(knot_selected_data["span_left"]),
                        float(knot_selected_data["span_right"]),
                        float(knot_selected_data["inserted_knot"]),
                        int(knot_selected_data["ndv_after"]) - int(knot_selected_data["ndv_before"]),
                    )
                )
                print(
                    "[PROGRESSIVE_BSPLINE] refinement | ndv_before={} ndv_after={} "
                    "selected side={} x={:.6f}".format(
                        int(knot_selected_data["ndv_before"]),
                        int(knot_selected_data["ndv_after"]),
                        str(knot_selected_data.get("side", "BOTH")).upper(),
                        float(knot_selected_data["inserted_knot"]),
                    )
                )
                print(
                    "[PROGRESSIVE_BSPLINE] KNOT_INSERTION transfer | "
                    "rms={:.6e} max={:.6e} rel={:.6e}".format(
                        float(knot_selected_data["transfer_rms_error"]),
                        float(knot_selected_data["transfer_max_error"]),
                        float(knot_selected_data["transfer_relative_error"]),
                    )
                )
                print(
                    "[PROGRESSIVE_BSPLINE] Coefficients transferred from previous level: kept={} added={} transfer_max={:.6e}".format(
                        int(knot_selected_data["ndv_before"]),
                        int(knot_selected_data["ndv_after"]) - int(knot_selected_data["ndv_before"]),
                        float(knot_selected_data["transfer_max_error"]),
                    )
                )
            else:
                print("[PROGRESSIVE_BSPLINE] KNOT_INSERTION refine | no positive knot span selected")
                refine_now = False
                current_modes = optimized_modes
        else:
            current_modes = optimized_modes

        level_summary = _level_summary_row(
            level,
            opt_rows,
            selected_modes,
            trigger_decision,
            refine_now,
            "ok",
        )
        if knot_refine_used and knot_selected_data:
            level_summary["n_added"] = int(knot_selected_data["ndv_after"]) - int(
                knot_selected_data["ndv_before"]
            )
            level_summary["selected_ids"] = "knot@{:.12g}".format(
                float(knot_selected_data["inserted_knot"])
            )
        with open(level.workdir / "level_summary.json", "w") as fp:
            json.dump(
                level_summary
                | {
                    "refine_mode": settings["refine_mode"],
                    "knot_score_mode": settings["knot_score_mode"],
                    "knot_refine_used": knot_refine_used,
                    "knot_refinement": knot_selected_data,
                    "optimizer_result": result,
                    "sensitivity_weighting": settings["sensitivity_weighting"],
                },
                fp,
                indent=2,
                sort_keys=True,
            )
            fp.write("\n")
        adaptive_rows.append(level_summary)
        _write_adaptive_history(adaptive_rows, workdir / "adaptive_history.csv")

        if not refine_now:
            break

    write_mode_spec(current_modes, workdir / "active_modes_final.json")
    return {
        "status": "ok",
        "workdir": str(workdir),
        "active_modes_final": str(workdir / "active_modes_final.json"),
        "levels": len(adaptive_rows),
    }


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Progressive/adaptive external B-spline SU2 optimization."
    )
    parser.add_argument("-f", "--case-config", default=None)
    parser.add_argument("-n", "--nproc", type=int, default=None)
    parser.add_argument("--modes", default=None)
    parser.add_argument(
        "--candidate-bank",
        default=None,
        help="Deprecated and ignored; candidate/generated refinement has been removed.",
    )
    parser.add_argument("--base-mesh", default=None)
    parser.add_argument("--marker", default=None)
    parser.add_argument("--def-template", default=None)
    parser.add_argument("--primal-template", default=None)
    parser.add_argument("--adjoint-template", default=None)
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--optimizer-config", default=None)
    parser.add_argument("--objective-column", default="CD")
    parser.add_argument("--mpi", default="")
    parser.add_argument("--nlevels", type=int, default=1)
    parser.add_argument("--nfinal", type=int, default=None)
    parser.add_argument("--max-iter-per-level", type=int, default=5)
    parser.add_argument("--refinement", default="ADAPTIVE")
    parser.add_argument("--refine-mode", default="KNOT_INSERTION")
    parser.add_argument("--refine-state", default=REFINE_STATE)
    parser.add_argument("--knot-score-mode", default="VIRTUAL_INSERTION", choices=ALLOWED_KNOT_SCORE_MODES)
    parser.add_argument("--knot-insertions-per-refine", default="1")
    parser.add_argument("--knot-min-span-width", type=float, default=1.0e-8)
    parser.add_argument("--transfer-method", default="BOEHM")
    parser.add_argument("--transfer-bound-policy", default="ERROR")
    parser.add_argument("--transfer-geometry-abs-tol", type=float, default=1.0e-10)
    parser.add_argument("--transfer-geometry-rel-tol", type=float, default=1.0e-8)
    parser.add_argument("--nadd-mode", default="GROWTH_RATIO")
    parser.add_argument("--batch-size-max", type=int, default=1)
    parser.add_argument("--growth-ratio", type=float, default=2.0)
    parser.add_argument("--fixed-nadd", type=int, default=1)
    parser.add_argument("--eval-layout", default="DSN", choices=ALLOWED_EVAL_LAYOUTS)
    parser.add_argument("--objective-adjoint", default="drag")
    parser.add_argument("--symmetry-coupling", default="NONE", choices=ALLOWED_SYMMETRY_COUPLINGS)
    parser.add_argument("--surface-mode", default="BOTH", choices=ALLOWED_SURFACE_MODES)
    parser.add_argument(
        "--deformation-direction",
        dest="deformation_direction_mode",
        default=None,
        choices=ALLOWED_DEFORMATION_DIRECTION_MODES,
    )
    parser.add_argument("--trigger", default="MAX_ITER")
    parser.add_argument("--trigger-window", dest="window", type=int, default=1)
    parser.add_argument("--trigger-ratio", dest="tol", type=float, default=0.2)
    parser.add_argument("--trigger-eps", dest="eps", type=float, default=1.0e-300)
    parser.add_argument("--slope-filter-tol", type=float, default=0.02)
    parser.add_argument("--slope-patience", type=int, default=1)
    parser.add_argument("--stagnation-window", dest="stag_window", type=int, default=3)
    parser.add_argument("--stagnation-rel-tol", dest="stag_tol", type=float, default=1.0e-3)
    parser.add_argument("--stagnation-band", dest="stag_band", type=float, default=0.02)
    parser.add_argument("--stagnation-patience", dest="stag_patience", type=int, default=1)
    parser.add_argument("--warmup-iter", type=int, default=0)
    parser.add_argument("--generate-initial-modes", default=None)
    parser.add_argument("--initial-nper-side", type=int, default=7)
    parser.add_argument("--initial-degree", type=int, default=3)
    parser.add_argument("--initial-coefficient", type=float, default=0.0)
    parser.add_argument("--initial-bound-lower", type=float, default=-0.01)
    parser.add_argument("--initial-bound-upper", type=float, default=0.01)
    parser.add_argument("--initial-class-shape", default="sqrt_x_one_minus_x")
    parser.add_argument("--initial-normalize-basis", default=True)
    parser.add_argument("--initial-normalization-mode", default="max")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--show-commands", action="store_true", default=False)
    parser.add_argument("--stream-solver-output", action="store_true", default=False)
    parser.add_argument("--quiet-driver", dest="show_commands", action="store_false")
    parser.add_argument("--no-optimizer-table", dest="print_optimizer_table", action="store_false")
    parser.set_defaults(print_optimizer_table=True)
    parser.add_argument("--log-active-modes", action="store_true", default=False)
    parser.add_argument("--auto-scale-bounds-to-geometry", action="store_true", default=False)
    parser.add_argument("--max-normal-displacement", type=float, default=None)
    parser.add_argument("--max-rms-normal-displacement", type=float, default=None)
    parser.add_argument("--min-bound-scale", type=float, default=0.0)
    parser.add_argument("--opt-bound-upper", type=float, default=None)
    parser.add_argument("--opt-bound-lower", type=float, default=None)
    parser.add_argument("--opt-relax-factor", type=float, default=1.0)
    parser.add_argument("--opt-gradient-factor", type=float, default=1.0)
    parser.add_argument("--opt-line-search-bound", type=float, default=None)
    parser.add_argument("--local-step-limit", action="store_true", default=False)
    parser.add_argument("--local-step-limit-ratio", type=float, default=200.0)
    parser.add_argument(
        "--le-safe-direction",
        action="store_true",
        default=False,
        help="Use the fixed-leading-edge-safe deformation direction near x/c=0",
    )
    parser.add_argument("--le-safe-x0", type=float, default=LE_SAFE_DEFAULT_X0)
    parser.add_argument("--le-safe-x1", type=float, default=LE_SAFE_DEFAULT_X1)
    parser.add_argument("--le-safe-power", type=float, default=LE_SAFE_DEFAULT_POWER)
    return parser


def _option_provided(argv, option):
    for item in list(argv or []):
        text = str(item)
        if text == option or text.startswith(option + "="):
            return True
    return False


def parse_adaptive_options(argv=None):
    parser = _build_arg_parser()
    argv_list = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)
    args._mpi_cli_provided = _option_provided(argv_list, "--mpi")
    if args.case_config and not args.optimizer_config:
        args.optimizer_config = args.case_config
    args = apply_optimizer_config_to_args(
        args,
        parser,
        argv,
        adaptive_options_from_config,
        warning_prefix="[PROGRESSIVE_BSPLINE]",
    )
    settings = prepare_bspline_launch_settings(_settings_from_args(args))
    return validate_adaptive_options(settings)


def main(argv=None):
    parser = _build_arg_parser()
    try:
        argv_list = list(sys.argv[1:] if argv is None else argv)
        args = parser.parse_args(argv)
        args._mpi_cli_provided = _option_provided(argv_list, "--mpi")
        if args.case_config and not args.optimizer_config:
            args.optimizer_config = args.case_config
        args = apply_optimizer_config_to_args(
            args,
            parser,
            argv,
            adaptive_options_from_config,
            warning_prefix="[PROGRESSIVE_BSPLINE]",
        )
        result = progressive_bspline_su2_shape_optimization(_settings_from_args(args))
    except (BSplineAdaptiveError, BSplineModeError, BSplineSU2DriverError, OSError, ValueError, NotImplementedError) as exc:
        parser.error(str(exc))

    print("[PROGRESSIVE_BSPLINE] Status: {}".format(result["status"]))
    if result.get("active_modes_final"):
        print("[PROGRESSIVE_BSPLINE] Wrote {}".format(result["active_modes_final"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
