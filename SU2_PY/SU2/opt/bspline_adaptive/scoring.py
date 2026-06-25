"""Knot-span scoring helpers for adaptive B-splines."""

import math

import numpy as np

from SU2.opt.bspline_dot import (
    match_sensitivities_to_metadata,
    read_metadata,
    read_sensitivity_file,
)

from .errors import BSplineAdaptiveError
from .knot_space import (
    insert_knot_midpoint,
    knot_insertion_spans,
    reduced_basis_matrix_for_space,
    regenerate_clamped_modes,
)

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

# Leading/trailing-edge closure nodes (x/c == 0 and == 1) are geometrically
# pinned and carry a degenerate surface normal. Their sensitivity — typically a
# trailing-edge adjoint singularity — cannot be reduced by knot insertion, so it
# must never drive refinement. These two endpoints are always dropped from the
# scoring point set (the geometry-transfer check still uses the full metadata).
SCORING_CLOSURE_NODE_EPS = 1.0e-6

def _drop_closure_nodes(metadata, signal, eps=SCORING_CLOSURE_NODE_EPS):
    signal = np.asarray(signal, dtype=float)
    kept_metadata = []
    kept_signal = []
    for row, value in zip(metadata, signal):
        x_over_c = float(row["x_over_c"])
        if x_over_c <= float(eps) or x_over_c >= 1.0 - float(eps):
            continue
        kept_metadata.append(row)
        kept_signal.append(float(value))
    return kept_metadata, np.asarray(kept_signal, dtype=float)

def score_knot_spans(space, metadata, signal, settings, regularization=1.0e-12):
    knot_score_mode = str(settings.get("knot_score_mode", "VIRTUAL_INSERTION")).upper()
    metadata, signal = _drop_closure_nodes(metadata, signal)
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
