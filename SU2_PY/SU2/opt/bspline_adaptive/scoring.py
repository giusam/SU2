"""Knot-span scoring helpers for adaptive B-splines."""

import math

import numpy as np

from SU2.opt.bspline_dot import (
    match_sensitivities_to_metadata,
    read_metadata,
    read_sensitivity_file,
)

from .errors import BSplineAdaptiveError
from .diagnostics import (
    candidate_id,
    old_basis_payload,
    record_scoring_pass,
    scoring_pass_active,
)
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

def _rank_incremental_columns(
    active_matrix,
    candidate_matrix,
    regularization,
    return_diagnostics=False,
):
    # Return an orthonormal basis for the truly new incremental subspace.
    #
    # Knot insertion can add several candidate columns, but after projection
    # onto the complement of the active basis many of them can be nearly
    # linearly dependent. Returning the raw residualized columns makes
    # Z.T @ Z nearly singular. We therefore residualize the block and compress
    # it with SVD to its numerical rank.
    residual_columns = []
    residual_norms_all = []

    for column in range(candidate_matrix.shape[1]):
        z = residualize_candidate(
            active_matrix,
            candidate_matrix[:, column],
            regularization=regularization,
        )
        h = float(np.dot(z, z))
        residual_norms_all.append(math.sqrt(max(h, 0.0)) if math.isfinite(h) else math.nan)
        if h > float(regularization) and math.isfinite(h):
            residual_columns.append(z)

    if not residual_columns:
        basis = np.zeros((candidate_matrix.shape[0], 0), dtype=float)
        if return_diagnostics:
            return basis, {
                "candidate_columns": int(candidate_matrix.shape[1]),
                "residual_columns_kept": 0,
                "rejected_columns": int(candidate_matrix.shape[1]),
                "residual_column_norms_all": residual_norms_all,
                "residual_column_norms_kept": [],
                "svd_backend": "none",
                "svd_singular_values_all": [],
                "svd_singular_values_kept": [],
                "svd_tol": None,
                "incremental_rank": 0,
                "rank_gap": None,
                "raw_incremental_condition": None,
                "gram_condition_after_svd": None,
                "u_orthogonality_error": None,
                "old_space_orthogonality": None,
            }
        return basis

    Z = np.column_stack(residual_columns)
    residual_norms_kept = [
        float(np.linalg.norm(Z[:, column]))
        for column in range(Z.shape[1])
    ]

    def _basis_diagnostics(basis, singular_values, tol, backend, rank):
        if int(rank) <= 0 or basis.shape[1] == 0:
            return {
                "candidate_columns": int(candidate_matrix.shape[1]),
                "residual_columns_kept": int(len(residual_columns)),
                "rejected_columns": int(candidate_matrix.shape[1] - len(residual_columns)),
                "residual_column_norms_all": residual_norms_all,
                "residual_column_norms_kept": residual_norms_kept,
                "svd_backend": backend,
                "svd_singular_values_all": [float(value) for value in singular_values],
                "svd_singular_values_kept": [],
                "svd_tol": None if tol is None else float(tol),
                "incremental_rank": 0,
                "rank_gap": None,
                "raw_incremental_condition": None,
                "gram_condition_after_svd": None,
                "u_orthogonality_error": None,
                "old_space_orthogonality": None,
            }
        singular_values = [float(value) for value in singular_values]
        kept = singular_values[: int(rank)]
        rank_gap = (
            kept[-1] / singular_values[int(rank)]
            if int(rank) < len(singular_values) and singular_values[int(rank)] != 0.0
            else None
        )
        raw_condition = (
            kept[0] / kept[-1]
            if kept and kept[-1] != 0.0
            else None
        )
        gram = basis.T.dot(basis)
        try:
            gram_condition = float(np.linalg.cond(gram))
        except Exception:
            gram_condition = math.inf
        try:
            u_error = float(np.linalg.norm(gram - np.eye(gram.shape[0]), ord="fro"))
        except Exception:
            u_error = math.inf
        denom = float(np.linalg.norm(active_matrix, ord="fro") * np.linalg.norm(basis, ord="fro")) + 1.0e-300
        try:
            old_orthogonality = float(np.linalg.norm(active_matrix.T.dot(basis), ord="fro") / denom)
        except Exception:
            old_orthogonality = math.inf
        return {
            "candidate_columns": int(candidate_matrix.shape[1]),
            "residual_columns_kept": int(len(residual_columns)),
            "rejected_columns": int(candidate_matrix.shape[1] - len(residual_columns)),
            "residual_column_norms_all": residual_norms_all,
            "residual_column_norms_kept": residual_norms_kept,
            "svd_backend": backend,
            "svd_singular_values_all": singular_values,
            "svd_singular_values_kept": kept,
            "svd_tol": None if tol is None else float(tol),
            "incremental_rank": int(rank),
            "rank_gap": rank_gap,
            "raw_incremental_condition": raw_condition,
            "gram_condition_after_svd": gram_condition,
            "u_orthogonality_error": u_error,
            "old_space_orthogonality": old_orthogonality,
        }

    try:
        U, S, _Vt = np.linalg.svd(Z, full_matrices=False)
    except np.linalg.LinAlgError:
        Q, R = np.linalg.qr(Z, mode="reduced")
        diag = np.abs(np.diag(R)) if R.ndim == 2 else np.array([])
        if diag.size == 0:
            basis = np.zeros((candidate_matrix.shape[0], 0), dtype=float)
            if return_diagnostics:
                return basis, _basis_diagnostics(basis, [], None, "qr_fallback", 0)
            return basis

        tol = max(
            float(regularization) ** 0.5,
            np.finfo(float).eps * max(Z.shape) * float(diag[0]),
        )
        rank = int(np.sum(diag > tol))

        if rank <= 0:
            basis = np.zeros((candidate_matrix.shape[0], 0), dtype=float)
            if return_diagnostics:
                return basis, _basis_diagnostics(basis, diag, tol, "qr_fallback", 0)
            return basis

        basis = Q[:, :rank]
        if return_diagnostics:
            return basis, _basis_diagnostics(basis, diag, tol, "qr_fallback", rank)
        return basis

    if S.size == 0:
        basis = np.zeros((candidate_matrix.shape[0], 0), dtype=float)
        if return_diagnostics:
            return basis, _basis_diagnostics(basis, S, None, "svd", 0)
        return basis

    tol = max(
        float(regularization) ** 0.5,
        np.finfo(float).eps * max(Z.shape) * float(S[0]),
    )
    rank = int(np.sum(S > tol))

    if rank <= 0:
        basis = np.zeros((candidate_matrix.shape[0], 0), dtype=float)
        if return_diagnostics:
            return basis, _basis_diagnostics(basis, S, tol, "svd", 0)
        return basis

    basis = U[:, :rank]
    if return_diagnostics:
        return basis, _basis_diagnostics(basis, S, tol, "svd", rank)
    return basis

def _score_virtual_insertion_on_basis(
    z_matrix,
    signal,
    regularization,
    rank=None,
    condition=None,
):
    if z_matrix.shape[1] == 0:
        return 0.0, 0.0, 0, 0, 0.0

    b = z_matrix.T.dot(signal)
    gram = z_matrix.T.dot(z_matrix)
    if condition is None:
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
    if rank is None:
        rank = int(np.linalg.matrix_rank(z_matrix))
    return score, score, int(rank), int(z_matrix.shape[1]), float(condition)


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
    return _score_virtual_insertion_on_basis(z_matrix, signal, regularization)

# Leading/trailing-edge closure nodes are geometrically pinned and carry a
# degenerate surface normal. The trailing edge can also carry an adjoint
# singularity just upstream of x/c = 1, so its exclusion band is wider.
SCORING_LE_CLOSURE_NODE_EPS = 1.0e-6
SCORING_TE_CLOSURE_NODE_EPS = 5.0e-3
SCORING_CLOSURE_NODE_EPS = SCORING_LE_CLOSURE_NODE_EPS

def scoring_node_mask(
    metadata,
    eps=None,
    le_eps=SCORING_LE_CLOSURE_NODE_EPS,
    te_eps=SCORING_TE_CLOSURE_NODE_EPS,
):
    if eps is not None:
        le_eps = eps
        te_eps = eps
    mask = []
    for row in metadata:
        x_over_c = float(row["x_over_c"])
        mask.append(
            not (
                x_over_c <= float(le_eps)
                or x_over_c >= 1.0 - float(te_eps)
            )
        )
    return np.asarray(mask, dtype=bool)

def _drop_closure_nodes(
    metadata,
    signal,
    eps=None,
    le_eps=SCORING_LE_CLOSURE_NODE_EPS,
    te_eps=SCORING_TE_CLOSURE_NODE_EPS,
):
    signal = np.asarray(signal, dtype=float)
    kept_metadata = []
    kept_signal = []
    for keep, row, value in zip(
        scoring_node_mask(metadata, eps=eps, le_eps=le_eps, te_eps=te_eps),
        metadata,
        signal,
    ):
        if not bool(keep):
            continue
        kept_metadata.append(row)
        kept_signal.append(float(value))
    return kept_metadata, np.asarray(kept_signal, dtype=float)

def score_knot_spans(space, metadata, signal, settings, regularization=1.0e-12):
    knot_score_mode = str(settings.get("knot_score_mode", "VIRTUAL_INSERTION")).upper()
    diagnostics_active = scoring_pass_active(settings)
    diagnostic_state = (
        settings.get("_scoring_diagnostics_state", {})
        if diagnostics_active
        else {}
    )
    diagnostic_context = diagnostic_state.get("context", {}) if diagnostics_active else {}
    diagnostic_level = int(diagnostic_context.get("level", 0)) if diagnostics_active else 0
    diagnostic_batch_step = int(settings.get("_diagnostic_batch_step", 1) or 1)
    diagnostic_pass_id = int(
        settings.get("_diagnostic_scoring_pass_id", diagnostic_batch_step)
        or diagnostic_batch_step
    )
    diagnostic_side = str(
        settings.get("_diagnostic_side")
        or (space.sides[0].upper() if len(space.sides) == 1 else "BOTH")
    ).upper()
    mask = scoring_node_mask(metadata)
    objective_signal = settings.get("_ikkt_objective_signal")
    if objective_signal is not None:
        objective_signal = np.asarray(objective_signal, dtype=float)
        if objective_signal.shape[0] != len(mask):
            raise BSplineAdaptiveError(
                "IKKT objective signal length does not match knot-scoring metadata"
            )
        objective_signal = objective_signal[mask]
    metadata = [row for keep, row in zip(mask, metadata) if bool(keep)]
    signal = np.asarray(signal, dtype=float)[mask]
    spans = knot_insertion_spans(
        space.knot_vector,
        min_width=settings.get("knot_min_span_width", 1.0e-8),
    )
    if not spans:
        raise BSplineAdaptiveError("KNOT_INSERTION found no non-degenerate knot spans")

    old_modes, old_matrix = reduced_basis_matrix_for_space(space, space.spec, metadata)
    del old_modes
    signal = np.asarray(signal, dtype=float)
    residual = signal - project_onto_basis(old_matrix, signal, regularization=regularization)
    candidate_side = space.sides[0].upper() if len(space.sides) == 1 else "BOTH"
    if settings.get("_diagnostic_side"):
        candidate_side = diagnostic_side
    primary_signal_name = (
        "ikkt_residual"
        if knot_score_mode == "IKKT_VIRTUAL_INSERTION"
        else "objective"
    )
    score_signal_norm = float(np.linalg.norm(signal))
    objective_signal_norm = (
        float(np.linalg.norm(objective_signal))
        if objective_signal is not None
        else None
    )
    pass_payload = None
    if diagnostics_active:
        pass_payload = {
            "batch_step": diagnostic_batch_step,
            "scoring_pass_id": diagnostic_pass_id,
            "side": diagnostic_side,
            "old_basis": (
                old_basis_payload(
                    old_matrix,
                    signal,
                    objective_signal if objective_signal is not None else signal,
                    primary_signal_name,
                    regularization,
                )
                if settings.get("scoring_diagnostic_basis", True)
                else {"status": "disabled"}
            ),
        }
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
        span_mask = []
        for row in metadata:
            x_over_c = float(row["x_over_c"])
            span_mask.append(left <= x_over_c <= right if last_span else left <= x_over_c < right)
        span_mask = np.asarray(span_mask, dtype=bool)
        span_upper = np.asarray(
            [str(row.get("side", "")).strip().lower() == "upper" for row in metadata],
            dtype=bool,
        )
        span_lower = np.asarray(
            [str(row.get("side", "")).strip().lower() == "lower" for row in metadata],
            dtype=bool,
        )
        z_matrix = np.zeros((len(metadata), 0), dtype=float)
        rank_diag = {}

        if knot_score_mode == "RESIDUAL_ENERGY":
            score_raw = float(residual_energy)
            score = score_raw
            rank = 0
            columns = 0
            condition = 0.0
        elif knot_score_mode in ("VIRTUAL_INSERTION", "IKKT_VIRTUAL_INSERTION"):
            if diagnostics_active and settings.get("scoring_diagnostic_svd", True):
                z_matrix, rank_diag = _rank_incremental_columns(
                    old_matrix,
                    new_matrix,
                    regularization,
                    return_diagnostics=True,
                )
            else:
                z_matrix = _rank_incremental_columns(
                    old_matrix,
                    new_matrix,
                    regularization,
                )
            score, score_raw, rank, columns, condition = _score_virtual_insertion_on_basis(
                z_matrix,
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

        score_objective = ""
        rank_objective = ""
        objective_projection = ""
        score_objective_normalized = ""
        score_ikkt = ""
        rank_ikkt = ""
        lagrangian_projection = ""
        score_ikkt_normalized = ""
        ikkt_objective_score_ratio = ""
        candidate_projection_cosine_ikkt_objective = ""
        if knot_score_mode == "IKKT_VIRTUAL_INSERTION":
            score_ikkt = float(score)
            score_ikkt_normalized = float(score) / (score_signal_norm * score_signal_norm + 1.0e-300)
            lagrangian_projection = math.sqrt(float(score_raw)) if float(score_raw) >= 0.0 else 0.0
            if objective_signal is not None:
                obj_score, obj_raw, _obj_rank, _obj_columns, obj_condition = _score_virtual_insertion_on_basis(
                    z_matrix,
                    objective_signal,
                    regularization,
                    rank=rank,
                    condition=condition,
                )
                if not math.isfinite(obj_condition) or obj_condition > 1.0e14:
                    obj_score = 0.0
                    obj_raw = 0.0
                score_objective = float(obj_score)
                score_objective_normalized = (
                    float(obj_score)
                    / (float(objective_signal_norm) * float(objective_signal_norm) + 1.0e-300)
                    if objective_signal_norm is not None
                    else ""
                )
                objective_projection = math.sqrt(float(obj_raw)) if float(obj_raw) >= 0.0 else 0.0
                ikkt_objective_score_ratio = float(score) / (float(obj_score) + 1.0e-300)
                p_l = z_matrix.T.dot(signal)
                p_j = z_matrix.T.dot(objective_signal)
                denom = float(np.linalg.norm(p_l) * np.linalg.norm(p_j)) + 1.0e-300
                candidate_projection_cosine_ikkt_objective = float(np.dot(p_l, p_j) / denom)

        support_mask = (
            np.linalg.norm(z_matrix, axis=1) > 1.0e-12
            if z_matrix.shape[1] > 0
            else np.zeros(len(metadata), dtype=bool)
        )
        residual_norms = rank_diag.get("residual_column_norms_kept", [])
        projection_norm = math.sqrt(float(score_raw)) if float(score_raw) >= 0.0 else 0.0
        row_payload = {
            "level": diagnostic_level if diagnostics_active else "",
            "batch_step": diagnostic_batch_step if diagnostics_active else "",
            "scoring_pass_id": diagnostic_pass_id if diagnostics_active else "",
            "candidate_id": (
                candidate_id(
                    diagnostic_level,
                    diagnostic_pass_id,
                    diagnostic_batch_step,
                    candidate_side,
                    left,
                    right,
                    inserted,
                )
                if diagnostics_active
                else ""
            ),
            "primary_signal_name": primary_signal_name,
            "span_node_count": int(np.sum(span_mask)),
            "span_node_count_upper": int(np.sum(span_mask & span_upper)),
            "span_node_count_lower": int(np.sum(span_mask & span_lower)),
            "incremental_support_node_count": int(np.sum(support_mask)),
            "incremental_support_node_count_upper": int(np.sum(support_mask & span_upper)),
            "incremental_support_node_count_lower": int(np.sum(support_mask & span_lower)),
            "pre_svd_columns": rank_diag.get("candidate_columns", int(new_matrix.shape[1])),
            "rejected_columns": rank_diag.get("rejected_columns", ""),
            "residual_column_norm_min": min(residual_norms) if residual_norms else "",
            "residual_column_norm_max": max(residual_norms) if residual_norms else "",
            "residual_column_norms": residual_norms,
            "svd_tol": rank_diag.get("svd_tol", ""),
            "svd_singular_values_all": rank_diag.get("svd_singular_values_all", []),
            "svd_singular_values_kept": rank_diag.get("svd_singular_values_kept", []),
            "rank_gap": rank_diag.get("rank_gap", ""),
            "raw_incremental_condition": rank_diag.get("raw_incremental_condition", ""),
            "gram_condition_after_svd": rank_diag.get("gram_condition_after_svd", ""),
            "u_orthogonality_error": rank_diag.get("u_orthogonality_error", ""),
            "old_space_orthogonality": rank_diag.get("old_space_orthogonality", ""),
            "score_signal_norm": score_signal_norm,
            "projection_norm": projection_norm,
            "score_normalized": float(score) / (score_signal_norm * score_signal_norm + 1.0e-300),
            "score_ikkt_normalized": score_ikkt_normalized,
            "score_objective_normalized": score_objective_normalized,
            "ikkt_objective_score_ratio": ikkt_objective_score_ratio,
            "candidate_projection_cosine_ikkt_objective": candidate_projection_cosine_ikkt_objective,
        }

        rows.append(
            {
                "batch_step": diagnostic_batch_step if diagnostics_active else "",
                "scoring_pass_id": diagnostic_pass_id if diagnostics_active else "",
                "candidate_id": row_payload["candidate_id"],
                "span_left": float(left),
                "span_right": float(right),
                "span_width": float(right) - float(left),
                "inserted_knot": float(inserted),
                "side": candidate_side,
                "score_mode": knot_score_mode,
                "score": float(score),
                "score_raw": float(score_raw),
                "residual_energy": float(residual_energy),
                "score_objective": score_objective,
                "score_ikkt": score_ikkt,
                "rank_objective": rank_objective,
                "rank_ikkt": rank_ikkt,
                "objective_projection": objective_projection,
                "lagrangian_projection": lagrangian_projection,
                "incremental_rank": rank,
                "incremental_columns": columns,
                "condition_number": condition,
                "selected": False,
                "status": "ok",
                **row_payload,
            }
        )

    rows.sort(key=lambda row: (-float(row["score"]), float(row["span_left"]), float(row["span_right"])))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        if knot_score_mode == "IKKT_VIRTUAL_INSERTION":
            row["rank_ikkt"] = rank
    if knot_score_mode == "IKKT_VIRTUAL_INSERTION":
        objective_rows = [
            row for row in rows
            if row.get("score_objective", "") != ""
            and math.isfinite(float(row.get("score_objective", 0.0)))
        ]
        objective_rows.sort(
            key=lambda row: (
                -float(row["score_objective"]),
                float(row["span_left"]),
                float(row["span_right"]),
            )
        )
        for rank, row in enumerate(objective_rows, start=1):
            row["rank_objective"] = rank
    if rows and float(rows[0]["score"]) > 0.0 and math.isfinite(float(rows[0]["score"])):
        rows[0]["selected"] = True
    if diagnostics_active and pass_payload is not None:
        record_scoring_pass(settings, pass_payload)
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
