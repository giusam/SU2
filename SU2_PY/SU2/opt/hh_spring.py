#!/usr/bin/env python

import math
import numpy as np


def _normalize_scores(scores):
    """
    Normalize scores into [0, 1] as in the paper:
        f_i = (a_i - a_min) / (a_max - a_min)
    """
    scores = [float(s) for s in scores]

    if not scores:
        return []

    a_min = min(scores)
    a_max = max(scores)

    if abs(a_max - a_min) < 1.0e-14:
        return [0.0 for _ in scores]

    return [(a - a_min) / (a_max - a_min) for a in scores]


def _paper_F(f):
    """
    Paper choice:
        F(f) = min(1.0, 2*sin(f))
    assuming f in [0, 1].
    """
    return min(1.0, 2.0 * math.sin(f))


def spring_redistribute_centers(centers, scores, A=20.0):
    """
    Redistribute 1D HH centers with the spring analogy.

    Governing equations:
        K_{i+1/2}(x_{i+1} - x_i) = K_{i-1/2}(x_i - x_{i-1})

    with stiffness:
        K_{i+1/2} = 1 + (A - 1) * F( (f_i + f_{i+1}) / 2 )

    where:
        f_i = (a_i - a_min) / (a_max - a_min)
        F(f) = min(1.0, 2*sin(f))

    Endpoints are kept fixed.
    """

    centers = [float(x) for x in centers]
    scores = [float(s) for s in scores]

    if len(centers) != len(scores):
        raise ValueError(
            f"centers/scores size mismatch: {len(centers)} vs {len(scores)}"
        )

    if len(centers) <= 2:
        return sorted(centers)

    # sort by x
    pairs = sorted(zip(centers, scores), key=lambda p: p[0])
    centers = [p[0] for p in pairs]
    scores = [p[1] for p in pairs]

    # normalized nodal indicators f_i
    f = _normalize_scores(scores)

    # spring stiffnesses K_{i+1/2}, one for each interval
    K = []
    for i in range(len(centers) - 1):
        f_avg = 0.5 * (f[i] + f[i + 1])
        k = 1.0 + (A - 1.0) * _paper_F(f_avg)
        K.append(k)

    N = len(centers)

    # Unknowns are internal nodes x_1 ... x_{N-2}
    A_mat = np.zeros((N - 2, N - 2))
    b_vec = np.zeros(N - 2)

    for i in range(1, N - 1):
        row = i - 1

        K_imh = K[i - 1]  # K_{i-1/2}
        K_iph = K[i]      # K_{i+1/2}

        # Equation:
        # -K_{i-1/2} x_{i-1} + (K_{i-1/2}+K_{i+1/2}) x_i - K_{i+1/2} x_{i+1} = 0

        if row > 0:
            A_mat[row, row - 1] = -K_imh

        A_mat[row, row] = K_imh + K_iph

        if row < N - 3:
            A_mat[row, row + 1] = -K_iph

        # boundary contributions
        if i == 1:
            b_vec[row] += K_imh * centers[0]

        if i == N - 2:
            b_vec[row] += K_iph * centers[-1]

    x_internal = np.linalg.solve(A_mat, b_vec)

    new_centers = [centers[0]] + list(x_internal) + [centers[-1]]

    # Clamp to open interval (0,1) and preserve ordering
    eps = 1.0e-6
    new_centers = [max(eps, min(1.0 - eps, x)) for x in new_centers]
    new_centers = sorted(new_centers)

    return new_centers


def apply_hh_spring_after_selection(
    prev_level,
    chosen,
    active_upper_scores,
    active_lower_scores,
    opts,
):
    """
    Apply spring analogy after candidate selection.

    Inputs:
      - prev_level.upper / prev_level.lower: existing active HH centers
      - chosen: selected candidate dicts with keys side, x, indicator
      - active_upper_scores / active_lower_scores: scores of existing active HH centers
      - opts["spring_A"]: stiffness ratio parameter A
    """

    A = float(opts.get("spring_A", 20.0))

    chosen_upper = [c for c in chosen if c["side"] == "UPPER"]
    chosen_lower = [c for c in chosen if c["side"] == "LOWER"]

    centers_upper = list(prev_level.upper) + [float(c["x"]) for c in chosen_upper]
    centers_lower = list(prev_level.lower) + [float(c["x"]) for c in chosen_lower]

    scores_upper = list(active_upper_scores) + [float(c["indicator"]) for c in chosen_upper]
    scores_lower = list(active_lower_scores) + [float(c["indicator"]) for c in chosen_lower]

    new_upper = spring_redistribute_centers(centers_upper, scores_upper, A=A)
    new_lower = spring_redistribute_centers(centers_lower, scores_lower, A=A)

    return new_upper, new_lower


def apply_hh_spring_after_selection_symmetric(
    prev_level,
    chosen,
    active_pair_scores,
    opts,
):
    A = float(opts.get("spring_A", 20.0))

    chosen_pair = [c for c in chosen if c["side"] == "PAIR"]
    centers = list(prev_level.upper) + [float(c["x"]) for c in chosen_pair]
    scores = list(active_pair_scores) + [
        float(c["indicator"]) for c in chosen_pair
    ]

    new_pair = spring_redistribute_centers(centers, scores, A=A)
    return new_pair, list(new_pair)
