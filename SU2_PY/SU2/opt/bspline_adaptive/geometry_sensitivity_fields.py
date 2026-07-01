"""Analytic geometry sensitivity fields for B-spline IKKT scoring."""

import numpy as np

from .errors import BSplineAdaptiveError


def _metadata_xy(metadata):
    x = np.asarray(
        [float(row.get("deformed_x", row.get("x"))) for row in metadata],
        dtype=float,
    )
    y = np.asarray(
        [float(row.get("deformed_y", row.get("y"))) for row in metadata],
        dtype=float,
    )
    return x, y


def _segments(npoint, closed=False):
    pairs = [(index, index + 1) for index in range(max(0, int(npoint) - 1))]
    if closed and int(npoint) > 2:
        pairs.append((int(npoint) - 1, 0))
    return pairs


def _station_hits(metadata, x_station, closed=False):
    x, y = _metadata_xy(metadata)
    x_station = float(x_station)
    hits = []
    tol = 1.0e-12

    for i0, i1 in _segments(len(metadata), closed=closed):
        x0, x1 = float(x[i0]), float(x[i1])
        y0, y1 = float(y[i0]), float(y[i1])
        if x_station < min(x0, x1) - tol or x_station > max(x0, x1) + tol:
            continue
        if abs(x1 - x0) <= tol:
            if abs(x_station - x0) <= tol:
                hits.append((y0, [(i0, 1.0)]))
                hits.append((y1, [(i1, 1.0)]))
            continue
        alpha = (x_station - x0) / (x1 - x0)
        if -tol <= alpha <= 1.0 + tol:
            alpha = max(0.0, min(1.0, float(alpha)))
            y_hit = (1.0 - alpha) * y0 + alpha * y1
            hits.append((float(y_hit), [(i0, 1.0 - alpha), (i1, alpha)]))
    return hits


def thickness_station_field(
    metadata,
    x_station,
    domain_mode="FULL",
    symmetry_y=0.0,
    closed=False,
):
    """Return the nodal field for the real progressive-thickness measure."""

    domain_mode = str(domain_mode or "FULL").strip().upper()
    if domain_mode not in ("FULL", "HALF_UPPER", "HALF_LOWER"):
        raise BSplineAdaptiveError(
            "thickness domain mode must be FULL, HALF_UPPER, or HALF_LOWER"
        )

    hits = _station_hits(metadata, x_station, closed=closed)
    field = np.zeros(len(metadata), dtype=float)
    if domain_mode == "FULL":
        if len(hits) < 2:
            raise BSplineAdaptiveError(
                f"Could not compute B-spline thickness field at x={float(x_station):.12g}"
            )
        upper = max(hits, key=lambda item: item[0])
        lower = min(hits, key=lambda item: item[0])
        for index, weight in upper[1]:
            field[int(index)] += float(weight)
        for index, weight in lower[1]:
            field[int(index)] -= float(weight)
        current_measure = float(upper[0]) - float(lower[0])
    elif domain_mode == "HALF_UPPER":
        if not hits:
            raise BSplineAdaptiveError(
                f"Could not compute B-spline upper half-thickness field at x={float(x_station):.12g}"
            )
        upper = max(hits, key=lambda item: item[0])
        for index, weight in upper[1]:
            field[int(index)] += float(weight)
        current_measure = float(upper[0]) - float(symmetry_y)
    else:
        if not hits:
            raise BSplineAdaptiveError(
                f"Could not compute B-spline lower half-thickness field at x={float(x_station):.12g}"
            )
        lower = min(hits, key=lambda item: item[0])
        for index, weight in lower[1]:
            field[int(index)] -= float(weight)
        current_measure = float(symmetry_y) - float(lower[0])

    return {
        "field": field,
        "current_measure": float(current_measure),
        "x": float(x_station),
        "domain_mode": domain_mode,
    }
