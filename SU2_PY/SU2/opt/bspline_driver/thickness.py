"""Thickness constraint for the B-spline/SU2 driver."""

import numpy as np

from SU2.opt.thickness_constraint import _normalize_gradient_mode
from .errors import BSplineSU2DriverError


class BSplineThicknessConstraint:
    def __init__(
        self,
        metadata,
        basis_matrix,
        mode_ids,
        reference_measure,
        x_stations,
        margin=0.0,
        domain_mode="FULL",
        symmetry_y=0.0,
        fd_eps=1.0e-6,
        gradient_mode="AUTO",
        closed=False,
        marker=None,
    ):
        self.metadata = list(metadata)
        self.basis_matrix = np.asarray(basis_matrix, dtype=float)
        self.mode_ids = [str(mode_id) for mode_id in mode_ids]
        self.reference_measure = np.asarray(reference_measure, dtype=float)
        self.reference_thickness = self.reference_measure
        self.x_stations = np.asarray(x_stations, dtype=float)
        self.margin = float(margin)
        self.domain_mode = str(domain_mode).upper()
        self.symmetry_y = float(symmetry_y)
        self.fd_eps = float(fd_eps)
        self.gradient_mode = _normalize_gradient_mode(gradient_mode)
        self.closed = bool(closed)
        self.marker = None if marker is None else str(marker)
        self._fallback_warned = False
        self._switch_warned = False

        if self.domain_mode not in ("FULL", "HALF_UPPER", "HALF_LOWER"):
            raise BSplineSU2DriverError(
                "PROGRESSIVE_THICKNESS_DOMAIN_MODE must be FULL, HALF_UPPER, or HALF_LOWER"
            )
        if self.basis_matrix.ndim != 2:
            raise BSplineSU2DriverError("thickness basis matrix must be two-dimensional")
        if self.basis_matrix.shape[0] != len(self.metadata):
            raise BSplineSU2DriverError(
                "thickness basis matrix row count must match metadata rows"
            )
        if self.basis_matrix.shape[1] != len(self.mode_ids):
            raise BSplineSU2DriverError(
                "thickness basis matrix column count must match active modes"
            )
        if len(self.reference_measure) != len(self.x_stations):
            raise BSplineSU2DriverError(
                "reference thickness count must match x station count"
            )

        self.x_base = np.asarray([float(row["x"]) for row in self.metadata], dtype=float)
        self.y_base = np.asarray([float(row["y"]) for row in self.metadata], dtype=float)
        # Reconstruct the geometry with the effective NORMAL, LE_SAFE, or
        # VERTICAL direction recorded by bspline_def.
        self.deform_dir_x = np.asarray(
            [float(row.get("deform_dir_x", row["normal_x"])) for row in self.metadata],
            dtype=float,
        )
        self.deform_dir_y = np.asarray(
            [float(row.get("deform_dir_y", row["normal_y"])) for row in self.metadata],
            dtype=float,
        )
        self._segments = [(index, index + 1) for index in range(len(self.metadata) - 1)]
        if self.closed and len(self.metadata) > 2:
            self._segments.append((len(self.metadata) - 1, 0))

    def _deformed_arrays(self, coefficients):
        coefficients = np.asarray(coefficients, dtype=float)
        if coefficients.shape != (len(self.mode_ids),):
            raise BSplineSU2DriverError(
                f"expected {len(self.mode_ids)} thickness coefficient(s)"
            )
        dn = self.basis_matrix.dot(coefficients)
        x_def = self.x_base + self.deform_dir_x * dn
        y_def = self.y_base + self.deform_dir_y * dn
        dx_da = self.deform_dir_x[:, None] * self.basis_matrix
        dy_da = self.deform_dir_y[:, None] * self.basis_matrix
        return x_def, y_def, dx_da, dy_da

    def _station_hits(self, x_station, x_def, y_def, dx_da=None, dy_da=None):
        hits = []
        tol = 1.0e-12
        x_station = float(x_station)

        for i0, i1 in self._segments:
            x0, x1 = float(x_def[i0]), float(x_def[i1])
            y0, y1 = float(y_def[i0]), float(y_def[i1])
            if x_station < min(x0, x1) - tol or x_station > max(x0, x1) + tol:
                continue

            if abs(x1 - x0) <= tol:
                if abs(x_station - x0) <= tol:
                    if dy_da is None:
                        hits.append((y0, None))
                        hits.append((y1, None))
                    else:
                        hits.append((y0, np.asarray(dy_da[i0], dtype=float)))
                        hits.append((y1, np.asarray(dy_da[i1], dtype=float)))
                continue

            t = (x_station - x0) / (x1 - x0)
            if -tol <= t <= 1.0 + tol:
                t = max(0.0, min(1.0, float(t)))
                y_hit = y0 + t * (y1 - y0)
                if dx_da is None or dy_da is None:
                    hits.append((y_hit, None))
                    continue

                dx0 = np.asarray(dx_da[i0], dtype=float)
                dx1 = np.asarray(dx_da[i1], dtype=float)
                dy0 = np.asarray(dy_da[i0], dtype=float)
                dy1 = np.asarray(dy_da[i1], dtype=float)
                dt_da = (((t - 1.0) * dx0) - t * dx1) / (x1 - x0)
                dy_hit_da = (1.0 - t) * dy0 + t * dy1 + (y1 - y0) * dt_da
                hits.append((y_hit, dy_hit_da))

        return hits

    def section_measure(self, coefficients):
        x_def, y_def, _dx_da, _dy_da = self._deformed_arrays(coefficients)
        values = []
        for x_station in self.x_stations:
            hits = self._station_hits(x_station, x_def, y_def)
            if self.domain_mode == "FULL":
                if len(hits) < 2:
                    raise BSplineSU2DriverError(
                        f"Could not compute B-spline thickness at x={float(x_station):.12g}"
                    )
                y_values = [hit[0] for hit in hits]
                values.append(max(y_values) - min(y_values))
            elif self.domain_mode == "HALF_UPPER":
                if not hits:
                    raise BSplineSU2DriverError(
                        f"Could not compute B-spline upper half-thickness at x={float(x_station):.12g}"
                    )
                values.append(max(hit[0] for hit in hits) - self.symmetry_y)
            else:
                if not hits:
                    raise BSplineSU2DriverError(
                        f"Could not compute B-spline lower half-thickness at x={float(x_station):.12g}"
                    )
                values.append(self.symmetry_y - min(hit[0] for hit in hits))
        return np.asarray(values, dtype=float)

    def values(self, coefficients):
        current = self.section_measure(coefficients)
        return current - self.reference_measure - self.margin

    def jacobian_analytic(self, coefficients):
        x_def, y_def, dx_da, dy_da = self._deformed_arrays(coefficients)
        jac = np.zeros((len(self.x_stations), len(self.mode_ids)), dtype=float)
        # Switch margin: if the upper/lower pair at a station are closer than
        # this in y, the (upper, lower) identity is unstable under small
        # perturbations of the coefficients and the analytic gradient below
        # is only a subgradient. Warn once per driver so the user can decide
        # whether to use the FD fallback or to add safety margin.
        switch_margin = 1.0e-6

        for i_x, x_station in enumerate(self.x_stations):
            hits = self._station_hits(x_station, x_def, y_def, dx_da=dx_da, dy_da=dy_da)
            if self.domain_mode == "FULL":
                if len(hits) < 2:
                    raise BSplineSU2DriverError(
                        f"Could not compute B-spline thickness gradient at x={float(x_station):.12g}"
                    )
                upper = max(hits, key=lambda item: item[0])
                lower = min(hits, key=lambda item: item[0])
                if (
                    not self._switch_warned
                    and abs(upper[0] - lower[0]) < switch_margin
                ):
                    print(
                        "[BSPLINE_SU2_DRIVER] WARNING: analytic B-spline thickness gradient "
                        f"is near a max/min switch at x={float(x_station):.12g} "
                        f"(|y_u - y_l| = {abs(upper[0] - lower[0]):.3e}); the gradient "
                        "is only a subgradient. Consider enabling PROGRESSIVE_THICKNESS_GRADIENT=FINITE_DIFFERENCE "
                        "or increasing the safety margin."
                    )
                    self._switch_warned = True
                jac[i_x, :] = upper[1] - lower[1]
            elif self.domain_mode == "HALF_UPPER":
                if not hits:
                    raise BSplineSU2DriverError(
                        f"Could not compute B-spline upper half-thickness gradient at x={float(x_station):.12g}"
                    )
                upper = max(hits, key=lambda item: item[0])
                jac[i_x, :] = upper[1]
            else:
                if not hits:
                    raise BSplineSU2DriverError(
                        f"Could not compute B-spline lower half-thickness gradient at x={float(x_station):.12g}"
                    )
                lower = min(hits, key=lambda item: item[0])
                jac[i_x, :] = -lower[1]
        return jac

    def jacobian_fd_physical(self, coefficients):
        coefficients = np.asarray(coefficients, dtype=float)
        g0 = np.asarray(self.values(coefficients), dtype=float)
        jac = np.zeros((len(g0), len(coefficients)), dtype=float)
        for j in range(len(coefficients)):
            trial = coefficients.copy()
            trial[j] += self.fd_eps
            jac[:, j] = (np.asarray(self.values(trial), dtype=float) - g0) / self.fd_eps
        return jac

    def jacobian_physical(self, coefficients):
        if self.gradient_mode == "FINITE_DIFFERENCE":
            return self.jacobian_fd_physical(coefficients)
        try:
            return self.jacobian_analytic(coefficients)
        except Exception as exc:
            if self.gradient_mode == "ANALYTIC":
                raise
            if not self._fallback_warned:
                print(
                    "[BSPLINE_SU2_DRIVER] WARNING: analytic B-spline thickness gradient unavailable; "
                    f"falling back to finite differences ({exc})"
                )
                self._fallback_warned = True
            return self.jacobian_fd_physical(coefficients)
