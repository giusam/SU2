"""Geometric constraint metrics for B-spline design variables."""

import numpy as np

from .errors import BSplineSU2DriverError, _normalized_name
from .thickness import BSplineThicknessConstraint

try:
    from SU2.io.tools import optnames_geo as _SU2_GEOMETRY_NAMES
except Exception:
    _SU2_GEOMETRY_NAMES = (
        "AIRFOIL_AREA",
        "AIRFOIL_THICKNESS",
        "AIRFOIL_CHORD",
        "AIRFOIL_LE_RADIUS",
        "AIRFOIL_TOC",
        "AIRFOIL_ALPHA",
    )


GEOMETRY_CONSTRAINT_NAMES = {
    _normalized_name(name).upper() for name in _SU2_GEOMETRY_NAMES
}
GEOMETRY_CANONICAL_NAMES = {
    _normalized_name(name).upper(): str(name).strip().upper()
    for name in _SU2_GEOMETRY_NAMES
}
SUPPORTED_BSPLINE_GEOMETRY_CONSTRAINTS = ("AIRFOIL_AREA", "AIRFOIL_THICKNESS")


def is_geometry_constraint_name(name):
    return _normalized_name(name).upper() in GEOMETRY_CONSTRAINT_NAMES


def normalize_geometry_constraint_name(name):
    normalized = _normalized_name(name).upper()
    if normalized in GEOMETRY_CANONICAL_NAMES:
        return GEOMETRY_CANONICAL_NAMES[normalized]
    for supported in SUPPORTED_BSPLINE_GEOMETRY_CONSTRAINTS:
        if _normalized_name(supported).upper() == normalized:
            return supported
    return str(name or "").strip().upper()


class BSplineAirfoilAreaMetric:
    def __init__(self, metadata, basis_matrix, mode_ids, closed=False):
        self.metadata = list(metadata)
        self.basis_matrix = np.asarray(basis_matrix, dtype=float)
        self.mode_ids = [str(mode_id) for mode_id in mode_ids]
        self.closed = bool(closed)
        if not self.closed:
            raise BSplineSU2DriverError(
                "AIRFOIL_AREA requires a closed B-spline marker"
            )
        if self.basis_matrix.ndim != 2:
            raise BSplineSU2DriverError("area basis matrix must be two-dimensional")
        if self.basis_matrix.shape[0] != len(self.metadata):
            raise BSplineSU2DriverError(
                "area basis matrix row count must match metadata rows"
            )
        if self.basis_matrix.shape[1] != len(self.mode_ids):
            raise BSplineSU2DriverError(
                "area basis matrix column count must match active modes"
            )
        self.x_base = np.asarray([float(row["x"]) for row in self.metadata], dtype=float)
        self.y_base = np.asarray([float(row["y"]) for row in self.metadata], dtype=float)
        self.deform_dir_x = np.asarray(
            [float(row.get("deform_dir_x", row["normal_x"])) for row in self.metadata],
            dtype=float,
        )
        self.deform_dir_y = np.asarray(
            [float(row.get("deform_dir_y", row["normal_y"])) for row in self.metadata],
            dtype=float,
        )

    def _deformed_arrays(self, coefficients):
        coefficients = np.asarray(coefficients, dtype=float)
        if coefficients.shape != (len(self.mode_ids),):
            raise BSplineSU2DriverError(
                f"expected {len(self.mode_ids)} area coefficient(s)"
            )
        dn = self.basis_matrix.dot(coefficients)
        x_def = self.x_base + self.deform_dir_x * dn
        y_def = self.y_base + self.deform_dir_y * dn
        dx_da = self.deform_dir_x[:, None] * self.basis_matrix
        dy_da = self.deform_dir_y[:, None] * self.basis_matrix
        return x_def, y_def, dx_da, dy_da

    def value_and_gradient(self, coefficients):
        x_def, y_def, dx_da, dy_da = self._deformed_arrays(coefficients)
        signed_area = 0.0
        signed_grad = np.zeros(len(self.mode_ids), dtype=float)
        npoint = len(x_def)
        for index in range(npoint):
            next_index = (index + 1) % npoint
            x0 = x_def[index]
            y0 = y_def[index]
            x1 = x_def[next_index]
            y1 = y_def[next_index]
            signed_area += 0.5 * (x0 * y1 - x1 * y0)
            signed_grad += 0.5 * (
                dx_da[index] * y1
                + x0 * dy_da[next_index]
                - dx_da[next_index] * y0
                - x1 * dy_da[index]
            )
        sign = -1.0 if signed_area < 0.0 else 1.0
        return abs(float(signed_area)), signed_grad * sign


class BSplineAirfoilMaxThicknessMetric:
    def __init__(
        self,
        metadata,
        basis_matrix,
        mode_ids,
        x_stations,
        fd_eps=1.0e-6,
        gradient_mode="AUTO",
        closed=False,
        marker=None,
    ):
        self.section_metric = BSplineThicknessConstraint(
            metadata,
            basis_matrix,
            mode_ids,
            reference_measure=np.zeros(len(x_stations), dtype=float),
            x_stations=x_stations,
            margin=0.0,
            domain_mode="FULL",
            symmetry_y=0.0,
            fd_eps=fd_eps,
            gradient_mode=gradient_mode,
            closed=closed,
            marker=marker,
        )

    def value_and_gradient(self, coefficients):
        values = np.asarray(
            self.section_metric.section_measure(coefficients),
            dtype=float,
        )
        if len(values) == 0:
            raise BSplineSU2DriverError(
                "AIRFOIL_THICKNESS requires at least one x station"
            )
        index = int(np.argmax(values))
        jacobian = np.asarray(
            self.section_metric.jacobian_physical(coefficients),
            dtype=float,
        )
        return float(values[index]), jacobian[index, :]
