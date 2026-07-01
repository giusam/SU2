"""IKKT/Lagrangian residual signal construction for B-spline refinement."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

from SU2.opt.bspline_driver.reduction import (
    build_reduced_variables,
    collapse_full_jacobian,
)
from SU2.opt.thickness_constraint import (
    _as_bool as _thickness_as_bool,
    _load_or_build_reference,
    _parse_x_stations,
    _resolve_from_cfg_dir,
    _x_stations_value_is_empty,
)

from .errors import BSplineAdaptiveError, _as_bool, _as_float
from .geometry_sensitivity_fields import thickness_station_field
from .mode_utils import evaluate_basis_matrix
from .scoring import scoring_node_mask


ALLOWED_IKKT_SCALING_MODES = ("PHYSICAL", "DRIVER")
ALLOWED_IKKT_SIGN_CONVENTIONS = ("SLSQP_GE_RAW", "HH_RAW")


@dataclass
class IKKTConstraintField:
    name: str
    source: str
    sign: str
    value: float
    target: float
    gap: float
    x: float
    domain_mode: str
    raw_field: np.ndarray
    fit_field: np.ndarray
    lambda_lower: float
    lambda_upper: float
    lambda_bounds_reason: str
    gradient_scale_factor: float


def normalize_ikkt_scaling_mode(value):
    mode = str(value or "PHYSICAL").strip().upper()
    if mode not in ALLOWED_IKKT_SCALING_MODES:
        raise BSplineAdaptiveError(
            "BSPLINE_IKKT_SCALING_MODE must be PHYSICAL or DRIVER"
        )
    return mode


def normalize_ikkt_sign_convention(value):
    convention = str(value or "SLSQP_GE_RAW").strip().upper()
    if convention not in ALLOWED_IKKT_SIGN_CONVENTIONS:
        raise BSplineAdaptiveError(
            "BSPLINE_IKKT_SIGN_CONVENTION must be SLSQP_GE_RAW or HH_RAW"
        )
    return convention


def lambda_bounds_for_sign(sign, sign_convention="SLSQP_GE_RAW"):
    sign = str(sign or "").strip()
    convention = normalize_ikkt_sign_convention(sign_convention)
    if convention == "SLSQP_GE_RAW":
        if sign == ">":
            return 0.0, math.inf
        if sign == "<":
            return -math.inf, 0.0
        if sign == "=":
            return -math.inf, math.inf
    if convention == "HH_RAW":
        if sign == ">":
            return -math.inf, 0.0
        if sign == "<":
            return 0.0, math.inf
        if sign == "=":
            return -math.inf, math.inf
    raise BSplineAdaptiveError(f"unsupported IKKT constraint sign {sign!r}")


def lambda_bounds_reason_for_sign(sign, sign_convention="SLSQP_GE_RAW"):
    sign = str(sign or "").strip()
    convention = normalize_ikkt_sign_convention(sign_convention)
    if convention == "SLSQP_GE_RAW":
        if sign == ">":
            return "SLSQP_GE_RAW: c = F - target >= 0, raw gF gives lambda >= 0"
        if sign == "<":
            return "SLSQP_GE_RAW: c = target - F >= 0, raw gF gives lambda <= 0"
        if sign == "=":
            return "SLSQP_GE_RAW: equality uses free lambda"
    if convention == "HH_RAW":
        if sign == ">":
            return "HH_RAW: raw F > target uses lambda <= 0"
        if sign == "<":
            return "HH_RAW: raw F < target uses lambda >= 0"
        if sign == "=":
            return "HH_RAW: equality uses free lambda"
    raise BSplineAdaptiveError(f"unsupported IKKT constraint sign {sign!r}")


def scale_objective_field_for_ikkt(objective_field, settings):
    mode = normalize_ikkt_scaling_mode(settings.get("ikkt_scaling_mode", "PHYSICAL"))
    factor = (
        1.0
        if mode == "PHYSICAL"
        else _as_float(settings.get("opt_gradient_factor", 1.0), "OPT_GRADIENT_FACTOR")
    )
    return np.asarray(objective_field, dtype=float) * float(factor), float(factor)


def scale_constraint_field_for_ikkt(raw_field, settings, source="PROGRESSIVE_THICKNESS"):
    mode = normalize_ikkt_scaling_mode(settings.get("ikkt_scaling_mode", "PHYSICAL"))
    if mode == "PHYSICAL":
        factor = 1.0
    elif str(source).upper() == "PROGRESSIVE_THICKNESS":
        factor = 1.0
    else:
        factor = _as_float(settings.get("opt_gradient_factor", 1.0), "OPT_GRADIENT_FACTOR")
    return np.asarray(raw_field, dtype=float) * float(factor), float(factor)


def estimate_ikkt_multipliers(
    objective_field,
    constraint_fields,
    design_basis_matrix,
    lambda_bounds,
):
    objective_field = np.asarray(objective_field, dtype=float)
    constraint_fields = [np.asarray(field, dtype=float) for field in constraint_fields]
    design_basis_matrix = np.asarray(design_basis_matrix, dtype=float)

    if len(constraint_fields) == 0:
        return np.zeros(0, dtype=float), {"status": "no_active_constraints"}
    if design_basis_matrix.ndim != 2:
        raise BSplineAdaptiveError("IKKT design basis matrix must be two-dimensional")
    if design_basis_matrix.shape[0] != objective_field.shape[0]:
        raise BSplineAdaptiveError("IKKT design basis row count does not match objective field")
    for field in constraint_fields:
        if field.shape != objective_field.shape:
            raise BSplineAdaptiveError("IKKT constraint field length mismatch")
    if design_basis_matrix.shape[1] == 0:
        return np.zeros(len(constraint_fields), dtype=float), {"status": "empty_design_basis"}

    grad_objective = design_basis_matrix.T.dot(objective_field)
    grad_constraints = [design_basis_matrix.T.dot(field) for field in constraint_fields]
    A = np.column_stack(grad_constraints)
    lower = np.asarray([float(bounds[0]) for bounds in lambda_bounds], dtype=float)
    upper = np.asarray([float(bounds[1]) for bounds in lambda_bounds], dtype=float)
    try:
        result = lsq_linear(A, grad_objective, bounds=(lower, upper), lsmr_tol="auto")
    except Exception as exc:
        raise BSplineAdaptiveError(f"IKKT multiplier least-squares solve failed: {exc}") from exc
    if not bool(result.success):
        raise BSplineAdaptiveError(
            f"IKKT multiplier least-squares solve failed: {result.message}"
        )
    return np.asarray(result.x, dtype=float), {
        "status": "ok",
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "active_mask": [int(value) for value in np.asarray(result.active_mask).tolist()],
        "grad_objective_norm": float(np.linalg.norm(grad_objective)),
        "grad_constraint_norms": [float(np.linalg.norm(item)) for item in grad_constraints],
    }


def build_ikkt_residual_field(objective_field, constraint_fields, lambdas):
    residual = np.asarray(objective_field, dtype=float).copy()
    for lam, field in zip(np.asarray(lambdas, dtype=float), constraint_fields):
        residual -= float(lam) * np.asarray(field, dtype=float)
    return residual


def _mode_design_counts(mode_spec):
    geometric = [
        mode for mode in mode_spec.get("modes", [])
        if mode.get("active", True) is not False
    ]
    design = [
        mode for mode in geometric
        if mode.get("frozen", False) is not True
    ]
    frozen = [
        mode for mode in geometric
        if mode.get("frozen", False) is True
    ]
    return geometric, design, frozen


def _design_basis_matrix(mode_spec, metadata, settings):
    _geometric_modes, design_modes, frozen_modes = _mode_design_counts(mode_spec)
    full = evaluate_basis_matrix(mode_spec, design_modes, metadata)
    coupling = str(settings.get("symmetry_coupling", "NONE") or "NONE").strip().upper()
    reduced_variables, warnings = build_reduced_variables(mode_spec, coupling)
    if warnings and coupling != "NONE":
        raise BSplineAdaptiveError(
            "IKKT_VIRTUAL_INSERTION cannot use inconsistent symmetry reduction: "
            + "; ".join(warnings)
        )
    reduced = collapse_full_jacobian(full, reduced_variables)
    return reduced, {
        "n_design": int(len(design_modes)),
        "n_frozen": int(len(frozen_modes)),
        "n_reduced_design": int(reduced.shape[1]),
        "symmetry_coupling": coupling,
        "design_mode_ids": [str(mode["id"]) for mode in design_modes],
        "reduced_variable_ids": [str(variable.id) for variable in reduced_variables],
    }


def _x_stations_from_options(options):
    x_stations_value = options.get("PROGRESSIVE_THICKNESS_X_STATIONS")
    if not _x_stations_value_is_empty(x_stations_value):
        return _parse_x_stations(x_stations_value)
    npoints = int(options.get("PROGRESSIVE_THICKNESS_NPOINTS", 101))
    xmin = float(options.get("PROGRESSIVE_THICKNESS_XMIN", 0.001))
    xmax = float(options.get("PROGRESSIVE_THICKNESS_XMAX", 0.999))
    if npoints < 2:
        raise BSplineAdaptiveError("PROGRESSIVE_THICKNESS_NPOINTS must be >= 2")
    if not xmin < xmax:
        raise BSplineAdaptiveError("PROGRESSIVE_THICKNESS_XMIN must be < XMAX")
    return np.linspace(xmin, xmax, npoints)


def _thickness_reference(options, settings, x_stations, domain_mode, symmetry_y, skipped):
    ref_mesh_value = options.get("PROGRESSIVE_THICKNESS_REF_MESH")
    if not ref_mesh_value:
        skipped.append(
            {
                "name": "PROGRESSIVE_THICKNESS",
                "reason": "reference_mesh_unavailable",
            }
        )
        return None
    cfg = type("_IKKTThicknessConfig", (), {})()
    if options.get("_optimizer_config_filename"):
        cfg._filename = str(options["_optimizer_config_filename"])
    elif settings.get("_case_config"):
        cfg._filename = str(settings["_case_config"])
    ref_mesh = _resolve_from_cfg_dir(cfg, ref_mesh_value)
    marker = str(options.get("PROGRESSIVE_THICKNESS_MARKER", settings.get("marker", "")))
    cache_value = options.get("PROGRESSIVE_THICKNESS_CACHE_FILE", "thickness_reference.npz")
    cache_file = _resolve_from_cfg_dir(cfg, cache_value) if cache_value else None
    try:
        return _load_or_build_reference(
            ref_mesh,
            marker,
            x_stations,
            cache_file,
            domain_mode,
            symmetry_y,
        )
    except Exception as exc:
        skipped.append(
            {
                "name": "PROGRESSIVE_THICKNESS",
                "reason": f"reference_measure_unavailable:{exc}",
            }
        )
        return None


def _build_progressive_thickness_fields(metadata, settings, skipped):
    options = dict(settings.get("thickness_options") or {})
    enabled = _thickness_as_bool(
        options.get("PROGRESSIVE_THICKNESS_CONSTRAINT", "NO"),
        default=False,
    )
    if not enabled:
        return []
    domain_mode = str(options.get("PROGRESSIVE_THICKNESS_DOMAIN_MODE", "FULL")).upper()
    symmetry_y = float(options.get("PROGRESSIVE_THICKNESS_SYMMETRY_Y", 0.0))
    margin = float(options.get("PROGRESSIVE_THICKNESS_MARGIN", 0.0))
    x_stations = _x_stations_from_options(options)
    reference = _thickness_reference(
        options,
        settings,
        x_stations,
        domain_mode,
        symmetry_y,
        skipped,
    )
    if reference is None:
        return []
    reference = np.asarray(reference, dtype=float)
    if len(reference) != len(x_stations):
        raise BSplineAdaptiveError(
            "IKKT progressive thickness reference count does not match x stations"
        )

    active_tol = _as_float(
        settings.get("ikkt_geom_thickness_active_tol", 1.0e-4),
        "BSPLINE_IKKT_GEOM_THICKNESS_ACTIVE_TOL",
    )
    sign_convention = normalize_ikkt_sign_convention(
        settings.get("ikkt_sign_convention", "SLSQP_GE_RAW")
    )
    fields = []
    closed = domain_mode == "FULL"
    for index, (x_station, ref_value) in enumerate(zip(x_stations, reference)):
        item = thickness_station_field(
            metadata,
            x_station,
            domain_mode=domain_mode,
            symmetry_y=symmetry_y,
            closed=closed,
        )
        target = float(ref_value) + float(margin)
        value = float(item["current_measure"])
        gap = value - target
        if gap > active_tol:
            continue
        lower, upper = lambda_bounds_for_sign(">", sign_convention)
        fit_field, gradient_scale = scale_constraint_field_for_ikkt(
            item["field"],
            settings,
            source="PROGRESSIVE_THICKNESS",
        )
        fields.append(
            IKKTConstraintField(
                name=f"PROGRESSIVE_THICKNESS[{index}]",
                source="PROGRESSIVE_THICKNESS",
                sign=">",
                value=value,
                target=target,
                gap=float(gap),
                x=float(x_station),
                domain_mode=domain_mode,
                raw_field=np.asarray(item["field"], dtype=float),
                fit_field=fit_field,
                lambda_lower=lower,
                lambda_upper=upper,
                lambda_bounds_reason=lambda_bounds_reason_for_sign(">", sign_convention),
                gradient_scale_factor=gradient_scale,
            )
        )
    return fields


def _validate_ikkt_mode(settings, metadata):
    direction = str(settings.get("deformation_direction_mode", "") or "").strip().upper()
    if not direction and metadata:
        direction = str(metadata[0].get("deformation_direction_mode", "")).strip().upper()
    if direction != "VERTICAL":
        raise BSplineAdaptiveError(
            "BSPLINE_KNOT_SCORE_MODE=IKKT_VIRTUAL_INSERTION requires "
            "BSPLINE_DEFORMATION_DIRECTION=VERTICAL"
        )


def _lambda_bound_value(value):
    if math.isinf(float(value)):
        return "inf" if float(value) > 0.0 else "-inf"
    return float(value)


def _constraint_dict(field, lam=None):
    return {
        "name": field.name,
        "source": field.source,
        "sign": field.sign,
        "inequality_representation": "c = F - target >= 0",
        "value": float(field.value),
        "target": float(field.target),
        "gap": float(field.gap),
        "x": float(field.x),
        "domain_mode": field.domain_mode,
        "lambda": None if lam is None else float(lam),
        "lambda_bounds": [
            _lambda_bound_value(field.lambda_lower),
            _lambda_bound_value(field.lambda_upper),
        ],
        "lambda_bounds_reason": field.lambda_bounds_reason,
        "gradient_norm": float(np.linalg.norm(field.fit_field)),
        "gradient_scale_factor": float(field.gradient_scale_factor),
    }


def _node_mask_diagnostics(mask):
    mask = np.asarray(mask, dtype=bool)
    included = [int(index) for index, keep in enumerate(mask) if bool(keep)]
    excluded = [int(index) for index, keep in enumerate(mask) if not bool(keep)]
    return {
        "total": int(len(mask)),
        "included": int(len(included)),
        "excluded": int(len(excluded)),
        "included_indices": included,
        "excluded_indices": excluded,
    }


def _diagnostics_base(
    status,
    constraints,
    unavailable,
    objective_fit,
    residual,
    mask,
    basis_diag,
    settings,
    objective_scale,
    lsq_diag=None,
    lambdas=None,
):
    objective_norm = float(np.linalg.norm(objective_fit))
    residual_norm = float(np.linalg.norm(residual))
    scaling_mode = normalize_ikkt_scaling_mode(settings.get("ikkt_scaling_mode", "PHYSICAL"))
    sign_convention = normalize_ikkt_sign_convention(
        settings.get("ikkt_sign_convention", "SLSQP_GE_RAW")
    )
    return {
        "score_mode": "IKKT_VIRTUAL_INSERTION",
        "status": status,
        "scaling_policy": {
            "scaling_mode": scaling_mode,
            "objective_scale_factor": float(objective_scale),
            "constraint_scale_uses_opt_constraint_scale": False,
        },
        "sign_convention": sign_convention,
        "included_constraints": [
            _constraint_dict(field, lam)
            for field, lam in zip(constraints, [] if lambdas is None else lambdas)
        ],
        "skipped_constraints": unavailable,
        "objective_norm": objective_norm,
        "lagrangian_residual_norm": residual_norm,
        "relative_lagrangian_residual_norm": (
            residual_norm / objective_norm if objective_norm > 0.0 else 0.0
        ),
        "least_squares": {} if lsq_diag is None else lsq_diag,
        "node_mask": _node_mask_diagnostics(mask),
        **basis_diag,
    }


def build_ikkt_score_signal(mode_spec, metadata, objective_signal, settings):
    _validate_ikkt_mode(settings, metadata)
    include_geometry = _as_bool(
        settings.get("ikkt_include_geometry_constraints", True),
        default=True,
    )
    include_aero = _as_bool(
        settings.get("ikkt_include_aero_constraints", False),
        default=False,
    )
    require_available = _as_bool(
        settings.get("ikkt_require_available_fields", True),
        default=True,
    )

    skipped = []
    unavailable = []
    constraints = []
    if include_geometry:
        constraints.extend(_build_progressive_thickness_fields(metadata, settings, skipped))
    if include_aero:
        unavailable.append(
            {
                "name": "AERO_CONSTRAINTS",
                "reason": "aero_constraint_fields_unavailable_v1",
            }
        )
    unavailable.extend(skipped)
    if require_available and unavailable:
        raise BSplineAdaptiveError(
            "IKKT_VIRTUAL_INSERTION could not build required constraint fields: "
            + "; ".join(f"{item['name']}:{item['reason']}" for item in unavailable)
        )

    objective_fit, objective_scale = scale_objective_field_for_ikkt(objective_signal, settings)
    mask = scoring_node_mask(metadata)
    design_basis, basis_diag = _design_basis_matrix(mode_spec, metadata, settings)
    if design_basis.shape[0] != len(objective_fit):
        raise BSplineAdaptiveError("IKKT design basis length mismatch")

    if not constraints:
        diagnostics = _diagnostics_base(
            "objective_only_no_active_constraints",
            [],
            unavailable,
            objective_fit,
            objective_fit,
            mask,
            basis_diag,
            settings,
            objective_scale,
        )
        return objective_fit, diagnostics

    lambdas, lsq_diag = estimate_ikkt_multipliers(
        objective_fit[mask],
        [field.fit_field[mask] for field in constraints],
        design_basis[mask, :],
        [(field.lambda_lower, field.lambda_upper) for field in constraints],
    )
    residual = build_ikkt_residual_field(
        objective_fit,
        [field.fit_field for field in constraints],
        lambdas,
    )
    diagnostics = _diagnostics_base(
        "ok",
        constraints,
        unavailable,
        objective_fit,
        residual,
        mask,
        basis_diag,
        settings,
        objective_scale,
        lsq_diag=lsq_diag,
        lambdas=lambdas,
    )
    return residual, diagnostics


def write_ikkt_diagnostics(filename, diagnostics):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fp:
        json.dump(diagnostics, fp, indent=2)
        fp.write("\n")
