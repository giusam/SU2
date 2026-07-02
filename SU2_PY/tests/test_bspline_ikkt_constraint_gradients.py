import json
import os
from pathlib import Path

import numpy as np
import pytest

from SU2.opt.bspline_adaptive.ikkt import (
    _airfoil_area_value_and_field,
    _build_native_constraint_fields,
    _design_basis_matrix,
    build_ikkt_score_signal,
)
from SU2.opt.bspline_adaptive.mode_utils import evaluate_basis_matrix
from SU2.opt.bspline_adaptive.scoring import load_adjoint_signal, scoring_node_mask
from SU2.opt.bspline_adaptive.settings import (
    adaptive_options_from_config,
    validate_adaptive_options,
)
from SU2.opt.bspline_dot import read_metadata
from SU2.opt.bspline_driver.config_parse import parse_optimizer_config
from SU2.opt.bspline_driver.geometry_constraints import is_geometry_constraint_name
from SU2.opt.bspline_modes import load_mode_spec


FIXTURE_DIR = Path(
    "/home/giuseppe/Desktop/Codes/tesi/test/rae2822/bspline/test/ikkt_v2_probe_test"
)


def _require_fixture():
    if not FIXTURE_DIR.exists():
        pytest.skip(f"RAE2822 IKKT V2 probe fixture not found: {FIXTURE_DIR}")
    return FIXTURE_DIR


def _settings_from_fixture_config():
    config_values = parse_optimizer_config(
        "Config_IKKT_probe_all_constraints.cfg",
        warning_prefix="[TEST_IKKT]",
    )
    options = adaptive_options_from_config(config_values)
    options.update(
        {
            "ikkt_active_tol": 3.0e-3,
            "ikkt_scaling_mode": "PHYSICAL",
            "ikkt_sign_convention": "SLSQP_GE_RAW",
            "ikkt_include_geometry_constraints": True,
            "ikkt_include_aero_constraints": True,
            "ikkt_require_available_fields": True,
        }
    )
    settings = validate_adaptive_options(options)
    settings["_ikkt_eval_dir"] = "."
    with open("ikkt_constraint_values.json", "r") as fp:
        settings["_ikkt_aero_constraint_values"] = json.load(fp)
    return settings


def _constraint_by_name(diagnostics):
    return {
        item["name"]: item
        for item in diagnostics.get("included_constraints", [])
    }


def _native_constraint_canonical_name(spec):
    source = "GEOMETRY" if is_geometry_constraint_name(spec.name) else "AERO"
    return f"{source}[{str(spec.name).strip().upper()}]"


def _assert_every_native_constraint_classified(settings, diagnostics):
    categorized = []
    for bucket in (
        diagnostics.get("included_constraints", []),
        diagnostics.get("skipped_constraints", []),
        diagnostics.get("unsupported_constraints", []),
    ):
        categorized.extend(item.get("name") for item in bucket)

    for spec in settings["native_constraints"]:
        name = _native_constraint_canonical_name(spec)
        assert categorized.count(name) == 1, (
            f"{name} should appear exactly once across IKKT diagnostic categories; "
            f"all categories={categorized}"
        )


def _perturb_metadata(metadata, displacement, epsilon):
    perturbed = []
    for row, value in zip(metadata, displacement):
        copied = dict(row)
        deform_x = float(row.get("deform_dir_x", row.get("normal_x")))
        deform_y = float(row.get("deform_dir_y", row.get("normal_y")))
        copied["deformed_x"] = (
            float(row.get("deformed_x", row.get("x")))
            + float(epsilon) * float(value) * deform_x
        )
        copied["deformed_y"] = (
            float(row.get("deformed_y", row.get("y")))
            + float(epsilon) * float(value) * deform_y
        )
        perturbed.append(copied)
    return perturbed


def _stable_central_fd_area(metadata, settings, displacement):
    samples = []
    for epsilon in (1.0e-5, 1.0e-6, 1.0e-7):
        area_plus, _field_plus = _airfoil_area_value_and_field(
            _perturb_metadata(metadata, displacement, epsilon),
            settings,
        )
        area_minus, _field_minus = _airfoil_area_value_and_field(
            _perturb_metadata(metadata, displacement, -epsilon),
            settings,
        )
        fd_value = (area_plus - area_minus) / (2.0 * epsilon)
        samples.append((epsilon, area_plus, area_minus, fd_value))
    median_fd = float(np.median([item[3] for item in samples]))
    return min(samples, key=lambda item: abs(item[3] - median_fd))


def _active_design_modes(mode_spec):
    return [
        mode
        for mode in mode_spec.get("modes", [])
        if mode.get("active", True) is not False
        and mode.get("frozen", False) is not True
    ]


def _projection(field, mode_spec, metadata, settings):
    design_basis, basis_diag = _design_basis_matrix(mode_spec, metadata, settings)
    mask = scoring_node_mask(metadata)
    projected = design_basis[mask, :].T.dot(np.asarray(field, dtype=float)[mask])
    return projected, basis_diag


def _maybe_named_gradient_from_csv(filename, function_name):
    text = Path(filename).read_text().splitlines()
    if not text:
        return None
    header = [part.strip() for part in text[0].split(",")]
    lower = [part.lower() for part in header]
    candidates = [
        f"gradient_{function_name}".lower(),
        f"{function_name}_gradient".lower(),
        function_name.lower(),
    ]
    index = None
    for candidate in candidates:
        if candidate in lower:
            index = lower.index(candidate)
            break
    if index is None:
        return None
    values = []
    for line in text[1:]:
        if not line.strip():
            continue
        values.append(float(line.split(",")[index]))
    return np.asarray(values, dtype=float)


def _verbose_enabled():
    return str(os.environ.get("IKKT_FD_VERBOSE", "")).strip() == "1"


def _fmt(value):
    if value in ("inf", "-inf", "nan"):
        return str(value)
    if value is None:
        return ""
    try:
        return f"{float(value):.16e}"
    except Exception:
        return str(value)


def _print_constraint_report(diagnostics):
    if not _verbose_enabled():
        return
    print("\n[IKKT_FD_VERBOSE] Constraint classification")
    for key in ("included_constraints", "skipped_constraints", "unsupported_constraints"):
        rows = diagnostics.get(key, [])
        print(f"{key}: {len(rows)}")
        for item in rows:
            print(
                "  "
                f"name={item.get('name', '')} "
                f"source={item.get('source', '')} "
                f"function_name={item.get('function_name', '')} "
                f"c_value={_fmt(item.get('c_value'))} "
                f"field_sign={_fmt(item.get('field_sign'))} "
                f"lambda_bounds={item.get('lambda_bounds', '')} "
                f"provider={item.get('provider', '')} "
                f"field_file={item.get('field_file', '')}"
            )


def _print_area_fd_report(rows):
    if not _verbose_enabled():
        return
    print("\n[IKKT_FD_VERBOSE] AIRFOIL_AREA central finite differences")
    print(
        "mode_id,side,eps,area_plus,area_minus,fd,analytic,abs_error,rel_error,tolerance"
    )
    for row in rows:
        print(
            "{mode_id},{side},{eps},{area_plus},{area_minus},{fd},{analytic},"
            "{abs_error},{rel_error},{tolerance}".format(
                mode_id=row["mode_id"],
                side=row["side"],
                eps=_fmt(row["eps"]),
                area_plus=_fmt(row["area_plus"]),
                area_minus=_fmt(row["area_minus"]),
                fd=_fmt(row["fd"]),
                analytic=_fmt(row["analytic"]),
                abs_error=_fmt(row["abs_error"]),
                rel_error=_fmt(row["rel_error"]),
                tolerance=_fmt(row["tolerance"]),
            )
        )


def _print_projection_report(constraint, projected):
    if not _verbose_enabled():
        return
    print(
        "\n[IKKT_FD_VERBOSE] Projection "
        f"name={constraint.get('name', '')} "
        f"dimension={len(projected)} "
        f"norm={np.linalg.norm(projected):.16e} "
        f"field_sign={_fmt(constraint.get('field_sign'))} "
        f"provider={constraint.get('provider', '')} "
        f"field_file={constraint.get('field_file', '')}"
    )


def test_unified_ikkt_opt_constraint_providers_and_gradients(monkeypatch):
    fixture = _require_fixture()
    monkeypatch.chdir(fixture)

    settings = _settings_from_fixture_config()
    mode_spec = load_mode_spec("modes_current.json")
    metadata, objective_signal = load_adjoint_signal(
        "adjoint_drag/surface_sens.csv",
        "bspline_surface_metadata.csv",
    )

    _ikkt_signal, diagnostics = build_ikkt_score_signal(
        mode_spec,
        metadata,
        objective_signal,
        settings,
    )
    included = _constraint_by_name(diagnostics)
    _print_constraint_report(diagnostics)

    assert set(included) >= {
        "AERO[LIFT]",
        "AERO[MOMENT_Z]",
        "GEOMETRY[AIRFOIL_AREA]",
    }
    skipped_names = {item["name"] for item in diagnostics.get("skipped_constraints", [])}
    assert "AERO[LIFT]" not in skipped_names
    assert "AERO[MOMENT_Z]" not in skipped_names
    assert "GEOMETRY[AIRFOIL_AREA]" not in skipped_names
    assert diagnostics.get("unsupported_constraints", []) == []
    _assert_every_native_constraint_classified(settings, diagnostics)

    lift = included["AERO[LIFT]"]
    assert lift["original_operator"] == "="
    assert lift["field_sign"] == pytest.approx(1.0)
    assert lift["lambda_bounds"] == ["-inf", "inf"]
    assert lift["provider"] == "adjoint_surface_sensitivity"
    assert lift["field_file"].endswith("adjoint_lift/surface_sens.csv")

    moment = included["AERO[MOMENT_Z]"]
    assert moment["original_operator"] == "<"
    assert moment["field_sign"] == pytest.approx(-1.0)
    assert moment["lambda_bounds"] == [0.0, "inf"]
    assert moment["provider"] == "adjoint_surface_sensitivity"
    assert moment["field_file"].endswith("adjoint_momentz/surface_sens.csv")

    area = included["GEOMETRY[AIRFOIL_AREA]"]
    assert area["original_operator"] == ">"
    assert area["field_sign"] == pytest.approx(1.0)
    assert area["lambda_bounds"] == [0.0, "inf"]
    assert area["provider"] == "analytic_airfoil_area"
    assert area["current_value"] == pytest.approx(0.0778, abs=5.0e-12)
    assert area["c_value"] == pytest.approx(0.0, abs=1.0e-12)

    skipped = []
    unsupported = []
    fields = _build_native_constraint_fields(metadata, settings, skipped, unsupported)
    assert skipped == []
    assert unsupported == []
    fields_by_name = {field.name: field for field in fields}
    area_field = fields_by_name["GEOMETRY[AIRFOIL_AREA]"]
    lift_field = fields_by_name["AERO[LIFT]"]
    moment_field = fields_by_name["AERO[MOMENT_Z]"]

    nonzero_fd_derivatives = []
    failures = []
    fd_report_rows = []
    for mode in _active_design_modes(mode_spec):
        phi = evaluate_basis_matrix(mode_spec, [mode], metadata)[:, 0]
        analytic = float(np.dot(phi, area_field.raw_field))
        epsilon, area_plus, area_minus, fd_value = _stable_central_fd_area(
            metadata,
            settings,
            phi,
        )
        abs_error = abs(fd_value - analytic)
        rel_error = abs_error / max(1.0, abs(fd_value), abs(analytic))
        tolerance = 1.0e-8 + 1.0e-4 * max(1.0, abs(fd_value), abs(analytic))
        fd_report_rows.append(
            {
                "mode_id": mode.get("id"),
                "side": mode.get("side", ""),
                "eps": epsilon,
                "area_plus": area_plus,
                "area_minus": area_minus,
                "fd": fd_value,
                "analytic": analytic,
                "abs_error": abs_error,
                "rel_error": rel_error,
                "tolerance": tolerance,
            }
        )
        if abs_error > tolerance:
            failures.append(
                {
                    "mode_id": mode.get("id"),
                    "side": mode.get("side"),
                    "eps": epsilon,
                    "area_plus": area_plus,
                    "area_minus": area_minus,
                    "fd": fd_value,
                    "analytic": analytic,
                    "abs_error": abs_error,
                    "rel_error": rel_error,
                    "tolerance": tolerance,
                }
            )
        if abs(analytic) > 1.0e-10 or abs(fd_value) > 1.0e-10:
            nonzero_fd_derivatives.append(mode.get("id"))

    _print_area_fd_report(fd_report_rows)
    assert failures == []
    assert nonzero_fd_derivatives, "AIRFOIL_AREA FD check passed vacuously with all zeros"
    assert any(str(mode_id).startswith("upper_") for mode_id in nonzero_fd_derivatives)
    assert any(str(mode_id).startswith("lower_") for mode_id in nonzero_fd_derivatives)

    lift_projected, basis_diag = _projection(
        lift_field.fit_field,
        mode_spec,
        metadata,
        settings,
    )
    assert lift_projected.shape == (basis_diag["n_reduced_design"],)
    assert np.all(np.isfinite(lift_projected))
    assert np.linalg.norm(lift_projected) > 0.0
    _print_projection_report(lift, lift_projected)

    moment_projected, basis_diag = _projection(
        moment_field.fit_field,
        mode_spec,
        metadata,
        settings,
    )
    assert moment_projected.shape == (basis_diag["n_reduced_design"],)
    assert np.all(np.isfinite(moment_projected))
    assert np.linalg.norm(moment_projected) > 0.0
    _print_projection_report(moment, moment_projected)

    lift_reference = _maybe_named_gradient_from_csv("bspline_gradients.csv", "LIFT")
    if lift_reference is not None and lift_reference.shape == lift_projected.shape:
        assert lift_projected == pytest.approx(lift_reference, rel=1.0e-4, abs=1.0e-8)

    moment_reference = _maybe_named_gradient_from_csv("bspline_gradients.csv", "MOMENT_Z")
    if moment_reference is not None and moment_reference.shape == moment_projected.shape:
        assert moment_projected == pytest.approx(
            -moment_reference,
            rel=1.0e-4,
            abs=1.0e-8,
        )
