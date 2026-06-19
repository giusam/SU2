import math

import numpy as np
import pytest

from SU2.opt.bspline_def import compute_deformed_surface, read_su2_mesh
from SU2.opt.bspline_modes import (
    BSplineModeError,
    LE_SAFE_DEFAULT_POWER,
    LE_SAFE_DEFAULT_X0,
    LE_SAFE_DEFAULT_X1,
    clamped_basis_count,
    deformation_direction,
    evaluate_mode_values,
    le_safe_direction,
    validate_le_safe_direction_options,
    validate_mode_spec,
)


def _base_spec(coefficient=0.0):
    return {
        "version": 1,
        "dimension": 2,
        "marker": "airfoil",
        "chord": {"mode": "auto", "x_le": None, "x_te": None},
        "normal_displacement": True,
        "class_shape": "sqrt_x_one_minus_x",
        "normalize_basis": True,
        "normalization_mode": "max",
        "modes": [
            {
                "id": "upper_B1",
                "side": "upper",
                "basis_type": "clamped",
                "degree": 3,
                "knot_vector": [0, 0, 0, 0, 1, 1, 1, 1],
                "basis_index": 1,
                "coefficient": coefficient,
                "bounds": [-0.01, 0.01],
            },
            {
                "id": "lower_B1",
                "side": "lower",
                "basis_type": "clamped",
                "degree": 3,
                "knot_vector": [0, 0, 0, 0, 1, 1, 1, 1],
                "basis_index": 1,
                "coefficient": coefficient,
                "bounds": [-0.01, 0.01],
            },
        ],
    }


def _write_synthetic_airfoil_mesh(path):
    points = [
        (1.0, 0.0),
        (0.75, 0.06),
        (0.5, 0.08),
        (0.25, 0.05),
        (0.0, 0.0),
        (0.25, -0.05),
        (0.5, -0.08),
        (0.75, -0.06),
    ]

    lines = ["NDIME= 2\n", "NPOIN= 8\n"]
    for i, (x, y) in enumerate(points):
        lines.append(f"{x:.16g} {y:.16g} {i}\n")
    lines.extend(["NMARK= 1\n", "MARKER_TAG= airfoil\n", "MARKER_ELEMS= 8\n"])
    for i in range(len(points)):
        lines.append(f"3 {i} {(i + 1) % len(points)}\n")

    path.write_text("".join(lines))
    return path


def test_clamped_cubic_no_internal_knots_gives_4_modes():
    assert clamped_basis_count(3, [0, 0, 0, 0, 1, 1, 1, 1]) == 4


def test_clamped_cubic_with_three_internal_knots_gives_7_modes():
    knots = [0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1]
    assert clamped_basis_count(3, knots) == 7


def test_class_shaped_modes_are_zero_at_chord_endpoints():
    mode = {
        "id": "B0",
        "side": "upper",
        "basis_type": "clamped",
        "degree": 3,
        "knot_vector": [0, 0, 0, 0, 1, 1, 1, 1],
        "basis_index": 0,
        "coefficient": 0.0,
    }
    values = evaluate_mode_values(mode, [0.0, 0.5, 1.0], normalize=False)
    assert values[0] == pytest.approx(0.0)
    assert values[-1] == pytest.approx(0.0)


def test_local_basis_type_is_rejected_in_modes_validation():
    spec = _base_spec()
    spec["modes"] = [
        {
            "id": "bad_local",
            "side": "upper",
            "basis_type": "local",
            "degree": 3,
            "support": [0.25, 0.75],
            "coefficient": 0.0,
        }
    ]
    with pytest.raises(BSplineModeError, match="clamped"):
        validate_mode_spec(spec)


def test_max_normalization_scales_nonzero_mode_to_one():
    mode = {
        "id": "B1",
        "side": "upper",
        "basis_type": "clamped",
        "degree": 3,
        "knot_vector": [0, 0, 0, 0, 1, 1, 1, 1],
        "basis_index": 1,
        "coefficient": 0.0,
    }
    values = evaluate_mode_values(mode, [i / 100 for i in range(101)])
    assert max(abs(value) for value in values) == pytest.approx(1.0, abs=2.0e-4)


def test_synthetic_airfoil_marker_gets_reasonable_normals_and_positive_weights(tmp_path):
    mesh_file = _write_synthetic_airfoil_mesh(tmp_path / "airfoil.su2")
    result = compute_deformed_surface(str(mesh_file), _base_spec())

    upper = [record for record in result["records"] if record["side"] == "upper"]
    lower = [record for record in result["records"] if record["side"] == "lower"]
    assert upper
    assert lower
    assert all(record["weight"] > 0.0 for record in result["records"])
    assert max(record["normal_y"] for record in upper) > 0.8
    assert min(record["normal_y"] for record in lower) < -0.8


def test_zero_coefficients_produce_unchanged_surface_positions(tmp_path):
    mesh_file = _write_synthetic_airfoil_mesh(tmp_path / "airfoil.su2")
    mesh = read_su2_mesh(str(mesh_file))
    result = compute_deformed_surface(str(mesh_file), _base_spec(coefficient=0.0))

    for record in result["records"]:
        point = mesh["points"][record["node_id"]]
        assert record["deformed_x"] == pytest.approx(point[0])
        assert record["deformed_y"] == pytest.approx(point[1])


def test_one_nonzero_coefficient_produces_zero_displacement_at_le_te(tmp_path):
    mesh_file = _write_synthetic_airfoil_mesh(tmp_path / "airfoil.su2")
    result = compute_deformed_surface(str(mesh_file), _base_spec(coefficient=0.01))

    endpoint_records = [
        record
        for record in result["records"]
        if math.isclose(record["x_over_c"], 0.0)
        or math.isclose(record["x_over_c"], 1.0)
    ]
    assert endpoint_records
    for record in endpoint_records:
        assert record["deformed_x"] == pytest.approx(record["x"])
        assert record["deformed_y"] == pytest.approx(record["y"])


# ---------------------------------------------------------------------------
# LE-safe deformation direction
# ---------------------------------------------------------------------------


def test_le_safe_direction_is_vertical_inside_le_band():
    # x/c <= x0 → (0, +1) upper, (0, -1) lower regardless of normal.
    for x in (0.0, 0.001, 0.004999):
        vx, vy = le_safe_direction(x, "upper", 0.5, -0.1)
        assert (vx, vy) == pytest.approx((0.0, 1.0))
        vx, vy = le_safe_direction(x, "lower", 0.5, 0.1)
        assert (vx, vy) == pytest.approx((0.0, -1.0))


def test_le_safe_direction_returns_original_normal_beyond_le_band():
    # x/c >= x1 → unchanged normal.
    nx, ny = 0.3, 0.95
    for x in (0.025, 0.05, 0.9):
        vx, vy = le_safe_direction(x, "upper", nx, ny)
        assert (vx, vy) == pytest.approx((nx, ny))


def test_le_safe_direction_blends_smoothly_between_vertical_and_normal():
    nx, ny = 0.4, 0.9
    x0, x1, power = 0.005, 0.025, 1.5
    samples = np.linspace(x0, x1, 9)

    prev_upper = (-1.0, -1.0)
    for x in samples:
        t = (x - x0) / (x1 - x0)
        expected_w = (3.0 * t * t - 2.0 * t * t * t) ** power
        vx_u = expected_w * nx
        vy_u = (1.0 - expected_w) * 1.0 + expected_w * ny
        vx, vy = le_safe_direction(x, "upper", nx, ny, x0=x0, x1=x1, power=power)
        assert (vx, vy) == pytest.approx((vx_u, vy_u))
        # Blend weight must be monotonic in x and bounded by (0, 1].
        assert 0.0 <= expected_w <= 1.0
        # Direction is NOT renormalized: only the magnitude at the two
        # endpoints (pure vertical at LE, pure normal beyond LE) equals 1.
        if 0.0 < t < 1.0:
            mag = math.hypot(vx, vy)
            assert mag < 1.0 + 1.0e-12
            assert mag > 0.0
        prev_upper = (vx, vy)


def test_le_safe_direction_is_not_renormalized_in_band():
    # The blend is intentionally NOT renormalized; magnitude varies with t.
    nx, ny = 0.6, 0.8
    mag = math.hypot(nx, ny)
    for x in (0.01, 0.015, 0.02):
        vx, vy = le_safe_direction(x, "upper", nx, ny)
        m = math.hypot(vx, vy)
        # Not renormalized: magnitude should differ from mag for interior points
        # unless the blend trivially equals the normal at x1.
        if x < LE_SAFE_DEFAULT_X1:
            assert abs(m - mag) > 1.0e-6 or abs(vy - 1.0) < 1.0e-12


def test_le_safe_direction_rejects_invalid_side():
    with pytest.raises(ValueError, match="invalid side"):
        le_safe_direction(0.01, "middle", 0.0, 1.0)


def test_le_safe_direction_rejects_non_increasing_x_band():
    with pytest.raises(ValueError, match="greater than"):
        le_safe_direction(0.01, "upper", 0.0, 1.0, x0=0.02, x1=0.02)


def test_deformation_direction_defaults_to_pure_normal():
    nx, ny = 0.3, 0.95
    vx, vy = deformation_direction(0.5, "upper", nx, ny)
    assert (vx, vy) == pytest.approx((nx, ny))


def test_deformation_direction_disabled_equals_pure_normal():
    nx, ny = 0.3, 0.95
    vx, vy = deformation_direction(
        0.5, "upper", nx, ny, use_le_safe_direction=False
    )
    assert (vx, vy) == pytest.approx((nx, ny))


def test_deformation_direction_enabled_uses_le_safe_direction():
    nx, ny = 0.3, 0.95
    vx, vy = deformation_direction(
        0.0, "upper", nx, ny, use_le_safe_direction=True
    )
    assert (vx, vy) == pytest.approx((0.0, 1.0))


def test_validate_le_safe_direction_options_defaults():
    options = validate_le_safe_direction_options()
    assert options == {
        "le_safe_direction": False,
        "le_safe_x0": LE_SAFE_DEFAULT_X0,
        "le_safe_x1": LE_SAFE_DEFAULT_X1,
        "le_safe_power": LE_SAFE_DEFAULT_POWER,
    }


def test_validate_le_safe_direction_options_accepts_explicit_values():
    options = validate_le_safe_direction_options(
        le_safe_direction=True,
        le_safe_x0=0.0,
        le_safe_x1=0.05,
        le_safe_power=2.0,
    )
    assert options == {
        "le_safe_direction": True,
        "le_safe_x0": 0.0,
        "le_safe_x1": 0.05,
        "le_safe_power": 2.0,
    }


def test_validate_le_safe_direction_options_rejects_invalid_band():
    with pytest.raises(BSplineModeError, match="0 <= x0 < x1 <= 1"):
        validate_le_safe_direction_options(
            le_safe_direction=True,
            le_safe_x0=0.05,
            le_safe_x1=0.005,
        )
    with pytest.raises(BSplineModeError, match="0 <= x0 < x1 <= 1"):
        validate_le_safe_direction_options(
            le_safe_direction=True,
            le_safe_x0=-0.01,
            le_safe_x1=0.02,
        )
    with pytest.raises(BSplineModeError, match="0 <= x0 < x1 <= 1"):
        validate_le_safe_direction_options(
            le_safe_direction=True,
            le_safe_x0=0.0,
            le_safe_x1=1.5,
        )


def test_validate_le_safe_direction_options_rejects_non_positive_power():
    with pytest.raises(BSplineModeError, match="BSPLINE_LE_SAFE_POWER"):
        validate_le_safe_direction_options(
            le_safe_direction=True,
            le_safe_x0=0.0,
            le_safe_x1=0.05,
            le_safe_power=0.0,
        )


def test_legacy_basis_types_are_rejected_by_validation():
    for legacy_basis in ("asymhh_le", "asymbeta_le", "asymbeta_te"):
        spec = _base_spec()
        spec["modes"] = [
            {
                "id": f"bad_{legacy_basis}",
                "side": "upper",
                "basis_type": legacy_basis,
                "degree": 3,
                "support": [0.0, 0.05],
                "xm": 0.005,
                "p": 8.0,
                "coefficient": 0.0,
            }
        ]
        with pytest.raises(BSplineModeError, match="clamped"):
            validate_mode_spec(spec)


def test_deformed_surface_with_le_safe_direction_keeps_le_x_constant(tmp_path):
    """With LE-safe direction enabled, knot insertions near the LE must not
    move the deformed LE point upstream of x_LE."""
    mesh_file = _write_synthetic_airfoil_mesh(tmp_path / "airfoil.su2")
    spec = _base_spec(coefficient=0.01)
    result = compute_deformed_surface(
        str(mesh_file),
        spec,
        le_safe_direction=True,
        le_safe_x0=0.0,
        le_safe_x1=0.05,
        le_safe_power=1.5,
    )
    le_records = [r for r in result["records"] if math.isclose(r["x_over_c"], 0.0)]
    assert le_records
    for record in le_records:
        # With LE-safe direction, vertical displacement at the LE does not
        # translate into upstream motion in x.
        assert record["deformed_x"] == pytest.approx(record["x"])
        # Direction is exactly (0, +1) for the upper LE point and (0, -1)
        # for the lower LE point.
        assert record["deform_dir_y"] in (pytest.approx(1.0), pytest.approx(-1.0))
        assert record["deform_dir_x"] == pytest.approx(0.0)

