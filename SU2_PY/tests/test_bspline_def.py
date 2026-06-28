import math

import numpy as np
import pytest

import SU2.opt.bspline_def as bspline_def
from SU2.opt.bspline_def import (
    BSplineDefError,
    classify_sides,
    compute_deformed_surface,
    read_su2_mesh,
)
from SU2.opt.bspline_modes import (
    BSplineModeError,
    LE_SAFE_DEFAULT_POWER,
    LE_SAFE_DEFAULT_X0,
    LE_SAFE_DEFAULT_X1,
    clamped_basis_count,
    deformation_direction,
    evaluate_mode_values,
    le_safe_direction,
    normalize_deformation_direction_mode,
    normalize_surface_mode,
    active_sides_from_surface_mode,
    validate_surface_mode_against_modes,
    validate_le_safe_direction_options,
    validate_mode_spec,
    vertical_direction,
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


def _write_airfoil_mesh(path, points, closed=True):
    lines = ["NDIME= 2\n", f"NPOIN= {len(points)}\n"]
    for i, (x, y) in enumerate(points):
        lines.append(f"{x:.16g} {y:.16g} {i}\n")

    nelems = len(points) if closed else len(points) - 1
    lines.extend(["NMARK= 1\n", "MARKER_TAG= airfoil\n", f"MARKER_ELEMS= {nelems}\n"])
    for i in range(nelems):
        j = (i + 1) % len(points)
        lines.append(f"3 {i} {j}\n")

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
    # The continuous normalization maximum need not lie on this 0.01 grid.
    assert max(abs(value) for value in values) == pytest.approx(1.0, abs=2.5e-4)


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


@pytest.mark.parametrize(
    "value,expected,sides",
    [
        (None, "BOTH", ["upper", "lower"]),
        ("FULL", "BOTH", ["upper", "lower"]),
        ("HALF_UPPER", "UPPER", ["upper"]),
        ("half-lower", "LOWER", ["lower"]),
    ],
)
def test_surface_mode_normalization_and_active_sides(value, expected, sides):
    assert normalize_surface_mode(value) == expected
    assert active_sides_from_surface_mode(value) == sides


def test_surface_mode_validation_rejects_absent_active_side():
    with pytest.raises(
        BSplineModeError,
        match="BSPLINE_SURFACE_MODE=UPPER requires all active modes to have side='upper'",
    ):
        validate_surface_mode_against_modes(_base_spec(), "UPPER")


@pytest.mark.parametrize(
    "surface_mode,side,vertical_sign",
    [("UPPER", "upper", 1.0), ("LOWER", "lower", -1.0)],
)
def test_half_domain_forces_every_marker_node_to_active_side(
    tmp_path,
    surface_mode,
    side,
    vertical_sign,
):
    mesh_file = _write_synthetic_airfoil_mesh(tmp_path / "airfoil.su2")
    spec = _base_spec(coefficient=0.01)
    spec["modes"] = [mode for mode in spec["modes"] if mode["side"] == side]
    spec["surface_mode"] = surface_mode

    result = compute_deformed_surface(
        str(mesh_file),
        spec,
        surface_mode=surface_mode,
        deformation_direction_mode="VERTICAL",
    )

    assert {record["side"] for record in result["records"]} == {side}
    assert {record["surface_mode"] for record in result["records"]} == {surface_mode}
    assert all(record["deform_dir_x"] == pytest.approx(0.0) for record in result["records"])
    assert all(record["deform_dir_y"] == pytest.approx(vertical_sign) for record in result["records"])


@pytest.mark.parametrize(
    "surface_mode,side",
    [("UPPER", "upper"), ("LOWER", "lower")],
)
def test_half_mode_unchanged_does_not_call_side_classifier(
    tmp_path,
    monkeypatch,
    surface_mode,
    side,
):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("classify_sides must not be called in half mode")

    monkeypatch.setattr(bspline_def, "classify_sides", fail_if_called)
    mesh_file = _write_synthetic_airfoil_mesh(tmp_path / "airfoil.su2")
    spec = _base_spec(coefficient=0.01)
    spec["modes"] = [mode for mode in spec["modes"] if mode["side"] == side]
    spec["surface_mode"] = surface_mode

    result = compute_deformed_surface(
        str(mesh_file),
        spec,
        surface_mode=surface_mode,
    )

    assert {record["side"] for record in result["records"]} == {side}


def test_cambered_both_closed_loop_has_no_aft_sawtooth(tmp_path):
    points = [
        (1.0, 0.006),
        (0.985, 0.008),
        (0.965, 0.014),
        (0.90, 0.026),
        (0.65, 0.060),
        (0.35, 0.065),
        (0.10, 0.035),
        (0.0, 0.0),
        (0.10, -0.025),
        (0.35, -0.012),
        (0.65, 0.002),
        (0.90, 0.010),
        (0.965, 0.011),
        (0.985, 0.006),
        (1.0, 0.002),
    ]
    mesh_file = _write_airfoil_mesh(tmp_path / "rae_like.su2", points)

    result = compute_deformed_surface(
        str(mesh_file),
        _base_spec(coefficient=0.005),
        deformation_direction_mode="VERTICAL",
        surface_mode="BOTH",
    )
    records = result["records"]
    x_over_c = [record["x_over_c"] for record in records]
    sides = [record["side"] for record in records]
    i_le = min(range(len(records)), key=lambda index: x_over_c[index])
    i_te = max(range(len(records)), key=lambda index: x_over_c[index])

    for index, side in enumerate(sides):
        if index in (i_le, i_te):
            continue
        previous = sides[(index - 1) % len(sides)]
        next_side = sides[(index + 1) % len(sides)]
        assert not (previous == next_side and side != previous)

    aft_indices = [
        index
        for index, record in enumerate(records)
        if 0.95 <= record["x_over_c"] <= 0.99
    ]
    assert aft_indices
    for index in aft_indices:
        if index in (i_le, i_te):
            continue
        previous = sides[(index - 1) % len(sides)]
        next_side = sides[(index + 1) % len(sides)]
        if previous == next_side:
            assert sides[index] == previous


def test_symmetric_both_classifies_by_surface_branch(tmp_path):
    mesh_file = _write_synthetic_airfoil_mesh(tmp_path / "airfoil.su2")
    result = compute_deformed_surface(
        str(mesh_file),
        _base_spec(),
        surface_mode="BOTH",
    )

    for record in result["records"]:
        if record["y"] > 0.0:
            assert record["side"] == "upper"
        elif record["y"] < 0.0:
            assert record["side"] == "lower"


def test_side_overrides_are_applied_after_topological_classification():
    node_ids = list(range(8))
    x_over_c = [1.0, 0.75, 0.5, 0.25, 0.0, 0.25, 0.5, 0.75]
    y_values = [0.0, 0.06, 0.08, 0.05, 0.0, -0.05, -0.08, -0.06]

    baseline = classify_sides(node_ids, x_over_c, y_values, closed=True)
    overridden = classify_sides(
        node_ids,
        x_over_c,
        y_values,
        side_overrides={1: "lower"},
        closed=True,
    )

    assert baseline[1] == "upper"
    assert overridden[1] == "lower"
    assert [
        index
        for index, (before, after) in enumerate(zip(baseline, overridden))
        if before != after
    ] == [1]

    with pytest.raises(BSplineDefError, match="Invalid side override"):
        classify_sides(
            node_ids,
            x_over_c,
            y_values,
            side_overrides={"1": "middle"},
            closed=True,
        )


def test_no_isolated_flip_guard_reports_node_context():
    with pytest.raises(BSplineDefError) as exc_info:
        bspline_def._raise_on_isolated_side_islands(
            [10, 11, 12, 13, 14],
            [0.1, 0.25, 0.5, 0.75, 0.9],
            [0.02, 0.04, -0.01, 0.03, 0.02],
            ["upper", "upper", "lower", "upper", "upper"],
            set(),
            closed=False,
        )

    message = str(exc_info.value)
    assert "node 12" in message
    assert "x_over_c=0.5" in message
    assert "y=-0.01" in message
    assert "previous side='upper'" in message
    assert "current side='lower'" in message
    assert "next side='upper'" in message


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


@pytest.mark.parametrize(
    "value,legacy_le_safe,expected",
    [
        (None, False, "NORMAL"),
        (None, True, "LE_SAFE"),
        ("vertical", False, "VERTICAL"),
        ("Y", False, "VERTICAL"),
        ("le-safe", False, "LE_SAFE"),
        ("normals", False, "NORMAL"),
    ],
)
def test_normalize_deformation_direction_mode(value, legacy_le_safe, expected):
    assert (
        normalize_deformation_direction_mode(
            value,
            le_safe_direction=legacy_le_safe,
        )
        == expected
    )


def test_vertical_direction_and_deformation_direction_follow_surface_side():
    assert vertical_direction("upper") == (0.0, 1.0)
    assert vertical_direction("lower") == (0.0, -1.0)
    assert deformation_direction(
        0.5,
        "upper",
        0.8,
        0.6,
        direction_mode="VERTICAL",
    ) == (0.0, 1.0)
    assert deformation_direction(
        0.5,
        "lower",
        -0.8,
        -0.6,
        direction_mode="VERTICAL",
    ) == (0.0, -1.0)


def test_vertical_deformation_keeps_x_fixed_and_changes_y(tmp_path):
    mesh_file = _write_synthetic_airfoil_mesh(tmp_path / "airfoil.su2")
    result = compute_deformed_surface(
        str(mesh_file),
        _base_spec(coefficient=0.01),
        deformation_direction_mode="VERTICAL",
    )

    changed_y = []
    for record in result["records"]:
        expected_y = 1.0 if record["side"] == "upper" else -1.0
        assert record["deform_dir_x"] == pytest.approx(0.0)
        assert record["deform_dir_y"] == pytest.approx(expected_y)
        assert record["deformation_direction_mode"] == "VERTICAL"
        assert record["deformed_x"] == pytest.approx(record["x"])
        changed_y.append(not math.isclose(record["deformed_y"], record["y"]))

    assert any(changed_y)


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
