import pytest

from SU2.opt.progressive_ffd_blending import (
    BSPLINE_UNIFORM,
    basis_values,
    evaluate_curve,
    invert_monotone_curve,
    make_blending_spec,
    open_uniform_knot_vector,
    parse_bspline_orders,
    validate_blending_spec,
)


def test_open_uniform_knots_match_su2_construction():
    assert open_uniform_knot_vector(5, 3) == pytest.approx(
        [0.0, 0.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0, 1.0, 1.0]
    )


def test_uniform_bspline_partition_unity_endpoints_and_local_support():
    spec = make_blending_spec(BSPLINE_UNIFORM, (4, 2, 2))
    for t in (0.0, 0.03, 0.25, 0.5, 0.91, 1.0):
        values = basis_values(9, t, spec, axis=0)
        assert sum(values) == pytest.approx(1.0, abs=1.0e-14)
        assert sum(abs(value) > 1.0e-14 for value in values) <= 4
    assert basis_values(9, 0.0, spec, axis=0)[0] == pytest.approx(1.0)
    assert basis_values(9, 1.0, spec, axis=0)[-1] == pytest.approx(1.0)


def test_uniform_bspline_curve_inversion_round_trip():
    spec = make_blending_spec(BSPLINE_UNIFORM, (4, 2, 2))
    controls = [0.0, 0.03, 0.14, 0.28, 0.48, 0.68, 0.83, 0.94, 1.0]
    for u in (0.0, 0.07, 0.31, 0.67, 0.93, 1.0):
        x = evaluate_curve(controls, u, spec, axis=0)
        recovered = invert_monotone_curve(controls, x, spec, axis=0)
        assert recovered == pytest.approx(u, abs=1.0e-12)


def test_bspline_order_parser_and_dual_validation():
    assert parse_bspline_orders("( 4, 2, 2 )") == (4, 2, 2)
    spec = make_blending_spec(BSPLINE_UNIFORM, (4, 2, 2))
    validate_blending_spec(spec, control_counts=(9, 2, 2), dual_2d=True)
    with pytest.raises(ValueError, match="exceeds control-point count"):
        validate_blending_spec(spec, control_counts=(3, 2, 2), dual_2d=True)
    with pytest.raises(ValueError, match="requires FFD_BSPLINE_ORDER"):
        validate_blending_spec(
            make_blending_spec(BSPLINE_UNIFORM, (4, 3, 2)),
            control_counts=(9, 2, 2),
            dual_2d=True,
        )
