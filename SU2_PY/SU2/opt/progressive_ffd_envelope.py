#!/usr/bin/env python

"""Basis-aware outer envelopes for progressive two-row FFD boxes."""

from dataclasses import dataclass
import math

from SU2.opt.progressive_ffd_blending import (
    FFDBlendingSpec,
    basis_values,
    evaluate_curve,
    invert_monotone_curve,
)


FIXED_OFFSET = "FIXED_OFFSET"
ADAPTIVE_CLEARANCE = "ADAPTIVE_CLEARANCE"
SUPPORTED_FFD_ENVELOPE_MODES = (FIXED_OFFSET, ADAPTIVE_CLEARANCE)


class FFDEnvelopeError(ValueError):
    """Raised when an adaptive FFD envelope cannot be constructed safely."""


@dataclass(frozen=True)
class FFDClearanceSpec:
    """Piecewise-linear chord-normalized clearance profile."""

    leading_chord: float = 0.005
    transition_start: float = 0.10
    transition_end: float = 0.20
    trailing_chord: float = 0.01

    def __post_init__(self):
        values = (
            self.leading_chord,
            self.transition_start,
            self.transition_end,
            self.trailing_chord,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise FFDEnvelopeError("FFD clearance parameters must be finite")
        if float(self.leading_chord) <= 0.0 or float(self.trailing_chord) <= 0.0:
            raise FFDEnvelopeError("FFD chord clearances must be positive")
        if not (
            0.0
            <= float(self.transition_start)
            < float(self.transition_end)
            <= 1.0
        ):
            raise FFDEnvelopeError(
                "FFD clearance transition must satisfy "
                "0 <= start < end <= 1"
            )

    def clearance_chord(self, x_over_c):
        x_over_c = float(x_over_c)
        if not math.isfinite(x_over_c):
            raise FFDEnvelopeError("FFD clearance location must be finite")
        if x_over_c <= float(self.transition_start):
            return float(self.leading_chord)
        if x_over_c >= float(self.transition_end):
            return float(self.trailing_chord)
        fraction = (
            (x_over_c - float(self.transition_start))
            / (float(self.transition_end) - float(self.transition_start))
        )
        return float(self.leading_chord) + fraction * (
            float(self.trailing_chord) - float(self.leading_chord)
        )

    def clearance(self, x, x_le, chord):
        return self.clearance_chord(
            (float(x) - float(x_le)) / float(chord)
        ) * float(chord)

    def as_dict(self):
        return {
            "leading_chord": float(self.leading_chord),
            "transition_start": float(self.transition_start),
            "transition_end": float(self.transition_end),
            "trailing_chord": float(self.trailing_chord),
        }


def normalize_ffd_envelope_mode(value):
    mode = str(value or FIXED_OFFSET).strip().upper()
    if mode not in SUPPORTED_FFD_ENVELOPE_MODES:
        raise FFDEnvelopeError(
            f"PROGRESSIVE_FFD_ENVELOPE_MODE must be one of "
            f"{SUPPORTED_FFD_ENVELOPE_MODES}, got {value!r}"
        )
    return mode


def _validated_side(side):
    side = str(side).strip().upper()
    if side not in ("UPPER", "LOWER"):
        raise FFDEnvelopeError(
            f"FFD envelope side must be UPPER or LOWER, got {side!r}"
        )
    return side


def _dot(left, right):
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _constraint_samples(
    columns,
    surface_y,
    surface_x,
    x_le,
    x_te,
    blending_spec,
    dense_count,
):
    dense_count = int(dense_count)
    if dense_count < 2:
        raise FFDEnvelopeError("FFD envelope dense sample count must be at least 2")
    chord = float(x_te) - float(x_le)
    if not math.isfinite(chord) or chord <= 0.0:
        raise FFDEnvelopeError("FFD envelope chord must be positive and finite")

    tolerance = 1.0e-12 * max(1.0, chord)
    x_values = {float(x_le), float(x_te)}
    for value in surface_x:
        x = float(value)
        if x < float(x_le) - tolerance or x > float(x_te) + tolerance:
            raise FFDEnvelopeError(
                f"Surface sample x={x:.16g} lies outside the FFD chord"
            )
        x_values.add(min(float(x_te), max(float(x_le), x)))
    for index in range(dense_count):
        x_values.add(float(x_le) + chord * float(index) / float(dense_count - 1))

    samples = []
    for x in sorted(x_values):
        try:
            u = invert_monotone_curve(columns, x, blending_spec, axis=0)
        except ValueError as exc:
            raise FFDEnvelopeError(
                f"Could not invert FFD x curve at x={x:.16g}: {exc}"
            ) from exc
        weights = basis_values(len(columns), u, blending_spec, axis=0)
        samples.append(
            {
                "x": x,
                "y": float(surface_y(x)),
                "u": float(u),
                "weights": [float(weight) for weight in weights],
            }
        )
    return samples


def build_adaptive_outer_row(
    *,
    columns,
    inner_controls,
    surface_y,
    surface_x,
    x_le,
    x_te,
    chord,
    side,
    blending_spec,
    clearance_spec,
    dense_count=4001,
):
    """Construct and verify a basis-aware outer row around one surface."""

    if not isinstance(blending_spec, FFDBlendingSpec):
        raise FFDEnvelopeError("blending_spec must be an FFDBlendingSpec")
    if not isinstance(clearance_spec, FFDClearanceSpec):
        raise FFDEnvelopeError("clearance_spec must be an FFDClearanceSpec")

    side = _validated_side(side)
    sign = 1.0 if side == "UPPER" else -1.0
    columns = [float(value) for value in columns]
    inner_controls = [float(value) for value in inner_controls]
    if len(columns) != len(inner_controls) or len(columns) < 2:
        raise FFDEnvelopeError(
            "FFD columns and inner controls must have the same non-trivial size"
        )
    if any(
        right - left <= 1.0e-12
        for left, right in zip(columns[:-1], columns[1:])
    ):
        raise FFDEnvelopeError("FFD envelope columns must be strictly increasing")
    chord = float(chord)
    expected_chord = float(x_te) - float(x_le)
    tolerance = 1.0e-10 * max(1.0, abs(chord), abs(expected_chord))
    if (
        not math.isfinite(chord)
        or chord <= 0.0
        or abs(chord - expected_chord) > tolerance
    ):
        raise FFDEnvelopeError("FFD envelope chord is inconsistent with LE/TE")

    surface_controls = [float(surface_y(x)) for x in columns]
    requested_controls = [
        clearance_spec.clearance(x, x_le, chord) for x in columns
    ]
    signed_outer = [
        sign * value + requested
        for value, requested in zip(surface_controls, requested_controls)
    ]
    samples = _constraint_samples(
        columns,
        surface_y,
        surface_x,
        x_le,
        x_te,
        blending_spec,
        dense_count,
    )
    numerical_pad = 1.0e-13 * max(1.0, chord)

    initial_deficits = []
    for sample in samples:
        requested = clearance_spec.clearance(sample["x"], x_le, chord)
        target = sign * sample["y"] + requested
        initial_deficits.append(target - _dot(sample["weights"], signed_outer))

    correction_count = 0
    for index in sorted(
        range(len(samples)),
        key=lambda item: initial_deficits[item],
        reverse=True,
    ):
        sample = samples[index]
        requested = clearance_spec.clearance(sample["x"], x_le, chord)
        target = sign * sample["y"] + requested
        deficit = target - _dot(sample["weights"], signed_outer)
        if deficit <= numerical_pad:
            continue
        denominator = sum(weight * weight for weight in sample["weights"])
        if denominator <= 0.0:
            raise FFDEnvelopeError(
                f"FFD basis has zero norm while correcting x={sample['x']:.16g}"
            )
        magnitude = (deficit + numerical_pad) / denominator
        signed_outer = [
            value + magnitude * weight
            for value, weight in zip(signed_outer, sample["weights"])
        ]
        correction_count += 1

    outer_controls = [sign * value for value in signed_outer]
    min_clearance = math.inf
    max_clearance = -math.inf
    min_v = math.inf
    max_v = -math.inf
    worst_sample = None
    row_tolerance = 1.0e-14 * max(1.0, chord)
    verification_tolerance = 5.0e-12 * max(1.0, chord)

    for sample in samples:
        y_inner = _dot(sample["weights"], inner_controls)
        y_outer = _dot(sample["weights"], outer_controls)
        clearance = sign * (y_outer - sample["y"])
        requested = clearance_spec.clearance(sample["x"], x_le, chord)
        margin = clearance - requested
        row_height = sign * (y_outer - y_inner)
        if row_height <= row_tolerance:
            raise FFDEnvelopeError(
                f"Adaptive FFD rows cross or collapse at x={sample['x']:.16g}: "
                f"height={row_height:.6e}"
            )
        if side == "UPPER":
            v = (sample["y"] - y_inner) / (y_outer - y_inner)
        else:
            v = (sample["y"] - y_outer) / (y_inner - y_outer)
        min_clearance = min(min_clearance, clearance)
        max_clearance = max(max_clearance, clearance)
        min_v = min(min_v, v)
        max_v = max(max_v, v)
        if worst_sample is None or margin < worst_sample["clearance_margin"]:
            worst_sample = {
                "x": float(sample["x"]),
                "x_over_c": (float(sample["x"]) - float(x_le)) / chord,
                "y": float(sample["y"]),
                "u": float(sample["u"]),
                "v": float(v),
                "requested_clearance": float(requested),
                "actual_clearance": float(clearance),
                "clearance_margin": float(margin),
            }

    if worst_sample is None:
        raise FFDEnvelopeError("Adaptive FFD envelope has no verification samples")
    min_margin = worst_sample["clearance_margin"]
    if min_margin < -verification_tolerance:
        raise FFDEnvelopeError(
            "Adaptive FFD envelope does not contain the requested surface: "
            f"margin={min_margin:.6e} at x={worst_sample['x']:.16g}"
        )
    if min_v < -verification_tolerance or max_v > 1.0 + verification_tolerance:
        raise FFDEnvelopeError(
            "Adaptive FFD envelope produced invalid surface parameters: "
            f"v_range=[{min_v:.16g},{max_v:.16g}]"
        )

    return {
        "side": side,
        "outer_controls": outer_controls,
        "surface_controls": surface_controls,
        "control_offsets": [
            sign * (outer - surface)
            for outer, surface in zip(outer_controls, surface_controls)
        ],
        "requested_control_clearances": requested_controls,
        "correction_count": correction_count,
        "sample_count": len(samples),
        "max_initial_deficit": max(0.0, max(initial_deficits)),
        "min_clearance": float(min_clearance),
        "max_clearance": float(max_clearance),
        "min_clearance_margin": float(min_margin),
        "min_v": float(min_v),
        "max_v": float(max_v),
        "worst_sample": worst_sample,
        "clearance_spec": clearance_spec.as_dict(),
    }


def sample_envelope_curves(
    columns,
    inner_controls,
    outer_controls,
    blending_spec,
    count=1001,
):
    count = int(count)
    if count < 2:
        raise FFDEnvelopeError("FFD envelope curve sample count must be at least 2")
    inner = []
    outer = []
    for index in range(count):
        u = float(index) / float(count - 1)
        x = evaluate_curve(columns, u, blending_spec, axis=0)
        inner.append(
            (x, evaluate_curve(inner_controls, u, blending_spec, axis=0))
        )
        outer.append(
            (x, evaluate_curve(outer_controls, u, blending_spec, axis=0))
        )
    return inner, outer
