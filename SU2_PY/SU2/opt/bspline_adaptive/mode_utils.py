"""Mode-spec helpers for adaptive B-spline levels."""

from pathlib import Path

import numpy as np

from SU2.opt.bspline_modes import (
    evaluate_all_modes,
    validate_mode_spec,
)
from SU2.opt.bspline_driver.reduction import write_mode_spec

from .models import BsplineLevel

def _active_modes(mode_spec):
    validate_mode_spec(mode_spec)
    return [
        mode
        for mode in mode_spec.get("modes", [])
        if mode.get("active", True) is not False
    ]

def _mode_support(mode):
    basis_type = str(mode.get("basis_type", "")).strip().lower()
    if basis_type == "clamped":
        knots = mode.get("knot_vector", mode.get("knots"))
        if knots is not None and "basis_index" in mode:
            degree = int(mode.get("degree", 3))
            index = int(mode["basis_index"])
            knots = [float(value) for value in knots]
            right_index = min(len(knots) - 1, index + degree + 1)
            left = knots[index]
            right = knots[right_index]
            return left, right, 0.5 * (left + right)

    return 0.0, 1.0, 0.5

def mode_sort_key(mode):
    side_order = {"upper": 0, "lower": 1}
    left, _right, center = _mode_support(mode)
    side = str(mode.get("side", "")).strip().lower()
    return (
        side_order.get(side, 2),
        float(center),
        float(left),
        str(mode.get("id", "")),
    )

def _copy_global_metadata(source_spec, modes):
    from .settings import GLOBAL_MODE_KEYS

    return {
        key: source_spec[key]
        for key in GLOBAL_MODE_KEYS
        if key in source_spec
    } | {"modes": list(modes)}

def _mode_with_zero_coefficient(mode):
    new_mode = dict(mode)
    new_mode["coefficient"] = 0.0
    new_mode["active"] = True
    return new_mode

def build_level(level_id, active_modes, workdir, initial_modes_source):
    level_dir = Path(workdir) / f"LEVEL_{int(level_id):03d}"
    return BsplineLevel(
        level_id=int(level_id),
        workdir=level_dir,
        opt_workdir=level_dir / "opt_run",
        active_modes=active_modes,
        active_modes_start_filename=level_dir / "active_modes_start.json",
        optimized_modes_filename=level_dir / "opt_run" / "optimized_modes.json",
        selection_metadata={},
        initial_modes_source=Path(initial_modes_source),
    )

def write_level_start(level):
    level.workdir.mkdir(parents=True, exist_ok=True)
    write_mode_spec(level.active_modes, level.active_modes_start_filename)

def build_next_active_modes(optimized_modes, selected_modes):
    optimized = validate_mode_spec(optimized_modes)
    existing_ids = set()
    modes = []
    for mode in _active_modes(optimized):
        copied = dict(mode)
        copied["active"] = True
        existing_ids.add(str(copied["id"]))
        modes.append(copied)

    for mode in selected_modes:
        mode_id = str(mode.get("id", ""))
        if mode_id in existing_ids:
            continue
        modes.append(_mode_with_zero_coefficient(mode))
        existing_ids.add(mode_id)

    ordered = sorted(modes, key=mode_sort_key)
    return validate_mode_spec(_copy_global_metadata(optimized, ordered))

def _spec_with_modes(template_spec, modes):
    prepared = []
    for mode in modes:
        copied = dict(mode)
        copied["active"] = True
        if "coefficient" not in copied:
            copied["coefficient"] = 0.0
        prepared.append(copied)
    return validate_mode_spec(_copy_global_metadata(template_spec, prepared))

def evaluate_basis_matrix(template_spec, modes, metadata):
    if not modes:
        return np.zeros((len(metadata), 0), dtype=float)
    spec = _spec_with_modes(template_spec, modes)
    x_over_c = [record["x_over_c"] for record in metadata]
    sides = [record["side"] for record in metadata]
    values = evaluate_all_modes(spec, x_over_c, sides=sides)
    columns = [np.asarray(values[str(mode["id"])], dtype=float) for mode in modes]
    return np.column_stack(columns) if columns else np.zeros((len(metadata), 0), dtype=float)
