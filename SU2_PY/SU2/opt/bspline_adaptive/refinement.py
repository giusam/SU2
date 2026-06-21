"""Knot-insertion refinement orchestration for one adaptive level."""



from SU2.opt.bspline_driver.reduction import active_mode_ids

from .boehm import (
    _check_transferred_coefficients_within_bounds,
    transfer_shape_to_inserted_space,
)
from .errors import BSplineAdaptiveError
from .knot_space import (
    _requested_knot_insertions,
    extract_clamped_knot_space,
    insert_knot_midpoint,
    knot_insertion_spans,
    reduced_ndv_for_knot_space,
    refinement_limit_ndv,
    regenerate_clamped_modes,
)
from .penalties import (
    _span_key,
    apply_knot_depth_penalty,
    get_or_initialize_knot_span_depths,
)
from .scoring import score_knot_spans

def build_next_knot_inserted_modes(optimized_modes, metadata, signal, settings):
    space = extract_clamped_knot_space(optimized_modes, settings)
    available_spans = len(
        knot_insertion_spans(
            space.knot_vector,
            min_width=settings.get("knot_min_span_width", 1.0e-8),
        )
    )
    n_insertions, batch_info = _requested_knot_insertions(
        space,
        settings,
        available_spans,
    )
    if n_insertions <= 0:
        return None, [], {
            "status": "no_valid_knot_span",
            "selected": False,
            "batch": batch_info,
        }

    print(
        "[PROGRESSIVE_BSPLINE] KNOT_INSERTION batch | "
        "mode={} current_reduced_ndv={} target_reduced_ndv={} insertions={}".format(
            batch_info["mode"],
            int(batch_info["current_reduced_ndv"]),
            int(batch_info["target_reduced_ndv"]),
            int(n_insertions),
        )
    )
    if batch_info.get("clamped"):
        print(
            "[PROGRESSIVE_BSPLINE] KNOT_INSERTION batch clamped | "
            "requested={} selected={} reason={}".format(
                int(batch_info.get("requested_insertions", n_insertions)),
                int(n_insertions),
                batch_info.get("clamp_reason", ""),
            )
        )

    score_rows = []
    selected_insertions = []
    current_spec = dict(optimized_modes)
    current_space = space
    current_knots = tuple(space.knot_vector)
    span_depths = get_or_initialize_knot_span_depths(
        current_spec,
        current_space.knot_vector,
        settings,
    )
    current_spec["knot_span_depths"] = dict(span_depths)

    for step in range(1, int(n_insertions) + 1):
        try:
            step_rows = score_knot_spans(current_space, metadata, signal, settings)
        except BSplineAdaptiveError:
            if selected_insertions:
                break
            raise
        for row in step_rows:
            row["batch_step"] = step
            row["selected"] = False
        step_rows = apply_knot_depth_penalty(step_rows, span_depths, settings)
        if not step_rows or float(step_rows[0].get("selection_score", 0.0)) <= 0.0:
            for row in step_rows:
                row["status"] = "nonpositive_score"
            score_rows.extend(step_rows)
            break

        selected = step_rows[0]
        selected["selected"] = True
        selected["status"] = "selected"
        old_knots = tuple(current_knots)
        new_knots = insert_knot_midpoint(
            current_space.knot_vector,
            (selected["span_left"], selected["span_right"], selected["inserted_knot"]),
        )
        parent_key = _span_key(selected["span_left"], selected["span_right"])
        parent_depth = int(span_depths[parent_key])
        child_depth = parent_depth + 1
        span_depths.pop(parent_key)
        span_depths[_span_key(selected["span_left"], selected["inserted_knot"])] = child_depth
        span_depths[_span_key(selected["inserted_knot"], selected["span_right"])] = child_depth
        selected_insertions.append(
            {
                "step": step,
                "span_key": str(selected.get("span_key", parent_key)),
                "span_left": float(selected["span_left"]),
                "span_right": float(selected["span_right"]),
                "inserted_knot": float(selected["inserted_knot"]),
                "side": str(selected.get("side", "BOTH")).upper(),
                "score": float(selected["score"]),
                "score_raw": float(selected["score_raw"]),
                "score_effective": float(selected.get("score_effective", selected["score"])),
                "selection_score": float(selected.get("selection_score", selected.get("score_effective", selected["score"]))),
                "parent_depth": int(selected.get("parent_depth", parent_depth)),
                "child_depth": int(selected.get("child_depth", child_depth)),
                "depth_penalty": float(selected.get("depth_penalty", 1.0)),
                "depth_penalty_mode": str(selected.get("depth_penalty_mode", "NONE")),
                "batch_depth": int(selected.get("batch_depth", 1)),
                "batch_penalty": float(selected.get("batch_penalty", 1.0)),
                "batch_penalty_mode": str(selected.get("batch_penalty_mode", "NONE")),
                "residual_energy": float(selected["residual_energy"]),
                "old_knot_vector": [float(value) for value in old_knots],
                "new_knot_vector": [float(value) for value in new_knots],
                "incremental_rank": int(selected["incremental_rank"]),
                "incremental_columns": int(selected["incremental_columns"]),
                "condition_number": float(selected["condition_number"]),
            }
        )
        score_rows.extend(step_rows)
        if str(selected.get("batch_penalty_mode", "NONE")).upper() == "NONE":
            print(
                "[PROGRESSIVE_BSPLINE] KNOT_INSERTION selected | "
                "step={} side={} span=[{:.6f},{:.6f}] knot={:.6f} score={:.6e}".format(
                    step,
                    str(selected.get("side", "BOTH")).upper(),
                    float(selected["span_left"]),
                    float(selected["span_right"]),
                    float(selected["inserted_knot"]),
                    float(selected["score"]),
                )
            )
        else:
            print(
                "[PROGRESSIVE_BSPLINE] KNOT_INSERTION selected | "
                "step={} side={} span=[{:.6f},{:.6f}] knot={:.6f} "
                "score_eff={:.6e} score_raw={:.6e} depth={} penalty={:.6e}".format(
                    step,
                    str(selected.get("side", "BOTH")).upper(),
                    float(selected["span_left"]),
                    float(selected["span_right"]),
                    float(selected["inserted_knot"]),
                    float(selected.get("score_effective", selected["score"])),
                    float(selected["score_raw"]),
                    int(selected.get("child_depth", selected.get("batch_depth", 1))),
                    float(selected.get("depth_penalty", selected.get("batch_penalty", 1.0))),
                )
            )
        current_spec = regenerate_clamped_modes(current_space, new_knots)
        current_spec["knot_span_depths"] = dict(span_depths)
        current_space = extract_clamped_knot_space(current_spec, settings)
        current_knots = tuple(new_knots)

    if not selected_insertions:
        return None, score_rows, {
            "status": "no_positive_knot_span",
            "selected": False,
            "batch": batch_info,
        }

    new_knots = current_knots
    coefficients_by_side, diagnostics = transfer_shape_to_inserted_space(
        space,
        metadata,
        new_knots,
        settings=settings,
    )
    next_modes = regenerate_clamped_modes(space, new_knots, coefficients_by_side)
    next_modes["knot_span_depths"] = dict(span_depths)
    _check_transferred_coefficients_within_bounds(next_modes, settings)
    if settings.get("nfinal") is not None and refinement_limit_ndv(next_modes, settings) > int(settings["nfinal"]):
        return None, score_rows, {
            "status": "nfinal_limit",
            "selected": False,
            "ndv_before": len(active_mode_ids(optimized_modes)),
            "ndv_after": len(active_mode_ids(next_modes)),
            "batch": batch_info,
            "selected_insertions": selected_insertions,
        }
    first = selected_insertions[0]
    selected_data = {
        "status": "ok",
        "selected": True,
        "refine_mode": "KNOT_INSERTION",
        "knot_score_mode": str(settings.get("knot_score_mode", "VIRTUAL_INSERTION")).upper(),
        "span_key": str(first.get("span_key", "")),
        "span_left": float(first["span_left"]),
        "span_right": float(first["span_right"]),
        "inserted_knot": float(first["inserted_knot"]),
        "side": str(first.get("side", "BOTH")).upper(),
        "score": float(first["score"]),
        "score_raw": float(first["score_raw"]),
        "score_effective": float(first.get("score_effective", first["score"])),
        "selection_score": float(first.get("selection_score", first.get("score_effective", first["score"]))),
        "parent_depth": int(first.get("parent_depth", 1)),
        "child_depth": int(first.get("child_depth", 1)),
        "depth_penalty": float(first.get("depth_penalty", 1.0)),
        "depth_penalty_mode": str(first.get("depth_penalty_mode", "NONE")),
        "residual_energy": float(first["residual_energy"]),
        "ndv_before": len(active_mode_ids(optimized_modes)),
        "ndv_after": len(active_mode_ids(next_modes)),
        "reduced_ndv_before": reduced_ndv_for_knot_space(space),
        "reduced_ndv_after": refinement_limit_ndv(next_modes, settings),
        "old_knot_vector": [float(value) for value in space.knot_vector],
        "new_knot_vector": [float(value) for value in new_knots],
        "selected_insertions": selected_insertions,
        "batch": batch_info | {"insertions": len(selected_insertions)},
    } | diagnostics
    return next_modes, score_rows, selected_data
