#!/usr/bin/env python

from SU2.opt.progressive_hh_core import (
    HHLevel,
    _as_bool,
    assert_symmetric_centers,
    expand_symmetric_dv,
    full_to_reduced_symmetric,
    get_progressive_hh_options,
    initial_centers,
    is_symmetric_reduced,
    project_full_to_symmetric,
    reduce_symmetric_gradient,
    refine_uniform,
    symmetric_n_pairs_from_level,
    _compute_adaptive_nadd,
    _select_top_candidates,
    select_adaptive_candidate_batch,
    select_candidates_by_nadd_mode,
    select_candidates_by_nadd_mode_symmetric,
    refine_adaptive,
    apply_post_opt_coefficient_spring,
    should_refine,
)

from SU2.opt.progressive_hh_levels import (
    _resolve_from_cfg_dir,
    build_initial_level,
    build_next_level,
    build_spring_reallocated_level,
    make_hh_definition,
    _remove_progressive_keys,
    _prepare_local_mesh,
    write_level_config,
    _find_history_file,
    _read_history_values,
    _find_final_mesh,
    collect_level_result,
    append_selection_history_csv,
)

from SU2.opt.progressive_hh_projection import (
    get_midpoint_candidates,
    _find_real_adjoint_assets,
    _build_extended_dot_config,
    _make_projection_state,
    _extract_constraint_names,
    _run_dot_for_function,
    _run_geo_gradient_for_function,
    _compute_ikkt_residual_vector,
    _compute_dot_candidate_scores,
)
