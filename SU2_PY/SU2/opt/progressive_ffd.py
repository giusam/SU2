#!/usr/bin/env python

from SU2.opt.progressive_ffd_core import (
    FFDLevel,
    build_ffd_mesh_columns,
    get_progressive_ffd_options,
    initial_ffd_columns_from_config,
    make_ffd_config_dump_compatible,
    make_dual_ffd_definition,
    make_ffd_definition,
    ordered_dual_ffd_records,
    validate_active_ffd_columns,
    refine_ffd_columns,
    select_ffd_candidates_by_nadd_mode,
    apply_post_opt_ffd_spring,
)

from SU2.opt.progressive_ffd_levels import (
    build_initial_ffd_level,
    build_next_ffd_level,
    build_ffd_spring_reallocated_level,
    refresh_ffd_scoring_baseline,
    write_ffd_level_config,
)

from SU2.opt.progressive_ffd_mesh import (
    FFDMeshError,
    read_ffd_box_columns,
    rewrite_ffd_box_with_columns_and_reembed,
    validate_mesh_ffd_columns,
)

from SU2.opt.progressive_ffd_prepare import (
    FFDPreparationError,
    prepare_progressive_ffd_input,
)

from SU2.opt.progressive_ffd_split import (
    rewrite_dual_ffd_boxes_with_columns_and_reembed,
)

from SU2.opt.progressive_ffd_projection import (
    _compute_ffd_dot_candidate_scores,
)

from SU2.opt.progressive_ffd_tangent import (
    COMPONENT,
    VIRTUAL_TANGENT,
    FFDTangentError,
    airfoil_area_value_and_field,
    airfoil_thickness_value_and_field,
    build_ffd_tangent_state,
    compare_tangent_spaces,
    fit_surface_ikkt_signal,
    load_surface_sensitivity_vector,
    project_surface_field,
)
