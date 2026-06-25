"""BSplineSU2Driver and projected-gradient optimization helpers."""

import csv
import json
import math
import shutil
import sys
from collections import deque
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from SU2.opt.bspline_def import (
    extract_marker_nodes,
    read_su2_mesh,
)
from SU2.opt.bspline_dot import (
    BSplineDotError,
    normalize_sensitivity_weighting,
    read_metadata,
)
from SU2.opt.bspline_modes import (
    BSplineModeError,
    LE_SAFE_DEFAULT_POWER,
    LE_SAFE_DEFAULT_X0,
    LE_SAFE_DEFAULT_X1,
    active_sides_from_surface_mode,
    evaluate_all_modes,
    load_mode_spec,
    normalize_deformation_direction_mode,
    normalize_surface_mode,
    validate_le_safe_direction_options,
    validate_surface_mode_against_modes,
)
from SU2.opt.progressive_trigger import (
    RefinementTriggered,
    record_objective_and_check,
    trigger_prefix,
)
from SU2.opt.thickness_constraint import (
    _as_bool as _thickness_as_bool,
    _load_or_build_reference,
    _normalize_gradient_mode,
    _parse_x_stations,
    _resolve_from_cfg_dir,
    _x_stations_value_is_empty,
)
from .commands import (
    _append_command_log,
    _normalize_eval_layout,
    _normalize_objective_adjoint,
    _relative_path,
    _with_mpi,
    build_eval_commands,
    build_eval_paths,
    create_eval_aliases,
    ensure_adjoint_solution_input,
    normalize_sensitivity_source,
    run_command,
)
from .config_apply import resolve_thickness_domain_mode
from .config_parse import (
    _ConfigDict,
    _format_config_atom,
    patch_config_template,
)
from .constants import (
    ALLOWED_SYMMETRY_COUPLINGS,
    DEFAULT_BOUNDS,
)
from .errors import (
    BSplineSU2DriverError,
    GradientGuardStop,
    TrustClipStop,
    _as_float,
    _normalized_name,
)
from .geometry_bounds import (
    _bounds_are_uniform,
    _bounds_summary,
    _bounds_to_list,
    _vector_summary,
    compute_geometry_aware_bound_scaling,
)
from .geometry_constraints import (
    BSplineAirfoilAreaMetric,
    BSplineAirfoilMaxThicknessMetric,
    SUPPORTED_BSPLINE_GEOMETRY_CONSTRAINTS,
    is_geometry_constraint_name,
    normalize_geometry_constraint_name,
)
from .guards import (
    ALLOWED_TRUST_CLIP_POLICIES,
    LAST_EVAL_CACHE_BLOCKED_TRUST_CLIP_CLASSES,
    SAFE_EVALUATION_STATUSES,
    _trust_clip_options,
    classify_clipped_trial,
    gradient_guard_triggered,
)
from .reduction import (
    _active_modes,
    _validated_bounds,
    active_bounds,
    active_coefficient_vector,
    active_mode_ids,
    build_reduced_variables,
    cache_key,
    collapse_full_gradient,
    collapse_full_jacobian,
    compress_full_coefficients,
    expand_reduced_coefficients,
    reduced_bounds_from_full_bounds,
    reduced_step_limits_from_modes,
    update_mode_coefficients,
    write_mode_spec,
)
from .native_constraints import normalize_native_constraints
from .tables import (
    FUNCTION_HISTORY_COLUMNS,
    history_column_for_function,
    read_gradient_vector,
    read_objective_from_history,
)
from .thickness import BSplineThicknessConstraint

ALLOWED_GEOMETRY_CONSTRAINT_GRADIENT_MODES = ("AUTO", "ANALYTIC", "SU2_GEO")


def normalize_geometry_constraint_gradient_mode(value):
    mode = str(value or "AUTO").strip().upper()
    if mode not in ALLOWED_GEOMETRY_CONSTRAINT_GRADIENT_MODES:
        raise BSplineSU2DriverError(
            "BSPLINE_GEOMETRY_CONSTRAINT_GRADIENT must be AUTO, ANALYTIC, "
            f"or SU2_GEO; got {mode!r}"
        )
    return mode


def _project_to_bounds(coefficients, bounds):
    clipped = []
    for value, (lower, upper) in zip(coefficients, bounds):
        clipped.append(min(max(float(value), lower), upper))
    return clipped








class BSplineSU2Driver:
    def __init__(
        self,
        modes_filename,
        base_mesh,
        marker,
        def_template,
        primal_template,
        adjoint_template,
        workdir,
        objective_column="CD",
        mpi_prefix="",
        default_bounds=DEFAULT_BOUNDS,
        cache_tol=1.0e-12,
        python_executable=None,
        show_commands=False,
        stream_solver_output=False,
        print_optimizer_table=True,
        auto_scale_bounds_to_geometry=False,
        max_normal_displacement=None,
        max_rms_normal_displacement=None,
        min_bound_scale=0.0,
        opt_accuracy=None,
        opt_bound_upper=None,
        opt_bound_lower=None,
        opt_relax_factor=1.0,
        opt_gradient_factor=1.0,
        gradient_guard=True,
        gradient_guard_factor=100.0,
        gradient_guard_window=5,
        gradient_guard_min_history=3,
        gradient_guard_floor=1.0e-14,
        gradient_guard_next_action="restart_same_level",
        refinement_available=None,
        trust_clip_policy="OFF",
        trust_clip_beta_tol=1.0e-12,
        trust_clip_legacy_beta_min=0.50,
        trust_clip_severe_beta=0.50,
        trust_clip_worsening_tol=0.05,
        trust_clip_soft_gnorm_factor=20.0,
        trust_clip_bad_patience=2,
        trust_clip_bad_window=5,
        trust_clip_stag_tol=1.0e-6,
        opt_line_search_bound=None,
        thickness_options=None,
        native_constraints=None,
        geometry_fd_eps=1.0e-6,
        geometry_constraint_gradient="AUTO",
        eval_layout="DSN",
        objective_adjoint="drag",
        symmetry_coupling="NONE",
        surface_mode="BOTH",
        sensitivity_weighting="NODAL",
        sensitivity_source="DOT_AD_TRANSFER",
        local_step_limit=False,
        local_step_limit_ratio=200.0,
        trigger_opts=None,
        progressive_label="PROGRESSIVE_BSPLINE",
        deformation_direction_mode=None,
        le_safe_direction=False,
        le_safe_x0=None,
        le_safe_x1=None,
        le_safe_power=None,
    ):
        self.modes_filename = Path(modes_filename).resolve()
        self.base_mesh = Path(base_mesh).resolve()
        self.marker = marker
        self.def_template = Path(def_template).resolve()
        self.primal_template = Path(primal_template).resolve()
        self.adjoint_template = Path(adjoint_template).resolve()
        self.workdir = Path(workdir).resolve()
        self.objective_column = objective_column
        self.eval_layout = _normalize_eval_layout(eval_layout)
        self.objective_adjoint = _normalize_objective_adjoint(objective_adjoint)
        try:
            self.surface_mode = normalize_surface_mode(surface_mode)
        except BSplineModeError as exc:
            raise BSplineSU2DriverError(str(exc))
        self.symmetry_coupling = str(symmetry_coupling or "NONE").strip().upper()
        if self.symmetry_coupling not in ALLOWED_SYMMETRY_COUPLINGS:
            raise BSplineSU2DriverError(
                f"BSPLINE_SYMMETRY_COUPLING must be one of {ALLOWED_SYMMETRY_COUPLINGS}; got {self.symmetry_coupling!r}"
            )
        if self.surface_mode != "BOTH" and self.symmetry_coupling != "NONE":
            raise BSplineSU2DriverError(
                "BSPLINE_SYMMETRY_COUPLING is only valid with BSPLINE_SURFACE_MODE=BOTH"
            )
        try:
            self.deformation_direction_mode = normalize_deformation_direction_mode(
                deformation_direction_mode,
                le_safe_direction=le_safe_direction,
            )
            self.le_safe_direction_options = validate_le_safe_direction_options(
                le_safe_direction=self.deformation_direction_mode == "LE_SAFE",
                le_safe_x0=le_safe_x0
                if self.deformation_direction_mode == "LE_SAFE"
                and le_safe_x0 is not None
                else LE_SAFE_DEFAULT_X0,
                le_safe_x1=le_safe_x1
                if self.deformation_direction_mode == "LE_SAFE"
                and le_safe_x1 is not None
                else LE_SAFE_DEFAULT_X1,
                le_safe_power=le_safe_power
                if self.deformation_direction_mode == "LE_SAFE"
                and le_safe_power is not None
                else LE_SAFE_DEFAULT_POWER,
            )
        except BSplineModeError as exc:
            raise BSplineSU2DriverError(str(exc))
        try:
            self.sensitivity_weighting = normalize_sensitivity_weighting(sensitivity_weighting)
        except BSplineDotError as exc:
            raise BSplineSU2DriverError(str(exc))
        self.sensitivity_source = normalize_sensitivity_source(sensitivity_source)
        self.mpi_prefix = mpi_prefix or ""
        self.default_bounds = _validated_bounds(default_bounds, "default_bounds")
        self.cache_tol = cache_tol
        self.python_executable = python_executable or sys.executable or "python3"
        self.show_commands = bool(show_commands)
        self.stream_solver_output = bool(stream_solver_output)
        self.print_optimizer_table = bool(print_optimizer_table)
        self.auto_scale_bounds_to_geometry = bool(auto_scale_bounds_to_geometry)
        self.max_normal_displacement = max_normal_displacement
        self.max_rms_normal_displacement = max_rms_normal_displacement
        self.min_bound_scale = 0.0 if min_bound_scale is None else float(min_bound_scale)
        self.opt_accuracy = None if opt_accuracy is None else float(opt_accuracy)
        self.opt_bound_upper = (
            None if opt_bound_upper is None else _as_float(opt_bound_upper, "OPT_BOUND_UPPER")
        )
        self.opt_bound_lower = (
            None if opt_bound_lower is None else _as_float(opt_bound_lower, "OPT_BOUND_LOWER")
        )
        self.opt_relax_factor = _as_float(
            1.0 if opt_relax_factor is None else opt_relax_factor,
            "OPT_RELAX_FACTOR",
        )
        if self.opt_relax_factor <= 0.0:
            raise BSplineSU2DriverError("OPT_RELAX_FACTOR must be positive")
        self.opt_gradient_factor = _as_float(
            1.0 if opt_gradient_factor is None else opt_gradient_factor,
            "OPT_GRADIENT_FACTOR",
        )
        if self.opt_gradient_factor <= 0.0:
            raise BSplineSU2DriverError("OPT_GRADIENT_FACTOR must be positive")
        self.gradient_guard_enabled = _thickness_as_bool(
            gradient_guard,
            default=True,
        )
        self.gradient_guard_factor = _as_float(
            gradient_guard_factor,
            "BSPLINE_GRADIENT_GUARD_FACTOR",
        )
        self.gradient_guard_window = int(gradient_guard_window)
        self.gradient_guard_min_history = int(gradient_guard_min_history)
        self.gradient_guard_floor = _as_float(
            gradient_guard_floor,
            "BSPLINE_GRADIENT_GUARD_FLOOR",
        )
        self.gradient_guard_next_action = str(
            gradient_guard_next_action or "restart_same_level"
        ).strip().lower()
        if self.gradient_guard_factor <= 0.0:
            raise BSplineSU2DriverError("BSPLINE_GRADIENT_GUARD_FACTOR must be positive")
        if self.gradient_guard_window < 1:
            raise BSplineSU2DriverError("BSPLINE_GRADIENT_GUARD_WINDOW must be >= 1")
        if self.gradient_guard_min_history < 1:
            raise BSplineSU2DriverError("BSPLINE_GRADIENT_GUARD_MIN_HISTORY must be >= 1")
        if self.gradient_guard_floor <= 0.0:
            raise BSplineSU2DriverError("BSPLINE_GRADIENT_GUARD_FLOOR must be positive")
        if self.gradient_guard_next_action not in (
            "refine",
            "restart_same_level",
            "terminate_last_safe",
        ):
            raise BSplineSU2DriverError(
                "gradient guard next action must be refine, restart_same_level, "
                "or terminate_last_safe"
            )
        self.refinement_available = (
            self.gradient_guard_next_action == "refine"
            if refinement_available is None
            else bool(refinement_available)
        )
        self.trust_clip_options = _trust_clip_options(
            {
                "policy": trust_clip_policy,
                "beta_tol": _as_float(trust_clip_beta_tol, "BSPLINE_TRUST_CLIP_BETA_TOL"),
                "legacy_beta_min": _as_float(
                    trust_clip_legacy_beta_min,
                    "BSPLINE_TRUST_CLIP_LEGACY_BETA_MIN",
                ),
                "severe_beta": _as_float(
                    trust_clip_severe_beta,
                    "BSPLINE_TRUST_CLIP_SEVERE_BETA",
                ),
                "worsening_tol": _as_float(
                    trust_clip_worsening_tol,
                    "BSPLINE_TRUST_CLIP_WORSENING_TOL",
                ),
                "soft_gnorm_factor": _as_float(
                    trust_clip_soft_gnorm_factor,
                    "BSPLINE_TRUST_CLIP_SOFT_GNORM_FACTOR",
                ),
                "bad_patience": int(trust_clip_bad_patience),
                "bad_window": int(trust_clip_bad_window),
                "stag_tol": _as_float(
                    trust_clip_stag_tol,
                    "BSPLINE_TRUST_CLIP_STAG_TOL",
                ),
                "gnorm_floor": self.gradient_guard_floor,
                "objective_floor": 1.0e-12,
            }
        )
        if self.trust_clip_options["policy"] not in ALLOWED_TRUST_CLIP_POLICIES:
            raise BSplineSU2DriverError(
                "BSPLINE_TRUST_CLIP_POLICY must be OFF or ACCEPT_RESTART"
            )
        for key in ("beta_tol", "worsening_tol", "stag_tol"):
            if float(self.trust_clip_options[key]) < 0.0:
                raise BSplineSU2DriverError(f"trust-clip {key} must be non-negative")
        for key in ("legacy_beta_min", "severe_beta"):
            if not 0.0 <= float(self.trust_clip_options[key]) <= 1.0:
                raise BSplineSU2DriverError(f"trust-clip {key} must be in [0, 1]")
        if float(self.trust_clip_options["soft_gnorm_factor"]) <= 0.0:
            raise BSplineSU2DriverError("trust-clip soft_gnorm_factor must be positive")
        if int(self.trust_clip_options["bad_patience"]) < 1:
            raise BSplineSU2DriverError("trust-clip bad_patience must be >= 1")
        if int(self.trust_clip_options["bad_window"]) < 1:
            raise BSplineSU2DriverError("trust-clip bad_window must be >= 1")
        self.opt_line_search_bound = (
            None
            if opt_line_search_bound is None
            else _as_float(opt_line_search_bound, "OPT_LINE_SEARCH_BOUND")
        )
        if self.opt_line_search_bound is not None and self.opt_line_search_bound <= 0.0:
            raise BSplineSU2DriverError("OPT_LINE_SEARCH_BOUND must be positive")
        self.local_step_limit = _thickness_as_bool(local_step_limit, default=False)
        self.local_step_limit_ratio = _as_float(
            local_step_limit_ratio,
            "BSPLINE_LOCAL_STEP_LIMIT_RATIO",
        )
        if self.local_step_limit_ratio <= 0.0:
            raise BSplineSU2DriverError("BSPLINE_LOCAL_STEP_LIMIT_RATIO must be positive")
        if (self.opt_bound_lower is None) != (self.opt_bound_upper is None):
            raise BSplineSU2DriverError(
                "OPT_BOUND_LOWER and OPT_BOUND_UPPER must be provided together"
            )
        self.thickness_options = dict(thickness_options or {})
        self.native_constraints = normalize_native_constraints(native_constraints)
        self.geometry_fd_eps = _as_float(
            1.0e-6 if geometry_fd_eps is None else geometry_fd_eps,
            "BSPLINE_GEOMETRY_FD_EPS",
        )
        if self.geometry_fd_eps <= 0.0:
            raise BSplineSU2DriverError("BSPLINE_GEOMETRY_FD_EPS must be positive")
        self.geometry_constraint_gradient_mode = (
            normalize_geometry_constraint_gradient_mode(geometry_constraint_gradient)
        )
        if any(
            normalize_geometry_constraint_name(spec.name) == "AIRFOIL_AREA"
            for spec in self.native_constraints
        ) and self.surface_mode != "BOTH":
            raise BSplineSU2DriverError(
                "AIRFOIL_AREA requires BSPLINE_SURFACE_MODE=BOTH"
            )
        self.thickness_constraint = None
        self._thickness_constraint_configured = False
        self._thickness_fallback_warned = False
        self._printed_iteration_header = False

        for filename in (
            self.modes_filename,
            self.base_mesh,
            self.def_template,
            self.primal_template,
            self.adjoint_template,
        ):
            if not filename.exists():
                raise BSplineSU2DriverError(f"required file was not found: {filename}")

        self.mode_spec = load_mode_spec(str(self.modes_filename))
        try:
            validate_surface_mode_against_modes(self.mode_spec, self.surface_mode)
        except BSplineModeError as exc:
            raise BSplineSU2DriverError(str(exc))
        if self.surface_mode != "BOTH" and "surface_mode" not in self.mode_spec:
            self.mode_spec["surface_mode"] = self.surface_mode
        self.mode_ids = active_mode_ids(self.mode_spec)
        if not self.mode_ids:
            raise BSplineSU2DriverError("bspline_modes.json has no active modes")
        original_initial_coefficients = active_coefficient_vector(self.mode_spec)
        if self.opt_bound_lower is not None and self.opt_bound_upper is not None:
            override_bounds = _validated_bounds(
                (self.opt_bound_lower, self.opt_bound_upper),
                "OPT_BOUND_LOWER/OPT_BOUND_UPPER",
            )
            self.bounds = [override_bounds for _mode_id in self.mode_ids]
        else:
            self.bounds = active_bounds(self.mode_spec, self.default_bounds)
        self.original_bounds = list(self.bounds)
        self.reduced_variables, symmetry_warnings = build_reduced_variables(
            self.mode_spec,
            self.symmetry_coupling,
        )
        for warning in symmetry_warnings:
            print(f"[BSPLINE_SU2_DRIVER] WARNING: {warning}")
        self.reduced_variable_ids = [variable.id for variable in self.reduced_variables]
        self.initial_reduced_coefficients = compress_full_coefficients(
            original_initial_coefficients,
            self.reduced_variables,
            warn=lambda message: print(f"[BSPLINE_SU2_DRIVER] WARNING: {message}"),
            coupling=self.symmetry_coupling,
        )
        self.initial_coefficients = expand_reduced_coefficients(
            self.initial_reduced_coefficients,
            self.reduced_variables,
            len(self.mode_ids),
        )
        self.reduced_bounds = reduced_bounds_from_full_bounds(
            self.bounds,
            self.reduced_variables,
        )
        self.reduced_local_step_limits = reduced_step_limits_from_modes(
            self.mode_spec,
            self.reduced_variables,
            self.local_step_limit_ratio,
        )
        self.geometry_bounds_scaling = None
        self._geometry_bounds_configured = False
        self._geometry_probe = None
        self._geometry_metric_cache = {}
        self._line_search_basis_matrix = None
        self._line_search_bound_configured = False
        self._line_search_anchor_physical = list(self.initial_coefficients)
        self._local_step_anchor_reduced = list(self.initial_reduced_coefficients)
        self._cache = {}
        self._constraint_cache = {}
        self._design_points = {}
        self._last_eval_physical_key = None
        self._last_eval_result = None
        self._last_eval_info = None
        self._history_records = []
        self.last_safe_entry = None
        self.best_safe_entry = None
        self.best_physical_entry = None
        self.recent_safe_raw_gnorms = deque(maxlen=self.gradient_guard_window)
        self.last_gradient_guard_stop = None
        self.anchor_entry = None
        self.recent_level_clip_events = deque(
            maxlen=int(self.trust_clip_options["bad_window"])
        )
        self._trust_clip_by_requested_key = {}
        self.last_trust_clip_stop = None
        self._slsqp_major_iter = 0
        self._run_eval_count = 0
        self._printed_commands_log_path = False
        self.trigger_project = SimpleNamespace(
            trigger_opts=dict(trigger_opts) if trigger_opts else None,
            trigger_history=[],
            trigger_state=None,
            refinement_triggered=False,
            progressive_label=str(progressive_label or "PROGRESSIVE_BSPLINE"),
        )
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._next_eval_id = self._initial_eval_id()

    def _probe_geometry_aware_bounds(self):
        if self._geometry_probe is not None:
            return self._geometry_probe

        probe_dir = self.workdir / "bounds_scaling_probe"
        probe_paths = build_eval_paths(
            probe_dir,
            eval_layout=self.eval_layout,
            objective_adjoint=self.objective_adjoint,
        )
        probe_dir.mkdir(parents=True, exist_ok=True)
        write_mode_spec(
            update_mode_coefficients(self.mode_spec, self.initial_coefficients),
            probe_paths.modes_current,
        )

        commands = build_eval_commands(
            probe_paths,
            self.base_mesh,
            self.marker,
            mpi_prefix=self.mpi_prefix,
            python_executable=self.python_executable,
            sensitivity_weighting=self.sensitivity_weighting,
            surface_mode=self.surface_mode,
            deformation_direction_mode=self.deformation_direction_mode,
            le_safe_direction=self.le_safe_direction_options["le_safe_direction"],
            le_safe_x0=self.le_safe_direction_options["le_safe_x0"],
            le_safe_x1=self.le_safe_direction_options["le_safe_x1"],
            le_safe_power=self.le_safe_direction_options["le_safe_power"],
        )
        _append_command_log(
            probe_paths.commands_log,
            "bounds_scaling_bspline_def",
            probe_paths.deform_dir,
            commands["bspline_def"],
        )
        _append_command_log(
            self.workdir / "commands.log",
            "bounds_scaling_bspline_def",
            probe_paths.deform_dir,
            commands["bspline_def"],
        )
        if self.show_commands and not self._printed_commands_log_path:
            print(
                "[BSPLINE_SU2_DRIVER] Commands are logged in {}".format(
                    self.workdir / "commands.log"
                )
            )
            self._printed_commands_log_path = True
        run_command(
            commands["bspline_def"],
            probe_paths.deform_dir,
            probe_paths.bspline_def_log,
            show_command=False,
            stream_output=self.stream_solver_output,
            stage="bounds_scaling_bspline_def",
        )

        metadata = read_metadata(probe_paths.metadata)
        x_over_c = [float(row["x_over_c"]) for row in metadata]
        sides = [str(row["side"]).strip().lower() for row in metadata]
        values_by_id = evaluate_all_modes(self.mode_spec, x_over_c, sides=sides)
        active_modes = _active_modes(self.mode_spec)
        columns = [
            np.asarray(values_by_id[str(mode["id"])], dtype=float)
            for mode in active_modes
        ]
        basis_matrix = (
            np.column_stack(columns) if columns else np.zeros((len(metadata), 0), dtype=float)
        )
        self._geometry_probe = (metadata, basis_matrix)
        return self._geometry_probe

    def configure_geometry_aware_bounds(self):
        if self._geometry_bounds_configured:
            return self.geometry_bounds_scaling

        self._geometry_bounds_configured = True
        if not self.auto_scale_bounds_to_geometry:
            return None
        if (
            self.max_normal_displacement is None
            and self.max_rms_normal_displacement is None
        ):
            raise BSplineSU2DriverError(
                "at least one of --max-normal-displacement or --max-rms-normal-displacement must be provided when --auto-scale-bounds-to-geometry is enabled"
            )

        _metadata, basis_matrix = self._probe_geometry_aware_bounds()
        summary = compute_geometry_aware_bound_scaling(
            basis_matrix,
            self.initial_coefficients,
            self.original_bounds,
            max_normal_displacement=self.max_normal_displacement,
            max_rms_normal_displacement=self.max_rms_normal_displacement,
            min_bound_scale=self.min_bound_scale,
        )
        self.bounds = [tuple(bounds) for bounds in summary["scaled_bounds"]]
        self.reduced_bounds = reduced_bounds_from_full_bounds(
            self.bounds,
            self.reduced_variables,
        )
        self.geometry_bounds_scaling = summary

        print("[BSPLINE_SU2_DRIVER] Geometry-aware bound scaling active")
        print(
            "[BSPLINE_SU2_DRIVER] max|dn| current = {:.15g}".format(
                summary["max_abs_dn_current"]
            )
        )
        print(
            "[BSPLINE_SU2_DRIVER] rms(dn) current = {:.15g}".format(
                summary["rms_dn_current"]
            )
        )
        print(
            "[BSPLINE_SU2_DRIVER] beta_safe = {:.15g}".format(
                summary["beta_safe"]
            )
        )
        print("[BSPLINE_SU2_DRIVER] coefficient bounds rescaled around current design")

        bounds_scaling_file = self.workdir / "bounds_scaling.json"
        with open(bounds_scaling_file, "w") as fp:
            json.dump(summary, fp, indent=2, sort_keys=True)
            fp.write("\n")
        return summary

    def physical_to_optimizer(self, coefficients):
        return [float(value) / self.opt_relax_factor for value in coefficients]

    def optimizer_to_physical(self, variables):
        return [float(value) * self.opt_relax_factor for value in variables]

    def expand_reduced_physical(self, reduced_coefficients):
        return expand_reduced_coefficients(
            reduced_coefficients,
            self.reduced_variables,
            len(self.mode_ids),
        )

    def compress_full_physical(self, coefficients, warn=False):
        # NOTE: coupling is intentionally NOT forwarded here. This wrapper is
        # called on coefficients that already came out of
        # expand_reduced_physical during optimization, so they are
        # antisymmetric by construction and the C1 unilateral-bump check
        # would only risk a spurious mid-run abort. The check is enforced
        # once, on the user-supplied initial coefficients, in __init__.
        return compress_full_coefficients(
            coefficients,
            self.reduced_variables,
            warn=(
                (lambda message: print(f"[BSPLINE_SU2_DRIVER] WARNING: {message}"))
                if warn
                else None
            ),
        )

    def collapse_gradient_to_reduced(self, gradient):
        return collapse_full_gradient(gradient, self.reduced_variables)

    def collapse_jacobian_to_reduced(self, jacobian):
        return collapse_full_jacobian(jacobian, self.reduced_variables)

    def optimizer_bounds(self):
        return [
            (
                float(lower) / self.opt_relax_factor,
                float(upper) / self.opt_relax_factor,
            )
            for lower, upper in self.reduced_bounds
        ]

    def _line_search_default_info(self):
        return {
            "line_search_beta": 1.0,
            "line_search_maxdiff": 0.0,
            "line_search_limited": 0,
            "line_search_beta_geometry": 1.0,
            "local_step_beta": 1.0,
            "local_step_limited": 0,
            "local_step_limiting_mode": "",
            "local_step_da": 0.0,
            "local_step_limit": 0.0,
        }

    def _configure_line_search_bound(self):
        if self._line_search_bound_configured:
            return self._line_search_basis_matrix

        self._line_search_bound_configured = True
        self._line_search_anchor_physical = list(self.initial_coefficients)
        self._local_step_anchor_reduced = list(self.initial_reduced_coefficients)
        if self.opt_line_search_bound is None:
            return None

        _metadata, basis_matrix = self._probe_geometry_aware_bounds()
        basis_matrix = np.asarray(basis_matrix, dtype=float)
        if basis_matrix.ndim != 2:
            raise BSplineSU2DriverError("basis matrix for OPT_LINE_SEARCH_BOUND must be two-dimensional")
        if basis_matrix.shape[1] != len(self.mode_ids):
            raise BSplineSU2DriverError(
                "basis matrix column count must match the number of active modes"
            )
        self._line_search_basis_matrix = basis_matrix
        print(
            "[BSPLINE_SU2_DRIVER] OPT_LINE_SEARCH_BOUND active: max accepted normal jump = {:.15g}".format(
                float(self.opt_line_search_bound)
            )
        )
        return self._line_search_basis_matrix

    def _marker_closed(self, marker):
        mesh = read_su2_mesh(str(self.base_mesh))
        _tag, _node_ids, closed = extract_marker_nodes(mesh, marker)
        return bool(closed)

    def _build_bspline_thickness_constraint(self):
        options = dict(self.thickness_options or {})
        enabled = _thickness_as_bool(
            options.get("PROGRESSIVE_THICKNESS_CONSTRAINT", "NO"),
            default=False,
        )
        if not enabled:
            return None

        ref_mesh_value = options.get("PROGRESSIVE_THICKNESS_REF_MESH")
        if not ref_mesh_value:
            raise BSplineSU2DriverError(
                "PROGRESSIVE_THICKNESS_REF_MESH is required when "
                "PROGRESSIVE_THICKNESS_CONSTRAINT=YES"
            )

        cfg = _ConfigDict(options)
        if options.get("_optimizer_config_filename"):
            cfg._filename = str(options["_optimizer_config_filename"])

        ref_mesh = _resolve_from_cfg_dir(cfg, ref_mesh_value)
        marker = str(options.get("PROGRESSIVE_THICKNESS_MARKER", self.marker))
        if marker.strip().lower() != str(self.marker).strip().lower():
            raise BSplineSU2DriverError(
                "B-spline thickness marker must match the optimizer marker "
                f"({marker!r} != {self.marker!r})"
            )

        domain_mode = resolve_thickness_domain_mode(
            self.surface_mode,
            options.get("PROGRESSIVE_THICKNESS_DOMAIN_MODE", "AUTO"),
        )
        symmetry_y = float(options.get("PROGRESSIVE_THICKNESS_SYMMETRY_Y", 0.0))
        margin = float(options.get("PROGRESSIVE_THICKNESS_MARGIN", 0.0))
        fd_eps = float(options.get("PROGRESSIVE_THICKNESS_FD_EPS", 1.0e-6))
        gradient_mode = _normalize_gradient_mode(
            options.get("PROGRESSIVE_THICKNESS_GRADIENT", "AUTO")
        )

        x_stations_value = options.get("PROGRESSIVE_THICKNESS_X_STATIONS")
        explicit_x_stations = not _x_stations_value_is_empty(x_stations_value)
        if explicit_x_stations:
            x_stations = _parse_x_stations(x_stations_value)
        else:
            npoints = int(options.get("PROGRESSIVE_THICKNESS_NPOINTS", 101))
            xmin = float(options.get("PROGRESSIVE_THICKNESS_XMIN", 0.001))
            xmax = float(options.get("PROGRESSIVE_THICKNESS_XMAX", 0.999))
            if npoints < 2:
                raise BSplineSU2DriverError("PROGRESSIVE_THICKNESS_NPOINTS must be >= 2")
            if not xmin < xmax:
                raise BSplineSU2DriverError("PROGRESSIVE_THICKNESS_XMIN must be < XMAX")
            x_stations = np.linspace(xmin, xmax, npoints)

        cache_value = options.get(
            "PROGRESSIVE_THICKNESS_CACHE_FILE",
            "thickness_reference.npz",
        )
        cache_file = _resolve_from_cfg_dir(cfg, cache_value) if cache_value else None
        reference = _load_or_build_reference(
            ref_mesh,
            marker,
            x_stations,
            cache_file,
            domain_mode,
            symmetry_y,
        )

        metadata, basis_matrix = self._probe_geometry_aware_bounds()
        closed = self._marker_closed(marker)
        constraint = BSplineThicknessConstraint(
            metadata,
            basis_matrix,
            self.mode_ids,
            reference,
            x_stations,
            margin=margin,
            domain_mode=domain_mode,
            symmetry_y=symmetry_y,
            fd_eps=fd_eps,
            gradient_mode=gradient_mode,
            closed=closed,
            marker=marker,
        )

        print("[BSPLINE_SU2_DRIVER] Thickness constraint active")
        print(f"[BSPLINE_SU2_DRIVER] thickness marker = {marker}")
        print(f"[BSPLINE_SU2_DRIVER] thickness domain mode = {domain_mode}")
        print(f"[PROGRESSIVE_BSPLINE][THICKNESS] domain = {domain_mode}")
        print(f"[PROGRESSIVE_BSPLINE][THICKNESS] symmetry_y = {symmetry_y}")
        print(f"[BSPLINE_SU2_DRIVER] thickness gradient mode = {gradient_mode}")
        print(f"[BSPLINE_SU2_DRIVER] thickness stations = {len(x_stations)}")
        print(
            "[BSPLINE_SU2_DRIVER] min reference thickness = {:.6e}".format(
                float(np.min(reference))
            )
        )
        return constraint

    def configure_thickness_constraint(self):
        if self._thickness_constraint_configured:
            return self.thickness_constraint
        self._thickness_constraint_configured = True
        self.thickness_constraint = self._build_bspline_thickness_constraint()
        return self.thickness_constraint

    def _thickness_values_for_physical(self, coefficients):
        if self.thickness_constraint is None:
            return None
        return np.asarray(self.thickness_constraint.values(coefficients), dtype=float)

    def _thickness_history_info(self, coefficients):
        values = self._thickness_values_for_physical(coefficients)
        if values is None:
            return {}
        min_value = float(np.min(values)) if len(values) else 0.0
        return {
            "min_thickness_constraint": min_value,
            "thickness_constraint_active": 1 if min_value <= 1.0e-10 else 0,
        }

    def _thickness_constraint_functions(self):
        if self.thickness_constraint is None:
            return []

        def thickness_fun(variables):
            reduced_trial = self.optimizer_to_physical(variables)
            physical_trial = self.expand_reduced_physical(reduced_trial)
            physical_eval, _info = self._apply_line_search_bound(physical_trial)
            return self._thickness_values_for_physical(physical_eval)

        def thickness_jac(variables):
            variables = np.asarray(variables, dtype=float)
            if self.thickness_constraint.gradient_mode == "FINITE_DIFFERENCE":
                return self._thickness_jacobian_fd_optimizer(variables, thickness_fun)

            reduced_trial = self.optimizer_to_physical(variables)
            physical_trial = self.expand_reduced_physical(reduced_trial)
            physical_eval, info = self._apply_line_search_bound(physical_trial)
            beta = float(info.get("line_search_beta", 1.0))
            try:
                jac_a = self.thickness_constraint.jacobian_analytic(physical_eval)
            except Exception as exc:
                if self.thickness_constraint.gradient_mode == "ANALYTIC":
                    raise
                if not self._thickness_fallback_warned:
                    print(
                        "[BSPLINE_SU2_DRIVER] WARNING: analytic B-spline thickness gradient unavailable; "
                        f"falling back to finite differences ({exc})"
                    )
                    self._thickness_fallback_warned = True
                return self._thickness_jacobian_fd_optimizer(variables, thickness_fun)
            jac_p = self.collapse_jacobian_to_reduced(jac_a)
            return np.asarray(jac_p, dtype=float) * self.opt_relax_factor * beta

        return [{"type": "ineq", "fun": thickness_fun, "jac": thickness_jac}]

    def _run_su2_geo_function_value(self, coefficients, function_name):
        function_name = normalize_geometry_constraint_name(function_name)
        eval_id, paths = self._next_paths(objective_adjoint=f"geo_{function_name}")
        # SU2_GEO only needs the deformed mesh; no adjoint is solved here, so prepare
        # only the shared deform/direct files (avoids a stray, unused adjoint dir).
        self._prepare_shared_files(coefficients, paths)

        geo_dir = paths.eval_dir / "geometry"
        geo_dir.mkdir(parents=True, exist_ok=True)
        geo_cfg = geo_dir / "geo.cfg"
        geo_log = geo_dir / "su2_geo.log"
        geo_value = geo_dir / "of_func.csv"

        patch_config_template(
            self.primal_template,
            geo_cfg,
            {
                "MESH_FILENAME": _relative_path(paths.deformed_mesh, geo_dir),
                "GEO_PARAM": function_name,
                "GEO_MODE": "FUNCTION",
                "VALUE_OBJFUNC_FILENAME": "of_func",
                "GRAD_OBJFUNC_FILENAME": "of_grad",
                "TABULAR_FORMAT": "CSV",
                "CONSOLE": "NONE",
            },
        )

        commands = build_eval_commands(
            paths,
            self.base_mesh,
            self.marker,
            mpi_prefix=self.mpi_prefix,
            python_executable=self.python_executable,
            sensitivity_weighting=self.sensitivity_weighting,
            surface_mode=self.surface_mode,
            deformation_direction_mode=self.deformation_direction_mode,
            le_safe_direction=self.le_safe_direction_options["le_safe_direction"],
            le_safe_x0=self.le_safe_direction_options["le_safe_x0"],
            le_safe_x1=self.le_safe_direction_options["le_safe_x1"],
            le_safe_power=self.le_safe_direction_options["le_safe_power"],
            sensitivity_source=self.sensitivity_source,
        )
        geo_command = _with_mpi(["SU2_GEO", geo_cfg.name], self.mpi_prefix)
        for stage, cwd, command in (
            ("geometry_bspline_def", paths.deform_dir, commands["bspline_def"]),
            ("geometry_su2_def", paths.deform_dir, commands["def"]),
            ("geometry_su2_geo", geo_dir, geo_command),
        ):
            _append_command_log(paths.commands_log, stage, cwd, command)
            _append_command_log(self.workdir / "commands.log", stage, cwd, command)

        run_command(
            commands["bspline_def"],
            paths.deform_dir,
            paths.bspline_def_log,
            show_command=False,
            stream_output=self.stream_solver_output,
            stage="geometry_bspline_def",
        )
        run_command(
            commands["def"],
            paths.deform_dir,
            paths.su2_def_log,
            show_command=False,
            stream_output=self.stream_solver_output,
            stage="geometry_su2_def",
        )
        run_command(
            geo_command,
            geo_dir,
            geo_log,
            show_command=False,
            stream_output=self.stream_solver_output,
            stage="geometry_su2_geo",
        )
        if not geo_value.exists():
            raise BSplineSU2DriverError(
                f"SU2_GEO completed but did not write {geo_value}"
            )
        value = read_objective_from_history(geo_value, function_name)
        return {
            "eval_id": eval_id,
            "eval_dir": str(paths.eval_dir),
            "geo_dir": str(geo_dir),
            "value": value,
        }

    def _geometry_backend_for(self, function_name):
        function_name = normalize_geometry_constraint_name(function_name)
        mode = self.geometry_constraint_gradient_mode
        supported = function_name in SUPPORTED_BSPLINE_GEOMETRY_CONSTRAINTS
        if function_name == "AIRFOIL_AREA" and self.surface_mode != "BOTH":
            raise BSplineSU2DriverError(
                "AIRFOIL_AREA requires BSPLINE_SURFACE_MODE=BOTH"
            )
        if mode == "SU2_GEO":
            return "SU2_GEO"
        if not supported:
            if mode == "ANALYTIC":
                raise BSplineSU2DriverError(
                    f"analytic geometry constraint gradient is unsupported for {function_name}"
                )
            return "SU2_GEO"
        if mode == "ANALYTIC":
            if self.deformation_direction_mode != "VERTICAL":
                raise BSplineSU2DriverError(
                    "analytic geometry constraint gradient requires "
                    "BSPLINE_DEFORMATION_DIRECTION=VERTICAL"
                )
            if function_name == "AIRFOIL_AREA":
                return "ANALYTIC"
            if function_name == "AIRFOIL_THICKNESS":
                if self.surface_mode != "BOTH":
                    raise BSplineSU2DriverError(
                        "analytic AIRFOIL_THICKNESS requires BSPLINE_SURFACE_MODE=BOTH"
                    )
                if not self._marker_closed(self.marker):
                    raise BSplineSU2DriverError(
                        "analytic AIRFOIL_THICKNESS requires a closed B-spline marker"
                    )
                return "ANALYTIC"
        if mode == "AUTO":
            if self.deformation_direction_mode != "VERTICAL":
                return "SU2_GEO"
            if function_name == "AIRFOIL_AREA":
                return "ANALYTIC"
            if (
                function_name == "AIRFOIL_THICKNESS"
                and self.surface_mode == "BOTH"
                and self._marker_closed(self.marker)
            ):
                return "ANALYTIC"
        return "SU2_GEO"

    def _geometry_metric_for(self, function_name):
        function_name = normalize_geometry_constraint_name(function_name)
        if function_name not in SUPPORTED_BSPLINE_GEOMETRY_CONSTRAINTS:
            return None
        metric = self._geometry_metric_cache.get(function_name)
        if metric is not None:
            return metric
        metadata, basis_matrix = self._probe_geometry_aware_bounds()
        marker = self.marker
        closed = self._marker_closed(marker)
        if function_name == "AIRFOIL_AREA":
            if self.surface_mode != "BOTH":
                raise BSplineSU2DriverError(
                    "AIRFOIL_AREA requires BSPLINE_SURFACE_MODE=BOTH"
                )
            metric = BSplineAirfoilAreaMetric(
                metadata,
                basis_matrix,
                self.mode_ids,
                closed=closed,
            )
        elif function_name == "AIRFOIL_THICKNESS":
            x_stations_value = (self.thickness_options or {}).get(
                "PROGRESSIVE_THICKNESS_X_STATIONS"
            )
            if not _x_stations_value_is_empty(x_stations_value):
                x_stations = _parse_x_stations(x_stations_value)
            else:
                x_stations = np.linspace(0.001, 0.999, 101)
            metric = BSplineAirfoilMaxThicknessMetric(
                metadata,
                basis_matrix,
                self.mode_ids,
                x_stations=x_stations,
                fd_eps=self.geometry_fd_eps,
                gradient_mode="ANALYTIC",
                closed=closed,
                marker=marker,
            )
        else:
            metric = None
        self._geometry_metric_cache[function_name] = metric
        return metric

    def evaluate_geometry_constraint_function(self, coefficients, function_name):
        coefficients = [_as_float(value, "coefficient") for value in coefficients]
        if len(coefficients) != len(self.mode_ids):
            raise BSplineSU2DriverError(
                f"expected {len(self.mode_ids)} coefficients, got {len(coefficients)}"
            )
        function_name = normalize_geometry_constraint_name(function_name)
        key = (cache_key(coefficients, self.cache_tol), _normalized_name(function_name))
        if key in self._constraint_cache:
            return self._constraint_cache[key]

        backend = self._geometry_backend_for(function_name)
        if backend == "ANALYTIC":
            metric = self._geometry_metric_for(function_name)
            if metric is None:
                raise BSplineSU2DriverError(
                    f"analytic geometry constraint gradient is unsupported for {function_name}"
                )
            value, gradient = metric.value_and_gradient(coefficients)
            result = {
                "eval_id": None,
                "eval_dir": None,
                "function": function_name,
                "value": float(value),
                "gradient": [float(item) for item in gradient],
                "coefficients": coefficients,
                "status": "ok",
                "source": "ANALYTIC",
            }
            self._append_geometry_constraint_record(result)
            self._constraint_cache[key] = result
            return result

        base = self._run_su2_geo_function_value(coefficients, function_name)
        value = float(base["value"])
        gradient = []
        for index in range(len(coefficients)):
            perturbed = list(coefficients)
            perturbed[index] += float(self.geometry_fd_eps)
            perturbed_value = float(
                self._run_su2_geo_function_value(perturbed, function_name)["value"]
            )
            gradient.append((perturbed_value - value) / float(self.geometry_fd_eps))

        result = {
            "eval_id": base["eval_id"],
            "eval_dir": base["eval_dir"],
            "function": function_name,
            "value": value,
            "gradient": [float(item) for item in gradient],
            "coefficients": coefficients,
            "status": "ok",
            "source": "SU2_GEO",
            "geometry_fd_eps": float(self.geometry_fd_eps),
        }
        self._append_geometry_constraint_record(result)
        self._constraint_cache[key] = result
        return result

    def _native_constraint_result_for_variables(self, variables, spec):
        reduced_trial = self.optimizer_to_physical(variables)
        physical_trial = self.expand_reduced_physical(reduced_trial)
        physical_eval, info = self._apply_line_search_bound(physical_trial)
        if is_geometry_constraint_name(spec.name):
            result = self.evaluate_geometry_constraint_function(physical_eval, spec.name)
        else:
            result = self.evaluate_constraint_function(physical_eval, spec.name)
        return result, info

    def _native_constraint_value(self, value, spec):
        factor = float(spec.scale) * float(self.opt_gradient_factor)
        delta = float(value) - float(spec.target)
        if spec.sign == "=":
            return delta * factor
        if spec.sign == "<":
            return -delta * factor
        if spec.sign == ">":
            return delta * factor
        raise BSplineSU2DriverError(
            f"unsupported OPT_CONSTRAINT sign {spec.sign!r} for {spec.name}"
        )

    def _native_constraint_gradient_factor(self, spec):
        # Match SU2.eval.design: constraint values are multiplied by SCALE,
        # while con_dceq/con_dcieq apply OPT_GRADIENT_FACTOR and sign but not
        # the per-constraint SCALE.
        factor = float(self.opt_gradient_factor)
        if spec.sign == "=":
            return factor
        if spec.sign == "<":
            return -factor
        if spec.sign == ">":
            return factor
        raise BSplineSU2DriverError(
            f"unsupported OPT_CONSTRAINT sign {spec.sign!r} for {spec.name}"
        )

    def _native_constraint_functions(self):
        constraints = []
        for spec in self.native_constraints:

            def constraint_fun(variables, spec=spec):
                result, _info = self._native_constraint_result_for_variables(
                    variables,
                    spec,
                )
                return self._native_constraint_value(result["value"], spec)

            def constraint_jac(variables, spec=spec):
                result, info = self._native_constraint_result_for_variables(
                    variables,
                    spec,
                )
                beta = float(info.get("line_search_beta", 1.0))
                reduced_gradient = self.collapse_gradient_to_reduced(result["gradient"])
                factor = (
                    self._native_constraint_gradient_factor(spec)
                    * float(self.opt_relax_factor)
                    * beta
                )
                return np.asarray(
                    [float(value) * factor for value in reduced_gradient],
                    dtype=float,
                )

            constraints.append(
                {
                    "type": "eq" if spec.sign == "=" else "ineq",
                    "fun": constraint_fun,
                    "jac": constraint_jac,
                }
            )
        return constraints

    def _thickness_jacobian_fd_optimizer(self, variables, thickness_fun):
        variables = np.asarray(variables, dtype=float)
        g0 = np.asarray(thickness_fun(variables), dtype=float)
        jac = np.zeros((len(g0), len(variables)), dtype=float)
        eps = float(self.thickness_constraint.fd_eps)
        for j in range(len(variables)):
            trial = variables.copy()
            trial[j] += eps
            jac[:, j] = (np.asarray(thickness_fun(trial), dtype=float) - g0) / eps
        return jac

    def _local_step_limit_info(self, physical_trial):
        info = {
            "local_step_beta": 1.0,
            "local_step_limited": 0,
            "local_step_limiting_mode": "",
            "local_step_da": 0.0,
            "local_step_limit": 0.0,
        }
        if not self.local_step_limit:
            return info
        reduced_trial = self.compress_full_physical(physical_trial)
        anchor = np.asarray(self._local_step_anchor_reduced, dtype=float)
        trial = np.asarray(reduced_trial, dtype=float)
        delta = trial - anchor
        beta = 1.0
        limiting_index = None
        limiting_da = 0.0
        limiting_limit = 0.0
        eps = 1.0e-30
        for index, (da, limit) in enumerate(zip(delta, self.reduced_local_step_limits)):
            limit = float(limit)
            if not math.isfinite(limit) or limit <= 0.0:
                continue
            abs_da = abs(float(da))
            if abs_da > limit and abs_da > eps:
                candidate_beta = limit / abs_da
                if candidate_beta < beta:
                    beta = candidate_beta
                    limiting_index = index
                    limiting_da = float(da)
                    limiting_limit = limit
        if limiting_index is not None:
            variable = self.reduced_variables[limiting_index]
            info.update(
                {
                    "local_step_beta": float(beta),
                    "local_step_limited": 1,
                    "local_step_limiting_mode": ",".join(variable.mode_ids),
                    "local_step_da": limiting_da,
                    "local_step_limit": limiting_limit,
                }
            )
        return info

    def _apply_line_search_bound(self, physical_trial):
        physical_trial = [_as_float(value, "coefficient") for value in physical_trial]
        if len(physical_trial) != len(self.mode_ids):
            raise BSplineSU2DriverError(
                f"expected {len(self.mode_ids)} coefficients, got {len(physical_trial)}"
            )
        info = self._line_search_default_info()

        local_info = self._local_step_limit_info(physical_trial)
        info.update(local_info)

        if self.opt_line_search_bound is not None and self._line_search_basis_matrix is None:
            self._configure_line_search_bound()

        anchor = np.asarray(self._line_search_anchor_physical, dtype=float)
        trial = np.asarray(physical_trial, dtype=float)
        delta = trial - anchor

        geometry_beta = 1.0
        maxdiff = 0.0
        geometry_limited = 0
        if self.opt_line_search_bound is not None:
            basis_matrix = self._line_search_basis_matrix
            delta_dn = basis_matrix.dot(delta)
            maxdiff = float(np.max(np.abs(delta_dn))) if len(delta_dn) else 0.0
            if maxdiff > float(self.opt_line_search_bound) and maxdiff > 0.0:
                geometry_beta = float(self.opt_line_search_bound) / maxdiff
                geometry_limited = 1

        beta = min(float(geometry_beta), float(info["local_step_beta"]))
        limited = int(geometry_limited or info["local_step_limited"])
        if beta < 1.0:
            trial = anchor + beta * delta
        info.update(
            {
                "line_search_beta": float(beta),
                "line_search_beta_geometry": float(geometry_beta),
                "line_search_maxdiff": float(maxdiff),
                "line_search_limited": int(limited),
            }
        )
        return [float(value) for value in trial], info

    def _update_line_search_anchor_from_optimizer_variables(self, variables):
        reduced_trial = self.optimizer_to_physical(variables)
        physical_trial = self.expand_reduced_physical(reduced_trial)
        physical_eval, _info = self._apply_line_search_bound(physical_trial)
        self._line_search_anchor_physical = list(physical_eval)
        self._local_step_anchor_reduced = self.compress_full_physical(physical_eval)

    def _optimizer_gradient_for_logging(self, raw_gradient, line_search_info=None):
        info = line_search_info or {}
        beta = float(info.get("line_search_beta", 1.0))
        values = [float(value) for value in raw_gradient]
        reduced = []
        for variable in self.reduced_variables:
            reduced.append(
                sum(
                    float(sign) * values[int(index)]
                    for index, sign in zip(variable.mode_indices, variable.signs)
                )
            )
        return np.asarray(reduced, dtype=float) * (
            self.opt_relax_factor * self.opt_gradient_factor * beta
        )

    def _gradient_entry(self, result, paths, line_search_info=None):
        info = {
            **self._line_search_default_info(),
            **(line_search_info or {}),
        }
        raw_gradient = np.asarray(result.get("gradient") or [], dtype=float)
        optimizer_gradient = self._optimizer_gradient_for_logging(
            raw_gradient,
            info,
        )
        evaluated = [float(value) for value in result.get("coefficients", [])]
        requested = [
            float(value)
            for value in info.get("requested_x", evaluated)
        ]
        beta_eff = float(info.get("line_search_beta", 1.0))
        return {
            "eval_id": int(result.get("eval_id", -1)),
            "objective": float(result["objective"]),
            "requested_x": requested,
            "evaluated_x": evaluated,
            "beta_eff": beta_eff,
            "was_clipped": bool(
                int(info.get("line_search_limited", 0)) or beta_eff < 1.0
            ),
            "gnorm_raw": float(np.linalg.norm(raw_gradient)),
            "gnorm_opt": float(np.linalg.norm(optimizer_gradient)),
            "eval_dir": Path(paths.eval_dir),
            "modes_file": Path(paths.modes_current),
        }

    def restore_modes_from_entry(self, entry):
        self.optimized_modes_filename.parent.mkdir(parents=True, exist_ok=True)
        source = None if entry is None else entry.get("modes_file")
        if source is not None and Path(source).exists():
            source = Path(source)
            if source.resolve() != self.optimized_modes_filename.resolve():
                shutil.copy2(source, self.optimized_modes_filename)
        else:
            self.write_optimized_modes(self.initial_coefficients)
        return self.optimized_modes_filename

    def _log_gradient_guard_stop(self, bad_entry, guard_info):
        rollback = self.best_safe_entry or self.last_safe_entry
        safe_eval = (
            None
            if rollback is None
            else rollback.get("eval_id")
        )
        print("GRADIENT_GUARD_STOP")
        print(f"  reason          = {guard_info.get('reason')}")
        print(f"  bad_eval        = {bad_entry.get('eval_id')}")
        print(f"  restore_eval    = {safe_eval}")
        print(f"  gnorm_raw_bad   = {bad_entry.get('gnorm_raw')}")
        print(f"  gnorm_raw_ref   = {guard_info.get('reference')}")
        print(f"  raw_ratio       = {guard_info.get('ratio')}")
        print(f"  gnorm_opt_bad   = {bad_entry.get('gnorm_opt')}")
        print(f"  beta_eff_bad    = {bad_entry.get('beta_eff')}")
        print("  action          = rollback_to_best_safe")
        print(f"  next_action     = {self.gradient_guard_next_action}")

    def _promote_safe_entry(self, entry, update_recent_raw_gnorm=True):
        self.last_safe_entry = entry
        if update_recent_raw_gnorm:
            self.recent_safe_raw_gnorms.append(float(entry["gnorm_raw"]))
        objective = float(entry["objective"])
        if np.isfinite(objective) and (
            self.best_safe_entry is None
            or objective < float(self.best_safe_entry["objective"])
        ):
            self.best_safe_entry = entry
            self.best_physical_entry = entry
        if self.anchor_entry is None:
            self.anchor_entry = entry

    def register_gradient_entry(self, entry, promote=True):
        triggered = False
        guard_info = {
            "reason": "disabled",
            "gnorm_raw": float(entry["gnorm_raw"]),
            "reference": None,
            "ratio": None,
        }
        if self.gradient_guard_enabled:
            triggered, guard_info = gradient_guard_triggered(
                entry,
                self.recent_safe_raw_gnorms,
                factor=self.gradient_guard_factor,
                window=self.gradient_guard_window,
                min_history=self.gradient_guard_min_history,
                floor=self.gradient_guard_floor,
            )
        if triggered:
            rollback_entry = self.best_safe_entry or self.last_safe_entry
            self.restore_modes_from_entry(rollback_entry)
            self._log_gradient_guard_stop(entry, guard_info)
            stop = GradientGuardStop(
                rollback_entry,
                bad_entry=entry,
                guard_info=guard_info,
            )
            self.last_gradient_guard_stop = stop
            raise stop

        if promote:
            self._promote_safe_entry(entry)
        return guard_info

    def _trust_clip_enabled(self):
        return self.trust_clip_options["policy"] == "ACCEPT_RESTART"

    def _requested_optimizer_key(self, variables):
        return cache_key(variables, self.cache_tol)

    def _line_search_anchor_key(self):
        return cache_key(list(self._line_search_anchor_physical), self.cache_tol)

    def _pending_trust_clip_key(self, optimizer_variables):
        return (
            self._requested_optimizer_key(optimizer_variables),
            self._line_search_anchor_key(),
        )

    def _prune_stale_trust_clip_pending(self):
        current_anchor_key = self._line_search_anchor_key()
        for key in list(self._trust_clip_by_requested_key):
            _request_key, anchor_key = key
            if anchor_key != current_anchor_key:
                del self._trust_clip_by_requested_key[key]

    def _classify_trust_clip_entry(self, entry):
        classification, diagnostics = classify_clipped_trial(
            entry,
            self.recent_safe_raw_gnorms,
            self.best_safe_entry,
            self.anchor_entry,
            self.recent_level_clip_events,
            self.trust_clip_options,
        )
        event = {
            "classification": classification,
            "clipped": bool(diagnostics["clipped"]),
            "weak_improvement": bool(diagnostics["weak_improvement"]),
            "toxic": classification == "rejected_toxic_clip",
            "clipped_stagnation_plateau": bool(
                diagnostics["clipped_stagnation_plateau"]
            ),
        }
        self.recent_level_clip_events.append(event)
        return classification, diagnostics

    def _trust_clip_status(self, classification):
        return {
            "not_clipped": "ok",
            "benign_clipped_legacy": "ok_clipped_benign",
            "weak_clipped_progress": "ok_clipped_weak",
            "accepted_clipped_restart": "accepted_clipped_restart",
            "rejected_toxic_clip": "rejected_toxic_clip",
        }[classification]

    def _log_trust_clip_stop(self, stop):
        diagnostics = stop.diagnostics
        print("TRUST_CLIP_STOP")
        print(f"  class           = {stop.classification}")
        print(f"  eval             = {stop.entry.get('eval_id')}")
        print(
            "  restore_eval     = "
            f"{None if stop.rollback_entry is None else stop.rollback_entry.get('eval_id')}"
        )
        print(f"  beta_eff         = {diagnostics.get('beta_eff')}")
        print(f"  improvement_rel  = {diagnostics.get('improvement_rel')}")
        print(f"  relative_worsen  = {diagnostics.get('relative_worsening')}")
        print(f"  gnorm_ratio      = {diagnostics.get('gnorm_ratio')}")
        print(f"  reasons          = {','.join(diagnostics.get('toxic_reasons', []))}")
        print(f"  action           = {stop.action}")

    def _trust_clip_callback(self, optimizer_variables):
        if not self._trust_clip_enabled():
            self._update_line_search_anchor_from_optimizer_variables(optimizer_variables)
            return
        key = self._pending_trust_clip_key(optimizer_variables)
        pending = self._trust_clip_by_requested_key.pop(key, None)
        if pending is None:
            self._update_line_search_anchor_from_optimizer_variables(optimizer_variables)
            self._prune_stale_trust_clip_pending()
            return
        classification = pending["classification"]
        entry = pending["entry"]
        diagnostics = pending["diagnostics"]
        if classification in (
            "not_clipped",
            "benign_clipped_legacy",
            "weak_clipped_progress",
        ):
            self._update_line_search_anchor_from_optimizer_variables(optimizer_variables)
            self.anchor_entry = entry
            self._prune_stale_trust_clip_pending()
            return
        if classification == "accepted_clipped_restart":
            self._promote_safe_entry(entry)
            self._update_line_search_anchor_from_optimizer_variables(optimizer_variables)
            self.anchor_entry = entry
            rollback_entry = entry
            action = "restart_from_evaluated"
        else:
            rollback_entry = self.best_safe_entry or self.last_safe_entry
            toxic_reasons = set(diagnostics.get("toxic_reasons", []))
            force_refine = bool(
                self.refinement_available
                and toxic_reasons.intersection(
                    {"toxic_clipped_repeated", "clipped_stagnation_plateau"}
                )
            )
            action = "refine" if force_refine else "rollback_best_safe_restart"
        self._prune_stale_trust_clip_pending()
        self.restore_modes_from_entry(rollback_entry)
        stop = TrustClipStop(
            classification,
            entry,
            rollback_entry,
            diagnostics=diagnostics,
            action=action,
        )
        self.last_trust_clip_stop = stop
        self._log_trust_clip_stop(stop)
        raise stop

    def _evaluate_optimizer_variables(self, variables):
        reduced_trial = self.optimizer_to_physical(variables)
        physical_trial = self.expand_reduced_physical(reduced_trial)
        physical_eval, info = self._apply_line_search_bound(physical_trial)
        info["requested_x"] = list(physical_trial)
        info["evaluated_x"] = list(physical_eval)
        info["requested_optimizer_x"] = [float(value) for value in variables]
        physical_key = cache_key(physical_eval, self.cache_tol)
        beta_tol = float(self.trust_clip_options.get("beta_tol", 1.0e-12))
        beta_now = float(info.get("beta_eff", info.get("line_search_beta", 1.0)))
        if (
            self._last_eval_physical_key == physical_key
            and self._last_eval_result is not None
            and self._last_eval_info is not None
        ):
            beta_prev = float(
                self._last_eval_info.get(
                    "beta_eff",
                    self._last_eval_info.get("line_search_beta", 1.0),
                )
            )
            prev_class = str(self._last_eval_result.get("trust_clip_class", ""))
            if (
                beta_prev >= 1.0 - beta_tol
                and beta_now >= 1.0 - beta_tol
                and prev_class not in LAST_EVAL_CACHE_BLOCKED_TRUST_CLIP_CLASSES
            ):
                cached_info = dict(self._last_eval_info)
                cached_info["cache_hit"] = True
                cached_info["last_eval_cache_hit"] = True
                cached_info["requested_optimizer_x"] = [float(value) for value in variables]
                cached_info["requested_x"] = list(physical_trial)
                cached_info["evaluated_x"] = list(physical_eval)
                return self._last_eval_result, cached_info
        if self._trust_clip_enabled():
            pending = self._trust_clip_by_requested_key.get(
                self._pending_trust_clip_key(variables)
            )
            if pending is not None:
                return pending["result"], info
        result = self.evaluate(physical_eval, line_search_info=info)
        self._last_eval_physical_key = physical_key
        self._last_eval_result = result
        self._last_eval_info = dict(info)
        return result, info

    def _evaluate_reduced_physical(self, reduced_coefficients):
        physical_trial = self.expand_reduced_physical(reduced_coefficients)
        physical_eval, info = self._apply_line_search_bound(physical_trial)
        info["requested_x"] = list(physical_trial)
        info["evaluated_x"] = list(physical_eval)
        result = self.evaluate(physical_eval, line_search_info=info)
        reduced_eval = self.compress_full_physical(physical_eval)
        return result, info, reduced_eval

    def _initial_eval_id(self):
        ids = []
        for path in self.workdir.glob("eval_[0-9][0-9][0-9][0-9]"):
            try:
                ids.append(int(path.name.split("_", 1)[1]))
            except Exception:
                pass
        return max(ids) + 1 if ids else 0

    @property
    def optimization_history_filename(self):
        return self.workdir / "optimization_history.csv"

    @property
    def optimized_modes_filename(self):
        return self.workdir / "optimized_modes.json"

    @property
    def geometry_constraint_history_filename(self):
        return self.workdir / "geometry_constraints.csv"

    def _append_geometry_constraint_record(self, result):
        fieldnames = [
            "function",
            "source",
            "value",
            "gradient",
            "coefficients",
            "eval_id",
            "eval_dir",
            "geometry_fd_eps",
        ]
        filename = self.geometry_constraint_history_filename
        write_header = not filename.exists()
        with open(filename, "a", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "function": result.get("function", ""),
                    "source": result.get("source", ""),
                    "value": _format_config_atom(result.get("value", "")),
                    "gradient": json.dumps(result.get("gradient", [])),
                    "coefficients": json.dumps(result.get("coefficients", [])),
                    "eval_id": ""
                    if result.get("eval_id") is None
                    else result.get("eval_id"),
                    "eval_dir": ""
                    if result.get("eval_dir") is None
                    else result.get("eval_dir"),
                    "geometry_fd_eps": _format_config_atom(
                        result.get("geometry_fd_eps", "")
                    ),
                }
            )

    def _next_paths(self, objective_adjoint=None):
        objective_adjoint = (
            self.objective_adjoint
            if objective_adjoint is None
            else _normalize_objective_adjoint(objective_adjoint)
        )
        while True:
            eval_id = self._next_eval_id
            self._next_eval_id += 1
            paths = build_eval_paths(
                self.workdir / f"eval_{eval_id:04d}",
                eval_layout=self.eval_layout,
                objective_adjoint=objective_adjoint,
            )
            if not paths.eval_dir.exists():
                return eval_id, paths

    def _shared_primal_updates(self, paths):
        return {
            "SOLUTION_FILENAME": paths.primal_solution.name,
            "RESTART_FILENAME": paths.primal_restart.name,
            "SOLUTION_ADJ_FILENAME": paths.adjoint_solution.name,
            "RESTART_ADJ_FILENAME": paths.adjoint_restart.name,
            "SURFACE_ADJ_FILENAME": paths.surface_adjoint.stem,
            "VOLUME_ADJ_FILENAME": paths.volume_adjoint.name,
            "TABULAR_FORMAT": "CSV",
        }

    def _prepare_shared_files(self, coefficients, paths):
        paths.eval_dir.mkdir(parents=True, exist_ok=True)
        for directory in {paths.deform_dir, paths.direct_dir}:
            directory.mkdir(parents=True, exist_ok=True)
        current_spec = update_mode_coefficients(self.mode_spec, coefficients)
        write_mode_spec(current_spec, paths.modes_current)

        patch_config_template(
            self.def_template,
            paths.def_cfg,
            {
                "MESH_FILENAME": str(self.base_mesh),
                "MESH_OUT_FILENAME": paths.deformed_mesh.name,
                "DV_KIND": "SURFACE_FILE",
                "DV_MARKER": [self.marker],
                "DV_FILENAME": paths.surface_positions.name,
            },
        )

        primal_mesh_filename = _relative_path(paths.deformed_mesh, paths.direct_dir)
        patch_config_template(
            self.primal_template,
            paths.primal_cfg,
            dict(
                self._shared_primal_updates(paths),
                MESH_FILENAME=primal_mesh_filename,
                MESH_OUT_FILENAME=paths.primal_mesh_out.name,
                CONV_FILENAME=paths.primal_history.stem,
                HISTORY_OUTPUT=["ITER", "RMS_RES", "AERO_COEFF"],
                SCREEN_OUTPUT=["INNER_ITER", "RMS_RES", "LIFT", "DRAG"],
            ),
        )

    def _prepare_func_adjoint_files(self, paths, objective_function=None):
        paths.eval_dir.mkdir(parents=True, exist_ok=True)
        paths.adjoint_dir.mkdir(parents=True, exist_ok=True)
        adjoint_mesh_filename = _relative_path(paths.deformed_mesh, paths.adjoint_dir)
        adjoint_flow_solution = _relative_path(paths.primal_solution, paths.adjoint_dir)
        adjoint_flow_restart = _relative_path(paths.primal_restart, paths.adjoint_dir)

        objective_updates = {}
        if objective_function:
            objective_updates["OBJECTIVE_FUNCTION"] = str(objective_function).strip().upper()
        patch_config_template(
            self.adjoint_template,
            paths.adjoint_cfg,
            dict(
                self._shared_primal_updates(paths),
                **objective_updates,
                MESH_FILENAME=adjoint_mesh_filename,
                SOLUTION_FILENAME=adjoint_flow_solution,
                RESTART_FILENAME=adjoint_flow_restart,
                SURFACE_ADJ_FILENAME=paths.surface_adjoint.stem,
                VOLUME_ADJ_FILENAME=paths.volume_adjoint.name,
                MESH_OUT_FILENAME=paths.adjoint_mesh_out.name,
                CONV_FILENAME=paths.adjoint_history.stem,
            ),
        )
        patch_config_template(
            self.adjoint_template,
            paths.dot_ad_cfg,
            dict(
                self._shared_primal_updates(paths),
                **objective_updates,
                MATH_PROBLEM="DISCRETE_ADJOINT",
                MESH_FILENAME=adjoint_mesh_filename,
                SOLUTION_FILENAME=adjoint_flow_solution,
                RESTART_FILENAME=adjoint_flow_restart,
                SOLUTION_ADJ_FILENAME=paths.adjoint_solution.name,
                RESTART_ADJ_FILENAME=paths.adjoint_restart.name,
                DV_KIND="SURFACE_FILE",
                DV_MARKER=[self.marker],
                DV_FILENAME=_relative_path(paths.surface_positions, paths.adjoint_dir),
                SURFACE_ADJ_FILENAME=paths.surface_adjoint.stem,
                VOLUME_ADJ_FILENAME=paths.volume_adjoint.name,
                SURFACE_SENS_FILENAME=paths.surface_sens.stem,
                VOLUME_SENS_FILENAME=paths.volume_sens.name,
                OUTPUT_FILES=["SURFACE_CSV"],
                TABULAR_FORMAT="CSV",
                OUTPUT_PRECISION=15,
                MESH_OUT_FILENAME=paths.adjoint_mesh_out.name,
                CONV_FILENAME=paths.adjoint_history.stem,
            ),
        )

    def _prepare_eval_files(self, coefficients, paths, objective_function=None):
        self._prepare_shared_files(coefficients, paths)
        self._prepare_func_adjoint_files(paths, objective_function=objective_function)

    def _build_commands(self, paths):
        return build_eval_commands(
            paths,
            self.base_mesh,
            self.marker,
            mpi_prefix=self.mpi_prefix,
            python_executable=self.python_executable,
            sensitivity_weighting=self.sensitivity_weighting,
            surface_mode=self.surface_mode,
            deformation_direction_mode=self.deformation_direction_mode,
            le_safe_direction=self.le_safe_direction_options["le_safe_direction"],
            le_safe_x0=self.le_safe_direction_options["le_safe_x0"],
            le_safe_x1=self.le_safe_direction_options["le_safe_x1"],
            le_safe_power=self.le_safe_direction_options["le_safe_power"],
            sensitivity_source=self.sensitivity_source,
        )

    def _log_eval_commands(self, paths, commands, stages):
        stage_cwds = {
            "bspline_def": paths.deform_dir,
            "def": paths.deform_dir,
            "primal": paths.direct_dir,
            "adjoint": paths.adjoint_dir,
            "dot_ad": paths.adjoint_dir,
            "bspline_dot": paths.eval_dir,
        }
        for stage in stages:
            if stage not in commands:
                continue
            cwd = stage_cwds[stage]
            _append_command_log(paths.commands_log, stage, cwd, commands[stage])
            _append_command_log(self.workdir / "commands.log", stage, cwd, commands[stage])

    def _maybe_print_commands_log_path(self):
        if self.show_commands and not self._printed_commands_log_path:
            print(
                "[BSPLINE_SU2_DRIVER] Commands are logged in {}".format(
                    self.workdir / "commands.log"
                )
            )
            self._printed_commands_log_path = True

    def _function_record_key(self, function_name):
        return _normalized_name(function_name).upper()

    def _history_column_record_key(self, column_name):
        return "COLUMN:" + _normalized_name(column_name).upper()

    def _read_primal_values(self, paths):
        values = {}
        for function_name, column_name in FUNCTION_HISTORY_COLUMNS.items():
            try:
                value = read_objective_from_history(paths.primal_history, column_name)
            except BSplineSU2DriverError:
                continue
            values[self._function_record_key(function_name)] = value
            values[self._history_column_record_key(column_name)] = value
        try:
            values[self._history_column_record_key(self.objective_column)] = (
                read_objective_from_history(paths.primal_history, self.objective_column)
            )
        except BSplineSU2DriverError:
            pass
        return values

    def _primal_value_for_function(self, primal_values, function_name):
        function_key = self._function_record_key(function_name)
        if function_key in primal_values:
            return primal_values[function_key]
        column_key = self._history_column_record_key(
            history_column_for_function(function_name)
        )
        if column_key in primal_values:
            return primal_values[column_key]
        raise BSplineSU2DriverError(
            "primal history has no value for function {}".format(function_name)
        )

    def _objective_value_from_primal_values(self, primal_values):
        column_key = self._history_column_record_key(self.objective_column)
        if column_key in primal_values:
            return primal_values[column_key]
        return self._primal_value_for_function(primal_values, self.objective_adjoint)

    def _design_point(self, coefficients, objective_adjoint=None):
        key = cache_key(coefficients, self.cache_tol)
        record = self._design_points.get(key)
        if record is not None:
            return key, record
        eval_id, paths = self._next_paths(objective_adjoint=objective_adjoint)
        record = {
            "eval_id": eval_id,
            "eval_dir": paths.eval_dir,
            "coefficients": list(coefficients),
            "core_done": False,
            "primal_values": {},
            "func_paths": {},
            "func_adjoints": {},
        }
        self._design_points[key] = record
        return key, record

    def _record_paths_for_function(self, record, function_name):
        function_key = self._function_record_key(function_name)
        paths = record["func_paths"].get(function_key)
        if paths is None:
            paths = build_eval_paths(
                record["eval_dir"],
                eval_layout=self.eval_layout,
                objective_adjoint=function_name,
            )
            record["func_paths"][function_key] = paths
        return paths

    def _run_shared_core(self, coefficients, paths, record=None):
        if record is not None and record.get("core_done"):
            return record["primal_values"]

        self._prepare_shared_files(coefficients, paths)
        commands = self._build_commands(paths)
        self._log_eval_commands(paths, commands, ("bspline_def", "def", "primal"))
        self._maybe_print_commands_log_path()

        run_command(
            commands["bspline_def"],
            paths.deform_dir,
            paths.bspline_def_log,
            show_command=False,
            stream_output=self.stream_solver_output,
            stage="bspline_def",
        )
        run_command(
            commands["def"],
            paths.deform_dir,
            paths.su2_def_log,
            show_command=False,
            stream_output=self.stream_solver_output,
            stage="su2_def",
        )
        run_command(
            commands["primal"],
            paths.direct_dir,
            paths.su2_cfd_log,
            show_command=False,
            stream_output=self.stream_solver_output,
            stage="su2_cfd",
        )
        ensure_adjoint_solution_input(paths.primal_restart, paths.primal_solution)
        primal_values = self._read_primal_values(paths)
        if record is not None:
            record["core_done"] = True
            record["primal_values"] = primal_values
        return primal_values

    def _run_function_adjoint(
        self,
        paths,
        function_name,
        record=None,
        allow_nonfinite=False,
    ):
        function_key = self._function_record_key(function_name)
        if record is not None and function_key in record["func_adjoints"]:
            return record["func_adjoints"][function_key]["gradient"]

        self._prepare_func_adjoint_files(paths, objective_function=function_name)
        commands = self._build_commands(paths)
        self._log_eval_commands(paths, commands, ("adjoint", "dot_ad", "bspline_dot"))

        run_command(
            commands["adjoint"],
            paths.adjoint_dir,
            paths.su2_cfd_ad_log,
            show_command=False,
            stream_output=self.stream_solver_output,
            stage="su2_cfd_ad",
        )
        if "dot_ad" in commands:
            run_command(
                commands["dot_ad"],
                paths.adjoint_dir,
                paths.su2_dot_ad_log,
                show_command=False,
                stream_output=self.stream_solver_output,
                stage="su2_dot_ad",
            )
            if not paths.surface_sens.exists():
                raise BSplineSU2DriverError(
                    "SU2_DOT_AD completed but did not write {}; check "
                    "SURFACE_SENS_FILENAME and OUTPUT_FILES in {}".format(
                        paths.surface_sens,
                        paths.dot_ad_cfg,
                    )
                )
        run_command(
            commands["bspline_dot"],
            paths.eval_dir,
            paths.bspline_dot_log,
            show_command=False,
            stream_output=self.stream_solver_output,
            stage="bspline_dot",
        )
        if paths.gradients.exists():
            shutil.copy2(paths.gradients, paths.adjoint_gradients)
        if paths.summary.exists():
            shutil.copy2(paths.summary, paths.adjoint_summary)
        # Root-level aliases (surface_sens.csv, ...) are created once for the
        # objective in evaluate(), so they deterministically reflect the objective
        # rather than whichever function ran last in the shared evaluation.
        gradient = read_gradient_vector(
            paths.gradients,
            self.mode_ids,
            allow_nonfinite=allow_nonfinite,
        )
        if record is not None:
            record["func_adjoints"][function_key] = {
                "gradient": gradient,
                "surface_sens": paths.surface_sens,
            }
        return gradient

    def _write_eval_summary_metadata(
        self,
        paths,
        line_search_info=None,
        objective_function=None,
    ):
        data = {}
        if paths.summary.exists():
            try:
                with open(paths.summary, "r") as fp:
                    data = json.load(fp)
            except Exception:
                data = {}
        line_search_info = {
            **self._line_search_default_info(),
            **(line_search_info or {}),
        }
        data.update(
            {
                "eval_layout": paths.eval_layout,
                "deform_dir": str(paths.deform_dir),
                "direct_dir": str(paths.direct_dir),
                "adjoint_dir": str(paths.adjoint_dir),
                "objective_adjoint": paths.objective_adjoint,
                "objective_function": (
                    str(objective_function).strip().upper()
                    if objective_function
                    else ""
                ),
                "symmetry_coupling": self.symmetry_coupling,
                "surface_mode": self.surface_mode,
                "sensitivity_weighting": self.sensitivity_weighting,
                "sensitivity_source": self.sensitivity_source,
                "gradient_sensitivity_file": str(
                    paths.surface_sens
                    if self.sensitivity_source == "DOT_AD_TRANSFER"
                    else paths.surface_adjoint
                ),
                "n_active_modes": len(self.mode_ids),
                "n_design_variables": len(self.reduced_variable_ids),
                "local_step_limit_enabled": bool(self.local_step_limit),
                "local_step_limit_ratio": float(self.local_step_limit_ratio),
                "local_step_beta": line_search_info.get("local_step_beta", 1.0),
                "local_step_limited": line_search_info.get("local_step_limited", 0),
                "line_search_beta": line_search_info.get("line_search_beta", 1.0),
            }
        )
        with open(paths.summary, "w") as fp:
            json.dump(data, fp, indent=2, sort_keys=True)
            fp.write("\n")

    def _write_gradient_guard_summary(self, paths, entry, guard_info, status):
        data = {}
        if paths.summary.exists():
            try:
                with open(paths.summary, "r") as fp:
                    data = json.load(fp)
            except Exception:
                data = {}
        data["gradient_guard"] = {
            "status": str(status),
            "reason": guard_info.get("reason"),
            "gnorm_raw": entry.get("gnorm_raw"),
            "gnorm_opt": entry.get("gnorm_opt"),
            "reference": guard_info.get("reference"),
            "ratio": guard_info.get("ratio"),
            "beta_eff": entry.get("beta_eff"),
            "was_clipped": bool(entry.get("was_clipped", False)),
        }
        with open(paths.summary, "w") as fp:
            json.dump(data, fp, indent=2, sort_keys=True)
            fp.write("\n")

    def _write_trust_clip_summary(self, paths, classification, diagnostics, action):
        data = {}
        if paths.summary.exists():
            try:
                with open(paths.summary, "r") as fp:
                    data = json.load(fp)
            except Exception:
                data = {}
        data["trust_clip"] = {
            "policy": self.trust_clip_options["policy"],
            "classification": classification,
            "action": action,
            **dict(diagnostics),
        }
        with open(paths.summary, "w") as fp:
            json.dump(data, fp, indent=2, sort_keys=True)
            fp.write("\n")

    def _append_history_record(
        self,
        eval_id,
        objective,
        coefficients,
        gradient,
        status,
        line_search_info=None,
        eval_dir=None,
        eval_index=None,
        gradient_guard_info=None,
        gradient_entry=None,
        trust_clip_classification="",
        trust_clip_diagnostics=None,
        trust_clip_action="",
    ):
        line_search_info = {
            **self._line_search_default_info(),
            **(line_search_info or {}),
        }
        reduced_coefficients = (
            self.compress_full_physical(coefficients)
            if self.symmetry_coupling != "NONE"
            else []
        )
        gradient_values = list(gradient or [])
        gradient_is_finite = all(np.isfinite(float(value)) for value in gradient_values)
        reduced_gradient = (
            self.collapse_gradient_to_reduced(gradient_values)
            if self.symmetry_coupling != "NONE"
            and gradient_values
            and gradient_is_finite
            else []
        )
        gradient_guard_info = dict(gradient_guard_info or {})
        gradient_entry = dict(gradient_entry or {})
        trust_clip_diagnostics = dict(trust_clip_diagnostics or {})
        record = {
            "eval_index": eval_index,
            "slsqp_iter": self._slsqp_major_iter,
            "eval_id": eval_id,
            "eval_dir": str(eval_dir) if eval_dir is not None else "",
            "objective": objective,
            "coefficients": list(coefficients),
            "reduced_coefficients": list(reduced_coefficients),
            "gradients": gradient_values,
            "gnorm_raw": gradient_entry.get("gnorm_raw", ""),
            "gnorm_opt": gradient_entry.get("gnorm_opt", ""),
            "gradient_guard_reason": gradient_guard_info.get("reason", ""),
            "gradient_guard_reference": gradient_guard_info.get("reference", ""),
            "gradient_guard_ratio": gradient_guard_info.get("ratio", ""),
            "trust_clip_class": trust_clip_classification,
            "trust_clip_action": trust_clip_action,
            "trust_clip_beta": trust_clip_diagnostics.get("beta_eff", ""),
            "trust_clip_improvement_rel": trust_clip_diagnostics.get(
                "improvement_rel", ""
            ),
            "trust_clip_relative_worsening": trust_clip_diagnostics.get(
                "relative_worsening", ""
            ),
            "trust_clip_gnorm_ratio": trust_clip_diagnostics.get("gnorm_ratio", ""),
            "trust_clip_reasons": ",".join(
                trust_clip_diagnostics.get("toxic_reasons", [])
                or trust_clip_diagnostics.get("accepted_reasons", [])
            ),
            "status": status,
            "line_search_beta": line_search_info["line_search_beta"],
            "line_search_maxdiff": line_search_info["line_search_maxdiff"],
            "line_search_limited": line_search_info["line_search_limited"],
            "local_step_beta": line_search_info["local_step_beta"],
            "local_step_limited": line_search_info["local_step_limited"],
            "local_step_limiting_mode": line_search_info["local_step_limiting_mode"],
            "local_step_da": line_search_info["local_step_da"],
            "local_step_limit": line_search_info["local_step_limit"],
        }
        if self.thickness_constraint is not None:
            try:
                record.update(self._thickness_history_info(coefficients))
            except Exception:
                record["min_thickness_constraint"] = ""
                record["thickness_constraint_active"] = ""
        for mode_id, coefficient in zip(self.mode_ids, coefficients):
            record[f"coeff__{mode_id}"] = coefficient
        if self.symmetry_coupling != "NONE":
            for reduced_id, coefficient in zip(self.reduced_variable_ids, reduced_coefficients):
                record[f"reduced_coeff__{reduced_id}"] = coefficient
        for mode_id, value in zip(self.mode_ids, gradient_values):
            record[f"grad__{mode_id}"] = value
        if self.symmetry_coupling != "NONE":
            for reduced_id, value in zip(self.reduced_variable_ids, reduced_gradient):
                record[f"reduced_grad__{reduced_id}"] = value
        self._history_records.append(record)
        self.write_optimization_history()

    def _history_fieldnames(self):
        fieldnames = (
            [
                "eval_index",
                "slsqp_iter",
                "eval_id",
                "eval_dir",
                "objective",
                "coefficients",
                "reduced_coefficients",
                "gradients",
                "gnorm_raw",
                "gnorm_opt",
                "gradient_guard_reason",
                "gradient_guard_reference",
                "gradient_guard_ratio",
                "trust_clip_class",
                "trust_clip_action",
                "trust_clip_beta",
                "trust_clip_improvement_rel",
                "trust_clip_relative_worsening",
                "trust_clip_gnorm_ratio",
                "trust_clip_reasons",
            ]
            + [f"coeff__{mode_id}" for mode_id in self.mode_ids]
            + (
                [f"reduced_coeff__{reduced_id}" for reduced_id in self.reduced_variable_ids]
                if self.symmetry_coupling != "NONE"
                else []
            )
            + [f"grad__{mode_id}" for mode_id in self.mode_ids]
            + (
                [f"reduced_grad__{reduced_id}" for reduced_id in self.reduced_variable_ids]
                if self.symmetry_coupling != "NONE"
                else []
            )
            + [
                "line_search_beta",
                "line_search_maxdiff",
                "line_search_limited",
                "local_step_beta",
                "local_step_limited",
                "local_step_limiting_mode",
                "local_step_da",
                "local_step_limit",
            ]
        )
        if self.thickness_constraint is not None:
            fieldnames += [
                "min_thickness_constraint",
                "thickness_constraint_active",
            ]
        fieldnames += ["status"]
        return fieldnames

    def _best_ok_history_record(self):
        ok_records = [
            record
            for record in self._history_records
            if str(record.get("status", "")).strip().lower()
            in SAFE_EVALUATION_STATUSES
            and record.get("objective") is not None
            and np.isfinite(float(record.get("objective")))
        ]
        if not ok_records:
            return None
        return min(ok_records, key=lambda record: float(record["objective"]))

    def write_optimization_history(self):
        fieldnames = self._history_fieldnames()
        with open(self.optimization_history_filename, "w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for record in self._history_records:
                writer.writerow(
                    {
                        field: _format_config_atom(record.get(field, ""))
                        for field in fieldnames
                    }
                )

    def evaluate_constraint_function(self, coefficients, function_name):
        coefficients = [_as_float(value, "coefficient") for value in coefficients]
        if len(coefficients) != len(self.mode_ids):
            raise BSplineSU2DriverError(
                f"expected {len(self.mode_ids)} coefficients, got {len(coefficients)}"
            )
        function_name = str(function_name or "").strip().upper()
        if not function_name:
            raise BSplineSU2DriverError("constraint function name must be non-empty")
        if is_geometry_constraint_name(function_name):
            return self.evaluate_geometry_constraint_function(coefficients, function_name)

        key = (cache_key(coefficients, self.cache_tol), _normalized_name(function_name))
        if key in self._constraint_cache:
            return self._constraint_cache[key]

        _design_key, record = self._design_point(
            coefficients,
            objective_adjoint=function_name,
        )
        paths = self._record_paths_for_function(record, function_name)
        try:
            primal_values = self._run_shared_core(coefficients, paths, record=record)
            value = self._primal_value_for_function(primal_values, function_name)
            gradient = self._run_function_adjoint(
                paths,
                function_name,
                record=record,
            )
            self._write_eval_summary_metadata(
                paths,
                objective_function=function_name,
            )
            result = {
                "eval_id": record["eval_id"],
                "eval_dir": str(paths.eval_dir),
                "function": function_name,
                "value": value,
                "gradient": gradient,
                "coefficients": coefficients,
                "status": "ok",
            }
            self._constraint_cache[key] = result
            return result
        except Exception as exc:
            if isinstance(exc, BSplineSU2DriverError):
                raise
            raise BSplineSU2DriverError(
                "constraint evaluation {} failed in {}: {}".format(
                    function_name,
                    paths.eval_dir,
                    exc,
                )
            ) from exc

    def _finalize_objective(
        self,
        result,
        paths,
        line_search_info,
        eval_id,
        eval_index,
        coefficients,
        objective,
        gradient,
        key,
    ):
        gradient_entry = self._gradient_entry(
            result,
            paths,
            line_search_info=line_search_info,
        )
        try:
            guard_info = self.register_gradient_entry(
                gradient_entry,
                promote=not self._trust_clip_enabled(),
            )
        except GradientGuardStop as stop:
            result.update(
                {
                    "status": "rejected_gradient_guard",
                    "gnorm_raw": gradient_entry["gnorm_raw"],
                    "gnorm_opt": gradient_entry["gnorm_opt"],
                    "gradient_guard_info": stop.guard_info,
                }
            )
            self._write_gradient_guard_summary(
                paths,
                gradient_entry,
                stop.guard_info,
                "rejected",
            )
            self._append_history_record(
                eval_id,
                objective,
                coefficients,
                gradient,
                "rejected_gradient_guard",
                line_search_info=line_search_info,
                eval_dir=paths.eval_dir,
                eval_index=eval_index,
                gradient_guard_info=stop.guard_info,
                gradient_entry=gradient_entry,
            )
            raise

        trust_clip_classification = ""
        trust_clip_diagnostics = {}
        trust_clip_action = ""
        status = "ok"
        cache_as_safe = True
        if self._trust_clip_enabled():
            (
                trust_clip_classification,
                trust_clip_diagnostics,
            ) = self._classify_trust_clip_entry(gradient_entry)
            status = self._trust_clip_status(trust_clip_classification)
            trust_clip_action = {
                "not_clipped": "continue",
                "benign_clipped_legacy": "legacy_continue",
                "weak_clipped_progress": "weak_continue",
                "accepted_clipped_restart": "defer_restart_to_callback",
                "rejected_toxic_clip": "defer_rollback_to_callback",
            }[trust_clip_classification]
            finite_safe_candidate = bool(
                np.isfinite(float(gradient_entry["objective"]))
                and np.isfinite(float(gradient_entry["gnorm_raw"]))
            )
            if trust_clip_classification in (
                "not_clipped",
                "benign_clipped_legacy",
                "weak_clipped_progress",
            ) and finite_safe_candidate:
                self._promote_safe_entry(
                    gradient_entry,
                    update_recent_raw_gnorm=(
                        trust_clip_classification != "weak_clipped_progress"
                    ),
                )
            cache_as_safe = (
                finite_safe_candidate
                and trust_clip_classification
                in (
                    "not_clipped",
                    "benign_clipped_legacy",
                    "weak_clipped_progress",
                )
            )
            requested_optimizer_x = (line_search_info or {}).get(
                "requested_optimizer_x"
            )
            if requested_optimizer_x is not None:
                request_key = self._pending_trust_clip_key(requested_optimizer_x)
                self._trust_clip_by_requested_key[request_key] = {
                    "classification": trust_clip_classification,
                    "diagnostics": trust_clip_diagnostics,
                    "entry": gradient_entry,
                    "result": result,
                }
            self._write_trust_clip_summary(
                paths,
                trust_clip_classification,
                trust_clip_diagnostics,
                trust_clip_action,
            )

        result.update(
            {
                "status": status,
                "gnorm_raw": gradient_entry["gnorm_raw"],
                "gnorm_opt": gradient_entry["gnorm_opt"],
                "gradient_guard_info": guard_info,
                "trust_clip_class": trust_clip_classification,
                "trust_clip_diagnostics": trust_clip_diagnostics,
                "trust_clip_action": trust_clip_action,
            }
        )
        self._write_gradient_guard_summary(
            paths,
            gradient_entry,
            guard_info,
            "safe",
        )
        if cache_as_safe:
            self._cache[key] = result
        self._print_iteration_row(result, line_search_info=line_search_info)
        self._append_history_record(
            eval_id,
            objective,
            coefficients,
            gradient,
            status,
            line_search_info=line_search_info,
            eval_dir=paths.eval_dir,
            eval_index=eval_index,
            gradient_guard_info=guard_info,
            gradient_entry=gradient_entry,
            trust_clip_classification=trust_clip_classification,
            trust_clip_diagnostics=trust_clip_diagnostics,
            trust_clip_action=trust_clip_action,
        )
        return result

    def evaluate(self, coefficients, line_search_info=None):
        coefficients = [_as_float(value, "coefficient") for value in coefficients]
        if len(coefficients) != len(self.mode_ids):
            raise BSplineSU2DriverError(
                f"expected {len(self.mode_ids)} coefficients, got {len(coefficients)}"
            )

        key = cache_key(coefficients, self.cache_tol)
        if key in self._cache and not self._trust_clip_enabled():
            return self._cache[key]

        _design_key, record = self._design_point(
            coefficients,
            objective_adjoint=self.objective_adjoint,
        )
        paths = self._record_paths_for_function(record, self.objective_adjoint)
        self._run_eval_count += 1
        eval_index = self._run_eval_count
        objective = None
        gradient = None
        try:
            primal_values = self._run_shared_core(coefficients, paths, record=record)
            objective = self._objective_value_from_primal_values(primal_values)
            gradient = self._run_function_adjoint(
                paths,
                self.objective_adjoint,
                record=record,
                allow_nonfinite=(
                    self.gradient_guard_enabled or self._trust_clip_enabled()
                ),
            )
            # Point the eval-root aliases at the objective's adjoint artifacts so the
            # adaptive scoring (which reads <eval>/surface_sens.csv) always uses the
            # objective sensitivity, independent of constraint evaluation order.
            create_eval_aliases(paths)
            self._write_eval_summary_metadata(paths, line_search_info=line_search_info)

            result = {
                "eval_index": eval_index,
                "eval_id": record["eval_id"],
                "eval_dir": str(paths.eval_dir),
                "objective": objective,
                "gradient": gradient,
                "coefficients": coefficients,
                "status": "ok",
            }
            return self._finalize_objective(
                result,
                paths,
                line_search_info=line_search_info,
                eval_id=record["eval_id"],
                eval_index=eval_index,
                coefficients=coefficients,
                objective=objective,
                gradient=gradient,
                key=key,
            )
        except GradientGuardStop:
            raise
        except Exception as exc:
            self._append_history_record(
                record["eval_id"],
                objective,
                coefficients,
                gradient,
                "failed",
                line_search_info=line_search_info,
                eval_dir=paths.eval_dir,
                eval_index=eval_index,
            )
            if isinstance(exc, BSplineSU2DriverError):
                raise
            raise BSplineSU2DriverError(
                f"evaluation {record['eval_id']} failed in {paths.eval_dir}: {exc}"
            ) from exc

    def write_optimized_modes(self, coefficients):
        optimized_spec = update_mode_coefficients(self.mode_spec, coefficients)
        write_mode_spec(optimized_spec, self.optimized_modes_filename)
        return optimized_spec

    def _print_slsqp_parameters(self, maxiter, optimizer_bounds=None):
        if not self.print_optimizer_table:
            return
        optimizer_bounds = list(self.optimizer_bounds() if optimizer_bounds is None else optimizer_bounds)
        initial_optimizer_variables = self.physical_to_optimizer(
            self.initial_reduced_coefficients
        )
        print("Sequential Least SQuares Programming (SLSQP) parameters:")
        print(f"Number of active modes: {len(self.mode_ids)}")
        print(f"Number of design variables: {len(self.reduced_variable_ids)}")
        print(f"Symmetry coupling: {self.symmetry_coupling}")
        print(f"[PROGRESSIVE_BSPLINE][SURFACE] mode = {self.surface_mode}")
        print(
            "[PROGRESSIVE_BSPLINE][SURFACE] active sides = "
            f"{active_sides_from_surface_mode(self.surface_mode)}"
        )
        print(f"[PROGRESSIVE_BSPLINE][SURFACE] ndv = {len(self.reduced_variable_ids)}")
        print(
            "[PROGRESSIVE_BSPLINE][SURFACE] deformation direction = "
            f"{self.deformation_direction_mode}"
        )
        print(f"Eval layout: {self.eval_layout}")
        print(f"Sensitivity weighting: {self.sensitivity_weighting}")
        print(f"Sensitivity source: {self.sensitivity_source}")
        print(
            "Raw-gradient guard: {} factor={} window={} min_history={} floor={}".format(
                "ON" if self.gradient_guard_enabled else "OFF",
                self.gradient_guard_factor,
                self.gradient_guard_window,
                self.gradient_guard_min_history,
                self.gradient_guard_floor,
            )
        )
        print(
            "Trust-clip policy: {} legacy_beta_min={} severe_beta={} "
            "bad_patience={}/{}".format(
                self.trust_clip_options["policy"],
                self.trust_clip_options["legacy_beta_min"],
                self.trust_clip_options["severe_beta"],
                self.trust_clip_options["bad_patience"],
                self.trust_clip_options["bad_window"],
            )
        )
        print(
            "Objective function scaling factor: [{:.15g}]".format(
                float(self.opt_gradient_factor)
            )
        )
        if self.native_constraints:
            formatted = [
                "{}{}{} * {}".format(
                    spec.name,
                    spec.sign,
                    "{:.15g}".format(float(spec.target)),
                    "{:.15g}".format(float(spec.scale)),
                )
                for spec in self.native_constraints
            ]
            print("Native SU2 constraints: " + "; ".join(formatted))
            if any(is_geometry_constraint_name(spec.name) for spec in self.native_constraints):
                if self.geometry_constraint_gradient_mode == "SU2_GEO":
                    print(
                        "Geometric constraint backend: SU2_GEO finite differences, eps={:.15g}".format(
                            float(self.geometry_fd_eps)
                        )
                    )
                else:
                    print(
                        "Geometric constraint gradient: {} (SU2_GEO fallback eps={:.15g})".format(
                            self.geometry_constraint_gradient_mode,
                            float(self.geometry_fd_eps),
                        )
                    )
        print(
            "Variable scaling: physical coefficient = optimizer variable * {:.15g}".format(
                float(self.opt_relax_factor)
            )
        )
        print(f"Maximum number of iterations: {int(maxiter)}")
        accuracy = 1.0e-10 if self.opt_accuracy is None else self.opt_accuracy
        print("Requested accuracy: {:.15g}".format(float(accuracy)))
        print(
            "Initial physical coefficients: "
            + _vector_summary(self.initial_coefficients)
        )
        print("Physical coefficient bounds: " + _bounds_summary(self.bounds))
        print(f"Local step limiter: {'ON' if self.local_step_limit else 'OFF'}")
        if self.local_step_limit:
            print("Local step limit ratio: {:.15g}".format(float(self.local_step_limit_ratio)))
        print(
            "Initial SLSQP variables: "
            + _vector_summary(initial_optimizer_variables)
        )
        print("SLSQP variable bounds: " + _bounds_summary(optimizer_bounds))
        if not _bounds_are_uniform(self.bounds) or not _bounds_are_uniform(optimizer_bounds):
            arrays_file = self.workdir / "slsqp_parameter_arrays.json"
            with open(arrays_file, "w") as fp:
                json.dump(
                    {
                        "initial_physical_coefficients": list(self.initial_coefficients),
                        "physical_bounds": _bounds_to_list(self.bounds),
                        "initial_slsqp_variables": list(initial_optimizer_variables),
                        "slsqp_bounds": _bounds_to_list(optimizer_bounds),
                    },
                    fp,
                    indent=2,
                    sort_keys=True,
                )
                fp.write("\n")
            print(f"Full nonuniform SLSQP parameter arrays: {arrays_file}")
        print(
            "Note: EVAL_ID is the eval_XXXX directory suffix and may not start from zero if the workdir is reused. "
            "FC is the current-run CFD evaluation counter."
        )
        print(
            "Note: SLSQP_IT is the latest accepted SLSQP major iteration known at evaluation time; "
            "evaluations printed before a callback use the previous accepted iteration."
        )
        print("")

    def _print_iteration_row(self, result, line_search_info=None):
        if not self.print_optimizer_table:
            return

        if not self._printed_iteration_header:
            print(
                "SLSQP_IT   FC   EVAL_ID      OBJFUN_PHYS     OBJFUN_SLSQP      "
                "GNORM_RAW        GNORM_OPT          LS_BETA LS_BETA_LOCAL"
            )
            self._printed_iteration_header = True

        gradient = result.get("gradient") or []
        gnorm_raw = math.sqrt(
            sum(float(value) * float(value) for value in gradient)
        )
        reduced_gradient = self.collapse_gradient_to_reduced(gradient) if gradient else []

        info = line_search_info or {}
        beta = float(info.get("line_search_beta", 1.0))
        local_beta = float(info.get("local_step_beta", 1.0))
        if int(info.get("local_step_limited", 0)):
            print(
                "[BSPLINE_SU2_DRIVER] Local step limiter: beta={:.6e}, limiting_mode={}, da={:.6e}, limit={:.6e}".format(
                    local_beta,
                    info.get("local_step_limiting_mode", ""),
                    float(info.get("local_step_da", 0.0)),
                    float(info.get("local_step_limit", 0.0)),
                )
            )
        if beta < 1.0:
            print(
                "[BSPLINE_SU2_DRIVER] WARNING: line_search_beta/local_step_beta < 1; "
                "the gradient passed to SLSQP is approximate because the evaluated design is clipped."
            )

        obj_phys = float(result["objective"])
        obj_slsqp = obj_phys * float(self.opt_gradient_factor)

        gnorm_opt = (
            math.sqrt(sum(float(value) * float(value) for value in reduced_gradient))
            * float(self.opt_relax_factor)
            * float(self.opt_gradient_factor)
            * beta
        )

        slsqp_iter = int(self._slsqp_major_iter)
        fc = int(result.get("eval_index", self._run_eval_count))
        eval_id = int(result.get("eval_id", -1))

        print(
            "{:8d} {:4d} {:9d} {:16.6E} {:16.6E} {:16.6E} {:16.6E} {:12.4E} {:13.4E}".format(
                slsqp_iter,
                fc,
                eval_id,
                obj_phys,
                obj_slsqp,
                gnorm_raw,
                gnorm_opt,
                beta,
                local_beta,
            )
        )

    def _apply_trigger_resume_state(self, trigger_resume_state=None):
        state = dict(trigger_resume_state or {})
        self.trigger_project.trigger_history = list(state.get("trigger_history", []))
        self.trigger_project.trigger_state = state.get("trigger_state", None)
        self.trigger_project.refinement_triggered = bool(
            state.get("refinement_triggered", False)
        )

    def _trigger_resume_state(self):
        return {
            "trigger_history": list(self.trigger_project.trigger_history),
            "trigger_state": self.trigger_project.trigger_state,
            "refinement_triggered": bool(self.trigger_project.refinement_triggered),
        }

    def _attach_trigger_state(self, result):
        result["trigger_history"] = list(self.trigger_project.trigger_history)
        result["trigger_state"] = self.trigger_project.trigger_state
        result["trigger_history_len"] = len(self.trigger_project.trigger_history)
        result["refinement_triggered"] = bool(self.trigger_project.refinement_triggered)
        return result

    def _controlled_gradient_guard_result(self, stop, optimizer="SLSQP"):
        safe = stop.last_safe_entry
        self.restore_modes_from_entry(safe)
        if safe is None:
            coefficients = list(self.initial_coefficients)
            objective = math.inf
            safe_eval_id = None
        else:
            coefficients = [float(value) for value in safe["evaluated_x"]]
            objective = float(safe["objective"])
            safe_eval_id = int(safe["eval_id"])
        refine = self.gradient_guard_next_action == "refine"
        if refine:
            self.trigger_project.refinement_triggered = True
        if self.print_optimizer_table:
            print("Raw-gradient guard stop    (controlled rollback)")
            print(f"            Restored evaluation: {safe_eval_id}")
            print(f"            Current function value: {objective:.12g}")
        return self._attach_trigger_state({
            "optimizer": optimizer,
            "success": True,
            "message": "Raw-gradient guard stop: restored last safe evaluation",
            "objective": objective,
            "coefficients": coefficients,
            "status": "gradient_guard_stop",
            "gradient_guard_triggered": True,
            "gradient_guard_info": dict(stop.guard_info),
            "gradient_guard_bad_eval_id": stop.bad_entry.get("eval_id"),
            "gradient_guard_restore_eval_id": safe_eval_id,
            "gradient_guard_next_action": self.gradient_guard_next_action,
            "early_refine_triggered": refine,
            "refinement_triggered": refine,
        })

    def _controlled_trust_clip_result(self, stop, optimizer="SLSQP"):
        rollback = stop.rollback_entry
        self.restore_modes_from_entry(rollback)
        if rollback is None:
            coefficients = list(self.initial_coefficients)
            objective = math.inf
            restore_eval_id = None
        else:
            coefficients = [float(value) for value in rollback["evaluated_x"]]
            objective = float(rollback["objective"])
            restore_eval_id = int(rollback["eval_id"])
        next_action = (
            "refine" if stop.action == "refine" else "restart_same_level"
        )
        refine = next_action == "refine"
        if refine:
            self.trigger_project.refinement_triggered = True
        return self._attach_trigger_state({
            "optimizer": optimizer,
            "success": True,
            "message": f"Trust-clip controlled stop: {stop.classification}",
            "objective": objective,
            "coefficients": coefficients,
            "status": "trust_clip_stop",
            "trust_clip_triggered": True,
            "trust_clip_class": stop.classification,
            "trust_clip_diagnostics": dict(stop.diagnostics),
            "trust_clip_action": stop.action,
            "trust_clip_bad_eval_id": stop.entry.get("eval_id"),
            "trust_clip_restore_eval_id": restore_eval_id,
            "trust_clip_next_action": next_action,
            "early_refine_triggered": refine,
            "refinement_triggered": refine,
        })

    def optimize(
        self,
        maxiter=5,
        fallback_step=0.1,
        gradient_tol=1.0e-8,
        trigger_resume_state=None,
    ):
        self._apply_trigger_resume_state(trigger_resume_state)
        self.configure_geometry_aware_bounds()
        self._configure_line_search_bound()
        self.configure_thickness_constraint()
        try:
            from scipy.optimize import minimize
        except Exception:
            if self.thickness_constraint is not None or self.native_constraints:
                raise BSplineSU2DriverError(
                    "SciPy is required when optimization constraints are active"
                )
            try:
                return self._attach_trigger_state(self._optimize_projected_gradient_descent(
                    maxiter=maxiter,
                    step_size=fallback_step,
                    gradient_tol=gradient_tol,
                ))
            except GradientGuardStop as stop:
                return self._controlled_gradient_guard_result(
                    stop,
                    optimizer="projected_gradient_descent",
                )

        x0 = self.physical_to_optimizer(self.initial_reduced_coefficients)
        bounds_u = self.optimizer_bounds()
        constraints = (
            self._thickness_constraint_functions()
            + self._native_constraint_functions()
        )
        self._print_slsqp_parameters(maxiter, optimizer_bounds=bounds_u)

        def fun(x):
            result, info = self._evaluate_optimizer_variables(list(x))
            if result.get("trust_clip_class") not in (
                "weak_clipped_progress",
                "accepted_clipped_restart",
                "rejected_toxic_clip",
            ) and not info.get("cache_hit", False):
                record_objective_and_check(
                    self.trigger_project,
                    float(result["objective"]),
                )
            return float(result["objective"]) * self.opt_gradient_factor

        def jac(x):
            result, info = self._evaluate_optimizer_variables(list(x))
            beta = float(info.get("line_search_beta", 1.0))
            reduced_gradient = self.collapse_gradient_to_reduced(result["gradient"])
            return [
                float(value) * self.opt_relax_factor * self.opt_gradient_factor * beta
                for value in reduced_gradient
            ]

        def callback(x):
            self._slsqp_major_iter += 1
            self._trust_clip_callback(list(x))

        options = {
            "maxiter": int(maxiter),
            "disp": False,
        }
        if self.opt_accuracy is not None:
            options["ftol"] = float(self.opt_accuracy) * self.opt_gradient_factor

        early_refine_triggered = False
        result = None
        try:
            result = minimize(
                fun,
                x0,
                jac=jac,
                bounds=bounds_u,
                constraints=constraints,
                method="SLSQP",
                callback=callback,
                options=options,
            )
        except TrustClipStop as stop:
            return self._controlled_trust_clip_result(stop, optimizer="SLSQP")
        except GradientGuardStop as stop:
            return self._controlled_gradient_guard_result(stop, optimizer="SLSQP")
        except RefinementTriggered:
            early_refine_triggered = True
            print(
                f"[{trigger_prefix(self.trigger_project)}] "
                "Optimization stopped early due to refinement trigger"
            )
        best_record = self._best_ok_history_record()
        if best_record is not None:
            final_coefficients = [
                float(best_record.get(f"coeff__{mode_id}", value))
                for mode_id, value in zip(
                    self.mode_ids,
                    self.initial_coefficients
                    if result is None
                    else self.expand_reduced_physical(self.optimizer_to_physical(result.x)),
                )
            ]
            final_objective = float(best_record["objective"])
        else:
            if result is not None:
                final_coefficients = self.expand_reduced_physical(
                    self.optimizer_to_physical(result.x)
                )
                final_objective = float(result.fun) / self.opt_gradient_factor
            else:
                final_coefficients = list(self.initial_coefficients)
                final_objective = math.inf
        self.write_optimized_modes(final_coefficients)
        if self.print_optimizer_table:
            if early_refine_triggered:
                print("Early refinement trigger    (Exit mode early_refine_trigger)")
            else:
                print("{}    (Exit mode {})".format(str(result.message), int(result.status)))
            print("            Current function value: {:.12g}".format(final_objective))
            print("            Iterations: {}".format(int(getattr(result, "nit", self._slsqp_major_iter))))
            print("            Function evaluations: {}".format(int(getattr(result, "nfev", len(self._history_records)))))
            print("            Gradient evaluations: {}".format(int(getattr(result, "njev", len(self._history_records)))))
        return self._attach_trigger_state({
            "optimizer": "SLSQP",
            "success": True if early_refine_triggered else bool(result.success),
            "message": (
                "Early refinement trigger"
                if early_refine_triggered
                else str(result.message)
            ),
            "objective": final_objective,
            "coefficients": final_coefficients,
            "status": (
                "early_refine_trigger"
                if early_refine_triggered
                else "ok"
            ),
            "early_refine_triggered": bool(early_refine_triggered),
            "refinement_triggered": bool(
                getattr(self.trigger_project, "refinement_triggered", False)
            ),
        })

    def _optimize_projected_gradient_descent(self, maxiter, step_size, gradient_tol):
        self.configure_geometry_aware_bounds()
        self._configure_line_search_bound()
        x = _project_to_bounds(self.initial_reduced_coefficients, self.reduced_bounds)
        self._print_slsqp_parameters(maxiter)
        current, _line_search_info, x = self._evaluate_reduced_physical(x)
        self._line_search_anchor_physical = list(current["coefficients"])
        self._local_step_anchor_reduced = list(x)
        step = float(step_size)

        for _ in range(int(maxiter)):
            gradient = self.collapse_gradient_to_reduced(current["gradient"])
            gradient_norm = math.sqrt(sum(value * value for value in gradient))
            if gradient_norm <= gradient_tol:
                break

            accepted = False
            trial_step = step
            for _inner in range(12):
                trial = _project_to_bounds(
                    [value - trial_step * grad for value, grad in zip(x, gradient)],
                    self.reduced_bounds,
                )
                if trial == x:
                    trial_step *= 0.5
                    continue
                trial_result, _line_search_info, trial_reduced_eval = self._evaluate_reduced_physical(trial)
                if trial_result["objective"] <= current["objective"]:
                    x = trial_reduced_eval
                    current = trial_result
                    self._line_search_anchor_physical = list(trial_result["coefficients"])
                    self._local_step_anchor_reduced = list(x)
                    step = min(trial_step * 1.25, 1.0)
                    accepted = True
                    break
                trial_step *= 0.5

            if not accepted:
                break

        best_record = self._best_ok_history_record()
        if best_record is not None:
            final_coefficients = [
                float(best_record.get(f"coeff__{mode_id}", value))
                for mode_id, value in zip(self.mode_ids, self.expand_reduced_physical(x))
            ]
            current = {
                "objective": float(best_record["objective"]),
                "gradient": best_record.get("gradient", current.get("gradient")),
            }
        else:
            final_coefficients = self.expand_reduced_physical(x)
        self.write_optimized_modes(final_coefficients)
        if self.print_optimizer_table:
            print("Optimization terminated successfully    (projected gradient fallback)")
            print("            Current function value: {:.12g}".format(float(current["objective"])))
            print("            Function evaluations: {}".format(len(self._history_records)))
            print("            Gradient evaluations: {}".format(len(self._history_records)))
        return {
            "optimizer": "projected_gradient_descent",
            "success": True,
            "message": "SciPy unavailable; used projected gradient descent fallback",
            "objective": current["objective"],
            "coefficients": final_coefficients,
        }


def run_bspline_su2_optimization(
    modes_filename,
    base_mesh,
    marker,
    def_template,
    primal_template,
    adjoint_template,
    workdir,
    objective_column="CD",
    maxiter=5,
    mpi_prefix="",
    default_bounds=DEFAULT_BOUNDS,
    cache_tol=1.0e-12,
    fallback_step=0.1,
    show_commands=False,
    stream_solver_output=False,
    print_optimizer_table=True,
    auto_scale_bounds_to_geometry=False,
    max_normal_displacement=None,
    max_rms_normal_displacement=None,
    min_bound_scale=0.0,
    opt_accuracy=None,
    opt_bound_upper=None,
    opt_bound_lower=None,
    opt_relax_factor=1.0,
    opt_gradient_factor=1.0,
    gradient_guard=True,
    gradient_guard_factor=100.0,
    gradient_guard_window=5,
    gradient_guard_min_history=3,
    gradient_guard_floor=1.0e-14,
    gradient_guard_next_action="restart_same_level",
    refinement_available=None,
    trust_clip_policy="OFF",
    trust_clip_beta_tol=1.0e-12,
    trust_clip_legacy_beta_min=0.50,
    trust_clip_severe_beta=0.50,
    trust_clip_worsening_tol=0.05,
    trust_clip_soft_gnorm_factor=20.0,
    trust_clip_bad_patience=2,
    trust_clip_bad_window=5,
    trust_clip_stag_tol=1.0e-6,
    opt_line_search_bound=None,
    thickness_options=None,
    native_constraints=None,
    geometry_fd_eps=1.0e-6,
    geometry_constraint_gradient="AUTO",
    eval_layout="DSN",
    objective_adjoint="drag",
    symmetry_coupling="NONE",
    surface_mode="BOTH",
    sensitivity_weighting="NODAL",
    sensitivity_source="DOT_AD_TRANSFER",
    local_step_limit=False,
    local_step_limit_ratio=200.0,
    trigger_opts=None,
    trigger_resume_state=None,
    progressive_label="PROGRESSIVE_BSPLINE",
    deformation_direction_mode=None,
    le_safe_direction=False,
    le_safe_x0=None,
    le_safe_x1=None,
    le_safe_power=None,
):
    """Run the fixed active-mode B-spline optimization and return its result."""

    driver = BSplineSU2Driver(
        modes_filename=modes_filename,
        base_mesh=base_mesh,
        marker=marker,
        def_template=def_template,
        primal_template=primal_template,
        adjoint_template=adjoint_template,
        workdir=workdir,
        objective_column=objective_column,
        mpi_prefix=mpi_prefix,
        default_bounds=default_bounds,
        cache_tol=cache_tol,
        show_commands=show_commands,
        stream_solver_output=stream_solver_output,
        print_optimizer_table=print_optimizer_table,
        auto_scale_bounds_to_geometry=auto_scale_bounds_to_geometry,
        max_normal_displacement=max_normal_displacement,
        max_rms_normal_displacement=max_rms_normal_displacement,
        min_bound_scale=min_bound_scale,
        opt_accuracy=opt_accuracy,
        opt_bound_upper=opt_bound_upper,
        opt_bound_lower=opt_bound_lower,
        opt_relax_factor=opt_relax_factor,
        opt_gradient_factor=opt_gradient_factor,
        gradient_guard=gradient_guard,
        gradient_guard_factor=gradient_guard_factor,
        gradient_guard_window=gradient_guard_window,
        gradient_guard_min_history=gradient_guard_min_history,
        gradient_guard_floor=gradient_guard_floor,
        gradient_guard_next_action=gradient_guard_next_action,
        refinement_available=refinement_available,
        trust_clip_policy=trust_clip_policy,
        trust_clip_beta_tol=trust_clip_beta_tol,
        trust_clip_legacy_beta_min=trust_clip_legacy_beta_min,
        trust_clip_severe_beta=trust_clip_severe_beta,
        trust_clip_worsening_tol=trust_clip_worsening_tol,
        trust_clip_soft_gnorm_factor=trust_clip_soft_gnorm_factor,
        trust_clip_bad_patience=trust_clip_bad_patience,
        trust_clip_bad_window=trust_clip_bad_window,
        trust_clip_stag_tol=trust_clip_stag_tol,
        opt_line_search_bound=opt_line_search_bound,
        thickness_options=thickness_options,
        native_constraints=native_constraints,
        geometry_fd_eps=geometry_fd_eps,
        geometry_constraint_gradient=geometry_constraint_gradient,
        eval_layout=eval_layout,
        objective_adjoint=objective_adjoint,
        symmetry_coupling=symmetry_coupling,
        surface_mode=surface_mode,
        sensitivity_weighting=sensitivity_weighting,
        sensitivity_source=sensitivity_source,
        local_step_limit=local_step_limit,
        local_step_limit_ratio=local_step_limit_ratio,
        trigger_opts=trigger_opts,
        progressive_label=progressive_label,
        deformation_direction_mode=deformation_direction_mode,
        le_safe_direction=le_safe_direction,
        le_safe_x0=le_safe_x0,
        le_safe_x1=le_safe_x1,
        le_safe_power=le_safe_power,
    )
    result = driver.optimize(
        maxiter=maxiter,
        fallback_step=fallback_step,
        trigger_resume_state=trigger_resume_state,
    )
    result["optimization_history"] = str(driver.optimization_history_filename)
    result["optimized_modes"] = str(driver.optimized_modes_filename)
    result["workdir"] = str(driver.workdir)
    return result
