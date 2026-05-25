#!/usr/bin/env python

## \file scipy_tools.py
#  \brief tools for interfacing with scipy
#  \author T. Lukaczyk, F. Palacios
#  \version 8.4.0 "Harrier"

import math
import sys

from .. import eval as su2eval
from numpy import array, zeros
from SU2.opt.progressive_hh_core import (
    expand_symmetric_dv,
    full_to_reduced_symmetric,
    reduce_symmetric_gradient,
)


class RefinementTriggered(Exception):
    pass


def _get_symmetry(project):
    symmetry = getattr(project, "progressive_hh_symmetry", None)
    if symmetry is None:
        symmetry = getattr(project, "progressive_hh_opts", None)
    if not symmetry:
        return {"mode": "NONE", "sign": -1.0}

    return {
        "mode": str(symmetry.get("mode", symmetry.get("symmetry_mode", "NONE"))).upper(),
        "sign": float(symmetry.get("sign", symmetry.get("symmetry_sign", -1.0))),
    }


def _is_reduced_symmetry(project):
    return _get_symmetry(project)["mode"] == "REDUCED"


def _symmetry_pair_count(project):
    n_full = sum(project.config["DEFINITION_DV"]["SIZE"])
    if n_full % 2 != 0:
        raise ValueError(
            "PROGRESSIVE_HH_SYMMETRY_MODE=REDUCED requires an even full HH DV count"
        )
    return n_full // 2


def _expand_if_needed(x, project):
    if not _is_reduced_symmetry(project):
        return x
    symmetry = _get_symmetry(project)
    return array(expand_symmetric_dv(x, symmetry["sign"]))


def _reduce_grad_if_needed(g, project):
    if not _is_reduced_symmetry(project):
        return array(g)
    symmetry = _get_symmetry(project)
    n_pairs = _symmetry_pair_count(project)
    return array(reduce_symmetric_gradient(g, n_pairs, symmetry["sign"]))


def _reduce_jac_if_needed(J, project):
    if not _is_reduced_symmetry(project):
        return array(J)

    n_pairs = _symmetry_pair_count(project)
    symmetry = _get_symmetry(project)
    return array(
        [
            reduce_symmetric_gradient(row, n_pairs, symmetry["sign"])
            for row in J
        ]
    )


def _validate_reduced_bounds(xb, n_pairs, sign):
    if len(xb) != 2 * n_pairs:
        raise ValueError(
            "Reduced symmetry bound size mismatch: "
            f"got {len(xb)}, expected {2 * n_pairs}"
        )

    tol = 1.0e-12
    sign = float(sign)
    if abs(sign) <= tol:
        raise ValueError("PROGRESSIVE_HH_SYMMETRY_SIGN must be non-zero")

    for i in range(n_pairs):
        u_lo, u_hi = [float(v) for v in xb[i]]
        l_lo, l_hi = [float(v) for v in xb[n_pairs + i]]

        if sign > 0.0:
            z_lo_from_lower = l_lo / sign
            z_hi_from_lower = l_hi / sign
        else:
            z_lo_from_lower = l_hi / sign
            z_hi_from_lower = l_lo / sign

        if u_lo < z_lo_from_lower - tol or u_hi > z_hi_from_lower + tol:
            raise ValueError(
                "Incompatible upper/lower bounds for "
                "PROGRESSIVE_HH_SYMMETRY_MODE=REDUCED at pair "
                f"{i}: upper z bounds=({u_lo}, {u_hi}), "
                f"lower-implied z bounds=({z_lo_from_lower}, {z_hi_from_lower}), "
                f"sign={sign}"
            )


def _init_trigger_state(project):
    if not hasattr(project, "trigger_state") or project.trigger_state is None:
        project.trigger_state = {
            "accepted_history": [],
            "best_obj": None,
            "sat_counter": 0,
        }


def _update_filtered_history(accepted_history, value, filter_tol):
    """
    Update the filtered history used by the slope trigger.

    Rules:
      - if improving: accept
      - if worsening but relative worsening <= filter_tol: accept
      - if worsening too much: reject
    """
    eps = 1.0e-14

    if not accepted_history:
        accepted_history.append(value)
        return True

    ref = accepted_history[-1]

    if value <= ref:
        accepted_history.append(value)
        return True

    rel_wors = (value - ref) / max(abs(ref), eps)

    if rel_wors <= filter_tol:
        accepted_history.append(value)
        return True

    return False


def _compute_smoothed_history(history, window):
    if window <= 1:
        return list(history)

    smooth = []
    for i in range(window - 1, len(history)):
        avg = sum(history[i - window + 1 : i + 1]) / float(window)
        smooth.append(avg)
    return smooth


def _check_slope_trigger(project, obj_value, opts):
    """
    New robust slope trigger:
      - filtered history
      - small worsenings tolerated
      - large spikes ignored
      - only positive decrements are used
    """
    _init_trigger_state(project)

    warmup_iter = int(opts.get("warmup_iter", 0))

    if len(project.trigger_history) <= warmup_iter:
        sys.stdout.write(
            "[PROGRESSIVE_HH] SLOPE_EFFICIENCY ONLINE | "
            f"warmup phase ({len(project.trigger_history)}/{warmup_iter})\n"
        )
        return

    w = max(1, int(opts.get("window", 1)))
    r = float(opts.get("tol", 0.2))
    filter_tol = float(opts.get("filter_tol", 0.02))

    accepted_history = project.trigger_state["accepted_history"]
    accepted_now = _update_filtered_history(accepted_history, obj_value, filter_tol)

    if not accepted_now:
        sys.stdout.write(
            "[PROGRESSIVE_HH] SLOPE_EFFICIENCY ONLINE | "
            "large worsening ignored in filtered history\n"
        )
        return

    smooth = _compute_smoothed_history(accepted_history, w)

    if len(smooth) < 2:
        return

    slopes = []
    for i in range(1, len(smooth)):
        dj = smooth[i - 1] - smooth[i]
        slopes.append(dj)

    if not slopes:
        return

    current_slope = slopes[-1]

    if current_slope <= 0.0:
        sys.stdout.write(
            "[PROGRESSIVE_HH] SLOPE_EFFICIENCY ONLINE | "
            "last accepted step not improving, skip trigger check\n"
        )
        return

    positive_slopes = [s for s in slopes if s > 0.0]

    if not positive_slopes:
        return

    max_slope = max(positive_slopes)

    if max_slope <= 1.0e-16:
        sys.stdout.write(
            "[PROGRESSIVE_HH] SLOPE_EFFICIENCY ONLINE | "
            "flat positive history, skip trigger check\n"
        )
        return

    ratio = current_slope / max_slope

    sys.stdout.write(
        "[PROGRESSIVE_HH] SLOPE_EFFICIENCY ONLINE | "
        f"ratio={ratio:.6e} threshold={r:.6e}\n"
    )

    if ratio < r:
        project.refinement_triggered = True
        sys.stdout.write("[PROGRESSIVE_HH] Efficiency trigger -> STOP\n")
        raise RefinementTriggered()


def _check_slope_best_log_trigger(project, obj_value, opts):
    _init_trigger_state(project)

    state = project.trigger_state
    state.setdefault("last_log_best", None)
    state.setdefault("improvements", [])
    state.setdefault("max_slope_seen", 0.0)
    state.setdefault("bad_count", 0)

    warmup_iter = int(opts.get("warmup_iter", 0))
    window = max(1, int(opts.get("window", 1)))
    tol = float(opts.get("tol", 0.2))
    eps = float(opts.get("eps", 1.0e-300))
    patience = max(1, int(opts.get("patience", 1)))

    best_obj = state.get("best_obj", None)
    if best_obj is None or obj_value < best_obj:
        best_obj = obj_value
        state["best_obj"] = best_obj

    y_k = math.log(max(best_obj, eps))

    if state["last_log_best"] is None:
        state["last_log_best"] = y_k
        return

    delta_k = state["last_log_best"] - y_k
    if delta_k < 0.0:
        delta_k = 0.0
    state["improvements"].append(delta_k)

    if len(project.trigger_history) <= warmup_iter:
        state["last_log_best"] = y_k
        sys.stdout.write(
            "[PROGRESSIVE_HH] SLOPE_EFFICIENCY_BEST_LOG | "
            f"warmup phase ({len(project.trigger_history)}/{warmup_iter})\n"
        )
        return

    if len(state["improvements"]) < window:
        state["last_log_best"] = y_k
        return

    recent = state["improvements"][-window:]
    recent_slope = sum(recent) / float(window)

    if recent_slope > 0.0:
        state["max_slope_seen"] = max(state["max_slope_seen"], recent_slope)

    ratio = recent_slope / max(state["max_slope_seen"], eps)

    sys.stdout.write(
        "[PROGRESSIVE_HH] SLOPE_EFFICIENCY_BEST_LOG | "
        f"ratio={ratio:.6e} threshold={tol:.6e} "
        f"bad_count={state['bad_count']}/{patience}\n"
    )

    if ratio < tol:
        state["bad_count"] += 1
    else:
        state["bad_count"] = 0

    state["last_log_best"] = y_k

    if state["bad_count"] >= patience:
        project.refinement_triggered = True
        sys.stdout.write("[PROGRESSIVE_HH] SLOPE_EFFICIENCY_BEST_LOG -> STOP\n")
        raise RefinementTriggered()


def _check_stagnation_trigger(project, obj_value, opts):
    """
    New stagnation trigger with a single saturation counter.

    Logic:
      - significant new best -> reset counter
      - small new best OR point near best -> increase counter
      - point far from best -> reset counter
      - trigger when counter reaches stag_window
    """
    _init_trigger_state(project)
    warmup_iter = int(opts.get("warmup_iter", 0))

    if len(project.trigger_history) <= warmup_iter:
        sys.stdout.write(
            "[PROGRESSIVE_HH] STAGNATION ONLINE | "
            f"warmup phase ({len(project.trigger_history)}/{warmup_iter})\n"
        )
        return
    eps = 1.0e-14
    stag_tol = float(opts.get("stag_tol", 1.0e-3))
    stag_band = float(opts.get("stag_band", 0.02))
    stag_window = int(opts.get("stag_window", 3))

    best_obj = project.trigger_state["best_obj"]
    sat_counter = project.trigger_state["sat_counter"]

    if best_obj is None:
        project.trigger_state["best_obj"] = obj_value
        project.trigger_state["sat_counter"] = 0
        return

    # New best
    if obj_value < best_obj:
        improvement = (best_obj - obj_value) / max(abs(best_obj), eps)
        project.trigger_state["best_obj"] = obj_value

        if improvement > stag_tol:
            project.trigger_state["sat_counter"] = 0
            sys.stdout.write(
                "[PROGRESSIVE_HH] STAGNATION ONLINE | "
                f"significant new best, reset counter (impr={improvement:.6e})\n"
            )
        else:
            project.trigger_state["sat_counter"] = sat_counter + 1
            sys.stdout.write(
                "[PROGRESSIVE_HH] STAGNATION ONLINE | "
                f"small new best, counter={project.trigger_state['sat_counter']} "
                f"(impr={improvement:.6e}, tol={stag_tol:.6e})\n"
            )
    else:
        gap = (obj_value - best_obj) / max(abs(best_obj), eps)

        if gap < stag_band:
            project.trigger_state["sat_counter"] = sat_counter + 1
            sys.stdout.write(
                "[PROGRESSIVE_HH] STAGNATION ONLINE | "
                f"near best, counter={project.trigger_state['sat_counter']} "
                f"(gap={gap:.6e}, band={stag_band:.6e})\n"
            )
        else:
            project.trigger_state["sat_counter"] = 0
            sys.stdout.write(
                "[PROGRESSIVE_HH] STAGNATION ONLINE | "
                f"outside band, reset counter (gap={gap:.6e}, band={stag_band:.6e})\n"
            )

    if project.trigger_state["sat_counter"] >= stag_window:
        project.refinement_triggered = True
        sys.stdout.write("[PROGRESSIVE_HH] Stagnation trigger -> STOP\n")
        raise RefinementTriggered()


# -------------------------------------------------------------------
#  Scipy SLSQP
# -------------------------------------------------------------------


def scipy_slsqp(project, x0=None, xb=None, its=100, accu=1e-10, grads=True):
    from scipy.optimize import fmin_slsqp

    if x0 is None:
        x0 = []
    if xb is None:
        xb = []

    func = obj_f
    f_eqcons = con_ceq
    f_ieqcons = con_cieq

    if project.config.get("GRADIENT_METHOD", "NONE") == "NONE":
        fprime = None
        fprime_eqcons = None
        fprime_ieqcons = None
    else:
        fprime = obj_df
        fprime_eqcons = con_dceq
        fprime_ieqcons = con_dcieq

    dv_size = project.config["DEFINITION_DV"]["SIZE"]
    n_dv = sum(dv_size)
    reduced_symmetry = _is_reduced_symmetry(project)
    symmetry = _get_symmetry(project)
    n_full = n_dv
    n_pairs = None
    if reduced_symmetry:
        n_pairs = _symmetry_pair_count(project)
        project.n_dv = n_pairs
        if len(x0) not in (0, n_full):
            raise ValueError(
                "PROGRESSIVE_HH_SYMMETRY_MODE=REDUCED requires full-length x0: "
                f"got {len(x0)}, expected {n_full}"
            )
        if len(xb) != n_full:
            raise ValueError(
                "PROGRESSIVE_HH_SYMMETRY_MODE=REDUCED requires full-length bounds: "
                f"got {len(xb)}, expected {n_full}"
            )
    else:
        project.n_dv = n_dv

    if not x0:
        x0 = [0.0] * n_dv

    dv_scales = project.config["DEFINITION_DV"]["SCALE"]
    k = 0
    for i, dv_scl in enumerate(dv_scales):
        for j in range(dv_size[i]):
            x0[k] = x0[k] / dv_scl
            k = k + 1

    if reduced_symmetry:
        _validate_reduced_bounds(xb, n_pairs, symmetry["sign"])
        z0 = full_to_reduced_symmetric(x0, n_pairs, symmetry["sign"])
        xb_reduced = xb[:n_pairs]
        x0_full_scaled = list(x0)
        x0 = z0
        xb = xb_reduced

    obj = project.config["OPT_OBJECTIVE"]
    obj_scale = []
    for this_obj in obj.keys():
        obj_scale = obj_scale + [obj[this_obj]["SCALE"]]

    if len(obj.keys()) == 1:
        accu = accu * obj_scale[0]

    eps = 1.0e-04

    sys.stdout.write("Sequential Least SQuares Programming (SLSQP) parameters:\n")
    if reduced_symmetry:
        sys.stdout.write(
            "Number of design variables: reduced "
            + str(n_pairs)
            + ", full "
            + str(n_full)
            + "\n"
        )
    else:
        sys.stdout.write(
            "Number of design variables: "
            + str(len(dv_size))
            + " ( "
            + str(n_dv)
            + " ) \n"
        )
    sys.stdout.write("Objective function scaling factor: " + str(obj_scale) + "\n")
    sys.stdout.write("Maximum number of iterations: " + str(its) + "\n")
    sys.stdout.write("Requested accuracy: " + str(accu) + "\n")
    sys.stdout.write("Initial guess for the independent variable(s): " + str(x0) + "\n")
    sys.stdout.write(
        "Lower and upper bound for each independent variable: " + str(xb) + "\n\n"
    )

    project.trigger_history = []
    project.refinement_triggered = False
    project.trigger_state = None

    # Store the last objective gradient seen by scipy
    project.last_obj_grad = None
    project.last_obj_grad_x = None
    project.last_obj_grad_full = None
    project.last_obj_grad_x_full = None
    project.last_dv_values = None
    project.last_reduced_dv_values = None
    project.opt_dv_values = None
    project.opt_reduced_dv_values = None

    if reduced_symmetry:
        project.last_dv_values = expand_symmetric_dv(x0, symmetry["sign"])
        project.last_reduced_dv_values = [float(v) for v in x0]
        project.initial_full_dv_values = x0_full_scaled

    if not hasattr(project, "trigger_opts"):
        project.trigger_opts = None

    sys.stdout.write("[DEBUG] trigger_opts = " + str(project.trigger_opts) + "\n")

    try:
        outputs = fmin_slsqp(
            x0=x0,
            func=func,
            f_eqcons=f_eqcons,
            f_ieqcons=f_ieqcons,
            fprime=fprime,
            fprime_eqcons=fprime_eqcons,
            fprime_ieqcons=fprime_ieqcons,
            args=(project,),
            bounds=xb,
            iter=its,
            iprint=2,
            full_output=True,
            acc=accu,
            epsilon=eps,
        )
    except RefinementTriggered:
        sys.stdout.write(
            "[PROGRESSIVE_HH] Optimization stopped early due to refinement trigger\n"
        )
        outputs = None

    if outputs is not None:
        try:
            if reduced_symmetry:
                z_opt = [float(v) for v in outputs[0]]
                project.opt_reduced_dv_values = z_opt
                project.opt_dv_values = expand_symmetric_dv(z_opt, symmetry["sign"])
            else:
                project.opt_dv_values = [float(v) for v in outputs[0]]
        except Exception:
            project.opt_dv_values = getattr(project, "last_dv_values", None)
    else:
        project.opt_dv_values = getattr(project, "last_dv_values", None)
        if reduced_symmetry:
            project.opt_reduced_dv_values = getattr(
                project,
                "last_reduced_dv_values",
                None,
            )

    return outputs


# -------------------------------------------------------------------
#  Scipy CG
# -------------------------------------------------------------------


def scipy_cg(project, x0=None, xb=None, its=100, accu=1e-10, grads=True):
    from scipy.optimize import fmin_cg

    if _is_reduced_symmetry(project):
        raise ValueError(
            "PROGRESSIVE_HH_SYMMETRY_MODE=REDUCED currently supports only SLSQP"
        )

    if x0 is None:
        x0 = []
    if xb is None:
        xb = []

    func = obj_f

    if project.config.get("GRADIENT_METHOD", "NONE") == "NONE":
        fprime = None
    else:
        fprime = obj_df

    n_dv = len(project.config["DEFINITION_DV"]["KIND"])
    project.n_dv = n_dv

    if not x0:
        x0 = [0.0] * n_dv

    dv_scales = project.config["DEFINITION_DV"]["SCALE"]
    x0 = [x0[i] / dv_scl for i, dv_scl in enumerate(dv_scales)]

    obj = project.config["OPT_OBJECTIVE"]
    obj_scale = obj[obj.keys()[0]]["SCALE"]
    accu = accu * obj_scale

    eps = 1.0e-04

    sys.stdout.write("Conjugate gradient (CG) parameters:\n")
    sys.stdout.write("Number of design variables: " + str(n_dv) + "\n")
    sys.stdout.write("Objective function scaling factor: " + str(obj_scale) + "\n")
    sys.stdout.write("Maximum number of iterations: " + str(its) + "\n")
    sys.stdout.write("Requested accuracy: " + str(accu) + "\n")
    sys.stdout.write("Initial guess for the independent variable(s): " + str(x0) + "\n")
    sys.stdout.write(
        "Lower and upper bound for each independent variable: " + str(xb) + "\n\n"
    )

    obj_f(x0, project)

    outputs = fmin_cg(
        x0=x0,
        f=func,
        fprime=fprime,
        args=(project,),
        gtol=accu,
        epsilon=eps,
        maxiter=its,
        full_output=True,
        disp=True,
        retall=True,
    )

    return outputs


# -------------------------------------------------------------------
#  Scipy BFGS
# -------------------------------------------------------------------


def scipy_bfgs(project, x0=None, xb=None, its=100, accu=1e-10, grads=True):
    from scipy.optimize import fmin_bfgs

    if _is_reduced_symmetry(project):
        raise ValueError(
            "PROGRESSIVE_HH_SYMMETRY_MODE=REDUCED currently supports only SLSQP"
        )

    if x0 is None:
        x0 = []
    if xb is None:
        xb = []

    func = obj_f

    if project.config.get("GRADIENT_METHOD", "NONE") == "NONE":
        fprime = None
    else:
        fprime = obj_df

    n_dv = len(project.config["DEFINITION_DV"]["KIND"])
    project.n_dv = n_dv

    if not x0:
        x0 = [0.0] * n_dv

    dv_scales = project.config["DEFINITION_DV"]["SCALE"]
    x0 = [x0[i] / dv_scl for i, dv_scl in enumerate(dv_scales)]

    obj = project.config["OPT_OBJECTIVE"]
    obj_scale = obj[obj.keys()[0]]["SCALE"]
    accu = accu * obj_scale

    eps = 1.0e-04

    sys.stdout.write("Broyden-Fletcher-Goldfarb-Shanno (BFGS) parameters:\n")
    sys.stdout.write("Number of design variables: " + str(n_dv) + "\n")
    sys.stdout.write("Objective function scaling factor: " + str(obj_scale) + "\n")
    sys.stdout.write("Maximum number of iterations: " + str(its) + "\n")
    sys.stdout.write("Requested accuracy: " + str(accu) + "\n")
    sys.stdout.write("Initial guess for the independent variable(s): " + str(x0) + "\n")
    sys.stdout.write(
        "Lower and upper bound for each independent variable: " + str(xb) + "\n\n"
    )

    obj_f(x0, project)

    outputs = fmin_bfgs(
        x0=x0,
        f=func,
        fprime=fprime,
        args=(project,),
        gtol=accu,
        epsilon=eps,
        maxiter=its,
        full_output=True,
        disp=True,
        retall=True,
    )

    return outputs


def scipy_powell(project, x0=None, xb=None, its=100, accu=1e-10, grads=False):
    from scipy.optimize import fmin_powell

    if _is_reduced_symmetry(project):
        raise ValueError(
            "PROGRESSIVE_HH_SYMMETRY_MODE=REDUCED currently supports only SLSQP"
        )

    if x0 is None:
        x0 = []

    func = obj_f

    n_dv = len(project.config["DEFINITION_DV"]["KIND"])
    project.n_dv = n_dv

    if not x0:
        x0 = [0.0] * n_dv

    dv_scales = project.config["DEFINITION_DV"]["SCALE"]
    x0 = [x0[i] / dv_scl for i, dv_scl in enumerate(dv_scales)]

    obj = project.config["OPT_OBJECTIVE"]
    obj_scale = obj[obj.keys()[0]]["SCALE"]
    accu = accu * obj_scale

    eps = 1.0e-04

    sys.stdout.write("Powells method parameters:\n")
    sys.stdout.write("Number of design variables: " + str(n_dv) + "\n")
    sys.stdout.write("Objective function scaling factor: " + str(obj_scale) + "\n")
    sys.stdout.write("Maximum number of iterations: " + str(its) + "\n")
    sys.stdout.write("Requested accuracy: " + str(accu) + "\n")

    obj_f(x0, project)

    outputs = fmin_powell(
        x0=x0,
        func=func,
        args=(project,),
        ftol=accu,
        maxiter=its,
        full_output=True,
        disp=True,
        retall=True,
    )

    return outputs


def obj_f(x, project):
    x_eval = _expand_if_needed(x, project)

    if _is_reduced_symmetry(project):
        project.last_reduced_dv_values = [float(v) for v in x]
        project.last_dv_values = [float(v) for v in x_eval]
    else:
        project.last_dv_values = [float(v) for v in x]

    obj_list = project.obj_f(x_eval)
    obj = 0
    for this_obj in obj_list:
        obj = obj + this_obj

    if not hasattr(project, "trigger_history"):
        project.trigger_history = []

    project.trigger_history.append(obj)

    opts = getattr(project, "trigger_opts", None)

    if opts:
        trigger = str(opts.get("trigger", "")).upper()

        if trigger in ("SLOPE_EFFICIENCY_TRIGGER", "SLOPE_EFFICIENCY_FILTERED"):
            _check_slope_trigger(project, obj, opts)

        elif trigger == "SLOPE_EFFICIENCY_BEST_LOG":
            _check_slope_best_log_trigger(project, obj, opts)

        elif trigger == "STAGNATION_TRIGGER":
            _check_stagnation_trigger(project, obj, opts)

    return obj


def obj_df(x, project):
    x_eval = _expand_if_needed(x, project)
    dobj_list = project.obj_df(x_eval)
    dobj = [0.0] * len(dobj_list[0])

    for this_dobj in dobj_list:
        idv = 0
        for this_dv_dobj in this_dobj:
            dobj[idv] = dobj[idv] + this_dv_dobj
            idv += 1
    dobj_full = array(dobj)
    dobj = _reduce_grad_if_needed(dobj_full, project)

    # Store the last objective gradient evaluated by scipy
    project.last_obj_grad = dobj.tolist()
    project.last_obj_grad_x = list(x)
    if _is_reduced_symmetry(project):
        project.last_obj_grad_full = dobj_full.tolist()
        project.last_obj_grad_x_full = [float(v) for v in x_eval]

    return dobj


def con_ceq(x, project):
    x_eval = _expand_if_needed(x, project)
    cons = project.con_ceq(x_eval)

    if cons:
        cons = array(cons)
    else:
        cons = zeros([0])

    return cons


def con_dceq(x, project):
    x_eval = _expand_if_needed(x, project)
    dcons = project.con_dceq(x_eval)

    dim = project.n_dv
    if dcons:
        dcons = _reduce_jac_if_needed(dcons, project)
    else:
        dcons = zeros([0, dim])

    return dcons


def con_cieq(x, project):
    x_eval = _expand_if_needed(x, project)
    cons = project.con_cieq(x_eval)

    if cons:
        cons = array(cons)
    else:
        cons = zeros([0])

    return -cons


def con_dcieq(x, project):
    x_eval = _expand_if_needed(x, project)
    dcons = project.con_dcieq(x_eval)

    dim = project.n_dv
    if dcons:
        dcons = _reduce_jac_if_needed(dcons, project)
    else:
        dcons = zeros([0, dim])

    return -dcons
