#!/usr/bin/env python

## \file scipy_tools.py
#  \brief tools for interfacing with scipy
#  \author T. Lukaczyk, F. Palacios
#  \version 8.4.0 "Harrier"

import sys

from .. import eval as su2eval
from numpy import array, zeros


class RefinementTriggered(Exception):
    pass


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
    project.n_dv = n_dv

    if not x0:
        x0 = [0.0] * n_dv

    dv_scales = project.config["DEFINITION_DV"]["SCALE"]
    k = 0
    for i, dv_scl in enumerate(dv_scales):
        for j in range(dv_size[i]):
            x0[k] = x0[k] / dv_scl
            k = k + 1

    obj = project.config["OPT_OBJECTIVE"]
    obj_scale = []
    for this_obj in obj.keys():
        obj_scale = obj_scale + [obj[this_obj]["SCALE"]]

    if len(obj.keys()) == 1:
        accu = accu * obj_scale[0]

    eps = 1.0e-04

    sys.stdout.write("Sequential Least SQuares Programming (SLSQP) parameters:\n")
    sys.stdout.write(
        "Number of design variables: " + str(len(dv_size)) + " ( " + str(n_dv) + " ) \n"
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

    return outputs


# -------------------------------------------------------------------
#  Scipy CG
# -------------------------------------------------------------------


def scipy_cg(project, x0=None, xb=None, its=100, accu=1e-10, grads=True):
    from scipy.optimize import fmin_cg

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
    obj_list = project.obj_f(x)
    obj = 0
    for this_obj in obj_list:
        obj = obj + this_obj

    if not hasattr(project, "trigger_history"):
        project.trigger_history = []

    project.trigger_history.append(obj)

    opts = getattr(project, "trigger_opts", None)

    if opts:
        trigger = str(opts.get("trigger", "")).upper()

        if trigger == "SLOPE_EFFICIENCY_TRIGGER":
            _check_slope_trigger(project, obj, opts)

        elif trigger == "STAGNATION_TRIGGER":
            _check_stagnation_trigger(project, obj, opts)

    return obj


def obj_df(x, project):
    dobj_list = project.obj_df(x)
    dobj = [0.0] * len(dobj_list[0])

    for this_dobj in dobj_list:
        idv = 0
        for this_dv_dobj in this_dobj:
            dobj[idv] = dobj[idv] + this_dv_dobj
            idv += 1
    dobj = array(dobj)

    # Store the last objective gradient evaluated by scipy
    project.last_obj_grad = dobj.tolist()
    project.last_obj_grad_x = list(x)

    return dobj


def con_ceq(x, project):
    cons = project.con_ceq(x)

    if cons:
        cons = array(cons)
    else:
        cons = zeros([0])

    return cons


def con_dceq(x, project):
    dcons = project.con_dceq(x)

    dim = project.n_dv
    if dcons:
        dcons = array(dcons)
    else:
        dcons = zeros([0, dim])

    return dcons


def con_cieq(x, project):
    cons = project.con_cieq(x)

    if cons:
        cons = array(cons)
    else:
        cons = zeros([0])

    return -cons


def con_dcieq(x, project):
    dcons = project.con_dcieq(x)

    dim = project.n_dv
    if dcons:
        dcons = array(dcons)
    else:
        dcons = zeros([0, dim])

    return -dcons