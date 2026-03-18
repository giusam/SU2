#!/usr/bin/env python

## \file scipy_tools.py
#  \brief tools for interfacing with scipy
#  \author T. Lukaczyk, F. Palacios
#  \version 8.4.0 "Harrier"
#
# SU2 Project Website: https://su2code.github.io
#
# The SU2 Project is maintained by the SU2 Foundation
# (http://su2foundation.org)
#
# Copyright 2012-2026, SU2 Contributors (cf. AUTHORS.md)
#
# SU2 is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# SU2 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with SU2. If not, see <http://www.gnu.org/licenses/>.

# -------------------------------------------------------------------
#  Imports
# -------------------------------------------------------------------

import sys

from .. import eval as su2eval
from numpy import array, zeros


class RefinementTriggered(Exception):
    pass


# -------------------------------------------------------------------
#  Scipy SLSQP
# -------------------------------------------------------------------


def scipy_slsqp(project, x0=None, xb=None, its=100, accu=1e-10, grads=True):
    """result = scipy_slsqp(project,x0=[],xb=[],its=100,accu=1e-10)

    Runs the Scipy implementation of SLSQP with
    an SU2 project

    Inputs:
        project - an SU2 project
        x0      - optional, initial guess
        xb      - optional, design variable bounds
        its     - max outer iterations, default 100
        accu    - accuracy, default 1e-10

    Outputs:
       result - the outputs from scipy.fmin_slsqp
    """

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
    if not hasattr(project, "trigger_opts"):
        project.trigger_opts = {
            "trigger": "",
            "window": 1,
            "tol": 0.1,
        }

    sys.stdout.write(
        "[DEBUG] trigger_opts = " + str(project.trigger_opts) + "\n"
    )

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
    """result = scipy_cg(project,x0=[],xb=[],its=100,accu=1e-10)

    Runs the Scipy implementation of CG with
    an SU2 project

    Inputs:
        project - an SU2 project
        x0      - optional, initial guess
        xb      - optional, design variable bounds
        its     - max outer iterations, default 100
        accu    - accuracy, default 1e-10

    Outputs:
       result - the outputs from scipy.fmin_slsqp
    """

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
    """result = scipy_bfgs(project,x0=[],xb=[],its=100,accu=1e-10)

    Runs the Scipy implementation of BFGS with
    an SU2 project

    Inputs:
        project - an SU2 project
        x0      - optional, initial guess
        xb      - optional, design variable bounds
        its     - max outer iterations, default 100
        accu    - accuracy, default 1e-10

    Outputs:
       result - the outputs from scipy.fmin_slsqp
    """

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
    """result = scipy_powell(project,x0=[],xb=[],its=100,accu=1e-10)

    Runs the Scipy implementation of Powell's method with
    an SU2 project

    Inputs:
        project - an SU2 project
        x0      - optional, initial guess
        xb      - optional, design variable bounds
        its     - max outer iterations, default 100
        accu    - accuracy, default 1e-10

    Outputs:
       result - the outputs from scipy.fmin_slsqp
    """

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
    """obj = obj_f(x,project)

    Objective Function
    SU2 Project interface to scipy.fmin_slsqp

    su2:         minimize f(x), list[nobj]
    scipy_slsqp: minimize f(x), float
    """

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
        history = project.trigger_history

        if trigger == "WINDOW_DROP":
            w = max(1, int(opts["window"]))
            tol = float(opts["tol"])

            if len(history) >= w + 1:
                j_old = history[-w - 1]
                j_new = history[-1]
                rel_drop = abs(j_old - j_new) / max(abs(j_new), 1.0e-14)

                sys.stdout.write(
                    "[PROGRESSIVE_HH] WINDOW_DROP ONLINE | "
                    f"rel_drop={rel_drop:.6e} threshold={tol:.6e}\n"
                )

                if rel_drop < tol:
                    sys.stdout.write(
                        "[PROGRESSIVE_HH] Window-drop trigger -> STOP\n"
                    )
                    raise RefinementTriggered()

        elif trigger == "ANDERSON":
            w = max(1, int(opts["window"]))
            r = float(opts["tol"])

            if len(history) >= w + 2:
                smooth = []
                for i in range(w - 1, len(history)):
                    avg = sum(history[i - w + 1 : i + 1]) / float(w)
                    smooth.append(avg)

                slopes = []
                for i in range(1, len(smooth)):
                    dj = smooth[i - 1] - smooth[i]
                    slopes.append(max(dj, 0.0))

                if slopes:
                    max_slope = max(slopes)
                    current_slope = slopes[-1]

                    if max_slope <= 1.0e-16:
                        sys.stdout.write(
                            "[PROGRESSIVE_HH] Anderson trigger (flat history) -> STOP\n"
                        )
                        raise RefinementTriggered()

                    ratio = current_slope / max_slope

                    sys.stdout.write(
                        "[PROGRESSIVE_HH] ANDERSON ONLINE | "
                        f"ratio={ratio:.6e} threshold={r:.6e}\n"
                    )

                    if ratio < r:
                        sys.stdout.write(
                            "[PROGRESSIVE_HH] Anderson trigger -> STOP\n"
                        )
                        raise RefinementTriggered()

    return obj


def obj_df(x, project):
    """dobj = obj_df(x,project)

    Objective Function Gradients
    SU2 Project interface to scipy.fmin_slsqp

    su2:         df(x), list[nobj x dim]
    scipy_slsqp: df(x), ndarray[dim]
    """

    dobj_list = project.obj_df(x)
    dobj = [0.0] * len(dobj_list[0])

    for this_dobj in dobj_list:
        idv = 0
        for this_dv_dobj in this_dobj:
            dobj[idv] = dobj[idv] + this_dv_dobj
            idv += 1
    dobj = array(dobj)

    return dobj


def con_ceq(x, project):
    """cons = con_ceq(x,project)

    Equality Constraint Functions
    SU2 Project interface to scipy.fmin_slsqp

    su2:         ceq(x) = 0.0, list[nceq]
    scipy_slsqp: ceq(x) = 0.0, ndarray[nceq]
    """

    cons = project.con_ceq(x)

    if cons:
        cons = array(cons)
    else:
        cons = zeros([0])

    return cons


def con_dceq(x, project):
    """dcons = con_dceq(x,project)

    Equality Constraint Gradients
    SU2 Project interface to scipy.fmin_slsqp

    su2:         dceq(x), list[nceq x dim]
    scipy_slsqp: dceq(x), ndarray[nceq x dim]
    """

    dcons = project.con_dceq(x)

    dim = project.n_dv
    if dcons:
        dcons = array(dcons)
    else:
        dcons = zeros([0, dim])

    return dcons


def con_cieq(x, project):
    """cons = con_cieq(x,project)

    Inequality Constraints
    SU2 Project interface to scipy.fmin_slsqp

    su2:         cieq(x) < 0.0, list[ncieq]
    scipy_slsqp: cieq(x) > 0.0, ndarray[ncieq]
    """

    cons = project.con_cieq(x)

    if cons:
        cons = array(cons)
    else:
        cons = zeros([0])

    return -cons


def con_dcieq(x, project):
    """dcons = con_dcieq(x,project)

    Inequality Constraint Gradients
    SU2 Project interface to scipy.fmin_slsqp

    su2:         dcieq(x), list[ncieq x dim]
    scipy_slsqp: dcieq(x), ndarray[ncieq x dim]
    """

    dcons = project.con_dcieq(x)

    dim = project.n_dv
    if dcons:
        dcons = array(dcons)
    else:
        dcons = zeros([0, dim])

    return -dcons
