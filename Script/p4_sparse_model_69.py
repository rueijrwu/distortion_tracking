"""Predeclared75-pose R1 extension:33 symmetry terms x{1,a}+3on-axis."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from p4_sparse_model import SparseModel as BaseModel, PARAM_NAMES, OUTER

SPATIAL = tuple((parameter, i, j) for parameter, powers in (
    (0, ((2, 0), (0, 2), (4, 0), (2, 2), (0, 4))),
    (1, ((2, 0), (0, 2), (4, 0), (2, 2), (0, 4))),
    (2, ((2, 0), (0, 2), (4, 0), (2, 2), (0, 4))),
    (3, ((1, 1), (3, 1), (1, 3))), (4, ((1, 1), (3, 1), (1, 3))),
    (5, ((1, 0), (3, 0), (1, 2))), (6, ((1, 0), (3, 0), (1, 2))),
    (7, ((0, 1), (2, 1), (0, 3))), (8, ((0, 1), (2, 1), (0, 3)))) for i, j in powers)
assert len(SPATIAL) == 33


def gaze_basis(ade, bde, derivatives=False):
    tx, ty = np.asarray(bde).ravel()/20., np.asarray(ade).ravel()/20.
    base = np.stack([tx**i*ty**j for _, i, j in SPATIAL], -1)
    if not derivatives:
        return base
    dade = np.stack([(j/20.)*tx**i*ty**(j-1) if j else np.zeros_like(tx) for _, i, j in SPATIAL], -1)
    dbde = np.stack([(i/20.)*tx**(i-1)*ty**j if i else np.zeros_like(tx) for _, i, j in SPATIAL], -1)
    return base, dade, dbde


class SparseModel69(BaseModel):
    def __post_init__(self):
        self.reference_mm = np.asarray(self.reference_mm, float)
        self.calibration = np.asarray(self.calibration, float)
        self.coefficients = np.asarray(self.coefficients, float)
        if self.accommodation_degree != 1 or self.composition != "projective_first":
            raise ValueError("Only the predeclared linear-A/original-R1 extension is supported")
        if self.calibration.shape != (3,) or self.coefficients.shape != (66,):
            raise ValueError("Expected3calibration+66gaze coefficients")
        self.parameter_for_coefficient = np.tile([p for p, _, _ in SPATIAL], 2)
        self.labels = [f"{PARAM_NAMES[p]}:a^{d}*tx^{i}*ty^{j}" for d in (0, 1) for p, i, j in SPATIAL]

    def maps(self, accommodation, ade, bde, derivatives=False):
        a = np.asarray(accommodation).ravel()/4.
        if derivatives:
            gaze, dade, dbde = gaze_basis(ade, bde, True)
        else:
            gaze = gaze_basis(ade, bde)
        basis = np.concatenate((gaze, a[:, None]*gaze), axis=1)
        pars = np.zeros((len(a), 9))
        if derivatives:
            derivs = (np.concatenate((np.zeros_like(gaze), gaze/4.), axis=1),
                      np.concatenate((dade, a[:, None]*dade), axis=1),
                      np.concatenate((dbde, a[:, None]*dbde), axis=1))
            dp = np.zeros((len(a), 9, 3))
        for parameter in range(9):
            ids = np.flatnonzero(self.parameter_for_coefficient == parameter)
            pars[:, parameter] = basis[:, ids]@self.coefficients[ids]
            if derivatives:
                for j, derivative in enumerate(derivs):
                    dp[:, parameter, j] = derivative[:, ids]@self.coefficients[ids]
        pars[:, :2] += 1.
        return (pars, basis, dp) if derivatives else (pars, basis)

    def prepare_gaze(self, ade, bde):
        return PreparedGaze69(self, ade, bde)


class PreparedGaze69:
    def __init__(self, model, ade, bde):
        self.model = model
        gaze = gaze_basis(ade, bde)
        theta = model.coefficients.reshape(2, 33)
        self.maps = np.zeros((len(gaze), 9, 2))
        for p in range(9):
            ids = [i for i, term in enumerate(SPATIAL) if term[0] == p]
            self.maps[:, p, :] = gaze[:, ids]@theta[:, ids].T

    def predict(self, accommodation, indices=None):
        A = np.asarray(accommodation).ravel()
        coefficients = self.maps if indices is None else self.maps[indices]
        pars = coefficients[:, :, 0]+(A/4.)[:, None]*coefficients[:, :, 1]
        pars = pars.copy()
        pars[:, :2] += 1.
        return self.model._spatial(A, pars)["prediction"]


def fit_model(states, centered, accommodation_degree=1, composition="projective_first"):
    states, centered = np.asarray(states), np.asarray(centered)
    if len(states) not in (72, 75):
        raise ValueError("69candidate is restricted to75calibration states or its72-state CV folds")
    anchor = np.flatnonzero(np.all(np.isclose(states, [0., 0., 0.], atol=1e-10), axis=1))
    if len(anchor) != 1:
        raise ValueError("Missing unique reference calibration")
    model = SparseModel69(centered[anchor[0]].copy(), np.zeros(3), np.zeros(66), accommodation_degree, composition)
    axis = np.all(np.isclose(states[:, 1:], 0., atol=1e-10), axis=1)
    if np.sum(axis) != 3:
        raise ValueError("Three on-axis calibration states are required")

    def cal_fun(c):
        model.calibration = np.asarray(c).copy()
        return ((model.evaluate(*states[axis].T)["prediction"][:, OUTER]-centered[axis][:, OUTER])*1000.).ravel()

    def cal_jac(c):
        model.calibration = np.asarray(c).copy()
        return (model.evaluate(*states[axis].T, calibration_jac=True)["calibration_jacobian"][:, OUTER]*1000.).reshape(-1, 3)

    cal = least_squares(cal_fun, np.zeros(3), jac=cal_jac, x_scale="jac", max_nfev=150,
                        ftol=1e-12, xtol=1e-12, gtol=1e-12)
    model.calibration = cal.x.copy()

    def fun(c):
        model.coefficients = np.asarray(c).copy()
        return ((model.evaluate(*states.T)["prediction"][:, OUTER]-centered[:, OUTER])*1000.).ravel()

    def jac(c):
        model.coefficients = np.asarray(c).copy()
        return (model.evaluate(*states.T, theta_jac=True)["theta_jacobian"][:, OUTER]*1000.).reshape(-1, 66)

    fit = least_squares(fun, np.zeros(66), jac=jac, x_scale="jac", max_nfev=200,
                        ftol=1e-11, xtol=1e-11, gtol=1e-11)
    model.coefficients = fit.x.copy()
    return model, {"calibration_success": bool(cal.success), "calibration_message": cal.message,
        "calibration_nfev": cal.nfev, "calibration_cost_um2_over_2": float(cal.cost),
        "gaze_success": bool(fit.success), "gaze_message": fit.message, "gaze_nfev": fit.nfev,
        "gaze_cost_um2_over_2": float(fit.cost), "gaze_optimality": float(fit.optimality),
        "initialization": "identity/zero; only75physical calibration poses; no dense prior"}
