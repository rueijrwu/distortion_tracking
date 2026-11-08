"""Globally shared, direct-reference R1 models for sparse P4 calibration.

The 12 lowest-order symmetry terms are identifiable on a 3x3 gaze lattice.
No barrel-removal transform or dense fitted prior is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

L_MM = 1.235429592627
OUTER = np.array([0, 1, 2, 3, 5, 6, 7, 8])
# Parameter order matches the existing R1: m11,m22,g,m12,m21,qx,p2,qy,p1.
SPATIAL = ((0, 2, 0), (0, 0, 2), (1, 2, 0), (1, 0, 2),
           (2, 2, 0), (2, 0, 2), (3, 1, 1), (4, 1, 1),
           (5, 1, 0), (6, 1, 0), (7, 0, 1), (8, 0, 1))
PARAM_NAMES = ("m11", "m22", "g", "m12", "m21", "qx", "p2", "qy", "p1")


def spatial_basis(ade, bde, derivatives=False):
    tx, ty = np.asarray(bde).ravel()/20., np.asarray(ade).ravel()/20.
    basis = np.stack([tx**i*ty**j for _, i, j in SPATIAL], -1)
    if not derivatives:
        return basis
    dade = np.stack([(j/20.)*tx**i*ty**(j-1) if j else np.zeros_like(tx) for _, i, j in SPATIAL], -1)
    dbde = np.stack([(i/20.)*tx**(i-1)*ty**j if i else np.zeros_like(tx) for _, i, j in SPATIAL], -1)
    return basis, dade, dbde


def _matrix(a, b, c, d):
    return np.stack((np.stack((a, b), -1), np.stack((c, d), -1)), -2)


def _project(x, pars, jac):
    m11, m22, _, m12, m21, qx, _, qy, _ = (pars[:, j, None] for j in range(9))
    vx, vy = x[..., 0], x[..., 1]
    den = 1.+qx*vx+qy*vy
    hx, hy = (m11*vx+m12*vy)/den, (m21*vx+m22*vy)/den
    h = np.stack((hx, hy), -1)
    if not jac:
        return h, den, None, None
    derivative = _matrix((m11-hx*qx)/den, (m12-hx*qy)/den,
                         (m21-hy*qx)/den, (m22-hy*qy)/den)
    local = np.zeros(h.shape+(9,))
    local[..., 0, 0], local[..., 1, 1] = vx/den, vy/den
    local[..., 0, 3], local[..., 1, 4] = vy/den, vx/den
    local[..., 5] = -h*(vx/den)[..., None]
    local[..., 7] = -h*(vy/den)[..., None]
    return h, den, derivative, local


def _brown(x, pars, jac):
    hx, hy = x[..., 0], x[..., 1]
    p1, p2 = pars[:, 8, None], pars[:, 6, None]
    wx = hx+2*p1*hx*hy+p2*(3*hx*hx+hy*hy)
    wy = hy+p1*(hx*hx+3*hy*hy)+2*p2*hx*hy
    w = np.stack((wx, wy), -1)
    xx, xy, yy = 1+2*p1*hy+6*p2*hx, 2*p1*hx+2*p2*hy, 1+6*p1*hy+2*p2*hx
    determinant = xx*yy-xy*xy
    if not jac:
        return w, determinant, None, None
    derivative = _matrix(xx, xy, xy, yy)
    local = np.zeros(w.shape+(9,))
    local[..., 6] = np.stack((3*hx*hx+hy*hy, 2*hx*hy), -1)
    local[..., 8] = np.stack((2*hx*hy, hx*hx+3*hy*hy), -1)
    return w, determinant, derivative, local


def _radial(x, kappa, jac):
    r2 = np.sum(x*x, axis=-1)
    factor = 1.+kappa[:, None]*r2
    radial = 1.+3.*kappa[:, None]*r2
    output = x*factor[..., None]
    if not jac:
        return output, factor, radial, None, None
    xx, xy = x[..., 0], x[..., 1]
    twice = 2.*kappa[:, None]
    derivative = _matrix(factor+twice*xx*xx, twice*xx*xy, twice*xx*xy, factor+twice*xy*xy)
    local = np.zeros(output.shape+(9,))
    local[..., 2] = x*r2[..., None]
    return output, factor, radial, derivative, local


@dataclass
class SparseModel:
    reference_mm: np.ndarray
    calibration: np.ndarray
    coefficients: np.ndarray
    accommodation_degree: int = 1
    composition: str = "projective_first"
    length_mm: float = L_MM

    def __post_init__(self):
        self.reference_mm = np.asarray(self.reference_mm, float)
        self.calibration = np.asarray(self.calibration, float)
        self.coefficients = np.asarray(self.coefficients, float)
        if self.accommodation_degree not in (1, 2) or self.composition not in ("projective_first", "distortion_first"):
            raise ValueError("Unsupported predeclared candidate")
        if self.calibration.shape != (3,) or self.coefficients.shape != (12*(self.accommodation_degree+1),):
            raise ValueError("Coefficient dimensions disagree with candidate")
        self.parameter_for_coefficient = np.tile([p for p, _, _ in SPATIAL], self.accommodation_degree+1)
        self.labels = [f"{PARAM_NAMES[p]}:a^{d}*tx^{i}*ty^{j}" for d in range(self.accommodation_degree+1) for p, i, j in SPATIAL]

    @property
    def parameter_count(self):
        return len(self.coefficients)+3

    def maps(self, accommodation, ade, bde, derivatives=False):
        a = np.asarray(accommodation).ravel()/4.
        if derivatives:
            gaze, dade, dbde = spatial_basis(ade, bde, True)
        else:
            gaze = spatial_basis(ade, bde)
        powers = np.stack([a**d for d in range(self.accommodation_degree+1)], -1)
        basis = (powers[:, :, None]*gaze[:, None, :]).reshape(len(a), -1)
        pars = np.zeros((len(a), 9))
        if derivatives:
            dpower = np.stack([d*a**(d-1)/4. if d else np.zeros_like(a) for d in range(self.accommodation_degree+1)], -1)
            dbA = (dpower[:, :, None]*gaze[:, None, :]).reshape(len(a), -1)
            dbade = (powers[:, :, None]*dade[:, None, :]).reshape(len(a), -1)
            dbbde = (powers[:, :, None]*dbde[:, None, :]).reshape(len(a), -1)
            dp = np.zeros((len(a), 9, 3))
        for parameter in range(9):
            ids = np.flatnonzero(self.parameter_for_coefficient == parameter)
            pars[:, parameter] = basis[:, ids]@self.coefficients[ids]
            if derivatives:
                for j, derivative in enumerate((dbA, dbade, dbbde)):
                    dp[:, parameter, j] = derivative[:, ids]@self.coefficients[ids]
        pars[:, :2] += 1.
        return (pars, basis, dp) if derivatives else (pars, basis)

    def _spatial(self, accommodation, pars, jac=False, diagnostics=False):
        a = np.asarray(accommodation).ravel()/4.
        scale = 1.+self.calibration[0]*a
        kappa = self.calibration[1]*a+self.calibration[2]*a*a+pars[:, 2]
        v = scale[:, None, None]*(self.reference_mm/self.length_mm)[None]
        if self.composition == "projective_first":
            h, den, hj, hp = _project(v, pars, jac)
            w, brown_det, bj, bp = _brown(h, pars, jac)
            output, rt, rr, rj, rp = _radial(w, kappa, jac)
            if jac:
                local = rj@bj@hp+rj@bp+rp
                dv = rj@bj@hj
        else:
            w, brown_det, bj, bp = _brown(v, pars, jac)
            d, rt, rr, rj, rp = _radial(w, kappa, jac)
            output, den, hj, hp = _project(d, pars, jac)
            if jac:
                local = hp+hj@rj@bp+hj@rp
                dv = hj@rj@bj
        result = {"prediction": self.length_mm*output}
        if diagnostics:
            result.update(denominator=den, radial_tangential=rt, radial_radial=rr, brown_determinant=brown_det)
        if jac:
            result["local_parameter_jacobian"] = self.length_mm*local
            result["input_jacobian"] = self.length_mm*dv
        return result

    def evaluate(self, accommodation, ade, bde, *, theta_jac=False, calibration_jac=False, state_jac=False, diagnostics=False):
        A, ade, bde = np.broadcast_arrays(np.asarray(accommodation, float), np.asarray(ade, float), np.asarray(bde, float))
        A, ade, bde = A.ravel(), ade.ravel(), bde.ravel()
        jac = theta_jac or calibration_jac or state_jac
        if state_jac:
            pars, basis, dp = self.maps(A, ade, bde, True)
        else:
            pars, basis = self.maps(A, ade, bde)
        result = self._spatial(A, pars, jac, diagnostics)
        if not jac:
            return result
        local, dv = result.pop("local_parameter_jacobian"), result.pop("input_jacobian")
        a, ref = A/4., self.reference_mm/self.length_mm
        if theta_jac:
            result["theta_jacobian"] = local[..., self.parameter_for_coefficient]*basis[:, None, None, :]
        if calibration_jac:
            cal = np.empty(result["prediction"].shape+(3,))
            cal[..., 0] = (dv@(a[:, None, None]*ref[None])[..., None])[..., 0]
            cal[..., 1], cal[..., 2] = local[..., 2]*a[:, None, None], local[..., 2]*(a*a)[:, None, None]
            result["calibration_jacobian"] = cal
        if state_jac:
            state = np.einsum("nfcp,npd->nfcd", local, dp, optimize=True)
            state[..., 0] += (dv@(self.calibration[0]/4.*ref)[None, ..., None])[..., 0]
            state[..., 0] += local[..., 2]*((self.calibration[1]+2*self.calibration[2]*a)/4.)[:, None, None]
            result["state_jacobian"] = state
        return result

    def prepare_gaze(self, ade, bde):
        return PreparedGaze(self, ade, bde)

    def save(self, path, **metadata):
        np.savez_compressed(path, reference_mm=self.reference_mm, calibration=self.calibration,
            coefficients=self.coefficients, accommodation_degree=np.array(self.accommodation_degree),
            composition=np.array(self.composition), length_mm=np.array(self.length_mm),
            coefficient_labels=np.array(self.labels), parameter_count=np.array(self.parameter_count), **metadata)

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            return cls(data["reference_mm"], data["calibration"], data["coefficients"],
                int(data["accommodation_degree"]), str(data["composition"]), float(data["length_mm"]))


class PreparedGaze:
    """Exact collapse of shared gaze terms, for repeated unknown-A trials."""
    def __init__(self, model, ade, bde):
        self.model = model
        gaze = spatial_basis(ade, bde)
        theta = model.coefficients.reshape(model.accommodation_degree+1, 12)
        self.maps = np.zeros((len(gaze), 9, model.accommodation_degree+1))
        for p in range(9):
            ids = [i for i, term in enumerate(SPATIAL) if term[0] == p]
            self.maps[:, p, :] = gaze[:, ids]@theta[:, ids].T

    def predict(self, accommodation, indices=None):
        A = np.asarray(accommodation).ravel()
        powers = np.stack([(A/4.)**d for d in range(self.model.accommodation_degree+1)], -1)
        coefficients = self.maps if indices is None else self.maps[indices]
        pars = np.einsum("npd,nd->np", coefficients, powers, optimize=True)
        pars[:, :2] += 1.
        return self.model._spatial(A, pars)["prediction"]


def fit_model(states, centered, accommodation_degree=1, composition="projective_first"):
    """Only caller-supplied calibration rows are ever used in the fit."""
    states, centered = np.asarray(states), np.asarray(centered)
    anchor = np.flatnonzero(np.all(np.isclose(states, [0., 0., 0.], atol=1e-10), axis=1))
    if len(anchor) != 1:
        raise ValueError("Calibration must contain exactly one A0/zero-gaze reference")
    reference = centered[anchor[0]].copy()
    model = SparseModel(reference, np.zeros(3), np.zeros(12*(accommodation_degree+1)), accommodation_degree, composition)
    axis = np.all(np.isclose(states[:, 1:], 0., atol=1e-10), axis=1)
    if np.sum(axis) != 3:
        raise ValueError("Exactly three on-axis calibration accommodations are required")

    def cal_fun(c):
        model.calibration = np.asarray(c).copy()
        return ((model.evaluate(*states[axis].T)["prediction"][:, OUTER]-centered[axis][:, OUTER])*1000.).ravel()

    def cal_jac(c):
        model.calibration = np.asarray(c).copy()
        return (model.evaluate(*states[axis].T, calibration_jac=True)["calibration_jacobian"][:, OUTER]*1000.).reshape(-1, 3)

    calfit = least_squares(cal_fun, np.zeros(3), jac=cal_jac, x_scale="jac", max_nfev=150,
                           ftol=1e-12, xtol=1e-12, gtol=1e-12)
    model.calibration = calfit.x.copy()

    def gaze_fun(theta):
        model.coefficients = np.asarray(theta).copy()
        return ((model.evaluate(*states.T)["prediction"][:, OUTER]-centered[:, OUTER])*1000.).ravel()

    def gaze_jac(theta):
        model.coefficients = np.asarray(theta).copy()
        return (model.evaluate(*states.T, theta_jac=True)["theta_jacobian"][:, OUTER]*1000.).reshape(-1, len(theta))

    fit = least_squares(gaze_fun, np.zeros_like(model.coefficients), jac=gaze_jac, x_scale="jac", max_nfev=200,
                        ftol=1e-11, xtol=1e-11, gtol=1e-11)
    model.coefficients = fit.x.copy()
    details = {"calibration_success": bool(calfit.success), "calibration_message": calfit.message,
        "calibration_nfev": calfit.nfev, "calibration_cost_um2_over_2": float(calfit.cost),
        "gaze_success": bool(fit.success), "gaze_message": fit.message, "gaze_nfev": fit.nfev,
        "gaze_cost_um2_over_2": float(fit.cost), "gaze_optimality": float(fit.optimality),
        "initialization": "zero coefficients and identity; calibration rows only"}
    return model, details
