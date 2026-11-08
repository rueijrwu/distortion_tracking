"""Two separate symmetry-preserving cubic gaze outputs, ten shared values.

Inputs are measured P4 central-ray minus P1-mean detector coordinates. No
accommodation, affine comparator, or gaze feature is supplied at inference.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from p4_a0_only_models import design_geometry

BDE_TERMS = ("x", "x*y", "x*y^2", "x^3")
ADE_TERMS = ("1", "y", "y^2", "y^3", "x^2", "x^2*y")


def bases(x, y):
    return (np.stack((x, x*y, x*y*y, x*x*x), axis=1),
            np.stack((np.ones_like(y), y, y*y, y*y*y, x*x, x*x*y), axis=1))


def reflection_check(q_mm, gaze_deg, tolerance_mm=1e-8):
    q, gaze = np.asarray(q_mm, float), np.asarray(gaze_deg, float)
    index = {tuple(np.round(g, 8)): row for row, g in enumerate(gaze)}
    pairs = []
    for row in np.flatnonzero(gaze[:, 1] > 1e-9):
        mirror = index.get(tuple(np.round([gaze[row, 0], -gaze[row, 1]], 8)))
        if mirror is None:
            raise ValueError("Missing calibration BDE-reflection pair")
        pairs.append((row, mirror))
    zero = np.isclose(gaze[:, 1], 0, atol=1e-9, rtol=0)
    if not pairs or not np.any(zero):
        raise ValueError("Reflection gate needs both paired and zero-BDE observations")
    x_error = max(abs(q[i, 0]+q[j, 0]) for i, j in pairs)
    y_error = max(abs(q[i, 1]-q[j, 1]) for i, j in pairs)
    zero_error = float(np.max(np.abs(q[zero, 0])))
    passed = bool(np.isfinite(q).all() and max(x_error, y_error, zero_error) <= tolerance_mm)
    return {"passed": passed, "tolerance_mm": tolerance_mm, "paired_locations": len(pairs),
            "max_qx_odd_error_mm": float(x_error), "max_qy_even_error_mm": float(y_error),
            "max_zero_BDE_qx_mm": zero_error}


@dataclass
class CrossAxisCubic:
    sx_mm: float
    mean_y_mm: float
    sy_mm: float
    bde_coefficients: np.ndarray
    ade_coefficients: np.ndarray

    def normalized(self, q_mm):
        q = np.asarray(q_mm)
        return q[:, 0]/self.sx_mm, (q[:, 1]-self.mean_y_mm)/self.sy_mm

    def predict(self, q_mm):
        bde, ade = bases(*self.normalized(q_mm))
        return np.stack((ade@self.ade_coefficients, bde@self.bde_coefficients), axis=1)

    def jacobian(self, q_mm):
        x, y = self.normalized(q_mm)
        b, a = self.bde_coefficients, self.ade_coefficients
        jac = np.empty((len(x), 2, 2))  # output[ADE,BDE], input[qx,qy]
        jac[:, 0, 0] = (2*a[4]*x+2*a[5]*x*y)/self.sx_mm
        jac[:, 0, 1] = (a[1]+2*a[2]*y+3*a[3]*y*y+a[5]*x*x)/self.sy_mm
        jac[:, 1, 0] = (b[0]+b[1]*y+b[2]*y*y+3*b[3]*x*x)/self.sx_mm
        jac[:, 1, 1] = (b[1]*x+2*b[2]*x*y)/self.sy_mm
        return jac

    @classmethod
    def fit(cls, q_mm, gaze_deg):
        q, gaze = np.asarray(q_mm, float), np.asarray(gaze_deg, float)
        if q.ndim != 2 or q.shape[1] != 2 or gaze.shape != q.shape or not np.isfinite(q).all():
            raise ValueError("Expected finite calibration q and true gaze[N,2]")
        sx = float(np.sqrt(np.mean(q[:, 0]**2)))  # x origin stays exactly zero
        mean_y, sy = float(q[:, 1].mean()), float(q[:, 1].std())
        if sx <= 0 or sy <= 0:
            raise ValueError("Degenerate detector normalization")
        normalized = bases(q[:, 0]/sx, (q[:, 1]-mean_y)/sy)
        raw = bases(q[:, 0], q[:, 1])
        b = np.linalg.lstsq(normalized[0], gaze[:, 1], rcond=None)[0]
        a = np.linalg.lstsq(normalized[1], gaze[:, 0], rcond=None)[0]
        geometry = {}
        for axis, label in enumerate(("BDE", "ADE")):
            geometry[label] = {"normalized_design": design_geometry(normalized[axis]),
                               "raw_detector_powers": design_geometry(raw[axis])}
        good = geometry["BDE"]["normalized_design"]["rank"] == 4 and geometry["ADE"]["normalized_design"]["rank"] == 6
        diagnostics = {"full_rank": bool(good), "geometry": geometry,
                       "normalization": {"qx_offset_mm": 0.0, "sx_RMS_mm": sx,
                                         "qy_mean_mm": mean_y, "sy_std_mm": sy}}
        return cls(sx, mean_y, sy, b, a), diagnostics

    def save(self, path):
        np.savez_compressed(path, sx_mm=np.array(self.sx_mm), mean_y_mm=np.array(self.mean_y_mm),
                            sy_mm=np.array(self.sy_mm), bde_coefficients=self.bde_coefficients,
                            ade_coefficients=self.ade_coefficients,
                            BDE_terms=np.array(BDE_TERMS), ADE_terms=np.array(ADE_TERMS),
                            coefficient_count=np.array(10), qx_offset_mm=np.array(0.0))

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            if float(data["qx_offset_mm"]) != 0 or int(data["coefficient_count"]) != 10:
                raise ValueError("C10 symmetry/count convention changed")
            return cls(float(data["sx_mm"]), float(data["mean_y_mm"]), float(data["sy_mm"]),
                       data["bde_coefficients"], data["ade_coefficients"])


def self_check():
    prototype = CrossAxisCubic(1.2, -1.3, .9,
        np.array([-7.1, .14, -.06, .09]), np.array([.2, 6.5, -.07, .04, .13, -.09]))
    q = np.array([(x, y) for x in (-2., -1., 0., 1., 2.) for y in (-3., -2., -1., 0., 1.)])
    target = prototype.predict(q)
    fitted, geometry = CrossAxisCubic.fit(q, target)
    probe = np.array([[-1.37, -.73], [.38, -2.26], [1.62, .19]])
    recovery = float(np.max(np.abs(fitted.predict(probe)-prototype.predict(probe))))
    analytic = fitted.jacobian(probe)
    numerical = np.empty_like(analytic)
    for axis in range(2):
        plus, minus = probe.copy(), probe.copy()
        plus[:, axis] += 1e-6
        minus[:, axis] -= 1e-6
        numerical[:, :, axis] = (fitted.predict(plus)-fitted.predict(minus))/(2e-6)
    derivative = float(np.max(np.abs(analytic-numerical)))
    reflected = probe.copy()
    reflected[:, 0] *= -1
    parity = float(np.max(np.abs(fitted.predict(reflected)-fitted.predict(probe)*[1., -1.])))
    if not geometry["full_rank"] or recovery > 1e-10 or derivative > 1e-7 or parity > 1e-12:
        raise AssertionError((recovery, derivative, parity, geometry["full_rank"]))
    return {"synthetic_cross_cubic_recovery_deg": recovery,
            "max_state_Jacobian_FD_error_deg_per_mm": derivative,
            "BDE_reflection_error_deg": parity, "rank_BDE": 4, "rank_ADE": 6}
