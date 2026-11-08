"""P1 gaze-only distortion-induced centroid bias; BDE parity only."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from p4_a0_only_models import design_geometry

GAZE_SCALE_DEG = 12.4


def features(gaze_deg, degree):
    if degree not in (2, 3):
        raise ValueError("Only predeclared P1 B2/B3 centroid models are supported")
    g = np.asarray(gaze_deg)
    x, y = g[:, 1]/GAZE_SCALE_DEG, g[:, 0]/GAZE_SCALE_DEG
    bx = (x, x*y, x*y*y, x*x*x)
    by = (y, y*y, x*x, y*y*y, x*x*y)
    return np.stack(bx[:2 if degree == 2 else 4], axis=1), np.stack(by[:3 if degree == 2 else 5], axis=1)


@dataclass
class P1CentroidBias:
    degree: int
    reference_bias_mm: np.ndarray
    coefficients_x_mm: np.ndarray
    coefficients_y_mm: np.ndarray

    @property
    def coefficient_count(self):
        return 5 if self.degree == 2 else 9

    def predict(self, gaze_deg):
        bx, by = features(gaze_deg, self.degree)
        return self.reference_bias_mm+np.stack((bx@self.coefficients_x_mm, by@self.coefficients_y_mm), axis=1)

    def jacobian(self, gaze_deg):
        x, y = np.asarray(gaze_deg).T[::-1]/GAZE_SCALE_DEG
        zx = np.zeros_like(x)
        dxa = np.stack((zx, x, 2*x*y, zx)[:2 if self.degree == 2 else 4], axis=1)
        dxb = np.stack((np.ones_like(x), y, y*y, 3*x*x)[:2 if self.degree == 2 else 4], axis=1)
        dya = np.stack((np.ones_like(y), 2*y, zx, 3*y*y, x*x)[:3 if self.degree == 2 else 5], axis=1)
        dyb = np.stack((zx, zx, 2*x, zx, 2*x*y)[:3 if self.degree == 2 else 5], axis=1)
        return np.stack((np.stack((dxa@self.coefficients_x_mm, dya@self.coefficients_y_mm), axis=1),
                         np.stack((dxb@self.coefficients_x_mm, dyb@self.coefficients_y_mm), axis=1)), axis=2)/GAZE_SCALE_DEG

    @classmethod
    def fit(cls, gaze_deg, measured_bias_mm, reference_bias_mm, degree):
        bx, by = features(gaze_deg, degree)
        delta = np.asarray(measured_bias_mm)-np.asarray(reference_bias_mm)
        cx = np.linalg.lstsq(bx, delta[:, 0], rcond=None)[0]
        cy = np.linalg.lstsq(by, delta[:, 1], rcond=None)[0]
        geometry = {"x": design_geometry(bx), "y": design_geometry(by)}
        full_rank = geometry["x"]["rank"] == bx.shape[1] and geometry["y"]["rank"] == by.shape[1]
        return cls(degree, np.asarray(reference_bias_mm).copy(), cx, cy), {
            "full_rank": bool(full_rank), "geometry": geometry,
            "reference_is_measured_not_fitted": True, "ADE_parity_imposed": False}

    def save(self, path):
        np.savez_compressed(path, degree=np.array(self.degree), reference_bias_mm=self.reference_bias_mm,
            coefficients_x_mm=self.coefficients_x_mm, coefficients_y_mm=self.coefficients_y_mm,
            scalar_coefficient_count=np.array(self.coefficient_count), gaze_scale_deg=np.array(GAZE_SCALE_DEG),
            x_terms=np.array(["x", "xy", "xy^2", "x^3"][:2 if self.degree == 2 else 4]),
            y_terms=np.array(["y", "y^2", "x^2", "y^3", "x^2y"][:3 if self.degree == 2 else 5]))

    @classmethod
    def load(cls, path):
        with np.load(path) as saved:
            degree = int(saved["degree"])
            if degree not in (2, 3) or float(saved["gaze_scale_deg"]) != GAZE_SCALE_DEG:
                raise ValueError("P1 centroid model convention changed")
            return cls(degree, saved["reference_bias_mm"], saved["coefficients_x_mm"], saved["coefficients_y_mm"])


def reflection_gate(gaze_deg, bias_mm, tolerance_mm=1e-8):
    gaze, bias = np.asarray(gaze_deg), np.asarray(bias_mm)
    lookup = {tuple(np.round(g, 8)): i for i, g in enumerate(gaze)}
    errors = []
    for i in np.flatnonzero(gaze[:, 1] > 1e-9):
        j = lookup[tuple(np.round([gaze[i, 0], -gaze[i, 1]], 8))]
        errors.append([abs(bias[i, 0]+bias[j, 0]), abs(bias[i, 1]-bias[j, 1])])
    maxima = np.max(errors, axis=0)
    zero_x = float(np.max(np.abs(bias[np.isclose(gaze[:, 1], 0, atol=1e-9, rtol=0), 0])))
    return {"passed": bool(max(maxima.max(), zero_x) <= tolerance_mm), "tolerance_mm": tolerance_mm,
            "BDE_odd_x_max_error_mm": float(maxima[0]), "BDE_even_y_max_error_mm": float(maxima[1]),
            "zero_BDE_x_bias_max_mm": zero_x, "ADE_symmetry_required": False}


def self_check():
    gaze = np.array([(a, b) for a in (-12.4, -6.4, 0., 6.4, 12.4) for b in (-12.4, -6.4, 0., 6.4, 12.4)])
    probe = np.array([[3.7, -8.2], [-10.6, 2.9], [11.3, -4.1]])
    results = {}
    for degree in (2, 3):
        nx, ny = (2, 3) if degree == 2 else (4, 5)
        model = P1CentroidBias(degree, np.array([0., -.058015]),
            np.array([.003, -.002, .001, -.0005])[:nx], np.array([.002, .001, -.003, .0004, -.0003])[:ny])
        fit, info = P1CentroidBias.fit(gaze, model.predict(gaze), model.reference_bias_mm, degree)
        error = float(np.max(np.abs(fit.predict(probe)-model.predict(probe))))
        numerical = np.empty((len(probe), 2, 2))
        for axis in range(2):
            plus, minus = probe.copy(), probe.copy()
            plus[:, axis] += 1e-5
            minus[:, axis] -= 1e-5
            numerical[:, :, axis] = (fit.predict(plus)-fit.predict(minus))/(2e-5)
        derivative = float(np.max(np.abs(numerical-fit.jacobian(probe))))
        gate = reflection_gate(gaze, fit.predict(gaze))
        if not info["full_rank"] or error > 1e-12 or derivative > 1e-9 or not gate["passed"]:
            raise AssertionError((degree, error, derivative))
        results[str(degree)] = {"recovery_error_mm": error, "Jacobian_FD_error_mm_per_deg": derivative,
                               "zero_bias_preserved": bool(np.array_equal(fit.predict(np.zeros((1, 2)))[0], model.reference_bias_mm))}
    return results
