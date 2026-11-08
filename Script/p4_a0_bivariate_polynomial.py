"""Full two-input order2/3 polynomial for each gaze output; no accommodation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from p4_a0_only_models import design_geometry

POWERS = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2),
          (3, 0), (2, 1), (1, 2), (0, 3))
TERMS = ("1", "x", "y", "x^2", "x*y", "y^2", "x^3", "x^2*y", "x*y^2", "y^3")


def powers_for(degree):
    if degree not in (2, 3):
        raise ValueError("Only requested full polynomial orders2 and3 are supported")
    return POWERS[:6] if degree == 2 else POWERS


def design(xy, degree):
    x, y = np.asarray(xy).T
    return np.stack([x**i*y**j for i, j in powers_for(degree)], axis=1)


@dataclass
class BivariatePolynomial:
    degree: int
    mean_mm: np.ndarray
    scale_mm: np.ndarray
    coefficients: np.ndarray  # monomials by independent[ADE,BDE] outputs

    @property
    def coefficient_count(self):
        return 2*len(powers_for(self.degree))

    def predict(self, q_mm):
        return design((np.asarray(q_mm)-self.mean_mm)/self.scale_mm, self.degree)@self.coefficients

    def jacobian(self, q_mm):
        x, y = ((np.asarray(q_mm)-self.mean_mm)/self.scale_mm).T
        dx = np.stack([i*x**(i-1)*y**j if i else np.zeros_like(x)
                       for i, j in powers_for(self.degree)], axis=1)
        dy = np.stack([j*x**i*y**(j-1) if j else np.zeros_like(y)
                       for i, j in powers_for(self.degree)], axis=1)
        return np.stack(((dx@self.coefficients)/self.scale_mm[0],
                         (dy@self.coefficients)/self.scale_mm[1]), axis=2)

    @classmethod
    def fit(cls, q_mm, true_gaze_deg, degree):
        q, gaze = np.asarray(q_mm, float), np.asarray(true_gaze_deg, float)
        if q.ndim != 2 or q.shape[1] != 2 or gaze.shape != q.shape or not np.isfinite(q).all():
            raise ValueError("Expected finite measured q and true gaze[N,2]")
        count = len(powers_for(degree))
        mean, scale = q.mean(axis=0), q.std(axis=0)
        if np.any(scale <= 0):
            raise ValueError("Degenerate training-only detector normalization")
        normalized = design((q-mean)/scale, degree)
        coefficients = np.stack([np.linalg.lstsq(normalized, gaze[:, axis], rcond=None)[0]
                                 for axis in range(2)], axis=1)
        geometry = design_geometry(normalized)
        info = {"full_rank": bool(geometry["rank"] == count),
                "required_rank_per_output": count,
                "rank_per_output": {"ADE": geometry["rank"], "BDE": geometry["rank"]},
                "normalized_design": geometry,
                "raw_detector_powers": design_geometry(design(q, degree)),
                "normalization_mean_mm": mean.tolist(), "normalization_scale_mm": scale.tolist()}
        return cls(degree, mean, scale, coefficients), info

    def save(self, path):
        count = len(powers_for(self.degree))
        np.savez_compressed(path, degree=np.array(self.degree), mean_mm=self.mean_mm,
                            scale_mm=self.scale_mm, coefficients=self.coefficients,
                            powers=np.array(powers_for(self.degree)), terms=np.array(TERMS[:count]),
                            output_axes=np.array(["ADE", "BDE"]), coefficient_count=np.array(2*count))

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            degree = int(data["degree"])
            if (int(data["coefficient_count"]) != 2*len(powers_for(degree))
                    or not np.array_equal(data["powers"], powers_for(degree))):
                raise ValueError("Full polynomial monomial/count convention changed")
            return cls(degree, data["mean_mm"], data["scale_mm"], data["coefficients"])


def self_check():
    coefficients = np.array([[.2, -.1], [.15, -7.1], [6.5, .12], [.08, -.04],
                              [-.09, .07], [-.06, .02], [.03, -.025], [-.015, .04],
                              [.02, -.03], [.01, -.012]])
    q = np.array([(x, y) for x in (-2., -1., 0., 1., 2.) for y in (-3., -2., -1., 0., 1.)])
    probe = np.array([[-1.37, -.73], [.38, -2.26], [1.62, .19]])
    results = {}
    for degree in (2, 3):
        count = len(powers_for(degree))
        prototype = BivariatePolynomial(degree, np.array([.3, -1.3]), np.array([1.2, .9]), coefficients[:count])
        fitted, geometry = BivariatePolynomial.fit(q, prototype.predict(q), degree)
        recovery = float(np.max(np.abs(fitted.predict(probe)-prototype.predict(probe))))
        analytic = fitted.jacobian(probe)
        numerical = np.empty_like(analytic)
        for axis in range(2):
            plus, minus = probe.copy(), probe.copy()
            plus[:, axis] += 1e-6
            minus[:, axis] -= 1e-6
            numerical[:, :, axis] = (fitted.predict(plus)-fitted.predict(minus))/(2e-6)
        derivative = float(np.max(np.abs(analytic-numerical)))
        x, y = ((probe-prototype.mean_mm)/prototype.scale_mm).T
        independent = np.column_stack((np.ones(len(x)), x, y, x*x, x*y, y*y,
                                        x*x*x, x*x*y, x*y*y, y*y*y))[:, :count]
        monomials = float(np.max(np.abs(prototype.predict(probe)-independent@coefficients[:count])))
        if not geometry["full_rank"] or recovery > 1e-10 or derivative > 1e-7 or monomials > 1e-12:
            raise AssertionError((degree, recovery, derivative, monomials))
        results[str(degree)] = {"synthetic_recovery_deg": recovery,
            "max_q_Jacobian_FD_error_deg_per_mm": derivative,
            "independent_monomial_check_deg": monomials, "rank_per_output": count}
    return results
