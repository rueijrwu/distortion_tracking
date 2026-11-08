"""Full2D cubic gaze map with shared accommodation-dependent coefficients.

At every estimated accommodation the two outputs use all ten spatial cubic
monomials. Accommodation changes their coefficients through order0/1/2
Legendre functions; no true state is required at inference.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from p4_a0_bivariate_polynomial import POWERS, TERMS, design
from p4_a0_only_models import design_geometry


def accommodation_basis(accommodation_d, order, derivative=False):
    if order not in (0, 1, 2):
        raise ValueError("Only predeclared accommodation orders0/1/2 are supported")
    A = np.asarray(accommodation_d)
    xi = A/2.0-1.0
    if derivative:
        terms = (np.zeros_like(xi), np.full_like(xi, .5), 1.5*xi)
    else:
        terms = (np.ones_like(xi), xi, (3*xi*xi-1)/2)
    return np.stack(terms[:order+1], axis=1)


@dataclass
class AccommodationCubic:
    accommodation_order: int
    mean_mm: np.ndarray
    scale_mm: np.ndarray
    coefficients: np.ndarray  # accommodation-block, ten spatial terms, outputs[ADE,BDE]

    @property
    def coefficient_count(self):
        return 20*(self.accommodation_order+1)

    def features(self, q_mm, current_accommodation_d):
        q = np.asarray(q_mm)
        A = np.broadcast_to(np.asarray(current_accommodation_d), (len(q),))
        spatial = design((q-self.mean_mm)/self.scale_mm, 3)
        legendre = accommodation_basis(A, self.accommodation_order)
        return np.concatenate([spatial*legendre[:, j, None]
                               for j in range(self.accommodation_order+1)], axis=1)

    def predict(self, q_mm, current_accommodation_d):
        return self.features(q_mm, current_accommodation_d)@self.coefficients

    def jacobian(self, q_mm, current_accommodation_d):
        q = np.asarray(q_mm)
        A = np.broadcast_to(np.asarray(current_accommodation_d), (len(q),))
        x, y = ((q-self.mean_mm)/self.scale_mm).T
        spatial = design(np.stack((x, y), axis=1), 3)
        dx = np.stack([i*x**(i-1)*y**j if i else np.zeros_like(x) for i, j in POWERS], axis=1)/self.scale_mm[0]
        dy = np.stack([j*x**i*y**(j-1) if j else np.zeros_like(y) for i, j in POWERS], axis=1)/self.scale_mm[1]
        legendre = accommodation_basis(A, self.accommodation_order)
        dlegendre = accommodation_basis(A, self.accommodation_order, derivative=True)
        columns = []
        for spatial_part, temporal in ((dx, legendre), (dy, legendre), (spatial, dlegendre)):
            features = np.concatenate([spatial_part*temporal[:, j, None]
                                       for j in range(self.accommodation_order+1)], axis=1)
            columns.append(features@self.coefficients)
        return np.stack(columns, axis=2)  # outputs[ADE,BDE], inputs[qx,qy,current A]

    @classmethod
    def fit(cls, q_mm, true_gaze_deg, known_calibration_A_d, accommodation_order):
        q, gaze = np.asarray(q_mm, float), np.asarray(true_gaze_deg, float)
        A = np.asarray(known_calibration_A_d, float)
        if q.ndim != 2 or q.shape[1] != 2 or gaze.shape != q.shape or A.shape != (len(q),):
            raise ValueError("Expected measured q[N,2], true calibration gaze[N,2], known calibration A[N]")
        if not np.isfinite(q).all() or not np.isfinite(gaze).all() or not np.isfinite(A).all():
            raise ValueError("Nonfinite calibration observation/label")
        mean, scale = q.mean(axis=0), q.std(axis=0)
        if np.any(scale <= 0):
            raise ValueError("Degenerate detector normalization")
        count = 10*(accommodation_order+1)
        model = cls(accommodation_order, mean, scale, np.zeros((count, 2)))
        matrix = model.features(q, A)
        coefficients = np.stack([np.linalg.lstsq(matrix, gaze[:, axis], rcond=None)[0]
                                 for axis in range(2)], axis=1)
        model.coefficients = coefficients
        geometry = design_geometry(matrix)
        raw_spatial = design(q, 3)
        temporal = accommodation_basis(A, accommodation_order)
        raw_matrix = np.concatenate([raw_spatial*temporal[:, j, None]
                                     for j in range(accommodation_order+1)], axis=1)
        info = {"full_rank": bool(geometry["rank"] == count), "required_rank_per_output": count,
                "rank_per_output": {"ADE": geometry["rank"], "BDE": geometry["rank"]},
                "normalized_design": geometry, "raw_detector_powers": design_geometry(raw_matrix),
                "q_normalization_mean_mm": mean.tolist(), "q_normalization_scale_mm": scale.tolist()}
        return model, info

    def save(self, path):
        np.savez_compressed(path, accommodation_order=np.array(self.accommodation_order),
            mean_mm=self.mean_mm, scale_mm=self.scale_mm, coefficients=self.coefficients,
            powers=np.array(POWERS), spatial_terms=np.array(TERMS),
            accommodation_basis=np.array(["1", "xi", "(3*xi^2-1)/2"][:self.accommodation_order+1]),
            xi_definition=np.array("xi=A/2-1; A in D"), feature_order=np.array("Legendre blocks, then ten spatial terms"),
            output_axes=np.array(["ADE", "BDE"]), coefficient_count=np.array(self.coefficient_count))

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            order = int(data["accommodation_order"])
            if order not in (0, 1, 2) or int(data["coefficient_count"]) != 20*(order+1):
                raise ValueError("Accommodation coefficient count changed")
            if not np.array_equal(data["powers"], POWERS):
                raise ValueError("Full2D cubic spatial convention changed")
            return cls(order, data["mean_mm"], data["scale_mm"], data["coefficients"])


def self_check():
    rng = np.random.default_rng(20261107)
    q = np.tile(np.array([(x, y) for x in (-2., -1., 0., 1., 2.)
                           for y in (-3., -2., -1., 0., 1.)]), (3, 1))
    A = np.repeat([0., 2., 4.], 25)
    probe = np.array([[-1.37, -.73], [.38, -2.26], [1.62, .19]])
    probe_A = np.array([.3, 2.7, 3.8])
    results = {}
    for order in (0, 1, 2):
        prototype = AccommodationCubic(order, np.array([.3, -1.3]), np.array([1.2, .9]),
                                      rng.normal(0, .2, (10*(order+1), 2)))
        target = prototype.predict(q, A)
        fitted, info = AccommodationCubic.fit(q, target, A, order)
        recovery = float(np.max(np.abs(fitted.predict(probe, probe_A)-prototype.predict(probe, probe_A))))
        analytic = fitted.jacobian(probe, probe_A)
        numerical = np.empty_like(analytic)
        for axis in range(3):
            plus, minus, plus_A, minus_A = probe.copy(), probe.copy(), probe_A.copy(), probe_A.copy()
            if axis < 2:
                plus[:, axis] += 1e-6
                minus[:, axis] -= 1e-6
            else:
                plus_A += 1e-6
                minus_A -= 1e-6
            numerical[:, :, axis] = (fitted.predict(plus, plus_A)-fitted.predict(minus, minus_A))/(2e-6)
        derivative = float(np.max(np.abs(analytic-numerical)))
        if not info["full_rank"] or recovery > 1e-9 or derivative > 1e-7:
            raise AssertionError((order, recovery, derivative))
        results[str(order)] = {"synthetic_recovery_deg": recovery,
            "max_q_and_A_Jacobian_FD_error": derivative, "rank_per_output": 10*(order+1)}
    return results
