"""P4 accommodation-dependent image normalization about its measured A=0 pivot."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from p4_a0_only_models import design_geometry


@dataclass(frozen=True)
class CentralMotion:
    order: int
    pivot_mm: np.ndarray
    coefficients: np.ndarray  # detector axis [x,y] by powers a..a^order

    def __post_init__(self):
        if self.order not in (1, 2) or np.shape(self.pivot_mm) != (2,) or np.shape(self.coefficients) != (2, self.order):
            raise ValueError("Expected M1/M2 diagonal scale about a two-coordinate pivot")

    @property
    def coefficient_count(self):
        return 2 * self.order

    def scales(self, accommodation_d):
        a = np.asarray(accommodation_d, float) / 4.0
        return 1.0 + sum(self.coefficients[:, power-1] * a[..., None]**power
                         for power in range(1, self.order+1))

    def forward(self, center_a0_mm, accommodation_d):
        center = np.asarray(center_a0_mm, float)
        return self.pivot_mm + self.scales(accommodation_d) * (center-self.pivot_mm)

    def inverse(self, center_a_mm, accommodation_d):
        center = np.asarray(center_a_mm, float)
        scale = self.scales(accommodation_d)
        if np.any(scale <= 0) or not np.isfinite(scale).all():
            raise ValueError("P4 central-motion scale is nonpositive/nonfinite")
        return self.pivot_mm + (center-self.pivot_mm) / scale

    def save(self, path):
        np.savez_compressed(path, order=np.array(self.order), pivot_mm=self.pivot_mm,
                            coefficients=self.coefficients, coefficient_count=np.array(self.coefficient_count),
                            accommodation_scale_d=np.array(4.),
                            formula=np.array("cA=o+diag(1+sum(k_axis,p*(A/4)^p))*(c0-o)"))

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            order = int(data["order"])
            if int(data["coefficient_count"]) != 2*order or float(data["accommodation_scale_d"]) != 4.:
                raise ValueError("P4 center-motion convention/count changed")
            return cls(order, data["pivot_mm"], data["coefficients"])


def fit_motion(accommodation_d, center_a_mm, center_a0_mm, pivot_mm, order):
    a = np.asarray(accommodation_d, float) / 4.
    observed = np.asarray(center_a_mm, float)
    reference = np.asarray(center_a0_mm, float)
    pivot = np.asarray(pivot_mm, float)
    if observed.shape != reference.shape or observed.shape != (len(a), 2) or pivot.shape != (2,):
        raise ValueError("Expected paired central-ray positions and one pivot")
    coefficients, geometry = [], {}
    for axis, name in enumerate(("x", "y")):
        design = np.stack([a**power*(reference[:, axis]-pivot[axis])
                           for power in range(1, order+1)], axis=1)
        target = observed[:, axis]-reference[:, axis]
        fitted, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
        coefficients.append(fitted)
        geometry[name] = design_geometry(design)
        geometry[name]["lstsq_rank"] = int(rank)
    model = CentralMotion(order, pivot, np.stack(coefficients))
    sampled_scale = model.scales(np.linspace(0., 4., 401))
    full_rank = all(geometry[name]["rank"] == order and geometry[name]["lstsq_rank"] == order
                    for name in ("x", "y"))
    positive = bool(np.isfinite(sampled_scale).all() and np.all(sampled_scale > 0))
    return model, {"full_rank": bool(full_rank), "positive_scales_0_to_4D": positive,
                   "scale_minima": sampled_scale.min(axis=0).tolist(),
                   "scale_maxima": sampled_scale.max(axis=0).tolist(),
                   "geometry": geometry}


def self_check():
    rng = np.random.default_rng(20261108)
    a = np.tile([0., 2., 4.], 20)
    c0 = rng.uniform(-3., 3., (len(a), 2))
    pivot = np.array([.2, -.1])
    results = {}
    for order in (1, 2):
        coefs = np.array([[.04, -.01], [-.03, .015]])[:, :order]
        truth = CentralMotion(order, pivot, coefs)
        observed = truth.forward(c0, a)
        fitted, diagnostics = fit_motion(a, observed, c0, pivot, order)
        error = float(np.max(np.abs(fitted.coefficients-coefs)))
        inverse = float(np.max(np.abs(fitted.inverse(observed, a)-c0)))
        if not diagnostics["full_rank"] or not diagnostics["positive_scales_0_to_4D"] or error > 1e-12 or inverse > 1e-12:
            raise AssertionError((order, error, inverse))
        results[f"M{order}"] = {"coefficient_recovery_max_abs": error,
                                "inverse_roundtrip_max_mm": inverse}
    return results
