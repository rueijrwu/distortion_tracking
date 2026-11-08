"""Independent gaze-only P4 shape and separate center-signal cubic models.

There is no accommodation input, parameter, optimizer, or fitted prior here.
The forward reference is a measured zero-gaze A0 calibration observation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

L_MM = 1.235429592627  # fixed coordinate-unit convention, not a fitted value
GAZE_SCALE_DEG = 12.4
OUTER = np.array([0, 1, 2, 3, 5, 6, 7, 8])
GROUPS12 = (
    ("m11", ((2, 0), (0, 2))), ("m22", ((2, 0), (0, 2))),
    ("g", ((2, 0), (0, 2))), ("m12", ((1, 1),)),
    ("m21", ((1, 1),)), ("qx", ((1, 0),)), ("p2", ((1, 0),)),
    ("qy", ((0, 1),)), ("p1", ((0, 1),)),
)
GROUPS33 = (
    ("m11", ((2, 0), (0, 2), (4, 0), (2, 2), (0, 4))),
    ("m22", ((2, 0), (0, 2), (4, 0), (2, 2), (0, 4))),
    ("g", ((2, 0), (0, 2), (4, 0), (2, 2), (0, 4))),
    ("m12", ((1, 1), (3, 1), (1, 3))),
    ("m21", ((1, 1), (3, 1), (1, 3))),
    ("qx", ((1, 0), (3, 0), (1, 2))),
    ("p2", ((1, 0), (3, 0), (1, 2))),
    ("qy", ((0, 1), (2, 1), (0, 3))),
    ("p1", ((0, 1), (2, 1), (0, 3))),
)


def design_geometry(matrix):
    x = np.asarray(matrix, dtype=float)
    singular = np.linalg.svd(x, compute_uv=False)
    tolerance = max(x.shape)*np.finfo(float).eps*(singular[0] if len(singular) else 0)
    rank = int(np.sum(singular > tolerance))
    norms = np.linalg.norm(x, axis=0)
    scaled = x/np.where(norms > 0, norms, 1.0)
    ss = np.linalg.svd(scaled, compute_uv=False)
    result = {
        "rows": x.shape[0], "columns": x.shape[1], "rank": rank,
        "rank_tolerance": float(tolerance), "singular_values": singular.tolist(),
        "raw_condition": float(singular[0]/singular[-1]) if rank == x.shape[1] else None,
        "column_scaled_singular_values": ss.tolist(),
        "column_scaled_condition": float(ss[0]/ss[-1]) if rank == x.shape[1] else None,
        "column_norms": norms.tolist(),
    }
    if rank == x.shape[1]:
        _, sv, vt = np.linalg.svd(x, full_matrices=False)
        geometry_cov = (vt.T/(sv*sv))@vt
        sd = np.sqrt(np.maximum(np.diag(geometry_cov), 0))
        corr = geometry_cov/np.outer(sd, sd)
        result["unit_residual_coefficient_correlation"] = corr.tolist()
        result["max_absolute_offdiagonal_correlation"] = float(np.max(np.abs(corr-np.eye(len(sd)))))
    return result


@dataclass
class GazeOnlyShape:
    reference_mm: np.ndarray
    coefficients: np.ndarray
    order: str = "PF"
    spatial_terms: int = 12

    def __post_init__(self):
        self.reference_mm = np.asarray(self.reference_mm, dtype=float)
        self.coefficients = np.asarray(self.coefficients)
        if self.reference_mm.shape != (9, 2) or self.spatial_terms not in (12, 33):
            raise ValueError("Expected nine-point reference and12 or33 gaze terms")
        if self.coefficients.shape != (self.spatial_terms,) or self.order not in ("PF", "DF"):
            raise ValueError("Invalid gaze-only coefficient count/order")
        if not np.array_equal(self.reference_mm[4], np.zeros(2)):
            raise ValueError("Reference must be centered on the labeled central field")

    @property
    def groups(self):
        return GROUPS12 if self.spatial_terms == 12 else GROUPS33

    @property
    def labels(self):
        return [f"{name}:tx^{i}ty^{j}" for name, powers in self.groups for i, j in powers]

    def evaluate(self, gaze_deg, coefficients=None, diagnostics=False):
        gaze = np.asarray(gaze_deg)
        if gaze.ndim != 2 or gaze.shape[1] != 2:
            raise ValueError("Gaze must have columns[ADE,BDE]")
        c = self.coefficients if coefficients is None else np.asarray(coefficients)
        tx, ty = gaze[:, 1]/GAZE_SCALE_DEG, gaze[:, 0]/GAZE_SCALE_DEG
        pars = {}
        offset = 0
        for name, powers in self.groups:
            basis = np.stack([tx**i*ty**j for i, j in powers], axis=1)
            pars[name] = basis@c[offset:offset+len(powers)]
            offset += len(powers)
        m11, m22 = 1+pars["m11"], 1+pars["m22"]
        m12, m21, qx, qy = (pars[k] for k in ("m12", "m21", "qx", "qy"))
        p1, p2, radial = (pars[k][:, None] for k in ("p1", "p2", "g"))
        b = np.broadcast_to(self.reference_mm/L_MM, (len(gaze), 9, 2))

        def project(x, y):
            den = 1+qx[:, None]*x+qy[:, None]*y
            return ((m11[:, None]*x+m12[:, None]*y)/den,
                    (m21[:, None]*x+m22[:, None]*y)/den, den)

        def distort(x, y):
            wx = x+2*p1*x*y+p2*(3*x*x+y*y)
            wy = y+p1*(x*x+3*y*y)+2*p2*x*y
            brown = ((1+2*p1*y+6*p2*x)*(1+6*p1*y+2*p2*x)
                     -(2*p1*x+2*p2*y)**2)
            radius2 = wx*wx+wy*wy  # analytic, no conjugation, for complex-step
            tangential = 1+radial*radius2
            radial_derivative = 1+3*radial*radius2
            return wx*tangential, wy*tangential, brown, tangential, radial_derivative

        if self.order == "PF":
            x, y, denominator = project(b[..., 0], b[..., 1])
            x, y, brown, tangential, radial_derivative = distort(x, y)
        else:
            x, y, brown, tangential, radial_derivative = distort(b[..., 0], b[..., 1])
            x, y, denominator = project(x, y)
        prediction = L_MM*np.stack((x, y), axis=-1)
        if not diagnostics:
            return prediction
        determinant = np.broadcast_to((m11*m22-m12*m21)[:, None], denominator.shape)
        checks = {"denominator": denominator, "brown_determinant": brown,
                  "radial_tangential": tangential, "radial_radial": radial_derivative,
                  "homography_determinant": determinant}
        valid = np.isfinite(prediction).all(axis=(1, 2))
        for value in checks.values():
            valid &= np.all(np.isfinite(value) & (value > 0), axis=1)
        return {"prediction": prediction, "valid": valid, **checks}

    def save(self, path):
        np.savez_compressed(path, reference_mm=self.reference_mm, coefficients=self.coefficients,
                            order=np.array(self.order), spatial_terms=np.array(self.spatial_terms),
                            gaze_scale_deg=np.array(GAZE_SCALE_DEG), L_mm=np.array(L_MM),
                            coefficient_labels=np.array(self.labels))

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            if float(data["gaze_scale_deg"]) != GAZE_SCALE_DEG or float(data["L_mm"]) != L_MM:
                raise ValueError("Gaze-only coordinate convention mismatch")
            return cls(data["reference_mm"], data["coefficients"],
                       str(data["order"].item()), int(data["spatial_terms"]))


@dataclass
class CenterCubic:
    mean_mm: np.ndarray
    scale_mm: np.ndarray
    coefficients_xy: np.ndarray  # x->BDE,y->ADE, four powers per axis

    def predict(self, q_mm):
        q = np.asarray(q_mm)
        z = (q-self.mean_mm)/self.scale_mm
        values = np.stack([sum(self.coefficients_xy[axis, k]*z[:, axis]**k
                               for k in range(4)) for axis in range(2)], axis=1)
        return values[:, ::-1]

    @classmethod
    def fit(cls, q_mm, gaze_deg):
        q, gaze = np.asarray(q_mm, float), np.asarray(gaze_deg, float)
        mean, scale = q.mean(axis=0), q.std(axis=0)
        if np.any(scale <= 0) or not np.isfinite(q).all():
            raise ValueError("Degenerate/nonfinite calibration center signal")
        z = (q-mean)/scale
        coefficients, diagnostics = [], {}
        for axis, label in enumerate(("x_to_BDE", "y_to_ADE")):
            design = np.stack([z[:, axis]**k for k in range(4)], axis=1)
            c = np.linalg.lstsq(design, gaze[:, 1-axis], rcond=None)[0]
            coefficients.append(c)
            raw_design = np.stack([q[:, axis]**k for k in range(4)], axis=1)
            diagnostics[label] = {"standardized": design_geometry(design),
                                  "raw_detector_powers": design_geometry(raw_design)}
        return cls(mean, scale, np.array(coefficients)), diagnostics

    def save(self, path):
        np.savez_compressed(path, mean_mm=self.mean_mm, scale_mm=self.scale_mm,
                            coefficients_xy=self.coefficients_xy, powers=np.arange(4),
                            output_axes=np.array(["BDE", "ADE"]))

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            return cls(data["mean_mm"], data["scale_mm"], data["coefficients_xy"])


def self_check():
    reference = np.array([(x, y) for x in (-.8, 0., .8) for y in (-.9, 0., .9)])
    gaze = np.array([[3.1, -7.2], [-8.4, 5.7], [11.6, -4.3]])
    rng = np.random.default_rng(20261107)
    results = {}
    for count in (12, 33):
        for order in ("PF", "DF"):
            c = rng.normal(0, .003, count)
            model = GazeOnlyShape(reference, c, order, count)
            cs, fd = [], []
            for j in range(count):
                complex_c = c.astype(complex)
                complex_c[j] += 1e-25j
                cs.append(np.imag(model.evaluate(gaze, complex_c))/1e-25)
                plus, minus = c.copy(), c.copy()
                plus[j] += 1e-6
                minus[j] -= 1e-6
                fd.append((model.evaluate(gaze, plus)-model.evaluate(gaze, minus))/(2e-6))
            error = float(np.max(np.abs(np.array(cs)-np.array(fd))))
            zero_error = float(np.max(np.abs(model.evaluate(np.zeros((1, 2)))[0]-reference)))
            if error > 1e-8 or zero_error > 1e-14 or not model.evaluate(gaze, diagnostics=True)["valid"].all():
                raise AssertionError((count, order, error, zero_error))
            results[f"{order}{count}"] = {"max_coefficient_J_error_mm": error,
                                            "zero_gaze_identity_error_mm": zero_error}
    return results
