"""Frozen independent A=0 components for the combined P1/P4 model.

The P4 PF12 map supplies only a center-relative shape scaffold. Its predicted
mean is adjusted analytically to the independently fitted P4 B3 centroid bias,
while the labeled central field remains at zero. New accommodation terms live
in a separate extension and vanish at A=0.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from p1_centroid_bias import P1CentroidBias
from p4_a0_bivariate_polynomial import BivariatePolynomial
from p4_a0_only_models import GazeOnlyShape

ROOT = Path(__file__).resolve().parents[1]
INDEPENDENT_CAL = ROOT / "data/p1_p4_a0_independent_gaze/calibration/summary.json"
A0_SHAPE_CAL = ROOT / "data/p4_p1_a0_only/calibration/summary.json"
OUTER = np.array([0, 1, 2, 3, 5, 6, 7, 8])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class FrozenA0:
    p1_bias: P1CentroidBias
    p4_bias: P1CentroidBias
    p1_gaze: BivariatePolynomial
    p4_gaze: BivariatePolynomial
    p4_shape: GazeOnlyShape
    source_hashes: dict

    def p4_relative_shape(self, gaze_deg, diagnostics=False):
        """P4 field4=0 and mean=the frozen B4(g), by construction."""
        evaluated = self.p4_shape.evaluate(gaze_deg, diagnostics=True)
        shape = np.array(evaluated["prediction"], copy=True)
        bias = self.p4_bias.predict(gaze_deg)
        mean_correction = bias-shape.mean(axis=1)
        shape[:, OUTER] += (9./8.)*mean_correction[:, None, :]
        if not diagnostics:
            return shape
        mean_error = np.max(np.abs(shape.mean(axis=1)-bias), axis=1)
        center_error = np.max(np.abs(shape[:, 4]), axis=1)
        return {"prediction": shape,
                "valid": evaluated["valid"] & np.isfinite(shape).all(axis=(1, 2)),
                "mean_B4_identity_error_mm": mean_error,
                "central_field_zero_error_mm": center_error}


def load_frozen_a0(independent_summary=INDEPENDENT_CAL, shape_summary=A0_SHAPE_CAL):
    independent_summary, shape_summary = Path(independent_summary), Path(shape_summary)
    independent = json.loads(independent_summary.read_text(encoding="utf-8"))
    shape_info = json.loads(shape_summary.read_text(encoding="utf-8"))["cases"]["25"]["shape_candidates"]["PF12"]
    if (not independent["A0_only"] or independent["total_fitted_scalar_values"] != 58
            or not independent["P1_and_P4_coefficients_and_normalizers_independent"]):
        raise RuntimeError("Expected the frozen independently fitted A=0 core")
    paths = {}
    for name in ("P1", "P4"):
        for part in ("bias", "gaze"):
            info = independent["channels"][name][part]
            path = Path(info["path"])
            if digest(path) != info["sha256"]:
                raise RuntimeError(f"Frozen {name} {part} changed: {path}")
            paths[f"{name}_{part}"] = path
    p4_shape_path = Path(shape_info["path"])
    if digest(p4_shape_path) != shape_info["sha256"]:
        raise RuntimeError("Frozen A=0 PF12 P4 shape changed")
    model = FrozenA0(
        P1CentroidBias.load(paths["P1_bias"]), P1CentroidBias.load(paths["P4_bias"]),
        BivariatePolynomial.load(paths["P1_gaze"]), BivariatePolynomial.load(paths["P4_gaze"]),
        GazeOnlyShape.load(p4_shape_path),
        {"independent_summary": digest(independent_summary), "shape_summary": digest(shape_summary),
         **{key: digest(path) for key, path in paths.items()}, "P4_shape": digest(p4_shape_path)})
    if (model.p1_bias.coefficient_count != 9 or model.p4_bias.coefficient_count != 9
            or model.p1_gaze.coefficient_count != 20 or model.p4_gaze.coefficient_count != 20
            or model.p4_shape.spatial_terms != 12 or model.p4_shape.order != "PF"):
        raise RuntimeError("Frozen A=0 model form/count changed")
    zero = np.zeros((1, 2))
    checked = model.p4_relative_shape(zero, diagnostics=True)
    if (not checked["valid"].all() or checked["mean_B4_identity_error_mm"].max() > 1e-12
            or checked["central_field_zero_error_mm"].max() > 1e-12):
        raise RuntimeError("P4 A=0 shape scaffold failed its B4/central-field identities")
    return model
