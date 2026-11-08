"""Fit one A=0 center-to-center cubic with frozen A=0 P4/P1 corrections."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from fit_p4_p1_both_centers import load_p1
from p1_centroid_bias import P1CentroidBias
from p4_a0_bivariate_polynomial import BivariatePolynomial, design, self_check as cubic_check
from p4_a0_only_models import GazeOnlyShape, design_geometry

ROOT = Path(__file__).resolve().parents[1]
A0_CAL = ROOT / "data/p4_p1_a0_only/calibration"
P1_CAL = ROOT / "data/p4_p1_both_centers/calibration"
SOURCE = ROOT / "data/p4_p1_limited_gaze_12p5/source_subset.npz"
OUT = ROOT / "data/p4_p1_a0_both_centers/calibration"
STENCIL = np.array([-12.4, -6.4, 0., 6.4, 12.4])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stencil_mask(gaze):
    return np.all(np.any(np.isclose(gaze[:, :, None], STENCIL, atol=1e-9, rtol=0), axis=2), axis=1)


def fit_joint_cubic(q, gaze):
    """A single 25x10 least-squares design produces the two gaze columns."""
    q, gaze = np.asarray(q, float), np.asarray(gaze, float)
    if q.ndim != 2 or q.shape[1] != 2 or gaze.shape != q.shape or not np.isfinite(q).all():
        raise ValueError("Expected finite center difference and gaze [N,2]")
    mean, scale = q.mean(axis=0), q.std(axis=0)
    if np.any(scale <= 0):
        raise ValueError("Degenerate training-only detector normalization")
    matrix = design((q - mean) / scale, 3)
    coefficients, _, rank, _ = np.linalg.lstsq(matrix, gaze, rcond=None)
    geometry = design_geometry(matrix)
    model = BivariatePolynomial(3, mean, scale, coefficients)
    return model, {"full_rank": bool(rank == 10 and geometry["rank"] == 10),
                   "joint_coefficient_shape": [10, 2], "SVD_rank": int(rank),
                   "normalized_design": geometry,
                   "raw_mm_design": design_geometry(design(q, 3)),
                   "training_mean_mm": mean.tolist(), "training_scale_mm": scale.tolist()}


def calibration_cv(q, gaze):
    folds = []
    for row in np.flatnonzero(np.any(gaze != 0, axis=1)):
        train = np.arange(len(gaze)) != row
        fitted, geometry = fit_joint_cubic(q[train], gaze[train])
        prediction = fitted.predict(q[row:row+1])[0]
        error = prediction - gaze[row]
        folds.append({"heldout_calibration_row": int(row), "heldout_gaze_deg": gaze[row].tolist(),
                      "prediction_gaze_deg": prediction.tolist(),
                      "coordinate_MSE_deg2": float(np.mean(error * error)),
                      "radial_error_deg": float(np.linalg.norm(error)), "geometry": geometry})
    if len(folds) != 24 or not all(f["geometry"]["full_rank"] for f in folds):
        raise RuntimeError("The 24 nonzero-gaze folds must all retain full rank and zero reference")
    loss = np.array([f["coordinate_MSE_deg2"] for f in folds])
    return {"folds": folds, "fold_count": 24, "all_folds_full_rank": True,
            "coordinate_MSE_deg2": float(np.mean(loss)),
            "SE_coordinate_MSE_deg2": float(np.std(loss, ddof=1) / np.sqrt(len(loss))),
            "pooled_radial_RMS_deg": float(np.sqrt(2 * np.mean(loss)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    checks = cubic_check()
    if args.check_only:
        print(json.dumps({"cubic_checks": checks}), flush=True)
        return

    a0_summary_path = A0_CAL / "summary.json"
    a0_summary = json.loads(a0_summary_path.read_text(encoding="utf-8"))
    p1_manifest_path = P1_CAL / "manifest.json"
    p1_manifest = json.loads(p1_manifest_path.read_text(encoding="utf-8"))
    source_path = Path(a0_summary["source_subset_path"])
    obs_path = A0_CAL / "a0_observations.npz"
    paired_path = P1_CAL / "paired_data.npz"
    shape_path = Path(a0_summary["cases"]["25"]["shape_candidates"]["PF12"]["path"])
    bias_path = Path(p1_manifest["P1_candidates"]["B3"]["path"])
    for path, expected in ((source_path, a0_summary["source_subset_sha256"]),
                           (obs_path, a0_summary["observation_artifact_sha256"]),
                           (paired_path, p1_manifest["paired_data_sha256"]),
                           (shape_path, a0_summary["cases"]["25"]["shape_candidates"]["PF12"]["sha256"]),
                           (bias_path, p1_manifest["P1_candidates"]["B3"]["sha256"])):
        if digest(path) != expected:
            raise RuntimeError(f"Frozen source/model changed: {path}")
    if source_path.resolve() != SOURCE.resolve() or p1_manifest["P1_selection"]["selected"] != "B3":
        raise RuntimeError("Unexpected paired source or selected P1 correction")

    # The first new-data operation is the A=0 subset. All references, fits,
    # normalizers, CV folds, and final scoring are downstream of this filter.
    with np.load(obs_path) as obs:
        gaze = obs["gaze_deg"].copy()
        y4 = obs["absolute_p4_mm"].copy()
        m1 = obs["mean_p1_absolute_mm"].copy()
        ids = obs["source_indices"].copy()
        raw_rows = obs["raw_source_subset_rows"].copy()
        cal = obs["calibration25"].astype(bool)
        holdout = obs["holdout25"].astype(bool)
        reference_row = int(obs["reference_row"])
    with np.load(source_path) as source:
        source_a0 = np.flatnonzero(source["states"][:, 0] == 0)
        source_gaze = source["states"][source_a0, 1:]
        source_y4 = source["absolute_p4_mm"][source_a0]
        source_m1 = source["mean_p1_absolute_mm"][source_a0]
        source_ids = source["source_indices"][source_a0]
    if (len(gaze) != 3969 or not np.array_equal(raw_rows, source_a0)
            or not np.array_equal(gaze, source_gaze) or not np.array_equal(y4, source_y4)
            or not np.array_equal(m1, source_m1) or not np.array_equal(ids, source_ids)):
        raise RuntimeError("A=0 observation artifact does not reproduce the raw A=0 subset")
    if (cal.sum() != 25 or holdout.sum() != 3944 or np.any(cal & holdout)
            or not np.array_equal(cal, stencil_mask(gaze))
            or len(np.unique(gaze, axis=0)) != 3969):
        raise AssertionError("A=0 calibration/test gaze identities differ")
    expected25 = np.array([(a, b) for a in STENCIL for b in STENCIL])
    if not np.allclose(np.unique(gaze[cal], axis=0), expected25, atol=1e-9, rtol=0):
        raise AssertionError("Calibration must have the exact shared 25 physical gaze labels")
    zero = np.flatnonzero(np.all(gaze == 0, axis=1))
    if len(zero) != 1 or reference_row != zero[0] or not cal[zero[0]]:
        raise AssertionError("Measured zero-gaze A=0 reference differs")

    # Re-read the raw P1 all-nine grid and verify the older P1-only artifact
    # exactly. Its gaze coordinates have no accommodation dimension.
    raw_p1_gaze, raw_p1_grids, trace_delta = load_p1()
    with np.load(paired_path) as paired:
        p1_gaze = paired["p1_gaze_deg"].copy()
        p1_absolute = paired["p1_absolute_mm"].copy()
        p1_cal = paired["p1_calibration25"].astype(bool)
        p1_bias = paired["p1_bias_mm"].copy()
    if (not np.array_equal(raw_p1_gaze, p1_gaze)
            or not np.array_equal(raw_p1_grids, p1_absolute)
            or p1_cal.sum() != 25 or not np.array_equal(p1_cal, stencil_mask(p1_gaze))
            or not np.allclose(np.unique(p1_gaze[p1_cal], axis=0), expected25, atol=1e-9, rtol=0)):
        raise RuntimeError("Raw P1 and paired calibration identities differ")
    p1_mean = p1_absolute.mean(axis=1)
    p1_center = p1_absolute[:, 4]
    if np.max(np.abs(p1_mean - p1_center - p1_bias)) > 1e-12:
        raise RuntimeError("Saved P1 mean-to-center bias differs from raw nine-point rays")
    p1_lookup = {tuple(np.round(row, 8)): i for i, row in enumerate(p1_gaze)}
    joined = np.array([p1_lookup[tuple(np.round(row, 8))] for row in gaze])
    mean_delta = float(np.max(np.abs(p1_mean[joined] - m1)))
    if mean_delta > 1e-10 or not np.allclose(gaze[cal], p1_gaze[joined[cal]], atol=1e-9, rtol=0):
        raise RuntimeError("P1 all-nine measured means or shared 25 gaze labels differ")

    shape = GazeOnlyShape.load(shape_path)
    if shape.spatial_terms != 12 or shape.order != "PF":
        raise RuntimeError("Expected frozen A=0 PF12 shape")
    reference = y4[zero[0]] - y4[zero[0], 4]
    if not np.array_equal(reference, shape.reference_mm):
        raise RuntimeError("P4 shape uses a different measured A=0 zero-gaze reference")
    bias = P1CentroidBias.load(bias_path)
    fitted_bias, bias_geometry = P1CentroidBias.fit(p1_gaze[p1_cal], p1_bias[p1_cal],
                                                    p1_bias[joined[zero[0]]], 3)
    if (bias.degree != 3 or not bias_geometry["full_rank"]
            or not np.array_equal(bias.reference_bias_mm, p1_bias[joined[zero[0]]])
            or np.max(np.abs(fitted_bias.coefficients_x_mm - bias.coefficients_x_mm)) > 1e-13
            or np.max(np.abs(fitted_bias.coefficients_y_mm - bias.coefficients_y_mm)) > 1e-13):
        raise RuntimeError("B3 coefficients do not reproduce the exact 25 P1-only poses")

    q_cal = y4[cal, 4] - p1_center[joined[cal]]
    cubic, geometry = fit_joint_cubic(q_cal, gaze[cal])
    cv = calibration_cv(q_cal, gaze[cal])
    if not geometry["full_rank"] or not np.isfinite(cubic.coefficients).all():
        raise RuntimeError("A=0 center-to-center cubic is not full rank/finite")
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(f"Fresh output directory required: {out}")
    out.mkdir(parents=True)
    cubic_path = out / "center_to_center_C20.npz"
    cubic.save(cubic_path)
    np.savez_compressed(out / "calibration_data.npz", calibration_gaze_deg=gaze[cal],
                        calibration_source_indices=ids[cal], calibration_raw_source_rows=raw_rows[cal],
                        calibration_p4_center_mm=y4[cal, 4],
                        calibration_p1_center_mm=p1_center[joined[cal]],
                        calibration_q_mm=q_cal, calibration_prediction_deg=cubic.predict(q_cal),
                        holdout_source_indices=ids[holdout])
    summary = {"experiment": "A=0 P4 PF12 plus P1 B3 mean-to-center plus newly fitted center-to-center C20",
        "source_A0_only": True, "A0_states": 3969, "calibration_poses": 25,
        "holdout_including_user_exclusion": 3944, "primary_holdout_after_user_exclusion": 3943,
        "calibration_gaze_deg": gaze[cal].tolist(), "reference_gaze_deg": [0., 0.],
        "reference_source_id": int(ids[zero[0]]),
        "P1_source_sha256": p1_manifest["P1_source"]["sha256"],
        "paired_source_sha256": digest(source_path), "A0_observations_sha256": digest(obs_path),
        "P1_paired_data_sha256": digest(paired_path), "P1_trace_center_delta_mm": trace_delta,
        "P1_mean_join_max_abs_mm": mean_delta,
        "P4_shape": {"path": str(shape_path), "sha256": digest(shape_path), "coefficients": 12},
        "P1_bias": {"path": str(bias_path), "sha256": digest(bias_path), "coefficients": 9,
                     "measured_zero_bias_mm": bias.reference_bias_mm.tolist(),
                     "shared25_refit_full_rank": bias_geometry["full_rank"]},
        "gaze_C20": {"path": str(cubic_path), "sha256": digest(cubic_path),
                     "input": "P4 labeled center minus P1 labeled center",
                     "output": "[ADE,BDE]", "coefficients": 20,
                     "joint_matrix_fit": True, "geometry": geometry, "calibration_CV": cv},
        "total_fitted_scalar_values": 41, "reference_and_normalizer_constants_counted_separately": True,
        "no_accommodation_features_or_runtime_input": True,
        "calibration_data_sha256": digest(out / "calibration_data.npz"),
        "trainer_sha256": digest(__file__), "checks": checks}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "full_rank": geometry["full_rank"],
                      "CV_radial_RMS_deg": cv["pooled_radial_RMS_deg"],
                      "total_fitted_scalar_values": 41}, indent=2), flush=True)


if __name__ == "__main__":
    main()
