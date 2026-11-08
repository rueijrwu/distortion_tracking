"""A=0 calibration of independent, equal-form P1 and P4 gaze pipelines.

Each instrument gets its own anchored B3 centroid bias and its own full 10-term
two-output cubic on its recorded absolute labeled central-ray coordinates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from fit_p4_p1_a0_both_centers import calibration_cv, digest, fit_joint_cubic, stencil_mask
from fit_p4_p1_both_centers import load_p1
from p1_centroid_bias import P1CentroidBias, reflection_gate, self_check as bias_check
from p4_a0_bivariate_polynomial import self_check as cubic_check

ROOT = Path(__file__).resolve().parents[1]
A0_CAL = ROOT / "data/p4_p1_a0_only/calibration"
P1_CAL = ROOT / "data/p4_p1_both_centers/calibration"
SOURCE = ROOT / "data/p4_p1_limited_gaze_12p5/source_subset.npz"
OUT = ROOT / "data/p1_p4_a0_independent_gaze/calibration"
STENCIL = np.array([-12.4, -6.4, 0., 6.4, 12.4])


def bias_calibration_cv(gaze, measured_bias, reference):
    folds = []
    for row in np.flatnonzero(np.any(gaze != 0, axis=1)):
        train = np.arange(len(gaze)) != row
        model, geometry = P1CentroidBias.fit(gaze[train], measured_bias[train], reference, 3)
        prediction = model.predict(gaze[row:row+1])[0]
        error_um = (prediction - measured_bias[row]) * 1000
        folds.append({"heldout_calibration_row": int(row), "heldout_gaze_deg": gaze[row].tolist(),
                      "coordinate_MSE_um2": float(np.mean(error_um * error_um)),
                      "vector_error_um": float(np.linalg.norm(error_um)),
                      "full_rank": geometry["full_rank"]})
    if len(folds) != 24 or not all(f["full_rank"] for f in folds):
        raise RuntimeError("Independent B3 must retain full rank in 24 nonzero-gaze folds")
    loss = np.array([f["coordinate_MSE_um2"] for f in folds])
    return {"folds": folds, "fold_count": 24, "all_folds_full_rank": True,
            "coordinate_MSE_um2": float(loss.mean()),
            "SE_coordinate_MSE_um2": float(loss.std(ddof=1) / np.sqrt(len(loss))),
            "vector_RMS_um": float(np.sqrt(2 * loss.mean()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    checks = {"B3": bias_check(), "G20": cubic_check()}
    if args.check_only:
        print(json.dumps({"checks": checks}), flush=True)
        return
    a0_summary_path = A0_CAL / "summary.json"
    a0_summary = json.loads(a0_summary_path.read_text(encoding="utf-8"))
    p1_manifest_path = P1_CAL / "manifest.json"
    p1_manifest = json.loads(p1_manifest_path.read_text(encoding="utf-8"))
    obs_path = A0_CAL / "a0_observations.npz"
    source_path = Path(a0_summary["source_subset_path"])
    paired_path = P1_CAL / "paired_data.npz"
    for path, expected in ((obs_path, a0_summary["observation_artifact_sha256"]),
                           (source_path, a0_summary["source_subset_sha256"]),
                           (paired_path, p1_manifest["paired_data_sha256"])):
        if digest(path) != expected:
            raise RuntimeError(f"Pinned raw data changed: {path}")
    if source_path.resolve() != SOURCE.resolve():
        raise RuntimeError("Unexpected A=0 paired source")

    # Restrict P4 to A=0 before new reference construction or fitting.
    with np.load(obs_path) as obs:
        gaze = obs["gaze_deg"].copy()
        p4_grid = obs["absolute_p4_mm"].copy()
        paired_p1_mean = obs["mean_p1_absolute_mm"].copy()
        ids = obs["source_indices"].copy()
        source_rows = obs["raw_source_subset_rows"].copy()
        cal = obs["calibration25"].astype(bool)
        holdout = obs["holdout25"].astype(bool)
        reference_row = int(obs["reference_row"])
    with np.load(source_path) as raw:
        a0_rows = np.flatnonzero(raw["states"][:, 0] == 0)
        if (not np.array_equal(a0_rows, source_rows)
                or not np.array_equal(raw["states"][a0_rows, 1:], gaze)
                or not np.array_equal(raw["absolute_p4_mm"][a0_rows], p4_grid)
                or not np.array_equal(raw["mean_p1_absolute_mm"][a0_rows], paired_p1_mean)
                or not np.array_equal(raw["source_indices"][a0_rows], ids)):
            raise RuntimeError("A=0 subset differs from pinned raw P4/P1 observations")
    if (len(gaze) != 3969 or cal.sum() != 25 or holdout.sum() != 3944
            or np.any(cal & holdout) or not np.array_equal(cal, stencil_mask(gaze))):
        raise AssertionError("Expected exactly 25 A=0 calibration and 3944 heldout poses")
    expected25 = np.array([(a, b) for a in STENCIL for b in STENCIL])
    if not np.allclose(np.unique(gaze[cal], axis=0), expected25, atol=1e-9, rtol=0):
        raise AssertionError("P4 calibration angular labels changed")
    zero = np.flatnonzero(np.all(gaze == 0, axis=1))
    if len(zero) != 1 or int(zero[0]) != reference_row or not cal[zero[0]]:
        raise AssertionError("A=0 zero-gaze calibration anchor changed")

    p1_gaze_raw, p1_grid_raw, p1_trace_delta = load_p1()
    with np.load(paired_path) as paired:
        p1_gaze = paired["p1_gaze_deg"].copy()
        p1_grid = paired["p1_absolute_mm"].copy()
        p1_cal = paired["p1_calibration25"].astype(bool)
    if (not np.array_equal(p1_gaze, p1_gaze_raw) or not np.array_equal(p1_grid, p1_grid_raw)
            or p1_cal.sum() != 25 or not np.array_equal(p1_cal, stencil_mask(p1_gaze))
            or not np.allclose(np.unique(p1_gaze[p1_cal], axis=0), expected25, atol=1e-9, rtol=0)):
        raise RuntimeError("Independent P1 raw grid or 25-pose identities differ")
    p1_lookup = {tuple(np.round(g, 8)): i for i, g in enumerate(p1_gaze)}
    joined = np.array([p1_lookup[tuple(np.round(g, 8))] for g in gaze])
    p1_grid = p1_grid[joined]
    if not np.allclose(p1_gaze[joined], gaze, atol=1e-9, rtol=0):
        raise RuntimeError("Paired P1/P4 gaze labels differ")
    p1_mean = p1_grid.mean(axis=1)
    if np.max(np.abs(p1_mean - paired_p1_mean)) > 1e-10:
        raise RuntimeError("Paired P1 mean differs from its own nine raw rays")
    grids = {"P1": p1_grid, "P4": p4_grid}
    bias_models, gaze_models, channel_summaries = {}, {}, {}
    for name, grid in grids.items():
        mean = grid.mean(axis=1)
        center = grid[:, 4]
        measured_bias = mean - center
        reference_bias = measured_bias[zero[0]].copy()
        gate = reflection_gate(gaze[cal], measured_bias[cal])
        if not gate["passed"]:
            raise RuntimeError(f"{name} calibration-only BDE symmetry gate failed: {gate}")
        bias, bias_geometry = P1CentroidBias.fit(gaze[cal], measured_bias[cal], reference_bias, 3)
        bias_cv = bias_calibration_cv(gaze[cal], measured_bias[cal], reference_bias)
        gaze_model, gaze_geometry = fit_joint_cubic(center[cal], gaze[cal])
        gaze_cv = calibration_cv(center[cal], gaze[cal])
        if not bias_geometry["full_rank"] or not gaze_geometry["full_rank"]:
            raise RuntimeError(f"{name} B3/G20 calibration is not full rank")
        bias_models[name], gaze_models[name] = bias, gaze_model
        channel_summaries[name] = {
            "bias": {"form": "anchored B3 with 9 fitted scalars", "fit": bias_geometry,
                     "CV": bias_cv, "reflection_gate": gate,
                     "measured_zero_bias_mm": reference_bias.tolist()},
            "gaze": {"form": "full two-input degree-3 polynomial, 10 monomials x 2 outputs",
                     "fit": gaze_geometry, "CV": gaze_cv,
                     "absolute_center_calibration_range_mm": np.stack((center[cal].min(axis=0),
                                                                          center[cal].max(axis=0))).tolist()},
            "fitted_scalar_values": 29}
    old_b3_path = Path(p1_manifest["P1_candidates"]["B3"]["path"])
    if digest(old_b3_path) != p1_manifest["P1_candidates"]["B3"]["sha256"]:
        raise RuntimeError("Historical independently fitted P1 B3 changed")
    old_b3 = P1CentroidBias.load(old_b3_path)
    p1_refit_delta = float(max(np.max(np.abs(bias_models["P1"].coefficients_x_mm - old_b3.coefficients_x_mm)),
                               np.max(np.abs(bias_models["P1"].coefficients_y_mm - old_b3.coefficients_y_mm)),
                               np.max(np.abs(bias_models["P1"].reference_bias_mm - old_b3.reference_bias_mm))))
    if p1_refit_delta > 1e-13:
        raise RuntimeError("P1 B3 independent refit differs from prior 25-pose model")

    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(f"Fresh calibration directory required: {out}")
    out.mkdir(parents=True)
    for name in ("P1", "P4"):
        bias_path = out / f"{name.lower()}_bias_B3.npz"
        gaze_path = out / f"{name.lower()}_absolute_center_G20.npz"
        bias_models[name].save(bias_path)
        gaze_models[name].save(gaze_path)
        channel_summaries[name]["bias"].update(path=str(bias_path), sha256=digest(bias_path))
        channel_summaries[name]["gaze"].update(path=str(gaze_path), sha256=digest(gaze_path))
    np.savez_compressed(out / "calibration_data.npz", calibration_gaze_deg=gaze[cal],
                        calibration_p4_source_indices=ids[cal], calibration_source_subset_rows=source_rows[cal],
                        p1_absolute_center_mm=p1_grid[cal, 4], p4_absolute_center_mm=p4_grid[cal, 4],
                        p1_measured_bias_mm=p1_grid[cal].mean(axis=1)-p1_grid[cal, 4],
                        p4_measured_bias_mm=p4_grid[cal].mean(axis=1)-p4_grid[cal, 4],
                        holdout_p4_source_indices=ids[holdout])
    summary = {"experiment": "Independent A=0 P1 and P4 gaze channels with equal model forms",
        "A0_only": True, "calibration_poses_per_channel": 25,
        "primary_holdout": 3943, "inclusive_diagnostic_holdout": 3944,
        "excluded_gaze_deg": [12., 9.6], "excluded_source_id": 7554,
        "P1_and_P4_coefficients_and_normalizers_independent": True,
        "no_pairwise_gaze_map_or_automatic_fusion": True,
        "total_fitted_scalar_values": 58, "fitted_scalar_values_per_channel": 29,
        "measured_zero_bias_anchors_counted_separately": True,
        "channels": channel_summaries,
        "P1_old_B3_refit_max_abs_delta_mm": p1_refit_delta,
        "P1_center_trace_vs_labeled_ray_max_mm": p1_trace_delta,
        "P1_raw_sha256": p1_manifest["P1_source"]["sha256"],
        "P1_paired_data_sha256": digest(paired_path),
        "A0_observations_sha256": digest(obs_path),
        "raw_paired_source_sha256": digest(source_path),
        "calibration_data_sha256": digest(out / "calibration_data.npz"),
        "trainer_sha256": digest(__file__), "preflight_checks": checks,
        "frame_assumption": "Each channel stays in its recorded absolute coordinates; no fitted registration"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "total_fitted_scalar_values": 58,
                      "G20_CV_radial_RMS_deg": {name: channel_summaries[name]["gaze"]["CV"]["pooled_radial_RMS_deg"]
                                                   for name in ("P1", "P4")},
                      "B3_CV_vector_RMS_um": {name: channel_summaries[name]["bias"]["CV"]["vector_RMS_um"]
                                                for name in ("P1", "P4")}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
