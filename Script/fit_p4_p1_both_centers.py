"""Calibrate P1 centroid correction and a new P4-center minus P1-center cubic."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from p1_centroid_bias import P1CentroidBias, reflection_gate, self_check
from p4_a0_bivariate_polynomial import BivariatePolynomial
from fit_p4_p1_full_accommodation_fixed_c20 import fit_joint_vector_cubic

ROOT = Path(__file__).resolve().parents[1]
P1 = ROOT/"data/distortion_grid_p1_biaxial_macro/distortion_grid.pkl"
P1_SHA = "1a57a8d850d021f3d23dda201f2fc47bd1aec6eb60860e42990be990cfe7f835"
SOURCE = ROOT/"data/p4_p1_limited_gaze_12p5/source_subset.npz"
SOURCE_SHA = "97c6db1b455f9abf7354351fa768cca7eea67198d8a4bfd14bbe684475462daf"
F27 = ROOT/"data/p4_p1_limited_gaze_12p5/models/new75_DF27.npz"
F27_SHA = "a590fb36e4561bb7f23af016d49330dabc2b038e4b7dee41288fe930445d52ab"
OLD_G = ROOT/"data/p4_p1_full_accommodation_fixed_c20/calibration/fixed_C20.npz"
OLD_G_SHA = "11fc9b3d07d27d7a080574a08bc3ff013a4c84becfe74edc4bdf959315044bb2"
STENCIL = np.array([-12.4, -6.4, 0., 6.4, 12.4])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vector_metric(error_mm):
    r = np.linalg.norm(np.asarray(error_mm)*1000., axis=1)
    return {"N": len(r), "vector_RMS_um": float(np.sqrt(np.mean(r*r))),
            "median_um": float(np.median(r)), "P95_um": float(np.percentile(r, 95)), "max_um": float(r.max())}


def load_p1():
    if digest(P1) != P1_SHA:
        raise RuntimeError("Pinned P1 raw grid changed")
    raw = pickle.loads(P1.read_bytes())
    order = np.lexsort((raw["field_y_relative"], raw["field_x_relative"],
                        raw["eye_rotation_x_deg"], raw["eye_rotation_deg"]))
    d = raw[order]
    ade, bde = np.unique(d["eye_rotation_deg"]), np.unique(d["eye_rotation_x_deg"])
    if len(d) != 101*101*9 or len(ade) != 101 or len(bde) != 101:
        raise AssertionError("Expected one101x101 P1 gaze grid, without accommodation")
    shape = (101, 101, 9)
    field = np.stack((d["field_x_relative"], d["field_y_relative"]), axis=-1).reshape(shape+(2,))
    expected = np.array([(x, y) for x in (-1., 0., 1.) for y in (-1., 0., 1.)])
    if not np.array_equal(field, np.broadcast_to(expected, field.shape)):
        raise AssertionError("P1 physical field identity/order differs")
    center = np.stack((d["center_absolute_x_mm"], d["center_absolute_y_mm"]), axis=-1).reshape(shape+(2,))
    relative = np.stack((d["real_x_mm"], d["real_y_mm"]), axis=-1).reshape(shape+(2,))
    absolute = center+relative
    if not np.isfinite(absolute).all():
        raise AssertionError("No changing-field averages: allnine real P1 rays mustbefinite")
    keepa, keepb = np.flatnonzero(np.abs(ade) <= 12.5), np.flatnonzero(np.abs(bde) <= 12.5)
    aa, bb = np.meshgrid(ade[keepa], bde[keepb], indexing="ij")
    gaze = np.stack((aa.ravel(), bb.ravel()), axis=1)
    grids = absolute[keepa[:, None], keepb[None, :]].reshape(-1, 9, 2)
    central_trace_delta = float(np.max(np.abs(absolute[..., 4, :]-center[..., 4, :])))
    return gaze, grids, central_trace_delta


def fit_bias_case(gaze, bias, reference_bias, degree):
    model, full = P1CentroidBias.fit(gaze, bias, reference_bias, degree)
    folds = []
    for test in np.flatnonzero(np.any(gaze != 0, axis=1)):
        train = np.arange(len(gaze)) != test
        fitted, info = P1CentroidBias.fit(gaze[train], bias[train], reference_bias, degree)
        error = (fitted.predict(gaze[test:test+1])[0]-bias[test])*1000.0
        folds.append({"test_gaze_deg": gaze[test].tolist(), "coordinate_MSE_um2": float(np.mean(error*error)),
                      "full_rank": info["full_rank"], "geometry": info})
    loss = np.array([fold["coordinate_MSE_um2"] for fold in folds])
    eligible = bool(full["full_rank"] and all(fold["full_rank"] for fold in folds) and np.isfinite(loss).all())
    return model, {"degree": degree, "coefficient_count": model.coefficient_count, "full_geometry": full,
        "folds": folds, "eligible": eligible, "CV_coordinate_MSE_um2": float(loss.mean()),
        "CV_SE_um2": float(loss.std(ddof=1)/np.sqrt(len(loss))), "CV_vector_RMS_um": float(np.sqrt(2*loss.mean()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"data/p4_p1_both_centers/calibration")
    args = parser.parse_args()
    checks = self_check()
    print(json.dumps({"bias_checks": checks}), flush=True)
    if args.check_only:
        return
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(f"Fresh output required:{out}")
    for path, expected in ((SOURCE, SOURCE_SHA), (F27, F27_SHA), (OLD_G, OLD_G_SHA)):
        if digest(path) != expected:
            raise RuntimeError(f"Frozen input changed:{path}")
    p1_gaze, p1_grids, trace_delta = load_p1()
    if len(p1_gaze) != 3969:
        raise AssertionError("Expected3969 P1 states inactual±12.4 support")
    p1_mean, p1_center = p1_grids.mean(axis=1), p1_grids[:, 4]
    p1_bias = p1_mean-p1_center
    p1_cal = np.all(np.any(np.isclose(p1_gaze[:, :, None], STENCIL, atol=1e-9, rtol=0), axis=2), axis=1)
    zero = np.flatnonzero(np.all(p1_gaze == 0, axis=1))
    excluded_p1 = np.all(np.isclose(p1_gaze, [12., 9.6], atol=1e-9, rtol=0), axis=1)
    if p1_cal.sum() != 25 or len(zero) != 1 or excluded_p1.sum() != 1 or np.any(p1_cal & excluded_p1):
        raise AssertionError("P1 calibration/reference/exclusion labels differ")
    reference_bias = p1_bias[zero[0]].copy()
    reference_grid = p1_grids[zero[0]]-p1_center[zero[0]]
    outer = np.array([0, 1, 2, 3, 5, 6, 7, 8])
    reference_RMS_mm = float(np.sqrt(np.mean(np.sum(reference_grid[outer]**2, axis=1))))
    gate = reflection_gate(p1_gaze[p1_cal], p1_bias[p1_cal])
    if not gate["passed"]:
        raise RuntimeError(f"P1 calibration-only BDE parity fails:{gate}")
    models, candidates = {}, {}
    for degree in (2, 3):
        name = f"B{degree}"
        models[name], candidates[name] = fit_bias_case(p1_gaze[p1_cal], p1_bias[p1_cal], reference_bias, degree)
    eligible = [name for name, info in candidates.items() if info["eligible"]]
    if not eligible:
        raise RuntimeError("No full-rank P1 centroid candidate")
    best = min(eligible, key=lambda name: candidates[name]["CV_coordinate_MSE_um2"])
    threshold = candidates[best]["CV_coordinate_MSE_um2"]+candidates[best]["CV_SE_um2"]
    within = [name for name in eligible if candidates[name]["CV_coordinate_MSE_um2"] <= threshold]
    selected = min(within, key=lambda name: candidates[name]["coefficient_count"])
    selection = {"selected": selected, "best_CV": best, "within_one_SE": within, "one_SE_threshold_um2": threshold,
                 "heldout_used_for_selection": False, "rule": "Fewest P1 coefficients within oneSE of best24 calibration folds"}
    with np.load(SOURCE) as source:
        states, ids = source["states"].copy(), source["source_indices"].copy()
        common = source["common_holdout"].astype(bool)
        cal75 = np.all(np.any(np.isclose(states[:, 1:, None], STENCIL, atol=1e-9, rtol=0), axis=2), axis=1)
        cal75 &= np.any(np.isclose(states[:, 0, None], [0., 2., 4.], atol=1e-12, rtol=0), axis=1)
        p4_cal_centers = source["absolute_p4_mm"][cal75, 4].copy()
        measured_p1_mean = source["mean_p1_absolute_mm"].copy()
    lookup = {tuple(np.round(g, 8)): i for i, g in enumerate(p1_gaze)}
    joined = np.array([lookup[tuple(np.round(g, 8))] for g in states[:, 1:]], int)
    mean_delta = float(np.max(np.abs(p1_mean[joined]-measured_p1_mean)))
    if mean_delta > 1e-10:
        raise AssertionError("Reconstructed P1 allnine mean differs frompaired source")
    excluded = excluded_p1[joined]
    if cal75.sum() != 75 or common.sum() != 83250 or (common & ~excluded).sum() != 83229 or excluded.sum() != 21:
        raise AssertionError("Expected75cal,83,250diagnostic,83,229primary and21excluded rows")
    if np.any(cal75 & excluded) or np.any(cal75 & common):
        raise AssertionError("Calibration/test/exclusion overlap")
    q = p4_cal_centers-p1_center[joined[cal75]]
    cal_states = states[cal75]
    G, geometry = fit_joint_vector_cubic(q, cal_states[:, 1:])
    Gfolds = []
    for location in np.unique(cal_states[:, 1:], axis=0):
        if np.all(location == 0):
            continue
        test = np.all(cal_states[:, 1:] == location, axis=1)
        fitted, info = fit_joint_vector_cubic(q[~test], cal_states[~test, 1:])
        loss = float(np.mean((fitted.predict(q[test])-cal_states[test, 1:])**2))
        Gfolds.append({"test_gaze_deg": location.tolist(), "test_A_D": cal_states[test, 0].tolist(),
                       "coordinate_MSE_deg2": loss, "full_rank": info["full_rank"], "geometry": info})
    if not geometry["full_rank"] or not all(fold["full_rank"] for fold in Gfolds):
        raise RuntimeError("Newcenter-to-center C20 failsfull/foldrank10")
    # Reproduce the fair old-domain baseline using the exactsame75 observations.
    old_check, _ = fit_joint_vector_cubic(p4_cal_centers-measured_p1_mean[cal75], cal_states[:, 1:])
    old = BivariatePolynomial.load(OLD_G)
    old_delta = float(np.max(np.abs(old_check.coefficients-old.coefficients)))
    if old_delta > 1e-10:
        raise AssertionError("Old P1-mean baseline cannotbereproduced onmatched75cal")
    out.mkdir(parents=True)
    for name, model in models.items():
        path = out/f"p1_{name}.npz"
        model.save(path)
        candidates[name].update(path=str(path), sha256=digest(path))
    Gpath = out/"center_to_center_C20.npz"
    G.save(Gpath)
    # Selection is frozen above; these are independent heldout P1 diagnostics.
    p1_test = ~p1_cal
    p1_primary = p1_test & ~excluded_p1
    if p1_test.sum() != 3944 or p1_primary.sum() != 3943:
        raise AssertionError("P1-only test counts differ")
    p1_scores = {"uncorrected_mean_as_center": vector_metric(p1_bias[p1_primary])}
    for name, model in models.items():
        error = p1_bias-model.predict(p1_gaze)
        p1_scores[name] = {"primary3943": vector_metric(error[p1_primary]),
                           "diagnostic3944": vector_metric(error[p1_test])}
    np.savez_compressed(out/"paired_data.npz", p1_gaze_deg=p1_gaze, p1_absolute_mm=p1_grids,
        p1_mean_mm=p1_mean, p1_labeled_center_mm=p1_center, p1_bias_mm=p1_bias,
        p1_calibration25=p1_cal, p1_primary3943=p1_primary, p1_diagnostic3944=p1_test,
        p1_reference_centered_grid_mm=reference_grid, source_p1_rows=joined,
        common_source_subset_rows=np.flatnonzero(common), common_source_indices=ids[common],
        primary_within_common=(~excluded)[common], calibration_source_rows=np.flatnonzero(cal75),
        calibration_source_indices=ids[cal75], calibration_q_center_to_center_mm=q,
        calibration_truth_states=cal_states)
    manifest = {"experiment": "Independent P1 gaze-only centroid correction plusnewcenter-to-center cubic",
        "P1_source": {"path": str(P1), "sha256": P1_SHA, "unique_calibration_gazes": 25},
        "paired_source": {"path": str(SOURCE), "sha256": SOURCE_SHA},
        "forward_P4": {"path": str(F27), "sha256": F27_SHA, "coefficient_count": 27},
        "P1_reference_bias_mm": reference_bias.tolist(), "reference_bias_is_measured_not_fitted": True,
        "P1_reference_outer_RMS_mm": reference_RMS_mm, "P1_zero_labeled_center_mm": p1_center[zero[0]].tolist(),
        "P1_center_trace_vs_labeled_ray_max_mm": trace_delta, "P1_reconstructed_mean_delta_mm": mean_delta,
        "P1_parity_gate": gate, "P1_candidates": candidates, "P1_selection": selection,
        "P1_heldout_scores": p1_scores,
        "gaze_C20": {"path": str(Gpath), "sha256": digest(Gpath), "vector_coefficients": 10,
            "scalar_coefficients": 20, "API": "predict(q_mm)", "input": "P4 labeledcenter minusP1 labeledcenter",
            "full_geometry": geometry, "CV_folds": Gfolds,
            "CV_radial_RMS_deg": float(np.sqrt(2*np.mean([f["coordinate_MSE_deg2"] for f in Gfolds])))},
        "baseline_before_P1_correction": {"map_path": str(OLD_G), "map_sha256": OLD_G_SHA,
            "input": "P4 center minusP1 mean", "matched_calibration_coefficient_delta": old_delta,
            "full_runtime_path": str(ROOT/"data/p4_p1_compact_full_accommodation/full/runtime_shared75.npz"),
            "full_runtime_sha256": "064f0e73e7df9688eb40fd417e0367e2c6abc89922102b4085af5f4393d9237e"},
        "paired_data_sha256": digest(out/"paired_data.npz"), "common_diagnostic_count": 83250,
        "primary_count": 83229, "excluded_P1_gaze_deg": [12, 9.6], "excluded_repeated_rows": 21,
        "selected_total_fitted_coefficients": 47+candidates[selected]["coefficient_count"],
        "reference_and_normalizer_constants_listed_separately": True,
        "bias_model_source_sha256": digest(ROOT/"Script/p1_centroid_bias.py"),
        "trainer_source_sha256": digest(__file__), "checks": checks,
        "frame_assumption": "User-specified mirrored common source; nominal shared SI coordinates; noextra registration fit or externalregistration claim"}
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "P1_selection": selection,
        "P1_zero_bias_mm": reference_bias.tolist(), "P1_heldout": p1_scores,
        "C20_CV_radial_RMS_deg": manifest["gaze_C20"]["CV_radial_RMS_deg"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
