"""Fit the requested full2D order2/order3 gaze maps on the same25 A0 poses."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from p4_a0_bivariate_polynomial import BivariatePolynomial, self_check

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"data/p4_p1_a0_only/calibration"
MODEL_SOURCE_SHA = "5a98a8a4e4727c1cef47de99bd65454afe71d2d198ba6fa2bb28bb4e1a6f9dc7"
C8_SHA = "92d8c5426ef216a7112d72fffb6e5e0b607cfd5c5e37d5b1e127d61ffd8ad929"
SHAPE_PINS = {"PF12": "b4d5aa1e6a49bd10ee80fa053feb6b73eb714fa1d5437a4f4f175c11079f5fb2",
              "PF33": "c13216dcee046e56a6fd19e3d521feb75ac6310b98eb1a5b0fcc392d799be3b9"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def calibration_cv(q, gaze, degree):
    folds = []
    for test in np.flatnonzero(np.any(gaze != 0, axis=1)):
        train = np.arange(len(q)) != test
        fitted, info = BivariatePolynomial.fit(q[train], gaze[train], degree)
        prediction = fitted.predict(q[test:test+1])[0]
        loss = float(np.mean((prediction-gaze[test])**2))
        folds.append({"test_calibration_row": int(test), "test_gaze_deg": gaze[test].tolist(),
                      "prediction_gaze_deg": prediction.tolist(), "coordinate_MSE_deg2": loss,
                      "full_rank": info["full_rank"], "geometry": info})
    losses = np.array([fold["coordinate_MSE_deg2"] for fold in folds])
    good = bool(all(fold["full_rank"] for fold in folds) and np.isfinite(losses).all())
    return {"folds": folds, "all_folds_full_rank": good,
            "coordinate_MSE_deg2": float(losses.mean()),
            "SE_coordinate_MSE_deg2": float(losses.std(ddof=1)/np.sqrt(len(losses))),
            "pooled_radial_RMS_deg": float(np.sqrt(2*losses.mean()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"data/p4_p1_a0_bivariate/calibration")
    args = parser.parse_args()
    checks = self_check()
    print(json.dumps({"polynomial_checks": checks}), flush=True)
    if args.check_only:
        return
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(f"Fresh output required:{out}")
    base = json.loads((BASE/"summary.json").read_text(encoding="utf-8"))
    if digest(ROOT/"Script/p4_a0_only_models.py") != MODEL_SOURCE_SHA or base["model_source_sha256"] != MODEL_SOURCE_SHA:
        raise RuntimeError("Frozen independent A0 model source changed")
    obs_path = BASE/"a0_observations.npz"
    if digest(obs_path) != base["observation_artifact_sha256"] or base["A0_rows"] != 3969:
        raise RuntimeError("Pinned A0-only observations changed")
    with np.load(obs_path) as obs:
        calibration = obs["calibration25"].astype(bool)
        gaze = obs["gaze_deg"][calibration].copy()
        q = (obs["absolute_p4_mm"][calibration, 4]-obs["mean_p1_absolute_mm"][calibration]).copy()
        source_ids = obs["source_indices"][calibration].copy()
        excluded = np.all(np.isclose(obs["gaze_deg"], [12., 9.6], atol=1e-9, rtol=0), axis=1)
        if calibration.sum() != 25 or excluded.sum() != 1 or np.any(excluded & calibration):
            raise AssertionError("Expected25cal poses andone exactexcluded noncal state")
        if not np.array_equal(obs["source_indices"][excluded], [7554]):
            raise AssertionError("User-excluded source identity differs")
    info25 = base["cases"]["25"]
    C8path = Path(info25["center_C8"]["path"])
    if digest(C8path) != C8_SHA:
        raise RuntimeError("Frozen separate-axis C8 baseline changed")
    comparisons, fitted = {}, {}
    for degree, name in ((2, "Q12"), (3, "C20")):
        model, geometry = BivariatePolynomial.fit(q, gaze, degree)
        cv = calibration_cv(q, gaze, degree)
        cv.update(degree=degree, coefficient_count=model.coefficient_count,
                  full_geometry=geometry, eligible=bool(geometry["full_rank"] and cv["all_folds_full_rank"]))
        comparisons[name], fitted[name] = cv, model
    # The same explicit calibration fold labels are checked against C8/C10;
    # those are historical diagnostics, not extra candidates in degree choice.
    baseline_folds = info25["center_C8"]["folds"]
    for reference, current in zip(baseline_folds, comparisons["C20"]["folds"]):
        if reference["test_gaze_deg"] != current["test_gaze_deg"]:
            raise AssertionError("Requested maps andC8 do not share calibration folds")
    if len(baseline_folds) != 24:
        raise AssertionError("Expected24 nonzero-gaze CV folds")
    eligible = [name for name, info in comparisons.items() if info["eligible"]]
    if not eligible:
        raise RuntimeError("Neither requested full2D degree is full rank/finite")
    best = min(eligible, key=lambda name: comparisons[name]["coordinate_MSE_deg2"])
    threshold = comparisons[best]["coordinate_MSE_deg2"]+comparisons[best]["SE_coordinate_MSE_deg2"]
    within = [name for name in eligible if comparisons[name]["coordinate_MSE_deg2"] <= threshold]
    selected = min(within, key=lambda name: comparisons[name]["coefficient_count"])
    selection = {"selected": selected, "best_CV": best, "one_SE_threshold_deg2": threshold,
                 "within_one_SE": within, "candidate_set": ["Q12", "C20"],
                 "rule": "Smaller full2D degree within oneSE of best24-fold calibration MSE",
                 "both_requested_degrees_reported_regardless_selection": True,
                 "heldout_used_for_selection": False}
    shapes = {}
    for name, expected in SHAPE_PINS.items():
        path = Path(info25["shape_candidates"][name]["path"])
        if digest(path) != expected:
            raise RuntimeError(f"Frozen A0-only shape changed:{name}")
        shapes[name] = {"path": str(path), "sha256": expected}
    optional_C10 = None
    optional_path = ROOT/"data/p4_p1_a0_cross_cubic/calibration/summary.json"
    if optional_path.exists():
        optional = json.loads(optional_path.read_text(encoding="utf-8"))
        if optional["A0_observation_sha256"] == digest(obs_path):
            optional_C10 = {"summary_path": str(optional_path), "summary_sha256": digest(optional_path),
                            "CV": optional["comparisons"]["C10"], "role": "separate symmetry-reduced historical comparison"}
    out.mkdir(parents=True)
    for name, model in fitted.items():
        path = out/f"map_{name}.npz"
        model.save(path)
        comparisons[name].update(path=str(path), sha256=digest(path),
                                 coefficients_ADE_BDE=model.coefficients.tolist())
    np.savez_compressed(out/"calibration_data.npz", measured_center_signal_mm=q,
                        truth_gaze_deg=gaze, source_indices=source_ids,
                        predictions_Q12=fitted["Q12"].predict(q), predictions_C20=fitted["C20"].predict(q))
    summary = {"experiment": "Requested full2D polynomial orders2 and3, calibrated on25 A0 poses",
               "selection": selection, "maps": comparisons, "shapes": shapes,
               "C8": {"path": str(C8path), "sha256": C8_SHA,
                      "calibration_CV": info25["center_C8"], "role": "frozen separate-axis cubic baseline"},
               "optional_C10": optional_C10, "A0_observation_path": str(obs_path),
               "A0_observation_sha256": digest(obs_path),
               "base_calibration_summary_sha256": digest(BASE/"summary.json"),
               "model_source_sha256": digest(ROOT/"Script/p4_a0_bivariate_polynomial.py"),
               "trainer_source_sha256": digest(__file__), "checks": checks,
               "primary_test_count": 3943, "unfiltered_diagnostic_test_count": 3944,
               "excluded_by_user": {"A_D": 0, "ADE_deg": 12, "BDE_deg": 9.6, "source_id": 7554},
               "frame_assumption": base["frame_assumption"]}
    (out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "selection": selection,
        "CV": {name: {key: info[key] for key in ("coordinate_MSE_deg2", "SE_coordinate_MSE_deg2", "pooled_radial_RMS_deg", "eligible")}
               for name, info in comparisons.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
