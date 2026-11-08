"""Calibrate predeclared C10 and compare calibration-only CV with frozen C8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from p4_a0_cross_axis_cubic import CrossAxisCubic, reflection_check, self_check
from p4_a0_only_models import CenterCubic

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"data/p4_p1_a0_only/calibration"
C8_SHA = "92d8c5426ef216a7112d72fffb6e5e0b607cfd5c5e37d5b1e127d61ffd8ad929"
MODEL_SOURCE_SHA = "5a98a8a4e4727c1cef47de99bd65454afe71d2d198ba6fa2bb28bb4e1a6f9dc7"
SHAPE_PINS = {"PF12": "b4d5aa1e6a49bd10ee80fa053feb6b73eb714fa1d5437a4f4f175c11079f5fb2",
              "PF33": "c13216dcee046e56a6fd19e3d521feb75ac6310b98eb1a5b0fcc392d799be3b9"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def full_rank_c8(info):
    return all(axis["standardized"]["rank"] == 4 for axis in info.values())


def cv(q, gaze, kind):
    folds = []
    for test in np.flatnonzero(np.any(gaze != 0, axis=1)):
        train = np.arange(len(q)) != test
        if kind == "C8":
            fitted, info = CenterCubic.fit(q[train], gaze[train])
            rank_ok = full_rank_c8(info)
        else:
            fitted, info = CrossAxisCubic.fit(q[train], gaze[train])
            rank_ok = info["full_rank"]
        prediction = fitted.predict(q[test:test+1])[0]
        loss = float(np.mean((prediction-gaze[test])**2))
        folds.append({"test_calibration_row": int(test), "test_gaze_deg": gaze[test].tolist(),
                      "prediction_gaze_deg": prediction.tolist(), "coordinate_MSE_deg2": loss,
                      "full_rank": bool(rank_ok), "geometry": info})
    losses = np.array([row["coordinate_MSE_deg2"] for row in folds])
    good = bool(all(row["full_rank"] for row in folds) and np.isfinite(losses).all())
    return {"folds": folds, "all_folds_full_rank": good,
            "coordinate_MSE_deg2": float(losses.mean()),
            "SE_coordinate_MSE_deg2": float(losses.std(ddof=1)/np.sqrt(len(losses))),
            "pooled_radial_RMS_deg": float(np.sqrt(2*losses.mean()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"data/p4_p1_a0_cross_cubic/calibration")
    args = parser.parse_args()
    checks = self_check()
    print(json.dumps({"C10_checks": checks}), flush=True)
    if args.check_only:
        return
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(f"Fresh output required: {out}")
    base = json.loads((BASE/"summary.json").read_text(encoding="utf-8"))
    if digest(ROOT/"Script/p4_a0_only_models.py") != MODEL_SOURCE_SHA or base["model_source_sha256"] != MODEL_SOURCE_SHA:
        raise RuntimeError("Frozen A0-only model source changed")
    obs_path = BASE/"a0_observations.npz"
    if digest(obs_path) != base["observation_artifact_sha256"]:
        raise RuntimeError("A0-only observations changed")
    with np.load(obs_path) as obs:
        mask = obs["calibration25"].astype(bool)
        gaze = obs["gaze_deg"][mask].copy()
        q = (obs["absolute_p4_mm"][mask, 4]-obs["mean_p1_absolute_mm"][mask]).copy()
        cal_source_ids = obs["source_indices"][mask].copy()
        exclusion = np.all(np.isclose(obs["gaze_deg"], [12., 9.6], atol=1e-9, rtol=0), axis=1)
        if mask.sum() != 25 or exclusion.sum() != 1 or np.any(exclusion & mask):
            raise AssertionError("Expected25cal points and exactnoncal user-excludedstate")
        if not np.array_equal(obs["source_indices"][exclusion], [7554]):
            raise AssertionError("User-excluded physical state/source identity differs")
    symmetry = reflection_check(q, gaze)
    if not symmetry["passed"]:
        raise RuntimeError(f"C10 calibration reflection gate failed: {symmetry}")
    info25 = base["cases"]["25"]
    C8path = Path(info25["center_C8"]["path"])
    if digest(C8path) != C8_SHA or info25["center_C8"]["sha256"] != C8_SHA:
        raise RuntimeError("Frozen baseline C8 changed")
    baseline = CenterCubic.load(C8path)
    recomputed, baseline_geometry = CenterCubic.fit(q, gaze)
    coefficient_delta = float(np.max(np.abs(recomputed.coefficients_xy-baseline.coefficients_xy)))
    prediction_delta = float(np.max(np.abs(recomputed.predict(q)-baseline.predict(q))))
    normalizer_delta = float(max(np.max(np.abs(recomputed.mean_mm-baseline.mean_mm)),
                                 np.max(np.abs(recomputed.scale_mm-baseline.scale_mm))))
    if max(coefficient_delta, prediction_delta, normalizer_delta) > 1e-11:
        raise RuntimeError("Recomputed calibration-only C8 differs from frozen baseline")
    fitted, full_geometry = CrossAxisCubic.fit(q, gaze)
    comparisons = {"C8": cv(q, gaze, "C8"), "C10": cv(q, gaze, "C10")}
    old_folds = info25["center_C8"]["folds"]
    if len(old_folds) != 24:
        raise AssertionError("Frozen C8 does not have24calibration folds")
    old_delta = 0.0
    for original, current in zip(old_folds, comparisons["C8"]["folds"]):
        if original["test_gaze_deg"] != current["test_gaze_deg"]:
            raise AssertionError("C8/C10 calibration fold labels differ")
        old_delta = max(old_delta, abs(original["coordinate_MSE_deg2"]-current["coordinate_MSE_deg2"]))
    if old_delta > 1e-11:
        raise RuntimeError("C8 fold scores differ from frozen A0-only baseline")
    comparisons["C8"].update(coefficient_count=8,
        eligible=bool(full_rank_c8(baseline_geometry) and comparisons["C8"]["all_folds_full_rank"]))
    comparisons["C10"].update(coefficient_count=10,
        eligible=bool(full_geometry["full_rank"] and comparisons["C10"]["all_folds_full_rank"]))
    eligible = [key for key, value in comparisons.items() if value["eligible"]]
    if not eligible:
        raise RuntimeError("No full-rank finite map candidate")
    best = min(eligible, key=lambda key: comparisons[key]["coordinate_MSE_deg2"])
    threshold = comparisons[best]["coordinate_MSE_deg2"]+comparisons[best]["SE_coordinate_MSE_deg2"]
    within = [key for key in eligible if comparisons[key]["coordinate_MSE_deg2"] <= threshold]
    selected = min(within, key=lambda key: comparisons[key]["coefficient_count"])
    selection = {"selected": selected, "best_CV": best, "one_SE_threshold_deg2": threshold,
                 "within_one_SE": within, "rule": "Fewest coefficients within oneSE of best24-fold calibration-only coordinate MSE",
                 "test_used_for_selection": False}
    shapes = {}
    for name, expected in SHAPE_PINS.items():
        path = Path(info25["shape_candidates"][name]["path"])
        if digest(path) != expected:
            raise RuntimeError(f"Frozen A0-only shape changed:{name}")
        shapes[name] = {"path": str(path), "sha256": expected}
    out.mkdir(parents=True)
    model_path = out/"center_C10.npz"
    fitted.save(model_path)
    np.savez_compressed(out/"calibration_data.npz", measured_center_signal_mm=q, truth_gaze_deg=gaze,
                        source_indices=cal_source_ids, predictions_C8=baseline.predict(q), predictions_C10=fitted.predict(q))
    summary = {"experiment": "Predeclared cross-axis symmetry cubics; A0-only25 calibration observations",
               "reflection_gate": symmetry, "selection": selection, "comparisons": comparisons,
               "full_C10_geometry": full_geometry, "C10_coefficients": {
                   "BDE": fitted.bde_coefficients.tolist(), "ADE": fitted.ade_coefficients.tolist()},
               "C10": {"path": str(model_path), "sha256": digest(model_path)},
               "C8": {"path": str(C8path), "sha256": C8_SHA}, "shapes": shapes,
               "baseline_reconstruction": {"max_coefficient_delta": coefficient_delta,
                   "max_calibration_prediction_delta_deg": prediction_delta, "max_normalizer_delta_mm": normalizer_delta,
                   "max_original_fold_MSE_delta_deg2": old_delta},
               "A0_observation_path": str(obs_path), "A0_observation_sha256": digest(obs_path),
               "base_calibration_summary_sha256": digest(BASE/"summary.json"),
               "model_source_sha256": digest(ROOT/"Script/p4_a0_cross_axis_cubic.py"),
               "trainer_source_sha256": digest(__file__), "checks": checks,
               "primary_test_count": 3943, "unfiltered_diagnostic_test_count": 3944,
               "excluded_by_user": {"A_D": 0, "ADE_deg": 12, "BDE_deg": 9.6, "source_id": 7554},
               "frame_assumption": base["frame_assumption"]}
    (out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "reflection": symmetry, "selection": selection,
                      "CV": {key: {metric: value[metric] for metric in ("coordinate_MSE_deg2", "SE_coordinate_MSE_deg2", "pooled_radial_RMS_deg", "eligible")}
                             for key, value in comparisons.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
