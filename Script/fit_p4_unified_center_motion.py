"""Fit and select measured-pivot P4 A-normalization from 75 known-A poses."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from p4_unified_center_motion import fit_motion, self_check

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/p4_p1_limited_gaze_12p5/source_subset.npz"
A0_SUMMARY = ROOT / "data/p4_p1_a0_only/calibration/summary.json"
OUT = ROOT / "data/p4_unified_combined/calibration/center_motion"
STENCIL = np.array([-12.4, -6.4, 0., 6.4, 12.4])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def calibration_cv(accommodation, gaze, observed, reference, pivot, order):
    folds = []
    for location in np.unique(gaze, axis=0):
        if np.all(location == 0):
            continue
        test = np.all(np.isclose(gaze, location, atol=1e-9, rtol=0), axis=1)
        if test.sum() != 3 or not np.array_equal(np.sort(accommodation[test]), [0., 2., 4.]):
            raise AssertionError("Each heldout gaze fold must contain all three A observations")
        model, fit = fit_motion(accommodation[~test], observed[~test], reference[~test], pivot, order)
        normalized = model.inverse(observed[test], accommodation[test])
        error = normalized-reference[test]
        folds.append({"heldout_gaze_deg": location.tolist(), "heldout_A_d": accommodation[test].tolist(),
                      "coordinate_MSE_mm2": float(np.mean(error*error)),
                      "vector_RMS_mm": float(np.sqrt(np.mean(np.sum(error*error, axis=1)))),
                      "fit": fit})
    if len(folds) != 24 or not all(f["fit"]["full_rank"] and f["fit"]["positive_scales_0_to_4D"] for f in folds):
        raise RuntimeError("M1/M2 must have 24 full-rank positive-scale folds")
    loss = np.array([f["coordinate_MSE_mm2"] for f in folds])
    return {"folds": folds, "fold_count": 24, "all_folds_full_rank": True,
            "all_folds_positive_scales": True,
            "coordinate_MSE_mm2": float(loss.mean()),
            "SE_coordinate_MSE_mm2": float(loss.std(ddof=1)/np.sqrt(len(loss))),
            "normalized_center_vector_RMS_mm": float(np.sqrt(2*loss.mean()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    checks = self_check()
    if args.check_only:
        print(json.dumps({"checks": checks}), flush=True)
        return
    source_summary = json.loads(A0_SUMMARY.read_text(encoding="utf-8"))
    if digest(SOURCE) != source_summary["source_subset_sha256"]:
        raise RuntimeError("Pinned paired P4 source changed")
    with np.load(SOURCE) as raw:
        states = raw["states"].copy()
        ids = raw["source_indices"].copy()
        central = raw["absolute_p4_mm"][:, 4].copy()
    gaze_cal = np.all(np.any(np.isclose(states[:, 1:, None], STENCIL, atol=1e-9, rtol=0), axis=2), axis=1)
    a_cal = np.any(np.isclose(states[:, 0, None], [0., 2., 4.], atol=1e-12, rtol=0), axis=1)
    cal = gaze_cal & a_cal
    if cal.sum() != 75:
        raise AssertionError("Expected exact 25 gaze x 3 known-A calibration")
    state = states[cal]
    y4center = central[cal]
    source_ids = ids[cal]
    zero = np.all(state[:, 1:] == 0, axis=1)
    if (zero.sum() != 3 or not np.array_equal(np.sort(state[zero, 0]), [0., 2., 4.])
            or not np.array_equal(y4center[zero], np.zeros((3, 2)))):
        raise RuntimeError("Measured on-axis P4 center moved; no-translation model is invalid")
    pivot = y4center[zero & (state[:, 0] == 0)][0].copy()
    a0_by_gaze = {tuple(np.round(g, 8)): c for g, c in zip(state[state[:, 0] == 0, 1:],
                                                               y4center[state[:, 0] == 0])}
    if len(a0_by_gaze) != 25:
        raise AssertionError("A=0 calibration reference gaze set changed")
    reference = np.array([a0_by_gaze[tuple(np.round(g, 8))] for g in state[:, 1:]])
    if not np.array_equal(reference[state[:, 0] == 0], y4center[state[:, 0] == 0]):
        raise AssertionError("A=0 central reference was not matched by physical gaze label")
    candidates, models = {}, {}
    for order in (1, 2):
        name = f"M{order}"
        model, fit = fit_motion(state[:, 0], y4center, reference, pivot, order)
        if not fit["full_rank"] or not fit["positive_scales_0_to_4D"]:
            raise RuntimeError(f"{name} has rank-deficient or nonpositive central motion")
        cv = calibration_cv(state[:, 0], state[:, 1:], y4center, reference, pivot, order)
        train_error = model.inverse(y4center, state[:, 0])-reference
        candidates[name] = {"order": order, "coefficient_count": model.coefficient_count,
                            "fit": fit, "CV": cv,
                            "training_normalized_center_vector_RMS_mm": float(np.sqrt(np.mean(np.sum(train_error**2, axis=1)))),
                            "coefficients_axis_by_power": model.coefficients.tolist()}
        models[name] = model
    best = min(candidates, key=lambda name: candidates[name]["CV"]["coordinate_MSE_mm2"])
    threshold = (candidates[best]["CV"]["coordinate_MSE_mm2"]+
                 candidates[best]["CV"]["SE_coordinate_MSE_mm2"])
    within = [name for name in candidates if candidates[name]["CV"]["coordinate_MSE_mm2"] <= threshold]
    selected = min(within, key=lambda name: candidates[name]["coefficient_count"])
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(f"Fresh output directory required: {out}")
    out.mkdir(parents=True)
    for name, model in models.items():
        path = out / f"central_motion_{name}.npz"
        model.save(path)
        candidates[name].update(path=str(path), sha256=digest(path))
    np.savez_compressed(out / "calibration_data.npz", states=state, source_indices=source_ids,
                        true_p4_center_A_mm=y4center, matched_p4_center_A0_mm=reference,
                        pivot_mm=pivot)
    summary = {"experiment": "P4 central-motion normalization about measured A=0 on-axis center",
        "known_A_calibration_poses": 75, "unique_gaze_locations": 25,
        "A_calibration_D": [0., 2., 4.], "pivot_A0_onaxis_mm": pivot.tolist(),
        "measured_onaxis_centers_A0_A2_A4_mm": y4center[zero].tolist(),
        "onaxis_translation_measured_zero": True,
        "candidates": candidates,
        "selection": {"selected": selected, "best_CV": best,
                      "one_SE_threshold_coordinate_MSE_mm2": threshold,
                      "within_one_SE": within,
                      "rule": "Fewest positive-scale full-rank coefficients within one SE of best 24-fold normalized-center MSE"},
        "calibration_data_sha256": digest(out / "calibration_data.npz"),
        "source_sha256": digest(SOURCE), "trainer_sha256": digest(__file__),
        "model_source_sha256": digest(ROOT / "Script/p4_unified_center_motion.py"),
        "checks": checks, "no_G4_refit": True,
        "runtime_formula": "c_ref=o+(c_A-o)/s(A), s_i=1+sum(k_i,p*(A/4)^p)"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "selected": selected,
                      "candidates": {name: {"CV_normalized_vector_RMS_mm": value["CV"]["normalized_center_vector_RMS_mm"],
                                            "coefficients": value["coefficient_count"]}
                                     for name, value in candidates.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
