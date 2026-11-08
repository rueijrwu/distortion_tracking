"""Calibrate full cubic gaze coefficients versus accommodation on75 known poses."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from p4_accommodation_bivariate_cubic import AccommodationCubic, self_check

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/"data/p4_p1_limited_gaze_12p5/source_subset.npz"
SOURCE_SHA = "97c6db1b455f9abf7354351fa768cca7eea67198d8a4bfd14bbe684475462daf"
SHAPES = {
    "F27": (ROOT/"data/p4_p1_limited_gaze_12p5/models/new75_DF27.npz",
             "a590fb36e4561bb7f23af016d49330dabc2b038e4b7dee41288fe930445d52ab"),
    "F69": (ROOT/"data/p4_p1_limited_gaze_12p5/models/new75_PF69.npz",
             "ce1e9710c348e2874fa5d82060ee8d85beed0cf17cc8fe609bf92c0db44a05b4"),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def grouped_cv(q, gaze, A, order):
    folds = []
    for location in np.unique(gaze, axis=0):
        if np.all(location == 0):
            continue
        test = np.all(gaze == location, axis=1)
        if test.sum() != 3 or not np.array_equal(np.unique(A[test]), [0., 2., 4.]):
            raise AssertionError("Each gaze fold mustholdallthree known accommodation levels")
        model, info = AccommodationCubic.fit(q[~test], gaze[~test], A[~test], order)
        prediction = model.predict(q[test], A[test])
        error = prediction-gaze[test]
        loss = float(np.mean(error*error))
        folds.append({"test_gaze_deg": location.tolist(), "test_A_D": A[test].tolist(),
            "train_count": int((~test).sum()), "test_count": int(test.sum()),
            "coordinate_MSE_deg2": loss, "predicted_gaze_deg": prediction.tolist(),
            "full_rank": info["full_rank"], "geometry": info})
    losses = np.array([fold["coordinate_MSE_deg2"] for fold in folds])
    if len(folds) != 24:
        raise AssertionError("Expected24 nonzero-gaze-location folds")
    return {"folds": folds, "all_folds_full_rank": bool(all(f["full_rank"] for f in folds)),
            "all_finite": bool(np.isfinite(losses).all()),
            "coordinate_MSE_deg2": float(losses.mean()),
            "SE_coordinate_MSE_deg2": float(losses.std(ddof=1)/np.sqrt(len(losses))),
            "pooled_radial_RMS_deg": float(np.sqrt(2*losses.mean()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"data/p4_p1_accommodation_bivariate/calibration")
    args = parser.parse_args()
    checks = self_check()
    print(json.dumps({"map_checks": checks}), flush=True)
    if args.check_only:
        return
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(f"Fresh output required:{out}")
    if digest(SOURCE) != SOURCE_SHA:
        raise RuntimeError("Pinned source observations changed")
    with np.load(SOURCE) as source:
        states = source["states"]
        stencil = np.array([-12.4, -6.4, 0., 6.4, 12.4])
        on_gaze_stencil = np.all(np.any(np.isclose(states[:, 1:, None], stencil, atol=1e-9, rtol=0), axis=2), axis=1)
        on_A_stencil = np.any(np.isclose(states[:, 0, None], [0., 2., 4.], atol=1e-12, rtol=0), axis=1)
        calibration = on_gaze_stencil & on_A_stencil
        calibration_rows = np.flatnonzero(calibration)
        cal_states = states[calibration].copy()
        q = (source["absolute_p4_mm"][calibration, 4]-source["mean_p1_absolute_mm"][calibration]).copy()
        common = source["common_holdout"].astype(bool)
        excluded = np.all(np.isclose(states[:, 1:], [12., 9.6], atol=1e-9, rtol=0), axis=1)
        source_ids = source["source_indices"].copy()
        if len(states) != 83349 or calibration.sum() != 75 or common.sum() != 83250:
            raise AssertionError("Expected83,349 states,75cal and83,250 common holdout")
        if excluded.sum() != 21 or np.any(calibration & excluded) or np.any(calibration & common):
            raise AssertionError("User-excluded P1 gaze mustremain21 noncal heldout rows")
        if not np.all(common[excluded]) or (common & ~excluded).sum() != 83229:
            raise AssertionError("Expected83,229 primary and83,250 diagnostic rows")
        if not np.array_equal(np.unique(states[excluded, 0]), np.linspace(0, 4, 21)):
            raise AssertionError("Excluded P1 gaze is notrepeated atall21 accommodation labels")
        zero_rows = np.flatnonzero(np.all(states == [0., 0., 0.], axis=1))
        if len(zero_rows) != 1 or not calibration[zero_rows[0]]:
            raise AssertionError("Raw A0 zero-gaze reference is not a calibration anchor")
    if not np.array_equal(np.unique(cal_states[:, 0]), [0., 2., 4.]):
        raise AssertionError("Map fitting mustuseonly known0/2/4D labels")
    shape_metadata = {}
    for name, (path, expected) in SHAPES.items():
        if digest(path) != expected:
            raise RuntimeError(f"Frozen75-pose shape changed:{name}")
        shape_metadata[name] = {"path": str(path), "sha256": expected,
                                "coefficient_count": int(name[1:])}
    comparisons, fitted = {}, {}
    for order in (0, 1, 2):
        name = f"C{20*(order+1)}"
        model, info = AccommodationCubic.fit(q, cal_states[:, 1:], cal_states[:, 0], order)
        cv = grouped_cv(q, cal_states[:, 1:], cal_states[:, 0], order)
        cv.update(accommodation_order=order, coefficient_count=model.coefficient_count,
                  full_geometry=info, eligible=bool(info["full_rank"] and cv["all_folds_full_rank"] and cv["all_finite"]))
        comparisons[name], fitted[name] = cv, model
    eligible = [name for name, info in comparisons.items() if info["eligible"]]
    if not eligible:
        raise RuntimeError("No full-rank finite accommodation-dependent cubic map")
    best = min(eligible, key=lambda name: comparisons[name]["coordinate_MSE_deg2"])
    threshold = comparisons[best]["coordinate_MSE_deg2"]+comparisons[best]["SE_coordinate_MSE_deg2"]
    within = [name for name in eligible if comparisons[name]["coordinate_MSE_deg2"] <= threshold]
    selected = min(within, key=lambda name: comparisons[name]["coefficient_count"])
    selection = {"selected": selected, "best_CV": best, "one_SE_threshold_deg2": threshold,
                 "within_one_SE": within, "rule": "Fewest gaze coefficients within oneSE of best24 grouped calibration-fold MSE",
                 "heldout_used_for_selection": False}
    out.mkdir(parents=True)
    for name, model in fitted.items():
        path = out/f"map_{name}.npz"
        model.save(path)
        comparisons[name].update(path=str(path), sha256=digest(path), coefficients_ADE_BDE=model.coefficients.tolist())
    np.savez_compressed(out/"calibration_and_scope.npz", calibration_source_subset_rows=calibration_rows,
        calibration_source_indices=source_ids[calibration], truth_calibration_states=cal_states,
        measured_center_signal_mm=q, common_source_subset_rows=np.flatnonzero(common),
        common_source_indices=source_ids[common], primary_within_common=(~excluded)[common],
        excluded_source_subset_rows=np.flatnonzero(excluded), excluded_source_indices=source_ids[excluded],
        reference_source_index=source_ids[zero_rows[0]])
    summary = {"experiment": "Full2D cubic gaze with coefficients varying with estimated accommodation",
        "selection": selection, "maps": comparisons, "shapes": shape_metadata,
        "calibration_poses": 75, "calibration_A_D": [0, 2, 4], "gaze_stencil_deg": [-12.4, -6.4, 0, 6.4, 12.4],
        "source_path": str(SOURCE), "source_sha256": SOURCE_SHA,
        "scope_artifact_sha256": digest(out/"calibration_and_scope.npz"),
        "spatial_degree": 3, "accommodation_basis": "L0=1;L1=2*(A/4)-1;L2=(3*L1^2-1)/2",
        "feature_order": "Legendre blocks, then1,x,y,x^2,xy,y^2,x^3,x^2y,xy^2,y^3",
        "LOAO_limitation": "No leave-one-accommodation-level validation claimed: order2 requiresallthree0/2/4D levels; two levels cannotidentify it",
        "common_diagnostic_count": 83250, "primary_count": 83229, "excluded_count": 21,
        "excluded_P1_gaze_deg": [12, 9.6], "exclusion_reason": "User-excluded A-independent P1 gaze isreused atall21 A labels; allrows preserveddiagnostically",
        "reference_source_index": int(source_ids[zero_rows[0]]),
        "map_source_sha256": digest(ROOT/"Script/p4_accommodation_bivariate_cubic.py"),
        "trainer_source_sha256": digest(__file__), "checks": checks,
        "frame_assumption": "Pinned P1/P4 nominal SI coordinates; external physical registration notindependently proved",
        "noise_scope": "Noiseless calibration/geometry diagnostics; no instrument-noise accuracy guarantee"}
    (out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "selection": selection,
        "CV": {name: {key: info[key] for key in ("coordinate_MSE_deg2", "SE_coordinate_MSE_deg2", "pooled_radial_RMS_deg", "eligible")}
               for name, info in comparisons.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
