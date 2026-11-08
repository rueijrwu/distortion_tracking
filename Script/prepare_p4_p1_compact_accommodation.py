"""Freeze compact full-A inputs and compare calibration strategies informationally."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from fit_p4_p1_full_accommodation_fixed_c20 import fit_joint_vector_cubic
from p4_a0_bivariate_polynomial import BivariatePolynomial

ROOT = Path(__file__).resolve().parents[1]
A0_CAL = ROOT/"data/p4_p1_a0_bivariate/calibration"
FULL_CAL = ROOT/"data/p4_p1_full_accommodation_fixed_c20/calibration"
SOURCE_SHA = "97c6db1b455f9abf7354351fa768cca7eea67198d8a4bfd14bbe684475462daf"
FORWARD_SHA = "a590fb36e4561bb7f23af016d49330dabc2b038e4b7dee41288fe930445d52ab"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_folds(q, states):
    folds = []
    for gaze in np.unique(states[:, 1:], axis=0):
        if np.all(gaze == 0):
            continue
        test = np.all(states[:, 1:] == gaze, axis=1)
        rows = {}
        for variant in ("A0_frozen", "shared75"):
            train = ~test
            if variant == "A0_frozen":
                train &= states[:, 0] == 0
            model, geometry = fit_joint_vector_cubic(q[train], states[train, 1:])
            error = model.predict(q[test])-states[test, 1:]
            rows[variant] = {"train_count": int(train.sum()), "full_rank": geometry["full_rank"],
                "coordinate_MSE_deg2": float(np.mean(error*error)), "geometry": geometry}
        folds.append({"heldout_gaze_deg": gaze.tolist(), "heldout_A_D": states[test, 0].tolist(),
                      "methods": rows})
    summaries = {}
    for variant in ("A0_frozen", "shared75"):
        losses = np.array([fold["methods"][variant]["coordinate_MSE_deg2"] for fold in folds])
        summaries[variant] = {"mean_coordinate_MSE_deg2": float(losses.mean()),
            "SE_coordinate_MSE_deg2": float(losses.std(ddof=1)/np.sqrt(len(losses))),
            "pooled_radial_RMS_deg": float(np.sqrt(2*losses.mean())),
            "all_folds_full_rank": bool(all(fold["methods"][variant]["full_rank"] for fold in folds))}
    return {"folds": folds, "summary": summaries, "role": "Informational; primary remains exact A0 map regardlessscores"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"data/p4_p1_compact_full_accommodation/calibration")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(f"Fresh output required:{out}")
    pure = json.loads((FULL_CAL/"summary.json").read_text(encoding="utf-8"))
    a0 = json.loads((A0_CAL/"summary.json").read_text(encoding="utf-8"))
    source_path = Path(pure["source_path"])
    if digest(source_path) != SOURCE_SHA or pure["source_sha256"] != SOURCE_SHA:
        raise RuntimeError("Pinned paired source changed")
    forward = pure["forward_model"]
    if forward["coefficient_count"] != 27 or digest(forward["path"]) != FORWARD_SHA:
        raise RuntimeError("Compact75-pose forward artifact changed")
    if digest(FULL_CAL/"calibration_and_scope.npz") != pure["scope_sha256"]:
        raise RuntimeError("Verified75-pose source scope changed")
    map_info = {
        "A0_frozen": {"path": a0["maps"]["C20"]["path"], "sha256": a0["maps"]["C20"]["sha256"],
                       "role": "primary; exact accepted A0 gaze coefficients preserved", "calibration_poses": 25},
        "shared75": {"path": pure["gaze_model"]["path"], "sha256": pure["gaze_model"]["sha256"],
                      "role": "predeclared75-pose shared-map control", "calibration_poses": 75},
    }
    with np.load(FULL_CAL/"calibration_and_scope.npz") as scope:
        cal_rows = scope["calibration_rows"]
        states = scope["truth_calibration_states"].copy()
        q = scope["measured_center_signal_mm"].copy()
        common_rows = scope["common_source_subset_rows"].copy()
        common_ids = scope["common_source_indices"].copy()
        primary = scope["primary_within_common"].copy()
    if len(states) != 75 or len(common_rows) != 83250 or primary.sum() != 83229:
        raise AssertionError("Expected75cal,83,250diagnostic,83,229primary")
    reconstruction = {}
    for variant, info in map_info.items():
        if digest(info["path"]) != info["sha256"]:
            raise RuntimeError(f"Frozen map changed:{variant}")
        model = BivariatePolynomial.load(info["path"])
        if model.degree != 3 or model.coefficient_count != 20 or model.coefficients.shape != (10, 2):
            raise AssertionError("Each map mustbeexactlyone10x2 fullcubic matrix")
        train = states[:, 0] == 0 if variant == "A0_frozen" else np.ones(len(states), bool)
        check, geometry = fit_joint_vector_cubic(q[train], states[train, 1:])
        delta = float(np.max(np.abs(check.coefficients-model.coefficients)))
        pred_delta = float(np.max(np.abs(check.predict(q)-model.predict(q))))
        if max(delta, pred_delta) > 1e-10:
            raise AssertionError(f"Canonical map not reproduced from authorizedcal data:{variant}")
        reconstruction[variant] = {"coefficient_max_delta": delta, "prediction_max_delta_deg": pred_delta,
                                    "full_geometry": geometry}
    cv = compare_folds(q, states)
    out.mkdir(parents=True)
    np.savez_compressed(out/"scope.npz", calibration_rows=cal_rows,
        common_source_subset_rows=common_rows, common_source_indices=common_ids,
        primary_within_common=primary)
    manifest = {"experiment": "Compact full-accommodation extension of accepted A0 vector cubic",
        "primary_variant": "A0_frozen", "control_variant": "shared75", "maps": map_info,
        "gaze_vector_coefficients": 10, "gaze_scalar_components": 20,
        "gaze_API": "predict(q_mm)", "gaze_accommodation_features": False,
        "forward_model": forward, "total_scalar_coefficients_per_variant": 47,
        "forward_change": "A0 compact PF12 to prior calibration-selected full-A DF27; the forward coefficients/order are not claimed unchanged",
        "source_path": str(source_path), "source_sha256": SOURCE_SHA,
        "common_diagnostic_count": 83250, "primary_count": 83229,
        "excluded_P1_gaze_deg": [12, 9.6], "excluded_rows": 21,
        "scope_sha256": digest(out/"scope.npz"), "canonical_map_reconstruction": reconstruction,
        "calibration_strategy_CV": cv, "CV_cannot_replace_primary": True,
        "pure75_calibration_summary_sha256": digest(FULL_CAL/"summary.json"),
        "A0_calibration_summary_sha256": digest(A0_CAL/"summary.json"),
        "preparation_source_sha256": digest(__file__),
        "frame_assumption": pure["frame_assumption"]}
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "primary": "A0_frozen", "scalar_coefficients": 47,
                      "informational_CV": cv["summary"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
