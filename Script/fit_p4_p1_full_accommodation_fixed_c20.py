"""Fit exactly20 shared cubic gaze values; accommodation belongs only to F27."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from p4_a0_bivariate_polynomial import BivariatePolynomial, design, self_check
from p4_a0_only_models import design_geometry

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/"data/p4_p1_limited_gaze_12p5/source_subset.npz"
SOURCE_SHA = "97c6db1b455f9abf7354351fa768cca7eea67198d8a4bfd14bbe684475462daf"
FORWARD = ROOT/"data/p4_p1_limited_gaze_12p5/models/new75_DF27.npz"
FORWARD_SHA = "a590fb36e4561bb7f23af016d49330dabc2b038e4b7dee41288fe930445d52ab"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_joint_vector_cubic(q, true_gaze):
    """One design matrix and one jointly solved10x2 vector-coefficient matrix."""
    mean, scale = q.mean(axis=0), q.std(axis=0)
    if np.any(scale <= 0):
        raise ValueError("Degenerate training-only detector normalization")
    matrix = design((q-mean)/scale, 3)
    coefficients = np.linalg.lstsq(matrix, true_gaze, rcond=None)[0]
    model = BivariatePolynomial(3, mean, scale, coefficients)
    geometry = design_geometry(matrix)
    return model, {"full_rank": bool(geometry["rank"] == 10),
        "rank_per_output": {"ADE": geometry["rank"], "BDE": geometry["rank"]},
        "normalized_design": geometry, "raw_detector_powers": design_geometry(design(q, 3)),
        "normalization_mean_mm": mean.tolist(), "normalization_scale_mm": scale.tolist(),
        "coefficient_matrix_shape": [10, 2]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"data/p4_p1_full_accommodation_fixed_c20/calibration")
    args = parser.parse_args()
    checks = self_check()
    if len(inspect.signature(BivariatePolynomial.predict).parameters) != 2:
        raise AssertionError("Gaze API must be predict(self,q), without accommodation")
    print(json.dumps({"full2D_checks": checks}), flush=True)
    if args.check_only:
        return
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(f"Fresh output required:{out}")
    if digest(SOURCE) != SOURCE_SHA or digest(FORWARD) != FORWARD_SHA:
        raise RuntimeError("Frozen raw source or75-pose F27 changed")
    with np.load(SOURCE) as source:
        states = source["states"]
        grid = np.array([-12.4, -6.4, 0., 6.4, 12.4])
        gaze_cal = np.all(np.any(np.isclose(states[:, 1:, None], grid, atol=1e-9, rtol=0), axis=2), axis=1)
        A_cal = np.any(np.isclose(states[:, 0, None], [0., 2., 4.], atol=1e-12, rtol=0), axis=1)
        cal = gaze_cal & A_cal
        cal_rows = np.flatnonzero(cal)
        cal_states = states[cal].copy()
        q = (source["absolute_p4_mm"][cal, 4]-source["mean_p1_absolute_mm"][cal]).copy()
        ids = source["source_indices"].copy()
        common = source["common_holdout"].astype(bool)
        excluded = np.all(np.isclose(states[:, 1:], [12., 9.6], atol=1e-9, rtol=0), axis=1)
        zero = np.flatnonzero(np.all(states == [0., 0., 0.], axis=1))
        reference = source["absolute_p4_mm"][zero[0]]-source["absolute_p4_mm"][zero[0], 4]
    if len(states) != 83349 or cal.sum() != 75 or common.sum() != 83250:
        raise AssertionError("Expected full limited source,75 calibration and83,250 common rows")
    if excluded.sum() != 21 or np.any(excluded & cal) or not np.all(common[excluded]):
        raise AssertionError("Exact user-excluded P1 gaze mustcover21 heldout rows only")
    if (common & ~excluded).sum() != 83229 or len(zero) != 1 or not cal[zero[0]]:
        raise AssertionError("Primary count/reference anchor mismatch")
    # Verify the forward artifact's learned values came from these exact75
    # physical labels/reference. Its coefficients are reused unchanged.
    with np.load(FORWARD) as saved:
        forward_states = saved["calibration_states"]
        forward_ids = saved["calibration_source_indices"]
        if len(forward_states) != 75 or not np.array_equal(np.sort(forward_ids), np.sort(ids[cal])):
            raise AssertionError("F27 calibration observation identities differ")
        state_by_id = {int(index): state for index, state in zip(ids[cal], cal_states)}
        if not all(np.array_equal(state_by_id[int(index)], state) for index, state in zip(forward_ids, forward_states)):
            raise AssertionError("F27 calibration state labels differ")
        reference_delta = float(np.max(np.abs(saved["reference_mm"]-reference)))
        if reference_delta != 0:
            raise AssertionError("F27 raw zero-gaze reference differs")
    # Known A labels select calibration rows and groups only. They are NEVER
    # features in this fit or in the resulting pure predict(q) model.
    model, geometry = fit_joint_vector_cubic(q, cal_states[:, 1:])
    folds = []
    for gaze in np.unique(cal_states[:, 1:], axis=0):
        if np.all(gaze == 0):
            continue
        test = np.all(cal_states[:, 1:] == gaze, axis=1)
        if test.sum() != 3:
            raise AssertionError("Each fold mustholdallthreeA observations forone gaze")
        fitted, info = fit_joint_vector_cubic(q[~test], cal_states[~test, 1:])
        prediction = fitted.predict(q[test])
        loss = float(np.mean((prediction-cal_states[test, 1:])**2))
        folds.append({"test_gaze_deg": gaze.tolist(), "test_A_D": cal_states[test, 0].tolist(),
            "coordinate_MSE_deg2": loss, "full_rank": info["full_rank"], "geometry": info})
    losses = np.array([fold["coordinate_MSE_deg2"] for fold in folds])
    eligible = bool(geometry["full_rank"] and len(folds) == 24 and all(f["full_rank"] for f in folds) and np.isfinite(losses).all())
    if not eligible:
        raise RuntimeError("Exact20-value cubic isnotfullrank/finite inallcalibrationfolds")
    # Historical A-order0 is checked only AFTER this independent fit. It is
    # never copied into the new fit, supplied as a prior, or used for selection.
    historical_path = ROOT/"data/p4_p1_accommodation_bivariate/calibration/map_C20.npz"
    history_check = None
    if historical_path.exists():
        with np.load(historical_path) as historical:
            if int(historical["accommodation_order"]) != 0:
                raise AssertionError("Historical comparison isnotorder0")
            coefficient_delta = float(np.max(np.abs(historical["coefficients"]-model.coefficients)))
            normalizer_delta = float(max(np.max(np.abs(historical["mean_mm"]-model.mean_mm)),
                                         np.max(np.abs(historical["scale_mm"]-model.scale_mm))))
            if max(coefficient_delta, normalizer_delta) > 1e-11:
                raise AssertionError("Pure20-value fit differs from historical order0 verification")
            history_check = {"artifact": str(historical_path), "sha256": digest(historical_path),
                "max_coefficient_delta": coefficient_delta, "max_normalizer_delta_mm": normalizer_delta,
                "role": "post-fit identity verification only; unused expanded-A candidates are not deployed"}
    out.mkdir(parents=True)
    model_path = out/"fixed_C20.npz"
    model.save(model_path)
    np.savez_compressed(out/"calibration_and_scope.npz", calibration_rows=cal_rows,
        calibration_source_indices=ids[cal], truth_calibration_states=cal_states,
        measured_center_signal_mm=q, common_source_subset_rows=np.flatnonzero(common),
        common_source_indices=ids[common], primary_within_common=(~excluded)[common],
        excluded_source_subset_rows=np.flatnonzero(excluded), excluded_source_indices=ids[excluded])
    summary = {"experiment": "Exactly20 shared full2D cubic gaze values with full-accommodation F27",
        "gaze_model": {"path": str(model_path), "sha256": digest(model_path), "coefficient_count": 20,
            "degree": 3, "API": "predict(q_mm)", "accommodation_features": False,
            "vector_coefficients": 10, "scalar_components": 20, "joint_matrix_fit": True,
            "coefficients_ADE_BDE": model.coefficients.tolist(), "geometry": geometry},
        "forward_model": {"path": str(FORWARD), "sha256": FORWARD_SHA, "coefficient_count": 27,
            "calibration_count": 75, "reference_max_abs_delta_mm": reference_delta},
        "total_shared_coefficients": 47, "map_choice": "User-fixed C20; noA-order model selection",
        "calibration_CV": {"folds": folds, "coordinate_MSE_deg2": float(losses.mean()),
            "SE_coordinate_MSE_deg2": float(losses.std(ddof=1)/np.sqrt(len(losses))),
            "pooled_radial_RMS_deg": float(np.sqrt(2*losses.mean())), "eligible": eligible},
        "post_fit_history_verification": history_check,
        "source_path": str(SOURCE), "source_sha256": SOURCE_SHA,
        "scope_sha256": digest(out/"calibration_and_scope.npz"),
        "common_diagnostic_count": 83250, "primary_count": 83229, "excluded_count": 21,
        "excluded_P1_gaze_deg": [12, 9.6], "all_source_observations_preserved": True,
        "model_source_sha256": digest(ROOT/"Script/p4_a0_bivariate_polynomial.py"),
        "trainer_source_sha256": digest(__file__), "checks": checks,
        "frame_assumption": "Pinned P1/P4 nominal SI coordinates; externalregistration notindependently proved"}
    (out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "total_shared_coefficients": 47,
        "gaze_CV_radial_RMS_deg": summary["calibration_CV"]["pooled_radial_RMS_deg"],
        "pure_C20_verified": history_check}, indent=2), flush=True)


if __name__ == "__main__":
    main()
