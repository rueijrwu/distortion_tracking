"""Compare requested full2D quadratic/cubic gaze feedback with frozen C8.

Polynomials determine gaze. Frozen forward models only correct the observed
grid mean to the central-ray estimate; converged primary paths are preserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "4")

import numpy as np

from p4_a0_bivariate_polynomial import BivariatePolynomial, self_check
from p4_a0_only_models import CenterCubic, GazeOnlyShape
from evaluate_p4_p1_a0_only import center_metrics, choose_roots, metrics, solve_gaze_only

ROOT = Path(__file__).resolve().parents[1]
BASE_RUNTIME = ROOT/"data/p4_p1_a0_only/runtime"
BASE_FILTERED = BASE_RUNTIME/"summary_excluding_A0_ADE12_BDE9p6.json"
BASE_FILTERED_SHA = "b9ff31c28b09461c5739df7abe79011ccfc3e39c9a8d05a2c95bbd10346626bd"
RESULT_KEYS = ("gaze_deg", "center_mm", "converged", "valid", "selected_start", "shape_sse_mm2",
    "fixedpoint_residual_deg", "raw_initial_gaze_deg", "bounded_initial_gaze_deg",
    "first_corrected_raw_gaze_deg", "first_corrected_gaze_deg", "primary_gaze_deg", "primary_converged",
    "primary_center_mm", "alternate_start_rescue", "distinct_converged_roots", "bound_ADE_BDE",
    "outside_sampled_12p4", "all_cycle_count", "all_growth_count", "all_iterations",
    "all_ever_cubic_bound_pressure")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gaze_metrics(estimate, truth):
    answer = metrics(estimate, truth)
    error = np.asarray(estimate)-np.asarray(truth)
    finite = np.isfinite(error).all(axis=1)
    for axis, label in enumerate(("ADE", "BDE")):
        if finite.any():
            answer[label]["median_abs_deg"] = float(np.median(np.abs(error[finite, axis])))
    return answer


def feedback_check():
    reference = np.array([(x, y) for x in (-.8, 0., .8) for y in (-.9, 0., .9)])
    shape = GazeOnlyShape(reference, np.array([.005, -.003, -.004, .006, .002, -.001,
                                              .002, -.003, .012, .004, -.010, .003]), "DF", 12)
    full = np.array([[.1, 0.], [.03, -6.], [5., .04], [.008, -.004], [-.006, .005],
                     [-.004, .003], [.001, -.001], [-.002, .0015], [.001, .001], [.001, -.001]])
    truth = np.array([[3.7, -8.2], [-11.6, 2.9], [12.5, -12.5], [-12.5, 12.5], [0., 0.], [12.5, 4.1]])
    mean_p1 = np.array([1.3, -.7])+np.arange(len(truth))[:, None]*[.02, -.01]
    results = {}
    for degree in (2, 3):
        count = 6 if degree == 2 else 10
        polynomial = BivariatePolynomial(degree, np.array([0., -1.]), np.ones(2), full[:count])
        q = np.stack((-truth[:, 1]/6, -1+(truth[:, 0]-.1)/5), axis=1)
        for _ in range(30):
            q -= np.linalg.solve(polynomial.jacobian(q), (polynomial.predict(q)-truth)[..., None])[..., 0]
        F = shape.evaluate(truth)
        observed = mean_p1[:, None, :]+q[:, None, :]+F
        inferred, calls = solve_gaze_only(observed, mean_p1, shape, polynomial)
        error = float(np.max(np.abs(inferred["gaze_deg"]-truth)))
        center_error = float(np.max(np.abs(inferred["center_mm"]-(mean_p1+q))))
        identity = float(np.max(np.abs(observed.mean(axis=1)-F.mean(axis=1)-(mean_p1+q))))
        if not inferred["primary_converged"].all() or error > 1e-4 or center_error > 1e-5 or identity > 1e-12:
            raise AssertionError((degree, error, center_error, identity))
        results[str(degree)] = {"max_blind_gaze_component_error_deg": error,
            "max_center_component_error_mm": center_error, "center_identity_error_mm": identity,
            "all_primary_converged": True, "map_calls": calls}
    conv = np.ones((3, 1), dtype=bool)
    if choose_roots(conv, conv, np.zeros((3, 1)), np.array([[10.], [1.], [.1]]))[0] != 0:
        raise AssertionError("A lower-SSE alternate replaced a converged primary")
    results["primary_preserved_despite_lower_alternate_SSE"] = True
    return results


def score_case(result, observed, mean_p1, truth, shape, polynomial, primary, excluded):
    central = observed[:, 4]
    oracle_center_gaze = polynomial.predict(central-mean_p1)
    oracle_forward_center = observed.mean(axis=1)-shape.evaluate(truth).mean(axis=1)
    oracle_forward_gaze = polynomial.predict(oracle_forward_center-mean_p1)
    stages = {"raw_unclipped": result["raw_initial_gaze_deg"], "raw_bounded": result["bounded_initial_gaze_deg"],
              "one_correction_unclipped": result["first_corrected_raw_gaze_deg"],
              "one_correction_bounded": result["first_corrected_gaze_deg"],
              "final_primary": result["primary_gaze_deg"], "final_selected": result["gaze_deg"],
              "oracle_true_central_ray": oracle_center_gaze, "oracle_forward_at_true_gaze": oracle_forward_gaze}
    groups = {}
    for label, mask in (("primary_user_exclusion_3943", primary),
                        ("unfiltered_diagnostic_3944", np.ones(len(truth), bool)),
                        ("excluded_one_state_diagnostic", excluded)):
        scores = {stage: gaze_metrics(values[mask], truth[mask]) for stage, values in stages.items()}
        scores["center_raw_grid_mean"] = center_metrics(observed.mean(axis=1)[mask], central[mask])
        scores["center_final_primary"] = center_metrics(result["primary_center_mm"][mask], central[mask])
        scores["center_final_selected"] = center_metrics(result["center_mm"][mask], central[mask])
        scores["oracle_forward_center"] = center_metrics(oracle_forward_center[mask], central[mask])
        delta = result["gaze_deg"][mask]-result["bounded_initial_gaze_deg"][mask]
        scores["correction_change_from_bounded_raw"] = gaze_metrics(delta, np.zeros_like(delta))
        scores["telemetry"] = {
            "attempted": int(mask.sum()), "primary_converged": int(result["primary_converged"][mask].sum()),
            "selected_converged": int(result["converged"][mask].sum()),
            "nonconverged": int((~result["converged"][mask]).sum()), "invalid": int((~result["valid"][mask]).sum()),
            "alternate_rescue": int(result["alternate_start_rescue"][mask].sum()),
            "distinct_roots": int(result["distinct_converged_roots"][mask].sum()),
            "converged_primary_replaced": int(np.sum(result["primary_converged"][mask] & (result["selected_start"][mask] != 0))),
            "bound_ADE_BDE": result["bound_ADE_BDE"][mask].sum(axis=0).tolist(),
            "unsampled_12p4_to_12p5_strip": int(result["outside_sampled_12p4"][mask].sum()),
            "raw_initial_bound_pressure": int(np.any(np.abs(result["raw_initial_gaze_deg"][mask]) > 12.5, axis=1).sum()),
            "all_start_cycle_events": int(result["all_cycle_count"][:, mask].sum()),
            "all_start_growth_events": int(result["all_growth_count"][:, mask].sum()),
            "max_evaluations_per_start": int(result["all_iterations"][:, mask].max()),
            "all_start_cubic_bound_pressure": int(np.any(result["all_ever_cubic_bound_pressure"][:, mask], axis=2).sum()),
        }
        groups[label] = scores
    return groups, oracle_center_gaze, oracle_forward_center, oracle_forward_gaze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--calibration-dir", type=Path, default=ROOT/"data/p4_p1_a0_bivariate/calibration")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"data/p4_p1_a0_bivariate/runtime")
    args = parser.parse_args()
    checks = {"polynomial": self_check(), "feedback": feedback_check()}
    print(json.dumps({"checks": checks}), flush=True)
    if args.check_only:
        return
    calibration_path = args.calibration_dir/"summary.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if digest(ROOT/"Script/p4_a0_bivariate_polynomial.py") != calibration["model_source_sha256"]:
        raise RuntimeError("Polynomial source changed since fitting")
    obs_path = Path(calibration["A0_observation_path"])
    if digest(obs_path) != calibration["A0_observation_sha256"]:
        raise RuntimeError("A0 observations changed")
    if digest(BASE_FILTERED) != BASE_FILTERED_SHA:
        raise RuntimeError("Pinned existing user-exclusion baseline summary changed")
    with np.load(obs_path) as obs:
        rows = np.flatnonzero(obs["holdout25"])
        truth = obs["gaze_deg"][rows].copy()
        observed = obs["absolute_p4_mm"][rows].copy()
        mean_p1 = obs["mean_p1_absolute_mm"][rows].copy()
        ids = obs["source_indices"][rows].copy()
        excluded = np.all(np.isclose(truth, [12., 9.6], atol=1e-9, rtol=0), axis=1)
        if np.any(obs["calibration25"][rows]):
            raise AssertionError("Calibration/test overlap")
    primary = ~excluded
    if len(rows) != 3944 or excluded.sum() != 1 or primary.sum() != 3943 or not np.array_equal(ids[excluded], [7554]):
        raise AssertionError("Exact requested exclusion/test identities differ")
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(f"Fresh output directory required:{out}")
    out.mkdir(parents=True)
    cases = {}
    C8path = Path(calibration["C8"]["path"])
    if digest(C8path) != calibration["C8"]["sha256"]:
        raise RuntimeError("Frozen C8 changed")
    baseline_map = CenterCubic.load(C8path)
    for shape_name in ("PF12", "PF33"):
        shape_info = calibration["shapes"][shape_name]
        shape_path = Path(shape_info["path"])
        if digest(shape_path) != shape_info["sha256"]:
            raise RuntimeError("Frozen A0 shape changed")
        shape = GazeOnlyShape.load(shape_path)
        for map_name in ("C8", "Q12", "C20"):
            case_name = f"{shape_name}_{map_name}"
            if map_name == "C8":
                source_path = BASE_RUNTIME/f"runtime_{shape_name}_C8.npz"
                with np.load(source_path) as saved:
                    if not np.array_equal(saved["source_indices"], ids) or not np.array_equal(saved["truth_gaze_deg"], truth):
                        raise AssertionError("Saved C8 andrequested maps use different test labels")
                    result = {key: saved[key].copy() for key in RESULT_KEYS}
                polynomial, wall = baseline_map, None
                center_reconstruction = observed.mean(axis=1)-shape.evaluate(result["gaze_deg"]).mean(axis=1)
                center_delta = float(np.max(np.abs(center_reconstruction-result["center_mm"])))
                raw_delta = float(np.max(np.abs(polynomial.predict(observed.mean(axis=1)-mean_p1)-result["raw_initial_gaze_deg"])))
                if max(center_delta, raw_delta) > 1e-10:
                    raise RuntimeError("Frozen C8 observation/model reconstruction differs")
                provenance = {"role": "frozen separate-axis cubic baseline rescored onidentical labels",
                    "source_path": str(source_path), "source_sha256": digest(source_path),
                    "center_reconstruction_max_abs_mm": center_delta, "raw_prediction_delta_deg": raw_delta}
                count, calls = 8, None
            else:
                info = calibration["maps"][map_name]
                if not info["eligible"] or digest(info["path"]) != info["sha256"]:
                    raise RuntimeError(f"Requested map is not a full-rank frozen calibration:{map_name}")
                polynomial = BivariatePolynomial.load(info["path"])
                began = time.perf_counter()
                result, calls = solve_gaze_only(observed, mean_p1, shape, polynomial)
                wall = time.perf_counter()-began
                provenance = {"role": "requested full2D polynomial", "map_path": info["path"], "map_sha256": info["sha256"]}
                count = polynomial.coefficient_count
            scores, oracle_center, oracle_forward_center, oracle_forward_gaze = score_case(
                result, observed, mean_p1, truth, shape, polynomial, primary, excluded)
            if map_name != "C8":
                np.savez_compressed(out/f"runtime_{case_name}.npz", **result,
                    source_indices=ids, a0_observation_rows=rows, truth_gaze_deg=truth,
                    primary_user_exclusion_mask=primary, user_excluded_state_mask=excluded,
                    gaze_error_deg=result["gaze_deg"]-truth,
                    center_error_mm=result["center_mm"]-observed[:, 4],
                    oracle_true_center_gaze_deg=oracle_center,
                    oracle_forward_center_mm=oracle_forward_center, oracle_forward_gaze_deg=oracle_forward_gaze)
            cases[case_name] = {"shape_coefficients": shape.spatial_terms, "gaze_coefficients": count,
                "total_coefficients": shape.spatial_terms+count, "runtime_seconds": wall,
                "model_calls": calls, "provenance": provenance, "scores": scores,
                "shape_sha256": digest(shape_path)}
            print(json.dumps({"case": case_name, "total_coefficients": shape.spatial_terms+count,
                "runtime_seconds": wall, "primary_final": scores["primary_user_exclusion_3943"]["final_selected"],
                "primary_status": scores["primary_user_exclusion_3943"]["telemetry"]}, indent=2), flush=True)
    summary = {"experiment": "Requested full2D order2/order3 gaze with centroid-only forward correction",
        "primary_count": 3943, "unfiltered_diagnostic_count": 3944,
        "excluded_by_user": calibration["excluded_by_user"], "source_data_deleted": False,
        "calibration_degree_selection": calibration["selection"],
        "all_requested_degrees_evaluated_regardless_selection": True,
        "gaze_determined_by_polynomial_primary": True,
        "runtime_bound_deg": 12.5, "actual_ray_support_deg": 12.4,
        "cases": cases, "checks": checks,
        "calibration_summary_sha256": digest(calibration_path),
        "A0_observation_sha256": digest(obs_path), "C8_baseline_exclusion_summary_sha256": BASE_FILTERED_SHA,
        "runtime_source_sha256": digest(__file__),
        "polynomial_source_sha256": digest(ROOT/"Script/p4_a0_bivariate_polynomial.py"),
        "frozen_feedback_source_sha256": digest(ROOT/"Script/evaluate_p4_p1_a0_only.py"),
        "no_accommodation_input_parameters_or_optimization": True,
        "frame_assumption": calibration["frame_assumption"],
        "noise_scope": "Noiseless simulated observations/model mismatch; noinstrument-noise claim"}
    (out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
