"""Gaze from new A0-only cubics, iteratively corrected by the A0-only shape mean.

The cubic is the gaze estimator. The forward shape supplies only the predicted
centroid offset; observed whole-grid shape never replaces a converged primary.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "4")

import numpy as np

from p4_a0_only_models import CenterCubic, GazeOnlyShape, self_check
from p4_fixed_a0_gaze import solve_fixed_a0

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose_roots(converged, valid, residual, shape_sse):
    """A converged raw-mean primary always wins, regardless of another root's SSE."""
    converged, valid = np.asarray(converged, bool), np.asarray(valid, bool)
    selected = np.zeros(converged.shape[1], dtype=int)
    for row in range(len(selected)):
        if converged[0, row] and valid[0, row]:
            continue
        alternate = np.flatnonzero(converged[1:, row] & valid[1:, row])+1
        if len(alternate):
            selected[row] = alternate[np.argmin(shape_sse[alternate, row])]
        else:
            finite = np.flatnonzero(valid[:, row] & np.isfinite(residual[:, row]))
            if len(finite):
                selected[row] = finite[np.lexsort((shape_sse[finite, row], residual[finite, row]))[0]]
    return selected


def solve_gaze_only(y4_mm, m1_mm, shape, cubic, *, keep_history=True):
    calls = {"forward_shape": 0, "separate_center_cubics": 0}

    def forward(gaze):
        calls["forward_shape"] += 1
        evaluated = shape.evaluate(gaze, diagnostics=True)
        return evaluated["prediction"], evaluated["valid"]

    def inverse_center(q):
        calls["separate_center_cubics"] += 1
        return cubic.predict(q)

    # Reuse only the observation-only Picard safeguards, not any learned model.
    # These callables have no accommodation input or optimization interface.
    result = solve_fixed_a0(y4_mm, m1_mm, forward, inverse_center, keep_history=keep_history)
    selected = choose_roots(result["all_converged"], result["all_valid"],
                            result["all_fixedpoint_residual_deg"], result["all_shape_sse_mm2"])
    rows = np.arange(len(selected))
    valid = result["all_valid"][selected, rows]
    result["selected_start"] = selected
    result["gaze_deg"] = np.where(valid[:, None], result["all_gaze_deg"][selected, rows], np.nan)
    result["center_mm"] = np.where(valid[:, None], result["all_center_mm"][selected, rows], np.nan)
    result["converged"] = result["all_converged"][selected, rows] & valid
    result["valid"] = valid
    result["shape_sse_mm2"] = result["all_shape_sse_mm2"][selected, rows]
    result["fixedpoint_residual_deg"] = result["all_fixedpoint_residual_deg"][selected, rows]
    result["primary_gaze_deg"] = np.where(result["all_valid"][0, :, None], result["all_gaze_deg"][0], np.nan)
    result["primary_converged"] = result["all_converged"][0] & result["all_valid"][0]
    result["alternate_start_rescue"] = ~result["primary_converged"] & result["converged"]
    result["bound_ADE_BDE"] = np.isclose(np.abs(result["gaze_deg"]), 12.5, atol=1e-7, rtol=0)
    result["outside_sampled_12p4"] = np.any(np.abs(result["gaze_deg"]) > 12.4+1e-10, axis=1)
    result["primary_preserved"] = result["primary_converged"] & (selected == 0)
    for key in tuple(result):
        if key.startswith("accommodation_"):
            del result[key]
    if np.any(result["primary_converged"] & (selected != 0)):
        raise AssertionError("A converged polynomial-primary path was replaced")
    return result, calls


def metrics(estimate, truth):
    e = np.asarray(estimate)-np.asarray(truth)
    finite = np.isfinite(e).all(axis=1)
    good = e[finite]
    result = {"attempted": len(e), "finite": int(finite.sum()), "nonfinite": int((~finite).sum())}
    if not len(good):
        return result
    for axis, name in enumerate(("ADE", "BDE")):
        x = good[:, axis]
        result[name] = {"bias_deg": float(x.mean()), "RMS_deg": float(np.sqrt(np.mean(x*x))),
                        "MAE_deg": float(np.abs(x).mean()), "P95_abs_deg": float(np.percentile(np.abs(x), 95)),
                        "max_abs_deg": float(np.abs(x).max())}
    r = np.linalg.norm(good, axis=1)
    result["radial"] = {"RMS_deg": float(np.sqrt(np.mean(r*r))), "median_deg": float(np.median(r)),
                         "P95_deg": float(np.percentile(r, 95)), "P99_deg": float(np.percentile(r, 99)),
                         "max_deg": float(r.max()), "fraction_le_0p5_all_attempted": float(np.sum(r <= .5)/len(e))}
    return result


def center_metrics(estimated, labeled):
    e = (np.asarray(estimated)-np.asarray(labeled))*1000.0
    good = e[np.isfinite(e).all(axis=1)]
    if not len(good):
        return {"attempted": len(e), "finite": 0}
    r = np.linalg.norm(good, axis=1)
    return {"attempted": len(e), "finite": len(good), "bias_xy_um": good.mean(axis=0).tolist(),
            "vector_RMS_um": float(np.sqrt(np.mean(r*r))), "median_um": float(np.median(r)),
            "P95_um": float(np.percentile(r, 95)), "max_um": float(r.max())}


def runtime_self_check():
    # Actual new no-A shape, nontrivial independent cubics and centroid bias.
    reference = np.array([(x, y) for x in (-.8, 0., .8) for y in (-.9, 0., .9)])
    coefficients = np.array([.005, -.003, -.004, .006, .002, -.001,
                             .002, -.003, .012, .004, -.010, .003])
    shape = GazeOnlyShape(reference, coefficients, "DF", 12)
    cubic = CenterCubic(np.zeros(2), np.ones(2), np.array([[0., -6., 0., -.12], [.1, 5., 0., .05]]))
    truth = np.array([[3.7, -8.2], [-11.6, 2.9], [12.5, -12.5], [-12.5, 12.5], [0., 0.], [12.5, 4.1]])
    q = np.stack((-truth[:, 1]/6, (truth[:, 0]-.1)/5), axis=1)
    for _ in range(16):
        q[:, 0] -= (-6*q[:, 0]-.12*q[:, 0]**3-truth[:, 1])/(-6-.36*q[:, 0]**2)
        q[:, 1] -= (.1+5*q[:, 1]+.05*q[:, 1]**3-truth[:, 0])/(5+.15*q[:, 1]**2)
    mean_p1 = np.array([1.3, -.7])+np.arange(len(truth))[:, None]*[.02, -.01]
    F = shape.evaluate(truth)
    observed = mean_p1[:, None, :]+q[:, None, :]+F
    inferred, calls = solve_gaze_only(observed, mean_p1, shape, cubic)
    max_error = float(np.max(np.abs(inferred["gaze_deg"]-truth)))
    identity = float(np.max(np.abs(observed.mean(axis=1)-F.mean(axis=1)-(mean_p1+q))))
    center_error = float(np.max(np.abs(inferred["center_mm"]-(mean_p1+q))))
    if not inferred["converged"].all() or max_error > 1e-4 or identity > 1e-12 or center_error > 1e-5:
        raise AssertionError((max_error, identity, center_error))
    # Explicitly test that a lower-SSE alternate cannot replace a converged primary.
    converged = np.array([[True, False, False], [True, True, False], [True, True, False]])
    valid = np.ones_like(converged)
    sse = np.array([[10., 10., 9.], [1., 2., 4.], [.1, 1., 1.]])
    residual = np.array([[0., .1, .3], [0., 0., .2], [0., 0., .1]])
    if not np.array_equal(choose_roots(converged, valid, residual, sse), [0, 2, 2]):
        raise AssertionError("Primary-preserving/alternate-rescue selector failed")
    if "A" in inspect.signature(shape.evaluate).parameters or len(inspect.signature(cubic.predict).parameters) != 1:
        raise AssertionError("Unexpected accommodation interface")
    return {"all_converged": True, "max_gaze_component_error_deg": max_error,
            "center_identity_error_mm": identity, "max_center_component_error_mm": center_error,
            "primary_preserved_despite_lower_alternate_SSE": True, "map_calls": calls,
            "synthetic_centroid_bias_max_mm": float(np.max(np.linalg.norm(F.mean(axis=1), axis=1)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--calibration-dir", type=Path, default=ROOT/"data/p4_p1_a0_only/calibration")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"data/p4_p1_a0_only/runtime")
    args = parser.parse_args()
    checks = {"model": self_check(), "runtime": runtime_self_check()}
    print(json.dumps({"checks": checks}), flush=True)
    if args.check_only:
        return
    calibration = json.loads((args.calibration_dir/"summary.json").read_text(encoding="utf-8"))
    if not calibration["no_accommodation_parameters"] or not calibration["no_prior_learned_coefficients"]:
        raise AssertionError("Calibration did not declare independent A0-only fitting")
    if digest(ROOT/"Script/p4_a0_only_models.py") != calibration["model_source_sha256"]:
        raise RuntimeError("No-A model source changed since fitting")
    obs_path = args.calibration_dir/"a0_observations.npz"
    if digest(obs_path) != calibration["observation_artifact_sha256"]:
        raise RuntimeError("A0 observations changed since fitting")
    with np.load(obs_path) as data:
        test_rows = np.flatnonzero(data["holdout25"])
        gaze_truth = data["gaze_deg"][test_rows].copy()
        observed = data["absolute_p4_mm"][test_rows].copy()
        mean_p1 = data["mean_p1_absolute_mm"][test_rows].copy()
        source_ids = data["source_indices"][test_rows].copy()
        flagged = data["quality_flagged"][test_rows].copy()
        prior_comparison = data["prior_comparison3936"][test_rows].copy()
    if len(test_rows) != 3944 or flagged.sum() != 1 or prior_comparison.sum() != 3936:
        raise AssertionError("Expected own3,944 heldout rows, one flag, and3,936 old-comparison rows")
    case = calibration["cases"]["25"]
    cubic_info = case["center_C8"]
    if not cubic_info["eligible"]:
        raise RuntimeError("Center cubic is not full rank in full calibration and every fold")
    Cpath = Path(cubic_info["path"])
    if digest(Cpath) != cubic_info["sha256"]:
        raise RuntimeError("A0-only center cubic changed")
    cubic = CenterCubic.load(Cpath)
    winner = case["shape_selection"]["selected"]
    low = [name for name, value in case["shape_candidates"].items()
           if value["eligible"] and value["spatial_coefficients"] == 12]
    names = [winner]
    if case["shape_candidates"][winner]["spatial_coefficients"] == 33 and low:
        names.append(min(low, key=lambda name: case["shape_candidates"][name]["CV_coordinate_MSE_um2"]))
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(f"Fresh runtime directory required: {out}")
    out.mkdir(parents=True)
    summaries = {}
    for name in names:
        info = case["shape_candidates"][name]
        path = Path(info["path"])
        if digest(path) != info["sha256"]:
            raise RuntimeError("A0-only shape coefficients changed")
        shape = GazeOnlyShape.load(path)
        started = time.perf_counter()
        result, calls = solve_gaze_only(observed, mean_p1, shape, cubic)
        wall = time.perf_counter()-started
        # Truth enters only these diagnostics, after the inference returns.
        center_truth = observed[:, 4]
        oracle_center_gaze = cubic.predict(center_truth-mean_p1)
        true_gaze_F = shape.evaluate(gaze_truth)
        oracle_forward_center = observed.mean(axis=1)-true_gaze_F.mean(axis=1)
        oracle_forward_gaze = cubic.predict(oracle_forward_center-mean_p1)
        stages = {"raw_mean_unclipped": result["raw_initial_gaze_deg"],
                  "raw_mean_bounded": result["bounded_initial_gaze_deg"],
                  "one_correction_raw": result["first_corrected_raw_gaze_deg"],
                  "one_correction_bounded": result["first_corrected_gaze_deg"],
                  "final_primary": result["primary_gaze_deg"], "final_selected": result["gaze_deg"],
                  "oracle_labeled_center": oracle_center_gaze, "oracle_forward_at_true_gaze": oracle_forward_gaze}
        scores = {}
        for label, mask in (("all", np.ones(len(test_rows), bool)), ("ordinary", ~flagged),
                            ("flagged", flagged), ("prior_comparison3936", prior_comparison)):
            scores[label] = {stage: metrics(values[mask], gaze_truth[mask]) for stage, values in stages.items()}
            scores[label]["center_raw_grid_mean"] = center_metrics(observed.mean(axis=1)[mask], center_truth[mask])
            scores[label]["center_primary"] = center_metrics(result["primary_center_mm"][mask], center_truth[mask])
            scores[label]["center_selected"] = center_metrics(result["center_mm"][mask], center_truth[mask])
            scores[label]["oracle_forward_center"] = center_metrics(oracle_forward_center[mask], center_truth[mask])
            delta = result["gaze_deg"]-result["bounded_initial_gaze_deg"]
            scores[label]["correction_change_from_raw_bounded"] = metrics(delta[mask], np.zeros((int(mask.sum()), 2)))
        telemetry = {
            "attempted": len(test_rows), "primary_converged": int(result["primary_converged"].sum()),
            "selected_converged": int(result["converged"].sum()), "nonconverged": int((~result["converged"]).sum()),
            "invalid_returned": int((~result["valid"]).sum()),
            "alternate_rescue": int(result["alternate_start_rescue"].sum()),
            "distinct_converged_roots": int(result["distinct_converged_roots"].sum()),
            "selected_seed_counts_raw_zero_negative_raw": np.bincount(result["selected_start"], minlength=3).tolist(),
            "primary_replaced_despite_converged": int(np.sum(result["primary_converged"] & (result["selected_start"] != 0))),
            "bound_ADE_BDE_counts": result["bound_ADE_BDE"].sum(axis=0).tolist(),
            "estimated_in_unsampled_12p4_to_12p5_strip": int(result["outside_sampled_12p4"].sum()),
            "raw_initializer_projected_count": int(np.any(np.abs(result["raw_initial_gaze_deg"]) > 12.5, axis=1).sum()),
            "all_start_cycles": int(result["all_cycle_count"].sum()),
            "all_start_growth_events": int(result["all_growth_count"].sum()),
            "maximum_evaluations_per_start": int(result["all_iterations"].max()),
        }
        np.savez_compressed(out/f"runtime_{name}_C8.npz", **result,
                            source_indices=source_ids, a0_observation_rows=test_rows,
                            truth_gaze_deg=gaze_truth, labeled_center_mm=center_truth,
                            gaze_error_deg=result["gaze_deg"]-gaze_truth,
                            center_error_mm=result["center_mm"]-center_truth,
                            quality_flagged=flagged, prior_comparison3936=prior_comparison,
                            oracle_center_gaze_deg=oracle_center_gaze,
                            oracle_forward_center_mm=oracle_forward_center,
                            oracle_forward_gaze_deg=oracle_forward_gaze)
        summaries[name] = {"role": "calibration_selected" if name == winner else "predeclared_low_order_same_C8_control",
                           "shape_coefficients": shape.spatial_terms, "center_cubic_coefficients": 8,
                           "total_fitted_coefficients": shape.spatial_terms+8,
                           "runtime_seconds": wall, "model_calls": calls, "telemetry": telemetry,
                           "metrics": scores, "shape_sha256": digest(path), "center_cubic_sha256": digest(Cpath)}
        print(json.dumps({"case": name, "runtime_seconds": wall, "telemetry": telemetry,
                          "ordinary_final": scores["ordinary"]["final_selected"],
                          "all_final": scores["all"]["final_selected"]}, indent=2), flush=True)
    summary = {"experiment": "Genuinely A0-only calibration and polynomial-primary gaze inference",
               "selected_shape": winner, "heldout": 3944, "retained_flagged": 1,
               "actual_data_support_deg": 12.4, "runtime_bound_deg": 12.5,
               "gaze_update": "C8(mean(observed P4)-mean(F(current gaze))-mean(observed P1)); forward supplies centroid offset only",
               "root_selection": "Keep converged primary; alternate root may rescue only failed primary",
               "no_accommodation_input_or_parameters_or_profile": True,
               "calibration_summary_sha256": digest(args.calibration_dir/"summary.json"),
               "a0_observations_sha256": digest(obs_path),
               "model_source_sha256": digest(ROOT/"Script/p4_a0_only_models.py"),
               "runtime_source_sha256": digest(__file__),
               "picard_engine_sha256": digest(ROOT/"Script/p4_fixed_a0_gaze.py"),
               "checks": checks, "cases": summaries,
               "noise_scope": "Noiseless simulation/model mismatch only; no instrument noise accuracy claim",
               "frame_assumption": calibration["frame_assumption"]}
    (out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
