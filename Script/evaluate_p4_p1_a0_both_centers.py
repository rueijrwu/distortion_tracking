"""Compare A=0 PF12+B3+new C20 with frozen A=0 PF12+old C20."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from evaluate_p4_p1_a0_only import choose_roots, center_metrics, metrics
from p1_centroid_bias import P1CentroidBias
from p4_a0_bivariate_polynomial import BivariatePolynomial
from p4_a0_only_models import GazeOnlyShape
from p4_fixed_a0_both_centers import solve_fixed_a0, synthetic_check

ROOT = Path(__file__).resolve().parents[1]
CAL = ROOT / "data/p4_p1_a0_both_centers/calibration"
OLD_CAL = ROOT / "data/p4_p1_a0_bivariate/calibration"
OLD_RUNTIME = ROOT / "data/p4_p1_a0_bivariate/runtime/runtime_PF12_C20.npz"
OUT = ROOT / "data/p4_p1_a0_both_centers/runtime"
EXCLUDED_GAZE = np.array([12., 9.6])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def infer(observed, mean_p1, shape, bias, cubic):
    """The only measured inputs are the P4 nine-ray grid and P1 mean."""
    calls = {"p4_forward": 0, "p1_bias": 0, "gaze_cubic": 0}

    def forward(gaze):
        calls["p4_forward"] += 1
        evaluated = shape.evaluate(gaze, diagnostics=True)
        return evaluated["prediction"], evaluated["valid"]

    def bias_at_gaze(gaze):
        calls["p1_bias"] += 1
        return bias.predict(gaze)

    def inverse(q):
        calls["gaze_cubic"] += 1
        return cubic.predict(q)

    result = solve_fixed_a0(observed, mean_p1, forward, inverse,
                            p1_bias_at_zero=bias_at_gaze, keep_history=False)
    selected = choose_roots(result["all_converged"], result["all_valid"],
                            result["all_fixedpoint_residual_deg"], result["all_shape_sse_mm2"])
    rows = np.arange(len(selected))
    valid = result["all_valid"][selected, rows]
    gaze = np.where(valid[:, None], result["all_gaze_deg"][selected, rows], np.nan)
    c4 = np.where(valid[:, None], result["all_center_mm"][selected, rows], np.nan)
    c1 = mean_p1 - bias.predict(gaze)
    q = c4 - c1
    initial = result["bounded_initial_gaze_deg"]
    first_c4 = observed.mean(axis=1) - shape.evaluate(initial).mean(axis=1)
    first_c1 = mean_p1 - bias.predict(initial)
    result.update({
        "selected_start": selected, "gaze_deg": gaze, "center_mm": c4,
        "p4_center_mm": c4, "p1_center_mm": c1, "q_mm": q,
        "converged": result["all_converged"][selected, rows] & valid,
        "valid": valid,
        "shape_sse_mm2": result["all_shape_sse_mm2"][selected, rows],
        "fixedpoint_residual_deg": result["all_fixedpoint_residual_deg"][selected, rows],
        "primary_gaze_deg": np.where(result["all_valid"][0, :, None], result["all_gaze_deg"][0], np.nan),
        "primary_converged": result["all_converged"][0] & result["all_valid"][0],
        "primary_p4_center_mm": result["all_center_mm"][0],
        "primary_p1_center_mm": mean_p1 - bias.predict(result["all_gaze_deg"][0]),
        "raw_p4_center_mm": observed.mean(axis=1), "raw_p1_center_mm": mean_p1,
        "raw_q_mm": observed.mean(axis=1) - mean_p1,
        "first_p4_center_mm": first_c4, "first_p1_center_mm": first_c1,
        "first_q_mm": first_c4 - first_c1,
        "alternate_start_rescue": ~(result["all_converged"][0] & result["all_valid"][0])
                                  & result["all_converged"][selected, rows] & valid,
        "bound_ADE_BDE": np.isclose(np.abs(gaze), 12.5, atol=1e-7, rtol=0),
        "outside_sampled_12p4": np.any(np.abs(gaze) > 12.4 + 1e-10, axis=1),
    })
    if np.any(result["primary_converged"] & (selected != 0)):
        raise AssertionError("Converged raw-mean primary was replaced")
    if np.any(result["valid"] & (np.max(np.abs(result["q_mm"] -
        (observed.mean(axis=1)-shape.evaluate(gaze).mean(axis=1)-mean_p1+bias.predict(gaze))), axis=1) > 1e-10)):
        raise AssertionError("Both-center q sign/identity failed")
    return result, calls


def gaze_metrics(estimate, truth):
    scored = metrics(estimate, truth)
    error = np.asarray(estimate) - np.asarray(truth)
    finite = np.isfinite(error).all(axis=1)
    if finite.any():
        for axis, name in enumerate(("ADE", "BDE")):
            scored[name]["median_abs_deg"] = float(np.median(np.abs(error[finite, axis])))
    return scored


def telemetry(result, mask):
    selected = result["selected_start"][mask]
    return {"attempted": int(mask.sum()),
            "primary_converged": int(result["primary_converged"][mask].sum()),
            "selected_converged": int(result["converged"][mask].sum()),
            "nonconverged": int((~result["converged"][mask]).sum()),
            "invalid": int((~result["valid"][mask]).sum()),
            "alternate_rescue": int(result["alternate_start_rescue"][mask].sum()),
            "converged_primary_replaced": int(np.sum(result["primary_converged"][mask] & (selected != 0))),
            "selected_seed_counts": np.bincount(selected, minlength=3).tolist(),
            "bound_ADE_BDE": result["bound_ADE_BDE"][mask].sum(axis=0).tolist(),
            "unsampled_12p4_to_12p5_strip": int(result["outside_sampled_12p4"][mask].sum()),
            "raw_initializer_bound_pressure": int(np.any(np.abs(result["raw_initial_gaze_deg"][mask]) > 12.5, axis=1).sum()),
            "all_start_cycle_events": int(result["all_cycle_count"][:, mask].sum()),
            "all_start_growth_events": int(result["all_growth_count"][:, mask].sum()),
            "max_evaluations_per_start": int(result["all_iterations"][:, mask].max())}


def score(result, truth, c4_true, c1_true, masks, *, oracle=None):
    stages = {"raw_unbounded": result["raw_initial_gaze_deg"],
              "raw_bounded": result["bounded_initial_gaze_deg"],
              "one_correction_unbounded": result["first_corrected_raw_gaze_deg"],
              "one_correction_bounded": result["first_corrected_gaze_deg"],
              "final_primary": result["primary_gaze_deg"], "final_selected": result["gaze_deg"]}
    if oracle is not None:
        stages["oracle_true_both_centers_scoring_only"] = oracle
    groups = {}
    for name, mask in masks.items():
        row = {stage: gaze_metrics(values[mask], truth[mask]) for stage, values in stages.items()}
        row["P4_center_raw_grid_mean"] = center_metrics(result["raw_p4_center_mm"][mask], c4_true[mask])
        row["P4_center_first"] = center_metrics(result["first_p4_center_mm"][mask], c4_true[mask])
        row["P4_center_final"] = center_metrics(result["p4_center_mm"][mask], c4_true[mask])
        row["P1_center_raw_mean"] = center_metrics(result["raw_p1_center_mm"][mask], c1_true[mask])
        row["P1_center_first"] = center_metrics(result["first_p1_center_mm"][mask], c1_true[mask])
        row["P1_center_final"] = center_metrics(result["p1_center_mm"][mask], c1_true[mask])
        change = result["gaze_deg"][mask] - result["bounded_initial_gaze_deg"][mask]
        row["gaze_correction_from_bounded_raw"] = gaze_metrics(change, np.zeros_like(change))
        row["P4_center_correction_from_raw"] = center_metrics(result["p4_center_mm"][mask], result["raw_p4_center_mm"][mask])
        row["P1_center_correction_from_raw"] = center_metrics(result["p1_center_mm"][mask], result["raw_p1_center_mm"][mask])
        row["telemetry"] = telemetry(result, mask)
        groups[name] = row
    return groups


def preflight_check():
    checks = synthetic_check()
    converged = np.array([[True, False, False], [True, True, False], [True, True, False]])
    valid = np.ones_like(converged)
    sse = np.array([[10., 10., 9.], [1., 2., 4.], [.1, 1., 1.]])
    residual = np.array([[0., .1, .3], [0., 0., .2], [0., 0., .1]])
    if not np.array_equal(choose_roots(converged, valid, residual, sse), [0, 2, 2]):
        raise AssertionError("Converged primary must survive a lower-SSE alternate")
    checks["primary_preserved_despite_lower_alternate_SSE"] = True
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--calibration-dir", type=Path, default=CAL)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    checks = preflight_check()
    print(json.dumps({"preflight": checks}), flush=True)
    if args.check_only:
        return
    calibration_path = args.calibration_dir / "summary.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration["total_fitted_scalar_values"] != 41 or not calibration["source_A0_only"]:
        raise RuntimeError("Expected independently calibrated A=0 41-value pipeline")
    obs_path = ROOT / "data/p4_p1_a0_only/calibration/a0_observations.npz"
    paired_path = ROOT / "data/p4_p1_both_centers/calibration/paired_data.npz"
    for path, sha in ((obs_path, calibration["A0_observations_sha256"]),
                      (paired_path, calibration["P1_paired_data_sha256"])):
        if digest(path) != sha:
            raise RuntimeError(f"Frozen physical data changed: {path}")
    with np.load(obs_path) as obs:
        rows = np.flatnonzero(obs["holdout25"])
        gaze_truth = obs["gaze_deg"][rows].copy()
        observed = obs["absolute_p4_mm"][rows].copy()
        mean_p1 = obs["mean_p1_absolute_mm"][rows].copy()
        ids = obs["source_indices"][rows].copy()
    excluded = np.all(np.isclose(gaze_truth, EXCLUDED_GAZE, atol=1e-9, rtol=0), axis=1)
    primary = ~excluded
    if (len(rows) != 3944 or excluded.sum() != 1 or primary.sum() != 3943
            or not np.array_equal(ids[excluded], [7554])):
        raise AssertionError("Exact A=0 matched holdout/exclusion identities differ")
    with np.load(paired_path) as paired:
        p1_gaze = paired["p1_gaze_deg"].copy()
        p1_labeled = paired["p1_labeled_center_mm"].copy()
        p1_mean = paired["p1_mean_mm"].copy()
    lookup = {tuple(np.round(g, 8)): i for i, g in enumerate(p1_gaze)}
    p1_rows = np.array([lookup[tuple(np.round(g, 8))] for g in gaze_truth])
    c1_true = p1_labeled[p1_rows]
    c4_true = observed[:, 4]
    if np.max(np.abs(p1_mean[p1_rows] - mean_p1)) > 1e-10:
        raise RuntimeError("P1 mean on runtime rows does not match the raw nine-ray P1 grid")
    shape = GazeOnlyShape.load(calibration["P4_shape"]["path"])
    bias = P1CentroidBias.load(calibration["P1_bias"]["path"])
    cubic = BivariatePolynomial.load(calibration["gaze_C20"]["path"])
    for key in ("P4_shape", "P1_bias", "gaze_C20"):
        info = calibration[key]
        if digest(info["path"]) != info["sha256"]:
            raise RuntimeError(f"Frozen new component changed: {key}")
    if shape.spatial_terms != 12 or shape.order != "PF" or bias.coefficient_count != 9 or cubic.coefficient_count != 20:
        raise RuntimeError("Expected only compact PF12+B3+C20")
    old_cal_path = OLD_CAL / "summary.json"
    old_cal = json.loads(old_cal_path.read_text(encoding="utf-8"))
    old_map_path = Path(old_cal["maps"]["C20"]["path"])
    if digest(old_map_path) != old_cal["maps"]["C20"]["sha256"]:
        raise RuntimeError("Old A=0 cubic changed")
    old_map = BivariatePolynomial.load(old_map_path)
    with np.load(OLD_RUNTIME) as prior:
        if (not np.array_equal(prior["source_indices"], ids)
                or not np.array_equal(prior["truth_gaze_deg"], gaze_truth)
                or not np.array_equal(prior["primary_user_exclusion_mask"], primary)):
            raise RuntimeError("Old PF12+C20 runtime does not match these exact holdout rows")
        old = {key: prior[key].copy() for key in prior.files if key not in ("source_indices", "truth_gaze_deg")}
    old_raw_delta = float(np.max(np.abs(old_map.predict(observed.mean(axis=1)-mean_p1)
                                        -old["raw_initial_gaze_deg"])))
    if old_raw_delta > 1e-10:
        raise RuntimeError("Old baseline was not calibrated on P4 center minus P1 mean")
    old["p4_center_mm"] = old["center_mm"]
    old["p1_center_mm"] = mean_p1
    old["raw_p4_center_mm"] = observed.mean(axis=1)
    old["raw_p1_center_mm"] = mean_p1
    old["first_p4_center_mm"] = observed.mean(axis=1) - shape.evaluate(old["bounded_initial_gaze_deg"]).mean(axis=1)
    old["first_p1_center_mm"] = mean_p1
    masks = {"primary_user_exclusion_3943": primary,
             "unfiltered_diagnostic_3944": np.ones(len(rows), bool),
             "excluded_one_state_diagnostic": excluded}

    begun = time.perf_counter()
    new, calls = infer(observed, mean_p1, shape, bias, cubic)
    wall = time.perf_counter() - begun
    oracle = cubic.predict(c4_true - c1_true)  # scoring only, after inference
    new_scores = score(new, gaze_truth, c4_true, c1_true, masks, oracle=oracle)
    old_scores = score(old, gaze_truth, c4_true, c1_true, masks)
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(f"Fresh output directory required: {out}")
    out.mkdir(parents=True)
    np.savez_compressed(out / "runtime_PF12_B3_C20.npz", **new,
                        source_indices=ids, a0_observation_rows=rows, truth_gaze_deg=gaze_truth,
                        true_p4_labeled_center_mm=c4_true, true_p1_labeled_center_mm=c1_true,
                        primary_user_exclusion_mask=primary, user_excluded_mask=excluded,
                        gaze_error_deg=new["gaze_deg"]-gaze_truth,
                        p4_center_error_mm=new["p4_center_mm"]-c4_true,
                        p1_center_error_mm=new["p1_center_mm"]-c1_true,
                        oracle_true_both_center_gaze_deg=oracle)
    summary = {"experiment": "A=0 both-center correction versus frozen A=0 P1-mean baseline",
        "new_pipeline": "PF12+B3+C20", "new_fitted_scalar_values": 41,
        "baseline_pipeline": "PF12+C20", "baseline_fitted_scalar_values": 32,
        "baseline_gaze_input": "P4 labeled center minus measured P1 mean",
        "new_gaze_input": "P4 corrected labeled center minus P1 corrected labeled center",
        "calibration_poses": 25, "primary_holdout": 3943,
        "diagnostic_holdout": 3944, "excluded_gaze_deg": EXCLUDED_GAZE.tolist(),
        "excluded_source_id": 7554, "no_accommodation_input_or_optimizer": True,
        "runtime_bound_deg": 12.5, "observed_support_deg": 12.4,
        "new_runtime_seconds": wall, "new_model_calls": calls,
        "scores": {"new": new_scores, "baseline": old_scores},
        "new_calibration_sha256": digest(calibration_path),
        "old_calibration_sha256": digest(old_cal_path),
        "old_runtime_sha256": digest(OLD_RUNTIME),
        "old_map_sha256": digest(old_map_path), "old_raw_input_reconstruction_max_deg": old_raw_delta,
        "A0_observations_sha256": digest(obs_path),
        "P1_paired_data_sha256": digest(paired_path),
        "new_runtime_sha256": digest(out / "runtime_PF12_B3_C20.npz"),
        "solver_sha256": digest(ROOT / "Script/p4_fixed_a0_both_centers.py"),
        "driver_sha256": digest(__file__), "preflight": checks,
        "noise_scope": "Noiseless simulation/model mismatch only"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    primary_new = new_scores["primary_user_exclusion_3943"]["final_selected"]["radial"]
    primary_old = old_scores["primary_user_exclusion_3943"]["final_selected"]["radial"]
    print(json.dumps({"output": str(out), "new_radial": primary_new,
                      "baseline_radial": primary_old,
                      "new_telemetry": new_scores["primary_user_exclusion_3943"]["telemetry"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
