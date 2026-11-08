"""Infer A=0 gaze independently from P1 and P4 with equal-form B3/G20 maps."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from evaluate_p4_p1_a0_only import center_metrics, metrics
from p1_centroid_bias import P1CentroidBias
from p4_a0_bivariate_polynomial import BivariatePolynomial
from p4_fixed_a0_gaze import solve_fixed_a0

ROOT = Path(__file__).resolve().parents[1]
CAL = ROOT / "data/p1_p4_a0_independent_gaze/calibration"
A0_OBS = ROOT / "data/p4_p1_a0_only/calibration/a0_observations.npz"
P1_PAIRED = ROOT / "data/p4_p1_both_centers/calibration/paired_data.npz"
HISTORICAL = ROOT / "data/p4_p1_a0_bivariate/runtime/runtime_PF12_C20.npz"
OUT = ROOT / "data/p1_p4_a0_independent_gaze/runtime"
EXCLUDED_GAZE = np.array([12., 9.6])
SAVE_KEYS = ("gaze_deg", "center_mm", "raw_center_mm", "first_center_mm", "bias_mm",
    "converged", "valid", "selected_start", "fixedpoint_residual_deg",
    "raw_initial_gaze_deg", "bounded_initial_gaze_deg", "first_corrected_raw_gaze_deg",
    "first_corrected_gaze_deg", "primary_gaze_deg", "primary_center_mm", "primary_converged",
    "alternate_start_rescue", "bound_ADE_BDE", "outside_sampled_12p4",
    "all_converged", "all_valid", "all_invalid", "all_iterations", "all_alpha",
    "all_cycle_count", "all_growth_count", "all_fixedpoint_residual_deg",
    "all_ever_cubic_bound_pressure", "all_initial_gaze_deg")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose_by_fixedpoint(converged, valid, residual):
    """Prefer the raw-mean primary; never use centered-grid SSE to rank roots."""
    converged, valid, residual = np.asarray(converged, bool), np.asarray(valid, bool), np.asarray(residual)
    selected = np.zeros(converged.shape[1], dtype=int)
    for row in range(len(selected)):
        if converged[0, row] and valid[0, row]:
            continue
        alternatives = np.flatnonzero(converged[1:, row] & valid[1:, row]) + 1
        if len(alternatives):
            selected[row] = alternatives[np.argmin(residual[alternatives, row])]
        else:
            finite = np.flatnonzero(valid[:, row] & np.isfinite(residual[:, row]))
            if len(finite):
                selected[row] = finite[np.argmin(residual[finite, row])]
    return selected


def infer_channel(observed_grid_mm, bias_model, gaze_model):
    """Centroid-only adapter: the nine broadcast values are not a ray prediction."""
    grid = np.asarray(observed_grid_mm, float)
    if grid.ndim != 3 or grid.shape[1:] != (9, 2) or not np.isfinite(grid).all():
        raise ValueError("Expected finite observed nine-point grid")
    mean = grid.mean(axis=1)
    count = len(mean)
    # The Picard core consumes a nine-position array to compute its mean. Pass
    # the measured mean repeated nine times, so its grid-shape SSE is constant
    # and carries no hidden full-field model or root-selection information.
    centroid_observation = np.broadcast_to(mean[:, None, :], (count, 9, 2))
    zero_reference = np.zeros((count, 2))
    calls = {"centroid_bias": 0, "gaze_cubic": 0}

    def centroid_callback(gaze):
        calls["centroid_bias"] += 1
        bias = bias_model.predict(gaze)
        return np.broadcast_to(bias[:, None, :], (len(gaze), 9, 2)), np.isfinite(bias).all(axis=1)

    def inverse_center(center):
        calls["gaze_cubic"] += 1
        return gaze_model.predict(center)

    core = solve_fixed_a0(centroid_observation, zero_reference,
                          centroid_callback, inverse_center, keep_history=False)
    selected = choose_by_fixedpoint(core["all_converged"], core["all_valid"],
                                    core["all_fixedpoint_residual_deg"])
    rows = np.arange(count)
    valid = core["all_valid"][selected, rows]
    gaze = np.where(valid[:, None], core["all_gaze_deg"][selected, rows], np.nan)
    center = mean - bias_model.predict(gaze)
    initial = core["bounded_initial_gaze_deg"]
    first_center = mean - bias_model.predict(initial)
    primary_valid = core["all_valid"][0]
    primary_gaze = np.where(primary_valid[:, None], core["all_gaze_deg"][0], np.nan)
    result = {"gaze_deg": gaze, "center_mm": center,
        "raw_center_mm": mean, "first_center_mm": first_center,
        "bias_mm": mean-center,
        "converged": core["all_converged"][selected, rows] & valid,
        "valid": valid, "selected_start": selected,
        "fixedpoint_residual_deg": core["all_fixedpoint_residual_deg"][selected, rows],
        "raw_initial_gaze_deg": core["raw_initial_gaze_deg"],
        "bounded_initial_gaze_deg": initial,
        "first_corrected_raw_gaze_deg": core["first_corrected_raw_gaze_deg"],
        "first_corrected_gaze_deg": core["first_corrected_gaze_deg"],
        "primary_gaze_deg": primary_gaze,
        "primary_center_mm": mean-bias_model.predict(primary_gaze),
        "primary_converged": core["all_converged"][0] & primary_valid,
        "alternate_start_rescue": ~(core["all_converged"][0] & primary_valid)
                                  & core["all_converged"][selected, rows] & valid,
        "bound_ADE_BDE": np.isclose(np.abs(gaze), 12.5, atol=1e-7, rtol=0),
        "outside_sampled_12p4": np.any(np.abs(gaze) > 12.4+1e-10, axis=1),
        "all_converged": core["all_converged"], "all_valid": core["all_valid"],
        "all_invalid": core["all_invalid"], "all_iterations": core["all_iterations"],
        "all_alpha": core["all_alpha"], "all_cycle_count": core["all_cycle_count"],
        "all_growth_count": core["all_growth_count"],
        "all_fixedpoint_residual_deg": core["all_fixedpoint_residual_deg"],
        "all_ever_cubic_bound_pressure": core["all_ever_cubic_bound_pressure"],
        "all_initial_gaze_deg": core["all_initial_gaze_deg"]}
    if np.any(result["primary_converged"] & (selected != 0)):
        raise AssertionError("A converged raw-mean primary was replaced")
    good = result["valid"]
    if np.any(good):
        consistency = np.max(np.abs(gaze_model.predict(center[good])-gaze[good]), axis=1)
        if np.any(consistency > 1e-4+1e-9):
            # Bound projections can make raw G(center) exceed the returned gaze.
            bounded = np.clip(gaze_model.predict(center[good]), -12.5, 12.5)
            if np.any(np.max(np.abs(bounded-gaze[good]), axis=1) > 1e-4+1e-9):
                raise AssertionError("Centroid-only fixed-point identity failed")
    return result, calls


def gaze_metrics(prediction, truth):
    scored = metrics(prediction, truth)
    error = np.asarray(prediction)-np.asarray(truth)
    finite = np.isfinite(error).all(axis=1)
    if finite.any():
        for axis, label in enumerate(("ADE", "BDE")):
            scored[label]["median_abs_deg"] = float(np.median(np.abs(error[finite, axis])))
    return scored


def telemetry(result, mask):
    chosen = result["selected_start"][mask]
    return {"attempted": int(mask.sum()),
        "primary_converged": int(result["primary_converged"][mask].sum()),
        "selected_converged": int(result["converged"][mask].sum()),
        "nonconverged": int((~result["converged"][mask]).sum()),
        "invalid": int((~result["valid"][mask]).sum()),
        "alternate_rescue": int(result["alternate_start_rescue"][mask].sum()),
        "converged_primary_replaced": int(np.sum(result["primary_converged"][mask] & (chosen != 0))),
        "selected_seed_counts": np.bincount(chosen, minlength=3).tolist(),
        "bound_ADE_BDE": result["bound_ADE_BDE"][mask].sum(axis=0).tolist(),
        "unsampled_12p4_to_12p5_strip": int(result["outside_sampled_12p4"][mask].sum()),
        "raw_initializer_bound_pressure": int(np.any(np.abs(result["raw_initial_gaze_deg"][mask]) > 12.5, axis=1).sum()),
        "all_start_cycle_events": int(result["all_cycle_count"][:, mask].sum()),
        "all_start_growth_events": int(result["all_growth_count"][:, mask].sum()),
        "max_evaluations_per_start": int(result["all_iterations"][:, mask].max())}


def score_channel(result, truth, labeled_center, masks, oracle):
    stages = {"raw_unbounded": result["raw_initial_gaze_deg"],
        "raw_bounded": result["bounded_initial_gaze_deg"],
        "one_correction_unbounded": result["first_corrected_raw_gaze_deg"],
        "one_correction_bounded": result["first_corrected_gaze_deg"],
        "final_primary": result["primary_gaze_deg"], "final_selected": result["gaze_deg"],
        "oracle_true_labeled_center_scoring_only": oracle}
    scored = {}
    for name, mask in masks.items():
        row = {stage: gaze_metrics(value[mask], truth[mask]) for stage, value in stages.items()}
        row["center_raw_grid_mean"] = center_metrics(result["raw_center_mm"][mask], labeled_center[mask])
        row["center_one_correction"] = center_metrics(result["first_center_mm"][mask], labeled_center[mask])
        row["center_final"] = center_metrics(result["center_mm"][mask], labeled_center[mask])
        row["center_correction_from_raw"] = center_metrics(result["center_mm"][mask], result["raw_center_mm"][mask])
        change = result["gaze_deg"][mask]-result["bounded_initial_gaze_deg"][mask]
        row["gaze_correction_from_raw_bounded"] = gaze_metrics(change, np.zeros_like(change))
        row["telemetry"] = telemetry(result, mask)
        scored[name] = row
    return scored


def agreement_metrics(p1, p4, mask):
    delta = p1[mask]-p4[mask]
    finite = np.isfinite(delta).all(axis=1)
    radial = np.linalg.norm(delta[finite], axis=1)
    return {"attempted": int(mask.sum()), "finite": int(finite.sum()),
            "bias_ADE_BDE_deg": delta[finite].mean(axis=0).tolist(),
            "radial_RMS_deg": float(np.sqrt(np.mean(radial*radial))),
            "radial_median_deg": float(np.median(radial)),
            "radial_P95_deg": float(np.percentile(radial, 95)),
            "radial_max_deg": float(radial.max())}


def synthetic_check():
    truth = np.array([[3.7, -8.2], [-10.6, 2.9], [12.5, -12.5], [0., 0.]])
    field = np.array([(x, y) for x in (-1., 0., 1.) for y in (-1., 0., 1.)], float)
    answers = {}
    from p4_a0_bivariate_polynomial import BivariatePolynomial
    for name, offset, slope in (("P1", 1.7, 5.), ("P4", 0., 6.)):
        coefficients = np.zeros((10, 2))
        coefficients[0] = [.1, -.2]
        coefficients[1, 1] = -slope
        coefficients[2, 0] = slope
        coefficients[4] = [.012, -.009]
        G = BivariatePolynomial(3, np.array([0., offset]), np.ones(2), coefficients)
        B = P1CentroidBias(3, np.array([0., -.05 if name == "P1" else 0.]),
                           np.array([.003, -.002, .001, -.0005]),
                           np.array([.002, .001, -.003, .0004, -.0003]))
        center = np.column_stack((-truth[:, 1]/slope, offset+truth[:, 0]/slope))
        for _ in range(12):
            center -= np.linalg.solve(G.jacobian(center), (G.predict(center)-truth)[..., None])[..., 0]
        bias = B.predict(truth)
        grid = center[:, None, :]+field[None, :, :]*.2
        grid[:, [0, 1, 2, 3, 5, 6, 7, 8]] += bias[:, None, :]*9/8
        result, _ = infer_channel(grid, B, G)
        error = float(np.max(np.abs(result["gaze_deg"]-truth)))
        center_error = float(np.max(np.abs(result["center_mm"]-center)))
        if not result["primary_converged"].all() or error > 1e-4 or center_error > 1e-5:
            raise AssertionError((name, error, center_error))
        # Centered shape changes with the same mean cannot change inference.
        changed = grid.copy()
        changed[:, 0] += [.04, -.03]
        changed[:, 8] -= [.04, -.03]
        again, _ = infer_channel(changed, B, G)
        shape_invariance = float(np.max(np.abs(again["gaze_deg"]-result["gaze_deg"])))
        if shape_invariance > 1e-10:
            raise AssertionError("Centroid-only callback depended on nine-ray shape")
        answers[name] = {"gaze_max_component_error_deg": error,
                         "center_max_component_error_mm": center_error,
                         "shape_invariance_error_deg": shape_invariance,
                         "nonzero_bias_max_mm": float(np.max(np.abs(bias)))}
    converged = np.array([[True, False, False], [True, True, False], [True, True, False]])
    valid = np.ones_like(converged)
    residual = np.array([[10., 10., 9.], [1., 2., 4.], [.1, 1., 1.]])
    if not np.array_equal(choose_by_fixedpoint(converged, valid, residual), [0, 2, 2]):
        raise AssertionError("Primary-preserving fixed-point selector failed")
    answers["primary_preserved"] = True
    return answers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--calibration-dir", type=Path, default=CAL)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    checks = synthetic_check()
    print(json.dumps({"preflight": checks}), flush=True)
    if args.check_only:
        return
    calibration_path = args.calibration_dir / "summary.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if (not calibration["A0_only"] or calibration["total_fitted_scalar_values"] != 58
            or not calibration["P1_and_P4_coefficients_and_normalizers_independent"]):
        raise RuntimeError("Expected the separate 58-value A=0 calibration")
    for path, expected in ((A0_OBS, calibration["A0_observations_sha256"]),
                           (P1_PAIRED, calibration["P1_paired_data_sha256"])):
        if digest(path) != expected:
            raise RuntimeError(f"Raw observation source changed: {path}")
    with np.load(A0_OBS) as obs:
        rows = np.flatnonzero(obs["holdout25"])
        truth = obs["gaze_deg"][rows].copy()
        p4_grid = obs["absolute_p4_mm"][rows].copy()
        paired_p1_mean = obs["mean_p1_absolute_mm"][rows].copy()
        ids = obs["source_indices"][rows].copy()
    excluded = np.all(np.isclose(truth, EXCLUDED_GAZE, atol=1e-9, rtol=0), axis=1)
    primary = ~excluded
    if (len(rows) != 3944 or excluded.sum() != 1 or primary.sum() != 3943
            or not np.array_equal(ids[excluded], [7554])):
        raise AssertionError("Matched A=0 primary/diagnostic holdout changed")
    with np.load(P1_PAIRED) as paired:
        p1_gaze = paired["p1_gaze_deg"].copy()
        p1_grid_all = paired["p1_absolute_mm"].copy()
    lookup = {tuple(np.round(g, 8)): i for i, g in enumerate(p1_gaze)}
    joined = np.array([lookup[tuple(np.round(g, 8))] for g in truth])
    p1_grid = p1_grid_all[joined]
    if np.max(np.abs(p1_grid.mean(axis=1)-paired_p1_mean)) > 1e-10:
        raise RuntimeError("Independent P1 observed grid mean differs from paired A=0 data")
    masks = {"primary_user_exclusion_3943": primary,
             "unfiltered_diagnostic_3944": np.ones(len(rows), bool),
             "excluded_one_state_diagnostic": excluded}
    grid_by_name = {"P1": p1_grid, "P4": p4_grid}
    results, scores, calls, seconds, oracles = {}, {}, {}, {}, {}
    for name, grid in grid_by_name.items():
        info = calibration["channels"][name]
        bpath, gpath = Path(info["bias"]["path"]), Path(info["gaze"]["path"])
        if digest(bpath) != info["bias"]["sha256"] or digest(gpath) != info["gaze"]["sha256"]:
            raise RuntimeError(f"Frozen {name} B3/G20 component changed")
        B, G = P1CentroidBias.load(bpath), BivariatePolynomial.load(gpath)
        if B.coefficient_count != 9 or G.coefficient_count != 20:
            raise RuntimeError(f"Expected only {name} B3/G20")
        started = time.perf_counter()
        result, call_count = infer_channel(grid, B, G)
        seconds[name] = time.perf_counter()-started
        # True labeled centers and gaze labels first enter after inference.
        labeled = grid[:, 4]
        oracle = G.predict(labeled)
        results[name], calls[name], oracles[name] = result, call_count, oracle
        scores[name] = score_channel(result, truth, labeled, masks, oracle)
    # Historical pairwise map is read only and cannot enter either new channel.
    with np.load(HISTORICAL) as old:
        if (not np.array_equal(old["source_indices"], ids)
                or not np.array_equal(old["truth_gaze_deg"], truth)
                or not np.array_equal(old["primary_user_exclusion_mask"], primary)):
            raise RuntimeError("Historical pairwise PF12+C20 uses different heldout labels")
        historical_gaze = old["gaze_deg"].copy()
    historical_scores = {name: gaze_metrics(historical_gaze[mask], truth[mask])
                         for name, mask in masks.items()}
    agreement = {name: agreement_metrics(results["P1"]["gaze_deg"], results["P4"]["gaze_deg"], mask)
                 for name, mask in masks.items()}
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(f"Fresh runtime directory required: {out}")
    out.mkdir(parents=True)
    saved = {f"{name.lower()}_{key}": value for name, result in results.items()
             for key, value in result.items() if key in SAVE_KEYS}
    np.savez_compressed(out / "runtime_independent_P1_P4.npz", **saved,
        source_indices=ids, a0_observation_rows=rows, truth_gaze_deg=truth,
        p1_true_labeled_center_mm=p1_grid[:, 4], p4_true_labeled_center_mm=p4_grid[:, 4],
        primary_user_exclusion_mask=primary, excluded_state_mask=excluded,
        p1_oracle_true_center_gaze_deg=oracles["P1"],
        p4_oracle_true_center_gaze_deg=oracles["P4"],
        historical_pairwise_gaze_deg=historical_gaze)
    summary = {"experiment": "Independent A=0 P1 and P4 gaze estimates from equal-form B3/G20 maps",
        "no_automatic_fusion_or_pairwise_gaze_input": True,
        "P1_and_P4_models_fitted_separately": True,
        "total_fitted_scalar_values": 58, "per_channel_fitted_scalar_values": 29,
        "calibration_poses_per_channel": 25, "primary_holdout": 3943,
        "inclusive_diagnostic_holdout": 3944,
        "excluded_gaze_deg": EXCLUDED_GAZE.tolist(), "excluded_source_id": 7554,
        "runtime_input_each_channel": "own observed nine-point grid mean; no true gaze/center",
        "centroid_callback_note": "Nine repeated callback values carry only predicted centroid bias; no nine-ray shape model or shape-SSE root ranking",
        "runtime_bound_deg": 12.5, "observed_support_deg": 12.4,
        "channels": {name: {"runtime_seconds": seconds[name], "model_calls": calls[name],
                            "scores": scores[name]} for name in ("P1", "P4")},
        "P1_P4_gaze_agreement_diagnostic_only": agreement,
        "historical_pairwise_PF12_C20_32_values": {"source_path": str(HISTORICAL),
            "source_sha256": digest(HISTORICAL), "scores": historical_scores},
        "calibration_summary_sha256": digest(calibration_path),
        "A0_observations_sha256": digest(A0_OBS), "P1_paired_data_sha256": digest(P1_PAIRED),
        "runtime_artifact_sha256": digest(out / "runtime_independent_P1_P4.npz"),
        "driver_sha256": digest(__file__), "Picard_core_sha256": digest(ROOT / "Script/p4_fixed_a0_gaze.py"),
        "preflight": checks, "no_accommodation_input_or_profile": True,
        "noise_scope": "Noiseless simulation/model mismatch only"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out),
        "primary_radial_RMS_deg": {name: scores[name]["primary_user_exclusion_3943"]["final_selected"]["radial"]["RMS_deg"]
                                    for name in ("P1", "P4")},
        "primary_agreement_radial_RMS_deg": agreement["primary_user_exclusion_3943"]["radial_RMS_deg"],
        "primary_telemetry": {name: scores[name]["primary_user_exclusion_3943"]["telemetry"]
                              for name in ("P1", "P4")}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
