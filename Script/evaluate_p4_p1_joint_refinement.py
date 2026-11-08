"""Final observed-data joint refinement after sparse P4/P1 calibration.

This is a separate experiment; the earlier alternating-pilot artifacts remain
unchanged. Truth is used only for coverage selection and post-estimation scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "6"

import numpy as np
from threadpoolctl import threadpool_limits

from p4_sparse_model import SparseModel
from p4_sparse_model_69 import SparseModel69
from p4_centroid_iteration import CenterSignalMap
from p4_joint_observed_solver import solve_observed, FIXED_STARTS, GRADIENT_TOL_MM
from fit_p4_sparse_calibration import calibration_masks, scalar_metrics, write_json, log
from evaluate_p4_p1_centroid_iteration import (pair_inputs, gaze_metrics, pilot_indices,
    stratified_indices, P1_PATH, SOURCE_FIT)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT/"data/p4_p1_joint_refinement"
DEFAULT_SIGNALS = ROOT/"data/p4_p1_sparse_centroid_iteration_pilot"
DEFAULT_MODEL75 = ROOT/"data/p4_sparse_calibration_69/cal75_projective_first_69.npz"
HISTORY_KEYS = ("state_history", "corrected_center_history_mm", "predicted_center_history_mm",
                "raw_proposal_history", "sse_history_mm2", "accepted_history")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_summary(values):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"finite": 0, "nonfinite": int(values.size)}
    return {"finite": int(len(finite)), "nonfinite": int(values.size-len(finite)),
            "median": float(np.median(finite)), "p95": float(np.percentile(finite, 95)),
            "max": float(np.max(finite))}


def score(result, truth, true_center, mask):
    take = np.asarray(mask)
    error = result["estimated_state"][take]-truth[take]
    raw_initial_gaze = result["all_seed_initial_states"][take, 1, 1:]-truth[take, 1:]
    corrected_seed_gaze = result["all_seed_centroid_corrected_states"][take, 1, 1:]-truth[take, 1:]
    corrected_center_error = result["corrected_center_absolute_mm"][take]-true_center[take]
    predicted_center_error = result["predicted_center_absolute_mm"][take]-true_center[take]
    convergence = result["converged"][take]
    return {"states": int(np.sum(take)), "A_d": scalar_metrics(error[:, 0]),
        "fixed_A2_raw_mean_initial_gaze": gaze_metrics(raw_initial_gaze),
        "fixed_A2_one_centroid_correction_gaze": gaze_metrics(corrected_seed_gaze),
        "final_gaze": gaze_metrics(error[:, 1:]),
        "converged": int(np.sum(convergence)), "nonconverged": int(np.sum(~convergence)),
        "no_converged_valid_seed": int(np.sum(result["no_converged_valid_seed"][take])),
        "invalid_map": int(np.sum(~result["valid_map"][take])),
        "all_seed_nonfinite_objective": int(np.sum(result["all_seeds_nonfinite_objective"][take])),
        "rank_deficient": int(np.sum(result["normalized_jacobian_rank"][take] < 3)),
        "boundary_A_ADE_BDE": np.sum(result["boundary"][take], axis=0).tolist(),
        "median_iterations": float(np.median(result["iterations"][take])),
        "p95_iterations": float(np.percentile(result["iterations"][take], 95)),
        "normalized_gradient_mm": finite_summary(result["projected_normalized_gradient_mm"][take]),
        "normalized_jacobian_condition": finite_summary(result["normalized_jacobian_condition"][take]),
        "selected_minus_best_any_sse_mm2": finite_summary(result["selected_minus_best_any_sse_mm2"][take]),
        "distinct_converged_alternative_minima": int(np.sum(np.isfinite(result["distinct_converged_alternative_sse_gap_mm2"][take]))),
        "corrected_center_rms_um": float(np.sqrt(np.mean(np.sum(corrected_center_error**2, axis=1)))*1000.),
        "predicted_center_rms_um": float(np.sqrt(np.mean(np.sum(predicted_center_error**2, axis=1)))*1000.),
        "center_consistency_rms_um": float(np.sqrt(np.mean(np.sum(result["center_residual_mm"][take]**2, axis=1)))*1000.),
        "observed_vertex_rms_um": float(np.sqrt(np.mean(result["observable_joint_sse_mm2"][take]/9.))*1000.),
        "selected_seed_counts_A0_A2_A4": np.bincount(result["selected_seed_index"][take], minlength=3).tolist(),
        "converged_only_A_d": scalar_metrics(error[convergence, 0]) if np.any(convergence) else None}


def run(model, signal, y4, m1, indices, out, tag, chunk=1024):
    arrays = {}
    trace_dir = out/f"traces_{tag}"
    trace_dir.mkdir(exist_ok=True)
    started = time.perf_counter()
    for lower in range(0, len(indices), chunk):
        upper = min(lower+chunk, len(indices))
        ids = indices[lower:upper]
        result = solve_observed(model, signal, y4[ids], m1[ids], record_history=True)
        histories = {key: result.pop(key) for key in HISTORY_KEYS}
        np.savez_compressed(trace_dir/f"frames_{lower:06d}_{upper-1:06d}.npz",
                            selection_indices=ids, **histories)
        for key, value in result.items():
            if key not in arrays:
                arrays[key] = np.empty((len(indices),)+value.shape[1:], dtype=value.dtype)
            arrays[key][lower:upper] = value
        if upper == len(indices) or upper % 4096 == 0:
            elapsed = time.perf_counter()-started
            log(f"joint {tag}: {upper:,}/{len(indices):,}; elapsed {elapsed:.1f}s; projected {elapsed*len(indices)/upper:.1f}s")
    return arrays, time.perf_counter()-started


def write_report(out, summary):
    lines = ["# Observed-data P4−P1 joint refinement", "",
        "The frozen sparse shape and center-signal calibrations are unchanged. This separate solver uses one coherent observed-coordinate residual: `R_j=y4_j−mean(P1)−D(A,g)−F_j(A,g)` over all nine fields. Its SSE equals the mean-centered shape SSE plus nine times the centroid-signal SSE. It is an unweighted empirical objective, not a noise-calibrated likelihood.", "",
        "Three fixed blind A starts (0, 2, 4 D) seed gaze from the measured mean difference and the same calibrated D map, then apply one model centroid correction. Bounded joint LM updates A/ADE/BDE together. No true state or true center enters initialization, objective, convergence, or candidate selection. Only projected/KKT stationarity counts as convergence. The lowest-cost converged valid seed is selected; if none converges, the best finite estimate is retained and flagged.", "",
        summary["provenance"]["frame_assumption"]+". No heldout registration transform is fitted.", "",
        "| Calibration states | Shape values | Center/gaze values | Deployed total | Common heldout N | A RMSE (D) | A P95 abs (D) | Within ±0.1 D | Gaze parameter RMSE (deg) | Converged |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for size, row in summary["models"].items():
        if "runtime" not in row:
            continue
        q = row["runtime"]["metrics"]["common_heldout"]
        a = q["A_d"]
        lines.append(f"| {size} | {row['shape_coefficients']} | {row['signal_coefficients']} | {row['deployed_coefficients']} | {q['states']:,} | {a['rmse']:.7g} | {a['p95_abs']:.7g} | {100*a['within_0_1D_fraction_all_attempted']:.3f}% | {q['final_gaze']['paired_parameter_rmse_deg']:.7g} | {q['converged']:,} |")
    lines += ["", "Separate interior-A and endpoint-A, random-sample and complete-boundary results, initial-to-final gaze errors, map/rank/conditioning, every seed's costs/statuses, corrected and predicted centers, and selected-vs-best-any costs are in summary.json and NPZ files. Selected-seed state/center/objective histories are retained by chunk under traces_*.", "",
        "The ±0.1 D fraction is a comparison reference, not a pass/fail criterion. Source A spacing of 0.2 D does not itself establish 0.1 D resolution. Exact-supplied-gaze results are a separate controlled benchmark; the earlier alternating pilot remains an unchanged diagnostic. These are conditional simulation results, not a claim of physical frame registration or real sensor-noise accuracy."]
    (out/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-fit", type=Path, default=SOURCE_FIT)
    parser.add_argument("--signal-dir", type=Path, default=DEFAULT_SIGNALS)
    parser.add_argument("--model27", type=Path)
    parser.add_argument("--model75", type=Path, default=DEFAULT_MODEL75)
    parser.add_argument("--P1", type=Path, default=P1_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--chunk", type=int, default=1024)
    args = parser.parse_args()
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=6):
        states, y4, m1, centers, valid, provenance = pair_inputs(args.source_fit, args.P1)
        source_summary = json.loads((args.source_fit/"summary.json").read_text(encoding="utf-8"))
        path27 = args.model27 or args.source_fit/f"{source_summary['selected']['27']}.npz"
        paths = {27: path27, 75: args.model75}
        models, signals = {}, {}
        summary = {"provenance": provenance, "solver": {"fixed_A_starts_D": FIXED_STARTS.tolist(),
            "max_iterations": 80, "projected_gradient_tolerance_mm": GRADIENT_TOL_MM,
            "source_sha256": file_hash(ROOT/"Script/p4_joint_observed_solver.py"),
            "runner_sha256": file_hash(__file__), "shape_core_sha256": file_hash(ROOT/"Script/p4_sparse_model.py"),
            "shape69_core_sha256": file_hash(ROOT/"Script/p4_sparse_model_69.py")}, "models": {}}
        common = valid & ~calibration_masks(75)["calibration"]
        pilot_take = pilot_indices(states, common)
        projection = {}
        for size in (27, 75):
            with np.load(paths[size]) as stored:
                extended = len(stored["coefficients"]) == 66
            model = SparseModel69.load(paths[size]) if extended else SparseModel.load(paths[size])
            signal_path = args.signal_dir/f"center_signal_{size}.npz"
            signal = CenterSignalMap.load(signal_path)
            models[size], signals[size] = model, signal
            result, seconds = run(model, signal, y4, m1, pilot_take, out, f"pilot_{size}", chunk=512)
            np.savez_compressed(out/f"pilot_{size}.npz", selection_indices=pilot_take,
                                truth_state=states[pilot_take], **result)
            estimate = seconds*int(np.sum(common))/len(pilot_take)
            projection[size] = estimate
            summary["models"][str(size)] = {"shape_path": str(paths[size]), "shape_sha256": file_hash(paths[size]),
                "signal_path": str(signal_path), "signal_sha256": file_hash(signal_path),
                "shape_coefficients": model.parameter_count, "signal_coefficients": signal.parameter_count,
                "deployed_coefficients": model.parameter_count+signal.parameter_count,
                "pilot": {"states": 512, "wall_seconds": seconds,
                    "projected_full_common_heldout_seconds": estimate,
                    "metrics": score(result, states[pilot_take], centers[pilot_take], np.ones(512, bool))}}
            write_json(out/"summary.json", summary)
            write_report(out, summary)
            log(f"joint cal{size} pilot: {seconds:.3f}s; projected common heldout {estimate:.1f}s")
        if args.pilot_only:
            return
        full = max(projection.values()) <= 600.
        random_take = stratified_indices(states, common, 1000, 20261107)
        boundary = np.any(np.isclose(np.abs(states[:, 1:]), 20., atol=1e-9), axis=1)
        stress_take = np.flatnonzero(common & boundary)
        calibration_take = np.flatnonzero(valid & calibration_masks(75)["calibration"])
        selection = np.flatnonzero(valid) if full else np.unique(np.r_[random_take, stress_take, calibration_take])
        summary["scope"] = {"full_source": bool(full), "states": len(selection),
            "common_heldout_states_available": int(np.sum(common)),
            "random_common_heldout_states": len(random_take), "random_per_A": 1000, "random_seed": 20261107,
            "complete_boundary_stress_states": len(stress_take),
            "distribution_note": "Stratified random and complete boundary stress sets are scored separately; sampled maxima are not full-grid maxima."}
        for size in (27, 75):
            result, seconds = run(models[size], signals[size], y4, m1, selection, out, f"runtime_{size}", args.chunk)
            np.savez_compressed(out/f"runtime_{size}.npz", selection_indices=selection,
                truth_state=states[selection], true_center_absolute_mm=centers[selection],
                signed_state_error=result["estimated_state"]-states[selection], **result)
            masks = {name: mask[selection] for name, mask in calibration_masks(size).items()}
            masks["common_heldout"] = common[selection]
            masks["random_common_heldout"] = np.isin(selection, random_take)
            masks["complete_boundary_stress"] = np.isin(selection, stress_take)
            endpoints = np.isclose(states[selection, 0], 0.) | np.isclose(states[selection, 0], 4.)
            masks["common_heldout_interior_A"] = common[selection] & ~endpoints
            masks["common_heldout_endpoint_A"] = common[selection] & endpoints
            summary["models"][str(size)]["runtime"] = {"wall_seconds": seconds,
                "metrics": {name: score(result, states[selection], centers[selection], mask)
                            for name, mask in masks.items() if np.any(mask)},
                "per_A": {f"{value:.1f}": score(result, states[selection], centers[selection],
                    common[selection] & np.isclose(states[selection, 0], value)) for value in np.unique(states[:, 0])}}
            write_json(out/"summary.json", summary)
            write_report(out, summary)
        log(f"observed-data joint refinement complete: {out}")


if __name__ == "__main__":
    main()
