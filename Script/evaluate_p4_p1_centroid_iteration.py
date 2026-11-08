"""Sparse-calibrated, conditional simulation test of iterative P4-P1 inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "6"

import numpy as np
from threadpoolctl import threadpool_limits

from p4_sparse_model import SparseModel
from fit_p4_sparse_calibration import calibration_masks, scalar_metrics, write_json, log
from p4_centroid_iteration import fit_signal, select_signal_map, iterate_centroids

ROOT = Path(__file__).resolve().parents[1]
P1_PATH = ROOT/"data/distortion_grid_p1_biaxial_macro/distortion_grid.pkl"
P1_SHA = "1a57a8d850d021f3d23dda201f2fc47bd1aec6eb60860e42990be990cfe7f835"
SOURCE_FIT = ROOT/"data/p4_sparse_calibration"
DEFAULT_OUT = ROOT/"data/p4_p1_sparse_centroid_iteration"


def pair_inputs(source_dir, p1_path):
    with np.load(source_dir/"source_tensors.npz") as source:
        states, relative, centers = source["states"], source["relative_mm"], source["center_absolute_mm"]
        A, ADE, BDE = source["A"], source["ADE"], source["BDE"]
        p4_sha = str(source["grid_sha256"])
    raw = p1_path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != P1_SHA:
        raise RuntimeError("P1 pickle differs from the advisor-pinned audited input")
    p1 = pickle.loads(raw)
    assert len(p1) == 101*101*9
    a1, b1 = np.unique(p1["eye_rotation_deg"]), np.unique(p1["eye_rotation_x_deg"])
    assert np.allclose(a1, ADE, atol=1e-9, rtol=0) and np.allclose(b1, BDE, atol=1e-9, rtol=0)
    di, bi = np.searchsorted(a1, p1["eye_rotation_deg"]), np.searchsorted(b1, p1["eye_rotation_x_deg"])
    order = np.lexsort((p1["field_y_relative"], p1["field_x_relative"], di, bi))
    p1 = p1[order]
    fields = np.stack((p1["field_x_relative"], p1["field_y_relative"]), -1).reshape(101, 101, 9, 2)
    expected = np.array([(x, y) for x in (-1., 0., 1.) for y in (-1., 0., 1.)])
    assert np.array_equal(fields, np.broadcast_to(expected, fields.shape))
    rel1 = np.stack((p1["real_x_mm"], p1["real_y_mm"]), -1).reshape(101, 101, 9, 2)
    center1 = np.stack((p1["center_absolute_x_mm"], p1["center_absolute_y_mm"]), -1).reshape(101, 101, 9, 2)
    assert np.array_equal(center1, np.broadcast_to(center1[..., :1, :], center1.shape))
    absolute1 = rel1+center1
    valid1 = np.all(np.isfinite(absolute1), axis=(2, 3))
    # Uniform means over the same nine measured fields; never drop a field.
    mean1_grid = np.mean(absolute1, axis=2)
    mean1 = np.broadcast_to(mean1_grid[None], (len(A), 101, 101, 2)).reshape(-1, 2).copy()
    valid = np.broadcast_to(valid1[None], (len(A), 101, 101)).ravel() & np.all(np.isfinite(relative), axis=(1, 2))
    absolute4 = relative+centers[:, None, :]
    provenance = {"P1_path": str(p1_path), "P1_sha256": sha, "P4_sha256": p4_sha,
        "paired_states": int(np.sum(valid)), "excluded_gaze_states": int(np.sum(~valid1)),
        "excluded_P4_A_gaze_states": int(np.sum(~valid)),
        "P1_reuse": "one measured A-independent gaze grid reused across P4 A levels, per user specification",
        "frame_assumption": "conditional common-SI-coordinate simulation; identical nominal rear-camera chain/FOV, no fitted registration transform; physical registration independently unproved",
        "source_paths": "P4 loaded binary model; P1 Lens/p1_ME.seq with SRC_ROT ADE=-20; source paths differ",
        "absolute_reconstruction": "center_absolute + real_relative for every ray"}
    return states, absolute4, mean1, centers, valid, provenance


def gaze_metrics(error):
    norm = np.linalg.norm(error, axis=1)
    ade, bde = scalar_metrics(error[:, 0]), scalar_metrics(error[:, 1])
    ade.pop("within_0_1D_fraction_all_attempted", None)
    bde.pop("within_0_1D_fraction_all_attempted", None)
    return {"ADE_deg": ade, "BDE_deg": bde,
        "paired_parameter_rmse_deg": float(np.sqrt(np.mean(norm**2))),
        "paired_parameter_p95_deg": float(np.percentile(norm, 95)), "paired_parameter_max_deg": float(np.max(norm)),
        "fraction_within_0_5deg_vector": float(np.mean(norm <= .5))}


def runtime_metrics(result, truth, true_centers, mask):
    take = np.asarray(mask)
    error = result["estimated_state"][take]-truth[take]
    initial = result["initial_state"][take, 1:]-truth[take, 1:]
    center_error = result["estimated_center_absolute_mm"][take]-true_centers[take]
    return {"states": int(np.sum(take)), "A_d": scalar_metrics(error[:, 0]),
        "initial_gaze": gaze_metrics(initial), "corrected_gaze": gaze_metrics(error[:, 1:]),
        "center_rms_um": float(np.sqrt(np.mean(np.sum(center_error**2, axis=1)))*1000.),
        "converged": int(np.sum(result["converged"][take])), "cycles": int(np.sum(result["cycle"][take])),
        "stagnated": int(np.sum(result["stagnation"][take])), "iteration_cap": int(np.sum(result["max_iterations"][take])),
        "invalid_map": int(np.sum(~result["valid_map"][take])),
        "invalid_signal_map": int(np.sum(~result["signal_map_valid"][take])),
        "out_of_domain_updates": int(np.sum(result["unconstrained_out_of_domain_updates"][take])),
        "frames_with_out_of_domain_update": int(np.sum(result["unconstrained_out_of_domain_updates"][take] > 0)),
        "backtracked_updates": int(np.sum(result["backtracked_updates"][take])),
        "bound_hits_A_ADE_BDE": np.sum(result["boundary"][take], axis=0).tolist(),
        "median_iterations": float(np.median(result["iterations"][take])),
        "p95_iterations": float(np.percentile(result["iterations"][take], 95)),
        "joint_vertex_rms_um": float(np.sqrt(np.mean(result["observable_joint_sse_mm2"][take]/9.))*1000.)}


def stratified_indices(states, allowed, per_A, seed):
    rng = np.random.default_rng(seed)
    arrays = []
    for value in np.unique(states[:, 0]):
        pool = np.flatnonzero(allowed & (states[:, 0] == value))
        arrays.append(rng.choice(pool, min(per_A, len(pool)), replace=False))
    return np.sort(np.concatenate(arrays))


def pilot_indices(states, allowed):
    rng = np.random.default_rng(20261107)
    arrays = []
    for i, value in enumerate(np.unique(states[:, 0])):
        pool = np.flatnonzero(allowed & (states[:, 0] == value))
        arrays.append(rng.choice(pool, 24+int(i < 8), replace=False))
    output = np.sort(np.concatenate(arrays))
    assert len(output) == 512
    return output


def combine_starts(results):
    costs = np.stack([result["observable_joint_sse_mm2"] for result in results], axis=1)
    choice = np.argmin(np.where(np.isfinite(costs), costs, np.inf), axis=1)
    combined = {key: np.stack([result[key] for result in results], axis=1)[np.arange(len(choice)), choice]
                for key in results[0]}
    combined["selected_start_index"] = choice
    combined["all_start_costs_mm2"] = costs
    combined["all_start_states"] = np.stack([result["estimated_state"] for result in results], axis=1)
    combined["all_start_converged"] = np.stack([result["converged"] for result in results], axis=1)
    return combined


def run_in_chunks(model, signal, absolute4, mean1, starts, chunk=512):
    outputs = {}
    t0 = time.perf_counter()
    for lower in range(0, len(mean1), chunk):
        upper = min(lower+chunk, len(mean1))
        attempts = [iterate_centroids(model, signal, absolute4[lower:upper], mean1[lower:upper], initial_A=start) for start in starts]
        merged = combine_starts(attempts)
        # Keep the prescribed A=2 result explicitly even when fixed-start
        # robustness controls are used and objective-based selection differs.
        primary = attempts[starts.index(2.)]
        merged["primary_A2_state"] = primary["estimated_state"]
        merged["primary_A2_converged"] = primary["converged"]
        for key, value in merged.items():
            if key not in outputs:
                outputs[key] = np.empty((len(mean1),)+value.shape[1:], dtype=value.dtype)
            outputs[key][lower:upper] = value
        if upper == len(mean1) or upper % 2048 == 0:
            elapsed = time.perf_counter()-t0
            log(f"paired iteration{upper:,}/{len(mean1):,};elapsed{elapsed:.1f}s;projected{elapsed*len(mean1)/upper:.1f}s")
    return outputs, time.perf_counter()-t0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-fit", type=Path, default=SOURCE_FIT)
    parser.add_argument("--P1", type=Path, default=P1_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model75", type=Path, help="Explicit separately audited75-pose candidate/comparison, e.g. the predeclared69-value model")
    parser.add_argument("--pilot-only", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=6):
        states, absolute4, mean1, true_centers, valid, provenance = pair_inputs(args.source_fit, args.P1)
        fit_summary = json.loads((args.source_fit/"summary.json").read_text(encoding="utf-8"))
        summary = {"provenance": provenance, "calibrations": {}}
        summary["provenance"]["runtime_engine_sha256"] = hashlib.sha256((ROOT/"Script/p4_centroid_iteration.py").read_bytes()).hexdigest()
        summary["provenance"]["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        raw_difference = np.mean(absolute4, axis=1)-mean1
        corrected_difference = true_centers-mean1
        common = valid & ~calibration_masks(75)["calibration"]
        pilot_take = pilot_indices(states, common)
        random_take = stratified_indices(states, common, 1000, 20261107)
        boundary = np.any(np.isclose(np.abs(states[:, 1:]), 20., atol=1e-9), axis=1)
        stress_take = np.flatnonzero(common & boundary)
        models, signals, starts_for_size, projections = {}, {}, {}, {}
        for size in (27, 75):
            cal = calibration_masks(size)["calibration"]
            if np.any(cal & ~valid):
                raise RuntimeError(f"Required{size}-state calibration has invalid paired P1/P4 real rays")
            model_name = fit_summary["selected"][str(size)]
            model_path = args.model75 if size == 75 and args.model75 else args.source_fit/f"{model_name}.npz"
            with np.load(model_path) as model_data:
                extended = len(model_data["coefficients"]) == 66
            if extended:
                from p4_sparse_model_69 import SparseModel69
                model = SparseModel69.load(model_path)
            else:
                model = SparseModel.load(model_path)
            signal, selection = select_signal_map(states[cal], corrected_difference[cal])
            raw_map, raw_rank = fit_signal(states[cal, 0], states[cal, 1:], raw_difference[cal], 0)
            signal.save(args.output_dir/f"center_signal_{size}.npz", calibration_states=states[cal])
            raw_map.save(args.output_dir/f"raw_mean_linearity_{size}.npz", diagnostic_only=np.array(True))
            # Calibration-only selection is complete before any dense error is scored.
            raw_gaze, raw_unbounded, _, _ = raw_map.inverse(states[:, 0], raw_difference)
            corrected_gaze, corrected_unbounded, _, _ = signal.inverse(states[:, 0], corrected_difference)
            masks = calibration_masks(size)
            diagnostic = {"raw_mean_map_coefficients": 6, "raw_mean_map_deployed": False,
                "shape_model_path": str(model_path), "explicit75_comparison_override": bool(size == 75 and args.model75),
                "raw_mean_rank": raw_rank, "corrected_signal_selection": selection,
                "shape_coefficients": model.parameter_count, "center_signal_coefficients": signal.parameter_count,
                "total_deployed_coefficients": model.parameter_count+signal.parameter_count,
                "raw_mean_gaze_error": {}, "raw_mean_bounded_gaze_error": {},
                "raw_mean_unconstrained_out_of_domain_frames": int(np.sum(np.any(np.abs(raw_unbounded)>20., axis=1))),
                "corrected_center_gaze_error_given_true_A_diagnostic": {},
                "raw_mean_error_by_A": {}}
            for name, take in masks.items():
                take = take & valid
                if np.any(take):
                    diagnostic["raw_mean_gaze_error"][name] = gaze_metrics(raw_unbounded[take]-states[take, 1:])
                    diagnostic["raw_mean_bounded_gaze_error"][name] = gaze_metrics(raw_gaze[take]-states[take, 1:])
                    diagnostic["corrected_center_gaze_error_given_true_A_diagnostic"][name] = gaze_metrics(corrected_unbounded[take]-states[take, 1:])
            for av in np.unique(states[:, 0]):
                take = valid & (states[:, 0] == av) & masks["heldout"]
                diagnostic["raw_mean_error_by_A"][f"{av:.1f}"] = gaze_metrics(raw_unbounded[take]-states[take, 1:])
            pilot_results = []
            starts = [2.]
            begun = time.perf_counter()
            first = iterate_centroids(model, signal, absolute4[pilot_take], mean1[pilot_take], initial_A=2., log=log)
            pilot_results.append(first)
            if np.any(~first["converged"]):
                starts = [0., 2., 4.]
                pilot_results = [iterate_centroids(model, signal, absolute4[pilot_take], mean1[pilot_take], initial_A=start)
                                 if start != 2. else first for start in starts]
            pilot_wall = time.perf_counter()-begun
            combined = combine_starts(pilot_results)
            np.savez_compressed(args.output_dir/f"pilot_{size}.npz", selection_indices=pilot_take,
                truth_state=states[pilot_take], fixed_A_starts=np.array(starts), **combined)
            projection = pilot_wall*int(np.sum(common))/len(pilot_take)
            diagnostic["pilot"] = {"states": len(pilot_take), "wall_seconds": pilot_wall,
                "projected_full_common_heldout_seconds": projection, "fixed_A_starts": starts,
                "metrics": runtime_metrics(combined, states[pilot_take], true_centers[pilot_take], np.ones(len(pilot_take), bool)),
                "multistart_trigger": "any nonconverged fixed-A2 pilot frame; alternatives chosen only by observable joint SSE"}
            summary["calibrations"][str(size)] = diagnostic
            models[size], signals[size], starts_for_size[size], projections[size] = model, signal, starts, projection
            write_json(args.output_dir/"summary.json", summary)
            log(f"cal{size}512pilot{pilot_wall:.2f}s;fullheldoutprojection{projection:.1f}s;starts{starts}")
        if args.pilot_only:
            return
        full = max(projections.values()) <= 600.
        calibration_take = np.flatnonzero(valid & calibration_masks(75)["calibration"])
        selection = (np.flatnonzero(valid) if full else np.unique(np.r_[random_take, stress_take, calibration_take]))
        scope = {"all_valid_source_states": bool(full), "runtime_states": len(selection),
            "uniform_random_common_heldout_states": len(random_take), "complete_boundary_stress_states": len(stress_take),
            "seed": 20261107, "random_count_per_A": 1000,
            "distribution_note": "random per-A sample and complete boundary stress set reported separately; no pooled full-grid claim"}
        summary["scope"] = scope
        for size in (27, 75):
            result, seconds = run_in_chunks(models[size], signals[size], absolute4[selection], mean1[selection], starts_for_size[size])
            np.savez_compressed(args.output_dir/f"runtime_{size}.npz", selection_indices=selection,
                truth_state=states[selection], truth_center_absolute_mm=true_centers[selection],
                fixed_A_starts=np.array(starts_for_size[size]), signed_state_error=result["estimated_state"]-states[selection], **result)
            masks = {name: mask[selection] for name, mask in calibration_masks(size).items()}
            masks["common_random_heldout"] = np.isin(selection, random_take)
            masks["complete_boundary_stress"] = np.isin(selection, stress_take)
            endpoints = np.isclose(states[selection, 0], 0.) | np.isclose(states[selection, 0], 4.)
            masks["heldout_interior_A"] = masks["heldout"] & ~endpoints
            masks["heldout_endpoint_A"] = masks["heldout"] & endpoints
            summary["calibrations"][str(size)]["runtime"] = {
                "wall_seconds": seconds, "metrics": {name: runtime_metrics(result, states[selection], true_centers[selection], mask)
                    for name, mask in masks.items() if np.any(mask)}}
            write_json(args.output_dir/"summary.json", summary)
        lines = ["# Iterative P4−P1 centroid inference", "",
            provenance["frame_assumption"]+". No synthetic P1 signal or heldout registration transform is used.", "",
            "Each runtime call receives only measured P4 rays, measured P1 mean and calibrated shared maps. Initial A is fixed2D; gaze comes from D(2,g) inverted on the raw mean difference. The model predicts the P4 centroid bias, updates corrected-center gaze, and profiles A from all9 mean-centered rays. Two small undamped proposals define convergence; cycles, stagnation, bounds and all failures remain saved.", "",
            "The separate6-coefficient raw-mean affine fit is a linearity diagnostic and is excluded from deployed counts. The corrected-center D0/D1 map is chosen using calibration-only CV; its6/12 values are counted in addition to shape-model values.", "",
            "| Cal states | Shape values | Center/gaze values | Deployed total | Heldout N | A RMSE(D) | P95 abs(D) | Within0.1D | Gaze paired RMSE(deg) | Converged |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for size in (27, 75):
            row = summary["calibrations"][str(size)]
            q = row["runtime"]["metrics"]["heldout"]
            a = q["A_d"]
            lines.append(f"|{size}|{row['shape_coefficients']}|{row['center_signal_coefficients']}|{row['total_deployed_coefficients']}|{q['states']}|{a['rmse']:.6g}|{a['p95_abs']:.6g}|{100*a['within_0_1D_fraction_all_attempted']:.3f}%|{q['corrected_gaze']['paired_parameter_rmse_deg']:.6g}|{q['converged']}|")
        lines += ["", "Numerical arrays and unrounded estimates, raw/accepted proposals, objective histories, statuses, source hashes and separate boundary/random/interior-A/endpoint-A scores are retained. This is a conditional simulation accuracy test; it does not prove physical frame registration, sensor-noise performance or0.1D resolution."]
        (args.output_dir/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
        log(f"P4-P1 iterationcomplete:{args.output_dir}")


if __name__ == "__main__":
    main()
