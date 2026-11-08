"""Fair old/new shared-model inference on the83,250 restricted common holdout."""
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
from p4_cubic_center_signal import CubicCenterSignal
from p4_joint_observed_solver_limited import solve_observed
from evaluate_p4_p1_joint_refinement import score, HISTORY_KEYS
from evaluate_p4_p1_centroid_iteration import pilot_indices
from fit_p4_limited_gaze import DEFAULT_OUT
from fit_p4_sparse_calibration import write_json, log

ROOT = Path(__file__).resolve().parents[1]
OLD_SHAPE = ROOT/"data/p4_sparse_calibration_69/cal75_projective_first_69.npz"
OLD_MAP = ROOT/"data/p4_p1_cubic_center_full/center_signal_D15.npz"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_shape(path):
    with np.load(path) as stored:
        extended = len(stored["coefficients"]) == 66
    return SparseModel69.load(path) if extended else SparseModel.load(path)


def load_signal(path):
    with np.load(path) as stored:
        cubic = "x_coefficients" in stored
    return CubicCenterSignal.load(path) if cubic else CenterSignalMap.load(path)


def subset_masks(states):
    endpoints = np.isclose(states[:, 0], 0.) | np.isclose(states[:, 0], 4.)
    radius = np.max(np.abs(states[:, 1:]), axis=1)
    masks = {"common_holdout": np.ones(len(states), bool), "interior_A": ~endpoints, "endpoint_A": endpoints,
        "sampled_gaze_edge": np.any(np.isclose(np.abs(states[:, 1:]), 12.4, atol=1e-9, rtol=0), axis=1)}
    for lo, hi in ((0., 4.), (4., 8.), (8., 12.4)):
        masks[f"max_abs_gaze_{lo:g}_{hi:g}"] = (radius >= lo if lo == 0. else radius > lo) & (radius <= hi+1e-9)
    return masks


def score_limited(result, truth, centers, mask):
    row = score(result, truth, centers, mask)
    gaze = result["estimated_state"][mask, 1:]
    row["estimates_in_unsampled_12p4_to_12p5_strip"] = int(np.sum(np.any(np.abs(gaze) > 12.4+1e-9, axis=1)))
    row["estimates_outside_requested_12p5_bound"] = int(np.sum(np.any(np.abs(gaze) > 12.5+1e-8, axis=1)))
    row["initial_unbounded_seeds_outside_requested_domain"] = int(np.sum(np.any(np.abs(
        result["all_seed_initial_unconstrained_gaze"][mask]) > 12.5+1e-9, axis=2)))
    return row


def run(model, signal, y4, m1, selection, source_indices, out, tag, chunk):
    arrays = {}
    traces = out/f"traces_{tag}"
    traces.mkdir(exist_ok=True)
    started = time.perf_counter()
    for lower in range(0, len(selection), chunk):
        upper = min(lower+chunk, len(selection))
        ids = selection[lower:upper]
        result = solve_observed(model, signal, y4[ids], m1[ids], gaze_bound_deg=12.5, record_history=True)
        histories = {key: result.pop(key) for key in HISTORY_KEYS}
        np.savez_compressed(traces/f"frames_{lower:06d}_{upper-1:06d}.npz", source_indices=source_indices[ids], **histories)
        for key, value in result.items():
            if key not in arrays:
                arrays[key] = np.empty((len(selection),)+value.shape[1:], dtype=value.dtype)
            arrays[key][lower:upper] = value
        if upper == len(selection) or upper % 4096 == 0:
            elapsed = time.perf_counter()-started
            log(f"{tag}: {upper:,}/{len(selection):,}; elapsed{elapsed:.1f}s; projected{elapsed*len(selection)/upper:.1f}s")
    return arrays, time.perf_counter()-started


def write_report(out, summary):
    lines = ["# Limited-gaze shared calibration and observed-data inference", "",
        "Requested gaze range:±12.5 degrees. Existing ray data cover exactly±12.4 on63×63 gaze nodes. Both old and new inversions use the SAME physical±12.5 bound; coefficients retain t=gaze/20 normalization. Estimates in the unobserved0.1-degree edge strip are counted separately and are not validated ray samples.", "",
        "New calibration uses A=0,2,4D and either3×3 gaze nodes−12.4,0,12.4 or5×5 nodes−12.4,−6.4,0,6.4,12.4. New coefficients are fitted only from those raw observations with identity/zero starts. Shape/map selection uses calibration-only CV/rank and the one-SE rule. The83,250 common holdout excludes the union of99 old/new training rows; no P1 anomaly or large error is removed.", "",
        "| Case | Cal poses | Shape values | Map values | Total | Common N | A RMS(D) | P95 abs(D) | P99 abs(D) | Max abs(D) | Gaze RMS(deg) | Converged |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in summary["cases"].items():
        if "runtime" not in row:
            continue
        q = row["runtime"]["metrics"]["common_holdout"]
        a = q["A_d"]
        lines.append(f"|{name}|{row['calibration_poses']}|{row['shape_count']}|{row['map_count']}|{row['total_count']}|{q['states']:,}|{a['rmse']:.7g}|{a['p95_abs']:.7g}|{a['p99_abs']:.7g}|{a['max_abs']:.7g}|{q['final_gaze']['paired_parameter_rmse_deg']:.7g}|{q['converged']:,}|")
    lines += ["", "Inference uses measured absolute P4 points and measured P1 mean only, three fixed A starts, one model centroid correction, and coherent joint observed-coordinate LM. True A, gaze and central-ray center are used only to score. Initial-to-final gaze, A endpoint/interior and gaze-bin errors, both center estimates, raw gradients, all seed costs/statuses, alternative minima and selected trajectories are retained.", "",
        "The nominal common-SI-coordinate simulation assumption is unchanged. ±0.1D is a comparison reference, not a pass/fail test or a resolution guarantee. Calibration noise and quality-flag subsets are separate diagnostics and do not alter primary all-frame results."]
    (out/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--old-shape", type=Path, default=OLD_SHAPE)
    parser.add_argument("--old-map", type=Path, default=OLD_MAP)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--chunk", type=int, default=1024)
    args = parser.parse_args()
    out = args.output_dir or args.calibration_dir/"joint_runtime"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=6):
        fitted = json.loads((args.calibration_dir/"summary.json").read_text(encoding="utf-8"))
        with np.load(args.calibration_dir/"source_subset.npz") as stored:
            states, y4, m1, centers, source_indices = (stored[key] for key in
                ("states", "absolute_p4_mm", "mean_p1_absolute_mm", "center_absolute_mm", "source_indices"))
            common = stored["common_holdout"]
        assert len(states) == 83349 and np.sum(common) == 83250
        definitions = [("old_fullrange75_F69_D15", args.old_shape, args.old_map, 75)]
        for case, shape_key, map_key, poses in (("new27_selected", "27", "27", 27),
                ("new75_low", "75_low", "75", 75), ("new75_selected", "75", "75", 75)):
            shape_name = fitted["shape_selection"][shape_key]["selected"]
            shape_path = Path(fitted["shape_candidates"][shape_name]["model_path"])
            map_name = fitted["map_selection"][map_key]["selected"]
            map_path = Path(fitted["map_selection"][map_key]["artifacts"][map_name]["path"])
            definitions.append((case, shape_path, map_path, poses))
        summary = {"provenance": fitted["provenance"], "bounds": {"requested_runtime_deg": 12.5,
            "sampled_ray_support_deg": 12.4, "coordinate_normalization_deg": 20.},
            "source_hashes": {name: digest(ROOT/"Script"/name) for name in
                ("evaluate_p4_p1_limited_gaze.py", "p4_joint_observed_solver_limited.py", "p4_limited_bounds.py")},
            "common_holdout_states": 83250, "calibration_comparison_source": str(args.calibration_dir/"summary.json"), "cases": {}}
        selection = np.flatnonzero(common)
        pilot_take = pilot_indices(states, common)
        prepared = {}
        for name, shape_path, map_path, poses in definitions:
            model, signal = load_shape(shape_path), load_signal(map_path)
            prepared[name] = model, signal
            result, seconds = run(model, signal, y4, m1, pilot_take, source_indices, out, f"pilot_{name}", 512)
            np.savez_compressed(out/f"pilot_{name}.npz", source_indices=source_indices[pilot_take], truth_state=states[pilot_take], **result)
            summary["cases"][name] = {"shape_path": str(shape_path), "shape_sha256": digest(shape_path),
                "map_path": str(map_path), "map_sha256": digest(map_path), "calibration_poses": poses,
                "shape_count": model.parameter_count, "map_count": signal.parameter_count,
                "total_count": model.parameter_count+signal.parameter_count,
                "pilot": {"wall_seconds": seconds, "projected_full_seconds": seconds*83250/512.,
                    "metrics": score_limited(result, states[pilot_take], centers[pilot_take], np.ones(512, bool))}}
            write_json(out/"summary.json", summary)
            log(f"{name}512pilot{seconds:.3f}s;full83,250 projection{seconds*83250/512.:.1f}s")
        write_report(out, summary)
        if args.pilot_only:
            return
        reused = {}
        for name, shape_path, map_path, poses in definitions:
            key = (digest(shape_path), digest(map_path))
            if key in reused:
                result, seconds, source_name = reused[key], 0., next(k for k, v in summary["cases"].items() if k != name and v.get("runtime", {}).get("model_pair_sha256") == list(key))
            else:
                result, seconds = run(*prepared[name], y4, m1, selection, source_indices, out, name, args.chunk)
                reused[key], source_name = result, None
            np.savez_compressed(out/f"runtime_{name}.npz", source_indices=source_indices[selection],
                truth_state=states[selection], true_center_absolute_mm=centers[selection],
                signed_state_error=result["estimated_state"]-states[selection], **result)
            masks = subset_masks(states[selection])
            summary["cases"][name]["runtime"] = {"wall_seconds": seconds, "reused_identical_model_case": source_name,
                "model_pair_sha256": list(key),
                "metrics": {k: score_limited(result, states[selection], centers[selection], take) for k, take in masks.items() if np.any(take)},
                "per_A": {f"{a:.1f}": score_limited(result, states[selection], centers[selection], np.isclose(states[selection, 0], a)) for a in np.unique(states[:, 0])}}
            write_json(out/"summary.json", summary)
            write_report(out, summary)
        log(f"limited-gaze observed-data comparison complete: {out}")


if __name__ == "__main__":
    main()
