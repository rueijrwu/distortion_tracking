"""Latency-only check on the exact previously frozen16 ordinary frames."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from p4_aaware_center_cubic import AwareCenterCubic, DEFAULT_ROOT
from p4_center_cubic_fixedpoint_aware import solve_fixedpoint
from fit_p4_limited_gaze import DEFAULT_OUT as SOURCE_ROOT
from evaluate_p4_p1_limited_gaze import load_shape
from fit_p4_p1_axis_cubic_gaze import frozen_quality_flags
from fit_p4_sparse_calibration import write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IDS = ROOT/"data/p4_p1_center_cubic_integration/comparison_sample_21k_predeclared.npz"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_ROOT/"calibration")
    parser.add_argument("--latency-ids", type=Path, default=DEFAULT_IDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT/"latency16_frozen")
    args = parser.parse_args()
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=6):
        source = args.source_dir/"source_subset.npz"
        with np.load(source) as stored:
            states, y4, m1, global_ids, common = (stored[key] for key in
                ("states", "absolute_p4_mm", "mean_p1_absolute_mm", "source_indices", "common_holdout"))
        with np.load(args.latency_ids) as frozen:
            ids = frozen["latency_source_indices"].astype(int)
            expected_states = frozen["latency_states"]
            seed = int(frozen["latency_seed"])
        lookup = np.full(21*101*101, -1, int)
        lookup[global_ids] = np.arange(len(global_ids))
        local = lookup[ids]
        if len(ids) != 16 or np.any(local < 0) or not np.all(common[local]) or not np.array_equal(states[local], expected_states):
            raise RuntimeError("Saved latency IDs/labels are not the exact common16 frames")
        if np.any(frozen_quality_flags(states)[local]):
            raise RuntimeError("The saved16 were specified as ordinary/unflagged; do not silently replace them")
        calibration = json.loads((args.calibration_dir/"summary.json").read_text(encoding="utf-8"))
        selected = calibration["selection"]["selected"]
        Cpath = Path(calibration["models"][selected]["path"])
        Fpath = args.source_dir/"models/new75_DF27.npz"
        model, C = load_shape(Fpath), AwareCenterCubic.load(Cpath)
        result = {"predeclared_ID_artifact": str(args.latency_ids), "ID_artifact_sha256": digest(args.latency_ids),
            "latency_seed": seed, "source_indices": ids.tolist(), "model": selected,
            "F_sha256": digest(Fpath), "C_sha256": digest(Cpath), "three_fixed_A_starts": [0., 2., 4.],
            "includes_all_fixedpoint_iterations": True, "disk_I_O_excluded": True,
            "warmup": "first ofsame16, once perprofile solver", "profile_methods": {},
            "source_hashes": {name: digest(ROOT/"Script"/name) for name in
                ("p4_center_cubic_fixedpoint_aware.py", "p4_fast_accommodation_profile.py", "benchmark_p4_aaware_latency.py")}}
        for method in ("reference401", "fast41"):
            solve_fixedpoint(model, C, y4[local[:1]], m1[local[:1]], profile_solver=method, record_history=False)
            times, estimates, statuses, calls, fallbacks = [], [], [], [], []
            for i in local:
                start = time.perf_counter()
                solved = solve_fixedpoint(model, C, y4[i:i+1], m1[i:i+1], profile_solver=method, record_history=False)
                times.append(time.perf_counter()-start)
                estimates.append(solved["estimated_state"][0])
                statuses.append(bool(solved["converged"][0]))
                calls.append(solved["all_start_profile_calls"][0])
                fallbacks.append(solved["all_start_profile401_fallback_calls"][0])
            result["profile_methods"][method] = {"seconds": times, "median_seconds": float(np.median(times)),
                "P95_seconds": float(np.percentile(times, 95)), "max_seconds": float(np.max(times)),
                "converged_frames": int(np.sum(statuses))}
            np.savez_compressed(out/f"latency_{method}.npz", source_indices=ids, states_for_scoring_only=states[local],
                seconds=np.array(times), estimated_state=np.array(estimates), converged=np.array(statuses),
                all_start_profile_calls=np.array(calls), all_start_fallback_calls=np.array(fallbacks))
        write_json(out/"summary.json", result)


if __name__ == "__main__":
    main()
