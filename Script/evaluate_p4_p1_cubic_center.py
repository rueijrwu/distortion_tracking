"""Same75-pose F69+D15 versus F69+D12, with calibration-only map selection."""
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

from p4_sparse_model_69 import SparseModel69
from p4_centroid_iteration import CenterSignalMap
from p4_cubic_center_signal import compare_center_maps
from p4_joint_observed_solver_cubic import solve_observed
from evaluate_p4_p1_joint_refinement import score, HISTORY_KEYS, DEFAULT_SIGNALS, DEFAULT_MODEL75
from evaluate_p4_p1_centroid_iteration import pair_inputs, pilot_indices, SOURCE_FIT, P1_PATH
from fit_p4_sparse_calibration import calibration_masks, write_json, log

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT/"data/p4_p1_cubic_center"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(model, signal, y4, m1, selection, out, tag, chunk):
    arrays = {}
    traces = out/f"traces_{tag}"
    traces.mkdir(exist_ok=True)
    started = time.perf_counter()
    for lower in range(0, len(selection), chunk):
        upper = min(lower+chunk, len(selection))
        ids = selection[lower:upper]
        result = solve_observed(model, signal, y4[ids], m1[ids], record_history=True)
        history = {key: result.pop(key) for key in HISTORY_KEYS}
        np.savez_compressed(traces/f"frames_{lower:06d}_{upper-1:06d}.npz", selection_indices=ids, **history)
        for key, value in result.items():
            if key not in arrays:
                arrays[key] = np.empty((len(selection),)+value.shape[1:], dtype=value.dtype)
            arrays[key][lower:upper] = value
        if upper == len(selection) or upper % 4096 == 0:
            elapsed = time.perf_counter()-started
            log(f"{tag}: {upper:,}/{len(selection):,}; elapsed{elapsed:.1f}s; projected{elapsed*len(selection)/upper:.1f}s")
    return arrays, time.perf_counter()-started


def saved_D12(path, expected_selection, common, shape_hash, signal_hash):
    """Reuse only an audited full control with exactly matching frozen inputs."""
    metadata = json.loads((path.parent/"summary.json").read_text(encoding="utf-8"))
    model = metadata["models"]["75"]
    if model["shape_sha256"] != shape_hash or model["signal_sha256"] != signal_hash:
        raise RuntimeError("Existing D12 control does not have identical pinned F69/D12 inputs")
    with np.load(path) as stored:
        indices = stored["selection_indices"]
        take = common[indices]
        if not np.array_equal(indices[take], expected_selection):
            raise RuntimeError("Existing D12 control does not cover the exact complete common heldout set")
        result = {key: stored[key][take].copy() for key in stored.files
            if key not in ("selection_indices", "truth_state", "true_center_absolute_mm", "signed_state_error")
            and stored[key].ndim and stored[key].shape[0] == len(indices)}
    return result


def report(out, summary):
    lines = ["# Same75-pose cubic center/gaze-map comparison", "",
        "The69-value P4 shape model remains frozen. Only the center/gaze map is changed: D15 uses seven predeclared x-output terms and eight y-output terms, with no extra fitted initializer. D6/D12/D15 are compared by calibration-only leave-one-nonzero-gaze-location CV, preserving the three zero-gaze anchors; the lower-count full-rank map within one SE is selected before dense scoring.", "",
        "D15 x basis:1,a,tx,tx*ty,tx³,tx*ty²,a*tx. D15 y basis:1,ty,tx²,ty²,tx²*ty,ty³,a,a*ty. Here a=A/4,tx=BDE/20,ty=ADE/20. The nonlinear inverse starts from this map's own zero-gaze linearization and stays within the calibrated gaze bounds.", "",
        f"Calibration-only selected center map: `{summary['calibration_map_comparison']['selected']}`. Both predeclared D12 and D15 experiments are retained regardless of selection. The observed-data LM algorithm and three blind A starts remain unchanged; no true state or center enters inference.", "",
        "| Experiment | Shape values | Center/gaze values | Total | Common heldout N | A RMSE(D) | A P95 abs(D) | Gaze RMSE(deg) | Converged |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in summary.get("runtime", {}).items():
        q = row["metrics"]["common_heldout"]
        lines.append(f"| F69+{name} |69|{row['signal_values']}|{69+row['signal_values']}|{q['states']:,}|{q['A_d']['rmse']:.7g}|{q['A_d']['p95_abs']:.7g}|{q['final_gaze']['paired_parameter_rmse_deg']:.7g}|{q['converged']:,}|")
    lines += ["", "Every common heldout state, including finite anomalous P1 states, is retained. Interior-A, endpoint-A and gaze-boundary scores, all seed statuses/costs, rank/condition, both center estimates, gradients and chunked histories are saved. The same nominal-SI-coordinate simulation assumption applies; no registration is learned from heldout data.", "",
        "D15 design covariance matrices describe calibration geometry under an explicitly stated unit-IID signal-target assumption. The separate100-draw1µm study is shape-calibration sensitivity; it is not a full D15/pipeline-noise experiment. ±0.1D remains a comparison reference, not a pass/fail or resolution guarantee."]
    (out/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-fit", type=Path, default=SOURCE_FIT)
    parser.add_argument("--model69", type=Path, default=DEFAULT_MODEL75)
    parser.add_argument("--signal-dir", type=Path, default=DEFAULT_SIGNALS)
    parser.add_argument("--P1", type=Path, default=P1_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--existing-D12-result", type=Path)
    parser.add_argument("--chunk", type=int, default=1024)
    args = parser.parse_args()
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=6):
        states, y4, m1, centers, valid, provenance = pair_inputs(args.source_fit, args.P1)
        cal = calibration_masks(75)["calibration"]
        if np.any(cal & ~valid):
            raise RuntimeError("All75 actual calibration poses must be valid; no filling or outlier deletion")
        common = valid & ~cal
        selection = np.flatnonzero(common)
        if len(selection) != 214146:
            raise RuntimeError("Expected all214146 common heldout states; no outlier filtering is authorized")
        model = SparseModel69.load(args.model69)
        signal_path = args.signal_dir/"center_signal_75.npz"
        D12 = CenterSignalMap.load(signal_path)
        if model.parameter_count != 69 or D12.parameter_count != 12:
            raise RuntimeError("Comparison requires frozen F69 and75-pose D12")
        with np.load(signal_path) as stored:
            if not np.array_equal(stored["calibration_states"], states[cal]):
                raise RuntimeError("The D12 control must come from these exact75 poses")
        maps, cv = compare_center_maps(states[cal], centers[cal]-m1[cal])
        D15 = maps["D15"]
        D15.save(out/"center_signal_D15.npz", calibration_states=states[cal],
                 calibration_targets_mm=centers[cal]-m1[cal])
        D12.save(out/"center_signal_D12_control.npz", calibration_states=states[cal])
        summary = {"provenance": provenance, "shape_path": str(args.model69),
            "shape_sha256": digest(args.model69), "D12_path": str(signal_path), "D12_sha256": digest(signal_path),
            "D15_sha256": digest(out/"center_signal_D15.npz"),
            "source_hashes": {name: digest(ROOT/"Script"/name) for name in
                ("p4_cubic_center_signal.py", "p4_joint_observed_solver_cubic.py", "evaluate_p4_p1_cubic_center.py")},
            "calibration_poses": 75, "calibration_map_comparison": cv,
            "refitted_D12_vs_frozen_max_abs_coefficient_difference": float(np.max(np.abs(maps["D12"].coefficients-D12.coefficients))),
            "pilot": {}, "runtime": {}}
        write_json(out/"summary.json", summary)
        pilot_take = pilot_indices(states, common)
        for name, signal in (("D12", D12), ("D15", D15)):
            result, seconds = run(model, signal, y4, m1, pilot_take, out, f"pilot_{name}", 512)
            np.savez_compressed(out/f"pilot_{name}.npz", selection_indices=pilot_take, truth_state=states[pilot_take], **result)
            summary["pilot"][name] = {"wall_seconds": seconds,
                "projected_full_seconds": seconds*len(selection)/512.,
                "metrics": score(result, states[pilot_take], centers[pilot_take], np.ones(512, bool))}
            write_json(out/"summary.json", summary)
            log(f"F69+{name}512pilot{seconds:.3f}s; full projection{seconds*len(selection)/512.:.1f}s")
        report(out, summary)
        if args.pilot_only:
            return
        endpoints = np.isclose(states[selection, 0], 0.) | np.isclose(states[selection, 0], 4.)
        boundary = np.any(np.isclose(np.abs(states[selection, 1:]), 20., atol=1e-9), axis=1)
        masks = {"common_heldout": np.ones(len(selection), bool), "interior_A": ~endpoints,
                 "endpoint_A": endpoints, "gaze_boundary": boundary, "gaze_interior": ~boundary}
        for name, signal in (("D12", D12), ("D15", D15)):
            reused = name == "D12" and args.existing_D12_result is not None
            if reused:
                result = saved_D12(args.existing_D12_result, selection, common, digest(args.model69), digest(signal_path))
                seconds = 0.
            else:
                result, seconds = run(model, signal, y4, m1, selection, out, f"runtime_{name}", args.chunk)
            np.savez_compressed(out/f"runtime_{name}.npz", selection_indices=selection,
                truth_state=states[selection], true_center_absolute_mm=centers[selection],
                signed_state_error=result["estimated_state"]-states[selection], **result)
            summary["runtime"][name] = {"signal_values": signal.parameter_count, "wall_seconds": seconds,
                "reused_control": str(args.existing_D12_result) if reused else None,
                "metrics": {key: score(result, states[selection], centers[selection], take) for key, take in masks.items()},
                "per_A": {f"{a:.1f}": score(result, states[selection], centers[selection], np.isclose(states[selection, 0], a))
                          for a in np.unique(states[:, 0])}}
            write_json(out/"summary.json", summary)
            report(out, summary)
        log(f"F69+D15 comparison complete: {out}")


if __name__ == "__main__":
    main()
