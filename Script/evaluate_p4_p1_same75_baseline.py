"""Thin same75-pose baseline: old27-value shape plus the75-pose D12 map."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from p4_sparse_model import SparseModel
from p4_centroid_iteration import CenterSignalMap
from evaluate_p4_p1_centroid_iteration import pair_inputs, SOURCE_FIT, P1_PATH
from evaluate_p4_p1_joint_refinement import run, score, DEFAULT_SIGNALS
from fit_p4_sparse_calibration import calibration_masks, write_json, log

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT/"data/p4_p1_same75_loworder_baseline"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-fit", type=Path, default=SOURCE_FIT)
    parser.add_argument("--signal-dir", type=Path, default=DEFAULT_SIGNALS)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--P1", type=Path, default=P1_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--chunk", type=int, default=1024)
    args = parser.parse_args()
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=6):
        states, y4, m1, centers, valid, provenance = pair_inputs(args.source_fit, args.P1)
        fitted = json.loads((args.source_fit/"summary.json").read_text(encoding="utf-8"))
        model_path = args.model or args.source_fit/f"{fitted['selected']['75']}.npz"
        signal_path = args.signal_dir/"center_signal_75.npz"
        shape = SparseModel.load(model_path)
        signal = CenterSignalMap.load(signal_path)
        if shape.parameter_count != 27 or signal.parameter_count != 12:
            raise RuntimeError("This control specifically requires the OLD75-pose27-value shape and75-pose D12 map")
        cal = calibration_masks(75)["calibration"]
        expected = states[cal]
        for path in (model_path, signal_path):
            with np.load(path) as saved:
                if "calibration_states" not in saved or not np.array_equal(saved["calibration_states"], expected):
                    raise RuntimeError(f"Input does not document the exact same75 calibration poses: {path}")
        common = valid & ~cal
        selection = np.flatnonzero(common)
        if len(selection) != 214146:
            raise RuntimeError(f"Expected all214146 common heldout states, found{len(selection)}; no outlier filtering is authorized")
        result, seconds = run(shape, signal, y4, m1, selection, out, "same75_baseline", args.chunk)
        np.savez_compressed(out/"runtime_same75_baseline.npz", selection_indices=selection,
            truth_state=states[selection], true_center_absolute_mm=centers[selection],
            signed_state_error=result["estimated_state"]-states[selection], **result)
        endpoint = np.isclose(states[selection, 0], 0.) | np.isclose(states[selection, 0], 4.)
        boundary = np.any(np.isclose(np.abs(states[selection, 1:]), 20., atol=1e-9), axis=1)
        masks = {"common_heldout": np.ones(len(selection), bool), "interior_A": ~endpoint,
                 "endpoint_A": endpoint, "gaze_boundary": boundary, "gaze_interior": ~boundary}
        summary = {"provenance": provenance, "shape_path": str(model_path), "shape_sha256": digest(model_path),
            "signal_path": str(signal_path), "signal_sha256": digest(signal_path),
            "driver_sha256": digest(__file__), "solver_sha256": digest(ROOT/"Script/p4_joint_observed_solver.py"),
            "calibration_context": {"physical_poses": 75, "A_D": [0., 2., 4.],
                "ADE_BDE_grid_deg": [-20., -10., 0., 10., 20.], "shape_values": 27,
                "center_signal_values": 12, "deployed_values": 39,
                "center_map_source": "center_signal_75; center_signal_27 is explicitly not used",
                "selection": "previously frozen calibration-only selected75-pose low-order shape; no refit or new selection"},
            "heldout_states": len(selection), "wall_seconds": seconds,
            "metrics": {name: score(result, states[selection], centers[selection], mask) for name, mask in masks.items()},
            "per_A": {f"{a:.1f}": score(result, states[selection], centers[selection], np.isclose(states[selection, 0], a))
                       for a in np.unique(states[:, 0])}}
        write_json(out/"summary.json", summary)
        q = summary["metrics"]["common_heldout"]
        text = ("# Same75-pose low-order baseline\n\n"
            "This control uses the OLD27-value shape model calibrated on the SAME75 physical poses as the69-value model, plus the SAME75-pose D12 center/gaze map. It uses the unchanged observed-data solver and all214,146 common heldout states, including every finite anomalous P1 state. No coefficients are refitted and no data are filtered.\n\n"
            f"Deployed count:27+12=39 values. A RMSE{q['A_d']['rmse']:.7g}D; P95abs{q['A_d']['p95_abs']:.7g}D; paired gaze-parameter RMSE{q['final_gaze']['paired_parameter_rmse_deg']:.7g}deg; {q['converged']:,}/{q['states']:,} converged.\n\n"
            "This isolates shape-family capacity from calibration-pose count when compared with the75-pose69+D12 and69+D15 experiments. The common-SI-coordinate simulation assumption remains explicit.\n")
        (out/"report.md").write_text(text, encoding="utf-8")
        log(f"same75-pose baseline complete: {out}")


if __name__ == "__main__":
    main()
