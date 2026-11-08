"""Explicit42-value DF27+D15 restricted control, with no calibration refit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from fit_p4_limited_gaze import DEFAULT_OUT
from evaluate_p4_p1_limited_gaze import load_shape, load_signal, run, score_limited, subset_masks
from evaluate_p4_p1_centroid_iteration import pilot_indices
from fit_p4_sparse_calibration import write_json, log


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--shape", type=Path)
    parser.add_argument("--map", type=Path)
    parser.add_argument("--reference-runtime-summary", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--chunk", type=int, default=1024)
    args = parser.parse_args()
    shape_path = args.shape or args.calibration_dir/"models/new75_DF27.npz"
    map_path = args.map or args.calibration_dir/"models/new75_D15.npz"
    out = args.output_dir or args.calibration_dir/"runtime_same_map42"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=6):
        with np.load(args.calibration_dir/"source_subset.npz") as saved:
            states, y4, m1, centers, source_indices, common, cal = (saved[key] for key in
                ("states", "absolute_p4_mm", "mean_p1_absolute_mm", "center_absolute_mm", "source_indices", "common_holdout", "cal75"))
        shape, signal = load_shape(shape_path), load_signal(map_path)
        if shape.parameter_count != 27 or shape.composition != "distortion_first" or signal.parameter_count != 15:
            raise RuntimeError("The explicit same-map control is DF27+D15=42")
        for path in (shape_path, map_path):
            with np.load(path) as saved:
                if not np.array_equal(saved["calibration_states"], states[cal]):
                    raise RuntimeError("Both inputs must document the exact same new75 physical calibration poses")
        map_sha = digest(map_path)
        reference_check = None
        if args.reference_runtime_summary:
            reference = json.loads(args.reference_runtime_summary.read_text(encoding="utf-8"))
            row = reference["cases"]["new75_selected"]
            if row["shape_count"] != 69 or row["map_count"] != 15 or row["map_sha256"] != map_sha:
                raise RuntimeError("Reference PF69+D15 case does not use the identical D15 map")
            reference_check = str(args.reference_runtime_summary)
        selection = np.flatnonzero(common)
        assert len(selection) == 83250
        pilot_take = pilot_indices(states, common)
        pilot, pilot_seconds = run(shape, signal, y4, m1, pilot_take, source_indices, out, "pilot_DF27_D15", 512)
        np.savez_compressed(out/"pilot_DF27_D15.npz", source_indices=source_indices[pilot_take], truth_state=states[pilot_take], **pilot)
        summary = {"case": "new75_DF27_D15_same_map", "calibration_poses": 75,
            "shape_count": 27, "map_count": 15, "total_count": 42,
            "shape_path": str(shape_path), "shape_sha256": digest(shape_path), "map_path": str(map_path), "map_sha256": map_sha,
            "same_D15_reference_summary": reference_check, "requested_inverse_bound_deg": 12.5,
            "actual_ray_support_bound_deg": 12.4, "common_holdout_states": 83250,
            "calibration_action": "none; frozen75-pose shape and map loaded without refitting",
            "pilot": {"wall_seconds": pilot_seconds, "projected_full_seconds": pilot_seconds*83250/512.,
                "metrics": score_limited(pilot, states[pilot_take], centers[pilot_take], np.ones(512, bool))}}
        write_json(out/"summary.json", summary)
        if args.pilot_only:
            return
        result, seconds = run(shape, signal, y4, m1, selection, source_indices, out, "DF27_D15", args.chunk)
        np.savez_compressed(out/"runtime_new75_DF27_D15_same_map.npz", source_indices=source_indices[selection],
            truth_state=states[selection], true_center_absolute_mm=centers[selection],
            signed_state_error=result["estimated_state"]-states[selection], **result)
        masks = subset_masks(states[selection])
        summary["runtime"] = {"wall_seconds": seconds,
            "metrics": {name: score_limited(result, states[selection], centers[selection], take) for name, take in masks.items() if np.any(take)},
            "per_A": {f"{a:.1f}": score_limited(result, states[selection], centers[selection], np.isclose(states[selection, 0], a)) for a in np.unique(states[:, 0])}}
        write_json(out/"summary.json", summary)
        q = summary["runtime"]["metrics"]["common_holdout"]
        (out/"report.md").write_text(
            "# Same-map limited-gaze shape-capacity control\n\n"
            "This is the explicit75-pose DF27+D15=42 control. The identical D15 file is used by the PF69+D15=84 comparison; no model/map is refitted. All83,250 common heldout frames are retained with the same±12.5 inverse bounds and actual±12.4 source support.\n\n"
            f"A RMSE{q['A_d']['rmse']:.7g}D, P95abs{q['A_d']['p95_abs']:.7g}D, gaze RMS{q['final_gaze']['paired_parameter_rmse_deg']:.7g}deg; {q['converged']:,}/{q['states']:,}converged.\n\n"
            "The separate DF27+D12=39 case remains an affine-map ablation. This42-vs84 comparison isolates shape capacity at equal calibration poses and equal center/gaze map.\n", encoding="utf-8")
        log(f"same-map42-value limited control complete: {out}")


if __name__ == "__main__":
    main()
