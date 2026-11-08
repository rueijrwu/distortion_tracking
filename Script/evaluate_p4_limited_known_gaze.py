"""Exact/controlled supplied-gaze A controls for the frozen limited models."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from fit_p4_limited_gaze import DEFAULT_OUT, evaluation_masks
from evaluate_p4_p1_limited_gaze import load_shape, OLD_SHAPE
from p4_known_gaze_inverse import estimate_accommodation
from fit_p4_sparse_calibration import score_inverse, scalar_metrics, forward_metrics, write_json, log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--old-shape", type=Path, default=OLD_SHAPE)
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--selected-only", action="store_true")
    args = parser.parse_args()
    out = args.output_dir or args.calibration_dir/("known_gaze_stress" if args.stress else "known_gaze")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    fitted = json.loads((args.calibration_dir/"summary.json").read_text(encoding="utf-8"))
    with np.load(args.calibration_dir/"source_subset.npz") as saved:
        states, y4, source_indices, common, cal27, cal75, old_cal = (saved[key] for key in
            ("states", "absolute_p4_mm", "source_indices", "common_holdout", "cal27", "cal75", "old_calibration_inside"))
    selected = {fitted["shape_selection"][key]["selected"] for key in ("27", "75_low", "75")}
    cases = [(name, Path(row["model_path"]), cal27 if name.startswith("new27_") else cal75)
             for name, row in fitted["shape_candidates"].items() if not args.selected_only or name in selected]
    cases.append(("old_fullrange75_F69", args.old_shape, old_cal))
    # All eight supplied offsets must stay within actual sampled support.
    offsets = .5*np.stack((np.cos(np.arange(8)*np.pi/4.), np.sin(np.arange(8)*np.pi/4.)), -1)
    interior = common & np.all(np.abs(states[:, None, 1:]+offsets[None]) <= 12.4+1e-9, axis=(1, 2))
    assert np.sum(interior) == 73050
    subset = np.flatnonzero(interior)
    summary = {"source_subset_states": len(states), "common_holdout_states": int(np.sum(common)),
        "controlled_gaze_common_interior_states": len(subset),
        "controlled_gaze_excluded_common_states": int(np.sum(common)-len(subset)),
        "offsets_ADE_BDE_deg": offsets.tolist(), "cases": {},
        "scope": "controlled supplied gaze only; mean-centered all9 objective; no true center supplied; no Gaussian model/no clipping"}
    with threadpool_limits(limits=6):
        for name, path, cal in cases:
            model = load_shape(path)
            begun = time.perf_counter()
            result, timing = estimate_accommodation(model, y4, states[:, 1:], mean_centered=True)
            masks = evaluation_masks(states, cal, common)
            masks["same_stress_interior"] = interior
            metrics, error = score_inverse(result, states[:, 0], masks)
            np.savez_compressed(out/f"exact_{name}.npz", source_indices=source_indices,
                truth_state=states, signed_A_error_d=error, **result)
            prediction = np.empty_like(y4)
            for low in range(0, len(states), 8192):
                high = min(low+8192, len(states))
                prediction[low:high] = model.evaluate(*states[low:high].T)["prediction"]
            centered = y4-y4[:, 4:5]
            outer = [0, 1, 2, 3, 5, 6, 7, 8]
            row = {"shape_path": str(path), "shape_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "shape_count": model.parameter_count, "exact_known_gaze": metrics, "timing": timing,
                "forward_common": forward_metrics((prediction[:, outer]-centered[:, outer])[common]),
                "wall_exact_seconds": time.perf_counter()-begun}
            if args.stress:
                total, added, directions = [], [], []
                for k, offset in enumerate(offsets):
                    supplied = states[subset, 1:]+offset
                    assert np.all(np.abs(supplied) <= 12.4+1e-9)
                    perturbed, scenario_time = estimate_accommodation(model, y4[subset], supplied, mean_centered=True)
                    err = perturbed["estimated_A_d"]-states[subset, 0]
                    extra = perturbed["estimated_A_d"]-result["estimated_A_d"][subset]
                    total.append(err)
                    added.append(extra)
                    directions.append({"offset_ADE_BDE_deg": offset.tolist(), "total_A_error": scalar_metrics(err),
                        "added_A_error_vs_exact_gaze": scalar_metrics(extra), "timing": scenario_time,
                        "nonconverged": int(np.sum(~perturbed["converged"]))})
                    log(f"{name} restricted0.5deg direction{k+1}/8 complete")
                total, added = np.asarray(total), np.asarray(added)
                row["gaze_stress"] = {"states": len(subset), "directions": directions,
                    "max_abs_A_error_across8_tested_directions": scalar_metrics(np.max(np.abs(total), axis=0)),
                    "max_added_abs_A_error_across8_tested_directions": scalar_metrics(np.max(np.abs(added), axis=0)),
                    "scope": "finite8-direction comparison, not a certified continuous worst case"}
                np.savez_compressed(out/f"stress_{name}.npz", source_indices=source_indices[subset],
                    truth_state=states[subset], exact_gaze_A_d=result["estimated_A_d"][subset],
                    total_signed_A_error_d=total, added_signed_A_error_d=added, offsets_ADE_BDE_deg=offsets)
            summary["cases"][name] = row
            write_json(out/"summary.json", summary)
            log(f"{name} limited supplied-gaze controls complete")


if __name__ == "__main__":
    main()
