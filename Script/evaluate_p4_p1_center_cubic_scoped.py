"""Explicit per-case scopes for the frozen corrected-C8 fixed-point baseline."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from fit_p4_limited_gaze import DEFAULT_OUT as SOURCE_ROOT
from fit_p4_p1_corrected_axis_cubic import DEFAULT_ROOT
from fit_p4_p1_axis_cubic_gaze import AxisCubic, frozen_quality_flags
from evaluate_p4_p1_limited_gaze import load_shape
from evaluate_p4_p1_center_cubic_iteration import run, score
from fit_p4_sparse_calibration import write_json

CASES = {"cal75_F27_C8": ("new75_DF27.npz", 75, True),
         "cal27_F27_C8": ("new27_DF27.npz", 27, False),
         "cal75_F69_C8": ("new75_PF69.npz", 75, False)}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(CASES), required=True)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--cubic-dir", type=Path, default=DEFAULT_ROOT/"calibration")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk", type=int, default=512)
    args = parser.parse_args()
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=6):
        source = args.source_dir/"source_subset.npz"
        with np.load(source) as stored:
            states, y4, m1, common, indices = (stored[key] for key in
                ("states", "absolute_p4_mm", "mean_p1_absolute_mm", "common_holdout", "source_indices"))
        assert np.sum(common) == 83250
        rng = np.random.default_rng(20261107)
        random = np.sort(np.concatenate([rng.choice(np.flatnonzero(common & np.isclose(states[:, 0], a)), 1000, replace=False)
                                         for a in np.unique(states[:, 0])]))
        assert len(random) == 21000 and len(np.unique(random)) == 21000
        flagged_mask = common & frozen_quality_flags(states)
        flagged = np.flatnonzero(flagged_mask)
        filename, budget, full = CASES[args.case]
        selection = np.flatnonzero(common) if full else np.unique(np.r_[random, flagged])
        np.savez_compressed(out/"frozen_selection.npz", random21000_source_indices=indices[random],
            flagged_source_indices=indices[flagged], evaluated_source_indices=indices[selection])
        shape_path, C_path = args.source_dir/"models"/filename, args.cubic_dir/f"corrected_axis_cubic_{budget}.npz"
        model, C = load_shape(shape_path), AxisCubic.load(C_path)
        result, seconds = run(model, C, y4, m1, selection, indices, out, args.case, args.chunk)
        oracle = C.predict(y4[selection, 4]-m1[selection])  # scoring only, never passed into run
        np.savez_compressed(out/f"runtime_{args.case}.npz", source_indices=indices[selection], truth_state=states[selection],
            true_central_ray_mm=y4[selection, 4], signed_state_error=result["estimated_state"]-states[selection], **result)
        random_take = np.isin(selection, random)
        flag_take = flagged_mask[selection]
        endpoints = np.isclose(states[selection, 0], 0.) | np.isclose(states[selection, 0], 4.)
        masks = {"full_common" if full else "evaluated_union_not_population": np.ones(len(selection), bool),
                 "random21000_common": random_take, "flagged_all": flag_take,
                 "random_interior_A": random_take & ~endpoints, "random_endpoint_A": random_take & endpoints}
        if full:
            masks.update(full_interior_A=~endpoints, full_endpoint_A=endpoints, full_unflagged=~flag_take)
        summary = {"case": args.case, "shape_values": model.parameter_count, "center_cubic_values": 8,
            "total_values": model.parameter_count+8, "shape_sha256": digest(shape_path), "center_cubic_sha256": digest(C_path),
            "source_subset_sha256": digest(source), "frozen_solver_sha256": digest(Path(__file__).with_name("p4_center_cubic_fixedpoint.py")),
            "driver_sha256": digest(__file__), "full_common": full, "evaluated_states": len(selection),
            "population_common_states": 83250, "random_states": 21000, "random_seed": 20261107,
            "flagged_stress_states": len(flagged), "wall_seconds": seconds,
            "scope_note": "Random balanced1000/A estimates and complete flagged stress set are separate; pooled oversampled union is not a population estimate",
            "metrics": {name: score(result, states[selection], y4[selection, 4], oracle, mask) for name, mask in masks.items() if np.any(mask)}}
        write_json(out/"summary.json", summary)


if __name__ == "__main__":
    main()
