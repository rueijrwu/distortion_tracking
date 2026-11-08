"""Exploratory P1 consistency diagnostics; never filters primary inference."""
from __future__ import annotations

import argparse
import hashlib
import pickle
from pathlib import Path

import numpy as np

from fit_p4_limited_gaze import DEFAULT_OUT
from fit_p4_sparse_calibration import scalar_metrics, write_json
from evaluate_p4_p1_centroid_iteration import P1_PATH, P1_SHA, gaze_metrics

ROOT = Path(__file__).resolve().parents[1]
OLD_FLAG_ARTIFACT = ROOT/"data/p4_p1_cubic_center_full/p1_grid_quality_flags_D15.json"
OLD_FLAG_SHA = "ff951d328e05b10248ad276246a56e502466f78c3e16fb59f4933a7e147f6dca"
OLD_PAIR_MM, OLD_CENTER_MM = .05732008242582714, .09006296602113578
EQUALITY_TOL_MM = 1e-12


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    runtime = args.runtime_dir or args.calibration_dir/"joint_runtime"
    out = args.output_dir or args.calibration_dir/"quality_diagnostics"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    with np.load(args.calibration_dir/"source_subset.npz") as saved:
        states, source_indices, cal75, common = (saved[key] for key in ("states", "source_indices", "cal75", "common_holdout"))
    if hashlib.sha256(OLD_FLAG_ARTIFACT.read_bytes()).hexdigest() != OLD_FLAG_SHA:
        raise RuntimeError("Frozen full-range flag artifact changed")
    raw = P1_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != P1_SHA:
        raise RuntimeError("Pinned P1 source changed")
    p1 = pickle.loads(raw)
    ade, bde = np.unique(p1["eye_rotation_deg"]), np.unique(p1["eye_rotation_x_deg"])
    ai, bi = np.searchsorted(ade, p1["eye_rotation_deg"]), np.searchsorted(bde, p1["eye_rotation_x_deg"])
    order = np.lexsort((p1["field_y_relative"], p1["field_x_relative"], ai, bi))
    p1 = p1[order]
    xy = np.stack((p1["real_x_mm"]+p1["center_absolute_x_mm"],
                   p1["real_y_mm"]+p1["center_absolute_y_mm"]), -1).reshape(len(bde), len(ade), 9, 2)
    if not np.all(np.isfinite(xy)):
        raise RuntimeError("P1 quality means require all nine finite fields")
    midpoint = .5*(xy[:, :, [0, 1, 2, 3], :]+xy[:, :, [8, 7, 6, 5], :])
    mean = np.mean(midpoint, axis=2)
    q_pair = np.sqrt(np.mean(np.sum((midpoint-mean[:, :, None, :])**2, axis=3), axis=2))
    q_center = np.linalg.norm(xy[:, :, 4, :]-mean, axis=2)
    ii, jj = np.searchsorted(bde, states[:, 2]), np.searchsorted(ade, states[:, 1])
    pair, center = q_pair[ii, jj], q_center[ii, jj]
    new_pair, new_center = float(np.max(pair[cal75])), float(np.max(center[cal75]))
    old_flag = (pair > OLD_PAIR_MM+EQUALITY_TOL_MM) | (center > OLD_CENTER_MM+EQUALITY_TOL_MM)
    new_flag = (pair > new_pair+EQUALITY_TOL_MM) | (center > new_center+EQUALITY_TOL_MM)
    output = {"scope": "Exploratory fixed diagnostics; all-frame primary data are unchanged; no operational guarantee",
        "equality_tolerance_mm": EQUALITY_TOL_MM,
        "old_thresholds": {"pair_mm": OLD_PAIR_MM, "center_mm": OLD_CENTER_MM,
            "source": str(OLD_FLAG_ARTIFACT), "source_sha256": OLD_FLAG_SHA},
        "new75_calibration_only_thresholds": {"pair_mm": new_pair, "center_mm": new_center,
            "calibration_rows": int(np.sum(cal75)), "unique_gaze_poses": len(np.unique(states[cal75, 1:], axis=0))},
        "common_states": int(np.sum(common)), "old_flagged_common_states": int(np.sum(old_flag & common)),
        "new_flagged_common_states": int(np.sum(new_flag & common)), "models": {}}
    np.savez_compressed(out/"quality_flags.npz", source_indices=source_indices, q_pair_mm=pair,
        q_center_mm=center, old_fullrange_flag=old_flag, new_calibration_flag=new_flag, common_holdout=common)
    lookup = np.full(21*101*101, -1, int)
    lookup[source_indices] = np.arange(len(source_indices))
    for path in sorted(runtime.glob("runtime_*.npz")):
        with np.load(path) as saved:
            where = lookup[saved["source_indices"]]
            if np.any(where < 0):
                raise RuntimeError("Runtime frame outside fixed source subset")
            truth, estimate = saved["truth_state"], saved["estimated_state"]
            error = estimate-truth
        total_A_sse, total_g_sse = float(np.sum(error[:, 0]**2)), float(np.sum(error[:, 1:]**2))
        rows = {}
        for flag_name, flag in (("old_fullrange", old_flag[where]), ("new75_calibration", new_flag[where])):
            groups = {}
            for group, mask in (("all", np.ones(len(where), bool)), ("unflagged", ~flag), ("flagged", flag)):
                if np.any(mask):
                    groups[group] = {"states": int(np.sum(mask)), "coverage": float(np.mean(mask)),
                        "A_d": scalar_metrics(error[mask, 0]), "gaze": gaze_metrics(error[mask, 1:]),
                        "A_squared_error_share": float(np.sum(error[mask, 0]**2)/total_A_sse) if total_A_sse else 0.,
                        "gaze_squared_error_share": float(np.sum(error[mask, 1:]**2)/total_g_sse) if total_g_sse else 0.}
            rows[flag_name] = groups
        output["models"][path.stem] = rows
    for label, flag in (("old", old_flag), ("new", new_flag)):
        gazes = np.unique(states[flag & common, 1:], axis=0)
        output[f"{label}_flagged_gaze_ADE_BDE_deg"] = gazes.tolist()
    write_json(out/"quality_diagnostics.json", output)


if __name__ == "__main__":
    main()
