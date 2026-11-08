"""Fit C8 on the calibrated central P4 ray minus measured P1 mean.

This is a NEW observable-specific map. The prior raw-mean cubic is not reused.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from fit_p4_limited_gaze import DEFAULT_OUT as SOURCE_ROOT
from fit_p4_p1_axis_cubic_gaze import fit, cv, metrics
from fit_p4_sparse_calibration import write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT/"data/p4_p1_center_cubic_integration"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT/"calibration")
    args = parser.parse_args()
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    source = args.source_dir/"source_subset.npz"
    with np.load(source) as saved:
        states, y4, m1, trace_center, common, cal27, cal75, indices = (saved[key] for key in
            ("states", "absolute_p4_mm", "mean_p1_absolute_mm", "center_absolute_mm", "common_holdout", "cal27", "cal75", "source_indices"))
    assert len(states) == 83349 and np.sum(common) == 83250
    central_ray = y4[:, 4]
    q = central_ray-m1
    d = np.mean(y4, axis=1)-m1
    summary = {"source_subset_path": str(source), "source_subset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "calibration_feature": "q=absolute labeled(0,0)P4 gridray(index4)-mean(all9 absoluteP1)",
        "coefficients": 8, "runtime_A_input": False, "runtime_truecenter_input": False,
        "reference_trace_difference_max_mm": float(np.max(np.linalg.norm(central_ray-trace_center, axis=1))), "budgets": {}}
    for budget, cal in ((27, cal27), (75, cal75)):
        assert np.sum(cal) == budget
        model, identity = fit(q[cal], states[cal, 1:])
        cv_info, cv_prediction = cv(q[cal], states[cal, 1:])
        path = out/f"corrected_axis_cubic_{budget}.npz"
        np.savez_compressed(path, mean_mm=model.mean_mm, scale_mm=model.scale_mm,
            x_to_BDE=model.x_to_BDE, y_to_ADE=model.y_to_ADE, powers=np.arange(4), parameter_count=np.array(8),
            input_order=np.array(["corrected_center_minus_P1_x_mm", "corrected_center_minus_P1_y_mm"]),
            output_order=np.array(["ADE_deg", "BDE_deg"]), calibration_source_indices=indices[cal],
            calibration_states=states[cal], calibration_q_mm=q[cal], ground_truth_gaze_deg=states[cal, 1:])
        oracle = model.predict(q)
        initial = model.predict(d)
        summary["budgets"][str(budget)] = {"model_path": str(path), "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "fit": identity, "calibration_CV": cv_info,
            "calibration_metrics": metrics(model.predict(q[cal]), states[cal, 1:]),
            "scoring_only_oracle_central_ray_input": metrics(oracle[common], states[common, 1:]),
            "deliberately_uncorrected_rawmean_initialization": metrics(initial[common], states[common, 1:])}
        np.savez_compressed(out/f"calibration_diagnostics_{budget}.npz", source_indices=indices,
            oracle_gaze_ADE_BDE_deg=oracle, rawmean_initial_gaze_ADE_BDE_deg=initial,
            calibration_CV_prediction=cv_prediction, common_holdout_mask=common)
        write_json(out/"summary.json", summary)
        print(f"corrected C8 cal{budget}: rankfull={identity['full_rank']}; calibrated feature distinct fromrawmean", flush=True)


if __name__ == "__main__":
    main()
