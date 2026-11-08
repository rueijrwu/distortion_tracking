"""Optional matched1um raw-calibration noise for ONLY the separate-axis cubics."""
from __future__ import annotations

import argparse
import hashlib
import math
import pickle
from pathlib import Path

import numpy as np

from fit_p4_p1_axis_cubic_gaze import AxisCubic, fit, metrics, DEFAULT_OUT, CALIBRATION_ROOT, P1_PATH, P1_SHA
from fit_p4_sparse_calibration import write_json


def raw_coefficients(beta, mean, scale):
    result = np.zeros(4)
    for degree in range(4):
        for power in range(degree+1):
            result[power] += beta[degree]*math.comb(degree, power)*(-mean)**(degree-power)/scale**degree
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, default=CALIBRATION_ROOT)
    parser.add_argument("--fit-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--draws", type=int, default=100)
    args = parser.parse_args()
    if args.draws < 2:
        parser.error("At least2 draws are required")
    out = args.output_dir or args.fit_dir/f"noise_{args.draws}"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    source = args.calibration_dir/"source_subset.npz"
    with np.load(source) as saved:
        states, y4, m1, common, cal27, cal75, indices = (saved[key] for key in
            ("states", "absolute_p4_mm", "mean_p1_absolute_mm", "common_holdout", "cal27", "cal75", "source_indices"))
    rng_sample = np.random.default_rng(20261107)
    selected = np.sort(np.concatenate([rng_sample.choice(np.flatnonzero(common & np.isclose(states[:, 0], a)), 100, replace=False)
                                      for a in np.unique(states[:, 0])]))
    assert len(selected) == 2100 and np.all(common[selected])
    raw = P1_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != P1_SHA:
        raise RuntimeError("P1 input changed")
    p1 = pickle.loads(raw)
    ade, bde = np.unique(p1["eye_rotation_deg"]), np.unique(p1["eye_rotation_x_deg"])
    ai, bi = np.searchsorted(ade, p1["eye_rotation_deg"]), np.searchsorted(bde, p1["eye_rotation_x_deg"])
    p1 = p1[np.lexsort((p1["field_y_relative"], p1["field_x_relative"], ai, bi))]
    absolute1 = np.stack((p1["real_x_mm"]+p1["center_absolute_x_mm"], p1["real_y_mm"]+p1["center_absolute_y_mm"]), -1).reshape(101, 101, 9, 2)
    unique_gaze, inverse = np.unique(states[cal75, 1:], axis=0, return_inverse=True)
    unique1 = absolute1[np.searchsorted(bde, unique_gaze[:, 1]), np.searchsorted(ade, unique_gaze[:, 0])]
    assert len(unique1) == 25 and np.allclose(np.mean(unique1, axis=1)[inverse], m1[cal75], atol=1e-12, rtol=0)
    sub27 = cal27[cal75]
    clean_d = np.mean(y4[selected], axis=1)-m1[selected]
    targets = states[cal75, 1:]
    results = {budget: {"prediction": [], "clipped": [], "coefficients": [], "normalizers": [],
                        "ranks": [], "conditions": [], "success": []} for budget in (27, 75)}
    noise = np.random.default_rng(20261108)
    for draw in range(args.draws):
        noisy4 = y4[cal75]+noise.normal(0., .001, y4[cal75].shape)
        noisy1 = unique1+noise.normal(0., .001, unique1.shape)
        noisy_d = np.mean(noisy4, axis=1)-np.mean(noisy1, axis=1)[inverse]
        for budget, take in ((27, sub27), (75, np.ones(75, bool))):
            model, identity = fit(noisy_d[take], targets[take])
            pred = model.predict(clean_d)
            record = results[budget]
            record["prediction"].append(pred)
            record["clipped"].append(np.clip(pred, -12.5, 12.5))
            record["coefficients"].append(np.r_[raw_coefficients(model.x_to_BDE, model.mean_mm[0], model.scale_mm[0]),
                                                       raw_coefficients(model.y_to_ADE, model.mean_mm[1], model.scale_mm[1])])
            record["normalizers"].append(np.r_[model.mean_mm, model.scale_mm])
            record["ranks"].append([identity[k]["SVD_rank"] for k in ("detector_x_to_BDE", "detector_y_to_ADE")])
            record["conditions"].append([identity[k]["normalized_design"]["condition"] for k in ("detector_x_to_BDE", "detector_y_to_ADE")])
            record["success"].append(identity["full_rank"] and np.all(np.isfinite(pred)))
        if draw in (0, 4) or (draw+1) % 25 == 0:
            print(f"separate-axis cubic matched raw1um draw{draw+1}/{args.draws}", flush=True)
    summary = {"draws": args.draws, "evaluation_states": 2100, "selection_seed": 20261107, "noise_seed": 20261108,
        "sample_source_indices": indices[selected].tolist(), "source_subset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "raw_coordinate_sigma_um": 1., "P1_noise_policy": "one independent9-point draw per unique calibration gaze, reused across3A and both budgets",
        "P4_noise_policy": "independent raw9-point draws per75 calibration state, overlapping27 observations shared acrossbudgets",
        "scope": "calibration-noise-only; noiseless test means/ground-truth labels; no runtime sensor uncertainty; no family selection", "cases": {}}
    for budget in (27, 75):
        record = {key: np.asarray(value) for key, value in results[budget].items()}
        pred, clipped, coefficients = record["prediction"], record["clipped"], record["coefficients"]
        truth = states[selected, 1:]
        baseline = AxisCubic.load(args.fit_dir/f"axis_cubic_{budget}.npz").predict(clean_d)
        error = pred-truth[None]
        draw_rmse = np.sqrt(np.mean(np.sum(error**2, axis=2), axis=1))
        sd = np.std(coefficients, axis=0, ddof=1)
        covariance = np.cov(coefficients, rowvar=False, ddof=1)
        correlation = np.divide(covariance, sd[:, None]*sd[None], out=np.zeros_like(covariance), where=sd[:, None]*sd[None] > 0)
        np.savez_compressed(out/f"axis_cubic_noise_{budget}.npz", source_indices=indices[selected], truth_state=states[selected],
            baseline_gaze=baseline, coefficient_SD_raw_mm_basis=sd, coefficient_correlation_raw_mm_basis=correlation, **record)
        summary["cases"][str(budget)] = {"coefficients": 8, "failed_or_rank_deficient_draws": int(np.sum(~record["success"])),
            "rank_x_BDE_y_ADE_by_draw": record["ranks"].tolist(), "normalized_condition_x_y_by_draw": record["conditions"].tolist(),
            "pooled_unbounded_metrics": metrics(pred.reshape(-1, 2), np.tile(truth, (args.draws, 1))),
            "pooled_clipped_metrics": metrics(clipped.reshape(-1, 2), np.tile(truth, (args.draws, 1))),
            "mean_draw_radial_RMSE_deg": float(np.mean(draw_rmse)), "SD_draw_radial_RMSE_deg": float(np.std(draw_rmse, ddof=1)),
            "prediction_SD_ADE_BDE_deg_mean": np.mean(np.std(pred, axis=0, ddof=1), axis=0).tolist(),
            "coefficient_SD_common_raw_mm_basis": sd.tolist(), "coefficient_correlation_common_raw_mm_basis": correlation.tolist(),
            "noiseless_same_sample_metrics": metrics(baseline, truth)}
    write_json(out/"summary.json", summary)


if __name__ == "__main__":
    main()
