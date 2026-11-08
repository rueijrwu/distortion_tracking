"""Matched hypothetical raw1um shape-calibration sensitivity in limited gaze."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from fit_p4_limited_gaze import DEFAULT_OUT
from evaluate_p4_p1_limited_gaze import load_shape
from p4_sparse_model import fit_model
from p4_sparse_model_69 import fit_model as fit69
from p4_known_gaze_inverse import estimate_accommodation
from fit_p4_sparse_calibration import scalar_metrics, write_json, log, identifiability


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--draws", type=int, default=5)
    args = parser.parse_args()
    if args.draws < 2:
        parser.error("At least2 draws are needed for variance")
    out = args.output_dir or args.calibration_dir/f"noise_{args.draws}_draws"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    fitted = json.loads((args.calibration_dir/"summary.json").read_text(encoding="utf-8"))
    with np.load(args.calibration_dir/"source_subset.npz") as saved:
        states, y4, source_indices, common, cal = (saved[key] for key in
            ("states", "absolute_p4_mm", "source_indices", "common_holdout", "cal75"))
    rng_selection = np.random.default_rng(20261107)
    chosen = np.sort(np.concatenate([rng_selection.choice(np.flatnonzero(common & np.isclose(states[:, 0], a)), 100, replace=False)
                                    for a in np.unique(states[:, 0])]))
    assert len(chosen) == 2100
    names = list(dict.fromkeys((fitted["shape_selection"]["75_low"]["simplest_full_rank"],
                               fitted["shape_selection"]["75"]["selected"])))
    summary = {"draws": args.draws, "raw_point_coordinate_sigma_um": 1.,
        "evaluation_states": 2100, "selection_seed": 20261107, "noise_seed": 20261108,
        "scope": "hypothetical shape-calibration-only noise; exact supplied gaze/noiseless test; excludes D,P1 and runtime noise",
        "reference_policy": "one noisy A0zero-gaze reference per replicate shared everywhere; each noisy central point subtracted from every field in its state",
        "family_selection": "frozen before noise; simplest full-rank low-order and selected75-pose shape; no reselection",
        "models": {}}
    with threadpool_limits(limits=6):
        for name in names:
            path = Path(fitted["shape_candidates"][name]["model_path"])
            model = load_shape(path)
            exact, _ = estimate_accommodation(model, y4[chosen], states[chosen, 1:], mean_centered=True)
            baseline_prediction = model.evaluate(*states[chosen].T)["prediction"]
            noise_rng = np.random.default_rng(20261108)
            estimates, predictions, coefficients, successes, invalid, timings = [], [], [], [], [], []
            for draw in range(args.draws):
                begun = time.perf_counter()
                noisy_raw = y4[cal]+noise_rng.normal(0., .001, y4[cal].shape)
                noisy_centered = noisy_raw-noisy_raw[:, 4:5]
                refit, info = (fit69(states[cal], noisy_centered) if model.parameter_count == 69 else
                    fit_model(states[cal], noisy_centered, model.accommodation_degree, model.composition))
                inverse, _ = estimate_accommodation(refit, y4[chosen], states[chosen, 1:], mean_centered=True)
                prediction = refit.evaluate(*states[chosen].T, diagnostics=True)
                valid = np.all(np.stack([prediction[key] > 0 for key in
                    ("denominator", "radial_tangential", "radial_radial", "brown_determinant")]), axis=(0, 2))
                estimates.append(inverse["estimated_A_d"])
                predictions.append(prediction["prediction"])
                coefficients.append(np.r_[refit.calibration, refit.coefficients])
                successes.append(info["gaze_success"] and info["calibration_success"] and np.all(inverse["converged"]))
                invalid.append(int(np.sum(~valid)))
                timings.append(time.perf_counter()-begun)
                if draw in (0, 4) or (draw+1) % 25 == 0:
                    log(f"{name} matched raw1um draw{draw+1}/{args.draws}; projected100 draws{np.mean(timings)*100:.1f}s")
            estimates, predictions, coefficients = np.asarray(estimates), np.asarray(predictions), np.asarray(coefficients)
            A_sd = np.std(estimates, axis=0, ddof=1)
            coordinate_sd = np.std(predictions, axis=0, ddof=1)
            point_sd = np.linalg.norm(coordinate_sd, axis=2)*1000.
            coefficient_sd = np.std(coefficients, axis=0, ddof=1)
            covariance = np.cov(coefficients, rowvar=False, ddof=1)
            correlation = np.divide(covariance, coefficient_sd[:, None]*coefficient_sd[None, :],
                out=np.zeros_like(covariance), where=coefficient_sd[:, None]*coefficient_sd[None, :] > 0)
            np.savez_compressed(out/f"{name}.npz", source_indices=source_indices[chosen], truth_state=states[chosen],
                estimated_A_d=estimates, predicted_coordinates_mm=predictions, coefficients=coefficients,
                coefficient_labels=np.array(["s1", "k1", "k2"]+model.labels), fit_success=np.array(successes),
                invalid_sampled_map_states=np.array(invalid), coefficient_sd=coefficient_sd,
                coefficient_covariance=covariance, coefficient_correlation=correlation,
                exact_calibration_A_d=exact["estimated_A_d"], exact_calibration_prediction_mm=baseline_prediction)
            summary["models"][name] = {"shape_values": model.parameter_count,
                "fit_failures": int(np.sum(~np.asarray(successes))), "invalid_map_states_by_draw": invalid,
                "wall_seconds": float(np.sum(timings)), "projected100_draw_seconds": float(np.mean(timings)*100),
                "A_error_all_draws": scalar_metrics((estimates-states[chosen, 0]).ravel()),
                "added_A_error_vs_exact_calibration": scalar_metrics((estimates-exact["estimated_A_d"]).ravel()),
                "A_estimate_SD_D": {"mean": float(np.mean(A_sd)), "p95": float(np.percentile(A_sd, 95)), "max": float(np.max(A_sd))},
                "predicted_point_SD_um": {"rms": float(np.sqrt(np.mean(point_sd**2))), "p95": float(np.percentile(point_sd, 95)), "max": float(np.max(point_sd))},
                "coefficient_SD": coefficient_sd.tolist(), "coefficient_correlation": correlation.tolist(),
                "noise_free_identifiability": identifiability(model, states[cal]),
                "interpretation": "raw and scaled rank, coefficient variability and prediction/A variability are reported together; coefficient cancellation is possible within range"}
            write_json(out/"summary.json", summary)


if __name__ == "__main__":
    main()
