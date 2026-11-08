"""Predeclared69-value comparison on the75 physical P4 calibration states."""
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

from p4_sparse_model_69 import SparseModel69, fit_model
from p4_known_gaze_inverse import estimate_accommodation
from fit_p4_sparse_calibration import (calibration_masks, identifiability, forward_metrics,
    scalar_metrics, fixed_subset, evaluate_known, write_json, log)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT/"data/p4_sparse_calibration"
DEFAULT_OUT = ROOT/"data/p4_sparse_calibration_69"


def cv69(states, observed):
    folds, losses = [], []
    for gaze in np.unique(states[:, 1:], axis=0):
        if np.all(np.isclose(gaze, 0., atol=1e-10)):
            continue
        held = np.all(np.isclose(states[:, 1:], gaze, atol=1e-10), axis=1)
        model, info = fit_model(states[~held], observed[~held])
        rank = identifiability(model, states[~held])["full_scaled_rank"]
        prediction = model.evaluate(*states[held].T)["prediction"][:, [0, 1, 2, 3, 5, 6, 7, 8]]
        loss = float(np.mean(((prediction-observed[held][:, [0, 1, 2, 3, 5, 6, 7, 8]])*1000.)**2))
        losses.append(loss)
        folds.append({"gaze_ADE_BDE": gaze.tolist(), "coordinate_mse_um2": loss,
            "heldout_states": int(np.sum(held)), "rank": rank,
            "success": bool(info["calibration_success"] and info["gaze_success"] and rank == 69)})
    return {"mean_coordinate_mse_um2": float(np.mean(losses)),
        "se_coordinate_mse_um2": float(np.std(losses, ddof=1)/np.sqrt(len(losses))),
        "folds": folds, "all_folds_full_rank_success": all(row["success"] for row in folds),
        "selection_data": "75calibrationstates only; threezero-gazeanchors retained"}


def noise69(model, states, relative, centers, cal, out, draws):
    rng = np.random.default_rng(20261107+75)
    chosen = fixed_subset(states, ~cal, 3, 20261107)[:64]
    baseline, _ = estimate_accommodation(model, relative[chosen], states[chosen, 1:], mean_centered=True)
    raw = relative[cal]+centers[cal, None]
    estimates, parameters, successes, times = [], [], [], []
    for i in range(draws):
        start = time.perf_counter()
        noisy = raw+rng.normal(0., .001, raw.shape)
        centered = noisy-noisy[:, 4:5]
        fitted, info = fit_model(states[cal], centered)
        inverse, _ = estimate_accommodation(fitted, relative[chosen], states[chosen, 1:], mean_centered=True)
        estimates.append(inverse["estimated_A_d"])
        parameters.append(np.r_[fitted.calibration, fitted.coefficients])
        successes.append(info["gaze_success"] and info["calibration_success"] and np.all(inverse["converged"]))
        times.append(time.perf_counter()-start)
        if i in (0, 4) or (i+1) % 25 == 0:
            log(f"69candidate illustrative raw1um calibration-noise draw{i+1}/{draws};projected{np.mean(times)*draws:.1f}s")
    estimates, parameters = np.asarray(estimates), np.asarray(parameters)
    np.savez_compressed(out/"shape_calibration_noise_75_69.npz", selection_indices=chosen,
        truth_state=states[chosen], estimated_A_d=estimates, parameters=parameters,
        fit_success=np.array(successes), exact_calibration_A_d=baseline["estimated_A_d"],
        raw_point_sigma_mm=np.array(.001))
    return {"draws": draws, "states": len(chosen), "fit_failures": int(np.sum(~np.asarray(successes))),
        "wall_seconds": float(np.sum(times)), "mean_state_std_A_d": float(np.mean(np.std(estimates, axis=0, ddof=1))),
        "p95_state_std_A_d": float(np.percentile(np.std(estimates, axis=0, ddof=1), 95)),
        "total_A_error": scalar_metrics((estimates-states[chosen, 0]).ravel()),
        "added_A_error": scalar_metrics((estimates-baseline["estimated_A_d"]).ravel()),
        "empirical_parameter_correlation_matrix": np.corrcoef(parameters, rowvar=False).tolist(),
        "scope": "hypothetical shape-calibration noise only; one noisyrawreference perreplicate, shared consistently"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-fit", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--exact-only", action="store_true")
    parser.add_argument("--noise-draws", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    out = args.output_dir
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=6):
        with np.load(args.source_fit/"source_tensors.npz") as source:
            states, relative, centers = source["states"], source["relative_mm"], source["center_absolute_mm"]
            grid_sha = str(source["grid_sha256"])
        observed = relative-relative[:, 4:5]
        masks = calibration_masks(75)
        cal = masks["calibration"]
        model_path = out/"cal75_projective_first_69.npz"
        if args.resume and model_path.exists():
            model = SparseModel69.load(model_path)
            row = json.loads((out/"summary.json").read_text(encoding="utf-8"))["candidate69"]
        else:
            start = time.perf_counter()
            model, info = fit_model(states[cal], observed[cal])
            cv = cv69(states[cal], observed[cal])
            rank = identifiability(model, states[cal])
            model.save(model_path, grid_sha256=np.array(grid_sha), calibration_states=states[cal], calibration_indices=np.flatnonzero(cal))
            prediction = np.empty_like(observed)
            regularity = {name: np.inf for name in ("denominator", "radial_tangential", "radial_radial", "brown_determinant")}
            for low in range(0, len(states), 8192):
                high = min(low+8192, len(states))
                part = model.evaluate(*states[low:high].T, diagnostics=True)
                prediction[low:high] = part["prediction"]
                for key in regularity:
                    regularity[key] = min(regularity[key], float(np.min(part[key])))
            outer = [0, 1, 2, 3, 5, 6, 7, 8]
            error = prediction[:, outer]-observed[:, outer]
            row = {"parameter_count": 69, "fit": info, "cv": cv, "identifiability": rank,
                "forward": {name: forward_metrics(error[take]) for name, take in masks.items()},
                "regularity": regularity, "fit_CV_forward_seconds": time.perf_counter()-start}
            np.savez_compressed(out/"forward_69.npz", prediction_mm=prediction, calibration_mask=cal, truth_state=states)
        prior = json.loads((args.source_fit/"summary.json").read_text(encoding="utf-8"))
        candidates = {name: item for name, item in prior["candidates"].items() if name.startswith("cal75_")}
        candidates["cal75_projective_first_69"] = row
        eligible = [name for name, item in candidates.items() if item["identifiability"]["full_scaled_rank"] == item["parameter_count"] and item["cv"]["all_folds_full_rank_success"] and item["fit"]["gaze_success"]]
        best = min(eligible, key=lambda name: candidates[name]["cv"]["mean_coordinate_mse_um2"])
        threshold = candidates[best]["cv"]["mean_coordinate_mse_um2"]+candidates[best]["cv"]["se_coordinate_mse_um2"]
        selected = min([name for name in eligible if candidates[name]["cv"]["mean_coordinate_mse_um2"] <= threshold],
            key=lambda name: (candidates[name]["parameter_count"], candidates[name]["cv"]["mean_coordinate_mse_um2"]))
        summary = {"candidate69": row, "calibration_only_comparison": {"candidates": candidates,
            "selected_name": selected, "one_SE_threshold_coordinate_mse_um2": threshold},
            "provenance": {"grid_sha256": grid_sha,
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "model_sha256": hashlib.sha256((ROOT/"Script/p4_sparse_model_69.py").read_bytes()).hexdigest(),
                "predeclaration": "33symmetricgaze terms x{1,a}+3TRAIN-onlyon-axis;75poses;no heldout-driven terms"}}
        write_json(out/"summary.json", summary)
        log(f"69candidate:rank{row['identifiability']['full_scaled_rank']}/69,CVmse{row['cv']['mean_coordinate_mse_um2']:.6g};cal-onlyselection{selected}")
        common = np.all(np.abs(states[:, 1:]) <= 19.2+1e-9, axis=1) & ~cal
        scenario_sample = None if args.exact_only else fixed_subset(states, common, 1400, 20261107)
        known, _ = evaluate_known(model, "cal75_projective_first_69", 75, states, observed, masks, out, scenario_sample, args.resume)
        summary["known_gaze"] = known
        if args.noise_draws and not args.exact_only:
            summary["shape_calibration_noise"] = noise69(model, states, relative, centers, cal, out, args.noise_draws)
        write_json(out/"summary.json", summary)
        q = known["metrics"]["heldout"]
        text = ("# Predeclared75-pose69-coefficient R1 comparison\n\n"
            "This retains the same direct-reference spatial composition and three calibration-only on-axis values. It adds identifiable fourth-order symmetric gaze terms on the5×5 calibration lattice; all starts are identity/zero. The previous27/39 families and their results remain unchanged.\n\n"
            f"Calibration-only one-SE selection after including this candidate: `{selected}`. Full scaled rank: {row['identifiability']['full_scaled_rank']}/69; scaled condition: {row['identifiability']['scaled_condition']:.6g}.\n\n"
            f"All{q['attempted']:,}heldout exact-supplied-gaze states: A RMSE{q['rmse']:.6g}D, P95abs{q['p95_abs']:.6g}D, {100*q['within_0_1D_fraction_all_attempted']:.3f}%within±0.1D. This comparison line is not a pass/fail gate or a resolution guarantee.\n\n"
            "The corrected-center/gaze-map coefficients used by an iterative deployment must be counted separately. Scenario/noise arrays and source/rank/CV evidence are preserved alongside summary.json.\n")
        (out/"report.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
