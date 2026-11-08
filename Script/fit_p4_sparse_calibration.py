"""Sparse P4 calibration: 27/75 physical states, shared models, blind A search."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "6"

import numpy as np
from threadpoolctl import threadpool_limits

from p4_sparse_model import SparseModel, fit_model, OUTER
from p4_known_gaze_inverse import estimate_accommodation

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/"data/distortion_grid_p4_biaxial_macro/distortion_grid.pkl"
DEFAULT_OUT = ROOT/"data/p4_sparse_calibration"
CAL_A = np.array([0, 10, 20])
CAL_GAZE = {27: np.array([0, 50, 100]), 75: np.array([0, 25, 50, 75, 100])}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def log(text):
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def load_source(path=SOURCE):
    raw = Path(path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    data = pickle.loads(raw)
    del raw
    A, ADE, BDE = (np.unique(data[name]) for name in ("accommodation_d", "eye_rotation_deg", "eye_rotation_x_deg"))
    assert (len(A), len(ADE), len(BDE), len(data)) == (21, 101, 101, 1927989)
    ai, bi, di = (np.searchsorted(axis, data[name]) for axis, name in
                  ((A, "accommodation_d"), (BDE, "eye_rotation_x_deg"), (ADE, "eye_rotation_deg")))
    order = np.lexsort((data["field_y_relative"], data["field_x_relative"], di, bi, ai))
    ordered = data[order]
    shape = (21, 101, 101, 9)
    relative = np.stack((ordered["real_x_mm"], ordered["real_y_mm"]), -1).reshape(shape+(2,))
    centers = np.stack((ordered["center_absolute_x_mm"], ordered["center_absolute_y_mm"]), -1).reshape(shape+(2,))
    assert np.array_equal(centers, np.broadcast_to(centers[..., :1, :], centers.shape))
    fields = np.stack((ordered["field_x_relative"], ordered["field_y_relative"]), -1).reshape(shape+(2,))
    expected_fields = np.array([(x, y) for x in (-1., 0., 1.) for y in (-1., 0., 1.)])
    assert np.array_equal(fields, np.broadcast_to(expected_fields, fields.shape))
    probe = np.any((~np.isfinite(ordered["paraxial_x_mm"]) | ~np.isfinite(ordered["paraxial_y_mm"])).reshape(shape), axis=-1)
    assert np.all(np.isfinite(relative)) and np.all(np.isfinite(centers))
    aa, bb, dd = np.meshgrid(A, BDE, ADE, indexing="ij")
    states = np.stack((aa.ravel(), dd.ravel(), bb.ravel()), -1)
    return (A, ADE, BDE), states, relative.reshape(-1, 9, 2), centers[..., 0, :].reshape(-1, 2), probe.ravel(), sha


def calibration_masks(size):
    ia, ib, id_ = np.indices((21, 101, 101))
    seen_a = np.isin(ia, CAL_A)
    seen_gaze = np.isin(ib, CAL_GAZE[size]) & np.isin(id_, CAL_GAZE[size])
    cal = seen_a & seen_gaze
    assert np.sum(cal) == size
    return {"calibration": cal.ravel(), "seen_A_new_gaze": (seen_a & ~seen_gaze).ravel(),
            "new_A_seen_gaze": (~seen_a & seen_gaze).ravel(),
            "both_new": (~seen_a & ~seen_gaze).ravel(), "heldout": (~cal).ravel(),
            "all": np.ones(cal.size, bool)}


def scalar_metrics(values):
    attempted = len(values)
    finite = np.isfinite(values)
    v = np.asarray(values)[finite]
    if not len(v):
        return {"attempted": attempted, "finite": 0, "within_0_1D_fraction_all_attempted": 0.}
    return {"attempted": attempted, "finite": int(np.sum(finite)), "bias": float(np.mean(v)),
            "mae": float(np.mean(np.abs(v))), "rmse": float(np.sqrt(np.mean(v*v))),
            "median_abs": float(np.median(np.abs(v))), "p95_abs": float(np.percentile(np.abs(v), 95)),
            "p99_abs": float(np.percentile(np.abs(v), 99)), "max_abs": float(np.max(np.abs(v))),
            "within_0_1D_fraction_all_attempted": float(np.sum(np.abs(v) <= .1)/attempted)}


def forward_metrics(error):
    e = error*1000.
    norm = np.linalg.norm(e, axis=-1)
    return {"states": len(error), "vertex_rms_um": float(np.sqrt(np.mean(norm*norm))),
            "x_rmse_um": float(np.sqrt(np.mean(e[..., 0]**2))), "y_rmse_um": float(np.sqrt(np.mean(e[..., 1]**2))),
            "p95_point_um": float(np.percentile(norm, 95)), "max_point_um": float(np.max(norm))}


def identifiability(model, states):
    result = model.evaluate(*states.T, theta_jac=True, calibration_jac=True)
    J = np.concatenate((result["calibration_jacobian"][:, OUTER], result["theta_jacobian"][:, OUTER]), -1).reshape(-1, model.parameter_count)
    norms = np.linalg.norm(J, axis=0)
    normalized = J/np.maximum(norms, 1e-30)
    raw_sv = np.linalg.svd(J, compute_uv=False)
    sv = np.linalg.svd(normalized, compute_uv=False)
    rank = int(np.sum(sv > sv[0]*1e-10))
    covariance = np.linalg.pinv(normalized.T@normalized, rcond=1e-14)
    sd = np.sqrt(np.maximum(np.diag(covariance), 0.))
    corr = covariance/np.maximum(sd[:, None]*sd[None, :], 1e-30)
    np.fill_diagonal(corr, 0.)
    return {"columns": model.parameter_count, "rows": len(J), "full_scaled_rank": rank,
            "rank_relative_tolerance": 1e-10, "scaled_singular_values": sv.tolist(),
            "unscaled_singular_values": raw_sv.tolist(), "column_norms": norms.tolist(),
            "scaled_condition": float(sv[0]/sv[-1]), "unscaled_condition": float(raw_sv[0]/raw_sv[-1]),
            "max_abs_normal_matrix_parameter_correlation": float(np.max(np.abs(corr))),
            "normal_matrix_correlation_assumption": "geometric IID-centered-residual diagnostic; raw noise is tested separately",
            "parameter_correlation_matrix": corr.tolist()}


def candidate_cv(states, observed, degree, composition):
    gaze_locations = np.unique(states[:, 1:], axis=0)
    losses, folds = [], []
    for gaze in gaze_locations:
        if np.all(np.isclose(gaze, 0., atol=1e-10)):
            continue
        heldout = np.all(np.isclose(states[:, 1:], gaze, atol=1e-10), axis=1)
        model, info = fit_model(states[~heldout], observed[~heldout], degree, composition)
        rank = identifiability(model, states[~heldout])["full_scaled_rank"]
        success = info["gaze_success"] and info["calibration_success"] and rank == model.parameter_count
        prediction = model.evaluate(*states[heldout].T)["prediction"][:, OUTER]
        loss = float(np.mean(((prediction-observed[heldout][:, OUTER])*1000.)**2))
        losses.append(loss)
        folds.append({"gaze_ADE_BDE": gaze.tolist(), "heldout_states": int(np.sum(heldout)),
                      "coordinate_mse_um2": loss, "success": bool(success), "rank": rank})
    return {"mean_coordinate_mse_um2": float(np.mean(losses)),
            "se_coordinate_mse_um2": float(np.std(losses, ddof=1)/np.sqrt(len(losses))),
            "folds": folds, "all_folds_full_rank_success": all(row["success"] for row in folds),
            "limitation": "3x3 folds mostly test reflected repetitions; CV alone does not establish unsampled-gaze interpolation"}


def score_inverse(result, truth_A, masks):
    error = result["estimated_A_d"]-truth_A
    output = {}
    for name, mask in masks.items():
        if not np.any(mask):
            continue
        row = scalar_metrics(error[mask])
        row.update(nonconverged=int(np.sum(~result["converged"][mask])),
                   invalid_map=int(np.sum(~result["valid_map"][mask])), boundary_A=int(np.sum(result["boundary_A"][mask])))
        output[name] = row
    return output, error


def fixed_subset(states, allowed, count_per_A, seed):
    rng = np.random.default_rng(seed)
    selected = []
    for av in np.unique(states[:, 0]):
        pool = np.flatnonzero(allowed & (states[:, 0] == av))
        count = min(count_per_A, len(pool))
        # Broad coverage is fixed before any error is examined.
        edge = pool[np.max(np.abs(states[pool, 1:]), axis=1) >= np.max(np.abs(states[pool, 1:]))-1e-9]
        edge = edge[:min(len(edge), count)]
        interior = np.setdiff1d(pool, edge)
        selected.append(np.r_[edge, rng.choice(interior, count-len(edge), replace=False)])
    return np.sort(np.concatenate(selected))


def evaluate_known(model, name, size, states, observed, masks, out, scenario_sample, reuse=False):
    started = time.perf_counter()
    known_path = out/f"known_gaze_{size}.npz"
    if reuse and known_path.exists():
        with np.load(known_path) as saved:
            result = {key: saved[key].copy() for key in saved.files if key not in
                      ("truth_state", "signed_A_error_d", "supplied_gaze", "objective")}
        timing = {"reused_saved_exact_gaze_inverse": str(known_path)}
    else:
        result, timing = estimate_accommodation(model, observed, states[:, 1:], log=log, mean_centered=True)
    extra = dict(masks)
    boundary_g = np.any(np.isclose(np.abs(states[:, 1:]), 20., atol=1e-9), axis=1)
    boundary_a = np.isclose(states[:, 0], 0.) | np.isclose(states[:, 0], 4.)
    extra.update(heldout_boundary_gaze=masks["heldout"] & boundary_g,
                 heldout_interior_gaze=masks["heldout"] & ~boundary_g,
                 heldout_boundary_A=masks["heldout"] & boundary_a,
                 heldout_interior_A=masks["heldout"] & ~boundary_a)
    metrics, error = score_inverse(result, states[:, 0], extra)
    if not (reuse and known_path.exists()):
        np.savez_compressed(known_path, truth_state=states, signed_A_error_d=error,
                            supplied_gaze=states[:, 1:], objective=np.array("all9 mean-centered; true center never used"), **result)
    if scenario_sample is None:
        return {"model": name, "metrics": metrics, "exact_gaze_timing": timing,
                "exact_and_scenarios_wall_seconds": time.perf_counter()-started,
                "scenario_states": 0, "scenarios": {}}, result
    common = scenario_sample
    scenarios, all_errors, added_errors = {}, [], []
    for k in range(8):
        offset = .5*np.array([np.cos(k*np.pi/4.), np.sin(k*np.pi/4.)])
        supplied = states[common, 1:]+offset
        assert np.all(np.abs(supplied) < 20.+1e-10)
        perturbed, scenario_time = estimate_accommodation(model, observed[common], supplied, log=None, mean_centered=True)
        delta = perturbed["estimated_A_d"]-states[common, 0]
        added = perturbed["estimated_A_d"]-result["estimated_A_d"][common]
        scenarios[f"direction_{k}"] = {"offset_ADE_BDE_deg": offset.tolist(), "states": len(common),
            "total_A_error": scalar_metrics(delta), "added_vs_exact_gaze_A_error": scalar_metrics(added),
            "nonconverged": int(np.sum(~perturbed["converged"])), "timing": scenario_time}
        all_errors.append(delta)
        added_errors.append(added)
        log(f"{name} deterministic0.5deg scenario{k+1}/8 complete")
    total, added = np.asarray(all_errors), np.asarray(added_errors)
    np.savez_compressed(out/f"gaze_error_scenarios_{size}.npz", selection_indices=common,
        truth_state=states[common], total_signed_A_errors_d=total, added_signed_A_errors_d=added,
        exact_gaze_A_estimate=result["estimated_A_d"][common],
        offsets_ADE_BDE_deg=.5*np.stack((np.cos(np.arange(8)*np.pi/4.), np.sin(np.arange(8)*np.pi/4.)), -1))
    return {"model": name, "metrics": metrics, "exact_gaze_timing": timing,
            "exact_and_scenarios_wall_seconds": time.perf_counter()-started,
            "scenario_states": len(common), "scenarios": scenarios,
            "max_abs_across8_tested_directions": scalar_metrics(np.max(np.abs(total), axis=0)),
            "max_added_abs_across8_tested_directions": scalar_metrics(np.max(np.abs(added), axis=0)),
            "scenario_scope": "fixed common interior heldout states, |true gaze axes|<=19.2deg; no clipping; not a continuous-direction worst-case guarantee"}, result


def noise_sensitivity(model, size, states, relative, centers, calmask, out, draws=100):
    # One noisy raw9-point measurement per calibration state; the reference is
    # drawn ONCE per replicate and shared through the fitted model.
    rng = np.random.default_rng(20261107+size)
    selection = fixed_subset(states, ~calmask, 3, 20261107)[:64]
    exact, _ = estimate_accommodation(model, relative[selection], states[selection, 1:], mean_centered=True)
    absolute_cal = relative[calmask]+centers[calmask, None, :]
    estimates, parameters, success, seconds = [], [], [], []
    for draw in range(draws):
        start = time.perf_counter()
        noisy_raw = absolute_cal+rng.normal(0., .001, absolute_cal.shape)
        noisy_centered = noisy_raw-noisy_raw[:, 4:5, :]
        fitted, info = fit_model(states[calmask], noisy_centered, model.accommodation_degree, model.composition)
        predicted, _ = estimate_accommodation(fitted, relative[selection], states[selection, 1:], mean_centered=True)
        estimates.append(predicted["estimated_A_d"])
        parameters.append(np.r_[fitted.calibration, fitted.coefficients])
        success.append(info["gaze_success"] and info["calibration_success"] and np.all(predicted["converged"]))
        seconds.append(time.perf_counter()-start)
        if draw in (0, 4) or (draw+1) % 25 == 0:
            log(f"cal{size} hypothetical raw1um noise draw{draw+1}/{draws}; projected total{np.mean(seconds)*(draws):.1f}s")
    estimates = np.asarray(estimates)
    np.savez_compressed(out/f"shape_calibration_noise_{size}.npz", selection_indices=selection,
        truth_state=states[selection], estimated_A_d=estimates, parameters=np.asarray(parameters),
        fit_success=np.asarray(success), exact_calibration_A_d=exact["estimated_A_d"],
        raw_point_sigma_mm=np.array(.001), reference_noise_policy=np.array("one shared noisy A0zero reference per replicate"))
    return {"draws": draws, "evaluation_states": len(selection), "sigma_raw_coordinate_um": 1.,
        "fit_failures": int(np.sum(~np.asarray(success))), "wall_seconds": float(np.sum(seconds)),
        "mean_state_std_A_d": float(np.mean(np.std(estimates, axis=0, ddof=1))),
        "p95_state_std_A_d": float(np.percentile(np.std(estimates, axis=0, ddof=1), 95)),
        "total_A_errors_all_draws": scalar_metrics((estimates-states[selection, 0]).ravel()),
        "added_A_errors_vs_noise_free_calibration": scalar_metrics((estimates-exact["estimated_A_d"]).ravel()),
        "scope": "hypothetical shape-calibration noise sensitivity only; P4-P1 signal maps are not refitted here"}


def write_report(out, summary):
    lines = ["# Practical sparse P4 calibration", "",
        "The 27-state set uses three A levels (0,2,4 D) and 3×3 gaze positions (−20,0,20 degrees). The75-state set uses the same A levels and5×5 gaze positions (−20,−10,0,10,20). All other source states are disjoint evaluation. No dense fitted coefficients initialize these experiments.", "",
        "The real reference is used directly, with no barrel removal. s=1+s1*a and relative κ_A=k1*a+k2*a² are fitted on the three calibration on-axis states and fixed. Twelve identifiable lowest-order symmetric gaze terms are multiplied by{1,a}, giving27 shared values; the predeclared75-state quadratic-A candidate has39. Both H→Brown→radial and a same-count Brown→radial→H empirical order control are tested. These are empirical curved-mirror image maps, not a physical mirror law or Wang's exact sensor model.", "",
        "Selection uses calibration-only leave-one-nonzero-gaze-location CV, retaining all three zero-gaze anchors. The lower-count full-rank family within one standard error of best is selected. The3×3 folds mostly test reflected repetitions and cannot establish unsampled-gaze curvature accuracy.", "",
        "| Candidate | Shared values | CV coordinate MSE (µm²) | Full scaled rank | Heldout vertex RMS (µm) | Selected |",
        "|---|---:|---:|---:|---:|---|"]
    for name, row in summary["candidates"].items():
        lines.append(f"|{name}|{row['parameter_count']}|{row['cv']['mean_coordinate_mse_um2']:.6g}|{row['identifiability']['full_scaled_rank']}|{row['forward']['heldout']['vertex_rms_um']:.6g}|{'yes' if name in summary['selected'].values() else ''}|")
    lines += ["", "## Supplied-gaze accommodation estimation", "",
        "The estimator receives the measured nine-point shape and supplied gaze. It subtracts the observed and predicted means separately at every A candidate, so no stored true center is used. A fixed0…4 D scan at0.01 D is refined with bounded golden search. Estimates are continuous and unrounded; true A is used only for scoring.", "",
        "| Calibration states | Selected family | Heldout A RMSE (D) | P95 abs (D) | Maximum abs (D) | Within0.1 D |",
        "|---|---|---:|---:|---:|---:|"]
    for size, row in summary.get("known_gaze", {}).items():
        q = row["metrics"]["heldout"]
        lines.append(f"|{size}|{row['model']}|{q['rmse']:.6g}|{q['p95_abs']:.6g}|{q['max_abs']:.6g}|{100*q['within_0_1D_fraction_all_attempted']:.3f}%|")
    lines += ["", "Eight deterministic gaze-error vectors have total magnitude0.5 degrees, at45-degree intervals. Diagonal components are0.353553 degrees. They are not a Gaussian noise model. Exactly the same interior heldout observations are used for all directions and their exact-gaze comparison; no gaze clipping is performed. Saved results include total errors and added errors relative to exact supplied gaze. The maximum across those eight directions is not a certified maximum across every direction.", "",
        "The illustrative100-draw1µm trial adds noise to raw calibration point coordinates, then shares each noisy center and the one noisy reference consistently. It is shape-calibration sensitivity, not full-instrument noise performance. Source0.2D spacing does not establish0.1D resolving power.", "",
        "Full provenance, coefficient correlations, rank/condition, CV folds, statuses, per-direction metrics and noise results are in summary.json and the NPZ artifacts. P4−P1 runtime evaluation is reported separately after paired absolute-coordinate validation."]
    (out/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fit-only", action="store_true")
    parser.add_argument("--exact-only", action="store_true", help="Obtain primary exact-supplied-gaze results before sensitivity runs")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--scenario-per-A", type=int, default=1400, help="fixed common stratum sample;0 selects full common interior")
    parser.add_argument("--noise-draws", type=int, default=100)
    args = parser.parse_args()
    out = args.output_dir
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=6):
        axes, states, relative, centers, probe, sha = load_source()
        centered = relative-relative[:, 4:5, :]
        summary = {"provenance": {"grid_sha256": sha, "source_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "model_source_sha256": hashlib.sha256((ROOT/"Script/p4_sparse_model.py").read_bytes()).hexdigest(),
            "source_states": len(states), "probe_failed_states_retained": int(np.sum(probe)),
            "calibration_only_initialization": "identity and zero; no dense learned prior"}, "candidates": {}, "selected": {}}
        selected_models = {}
        for size in (27, 75):
            masks = calibration_masks(size)
            cal = masks["calibration"]
            definitions = [(1, "projective_first"), (1, "distortion_first")]+([(2, "projective_first")] if size == 75 else [])
            names = []
            for degree, composition in definitions:
                name = f"cal{size}_{composition}_Adegree{degree}"
                names.append(name)
                path = out/f"{name}.npz"
                started = time.perf_counter()
                if args.resume and path.exists() and (out/"summary.json").exists():
                    saved = json.loads((out/"summary.json").read_text(encoding="utf-8"))
                    model, row = SparseModel.load(path), saved["candidates"][name]
                else:
                    model, info = fit_model(states[cal], centered[cal], degree, composition)
                    cv = candidate_cv(states[cal], centered[cal], degree, composition)
                    identity = identifiability(model, states[cal])
                    prediction = np.empty_like(centered)
                    regularity = {key: np.inf for key in ("denominator", "radial_tangential", "radial_radial", "brown_determinant")}
                    for lower in range(0, len(states), 8192):
                        upper = min(lower+8192, len(states))
                        predicted = model.evaluate(*states[lower:upper].T, diagnostics=True)
                        prediction[lower:upper] = predicted["prediction"]
                        for key in regularity:
                            regularity[key] = min(regularity[key], float(np.min(predicted[key])))
                    error = prediction[:, OUTER]-centered[:, OUTER]
                    row = {"parameter_count": model.parameter_count, "fit": info, "cv": cv,
                        "identifiability": identity, "regularity": regularity,
                        "forward": {key: forward_metrics(error[mask]) for key, mask in masks.items()},
                        "wall_fit_CV_forward_seconds": time.perf_counter()-started}
                    model.save(path, grid_sha256=np.array(sha), calibration_state_indices=np.flatnonzero(cal),
                               calibration_states=states[cal], calibration_centered_mm=centered[cal])
                    np.savez_compressed(out/f"forward_{name}.npz", prediction_mm=prediction,
                                        calibration_mask=cal, truth_state=states)
                summary["candidates"][name] = row
                log(f"{name}:CVmse{row['cv']['mean_coordinate_mse_um2']:.6g}; rank{row['identifiability']['full_scaled_rank']}/{model.parameter_count}; heldoutRMS{row['forward']['heldout']['vertex_rms_um']:.5g}um")
                write_json(out/"summary.json", summary)
            eligible = [name for name in names if summary["candidates"][name]["identifiability"]["full_scaled_rank"] == summary["candidates"][name]["parameter_count"] and summary["candidates"][name]["cv"]["all_folds_full_rank_success"] and summary["candidates"][name]["fit"]["gaze_success"] and summary["candidates"][name]["fit"]["calibration_success"]]
            if not eligible:
                raise RuntimeError(f"No full-rank, successful calibration-only candidate for{size}states")
            best = min(eligible, key=lambda n: summary["candidates"][n]["cv"]["mean_coordinate_mse_um2"])
            threshold = summary["candidates"][best]["cv"]["mean_coordinate_mse_um2"]+summary["candidates"][best]["cv"]["se_coordinate_mse_um2"]
            within = [n for n in eligible if summary["candidates"][n]["cv"]["mean_coordinate_mse_um2"] <= threshold]
            selected = min(within, key=lambda n: (summary["candidates"][n]["parameter_count"], summary["candidates"][n]["cv"]["mean_coordinate_mse_um2"]))
            summary["selected"][str(size)] = selected
            selected_models[size] = SparseModel.load(out/f"{selected}.npz")
            write_json(out/"summary.json", summary)
        np.savez_compressed(out/"source_tensors.npz", A=axes[0], ADE=axes[1], BDE=axes[2],
            states=states, relative_mm=relative, center_absolute_mm=centers, probe_failed=probe, grid_sha256=np.array(sha))
        write_report(out, summary)
        if args.fit_only:
            return
        common = np.all(np.abs(states[:, 1:]) <= 19.2+1e-9, axis=1) & ~calibration_masks(75)["calibration"]
        assert np.sum(common) == 197562
        scenario_sample = (np.flatnonzero(common) if args.scenario_per_A == 0 else
                           fixed_subset(states, common, args.scenario_per_A, 20261107))
        summary["common_interior_heldout_states"] = int(np.sum(common))
        summary["controlled_gaze_scenario_states"] = int(len(scenario_sample))
        summary["known_gaze"], summary["shape_calibration_noise"] = {}, {}
        for size, model in selected_models.items():
            name = summary["selected"][str(size)]
            row, _ = evaluate_known(model, name, size, states, centered, calibration_masks(size), out,
                                    None if args.exact_only else scenario_sample, reuse=args.resume)
            summary["known_gaze"][str(size)] = row
            write_json(out/"summary.json", summary)
            write_report(out, summary)
            if args.noise_draws and not args.exact_only:
                summary["shape_calibration_noise"][str(size)] = noise_sensitivity(model, size, states, relative, centers,
                    calibration_masks(size)["calibration"], out, args.noise_draws)
            write_json(out/"summary.json", summary)
            write_report(out, summary)
        log(f"sparse calibration complete:{out}")


if __name__ == "__main__":
    main()
