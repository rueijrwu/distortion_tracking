"""Refit sparse shared P4/P1 calibration for requested gaze range±12.5deg.

Actual existing ray coverage is±12.4deg on63x63 nodes. No CODE V is used.
Every new fitted coefficient comes only from the declared27/75 observations.
"""
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

from p4_sparse_model import SparseModel, fit_model, OUTER
from p4_sparse_model_69 import fit_model as fit69
from fit_p4_sparse_calibration_69 import cv69
from fit_p4_sparse_calibration import candidate_cv, identifiability, forward_metrics, write_json, log
from p4_centroid_iteration import select_signal_map
from p4_cubic_center_signal import compare_center_maps
from evaluate_p4_p1_centroid_iteration import pair_inputs, SOURCE_FIT, P1_PATH
from p4_known_gaze_inverse import estimate_accommodation

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT/"data/p4_p1_limited_gaze_12p5"
GAZE_NODES = {27: np.array([-12.4, 0., 12.4]), 75: np.array([-12.4, -6.4, 0., 6.4, 12.4])}


def member(values, nodes):
    return np.any(np.isclose(np.asarray(values)[:, None], nodes[None, :], atol=1e-9, rtol=0), axis=1)


def masks_for(states):
    allowed = np.all(np.abs(states[:, 1:]) <= 12.5+1e-10, axis=1)
    seen_a = member(states[:, 0], np.array([0., 2., 4.]))
    cal = {}
    for size, nodes in GAZE_NODES.items():
        cal[size] = seen_a & member(states[:, 1], nodes) & member(states[:, 2], nodes)
        assert np.sum(cal[size]) == size
    old = seen_a & member(states[:, 1], np.array([-20., -10., 0., 10., 20.])) & member(states[:, 2], np.array([-20., -10., 0., 10., 20.]))
    old_inside = old & allowed
    union = cal[75] | old_inside
    common = allowed & ~union
    assert np.sum(allowed) == 83349 and np.sum(old_inside) == 27
    assert np.sum(cal[75] & old_inside) == 3 and np.sum(union) == 99
    assert np.sum(common) == 83250 and np.all(~cal[27] | cal[75])
    return allowed, cal, old_inside, common


def evaluation_masks(states, cal, common):
    seen_a = member(states[:, 0], np.array([0., 2., 4.]))
    seen_gaze = np.any(np.all(np.isclose(states[:, None, 1:], np.unique(states[cal, 1:], axis=0)[None, :, :], atol=1e-9, rtol=0), axis=2), axis=1)
    endpoints = member(states[:, 0], np.array([0., 4.]))
    edge = np.any(np.isclose(np.abs(states[:, 1:]), 12.4, atol=1e-9, rtol=0), axis=1)
    return {"calibration": cal, "own_holdout": ~cal, "common_holdout": common,
        "seen_A_new_gaze": seen_a & ~seen_gaze, "new_A_seen_gaze": ~seen_a & seen_gaze,
        "both_new": ~seen_a & ~seen_gaze, "common_interior_A": common & ~endpoints,
        "common_endpoint_A": common & endpoints, "common_gaze_edge": common & edge,
        "common_gaze_interior": common & ~edge}


def selected_name(candidates, names):
    eligible = [name for name in names if candidates[name]["fit"]["gaze_success"] and candidates[name]["fit"]["calibration_success"]
        and candidates[name]["identifiability"]["full_scaled_rank"] == candidates[name]["parameter_count"]
        and candidates[name]["cv"]["all_folds_full_rank_success"]]
    if not eligible:
        raise RuntimeError("No successful full-rank calibration-only candidate")
    best = min(eligible, key=lambda name: candidates[name]["cv"]["mean_coordinate_mse_um2"])
    threshold = candidates[best]["cv"]["mean_coordinate_mse_um2"]+candidates[best]["cv"]["se_coordinate_mse_um2"]
    within = [name for name in eligible if candidates[name]["cv"]["mean_coordinate_mse_um2"] <= threshold]
    selected = min(within, key=lambda name: (candidates[name]["parameter_count"], candidates[name]["cv"]["mean_coordinate_mse_um2"]))
    simplest = min(eligible, key=lambda name: (candidates[name]["parameter_count"], candidates[name]["cv"]["mean_coordinate_mse_um2"]))
    return selected, {"eligible": eligible, "best_CV": best, "one_SE_threshold_coordinate_mse_um2": threshold,
                      "selected": selected, "simplest_full_rank": simplest}


def parity_calibration(states, targets):
    values = {(round(float(A), 9), round(float(ade), 9), round(float(bde), 9)): target
              for (A, ade, bde), target in zip(states, targets)}
    odd, even, zero_x = [], [], []
    for (A, ade, bde), target in values.items():
        opposite, center = values[(A, ade, -bde)], values[(A, ade, 0.)]
        odd.append(target[0]+opposite[0]-2*center[0])
        even.append(target[1]-opposite[1])
        zero_x.append(center[0]-values[(A, 0., 0.)][0])
    return {"calibration_states": len(states), "equality_tolerance_mm": 1e-12,
            "Dx_odd_about_BDE0_max_abs_mm": float(np.max(np.abs(odd))),
            "Dy_BDE_even_max_abs_mm": float(np.max(np.abs(even))),
            "BDE0_Dx_dependence_on_ADE_max_abs_mm": float(np.max(np.abs(zero_x)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-fit", type=Path, default=SOURCE_FIT)
    parser.add_argument("--P1", type=Path, default=P1_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--known-gaze", action="store_true")
    args = parser.parse_args()
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    (out/"models").mkdir(exist_ok=True)
    with threadpool_limits(limits=6):
        full_states, full_y4, full_m1, full_centers, full_valid, provenance = pair_inputs(args.source_fit, args.P1)
        allowed, full_cal, old_inside, full_common = masks_for(full_states)
        if np.any(allowed & ~full_valid):
            raise RuntimeError("Invalid real rays inside requested box; report rather than silently filter")
        source_indices = np.flatnonzero(allowed)
        states, y4, m1, centers = full_states[allowed], full_y4[allowed], full_m1[allowed], full_centers[allowed]
        observed = y4-y4[:, 4:5]
        cal = {size: full_cal[size][allowed] for size in (27, 75)}
        common = full_common[allowed]
        assert np.sum(~cal[27]) == 83322 and np.sum(~cal[75]) == 83274
        np.savez_compressed(out/"source_subset.npz", source_indices=source_indices, states=states,
            absolute_p4_mm=y4, mean_p1_absolute_mm=m1, center_absolute_mm=centers,
            cal27=cal[27], cal75=cal[75], old_calibration_inside=old_inside[allowed], common_holdout=common)
        summary = {"provenance": provenance, "requested_gaze_bound_deg": 12.5,
            "actual_ray_support_bound_deg": 12.4, "gaze_polynomial_normalization_deg": 20.,
            "source_subset_states": len(states), "common_holdout_states": int(np.sum(common)),
            "old_new_calibration_union": {"new75": 75, "old_inside": 27, "overlap": 3, "union": 99},
            "calibration_nodes": {str(k): v.tolist() for k, v in GAZE_NODES.items()},
            "source_hashes": {name: hashlib.sha256((ROOT/"Script"/name).read_bytes()).hexdigest() for name in
                ("fit_p4_limited_gaze.py", "p4_sparse_model.py", "p4_sparse_model_69.py", "p4_cubic_center_signal.py")},
            "new_initialization": "zero/identity starts and raw calibration anchor only; no old learned coefficients",
            "shape_candidates": {}, "shape_selection": {}, "map_selection": {}}
        families = {27: [("PF27", 1, "projective_first", False), ("DF27", 1, "distortion_first", False)],
                    75: [("PF27", 1, "projective_first", False), ("DF27", 1, "distortion_first", False),
                         ("PF39", 2, "projective_first", False), ("PF69", 1, "projective_first", True)]}
        for size in (27, 75):
            names = []
            for label, degree, composition, extended in families[size]:
                name = f"new{size}_{label}"
                names.append(name)
                begun = time.perf_counter()
                model, fit = (fit69(states[cal[size]], observed[cal[size]]) if extended else
                    fit_model(states[cal[size]], observed[cal[size]], degree, composition))
                cv = cv69(states[cal[size]], observed[cal[size]]) if extended else candidate_cv(states[cal[size]], observed[cal[size]], degree, composition)
                identity = identifiability(model, states[cal[size]])
                prediction = np.empty_like(observed)
                margins = {key: np.inf for key in ("denominator", "radial_tangential", "radial_radial", "brown_determinant")}
                for lower in range(0, len(states), 8192):
                    upper = min(lower+8192, len(states))
                    result = model.evaluate(*states[lower:upper].T, diagnostics=True)
                    prediction[lower:upper] = result["prediction"]
                    for key in margins:
                        margins[key] = min(margins[key], float(np.min(result[key])))
                errors = prediction[:, OUTER]-observed[:, OUTER]
                masks = evaluation_masks(states, cal[size], common)
                model_path = out/"models"/f"{name}.npz"
                model.save(model_path, calibration_states=states[cal[size]],
                    calibration_source_indices=source_indices[cal[size]], grid_sha256=np.array(provenance["P4_sha256"]))
                np.savez_compressed(out/f"forward_{name}.npz", source_indices=source_indices,
                                    prediction_mm=prediction, signed_error_mm=prediction-observed)
                row = {"parameter_count": model.parameter_count, "model_path": str(model_path),
                    "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(), "fit": fit, "cv": cv,
                    "identifiability": identity, "sampled_map_margins": margins,
                    "forward": {key: forward_metrics(errors[take]) for key, take in masks.items() if np.any(take)},
                    "forward_by_A_common": {f"{a:.1f}": forward_metrics(errors[common & np.isclose(states[:, 0], a)]) for a in np.unique(states[:, 0])},
                    "wall_seconds": time.perf_counter()-begun}
                if args.known_gaze:
                    inverse, timing = estimate_accommodation(model, y4, states[:, 1:], mean_centered=True)
                    from fit_p4_sparse_calibration import score_inverse
                    row["exact_known_gaze"], error = score_inverse(inverse, states[:, 0], masks)
                    row["exact_known_gaze_timing"] = timing
                    np.savez_compressed(out/f"known_gaze_{name}.npz", source_indices=source_indices,
                        truth_state=states, signed_A_error_d=error, **inverse)
                summary["shape_candidates"][name] = row
                write_json(out/"summary.json", summary)
                log(f"{name}: CV{cv['mean_coordinate_mse_um2']:.6g}; rank{identity['full_scaled_rank']}/{model.parameter_count}; common RMS{row['forward']['common_holdout']['vertex_rms_um']:.5g}um")
            chosen, choice = selected_name(summary["shape_candidates"], names)
            summary["shape_selection"][str(size)] = choice
            if size == 75:
                _, low = selected_name(summary["shape_candidates"], names[:3])
                summary["shape_selection"]["75_low"] = low
            targets = centers[cal[size]]-m1[cal[size]]
            if size == 27:
                signal, selection_info = select_signal_map(states[cal[size]], targets)
                signal_name = f"D{signal.parameter_count}"
                maps = {signal_name: signal}
                selection_info["selected"] = signal_name
            else:
                maps, selection_info = compare_center_maps(states[cal[size]], targets)
                signal_name = selection_info["selected"]
                summary["new75_calibration_BDE_parity"] = parity_calibration(states[cal[size]], targets)
            paths = {}
            for name, signal in maps.items():
                path = out/"models"/f"new{size}_{name}.npz"
                signal.save(path, calibration_states=states[cal[size]], calibration_targets_mm=targets)
                paths[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                               "parameter_count": signal.parameter_count}
            summary["map_selection"][str(size)] = {"selected": signal_name, "comparison": selection_info, "artifacts": paths}
            write_json(out/"summary.json", summary)
        log(f"limited-range calibration complete: {out}")


if __name__ == "__main__":
    main()
