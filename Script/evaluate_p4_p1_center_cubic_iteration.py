"""Compact35/35/77-value corrected-center cubic/shape fixed-point experiment."""
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

from fit_p4_limited_gaze import DEFAULT_OUT as SOURCE_ROOT
from fit_p4_p1_corrected_axis_cubic import DEFAULT_ROOT
from fit_p4_p1_axis_cubic_gaze import AxisCubic, metrics as gaze_metrics, frozen_quality_flags, design
from p4_sparse_model import SparseModel
from evaluate_p4_p1_limited_gaze import load_shape
from p4_center_cubic_fixedpoint import solve_fixedpoint, START_A, boundary_flags
from evaluate_p4_p1_centroid_iteration import pilot_indices
from fit_p4_sparse_calibration import scalar_metrics, write_json, log


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(result, truth, central_ray, oracle_gaze, mask):
    error = result["estimated_state"][mask]-truth[mask]
    center_error = result["corrected_center_absolute_mm"][mask]-central_ray[mask]
    return {"states": int(np.sum(mask)), "A_d": scalar_metrics(error[:, 0]),
        "raw_initial_gaze": gaze_metrics(result["raw_initial_gaze"][mask], truth[mask, 1:]),
        "bounded_initial_gaze": gaze_metrics(result["bounded_initial_gaze"][mask], truth[mask, 1:]),
        "first_corrected_gaze_A2": gaze_metrics(result["first_bounded_corrected_gaze"][mask], truth[mask, 1:]),
        "final_gaze": gaze_metrics(result["estimated_state"][mask, 1:], truth[mask, 1:]),
        "scoring_only_oracle_central_ray_gaze": gaze_metrics(oracle_gaze[mask], truth[mask, 1:]),
        "primary_A2_A_d": scalar_metrics(result["primary_A2_state"][mask, 0]-truth[mask, 0]),
        "primary_A2_gaze": gaze_metrics(result["primary_A2_state"][mask, 1:], truth[mask, 1:]),
        "converged": int(np.sum(result["converged"][mask])), "nonconverged": int(np.sum(~result["converged"][mask])),
        "primary_A2_converged": int(np.sum(result["primary_A2_converged"][mask])),
        "no_valid_start": int(np.sum(result["no_valid_start"][mask])),
        "bounded_fixedpoint_raw_pressure": int(np.sum(result["bounded_fixedpoint_with_raw_pressure"][mask])),
        "interior_fixedpoint": int(np.sum(result["interior_fixedpoint"][mask])),
        "cycle_event_frames": int(np.sum(result["cycle_events"][mask] > 0)),
        "stagnation": int(np.sum(result["stagnation"][mask])), "iteration_cap": int(np.sum(result["iteration_cap"][mask])),
        "invalid_update": int(np.sum(result["invalid_update"][mask])),
        "distinct_converged_roots": int(np.sum(result["distinct_converged_roots"][mask])),
        "multiple_A_profile_minima": int(np.sum(result["profile_local_minima"][mask] > 1)),
        "median_T_evaluations": float(np.median(result["iterations"][mask])),
        "p95_T_evaluations": float(np.percentile(result["iterations"][mask], 95)),
        "center_RMS_um": float(np.sqrt(np.mean(np.sum(center_error**2, axis=1)))*1000.),
        "observed_shape_vertex_RMS_um": float(np.sqrt(np.mean(result["observed_shape_sse_mm2"][mask]/9.))*1000.),
        "bounds_A_ADE_BDE": np.sum(result["boundary_A_ADE_BDE"][mask], axis=0).tolist(),
        "unsampled_12p4_to_12p5_strip_estimates": int(np.sum(np.any(np.abs(result["estimated_state"][mask, 1:]) > 12.4+1e-9, axis=1)))}


def run(model, cubic, y4, m1, selection, indices, out, tag, chunk=512):
    arrays = {}
    traces = out/f"traces_{tag}"
    traces.mkdir(exist_ok=True)
    begun = time.perf_counter()
    for lower in range(0, len(selection), chunk):
        upper = min(lower+chunk, len(selection))
        ids = selection[lower:upper]
        result = solve_fixedpoint(model, cubic, y4[ids], m1[ids], record_history=True)
        histories = {key: result.pop(key) for key in list(result) if "history" in key}
        np.savez_compressed(traces/f"frames_{lower:06d}_{upper-1:06d}.npz", source_indices=indices[ids], **histories)
        for key, value in result.items():
            if key not in arrays:
                arrays[key] = np.empty((len(selection),)+value.shape[1:], dtype=value.dtype)
            arrays[key][lower:upper] = value
        elapsed = time.perf_counter()-begun
        log(f"{tag}: {upper:,}/{len(selection):,}; elapsed{elapsed:.1f}s; projected{elapsed*len(selection)/upper:.1f}s")
    return arrays, time.perf_counter()-begun


def synthetic_check():
    edge_states = np.array([[0., 1., -2.], [2., -12.5, 0.], [4., 0., 12.5], [2., -12.4, 12.4]])
    assert np.array_equal(boundary_flags(edge_states), [[True, False, False], [False, True, False],
                                                       [True, False, True], [False, False, False]])
    fields = np.array([(x, y) for x in (-1., 0., 1.) for y in (-1., 0., 1.)])
    model = SparseModel(fields*.8, np.array([-.08, .004, -.001]), np.linspace(-.015, .015, 24), 1, "distortion_first")
    C = AxisCubic(np.zeros(2), np.ones(2), np.array([0., -7., .01, .005]), np.array([0., 6.5, -.02, .004]))
    truth = np.array([[.73, -6.2, 8.1], [2.37, 9.7, -7.8], [3.43, -11.1, 5.2]])
    q = np.column_stack((truth[:, 2]/-7., truth[:, 1]/6.5))
    for _ in range(15):
        for axis, beta, target in ((0, C.x_to_BDE, truth[:, 2]), (1, C.y_to_ADE, truth[:, 1])):
            derivative = beta[1]+2*beta[2]*q[:, axis]+3*beta[3]*q[:, axis]**2
            q[:, axis] -= (design(q[:, axis])@beta-target)/derivative
    m1 = np.array([[.2, -.1], [.3, -.2], [.4, -.3]])
    true_center = m1+q
    F = model.evaluate(*truth.T)["prediction"]
    y4 = true_center[:, None]+F
    assert np.max(np.abs(np.mean(y4, axis=1)-np.mean(F, axis=1)-true_center)) < 1e-14
    solved = solve_fixedpoint(model, C, y4, m1)
    error = np.max(np.abs(solved["estimated_state"]-truth), axis=0)
    if not np.all(solved["converged"]) or not np.all(error < [1e-3, 1e-3, 1e-3]):
        raise AssertionError(f"Synthetic fixed-point recovery failed:{error},status{solved['converged']}")
    assert np.all(solved["estimated_state"][:, 0] >= 0.) and np.all(solved["estimated_state"][:, 0] <= 4.)
    assert np.all(np.abs(solved["estimated_state"][:, 1:]) <= 12.5)
    print(f"coupled synthetic recovery/center identity PASS,maxA-ADE-BDE errors{error}", flush=True)


def write_report(out, summary):
    lines = ["# Corrected-center cubic plus forward-shape fixed point", "",
        "A new8-coefficient C map is trained on labeled central P4 ray−mean(P1), directly against known calibration gaze. It is not the earlier raw-mean cubic. The SAME C is deliberately applied to rawmean difference for initialization, so there are no extra initializer coefficients. Frozen shape plus C gives35,35 or77 fitted values.", "",
        "Runtime sees only nine measured absolute P4 points and measured P1 mean. Current model centroid bias estimates the central-ray center; C updates gaze; the mean-centered nine-point shape profiles accommodation. Damped center-first Picard uses the undamped fixed-point residual for safeguards/convergence, never the old D-based SSE gate. A2 is primary with fixed0/4 controls; valid converged roots are chosen by observed shape SSE. True state/center are scoring-only.", "",
        "| Case | Total values | Common N | A RMSE(D) | A P95 abs(D) | Final gaze RMSE(deg) | Rawinitial gaze RMSE | Oraclecenter gaze RMSE | Converged |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in summary["cases"].items():
        if "runtime" not in row:
            continue
        q = row["runtime"]["metrics"]["all_common"]
        lines.append(f"|{name}|{row['total_values']}|{q['states']:,}|{q['A_d']['rmse']:.7g}|{q['A_d']['p95_abs']:.7g}|{q['final_gaze']['radial_RMSE_deg']:.7g}|{q['raw_initial_gaze']['radial_RMSE_deg']:.7g}|{q['scoring_only_oracle_central_ray_gaze']['radial_RMSE_deg']:.7g}|{q['converged']:,}|")
    lines += ["", "All83,250 common heldout rows, including flagged P1 observations, remain primary. Separate flag/interior/endpoint diagnostics, pressure atbounds, the unsampled12.4–12.5° strip, A-profile minima, allstartroots/statuses and every raw C/T/state/alpha/center residual history are saved. The oracle central-ray calculation is performed only by reporting code. Nominal shared SI-frame remains an assumption; no runtime-noise accuracy or0.1D resolution is claimed."]
    (out/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--cubic-dir", type=Path, default=DEFAULT_ROOT/"calibration")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT/"runtime")
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--chunk", type=int, default=512)
    args = parser.parse_args()
    with threadpool_limits(limits=6):
        synthetic_check()
        if args.check_only:
            return
        out = args.output_dir
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(out)
        out.mkdir(parents=True, exist_ok=True)
        source = args.source_dir/"source_subset.npz"
        with np.load(source) as stored:
            states, y4, m1, common, indices = (stored[key] for key in
                ("states", "absolute_p4_mm", "mean_p1_absolute_mm", "common_holdout", "source_indices"))
        assert np.sum(common) == 83250
        centers = y4[:, 4]
        flags = frozen_quality_flags(states)
        selection, pilot = np.flatnonzero(common), pilot_indices(states, common)
        flagged = np.flatnonzero(common & flags)
        definitions = [("cal27_F27_C8", "new27_DF27.npz", 27),
                       ("cal75_F27_C8", "new75_DF27.npz", 75), ("cal75_F69_C8", "new75_PF69.npz", 75)]
        summary = {"source_subset_sha256": digest(source), "runtime_core_sha256": digest(Path(__file__).with_name("p4_center_cubic_fixedpoint.py")),
            "runner_sha256": digest(__file__), "protocol": "center-first C8 then mean-centered A profile; damped Picard; noD-SSE gate",
            "fixed_A_starts": START_A.tolist(), "common_holdout": 83250, "cases": {}}
        prepared = {}
        for name, filename, budget in definitions:
            shape_path = args.source_dir/"models"/filename
            cubic_path = args.cubic_dir/f"corrected_axis_cubic_{budget}.npz"
            model, C = load_shape(shape_path), AxisCubic.load(cubic_path)
            prepared[name] = model, C
            # Oracle is outside the inference API and used ONLY below for scores.
            oracle = C.predict(centers-m1)
            result, seconds = run(model, C, y4, m1, pilot, indices, out, f"pilot_{name}", 512)
            np.savez_compressed(out/f"pilot_{name}.npz", source_indices=indices[pilot], truth_state=states[pilot], **result)
            row = {"shape_path": str(shape_path), "shape_sha256": digest(shape_path), "center_cubic_path": str(cubic_path),
                "center_cubic_sha256": digest(cubic_path), "shape_values": model.parameter_count, "center_values": 8,
                "total_values": model.parameter_count+8,
                "pilot": {"wall_seconds": seconds, "projected_full_seconds": seconds*83250/512.,
                    "metrics": score(result, states[pilot], centers[pilot], oracle[pilot], np.ones(len(pilot), bool))}}
            if len(flagged):
                diagnostic, flag_seconds = run(model, C, y4, m1, flagged, indices, out, f"flagged_pilot_{name}", len(flagged))
                np.savez_compressed(out/f"flagged_pilot_{name}.npz", source_indices=indices[flagged], truth_state=states[flagged], **diagnostic)
                row["flagged_pilot"] = {"wall_seconds": flag_seconds,
                    "metrics": score(diagnostic, states[flagged], centers[flagged], oracle[flagged], np.ones(len(flagged), bool))}
            summary["cases"][name] = row
            write_json(out/"summary.json", summary)
            log(f"{name}512pilot{seconds:.2f}s; fullprojection{seconds*83250/512.:.1f}s")
        write_report(out, summary)
        if args.pilot_only:
            return
        failures = [name for name, row in summary["cases"].items() if
            row["pilot"]["metrics"]["nonconverged"] or row["pilot"]["metrics"]["distinct_converged_roots"] or
            row.get("flagged_pilot", {}).get("metrics", {}).get("nonconverged", 0)]
        if failures:
            raise RuntimeError(f"Pilot requires convergence audit before full run; preserved cases:{failures}")
        endpoint = np.isclose(states[selection, 0], 0.) | np.isclose(states[selection, 0], 4.)
        for name, _, _ in definitions:
            model, C = prepared[name]
            result, seconds = run(model, C, y4, m1, selection, indices, out, name, args.chunk)
            oracle = C.predict(centers[selection]-m1[selection])
            np.savez_compressed(out/f"runtime_{name}.npz", source_indices=indices[selection], truth_state=states[selection],
                true_central_ray_mm=centers[selection], signed_state_error=result["estimated_state"]-states[selection], **result)
            masks = {"all_common": np.ones(len(selection), bool), "unflagged": ~flags[selection], "flagged": flags[selection],
                     "interior_A": ~endpoint, "endpoint_A": endpoint}
            summary["cases"][name]["runtime"] = {"wall_seconds": seconds,
                "metrics": {key: score(result, states[selection], centers[selection], oracle, mask) for key, mask in masks.items() if np.any(mask)},
                "per_A": {f"{av:.1f}": score(result, states[selection], centers[selection], oracle, np.isclose(states[selection, 0], av)) for av in np.unique(states[:, 0])}}
            write_json(out/"summary.json", summary)
            write_report(out, summary)


if __name__ == "__main__":
    main()
