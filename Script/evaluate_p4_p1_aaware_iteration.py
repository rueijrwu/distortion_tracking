"""Independent strict/fast pilots and scoped runtime for75-pose C12/C16."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from p4_aaware_center_cubic import AwareCenterCubic, FrozenC8Adapter, DEFAULT_ROOT, features
from p4_center_cubic_fixedpoint_aware import solve_fixedpoint
from p4_sparse_model import SparseModel
from evaluate_p4_p1_limited_gaze import load_shape
from evaluate_p4_p1_center_cubic_iteration import score
from fit_p4_p1_axis_cubic_gaze import AxisCubic, frozen_quality_flags
from fit_p4_limited_gaze import DEFAULT_OUT as SOURCE_ROOT
from evaluate_p4_p1_centroid_iteration import pilot_indices
from fit_p4_sparse_calibration import write_json, log

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def oracle(C, y4, m1, states):
    return C.predict(y4[:, 4]-m1, states[:, 0])  # reporting only, never an inference seed/update


def run(model, C, y4, m1, ids, global_ids, out, tag, solver, chunk=512):
    arrays = {}
    folder = out/f"traces_{tag}"
    folder.mkdir(exist_ok=True)
    begun = time.perf_counter()
    for lower in range(0, len(ids), chunk):
        upper = min(lower+chunk, len(ids))
        take = ids[lower:upper]
        result = solve_fixedpoint(model, C, y4[take], m1[take], profile_solver=solver, record_history=True)
        history = {key: result.pop(key) for key in list(result) if "history" in key}
        np.savez_compressed(folder/f"frames_{lower:06d}_{upper-1:06d}.npz", source_indices=global_ids[take], **history)
        for key, value in result.items():
            if key not in arrays:
                arrays[key] = np.empty((len(ids),)+value.shape[1:], dtype=value.dtype)
            arrays[key][lower:upper] = value
        elapsed = time.perf_counter()-begun
        log(f"{tag}:{upper}/{len(ids)},elapsed{elapsed:.2f}s,projected{elapsed*len(ids)/upper:.2f}s")
    return arrays, time.perf_counter()-begun


def compare(strict, fast):
    delta = np.abs(strict["estimated_state"]-fast["estimated_state"])
    center = np.max(np.abs(strict["corrected_center_absolute_mm"]-fast["corrected_center_absolute_mm"]), axis=1)
    sse = np.abs(strict["observed_shape_sse_mm2"]-fast["observed_shape_sse_mm2"])
    mismatch = ((delta[:, 0] > 1e-5) | np.any(delta[:, 1:] > 1e-4, axis=1) | (center > 1e-6) |
        (sse > 1e-12+1e-8*np.abs(strict["observed_shape_sse_mm2"])) |
        (strict["converged"] != fast["converged"]) |
        np.any(strict["all_start_converged"] != fast["all_start_converged"], axis=1))
    return {"all_pass": not bool(np.any(mismatch)), "material_mismatch_frames": int(np.sum(mismatch)),
        "max_delta_A_D": float(np.max(delta[:, 0])), "max_delta_ADE_BDE_deg": np.max(delta[:, 1:], axis=0).tolist(),
        "max_delta_center_mm": float(np.max(center)), "max_delta_shape_SSE_mm2": float(np.max(sse)),
        "convergence_status_mismatches": int(np.sum(strict["converged"] != fast["converged"]))}, mismatch


def synthetic_check():
    grid = np.array([(x, y) for x in (-1., 0., 1.) for y in (-1., 0., 1.)])
    model = SparseModel(grid*.8, np.array([-.08, .004, -.001]), np.linspace(-.015, .015, 24), 1, "distortion_first")
    C = AwareCenterCubic(np.zeros(2), np.ones(2), np.array([0., -7., .01, .005, .03, .07]),
                         np.array([0., 6.5, -.02, .004, -.04, .05]), 12)
    truth = np.array([[.73, -6.2, 8.1], [2.37, 9.7, -7.8], [3.43, -11.1, 5.2]])
    q = np.column_stack((truth[:, 2]/-7., truth[:, 1]/6.5))
    for _ in range(20):
        predicted = C.predict(q, truth[:, 0])
        a = truth[:, 0]/4.
        dx = C.x_to_BDE[1]+2*C.x_to_BDE[2]*q[:, 0]+3*C.x_to_BDE[3]*q[:, 0]**2+a*C.x_to_BDE[5]
        dy = C.y_to_ADE[1]+2*C.y_to_ADE[2]*q[:, 1]+3*C.y_to_ADE[3]*q[:, 1]**2+a*C.y_to_ADE[5]
        q[:, 0] -= (predicted[:, 1]-truth[:, 2])/dx
        q[:, 1] -= (predicted[:, 0]-truth[:, 1])/dy
    m1 = np.array([[.2, -.1], [.3, -.2], [.4, -.3]])
    y4 = (m1+q)[:, None]+model.evaluate(*truth.T)["prediction"]
    solved = {}
    for solver in ("reference401", "fast41"):
        result = solve_fixedpoint(model, C, y4, m1, profile_solver=solver)
        error = np.max(np.abs(result["estimated_state"]-truth), axis=0)
        assert np.all(result["converged"]) and np.all(error < [1e-3, 1e-3, 1e-3]), (solver, error)
        solved[solver] = result
    assert compare(solved["reference401"], solved["fast41"])[0]["all_pass"]
    print("A-aware coupled synthetic strict/fast blind recovery PASS", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_ROOT/"calibration")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT/"runtime_pilot")
    parser.add_argument("--phase", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--case", choices=("selected", "C8", "C12", "C16"), default="selected")
    parser.add_argument("--profile-solver", choices=("reference401", "fast41"), default="fast41")
    parser.add_argument("--pilot-dir", type=Path, default=DEFAULT_ROOT/"runtime_pilot")
    parser.add_argument("--profile-validation", type=Path, default=DEFAULT_ROOT/"profile_validation/summary.json")
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
            states, y4, m1, common, global_ids = (stored[key] for key in
                ("states", "absolute_p4_mm", "mean_p1_absolute_mm", "common_holdout", "source_indices"))
        calibrated = json.loads((args.calibration_dir/"summary.json").read_text(encoding="utf-8"))
        shape_path = args.source_dir/"models/new75_DF27.npz"
        model = load_shape(shape_path)
        flags = common & frozen_quality_flags(states)
        flagged = np.flatnonzero(flags)
        pilot = pilot_indices(states, common)
        rng = np.random.default_rng(20261107)
        random = np.sort(np.concatenate([rng.choice(np.flatnonzero(common & np.isclose(states[:, 0], a)), 1000, replace=False)
                                         for a in np.unique(states[:, 0])]))
        assert len(random) == 21000
        summary = {"shape_path": str(shape_path), "shape_sha256": digest(shape_path),
            "source_sha256": digest(source), "aware_core_sha256": digest(Path(__file__).with_name("p4_center_cubic_fixedpoint_aware.py")),
            "fast_profile_sha256": digest(Path(__file__).with_name("p4_fast_accommodation_profile.py")),
            "calibration_selection": calibrated["selection"], "phase": args.phase, "cases": {}}
        if args.phase == "pilot":
            data = {}
            for name in ("C12", "C16"):
                Cpath = Path(calibrated["models"][name]["path"])
                C = AwareCenterCubic.load(Cpath)
                for solver in ("reference401", "fast41"):
                    case = f"{name}_{solver}"
                    result, seconds = run(model, C, y4, m1, pilot, global_ids, out, f"pilot_{case}", solver, 512)
                    np.savez_compressed(out/f"pilot_{case}.npz", source_indices=global_ids[pilot], truth_state=states[pilot], **result)
                    oracle_values = oracle(C, y4[pilot], m1[pilot], states[pilot])
                    row = {"shape_path": str(shape_path), "shape_sha256": digest(shape_path), "center_cubic_path": str(Cpath),
                        "center_cubic_sha256": digest(Cpath), "profile_solver": solver, "total_values": 27+C.parameter_count,
                        "pilot": {"wall_seconds": seconds, "projected_full_seconds": seconds*83250/512.,
                            "metrics": score(result, states[pilot], y4[pilot, 4], oracle_values, np.ones(512, bool))}}
                    stress, flag_seconds = run(model, C, y4, m1, flagged, global_ids, out, f"flagged_pilot_{case}", solver, max(1, len(flagged)))
                    np.savez_compressed(out/f"flagged_pilot_{case}.npz", source_indices=global_ids[flagged], truth_state=states[flagged], **stress)
                    row["flagged_pilot"] = {"wall_seconds": flag_seconds,
                        "metrics": score(stress, states[flagged], y4[flagged, 4], oracle(C, y4[flagged], m1[flagged], states[flagged]), np.ones(len(flagged), bool))}
                    summary["cases"][case] = row
                    data[(name, solver)] = result, stress
                    write_json(out/"summary.json", summary)
                comparison, mismatch = compare(data[(name, "reference401")][0], data[(name, "fast41")][0])
                stress_comparison, stress_mismatch = compare(data[(name, "reference401")][1], data[(name, "fast41")][1])
                summary.setdefault("strict_fast_equivalence", {})[name] = {"pilot": comparison, "flagged": stress_comparison}
                np.savez_compressed(out/f"equivalence_{name}.npz", pilot_mismatch=mismatch, flagged_mismatch=stress_mismatch)
                write_json(out/"summary.json", summary)
            # Predeclared16 ordinary frames: latency includes all3starts/iterations,
            # excludes serialization and does not substitute batch throughput.
            ordinary = np.flatnonzero(common & ~flags)
            latency_ids = np.random.default_rng(20261107).choice(ordinary, 16, replace=False)
            chosen = calibrated["selection"]["selected"]
            C = AwareCenterCubic.load(calibrated["models"][chosen]["path"])
            summary["single_frame_latency"] = {"source_indices": global_ids[latency_ids].tolist(), "seed": 20261107,
                "model": chosen, "three_fixed_A_starts": True, "disk_excluded": True}
            for solver in ("reference401", "fast41"):
                solve_fixedpoint(model, C, y4[latency_ids[:1]], m1[latency_ids[:1]], profile_solver=solver, record_history=False)
                timings = []
                for i in latency_ids:
                    start = time.perf_counter()
                    solve_fixedpoint(model, C, y4[i:i+1], m1[i:i+1], profile_solver=solver, record_history=False)
                    timings.append(time.perf_counter()-start)
                summary["single_frame_latency"][solver] = {"seconds": timings, "median_seconds": float(np.median(timings)),
                    "P95_seconds": float(np.percentile(timings, 95)), "max_seconds": float(np.max(timings))}
            write_json(out/"summary.json", summary)
            if not all(row[part]["all_pass"] for row in summary["strict_fast_equivalence"].values() for part in ("pilot", "flagged")):
                raise RuntimeError("Material strict/fast controller disagreement; no fastfull evaluation")
            return
        chosen = calibrated["selection"]["selected"] if args.case == "selected" else args.case
        Cpath = Path(calibrated["models"][chosen]["path"])
        C = AwareCenterCubic.load(Cpath)
        if args.profile_solver == "fast41":
            profile_check = json.loads(args.profile_validation.read_text(encoding="utf-8"))
            pilot_check = json.loads((args.pilot_dir/"summary.json").read_text(encoding="utf-8"))
            if not profile_check["all_pass"] or profile_check["fast_source_sha256"] != summary["fast_profile_sha256"]:
                raise RuntimeError("Missing/mismatched fastprofile equivalence audit")
            if chosen in ("C12", "C16") and not all(pilot_check["strict_fast_equivalence"][chosen][part]["all_pass"] for part in ("pilot", "flagged")):
                raise RuntimeError("Missing controller strict/fast equivalence")
            if pilot_check["aware_core_sha256"] != summary["aware_core_sha256"]:
                raise RuntimeError("Awarecontroller changed afterpilot")
        full = chosen == calibrated["selection"]["selected"]
        selection = np.flatnonzero(common) if full else np.unique(np.r_[random, flagged])
        result, seconds = run(model, C, y4, m1, selection, global_ids, out, f"runtime_{chosen}", args.profile_solver, args.chunk)
        np.savez_compressed(out/f"runtime_{chosen}.npz", source_indices=global_ids[selection], truth_state=states[selection],
            true_central_ray_mm=y4[selection, 4], signed_state_error=result["estimated_state"]-states[selection], **result)
        oracle_values = oracle(C, y4[selection], m1[selection], states[selection])
        endpoint = np.isclose(states[selection, 0], 0.) | np.isclose(states[selection, 0], 4.)
        masks = {"full_common" if full else "evaluated_union_not_population": np.ones(len(selection), bool),
            "random21000": np.isin(selection, random), "flagged": flags[selection], "unflagged": ~flags[selection],
            "interior_A": ~endpoint, "endpoint_A": endpoint}
        summary.update(selected_case=chosen, full_common=full, profile_solver=args.profile_solver,
            total_values=27+C.parameter_count, center_cubic_path=str(Cpath), center_cubic_sha256=digest(Cpath),
            runtime={"wall_seconds": seconds, "metrics": {name: score(result, states[selection], y4[selection, 4], oracle_values, mask)
                for name, mask in masks.items() if np.any(mask)}})
        write_json(out/"summary.json", summary)


if __name__ == "__main__":
    main()
