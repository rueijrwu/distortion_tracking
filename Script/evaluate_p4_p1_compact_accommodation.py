"""Compact full-accommodation feedback with an unchanged q-only vector cubic."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "4")

import numpy as np

from p4_a0_bivariate_polynomial import BivariatePolynomial
from evaluate_p4_p1_limited_gaze import load_shape
import p4_shared_gaze_accommodation_fixedpoint as engine

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"data/p4_p1_compact_full_accommodation"
VARIANTS = ("A0_frozen", "shared75")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_metrics(estimated, truth):
    error = np.asarray(estimated)-np.asarray(truth)
    finite = np.isfinite(error).all(axis=1)
    e = error[finite]
    answer = {"attempted": len(error), "finite": int(finite.sum()), "nonfinite": int((~finite).sum())}
    if not len(e):
        return answer
    for col, name in enumerate(("A_D", "ADE_deg", "BDE_deg")):
        x = e[:, col]
        answer[name] = {"bias": float(x.mean()), "RMS": float(np.sqrt(np.mean(x*x))),
            "MAE": float(np.abs(x).mean()), "median_abs": float(np.median(np.abs(x))),
            "P95_abs": float(np.percentile(np.abs(x), 95)), "max_abs": float(np.abs(x).max())}
    r = np.linalg.norm(e[:, 1:], axis=1)
    answer["gaze_radial_deg"] = {"RMS": float(np.sqrt(np.mean(r*r))), "median": float(np.median(r)),
        "P95": float(np.percentile(r, 95)), "max": float(r.max()),
        "fraction_le_0p5_all_attempted": float(np.sum(r <= .5)/len(error))}
    answer["fraction_A_abs_le_0p1D_all_attempted"] = float(np.sum(np.abs(e[:, 0]) <= .1)/len(error))
    return answer


def center_metrics(estimated, labeled):
    e = (np.asarray(estimated)-np.asarray(labeled))*1000.0
    good = e[np.isfinite(e).all(axis=1)]
    if not len(good):
        return {"attempted": len(e), "finite": 0}
    r = np.linalg.norm(good, axis=1)
    return {"attempted": len(e), "finite": len(good), "bias_xy_um": good.mean(axis=0).tolist(),
        "vector_RMS_um": float(np.sqrt(np.mean(r*r))), "median_um": float(np.median(r)),
        "P95_um": float(np.percentile(r, 95)), "max_um": float(r.max())}


def histories_and_last_two(result):
    history = {key: value for key, value in result.items() if "history" in key}
    compact = {key: value for key, value in result.items() if "history" not in key}
    iterations = result["all_start_iterations"]
    n = len(iterations)
    indices = np.stack((np.maximum(iterations-2, 0), np.maximum(iterations-1, 0)), axis=2)
    for key, value in history.items():
        if key.startswith("all_start_"):
            compact["last_two_"+key] = value[np.arange(n)[:, None, None], np.arange(3)[None, :, None], indices]
    compact["last_two_recorded_evaluations"] = np.minimum(iterations, 2)
    compact["primary_first_state"] = result["all_start_T_history"][:, 1, 0]
    delta = compact["last_two_all_start_fixedpoint_delta_history"]
    compact["last_two_A_tolerance_pass"] = np.abs(delta[..., 0]) <= 1e-5
    compact["last_two_gaze_tolerance_pass"] = np.linalg.norm(delta[..., 1:], axis=-1) <= 1e-4
    tested = compact["last_two_A_tolerance_pass"] & compact["last_two_gaze_tolerance_pass"]
    compact["convergence_history_consistent"] = (~result["all_start_converged"]
        | ((iterations >= 2) & np.all(tested, axis=2)))
    return compact, history


def score(result, observed, truth, primary, model):
    initial = np.column_stack((np.full(len(truth), 2.), result["bounded_initial_gaze"]))
    raw = np.column_stack((np.full(len(truth), 2.), result["raw_initial_gaze"]))
    selected = result["estimated_state"]
    primary_state = result["primary_A2_state"]
    primary_center = engine.shape_state(model, observed, primary_state)["center"]
    stages = {"raw_initial_unbounded": raw, "raw_initial_bounded": initial,
              "first_update_primary": result["primary_first_state"],
              "final_primary_A2": primary_state, "final_selected": selected}
    groups = {}
    for name, mask in (("primary", primary), ("all_diagnostic", np.ones(len(truth), bool)), ("excluded_diagnostic", ~primary)):
        groups[name] = {stage: state_metrics(value[mask], truth[mask]) for stage, value in stages.items()}
        groups[name]["center_selected"] = center_metrics(result["corrected_center_absolute_mm"][mask], observed[mask, 4])
        groups[name]["center_primary"] = center_metrics(primary_center[mask], observed[mask, 4])
        groups[name]["telemetry"] = {
            "attempted": int(mask.sum()), "converged": int(result["converged"][mask].sum()),
            "primary_converged": int(result["primary_A2_converged"][mask].sum()),
            "no_valid_start": int(result["no_valid_start"][mask].sum()),
            "alternate_rescue": int(np.sum(result["converged"][mask] & ~result["primary_A2_converged"][mask])),
            "converged_primary_replaced": int(np.sum(result["primary_A2_converged"][mask] & (result["selected_start_index"][mask] != 1))),
            "distinct_roots": int(result["distinct_converged_roots"][mask].sum()),
            "bounds_A_ADE_BDE": result["boundary_A_ADE_BDE"][mask].sum(axis=0).tolist(),
            "unsampled_gaze_edge_strip": int(np.any(np.abs(selected[mask, 1:]) > 12.4+1e-10, axis=1).sum()),
            "iteration_cap": int(result["iteration_cap"][mask].sum()),
            "profile_calls_all_starts": int(result["all_start_profile_calls"][mask].sum()),
            "reference401_fallbacks_all_starts": int(result["all_start_profile401_fallback_calls"][mask].sum()),
            "cycles_all_starts": int(result["all_start_cycle_events"][mask].sum()),
            "convergence_history_inconsistent": int((~result["convergence_history_consistent"][mask]).sum()),
        }
    groups["per_true_A"] = {f"{A:.1f}": state_metrics(selected[primary & (truth[:, 0] == A)], truth[primary & (truth[:, 0] == A)])
                            for A in np.unique(truth[primary, 0])}
    return groups


def physical_equivalence(reference, fast, y4, primary, model):
    a, b = reference["estimated_state"], fast["estimated_state"]
    Fa = model.evaluate(*a.T, diagnostics=True)["prediction"]
    Fb = model.evaluate(*b.T, diagnostics=True)["prediction"]
    observed = y4-y4.mean(axis=1, keepdims=True)
    ra, rb = observed-(Fa-Fa.mean(axis=1, keepdims=True)), observed-(Fb-Fb.mean(axis=1, keepdims=True))
    dr = rb-ra
    na, nd = np.linalg.norm(ra, axis=(1, 2)), np.linalg.norm(dr, axis=(1, 2))
    dsse = np.abs(np.sum(rb*rb, axis=(1, 2))-np.sum(ra*ra, axis=(1, 2)))
    bound = 2*na*nd+nd*nd
    point_shift_um = np.max(np.abs(Fa-Fb), axis=(1, 2))*1000.0
    delta = np.abs(a-b)
    center_delta = np.max(np.abs(reference["corrected_center_absolute_mm"]-fast["corrected_center_absolute_mm"]), axis=1)
    same_status = (reference["converged"] == fast["converged"]) & np.all(reference["all_start_converged"] == fast["all_start_converged"], axis=1)
    good = ((delta[:, 0] <= 1e-5) & np.all(delta[:, 1:] <= 1e-4, axis=1)
            & (center_delta <= 1e-6) & (point_shift_um <= 1e-4) & same_status & (dsse <= bound+1e-18))
    answer = {}
    for name, mask in (("primary", primary), ("excluded_diagnostic", ~primary)):
        answer[name] = {"attempted": int(mask.sum()), "passed": bool(good[mask].all()),
            "failure_count": int((~good[mask]).sum()), "max_state_component_delta": delta[mask].max(axis=0).tolist(),
            "max_center_component_delta_mm": float(center_delta[mask].max()),
            "max_prediction_coordinate_shift_um": float(point_shift_um[mask].max()),
            "status_difference_count": int((~same_status[mask]).sum()),
            "max_SSE_Cauchy_bound_violation_mm2": float(np.maximum(dsse-bound, 0)[mask].max())}
    return answer


def identical_query_audit(reference, y4, primary, model, batch_size):
    T = reference["all_start_T_history"]
    states = reference["all_start_state_history"]
    successful = reference["all_start_A_profile_success_history"]
    valid = np.isfinite(T).all(axis=3) & np.isfinite(reference["all_start_profile_sse_history_mm2"]) & successful
    owners = np.broadcast_to(np.arange(len(y4))[:, None, None], valid.shape)[valid]
    targets = T[valid]
    reference_sse = reference["all_start_profile_sse_history_mm2"][valid]
    keys = np.column_stack((owners, targets[:, 1:]))
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    fast_A = np.empty(len(first)); fast_sse = np.empty(len(first)); ok = np.empty(len(first), bool)
    reasons = np.empty(len(first), np.int16)
    for lo in range(0, len(first), batch_size):
        take = first[lo:lo+batch_size]
        estimate, _ = engine.estimate_accommodation_fast(model, y4[owners[take]], targets[take, 1:],
            mean_centered=True, refinement_iterations=32)
        fast_A[lo:lo+len(take)] = estimate["estimated_A_d"]
        fast_sse[lo:lo+len(take)] = 9*(estimate["fit_vertex_rms_um"]/1000.0)**2
        ok[lo:lo+len(take)] = estimate["converged"] & estimate["valid_map"]
        reasons[lo:lo+len(take)] = estimate["fallback_reason_bits"]
    deltaA = np.abs(fast_A[inverse]-targets[:, 0])
    deltaSSE = np.abs(fast_sse[inverse]-reference_sse)
    passed = ok[inverse] & (deltaA <= 1e-5) & (deltaSSE <= 1e-12+1e-8*np.abs(reference_sse))
    answer = {"successful_recorded_queries": len(owners), "unique_queries": len(first)}
    finite_states = np.isfinite(states).all(axis=3)
    for label, mask in (("primary", primary[owners]), ("excluded_diagnostic", ~primary[owners])):
        frame_mask = primary if label == "primary" else ~primary
        unsuccessful = finite_states[frame_mask] & ~successful[frame_mask]
        answer[label] = {"queries": int(mask.sum()), "passed": bool(passed[mask].all()),
            "failure_count": int((~passed[mask]).sum()), "max_A_delta_D": float(deltaA[mask].max(initial=0)),
            "max_scalar_SSE_delta_mm2": float(deltaSSE[mask].max(initial=0)),
            "recorded_unsuccessful_profiles": int(unsuccessful.sum())}
    answer["fallback_reason_bits_counts_unique"] = {str(int(bit)): int(np.sum(reasons == bit)) for bit in np.unique(reasons)}
    return answer


def pick_pilot(states, common_rows, primary, seed=20261107):
    rng = np.random.default_rng(seed)
    eligible = common_rows[primary]
    selected = []
    for plane, A in enumerate(np.unique(states[eligible, 0])):
        pool = eligible[states[eligible, 0] == A]
        boundary = pool[np.any(np.isclose(np.abs(states[pool, 1:]), 12.4, atol=1e-9, rtol=0), axis=1)]
        edge = rng.choice(boundary, min(4, len(boundary)), replace=False)
        count = 24+int(plane < 8)
        interior = np.setdiff1d(pool, edge)
        selected.extend(edge.tolist()+rng.choice(interior, count-len(edge), replace=False).tolist())
    if len(selected) != 512 or len(np.unique(selected)) != 512:
        raise AssertionError("Expected512 deterministic stratified primary states")
    return np.array(sorted(selected), int)


def synthetic_check(model, maps):
    truth = np.array([[0., 3.7, -8.2], [1.3, -10.6, 2.9], [2.7, 12.5, -12.5], [4., -12.5, 12.5]])
    m1 = np.array([1.2, -.7])+np.arange(len(truth))[:, None]*[.02, -.01]
    answer = {}
    for variant, G in maps.items():
        q = np.broadcast_to(G.mean_mm, (len(truth), 2)).copy()
        for _ in range(40):
            q -= np.linalg.solve(G.jacobian(q), (G.predict(q)-truth[:, 1:])[..., None])[..., 0]
        if np.max(np.abs(G.predict(q)-truth[:, 1:])) > 1e-9:
            raise AssertionError("Synthetic vector polynomial inversion failed")
        F = model.evaluate(*truth.T, diagnostics=True)["prediction"]
        y4 = m1[:, None]+q[:, None]+F
        result = engine.solve_fixedpoint(model, G, y4, m1, record_history=True, profile_solver="reference401")
        delta = np.max(np.abs(result["estimated_state"]-truth), axis=0)
        if not result["primary_A2_converged"].all() or np.any(delta > [1e-4, 1e-3, 1e-3]):
            raise AssertionError((variant, delta, result["primary_A2_converged"]))
        if np.any(result["primary_A2_converged"] & (result["selected_start_index"] != 1)):
            raise AssertionError("Converged A2 primary replaced")
        answer[variant] = {"max_state_component_error": delta.tolist(), "all_primary_converged": True}
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("check", "pilot", "full"), default="pilot")
    parser.add_argument("--calibration-dir", type=Path, default=BASE/"calibration")
    parser.add_argument("--pilot-dir", type=Path, default=BASE/"pilot")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    manifest_path = args.calibration_dir/"manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if digest(manifest["source_path"]) != manifest["source_sha256"]:
        raise RuntimeError("Source changed")
    if digest(args.calibration_dir/"scope.npz") != manifest["scope_sha256"]:
        raise RuntimeError("Scope changed")
    Finfo = manifest["forward_model"]
    if digest(Finfo["path"]) != Finfo["sha256"]:
        raise RuntimeError("F27 changed")
    model = load_shape(Path(Finfo["path"]))
    maps = {}
    for variant in VARIANTS:
        info = manifest["maps"][variant]
        if digest(info["path"]) != info["sha256"]:
            raise RuntimeError("Frozen gaze map changed")
        maps[variant] = BivariatePolynomial.load(info["path"])
        if maps[variant].degree != 3 or maps[variant].coefficients.shape != (10, 2):
            raise AssertionError("Expected one10x2 cubic withnoA input")
    checks = synthetic_check(model, maps)
    print(json.dumps({"synthetic_checks": checks}), flush=True)
    if args.phase == "check":
        return
    with np.load(manifest["source_path"]) as source:
        states, all_ids = source["states"].copy(), source["source_indices"].copy()
        all_y4, all_m1 = source["absolute_p4_mm"].copy(), source["mean_p1_absolute_mm"].copy()
    with np.load(args.calibration_dir/"scope.npz") as scope:
        common = scope["common_source_subset_rows"].copy()
        primary_common = scope["primary_within_common"].copy()
    if len(common) != 83250 or primary_common.sum() != 83229:
        raise AssertionError("Expected83,250 diagnostic/83,229 primary rows")
    excluded = common[~primary_common]
    fingerprints = {"manifest": digest(manifest_path), "core": digest(ROOT/"Script/p4_shared_gaze_accommodation_fixedpoint.py"),
        "fast_profile": digest(ROOT/"Script/p4_fast_accommodation_profile.py"), "runner": digest(__file__)}
    out = (args.output_dir or BASE/args.phase).resolve()
    if out.exists():
        raise FileExistsError(f"Fresh output required:{out}")
    out.mkdir(parents=True)
    summaries = {}
    if args.phase == "pilot":
        pilot = pick_pilot(states, common, primary_common)
        rows = np.concatenate((pilot, excluded))
        primary = np.arange(len(rows)) < 512
        np.savez_compressed(out/"predeclared_ids.npz", source_subset_rows=rows, source_indices=all_ids[rows],
            truth_states=states[rows], primary=primary, seed=np.array(20261107))
        for variant, G in maps.items():
            results, timings = {}, {}
            for method in ("reference401", "fast41"):
                started = time.perf_counter()
                raw = engine.solve_fixedpoint(model, G, all_y4[rows], all_m1[rows], record_history=True, profile_solver=method)
                timings[method] = time.perf_counter()-started
                compact, _ = histories_and_last_two(raw)
                results[method] = raw
                scores = score(compact, all_y4[rows], states[rows], primary, model)
                np.savez_compressed(out/f"{variant}_{method}.npz", **raw, source_indices=all_ids[rows],
                    source_subset_rows=rows, truth_states=states[rows], primary_mask=primary)
                summaries[f"{variant}_{method}"] = {"seconds": timings[method], "scores": scores}
            physical = physical_equivalence(results["reference401"], results["fast41"], all_y4[rows], primary, model)
            queries = identical_query_audit(results["reference401"], all_y4[rows], primary, model, args.batch_size)
            summaries[variant] = {"physical_equivalence": physical, "identical_query_equivalence": queries,
                "primary_numerical_pass": bool(physical["primary"]["passed"] and queries["primary"]["passed"]),
                "projected_full_fast_seconds": timings["fast41"]/len(rows)*len(common)}
        overall = all(summaries[v]["primary_numerical_pass"] for v in VARIANTS)
        summary = {"phase": "pilot", "fingerprints": fingerprints, "checks": checks,
            "primary_count": 512, "excluded_diagnostic_count": 21, "cases": summaries,
            "primary_numerical_pass": overall, "IDs_sha256": digest(out/"predeclared_ids.npz")}
    else:
        pilot_summary = json.loads((args.pilot_dir/"summary.json").read_text(encoding="utf-8"))
        if not pilot_summary["primary_numerical_pass"] or pilot_summary["fingerprints"] != fingerprints:
            raise RuntimeError("Fullrun requires source-pinned primary strict/fast pilot PASS")
        with np.load(args.pilot_dir/"predeclared_ids.npz") as ids:
            traced_ids = ids["source_indices"]
        for variant, G in maps.items():
            pieces, total_seconds = {}, 0.0
            trace_dir = out/f"{variant}_traces"
            trace_dir.mkdir()
            for batch, lo in enumerate(range(0, len(common), args.batch_size)):
                rows = common[lo:lo+args.batch_size]
                started = time.perf_counter()
                raw = engine.solve_fixedpoint(model, G, all_y4[rows], all_m1[rows], record_history=True, profile_solver="fast41")
                total_seconds += time.perf_counter()-started
                compact, history = histories_and_last_two(raw)
                interesting = (np.isin(all_ids[rows], traced_ids) | ~raw["converged"]
                    | raw["distinct_converged_roots"] | np.any(raw["all_start_cycle_events"] > 0, axis=1)
                    | np.any(raw["all_start_profile401_fallback_calls"] > 0, axis=1))
                if interesting.any():
                    np.savez_compressed(trace_dir/f"batch_{batch:04d}.npz", **{key: value[interesting] for key, value in history.items()},
                        source_indices=all_ids[rows][interesting], source_subset_rows=rows[interesting])
                for key, value in compact.items():
                    pieces.setdefault(key, []).append(value)
                if batch % 16 == 0:
                    print(json.dumps({"variant": variant, "processed": min(lo+args.batch_size, len(common)),
                                      "inference_seconds": total_seconds}), flush=True)
            result = {key: np.concatenate(value, axis=0) for key, value in pieces.items()}
            scores = score(result, all_y4[common], states[common], primary_common, model)
            np.savez_compressed(out/f"runtime_{variant}.npz", **result, source_indices=all_ids[common],
                source_subset_rows=common, truth_states=states[common], primary_mask=primary_common)
            summaries[variant] = {"inference_seconds": total_seconds, "scores": scores, "scalar_coefficients": 47}
        summary = {"phase": "full", "fingerprints": fingerprints, "checks": checks,
            "primary_variant": "A0_frozen", "control_variant": "shared75", "cases": summaries,
            "diagnostic_count": 83250, "primary_count": 83229,
            "pilot_summary_sha256": digest(args.pilot_dir/"summary.json")}
    summary.update(gaze_vector_coefficients=10, gaze_scalar_components=20, forward_scalar_coefficients=27,
        total_scalar_coefficients=47, gaze_accommodation_features=False,
        forward_change=manifest["forward_change"], frame_assumption=manifest["frame_assumption"])
    (out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "phase": args.phase,
                      "primary_numerical_pass": summary.get("primary_numerical_pass")}), flush=True)


if __name__ == "__main__":
    main()
