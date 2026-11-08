"""Paired P4/P1 estimated-center feedback with an independently fitted P1 bias."""
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

from p1_centroid_bias import P1CentroidBias
from p4_a0_bivariate_polynomial import BivariatePolynomial
from evaluate_p4_p1_limited_gaze import load_shape
import evaluate_p4_p1_compact_accommodation as base
import p4_p1_both_center_fixedpoint as engine

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"data/p4_p1_both_centers"
SCORE_KEYS = ("estimated_state", "primary_A2_state", "primary_first_state", "raw_initial_gaze",
    "bounded_initial_gaze", "corrected_center_absolute_mm", "converged", "primary_A2_converged",
    "no_valid_start", "selected_start_index", "distinct_converged_roots", "boundary_A_ADE_BDE",
    "iteration_cap", "all_start_profile_calls", "all_start_profile401_fallback_calls",
    "all_start_cycle_events", "convergence_history_consistent")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gaze_metric(estimate, truth):
    e = np.asarray(estimate)-np.asarray(truth)
    good = e[np.isfinite(e).all(axis=1)]
    if not len(good):
        return {"N": len(e), "finite": 0}
    r = np.linalg.norm(good, axis=1)
    return {"N": len(e), "finite": len(good), "bias_ADE_BDE_deg": good.mean(axis=0).tolist(),
        "RMS_ADE_BDE_deg": np.sqrt(np.mean(good*good, axis=0)).tolist(),
        "vector_RMS_deg": float(np.sqrt(np.mean(r*r))), "median_deg": float(np.median(r)),
        "P95_deg": float(np.percentile(r, 95)), "max_deg": float(r.max())}


def dual_scores(result, y4, m1, truth, labeled_c1, primary, F4, G, B1):
    scores = base.score(result, y4, truth, primary, F4)
    if B1 is None:
        c1_selected = c1_primary = c1_first = m1
        oracle_input = y4[:, 4]-m1  # old G retains its correctly calibrated domain
        conditional_c1 = m1
    else:
        c1_selected = m1-B1.predict(result["estimated_state"][:, 1:])
        c1_primary = m1-B1.predict(result["primary_A2_state"][:, 1:])
        c1_first = result["first_corrected_p1_center_mm"]
        conditional_c1 = m1-B1.predict(truth[:, 1:])
        oracle_input = y4[:, 4]-labeled_c1
    oracle_gaze = G.predict(oracle_input)
    true_difference = y4[:, 4]-labeled_c1
    estimated_difference = result["corrected_center_absolute_mm"]-c1_selected
    for name, mask in (("primary", primary), ("all_diagnostic", np.ones(len(truth), bool)),
                        ("excluded_diagnostic", ~primary)):
        scores[name]["P1_uncorrected_mean_as_center"] = base.center_metrics(m1[mask], labeled_c1[mask])
        scores[name]["P1_first_center"] = base.center_metrics(c1_first[mask], labeled_c1[mask])
        scores[name]["P1_operational_final_primary_center"] = base.center_metrics(c1_primary[mask], labeled_c1[mask])
        scores[name]["P1_operational_final_selected_center"] = base.center_metrics(c1_selected[mask], labeled_c1[mask])
        scores[name]["P1_conditional_supplied_true_gaze_diagnostic"] = base.center_metrics(conditional_c1[mask], labeled_c1[mask])
        scores[name]["operational_center_difference"] = base.center_metrics(estimated_difference[mask], true_difference[mask])
        scores[name]["oracle_calibration_style_center_input_gaze"] = gaze_metric(oracle_gaze[mask], truth[mask, 1:])
    return scores


def synthetic_check(F4, G, biases):
    truth = np.array([[0., 3.7, -8.2], [1.3, -10.6, 2.9], [2.7, 12.5, -12.5], [4., -12.5, 12.5]])
    q = np.broadcast_to(G.mean_mm, (len(truth), 2)).copy()
    for _ in range(40):
        q -= np.linalg.solve(G.jacobian(q), (G.predict(q)-truth[:, 1:])[..., None])[..., 0]
    c1 = np.array([.4, 1.7])+np.arange(len(truth))[:, None]*[.02, -.01]
    c4 = c1+q
    F = F4.evaluate(*truth.T, diagnostics=True)["prediction"]
    y4 = c4[:, None]+F
    answer = {}
    for name, B1 in biases.items():
        bias = B1.predict(truth[:, 1:])
        m1 = c1+bias
        identity4 = np.max(np.abs(y4.mean(axis=1)-F.mean(axis=1)-c4))
        identity1 = np.max(np.abs(m1-bias-c1))
        q_identity = np.max(np.abs(y4.mean(axis=1)-m1-F.mean(axis=1)+bias-q))
        raw = engine.solve_fixedpoint(F4, G, y4, m1, p1_bias_model=B1, record_history=True, profile_solver="reference401")
        error = np.max(np.abs(raw["estimated_state"]-truth), axis=0)
        center1_error = np.max(np.abs(raw["corrected_p1_center_absolute_mm"]-c1))
        center4_error = np.max(np.abs(raw["corrected_center_absolute_mm"]-c4))
        if (not raw["primary_A2_converged"].all() or np.any(error > [1e-4, 1e-3, 1e-3])
                or max(identity1, identity4, q_identity) > 1e-12 or max(center1_error, center4_error) > 1e-4):
            raise AssertionError((name, error, identity1, identity4, q_identity, center1_error, center4_error))
        if np.any(raw["primary_A2_converged"] & (raw["selected_start_index"] != 1)):
            raise AssertionError("A lower-SSE alternate replaced the converged primary")
        answer[name] = {"max_state_component_error": error.tolist(), "P1_center_error_mm": float(center1_error),
            "P4_center_error_mm": float(center4_error), "P1_identity_mm": float(identity1),
            "P4_identity_mm": float(identity4), "correct_P1_plus_sign_identity_mm": float(q_identity),
            "nonzero_P1_reference_bias_mm": B1.reference_bias_mm.tolist()}
    return answer


def paired_equivalence(reference, fast, y4, primary, F4):
    comparison = base.physical_equivalence(reference, fast, y4, primary, F4)
    d1 = np.max(np.abs(reference["corrected_p1_center_absolute_mm"]-fast["corrected_p1_center_absolute_mm"]), axis=1)
    for name, mask in (("primary", primary), ("excluded_diagnostic", ~primary)):
        comparison[name]["max_P1_center_component_delta_mm"] = float(d1[mask].max())
        comparison[name]["passed"] = bool(comparison[name]["passed"] and np.all(d1[mask] <= 1e-6))
    return comparison


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
    for info in (manifest["paired_source"], manifest["P1_source"], manifest["forward_P4"], manifest["gaze_C20"]):
        if digest(info["path"]) != info["sha256"]:
            raise RuntimeError("Frozen calibrated input changed")
    if digest(args.calibration_dir/"paired_data.npz") != manifest["paired_data_sha256"]:
        raise RuntimeError("Paired P1/P4 labels/coordinates changed")
    F4 = load_shape(Path(manifest["forward_P4"]["path"]))
    G = BivariatePolynomial.load(manifest["gaze_C20"]["path"])
    names = [manifest["P1_selection"]["selected"]]
    if names[0] == "B3":
        names.append("B2")
    biases = {}
    for name in names:
        info = manifest["P1_candidates"][name]
        if not info["eligible"] or digest(info["path"]) != info["sha256"]:
            raise RuntimeError("P1 selected/control model isnotfullrank/frozen")
        biases[name] = P1CentroidBias.load(info["path"])
    checks = synthetic_check(F4, G, biases)
    print(json.dumps({"coupled_synthetic_checks": checks}), flush=True)
    if args.phase == "check":
        return
    with np.load(manifest["paired_source"]["path"]) as source:
        states, ids = source["states"].copy(), source["source_indices"].copy()
        y4_all, m1_all = source["absolute_p4_mm"].copy(), source["mean_p1_absolute_mm"].copy()
    with np.load(args.calibration_dir/"paired_data.npz") as paired:
        common = paired["common_source_subset_rows"].copy()
        primary_common = paired["primary_within_common"].copy()
        c1_all = paired["p1_labeled_center_mm"][paired["source_p1_rows"]].copy()
    if len(common) != 83250 or primary_common.sum() != 83229:
        raise AssertionError("Expected83,250diagnostic/83,229primary states")
    excluded = common[~primary_common]
    fingerprints = {"manifest": digest(manifest_path), "core": digest(ROOT/"Script/p4_p1_both_center_fixedpoint.py"),
        "P1_bias_source": digest(ROOT/"Script/p1_centroid_bias.py"),
        "fast_profile": digest(ROOT/"Script/p4_fast_accommodation_profile.py"), "runner": digest(__file__),
        "validation_helpers": digest(ROOT/"Script/evaluate_p4_p1_compact_accommodation.py")}
    out = (args.output_dir or BASE/args.phase).resolve()
    if out.exists():
        raise FileExistsError(f"Fresh output required:{out}")
    out.mkdir(parents=True)
    summaries = {}
    if args.phase == "pilot":
        pilot = base.pick_pilot(states, common, primary_common)
        rows = np.concatenate((pilot, excluded))
        primary = np.arange(len(rows)) < 512
        np.savez_compressed(out/"predeclared_ids.npz", source_subset_rows=rows, source_indices=ids[rows],
            truth_states=states[rows], primary=primary, seed=np.array(20261107))
        for name, B1 in biases.items():
            results, timings = {}, {}
            for method in ("reference401", "fast41"):
                started = time.perf_counter()
                raw = engine.solve_fixedpoint(F4, G, y4_all[rows], m1_all[rows], p1_bias_model=B1,
                    record_history=True, profile_solver=method)
                timings[method] = time.perf_counter()-started
                compact, _ = base.histories_and_last_two(raw)
                results[method] = raw
                scores = dual_scores(compact, y4_all[rows], m1_all[rows], states[rows], c1_all[rows], primary, F4, G, B1)
                np.savez_compressed(out/f"{name}_{method}.npz", **raw, source_indices=ids[rows],
                    source_subset_rows=rows, truth_states=states[rows], primary_mask=primary)
                summaries[f"{name}_{method}"] = {"seconds": timings[method], "scores": scores}
            physical = paired_equivalence(results["reference401"], results["fast41"], y4_all[rows], primary, F4)
            queries = base.identical_query_audit(results["reference401"], y4_all[rows], primary, F4, args.batch_size)
            summaries[name] = {"physical_equivalence": physical, "identical_query_equivalence": queries,
                "primary_numerical_pass": bool(physical["primary"]["passed"] and queries["primary"]["passed"]),
                "projected_full_fast_seconds": timings["fast41"]/len(rows)*len(common)}
        summary = {"phase": "pilot", "fingerprints": fingerprints, "checks": checks, "cases": summaries,
            "primary_numerical_pass": all(summaries[name]["primary_numerical_pass"] for name in names),
            "IDs_sha256": digest(out/"predeclared_ids.npz"), "primary_count": 512, "excluded_diagnostic_count": 21}
    else:
        pilot = json.loads((args.pilot_dir/"summary.json").read_text(encoding="utf-8"))
        if not pilot["primary_numerical_pass"] or pilot["fingerprints"] != fingerprints:
            raise RuntimeError("Fullrun requires frozen-source strict/fastpilotPASS")
        with np.load(args.pilot_dir/"predeclared_ids.npz") as data:
            trace_ids = data["source_indices"]
        for name, B1 in biases.items():
            pieces, seconds = {}, 0.
            trace_dir = out/f"{name}_traces"
            trace_dir.mkdir()
            for batch, lo in enumerate(range(0, len(common), args.batch_size)):
                rows = common[lo:lo+args.batch_size]
                started = time.perf_counter()
                raw = engine.solve_fixedpoint(F4, G, y4_all[rows], m1_all[rows], p1_bias_model=B1,
                    record_history=True, profile_solver="fast41")
                seconds += time.perf_counter()-started
                compact, history = base.histories_and_last_two(raw)
                interesting = (np.isin(ids[rows], trace_ids) | ~raw["converged"] | raw["distinct_converged_roots"]
                    | np.any(raw["all_start_cycle_events"] > 0, axis=1)
                    | np.any(raw["all_start_profile401_fallback_calls"] > 0, axis=1))
                if interesting.any():
                    np.savez_compressed(trace_dir/f"batch_{batch:04d}.npz", **{k: v[interesting] for k, v in history.items()},
                        source_indices=ids[rows][interesting], source_subset_rows=rows[interesting])
                for key, value in compact.items():
                    pieces.setdefault(key, []).append(value)
                if batch % 16 == 0:
                    print(json.dumps({"case": name, "processed": min(lo+args.batch_size, len(common)),
                                      "inference_seconds": seconds}), flush=True)
            result = {key: np.concatenate(value, axis=0) for key, value in pieces.items()}
            scores = dual_scores(result, y4_all[common], m1_all[common], states[common], c1_all[common], primary_common, F4, G, B1)
            np.savez_compressed(out/f"runtime_{name}.npz", **result, source_indices=ids[common],
                source_subset_rows=common, truth_states=states[common], labeled_P1_center_mm=c1_all[common],
                primary_mask=primary_common, P1_center_error_mm=result["corrected_p1_center_absolute_mm"]-c1_all[common])
            summaries[name] = {"inference_seconds": seconds, "scores": scores,
                "P1_fitted_coefficients": B1.coefficient_count, "total_fitted_coefficients": 47+B1.coefficient_count}
        # Fair saved baseline: its own q=c4-P1mean-trained G and identical IDs.
        old = manifest["baseline_before_P1_correction"]
        if digest(old["full_runtime_path"]) != old["full_runtime_sha256"] or digest(old["map_path"]) != old["map_sha256"]:
            raise RuntimeError("Frozen matched before-P1 baseline changed")
        with np.load(old["full_runtime_path"]) as saved:
            if not np.array_equal(saved["source_indices"], ids[common]) or not np.array_equal(saved["truth_states"], states[common]):
                raise AssertionError("Matched baseline/testlabels differ")
            baseline = {key: saved[key].copy() for key in SCORE_KEYS}
        oldG = BivariatePolynomial.load(old["map_path"])
        summaries["before_P1_correction_baseline47"] = {"total_fitted_coefficients": 47,
            "scores": dual_scores(baseline, y4_all[common], m1_all[common], states[common], c1_all[common], primary_common, F4, oldG, None),
            "source_sha256": old["full_runtime_sha256"], "own_correctly_calibrated_G_retained": True}
        summary = {"phase": "full", "fingerprints": fingerprints, "checks": checks, "cases": summaries,
            "primary_count": 83229, "diagnostic_count": 83250,
            "pilot_summary_sha256": digest(args.pilot_dir/"summary.json")}
    summary.update(P1_selected_by_calibration=names[0], P1_reference_bias_mm=manifest["P1_reference_bias_mm"],
        gaze_vector_coefficients=10, gaze_scalar_coefficients=20, P4_forward_coefficients=27,
        P1_reference_vector_recorded_separately=True, P1_accommodation_features=False,
        gaze_accommodation_features=False, frame_assumption=manifest["frame_assumption"])
    (out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"phase": args.phase, "output": str(out),
                      "primary_numerical_pass": summary.get("primary_numerical_pass")}), flush=True)


if __name__ == "__main__":
    main()
