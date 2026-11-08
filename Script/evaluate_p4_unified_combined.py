"""Blind joint P4 accommodation/gaze with independent P1 gaze and fixed-A control."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from evaluate_p1_p4_a0_independent_gaze import infer_channel, gaze_metrics
from evaluate_p4_p1_a0_only import center_metrics
from p4_unified_a_gaze_model import load_frozen_a0
from p4_unified_center_motion import CentralMotion
from p4_unified_fullshape import UnifiedP4Field
from p4_unified_joint_solver import solve_fixed_A_control, solve_joint

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/p4_p1_limited_gaze_12p5/source_subset.npz"
A0_OBS = ROOT / "data/p4_p1_a0_only/calibration/a0_observations.npz"
PILOT_IDS = ROOT / "data/p4_p1_compact_full_accommodation/pilot/predeclared_ids.npz"
OLD_A0_RUNTIME = ROOT / "data/p1_p4_a0_independent_gaze/runtime/runtime_independent_P1_P4.npz"
MOTION_SUMMARY = ROOT / "data/p4_unified_combined/calibration/center_motion/summary.json"
FIELD_SUMMARY = ROOT / "data/p4_unified_combined/calibration/full_field_profile_final/summary.json"
OUT = ROOT / "data/p4_unified_combined/runtime"
EXCLUDED_GAZE = np.array([12., 9.6])
JOINT_KEYS = ("estimated_state", "estimated_A_d", "gaze_deg", "center_mm",
    "center_A0_reference_mm", "converged", "valid", "selected_start", "shape_sse_mm2",
    "fixedpoint_residual_normalized", "raw_G4_deg", "bounded_proposal_state",
    "profile_vertex_rms_um", "profile_boundary_A", "profile_local_minima",
    "initial_P1_conditioned_profile_A_d", "initial_P1_conditioned_profile_valid",
    "initial_P1_conditioned_profile_vertex_rms_um", "initial_P1_conditioned_profile_boundary_A",
    "initial_P1_conditioned_profile_local_minima", "raw_independent_G4_from_mean_deg",
    "first_center_mm", "first_center_A0_reference_mm", "first_raw_G4_deg",
    "first_proposal_state", "first_shape_sse_mm2", "primary_state",
    "primary_converged", "primary_valid", "alternate_start_rescue", "all_initial_state",
    "all_best_state", "all_converged", "all_valid", "all_invalid", "all_iterations",
    "all_alpha", "all_growth_events", "all_cycle_events", "all_bound_pressure",
    "all_shape_sse_mm2", "all_fixedpoint_residual_normalized", "all_profile_calls",
    "all_profile_fallbacks", "bound_A_ADE_BDE")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scalar_metrics(predicted, truth):
    error = np.asarray(predicted)-np.asarray(truth)
    finite = np.isfinite(error)
    good = error[finite]
    if not len(good):
        return {"attempted": len(error), "finite": 0}
    return {"attempted": len(error), "finite": int(finite.sum()),
            "bias_d": float(good.mean()), "RMS_d": float(np.sqrt(np.mean(good*good))),
            "MAE_d": float(np.mean(np.abs(good))),
            "median_abs_d": float(np.median(np.abs(good))),
            "P95_abs_d": float(np.percentile(np.abs(good), 95)),
            "max_abs_d": float(np.max(np.abs(good)))}


def joint_telemetry(result, mask):
    selected = result["selected_start"][mask]
    return {"attempted": int(mask.sum()),
        "primary_converged": int(result["primary_converged"][mask].sum()),
        "selected_converged": int(result["converged"][mask].sum()),
        "nonconverged": int((~result["converged"][mask]).sum()),
        "invalid": int((~result["valid"][mask]).sum()),
        "alternate_rescue": int(result["alternate_start_rescue"][mask].sum()),
        "converged_primary_replaced": int(np.sum(result["primary_converged"][mask] & (selected != 1))),
        "selected_start_0_primary_2": np.bincount(selected, minlength=3).tolist(),
        "final_bound_A_ADE_BDE": result["bound_A_ADE_BDE"][mask].sum(axis=0).tolist(),
        "all_start_profile_calls": int(result["all_profile_calls"][:, mask].sum()),
        "all_start_fast_to_strict_fallbacks": int(result["all_profile_fallbacks"][:, mask].sum()),
        "all_start_growth_events": int(result["all_growth_events"][:, mask].sum()),
        "all_start_cycle_events": int(result["all_cycle_events"][:, mask].sum()),
        "max_iterations_per_start": int(result["all_iterations"][:, mask].max())}


def score(result, control, p1_gaze, p1_center, truth, true_p4_center, true_p1_center,
          true_a0_p4_center, masks):
    groups = {}
    for name, mask in masks.items():
        if not np.any(mask):
            continue
        group = {
            "joint_A": scalar_metrics(result["estimated_A_d"][mask], truth[mask, 0]),
            "joint_gaze": gaze_metrics(result["gaze_deg"][mask], truth[mask, 1:]),
            "joint_P4_center": center_metrics(result["center_mm"][mask], true_p4_center[mask]),
            "joint_P4_center_A0_normalized_scoring_only": center_metrics(
                result["center_A0_reference_mm"][mask], true_a0_p4_center[mask]),
            "initial_P1_conditioned_A": scalar_metrics(
                result["initial_P1_conditioned_profile_A_d"][mask], truth[mask, 0]),
            "raw_independent_G4_from_mean": gaze_metrics(
                result["raw_independent_G4_from_mean_deg"][mask], truth[mask, 1:]),
            "first_primary_proposal_gaze": gaze_metrics(
                result["first_proposal_state"][1, mask, 1:], truth[mask, 1:]),
            "P1_independent_gaze": gaze_metrics(p1_gaze[mask], truth[mask, 1:]),
            "P1_independent_center": center_metrics(p1_center[mask], true_p1_center[mask]),
            "P1_P4_gaze_agreement_diagnostic_only": gaze_metrics(
                result["gaze_deg"][mask], p1_gaze[mask]),
            "fixed_P1_conditioned_A_control_gaze": gaze_metrics(
                control["gaze_deg"][mask], truth[mask, 1:]),
            "fixed_P1_conditioned_A_control_center": center_metrics(
                control["center_mm"][mask], true_p4_center[mask]),
            "fixed_P1_conditioned_A_control_A": scalar_metrics(
                control["held_A_d"][mask], truth[mask, 0]),
            "telemetry": joint_telemetry(result, mask),
            "control_telemetry": {"converged": int(control["converged"][mask].sum()),
                                  "invalid": int((~control["valid"][mask]).sum()),
                                  "bound_ADE_BDE": control["bound_ADE_BDE"][mask].sum(axis=0).tolist()},
        }
        shape_rms = np.sqrt(np.maximum(result["shape_sse_mm2"][mask], 0)/9.)*1000.
        finite = shape_rms[np.isfinite(shape_rms)]
        group["joint_centered_P4_grid_fit_vertex_RMS_um"] = float(np.sqrt(np.mean(finite*finite))) if len(finite) else None
        groups[name] = group
    return groups


def load_models(field_summary_path, motion_summary_path):
    frozen = load_frozen_a0()
    field_summary = json.loads(Path(field_summary_path).read_text(encoding="utf-8"))
    artifact = Path(field_summary["artifact_path"])
    if digest(artifact) != field_summary["artifact_sha256"] or field_summary["new_extension_values"] != 15:
        raise RuntimeError("Frozen 15-value P4 field extension changed")
    with np.load(artifact) as saved:
        extension = saved["extension15"].copy()
        if (int(saved["full_family_parameter_count"]) != 27
                or not np.array_equal(saved["frozen_base12"], frozen.p4_shape.coefficients *
                    np.array([(20/12.4)**(ix+iy) for _, ix, iy in __import__("p4_sparse_model").SPATIAL]))):
            raise RuntimeError("Frozen A=0 P4 basis conversion changed")
    field = UnifiedP4Field(frozen, extension)
    motion_summary = json.loads(Path(motion_summary_path).read_text(encoding="utf-8"))
    selected = motion_summary["selection"]["selected"]
    motion_info = motion_summary["candidates"][selected]
    if selected != "M1" or digest(motion_info["path"]) != motion_info["sha256"]:
        raise RuntimeError("Expected selected two-value positive M1 central motion")
    motion = CentralMotion.load(motion_info["path"])
    if motion.coefficient_count != 2 or not np.array_equal(motion.pivot_mm, np.zeros(2)):
        raise RuntimeError("Expected measured-zero-pivot M1")
    return frozen, field, motion, {"field_summary_sha256": digest(field_summary_path),
                                   "field_artifact_sha256": digest(artifact),
                                   "motion_summary_sha256": digest(motion_summary_path),
                                   "motion_artifact_sha256": digest(motion_info["path"]),
                                   "frozen_A0_hashes": frozen.source_hashes}


def forced_a0_identity(frozen, field, motion):
    """Known-A=0 control; does not feed true A into the unknown-A solver."""
    with np.load(A0_OBS) as obs:
        rows = np.flatnonzero(obs["holdout25"])
        y4 = obs["absolute_p4_mm"][rows].copy()
        ids = obs["source_indices"][rows].copy()
        gaze = obs["gaze_deg"][rows].copy()
    forced, _ = infer_channel(y4, frozen.p4_bias, frozen.p4_gaze)
    f0 = field.evaluate(np.zeros(len(y4)), forced["gaze_deg"])
    mean_delta = float(np.max(np.abs(f0.mean(axis=1)-frozen.p4_bias.predict(forced["gaze_deg"]))))
    normalize_delta = float(np.max(np.abs(motion.inverse(forced["center_mm"], np.zeros(len(y4)))-forced["center_mm"])))
    with np.load(OLD_A0_RUNTIME) as saved:
        if not np.array_equal(saved["source_indices"], ids):
            raise RuntimeError("Old independent P4 A=0 runtime has different source IDs")
        old_gaze = saved["p4_gaze_deg"].copy()
        old_center = saved["p4_center_mm"].copy()
    gaze_delta = float(np.max(np.abs(forced["gaze_deg"]-old_gaze)))
    center_delta = float(np.max(np.abs(forced["center_mm"]-old_center)))
    if max(mean_delta, normalize_delta, gaze_delta, center_delta) > 1e-10:
        raise RuntimeError("Forced-known-A0 reduction differs from frozen independent P4 runtime")
    excluded = np.all(np.isclose(gaze, EXCLUDED_GAZE, atol=1e-9, rtol=0), axis=1)
    return {"heldout": len(rows), "primary_after_exclusion": int((~excluded).sum()),
            "P4_F0_mean_vs_B4_max_abs_mm": mean_delta,
            "M1_inverse_at_A0_identity_max_abs_mm": normalize_delta,
            "frozen_runtime_gaze_max_abs_deg": gaze_delta,
            "frozen_runtime_center_max_abs_mm": center_delta,
            "forced_A0_primary_gaze": gaze_metrics(forced["gaze_deg"][~excluded], gaze[~excluded]),
            "source_sha256": digest(OLD_A0_RUNTIME)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("pilot", "full"), required=True)
    parser.add_argument("--profile-method", choices=("strict401", "fast41"), required=True)
    parser.add_argument("--field-summary", type=Path, default=FIELD_SUMMARY)
    parser.add_argument("--motion-summary", type=Path, default=MOTION_SUMMARY)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    out = (args.output_dir or OUT / f"{args.scope}_{args.profile_method}").resolve()
    if out.exists():
        raise FileExistsError(f"Fresh output directory required: {out}")
    frozen, field, motion, model_hashes = load_models(args.field_summary, args.motion_summary)
    source_sha = digest(SOURCE)
    if source_sha != "97c6db1b455f9abf7354351fa768cca7eea67198d8a4bfd14bbe684475462daf":
        raise RuntimeError("Pinned paired source changed")
    with np.load(SOURCE) as raw:
        all_states = raw["states"].copy()
        all_ids = raw["source_indices"].copy()
        all_y4 = raw["absolute_p4_mm"].copy()
        all_m1 = raw["mean_p1_absolute_mm"].copy()
        common = raw["common_holdout"].astype(bool)
    if args.scope == "pilot":
        with np.load(PILOT_IDS) as pilot:
            rows = pilot["source_subset_rows"].copy()
            if (len(rows) != 533 or not np.array_equal(all_ids[rows], pilot["source_indices"])
                    or not np.array_equal(all_states[rows], pilot["truth_states"])):
                raise RuntimeError("Predeclared 512+21 pilot source identity changed")
    else:
        rows = np.flatnonzero(common)
        if len(rows) != 83250:
            raise RuntimeError("Expected complete common holdout population")
    states, ids = all_states[rows], all_ids[rows]
    y4, m1 = all_y4[rows], all_m1[rows]
    excluded = np.all(np.isclose(states[:, 1:], EXCLUDED_GAZE, atol=1e-9, rtol=0), axis=1)
    primary = ~excluded
    if (excluded.sum() != 21 or (args.scope == "pilot" and primary.sum() != 512)
            or (args.scope == "full" and primary.sum() != 83229)):
        raise AssertionError("Primary/excluded scope differs from frozen population")
    # The P1 loop sees measured P1 means only. Deduplicate identical measured
    # observations, without using A or known gaze labels, for repeated A planes.
    unique_m1, inverse_rows = np.unique(m1, axis=0, return_inverse=True)
    p1_observed_centroids = np.broadcast_to(unique_m1[:, None, :], (len(unique_m1), 9, 2))
    p1_unique, _ = infer_channel(p1_observed_centroids, frozen.p1_bias, frozen.p1_gaze)
    g1 = p1_unique["gaze_deg"][inverse_rows]
    c1 = p1_unique["center_mm"][inverse_rows]
    started = time.perf_counter()
    joint = solve_joint(y4, g1, field, motion, frozen.p4_gaze, profile_method=args.profile_method)
    joint_seconds = time.perf_counter()-started
    control_started = time.perf_counter()
    control = solve_fixed_A_control(y4, joint["initial_P1_conditioned_profile_A_d"], g1,
                                    field, motion, frozen.p4_gaze)
    control_seconds = time.perf_counter()-control_started
    # Ground-truth labels and labeled centers enter only here, after inference.
    true_p4_center = y4[:, 4]
    from fit_p4_p1_both_centers import load_p1
    p1_gaze_labels, p1_grid, _ = load_p1()
    p1_lookup = {tuple(np.round(g, 8)): i for i, g in enumerate(p1_gaze_labels)}
    p1_rows = np.array([p1_lookup[tuple(np.round(g, 8))] for g in states[:, 1:]])
    true_p1_center = p1_grid[p1_rows, 4]
    with np.load(A0_OBS) as obs:
        a0_lookup = {tuple(np.round(g, 8)): c for g, c in zip(obs["gaze_deg"], obs["absolute_p4_mm"][:, 4])}
    true_a0_center = np.array([a0_lookup[tuple(np.round(g, 8))] for g in states[:, 1:]])
    masks = {"primary": primary, "inclusive_diagnostic": np.ones(len(rows), bool),
             "excluded21_diagnostic": excluded,
             "true_A0_primary": primary & np.isclose(states[:, 0], 0., atol=1e-12, rtol=0),
             "primary_interior_A": primary & (states[:, 0] > 0) & (states[:, 0] < 4),
             "primary_endpoint_A": primary & ((states[:, 0] == 0) | (states[:, 0] == 4))}
    scores = score(joint, control, g1, c1, states, true_p4_center, true_p1_center,
                   true_a0_center, masks)
    if args.scope == "full":
        scores["by_true_A_scoring_only"] = {f"{a:.1f}": score(joint, control, g1, c1,
            states, true_p4_center, true_p1_center, true_a0_center,
            {"slice": primary & np.isclose(states[:, 0], a, atol=1e-12, rtol=0)})["slice"]
            for a in np.unique(states[:, 0])}
    forced = forced_a0_identity(frozen, field, motion)
    out.mkdir(parents=True)
    saved = {f"joint_{key}": value for key, value in joint.items() if key in JOINT_KEYS}
    saved.update({f"control_{key}": value for key, value in control.items()})
    np.savez_compressed(out / "runtime.npz", **saved, source_indices=ids,
        source_subset_rows=rows, truth_state_scoring_only=states,
        primary_mask=primary, excluded_mask=excluded,
        observed_P1_mean_mm=m1, observed_P4_mean_mm=y4.mean(axis=1),
        independent_P1_gaze_deg=g1, independent_P1_center_mm=c1,
        true_P1_labeled_center_scoring_only_mm=true_p1_center,
        true_P4_labeled_center_scoring_only_mm=true_p4_center,
        true_A0_P4_center_scoring_only_mm=true_a0_center)
    summary = {"experiment": "Unified P4 accommodation/gaze using frozen independent P1 seed",
        "scope": args.scope, "profile_method": args.profile_method,
        "selection": {"attempted": len(rows), "primary": int(primary.sum()),
                      "excluded": int(excluded.sum()),
                      "pilot_source_sha256": digest(PILOT_IDS) if args.scope == "pilot" else None},
        "P1_gaze_remains_independent": True, "no_truth_A_gaze_or_center_runtime_input": True,
        "P4_G20_coefficients_frozen": True,
        "model_counts": {"frozen_independent_B1_B4_G1_G4": 58,
                         "frozen_P4_A0_shape": 12, "new_P4_field_extension": 15,
                         "new_selected_M1_motion": 2, "total_fitted_family": 87,
                         "newly_fitted_this_extension": 17},
        "root_policy": "P1-conditioned A profile is start1 primary; converged alternatives rescue by observed P4 shape SSE",
        "profile_objective": "All-nine mean-centered observed P4 grid versus corrected full F4(A,g)",
        "joint_seconds": joint_seconds, "fixed_A_control_seconds": control_seconds,
        "scores": scores, "forced_known_A0_identity_control": forced,
        "source_sha256": source_sha, "A0_observations_sha256": digest(A0_OBS),
        "runtime_npz_sha256": digest(out / "runtime.npz"),
        "model_hashes": model_hashes,
        "solver_sha256": digest(ROOT / "Script/p4_unified_joint_solver.py"),
        "driver_sha256": digest(__file__),
        "noise_scope": "Noiseless simulation/model mismatch only; no instrument-noise claim"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "scope": args.scope,
        "profile_method": args.profile_method, "joint_seconds": joint_seconds,
        "control_seconds": control_seconds,
        "primary_A_RMS_d": scores["primary"]["joint_A"]["RMS_d"],
        "primary_gaze_radial_RMS_deg": scores["primary"]["joint_gaze"]["radial"]["RMS_deg"],
        "primary_status": scores["primary"]["telemetry"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
