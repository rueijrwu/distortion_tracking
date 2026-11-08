"""Evaluate fixed-A=0 gaze inference on the unchanged limited common holdout."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "4")

import numpy as np

from p4_aaware_center_cubic import AwareCenterCubic
from p4_fixed_a0_gaze import solve_fixed_a0, synthetic_check
from evaluate_p4_p1_limited_gaze import load_shape

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/"data/p4_p1_limited_gaze_12p5/source_subset.npz"
SHAPE = ROOT/"data/p4_p1_limited_gaze_12p5/models/new75_DF27.npz"
CUBIC = ROOT/"data/p4_p1_aaware_center_cubic/calibration/corrected_C12.npz"
QUALITY = ROOT/"data/p4_p1_limited_gaze_12p5/quality_diagnostics/quality_flags.npz"
PINS = {
    "source_subset": (SOURCE, "97c6db1b455f9abf7354351fa768cca7eea67198d8a4bfd14bbe684475462daf"),
    "shape_F27": (SHAPE, "a590fb36e4561bb7f23af016d49330dabc2b038e4b7dee41288fe930445d52ab"),
    "center_C12": (CUBIC, "5a9da4f4a4f5e41d484999f1b19e3cbe514cc8c8e48f6d6c8a888646b6f8d421"),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gaze_metrics(estimate, truth):
    error = np.asarray(estimate)-np.asarray(truth)
    finite = np.isfinite(error).all(axis=1)
    e = error[finite]
    n = len(error)
    if not len(e):
        return {"attempted": n, "finite": 0, "nonfinite": n}
    radial = np.linalg.norm(e, axis=1)
    result = {"attempted": n, "finite": int(finite.sum()), "nonfinite": int((~finite).sum())}
    for axis, label in enumerate(("ADE", "BDE")):
        x = e[:, axis]
        result[label] = {
            "bias_deg": float(x.mean()), "RMSE_deg": float(np.sqrt(np.mean(x*x))),
            "MAE_deg": float(np.mean(np.abs(x))),
            "median_abs_deg": float(np.median(np.abs(x))),
            "P95_abs_deg": float(np.percentile(np.abs(x), 95)),
            "P99_abs_deg": float(np.percentile(np.abs(x), 99)),
            "max_abs_deg": float(np.max(np.abs(x))),
        }
    result["radial"] = {
        "RMS_deg": float(np.sqrt(np.mean(radial*radial))),
        "median_deg": float(np.median(radial)), "P95_deg": float(np.percentile(radial, 95)),
        "P99_deg": float(np.percentile(radial, 99)), "max_deg": float(radial.max()),
        "fraction_le_0p5_deg_all_attempted": float(np.sum(radial <= .5)/n),
    }
    return result


def center_metrics(estimate, labeled_center):
    e = (np.asarray(estimate)-np.asarray(labeled_center))*1000.0
    finite = np.isfinite(e).all(axis=1)
    if not np.any(finite):
        return {"attempted": len(e), "finite": 0}
    r = np.linalg.norm(e[finite], axis=1)
    return {"attempted": len(e), "finite": int(finite.sum()),
            "bias_xy_um": e[finite].mean(axis=0).tolist(),
            "vector_RMS_um": float(np.sqrt(np.mean(r*r))),
            "median_um": float(np.median(r)), "P95_um": float(np.percentile(r, 95)),
            "max_um": float(r.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT/"data/p4_p1_fixed_a0_gaze")
    args = parser.parse_args()
    check = synthetic_check()
    print(json.dumps({"synthetic_check": check}), flush=True)
    if args.check_only:
        return
    provenance = {}
    for name, (path, expected) in PINS.items():
        actual = digest(path)
        if actual != expected:
            raise RuntimeError(f"Frozen input changed: {name}")
        provenance[name] = {"path": str(path), "sha256": actual}
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(f"Fresh output directory required: {out}")
    source = np.load(SOURCE)
    states = source["states"]
    common = np.asarray(source["common_holdout"], dtype=bool)
    selection = np.flatnonzero(common & np.isclose(states[:, 0], 0.0, atol=1e-12, rtol=0))
    if len(selection) != 3936 or not np.all(states[selection, 0] == 0):
        raise AssertionError("Expected exactly 3,936 physical-label matched A0 common rows")
    y4 = source["absolute_p4_mm"][selection]
    m1 = source["mean_p1_absolute_mm"][selection]
    source_indices = source["source_indices"][selection]
    quality = np.load(QUALITY)
    if not np.array_equal(quality["source_indices"], source["source_indices"]):
        raise AssertionError("Quality/source row identities differ")
    flagged = quality["new_calibration_flag"][selection].astype(bool)
    if int(flagged.sum()) != 1:
        raise AssertionError("Expected the single retained P1 flagged A0 frame")
    model = load_shape(SHAPE)
    center_cubic = AwareCenterCubic.load(CUBIC)
    calls = {"forward": 0, "center_cubic": 0, "accommodation_profile": 0,
             "nonzero_accommodation_inputs": 0}

    def forward_at_zero(gaze):
        # Accommodation is constructed here, never estimated or read from truth.
        zero_A = np.zeros(len(gaze), dtype=float)
        if not np.all(zero_A == 0):
            raise AssertionError("Nonzero accommodation entered fixed-A0 inference")
        calls["forward"] += 1
        evaluated = model.evaluate(zero_A, gaze[:, 0], gaze[:, 1], diagnostics=True)
        pred = evaluated["prediction"]
        valid = np.isfinite(pred).all(axis=(1, 2))
        for name in ("denominator", "radial_tangential", "radial_radial", "brown_determinant"):
            v = np.asarray(evaluated[name])
            valid &= np.all(np.isfinite(v) & (v > 0), axis=tuple(range(1, v.ndim)))
        return pred, valid

    def cubic_at_zero(q):
        zero_A = np.zeros(len(q), dtype=float)
        if not np.all(zero_A == 0):
            raise AssertionError("Nonzero accommodation entered fixed-A0 cubic")
        calls["center_cubic"] += 1
        return center_cubic.predict(q, zero_A)

    started = time.perf_counter()
    result = solve_fixed_a0(y4, m1, forward_at_zero, cubic_at_zero, keep_history=True)
    wall = time.perf_counter()-started
    if calls["accommodation_profile"] or not np.all(result["accommodation_d"] == 0):
        raise AssertionError("A0 experiment called accommodation inference")
    # Ground truth is used only after inference, for scoring and the explicitly
    # labeled oracle-center diagnostic. It is never passed into the solver.
    truth = states[selection, 1:3]
    labeled_center = y4[:, 4, :]
    oracle_gaze = center_cubic.predict(labeled_center-m1, np.zeros(len(selection)))
    scores = {}
    stages = {
        "raw_C12_at_A0_unclipped": result["raw_initial_gaze_deg"],
        "raw_C12_at_A0_bounded": result["bounded_initial_gaze_deg"],
        "first_corrected_primary_bounded": result["first_corrected_gaze_deg"],
        "fixed_A0_primary_raw_seed": result["primary_gaze_deg"],
        "fixed_A0_selected_root": result["gaze_deg"],
        "oracle_labeled_center_C12_at_A0": oracle_gaze,
    }
    for label, mask in (("all", np.ones(len(selection), dtype=bool)),
                        ("quality_unflagged", ~flagged), ("quality_flagged", flagged)):
        scores[label] = {stage: gaze_metrics(value[mask], truth[mask])
                         for stage, value in stages.items()}
        scores[label]["center_selected"] = center_metrics(result["center_mm"][mask], labeled_center[mask])
        scores[label]["center_primary"] = center_metrics(result["primary_center_mm"][mask], labeled_center[mask])
    convergence = {
        "attempted": len(selection), "selected_converged": int(result["converged"].sum()),
        "primary_converged": int(result["primary_converged"].sum()),
        "nonconverged": int((~result["converged"]).sum()),
        "invalid_returned": int((~result["valid"]).sum()),
        "alternate_start_rescue": int(result["alternate_start_rescue"].sum()),
        "selected_start_counts_raw_zero_negative_raw": np.bincount(result["selected_start"], minlength=3).tolist(),
        "distinct_converged_roots": int(result["distinct_converged_roots"].sum()),
        "bound_ADE_BDE_counts": result["bound_ADE_BDE"].sum(axis=0).tolist(),
        "outside_sampled_12p4_count": int(result["outside_sampled_12p4"].sum()),
        "raw_initial_outside_12p5_count": int(np.any(np.abs(result["raw_initial_gaze_deg"]) > 12.5, axis=1).sum()),
        "all_start_cycle_events": int(result["all_cycle_count"].sum()),
        "all_start_growth_events": int(result["all_growth_count"].sum()),
        "max_iterations": int(result["all_iterations"].max()),
    }
    summary = {
        "experiment": "A=0 fixed in every forward/cubic call; gaze only, no accommodation inference",
        "evaluated_states": len(selection), "retained_quality_flagged_states": int(flagged.sum()),
        "shape_values": 27, "center_cubic_values": 12, "total_shared_values": 39,
        "runtime_inputs": ["nine absolute observed P4 points", "observed nine-point P1 mean"],
        "requested_bound_deg": 12.5, "actual_ray_support_deg": 12.4,
        "synthetic_check": check, "provenance": provenance,
        "core_sha256": digest(ROOT/"Script/p4_fixed_a0_gaze.py"),
        "runner_sha256": digest(__file__),
        "feature_source_sha256": digest(ROOT/"Script/p4_aaware_center_cubic.py"),
        "quality_artifact_sha256": digest(QUALITY),
        "common_holdout_count": int(common.sum()), "A_fixed_D": 0.0,
        "map_call_counts": calls, "runtime_wall_seconds": wall,
        "convergence": convergence, "metrics": scores,
        "frame_assumption": "Pinned P1/P4 rays share nominal SI coordinates; external registration not independently established",
        "noise_scope": "Noise-free simulated observations with empirical-model mismatch; all flagged observations retained",
        "selection": "Lowest observed mean-centered shape SSE among converged valid roots; otherwise smallest finite undamped fixed-point residual",
    }
    out.mkdir(parents=True)
    np.savez_compressed(out/"runtime_fixed_A0.npz", **result,
                        source_subset_row_indices=selection, source_indices=source_indices,
                        truth_state=states[selection], truth_labeled_center_mm=labeled_center,
                        gaze_error_deg=result["gaze_deg"]-truth,
                        center_error_mm=result["center_mm"]-labeled_center,
                        oracle_labeled_center_gaze_deg=oracle_gaze, quality_flagged=flagged)
    (out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "runtime_seconds": wall,
                      "convergence": convergence, "all_frame_metrics": scores["all"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
