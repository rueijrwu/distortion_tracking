"""Explicit physical-equivalence gate for the unchanged fast39/43 runtime.

The original SSE-only pilot failures are preserved. This launcher consumes a
separate, hash-pinned physical-units audit; it changes no numerical tolerance,
estimator coefficient, scalar search or fixed-point update.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from p4_aaware_center_cubic import AwareCenterCubic, DEFAULT_ROOT
from evaluate_p4_p1_aaware_iteration import run, oracle
from evaluate_p4_p1_center_cubic_iteration import score
from evaluate_p4_p1_limited_gaze import load_shape
from fit_p4_limited_gaze import DEFAULT_OUT as SOURCE_ROOT
from fit_p4_p1_axis_cubic_gaze import frozen_quality_flags
from fit_p4_sparse_calibration import write_json

ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT/"data/p4_p1_center_cubic_integration/full_cal75_F27_C8/independent_raw_audit.json"
AUDIT_SHA = "bcf48e55307475e66f15a6ea2cb0562d4511fdd2d1d2dfa3d6692d1020fe92a9"
SAMPLE_PATH = ROOT/"data/p4_p1_center_cubic_integration/comparison_sample_21k_predeclared.npz"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strings(value):
    if isinstance(value, dict):
        return [item for child in value.values() for item in strings(child)]
    if isinstance(value, list):
        return [item for child in value for item in strings(child)]
    return [value] if isinstance(value, str) else []


def maximum_number(value):
    if isinstance(value, dict):
        return max(maximum_number(child) for child in value.values())
    return float(value)


def verify_physical_audit(audit_path, required_inputs):
    if digest(audit_path) != AUDIT_SHA:
        raise RuntimeError("Physical-equivalence artifact differs from the explicitly approved frozen SHA")
    audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    if audit.get("overall_pass") is not True:
        raise RuntimeError("Explicit physical-equivalence overall_pass is required")
    physical = audit["strict_fast_physical_equivalence"]
    thresholds = physical["physical_thresholds"]
    numeric_thresholds = [float(v) for v in thresholds.values() if isinstance(v, (int, float))]
    if len(numeric_thresholds) < 2 or not all(np.isclose(v, 1e-4, atol=0., rtol=1e-12) for v in numeric_thresholds):
        raise RuntimeError("Approved coordinate/RMS thresholds must explicitly remain1e-4um")
    if maximum_number(physical["observed_max_prediction_coordinate_shift_um"]) > 1e-4 or maximum_number(physical["observed_max_vertex_RMS_difference_um"]) > 1e-4:
        raise RuntimeError("Physical prediction/RMS equivalence threshold failed")
    if maximum_number(physical["SSE_cauchy_bound_max_violation_mm2"]) > 1e-18 or physical.get("SSE_gate_relaxed") is not False:
        raise RuntimeError("Residual bound/original-gate preservation is not verified")
    pinned = set(strings(audit["hashes"]))
    checked = {}
    for name, path in required_inputs.items():
        sha = digest(path)
        if sha not in pinned:
            raise RuntimeError(f"Current audited input hash is missing from the frozen audit:{name}:{path}")
        checked[name] = {"path": str(path), "sha256": sha}
    return audit, checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("C12", "C16"), default="C12")
    parser.add_argument("--source-dir", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_ROOT/"calibration")
    parser.add_argument("--pilot-dir", type=Path, default=DEFAULT_ROOT/"runtime_pilot")
    parser.add_argument("--audit", type=Path, default=AUDIT_PATH)
    parser.add_argument("--sample-ids", type=Path, default=SAMPLE_PATH)
    parser.add_argument("--profile-validation", type=Path, default=DEFAULT_ROOT/"profile_validation/summary.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk", type=int, default=512)
    args = parser.parse_args()
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    with threadpool_limits(limits=6):
        calibration_path = args.calibration_dir/"summary.json"
        calibrated = json.loads(calibration_path.read_text(encoding="utf-8"))
        if calibrated["selection"]["selected"] != "C12":
            raise RuntimeError("This explicit approval is for calibration-only selected C12/full and C16/sample")
        source = args.source_dir/"source_subset.npz"
        Fpath = args.source_dir/"models/new75_DF27.npz"
        Cpath = Path(calibrated["models"][args.case]["path"])
        inputs = {"source_subset": source, "F27": Fpath,
            "aware_core": ROOT/"Script/p4_center_cubic_fixedpoint_aware.py",
            "aware_runner": ROOT/"Script/evaluate_p4_p1_aaware_iteration.py",
            "fast_profile": ROOT/"Script/p4_fast_accommodation_profile.py",
            "profile_replay_audit": args.profile_validation}
        audit, pins = verify_physical_audit(args.audit, inputs)
        replay = json.loads(args.profile_validation.read_text(encoding="utf-8"))
        reference_path = ROOT/"Script/p4_known_gaze_inverse.py"
        if replay.get("all_pass") is not True or replay["reference_source_sha256"] != digest(reference_path):
            raise RuntimeError("Reference401 source differs from the hash-pinned replay audit")
        pins["reference_profile_transitively_pinned_by_replay"] = {"path": str(reference_path), "sha256": digest(reference_path)}
        if digest(Cpath) != calibrated["models"][args.case]["sha256"]:
            raise RuntimeError("Current corrected-center model differs from calibration metadata")
        feature_path = ROOT/"Script/p4_aaware_center_cubic.py"
        feature_bytes = feature_path.read_bytes()
        feature_sha = hashlib.sha256(feature_bytes).hexdigest()
        feature_provenance = {
            "calibration_source_sha256": calibrated["script_sha256"],
            "runtime_source_sha256": feature_sha,
            "compatibility": "exact original source plus audited unused adapter insertion",
        }
        # Permit only this independently audited, unused class insertion.
        adapter_start, adapter_stop = 654, 979
        audited_sha = "758d6593d88dba0ae69fa98c631423a8e25c830116debad8af6f910d7f32b32c"
        original_sha = "1111f3470fa0dd4d4d32d644da0e13f4671bc92a5d8e11a1ca69e4e10e467b43"
        stripped_bytes = feature_bytes[:adapter_start] + feature_bytes[adapter_stop:]
        if (feature_sha != audited_sha or calibrated["script_sha256"] != original_sha
                or hashlib.sha256(stripped_bytes).hexdigest() != original_sha):
            raise RuntimeError("A-aware feature/prediction source changed since calibration")
        feature_provenance.update(
            adapter_byte_range=[adapter_start, adapter_stop],
            adapter_sha256=hashlib.sha256(feature_bytes[adapter_start:adapter_stop]).hexdigest(),
        )
        pins["feature_source_calibration_compatibility"] = feature_provenance
        pilot = json.loads((args.pilot_dir/"summary.json").read_text(encoding="utf-8"))
        for method in ("reference401", "fast41"):
            case = pilot["cases"][f"{args.case}_{method}"]
            if case["shape_sha256"] != digest(Fpath) or case["center_cubic_sha256"] != digest(Cpath):
                raise RuntimeError("Model differs from the strict/fast physical-audited pilot")
        pins[f"{args.case}_cross_checked_against_calibration_and_both_pilots"] = {"path": str(Cpath), "sha256": digest(Cpath)}
        with np.load(source) as stored:
            states, y4, m1, common, global_ids = (stored[key] for key in
                ("states", "absolute_p4_mm", "mean_p1_absolute_mm", "common_holdout", "source_indices"))
        assert np.sum(common) == 83250
        with np.load(args.sample_ids) as frozen:
            sample_global = frozen["source_indices"].astype(int)
        lookup = np.full(21*101*101, -1, int)
        lookup[global_ids] = np.arange(len(global_ids))
        random = lookup[sample_global]
        if len(random) != 21000 or len(np.unique(random)) != 21000 or np.any(random < 0) or not np.all(common[random]):
            raise RuntimeError("The exact predeclared21k sample must match common restricted rows")
        flags = common & frozen_quality_flags(states)
        flagged = np.flatnonzero(flags)
        full = args.case == "C12"
        selection = np.flatnonzero(common) if full else np.unique(np.r_[random, flagged])
        model, C = load_shape(Fpath), AwareCenterCubic.load(Cpath)
        if model.parameter_count != 27 or C.parameter_count != (12 if full else 16):
            raise RuntimeError("Unexpected approved model capacity")
        out.mkdir(parents=True, exist_ok=True)
        summary = {"case": args.case, "shape_values": 27, "center_values": C.parameter_count,
            "total_values": 27+C.parameter_count, "full_common": full, "evaluated_states": len(selection),
            "population_common": 83250, "sample_states": 21000, "complete_flagged_stress_states": len(flagged),
            "equivalence_approval": {"artifact": str(args.audit), "sha256": AUDIT_SHA,
                "basis": "separate physical-units/residual-bound PASS; original SSE-only failures remain unchanged",
                "physical_thresholds": audit["strict_fast_physical_equivalence"]["physical_thresholds"], "checked_inputs": pins},
            "profile_solver": "fast41 with unchanged401 fallback", "solver_or_model_change": False,
            "original_pilot_guard": pilot.get("strict_fast_equivalence"), "calibration_selection": calibrated["selection"],
            "sample_ID_artifact_sha256": digest(args.sample_ids), "launcher_sha256": digest(__file__)}
        write_json(out/"summary.json", summary)
        np.savez_compressed(out/"frozen_selection.npz", evaluated_source_indices=global_ids[selection],
            random21000_source_indices=sample_global, flagged_source_indices=global_ids[flagged])
        result, seconds = run(model, C, y4, m1, selection, global_ids, out, f"runtime_{args.case}", "fast41", args.chunk)
        np.savez_compressed(out/f"runtime_{args.case}.npz", source_indices=global_ids[selection], truth_state=states[selection],
            true_central_ray_mm=y4[selection, 4], signed_state_error=result["estimated_state"]-states[selection], **result)
        oracle_values = oracle(C, y4[selection], m1[selection], states[selection])  # score only
        endpoint = np.isclose(states[selection, 0], 0.) | np.isclose(states[selection, 0], 4.)
        random_mask = np.isin(selection, random)
        masks = {"full_common" if full else "evaluated_union_not_population": np.ones(len(selection), bool),
            "random21000": random_mask, "flagged": flags[selection], "unflagged": ~flags[selection],
            "interior_A": ~endpoint, "endpoint_A": endpoint}
        if not full:
            masks["random_unflagged"] = random_mask & ~flags[selection]
        summary["runtime"] = {"wall_seconds": seconds,
            "all_start_profile_queries": int(np.sum(result["all_start_profile_calls"])),
            "all_start_401_fallback_queries": int(np.sum(result["all_start_profile401_fallback_calls"])),
            "metrics": {name: score(result, states[selection], y4[selection, 4], oracle_values, mask)
                        for name, mask in masks.items() if np.any(mask)}}
        write_json(out/"summary.json", summary)


if __name__ == "__main__":
    main()
