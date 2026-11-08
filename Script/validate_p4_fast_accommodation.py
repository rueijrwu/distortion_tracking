"""Exhaustive saved-pilot query replay of fast A search against frozen401."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from p4_fast_accommodation_profile import estimate_accommodation_fast
from p4_known_gaze_inverse import estimate_accommodation as reference401
from evaluate_p4_p1_limited_gaze import load_shape
from fit_p4_limited_gaze import DEFAULT_OUT as SOURCE_ROOT
from fit_p4_sparse_calibration import write_json, log

ROOT = Path(__file__).resolve().parents[1]


class ScalarCostModel:
    """Unit-test curve embedded in nine centered xy points; not an optics fit."""
    def __init__(self, kind="linear"):
        self.kind = kind
        self.pattern = .2*np.array([(x, y) for x in (-1., 0., 1.) for y in (-1., 0., 1.)])

    def evaluate(self, A, ade, bde, state_jac=False, diagnostics=False):
        A = np.asarray(A).ravel()
        f = A if self.kind == "linear" else ((A-1.)*(A-3.) if self.kind == "multimodal" else np.zeros_like(A))
        df = np.ones_like(A) if self.kind == "linear" else (2.*A-4. if self.kind == "multimodal" else np.zeros_like(A))
        pred = (1.+.05*f[:, None, None])*self.pattern[None]
        result = {"prediction": pred}
        if diagnostics:
            result.update({key: np.ones((len(A), 9)) for key in
                ("denominator", "radial_tangential", "radial_radial", "brown_determinant")})
        if state_jac:
            J = np.zeros(pred.shape+(3,))
            J[..., 0] = .05*df[:, None, None]*self.pattern
            result["state_jacobian"] = J
        return result

    def prepare_gaze(self, ade, bde):
        outer = self
        class Prepared:
            def predict(self, A, indices=None):
                return outer.evaluate(A, np.zeros(len(A)), np.zeros(len(A)))["prediction"]
        return Prepared()


def scalar_tests():
    result = {}
    for kind, underlying in (("linear", np.array([-.2, 0., .00017, 1.234567, 3.99983, 4., 4.2])),
                             ("multimodal", np.array([1.])), ("flat", np.array([2.]))):
        model = ScalarCostModel(kind)
        gaze = np.zeros((len(underlying), 2))
        y = model.evaluate(underlying, gaze[:, 0], gaze[:, 1])["prediction"]
        ref, _ = reference401(model, y, gaze, mean_centered=True)
        fast, _ = estimate_accommodation_fast(model, y, gaze)
        delta = np.abs(ref["estimated_A_d"]-fast["estimated_A_d"])
        if np.max(delta) > 1e-5:
            raise AssertionError(f"{kind} scalar A mismatch:{delta}")
        if kind != "linear" and not np.all(fast["reference401_fallback"]):
            raise AssertionError(f"{kind} must use401fallback")
        result[kind] = {"reference_A": ref["estimated_A_d"].tolist(), "fast_A": fast["estimated_A_d"].tolist(),
            "max_delta_A_D": float(np.max(delta)), "fallback": fast["reference401_fallback"].tolist(),
            "reasons": fast["fallback_reason_bits"].tolist()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--pilot-dir", type=Path, default=ROOT/"data/p4_p1_center_cubic_integration/runtime_pilot")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"data/p4_p1_aaware_center_cubic/profile_validation")
    parser.add_argument("--single-repeat", type=int, default=20)
    args = parser.parse_args()
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=6):
        tests = scalar_tests()
        source = args.source_dir/"source_subset.npz"
        with np.load(source) as stored:
            indices, y4 = stored["source_indices"], stored["absolute_p4_mm"]
        lookup = np.full(21*101*101, -1, int)
        lookup[indices] = np.arange(len(indices))
        pilot_summary = json.loads((args.pilot_dir/"summary.json").read_text(encoding="utf-8"))
        summary = {"scalar_tests": tests, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "fast_source_sha256": hashlib.sha256((ROOT/"Script/p4_fast_accommodation_profile.py").read_bytes()).hexdigest(),
            "reference_source_sha256": hashlib.sha256((ROOT/"Script/p4_known_gaze_inverse.py").read_bytes()).hexdigest(),
            "A_tolerance_D": 1e-5, "SSE_tolerance": "1e-12mm²+1e-8*abs(referenceSSE)",
            "scope": "EVERY finite saved all-start A-profile query from512pilot andflagged21, including alliterations",
            "cases": {}}
        failures = []
        for name, metadata in pilot_summary["cases"].items():
            shape_path = Path(metadata["shape_path"])
            if hashlib.sha256(shape_path.read_bytes()).hexdigest() != metadata["shape_sha256"]:
                raise RuntimeError("Pinned shape changed since savedpilot")
            model = load_shape(shape_path)
            records = []
            for prefix in ("pilot_", "flagged_pilot_"):
                folder = args.pilot_dir/f"traces_{prefix}{name}"
                for path in sorted(folder.glob("*.npz")):
                    with np.load(path) as trace:
                        T = trace["all_start_T_history"]
                        profile_sse = trace["all_start_profile_sse_history_mm2"]
                        valid = np.all(np.isfinite(T), axis=3)
                        frame = np.broadcast_to(trace["source_indices"][:, None, None], T.shape[:3])
                        records.append(np.column_stack((frame[valid], T[..., 1:][valid], T[..., 0][valid], profile_sse[valid])))
            if not records:
                raise RuntimeError(f"No complete all-start query histories for{name}")
            records = np.concatenate(records)
            unique, inverse = np.unique(records[:, :3], axis=0, return_inverse=True)
            frame_indices, gaze = unique[:, 0].astype(int), unique[:, 1:3]
            local = lookup[frame_indices]
            if np.any(local < 0):
                raise RuntimeError("Saved query outside pinned source subset")
            observed = y4[local]
            start = time.perf_counter()
            reference, _ = reference401(model, observed, gaze, mean_centered=True)
            reference_time = time.perf_counter()-start
            start = time.perf_counter()
            fast, fast_timing = estimate_accommodation_fast(model, observed, gaze)
            fast_time = time.perf_counter()-start
            ref_sse = 9.*(reference["fit_vertex_rms_um"]/1000.)**2
            fast_sse = 9.*(fast["fit_vertex_rms_um"]/1000.)**2
            dA, dSSE = np.abs(fast["estimated_A_d"]-reference["estimated_A_d"]), np.abs(fast_sse-ref_sse)
            bad = (dA > 1e-5) | (dSSE > 1e-12+1e-8*np.abs(ref_sse)) | (fast["valid_map"] != reference["valid_map"])
            if np.any(bad):
                failures.append(name)
            j = len(unique)//2
            latency = {}
            for method_name, method in (("reference401", reference401), ("fast41", estimate_accommodation_fast)):
                method(model, observed[j:j+1], gaze[j:j+1], mean_centered=True)
                stamps = []
                for _ in range(args.single_repeat):
                    start = time.perf_counter()
                    method(model, observed[j:j+1], gaze[j:j+1], mean_centered=True)
                    stamps.append(time.perf_counter()-start)
                latency[method_name] = {"median_seconds": float(np.median(stamps)), "P95_seconds": float(np.percentile(stamps, 95))}
            np.savez_compressed(out/f"profile_replay_{name}.npz", source_indices=frame_indices, supplied_gaze=gaze,
                saved_query_records=records, record_to_unique_query=inverse,
                reference_A_d=reference["estimated_A_d"], fast_A_d=fast["estimated_A_d"],
                reference_sse_mm2=ref_sse, fast_sse_mm2=fast_sse,
                fallback=fast["reference401_fallback"], fallback_reason_bits=fast["fallback_reason_bits"], disagreement=bad)
            summary["cases"][name] = {"saved_queries": len(records), "unique_queries": len(unique),
                "material_disagreements": int(np.sum(bad)), "max_delta_A_D": float(np.max(dA)),
                "max_delta_SSE_mm2": float(np.max(dSSE)),
                "saved_vs_recomputed401_max_A_D": float(np.max(np.abs(records[:, 3]-reference["estimated_A_d"][inverse]))),
                "saved_vs_recomputed401_max_SSE_mm2": float(np.max(np.abs(records[:, 4]-ref_sse[inverse]))),
                "reference_wall_seconds": reference_time, "fast_wall_seconds": fast_time,
                "speedup": reference_time/fast_time, "fast_status": fast_timing, "single_query_latency": latency}
            write_json(out/"summary.json", summary)
            log(f"{name}: every{len(records)}query PASS={not np.any(bad)}; maxdA{np.max(dA):.17g}; maxdSSE{np.max(dSSE):.17g}; speedup{reference_time/fast_time:.2f}")
        summary["all_pass"] = not failures
        write_json(out/"summary.json", summary)
        if failures:
            raise RuntimeError(f"Fastprofile must be tightened beforeuse; failedcases:{failures}")


if __name__ == "__main__":
    main()
