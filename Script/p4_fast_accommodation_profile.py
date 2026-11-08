"""Independent batched41-point A profile with audited401-point fallback."""
from __future__ import annotations

import time

import numpy as np

from p4_known_gaze_inverse import estimate_accommodation as reference401

SCAN_A = np.linspace(0., 4., 41)
GOLDEN_ITERATIONS = 32


def vectors(prediction):
    prediction = prediction-np.mean(prediction, axis=1, keepdims=True)
    return prediction.reshape(len(prediction), 18)


def profile_minima(cost):
    interior = np.sum((cost[:, 1:-1] < cost[:, :-2]) & (cost[:, 1:-1] <= cost[:, 2:]), axis=1)
    total = interior+(cost[:, 0] <= cost[:, 1])+(cost[:, -1] < cost[:, -2])
    return interior, total


def estimate_accommodation_fast(model, observations, supplied_gaze, *, chunk=256,
                                 mean_centered=True, refinement_iterations=32, log=None):
    """Same observed9-field mean-centered SSE; no true accommodation input.

    Candidate A is evaluated in batched per-observation arrays, avoiding the
    reference implementation's Python loop over almost-unique feedback gazes.
    Multiple/irregular coarse minima and invalid refinement use reference401.
    """
    if not mean_centered or refinement_iterations != GOLDEN_ITERATIONS:
        raise ValueError("This audited method is fixed to all9 mean-centered fields and32 golden iterations")
    y, gaze = np.asarray(observations, float), np.asarray(supplied_gaze, float)
    if y.shape != (len(gaze), 9, 2) or gaze.shape[1:] != (2,) or not np.all(np.isfinite(y)) or not np.all(np.isfinite(gaze)):
        raise ValueError("Expected finite measured9-point shapes and supplied(ADE,BDE)")
    if np.any(np.abs(gaze) > 12.5+1e-9):
        raise ValueError("Supplied gaze outside the explicit±12.5deg domain")
    n = len(y)
    result = {"estimated_A_d": np.empty(n), "fit_vertex_rms_um": np.empty(n),
        "converged": np.zeros(n, bool), "valid_map": np.zeros(n, bool),
        "profile_local_minima": np.zeros(n, np.int16), "coarse_profile_interior_minima": np.zeros(n, np.int16),
        "coarse_profile_flat": np.zeros(n, bool),
        "final_bracket_width_d": np.empty(n), "coarse_A_d": np.empty(n), "coarse_sse_mm2": np.empty(n),
        "derivative_coordinate_norm_mm_per_D": np.empty(n),
        "reference401_fallback": np.zeros(n, bool), "fallback_reason_bits": np.zeros(n, np.int16)}
    started = time.perf_counter()
    ratio = (np.sqrt(5.)-1.)/2.
    for lower in range(0, n, chunk):
        upper = min(lower+chunk, n)
        target = vectors(y[lower:upper])
        g = gaze[lower:upper]
        prepared = model.prepare_gaze(g[:, 0], g[:, 1])
        size = len(g)
        indices = np.repeat(np.arange(size), len(SCAN_A))
        profiles = vectors(prepared.predict(np.tile(SCAN_A, size), indices=indices)).reshape(size, len(SCAN_A), 18)
        cost = np.sum((profiles-target[:, None])**2, axis=2)
        finite = np.all(np.isfinite(cost), axis=1)
        safe = np.where(np.isfinite(cost), cost, np.inf)
        best = np.argmin(safe, axis=1)
        best_cost = safe[np.arange(size), best]
        interior, minima = profile_minima(safe)
        reason = np.zeros(size, np.int16)
        reason[~finite] |= 1
        reason[interior > 1] |= 2
        finite_cost = np.where(np.isfinite(cost), cost, 0.)
        flat = finite & (np.ptp(finite_cost, axis=1) <= 1e-14+1e-10*np.max(np.abs(finite_cost), axis=1))
        # An endpoint plus another minimum, or a fully flat profile, is also
        # conservatively classified as irregular/multiple and uses401.
        reason[(minima > 1) | flat] |= 8
        lo, hi = SCAN_A[np.maximum(best-1, 0)].copy(), SCAN_A[np.minimum(best+1, len(SCAN_A)-1)].copy()
        c, d = hi-ratio*(hi-lo), lo+ratio*(hi-lo)
        def objective(A):
            return np.sum((vectors(prepared.predict(A))-target)**2, axis=1)
        fc, fd = objective(c), objective(d)
        for _ in range(GOLDEN_ITERATIONS):
            left = fc <= fd
            new_hi, new_lo = np.where(left, d, hi), np.where(left, lo, c)
            new_c = np.where(left, new_hi-ratio*(new_hi-new_lo), d)
            new_d = np.where(left, c, new_lo+ratio*(new_hi-new_lo))
            value = objective(np.where(left, new_c, new_d))
            fc, fd = np.where(left, value, fd), np.where(left, fc, value)
            lo, hi, c, d = new_lo, new_hi, new_c, new_d
        estimate = (lo+hi)/2.
        refined_cost = objective(estimate)
        use_coarse = best_cost <= refined_cost
        estimate = np.where(use_coarse, SCAN_A[best], estimate)
        selected_cost = np.where(use_coarse, best_cost, refined_cost)
        # Explicit exact endpoints, with deterministic lower-end tie behavior.
        use_zero = safe[:, 0] <= selected_cost
        estimate, selected_cost = np.where(use_zero, 0., estimate), np.where(use_zero, safe[:, 0], selected_cost)
        use_four = safe[:, -1] < selected_cost
        estimate, selected_cost = np.where(use_four, 4., estimate), np.where(use_four, safe[:, -1], selected_cost)
        final = model.evaluate(estimate, g[:, 0], g[:, 1], state_jac=True, diagnostics=True)
        valid = np.all(np.stack([final[key] > 0 for key in
            ("denominator", "radial_tangential", "radial_radial", "brown_determinant")]), axis=(0, 2))
        success = valid & np.isfinite(selected_cost) & np.isfinite(estimate) & (hi-lo < 1e-6)
        reason[~success] |= 4
        derivative = final["state_jacobian"][..., 0]
        derivative -= np.mean(derivative, axis=1, keepdims=True)
        sl = slice(lower, upper)
        result["estimated_A_d"][sl], result["fit_vertex_rms_um"][sl] = estimate, np.sqrt(selected_cost/9.)*1000.
        result["converged"][sl], result["valid_map"][sl] = success, valid
        result["profile_local_minima"][sl], result["coarse_profile_interior_minima"][sl] = minima, interior
        result["coarse_profile_flat"][sl] = flat
        result["final_bracket_width_d"][sl] = hi-lo
        result["coarse_A_d"][sl], result["coarse_sse_mm2"][sl] = SCAN_A[best], best_cost
        result["derivative_coordinate_norm_mm_per_D"][sl] = np.linalg.norm(derivative.reshape(size, 18), axis=1)
        result["fallback_reason_bits"][sl] = reason
        result["reference401_fallback"][sl] = reason != 0
        if log:
            log(f"fastA profile{upper}/{n}")
    fallback = result["reference401_fallback"]
    if np.any(fallback):
        reference, _ = reference401(model, y[fallback], gaze[fallback], mean_centered=True)
        for key in ("estimated_A_d", "fit_vertex_rms_um", "converged", "valid_map", "profile_local_minima",
                    "final_bracket_width_d", "coarse_A_d", "coarse_sse_mm2", "derivative_coordinate_norm_mm_per_D"):
            result[key][fallback] = reference[key]
    timing = {"wall_seconds": time.perf_counter()-started, "attempted_states": n, "scan_candidates": 41,
        "golden_iterations": 32, "reference_fallback_states": int(np.sum(fallback)),
        "fallback_reason_bits": "1=nonfinite scan;2=multiple interior minima;4=invalid refinement;8=flat/multiple/irregular including endpoints",
        "objective": "same all9 mean-centered coordinate SSE; A0..4; exactendpoint comparison"}
    return result, timing
