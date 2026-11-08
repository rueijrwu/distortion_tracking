"""Observation-only accommodation inversion with externally supplied gaze.

The model must provide evaluate(A,ADE,BDE) and optionally prepare_gaze(ADE,BDE).
The search never accepts true accommodation values. Gaze is a measured input
or an explicitly labelled controlled exact-gaze benchmark.
"""
from __future__ import annotations

import time

import numpy as np

OUTER = np.array([0, 1, 2, 3, 5, 6, 7, 8])
SCAN_A = np.linspace(0., 4., 401)


class _GenericPrepared:
    def __init__(self, model, ade, bde):
        self.model = model
        self.ade, self.bde = np.asarray(ade), np.asarray(bde)

    def predict(self, accommodation, indices=None):
        if indices is None:
            indices = np.arange(len(self.ade))
        return self.model.evaluate(accommodation, self.ade[indices], self.bde[indices])["prediction"]


def _prepare(model, ade, bde):
    return model.prepare_gaze(ade, bde) if hasattr(model, "prepare_gaze") else _GenericPrepared(model, ade, bde)


def _vectors(prediction, fields, mean_centered):
    if mean_centered:
        prediction = prediction-np.mean(prediction, axis=1, keepdims=True)
    return prediction[:, fields].reshape(len(prediction), -1)


def _cost(prediction, observed, fields=OUTER, mean_centered=False):
    return np.sum((_vectors(prediction, fields, mean_centered)-observed)**2, axis=1)


def estimate_accommodation(model, observations, supplied_gaze, *, profile_chunk=32,
                           refine_chunk=8192, refinement_iterations=60, log=None,
                           mean_centered=False):
    """Global 0.01D profile scan and vectorized local golden refinement.

    Parameters contain measured centered coordinates and supplied(ADE,BDE),
    never true A. Identical supplied gaze pairs share their 401 model profiles.
    All candidate As are continuous fixed search coordinates independent of
    any ground-truth accommodation grid.
    """
    observations = np.asarray(observations, float)
    supplied_gaze = np.asarray(supplied_gaze, float)
    if observations.shape != (len(supplied_gaze), 9, 2) or supplied_gaze.shape[1:] != (2,):
        raise ValueError("Expected observations(N,9,2) and supplied gaze(N,2) in ADE,BDE order")
    if np.any(~np.isfinite(observations)) or np.any(~np.isfinite(supplied_gaze)):
        raise ValueError("Nonfinite observation or supplied gaze")
    if np.any(np.abs(supplied_gaze) > 20. + 1e-10):
        raise ValueError("Supplied gaze is outside the calibrated ±20-degree domain; no clipping/extrapolation")
    n = len(observations)
    fields = np.arange(9) if mean_centered else OUTER
    width = len(fields)*2
    observed = _vectors(observations, fields, mean_centered)
    cost_function = lambda prediction, target: _cost(prediction, target, fields, mean_centered)
    unique_gaze, group = np.unique(supplied_gaze, axis=0, return_inverse=True)
    ordering = np.argsort(group, kind="stable")
    edges = np.r_[0, np.cumsum(np.bincount(group, minlength=len(unique_gaze)))]
    coarse_index = np.empty(n, np.int16)
    coarse_sse = np.empty(n)
    minima = np.empty(n, np.int16)
    started = time.perf_counter()
    for start in range(0, len(unique_gaze), profile_chunk):
        stop = min(start + profile_chunk, len(unique_gaze))
        gaze = unique_gaze[start:stop]
        prepared = _prepare(model, gaze[:, 0], gaze[:, 1])
        indices = np.repeat(np.arange(len(gaze)), len(SCAN_A))
        prediction = prepared.predict(np.tile(SCAN_A, len(gaze)), indices=indices)
        profiles = _vectors(prediction, fields, mean_centered).reshape(len(gaze), len(SCAN_A), width)
        for local, gi in enumerate(range(start, stop)):
            ids = ordering[edges[gi]:edges[gi + 1]]
            # Shape-only SSE on the same eight physical vertices as the fit.
            costs = np.sum((profiles[local][None, :, :] - observed[ids, None, :])**2, axis=2)
            best = np.argmin(costs, axis=1)
            coarse_index[ids] = best
            coarse_sse[ids] = costs[np.arange(len(ids)), best]
            minima[ids] = (np.sum((costs[:, 1:-1] < costs[:, :-2]) &
                                  (costs[:, 1:-1] <= costs[:, 2:]), axis=1) +
                           (costs[:, 0] <= costs[:, 1]) + (costs[:, -1] < costs[:, -2]))
        if log and (stop == len(unique_gaze) or stop % 1024 == 0):
            log(f"known-gaze scan profiles {stop:,}/{len(unique_gaze):,}; {time.perf_counter()-started:.1f}s")
    scan_seconds = time.perf_counter() - started
    result = {"estimated_A_d": np.empty(n), "fit_vertex_rms_um": np.empty(n),
              "converged": np.zeros(n, bool), "valid_map": np.zeros(n, bool),
              "coarse_A_d": SCAN_A[coarse_index], "coarse_sse_mm2": coarse_sse,
              "profile_local_minima": minima, "final_bracket_width_d": np.empty(n),
              "derivative_coordinate_norm_mm_per_D": np.empty(n)}
    ratio = (np.sqrt(5.) - 1.) / 2.
    refine_started = time.perf_counter()
    for start in range(0, n, refine_chunk):
        stop = min(start + refine_chunk, n)
        take = slice(start, stop)
        gaze = supplied_gaze[take]
        target = observed[take]
        prepared = _prepare(model, gaze[:, 0], gaze[:, 1])
        lo = SCAN_A[np.maximum(coarse_index[take] - 1, 0)].copy()
        hi = SCAN_A[np.minimum(coarse_index[take] + 1, len(SCAN_A) - 1)].copy()
        c, d = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
        fc, fd = cost_function(prepared.predict(c), target), cost_function(prepared.predict(d), target)
        for _ in range(refinement_iterations):
            left = fc <= fd
            next_hi, next_lo = np.where(left, d, hi), np.where(left, lo, c)
            next_c = np.where(left, next_hi - ratio * (next_hi - next_lo), d)
            next_d = np.where(left, c, next_lo + ratio * (next_hi - next_lo))
            next_value = np.where(left, next_c, next_d)
            next_cost = cost_function(prepared.predict(next_value), target)
            fc, fd = np.where(left, next_cost, fd), np.where(left, fc, next_cost)
            lo, hi, c, d = next_lo, next_hi, next_c, next_d
        estimate = (lo + hi) / 2.
        cost = cost_function(prepared.predict(estimate), target)
        # Retain exact constrained endpoints when the observed objective favors
        # a coarse endpoint. No rounding to calibration/source A labels occurs.
        coarse = result["coarse_A_d"][take]
        prefer_coarse = coarse_sse[take] <= cost
        estimate = np.where(prefer_coarse, coarse, estimate)
        cost = np.where(prefer_coarse, coarse_sse[take], cost)
        final = model.evaluate(estimate, gaze[:, 0], gaze[:, 1], state_jac=True, diagnostics=True)
        result["estimated_A_d"][take] = estimate
        result["fit_vertex_rms_um"][take] = np.sqrt(cost / len(fields)) * 1000.
        result["final_bracket_width_d"][take] = hi - lo
        result["converged"][take] = np.isfinite(cost) & (hi - lo < 1e-9)
        result["valid_map"][take] = np.all(np.stack([final[name] > 0 for name in
            ("denominator", "radial_tangential", "radial_radial", "brown_determinant")]), axis=(0, 2))
        derivative = final["state_jacobian"][..., 0]
        if mean_centered:
            derivative = derivative-np.mean(derivative, axis=1, keepdims=True)
        result["derivative_coordinate_norm_mm_per_D"][take] = np.linalg.norm(derivative[:, fields].reshape(-1, width), axis=1)
        if log:
            log(f"known-gaze refine {stop:,}/{n:,}; {time.perf_counter()-refine_started:.1f}s")
    result["boundary_A"] = (result["estimated_A_d"] <= 1e-7) | (result["estimated_A_d"] >= 4. - 1e-7)
    return result, {"scan_seconds": scan_seconds, "refine_seconds": time.perf_counter()-refine_started,
                    "unique_supplied_gaze_pairs": len(unique_gaze), "attempted_states": n,
                    "scan_candidates": len(SCAN_A), "golden_iterations": refinement_iterations,
                    "shape_objective": "all9 mean-centered; center profiled out" if mean_centered else "outer8 center-relative"}
