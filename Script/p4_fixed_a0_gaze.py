"""Gaze-only, observed-data fixed point with accommodation fixed identically to 0 D.

The two supplied callables evaluate the frozen forward and center-cubic maps at
A=0. This module has no accommodation optimizer and accepts no true state/center.
"""
from __future__ import annotations

import numpy as np


def solve_fixed_a0(observed_p4_mm, mean_p1_mm, forward_at_zero, cubic_at_zero,
                   *, gaze_bound_deg=12.5, max_iterations=40,
                   gaze_tolerance_deg=1e-4, keep_history=True):
    y4 = np.asarray(observed_p4_mm, dtype=float)
    m1 = np.asarray(mean_p1_mm, dtype=float)
    if y4.ndim != 3 or y4.shape[1:] != (9, 2) or m1.shape != (len(y4), 2):
        raise ValueError("Expected observed P4[n,9,2] and measured P1 mean[n,2]")
    n, starts = len(y4), 3
    m4 = y4.mean(axis=1)
    d_raw = m4 - m1
    raw_initial = np.asarray(cubic_at_zero(d_raw), dtype=float)
    if raw_initial.shape != (n, 2):
        raise ValueError("Center cubic must return [ADE,BDE] pairs")
    finite_observation = np.isfinite(y4).all(axis=(1, 2)) & np.isfinite(m1).all(axis=1)
    initial_finite = np.isfinite(raw_initial).all(axis=1)
    initial_bounded = np.clip(np.nan_to_num(raw_initial), -gaze_bound_deg, gaze_bound_deg)
    # The raw-mean path is primary; the other two seeds are blind diagnostics.
    gaze = np.stack((initial_bounded, np.zeros_like(initial_bounded), -initial_bounded), axis=0)
    alive = np.broadcast_to(finite_observation, (starts, n)).copy()
    alive[[0, 2]] &= initial_finite
    converged = np.zeros((starts, n), dtype=bool)
    invalid = ~alive.copy()
    consecutive = np.zeros((starts, n), dtype=np.int16)
    iterations = np.zeros((starts, n), dtype=np.int16)
    cycle_count = np.zeros((starts, n), dtype=np.int16)
    growth_count = np.zeros((starts, n), dtype=np.int16)
    alpha = np.ones((starts, n))
    previous_norm = np.full((starts, n), np.inf)
    previous_state = np.full_like(gaze, np.nan)
    best_norm = np.full((starts, n), np.inf)
    best_sse = np.full((starts, n), np.inf)
    best_gaze = gaze.copy()
    first_raw = np.full_like(gaze, np.nan)
    first_bounded = np.full_like(gaze, np.nan)
    ever_pressure = np.zeros((starts, n, 2), dtype=bool)
    histories = []
    observed_shape = y4 - m4[:, None, :]

    def evaluate(points):
        pred, valid = forward_at_zero(points.reshape(-1, 2))
        pred = np.asarray(pred, dtype=float).reshape(starts, n, 9, 2)
        valid = np.asarray(valid, dtype=bool).reshape(starts, n)
        valid &= np.isfinite(pred).all(axis=(2, 3))
        mu = pred.mean(axis=2)
        center = m4[None, :, :] - mu
        raw = np.asarray(cubic_at_zero((center-m1[None, :, :]).reshape(-1, 2)),
                         dtype=float).reshape(starts, n, 2)
        valid &= np.isfinite(raw).all(axis=2)
        proposal = np.clip(raw, -gaze_bound_deg, gaze_bound_deg)
        delta = proposal - points
        norm_deg = np.linalg.norm(delta, axis=2)
        normalized = norm_deg / 20.0
        residual = observed_shape[None, :, :, :] - (pred-mu[:, :, None, :])
        sse = np.sum(residual*residual, axis=(2, 3))
        valid &= np.isfinite(sse) & np.isfinite(normalized)
        return pred, valid, center, raw, proposal, delta, norm_deg, normalized, sse

    for it in range(max_iterations):
        if not np.any(alive):
            break
        pred, valid, center, raw, proposal, delta, norm_deg, normalized, sse = evaluate(gaze)
        active = alive.copy()
        valid_active = active & valid
        invalid |= active & ~valid
        alive &= valid
        iterations[active] = it+1
        if it == 0:
            first_raw[:] = raw
            first_bounded[:] = proposal
        ever_pressure |= active[:, :, None] & (np.abs(raw) > gaze_bound_deg)
        improved = valid_active & ((normalized < best_norm)
                    | ((normalized == best_norm) & (sse < best_sse)))
        best_norm[improved] = normalized[improved]
        best_sse[improved] = sse[improved]
        best_gaze[improved] = gaze[improved]
        consecutive = np.where(valid_active & (norm_deg <= gaze_tolerance_deg),
                               consecutive+1, 0)
        just_converged = alive & (consecutive >= 2)
        converged |= just_converged
        best_gaze[just_converged] = gaze[just_converged]
        best_norm[just_converged] = normalized[just_converged]
        best_sse[just_converged] = sse[just_converged]
        growth = alive & np.isfinite(previous_norm) & (normalized > 1.2*previous_norm)
        cycle = (alive & np.isfinite(previous_state).all(axis=2)
                 & (np.linalg.norm(proposal-previous_state, axis=2) <= gaze_tolerance_deg)
                 & (norm_deg > gaze_tolerance_deg))
        growth_count += growth
        cycle_count += cycle
        alpha = np.where(growth | cycle, np.maximum(alpha/2.0, 1.0/32.0), alpha)
        if keep_history:
            histories.append({
                "gaze_deg": gaze.copy(), "raw_cubic_deg": raw.copy(),
                "bounded_proposal_deg": proposal.copy(), "center_mm": center.copy(),
                "alpha": alpha.copy(), "fixedpoint_residual_deg": norm_deg.copy(),
                "normalized_fixedpoint_residual": normalized.copy(), "shape_sse_mm2": sse.copy(),
                "map_valid": valid.copy(), "active": active.copy(),
                "growth": growth.copy(), "two_cycle": cycle.copy(),
            })
        alive &= ~just_converged
        previous_state[:] = gaze
        previous_norm = np.where(valid_active, normalized, previous_norm)
        updated = gaze + alpha[:, :, None]*delta
        gaze = np.where(alive[:, :, None], updated, gaze)

    # Re-evaluate the returned best valid state for every seed. Small damped
    # steps never change the convergence flag or the undamped stopping rule.
    pred, valid, center, raw, proposal, delta, norm_deg, normalized, sse = evaluate(best_gaze)
    finite = valid & np.isfinite(best_norm)
    converged &= finite & (norm_deg <= gaze_tolerance_deg)
    costs = np.where(converged, sse, np.inf)
    has_converged = converged.any(axis=0)
    selected = np.argmin(costs, axis=0)
    # If no seed converges, choose the smallest undamped fixed-point residual,
    # with observed shape SSE breaking exact ties; never use scoring labels.
    fallback_norm = np.where(finite, normalized, np.inf)
    for row in np.flatnonzero(~has_converged):
        candidates = np.flatnonzero(finite[:, row])
        if len(candidates):
            selected[row] = candidates[np.lexsort((sse[candidates, row],
                                                   fallback_norm[candidates, row]))[0]]
    rows = np.arange(n)
    chosen_gaze = best_gaze[selected, rows]
    chosen_finite = finite[selected, rows]
    chosen_gaze = np.where(chosen_finite[:, None], chosen_gaze, np.nan)
    bound_flags = np.isclose(np.abs(chosen_gaze), gaze_bound_deg, atol=1e-7, rtol=0)
    distinct = np.zeros(n, dtype=bool)
    for left in range(starts):
        for right in range(left+1, starts):
            distinct |= (converged[left] & converged[right]
                         & (np.linalg.norm(best_gaze[left]-best_gaze[right], axis=1) > 1e-3))
    result = {
        "gaze_deg": chosen_gaze, "center_mm": center[selected, rows],
        "converged": has_converged, "valid": chosen_finite,
        "selected_start": selected, "shape_sse_mm2": sse[selected, rows],
        "fixedpoint_residual_deg": norm_deg[selected, rows],
        "raw_initial_gaze_deg": raw_initial, "bounded_initial_gaze_deg": initial_bounded,
        "first_corrected_raw_gaze_deg": first_raw[0],
        "first_corrected_gaze_deg": first_bounded[0],
        "primary_gaze_deg": best_gaze[0], "primary_converged": converged[0],
        "primary_center_mm": center[0], "primary_shape_sse_mm2": sse[0],
        "alternate_start_rescue": ~converged[0] & has_converged,
        "distinct_converged_roots": distinct,
        "bound_ADE_BDE": bound_flags,
        "outside_sampled_12p4": np.any(np.abs(chosen_gaze) > 12.4+1e-10, axis=1),
        "all_gaze_deg": best_gaze, "all_center_mm": center,
        "all_converged": converged, "all_valid": finite, "all_invalid": invalid,
        "all_iterations": iterations, "all_alpha": alpha,
        "all_cycle_count": cycle_count, "all_growth_count": growth_count,
        "all_shape_sse_mm2": sse, "all_fixedpoint_residual_deg": norm_deg,
        "all_raw_cubic_deg": raw, "all_bounded_proposal_deg": proposal,
        "all_ever_cubic_bound_pressure": ever_pressure,
        "all_initial_gaze_deg": np.stack((initial_bounded, np.zeros_like(initial_bounded),
                                          -initial_bounded), axis=0),
        "accommodation_d": np.zeros(n),
        "accommodation_optimizer_calls": np.array(0),
    }
    if keep_history:
        for key in histories[0] if histories else ():
            result["history_"+key] = np.stack([entry[key] for entry in histories], axis=0)
    return result


def synthetic_check():
    """Blind coupled recovery, including asymmetric points and exact bounds."""
    truth = np.array([[3.7, -8.2], [-11.6, 2.9], [12.5, -12.5],
                      [-12.5, 12.5], [0.0, 0.0], [12.5, 4.1]])
    fields = np.array([(x, y) for x in (-1., 0., 1.) for y in (-1., 0., 1.)])
    weight = np.sum(fields*fields, axis=1)
    calls = {"forward": 0, "cubic": 0}

    def cubic(q):
        calls["cubic"] += 1
        q = np.asarray(q)
        # Nontrivial separate cubics, returned in [ADE,BDE] order.
        return np.stack((5.0*(q[:, 1]+.01*q[:, 1]**3)+.1,
                         -6.0*(q[:, 0]+.02*q[:, 0]**3)), axis=1)

    def forward(g):
        calls["forward"] += 1
        g = np.asarray(g)
        pred = np.broadcast_to(.8*fields, (len(g), 9, 2)).copy()
        pred[..., 0] += .004*g[:, 1, None]*weight
        pred[..., 1] -= .003*g[:, 0, None]*weight
        return pred, np.ones(len(g), dtype=bool)

    q = np.stack((-truth[:, 1]/6.0, (truth[:, 0]-.1)/5.0), axis=1)
    for _ in range(12):
        q[:, 0] -= ((q[:, 0]+.02*q[:, 0]**3)+truth[:, 1]/6.0)/(1+.06*q[:, 0]**2)
        q[:, 1] -= ((q[:, 1]+.01*q[:, 1]**3)-(truth[:, 0]-.1)/5.0)/(1+.03*q[:, 1]**2)
    m1 = np.array([1.7, -.4]) + np.arange(len(truth))[:, None]*[.01, -.02]
    shape, _ = forward(truth)
    y4 = m1[:, None, :]+q[:, None, :]+shape
    result = solve_fixed_a0(y4, m1, forward, cubic)
    error = np.max(np.abs(result["gaze_deg"]-truth))
    center_error = np.max(np.abs(result["center_mm"]-(m1+q)))
    identity = np.max(np.abs(y4.mean(axis=1)-shape.mean(axis=1)-(m1+q)))
    if not result["converged"].all() or error > 1e-4 or center_error > 1e-5 or identity > 1e-12:
        raise AssertionError((error, center_error, identity, result["converged"]))
    if not np.all(result["accommodation_d"] == 0) or int(result["accommodation_optimizer_calls"]) != 0:
        raise AssertionError("Accommodation was not fixed identically to zero")
    return {"all_converged": True, "max_gaze_component_error_deg": float(error),
            "max_center_component_error_mm": float(center_error),
            "mean_center_identity_error_mm": float(identity), "map_calls": calls,
            "boundary_states": 3, "profile_calls": 0}


if __name__ == "__main__":
    import json
    print(json.dumps(synthetic_check(), indent=2))
