"""Center-first C8/shape fixed point using measured P4/P1 observations only."""
from __future__ import annotations

import numpy as np

from p4_known_gaze_inverse import estimate_accommodation
from p4_fast_accommodation_profile import estimate_accommodation_fast

START_A = np.array([0., 2., 4.])
SCALE = np.array([4., 20., 20.])
A_TOL = 1e-5
G_TOL = 1e-4
ALPHA_FLOOR = 1./32.


def boundary_flags(state, gaze_bound_deg=12.5):
    """Independent per-coordinate flags, including mixed-sign edges."""
    flags = np.zeros(np.asarray(state).shape, bool)
    flags[:, 0] = np.isclose(state[:, 0], 0., atol=1e-7, rtol=0) | np.isclose(state[:, 0], 4., atol=1e-7, rtol=0)
    flags[:, 1] = np.isclose(np.abs(state[:, 1]), gaze_bound_deg, atol=1e-7, rtol=0)
    flags[:, 2] = np.isclose(np.abs(state[:, 2]), gaze_bound_deg, atol=1e-7, rtol=0)
    return flags


def shape_state(model, measured_p4, state):
    prediction = model.evaluate(*state.T, diagnostics=True)
    F = prediction["prediction"]
    meanF = np.mean(F, axis=1)
    mean4 = np.mean(measured_p4, axis=1)
    residual = (measured_p4-mean4[:, None])-(F-meanF[:, None])
    valid = np.all(np.isfinite(F), axis=(1, 2)) & np.all(np.stack([prediction[key] > 0 for key in
        ("denominator", "radial_tangential", "radial_radial", "brown_determinant")]), axis=(0, 2))
    return {"meanF": meanF, "center": mean4-meanF,
            "shape_sse_mm2": np.sum(residual**2, axis=(1, 2)), "valid": valid}


def evaluate_T(model, center_cubic, measured_p4, measured_p1_mean, state, gaze_bound_deg=12.5,
               profile_solver="reference401"):
    """Final approved order: current center -> gaze update -> A profile."""
    current = shape_state(model, measured_p4, state)
    q = current["center"]-measured_p1_mean
    # One shared vector-valued cubic; accommodation is not a gaze-map input.
    raw_gaze = center_cubic.predict(q)
    bounded_gaze = np.clip(raw_gaze, -gaze_bound_deg, gaze_bound_deg)
    eligible = current["valid"] & np.all(np.isfinite(raw_gaze), axis=1)
    T = np.column_stack((np.full(len(state), np.nan), bounded_gaze))
    success, profile_valid = np.zeros(len(state), bool), np.zeros(len(state), bool)
    profile_sse = np.full(len(state), np.nan)
    minima = np.zeros(len(state), np.int16)
    fallback, fallback_reason = np.zeros(len(state), bool), np.zeros(len(state), np.int16)
    if np.any(eligible):
        if profile_solver == "reference401":
            inverse, _ = estimate_accommodation(model, measured_p4[eligible], bounded_gaze[eligible],
                                                mean_centered=True, refinement_iterations=60)
        elif profile_solver == "fast41":
            inverse, _ = estimate_accommodation_fast(model, measured_p4[eligible], bounded_gaze[eligible],
                                                     mean_centered=True, refinement_iterations=32)
            fallback[eligible] = inverse["reference401_fallback"]
            fallback_reason[eligible] = inverse["fallback_reason_bits"]
        else:
            raise ValueError("Explicit profile solver must be reference401 orfast41")
        T[eligible, 0] = inverse["estimated_A_d"]
        success[eligible] = inverse["converged"]
        profile_valid[eligible] = inverse["valid_map"]
        profile_sse[eligible] = 9.*(inverse["fit_vertex_rms_um"]/1000.)**2
        minima[eligible] = inverse["profile_local_minima"]
    delta = T-state
    norm = np.linalg.norm(delta/SCALE, axis=1)
    return dict(current, T=T, raw_gaze=raw_gaze, bounded_gaze=bounded_gaze, q=q,
        delta=delta, residual_norm=norm, profile_success=success, profile_valid=profile_valid,
        profile_sse_mm2=profile_sse, profile_local_minima=minima,
        profile_attempted=eligible, profile_fallback=fallback, profile_fallback_reason=fallback_reason,
        update_valid=eligible & success & profile_valid & np.all(np.isfinite(T), axis=1))


def solve_fixedpoint(model, center_cubic, measured_p4, measured_p1_mean, *, max_iterations=40,
                     gaze_bound_deg=12.5, record_history=True, profile_solver="reference401"):
    """No true A/gaze/center arguments; no incompatible D-based SSE gate."""
    y4, m1 = np.asarray(measured_p4), np.asarray(measured_p1_mean)
    if y4.shape != (len(m1), 9, 2) or m1.shape[1:] != (2,) or not np.all(np.isfinite(y4)) or not np.all(np.isfinite(m1)):
        raise ValueError("Runtime requires all nine finite measured P4 rays and measured P1 mean")
    n, k = len(m1), 3
    if profile_solver not in ("reference401", "fast41"):
        raise ValueError("Explicit profile solver must be reference401 orfast41")
    initial_A = np.tile(START_A, n)
    raw_initial_all = center_cubic.predict(np.repeat(np.mean(y4, axis=1)-m1, k, axis=0)).reshape(n, k, 2)
    bounded_initial_all = np.clip(raw_initial_all, -gaze_bound_deg, gaze_bound_deg)
    raw_initial, initial_gaze = raw_initial_all[:, 1], bounded_initial_all[:, 1]
    initial = np.column_stack((initial_A, bounded_initial_all.reshape(n*k, 2)))
    state = initial.copy()
    targets4, targets1 = np.repeat(y4, k, axis=0), np.repeat(m1, k, axis=0)
    count = len(state)
    active = np.all(np.isfinite(state), axis=1)
    converged = np.zeros(count, bool)
    invalid_update = np.zeros(count, bool)
    alpha = np.ones(count)
    consecutive = np.zeros(count, np.int16)
    iterations, growth_events, cycle_events = (np.zeros(count, np.int16) for _ in range(3))
    profile_calls, profile_fallbacks = np.zeros(count, np.int16), np.zeros(count, np.int16)
    previous_norm = np.full(count, np.inf)
    previous_state = np.full_like(state, np.nan)
    best_norm, best_sse = np.full(count, np.inf), np.full(count, np.inf)
    best_state, best_center = state.copy(), np.full((count, 2), np.nan)
    best_raw_gaze = np.full((count, 2), np.nan)
    best_profile_minima = np.zeros(count, np.int16)
    first_raw, first_bounded, first_center = (np.full((count, 2), np.nan) for _ in range(3))
    history = {}
    for name, width in (("state_history", 3), ("T_history", 3), ("raw_cubic_gaze_history", 2),
                        ("corrected_center_history_mm", 2), ("fixedpoint_delta_history", 3)):
        history[name] = np.full((count, max_iterations, width), np.nan)
    for name in ("fixedpoint_norm_history", "alpha_history", "shape_sse_history_mm2", "profile_sse_history_mm2"):
        history[name] = np.full((count, max_iterations), np.nan)
    history["A_profile_success_history"] = np.zeros((count, max_iterations), bool)
    history["valid_update_history"] = np.zeros((count, max_iterations), bool)
    history["A_profile_fallback_history"] = np.zeros((count, max_iterations), bool)
    history["A_profile_fallback_reason_history"] = np.zeros((count, max_iterations), np.int16)
    for iteration in range(max_iterations):
        ids = np.flatnonzero(active)
        if not len(ids):
            break
        evaluated = evaluate_T(model, center_cubic, targets4[ids], targets1[ids], state[ids], gaze_bound_deg, profile_solver)
        profile_calls[ids] += evaluated["profile_attempted"]
        profile_fallbacks[ids] += evaluated["profile_fallback"]
        norm, sse = evaluated["residual_norm"], evaluated["shape_sse_mm2"]
        valid = evaluated["update_valid"] & np.isfinite(norm) & np.isfinite(sse)
        if iteration == 0:
            first_raw[ids], first_bounded[ids], first_center[ids] = evaluated["raw_gaze"], evaluated["bounded_gaze"], evaluated["center"]
        better = valid & ((norm < best_norm[ids]-1e-12) |
            (np.isclose(norm, best_norm[ids], rtol=1e-9, atol=1e-12) & (sse < best_sse[ids])))
        use = ids[better]
        best_norm[use], best_sse[use] = norm[better], sse[better]
        best_state[use], best_center[use], best_raw_gaze[use] = state[use], evaluated["center"][better], evaluated["raw_gaze"][better]
        best_profile_minima[use] = evaluated["profile_local_minima"][better]
        delta = evaluated["delta"]
        small = valid & (np.abs(delta[:, 0]) <= A_TOL) & (np.linalg.norm(delta[:, 1:], axis=1) <= G_TOL)
        consecutive[ids] = np.where(small, consecutive[ids]+1, 0)
        done = valid & (consecutive[ids] >= 2)
        if np.any(done):
            use = ids[done]
            converged[use], active[use] = True, False
            # Root itself, not an earlier lowest-residual iterate, is retained.
            best_state[use], best_center[use], best_raw_gaze[use] = state[use], evaluated["center"][done], evaluated["raw_gaze"][done]
            best_norm[use], best_sse[use] = norm[done], sse[done]
            best_profile_minima[use] = evaluated["profile_local_minima"][done]
        grown = valid & np.isfinite(previous_norm[ids]) & (norm > 1.2*previous_norm[ids])
        previous = previous_state[ids]
        cycle = valid & np.all(np.isfinite(previous), axis=1) & ~small & (
            np.abs(evaluated["T"][:, 0]-previous[:, 0]) <= A_TOL) & (
            np.linalg.norm(evaluated["T"][:, 1:]-previous[:, 1:], axis=1) <= G_TOL)
        growth_events[ids] += grown
        cycle_events[ids] += cycle
        alpha[ids[grown | cycle]] = np.maximum(alpha[ids[grown | cycle]]/2., ALPHA_FLOOR)
        history["state_history"][ids, iteration] = state[ids]
        history["T_history"][ids, iteration] = evaluated["T"]
        history["raw_cubic_gaze_history"][ids, iteration] = evaluated["raw_gaze"]
        history["corrected_center_history_mm"][ids, iteration] = evaluated["center"]
        history["fixedpoint_delta_history"][ids, iteration] = delta
        history["fixedpoint_norm_history"][ids, iteration] = norm
        history["alpha_history"][ids, iteration] = alpha[ids]
        history["shape_sse_history_mm2"][ids, iteration] = sse
        history["profile_sse_history_mm2"][ids, iteration] = evaluated["profile_sse_mm2"]
        history["A_profile_success_history"][ids, iteration] = evaluated["profile_success"]
        history["valid_update_history"][ids, iteration] = valid
        history["A_profile_fallback_history"][ids, iteration] = evaluated["profile_fallback"]
        history["A_profile_fallback_reason_history"][ids, iteration] = evaluated["profile_fallback_reason"]
        iterations[ids] = iteration+1
        invalid_update[ids[~valid]], active[ids[~valid]] = True, False
        update = valid & ~done
        update_ids = ids[update]
        previous_state[ids], previous_norm[ids] = state[ids], norm
        if iteration < max_iterations-1:
            state[update_ids] += alpha[update_ids, None]*delta[update]
    valid_candidate = np.isfinite(best_norm) & np.isfinite(best_sse)
    tail_indices = np.maximum(iterations[:, None]-np.arange(1, 9)[None, :], 0)
    tail = np.take_along_axis(history["fixedpoint_norm_history"], tail_indices, axis=1)
    finite_tail = np.all(np.isfinite(tail), axis=1)
    stagnation = (~converged) & (iterations >= 8) & finite_tail & (
        np.ptp(tail, axis=1) <= np.maximum(1e-10, 1e-6*np.min(np.where(np.isfinite(tail), tail, np.inf), axis=1)))
    conv = (converged & valid_candidate).reshape(n, k)
    costs, norms = best_sse.reshape(n, k), best_norm.reshape(n, k)
    any_conv = np.any(conv, axis=1)
    converged_choice = np.argmin(np.where(conv, costs, np.inf), axis=1)
    # For nonconverged controls, tie residuals by observed shape SSE.
    fallback = np.array([np.lexsort((costs[i], norms[i]))[0] for i in range(n)])
    # The raw-mean A2 primary is the estimator. Alternative starts may rescue
    # only a failed primary; lower shape SSE cannot replace a converged one.
    choice = np.where(conv[:, 1], 1, np.where(any_conv, converged_choice, fallback))
    selected = np.arange(n)*k+choice
    roots = best_state.reshape(n, k, 3)
    root_difference = roots-roots[np.arange(n), choice, None, :]
    distinct = conv & ((np.abs(root_difference[:, :, 0]) > 1e-3) | (np.linalg.norm(root_difference[:, :, 1:], axis=2) > 1e-3))
    pressure = np.abs(best_raw_gaze) > gaze_bound_deg+1e-9
    selected_state = best_state[selected]
    result = {"estimated_state": selected_state, "initial_state": initial[selected],
        "raw_initial_gaze": raw_initial, "bounded_initial_gaze": initial_gaze,
        "all_start_raw_initial_gaze": raw_initial_all, "all_start_bounded_initial_gaze": bounded_initial_all,
        "first_raw_corrected_gaze": first_raw.reshape(n, k, 2)[:, 1],
        "first_bounded_corrected_gaze": first_bounded.reshape(n, k, 2)[:, 1],
        "first_corrected_center_mm": first_center.reshape(n, k, 2)[:, 1],
        "corrected_center_absolute_mm": best_center[selected], "raw_final_cubic_gaze": best_raw_gaze[selected],
        "converged": conv[np.arange(n), choice], "no_converged_start": ~any_conv,
        "no_valid_start": ~np.any(valid_candidate.reshape(n, k), axis=1),
        "bounded_fixedpoint_with_raw_pressure": conv[np.arange(n), choice] & np.any(pressure[selected], axis=1),
        "interior_fixedpoint": conv[np.arange(n), choice] & ~np.any(pressure[selected], axis=1),
        "normalized_fixedpoint_residual": best_norm[selected], "observed_shape_sse_mm2": best_sse[selected],
        "iterations": iterations[selected], "selected_start_index": choice,
        "growth_events": growth_events[selected], "cycle_events": cycle_events[selected], "final_alpha": alpha[selected],
        "invalid_update": invalid_update[selected], "iteration_cap": (~converged[selected]) & (iterations[selected] >= max_iterations),
        "stagnation": stagnation[selected],
        "profile_calls": profile_calls[selected], "profile401_fallback_calls": profile_fallbacks[selected],
        "all_start_profile_calls": profile_calls.reshape(n, k), "all_start_profile401_fallback_calls": profile_fallbacks.reshape(n, k),
        "profile_local_minima": best_profile_minima[selected], "distinct_converged_roots": np.any(distinct, axis=1),
        "all_start_states": roots, "all_start_converged": conv, "all_start_valid": valid_candidate.reshape(n, k),
        "all_start_shape_sse_mm2": costs, "all_start_fixedpoint_residual": norms,
        "all_start_iterations": iterations.reshape(n, k), "all_start_cycle_events": cycle_events.reshape(n, k),
        "all_start_growth_events": growth_events.reshape(n, k), "all_start_alpha": alpha.reshape(n, k),
        "all_start_invalid_update": invalid_update.reshape(n, k), "all_start_stagnation": stagnation.reshape(n, k),
        "all_start_raw_pressure": pressure.reshape(n, k, 2), "primary_A2_state": roots[:, 1],
        "primary_A2_converged": conv[:, 1],
        "boundary_A_ADE_BDE": boundary_flags(selected_state, gaze_bound_deg)}
    if record_history:
        result.update({name: value[selected] for name, value in history.items()})
        result.update({f"all_start_{name}": value.reshape((n, k)+value.shape[1:]) for name, value in history.items()})
    return result
