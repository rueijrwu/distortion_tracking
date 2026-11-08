"""D15-aware copy of the frozen observed-coordinate P4-P1 solver.

No true accommodation, gaze, or center is accepted by this runtime API.
The residual is measured_P4 - measured_P1_mean - D(A,g) - F_j(A,g).
"""
from __future__ import annotations

import numpy as np

from p4_limited_bounds import BoundedSignalMap

STATE_SCALE = np.array([4., 20., 20.])
LOWER = np.array([0., -1., -1.])
UPPER = np.array([1., 1., 1.])
FIXED_STARTS = np.array([0., 2., 4.])
GRADIENT_TOL_MM = 1e-8


def signal_state_jacobian(signal_map, state):
    """Physical columns are A(D), ADE(deg), BDE(deg)."""
    if hasattr(signal_map, "state_jacobian"):
        return signal_map.state_jacobian(state)
    matrix, _ = signal_map.matrix_offset(state[:, 0])
    derivative = np.zeros((len(state), 2, 3))
    derivative[:, :, 1:] = matrix
    if signal_map.accommodation_degree:
        derivative[:, :, 0] = (signal_map.coefficients[3]+state[:, 1:]@signal_map.coefficients[4:6])/4.
    return derivative


def evaluate_observed(model, signal_map, measured_p4, measured_p1_mean, state, *, jacobian=False):
    """Evaluate the same observable objective for every seed and trial."""
    prediction = model.evaluate(*np.asarray(state).T, state_jac=jacobian, diagnostics=True)
    F = prediction["prediction"]
    D = signal_map.forward(state[:, 0], state[:, 1:])
    residual = measured_p4-measured_p1_mean[:, None, :]-D[:, None, :]-F
    finite = np.all(np.isfinite(residual), axis=(1, 2))
    valid = finite & np.all(np.stack([prediction[key] > 0 for key in
        ("denominator", "radial_tangential", "radial_radial", "brown_determinant")]), axis=(0, 2))
    mean4, meanF = np.mean(measured_p4, axis=1), np.mean(F, axis=1)
    result = {"residual_mm": residual.reshape(len(state), 18),
        "sse_mm2": np.sum(residual**2, axis=(1, 2)), "valid_map": valid,
        "predicted_centroid_bias_mm": meanF,
        "corrected_center_absolute_mm": mean4-meanF,
        "predicted_center_absolute_mm": measured_p1_mean+D,
        "center_residual_mm": mean4-meanF-measured_p1_mean-D}
    if jacobian:
        # Residual uses observed minus predicted, hence the minus sign.
        result["state_jacobian"] = -(prediction["state_jacobian"]+
            signal_state_jacobian(signal_map, state)[:, None, :, :]).reshape(len(state), 18, 3)
    return result


def _projected_gradient(z, gradient, diagonal, lower=LOWER, upper=UPPER):
    blocked = (((z <= lower+1e-10) & (gradient > 0.)) |
               ((z >= upper-1e-10) & (gradient < 0.)))
    projected = np.where(blocked, 0., gradient)
    normalized = np.max(np.abs(projected)/np.sqrt(np.maximum(diagonal, 1e-18)), axis=1)
    return projected, normalized, blocked


def solve_observed(model, signal_map, measured_p4, measured_p1_mean, *, max_iterations=80,
                   record_history=True, gaze_bound_deg=12.5):
    """Three blind fixed-A starts; joint bounded damped Gauss-Newton/LM.

    Accepted trials always decrease the SAME observed-coordinate SSE and have
    valid sampled spatial maps. Only verified projected stationarity counts as
    convergence; rejected or arbitrarily tiny damped steps do not count.
    """
    gaze_bound_deg = float(gaze_bound_deg)
    if not 0. < gaze_bound_deg <= 20.:
        raise ValueError("Explicit physical gaze bound must be in(0,20]degrees")
    lower = np.array([0., -gaze_bound_deg/20., -gaze_bound_deg/20.])
    upper = np.array([1., gaze_bound_deg/20., gaze_bound_deg/20.])
    signal_map = BoundedSignalMap(signal_map, gaze_bound_deg)
    y4, m1 = np.asarray(measured_p4, float), np.asarray(measured_p1_mean, float)
    if y4.shape != (len(m1), 9, 2) or m1.shape[1:] != (2,):
        raise ValueError("Expected measured P4(N,9,2) and measured P1 mean(N,2)")
    if not np.all(np.isfinite(y4)) or not np.all(np.isfinite(m1)):
        raise ValueError("All nine measured real fields must be finite")
    n, k = len(m1), len(FIXED_STARTS)
    difference = np.mean(y4, axis=1)-m1
    initial_A = np.tile(FIXED_STARTS, n)
    raw_difference = np.repeat(difference, k, axis=0)
    raw_gaze, unbounded_gaze, signal_valid, signal_condition = signal_map.inverse(initial_A, raw_difference)
    initial_state = np.column_stack((initial_A, raw_gaze))
    target4, target1 = np.repeat(y4, k, axis=0), np.repeat(m1, k, axis=0)
    first = evaluate_observed(model, signal_map, target4, target1, initial_state)
    corrected_difference = raw_difference-first["predicted_centroid_bias_mm"]
    corrected_gaze, correction_unbounded, correction_valid, _ = signal_map.inverse(initial_A, corrected_difference)
    corrected_seed = np.column_stack((initial_A, corrected_gaze))
    check = evaluate_observed(model, signal_map, target4, target1, corrected_seed)
    use_correction = signal_valid & correction_valid & first["valid_map"] & check["valid_map"]
    seed = initial_state.copy()
    seed[use_correction] = corrected_seed[use_correction]
    z = np.clip(seed/STATE_SCALE, lower, upper)
    size = len(z)
    damping = np.full(size, 1e-3)
    converged = np.zeros(size, bool)
    active = np.ones(size, bool)
    iterations = np.zeros(size, np.int16)
    accepted_steps, rejected_steps = np.zeros(size, np.int16), np.zeros(size, np.int16)
    invalid_trial_steps = np.zeros(size, np.int16)
    projected_trial_steps = np.zeros((size, 3), np.int16)
    history = {}
    if record_history:
        for key, width in (("state_history", 3), ("corrected_center_history_mm", 2),
                           ("predicted_center_history_mm", 2), ("raw_proposal_history", 3)):
            history[key] = np.full((size, max_iterations+1, width), np.nan)
        history["sse_history_mm2"] = np.full((size, max_iterations+1), np.nan)
        history["accepted_history"] = np.zeros((size, max_iterations+1), bool)
        initial = evaluate_observed(model, signal_map, target4, target1, seed)
        history["state_history"][:, 0] = seed
        history["corrected_center_history_mm"][:, 0] = initial["corrected_center_absolute_mm"]
        history["predicted_center_history_mm"][:, 0] = initial["predicted_center_absolute_mm"]
        history["sse_history_mm2"][:, 0] = initial["sse_mm2"]
    for iteration in range(max_iterations):
        ids = np.flatnonzero(active)
        if not len(ids):
            break
        current = evaluate_observed(model, signal_map, target4[ids], target1[ids], z[ids]*STATE_SCALE, jacobian=True)
        residual = current["residual_mm"]
        jac = current["state_jacobian"]*STATE_SCALE
        hessian = np.einsum("nfi,nfj->nij", jac, jac, optimize=True)
        gradient = np.einsum("nfi,nf->ni", jac, residual, optimize=True)
        diagonal = np.maximum(np.diagonal(hessian, axis1=1, axis2=2), 1e-18)
        _, norm, blocked = _projected_gradient(z[ids], gradient, diagonal, lower, upper)
        stationary = current["valid_map"] & np.isfinite(norm) & (norm <= GRADIENT_TOL_MM)
        converged[ids[stationary]], active[ids[stationary]] = True, False
        invalid_current = ~current["valid_map"]
        active[ids[invalid_current]] = False
        use = ~stationary & ~invalid_current
        if not np.any(use):
            continue
        ids, hessian, gradient, diagonal, blocked = ids[use], hessian[use], gradient[use], diagonal[use], blocked[use]
        old_cost = current["sse_mm2"][use]
        for parameter in range(3):
            rows = np.flatnonzero(blocked[:, parameter])
            hessian[rows, parameter, :], hessian[rows, :, parameter] = 0., 0.
            hessian[rows, parameter, parameter], gradient[rows, parameter] = 1., 0.
        damped = hessian.copy()
        damped[:, np.arange(3), np.arange(3)] += damping[ids, None]*diagonal+1e-18
        step = np.linalg.solve(damped, -gradient[..., None])[..., 0]
        step /= np.maximum(1., np.max(np.abs(step), axis=1)/.25)[:, None]
        proposed = z[ids]+step
        projected_trial_steps[ids] += (proposed < lower) | (proposed > upper)
        trial_z = np.clip(proposed, lower, upper)
        trial = evaluate_observed(model, signal_map, target4[ids], target1[ids], trial_z*STATE_SCALE)
        accepted = trial["valid_map"] & np.isfinite(trial["sse_mm2"]) & (trial["sse_mm2"] < old_cost)
        invalid_trial_steps[ids] += ~trial["valid_map"]
        z[ids[accepted]] = trial_z[accepted]
        damping[ids[accepted]] = np.maximum(damping[ids[accepted]]/3., 1e-12)
        damping[ids[~accepted]] = np.minimum(damping[ids[~accepted]]*10., 1e15)
        accepted_steps[ids] += accepted
        rejected_steps[ids] += ~accepted
        iterations[ids] = iteration+1
        if record_history:
            accepted_state = evaluate_observed(model, signal_map, target4[ids], target1[ids], z[ids]*STATE_SCALE)
            history["state_history"][ids, iteration+1] = z[ids]*STATE_SCALE
            history["raw_proposal_history"][ids, iteration+1] = proposed*STATE_SCALE
            history["corrected_center_history_mm"][ids, iteration+1] = accepted_state["corrected_center_absolute_mm"]
            history["predicted_center_history_mm"][ids, iteration+1] = accepted_state["predicted_center_absolute_mm"]
            history["sse_history_mm2"][ids, iteration+1] = accepted_state["sse_mm2"]
            history["accepted_history"][ids, iteration+1] = accepted

    final = evaluate_observed(model, signal_map, target4, target1, z*STATE_SCALE, jacobian=True)
    jac = final["state_jacobian"]*STATE_SCALE
    gradient = np.einsum("nfi,nf->ni", jac, final["residual_mm"], optimize=True)
    diagonal = np.sum(jac*jac, axis=1)
    projected_gradient, normalized_gradient, _ = _projected_gradient(z, gradient, diagonal, lower, upper)
    converged |= final["valid_map"] & (normalized_gradient <= GRADIENT_TOL_MM)
    valid_converged = converged & final["valid_map"]
    costs = np.where(np.isfinite(final["sse_mm2"]), final["sse_mm2"], np.inf).reshape(n, k)
    eligible_cost = np.where(valid_converged.reshape(n, k), costs, np.inf)
    any_converged = np.any(np.isfinite(eligible_cost), axis=1)
    best_any = np.argmin(costs, axis=1)
    choice = np.where(any_converged, np.argmin(eligible_cost, axis=1), best_any)
    best = np.arange(n)*k+choice
    best_any_flat = np.arange(n)*k+best_any
    singular = np.full((n, 3), np.nan)
    finite_jacobian = np.all(np.isfinite(jac[best]), axis=(1, 2))
    singular[finite_jacobian] = np.linalg.svd(jac[best][finite_jacobian], compute_uv=False)
    all_state = (z*STATE_SCALE).reshape(n, k, 3)
    separation = np.max(np.abs((all_state-all_state[np.arange(n), choice, None, :])/STATE_SCALE), axis=2)
    distinct = (separation > 1e-3) & valid_converged.reshape(n, k)
    alternate = np.min(np.where(distinct, costs, np.inf), axis=1)
    result = {"estimated_state": z[best]*STATE_SCALE,
        "initial_state": initial_state[best], "centroid_corrected_seed_state": seed[best],
        "converged": valid_converged[best], "no_converged_valid_seed": ~any_converged,
        "all_seeds_nonfinite_objective": ~np.any(np.isfinite(costs), axis=1),
        "valid_map": final["valid_map"][best], "iterations": iterations[best],
        "accepted_steps": accepted_steps[best], "rejected_steps": rejected_steps[best],
        "invalid_trial_steps": invalid_trial_steps[best],
        "observable_joint_sse_mm2": final["sse_mm2"][best],
        "corrected_center_absolute_mm": final["corrected_center_absolute_mm"][best],
        "predicted_center_absolute_mm": final["predicted_center_absolute_mm"][best],
        "center_residual_mm": final["center_residual_mm"][best],
        "predicted_centroid_bias_mm": final["predicted_centroid_bias_mm"][best],
        "raw_gradient_z_mm2": gradient[best], "projected_gradient_z_mm2": projected_gradient[best],
        "projected_normalized_gradient_mm": normalized_gradient[best],
        "normalized_jacobian_singular_values": singular,
        "normalized_jacobian_rank": np.sum(singular > singular[:, :1]*1e-10, axis=1),
        "normalized_jacobian_condition": singular[:, 0]/np.maximum(singular[:, -1], 1e-30),
        "selected_seed_index": choice, "best_any_seed_index": best_any,
        "selected_minus_best_any_sse_mm2": final["sse_mm2"][best]-final["sse_mm2"][best_any_flat],
        "distinct_converged_alternative_sse_gap_mm2": alternate-final["sse_mm2"][best],
        "all_seed_estimated_states": all_state, "all_seed_sse_mm2": costs,
        "all_seed_converged": valid_converged.reshape(n, k),
        "all_seed_valid_map": final["valid_map"].reshape(n, k),
        "all_seed_iterations": iterations.reshape(n, k),
        "all_seed_projected_normalized_gradient_mm": normalized_gradient.reshape(n, k),
        "all_seed_initial_states": initial_state.reshape(n, k, 3),
        "all_seed_centroid_corrected_states": seed.reshape(n, k, 3),
        "all_seed_initial_unconstrained_gaze": unbounded_gaze.reshape(n, k, 2),
        "all_seed_correction_unconstrained_gaze": correction_unbounded.reshape(n, k, 2),
        "all_seed_centroid_correction_used": use_correction.reshape(n, k),
        "all_seed_signal_inverse_valid": signal_valid.reshape(n, k),
        "all_seed_signal_condition": signal_condition.reshape(n, k),
        "projected_trial_steps": projected_trial_steps[best],
        "boundary": (z[best] <= lower+1e-7) | (z[best] >= upper-1e-7)}
    if record_history:
        result.update({key: values[best] for key, values in history.items()})
    return result
