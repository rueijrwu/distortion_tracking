"""Truth-free joint P4 accommodation/gaze fixed point with independent P1 seed."""
from __future__ import annotations

import time

import numpy as np

from p4_fast_accommodation_profile import estimate_accommodation_fast
from p4_known_gaze_inverse import estimate_accommodation
from p4_unified_fullshape import ProfileAdapter

SCALE = np.array([4., 20., 20.])
A_TOL_D = 1e-5
GAZE_TOL_DEG = 1e-4
GAZE_BOUND_DEG = 12.5
MAX_ITERATIONS = 40


def profile_accommodation(adapter, observed, gaze, method):
    if method == "strict401":
        values, timing = estimate_accommodation(adapter, observed, gaze,
                                                mean_centered=True,
                                                refinement_iterations=60)
    elif method == "fast41":
        values, timing = estimate_accommodation_fast(adapter, observed, gaze,
                                                     mean_centered=True,
                                                     refinement_iterations=32)
    else:
        raise ValueError("Profile method must be strict401 or fast41")
    return values, timing


def select_roots(converged, valid, shape_sse, residual):
    """Primary index 1 wins; alternatives rescue by observable shape SSE."""
    selected = np.ones(converged.shape[1], dtype=int)
    for row in range(len(selected)):
        if converged[1, row] and valid[1, row]:
            continue
        alternatives = np.flatnonzero(converged[[0, 2], row] & valid[[0, 2], row])
        if len(alternatives):
            candidates = np.array([0, 2])[alternatives]
            selected[row] = candidates[np.lexsort((candidates, residual[candidates, row],
                                                    shape_sse[candidates, row]))[0]]
        else:
            candidates = np.flatnonzero(valid[:, row] & np.isfinite(residual[:, row]))
            if len(candidates):
                selected[row] = candidates[np.lexsort((candidates, shape_sse[candidates, row],
                                                       residual[candidates, row]))[0]]
    return selected


def solve_joint(observed_p4_mm, p1_gaze_deg, field, motion, p4_gaze_model,
                *, profile_method="strict401", keep_history=False):
    """Only observed P4 grid and independently inferred P1 gaze enter runtime."""
    y4 = np.asarray(observed_p4_mm, float)
    g1 = np.asarray(p1_gaze_deg, float)
    if y4.ndim != 3 or y4.shape[1:] != (9, 2) or g1.shape != (len(y4), 2):
        raise ValueError("Expected P4[N,9,2] and independent observed P1 gaze[N,2]")
    n, starts = len(y4), 3
    m4 = y4.mean(axis=1)
    observed_shape = y4-m4[:, None, :]
    adapter = ProfileAdapter(field)
    profile_started = time.perf_counter()
    initial_profile, _ = profile_accommodation(adapter, y4, np.clip(g1, -GAZE_BOUND_DEG, GAZE_BOUND_DEG),
                                               profile_method)
    initial_profile_seconds = time.perf_counter()-profile_started
    a_seed = np.asarray(initial_profile["estimated_A_d"], float)
    seed_valid = (np.asarray(initial_profile["converged"], bool)
                  & np.asarray(initial_profile["valid_map"], bool) & np.isfinite(a_seed)
                  & np.isfinite(g1).all(axis=1) & np.isfinite(y4).all(axis=(1, 2)))
    initial = np.zeros((starts, n, 3))
    initial[:, :, 1:] = np.clip(g1[None, :, :], -GAZE_BOUND_DEG, GAZE_BOUND_DEG)
    initial[0, :, 0] = 0.
    initial[1, :, 0] = np.clip(np.nan_to_num(a_seed, nan=2.), 0., 4.)
    initial[2, :, 0] = 4.
    current = initial.reshape(-1, 3).copy()
    observation_index = np.tile(np.arange(n), starts)
    count = starts*n
    alive = np.isfinite(current).all(axis=1) & np.isfinite(y4[observation_index]).all(axis=(1, 2))
    alive[n:2*n] &= seed_valid
    converged = np.zeros(count, bool)
    invalid = ~alive.copy()
    consecutive = np.zeros(count, np.int16)
    iterations = np.zeros(count, np.int16)
    growth_events = np.zeros(count, np.int16)
    cycle_events = np.zeros(count, np.int16)
    bound_pressure = np.zeros((count, 3), bool)
    alpha = np.ones(count)
    previous_norm = np.full(count, np.inf)
    previous_state = np.full((count, 3), np.nan)
    best_norm, best_sse = np.full(count, np.inf), np.full(count, np.inf)
    best_state = current.copy()
    best_center = np.full((count, 2), np.nan)
    best_center_ref = np.full((count, 2), np.nan)
    best_raw_gaze = np.full((count, 2), np.nan)
    best_proposal = np.full((count, 3), np.nan)
    best_profile_rms_um = np.full(count, np.nan)
    best_profile_boundary = np.zeros(count, bool)
    best_profile_minima = np.zeros(count, np.int16)
    profile_calls = np.zeros(count, np.int16)
    profile_fallbacks = np.zeros(count, np.int16)
    first_center = np.full((count, 2), np.nan)
    first_center_ref = np.full((count, 2), np.nan)
    first_raw_gaze = np.full((count, 2), np.nan)
    first_proposal = np.full((count, 3), np.nan)
    first_shape_sse = np.full(count, np.nan)
    histories = []

    def evaluate(indices, states):
        rows = observation_index[indices]
        a, g = states[:, 0], states[:, 1:]
        checked = field.evaluate(a, g, diagnostics=True)
        forward = checked["prediction"]
        mean_forward = forward.mean(axis=1)
        center = m4[rows]-mean_forward
        center_ref = motion.inverse(center, a)
        raw_gaze = p4_gaze_model.predict(center_ref)
        bounded_gaze = np.clip(raw_gaze, -GAZE_BOUND_DEG, GAZE_BOUND_DEG)
        valid = checked["valid"] & np.isfinite(center_ref).all(axis=1) & np.isfinite(raw_gaze).all(axis=1)
        shape_residual = observed_shape[rows]-(forward-mean_forward[:, None, :])
        shape_sse = np.sum(shape_residual*shape_residual, axis=(1, 2))
        valid &= np.isfinite(shape_sse)
        proposed_a = np.full(len(indices), np.nan)
        profile_rms = np.full(len(indices), np.nan)
        profile_boundary = np.zeros(len(indices), bool)
        profile_minima = np.zeros(len(indices), np.int16)
        fallback = np.zeros(len(indices), bool)
        eligible = valid & np.isfinite(bounded_gaze).all(axis=1)
        if np.any(eligible):
            prof, _ = profile_accommodation(adapter, y4[rows[eligible]], bounded_gaze[eligible],
                                            profile_method)
            proposed_a[eligible] = prof["estimated_A_d"]
            profile_rms[eligible] = prof["fit_vertex_rms_um"]
            profile_boundary[eligible] = prof.get("boundary_A",
                np.isclose(prof["estimated_A_d"], 0., atol=1e-7, rtol=0)
                | np.isclose(prof["estimated_A_d"], 4., atol=1e-7, rtol=0))
            profile_minima[eligible] = prof["profile_local_minima"]
            if "reference401_fallback" in prof:
                fallback[eligible] = prof["reference401_fallback"]
            profile_ok = np.zeros(len(indices), bool)
            profile_ok[eligible] = prof["converged"] & prof["valid_map"]
            valid &= profile_ok & np.isfinite(proposed_a)
        proposal = np.column_stack((np.clip(proposed_a, 0., 4.), bounded_gaze))
        delta = proposal-states
        residual_norm = np.linalg.norm(delta/SCALE, axis=1)
        valid &= np.isfinite(residual_norm)
        return {"valid": valid, "center": center, "center_ref": center_ref,
                "raw_gaze": raw_gaze, "proposal": proposal, "delta": delta,
                "residual_norm": residual_norm, "shape_sse": shape_sse,
                "profile_rms_um": profile_rms, "profile_boundary": profile_boundary,
                "profile_minima": profile_minima, "fallback": fallback,
                "eligible": eligible}

    for iteration in range(MAX_ITERATIONS):
        indices = np.flatnonzero(alive)
        if not len(indices):
            break
        ev = evaluate(indices, current[indices])
        valid = ev["valid"]
        valid_indices = indices[valid]
        invalid[indices[~valid]] = True
        alive[indices[~valid]] = False
        iterations[indices] = iteration+1
        profile_calls[indices[ev["eligible"]]] += 1
        profile_fallbacks[indices[ev["fallback"]]] += 1
        if iteration == 0:
            first_center[indices] = ev["center"]
            first_center_ref[indices] = ev["center_ref"]
            first_raw_gaze[indices] = ev["raw_gaze"]
            first_proposal[indices] = ev["proposal"]
            first_shape_sse[indices] = ev["shape_sse"]
        bound_pressure[indices, 1:] |= np.abs(ev["raw_gaze"]) > GAZE_BOUND_DEG
        bound_pressure[indices, 0] |= ev["profile_boundary"]
        local_norm = ev["residual_norm"]
        improved = valid & ((local_norm < best_norm[indices])
                            | ((local_norm == best_norm[indices]) & (ev["shape_sse"] < best_sse[indices])))
        take = indices[improved]
        best_norm[take] = local_norm[improved]
        best_sse[take] = ev["shape_sse"][improved]
        best_state[take] = current[take]
        best_center[take] = ev["center"][improved]
        best_center_ref[take] = ev["center_ref"][improved]
        best_raw_gaze[take] = ev["raw_gaze"][improved]
        best_proposal[take] = ev["proposal"][improved]
        best_profile_rms_um[take] = ev["profile_rms_um"][improved]
        best_profile_boundary[take] = ev["profile_boundary"][improved]
        best_profile_minima[take] = ev["profile_minima"][improved]
        step_ok = valid & (np.abs(ev["delta"][:, 0]) <= A_TOL_D) & (
            np.linalg.norm(ev["delta"][:, 1:], axis=1) <= GAZE_TOL_DEG)
        consecutive[indices] = np.where(step_ok, consecutive[indices]+1, 0)
        just_converged = indices[valid & (consecutive[indices] >= 2)]
        converged[just_converged] = True
        if len(just_converged):
            loc = np.searchsorted(indices, just_converged)
            best_state[just_converged] = current[just_converged]
            best_norm[just_converged] = local_norm[loc]
            best_sse[just_converged] = ev["shape_sse"][loc]
            best_center[just_converged] = ev["center"][loc]
            best_center_ref[just_converged] = ev["center_ref"][loc]
            best_raw_gaze[just_converged] = ev["raw_gaze"][loc]
            best_proposal[just_converged] = ev["proposal"][loc]
            best_profile_rms_um[just_converged] = ev["profile_rms_um"][loc]
            best_profile_boundary[just_converged] = ev["profile_boundary"][loc]
            best_profile_minima[just_converged] = ev["profile_minima"][loc]
        growth = valid & np.isfinite(previous_norm[indices]) & (local_norm > 1.2*previous_norm[indices])
        cycle = (valid & np.isfinite(previous_state[indices]).all(axis=1)
                 & (np.linalg.norm((ev["proposal"]-previous_state[indices])/SCALE, axis=1) <= 1e-5)
                 & (local_norm > 1e-5))
        growth_events[indices[growth]] += 1
        cycle_events[indices[cycle]] += 1
        damp = indices[growth | cycle]
        alpha[damp] = np.maximum(alpha[damp]/2., 1./32.)
        if keep_history:
            histories.append({"indices": indices.copy(), "state": current[indices].copy(),
                              "proposal": ev["proposal"].copy(), "valid": valid.copy(),
                              "residual_norm": local_norm.copy(), "alpha": alpha[indices].copy()})
        previous_state[valid_indices] = current[valid_indices]
        previous_norm[valid_indices] = local_norm[valid]
        alive[just_converged] = False
        updating = indices[valid & alive[indices]]
        if len(updating):
            loc = np.searchsorted(indices, updating)
            current[updating] += alpha[updating, None]*ev["delta"][loc]
            current[updating, 0] = np.clip(current[updating, 0], 0., 4.)
            current[updating, 1:] = np.clip(current[updating, 1:], -GAZE_BOUND_DEG, GAZE_BOUND_DEG)

    all_converged = converged.reshape(starts, n)
    all_valid = np.isfinite(best_norm).reshape(starts, n)
    all_sse = best_sse.reshape(starts, n)
    all_norm = best_norm.reshape(starts, n)
    selected = select_roots(all_converged, all_valid, all_sse, all_norm)
    chosen = selected*n+np.arange(n)
    valid = all_valid[selected, np.arange(n)]
    state = np.where(valid[:, None], best_state[chosen], np.nan)
    result = {
        "estimated_state": state, "estimated_A_d": state[:, 0], "gaze_deg": state[:, 1:],
        "center_mm": np.where(valid[:, None], best_center[chosen], np.nan),
        "center_A0_reference_mm": np.where(valid[:, None], best_center_ref[chosen], np.nan),
        "converged": all_converged[selected, np.arange(n)] & valid,
        "valid": valid, "selected_start": selected,
        "shape_sse_mm2": best_sse[chosen],
        "fixedpoint_residual_normalized": best_norm[chosen],
        "raw_G4_deg": best_raw_gaze[chosen],
        "bounded_proposal_state": best_proposal[chosen],
        "profile_vertex_rms_um": best_profile_rms_um[chosen],
        "profile_boundary_A": best_profile_boundary[chosen],
        "profile_local_minima": best_profile_minima[chosen],
        "initial_P1_conditioned_profile_A_d": a_seed,
        "initial_P1_conditioned_profile_valid": seed_valid,
        "initial_P1_conditioned_profile_vertex_rms_um": initial_profile["fit_vertex_rms_um"],
        "initial_P1_conditioned_profile_boundary_A": initial_profile.get("boundary_A",
            np.isclose(a_seed, 0., atol=1e-7, rtol=0)
            | np.isclose(a_seed, 4., atol=1e-7, rtol=0)),
        "initial_P1_conditioned_profile_local_minima": initial_profile["profile_local_minima"],
        "raw_independent_G4_from_mean_deg": p4_gaze_model.predict(m4),
        "first_center_mm": first_center.reshape(starts, n, 2),
        "first_center_A0_reference_mm": first_center_ref.reshape(starts, n, 2),
        "first_raw_G4_deg": first_raw_gaze.reshape(starts, n, 2),
        "first_proposal_state": first_proposal.reshape(starts, n, 3),
        "first_shape_sse_mm2": first_shape_sse.reshape(starts, n),
        "primary_state": best_state.reshape(starts, n, 3)[1],
        "primary_converged": all_converged[1] & all_valid[1],
        "primary_valid": all_valid[1],
        "alternate_start_rescue": ~(all_converged[1] & all_valid[1]) & all_converged[selected, np.arange(n)] & valid,
        "all_initial_state": initial,
        "all_best_state": best_state.reshape(starts, n, 3),
        "all_converged": all_converged, "all_valid": all_valid,
        "all_invalid": invalid.reshape(starts, n),
        "all_iterations": iterations.reshape(starts, n),
        "all_alpha": alpha.reshape(starts, n),
        "all_growth_events": growth_events.reshape(starts, n),
        "all_cycle_events": cycle_events.reshape(starts, n),
        "all_bound_pressure": bound_pressure.reshape(starts, n, 3),
        "all_shape_sse_mm2": all_sse,
        "all_fixedpoint_residual_normalized": all_norm,
        "all_profile_calls": profile_calls.reshape(starts, n),
        "all_profile_fallbacks": profile_fallbacks.reshape(starts, n),
        "bound_A_ADE_BDE": np.column_stack((np.isclose(state[:, 0], 0., atol=1e-7, rtol=0)
                            | np.isclose(state[:, 0], 4., atol=1e-7, rtol=0),
                            np.isclose(np.abs(state[:, 1:]), GAZE_BOUND_DEG, atol=1e-7, rtol=0))),
        "wall_initial_profile_seconds": initial_profile_seconds,
    }
    if np.any(result["primary_converged"] & (selected != 1)):
        raise AssertionError("Converged P1-conditioned primary was replaced")
    if keep_history:
        result["history_chunks"] = np.array(histories, dtype=object)
    return result


def solve_fixed_A_control(observed_p4_mm, held_A_d, p1_gaze_deg, field, motion, p4_gaze_model):
    """Hold the P1-conditioned profile A and iterate only the P4 center gaze."""
    y4 = np.asarray(observed_p4_mm, float)
    m4 = y4.mean(axis=1)
    held_a = np.asarray(held_A_d, float)
    gaze = np.clip(np.asarray(p1_gaze_deg, float), -GAZE_BOUND_DEG, GAZE_BOUND_DEG).copy()
    n = len(gaze)
    alive = np.isfinite(gaze).all(axis=1) & np.isfinite(held_a) & np.isfinite(y4).all(axis=(1, 2))
    converged, invalid = np.zeros(n, bool), ~alive.copy()
    consecutive = np.zeros(n, np.int16)
    iterations = np.zeros(n, np.int16)
    alpha = np.ones(n)
    previous_norm = np.full(n, np.inf)
    previous_gaze = np.full_like(gaze, np.nan)
    growth_count, cycle_count = np.zeros(n, np.int16), np.zeros(n, np.int16)
    best_norm = np.full(n, np.inf)
    best_gaze = gaze.copy()
    best_center = np.full((n, 2), np.nan)
    best_ref = np.full((n, 2), np.nan)

    for iteration in range(MAX_ITERATIONS):
        indices = np.flatnonzero(alive)
        if not len(indices):
            break
        evaluated = field.evaluate(held_a[indices], gaze[indices], diagnostics=True)
        center = m4[indices]-evaluated["prediction"].mean(axis=1)
        center_ref = motion.inverse(center, held_a[indices])
        raw = p4_gaze_model.predict(center_ref)
        proposal = np.clip(raw, -GAZE_BOUND_DEG, GAZE_BOUND_DEG)
        delta = proposal-gaze[indices]
        norm = np.linalg.norm(delta, axis=1)
        valid = evaluated["valid"] & np.isfinite(proposal).all(axis=1) & np.isfinite(norm)
        invalid[indices[~valid]] = True
        alive[indices[~valid]] = False
        iterations[indices] = iteration+1
        improved = valid & (norm < best_norm[indices])
        take = indices[improved]
        best_norm[take] = norm[improved]
        best_gaze[take] = gaze[take]
        best_center[take] = center[improved]
        best_ref[take] = center_ref[improved]
        consecutive[indices] = np.where(valid & (norm <= GAZE_TOL_DEG), consecutive[indices]+1, 0)
        just = indices[valid & (consecutive[indices] >= 2)]
        converged[just] = True
        if len(just):
            loc = np.searchsorted(indices, just)
            best_norm[just] = norm[loc]
            best_gaze[just] = gaze[just]
            best_center[just] = center[loc]
            best_ref[just] = center_ref[loc]
        growth = valid & np.isfinite(previous_norm[indices]) & (norm > 1.2*previous_norm[indices])
        cycle = (valid & np.isfinite(previous_gaze[indices]).all(axis=1)
                 & (np.linalg.norm(proposal-previous_gaze[indices], axis=1) <= GAZE_TOL_DEG)
                 & (norm > GAZE_TOL_DEG))
        growth_count[indices[growth]] += 1
        cycle_count[indices[cycle]] += 1
        damp = indices[growth | cycle]
        alpha[damp] = np.maximum(alpha[damp]/2., 1./32.)
        good = indices[valid]
        previous_gaze[good] = gaze[good]
        previous_norm[good] = norm[valid]
        alive[just] = False
        update = indices[valid & alive[indices]]
        if len(update):
            loc = np.searchsorted(indices, update)
            gaze[update] = np.clip(gaze[update]+alpha[update, None]*delta[loc],
                                   -GAZE_BOUND_DEG, GAZE_BOUND_DEG)
    finite = np.isfinite(best_norm)
    return {"held_A_d": held_a, "gaze_deg": np.where(finite[:, None], best_gaze, np.nan),
            "center_mm": best_center, "center_A0_reference_mm": best_ref,
            "converged": converged & finite, "valid": finite,
            "fixedpoint_residual_deg": best_norm,
            "iterations": iterations, "invalid": invalid,
            "growth_events": growth_count, "cycle_events": cycle_count,
            "bound_ADE_BDE": np.isclose(np.abs(best_gaze), GAZE_BOUND_DEG, atol=1e-7, rtol=0)}
