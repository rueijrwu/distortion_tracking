"""Calibrated P4-P1 centroid correction with no true-state runtime inputs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from p4_known_gaze_inverse import estimate_accommodation


def signal_features(accommodation, gaze, degree):
    gaze = np.asarray(gaze, float)
    base = np.column_stack((np.ones(len(gaze)), gaze))
    return np.concatenate((base, (np.asarray(accommodation)/4.)[:, None]*base), axis=1) if degree else base


@dataclass
class CenterSignalMap:
    coefficients: np.ndarray
    accommodation_degree: int = 0

    @property
    def parameter_count(self):
        return self.coefficients.size

    def matrix_offset(self, accommodation):
        a = np.asarray(accommodation).ravel()/4.
        b = np.broadcast_to(self.coefficients[0], (len(a), 2)).copy()
        matrix = np.broadcast_to(self.coefficients[1:3].T, (len(a), 2, 2)).copy()
        if self.accommodation_degree:
            b += a[:, None]*self.coefficients[3]
            matrix += a[:, None, None]*self.coefficients[4:6].T
        return matrix, b

    def forward(self, accommodation, gaze):
        matrix, b = self.matrix_offset(accommodation)
        return b+np.einsum("nij,nj->ni", matrix, np.asarray(gaze), optimize=True)

    def inverse(self, accommodation, difference):
        """Box-constrained 2D least squares, preserving unconstrained proposals."""
        matrix, b = self.matrix_offset(accommodation)
        target = np.asarray(difference)-b
        singular = np.linalg.svd(matrix, compute_uv=False)
        valid = singular[:, -1] > 1e-10*np.maximum(singular[:, 0], 1e-30)
        unconstrained = np.einsum("nij,nj->ni", np.linalg.pinv(matrix, rcond=1e-12), target, optimize=True)
        candidates = [np.clip(unconstrained, -20., 20.)]
        for fixed in range(2):
            free = 1-fixed
            for value in (-20., 20.):
                residual = target-matrix[:, :, fixed]*value
                free_value = np.sum(matrix[:, :, free]*residual, axis=1)/np.maximum(np.sum(matrix[:, :, free]**2, axis=1), 1e-30)
                candidate = np.empty_like(unconstrained)
                candidate[:, fixed], candidate[:, free] = value, np.clip(free_value, -20., 20.)
                candidates.append(candidate)
        candidates = np.stack(candidates, axis=1)
        residual = np.einsum("nij,nkj->nki", matrix, candidates, optimize=True)-target[:, None, :]
        choice = np.argmin(np.sum(residual**2, axis=2), axis=1)
        bounded = candidates[np.arange(len(matrix)), choice]
        return bounded, unconstrained, valid, singular[:, 0]/np.maximum(singular[:, -1], 1e-30)

    def save(self, path, **metadata):
        np.savez_compressed(path, coefficients=self.coefficients,
            accommodation_degree=np.array(self.accommodation_degree), parameter_count=np.array(self.parameter_count), **metadata)

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            return cls(data["coefficients"], int(data["accommodation_degree"]))


def fit_signal(accommodation, gaze, observed_difference, degree):
    X = signal_features(accommodation, gaze, degree)
    beta, _, rank, singular = np.linalg.lstsq(X, observed_difference, rcond=None)
    norms = np.linalg.norm(X, axis=0)
    scaled = np.linalg.svd(X/np.maximum(norms, 1e-30), compute_uv=False)
    return CenterSignalMap(beta, degree), {"feature_rank": int(rank), "features": X.shape[1],
        "parameter_count": int(beta.size), "singular_values": singular.tolist(),
        "scaled_singular_values": scaled.tolist(), "scaled_condition": float(scaled[0]/scaled[-1])}


def select_signal_map(states, corrected_center_difference):
    """Calibration-only CV; retain all zero-gaze anchors in every fold."""
    rows, maps = {}, {}
    gazes = np.unique(states[:, 1:], axis=0)
    for degree in (0, 1):
        model, rank = fit_signal(states[:, 0], states[:, 1:], corrected_center_difference, degree)
        folds = []
        for gaze in gazes:
            if np.all(np.isclose(gaze, 0., atol=1e-10)):
                continue
            held = np.all(np.isclose(states[:, 1:], gaze, atol=1e-10), axis=1)
            fitted, identity = fit_signal(states[~held, 0], states[~held, 1:], corrected_center_difference[~held], degree)
            prediction = fitted.forward(states[held, 0], states[held, 1:])
            loss = float(np.mean((prediction-corrected_center_difference[held])**2))
            folds.append({"gaze_ADE_BDE": gaze.tolist(), "coordinate_mse_mm2": loss,
                          "full_rank": identity["feature_rank"] == identity["features"]})
        losses = np.array([fold["coordinate_mse_mm2"] for fold in folds])
        rows[str(degree)] = {"identifiability": rank, "folds": folds,
            "mean_coordinate_mse_mm2": float(np.mean(losses)),
            "se_coordinate_mse_mm2": float(np.std(losses, ddof=1)/np.sqrt(len(losses))),
            "all_folds_full_rank": all(fold["full_rank"] for fold in folds)}
        maps[degree] = model
    eligible = [d for d in (0, 1) if rows[str(d)]["identifiability"]["feature_rank"] == rows[str(d)]["identifiability"]["features"] and rows[str(d)]["all_folds_full_rank"]]
    if not eligible:
        raise RuntimeError("No identifiable center-signal calibration")
    best = min(eligible, key=lambda d: rows[str(d)]["mean_coordinate_mse_mm2"])
    threshold = rows[str(best)]["mean_coordinate_mse_mm2"]+rows[str(best)]["se_coordinate_mse_mm2"]
    selected = min(d for d in eligible if rows[str(d)]["mean_coordinate_mse_mm2"] <= threshold)
    return maps[selected], {"selected_degree": selected, "candidate_CV": rows,
                             "selection": "lower-count full-rank map within one SE; calibration only"}


def observable_residual(model, signal_map, absolute_p4, mean_p1, state):
    """Equivalent to all9 absolute P4-ray residuals relative to measured P1 mean."""
    prediction = model.evaluate(*np.asarray(state).T, diagnostics=True)
    F = prediction["prediction"]
    meanF = np.mean(F, axis=1)
    mean4 = np.mean(absolute_p4, axis=1)
    difference = mean4-mean_p1
    signal = signal_map.forward(state[:, 0], state[:, 1:])
    shape = (absolute_p4-mean4[:, None, :])-(F-meanF[:, None, :])
    mean_residual = difference-signal-meanF
    sse = np.sum(shape**2, axis=(1, 2))+9.*np.sum(mean_residual**2, axis=1)
    valid = np.all(np.stack([prediction[key] > 0 for key in
        ("denominator", "radial_tangential", "radial_radial", "brown_determinant")]), axis=(0, 2))
    return sse, meanF, mean4-meanF, valid


def iterate_centroids(model, signal_map, absolute_p4, mean_p1, *, initial_A=2., max_iterations=30, log=None):
    """Runtime API accepts measured rays/maps only; there are no truth labels."""
    absolute_p4, mean_p1 = np.asarray(absolute_p4, float), np.asarray(mean_p1, float)
    if absolute_p4.shape != (len(mean_p1), 9, 2) or mean_p1.shape[1:] != (2,):
        raise ValueError("Expected nine measured P4 points and measured P1 mean")
    if not np.all(np.isfinite(absolute_p4)) or not np.all(np.isfinite(mean_p1)):
        raise ValueError("All nine real rays must be finite; changing-field means are prohibited")
    n = len(mean_p1)
    raw_difference = np.mean(absolute_p4, axis=1)-mean_p1
    A = np.full(n, float(initial_A))
    gaze, unconstrained, signal_valid, _ = signal_map.inverse(A, raw_difference)
    state = np.column_stack((A, gaze))
    initial_state = state.copy()
    sse, meanF, center, valid_map = observable_residual(model, signal_map, absolute_p4, mean_p1, state)
    state_history = np.full((n, max_iterations+1, 3), np.nan)
    proposal_history = np.full((n, max_iterations, 3), np.nan)
    unconstrained_gaze_history = np.full((n, max_iterations, 2), np.nan)
    sse_history = np.full((n, max_iterations+1), np.nan)
    damping_history = np.full((n, max_iterations), np.nan)
    state_history[:, 0], sse_history[:, 0] = state, sse
    active = signal_valid & valid_map
    converged, cycle, stagnation = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
    iterations = np.zeros(n, np.int16)
    consecutive_small, consecutive_stall = np.zeros(n, np.int16), np.zeros(n, np.int16)
    out_of_domain = np.any(np.abs(unconstrained) > 20.+1e-10, axis=1).astype(np.int16)
    backtracks = np.zeros(n, np.int16)
    scalar_failures = np.zeros(n, np.int16)
    for iteration in range(max_iterations):
        ids = np.flatnonzero(active)
        if not len(ids):
            break
        new_gaze, raw_gaze, signal_ok, _ = signal_map.inverse(state[ids, 0], raw_difference[ids]-meanF[ids])
        signal_valid[ids] &= signal_ok
        proposal_history[ids, iteration, 1:] = new_gaze
        unconstrained_gaze_history[ids, iteration] = raw_gaze
        out_of_domain[ids] += np.any(np.abs(raw_gaze) > 20.+1e-10, axis=1)
        inverse, _ = estimate_accommodation(model, absolute_p4[ids], new_gaze,
            mean_centered=True, log=None, refinement_iterations=60)
        scalar_failures[ids] += ~inverse["converged"]
        proposal = np.column_stack((inverse["estimated_A_d"], new_gaze))
        proposal_history[ids, iteration] = proposal
        change = np.abs(proposal-state[ids])
        small = (change[:, 0] <= 1e-4) & (np.max(change[:, 1:], axis=1) <= 1e-3)
        consecutive_small[ids] = np.where(small, consecutive_small[ids]+1, 0)
        proposed_cost, _, _, proposed_valid = observable_residual(model, signal_map, absolute_p4[ids], mean_p1[ids], proposal)
        accepted = np.isfinite(proposed_cost) & proposed_valid & signal_ok & (proposed_cost <= sse[ids]*(1.+1e-12)+1e-20)
        chosen = state[ids].copy()
        chosen[accepted] = proposal[accepted]
        alpha = np.zeros(len(ids))
        alpha[accepted] = 1.
        remaining = ~accepted & signal_ok
        for step in range(1, 9):
            if not np.any(remaining):
                break
            local_ids = np.flatnonzero(remaining)
            factor = 2.**(-step)
            candidate = state[ids[local_ids]]+factor*(proposal[local_ids]-state[ids[local_ids]])
            cost, _, _, candidate_valid = observable_residual(model, signal_map,
                absolute_p4[ids[local_ids]], mean_p1[ids[local_ids]], candidate)
            good = candidate_valid & np.isfinite(cost) & (cost <= sse[ids[local_ids]]*(1.+1e-12)+1e-20)
            use = local_ids[good]
            chosen[use], alpha[use], accepted[use], remaining[use] = candidate[good], factor, True, False
        backtracks[ids] += (alpha > 0.) & (alpha < 1.)
        consecutive_stall[ids] = np.where(accepted, 0, consecutive_stall[ids]+1)
        damping_history[ids, iteration] = alpha
        state[ids] = chosen
        sse[ids], meanF[ids], center[ids], valid_map[ids] = observable_residual(model, signal_map, absolute_p4[ids], mean_p1[ids], state[ids])
        iterations[ids] = iteration+1
        state_history[ids, iteration+1], sse_history[ids, iteration+1] = state[ids], sse[ids]
        done = (consecutive_small[ids] >= 2) & signal_ok & valid_map[ids]
        converged[ids[done]], active[ids[done]] = True, False
        stalled = consecutive_stall[ids] >= 3
        stagnation[ids[stalled]], active[ids[stalled]] = True, False
        if iteration >= 1:
            two_back = state_history[ids, iteration-1]
            distance = np.abs(state[ids]-two_back)
            cycling = ((distance[:, 0] <= 1e-4) & (np.max(distance[:, 1:], axis=1) <= 1e-3) &
                       ~small & (alpha > 0.))
            cycle[ids[cycling]], active[ids[cycling]] = True, False
        active[ids[~signal_ok | ~valid_map[ids]]] = False
        if log:
            log(f"centroid iteration{iteration+1}:active{np.sum(active):,},converged{np.sum(converged):,},stagnated{np.sum(stagnation):,},cycles{np.sum(cycle):,}")
    return {"estimated_state": state, "initial_state": initial_state,
        "initial_unconstrained_gaze": unconstrained, "estimated_center_absolute_mm": center,
        "predicted_centroid_bias_mm": meanF, "observable_joint_sse_mm2": sse,
        "converged": converged, "cycle": cycle, "stagnation": stagnation,
        "max_iterations": active, "valid_map": valid_map, "signal_map_valid": signal_valid,
        "iterations": iterations, "unconstrained_out_of_domain_updates": out_of_domain,
        "backtracked_updates": backtracks, "scalar_inverse_failures": scalar_failures,
        "boundary": np.isclose(state, [0., -20., -20.], atol=1e-7, rtol=0) | np.isclose(state, [4., 20., 20.], atol=1e-7, rtol=0),
        "state_history": state_history, "undamped_proposal_history": proposal_history,
        "unconstrained_gaze_history": unconstrained_gaze_history, "joint_sse_history": sse_history,
        "accepted_step_fraction_history": damping_history}
