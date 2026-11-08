"""Explicit physical gaze bounds for old and new limited-range experiments."""
from __future__ import annotations

import numpy as np

from p4_joint_observed_solver_cubic import signal_state_jacobian


class BoundedSignalMap:
    """A coefficient-preserving adapter; adds no calibrated parameters."""
    def __init__(self, signal_map, gaze_bound_deg=12.5):
        self.signal_map = signal_map
        self.gaze_bound_deg = float(gaze_bound_deg)
        if not 0. < self.gaze_bound_deg <= 20.:
            raise ValueError("Invalid explicit physical gaze bound")

    @property
    def parameter_count(self):
        return self.signal_map.parameter_count

    def forward(self, accommodation, gaze):
        return self.signal_map.forward(accommodation, gaze)

    def state_jacobian(self, state):
        return signal_state_jacobian(self.signal_map, state)

    def inverse(self, accommodation, difference):
        A, target = np.asarray(accommodation).ravel(), np.asarray(difference)
        bound = self.gaze_bound_deg
        zero_state = np.column_stack((A, np.zeros((len(A), 2))))
        offset = self.forward(A, zero_state[:, 1:])
        matrix = self.state_jacobian(zero_state)[:, :, 1:]
        singular = np.linalg.svd(matrix, compute_uv=False)
        initial_valid = singular[:, -1] > 1e-10*np.maximum(singular[:, 0], 1e-30)
        unbounded = np.einsum("nij,nj->ni", np.linalg.pinv(matrix, rcond=1e-12), target-offset, optimize=True)
        if not hasattr(self.signal_map, "state_jacobian"):
            # Exact constrained2D linear least squares, with THESE bounds.
            candidates = [np.clip(unbounded, -bound, bound)]
            for fixed in range(2):
                free = 1-fixed
                for value in (-bound, bound):
                    residual = target-offset-matrix[:, :, fixed]*value
                    free_value = np.sum(matrix[:, :, free]*residual, axis=1)/np.maximum(np.sum(matrix[:, :, free]**2, axis=1), 1e-30)
                    candidate = np.empty_like(unbounded)
                    candidate[:, fixed], candidate[:, free] = value, np.clip(free_value, -bound, bound)
                    candidates.append(candidate)
            candidates = np.stack(candidates, axis=1)
            residual = np.einsum("nij,nkj->nki", matrix, candidates, optimize=True)+offset[:, None, :]-target[:, None, :]
            choice = np.argmin(np.sum(residual**2, axis=2), axis=1)
            return candidates[np.arange(len(A)), choice], unbounded, initial_valid, singular[:, 0]/np.maximum(singular[:, -1], 1e-30)
        z = np.clip(unbounded/bound, -1., 1.)
        damping = np.full(len(A), 1e-3)
        active = initial_valid.copy()
        for _ in range(50):
            ids = np.flatnonzero(active)
            if not len(ids):
                break
            state = np.column_stack((A[ids], z[ids]*bound))
            residual = self.forward(A[ids], z[ids]*bound)-target[ids]
            J = self.state_jacobian(state)[:, :, 1:]*bound
            H = np.einsum("nfi,nfj->nij", J, J, optimize=True)
            gradient = np.einsum("nfi,nf->ni", J, residual, optimize=True)
            diagonal = np.maximum(np.diagonal(H, axis1=1, axis2=2), 1e-18)
            blocked = ((z[ids] <= -1.+1e-10) & (gradient > 0)) | ((z[ids] >= 1.-1e-10) & (gradient < 0))
            norm = np.max(np.abs(np.where(blocked, 0., gradient))/np.sqrt(diagonal), axis=1)
            done = norm <= 1e-9
            active[ids[done]] = False
            use = ~done
            ids, H, gradient, diagonal, blocked, residual = ids[use], H[use], gradient[use], diagonal[use], blocked[use], residual[use]
            if not len(ids):
                continue
            for p in range(2):
                rows = np.flatnonzero(blocked[:, p])
                H[rows, p, :], H[rows, :, p] = 0., 0.
                H[rows, p, p], gradient[rows, p] = 1., 0.
            H[:, np.arange(2), np.arange(2)] += damping[ids, None]*diagonal+1e-18
            step = np.linalg.solve(H, -gradient[..., None])[..., 0]
            step /= np.maximum(1., np.max(np.abs(step), axis=1)/.25)[:, None]
            trial = np.clip(z[ids]+step, -1., 1.)
            cost = np.sum((self.forward(A[ids], trial*bound)-target[ids])**2, axis=1)
            accept = np.isfinite(cost) & (cost < np.sum(residual*residual, axis=1))
            z[ids[accept]] = trial[accept]
            damping[ids[accept]] = np.maximum(damping[ids[accept]]/3., 1e-12)
            damping[ids[~accept]] = np.minimum(damping[ids[~accept]]*10., 1e15)
        state = np.column_stack((A, z*bound))
        J = self.state_jacobian(state)[:, :, 1:]
        singular = np.linalg.svd(J, compute_uv=False)
        residual = self.forward(A, z*bound)-target
        gradient = np.einsum("nfi,nf->ni", J*bound, residual, optimize=True)
        diagonal = np.maximum(np.sum((J*bound)**2, axis=1), 1e-18)
        blocked = ((z <= -1.+1e-10) & (gradient > 0)) | ((z >= 1.-1e-10) & (gradient < 0))
        norm = np.max(np.abs(np.where(blocked, 0., gradient))/np.sqrt(diagonal), axis=1)
        valid = initial_valid & (norm <= 1e-9) & (singular[:, -1] > singular[:, 0]*1e-10)
        return z*bound, unbounded, valid, singular[:, 0]/np.maximum(singular[:, -1], 1e-30)
