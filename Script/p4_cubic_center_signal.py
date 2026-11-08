"""Predeclared15-value center/gaze map for the75-pose P4 calibration.

The two output axes use different, explicitly declared polynomial bases. This
adds no shape parameters and no independent initializer coefficients.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from p4_centroid_iteration import fit_signal

X_LABELS = ("1", "a", "tx", "tx*ty", "tx^3", "tx*ty^2", "a*tx")
Y_LABELS = ("1", "ty", "tx^2", "ty^2", "tx^2*ty", "ty^3", "a", "a*ty")


def design(accommodation, gaze):
    a = np.asarray(accommodation).ravel()/4.
    ty, tx = np.asarray(gaze)[:, 0]/20., np.asarray(gaze)[:, 1]/20.
    one = np.ones_like(a)
    return (np.stack((one, a, tx, tx*ty, tx**3, tx*ty**2, a*tx), -1),
            np.stack((one, ty, tx**2, ty**2, tx**2*ty, ty**3, a, a*ty), -1))


def design_diagnostics(X):
    singular = np.linalg.svd(X, compute_uv=False)
    norms = np.linalg.norm(X, axis=0)
    scaled = np.linalg.svd(X/np.maximum(norms, 1e-30), compute_uv=False)
    covariance = np.linalg.pinv(X.T@X, rcond=1e-14)
    sd = np.sqrt(np.maximum(np.diag(covariance), 0.))
    correlation = covariance/np.maximum(sd[:, None]*sd[None, :], 1e-30)
    return {"rows": len(X), "columns": X.shape[1],
        "rank": int(np.sum(scaled > scaled[0]*1e-10)), "rank_relative_tolerance": 1e-10,
        "singular_values": singular.tolist(), "scaled_singular_values": scaled.tolist(),
        "condition": float(singular[0]/singular[-1]), "scaled_condition": float(scaled[0]/scaled[-1]),
        "column_norms": norms.tolist(), "unit_target_noise_design_covariance": covariance.tolist(),
        "design_parameter_correlation": correlation.tolist(),
        "covariance_scope": "design geometry under IID unit-variance signal-target errors; repeated P1/calibration correlations are not a physical-noise model"}


@dataclass
class CubicCenterSignal:
    x_coefficients: np.ndarray
    y_coefficients: np.ndarray

    @property
    def parameter_count(self):
        return 15

    @property
    def accommodation_degree(self):
        return 1

    def forward(self, accommodation, gaze):
        X, Y = design(accommodation, gaze)
        return np.column_stack((X@self.x_coefficients, Y@self.y_coefficients))

    def state_jacobian(self, state):
        a, ty, tx = state[:, 0]/4., state[:, 1]/20., state[:, 2]/20.
        b, c = self.x_coefficients, self.y_coefficients
        derivative = np.empty((len(state), 2, 3))
        derivative[:, 0, 0] = (b[1]+b[6]*tx)/4.
        derivative[:, 0, 1] = (b[3]*tx+2*b[5]*tx*ty)/20.
        derivative[:, 0, 2] = (b[2]+b[3]*ty+3*b[4]*tx**2+b[5]*ty**2+b[6]*a)/20.
        derivative[:, 1, 0] = (c[6]+c[7]*ty)/4.
        derivative[:, 1, 1] = (c[1]+2*c[3]*ty+c[4]*tx**2+3*c[5]*ty**2+c[7]*a)/20.
        derivative[:, 1, 2] = (2*c[2]*tx+2*c[4]*tx*ty)/20.
        return derivative

    def matrix_offset(self, accommodation):
        """Own zero-gaze linearization, used only to initialize the inverse."""
        A = np.asarray(accommodation).ravel()
        zero = np.zeros((len(A), 2))
        state = np.column_stack((A, zero))
        return self.state_jacobian(state)[:, :, 1:], self.forward(A, zero)

    def inverse(self, accommodation, difference):
        """Bounded2D LM from this map's own zero-gaze linearization."""
        A, target = np.asarray(accommodation).ravel(), np.asarray(difference)
        matrix, offset = self.matrix_offset(A)
        singular = np.linalg.svd(matrix, compute_uv=False)
        linear_valid = singular[:, -1] > 1e-10*np.maximum(singular[:, 0], 1e-30)
        unbounded = np.einsum("nij,nj->ni", np.linalg.pinv(matrix, rcond=1e-12), target-offset, optimize=True)
        z = np.clip(unbounded/20., -1., 1.)
        damping = np.full(len(A), 1e-3)
        active = linear_valid.copy()
        converged = np.zeros(len(A), bool)
        for _ in range(50):
            ids = np.flatnonzero(active)
            if not len(ids):
                break
            state = np.column_stack((A[ids], z[ids]*20.))
            residual = self.forward(A[ids], z[ids]*20.)-target[ids]
            J = self.state_jacobian(state)[:, :, 1:]*20.
            hessian = np.einsum("nfi,nfj->nij", J, J, optimize=True)
            gradient = np.einsum("nfi,nf->ni", J, residual, optimize=True)
            diag = np.maximum(np.diagonal(hessian, axis1=1, axis2=2), 1e-18)
            blocked = ((z[ids] <= -1.+1e-10) & (gradient > 0)) | ((z[ids] >= 1.-1e-10) & (gradient < 0))
            projected = np.where(blocked, 0., gradient)
            done = np.max(np.abs(projected)/np.sqrt(diag), axis=1) <= 1e-9
            converged[ids[done]], active[ids[done]] = True, False
            use = ~done
            ids, hessian, gradient, diag, blocked, residual = ids[use], hessian[use], gradient[use], diag[use], blocked[use], residual[use]
            if not len(ids):
                continue
            for p in range(2):
                rows = np.flatnonzero(blocked[:, p])
                hessian[rows, p, :], hessian[rows, :, p] = 0., 0.
                hessian[rows, p, p], gradient[rows, p] = 1., 0.
            hessian[:, np.arange(2), np.arange(2)] += damping[ids, None]*diag+1e-18
            step = np.linalg.solve(hessian, -gradient[..., None])[..., 0]
            step /= np.maximum(1., np.max(np.abs(step), axis=1)/.25)[:, None]
            trial = np.clip(z[ids]+step, -1., 1.)
            cost = np.sum((self.forward(A[ids], trial*20.)-target[ids])**2, axis=1)
            accept = np.isfinite(cost) & (cost < np.sum(residual*residual, axis=1))
            z[ids[accept]] = trial[accept]
            damping[ids[accept]] = np.maximum(damping[ids[accept]]/3., 1e-12)
            damping[ids[~accept]] = np.minimum(damping[ids[~accept]]*10., 1e15)
        final_state = np.column_stack((A, z*20.))
        final_J = self.state_jacobian(final_state)[:, :, 1:]
        final_singular = np.linalg.svd(final_J, compute_uv=False)
        final_residual = self.forward(A, z*20.)-target
        gradient = np.einsum("nfi,nf->ni", final_J*20., final_residual, optimize=True)
        diagonal = np.maximum(np.sum((final_J*20.)**2, axis=1), 1e-18)
        blocked = ((z <= -1.+1e-10) & (gradient > 0)) | ((z >= 1.-1e-10) & (gradient < 0))
        normalized = np.max(np.abs(np.where(blocked, 0., gradient))/np.sqrt(diagonal), axis=1)
        converged |= normalized <= 1e-9
        valid = (linear_valid & converged & np.all(np.isfinite(z), axis=1) &
                 (final_singular[:, -1] > final_singular[:, 0]*1e-10))
        # The fourth result is local inverse conditioning. Initial unbounded
        # values are retained separately by the final3D runtime solver.
        return z*20., unbounded, valid, final_singular[:, 0]/np.maximum(final_singular[:, -1], 1e-30)

    def save(self, path, **metadata):
        np.savez_compressed(path, x_coefficients=self.x_coefficients, y_coefficients=self.y_coefficients,
                            x_labels=np.array(X_LABELS), y_labels=np.array(Y_LABELS),
                            parameter_count=np.array(15), **metadata)

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            return cls(data["x_coefficients"], data["y_coefficients"])


def fit_cubic(accommodation, gaze, target):
    X, Y = design(accommodation, gaze)
    bx = np.linalg.lstsq(X, target[:, 0], rcond=None)[0]
    by = np.linalg.lstsq(Y, target[:, 1], rcond=None)[0]
    diagnostics = {"x": design_diagnostics(X), "y": design_diagnostics(Y),
                   "parameter_count": 15}
    diagnostics["rank"] = diagnostics["x"]["rank"]+diagnostics["y"]["rank"]
    return CubicCenterSignal(bx, by), diagnostics


def compare_center_maps(states, target):
    """D6,D12,D15 selection uses ONLY the supplied75 calibration states."""
    if len(states) != 75:
        raise ValueError("D15 comparison requires exactly75 calibration poses")
    maps, details = {}, {}
    for name in ("D6", "D12", "D15"):
        if name == "D15":
            model, identity = fit_cubic(states[:, 0], states[:, 1:], target)
            rank = identity["rank"]
        else:
            model, identity = fit_signal(states[:, 0], states[:, 1:], target, int(name == "D12"))
            rank = 2*identity["feature_rank"]
        folds, losses = [], []
        for gaze in np.unique(states[:, 1:], axis=0):
            if np.all(np.isclose(gaze, 0., atol=1e-10)):
                continue
            held = np.all(np.isclose(states[:, 1:], gaze, atol=1e-10), axis=1)
            if name == "D15":
                fitted, ident = fit_cubic(states[~held, 0], states[~held, 1:], target[~held])
                fold_rank = ident["rank"]
            else:
                fitted, ident = fit_signal(states[~held, 0], states[~held, 1:], target[~held], int(name == "D12"))
                fold_rank = 2*ident["feature_rank"]
            loss = float(np.mean((fitted.forward(states[held, 0], states[held, 1:])-target[held])**2))
            losses.append(loss)
            folds.append({"gaze_ADE_BDE": gaze.tolist(), "coordinate_mse_mm2": loss,
                          "rank": fold_rank, "full_rank": fold_rank == fitted.parameter_count})
        details[name] = {"parameter_count": model.parameter_count, "rank": rank,
            "identifiability": identity, "folds": folds,
            "mean_coordinate_mse_mm2": float(np.mean(losses)),
            "se_coordinate_mse_mm2": float(np.std(losses, ddof=1)/np.sqrt(len(losses))),
            "all_folds_full_rank": all(fold["full_rank"] for fold in folds)}
        maps[name] = model
    eligible = [name for name, row in details.items() if row["rank"] == row["parameter_count"] and row["all_folds_full_rank"]]
    best = min(eligible, key=lambda name: details[name]["mean_coordinate_mse_mm2"])
    threshold = details[best]["mean_coordinate_mse_mm2"]+details[best]["se_coordinate_mse_mm2"]
    selected = min([name for name in eligible if details[name]["mean_coordinate_mse_mm2"] <= threshold],
                   key=lambda name: details[name]["parameter_count"])
    return maps, {"candidates": details, "selected": selected,
        "one_SE_threshold_coordinate_mse_mm2": threshold,
        "selection_data": "75calibration poses only; all three zero-gaze anchors retained in every fold"}
