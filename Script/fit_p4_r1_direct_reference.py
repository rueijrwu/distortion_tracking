"""Fit and invert the P4 R1 direct-reference model without barrel removal.

Only the stored real-ray grid is read.  No CODE V import or optical commands
are used.  Run --check-only for derivative and observation-only inverse checks.
The primary experiment uses 38 shared calibration coefficients and estimates
all three unknowns (accommodation, ADE, BDE) from eight centered real vertices.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "6"

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "data/distortion_grid_p4_biaxial_macro/distortion_grid.pkl"
DEFAULT_OUT = ROOT / "data/p4_r1_direct_reference"
L_MM = 1.235429592627
C0_MM2 = -0.0185004967
OUTER = np.array([0, 1, 2, 3, 5, 6, 7, 8])
FIELDS = np.array([(x, y) for x in (-1., 0., 1.) for y in (-1., 0., 1.)])
GROUPS = (
    ("m11", ((2, 0), (0, 2), (4, 0), (2, 2), (0, 4))),
    ("m22", ((2, 0), (0, 2), (4, 0), (2, 2), (0, 4))),
    ("g", ((2, 0), (0, 2), (4, 0), (2, 2), (0, 4))),
    ("m12", ((1, 1), (3, 1), (1, 3))),
    ("m21", ((1, 1), (3, 1), (1, 3))),
    ("qx", ((1, 0), (3, 0), (1, 2))),
    ("p2", ((1, 0), (3, 0), (1, 2))),
    ("qy", ((0, 1), (2, 1), (0, 3))),
    ("p1", ((0, 1), (2, 1), (0, 3))),
)
SPANS = ((0, 5), (5, 10), (10, 15), (15, 18), (18, 21), (21, 24), (24, 27), (27, 30), (30, 33))
PARAM_FOR_COEFF = np.repeat(np.arange(9), [hi - lo for lo, hi in SPANS])
LABELS = [f"{name}:tx^{i}ty^{j}" for name, powers in GROUPS for i, j in powers]
LOWER = np.array([0., -1., -1.])
UPPER = np.array([1., 1., 1.])
STATE_SCALE = np.array([4., 20., 20.])


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_grid(path=GRID):
    """Return all real-ray states, including failed paraxial/probe states."""
    raw = Path(path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    data = pickle.loads(raw)
    del raw
    A, ADE, BDE = (np.unique(data[k]) for k in
                   ("accommodation_d", "eye_rotation_deg", "eye_rotation_x_deg"))
    assert len(data) == 21 * 101 * 101 * 9
    for actual, expected in ((A, np.linspace(0, 4, 21)),
                             (ADE, np.linspace(-20, 20, 101)),
                             (BDE, np.linspace(-20, 20, 101))):
        assert np.allclose(actual, expected, atol=1e-9, rtol=0)
    ai = np.searchsorted(A, data["accommodation_d"])
    di = np.searchsorted(ADE, data["eye_rotation_deg"])
    bi = np.searchsorted(BDE, data["eye_rotation_x_deg"])
    order = np.lexsort((data["field_y_relative"], data["field_x_relative"], di, bi, ai))
    ordered = data[order]
    del data, ai, di, bi, order
    shape = (len(A), len(BDE), len(ADE), 9)
    fields = np.stack((ordered["field_x_relative"], ordered["field_y_relative"]), -1).reshape(shape + (2,))
    assert np.array_equal(fields, np.broadcast_to(FIELDS, fields.shape))
    for name, values in (("accommodation_d", A[:, None, None, None]),
                         ("eye_rotation_x_deg", BDE[None, :, None, None]),
                         ("eye_rotation_deg", ADE[None, None, :, None])):
        assert np.array_equal(ordered[name].reshape(shape), np.broadcast_to(values, shape))
    real = np.stack((ordered["real_x_mm"], ordered["real_y_mm"]), -1).reshape(shape + (2,))
    probe_failed = np.any((~np.isfinite(ordered["paraxial_x_mm"]) |
                           ~np.isfinite(ordered["paraxial_y_mm"])).reshape(shape), axis=-1)
    assert np.all(np.isfinite(real)), "Real-ray observation is nonfinite"
    centered = real - real[..., 4:5, :]
    return A, ADE, BDE, centered, probe_failed, sha


def gaze_basis(ade, bde, derivative=False):
    """Exact 33-term symmetric R1 basis; derivatives use physical degrees."""
    tx, ty = np.asarray(bde).ravel() / 20., np.asarray(ade).ravel() / 20.
    terms, dax, day = [], [], []
    for _, powers in GROUPS:
        for i, j in powers:
            terms.append(tx**i * ty**j)
            if derivative:
                dax.append((i / 20.) * tx**(i - 1) * ty**j if i else np.zeros_like(tx))
                day.append((j / 20.) * tx**i * ty**(j - 1) if j else np.zeros_like(tx))
    basis = np.stack(terms, -1)
    if derivative:
        return basis, np.stack(day, -1), np.stack(dax, -1)
    return basis


def inverse_fixed_reference_barrel(reference):
    """Declared fixed-c0 gauge for the separately labelled original-R1 control."""
    radius = np.linalg.norm(reference, axis=-1)
    undistorted = radius.copy()
    for _ in range(16):
        undistorted -= (undistorted + C0_MM2 * undistorted**3 - radius) / (1. + 3. * C0_MM2 * undistorted**2)
    assert np.all(1. + 3. * C0_MM2 * undistorted**2 > 0)
    result = reference * np.divide(undistorted, radius, out=np.ones_like(radius), where=radius > 0)[:, None]
    assert np.max(np.abs(result * (1. + C0_MM2 * np.sum(result**2, axis=-1, keepdims=True)) - reference)) < 5e-13
    return result


@dataclass
class Model:
    reference_mm: np.ndarray
    calibration: np.ndarray
    coefficients: np.ndarray
    length_mm: float = L_MM
    variant: str = "direct"

    def __post_init__(self):
        if self.variant not in ("direct", "original"):
            raise ValueError(self.variant)
        self.input_reference_mm = (self.reference_mm.copy() if self.variant == "direct" else
                                   inverse_fixed_reference_barrel(self.reference_mm))
        self.base_kappa = 0. if self.variant == "direct" else C0_MM2 * self.length_mm**2

    def evaluate(self, accommodation, ade, bde, *, theta_jac=False, state_jac=False, diagnostics=False):
        """Predict centered mm; optionally differentiate coefficients/states.

        State columns are physical (A in D, ADE and BDE in degrees).  The
        reference is the measured real A=0, zero-gaze grid, used directly.
        """
        A, ade, bde = np.broadcast_arrays(np.asarray(accommodation, float),
                                         np.asarray(ade, float), np.asarray(bde, float))
        A, ade, bde = A.ravel(), ade.ravel(), bde.ravel()
        a = A / 4.
        if state_jac:
            basis, b_ade, b_bde = gaze_basis(ade, bde, derivative=True)
        else:
            basis = gaze_basis(ade, bde)
        pars = np.zeros((len(A), 9))
        if state_jac:
            pars_d = np.zeros((len(A), 9, 2))
        for j, (lo, hi) in enumerate(SPANS):
            pars[:, j] = basis[:, lo:hi] @ self.coefficients[lo:hi]
            if state_jac:
                pars_d[:, j, 0] = b_ade[:, lo:hi] @ self.coefficients[lo:hi]
                pars_d[:, j, 1] = b_bde[:, lo:hi] @ self.coefficients[lo:hi]
        pars[:, :2] += 1.
        s = 1. + self.calibration[0] * a
        kap_a = self.base_kappa + sum(self.calibration[j] * a**j for j in range(1, 5))
        reference = self.input_reference_mm / self.length_mm
        vx = s[:, None] * reference[None, :, 0]
        vy = s[:, None] * reference[None, :, 1]
        m11, m22, g, m12, m21, qx, p2, qy, p1 = (pars[:, j, None] for j in range(9))
        den = 1. + qx * vx + qy * vy
        hx = (m11 * vx + m12 * vy) / den
        hy = (m21 * vx + m22 * vy) / den
        wx = hx + 2. * p1 * hx * hy + p2 * (3. * hx * hx + hy * hy)
        wy = hy + p1 * (hx * hx + 3. * hy * hy) + 2. * p2 * hx * hy
        r2 = wx * wx + wy * wy
        kap = kap_a[:, None] + g
        radial = 1. + kap * r2
        prediction = self.length_mm * radial[..., None] * np.stack((wx, wy), -1)
        result = {"prediction": prediction}
        if diagnostics:
            brown_xx = 1. + 2. * p1 * hy + 6. * p2 * hx
            brown_xy = 2. * p1 * hx + 2. * p2 * hy
            brown_yy = 1. + 6. * p1 * hy + 2. * p2 * hx
            result.update(denominator=den, radial_tangential=radial,
                          radial_radial=1. + 3. * kap * r2,
                          brown_determinant=brown_xx * brown_yy - brown_xy * brown_xy)
        if not (theta_jac or state_jac):
            return result

        # Chain derivatives of radial(Brown(projective(v))).
        rxx = self.length_mm * (radial + 2. * kap * wx * wx)
        rxy = self.length_mm * 2. * kap * wx * wy
        ryy = self.length_mm * (radial + 2. * kap * wy * wy)
        bxx = 1. + 2. * p1 * hy + 6. * p2 * hx
        bxy = 2. * p1 * hx + 2. * p2 * hy
        byy = 1. + 6. * p1 * hy + 2. * p2 * hx
        jxx = rxx * bxx + rxy * bxy
        jxy = rxx * bxy + rxy * byy
        jyx = rxy * bxx + ryy * bxy
        jyy = rxy * bxy + ryy * byy
        local = np.zeros(prediction.shape + (9,))

        def assign(index, dx, dy):
            local[..., 0, index] = dx
            local[..., 1, index] = dy

        assign(0, jxx * vx / den, jyx * vx / den)
        assign(1, jxy * vy / den, jyy * vy / den)
        assign(2, self.length_mm * r2 * wx, self.length_mm * r2 * wy)
        assign(3, jxx * vy / den, jyx * vy / den)
        assign(4, jxy * vx / den, jyy * vx / den)
        projected_x = jxx * hx + jxy * hy
        projected_y = jyx * hx + jyy * hy
        assign(5, -projected_x * vx / den, -projected_y * vx / den)
        px, py = 3. * hx * hx + hy * hy, 2. * hx * hy
        assign(6, rxx * px + rxy * py, rxy * px + ryy * py)
        assign(7, -projected_x * vy / den, -projected_y * vy / den)
        px, py = 2. * hx * hy, hx * hx + 3. * hy * hy
        assign(8, rxx * px + rxy * py, rxy * px + ryy * py)
        if theta_jac:
            result["theta_jacobian"] = local[..., PARAM_FOR_COEFF] * basis[:, None, None, :]
        if state_jac:
            state = np.zeros(prediction.shape + (3,))
            # Accommodation acts through scale before the projective map and
            # through the fitted relative radial coefficient at the end.
            hvxx, hvxy = (m11 - hx * qx) / den, (m12 - hx * qy) / den
            hvyx, hvyy = (m21 - hy * qx) / den, (m22 - hy * qy) / den
            dvx = self.calibration[0] * reference[None, :, 0] / 4.
            dvy = self.calibration[0] * reference[None, :, 1] / 4.
            dhx, dhy = hvxx * dvx + hvxy * dvy, hvyx * dvx + hvyy * dvy
            dk = sum(j * self.calibration[j] * a**(j - 1) for j in range(1, 5)) / 4.
            state[..., 0, 0] = jxx * dhx + jxy * dhy + local[..., 0, 2] * dk[:, None]
            state[..., 1, 0] = jyx * dhx + jyy * dhy + local[..., 1, 2] * dk[:, None]
            state[..., 1:] = np.einsum("nfcp,npd->nfcd", local, pars_d, optimize=True)
            result["state_jacobian"] = state
        return result

    def save(self, path, **metadata):
        np.savez_compressed(path, reference_mm=self.reference_mm, calibration=self.calibration,
                            coefficients=self.coefficients, length_mm=np.array(self.length_mm),
                            coefficient_labels=np.array(LABELS), variant=np.array(self.variant),
                            fixed_c0_mm2=np.array(C0_MM2 if self.variant == "original" else 0.), **metadata)

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            return cls(data["reference_mm"], data["calibration"], data["coefficients"],
                       float(data["length_mm"]), str(data["variant"]))


def calibration_prediction(calibration, reference, accommodation, with_jac=False, base_kappa=0.):
    a = np.asarray(accommodation) / 4.
    s = 1. + calibration[0] * a
    kap = base_kappa + sum(calibration[j] * a**j for j in range(1, 5))
    b = reference / L_MM
    b2 = np.sum(b * b, axis=-1)
    factor = s[:, None] * (1. + kap[:, None] * s[:, None]**2 * b2[None, :])
    pred = L_MM * factor[..., None] * b[None, :, :]
    if not with_jac:
        return pred
    jac = np.empty(pred.shape + (5,))
    ds = L_MM * (1. + 3. * kap[:, None] * s[:, None]**2 * b2) * a[:, None]
    jac[..., 0] = ds[..., None] * b
    for j in range(1, 5):
        factor_j = L_MM * s[:, None]**3 * b2 * a[:, None]**j
        jac[..., j] = factor_j[..., None] * b
    return pred, jac


def fit_calibration(model, A, target, train_accommodation):
    av = A[train_accommodation]
    observed = target[train_accommodation][:, OUTER]
    ref = model.input_reference_mm[OUTER]
    fun = lambda c: ((calibration_prediction(c, ref, av, base_kappa=model.base_kappa) - observed) * 1000.).ravel()
    jac = lambda c: (calibration_prediction(c, ref, av, True, model.base_kappa)[1] * 1000.).reshape(-1, 5)
    fit = least_squares(fun, [-.08, 0., 0., 0., 0.], jac=jac, x_scale="jac",
                        ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=300)
    if not fit.success:
        raise RuntimeError(f"On-axis calibration failed: {fit.message}")
    return fit


class GazeObjective:
    def __init__(self, model, states, observed):
        self.model, self.states, self.observed = model, states, observed
        self.last = self.value = self.derivative = None

    def update(self, theta):
        if self.last is None or not np.array_equal(theta, self.last):
            self.model.coefficients = np.asarray(theta).copy()
            result = self.model.evaluate(*self.states.T, theta_jac=True)
            self.value = ((result["prediction"][:, OUTER] - self.observed[:, OUTER]) * 1000.).ravel()
            self.derivative = (result["theta_jacobian"][:, OUTER] * 1000.).reshape(-1, 33)
            self.last = np.asarray(theta).copy()

    def fun(self, theta):
        self.update(theta)
        return self.value

    def jac(self, theta):
        self.update(theta)
        return self.derivative


def partitions(shape):
    ia, ib, id_ = np.indices(shape)
    seen_a = ia % 2 == 0
    seen_gaze = (ib % 2 == 0) & (id_ % 2 == 0)
    return {"train": seen_a & seen_gaze,
            "unseen_gaze": seen_a & ~seen_gaze,
            "unseen_accommodation": ~seen_a & seen_gaze,
            "unseen_both": ~seen_a & ~seen_gaze,
            "heldout_all": ~(seen_a & seen_gaze), "all": np.ones(shape, bool)}


def summarize_scalar(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {name: None for name in ("bias", "rmse", "mae", "median_abs", "p95_abs", "p99_abs", "max_abs")}
    return {"bias": float(np.mean(values)), "rmse": float(np.sqrt(np.mean(values**2))),
            "mae": float(np.mean(np.abs(values))), "median_abs": float(np.median(np.abs(values))),
            "p95_abs": float(np.percentile(np.abs(values), 95)),
            "p99_abs": float(np.percentile(np.abs(values), 99)), "max_abs": float(np.max(np.abs(values)))}


def point_metrics(error_mm):
    components = np.asarray(error_mm) * 1000.
    norms = np.linalg.norm(components, axis=-1)
    return {"states": int(len(norms)), "vertex_rms_um": float(np.sqrt(np.mean(norms**2))),
            "x_rmse_um": float(np.sqrt(np.mean(components[..., 0]**2))),
            "y_rmse_um": float(np.sqrt(np.mean(components[..., 1]**2))),
            "median_point_um": float(np.median(norms)), "p95_point_um": float(np.percentile(norms, 95)),
            "p99_point_um": float(np.percentile(norms, 99)), "max_point_um": float(np.max(norms))}


def evaluate_forward(model, states, observed, chunk=8192):
    prediction = np.empty_like(observed)
    minimum = {"denominator": np.inf, "radial_tangential": np.inf,
               "radial_radial": np.inf, "brown_determinant": np.inf}
    for start in range(0, len(states), chunk):
        stop = min(start + chunk, len(states))
        result = model.evaluate(*states[start:stop].T, diagnostics=True)
        prediction[start:stop] = result["prediction"]
        for name in minimum:
            minimum[name] = min(minimum[name], float(np.min(result[name])))
    return prediction, minimum


class ObservationInverse:
    """Observation-only dictionary initialization followed by bounded batched LM."""
    def __init__(self, model, seeds=3, max_iterations=60, workers=6):
        self.model, self.nearest_seeds = model, seeds
        self.seeds, self.max_iterations, self.workers = seeds + 1, max_iterations, workers
        # This grid is fixed independently of every measured/true test state.
        aa, dd, bb = np.meshgrid(np.linspace(0., 4., 17), np.linspace(-20., 20., 21),
                                 np.linspace(-20., 20., 21), indexing="ij")
        self.dictionary = np.stack((aa.ravel(), dd.ravel(), bb.ravel()), -1)
        prediction = model.evaluate(*self.dictionary.T)["prediction"][:, OUTER].reshape(-1, 16)
        self.tree = cKDTree(prediction)

    def solve(self, observations):
        """Input contains centered measured vertices only; no state labels."""
        observed = np.asarray(observations)[:, OUTER].reshape(-1, 16)
        _, index = self.tree.query(observed, k=self.nearest_seeds, workers=self.workers)
        index = np.asarray(index).reshape(len(observed), self.nearest_seeds)
        nearest = self.dictionary[index] / STATE_SCALE
        initial = np.concatenate((nearest, np.broadcast_to([.5, 0., 0.], (len(observed), 1, 3))), axis=1)
        z = initial.reshape(-1, 3).copy()
        target = np.repeat(observed, self.seeds, axis=0)
        n = len(z)
        damping = np.full(n, 1e-3)
        active = np.ones(n, bool)
        converged = np.zeros(n, bool)
        iteration_count = np.zeros(n, np.int16)
        for iteration in range(self.max_iterations):
            ids = np.flatnonzero(active)
            if not len(ids):
                break
            current = z[ids] * STATE_SCALE
            result = self.model.evaluate(*current.T, state_jac=True)
            residual = result["prediction"][:, OUTER].reshape(-1, 16) - target[ids]
            jac = result["state_jacobian"][:, OUTER].reshape(-1, 16, 3) * STATE_SCALE
            hess = np.einsum("nfi,nfj->nij", jac, jac, optimize=True)
            gradient = np.einsum("nfi,nf->ni", jac, residual, optimize=True)
            diag = np.maximum(np.diagonal(hess, axis1=1, axis2=2), 1e-16)
            projected = gradient.copy()
            projected[(z[ids] <= LOWER + 1e-10) & (gradient > 0)] = 0.
            projected[(z[ids] >= UPPER - 1e-10) & (gradient < 0)] = 0.
            stationary = np.max(np.abs(projected) / np.sqrt(diag), axis=1) < 1e-10
            if np.any(stationary):
                converged[ids[stationary]] = True
                active[ids[stationary]] = False
            remaining = ~stationary
            if not np.any(remaining):
                continue
            ids, hess, gradient, diag = ids[remaining], hess[remaining], gradient[remaining], diag[remaining]
            residual, jac = residual[remaining], jac[remaining]
            blocked = (((z[ids] <= LOWER + 1e-10) & (gradient > 0)) |
                       ((z[ids] >= UPPER - 1e-10) & (gradient < 0)))
            for parameter in range(3):
                rows = np.flatnonzero(blocked[:, parameter])
                hess[rows, parameter, :] = 0.
                hess[rows, :, parameter] = 0.
                hess[rows, parameter, parameter] = 1.
                gradient[rows, parameter] = 0.
            damped = hess.copy()
            damped[:, np.arange(3), np.arange(3)] += damping[ids, None] * diag + 1e-16
            step = np.linalg.solve(damped, -gradient[..., None])[..., 0]
            step /= np.maximum(1., np.max(np.abs(step), axis=1) / .25)[:, None]
            trial = np.clip(z[ids] + step, LOWER, UPPER)
            trial_pred = self.model.evaluate(*(trial * STATE_SCALE).T)["prediction"][:, OUTER].reshape(-1, 16)
            trial_cost = np.sum((trial_pred - target[ids])**2, axis=1)
            cost = np.sum(residual**2, axis=1)
            accept = np.isfinite(trial_cost) & (trial_cost < cost)
            actual_step = np.max(np.abs(trial - z[ids]), axis=1)
            z[ids[accept]] = trial[accept]
            damping[ids[accept]] = np.maximum(damping[ids[accept]] / 3., 1e-12)
            damping[ids[~accept]] = np.minimum(damping[ids[~accept]] * 10., 1e12)
            iteration_count[ids] = iteration + 1
            small = (accept & (actual_step < 1e-9) &
                     (np.max(np.abs(gradient) / np.sqrt(diag), axis=1) < 1e-8))
            converged[ids[small]] = True
            active[ids[small]] = False

        final = self.model.evaluate(*(z * STATE_SCALE).T, state_jac=True, diagnostics=True)
        residual = final["prediction"][:, OUTER].reshape(-1, 16) - target
        all_jac = final["state_jacobian"][:, OUTER].reshape(-1, 16, 3) * STATE_SCALE
        all_gradient = np.einsum("nfi,nf->ni", all_jac, residual, optimize=True)
        all_diag = np.maximum(np.sum(all_jac**2, axis=1), 1e-16)
        all_gradient[(z <= LOWER + 1e-10) & (all_gradient > 0)] = 0.
        all_gradient[(z >= UPPER - 1e-10) & (all_gradient < 0)] = 0.
        projected_gradient = np.max(np.abs(all_gradient) / np.sqrt(all_diag), axis=1)
        converged |= projected_gradient < 1e-10
        costs = np.sum(residual**2, axis=1)
        costs = np.where(np.isfinite(costs), costs, np.inf).reshape(len(observed), self.seeds)
        choice = np.argmin(costs, axis=1)
        best = np.arange(len(observed)) * self.seeds + choice
        jac = final["state_jacobian"][:, OUTER].reshape(-1, 16, 3)[best]
        singular = np.linalg.svd(jac * STATE_SCALE, compute_uv=False)
        hess = np.einsum("nfi,nfj->nij", jac, jac, optimize=True)
        # Optional local diagnostic: center assumed exact, with independent
        # 1 micrometre noise on already-centered coordinates. This is not the
        # covariance of independent noise on all nine raw measured vertices.
        covariance = np.linalg.pinv(hess, rcond=1e-14) * 1e-6
        sigma = np.sqrt(np.maximum(np.diagonal(covariance, axis1=1, axis2=2), 0.))
        correlation = covariance / np.maximum(sigma[:, :, None] * sigma[:, None, :], 1e-30)
        best_normalized = z[best]
        all_z = z.reshape(len(observed), self.seeds, 3)
        separation = np.max(np.abs(all_z - best_normalized[:, None, :]), axis=2)
        distinct = (separation > 1e-3) & converged.reshape(len(observed), self.seeds)
        alternate_cost = np.min(np.where(distinct, costs, np.inf), axis=1)
        return {"estimated_state": best_normalized * STATE_SCALE,
                "fit_vertex_rms_um": np.sqrt(costs[np.arange(len(observed)), choice] / 8.) * 1000.,
                "converged": converged[best], "iterations": iteration_count[best],
                "initial_state": initial.reshape(-1, 3)[best] * STATE_SCALE,
                "normalized_singular_values": singular, "local_sigma_1um": sigma,
                "normalized_jacobian_rank": np.sum(singular > singular[:, :1] * 1e-10, axis=1),
                "parameter_correlations_A_ADE_A_BDE_ADE_BDE": correlation[:, [0, 0, 1], [1, 2, 2]],
                "projected_gradient_mm": projected_gradient[best],
                "valid_map": np.all(np.stack([final[name] > 0 for name in
                    ("denominator", "radial_tangential", "radial_radial", "brown_determinant")]), axis=(0, 2))[best],
                "distinct_alternative_sse_gap_mm2": alternate_cost - costs[np.arange(len(observed)), choice],
                "all_seed_costs_mm2": costs,
                "all_seed_states": all_z * STATE_SCALE,
                "all_seed_initial_states": initial * STATE_SCALE,
                "all_seed_converged": converged.reshape(len(observed), self.seeds),
                "all_seed_iterations": iteration_count.reshape(len(observed), self.seeds),
                "maximum_seed_normalized_parameter_separation": np.max(separation, axis=1),
                "boundary": (best_normalized <= LOWER + 1e-7) | (best_normalized >= UPPER - 1e-7)}


def run_inverse(inverter, observations, chunk, progress=True):
    results = {}
    started = time.perf_counter()
    for start in range(0, len(observations), chunk):
        stop = min(start + chunk, len(observations))
        part = inverter.solve(observations[start:stop])
        for key, value in part.items():
            if key not in results:
                results[key] = np.empty((len(observations),) + value.shape[1:], dtype=value.dtype)
            results[key][start:stop] = value
        if progress:
            elapsed = time.perf_counter() - started
            log(f"inverse {stop:,}/{len(observations):,}; {elapsed:.1f}s elapsed; projected total {elapsed * len(observations) / stop:.1f}s")
    return results


def inverse_metrics(result, truth, masks):
    error = result["estimated_state"] - truth
    metrics = {}
    for name, mask in masks.items():
        take = np.asarray(mask).ravel()
        if not np.any(take):
            continue
        e = error[take]
        joint = np.linalg.norm(e[:, 1:], axis=1)
        finite = np.all(np.isfinite(e), axis=1)
        converged = result["converged"][take] & finite
        metrics[name] = {"states": int(np.sum(take)),
                         "finite_estimates": int(np.sum(finite)), "nonfinite": int(np.sum(~finite)),
                         "accommodation_d": summarize_scalar(e[:, 0]),
                         "ADE_deg": summarize_scalar(e[:, 1]), "BDE_deg": summarize_scalar(e[:, 2]),
                         "joint_gaze_parameter_rmse_deg": float(np.sqrt(np.mean(joint**2))),
                         "joint_gaze_parameter_p95_deg": float(np.percentile(joint, 95)),
                         "joint_gaze_parameter_max_deg": float(np.max(joint)),
                         "fraction_A_within_0_1D_all_attempted": float(np.sum(finite & (np.abs(e[:, 0]) <= .1)) / len(e)),
                         "nonconverged": int(np.sum(~result["converged"][take])),
                         "invalid_map": int(np.sum(~result["valid_map"][take])),
                         "rank_deficient": int(np.sum(result["normalized_jacobian_rank"][take] < 3)),
                         "boundary_A": int(np.sum(result["boundary"][take, 0])),
                         "boundary_ADE": int(np.sum(result["boundary"][take, 1])),
                         "boundary_BDE": int(np.sum(result["boundary"][take, 2])),
                         "fit_vertex_rms_um": float(np.sqrt(np.mean(result["fit_vertex_rms_um"][take]**2))),
                         "converged_only": {"states": int(np.sum(converged)),
                             "accommodation_d": summarize_scalar(e[converged, 0]),
                             "ADE_deg": summarize_scalar(e[converged, 1]),
                             "BDE_deg": summarize_scalar(e[converged, 2])}}
    return metrics, error


def inverse_sampling(shape):
    """Frozen broad sample and 512-state pilot; labels only define coverage."""
    ia, ib, id_ = np.indices(shape)
    flat_a, flat_b, flat_d = ia.ravel(), ib.ravel(), id_.ravel()
    gaze_seen = (flat_b % 2 == 0) & (flat_d % 2 == 0)
    gaze_boundary = ((flat_b == 0) | (flat_b == shape[1] - 1) |
                     (flat_d == 0) | (flat_d == shape[2] - 1))
    random = np.random.default_rng(20261107)
    pools, sample = [], []
    for ai in range(shape[0]):
        gaze_strata = ((False, 1400),) if ai % 2 == 0 else ((True, 700), (False, 700))
        for seen, count in gaze_strata:
            pool = np.flatnonzero((flat_a == ai) & (gaze_seen == seen))
            pools.append(pool)
            edge = pool[gaze_boundary[pool]]
            interior = pool[~gaze_boundary[pool]]
            assert len(edge) <= count <= len(pool)
            sample.append(np.concatenate((edge, random.choice(interior, count - len(edge), replace=False))))
    fixed_sample = np.sort(np.concatenate(sample))
    assert len(fixed_sample) == 29400 and len(np.unique(fixed_sample)) == 29400
    # Sixteen states per A/gaze stratum, with a seventeenth in the first sixteen
    # strata: 31*16+16=512. Four deterministic boundary picks per stratum are
    # included, followed by independent seeded interior picks.
    pilot = []
    for i, pool in enumerate(pools):
        count = 16 + int(i < 16)
        edge = pool[gaze_boundary[pool]]
        interior = pool[~gaze_boundary[pool]]
        chosen_edge = edge[np.linspace(0, len(edge) - 1, 4, dtype=int)]
        pilot.append(np.concatenate((chosen_edge, random.choice(interior, count - 4, replace=False))))
    pilot = np.sort(np.concatenate(pilot))
    assert len(pilot) == 512 and len(np.unique(pilot)) == 512
    return pilot, fixed_sample


def synthetic_checks(variant="direct"):
    reference = np.stack((FIELDS[:, 0], FIELDS[:, 1]), -1) * .8
    theta = np.linspace(-.02, .02, 33)
    theta[21:24] += [.08, 0., 0.]
    theta[27:30] += [.07, 0., 0.]
    model = Model(reference, np.array([-.08, .002, -.001, .0004, -.0001]), theta, variant=variant)
    states = np.array([[1.13, -6.7, 9.3], [2.27, 12.1, -11.3], [.61, 4.9, 3.7]])
    analytic = model.evaluate(*states.T, theta_jac=True, state_jac=True)
    h = 1e-6
    worst_state = worst_theta = 0.
    for j in range(3):
        plus, minus = states.copy(), states.copy()
        plus[:, j] += h
        minus[:, j] -= h
        numerical = (model.evaluate(*plus.T)["prediction"] - model.evaluate(*minus.T)["prediction"]) / (2 * h)
        worst_state = max(worst_state, float(np.max(np.abs(numerical - analytic["state_jacobian"][..., j]))))
    for j in range(33):
        original = model.coefficients[j]
        model.coefficients[j] = original + h
        plus = model.evaluate(*states.T)["prediction"]
        model.coefficients[j] = original - h
        minus = model.evaluate(*states.T)["prediction"]
        model.coefficients[j] = original
        numerical = (plus - minus) / (2 * h)
        worst_theta = max(worst_theta, float(np.max(np.abs(numerical - analytic["theta_jacobian"][..., j]))))
    cal_ref = model.input_reference_mm
    cal_pred, cal_jac = calibration_prediction(model.calibration, cal_ref, states[:, 0], True, model.base_kappa)
    worst_cal = 0.
    for j in range(5):
        cp, cm = model.calibration.copy(), model.calibration.copy()
        cp[j] += h
        cm[j] -= h
        numerical = (calibration_prediction(cp, cal_ref, states[:, 0], base_kappa=model.base_kappa) -
                     calibration_prediction(cm, cal_ref, states[:, 0], base_kappa=model.base_kappa)) / (2 * h)
        worst_cal = max(worst_cal, float(np.max(np.abs(numerical - cal_jac[..., j]))))
    assert worst_state < 1e-8 and worst_theta < 1e-8 and worst_cal < 1e-8
    zero = model.evaluate([0.], [0.], [0.])["prediction"][0]
    assert np.max(np.abs(zero - reference)) < 1e-14
    inverse = ObservationInverse(model).solve(analytic["prediction"])
    recovery = np.max(np.abs(inverse["estimated_state"] - states), axis=0)
    assert np.all(recovery < [1e-5, 1e-4, 1e-4]), recovery
    log(f"{variant} synthetic checks passed: state Jacobian {worst_state:.3g}, coefficient Jacobian {worst_theta:.3g}, calibration Jacobian {worst_cal:.3g}; recovery {recovery}")


def json_write(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_report(path, model, metrics, timings, sha, inverse_coverage):
    variant_title = "direct-reference, no barrel removal" if model.variant == "direct" else "original dewarped-reference control"
    reference_description = ("The measured real A=0, ADE=BDE=0 reference is used directly. No inverse barrel transform is applied to reference or observations." if model.variant == "direct" else
                             f"This separately labelled control uses `u0=D_c0^-1(r0)` with fixed c0={C0_MM2:.11g} mm^-2 and retains absolute `k(A)L²` in the last radial stage. It is not the requested no-removal variant.")
    lines = [f"# P4 R1 {variant_title}: fit and joint inversion", "",
             reference_description + " Field 4 is subtracted from each measured observation to remove translation; all eight remaining real-ray vertices are retained, including states with failed paraxial probes. If noise were added before center subtraction, that operation would correlate the centered errors.", "",
             f"Source grid SHA-256: `{sha}`. Source grid: 21 accommodations × 101 BDE × 101 ADE = 214,221 states, nine fields. Units: mm, D, degrees. Inverse coverage: {inverse_coverage}.", "",
             "## Model", "",
             "Let `a=A/4`, `tx=BDE/20`, `ty=ADE/20`. For the primary direct-reference variant, `b=r0/L` and `κ_A=d1*a+d2*a²+d3*a³+d4*a⁴`. For the original-R1 control, `b=D_c0^-1(r0)/L` and `κ_A=c0*L²+d1*a+d2*a²+d3*a³+d4*a⁴`, equivalent to absolute k(A)L². In both variants `s=1+s1*a`; `v=s*b`; `h=M(t)v/(1+qx*vx+qy*vy)`; `w=Brown_p(t)(h)`; `rhat=L*(1+(κ_A+g(t))*|w|²)*w`. At A=0 and zero gaze the map equals the real reference exactly. Direct-reference κ_A is relative and is not the old absolute physical k(A).", "",
             f"L={model.length_mm:.13g} mm. Five on-axis calibration values: `{model.calibration.tolist()}`. The 33 gaze functions use the exact existing R1 symmetry basis, with no accommodation dependence in those functions. Total: 38 shared calibrated values. The five on-axis values are fitted only at even-index accommodation levels and held fixed in the gaze fit.", "",
             "## Calibration and forward validation", "",
             "Training requires an even accommodation index AND even BDE and ADE indices (28,611 states). The disjoint test strata hold out gaze only (83,600), accommodation only (26,010), or both (76,000). A=0 and zero gaze are in calibration. Forward errors below use all eight noncenter vertices and physical Euclidean point error.", "",
             "| Split | States | Vertex RMS (µm) | P95 point (µm) | Maximum point (µm) |",
             "|---|---:|---:|---:|---:|"]
    for name, row in metrics["forward"].items():
        lines.append(f"| {name} | {row['states']:,} | {row['vertex_rms_um']:.6f} | {row['p95_point_um']:.6f} | {row['max_point_um']:.6f} |")
    lines += ["", "On-axis calibration validation: `" + json.dumps(metrics["on_axis"]) + "`.", "",
              "Minimum sampled transform margins: `" + json.dumps(metrics["regularity"]) + "`.", "",
              "## Joint accommodation and gaze estimation", "",
              "The estimator receives only the eight observed center-relative real vertices. Its fixed model dictionary spans A=0…4 D in 0.25 D increments and both gaze axes −20…20° in 2° increments. The three nearest dictionary shapes and a fixed domain-center start seed bounded batched Levenberg–Marquardt fits of all three unknowns. Analytic derivatives are used; the state labels never enter the objective, initialization, or seed selection. The minimum eight-vertex SSE selects the result, including finite candidates that have not met the convergence tolerance. Bound-aware projected gradients are used. Test truth is used only for scoring after estimation. Bounds are A∈[0,4] D, ADE/BDE∈[−20,20]°.", "",
              "Signed errors are estimate minus ground truth. Unrounded estimates, signed errors, seed diagnostics, convergence flags and boundary flags are saved. Joint gaze error below is the Euclidean norm of the ADE/BDE parameter errors; no unverified 3D rotation convention is assumed.", "",
              "| Split | States | A RMSE (D) | A P95 abs (D) | A max abs (D) | Within 0.1 D | ADE RMSE (°) | BDE RMSE (°) | Joint gaze RMSE (°) | Nonconverged |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in metrics.get("inverse", {}).items():
        aa = row["accommodation_d"]
        lines.append(f"| {name} | {row['states']:,} | {aa['rmse']:.7f} | {aa['p95_abs']:.7f} | {aa['max_abs']:.7f} | {100 * row['fraction_A_within_0_1D_all_attempted']:.3f}% | {row['ADE_deg']['rmse']:.7f} | {row['BDE_deg']['rmse']:.7f} | {row['joint_gaze_parameter_rmse_deg']:.7f} | {row['nonconverged']:,} |")
    lines += ["", "Boundary and ambiguity diagnostics: `" + json.dumps(metrics.get("inverse_diagnostics", {})) + "`.", "",
              "These are noiseless stored-ray reconstruction errors, including model approximation error. They do not establish camera, calibration, or sensor-noise performance. The saved optional local `local_sigma_1um` diagnostic assumes an exact center and IID 1 µm noise on the already-centered coordinates; it does not model independent noise on all nine raw vertices.", "",
              "The 0.2 D source sampling does not establish discrimination between states separated by 0.1 D. The reported within-0.1-D fraction scores continuous unrounded estimates against the available ground truth.", "",
              "## Reproducibility", "", "Runtimes in seconds: `" + json.dumps(timings) + "`.", "",
              "Run with the project Python executable and this script. `--check-only` verifies the analytic Jacobians, direct-reference anchor, and exact synthetic three-unknown recovery. The existing gated R1 radial-merge result is preserved and is a different composition.", "",
              "## Gaze coefficients", "", "| Coefficient | Value |", "|---|---:|"]
    lines.extend(f"| {label} | {value:.14g} |" for label, value in zip(LABELS, model.coefficients))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_overview(path, axes, truth, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    A, ADE, BDE = axes
    estimated = result["estimated_state"]
    error = estimated - truth
    fig, panels = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    take = np.linspace(0, len(truth) - 1, min(len(truth), 20000), dtype=int)
    for j, (name, unit, bounds) in enumerate((("Accommodation", "D", (0, 4)),
                                              ("ADE", "deg", (-20, 20)),
                                              ("BDE", "deg", (-20, 20)))):
        ax = panels[0, j]
        ax.scatter(truth[take, j], estimated[take, j], s=1, alpha=.15, rasterized=True)
        ax.plot(bounds, bounds, "k--", lw=1)
        ax.set(xlabel=f"Ground truth ({unit})", ylabel=f"Estimate ({unit})", title=name)
    means, p95, biases = [], [], []
    for av in A:
        take_a = np.isclose(truth[:, 0], av, atol=1e-9)
        means.append(np.mean(np.abs(error[take_a, 0])))
        p95.append(np.percentile(np.abs(error[take_a, 0]), 95))
        biases.append(np.mean(error[take_a, 0]))
    ax = panels[1, 0]
    ax.plot(A, means, "o-", label="MAE")
    ax.plot(A, p95, "s-", label="P95 abs")
    ax.plot(A, biases, ".--", label="Signed bias")
    ax.axhline(.1, color="k", ls=":", lw=1)
    ax.set(xlabel="Ground-truth accommodation (D)", ylabel="A error (D)", title="Per-accommodation errors")
    ax.legend(fontsize=8)
    values = np.sort(np.abs(error[:, 0]))
    ax = panels[1, 1]
    ax.plot(values, np.arange(1, len(values) + 1) / len(values))
    ax.axvline(.1, color="k", ls="--", lw=1, label="0.1 D")
    ax.set(xlabel="Absolute A error (D)", ylabel="Cumulative fraction", title=f"Within 0.1 D: {100*np.mean(values <= .1):.2f}%")
    ax.legend(fontsize=8)
    values = np.sort(np.linalg.norm(error[:, 1:], axis=1))
    panels[1, 2].plot(values, np.arange(1, len(values) + 1) / len(values))
    panels[1, 2].set(xlabel="Joint ADE/BDE parameter error (deg)", ylabel="Cumulative fraction", title="Gaze parameter error CDF")
    fig.suptitle(f"P4 R1 {result.get('variant_label', 'joint estimation')}: {len(truth):,} attempted states")
    fig.savefig(path, dpi=160)
    plt.close(fig)

    ib = np.searchsorted(BDE, truth[:, 2])
    id_ = np.searchsorted(ADE, truth[:, 1])
    counts = np.zeros((len(BDE), len(ADE)))
    np.add.at(counts, (ib, id_), 1.)
    maps = []
    for values in (error[:, 1], error[:, 2], np.abs(error[:, 0])):
        total = np.zeros_like(counts)
        np.add.at(total, (ib, id_), values)
        maps.append(np.divide(total, counts, out=np.full_like(total, np.nan), where=counts > 0))
    fig, panels = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    titles = ("Mean signed ADE error (deg)", "Mean signed BDE error (deg)", "Mean absolute A error (D)")
    for ax, values, title in zip(panels, maps, titles):
        im = ax.imshow(values, origin="lower", extent=[ADE[0], ADE[-1], BDE[0], BDE[-1]], aspect="auto")
        ax.set(xlabel="ADE ground truth (deg)", ylabel="BDE ground truth (deg)", title=title)
        fig.colorbar(im, ax=ax)
    fig.suptitle("Averaged over attempted accommodation states; blank pixels were not sampled")
    fig.savefig(Path(path).with_name("gaze_error_maps.png"), dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--variant", choices=("direct", "original"), default="direct")
    parser.add_argument("--reuse-model", type=Path)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--skip-inverse", action="store_true")
    parser.add_argument("--inverse-chunk", type=int, default=2048)
    parser.add_argument("--inverse-seeds", type=int, default=3)
    parser.add_argument("--inverse-max-iterations", type=int, default=60)
    parser.add_argument("--inverse-limit", type=int, help="Deterministic broad subset for timing/debugging; not full coverage")
    args = parser.parse_args()
    with threadpool_limits(limits=6):
        synthetic_checks(args.variant)
        if args.check_only:
            return
        out = args.output_dir or (DEFAULT_OUT if args.variant == "direct" else ROOT / "data/p4_r1_original_control")
        if out.exists() and any(out.iterdir()) and args.reuse_model is None:
            raise FileExistsError(f"Refusing to overwrite existing experiment: {out}")
        out.mkdir(parents=True, exist_ok=True)
        timings, metrics = {}, {}
        started = time.perf_counter()
        A, ADE, BDE, centered, probe_failed, sha = load_grid()
        timings["load"] = time.perf_counter() - started
        shape = centered.shape[:3]
        aa, bb, dd = np.meshgrid(A, BDE, ADE, indexing="ij")
        states = np.stack((aa.ravel(), dd.ravel(), bb.ravel()), -1)
        observed = centered.reshape(-1, 9, 2)
        masks = partitions(shape)
        log(f"loaded {len(states):,} real-ray states; {np.sum(probe_failed):,} probe-failed states retained")
        if args.reuse_model:
            model = Model.load(args.reuse_model)
            assert model.variant == args.variant
            with np.load(args.reuse_model) as saved:
                assert str(saved["grid_sha256"]) == sha, "Reused model has different source provenance"
            assert np.array_equal(model.reference_mm, centered[0, 50, 50])
            prior_metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8")) if (out / "metrics.json").exists() else {}
            prior_timings = json.loads((out / "timings.json").read_text(encoding="utf-8")) if (out / "timings.json").exists() else {}
            timings["calibration_fit"] = prior_timings.get("calibration_fit", 0.)
            timings["gaze_fit"] = prior_timings.get("gaze_fit", 0.)
            metrics["fit_status"] = prior_metrics.get("fit_status", {"reused_model": str(args.reuse_model)})
            metrics["calibration_fit_status"] = prior_metrics.get("calibration_fit_status", {})
        else:
            reference = centered[0, 50, 50].copy()
            model = Model(reference, np.zeros(5), np.zeros(33), variant=args.variant)
            started = time.perf_counter()
            fit_cal = fit_calibration(model, A, centered[:, 50, 50], np.arange(len(A)) % 2 == 0)
            timings["calibration_fit"] = time.perf_counter() - started
            log(f"on-axis calibration fitted: {fit_cal.x.tolist()}")
            model.calibration = fit_cal.x.copy()
            metrics["calibration_fit_status"] = {"message": fit_cal.message, "nfev": fit_cal.nfev,
                "cost_um2_over_2": float(fit_cal.cost), "optimality": float(fit_cal.optimality)}
            axis_prediction = model.evaluate(A, np.zeros_like(A), np.zeros_like(A))["prediction"]
            axis_error = axis_prediction[:, OUTER] - centered[:, 50, 50][:, OUTER]
            log("on-axis train/unseen-A errors before gaze fit: " + json.dumps({
                "train": point_metrics(axis_error[np.arange(len(A)) % 2 == 0]),
                "unseen_A": point_metrics(axis_error[np.arange(len(A)) % 2 == 1])}))
            train = masks["train"].ravel()
            objective = GazeObjective(model, states[train], observed[train])
            initializer = np.zeros(33)
            initializer_source = "zeros"
            previous = ROOT / "data/p4_r1_radial_merge/step1_a0.npz"
            if previous.exists():
                with np.load(previous) as old:
                    key = "control_coefficients" if model.variant == "direct" else "coefficients"
                    if old[key].shape == (33,):
                        initializer = old[key].copy()
                        initializer_source = f"{previous.name}:{key} (A0 even/even training only)"
            model.coefficients = initializer.copy()
            started = time.perf_counter()
            model.evaluate(*states[train][:512].T, theta_jac=True)
            timings["train512_residual_jacobian_eval"] = time.perf_counter() - started
            jacobian_mb = int(np.sum(train)) * 16 * 33 * 8 / 1e6
            log(f"train512 residual/Jacobian evaluation {timings['train512_residual_jacobian_eval']:.4f}s; projected full evaluation {timings['train512_residual_jacobian_eval']*np.sum(train)/512:.3f}s; dense fit Jacobian {jacobian_mb:.1f}MB; estimated fit peak <1.2GB")
            started = time.perf_counter()
            fit = least_squares(objective.fun, initializer, jac=objective.jac, x_scale="jac",
                                ftol=1e-11, xtol=1e-11, gtol=1e-11, max_nfev=200, verbose=1)
            timings["gaze_fit"] = time.perf_counter() - started
            if not fit.success:
                raise RuntimeError(f"Gaze fit failed: {fit.message}")
            model.coefficients = fit.x.copy()
            metrics["fit_status"] = {"message": fit.message, "nfev": fit.nfev, "njev": fit.njev,
                                     "cost_um2_over_2": float(fit.cost), "optimality": float(fit.optimality),
                                     "initializer": initializer_source}
            del objective
            log(f"gaze fit completed in {timings['gaze_fit']:.1f}s, nfev={fit.nfev}")
        metrics["provenance"] = {"grid_sha256": sha, "source_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "variant": model.variant, "parameter_count": 38, "training_states": int(np.sum(masks["train"])),
            "center_handling": "subtract measured field4 independently in each observation; no state labels used"}
        model.save(out / "model.npz", grid_sha256=np.array(sha), A=A, ADE=ADE, BDE=BDE,
                   model_variant=np.array("R1-direct-reference-no-barrel-removal" if model.variant == "direct" else "R1-original-dewarped-reference-control"),
                   parameter_count=np.array(38), training_rule=np.array("even_A_and_even_BDE_and_even_ADE"))
        started = time.perf_counter()
        prediction, regularity = evaluate_forward(model, states, observed)
        timings["forward_evaluation"] = time.perf_counter() - started
        forward_error = prediction[:, OUTER] - observed[:, OUTER]
        metrics["forward"] = {name: point_metrics(forward_error[mask.ravel()]) for name, mask in masks.items()}
        metrics["forward_all9"] = {name: point_metrics((prediction - observed)[mask.ravel()]) for name, mask in masks.items()}
        metrics["forward_by_A"] = {f"{av:.1f}": point_metrics(forward_error[states[:, 0] == av]) for av in A}
        gaze_magnitude = np.max(np.abs(states[:, 1:]), axis=1)
        metrics["forward_by_gaze_bin"] = {f"{lo:g}_{hi:g}_max_abs_deg": point_metrics(forward_error[
            (gaze_magnitude > lo if lo else gaze_magnitude >= lo) & (gaze_magnitude <= hi)])
            for lo, hi in ((0, 5), (5, 10), (10, 15), (15, 20))}
        metrics["regularity"] = regularity
        axis_prediction = model.evaluate(A, np.zeros_like(A), np.zeros_like(A))["prediction"]
        on_axis_error = axis_prediction[:, OUTER] - centered[:, 50, 50][:, OUTER]
        metrics["on_axis"] = {"train": point_metrics(on_axis_error[np.arange(len(A)) % 2 == 0]),
                               "unseen_accommodation": point_metrics(on_axis_error[np.arange(len(A)) % 2 == 1])}
        np.savez_compressed(out / "forward.npz", A=A, ADE=ADE, BDE=BDE, fields=FIELDS,
                            prediction_mm=prediction.reshape(centered.shape), target_centered_mm=centered,
                            probe_failed_mask=probe_failed,
                            **{f"{name}_mask": mask for name, mask in masks.items()})
        del prediction, forward_error
        json_write(out / "metrics.json", metrics)
        write_report(out / "report.md", model, metrics, timings, sha, "not run yet")
        if args.skip_inverse:
            log(f"forward artifacts written to {out}")
            return
        inverter = ObservationInverse(model, args.inverse_seeds, args.inverse_max_iterations)
        pilot_selection, fixed_sample = inverse_sampling(shape)
        started = time.perf_counter()
        pilot = run_inverse(inverter, observed[pilot_selection], min(args.inverse_chunk, 512), progress=False)
        timings["inverse_pilot_512"] = time.perf_counter() - started
        projected_all = timings["inverse_pilot_512"] * len(states) / 512.
        projected_heldout = timings["inverse_pilot_512"] * int(np.sum(masks["heldout_all"])) / 512.
        metrics["pilot"] = {"states": 512, "runtime_seconds": timings["inverse_pilot_512"],
                            "projected_all_seconds": projected_all, "projected_heldout_seconds": projected_heldout,
                            "nonconverged": int(np.sum(~pilot["converged"])),
                            "estimated_peak_solver_array_MB_per_2048_chunk": 100.}
        np.savez_compressed(out / "pilot.npz", selection_indices=pilot_selection, truth_state=states[pilot_selection], **pilot)
        json_write(out / "metrics.json", metrics)
        json_write(out / "timings.json", timings)
        write_report(out / "report.md", model, metrics, timings, sha, "512-state timing pilot only")
        log(f"512-state pilot {timings['inverse_pilot_512']:.2f}s; projected full {projected_all:.1f}s / heldout {projected_heldout:.1f}s; nonconverged {metrics['pilot']['nonconverged']}")
        if args.pilot_only:
            return
        if projected_all <= 900.:
            selection = np.arange(len(states))
            sampling_description = "all 214221 source states"
        elif projected_heldout <= 900.:
            selection = np.flatnonzero(masks["heldout_all"].ravel())
            sampling_description = "all 185610 heldout source states"
        else:
            selection = fixed_sample
            sampling_description = "fixed 29400 heldout states; seed20261107; 1400/seenA,700/stratum/unseenA; all gaze boundaries before random interior"
        if args.inverse_limit and args.inverse_limit < len(states):
            selection = np.linspace(0, len(states) - 1, args.inverse_limit, dtype=int)
            sampling_description = "explicit --inverse-limit deterministic broad timing/debug subset"
        log(f"selected inverse coverage: {sampling_description}")
        started = time.perf_counter()
        result = run_inverse(inverter, observed[selection], args.inverse_chunk)
        timings["joint_inverse"] = time.perf_counter() - started
        score_masks = {name: mask.ravel()[selection] for name, mask in masks.items()}
        score_masks["probe_failed"] = probe_failed.ravel()[selection]
        boundary_truth = np.any(np.isclose(states[selection], [0., -20., -20.], atol=1e-9) |
                                np.isclose(states[selection], [4., 20., 20.], atol=1e-9), axis=1)
        score_masks["interior_truth"] = ~boundary_truth
        score_masks["boundary_truth"] = boundary_truth
        score_masks["heldout_interior"] = score_masks["heldout_all"] & ~boundary_truth
        score_masks["heldout_boundary"] = score_masks["heldout_all"] & boundary_truth
        metrics["inverse"], error = inverse_metrics(result, states[selection], score_masks)
        estimates = result["estimated_state"]
        interior = (states[selection] > [0., -20., -20.]) & (states[selection] < [4., 20., 20.])
        finite_gap = np.isfinite(result["distinct_alternative_sse_gap_mm2"])
        metrics["inverse_diagnostics"] = {
            "coverage_states": int(len(selection)), "full_grid": bool(len(selection) == len(states)),
            "sampling_description": sampling_description,
            "probe_failed_states_retained": int(np.sum(probe_failed.ravel()[selection])),
            "interior_truth_boundary_estimates_by_parameter": np.sum(interior & result["boundary"], axis=0).tolist(),
            "distinct_multistart_solutions": int(np.sum(finite_gap)),
            "minimum_distinct_alternative_sse_gap_mm2": float(np.min(result["distinct_alternative_sse_gap_mm2"][finite_gap])) if np.any(finite_gap) else None,
            "median_normalized_jacobian_condition": float(np.median(result["normalized_singular_values"][:, 0] / result["normalized_singular_values"][:, -1])),
            "maximum_normalized_jacobian_condition": float(np.max(result["normalized_singular_values"][:, 0] / result["normalized_singular_values"][:, -1])),
            "rank_deficient": int(np.sum(result["normalized_jacobian_rank"] < 3)),
            "median_parameter_correlations_A_ADE_A_BDE_ADE_BDE": np.median(result["parameter_correlations_A_ADE_A_BDE_ADE_BDE"], axis=0).tolist(),
            "p95_abs_parameter_correlations_A_ADE_A_BDE_ADE_BDE": np.percentile(np.abs(result["parameter_correlations_A_ADE_A_BDE_ADE_BDE"]), 95, axis=0).tolist(),
            "local_sigma_assumption": "exact center; IID 1um noise on already-centered coordinates; local linearization only",
            "median_local_sigma_1um_center_exact_A_ADE_BDE": np.median(result["local_sigma_1um"], axis=0).tolist(),
            "p95_local_sigma_1um_center_exact_A_ADE_BDE": np.percentile(result["local_sigma_1um"], 95, axis=0).tolist(),
        }
        np.savez_compressed(out / "inverse.npz", A=A, ADE=ADE, BDE=BDE, selection_indices=selection,
                            truth_state=states[selection], signed_error=error,
                            initialization=np.array("fixed dictionary cKDTree; observed coordinates only"), **result)
        plot_overview(out / "overview.png", (A, ADE, BDE), states[selection], result)
        metrics["per_accommodation"] = {}
        for av in A:
            take = states[selection, 0] == av
            if np.any(take):
                metrics["per_accommodation"][f"{av:.1f}"] = {
                    "states": int(np.sum(take)), "A_d": summarize_scalar(error[take, 0]),
                    "ADE_deg": summarize_scalar(error[take, 1]), "BDE_deg": summarize_scalar(error[take, 2]),
                    "fraction_A_within_0_1D_all_attempted": float(np.mean(np.abs(error[take, 0]) <= .1))}
        with (out / "per_accommodation.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["A_D", "states", "A_bias_D", "A_MAE_D", "A_RMSE_D", "A_p95_abs_D", "A_max_abs_D", "A_within_0.1D_fraction", "ADE_RMSE_deg", "BDE_RMSE_deg"])
            for av, row in metrics["per_accommodation"].items():
                q = row["A_d"]
                writer.writerow([av, row["states"], q["bias"], q["mae"], q["rmse"], q["p95_abs"], q["max_abs"], row["fraction_A_within_0_1D_all_attempted"], row["ADE_deg"]["rmse"], row["BDE_deg"]["rmse"]])
        with (out / "inverse_summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["split", "states", "A_bias_D", "A_MAE_D", "A_RMSE_D", "A_p95_abs_D", "A_max_abs_D", "within_0.1D_fraction", "ADE_RMSE_deg", "BDE_RMSE_deg", "joint_gaze_parameter_RMSE_deg", "nonconverged", "invalid_map"])
            for name, row in metrics["inverse"].items():
                q = row["accommodation_d"]
                writer.writerow([name, row["states"], q["bias"], q["mae"], q["rmse"], q["p95_abs"], q["max_abs"], row["fraction_A_within_0_1D_all_attempted"], row["ADE_deg"]["rmse"], row["BDE_deg"]["rmse"], row["joint_gaze_parameter_rmse_deg"], row["nonconverged"], row["invalid_map"]])
        metrics["worst_accommodation_states"] = [
            {"truth_A_ADE_BDE": states[selection[i]].tolist(), "estimate_A_ADE_BDE": estimates[i].tolist(),
             "signed_error_A_ADE_BDE": error[i].tolist(), "fit_vertex_rms_um": float(result["fit_vertex_rms_um"][i]),
             "converged": bool(result["converged"][i]), "boundary": result["boundary"][i].tolist()}
            for i in np.argsort(np.abs(error[:, 0]))[-20:][::-1]]
        json_write(out / "metrics.json", metrics)
        json_write(out / "timings.json", timings)
        coverage = f"{len(selection):,} / {len(states):,} states; {sampling_description}"
        write_report(out / "report.md", model, metrics, timings, sha, coverage)
        log(f"completed; report {out / 'report.md'}")


if __name__ == "__main__":
    main()
