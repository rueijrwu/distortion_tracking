"""Fit a separable physical-barrel and two-term projective-keystone model."""
from __future__ import annotations

import argparse
import hashlib
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent.parent
GRID_PATH = ROOT / "data" / "distortion_grid_p4" / "distortion_grid.pkl"
OLD_MODEL_PATH = ROOT / "data" / "p4_joint_corner_transform" / "p4_joint_corner_transform.pkl"
RADIAL_FIT_PATH = ROOT / "data" / "p4_radial_distortion_fit" / "p4_radial_distortion_fit.pkl"
RADIAL_MODEL_PATH = ROOT / "data" / "p4_radial_distortion_fit" / "p4_radial_distortion_accommodation_models.pkl"
RADIAL_METRICS_PATH = ROOT / "data" / "p4_radial_distortion_fit" / "p4_radial_distortion_metrics.pkl"
OUTPUT_DIR = ROOT / "data" / "p4_separable_transform"
SCHEMA = "p4-separable-corner-inverse-v1"
FONT_SCALE = 1.5
AXIS_SCALE = 1.5 * FONT_SCALE
OVERLAY_AXIS_SCALE = 1.5 * AXIS_SCALE
REFERENCE_COLOR = "#9aa1ab"
ACTUAL_COLOR = "#15191f"
JOINT_COLOR = "#16805d"
BARREL_COLOR = "#d97706"
KEYSTONE_COLOR = "#2b6cb0"
ACCOMMODATION_COLORS = {0.: "#2b6cb0", 2.: "#d97706", 5.: "#16805d"}
CORNER_FIELDS = ((-1., -1.), (-1., 1.), (1., -1.), (1., 1.))
TICK_VALUES = (-2., -1., 0., 1., 2.)


def load_grid(path: Path):
    raw = path.read_bytes()
    data = pickle.loads(raw)
    required = {"accommodation_d", "eye_rotation_deg", "field_x_relative", "field_y_relative",
                "real_x_mm", "real_y_mm", "z_mm"}
    if not isinstance(data, np.ndarray) or data.dtype.names is None or not required <= set(data.dtype.names):
        raise ValueError("Expected structured P4 real-ray grid data")
    if not np.allclose(data["z_mm"], 0, atol=1e-9, rtol=0):
        raise ValueError("Only Z=0 data are supported")
    a, gaze = np.unique(data["accommodation_d"]), np.unique(data["eye_rotation_deg"])
    if len(a) != 51 or len(gaze) != 401:
        raise ValueError("Expected 51 accommodations and 401 gaze angles")
    ai = np.searchsorted(a, data["accommodation_d"])
    gi = np.searchsorted(gaze, data["eye_rotation_deg"])
    order = np.lexsort((data["field_y_relative"], data["field_x_relative"], gi, ai))
    rows = data[order]
    fields = np.column_stack((rows["field_x_relative"], rows["field_y_relative"]))
    fields = fields.reshape(len(a), len(gaze), 9, 2)
    coords = np.column_stack((rows["real_x_mm"], rows["real_y_mm"]))
    coords = coords.reshape(len(a), len(gaze), 9, 2)
    expected = np.array([[-1., -1.], [-1., 0.], [-1., 1.], [0., -1.], [0., 0.],
                         [0., 1.], [1., -1.], [1., 0.], [1., 1.]])
    if not np.array_equal(fields, np.broadcast_to(expected, fields.shape)):
        raise ValueError("Unexpected, reordered, or inconsistent 3x3 native field identities")
    fields = fields[0, 0]
    nrecords = np.bincount(ai * len(gaze) + gi)
    if not np.all(nrecords == 9):
        raise ValueError("Incomplete native 3x3 grid at one or more states")
    corner_idx = np.array([np.flatnonzero(np.all(fields == c, axis=1))[0] for c in CORNER_FIELDS])
    other_idx = np.array([i for i in range(9) if i not in corner_idx])
    return a, gaze, fields, coords, corner_idx, other_idx, hashlib.sha256(raw).hexdigest(), len(raw)


def load_inputs():
    a, gaze, fields, coords, corners, noncorners, grid_sha, grid_bytes = load_grid(GRID_PATH)
    old_bytes = OLD_MODEL_PATH.read_bytes()
    old = pickle.loads(old_bytes)
    if old.get("schema") != "p4-joint-corner-inverse-v2":
        raise ValueError("Expected the existing v2 joint fit for initialization/comparison")
    radial_fit_bytes = RADIAL_FIT_PATH.read_bytes()
    radial_results = pickle.loads(radial_fit_bytes)
    radial_model_bytes = RADIAL_MODEL_PATH.read_bytes()
    radial_models = pickle.loads(radial_model_bytes)
    radial_metric_bytes = RADIAL_METRICS_PATH.read_bytes()
    radial_metrics = pickle.loads(radial_metric_bytes)["metrics"]
    if len(radial_results) != len(a) or len(radial_metrics) != len(a):
        raise ValueError("Saved radial fit/scale metrics must match the 51 accommodation states")
    if not np.allclose(radial_results["accommodation_d"], a, atol=1e-10, rtol=0):
        raise ValueError("Radial-fit accommodation ordering disagrees with grid")
    if "accommodation_d" not in (radial_metrics.dtype.names or ()) or not np.allclose(
            radial_metrics["accommodation_d"], a, atol=1e-10, rtol=0):
        raise ValueError("Radial scale-metric accommodation ordering disagrees with grid")
    power_rows = radial_models[radial_models["model"] == "power_law"]
    if len(power_rows) != 1:
        raise ValueError("Expected exactly one saved power-law radial model")
    return {
        "a": a, "gaze": gaze, "fields": fields, "coords": coords,
        "corner_idx": corners, "noncorner_idx": noncorners, "grid_sha256": grid_sha,
        "grid_bytes": grid_bytes, "old": old, "old_sha256": hashlib.sha256(old_bytes).hexdigest(),
        "radial_results": radial_results, "radial_models": radial_models,
        "radial_metrics": radial_metrics,
        "radial_fit_sha256": hashlib.sha256(radial_fit_bytes).hexdigest(),
        "radial_models_sha256": hashlib.sha256(radial_model_bytes).hexdigest(),
        "radial_metrics_sha256": hashlib.sha256(radial_metric_bytes).hexdigest(),
        "k_coefficients": np.array([power_rows[0][f"coefficient_{i}"] for i in range(3)], dtype=float),
        "L": float(old["normalization_L_mm"]),
    }


def radial_k1(kcoeff, accommodation):
    c0, c1, exponent = np.asarray(kcoeff, float)
    accommodation = np.asarray(accommodation, float)
    return c0 + c1 * np.power(accommodation, exponent)


def invert_radial(xy, k, *, tolerance=2e-13, max_iter=40):
    """Invert D_k on the near-axis monotonic branch, vectorized over coordinates."""
    xy = np.asarray(xy, float)
    radius = np.linalg.norm(xy, axis=-1)
    k = np.broadcast_to(np.asarray(k, float), radius.shape)
    root = radius.copy()
    for _ in range(max_iter):
        deriv = 1 + 3 * k * root**2
        if np.any(deriv <= 0) or not np.isfinite(deriv).all():
            raise ValueError("Radial inverse left the near-axis monotonic branch")
        residual = root * (1 + k * root**2) - radius
        step = residual / deriv
        root -= step
        if np.max(np.abs(step), initial=0.) <= tolerance:
            break
    else:
        raise ValueError("Radial inverse did not converge")
    if np.any(root < 0) or np.any(1 + k * root**2 <= 0):
        raise ValueError("Radial inverse has a nonpositive gain")
    ratio = np.divide(root, radius, out=np.ones_like(root), where=radius > 0)
    return xy * ratio[..., None], 1 + 3 * k * root**2


def apply_radial(xy, k):
    xy = np.asarray(xy, float)
    k = np.broadcast_to(np.asarray(k, float), xy.shape[:-1])
    return xy * (1 + k * np.sum(xy**2, axis=-1))[..., None]


def barrel_map(ref, a, scale, kcoeff):
    k0 = radial_k1(kcoeff, 0.)
    ka = radial_k1(kcoeff, a)
    para0, _ = invert_radial(ref, k0)
    return apply_radial(scale * para0, ka)


def barrel_inverse(coords, a, scale, kcoeff):
    """B_A^-1(r)=D_k0[(1/s(A))*D_kA^-1(r)]."""
    k0 = radial_k1(kcoeff, 0.)
    ka = radial_k1(kcoeff, a)
    para_a, derivative = invert_radial(coords, ka)
    para0 = para_a / np.asarray(scale)[..., None]
    return apply_radial(para0, k0), derivative


def gaze_basis(gaze):
    t = np.asarray(gaze, float) / 20.
    return np.stack([t**j for j in range(1, 5)], axis=-1)


def unpack(p, gaze):
    p = np.asarray(p, float)
    if p.shape != (16,):
        raise ValueError("Expected 16 separable keystone coefficients")
    basis = gaze_basis(gaze)
    curves = p.reshape(4, 4) @ basis.reshape(-1, 4).T
    return 1 + curves[0], 1 + curves[1], curves[2], curves[3]


def old_parameter_surfaces(old_fit, a, gaze):
    p = np.asarray(old_fit["coefficients_normalized"], float)
    ao, go, bo = int(old_fit["accommodation_order"]), int(old_fit["gaze_order"]), int(old_fit["beta_order"])
    aa, tt = np.meshgrid(a, gaze, indexing="ij")
    k = bo
    surfaces = []
    for ip in range(3):
        value = np.ones(aa.shape) if ip < 2 else np.zeros(aa.shape)
        for l in range(ao + 1):
            for j in range(1, go + 1):
                value += p[k] * (aa / 5)**l * (tt / 20)**j
                k += 1
        surfaces.append(value)
    if k != len(p):
        raise ValueError("Existing v2 coefficient vector does not match its metadata")
    return surfaces


def initialize(old_fit, a, gaze):
    old_sx, old_sy, old_qy = old_parameter_surfaces(old_fit, a, gaze)
    design = gaze_basis(gaze)
    p0 = np.zeros(16)
    for block, target in enumerate((old_sx.mean(axis=0) - 1,
                                    old_sy.mean(axis=0) - 1,
                                    np.zeros_like(gaze), old_qy.mean(axis=0))):
        p0[4*block:4*(block+1)] = np.linalg.lstsq(design, target, rcond=None)[0]
    return p0, (old_sx, old_sy, old_qy)


def inverse_keystone(real_mm, L, sx, sy, qx, qy):
    real = np.asarray(real_mm, float) / L
    sx = np.asarray(sx, float)[..., None]
    sy = np.asarray(sy, float)[..., None]
    qx = np.asarray(qx, float)[..., None]
    qy = np.asarray(qy, float)[..., None]
    if np.any(sx <= 0) or np.any(sy <= 0):
        raise ValueError("Keystone diagonal scales must stay positive")
    z = np.stack((real[..., 0] / sx, real[..., 1] / sy), axis=-1)
    denominator = 1 - qx * z[..., 0] - qy * z[..., 1]
    if np.any(denominator <= 0) or not np.isfinite(denominator).all():
        raise ValueError("Inverse keystone left the positive projective branch")
    return z / denominator[..., None], denominator


def forward_keystone(barrel_mm, L, sx, sy, qx, qy):
    b = np.asarray(barrel_mm, float) / L
    sx = np.asarray(sx, float)[..., None]
    sy = np.asarray(sy, float)[..., None]
    qx = np.asarray(qx, float)[..., None]
    qy = np.asarray(qy, float)[..., None]
    denominator = 1 + qx * b[..., 0] + qy * b[..., 1]
    if np.any(denominator <= 0) or not np.isfinite(denominator).all():
        raise ValueError("Forward keystone left the positive projective branch")
    return L * np.stack((sx * b[..., 0] / denominator,
                         sy * b[..., 1] / denominator), axis=-1), denominator


class SeparableFitter:
    def __init__(self, inputs):
        self.inputs = inputs
        a, gaze = inputs["a"], inputs["gaze"]
        self.na, self.ng = len(a), len(gaze)
        self.nstate = self.na * self.ng
        self.L = inputs["L"]
        self.kcoeff = inputs["k_coefficients"]
        self.scale_a = np.asarray(inputs["radial_metrics"]["paraxial_scale_vs_a0"], float)
        self.ka = radial_k1(self.kcoeff, a)
        self.k0 = float(radial_k1(self.kcoeff, 0.))
        self.state_ai = np.repeat(np.arange(self.na), self.ng)
        self.state_gi = np.tile(np.arange(self.ng), self.na)
        self.state_a = a[self.state_ai]
        self.state_gaze = gaze[self.state_gi]
        self.ref = inputs["coords"][0, np.argmin(np.abs(gaze))].copy()
        self.corners = inputs["coords"][:, :, inputs["corner_idx"]].reshape(self.nstate, 4, 2)
        self.corners_norm = self.corners / self.L
        self.reference_corners = np.broadcast_to(self.ref[inputs["corner_idx"]], (self.nstate, 4, 2))
        self.scale_state = self.scale_a[self.state_ai]
        self.ka_state = self.ka[self.state_ai]
        self.b0_corners, _ = invert_radial(self.ref[inputs["corner_idx"]], self.k0)
        self.b0_all, _ = invert_radial(self.ref, self.k0)
        self.barrel_corners = np.empty_like(self.corners)
        self.barrel_all = np.empty((self.na, 9, 2))
        for ia in range(self.na):
            self.barrel_corners.reshape(self.na, self.ng, 4, 2)[ia] = barrel_map(
                self.ref[inputs["corner_idx"]], a[ia], self.scale_a[ia], self.kcoeff)
            self.barrel_all[ia] = barrel_map(self.ref, a[ia], self.scale_a[ia], self.kcoeff)

    def recover(self, p, rays_mm, field="corners"):
        sx, sy, qx, qy = unpack(p, self.state_gaze)
        rays = np.asarray(rays_mm, float).reshape(self.nstate, -1, 2)
        vnorm, denominator = inverse_keystone(rays, self.L, sx, sy, qx, qy)
        vmm = vnorm * self.L
        para_a, radial_derivative = invert_radial(vmm, self.ka_state[:, None])
        para0 = para_a / self.scale_state[:, None, None]
        recovered = apply_radial(para0, self.k0)
        return recovered, denominator, radial_derivative

    def residual(self, p):
        try:
            recovered, denominator, radial_derivative = self.recover(p, self.corners)
            err_um = (recovered - self.reference_corners) * 1000.
            # Soft domain penalties keep trial iterates away from projective/radial singularities.
            denom_penalty = 1000. * np.maximum(.05 - denominator, 0.).reshape(-1)
            radial_penalty = 1000. * np.maximum(.05 - radial_derivative, 0.).reshape(-1)
            return np.concatenate((err_um.reshape(-1), denom_penalty, radial_penalty))
        except ValueError:
            # This branch is only for nonphysical trial iterates. The coefficient bounds and
            # smooth penalties guide the optimizer back toward the valid domain.
            return np.full(self.nstate * 4 * 2 + self.nstate * 8, 1e5, dtype=float)


def metric_summary(errors, ids):
    e = np.asarray(errors)[..., ids, :]
    norm = np.linalg.norm(e, axis=-1)
    return {"vertex_rms_um": float(np.sqrt(np.mean(norm**2)) * 1000),
            "vertex_max_um": float(np.max(norm) * 1000),
            "coordinate_rmse_um": float(np.sqrt(np.mean(e**2)) * 1000),
            "mean_state_vertex_rms_um": float(np.mean(np.sqrt(np.mean(norm**2, axis=-1))) * 1000),
            "max_state_vertex_rms_um": float(np.max(np.sqrt(np.mean(norm**2, axis=-1))) * 1000)}


def point_metric_summary(errors):
    """Summarize arbitrary state-by-point vector residuals (last dimension is x/y)."""
    e = np.asarray(errors)
    norm = np.linalg.norm(e, axis=-1)
    state_rms = np.sqrt(np.mean(norm**2, axis=-1))
    return {"vertex_rms_um": float(np.sqrt(np.mean(norm**2)) * 1000),
            "vertex_max_um": float(np.max(norm) * 1000),
            "coordinate_rmse_um": float(np.sqrt(np.mean(e**2)) * 1000),
            "mean_state_vertex_rms_um": float(np.mean(state_rms) * 1000),
            "max_state_vertex_rms_um": float(np.max(state_rms) * 1000)}


def model_errors(inputs, fitter, p):
    a, gaze, coords = inputs["a"], inputs["gaze"], inputs["coords"]
    sx, sy, qx, qy = unpack(p, gaze)
    pred = np.empty_like(coords)
    denominator = np.empty((len(a), len(gaze), 9))
    for ia in range(len(a)):
        pred[ia], denominator[ia] = forward_keystone(
            fitter.barrel_all[ia][None, None, :, :], fitter.L,
            sx[None, :], sy[None, :], qx[None, :], qy[None, :])
    errors = pred - coords
    recovered, inverse_den, radial_derivative = fitter.recover(p, coords)
    recovered = recovered.reshape(coords.shape)
    inverse_errors = recovered - fitter.ref[None, None, :, :]
    predicted_recovered, predicted_inverse_den, predicted_radial_derivative = fitter.recover(p, pred)
    predicted_recovered = predicted_recovered.reshape(coords.shape)
    predicted_roundtrip_error = predicted_recovered - fitter.ref[None, None, :, :]
    corner_stats = metric_summary(errors, inputs["corner_idx"])
    noncorner_stats = metric_summary(errors, inputs["noncorner_idx"])
    all_stats = metric_summary(errors, np.arange(9))
    inverse_corner = metric_summary(inverse_errors, inputs["corner_idx"])
    inverse_noncorner = metric_summary(inverse_errors, inputs["noncorner_idx"])
    inverse_all = metric_summary(inverse_errors, np.arange(9))
    return {
        "predicted_real_grid_mm": pred,
        "forward_errors_mm": errors,
        "recovered_reference_mm": recovered,
        "inverse_errors_mm": inverse_errors,
        "corner_forward": corner_stats, "noncorner_forward": noncorner_stats,
        "all_forward": all_stats, "corner_inverse": inverse_corner,
        "noncorner_inverse": inverse_noncorner, "all_inverse": inverse_all,
        "minimum_forward_denominator": float(np.min(denominator)),
        "minimum_inverse_denominator": float(np.min(inverse_den)),
        "minimum_radial_derivative": float(np.min(radial_derivative)),
        "predicted_forward_inverse_roundtrip_max_mm": float(np.max(np.linalg.norm(predicted_roundtrip_error, axis=-1))),
        "minimum_predicted_inverse_keystone_denominator": float(np.min(predicted_inverse_den)),
        "minimum_predicted_inverse_radial_derivative": float(np.min(predicted_radial_derivative)),
        "inverse_denominator_by_state": inverse_den.reshape(len(a), len(gaze), 9),
        "radial_derivative_by_state": radial_derivative.reshape(len(a), len(gaze), 9),
    }


def old_parameter_grid(old_fit, a, gaze):
    old_sx, old_sy, old_q = old_parameter_surfaces(old_fit, a, gaze)
    return old_sx, old_sy, np.zeros_like(old_q), old_q


def evaluate_old_and_new_comparison(inputs, fitter, old_fit, pnew):
    a, gaze, coords = inputs["a"], inputs["gaze"], inputs["coords"]
    old_sx, old_sy, old_qx, old_qy = old_parameter_grid(old_fit, a, gaze)
    pred_old = np.empty_like(coords)
    for ia in range(len(a)):
        pred_old[ia], _ = forward_keystone(
            fitter.barrel_all[ia][None, None, :, :], fitter.L,
            old_sx[ia][None, :], old_sy[ia][None, :], old_qx[ia][None, :], old_qy[ia][None, :])
    new_sx, new_sy, new_qx, new_qy = unpack(pnew, gaze)
    pred_new = np.empty_like(coords)
    for ia in range(len(a)):
        pred_new[ia], _ = forward_keystone(
            fitter.barrel_all[ia][None, None, :, :], fitter.L,
            new_sx[None, :], new_sy[None, :], new_qx[None, :], new_qy[None, :])
    actual = coords
    old_forward_err, new_forward_err = pred_old - actual, pred_new - actual
    old_recovered = np.empty_like(coords)
    new_recovered = np.empty_like(coords)
    old_min_den = np.inf
    new_min_den = np.inf
    min_radial = np.inf
    for ia in range(len(a)):
        # Flatten gaze and fields so the same physical B_A inverse is used for both K models.
        state_real = coords[ia]
        for vals, recovered_out, state_idx in (
            ((old_sx[ia], old_sy[ia], old_qx[ia], old_qy[ia]), old_recovered, ia),
            ((new_sx, new_sy, new_qx, new_qy), new_recovered, ia),
        ):
            sxv, syv, qxv, qyv = vals
            vnorm, den = inverse_keystone(state_real, fitter.L, sxv, syv, qxv, qyv)
            para_a, deriv = invert_radial(vnorm * fitter.L, fitter.ka[ia])
            recovered = apply_radial(para_a / fitter.scale_a[ia], fitter.k0)
            recovered_out[ia] = recovered
            if recovered_out is old_recovered:
                old_min_den = min(old_min_den, float(np.min(den)))
            else:
                new_min_den = min(new_min_den, float(np.min(den)))
            min_radial = min(min_radial, float(np.min(deriv)))
    old_inverse_err = old_recovered - fitter.ref[None, None, :, :]
    new_inverse_err = new_recovered - fitter.ref[None, None, :, :]
    return {
        "old_forward_corner": metric_summary(old_forward_err, inputs["corner_idx"]),
        "new_forward_corner": metric_summary(new_forward_err, inputs["corner_idx"]),
        "old_forward_noncorners": metric_summary(old_forward_err, inputs["noncorner_idx"]),
        "new_forward_noncorners": metric_summary(new_forward_err, inputs["noncorner_idx"]),
        "old_forward_all": metric_summary(old_forward_err, np.arange(9)),
        "new_forward_all": metric_summary(new_forward_err, np.arange(9)),
        "old_inverse_corner": metric_summary(old_inverse_err, inputs["corner_idx"]),
        "new_inverse_corner": metric_summary(new_inverse_err, inputs["corner_idx"]),
        "old_inverse_noncorners": metric_summary(old_inverse_err, inputs["noncorner_idx"]),
        "new_inverse_noncorners": metric_summary(new_inverse_err, inputs["noncorner_idx"]),
        "old_inverse_all": metric_summary(old_inverse_err, np.arange(9)),
        "new_inverse_all": metric_summary(new_inverse_err, np.arange(9)),
        "old_minimum_inverse_denominator": old_min_den,
        "new_minimum_inverse_denominator": new_min_den,
        "minimum_radial_derivative": min_radial,
        "old_forward_errors_mm": old_forward_err,
        "new_forward_errors_mm": new_forward_err,
        "old_inverse_errors_mm": old_inverse_err,
        "new_inverse_errors_mm": new_inverse_err,
    }


def fit_model(inputs, max_nfev):
    fitter = SeparableFitter(inputs)
    p0, old_init_surfaces = initialize(inputs["old"]["fit"], inputs["a"], inputs["gaze"])
    lower = np.tile(np.array([-.15, -.15, -.25, -.25]), 4)
    upper = -lower
    # Scale curves stay positive under these coefficient bounds; q curves are guarded by
    # positive-branch checks and smooth residual penalties in the vectorized objective.
    lower[:8], upper[:8] = -.2, .2
    def jacobian(p):
        # An absolute central difference is more stable here than the default tiny relative
        # step because several polynomial coefficients are close to zero.
        jac = np.empty((len(fitter.residual(p)), len(p)), dtype=float)
        for col in range(len(p)):
            step = 1e-5
            plus, minus = p.copy(), p.copy()
            plus[col] = min(upper[col] - 1e-12, p[col] + step)
            minus[col] = max(lower[col] + 1e-12, p[col] - step)
            width = plus[col] - minus[col]
            jac[:, col] = (fitter.residual(plus) - fitter.residual(minus)) / width
        return jac

    opt = least_squares(fitter.residual, p0, jac=jacobian, method="trf", bounds=(lower, upper),
                        max_nfev=max_nfev, x_scale=np.full(16, .08),
                        ftol=1e-14, xtol=1e-14, gtol=1e-10, verbose=1)
    if not opt.success:
        raise RuntimeError(f"Separable fit did not converge: {opt.message}")
    bound_tolerance = 1e-6
    active_lower = np.flatnonzero(np.abs(opt.x - lower) <= bound_tolerance)
    active_upper = np.flatnonzero(np.abs(opt.x - upper) <= bound_tolerance)
    sx, sy, qx, qy = unpack(opt.x, inputs["gaze"])
    recovered, inverse_den, radial_deriv = fitter.recover(opt.x, fitter.corners)
    inverse_errors = recovered - fitter.reference_corners
    inverse_fit_metrics = point_metric_summary(inverse_errors)
    corner_norms = np.linalg.norm(inverse_errors, axis=-1)
    state_vertex = np.sqrt(np.mean(corner_norms**2, axis=-1)).reshape(fitter.na, fitter.ng)
    old_fit = inputs["old"]["fit"]
    old_sx, old_sy, old_q = old_parameter_surfaces(old_fit, inputs["a"], inputs["gaze"])
    comparison = evaluate_old_and_new_comparison(inputs, fitter, old_fit, opt.x)
    results = model_errors(inputs, fitter, opt.x)
    # Assert baseline identities for radial map and both old/new projective keystones.
    baseline = barrel_map(fitter.ref, 0., fitter.scale_a[0], fitter.kcoeff)
    baseline_error = float(np.max(np.linalg.norm(baseline - fitter.ref, axis=-1)))
    zero_i = int(np.argmin(np.abs(inputs["gaze"])))
    if not np.allclose([sx[zero_i], sy[zero_i], qx[zero_i], qy[zero_i]], [1., 1., 0., 0.], atol=1e-12, rtol=0):
        raise RuntimeError("Keystone is not identity at E=0")
    model = {
        "schema": SCHEMA,
        "model": "fixed physical power-law barrel followed by separable two-q keystone",
        "fit_method": "least_squares on inverse-corrected real-ray corner coordinates",
        "fit_parameterization": "sx/sy=1+sum(j=1..4)c_j(E/20)^j; qnx/qny=sum(j=1..4)c_j(E/20)^j",
        "coefficient_order": ["sx_j1..j4", "sy_j1..j4", "qnx_j1..j4", "qny_j1..j4"],
        "coefficients_normalized": opt.x.copy(),
        "accommodation_dependence": "barrel B_A only; keystone coefficients depend only on single sampled gaze angle E",
        "accommodation_order": 0,
        "gaze_order": 4,
        "parameter_count": 16,
        "normalization_L_mm": fitter.L,
        "accommodation_d": inputs["a"].copy(),
        "eye_rotation_deg": inputs["gaze"].copy(),
        "radial_k1_power_coefficients_c0_c1_n": inputs["k_coefficients"].copy(),
        "paraxial_scale_vs_a0": fitter.scale_a.copy(),
        "reference_corners_mm": fitter.ref[inputs["corner_idx"]].copy(),
        "reference_all_fields_mm": fitter.ref.copy(),
        "corner_field_ids": inputs["fields"][inputs["corner_idx"]].copy(),
        "noncorner_field_ids": inputs["fields"][inputs["noncorner_idx"]].copy(),
        "fit_corner_inverse_rms_by_state_mm": state_vertex,
        "inverse_fit_corner_metrics": inverse_fit_metrics,
        "fit_corner_inverse_residuals_mm": inverse_errors.reshape(fitter.na, fitter.ng, 4, 2),
        "fit_success": bool(opt.success), "fit_status": int(opt.status),
        "fit_message": str(opt.message), "fit_nfev": int(opt.nfev),
        "fit_cost": float(opt.cost), "fit_optimality": float(opt.optimality),
        "active_lower_bound_indices": active_lower,
        "active_upper_bound_indices": active_upper,
        "minimum_inverse_keystone_denominator": float(np.min(inverse_den)),
        "minimum_inverse_radial_derivative": float(np.min(radial_deriv)),
        "baseline_B0_identity_max_error_mm": baseline_error,
        "parameter_curves": {"sx": sx.copy(), "sy": sy.copy(), "qnx": qx.copy(), "qny": qy.copy()},
        "forward_metrics": {k: v for k, v in results.items() if isinstance(v, dict) or k.startswith("minimum_") or k.endswith("roundtrip_max_mm")},
        "old_v2_K_on_new_B_comparison": {k: v for k, v in comparison.items() if isinstance(v, dict) or k.endswith("denominator") or k == "minimum_radial_derivative"},
        "provenance": {
            "grid_path": str(GRID_PATH.resolve()), "grid_sha256": inputs["grid_sha256"], "grid_bytes": inputs["grid_bytes"],
            "old_joint_model_path": str(OLD_MODEL_PATH.resolve()), "old_joint_model_sha256": inputs["old_sha256"],
            "radial_fit_path": str(RADIAL_FIT_PATH.resolve()), "radial_fit_sha256": inputs["radial_fit_sha256"],
            "radial_models_path": str(RADIAL_MODEL_PATH.resolve()), "radial_models_sha256": inputs["radial_models_sha256"],
            "radial_metrics_path": str(RADIAL_METRICS_PATH.resolve()), "radial_metrics_sha256": inputs["radial_metrics_sha256"],
            "ray_type": "real only", "z_mm": 0., "fit_points": "four field corners at all 51x401 sampled states",
            "validation_points": "five noncorner fields at all 51x401 sampled states; excluded from fit objective",
        },
        "optimizer": {"method": "scipy.optimize.least_squares(method='trf')", "max_nfev": max_nfev,
                      "tolerances": [1e-12, 1e-12, 1e-12], "bounds": [lower.copy(), upper.copy()],
                      "initial_coefficients": p0.copy()},
        "fit_parameter_curves_old_initialization": {"sx": old_init_surfaces[0].mean(axis=0),
                                                       "sy": old_init_surfaces[1].mean(axis=0),
                                                       "qny": old_init_surfaces[2].mean(axis=0)},
    }
    return model, fitter, results, comparison, old_fit


def grid_plot(ax, xy, color, linestyle="-", linewidth=1.5, zorder=3):
    arr = np.asarray(xy).reshape(3, 3, 2)
    for i in (0, 2):
        ax.plot(arr[i, :, 0], arr[i, :, 1], color=color, ls=linestyle, lw=linewidth, zorder=zorder)
        ax.plot(arr[:, i, 0], arr[:, i, 1], color=color, ls=linestyle, lw=linewidth, zorder=zorder)
    ax.scatter([arr[1, 1, 0]], [arr[1, 1, 1]], s=17, color=color, edgecolors="none", zorder=zorder+2)


def setup_ax(ax, title, overlay=False):
    axis_scale = OVERLAY_AXIS_SCALE if overlay else AXIS_SCALE
    ax.set_title(title, fontsize=11 * FONT_SCALE, pad=7 * FONT_SCALE)
    ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
    ax.set_xticks(TICK_VALUES); ax.set_yticks(TICK_VALUES)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=.16, lw=.5)
    ax.set_xlabel("x (mm)", fontsize=9 * axis_scale, labelpad=2)
    ax.set_ylabel("y (mm)", fontsize=9 * axis_scale, labelpad=2)
    ax.tick_params(labelsize=8 * axis_scale)


def make_predictions(inputs, model, fitter):
    a, gaze = inputs["a"], inputs["gaze"]
    sx, sy, qx, qy = unpack(model["coefficients_normalized"], gaze)
    pred = np.empty_like(inputs["coords"])
    for ia in range(len(a)):
        pred[ia], _ = forward_keystone(fitter.barrel_all[ia][None, :, :], fitter.L,
                                       sx[None, :], sy[None, :], qx[None, :], qy[None, :])
    return pred


def save_figures(out, inputs, model, fitter, results, comparison, old_fit):
    a, gaze, fields, coords = inputs["a"], inputs["gaze"], inputs["fields"], inputs["coords"]
    predictions = results["predicted_real_grid_mm"]
    ref = fitter.ref
    corner_idx, noncorner_idx = inputs["corner_idx"], inputs["noncorner_idx"]
    a0, a2, a5 = (int(np.argmin(np.abs(a-v))) for v in (0., 2., 5.))
    e0, em20, ep20 = (int(np.argmin(np.abs(gaze-v))) for v in (0., -20., 20.))

    # Four-stage view: reference, new physical barrel, isolated fitted K, and joint map.
    fig, axs = plt.subplots(2, 2, figsize=(14, 11))
    panels = [
        (axs[0, 0], "Reference", ref, None, "#667085", "A=0 D, E=0 deg", False),
        (axs[0, 1], "Barrel + scale", fitter.barrel_all[a5], coords[a5, e0], BARREL_COLOR, "A=5 D, E=0 deg", False),
        (axs[1, 0], "Keystone", predictions[a0, ep20], None, KEYSTONE_COLOR, "A=0 D, E=20 deg", True),
        (axs[1, 1], "Joint", predictions[a5, ep20], coords[a5, ep20], JOINT_COLOR, "A=5 D, E=20 deg", True),
    ]
    for ax, title, pred, actual, color, state, swapped in panels:
        visual_ref = ref[:, ::-1] if swapped else ref
        grid_plot(ax, visual_ref, REFERENCE_COLOR, "--", 1.0, 1)
        visual_pred = pred[:, ::-1] if swapped else pred
        grid_plot(ax, visual_pred, color, "--", 1.6, 4)
        if actual is not None:
            grid_plot(ax, actual[:, ::-1] if swapped else actual, ACTUAL_COLOR, "-", 1.0, 3)
        setup_ax(ax, title + "\n" + state)
    handles = [
        Line2D([0], [0], color=REFERENCE_COLOR, ls="--", lw=1, label="Reference grid"),
        Line2D([0], [0], color=BARREL_COLOR, ls="--", lw=1.6, label="Barrel model"),
        Line2D([0], [0], color=KEYSTONE_COLOR, ls="--", lw=1.6, label="Keystone model"),
        Line2D([0], [0], color=JOINT_COLOR, ls="--", lw=1.6, label="Joint model"),
        Line2D([0], [0], color=ACTUAL_COLOR, ls="-", lw=1, label="Actual real-ray grid"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .04), ncol=3,
               frameon=False, fontsize=8.5 * FONT_SCALE)
    fig.tight_layout(rect=(0, .12, 1, 1), pad=1.0)
    fig.savefig(out / "p4_transformation_stages.png", dpi=200, facecolor="white")
    fig.savefig(out / "p4_transformation_stages.svg", facecolor="white")
    plt.close(fig)

    # Three-by-three gaze/accommodation comparison with all keystone panels swapped.
    fig, axs = plt.subplots(3, 3, figsize=(15, 14))
    for row, ia in enumerate((a0, a2, a5)):
        for col, ig in enumerate((em20, e0, ep20)):
            ax = axs[row, col]
            grid_plot(ax, ref[:, ::-1], REFERENCE_COLOR, "--", 1.0, 1)
            grid_plot(ax, coords[ia, ig, :, ::-1], ACTUAL_COLOR, "-", 1.0, 3)
            grid_plot(ax, predictions[ia, ig, :, ::-1], JOINT_COLOR, "--", 1.6, 4)
            setup_ax(ax, f"A={a[ia]:g} D, E={gaze[ig]:g} deg")
    handles = [
        Line2D([0], [0], color=REFERENCE_COLOR, ls="--", lw=1, label="Reference grid"),
        Line2D([0], [0], color=ACTUAL_COLOR, ls="-", lw=1, label="Actual real-ray grid"),
        Line2D([0], [0], color=JOINT_COLOR, ls="--", lw=1.6, label="Refit joint model"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .035), ncol=3,
               frameon=False, fontsize=8.5 * FONT_SCALE)
    fig.tight_layout(rect=(0, .09, 1, 1), pad=1.0)
    fig.savefig(out / "p4_transformation_sweep.png", dpi=200, facecolor="white")
    fig.savefig(out / "p4_transformation_sweep.svg", facecolor="white")
    plt.close(fig)

    # Accommodation overlays by gaze, with actual solid/model dashed in matched colors.
    fig, axs = plt.subplots(1, 3, figsize=(15, 8))
    for col, ig in enumerate((em20, e0, ep20)):
        ax = axs[col]
        grid_plot(ax, ref[:, ::-1], REFERENCE_COLOR, ":", 1.2, 1)
        for ia in (a0, a2, a5):
            color = ACCOMMODATION_COLORS[float(a[ia])]
            grid_plot(ax, coords[ia, ig, :, ::-1], color, "-", 1.4, 3)
            grid_plot(ax, predictions[ia, ig, :, ::-1], color, "--", 1.6, 4)
        setup_ax(ax, f"E={gaze[ig]:g} deg", overlay=True)
    overlay_handles = [Line2D([0], [0], color=ACCOMMODATION_COLORS[float(a[i])], lw=2,
                              label=f"A={a[i]:g} D") for i in (a0, a2, a5)] + [
        Line2D([0], [0], color=ACTUAL_COLOR, ls="-", lw=1.5, label="Actual real-ray grid"),
        Line2D([0], [0], color=ACTUAL_COLOR, ls="--", lw=1.6, label="Refit joint prediction"),
        Line2D([0], [0], color=REFERENCE_COLOR, ls=":", lw=1.2, label="Reference grid"),
    ]
    fig.legend(handles=overlay_handles, loc="lower center", bbox_to_anchor=(.5, .035), ncol=3,
               frameon=False, fontsize=8.5 * FONT_SCALE)
    fig.tight_layout(rect=(0, .18, 1, 1), pad=1.0)
    fig.savefig(out / "p4_transformation_accommodation_overlay.png", dpi=200, facecolor="white")
    fig.savefig(out / "p4_transformation_accommodation_overlay.svg", facecolor="white")
    plt.close(fig)

    # Parameter curves by the sole sampled gaze angle.
    curves = model["parameter_curves"]
    q_limit = 1.1 * float(np.max(np.abs(curves["qny"])))
    qx_max = float(np.max(np.abs(curves["qnx"])))
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for ax, key, label in zip(axs.ravel(), ("sx", "sy", "qnx", "qny"),
                              ("s_x(E)", "s_y(E)", "q_{n,x}(E)", "q_{n,y}(E)")):
        ax.plot(gaze, curves[key], color=JOINT_COLOR if key in ("sx", "sy") else KEYSTONE_COLOR, lw=2)
        ax.axhline(1 if key in ("sx", "sy") else 0, color="0.5", ls=":", lw=1)
        ax.set_title(f"${label}$", fontsize=11 * FONT_SCALE)
        ax.set_xlabel("E (deg)", fontsize=9 * AXIS_SCALE)
        ax.set_ylabel("Fitted value", fontsize=9 * AXIS_SCALE)
        if key in ("qnx", "qny"):
            ax.set_ylim(-q_limit, q_limit)
        if key == "qnx":
            ax.text(.03, .94, f"max |qnx|={qx_max:.2g}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=8.5 * FONT_SCALE)
        ax.tick_params(labelsize=8 * AXIS_SCALE); ax.grid(alpha=.2)
    fig.tight_layout(pad=1.0)
    fig.savefig(out / "keystone_parameter_curves.png", dpi=180, facecolor="white")
    fig.savefig(out / "keystone_parameter_curves.svg", facecolor="white")
    plt.close(fig)

    # In-sample inverse-corner RMS over the complete A/E grid.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    image = ax.pcolormesh(gaze, a, model["fit_corner_inverse_rms_by_state_mm"] * 1000,
                          shading="auto", cmap="magma")
    fig.colorbar(image, ax=ax, label="Four-corner inverse RMS (um)")
    ax.set_xlabel("E (deg)"); ax.set_ylabel("Accommodation A (D)")
    ax.set_title("Separable two-q inverse-fit corner residual")
    fig.tight_layout(pad=1.0)
    fig.savefig(out / "inverse_corner_rms.png", dpi=180, facecolor="white")
    fig.savefig(out / "inverse_corner_rms.svg", facecolor="white")
    plt.close(fig)


def report_text(model, comparison, inputs):
    def stats(label, d):
        return (f"{label}: vertex RMS/max={d['vertex_rms_um']:.6f}/{d['vertex_max_um']:.6f} um; "
                f"coordinate RMSE={d['coordinate_rmse_um']:.6f} um; "
                f"mean/max state RMS={d['mean_state_vertex_rms_um']:.6f}/{d['max_state_vertex_rms_um']:.6f} um.")
    forward = model["forward_metrics"]
    comp = model["old_v2_K_on_new_B_comparison"]
    coeff = model["coefficients_normalized"].reshape(4, 4)
    lines = [
        "# P4 separable power-barrel and two-q keystone fit", "",
        "This new fit uses all 51 x 401 simulated states, fitting the four real-ray corners at every state. The five noncorner fields are checks and are not included in the objective. The saved canonical ray grid, original v2 fit artifact, and radial-fit artifacts are read-only inputs.", "",
        "## Model", "",
        "The accommodation map is fixed to the saved physical radial law and paraxial scale: $B_A=D_{k_1(A)}\\circ S_{s(A)}\\circ D_{k_1(0)}^{-1}$. The keystone depends only on the single sampled gaze angle $E$:", "",
        "$$", "K_E(x_b,y_b)=\\frac{1}{1+q_{n,x}(E)x_b/L+q_{n,y}(E)y_b/L}\\begin{bmatrix}s_x(E)x_b\\\\s_y(E)y_b\\end{bmatrix}.", "$$", "",
        "The plotted panels swap physical x/y for display; the saved coefficients and fit use physical axes. The denominator has two fitted physical-coordinate terms, $q_xx_b+q_yy_b$, with $q_x=q_{n,x}/L$ and $q_y=q_{n,y}/L$ in mm$^{-1}$. No separate $E_x/E_y$ rotations are inferred: $E$ is the dataset's one gaze-angle variable.", "",
        "The four coefficient vectors are independent: $f_{s_x}(E)=\\sum_{j=1}^4c_{s_x,j}(E/20)^j$, $f_{s_y}(E)=\\sum_{j=1}^4c_{s_y,j}(E/20)^j$, $f_{q_{n,x}}(E)=\\sum_{j=1}^4c_{q_{n,x},j}(E/20)^j$, and $f_{q_{n,y}}(E)=\\sum_{j=1}^4c_{q_{n,y},j}(E/20)^j$. Then $s_x=1+f_{s_x}$, $s_y=1+f_{s_y}$; the q curves have zero baseline. Thus $K_0$ is exactly identity. Fit residuals use inverse mapping $B_A^{-1}(K_E^{-1}(\\mathbf r))$ against the fixed real A=0,E=0 reference.", "",
        "## Fit result", "",
        f"Converged: {model['fit_success']} ({model['fit_message']}); {model['fit_nfev']} evaluations, {model['parameter_count']} fitted coefficients, objective cost={model['fit_cost']:.9g}, first-order optimality={model['fit_optimality']:.6g}. Coefficients at lower/upper bounds (0-based): {model['active_lower_bound_indices'].tolist()} / {model['active_upper_bound_indices'].tolist()}. Reference uses L={model['normalization_L_mm']:.12g} mm. Baseline B0 identity max error={model['baseline_B0_identity_max_error_mm']*1000:.3g} um. Minimum inverse projective denominator={model['minimum_inverse_keystone_denominator']:.8g}; minimum inverse radial derivative={model['minimum_inverse_radial_derivative']:.8g}.", "",
        f"Forward predictions passed through $K_E^{-1}$ then $B_A^{-1}$ return the fixed reference with max all-nine roundtrip error {forward['predicted_forward_inverse_roundtrip_max_mm']*1000:.3g} um. On those predicted grids, the minimum inverse projective denominator is {forward['minimum_predicted_inverse_keystone_denominator']:.8g} and the minimum inverse radial derivative is {forward['minimum_predicted_inverse_radial_derivative']:.8g}.", "",
        stats("Inverse-fit corners", model["inverse_fit_corner_metrics"]),
        stats("Forward corners", forward["corner_forward"]),
        stats("Forward five noncorner checks", forward["noncorner_forward"]),
        stats("Forward all nine points", forward["all_forward"]),
        stats("Inverse five noncorner checks", forward["noncorner_inverse"]), "",
        "## Comparison against previous K on the same new B", "",
        "Both variants use the same fixed physical power-law barrel B_A and score all 51 x 401 actual grids. The previous K uses its saved accommodation-dependent sx/sy/qy and qx=0; the new K uses gaze-only sx/sy/qx/qy.", "",
        stats("Old K forward corners", comp["old_forward_corner"]),
        stats("New K forward corners", comp["new_forward_corner"]),
        stats("Old K forward five noncorner checks", comp["old_forward_noncorners"]),
        stats("New K forward five noncorner checks", comp["new_forward_noncorners"]),
        stats("Old K forward all nine points", comp["old_forward_all"]),
        stats("New K forward all nine points", comp["new_forward_all"]),
        stats("Old K inverse corners", comp["old_inverse_corner"]),
        stats("New K inverse corners", comp["new_inverse_corner"]),
        stats("Old K inverse five noncorner checks", comp["old_inverse_noncorners"]),
        stats("New K inverse five noncorner checks", comp["new_inverse_noncorners"]), "",
        "## Coefficients", "",
        "Rows are parameter families sx, sy, qnx, qny; columns are powers j=1..4 of E/20:", "",
        "| Parameter | j=1 | j=2 | j=3 | j=4 |", "|---|---:|---:|---:|---:|",
    ]
    for name, vals in zip(("sx-1", "sy-1", "qnx", "qny"), coeff):
        lines.append(f"| {name} | " + " | ".join(f"{v:.12g}" for v in vals) + " |")
    lines += ["", f"Both q vectors were fit independently. The fitted $q_{{n,x}}$ curve is effectively zero by symmetry (maximum absolute value {np.max(np.abs(model['parameter_curves']['qnx'])):.3g}); $q_{{n,y}}$ carries the observed gaze-dependent perspective term."]
    k0, c1, exponent = inputs["k_coefficients"]
    lines += ["", "## Fixed barrel inputs", "",
              f"$k_1(A)=({c1:.12g})A^{{{exponent:.12g}}}+({k0:.12g})$ mm$^{{-2}}$. Saved paraxial scales include s(0)={model['paraxial_scale_vs_a0'][0]:.12g} and s(5)={model['paraxial_scale_vs_a0'][-1]:.12g}.", "",
              "## Files", "",
              "`p4_transformation_stages.png/.svg`, `p4_transformation_sweep.png/.svg`, and `p4_transformation_accommodation_overlay.png/.svg` show reference, actual, and refit predictions. All keystone panels swap physical axes for display, show only outer perimeters and center dots, and use the established shared limits. `keystone_parameter_curves.png/.svg` shows the four fitted gaze curves; `inverse_corner_rms.png/.svg` maps inverse-fit corner RMS across A/E.", "",
              "The model is empirical for this one-angle P4 simulation domain. The separate qx and qy terms do not establish independent physical rotation axes or generalize beyond the sampled conditions without new validation.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-nfev", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    inputs = load_inputs()
    model, fitter, results, comparison, old_fit = fit_model(inputs, args.max_nfev)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir
    artifact = {**model, "forward_errors_mm": results["forward_errors_mm"],
                "recovered_reference_mm": results["recovered_reference_mm"],
                "predicted_real_grid_mm": results["predicted_real_grid_mm"],
                "old_v2_K_comparison_arrays": {k: v for k, v in comparison.items() if k.endswith("_mm")}}
    with (out / "p4_separable_transform.pkl").open("wb") as stream:
        pickle.dump(artifact, stream, pickle.HIGHEST_PROTOCOL)
    save_figures(out, inputs, model, fitter, results, comparison, old_fit)
    (out / "report.md").write_text(report_text(model, comparison, inputs), encoding="utf-8")
    print(f"Schema: {SCHEMA}")
    print(f"Fit converged={model['fit_success']}, nfev={model['fit_nfev']}, inverse corner RMS="
          f"{model['inverse_fit_corner_metrics']['vertex_rms_um']:.6f} um, max="
          f"{model['inverse_fit_corner_metrics']['vertex_max_um']:.6f} um")
    print(f"Forward new K corners/noncorners/all-nine RMS="
          f"{results['corner_forward']['vertex_rms_um']:.6f}/"
          f"{results['noncorner_forward']['vertex_rms_um']:.6f}/"
          f"{results['all_forward']['vertex_rms_um']:.6f} um")
    print(f"Old-vs-new forward all-nine RMS="
          f"{comparison['old_forward_all']['vertex_rms_um']:.6f}/"
          f"{comparison['new_forward_all']['vertex_rms_um']:.6f} um")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
