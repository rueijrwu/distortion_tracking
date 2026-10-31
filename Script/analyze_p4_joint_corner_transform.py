"""Fit a joint real-ray four-corner inverse barrel and keystone transform."""
from __future__ import annotations

import argparse
import hashlib
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "distortion_grid_p4" / "distortion_grid.pkl"
DEFAULT_OUTPUT = ROOT / "data" / "p4_joint_corner_transform"
SCHEMA = "p4-joint-corner-inverse-v2"


def load_data(path: Path):
    raw = path.read_bytes()
    data = pickle.loads(raw)
    required = {"accommodation_d", "eye_rotation_deg", "field_x_relative", "field_y_relative",
                "real_x_mm", "real_y_mm", "z_mm"}
    if not isinstance(data, np.ndarray) or data.dtype.names is None or not required <= set(data.dtype.names):
        raise ValueError("Expected the structured P4 distortion grid")
    a, theta = np.unique(data["accommodation_d"]), np.unique(data["eye_rotation_deg"])
    if not np.allclose(a, np.linspace(0, 5, 51), atol=1e-9, rtol=0):
        raise ValueError("Expected 51 accommodation states from 0 to 5 D")
    if not np.allclose(theta, np.linspace(-20, 20, 401), atol=1e-9, rtol=0):
        raise ValueError("Expected 401 rotations from -20 to 20 degrees")
    if not np.allclose(data["z_mm"], 0, atol=1e-9, rtol=0):
        raise ValueError("Only Z=0 data are supported")
    ai = np.searchsorted(a, data["accommodation_d"])
    ti = np.searchsorted(theta, data["eye_rotation_deg"])
    order = np.lexsort((data["field_y_relative"], data["field_x_relative"], ti, ai))
    d = data[order]
    fields = np.column_stack((d["field_x_relative"], d["field_y_relative"])).reshape(len(a), len(theta), 9, 2)
    coords = np.column_stack((d["real_x_mm"], d["real_y_mm"])).reshape(len(a), len(theta), 9, 2)
    expected = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    corners_idx = np.flatnonzero(np.all(np.abs(fields[0, 0]) == 1, axis=1))
    if not np.array_equal(fields[0, 0, corners_idx], expected):
        raise ValueError("Corner field identity/order mismatch")
    if not np.all(fields == fields[0:1, 0:1]):
        raise ValueError("Field identities vary across states")
    if not np.isfinite(coords).all():
        raise ValueError("Nonfinite real-ray coordinates")
    if not np.all(np.bincount(ai * len(theta) + ti) == 9):
        raise ValueError("Incomplete 3x3 state grid")
    return a, theta, coords[:, :, corners_idx], hashlib.sha256(raw).hexdigest(), len(raw)


def basis(a, t, accommodation_order, gaze_order, beta_order):
    """Columns: beta A terms, then sx, sy, q terms in (A/5)^l(theta/20)^j."""
    columns = [(a / 5) ** l for l in range(1, beta_order + 1)]
    columns += [(a / 5) ** l * (t / 20) ** j
                for _ in range(3) for l in range(accommodation_order + 1)
                for j in range(1, gaze_order + 1)]
    return np.column_stack(columns)


def unpack(p, a, t, accommodation_order, gaze_order, beta_order):
    k = 0
    beta = np.zeros_like(a, dtype=float)
    for l in range(1, beta_order + 1):
        beta += p[k] * (a / 5) ** l
        k += 1
    values = []
    for param in range(3):
        value = np.ones_like(a, dtype=float) if param < 2 else np.zeros_like(a, dtype=float)
        for l in range(accommodation_order + 1):
            for j in range(1, gaze_order + 1):
                value += p[k] * (a / 5) ** l * (t / 20) ** j
                k += 1
        values.append(value)
    return beta, *values


def inverse_map(real, pars):
    X, Y = real[..., 0], real[..., 1]
    sx, sy, q = (pars[..., i, None] for i in range(3))
    denominator = sy - q * Y
    if np.any(denominator <= 0) or np.any(sx <= 0) or np.any(sy <= 0):
        raise ValueError("Invalid inverse keystone domain")
    return np.stack((X * sy / (sx * denominator), Y / denominator), axis=-1), denominator


def inverse_barrel(v, beta):
    radius = np.linalg.norm(v, axis=-1)
    recovered_radius = radius.copy()
    for _ in range(14):
        f = recovered_radius * (1 + beta[..., None] * recovered_radius**2) - radius
        derivative = 1 + 3 * beta[..., None] * recovered_radius**2
        if np.any(derivative <= 0):
            raise ValueError("Radial inverse left the monotone branch")
        recovered_radius -= f / derivative
        if np.max(np.abs(f)) < 1e-13:
            break
    gain = 1 + beta[..., None] * recovered_radius**2
    if np.any(gain <= 0):
        raise ValueError("Radial inverse has nonpositive gain")
    return v * np.divide(recovered_radius, radius, out=np.ones_like(radius), where=radius > 0)[..., None], recovered_radius


def evaluate(p, a, t, real, ref, accommodation_order, gaze_order, beta_order, jac=False):
    beta, sx, sy, q = unpack(p, a, t, accommodation_order, gaze_order, beta_order)
    pars = np.column_stack((sx, sy, q))
    v, _ = inverse_map(real, pars)
    u, ru = inverse_barrel(v, beta)
    residual = ((u - ref) * SCALE).reshape(-1)
    if not jac:
        return residual

    X, Y = real[..., 0], real[..., 1]
    sxv, syv, qv = sx[:, None], sy[:, None], q[:, None]
    d = syv - qv * Y
    vx = X * syv / (sxv * d)
    vy = Y / d
    Jk = np.zeros((*X.shape, 2, 3))
    Jk[..., 0, 0] = -vx / sxv
    Jk[..., 0, 1] = -X * qv * Y / (sxv * d**2)
    Jk[..., 1, 1] = -Y / d**2
    Jk[..., 0, 2] = X * syv * Y / (sxv * d**2)
    Jk[..., 1, 2] = Y**2 / d**2
    denom = 1 + beta[:, None] * ru**2
    radial_derivative = 1 + 3 * beta[:, None] * ru**2
    Jr = (np.eye(2)[None, None, :, :] / denom[..., None, None]
          - (2 * beta[:, None] / (denom * radial_derivative))[..., None, None]
          * (u[..., :, None] * u[..., None, :]))
    Jp = Jr @ Jk
    Jb = (-(ru**2)[..., None] * u / radial_derivative[..., None])[..., None]
    B = basis(a, t, accommodation_order, gaze_order, beta_order)
    nstate, ncorner = len(a), real.shape[1]
    out = np.zeros((nstate * ncorner * 2, len(p)))
    c = 0
    for _ in range(beta_order):
        out[:, c] = (Jb[..., 0] * B[:, c, None, None]).reshape(-1) * SCALE
        c += 1
    for param in range(3):
        for l in range(accommodation_order + 1):
            for j in range(1, gaze_order + 1):
                bi = beta_order + param * ((accommodation_order + 1) * gaze_order) + l * gaze_order + j - 1
                out[:, c] = (Jp[..., param] * B[:, bi, None, None]).reshape(-1) * SCALE
                c += 1
    return out


def physical_terms(coefficients, L, accommodation_order, gaze_order, beta_order):
    rows = []
    k = 0
    for l in range(1, beta_order + 1):
        rows.append({"parameter": "beta", "A_power": l, "theta_power": 0,
                     "scaled_basis_coefficient": float(coefficients[k]),
                     "physical_A_theta_coefficient_mm^-2_D^-A_power":
                         float(coefficients[k] / (L**2 * 5**l))})
        k += 1
    for name in ("sx", "sy", "q_n"):
        for l in range(accommodation_order + 1):
            for j in range(1, gaze_order + 1):
                val = float(coefficients[k])
                factor = L if name == "q_n" else 1.0
                row = {"parameter": name, "A_power": l, "theta_power": j,
                       "scaled_basis_coefficient": val,
                       "physical_A_theta_coefficient": val / (5**l * 20**j * factor)}
                if name == "q_n":
                    row["physical_units"] = "mm^-1 D^-A_power degree^-theta_power"
                elif name in ("sx", "sy"):
                    row["physical_units"] = "D^-A_power degree^-theta_power"
                rows.append(row)
                k += 1
    return rows


def fit_joint(a, theta, realmm, accommodation_order, gaze_order, beta_order, max_nfev):
    global SCALE
    iz = int(np.argmin(np.abs(theta)))
    reference_mm = realmm[0, iz].copy()
    SCALE = float(abs(reference_mm[0, 0]))
    real = realmm / SCALE
    reference = reference_mm / SCALE
    aa, tt = np.meshgrid(a, theta, indexing="ij")
    af, tf = aa.ravel(), tt.ravel()
    rays = real.reshape(-1, 4, 2)
    refs = np.broadcast_to(reference, (len(a) * len(theta), 4, 2))
    B = basis(af, tf, accommodation_order, gaze_order, beta_order)
    p0 = np.zeros(B.shape[1])
    # Initialize beta from the measured theta-zero corner gain.
    baseline = real[:, iz]
    gain = np.sum(reference[None, :, :] * baseline, axis=(1, 2)) / np.sum(reference**2)
    beta_anchor = (gain - 1) / 2
    beta_design = np.column_stack([(a / 5) ** l for l in range(1, beta_order + 1)])
    p0[:beta_order] = np.linalg.lstsq(beta_design, beta_anchor, rcond=None)[0]
    # Seed the joint fit by least-squares polynomials through direct per-state inverses.
    direct = np.empty((len(a), len(theta), 3))
    for ia in range(len(a)):
        for it in range(len(theta)):
            X, Y = real[ia, it, :, 0], real[ia, it, :, 1]
            pars = np.array([1.0, 1.0, 0.0])
            for _ in range(18):
                sx0, sy0, q0 = pars
                d0 = sy0 - q0 * Y
                px, py = X * sy0 / (sx0 * d0), Y / d0
                residual0 = np.stack((px, py), axis=-1) - reference
                J = np.zeros((8, 3))
                J[0::2, 0] = -px / sx0
                J[0::2, 1] = -X * q0 * Y / (sx0 * d0**2)
                J[1::2, 1] = -Y / d0**2
                J[0::2, 2] = X * sy0 * Y / (sx0 * d0**2)
                J[1::2, 2] = Y**2 / d0**2
                step = np.linalg.lstsq(J, residual0.reshape(-1), rcond=None)[0]
                pars -= step
                if np.max(np.abs(step)) < 1e-12:
                    break
            direct[ia, it] = pars
    for param in range(3):
        target = direct[:, :, param] - (1.0 if param < 2 else 0.0)
        design = np.column_stack([((aa / 5) ** l * (tt / 20) ** j).ravel()
                                  for l in range(accommodation_order + 1)
                                  for j in range(1, gaze_order + 1)])
        start = beta_order + param * (accommodation_order + 1) * gaze_order
        stop = start + (accommodation_order + 1) * gaze_order
        p0[start:stop] = np.linalg.lstsq(design, target.ravel(), rcond=None)[0]
    sample = np.linspace(0, len(af) - 1, min(64, len(af)), dtype=int)
    analytic = evaluate(p0, af[sample], tf[sample], rays[sample], refs[sample],
                        accommodation_order, gaze_order, beta_order, True)
    max_jac_rel = 0.0
    for col in range(len(p0)):
        step = 1e-6 * max(1.0, abs(p0[col]))
        pp, pm = p0.copy(), p0.copy()
        pp[col] += step
        pm[col] -= step
        fd = (evaluate(pp, af[sample], tf[sample], rays[sample], refs[sample], accommodation_order, gaze_order, beta_order)
              - evaluate(pm, af[sample], tf[sample], rays[sample], refs[sample], accommodation_order, gaze_order, beta_order)) / (2 * step)
        max_jac_rel = max(max_jac_rel, float(np.max(np.abs(fd - analytic[:, col])) /
                                               max(1e-8, float(np.max(np.abs(fd))))))
    if max_jac_rel > 2e-5:
        raise RuntimeError(f"Analytic Jacobian finite-difference audit failed: {max_jac_rel:g}")
    fun = lambda p: evaluate(p, af, tf, rays, refs, accommodation_order, gaze_order, beta_order)
    jac = lambda p: evaluate(p, af, tf, rays, refs, accommodation_order, gaze_order, beta_order, True)
    opt = least_squares(fun, p0, jac=jac, method="trf", max_nfev=max_nfev, x_scale="jac", verbose=0)
    beta, sx, sy, q = unpack(opt.x, af, tf, accommodation_order, gaze_order, beta_order)
    pars = np.column_stack((sx, sy, q))
    recovered, denominator = inverse_map(rays, pars)
    recovered, radius = inverse_barrel(recovered, beta)
    residual = (recovered - refs) * SCALE
    vertex = np.sqrt(np.mean(np.sum(residual.reshape(-1, 4, 2)**2, axis=-1), axis=-1)).reshape(len(a), len(theta))
    roundtrip = recovered * (1 + beta[:, None, None] * np.sum(recovered**2, axis=-1)[..., None])
    xx, yy = roundtrip[..., 0], roundtrip[..., 1]
    roundtrip = np.stack((sx[:, None] * xx / (1 + q[:, None] * yy),
                          sy[:, None] * yy / (1 + q[:, None] * yy)), axis=-1)
    # Reference normalized coordinates in the canonical artifact.
    result = {
        "schema": SCHEMA, "model": "joint", "accommodation_order": accommodation_order,
        "gaze_order": gaze_order, "beta_order": beta_order,
        "parameter_count": len(opt.x), "success": bool(opt.success), "status": int(opt.status),
        "message": str(opt.message), "nfev": int(opt.nfev), "cost": float(opt.cost),
        "optimality": float(opt.optimality), "coefficients_normalized": opt.x.copy(),
        "coefficient_basis": "beta=sum b_l(A/5)^l; sx/sy/q_n=identity+sum c_lj(A/5)^l(theta/20)^j",
        "coefficient_units": "beta dimensionless; q_n normalized; sx/sy unitless",
        "polynomial_terms_physical_basis": physical_terms(opt.x, SCALE, accommodation_order, gaze_order, beta_order),
        "beta_effective_dimensionless": beta.reshape(len(a), len(theta)),
        "sx": sx.reshape(len(a), len(theta)), "sy": sy.reshape(len(a), len(theta)),
        "q_normalized": q.reshape(len(a), len(theta)),
        "beta_effective_mm^-2": beta.reshape(len(a), len(theta)) / SCALE**2,
        "q_mm^-1": q.reshape(len(a), len(theta)) / SCALE,
        "recovered_vertices_mm": (recovered * SCALE).reshape(len(a), len(theta), 4, 2),
        "correct_vertex_rms_mm": vertex, "coordinate_rmse_mm": np.sqrt(np.mean(residual**2, axis=(-2, -1))).reshape(len(a), len(theta)),
        "maximum_point_error_mm": np.linalg.norm(residual, axis=-1).reshape(len(a), len(theta), 4).max(axis=-1),
        "minimum_inverse_keystone_denominator": float(np.min(denominator)),
        "minimum_radial_derivative": float(np.min(1 + 3 * beta[:, None] * radius**2)),
        "fraction_states_vertex_rms_le_1um": float(np.mean(vertex <= .001)),
        "mean_state_vertex_rms_mm": float(np.mean(vertex)),
        "max_state_vertex_rms_mm": float(np.max(vertex)),
        "global_pooled_vertex_rms_mm": float(np.sqrt(np.mean(vertex**2))),
        "theta0_max_vertex_rms_mm": float(np.max(vertex[:, iz])),
        "theta0_mean_vertex_rms_mm": float(np.mean(vertex[:, iz])),
        "analytic_jacobian_max_relative_fd_error": max_jac_rel,
        "max_roundtrip_error_mm": float(np.max(np.linalg.norm((roundtrip - rays) * SCALE, axis=-1))),
        "initialization_beta_anchor_dimensionless": beta_anchor.copy(),
    }
    return result, reference_mm


def canonicalize_legacy(raw):
    """Keep only the existing joint order-4 fit and its scientifically useful fields."""
    if raw.get("schema") == SCHEMA:
        return raw
    model = raw["models"]["gaze_order_4"]["coupled"]
    keep = ("accommodation_order", "gaze_order", "parameter_count", "success", "status", "message",
            "nfev", "cost", "optimality", "coefficients_normalized", "coefficient_basis", "coefficient_units",
            "polynomial_terms_physical_basis", "beta_effective_dimensionless", "sx", "sy", "q_normalized",
            "beta_effective_mm^-2", "q_mm^-1", "recovered_vertices_mm", "correct_vertex_rms_mm",
            "coordinate_rmse_mm", "maximum_point_error_mm", "minimum_inverse_keystone_denominator",
            "minimum_radial_derivative", "fraction_states_vertex_rms_le_1um", "mean_state_vertex_rms_mm",
            "max_state_vertex_rms_mm", "global_pooled_vertex_rms_mm", "theta0_max_vertex_rms_mm",
            "theta0_mean_vertex_rms_mm", "analytic_jacobian_max_relative_fd_error", "max_roundtrip_error_mm")
    cleaned = {key: model[key] for key in keep}
    cleaned.update({"schema": SCHEMA, "model": "joint", "beta_order": model["accommodation_order"]})
    return {"schema": SCHEMA, "model": "joint", "input_path": raw["input_path"],
            "input_sha256": raw["input_sha256"], "input_bytes": raw["input_bytes"],
            "ray_type": raw["ray_type"], "z_mm": raw["z_mm"], "field_ids": raw["field_ids"],
            "normalization_L_mm": raw["normalization_L_mm"],
            "accommodation_d": raw["accommodation_d"], "eye_rotation_deg": raw["eye_rotation_deg"],
            "reference_corners_mm": raw["reference_corners_mm"],
            "reference_corners_normalized": raw["reference_corners_mm"] / raw["normalization_L_mm"],
            "fit": cleaned}


def joint_values(fit, a, theta):
    aa, tt = np.meshgrid(a, theta, indexing="ij")
    beta, sx, sy, q = unpack(fit["coefficients_normalized"], aa.ravel(), tt.ravel(),
                             fit["accommodation_order"], fit["gaze_order"], fit["beta_order"])
    shape = (len(a), len(theta))
    return {"beta_effective_dimensionless": beta.reshape(shape), "sx": sx.reshape(shape),
            "sy": sy.reshape(shape), "q_normalized": q.reshape(shape)}


def plot_results(out, a, theta, result):
    fit = result["fit"]
    colors = plt.cm.viridis(np.linspace(0, 1, 6))
    selected_a = np.arange(0, len(a), 10)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    for ax, key, title in zip(axs, ("sx", "sy", "q_normalized"), ("$s_x$", "$s_y$", "$q_n$")):
        for color, ai in zip(colors, selected_a):
            ax.plot(theta, fit[key][ai], color=color, lw=1.5, label=f"A={a[ai]:g} D")
        ax.set(title=title, xlabel="Eye rotation $\\theta$ (degrees)")
        ax.grid(alpha=.25)
    axs[0].set_ylabel("Fitted parameter")
    axs[0].legend(fontsize=8, ncol=2)
    fig.suptitle("Joint keystone parameters across eye rotation")
    fig.savefig(out / "parameters_vs_gaze.png", dpi=180)
    plt.close(fig)

    ti = [int(np.argmin(np.abs(theta - x))) for x in (-20, -10, 0, 10, 20)]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    for ax, key, title in zip(axs, ("sx", "sy", "q_normalized"), ("$s_x$", "$s_y$", "$q_n$")):
        for ii, x in zip(ti, (-20, -10, 0, 10, 20)):
            ax.plot(a, fit[key][:, ii], marker=".", ms=3, label=f"$\\theta={x}^\\circ$")
        ax.set(title=title, xlabel="Accommodation $A$ (D)")
        ax.grid(alpha=.25)
    axs[0].set_ylabel("Fitted parameter")
    axs[0].legend(fontsize=8)
    fig.suptitle("Joint keystone parameters across accommodation")
    fig.savefig(out / "parameters_vs_accommodation.png", dpi=180)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, key, title in zip(axs, ("sx", "sy", "q_normalized"), ("$s_x$", "$s_y$", "$q_n$")):
        image = ax.pcolormesh(theta, a, fit[key], shading="auto", cmap="viridis")
        fig.colorbar(image, ax=ax, label="Fitted parameter")
        ax.set(title=title, xlabel="Eye rotation $\\theta$ (degrees)", ylabel="Accommodation $A$ (D)")
    fig.suptitle("Joint keystone parameter surfaces")
    fig.savefig(out / "joint_parameter_heatmaps.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.7), constrained_layout=True)
    image = ax.pcolormesh(theta, a, fit["correct_vertex_rms_mm"] * 1000, shading="auto", cmap="magma")
    fig.colorbar(image, ax=ax, label="Four-corner vertex RMS ($\\mu$m)")
    ax.set(title="Joint-fit corner residuals", xlabel="Eye rotation $\\theta$ (degrees)", ylabel="Accommodation $A$ (D)")
    fig.savefig(out / "joint_vertex_rms.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(a, fit["beta_effective_dimensionless"][:, int(np.argmin(np.abs(theta)))], color="navy", lw=2)
    ax.set(title="Joint relative-barrel coefficient at $\\theta=0$", xlabel="Accommodation $A$ (D)",
           ylabel="Effective relative $\\beta$ (normalized)")
    ax.grid(alpha=.25)
    fig.savefig(out / "barrel_parameter_vs_accommodation.png", dpi=180)
    plt.close(fig)


def write_report(path, result):
    fit = result["fit"]
    terms = fit["polynomial_terms_physical_basis"]
    terms_by = {}
    for row in terms:
        terms_by.setdefault(row["parameter"], []).append(row)
    lines = [
        "# P4 joint real-ray four-corner transform", "",
        "This single joint fit uses the 51 by 401 sampled states at $Z=0$ and only the four real-ray field corners. Its fixed reference is the measured corner pattern at $A=0$, $\\theta=0$. Coordinates are normalized by $L="
        f"{result['normalization_L_mm']:.12g}\\,\\mathrm{{mm}}$; no paraxial coordinates or extra magnification parameter enter the objective.", "",
        "The forward map is $K\\circ B_\\beta$: relative barrel first, then keystone. For a real corner $\\mathbf R=(X,Y)$, recovery applies the inverse keystone and then inverse barrel:", "",
        "$$", "\\mathbf v=K^{-1}(\\mathbf R),\\qquad \\mathbf u=B_\\beta^{-1}(\\mathbf v),\\qquad \\min \\operatorname{RMS}_{corners}(\\mathbf u-\\mathbf u_0).", "$$", "",
        "The normalized keystone is $X=s_xx/(1+q_ny)$, $Y=s_yy/(1+q_ny)$ with $H_{33}=1$. The fitted scales $s_x,s_y$ are independent; no unit-determinant constraint is imposed. The inverse uses $y=Y/(s_y-q_nY)$ and $x=Xs_y/[s_x(s_y-q_nY)]$. The physical keystone coefficient is $q=q_n/L$ in $\\mathrm{mm}^{-1}$.", "",
        "The relative barrel is $B_\\beta(\\mathbf u)=(1+\\beta r^2)\\mathbf u$, where $\\beta(A)=b_1(A/5)+b_2(A/5)^2$. Here $\\beta$ is dimensionless in normalized coordinates and converts to $\\mathrm{mm}^{-2}$ by dividing by $L^2$. Since all fitted corners share $r^2=2$, these data constrain the relative radial effect at that radius; they do not identify absolute physical $k_1$ separately from isotropic scale.", "",
        "The joint parameter family is", "", "$$",
        "s_x=1+\\sum_{l=0}^{2}\\sum_{j=1}^{4}c^x_{lj}(A/5)^l(\\theta/20)^j,\\qquad",
        "s_y=1+\\sum_{l=0}^{2}\\sum_{j=1}^{4}c^y_{lj}(A/5)^l(\\theta/20)^j,",
        "$$", "", "$$",
        "q_n=\\sum_{l=0}^{2}\\sum_{j=1}^{4}c^q_{lj}(A/5)^l(\\theta/20)^j.",
        "$$", "",
        "The order-4 gaze terms and order-2 accommodation terms are fitted together against the recovered-corner residuals. The theta-zero keystone is the identity by construction, and $\\beta(0)=0$.", "",
        "## Parameter dependence", "",
        "The dominant fitted dependence, with rounded coefficients for interpretation, is", "", "$$",
        "s_x\\approx1+f_{x2}(a)t^2+f_{x4}(a)t^4,\\qquad s_y\\approx1+f_{y2}(a)t^2+f_{y4}(a)t^4,",
        "$$", "", "$$", "q_n\\approx f_{q1}(a)t+f_{q3}(a)t^3,\\qquad a=A/5,\\qquad t=\\theta/20.", "$$", "",
        "Each function is quadratic in normalized accommodation:", "", "$$",
        "\\begin{aligned}",
        "f_{x2}(a)&=0.0594091+0.0342887a+0.000354141a^2,\\quad&f_{x4}(a)&=0.00191096+0.000448253a+0.00287106a^2,\\\\",
        "f_{y2}(a)&=0.122914+0.111543a+0.00729162a^2,\\quad&f_{y4}(a)&=-0.00353496+0.00351620a+0.00833102a^2,\\\\",
        "f_{q1}(a)&=-0.0630503-0.0603090a-0.0137921a^2,\\quad&f_{q3}(a)&=0.00811845+0.00575111a+0.00296676a^2.",
        "\\end{aligned}", "$$", "",
        "These are approximate summaries of the dominant gaze terms; every coefficient in the full fitted model remains in the saved prediction. The relative-barrel accommodation polynomial is $\\beta=-0.0428500981a-9.70843337\\times10^{-5}a^2$. At $A=5$ D ($a=1$), its quadratic coefficient contributes about 0.23% of the linear coefficient magnitude. This coefficient comparison does not validate removing the quadratic term.", "",
        "## Sampled fit summary", "",
        f"The fit converged: {fit['success']} ({fit['message']}); {fit['nfev']} evaluations for {fit['parameter_count']} coefficients. Mean state four-corner vertex RMS is {fit['mean_state_vertex_rms_mm']*1000:.6f} $\\mu\\mathrm{{m}}$ and the worst sampled state is {fit['max_state_vertex_rms_mm']*1000:.6f} $\\mu\\mathrm{{m}}$. The minimum inverse-keystone denominator is {fit['minimum_inverse_keystone_denominator']:.8g}; the minimum inverse-barrel radial derivative is {fit['minimum_radial_derivative']:.8g}. The analytic Jacobian finite-difference audit error is {fit['analytic_jacobian_max_relative_fd_error']:.3g}, and the maximum forward/inverse round-trip discrepancy is {fit['max_roundtrip_error_mm']*1000:.3g} $\\mu\\mathrm{{m}}$.", "",
        "These are in-sample residuals on the simulated grid. They summarize this corner-only transform family and do not establish a unique physical barrel mechanism.", "",
        "## Figures", "",
        "`parameters_vs_gaze.png` shows $s_x,s_y,q_n$ versus gaze rotation at accommodation overlays. `parameters_vs_accommodation.png` shows the same parameters versus accommodation at selected gaze angles. `joint_parameter_heatmaps.png` shows their full joint surfaces; `barrel_parameter_vs_accommodation.png` isolates the fitted relative-barrel dependence; `joint_vertex_rms.png` shows residuals over the sampled domain.", "",
        "## Full physical coefficient tables", "",
        "The complete fitted coefficients on physical monomials $A^l\\theta^j$ appear below. Scale coefficients have units $\\mathrm{D}^{-l}\\mathrm{degree}^{-j}$; $q=q_n/L$ coefficients have units $\\mathrm{mm}^{-1}\\mathrm{D}^{-l}\\mathrm{degree}^{-j}$. The full scaled-basis coefficient vector is stored in the pickle.", "",
    ]
    for name, units in (("sx", "\\mathrm{D}^{-l}\\mathrm{degree}^{-j}"), ("sy", "\\mathrm{D}^{-l}\\mathrm{degree}^{-j}"), ("q=q_n/L", "\\mathrm{mm}^{-1}\\mathrm{D}^{-l}\\mathrm{degree}^{-j}")):
        term_name = "q_n"
        lines += [f"### {name}", "", f"Physical coefficient units: ${units}$.", "",
                  "| $A$ power $l$ | $\\theta$ power $j$ | Physical coefficient |", "|---:|---:|---:|"]
        for row in terms_by[term_name]:
            val = row["physical_A_theta_coefficient"]
            lines.append(f"| {row['A_power']} | {row['theta_power']} | {val:.10g} |")
        lines.append("")
    bterms = terms_by["beta"]
    lines += ["### Relative-barrel accommodation polynomial", "",
              "| $A$ power | Scaled coefficient | Physical coefficient ($\\mathrm{mm}^{-2}\\,\\mathrm{D}^{-l}$) |",
              "|---:|---:|---:|"]
    for row in bterms:
        lines.append(f"| {row['A_power']} | {row['scaled_basis_coefficient']:.12g} | {row['physical_A_theta_coefficient_mm^-2_D^-A_power']:.10g} |")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gaze-order", type=int, default=4)
    parser.add_argument("--accommodation-order", type=int, default=2)
    parser.add_argument("--max-nfev", type=int, default=500)
    parser.add_argument("--replot", action="store_true", help="Migrate/replot the stored order-4 joint fit without optimizing")
    args = parser.parse_args()
    if not 1 <= args.gaze_order <= 8 or not 0 <= args.accommodation_order <= 5:
        parser.error("gaze order must be 1..8 and accommodation order 0..5")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_pickle = args.output_dir / "p4_joint_corner_transform.pkl"
    if args.replot:
        with output_pickle.open("rb") as stream:
            raw = pickle.load(stream)
        a, theta = np.asarray(raw["accommodation_d"]), np.asarray(raw["eye_rotation_deg"])
        if raw.get("schema") == SCHEMA:
            result = raw
        else:
            result = canonicalize_legacy(raw)
            with output_pickle.open("wb") as stream:
                pickle.dump(result, stream, pickle.HIGHEST_PROTOCOL)
        plot_results(args.output_dir, a, theta, result)
        write_report(args.output_dir / "report.md", result)
        fit = result["fit"]
    else:
        a, theta, corners_mm, input_sha, input_bytes = load_data(args.input)
        fit, reference_mm = fit_joint(a, theta, corners_mm, args.accommodation_order,
                                      args.gaze_order, 2, args.max_nfev)
        values = joint_values(fit, a, theta)
        fit.update(values)
        result = {"schema": SCHEMA, "model": "joint", "input_path": str(args.input.resolve()),
                  "input_sha256": input_sha, "input_bytes": input_bytes, "ray_type": "real only", "z_mm": 0.0,
                  "field_ids": [[-1, -1], [-1, 1], [1, -1], [1, 1]], "normalization_L_mm": SCALE,
                  "accommodation_d": a, "eye_rotation_deg": theta,
                  "reference_corners_mm": reference_mm, "reference_corners_normalized": reference_mm / SCALE,
                  "fit": fit}
        with output_pickle.open("wb") as stream:
            pickle.dump(result, stream, pickle.HIGHEST_PROTOCOL)
        plot_results(args.output_dir, a, theta, result)
        write_report(args.output_dir / "report.md", result)
    print(f"joint: gaze order {fit['gaze_order']}, accommodation order {fit['accommodation_order']}; "
          f"vertex RMS mean/max {fit['mean_state_vertex_rms_mm']*1000:.6f}/"
          f"{fit['max_state_vertex_rms_mm']*1000:.6f} um; success={fit['success']}; nfev={fit['nfev']}", flush=True)


if __name__ == "__main__":
    main()
