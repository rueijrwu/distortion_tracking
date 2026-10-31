"""Fit the coupled physical-barrel and two-q P4 joint transform."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.optimize import least_squares

import refit_p4_separable_transform as common

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "data" / "p4_joint_power_transform"
SCHEMA = "p4-joint-powerbarrel-twoq-inverse-v1"


def joint_basis(accommodation, gaze):
    """The 12 columns (A/5)^l(E/20)^j, l=0..2 and j=1..4."""
    a, e = np.broadcast_arrays(np.asarray(accommodation, float), np.asarray(gaze, float))
    return np.stack([(a / 5.)**l * (e / 20.)**j
                     for l in range(3) for j in range(1, 5)], axis=-1)


def unpack_joint(coefficients, accommodation, gaze):
    """Return sx, sy, qnx, qny for the 48-coefficient coupled model."""
    p = np.asarray(coefficients, float)
    if p.shape != (48,):
        raise ValueError("Expected 48 joint coefficients")
    basis = joint_basis(accommodation, gaze)
    vals = p.reshape(4, 12) @ basis.reshape(-1, 12).T
    shape = basis.shape[:-1]
    vals = vals.reshape((4,) + shape)
    return 1. + vals[0], 1. + vals[1], vals[2], vals[3]


def forward_joint(reference_mm, accommodation, gaze, L, kcoeff, paraxial_scale, coefficients):
    """Integrated F(A,E)=K_A,E(B_A(r0)) in physical millimeter coordinates.

    w=s(A)D_k0^-1(r0), gamma=1+k1(A)||w||^2, and
    F=gamma*diag(sx(A,E),sy(A,E))*w /
      [1+gamma*(qnx(A,E)wx/L+qny(A,E)wy/L)].
    """
    aa, ee = np.broadcast_arrays(np.asarray(accommodation, float), np.asarray(gaze, float))
    state_shape = aa.shape
    flat_a, flat_e = aa.reshape(-1), ee.reshape(-1)
    sx, sy, qnx, qny = unpack_joint(coefficients, flat_a, flat_e)
    ref = np.asarray(reference_mm, float)
    k0 = float(common.radial_k1(kcoeff, 0.))
    para0, _ = common.invert_radial(ref, k0)
    # The caller supplies each requested state's saved A-specific scale.
    scales = np.broadcast_to(np.asarray(paraxial_scale, float), state_shape).reshape(-1)
    kA = common.radial_k1(kcoeff, flat_a)
    w = scales[:, None, None] * para0[None, :, :]
    gamma = 1. + kA[:, None] * np.sum(w*w, axis=-1)
    denominator = 1. + gamma * (qnx[:, None] * w[..., 0] / L + qny[:, None] * w[..., 1] / L)
    if np.any(sx <= 0) or np.any(sy <= 0) or np.any(denominator <= 0):
        raise ValueError("Forward joint map left its positive physical branch")
    out = gamma[..., None] * np.stack((sx[:, None] * w[..., 0], sy[:, None] * w[..., 1]), axis=-1)
    out /= denominator[..., None]
    return out.reshape(state_shape + (len(ref), 2))


def staged_forward(reference_mm, accommodation, gaze, L, kcoeff, paraxial_scale, coefficients):
    """Numerically compose B_A then K_A,E as an independent audit comparator."""
    aa, ee = np.broadcast_arrays(np.asarray(accommodation, float), np.asarray(gaze, float))
    state_shape = aa.shape
    flat_a, flat_e = aa.reshape(-1), ee.reshape(-1)
    scales = np.broadcast_to(np.asarray(paraxial_scale, float), state_shape).reshape(-1)
    sx, sy, qnx, qny = unpack_joint(coefficients, flat_a, flat_e)
    ref = np.asarray(reference_mm, float)
    para0, _ = common.invert_radial(ref, float(common.radial_k1(kcoeff, 0.)))
    w = scales[:, None, None] * para0[None, :, :]
    kA = common.radial_k1(kcoeff, flat_a)
    b = common.apply_radial(w, kA[:, None])
    out, _ = common.forward_keystone(b, L, sx, sy, qnx, qny)
    return out.reshape(state_shape + (len(reference_mm), 2))


def inverse_joint(real_mm, accommodation, gaze, L, kcoeff, paraxial_scale, coefficients):
    """Recover reference coordinates in the prescribed inverse order K^-1 then B_A^-1."""
    aa, ee = np.broadcast_arrays(np.asarray(accommodation, float), np.asarray(gaze, float))
    state_shape = aa.shape
    flat_a, flat_e = aa.reshape(-1), ee.reshape(-1)
    rays = np.asarray(real_mm, float).reshape(len(flat_a), -1, 2)
    scales = np.broadcast_to(np.asarray(paraxial_scale, float), state_shape).reshape(-1)
    sx, sy, qnx, qny = unpack_joint(coefficients, flat_a, flat_e)
    vnorm, inverse_den = common.inverse_keystone(rays, L, sx, sy, qnx, qny)
    ka = common.radial_k1(kcoeff, flat_a)
    para_a, radial_derivative = common.invert_radial(vnorm * L, ka[:, None])
    para0 = para_a / scales[:, None, None]
    recovered = common.apply_radial(para0, common.radial_k1(kcoeff, 0.))
    return (recovered.reshape(state_shape + (rays.shape[1], 2)),
            inverse_den.reshape(state_shape + (rays.shape[1],)),
            radial_derivative.reshape(state_shape + (rays.shape[1],)))


class JointObjective:
    def __init__(self, inputs, base):
        self.inputs = inputs
        self.base = base
        self.na, self.ng = len(inputs["a"]), len(inputs["gaze"])
        self.nstate = self.na * self.ng
        self.L = inputs["L"]
        self.a = np.repeat(inputs["a"], self.ng)
        self.gaze = np.tile(inputs["gaze"], self.na)
        self.rays = base.corners
        self.reference = base.reference_corners
        self.scales = base.scale_state
        self.kA = base.ka_state
        self.penalty_scale = np.full(self.nstate * 2, 1000.)
        self.penalty_den = np.full(self.nstate * 4, 1000.)
        self.penalty_radial = np.full(self.nstate * 4, 1000.)
        self.residual_count = self.rays.size + len(self.penalty_scale) + len(self.penalty_den) + len(self.penalty_radial)

    def residual(self, p):
        sx, sy, qnx, qny = unpack_joint(p, self.a, self.gaze)
        scale_penalty = np.concatenate((np.maximum(.05 - sx, 0.), np.maximum(.05 - sy, 0.))) * 1000.
        safe_sx, safe_sy = np.maximum(sx, .05), np.maximum(sy, .05)
        real = self.rays / self.L
        z = np.stack((real[..., 0] / safe_sx[:, None], real[..., 1] / safe_sy[:, None]), axis=-1)
        denominator = 1. - qnx[:, None] * z[..., 0] - qny[:, None] * z[..., 1]
        den_penalty = np.maximum(.05 - denominator, 0.).reshape(-1) * 1000.
        safe_den = np.maximum(denominator, .05)
        vnorm = z / safe_den[..., None]
        try:
            para_a, radial_deriv = common.invert_radial(vnorm * self.L, self.kA[:, None])
        except ValueError:
            # Large but finite penalties make invalid trial iterates rejectable by TRF.
            return np.full(self.residual_count, 1e5, dtype=float)
        para0 = para_a / self.scales[:, None, None]
        recovered = common.apply_radial(para0, common.radial_k1(self.inputs["k_coefficients"], 0.))
        corner_resid_um = (recovered - self.reference) * 1000.
        radial_penalty = np.maximum(.05 - radial_deriv, 0.).reshape(-1) * 1000.
        return np.concatenate((corner_resid_um.reshape(-1), scale_penalty,
                               den_penalty, radial_penalty))


def old_coefficients(old_fit):
    old = np.asarray(old_fit["coefficients_normalized"], float)
    # v2 order: beta(2), sx(12), sy(12), q_n(12); map to sx,sy,qnx,qny.
    p = np.zeros(48)
    p[:12] = old[2:14]
    p[12:24] = old[14:26]
    p[36:48] = old[26:38]
    return p


def active_bounds(p, lower, upper, tol=1e-6):
    return np.flatnonzero(np.abs(p - lower) <= tol), np.flatnonzero(np.abs(p - upper) <= tol)


def fit_joint(inputs, max_nfev):
    base = common.SeparableFitter(inputs)
    objective = JointObjective(inputs, base)
    p0 = old_coefficients(inputs["old"]["fit"])
    lower = np.full(48, -.5)
    upper = np.full(48, .5)

    def jacobian(p):
        jac = np.empty((objective.residual_count, len(p)), dtype=float)
        for col in range(len(p)):
            step = 1e-6
            plus, minus = p.copy(), p.copy()
            can_plus, can_minus = p[col] + step < upper[col], p[col] - step > lower[col]
            if can_plus and can_minus:
                plus[col] += step; minus[col] -= step
                jac[:, col] = (objective.residual(plus) - objective.residual(minus)) / (2 * step)
            elif can_plus:
                plus[col] += step
                jac[:, col] = (objective.residual(plus) - objective.residual(p)) / step
            else:
                minus[col] -= step
                jac[:, col] = (objective.residual(p) - objective.residual(minus)) / step
        return jac

    result = least_squares(objective.residual, p0, jac=jacobian, method="trf",
                           bounds=(lower, upper), max_nfev=max_nfev,
                           x_scale=np.full(48, .08), ftol=1e-14, xtol=1e-14,
                           gtol=1e-9, verbose=1)
    if not result.success:
        raise RuntimeError(f"Joint coupled fit failed: {result.message}")
    active_lo, active_hi = active_bounds(result.x, lower, upper)
    return result, base, objective, p0, lower, upper, active_lo, active_hi


def metric_vectors(errors):
    e = np.asarray(errors)
    norm = np.linalg.norm(e, axis=-1)
    return {"vertex_rms_um": float(np.sqrt(np.mean(norm**2))*1000),
            "vertex_max_um": float(np.max(norm)*1000),
            "coordinate_rmse_um": float(np.sqrt(np.mean(e**2))*1000)}


def grid_metrics(errors, ids):
    return common.metric_summary(errors, ids)


def selected_state_metrics(errors, ia, ie, ids):
    return metric_vectors(errors[ia, ie, ids, :])


def state_parameter_surfaces(p, a, gaze):
    aa, ee = np.meshgrid(a, gaze, indexing="ij")
    vals = unpack_joint(p, aa, ee)
    return {name: value for name, value in zip(("sx", "sy", "qnx", "qny"), vals)}


def forward_all(ref, p, inputs, base):
    aa, ee = np.meshgrid(inputs["a"], inputs["gaze"], indexing="ij")
    return forward_joint(ref, aa, ee, base.L, inputs["k_coefficients"],
                         base.scale_a[:, None], p)


def forward_from_surfaces(ref, surfaces, inputs, base):
    a, gaze = inputs["a"], inputs["gaze"]
    out = np.empty_like(inputs["coords"])
    for ia in range(len(a)):
        b = base.barrel_all[ia] / base.L
        den = 1. + surfaces["qx"][ia, :, None] * b[None, :, 0] + surfaces["qy"][ia, :, None] * b[None, :, 1]
        out[ia] = base.L * np.stack((surfaces["sx"][ia, :, None] * b[None, :, 0] / den,
                                     surfaces["sy"][ia, :, None] * b[None, :, 1] / den), axis=-1)
    return out


def recover_from_surfaces(real, surfaces, inputs, base):
    na, ng = len(inputs["a"]), len(inputs["gaze"])
    output = np.empty_like(real)
    denmin, radmin = np.inf, np.inf
    for ia in range(na):
        sx, sy = surfaces["sx"][ia], surfaces["sy"][ia]
        qx, qy = surfaces["qx"][ia], surfaces["qy"][ia]
        vnorm, den = common.inverse_keystone(real[ia], base.L, sx, sy, qx, qy)
        para, deriv = common.invert_radial(vnorm * base.L, base.ka[ia])
        output[ia] = common.apply_radial(para / base.scale_a[ia], base.k0)
        denmin = min(denmin, float(np.min(den)))
        radmin = min(radmin, float(np.min(deriv)))
    return output, denmin, radmin


def evaluate_variants(inputs, base, p_new, separable_artifact):
    a, gaze, coords = inputs["a"], inputs["gaze"], inputs["coords"]
    old_fit = inputs["old"]["fit"]
    old_sx, old_sy, old_q = common.old_parameter_surfaces(old_fit, a, gaze)
    old_surfaces = {"sx": old_sx, "sy": old_sy, "qx": np.zeros_like(old_q), "qy": old_q}
    new_surfaces = {**state_parameter_surfaces(p_new, a, gaze),
                    "qx": state_parameter_surfaces(p_new, a, gaze)["qnx"],
                    "qy": state_parameter_surfaces(p_new, a, gaze)["qny"]}
    sep_p = separable_artifact["coefficients_normalized"]
    sep_curves = separable_artifact["parameter_curves"]
    sep_surfaces = {"sx": np.broadcast_to(sep_curves["sx"], (len(a), len(gaze))),
                    "sy": np.broadcast_to(sep_curves["sy"], (len(a), len(gaze))),
                    "qx": np.broadcast_to(sep_curves["qnx"], (len(a), len(gaze))),
                    "qy": np.broadcast_to(sep_curves["qny"], (len(a), len(gaze)))}
    # All three variants use the identical fixed new physical barrel grid.
    predictions = {"old_hybrid": forward_from_surfaces(base.ref, old_surfaces, inputs, base),
                  "separable": forward_from_surfaces(base.ref, sep_surfaces, inputs, base),
                  "joint": forward_all(base.ref, p_new, inputs, base)}
    result = {}
    for name, pred in predictions.items():
        chosen = (old_surfaces if name == "old_hybrid" else sep_surfaces if name == "separable" else
                  {"sx": new_surfaces["sx"], "sy": new_surfaces["sy"],
                   "qx": new_surfaces["qx"], "qy": new_surfaces["qy"]})
        # Inverse errors are scored on actual ray-traced samples, not on each
        # model's own predictions (which would be zero by construction).
        rec, denmin, radmin = recover_from_surfaces(coords, chosen, inputs, base)
        forward_err = pred - coords
        inverse_err = rec - base.ref[None, None, :, :]
        result[name] = {
            "predicted_real_grid_mm": pred, "forward_errors_mm": forward_err,
            "recovered_reference_mm": rec, "inverse_errors_mm": inverse_err,
            "corner_forward": grid_metrics(forward_err, inputs["corner_idx"]),
            "noncorner_forward": grid_metrics(forward_err, inputs["noncorner_idx"]),
            "all_forward": grid_metrics(forward_err, np.arange(9)),
            "corner_inverse": grid_metrics(inverse_err, inputs["corner_idx"]),
            "noncorner_inverse": grid_metrics(inverse_err, inputs["noncorner_idx"]),
            "all_inverse": grid_metrics(inverse_err, np.arange(9)),
            "minimum_inverse_denominator": denmin,
            "minimum_inverse_radial_derivative": radmin,
        }
    return result, {"old": old_surfaces, "separable": sep_surfaces, "joint": {
        "sx": new_surfaces["sx"], "sy": new_surfaces["sy"],
        "qx": new_surfaces["qx"], "qy": new_surfaces["qy"]}}


def selected_state_report(inputs, variants):
    a, gaze = inputs["a"], inputs["gaze"]
    ia, ie = int(np.argmin(np.abs(a-5))), int(np.argmin(np.abs(gaze-20)))
    lines = []
    for name, data in variants.items():
        for metric in ("forward_errors_mm", "inverse_errors_mm"):
            errors = data[metric]
            for group, ids in (("corners", inputs["corner_idx"]),
                               ("five noncorner checks", inputs["noncorner_idx"]),
                               ("all nine", np.arange(9))):
                m = selected_state_metrics(errors, ia, ie, ids)
                lines.append(f"A=5 D,E=20 deg {name} {metric.replace('_errors_mm','')} {group}: RMS/max={m['vertex_rms_um']:.6f}/{m['vertex_max_um']:.6f} um.")
    return lines


def grid_plot(ax, xy, color, linestyle="-", linewidth=1.5, zorder=3):
    arr = np.asarray(xy).reshape(3, 3, 2)
    for i in (0, 2):
        ax.plot(arr[i, :, 0], arr[i, :, 1], color=color, ls=linestyle,
                lw=linewidth, zorder=zorder)
        ax.plot(arr[:, i, 0], arr[:, i, 1], color=color, ls=linestyle,
                lw=linewidth, zorder=zorder)
    ax.scatter([arr[1, 1, 0]], [arr[1, 1, 1]], s=17, color=color,
               edgecolors="none", zorder=zorder + 2)


def plot_axes(ax, title, overlay=False):
    axis_scale = common.OVERLAY_AXIS_SCALE if overlay else common.AXIS_SCALE
    ax.set_title(title, fontsize=11 * common.FONT_SCALE, pad=7 * common.FONT_SCALE)
    ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
    ax.set_xticks(common.TICK_VALUES); ax.set_yticks(common.TICK_VALUES)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=.16, lw=.5)
    ax.set_xlabel("x (mm)", fontsize=9 * axis_scale, labelpad=2)
    ax.set_ylabel("y (mm)", fontsize=9 * axis_scale, labelpad=2)
    ax.tick_params(labelsize=8 * axis_scale)


def save_figures(out, inputs, base, joint_surface, variants, fit_metrics):
    a, gaze, coords = inputs["a"], inputs["gaze"], inputs["coords"]
    ref = base.ref
    pred = variants["joint"]["predicted_real_grid_mm"]
    i0, i2, i5 = (int(np.argmin(np.abs(a-x))) for x in (0., 2., 5.))
    im20, iE0, ip20 = (int(np.argmin(np.abs(gaze-x))) for x in (-20., 0., 20.))

    # Stage figure uses the new fixed B and newly fitted joint K.
    fig, axs = plt.subplots(2, 2, figsize=(14, 11))
    barrel5 = base.barrel_all[i5]
    keystone0p20 = pred[i0, ip20]
    panels = [
        (axs[0, 0], "Reference", ref, None, "#667085", "A=0 D, E=0 deg", False),
        (axs[0, 1], "Barrel + scale", barrel5, coords[i5, iE0], common.BARREL_COLOR, "A=5 D, E=0 deg", False),
        (axs[1, 0], "Keystone", keystone0p20, None, common.KEYSTONE_COLOR, "A=0 D, E=20 deg", True),
        (axs[1, 1], "Joint", pred[i5, ip20], coords[i5, ip20], common.JOINT_COLOR, "A=5 D, E=20 deg", True),
    ]
    for ax, title, coords_pred, actual, color, state, swap in panels:
        grid_plot(ax, ref[:, ::-1] if swap else ref, common.REFERENCE_COLOR, "--", 1., 1)
        grid_plot(ax, coords_pred[:, ::-1] if swap else coords_pred, color, "--", 1.6, 4)
        if actual is not None:
            grid_plot(ax, actual[:, ::-1] if swap else actual, common.ACTUAL_COLOR, "-", 1., 3)
        plot_axes(ax, title + "\n" + state)
    stage_handles = [Line2D([0], [0], color=common.REFERENCE_COLOR, ls="--", label="Reference grid"),
                     Line2D([0], [0], color=common.BARREL_COLOR, ls="--", label="Barrel model"),
                     Line2D([0], [0], color=common.KEYSTONE_COLOR, ls="--", label="Keystone model"),
                     Line2D([0], [0], color=common.JOINT_COLOR, ls="--", label="Joint model"),
                     Line2D([0], [0], color=common.ACTUAL_COLOR, ls="-", label="Actual real-ray grid")]
    fig.legend(handles=stage_handles, loc="lower center", bbox_to_anchor=(.5, .04), ncol=3,
               frameon=False, fontsize=8.5*common.FONT_SCALE)
    fig.tight_layout(rect=(0, .12, 1, 1), pad=1.)
    fig.savefig(out / "p4_transformation_stages.png", dpi=200, facecolor="white")
    fig.savefig(out / "p4_transformation_stages.svg", facecolor="white")
    plt.close(fig)

    # Required joint sweep.
    fig, axs = plt.subplots(3, 3, figsize=(15, 14))
    for row, ia in enumerate((i0, i2, i5)):
        for col, ie in enumerate((im20, iE0, ip20)):
            ax = axs[row, col]
            grid_plot(ax, ref[:, ::-1], common.REFERENCE_COLOR, "--", 1., 1)
            grid_plot(ax, coords[ia, ie, :, ::-1], common.ACTUAL_COLOR, "-", 1., 3)
            grid_plot(ax, pred[ia, ie, :, ::-1], common.JOINT_COLOR, "--", 1.6, 4)
            plot_axes(ax, f"A={a[ia]:g} D, E={gaze[ie]:g} deg")
    sweep_handles = [Line2D([0], [0], color=common.REFERENCE_COLOR, ls="--", label="Reference grid"),
                     Line2D([0], [0], color=common.ACTUAL_COLOR, ls="-", label="Actual real-ray grid"),
                     Line2D([0], [0], color=common.JOINT_COLOR, ls="--", label="Coupled joint model")]
    fig.legend(handles=sweep_handles, loc="lower center", bbox_to_anchor=(.5, .035), ncol=3,
               frameon=False, fontsize=8.5*common.FONT_SCALE)
    fig.tight_layout(rect=(0, .09, 1, 1), pad=1.)
    fig.savefig(out / "p4_transformation_sweep.png", dpi=200, facecolor="white")
    fig.savefig(out / "p4_transformation_sweep.svg", facecolor="white")
    plt.close(fig)

    # Accommodation overlay by gaze.
    fig, axs = plt.subplots(1, 3, figsize=(15, 8))
    for col, ie in enumerate((im20, iE0, ip20)):
        ax = axs[col]
        grid_plot(ax, ref[:, ::-1], common.REFERENCE_COLOR, ":", 1.2, 1)
        for ia in (i0, i2, i5):
            color = common.ACCOMMODATION_COLORS[float(a[ia])]
            grid_plot(ax, coords[ia, ie, :, ::-1], color, "-", 1.4, 3)
            grid_plot(ax, pred[ia, ie, :, ::-1], color, "--", 1.6, 4)
        plot_axes(ax, f"E={gaze[ie]:g} deg", overlay=True)
    overlay_handles = [Line2D([0], [0], color=common.ACCOMMODATION_COLORS[float(a[i])], lw=2,
                              label=f"A={a[i]:g} D") for i in (i0, i2, i5)] + [
        Line2D([0], [0], color=common.ACTUAL_COLOR, ls="-", label="Actual real-ray grid"),
        Line2D([0], [0], color=common.ACTUAL_COLOR, ls="--", label="Coupled joint prediction"),
        Line2D([0], [0], color=common.REFERENCE_COLOR, ls=":", label="Reference grid")]
    fig.legend(handles=overlay_handles, loc="lower center", bbox_to_anchor=(.5, .035), ncol=3,
               frameon=False, fontsize=8.5*common.FONT_SCALE)
    fig.tight_layout(rect=(0, .18, 1, 1), pad=1.)
    fig.savefig(out / "p4_transformation_accommodation_overlay.png", dpi=200, facecolor="white")
    fig.savefig(out / "p4_transformation_accommodation_overlay.svg", facecolor="white")
    plt.close(fig)

    # Parameter surfaces: use one shared symmetric scale for qnx/qny.
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    qlim = 1.1 * float(np.max(np.abs(joint_surface["qny"])))
    for ax, key, label in zip(axs.ravel(), ("sx", "sy", "qnx", "qny"),
                              ("$s_x(A,E)$", "$s_y(A,E)$", "$q_{n,x}(A,E)$", "$q_{n,y}(A,E)$")):
        surface = joint_surface[key]
        vmin, vmax = ((-qlim, qlim) if key in ("qnx", "qny") else (float(surface.min()), float(surface.max())))
        img = ax.pcolormesh(gaze, a, surface, shading="auto", cmap="coolwarm" if key in ("qnx", "qny") else "viridis", vmin=vmin, vmax=vmax)
        ax.set_title(label, fontsize=11*common.FONT_SCALE)
        ax.set_xlabel("E (deg)", fontsize=9*common.AXIS_SCALE)
        ax.set_ylabel("A (D)", fontsize=9*common.AXIS_SCALE)
        ax.tick_params(labelsize=8*common.AXIS_SCALE)
        fig.colorbar(img, ax=ax, shrink=.84)
    fig.tight_layout(pad=1.)
    fig.savefig(out / "joint_parameter_surfaces.png", dpi=180, facecolor="white")
    fig.savefig(out / "joint_parameter_surfaces.svg", facecolor="white")
    plt.close(fig)

    # Corner inverse residual surface.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    image = ax.pcolormesh(inputs["gaze"], inputs["a"], fit_metrics["corner_rms_by_state_mm"]*1000,
                          shading="auto", cmap="magma")
    fig.colorbar(image, ax=ax, label="Four-corner inverse RMS (um)")
    ax.set_xlabel("E (deg)"); ax.set_ylabel("A (D)")
    ax.set_title("Coupled joint inverse-fit corner residual")
    fig.tight_layout(pad=1.)
    fig.savefig(out / "inverse_corner_rms.png", dpi=180, facecolor="white")
    fig.savefig(out / "inverse_corner_rms.svg", facecolor="white")
    plt.close(fig)


def build_artifact(inputs, result, base, objective, p0, lower, upper, active_lo, active_hi,
                   joint_surface, variants, staged_forward_error):
    sx, sy, qnx, qny = (joint_surface[k] for k in ("sx", "sy", "qnx", "qny"))
    fit_resid = objective.residual(result.x)[:objective.rays.size].reshape(objective.rays.shape)
    corner_norm = np.linalg.norm(fit_resid / 1000., axis=-1)
    state_rms = np.sqrt(np.mean(corner_norm**2, axis=-1)).reshape(base.na, base.ng)
    metrics = {
        "inverse_fit_corner": common.point_metric_summary(fit_resid / 1000.),
        "forward_corner": variants["joint"]["corner_forward"],
        "forward_five_noncorner_checks": variants["joint"]["noncorner_forward"],
        "forward_all_nine": variants["joint"]["all_forward"],
        "inverse_five_noncorner_checks": variants["joint"]["noncorner_inverse"],
        "inverse_all_nine": variants["joint"]["all_inverse"],
        "minimum_projective_denominator": variants["joint"]["minimum_inverse_denominator"],
        "minimum_radial_derivative": variants["joint"]["minimum_inverse_radial_derivative"],
    }
    return {
        "schema": SCHEMA,
        "model_definition": "F(A,E)=K_E(B_A(r0)); B_A=D_k1(A) o S_s(A) o D_k1(0)^-1; K has diagonal sx/sy numerator and independent qnx/qny linear denominator",
        "fit_parameterization": "sx/sy=1+sum_l=0..2 sum_j=1..4 c_lj(A/5)^l(E/20)^j; qnx/qny=sum_l=0..2 sum_j=1..4 c_lj(A/5)^l(E/20)^j",
        "coefficient_order": [f"{family}_l{l}_j{j}" for family in ("sx", "sy", "qnx", "qny") for l in range(3) for j in range(1, 5)],
        "coefficients_normalized": result.x.copy(),
        "normalization_L_mm": inputs["L"],
        "accommodation_d": inputs["a"].copy(), "eye_rotation_deg": inputs["gaze"].copy(),
        "radial_k1_power_coefficients_c0_c1_n": inputs["k_coefficients"].copy(),
        "paraxial_scale_vs_a0": base.scale_a.copy(),
        "reference_all_fields_mm": base.ref.copy(),
        "reference_corners_mm": base.ref[inputs["corner_idx"]].copy(),
        "corner_field_ids": inputs["fields"][inputs["corner_idx"]].copy(),
        "noncorner_field_ids": inputs["fields"][inputs["noncorner_idx"]].copy(),
        "parameter_surfaces": {"sx": sx.copy(), "sy": sy.copy(), "qnx": qnx.copy(), "qny": qny.copy()},
        "fit_corner_inverse_rms_by_state_mm": state_rms,
        "fit_corner_inverse_residuals_mm": fit_resid / 1000.,
        "metrics": metrics,
        "variant_comparison_metrics": {name: {k: v for k, v in data.items() if isinstance(v, dict) or k.startswith("minimum_")}
                                       for name, data in variants.items()},
        "staged_vs_integrated_forward_max_difference_mm": staged_forward_error,
        "fit_success": bool(result.success), "fit_status": int(result.status), "fit_message": str(result.message),
        "fit_nfev": int(result.nfev), "fit_cost": float(result.cost), "fit_optimality": float(result.optimality),
        "active_lower_bound_indices": active_lo, "active_upper_bound_indices": active_hi,
        "optimizer": {"method": "scipy.optimize.least_squares(method='trf')", "jacobian": "vectorized residual forward differences, absolute step 1e-5",
                      "max_nfev": int(result.nfev), "lower_bounds": lower, "upper_bounds": upper,
                      "initial_coefficients": p0},
        "provenance": {
            "grid_path": str(common.GRID_PATH.resolve()), "grid_sha256": inputs["grid_sha256"],
            "old_joint_model_path": str(common.OLD_MODEL_PATH.resolve()), "old_joint_model_sha256": inputs["old_sha256"],
            "radial_fit_sha256": inputs["radial_fit_sha256"], "radial_models_sha256": inputs["radial_models_sha256"],
            "radial_metrics_sha256": inputs["radial_metrics_sha256"],
            "separable_model_path": str((ROOT / "data" / "p4_separable_transform" / "p4_separable_transform.pkl").resolve()),
            "ray_type": "real only", "z_mm": 0., "fit_points": "four corners in every 51 x 401 state",
            "checks": "five noncorner samples in every state; excluded from fit objective",
        },
    }


def make_report(artifact, variants, inputs, selected):
    def fmt(label, stats):
        return (f"{label}: vertex RMS/max={stats['vertex_rms_um']:.6f}/{stats['vertex_max_um']:.6f} um; "
                f"coordinate RMSE={stats['coordinate_rmse_um']:.6f} um; "
                f"mean/max state RMS={stats['mean_state_vertex_rms_um']:.6f}/{stats['max_state_vertex_rms_um']:.6f} um.")
    p = artifact["coefficients_normalized"].reshape(4, 3, 4)
    c0, c1, n = inputs["k_coefficients"]
    lines = [
        "# P4 coupled physical-barrel and two-q keystone fit", "",
        "Fit uses four real-ray corners from every one of the 51 x 401 simulated accommodation/gaze states. All five noncorner fields are checks and are excluded from the objective; no train/test split is used. The existing canonical ray grid, v2 joint fit, radial fit, and separable fit artifacts are read-only.", "",
        "## Integrated transformation", "",
        "Let $D_k(\\mathbf u)=(1+k\\|\\mathbf u\\|^2)\\mathbf u$, $k_0=k_1(0)$, $w=s(A)D_{k_0}^{-1}(\\mathbf r_0)$, and $\\gamma=1+k_1(A)\\|w\\|^2$. The complete physical forward map is implemented as the single expression", "",
        "$$", "F(A,E;\\mathbf r_0)=\\frac{\\gamma\\begin{bmatrix}s_x(A,E)w_x\\\\s_y(A,E)w_y\\end{bmatrix}}{1+\\gamma\\left(q_{n,x}(A,E)w_x/L+q_{n,y}(A,E)w_y/L\\right)}.", "$$", "",
        "Equivalently $F=K_{A,E}(B_A(\\mathbf r_0))$, where $B_A=D_{k_1(A)}\\circ S_{s(A)}\\circ D_{k_0}^{-1}$ and $K$ has diagonal numerator $\\operatorname{diag}(s_x,s_y)$ and both $q_xx_b+q_yy_b$ denominator terms. Physical $q_x=q_{n,x}/L$, $q_y=q_{n,y}/L$. The single simulated gaze variable E remains one axis; no separate physical $E_x,E_y$ interpretation is inferred.", "",
        "Inverse correction applies $K_{A,E}^{-1}$ first, then $B_A^{-1}$, and minimizes recovered-corner coordinates against the fixed real A=0,E=0 reference. The inverse projective branch must have positive denominator; the radial inverse stays on $1+3k\\rho^2>0$.", "",
        "For each parameter family independently, $f(A,E)=\\sum_{l=0}^2\\sum_{j=1}^4c_{lj}(A/5)^l(E/20)^j$. Then $s_x=1+f_{s_x}$, $s_y=1+f_{s_y}$, and $q_{n,x}=f_{q_{n,x}}$, $q_{n,y}=f_{q_{n,y}}$. This gives 48 coefficients and exact K identity at E=0.", "",
        "## Fit and validation", "",
        f"Converged={artifact['fit_success']} ({artifact['fit_message']}), nfev={artifact['fit_nfev']}, first-order optimality={artifact['fit_optimality']:.6g}, objective cost={artifact['fit_cost']:.9g}. Active coefficient bounds (0-based) lower/upper: {artifact['active_lower_bound_indices'].tolist()} / {artifact['active_upper_bound_indices'].tolist()}.",
        f"The integrated expression and staged B-then-K composition agree over every A/E state and all nine reference fields to max {artifact['staged_vs_integrated_forward_max_difference_mm']:.3g} mm. New forward predictions round-trip through inverse K then inverse B to max {variants['joint']['roundtrip_max_mm']:.3g} mm. Minimum inverse projective denominator={variants['joint']['minimum_inverse_denominator']:.8g}; minimum radial derivative={variants['joint']['minimum_inverse_radial_derivative']:.8g}.", "",
        fmt("New coupled inverse-fit corners", artifact["metrics"]["inverse_fit_corner"]),
        fmt("New coupled forward corners", artifact["metrics"]["forward_corner"]),
        fmt("New coupled forward five noncorner checks", artifact["metrics"]["forward_five_noncorner_checks"]),
        fmt("New coupled forward all nine", artifact["metrics"]["forward_all_nine"]),
        fmt("New coupled inverse five noncorner checks", artifact["metrics"]["inverse_five_noncorner_checks"]), "",
        "## Same-B model comparison", "",
        "Every variant below uses the identical corrected physical $B_A$, canonical ray data, and all 51 x 401 states. `old_hybrid` reuses A/E-dependent v2 K; `separable` reuses the new gaze-only 16-coefficient K; `joint` is the new 48-coefficient K(A,E). RMS/max are over actual grid vertices.", "",
    ]
    for name in ("old_hybrid", "separable", "joint"):
        data = variants[name]
        lines += [f"### {name}", "", fmt("Forward corners", data["corner_forward"]),
                  fmt("Forward five noncorner checks", data["noncorner_forward"]),
                  fmt("Forward all nine", data["all_forward"]),
                  fmt("Inverse corners", data["corner_inverse"]),
                  fmt("Inverse five noncorner checks", data["noncorner_inverse"]),
                  fmt("Inverse all nine", data["all_inverse"]), ""]
    lines += ["## A=5 D, E=20 deg pointwise-state comparison", ""] + selected + ["", "## Fitted coefficient table", "",
              "Rows are independent parameter families; columns are accommodation powers l=0,1,2, each containing gaze powers j=1..4:", ""]
    for i, name in enumerate(("sx-1", "sy-1", "qnx", "qny")):
        lines += [f"### {name}", "", "| l | j=1 | j=2 | j=3 | j=4 |", "|---:|---:|---:|---:|---:|"]
        for l in range(3):
            lines.append(f"| {l} | " + " | ".join(f"{v:.12g}" for v in p[i, l]) + " |")
        lines.append("")
    lines += [f"Saved physical barrel: $k_1(A)={c1:.12g}A^{{{n:.12g}}}{c0:+.12g}$ mm$^{{-2}}$; saved $s(0)={artifact['paraxial_scale_vs_a0'][0]:.12g}$ and $s(5)={artifact['paraxial_scale_vs_a0'][-1]:.12g}$.",
              f"The fitted $q_{{n,x}}$ surface max absolute value is {np.max(np.abs(artifact['parameter_surfaces']['qnx'])):.6g}; the fitted $q_{{n,y}}$ surface max absolute value is {np.max(np.abs(artifact['parameter_surfaces']['qny'])):.6g}.", "",
              "Figures include the 3 x 3 state sweep, A-overlay by E, stages, A/E coefficient surfaces, and inverse corner residual surface. Keystone panels swap physical x/y for display only. Each grid shows its outer perimeter and center dot; middle native cross-lines and other markers are omitted.", "",
              "The fit is empirical over this one-angle P4 simulation domain and does not establish independent physical rotation axes or performance beyond the sampled conditions.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-nfev", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    inputs = common.load_inputs()
    separable = pickle.loads((ROOT / "data" / "p4_separable_transform" / "p4_separable_transform.pkl").read_bytes())
    if separable.get("schema") != "p4-separable-corner-inverse-v1":
        raise ValueError("Expected the current gaze-only separable artifact for comparison")
    result, base, objective, p0, lower, upper, active_lo, active_hi = fit_joint(inputs, args.max_nfev)
    params = state_parameter_surfaces(result.x, inputs["a"], inputs["gaze"])
    # Audit the explicit integrated expression against staged B_A followed by K_E at all fields.
    integrated = forward_all(base.ref, result.x, inputs, base)
    staged = staged_forward(base.ref, np.repeat(inputs["a"][:, None], len(inputs["gaze"]), axis=1),
                            np.repeat(inputs["gaze"][None, :], len(inputs["a"]), axis=0),
                            base.L, inputs["k_coefficients"], base.scale_a[:, None], result.x)
    staged_difference = float(np.max(np.abs(integrated - staged)))
    if staged_difference > 1e-12:
        raise RuntimeError(f"Integrated/staged forward formulas disagree: {staged_difference:g} mm")
    variants, surfaces = evaluate_variants(inputs, base, result.x, separable)
    # Explicit full-grid roundtrip check on the new integrated forward prediction.
    rec, pred_den, pred_rad = inverse_joint(variants["joint"]["predicted_real_grid_mm"],
                                            np.repeat(inputs["a"][:, None], len(inputs["gaze"]), axis=1),
                                            np.repeat(inputs["gaze"][None, :], len(inputs["a"]), axis=0),
                                            base.L, inputs["k_coefficients"], base.scale_a[:, None], result.x)
    roundtrip_error = float(np.max(np.linalg.norm(rec - base.ref[None, None, :, :], axis=-1)))
    if roundtrip_error > 1e-9:
        raise RuntimeError(f"Forward/inverse all-nine roundtrip is too large: {roundtrip_error:g} mm")
    # Fitted inverse objective residuals and the optimizer cost must agree exactly (penalties inactive).
    residual = objective.residual(result.x)
    corner_resid = residual[:objective.rays.size]
    cost_recomputed = .5 * float(corner_resid @ corner_resid)
    if abs(cost_recomputed - float(result.cost)) > 1e-6 * max(1., float(result.cost)):
        raise RuntimeError("Saved least-squares cost does not equal the corner residual objective")
    agrid, egrid = np.meshgrid(inputs["a"], inputs["gaze"], indexing="ij")
    sx, sy, qnx, qny = unpack_joint(result.x, agrid, egrid)
    if np.any(sx <= 0) or np.any(sy <= 0):
        raise RuntimeError("Fitted scale surface left positive domain")
    if not np.allclose([sx[0, 200], sy[0, 200], qnx[0, 200], qny[0, 200]], [1., 1., 0., 0.], atol=1e-12, rtol=0):
        raise RuntimeError("Keystone does not equal identity at E=0")
    # Add domain/roundtrip metrics from predicted and observed all-nine grids.
    variants["joint"]["roundtrip_max_mm"] = roundtrip_error
    variants["joint"]["minimum_predicted_inverse_denominator"] = float(np.min(pred_den))
    variants["joint"]["minimum_predicted_inverse_radial_derivative"] = float(np.min(pred_rad))
    # Attach pointwise-state diagnostics for the requested A5,E20 state.
    selected = selected_state_report(inputs, variants)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    artifact = build_artifact(inputs, result, base, objective, p0, lower, upper, active_lo, active_hi,
                              params, variants, staged_difference)
    artifact["fit_cost_recomputed_from_corner_residuals"] = cost_recomputed
    artifact["fit_corner_inverse_residuals_um"] = corner_resid.reshape(objective.rays.shape)
    artifact["predicted_inverse_roundtrip_max_mm"] = roundtrip_error
    artifact["minimum_predicted_inverse_denominator_all_nine"] = float(np.min(pred_den))
    artifact["minimum_predicted_inverse_radial_derivative_all_nine"] = float(np.min(pred_rad))
    artifact["variant_arrays"] = {name: {k: v for k, v in data.items() if k.endswith("_mm") and isinstance(v, np.ndarray)}
                                  for name, data in variants.items()}
    with (out / "p4_joint_power_transform.pkl").open("wb") as stream:
        pickle.dump(artifact, stream, pickle.HIGHEST_PROTOCOL)
    save_figures(out, inputs, base, params, variants, {"corner_rms_by_state_mm": artifact["fit_corner_inverse_rms_by_state_mm"]})
    (out / "report.md").write_text(make_report(artifact, variants, inputs, selected), encoding="utf-8")
    print(f"Schema: {SCHEMA}")
    print(f"Converged={result.success}, nfev={result.nfev}, cost={result.cost:.9g}, optimality={result.optimality:.6g}, active bounds={len(active_lo)+len(active_hi)}")
    print(f"Inverse-fit corner RMS/max={artifact['metrics']['inverse_fit_corner']['vertex_rms_um']:.6f}/{artifact['metrics']['inverse_fit_corner']['vertex_max_um']:.6f} um")
    print(f"Forward all-nine RMS/max coupled={variants['joint']['all_forward']['vertex_rms_um']:.6f}/{variants['joint']['all_forward']['vertex_max_um']:.6f} um; old={variants['old_hybrid']['all_forward']['vertex_rms_um']:.6f}/{variants['old_hybrid']['all_forward']['vertex_max_um']:.6f} um; separable={variants['separable']['all_forward']['vertex_rms_um']:.6f}/{variants['separable']['all_forward']['vertex_max_um']:.6f} um")
    print(f"Roundtrip max={roundtrip_error:.3g} mm; staged/integrated max difference={staged_difference:.3g} mm; output={out}")


if __name__ == "__main__":
    main()
