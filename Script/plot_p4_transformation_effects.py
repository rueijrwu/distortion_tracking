"""Plot staged P4 barrel and keystone transformations from saved real-ray data."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "distortion_grid_p4" / "distortion_grid.pkl"
DEFAULT_MODEL = ROOT / "data" / "p4_joint_corner_transform" / "p4_joint_corner_transform.pkl"
DEFAULT_RADIAL_FIT = ROOT / "data" / "p4_radial_distortion_fit" / "p4_radial_distortion_fit.pkl"
DEFAULT_RADIAL_MODELS = ROOT / "data" / "p4_radial_distortion_fit" / "p4_radial_distortion_accommodation_models.pkl"
DEFAULT_RADIAL_METRICS = ROOT / "data" / "p4_radial_distortion_fit" / "p4_radial_distortion_metrics.pkl"
DEFAULT_OUTPUT = ROOT / "data" / "p4_transformation_effects"
CORNER_IDS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
REF_COLOR = "#9aa1ab"
ACT_COLOR = "#15191f"
COLORS = {"barrel": "#d97706", "key": "#2b6cb0", "joint": "#16805d"}
FONT_SCALE = 1.5
AXIS_LABEL_SCALE = 1.5 * FONT_SCALE
OVERLAY_AXIS_LABEL_SCALE = 1.5 * AXIS_LABEL_SCALE


def load_grid(path: Path):
    d = pickle.loads(path.read_bytes())
    if not isinstance(d, np.ndarray) or d.dtype.names is None:
        raise ValueError("Input must be the structured real-ray distortion grid")
    needed = {"accommodation_d", "eye_rotation_deg", "field_x_relative", "field_y_relative", "real_x_mm", "real_y_mm", "z_mm"}
    if not needed.issubset(d.dtype.names):
        raise ValueError(f"Grid is missing fields: {sorted(needed - set(d.dtype.names))}")
    if not np.allclose(d["z_mm"], 0, atol=1e-9, rtol=0):
        raise ValueError("Expected P4 real-ray data at Z=0")
    fields = np.unique(np.column_stack((d["field_x_relative"], d["field_y_relative"])), axis=0)
    if fields.shape != (9, 2):
        raise ValueError("Expected nine native 3x3 field points")
    a, theta = np.unique(d["accommodation_d"]), np.unique(d["eye_rotation_deg"])
    ai = np.searchsorted(a, d["accommodation_d"])
    ti = np.searchsorted(theta, d["eye_rotation_deg"])
    fields = fields[np.lexsort((fields[:, 1], fields[:, 0]))]
    order = np.lexsort((d["field_y_relative"], d["field_x_relative"], ti, ai))
    s = d[order]
    coords = np.column_stack((s["real_x_mm"], s["real_y_mm"])).reshape(len(a), len(theta), 9, 2)
    fcheck = np.column_stack((s["field_x_relative"], s["field_y_relative"])).reshape(len(a), len(theta), 9, 2)
    if not np.array_equal(fcheck[0, 0], fields) or not np.all(fcheck == fields[None, None, :, :]):
        raise ValueError("Field ordering or identity could not be verified")
    return a, theta, fields, coords


def parameter(fit, a, theta):
    aa, tt = np.broadcast_arrays(np.asarray(a, float), np.asarray(theta, float))
    p = np.asarray(fit["coefficients_normalized"], float)
    ao, go, bo = int(fit["accommodation_order"]), int(fit["gaze_order"]), int(fit["beta_order"])
    k = 0
    beta = np.zeros(aa.shape)
    for l in range(1, bo + 1):
        beta += p[k] * (aa / 5) ** l
        k += 1
    vals = []
    for ip in range(3):
        v = np.ones(aa.shape) if ip < 2 else np.zeros(aa.shape)
        for l in range(ao + 1):
            for j in range(1, go + 1):
                v += p[k] * (aa / 5) ** l * (tt / 20) ** j
                k += 1
        vals.append(v)
    if k != len(p):
        raise ValueError("Coefficient vector length does not match saved model orders")
    return beta, *vals


def idx_exact(values, value, label):
    hits = np.flatnonzero(np.isclose(values, value, atol=1e-10, rtol=0))
    if len(hits) != 1:
        raise ValueError(f"Requested {label}={value} is not uniquely stored")
    return int(hits[0])


def model_transform(ref_mm, L, beta, sx, sy, q):
    u = ref_mm / L
    barrel = (1 + beta * np.sum(u * u, axis=-1))[..., None] * u
    den = 1 + q * barrel[..., 1]
    return L * np.stack((sx * barrel[..., 0] / den, sy * barrel[..., 1] / den), axis=-1)


def invert_radial(real_mm, k1):
    """Invert r_real=r_para*(1+k1*r_para^2) on the physical near-axis branch."""
    real_mm = np.asarray(real_mm, float)
    r_real = np.linalg.norm(real_mm, axis=-1)
    r_para = r_real.copy()
    for _ in range(30):
        f = r_para * (1 + k1 * r_para**2) - r_real
        fp = 1 + 3 * k1 * r_para**2
        step = f / fp
        r_para -= step
        if np.max(np.abs(step)) < 1e-14:
            break
    if np.any(r_para < 0) or np.any(1 + 3*k1*r_para**2 <= 0):
        raise ValueError("Baseline radial inversion left the monotonic near-axis branch")
    ratio = np.divide(r_para, r_real, out=np.ones_like(r_para), where=r_real > 0)
    return real_mm * ratio[..., None]


def apply_radial(paraxial_mm, k1):
    paraxial_mm = np.asarray(paraxial_mm, float)
    r2 = np.sum(paraxial_mm**2, axis=-1)
    return paraxial_mm * (1 + k1 * r2)[..., None]


def power_k1(models, accommodation):
    row = models[models["model"] == "power_law"]
    if len(row) != 1:
        raise ValueError("Expected exactly one saved power_law accommodation model")
    row = row[0]
    return row["coefficient_0"] + row["coefficient_1"] * np.power(accommodation, row["coefficient_2"])


def power_barrel(ref, accommodation, radial_results, power_models, radial_metrics):
    """B_A = D_k(A) o S_s(A) o D_k(0)^-1, anchored at the saved real A=0 grid."""
    a_values = radial_results["accommodation_d"]
    hits = np.flatnonzero(np.isclose(a_values, accommodation, atol=1e-10, rtol=0))
    if len(hits) != 1:
        raise ValueError(f"Accommodation {accommodation:g} is not in saved radial fit")
    idx = int(hits[0])
    scale = float(radial_metrics[idx]["paraxial_scale_vs_a0"])
    k0 = float(power_k1(power_models, 0.0))
    kA = float(power_k1(power_models, accommodation))
    para0 = invert_radial(ref, k0)
    return apply_radial(scale * para0, kA), scale, k0, kA


def error_summary(pred, actual, corners, other):
    e = np.linalg.norm(pred - actual, axis=1)
    def summary(ids):
        return (float(np.sqrt(np.mean(e[ids]**2))*1000), float(np.max(e[ids])*1000))
    return summary(corners), summary(other), summary(np.arange(len(e)))


def plot_grid(ax, coords, color, linestyle="-", lw=1.5, zorder=3):
    """Draw the outer native-grid perimeter and center sample, omitting middle cross-lines."""
    arr = np.asarray(coords).reshape(3, 3, 2)
    for i in (0, 2):
        ax.plot(arr[i, :, 0], arr[i, :, 1], color=color, lw=lw, ls=linestyle, zorder=zorder)
        ax.plot(arr[:, i, 0], arr[:, i, 1], color=color, lw=lw, ls=linestyle, zorder=zorder)
    ax.scatter([arr[1, 1, 0]], [arr[1, 1, 1]], s=17, color=color, edgecolors="none", zorder=zorder + 2)


def transpose_coords(coords):
    return np.asarray(coords)[..., ::-1]


def corner_and_inner(fields):
    corners = np.array([np.flatnonzero(np.all(fields == c, axis=1))[0] for c in CORNER_IDS])
    inner = np.array([i for i in range(9) if i not in corners])
    return corners, inner


def draw_model(ax, coords, fields, color, linestyle="-", markers=True):
    plot_grid(ax, coords, color, linestyle, 1.6, 4)


def draw_actual(ax, coords):
    plot_grid(ax, coords, ACT_COLOR, "-", 1.0, 3)


def setup_ax(ax, lim, title, axis_scale=AXIS_LABEL_SCALE):
    ax.set_title(title, fontsize=11 * FONT_SCALE, pad=7 * FONT_SCALE)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=.16, lw=.5)
    ax.set_xticks((-2, -1, 0, 1, 2))
    ax.set_yticks((-2, -1, 0, 1, 2))
    ax.set_xlabel("x (mm)", fontsize=9 * axis_scale, labelpad=2)
    ax.set_ylabel("y (mm)", fontsize=9 * axis_scale, labelpad=2)
    ax.tick_params(labelsize=8 * axis_scale)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--radial-fit", type=Path, default=DEFAULT_RADIAL_FIT)
    ap.add_argument("--radial-models", type=Path, default=DEFAULT_RADIAL_MODELS)
    ap.add_argument("--radial-metrics", type=Path, default=DEFAULT_RADIAL_METRICS)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()
    raw = pickle.loads(args.model.read_bytes())
    if raw.get("schema") != "p4-joint-corner-inverse-v2" or raw.get("fit", {}).get("model") != "joint":
        raise ValueError("Expected the saved p4-joint-corner-inverse-v2 joint fit")
    a, theta, fields, coords = load_grid(args.input)
    radial_results = pickle.loads(args.radial_fit.read_bytes())
    power_models = pickle.loads(args.radial_models.read_bytes())
    radial_metrics = pickle.loads(args.radial_metrics.read_bytes())["metrics"]
    if not isinstance(radial_results, np.ndarray) or "k1_per_mm2" not in (radial_results.dtype.names or ()):
        raise ValueError("Expected saved per-accommodation all-grid radial-fit results")
    if len(radial_metrics) != len(radial_results):
        raise ValueError("Radial fit and scale metrics do not align")
    L, fit = float(raw["normalization_L_mm"]), raw["fit"]
    ref = coords[idx_exact(a, 0, "reference accommodation"), idx_exact(theta, 0, "reference gaze")].copy()
    corners, _ = corner_and_inner(fields)
    if not np.allclose(ref[corners], raw["reference_corners_mm"], atol=2e-10, rtol=0):
        raise ValueError("Input reference corners disagree with saved model")
    ai5, ti0, ti20 = idx_exact(a, 5, "accommodation"), idx_exact(theta, 0, "gaze"), idx_exact(theta, 20, "gaze")
    actual50, actual520 = coords[ai5, ti0].copy(), coords[ai5, ti20].copy()
    beta5 = float(parameter(fit, np.array([5.]), np.array([0.]))[0][0])
    old_barrel5 = model_transform(ref, L, beta5, 1., 1., 0.)
    pbarrel, scale5, k0_power, k5_power = power_barrel(ref, 5., radial_results, power_models, radial_metrics)
    pbarrel0, scale0, _, k0_power_check = power_barrel(ref, 0., radial_results, power_models, radial_metrics)
    if not np.allclose(pbarrel0, ref, atol=2e-12, rtol=0):
        raise ValueError("Power-law baseline mapping failed identity anchoring at A=0")
    beta, sx, sy, q = parameter(fit, np.array([5.]), np.array([20.]))
    pkeystone = model_transform(ref, L, 0., float(sx[0]), float(sy[0]), float(q[0]))
    pjoint = model_transform(pbarrel, L, 0., float(sx[0]), float(sy[0]), float(q[0]))
    old_pjoint = model_transform(ref, L, float(beta[0]), float(sx[0]), float(sy[0]), float(q[0]))
    para0_roundtrip = apply_radial(invert_radial(ref, k0_power), k0_power)
    all_points = np.concatenate([ref, actual50, actual520, pbarrel, pkeystone, pjoint])
    lim = (-2., 2.)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Four explanatory stages.
    fig, axs = plt.subplots(2, 2, figsize=(14, 11))
    panels = [
        (axs[0, 0], "Reference", ref, None, "#667085", "A=0 D, E=0 deg"),
        (axs[0, 1], "Barrel + scale", pbarrel, actual50, COLORS["barrel"], "A=5 D, E=0 deg"),
        (axs[1, 0], "Keystone", pkeystone, None, COLORS["key"], "A=5 D, E=20 deg"),
        (axs[1, 1], "Joint", pjoint, actual520, COLORS["joint"], "A=5 D, E=20 deg"),
    ]
    for ax, title, pred, actual, color, state in panels:
        swapped = title.startswith("Keystone") or title.startswith("Joint")
        visual_ref = transpose_coords(ref) if swapped else ref
        visual_fields = fields[:, ::-1] if swapped else fields
        plot_grid(ax, visual_ref, REF_COLOR, "--", 1.0, 1)

        visual_pred = transpose_coords(pred) if swapped else pred
        draw_model(ax, visual_pred, visual_fields, color)
        if actual is not None:
            draw_actual(ax, transpose_coords(actual) if swapped else actual)
        setup_ax(ax, lim, title + "\n" + state)
    handles = [
        Line2D([0], [0], color=REF_COLOR, ls="--", lw=1, label="Reference grid"),
        Line2D([0], [0], color=COLORS["barrel"], lw=1.6, label="Barrel model"),
        Line2D([0], [0], color=COLORS["key"], lw=1.6, label="Keystone model"),
        Line2D([0], [0], color=COLORS["joint"], lw=1.6, label="Joint model"),
        Line2D([0], [0], color=ACT_COLOR, ls="-", label="Actual real-ray grid"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .04), ncol=3, frameon=False, fontsize=8.5 * FONT_SCALE)
    fig.tight_layout(rect=(0, .12, 1, 1), pad=1.0)
    main_png = args.output_dir / "p4_transformation_stages.png"
    fig.savefig(main_png, dpi=200, facecolor="white")
    fig.savefig(args.output_dir / "p4_transformation_stages.svg", facecolor="white")
    plt.close(fig)

    # Joint-model gaze sweep at each accommodation.
    fig, axs = plt.subplots(3, 3, figsize=(15, 14))
    sweep_points = [ref]
    for row, av in enumerate((0., 2., 5.)):
        ia = idx_exact(a, av, "accommodation")
        stage_barrel, _, _, _ = power_barrel(ref, av, radial_results, power_models, radial_metrics)
        for col, tv in enumerate((-20., 0., 20.)):
            it = idx_exact(theta, tv, "gaze")
            act = coords[ia, it]
            _, sx, sy, q = parameter(fit, np.array([av]), np.array([tv]))
            pred = model_transform(stage_barrel, L, 0., float(sx[0]), float(sy[0]), float(q[0]))
            sweep_points.extend([act, pred])
            ax = axs[row, col]
            visual_ref, visual_act, visual_pred = transpose_coords(ref), transpose_coords(act), transpose_coords(pred)
            plot_grid(ax, visual_ref, REF_COLOR, "--", 1.0, 1)
            draw_actual(ax, visual_act)
            draw_model(ax, visual_pred, fields[:, ::-1], COLORS["joint"], "--")
            setup_ax(ax, lim, f"A={av:g} D, E={tv:g} deg")

    handles = [
        Line2D([0], [0], color=REF_COLOR, ls="--", lw=1, label="Reference grid"),
        Line2D([0], [0], color=ACT_COLOR, ls="-", lw=1, label="Actual real-ray grid"),
        Line2D([0], [0], color=COLORS["joint"], ls="--", lw=1.6, label="Joint model"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .035), ncol=3, frameon=False, fontsize=8.5 * FONT_SCALE)
    fig.tight_layout(rect=(0, .09, 1, 1), pad=1.0)
    sweep_png = args.output_dir / "p4_transformation_sweep.png"
    fig.savefig(sweep_png, dpi=200, facecolor="white")
    fig.savefig(args.output_dir / "p4_transformation_sweep.svg", facecolor="white")
    plt.close(fig)

    # Overlay accommodations at each gaze angle in a compact three-panel figure.
    fig, axs = plt.subplots(1, 3, figsize=(15, 8))
    accommodation_colors = {0.: "#2b6cb0", 2.: "#d97706", 5.: "#16805d"}
    for col, tv in enumerate((-20., 0., 20.)):
        ax = axs[col]
        it = idx_exact(theta, tv, "gaze")
        visual_ref = transpose_coords(ref)
        plot_grid(ax, visual_ref, REF_COLOR, ":", 1.2, 1)
        for av in (0., 2., 5.):
            ia = idx_exact(a, av, "accommodation")
            actual = transpose_coords(coords[ia, it])
            stage_barrel, _, _, _ = power_barrel(ref, av, radial_results, power_models, radial_metrics)
            _, sx, sy, q = parameter(fit, np.array([av]), np.array([tv]))
            predicted = model_transform(stage_barrel, L, 0., float(sx[0]), float(sy[0]), float(q[0]))
            predicted = transpose_coords(predicted)
            color = accommodation_colors[av]
            plot_grid(ax, actual, color, "-", 1.4, 3)
            plot_grid(ax, predicted, color, "--", 1.6, 4)
        setup_ax(ax, lim, f"E={tv:g} deg", axis_scale=OVERLAY_AXIS_LABEL_SCALE)
    overlay_handles = [
        Line2D([0], [0], color=accommodation_colors[av], lw=2,
               label=f"A={av:g} D") for av in (0., 2., 5.)
    ] + [
        Line2D([0], [0], color=ACT_COLOR, ls="-", lw=1.5, label="Actual real-ray grid"),
        Line2D([0], [0], color=ACT_COLOR, ls="--", lw=1.6, label="Joint prediction"),
        Line2D([0], [0], color=REF_COLOR, ls=":", lw=1.2, label="Reference grid"),
    ]
    fig.legend(handles=overlay_handles, loc="lower center", bbox_to_anchor=(.5, .035),
               ncol=3, frameon=False, fontsize=8.5 * FONT_SCALE)
    fig.tight_layout(rect=(0, .18, 1, 1), pad=1.0)
    overlay_png = args.output_dir / "p4_transformation_accommodation_overlay.png"
    fig.savefig(overlay_png, dpi=200, facecolor="white")
    fig.savefig(args.output_dir / "p4_transformation_accommodation_overlay.svg", facecolor="white")
    plt.close(fig)

    other = np.array([i for i in range(9) if i not in corners])
    new_joint_err = error_summary(pjoint, actual520, corners, other)
    old_joint_err = error_summary(old_pjoint, actual520, corners, other)
    new_barrel_err = error_summary(pbarrel, actual50, corners, other)
    old_barrel_err = error_summary(old_barrel5, actual50, corners, other)
    barrel_change = error_summary(pbarrel, old_barrel5, corners, other)
    roundtrip_max = float(np.max(np.linalg.norm(para0_roundtrip-ref, axis=1))*1000)
    power_row = power_models[power_models["model"] == "power_law"][0]
    def fmt_stats(label, stats):
        c, n, alln = stats
        return f"{label}: corners RMS/max={c[0]:.6f}/{c[1]:.6f} um; five other points RMS/max={n[0]:.6f}/{n[1]:.6f} um; all-nine RMS/max={alln[0]:.6f}/{alln[1]:.6f} um."
    report = [
        "# P4 transformation effect figures", "",
        "Generated from saved Z=0 real-ray coordinates, the saved all-grid radial fit, saved paraxial scale metrics, and saved v2 joint-fit K parameters. No CODE V run or fit was performed during plotting; the saved K parameters were not refit.", "",
        f"Power law: k1(A)=c1*A^n+c0 with c0={power_row['coefficient_0']:.12g} mm^-2, c1={power_row['coefficient_1']:.12g} mm^-2/D^n, n={power_row['coefficient_2']:.12g}. A=0 baseline inverse uses fitted c0 and then reapplies c0, enforcing identity at A=0; inverse radial radius is solved on the monotonic near-axis branch. Baseline roundtrip maximum error={roundtrip_max:.3g} um.",
        f"Barrel composition is B_A = D_k(A) o S_s(A) o D_k(0)^-1: invert the fitted physical radial law at A=0, apply the saved E=0 paraxial scale ratio s(A), then apply the power-law physical k1(A). At A=5 D, s={scale5:.12g}, k1(A)={k5_power:.12g} mm^-2; at A=0, s={scale0:.12g}, k1(A)={k0_power_check:.12g} mm^-2.",
        f"At A=5 D, new barrel versus old joint-fit beta barrel baseline: corner RMS/max difference={barrel_change[0][0]:.6f}/{barrel_change[0][1]:.6f} um; all-nine RMS/max difference={barrel_change[2][0]:.6f}/{barrel_change[2][1]:.6f} um. Old beta={beta5:.12g} normalized; its corner calibration is retained through the previously fitted sx, sy, q, applied after the new barrel map.", "",
        "A=5 D, E=0 deg barrel prediction versus actual:", fmt_stats("New power-law barrel", new_barrel_err), fmt_stats("Old beta barrel", old_barrel_err), "",
        "A=5 D, E=20 deg joint prediction versus actual:", fmt_stats("New power barrel + saved K", new_joint_err), fmt_stats("Old beta barrel + saved K", old_joint_err), "",
        "The four corners were used for the saved K fit; the other five native field points are checks. Predicted point markers are omitted. Every keystone panel plots x=physical y and y=physical x. The isolated keystone panel applies saved K to the unchanged reference and is not a standalone optical state. Each trace shows the outer perimeter lines and its center sample as a dot; the middle native row and column are omitted.", "",
    ]
    (args.output_dir / "figure_notes.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Main figure: {main_png}")
    print(f"Sweep figure: {sweep_png}")
    print(fmt_stats("New power-law barrel A=5,E=0", new_barrel_err))
    print(fmt_stats("Old beta barrel A=5,E=0", old_barrel_err))
    print(fmt_stats("New power barrel + saved K A=5,E=20", new_joint_err))
    print(fmt_stats("Old beta barrel + saved K A=5,E=20", old_joint_err))


if __name__ == "__main__":
    main()







