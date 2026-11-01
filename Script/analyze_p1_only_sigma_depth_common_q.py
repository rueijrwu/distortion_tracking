"""Test a shared image-translation gain and estimate theta/h from all zooms."""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "p1_only_sigma_depth_raytracing" / "p1_only_sigma_depth_raytracing_cleaned.pkl"
OUT_DIR = ROOT / "data" / "p1_only_sigma_depth_common_q_analysis"
TEST_H = (-1.0, 1.0)
SOURCE_H = (-1.0, 0.0, 1.0)
ZOOMS = (1, 2, 3)
ZOOM_NAMES = {1: "z1 Inf (S1 ADE -15 deg, YDE 0)", 2: "z2 finite (S1 YDE +5)", 3: "z3 finite (S1 YDE -5)"}
H_STYLES = {-1.0: "--", 1.0: ":"}
H_COLORS = {-1.0: "#1f77b4", 1.0: "#d95f02"}
ZOOM_COLORS = {1: "#1f77b4", 2: "#d95f02", 3: "#2ca02c"}


def metrics(errors: np.ndarray) -> dict[str, float | int]:
    errors = errors[np.isfinite(errors)]
    if errors.size == 0:
        return {"count": 0, "bias": float("nan"), "rms": float("nan"), "max_abs": float("nan")}
    return {
        "count": int(errors.size),
        "bias": float(np.mean(errors)),
        "rms": float(np.sqrt(np.mean(errors**2))),
        "max_abs": float(np.max(np.abs(errors))),
    }


def main() -> None:
    with INPUT.open("rb") as stream:
        data = pickle.load(stream)
    required = {"h_translation_mm", "rotation_deg", "zoom_number", "real_y_si_mm", "trace_success"}
    if not isinstance(data, np.ndarray) or data.ndim != 1 or data.dtype.names is None:
        raise ValueError("Input must be a one-dimensional structured NumPy array")
    missing = required.difference(data.dtype.names)
    if missing:
        raise ValueError(f"Input pickle is missing fields: {', '.join(sorted(missing))}")
    if not np.all(data["trace_success"]):
        raise ValueError("Input contains failed traces")
    if not np.isfinite(data["rotation_deg"]).all() or np.isinf(data["real_y_si_mm"]).any():
        raise ValueError("Input contains nonfinite rotations or infinite SI Y values")

    rows_by_key: dict[tuple[float, int], np.ndarray] = {}
    angles_ref: np.ndarray | None = None
    for h in SOURCE_H:
        for zoom in ZOOMS:
            rows = np.sort(data[
                np.isclose(data["h_translation_mm"], h, rtol=0.0, atol=1e-8)
                & (data["zoom_number"] == zoom)
            ], order="rotation_deg")
            if rows.size == 0 or np.unique(rows["rotation_deg"]).size != rows.size:
                raise ValueError(f"Missing or duplicate samples for h={h:+g}, z{zoom}")
            if angles_ref is None:
                angles_ref = rows["rotation_deg"]
            elif not np.array_equal(rows["rotation_deg"], angles_ref):
                raise ValueError(f"Rotation samples differ for h={h:+g}, z{zoom}")
            rows_by_key[(h, zoom)] = rows
    assert angles_ref is not None

    # Use one common finite h=0 angle set for all per-zoom linear calibrations.
    y0 = {z: rows_by_key[(0.0, z)]["real_y_si_mm"] for z in ZOOMS}
    calibration_mask = np.logical_and.reduce([np.isfinite(y0[z]) for z in ZOOMS])
    theta_cal = angles_ref[calibration_mask]
    if theta_cal.size < 2:
        raise ValueError("Too few common finite h=0 rotations for calibration")

    calibration: dict[int, dict[str, float | int]] = {}
    fitted_y0: dict[int, np.ndarray] = {}
    for z in ZOOMS:
        values = y0[z][calibration_mask]
        design = np.column_stack((theta_cal, np.ones(theta_cal.size)))
        (m, c), *_ = np.linalg.lstsq(design, values, rcond=None)
        residual = values - (m * theta_cal + c)
        sse = float(np.sum(residual**2))
        sst = float(np.sum((values - np.mean(values))**2))
        calibration[z] = {
            "forward_gain_mm_per_deg": float(m),
            "intercept_mm": float(c),
            "matched_h0_sample_count": int(theta_cal.size),
            "residual_bias_mm": float(np.mean(residual)),
            "residual_rms_mm": float(np.sqrt(np.mean(residual**2))),
            "residual_max_abs_mm": float(np.max(np.abs(residual))),
            "r_squared": float(1.0 - sse / sst) if sst > 0 else float("nan"),
        }
        fitted_y0[z] = m * theta_cal + c

    # For a fair q comparison, use only rotations finite at every zoom and
    # every h in {-1,0,+1}. Estimate a shared q from delta-y = q*h.
    common_mask = calibration_mask.copy()
    for h in TEST_H:
        common_mask &= np.logical_and.reduce([
            np.isfinite(rows_by_key[(h, z)]["real_y_si_mm"]) for z in ZOOMS
        ])
    if np.count_nonzero(common_mask) < 2:
        raise ValueError("Too few all-zoom/all-translation finite angles to estimate q")
    common_angles = angles_ref[common_mask]
    q_by_zoom: dict[int, float] = {}
    q_residuals: dict[int, dict[str, float | int]] = {}
    q_num = 0.0
    q_den = 0.0
    q_local: dict[int, np.ndarray] = {}
    for z in ZOOMS:
        ybase = rows_by_key[(0.0, z)]["real_y_si_mm"][common_mask]
        deltas = []
        for h in TEST_H:
            yh = rows_by_key[(h, z)]["real_y_si_mm"][common_mask]
            delta = yh - ybase
            deltas.append(delta / h)
            q_num += float(np.sum(h * delta))
            q_den += float(common_angles.size * h**2)
        q_local[z] = np.mean(np.vstack(deltas), axis=0)
        q_by_zoom[z] = float(np.mean(np.concatenate(deltas)))
    common_q = q_num / q_den
    for z in ZOOMS:
        residual_list = []
        for h in TEST_H:
            yh = rows_by_key[(h, z)]["real_y_si_mm"][common_mask]
            ybase = rows_by_key[(0.0, z)]["real_y_si_mm"][common_mask]
            residual_list.append(yh - ybase - common_q * h)
        q_residuals[z] = metrics(np.concatenate(residual_list))

    q_values = np.array([q_by_zoom[z] for z in ZOOMS])
    q_spread_abs = float(np.max(q_values) - np.min(q_values))
    q_spread_relative = float(q_spread_abs / abs(common_q)) if common_q != 0 else float("inf")

    # Solve all three measurements jointly for each h=-1/+1 sample.
    m_vector = np.array([calibration[z]["forward_gain_mm_per_deg"] for z in ZOOMS], dtype=float)
    c_vector = np.array([calibration[z]["intercept_mm"] for z in ZOOMS], dtype=float)
    design_joint = np.column_stack((m_vector, np.full(3, common_q)))
    if np.linalg.matrix_rank(design_joint) < 2:
        raise ValueError("Joint theta/h design matrix is rank deficient")
    joint_singular_values = np.linalg.svd(design_joint, compute_uv=False)
    joint_condition_number = float(joint_singular_values[0] / joint_singular_values[-1])
    pseudo_inverse = np.linalg.pinv(design_joint)

    # Synthetic algebra validation with the fitted common-q model.
    theta_synth, h_synth = 2.3, 0.7
    y_synth = m_vector * theta_synth + c_vector + common_q * h_synth
    theta_h_check = pseudo_inverse @ (y_synth - c_vector)
    if not (np.isclose(theta_h_check[0], theta_synth, atol=1e-10, rtol=0.0)
            and np.isclose(theta_h_check[1], h_synth, atol=1e-10, rtol=0.0)):
        raise AssertionError("Synthetic joint theta/h validation failed")

    result_dtype = np.dtype([
        ("h_true_mm", "f8"), ("rotation_true_deg", "f8"),
        ("theta_est_deg", "f8"), ("h_est_mm", "f8"),
        ("valid_all_zooms", "?"),
    ])
    result_rows = []
    estimates: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    eval_metrics: dict[float, dict[str, dict[str, float | int]]] = {}
    for h in TEST_H:
        ys = np.vstack([rows_by_key[(h, z)]["real_y_si_mm"] for z in ZOOMS])
        valid = np.isfinite(ys).all(axis=0)
        theta_est = np.full(angles_ref.size, np.nan)
        h_est = np.full(angles_ref.size, np.nan)
        solved = pseudo_inverse @ (ys[:, valid] - c_vector[:, None])
        theta_est[valid] = solved[0]
        h_est[valid] = solved[1]
        estimates[h] = (angles_ref, theta_est, h_est)
        eval_metrics[h] = {
            "theta_error_deg": metrics(theta_est[valid] - angles_ref[valid]),
            "h_error_mm": metrics(h_est[valid] - h),
        }
        for k, angle in enumerate(angles_ref):
            result_rows.append((h, float(angle), theta_est[k], h_est[k], bool(valid[k])))
    results = np.array(result_rows, dtype=result_dtype)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = OUT_DIR / "p1_only_sigma_depth_common_q_estimates.pkl"
    coeff_path = OUT_DIR / "p1_only_sigma_depth_common_q_calibration.pkl"
    main_svg = OUT_DIR / "p1_only_sigma_depth_common_q_joint_estimation.svg"
    diagnostic_svg = OUT_DIR / "p1_only_sigma_depth_common_q_diagnostics.svg"
    report_path = OUT_DIR / "p1_only_sigma_depth_common_q_report.md"
    with result_path.open("wb") as stream:
        pickle.dump(results, stream, protocol=pickle.HIGHEST_PROTOCOL)
    calibration_payload = {
        "model": "y_i_mm = m_i_mm_per_deg * theta_deg + c_i_mm + q_mm_per_mm * h_mm",
        "input_pickle": str(INPUT),
        "calibration_common_h0_angle_count": int(theta_cal.size),
        "common_q_fit_angle_count": int(common_angles.size),
        "calibration_by_zoom": calibration,
        "common_q_mm_per_mm": float(common_q),
        "q_by_zoom_mm_per_mm": q_by_zoom,
        "q_residual_metrics_by_zoom": q_residuals,
        "q_absolute_spread_mm_per_mm": q_spread_abs,
        "q_relative_spread_fraction": q_spread_relative,
        "joint_design_singular_values": joint_singular_values,
        "joint_design_condition_number": joint_condition_number,
        "evaluation_h_values_mm": list(TEST_H),
        "evaluation_metrics_by_h": eval_metrics,
        "synthetic_validation": {"theta_deg": theta_synth, "h_mm": h_synth, "passed": True},
    }
    with coeff_path.open("wb") as stream:
        pickle.dump(calibration_payload, stream, protocol=pickle.HIGHEST_PROTOCOL)

    # Main 1x2 plot, with h-only series because all three zooms enter jointly.
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.8))
    h_handles = [Line2D([0], [0], color=H_COLORS[h], linewidth=1.8,
                        linestyle=H_STYLES[h], label=f"h = {h:+g} mm") for h in TEST_H]
    for h in TEST_H:
        angles, theta_est, h_est = estimates[h]
        axes[0].plot(angles, theta_est, color=H_COLORS[h], linestyle=H_STYLES[h], linewidth=1.6)
        axes[1].plot(angles, h_est, color=H_COLORS[h], linestyle=H_STYLES[h], linewidth=1.6)
    axes[0].plot([-20.0, 20.0], [-20.0, 20.0], color="#555555", linestyle="--", linewidth=1.0,
                 alpha=0.7, label="truth θ = x")
    for h in TEST_H:
        axes[1].axhline(h, color=H_COLORS[h], linestyle=H_STYLES[h], linewidth=0.9, alpha=0.45)
    axes[0].set_xlim(-20.0, 20.0)
    axes[0].set_xlabel("Ground-truth eye rotation (deg)")
    axes[0].set_ylabel("Estimated eye rotation θ (deg)")
    axes[0].set_title("Three-zoom joint eye-rotation estimate")
    axes[0].grid(True, color="#dddddd", linewidth=0.7)
    axes[0].legend(handles=h_handles, title="Head translation", loc="upper left", frameon=True)
    axes[1].set_xlim(-20.0, 20.0)
    axes[1].set_xlabel("Ground-truth eye rotation (deg)")
    axes[1].set_ylabel("Estimated head translation h (mm)")
    axes[1].set_title("Three-zoom joint translation estimate")
    axes[1].grid(True, color="#dddddd", linewidth=0.7)
    axes[1].legend(handles=h_handles, title="Head translation", loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(main_svg, facecolor="white")
    plt.close(fig)

    # Diagnostic: q_i(theta) from matched ±1 vs h=0 shifts, plus the common q.
    diag_fig, diag_ax = plt.subplots(figsize=(9.2, 5.6))
    for z in ZOOMS:
        diag_ax.plot(common_angles, q_local[z], color=ZOOM_COLORS[z],
                     linewidth=1.35, label=f"{ZOOM_NAMES[z]} q_i(θ)")
    diag_ax.axhline(common_q, color="#222222", linestyle="--", linewidth=1.5,
                    label=f"common q = {common_q:.6g} mm/mm")
    diag_ax.set_xlim(-20.0, 20.0)
    diag_ax.set_xlabel("Ground-truth eye rotation θ (deg)")
    diag_ax.set_ylabel("Image translation gain q_i (mm/mm)")
    diag_ax.set_title("Per-zoom translation gain vs common-q model")
    diag_ax.grid(True, color="#dddddd", linewidth=0.7)
    diag_ax.legend(title="Per-zoom gain / common fit", loc="best", frameon=True)
    diag_fig.tight_layout()
    diag_fig.savefig(diagnostic_svg, facecolor="white")
    plt.close(diag_fig)

    lines = [
        "# Common image-translation gain test",
        "",
        f"Input: `{INPUT}`",
        "",
        "## Model and matched samples",
        "",
        "The tested model is `y_i = m_i*theta + c_i + q*h`, with one shared q across zooms. Each `m_i,c_i` was fit by least squares on h=0 samples using the same rotations finite for all three zooms. q was fit from `y_i(h)-y_i(0)=q*h` using the common finite intersection across all three zooms at h=-1, 0, and +1. Calibration and evaluation use the same sweep; this is not a held-out test.",
        "",
        f"Matched h=0 calibration angles: {theta_cal.size}; common q angles finite across all h/zooms: {common_angles.size}; evaluation uses h=-1,+1 only.",
        "",
        "## Forward image calibrations",
        "",
        "| Zoom | m_i (mm/deg) | c_i (mm) | n | Residual bias (mm) | RMS (mm) | Max abs (mm) | R^2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for z in ZOOMS:
        c = calibration[z]
        lines.append(
            f"| {ZOOM_NAMES[z]} | {c['forward_gain_mm_per_deg']:.10g} | {c['intercept_mm']:.10g} | "
            f"{c['matched_h0_sample_count']} | {c['residual_bias_mm']:.6g} | {c['residual_rms_mm']:.6g} | "
            f"{c['residual_max_abs_mm']:.6g} | {c['r_squared']:.8f} |"
        )
    lines += [
        "",
        "## Translation gain comparison",
        "",
        "| Zoom | q_i (mm/mm) | Common-q residual bias (mm) | Common-q residual RMS (mm) | Common-q max abs residual (mm) |",
        "|---|---:|---:|---:|---:|",
    ]
    for z in ZOOMS:
        r = q_residuals[z]
        lines.append(
            f"| {ZOOM_NAMES[z]} | {q_by_zoom[z]:.10g} | {r['bias']:.6g} | "
            f"{r['rms']:.6g} | {r['max_abs']:.6g} |"
        )
    lines += [
        "",
        f"Common q: {common_q:.10g} mm/mm. Absolute q spread: {q_spread_abs:.6g} mm/mm ({q_spread_relative:.4%} of |common q|).",
        f"Joint design singular values: {joint_singular_values[0]:.6g}, {joint_singular_values[1]:.6g}; condition number: {joint_condition_number:.6g}.",
        "",
        "## Common-q assessment",
        "",
        f"The shared-q assumption is not supported by these same-sweep data: per-zoom q_i span {q_spread_abs:.6g} mm/mm ({q_spread_relative:.2%} of |common q|). Under the shared fit, z1 and z3 differential-y RMS residuals are {q_residuals[1]['rms']:.6g} and {q_residuals[3]['rms']:.6g} mm, versus {q_residuals[2]['rms']:.6g} mm for z2. The joint estimates also fail to recover the known translations (h RMSE {eval_metrics[-1.0]['h_error_mm']['rms']:.6g} mm at h=-1 and {eval_metrics[1.0]['h_error_mm']['rms']:.6g} mm at h=+1). This is descriptive evidence from calibration and evaluation on the same sweep, not a held-out validation.",
        "",
        "## Joint estimate errors (h = -1,+1 only)",
        "",
        "| True h (mm) | n | θ bias (deg) | θ RMSE (deg) | h bias (mm) | h RMSE (mm) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for h in TEST_H:
        t = eval_metrics[h]["theta_error_deg"]
        he = eval_metrics[h]["h_error_mm"]
        lines.append(f"| {h:+g} | {t['count']} | {t['bias']:.6g} | {t['rms']:.6g} | {he['bias']:.6g} | {he['rms']:.6g} |")
    lines += [
        "",
        "Synthetic algebra validation recovered theta=2.3 deg and h=0.7 mm: PASS.",
        "",
        "## Outputs",
        "",
        f"- Joint estimates: `{result_path}`",
        f"- Calibration and q metrics: `{coeff_path}`",
        f"- Joint-estimate figure: `{main_svg}`",
        f"- q diagnostic figure: `{diagnostic_svg}`",
        f"- This report: `{report_path}`",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Matched h0 calibration angles={theta_cal.size}; all-data q angles={common_angles.size}")
    print("Forward calibration:")
    for z in ZOOMS:
        c = calibration[z]
        print(f"  z{z}: m={c['forward_gain_mm_per_deg']:.10g} mm/deg, c={c['intercept_mm']:.10g} mm, RMS={c['residual_rms_mm']:.6g} mm")
    print("Per-zoom q:", {ZOOM_NAMES[z]: q_by_zoom[z] for z in ZOOMS})
    print(f"Common q={common_q:.10g} mm/mm; relative spread={q_spread_relative:.4%}")
    print(f"Joint design singular values={joint_singular_values}; condition={joint_condition_number:.6g}")
    print("Joint errors:", eval_metrics)
    print(f"SVG: {main_svg}")
    print(f"Diagnostic SVG: {diagnostic_svg}")
    print(f"Results pickle: {result_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
