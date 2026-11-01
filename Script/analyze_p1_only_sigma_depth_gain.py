"""Fit h=0 inverse gains and plot the literal requested gain-normalized curves."""

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
OUT_DIR = ROOT / "data" / "p1_only_sigma_depth_gain_analysis"
TRANSLATIONS = (-1.0, 0.0, 1.0)
ZOOMS = (1, 2, 3)
ZOOM_NAMES = {1: "z1 Inf (S1 ADE -15 deg, YDE 0)", 2: "z2 finite (S1 YDE +5)", 3: "z3 finite (S1 YDE -5)"}
STYLES = {-1.0: "--", 0.0: "-", 1.0: ":"}
COLORS = {-1.0: "#1f77b4", 0.0: "#555555", 1.0: "#d95f02"}


def main() -> None:
    with INPUT.open("rb") as stream:
        data = pickle.load(stream)
    required = {"h_translation_mm", "rotation_deg", "zoom_number", "real_y_si_mm", "trace_success"}
    if not isinstance(data, np.ndarray) or data.ndim != 1 or data.dtype.names is None:
        raise ValueError("Input pickle must contain a one-dimensional structured NumPy array")
    missing = required.difference(data.dtype.names)
    if missing:
        raise ValueError(f"Input pickle is missing fields: {', '.join(sorted(missing))}")
    if not np.all(data["trace_success"]):
        raise ValueError("Input contains failed rays")
    if not np.isfinite(data["rotation_deg"]).all() or np.isinf(data["real_y_si_mm"]).any():
        raise ValueError("Input contains nonfinite rotations or infinite SI Y values")

    rows_by_key: dict[tuple[float, int], np.ndarray] = {}
    angles_ref: np.ndarray | None = None
    for h in TRANSLATIONS:
        for zoom in ZOOMS:
            rows = data[
                np.isclose(data["h_translation_mm"], h, rtol=0.0, atol=1e-8)
                & (data["zoom_number"] == zoom)
            ]
            if rows.size == 0:
                raise ValueError(f"No samples for h={h:+g}, z{zoom}")
            rows = np.sort(rows, order="rotation_deg")
            angles = rows["rotation_deg"]
            if np.unique(angles).size != angles.size:
                raise ValueError(f"Duplicate rotation samples for h={h:+g}, z{zoom}")
            if angles_ref is None:
                angles_ref = angles
            elif not np.array_equal(angles, angles_ref):
                raise ValueError(f"Rotation samples differ for h={h:+g}, z{zoom}")
            rows_by_key[(h, zoom)] = rows

    assert angles_ref is not None
    coefficients: dict[int, dict[str, float | int]] = {}
    for zoom in ZOOMS:
        rows = rows_by_key[(0.0, zoom)]
        theta = rows["rotation_deg"]
        y = rows["real_y_si_mm"]
        finite = np.isfinite(y) & np.isfinite(theta)
        if np.count_nonzero(finite) < 2:
            raise ValueError(f"Not enough finite h=0 samples to fit z{zoom}")
        design = np.column_stack((y[finite], np.ones(np.count_nonzero(finite))))
        (a, b), *_ = np.linalg.lstsq(design, theta[finite], rcond=None)
        predicted = a * y[finite] + b
        residuals = theta[finite] - predicted
        sse = float(np.sum(residuals**2))
        sst = float(np.sum((theta[finite] - np.mean(theta[finite]))**2))
        coefficients[zoom] = {
            "inverse_gain_deg_per_mm": float(a),
            "intercept_deg": float(b),
            "finite_sample_count": int(np.count_nonzero(finite)),
            "rotation_min_deg": float(np.min(theta[finite])),
            "rotation_max_deg": float(np.max(theta[finite])),
            "residual_mean_deg": float(np.mean(residuals)),
            "residual_rms_deg": float(np.sqrt(np.mean(residuals**2))),
            "residual_max_abs_deg": float(np.max(np.abs(residuals))),
            "r_squared": float(1.0 - sse / sst) if sst > 0 else float("nan"),
        }

    # Follow the requested expressions literally: y_i/a_i, with a_i in deg/mm.
    # Therefore each plotted difference has units mm^2/deg, not a rotation.
    values_by_h: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for h in TRANSLATIONS:
        arrays = {z: rows_by_key[(h, z)]["real_y_si_mm"] for z in ZOOMS}
        left = arrays[2] / float(coefficients[2]["inverse_gain_deg_per_mm"]) - arrays[1] / float(
            coefficients[1]["inverse_gain_deg_per_mm"]
        )
        right = arrays[3] / float(coefficients[3]["inverse_gain_deg_per_mm"]) - arrays[1] / float(
            coefficients[1]["inverse_gain_deg_per_mm"]
        )
        cross = arrays[3] / float(coefficients[3]["inverse_gain_deg_per_mm"]) - arrays[2] / float(
            coefficients[2]["inverse_gain_deg_per_mm"]
        )
        values_by_h[h] = (left, right, cross)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    svg = OUT_DIR / "p1_only_sigma_depth_gain_normalized_differences.svg"
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.6))
    panel_specs = (
        (axes[0], 0, r"$y_2/a_2 - y_1/a_1$", "z2 fin+ vs z1 Inf"),
        (axes[1], 1, r"$y_3/a_3 - y_1/a_1$", "z3 fin- vs z1 Inf"),
        (axes[2], 2, r"$y_3/a_3 - y_2/a_2$", "z3 fin- vs z2 fin+"),
    )
    handles = [
        Line2D([0], [0], color=COLORS[h], lw=1.8, linestyle=STYLES[h],
               label=f"h = {h:+g} mm")
        for h in TRANSLATIONS
    ]
    for ax, side, expression, title in panel_specs:
        for h in TRANSLATIONS:
            ax.plot(
                angles_ref, values_by_h[h][side], color=COLORS[h],
                linestyle=STYLES[h], linewidth=1.55,
            )
        ax.axvline(0.0, color="#aaaaaa", linewidth=0.7, alpha=0.6)
        ax.axhline(0.0, color="#777777", linewidth=0.8, alpha=0.65)
        ax.set_xlim(-20.0, 20.0)
        ax.set_xlabel("RC eye rotation (deg)")
        ax.set_ylabel(r"Literal normalized difference (mm$^2$/deg)")
        ax.set_title(f"{title}: {expression}")
        ax.grid(True, color="#dddddd", linewidth=0.7)
        ax.legend(handles=handles, title="RC Y translation", frameon=True)
    fig.tight_layout()
    fig.savefig(svg, facecolor="white")
    plt.close(fig)

    coeff_path = OUT_DIR / "p1_only_sigma_depth_gain_coefficients.pkl"
    payload = {
        "fit_model": "theta_deg = a_i * real_y_si_mm + b_i",
        "gain_definition": "inverse gain a_i in deg/mm",
        "fit_translation_mm": 0.0,
        "input_pickle": str(INPUT),
        "coefficients_by_zoom": coefficients,
        "requested_plot_expressions": {
            "left": "y2/a2 - y1/a1",
            "middle": "y3/a3 - y1/a1",
            "right": "y3/a3 - y2/a2",
            "units": "mm^2/deg",
        },
    }
    with coeff_path.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)

    report_path = OUT_DIR / "p1_only_sigma_depth_gain_report.md"
    lines = [
        "# Sigma depth linear gain calibration",
        "",
        f"Input: `{INPUT}`",
        "",
        "## Fit definition",
        "",
        "For each zoom independently, finite cleaned samples at h = 0 mm over the available -20 to +20 deg rotations were fit by least squares to",
        "",
        "```text",
        "theta_deg = a_i * y_i_mm + b_i",
        "```",
        "",
        "The requested gain is preserved as the inverse gain `a_i` in deg/mm. The plotted expressions are evaluated literally as `y2/a2 - y1/a1`, `y3/a3 - y1/a1`, and `y3/a3 - y2/a2`; their units are mm^2/deg. They are not a rotation-normalized quantity.",
        "",
        "## Coefficients and fit residuals",
        "",
        "| Zoom | a_i (deg/mm) | b_i (deg) | n finite | Rotation span (deg) | RMS residual (deg) | Max abs residual (deg) | R^2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for zoom in ZOOMS:
        c = coefficients[zoom]
        lines.append(
            f"| {ZOOM_NAMES[zoom]} | {c['inverse_gain_deg_per_mm']:.10g} | {c['intercept_deg']:.10g} | "
            f"{c['finite_sample_count']} | {c['rotation_min_deg']:.1f} to {c['rotation_max_deg']:.1f} | "
            f"{c['residual_rms_deg']:.6g} | {c['residual_max_abs_deg']:.6g} | {c['r_squared']:.8f} |"
        )
    lines += [
        "",
        "## Outputs",
        "",
        f"- Figure: `{svg}`",
        f"- Coefficients: `{coeff_path}`",
        f"- This report: `{report_path}`",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("h=0 calibration:")
    for zoom in ZOOMS:
        c = coefficients[zoom]
        print(
            f"  z{zoom} ({ZOOM_NAMES[zoom]}): a={c['inverse_gain_deg_per_mm']:.10g} deg/mm, "
            f"b={c['intercept_deg']:.10g} deg, n={c['finite_sample_count']}, "
            f"RMS residual={c['residual_rms_deg']:.6g} deg, R2={c['r_squared']:.8f}"
        )
    print(f"SVG: {svg}")
    print(f"Coefficients pickle: {coeff_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
