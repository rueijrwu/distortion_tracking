"""Fit h=0 calibrations and solve eye rotation/head translation from zoom pairs."""

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
OUT_DIR = ROOT / "data" / "p1_only_sigma_depth_joint_estimation"
TRANSLATIONS = (-1.0, 0.0, 1.0)
ZOOMS = (1, 2, 3)
ZOOM_NAMES = {1: "z1 Inf (S1 ADE -15 deg, YDE 0)", 2: "z2 finite (S1 YDE +5)", 3: "z3 finite (S1 YDE -5)"}
PAIRS = ((1, 2), (1, 3), (2, 3))
PAIR_NAMES = {(1, 2): "z1 + z2", (1, 3): "z1 + z3", (2, 3): "z2 + z3"}
PAIR_COLORS = {(1, 2): "#1f77b4", (1, 3): "#d95f02", (2, 3): "#2ca02c"}
H_STYLES = {-1.0: "--", 0.0: "-", 1.0: ":"}


def solve_pair(y_i: float, y_j: float, a_i: float, b_i: float,
               a_j: float, b_j: float) -> tuple[float, float]:
    """Return theta (deg), h (mm) for theta=a_k*(y_k+h)+b_k."""
    c_i = a_i * y_i + b_i
    c_j = a_j * y_j + b_j
    denominator = a_j - a_i
    if denominator == 0.0:
        raise ZeroDivisionError("The selected zoom pair has identical inverse gains")
    h = (c_i - c_j) / denominator
    theta = c_i + a_i * h
    return float(theta), float(h)


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
        raise ValueError("Input contains failed traces")
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
            rows = np.sort(rows, order="rotation_deg")
            angles = rows["rotation_deg"]
            if rows.size == 0 or np.unique(angles).size != angles.size:
                raise ValueError(f"Missing or duplicate samples for h={h:+g}, z{zoom}")
            if angles_ref is None:
                angles_ref = angles
            elif not np.array_equal(angles, angles_ref):
                raise ValueError(f"Rotation samples differ for h={h:+g}, z{zoom}")
            rows_by_key[(h, zoom)] = rows
    assert angles_ref is not None

    # Per-zoom h=0 fit: theta_deg = a_i*y_i_mm + b_i.
    calibrations: dict[int, dict[str, float | int]] = {}
    for zoom in ZOOMS:
        rows = rows_by_key[(0.0, zoom)]
        y, theta = rows["real_y_si_mm"], rows["rotation_deg"]
        finite = np.isfinite(y) & np.isfinite(theta)
        if np.count_nonzero(finite) < 2:
            raise ValueError(f"Not enough finite h=0 calibration samples for z{zoom}")
        design = np.column_stack((y[finite], np.ones(np.count_nonzero(finite))))
        (a, b), *_ = np.linalg.lstsq(design, theta[finite], rcond=None)
        residual = theta[finite] - (a * y[finite] + b)
        sse = float(np.sum(residual**2))
        sst = float(np.sum((theta[finite] - np.mean(theta[finite]))**2))
        calibrations[zoom] = {
            "a_deg_per_mm": float(a),
            "b_deg": float(b),
            "finite_sample_count": int(np.count_nonzero(finite)),
            "residual_rms_deg": float(np.sqrt(np.mean(residual**2))),
            "residual_max_abs_deg": float(np.max(np.abs(residual))),
            "r_squared": float(1.0 - sse / sst) if sst > 0 else float("nan"),
        }

    # Check the closed-form pair solution against known synthetic theta and h.
    theta_true, h_true = 1.25, 0.4
    for i, j in PAIRS:
        ai, bi = calibrations[i]["a_deg_per_mm"], calibrations[i]["b_deg"]
        aj, bj = calibrations[j]["a_deg_per_mm"], calibrations[j]["b_deg"]
        yi = (theta_true - bi) / ai - h_true
        yj = (theta_true - bj) / aj - h_true
        theta_check, h_check = solve_pair(yi, yj, ai, bi, aj, bj)
        if not (np.isclose(theta_check, theta_true, atol=1e-10, rtol=0.0)
                and np.isclose(h_check, h_true, atol=1e-10, rtol=0.0)):
            raise AssertionError(f"Synthetic pair validation failed for z{i}+z{j}")

    dtype = np.dtype([
        ("h_true_mm", "f8"), ("rotation_true_deg", "f8"),
        ("pair_code", "i4"), ("zoom_i", "i4"), ("zoom_j", "i4"),
        ("y_i_si_mm", "f8"), ("y_j_si_mm", "f8"),
        ("theta_est_deg", "f8"), ("h_est_mm", "f8"), ("valid_pair", "?"),
    ])
    rows_out = []
    valid_counts = {pair: 0 for pair in PAIRS}
    total_counts = {pair: 0 for pair in PAIRS}
    curves: dict[tuple[float, tuple[int, int]], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for h in TRANSLATIONS:
        for pair_code, (i, j) in enumerate(PAIRS, 1):
            ri, rj = rows_by_key[(h, i)], rows_by_key[(h, j)]
            yi, yj = ri["real_y_si_mm"], rj["real_y_si_mm"]
            if not np.array_equal(ri["rotation_deg"], rj["rotation_deg"]):
                raise ValueError(f"Pair z{i}+z{j} rotations do not align at h={h:+g}")
            ai, bi = calibrations[i]["a_deg_per_mm"], calibrations[i]["b_deg"]
            aj, bj = calibrations[j]["a_deg_per_mm"], calibrations[j]["b_deg"]
            theta_est = np.full(len(angles_ref), np.nan)
            h_est = np.full(len(angles_ref), np.nan)
            valid = np.isfinite(yi) & np.isfinite(yj)
            if np.any(valid):
                ci = ai * yi[valid] + bi
                cj = aj * yj[valid] + bj
                h_est[valid] = (ci - cj) / (aj - ai)
                theta_est[valid] = ci + ai * h_est[valid]
            curves[(h, (i, j))] = (angles_ref, theta_est, h_est)
            total_counts[(i, j)] += len(angles_ref)
            valid_counts[(i, j)] += int(np.count_nonzero(valid))
            for k, angle in enumerate(angles_ref):
                rows_out.append((float(h), float(angle), pair_code, i, j,
                                 yi[k], yj[k], theta_est[k], h_est[k], bool(valid[k])))

    result = np.array(rows_out, dtype=dtype)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = OUT_DIR / "p1_only_sigma_depth_joint_estimates.pkl"
    coeff_path = OUT_DIR / "p1_only_sigma_depth_joint_calibrations.pkl"
    svg_path = OUT_DIR / "p1_only_sigma_depth_joint_estimation.svg"
    report_path = OUT_DIR / "p1_only_sigma_depth_joint_estimation_report.md"

    with result_path.open("wb") as stream:
        pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)
    coeff_payload = {
        "model": "theta_deg = a_i_deg_per_mm * (y_i_mm + h_mm) + b_i_deg",
        "input_pickle": str(INPUT),
        "calibrations_by_zoom": calibrations,
        "pair_order": list(PAIRS),
        "pair_names": PAIR_NAMES,
        "synthetic_validation": {
            "theta_true_deg": theta_true,
            "h_true_mm": h_true,
            "pairs_validated": list(PAIRS),
            "tolerance": 1e-10,
            "passed": True,
        },
        "valid_pair_counts": valid_counts,
        "total_pair_counts": total_counts,
    }
    with coeff_path.open("wb") as stream:
        pickle.dump(coeff_payload, stream, protocol=pickle.HIGHEST_PROTOCOL)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.2))
    pair_handles = [Line2D([0], [0], color=PAIR_COLORS[pair], lw=2,
                           label=PAIR_NAMES[pair]) for pair in PAIRS]
    h_handles = [Line2D([0], [0], color="#444444", lw=1.8,
                        linestyle=H_STYLES[h], label=f"h = {h:+g} mm") for h in TRANSLATIONS]
    for pair in PAIRS:
        for h in TRANSLATIONS:
            angles, theta_est, h_est = curves[(h, pair)]
            label = PAIR_NAMES[pair]
            axes[0].plot(angles, theta_est, color=PAIR_COLORS[pair],
                         linestyle=H_STYLES[h], linewidth=1.35)
            axes[1].plot(angles, h_est, color=PAIR_COLORS[pair],
                         linestyle=H_STYLES[h], linewidth=1.35)

    axes[0].plot([-20.0, 20.0], [-20.0, 20.0], color="#666666", linewidth=1.0,
                 linestyle="--", alpha=0.7, label="identity")
    axes[0].set_xlim(-20.0, 20.0)
    axes[0].set_xlabel("Ground-truth eye rotation (deg)")
    axes[0].set_ylabel("Estimated eye rotation θ (deg)")
    axes[0].set_title("Pairwise eye-rotation estimates")
    axes[0].grid(True, color="#dddddd", linewidth=0.7)
    axes[0].legend(handles=pair_handles, title="Zoom pair", loc="upper left", frameon=True)

    axes[1].set_xlim(-20.0, 20.0)
    axes[1].set_xlabel("Ground-truth eye rotation (deg)")
    axes[1].set_ylabel("Estimated head translation h (mm)")
    axes[1].set_title("Pairwise head-translation estimates")
    axes[1].grid(True, color="#dddddd", linewidth=0.7)
    axes[1].legend(handles=pair_handles, title="Zoom pair", loc="upper left", frameon=True)
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))
    fig.legend(handles=h_handles, title="True head translation (line style)",
               loc="lower center", ncol=3, frameon=True)
    fig.savefig(svg_path, facecolor="white")
    plt.close(fig)

    lines = [
        "# Sigma depth pairwise joint estimation",
        "",
        f"Input: `{INPUT}`",
        "",
        "## Model and pair solution",
        "",
        "Per zoom, h=0 finite cleaned samples were fit to `theta = a_i*y_i + b_i`. Pair estimates then use the exact model `theta = a_i*(y_i+h) + b_i`.",
        "",
        "For a pair i,j, let `c_i=a_i*y_i+b_i` and `c_j=a_j*y_j+b_j`. The closed-form solution is `h=(c_i-c_j)/(a_j-a_i)` and `theta=c_i+a_i*h`.",
        "",
        "A pair is skipped when either source y is NaN; resulting estimate fields remain NaN at that rotation.",
        "",
        "## Calibration fits",
        "",
        "| Zoom | a_i (deg/mm) | b_i (deg) | Finite samples | RMS residual (deg) | Max abs residual (deg) | R^2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for z in ZOOMS:
        c = calibrations[z]
        lines.append(
            f"| {ZOOM_NAMES[z]} | {c['a_deg_per_mm']:.10g} | {c['b_deg']:.10g} | "
            f"{c['finite_sample_count']} | {c['residual_rms_deg']:.6g} | "
            f"{c['residual_max_abs_deg']:.6g} | {c['r_squared']:.8f} |"
        )
    lines += ["", "## Pair availability", "", "| Pair | Valid estimates | Total samples |", "|---|---:|---:|"]
    for pair in PAIRS:
        lines.append(f"| {PAIR_NAMES[pair]} | {valid_counts[pair]} | {total_counts[pair]} |")
    lines += [
        "",
        "## Synthetic algebra check",
        "",
        f"For theta={theta_true} deg and h={h_true} mm, all three zoom pairs recovered both values within 1e-10: PASS.",
        "",
        "## Outputs",
        "",
        f"- Figure: `{svg_path}`",
        f"- Estimates: `{result_path}`",
        f"- Calibration metadata: `{coeff_path}`",
        f"- Report: `{report_path}`",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("Calibration coefficients:")
    for z in ZOOMS:
        c = calibrations[z]
        print(f"  z{z}: a={c['a_deg_per_mm']:.10g} deg/mm, b={c['b_deg']:.10g} deg, n={c['finite_sample_count']}")
    print("Valid pair estimates:")
    for pair in PAIRS:
        print(f"  {PAIR_NAMES[pair]}: {valid_counts[pair]}/{total_counts[pair]}")
    print("Synthetic algebra check: passed for all pairs")
    print(f"SVG: {svg_path}")
    print(f"Estimates pickle: {result_path}")
    print(f"Calibration pickle: {coeff_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
