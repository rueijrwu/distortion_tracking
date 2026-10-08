"""Fit and evaluate a quadratic forward model for five-point image shifts.

The script fits the saved CODE V sweep without starting CODE V. For each of
the center and four corner fields, it models the real image-coordinate shift
from the zero-degree pattern as a quadratic function of eye rotation.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_INPUT = PROJECT_DIR / "data" / "distortion_grid" / "distortion_grid.pkl"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "polynomial_distortion_fit"
POINTS = (
    ("center", 0.0, 0.0),
    ("top-left", -1.0, 1.0),
    ("top-right", 1.0, 1.0),
    ("bottom-left", -1.0, -1.0),
    ("bottom-right", 1.0, -1.0),
)
REQUIRED_FIELDS = (
    "eye_rotation_deg", "field_x_relative", "field_y_relative", "real_x_mm", "real_y_mm",
)


def load_five_point_sweep(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted angles and absolute center-relative image coordinates."""
    if not path.is_file():
        raise FileNotFoundError(f"Sweep pickle not found: {path}")
    with path.open("rb") as stream:
        data = pickle.load(stream)
    if not isinstance(data, np.ndarray) or data.ndim != 1 or data.dtype.names is None:
        raise ValueError("Input must be a one-dimensional structured NumPy array.")
    missing = [name for name in REQUIRED_FIELDS if name not in data.dtype.names]
    if missing:
        raise ValueError(f"Input is missing fields: {', '.join(missing)}")

    angles = np.unique(data["eye_rotation_deg"]).astype(float)
    if angles.size < 3 or not np.all(np.isfinite(angles)):
        raise ValueError("At least three finite rotation angles are required.")
    coordinates = np.empty((angles.size, len(POINTS), 2), dtype=float)
    for ai, angle in enumerate(angles):
        rows = data[data["eye_rotation_deg"] == angle]
        for pi, (name, fx, fy) in enumerate(POINTS):
            mask = (np.isclose(rows["field_x_relative"], fx, rtol=0, atol=1e-9)
                    & np.isclose(rows["field_y_relative"], fy, rtol=0, atol=1e-9))
            selected = rows[mask]
            if selected.size != 1:
                raise ValueError(
                    f"Rotation {angle:g} needs exactly one {name} field ({fx:g},{fy:g})."
                )
            coordinates[ai, pi] = (selected[0]["real_x_mm"], selected[0]["real_y_mm"])
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("The five-point coordinates contain non-finite values.")
    if not np.any(angles == 0.0):
        raise ValueError("The sweep must contain an exact 0 degree baseline.")
    return angles, coordinates


def design_matrix(angles_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return scaled and physical quadratic bases, both ordered [theta^2, theta, 1]."""
    angles_deg = np.asarray(angles_deg, dtype=float)
    scaled_theta = angles_deg / 20.0
    scaled_basis = np.column_stack((scaled_theta ** 2, scaled_theta, np.ones_like(angles_deg)))
    physical_basis = np.column_stack((angles_deg ** 2, angles_deg, np.ones_like(angles_deg)))
    return scaled_basis, physical_basis


def fit_coefficients(angles_deg: np.ndarray, shifts_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit all point/axis columns and return scaled and physical coefficients."""
    scaled_basis, _ = design_matrix(angles_deg)
    flat = shifts_mm.reshape(angles_deg.size, -1)
    scaled_coefficients = np.linalg.lstsq(scaled_basis, flat, rcond=None)[0]
    physical_coefficients = scaled_coefficients.copy()
    physical_coefficients[0] /= 20.0 ** 2
    physical_coefficients[1] /= 20.0
    return scaled_coefficients, physical_coefficients


def predict(angles_deg: np.ndarray, scaled_coefficients: np.ndarray) -> np.ndarray:
    scaled_basis, _ = design_matrix(angles_deg)
    prediction = scaled_basis @ scaled_coefficients
    return prediction.reshape(angles_deg.size, len(POINTS), 2)


def _axis_metrics(errors_mm: np.ndarray, angles_deg: np.ndarray) -> dict:
    metrics = {}
    for point_index, (name, _, _) in enumerate(POINTS):
        metrics[name] = {}
        for axis_index, axis_name in enumerate(("x", "y")):
            error = errors_mm[:, point_index, axis_index]
            worst_index = int(np.argmax(np.abs(error)))
            metrics[name][axis_name] = {
                "rmse_mm": float(np.sqrt(np.mean(error ** 2))),
                "mae_mm": float(np.mean(np.abs(error))),
                "max_abs_mm": float(np.max(np.abs(error))),
                "max_abs_angle_deg": float(angles_deg[worst_index]),
            }
    return metrics


def summarize_errors(errors_mm: np.ndarray, angles_deg: np.ndarray) -> dict:
    """Compute full-sweep coordinate and vector error summaries."""
    corners = errors_mm[:, 1:, :]
    flat_corner = corners.reshape(-1)
    max_angle_index, max_corner_index, max_axis_index = np.unravel_index(
        int(np.argmax(np.abs(corners))), corners.shape
    )
    vector_norm = np.linalg.norm(corners, axis=2)
    max_vector_angle_index, max_vector_point_index = np.unravel_index(
        int(np.argmax(vector_norm)), vector_norm.shape
    )
    return {
        "corner_coordinate_rmse_mm": float(np.sqrt(np.mean(flat_corner ** 2))),
        "corner_coordinate_mae_mm": float(np.mean(np.abs(flat_corner))),
        "corner_coordinate_max_abs_mm": float(np.max(np.abs(flat_corner))),
        "corner_coordinate_max_abs_um": float(np.max(np.abs(flat_corner)) * 1000.0),
        "worst_coordinate": {
            "angle_deg": float(angles_deg[max_angle_index]),
            "point": POINTS[max_corner_index + 1][0],
            "axis": ("x", "y")[max_axis_index],
            "error_mm": float(corners[max_angle_index, max_corner_index, max_axis_index]),
        },
        "corner_vector_rmse_mm": float(np.sqrt(np.mean(vector_norm ** 2))),
        "corner_vector_median_mm": float(np.median(vector_norm)),
        "corner_vector_p95_mm": float(np.percentile(vector_norm, 95)),
        "corner_vector_max_mm": float(np.max(vector_norm)),
        "corner_vector_max_um": float(np.max(vector_norm) * 1000.0),
        "worst_vector": {
            "angle_deg": float(angles_deg[max_vector_angle_index]),
            "point": POINTS[max_vector_point_index + 1][0],
            "error_norm_mm": float(vector_norm[max_vector_angle_index, max_vector_point_index]),
        },
        "coordinates_by_point_axis": _axis_metrics(errors_mm, angles_deg),
        "center_coordinate_rmse_mm": float(np.sqrt(np.mean(errors_mm[:, 0, :] ** 2))),
        "center_max_abs_mm": float(np.max(np.abs(errors_mm[:, 0, :]))),
    }


def _format_mm_and_um(value_mm: float) -> str:
    return f"{value_mm:.9g} mm ({value_mm * 1000.0:.6g} µm)"


def make_report(path: Path, angles: np.ndarray, shifts: np.ndarray,
                metrics: dict, theta_coefficients: np.ndarray) -> None:
    point_metrics = metrics["coordinates_by_point_axis"]
    lines = [
        "# Quadratic fit of five-point distortion shifts",
        "",
        "## Model and data",
        "",
        f"Fit the saved ground-truth sweep at {angles.size} rotations from "
        f"{angles.min():g}° to {angles.max():g}°. The input contains {shifts.shape[0]} "
        "five-point samples; no CODE V session was started.",
        "",
        "For each point and coordinate, the modeled shift is its center-relative "
        "real image coordinate minus the corresponding coordinate at 0°:",
        "",
        "```text",
        "Δ_i(θ) = [θ², θ, 1] C_i,     Δ_i = [dx_i, dy_i] in mm",
        "D = B C, with B[n] = [θ_n², θ_n, 1] and D containing 10 X/Y columns.",
        "```",
        "",
        "The fit uses the numerically scaled angle t=θ/20 and `numpy.linalg.lstsq`; "
        "coefficients are also saved in physical [θ², θ, 1] units. The intercept "
        "was fitted freely and is reported below as a check on the zero-shift baseline.",
        "The center and four corners use fields (0,0), (-1,+1), (+1,+1), (-1,-1), (+1,-1).",
        "",
        "## Full-sweep fit errors",
        "",
        "These residuals compare the quadratic fitted to all 401 ground-truth angles "
        "with those same ground-truth samples. They measure approximation error over "
        "this sweep and do not measure eye-rotation precision.",
        "",
        "| Corner coordinate RMSE | MAE | Maximum absolute coordinate error | Corner vector RMSE | Vector error P95 | Maximum vector error |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {_format_mm_and_um(metrics['corner_coordinate_rmse_mm'])} "
        f"| {_format_mm_and_um(metrics['corner_coordinate_mae_mm'])} "
        f"| {_format_mm_and_um(metrics['corner_coordinate_max_abs_mm'])} "
        f"| {_format_mm_and_um(metrics['corner_vector_rmse_mm'])} "
        f"| {_format_mm_and_um(metrics['corner_vector_p95_mm'])} "
        f"| {_format_mm_and_um(metrics['corner_vector_max_mm'])} |",
        "",
        f"The largest absolute coordinate residual is {metrics['worst_coordinate']['error_mm'] * 1000:.6g} µm "
        f"at {metrics['worst_coordinate']['point']} {metrics['worst_coordinate']['axis'].upper()}, "
        f"θ={metrics['worst_coordinate']['angle_deg']:+g}°.",
        "",
        "## Per-coordinate errors",
        "",
        "Values are RMSE / maximum absolute error in µm. The center is included; "
        "its ideal shifts are zero by construction.",
        "",
        "| Point | Axis | RMSE / max absolute error (µm) |",
        "|---|---|---:|",
    ]
    for point, _, _ in POINTS:
        for axis in ("x", "y"):
            result = point_metrics[point][axis]
            lines.append(
                f"| {point} | {axis.upper()} | {result['rmse_mm'] * 1000:.6g} / "
                f"{result['max_abs_mm'] * 1000:.6g} |"
            )
    lines.extend((
        "",
        "## Fitted coefficients",
        "",
        "Coefficients use the Theory.md basis `[θ², θ, 1]`. For each coordinate, "
        "`dx = a_x θ² + b_x θ + c_x` and similarly for `dy`; units are "
        "mm/degree², mm/degree, and mm.",
        "",
        "| Point | a_x | b_x | c_x | a_y | b_y | c_y |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ))
    for point_index, (point, _, _) in enumerate(POINTS):
        cx = theta_coefficients[:, point_index * 2]
        cy = theta_coefficients[:, point_index * 2 + 1]
        values = (cx[0], cx[1], cx[2], cy[0], cy[1], cy[2])
        lines.append("| " + point + " | " + " | ".join(f"{v:.9g}" for v in values) + " |")
    lines.extend((
        "",
        "## Interpretation",
        "",
        "Compare these fit residuals with the assumed 1 µm per-coordinate localization "
        "noise from the existing eye-rotation study. The corner-coordinate RMSE is "
        f"{metrics['corner_coordinate_rmse_mm'] * 1000:.3g} µm and the maximum absolute "
        f"error is {metrics['corner_coordinate_max_abs_mm'] * 1000:.3g} µm. These values "
        "describe how closely a quadratic represents the ideal saved sweep. They do "
        "not establish rotation-estimation precision and omit detector calibration, "
        "alignment, localization bias, and other physical errors.",
        "",
        "## Files",
        "",
        "- `polynomial_distortion_fit.pkl`: ground truth, fitted coefficients, predictions, residuals, and full-sweep metrics.",
        "- `truth_vs_quadratic.png`: ground-truth shifts and quadratic predictions.",
        "- `quadratic_residuals.png`: full-fit coordinate residuals in µm.",
        "",
    ))
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_fit(angles: np.ndarray, shifts: np.ndarray, fitted: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True)
    colors = ("tab:blue", "tab:orange")
    for corner_index in range(4):
        for axis_index, ax in enumerate(axes[:, corner_index]):
            ax.plot(angles, shifts[:, corner_index + 1, axis_index], color=colors[axis_index],
                    linewidth=1.1, label="Ground truth")
            ax.plot(angles, fitted[:, corner_index + 1, axis_index], color=colors[axis_index],
                    linestyle="--", linewidth=1.3, label="Quadratic fit")
            ax.axvline(0, color="0.5", linestyle=":", linewidth=0.7)
            ax.grid(True, color="0.9", linewidth=0.6)
            ax.set_title(f"{POINTS[corner_index + 1][0]} · {'ΔX' if axis_index == 0 else 'ΔY'}")
            ax.set_ylabel("Shift from 0° (mm)")
            if axis_index == 1:
                ax.set_xlabel("Eye rotation θ (degrees)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Quadratic forward model fitted to ground-truth five-point shifts")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(angles: np.ndarray, residuals: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey="row")
    for corner_index in range(4):
        for axis_index, ax in enumerate(axes[:, corner_index]):
            error_um = residuals[:, corner_index + 1, axis_index] * 1000.0
            ax.plot(angles, error_um, color="tab:blue", linewidth=0.9)
            ax.axhline(0, color="0.35", linewidth=0.7)
            ax.axhline(1, color="0.55", linestyle=":", linewidth=0.65)
            ax.axhline(-1, color="0.55", linestyle=":", linewidth=0.65)
            ax.grid(True, color="0.9", linewidth=0.6)
            ax.set_title(f"{POINTS[corner_index + 1][0]} · {'X' if axis_index == 0 else 'Y'}")
            ax.set_ylabel("Prediction error (µm)")
            if axis_index == 1:
                ax.set_xlabel("Eye rotation θ (degrees)")
    fig.suptitle("Quadratic fit residuals; dotted lines mark ±1 µm")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(input_path: Path, output_dir: Path) -> dict:
    angles, coordinates = load_five_point_sweep(input_path)
    zero_index = int(np.flatnonzero(angles == 0.0)[0])
    shifts = coordinates - coordinates[zero_index:zero_index + 1]

    scaled_coefficients, theta_coefficients = fit_coefficients(angles, shifts)
    fitted = predict(angles, scaled_coefficients)
    residuals = fitted - shifts
    metrics = summarize_errors(residuals, angles)

    output_dir.mkdir(parents=True, exist_ok=True)
    fit_plot = output_dir / "truth_vs_quadratic.png"
    residual_plot = output_dir / "quadratic_residuals.png"
    model_path = output_dir / "polynomial_distortion_fit.pkl"
    report_path = output_dir / "report.md"
    plot_fit(angles, shifts, fitted, fit_plot)
    plot_residuals(angles, residuals, residual_plot)

    artifact = {
        "model": "quadratic forward model, Δ_i(θ) = [θ², θ, 1] C_i",
        "coefficient_basis": ("theta_squared", "theta", "constant"),
        "scaled_basis": ("(theta/20)^2", "theta/20", "constant"),
        "point_order": tuple(point[0] for point in POINTS),
        "coordinate_order": ("x", "y"),
        "angles_deg": angles,
        "coordinates_mm": coordinates,
        "zero_baseline_shifts_mm": shifts,
        "coefficients_scaled_by_point_xy": scaled_coefficients.reshape(3, len(POINTS), 2),
        "coefficients_theta_basis_by_point_xy": theta_coefficients.reshape(3, len(POINTS), 2),
        "full_fit_predictions_mm": fitted,
        "full_fit_residuals_mm": residuals,
        "full_fit_metrics": metrics,
        "input_pickle": str(input_path),
    }
    with model_path.open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
    make_report(report_path, angles, shifts, metrics, theta_coefficients)
    return {"artifact": artifact, "model_path": model_path, "report_path": report_path,
            "fit_plot": fit_plot, "residual_plot": residual_plot}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pkl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run(args.input_pkl, args.output_dir)
    metrics = result["artifact"]["full_fit_metrics"]
    print(f"full-fit corner coordinate RMSE={metrics['corner_coordinate_rmse_mm'] * 1000:.6g} µm, "
          f"MAE={metrics['corner_coordinate_mae_mm'] * 1000:.6g} µm, "
          f"max_abs={metrics['corner_coordinate_max_abs_mm'] * 1000:.6g} µm; "
          f"vector RMSE={metrics['corner_vector_rmse_mm'] * 1000:.6g} µm, "
          f"P95={metrics['corner_vector_p95_mm'] * 1000:.6g} µm, "
          f"max={metrics['corner_vector_max_mm'] * 1000:.6g} µm")
    worst = metrics["worst_coordinate"]
    print(f"  worst coordinate: {worst['point']} {worst['axis'].upper()} at "
          f"{worst['angle_deg']:+g}° = {worst['error_mm'] * 1000:.6g} µm")
    print(f"Wrote {result['model_path']}")
    print(f"Wrote {result['report_path']}")
    print(f"Wrote {result['fit_plot']}")
    print(f"Wrote {result['residual_plot']}")


if __name__ == "__main__":
    main()
