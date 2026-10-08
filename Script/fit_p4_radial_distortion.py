"""Fit one radial k1 coefficient to P4's zero-rotation accommodation grids."""

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
DEFAULT_INPUT = PROJECT_DIR / "data" / "distortion_grid_p4" / "distortion_grid.pkl"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "p4_radial_distortion_fit"
REQUIRED_FIELDS = (
    "eye_rotation_deg",
    "accommodation_d",
    "paraxial_x_mm",
    "paraxial_y_mm",
    "real_x_mm",
    "real_y_mm",
)
RESULT_DTYPE = np.dtype(
    [
        ("accommodation_d", np.float64),
        ("k1_per_mm2", np.float64),
        ("coordinate_rmse_mm", np.float64),
        ("max_abs_residual_mm", np.float64),
    ]
)
MODEL_DTYPE = np.dtype(
    [("model", "U32"), ("coefficient_0", np.float64), ("coefficient_1", np.float64),
     ("coefficient_2", np.float64), ("rmse_k1_per_mm2", np.float64),
     ("max_abs_error_k1_per_mm2", np.float64), ("rmse_fraction_of_k1_span", np.float64)]
)


def load_zero_rotation_groups(input_path: Path) -> list[tuple[float, np.ndarray]]:
    if not input_path.is_file():
        raise FileNotFoundError(f"P4 distortion pickle not found: {input_path}")
    with input_path.open("rb") as stream:
        data = pickle.load(stream)
    if not isinstance(data, np.ndarray) or data.ndim != 1 or data.dtype.names is None:
        raise ValueError("Input pickle must contain a 1D structured NumPy array.")
    missing = [name for name in REQUIRED_FIELDS if name not in data.dtype.names]
    if missing:
        raise ValueError(f"Input pickle is missing fields: {', '.join(missing)}")

    zero_rotation = data[
        np.isclose(data["eye_rotation_deg"], 0.0, rtol=0.0, atol=1e-8)
    ]
    accommodations = np.unique(zero_rotation["accommodation_d"])
    if accommodations.size != 51:
        raise ValueError(
            f"Expected 51 zero-rotation accommodation groups, found {accommodations.size}."
        )

    groups = []
    for accommodation in accommodations:
        rows = zero_rotation[
            np.isclose(
                zero_rotation["accommodation_d"], accommodation, rtol=0.0, atol=1e-8
            )
        ]
        if len(rows) != 9:
            raise ValueError(
                f"Accommodation {accommodation:g} D has {len(rows)} points; expected 9."
            )
        coordinates = np.column_stack(
            [rows[name] for name in REQUIRED_FIELDS[2:]]
        )
        if not np.all(np.isfinite(coordinates)):
            raise ValueError(
                f"Accommodation {accommodation:g} D contains nonfinite coordinates."
            )
        groups.append((float(accommodation), rows))
    return groups


def fit_group(accommodation: float, rows: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(rows["paraxial_x_mm"], dtype=np.float64)
    y = np.asarray(rows["paraxial_y_mm"], dtype=np.float64)
    real_x = np.asarray(rows["real_x_mm"], dtype=np.float64)
    real_y = np.asarray(rows["real_y_mm"], dtype=np.float64)

    radius_squared = x**2 + y**2
    design = np.concatenate([x * radius_squared, y * radius_squared])
    displacement = np.concatenate([real_x - x, real_y - y])
    denominator = float(np.dot(design, design))
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError(f"Accommodation {accommodation:g} D has no fit leverage.")

    k1 = float(np.dot(design, displacement) / denominator)
    residual = displacement - k1 * design
    coordinate_rmse = float(np.sqrt(np.mean(residual**2)))
    max_abs_residual = float(np.max(np.abs(residual)))
    return k1, coordinate_rmse, max_abs_residual


def fit_accommodation_models(results: np.ndarray) -> np.ndarray:
    accommodation = results["accommodation_d"]
    k1 = results["k1_per_mm2"]
    quadratic_design = np.column_stack([np.ones_like(accommodation), accommodation, accommodation**2])
    quadratic_coeff = np.linalg.lstsq(quadratic_design, k1, rcond=None)[0]
    quadratic_prediction = quadratic_design @ quadratic_coeff

    def profiled_exponential(slope: float) -> tuple[float, float, float, float]:
        # k1 = q0 + q1*expm1(b*A)/b equals d + a*exp(b*A), with
        # d=q0-q1/b and a=q1/b. The basis tends smoothly to A as b->0.
        if abs(slope) < 1e-12:
            basis = accommodation
        else:
            basis = np.expm1(slope * accommodation) / slope
        design = np.column_stack([np.ones_like(accommodation), basis])
        q0, q1 = np.linalg.lstsq(design, k1, rcond=None)[0]
        prediction = design @ np.array([q0, q1])
        residual = prediction - k1
        return float(np.dot(residual, residual)), float(q0), float(q1), float(np.max(np.abs(residual)))

    def minimize_bracket(left: float, right: float) -> tuple[float, float]:
        phi = (np.sqrt(5.0) - 1.0) / 2.0
        x1, x2 = right - phi * (right - left), left + phi * (right - left)
        f1, f2 = profiled_exponential(x1)[0], profiled_exponential(x2)[0]
        for _ in range(100):
            if f1 <= f2:
                right, x2, f2 = x2, x1, f1
                x1 = right - phi * (right - left)
                f1 = profiled_exponential(x1)[0]
            else:
                left, x1, f1 = x1, x2, f2
                x2 = left + phi * (right - left)
                f2 = profiled_exponential(x2)[0]
        slope = (left + right) / 2.0
        return slope, profiled_exponential(slope)[0]

    # Scan both signs over a broad range, then refine each interior local
    # minimum. Expand if the best candidate lands on the scan boundary.
    scan_low, scan_high = -2.0, 2.0
    best_slope = 0.0
    boundary_winner = True
    for _ in range(5):
        scan = np.linspace(scan_low, scan_high, 4001)
        objectives = np.array([profiled_exponential(float(b))[0] for b in scan])
        minima = [i for i in range(1, len(scan) - 1)
                  if objectives[i] <= objectives[i - 1] and objectives[i] <= objectives[i + 1]]
        candidates = [(float(objectives[0]), float(scan[0])),
                      (float(objectives[-1]), float(scan[-1]))]
        candidates.extend(
            (minimize_bracket(float(scan[i - 1]), float(scan[i + 1]))[1],
             minimize_bracket(float(scan[i - 1]), float(scan[i + 1]))[0])
            for i in minima
        )
        objective, best_slope = min(candidates)
        boundary_winner = best_slope in (float(scan[0]), float(scan[-1]))
        if not boundary_winner:
            break
        scan_low *= 2.0
        scan_high *= 2.0
    if boundary_winner:
        raise RuntimeError("Offset exponential fit remained at a slope-search boundary.")

    _, q0, q1, _ = profiled_exponential(best_slope)
    exp_a = q1 / best_slope
    exp_d = q0 - exp_a
    exp_prediction = exp_d + exp_a * np.exp(best_slope * accommodation)

    models = np.empty(2, dtype=MODEL_DTYPE)
    for index, (name, c0, c1, c2, prediction) in enumerate(
        (("offset_exponential", exp_d, exp_a, best_slope, exp_prediction),
         ("quadratic_A", quadratic_coeff[0], quadratic_coeff[1], quadratic_coeff[2], quadratic_prediction))
    ):
        residual = prediction - k1
        rmse = float(np.sqrt(np.mean(residual**2)))
        models[index] = (name, c0, c1, c2, rmse, np.max(np.abs(residual)), rmse / np.ptp(k1))
    return models


def model_predictions(models: np.ndarray, accommodation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    exp_model = models[models["model"] == "offset_exponential"][0]
    quadratic_model = models[models["model"] == "quadratic_A"][0]
    exp_prediction = exp_model["coefficient_0"] + exp_model["coefficient_1"] * np.exp(exp_model["coefficient_2"] * accommodation)
    quadratic_prediction = (
        quadratic_model["coefficient_0"]
        + quadratic_model["coefficient_1"] * accommodation
        + quadratic_model["coefficient_2"] * accommodation**2
    )
    return exp_prediction, quadratic_prediction


def write_plot(results: np.ndarray, models: np.ndarray, path: Path) -> None:
    fig, (ax, residual_ax) = plt.subplots(
        2, 1, figsize=(9.2, 7.0), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax.scatter(
        results["accommodation_d"],
        results["k1_per_mm2"],
        marker="o",
        s=14,
        color="#225ea8",
        label="Per-accommodation k1 fits",
    )
    accommodation_grid = np.linspace(float(np.min(results["accommodation_d"])), float(np.max(results["accommodation_d"])), 301)
    prediction_exp, prediction_quadratic = model_predictions(models, accommodation_grid)
    curves = [
        ("offset_exponential", prediction_exp, "#238b45", "d + a exp(b A)"),
        ("quadratic_A", prediction_quadratic, "#756bb1", "p0 + p1 A + p2 A²"),
    ]
    for name, prediction, color, label in curves:
        ax.plot(accommodation_grid, prediction, color=color, label=label)
        model = models[models["model"] == name][0]
        if name == "offset_exponential":
            sample_prediction = model["coefficient_0"] + model["coefficient_1"] * np.exp(model["coefficient_2"] * results["accommodation_d"])
        else:
            a = results["accommodation_d"]
            sample_prediction = model["coefficient_0"] + model["coefficient_1"] * a + model["coefficient_2"] * a**2
        residual_ax.plot(results["accommodation_d"], (sample_prediction - results["k1_per_mm2"]) * 1e4, color=color, label=label)
    residual_ax.axhline(0, color="black", linewidth=0.8)
    residual_ax.set_ylabel(r"k1 error ($10^{-4}$ mm$^{-2}$)")
    residual_ax.grid(True, alpha=0.3)
    residual_ax.legend(fontsize=8, ncol=2, loc="upper center")
    residual_ax.set_xlabel("Accommodation (D)")
    ax.set_ylabel(r"Radial coefficient $k_1$ (mm$^{-2}$)")
    ax.set_title("P4 radial distortion fit at 0° eye rotation")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(results: np.ndarray, models: np.ndarray, path: Path) -> None:
    errors = results["coordinate_rmse_mm"]
    maxima = results["max_abs_residual_mm"]
    lines = [
        "# P4 radial distortion fit",
        "",
        "This fit uses only the existing P4 grids at eye rotation 0° and accommodation 0–5 D. Each accommodation group contains nine field points.",
        "",
        "The forward model is `real_x = x * (1 + k1*r^2)` and `real_y = y * (1 + k1*r^2)`, where `(x, y)` are paraxial image coordinates and `r^2 = x^2 + y^2`. This maps ideal/paraxial coordinates to real distorted coordinates. Coordinates are in millimeters, so `k1` is in mm^-2. One coefficient is fit by pooled least squares over all 18 x/y coordinates; coordinate RMSE is sqrt(mean of 18 squared coordinate residuals), including the center's two zero residuals. The stored real coordinates are already relative to the traced center ray.",
        "The 3 × 3 grid has only two distinct nonzero sampled radii (edge midpoints and corners); the center has zero radius and contributes no leverage. The paraxial grid scale also varies with accommodation, so the model is evaluated against each group's own stored ideal coordinates.",
        "",
        "## Accommodation models for fitted k1",
        "",
        "The accommodation curves are fit to the 51 per-accommodation k1 estimates. The offset exponential is `k1 = d + a*exp(b*A)` with signed, unconstrained coefficients and direct least squares in k1 space (coefficient order d, a, b). It is optimized by scanning b across [-2, 2] D^-1, refining every local minimum, and expanding the interval if the best result is at a boundary; the selected optimum is interior. A stable profiled basis `expm1(b*A)/b` is used, with linear limit A at b=0. For the exponential, d and a are in mm^-2 and b is in D^-1. The quadratic is `k1 = p0 + p1*A + p2*A^2` (coefficient order p0, p1, p2; units mm^-2, mm^-2/D and mm^-2/D^2). Model RMSE is sqrt(mean((prediction-k1)^2)) in mm^-2; the percent column divides RMSE by the observed k1 range (max-min). These are coefficient-curve errors, distinct from coordinate RMSE in mm above. The lower plot panel shows coefficient errors in units of 10^-4 mm^-2.",
        "",
        "| Model | Coefficient 0 | Coefficient 1 | Coefficient 2 | RMSE k1 (mm^-2) | RMSE (% of k1 range) | Maximum absolute k1 error (mm^-2) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['model']} | {row['coefficient_0']:.12g} | {row['coefficient_1']:.12g} | {row['coefficient_2']:.12g} | {row['rmse_k1_per_mm2']:.8g} | {100*row['rmse_fraction_of_k1_span']:.5g}% | {row['max_abs_error_k1_per_mm2']:.8g} |"
        for row in models
    )
    lines.extend([
        "",
        f"Across {len(results)} accommodation groups, coordinate RMSE ranges from {np.min(errors):.6g} to {np.max(errors):.6g} mm (mean {np.mean(errors):.6g} mm). Maximum absolute coordinate residual ranges from {np.min(maxima):.6g} to {np.max(maxima):.6g} mm (mean {np.mean(maxima):.6g} mm). These residuals show the adequacy of this one-parameter radial model for the sampled nine-point grids.",
        "",
        "| Accommodation (D) | k1 (mm^-2) | Coordinate RMSE (mm) | Maximum absolute residual (mm) |",
        "|---:|---:|---:|---:|",
    ])
    lines.extend(
        f"| {row['accommodation_d']:.1f} | {row['k1_per_mm2']:.10g} | {row['coordinate_rmse_mm']:.6g} | {row['max_abs_residual_mm']:.6g} |"
        for row in results
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    groups = load_zero_rotation_groups(args.input)
    results = np.empty(len(groups), dtype=RESULT_DTYPE)
    for index, (accommodation, rows) in enumerate(groups):
        k1, rmse, maximum = fit_group(accommodation, rows)
        results[index] = (accommodation, k1, rmse, maximum)
    models = fit_accommodation_models(results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pickle_path = args.output_dir / "p4_radial_distortion_fit.pkl"
    with pickle_path.open("wb") as stream:
        pickle.dump(results, stream, protocol=pickle.HIGHEST_PROTOCOL)
    model_pickle_path = args.output_dir / "p4_radial_distortion_accommodation_models.pkl"
    with model_pickle_path.open("wb") as stream:
        pickle.dump(models, stream, protocol=pickle.HIGHEST_PROTOCOL)
    plot_path = args.output_dir / "p4_radial_distortion_k1.png"
    report_path = args.output_dir / "p4_radial_distortion_fit.md"
    write_plot(results, models, plot_path)
    write_report(results, models, report_path)

    print(f"Validated and fit {len(results)} groups with 9 points each.")
    print(f"k1 range: {np.min(results['k1_per_mm2']):.10g} to {np.max(results['k1_per_mm2']):.10g} mm^-2")
    print(f"Coordinate RMSE range: {np.min(results['coordinate_rmse_mm']):.6g} to {np.max(results['coordinate_rmse_mm']):.6g} mm")
    print(f"Max absolute residual range: {np.min(results['max_abs_residual_mm']):.6g} to {np.max(results['max_abs_residual_mm']):.6g} mm")
    for model in models:
        print(f"{model['model']}: coefficient_0={model['coefficient_0']:.12g}, coefficient_1={model['coefficient_1']:.12g}, coefficient_2={model['coefficient_2']:.12g}, RMSE={model['rmse_k1_per_mm2']:.8g} mm^-2 ({100*model['rmse_fraction_of_k1_span']:.5g}% of range), max_abs={model['max_abs_error_k1_per_mm2']:.8g} mm^-2")
    print(f"Saved {pickle_path.resolve()}")
    print(f"Saved {model_pickle_path.resolve()}")
    print(f"Saved {plot_path.resolve()}")
    print(f"Saved {report_path.resolve()}")


if __name__ == "__main__":
    main()
