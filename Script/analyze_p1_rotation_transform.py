"""Fit P1 baseline barrel plus vertical keystone across eye rotation.

Uses the actual P1 theta=0 real grid as a fixed source and never starts CODE V.
The batched solver and quadratic fitter are shared with the P4 analysis.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_p4_rotation_transform import fit_batched, fit_quadratic, predict


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
INPUT_PATH = PROJECT_DIR / "data" / "distortion_grid" / "distortion_grid.pkl"
OUTPUT_DIR = PROJECT_DIR / "data" / "p1_rotation_transform"
PARAMETER_NAMES = ("sx", "sy", "q")
PARAMETER_UNITS = ("unitless", "unitless", "mm^-1")


def write_plots(output: Path, theta: np.ndarray, parameters: np.ndarray,
                quadratic: np.ndarray, coordinate_rmse: np.ndarray,
                quadratic_rmse: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for i, axis in enumerate(axes):
        axis.plot(theta, parameters[:, i], color="tab:blue", linewidth=1.5)
        axis.plot(theta, quadratic[:, i], color="tab:orange", linestyle="--", linewidth=1.3)
        axis.set_title(f"{PARAMETER_NAMES[i]} ({PARAMETER_UNITS[i]})")
        axis.set_xlabel("Eye rotation, theta (degrees)")
        axis.grid(True, alpha=0.25)
    axes[0].legend(("direct per-angle fit", "quadratic coefficient fit"), fontsize=8)
    fig.suptitle("P1 baseline barrel map plus vertical keystone")
    fig.savefig(output / "keystone_coefficients.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    axis.plot(theta, coordinate_rmse * 1000, color="tab:blue", linewidth=1.5)
    axis.plot(theta, quadratic_rmse * 1000, color="tab:orange", linestyle="--", linewidth=1.3)
    axis.legend(("direct fit", "quadratic-composed prediction"), fontsize=8)
    axis.set_xlabel("Eye rotation, theta (degrees)")
    axis.set_ylabel("Coordinate RMSE over 9 points and x/y (micrometers)")
    axis.set_title("P1 composed-model spatial error")
    axis.grid(True, alpha=0.25)
    fig.savefig(output / "coordinate_rmse.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    with args.input.open("rb") as stream:
        records = pickle.load(stream)
    required = {"eye_rotation_deg", "field_x_relative", "field_y_relative",
                "paraxial_x_mm", "paraxial_y_mm", "real_x_mm", "real_y_mm"}
    if not isinstance(records, np.ndarray) or records.dtype.names is None or not required.issubset(records.dtype.names):
        raise ValueError("Input is not the expected structured P1 grid array")
    theta = np.unique(records["eye_rotation_deg"])
    theta.sort()
    if theta.size != 401 or not np.allclose(theta, np.linspace(-20, 20, 401), rtol=0, atol=1e-9):
        raise ValueError("Expected all 401 rotations from -20 to +20 degrees")
    zero_idx = int(np.argmin(np.abs(theta)))
    angle_index = np.searchsorted(theta, records["eye_rotation_deg"])
    counts = np.bincount(angle_index, minlength=len(theta))
    if not np.all(counts == 9):
        raise ValueError(f"Expected nine records per angle, got counts {np.unique(counts)}")
    order = np.lexsort((records["field_y_relative"], records["field_x_relative"], angle_index))
    data = records[order]
    fields = np.column_stack((data["field_x_relative"], data["field_y_relative"])).reshape(401, 9, 2)
    if not np.all(fields == fields[0:1]):
        raise ValueError("Field identities do not match across rotations")
    real = np.stack((data["real_x_mm"], data["real_y_mm"]), axis=1).reshape(401, 9, 2)
    paraxial = np.stack((data["paraxial_x_mm"], data["paraxial_y_mm"]), axis=1).reshape(401, 9, 2)
    if not np.isfinite(real).all() or not np.isfinite(paraxial).all():
        raise ValueError("Input coordinates contain nonfinite values")

    baseline_real = real[zero_idx].copy()
    baseline_paraxial = paraxial[zero_idx].copy()
    source = np.broadcast_to(baseline_real, real.shape).copy()
    parameters, diagnostics = fit_batched(source, real)
    prediction, denominators = predict(parameters, source)
    residuals = prediction - real
    coordinate_rmse = np.sqrt(np.mean(residuals**2, axis=(1, 2)))
    max_coordinate_error = np.max(np.abs(residuals), axis=(1, 2))
    point_errors = np.linalg.norm(residuals, axis=2)
    point_rms_error = np.sqrt(np.mean(point_errors**2, axis=1))
    max_point_index = np.argmax(point_errors, axis=1)
    max_point_error = np.max(point_errors, axis=1)
    max_point_field = fields[np.arange(len(theta)), max_point_index]

    scaled_coefficients, quadratic_parameters, coefficient_metrics = fit_quadratic(parameters, theta)
    physical_coefficients = scaled_coefficients.copy()
    physical_coefficients[1] /= 20.0
    physical_coefficients[2] /= 400.0
    quadratic_prediction, quadratic_denominator = predict(quadratic_parameters, source)
    if np.any(quadratic_denominator <= 0):
        raise ValueError("Quadratic coefficient curve has a nonpositive sampled denominator")
    quadratic_residuals = quadratic_prediction - real
    quadratic_coordinate_rmse = np.sqrt(np.mean(quadratic_residuals**2, axis=(1, 2)))
    quadratic_point_errors = np.linalg.norm(quadratic_residuals, axis=2)
    quadratic_point_rms_error = np.sqrt(np.mean(quadratic_point_errors**2, axis=1))
    quadratic_max_point_error = np.max(quadratic_point_errors, axis=1)
    homographies = np.zeros((len(theta), 3, 3))
    homographies[:, 0, 0] = parameters[:, 0]
    homographies[:, 1, 1] = parameters[:, 1]
    homographies[:, 2, 1] = parameters[:, 2]
    homographies[:, 2, 2] = 1.0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_plots(args.output_dir, theta, parameters, quadratic_parameters,
                coordinate_rmse, quadratic_coordinate_rmse)
    result = {
        "theta_deg": theta,
        "field_coordinates_relative": fields[0],
        "baseline_theta0_real_xy_mm": baseline_real,
        "baseline_theta0_paraxial_xy_mm": baseline_paraxial,
        "baseline_mapping_assumption": "actual P1 theta=0 real coordinates are fixed source; baseline radial/barrel distortion is preserved at nine samples",
        "composition_formula": "X=sx*x0/(1+q*y0); Y=sy*y0/(1+q*y0)",
        "parameter_names": PARAMETER_NAMES,
        "parameter_units": PARAMETER_UNITS,
        "parameters_by_theta": parameters,
        "quadratic_parameters_by_theta": quadratic_parameters,
        "quadratic_physical_degree_coefficients_constant_linear_quadratic": physical_coefficients,
        "restricted_homography_matrices_3x3": homographies,
        "predicted_real_xy_mm": prediction,
        "point_residuals_xy_mm": residuals,
        "coordinate_rmse_mm": coordinate_rmse,
        "maximum_absolute_coordinate_error_mm": max_coordinate_error,
        "point_euclidean_errors_mm": point_errors,
        "rms_euclidean_point_error_mm": point_rms_error,
        "maximum_point_error_mm": max_point_error,
        "maximum_point_field_coordinates_relative": max_point_field,
        "quadratic_predicted_real_xy_mm": quadratic_prediction,
        "quadratic_point_residuals_xy_mm": quadratic_residuals,
        "quadratic_coordinate_rmse_mm": quadratic_coordinate_rmse,
        "quadratic_point_rms_error_mm": quadratic_point_rms_error,
        "quadratic_maximum_point_error_mm": quadratic_max_point_error,
        "coefficient_curve_rmse": coefficient_metrics["rmse_by_parameter"],
        "coefficient_curve_max_abs_error": coefficient_metrics["max_abs_by_parameter"],
        "minimum_denominator_by_theta": np.min(denominators, axis=1),
        "quadratic_minimum_denominator_by_theta": np.min(quadratic_denominator, axis=1),
        "solver_diagnostics": diagnostics,
    }
    with (args.output_dir / "p1_rotation_transform.pkl").open("wb") as stream:
        pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)
    coefficient_artifact = {
        "parameter_names": PARAMETER_NAMES,
        "parameter_units": PARAMETER_UNITS,
        "basis_scaled": "[1,t,t^2], t=theta_deg/20",
        "basis_physical": "[1,theta_deg,theta_deg^2]",
        "scaled_coefficients_rows_basis_columns": scaled_coefficients,
        "physical_degree_coefficients_constant_linear_quadratic": physical_coefficients,
        "coefficient_curve_rmse_by_parameter": coefficient_metrics["rmse_by_parameter"],
        "coefficient_curve_max_abs_error_by_parameter": coefficient_metrics["max_abs_by_parameter"],
        "fit_rank": coefficient_metrics["rank"],
        "fit_condition_number": coefficient_metrics["condition_number"],
        "fit_scope": "unconstrained ordinary least squares over all 401 direct coefficient estimates; in-sample",
    }
    with (args.output_dir / "coefficient_quadratics.pkl").open("wb") as stream:
        pickle.dump(coefficient_artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)

    samples = [int(np.argmin(np.abs(theta - angle))) for angle in (-20, -15, -10, -5, 0, 5, 10, 15, 20)]
    lines = [
        "# P1 composed baseline barrel and vertical keystone", "",
        "All 401 angles from -20 to +20 degrees and all nine fields were used. P1 has no accommodation dimension.", "",
        "## Model and fit scope", "",
        "The fixed source is the actual simulated real grid at theta=0. It preserves the P1 baseline barrel distortion at the sampled fields; baseline paraxial coordinates are retained as context and are not the transform source. The rotation-dependent mapping is `X=sx*x0/(1+q*y0)`, `Y=sy*y0/(1+q*y0)`, with homography `[[sx,0,0],[0,sy,0],[0,q,1]]`. The origin is centered, so translation is zero. P1 shows exact left-right reflection parity while top-bottom parity changes with theta, supporting the vertical-keystone orientation. This reduced model excludes shear and horizontal perspective.", "",
        "The shared P4 implementation supplies the batched algebraic initializer and damped Cartesian Gauss-Newton refinement, plus the ordinary least-squares coefficient fitter. There is one batched solve across 401 angle grids and no per-angle solve loop. The coefficient quadratics use every direct fitted angle in-sample and the scaled basis `[1,t,t^2]`, `t=theta/20`; reported coefficients use physical theta degrees.", "",
        "## Physical-degree coefficient fits", "",
        "| Parameter | Units | c0 | c1 per degree | c2 per degree^2 | Coefficient RMSE | Maximum coefficient error |", "|---|---|---:|---:|---:|---:|---:|",
    ]
    for j, name in enumerate(PARAMETER_NAMES):
        lines.append(f"| {name} | {PARAMETER_UNITS[j]} | {physical_coefficients[0,j]:.12g} | {physical_coefficients[1,j]:.12g} | {physical_coefficients[2,j]:.12g} | {coefficient_metrics['rmse_by_parameter'][j]:.8g} | {coefficient_metrics['max_abs_by_parameter'][j]:.8g} |")
    lines += ["", "## Spatial errors", "", "Coordinate RMSE is sqrt(mean(rx^2, ry^2)) over nine points and both coordinates (18 scalar residuals). RMS Euclidean point error is sqrt(mean(rx^2+ry^2)) over nine points. Maximum point error is the largest Euclidean residual among the nine points. Thus point RMS is sqrt(2) times coordinate RMSE for this common 18-coordinate population; they are reported separately for clarity.", "", "| theta deg | Direct coordinate RMSE um | Quadratic coordinate RMSE um | Direct RMS point um | Quadratic RMS point um | Direct max point um | Quadratic max point um |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for i in samples:
        lines.append(f"| {theta[i]:g} | {coordinate_rmse[i]*1000:.6g} | {quadratic_coordinate_rmse[i]*1000:.6g} | {point_rms_error[i]*1000:.6g} | {quadratic_point_rms_error[i]*1000:.6g} | {max_point_error[i]*1000:.6g} | {quadratic_max_point_error[i]*1000:.6g} |")
    zero_idx = int(np.argmin(np.abs(theta)))
    largest_field = max_point_field[np.argmax(max_point_error)]
    lines += ["", f"At theta=0, the direct baseline-to-itself fit estimates sx={parameters[zero_idx,0]:.12g}, sy={parameters[zero_idx,1]:.12g}, q={parameters[zero_idx,2]:.12g} mm^-1. It was not forced to identity.", "", f"Across the sweep direct coordinate RMSE is {coordinate_rmse.min()*1000:.6g}–{coordinate_rmse.max()*1000:.6g} um (mean {coordinate_rmse.mean()*1000:.6g}); RMS Euclidean point error is {point_rms_error.min()*1000:.6g}–{point_rms_error.max()*1000:.6g} um (mean {point_rms_error.mean()*1000:.6g}); maximum point error is {max_point_error.min()*1000:.6g}–{max_point_error.max()*1000:.6g} um. Quadratic-composed coordinate RMSE is {quadratic_coordinate_rmse.min()*1000:.6g}–{quadratic_coordinate_rmse.max()*1000:.6g} um (mean {quadratic_coordinate_rmse.mean()*1000:.6g}); RMS Euclidean point error is {quadratic_point_rms_error.min()*1000:.6g}–{quadratic_point_rms_error.max()*1000:.6g} um (mean {quadratic_point_rms_error.mean()*1000:.6g}); maximum point error is {quadratic_max_point_error.min()*1000:.6g}–{quadratic_max_point_error.max()*1000:.6g} um.", "", f"The largest direct point residual is {max_point_error.max()*1000:.6g} um at theta={theta[np.argmax(max_point_error)]:g} degrees, relative field ({float(largest_field[0]):g}, {float(largest_field[1]):g}).", f"Initializer rank is {np.unique(diagnostics['initializer_rank'])}; condition number range is {diagnostics['initializer_condition_number'].min():.6g}–{diagnostics['initializer_condition_number'].max():.6g}. All nonlinear fits converged: {bool(np.all(diagnostics['converged']))}; accepted refinement steps range {diagnostics['accepted_refinement_steps'].min()}–{diagnostics['accepted_refinement_steps'].max()}.", f"Minimum denominator over sampled fields and angles is {diagnostics['minimum_denominator_by_theta'].min():.9g}; quadratic-curve minimum is {np.min(quadratic_denominator):.9g}.", "", "## Artifacts", "", "- `p1_rotation_transform.pkl`: baseline grids, direct and quadratic coefficient series, homographies, predictions, residuals, metrics, and diagnostics.", "- `coefficient_quadratics.pkl`: scaled and physical-degree polynomial coefficient artifact.", "- `keystone_coefficients.png`: three coefficient panels with quadratic overlays.", "- `coordinate_rmse.png`: direct and quadratic-composed spatial RMSE.", ""]
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote P1 composed baseline-plus-keystone analysis to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
