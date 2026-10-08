"""Fit baseline-barrel plus vertical keystone transforms across P4 eye rotation.

The fixed source is the measured real A=0 grid at theta=0. This script never
starts CODE V and fits one direct three-parameter transform at each angle.
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
INPUT_PATH = PROJECT_DIR / "data" / "distortion_grid_p4" / "distortion_grid.pkl"
OUTPUT_DIR = PROJECT_DIR / "data" / "p4_rotation_transform"
PARAMETER_NAMES = ("sx", "sy", "q")
PARAMETER_UNITS = ("unitless", "unitless", "mm^-1")


def predict(parameters: np.ndarray, source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sx, sy, q = (parameters[:, i, None] for i in range(3))
    x, y = source[:, :, 0], source[:, :, 1]
    denominator = 1.0 + q * y
    if np.any(denominator <= 0) or not np.isfinite(denominator).all():
        raise ValueError("Vertical keystone has a nonpositive/nonfinite sampled denominator")
    projected = np.stack((sx * x / denominator, sy * y / denominator), axis=2)
    return projected, denominator


def residual_jacobian(parameters: np.ndarray, source: np.ndarray,
                      target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prediction, denominator = predict(parameters, source)
    sx, sy, _ = (parameters[:, i, None] for i in range(3))
    x, y = source[:, :, 0], source[:, :, 1]
    d = denominator
    jx = np.stack((x / d, np.zeros_like(x), -sx * x * y / d**2), axis=2)
    jy = np.stack((np.zeros_like(y), y / d, -sy * y**2 / d**2), axis=2)
    jacobian = np.stack((jx, jy), axis=2).reshape(len(source), 18, 3)
    residual = (prediction - target).reshape(len(source), 18)
    return residual, jacobian


def fit_batched(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict]:
    """Batched algebraic initializer and damped Cartesian Gauss-Newton refinement."""
    x, y = source[:, :, 0], source[:, :, 1]
    X, Y = target[:, :, 0], target[:, :, 1]
    # Rearrange X=sx*x/(1+q*y), Y=sy*y/(1+q*y) into linear least squares.
    design_x = np.stack((x, np.zeros_like(x), -X * y), axis=2)
    design_y = np.stack((np.zeros_like(y), y, -Y * y), axis=2)
    design = np.stack((design_x, design_y), axis=2).reshape(len(source), 18, 3)
    rhs = target.reshape(len(source), 18, 1)
    singular = np.linalg.svd(design, compute_uv=False)
    ranks = np.linalg.matrix_rank(design)
    if np.any(ranks < 3):
        raise ValueError(f"Algebraic initializer rank below three: {np.unique(ranks)}")
    condition = singular[:, 0] / singular[:, -1]
    parameters = (np.linalg.pinv(design) @ rhs)[:, :, 0]

    residual, jacobian = residual_jacobian(parameters, source, target)
    costs = np.sum(residual**2, axis=1)
    damping = np.full(len(source), 1e-6)
    converged = np.zeros(len(source), dtype=bool)
    accepted_steps = np.zeros(len(source), dtype=np.int32)
    for _ in range(100):
        jt = np.swapaxes(jacobian, 1, 2)
        normal = jt @ jacobian
        gradient = (jt @ residual[:, :, None])[:, :, 0]
        diagonal = np.maximum(np.diagonal(normal, axis1=1, axis2=2), 1e-12)
        damped = normal + damping[:, None, None] * np.eye(3)[None, :, :] * diagonal[:, None, :]
        step = np.linalg.solve(damped, gradient[:, :, None])[:, :, 0]
        candidate = parameters - step
        try:
            candidate_residual, candidate_jacobian = residual_jacobian(candidate, source, target)
            candidate_cost = np.sum(candidate_residual**2, axis=1)
            valid = np.isfinite(candidate_cost) & np.isfinite(candidate).all(axis=1)
        except ValueError:
            candidate_residual, candidate_jacobian = residual, jacobian
            candidate_cost = np.full_like(costs, np.inf)
            valid = np.zeros(len(source), dtype=bool)
        accepted = valid & (candidate_cost < costs)
        parameters[accepted] = candidate[accepted]
        residual[accepted] = candidate_residual[accepted]
        jacobian[accepted] = candidate_jacobian[accepted]
        costs[accepted] = candidate_cost[accepted]
        accepted_steps[accepted] += 1
        damping[accepted] = np.maximum(damping[accepted] / 3.0, 1e-12)
        damping[~accepted] = np.minimum(damping[~accepted] * 10.0, 1e12)
        step_norm = np.max(np.abs(step) / np.maximum(np.abs(parameters), 1.0), axis=1)
        converged |= valid & (step_norm < 1e-11)
        if np.all(converged | (damping >= 1e12)):
            break

    prediction, denominator = predict(parameters, source)
    error = prediction - target
    return parameters, {
        "initializer_rank": ranks,
        "initializer_condition_number": condition,
        "converged": converged,
        "accepted_refinement_steps": accepted_steps,
        "minimum_denominator_by_theta": np.min(denominator, axis=1),
        "sum_squared_coordinate_error_mm2": np.sum(error**2, axis=(1, 2)),
    }


def fit_quadratic(values: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit independent ordinary quadratics against t=theta/20 over all angles."""
    scaled_theta = theta / 20.0
    design = np.column_stack((np.ones_like(theta), scaled_theta, scaled_theta**2))
    coefficients, _, rank, singular = np.linalg.lstsq(design, values, rcond=None)
    fitted = design @ coefficients
    residual = fitted - values
    metrics = {
        "rank": int(rank),
        "condition_number": float(singular[0] / singular[-1]),
        "rmse_by_parameter": np.sqrt(np.mean(residual**2, axis=0)),
        "max_abs_by_parameter": np.max(np.abs(residual), axis=0),
    }
    return coefficients, fitted, metrics


def make_plots(theta: np.ndarray, parameters: np.ndarray, quadratic: np.ndarray,
               coordinate_rmse: np.ndarray, quadratic_rmse: np.ndarray, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for i, axis in enumerate(axes):
        axis.plot(theta, parameters[:, i], color="tab:blue", linewidth=1.5)
        axis.plot(theta, quadratic[:, i], color="tab:orange", linestyle="--", linewidth=1.3)
        axis.set_title(f"{PARAMETER_NAMES[i]} ({PARAMETER_UNITS[i]})")
        axis.set_xlabel("Eye rotation, theta (degrees)")
        axis.grid(True, alpha=0.25)
    fig.suptitle("Baseline barrel map plus vertical keystone, accommodation 0 D")
    axes[0].legend(("direct per-angle fit", "quadratic coefficient fit"), fontsize=8)
    fig.savefig(output / "keystone_coefficients.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    axis.plot(theta, coordinate_rmse * 1000.0, color="tab:blue", linewidth=1.5)
    axis.plot(theta, quadratic_rmse * 1000.0, color="tab:orange", linestyle="--", linewidth=1.3)
    axis.legend(("direct fit", "quadratic-composed prediction"), fontsize=8)
    axis.set_xlabel("Eye rotation, theta (degrees)")
    axis.set_ylabel("Coordinate RMSE over 9 points and x/y (µm)")
    axis.set_title("Single composed-model residual, accommodation 0 D")
    axis.grid(True, alpha=0.25)
    fig.savefig(output / "coordinate_rmse.png", dpi=180)
    plt.close(fig)


def write_report(output: Path, theta: np.ndarray, parameters: np.ndarray,
                 quadratic_coefficients_physical: np.ndarray,
                 coefficient_rmse: np.ndarray, coefficient_max: np.ndarray,
                 coordinate_rmse: np.ndarray, max_coordinate_error: np.ndarray,
                 point_rms_error: np.ndarray, max_point_error: np.ndarray,
                 quadratic_coordinate_rmse: np.ndarray,
                 quadratic_point_rms_error: np.ndarray,
                 quadratic_max_point_error: np.ndarray,
                 max_point_field: np.ndarray, diagnostics: dict) -> None:
    zero_idx = int(np.argmin(np.abs(theta)))
    lines = [
        "# P4 composed baseline barrel and vertical keystone at 0 D", "",
        "The analysis selects accommodation with `isclose(0, rtol=0, atol=1e-8)` and matches all 401 rotations to theta=0 by `(field_x_relative, field_y_relative)`.", "",
        "## Composition and model", "",
        "The fixed source points are the actual simulated **real** coordinates at accommodation 0 D and theta=0. Those nine coordinates and their paraxial baseline coordinates are saved in the pickle. This preserves the measured baseline barrel distortion exactly at the sampled points; each rotation's paraxial coordinates are not used as the transform source.", "",
        "The only rotation-dependent mapping is reduced vertical keystone: `X=sx*x0/(1+q*y0)`, `Y=sy*y0/(1+q*y0)`, with homography `[[sx,0,0],[0,sy,0],[0,q,1]]`. Center-relative coordinates set translations to zero; symmetry excludes shear and horizontal perspective. The shared denominator changes horizontal width with y as well as vertical position. `sx` and `sy` are dimensionless; `q` is mm^-1.", "",
        "A batched algebraic least-squares estimate initializes a damped Gauss-Newton refinement against Cartesian squared reprojection error. All 401 fits are processed together. The theta=0 baseline-to-itself coefficients are fitted from data, not assigned by hand; sampled denominators must remain positive.", "",
        "## Coefficient dependence", "",
        "The blue curve is the independently fitted coefficient at each rotation. The dashed curve is a separate unconstrained ordinary least-squares quadratic fitted to all 401 direct coefficients (all ground-truth angles, in-sample). Its internal basis is `[1,t,t²]` with `t=theta/20`; the report table converts this to `c(theta)=c0+c1*theta+c2*theta²`. No symmetry constraints or held-out angles are used, and the fit does not force the theta=0 intercept to identity.", "",
        "| Parameter | Units | Constant c0 | Linear c1 (per degree) | Quadratic c2 (per degree²) | Coefficient RMSE | Coefficient max error |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for j, name in enumerate(PARAMETER_NAMES):
        lines.append(f"| {name} | {PARAMETER_UNITS[j]} | {quadratic_coefficients_physical[0,j]:.12g} | "
                     f"{quadratic_coefficients_physical[1,j]:.12g} | {quadratic_coefficients_physical[2,j]:.12g} | "
                     f"{coefficient_rmse[j]:.8g} | {coefficient_max[j]:.8g} |")
    lines.extend(["", "The reported coefficient order is `(sx, sy, q)`. Coefficient-fit RMSE and maximum error use each parameter's base units. Units are unitless for sx/sy and mm^-1 for q; c1 and c2 carry the corresponding per-degree and per-degree-squared factors. The coefficients are stored both in the scaled basis and in physical-degree form.", "",
                  "## Per-angle spatial errors", "",
                  "Coordinate RMSE is `sqrt(mean(rx², ry²))` over nine points and both coordinates (18 scalar residuals). RMS Euclidean point error is `sqrt(mean(rx²+ry²))` over the nine points. Maximum Euclidean point error is the largest `sqrt(rx²+ry²)` among the nine points. Direct-fit columns come from each independent transform; quadratic columns evaluate the quadratic coefficient transform against the same actual grid.", "",
                  "| theta (degrees) | Direct coordinate RMSE (µm) | Quadratic coordinate RMSE (µm) | Direct RMS point (µm) | Quadratic RMS point (µm) | Direct max point (µm) | Quadratic max point (µm) |",
                  "|---:|---:|---:|---:|---:|---:|---:|"])
    samples = [int(np.argmin(np.abs(theta - angle))) for angle in (-20, -15, -10, -5, 0, 5, 10, 15, 20)]
    for i in samples:
        lines.append(f"| {theta[i]:g} | {coordinate_rmse[i]*1000:.6g} | {quadratic_coordinate_rmse[i]*1000:.6g} | "
                     f"{point_rms_error[i]*1000:.6g} | {quadratic_point_rms_error[i]*1000:.6g} | "
                     f"{max_point_error[i]*1000:.6g} | {quadratic_max_point_error[i]*1000:.6g} |")
    lines.extend(["", f"At theta=0, the direct baseline-to-itself fit gives `sx={parameters[zero_idx,0]:.12g}`, `sy={parameters[zero_idx,1]:.12g}`, `q={parameters[zero_idx,2]:.12g} mm^-1`; these values were estimated rather than forced. The quadratic intercept is reported in the coefficient table without being constrained to these values.", "",
                  "## Residual definitions and fit accuracy", "",
                  "Coordinate RMSE is `sqrt(mean(rx², ry²))` over nine points and both coordinates (18 scalar residuals). RMS Euclidean point error is `sqrt(mean(rx²+ry²))` over the nine points. Maximum Euclidean point error is the largest `sqrt(rx²+ry²)` among the nine points.", "",
                  f"Coordinate RMSE is {coordinate_rmse.min()*1000:.6g}–{coordinate_rmse.max()*1000:.6g} µm (mean {coordinate_rmse.mean()*1000:.6g} µm).",
                  f"RMS Euclidean point error is {point_rms_error.min()*1000:.6g}–{point_rms_error.max()*1000:.6g} µm (mean {point_rms_error.mean()*1000:.6g} µm).",
                  f"Maximum Euclidean point error is {max_point_error.min()*1000:.6g}–{max_point_error.max()*1000:.6g} µm.",
                  f"Quadratic-composed coordinate RMSE is {quadratic_coordinate_rmse.min()*1000:.6g}–{quadratic_coordinate_rmse.max()*1000:.6g} µm (mean {quadratic_coordinate_rmse.mean()*1000:.6g} µm).",
                  f"Quadratic-composed maximum Euclidean point error is {quadratic_max_point_error.min()*1000:.6g}–{quadratic_max_point_error.max()*1000:.6g} µm.",
                  f"Largest point residual: {max_point_error.max()*1000:.6g} µm at theta={theta[np.argmax(max_point_error)]:g}°, field ({max_point_field[np.argmax(max_point_error),0]:g}, {max_point_field[np.argmax(max_point_error),1]:g}).",
                  f"Batched algebraic initializer rank is 3 at every angle; condition number range {diagnostics['initializer_condition_number'].min():.6g}–{diagnostics['initializer_condition_number'].max():.6g}.",
                  f"All nonlinear fits converged: {bool(np.all(diagnostics['converged']))}; accepted damped steps range {diagnostics['accepted_refinement_steps'].min()}–{diagnostics['accepted_refinement_steps'].max()}.",
                  f"Minimum fitted denominator across sampled points: {diagnostics['minimum_denominator_by_theta'].min():.8g}.", "",
                  "## Artifacts", "",
                  "- `p4_rotation_transform.pkl`: baseline grids, direct and quadratic coefficient series, restricted homographies, predictions, residuals, errors, and solver diagnostics.",
                  "- `coefficient_quadratics.pkl`: scaled and physical-degree coefficient fits, basis/order/units, and curve errors.",
                  "- `keystone_coefficients.png`: direct `sx`, `sy`, and `q` dependence plus quadratic coefficient curves.",
                  "- `coordinate_rmse.png`: direct and quadratic-composed coordinate RMSE over the sweep.", ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    with args.input.open("rb") as stream:
        records = pickle.load(stream)
    required = {"eye_rotation_deg", "accommodation_d", "field_x_relative", "field_y_relative",
                "paraxial_x_mm", "paraxial_y_mm", "real_x_mm", "real_y_mm"}
    if not isinstance(records, np.ndarray) or records.dtype.names is None or not required.issubset(records.dtype.names):
        raise ValueError("Input is not the expected structured P4 grid array")
    data = records[np.isclose(records["accommodation_d"], 0.0, rtol=0.0, atol=1e-8)]
    theta = np.unique(data["eye_rotation_deg"])
    theta.sort()
    if theta.size != 401 or not np.allclose(theta, np.linspace(-20, 20, 401), rtol=0, atol=1e-9):
        raise ValueError("Expected all 401 rotations from -20 to +20 degrees at accommodation 0 D")
    zero_idx = int(np.argmin(np.abs(theta)))
    angle_index = np.searchsorted(theta, data["eye_rotation_deg"])
    counts = np.bincount(angle_index, minlength=len(theta))
    if not np.all(counts == 9):
        raise ValueError(f"Expected nine records per angle, got counts {np.unique(counts)}")
    order = np.lexsort((data["field_y_relative"], data["field_x_relative"], angle_index))
    data = data[order]
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
    # scaled columns are [constant, linear-in-t, quadratic-in-t], t=theta/20.
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
    make_plots(theta, parameters, quadratic_parameters, coordinate_rmse,
               quadratic_coordinate_rmse, args.output_dir)
    result = {
        "theta_deg": theta,
        "accommodation_selected_d": 0.0,
        "accommodation_atol_d": 1e-8,
        "field_coordinates_relative": fields[0],
        "baseline_theta0_real_xy_mm": baseline_real,
        "baseline_theta0_paraxial_xy_mm": baseline_paraxial,
        "baseline_mapping_assumption": "actual A=0 theta=0 real coordinates are fixed source; baseline radial/barrel distortion is preserved at nine samples",
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
    with (args.output_dir / "p4_rotation_transform.pkl").open("wb") as stream:
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
    write_report(args.output_dir, theta, parameters,
                 physical_coefficients, coefficient_metrics["rmse_by_parameter"],
                 coefficient_metrics["max_abs_by_parameter"], coordinate_rmse, max_coordinate_error,
                 point_rms_error, max_point_error, quadratic_coordinate_rmse,
                 quadratic_point_rms_error, quadratic_max_point_error,
                 max_point_field, diagnostics)
    print(f"Wrote single composed baseline-plus-keystone analysis to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
