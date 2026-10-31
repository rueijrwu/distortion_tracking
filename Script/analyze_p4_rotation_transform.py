"""Analyze P4 empirical theta-zero barrel baselines and reduced keystone maps.

For each accommodation, the measured real grid at theta=0 is the source grid.
The script fits a three-parameter vertical-keystone transform to each rotation,
then characterizes the theta dependence with independent quadratics.
"""

from __future__ import annotations

import argparse
import hashlib
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
    """Apply X=sx*x/(1+q*y), Y=sy*y/(1+q*y) to a batch of grids."""
    sx, sy, q = (parameters[:, i, None] for i in range(3))
    x, y = source[:, :, 0], source[:, :, 1]
    denominator = 1.0 + q * y
    if np.any(denominator <= 0) or not np.isfinite(denominator).all():
        raise ValueError("Vertical keystone has a nonpositive/nonfinite sampled denominator")
    return np.stack((sx * x / denominator, sy * y / denominator), axis=2), denominator


def residual_jacobian(parameters: np.ndarray, source: np.ndarray,
                      target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prediction, denominator = predict(parameters, source)
    sx, sy, _ = (parameters[:, i, None] for i in range(3))
    x, y = source[:, :, 0], source[:, :, 1]
    d = denominator
    jx = np.stack((x / d, np.zeros_like(x), -sx * x * y / d**2), axis=2)
    jy = np.stack((np.zeros_like(y), y / d, -sy * y**2 / d**2), axis=2)
    jacobian = np.stack((jx, jy), axis=2).reshape(len(source), 18, 3)
    return (prediction - target).reshape(len(source), 18), jacobian


def fit_batched(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict]:
    """Batched algebraic initializer and damped Cartesian Gauss-Newton fit.

    API retained for analyze_p1_rotation_transform.py.
    """
    x, y = source[:, :, 0], source[:, :, 1]
    X, Y = target[:, :, 0], target[:, :, 1]
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
    """Fit independent ordinary quadratics using t=theta/20; P1-compatible API."""
    design = np.column_stack((np.ones_like(theta), theta / 20.0, (theta / 20.0) ** 2))
    coefficients, _, rank, singular = np.linalg.lstsq(design, values, rcond=None)
    fitted = design @ coefficients
    residual = fitted - values
    return coefficients, fitted, {
        "rank": int(rank),
        "condition_number": float(singular[0] / singular[-1]),
        "rmse_by_parameter": np.sqrt(np.mean(residual**2, axis=0)),
        "max_abs_by_parameter": np.max(np.abs(residual), axis=0),
    }


def spatial_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict:
    residual = prediction - truth
    points = np.linalg.norm(residual, axis=-1)
    return {
        "residual_xy_mm": residual,
        "coordinate_rmse_mm": np.sqrt(np.mean(residual**2, axis=(-2, -1))),
        "maximum_absolute_coordinate_error_mm": np.max(np.abs(residual), axis=(-2, -1)),
        "point_euclidean_error_mm": points,
        "rms_euclidean_point_error_mm": np.sqrt(np.mean(points**2, axis=-1)),
        "maximum_euclidean_point_error_mm": np.max(points, axis=-1),
    }


def make_plots(theta: np.ndarray, accommodation: np.ndarray, direct: np.ndarray,
               quadratic: np.ndarray, physical_coeff: np.ndarray,
               metrics: dict, accommodation_metrics: dict,
               center_abs: np.ndarray, center_relative: np.ndarray, output: Path) -> None:
    colors = plt.cm.viridis(np.linspace(0, 1, len(accommodation)))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for j, ax in enumerate(axes):
        for ai, a in enumerate(accommodation):
            ax.plot(theta, direct[ai, :, j], color=colors[ai], lw=1.0, alpha=0.8)
        for ai in (0, 10, 20, 30, 40, 50):
            ax.plot(theta, quadratic[ai, :, j], "--", color=colors[ai], lw=1.6,
                    label=f"A={accommodation[ai]:g} D quadratic")
        ax.set_title(f"{PARAMETER_NAMES[j]} ({PARAMETER_UNITS[j]})")
        ax.set_xlabel("Eye rotation, theta (degrees)")
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("P4 per-accommodation empirical-baseline keystone coefficients")
    fig.savefig(output / "keystone_coefficients_by_accommodation.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(15, 10), constrained_layout=True)
    basis_names = ("constant", "linear in theta (degree⁻¹)", "quadratic in theta (degree⁻²)")
    for j, name in enumerate(PARAMETER_NAMES):
        for k, basis_name in enumerate(basis_names):
            ax = axes[j, k]
            ax.plot(accommodation, physical_coeff[:, k, j], marker=".", ms=3)
            ax.set_title(f"{name} {basis_name}\n({PARAMETER_UNITS[j]})")
            ax.set_xlabel("Accommodation (D)")
            ax.grid(True, alpha=0.25)
    fig.suptitle("Physical-degree theta-quadratic coefficients by parameter and basis")
    fig.savefig(output / "physical_quadratic_coefficients_vs_accommodation.png", dpi=180)
    plt.close(fig)

    for name, values in (("direct_fit_error", metrics["direct"]["maximum_euclidean_point_error_mm"]),
                         ("polynomial_total_error", metrics["polynomial"]["maximum_euclidean_point_error_mm"]),
                         ("polynomial_additional_error", metrics["quadratic_vs_direct"]["maximum_euclidean_point_error_mm"]),
                         ("ignored_accommodation_additional_error", accommodation_metrics["additional"]["maximum_euclidean_point_error_mm"]),
                         ("ignored_accommodation_total_error", accommodation_metrics["total"]["maximum_euclidean_point_error_mm"])):
        fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
        mesh = ax.pcolormesh(theta, accommodation, values * 1000.0, shading="nearest", cmap="magma")
        cb = fig.colorbar(mesh, ax=ax)
        cb.set_label("Maximum Euclidean point error (µm)")
        ax.contour(theta, accommodation, values * 1000.0, levels=[1.0], colors="cyan", linewidths=1.4)
        ax.set_xlabel("Eye rotation, theta (degrees)")
        ax.set_ylabel("Accommodation (D)")
        ax.set_title(name.replace("_", " ").title() + " (cyan contour = 1 µm)")
        fig.savefig(output / f"{name}_heatmap.png", dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for ax, title, values in zip(axes,
                                 ("Direct coordinate RMSE", "Polynomial total coordinate RMSE",
                                  "Polynomial added coordinate RMSE"),
                                 (metrics["direct"]["coordinate_rmse_mm"],
                                  metrics["polynomial"]["coordinate_rmse_mm"],
                                  metrics["quadratic_vs_direct"]["coordinate_rmse_mm"])):
        mesh = ax.pcolormesh(theta, accommodation, values * 1000.0, shading="nearest", cmap="magma")
        fig.colorbar(mesh, ax=ax, label="Coordinate RMSE (µm)")
        ax.contour(theta, accommodation, values * 1000.0, levels=[1.0], colors="cyan", linewidths=1.3)
        ax.set_title(f"{title}\n(cyan contour = 1 µm)")
        ax.set_xlabel("Eye rotation, theta (degrees)")
        ax.set_ylabel("Accommodation (D)")
    fig.savefig(output / "coordinate_rmse_heatmaps.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ai in (0, 10, 20, 30, 40, 50):
        label = f"A={accommodation[ai]:g} D"
        axes[0, 0].plot(theta, center_abs[ai, :, 0], label=label)
        axes[0, 1].plot(theta, center_abs[ai, :, 1], label=label)
        axes[1, 0].plot(theta, center_relative[ai, :, 0], label=label)
        axes[1, 1].plot(theta, center_relative[ai, :, 1], label=label)
    for ax, title, coord in zip(axes.ravel(), ("Absolute center x", "Absolute center y",
                                               "Same-A theta-zero-relative center x",
                                               "Same-A theta-zero-relative center y"),
                                ("mm", "mm", "mm", "mm")):
        ax.axhline(0, color="black", lw=0.7, alpha=0.5)
        ax.set_title(f"{title} ({coord})")
        ax.set_xlabel("Eye rotation, theta (degrees)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    fig.savefig(output / "absolute_and_relative_centers.png", dpi=180)
    plt.close(fig)


def load_and_validate(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    raw_bytes = path.read_bytes()
    sha = hashlib.sha256(raw_bytes).hexdigest()
    records = pickle.loads(raw_bytes)
    required = {"eye_rotation_deg", "accommodation_d", "field_x_relative", "field_y_relative",
                "paraxial_x_mm", "paraxial_y_mm", "real_x_mm", "real_y_mm",
                "center_absolute_x_mm", "center_absolute_y_mm", "z_mm"}
    if not isinstance(records, np.ndarray) or records.dtype.names is None or not required.issubset(records.dtype.names):
        raise ValueError("Input is not the expected structured P4 grid array with absolute center fields")
    accommodation = np.unique(records["accommodation_d"])
    theta = np.unique(records["eye_rotation_deg"])
    accommodation.sort(); theta.sort()
    expected_a = np.linspace(0.0, 5.0, 51)
    expected_theta = np.linspace(-20.0, 20.0, 401)
    if not np.allclose(accommodation, expected_a, rtol=0, atol=1e-9):
        raise ValueError("Expected A=0..5 D at 0.1 D intervals")
    if not np.allclose(theta, expected_theta, rtol=0, atol=1e-9):
        raise ValueError("Expected theta=-20..20 degrees at 0.1 degree intervals")
    z_values = np.unique(records["z_mm"] if "z_mm" in records.dtype.names else np.array([0.0]))
    if not np.allclose(z_values, 0.0, rtol=0, atol=1e-9):
        raise ValueError(f"Expected z=0 data, found {z_values}")
    a_idx = np.searchsorted(accommodation, records["accommodation_d"])
    t_idx = np.searchsorted(theta, records["eye_rotation_deg"])
    counts = np.bincount(a_idx * len(theta) + t_idx, minlength=len(accommodation) * len(theta))
    if not np.all(counts == 9):
        raise ValueError(f"Expected nine records per A/theta state, counts are {np.unique(counts)}")
    order = np.lexsort((records["field_y_relative"], records["field_x_relative"], t_idx, a_idx))
    data = records[order]
    fields = np.column_stack((data["field_x_relative"], data["field_y_relative"])).reshape(len(accommodation), len(theta), 9, 2)
    expected_fields = np.array([(x, y) for x in (-1.0, 0.0, 1.0) for y in (-1.0, 0.0, 1.0)])
    if not np.isfinite(fields).all() or not np.array_equal(fields[0, 0], expected_fields) or not np.all(fields == fields[0:1, 0:1]):
        raise ValueError("Field identities/order do not match across every state or contain nonfinite values")
    shape = (len(accommodation), len(theta), 9)
    real = np.stack((data["real_x_mm"], data["real_y_mm"]), axis=1).reshape(*shape, 2)
    paraxial = np.stack((data["paraxial_x_mm"], data["paraxial_y_mm"]), axis=1).reshape(*shape, 2)
    center_raw = np.stack((data["center_absolute_x_mm"], data["center_absolute_y_mm"]), axis=1).reshape(*shape, 2)
    if not np.isfinite(real).all() or not np.isfinite(paraxial).all() or not np.isfinite(center_raw).all():
        raise ValueError("Input coordinates/centers contain nonfinite values")
    center = center_raw[:, :, 0, :]
    if not np.allclose(center_raw, center[:, :, None, :], rtol=0, atol=1e-10):
        raise ValueError("Absolute center is not repeated identically across the nine rows of a state")
    meta = {"input_sha256": sha, "input_bytes": len(raw_bytes), "z_values_mm": z_values,
            "records_per_state": 9, "raw_record_count": len(records),
            "selected_a_count": len(accommodation), "selected_theta_count": len(theta)}
    arrays = {"real": real, "paraxial": paraxial, "center_absolute": center}
    return accommodation, theta, fields[:, 0], {"provenance": meta, **arrays}


def error_dict_from_metrics(metrics: dict) -> dict:
    return {key: value for key, value in metrics.items() if key != "residual_xy_mm"}


def write_report(output: Path, a: np.ndarray, theta: np.ndarray, result: dict) -> None:
    d, p, q = (result["direct_metrics"], result["polynomial_total_metrics"], result["polynomial_additional_metrics"])
    poly = result["quadratic_coefficient_fit_metrics"]
    ignored = result["ignored_accommodation_metrics"]
    deg = result["quadratic_physical_degree_coefficients_constant_linear_quadratic"]
    max_where = np.unravel_index(np.argmax(d["maximum_euclidean_point_error_mm"]), d["maximum_euclidean_point_error_mm"].shape)
    lines = ["# P4 keystone analysis across accommodation and eye rotation", "",
             f"Input: `{result['input_path']}` ({result['provenance']['input_bytes']} bytes), SHA256 `{result['provenance']['input_sha256']}`. The structured array was validated for A=0..5 D by 0.1 D, theta=-20..20 degrees by 0.1 degrees, z=0, exactly the normalized 3x3 field identities in every state, finite coordinates, and repeated absolute centers.", "",
             "## Empirical baseline and fit", "",
             "At every accommodation A, the nine measured real coordinates at theta=0 are the source grid. The direct transform is fitted independently for each (A, theta): `X=sx*x0/(1+q*y0)`, `Y=sy*y0/(1+q*y0)`. This treats the measured theta-zero map as the empirical barrel baseline; no additional radial k1 term is added to those baseline coordinates. The fit has three unconstrained parameters and no translation/shear. The fit and quadratic use all angles in-sample.", "",
             "## Error definitions", "",
             "Direct error compares each angle's direct fitted prediction with its measured real grid. The order-2 parameter curves are fitted independently in theta for sx, sy, and q. Polynomial total error compares the composition of those coefficient curves with measured truth. Polynomial additional error compares that polynomial-composed prediction with the same-angle direct prediction, isolating coefficient-curve approximation error. They are stored separately. Coordinate RMSE pools both coordinates across the nine points; max absolute coordinate is over all 18 scalar errors; maximum Euclidean point error is the worst of nine 2D point errors. All reported errors below are in micrometers.", "",
             "| Metric | Minimum | Mean | Maximum |", "|---|---:|---:|---:|"]
    for label, obj in (("Direct coordinate RMSE", d), ("Direct max abs coordinate", d), ("Direct max Euclidean point", d),
                       ("Polynomial total coordinate RMSE", p), ("Polynomial total max Euclidean point", p),
                       ("Polynomial additional max Euclidean point", q)):
        key = {"Direct coordinate RMSE":"coordinate_rmse_mm", "Direct max abs coordinate":"maximum_absolute_coordinate_error_mm",
               "Direct max Euclidean point":"maximum_euclidean_point_error_mm", "Polynomial total coordinate RMSE":"coordinate_rmse_mm",
               "Polynomial total max Euclidean point":"maximum_euclidean_point_error_mm",
               "Polynomial additional max Euclidean point":"maximum_euclidean_point_error_mm"}[label]
        arr = obj[key] * 1000
        lines.append(f"| {label} (µm) | {np.min(arr):.6g} | {np.mean(arr):.6g} | {np.max(arr):.6g} |")
    lines += ["", f"Largest direct max-point error: {d['maximum_euclidean_point_error_mm'][max_where]*1000:.6g} µm at A={a[max_where[0]]:g} D, theta={theta[max_where[1]]:g} degrees.",
              f"Direct algebraic initializer rank values: {np.unique(result['solver_diagnostics']['initializer_rank'])}; condition range {np.min(result['solver_diagnostics']['initializer_condition_number']):.6g} to {np.max(result['solver_diagnostics']['initializer_condition_number']):.6g}. Converged fraction: {np.mean(result['solver_diagnostics']['converged']):.6%}; accepted steps range {np.min(result['solver_diagnostics']['accepted_refinement_steps'])} to {np.max(result['solver_diagnostics']['accepted_refinement_steps'])}. Minimum direct sampled denominator: {np.min(result['minimum_denominator_by_state']):.9g}; polynomial minimum: {np.min(result['polynomial_minimum_denominator_by_state']):.9g}.", "",
              "## Order-2 theta coefficient model", "",
              "Each parameter uses an independent unconstrained least-squares polynomial. Coefficients are stored in scaled basis `[1,t,t²]`, `t=theta/20`, and physical-degree basis `[1,theta,theta²]`; physical coefficients are also plotted against accommodation. This is an in-sample fit over all 401 angles. No even/odd symmetry or identity-at-zero constraint is imposed.", "",
              "At theta=0 the direct per-A fit maps each empirical baseline to itself. The fitted parameters are estimated and not constrained to identity; their ranges are:"]
    for j, name in enumerate(PARAMETER_NAMES):
        arr = result["direct_parameters_by_accommodation_theta"][:, int(np.argmin(np.abs(theta))), j]
        lines.append(f"- `{name}`: {np.min(arr):.12g} to {np.max(arr):.12g} ({PARAMETER_UNITS[j]}).")
    lines += ["", "Per-A spatial-error summary (mean and worst over theta):", "",
              "| A (D) | Direct mean max-point (µm) | Direct worst (µm) | Quadratic total mean max-point (µm) | Quadratic total worst (µm) | Quadratic added mean max-point (µm) | Quadratic added worst (µm) |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for ai in range(len(a)):
        lines.append(f"| {a[ai]:g} | {np.mean(d['maximum_euclidean_point_error_mm'][ai])*1000:.6g} | {np.max(d['maximum_euclidean_point_error_mm'][ai])*1000:.6g} | "
                     f"{np.mean(p['maximum_euclidean_point_error_mm'][ai])*1000:.6g} | {np.max(p['maximum_euclidean_point_error_mm'][ai])*1000:.6g} | "
              f"{np.mean(q['maximum_euclidean_point_error_mm'][ai])*1000:.6g} | {np.max(q['maximum_euclidean_point_error_mm'][ai])*1000:.6g} |")
    lines += ["", "Quadratic additional coordinate-RMSE summary (µm):",
              f"Minimum {np.min(q['coordinate_rmse_mm'])*1000:.6g}; mean {np.mean(q['coordinate_rmse_mm'])*1000:.6g}; maximum {np.max(q['coordinate_rmse_mm'])*1000:.6g}.",
              "Fractions of all A/theta states below 1 µm, by distinct error metric:"]
    for label, metric_set, key in (("direct coordinate RMSE", d, "coordinate_rmse_mm"),
                                   ("polynomial total coordinate RMSE", p, "coordinate_rmse_mm"),
                                   ("polynomial additional coordinate RMSE", q, "coordinate_rmse_mm"),
                                   ("direct maximum point error", d, "maximum_euclidean_point_error_mm"),
                                   ("polynomial total maximum point error", p, "maximum_euclidean_point_error_mm"),
                                   ("polynomial additional maximum point error", q, "maximum_euclidean_point_error_mm")):
        lines.append(f"- {label}: {np.mean(metric_set[key] < 0.001):.3%}.")
    lines += ["", "Representative physical-degree coefficients use `c0 + c1*theta_deg + c2*theta_deg^2`; c1 and c2 include per-degree and per-degree-squared units.",
              "", "| A (D) | Parameter | c0 | c1 (per degree) | c2 (per degree²) |", "|---:|---|---:|---:|---:|"]
    for ai in (0, 10, 20, 30, 40, 50):
        for j, name in enumerate(PARAMETER_NAMES):
            lines.append(f"| {a[ai]:g} | {name} ({PARAMETER_UNITS[j]}) | {deg[ai,0,j]:.10g} | {deg[ai,1,j]:.10g} | {deg[ai,2,j]:.10g} |")
    lines += ["", "At each A, the quadratic predictions are compared with direct and measured predictions separately below.", "",
              "At each A, coefficient fit RMSE by parameter (base parameter units):"]
    lines += ["", "| A (D) | sx RMSE | sy RMSE | q RMSE (mm^-1) | max coefficient condition |", "|---:|---:|---:|---:|---:|"]
    for ai in (0, 10, 20, 30, 40, 50):
        lines.append(f"| {a[ai]:g} | {poly['rmse_by_parameter'][ai,0]:.6g} | {poly['rmse_by_parameter'][ai,1]:.6g} | {poly['rmse_by_parameter'][ai,2]:.6g} | {poly['condition_number'][ai]:.6g} |")
    lines += ["", "## Ignoring accommodation", "", "For this diagnostic, each A's own measured theta-zero source grid is retained. Parameters fitted at A=0 are applied at every angle to each accommodation's own source. Additional error compares this A=0-parameter prediction to the per-A direct prediction; total error compares it to measured truth.", "",
              f"Ignoring-A additional max point error range: {ignored['additional']['maximum_euclidean_point_error_mm'].min()*1000:.6g}–{ignored['additional']['maximum_euclidean_point_error_mm'].max()*1000:.6g} µm; total max point error range: {ignored['total']['maximum_euclidean_point_error_mm'].min()*1000:.6g}–{ignored['total']['maximum_euclidean_point_error_mm'].max()*1000:.6g} µm.", "",
              "## Absolute center", "", "Absolute center x/y are validated as repeated across the nine rows per state. Raw center curves are stored and plotted separately from same-A changes `c(A,theta)-c(A,0)`. Accommodation baseline is separately stored as `c(A,0)-c(0,0)`. No center polynomial is imposed.",
              f"Across all A and theta, the largest absolute same-A theta-zero-relative center component is {np.max(np.abs(result['center_same_accommodation_theta0_relative_xy_mm']))*1000:.6g} µm. The largest A-baseline component at theta=0 relative to A=0 is {np.max(np.abs(result['center_accommodation_baseline_relative_to_A0_theta0_xy_mm']))*1000:.6g} µm.",
              "", "## Artifacts", "",
              "- `p4_rotation_transform.pkl`: direct and polynomial predictions, residual/error arrays, parameters, diagnostics, center arrays, and provenance.",
              "- `coefficient_quadratics.pkl`: scaled and physical-degree coefficient fits by A.",
              "- `keystone_coefficients_by_accommodation.png`, `physical_quadratic_coefficients_vs_accommodation.png`.",
              "- Error heatmaps for direct, polynomial total/additional, and ignoring-accommodation total/additional errors; 1 µm contours are cyan.",
              "- `absolute_and_relative_centers.png`.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    accommodation, theta, fields, loaded = load_and_validate(args.input)
    real, paraxial = loaded["real"], loaded["paraxial"]
    centers = loaded["center_absolute"]
    zero = int(np.argmin(np.abs(theta)))
    baseline = real[:, zero].copy()
    source = np.broadcast_to(baseline[:, None, :, :], real.shape).copy()
    flat_source, flat_real = source.reshape(-1, 9, 2), real.reshape(-1, 9, 2)
    parameters_flat, diagnostics_flat = fit_batched(flat_source, flat_real)
    parameters = parameters_flat.reshape(len(accommodation), len(theta), 3)
    direct_prediction_flat, denominator_flat = predict(parameters_flat, flat_source)
    direct_prediction = direct_prediction_flat.reshape(real.shape)
    direct_metrics_flat = spatial_metrics(direct_prediction_flat, flat_real)
    direct_metrics = {k: v.reshape(len(accommodation), len(theta), 9) if k == "point_euclidean_error_mm"
                      else v.reshape(len(accommodation), len(theta), *v.shape[1:]) for k, v in direct_metrics_flat.items()}

    scaled_coeff = np.empty((len(accommodation), 3, 3))
    quadratic = np.empty_like(parameters)
    coefficient_metrics = {"rmse_by_parameter": np.empty((len(accommodation), 3)),
                           "max_abs_by_parameter": np.empty((len(accommodation), 3)),
                           "rank": np.empty(len(accommodation), dtype=int),
                           "condition_number": np.empty(len(accommodation))}
    for ai in range(len(accommodation)):
        c, fitted, m = fit_quadratic(parameters[ai], theta)
        scaled_coeff[ai], quadratic[ai] = c, fitted
        for k in ("rmse_by_parameter", "max_abs_by_parameter"):
            coefficient_metrics[k][ai] = m[k]
        coefficient_metrics["rank"][ai] = m["rank"]
        coefficient_metrics["condition_number"][ai] = m["condition_number"]
    physical_coeff = scaled_coeff.copy()
    physical_coeff[:, 1] /= 20
    physical_coeff[:, 2] /= 400
    poly_prediction_flat, poly_denominator_flat = predict(quadratic.reshape(-1, 3), flat_source)
    poly_prediction = poly_prediction_flat.reshape(real.shape)
    poly_metrics_flat = spatial_metrics(poly_prediction_flat, flat_real)
    poly_metrics = {k: v.reshape(len(accommodation), len(theta), 9) if k == "point_euclidean_error_mm"
                    else v.reshape(len(accommodation), len(theta), *v.shape[1:]) for k, v in poly_metrics_flat.items()}
    quad_direct = predict(quadratic.reshape(-1, 3), flat_source)[0].reshape(real.shape)
    additional_poly = spatial_metrics(quad_direct, direct_prediction)
    additional_poly = {k: v.reshape(len(accommodation), len(theta), 9) if k == "point_euclidean_error_mm"
                       else v.reshape(real.shape) if k == "residual_xy_mm"
                       else v.reshape(len(accommodation), len(theta)) for k, v in additional_poly.items()}

    a0_parameters = parameters[0]
    ignored_pred = predict(np.broadcast_to(a0_parameters, parameters.shape).reshape(-1, 3), flat_source)[0].reshape(real.shape)
    ignored_total = spatial_metrics(ignored_pred, real)
    ignored_additional = spatial_metrics(ignored_pred, direct_prediction)
    ignored = {"total": {k: v for k, v in ignored_total.items() if k != "residual_xy_mm"},
               "additional": {k: v for k, v in ignored_additional.items() if k != "residual_xy_mm"},
               "prediction_xy_mm": ignored_pred,
               "total_residual_xy_mm": ignored_total["residual_xy_mm"],
               "additional_residual_xy_mm": ignored_additional["residual_xy_mm"]}

    direct_diag = {k: v.reshape(len(accommodation), len(theta)) for k, v in diagnostics_flat.items()
                   if k != "minimum_denominator_by_theta"}
    direct_diag["minimum_denominator_by_state"] = denominator_flat.min(axis=1).reshape(len(accommodation), len(theta))
    center_relative = centers - centers[:, zero:zero+1, :]
    center_accommodation_baseline = centers[:, zero, :] - centers[0, zero, :]
    result = {
        "input_path": str(args.input.resolve()), "provenance": loaded["provenance"],
        "accommodation_d": accommodation, "theta_deg": theta, "field_coordinates_relative": fields,
        "baseline_theta0_real_xy_mm_by_accommodation": baseline,
        "baseline_theta0_paraxial_xy_mm_by_accommodation": paraxial[:, zero],
        "baseline_mapping_assumption": "At each A, theta=0 measured real points are the empirical barrel baseline; no additional radial coefficient is added.",
        "composition_formula": "X=sx*x0/(1+q*y0); Y=sy*y0/(1+q*y0)",
        "parameter_names": PARAMETER_NAMES, "parameter_units": PARAMETER_UNITS,
        "direct_parameters_by_accommodation_theta": parameters,
        "quadratic_parameters_by_accommodation_theta": quadratic,
        "quadratic_scaled_coefficients_rows_basis_columns": scaled_coeff,
        "quadratic_physical_degree_coefficients_constant_linear_quadratic": physical_coeff,
        "predicted_real_xy_mm": direct_prediction,
        "direct_metrics": error_dict_from_metrics(direct_metrics),
        "direct_residual_xy_mm": direct_metrics["residual_xy_mm"],
        "polynomial_predicted_real_xy_mm": poly_prediction,
        "polynomial_total_metrics": error_dict_from_metrics(poly_metrics),
        "polynomial_total_residual_xy_mm": poly_metrics["residual_xy_mm"],
        "polynomial_additional_metrics": error_dict_from_metrics(additional_poly),
        "polynomial_additional_residual_xy_mm": additional_poly["residual_xy_mm"],
        "direct_point_euclidean_error_mm": direct_metrics["point_euclidean_error_mm"],
        "polynomial_total_point_euclidean_error_mm": poly_metrics["point_euclidean_error_mm"],
        "polynomial_additional_point_euclidean_error_mm": additional_poly["point_euclidean_error_mm"],
        "minimum_denominator_by_state": denominator_flat.min(axis=1).reshape(len(accommodation), len(theta)),
        "polynomial_minimum_denominator_by_state": poly_denominator_flat.min(axis=1).reshape(len(accommodation), len(theta)),
        "solver_diagnostics": direct_diag,
        "quadratic_coefficient_fit_metrics": coefficient_metrics,
        "ignored_accommodation_metrics": ignored,
        "center_absolute_xy_mm": centers,
        "center_same_accommodation_theta0_relative_xy_mm": center_relative,
        "center_accommodation_baseline_relative_to_A0_theta0_xy_mm": center_accommodation_baseline,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    make_plots(theta, accommodation, parameters, quadratic, physical_coeff,
               {"direct": result["direct_metrics"], "polynomial": result["polynomial_total_metrics"],
                "quadratic_vs_direct": result["polynomial_additional_metrics"]}, ignored,
               centers, center_relative, args.output_dir)
    with (args.output_dir / "p4_rotation_transform.pkl").open("wb") as stream:
        pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)
    with (args.output_dir / "coefficient_quadratics.pkl").open("wb") as stream:
        pickle.dump({"parameter_names": PARAMETER_NAMES, "parameter_units": PARAMETER_UNITS,
                     "accommodation_d": accommodation, "theta_deg": theta,
                     "basis_scaled": "[1,t,t^2], t=theta_deg/20", "basis_physical": "[1,theta_deg,theta_deg^2]",
                     "scaled_coefficients_rows_basis_columns": scaled_coeff,
                     "physical_degree_coefficients_constant_linear_quadratic": physical_coeff,
                     **coefficient_metrics, "fit_scope": "per-accommodation unconstrained OLS over all 401 angle estimates"}, stream,
                    protocol=pickle.HIGHEST_PROTOCOL)
    write_report(args.output_dir, accommodation, theta, result)
    ranks = np.unique(direct_diag["initializer_rank"])
    print(f"Input SHA256: {loaded['provenance']['input_sha256']}")
    print(f"Wrote P4 accommodation/rotation analysis for {len(accommodation)} x {len(theta)} states to {args.output_dir.resolve()}")
    print(f"Direct max-point error range: {direct_metrics['maximum_euclidean_point_error_mm'].min()*1000:.6g}–{direct_metrics['maximum_euclidean_point_error_mm'].max()*1000:.6g} um")
    print(f"Polynomial-total max-point error range: {poly_metrics['maximum_euclidean_point_error_mm'].min()*1000:.6g}–{poly_metrics['maximum_euclidean_point_error_mm'].max()*1000:.6g} um")
    print(f"Polynomial-additional max-point error max: {additional_poly['maximum_euclidean_point_error_mm'].max()*1000:.6g} um")
    print(f"Ignoring accommodation additional max-point error max: {ignored_additional['maximum_euclidean_point_error_mm'].max()*1000:.6g} um")
    print(f"Rank values {ranks}; convergence {np.mean(direct_diag['converged']):.6%}; minimum sampled denominator {denominator_flat.min():.9g}")


if __name__ == "__main__":
    main()
