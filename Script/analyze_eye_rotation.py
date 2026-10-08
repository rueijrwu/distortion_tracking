"""Study eye-rotation estimation from five measured grid points.

The script reads the completed CODE V sweep and its saved quadratic model. It
never imports or starts CODE V. The reusable ``estimate_rotation_polynomial``
function accepts five absolute measured image coordinates in millimeters,
ordered center, top-left, top-right, bottom-left, bottom-right.
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
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "eye_rotation_analysis"
DEFAULT_MODEL_PICKLE = PROJECT_DIR / "data" / "polynomial_distortion_fit" / "polynomial_distortion_fit.pkl"
CORNER_NAMES = ("top-left", "top-right", "bottom-left", "bottom-right")
CORNER_FIELDS = ((-1.0, 1.0), (1.0, 1.0), (-1.0, -1.0), (1.0, -1.0))
POINT_ORDER = ("center", *CORNER_NAMES)
SCALED_BASIS = ("(theta/20)^2", "theta/20", "constant")
PHYSICAL_BASIS = ("theta_squared", "theta", "constant")
ANGLE_SCALE_DEG = 20.0
DEG_TO_ARCMIN = 60.0
SIGMA_MM = (0.001,)  # assumed 1 micrometre standard deviation per X/Y coordinate
SEED = 20261008
REQUIRED_FIELDS = (
    "eye_rotation_deg", "field_x_relative", "field_y_relative", "real_x_mm",
    "real_y_mm", "radial_distortion_pct", "tangential_distortion_pct",
)


def load_sweep(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Sweep pickle not found: {path}")
    with path.open("rb") as stream:
        data = pickle.load(stream)
    if not isinstance(data, np.ndarray) or data.ndim != 1 or data.dtype.names is None:
        raise ValueError("Input must be a one-dimensional structured NumPy array.")
    missing = [name for name in REQUIRED_FIELDS if name not in data.dtype.names]
    if missing:
        raise ValueError(f"Input is missing fields: {', '.join(missing)}")
    angles = np.unique(data["eye_rotation_deg"])
    if angles.size < 3 or not np.all(np.isfinite(angles)):
        raise ValueError("At least three finite rotation templates are required.")

    # Stored real image coordinates are relative to the traced center at each
    # rotation. Keep field matching explicit so row ordering is irrelevant.
    xy = np.empty((angles.size, 4, 2), dtype=float)
    distortion = np.empty((angles.size, 4, 2), dtype=float)
    for ai, angle in enumerate(angles):
        rows = data[data["eye_rotation_deg"] == angle]
        for ci, (fx, fy) in enumerate(CORNER_FIELDS):
            mask = (np.isclose(rows["field_x_relative"], fx, rtol=0, atol=1e-9)
                    & np.isclose(rows["field_y_relative"], fy, rtol=0, atol=1e-9))
            selected = rows[mask]
            if selected.size != 1:
                raise ValueError(f"RC={angle:g} needs exactly one field point ({fx:g},{fy:g}).")
            row = selected[0]
            xy[ai, ci] = (row["real_x_mm"], row["real_y_mm"])
            distortion[ai, ci] = (row["radial_distortion_pct"], row["tangential_distortion_pct"])
    if not np.all(np.isfinite(xy)) or not np.all(np.isfinite(distortion)):
        raise ValueError("Corner templates contain non-finite values.")
    if not np.any(angles == 0.0):
        raise ValueError("The sweep must include an exact 0 degree baseline.")
    return angles.astype(float), xy, distortion


def weighted_distance_sq(a: np.ndarray, b: np.ndarray, weight: np.ndarray) -> float:
    delta = np.asarray(a) - np.asarray(b)
    return float(sum(delta[:, axis] @ weight @ delta[:, axis] for axis in range(2)))


def load_polynomial_model(path: Path, angles: np.ndarray, xy: np.ndarray) -> dict:
    """Load coefficients, verify their basis/order, and match them to the sweep."""
    if not path.is_file():
        raise FileNotFoundError(f"Polynomial model pickle not found: {path}")
    with path.open("rb") as stream:
        model = pickle.load(stream)
    if not isinstance(model, dict):
        raise ValueError("Polynomial model pickle must contain a dictionary.")
    if tuple(model.get("scaled_basis", ())) != SCALED_BASIS:
        raise ValueError(f"Expected scaled monomial basis {SCALED_BASIS}.")
    if tuple(model.get("coefficient_basis", ())) != PHYSICAL_BASIS:
        raise ValueError(f"Expected physical coefficient basis {PHYSICAL_BASIS}.")
    if tuple(model.get("point_order", ())) != POINT_ORDER:
        raise ValueError(f"Model point order must be {POINT_ORDER}.")
    if tuple(model.get("coordinate_order", ())) != ("x", "y"):
        raise ValueError("Model coordinate order must be ('x', 'y').")

    model_angles = np.asarray(model.get("angles_deg"), dtype=float)
    model_coordinates = np.asarray(model.get("coordinates_mm"), dtype=float)
    coefficients = np.asarray(model.get("coefficients_scaled_by_point_xy"), dtype=float)
    physical_coefficients = np.asarray(model.get("coefficients_theta_basis_by_point_xy"), dtype=float)
    if not np.array_equal(model_angles, angles):
        raise ValueError("Polynomial model angles do not exactly match the ground-truth sweep.")
    if model_coordinates.shape != (angles.size, 5, 2) or coefficients.shape != (3, 5, 2):
        raise ValueError("Polynomial model has unexpected coordinate or coefficient dimensions.")
    if physical_coefficients.shape != coefficients.shape:
        raise ValueError("Physical-basis coefficient array has unexpected dimensions.")
    if not np.allclose(model_coordinates[:, 1:, :], xy, rtol=0.0, atol=1e-12):
        raise ValueError("Polynomial fit coordinates do not match the input sweep corners.")

    scaled_theta = model_angles / ANGLE_SCALE_DEG
    scaled_design = np.column_stack((scaled_theta ** 2, scaled_theta, np.ones_like(scaled_theta)))
    scaled_prediction = (scaled_design @ coefficients.reshape(3, -1)).reshape(angles.size, 5, 2)
    stored_prediction = np.asarray(model.get("full_fit_predictions_mm"), dtype=float)
    if stored_prediction.shape != scaled_prediction.shape or not np.allclose(
        scaled_prediction, stored_prediction, rtol=0.0, atol=1e-12
    ):
        raise ValueError("Scaled coefficients do not reproduce the model's stored predictions.")

    physical_design = np.column_stack((model_angles ** 2, model_angles, np.ones_like(model_angles)))
    physical_prediction = (
        physical_design @ physical_coefficients.reshape(3, -1)
    ).reshape(angles.size, 5, 2)
    if not np.allclose(physical_prediction, scaled_prediction, rtol=0.0, atol=1e-12):
        raise ValueError("Physical coefficients do not match the scaled coefficients.")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("Polynomial coefficients contain non-finite values.")
    zero_index = int(np.flatnonzero(model_angles == 0.0)[0])
    return {
        "coefficients_scaled_by_point_xy": coefficients,
        "baseline_corner_coordinates_mm": model_coordinates[zero_index, 1:, :].copy(),
        "fit_metrics": model["full_fit_metrics"],
        "model_pickle": str(path.resolve()),
        "scaled_basis": SCALED_BASIS,
        "physical_basis": PHYSICAL_BASIS,
        "point_order": POINT_ORDER,
        "coordinate_order": ("x", "y"),
        "max_scaled_prediction_check_error_mm": float(np.max(np.abs(scaled_prediction - stored_prediction))),
        "max_basis_conversion_check_error_mm": float(np.max(np.abs(physical_prediction - scaled_prediction))),
    }


def predict_polynomial_shift(theta_deg: float, coefficients_scaled_by_point_xy: np.ndarray) -> np.ndarray:
    """Predict center-relative shifts using scaled monomial order [t^2, t, 1]."""
    t = float(theta_deg) / ANGLE_SCALE_DEG
    basis = np.array((t * t, t, 1.0))
    return np.tensordot(basis, coefficients_scaled_by_point_xy, axes=(0, 0))


def _polynomial_objective(observed_shift_mm: np.ndarray, coefficients: np.ndarray,
                          t: float, weight: np.ndarray) -> float:
    predicted = predict_polynomial_shift(t * ANGLE_SCALE_DEG, coefficients)[1:, :]
    return weighted_distance_sq(predicted, observed_shift_mm, weight)


def _polynomial_candidates(observed_shift_mm: np.ndarray, coefficients: np.ndarray,
                           bounds_deg: tuple[float, float]) -> tuple[list[float], int]:
    """Return all admissible objective candidates in normalized t coordinates.

    The weighted squared residual is quartic in t, so every bounded global
    minimum lies at a range endpoint or at a real root of its cubic derivative.
    """
    observed = np.asarray(observed_shift_mm, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    if observed.shape != (4, 2) or not np.all(np.isfinite(observed)):
        raise ValueError("Observed corner shifts must be a finite (4, 2) array in mm.")
    if coefficients.shape != (3, 5, 2) or not np.all(np.isfinite(coefficients)):
        raise ValueError("Scaled polynomial coefficients must have shape (3, 5, 2).")
    lower_deg, upper_deg = map(float, bounds_deg)
    if not np.isfinite(lower_deg + upper_deg) or lower_deg >= upper_deg:
        raise ValueError("Rotation bounds must be finite and increasing.")
    lower_t, upper_t = lower_deg / ANGLE_SCALE_DEG, upper_deg / ANGLE_SCALE_DEG

    # Residual coefficients in increasing powers of t, for the four corners.
    corner_coefficients = coefficients[:, 1:, :]
    residual_coefficients = np.stack(
        (corner_coefficients[2] - observed, corner_coefficients[1], corner_coefficients[0]),
        axis=0,
    )  # (power, corner, axis)
    weight = np.eye(4) - np.ones((4, 4)) / 5.0
    objective_coefficients = np.zeros(5, dtype=float)
    for axis in range(2):
        for i in range(4):
            for j in range(4):
                product = np.polynomial.polynomial.polymul(
                    residual_coefficients[:, i, axis], residual_coefficients[:, j, axis]
                )
                objective_coefficients[:product.size] += weight[i, j] * product

    coefficient_scale = float(np.max(np.abs(objective_coefficients)))
    candidates = [lower_t, upper_t]
    if coefficient_scale <= np.finfo(float).tiny:
        # A flat objective has no rotation information; return its midpoint.
        candidates.append((lower_t + upper_t) / 2.0)
        return candidates, 0
    normalized_objective = objective_coefficients / coefficient_scale
    derivative = np.polynomial.polynomial.polyder(normalized_objective)
    derivative_scale = float(np.max(np.abs(derivative))) if derivative.size else 0.0
    stationary_count = 0
    if derivative_scale > 0.0:
        derivative = derivative / derivative_scale
        nonzero = np.flatnonzero(np.abs(derivative) > 1e-13)
        if nonzero.size >= 2:
            derivative = derivative[:nonzero[-1] + 1]
            roots = np.polynomial.polynomial.polyroots(derivative)
            for root in roots:
                tolerance = 1e-9 * (1.0 + abs(float(np.real(root))))
                if abs(float(np.imag(root))) <= tolerance:
                    real_root = float(np.real(root))
                    if lower_t - 1e-10 <= real_root <= upper_t + 1e-10:
                        candidates.append(float(np.clip(real_root, lower_t, upper_t)))
                        stationary_count += 1
    # Deduplicate close numerical roots before evaluating the direct objective.
    candidates = sorted(candidates)
    unique_candidates = []
    for candidate in candidates:
        if not unique_candidates or abs(candidate - unique_candidates[-1]) > 1e-10:
            unique_candidates.append(candidate)
    return unique_candidates, stationary_count


def _estimate_shift_polynomial(observed_shift_mm: np.ndarray, coefficients: np.ndarray,
                               bounds_deg: tuple[float, float] = (-20.0, 20.0)) -> tuple[float, float, int]:
    weight = np.eye(4) - np.ones((4, 4)) / 5.0
    candidates_t, stationary_count = _polynomial_candidates(observed_shift_mm, coefficients, bounds_deg)
    costs = [
        _polynomial_objective(observed_shift_mm, coefficients, t, weight)
        for t in candidates_t
    ]
    best_index = int(np.argmin(costs))
    theta_deg = candidates_t[best_index] * ANGLE_SCALE_DEG
    weighted_rms_mm = float(np.sqrt(max(costs[best_index], 0.0) / 8.0))
    return float(theta_deg), weighted_rms_mm, stationary_count


def estimate_rotation_polynomial(
    measured_absolute_xy_mm: np.ndarray,
    coefficients_scaled_by_point_xy: np.ndarray,
    baseline_corner_coordinates_mm: np.ndarray,
    bounds_deg: tuple[float, float] = (-20.0, 20.0),
) -> tuple[float, float]:
    """Estimate eye rotation from five absolute points using the quadratic model.

    Points are ordered center, top-left, top-right, bottom-left, bottom-right.
    The polynomial predicts corner shifts from their zero-degree positions.
    Returns the minimizing angle in degrees and weighted RMS residual in mm.
    """
    measured = np.asarray(measured_absolute_xy_mm, dtype=float)
    baseline = np.asarray(baseline_corner_coordinates_mm, dtype=float)
    if measured.shape != (5, 2) or not np.all(np.isfinite(measured)):
        raise ValueError("Supply five finite absolute (x,y) image coordinates in mm.")
    if baseline.shape != (4, 2) or not np.all(np.isfinite(baseline)):
        raise ValueError("Zero-degree corner coordinates must have shape (4, 2).")
    if np.asarray(coefficients_scaled_by_point_xy).shape != (3, 5, 2):
        raise ValueError("Polynomial coefficients must have shape (3, 5, 2).")
    observed_shift = measured[1:] - measured[0] - baseline
    estimate, residual, _ = _estimate_shift_polynomial(
        observed_shift, coefficients_scaled_by_point_xy, bounds_deg
    )
    return estimate, residual


def analyze(angles: np.ndarray, xy: np.ndarray, distortion: np.ndarray,
           model: dict, bounds_deg: tuple[float, float] = (-20.0, 20.0)):
    baseline_idx = int(np.flatnonzero(angles == 0.0)[0])
    base_xy, base_d = xy[baseline_idx], distortion[baseline_idx]
    delta_xy = xy - base_xy[None, :, :]
    delta_d = distortion - base_d[None, :, :]
    weight = np.eye(4) - np.ones((4, 4)) / 5.0
    coefficients = model["coefficients_scaled_by_point_xy"]
    baseline_corners = model["baseline_corner_coordinates_mm"]

    # Pairwise separation after removing common center motion. Examine pairs at
    # least one degree apart so adjacent samples do not dominate the ambiguity check.
    closest = (np.inf, None, None)
    for i in range(angles.size):
        for j in range(i + 1, angles.size):
            if angles[j] - angles[i] < 1.0 - 1e-9:
                continue
            rms = np.sqrt(weighted_distance_sq(xy[i], xy[j], weight) / 8.0)
            if rms < closest[0]:
                closest = (float(rms), float(angles[i]), float(angles[j]))

    # Local Fisher information from the forward polynomial derivative. The
    # coefficients are in powers [t^2, t, 1], where t=theta/20 degrees.
    scaled_angles = angles / ANGLE_SCALE_DEG
    derivatives = (
        2.0 * scaled_angles[:, None, None] * coefficients[0, 1:, :]
        + coefficients[1, 1:, :]
    ) / ANGLE_SCALE_DEG
    dx_dtheta = derivatives[:, :, 0]
    dy_dtheta = derivatives[:, :, 1]
    fisher_unit = np.array([
        dx_dtheta[i] @ weight @ dx_dtheta[i] + dy_dtheta[i] @ weight @ dy_dtheta[i]
        for i in range(angles.size)
    ])
    if np.any(fisher_unit <= 0):
        raise ValueError("Some angles have zero local polynomial sensitivity.")
    fisher_sigma = {sigma: sigma / np.sqrt(fisher_unit) for sigma in SIGMA_MM}

    # Invert noiseless CODE V points with the fitted polynomial. The residual
    # bias reflects the model approximation error, not measurement noise.
    polynomial_noiseless_estimates = np.empty(angles.size, dtype=float)
    polynomial_noiseless_residuals = np.empty(angles.size, dtype=float)
    stationary_candidate_counts = np.empty(angles.size, dtype=int)
    for i, angle in enumerate(angles):
        observed_shift = xy[i] - baseline_corners
        estimate, residual, stationary_count = _estimate_shift_polynomial(
            observed_shift, coefficients, bounds_deg
        )
        polynomial_noiseless_estimates[i] = estimate
        polynomial_noiseless_residuals[i] = residual
        stationary_candidate_counts[i] = stationary_count
    polynomial_noiseless_bias = polynomial_noiseless_estimates - angles
    if not np.all(np.isfinite(polynomial_noiseless_estimates)):
        raise RuntimeError("Polynomial inversion returned a non-finite noiseless estimate.")

    # Seeded idealized Monte Carlo: generate independent absolute point noise,
    # including the center, then apply the quadratic forward-model inverse.
    rng = np.random.default_rng(SEED)
    truth_indices = np.unique(np.round(np.linspace(0, angles.size - 1, 101)).astype(int))
    truth_angles = angles[truth_indices]
    mc = {}
    for sigma in SIGMA_MM:
        errors = []
        estimates = []
        per_truth_median = []
        per_truth_p95 = []
        for idx in truth_indices:
            truth_abs = np.vstack((np.zeros((1, 2)), xy[idx]))
            noise = rng.normal(0.0, sigma, size=(100, 5, 2))
            measured = truth_abs[None, :, :] + noise
            observed_shifts = measured[:, 1:, :] - measured[:, 0:1, :] - baseline_corners
            estimated = np.empty(100, dtype=float)
            for trial in range(100):
                estimated[trial], _, _ = _estimate_shift_polynomial(
                    observed_shifts[trial], coefficients, bounds_deg
                )
            trial_errors = np.abs(estimated - angles[idx])
            estimates.extend(estimated.tolist())
            errors.extend(trial_errors.tolist())
            per_truth_median.append(float(np.median(trial_errors)))
            per_truth_p95.append(float(np.percentile(trial_errors, 95)))
        errors = np.asarray(errors)
        estimates = np.asarray(estimates)
        truth_for_trial = np.repeat(truth_angles, 100)
        worst_trial = int(np.argmax(errors))
        mc[sigma] = {
            "median_abs_error_deg": float(np.median(errors)),
            "p95_abs_error_deg": float(np.percentile(errors, 95)),
            "rmse_deg": float(np.sqrt(np.mean((estimates - truth_for_trial) ** 2))),
            "max_abs_error_deg": float(np.max(errors)),
            "fraction_over_1deg": float(np.mean(errors > 1.0)),
            "fraction_over_5deg": float(np.mean(errors > 5.0)),
            "clamp_fraction": float(np.mean(np.isclose(np.abs(estimates), 20.0, atol=1e-10))),
            "lower_clamp_fraction": float(np.mean(np.isclose(estimates, -20.0, atol=1e-10))),
            "upper_clamp_fraction": float(np.mean(np.isclose(estimates, 20.0, atol=1e-10))),
            "trial_count": int(errors.size),
            "truth_angles_deg": truth_angles,
            "per_truth_median_abs_error_deg": np.asarray(per_truth_median),
            "per_truth_p95_abs_error_deg": np.asarray(per_truth_p95),
            "worst_error_truth_deg": float(truth_for_trial[worst_trial]),
            "worst_error_estimate_deg": float(estimates[worst_trial]),
        }

    model_fit_metrics = model["fit_metrics"]
    return {
        "delta_coordinates_mm": delta_xy,
        "delta_distortion_percentage_points": delta_d,
        "dx_dtheta_mm_per_deg": dx_dtheta,
        "dy_dtheta_mm_per_deg": dy_dtheta,
        "fisher_information_deg2_per_mm2": fisher_unit,
        "fisher_sigma_deg": fisher_sigma,
        "closest_nonlocal_signature_rms_mm": closest[0],
        "closest_nonlocal_angle_pair_deg": closest[1:],
        "polynomial_noiseless_estimates_deg": polynomial_noiseless_estimates,
        "polynomial_noiseless_bias_deg": polynomial_noiseless_bias,
        "polynomial_noiseless_weighted_rms_residual_mm": polynomial_noiseless_residuals,
        "polynomial_noiseless_median_abs_bias_deg": float(np.median(np.abs(polynomial_noiseless_bias))),
        "polynomial_noiseless_p95_abs_bias_deg": float(np.percentile(np.abs(polynomial_noiseless_bias), 95)),
        "polynomial_noiseless_max_abs_bias_deg": float(np.max(np.abs(polynomial_noiseless_bias))),
        "polynomial_noiseless_worst_truth_angle_deg": float(
            angles[np.argmax(np.abs(polynomial_noiseless_bias))]
        ),
        "polynomial_noiseless_max_stationary_roots": int(np.max(stationary_candidate_counts)),
        "polynomial_noiseless_max_weighted_rms_residual_mm": float(
            np.max(polynomial_noiseless_residuals)
        ),
        "forward_model_fit_metrics": model_fit_metrics,
        "model_metadata": {key: model[key] for key in (
            "model_pickle", "scaled_basis", "physical_basis", "point_order", "coordinate_order",
            "max_scaled_prediction_check_error_mm", "max_basis_conversion_check_error_mm",
        )},
        "rotation_bounds_deg": tuple(map(float, bounds_deg)),
        "monte_carlo": mc,
        "corner_names": CORNER_NAMES,
        "noise_sigmas_mm": SIGMA_MM,
    }


def plot_dependencies(angles: np.ndarray, results: dict, path: Path) -> None:
    coord = results["delta_coordinates_mm"]
    # The stored corner coordinates are already center-referenced at each
    # rotation. Add the measured center as an explicit, zero-valued fifth point.
    values_x = np.column_stack((np.zeros(angles.size), coord[:, :, 0]))
    values_y = np.column_stack((np.zeros(angles.size), coord[:, :, 1]))
    point_names = ("center", *CORNER_NAMES)
    colors = plt.get_cmap("tab10").colors[:5]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    specs = ((axes[0], values_x, "Horizontal shift ΔX", "Shift from 0° (mm)"),
             (axes[1], values_y, "Vertical shift ΔY", "Shift from 0° (mm)"))
    for ax, values, title, ylabel in specs:
        for pi, name in enumerate(point_names):
            ax.plot(angles, values[:, pi], color=colors[pi], linewidth=1.25,
                    linestyle=":" if pi == 0 else "-", label=name)
        ax.axhline(0, color="0.4", linewidth=0.7)
        ax.axvline(0, color="0.4", linestyle=":", linewidth=0.7)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, color="0.9", linewidth=0.6)
        ax.legend(frameon=False, ncol=5, fontsize=8)
    axes[1].set_xlabel("Ground-truth eye rotation (degrees)")
    fig.suptitle("Five-point fingerprint relative to 0 degrees")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_precision(angles: np.ndarray, results: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    sigma = SIGMA_MM[0]
    values = results["fisher_sigma_deg"][sigma] * DEG_TO_ARCMIN
    axes[0].plot(angles, values, label=f"sigma={sigma*1000:g} um per coordinate")
    axes[0].set_title("Local Fisher estimate")
    axes[0].set_xlabel("Ground-truth eye rotation (degrees)")
    axes[0].set_ylabel("Local 1-sigma angle uncertainty (arcmin)")
    axes[0].grid(True, color="0.9", linewidth=0.6)
    axes[0].legend(frameon=False)
    mc = results["monte_carlo"][sigma]
    axes[1].plot(mc["truth_angles_deg"], mc["per_truth_median_abs_error_deg"] * DEG_TO_ARCMIN,
                 label="Per-angle median", linewidth=1.1)
    axes[1].plot(mc["truth_angles_deg"], mc["per_truth_p95_abs_error_deg"] * DEG_TO_ARCMIN,
                 label="Per-angle 95th percentile", linewidth=1.1)
    axes[1].set_title("Error varies by true angle")
    axes[1].set_xlabel("Ground-truth eye rotation (degrees)")
    axes[1].set_ylabel("Absolute angle error (arcmin)")
    axes[1].grid(True, color="0.9", linewidth=0.6)
    axes[1].legend(frameon=False)
    axes[2].bar([0, 1], [mc["median_abs_error_deg"] * DEG_TO_ARCMIN,
                         mc["p95_abs_error_deg"] * DEG_TO_ARCMIN],
                color=["#1769aa", "#e07a24"])
    axes[2].set_xticks([0, 1], ["Median", "95th percentile"])
    axes[2].set_title("Pooled 10,100-trial errors")
    axes[2].set_ylabel("Absolute angle error (arcmin)")
    axes[2].grid(True, axis="y", color="0.9", linewidth=0.6)
    fig.suptitle("Idealized precision from the quadratic five-point estimator")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(path: Path, angles: np.ndarray, results: dict) -> None:
    pair = results["closest_nonlocal_angle_pair_deg"]
    coordinates = results["delta_coordinates_mm"]
    distortion = results["delta_distortion_percentage_points"]
    mc = results["monte_carlo"][SIGMA_MM[0]]
    fisher = results["fisher_sigma_deg"][SIGMA_MM[0]] * DEG_TO_ARCMIN
    fit = results["forward_model_fit_metrics"]
    bias = results["polynomial_noiseless_bias_deg"]

    def peak(values: np.ndarray) -> tuple[float, float, str]:
        idx = np.unravel_index(np.argmax(np.abs(values)), values.shape)
        return float(values[idx]), float(angles[idx[0]]), CORNER_NAMES[idx[1]]

    peak_x = peak(coordinates[:, :, 0])
    peak_y = peak(coordinates[:, :, 1])
    peak_radial = peak(distortion[:, :, 0])
    peak_tangential = peak(distortion[:, :, 1])
    per_angle_p95 = mc["per_truth_p95_abs_error_deg"] * DEG_TO_ARCMIN
    worst_p95_index = int(np.argmax(per_angle_p95))
    fisher_best_idx = int(np.argmin(fisher))
    fisher_worst_idx = int(np.argmax(fisher))
    bias_abs = np.abs(bias)
    bias_worst_idx = int(np.argmax(bias_abs))
    metadata = results["model_metadata"]
    lines = [
        "# Five-point eye-rotation analysis with a quadratic forward model", "",
        "**All estimation-error and precision values are in arcminutes (arcmin), with 1 degree = 60 arcmin. "
        "Truth and estimated rotation values remain in degrees.**",
        "## Data and model", "",
        f"The analysis uses {angles.size} CODE V ground-truth rotations from {angles[0]:g} to "
        f"{angles[-1]:g} degrees in {np.median(np.diff(angles)):.6g}-degree steps.",
        "The measured points are ordered center, top-left, top-right, bottom-left, bottom-right. "
        "The sweep coordinates already subtract the optical center at each rotation, so the center "
        "anchors the absolute measurements while the four relative corner X/Y values carry the angle pattern.",
        f"The model comes from `{metadata['model_pickle']}`. Its scaled basis is "
        "`(theta/20)^2, theta/20, 1`, with coefficient order `(power, point, x/y)`, powers "
        "`t^2, t, constant`, and points center/TL/TR/BL/BR. The artifact also stores physical "
        "coefficients in `[theta^2, theta, 1]` order. Both bases and point order were checked against "
        f"the artifact predictions (maximum differences {metadata['max_scaled_prediction_check_error_mm']:.3g} "
        f"mm scaled and {metadata['max_basis_conversion_check_error_mm']:.3g} mm physical).",
        f"Full-sweep forward coordinate residual: corner RMSE {fit['corner_coordinate_rmse_mm']*1000:.6g} um, "
        f"MAE {fit['corner_coordinate_mae_mm']*1000:.6g} um, and maximum absolute residual "
        f"{fit['corner_coordinate_max_abs_mm']*1000:.6g} um. These are forward coordinate fit errors, "
        "not rotation errors.",
        "", "## Inverse", "",
        "For a measured five-point set, subtract the measured center from each corner and subtract the "
        "zero-degree corner positions. Fit the resulting four-by-two shift to the quadratic forward model.",
        "With independent noise on the five absolute point coordinates, center subtraction gives each axis "
        "corner covariance `sigma^2 (I + 11^T)`. The weighted least-squares precision matrix is "
        "`W = I - 11^T/5` for each axis.",
        "The objective is quartic in normalized angle `t=theta/20`, so its derivative is cubic. The inverse "
        "evaluates every real stationary root within the angle bounds and both endpoints, then chooses the "
        "candidate with the smallest weighted residual. This is a global minimization of the fitted quadratic "
        "objective over the bounded interval, independent of angle-template spacing.",
        "", "## Noiseless inversion", "",
        f"Inverting all {angles.size} exact CODE V corner patterns gives median absolute angle bias "
        f"{np.median(bias_abs)*DEG_TO_ARCMIN:.6g} arcmin, P95 "
        f"{np.percentile(bias_abs, 95)*DEG_TO_ARCMIN:.6g} arcmin, and maximum "
        f"{bias_abs[bias_worst_idx]*DEG_TO_ARCMIN:.6g} arcmin at truth {angles[bias_worst_idx]:+g} degrees "
        f"(estimate {results['polynomial_noiseless_estimates_deg'][bias_worst_idx]:+g} degrees). "
        "This bias comes from the quadratic approximation to the CODE V response. It is distinct from the "
        "sub-micrometer forward coordinate residual above.",
        f"Largest absolute coordinate changes from 0 degrees are dX={peak_x[0]:.6g} mm at "
        f"{peak_x[1]:+g} degrees ({peak_x[2]}) and dY={peak_y[0]:.6g} mm at {peak_y[1]:+g} degrees "
        f"({peak_y[2]}). Largest radial/tangential distortion changes are {peak_radial[0]:.6g} and "
        f"{peak_tangential[0]:.6g} percentage points. These distortion values are diagnostic only; the "
        "rotation inverse uses image coordinates.",
        f"The closest sampled ground-truth corner patterns at least 1 degree apart are {pair[0]:+g} and "
        f"{pair[1]:+g} degrees, with center-corrected weighted RMS separation "
        f"{results['closest_nonlocal_signature_rms_mm']:.6g} mm per coordinate feature.",
        "", "## Conditional 1 um coordinate-noise precision", "",
        "Assume independent Gaussian noise with standard deviation 0.001 mm (1 um) on each X and Y "
        "coordinate of all five absolute points, including the center. The seeded Monte Carlo uses 101 true "
        "angles, 100 trials per angle, and exact CODE V coordinates as the noise-free means. Thus results "
        "include both coordinate noise and quadratic approximation bias. Error thresholds of 60 and 300 arcmin "
        "correspond to 1 and 5 degrees, respectively.",
        "", "| Statistic | Result |", "|---|---:|",
        f"| Median absolute angle error | {mc['median_abs_error_deg']*DEG_TO_ARCMIN:.6g} arcmin |",
        f"| Pooled 95th-percentile absolute angle error | {mc['p95_abs_error_deg']*DEG_TO_ARCMIN:.6g} arcmin |",
        f"| RMSE | {mc['rmse_deg']*DEG_TO_ARCMIN:.6g} arcmin |",
        f"| Maximum absolute error | {mc['max_abs_error_deg']*DEG_TO_ARCMIN:.6g} arcmin |",
        f"| Trials above 60 arcmin (1 degree) | {100*mc['fraction_over_1deg']:.4g}% |",
        f"| Trials above 300 arcmin (5 degrees) | {100*mc['fraction_over_5deg']:.4g}% |",
        f"| Endpoint clamp rate (-20 / +20 degrees) | {100*mc['lower_clamp_fraction']:.4g}% / {100*mc['upper_clamp_fraction']:.4g}% |",
        f"| Worst per-angle P95 | {np.max(per_angle_p95):.6g} arcmin at {mc['truth_angles_deg'][worst_p95_index]:+g} degrees |",
        f"| Local Fisher 1-sigma range | {np.min(fisher):.6g} to {np.max(fisher):.6g} arcmin |",
        f"| Best / worst local Fisher angle | {angles[fisher_best_idx]:+g} / {angles[fisher_worst_idx]:+g} degrees |",
        "", "The pooled and per-angle Monte Carlo percentiles have finite-sample uncertainty. The local Fisher "
        "estimate describes small errors near the true angle and does not capture wrong-branch outcomes. "
        "The 0.1-degree ground-truth spacing is sweep sampling, not demonstrated measurement accuracy.",
        "This simulation assumes known point identities and an exact calibration apart from the fitted "
        "quadratic approximation. It omits detector calibration, alignment, localization bias, and other "
        "systematic errors; it is not measured hardware precision.",
        "", "## Reusable estimator", "",
        "`Script/analyze_eye_rotation.py` exposes `estimate_rotation_polynomial(measured_absolute_xy_mm, "
        "coefficients_scaled_by_point_xy, baseline_corner_coordinates_mm, bounds_deg=(-20, 20))`. "
        "Supply a `(5, 2)` array ordered center, TL, TR, BL, BR. It returns the fitted angle in degrees and "
        "weighted RMS residual in mm.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model-pickle", type=Path, default=DEFAULT_MODEL_PICKLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    angles, xy, distortion = load_sweep(args.input)
    model = load_polynomial_model(args.model_pickle, angles, xy)
    results = analyze(angles, xy, distortion, model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "eye_rotation_observability.pkl").open("wb") as stream:
        pickle.dump({
            "method": "bounded global weighted least-squares inverse of the quadratic forward model",
            "angles_deg": angles,
            "corner_coordinates_mm": xy,
            "corner_distortion_pct": distortion,
            "polynomial_model": {
                "coefficients_scaled_by_point_xy": model["coefficients_scaled_by_point_xy"],
                "baseline_corner_coordinates_mm": model["baseline_corner_coordinates_mm"],
                "metadata": results["model_metadata"],
            },
            "analysis": results,
        },
                    stream, protocol=pickle.HIGHEST_PROTOCOL)
    plot_dependencies(angles, results, args.output_dir / "dependencies_vs_rotation.png")
    plot_precision(angles, results, args.output_dir / "precision_estimates.png")
    write_report(args.output_dir / "report.md", angles, results)
    print(f"Saved outputs to {args.output_dir.resolve()}")
    print(f"Loaded verified polynomial coefficients from {model['model_pickle']}")
    print(f"Forward fit corner-coordinate RMSE={model['fit_metrics']['corner_coordinate_rmse_mm']*1000:.6g} um; "
          f"physical/scaled prediction check={model['max_basis_conversion_check_error_mm']:.3g} mm")
    pair = results["closest_nonlocal_angle_pair_deg"]
    print(f"Closest >=1 degree pair: {pair[0]:+g}, {pair[1]:+g} deg; "
          f"weighted RMS separation={results['closest_nonlocal_signature_rms_mm']:.6g} mm")
    print("Noiseless polynomial inversion median/P95/max absolute bias="
          f"{results['polynomial_noiseless_median_abs_bias_deg']:.6g}/"
          f"{results['polynomial_noiseless_p95_abs_bias_deg']:.6g}/"
          f"{results['polynomial_noiseless_max_abs_bias_deg']:.6g} deg; "
          f"max at {results['polynomial_noiseless_worst_truth_angle_deg']:+g} deg")
    for sigma in SIGMA_MM:
        fisher = results["fisher_sigma_deg"][sigma]
        mc = results["monte_carlo"][sigma]
        print(f"sigma={sigma*1000:g} um/coordinate: Fisher 1sigma={np.min(fisher):.6g}..{np.max(fisher):.6g} deg; "
              f"MC median/P95/max={mc['median_abs_error_deg']:.6g}/"
              f"{mc['p95_abs_error_deg']:.6g}/{mc['max_abs_error_deg']:.6g} deg; "
              f"RMSE={mc['rmse_deg']:.6g} deg; endpoint clamp -/+="
              f"{100*mc['lower_clamp_fraction']:.3g}%/{100*mc['upper_clamp_fraction']:.3g}%")
        print(f"  >1 deg={100*mc['fraction_over_1deg']:.3g}%; >5 deg="
              f"{100*mc['fraction_over_5deg']:.3g}%; worst error truth/estimate="
              f"{mc['worst_error_truth_deg']:+g}/{mc['worst_error_estimate_deg']:+g} deg; "
              f"worst per-angle P95={np.max(mc['per_truth_p95_abs_error_deg']):.6g} deg at "
              f"{mc['truth_angles_deg'][np.argmax(mc['per_truth_p95_abs_error_deg'])]:+g} deg")


if __name__ == "__main__":
    main()
