"""Study whether five measured grid points identify eye rotation.

The script reads the completed CODE V sweep from its pickle. It never imports
or starts CODE V. The reusable ``estimate_rotation`` function accepts five
absolute measured image coordinates in mm ordered as center, top-left,
top-right, bottom-left, bottom-right.
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
CORNER_NAMES = ("top-left", "top-right", "bottom-left", "bottom-right")
CORNER_FIELDS = ((-1.0, 1.0), (1.0, 1.0), (-1.0, -1.0), (1.0, -1.0))
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


def estimate_rotation(measured_absolute_xy_mm: np.ndarray, angles_deg: np.ndarray,
                      corner_templates_mm: np.ndarray) -> tuple[float, float]:
    """Fit five measured absolute points to the sweep; return angle and RMS mm.

    ``measured_absolute_xy_mm`` must have shape (5, 2), ordered center, top-left,
    top-right, bottom-left, bottom-right. The center is subtracted because the
    stored sweep itself is center-referenced.
    """
    measured = np.asarray(measured_absolute_xy_mm, dtype=float)
    if measured.shape != (5, 2) or not np.all(np.isfinite(measured)):
        raise ValueError("Supply five finite absolute (x,y) image coordinates in mm.")
    angles = np.asarray(angles_deg, dtype=float)
    templates = np.asarray(corner_templates_mm, dtype=float)
    if templates.shape != (angles.size, 4, 2) or angles.size < 2:
        raise ValueError("Templates must have shape (number of angles, 4, 2).")
    observed_relative = measured[1:] - measured[0]
    return _estimate_relative(observed_relative, angles, templates)


def _estimate_relative(observed: np.ndarray, angles: np.ndarray,
                       templates: np.ndarray) -> tuple[float, float]:
    estimate, residual = _estimate_relative_batch(np.asarray(observed)[None, ...], angles, templates)
    return float(estimate[0]), float(residual[0])


def _estimate_relative_batch(observed: np.ndarray, angles: np.ndarray,
                             templates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized weighted projection of one or more measurements on segments."""
    # Center-subtracted corner noise has covariance sigma^2(I + 11^T) per
    # coordinate. Its inverse, with sigma factored out, is I - 11^T/5.
    weight = np.eye(4) - np.ones((4, 4)) / 5.0
    observed = np.asarray(observed, dtype=float)
    p0 = templates[:-1]
    slope = templates[1:] - p0
    delta = observed[:, None, :, :] - p0[None, :, :, :]
    weighted_delta = np.einsum("cd,bsda->bsca", weight, delta, optimize=True)
    numerator = np.einsum("sci,bsci->bs", slope, weighted_delta, optimize=True)
    denominator = np.einsum("sci,cd,sdi->s", slope, weight, slope, optimize=True)
    fraction = np.divide(numerator, denominator[None, :], out=np.zeros_like(numerator),
                         where=denominator[None, :] > np.finfo(float).tiny)
    fraction = np.clip(fraction, 0.0, 1.0)
    fitted = p0[None, :, :, :] + fraction[:, :, None, None] * slope[None, :, :, :]
    residual = observed[:, None, :, :] - fitted
    weighted_residual = np.einsum("cd,bsda->bsca", weight, residual, optimize=True)
    cost = np.einsum("bsci,bsci->bs", residual, weighted_residual, optimize=True)
    best = np.argmin(cost, axis=1)
    rows = np.arange(observed.shape[0])
    fitted_fraction = fraction[rows, best]
    estimated_angles = angles[best] + fitted_fraction * (angles[best + 1] - angles[best])
    rms_residual = np.sqrt(cost[rows, best] / 8.0)
    return estimated_angles, rms_residual


def analyze(angles: np.ndarray, xy: np.ndarray, distortion: np.ndarray):
    baseline_idx = int(np.flatnonzero(angles == 0.0)[0])
    base_xy, base_d = xy[baseline_idx], distortion[baseline_idx]
    delta_xy = xy - base_xy[None, :, :]
    delta_d = distortion - base_d[None, :, :]
    weight = np.eye(4) - np.ones((4, 4)) / 5.0

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

    # Local Fisher information for five independent absolute point measurements
    # with the same isotropic coordinate noise sigma.
    dx_dtheta = np.gradient(xy[:, :, 0], angles, axis=0, edge_order=2)
    dy_dtheta = np.gradient(xy[:, :, 1], angles, axis=0, edge_order=2)
    fisher_unit = np.array([
        dx_dtheta[i] @ weight @ dx_dtheta[i] + dy_dtheta[i] @ weight @ dy_dtheta[i]
        for i in range(angles.size)
    ])
    if np.any(fisher_unit <= 0):
        raise ValueError("Some angles have zero local coordinate sensitivity.")
    fisher_sigma = {sigma: sigma / np.sqrt(fisher_unit) for sigma in SIGMA_MM}

    # Leave one angle out, then estimate it from the remaining piecewise-linear
    # templates. This quantifies interpolation/model discretization error only.
    loo_errors = []
    for i in range(1, angles.size - 1):
        keep = np.arange(angles.size) != i
        estimate, _ = _estimate_relative(xy[i], angles[keep], xy[keep])
        loo_errors.append(abs(estimate - angles[i]))
    loo_errors = np.asarray(loo_errors)

    # Seeded idealized Monte Carlo: generate independent absolute point noise,
    # including the center, then invoke the same center-subtracted inverse.
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
            observed = measured[:, 1:, :] - measured[:, 0:1, :]
            estimated, _ = _estimate_relative_batch(observed, angles, xy)
            trial_errors = np.abs(estimated - angles[idx])
            estimates.extend(estimated)
            errors.extend(trial_errors)
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

    return {
        "delta_coordinates_mm": delta_xy,
        "delta_distortion_percentage_points": delta_d,
        "dx_dtheta_mm_per_deg": dx_dtheta,
        "dy_dtheta_mm_per_deg": dy_dtheta,
        "fisher_information_per_mm2": fisher_unit,
        "fisher_sigma_deg": fisher_sigma,
        "closest_nonlocal_signature_rms_mm": closest[0],
        "closest_nonlocal_angle_pair_deg": closest[1:],
        "loo_abs_error_deg": loo_errors,
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
    values = results["fisher_sigma_deg"][sigma]
    axes[0].plot(angles, values * 1000, label=f"σ={sigma*1000:g} µm per coordinate")
    axes[0].set_title("Local Fisher estimate")
    axes[0].set_xlabel("Ground-truth eye rotation (degrees)")
    axes[0].set_ylabel("Local 1σ angle uncertainty (millidegrees)")
    axes[0].grid(True, color="0.9", linewidth=0.6)
    axes[0].legend(frameon=False)
    mc = results["monte_carlo"][sigma]
    axes[1].plot(mc["truth_angles_deg"], mc["per_truth_median_abs_error_deg"],
                 label="Per-angle median", linewidth=1.1)
    axes[1].plot(mc["truth_angles_deg"], mc["per_truth_p95_abs_error_deg"],
                 label="Per-angle 95th percentile", linewidth=1.1)
    axes[1].set_title("Error varies by true angle")
    axes[1].set_xlabel("Ground-truth eye rotation (degrees)")
    axes[1].set_ylabel("Absolute angle error (degrees)")
    axes[1].grid(True, color="0.9", linewidth=0.6)
    axes[1].legend(frameon=False)
    axes[2].bar([0, 1], [mc["median_abs_error_deg"], mc["p95_abs_error_deg"]],
                color=["#1769aa", "#e07a24"])
    axes[2].set_xticks([0, 1], ["Median", "95th percentile"])
    axes[2].set_title("Pooled 10,100-trial errors")
    axes[2].set_ylabel("Absolute angle error (degrees)")
    axes[2].grid(True, axis="y", color="0.9", linewidth=0.6)
    fig.suptitle("Idealized precision from five measured points")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(path: Path, angles: np.ndarray, results: dict) -> None:
    pair = results["closest_nonlocal_angle_pair_deg"]
    coord = results["delta_coordinates_mm"]
    delta_dist = results["delta_distortion_percentage_points"]
    def peak(values):
        index = np.unravel_index(np.argmax(np.abs(values)), values.shape)
        return float(values[index]), float(angles[index[0]]), CORNER_NAMES[index[1]]

    peak_x = peak(coord[:, :, 0])
    peak_y = peak(coord[:, :, 1])
    peak_radial = peak(delta_dist[:, :, 0])
    peak_tangential = peak(delta_dist[:, :, 1])
    fisher = results["fisher_sigma_deg"][SIGMA_MM[0]]
    best_fisher_angle = float(angles[np.argmin(fisher)])
    worst_fisher_angle = float(angles[np.argmax(fisher)])
    mc_one = results["monte_carlo"][SIGMA_MM[0]]
    per_angle_p95 = mc_one["per_truth_p95_abs_error_deg"]
    per_angle_median = mc_one["per_truth_median_abs_error_deg"]
    p95_truth_angles = mc_one["truth_angles_deg"]
    worst_p95_index = int(np.argmax(per_angle_p95))
    lines = [
        "# Five-point eye-rotation observability", "",
        "## Data and method", "",
        f"Read {angles.size} ground-truth rotations from {angles[0]:g}° to {angles[-1]:g}° "
        f"in {np.median(np.diff(angles)):.6g}° steps, using four corner fields plus the center.",
        "The collector already subtracts the center ray at each rotation. The center is therefore "
        "an anchor for measured coordinates; its absolute translation carries no rotation information "
        "in this saved dataset. The inverse uses the four corner coordinates relative to the measured center.",
        "The four corners are top-left, top-right, bottom-left, and bottom-right at relative fields "
        "(-1,+1), (+1,+1), (-1,-1), and (+1,-1). Their image X/Y coordinates form an eight-value fingerprint.",
        "Radial and tangential corner distortion are also plotted as separate diagnostics. The proposed "
        "coordinate inverse does not treat those distortion percentages as directly measured inputs.",
        "", "## Dependence and ambiguity", "",
        f"Closest pair of fingerprints separated by at least 1°: RC {pair[0]:+g}° and {pair[1]:+g}°, "
        f"center-corrected weighted RMS separation {results['closest_nonlocal_signature_rms_mm']:.9g} mm "
        "per coordinate feature. This checks sampled global ambiguity; finite measurement noise can still cause errors.",
        f"Largest absolute corner-coordinate changes from 0° are ΔX={peak_x[0]:.6g} mm "
        f"at RC {peak_x[1]:+g}° ({peak_x[2]}) and ΔY={peak_y[0]:.6g} mm at RC {peak_y[1]:+g}° "
        f"({peak_y[2]}). Largest absolute distortion changes are radial {peak_radial[0]:.6g} pp "
        f"at RC {peak_radial[1]:+g}° ({peak_radial[2]}) and tangential "
        f"{peak_tangential[0]:.6g} pp at RC {peak_tangential[1]:+g}° ({peak_tangential[2]}).",
        f"Leave-one-out piecewise-linear template inversion across interior sweep samples gives median "
        f"absolute error {np.median(results['loo_abs_error_deg']):.9g}° and maximum "
        f"{np.max(results['loo_abs_error_deg']):.9g}°. This is interpolation/model discretization error "
        "for noiseless synthetic templates, not sensor precision.",
        "", "## Conditional precision", "",
        "Assume independent, isotropic Gaussian X/Y errors on all five absolute measured points, with "
        "σ = 0.001 mm (1 µm standard deviation) for each individual X and Y coordinate, including the center. "
        "After subtracting the noisy center, each coordinate's "
        "four-corner covariance is σ²(I + 11ᵀ); its inverse apart from σ² is I - 11ᵀ/5. The local "
        "Fisher estimate uses the numerical derivative of the sampled corner fingerprint.",
        "The Monte Carlo study uses seed 20261008, 101 evenly spaced ground-truth angles, and 100 trials "
        "per angle and noise level. It simulates errors on all five points and fits the piecewise-linear "
        "sweep by weighted least squares. The empirical percentiles have Monte Carlo sampling uncertainty "
        "because this is a finite 10,100-trial sample. This assumes the optical model/templates are exact "
        "and the five field identities are known; it omits detector calibration, alignment, and other systematic errors.",
        "All 401 sampled templates are distinct. This does not prove the underlying continuous physical "
        "mapping is injective between samples.",
        f"Across all 10,100 trials, {mc_one['fraction_over_1deg']*100:.3g}% exceed 1° absolute error "
        f"and {mc_one['fraction_over_5deg']*100:.3g}% exceed 5°. The largest observed error is "
        f"{mc_one['max_abs_error_deg']:.6g}° (truth {mc_one['worst_error_truth_deg']:+g}°, "
        f"estimate {mc_one['worst_error_estimate_deg']:+g}°). Per-angle median errors range "
        f"{np.min(per_angle_median):.6g}–{np.max(per_angle_median):.6g}°; per-angle 95th percentiles "
        f"range {np.min(per_angle_p95):.6g}–{np.max(per_angle_p95):.6g}°, with the worst at "
        f"truth RC {p95_truth_angles[worst_p95_index]:+g}°. The pooled Monte Carlo percentile is not "
        "a guaranteed error bound at every eye rotation.", "",
        "| Assumed per-coordinate point noise | Fisher 1σ range over sweep | Monte Carlo median | "
        "Monte Carlo 95th percentile | Monte Carlo RMSE | Max error | Endpoint clamp rate (−20° / +20°) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sigma in SIGMA_MM:
        fisher = results["fisher_sigma_deg"][sigma]
        mc = results["monte_carlo"][sigma]
        lines.append(
            f"| {sigma*1000:g} µm/coordinate | {np.min(fisher):.6g}–{np.max(fisher):.6g}° | "
            f"{mc['median_abs_error_deg']:.6g}° | {mc['p95_abs_error_deg']:.6g}° | "
            f"{mc['rmse_deg']:.6g}° | {mc['max_abs_error_deg']:.6g}° | "
            f"{mc['lower_clamp_fraction']*100:.3g}% / {mc['upper_clamp_fraction']*100:.3g}% |"
        )
    lines.append(
        f"At 1 µm per coordinate, the local Fisher 1σ estimate is best at RC {best_fisher_angle:+g}° "
        f"({np.min(fisher):.6g}°) and worst at RC {worst_fisher_angle:+g}° ({np.max(fisher):.6g}°)."
    )
    lines += [
        "", "## Limits", "",
        "These are conditional numerical results from the existing CODE V sweep, not measured hardware "
        "precision. Actual accuracy depends on point-localization noise, calibration, alignment, model "
        "error, and whether the five points can be detected without bias. The sweep spacing is 0.1°; "
        "interpolation permits a continuous estimate but does not establish 0.1° (or finer) experimental accuracy.",
        "The stored corner positions are center-relative by construction. The center adds no angle feature "
        "after recentering, but its noisy measurement is subtracted from all four corners and therefore "
        "increases their covariance; the inverse accounts for this with W = I − 11ᵀ/5. Any rotation "
        "information in absolute center displacement is absent and was not included.", "",
        "## Reusable inverse", "",
        "`Script/analyze_eye_rotation.py` exposes `estimate_rotation(measured_absolute_xy_mm, angles_deg, "
        "corner_templates_mm)`. Pass a (5, 2) array in center, top-left, top-right, bottom-left, bottom-right "
        "order. It returns the fitted rotation in degrees and center-corrected weighted RMS residual in mm.", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    angles, xy, distortion = load_sweep(args.input)
    results = analyze(angles, xy, distortion)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "eye_rotation_observability.pkl").open("wb") as stream:
        pickle.dump({"angles_deg": angles, "corner_coordinates_mm": xy,
                     "corner_distortion_pct": distortion, "analysis": results},
                    stream, protocol=pickle.HIGHEST_PROTOCOL)
    plot_dependencies(angles, results, args.output_dir / "dependencies_vs_rotation.png")
    plot_precision(angles, results, args.output_dir / "precision_estimates.png")
    write_report(args.output_dir / "report.md", angles, results)
    print(f"Saved outputs to {args.output_dir.resolve()}")
    pair = results["closest_nonlocal_angle_pair_deg"]
    print(f"Closest >=1 degree pair: {pair[0]:+g}, {pair[1]:+g} deg; "
          f"weighted RMS separation={results['closest_nonlocal_signature_rms_mm']:.6g} mm")
    print(f"LOO interpolation error median/max={np.median(results['loo_abs_error_deg']):.6g}/"
          f"{np.max(results['loo_abs_error_deg']):.6g} deg")
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
