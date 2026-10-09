"""Fit P1 radial k1 against absolute THI Z at zero eye rotation."""

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
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "p1_radial_distortion_fit"
REQUIRED_FIELDS = (
    "eye_rotation_deg", "z_mm", "paraxial_x_mm", "paraxial_y_mm",
    "real_x_mm", "real_y_mm", "field_x_relative", "field_y_relative",
    "radial_distortion_pct",
)
RESULT_DTYPE = np.dtype([
    ("z_mm", np.float64), ("k1_per_mm2", np.float64),
    ("coordinate_rmse_mm", np.float64), ("max_abs_residual_mm", np.float64),
    ("paraxial_edge_radius_mm", np.float64), ("normalized_k1", np.float64),
    ("mean_edge_radial_distortion_pct", np.float64),
    ("mean_corner_radial_distortion_pct", np.float64),
])
MODEL_DTYPE = np.dtype([
    ("model", "U32"), ("coefficient_0", np.float64), ("coefficient_1", np.float64),
    ("coefficient_2", np.float64), ("rmse_k1_per_mm2", np.float64),
    ("max_abs_error_k1_per_mm2", np.float64), ("rmse_fraction_of_k1_span", np.float64),
])
MAGNIFICATION_DTYPE = np.dtype([
    ("z_mm", np.float64), ("real_magnification", np.float64),
    ("paraxial_magnification", np.float64),
    ("coordinate_rmse_mm", np.float64), ("max_abs_residual_mm", np.float64),
    ("max_point_residual_um", np.float64), ("edge_ratio_mean", np.float64),
    ("corner_ratio_mean", np.float64), ("edge_corner_ratio_difference", np.float64),
    ("normalized_shape_coordinate_rmse_um", np.float64),
    ("normalized_shape_max_point_error_um", np.float64),
    ("normalized_k1_scaled", np.float64),
])
MAG_MODEL_DTYPE = np.dtype([
    ("model", "U16"), ("coefficient_0", np.float64), ("coefficient_1", np.float64),
    ("coefficient_2", np.float64), ("rmse", np.float64), ("max_abs_error", np.float64),
])


def load_zero_rotation_groups(input_path: Path) -> list[tuple[float, np.ndarray]]:
    with input_path.open("rb") as stream:
        data = pickle.load(stream)
    if not isinstance(data, np.ndarray) or data.ndim != 1 or data.dtype.names is None:
        raise ValueError("Input pickle must contain a 1D structured NumPy array.")
    missing = [name for name in REQUIRED_FIELDS if name not in data.dtype.names]
    if missing:
        raise ValueError(f"Input pickle is missing fields: {', '.join(missing)}")
    selected = data[np.isclose(data["eye_rotation_deg"], 0.0, rtol=0.0, atol=1e-8)]
    levels = np.unique(selected["z_mm"])
    if levels.size != 51:
        raise ValueError(f"Expected 51 zero-rotation Z groups, found {levels.size}.")
    groups = []
    for z in levels:
        rows = selected[np.isclose(selected["z_mm"], z, rtol=0.0, atol=1e-8)]
        if len(rows) != 9:
            raise ValueError(f"Z={z:g} mm has {len(rows)} points; expected 9.")
        coordinates = np.column_stack([rows[name] for name in REQUIRED_FIELDS[2:]])
        if not np.all(np.isfinite(coordinates)):
            raise ValueError(f"Z={z:g} mm contains nonfinite coordinates.")
        groups.append((float(z), rows))
    return groups


def fit_group(z_mm: float, rows: np.ndarray) -> tuple[float, float, float]:
    x, y = rows["paraxial_x_mm"], rows["paraxial_y_mm"]
    real_x, real_y = rows["real_x_mm"], rows["real_y_mm"]
    r2 = x**2 + y**2
    design = np.concatenate([x * r2, y * r2])
    displacement = np.concatenate([real_x - x, real_y - y])
    denominator = float(np.dot(design, design))
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError(f"Z={z_mm:g} mm has no fit leverage.")
    k1 = float(np.dot(design, displacement) / denominator)
    residual = displacement - k1 * design
    return k1, float(np.sqrt(np.mean(residual**2))), float(np.max(np.abs(residual)))


def scale_and_distortion(rows: np.ndarray, k1: float) -> tuple[float, float, float, float]:
    """Return axis edge radius, dimensionless k1*r² and sampled radial means."""
    axis_edge = (np.isclose(np.abs(rows["field_x_relative"]), 1.0, atol=1e-8, rtol=0.0)
                 & np.isclose(rows["field_y_relative"], 0.0, atol=1e-8, rtol=0.0))
    if np.count_nonzero(axis_edge) != 2:
        raise ValueError("Expected two axis edge points for normalized K1.")
    radius = float(np.mean(np.abs(rows["paraxial_x_mm"][axis_edge])))
    if radius <= 0:
        raise ValueError("Axis edge paraxial radius must be positive.")
    corner = (np.isclose(np.abs(rows["field_x_relative"]), 1.0, atol=1e-8, rtol=0.0)
              & np.isclose(np.abs(rows["field_y_relative"]), 1.0, atol=1e-8, rtol=0.0))
    return (radius, k1*radius**2,
            float(np.mean(rows["radial_distortion_pct"][axis_edge])),
            float(np.mean(rows["radial_distortion_pct"][corner])))


def fit_z_models(results: np.ndarray) -> np.ndarray:
    z, k1 = results["z_mm"], results["k1_per_mm2"]
    quad_design = np.column_stack([np.ones_like(z), z, z**2])
    quad_coeff = np.linalg.lstsq(quad_design, k1, rcond=None)[0]
    quad_prediction = quad_design @ quad_coeff

    # Profile the linear parameters in k1 = q0 + q1*expm1(b*Z)/b.
    # This is algebraically d + a*exp(b*Z), with a finite linear limit at b=0.
    def profile(b: float) -> tuple[float, float, float]:
        basis = z if abs(b) < 1e-12 else np.expm1(b * z) / b
        q0, q1 = np.linalg.lstsq(np.column_stack([np.ones_like(z), basis]), k1, rcond=None)[0]
        pred = q0 + q1 * basis
        return float(np.sum((pred - k1)**2)), float(q0), float(q1)

    def refine(left: float, right: float) -> tuple[float, float]:
        phi = (np.sqrt(5.0) - 1.0) / 2.0
        x1, x2 = right - phi*(right-left), left + phi*(right-left)
        f1, f2 = profile(x1)[0], profile(x2)[0]
        for _ in range(120):
            if f1 <= f2:
                right, x2, f2 = x2, x1, f1
                x1 = right - phi*(right-left)
                f1 = profile(x1)[0]
            else:
                left, x1, f1 = x1, x2, f2
                x2 = left + phi*(right-left)
                f2 = profile(x2)[0]
        b = (left+right)/2
        return b, profile(b)[0]

    low, high = -2.0, 2.0
    boundary = True
    best_b = 0.0
    for _ in range(6):
        scan = np.linspace(low, high, 8001)
        losses = np.array([profile(float(b))[0] for b in scan])
        local = [i for i in range(1, len(scan)-1)
                 if losses[i] <= losses[i-1] and losses[i] <= losses[i+1]]
        candidates = [(float(losses[0]), float(scan[0])), (float(losses[-1]), float(scan[-1]))]
        for i in local:
            b, loss = refine(float(scan[i-1]), float(scan[i+1]))
            candidates.append((loss, b))
        _, best_b = min(candidates)
        boundary = best_b in (float(scan[0]), float(scan[-1]))
        if not boundary:
            break
        low *= 2.0
        high *= 2.0

    if boundary:
        # The best profile can sit at the near-linear limit. Report its exact
        # stable limit instead of converting q1/b into unstable huge coefficients.
        loss0, q0, q1 = profile(0.0)
        best_b = 0.0
        exp_prediction = q0 + q1*z
        exp_d, exp_a = q0, q1
    else:
        _, q0, q1 = profile(best_b)
        exp_a = q1 / best_b
        exp_d = q0 - exp_a
        exp_prediction = exp_d + exp_a*np.exp(best_b*z)

    models = np.empty(2, dtype=MODEL_DTYPE)
    for i, (name, c0, c1, c2, pred) in enumerate([
        ("offset_exponential", exp_d, exp_a, best_b, exp_prediction),
        ("quadratic_Z", *quad_coeff, quad_prediction),
    ]):
        error = pred-k1
        rmse = float(np.sqrt(np.mean(error**2)))
        models[i] = (name, c0, c1, c2, rmse, float(np.max(np.abs(error))), rmse/np.ptp(k1))
    return models


def fit_magnification(reference: np.ndarray, rows: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Fit isotropic pooled XY scale and return scale, paraxial scale, residual vectors, ratios."""
    r0 = np.column_stack([reference["real_x_mm"], reference["real_y_mm"]])
    rz = np.column_stack([rows["real_x_mm"], rows["real_y_mm"]])
    p0 = np.column_stack([reference["paraxial_x_mm"], reference["paraxial_y_mm"]])
    pz = np.column_stack([rows["paraxial_x_mm"], rows["paraxial_y_mm"]])
    denom = float(np.sum(r0*r0))
    pdenom = float(np.sum(p0*p0))
    if denom <= 0 or pdenom <= 0:
        raise ValueError("Reference grid has no magnification leverage.")
    m = float(np.sum(r0*rz)/denom)
    mp = float(np.sum(p0*pz)/pdenom)
    residual = rz-m*r0
    noncentral = np.sum(r0*r0, axis=1) > 1e-20
    ratios = np.sum(r0[noncentral]*rz[noncentral], axis=1)/np.sum(r0[noncentral]**2, axis=1)
    return m, mp, residual, ratios


def magnification_models(z: np.ndarray, m: np.ndarray) -> np.ndarray:
    linear = np.polyfit(z, m, 1)[::-1]
    quadratic = np.polyfit(z, m, 2)[::-1]
    models = np.empty(2, dtype=MAG_MODEL_DTYPE)
    for i, (name, coeff) in enumerate((("linear_Z", np.r_[linear, 0.0]), ("quadratic_Z", quadratic))):
        pred = coeff[0]+coeff[1]*z+coeff[2]*z*z
        error = pred-m
        models[i] = (name, *coeff, float(np.sqrt(np.mean(error**2))), float(np.max(np.abs(error))))
    return models


def write_magnification_plot(data: np.ndarray, models: np.ndarray, path: Path) -> None:
    fig, (ax, residual_ax) = plt.subplots(2, 1, figsize=(9.2, 7), sharex=True,
        constrained_layout=True, gridspec_kw={"height_ratios": [3, 1]})
    z, m = data["z_mm"], data["real_magnification"]
    ax.plot(z, m, "o", ms=3, color="#225ea8", label="Pooled real-grid ratio")
    grid = np.linspace(float(z.min()), float(z.max()), 301)
    colors = {"linear_Z": "#238b45", "quadratic_Z": "#756bb1"}
    for model in models:
        c0, c1, c2 = model["coefficient_0"], model["coefficient_1"], model["coefficient_2"]
        predg, pred = c0+c1*grid+c2*grid**2, c0+c1*z+c2*z**2
        ax.plot(grid, predg, color=colors[model["model"]], label=model["model"].replace("_", " "))
        residual_ax.plot(z, (pred-m)*1e6, color=colors[model["model"]], label=model["model"].replace("_", " "))
    residual_ax.axhline(0, color="black", linewidth=.8)
    residual_ax.set_ylabel("Curve error (ppm)")
    residual_ax.set_xlabel("Absolute THI Z (mm)")
    residual_ax.grid(True, alpha=.3)
    residual_ax.legend(fontsize=8, ncol=2)
    ax.set_ylabel("Real-grid magnification ratio m(Z)")
    ax.set_title("P1 matched-field magnification relative to the Z=0 grid")
    ax.grid(True, alpha=.3)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def add_magnification_report(lines: list[str], data: np.ndarray, models: np.ndarray) -> None:
    m, mp = data["real_magnification"], data["paraxial_magnification"]
    ratio_delta = data["edge_corner_ratio_difference"]
    lines.extend(["", "## Matched-field Z magnification", "",
        "The fixed reference is the actual nine-point real-coordinate grid at Z=0. For each Z, one isotropic scale is fit by pooled XY least squares: `m(Z)=sum(R0·RZ)/sum(R0·R0)`, with predicted points `m(Z)R0`. The center is included in the pooled fit (it contributes zero when centered); per-point projected ratios use only the eight noncentral points, avoiding division by a zero reference coordinate. The parallel paraxial ratio uses the same fit against the Z=0 paraxial grid. Coordinate RMSE is over all 18 predicted coordinate residuals; maximum point error is the largest Euclidean residual across nine points.", "",
        "| Curve | c0 | c1 (mm^-1) | c2 (mm^-2) | Ratio RMSE | Maximum absolute ratio error |", "|---|---:|---:|---:|---:|---:|"])
    for model in models:
        lines.append(f"| {model['model']} | {model['coefficient_0']:.12g} | {model['coefficient_1']:.12g} | {model['coefficient_2']:.12g} | {model['rmse']:.8g} | {model['max_abs_error']:.8g} |")
    lines.append("")
    lines.append(f"Across all 51 fitted Z planes, m(Z) spans {m.min():.10g} to {m.max():.10g}; the largest real-grid coordinate RMSE is {data['coordinate_rmse_mm'].max()*1000:.6g} µm and largest point error is {data['max_point_residual_um'].max():.6g} µm. The edge-to-corner difference in mean projected point ratio ranges from {ratio_delta.min():.8g} to {ratio_delta.max():.8g}, assessing whether a common scale describes the full field. After dividing each Z grid by m(Z), the normalized-grid coordinate RMSE relative to Z=0 ranges from {data['normalized_shape_coordinate_rmse_um'].min():.6g} to {data['normalized_shape_coordinate_rmse_um'].max():.6g} µm. This directly tests the residual barrel-shape change after removing the fitted scale.")
    lines.append(f"The paraxial ratio spans {mp.min():.10g} to {mp.max():.10g}. The real-grid ratio is therefore an empirical image magnification relative to the actual Z=0 image, while the paraxial ratio isolates the corresponding reference-ray scale. `normalized_k1_scaled = k1(Z)*m_paraxial(Z)^2` has span {np.ptp(data['normalized_k1_scaled']):.8g}; its comparison with k1(0) checks the expected inverse-square coefficient scaling under a common image scale.")
    lines.extend(["", "| Z (mm) | Real m(Z) | Paraxial m(Z) | Coordinate RMSE (µm) | Max point error (µm) | Edge ratio mean | Corner ratio mean | Edge-corner difference | Normalized-grid RMSE (µm) | k1·mP² |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for target in (-5.0, -2.6, 0.0, 2.6, 5.0):
        row = data[np.argmin(np.abs(data["z_mm"]-target))]
        lines.append(f"| {row['z_mm']:.1f} | {row['real_magnification']:.10g} | {row['paraxial_magnification']:.10g} | {row['coordinate_rmse_mm']*1000:.7g} | {row['max_point_residual_um']:.7g} | {row['edge_ratio_mean']:.10g} | {row['corner_ratio_mean']:.10g} | {row['edge_corner_ratio_difference']:.8g} | {row['normalized_shape_coordinate_rmse_um']:.7g} | {row['normalized_k1_scaled']:.10g} |")


def model_predictions(models: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    exp = models[models["model"] == "offset_exponential"][0]
    quad = models[models["model"] == "quadratic_Z"][0]
    if abs(exp["coefficient_2"]) < 1e-12:
        pred_exp = exp["coefficient_0"] + exp["coefficient_1"]*z
    else:
        pred_exp = exp["coefficient_0"] + exp["coefficient_1"]*np.exp(exp["coefficient_2"]*z)
    pred_quad = quad["coefficient_0"] + quad["coefficient_1"]*z + quad["coefficient_2"]*z**2
    return pred_exp, pred_quad


def write_plot(results: np.ndarray, models: np.ndarray, path: Path) -> None:
    fig, (ax, residual_ax) = plt.subplots(2, 1, figsize=(9.2, 7), sharex=True,
        constrained_layout=True, gridspec_kw={"height_ratios": [3, 1]})
    z = results["z_mm"]
    ax.scatter(z, results["k1_per_mm2"], s=14, color="#225ea8", label="Per-Z k1 fits")
    grid = np.linspace(float(z.min()), float(z.max()), 301)
    curves = [("offset_exponential", "d + a exp(b Z)", "#238b45"),
              ("quadratic_Z", "c0 + c1 Z + c2 Z²", "#756bb1")]
    for name, label, color in curves:
        pred_grid = model_predictions(models, grid)[0 if name == "offset_exponential" else 1]
        pred_sample = model_predictions(models, z)[0 if name == "offset_exponential" else 1]
        ax.plot(grid, pred_grid, color=color, label=label)
        residual_ax.plot(z, (pred_sample-results["k1_per_mm2"])*1e4, color=color, label=label)
    residual_ax.axhline(0, color="black", linewidth=.8)
    residual_ax.set_ylabel(r"k1 error ($10^{-4}$ mm$^{-2}$)")
    residual_ax.set_xlabel("Absolute THI Z (mm)")
    residual_ax.grid(True, alpha=.3)
    residual_ax.legend(fontsize=8, ncol=2, loc="upper center")
    ax.set_ylabel(r"Radial coefficient $k_1$ (mm$^{-2}$)")
    ax.set_title("P1 radial distortion versus absolute THI Z at 0° eye rotation")
    ax.grid(True, alpha=.3)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(results: np.ndarray, models: np.ndarray, magnification: np.ndarray,
                 mag_models: np.ndarray, path: Path) -> None:
    lines = [
        "# P1 radial distortion versus absolute THI Z", "",
        "Scope: all 51 in-sample Z levels from -5 to +5 mm, eye rotation 0° only; each group has nine field points. Z is the absolute THI `Cornea_ENT_D` coordinate. The source sweep baseline is 1 mm; this report does not convert Z to baseline displacement.", "",
        "At each Z, one coefficient is fit by pooled least squares over all 18 x/y coordinates using `real_x = x*(1+k1*r²)` and `real_y = y*(1+k1*r²)`, where `(x,y)` are that Z group's paraxial image coordinates and `r²=x²+y²` in mm². Thus k1 has units mm^-2. Coordinate RMSE is over the 18 coordinate residuals. These spatial residuals measure the radial model fit to each sampled grid. Since paraxial image scale varies with Z, k1 alone does not measure scale-independent barrel strength. Dimensionless normalized K1 is `k1*r_edge²`, with `r_edge` the paraxial radius of the two horizontal noncorner axis-edge points. Edge and corner radial percentages are the stored sweep values averaged over symmetric field points.", "",
        "The curves fit the 51 k1 estimates directly in k1 space. The offset exponential is `k1=d+a*exp(b*Z)` (coefficient order d, a, b; d/a in mm^-2, b in mm^-1). Its signed b is scanned across both signs and interior minima refined using a stable profiled basis `expm1(b*Z)/b`; its b→0 limit is reported as a line to avoid unstable huge d/a values. The quadratic is `k1=c0+c1*Z+c2*Z²` with coefficient units mm^-2, mm^-3, mm^-4. Curve RMSE/max error are in mm^-2 and are distinct from coordinate grid RMSE in mm.", "",
        "## Z-dependent curve fits", "",
        "| Model | Coefficient 0 | Coefficient 1 | Coefficient 2 | k1 RMSE (mm^-2) | k1 RMSE (% of span) | Max absolute k1 error (mm^-2) |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for m in models:
        lines.append(f"| {m['model']} | {m['coefficient_0']:.12g} | {m['coefficient_1']:.12g} | {m['coefficient_2']:.12g} | {m['rmse_k1_per_mm2']:.8g} | {100*m['rmse_fraction_of_k1_span']:.5g}% | {m['max_abs_error_k1_per_mm2']:.8g} |")
    errors, maxima = results["coordinate_rmse_mm"], results["max_abs_residual_mm"]
    norm, edge, corner = results["normalized_k1"], results["mean_edge_radial_distortion_pct"], results["mean_corner_radial_distortion_pct"]
    lines += ["", f"Across {len(results)} Z groups, grid coordinate RMSE ranges {errors.min():.8g}–{errors.max():.8g} mm (mean {errors.mean():.8g} mm); maximum absolute coordinate residual ranges {maxima.min():.8g}–{maxima.max():.8g} mm (mean {maxima.mean():.8g} mm). Normalized K1 ranges {norm.min():.10g} to {norm.max():.10g} (span {np.ptp(norm):.6g}, {100*np.ptp(norm)/abs(np.mean(norm)):.5g}% of absolute mean). Mean edge radial distortion spans {edge.min():.8g}% to {edge.max():.8g}% ({np.ptp(edge):.6g} percentage points); mean corner distortion spans {corner.min():.8g}% to {corner.max():.8g}% ({np.ptp(corner):.6g} points). These dimensionless measures show how the sampled barrel distortion changes after accounting for paraxial scale.", "", "## Selected Z samples", "", "| Absolute THI Z (mm) | k1 (mm^-2) | Edge radius (mm) | normalized K1 | Edge radial distortion (%) | Corner radial distortion (%) | Coordinate RMSE (mm) | Max absolute residual (mm) |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for target in (-5.0, -2.6, 0.0, 2.6, 5.0):
        row = results[np.argmin(np.abs(results["z_mm"]-target))]
        lines.append(f"| {row['z_mm']:.1f} | {row['k1_per_mm2']:.10g} | {row['paraxial_edge_radius_mm']:.8g} | {row['normalized_k1']:.10g} | {row['mean_edge_radial_distortion_pct']:.8g} | {row['mean_corner_radial_distortion_pct']:.8g} | {row['coordinate_rmse_mm']:.8g} | {row['max_abs_residual_mm']:.8g} |")
    add_magnification_report(lines, magnification, mag_models)
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    groups = load_zero_rotation_groups(args.input)
    results = np.empty(len(groups), dtype=RESULT_DTYPE)
    for i, (z, rows) in enumerate(groups):
        k1, rmse, maximum = fit_group(z, rows)
        results[i] = (z, k1, rmse, maximum, *scale_and_distortion(rows, k1))
    models = fit_z_models(results)
    reference = next(rows for z, rows in groups if np.isclose(z, 0.0, atol=1e-8, rtol=0.0))
    mag_results = np.empty(len(groups), dtype=MAGNIFICATION_DTYPE)
    for i, (z, rows) in enumerate(groups):
        if not np.array_equal(rows["field_x_relative"], reference["field_x_relative"]) or not np.array_equal(rows["field_y_relative"], reference["field_y_relative"]):
            raise ValueError(f"Z={z:g} mm field order/grid does not match the Z=0 reference.")
        m, mp, residual, ratios = fit_magnification(reference, rows)
        point_norm = np.linalg.norm(residual, axis=1)
        normalized = np.column_stack([rows["real_x_mm"], rows["real_y_mm"]])/m
        baseline = np.column_stack([reference["real_x_mm"], reference["real_y_mm"]])
        shape_delta = normalized-baseline
        edge = (np.isclose(np.abs(rows["field_x_relative"]), 1.0, atol=1e-8, rtol=0.0)
                & np.isclose(rows["field_y_relative"], 0.0, atol=1e-8, rtol=0.0))
        corner = (np.isclose(np.abs(rows["field_x_relative"]), 1.0, atol=1e-8, rtol=0.0)
                  & np.isclose(np.abs(rows["field_y_relative"]), 1.0, atol=1e-8, rtol=0.0))
        # ratios are emitted in row order after excluding center; map them by the same mask.
        mask = np.sum(baseline*baseline, axis=1) > 1e-20
        edge_mean, corner_mean = float(np.mean(ratios[edge[mask]])), float(np.mean(ratios[corner[mask]]))
        k1 = results[i]["k1_per_mm2"]
        mag_results[i] = (z, m, mp, float(np.sqrt(np.mean(residual**2))),
            float(np.max(np.abs(residual))), float(np.max(point_norm)*1000), edge_mean,
            corner_mean, edge_mean-corner_mean, float(np.sqrt(np.mean(shape_delta**2))*1000),
            float(np.max(np.linalg.norm(shape_delta, axis=1))*1000), k1*mp**2)
    mag_models = magnification_models(mag_results["z_mm"], mag_results["real_magnification"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        ("p1_radial_distortion_fit.pkl", results),
        ("p1_radial_distortion_z_models.pkl", models),
        ("p1_z_magnification.pkl", {"baseline_z_mm": 0.0, "real_grid_ratio": mag_results["real_magnification"],
            "paraxial_grid_ratio": mag_results["paraxial_magnification"], "metrics": mag_results,
            "curve_models": mag_models}),
    ]
    for filename, obj in outputs:
        with (args.output_dir/filename).open("wb") as stream:
            pickle.dump(obj, stream, protocol=pickle.HIGHEST_PROTOCOL)
    plot = args.output_dir/"p1_radial_distortion_k1.png"
    report = args.output_dir/"p1_radial_distortion_fit.md"
    write_plot(results, models, plot)
    mag_plot = args.output_dir/"p1_z_magnification.png"
    write_magnification_plot(mag_results, mag_models, mag_plot)
    write_report(results, models, mag_results, mag_models, report)
    print(f"Validated and fit {len(results)} groups with nine points each at eye rotation 0 deg.")
    print(f"k1 range: {results['k1_per_mm2'].min():.10g} to {results['k1_per_mm2'].max():.10g} mm^-2")
    print(f"Grid coordinate RMSE range: {results['coordinate_rmse_mm'].min():.8g} to {results['coordinate_rmse_mm'].max():.8g} mm")
    print(f"Grid maximum residual range: {results['max_abs_residual_mm'].min():.8g} to {results['max_abs_residual_mm'].max():.8g} mm")
    for model in models:
        print(f"{model['model']}: coefficients=({model['coefficient_0']:.12g}, {model['coefficient_1']:.12g}, {model['coefficient_2']:.12g}), k1 RMSE={model['rmse_k1_per_mm2']:.8g} mm^-2 ({100*model['rmse_fraction_of_k1_span']:.5g}% span), max={model['max_abs_error_k1_per_mm2']:.8g} mm^-2")
    print(f"Saved {plot.resolve()}")
    print(f"Saved {report.resolve()}")
    print(f"Saved {mag_plot.resolve()}")


if __name__ == "__main__":
    main()
