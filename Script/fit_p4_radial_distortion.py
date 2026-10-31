"""Fit and assess P4's one-coefficient radial distortion model at zero rotation."""

from __future__ import annotations

import argparse
import hashlib
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_INPUT = PROJECT_DIR / "data" / "distortion_grid_p4" / "distortion_grid.pkl"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "p4_radial_distortion_fit"
REQUIRED_FIELDS = ("eye_rotation_deg", "accommodation_d", "z_mm", "field_x_relative", "field_y_relative",
                   "paraxial_x_mm", "paraxial_y_mm", "real_x_mm", "real_y_mm")
RESULT_DTYPE = np.dtype([("accommodation_d", "f8"), ("k1_per_mm2", "f8"), ("coordinate_rmse_mm", "f8"),
                         ("max_abs_residual_mm", "f8"), ("max_euclidean_error_mm", "f8")])
MODEL_DTYPE = np.dtype([("model", "U32"), ("coefficient_0", "f8"), ("coefficient_1", "f8"),
                        ("coefficient_2", "f8"), ("rmse_k1_per_mm2", "f8"),
                        ("max_abs_error_k1_per_mm2", "f8"), ("rmse_fraction_of_k1_span", "f8")])


def load_zero_rotation_groups(input_path: Path, z_mm: float = 0.0) -> list[tuple[float, np.ndarray]]:
    if not input_path.is_file():
        raise FileNotFoundError(f"P4 distortion pickle not found: {input_path}")
    with input_path.open("rb") as stream:
        data = pickle.load(stream)
    if not isinstance(data, np.ndarray) or data.ndim != 1 or data.dtype.names is None:
        raise ValueError("Input pickle must contain a 1D structured NumPy array.")
    missing = [name for name in REQUIRED_FIELDS if name not in data.dtype.names]
    if missing:
        raise ValueError(f"Input pickle is missing fields: {', '.join(missing)}")
    selected = data[np.isclose(data["eye_rotation_deg"], 0, rtol=0, atol=1e-8) &
                     np.isclose(data["z_mm"], z_mm, rtol=0, atol=1e-8)]
    accommodations = np.unique(selected["accommodation_d"])
    expected_a = np.arange(51, dtype=float) / 10
    if not np.allclose(accommodations, expected_a, rtol=0, atol=1e-8):
        raise ValueError(f"Expected accommodation values 0.0 through 5.0 D in 0.1 D steps; found {len(accommodations)} values.")
    expected_fields = np.unique(np.array([(x, y) for y in (-1., 0., 1.) for x in (-1., 0., 1.)]), axis=0)
    groups = []
    for accommodation in accommodations:
        rows = selected[np.isclose(selected["accommodation_d"], accommodation, rtol=0, atol=1e-8)]
        fields = np.column_stack((rows["field_x_relative"], rows["field_y_relative"]))
        if len(rows) != 9 or not np.array_equal(np.unique(fields, axis=0), expected_fields):
            raise ValueError(f"Accommodation {accommodation:g} D must contain exactly the expected 3x3 field identities.")
        coords = np.column_stack([rows[name] for name in REQUIRED_FIELDS[5:]])
        if not np.all(np.isfinite(coords)):
            raise ValueError(f"Accommodation {accommodation:g} D contains nonfinite coordinates.")
        # Canonical order makes saved per-field arrays reproducible.
        order = np.lexsort((rows["field_x_relative"], rows["field_y_relative"]))
        groups.append((float(accommodation), rows[order]))
    return groups


def fit_group(accommodation: float, rows: np.ndarray) -> tuple[float, float, float]:
    x, y = np.asarray(rows["paraxial_x_mm"], float), np.asarray(rows["paraxial_y_mm"], float)
    rx, ry = np.asarray(rows["real_x_mm"], float), np.asarray(rows["real_y_mm"], float)
    r2 = x*x + y*y
    design = np.concatenate((x*r2, y*r2))
    displacement = np.concatenate((rx-x, ry-y))
    denom = float(design @ design)
    if not np.isfinite(denom) or denom <= 0:
        raise ValueError(f"Accommodation {accommodation:g} D has no fit leverage.")
    k1 = float(design @ displacement / denom)
    resid = displacement - k1*design
    return k1, float(np.sqrt(np.mean(resid**2))), float(np.max(np.abs(resid)))


def coordinate_errors(rows: np.ndarray, k1: float) -> tuple[float, float, float, np.ndarray]:
    x, y = rows["paraxial_x_mm"], rows["paraxial_y_mm"]
    predx, predy = x*(1+k1*(x*x+y*y)), y*(1+k1*(x*x+y*y))
    err = np.column_stack((predx-rows["real_x_mm"], predy-rows["real_y_mm"]))
    return float(np.sqrt(np.mean(err**2))), float(np.max(np.abs(err))), float(np.max(np.linalg.norm(err, axis=1))), err


def fit_accommodation_models(results: np.ndarray) -> np.ndarray:
    a, k = results["accommodation_d"], results["k1_per_mm2"]
    quad = np.linalg.lstsq(np.column_stack((np.ones_like(a), a, a*a)), k, rcond=None)[0]
    qp = quad[0] + quad[1]*a + quad[2]*a*a

    # Fit c0, c1, and n directly to the 51 measured accommodation k1 values.
    # The positive lower bound defines 0**n as zero at the A=0 sample.
    power_initial = np.array([k[0], (k[-1] - k[0]) / a[-1], 1.0])
    power_fit = least_squares(
        lambda p: p[0] + p[1] * np.power(a, p[2]) - k,
        power_initial,
        bounds=([-np.inf, -np.inf, 1e-6], [np.inf, np.inf, 10.0]),
        x_scale=np.array([0.02, 0.001, 1.0]),
        ftol=1e-14,
        xtol=1e-14,
        gtol=1e-14,
        max_nfev=10000,
    )
    if not power_fit.success:
        raise RuntimeError(f"Power-law accommodation fit failed: {power_fit.message}")
    power_c0, power_c1, power_n = power_fit.x
    powerp = power_c0 + power_c1 * np.power(a, power_n)

    # Profile the linear coefficients for each slope, using a basis whose b->0 limit is A.
    def profile(b: float):
        basis = a if abs(b) < 1e-8 else np.expm1(b*a)/b
        c0, c1 = np.linalg.lstsq(np.column_stack((np.ones_like(a), basis)), k, rcond=None)[0]
        p = c0+c1*basis
        return float(np.sum((p-k)**2)), float(c0), float(c1), p

    phi = (np.sqrt(5)-1)/2
    def refine(left: float, right: float) -> tuple[float, float]:
        x1, x2 = right-phi*(right-left), left+phi*(right-left)
        f1, f2 = profile(x1)[0], profile(x2)[0]
        for _ in range(100):
            if f1 <= f2:
                right, x2, f2 = x2, x1, f1
                x1 = right-phi*(right-left); f1 = profile(x1)[0]
            else:
                left, x1, f1 = x1, x2, f2
                x2 = left+phi*(right-left); f2 = profile(x2)[0]
        b = (left+right)/2
        return b, profile(b)[0]

    low, high = -2.0, 2.0
    for _ in range(5):
        grid = np.linspace(low, high, 4001)
        scores = np.array([profile(float(b))[0] for b in grid])
        minima = [i for i in range(1, len(grid)-1) if scores[i] <= scores[i-1] and scores[i] <= scores[i+1]]
        candidates = [(float(scores[0]), float(grid[0])), (float(scores[-1]), float(grid[-1]))]
        candidates.extend((refine(float(grid[i-1]), float(grid[i+1]))[1], refine(float(grid[i-1]), float(grid[i+1]))[0]) for i in minima)
        _, b = min(candidates)
        if b not in (float(grid[0]), float(grid[-1])):
            break
        low *= 2; high *= 2
    else:
        raise RuntimeError("Offset exponential fit remained at a slope-search boundary after expansion.")
    _, q0, q1, _ = profile(b)
    # Preserve the exact linear-limit law rather than reporting it as d+a*exp(bA).
    if abs(b) < 1e-8:
        exp_name, exp_c0, exp_c1, ep = "offset_exponential_linear_limit", q0, q1, q0+q1*a
        exp_b = 0.0
    else:
        exp_name, exp_c1, exp_c0 = "offset_exponential", q1/b, q0-q1/b
        ep = exp_c0+exp_c1*np.exp(b*a)
        exp_b = b
    models = np.empty(3, dtype=MODEL_DTYPE)
    for i, (name, coeff, pred) in enumerate(((exp_name, (exp_c0, exp_c1, exp_b), ep),
                                               ("quadratic_A", quad, qp),
                                               ("power_law", (power_c0, power_c1, power_n), powerp))):
        residual = pred-k; rmse = float(np.sqrt(np.mean(residual**2)))
        models[i] = (name, *coeff, rmse, np.max(np.abs(residual)), rmse/np.ptp(k))
    return models


def model_predictions(models: np.ndarray, accommodation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    exponential_rows = models[np.isin(models["model"], ["offset_exponential", "offset_exponential_linear_limit"])]
    exp = exponential_rows[0]
    quad = models[models["model"] == "quadratic_A"][0]
    power = models[models["model"] == "power_law"][0]
    if exp["model"] == "offset_exponential_linear_limit":
        ep = exp["coefficient_0"]+exp["coefficient_1"]*accommodation
    else:
        ep = exp["coefficient_0"]+exp["coefficient_1"]*np.exp(exp["coefficient_2"]*accommodation)
    qp = quad["coefficient_0"]+quad["coefficient_1"]*accommodation+quad["coefficient_2"]*accommodation**2
    powerp = power["coefficient_0"] + power["coefficient_1"] * np.power(accommodation, power["coefficient_2"])
    return ep, qp, powerp


def write_plot(results: np.ndarray, models: np.ndarray, path: Path, curve_metrics: np.ndarray | None = None) -> None:
    fig, (ax, residual_ax, added_ax) = plt.subplots(3, 1, figsize=(9.2, 8.2), sharex=True, constrained_layout=False,
                                        gridspec_kw={"height_ratios": [3, 1.4, 1]})
    a, k = results["accommodation_d"], results["k1_per_mm2"]
    ax.scatter(a, k, s=14, color="#225ea8", label="Pooled direct k1 fits")
    grid = np.linspace(a.min(), a.max(), 301)
    curves = model_predictions(models, grid)
    exp_name = "offset_exponential_linear_limit" if "offset_exponential_linear_limit" in models["model"] else "offset_exponential"
    exp_label = "c0 + c1 A (linear limit)" if exp_name == "offset_exponential_linear_limit" else "d + a exp(b A)"
    for j, (name, color, label) in enumerate(((exp_name, "#238b45", exp_label),
                                               ("quadratic_A", "#756bb1", "p0 + p1 A + p2 A^2"),
                                               ("power_law", "#d95f0e", "c1 A^n + c0"))):
        ax.plot(grid, curves[j], color=color, label=label)
    if curve_metrics is not None:
        residual_ax.plot(a, curve_metrics["direct_rmse_mm"]*1000, color="#225ea8", label="Direct-fit coordinate RMSE")
        residual_ax.plot(a, curve_metrics["direct_max_euclidean_mm"]*1000, color="#111111", linestyle=":", label="Direct-fit max point error")
        residual_ax.plot(a, curve_metrics["quadratic_total_rmse_mm"]*1000, color="#756bb1", label="Quadratic-curve total RMSE")
        residual_ax.plot(a, curve_metrics["exponential_total_rmse_mm"]*1000, color="#238b45", label="Exponential-curve total RMSE")
        residual_ax.plot(a, curve_metrics["power_total_rmse_mm"]*1000, color="#d95f0e", label="Power-curve total RMSE")
        added_ax.plot(a, curve_metrics["quadratic_added_rmse_mm"]*1000, color="#756bb1", label="Quadratic added RMSE")
        added_ax.plot(a, curve_metrics["exponential_added_rmse_mm"]*1000, color="#238b45", label="Offset exponential added RMSE")
        added_ax.plot(a, curve_metrics["power_added_rmse_mm"]*1000, color="#d95f0e", label="Power-law added RMSE")
    residual_ax.axhline(1, color="#d7301f", linestyle="--", linewidth=.9, label="1 um")
    residual_ax.set_ylabel("Coordinate error (um)"); residual_ax.set_xlabel("Accommodation (D)")
    residual_ax.grid(True, alpha=.3); residual_ax.legend(fontsize=7, ncol=2)
    added_ax.set_ylabel("Added error (um)"); added_ax.set_xlabel("Accommodation (D)")
    added_ax.grid(True, alpha=.3); added_ax.legend(fontsize=7, ncol=2)
    ax.set_ylabel("Radial coefficient k1 (mm^-2)"); ax.set_title("P4 radial distortion at eye rotation 0 deg, z=0 mm")
    ax.grid(True, alpha=.3); ax.legend()
    fig.tight_layout(rect=(0, .01, 1, .97), pad=1.0)
    fig.savefig(path, dpi=180); plt.close(fig)


def write_report(results: np.ndarray, models: np.ndarray, metrics: np.ndarray, z_mm: float, sha: str, path: Path) -> None:
    scope = (f"Input SHA256: `{sha}`. In-sample scope: eye_rotation_deg=0, z_mm={z_mm:g}; "
             "accommodation 0.0-5.0 D in 0.1 D steps; normalized field identities ordered by y then x over [-1, 0, 1]. "
             "Each of the 51 groups contains exactly those nine field identities.")
    details = ("Forward coordinate law: real_x=x(1+k1*r^2), real_y=y(1+k1*r^2), where r^2=x^2+y^2 "
               "in mm^2 and k1 is in mm^-2. One signed k1 per accommodation was estimated by pooled Cartesian "
               "least squares over the nine points and both coordinates, using that accommodation's own stored "
               "paraxial coordinates as reference. The 3x3 sampled grid has two nonzero radial classes: edge "
               "midpoints and corners. Fit errors describe simulated coordinate residuals; they do not measure "
               "angular precision. Paraxial scale changes with accommodation and is a baseline magnification "
               "diagnostic, distinct from the fitted change in k1.")
    curves_text = ("The in-sample quadratic accommodation curve is k1=p0+p1*A+p2*A^2, coefficient order "
                   "(p0,p1,p2), units mm^-2, mm^-2/D, and mm^-2/D^2. The signed offset exponential is "
                   "k1=d+a*exp(b*A), coefficient order (d,a,b), units mm^-2, mm^-2, and D^-1. "
                   "The power law is k1=c1*A^n+c0, with all three parameters fitted directly to the 51 direct "
                   "per-accommodation fitted k1 values from the simulated grid. "
                   "c0 has units mm^-2, c1 has units mm^-2/D^n, and n is dimensionless. "
                   "The nonlinear least-squares initialization is c0=k1(0), c1=(k1(5)-k1(0))/5, n=1; bounds are "
                   "1e-6 <= n <= 10 with c0 and c1 unbounded. In the coefficient table, power_law columns 0, 1, "
                   "and 2 are c0, c1, and n respectively. The bounded nonlinear least-squares fit converged successfully. "
                   "If the optimum is numerically at the linear limit, the row is explicitly named "
                   "offset_exponential_linear_limit and means k1=c0+c1*A, coefficients (c0,c1,0).")
    error_text = ("Direct-fit errors compare predicted coordinates with stored real coordinates. Curve-added error "
                  "is the coordinate difference between the curve's k1 prediction and direct k1 prediction; total "
                  "truth error compares the curve prediction with stored real coordinates. Metrics use all nine "
                  "points (18 coordinate residuals); point maximum is the largest 2D Euclidean error. The 1 um "
                  "reference applies to coordinate error.")
    lines = ["# P4 radial distortion fit", "", scope, "", details, "", curves_text, "", error_text, "",
             "## Accommodation curves", "",
             "| Model | Coefficient 0 | Coefficient 1 | Coefficient 2 | k1 RMSE (mm^-2) | Maximum absolute k1 error (mm^-2) |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in models:
        lines.append(f"| {row['model']} | {row['coefficient_0']:.12g} | {row['coefficient_1']:.12g} | {row['coefficient_2']:.12g} | {row['rmse_k1_per_mm2']:.8g} | {row['max_abs_error_k1_per_mm2']:.8g} |")
    names = ("direct_rmse_mm", "direct_max_abs_mm", "direct_max_euclidean_mm",
             "quadratic_added_rmse_mm", "quadratic_added_max_euclidean_mm", "quadratic_total_rmse_mm",
             "quadratic_total_max_euclidean_mm", "exponential_added_rmse_mm",
             "exponential_added_max_euclidean_mm", "exponential_total_rmse_mm",
             "exponential_total_max_euclidean_mm", "power_added_rmse_mm",
             "power_added_max_euclidean_mm", "power_total_rmse_mm", "power_total_max_euclidean_mm")
    headers = ("A (D)", "paraxial scale vs A0", "scale residual RMSE", "direct RMSE",
               "direct max abs coord", "direct max point Euclidean", "quadratic added RMSE",
               "quadratic added max point", "quadratic total RMSE", "quadratic total max point",
               "exponential added RMSE", "exponential added max point", "exponential total RMSE",
               "exponential total max point", "power added RMSE", "power added max point",
               "power total RMSE", "power total max point")
    lines += ["", "## Per-accommodation coordinate metrics", "",
              "Coordinate errors and scale residual RMSE are in micrometers. Paraxial relative scale is a pooled "
              "dot-product fit to the A=0 paraxial coordinate vectors in matching field order. Its residual "
              "measures departures from one isotropic scale and is diagnostic only.", "",
              "| " + " | ".join(headers) + " |", "|" + "|".join(["---:"] * len(headers)) + "|"]
    for row in metrics:
        cells = [f"{row['accommodation_d']:.1f}", f"{row['paraxial_scale_vs_a0']:.9g}",
                 f"{row['scale_residual_rmse_mm']*1000:.6g}"]
        cells.extend(f"{row[name]*1000:.6g}" for name in names)
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

METRIC_DTYPE = np.dtype([("accommodation_d", "f8"), ("paraxial_scale_vs_a0", "f8"), ("scale_residual_rmse_mm", "f8"), ("direct_rmse_mm", "f8"), ("direct_max_abs_mm", "f8"), ("direct_max_euclidean_mm", "f8"), ("quadratic_added_rmse_mm", "f8"), ("quadratic_added_max_euclidean_mm", "f8"), ("quadratic_total_rmse_mm", "f8"), ("quadratic_total_max_euclidean_mm", "f8"), ("exponential_added_rmse_mm", "f8"), ("exponential_added_max_euclidean_mm", "f8"), ("exponential_total_rmse_mm", "f8"), ("exponential_total_max_euclidean_mm", "f8"), ("power_added_rmse_mm", "f8"), ("power_added_max_euclidean_mm", "f8"), ("power_total_rmse_mm", "f8"), ("power_total_max_euclidean_mm", "f8")])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--z-mm", type=float, default=0.0)
    args = parser.parse_args()
    groups = load_zero_rotation_groups(args.input, args.z_mm)
    results = np.empty(len(groups), dtype=RESULT_DTYPE)
    for i, (a, rows) in enumerate(groups):
        k1, _, _ = fit_group(a, rows)
        rmse, maxabs, maxeuclid, _ = coordinate_errors(rows, k1)
        results[i] = (a, k1, rmse, maxabs, maxeuclid)
    models = fit_accommodation_models(results)
    ep, qp, powerp = model_predictions(models, results["accommodation_d"])
    metrics = np.empty(len(groups), dtype=METRIC_DTYPE)
    baseline_xy = np.column_stack((groups[0][1]["paraxial_x_mm"], groups[0][1]["paraxial_y_mm"]))
    baseline_flat = baseline_xy.ravel()
    baseline_norm = float(baseline_flat @ baseline_flat)
    for i, (a, rows) in enumerate(groups):
        direct = results[i]["k1_per_mm2"]
        base_rmse, base_max, base_euclid, _ = coordinate_errors(rows, direct)
        current_xy = np.column_stack((rows["paraxial_x_mm"], rows["paraxial_y_mm"]))
        scale = float(baseline_flat @ current_xy.ravel() / baseline_norm)
        scale_resid = current_xy-scale*baseline_xy
        scale_rmse = float(np.sqrt(np.mean(scale_resid**2)))
        values = [a, scale, scale_rmse, base_rmse, base_max, base_euclid]
        for curve_k in (qp[i], ep[i], powerp[i]):
            # Added error is isolated from direct-fit residual by comparing predictions.
            x, y = rows["paraxial_x_mm"], rows["paraxial_y_mm"]; r2=x*x+y*y
            delta = (curve_k-direct)*np.column_stack((x*r2, y*r2))
            rmse_add = float(np.sqrt(np.mean(delta**2))); max_eu = float(np.max(np.linalg.norm(delta, axis=1)))
            rmse_tot, _, max_eu_tot, _ = coordinate_errors(rows, float(curve_k))
            values.extend((rmse_add, max_eu, rmse_tot, max_eu_tot))
        metrics[i] = tuple(values)
    sha = hashlib.sha256(args.input.read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {"input_path": str(args.input.resolve()), "input_sha256": sha, "eye_rotation_deg": 0.0, "z_mm": args.z_mm,
                  "accommodation_d": [0.0, 5.0, 0.1], "field_order": [[x, y] for y in (-1., 0., 1.) for x in (-1., 0., 1.)],
                  "model": "real=paraxial*(1+k1*r^2)", "fit": "pooled Cartesian least squares per accommodation"}
    # Preserve the original raw structured-array format in the two established artifacts.
    for name, value in (("p4_radial_distortion_fit.pkl", results), ("p4_radial_distortion_accommodation_models.pkl", models)):
        with (args.output_dir/name).open("wb") as stream: pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    metrics_artifact = {"provenance": provenance, "metrics": metrics,
                        "fit_pickle": "p4_radial_distortion_fit.pkl",
                        "accommodation_models_pickle": "p4_radial_distortion_accommodation_models.pkl"}
    with (args.output_dir/"p4_radial_distortion_metrics.pkl").open("wb") as stream:
        pickle.dump(metrics_artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
    plot = args.output_dir/"p4_radial_distortion_k1.png"; report = args.output_dir/"p4_radial_distortion_fit.md"
    write_plot(results, models, plot, metrics); write_report(results, models, metrics, args.z_mm, sha, report)
    print(f"Validated {len(results)} groups x 9 field identities at theta=0 deg, z={args.z_mm:g} mm; A=0..5 by 0.1.")
    print(f"Input SHA256: {sha}")
    for key, label in (("coordinate_rmse_mm", "Direct coordinate RMSE"), ("max_abs_residual_mm", "Direct maximum coordinate error"), ("max_euclidean_error_mm", "Direct maximum Euclidean point error")):
        v=results[key]*1000; print(f"{label}: range {v.min():.6g}-{v.max():.6g} um; mean {v.mean():.6g} um")
    for m in models:
        third_label = "n" if m["model"] == "power_law" else "c2"
        print(f"{m['model']}: coefficient_0={m['coefficient_0']:.12g}, coefficient_1={m['coefficient_1']:.12g}, {third_label}={m['coefficient_2']:.12g}; k1 RMSE={m['rmse_k1_per_mm2']:.8g} mm^-2; max |k1 error|={m['max_abs_error_k1_per_mm2']:.8g} mm^-2")
    for label, added, total in (("quadratic", "quadratic_added_rmse_mm", "quadratic_total_rmse_mm"), ("offset exponential", "exponential_added_rmse_mm", "exponential_total_rmse_mm"), ("power law", "power_added_rmse_mm", "power_total_rmse_mm")):
        print(f"{label} curve: accommodation-averaged added coordinate RMSE={np.mean(metrics[added])*1000:.6g} um; total truth RMSE={np.mean(metrics[total])*1000:.6g} um; max per-A total RMSE={np.max(metrics[total])*1000:.6g} um")
    for p in (args.output_dir/"p4_radial_distortion_fit.pkl", args.output_dir/"p4_radial_distortion_accommodation_models.pkl", args.output_dir/"p4_radial_distortion_metrics.pkl", plot, report): print(f"Saved {p.resolve()}")


if __name__ == "__main__": main()
