"""Evaluate noisy joint theta/head-translation estimates on random P1 states."""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import estimate_p1_sigma_depth_nonlinear as base


ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "p1_only_sigma_depth_raytracing" / "p1_only_sigma_depth_raytracing.pkl"
OUT_DIR = ROOT / "data" / "p1_only_sigma_depth_random_measurements"
RNG_SEED = 20261031
THETA_CAL_ORDER = 6
RESIDUAL_CANDIDATES = ((3, 2), (4, 2), (5, 3), (6, 3))
N_VALIDATION_H = 20
N_UNIFORM = 2000
N_NEAR_ZERO = 700
N_THETA0_DRAWS_PER_H = 100
EVAL_NOISE_MM = 0.001
SAMPLE_GROUPS = ("uniform_test", "near_zero_test", "theta0_stress")
GROUP_COLORS = {"uniform_test": "#1f77b4", "near_zero_test": "#d95f02", "theta0_stress": "#2ca02c"}
GROUP_LABELS = {
    "uniform_test": "Uniform held-out h/angle states",
    "near_zero_test": "Extra held-out |theta| <= 1 deg",
    "theta0_stress": "Exact theta=0 stress (training h levels)",
}


def add_true_theta(data: np.ndarray) -> np.ndarray:
    out = np.empty(data.shape, dtype=data.dtype.descr + [("true_theta_rad", "f8")])
    for name in data.dtype.names:
        out[name] = data[name]
    out["true_theta_rad"] = -np.deg2rad(out["rotation_deg"])
    return out


def validate_data(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = {
        "h_translation_mm", "rotation_deg", "zoom_number", "real_y_si_mm", "trace_success",
        *base.ZOOM_METADATA_FIELDS,
    }
    if not isinstance(data, np.ndarray) or data.ndim != 1 or data.dtype.names is None:
        raise ValueError("Input must be a one-dimensional structured NumPy array")
    missing = required.difference(data.dtype.names)
    if missing:
        raise ValueError(f"Input missing fields: {', '.join(sorted(missing))}")
    if not np.all(data["trace_success"]) or not np.isfinite(data["real_y_si_mm"]).all():
        raise ValueError("Input must contain only successful finite raw SI Y values")
    h_values = np.unique(data["h_translation_mm"])
    ade_values = np.unique(data["rotation_deg"])
    if data.size != 241803 or h_values.size != 201 or ade_values.size != 401:
        raise ValueError(f"Expected 201 x 401 x 3 raw sweep; got {data.size} rows")
    if not np.allclose(h_values, np.linspace(-1.0, 1.0, 201), rtol=0.0, atol=1e-12):
        raise ValueError("Unexpected translation grid")
    if not np.allclose(ade_values, np.linspace(-20.0, 20.0, 401), rtol=0.0, atol=1e-12):
        raise ValueError("Unexpected rotation grid")
    if set(np.unique(data["zoom_number"])) != set(base.ZOOMS):
        raise ValueError("Expected zooms 1, 2, and 3")
    for zoom, expected in base.EXPECTED_ZOOM_METADATA.items():
        rows = data["zoom_number"] == zoom
        for field, value in zip(base.ZOOM_METADATA_FIELDS, expected):
            if not np.array_equal(np.unique(data[field][rows]), np.array([value])):
                raise ValueError(f"Unexpected {field} metadata for z{zoom}")
    keys = np.stack([data["h_translation_mm"], data["rotation_deg"], data["zoom_number"]], axis=1)
    if np.unique(keys, axis=0).shape[0] != data.size:
        raise ValueError("Raw sweep contains duplicate h/rotation/zoom rows")
    return h_values, ade_values, np.flatnonzero(data["zoom_number"] == 1)


def polynomial_design(u: np.ndarray, degree: int) -> np.ndarray:
    return np.column_stack([u**j for j in range(degree + 1)])


def fit_baseline_h0(data: np.ndarray, h0: float) -> dict[int, np.ndarray]:
    baseline = {}
    for zoom in base.ZOOMS:
        use = (data["zoom_number"] == zoom) & np.isclose(data["h_translation_mm"], h0, atol=1e-12, rtol=0.0)
        u = data["true_theta_rad"][use] / base.THETA_SCALE_RAD
        X = polynomial_design(u, THETA_CAL_ORDER)
        coeff, *_ = np.linalg.lstsq(X, data["real_y_si_mm"][use], rcond=None)
        baseline[zoom] = coeff
    return baseline


def make_coefficients(
    data: np.ndarray,
    baseline: dict[int, np.ndarray],
    train_h_mask: np.ndarray,
    h_values: np.ndarray,
    q_degree: int,
    r_degree: int,
) -> tuple[dict[int, np.ndarray], tuple[int, int, int]]:
    orders = (THETA_CAL_ORDER, q_degree, r_degree)
    h_index = np.searchsorted(h_values, data["h_translation_mm"])
    eligible = train_h_mask[h_index] & (np.abs(data["h_translation_mm"]) > 1e-12)
    coeffs: dict[int, np.ndarray] = {}
    for zoom in base.ZOOMS:
        use = eligible & (data["zoom_number"] == zoom)
        u = data["true_theta_rad"][use] / base.THETA_SCALE_RAD
        h = data["h_translation_mm"][use]
        y = data["real_y_si_mm"][use]
        b = baseline[zoom]
        residual = y - np.polynomial.polynomial.polyval(u, b)
        X = np.column_stack([
            *(h * u**j for j in range(q_degree + 1)),
            *(h**2 * u**j for j in range(r_degree + 1)),
        ])
        qr, *_ = np.linalg.lstsq(X, residual, rcond=None)
        coeffs[zoom] = np.concatenate((b, qr))
    return coeffs, orders


def predict_rows(data: np.ndarray, coeffs: dict[int, np.ndarray], orders: tuple[int, int, int]) -> np.ndarray:
    predicted = np.full(data.size, np.nan, dtype=float)
    u = data["true_theta_rad"] / base.THETA_SCALE_RAD
    h = data["h_translation_mm"]
    for zoom in base.ZOOMS:
        use = data["zoom_number"] == zoom
        b, q, r = base.split_coefficients(coeffs[zoom], orders)
        predicted[use] = (
            np.polynomial.polynomial.polyval(u[use], b)
            + np.polynomial.polynomial.polyval(u[use], q) * h[use]
            + np.polynomial.polynomial.polyval(u[use], r) * h[use] ** 2
        )
    return predicted


def state_observations(data: np.ndarray) -> dict[tuple[float, float], np.ndarray]:
    grouped: dict[tuple[float, float], dict[int, float]] = {}
    for row in data:
        key = (float(row["h_translation_mm"]), float(row["rotation_deg"]))
        grouped.setdefault(key, {})[int(row["zoom_number"])] = float(row["real_y_si_mm"])
    if any(set(values) != set(base.ZOOMS) for values in grouped.values()):
        raise ValueError("Each sampled h/rotation state must have all three zoom observations")
    return {key: np.array([value[z] for z in base.ZOOMS]) for key, value in grouped.items()}


def validation_indices(h_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(h_values.size)
    even = indices[indices % 2 == 0]
    odd = indices[indices % 2 == 1]
    candidates = even[(even != 0) & (even != h_values.size - 1) & ~np.isclose(h_values[even], 0.0)]
    positions = np.rint(np.linspace(0, candidates.size - 1, N_VALIDATION_H)).astype(int)
    validation = candidates[np.unique(positions)]
    train_fit = even[~np.isin(even, validation)]
    if 0 not in train_fit or (h_values.size - 1) not in train_fit or not np.any(np.isclose(h_values[train_fit], 0.0)):
        raise RuntimeError("Training split must retain both endpoints and h=0")
    return train_fit, validation, odd


def forward_metrics(errors_mm: np.ndarray) -> dict[str, float]:
    errors_um = 1000.0 * errors_mm[np.isfinite(errors_mm)]
    return {
        "n": int(errors_um.size),
        "rmse_um": float(np.sqrt(np.mean(errors_um**2))),
        "mae_um": float(np.mean(np.abs(errors_um))),
        "max_abs_um": float(np.max(np.abs(errors_um))),
        "p95_abs_um": float(np.quantile(np.abs(errors_um), 0.95)),
    }


def pick_model(data: np.ndarray, baseline: dict[int, np.ndarray], h_values: np.ndarray,
               train_h_idx: np.ndarray, validation_h_idx: np.ndarray) -> tuple[dict, list[dict]]:
    train_mask = np.zeros(h_values.size, dtype=bool)
    train_mask[train_h_idx] = True
    validation_mask = np.zeros(h_values.size, dtype=bool)
    validation_mask[validation_h_idx] = True
    h_index = np.searchsorted(h_values, data["h_translation_mm"])
    validation_rows = validation_mask[h_index]
    all_results = []
    for q_deg, r_deg in RESIDUAL_CANDIDATES:
        coeffs, orders = make_coefficients(data, baseline, train_mask, h_values, q_deg, r_deg)
        prediction = predict_rows(data, coeffs, orders)
        errors = prediction[validation_rows] - data["real_y_si_mm"][validation_rows]
        item = {
            "q_degree": q_deg,
            "r_degree": r_deg,
            "orders": orders,
            "validation_overall": forward_metrics(errors),
            "validation_by_zoom": {
                z: forward_metrics(errors[data["zoom_number"][validation_rows] == z])
                for z in base.ZOOMS
            },
            "coefficients": coeffs,
        }
        all_results.append(item)
    best = min(all_results, key=lambda item: item["validation_overall"]["rmse_um"])
    return best, all_results


def make_samples(data: np.ndarray, h_values: np.ndarray, ade_values: np.ndarray,
                 odd_h_idx: np.ndarray, rng: np.random.Generator) -> tuple[list[dict], dict[tuple[float, float], np.ndarray]]:
    observations = state_observations(data)
    test_h = h_values[odd_h_idx]
    test_states = [(float(h), float(ade)) for h in test_h for ade in ade_values]
    uniform_idx = rng.choice(len(test_states), size=N_UNIFORM, replace=False)
    uniform_states = {test_states[i] for i in uniform_idx}
    near_pool = [
        state for state in test_states
        if abs(-state[1]) <= 1.0 + 1e-10 and state not in uniform_states
    ]
    near_idx = rng.choice(len(near_pool), size=N_NEAR_ZERO, replace=False)
    samples: list[dict] = []
    for group, selected in (("uniform_test", [test_states[i] for i in uniform_idx]),
                            ("near_zero_test", [near_pool[i] for i in near_idx])):
        for h, ade in selected:
            y = observations[(h, ade)]
            samples.append({
                "group": group, "h_true": h, "ade_true": ade,
                "theta_true_deg": -ade, "y_true": y,
            })
    for h in (-1.0, 0.0, 1.0):
        y = observations[(h, 0.0)]
        for _ in range(N_THETA0_DRAWS_PER_H):
            samples.append({
                "group": "theta0_stress", "h_true": h, "ade_true": 0.0,
                "theta_true_deg": 0.0, "y_true": y,
            })
    return samples, observations


def run_inversions(samples: list[dict], coeffs: dict[int, np.ndarray], orders: tuple[int, int, int],
                   rng: np.random.Generator) -> np.ndarray:
    templates = base.prepare_templates(coeffs, orders)
    dtype = np.dtype([
        ("sample_group", "U20"), ("h_true_mm", "f8"), ("ADE_true_deg", "f8"),
        ("theta_true_deg", "f8"),
        ("y1_true_mm", "f8"), ("y2_true_mm", "f8"), ("y3_true_mm", "f8"),
        ("noise1_mm", "f8"), ("noise2_mm", "f8"), ("noise3_mm", "f8"),
        ("y1_noisy_mm", "f8"), ("y2_noisy_mm", "f8"), ("y3_noisy_mm", "f8"),
        ("theta_est_noiseless_deg", "f8"), ("h_est_noiseless_mm", "f8"),
        ("theta_est_noisy_deg", "f8"), ("h_est_noisy_mm", "f8"),
        ("noiseless_chi2", "f8"), ("noisy_chi2", "f8"),
        ("noiseless_ambiguous", "?"), ("noisy_ambiguous", "?"),
        ("noiseless_alt_minima", "i4"), ("noisy_alt_minima", "i4"),
        ("noiseless_theta_bound", "?"), ("noisy_theta_bound", "?"),
        ("noiseless_h_bound", "?"), ("noisy_h_bound", "?"),
    ])
    result_rows = np.empty(len(samples), dtype=dtype)
    for i, sample in enumerate(samples, 1):
        y_true = sample["y_true"]
        noise = rng.normal(0.0, EVAL_NOISE_MM, size=3)
        y_noisy = y_true + noise
        fit0 = base.solve_group(y_true, coeffs, orders, templates)
        fitn = base.solve_group(y_noisy, coeffs, orders, templates)
        result_rows[i - 1] = (
            sample["group"], sample["h_true"], sample["ade_true"], sample["theta_true_deg"],
            *y_true, *noise, *y_noisy,
            fit0["theta_deg"], fit0["h"], fitn["theta_deg"], fitn["h"],
            fit0["chi2"], fitn["chi2"], fit0["ambiguous_multistart"], fitn["ambiguous_multistart"],
            fit0["alt_minima_count"], fitn["alt_minima_count"],
            fit0["theta_bound_hit"], fitn["theta_bound_hit"],
            fit0["h_bound_hit"], fitn["h_bound_hit"],
        )
        if i == 1 or i % 100 == 0 or i == len(samples):
            print(f"Inverse sample trials: {i}/{len(samples)}", flush=True)
    return result_rows


def error_metrics(rows: np.ndarray, group: str, noisy: bool) -> dict[str, float | int]:
    use = rows["sample_group"] == group
    subset = rows[use]
    theta_key = "theta_est_noisy_deg" if noisy else "theta_est_noiseless_deg"
    h_key = "h_est_noisy_mm" if noisy else "h_est_noiseless_mm"
    theta_error = subset[theta_key] - subset["theta_true_deg"]
    h_error = subset[h_key] - subset["h_true_mm"]
    return {
        "n": int(subset.size),
        "theta_bias_deg": float(np.mean(theta_error)),
        "theta_rmse_deg": float(np.sqrt(np.mean(theta_error**2))),
        "theta_p95_abs_deg": float(np.quantile(np.abs(theta_error), 0.95)),
        "theta_max_abs_deg": float(np.max(np.abs(theta_error))),
        "h_bias_mm": float(np.mean(h_error)),
        "h_rmse_mm": float(np.sqrt(np.mean(h_error**2))),
        "h_p95_abs_mm": float(np.quantile(np.abs(h_error), 0.95)),
        "h_max_abs_mm": float(np.max(np.abs(h_error))),
        "ambiguous": int(np.count_nonzero(subset["noisy_ambiguous" if noisy else "noiseless_ambiguous"])),
        "theta_bound_hits": int(np.count_nonzero(subset["noisy_theta_bound" if noisy else "noiseless_theta_bound"])),
        "h_bound_hits": int(np.count_nonzero(subset["noisy_h_bound" if noisy else "noiseless_h_bound"])),
    }


def binned_metric(rows: np.ndarray, group: str, noisy: bool, param: str, metric: str,
                  edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    subset = rows[rows["sample_group"] == group]
    theta = subset["theta_true_deg"]
    est_key = "theta_est_noisy_deg" if noisy else "theta_est_noiseless_deg"
    h_key = "h_est_noisy_mm" if noisy else "h_est_noiseless_mm"
    if param == "theta":
        error = subset[est_key] - theta
    else:
        error = subset[h_key] - subset["h_true_mm"]
    values = np.full(edges.size - 1, np.nan)
    counts = np.zeros(edges.size - 1, dtype=int)
    for i in range(edges.size - 1):
        mask = (theta >= edges[i]) & (theta < edges[i + 1] if i < edges.size - 2 else theta <= edges[i + 1])
        if np.any(mask):
            counts[i] = np.count_nonzero(mask)
            if metric == "rmse":
                values[i] = np.sqrt(np.mean(error[mask] ** 2))
            else:
                values[i] = np.quantile(np.abs(error[mask]), 0.95)
    return values, counts


def make_figures(rows: np.ndarray, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    ax = axes[0]
    for group in SAMPLE_GROUPS:
        subset = rows[rows["sample_group"] == group]
        ax.scatter(subset["theta_true_deg"], subset["theta_est_noiseless_deg"],
                   s=8, alpha=0.18, color=GROUP_COLORS[group], label=f"{GROUP_LABELS[group]} (noiseless)")
        ax.scatter(subset["theta_true_deg"], subset["theta_est_noisy_deg"],
                   s=8, alpha=0.18, color=GROUP_COLORS[group], marker="x", label=f"{GROUP_LABELS[group]} (1 um noise)")
    ax.plot([-20, 20], [-20, 20], "k--", lw=1, label="identity")
    ax.set(xlabel="True physical eye rotation theta (deg)", ylabel="Estimated theta (deg)",
           title="Eye rotation estimate")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="upper left")
    ax = axes[1]
    for group in SAMPLE_GROUPS:
        subset = rows[rows["sample_group"] == group]
        ax.scatter(subset["h_true_mm"], subset["h_est_noiseless_mm"],
                   s=8, alpha=0.18, color=GROUP_COLORS[group], label=f"{GROUP_LABELS[group]} (noiseless)")
        ax.scatter(subset["h_true_mm"], subset["h_est_noisy_mm"],
                   s=8, alpha=0.18, color=GROUP_COLORS[group], marker="x", label=f"{GROUP_LABELS[group]} (1 um noise)")
    ax.plot([-1, 1], [-1, 1], "k--", lw=1, label="identity")
    ax.set(xlabel="True RC Y translation h (mm)", ylabel="Estimated h (mm)", title="Head translation estimate")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    p = output_dir / "p1_sigma_depth_random_estimates.svg"
    fig.savefig(p, format="svg")
    plt.close(fig)
    outputs.append(p)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for group in SAMPLE_GROUPS:
        subset = rows[rows["sample_group"] == group]
        color = GROUP_COLORS[group]
        axes[0].scatter(subset["theta_true_deg"], subset["theta_est_noiseless_deg"] - subset["theta_true_deg"],
                        s=8, alpha=0.18, color=color, label=f"{GROUP_LABELS[group]} (noiseless)")
        axes[0].scatter(subset["theta_true_deg"], subset["theta_est_noisy_deg"] - subset["theta_true_deg"],
                        s=8, alpha=0.18, color=color, marker="x", label=f"{GROUP_LABELS[group]} (1 um noise)")
        axes[1].scatter(subset["theta_true_deg"], subset["h_est_noiseless_mm"] - subset["h_true_mm"],
                        s=8, alpha=0.18, color=color, label=f"{GROUP_LABELS[group]} (noiseless)")
        axes[1].scatter(subset["theta_true_deg"], subset["h_est_noisy_mm"] - subset["h_true_mm"],
                        s=8, alpha=0.18, color=color, marker="x", label=f"{GROUP_LABELS[group]} (1 um noise)")
    axes[0].axhline(0, color="black", ls="--", lw=1)
    axes[1].axhline(0, color="black", ls="--", lw=1)
    axes[0].set(xlabel="True physical eye rotation theta (deg)", ylabel="Theta estimation error (deg)", title="Theta error")
    axes[1].set(xlabel="True physical eye rotation theta (deg)", ylabel="h estimation error (mm)", title="h error")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    p = output_dir / "p1_sigma_depth_random_errors.svg"
    fig.savefig(p, format="svg")
    plt.close(fig)
    outputs.append(p)

    edges = np.linspace(-20.0, 20.0, 21)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)
    specs = (("theta", "rmse", "Theta RMSE (deg)"), ("theta", "p95", "Theta p95 absolute error (deg)"),
             ("h", "rmse", "h RMSE (mm)"), ("h", "p95", "h p95 absolute error (mm)"))
    for ax, (param, metric, title) in zip(axes.flat, specs):
        for group in SAMPLE_GROUPS:
            for noisy, linestyle in ((False, "-"), (True, "--")):
                vals, counts = binned_metric(rows, group, noisy, param, metric, edges)
                label = f"{GROUP_LABELS[group]} ({'1 um noise' if noisy else 'noiseless'})"
                ax.plot(centers, vals, color=GROUP_COLORS[group], linestyle=linestyle, lw=1.6,
                        marker="o", ms=3, label=label)
        ax.axvline(0.0, color="black", lw=0.8, alpha=0.5)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    for ax in axes[1, :]:
        ax.set_xlabel("True physical eye rotation theta (deg)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Error metric")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    p = output_dir / "p1_sigma_depth_random_binned_errors.svg"
    fig.savefig(p, format="svg")
    plt.close(fig)
    outputs.append(p)
    return outputs


def make_report(best: dict, candidates: list[dict], train_h: np.ndarray, validation_h: np.ndarray,
                test_h: np.ndarray, baseline_fit: dict, forward: dict, rows: np.ndarray,
                figures: list[Path]) -> str:
    lines = [
        "# Random-measurement theta and translation evaluation",
        "",
        f"Input: {INPUT}",
        "",
        "## Sampling and calibration",
        "",
        f"The raw simulator grid contains 201 h slices, 401 angles, and three zooms (241,803 observations). Sorted even-index h slices form the training pool; odd-index h slices form the untouched test pool. h=0 and both h endpoints remain in training. A deterministic subset of {len(validation_h)} interior nonzero even h slices is used only to choose residual polynomial order. The final coefficients are refit on all even h slices before sampling test measurements.",
        "",
        "For each zoom, the baseline b_i(theta) is calibrated at h=0 with a degree-6 polynomial in u=theta/20 deg. Translation residuals are then fit separately as h*q_i(u)+h^2*r_i(u) from nonzero training h slices. Thus h=0 identifies b_i only; the translation response q/r is learned from simulator data at h != 0. This is simulator calibration, not an h=0-only estimate of translation response.",
        "",
        f"Training h slices: {len(train_h)}; validation h slices: {len(validation_h)}; held-out test h slices: {len(test_h)}. Validation h values: {', '.join(f'{x:+.2f}' for x in validation_h)}.",
        "",
        "### Residual-order validation",
        "",
        "| q degree | r degree | validation RMSE (um) | validation max abs (um) | z1 RMSE | z2 RMSE | z3 RMSE |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in candidates:
        v = item["validation_overall"]
        by = item["validation_by_zoom"]
        lines.append(f"| {item['q_degree']} | {item['r_degree']} | {v['rmse_um']:.6g} | {v['max_abs_um']:.6g} | {by[1]['rmse_um']:.6g} | {by[2]['rmse_um']:.6g} | {by[3]['rmse_um']:.6g} |")
    lines += [
        "",
        f"Selected residual order: q={best['q_degree']}, r={best['r_degree']}. Held-out odd-h forward prediction: RMSE={forward['overall']['rmse_um']:.6g} um, p95 absolute={forward['overall']['p95_abs_um']:.6g} um, max absolute={forward['overall']['max_abs_um']:.6g} um.",
        "",
        "h=0 baseline fit RMSE by zoom (um): " + ", ".join(f"z{z}={baseline_fit[z]['rmse_um']:.6g}" for z in base.ZOOMS) + ".",
        "",
        "## Monte Carlo measurement evaluation",
        "",
        f"Samples: {N_UNIFORM} uniformly selected held-out (h,angle) states; {N_NEAR_ZERO} additional held-out states with |theta|<=1 deg; and {N_THETA0_DRAWS_PER_H} exact-theta-zero Gaussian draws at each h in (-1,0,+1) mm. This is {rows.size} trials total. Each trial observes all three zooms together. A separate independent Gaussian draw with sigma={EVAL_NOISE_MM*1000:g} um is added to each zoom channel. No truth values are used to seed or constrain the inverse; bounded multistart is initialized from model templates only.",
        "",
        "| Sample group | Noise | n | theta RMSE (deg) | theta p95 abs (deg) | h RMSE (mm) | h p95 abs (mm) | ambiguous | theta-bound hits | h-bound hits |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in SAMPLE_GROUPS:
        for noisy in (False, True):
            met = error_metrics(rows, group, noisy)
            lines.append(f"| {GROUP_LABELS[group]} | {'1 um iid' if noisy else 'noise-free'} | {met['n']} | {met['theta_rmse_deg']:.6g} | {met['theta_p95_abs_deg']:.6g} | {met['h_rmse_mm']:.6g} | {met['h_p95_abs_mm']:.6g} | {met['ambiguous']} | {met['theta_bound_hits']} | {met['h_bound_hits']} |")
    lines += [
        "",
        "The 1 um Gaussian noise is explicitly injected for this requested simulation evaluation. Its value is a scenario assumption, not a claim about measured detector noise. Held-out samples test unseen h slices but reuse the same angles as calibration; the exact-theta-zero stress draws include h values used in training and are reported separately. The binned plots use fixed 2-degree bins and display RMSE and p95 absolute error separately.",
        "",
        "## Outputs",
        "",
        f"- Full coefficients, settings, forward metrics, and all truth/observation/estimate rows: {OUT_DIR / 'p1_sigma_depth_random_measurements.pkl'}",
    ]
    lines.extend(f"- {p.name}" for p in figures)
    return "\n".join(lines) + "\n"


def main() -> None:
    with INPUT.open("rb") as stream:
        raw = pickle.load(stream)
    h_values, ade_values, _ = validate_data(raw)
    data = add_true_theta(raw)
    train_h_idx, validation_h_idx, test_h_idx = validation_indices(h_values)
    h0_index = int(np.argmin(np.abs(h_values)))
    if not np.isclose(h_values[h0_index], 0.0, atol=1e-12, rtol=0.0):
        raise RuntimeError("The raw h grid must include exact h=0")
    baseline = fit_baseline_h0(data, float(h_values[h0_index]))
    baseline_fit = {}
    for z in base.ZOOMS:
        rows = (data["zoom_number"] == z) & np.isclose(data["h_translation_mm"], 0.0, atol=1e-12, rtol=0.0)
        u = data["true_theta_rad"][rows] / base.THETA_SCALE_RAD
        pred = np.polynomial.polynomial.polyval(u, baseline[z])
        baseline_fit[z] = forward_metrics(pred - data["real_y_si_mm"][rows])
    best, candidate_results = pick_model(data, baseline, h_values, train_h_idx, validation_h_idx)
    final_train_mask = np.zeros(h_values.size, dtype=bool)
    final_train_mask[train_h_idx] = True
    final_coeffs, orders = make_coefficients(
        data, baseline, final_train_mask, h_values, best["q_degree"], best["r_degree"]
    )

    h_index = np.searchsorted(h_values, data["h_translation_mm"])
    test_rows = np.isin(h_index, test_h_idx)
    prediction = predict_rows(data, final_coeffs, orders)
    forward_error = prediction[test_rows] - data["real_y_si_mm"][test_rows]
    forward = {
        "overall": forward_metrics(forward_error),
        "by_zoom": {
            z: forward_metrics(forward_error[data["zoom_number"][test_rows] == z])
            for z in base.ZOOMS
        },
    }

    rng = np.random.default_rng(RNG_SEED)
    samples, _ = make_samples(data, h_values, ade_values, test_h_idx, rng)
    rows = run_inversions(samples, final_coeffs, orders, rng)
    figures = make_figures(rows, OUT_DIR)
    report = make_report(best, candidate_results, h_values[train_h_idx], h_values[validation_h_idx],
                         h_values[test_h_idx], baseline_fit, forward, rows, figures)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "input": str(INPUT),
        "rng_seed": RNG_SEED,
        "assumed_and_injected_noise_sigma_mm_per_zoom": EVAL_NOISE_MM,
        "theta_convention": "theta_deg = -rotation_deg (RC ADE); theta_rad = -deg2rad(ADE)",
        "h_split": {"train_h_mm": h_values[train_h_idx], "validation_h_mm": h_values[validation_h_idx],
                    "test_h_mm": h_values[test_h_idx]},
        "baseline_order": THETA_CAL_ORDER,
        "selected_q_degree": best["q_degree"],
        "selected_r_degree": best["r_degree"],
        "orders_b_q_r": orders,
        "baseline_coefficients_h0": baseline,
        "final_coefficients": final_coeffs,
        "validation_candidates": [
            {k: v for k, v in item.items() if k != "coefficients"} for item in candidate_results
        ],
        "heldout_forward_metrics": forward,
        "h0_baseline_fit_metrics": baseline_fit,
        "measurement_trials": rows,
        "trial_error_metrics": {
            (group, noisy): error_metrics(rows, group, noisy)
            for group in SAMPLE_GROUPS for noisy in (False, True)
        },
        "figures": [str(p) for p in figures],
        "interpretation": (
            "h=0 calibrates b_i(theta) only; q_i/r_i translation residual is fitted on nonzero training h slices. "
            "No truth values initialize/constrain estimates. The 1 um Gaussian is injected only for the noisy scenario."
        ),
    }
    result_path = OUT_DIR / "p1_sigma_depth_random_measurements.pkl"
    with result_path.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    report_path = OUT_DIR / "p1_sigma_depth_random_measurements_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Selected q/r order: {best['q_degree']}/{best['r_degree']}", flush=True)
    print(f"Heldout forward RMSE: {forward['overall']['rmse_um']:.6g} um", flush=True)
    for group in SAMPLE_GROUPS:
        for noisy in (False, True):
            met = error_metrics(rows, group, noisy)
            print(f"{group} {'noisy' if noisy else 'noise-free'}: n={met['n']} thetaRMSE={met['theta_rmse_deg']:.6g}deg hRMSE={met['h_rmse_mm']:.6g}mm amb={met['ambiguous']}", flush=True)
    print(f"Saved {result_path}", flush=True)
    print(f"Saved {report_path}", flush=True)
    for path in figures:
        print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()


