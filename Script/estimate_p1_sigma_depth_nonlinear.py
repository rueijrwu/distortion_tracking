"""Fit and evaluate a translation-dependent nonlinear model for P1 sigma data."""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.optimize import least_squares, minimize_scalar


ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "p1_only_sigma_depth_raytracing" / "p1_only_sigma_depth_raytracing.pkl"
OUT_DIR = ROOT / "data" / "p1_only_sigma_depth_nonlinear_estimation"
NOISE_SIGMA_MM = 0.001
THETA_MAX_DEG = 20.0
THETA_SCALE_RAD = np.deg2rad(THETA_MAX_DEG)
H_BOUNDS = (-1.0, 1.0)
BASE_CANDIDATES = (
    (2, 1, 0),
    (3, 2, 0),
    (3, 2, 1),
    (4, 3, 2),
)
HIGHER_CANDIDATES = ((5, 3, 2), (6, 4, 2))
N_FOLDS = 5
ZOOMS = (1, 2, 3)
H_VALUES = (-1.0, 0.0, 1.0)
EXPECTED_ZOOM_METADATA = {
    1: (0.0, 1.0e13, 50.0, -15.0, 0.0),
    2: (0.0, 0.0, 50.0, 0.0, 5.0),
    3: (0.0, 0.0, 25.0, 0.0, -5.0),
}
ZOOM_METADATA_FIELDS = (
    "field_yob_mm", "object_thi_s0_mm", "s1_thickness_mm", "s1_ade_deg", "s1_yde_mm",
)
H_COLORS = {-1.0: "#1f77b4", 0.0: "#555555", 1.0: "#d95f02"}
H_STYLES = {-1.0: "--", 0.0: ":", 1.0: "-"}
ZOOM_COLORS = {1: "#1f77b4", 2: "#d95f02", 3: "#2ca02c"}


def design_matrix(u: np.ndarray, h: np.ndarray, orders: tuple[int, int, int]) -> np.ndarray:
    db, dq, dr = orders
    cols = [u**j for j in range(db + 1)]
    cols += [h * u**j for j in range(dq + 1)]
    cols += [h**2 * u**j for j in range(dr + 1)]
    return np.column_stack(cols)


def split_coefficients(coeff: np.ndarray, orders: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    db, dq, dr = orders
    ib = db + 1
    iq = ib + dq + 1
    return coeff[:ib], coeff[ib:iq], coeff[iq:iq + dr + 1]


def poly_value(c: np.ndarray, u: float) -> float:
    return float(np.polynomial.polynomial.polyval(u, c))


def poly_derivative(c: np.ndarray, u: float) -> float:
    if c.size < 2:
        return 0.0
    dc = np.arange(1, c.size, dtype=float) * c[1:]
    return float(np.polynomial.polynomial.polyval(u, dc))


def predict_one(coeff: np.ndarray, orders: tuple[int, int, int], u: float, h: float) -> float:
    b, q, r = split_coefficients(coeff, orders)
    return poly_value(b, u) + poly_value(q, u) * h + poly_value(r, u) * h * h


def predict_vector(
    coeffs: dict[int, np.ndarray], orders: tuple[int, int, int], u: float, h: float
) -> np.ndarray:
    return np.array([predict_one(coeffs[z], orders, u, h) for z in ZOOMS], dtype=float)


def jacobian_vector(
    coeffs: dict[int, np.ndarray], orders: tuple[int, int, int], u: float, h: float
) -> np.ndarray:
    rows = []
    for z in ZOOMS:
        b, q, r = split_coefficients(coeffs[z], orders)
        dy_du = poly_derivative(b, u) + h * poly_derivative(q, u) + h * h * poly_derivative(r, u)
        dy_dh = poly_value(q, u) + 2.0 * h * poly_value(r, u)
        rows.append((dy_du, dy_dh))
    return np.asarray(rows, dtype=float)


def fit_zoom(data: np.ndarray, zoom: int, orders: tuple[int, int, int], train_mask: np.ndarray) -> np.ndarray:
    use = (data["zoom_number"] == zoom) & train_mask
    u = data["true_theta_rad"][use] / THETA_SCALE_RAD
    h = data["h_translation_mm"][use]
    y = data["real_y_si_mm"][use]
    x = design_matrix(u, h, orders)
    coeff, *_ = np.linalg.lstsq(x, y, rcond=None)
    return coeff


def fit_all(data: np.ndarray, orders: tuple[int, int, int], train_mask: np.ndarray) -> dict[int, np.ndarray]:
    return {z: fit_zoom(data, z, orders, train_mask) for z in ZOOMS}


def fold_assignments(data: np.ndarray) -> tuple[dict[float, int], np.ndarray]:
    angles = np.unique(data["rotation_deg"])
    folds = np.array_split(np.arange(angles.size), N_FOLDS)
    angle_to_fold: dict[float, int] = {}
    for fold, indices in enumerate(folds):
        for i in indices:
            angle_to_fold[float(angles[i])] = fold
    row_fold = np.array([angle_to_fold[float(a)] for a in data["rotation_deg"]], dtype=np.int16)
    return angle_to_fold, row_fold


def evaluate_candidate(
    data: np.ndarray,
    orders: tuple[int, int, int],
    row_fold: np.ndarray,
) -> dict:
    residuals = np.full(data.size, np.nan, dtype=float)
    fold_coeffs: dict[int, dict[int, np.ndarray]] = {}
    fold_metrics = []
    weak_projections = []
    weak_singular_values = []
    state_groups: dict[tuple[float, float], dict[int, int]] = {}
    for idx, row in enumerate(data):
        key = (float(row["h_translation_mm"]), float(row["rotation_deg"]))
        state_groups.setdefault(key, {})[int(row["zoom_number"])] = idx
    for fold in range(N_FOLDS):
        train = row_fold != fold
        test = row_fold == fold
        coeffs = fit_all(data, orders, train)
        fold_coeffs[fold] = coeffs
        pred = np.full(np.count_nonzero(test), np.nan, dtype=float)
        test_indices = np.flatnonzero(test)
        for z in ZOOMS:
            take = data["zoom_number"][test_indices] == z
            idx = test_indices[take]
            u = data["true_theta_rad"][idx] / THETA_SCALE_RAD
            h = data["h_translation_mm"][idx]
            yhat = np.array([predict_one(coeffs[z], orders, float(ui), float(hi)) for ui, hi in zip(u, h)])
            residuals[idx] = data["real_y_si_mm"][idx] - yhat
        for (h, ade), zoom_rows in state_groups.items():
            if int(row_fold[zoom_rows[1]]) != fold:
                continue
            observed = np.array([data[zoom_rows[z]]["real_y_si_mm"] for z in ZOOMS])
            predicted = np.array([
                predict_one(coeffs[z], orders,
                            float(data[zoom_rows[z]]["true_theta_rad"] / THETA_SCALE_RAD), h)
                for z in ZOOMS
            ])
            u = float(-np.deg2rad(ade) / THETA_SCALE_RAD)
            jac = jacobian_vector(coeffs, orders, u, h)
            left, singular, _ = np.linalg.svd(jac, full_matrices=False)
            weak_projections.append(float((observed - predicted) @ left[:, -1]))
            weak_singular_values.append((abs(ade), float(singular[-1])))
        fold_metrics.append({
            "fold": fold,
            "n": int(np.count_nonzero(test)),
            "rmse_mm": float(np.sqrt(np.mean(residuals[test] ** 2))),
        })
    per_zoom = {}
    for z in ZOOMS:
        err = residuals[data["zoom_number"] == z]
        per_zoom[z] = float(np.sqrt(np.mean(err**2)))
    weak_projections = np.asarray(weak_projections, dtype=float)
    near_zero_s = np.asarray([value for ade, value in weak_singular_values if ade <= 1.0])
    return {
        "orders": orders,
        "cv_rmse_mm": float(np.sqrt(np.mean(residuals**2))),
        "cv_rmse_um": float(1000.0 * np.sqrt(np.mean(residuals**2))),
        "cv_rmse_um_by_zoom": {z: 1000.0 * per_zoom[z] for z in ZOOMS},
        "cv_rmse_um_by_fold": [1000.0 * x["rmse_mm"] for x in fold_metrics],
        "weak_projection_cv_rmse_um": float(1000.0 * np.sqrt(np.mean(weak_projections**2))),
        "weak_projection_cv_p95_um": float(1000.0 * np.quantile(np.abs(weak_projections), 0.95)),
        "weak_singular_median_mm_per_scaled_unit": float(np.median([x[1] for x in weak_singular_values])),
        "near_zero_weak_singular_median_mm_per_scaled_unit": float(np.median(near_zero_s)),
        "near_zero_noise_over_singular_median_scaled": float(NOISE_SIGMA_MM / np.median(near_zero_s)),
        "cv_residuals_mm": residuals,
        "fold_coeffs": fold_coeffs,
    }


def fit_selection(data: np.ndarray, row_fold: np.ndarray) -> tuple[tuple[int, int, int], list[dict], dict]:
    candidates = list(BASE_CANDIDATES) + list(HIGHER_CANDIDATES)
    results = [evaluate_candidate(data, order, row_fold) for order in candidates]
    best_index = int(np.argmin([item["cv_rmse_mm"] for item in results]))
    return results[best_index]["orders"], results, results[best_index]


def prepare_templates(
    coeffs: dict[int, np.ndarray], orders: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u_grid = np.linspace(-1.0, 1.0, 81)
    h_grid = np.linspace(H_BOUNDS[0], H_BOUNDS[1], 9)
    uu, hh = np.meshgrid(u_grid, h_grid, indexing="ij")
    points = np.column_stack((uu.ravel(), hh.ravel()))
    pred = np.vstack([predict_vector(coeffs, orders, float(u), float(h)) for u, h in points])
    return points, pred, np.array([u_grid.size, h_grid.size], dtype=int)


def parameter_covariance(
    coeffs: dict[int, np.ndarray], orders: tuple[int, int, int], u: float, h: float
) -> dict[str, float | np.ndarray]:
    jac = jacobian_vector(coeffs, orders, u, h)
    _, singular, vt = np.linalg.svd(jac, full_matrices=False)
    if singular.size < 2 or singular[-1] <= np.finfo(float).eps * singular[0]:
        return {
            "condition_number": float("inf"),
            "singular_min": float(singular[-1]) if singular.size else 0.0,
            "singular_max": float(singular[0]) if singular.size else 0.0,
            "sigma_u": float("inf"),
            "sigma_h": float("inf"),
            "weak_sigma_scaled": float("inf"),
            "weak_vector": vt[-1] if singular.size else np.array([np.nan, np.nan]),
        }
    cov = vt.T @ np.diag((NOISE_SIGMA_MM / singular) ** 2) @ vt
    return {
        "condition_number": float(singular[0] / singular[-1]),
        "singular_min": float(singular[-1]),
        "singular_max": float(singular[0]),
        "sigma_u": float(np.sqrt(max(cov[0, 0], 0.0))),
        "sigma_h": float(np.sqrt(max(cov[1, 1], 0.0))),
        "weak_sigma_scaled": float(NOISE_SIGMA_MM / singular[-1]),
        "weak_vector": vt[-1].copy(),
    }


RESULT_DTYPE = np.dtype([
    ("h_true_mm", "f8"),
    ("true_ADE_deg", "f8"),
    ("true_theta_rad", "f8"),
    ("true_theta_deg", "f8"),
    ("theta_est_rad", "f8"),
    ("theta_est_deg", "f8"),
    ("ADE_est_deg", "f8"),
    ("h_est_mm", "f8"),
    ("residual_rms_mm", "f8"),
    ("chi2", "f8"),
    ("nfev_total", "i4"),
    ("solver_success", "?"),
    ("solver_retry_count", "i4"),
    ("theta_bound_hit", "?"),
    ("h_bound_hit", "?"),
    ("ambiguous_multistart", "?"),
    ("alt_minima_count", "i4"),
    ("alt_theta_spread_deg", "f8"),
    ("alt_h_spread_mm", "f8"),
    ("condition_number_scaled", "f8"),
    ("singular_min_mm", "f8"),
    ("singular_max_mm", "f8"),
    ("sigma_theta_deg_1um", "f8"),
    ("sigma_h_mm_1um", "f8"),
    ("weak_sigma_scaled_1um", "f8"),
    ("weak_v_u", "f8"),
    ("weak_v_h", "f8"),
])


def solve_group(
    observed: np.ndarray,
    coeffs: dict[int, np.ndarray],
    orders: tuple[int, int, int],
    templates: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, float | int | bool]:
    points, pred_grid, _ = templates
    chi_grid = np.sum(((pred_grid - observed[None, :]) / NOISE_SIGMA_MM) ** 2, axis=1)
    nearest = np.argsort(chi_grid)[:2]
    seeds = [points[int(i)].copy() for i in nearest]
    seeds += [np.array([-1.0, -1.0]), np.array([-1.0, 1.0]),
              np.array([1.0, -1.0]), np.array([1.0, 1.0]),
              np.array([0.0, 0.0])]
    lower = np.array([-1.0, H_BOUNDS[0]])
    upper = np.array([1.0, H_BOUNDS[1]])

    def residual(x: np.ndarray) -> np.ndarray:
        return (predict_vector(coeffs, orders, float(x[0]), float(x[1])) - observed) / NOISE_SIGMA_MM

    def jac(x: np.ndarray) -> np.ndarray:
        return jacobian_vector(coeffs, orders, float(x[0]), float(x[1])) / NOISE_SIGMA_MM

    minima = []
    unfinished = []
    nfev_total = 0
    for seed in seeds:
        result = least_squares(
            residual, np.clip(seed, lower, upper), jac=jac,
            bounds=(lower, upper), max_nfev=80, ftol=1e-11, xtol=1e-11, gtol=1e-11,
        )
        nfev_total += int(result.nfev)
        if result.success and np.all(np.isfinite(result.x)) and np.isfinite(result.cost):
            minima.append((float(2.0 * result.cost), result.x.copy()))
        elif np.all(np.isfinite(result.x)):
            unfinished.append(result.x.copy())
    retry_count = 0
    for seed in unfinished:
        result = least_squares(
            residual, np.clip(seed, lower, upper), jac=jac,
            bounds=(lower, upper), max_nfev=500, ftol=1e-11, xtol=1e-11, gtol=1e-11,
        )
        retry_count += 1
        nfev_total += int(result.nfev)
        if result.success and np.all(np.isfinite(result.x)) and np.isfinite(result.cost):
            minima.append((float(2.0 * result.cost), result.x.copy()))
    if not minima:
        raise RuntimeError("All bounded multistart inverse solutions failed")
    minima.sort(key=lambda item: item[0])
    best_chi, best_x = minima[0]
    distinct = []
    for chi, x in minima:
        if chi <= best_chi + 5.99 and not any(np.linalg.norm(x - old) < 1e-4 for old in distinct):
            distinct.append(x)
    spread = np.ptp(np.vstack(distinct), axis=0) if len(distinct) > 1 else np.zeros(2)
    ambiguity = bool(len(distinct) > 1 and np.linalg.norm(spread) > 0.05)

    u, h_est = map(float, best_x)
    cov = parameter_covariance(coeffs, orders, u, h_est)
    pred = predict_vector(coeffs, orders, u, h_est)
    resid = pred - observed
    true_theta_rad = float("nan")
    return {
        "u": u,
        "h": h_est,
        "theta_rad": u * THETA_SCALE_RAD,
        "theta_deg": u * THETA_MAX_DEG,
        "ADE_deg": -u * THETA_MAX_DEG,
        "residual_rms_mm": float(np.sqrt(np.mean(resid**2))),
        "chi2": float(np.sum((resid / NOISE_SIGMA_MM) ** 2)),
        "nfev_total": nfev_total,
        "solver_success": True,
        "solver_retry_count": retry_count,
        "theta_bound_hit": bool(abs(abs(u) - 1.0) < 1e-4),
        "h_bound_hit": bool(abs(abs(h_est) - 1.0) < 1e-4),
        "ambiguous_multistart": ambiguity,
        "alt_minima_count": len(distinct),
        "alt_theta_spread_deg": float(spread[0] * THETA_MAX_DEG),
        "alt_h_spread_mm": float(spread[1]),
        "condition_number_scaled": float(cov["condition_number"]),
        "singular_min_mm": float(cov["singular_min"]),
        "singular_max_mm": float(cov["singular_max"]),
        "sigma_theta_deg_1um": float(cov["sigma_u"] * THETA_MAX_DEG),
        "sigma_h_mm_1um": float(cov["sigma_h"]),
        "weak_sigma_scaled_1um": float(cov["weak_sigma_scaled"]),
        "weak_v_u": float(np.asarray(cov["weak_vector"])[0]),
        "weak_v_h": float(np.asarray(cov["weak_vector"])[1]),
    }


def inverse_dataset(
    data: np.ndarray,
    orders: tuple[int, int, int],
    full_coeffs: dict[int, np.ndarray],
    row_fold: np.ndarray,
    fold_coeffs: dict[int, dict[int, np.ndarray]],
    blocked: bool,
) -> np.ndarray:
    group_rows: dict[tuple[float, float], dict[int, int]] = {}
    for idx, row in enumerate(data):
        key = (float(row["h_translation_mm"]), float(row["rotation_deg"]))
        group_rows.setdefault(key, {})[int(row["zoom_number"])] = idx
    if any(set(v) != set(ZOOMS) for v in group_rows.values()):
        raise ValueError("Every h/rotation point must have all three zoom observations")
    templates_by_fold = {}
    coefficient_by_fold = {}
    for fold in range(N_FOLDS):
        coeffs = fold_coeffs[fold] if blocked else full_coeffs
        coefficient_by_fold[fold] = coeffs
        templates_by_fold[fold] = prepare_templates(coeffs, orders)
    rows = []
    for (h_true, ade), zoom_index in sorted(group_rows.items()):
        theta_true = -np.deg2rad(ade)
        fold = int(row_fold[zoom_index[1]])
        coeffs = coefficient_by_fold[fold]
        observed = np.array([data[zoom_index[z]]["real_y_si_mm"] for z in ZOOMS], dtype=float)
        result = solve_group(observed, coeffs, orders, templates_by_fold[fold])
        rows.append((
            h_true, ade, theta_true, np.rad2deg(theta_true),
            result["theta_rad"], result["theta_deg"], result["ADE_deg"], result["h"],
            result["residual_rms_mm"], result["chi2"], result["nfev_total"],
            result["solver_success"], result["solver_retry_count"],
            result["theta_bound_hit"], result["h_bound_hit"], result["ambiguous_multistart"],
            result["alt_minima_count"], result["alt_theta_spread_deg"], result["alt_h_spread_mm"],
            result["condition_number_scaled"], result["singular_min_mm"], result["singular_max_mm"],
            result["sigma_theta_deg_1um"], result["sigma_h_mm_1um"], result["weak_sigma_scaled_1um"],
            result["weak_v_u"], result["weak_v_h"],
        ))
        if len(rows) % 100 == 0 or len(rows) == len(group_rows):
            label = "blocked holdout" if blocked else "same-sweep"
            print(f"Inverse {label}: {len(rows)}/{len(group_rows)} states", flush=True)
    return np.asarray(rows, dtype=RESULT_DTYPE)


def error_metrics(results: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    rows = results[mask]
    theta_err = rows["theta_est_deg"] - rows["true_theta_deg"]
    h_err = rows["h_est_mm"] - rows["h_true_mm"]
    return {
        "n": int(rows.size),
        "theta_bias_deg": float(np.mean(theta_err)),
        "theta_rmse_deg": float(np.sqrt(np.mean(theta_err**2))),
        "theta_max_abs_deg": float(np.max(np.abs(theta_err))),
        "h_bias_mm": float(np.mean(h_err)),
        "h_rmse_mm": float(np.sqrt(np.mean(h_err**2))),
        "h_max_abs_mm": float(np.max(np.abs(h_err))),
    }


def profile_examples(
    data: np.ndarray,
    results: np.ndarray,
    coeffs: dict[int, np.ndarray],
    orders: tuple[int, int, int],
) -> tuple[list[dict], list[dict]]:
    index_by_key = {
        (float(row["h_translation_mm"]), float(row["rotation_deg"]), int(row["zoom_number"])): i
        for i, row in enumerate(data)
    }
    estimate_by_key = {
        (float(row["h_true_mm"]), float(row["true_ADE_deg"])): row
        for row in results
    }
    requests = ((0.0, 0.0), (1.0, -10.0), (-1.0, 10.0))
    profile_records = []
    summaries = []
    for h, ade in requests:
        obs = np.array([
            data[index_by_key[(h, ade, z)]]['real_y_si_mm'] for z in ZOOMS
        ], dtype=float)
        estimate = estimate_by_key[(h, ade)]
        xhat = np.array([estimate["theta_est_deg"] / THETA_MAX_DEG, estimate["h_est_mm"]])
        j = jacobian_vector(coeffs, orders, float(xhat[0]), float(xhat[1]))
        _, singular, vt = np.linalg.svd(j, full_matrices=False)
        h_grid = np.linspace(H_BOUNDS[0], H_BOUNDS[1], 201)
        u_grid = np.linspace(-1.0, 1.0, 81)
        profile_chi = []
        profile_theta = []
        for h_profile in h_grid:
            chi_grid = np.array([
                np.sum(((predict_vector(coeffs, orders, float(u), float(h_profile)) - obs)
                        / NOISE_SIGMA_MM) ** 2)
                for u in u_grid
            ])
            candidates = []
            for idx in np.argsort(chi_grid)[:3]:
                left = u_grid[max(0, int(idx) - 1)]
                right = u_grid[min(u_grid.size - 1, int(idx) + 1)]
                if right > left:
                    opt = minimize_scalar(
                        lambda u: float(np.sum(((predict_vector(coeffs, orders, float(u), float(h_profile))
                                                - obs) / NOISE_SIGMA_MM) ** 2)),
                        bounds=(float(left), float(right)), method="bounded",
                        options={"xatol": 1e-11, "maxiter": 120},
                    )
                    candidates.append((float(opt.fun), float(opt.x)))
                candidates.append((float(chi_grid[idx]), float(u_grid[idx])))
            best_chi, best_u = min(candidates, key=lambda item: item[0])
            profile_chi.append(best_chi)
            profile_theta.append(best_u * THETA_MAX_DEG)
        profile_chi = np.asarray(profile_chi)
        profile_chi -= float(np.min(profile_chi))
        profile_theta = np.asarray(profile_theta)
        for h_profile, theta_profile, value in zip(h_grid, profile_theta, profile_chi):
            profile_records.append({
                "h_true_mm": h, "ADE_deg": ade,
                "theta_true_deg": -ade, "profile_h_mm": float(h_profile),
                "profile_h_offset_mm": float(h_profile - h),
                "profile_theta_deg": float(theta_profile),
                "feasible": True,
                "delta_chi2": float(value),
            })
        intervals = {}
        for label, cutoff in (("68", 1.0), ("95", 3.84)):
            inside = h_grid[profile_chi <= cutoff]
            intervals[label] = [float(inside.min()), float(inside.max())] if inside.size else [float("nan"), float("nan")]
        summaries.append({
            "h_true_mm": h, "ADE_deg": ade, "theta_true_deg": -ade,
            "singular_values_mm": singular.tolist(),
            "condition_number_scaled": float(singular[0] / singular[-1]),
            "weak_vector_u_h": vt[-1].tolist(),
            "profile_best_h_mm": float(h_grid[int(np.argmin(profile_chi))]),
            "profile_delta_chi2_at_true_h": float(np.interp(h, h_grid, profile_chi)),
            "profile_68_interval_mm_clipped_to_domain": intervals["68"],
            "profile_95_interval_mm_clipped_to_domain": intervals["95"],
            "profile_touches_domain_edge_68": bool(np.any(np.isclose(intervals["68"], H_BOUNDS[0])) or np.any(np.isclose(intervals["68"], H_BOUNDS[1]))),
            "profile_touches_domain_edge_95": bool(np.any(np.isclose(intervals["95"], H_BOUNDS[0])) or np.any(np.isclose(intervals["95"], H_BOUNDS[1]))),
        })
    return profile_records, summaries


def draw_figures(
    full_results: np.ndarray,
    cv_results: np.ndarray,
    candidate_results: list[dict],
    profile_records: list[dict],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    h_handles = [Line2D([0], [0], color=H_COLORS[h], linestyle="-", lw=1.8,
                         label=f"h = {h:+g} mm") for h in (-1.0, 1.0)]
    fit_handles = [
        Line2D([0], [0], color="#333333", linestyle="-", lw=1.8, label="Blocked-angle holdout"),
        Line2D([0], [0], color="#333333", linestyle="--", lw=1.2, label="Same-sweep fit"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for h in (-1.0, 1.0):
        for arr, style, width in ((cv_results, "-", 1.55), (full_results, "--", 1.0)):
            subset = arr[np.isclose(arr["h_true_mm"], h)]
            order = np.argsort(subset["true_theta_deg"])
            axes[0].plot(subset["true_theta_deg"][order], subset["theta_est_deg"][order],
                         color=H_COLORS[h], linestyle=style, lw=width)
            axes[1].plot(subset["true_theta_deg"][order], subset["h_est_mm"][order],
                         color=H_COLORS[h], linestyle=style, lw=width)
    limits = [-THETA_MAX_DEG, THETA_MAX_DEG]
    axes[0].plot(limits, limits, color="#555555", lw=1.0, linestyle=":", label="identity")
    axes[0].set(xlim=limits, xlabel="True eye rotation θ = −ADE (deg)",
                ylabel="Estimated eye rotation θ (deg)", title="Joint estimate of θ")
    axes[0].grid(True, color="#dddddd", linewidth=0.7)
    axes[1].set(xlim=limits, xlabel="True eye rotation θ = −ADE (deg)",
                ylabel="Estimated physical translation t (mm)", title="Joint estimate of t")
    for h in (-1.0, 1.0):
        axes[1].axhline(h, color=H_COLORS[h], linestyle=":", lw=0.9, alpha=0.65)
    axes[1].grid(True, color="#dddddd", linewidth=0.7)
    fig.tight_layout(rect=(0.0, 0.16, 1.0, 1.0))
    fig.legend(handles=h_handles + fit_handles, loc="lower center", ncol=4,
               fontsize=8, frameon=True)
    path = output_dir / "p1_only_sigma_depth_joint_estimates.svg"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    outputs.append(path)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), sharex=True)
    for h in (-1.0, 1.0):
        for arr, style in ((cv_results, "-"), (full_results, "--")):
            subset = arr[np.isclose(arr["h_true_mm"], h)]
            order = np.argsort(subset["true_theta_deg"])
            theta_error = subset["theta_est_deg"][order] - subset["true_theta_deg"][order]
            h_error = subset["h_est_mm"][order] - subset["h_true_mm"][order]
            axes[0, 0].plot(subset["true_theta_deg"][order], theta_error,
                            color=H_COLORS[h], linestyle=style, lw=1.2)
            axes[0, 1].plot(subset["true_theta_deg"][order], h_error,
                            color=H_COLORS[h], linestyle=style, lw=1.2)
        subset = cv_results[np.isclose(cv_results["h_true_mm"], h)]
        order = np.argsort(subset["true_theta_deg"])
        axes[1, 0].plot(subset["true_theta_deg"][order],
                        subset["sigma_theta_deg_1um"][order],
                        color=H_COLORS[h], linestyle="-", lw=1.4)
        axes[1, 1].plot(subset["true_theta_deg"][order],
                        subset["sigma_h_mm_1um"][order],
                        color=H_COLORS[h], linestyle="-", lw=1.4)
    axes[0, 0].axhline(0.0, color="#777777", lw=0.8)
    axes[0, 1].axhline(0.0, color="#777777", lw=0.8)
    axes[0, 0].set(ylabel="θ error (deg)", title="Estimate error")
    axes[0, 1].set(ylabel="t error (mm)", title="Estimate error")
    axes[1, 0].set(ylabel="Local 1 μm σθ (deg)", title="Linearized noise scale")
    axes[1, 1].set(ylabel="Local 1 μm σt (mm)", title="Linearized noise scale")
    for ax in axes.ravel():
        ax.set_xlim(*limits)
        ax.grid(True, color="#dddddd", linewidth=0.7)
    for ax in axes[1]:
        ax.set_xlabel("True eye rotation θ = −ADE (deg)")
    fig.tight_layout(rect=(0.0, 0.11, 1.0, 1.0))
    fig.legend(handles=h_handles + fit_handles, loc="lower center", ncol=4,
               fontsize=8, frameon=True)
    path = output_dir / "p1_only_sigma_depth_joint_errors_uncertainty.svg"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    outputs.append(path)

    fig, ax = plt.subplots(figsize=(9, 5.4))
    names = [f"{x['orders'][0]}/{x['orders'][1]}/{x['orders'][2]}" for x in candidate_results]
    xs = np.arange(len(names))
    width = 0.23
    for j, z in enumerate(ZOOMS):
        values = [x["cv_rmse_um_by_zoom"][z] for x in candidate_results]
        ax.bar(xs + (j - 1) * width, values, width, color=ZOOM_COLORS[z], label=f"z{z}")
    ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1.0,
               label="1 μm likelihood scale")
    ax.set_yscale("log")
    ax.set_xticks(xs, names)
    ax.set_xlabel("Polynomial degrees (b, q, r) in scaled θ")
    ax.set_ylabel("Blocked-angle CV forward RMSE (μm)")
    ax.set_title("Forward-model order selection")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.legend(frameon=True, ncol=2)
    fig.tight_layout()
    path = output_dir / "p1_only_sigma_depth_forward_model_cv.svg"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    outputs.append(path)

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    profile_keys = sorted({(x["h_true_mm"], x["ADE_deg"]) for x in profile_records})
    labels = []
    for h, ade in profile_keys:
        values = [x for x in profile_records if x["h_true_mm"] == h and x["ADE_deg"] == ade]
        values.sort(key=lambda x: x["profile_h_offset_mm"])
        ax.plot([x["profile_h_offset_mm"] for x in values],
                [np.log10(1.0 + x["delta_chi2"]) for x in values], lw=1.6,
                color=H_COLORS[h], linestyle=H_STYLES[h])
        labels.append(f"h={h:+g}, ADE={ade:+g}°")
    ax.axhline(np.log10(2.0), color="#555555", linestyle=":", lw=0.9, label="Δχ²=1")
    ax.axhline(np.log10(6.99), color="#777777", linestyle="--", lw=0.9, label="Δχ²=5.99")
    ax.axvline(0.0, color="#777777", linestyle=":", lw=0.8)
    ax.set_xlabel("Profiled translation offset from true h (mm)")
    ax.set_ylabel("log10(1 + profile Δχ²)")
    ax.set_title("Bounded h profiles near center and off axis")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.legend(labels + ["Δχ²=1", "Δχ²=5.99"], fontsize=8, frameon=True)
    fig.tight_layout()
    path = output_dir / "p1_only_sigma_depth_h_profiles.svg"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    outputs.append(path)
    return outputs


def build_report(
    orders: tuple[int, int, int],
    candidate_results: list[dict],
    full_coeffs: dict[int, np.ndarray],
    full_fit_rmse_um_by_zoom: dict[int, float],
    full_results: np.ndarray,
    cv_results: np.ndarray,
    profile_summaries: list[dict],
) -> str:
    lines = [
        "# P1 sigma-depth nonlinear forward model and inverse estimates",
        "",
        f"Input: {INPUT}",
        "",
        "## Conventions and scope",
        "",
        "The measured sweep stores RC ADE in degrees. Physical eye rotation is theta = -ADE*pi/180 radians; physical head translation is h_translation_mm. The sequence settings are z1: S0 THI 1e13 mm, S1 THI 50 mm, S1 ADE -15 deg, S1 YDE 0 mm; z2: S0 THI 0 mm, S1 THI 50 mm, S1 ADE 0 deg, S1 YDE +5 mm; z3: S0 THI 0 mm, S1 THI 25 mm, S1 ADE 0 deg, S1 YDE -5 mm. F1 YOB readback is 0 mm for all zooms. The forward model is fitted only on observed h levels -1, 0, +1 mm; interpolation in h is not independently validated.",
        "",
        "The 1 μm per-zoom value is an assumed likelihood/noise scale for local uncertainty calculations, not a measured instrument noise claim. Calibration and evaluation use the same optical sweep; only angle blocks are withheld, so these are not independent experimental validation.",
        "",
        "## Forward model and order selection",
        "",
        "For each zoom i, the candidate forward model is",
        "",
        "$$",
        "G_i(\\theta,t)=b_i(\\theta)+q_i(\\theta)t+r_i(\\theta)t^2.",
        "$$",
        "",
        "Each coefficient is a polynomial in scaled angle u=θ/(20° in radians), and fitting is ordinary least squares with equal 1 μm channel weights.",
        "",
        "Five contiguous angle folds were used. Every h value and every zoom at an angle was withheld together. Model order was selected by the lowest aggregate blocked-angle cross-validation RMSE. Because selection uses these same folds, the selected score is a model-selection statistic, not an independent final test. Orders are reported as (degree b, degree q, degree r).",
        "",
        "| Orders | CV RMSE (μm) | z1 | z2 | z3 | Weak-output projection RMSE (μm) | Projection p95 abs (μm) | Median near-zero smin (mm/scaled unit) | 1 μm / smin (scaled units) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in candidate_results:
        v = item["cv_rmse_um_by_zoom"]
        marker = " **selected**" if item["orders"] == orders else ""
        lines.append(
            f"| {item['orders'][0]}/{item['orders'][1]}/{item['orders'][2]}{marker} | "
            f"{item['cv_rmse_um']:.6g} | {v[1]:.6g} | {v[2]:.6g} | {v[3]:.6g} | "
            f"{item['weak_projection_cv_rmse_um']:.6g} | {item['weak_projection_cv_p95_um']:.6g} | "
            f"{item['near_zero_weak_singular_median_mm_per_scaled_unit']:.6g} | "
            f"{item['near_zero_noise_over_singular_median_scaled']:.6g} |"
        )
    lines += [
        "",
        "Selected full-sweep coefficients use powers of scaled angle u=θ/(20° in radians):",
        "",
        "| Object | b_j coefficients (mm) | q_j coefficients (mm/mm) | r_j coefficients (mm/mm²) |",
        "|---|---|---|---|",
    ]
    for z in ZOOMS:
        b, q, r = split_coefficients(full_coeffs[z], orders)
        fmt = lambda values: ", ".join(f"{float(x):.12g}" for x in values)
        lines.append(f"| z{z} | [{fmt(b)}] | [{fmt(q)}] | [{fmt(r)}] |")
    lines += [
        "",
        "Same-sweep forward prediction RMSE (μm): "
        + ", ".join(f"z{z}={full_fit_rmse_um_by_zoom[z]:.6g}" for z in ZOOMS),
        "",
        "## Inverse estimates",
        "",
        "The inverse solves the three-channel least-squares problem for θ and physical translation t within the calibrated θ∈[−20°, +20°], t∈[−1,+1] domain. Initialization uses only a coarse grid of forward predictions plus diverse boundary/center starts; it does not use the true ADE or t labels. Unsuccessful optimizer starts are retried from their last iterate with a larger evaluation budget; only converged finite solutions are eligible. Near-equal minima use Δχ²≤5.99 and parameter separation >0.05 in scaled coordinates.",
        "",
        "| Evaluation | True t (mm) | n | θ bias (deg) | θ RMSE (deg) | θ max abs (deg) | t bias (mm) | t RMSE (mm) | t max abs (mm) | θ-bound hits | t-bound hits | ambiguous |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, arr in (("Same-sweep fit", full_results), ("Blocked-angle holdout", cv_results)):
        for h in H_VALUES:
            mask = np.isclose(arr["h_true_mm"], h)
            m = error_metrics(arr, mask)
            rows = arr[mask]
            lines.append(
                f"| {name} | {h:+g} | {m['n']} | {m['theta_bias_deg']:.6g} | {m['theta_rmse_deg']:.6g} | {m['theta_max_abs_deg']:.6g} | {m['h_bias_mm']:.6g} | {m['h_rmse_mm']:.6g} | {m['h_max_abs_mm']:.6g} | {int(rows['theta_bound_hit'].sum())} | {int(rows['h_bound_hit'].sum())} | {int(rows['ambiguous_multistart'].sum())} |"
            )
    lines.append("")
    for name, arr in (("Same-sweep fit", full_results), ("Blocked-angle holdout", cv_results)):
        lines.append(
            f"{name}: converged rows={int(arr['solver_success'].sum())}/{arr.size}; "
            f"rows with at least one retry={int(np.count_nonzero(arr['solver_retry_count']))}; "
            f"total retried starts={int(arr['solver_retry_count'].sum())}."
        )
    for name, arr in (("Same-sweep fit", full_results), ("Blocked-angle holdout", cv_results)):
        lines += [
            "",
            f"{name}: maximum local condition number {np.nanmax(arr['condition_number_scaled']):.6g}; "
            f"median 1 μm local σθ={np.nanmedian(arr['sigma_theta_deg_1um']):.6g} deg, "
            f"median 1 μm local σt={np.nanmedian(arr['sigma_h_mm_1um']):.6g} mm; "
            f"median weak-direction 1σ distance={np.nanmedian(arr['weak_sigma_scaled_1um']):.6g} "
            "in scaled (θ/(20°), t/mm) coordinates.",
        ]
    lines += [
        "",
        "## Local Jacobian and profile diagnostics",
        "",
        "For each estimate, the three-output Jacobian is taken with respect to scaled parameters (u=θ/(20°), t/mm). Its SVD gives the local condition number and, using independent 1 μm output noise, the linearized σθ and σt. These are local likelihood scales and can be large near a weak direction; they are not posterior intervals.",
        "",
        "The bounded h profiles below use representative observations from the full sweep fit at the center and at ±10°; they are descriptive in-sample diagnostics. Profile intervals use Δχ²=1 and 3.84 and are clipped to the calibrated h domain [−1,+1] mm.",
        "",
        "| State | Singular values (mm/scaled unit) | Condition | Profile best t (mm) | Δχ² at true t | 68% t profile (mm) | 95% t profile (mm) | Domain clipped? |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    for item in profile_summaries:
        sv = item["singular_values_mm"]
        ci68 = item["profile_68_interval_mm_clipped_to_domain"]
        ci95 = item["profile_95_interval_mm_clipped_to_domain"]
        lines.append(
            f"| ADE={item['ADE_deg']:+g}°, t={item['h_true_mm']:+g} | "
            f"{sv[0]:.6g}, {sv[1]:.6g} | {item['condition_number_scaled']:.6g} | "
            f"{item['profile_best_h_mm']:.6g} | {item['profile_delta_chi2_at_true_h']:.6g} | "
            f"[{ci68[0]:.6g}, {ci68[1]:.6g}] | [{ci95[0]:.6g}, {ci95[1]:.6g}] | "
            f"68%={item['profile_touches_domain_edge_68']}, 95%={item['profile_touches_domain_edge_95']} |"
        )
    lines += [
        "",
        "## Files",
        "",
        f"- Coefficients, CV results, same-sweep and heldout estimates, Jacobian uncertainties, profile summaries: {OUT_DIR / 'p1_only_sigma_depth_nonlinear_results.pkl'}",
        f"- Primary estimates: {OUT_DIR / 'p1_only_sigma_depth_joint_estimates.svg'}",
        f"- Error and uncertainty diagnostics: {OUT_DIR / 'p1_only_sigma_depth_joint_errors_uncertainty.svg'}",
        f"- Forward-model CV diagnostics: {OUT_DIR / 'p1_only_sigma_depth_forward_model_cv.svg'}",
        f"- Bounded translation profiles: {OUT_DIR / 'p1_only_sigma_depth_h_profiles.svg'}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    with args.input.open("rb") as stream:
        data = pickle.load(stream)
    if not isinstance(data, np.ndarray) or data.dtype.names is None:
        raise ValueError("Input must be a structured NumPy array")
    required = {"h_translation_mm", "rotation_deg", "zoom_number", "real_y_si_mm", "trace_success",
                *ZOOM_METADATA_FIELDS}
    if not required.issubset(data.dtype.names):
        raise ValueError(f"Missing required input fields: {required.difference(data.dtype.names)}")
    if not np.all(data["trace_success"]) or not np.isfinite(data["real_y_si_mm"]).all():
        raise ValueError("Input must contain only successful finite raw observations")
    if data.size != 3609 or set(np.unique(data["h_translation_mm"])) != set(H_VALUES):
        raise ValueError("Expected corrected 3 x 401 x 3 sweep with h=-1,0,+1")
    if np.unique(data["rotation_deg"]).size != 401 or set(np.unique(data["zoom_number"])) != set(ZOOMS):
        raise ValueError("Unexpected corrected sweep coverage")
    for zoom, expected_values in EXPECTED_ZOOM_METADATA.items():
        rows = data["zoom_number"] == zoom
        for field, expected_value in zip(ZOOM_METADATA_FIELDS, expected_values):
            actual_values = np.unique(data[field][rows])
            if not np.array_equal(actual_values, np.array([expected_value])):
                raise ValueError(
                    f"Unexpected sequence metadata {field} at z{zoom}: "
                    f"{actual_values}; expected {expected_value}"
                )

    original = data
    data = np.empty(original.shape, dtype=original.dtype.descr + [("true_theta_rad", "f8")])
    for name in original.dtype.names:
        data[name] = original[name]
    data["true_theta_rad"] = -np.deg2rad(data["rotation_deg"])
    angle_to_fold, row_fold = fold_assignments(data)
    selected, candidate_results, best = fit_selection(data, row_fold)
    full_coeffs = fit_all(data, selected, np.ones(data.size, dtype=bool))
    cv_coeffs = best["fold_coeffs"]

    print("Forward-model candidates (degrees b/q/r) and blocked-angle CV RMSE:", flush=True)
    for item in candidate_results:
        print(
            f"  {item['orders']}: {item['cv_rmse_um']:.6g} um; "
            f"by zoom={item['cv_rmse_um_by_zoom']}", flush=True
        )
    print(f"Selected orders: {selected}", flush=True)
    print("Running same-sweep and blocked-angle inverse fits with coarse-template multistart...", flush=True)
    same = inverse_dataset(data, selected, full_coeffs, row_fold, cv_coeffs, blocked=False)
    cv = inverse_dataset(data, selected, full_coeffs, row_fold, cv_coeffs, blocked=True)

    profile_records, profile_summaries = profile_examples(data, same, full_coeffs, selected)
    figures = draw_figures(same, cv, candidate_results, profile_records, args.output_dir)
    full_fit_rmse_um_by_zoom = {}
    for z in ZOOMS:
        use = data["zoom_number"] == z
        u = data["true_theta_rad"][use] / THETA_SCALE_RAD
        h = data["h_translation_mm"][use]
        prediction = np.array([
            predict_one(full_coeffs[z], selected, float(ui), float(hi))
            for ui, hi in zip(u, h)
        ])
        err = data["real_y_si_mm"][use] - prediction
        full_fit_rmse_um_by_zoom[z] = float(1000.0 * np.sqrt(np.mean(err**2)))
    report = build_report(
        selected, candidate_results, full_coeffs, full_fit_rmse_um_by_zoom,
        same, cv, profile_summaries,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    coeff_file = args.output_dir / "p1_only_sigma_depth_nonlinear_results.pkl"
    result_payload = {
        "input": str(args.input),
        "theta_convention": "theta_rad = -rotation_deg*pi/180; ADE is stored in rotation_deg",
        "translation_convention": "h_translation_mm is physical +Y eye translation t",
        "orders_selected_b_q_r": selected,
        "theta_scale_rad": THETA_SCALE_RAD,
        "assumed_sigma_mm_per_zoom": NOISE_SIGMA_MM,
        "candidate_cv": [
            {k: v for k, v in item.items() if k not in {"cv_residuals_mm", "fold_coeffs"}}
            for item in candidate_results
        ],
        "full_sweep_coefficients": full_coeffs,
        "blocked_fold_coefficients": cv_coeffs,
        "same_sweep_estimates": same,
        "blocked_angle_holdout_estimates": cv,
        "profile_examples": profile_summaries,
        "profile_curves": profile_records,
        "figure_paths": [str(p) for p in figures],
    }
    tmp = coeff_file.with_name(coeff_file.name + ".tmp")
    with tmp.open("wb") as stream:
        pickle.dump(result_payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, coeff_file)
    report_path = args.output_dir / "p1_only_sigma_depth_nonlinear_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Saved results to {coeff_file}", flush=True)
    print(f"Saved report to {report_path}", flush=True)
    for figure in figures:
        print(f"Saved SVG to {figure}", flush=True)


if __name__ == "__main__":
    main()

