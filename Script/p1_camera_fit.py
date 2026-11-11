#!/usr/bin/env python3
"""Fit nested camera-projection models to the verified p1 retrace dataset.

This module deliberately does not load or start CODE V.  ``--smoke`` runs the
same validation and fitting pipeline on a deterministic 21 by 21 state patch
and writes its outputs under the project's ``runs`` directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PICKLE = Path("data/p1_raytrace/p1_raytrace_251x251_12p5.pkl")
LENS_RELATIVE = Path("Lens/p1_ME_no_DM.seq")
LENS_HASH = "1D0C27FBB1808ABE7CD0C159E08811D73F83064846837ED6AA43FEA2719C4DBE"
N_ADE = N_BDE = 251
EXPECTED_ROWS = 567009
RAYS = np.arange(9, dtype=np.int64)
LEVEL_COUNTS = (3, 4, 5, 7, 9)
LEVEL_NAMES = ("E0", "E1", "E2", "E3", "E4")
SELECTION_RULE = "smallest model with holdout maximum point error <= 1 um"
REQUIRED = ("state_index", "ade_deg", "bde_deg", "ray_index", "field_x_normalized", "field_y_normalized",
            "xsi", "ysi", "l8", "m8", "n8", "theta_x_deg", "theta_y_deg",
            "geometric_theta_x_deg", "geometric_theta_y_deg", "rc_ade_readback_deg",
            "rc_bde_readback_deg", "rayrsi_status", "x8", "y8", "z8", "zsi")
PARAM_LABELS = ("f", "cx", "cy", "k1", "k2", "thetas", "phis", "q1", "q2")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _field_names(dtype: np.dtype) -> set[str]:
    return set(dtype.names or ())


def _need_finite(a: np.ndarray, label: str) -> None:
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{label} contains NaN or infinite values")


def _validation(data: np.ndarray) -> dict[str, Any]:
    if not isinstance(data, np.ndarray) or data.dtype.names is None:
        raise ValueError("pickle must contain a structured NumPy array")
    if data.shape != (EXPECTED_ROWS,):
        raise ValueError(f"expected {EXPECTED_ROWS} rows, found shape {data.shape}")
    if data.dtype.itemsize != 28 * 8 or len(data.dtype.names) != 28:
        raise ValueError("expected exactly 28 float64 fields (224-byte records)")
    if any(data.dtype.fields[n][0] != np.dtype("float64") for n in data.dtype.names):
        raise ValueError("all 28 structured fields must be float64")
    missing = sorted(set(REQUIRED) - _field_names(data.dtype))
    if missing:
        raise ValueError("missing required fields: " + ", ".join(missing))
    for name in data.dtype.names:
        _need_finite(data[name], name)

    if not np.all(data["rayrsi_status"] == 0):
        raise ValueError("rayrsi_status must be zero for every row")
    if not np.all(data["n8"] < 0) or not np.all(np.abs(data["n8"]) >= .9999):
        raise ValueError("n8 convention check failed: all n8 must be negative and abs(n8)>=0.9999")
    if not np.allclose(data["rc_ade_readback_deg"], data["ade_deg"], rtol=0, atol=1e-9):
        raise ValueError("ADE values do not match RC readbacks within 1e-9 degrees")
    if not np.allclose(data["rc_bde_readback_deg"], data["bde_deg"], rtol=0, atol=1e-9):
        raise ValueError("BDE values do not match RC readbacks within 1e-9 degrees")
    if np.ptp(data["zsi"]) > 0:
        raise ValueError("zsi must be a single value")
    for field in ("x8", "y8", "z8"):
        if np.max(np.abs(data[field])) > 1e-9:
            raise ValueError(f"{field} must be within 1e-9 mm of zero")

    ax = np.rint((data["ade_deg"] + 12.5) / .1).astype(np.int64)
    bx = np.rint((data["bde_deg"] + 12.5) / .1).astype(np.int64)
    if np.any((ax < 0) | (ax >= N_ADE) | (bx < 0) | (bx >= N_BDE)):
        raise ValueError("ADE/BDE angle index is outside 0..250")
    if not np.allclose(data["ade_deg"], ax * .1 - 12.5, rtol=0, atol=1e-8):
        raise ValueError("ADE angles are not on the 0.1-degree lattice")
    if not np.allclose(data["bde_deg"], bx * .1 - 12.5, rtol=0, atol=1e-8):
        raise ValueError("BDE angles are not on the 0.1-degree lattice")
    if not np.array_equal(data["state_index"], (bx * N_ADE + ax).astype(np.float64)):
        raise ValueError("stored state_index does not match angle-derived indices")
    rays = data["ray_index"].astype(np.int64)
    if not np.array_equal(data["ray_index"], rays) or np.any((rays < 0) | (rays > 8)):
        raise ValueError("ray_index must be integer 0..8")
    state = ax * N_BDE + bx
    order = np.lexsort((rays, state))
    state_sorted, ray_sorted = state[order], rays[order]
    starts = np.r_[0, np.flatnonzero(np.diff(state_sorted)) + 1]
    ends = np.r_[starts[1:], len(data)]
    if len(starts) != N_ADE * N_BDE or np.any(ends - starts != 9):
        raise ValueError("every ADE/BDE pair must contain exactly nine rays")
    if not np.array_equal(ray_sorted.reshape(-1, 9), np.tile(RAYS, (len(starts), 1))):
        raise ValueError("each ADE/BDE pair must contain ray_index 0..8 exactly once")

    tan_x = np.tan(np.deg2rad(data["theta_x_deg"]))
    tan_y = np.tan(np.deg2rad(data["theta_y_deg"]))
    ratio_x, ratio_y = data["l8"] / data["n8"], data["m8"] / data["n8"]
    dx, dy = tan_x - ratio_x, tan_y - ratio_y
    if not np.allclose(tan_x, ratio_x, rtol=1e-12, atol=1e-15):
            raise ValueError("theta_x tangent does not match l8/n8")
    if not np.allclose(tan_y, ratio_y, rtol=1e-12, atol=1e-15):
            raise ValueError("theta_y tangent does not match m8/n8")
    grouped = data[order].reshape(-1, 9)
    fields: dict[str, list[float]] = {}
    for k in RAYS:
        fx, fy = grouped[:, k]["field_x_normalized"], grouped[:, k]["field_y_normalized"]
        if np.any(fx != fx[0]) or np.any(fy != fy[0]):
            raise ValueError(f"ray {k} field coordinates change across angle states")
        fields[str(k)] = [float(fx[0]), float(fy[0])]
    if abs(fields["4"][0]) > 1e-12 or abs(fields["4"][1]) > 1e-12:
        raise ValueError("ray 4 field coordinates must be (0,0)")
    return {"angle_index_ade": ax, "angle_index_bde": bx, "ray_index": rays,
            "state_id": state, "theta_tangent_max_abs_difference": {
                "x": float(np.max(np.abs(dx))), "y": float(np.max(np.abs(dy)))},
            "theta_geometric_difference_degrees": {
                "x": float(np.max(np.abs(data["theta_x_deg"] - data["geometric_theta_x_deg"]))),
                "y": float(np.max(np.abs(data["theta_y_deg"] - data["geometric_theta_y_deg"])))} ,
            "ray_field_by_index_normalized": fields,
            "zsi_mm": float(data["zsi"][0]), "n8_convention": "n8 negative (post-reflection CODE V convention)"}


def _project(params: np.ndarray, u: np.ndarray, v: np.ndarray, level: int) -> tuple[np.ndarray, np.ndarray]:
    f, cx, cy = params[:3]
    k1 = params[3] if level >= 1 else 0.0
    k2 = params[4] if level >= 2 else 0.0
    th = params[5] if level >= 3 else 0.0
    ph = params[6] if level >= 3 else 0.0
    q1 = params[7] if level >= 4 else 0.0
    q2 = params[8] if level >= 4 else 0.0
    r2 = u*u + v*v
    radial = 1.0 + k1*r2 + k2*r2*r2
    ur, vr = radial*u, radial*v
    up, vp = ur + q1*(ur*ur + vr*vr), vr + q2*(ur*ur + vr*vr)
    den = 1.0 + th*vp - ph*up
    return f*up/den + cx, f*vp/den + cy


def _analytic_jacobian(params: np.ndarray, u: np.ndarray, v: np.ndarray, level: int) -> np.ndarray:
    """Exact residual Jacobian, interleaved x/y, for the active parameters."""
    f, _, _ = params[:3]
    k1 = params[3] if level >= 1 else 0.0
    k2 = params[4] if level >= 2 else 0.0
    th = params[5] if level >= 3 else 0.0
    ph = params[6] if level >= 3 else 0.0
    q1 = params[7] if level >= 4 else 0.0
    q2 = params[8] if level >= 4 else 0.0
    r2 = u*u + v*v
    radial = 1 + k1*r2 + k2*r2*r2
    ur, vr = radial*u, radial*v
    ur_k1, vr_k1 = u*r2, v*r2
    ur_k2, vr_k2 = u*r2*r2, v*r2*r2
    s = ur*ur + vr*vr
    up, vp = ur + q1*s, vr + q2*s
    den = 1 + th*vp - ph*up
    n = len(u); m = LEVEL_COUNTS[level]
    jac = np.zeros((2*n, m), dtype=np.float64)
    def put(col: int, dup: np.ndarray, dvp: np.ndarray) -> None:
        dden = th*dvp - ph*dup
        jac[0::2, col] = f*(dup*den - up*dden)/(den*den)
        jac[1::2, col] = f*(dvp*den - vp*dden)/(den*den)
    jac[0::2, 0] = up/den; jac[1::2, 0] = vp/den
    jac[0::2, 1] = 1; jac[1::2, 2] = 1
    if level >= 1:
        du, dv = ur_k1, vr_k1
        put(3, du + q1*2*(ur*du + vr*dv), dv + q2*2*(ur*du + vr*dv))
    if level >= 2:
        du, dv = ur_k2, vr_k2
        put(4, du + q1*2*(ur*du + vr*dv), dv + q2*2*(ur*du + vr*dv))
    if level >= 3:
        jac[0::2, 5] = -f*up*vp/(den*den)
        jac[1::2, 5] = -f*vp*vp/(den*den)
        jac[0::2, 6] = f*up*up/(den*den)
        jac[1::2, 6] = f*vp*up/(den*den)
    if level >= 4:
        put(7, s, np.zeros_like(s))
        put(8, np.zeros_like(s), s)
    return jac


def _residual_jac(params: np.ndarray, u: np.ndarray, v: np.ndarray, level: int) -> np.ndarray:
    return _analytic_jacobian(params, u, v, level)


def _closed_e0(u: np.ndarray, v: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    a = np.zeros((2*len(u), 3), dtype=np.float64)
    a[0::2, 0] = u; a[0::2, 1] = 1
    a[1::2, 0] = v; a[1::2, 2] = 1
    b = np.empty(2*len(u)); b[0::2] = x; b[1::2] = y
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _metrics(predx: np.ndarray, predy: np.ndarray, tx: np.ndarray, ty: np.ndarray,
             rays: np.ndarray, state_ids: np.ndarray, radius: np.ndarray) -> dict[str, Any]:
    dx, dy = predx - tx, predy - ty
    point = np.hypot(dx, dy)
    order = np.lexsort((rays, state_ids))
    dxg, dyg, pg = dx[order].reshape(-1, 9), dy[order].reshape(-1, 9), point[order].reshape(-1, 9)
    unique = state_ids[order].reshape(-1, 9)[:, 0]
    if not np.array_equal(np.tile(RAYS, (len(unique), 1)), rays[order].reshape(-1, 9)):
        raise ValueError("metric input does not contain ray_index 0..8 for every state")
    state_rms = np.sqrt(np.mean(pg**2, axis=1))
    bias = np.hypot(np.mean(dxg, axis=1) - dxg[:, 4], np.mean(dyg, axis=1) - dyg[:, 4])
    center = pg[:, 4]
    bins = np.linspace(0, max(float(radius.max()), 1e-15), 11)
    bin_data = []
    medrad = np.hypot(radius, 0)
    for lo, hi in zip(bins[:-1], bins[1:]):
        ix = (medrad >= lo) & (medrad <= hi if hi == bins[-1] else medrad < hi)
        if np.any(ix):
            bin_data.append({"radius_normalized_range": [float(lo), float(hi)],
                             "count": int(ix.sum()), "x_rms_um": float(np.sqrt(np.mean(dx[ix]**2))*1000),
                             "y_rms_um": float(np.sqrt(np.mean(dy[ix]**2))*1000),
                             "x_mean_um": float(np.mean(dx[ix])*1000), "y_mean_um": float(np.mean(dy[ix])*1000),
                             "point_rms_um": float(np.sqrt(np.mean(point[ix]**2))*1000),
                             "point_median_um": float(np.median(point[ix])*1000),
                             "point_p95_um": float(np.percentile(point[ix], 95)*1000)})
    return {"x_rmse_um": float(np.sqrt(np.mean(dx**2))*1000),
            "y_rmse_um": float(np.sqrt(np.mean(dy**2))*1000),
            "euclidean_rms_um": float(np.sqrt(np.mean(point**2))*1000),
            "point_max_um": float(point.max()*1000),
            "state_9ray_rms_um": {"median": float(np.median(state_rms)*1000),
                                  "p95": float(np.percentile(state_rms,95)*1000), "max": float(state_rms.max()*1000)},
            "center_ray4_um": {"rms": float(np.sqrt(np.mean(center**2))*1000), "max": float(center.max()*1000)},
            "per_ray_rms_um": {str(k): float(np.sqrt(np.mean(pg[:, k]**2))*1000) for k in RAYS},
            "bias_error_mean9_minus_ray4_um": {"rms": float(np.sqrt(np.mean(bias**2))*1000), "max": float(bias.max()*1000)},
            "residual_vs_radius_normalized": bin_data,
            "_per_state_rms_mm": state_rms, "_per_state_bias_mm": bias, "_state_ids": unique}


def _fit_level(level: int, init: np.ndarray, u: np.ndarray, v: np.ndarray,
               x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, Any]:
    count = LEVEL_COUNTS[level]
    def fun(p: np.ndarray) -> np.ndarray:
        px, py = _project(p, u, v, level)
        out = np.empty(2*len(u)); out[0::2] = px-x; out[1::2] = py-y
        return out
    def jac(p: np.ndarray) -> np.ndarray:
        return _residual_jac(p, u, v, level)
    start = _closed_e0(u, v, x, y) if level == 0 else init[:count]
    result = least_squares(fun, start, jac=jac, method="trf", x_scale="jac",
                           ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=2000)
    return result.x, result


def _check_analytic_jacobian() -> float:
    p = np.array([-.35, .001, -.002, .03, -.01, .02, -.015, .01, -.02], dtype=np.float64)
    u = np.array([-.31, -.12, .07, .24, .33], dtype=np.float64)
    v = np.array([.18, -.29, .04, .22, -.11], dtype=np.float64)
    analytic = _residual_jac(p, u, v, 4)
    numeric = np.empty_like(analytic)
    h = 1e-6
    for j in range(9):
        step = np.zeros(9); step[j] = h
        xp, yp = _project(p + step, u, v, 4)
        xm, ym = _project(p - step, u, v, 4)
        numeric[0::2, j] = (xp - xm) / (2*h)
        numeric[1::2, j] = (yp - ym) / (2*h)
    diff = float(np.max(np.abs(analytic - numeric)))
    if not np.allclose(analytic, numeric, rtol=1e-6, atol=1e-9):
        raise AssertionError(f"analytic Jacobian differs from central differences (max abs {diff:g})")
    return diff


def _serial_params(p: np.ndarray, T: float, level: int) -> dict[str, Any]:
    physical: dict[str, Any] = {}
    norms: dict[str, Any] = {}
    physical_units = {"f": "mm per physical tangent", "cx": "mm", "cy": "mm",
                      "k1": "physical tangent^-2", "k2": "physical tangent^-4",
                      "thetas": "physical tangent^-1", "phis": "physical tangent^-1",
                      "q1": "physical tangent^-1", "q2": "physical tangent^-1"}
    normalized_units = {"f": "mm per normalized tangent", "cx": "mm", "cy": "mm",
                        "k1": "normalized tangent^-2", "k2": "normalized tangent^-4",
                        "thetas": "normalized tangent^-1", "phis": "normalized tangent^-1",
                        "q1": "normalized tangent^-1", "q2": "normalized tangent^-1"}
    for i, name in enumerate(PARAM_LABELS[:LEVEL_COUNTS[level]]):
        val = float(p[i]); norms[name] = val
        physical_val = val
        if name == "f": physical_val = val / T
        elif name in ("k1",): physical_val = val / (T*T)
        elif name == "k2": physical_val = val / (T**4)
        elif name in ("thetas", "phis", "q1", "q2"): physical_val = val / T
        physical[name] = {"value": physical_val, "unit": physical_units[name]}
        norms[name] = {"value": val, "unit": normalized_units[name]}
    return {"normalized": norms, "physical": physical,
            "normalization": {"T": T, "definition": "max(abs(tx),abs(ty)) over active training rays",
                             "conversion": "f=f_norm/T; k1=k1_norm/T^2; k2=k2_norm/T^4; thetas/phis/q1/q2 = normalized coefficient/T"}}


def _write_plot(path: Path, metrics_by_split: dict[str, dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), constrained_layout=True)
    # Plot cells at their actual ADE/BDE centers; BDE runs horizontally.
    for ax, field, title in ((axes[0], "rms", "Per-state 9-ray RMS residual (µm)"),
                             (axes[1], "bias", "Bias error: mean 9 rays − ray 4 (µm)")):
        values: dict[tuple[int, int], float] = {}
        ade_values: list[int] = []; bde_values: list[int] = []
        for group in metrics_by_split.values():
            vals = group["_per_state_rms_mm"] if field == "rms" else group["_per_state_bias_mm"]
            sids = group["_state_ids"]
            for si, val in zip(sids, vals):
                ai, bi = int(si % N_ADE), int(si // N_ADE)
                values[(ai, bi)] = float(val * 1000); ade_values.append(ai); bde_values.append(bi)
        agrid, bgrid = np.unique(ade_values), np.unique(bde_values)
        arr = np.full((len(agrid), len(bgrid)), np.nan)
        apos, bpos = {v:i for i,v in enumerate(agrid)}, {v:i for i,v in enumerate(bgrid)}
        for (ai, bi), val in values.items(): arr[apos[ai], bpos[bi]] = val
        ade_angles, bde_angles = agrid * .1 - 12.5, bgrid * .1 - 12.5
        im = ax.pcolormesh(bde_angles, ade_angles, arr, shading="nearest")
        ax.set_xlabel("BDE (degrees)"); ax.set_ylabel("ADE (degrees)"); ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=.82, label="µm")
    fig.savefig(path, dpi=150); plt.close(fig)


def _run(args: argparse.Namespace) -> int:
    prefix = "p1_camera_fit_smoke" if args.smoke else "p1_camera_fit"
    default_out = ROOT / "runs" / prefix
    out = resolve_path(args.out) if args.out else default_out
    if args.smoke and not out.name.endswith("_smoke"):
        out = out.with_name(out.name + "_smoke")
    if args.smoke and (out == ROOT / "data" or (ROOT / "data") in out.parents):
        raise ValueError(f"smoke directory must be outside project data: {out}")
    pickle_path = resolve_path(args.pickle)
    if not pickle_path.is_file():
        raise FileNotFoundError(f"pickle not found: {pickle_path}")
    actual_hash = sha256_file(pickle_path)
    print(f"Input SHA-256: {actual_hash}; check {'enabled' if args.expected_sha256 is not None else 'skipped (no --expected-sha256)'}", flush=True)
    if args.expected_sha256 is not None and actual_hash.lower() != args.expected_sha256.lower():
        raise ValueError(f"pickle SHA256 mismatch: expected {args.expected_sha256}, got {actual_hash}")
    # A supplied expected hash is checked before loading the pickle.
    data = np.load(pickle_path, allow_pickle=True)
    validation = _validation(data)
    global_train_states = ((validation["angle_index_ade"] % 2 == 0) & (validation["angle_index_bde"] % 2 == 0))
    if int(np.count_nonzero(global_train_states) // 9) != 15876 or int((len(global_train_states)-np.count_nonzero(global_train_states)) // 9) != 47125:
        raise ValueError("full-data train/holdout split counts differ from required 15,876 / 47,125 states")
    smoke_indices = np.unique(np.floor(np.linspace(0, 250, 21)).astype(np.int64)) if args.smoke else np.arange(251)
    is_state = np.isin(validation["angle_index_ade"], smoke_indices) & np.isin(validation["angle_index_bde"], smoke_indices)
    data = data[is_state]
    ade_i, bde_i = validation["angle_index_ade"][is_state], validation["angle_index_bde"][is_state]
    ray = validation["ray_index"][is_state]
    sid = bde_i * N_ADE + ade_i
    split_train = (ade_i % 2 == 0) & (bde_i % 2 == 0)
    expected_grid_n = 21 if args.smoke else 251
    if len(np.unique(ade_i)) != expected_grid_n or len(np.unique(bde_i)) != expected_grid_n:
        raise ValueError("smoke/full active grid has unexpected dimensions")
    if args.smoke and (int(split_train.sum() // 9) != 121 or int((~split_train).sum() // 9) != 320):
        raise ValueError("smoke subgrid split must contain 121 train and 320 holdout states")
    tx, ty = data["l8"] / data["n8"], data["m8"] / data["n8"]
    x, y = data["xsi"], data["ysi"]
    train = split_train
    T = float(max(np.max(np.abs(tx[train])), np.max(np.abs(ty[train]))))
    if T <= 0 or not math.isfinite(T):
        raise ValueError("training tangent scale T must be finite and positive")
    u, v = tx/T, ty/T
    rad = np.hypot(u, v)
    jac_check_max_abs = _check_analytic_jacobian()
    levels: list[dict[str, Any]] = []
    params_all: list[np.ndarray] = []
    previous = np.zeros(9, dtype=np.float64)
    previous[0:3] = _closed_e0(u[train], v[train], x[train], y[train])
    outputs: list[dict[str, Any]] = []
    print(f"Selection rule: {SELECTION_RULE}")
    print(f"Analytic Jacobian finite-difference check: PASS (max abs difference {jac_check_max_abs:.3g})")
    for level, name in enumerate(LEVEL_NAMES):
        if level:
            z = previous.copy()
            z[LEVEL_COUNTS[level-1]:LEVEL_COUNTS[level]] = 0.0
            px0, py0 = _project(previous, u, v, level-1)
            px1, py1 = _project(z, u, v, level)
            if not (np.array_equal(px0, px1) and np.array_equal(py0, py1)):
                raise AssertionError(f"{name} zero-addition nesting identity failed")
        p, solver = _fit_level(level, previous, u[train], v[train], x[train], y[train])
        if solver.status == 0:
            print(f"WARNING: {name} reached max_nfev without convergence")
        fullp = np.zeros(9); fullp[:len(p)] = p
        previous = fullp
        params_all.append(fullp.copy())
        predx, predy = _project(fullp, u, v, level)
        splits = {"train": train, "holdout": ~train, "all": np.ones(len(data), dtype=bool)}
        metrics: dict[str, Any] = {}
        state_plots: dict[str, dict[str, Any]] = {}
        for split, mask in splits.items():
            m = _metrics(predx[mask], predy[mask], x[mask], y[mask], ray[mask], sid[mask], rad[mask])
            metrics[split] = {k: val for k, val in m.items() if not k.startswith("_")}
            state_plots[split] = m
        J = _residual_jac(fullp[:LEVEL_COUNTS[level]], u[train], v[train], level)
        singular = np.linalg.svd(J, compute_uv=False)
        colnorm = np.linalg.norm(J, axis=0)
        scaled = J / np.where(colnorm == 0, 1, colnorm)
        scaled_singular = np.linalg.svd(scaled, compute_uv=False)
        cost = float(np.sum((predx[train]-x[train])**2 + (predy[train]-y[train])**2))
        if levels:
            previous_cost = levels[-1]["train_sse_mm2"]
            tolerance = 64 * np.finfo(float).eps * max(abs(previous_cost), abs(cost), np.finfo(float).tiny)
            if cost > previous_cost + tolerance:
                raise AssertionError(f"training cost increased at {name}: {previous_cost:.17g} -> {cost:.17g}; tolerance={tolerance:.3g}")
        grad = float(np.linalg.norm(solver.grad, ord=np.inf))
        rec = {"name": name, "parameter_count": LEVEL_COUNTS[level], "parameters": _serial_params(fullp, T, level),
               "train_sse_mm2": cost, "solver": {"status": int(solver.status),
                   "converged": bool(solver.status > 0),
                   "message": str(solver.message), "nfev": int(solver.nfev), "gradient_inf_norm": grad,
                   "jacobian_singular_values": singular.tolist(),
                   "jacobian_condition_number": float(singular[0]/singular[-1]) if singular[-1] else None,
                   "column_scaled_jacobian_condition_number": float(scaled_singular[0]/scaled_singular[-1]) if scaled_singular[-1] else None},
               "metrics": metrics}
        levels.append(rec); outputs.append(state_plots)
        print(f"{name}: train RMSE={metrics['train']['euclidean_rms_um']:.6f} um, holdout RMSE={metrics['holdout']['euclidean_rms_um']:.6f} um, holdout max={metrics['holdout']['point_max_um']:.6f} um")
    selected_rec = next((r for r in levels if r["metrics"]["holdout"]["point_max_um"] <= 1.0), None)
    selected = None if selected_rec is None else selected_rec["name"]
    e4_holdout = levels[-1]["metrics"]["holdout"]
    all_sids = np.unique(sid)
    residual_store: dict[str, np.ndarray] = {"state_indices_ade_bde": np.column_stack((all_sids % N_ADE, all_sids // N_ADE))}
    for lev, p in enumerate(params_all):
        px, py = _project(p, u, v, lev)
        residual_store[f"{LEVEL_NAMES[lev]}_residual_predicted_minus_traced_mm"] = np.column_stack((px-x, py-y))
    residual_store["original_state_index"] = data["state_index"].astype(np.int64)
    residual_store["i_ade"] = ade_i
    residual_store["i_bde"] = bde_i
    residual_store["ray_index"] = ray
    residual_store["is_train"] = train
    residual_store["radius_normalized"] = rad
    residual_store["tangent_x_physical"] = tx
    residual_store["tangent_y_physical"] = ty
    residual_store["target_x_mm"] = x
    residual_store["target_y_mm"] = y
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "residuals.npz", **residual_store)
    for lev, name in enumerate(LEVEL_NAMES):
        _write_plot(out / f"{name}_residual_maps.png", {"all": outputs[lev]["all"]})
    lens_path = ROOT / LENS_RELATIVE
    lens_info: dict[str, Any] = {"path": str(lens_path), "provided_hash": LENS_HASH}
    if lens_path.is_file():
        lh = sha256_file(lens_path)
        lens_info.update({"present": True, "sha256": lh,
                          "matches_provided_hash": lh == LENS_HASH})
    else:
        lens_info["present"] = False
    summary = {"input_pickle": str(pickle_path), "input_pickle_sha256": actual_hash,
               "expected_input_sha256": args.expected_sha256,
               "input_sha256_check": "verified" if args.expected_sha256 is not None else "skipped",
               "authoritative_lens": lens_info, "validation": {k:v for k,v in validation.items() if not isinstance(v,np.ndarray)},
               "training_split": {"rule": "both global i_ade and i_bde even", "train_states": int(train.sum()//9),
                                  "holdout_states": int((~train).sum()//9), "smoke_subgrid_global_indices": smoke_indices.tolist() if args.smoke else None},
               "normalization_T": T, "normalized_inputs": "u=tx/T, v=ty/T; radial coordinate r=sqrt(u^2+v^2)",
               "split_scope": "smoke 21x21 ADE/BDE sample spanning global indices 0..250" if args.smoke else "full grid",
               "analytic_jacobian_check_max_abs_difference": jac_check_max_abs,
               "all_levels_converged": all(r["solver"]["converged"] for r in levels),
               "models": levels, "selection": {"criterion": SELECTION_RULE, "selected": selected,
                   "selected_holdout_bias_error_max_um": None if selected_rec is None else selected_rec["metrics"]["holdout"]["bias_error_mean9_minus_ray4_um"]["max"],
                   "selected_bias_error_max_le_1um": None if selected_rec is None else selected_rec["metrics"]["holdout"]["bias_error_mean9_minus_ray4_um"]["max"] <= 1.0,
                   "no_model_passed": selected_rec is None,
                   "e4_floor_holdout_point_max_um": e4_holdout["point_max_um"] if selected_rec is None else None,
                   "e4_floor_holdout_point_rms_um": e4_holdout["euclidean_rms_um"] if selected_rec is None else None,
                   "e4_residual_structure": {"radius_bins": e4_holdout["residual_vs_radius_normalized"], "per_field_rms_um": e4_holdout["per_ray_rms_um"]} if selected_rec is None else None},
               "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
               "units": {"targets_and_residuals": "mm (metrics convert to micrometres)", "tangents": "dimensionless",
                         "angles": "degrees", "residual_sign": "predicted minus traced",
                         "bias": "mean of nine per-ray residual vectors minus ray-4 residual vector"},
               "output_path": str(out), "script_sha256": sha256_file(Path(__file__).resolve())}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Selected model: {selected}")
    print(f"Output: {out}")
    if args.smoke:
        print("Smoke checks: PASS (full input validation, 21x21 fit, nested-model identity, finite outputs)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pickle", default=str(DEFAULT_PICKLE), help="verified structured NumPy pickle")
    parser.add_argument("--expected-sha256", help="Expected input SHA-256; default: skip hash check and print the hash found")
    parser.add_argument("--output-dir", "--out", dest="out", help="Output folder; default: runs/p1_camera_fit (smoke: runs/p1_camera_fit_smoke)")
    parser.add_argument("--overwrite", action="store_true", help="compatibility option; output files are always overwritten")
    parser.add_argument("--smoke", action="store_true", help="fit the deterministic 21x21 validation subgrid")
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except Exception as exc:
        traceback.print_exc()
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
