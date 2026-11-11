#!/usr/bin/env python3
"""Fit the audited P4 stop-tangent camera and eye models without CODE V.

The eye map predicts stop tangents from field, gaze, and accommodation.  The
camera map projects those tangents to the S16-frame sensor.  Fits and all
selection decisions use complete nine-ray states on the declared even grid.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.linalg import qr, solve_triangular

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Script"))
from p1_camera_fit import (  # noqa: E402
    _project, _analytic_jacobian, _closed_e0, _fit_level, _metrics,
    _serial_params, _check_analytic_jacobian, sha256_file,
)

PICKLE = ROOT / "data" / "p4_raytrace" / "p4_raytrace_251x251_12p5_A4.pkl"
FIELDS = (
    "state_index", "ray_index", "accommodation_index", "ade_index", "bde_index",
    "accommodation_d", "ade_deg", "bde_deg", "field_x_relative", "field_y_relative",
    "rc_ade_readback_deg", "rc_bde_readback_deg", "cornea_ent_d_thi_readback_mm",
    "center_rayrsi_status", "rayrsi_status", "x13", "y13", "z13", "l13", "m13",
    "n13", "x16", "y16", "z16", "l16", "m16", "n16", "xsi", "ysi", "zsi",
    "xsi_local", "ysi_local", "center_xsi_local", "center_ysi_local",
)
CAM_NAMES = ("E0", "E1", "E2", "E3", "E4")
CAM_COUNTS = (3, 4, 5, 7, 9)
SPATIAL_COUNTS = (2, 8, 20, 40)
L_ZERO_MM = 181.7259434838
L_FILE_MM = 348.6586486894943
DEFAULT_DISTANCE = "167.93 mm, zero working distance"
N_A, N_ADE, N_BDE, N_RAY = 21, 101, 101, 9


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def output_dir(value: str | None, smoke: bool = False) -> Path:
    if value is None:
        return ROOT / "runs" / ("p4_fit_smoke" if smoke else "p4_fit")
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    if smoke and not path.name.endswith("_smoke"):
        path = path.with_name(path.name + "_smoke")
    return path


def _check_schema(rows: np.ndarray, shape: tuple[int, ...]) -> dict:
    if not isinstance(rows, np.ndarray) or rows.shape != shape:
        raise ValueError(f"expected structured array shape {shape}, got {getattr(rows, 'shape', None)}")
    if rows.dtype.names != FIELDS or rows.dtype.itemsize != 8 * len(FIELDS):
        raise ValueError("P4 structured dtype does not match the 34-field retrace schema")
    if any(rows.dtype.fields[name][0] != np.dtype("float64") for name in FIELDS):
        raise ValueError("every P4 field must be float64")
    metadata = (rows.dtype.metadata or {}).get("p4_retrace")
    if not isinstance(metadata, dict):
        raise ValueError("P4 retrace dtype metadata is missing")
    if metadata.get("record_count") != len(rows):
        raise ValueError("dtype metadata record count differs from array")
    for name in FIELDS:
        if not np.all(np.isfinite(rows[name])):
            raise ValueError(f"{name} contains nonfinite data")
    for name in ("rayrsi_status", "center_rayrsi_status"):
        if np.any(rows[name] != 0):
            raise ValueError(f"{name} contains a nonzero status")
    return metadata


def validate_full(rows: np.ndarray, pickle_hash: str, expected_sha256: str | None = None) -> tuple[np.ndarray, dict]:
    metadata = _check_schema(rows, (N_A * N_ADE * N_BDE * N_RAY,))
    if expected_sha256 is not None and pickle_hash.lower() != expected_sha256.lower():
        raise ValueError(f"full P4 input SHA-256 differs from expected value: {pickle_hash}")
    if metadata.get("profile") != "full":
        raise ValueError("input metadata is not the full retrace profile")
    for label, file in (("lens_sha256", ROOT / "Lens/p4_ME_no_DM.len"),
                        ("lens_seq_sha256", ROOT / "Lens/p4_ME_no_DM.seq")):
        if metadata.get(label, "").lower() != sha256_file(file).lower():
            raise ValueError(f"{label} does not match the current lens")
    cube = rows.reshape(N_A, N_ADE, N_BDE, N_RAY)
    ia = np.arange(N_A)[:, None, None, None]
    ii = np.arange(N_ADE)[None, :, None, None]
    ji = np.arange(N_BDE)[None, None, :, None]
    ri = np.arange(N_RAY)[None, None, None, :]
    for name, expect in (
        ("accommodation_index", ia), ("ade_index", ii), ("bde_index", ji),
        ("ray_index", ri),
        ("state_index", ((ia * N_ADE + ii) * N_BDE + ji)),
        ("accommodation_d", .2 * ia), ("ade_deg", -.25 * 50 + .25 * ii),
        ("bde_deg", -.25 * 50 + .25 * ji),
    ):
        tolerance = 1e-8 if name in ("accommodation_d", "ade_deg", "bde_deg") else 0
        if np.max(np.abs(cube[name] - expect)) > tolerance:
            raise ValueError(f"{name} lattice/order mismatch")
    for axis, expect in (("accommodation_d", np.arange(N_A) * .2),
                         ("ade_deg", np.arange(N_ADE) * .25 - 12.5),
                         ("bde_deg", np.arange(N_BDE) * .25 - 12.5)):
        if not np.allclose(metadata["axes"][axis], expect, rtol=0, atol=1e-8):
            raise ValueError(f"metadata {axis} axis mismatch")
    field = np.asarray(metadata["axes"]["field_xy_y_major"], dtype=float)
    if field.shape != (9, 2) or not np.array_equal(field[4], [0, 0]):
        raise ValueError("metadata has no nine-field grid with ray 4 at zero")
    for col, j in (("field_x_relative", 0), ("field_y_relative", 1)):
        if np.max(np.abs(cube[col] - field[None, None, None, :, j])) > 1e-12:
            raise ValueError(f"{col} differs from metadata fields")
    if np.max(np.abs(rows["rc_ade_readback_deg"] - rows["ade_deg"])) > 1e-7:
        raise ValueError("RC ADE readback mismatch")
    if np.max(np.abs(rows["rc_bde_readback_deg"] - rows["bde_deg"])) > 1e-7:
        raise ValueError("RC BDE readback mismatch")
    if np.max(np.abs(rows["cornea_ent_d_thi_readback_mm"])) > 1e-8:
        raise ValueError("input must have the audited zero Cornea_ENT_D thickness")
    summaries = metadata.get("worker_summaries", [])
    if len(summaries) != 21:
        raise ValueError("expected 21 accommodation worker summaries")
    fovs = np.asarray([s["fov_deg"] for s in summaries], dtype=float)
    if fovs.shape != (21, 2) or np.max(np.abs(fovs - fovs[0])) > 1e-9:
        raise ValueError("worker field-of-view values differ across accommodation")
    if not np.allclose(fovs[0], [20, 20], rtol=0, atol=1e-8):
        raise ValueError("field-of-view differs from declared 20 degree source")
    origin = np.maximum.reduce([np.abs(rows[f"{c}16"]) for c in "xyz"])
    if np.max(origin) > 1e-6:
        raise ValueError("S16 stop intersection differs from global origin")
    norm13 = np.sqrt(sum(rows[f"{c}13"] ** 2 for c in "lmn"))
    norm16 = np.sqrt(sum(rows[f"{c}16"] ** 2 for c in "lmn"))
    norm_error = max(np.max(np.abs(norm13 - 1)), np.max(np.abs(norm16 - 1)))
    if norm_error > 1e-8 or np.any(rows["n16"] >= 0):
        raise ValueError("LMN unit norm or n16 negative sign gate failed")
    lmn_difference = max(np.max(np.abs(rows[f"{c}13"] - rows[f"{c}16"])) for c in "lmn")
    if lmn_difference > 1e-8:
        raise ValueError("S13 and S16 ray direction differs")
    segment = np.column_stack([rows[f"{c}16"] - rows[f"{c}13"] for c in "xyz"])
    direction = np.column_stack([rows[f"{c}16"] for c in "lmn"])
    segment_norm = np.linalg.norm(segment, axis=1)
    if np.any(segment_norm <= 0):
        raise ValueError("zero-length S13 to S16 segment")
    alignment = np.linalg.norm(np.cross(segment, direction), axis=1) / segment_norm
    if np.max(alignment) > 1e-7:
        raise ValueError("S13 to S16 ray collinearity gate failed")
    if np.ptp(rows["zsi"]) > 1e-9:
        raise ValueError("sensor z varies across rays")
    validation = {
        "rows": len(rows), "states": len(rows) // 9, "input_sha256": pickle_hash,
        "expected_input_sha256": expected_sha256,
        "input_sha256_check": "verified" if expected_sha256 is not None else "skipped",
        "lens_sha256": metadata["lens_sha256"], "lens_seq_sha256": metadata["lens_seq_sha256"],
        "collection_script_sha256": metadata["script_sha256"],
        "fov_deg": fovs[0].tolist(), "source_fields": field.tolist(),
        "max_stop_origin_mm": float(np.max(origin)), "max_lmn_norm_error": float(norm_error),
        "max_s13_s16_lmn_difference": float(lmn_difference),
        "max_segment_cross_normalized": float(np.max(alignment)),
        "sensor_z_ptp_mm": float(np.ptp(rows["zsi"])),
        "min_abs_n16": float(np.min(np.abs(rows["n16"]))),
        "cornea_ent_d_thi_mm": float(np.mean(rows["cornea_ent_d_thi_readback_mm"])),
        "status": "passed",
    }
    return cube, validation


def slice_cube(cube: np.ndarray, smoke: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if smoke:
        ai = np.asarray([0, 5, 10, 15, 20])
        gi = np.arange(0, 101, 5)
        return np.take(np.take(np.take(cube, ai, axis=0), gi, axis=1), gi, axis=2), ai, gi
    return cube, np.arange(21), np.arange(101)


def splits(ai: np.ndarray, gi: np.ndarray) -> dict[str, np.ndarray]:
    aa = ai % 2 == 0
    gg = (gi[:, None] % 2 == 0) & (gi[None, :] % 2 == 0)
    train = aa[:, None, None] & gg[None, :, :]
    h1 = aa[:, None, None] & ~gg[None, :, :]
    h2 = ~aa[:, None, None] & gg[None, :, :]
    h3 = ~aa[:, None, None] & ~gg[None, :, :]
    shape = (len(ai), len(gi), len(gi))
    result = {name: np.broadcast_to(mask, shape) for name, mask in
              (("train", train), ("H1", h1), ("H2", h2), ("H3", h3))}
    expected = {"train": 28611, "H1": 83600, "H2": 26010, "H3": 76000}
    if not (sum(result.values()) == 1).all():
        raise AssertionError("split masks overlap or leave states out")
    if len(ai) == 21 and len(gi) == 101:
        for name, count in expected.items():
            if int(np.count_nonzero(result[name])) != count:
                raise AssertionError(f"{name} state count differs from declared split")
    return result


def _gamma(ade: np.ndarray, bde: np.ndarray, order: str, y_sign: int) -> np.ndarray:
    a, b = np.broadcast_arrays(np.deg2rad(ade), np.deg2rad(bde))
    if order == "b_then_a":
        return np.stack((np.sin(b), y_sign * np.sin(a) * np.cos(b)), axis=-1)
    return np.stack((np.sin(b) * np.cos(a), y_sign * np.sin(a)), axis=-1)


def _phi(fields: np.ndarray, fov: np.ndarray, y_sign: int) -> np.ndarray:
    tangent = np.tan(np.deg2rad(fields * fov[None, :]))
    tangent[:, 1] *= y_sign
    return tangent / np.sqrt(1 + np.sum(tangent * tangent, axis=1))[:, None]


def _parallel_error(t: np.ndarray, d: np.ndarray) -> np.ndarray:
    cross = np.abs(t[..., 0] * d[..., 1] - t[..., 1] * d[..., 0])
    return cross / np.maximum(np.linalg.norm(t, axis=-1) * np.linalg.norm(d, axis=-1), 1e-30)


def _signed_cosine(t: np.ndarray, d: np.ndarray) -> np.ndarray:
    dot = np.sum(t * d, axis=-1)
    return dot / np.maximum(np.linalg.norm(t, axis=-1) * np.linalg.norm(d, axis=-1), 1e-30)


def geometry_gates(cube: np.ndarray, ai: np.ndarray, gi: np.ndarray,
                   fields: np.ndarray, fov: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    t = np.stack((-cube["l16"] / cube["n16"], -cube["m16"] / cube["n16"]), axis=-1)
    ai0 = int(np.flatnonzero(gi == 50)[0])
    center = t[:, :, :, 4]
    ade = gi[:, None] * .25 - 12.5
    bde = gi[None, :] * .25 - 12.5
    mixed = (ade != 0) & (bde != 0)
    candidates = []
    for order in ("b_then_a", "a_then_b"):
        for sign in (-1, 1):
            g = _gamma(ade, bde, order, sign)
            err = float(np.max(_parallel_error(center[:, mixed], g[mixed][None, :])))
            candidates.append({"order": order, "y_sign": sign, "max_parallel_error": err})
    passed = [c for c in candidates if c["max_parallel_error"] < 1e-6]
    if len(passed) != 1:
        raise ValueError(f"gaze rotation candidate gate failed or ambiguous: {candidates}")
    choice = passed[0]
    gamma = _gamma(ade, bde, choice["order"], choice["y_sign"])
    gaze_cosine = _signed_cosine(center[:, mixed], gamma[mixed][None, :])
    field_candidates = []
    for sign in (-1, 1):
        p = _phi(fields.copy(), fov, sign)
        measured = t[:, ai0, ai0, :]
        score = float(np.max(_parallel_error(measured[:, np.arange(9) != 4, :],
                                              p[np.arange(9) != 4][None, :, :])))
        field_candidates.append({"y_sign": sign, "max_parallel_error": score})
    field_pass = [c for c in field_candidates if c["max_parallel_error"] < 1e-6]
    if len(field_pass) != 1:
        raise ValueError(f"source field direction candidate gate failed or ambiguous: {field_candidates}")
    p = _phi(fields.copy(), fov, field_pass[0]["y_sign"])
    field_cosine = _signed_cosine(
        measured[:, np.arange(9) != 4, :],
        p[np.arange(9) != 4][None, :, :])
    radii = np.linalg.norm(t[:, ai0, ai0], axis=-1)
    equal = {}
    for label, ray_indices in (("edge", [1, 3, 5, 7]), ("corner", [0, 2, 6, 8])):
        r = radii[:, ray_indices]
        equal[label] = float(np.max(np.ptp(r, axis=1) / np.mean(r, axis=1)))
        if equal[label] > 1e-9:
            raise ValueError(f"zero-gaze {label} tangent radii are unequal")
    return gamma, p, {
        "gaze_candidates": candidates, "selected_gaze": choice,
        "field_candidates": field_candidates, "selected_field": field_pass[0],
        "gaze_signed_cosine_range": [float(np.min(gaze_cosine)), float(np.max(gaze_cosine))],
        "field_signed_cosine_range": [float(np.min(field_cosine)), float(np.max(field_cosine))],
        "orientation_convention": "parallelism gate uses absolute cross product; antiparallel traced rays retain negative fitted coefficients",
        "zero_gaze_tangent_radius_relative_range": equal,
        "status": "passed",
    }


def spatial(gamma: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Nested S1/S3/S5/S7 rotation-equivariant, mirror-even design."""
    gg = np.sum(gamma * gamma, axis=1)
    gp = np.sum(gamma * phi, axis=1)
    pp = np.sum(phi * phi, axis=1)
    columns, names = [], []
    n = len(gamma)
    for degree in range(4):
        for i in range(degree + 1):
            for j in range(degree - i + 1):
                k = degree - i - j
                scalar = gg ** i * gp ** j * pp ** k
                for vector, label in ((gamma, "gamma"), (phi, "phi")):
                    col = np.empty(2 * n)
                    col[0::2] = vector[:, 0] * scalar
                    col[1::2] = vector[:, 1] * scalar
                    columns.append(col)
                    names.append(f"{label}*(g.g)^{i}*(g.p)^{j}*(p.p)^{k}")
    design = np.column_stack(columns)
    if design.shape[1] != 40:
        raise AssertionError("S7 basis must have 40 nominal columns")
    return design, names


def a_basis(indices: np.ndarray, degree: int, raw_power: bool) -> tuple[np.ndarray, list[str]]:
    A = .2 * indices
    normalized = (A - 2) / 2
    design = np.polynomial.legendre.legvander(normalized, degree)
    names = [f"P{j}((A-2)/2)" for j in range(degree + 1)]
    if raw_power:
        design = np.column_stack((design, A ** .81))
        names.append("A^0.81")
    return design, names


def qr_fit(X: np.ndarray, Y: np.ndarray, names: list[str]) -> tuple[np.ndarray, dict]:
    if Y.ndim == 1:
        Y = Y[:, None]
    scales = np.linalg.norm(X, axis=0)
    if np.any(scales == 0):
        scales = np.where(scales == 0, 1, scales)
    Z = X / scales
    Q, R, pivot = qr(Z, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(R))
    rank = int(np.count_nonzero(diagonal >= 1e-10 * diagonal[0])) if len(diagonal) else 0
    if rank == 0:
        raise ValueError("design has no identifiable columns")
    coef = np.zeros((X.shape[1], Y.shape[1]))
    reduced = solve_triangular(R[:rank, :rank], Q[:, :rank].T @ Y)
    coef[pivot[:rank]] = reduced / scales[pivot[:rank], None]
    kept = [int(j) for j in pivot[:rank]]
    dropped = [j for j in range(X.shape[1]) if j not in kept]
    singular = np.linalg.svd(Z, compute_uv=False)
    independent = Z[:, kept]
    gram_inverse = np.linalg.pinv(independent.T @ independent, rcond=1e-12)
    std = np.sqrt(np.maximum(np.diag(gram_inverse), 0))
    correlation = np.divide(gram_inverse, np.outer(std, std),
                            out=np.zeros_like(gram_inverse), where=np.outer(std, std) > 0)
    diagnostic = {
        "rank": rank, "nominal_columns": X.shape[1],
        "kept_indices": kept, "kept_names": [names[j] for j in kept],
        "dropped_indices": dropped, "dropped_names": [names[j] for j in dropped],
        "column_scales": scales.tolist(), "singular_values": singular.tolist(),
        "column_scaled_condition_number": float(singular[0] / singular[rank - 1]),
        "kept_parameter_correlation": correlation.tolist(),
        "max_abs_offdiagonal_correlation": float(np.max(np.abs(correlation - np.eye(rank)))),
    }
    return coef, diagnostic


def stats(dx: np.ndarray, dy: np.ndarray, mask: np.ndarray) -> dict:
    ex, ey = dx[mask].ravel(), dy[mask].ravel()
    if not len(ex):
        return {"rays": 0, "rms_um": None, "max_um": None}
    radius = np.hypot(ex, ey)
    return {
        "rays": int(len(ex)), "states": int(np.count_nonzero(mask)),
        "rms_um": float(np.sqrt(np.mean(radius ** 2)) * 1000),
        "max_um": float(np.max(radius) * 1000),
        "mean_x_um": float(np.mean(ex) * 1000), "mean_y_um": float(np.mean(ey) * 1000),
    }


def point_stats(error_um: np.ndarray, mask: np.ndarray) -> dict:
    selected = error_um[mask].ravel()
    if not len(selected):
        return {"points": 0, "rms_um": None, "max_um": None}
    return {"points": int(len(selected)),
            "rms_um": float(np.sqrt(np.mean(selected ** 2))),
            "max_um": float(np.max(selected))}


def score_strata(dx: np.ndarray, dy: np.ndarray, masks: dict[str, np.ndarray]) -> dict:
    return {name: stats(dx, dy, mask) for name, mask in masks.items()}


def per_a_max(error_um: np.ndarray, mask: np.ndarray) -> list[dict]:
    result = []
    for a in range(error_um.shape[0]):
        values = error_um[a][mask[a]]
        if len(values):
            result.append({"accommodation_index": a, "max_um": float(np.max(values)),
                           "rms_um": float(np.sqrt(np.mean(values ** 2)))})
    return result


def camera_fit(cube: np.ndarray, tangent: np.ndarray, masks: dict[str, np.ndarray],
               distance_label: str) -> tuple[list[np.ndarray], list[dict], float, int | None]:
    train = masks["train"]
    T = float(np.max(np.abs(tangent[train])))
    if T <= 0:
        raise ValueError("zero training tangent normalization")
    u, v = tangent[..., 0] / T, tangent[..., 1] / T
    x, y = cube["xsi"], cube["ysi"]
    ut, vt, xt, yt = (array[train].ravel() for array in (u, v, x, y))
    closed = _closed_e0(ut, vt, xt, yt)
    fx = float(np.dot(ut - np.mean(ut), xt - np.mean(xt)) / np.dot(ut - np.mean(ut), ut - np.mean(ut)))
    fy = float(np.dot(vt - np.mean(vt), yt - np.mean(yt)) / np.dot(vt - np.mean(vt), vt - np.mean(vt)))
    if fx * fy <= 0 or abs(fx / fy - 1) >= 1e-3:
        raise ValueError(f"camera axes have inconsistent signs or scales: fx={fx}, fy={fy}")
    if closed[0] <= 0:
        raise ValueError(f"P4 camera focal sign is negative: {closed[0]}")
    sign_gate = {"closed_E0_f_normalized": float(closed[0]), "fx_normalized": fx,
                 "fy_normalized": fy, "fx_fy_relative_difference": float(abs(fx / fy - 1)),
                 "status": "passed"}
    params, models = [], []
    previous = np.zeros(9)
    previous[:3] = closed
    holdout = ~train
    outer_radius = np.hypot(ut, vt)
    outer_cut = float(np.quantile(outer_radius, .9))
    for level, name in enumerate(CAM_NAMES):
        fit, solver = _fit_level(level, previous, ut, vt, xt, yt)
        current = np.zeros(9)
        current[:len(fit)] = fit
        previous = current
        px, py = _project(current, u.ravel(), v.ravel(), level)
        dx = px.reshape(x.shape) - x
        dy = py.reshape(y.shape) - y
        jac = _analytic_jacobian(fit, ut, vt, level)
        _, ident = qr_fit(jac, np.zeros((len(jac), 1)), list(_serial_params(fit, T, level)["normalized"]))
        error = np.hypot(dx, dy) * 1000
        outer = holdout & (np.hypot(u, v).max(axis=-1) >= outer_cut)
        record = {
            "name": name, "level": level, "parameter_count": CAM_COUNTS[level],
            "parameters": _serial_params(current, T, level), "solver_status": int(solver.status),
            "train_sse_mm2": float(np.sum(dx[train] ** 2 + dy[train] ** 2)),
            "metrics": score_strata(dx, dy, masks),
            "holdout": stats(dx, dy, holdout),
            "outer_tangent_radius_decile": {"threshold_normalized": outer_cut,
                                            **stats(dx, dy, outer)},
            "identifiability": ident, "distance_label": distance_label,
        }
        models.append(record)
        params.append(current)
        print(f"camera {name}: holdout max {record['holdout']['max_um']:.6g} um", flush=True)
    previous_sse = None
    for record in models:
        sse = record["train_sse_mm2"]
        if previous_sse is not None and sse > previous_sse + 1e-9 * max(previous_sse, 1e-16):
            raise AssertionError("nested camera training cost increased")
        previous_sse = sse
    passing = [r["level"] for r in models if r["holdout"]["max_um"] <= 1]
    return params, models, T, (passing[0] if passing else None), sign_gate


def eye_spatial_fit(design: np.ndarray, tangent: np.ndarray, mask_gaze: np.ndarray,
                    train_a: np.ndarray, count: int, names: list[str]) -> tuple[np.ndarray, dict]:
    n_a = int(np.count_nonzero(train_a))
    ray_mask = np.repeat(mask_gaze.ravel(), 9)
    X = design.reshape(-1, 2, 40)[ray_mask].reshape(-1, 40)[:, :count]
    Y = tangent[train_a][:, mask_gaze].reshape(n_a, -1).T
    coef, diag = qr_fit(X, Y, names[:count])
    return coef.T, diag


def eye_model(design: np.ndarray, spatial_coeff: np.ndarray, a_indices: np.ndarray,
              train_a: np.ndarray, spatial_diag: dict, spatial_count: int,
              spatial_names: list[str], degree: int, raw: bool) -> tuple[np.ndarray, dict]:
    ad, names = a_basis(a_indices, degree, raw)
    trajectory, adiag = qr_fit(ad[train_a], spatial_coeff, names)
    predicted_coeff = ad @ trajectory
    pred = (design[:, :spatial_count] @ predicted_coeff.T).T.reshape(
        len(a_indices), -1, 2)
    record = {
        "id": f"S{2 * (SPATIAL_COUNTS.index(spatial_count)) + 1}_P{degree}" + ("_A081" if raw else ""),
        "spatial_order": 2 * (SPATIAL_COUNTS.index(spatial_count)) + 1,
        "spatial_count_nominal": spatial_count,
        "accommodation_degree": degree, "extra_A_081": raw,
        "parameter_count": spatial_diag["rank"] * adiag["rank"],
        "spatial_qr": spatial_diag, "accommodation_qr": adiag,
        "spatial_names": spatial_names[:spatial_count],
        "accommodation_names": names,
        "coefficient_matrix": trajectory.tolist(),
        "per_accommodation_coefficients": predicted_coeff.tolist(),
        "training_accommodation_indices": a_indices[train_a].tolist(),
    }
    return pred, record


def dense_product_check(design: np.ndarray, tangent: np.ndarray, mask_gaze: np.ndarray,
                        train_a: np.ndarray, ai: np.ndarray, count: int,
                        product_prediction: np.ndarray) -> float:
    ray_mask = np.repeat(mask_gaze.ravel(), 9)
    X = design.reshape(-1, 2, 40)[ray_mask].reshape(-1, 40)[:, :count]
    ad, _ = a_basis(ai[train_a], 2, False)
    dense = np.kron(ad, X)
    Y = tangent[train_a][:, mask_gaze].reshape(-1)
    coef = np.linalg.lstsq(dense, Y, rcond=1e-10)[0]
    target = dense @ coef
    product = product_prediction[train_a][:, mask_gaze].reshape(-1)
    difference = float(np.max(np.abs(target - product)))
    if difference > 1e-9:
        raise AssertionError(f"separable and dense eye fits differ by {difference:g} tangent")
    return difference


def pair_score(pred: np.ndarray, true_t: np.ndarray, cube: np.ndarray, T: float,
               camera: np.ndarray, level: int, masks: dict[str, np.ndarray],
               ai: np.ndarray, gi: np.ndarray) -> dict:
    shape = cube["xsi"].shape
    px, py = _project(camera, pred[..., 0].ravel() / T, pred[..., 1].ravel() / T, level)
    dx, dy = px.reshape(shape) - cube["xsi"], py.reshape(shape) - cube["ysi"]
    error = np.hypot(dx, dy) * 1000
    hold = ~masks["train"]
    # Train-A levels use held gaze; held-A levels use both gaze strata.
    per_a = []
    for j, index in enumerate(ai):
        subset = error[j][hold[j]]
        per_a.append({"accommodation_index": int(index), "accommodation_d": float(.2 * index),
                      "max_um": float(np.max(subset)),
                      "rms_um": float(np.sqrt(np.mean(subset ** 2)))})
    f_physical = float(camera[0] / T)
    eye_difference = pred - true_t
    eye_error = np.linalg.norm(eye_difference, axis=-1) * abs(f_physical) * 1000
    gaze_angle = np.hypot(gi[:, None] * .25 - 12.5, gi[None, :] * .25 - 12.5)
    ring = np.broadcast_to(gaze_angle[None, :, :] >= 10, masks["train"].shape)
    return {
        "composite": {"strata": score_strata(dx, dy, masks), "holdout": stats(dx, dy, hold),
                      "per_accommodation_holdout": per_a,
                      "outer_gaze_ring": stats(dx, dy, ring & hold)},
        "eye_only": {"strata": {name: {
            "rays": int(np.count_nonzero(mask) * 9),
            "rms_um": float(np.sqrt(np.mean(eye_error[mask] ** 2))),
            "max_um": float(np.max(eye_error[mask])),
        } for name, mask in masks.items()},
            "holdout_rms_um": float(np.sqrt(np.mean(eye_error[hold] ** 2))),
            "holdout_max_um": float(np.max(eye_error[hold]))},
        "pass_1um": bool(np.max(error[hold]) <= 1 and all(p["max_um"] <= 1 for p in per_a)),
    }


def flatness(cube: np.ndarray, tangent: np.ndarray, gi: np.ndarray, T: float,
             camera: np.ndarray, level: int, masks: dict[str, np.ndarray]) -> dict:
    gaze = ((gi[:, None] % 2 == 0) & (gi[None, :] % 2 == 0))
    result = []
    u, v = tangent[..., 0] / T, tangent[..., 1] / T
    for a in range(len(cube)):
        p, _ = _fit_level(level, camera, u[a][gaze].ravel(), v[a][gaze].ravel(),
                          cube["xsi"][a][gaze].ravel(), cube["ysi"][a][gaze].ravel())
        f_drift = abs(p[0] - camera[0]) * 1000
        k_drift = abs(camera[0] * (p[3] - camera[3])) * 1000 if level >= 1 else 0.0
        px, py = _project(camera, u[a].ravel(), v[a].ravel(), level)
        dx, dy = px.reshape(cube["xsi"][a].shape) - cube["xsi"][a], py.reshape(cube["ysi"][a].shape) - cube["ysi"][a]
        result.append({
            "accommodation_index": int(cube["accommodation_index"][a, 0, 0, 0]),
            "shared_camera_residual_mean_x_um": float(np.mean(dx) * 1000),
            "shared_camera_residual_mean_y_um": float(np.mean(dy) * 1000),
            "shared_camera_residual_rms_um": float(np.sqrt(np.mean(dx ** 2 + dy ** 2)) * 1000),
            "f_refit_normalized": float(p[0]),
            "k1_refit_normalized": float(p[3]) if level >= 1 else None,
            "f_drift_at_T_um": float(f_drift), "k1_drift_at_T_um": float(k_drift),
            "max_component_drift_at_T_um": float(max(f_drift, k_drift)),
        })
    maximum = max(r["max_component_drift_at_T_um"] for r in result)
    return {"threshold_um": .01, "max_drift_um": maximum, "passed": maximum <= .01,
            "per_accommodation": result}


def residual_maps(out: Path, label: str, gi: np.ndarray, ai: np.ndarray,
                  camera_error: np.ndarray, composite_error: np.ndarray) -> None:
    for a, index in enumerate(ai):
        for name, data in (("camera", camera_error), ("composite", composite_error)):
            fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
            im = ax.imshow(np.max(data[a], axis=-1), origin="lower", cmap="magma",
                           extent=[gi[0] * .25 - 12.5, gi[-1] * .25 - 12.5,
                                   gi[0] * .25 - 12.5, gi[-1] * .25 - 12.5])
            ax.set(xlabel="BDE (deg)", ylabel="ADE (deg)",
                   title=f"{label}; A={index * .2:g} D; {name} max over nine fields")
            fig.colorbar(im, ax=ax, label="point error (um)")
            fig.savefig(out / f"{name}_A{index:02d}.png", dpi=130)
            plt.close(fig)


def coefficient_plot(out: Path, label: str, model: dict, floor_gap: list[dict]) -> None:
    coeff = np.asarray(model["per_accommodation_coefficients"])
    ai = np.arange(len(coeff)) if len(coeff) == 21 else np.asarray([0, 5, 10, 15, 20])
    free = np.asarray(model["free_per_training_accommodation_coefficients"])
    training_ai = np.asarray(model["training_accommodation_indices"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for j, name in enumerate(("c_gamma", "c_phi")):
        axes[j].plot(ai * .2, coeff[:, j], "-", label="A-basis fit")
        axes[j].plot(training_ai * .2, free[:, j], "x", label="free per-A fit")
        axes[j].set(xlabel="Accommodation (D)", ylabel="tangent per unit basis",
                    title=f"{name}; {label}")
        axes[j].legend()
    fig.savefig(out / "coefficient_trajectories.png", dpi=140)
    plt.close(fig)
    if floor_gap:
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        ax.plot([r["accommodation_d"] for r in floor_gap],
                [r["gap_rms_um"] for r in floor_gap], "o-")
        ax.set(xlabel="Accommodation (D)", ylabel="global minus per-A floor RMS (um)",
               title=f"A-law gap; {label}")
        fig.savefig(out / "A_law_floor_gap.png", dpi=140)
        plt.close(fig)


def run_fit(args, out: Path, summary: dict) -> None:
    source = Path(args.pickle)
    source = source.resolve() if source.is_absolute() else (ROOT / source).resolve()
    source_hash = sha256_file(source)  # Hash before unpickling.
    summary["input"] = {"path": str(source), "sha256": source_hash,
                        "expected_sha256": args.expected_sha256,
                        "sha256_check": "pending" if args.expected_sha256 is not None else "skipped"}
    print(f"Input SHA-256: {source_hash}; check {'enabled' if args.expected_sha256 is not None else 'skipped (no --expected-sha256)'}", flush=True)
    if args.expected_sha256 is not None and source_hash.lower() != args.expected_sha256.lower():
        summary["input"]["sha256_check"] = "failed"
        raise ValueError(f"input SHA-256 mismatch: {source_hash}")
    summary["input"]["sha256_check"] = "verified" if args.expected_sha256 is not None else "skipped"
    raw = np.load(source, allow_pickle=True)
    full, validation = validate_full(raw, source_hash, args.expected_sha256)
    summary["input_validation"] = validation
    cube, ai, gi = slice_cube(full, args.smoke)
    masks = splits(ai, gi)
    summary["split"] = {
        "state_counts": {name: int(np.count_nonzero(mask)) for name, mask in masks.items()},
        "scope": "smoke" if args.smoke else "full",
        "no_field_holdout": True,
        "interpolation_axes": ["ADE", "BDE", "accommodation"],
    }
    tangent = np.stack((-cube["l16"] / cube["n16"], -cube["m16"] / cube["n16"]), axis=-1)
    field = np.asarray(validation["source_fields"])
    fov = np.asarray(validation["fov_deg"])
    gamma, phi, geometry = geometry_gates(cube, ai, gi, field, fov)
    summary["geometry_gates"] = geometry
    summary["analytic_jacobian_max_abs_difference"] = _check_analytic_jacobian()
    cameras, camera_models, T, camera_only_choice, sign_gate = camera_fit(
        cube, tangent, masks, args.distance_label)
    summary["camera_sign_gate"] = sign_gate
    summary["T_training_tangent"] = T
    summary["camera_models"] = camera_models
    summary["camera_only_selected"] = CAM_NAMES[camera_only_choice] if camera_only_choice is not None else None

    n_gaze_rays = len(gi) ** 2 * 9
    g = np.broadcast_to(gamma[:, :, None, :], (len(gi), len(gi), 9, 2)).reshape(n_gaze_rays, 2)
    p = np.broadcast_to(phi[None, None, :, :], (len(gi), len(gi), 9, 2)).reshape(n_gaze_rays, 2)
    design, names = spatial(g, p)
    train_a = ai % 2 == 0
    train_gaze = (gi[:, None] % 2 == 0) & (gi[None, :] % 2 == 0)
    eye_designs = []
    for count in SPATIAL_COUNTS:
        coeff, diag = eye_spatial_fit(design, tangent, train_gaze, train_a, count, names)
        eye_designs.append((count, coeff, diag))
        print(f"eye S{2 * SPATIAL_COUNTS.index(count) + 1}: rank {diag['rank']}/{count}", flush=True)
    summary["eye_spatial_identifiability"] = {
        f"S{2 * j + 1}": diag for j, (_, _, diag) in enumerate(eye_designs)}

    models, pairs = [], []
    best_key, best_model, best_camera = None, None, None
    best_pred = None
    smoke_dense = None
    floor_by_order = {}
    for spatial_index, (count, per_a_coeff, spatial_diag) in enumerate(eye_designs):
        # A free coefficient vector at each training A is a non-predictive H1 floor.
        floor_pred = (design[:, :count] @ per_a_coeff.T).T.reshape(np.count_nonzero(train_a), n_gaze_rays, 2)
        floor_error_by_a = []
        for j, original_ai in enumerate(ai[train_a]):
            local_a = int(np.flatnonzero(ai == original_ai)[0])
            held = ~train_gaze
            eye_err = np.linalg.norm(
                floor_pred[j].reshape(len(gi), len(gi), 9, 2)[held] - tangent[local_a][held], axis=-1)
            floor_x, floor_y = _project(cameras[4], floor_pred[j, :, 0] / T,
                                        floor_pred[j, :, 1] / T, 4)
            sensor_err = np.hypot(
                floor_x.reshape(len(gi), len(gi), 9)[held] - cube["xsi"][local_a][held],
                floor_y.reshape(len(gi), len(gi), 9)[held] - cube["ysi"][local_a][held]) * 1000
            floor_error_by_a.append({
                "accommodation_index": int(original_ai), "accommodation_d": float(.2 * original_ai),
                "eye_only_rms_um": float(np.sqrt(np.mean(eye_err ** 2)) * abs(cameras[4][0] / T) * 1000),
                "eye_only_max_um": float(np.max(eye_err) * abs(cameras[4][0] / T) * 1000),
                "E4_composite_rms_um": float(np.sqrt(np.mean(sensor_err ** 2))),
                "E4_composite_max_um": float(np.max(sensor_err)),
            })
        floor_by_order[f"S{2 * spatial_index + 1}"] = floor_error_by_a
        for degree in range(1, 6):
            for raw_power in (False, True):
                pred_flat, model = eye_model(design, per_a_coeff, ai, train_a, spatial_diag,
                                             count, names, degree, raw_power)
                pred = pred_flat.reshape(tangent.shape)
                model["distance_label"] = args.distance_label
                if args.smoke and count == 8 and degree == 2 and not raw_power:
                    smoke_dense = dense_product_check(design, tangent, train_gaze, train_a,
                                                      ai, count, pred)
                model["train_tangent_sse"] = float(np.sum(
                    (pred[masks["train"]] - tangent[masks["train"]]) ** 2))
                models.append(model)
                for level, camera in enumerate(cameras):
                    score = pair_score(pred, tangent, cube, T, camera, level, masks, ai, gi)
                    record = {
                        "eye_model_id": model["id"], "camera_level": CAM_NAMES[level],
                        "camera_level_index": level,
                        "total_parameter_count": CAM_COUNTS[level] + model["parameter_count"],
                        "eye_parameter_count": model["parameter_count"],
                        **score, "distance_label": args.distance_label,
                    }
                    pairs.append(record)
                    key = (record["composite"]["holdout"]["max_um"],
                           record["total_parameter_count"], level, model["id"])
                    if best_key is None or key < best_key:
                        best_key, best_model, best_camera = key, model, level
                        best_pred = pred.copy()
                print(f"{model['id']}: best composite max "
                      f"{min(r['composite']['holdout']['max_um'] for r in pairs[-5:]):.6g} um",
                      flush=True)
    summary["dense_product_check_max_abs_tangent"] = smoke_dense
    summary["eye_models"] = models
    summary["pair_scores"] = pairs
    summary["free_per_A_floor_H1"] = floor_by_order
    eye_sse_checks = []
    for raw_power in (False, True):
        for spatial_order in (1, 3, 5, 7):
            ladder = [m for m in models if m["spatial_order"] == spatial_order and
                      m["extra_A_081"] == raw_power]
            for lower, higher in zip(ladder[:-1], ladder[1:]):
                tolerance = max(1e-20, 1e-8 * lower["train_tangent_sse"])
                if higher["train_tangent_sse"] > lower["train_tangent_sse"] + tolerance:
                    raise AssertionError(f"nested A ladder training SSE increased at {higher['id']}")
                eye_sse_checks.append([lower["id"], higher["id"]])
        for degree in range(1, 6):
            ladder = [m for m in models if m["accommodation_degree"] == degree and
                      m["extra_A_081"] == raw_power]
            for lower, higher in zip(ladder[:-1], ladder[1:]):
                tolerance = max(1e-20, 1e-8 * lower["train_tangent_sse"])
                if higher["train_tangent_sse"] > lower["train_tangent_sse"] + tolerance:
                    raise AssertionError(f"nested spatial ladder training SSE increased at {higher['id']}")
                eye_sse_checks.append([lower["id"], higher["id"]])
    summary["nested_eye_training_cost_checks"] = {"status": "passed", "comparisons": eye_sse_checks}
    ordered = sorted(pairs, key=lambda r: (
        r["total_parameter_count"], r["camera_level_index"],
        next(j for j, model in enumerate(models) if model["id"] == r["eye_model_id"])))
    selected = next((record for record in ordered if record["pass_1um"]), None)
    summary["selection"] = {
        "status": "passed" if selected else "no_model_passed",
        "selected_pair": {"camera": selected["camera_level"], "eye": selected["eye_model_id"],
                          "total_parameter_count": selected["total_parameter_count"]}
        if selected else None,
        "best_diagnostic_pair": {"camera": CAM_NAMES[best_camera], "eye": best_model["id"],
                                 "holdout_max_um": best_key[0]},
        "ordered_pair_keys": [[r["camera_level"], r["eye_model_id"],
                               r["total_parameter_count"]] for r in ordered],
        "global_and_each_A_max_limit_um": 1.0,
    }
    diagnostic = selected if selected else min(pairs, key=lambda r: r["composite"]["holdout"]["max_um"])
    chosen_model = next(m for m in models if m["id"] == diagnostic["eye_model_id"])
    chosen_model["free_per_training_accommodation_coefficients"] = next(
        coefficients.tolist() for count, coefficients, _ in eye_designs
        if count == chosen_model["spatial_count_nominal"])
    level = diagnostic["camera_level_index"]
    camera = cameras[level]
    if chosen_model["id"] == best_model["id"]:
        pred = best_pred
    else:
        coeff = np.asarray(chosen_model["per_accommodation_coefficients"])
        count = chosen_model["spatial_count_nominal"]
        pred = (design[:, :count] @ coeff.T).T.reshape(tangent.shape)
    flat = flatness(cube, tangent, gi, T, camera, level, masks)
    summary["camera_flatness_gate"] = flat
    if not flat["passed"]:
        summary["selection"]["status"] = "invalid_camera_flatness"
        summary["selection"]["selected_pair"] = None

    f_physical = float(camera[0] / T)
    gamma_coeff = np.asarray(chosen_model["per_accommodation_coefficients"])[:, 0]
    phi_coeff = np.asarray(chosen_model["per_accommodation_coefficients"])[:, 1]
    lever = L_ZERO_MM * gamma_coeff / (1 + gamma_coeff)
    rho = phi_coeff * (L_ZERO_MM - lever)
    summary["derived_first_order"] = {
        "distance_label": args.distance_label,
        "interpretation": "perspective interpretation unverified; signed coefficient convention retained",
        "L_mm": L_ZERO_MM,
        "per_accommodation": [
            {"accommodation_index": int(index), "accommodation_d": float(.2 * index),
             "c_gamma": float(cg), "c_phi": float(cp),
             "d_mm": float(d), "rho_mm_per_unit_direction_sine": float(r)}
            for index, cg, cp, d, r in zip(ai, gamma_coeff, phi_coeff, lever, rho)],
    }
    summary["diagnostic_model"] = {
        "camera_level": CAM_NAMES[level], "camera_parameters_normalized": camera.tolist(),
        "eye_model_id": chosen_model["id"], "eye_model": chosen_model,
    }
    # The P1 metrics helper supplies full state/ray and radius diagnostics for
    # the chosen composite; no P1 numeric fit is used.
    px, py = _project(camera, pred[..., 0].ravel() / T, pred[..., 1].ravel() / T, level)
    dx = px.reshape(cube["xsi"].shape) - cube["xsi"]
    dy = py.reshape(cube["ysi"].shape) - cube["ysi"]
    hold = ~masks["train"]
    ray_indices = np.broadcast_to(np.arange(9), cube.shape)
    state_ids = cube["state_index"]
    p1_metrics = _metrics(px.reshape(cube["xsi"].shape)[hold].ravel(),
                          py.reshape(cube["ysi"].shape)[hold].ravel(),
                          cube["xsi"][hold].ravel(), cube["ysi"][hold].ravel(),
                          ray_indices[hold].ravel(), state_ids[hold].ravel(),
                          np.linalg.norm(tangent[hold], axis=-1).ravel() / T)
    summary["diagnostic_composite_detail"] = {
        k: v for k, v in p1_metrics.items() if not k.startswith("_")}
    summary["diagnostic_composite_detail"]["distance_label"] = args.distance_label
    camera_px, camera_py = _project(camera, tangent[..., 0].ravel() / T,
                                    tangent[..., 1].ravel() / T, level)
    camera_dx = camera_px.reshape(cube["xsi"].shape) - cube["xsi"]
    camera_dy = camera_py.reshape(cube["ysi"].shape) - cube["ysi"]
    camera_error = np.hypot(camera_dx, camera_dy) * 1000
    composite_error = np.hypot(dx, dy) * 1000
    eye_error = np.linalg.norm(pred - tangent, axis=-1) * abs(f_physical) * 1000
    stage_errors = {"camera": camera_error, "eye": eye_error,
                    "composite": composite_error}
    summary["per_accommodation_errors"] = [
        {
            "accommodation_index": int(index), "accommodation_d": float(.2 * index),
            "distance_label": args.distance_label,
            **{name: {"train": point_stats(error[j], masks["train"][j]),
                      "holdout": point_stats(error[j], hold[j])}
               for name, error in stage_errors.items()},
        }
        for j, index in enumerate(ai)
    ]
    summary["per_ray_and_class"] = {
        name: {
            "per_ray": {str(k): point_stats(error[..., k], hold) for k in range(9)},
            "center": point_stats(error[..., 4], hold),
            "edges": point_stats(error[..., [1, 3, 5, 7]], hold),
            "corners": point_stats(error[..., [0, 2, 6, 8]], hold),
        }
        for name, error in stage_errors.items()
    }
    summary["per_ray_and_class"]["distance_label"] = args.distance_label
    np.savez_compressed(
        out / "residuals.npz", distance_label=np.asarray(args.distance_label),
        accommodation_index=cube["accommodation_index"].astype(np.uint8),
        ade_index=cube["ade_index"].astype(np.uint8), bde_index=cube["bde_index"].astype(np.uint8),
        ray_index=ray_indices.astype(np.uint8), train_state=masks["train"],
        camera_point_error_um=camera_error, eye_point_error_um=eye_error,
        composite_point_error_um=composite_error,
        composite_dx_um=dx * 1000, composite_dy_um=dy * 1000)
    residual_maps(out, args.distance_label, gi, ai, camera_error, composite_error)
    floor_gap = []
    for item in floor_by_order[f"S{chosen_model['spatial_order']}"]:
        a = item["accommodation_index"]
        j = int(np.flatnonzero(ai == a)[0])
        global_eye = eye_error[j][~train_gaze]
        floor_gap.append({
            "accommodation_index": a, "accommodation_d": item["accommodation_d"],
            "floor_rms_um": item["eye_only_rms_um"],
            "global_rms_um": float(np.sqrt(np.mean(global_eye ** 2))),
            "gap_rms_um": float(np.sqrt(np.mean(global_eye ** 2)) - item["eye_only_rms_um"]),
        })
    summary["A_law_floor_gap_H1"] = floor_gap
    coefficient_plot(out, args.distance_label, chosen_model, floor_gap)
    if not flat["passed"]:
        raise ValueError(f"camera flatness hard gate exceeded 0.01 um: {flat['max_drift_um']}")


def run_pilot(args, out: Path, summary: dict) -> None:
    fit_path = Path(args.fit_summary).resolve()
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    if fit.get("distance_label") != DEFAULT_DISTANCE or fit.get("input_validation", {}).get("status") != "passed":
        raise ValueError("pilot requires a validated zero-distance P4 fit summary")
    if fit.get("selection", {}).get("status") != "passed":
        raise ValueError("pilot requires a passing selected zero-distance composite model")
    source = Path(args.pilot_pickle).resolve()
    source_hash = sha256_file(source)
    rows = np.load(source, allow_pickle=True)
    metadata = _check_schema(rows, (450,))
    if metadata.get("profile") != "pilot" or metadata.get("cornea_ent_d_thi_mode") != "file":
        raise ValueError("pilot pickle must declare file-thickness pilot mode")
    target = float(metadata.get("cornea_ent_d_thi_target_mm"))
    if abs(target - (-166.9327052056943)) > 1e-8 or np.max(
            np.abs(rows["cornea_ent_d_thi_readback_mm"] - target)) > 1e-8:
        raise ValueError("pilot Cornea_ENT_D thickness readback mismatch")
    pilot = rows.reshape(2, 5, 5, 9)
    if not np.allclose(pilot[:, 0, 0, 0]["accommodation_d"], [0, 4], rtol=0, atol=1e-8):
        raise ValueError("pilot accommodation axis mismatch")
    angle = np.asarray([-12, -6, 0, 6, 12])
    if np.max(np.abs(pilot["ade_deg"] - angle[None, :, None, None])) > 1e-8 or np.max(
            np.abs(pilot["bde_deg"] - angle[None, None, :, None])) > 1e-8:
        raise ValueError("pilot gaze lattice mismatch")
    if metadata.get("lens_sha256", "").lower() != fit["input_validation"]["lens_sha256"].lower():
        raise ValueError("pilot lens hash differs from zero-distance fit")
    if metadata.get("lens_seq_sha256", "").lower() != fit["input_validation"]["lens_seq_sha256"].lower():
        raise ValueError("pilot companion sequence hash differs from zero-distance fit")
    model = fit["diagnostic_model"]["eye_model"]
    level = CAM_NAMES.index(fit["diagnostic_model"]["camera_level"])
    camera = np.asarray(fit["diagnostic_model"]["camera_parameters_normalized"])
    T = float(fit["T_training_tangent"])
    t_pilot = np.stack((-pilot["l16"] / pilot["n16"], -pilot["m16"] / pilot["n16"]), axis=-1)
    px, py = _project(camera, t_pilot[..., 0].ravel() / T,
                      t_pilot[..., 1].ravel() / T, level)
    camera_error = np.hypot(px.reshape(pilot.shape) - pilot["xsi"],
                             py.reshape(pilot.shape) - pilot["ysi"]) * 1000
    fov = np.asarray(fit["input_validation"]["fov_deg"])
    field = np.asarray(fit["input_validation"]["source_fields"])
    order = fit["geometry_gates"]["selected_gaze"]["order"]
    gsign = fit["geometry_gates"]["selected_gaze"]["y_sign"]
    psign = fit["geometry_gates"]["selected_field"]["y_sign"]
    g = _gamma(angle[:, None], angle[None, :], order, gsign)
    p = _phi(field, fov, psign)
    ggrid = np.broadcast_to(g[:, :, None, :], (5, 5, 9, 2)).reshape(-1, 2)
    pgrid = np.broadcast_to(p[None, None, :, :], (5, 5, 9, 2)).reshape(-1, 2)
    design, _ = spatial(ggrid, pgrid)
    degree = model["accommodation_degree"]
    raw = model["extra_A_081"]
    ab, _ = a_basis(np.asarray([0, 20]), degree, raw)
    coefficient = ab @ np.asarray(model["coefficient_matrix"])
    t_zero = (design[:, :model["spatial_count_nominal"]] @ coefficient.T).T.reshape(2, 5, 5, 9, 2)
    cg = coefficient[:, 0]
    d = L_ZERO_MM * cg / (1 + cg)
    pdepth = d[:, None, None, None] * np.sqrt(1 - np.sum(g * g, axis=-1))[None, :, :, None]
    scale = (L_ZERO_MM - pdepth) / (L_FILE_MM - pdepth)
    t_transport = t_zero * scale[..., None]
    sx, sy = _project(camera, t_transport[..., 0].ravel() / T,
                      t_transport[..., 1].ravel() / T, level)
    transport_error = np.hypot(sx.reshape(pilot.shape) - pilot["xsi"],
                                sy.reshape(pilot.shape) - pilot["ysi"]) * 1000
    camera_max = float(np.max(camera_error))
    transport_max = float(np.max(transport_error))
    outcome = ("camera_audit_stop" if camera_max > 1 else
               "transported_checked_on_50_states" if transport_max <= 1 else
               "full_file_thickness_retrace_and_refit_needed")
    summary["pilot"] = {
        "source": str(source), "sha256": source_hash, "fit_summary": str(fit_path),
        "fit_summary_sha256": sha256_file(fit_path), "state_count": 50,
        "camera_only_max_um": camera_max,
        "camera_only_rms_um": float(np.sqrt(np.mean(camera_error ** 2))),
        "zero_distance_camera_holdout_max_um": fit["camera_models"][level]["holdout"]["max_um"],
        "transport_composite_max_um": transport_max,
        "transport_composite_rms_um": float(np.sqrt(np.mean(transport_error ** 2))),
        "camera_1um_pass": camera_max <= 1, "transport_1um_pass": transport_max <= 1,
        "outcome": outcome, "distance_label": "334.86 mm, file thickness pilot",
        "transport_L_zero_mm": L_ZERO_MM, "transport_L_file_mm": L_FILE_MM,
    }
    np.savez_compressed(out / "residuals.npz", distance_label=np.asarray("334.86 mm, file thickness pilot"),
                        camera_error_um=camera_error,
                        transport_error_um=transport_error)
    if camera_max > 1:
        raise ValueError("D2 camera comparison failed; audit sign or vignetting")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pickle", default=str(PICKLE))
    parser.add_argument("--expected-sha256", help="Expected full input SHA-256; default: skip hash check and print the hash found")
    parser.add_argument("--output-dir", "--out", dest="out", help="Output folder; default: runs/p4_fit (smoke: runs/p4_fit_smoke)")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--distance-label", default=DEFAULT_DISTANCE)
    parser.add_argument("--pilot-pickle")
    parser.add_argument("--fit-summary")
    args = parser.parse_args(argv)
    if bool(args.pilot_pickle) != bool(args.fit_summary):
        parser.error("--pilot-pickle and --fit-summary must be supplied together")
    out = output_dir(args.out, args.smoke)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "distance_label": args.distance_label,
        "mode": "pilot_comparison" if args.pilot_pickle else "fit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "ascending total parameters, then lower camera level; first pair with composite holdout maximum <= 1 um globally and at every A",
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "p1_camera_fit_sha256": sha256_file(ROOT / "Script/p1_camera_fit.py"),
        "p4_retrace_current_sha256": sha256_file(ROOT / "Script/p4_retrace_full_detail.py"),
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__,
                     "matplotlib": matplotlib.__version__},
        "output_directory": str(out),
    }
    try:
        if args.pilot_pickle:
            run_pilot(args, out, summary)
        else:
            run_fit(args, out, summary)
        summary["status"] = "completed"
    except Exception as exc:
        summary["status"] = "failed"
        summary["failure"] = {"type": type(exc).__name__, "message": str(exc),
                              "traceback": traceback.format_exc()}
    save_json(out / "summary.json", summary)
    artifacts = {p.name: sha256_file(p) for p in sorted(out.iterdir()) if p.is_file()}
    save_json(out / "manifest.json", {"distance_label": args.distance_label,
                                      "artifacts_sha256": artifacts})
    print(f"P4 fit output: {out}; status: {summary['status']}", flush=True)
    if summary["status"] == "failed":
        print(summary["failure"]["traceback"], file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
