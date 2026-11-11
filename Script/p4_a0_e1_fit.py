#!/usr/bin/env python3
"""Fit P4 accommodation index 0 with the shared P1 E1 camera model."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "Script"
PICKLE = ROOT / "data/p4_raytrace/p4_raytrace_251x251_12p5_A4.pkl"
DISTANCE_LABEL = "167.93 mm, zero working distance"
N_ANGLE = 101
RAYS = np.arange(9, dtype=np.int64)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old
    return module


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def radius_metrics(dx: np.ndarray, dy: np.ndarray, radius: np.ndarray,
                   mask: np.ndarray) -> list[dict[str, Any]]:
    """Residual summaries by physical tangent radius for a selected split."""
    hi = float(np.max(radius[mask]))
    edges = np.linspace(0.0, hi, 11)
    result = []
    point = np.hypot(dx, dy)
    for index, (lo, upper) in enumerate(zip(edges[:-1], edges[1:])):
        selected = mask & (radius >= lo)
        selected &= radius <= upper if index == len(edges) - 2 else radius < upper
        if not np.any(selected):
            continue
        result.append({
            "radius_physical_tangent_range": [float(lo), float(upper)],
            "count_rays": int(np.count_nonzero(selected)),
            "x_rmse_um": float(np.sqrt(np.mean(dx[selected] ** 2)) * 1000),
            "y_rmse_um": float(np.sqrt(np.mean(dy[selected] ** 2)) * 1000),
            "point_rms_um": float(np.sqrt(np.mean(point[selected] ** 2)) * 1000),
            "point_max_um": float(np.max(point[selected]) * 1000),
        })
    return result


def class_metrics(point_um: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    groups = {
        "per_ray": {str(ray): (RAYS == ray) for ray in range(9)},
        "center_ray4": (RAYS == 4),
        "edge_rays_1_3_5_7": np.isin(RAYS, [1, 3, 5, 7]),
        "corner_rays_0_2_6_8": np.isin(RAYS, [0, 2, 6, 8]),
    }
    result: dict[str, Any] = {}
    for group, raymask in groups.items():
        if group == "per_ray":
            result[group] = {}
            for ray, select in raymask.items():
                values = point_um[mask & select[None, None, :]]
                result[group][ray] = {
                    "rms_um": float(np.sqrt(np.mean(values ** 2))),
                    "point_max_um": float(np.max(values)),
                }
        else:
            values = point_um[raymask[None, None, :] & mask]
            result[group] = {
                "rms_um": float(np.sqrt(np.mean(values ** 2))),
                "point_max_um": float(np.max(values)),
            }
    return result


def metrics(dx_um: np.ndarray, dy_um: np.ndarray, point_um: np.ndarray,
            mask: np.ndarray, rays: np.ndarray) -> dict[str, Any]:
    return {
        "x_rmse_um": float(np.sqrt(np.mean(dx_um[mask] ** 2))),
        "y_rmse_um": float(np.sqrt(np.mean(dy_um[mask] ** 2))),
        "euclidean_rms_um": float(np.sqrt(np.mean(point_um[mask] ** 2))),
        "point_max_um": float(np.max(point_um[mask])),
        "ray_classes": class_metrics(point_um, mask),
    }


def condition_summary(jacobian: np.ndarray) -> dict[str, Any]:
    """Condition/correlation for normalized parameters, using SVD covariance."""
    _, singular, vh = np.linalg.svd(jacobian, full_matrices=False)
    tolerance = np.finfo(float).eps * max(jacobian.shape) * singular[0]
    rank = int(np.count_nonzero(singular > tolerance))
    covariance = (vh.T / singular**2) @ vh
    scale = np.sqrt(np.maximum(np.diag(covariance), 0))
    denominator = scale[:, None] * scale[None, :]
    correlation = np.divide(covariance, denominator,
                            out=np.full_like(covariance, np.nan), where=denominator > 0)
    scaled = jacobian / np.maximum(np.linalg.norm(jacobian, axis=0), 1e-300)
    return {
        "parameter_order": ["f_normalized", "cx_mm", "cy_mm", "k1_normalized"],
        "rank": rank,
        "singular_values": singular.tolist(),
        "raw_jacobian_condition_number": float(singular[0] / singular[-1]),
        "column_scaled_jacobian_condition_number": float(np.linalg.cond(scaled)),
        "parameter_correlation_from_local_covariance": correlation.tolist(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-sha256", help="Expected input SHA-256; default: skip hash check and print the hash found")
    parser.add_argument("--smoke", action="store_true",
                        help="fit a parity-preserving sample, score all A0 states")
    parser.add_argument("--output-dir", "--out", dest="out", help="Output folder; default: runs/p4_a0_e1 (smoke: runs/p4_a0_e1_smoke)")
    args = parser.parse_args(argv)

    out = (Path(args.out).resolve() if args.out else
           ROOT / "runs" / ("p4_a0_e1_smoke" if args.smoke else "p4_a0_e1"))
    if args.smoke and not out.name.endswith("_smoke"):
        out = out.with_name(out.name + "_smoke")
    out.mkdir(parents=True, exist_ok=True)
    (ROOT / "runs").mkdir(parents=True, exist_ok=True)
    run_id = out.name

    def log(message: str) -> None:
        line = f"{datetime.now(timezone.utc).isoformat()} [{run_id}] {message}\n"
        with (ROOT / "runs/subagent_status.log").open("a", encoding="utf-8") as stream:
            stream.write(line)
        print(message, flush=True)

    log(f"run started; pid={os.getpid()}")

    summary: dict[str, Any] = {
        "distance_label": DISTANCE_LABEL,
        "distance_mm": 167.93,
        "working_distance_mm": 0,
        "model": "E1 P1 camera model at accommodation index 0",
        "accommodation_independence": "single A=0 fit; no accommodation-dependent parameters",
        "mapping": "tx=-l16/n16, ty=-m16/n16; shared P1 E1 projection to xsi/ysi",
        "accommodation_index": 0,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_directory": str(out),
        "pid": os.getpid(),
        "input": str(PICKLE),
        "expected_input_sha256": args.expected_sha256,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "smoke": args.smoke,
    }
    try:
        digest = sha256_file(PICKLE)
        summary["input_sha256"] = digest
        print(f"Input SHA-256: {digest}; check {'enabled' if args.expected_sha256 is not None else 'skipped (no --expected-sha256)'}", flush=True)
        if args.expected_sha256 is not None and digest.lower() != args.expected_sha256.lower():
            raise ValueError(f"input SHA256 mismatch: {digest}")
        summary["input_sha256_check"] = "verified" if args.expected_sha256 is not None else "skipped"
        log(f"input hash check {summary['input_sha256_check']}; pid={os.getpid()}")

        # The existing P4 validation is CODE V-free. It checks the complete
        # source once, then only accommodation index 0 is fitted below.
        sys.path.insert(0, str(SCRIPT))
        p4 = load_module("p4_fit_a0_validation", SCRIPT / "p4_fit.py")
        p1 = sys.modules["p1_camera_fit"]
        p1_camera_refit_summary = ROOT / "archive/2026-11-11/runs/p1_camera_refit_20261110T224852Z_cdb51f18/summary.json"
        p1_refit_max = json.loads(p1_camera_refit_summary.read_text(encoding="utf-8"))[
            "metrics"
        ]["holdout"]["point_max_um"]
        if not math.isclose(p1_refit_max, 0.001581, rel_tol=0, abs_tol=5e-7):
            raise ValueError(f"P1 comparison value differs from supplied figure: {p1_refit_max}")
        rows = np.load(PICKLE, allow_pickle=True)
        full_cube, validation = p4.validate_full(rows, digest, args.expected_sha256)
        cube = full_cube[0].copy()
        del rows, full_cube
        summary["input_validation"] = validation
        summary["protected_source_sha256"] = {
            name: sha256_file(SCRIPT / name)
            for name in ("p1_camera_fit.py", "p1_camera_refit.py", "p4_fit.py")
        }
        summary["p1_comparison_source_sha256"] = sha256_file(p1_camera_refit_summary)
        log("full P4 validation passed; A0 slice selected")
        summary["accommodation_axis_confirmed"] = {
            "index": 0,
            "accommodation_d": float(cube["accommodation_d"][0, 0, 0]),
        }
        a0_origin = max(float(np.max(np.abs(cube[name]))) for name in ("x16", "y16", "z16"))
        a0_norm = np.sqrt(cube["l16"]**2 + cube["m16"]**2 + cube["n16"]**2)
        summary["a0_gate_details"] = {
            "max_abs_stop_origin_mm": a0_origin,
            "max_lmn_unit_norm_error": float(np.max(np.abs(a0_norm - 1.0))),
            "n16_negative_all_rays": bool(np.all(cube["n16"] < 0)),
            "min_abs_n16": float(np.min(np.abs(cube["n16"]))),
            "ray_status_zero_all_rays": bool(np.all(cube["rayrsi_status"] == 0)),
            "center_ray_status_zero_all_states": bool(np.all(cube["center_rayrsi_status"] == 0)),
            "sensor_z_ptp_mm": float(np.ptp(cube["zsi"])),
        }
        if summary["accommodation_axis_confirmed"]["accommodation_d"] != 0.0:
            raise ValueError("accommodation index 0 does not map to A=0")

        ade_index = np.broadcast_to(np.arange(N_ANGLE)[:, None, None], cube.shape)
        bde_index = np.broadcast_to(np.arange(N_ANGLE)[None, :, None], cube.shape)
        ray_index = np.broadcast_to(RAYS, cube.shape)
        train = ((ade_index % 2 == 0) & (bde_index % 2 == 0))
        holdout = ~train
        fit_train = train.copy()
        if args.smoke:
            sample_indices = np.unique(np.rint(np.linspace(0, 100, 21)).astype(int))
            fit_train &= np.isin(ade_index, sample_indices) & np.isin(bde_index, sample_indices)

        train_states = int(np.count_nonzero(train[:, :, 0]))
        holdout_states = int(np.count_nonzero(holdout[:, :, 0]))
        expected_counts = (2601, 7600)
        if (train_states, holdout_states) != expected_counts:
            raise ValueError(f"split state counts differ: {(train_states, holdout_states)}")
        if np.count_nonzero(train) != train_states * 9 or np.count_nonzero(holdout) != holdout_states * 9:
            raise ValueError("split does not contain complete nine-ray states")

        tx = -cube["l16"] / cube["n16"]
        ty = -cube["m16"] / cube["n16"]
        target_x, target_y = cube["xsi"], cube["ysi"]
        T = float(max(np.max(np.abs(tx[fit_train])), np.max(np.abs(ty[fit_train]))))
        if not math.isfinite(T) or T <= 0:
            raise ValueError(f"invalid training tangent scale T={T}")
        u, v = tx / T, ty / T

        # Independent axis fits check sign and relative scale before the E1 fit.
        fx = np.linalg.lstsq(np.column_stack((tx[fit_train], np.ones(np.count_nonzero(fit_train)))),
                             target_x[fit_train], rcond=None)[0][0]
        fy = np.linalg.lstsq(np.column_stack((ty[fit_train], np.ones(np.count_nonzero(fit_train)))),
                             target_y[fit_train], rcond=None)[0][0]
        same_sign = bool(np.sign(fx) == np.sign(fy))
        relative_difference = float(abs(fx / fy - 1.0))
        sign_check = {
            "fx_mm_per_physical_tangent": float(fx),
            "fy_mm_per_physical_tangent": float(fy),
            "fx_normalized": float(fx * T),
            "fy_normalized": float(fy * T),
            "same_sign": same_sign,
            "relative_magnitude_difference": relative_difference,
            "limit": 1e-3,
        }
        if not same_sign or relative_difference >= 1e-3:
            raise ValueError(f"independent fx/fy axis check failed: {sign_check}")

        # Reuse the exact P1 closed-form pinhole initializer, then supply the
        # additional zero k1 coefficient required by the shared E1 fitter.
        e0 = p1._closed_e0(u[fit_train], v[fit_train],
                           target_x[fit_train], target_y[fit_train])
        initial = np.r_[e0, 0.0]
        log(f"fit started; training rays={np.count_nonzero(fit_train)}; T={T:.12g}")
        params, solver = p1._fit_level(1, initial, u[fit_train], v[fit_train],
                                       target_x[fit_train], target_y[fit_train])
        pred_x, pred_y = p1._project(params, u, v, 1)
        dx = pred_x - target_x
        dy = pred_y - target_y
        point = np.hypot(dx, dy)
        radius = np.hypot(tx, ty)
        analytic_jacobian = p1._analytic_jacobian(params, u[fit_train], v[fit_train], 1)

        # Shared metrics helper reports coordinate/point RMS; add requested
        # maxima and ray-class detail around its compact result.
        state_ids = (ade_index[:, :, 0] * N_ANGLE + bde_index[:, :, 0])[:, :, None]
        state_ids = np.broadcast_to(state_ids, cube.shape)
        metrics_by_split = {}
        for name, mask in (("train", train), ("holdout", holdout)):
            base = p1._metrics(pred_x[mask], pred_y[mask], target_x[mask], target_y[mask],
                               ray_index[mask], state_ids[mask], radius[mask] / T)
            metrics_by_split[name] = {
                key: value for key, value in base.items() if not key.startswith("_")
            }
            metrics_by_split[name]["point_max_um"] = float(np.max(point[mask]) * 1000)
            metrics_by_split[name]["ray_classes"] = class_metrics(point * 1000, mask)
            metrics_by_split[name]["residual_vs_radius"] = radius_metrics(dx, dy, radius, mask)

        serialized_params = p1._serial_params(params, T, 1)
        if params[0] < 0:
            serialized_params["signed_f_convention"] = (
                "negative f retained; agreeing axes; no image flip applied"
            )
            log("negative f retained under agreeing-axis convention; no image flip applied")
        summary.update({
            "status": "completed",
            "distance_context": DISTANCE_LABEL,
            "split": {
                "rule": "ADE and BDE both even train; all remaining A0 states holdout",
                "training_states": train_states,
                "training_rays": int(np.count_nonzero(train)),
                "holdout_states": holdout_states,
                "holdout_rays": int(np.count_nonzero(holdout)),
                "fit_training_states": int(np.count_nonzero(fit_train[:, :, 0])),
                "fit_training_rays": int(np.count_nonzero(fit_train)),
                "smoke_sample_only_changes_fit": bool(args.smoke),
            },
            "sign_check": sign_check,
            "parameters": serialized_params,
            "fit": {
                "solver_success": bool(solver.success),
                "solver_status": int(solver.status),
                "solver_message": str(solver.message),
                "cost": float(solver.cost),
                "optimality": float(solver.optimality),
                "nfev": int(solver.nfev),
                "njev": int(solver.njev),
                "conditioning": condition_summary(analytic_jacobian),
            },
            "train_metrics": metrics_by_split["train"],
            "holdout_metrics": metrics_by_split["holdout"],
            "comparison_context": {
                "p1_camera_refit_dataset_holdout_max_um": p1_refit_max,
                "p1_source": str(p1_camera_refit_summary.relative_to(ROOT)),
                "prior_p4_all_accommodation_e1_max_um": 0.502306,
                "comparison_note": "Different datasets and fitting scopes; no causal inference.",
            },
        })
        log(f"fit finished; holdout point max={metrics_by_split['holdout']['point_max_um']:.9g} um")

        np.savez_compressed(
            out / "residuals.npz",
            distance_label=np.asarray(DISTANCE_LABEL),
            residual_direction=np.asarray("predicted minus traced sensor coordinate"),
            coordinate_unit_mm=np.asarray("mm"),
            residual_unit_um=np.asarray("um"),
            accommodation_index=np.asarray(0),
            ade_index=np.arange(N_ANGLE),
            bde_index=np.arange(N_ANGLE),
            ray_index=ray_index.astype(np.uint8),
            train_state=train[:, :, 0],
            fit_train_state=fit_train[:, :, 0],
            state_index=(np.arange(N_ANGLE)[:, None] * N_ANGLE + np.arange(N_ANGLE)[None, :]),
            tx=tx, ty=ty, physical_tangent_radius=radius, T_training_tangent=np.asarray(T),
            target_x=target_x, target_y=target_y, prediction_x=pred_x, prediction_y=pred_y,
            dx_um=dx * 1000, dy_um=dy * 1000, point_um=point * 1000,
            params_normalized=params,
        )

        # Per-ray residual magnitude maps with a shared scale, plus signed
        # per-ray x/y maps for diagnosing directional structure.
        magnitude_um = point * 1000
        vmax = float(np.max(magnitude_um))
        fig, axes = plt.subplots(3, 3, figsize=(14, 13), constrained_layout=True)
        for ray, axis in enumerate(axes.flat):
            image = axis.imshow(magnitude_um[:, :, ray], origin="lower",
                                extent=(-12.625, 12.625, -12.625, 12.625),
                                vmin=0, vmax=vmax, aspect="equal")
            axis.set_title(f"{DISTANCE_LABEL}\nA=0 E1 ray {ray} point residual (µm)")
            axis.set_xlabel("BDE (degrees)")
            axis.set_ylabel("ADE (degrees)")
            fig.colorbar(image, ax=axis, shrink=0.8, label="µm")
        fig.savefig(out / "residual_maps.png", dpi=140)
        plt.close(fig)

        fig, axes = plt.subplots(3, 3, figsize=(14, 13), constrained_layout=True)
        images = []
        for ray, axis in enumerate(axes.flat):
            image = axis.imshow(dx[:, :, ray] * 1000, origin="lower",
                                extent=(-12.625, 12.625, -12.625, 12.625),
                                cmap="coolwarm", aspect="equal")
            images.append(image)
            axis.set_title(f"{DISTANCE_LABEL}\nA=0 E1 ray {ray} signed x residual (µm)")
            axis.set_xlabel("BDE (degrees)")
            axis.set_ylabel("ADE (degrees)")
        signed_limit = float(np.max(np.abs(dx * 1000)))
        for image in images:
            image.set_clim(-signed_limit, signed_limit)
        fig.colorbar(images[0], ax=axes, shrink=0.8, label="µm")
        fig.savefig(out / "signed_x_residual_maps.png", dpi=140)
        plt.close(fig)

        fig, axes = plt.subplots(3, 3, figsize=(14, 13), constrained_layout=True)
        images = []
        for ray, axis in enumerate(axes.flat):
            image = axis.imshow(dy[:, :, ray] * 1000, origin="lower",
                                extent=(-12.625, 12.625, -12.625, 12.625),
                                cmap="coolwarm", aspect="equal")
            images.append(image)
            axis.set_title(f"{DISTANCE_LABEL}\nA=0 E1 ray {ray} signed y residual (µm)")
            axis.set_xlabel("BDE (degrees)")
            axis.set_ylabel("ADE (degrees)")
        signed_limit = float(np.max(np.abs(dy * 1000)))
        for image in images:
            image.set_clim(-signed_limit, signed_limit)
        fig.colorbar(images[0], ax=axes, shrink=0.8, label="µm")
        fig.savefig(out / "signed_y_residual_maps.png", dpi=140)
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
        for split_name, split_mask in (("train", train), ("holdout", holdout)):
            bins = metrics_by_split[split_name]["residual_vs_radius"]
            centers = [np.mean(item["radius_physical_tangent_range"]) for item in bins]
            values = [item["point_rms_um"] for item in bins]
            axis.plot(centers, values, marker="o", label=split_name)
        axis.set_title(f"{DISTANCE_LABEL}\nA=0 E1 point RMS by tangent radius")
        axis.set_xlabel("Physical stop tangent radius")
        axis.set_ylabel("Point RMS (µm)")
        axis.grid(True, alpha=0.3)
        axis.legend()
        fig.savefig(out / "residual_vs_radius.png", dpi=150)
        plt.close(fig)
        log("residual archive and PNG outputs complete")
    except Exception as exc:
        summary["status"] = "failed"
        summary["failure"] = {
            "type": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        log(f"failed: {type(exc).__name__}: {exc}")

    (out / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    log(f"finished status={summary['status']}")
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
