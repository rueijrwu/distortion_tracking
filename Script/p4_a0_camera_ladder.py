#!/usr/bin/env python3
"""Fit the P4 A=0 camera ladder E1 through E4 at 167.93 mm, zero working distance."""
from __future__ import annotations

import importlib.util
import argparse
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
PICKLE = ROOT / "data/distortion_grid_p4/p4_full_detail.pkl"
EXPECTED_SHA256 = "aceb43c9744dec4898df558d4e21ab786dede141e464a47303b1dd13c0892c03"
DISTANCE_LABEL = "167.93 mm, zero working distance"
N_ANGLE = 101
RAYS = np.arange(9, dtype=np.int64)
LEVELS = (1, 2, 3, 4)
NAMES = ("E1", "E2", "E3", "E4")


def _load(name: str, path: Path):
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


def _sha(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _condition(jac: np.ndarray, residual: np.ndarray, labels: list[str]) -> dict[str, Any]:
    _, singular, vh = np.linalg.svd(jac, full_matrices=False)
    tol = np.finfo(float).eps * max(jac.shape) * singular[0]
    rank = int(np.count_nonzero(singular > tol))
    invs2 = np.zeros_like(singular)
    invs2[singular > tol] = 1 / singular[singular > tol] ** 2
    cov_unscaled = (vh.T * invs2) @ vh
    dof = max(1, len(residual) - jac.shape[1])
    residual_variance = float(residual @ residual / dof)
    cov = cov_unscaled * residual_variance
    sigma = np.sqrt(np.maximum(np.diag(cov), 0))
    denom = sigma[:, None] * sigma[None, :]
    corr = np.divide(cov, denom, out=np.full_like(cov, np.nan), where=denom > 0)
    scaled = jac / np.maximum(np.linalg.norm(jac, axis=0), 1e-300)
    return {
        "parameter_labels": labels, "rank": rank, "singular_values": singular.tolist(),
        "raw_jacobian_condition_number": float(singular[0] / singular[-1]) if singular[-1] else None,
        "column_scaled_jacobian_condition_number": float(np.linalg.cond(scaled)),
        "covariance_residual_variance_scaled": cov.tolist(), "standard_errors": sigma.tolist(),
        "correlation_matrix": corr.tolist(), "residual_rss_mm2": float(residual @ residual),
        "residual_degrees_of_freedom": dof,
        "residual_variance_mm2": residual_variance,
        "scaling_note": "Jacobian covariance (J^T J)^+ multiplied by RSS/(2*N-p), where residual components are x and y errors in mm.",
    }


def _metrics(p1, helpers, predx, predy, tx, ty, rays, states, radius, T, mask, point):
    dx, dy = predx - tx, predy - ty
    base = p1._metrics(predx[mask], predy[mask], tx[mask], ty[mask], rays[mask], states[mask], (radius/T)[mask])
    out = {k: v for k, v in base.items() if not k.startswith("_")}
    out["point_max_um"] = float(np.max(point[mask]) * 1000)
    out["ray_classes"] = {}
    classes = {"center_ray4": [4], "edge_rays_1_3_5_7": [1, 3, 5, 7], "corner_rays_0_2_6_8": [0, 2, 6, 8]}
    classes.update({f"ray_{i}": [i] for i in range(9)})
    for name, indices in classes.items():
        m = mask & np.isin(RAYS, indices)[None, None, :]
        out["ray_classes"][name] = {
            "x_rms_um": float(np.sqrt(np.mean(dx[m] ** 2)) * 1000),
            "y_rms_um": float(np.sqrt(np.mean(dy[m] ** 2)) * 1000),
            "point_rms_um": float(np.sqrt(np.mean(point[m] ** 2)) * 1000),
            "point_max_um": float(np.max(point[m]) * 1000),
        }
    # E1-compatible equal-width physical-radius deciles plus explicit quantile bins.
    equal_width = helpers.radius_metrics(dx, dy, radius, mask)
    out["radius_equal_width_deciles"] = [{"range": x["radius_physical_tangent_range"], "count": x["count_rays"],
        "x_rms_um":x["x_rmse_um"],"y_rms_um":x["y_rmse_um"],"point_rms_um":x["point_rms_um"],"point_max_um":x["point_max_um"]} for x in equal_width]
    vals = radius[mask]
    q = np.quantile(vals, np.linspace(0, 1, 11))
    out["radius_quantile_deciles"] = []
    for i, (lo, hi) in enumerate(zip(q[:-1], q[1:])):
        m = mask & (radius >= lo) & ((radius <= hi) if i == 9 else (radius < hi))
        if np.any(m):
            out["radius_quantile_deciles"].append({"range": [float(lo), float(hi)], "count": int(m.sum()),
                "x_rms_um":float(np.sqrt(np.mean(dx[m]**2))*1000),"y_rms_um":float(np.sqrt(np.mean(dy[m]**2))*1000),
                "point_rms_um": float(np.sqrt(np.mean(point[m] ** 2)) * 1000), "point_max_um": float(np.max(point[m]) * 1000)})
    return out


def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke",action="store_true",help="fit parity-preserving 21x21 sample; score full A=0 grid")
    parser.add_argument("--output-dir", "--out", dest="out", help="Output folder; default: runs/p4_a0_ladder (smoke: runs/p4_a0_ladder_smoke)")
    args=parser.parse_args(argv)
    out = Path(args.out).resolve() if args.out else ROOT / "runs" / ("p4_a0_ladder_smoke" if args.smoke else "p4_a0_ladder")
    if args.smoke and not out.name.endswith("_smoke"):
        out = out.with_name(out.name + "_smoke")
    out.mkdir(parents=True, exist_ok=True)
    (ROOT / "runs").mkdir(parents=True, exist_ok=True)
    run_id = out.name
    def log(msg: str):
        with (ROOT / "runs/subagent_status.log").open("a", encoding="utf-8") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()} [{run_id}] {msg}\n")
        print(msg, flush=True)
    log(f"run started pid={os.getpid()}")
    summary: dict[str, Any] = {"distance_label": DISTANCE_LABEL, "distance_mm": 167.93, "working_distance_mm": 0,
        "created_utc": datetime.now(timezone.utc).isoformat(), "output_directory": str(out), "input": str(PICKLE),
        "expected_input_sha256": EXPECTED_SHA256, "smoke": args.smoke,
        "script_sha256": _sha(Path(__file__).resolve()), "protected_e1_source_sha256": _sha(SCRIPT/"p4_a0_e1_fit.py")}
    try:
        digest = _sha(PICKLE)
        if digest.lower() != EXPECTED_SHA256.lower():
            raise ValueError(f"input SHA256 mismatch: {digest}")
        log(f"input hash verified; sha256={digest}")
        sys.path.insert(0, str(SCRIPT))
        p4 = _load("p4_fit_a0_validation", SCRIPT / "p4_fit.py")
        p1 = sys.modules["p1_camera_fit"]
        helpers = _load("p4_a0_e1_helpers", SCRIPT/"p4_a0_e1_fit.py")
        refpath = ROOT / "archive/2026-11-11/runs/p1_camera_refit_20261110T224852Z_cdb51f18/summary.json"
        p1_refit_max = json.loads(refpath.read_text(encoding="utf-8"))["metrics"]["holdout"]["point_max_um"]
        if not math.isclose(p1_refit_max, .001581, rel_tol=0, abs_tol=5e-7):
            raise ValueError(f"P1 comparison differs: {p1_refit_max}")
        all_a_path=ROOT/"archive/2026-11-11/runs/p4_fit_20261110T153705Z_4f23532d/summary.json"
        all_a=json.loads(all_a_path.read_text(encoding="utf-8"))
        all_models=all_a.get("camera_models",all_a.get("models",[]))
        all_e2=next((m for m in all_models if m.get("name")=="E2"),None)
        if all_e2 is None: raise ValueError("all-A reference has no E2 camera model")
        all_k2=float(all_e2["parameters"]["normalized"]["k2"]["value"])
        e1ref_path=ROOT/"archive/2026-11-11/runs/p4_a0_e1_20261110T231522Z_e65f3c61/summary.json"
        e1ref=json.loads(e1ref_path.read_text(encoding="utf-8"))
        e1_prior_max=float(e1ref["holdout_metrics"]["point_max_um"])
        rows = np.load(PICKLE, allow_pickle=True)
        full, validation = p4.validate_full(rows, digest)
        cube = full[0].copy()
        del rows, full
        summary["input_validation"] = validation
        log("full P4 validation passed; A0 slice selected")
        summary["accommodation_index"] = 0
        summary["accommodation_axis_confirmed"] = {"index":0,"accommodation_d":float(cube["accommodation_d"][0,0,0])}
        summary["protected_source_sha256"] = {n: _sha(SCRIPT / n) for n in ("p1_camera_fit.py", "p1_camera_refit.py", "p4_fit.py")}
        if float(cube["accommodation_d"][0, 0, 0]) != 0.0:
            raise ValueError("accommodation index 0 does not map to A=0")
        summary["a0_gate_details"] = {
            "max_abs_stop_origin_mm": max(float(np.max(np.abs(cube[n]))) for n in ("x16", "y16", "z16")),
            "max_lmn_unit_norm_error": float(np.max(np.abs(np.sqrt(cube["l16"]**2+cube["m16"]**2+cube["n16"]**2)-1))),
            "n16_negative_all_rays": bool(np.all(cube["n16"] < 0)),
            "min_abs_n16": float(np.min(np.abs(cube["n16"]))),
            "ray_status_zero_all_rays": bool(np.all(cube["rayrsi_status"] == 0)),
            "center_ray_status_zero_all_states": bool(np.all(cube["center_rayrsi_status"] == 0)),
            "sensor_z_ptp_mm": float(np.ptp(cube["zsi"])),
        }
        ai = np.broadcast_to(np.arange(N_ANGLE)[:, None, None], cube.shape)
        bi = np.broadcast_to(np.arange(N_ANGLE)[None, :, None], cube.shape)
        rays = np.broadcast_to(RAYS, cube.shape)
        train = (ai % 2 == 0) & (bi % 2 == 0)
        holdout = ~train
        fit_train = train.copy()
        sample = np.unique(np.rint(np.linspace(0, 100, 21)).astype(int)) if args.smoke else np.arange(N_ANGLE)
        if args.smoke: fit_train &= np.isin(ai, sample) & np.isin(bi, sample)
        if (np.count_nonzero(train[:, :, 0]), np.count_nonzero(holdout[:, :, 0])) != (2601, 7600):
            raise ValueError("split state counts differ from 2601/7600")
        if np.count_nonzero(train)!=2601*9 or np.count_nonzero(holdout)!=7600*9:
            raise ValueError("split does not contain complete nine-ray states")
        tx, ty = -cube["l16"]/cube["n16"], -cube["m16"]/cube["n16"]
        targetx, targety = cube["xsi"], cube["ysi"]
        T = float(max(np.max(np.abs(tx[fit_train])), np.max(np.abs(ty[fit_train]))))
        if not math.isfinite(T) or T<=0: raise ValueError(f"invalid training tangent scale T={T}")
        u, v = tx/T, ty/T
        fx = np.linalg.lstsq(np.column_stack((tx[fit_train], np.ones(np.count_nonzero(fit_train)))), targetx[fit_train], rcond=None)[0][0]
        fy = np.linalg.lstsq(np.column_stack((ty[fit_train], np.ones(np.count_nonzero(fit_train)))), targety[fit_train], rcond=None)[0][0]
        sign = {"fx_mm_per_physical_tangent": float(fx), "fy_mm_per_physical_tangent": float(fy),
            "fx_normalized": float(fx*T), "fy_normalized": float(fy*T), "same_sign": bool(np.sign(fx)==np.sign(fy)),
            "relative_magnitude_difference": float(abs(fx/fy-1)), "limit": 1e-3}
        if not sign["same_sign"] or sign["relative_magnitude_difference"] >= 1e-3:
            raise ValueError(f"independent fx/fy check failed: {sign}")
        e0 = p1._closed_e0(u[fit_train], v[fit_train], targetx[fit_train], targety[fit_train])
        prev = np.r_[e0, 0.]
        state = np.broadcast_to((ai[:, :, 0]*N_ANGLE+bi[:, :, 0])[:, :, None], cube.shape)
        levels = []
        arrays = {"distance_label":np.asarray(DISTANCE_LABEL),"coordinate_unit":np.asarray("mm"),"residual_unit":np.asarray("um"),"residual_direction":np.asarray("predicted minus traced"),"accommodation_index":np.asarray(0),
                  "T_training_tangent":np.asarray(T),"physical_tangent_radius":np.hypot(tx,ty),"ade_index":np.arange(N_ANGLE),"bde_index":np.arange(N_ANGLE),
                  "state_ids":(np.arange(N_ANGLE)[:,None]*N_ANGLE+np.arange(N_ANGLE)[None,:]),
                  "tx": tx, "ty": ty, "u": u, "v": v, "target_x": targetx, "target_y": targety,
                  "train_state": train[:, :, 0], "fit_train_state": fit_train[:, :, 0], "holdout_state": holdout[:, :, 0], "ray_index": rays.astype(np.uint8)}
        residuals = {}
        for level, name in zip(LEVELS, NAMES):
            initial=np.r_[e0,0.] if level==1 else np.r_[prev,np.zeros(p1.LEVEL_COUNTS[level]-len(prev))]
            log(f"{name} fit started; rays={np.count_nonzero(fit_train)}; T={T:.12g}")
            params, solver = p1._fit_level(level, initial, u[fit_train], v[fit_train], targetx[fit_train], targety[fit_train])
            prev = params.copy()
            px, py = p1._project(params, u, v, level)
            dx, dy = px-targetx, py-targety
            point = np.hypot(dx, dy)
            radius = np.hypot(tx, ty)
            J = p1._analytic_jacobian(params, u[fit_train], v[fit_train], level)
            r = np.empty(2*np.count_nonzero(fit_train)); r[0::2]=dx[fit_train]; r[1::2]=dy[fit_train]
            labels = [f"{n}_normalized" if n not in ("cx", "cy") else f"{n}_mm" for n in p1.PARAM_LABELS[:p1.LEVEL_COUNTS[level]]]
            cond = _condition(J, r, labels)
            cond["covariance_reliable"] = cond["rank"]==p1.LEVEL_COUNTS[level]
            splits = {s: _metrics(p1, helpers, px, py, targetx, targety, rays, state, radius, T, m, point) for s,m in (("train",train),("holdout",holdout))}
            levelrec = {"name": name, "role":"candidate" if level<3 else "floor diagnostic only",
                "distance_label": DISTANCE_LABEL, "level": level, "initial_parameters_normalized":initial.tolist(),
                "warm_start_from":None if level==1 else NAMES[level-2], "parameters": p1._serial_params(params,T,level),
                "solver": {"success": bool(solver.success), "status": int(solver.status), "message": str(solver.message), "nfev": int(solver.nfev), "njev": int(solver.njev), "optimality": float(solver.optimality)},
                "conditioning": cond, "metrics": splits, "fit_training_sse_mm2": float(r@r)}
            if params[0] < 0:
                levelrec["parameters"]["signed_f_convention"]="negative f retained; agreeing axes; no image flip applied"
            levels.append(levelrec)
            residuals[f"{name}_dx_um"] = dx*1000; residuals[f"{name}_dy_um"] = dy*1000; residuals[f"{name}_point_um"] = point*1000
            arrays[f"{name}_params_normalized"] = params
            arrays[f"{name}_prediction_x"] = px
            arrays[f"{name}_prediction_y"] = py
            log(f"{name} completed status={solver.status}; holdout max={splits['holdout']['point_max_um']:.9g} um")
        e1,e2=levels[:2]
        p2 = np.asarray(arrays["E2_params_normalized"])
        cond2=e2["conditioning"]
        k2=float(p2[4]); sig2=float(cond2["standard_errors"][4]); f=float(p2[0])
        cov_reliable=cond2["rank"]==len(p2) and math.isfinite(sig2) and sig2>0
        k2sig={"normalized_signed": k2, "normalized_abs": abs(k2), "physical_signed": k2/T**4,
            "physical_abs": abs(k2/T**4), "displacement_at_normalized_radius_1_signed_um": f*k2*1000,
            "displacement_at_normalized_radius_1_abs_um": abs(f*k2)*1000,
            "sigma_normalized": sig2, "sigma_physical": sig2/T**4,
            "significance_sigma": abs(k2)/sig2 if cov_reliable else None,"signed_k2_over_sigma":k2/sig2 if cov_reliable else None,"covariance_reliable":cov_reliable,
            "k1_k2_correlation": cond2["correlation_matrix"][3][4],
            "sign_matches_all_a":bool(np.sign(k2)==np.sign(all_k2)),"all_a_k2_normalized":all_k2}
        e1max=e1["metrics"]["holdout"]["point_max_um"]; e2max=e2["metrics"]["holdout"]["point_max_um"]
        e1rms=e1["metrics"]["holdout"]["euclidean_rms_um"]; e2rms=e2["metrics"]["holdout"]["euclidean_rms_um"]
        gates={"e2_holdout_max_le_0_025_um": e2max<=.025,
            "e2_holdout_train_max_ratio_below_1_5": e2max/max(e2["metrics"]["train"]["point_max_um"],1e-300)<1.5,
            "k2_at_least_10_sigma": bool(k2sig["significance_sigma"] is not None and k2sig["significance_sigma"]>=10)}
        solver_valid=bool(e2["solver"]["success"] and cond2["rank"]==len(p2) and cov_reliable)
        selected="E2" if all(gates.values()) and solver_valid else None
        diagnostics={"same_T_u_v_as_E1":True,"warm_starts":"E2=[E1,0]; E3=[E2,0,0]; E4=[E3,0,0]",
            "E3_E4_floor_diagnostics_only":True,"selected_level":selected,"smoke_acceptance_provisional":bool(args.smoke),
            "full_run_required_for_final_acceptance":bool(args.smoke),"e2_improves_e1_holdout_max":e2max<e1max,
            "e2_k2_nonzero":k2!=0.0,"e2_k2_sign_matches_all_a":bool(np.sign(k2)==np.sign(all_k2)),"e2_converged":bool(e2["solver"]["success"]),
            "tangent_arrays_shared_with_e1":True}
        if e2max>=e1max: diagnostics["if_no_improvement"]="Review nonzero k2/sign versus all-A plus warm-start and solver convergence above."
        if e2max>.025: diagnostics["above_threshold_review"]="Inspect equal-width and quantile physical-radius deciles, corner rays, and domain-edge residual; r^6 trend suggests k3 and edge-localized residual suggests corner extrapolation. No parameters added."
        e2point=residuals["E2_point_um"]/1000
        maxloc=np.unravel_index(np.argmax(np.where(holdout,e2point,-np.inf)),e2point.shape)
        dominant=max(((k,e2["metrics"]["holdout"]["ray_classes"][k]) for k in ("center_ray4","edge_rays_1_3_5_7","corner_rays_0_2_6_8")),key=lambda kv:kv[1]["point_max_um"])[0]
        outer=e2["metrics"]["holdout"]["radius_equal_width_deciles"]
        solver_valid=bool(e2["solver"]["success"] and cond2["rank"]==len(p2) and cov_reliable)
        summary.update({"status":"completed", "distance_context":DISTANCE_LABEL, "input_sha256":digest,
            "all_a_reference":{"source":str(all_a_path.relative_to(ROOT)),"source_sha256":_sha(all_a_path),"T_training_tangent":all_a.get("T_training_tangent"),"E2_k2_normalized":all_k2,"E2_k2_physical":all_e2["parameters"]["physical"]["k2"]["value"],"E2_holdout_max_um":all_e2["holdout"]["max_um"]},
            "prior_a0_e1_reference":{"source":str(e1ref_path.relative_to(ROOT)),"source_sha256":_sha(e1ref_path),"holdout_max_um":e1_prior_max},
            "p1_camera_refit_reference":{"source":str(refpath.relative_to(ROOT)),"source_sha256":_sha(refpath),"holdout_max_um":p1_refit_max},
            "p1_comparison_holdout_max_um":p1_refit_max, "split":{"train_states":int(train.sum()//9),"train_rays":int(train.sum()),"holdout_states":int(holdout.sum()//9),"holdout_rays":int(holdout.sum()),"fit_training_states":int(fit_train.sum()//9),"fit_training_rays":int(fit_train.sum()),"rule":"ADE and BDE both even train; smoke only restricts fit to parity-preserving 21x21 subgrid" if args.smoke else "ADE and BDE both even train; full fit uses all train states"},
            "T_training_tangent":T,"expected_T_context":0.0228425,"T_expected_difference":T-.0228425,
            "sign_check":sign,"levels":levels,"k2_significance":k2sig,
            "e1_e2_holdout_ratios":{"point_max_e1_over_e2":e1max/e2max if e2max else None,"point_rms_e1_over_e2":e1rms/e2rms if e2rms else None},
            "acceptance_gates":gates,"solver_valid":solver_valid,"acceptance_pass":all(gates.values()) and solver_valid,"selected_level":selected,
            "acceptance_context":{"predicted_e2_holdout_max_um":[.010,.018],"all_a_e2_holdout_max_um":.0175,"expected_improvement_ratio":[25,45],"corner_rays_expected_to_dominate":True},
            "diagnostics":{**diagnostics,"holdout_max_location":{"ade_index":int(maxloc[0]),"bde_index":int(maxloc[1]),"ray_index":int(maxloc[2])},
                "dominant_holdout_ray_class":dominant,"holdout_max_train_max_ratio":e2max/max(e2["metrics"]["train"]["point_max_um"],1e-300),
                "holdout_rms_train_rms_ratio":e2rms/e2["metrics"]["train"]["euclidean_rms_um"],
                "outer_equal_width_radius_decile":outer[-1] if outer else None,"outer_to_core_radius_rms_ratio":outer[-1]["point_rms_um"]/max(outer[0]["point_rms_um"],1e-300) if outer else None}})
        np.savez_compressed(out/"residuals.npz", **arrays, **residuals)
        for level in levels:
            name=level["name"]
            fig,axes=plt.subplots(3,3,figsize=(13,12),constrained_layout=True)
            values=residuals[f"{name}_point_um"]
            vmax=float(np.max(values))
            for ray,ax in enumerate(axes.flat):
                im=ax.imshow(values[:,:,ray],origin="lower",aspect="auto",vmin=0,vmax=vmax)
                ax.set_title(f"{DISTANCE_LABEL}\n{name} ray {ray} point residual (um)",fontsize=9)
                ax.set_xlabel("BDE index"); ax.set_ylabel("ADE index")
            fig.colorbar(im,ax=axes,label="um")
            fig.savefig(out/f"{name}_residual_maps.png",dpi=120); plt.close(fig)
        for name in ("E1","E2"):
            for coord,keycoord in (("x","dx"),("y","dy")):
                values=residuals[f"{name}_{keycoord}_um"]
                limit=float(np.max(np.abs(values)))
                fig,axes=plt.subplots(3,3,figsize=(13,12),constrained_layout=True)
                for ray,ax in enumerate(axes.flat):
                    im=ax.imshow(values[:,:,ray],origin="lower",aspect="auto",cmap="coolwarm",vmin=-limit,vmax=limit)
                    ax.set_title(f"{DISTANCE_LABEL}\n{name} ray {ray} signed {coord} residual (um)",fontsize=9)
                    ax.set_xlabel("BDE index"); ax.set_ylabel("ADE index")
                fig.colorbar(im,ax=axes,label="um")
                fig.savefig(out/f"{name}_signed_{coord}_maps.png",dpi=120); plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
        for name in ("E1","E2"):
            for split in ("train","holdout"):
                bins=next(level["metrics"][split]["radius_equal_width_deciles"] for level in levels if level["name"]==name)
                ax.plot([sum(b["range"])/2 for b in bins],[b["point_rms_um"] for b in bins],marker="o",label=f"{name} {split}")
        ax.set_title(f"{DISTANCE_LABEL} A=0 residual by physical tangent radius")
        ax.set_xlabel("Physical tangent radius"); ax.set_ylabel("Point RMS (um)"); ax.legend(); ax.grid(True,alpha=.3)
        fig.savefig(out/"E1_E2_radius_comparison.png",dpi=130); plt.close(fig)
    except Exception as exc:
        summary["status"]="failed";summary["failure"]={"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}
        log(f"failed: {type(exc).__name__}: {exc}")
    safe=helpers.json_safe(summary) if "helpers" in locals() else summary
    (out/"summary.json").write_text(json.dumps(safe,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    log(f"finished status={summary['status']}")
    return 0 if summary["status"]=="completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
