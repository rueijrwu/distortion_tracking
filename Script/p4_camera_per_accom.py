#!/usr/bin/env python3
"""Fit and compare P4 E0–E4 camera models independently by accommodation."""
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
N = 101
RAYS = np.arange(9, dtype=np.int64)
DEFAULT_INDICES = (0, 5, 10, 15, 20)
LEVEL_NAMES = {0: "E0", 1: "E1", 2: "E2", 3: "E3", 4: "E4"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


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


def _condition(jac: np.ndarray, residual: np.ndarray, labels: list[str]) -> dict[str, Any]:
    _, singular, vh = np.linalg.svd(jac, full_matrices=False)
    tol = np.finfo(float).eps * max(jac.shape) * singular[0]
    rank = int(np.count_nonzero(singular > tol))
    invs2 = np.zeros_like(singular)
    invs2[singular > tol] = 1.0 / singular[singular > tol] ** 2
    cov0 = (vh.T * invs2) @ vh
    dof = max(1, len(residual) - jac.shape[1])
    variance = float(residual @ residual / dof)
    cov = cov0 * variance
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    den = se[:, None] * se[None, :]
    corr = np.divide(cov, den, out=np.full_like(cov, np.nan), where=den > 0)
    scaled = jac / np.maximum(np.linalg.norm(jac, axis=0), 1e-300)
    return {"parameter_labels": labels, "rank": rank,
            "column_scaled_jacobian_condition_number": float(np.linalg.cond(scaled)),
            "standard_errors_normalized": se.tolist(), "covariance_normalized": cov.tolist(),
            "correlation_matrix": corr.tolist(), "residual_variance_mm2": variance,
            "residual_degrees_of_freedom": dof, "rss_mm2": float(residual @ residual)}


def _metrics(p1, helpers, px, py, tx, ty, rays, states, radius, T, mask, point):
    dx, dy = px - tx, py - ty
    base = p1._metrics(px[mask], py[mask], tx[mask], ty[mask], rays[mask], states[mask], (radius / T)[mask])
    out = {k: v for k, v in base.items() if not k.startswith("_")}
    out["point_max_um"] = float(np.max(point[mask]) * 1000)
    classes = {"center_ray4": [4], "edge_rays_1_3_5_7": [1, 3, 5, 7], "corner_rays_0_2_6_8": [0, 2, 6, 8]}
    classes.update({f"ray_{r}": [r] for r in range(9)})
    out["ray_classes"] = {}
    for name, ids in classes.items():
        m = mask & np.isin(RAYS, ids)[None, None, :]
        out["ray_classes"][name] = {"x_rms_um": float(np.sqrt(np.mean(dx[m]**2))*1000),
                                    "y_rms_um": float(np.sqrt(np.mean(dy[m]**2))*1000),
                                    "point_rms_um": float(np.sqrt(np.mean(point[m]**2))*1000),
                                    "point_max_um": float(np.max(point[m])*1000)}
    out["radius_equal_width_deciles"] = helpers.radius_metrics(dx, dy, radius, mask)
    vals = radius[mask]
    edges = np.quantile(vals, np.linspace(0, 1, 11))
    quant = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = mask & (radius >= lo) & ((radius <= hi) if i == 9 else (radius < hi))
        if np.any(m):
            quant.append({"range": [float(lo), float(hi)], "count_rays": int(m.sum()),
                          "x_rms_um": float(np.sqrt(np.mean(dx[m]**2))*1000),
                          "y_rms_um": float(np.sqrt(np.mean(dy[m]**2))*1000),
                          "point_rms_um": float(np.sqrt(np.mean(point[m]**2))*1000),
                          "point_max_um": float(np.max(point[m])*1000)})
    out["radius_quantile_deciles"] = quant
    return out


def _stats(p1, helpers, params, level, cube, T, mask, states, rays):
    tx = -cube["l16"] / cube["n16"]
    ty = -cube["m16"] / cube["n16"]
    px, py = p1._project(params, tx / T, ty / T, level)
    dx, dy = px - cube["xsi"], py - cube["ysi"]
    point = np.hypot(dx, dy)
    return px, py, dx, dy, point, {
        name: _metrics(p1, helpers, px, py, cube["xsi"], cube["ysi"], rays, states,
                       np.hypot(tx, ty), T, m, point)
        for name, m in mask.items()
    }


def _fit(p1, helpers, cube, T, train, fit_train, holdout, ai, bi, rays, smoke):
    tx, ty = -cube["l16"] / cube["n16"], -cube["m16"] / cube["n16"]
    x, y = cube["xsi"], cube["ysi"]
    u, v = tx / T, ty / T
    fx = np.linalg.lstsq(np.column_stack((tx[fit_train], np.ones(np.count_nonzero(fit_train)))), x[fit_train], rcond=None)[0][0]
    fy = np.linalg.lstsq(np.column_stack((ty[fit_train], np.ones(np.count_nonzero(fit_train)))), y[fit_train], rcond=None)[0][0]
    sign = {"fx": float(fx), "fy": float(fy), "same_sign": bool(np.sign(fx) == np.sign(fy)),
            "relative_magnitude_difference": float(abs(fx / fy - 1)), "limit": 1e-3}
    if not sign["same_sign"] or sign["relative_magnitude_difference"] >= 1e-3:
        raise ValueError(f"independent fx/fy check failed: {sign}")
    e0 = p1._closed_e0(u[fit_train], v[fit_train], x[fit_train], y[fit_train])
    previous = None
    states = np.broadcast_to((ai[:, :, 0] * N + bi[:, :, 0])[:, :, None], cube.shape)
    masks = {"train": train, "holdout": holdout}
    levels, arrays = [], {}
    for level in range(5):
        if level == 0:
            params = e0
            solver = None
        else:
            start = np.r_[e0, 0.] if level == 1 else np.r_[previous, np.zeros(p1.LEVEL_COUNTS[level] - len(previous))]
            params, solver = p1._fit_level(level, start, u[fit_train], v[fit_train], x[fit_train], y[fit_train])
        previous = params.copy()
        px, py = p1._project(params, u, v, level)
        dx, dy = px-x, py-y
        point = np.hypot(dx, dy)
        jac = p1._analytic_jacobian(params, u[fit_train], v[fit_train], level)
        residual = np.empty(2 * np.count_nonzero(fit_train)); residual[0::2] = dx[fit_train]; residual[1::2] = dy[fit_train]
        labels = list(p1.PARAM_LABELS[:p1.LEVEL_COUNTS[level]])
        cond = _condition(jac, residual, labels)
        if level >= 2:
            rec_corr = cond["correlation_matrix"][3][4]
            cond["k1_k2_correlation"] = rec_corr
        ser = p1._serial_params(params, T, level)
        # Keep the serial form and add standard errors in both parameter scales.
        for i, name in enumerate(labels):
            sn = cond["standard_errors_normalized"][i]
            power = {"f": 1, "k1": 2, "k2": 4, "thetas": 1, "phis": 1, "q1": 1, "q2": 1}.get(name, 0)
            ser["normalized"][name]["standard_error"] = sn
            ser["physical"][name]["standard_error"] = sn / T**power
            ser["physical"][name]["value"] = float(ser["physical"][name]["value"])
        metrics = {split: _metrics(p1, helpers, px, py, x, y, rays, states,
                                   np.hypot(tx, ty), T, mask, point)
                   for split, mask in masks.items()}
        rec = {"name": LEVEL_NAMES[level], "level": level,
               "role": "candidate" if level <= 2 else "floor diagnostic only",
               "distance_label": DISTANCE_LABEL, "parameters": ser,
               "conditioning": cond, "solver": None if solver is None else {
                   "success": bool(solver.success), "status": int(solver.status), "message": str(solver.message),
                   "nfev": int(solver.nfev), "njev": int(solver.njev), "optimality": float(solver.optimality)},
               "metrics": metrics, "fit_train_rays": int(fit_train.sum()),
               "fit_state_scope": "21x21 parity-preserving fit grid" if smoke else "all parity-even training states"}
        levels.append(rec)
        arrays[f"{LEVEL_NAMES[level]}_params_normalized"] = params
        arrays[f"{LEVEL_NAMES[level]}_prediction_x"] = px
        arrays[f"{LEVEL_NAMES[level]}_prediction_y"] = py
        arrays[f"{LEVEL_NAMES[level]}_dx_um"] = dx * 1000
        arrays[f"{LEVEL_NAMES[level]}_dy_um"] = dy * 1000
        arrays[f"{LEVEL_NAMES[level]}_point_um"] = point * 1000
    e2 = levels[2]
    e2p = np.asarray(arrays["E2_params_normalized"])
    sig = e2["conditioning"]["standard_errors_normalized"]
    k2_sigma = sig[4]
    k2_z = abs(float(e2p[4])) / k2_sigma if k2_sigma > 0 else None
    e2["k2_significance_sigma"] = k2_z
    e2["k1_k2_correlation"] = e2["conditioning"]["correlation_matrix"][3][4]
    e2["gate_details"] = {"holdout_max_le_0_025_um": e2["metrics"]["holdout"]["point_max_um"] <= .025,
        "holdout_train_max_ratio_lt_1_5": e2["metrics"]["holdout"]["point_max_um"] / max(e2["metrics"]["train"]["point_max_um"], 1e-300) < 1.5,
        "k2_ge_10_sigma": bool(k2_z is not None and k2_z >= 10),
        "solver_success": bool(e2["solver"]["success"]), "full_rank": e2["conditioning"]["rank"] == 5}
    return {"levels": levels, "arrays": arrays, "sign_check": sign, "T": T,
            "train": train, "fit_train": fit_train, "holdout": holdout, "states": states,
            "tangents": (tx, ty), "k2_z": k2_z}


def _physical(rec, name):
    return float(rec["levels"][2]["parameters"]["physical"][name]["value"])


def _physical_se(rec, name):
    return float(rec["levels"][2]["parameters"]["physical"][name]["standard_error"])


def _project_physical(params, tx, ty):
    f, cx, cy, k1, k2 = params[:5]
    r2 = tx*tx + ty*ty
    radial = 1 + k1*r2 + k2*r2*r2
    return cx + f*tx*radial, cy + f*ty*radial


def _fit_stat_se(p1, params, level, cube, T, mask, kind, cov):
    """Formal delta-method SE for deterministic holdout RMS or point maximum."""
    tx=-cube["l16"]/cube["n16"]; ty=-cube["m16"]/cube["n16"]
    u,v=tx/T,ty/T
    px,py=p1._project(params,u,v,level)
    dx,dy=px-cube["xsi"],py-cube["ysi"]
    jac=p1._analytic_jacobian(params,u[mask],v[mask],level).reshape(-1,2,len(params))
    cov=np.asarray(cov)
    if kind=="euclidean_rms_um":
        point=np.hypot(dx[mask],dy[mask]); rms=float(np.sqrt(np.mean(point**2)))
        if rms==0: return 0.0
        grad=np.sum(dx[mask,None]*jac[:,0,:]+dy[mask,None]*jac[:,1,:],axis=0)/(len(point)*rms)
    else:
        point=np.hypot(dx[mask],dy[mask]); imax=int(np.argmax(point))
        if point[imax]==0: return 0.0
        grad=(dx[mask][imax]*jac[imax,0,:]+dy[mask][imax]*jac[imax,1,:])/point[imax]
    return float(math.sqrt(max(float(grad@cov@grad),0))*1000)


def _physical_metrics(predx, predy, cube, mask, rays, states):
    dx, dy = predx-cube["xsi"], predy-cube["ysi"]
    point = np.hypot(dx, dy)
    base = {"x_rms_um": float(np.sqrt(np.mean(dx[mask]**2))*1000),
            "y_rms_um": float(np.sqrt(np.mean(dy[mask]**2))*1000),
            "euclidean_rms_um": float(np.sqrt(np.mean(point[mask]**2))*1000),
            "point_max_um": float(np.max(point[mask])*1000),
            "mean_signed_x_um": float(np.mean(dx[mask])*1000),
            "mean_signed_y_um": float(np.mean(dy[mask])*1000), "ray_classes": {}}
    classes = {"center_ray4": [4], "edge_rays_1_3_5_7": [1,3,5,7], "corner_rays_0_2_6_8": [0,2,6,8]}
    classes.update({f"ray_{r}": [r] for r in range(9)})
    for name, ids in classes.items():
        m = mask & np.isin(RAYS, ids)[None,None,:]
        base["ray_classes"][name] = {"point_max_um": float(np.max(point[m])*1000),
                                      "point_rms_um": float(np.sqrt(np.mean(point[m]**2))*1000)}
    return base


def _fixed_inside_outside(record, fixed_record, cube, T0, train, holdout):
    tx,ty=record["tangents"]
    domain=(np.abs(tx)>T0)|(np.abs(ty)>T0)
    out={}
    for split,mask in (("train",train),("holdout",holdout)):
        out[split]={}
        for name,region in (("inside_T0",~domain),("outside_T0",domain)):
            use=mask&region
            if np.any(use):
                dx=record["_fixed_px"]-cube["xsi"]; dy=record["_fixed_py"]-cube["ysi"]
                out[split][name]={"count_rays":int(use.sum()),
                    "euclidean_rms_um":float(np.sqrt(np.mean((dx[use]**2+dy[use]**2)))*1000),
                    "point_max_um":float(np.max(np.hypot(dx[use],dy[use]))*1000)}
            else: out[split][name]={"count_rays":0,"euclidean_rms_um":None,"point_max_um":None}
    return out


def _drift(a, zero, r, tangents):
    p0 = [_physical(zero,n) for n in ("f","cx","cy","k1","k2")]
    pa = [_physical(a,n) for n in ("f","cx","cy","k1","k2")]
    s0 = [_physical_se(zero,n) for n in ("f","cx","cy","k1","k2")]
    sa = [_physical_se(a,n) for n in ("f","cx","cy","k1","k2")]
    dc_mm=math.hypot(pa[1]-p0[1],pa[2]-p0[2])
    terms = {
        "d_f_um": (abs(pa[0]-p0[0])*r*1000, math.hypot(sa[0],s0[0])*r*1000),
        "d_k1_um": (abs(p0[0]*(pa[3]-p0[3]))*r**3*1000,
                    abs(r**3*1000)*math.hypot((pa[3]-p0[3])*s0[0],p0[0]*math.hypot(sa[3],s0[3]))),
        "d_k2_um": (abs(p0[0]*(pa[4]-p0[4]))*r**5*1000,
                    abs(r**5*1000)*math.hypot((pa[4]-p0[4])*s0[0],p0[0]*math.hypot(sa[4],s0[4]))),
        "d_c_um": (dc_mm*1000,0.0)}
    if a is not zero and dc_mm>0:
        cov0=np.asarray(zero["levels"][2]["conditioning"]["covariance_normalized"])
        cova=np.asarray(a["levels"][2]["conditioning"]["covariance_normalized"])
        # cx,cy are normalized and physical coordinates identically; use their covariance block.
        covxy=cov0[1:3,1:3]+cova[1:3,1:3]
        grad=np.asarray([pa[1]-p0[1],pa[2]-p0[2]])/dc_mm
        terms["d_c_um"]=(dc_mm*1000,math.sqrt(max(float(grad@covxy@grad),0))*1000)
    gate_value = max(terms[k][0] for k in ("d_f_um","d_k1_um","d_k2_um"))
    gate_se = terms[{"d_f_um":"d_f_um","d_k1_um":"d_k1_um","d_k2_um":"d_k2_um"}[max(("d_f_um","d_k1_um","d_k2_um"),key=lambda k:terms[k][0])]][1]
    tx, ty = tangents
    x0,y0 = _project_physical(p0,tx,ty); xa,ya = _project_physical(pa,tx,ty)
    point = np.hypot(xa-x0,ya-y0)
    imax = int(np.argmax(point))
    dtotal = float(point.flat[imax]*1000)
    if a is zero:
        dtotal_sigma = 0.0
    else:
        u0=np.asarray([tx.flat[imax]]); v0=np.asarray([ty.flat[imax]])
        # Convert the physical point to each fit's own normalized coordinates.
        T0=float(zero["T"]); Ta=float(a["T"])
        j0=a["_p1"]._analytic_jacobian(zero["arrays"]["E2_params_normalized"],u0/T0,v0/T0,2)
        ja=a["_p1"]._analytic_jacobian(a["arrays"]["E2_params_normalized"],u0/Ta,v0/Ta,2)
        s0=np.asarray(zero["levels"][2]["conditioning"]["standard_errors_normalized"])
        sa=np.asarray(a["levels"][2]["conditioning"]["standard_errors_normalized"])
        # Projection covariance in x/y, delta-method uncertainty of Euclidean displacement.
        cov0=np.asarray(zero["levels"][2]["conditioning"]["covariance_normalized"])
        cova=np.asarray(a["levels"][2]["conditioning"]["covariance_normalized"])
        j0xy=j0.reshape(2,5); jaxy=ja.reshape(2,5)
        c0=j0xy @ cov0 @ j0xy.T
        ca=jaxy @ cova @ jaxy.T
        delta=np.asarray([xa.flat[imax]-x0.flat[imax],ya.flat[imax]-y0.flat[imax]])
        norm=max(float(np.linalg.norm(delta)),1e-300)
        dtotal_sigma=float(math.sqrt(max(float((delta/norm) @ (c0+ca) @ (delta/norm)),0))*1000)
    return {**{k:{"value":v[0],"sigma":v[1]} for k,v in terms.items()},
            "gate_max_um":{"value":gate_value,"sigma":gate_se},"d_total_um":{"value":dtotal,"sigma":dtotal_sigma}}


def _trend(xs, ys, ses, name, derivative):
    xs, ys, ses = np.asarray(xs), np.asarray(ys), np.asarray(ses)
    if len(xs)<3: raise ValueError(f"weighted trend for {name} needs at least three accommodation points")
    ses=np.maximum(ses,max(float(np.max(np.abs(ses)))*1e-12,1e-15))
    X = np.column_stack((np.ones(len(xs)), xs))
    w = 1/ses**2
    cov = np.linalg.pinv(X.T @ (w[:,None]*X))
    beta = cov @ (X.T @ (w*ys))
    resid = ys-X@beta
    dof = max(1,len(xs)-2)
    chi = float(np.sum(w*resid**2)/dof)
    se = float(math.sqrt(max(cov[1,1],0))*math.sqrt(max(1,chi)))
    unweighted = np.linalg.lstsq(X,ys,rcond=None)[0]
    effect = float(beta[1]*4*derivative*1000)
    effect_se = float(se*4*abs(derivative)*1000)
    return {"parameter":name,"weighted_slope_per_D":float(beta[1]),"weighted_slope_se_per_D":se,
            "chi2_per_dof":chi,"unweighted_slope_per_D":float(unweighted[1]),
            "corner_effect_over_4D_um":effect,"corner_effect_sigma_um":effect_se,
            "significant":abs(effect)>3*effect_se and abs(effect)>.01}


def _gate_cube(cube):
    norm = np.sqrt(cube["l16"]**2+cube["m16"]**2+cube["n16"]**2)
    maxxy = float(np.max(np.hypot(cube["x16"],cube["y16"])))
    maxz = float(np.max(np.abs(cube["z16"])))
    return {"max_hypot_x16_y16_mm":maxxy,"max_abs_x16_mm":float(np.max(np.abs(cube["x16"]))),
            "max_abs_y16_mm":float(np.max(np.abs(cube["y16"]))),"max_abs_z16_mm":maxz,
            "max_lmn_unit_norm_error":float(np.max(np.abs(norm-1))),
            "n16_negative_all_rays":bool(np.all(cube["n16"]<0)),
            "ray_status_zero_all_rays":bool(np.all(cube["rayrsi_status"]==0)),
            "center_ray_status_zero_all_states":bool(np.all(cube["center_rayrsi_status"]==0)),
            "sensor_z_ptp_mm":float(np.ptp(cube["zsi"])),
            "G1_pass":bool(maxxy<=1e-6 and maxz<=1e-6 and np.max(np.abs(norm-1))<=1e-12 and
                           np.all(cube["n16"]<0) and np.all(cube["rayrsi_status"]==0) and
                           np.all(cube["center_rayrsi_status"]==0))}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-sha256",help="Expected input SHA-256; default: skip hash check and print the hash found")
    parser.add_argument("--accom-index",nargs="+",type=int,default=list(DEFAULT_INDICES))
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--output-dir", "--out", dest="out", help="Output folder; default: runs/p4_camera_per_accom (smoke: runs/p4_camera_per_accom_smoke)")
    args=parser.parse_args(argv)
    indices=list(dict.fromkeys(args.accom_index))
    if 0 not in indices: parser.error("--accom-index must include 0")
    if any(i not in range(21) for i in indices): parser.error("accommodation indices must be 0..20")
    indices=[0]+sorted(i for i in indices if i!=0)
    out=Path(args.out).resolve() if args.out else ROOT/"runs"/("p4_camera_per_accom_smoke" if args.smoke else "p4_camera_per_accom")
    if args.smoke and not out.name.endswith("_smoke"):
        out=out.with_name(out.name+"_smoke")
    runs=(ROOT/"runs").resolve()
    runs.mkdir(parents=True,exist_ok=True)
    out.mkdir(parents=True,exist_ok=True)
    run_id=out.name
    def log(msg):
        with (runs/"subagent_status.log").open("a",encoding="utf-8") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()} [{run_id}] {DISTANCE_LABEL}; {msg}\n")
        print(f"{DISTANCE_LABEL}; {msg}",flush=True)
    log(f"run started pid={os.getpid()}; smoke={args.smoke}; indices={indices}")
    summary={"distance_label":DISTANCE_LABEL,"distance_mm":167.93,"working_distance_mm":0,
             "output_directory":str(out),"input":str(PICKLE),"expected_input_sha256":args.expected_sha256,
             "created_utc":datetime.now(timezone.utc).isoformat(),"smoke":args.smoke,
             "requested_indices":indices,"script_sha256":sha256(Path(__file__).resolve()),
             "index_interpretation":"only requested valid indices are analyzed; complete 0–4 D verdict requires all five default indices",
             "trend_uncertainty_methods":{"camera_parameters":"formal parameter covariance from Jacobian RSS/(2N-p)",
                 "T":"descriptive across-A scatter; T is an observed max with no fit covariance",
                 "holdout_RMS_max":"first-order delta method from E2 parameter covariance; max uses derivative at winning point"}}
    # Enforce an optional expected hash before loading pickle-derived data.
    digest=sha256(PICKLE)
    summary["input_sha256"]=digest
    print(f"Input SHA-256: {digest}; check {'enabled' if args.expected_sha256 is not None else 'skipped (no --expected-sha256)'}",flush=True)
    if args.expected_sha256 is not None and digest.lower()!=args.expected_sha256.lower():
        raise ValueError(f"input SHA-256 mismatch: {digest}")
    summary["input_sha256_check"]="verified" if args.expected_sha256 is not None else "skipped"
    sys.path.insert(0,str(SCRIPT))
    p4=_load("p4_fit_for_per_accom",SCRIPT/"p4_fit.py")
    p1=_load("p1_camera_fit_for_per_accom",SCRIPT/"p1_camera_fit.py")
    helpers=_load("p4_a0_e1_helpers_for_per_accom",SCRIPT/"p4_a0_e1_fit.py")
    safe_json=helpers.json_safe
    summary["protected_source_sha256"]={n:sha256(SCRIPT/n) for n in ("p1_camera_fit.py","p4_fit.py","p4_a0_e1_fit.py","p4_a0_camera_ladder.py")}
    rows=np.load(PICKLE,allow_pickle=True)
    full,validation=p4.validate_full(rows,digest,args.expected_sha256)
    summary["input_validation"]=validation
    del rows
    ai=np.broadcast_to(np.arange(N)[:,None,None],(N,N,9))
    bi=np.broadcast_to(np.arange(N)[None,:,None],(N,N,9))
    rays=np.broadcast_to(RAYS,(N,N,9))
    train=(ai%2==0)&(bi%2==0); holdout=~train
    if (np.count_nonzero(train[:,:,0]),np.count_nonzero(holdout[:,:,0]))!=(2601,7600): raise ValueError("split state counts differ")
    if np.count_nonzero(train)!=2601*9 or np.count_nonzero(holdout)!=7600*9: raise ValueError("split ray counts differ")
    sample=np.unique(np.rint(np.linspace(0,100,21)).astype(int))
    fitmask=train & np.isin(ai,sample) & np.isin(bi,sample) if args.smoke else train.copy()
    records={}; residuals={"distance_label":np.asarray(DISTANCE_LABEL),"indices":np.asarray(indices),"train_state":train[:,:,0],"holdout_state":holdout[:,:,0]}
    try:
        invalid_indices={}
        for index in indices:
            cube=full[index].copy()
            if not np.allclose(cube["accommodation_d"],0.2*index,rtol=0,atol=1e-8): raise ValueError(f"index {index} accommodation_d mismatch")
            gate=_gate_cube(cube)
            if not gate["G1_pass"]:
                invalid_indices[index]={"distance_label":DISTANCE_LABEL,"accommodation_index":index,"accommodation_d":.2*index,"status":"invalid_G1","gates":gate}
                log(f"index {index} invalid G1; no fit performed")
                continue
            tx,ty=-cube["l16"]/cube["n16"],-cube["m16"]/cube["n16"]
            T=float(max(np.max(np.abs(tx[train])),np.max(np.abs(ty[train]))))
            if not math.isfinite(T) or T<=0: raise ValueError(f"index {index}: invalid T={T}")
            log(f"index {index} A={0.2*index:g} fit starting T={T:.12g}")
            fit=_fit(p1,helpers,cube,T,train,fitmask,holdout,ai,bi,rays,args.smoke)
            record={"distance_label":DISTANCE_LABEL,"accommodation_index":index,"accommodation_d":0.2*index,
                    "T_training_tangent":T,"T_over_T0_minus_1":None,"gates":gate,
                    "levels":fit["levels"],"arrays":fit["arrays"],"sign_check":fit["sign_check"],"T":T,
                    "train":fit["train"],"fit_train":fit["fit_train"],"holdout":fit["holdout"],
                    "states":fit["states"],"tangents":fit["tangents"],"k2_z":fit["k2_z"]}
            record["_p1"]=p1
            records[index]=record
            for k,v in fit["arrays"].items(): residuals[f"a{index}_{k}"]=v
            log(f"index {index} fit complete E2 holdout max={fit['levels'][2]['metrics']['holdout']['point_max_um']:.8g} um")
        summary["invalid_indices"]=invalid_indices
        if 0 not in records:
            summary["status"]="failed"
            summary["failure"]={"type":"InvalidReference","message":"A=0 failed G1; no fits attempted for the reference"}
            summary["verdict"]="invalid reference geometry; no camera verdict available"
            (out/"summary.json").write_text(json.dumps(safe_json(summary),indent=2,allow_nan=False)+"\n",encoding="utf-8")
            raise ValueError("A=0 failed G1; cannot establish reference camera")
        zero=records[0]; T0=zero["T"]
        for i,r in records.items(): r["T_over_T0_minus_1"]=r["T"]/T0-1
        # Fixed physical camera and reciprocal diagnostic.
        fixed={}; drifts={}; matched={}
        union_tx=np.concatenate([r["tangents"][0].ravel() for r in records.values()]); union_ty=np.concatenate([r["tangents"][1].ravel() for r in records.values()])
        Tmax=max(r["T"] for r in records.values())
        for i,r in records.items():
            cube=full[i].copy(); tx,ty=r["tangents"]
            p0=[_physical(zero,n) for n in ("f","cx","cy","k1","k2")]
            px,py=_project_physical(p0,tx,ty)
            r["_fixed_px"],r["_fixed_py"]=px,py
            fixed[i]={"distance_label":DISTANCE_LABEL,"train":_physical_metrics(px,py,cube,train,rays,r["states"]),
                      "holdout":_physical_metrics(px,py,cube,holdout,rays,r["states"])}
            if i==0:
                drifts[i]={"d_f_um":{"value":0.0,"sigma":0.0},"d_k1_um":{"value":0.0,"sigma":0.0},
                           "d_k2_um":{"value":0.0,"sigma":0.0},"d_c_um":{"value":0.0,"sigma":0.0},
                           "gate_max_um":{"value":0.0,"sigma":0.0},"d_total_um":{"value":0.0,"sigma":0.0},
                           "at_max_T":{"d_f_um":{"value":0.0,"sigma":0.0},"d_k1_um":{"value":0.0,"sigma":0.0},"d_k2_um":{"value":0.0,"sigma":0.0},"d_c_um":{"value":0.0,"sigma":0.0},"gate_max_um":{"value":0.0,"sigma":0.0},"d_total_um":{"value":0.0,"sigma":0.0}},
                           "at_diagonal_sqrt2_T0":{"d_f_um":{"value":0.0,"sigma":0.0},"d_k1_um":{"value":0.0,"sigma":0.0},"d_k2_um":{"value":0.0,"sigma":0.0},"d_c_um":{"value":0.0,"sigma":0.0},"gate_max_um":{"value":0.0,"sigma":0.0},"d_total_um":{"value":0.0,"sigma":0.0}},
                           "at_diagonal_sqrt2_Tmax":{"d_f_um":{"value":0.0,"sigma":0.0},"d_k1_um":{"value":0.0,"sigma":0.0},"d_k2_um":{"value":0.0,"sigma":0.0},"d_c_um":{"value":0.0,"sigma":0.0},"gate_max_um":{"value":0.0,"sigma":0.0},"d_total_um":{"value":0.0,"sigma":0.0}},"d_f_at_r_T0_radial_um":{"value":0.0,"sigma":0.0}}
                matched[i]={"distance_label":DISTANCE_LABEL,"common_box_tangent_limit":T0,"A0_train_rays":int(train.sum()),"A_train_rays":int(train.sum()),"T_normalization":T0,"drift":{"d_f_um":{"value":0.0,"sigma":0.0},"d_k1_um":{"value":0.0,"sigma":0.0},"d_k2_um":{"value":0.0,"sigma":0.0},"d_c_um":{"value":0.0,"sigma":0.0},"gate_max_um":{"value":0.0,"sigma":0.0},"d_total_um":{"value":0.0,"sigma":0.0}},"d_total_um":0.0,"E2_fit_params_A0_normalized":zero["arrays"]["E2_params_normalized"].tolist(),"E2_fit_params_A_normalized":zero["arrays"]["E2_params_normalized"].tolist(),"E1_solver_A0":{"success":True,"status":0},"E2_solver_A0":{"success":True,"status":0},"E1_solver_A":{"success":True,"status":0},"E2_solver_A":{"success":True,"status":0}}
            else:
                # Own camera on A=0 data.
                aparams=[_physical(r,n) for n in ("f","cx","cy","k1","k2")]
                c0=full[0].copy(); tx0,ty0=zero["tangents"]
                qx,qy=_project_physical(aparams,tx0,ty0)
                fixed[i]["converse_on_A0"]={"train":_physical_metrics(qx,qy,c0,train,rays,zero["states"]),"holdout":_physical_metrics(qx,qy,c0,holdout,rays,zero["states"])}
                drifts[i]=_drift(r,zero,T0,(union_tx,union_ty))
                drifts[i]["at_max_T"]=_drift(r,zero,Tmax,(union_tx,union_ty))
                drifts[i]["at_diagonal_sqrt2_T0"]=_drift(r,zero,math.sqrt(2)*T0,(union_tx,union_ty))
                drifts[i]["at_diagonal_sqrt2_Tmax"]=_drift(r,zero,math.sqrt(2)*Tmax,(union_tx,union_ty))
                drifts[i]["d_f_at_r_T0_radial_um"]={"value":drifts[i]["d_f_um"]["value"]*abs(1+_physical(zero,"k1")*T0**2+_physical(zero,"k2")*T0**4),"sigma":drifts[i]["d_f_um"]["sigma"]*abs(1+_physical(zero,"k1")*T0**2+_physical(zero,"k2")*T0**4)}
                # Same T0 normalization and common axis-aligned tangent box.
                lim=min(T0,r["T"])
                c0=full[0].copy(); ca=cube
                t0x,t0y=zero["tangents"]
                m0=train & (np.abs(t0x)<=lim) & (np.abs(t0y)<=lim)
                ma=train & (np.abs(tx)<=lim) & (np.abs(ty)<=lim)
                def matched_fit(c, m):
                    ux,uy=-c["l16"]/c["n16"]/T0,-c["m16"]/c["n16"]/T0
                    init=p1._closed_e0(ux[m],uy[m],c["xsi"][m],c["ysi"][m])
                    e1,s1=p1._fit_level(1,np.r_[init,0.],ux[m],uy[m],c["xsi"][m],c["ysi"][m])
                    e2,s2=p1._fit_level(2,np.r_[e1,0.],ux[m],uy[m],c["xsi"][m],c["ysi"][m])
                    px,py=p1._project(e2,ux[m],uy[m],2)
                    res=np.empty(2*int(m.sum())); res[0::2]=px-c["xsi"][m]; res[1::2]=py-c["ysi"][m]
                    jac=p1._analytic_jacobian(e2,ux[m],uy[m],2)
                    cond=_condition(jac,res,list(p1.PARAM_LABELS[:5]))
                    return e2,s1,s2,cond
                if args.smoke:
                    m0 &= np.isin(ai,sample) & np.isin(bi,sample)
                    ma &= np.isin(ai,sample) & np.isin(bi,sample)
                mp0,s10,s20,cnd0=matched_fit(c0,m0); mpa,s1a,s2a,cnda=matched_fit(ca,ma)
                mpa_phys=[mpa[0]/T0,mpa[1],mpa[2],mpa[3]/T0**2,mpa[4]/T0**4]
                mp0_phys=[mp0[0]/T0,mp0[1],mp0[2],mp0[3]/T0**2,mp0[4]/T0**4]
                mpx,mpy=_project_physical(mpa_phys,union_tx,union_ty)
                m0x,m0y=_project_physical(mp0_phys,union_tx,union_ty)
                matched0={"T":T0,"levels":[None,None,{"parameters":{"physical":{
                    "f":{"value":mp0[0]/T0,"standard_error":cnd0["standard_errors_normalized"][0]/T0},
                    "cx":{"value":mp0[1],"standard_error":cnd0["standard_errors_normalized"][1]},
                    "cy":{"value":mp0[2],"standard_error":cnd0["standard_errors_normalized"][2]},
                    "k1":{"value":mp0[3]/T0**2,"standard_error":cnd0["standard_errors_normalized"][3]/T0**2},
                    "k2":{"value":mp0[4]/T0**4,"standard_error":cnd0["standard_errors_normalized"][4]/T0**4}}},"conditioning":cnd0}],"arrays":{"E2_params_normalized":mp0},"_p1":p1}
                matcheda={"T":T0,"levels":[None,None,{"parameters":{"physical":{
                    "f":{"value":mpa[0]/T0,"standard_error":cnda["standard_errors_normalized"][0]/T0},
                    "cx":{"value":mpa[1],"standard_error":cnda["standard_errors_normalized"][1]},
                    "cy":{"value":mpa[2],"standard_error":cnda["standard_errors_normalized"][2]},
                    "k1":{"value":mpa[3]/T0**2,"standard_error":cnda["standard_errors_normalized"][3]/T0**2},
                    "k2":{"value":mpa[4]/T0**4,"standard_error":cnda["standard_errors_normalized"][4]/T0**4}}},"conditioning":cnda}],"arrays":{"E2_params_normalized":mpa},"_p1":p1}
                matched_delta=_drift(matcheda,matched0,T0,(union_tx,union_ty))
                matched[i]={"distance_label":DISTANCE_LABEL,"common_box_tangent_limit":lim,"A0_train_rays":int(m0.sum()),"A_train_rays":int(ma.sum()),
                            "T_normalization":T0,"drift":matched_delta,"d_total_um":matched_delta["d_total_um"]["value"],
                            "E2_fit_params_A0_normalized":mp0.tolist(),"E2_fit_params_A_normalized":mpa.tolist(),
                            "E1_solver_A0":{"success":bool(s10.success),"status":int(s10.status)},"E2_solver_A0":{"success":bool(s20.success),"status":int(s20.status)},
                            "E1_solver_A":{"success":bool(s1a.success),"status":int(s1a.status)},"E2_solver_A":{"success":bool(s2a.success),"status":int(s2a.status)}}
        # Strip runtime helpers and non-JSON arrays from the persisted records.
        persisted_records={}
        for i,r in records.items():
            persisted_records[i]={k:v for k,v in r.items() if k not in ("_p1","_fixed_px","_fixed_py","arrays","train","fit_train","holdout","states","tangents")}
        summary.update({"status":"completed","per_accommodation":persisted_records,"fixed_camera":fixed,"drift":drifts,"matched_domain":matched})
        # Weighted trend lines and formal corner sensitivities at r=T0.
        fit_indices=[i for i in indices if i in records]
        if len(fit_indices)<3: raise ValueError("at least three valid indices are required for trends")
        xs=[0.2*i for i in fit_indices]; trend=[]
        for name in ("f","cx","cy","k1","k2"):
            vals=[_physical(records[i],name) for i in fit_indices]; ses=[_physical_se(records[i],name) for i in fit_indices]
            fp0=_physical(zero,"f"); k10=_physical(zero,"k1"); k20=_physical(zero,"k2")
            derivative={"f":T0*(1+k10*T0**2+k20*T0**4),"cx":1.,"cy":1.,"k1":abs(fp0)*T0**3,"k2":abs(fp0)*T0**5}[name]
            trend.append(_trend(xs,vals,ses,name,derivative))
        T_se=max(float(np.std([records[j]["T"] for j in fit_indices],ddof=1)),1e-12)
        ttrend=_trend(xs,[records[i]["T"] for i in fit_indices],[T_se]*len(fit_indices),"T",
            _physical(zero,"f")*(1+3*_physical(zero,"k1")*T0**2+5*_physical(zero,"k2")*T0**4))
        ttrend["uncertainty_method"]="observed across-A T scatter; T is an exact domain scale, not a camera parameter"
        ttrend["corner_effect_interpretation"]="descriptive sensor movement from changing tangent-domain edge at r=T0; not included in G6"
        trend.append(ttrend)
        for key in ("euclidean_rms_um","point_max_um"):
            vals=[records[i]["levels"][2]["metrics"]["holdout"][key] for i in fit_indices]
            ses=[_fit_stat_se(p1,records[i]["arrays"]["E2_params_normalized"],2,full[i],records[i]["T"],records[i]["holdout"],key,records[i]["levels"][2]["conditioning"]["covariance_normalized"]) for i in fit_indices]
            ses=[max(s,1e-12) for s in ses]
            tr=_trend(xs,vals,ses,f"holdout_{key}",.001)
            tr["uncertainty_method"]="first-order delta method from E2 parameter covariance; max uses the winning point's local derivative"
            trend.append(tr)
        summary["trends"]=trend
        # Per-A and across-A gates.
        per_gates={}
        for i,r in records.items():
            e2=r["levels"][2]; g=e2["gate_details"]
            g["G1_pass"]=r["gates"]["G1_pass"]; g["G2_pass"]=all(g[k] for k in ("holdout_max_le_0_025_um","holdout_train_max_ratio_lt_1_5","solver_success","full_rank")); g["G3_pass"]=g["k2_ge_10_sigma"]
            g["G4_pass"]=i==0 or drifts[i]["gate_max_um"]["value"]<=.01
            g["G5_pass"]=fixed[i]["holdout"]["point_max_um"]<=.025
            per_gates[i]=g
        summary["gates"]=per_gates
        g6_tests=[t for t in trend if t["parameter"] in ("f","cx","cy","k1","k2")]
        g6=not any(abs(t["corner_effect_over_4D_um"])>.01 and abs(t["corner_effect_over_4D_um"])>3*t["corner_effect_sigma_um"] for t in g6_tests)
        for t in g6_tests: t["significant"] = abs(t["corner_effect_over_4D_um"])>.01 and abs(t["corner_effect_over_4D_um"])>3*t["corner_effect_sigma_um"]
        for g in per_gates.values(): g["G6_pass"]=g6
        fit_indices=[i for i in indices if i in records]
        summary["triage"]={"significant_parameter_trends":g6_tests,
            "domain_matched_control_d_total_um":{str(i):v["d_total_um"] for i,v in matched.items()},
            "T_trend_monotonic":bool(all(np.diff([records[i]["T"] for i in fit_indices])>=0) or all(np.diff([records[i]["T"] for i in fit_indices])<=0)),
            "fixed_camera_holdout_max_by_A_um":{str(i):fixed[i]["holdout"]["point_max_um"] for i in fit_indices},
            "outer_domain_exceeding_T0_ray_share":{str(i):float(np.mean((np.abs(records[i]["tangents"][0])>T0)|(np.abs(records[i]["tangents"][1])>T0))) for i in fit_indices},
            "fixed_residual_inside_outside_T0_um":{str(i):_fixed_inside_outside(records[i],fixed[i],full[i],T0,train,holdout) for i in fit_indices},
            "E2_E3_E4_holdout_max_um":{str(i):{f"E{level}":records[i]["levels"][level]["metrics"]["holdout"]["point_max_um"] for level in (2,3,4)} for i in fit_indices}}
        complete_domain=set(indices)==set(DEFAULT_INDICES) and not invalid_indices and len(records)==len(DEFAULT_INDICES)
        overall=complete_domain and all(all(g.get(k,False) for k in ("G1_pass","G2_pass","G3_pass","G4_pass","G5_pass","G6_pass")) for g in per_gates.values())
        base_verdict=("single A-independent E2 camera (A=0 parameters) is valid for A = 0–4 D, 167.93 mm, zero working distance" if overall else "Acceptance criteria not all met or requested domain incomplete; per-A E2 remains diagnostic and §2 triage is required before interpreting any drift")
        summary["verdict"]=(f"PROVISIONAL --smoke result only; full run required. {base_verdict}" if args.smoke else base_verdict)
        summary["provisional_smoke_only"]=bool(args.smoke)
        # Compact final report tables.
        summary["table_A"]=[{"distance_label":DISTANCE_LABEL,"A_D":0.2*i,"idx":i,"T_a":records[i]["T"],"f_p":records[i]["levels"][2]["parameters"]["physical"]["f"],"cx_um":{"value":records[i]["levels"][2]["parameters"]["physical"]["cx"]["value"]*1000,"standard_error":records[i]["levels"][2]["parameters"]["physical"]["cx"]["standard_error"]*1000},"cy_um":{"value":records[i]["levels"][2]["parameters"]["physical"]["cy"]["value"]*1000,"standard_error":records[i]["levels"][2]["parameters"]["physical"]["cy"]["standard_error"]*1000},"k1_p":records[i]["levels"][2]["parameters"]["physical"]["k1"],"k2_p":records[i]["levels"][2]["parameters"]["physical"]["k2"],"train":records[i]["levels"][2]["metrics"]["train"],"holdout":records[i]["levels"][2]["metrics"]["holdout"],"E1_holdout_max_um":records[i]["levels"][1]["metrics"]["holdout"]["point_max_um"],"E4_floor_max_um":records[i]["levels"][4]["metrics"]["holdout"]["point_max_um"]} for i in fit_indices]
        summary["table_B"]=[{"distance_label":DISTANCE_LABEL,"idx":i,"drift":drifts.get(i),"fixed_camera_holdout":fixed[i]["holdout"],"gates":per_gates[i],"matched_domain":matched.get(i)} for i in fit_indices]
        summary["table_C"]=trend
        # Residual maps collapse the ray dimension by maximum magnitude.
        for i,r in records.items(): residuals[f"a{i}_E2_max_residual_um"]=np.max(r["arrays"]["E2_point_um"],axis=2)
        np.savez_compressed(out/"residuals.npz",**residuals)
        _plots(out,records,fit_indices,drifts,fixed,trend)
        (out/"summary.json").write_text(json.dumps(safe_json(summary),indent=2,allow_nan=False)+"\n",encoding="utf-8")
        _write_report(out,summary,fit_indices)
        log("JSON, NPZ, report, and plots complete")
    except Exception as exc:
        summary["status"]="failed"; summary["failure"]={"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}
        log(f"failed: {type(exc).__name__}: {exc}")
        (out/"summary.json").write_text(json.dumps(safe_json(summary),indent=2,allow_nan=False)+"\n",encoding="utf-8")
        raise
    (out/"summary.json").write_text(json.dumps(safe_json(summary),indent=2,allow_nan=False)+"\n",encoding="utf-8")
    log(f"finished status={summary['status']}")
    return 0


def _plots(out, records, indices, drifts, fixed, trends):
    As=[.2*i for i in indices]
    # Parameter versus accommodation with formal errors.
    fig,axs=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
    names=[("f","f_p (mm/tan)"),("cx","cx (mm)"),("cy","cy (mm)"),("k1","k1_p"),("k2","k2_p"),("T","T (tan)")]
    for ax,(name,label) in zip(axs.flat,names):
        if name=="T": vals=[records[i]["T"] for i in indices]; errs=[0]*len(vals)
        else: vals=[_physical(records[i],name) for i in indices]; errs=[_physical_se(records[i],name) for i in indices]
        ax.errorbar(As,vals,yerr=errs,marker="o",capsize=3); ax.set_title(f"{DISTANCE_LABEL}\n{label}"); ax.set_xlabel("Accommodation (D)"); ax.grid(True,alpha=.3)
    fig.savefig(out/"parameters_vs_accommodation.png",dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True)
    vals=[drifts.get(i,{}).get("gate_max_um",{}).get("value",0) for i in indices]
    ax.plot(As,vals,marker="o"); ax.axhline(.01,color="red",linestyle="--",label="0.01 µm gate"); ax.set_title(f"{DISTANCE_LABEL}\nE2 drift gate vs accommodation"); ax.set_xlabel("Accommodation (D)"); ax.set_ylabel("Gate max (µm)"); ax.grid(True,alpha=.3); ax.legend(); fig.savefig(out/"drift_vs_accommodation.png",dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True)
    vals=[fixed[i]["holdout"]["point_max_um"] for i in indices]
    ax.plot(As,vals,marker="o"); ax.axhline(.025,color="red",linestyle="--",label="0.025 µm gate"); ax.set_title(f"{DISTANCE_LABEL}\nFixed A=0 camera holdout maximum"); ax.set_xlabel("Accommodation (D)"); ax.set_ylabel("Point maximum (µm)"); ax.grid(True,alpha=.3); ax.legend(); fig.savefig(out/"fixed_camera_holdout_max.png",dpi=150); plt.close(fig)
    fig,axs=plt.subplots(1,len(indices),figsize=(6*len(indices),5),constrained_layout=True,squeeze=False)
    for ax,i in zip(axs.flat,indices):
        arr=np.max(records[i]["arrays"]["E2_point_um"],axis=2)
        im=ax.imshow(arr,origin="lower",extent=(-12.5,12.5,-12.5,12.5),aspect="equal"); ax.set_title(f"{DISTANCE_LABEL}\nA={.2*i:g} D E2 max ray residual (µm)",fontsize=9); ax.set_xlabel("BDE (deg)"); ax.set_ylabel("ADE (deg)"); fig.colorbar(im,ax=ax,shrink=.8,label="µm")
    fig.savefig(out/"E2_per_accommodation_residual_maps.png",dpi=140); plt.close(fig)


def _write_report(out, summary, indices):
    lines=[f"P4 per-accommodation E2 camera fit — {DISTANCE_LABEL}","", "Table A — per-A E2 own fit (all residuals in µm):",
           "A (D) | idx | T_a | f_p ± σ | cx ± σ (µm) | cy ± σ (µm) | k1_p ± σ | k2_p ± σ | train RMS/max | holdout RMS/max | corner/edge/center max | E1 holdout max | E4 floor max"]
    for i in indices:
        rec=summary["per_accommodation"][i]; e2=rec["levels"][2]; p=e2["parameters"]; tr=e2["metrics"]["train"]; ho=e2["metrics"]["holdout"]
        q=lambda n: f"{p['physical'][n]['value']*1000:.8g} ± {p['physical'][n]['standard_error']*1000:.3g}" if n in ("cx","cy") else f"{p['physical'][n]['value']:.8g} ± {p['physical'][n]['standard_error']:.3g}"
        classes=ho["ray_classes"]
        lines.append(f"{rec['accommodation_d']:.1f} | {i} | {rec['T']:.8g} | {q('f')} | {q('cx')} | {q('cy')} | {q('k1')} | {q('k2')} | {tr['euclidean_rms_um']:.5g}/{tr['point_max_um']:.5g} | {ho['euclidean_rms_um']:.5g}/{ho['point_max_um']:.5g} | {classes['corner_rays_0_2_6_8']['point_max_um']:.5g}/{classes['edge_rays_1_3_5_7']['point_max_um']:.5g}/{classes['center_ray4']['point_max_um']:.5g} | {rec['levels'][1]['metrics']['holdout']['point_max_um']:.5g} | {rec['levels'][4]['metrics']['holdout']['point_max_um']:.5g}")
    lines += ["", "Table B — drift vs A=0 and fixed camera:", "A (D) | df | dk1 | dk2 | gate max ±σ | dc | d_total | matched d_total | fixed holdout RMS/max | G2 | G4 | G5"]
    for i in indices:
        d=summary["drift"].get(i); fixed=summary["fixed_camera"][i]["holdout"]; g=summary["gates"][i]
        if d is None: values="0 | 0 | 0 | 0 ± 0 | 0 | 0 | 0"
        else: values=f"{d['d_f_um']['value']:.5g} | {d['d_k1_um']['value']:.5g} | {d['d_k2_um']['value']:.5g} | {d['gate_max_um']['value']:.5g} ± {d['gate_max_um']['sigma']:.2g} | {d['d_c_um']['value']:.5g} | {d['d_total_um']['value']:.5g} | {summary['matched_domain'][i]['d_total_um']:.5g}"
        lines.append(f"{.2*i:.1f} | {values} | {fixed['euclidean_rms_um']:.5g}/{fixed['point_max_um']:.5g} | {g['G2_pass']} | {g['G4_pass']} | {g['G5_pass']}")
    lines += ["", "Table C — trends:", "parameter | slope per D ±σ | χ²/dof | corner effect over 4 D (µm) ±σ | significant?"]
    for t in summary["trends"]:
        lines.append(f"{t['parameter']} | {t['weighted_slope_per_D']:.6g} ± {t['weighted_slope_se_per_D']:.3g} | {t['chi2_per_dof']:.4g} | {t['corner_effect_over_4D_um']:.5g} ± {t['corner_effect_sigma_um']:.3g} | {t['significant']}")
    lines += ["", f"Verdict: {summary['verdict']}", f"Run directory: {out}", f"Input SHA-256: {summary['input_sha256']}", f"Geometry: {DISTANCE_LABEL}"]
    (out/"report.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":
    raise SystemExit(main())
