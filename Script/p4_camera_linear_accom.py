#!/usr/bin/env python3
"""Validate the proposed P4 E2 camera law from existing data, without CODE V.

Full mode uses all 21 levels. OLS and joint coefficients learn only from the
five integer-dioptre levels. --smoke reduces fitting rays but retains all
levels, full scoring, and full-training T; smoke cannot certify acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "Script"
PICKLE = ROOT / "data/p4_raytrace/p4_raytrace_251x251_12p5_A4.pkl"
# Identity of the archived baseline only, not the current input file.
BASELINE_INPUT_SHA256 = "aceb43c9744dec4898df558d4e21ab786dede141e464a47303b1dd13c0892c03"
BASELINE = ROOT / "archive/2026-11-11/runs/p4_camera_per_accom_20261110T235949Z_ea13bf95/summary.json"
BASELINE_SCRIPT_SHA256 = "07f3c5e3b5ec825dea3e6a79d8596e9ff514c297b1033f57241ae4b0d7949273"
ANCHORS = np.array([0, 5, 10, 15, 20], dtype=int)
INTERPOLATION = np.array([i for i in range(21) if i not in ANCHORS], dtype=int)
ACCOMMODATION = .2 * np.arange(21)
NAMES = ("f0", "f1", "k1_0", "k1_1", "k2_0", "k2_1")
UNITS = ("mm/tan", "mm/tan/D", "tan^-2", "tan^-2/D", "tan^-4", "tan^-4/D")
PROPOSED = np.array([425.137558059, 5.461957e-6, 2.943545178,
                     -8.055154e-5, -106.474146, .09274143])
MODELS = ("own", "constant_A0", "linear_ols", "linear_joint")
CLASSES = {"center": [4], "edges": [1, 3, 5, 7], "corners": [0, 2, 6, 8]}
PARAMETERS = ("f", "cx", "cy", "k1", "k2")
# Numerical rerun tolerances; V4 separately compares six significant digits.
V6_ATOL = {"T": 5e-14, "f": 5e-9, "cx": 1e-12, "cy": 1e-12,
           "k1": 5e-8, "k2": 5e-5, "point_max_um": 5e-7,
           "euclidean_rms_um": 5e-8}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_helper(name):
    spec = importlib.util.spec_from_file_location("_linear_accom_" + name, SCRIPT / (name + ".py"))
    if spec is None or spec.loader is None:
        raise ImportError(name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def finite(array, label):
    if not np.all(np.isfinite(array)):
        raise ValueError(f"nonfinite {label}")


def covariance(jac, residual):
    """Formal RSS/(n-p) covariance, with column scaling before the SVD."""
    finite(jac, "Jacobian")
    finite(residual, "residual")
    norms = np.linalg.norm(jac, axis=0)
    if np.any(norms == 0):
        raise ValueError("zero Jacobian column")
    _, singular, vh = np.linalg.svd(jac / norms, full_matrices=False)
    rank = int(np.count_nonzero(singular > np.finfo(float).eps * max(jac.shape) * singular[0]))
    if rank != jac.shape[1]:
        raise ValueError(f"rank-deficient camera Jacobian: {rank}/{jac.shape[1]}")
    dof = len(residual) - jac.shape[1]
    if dof <= 0:
        raise ValueError("nonpositive residual degrees of freedom")
    rss = float(residual @ residual)
    cov = ((vh.T / singular**2) @ vh) / norms[:, None] / norms[None, :] * (rss / dof)
    finite(cov, "covariance")
    return cov, {"rank": rank, "column_scaled_condition_number": float(singular[0] / singular[-1]),
                 "residual_degrees_of_freedom": dof, "rss": rss}


def solver_info(result):
    return {"success": bool(result.success), "status": int(result.status),
            "message": str(result.message), "nfev": int(result.nfev),
            "njev": None if result.njev is None else int(result.njev),
            "optimality": float(result.optimality)}


def normalized(physical, T):
    f, cx, cy, k1, k2 = physical
    return np.array([f*T, cx, cy, k1*T*T, k2*T**4])


def camera_at(coefficients, A):
    """Physical E2 parameters, with both centers fixed to zero."""
    f, k1, k2 = coefficients[0::2] + A * coefficients[1::2]
    return np.array([f, 0., 0., k1, k2])


def project(p1, physical, tx, ty, T):
    return p1._project(normalized(physical, T), tx/T, ty/T, 2)


def own_fit(p1, per_accom, cube, train, fitmask, smoke):
    tx, ty = -cube["l16"]/cube["n16"], -cube["m16"]/cube["n16"]
    gate = per_accom._gate_cube(cube)
    if not gate["G1_pass"]:
        raise ValueError(f"per-level geometry gate failed: {gate}")
    T = float(max(np.max(np.abs(tx[train])), np.max(np.abs(ty[train]))))
    if not np.isfinite(T) or T <= 0:
        raise ValueError(f"invalid training T: {T}")
    u, v = tx[fitmask]/T, ty[fitmask]/T
    x, y = cube["xsi"][fitmask], cube["ysi"][fitmask]
    # Match the original E0 -> E1 -> E2 ladder, including free cx and cy.
    e0 = p1._closed_e0(u, v, x, y)
    e1, s1 = p1._fit_level(1, np.r_[e0, 0.], u, v, x, y)
    e2, s2 = p1._fit_level(2, np.r_[e1, 0.], u, v, x, y)
    finite(e2, "own E2 parameters")
    px, py = p1._project(e2, u, v, 2)
    residual = np.column_stack((px-x, py-y)).ravel()
    cov, condition = covariance(p1._analytic_jacobian(e2, u, v, 2), residual)
    condition["rss_unit"] = "mm^2 (x and y sensor components)"
    conversion = np.array([1/T, 1., 1., T**-2, T**-4])
    physical = e2 * conversion
    physical_cov = cov * conversion[:, None] * conversion[None, :]
    se = np.sqrt(np.maximum(np.diag(physical_cov), 0.))
    fx = float(np.linalg.lstsq(np.column_stack((tx[fitmask], np.ones(len(x)))), x, rcond=None)[0][0])
    fy = float(np.linalg.lstsq(np.column_stack((ty[fitmask], np.ones(len(y)))), y, rcond=None)[0][0])
    sign_pass = bool(fx * fy > 0 and abs(fx/fy - 1) < 1e-3)
    k2_sigma = float(abs(physical[4])/se[4]) if se[4] > 0 else None
    rec = {"T_a": T, "geometry": gate, "own_parameters": {
        name: {"value": float(value), "standard_error": float(error)}
        for name, value, error in zip(PARAMETERS, physical, se)},
        "own_conditioning": condition, "own_solvers": {"E1": solver_info(s1), "E2": solver_info(s2)},
        "sign_check": {"fx": fx, "fy": fy, "passed": sign_pass},
        "k2_significance_sigma": k2_sigma, "fit_states": int(fitmask.sum()//9),
        "fit_rays": int(fitmask.sum()), "smoke": smoke}
    return physical, e2, physical_cov, rec


def ols_coefficients(own_parameters):
    """OLS on five own fits only, in physical tangent units."""
    X = np.column_stack((np.ones(5), ACCOMMODATION[ANCHORS]))
    Y = own_parameters[ANCHORS][:, [0, 3, 4]]
    B, _, rank, _ = np.linalg.lstsq(X, Y, rcond=None)
    if rank != 2:
        raise ValueError("OLS accommodation design is rank deficient")
    residual = Y - X @ B
    # Three line-residual df; retain between-parameter covariance.
    cov = np.kron(residual.T @ residual / 3, np.linalg.inv(X.T @ X))
    q = B.T.ravel()  # f0,f1,k1_0,k1_1,k2_0,k2_1
    finite(q, "OLS coefficients")
    finite(cov, "OLS covariance")
    return q, cov, residual


def joint_coefficients(p1, cube, records, fitmask, start):
    """Direct sensor-error fit of six coefficients using five training sets."""
    samples = []
    for i in ANCHORS:
        c = cube[i]
        samples.append((float(ACCOMMODATION[i]), records[i]["T_a"],
                        -c["l16"][fitmask]/c["n16"][fitmask],
                        -c["m16"][fitmask]/c["n16"][fitmask],
                        c["xsi"][fitmask], c["ysi"][fitmask]))
    # Analytic derivatives, coefficient scaling and um residuals keep the tiny
    # slopes visible to the optimizer and its gradient stopping criterion.
    scale = np.array([425., 1e-5, 3., 1e-4, 100., .1])

    def residual(z):
        q = z * scale
        parts = []
        for A, T, tx, ty, x, y in samples:
            px, py = project(p1, camera_at(q, A), tx, ty, T)
            parts.append(np.column_stack((px-x, py-y)).ravel()*1000)
        return np.concatenate(parts)

    def jacobian(z):
        q = z * scale
        parts = []
        for A, _, tx, ty, _, _ in samples:
            f, _, _, k1, k2 = camera_at(q, A)
            r2 = tx*tx + ty*ty
            radial = 1 + k1*r2 + k2*r2*r2
            j = np.empty((2*len(tx), 6))
            for offset, tangent in ((0, tx), (1, ty)):
                j[offset::2, 0] = tangent*radial
                j[offset::2, 1] = A*tangent*radial
                j[offset::2, 2] = f*tangent*r2
                j[offset::2, 3] = A*f*tangent*r2
                j[offset::2, 4] = f*tangent*r2*r2
                j[offset::2, 5] = A*f*tangent*r2*r2
            parts.append(j * scale * 1000)
        return np.concatenate(parts, axis=0)

    opt = least_squares(residual, start/scale, jac=jacobian, method="trf", x_scale="jac",
                        ftol=1e-14, xtol=1e-14, gtol=1e-14, max_nfev=2000)
    cov, condition = covariance(jacobian(opt.x), residual(opt.x))
    condition["rss_unit"] = "um^2 (x and y sensor components)"
    q = opt.x * scale
    finite(q, "joint coefficients")
    initial = residual(start/scale)
    return q, cov * scale[:, None] * scale[None, :], {
        "solver": solver_info(opt), "conditioning": condition,
        "OLS_start_training_rss_um2": float(initial @ initial),
        "training_indices": ANCHORS.tolist(), "fit_rays": int(fitmask.sum()*5)}


def metrics(dx_um, dy_um, state_mask):
    point = np.hypot(dx_um, dy_um)
    selected = point[state_mask, :]
    classes = dict(CLASSES)
    classes.update({f"ray_{i}": [i] for i in range(9)})
    return {"point_max_um": float(selected.max()),
            "euclidean_rms_um": float(np.sqrt(np.mean(selected**2))),
            "x_rms_um": float(np.sqrt(np.mean(dx_um[state_mask, :]**2))),
            "y_rms_um": float(np.sqrt(np.mean(dy_um[state_mask, :]**2))),
            "ray_classes": {name: {"point_max_um": float(point[..., ids][state_mask, :].max()),
                                      "euclidean_rms_um": float(np.sqrt(np.mean(point[..., ids][state_mask, :]**2)))}
                            for name, ids in classes.items()}}


def displacement(dx, dy, train_state):
    point = np.hypot(dx, dy)
    return {split: {"max_um": float(point[mask, :].max()),
                    "rms_um": float(np.sqrt(np.mean(point[mask, :]**2)))}
            for split, mask in (("all", np.ones(train_state.shape, dtype=bool)),
                                ("train", train_state), ("holdout", ~train_state))}


def score(p1, cube, records, own, ols, joint, train_state):
    residuals = {f"{name}_{axis}_um": np.empty(cube.shape, dtype=float)
                 for name in MODELS for axis in ("dx", "dy")}
    for i, rec in enumerate(records):
        c, T = cube[i], rec["T_a"]
        tx, ty = -c["l16"]/c["n16"], -c["m16"]/c["n16"]
        cameras = {"own": own[i], "constant_A0": own[0],
                   "linear_ols": camera_at(ols, ACCOMMODATION[i]),
                   "linear_joint": camera_at(joint, ACCOMMODATION[i])}
        rec["models"] = {}
        for name, params in cameras.items():
            px, py = project(p1, params, tx, ty, T)
            dx, dy = (px-c["xsi"])*1000, (py-c["ysi"])*1000
            finite(dx, name + " dx")
            finite(dy, name + " dy")
            residuals[f"{name}_dx_um"][i] = dx
            residuals[f"{name}_dy_um"][i] = dy
            rec["models"][name] = {split: metrics(dx, dy, mask)
                                   for split, mask in (("train", train_state), ("holdout", ~train_state))}
            if name != "own":
                # Compare sensor prediction VECTORS, not residual magnitudes.
                own_dx, own_dy = residuals["own_dx_um"][i], residuals["own_dy_um"][i]
                model = rec["models"][name]
                model["displacement_to_own"] = displacement(dx-own_dx, dy-own_dy, train_state)
                model["gap_to_own"] = {split: {
                    key: model[split][key]-rec["models"]["own"][split][key]
                    for key in ("point_max_um", "euclidean_rms_um")}
                    for split in ("train", "holdout")}
        rec["joint_to_ols_displacement"] = displacement(
            residuals["linear_joint_dx_um"][i]-residuals["linear_ols_dx_um"][i],
            residuals["linear_joint_dy_um"][i]-residuals["linear_ols_dy_um"][i], train_state)
        m = rec["models"]["own"]
        rec["original_E2_gate_details"] = {
            "geometry_G1": bool(rec["geometry"]["G1_pass"]),
            "same_focal_sign_and_relative_difference_lt_1e_3": rec["sign_check"]["passed"],
            "solver_success": all(s["success"] for s in rec["own_solvers"].values()),
            "full_rank_5": rec["own_conditioning"]["rank"] == 5,
            "k2_ge_10_sigma": bool(rec["k2_significance_sigma"] is not None and rec["k2_significance_sigma"] >= 10),
            "holdout_max_le_0_025_um": m["holdout"]["point_max_um"] <= .025,
            "holdout_train_max_ratio_lt_1_5": m["holdout"]["point_max_um"] /
                max(m["train"]["point_max_um"], 1e-300) < 1.5}
    return residuals


def baseline_comparison(records):
    result = {"path": str(BASELINE), "absolute_tolerances": V6_ATOL, "relative_tolerance": 0.,
              "identity_passed": False, "comparisons": [], "failed_indices": []}
    try:
        result["sha256"] = sha256(BASELINE)
        baseline = json.loads(BASELINE.read_text(encoding="utf-8-sig"))
        identity = {
            "historical_script": baseline["script_sha256"] == BASELINE_SCRIPT_SHA256,
            "reference_run": Path(baseline["output_directory"]).name == BASELINE.parent.name,
            "audited_pickle": baseline["input_sha256"] == BASELINE_INPUT_SHA256,
            "expected_pickle": baseline["expected_input_sha256"] == BASELINE_INPUT_SHA256,
            "full_run": baseline["status"] == "completed" and baseline["smoke"] is False,
            "five_indices": sorted(baseline["requested_indices"]) == ANCHORS.tolist(),
            "distance_167_93_mm": baseline["distance_mm"] == 167.93,
            "zero_working_distance": baseline["working_distance_mm"] == 0,
            "validated_input": baseline["input_validation"]["status"] == "passed"}
        result["identity"] = identity
        result["identity_passed"] = all(identity.values())
        result["reference_script_sha256"] = baseline["script_sha256"]
        for i in ANCHORS:
            ref, rec = baseline["per_accommodation"][str(i)], records[i]
            e2 = ref["levels"][2]
            checks = []

            def compare(name, actual, expected, tolerance):
                checks.append({"field": name, "actual": float(actual), "reference": float(expected),
                               "difference": float(actual-expected), "atol": tolerance,
                               "passed": bool(np.isfinite(actual) and np.isfinite(expected) and abs(actual-expected) <= tolerance)})

            compare("accommodation_index", rec["index"], ref["accommodation_index"], 0.)
            compare("accommodation_d", rec["A_D"], ref["accommodation_d"], 1e-8)
            compare("T", rec["T_a"], ref["T_training_tangent"], V6_ATOL["T"])
            for name in PARAMETERS:
                compare(name, rec["own_parameters"][name]["value"],
                        e2["parameters"]["physical"][name]["value"], V6_ATOL[name])
            for split in ("train", "holdout"):
                for name in ("point_max_um", "euclidean_rms_um"):
                    compare(split + "." + name, rec["models"]["own"][split][name],
                            e2["metrics"][split][name], V6_ATOL[name])
            passed = all(c["passed"] for c in checks)
            result["comparisons"].append({"index": int(i), "passed": passed, "fields": checks})
            if not passed:
                result["failed_indices"].append(int(i))
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["passed"] = bool(result["identity_passed"] and len(result["comparisons"]) == 5 and
                            not result["failed_indices"] and "error" not in result)
    return result


def acceptance(records, ols, baseline):
    gates = {}

    def per_level(name, criterion, values, limit, strict=False):
        values = [float(v) for v in values]
        failed = [i for i, value in enumerate(values)
                  if not np.isfinite(value) or (value >= limit if strict else value > limit)]
        gates[name] = {"criterion": criterion, "passed": not failed,
                       "limit": limit, "values_by_index": values, "failed_indices": failed}

    per_level("V1", "OLS linear holdout max <= 0.025 um at every level",
              [r["models"]["linear_ols"]["holdout"]["point_max_um"] for r in records], .025)
    per_level("V2", "OLS linear holdout max minus own holdout max <= 0.0005 um (signed)",
              [r["models"]["linear_ols"]["gap_to_own"]["holdout"]["point_max_um"] for r in records], .0005)
    per_level("V3", "OLS law-to-own vector displacement max <= 0.0005 um over ALL rays at every level",
              [r["models"]["linear_ols"]["displacement_to_own"]["all"]["max_um"] for r in records], .0005)
    checks = [{"parameter": name, "actual": float(value), "proposed": float(expected),
               "actual_6_significant_digits": format(value, ".6g"),
               "proposed_6_significant_digits": format(expected, ".6g"),
               "passed": bool(format(value, ".6g") == format(expected, ".6g"))}
              for name, value, expected in zip(NAMES, ols, PROPOSED)]
    gates["V4"] = {"criterion": "OLS six coefficients equal proposal when rounded to 6 significant digits",
                   "passed": all(c["passed"] for c in checks), "parameters": checks}
    per_level("V5", "abs(own cx) and abs(own cy) strictly below 1e-6 um",
              [max(abs(r["own_parameters"][n]["value"])*1000 for n in ("cx", "cy")) for r in records],
              1e-6, strict=True)
    gates["V6"] = {"criterion": "Five original levels reproduce ea13bf95 with declared V6 tolerances",
                   "passed": baseline["passed"], "failed_indices": baseline["failed_indices"],
                   "identity_passed": baseline["identity_passed"]}
    for name in ("V2", "V3"):
        failed = gates[name]["failed_indices"]
        gates[name]["fails_only_at_interpolation_levels"] = bool(failed and set(failed).issubset(INTERPOLATION.tolist()))
    return gates


def parameter_table(ols, ols_cov, joint, joint_cov):
    return [{"parameter": name, "unit": unit, "proposed": float(ref),
             "ols": float(a), "ols_standard_error": float(sa), "joint": float(b),
             "joint_standard_error": float(sb), "joint_minus_ols": float(b-a)}
            for name, unit, ref, a, sa, b, sb in zip(NAMES, UNITS, PROPOSED, ols,
                np.sqrt(np.maximum(np.diag(ols_cov), 0)), joint, np.sqrt(np.maximum(np.diag(joint_cov), 0)))]



def plots(out, summary, ols, joint):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    records = summary["per_accommodation"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for row, split in enumerate(("train", "holdout")):
        for col, metric in enumerate(("euclidean_rms_um", "point_max_um")):
            ax = axes[row, col]
            for name in MODELS:
                ax.plot(ACCOMMODATION, [r["models"][name][split][metric] for r in records], label=name)
            ax.set(xlabel="Accommodation (D)", ylabel="Sensor error (um)", title=f"{split}: {metric}")
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("167.93 mm, zero working distance")
    fig.savefig(out / "sensor_errors.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for name in MODELS[1:]:
        axes[0].plot(ACCOMMODATION, [r["models"][name]["displacement_to_own"]["all"]["max_um"] for r in records], label=name)
        axes[1].plot(ACCOMMODATION, [r["models"][name]["gap_to_own"]["holdout"]["point_max_um"] for r in records], label=name)
    for ax in axes:
        ax.axhline(.0005, color="black", linestyle="--", label="V2/V3 limit")
        ax.scatter(ACCOMMODATION[ANCHORS], np.zeros(5), marker="|", color="black", label="fit levels")
        ax.set(xlabel="Accommodation (D)", ylabel="um")
        ax.grid(alpha=.2)
    axes[0].set_title("Maximum prediction displacement to own fit (all rays)")
    axes[1].set_title("Holdout maximum minus own maximum (signed)")
    axes[0].legend(fontsize=8)
    fig.suptitle("167.93 mm, zero working distance")
    fig.savefig(out / "gaps_to_own.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for j, name in enumerate(("f", "k1", "k2")):
        values = np.array([r["own_parameters"][name]["value"] for r in records])
        offset = values[0]
        axes[j].plot(ACCOMMODATION, values-offset, ".", label="21 own fits")
        axes[j].plot(ACCOMMODATION[ANCHORS], values[ANCHORS]-offset, "o", fillstyle="none", label="five regression levels")
        for coeff, label in ((ols, "OLS"), (joint, "joint")):
            axes[j].plot(ACCOMMODATION, coeff[2*j]+coeff[2*j+1]*ACCOMMODATION-offset, label=label)
        axes[j].set(xlabel="Accommodation (D)", ylabel=f"{name} - A0 own value", title=name)
        axes[j].grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("167.93 mm, zero working distance")
    fig.savefig(out / "parameter_trends.png", dpi=160)
    plt.close(fig)


def report(summary):
    lines = ["P4 E2 camera linear in accommodation: 167.93 mm, zero working distance",
             "Output: " + summary["output_directory"], "Input SHA-256: " + summary["input_sha256"],
             "Mode: " + ("SMOKE: provisional; acceptance disabled" if summary["smoke"] else "full validation"),
             "Train = even ADE AND even BDE indices at EACH A; holdout = all remaining states.",
             "T_a = max(abs(tx),abs(ty)) over that A's FULL training rays, including in smoke mode.",
             "Own E2 has free cx/cy; constant retains A0 free centers; both laws fix cx=cy=0.",
             "OLS and joint fit learn only indices 0,5,10,15,20; other 16 are interpolation tests.",
             "No intermediate-level tuning, quadratic fallback, or CODE V. Sensor errors are Euclidean um.",
             "Smoke fitting uses ADE/BDE indices divisible by 10; full split is always scored.",
             "SEs are formal for deterministic data, not measurement-noise estimates.",
             "OLS SE: five-point line scatter with 3 df; joint/own SE: sensor RSS/(2N-p) Jacobian covariance.",
             "Slopes are effective E2 truncation values inside the traced tangent range, not physical camera drift.", "",
             "Six-parameter table (physical tangent units)",
             "parameter unit proposed OLS OLS_SE joint joint_SE joint_minus_OLS"]
    for row in summary["parameter_table"]:
        lines.append(f"{row['parameter']} {row['unit']} {row['proposed']:.12g} {row['ols']:.12g} "
                     f"{row['ols_standard_error']:.6g} {row['joint']:.12g} {row['joint_standard_error']:.6g} {row['joint_minus_ols']:.6g}")
    lines += ["", "V1-V6 (canonical OLS law)"]
    for name, gate in summary["gates"].items():
        lines.append(f"{name}: {'PASS' if gate['passed'] else 'FAIL'}; {gate['criterion']}; failed indices={gate.get('failed_indices', [])}")
    lines += ["V4 details: " + json.dumps(summary["gates"]["V4"]["parameters"], separators=(",", ":")),
              "V6 absolute tolerances (relative tolerance zero): " + json.dumps(V6_ATOL),
              "V6 baseline identity passed: " + str(summary["baseline_comparison"]["identity_passed"]),
              "Fit health passed: " + str(summary["fit_health_passed"]),
              "Accepted: " + str(summary["accepted"]), summary["interpolation_verdict"], "",
              "Per-A summary: errors in um; T in tangent units; f/k in physical units.",
              "R=regression anchor, I=interpolation test. Model columns are holdout RMS/max.",
              "idx A role T_a own_f own_k1 own_k2 cx_um cy_um own_RMS/max const_RMS/max OLS_RMS/max joint_RMS/max OLS_to_own_all_max OLS_minus_own_holdout_max"]
    for r in summary["per_accommodation"]:
        p, models = r["own_parameters"], r["models"]
        values = " ".join(f"{models[n]['holdout']['euclidean_rms_um']:.9g}/{models[n]['holdout']['point_max_um']:.9g}" for n in MODELS)
        lines.append(f"{r['index']:2d} {r['A_D']:.1f} {r['role'][0].upper()} {r['T_a']:.12g} "
                     f"{p['f']['value']:.12g} {p['k1']['value']:.12g} {p['k2']['value']:.12g} "
                     f"{p['cx']['value']*1000:.6g} {p['cy']['value']*1000:.6g} {values} "
                     f"{models['linear_ols']['displacement_to_own']['all']['max_um']:.9g} "
                     f"{models['linear_ols']['gap_to_own']['holdout']['point_max_um']:.9g}")
    lines += ["", "Per-A training RMS/max (own constant_A0 linear_ols linear_joint)"]
    for r in summary["per_accommodation"]:
        values = " ".join(f"{r['models'][n]['train']['euclidean_rms_um']:.9g}/{r['models'][n]['train']['point_max_um']:.9g}" for n in MODELS)
        lines.append(f"{r['A_D']:.1f} {values}")
    lines += ["", "Per-class maximum (um): A model split center edges corners"]
    for r in summary["per_accommodation"]:
        for name in MODELS:
            for split in ("train", "holdout"):
                classes = r["models"][name][split]["ray_classes"]
                lines.append(f"{r['A_D']:.1f} {name} {split} " + " ".join(f"{classes[c]['point_max_um']:.9g}" for c in CLASSES))
    lines += ["", "Joint-vs-OLS: " + json.dumps(summary["joint_comparison"], separators=(",", ":")),
              "V6 details: " + json.dumps(summary["baseline_comparison"], separators=(",", ":")),
              "Residuals: residuals.npz, axes [accommodation, ADE, BDE, ray], float64 dx/dy in um.",
              "No model change from proposal; additions: joint comparison, formal SE, fit-health checks, smoke mode."]
    return "\n".join(lines) + "\n"


def log_progress(out, message):
    print(message, flush=True)
    with (out / "run.log").open("a", encoding="utf-8") as stream:
        stream.write(message + "\n")


def run(args, out, summary):
    digest = sha256(PICKLE)
    summary["input_sha256"] = digest
    print(f"Input SHA-256: {digest}; check {'enabled' if args.expected_sha256 is not None else 'skipped (no --expected-sha256)'}", flush=True)
    if args.expected_sha256 is not None and digest.lower() != args.expected_sha256.lower():
        raise ValueError(f"pickle SHA-256 mismatch: {digest}")
    summary["input_sha256_check"] = "verified" if args.expected_sha256 is not None else "skipped"
    # Helpers also select Agg. Keep their font/config cache inside this run.
    os.environ["MPLCONFIGDIR"] = str(out / "mplconfig")
    p4, p1, per = (load_helper(name) for name in ("p4_fit", "p1_camera_fit", "p4_camera_per_accom"))
    summary["helper_sha256"] = {name: sha256(SCRIPT/name) for name in
                                ("p4_fit.py", "p1_camera_fit.py", "p4_camera_per_accom.py")}
    rows = np.load(PICKLE, allow_pickle=True)
    cube, validation = p4.validate_full(rows, digest, args.expected_sha256)
    summary["input_validation"] = validation
    ai, bi = np.indices((101, 101))
    train_state = (ai % 2 == 0) & (bi % 2 == 0)
    train = np.broadcast_to(train_state[..., None], (101, 101, 9))
    fitmask = train & ((ai % 10 == 0) & (bi % 10 == 0))[..., None] if args.smoke else train.copy()
    if int(train_state.sum()) != 2601 or int((~train_state).sum()) != 7600:
        raise ValueError("unexpected training/holdout state counts")
    summary["split"] = {"train_states_per_A": 2601, "holdout_states_per_A": 7600,
                         "train_rays_per_A": 23409, "holdout_rays_per_A": 68400,
                         "fit_states_per_A": int(fitmask.sum()//9), "fit_rays_per_A": int(fitmask.sum()),
                         "rule": "ADE even AND BDE even, independently at every A"}
    records, own, normalized_own, own_covs = [], [], [], []
    for i in range(21):
        log_progress(out, f"A={ACCOMMODATION[i]:.1f} D: own E2 fit")
        physical, params, cov, rec = own_fit(p1, per, cube[i], train, fitmask, args.smoke)
        rec.update(index=i, A_D=float(ACCOMMODATION[i]), role="regression" if i in ANCHORS else "interpolation")
        records.append(rec)
        own.append(physical)
        normalized_own.append(params)
        own_covs.append(cov)
    own = np.asarray(own)
    ols, ols_cov, ols_residual = ols_coefficients(own)
    log_progress(out, "Direct joint six-coefficient fit on five training levels")
    joint, joint_cov, joint_info = joint_coefficients(p1, cube, records, fitmask, ols)
    residuals = score(p1, cube, records, own, ols, joint, train_state)
    baseline = baseline_comparison(records)
    gates = acceptance(records, ols, baseline)
    fit_health = all(all(r["original_E2_gate_details"].values()) for r in records) and joint_info["solver"]["success"]
    accepted = bool(not args.smoke and fit_health and all(g["passed"] for g in gates.values()))
    interpolation_only = [name for name in ("V2", "V3") if gates[name]["fails_only_at_interpolation_levels"]]
    interpolation_verdict = ("Interpolation-only failures in " + ", ".join(interpolation_only) +
        "; reported without adding a quadratic term. A user decision is required for that model change."
        if interpolation_only else "V2/V3 have no failures confined to the 16 intermediate levels; no quadratic fitted.")
    comparison = {"joint_minus_ols_coefficients": (joint-ols).tolist(),
                  "max_prediction_displacement_um": max(r["joint_to_ols_displacement"]["all"]["max_um"] for r in records),
                  "by_scope": {scope: {name: {
                      "worst_holdout_max_um": max(records[i]["models"][name]["holdout"]["point_max_um"] for i in indices),
                      "worst_holdout_rms_um": max(records[i]["models"][name]["holdout"]["euclidean_rms_um"] for i in indices),
                      "max_displacement_to_own_um": max(records[i]["models"][name]["displacement_to_own"]["all"]["max_um"] for i in indices)}
                      for name in ("linear_ols", "linear_joint")}
                      for scope, indices in (("anchors", ANCHORS), ("interpolation", INTERPOLATION), ("all", range(21)))}}
    gates_passed = all(g["passed"] for g in gates.values())
    status = "completed" if gates_passed else ("smoke_gates_failed" if args.smoke else "gates_failed")
    summary.update(status=status, accepted=accepted, fit_health_passed=bool(fit_health),
                   validation_eligible=not args.smoke, gates=gates, baseline_comparison=baseline,
                   parameter_table=parameter_table(ols, ols_cov, joint, joint_cov),
                   joint_fit=joint_info, joint_comparison=comparison, per_accommodation=records,
                   interpolation_verdict=interpolation_verdict,
                   uncertainty={"caveat": "Formal only: data are deterministic; judge sensor effect size.",
                                "OLS": "unweighted five-point line scatter, 3 residual df; no holdout or intermediate data",
                                "joint_and_own": "linearized sensor residual Jacobian RSS/(2N-p)"},
                   residuals={"file": "residuals.npz", "axes": ["accommodation", "ADE", "BDE", "ray"],
                              "shape_per_model_component": list(cube.shape), "unit": "um",
                              "models": list(MODELS), "definition": "prediction minus measured sensor position"})
    np.savez_compressed(out / "residuals.npz", **residuals, accommodation_d=ACCOMMODATION,
                        distance_label=np.str_("167.93 mm, zero working distance"),
                        working_distance_mm=np.float64(0.0), distance_mm=np.float64(167.9259434838),
                        ade_index=np.arange(101), bde_index=np.arange(101), ray_index=np.arange(9),
                        ade_deg=np.arange(101)*.25-12.5, bde_deg=np.arange(101)*.25-12.5,
                        train_state=train_state, holdout_state=~train_state, fit_ray_mask=fitmask,
                        regression_indices=ANCHORS, interpolation_indices=INTERPOLATION,
                        T_a=np.array([r["T_a"] for r in records]),
                        own_physical=own, own_normalized=np.asarray(normalized_own),
                        own_covariance_physical=np.asarray(own_covs), coefficient_labels=np.array(NAMES),
                        ols_coefficients=ols, joint_coefficients=joint, ols_covariance=ols_cov,
                        joint_covariance=joint_cov, ols_anchor_parameter_residual=ols_residual)
    plots(out, summary, ols, joint)
    (out / "report.txt").write_text(report(summary), encoding="utf-8")
    return 0 if accepted or (args.smoke and fit_health and gates_passed) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-sha256", help="Expected input SHA-256; default: skip hash check and print the hash found")
    parser.add_argument("--smoke", action="store_true", help="Fit 11x11 states per A; full scoring, provisional gates only")
    parser.add_argument("--output-dir", type=Path, help="Output folder; default: runs/p4_camera_linear_accom (smoke: runs/p4_camera_linear_accom_smoke)")
    args = parser.parse_args(argv)
    now = datetime.now(timezone.utc)
    runs = ROOT / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    out = (args.output_dir.resolve() if args.output_dir is not None else
           runs / ("p4_camera_linear_accom_smoke" if args.smoke else "p4_camera_linear_accom"))
    if args.smoke and not out.name.endswith("_smoke"):
        out = out.with_name(out.name + "_smoke")
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.log").write_text("", encoding="utf-8")
    summary = {"status": "running", "created_utc": now.isoformat(), "output_directory": str(out),
               "script_sha256": sha256(Path(__file__)), "input": str(PICKLE),
               "expected_input_sha256": args.expected_sha256, "distance_mm": 167.93,
               "working_distance_mm": 0., "smoke": args.smoke, "accepted": False,
               "regression_indices": ANCHORS.tolist(), "interpolation_indices": INTERPOLATION.tolist(),
               "T_convention": "max absolute tx or ty over each A's full even/even training rays",
               "normalization": "u=tx/T_a; v=ty/T_a; f_norm=f*T_a; k1_norm=k1*T_a^2; k2_norm=k2*T_a^4",
               "constant_camera": "all five physical A0 own-fit parameters, including free centers",
               "law_centers": {"cx_mm": 0., "cy_mm": 0.},
               "proposal_deviations": ["Additional joint fit, uncertainty tables and smoke mode; canonical model and split unchanged."]}
    try:
        proposal = ROOT / "p4_model_proposal.md"
        summary["proposal_sha256"] = sha256(proposal) if proposal.is_file() else None
        summary["proposal_status"] = "present" if proposal.is_file() else "historical document unavailable; provenance only"
        code = run(args, out, summary)
    except Exception as exc:
        summary.update(status="failed", accepted=False, failure={"type": type(exc).__name__, "message": str(exc)})
        (out / "report.txt").write_text("P4 linear accommodation validation failed: 167.93 mm, zero working distance\n" + traceback.format_exc(), encoding="utf-8")
        code = 1
    # Summaries contain no ray arrays. Numerical arrays live in the NPZ.
    (out / "summary.json").write_text(json.dumps(summary, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    log_progress(out, f"{summary['status']}: {out}; accepted={summary['accepted']}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
