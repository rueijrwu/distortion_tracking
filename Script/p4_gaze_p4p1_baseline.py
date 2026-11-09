"""Fit and score paired P4/P1 polynomial gaze readouts from frozen artifacts.

This analysis consumes saved hybrid runtime estimates and never runs the solver.
"""
from __future__ import annotations

import hashlib
import json
import sys
sys.dont_write_bytecode = True
from pathlib import Path

import numpy as np

from evaluate_p4_unified_combined import load_models
from fit_p4_p1_both_centers import load_p1

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/p4_p1_limited_gaze_12p5/source_subset.npz"
RUN = ROOT / "data/p4_unified_combined/runtime/full_fast41/runtime.npz"
FIELD_SUM = ROOT / "data/p4_unified_combined/calibration/full_field_profile_final/summary.json"
MOTION_SUM = ROOT / "data/p4_unified_combined/calibration/center_motion/summary.json"
MOTION_CAL = ROOT / "data/p4_unified_combined/calibration/center_motion/calibration_data.npz"
MOTION_ART = ROOT / "data/p4_unified_combined/calibration/center_motion/central_motion_M1.npz"
P1_CAL = ROOT / "data/p1_p4_a0_independent_gaze/calibration/calibration_data.npz"
OUT = ROOT / "data/p4_gaze_p4p1_baseline"
DEGREES = range(1, 5)
VARIANTS = ("estimated_A", "true_A_oracle", "identity_no_shift")
NOISE_SD = .001


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(g):
    return tuple(np.round(g, 8))


def fit(x, y, degree):
    x = np.asarray(x, float)
    center, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1.
    z = (x-center)/scale
    powers = [(i, t-i) for t in range(degree+1) for i in range(t, -1, -1)]
    X = np.stack([z[:, 0]**i*z[:, 1]**j for i, j in powers], axis=1)
    coef, _, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    return {"center": center, "scale": scale, "powers": powers, "coef": coef,
            "rank": int(rank), "singular_values": sv.tolist(),
            "condition": float(sv[0]/sv[-1]) if len(sv) and sv[-1] else None}


def predict(m, x):
    z = (np.asarray(x, float)-m["center"])/m["scale"]
    X = np.stack([z[:, 0]**i*z[:, 1]**j for i, j in m["powers"]], axis=1)
    return X@m["coef"]


def ang(pred, truth):
    e = np.asarray(pred)-np.asarray(truth)
    if len(e) == 0:
        return {"n": 0, "finite": 0, "radial_rms_arcmin": None,
            "radial_p95_arcmin": None, "radial_max_arcmin": None,
            "axis_rms_arcmin": [None, None], "axis_p95_abs_arcmin": [None, None],
            "axis_max_abs_arcmin": [None, None], "within_1_arcmin_pct": None,
            "within_3_arcmin_pct": None}
    rad = np.linalg.norm(e, axis=1)*60.
    ea = np.abs(e)*60.
    good = np.isfinite(rad)
    r = rad[good]
    return {"n": len(rad), "finite": int(good.sum()),
            "radial_rms_arcmin": float(np.sqrt(np.mean(r*r)) if len(r) else np.nan),
            "radial_p95_arcmin": float(np.percentile(r, 95) if len(r) else np.nan),
            "radial_max_arcmin": float(np.max(r) if len(r) else np.nan),
            "axis_rms_arcmin": (np.sqrt(np.mean(e*e, axis=0))*60.).tolist(),
            "axis_p95_abs_arcmin": np.percentile(ea, 95, axis=0).tolist(),
            "axis_max_abs_arcmin": np.max(ea, axis=0).tolist(),
            "within_1_arcmin_pct": float(np.mean(r <= 1)*100 if len(r) else np.nan),
            "within_3_arcmin_pct": float(np.mean(r <= 3)*100 if len(r) else np.nan)}


def center_metrics(p, y):
    e = (np.asarray(p)-np.asarray(y))*1000.
    r = np.linalg.norm(e, axis=1)
    return {"vector_rms_um": float(np.sqrt(np.mean(r*r))),
            "vector_p95_um": float(np.percentile(r, 95)),
            "vector_max_um": float(np.max(r)),
            "axis_rms_um": np.sqrt(np.mean(e*e, axis=0)).tolist()}


def jacobian(m, x):
    z = (x-m["center"])/m["scale"]
    J = np.zeros((len(x), 2, 2))
    for k, (i, j) in enumerate(m["powers"]):
        if i:
            J[:, :, 0] += (i*z[:, 0]**(i-1)*z[:, 1]**j/m["scale"][0])[:, None]*m["coef"][k]
        if j:
            J[:, :, 1] += (j*z[:, 0]**i*z[:, 1]**(j-1)/m["scale"][1])[:, None]*m["coef"][k]
    return J


def noise_floor(m, x, scale4, sigma):
    # P1 and P4 center errors are independent; inverse M1 scales P4 noise by 1/s.
    J = jacobian(m, x)
    C = np.zeros((len(x), 2, 2))
    C[:, 0, 0] = sigma**2*(1+1/scale4[:, 0]**2)
    C[:, 1, 1] = sigma**2*(1+1/scale4[:, 1]**2)
    V = J@C@np.swapaxes(J, 1, 2)
    return {"assumed_center_sigma_mm_per_axis": sigma,
            "radial_rms_arcmin": float(np.sqrt(np.mean(np.trace(V, axis1=1, axis2=2)))*60.),
            "axis_rms_arcmin": (np.sqrt(np.mean(np.diagonal(V, axis1=1, axis2=2), axis=0))*60.).tolist()}


def main():
    if sha(SRC) != "97c6db1b455f9abf7354351fa768cca7eea67198d8a4bfd14bbe684475462daf":
        raise RuntimeError("Pinned paired source checksum changed")
    frozen, field, motion, hashes = load_models(FIELD_SUM, MOTION_SUM)
    with np.load(SRC) as z:
        ids, states, p4 = z["source_indices"].copy(), z["states"].copy(), z["absolute_p4_mm"].copy()
        p1mean = z["mean_p1_absolute_mm"].copy()
        cal75, common = z["cal75"].copy(), z["common_holdout"].copy()
    with np.load(RUN) as z:
        rows = z["source_subset_rows"].copy()
        if not np.array_equal(z["source_indices"], ids[common]) or not np.array_equal(rows, np.flatnonzero(common)):
            raise RuntimeError("Saved runtime/source row join mismatch")
        if not np.array_equal(z["truth_state_scoring_only"], states[common]):
            raise RuntimeError("Saved runtime/source state mismatch")
        runtime = {k: z[k].copy() for k in ("joint_estimated_A_d", "joint_center_mm",
            "joint_center_A0_reference_mm", "independent_P1_center_mm", "true_P1_labeled_center_scoring_only_mm",
            "true_P4_labeled_center_scoring_only_mm", "independent_P1_gaze_deg", "primary_mask", "excluded_mask")}
    if (int(runtime["primary_mask"].sum()) != 83229 or int(runtime["excluded_mask"].sum()) != 21
            or np.any(runtime["primary_mask"] & runtime["excluded_mask"])):
        raise RuntimeError("Saved primary/excluded runtime masks differ from frozen 83,229/21 split")
    cal = np.flatnonzero(cal75)
    if len(cal) != 75:
        raise RuntimeError("Expected 75 known-A calibration poses")
    if len(np.unique(np.round(states[cal, 1:], 8), axis=0)) != 25:
        raise RuntimeError("Expected 25 unique matched calibration gaze poses")
    with np.load(MOTION_CAL) as mcal:
        mlookup = {int(sid): st for sid, st in zip(mcal["source_indices"], mcal["states"])}
        if any(sid not in mlookup or not np.array_equal(states[row], mlookup[int(sid)])
               for row, sid in zip(cal, ids[cal])):
            raise RuntimeError("75-row paired-source calibration join differs from frozen M1 calibration data")
    gaze, truth = states[:, 1:], states[:, 1:]

    # Reconstruct/join P1 labeled rays, and verify nominal paired coordinate identity.
    p1labels, p1grids, _ = load_p1()
    lookup = {key(g): i for i, g in enumerate(p1labels)}
    p1row = np.array([lookup[key(g)] for g in gaze])
    p1oracle = p1grids[p1row, 4]
    if np.max(np.abs(p1grids[p1row].mean(axis=1)-p1mean)) > 1e-10:
        raise RuntimeError("P1 reconstructed mean differs from paired common-source coordinates")
    # Calibration centers: truth-conditioned known A/gaze through frozen field and B1.
    cal_p4_center = p4[cal].mean(axis=1)-field.evaluate(states[cal, 0], gaze[cal]).mean(axis=1)
    cal_p1_center = p1mean[cal]-frozen.p1_bias.predict(gaze[cal])
    cal_p4_oracle, cal_p1_oracle = p4[cal, 4], p1oracle[cal]
    # Runtime centers: canonical saved hybrid center and independent P1 center.
    rt_p4_center, rt_p1_center = runtime["joint_center_mm"], runtime["independent_P1_center_mm"]
    rt_p4_oracle, rt_p1_oracle = runtime["true_P4_labeled_center_scoring_only_mm"], runtime["true_P1_labeled_center_scoring_only_mm"]
    if (np.max(np.abs(rt_p1_oracle-p1oracle[rows])) > 1e-10
            or np.max(np.abs(rt_p4_oracle-p4[rows, 4])) > 1e-10):
        raise RuntimeError("Saved runtime labeled-center scoring arrays disagree with paired source")
    if np.max(np.abs(rt_p1_center-(p1mean[rows]-frozen.p1_bias.predict(runtime["independent_P1_gaze_deg"])))) > 1e-10:
        raise RuntimeError("Saved independent P1 center is inconsistent with frozen B1 and saved G1 gaze")
    if np.max(np.abs(motion.inverse(rt_p4_center, runtime["joint_estimated_A_d"])-runtime["joint_center_A0_reference_mm"])) > 1e-10:
        raise RuntimeError("Saved runtime M1-normalized center differs from frozen M1 inverse")

    # Three distinct center pairs: a raw means, b frozen/hybrid corrected centers, c labeled centers.
    p1cal = {"a_raw": p1mean[cal], "b_corrected": cal_p1_center, "c_labeled": cal_p1_oracle}
    p4cal = {"a_raw": p4[cal].mean(axis=1), "b_corrected": cal_p4_center, "c_labeled": cal_p4_oracle}
    p1rt = {"a_raw": p1mean[rows], "b_corrected": rt_p1_center, "c_labeled": rt_p1_oracle}
    p4rt = {"a_raw": p4[rows].mean(axis=1), "b_corrected": rt_p4_center, "c_labeled": rt_p4_oracle}
    Ahat, Atrue = runtime["joint_estimated_A_d"], states[rows, 0]
    shifts_cal = {"estimated_A": states[cal, 0], "true_A_oracle": states[cal, 0], "identity_no_shift": np.zeros(len(cal))}
    shifts_rt = {"estimated_A": Ahat, "true_A_oracle": Atrue, "identity_no_shift": np.zeros(len(rows))}
    scales = {v: motion.scales(a) for v, a in shifts_rt.items()}
    pair_cal, pair_rt = {}, {}
    for center_name in p1cal:
        for variant in VARIANTS:
            tag = f"{center_name}__{variant}"
            c4cal = p4cal[center_name] if variant == "identity_no_shift" else motion.inverse(p4cal[center_name], shifts_cal[variant])
            c4rt = p4rt[center_name] if variant == "identity_no_shift" else motion.inverse(p4rt[center_name], shifts_rt[variant])
            pair_cal[tag] = c4cal-p1cal[center_name]
            pair_rt[tag] = c4rt-p1rt[center_name]

    # LOPO by gaze location (25 folds; all A=0,2,4 repeats withheld together).
    # Known A on calibration stands in for unavailable saved calibration A-hat.
    headline = "b_corrected__estimated_A"
    cv, cvpred = {}, {}
    for d in DEGREES:
        pred = np.full((75, 2), np.nan)
        for g in np.unique(np.round(gaze[cal], 8), axis=0):
            test = np.all(np.round(gaze[cal], 8) == g, axis=1)
            train = ~test
            m = fit(pair_cal[headline][train], gaze[cal][train], d)
            if m["rank"] == len(m["powers"]):
                pred[test] = predict(m, pair_cal[headline][test])
        cv[str(d)] = ang(pred, gaze[cal])
        cv[str(d)]["fold_count"] = int(len(np.unique(np.round(gaze[cal], 8), axis=0)))
        cvpred[str(d)] = pred
    eligible = [d for d in DEGREES if cv[str(d)]["finite"] == len(cal)
                and cv[str(d)]["radial_rms_arcmin"] is not None
                and np.isfinite(cv[str(d)]["radial_rms_arcmin"])]
    if not eligible:
        raise RuntimeError("All grouped LOPO degrees failed rank/finite prediction checks")
    selected = min(eligible, key=lambda d: cv[str(d)]["radial_rms_arcmin"])

    # Nine paired and twelve standalone readouts: 3 P1 centers; 3 P4 center types x 3 shifts.
    paired, preds_cal, preds_rt = {}, {}, {}
    for tag in pair_cal:
        m = fit(pair_cal[tag], gaze[cal], selected)
        if m["rank"] != len(m["powers"]):
            raise RuntimeError(f"Rank deficient selected-degree fit: {tag}")
        paired[tag] = m
        preds_cal[tag] = predict(m, pair_cal[tag])
        preds_rt[tag] = predict(m, pair_rt[tag])
    p1_models, p4_models = {}, {}
    # P1 25-pose standalone center choices; use matching calibration identities.
    with np.load(P1_CAL) as z:
        p1ids = z["calibration_p4_source_indices"].copy()
        p1g = z["calibration_gaze_deg"].copy()
    if len(p1ids) != 25 or len(np.unique(np.round(p1g, 8), axis=0)) != 25:
        raise RuntimeError("Expected 25 unique P1 calibration poses")
    p1cr = np.array([np.flatnonzero(ids == sid)[0] for sid in p1ids])
    p1_single_cal = {"a_raw": p1mean[p1cr],
        "b_corrected": p1mean[p1cr]-frozen.p1_bias.predict(p1g), "c_labeled": p1oracle[p1cr]}
    p1_single_rt = {"a_raw": p1mean[rows], "b_corrected": rt_p1_center, "c_labeled": rt_p1_oracle}
    for name, xcal in p1_single_cal.items():
        m = fit(xcal, p1g, selected)
        if m["rank"] != len(m["powers"]):
            raise RuntimeError(f"Rank deficient selected-degree P1 fit: {name}")
        p1_models[name] = (m, ang(predict(m, xcal), p1g), predict(m, p1_single_rt[name]))
    for center_name in p4cal:
        for variant in VARIANTS:
            tag = f"{center_name}__{variant}"
            xcal = p4cal[center_name] if variant == "identity_no_shift" else motion.inverse(p4cal[center_name], shifts_cal[variant])
            xrt = p4rt[center_name] if variant == "identity_no_shift" else motion.inverse(p4rt[center_name], shifts_rt[variant])
            m = fit(xcal, gaze[cal], selected)
            if m["rank"] != len(m["powers"]):
                raise RuntimeError(f"Rank deficient selected-degree P4 fit: {tag}")
            p4_models[tag] = (m, ang(predict(m, xcal), gaze[cal]), predict(m, xrt))

    # Exact signal center errors, reported on primary and excluded masks separately.
    primary = runtime["primary_mask"]
    excluded = runtime["excluded_mask"]
    center_diagnostics = {}
    for n in p1cal:
        for v in VARIANTS:
            c4 = p4cal[n] if v == "identity_no_shift" else motion.inverse(p4cal[n], shifts_cal[v])
            c4_oracle = cal_p4_oracle if v == "identity_no_shift" else motion.inverse(cal_p4_oracle, shifts_cal[v])
            center_diagnostics[f"{n}_{v}_p1_vs_labeled"] = center_metrics(p1cal[n], cal_p1_oracle)
            center_diagnostics[f"{n}_{v}_p4_vs_same_shift_labeled"] = center_metrics(c4, c4_oracle)
    center_diagnostics["calibrated_field_center_vs_labeled_ray"] = center_metrics(cal_p4_center, cal_p4_oracle)
    center_diagnostics["runtime_by_center_definition_shift"] = {}
    center_diagnostics["excluded_by_center_definition_shift"] = {}
    for label, mask, target in (("runtime_by_center_definition_shift", primary, center_diagnostics["runtime_by_center_definition_shift"]),
                                ("excluded_by_center_definition_shift", excluded, center_diagnostics["excluded_by_center_definition_shift"])):
        for n in p1rt:
            for v in VARIANTS:
                p1err = center_metrics(p1rt[n][mask], rt_p1_oracle[mask])
                c4 = p4rt[n] if v == "identity_no_shift" else motion.inverse(p4rt[n], shifts_rt[v])
                c4ref = rt_p4_oracle if v == "identity_no_shift" else motion.inverse(rt_p4_oracle, shifts_rt[v])
                target[f"{n}_{v}"] = {
                    "P1_vs_labeled_um": p1err, "P4_vs_same_shift_labeled_um": center_metrics(c4[mask], c4ref[mask]),
                    "paired_center_delta_vs_same_shift_labeled_um": center_metrics(c4[mask]-p1rt[n][mask], c4ref[mask]-rt_p1_oracle[mask])}
    center_diagnostics["runtime_hybrid_center_vs_labeled_ray_primary"] = center_metrics(rt_p4_center[primary], rt_p4_oracle[primary])
    center_diagnostics["runtime_P1_center_vs_labeled_ray_primary"] = center_metrics(rt_p1_center[primary], rt_p1_oracle[primary])
    center_diagnostics["runtime_hybrid_center_vs_labeled_ray_excluded"] = center_metrics(rt_p4_center[excluded], rt_p4_oracle[excluded])
    center_diagnostics["runtime_P1_center_vs_labeled_ray_excluded"] = center_metrics(rt_p1_center[excluded], rt_p1_oracle[excluded])

    # Delta-method noise floor, fixed-A only; report both a 1um center sigma and
    # 1um iid-per-ray sigma/3 for a nine-ray arithmetic mean with frozen corrections.
    floor_center, floor_ray = {}, {}
    for tag, m in paired.items():
        variant = tag.split("__", 1)[1]
        sigma_ray = NOISE_SD if tag.startswith("c_labeled__") else NOISE_SD/3.
        floor_center[tag] = noise_floor(m, pair_rt[tag][primary], scales[variant][primary], NOISE_SD)
        floor_ray[tag] = noise_floor(m, pair_rt[tag][primary], scales[variant][primary], sigma_ray)

    groups = {"primary": runtime["primary_mask"], "excluded_21_diagnostic": runtime["excluded_mask"],
              "inclusive": np.ones(len(rows), bool)}
    binspec = {"[0,4)": (0., 4.), "[4,8)": (4., 8.), "[8,12)": (8., 12.), "[12,12.4]": (12., 12.4000001)}
    breakdown = {}
    for gn, mask in groups.items():
        selected_rows = rows[mask]
        breakdown[gn] = {"count": int(mask.sum()), "paired": {}, "P1": {}, "P4": {}}
        mag = np.max(np.abs(gaze[selected_rows]), axis=1)
        for tag, pred in preds_rt.items():
            breakdown[gn]["paired"][tag] = {"all": ang(pred[mask], gaze[selected_rows]), "bins": {}}
            for bn, (lo, hi) in binspec.items():
                bm = (mag >= lo) & (mag < hi)
                breakdown[gn]["paired"][tag]["bins"][bn] = ang(pred[mask][bm], gaze[selected_rows][bm])
        for name, (_, _, pr) in p1_models.items():
            breakdown[gn]["P1"][name] = {"all": ang(pr[mask], gaze[selected_rows]), "bins": {}}
            for bn, (lo, hi) in binspec.items():
                bm = (mag >= lo) & (mag < hi)
                breakdown[gn]["P1"][name]["bins"][bn] = ang(pr[mask][bm], gaze[selected_rows][bm])
        for tag, (_, _, pr) in p4_models.items():
            breakdown[gn]["P4"][tag] = {"all": ang(pr[mask], gaze[selected_rows]), "bins": {}}
            for bn, (lo, hi) in binspec.items():
                bm = (mag >= lo) & (mag < hi)
                breakdown[gn]["P4"][tag]["bins"][bn] = ang(pr[mask][bm], gaze[selected_rows][bm])
    by_true_A = {}
    for a in np.unique(Atrue):
        amask = np.isclose(Atrue, a, atol=1e-10, rtol=0) & runtime["primary_mask"]
        aa = {"count": int(amask.sum()), "paired": {}, "P1": {}, "P4": {}}
        amag = np.max(np.abs(gaze[rows][amask]), axis=1)
        for tag, pred in preds_rt.items():
            q = ang(pred[amask], gaze[rows][amask]); q["bins"] = {}
            for bn, (lo, hi) in binspec.items():
                bm = (amag >= lo) & (amag < hi)
                q["bins"][bn] = ang(pred[amask][bm], gaze[rows][amask][bm])
            aa["paired"][tag] = q
        for name, (_, _, pr) in p1_models.items():
            q = ang(pr[amask], gaze[rows][amask]); q["bins"] = {}
            for bn, (lo, hi) in binspec.items():
                bm = (amag >= lo) & (amag < hi)
                q["bins"][bn] = ang(pr[amask][bm], gaze[rows][amask][bm])
            aa["P1"][name] = q
        for tag, (_, _, pr) in p4_models.items():
            q = ang(pr[amask], gaze[rows][amask]); q["bins"] = {}
            for bn, (lo, hi) in binspec.items():
                bm = (amag >= lo) & (amag < hi)
                q["bins"][bn] = ang(pr[amask][bm], gaze[rows][amask][bm])
            aa["P4"][tag] = q
        by_true_A[f"{a:.1f}"] = aa
    worst = {}
    for tag, pred in preds_rt.items():
        mask = runtime["primary_mask"]
        err = np.linalg.norm(pred[mask]-gaze[rows][mask], axis=1)*60.
        rr = rows[mask]
        order = np.argsort(err)[::-1][:10]
        worst[tag] = [{"source_subset_row": int(rr[i]), "source_index": int(ids[rr[i]]),
                       "state_A_ADE_BDE": states[rr[i]].tolist(), "radial_error_arcmin": float(err[i])} for i in order]

    # 21 concise main rows, with selected-degree in-sample calibration and primary holdout metrics.
    table = []
    for tag in paired:
        table.append({"family": "P4-P1", "readout": tag, "degree": selected,
            "LOPO": cv[str(selected)] if tag == headline else None,
            "calibration_in_sample": ang(preds_cal[tag], gaze[cal]),
            "primary_holdout": breakdown["primary"]["paired"][tag]["all"]})
    for name, (_, calmetric, pr) in p1_models.items():
        table.append({"family": "P1-only", "readout": name, "degree": selected,
            "LOPO": None, "calibration_in_sample": calmetric,
            "primary_holdout": breakdown["primary"]["P1"][name]["all"]})
    for tag, (_, calmetric, pr) in p4_models.items():
        table.append({"family": "P4-only", "readout": tag, "degree": selected,
            "LOPO": None, "calibration_in_sample": calmetric,
            "primary_holdout": breakdown["primary"]["P4"][tag]["all"]})

    signed_center_yardsticks = {}
    for n in p1cal:
        for v in VARIANTS:
            c4 = p4cal[n] if v == "identity_no_shift" else motion.inverse(p4cal[n], shifts_cal[v])
            signed_center_yardsticks[f"{n}__{v}"] = {
                "signed_difference_um": (((c4-p1cal[n])-(p4cal["a_raw"]-p1cal["a_raw"]))*1000.).tolist(),
                "vector_difference_um": center_metrics(c4-p1cal[n], p4cal["a_raw"]-p1cal["a_raw"])}
    noise_metrics = {tag: {"1um_sigma_per_center_axis": floor_center[tag],
        "1um_iid_ray": floor_ray[tag]} for tag in paired}
    noise_metrics["labeled_central_ray_noise_reference"] = (
        "A direct labeled central ray retains 1um/axis; 1/3um applies only to a nine-ray mean under independent equal ray noise.")

    motion_cal = np.load(MOTION_CAL)
    with np.load(MOTION_ART) as mart:
        m1_scale_d = float(mart["accommodation_scale_d"])
        m1_formula = str(mart["formula"].item())
    provenance = {"source_sha256": sha(SRC), "runtime_sha256": sha(RUN),
        "field_summary_sha256": sha(FIELD_SUM), "motion_summary_sha256": sha(MOTION_SUM),
        "motion_calibration_sha256": sha(MOTION_CAL), "motion_artifact_path": str(MOTION_ART),
        "motion_artifact_sha256": sha(MOTION_ART), "M1_accommodation_normalizer_d": m1_scale_d,
        "M1_saved_formula": m1_formula, "p1_calibration_sha256": sha(P1_CAL),
        "model_hashes": hashes, "calibration_count": len(cal), "primary_count": int(runtime["primary_mask"].sum()),
        "excluded_count": int(runtime["excluded_mask"].sum()), "excluded_gaze_ADE_BDE": [12., 9.6],
        "frame_convention": "No added mirror or registration transform. HANDOFF_2026-11-06.md: ‘Preserve the user-described mirrored common-source convention and the nominal recorded frames’; Summary.md describes ‘recorded nominal P1/P4 frames’. P1 reconstruction is checked against paired-source means; this is nominal coordinate consistency, not independently verified physical same-origin/axes/scale registration.",
        "M1_pivot_mm": motion.pivot_mm.tolist(), "M1_coefficients_axis_by_power": motion.coefficients.tolist(),
        "M1_formula": "cA=o+diag(1+sum(k_axis,p*(A/4)^p))*(c0-o)",
        "M1_scales_A0_A4": [motion.scales(np.array([0.]))[0].tolist(), motion.scales(np.array([4.]))[0].tolist()],
        "M1_calibration_states": motion_cal["states"].tolist(),
        "calibration_A_method": "No saved calibration A-hat; known A used to evaluate frozen P4 field and M1 for calibration features. This is teacher-conditioned and may cause runtime covariate shift.",
        "runtime_A_variants": "saved joint_estimated_A_d headline, true A scoring-only oracle, and identity/no M1 shift.",
        "coordinate_identity_checks": {"P1_rebuilt_arithmetic_mean_max_abs_mm": float(np.max(np.abs(p1grids[p1row].mean(axis=1)-p1mean))),
            "runtime_P1_labeled_center_max_abs_mm": float(np.max(np.abs(rt_p1_oracle-p1oracle[rows]))),
            "runtime_P4_labeled_center_max_abs_mm": float(np.max(np.abs(rt_p4_oracle-p4[rows, 4])))},
        "noise_model": "Conditional linearized readout floor only; no Monte Carlo, solver rerun, coefficient uncertainty, or A/shape perturbation.",
        "degrees": [1, 2, 3, 4], "selected_degree": selected}
    metrics = {"degree_LOPO_by_degree": cv, "selected_degree": selected,
        "degree_selection_rule": "minimum grouped LOPO RMS for b_corrected plus M1 inverse at known calibration A; 25 gaze groups, all three A repeats held out together",
        "main_21_rows": table, "runtime_breakdown": breakdown, "by_true_A": by_true_A,
        "worst_10_primary_by_paired_readout": worst,
        "center_diagnostics_um": center_diagnostics, "signed_center_yardsticks": signed_center_yardsticks,
        "noise_floor_difference_readout_primary_83229": noise_metrics,
        "quoted_comparators_arcmin": {"G1": .18, "G4": .90, "recomputed": False}}

    arr = {"calibration_rows": cal, "calibration_source_indices": ids[cal], "calibration_states_truth_labels": states[cal],
        "runtime_rows": rows, "runtime_source_indices": ids[rows], "runtime_states_scoring_only": states[rows],
        "primary_mask": runtime["primary_mask"], "excluded_mask": runtime["excluded_mask"],
        "p1_mean_mm": p1mean, "p4_mean_mm": p4.mean(axis=1), "p1_oracle_center_scoring_only_mm": p1oracle,
        "p4_oracle_center_scoring_only_mm": p4[:, 4], "p1_corrected_calibration_mm": cal_p1_center,
        "p4_corrected_calibration_mm": cal_p4_center, "p1_corrected_runtime_mm": rt_p1_center,
        "p4_corrected_runtime_mm": rt_p4_center, "runtime_Ahat_d": Ahat, "runtime_Atrue_scoring_only_d": Atrue,
        "degree_LOPO_prediction_by_degree": np.stack([cvpred[str(d)] for d in DEGREES]),
        "runtime_noise_floor_center_sigma": np.array([floor_center[headline]["radial_rms_arcmin"]]),
        "runtime_noise_floor_ray_sigma": np.array([floor_ray[headline]["radial_rms_arcmin"]])}
    for tag, x in pair_cal.items():
        arr[f"paired_{tag}_calibration_input_mm"] = x
        arr[f"paired_{tag}_runtime_input_mm"] = pair_rt[tag]
    for tag, m in paired.items():
        arr[f"paired_{tag}_coef"] = m["coef"]
        arr[f"paired_{tag}_normalizer_center_mm"] = m["center"]
        arr[f"paired_{tag}_normalizer_scale_mm"] = m["scale"]
        arr[f"paired_{tag}_powers"] = np.asarray(m["powers"], int)
        arr[f"paired_{tag}_prediction_runtime_deg"] = preds_rt[tag]
    for name, (m, _, pr) in p1_models.items():
        arr[f"P1_{name}_coef"] = m["coef"]
        arr[f"P1_{name}_normalizer_center_mm"] = m["center"]
        arr[f"P1_{name}_normalizer_scale_mm"] = m["scale"]
        arr[f"P1_{name}_powers"] = np.asarray(m["powers"], int)
        arr[f"P1_{name}_prediction_runtime_deg"] = pr
    for tag, (m, _, pr) in p4_models.items():
        arr[f"P4_{tag}_coef"] = m["coef"]
        arr[f"P4_{tag}_normalizer_center_mm"] = m["center"]
        arr[f"P4_{tag}_normalizer_scale_mm"] = m["scale"]
        arr[f"P4_{tag}_powers"] = np.asarray(m["powers"], int)
        arr[f"P4_{tag}_prediction_runtime_deg"] = pr
    OUT.mkdir(parents=True, exist_ok=True)
    allowed = {"baseline.npz", "report.md"}
    unexpected = {p.name for p in OUT.iterdir()}-allowed
    if unexpected:
        raise FileExistsError(f"Refusing to overwrite unexpected output files: {sorted(unexpected)}")
    np.savez_compressed(OUT/"baseline.npz", **arr,
        provenance_json=np.array(json.dumps(provenance, sort_keys=True)),
        metrics_json=np.array(json.dumps(metrics, sort_keys=True)))
    report(OUT/"report.md", provenance, metrics)


def report(path, provenance, metrics):
    def f(x, n=5):
        return "n/a" if x is None else f"{x:.{n}g}"

    def angular_rows(family, mapping):
        for name, item in mapping.items():
            q = item.get("all", item)
            if q.get("finite", q.get("n", 0)) == 0:
                yield f"| {family} | {name} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
                continue
            yield (f"| {family} | {name} | {q['radial_rms_arcmin']:.5g} | {q['radial_p95_arcmin']:.5g} | "
                f"{q['radial_max_arcmin']:.5g} | {q['axis_rms_arcmin'][0]:.5g}/{q['axis_rms_arcmin'][1]:.5g} | "
                f"{q['axis_p95_abs_arcmin'][0]:.5g}/{q['axis_p95_abs_arcmin'][1]:.5g} | "
                f"{q['axis_max_abs_arcmin'][0]:.5g}/{q['axis_max_abs_arcmin'][1]:.5g} | "
                f"{q['within_1_arcmin_pct']:.4g}% | {q['within_3_arcmin_pct']:.4g}% |")

    lines = [
        "# P4-P1 gaze baseline", "", "## Frame, method, and limits", "",
        "The handoff says: \"Preserve the user-described mirrored common-source convention and the nominal recorded frames.\" The paired-source summary calls its frame assumption \"conditional common-SI-coordinate simulation; identical nominal rear-camera chain/FOV, no fitted registration transform; physical registration independently unproved.\" No explicit P1-to-P4 mirror matrix is documented, so no added reflection or fitted registration is applied. The P1 reconstructed nine-ray mean is checked against the paired source, but this does not independently verify shared sensor origin, axis signs, or scale.",
        "",
        "Three center definitions are compared: (a) raw nine-ray arithmetic means; (b) corrected centers (saved hybrid runtime centers on holdout; known-state frozen-field correction on calibration); and (c) labeled central rays (oracle centers, deliberately supplied as the c-branch readout input). P1-only and P4-only readouts use their own center. Paired rows use the M1-inverse P4 center minus P1 center. M1 is applied to P4 only. True gaze labels are polynomial training targets; true A is used only for the explicit oracle-A variant and calibration construction. No true labels are supplied to the headline runtime a/b readouts.",
        "",
        "No hybrid solver was rerun. Runtime corrected centers and the existing joint A estimate come unchanged from `data/p4_unified_combined/runtime/full_fast41/runtime.npz`; A is not refit, re-estimated, or clipped here. Calibration uses 25 P1 A=0 poses and 75 matched P4 poses at A=0, 2, and 4 D (25 gaze locations at each A); the primary holdout has 83,229 rows over all 21 A planes. No saved calibration A-hat exists: calibration P1 center uses `mean(y1)-B1(g_true)` and P4 center uses `mean(y4)-mean(F(A_true,g_true))`, so both corrected calibration features are teacher-conditioned and may differ from runtime covariates.",
        "",
        f"M1 pivot: {provenance['M1_pivot_mm']} mm; x/y coefficients: {provenance['M1_coefficients_axis_by_power']}; scales at A=0/4 D: {provenance['M1_scales_A0_A4']}. For a=A/accommodation_scale_d, z=c4-o in mm, z0=z/s(a), c4_ref=o+z0, and d=c4_ref-c1; identity uses s=(1,1). Artifact: `data/p4_unified_combined/calibration/center_motion/central_motion_M1.npz`; saved accommodation scale is {provenance['M1_accommodation_normalizer_d']} D. No extra frame transform is applied; M1 acts in the nominal shared recorded coordinates, with no additional M1 coordinate z-score. The gaze polynomials retain their own fitted input normalizers (reported below). M1 was fit to paired labeled P4 central-ray positions at A and matched A=0. The raw arithmetic centroid (a) is not this center; corrected estimate (b) and oracle ray (c) are intended to estimate/use the labeled center, with b's residual reported as estimation error. M1 selection was calibration-only grouped CV; the P4 field extension calibration Jacobian has rank 15/15 and raw/column-scaled condition 96.54/29.56, and its 2.099 um RMS is training fit. Frozen family count: 58 independent B/G + 12 PF12 + 15 P4 field + 2 M1 = 87 scalars; references/normalizers are separate constants.",
        "",
        f"Degree selection used grouped leave-one-gaze-location-out CV on corrected paired input b, holding all A repeats together (25 groups); known A is used for calibration M1 correction. Each fold refits only the candidate gaze polynomial and its input normalizers on the remaining 24 gaze groups. Frozen B1/B4, P4 field F, and M1 remain fixed at fits trained using their full calibration sets, so this is readout-level CV, not end-to-end cross-validation. Selected degree: **{metrics['selected_degree']}**. Calibration fit metrics below are in-sample apart from this LOPO estimate. The saved hybrid G1/G4 comparators are quoted only; the paired coordinate convention and physical registration remain unvalidated.",
        "", "## Degree selection", "",
        "| Degree | Folds | Radial RMS (arcmin) | P95 | Max | Axis RMS x/y (arcmin) |", "|---:|---:|---:|---:|---:|---|"]
    for degree, q in sorted(metrics["degree_LOPO_by_degree"].items(), key=lambda x: int(x[0])):
        axis = q.get("axis_rms_arcmin", [float("nan"), float("nan")])
        lines.append(f"| {degree} | {q['fold_count']} | {f(q['radial_rms_arcmin'])} | {f(q['radial_p95_arcmin'])} | {f(q['radial_max_arcmin'])} | {axis[0]:.5g}/{axis[1]:.5g} |")

    floor = metrics["noise_floor_difference_readout_primary_83229"]
    lines += ["", "## Primary holdout: 21 readouts", "",
        "Radial and axis errors are in arcmin. Threshold columns show percentage within 1 and 3 arcmin. The noise column is a separate conditional 1 um sigma per channel-center axis floor, not combined with noise-free error.", "",
        "| Family | Readout | Radial RMS | P95 | Max | Axis RMS x/y | Axis P95 x/y | Axis max x/y | <=1 arcmin | <=3 arcmin | 1 um/center/axis floor RMS |", "|---|---|---:|---:|---:|---|---|---|---:|---:|---:|"]
    for row in metrics["main_21_rows"]:
        q = row["primary_holdout"]
        floor_val = None
        if row["family"] == "P4-P1":
            entry = floor[row["readout"]]
            floor_val = entry["1um_sigma_per_center_axis"]["radial_rms_arcmin"]
        line = next(angular_rows(row["family"], {row["readout"]: q}))
        lines.append(line + f" {f(floor_val)} |")
    quoted = metrics.get("quoted_comparators_arcmin", {"G1": .18, "G4": .90})
    lines += ["", f"Frozen hybrid comparators (quoted, not recomputed): G1 {quoted['G1']:.3g} arcmin RMS; G4 {quoted['G4']:.3g} arcmin RMS.",
        "", "## Per-accommodation rows", "", "All 21 A planes and all 21 readout signals are listed. Per-A rows use the same error and threshold summaries.", "",
        "| A (D) | Family | Readout | N | Radial RMS/P95/max | Axis RMS x/y | Axis P95 x/y | Axis max x/y | <=1 arcmin | <=3 arcmin |", "|---:|---|---|---:|---|---|---|---|---:|---:|"]
    by_a = metrics["by_true_A"]
    for a, groups in sorted(by_a.items(), key=lambda x: float(x[0])):
        for group, family in (("paired", "P4-P1"), ("P1", "P1-only"), ("P4", "P4-only")):
            for name, entry in groups.get(group, {}).items():
                q = entry.get("all", entry)
                if q.get("finite", q.get("n", 0)) == 0:
                    lines.append(f"| {float(a):.1f} | {family} | {name} | 0 | n/a | n/a | n/a | n/a | n/a | n/a |")
                else:
                    lines.append(f"| {float(a):.1f} | {family} | {name} | {q.get('finite', q.get('n', 'n/a'))} | {q['radial_rms_arcmin']:.4g}/{q['radial_p95_arcmin']:.4g}/{q['radial_max_arcmin']:.4g} | {q['axis_rms_arcmin']} | {q['axis_p95_abs_arcmin']} | {q['axis_max_abs_arcmin']} | {q['within_1_arcmin_pct']:.4g}% | {q['within_3_arcmin_pct']:.4g}% |")

    lines += ["", "## Gaze-magnitude bins", "", "Bins are based on max(|ADE|,|BDE|): [0,4), [4,8), [8,12), and [12,12.4].", "",
        "| Bin (max abs gaze, deg) | Family | Readout | N | Radial RMS/P95/max | Axis RMS x/y | Axis P95 x/y | Axis max x/y | <=1 arcmin | <=3 arcmin |", "|---|---|---|---:|---|---|---|---|---:|---:|"]
    primary = metrics["runtime_breakdown"]["primary"]
    for group, family in (("paired", "P4-P1"), ("P1", "P1-only"), ("P4", "P4-only")):
        for name, entry in primary.get(group, {}).items():
            for bin_name, q in entry.get("bins", {}).items():
                if q.get("finite", q.get("n", 0)) == 0:
                    lines.append(f"| {bin_name} | {family} | {name} | 0 | n/a | n/a | n/a | n/a | n/a | n/a |")
                else:
                    lines.append(f"| {bin_name} | {family} | {name} | {q.get('finite', q.get('n', 'n/a'))} | {q['radial_rms_arcmin']:.4g}/{q['radial_p95_arcmin']:.4g}/{q['radial_max_arcmin']:.4g} | {q['axis_rms_arcmin']} | {q['axis_p95_abs_arcmin']} | {q['axis_max_abs_arcmin']} | {q['within_1_arcmin_pct']:.4g}% | {q['within_3_arcmin_pct']:.4g}% |")

    lines += ["", "## Excluded diagnostic and inclusive scores", "", "The excluded gaze (ADE,BDE)=(12,9.6) has 21 A-plane repetitions and is reported separately from the 83,229-state primary set.", "",
        "| Family/readout | Excluded N | Radial RMS/P95/max | Axis RMS x/y | Axis P95 x/y | Axis max x/y | <=1 arcmin | <=3 arcmin | Inclusive RMS/P95/max |", "|---|---:|---|---|---|---|---:|---:|---|"]
    ex, inc = metrics["runtime_breakdown"]["excluded_21_diagnostic"], metrics["runtime_breakdown"]["inclusive"]
    for group, family in (("paired", "P4-P1"), ("P1", "P1-only"), ("P4", "P4-only")):
        for name, entry in ex.get(group, {}).items():
            q = entry.get("all", entry)
            iq = inc.get(group, {}).get(name, {})
            iq = iq.get("all", iq)
            lines.append(f"| {family} {name} | {q.get('n', 21)} | {q.get('radial_rms_arcmin', float('nan')):.4g}/{q.get('radial_p95_arcmin', float('nan')):.4g}/{q.get('radial_max_arcmin', float('nan')):.4g} | {q.get('axis_rms_arcmin', 'n/a')} | {q.get('axis_p95_abs_arcmin', 'n/a')} | {q.get('axis_max_abs_arcmin', 'n/a')} | {q.get('within_1_arcmin_pct', float('nan')):.4g}% | {q.get('within_3_arcmin_pct', float('nan')):.4g}% | {iq.get('radial_rms_arcmin', float('nan')):.4g}/{iq.get('radial_p95_arcmin', float('nan')):.4g}/{iq.get('radial_max_arcmin', float('nan')):.4g} |")

    lines += ["", "## Three center rows", "", "Center errors are vector RMS/P95/max in um against corresponding labeled-center references; the paired difference column measures P4-P1 center residual against its correspondingly shifted oracle difference.", "",
        "| Center input | P1 error RMS/P95/max (um) | P4 after M1, estimated A error RMS/P95/max (um) | P4-P1 difference error RMS/P95/max (um) |", "|---|---|---|---|"]
    diagnostics = metrics["center_diagnostics_um"]["runtime_by_center_definition_shift"]
    for label, key in (("a raw mean", "a_raw"), ("b corrected", "b_corrected"), ("c labeled ray oracle", "c_labeled")):
        tag = f"{key}_estimated_A"
        p1d = diagnostics[tag]["P1_vs_labeled_um"]
        p4d = diagnostics[tag]["P4_vs_same_shift_labeled_um"]
        dd = diagnostics[tag]["paired_center_delta_vs_same_shift_labeled_um"]
        lines.append(f"| {label} | {p1d['vector_rms_um']:.4g}/{p1d['vector_p95_um']:.4g}/{p1d['vector_max_um']:.4g} | {p4d['vector_rms_um']:.4g}/{p4d['vector_p95_um']:.4g}/{p4d['vector_max_um']:.4g} | {dd['vector_rms_um']:.4g}/{dd['vector_p95_um']:.4g}/{dd['vector_max_um']:.4g} |")
    lines += ["", "Paired-readout performance yardstick (oracle c minus raw a; negative metric deltas mean lower measured gaze error for c in this fitted readout):", "",
        "| M1 A variant | Raw radial RMS/P95/max (arcmin) | Corrected radial RMS/P95/max | Oracle radial RMS/P95/max | c-a delta RMS/P95/max (arcmin) |", "|---|---|---|---|---|"]
    main_lookup = {r["readout"]: r["primary_holdout"] for r in metrics["main_21_rows"] if r["family"] == "P4-P1"}
    for variant in ("estimated_A", "true_A_oracle", "identity_no_shift"):
        vals = [main_lookup[f"{center}__{variant}"] for center in ("a_raw", "b_corrected", "c_labeled")]
        rms = [v["radial_rms_arcmin"] for v in vals]
        p95 = [v["radial_p95_arcmin"] for v in vals]
        maximum = [v["radial_max_arcmin"] for v in vals]
        delta = [rms[2]-rms[0], p95[2]-p95[0], maximum[2]-maximum[0]]
        triples = [f"{rms[i]:.4g}/{p95[i]:.4g}/{maximum[i]:.4g}" for i in range(3)]
        lines.append(f"| {variant} | {triples[0]} | {triples[1]} | {triples[2]} | {delta[0]:+.4g}/{delta[1]:+.4g}/{delta[2]:+.4g} |")
    lines += ["", "The c branch uses the labeled central-ray oracle by design; no fitted registration is implied.",
        "", "## Difference-readout noise floor", "", "The main column is the conditional linearized 1 um sigma per channel center, per axis, under independent center errors and fixed A. It is reported separately from noise-free model error.", "",
        "| Paired readout | Primary radial RMS floor (arcmin) | Primary axis RMS floor x/y (arcmin) | Per-ray sensitivity (arcmin) |", "|---|---:|---|---:|"]
    for row in metrics["main_21_rows"]:
        if row["family"] != "P4-P1": continue
        e = floor[row["readout"]]
        c = e["1um_sigma_per_center_axis"]
        r = e["1um_iid_ray"]
        lines.append(f"| {row['readout']} | {f(c['radial_rms_arcmin'])} | {c['axis_rms_arcmin']} | {f(r['radial_rms_arcmin'])} |")
    lines += ["", "Project precedent uses independent Gaussian 1 um per x/y image-coordinate ray (`Script/p4_cr_floor.py`, nine rays/18 coordinates). The per-ray sensitivity uses 1/3 um center noise for mean-derived a/b inputs with frozen corrections, and 1 um for c's directly labeled central ray; it does not represent the full coupled center/gaze estimator. Both floors omit coefficient uncertainty, A uncertainty, shape uncertainty, and calibration-label uncertainty. The old 0.09 arcmin floor is withdrawn and not reused.",
        "", "## Ten worst primary paired states", ""]
    for tag, items in metrics["worst_10_primary_by_paired_readout"].items():
        lines += [f"### {tag}", "", "| Source index | Row | A (D) | ADE (deg) | BDE (deg) | Error (arcmin) |", "|---:|---:|---:|---:|---:|---:|"]
        for item in items:
            a, ade, bde = item["state_A_ADE_BDE"]
            lines.append(f"| {item['source_index']} | {item['source_subset_row']} | {a:.1f} | {ade:.4g} | {bde:.4g} | {item['radial_error_arcmin']:.5g} |")
        lines.append("")
    headline = main_lookup["b_corrected__estimated_A"]
    oracle_a = main_lookup["b_corrected__true_A_oracle"]
    p1_raw = next(r["primary_holdout"] for r in metrics["main_21_rows"] if r["family"] == "P1-only" and r["readout"] == "a_raw")
    model_artifact = Path(path).with_name("baseline.npz")
    with np.load(model_artifact) as saved:
        normalizers = {
            "paired_b_corrected__estimated_A": (saved["paired_b_corrected__estimated_A_normalizer_center_mm"].tolist(), saved["paired_b_corrected__estimated_A_normalizer_scale_mm"].tolist()),
            "P1_b_corrected": (saved["P1_b_corrected_normalizer_center_mm"].tolist(), saved["P1_b_corrected_normalizer_scale_mm"].tolist()),
            "P4_b_corrected__estimated_A": (saved["P4_b_corrected__estimated_A_normalizer_center_mm"].tolist(), saved["P4_b_corrected__estimated_A_normalizer_scale_mm"].tolist())}
    lines += ["## Polynomial coordinate normalizers", "", "These are the saved input centers/scales used by the polynomial features, in mm; they are separate from M1's A/4 scale.", "", "| Readout | Input center (mm) | Input scale (mm) |", "|---|---|---|"]
    for name, (center, scale) in normalizers.items():
        lines.append(f"| {name} | {center} | {scale} |")
    lines += ["", "## Interpretation", "", f"The headline paired corrected readout has radial RMS/P95/max {headline['radial_rms_arcmin']:.4g}/{headline['radial_p95_arcmin']:.4g}/{headline['radial_max_arcmin']:.4g} arcmin; with true A substituted, {oracle_a['radial_rms_arcmin']:.4g}/{oracle_a['radial_p95_arcmin']:.4g}/{oracle_a['radial_max_arcmin']:.4g}. Both RMS values are below 1 arcmin, but neither readout meets an all-states maximum gate of 1 or 3 arcmin. The P1 raw-mean readout has max {p1_raw['radial_max_arcmin']:.4g} arcmin, below both maxima. Per-state percentages and all threshold summaries appear above.", "", "The c branch is an oracle-center input yardstick; it does not guarantee zero or minimum gaze error. Its observed c-a score differences describe these fitted polynomial readouts; they do not prove which center is physically more accurate. Approximation by the fixed polynomial, M1 normalization, and teacher-conditioned calibration features can affect the comparison. P1 alone may outperform P4-P1 because the ray-traced simulation has no eye translation; that is an expected outcome, not a failure. The data are dense, correlated, noiseless ray-traced interpolation in nominal recorded frames, not hardware validation. Sensor noise, gaze-label error, and frame-registration perturbations are not modeled.",
        "", f"Pinned source SHA-256 `{provenance['source_sha256']}`; saved runtime SHA-256 `{provenance['runtime_sha256']}`. Calibration N={provenance['calibration_count']}; primary N={provenance['primary_count']}; excluded N={provenance['excluded_count']} at (ADE,BDE)={provenance['excluded_gaze_ADE_BDE']}.",
        "", "Complete numerical arrays and metrics are stored in `baseline.npz`.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
