"""A = 0 D, BDE = 0, ADE-only: separable keystone K followed by a decentering (Brown-Conrady p1-type) term.

Data slice and loader: imported unchanged from fit_p4_ade_fixed_accommodation.py (A = 0 and BDE = 0 via
np.isclose atol 1e-9, 101 ADE x 9 fields, real_x/y_mm center-relative, physical x/y as stored in the pickle).

Model, in normalized coordinates (L = 1.235429592627 mm, coefficients dimensionless):
    b = r0 / L                                   (r0 = ADE = 0 real grid; barrel is the identity at A = 0)
    xk = s_x b_x / (1 + q b_y),  yk = s_y b_y / (1 + q b_y)            (K, as in the kept script)
    then, applied AFTER K:
    (a) p1 Brown-Conrady:   X = xk + 2 p1 xk yk,            Y = yk + p1 (xk^2 + 3 yk^2)
    (b) Y-only smile:       X = xk,                          Y = yk + c xk^2
    (c) Y-only general:     X = xk,                          Y = yk + c (xk^2 + d yk^2)
    (d) p1 + p2 BC:         X = xk + 2 p1 xk yk + p2 (3 xk^2 + yk^2),  Y = yk + p1 (xk^2 + 3 yk^2) + 2 p2 xk yk
    prediction = L * (X, Y).
Fits: per-state (all params jointly per ADE), then polynomial in t = ADE/20 for the chosen variant
(train on even ADE indices, test on odd).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "6")
import sys
sys.dont_write_bytecode = True

import hashlib
import time
import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Script"))
from fit_p4_ade_fixed_accommodation import load, metrics, per_state_metrics, BINS, GRID_PATH  # noqa: E402

COEF_PATH = ROOT / "data" / "p4_biaxial_joint" / "p4_biaxial_joint_coefficients.npz"
OUT = ROOT / "data" / "p4_ade_a0_decenter"
UM = 1e3
FIELD_CLASSES = {"corners": [0, 2, 6, 8], "x-axis edge mid (+-1,0) [1,7]": [1, 7],
                 "y-axis edge mid (0,+-1) [3,5]": [3, 5], "center [4]": [4], "all nine": list(range(9))}
XEDGE = [1, 7]
OFFC = [0, 1, 2, 3, 5, 6, 7, 8]

# variant -> names of extra params (after s_x, s_y, q)
VARIANTS = {"K only": [], "(a) BC p1": ["p1"], "(b) Y += c xk^2": ["c"],
            "(c) Y += c(xk^2 + d yk^2)": ["c", "d"], "(d) BC p1 + p2": ["p1", "p2"]}


def model(b, p, var):
    """b (9,2) normalized; p = (s_x, s_y, q, extras...) -> normalized prediction (9,2)."""
    sx, sy, q = p[0], p[1], p[2]
    den = 1 + q * b[:, 1]
    xk, yk = sx * b[:, 0] / den, sy * b[:, 1] / den
    X, Y = xk.copy(), yk.copy()
    if var == "(a) BC p1":
        p1 = p[3]
        X += 2 * p1 * xk * yk
        Y += p1 * (xk ** 2 + 3 * yk ** 2)
    elif var == "(b) Y += c xk^2":
        Y += p[3] * xk ** 2
    elif var == "(c) Y += c(xk^2 + d yk^2)":
        Y += p[3] * (xk ** 2 + p[4] * yk ** 2)
    elif var == "(d) BC p1 + p2":
        p1, p2 = p[3], p[4]
        X += 2 * p1 * xk * yk + p2 * (3 * xk ** 2 + yk ** 2)
        Y += p1 * (xk ** 2 + 3 * yk ** 2) + 2 * p2 * xk * yk
    return np.stack((X, Y), -1)


def per_state(b, real, L, ade, var):
    n, k = len(ade), 3 + len(VARIANTS[var])
    P = np.full((n, k), np.nan)
    cond = np.full(n, np.nan); cond_s = np.full(n, np.nan)
    corr = np.full((n, k, k), np.nan)
    i0 = int(np.flatnonzero(np.isclose(ade, 0, atol=1e-9))[0])
    order = [i0] + [i for kk in range(1, n) for i in (i0 + kk, i0 - kk) if 0 <= i < n]
    for i in order:
        nb = [j for j in (i - 1, i + 1) if 0 <= j < n and np.isfinite(P[j, 0])]
        x0 = P[nb[0]].copy() if nb else np.r_[1., 1., 0., np.zeros(k - 3)]
        if var.startswith("(c)") and not nb:
            x0[4] = 1.0  # d undefined when c = 0; start at 1
        f = lambda p: ((model(b, p, var) * L - real[i]) * UM).ravel()
        r = least_squares(f, x0, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=20000)
        P[i] = r.x
        J = r.jac
        cn = np.linalg.norm(J, axis=0)
        if np.any(cn < 1e-12 * cn.max()):   # degenerate column (e.g. d when c = 0 at ADE = 0)
            continue
        cond[i] = np.linalg.cond(J)
        cond_s[i] = np.linalg.cond(J / cn)
        try:
            C = np.linalg.inv(J.T @ J)
            sd = np.sqrt(np.diag(C))
            corr[i] = C / np.outer(sd, sd)
        except np.linalg.LinAlgError:
            pass
    pred = np.stack([model(b, P[i], var) for i in range(n)]) * L
    return P, pred - real, cond, cond_s, corr


# ----------------------------------------------------------------- polynomial in t
def poly_spec(extra, parity):
    """K: q odd {1,3}, s even {2,4} (parity) or all {1..4}; extras odd {1,3} (parity) or all {1..4}."""
    if parity:
        return {"qp": [1, 3], "sp": [2, 4], "ep": [1, 3], "extra": extra, "parity": True}
    return {"qp": [1, 2, 3, 4], "sp": [1, 2, 3, 4], "ep": [1, 2, 3, 4], "extra": extra, "parity": False}


def poly_eval(spec, p, t):
    B = lambda pw: np.stack([t ** j for j in pw], -1)
    nq, ns, ne = len(spec["qp"]), len(spec["sp"]), len(spec["ep"])
    i = 0
    q = B(spec["qp"]) @ p[i:i + nq]; i += nq
    sx = 1 + B(spec["sp"]) @ p[i:i + ns]; i += ns
    sy = 1 + B(spec["sp"]) @ p[i:i + ns]; i += ns
    ex = []
    for _ in spec["extra"]:
        ex.append(B(spec["ep"]) @ p[i:i + ne]); i += ne
    return sx, sy, q, ex


def poly_predict(spec, p, t, b, var_poly):
    sx, sy, q, ex = poly_eval(spec, p, t)
    out = []
    for k in range(len(t)):
        out.append(model_poly(b, sx[k], sy[k], q[k], [e[k] for e in ex], var_poly))
    return np.stack(out)


def model_poly(b, sx, sy, q, ex, var_poly):
    # for (c) the polynomial is on c1 = c and c2 = c*d (both odd); map back to model params
    if var_poly.startswith("(c)"):
        c1, c2 = ex
        den = 1 + q * b[:, 1]
        xk, yk = sx * b[:, 0] / den, sy * b[:, 1] / den
        return np.stack((xk, yk + c1 * xk ** 2 + c2 * yk ** 2), -1)
    return model(b, np.r_[sx, sy, q, ex], var_poly)


def poly_fit(spec, var_poly, b, real, L, t, P, mask):
    B = lambda pw: np.stack([t[mask] ** j for j in pw], -1)
    targets = [P[mask, 2], P[mask, 0] - 1, P[mask, 1] - 1]
    if var_poly.startswith("(c)"):
        targets += [P[mask, 3], P[mask, 3] * P[mask, 4]]
    else:
        targets += [P[mask, 3 + k] for k in range(len(spec["extra"]))]
    pws = [spec["qp"], spec["sp"], spec["sp"]] + [spec["ep"]] * len(spec["extra"])
    x0 = np.concatenate([np.linalg.lstsq(B(pw), y, rcond=None)[0] for pw, y in zip(pws, targets)])

    def f(p):
        return ((poly_predict(spec, p, t[mask], b, var_poly) * L - real[mask]) * UM).ravel()
    r = least_squares(f, x0, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=50000)
    return r.x, float(np.linalg.cond(r.jac)), int(r.status)


def under1_range(ade, v):
    """largest A such that every state with |ADE| <= A has v < 1 um."""
    aa = np.abs(ade)
    best = None
    for a in np.unique(aa):
        if np.all(v[aa <= a + 1e-9] < 1.0):
            best = a
        else:
            break
    return best


def bin_mask(aa, lo, hi):
    return (aa <= hi + 1e-9) if lo == 0 else (aa > lo + 1e-9) & (aa <= hi + 1e-9)


# --------------------------------------------------------------------------- main
def main():
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ade, fields, real, sha, nrows = load()
    t = ade / 20.0
    L = float(np.load(COEF_PATH, allow_pickle=True)["L_mm"])
    i0 = int(np.flatnonzero(np.isclose(ade, 0, atol=1e-9))[0])
    r0 = real[i0]
    b = r0 / L
    aa = np.abs(ade)
    ip20, im20 = int(np.argmin(np.abs(ade - 20))), int(np.argmin(np.abs(ade + 20)))
    even = np.zeros(101, bool); even[::2] = True
    odd = ~even

    res = {}
    for var in VARIANTS:
        P, err, cond, cond_s, corr = per_state(b, real, L, ade, var)
        res[var] = {"P": P, "err": err, "cond": cond, "cond_s": cond_s, "corr": corr}

    # ---- choose variant: lowest pooled ceiling vRMS, prefer fewer params when within 5 %
    cand = [v for v in VARIANTS if v != "K only"]
    vr = {v: metrics(res[v]["err"])[0] for v in cand}
    best = min(cand, key=lambda v: vr[v])
    chosen = min([v for v in cand if vr[v] <= 1.05 * vr[best]], key=lambda v: (len(VARIANTS[v]), vr[v]))
    # decisive-check numbers
    def ycheck(err, i):
        e = err[i] * UM
        return e[1, 1], e[7, 1], e[OFFC, 1].mean()
    collapse = {}
    for v in VARIANTS:
        ys = [abs(x) for i in (ip20, im20) for x in ycheck(res[v]["err"], i)]
        collapse[v] = max(ys)
    k_before = collapse["K only"]

    # ---- polynomials for the chosen variant (and K-only reference)
    polys = []
    for var_poly, parity in [("K only", True), ("K only", False), (chosen, True), (chosen, False)]:
        extra = VARIANTS[var_poly] if not var_poly.startswith("(c)") else ["c1", "c2=c*d"]
        spec = poly_spec(extra, parity)
        P = res[var_poly]["P"]
        p_all, cond_all, st_all = poly_fit(spec, var_poly, b, real, L, t, P, np.ones(101, bool))
        p_h, cond_h, st_h = poly_fit(spec, var_poly, b, real, L, t, P, even)
        err_all = poly_predict(spec, p_all, t, b, var_poly) * L - real
        err_h = poly_predict(spec, p_h, t, b, var_poly) * L - real
        polys.append({"var": var_poly, "spec": spec, "p": p_all, "ph": p_h, "cond": cond_all, "condh": cond_h,
                      "status": (st_all, st_h), "err": err_all, "errh": err_h,
                      "name": f"{var_poly} poly {'parity' if parity else 'all powers 1..4'}"})

    # ---- npz
    sv = {}
    for k, v in enumerate(VARIANTS):
        sv[f"variant{k}_per_state_params"] = res[v]["P"]
        sv[f"variant{k}_per_state_vrms_um"] = per_state_metrics(res[v]["err"])[0]
        sv[f"variant{k}_per_state_maxpt_um"] = per_state_metrics(res[v]["err"])[2]
        sv[f"variant{k}_cond_J"] = res[v]["cond"]
        sv[f"variant{k}_cond_J_colscaled"] = res[v]["cond_s"]
        sv[f"variant{k}_param_corr"] = res[v]["corr"]
    for k, d in enumerate(polys):
        sv[f"poly{k}_coefficients_all"] = d["p"]
        sv[f"poly{k}_coefficients_holdout_even"] = d["ph"]
    np.savez(OUT / "p4_ade_a0_decenter.npz", ade_deg=ade, t=t, fields=fields, real_mm=real, r0_mm=r0, L_mm=L,
             variant_names=np.array(list(VARIANTS)),
             variant_param_names=np.array([",".join(["s_x", "s_y", "q_n"] + VARIANTS[v]) for v in VARIANTS]),
             poly_names=np.array([d["name"] for d in polys]), chosen_variant=chosen, grid_sha256=sha, **sv)

    plot(ade, res, polys, chosen, t)

    # ---- report
    R = []; w = R.append
    w("# P4, A = 0 D, ADE only: keystone K + decentering (Brown-Conrady p1-type) term\n")
    w(f"Generated by `Script/fit_p4_ade_a0_decenter.py` ({time.strftime('%Y-%m-%d %H:%M')}, run time "
      f"{time.time() - t_start:.1f} s). Input `data/distortion_grid_p4_biaxial_macro/distortion_grid.pkl` "
      f"(sha256 {sha[:16]}...), loader imported unchanged from `Script/fit_p4_ade_fixed_accommodation.py`: "
      f"A = 0 and BDE = 0 via `np.isclose(atol=1e-9)`, {nrows} rows = 101 ADE x 9 fields.\n")
    w("**Coordinate convention.** real_x_mm / real_y_mm are center-relative and used **as stored in the pickle** "
      "(physical x, y of the grid file), not the display-swapped axes of the Theory.md figures. 'x-axis edge "
      "midpoints' = fields with field_x_relative = +-1, field_y_relative = 0 (indices 1, 7), which lie on the stored "
      "real x axis; 'y-axis edge midpoints' = (0, +-1) (indices 3, 5). Errors are pred - real.\n")
    w("**Model.** b = r0/L (L = %.12f mm; r0 = ADE = 0 grid; barrel is the identity at A = 0). K: xk = s_x b_x/(1+q b_y), "
      "yk = s_y b_y/(1+q b_y). Decentering applied after K, on (xk, yk), dimensionless:\n" % L)
    w("```\n(a) X = xk + 2 p1 xk yk,                    Y = yk + p1 (xk^2 + 3 yk^2)            [s_x,s_y,q,p1]\n"
      "(b) X = xk,                                  Y = yk + c xk^2                        [s_x,s_y,q,c]\n"
      "(c) X = xk,                                  Y = yk + c (xk^2 + d yk^2)             [s_x,s_y,q,c,d]\n"
      "(d) X = xk + 2 p1 xk yk + p2 (3xk^2 + yk^2), Y = yk + p1 (xk^2 + 3yk^2) + 2 p2 xk yk [s_x,s_y,q,p1,p2]\n"
      "prediction = L (X, Y)\n```\n")
    w("Metrics (um): vertex RMS = sqrt(mean |e|^2), coord RMSE = vertex RMS/sqrt(2), max = max point |e|.\n")

    # 1. ceilings
    w("## 1. Per-state ceilings (all params fitted jointly per ADE, LM, warm start outward from ADE = 0)\n")
    w("| variant | params/state | vRMS | cRMSE | max pt | vRMS vs K-only |")
    w("|---|---|---|---|---|---|")
    m0 = metrics(res["K only"]["err"])
    for v in VARIANTS:
        m = metrics(res[v]["err"])
        w(f"| {v} | {3 + len(VARIANTS[v])} | {m[0]:.4f} | {m[1]:.4f} | {m[2]:.4f} | {m[0] / m0[0] * 100:.1f} % |")
    w("")
    w("By |ADE| bin (vRMS / max pt, um):\n")
    w("| |ADE| bin | " + " | ".join(VARIANTS) + " |")
    w("|---|" + "---|" * len(VARIANTS))
    for lo, hi in BINS:
        mk = bin_mask(aa, lo, hi)
        w(f"| {lo}-{hi} ({mk.sum()} st) | " + " | ".join(
            "%.3f / %.3f" % (metrics(res[v]["err"][mk])[0], metrics(res[v]["err"][mk])[2]) for v in VARIANTS) + " |")
    w("\nBy field class, pooled over 101 ADE (vRMS / max pt, um):\n")
    w("| class | " + " | ".join(VARIANTS) + " |")
    w("|---|" + "---|" * len(VARIANTS))
    for cn, ix in FIELD_CLASSES.items():
        w(f"| {cn} | " + " | ".join(
            "%.3f / %.3f" % (metrics(res[v]["err"], ix)[0], metrics(res[v]["err"], ix)[2]) for v in VARIANTS) + " |")
    w("")

    # 2. decisive check
    w("## 2. Decisive check at ADE = +-20: residual Y at the x-axis edge midpoints and common Y offset\n")
    w("Residual Y (pred - real, um) at fields (-1,0) [idx 1] and (+1,0) [idx 7]; common Y offset = mean residual Y "
      "over the 8 off-center points.\n")
    w("| variant | ADE | eY(-1,0) | eY(+1,0) | mean eY off-center | max point at this ADE |")
    w("|---|---|---|---|---|---|")
    for v in VARIANTS:
        for i in (im20, ip20):
            a1, a2, a3 = ycheck(res[v]["err"], i)
            w(f"| {v} | {ade[i]:+.0f} | {a1:+.4f} | {a2:+.4f} | {a3:+.4f} | "
              f"{per_state_metrics(res[v]['err'])[2][i]:.4f} |")
    w("")
    w("Max |value| among those three quantities at ADE = +-20: " +
      ", ".join(f"{v}: {collapse[v]:.4f} um" for v in VARIANTS) + ".\n")
    collapsed = {v: collapse[v] < 0.1 * k_before for v in cand}
    kx = abs(ycheck(res["K only"]["err"], ip20)[1]); ko = abs(ycheck(res["K only"]["err"], ip20)[2])
    w("Verdict (collapse = every one of the three drops below 10 % of the K-only value): " +
      ", ".join(f"{v}: {'collapses' if collapsed[v] else 'does NOT collapse'} (x-edge eY "
                f"{abs(ycheck(res[v]['err'], ip20)[1]) / kx * 100:.0f} %, common offset "
                f"{abs(ycheck(res[v]['err'], ip20)[2]) / ko * 100:.0f} % of K-only)" for v in cand) + ".\n")
    # even-even part of the K-only Y residual at +20 by field class (what an even Y term must supply)
    e = res["K only"]["err"][ip20] * UM
    ce, xe, ye = e[[0, 2, 6, 8], 1].mean(), e[XEDGE, 1].mean(), e[[3, 5], 1].mean()
    w(f"Even-even (in x and y) part of the K-only Y residual at ADE = +20: corners {ce:+.3f}, x-edge {xe:+.3f}, "
      f"y-edge {ye:+.3f} um. A pure quadratic c(x^2 + d y^2) would need corners = x-edge + y-edge = "
      f"{xe + ye:+.3f} um; the shortfall {ce - xe - ye:+.3f} um is an x^2 y^2-type (or higher-order) "
      "component, which no p1/quadratic term supplies (diagnostic only; nothing added, and K re-fits jointly so "
      "this is indicative, not exact).\n")

    # 3. parameters, correlation, conditioning
    w("## 3. Per-state parameters, correlation with q, Jacobian conditioning\n")
    for v in VARIANTS:
        P = res[v]["P"]
        names = ["s_x", "s_y", "q_n"] + VARIANTS[v]
        w(f"### {v}\n")
        w("| ADE | " + " | ".join(names) + " | vRMS | max pt | cond(J) | cond(J col-scaled) | "
          + " | ".join(f"corr({n},q)" for n in names if n not in ("q_n",)) + " |")
        w("|---|" + "---|" * (len(names) + 4 + len(names) - 1))
        vv, _, mm = per_state_metrics(res[v]["err"])
        for a in (-20, -16, -12, -8, -4, -0.4, 0, 0.4, 4, 8, 12, 16, 20):
            i = int(np.argmin(np.abs(ade - a)))
            cr = res[v]["corr"][i]
            cells = [f"{P[i, k]:+.6e}" for k in range(len(names))]
            cc = [f"{cr[k, 2]:+.4f}" for k in range(len(names)) if k != 2]
            w(f"| {ade[i]:+.1f} | " + " | ".join(cells) + f" | {vv[i]:.4f} | {mm[i]:.4f} | {res[v]['cond'][i]:.2e} | "
              f"{res[v]['cond_s'][i]:.2e} | " + " | ".join(cc) + " |")
        nz = ~np.isclose(ade, 0, atol=1e-9)
        if VARIANTS[v]:
            mc = [np.nanmax(np.abs(res[v]["corr"][nz, 3 + k, 2])) for k in range(len(VARIANTS[v]))]
            w("\nmax over ADE != 0: " + ", ".join(f"|corr({n},q)| = {c:.4f}" for n, c in zip(VARIANTS[v], mc))
              + f"; cond(J) range {np.nanmin(res[v]['cond']):.2e} .. {np.nanmax(res[v]['cond']):.2e}; "
              f"col-scaled {np.nanmin(res[v]['cond_s']):.2e} .. {np.nanmax(res[v]['cond_s']):.2e}.\n")
            par = []
            for k, n in enumerate(VARIANTS[v]):
                x = P[:, 3 + k]
                par.append(f"{n}: max|f(ADE)+f(-ADE)| = {np.max(np.abs(x + x[::-1])):.2e}, "
                           f"max|f(ADE)-f(-ADE)| = {np.max(np.abs(x - x[::-1])):.2e}")
            w("Parity of extra per-state params: " + "; ".join(par) + ".\n")
        else:
            w("")
    w("cond(J): raw Jacobian of the 18 residuals (um) w.r.t. the params; col-scaled = columns normalized to unit "
      "norm (removes the scale differences between s, q and the extra terms). corr from (J^T J)^-1.\n")

    # 4. polynomial
    w("## 4. Polynomial in t = ADE/20 for the chosen variant\n")
    w(f"Variant chosen: **{chosen}** (lowest pooled ceiling vRMS {vr[best]:.4f} um is {best}; rule: fewest params "
      f"within 5 % of that). K terms: q odd (t, t^3), s_x, s_y even (t^2, t^4) as in the kept order-4 K polynomial; "
      "extra term odd (e1 t + e3 t^3)" + (" for both c1 = c and c2 = c*d" if chosen.startswith("(c)") else "") +
      ". 'all powers' = every coefficient on t..t^4 (parity check). Holdout: fit on the 51 even ADE indices, "
      "test on the 50 odd ones (ADE = +-20 are training points).\n")
    w("| model | n coef | in-sample vRMS | in cRMSE | in max | holdout-odd vRMS | holdout-odd cRMSE | holdout-odd max | "
      "cond(J) | LM status |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for d in polys:
        m1, m2 = metrics(d["err"]), metrics(d["errh"][odd])
        w(f"| {d['name']} | {len(d['p'])} | {m1[0]:.4f} | {m1[1]:.4f} | {m1[2]:.4f} | {m2[0]:.4f} | {m2[1]:.4f} | "
          f"{m2[2]:.4f} | {d['cond']:.1e} | {d['status']} |")
    mc_ = metrics(res[chosen]["err"])
    w(f"| {chosen} per-state ceiling | {101 * (3 + len(VARIANTS[chosen]))} | {mc_[0]:.4f} | {mc_[1]:.4f} | "
      f"{mc_[2]:.4f} | | | | | |")
    w("\nReference from the kept K-only script: order-4 K polynomial holdout 2.556 vRMS / 10.0 max um.\n")
    w("### Coefficients (fit to all 101 ADE; holdout-fit coefficients in the npz)\n")
    for d in polys:
        sp, p = d["spec"], d["p"]
        i = 0; parts = []
        for lab, pw in [("q_n", sp["qp"]), ("s_x-1", sp["sp"]), ("s_y-1", sp["sp"])] + \
                [(e, sp["ep"]) for e in sp["extra"]]:
            parts.append(f"{lab} [" + ", ".join(f"t^{j}: {c:+.6e}" for j, c in zip(pw, p[i:i + len(pw)])) + "]")
            i += len(pw)
        w(f"- **{d['name']}**: " + "; ".join(parts))
    dall = polys[3]; sp = dall["spec"]; p = dall["p"]
    wrong, right = [], []
    i = 0
    for kind, pw in [("odd", sp["qp"]), ("even", sp["sp"]), ("even", sp["sp"])] + [("odd", sp["ep"])] * len(sp["extra"]):
        for j, c in zip(pw, p[i:i + len(pw)]):
            ok = (j % 2 == 1) if kind == "odd" else (j % 2 == 0)
            (right if ok else wrong).append(abs(c))
        i += len(pw)
    w(f"\nParity check ({dall['name']}): parity-forbidden coefficients max |coef| {max(wrong):.1e}; allowed "
      f"coefficients min |coef| {min(right):.1e}.\n")

    w("### Polynomial errors by |ADE| bin (vRMS / max pt, um)\n")
    show = [("K-only ceiling", res["K only"]["err"], None), (f"{chosen} ceiling", res[chosen]["err"], None)] + \
           [(d["name"] + " in-sample", d["err"], None) for d in polys] + \
           [(d["name"] + " holdout (odd)", d["errh"], odd) for d in polys]
    w("| |ADE| bin | " + " | ".join(n for n, _, _ in show) + " |")
    w("|---|" + "---|" * len(show))
    for lo, hi in BINS:
        mk = bin_mask(aa, lo, hi)
        cells = []
        for n, e, sel in show:
            mm_ = mk & sel if sel is not None else mk
            m1 = metrics(e[mm_])
            cells.append(f"{m1[0]:.3f} / {m1[2]:.3f}")
        w(f"| {lo}-{hi} | " + " | ".join(cells) + " |")
    w("\n### Polynomial errors by field class, pooled (vRMS / max pt, um)\n")
    w("| class | " + " | ".join(n for n, _, _ in show) + " |")
    w("|---|" + "---|" * len(show))
    for cn, ix in FIELD_CLASSES.items():
        cells = []
        for n, e, sel in show:
            ee = e[sel] if sel is not None else e
            m1 = metrics(ee, ix)
            cells.append(f"{m1[0]:.3f} / {m1[2]:.3f}")
        w(f"| {cn} | " + " | ".join(cells) + " |")

    w("\n### |ADE| range staying under 1 um (every state with |ADE| <= A)\n")
    w("| model | vRMS < 1 um up to | max pt < 1 um up to |")
    w("|---|---|---|")
    rows = [("K-only ceiling", res["K only"]["err"]), (f"{chosen} ceiling", res[chosen]["err"])] + \
           [(d["name"] + " (holdout coefficients, all 101 states)", d["errh"]) for d in polys]
    for n, e in rows:
        v_, _, m_ = per_state_metrics(e)
        a1, a2 = under1_range(ade, v_), under1_range(ade, m_)
        w(f"| {n} | {f'{a1:.1f}' if a1 is not None else 'none'} deg | {f'{a2:.1f}' if a2 is not None else 'none'} deg |")
    w("")

    # 5. residual pattern at +20 for chosen variant
    w("## 5. Residual pattern at ADE = +20 (pred - real, um, fields 0..8)\n")
    for v in ("K only", chosen):
        e = res[v]["err"][ip20] * UM
        w(f"- {v}: " + "; ".join(f"{i}{tuple(int(x) for x in fields[i])}: ({e[i, 0]:+.3f}, {e[i, 1]:+.3f})"
                                 for i in range(9)))
    w("")
    w("## 6. Conclusion\n")
    if not any(collapsed.values()):
        w("The x-axis-edge Y residual at ADE = +-20 does **not** collapse under any p1-type variant (with (a)/(d) "
          "it just flips sign at about the same size; (c) reduces it by about half), even though the common "
          "Y offset is largely absorbed by (a), (c) and (d). **p1 (decentering) is not the missing term.** p2 comes out "
          "identically 0 (x-mirror symmetry). No further terms were added.\n")
    else:
        w("Collapsing variants: " + ", ".join(v for v in cand if collapsed[v]) + ". See tables for the remaining "
          "residual.\n")
    (OUT / "report.md").write_text("\n".join(R) + "\n", encoding="utf-8")
    print("\n".join(R))


def plot(ade, res, polys, chosen, t):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid = "#1f2328", "#59636e", "#e6e8eb"
    cols = {"K only": "#59636e", "(a) BC p1": "#2b6cb0", "(b) Y += c xk^2": "#d97706",
            "(c) Y += c(xk^2 + d yk^2)": "#2f855a", "(d) BC p1 + p2": "#9b2c2c"}
    fig, axs = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    panels = [("q_n", 2, 1.0, 0.0), ("s_x - 1 (ppm)", 0, 1e6, 1.0), ("s_y - 1 (ppm)", 1, 1e6, 1.0)]
    for ax, (lab, k, sc, off) in zip(axs[0], panels):
        for v in res:
            ax.plot(ade, (res[v]["P"][:, k] - off) * sc, color=cols[v], lw=1.6, label=v)
        ax.set_title(lab, loc="left", color=ink, fontsize=11)
    ax = axs[1, 0]
    for v in res:
        names = res and None
    for v, names in [("(a) BC p1", ["p1"]), ("(b) Y += c xk^2", ["c"]), ("(c) Y += c(xk^2 + d yk^2)", ["c"]),
                     ("(d) BC p1 + p2", ["p1", "p2"])]:
        for k, n in enumerate(names):
            ax.plot(ade, res[v]["P"][:, 3 + k], color=cols[v], lw=1.6, ls="-" if k == 0 else ":", label=f"{v}: {n}")
    dpar = [d for d in polys if d["var"] == chosen and d["spec"]["parity"]][0]
    sx, sy, q, ex = poly_eval(dpar["spec"], dpar["p"], t)
    ax.plot(ade, ex[0], color=ink, lw=1, ls="--", label=f"{chosen} odd poly (first extra)")
    ax.set_title("extra coefficient (dimensionless)", loc="left", color=ink, fontsize=11)
    ax.legend(frameon=False, fontsize=7, labelcolor=ink)
    ax = axs[1, 1]
    v = "(c) Y += c(xk^2 + d yk^2)"
    nz = ~np.isclose(ade, 0, atol=1e-9)
    ax.plot(ade[nz], res[v]["P"][nz, 4], color=cols[v], lw=1.6)
    ax.set_title("(c): d per state", loc="left", color=ink, fontsize=11)
    ax = axs[1, 2]
    for v in res:
        ax.semilogy(ade, per_state_metrics(res[v]["err"])[0], color=cols[v], lw=1.6, label=v)
    ax.set_title("per-state ceiling vertex RMS (um)", loc="left", color=ink, fontsize=11)
    axs[0, 0].legend(frameon=False, fontsize=8, labelcolor=ink)
    for ax in axs.ravel():
        ax.set_xlabel("ADE (deg)", color=muted)
        ax.grid(color=grid, lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(muted)
        ax.tick_params(colors=muted)
    fig.suptitle("P4, A = 0 D, BDE = 0: K + decentering variants, per-state parameters vs ADE",
                 color=ink, fontsize=12, x=0.01, ha="left")
    fig.savefig(OUT / "p4_ade_a0_decenter_params.png", dpi=150)
    plt.close(fig)


def load_biaxial(path=GRID_PATH):
    """Load all A=0 ADE x BDE states, centered on each measured field center."""
    import pickle
    path = Path(path)
    raw = path.read_bytes()
    d = pickle.loads(raw)
    m = np.isclose(d["accommodation_d"], 0, atol=1e-9, rtol=0)
    s = d[m]
    ade = np.unique(s["eye_rotation_deg"])
    bde = np.unique(s["eye_rotation_x_deg"])
    na, nb = len(ade), len(bde)
    if len(s) != na * nb * 9 or not np.allclose(ade, np.linspace(-20, 20, na), atol=1e-9, rtol=0) \
            or not np.allclose(bde, np.linspace(-20, 20, nb), atol=1e-9, rtol=0):
        raise ValueError(f"unexpected A=0 slice: {len(s)} rows, axes {len(ade)} x {len(bde)}")
    order = np.lexsort((s["field_y_relative"], s["field_x_relative"],
                        s["eye_rotation_deg"], s["eye_rotation_x_deg"]))
    s = s[order]
    fields = np.stack((s["field_x_relative"], s["field_y_relative"]), -1).reshape(nb, na, 9, 2)[0, 0]
    expected = np.array([[x, y] for x in (-1., 0., 1.) for y in (-1., 0., 1.)])
    if not np.array_equal(fields, expected):
        raise ValueError("unexpected A=0 biaxial field order")
    z = np.stack((s["real_x_mm"], s["real_y_mm"]), -1).reshape(nb, na, 9, 2)
    # Structured ordering is BDE-major, ADE-minor, then x/y field, matching the polynomial mask.
    # Audit asymmetric labels against source records so axis transposition cannot pass by symmetry.
    for bi, ai, fi in ((int(round((nb - 1) * .63)), int(round((na - 1) * .17)), 6),
                       (int(round((nb - 1) * .25)), int(round((na - 1) * .76)), 2)):
        sx = s[(s["eye_rotation_x_deg"] == bde[bi]) & (s["eye_rotation_deg"] == ade[ai])
              & (s["field_x_relative"] == fields[fi, 0]) & (s["field_y_relative"] == fields[fi, 1])]
        source_xy = np.array([sx["real_x_mm"][0], sx["real_y_mm"][0]])
        if len(sx) != 1 or not np.array_equal(z[bi, ai, fi], source_xy):
            raise ValueError(f"BDE/ADE/field label alignment failed at {(bde[bi], ade[ai], fields[fi])}")
    return ade, bde, fields, z, hashlib.sha256(raw).hexdigest()


def bi_forward(b, p, variant):
    """Normalized center-preserving H, then optional Brown tangential and r^2 radial gain."""
    m11, m12, m21, m22, qx, qy = [p[..., i] for i in range(6)]
    x, y = b[..., 0], b[..., 1]
    den = 1 + qx * x + qy * y
    u = np.stack(((m11 * x + m12 * y) / den, (m21 * x + m22 * y) / den), -1)
    if variant != "projective":
        p1, p2 = p[..., 6], p[..., 7]
        ux, uy = u[..., 0], u[..., 1]
        X = ux + 2 * p1 * ux * uy + p2 * (3 * ux * ux + uy * uy)
        Y = uy + p1 * (ux * ux + 3 * uy * uy) + 2 * p2 * ux * uy
        u = np.stack((X, Y), -1)
    if variant == "BC + radial r2":
        g = p[..., 8]
        r2 = u[..., 0] ** 2 + u[..., 1] ** 2
        u = u * (1 + g * r2)[..., None]
    return u, den


def bi_fit_states(b, target, variant, max_iter=120, initial=None):
    """Batched damped LM for independent 8-point fits; center field is excluded."""
    names = {"projective": ("m11", "m12", "m21", "m22", "qx", "qy"),
             "projective + BC": ("m11", "m12", "m21", "m22", "qx", "qy", "p1", "p2"),
             "BC + radial r2": ("m11", "m12", "m21", "m22", "qx", "qy", "p1", "p2", "g_r2")}
    nstate, nfield = target.shape[:2]
    npar = len(names[variant])
    P = np.zeros((nstate, npar)); P[:, 0] = P[:, 3] = 1.
    if initial is not None:
        P[:] = initial
    eye = np.eye(npar)
    lam = np.full(nstate, 1e-3)
    done = np.zeros(nstate, bool)
    status = np.zeros(nstate, np.int8)  # 0 cap, 1 converged, 2 no accepted descent
    iters = np.zeros(nstate, np.int16)

    def resid(pp, idx):
        pred, den = bi_forward(b[idx], pp[:, None, :], variant)
        return ((pred - target[idx]) * L_MM * UM).reshape(len(idx), -1), den

    idxall = np.arange(nstate)
    r, den = resid(P, idxall)
    cost = np.sum(r * r, axis=1)
    if np.any(np.min(den, axis=1) <= .05):
        raise RuntimeError(f"{variant}: identity start violates projective denominator guard")
    for it in range(max_iter):
        idx = np.flatnonzero(~done)
        if not idx.size:
            break
        Pi = P[idx]
        ri, _ = resid(Pi, idx)
        J = np.empty((len(idx), ri.shape[1], npar))
        for j in range(npar):
            Pc = Pi.astype(complex); Pc[:, j] += 1e-30j
            J[:, :, j] = resid(Pc, idx)[0].imag / 1e-30
        JTJ = np.einsum("sri,srj->sij", J, J)
        grad = np.einsum("sri,sr->si", J, ri)
        scale = np.sqrt(np.maximum(np.einsum("sii->si", JTJ), 1e-30))
        scaled_grad = np.max(np.abs(grad / scale), axis=1)
        conv = scaled_grad < 1e-7
        done[idx[conv]] = True; status[idx[conv]] = 1
        iters[idx[conv]] = it + 1
        active = np.flatnonzero(~conv)
        if not len(active):
            continue
        ia = idx[active]; Ja = J[active]; ra = ri[active]; Pa = Pi[active]
        sa = scale[active]; H = JTJ[active] / sa[:, :, None] / sa[:, None, :]
        gg = grad[active] / sa
        accepted = np.zeros(len(active), bool)
        for _ in range(12):
            todo = np.flatnonzero(~accepted & (lam[ia] < 1e12))
            if not len(todo):
                break
            step_scaled = -np.linalg.solve(H[todo] + lam[ia[todo], None, None] * eye,
                                           gg[todo, :, None])[..., 0]
            step = step_scaled / sa[todo]
            cand = Pa[todo] + step
            rc, dc = resid(cand, ia[todo])
            cc = np.sum(rc * rc, axis=1)
            ok = np.isfinite(cc) & (cc < cost[ia[todo]]) & (np.min(dc.real, axis=1) > .05)
            if np.any(ok):
                ids = ia[todo[ok]]
                P[ids] = cand[ok]; cost[ids] = cc[ok]
                lam[ids] = np.maximum(lam[ids] / 3, 1e-12)
                accepted[todo[ok]] = True
            lam[ia[todo[~ok]]] *= 10
        stuck = ~accepted
        if np.any(stuck):
            ids = ia[stuck]; done[ids] = True; status[ids] = 2; iters[ids] = it + 1
    # At cap, preserve status 0. Recompute residuals over all states.
    pred, den = bi_forward(b, P[:, None, :], variant)
    err = (pred - target) * L_MM * UM
    return P, err, status, iters, np.min(den.real, axis=1)


def bi_poly_basis(tx, ty):
    powers = [(i, j) for total in range(5) for i in range(total + 1) for j in [total - i]]
    return np.stack([tx ** i * ty ** j for i, j in powers], axis=-1), powers


def bi_metrics(err, mask, fields, include_center=False):
    # err: state x field x xy, in um; metrics are over states and selected fields.
    ix = np.arange(9) if include_center else np.array([i for i in range(9) if i != 4])
    e = err[mask][:, ix]
    d = np.linalg.norm(e, axis=-1)
    out = {"xrmse": float(np.sqrt(np.mean(e[..., 0] ** 2))),
           "yrmse": float(np.sqrt(np.mean(e[..., 1] ** 2))),
           "vrms": float(np.sqrt(np.mean(d ** 2))),
           "max": float(np.max(d))}
    return out


def bi_stationarity(b, target, P, variant):
    """Dimensionless max component of J.T r, normalized by column/r norms."""
    nstate, npar = P.shape
    J = np.empty((nstate, target.shape[1] * 2, npar))
    for j in range(npar):
        Pc = P.astype(complex); Pc[:, j] += 1e-30j
        pred, _ = bi_forward(b, Pc[:, None, :], variant)
        J[:, :, j] = ((pred - target) * L_MM * UM).reshape(nstate, -1).imag / 1e-30
    pred, _ = bi_forward(b, P[:, None, :], variant)
    r = ((pred - target) * L_MM * UM).reshape(nstate, -1)
    grad = np.einsum("sri,sr->si", J, r)
    jn = np.sqrt(np.sum(J * J, axis=1))
    rn = np.maximum(np.linalg.norm(r, axis=1), 1.0)
    return np.max(np.abs(grad) / np.maximum(jn * rn[:, None], 1e-30), axis=1)


def main_biaxial(input_path=GRID_PATH, output_dir=OUT, coarse_model_path=None, batch_size=8192):
    global L_MM, OUT
    input_path, output_dir = Path(input_path), Path(output_dir)
    coarse_model_path = Path(coarse_model_path) if coarse_model_path else OUT / "p4_ade_a0_biaxial.npz"
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    OUT = output_dir
    coef = np.load(COEF_PATH)
    L_MM = float(coef["L_mm"])
    ade, bde, fields, z, sha = load_biaxial(input_path)
    na, nb = len(ade), len(bde)
    # Arrange states as (BDE, ADE, field, xy), matching tx/ty surface convention.
    B, A = np.meshgrid(bde / 20, ade / 20, indexing="ij")
    idxc = int(np.flatnonzero(np.all(fields == (0, 0), axis=1))[0])
    ia0 = int(np.flatnonzero(np.isclose(ade, 0, atol=1e-9, rtol=0))[0])
    ib0 = int(np.flatnonzero(np.isclose(bde, 0, atol=1e-9, rtol=0))[0])
    c00 = z[ib0, ia0, idxc].copy()
    ref = z[ib0, ia0]
    b = (ref - c00) / L_MM
    centers = z[:, :, idxc]
    target = (z - centers[:, :, None, :]) / L_MM
    valid_state = np.all(np.isfinite(z), axis=(2, 3))
    if not valid_state[ib0, ia0]:
        raise ValueError("zero-gaze reference has nonfinite real coordinates")
    n_excluded = int(np.count_nonzero(~valid_state))
    if n_excluded:
        print(f"Excluding {n_excluded} states with incomplete real coordinates; probe/status failures do not affect this mask", flush=True)
    # states flat in BDE-major, ADE-minor order
    outer = np.array([i for i in range(9) if i != idxc])
    bstate_all = np.broadcast_to(b[None, outer, :], (nb * na, 8, 2))
    tarfit_all = target[:, :, outer, :].reshape(-1, 8, 2)
    tarall = target.reshape(-1, 9, 2)
    valid_flat = valid_state.ravel()
    train_even = ((np.arange(nb)[:, None] % 2 == 0) & (np.arange(na)[None, :] % 2 == 0))
    train2 = train_even.ravel() & valid_flat
    test2 = (~train_even).ravel() & valid_flat
    ntrain = int(train_even.sum())
    if int(train2.sum()) != ntrain or int(test2.sum()) != nb * na - ntrain - n_excluded:
        raise ValueError("unexpected even/even train/test split or valid-state count")
    basis, powers = bi_poly_basis(B.ravel(), A.ravel())
    variants = ["projective", "projective + BC", "BC + radial r2"]
    fits, residuals, coeffs, polyres, stationarity, scipy_checks, poly_min_den = {}, {}, {}, {}, {}, {}, {}
    incumbent = None
    for variant in variants:
        t0 = time.time()
        npar = {"projective":6,"projective + BC":8,"BC + radial r2":9}[variant]
        P = np.full((nb * na, npar), np.nan)
        status = np.full(nb * na, -1, dtype=np.int8)
        iters = np.full(nb * na, -1, dtype=np.int16)
        min_den = np.full(nb * na, np.nan)
        grad = np.full(nb * na, np.nan)
        err = np.full((nb * na, 9, 2), np.nan)
        err[valid_flat, idxc, :] = 0.0
        valid_indices = np.flatnonzero(valid_flat)
        for start in range(0, len(valid_indices), batch_size):
            ids = valid_indices[start:start + batch_size]
            initial = None if incumbent is None else np.pad(
                incumbent[ids], ((0, 0), (0, npar - incumbent.shape[1])))
            Pb, eb, sb, ib, db = bi_fit_states(bstate_all[ids], tarfit_all[ids], variant, initial=initial)
            P[ids], status[ids], iters[ids], min_den[ids] = Pb, sb, ib, db
            err[ids[:, None], outer[None, :], :] = eb
            grad[ids] = bi_stationarity(bstate_all[ids], tarfit_all[ids], Pb, variant)
        stationarity[variant] = grad
        # Independent nonlinear refits of the three stuck states with largest normalized gradient.
        checks = []
        stuck = np.flatnonzero(status == 2)
        if len(stuck):
            sample = stuck[np.argsort(stationarity[variant][stuck])[-3:]]
            for si in sample:
                f = lambda pp: ((bi_forward(bstate_all[si], pp[None, :], variant)[0] - tarfit_all[si]) * L_MM * UM).ravel()
                rr = least_squares(f, P[si], method="lm", max_nfev=10000, xtol=1e-14, ftol=1e-14, gtol=1e-14)
                c0 = float(np.sum(f(P[si])**2)); c1 = float(np.sum(f(rr.x)**2))
                checks.append((int(si), float(stationarity[variant][si]), c0, c1, int(rr.nfev)))
        scipy_checks[variant] = checks
        incumbent = P
        fits[variant] = (P, status, iters, min_den)
        residuals[variant] = err
        # Train coefficient surfaces on even/even state parameters; evaluate whole plane.
        cf = np.linalg.lstsq(basis[train2], P[train2], rcond=None)[0]
        coeffs[variant] = cf
        pp = basis @ cf
        ball = np.broadcast_to(b[None, :, :], (len(P),9,2))
        pred, den = bi_forward(ball, pp[:, None, :], variant)
        poly_min_den[variant] = float(np.min(den.real))
        polyres[variant] = (pred - tarall) * L_MM * UM
        polyres[variant][~valid_flat] = np.nan
        print(f"{variant}: {time.time()-t0:.1f}s; status counts {np.unique(status[valid_flat], return_counts=True)}; min denom={np.nanmin(min_den):.4g}", flush=True)

    # Evaluate the coarse-grid coefficient surfaces on these same fine-grid states.
    if not coarse_model_path.is_file():
        raise FileNotFoundError(f"coarse coefficient NPZ not found: {coarse_model_path}")
    coarse = np.load(coarse_model_path, allow_pickle=False)
    if not np.array_equal(coarse["polynomial_powers"], np.asarray(powers, dtype=int)):
        raise ValueError("coarse coefficient polynomial basis does not match this fitter")
    if not np.isclose(float(coarse["L_mm"]), L_MM, atol=1e-12, rtol=0):
        raise ValueError("coarse coefficient L does not match this fit")
    if not np.array_equal(coarse["center_reference_mm"], c00):
        raise ValueError("fine zero-gaze center does not exactly match coarse model reference")
    if not np.array_equal(coarse["reference_mm"], ref):
        raise ValueError("fine zero-gaze field pattern does not exactly match coarse model reference")
    coarse_sha = hashlib.sha256(coarse_model_path.read_bytes()).hexdigest()
    baseline_residuals, baseline_min_den = {}, {}
    ball = np.broadcast_to(b[None, :, :], (nb * na, 9, 2))
    for variant in variants:
        cf = np.asarray(coarse[f"coefficients_{variant}"], dtype=float)
        if cf.shape != coeffs[variant].shape:
            raise ValueError(f"coarse coefficient shape mismatch for {variant}: {cf.shape}")
        pred, den = bi_forward(ball, (basis @ cf)[:, None, :], variant)
        baseline_residuals[variant] = (pred - tarall) * L_MM * UM
        baseline_residuals[variant][~valid_flat] = np.nan
        baseline_min_den[variant] = float(np.nanmin(den.real))

    # Reporting, coefficients, metric grids and plots.
    OUT.mkdir(parents=True, exist_ok=True)
    report = ["# P4 A=0 biaxial gaze-induced distortion fit", "",
              "This report supersedes only the earlier A=0, BDE=0 ADE-only analysis; that report remains at `report.md`.",
              f"Data: {int(valid_flat.sum()):,} valid A=0 states ({na} ADE x {nb} BDE nominal states) x 9 fields; gaze spacing {ade[1]-ade[0]:g} deg x {bde[1]-bde[0]:g} deg. Exactly {n_excluded} whole states with incomplete real-coordinate observations are excluded from fits and metrics. The 17 probe/paraxial-only failed states with finite real coordinates are retained; failure sidecar flags do not define this mask. Reference pattern is measured at ADE=BDE=0; `c00` is its measured center. For valid gaze states, `b_j=(z_00,j-c00)/L`, target `(z_gj-c_g)/L`, and reconstruction `zhat_gj=c_g+L*f_g(b_j)`. Fits describe centered distortion with measured `c_g` supplied; they do not predict absolute gaze-dependent center motion.",
              f"State tensors and flat rows use BDE-major, ADE-minor order: `state_k = i_BDE*{na} + i_ADE`; field index is innermost. The loader checks two asymmetric states against source records. Source pickle `{input_path}` SHA-256 `{sha}`.",
              f"`L={L_MM:.12f} mm`; measured center displacement relative to c00: x range {((centers-c00)[:,:,0].min()*1e3):+.4f} to {((centers-c00)[:,:,0].max()*1e3):+.4f} um; y range {((centers-c00)[:,:,1].min()*1e3):+.4f} to {((centers-c00)[:,:,1].max()*1e3):+.4f} um.",
              "All primary metrics use the eight noncentral fields; comparable nine-field values are also given. Center excluded from fits. Vertex RMS is sqrt(mean(ex²+ey²)); xRMSE/yRMSE are coordinate RMSE.", "",
              "## Model families", "",
              "Model order is H then Brown then optional radial: H(b)=M b/(1+qx*bx+qy*by), M=[[m11,m12],[m21,m22]]. Let u=H(b); BC gives w_x=u_x+2p1*u_x*u_y+p2*(3u_x^2+u_y^2), w_y=u_y+p1*(u_x^2+3u_y^2)+2p2*u_x*u_y. The third family is f=(1+g_r2*||w||^2)*w. Parameter counts are 6, 8, and 9. The fitted radial gain is position dependent and not a separate physical attribution.",
              f"Polynomial surfaces use all 15 total-degree<=4 monomials in tx=BDE/20, ty=ADE/20, unconstrained. Even/even indices ({int(train2.sum()):,} valid states) train; remaining {int(test2.sum()):,} valid states test; incomplete-real states are excluded. This is held-out gaze interpolation within +/-20 deg on the same nine sampled fields; it does not validate unseen field positions or extrapolation beyond the sampled range. Independent per-state fits are ceilings, not deployable gaze surfaces. Coarse baseline coefficients `{coarse_model_path}` SHA-256 `{coarse_sha}` are evaluated on the same fine heldout states.", "",
              "## Performance"]
    report.append("| family | split | field count | xRMSE | yRMSE | vertex RMS | max vertex |")
    report.append("|---|---|---:|---:|---:|---:|---:|")
    for v in variants:
        for kind, e in [("per-state ceiling", residuals[v]), ("polynomial", polyres[v])]:
            for split, mask in [("train", train2), ("test", test2), ("all", valid_flat)]:
                for inc, nfield in [(False, 8), (True, 9)]:
                    m = bi_metrics(e, mask, fields, inc)
                    report.append(f"| {v} {kind} | {split} | {nfield} | {m['xrmse']:.4f} | {m['yrmse']:.4f} | {m['vrms']:.4f} | {m['max']:.4f} |")
    report += ["", "## Saved coarse coefficients evaluated on fine heldout states", "",
               "All entries below use the same valid fine test states and target as the refit; these are baseline-vs-refit comparisons, not coarse-grid test metrics.", "",
               "| family | source coefficients | fields | xRMSE | yRMSE | vertex RMS | max vertex |", "|---|---|---:|---:|---:|---:|---:|"]
    for v in variants:
        for label, e in [("saved coarse fit", baseline_residuals[v]), ("fine refit", polyres[v])]:
            for inc, nfield in [(False, 8), (True, 9)]:
                mm = bi_metrics(e, test2, fields, inc)
                report.append(f"| {v} | {label} | {nfield} | {mm['xrmse']:.4f} | {mm['yrmse']:.4f} | {mm['vrms']:.4f} | {mm['max']:.4f} |")

    report += ["", "## Per-field-class heldout test metrics (eight noncenter fields)", "",
               "| family | class | n fields | xRMSE | yRMSE | vertex RMS | max vertex |", "|---|---|---:|---:|---:|---:|---:|"]
    classes = {"corners": [0,2,6,8], "x edges": [1,7], "y edges": [3,5]}
    for v in variants:
        for cls, ids in classes.items():
            e = polyres[v][test2][:, ids]
            d = np.linalg.norm(e, axis=-1)
            report.append(f"| {v} | {cls} | {len(ids)} | {np.sqrt(np.mean(e[...,0]**2)):.4f} | {np.sqrt(np.mean(e[...,1]**2)):.4f} | {np.sqrt(np.mean(d**2)):.4f} | {d.max():.4f} |")
    report += ["", "## Zero gaze and convergence", "", "At zero gaze the fixed-reference induced distortion is zero by construction. The polynomial fit has a free intercept; its predicted zero-gaze residual is reported below. The measured center displacement is zero within the stored grid precision, so no center-motion surface is needed for absolute coordinates. Denominators were checked over all fields and gaze states.", "", "| family | converged | capped | stuck/no descent | max iterations | min denominator per-state / fine polynomial / coarse polynomial | max normalized gradient among stuck | poly zero-gaze max field residual (µm) |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    zeroidx = ib0 * na + ia0
    for v in variants:
        P, st, it, den = fits[v]
        report.append(f"| {v} | {int(np.sum(st==1))} | {int(np.sum(st==0))} | {int(np.sum(st==2))} | {int(it.max())} | {np.nanmin(fits[v][3]):.4f} / {poly_min_den[v]:.4f} / {baseline_min_den[v]:.4f} | {np.max(stationarity[v][st==2]) if np.any(st==2) else 0:.3g} | {np.linalg.norm(polyres[v][zeroidx],axis=1).max():.6g} |")
    report += ["", "For the three highest-gradient stuck states per family, independent SciPy LM was started from the batched result. Columns show flat state index, normalized gradient, starting SSE, refitted SSE (µm²), and SciPy nfev.", "", "| family | state | norm grad | start SSE | refit SSE | nfev |", "|---|---:|---:|---:|---:|---:|"]
    for v in variants:
        for si, gr, c0, c1, nf in scipy_checks[v]:
            report.append(f"| {v} | {si} | {gr:.3g} | {c0:.6g} | {c1:.6g} | {nf} |")
    report += ["", "Nesting and ceiling checks (tolerance 1e-7 µm² per state):", ""]
    for va, vb in zip(variants[1:], variants[:-1]):
        ca = np.nansum(residuals[va]**2, axis=(1,2)); cb = np.nansum(residuals[vb]**2, axis=(1,2))
        report.append(f"- {va} worse than {vb}: {int(np.sum((ca > cb + 1e-7) & valid_flat))} states.")
    for v in variants:
        ca = np.nansum(residuals[v]**2, axis=(1,2)); cb = np.nansum(polyres[v]**2, axis=(1,2))
        report.append(f"- {v} per-state residual exceeds its polynomial fit: {int(np.sum((ca > cb + 1e-7) & valid_flat))} states.")
    report += ["", "Coverage is evaluated over heldout states; each state is counted if all eight noncentral field-point errors meet the threshold.", "", "| family | heldout states <1 µm vertex RMS | heldout states max point <1 µm |", "|---|---:|---:|"]
    for v in variants:
        e = polyres[v][test2]
        d = np.linalg.norm(e[:, [i for i in range(9) if i != 4]], axis=-1)
        vr = np.sqrt(np.mean(d*d, axis=1)); mx = d.max(axis=1)
        report.append(f"| {v} | {np.sum(vr<1)}/{len(vr)} | {np.sum(mx<1)}/{len(mx)} |")
    report += ["", "## Coefficient parity diagnostics", "", "No parity constraints were imposed. The table separates maximum coefficient magnitudes on odd-total-degree versus even-total-degree monomials in the unconstrained polynomial basis.", "", "| family | parameter | max odd-degree coefficient | max even-degree coefficient |", "|---|---|---:|---:|"]
    for v in variants:
        pnames = {"projective": ("m11","m12","m21","m22","qx","qy"),
                  "projective + BC": ("m11","m12","m21","m22","qx","qy","p1","p2"),
                  "BC + radial r2": ("m11","m12","m21","m22","qx","qy","p1","p2","g_r2")}[v]
        odd = np.array([(i+j)%2 == 1 for i,j in powers]); even = ~odd
        for k, name in enumerate(pnames):
            report.append(f"| {v} | {name} | {np.max(np.abs(coeffs[v][odd,k])):.4e} | {np.max(np.abs(coeffs[v][even,k])):.4e} |")
    report += ["", "## Artifacts", "", "- `p4_ade_a0_biaxial.npz`: measured centers, per-state parameters and residuals, polynomial coefficients/predictions, split masks, axes, fit statuses.", "- `p4_ade_a0_biaxial_heatmaps.png`: heldout xRMSE, yRMSE, and vertex RMS grids for all three families.", "- `p4_ade_a0_biaxial_residuals.png`: representative corner and axis residuals at the gaze corners.", ""]
    (OUT / "biaxial_report.md").write_text("\n".join(report), encoding="utf-8")
    np.savez_compressed(OUT / "p4_ade_a0_biaxial.npz", ADE=ade, BDE=bde, state_ADE=A.ravel(), state_BDE=B.ravel(), fields=fields, L_mm=L_MM,
                        source_pickle=np.array(str(input_path)), source_sha256=np.array(sha), coarse_coefficients=np.array(str(coarse_model_path)), coarse_sha256=np.array(coarse_sha), real_mm=z, valid_state=valid_state, valid_state_flat=valid_flat, excluded_state_mask=~valid_state,
                        centers_mm=centers, center_reference_mm=c00, reference_mm=ref, train_mask=train2,
                        test_mask=test2, polynomial_powers=np.array(powers), **{f"params_{v}":fits[v][0] for v in variants},
                        **{f"status_{v}":fits[v][1] for v in variants}, **{f"iterations_{v}":fits[v][2] for v in variants},
                        **{f"coefficients_{v}":coeffs[v] for v in variants},
                        **{f"per_state_residual_um_{v}":residuals[v] for v in variants},
                        **{f"polynomial_residual_um_{v}":polyres[v] for v in variants},
                        **{f"coarse_baseline_residual_um_{v}":baseline_residuals[v] for v in variants},
                        **{f"stationarity_{v}":stationarity[v] for v in variants},
                        **{f"polynomial_min_den_{v}":np.array(poly_min_den[v]) for v in variants})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 3, figsize=(13, 11), constrained_layout=True)
    fields_no_center = [i for i in range(9) if i != 4]
    for j, v in enumerate(variants):
        e = polyres[v].reshape(nb, na, 9, 2)
        for r, (label, val) in enumerate([("xRMSE", np.sqrt(np.mean(e[...,fields_no_center,0]**2,axis=2))),
                                           ("yRMSE", np.sqrt(np.mean(e[...,fields_no_center,1]**2,axis=2))),
                                           ("vertex RMS", np.sqrt(np.mean(np.sum(e[...,fields_no_center,:]**2,axis=-1),axis=2))) ]):
            val = np.where(test2.reshape(nb,na), val, np.nan)
            im=axs[r,j].imshow(val, origin="lower", extent=[-20,20,-20,20], aspect="auto", vmin=0, vmax=8, cmap="magma")
            axs[r,j].set_title(f"{v}: {label} (test µm)")
            axs[r,j].set_xlabel("ADE (deg)"); axs[r,j].set_ylabel("BDE (deg)"); fig.colorbar(im,ax=axs[r,j],shrink=.8)
    fig.savefig(OUT / "p4_ade_a0_biaxial_heatmaps.png", dpi=160); plt.close(fig)
    fig, axs = plt.subplots(3, 3, figsize=(13, 11), constrained_layout=True)
    for j, v in enumerate(variants):
        e = polyres[v].reshape(nb,na,9,2)
        for r,(ia,ib) in enumerate([(0,0),(0,nb-1),(na-1,0)]):
            ax=axs[r,j]; ex=e[ib,ia]
            # Residual vectors are in micrometres while the axes are normalized field positions.
            # Scale arrows for legibility; the adjacent text retains the actual signed values.
            ax.quiver(fields[:,0],fields[:,1],ex[:,0]*0.015,ex[:,1]*0.015,angles="xy",scale_units="xy",scale=1,color="#2b6cb0")
            ax.set_title(f"{v}; ADE={ade[ia]:g}, BDE={bde[ib]:g}°")
            ax.set_xlim(-1.3,1.3); ax.set_ylim(-1.3,1.3); ax.set_aspect("equal"); ax.grid(True,alpha=.25)
            for k,(x,y) in enumerate(fields): ax.text(x,y,f"{ex[k,0]:+.1f},{ex[k,1]:+.1f} µm",fontsize=6)
    fig.savefig(OUT / "p4_ade_a0_biaxial_residuals.png", dpi=160); plt.close(fig)
    print(f"Wrote {OUT / 'biaxial_report.md'} and NPZ/figures")


if __name__ == "__main__":
    if "--biaxial" in sys.argv:
        parser = argparse.ArgumentParser(description="Fit A=0 biaxial P4 gaze distortion")
        parser.add_argument("--biaxial", action="store_true")
        parser.add_argument("--input", type=Path, default=GRID_PATH)
        parser.add_argument("--output-dir", type=Path, default=OUT)
        parser.add_argument("--coarse-model", type=Path, default=OUT / "p4_ade_a0_biaxial.npz")
        parser.add_argument("--batch-size", type=int, default=8192)
        args = parser.parse_args()
        main_biaxial(args.input, args.output_dir, args.coarse_model, args.batch_size)
    else:
        main()
