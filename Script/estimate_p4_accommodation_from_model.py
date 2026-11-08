"""Invert the fitted P4 biaxial joint model to estimate accommodation A from the 3x3 grid (known gaze).

For each simulated state (A_true, ADE, BDE) the observation is the 9 center-relative real-ray coordinates
(18 numbers, mm).  Gaze (ADE, BDE) is assumed known.  A_hat in [0, 4] D minimizes the sum of squared
forward-model residuals over the selected fields:

    coarse scan on 0..4 D step 0.01 D (vectorized), then vectorized golden-section refinement on the
    bracket [A_i - 0.01, A_i + 0.01] clipped to [0, 4] (60 iterations -> bracket < 1e-14 D).

Forward model = Script/fit_p4_biaxial_joint.py, made continuous in A:
    k1(A) = c0 + c1*A^n   (stored power law),   s(A) = cubic spline (not-a-knot) through the measured
    per-accommodation paraxial scale values (the values used in training), r0 fixed.
At the 21 grid accommodations the continuous model therefore equals the trained model exactly.

Models: offset model (main_even, 336 coef) and center-exact model (control_no_T, T=0, 252 coef), both
trained on even-ADE/even-BDE states; plus the leave-out-1.0/3.0-D refit (offset model only; the T=0
leave-out fit was never run/stored).  Read-only use of existing artifacts; writes only
data/p4_biaxial_joint/accommodation_estimation.md and accommodation_estimation_results.npz.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "6"

import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Script"))
import fit_p4_biaxial_joint as fj  # noqa: E402
import refit_p4_separable_transform as sep  # noqa: E402

OUT = ROOT / "data" / "p4_biaxial_joint"
COEF_NPZ = OUT / "p4_biaxial_joint_coefficients.npz"
FIT_PKL = OUT / "p4_biaxial_joint.pkl"
REPORT = OUT / "accommodation_estimation.md"
RESULTS = OUT / "accommodation_estimation_results.npz"

A_GRID = np.linspace(0.0, 4.0, 401)
SIGMA_MM = 1e-3                       # 1 um per coordinate
CHUNK = 1000
FIELD_SETS = {"all9": list(range(9)), "corners4": [0, 2, 6, 8], "field(-1,-1)": [0]}
GOLD = (np.sqrt(5.0) - 1.0) / 2.0


def log(m):
    print(m, flush=True)


# --------------------------------------------------------------------------- continuous model
class Barrel:
    """b(A) = B_A(r0)/L with k1(A) power law and s(A) cubic spline; returns value and dA derivative."""

    def __init__(self, ref, kcoef, acc_nodes, s_nodes, L):
        self.c0, self.c1, self.n = (float(v) for v in kcoef)
        self.p0, _ = sep.invert_radial(ref, sep.radial_k1(kcoef, 0.0))      # (9, 2)
        self.r2 = np.sum(self.p0 ** 2, axis=-1)                              # (9,)
        self.spl = CubicSpline(acc_nodes, s_nodes)
        self.dspl = self.spl.derivative()
        self.L = L

    def __call__(self, A):
        A = np.asarray(A, float)
        s, ds = self.spl(A)[..., None], self.dspl(A)[..., None]
        k = (self.c0 + self.c1 * np.power(A, self.n))[..., None]
        dk = (self.c1 * self.n * np.power(A, self.n - 1.0))[..., None]
        g = 1.0 + k * s * s * self.r2
        dg = dk * s * s * self.r2 + 2.0 * k * s * ds * self.r2
        b = (s * g)[..., None] * self.p0 / self.L
        db = (ds * g + s * dg)[..., None] * self.p0 / self.L
        return b, db                                                          # A.shape + (9, 2)


def state_poly(C, free, tx, ty):
    """Per-state coefficients of each family as a quadratic in a=A/5: dict f -> (S, 3)."""
    mono = np.stack([tx ** i * ty ** j for (i, j) in fj.MONOS], axis=-1)    # (S, 14)
    out = {}
    for f in fj.FAMILIES:
        if f in free:
            c = C[list(free).index(f)].reshape(fj.NL, fj.NMONO)
            out[f] = mono @ c.T
        else:
            out[f] = None
    return out


def forward(A, gp, barrel, sl=slice(None)):
    """A broadcastable to (S, K). Returns pred, dpred/dA with shape (S, K, 9, 2) (mm, mm/D)."""
    b, db = barrel(A)
    a = np.asarray(A, float) / fj.A_NORM
    fv, dfv = {}, {}
    for f in fj.FAMILIES:
        base = fj.BASE.get(f, 0.0)
        g = gp[f]
        if g is None:
            fv[f], dfv[f] = base, 0.0
            continue
        g = g[sl]
        g0, g1, g2 = g[:, 0, None], g[:, 1, None], g[:, 2, None]
        fv[f] = (base + g0 + g1 * a + g2 * a * a)[..., None]
        dfv[f] = ((g1 + 2.0 * g2 * a) / fj.A_NORM)[..., None]
    xb, yb, dxb, dyb = b[..., 0], b[..., 1], db[..., 0], db[..., 1]
    L = barrel.L
    den = 1.0 + fv["qx"] * xb + fv["qy"] * yb
    dden = dfv["qx"] * xb + fv["qx"] * dxb + dfv["qy"] * yb + fv["qy"] * dyb
    nu = fv["m11"] * xb + fv["m12"] * yb
    nv = fv["m21"] * xb + fv["m22"] * yb
    dnu = dfv["m11"] * xb + fv["m11"] * dxb + dfv["m12"] * yb + fv["m12"] * dyb
    dnv = dfv["m21"] * xb + fv["m21"] * dxb + dfv["m22"] * yb + fv["m22"] * dyb
    X = L * nu / den + fv["Tx"]
    Y = L * nv / den + fv["Ty"]
    dX = L * (dnu * den - nu * dden) / den ** 2 + dfv["Tx"]
    dY = L * (dnv * den - nv * dden) / den ** 2 + dfv["Ty"]
    shp = np.broadcast_shapes(np.shape(X), np.shape(Y))
    return (np.stack(np.broadcast_arrays(X, Y), -1).reshape(shp + (2,)),
            np.stack(np.broadcast_arrays(dX, dY), -1).reshape(shp + (2,)))


def cost(A, gp, barrel, obs, fields, sl):
    pred, _ = forward(A, gp, barrel, sl)
    r = pred[:, :, fields, :] - obs[:, None, fields, :]
    return np.sum(r * r, axis=(-2, -1))


def estimate(obs, gp, barrel, fields):
    """Coarse scan + golden section. obs (S, 9, 2). Returns A_hat, n_local_minima_in_scan."""
    S = len(obs)
    Ahat = np.empty(S)
    nloc = np.empty(S, int)
    for s0 in range(0, S, CHUNK):
        sl = slice(s0, min(S, s0 + CHUNK))
        o = obs[sl]
        c = cost(A_GRID[None, :], gp, barrel, o, fields, sl)
        i = np.argmin(c, axis=1)
        inner = (c[:, 1:-1] < c[:, :-2]) & (c[:, 1:-1] < c[:, 2:])
        nloc[sl] = inner.sum(1) + (c[:, 0] < c[:, 1]) + (c[:, -1] < c[:, -2])
        lo = A_GRID[np.maximum(i - 1, 0)][:, None]
        hi = A_GRID[np.minimum(i + 1, len(A_GRID) - 1)][:, None]
        x1 = hi - GOLD * (hi - lo)
        x2 = lo + GOLD * (hi - lo)
        f1 = cost(x1, gp, barrel, o, fields, sl)
        f2 = cost(x2, gp, barrel, o, fields, sl)
        for _ in range(60):
            left = f1 < f2
            hi = np.where(left, x2, hi)
            lo = np.where(left, lo, x1)
            x2n = np.where(left, x1, lo + GOLD * (hi - lo))
            x1n = np.where(left, hi - GOLD * (hi - lo), x2)
            f2n = np.where(left, f1, np.nan)
            f1n = np.where(left, np.nan, f2)
            # evaluate the one new point per state
            newx = np.where(left, x1n, x2n)
            fn = cost(newx, gp, barrel, o, fields, sl)
            x1, x2 = x1n, x2n
            f1 = np.where(left, fn, f1n)
            f2 = np.where(left, f2n, fn)
        Ahat[sl] = 0.5 * (lo + hi)[:, 0]
        # compare against bracket endpoints (minimum on the bound)
        cand = np.concatenate([lo, hi, 0.5 * (lo + hi)], axis=1)
        cc = np.stack([cost(cand[:, k:k + 1], gp, barrel, o, fields, sl)[:, 0] for k in range(3)], 1)
        Ahat[sl] = cand[np.arange(len(cand)), np.argmin(cc, 1)]
    return Ahat, nloc


def crb_and_linbias(Atrue, obs, gp, barrel, fields):
    S = len(obs)
    sig = np.empty(S)
    lb = np.empty(S)
    for s0 in range(0, S, CHUNK):
        sl = slice(s0, min(S, s0 + CHUNK))
        pred, dp = forward(Atrue[sl, None], gp, barrel, sl)
        g = dp[:, 0, fields, :].reshape(pred.shape[0], -1)
        r = (pred[:, 0, fields, :] - obs[sl][:, fields, :]).reshape(pred.shape[0], -1)
        gg = np.sum(g * g, 1)
        sig[sl] = SIGMA_MM / np.sqrt(gg)
        lb[sl] = -np.sum(g * r, 1) / gg
    return sig, lb


# --------------------------------------------------------------------------------- statistics
def summ(e):
    e = np.asarray(e, float)
    ae = np.abs(e)
    return {"n": len(e), "bias": e.mean(), "rms": np.sqrt(np.mean(e ** 2)), "std": e.std(),
            "medabs": np.median(ae), "p95": np.percentile(ae, 95), "p99": np.percentile(ae, 99),
            "max": ae.max(), "f05": np.mean(ae < 0.05), "f10": np.mean(ae < 0.1),
            "f25": np.mean(ae < 0.25), "f50": np.mean(ae < 0.5)}


SUMM_HDR = ("| case | states | bias (mD) | RMS (mD) | std (mD) | median abs (mD) | p95 abs (mD) | p99 abs (mD) | "
            "max abs (mD) | <0.05 D | <0.1 D | <0.25 D | <0.5 D |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")


def summ_row(name, e):
    s = summ(e)
    return (f"| {name} | {s['n']:,} | {s['bias']*1e3:.2f} | {s['rms']*1e3:.2f} | {s['std']*1e3:.2f} | "
            f"{s['medabs']*1e3:.2f} | {s['p95']*1e3:.2f} | {s['p99']*1e3:.2f} | {s['max']*1e3:.2f} | "
            f"{s['f05']*100:.2f}% | {s['f10']*100:.2f}% | {s['f25']*100:.2f}% | {s['f50']*100:.2f}% |")


BINS = [(-0.1, 0.1, "0"), (0.1, 4.05, "(0,4]"), (4.05, 8.05, "(4,8]"), (8.05, 12.05, "(8,12]"),
        (12.05, 16.05, "(12,16]"), (16.05, 20.1, "(16,20]")]


def binmask(v, lo, hi):
    return (v > lo) & (v <= hi)


# ------------------------------------------------------------------------------------- main
def main():
    T0 = time.perf_counter()
    log("Loading grid ...")
    G = fj.load_biaxial(fj.GRID_PATH)
    Z = np.load(COEF_NPZ)
    P = pickle.loads(FIT_PKL.read_bytes())
    L = float(Z["L_mm"])
    acc = G["acc"]
    na, nt, nb = len(acc), len(G["ade"]), len(G["bde"])
    ai, ti, bi = np.nonzero(np.ones((na, nt, nb), bool))
    A_true = acc[ai]
    ADE, BDE = G["ade"][ti], G["bde"][bi]
    tx, ty = BDE / fj.T_NORM, ADE / fj.T_NORM
    obs = G["real"][ai, ti, bi]                                    # (S, 9, 2)
    even = (ti % 2 == 0) & (bi % 2 == 0)
    held = ~even
    rayfail = G["nan_state"][ai, ti, bi]
    S = len(A_true)
    log(f"  {S} states, even/even {even.sum()}, held-out {held.sum()}, ray-failure {rayfail.sum()}")

    br = P["barrel_main"]
    s_meas = br["s_measured"]
    barrel = Barrel(br["reference_mm"], Z["barrel_power_c0_c1_n"], acc, s_meas, L)
    models = {}
    for key, lab in (("main_even", "offset (336 coef, T free)"), ("control_no_T", "center-exact (252 coef, T=0)")):
        free = tuple(str(f) for f in Z[f"{key}_families"])
        models[key] = {"label": lab, "free": free, "gp": state_poly(Z[f"{key}_coefficients"], free, tx, ty),
                       "barrel": barrel}

    # ---------------- checks
    checks = {}
    bgrid, _ = barrel(acc)
    checks["barrel_vs_trained_max_mm"] = float(np.max(np.abs(bgrid * L - br["barrel_mm"])))
    for key, M in models.items():
        e = np.empty((S, 9, 2))
        for s0 in range(0, S, 20000):
            sl = slice(s0, min(S, s0 + 20000))
            pr, _ = forward(A_true[sl, None], M["gp"], barrel, sl)
            e[sl] = pr[:, 0] - obs[sl]
        checks[f"{key}_complement_rmse_um"] = float(np.sqrt(np.mean(e[held] ** 2)) * 1e3)
        checks[f"{key}_train_rmse_um"] = float(np.sqrt(np.mean(e[even] ** 2)) * 1e3)
        M["resid"] = e
    # derivative check by finite differences on a random subset
    rng = np.random.default_rng(7)
    sub = rng.choice(S, 500, replace=False)
    sub.sort()
    worst = 0.0
    for key, M in models.items():
        gps = {f: (None if v is None else v[sub]) for f, v in M["gp"].items()}
        Ac = np.clip(A_true[sub], 0.01, 3.99)[:, None]
        _, d = forward(Ac, gps, barrel)
        h = 1e-6
        fd = (forward(Ac + h, gps, barrel)[0] - forward(Ac - h, gps, barrel)[0]) / (2 * h)
        worst = max(worst, float(np.max(np.abs(fd - d)) / np.max(np.abs(fd))))
    checks["dA_fd_rel_err"] = worst
    # s(A) spline leave-one-out interpolation error
    loo = []
    for k in range(1, na - 1):
        m = np.ones(na, bool)
        m[k] = False
        loo.append(CubicSpline(acc[m], s_meas[m])(acc[k]) - s_meas[k])
    loo = np.array(loo)
    dsdA = float(np.mean(np.abs(barrel.dspl(acc))))
    checks["s_loo_max"] = float(np.max(np.abs(loo)))
    checks["s_loo_rms"] = float(np.sqrt(np.mean(loo ** 2)))
    checks["dsdA_mean_abs"] = dsdA
    d1 = np.diff(s_meas)
    checks["s_first_diff_range"] = (float(d1.min()), float(d1.max()))
    log(f"  checks {checks}")

    # ---------------- inversions
    res = {}
    for key, M in models.items():
        for fs, fields in FIELD_SETS.items():
            t = time.perf_counter()
            Ah, nl = estimate(obs, M["gp"], barrel, fields)
            sig, lb = crb_and_linbias(A_true, obs, M["gp"], barrel, fields)
            res[(key, fs)] = {"Ahat": Ah, "err": Ah - A_true, "nloc": nl, "sig": sig, "linb": lb}
            log(f"  {key} {fs}: RMS err {np.sqrt(np.mean((Ah-A_true)**2))*1e3:.2f} mD, "
                f"max {np.max(np.abs(Ah-A_true))*1e3:.1f} mD, {time.perf_counter()-t:.1f} s")

    # ---------------- leave-out 1.0/3.0 D (offset model only)
    ah = P["fits"]["acc_holdout"]
    trA = ah["barrel"]["train_accommodations"]
    mtr = np.isin(np.round(acc, 9), np.round(trA, 9))
    barrel_ah = Barrel(br["reference_mm"], ah["barrel"]["k_coefficients_c0_c1_n"], acc[mtr],
                       ah["barrel"]["s_measured"][mtr], L)
    s_ah_err = barrel_ah.spl(acc[~mtr]) - s_meas[~mtr]
    hoA = np.isclose(A_true, 1.0, atol=1e-9) | np.isclose(A_true, 3.0, atol=1e-9)
    idx_ho = np.flatnonzero(hoA)
    free_ah = tuple(str(f) for f in Z["acc_holdout_families"])
    gp_ah = state_poly(Z["acc_holdout_coefficients"], free_ah, tx[idx_ho], ty[idx_ho])
    Ah_ah, nl_ah = estimate(obs[idx_ho], gp_ah, barrel_ah, FIELD_SETS["all9"])
    sig_ah, lb_ah = crb_and_linbias(A_true[idx_ho], obs[idx_ho], gp_ah, barrel_ah, FIELD_SETS["all9"])
    err_ah = Ah_ah - A_true[idx_ho]
    log(f"  acc_holdout: RMS {np.sqrt(np.mean(err_ah**2))*1e3:.2f} mD")

    # ---------------- Monte Carlo (1 um noise), interior A, held-out states
    rng = np.random.default_rng(12345)
    pool = np.flatnonzero(held & (A_true > 0.1) & (A_true < 3.9))
    mc_states = np.sort(rng.choice(pool, 200, replace=False))
    ndraw = 50
    noise = rng.normal(scale=SIGMA_MM, size=(len(mc_states), ndraw, 9, 2))
    mc = {}
    for key, M in models.items():
        o = (obs[mc_states][:, None] + noise).reshape(-1, 9, 2)
        rep = np.repeat(mc_states, ndraw)
        gps = {f: (None if v is None else v[rep]) for f, v in M["gp"].items()}
        Ah, _ = estimate(o, gps, barrel, FIELD_SETS["all9"])
        e = (Ah - A_true[rep]).reshape(len(mc_states), ndraw)
        R = res[(key, "all9")]
        b0 = R["err"][mc_states]
        sg = R["sig"][mc_states]
        mc[key] = {"rms_total": float(np.sqrt(np.mean(e ** 2))),
                   "rms_bias": float(np.sqrt(np.mean(b0 ** 2))),
                   "rms_sigma_crb": float(np.sqrt(np.mean(sg ** 2))),
                   "rms_pred": float(np.sqrt(np.mean(b0 ** 2 + sg ** 2))),
                   "rms_noise_part": float(np.sqrt(np.mean((e - b0[:, None]) ** 2))),
                   "mean_shift": float(np.mean(e - b0[:, None])),
                   "emp_std_over_crb_median": float(np.median(e.std(1) / sg)),
                   "frac_bias_gt_sigma": float(np.mean(np.abs(b0) > sg)),
                   "max_abs": float(np.max(np.abs(e)))}
        log(f"  MC {key}: {mc[key]}")

    # ---------------- report
    Ld = []
    w = Ld.append
    mk = list(models)
    lab = {k: models[k]["label"] for k in mk}
    w("# Estimating accommodation by inverting the P4 biaxial joint model")
    w("")
    w("Generated by `Script/estimate_p4_accommodation_from_model.py` (read-only use of "
      "`data/p4_biaxial_joint/p4_biaxial_joint_coefficients.npz` / `.pkl`, data "
      "`data/distortion_grid_p4_biaxial_macro/distortion_grid.pkl`).")
    w("")
    w("**Scope / honesty note.** Everything here is in-simulation: the 'observations' are noise-free CODE V "
      "real-ray coordinates of the same lens the model was fit to, and gaze (ADE, BDE) is assumed known exactly. "
      "The noise-free A error is therefore pure model bias (model misfit, ~4.5-6.4 um coordinate RMSE, mapped "
      "through dX/dA). Noise results assume i.i.d. Gaussian 1 um per coordinate and nothing else (no gaze error, "
      "no calibration error, no lens/eye variation). Domain: 0..4 D, |ADE|,|BDE| <= 20 deg.")
    w("")
    w("## Method")
    w("")
    w("- Observation per state: the 9 center-relative `real_x/y_mm` (18 numbers).")
    w("- Estimator: A_hat = argmin over A in [0, 4] D of sum of squared forward residuals (selected fields). "
      "Vectorized coarse scan at 0.01 D (401 points), then vectorized golden-section on [A_i-0.01, A_i+0.01] "
      "clipped to [0,4] (60 iterations), final pick among bracket ends/midpoint (so a minimum on a bound is "
      "returned as the bound).")
    w("- Continuous forward model: as in `fit_p4_biaxial_joint.py` (b = B_A(r0)/L, projective keystone, "
      "families quadratic in a=A/5 times order-1..4 polynomials in t_x=BDE/20, t_y=ADE/20). "
      "k1(A)=c0+c1*A^n is the stored power law. **s(A) = interpolating cubic spline (scipy CubicSpline, "
      "not-a-knot) through the 21 measured paraxial scale values**, which are the values the keystone fit "
      "was trained with; so at every grid A the continuous model equals the trained model exactly and the "
      "noise-free A error contains no s(A)-interpolation confound. r0 (A=0, zero-gaze real grid) fixed.")
    w(f"- s(A) interpolation error (leave-one-out over the 19 interior nodes): max {checks['s_loo_max']:.2e}, "
      f"RMS {checks['s_loo_rms']:.2e}; with mean |ds/dA| = {checks['dsdA_mean_abs']:.5f} /D this corresponds to "
      f"max {checks['s_loo_max']/checks['dsdA_mean_abs']*1e3:.2f} mD of A (relevant only between grid "
      f"accommodations, which are not evaluated here). Note s(A) is not perfectly smooth: its first "
      f"differences over 0.2 D range {checks['s_first_diff_range'][0]:.6f}..{checks['s_first_diff_range'][1]:.6f}.")
    w("- Derivative dX/dA: analytic (dk/dA = c1*n*A^(n-1), finite at A=0); finite-difference check max "
      f"relative error {checks['dA_fd_rel_err']:.2e} on 500 random states.")
    w("- Cramer-Rao / Gauss-Newton: sigma_A = 1 um / sqrt(sum_k (dX_k/dA)^2) at A_true over the selected fields. "
      "Linearized bias cross-check: b_lin = -(g.r)/(g.g) with g = dX/dA, r = model residual at A_true.")
    w("- Models (both trained on even-ADE-index/even-BDE-index states, all 21 A): **offset** = `main_even` "
      "(8 families incl. Tx, Ty; 336 coef); **center-exact** = `control_no_T` (Tx=Ty=0; 252 coef). In the "
      "center-exact model the center field is predicted as exactly 0, so it contributes nothing to the cost "
      "or to dX/dA (its 2 observed coordinates carry no A information in that model).")
    w(f"- Sets: held-out = all {held.sum():,} states not in even/even training; even/even = {even.sum():,} "
      f"training states; ray-failure = {rayfail.sum()} states with NaN paraxial rows (real coords valid; "
      "these are included in held-out/even sets as appropriate and also reported separately).")
    w("")
    w("### Reproduction checks")
    w("")
    w(f"- Continuous barrel at the 21 grid A vs the trained `barrel_mm`: max |diff| = "
      f"{checks['barrel_vs_trained_max_mm']:.2e} mm.")
    for key in mk:
        w(f"- {lab[key]}: forward residual at true A, complement all-nine coord RMSE = "
          f"{checks[f'{key}_complement_rmse_um']:.4f} um, even/even = {checks[f'{key}_train_rmse_um']:.4f} um "
          f"(report.md: {'4.5467 / 4.6728' if key == 'main_even' else '6.3831 / 6.4759'} um).")
    w("")

    w("## 1. Noise-free A error (A_hat - A_true), all nine fields")
    w("")
    w(SUMM_HDR[0])
    w(SUMM_HDR[1])
    for key in mk:
        e = res[(key, "all9")]["err"]
        w(summ_row(f"{lab[key]}: held-out", e[held]))
        w(summ_row(f"{lab[key]}: even/even (training)", e[even]))
        w(summ_row(f"{lab[key]}: all states", e))
        w(summ_row(f"{lab[key]}: ray-failure states", e[rayfail]))
    w("")
    for key in mk:
        R = res[(key, "all9")]
        e = R["err"]
        nb0 = int(np.sum(R["Ahat"] <= 1e-9))
        nb4 = int(np.sum(R["Ahat"] >= 4 - 1e-9))
        ok = np.abs(R["linb"]) < 0.2
        dlin = np.abs(e - R["linb"])
        interior = ok & (R["Ahat"] > 1e-9) & (R["Ahat"] < 4 - 1e-9)
        w(f"- {lab[key]}: estimates on bound A_hat=0: {nb0:,} (true A=0 states: "
          f"{int(np.sum(np.isclose(A_true, 0) ))}), A_hat=4: {nb4:,}; states whose coarse-scan cost curve "
          f"has >1 local minimum: {int(np.sum(R['nloc'] > 1))}. Linearized bias vs nonlinear error "
          f"(interior estimates, |b_lin|<0.2 D, {interior.sum():,} states): max |diff| "
          f"{np.max(dlin[interior])*1e3:.2f} mD, RMS diff {np.sqrt(np.mean(dlin[interior]**2))*1e3:.3f} mD.")
    w("")
    w("Note: true A = 0 and 4 D lie on the search bounds, so errors there are one-sided (clipped); "
      "this flattens bias/RMS in those two bins.")
    w("")

    # by A
    w("### By true A (held-out states, all nine fields)")
    w("")
    hdr = "| A (D) | states |" + "".join(f" {lab[k].split(' (')[0]} bias / RMS / max (mD) | median sigma_A (mD) |" for k in mk)
    w(hdr)
    w("|---:|---:|" + "---:|---:|" * len(mk))
    for a in acc:
        s = held & np.isclose(A_true, a, atol=1e-9)
        row = f"| {a:.1f} | {s.sum():,} |"
        for k in mk:
            R = res[(k, "all9")]
            e = R["err"][s]
            row += f" {e.mean()*1e3:.2f} / {np.sqrt(np.mean(e**2))*1e3:.2f} / {np.max(np.abs(e))*1e3:.1f} | {np.median(R['sig'][s])*1e3:.3f} |"
        w(row)
    w("")
    for nm, v in (("|BDE|", np.abs(BDE)), ("|ADE|", np.abs(ADE))):
        w(f"### By {nm} (held-out states, all nine fields)")
        w("")
        w(f"| {nm} bin (deg) | states |" + "".join(f" {lab[k].split(' (')[0]} bias / RMS / p99 / max (mD) | median sigma_A (mD) |" for k in mk))
        w("|---|---:|" + "---:|---:|" * len(mk))
        for lo, hi, bl in BINS:
            s = held & binmask(v, lo, hi)
            row = f"| {bl} | {s.sum():,} |"
            for k in mk:
                R = res[(k, "all9")]
                e = R["err"][s]
                row += (f" {e.mean()*1e3:.2f} / {np.sqrt(np.mean(e**2))*1e3:.2f} / "
                        f"{np.percentile(np.abs(e), 99)*1e3:.1f} / {np.max(np.abs(e))*1e3:.1f} | "
                        f"{np.median(R['sig'][s])*1e3:.3f} |")
            w(row)
        w("")

    w("### 2-D (|ADE| bin x |BDE| bin), all states (training + held-out), all nine fields")
    w("")
    for k in mk:
        R = res[(k, "all9")]
        for stat in ("RMS", "max"):
            w(f"{lab[k]} - {stat} |A error| (mD); rows |ADE|, columns |BDE|:")
            w("")
            w("| |ADE| \\ |BDE| |" + "".join(f" {b[2]} |" for b in BINS))
            w("|---|" + "---:|" * len(BINS))
            for lo, hi, bl in BINS:
                row = f"| {bl} |"
                for lo2, hi2, _ in BINS:
                    s = binmask(np.abs(ADE), lo, hi) & binmask(np.abs(BDE), lo2, hi2)
                    e = R["err"][s]
                    val = np.sqrt(np.mean(e ** 2)) if stat == "RMS" else np.max(np.abs(e))
                    row += f" {val*1e3:.1f} |"
                w(row)
            w("")

    w("### Gaze corners: exact (|ADE|, |BDE|) values, all 21 A (these are even/even training states)")
    w("")
    w("| (|ADE|, |BDE|) | states |" + "".join(f" {lab[k].split(' (')[0]} bias / RMS / max (mD) | sigma_A median / max (mD) |" for k in mk))
    w("|---|---:|" + "---:|---:|" * len(mk))
    for ca, cb in ((0, 0), (0, 20), (20, 0), (20, 20), (10, 10)):
        s = np.isclose(np.abs(ADE), ca) & np.isclose(np.abs(BDE), cb)
        row = f"| ({ca}, {cb}) | {s.sum()} |"
        for k in mk:
            R = res[(k, "all9")]
            e = R["err"][s]
            row += (f" {e.mean()*1e3:.2f} / {np.sqrt(np.mean(e**2))*1e3:.2f} / {np.max(np.abs(e))*1e3:.1f} | "
                    f"{np.median(R['sig'][s])*1e3:.3f} / {np.max(R['sig'][s])*1e3:.3f} |")
        w(row)
    w("")

    for k in mk:
        R = res[(k, "all9")]
        w(f"### 10 worst states, {lab[k]} (all states, all nine fields)")
        w("")
        w("| A true (D) | ADE | BDE | A_hat (D) | error (mD) | b_lin (mD) | sigma_A (mD) | state model RMS (um) | in training | ray-failure |")
        w("|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
        order = np.argsort(-np.abs(R["err"]))[:10]
        for i in order:
            srms = np.sqrt(np.mean(models[k]["resid"][i] ** 2)) * 1e3
            w(f"| {A_true[i]:.1f} | {ADE[i]:g} | {BDE[i]:g} | {R['Ahat'][i]:.4f} | {R['err'][i]*1e3:.1f} | "
              f"{R['linb'][i]*1e3:.1f} | {R['sig'][i]*1e3:.3f} | {srms:.2f} | {bool(even[i])} | {bool(rayfail[i])} |")
        w("")

    # ---- leave-out A
    w("## 2. Leave-out-accommodation case (1.0 and 3.0 D held out of the fit)")
    w("")
    w("Coefficients `acc_holdout` (offset model, 336 coef, trained on even/even states of the 19 remaining "
      "accommodations) and its refit power law are stored, so this was run. s(A) here = cubic spline through "
      f"the 19 training s values; spline error at the held-out A: 1.0 D {s_ah_err[0]:.2e}, 3.0 D "
      f"{s_ah_err[1]:.2e} (= {s_ah_err[0]/checks['dsdA_mean_abs']*1e3:.2f} / {s_ah_err[1]/checks['dsdA_mean_abs']*1e3:.2f} mD "
      "equivalent). The center-exact (T=0) leave-out fit was never run/stored, so it is skipped (not refit here).")
    w("")
    w(SUMM_HDR[0])
    w(SUMM_HDR[1])
    Rm = res[("main_even", "all9")]
    for a in (1.0, 3.0, None):
        s = np.ones(len(idx_ho), bool) if a is None else np.isclose(A_true[idx_ho], a)
        tag = "1.0+3.0 D" if a is None else f"{a:.1f} D"
        w(summ_row(f"leave-out fit, A={tag}, all ADE x BDE", err_ah[s]))
        w(summ_row(f"main fit (A in training), A={tag}", Rm["err"][idx_ho][s]))
    w("")
    w(f"Leave-out fit: estimates on a bound: {int(np.sum((Ah_ah<=1e-9)|(Ah_ah>=4-1e-9)))}; "
      f"median sigma_A {np.median(sig_ah)*1e3:.3f} mD.")
    w("")

    # ---- noise
    w("## 3. Noise sensitivity (1 um i.i.d. Gaussian per coordinate)")
    w("")
    w("### Cramer-Rao sigma_A (all states, all nine fields; mD)")
    w("")
    w("| subset |" + "".join(f" {lab[k].split(' (')[0]} median / min / max |" for k in mk))
    w("|---|" + "---:|" * len(mk))

    def sigrow(name, s):
        r = f"| {name} |"
        for k in mk:
            sg = res[(k, "all9")]["sig"][s] * 1e3
            r += f" {np.median(sg):.3f} / {sg.min():.3f} / {sg.max():.3f} |"
        return r
    w(sigrow("all states", np.ones(S, bool)))
    for a in (0.0, 1.0, 2.0, 3.0, 4.0):
        w(sigrow(f"A = {a:.0f} D", np.isclose(A_true, a)))
    for lo, hi, bl in BINS:
        w(sigrow(f"|ADE| {bl}", binmask(np.abs(ADE), lo, hi)))
    for lo, hi, bl in BINS:
        w(sigrow(f"|BDE| {bl}", binmask(np.abs(BDE), lo, hi)))
    w("")
    w(f"### Monte Carlo: {len(mc_states)} held-out states with 0.2 <= A <= 3.8 D (seed 12345) x {ndraw} noise draws")
    w("")
    w("| model | RMS total A error (mD) | RMS noise-free bias, same states (mD) | RMS CRB sigma_A (mD) | "
      "sqrt(mean(bias^2+sigma^2)) (mD) | RMS noise part (A_hat_noisy - A_hat_clean) (mD) | median emp.std/CRB | "
      "fraction states |bias|>sigma_A | max abs (mD) |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k in mk:
        m = mc[k]
        w(f"| {lab[k]} | {m['rms_total']*1e3:.2f} | {m['rms_bias']*1e3:.2f} | {m['rms_sigma_crb']*1e3:.3f} | "
          f"{m['rms_pred']*1e3:.2f} | {m['rms_noise_part']*1e3:.3f} | {m['emp_std_over_crb_median']:.3f} | "
          f"{m['frac_bias_gt_sigma']*100:.1f}% | {m['max_abs']*1e3:.2f} |")
    w("")

    # ---- field subsets
    w("## 4. Information content: field subsets (noise-free bias and CRB)")
    w("")
    w("| model | fields | held-out RMS (mD) | held-out p99 (mD) | held-out max (mD) | even/even RMS (mD) | "
      "median sigma_A (mD) | max sigma_A (mD) | sigma_A median at ADE=BDE=0 (mD) | states with >1 local min |")
    w("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    z0 = np.isclose(ADE, 0) & np.isclose(BDE, 0)
    for k in mk:
        for fs in FIELD_SETS:
            R = res[(k, fs)]
            e = R["err"]
            w(f"| {lab[k].split(' (')[0]} | {fs} | {np.sqrt(np.mean(e[held]**2))*1e3:.2f} | "
              f"{np.percentile(np.abs(e[held]), 99)*1e3:.1f} | {np.max(np.abs(e[held]))*1e3:.1f} | "
              f"{np.sqrt(np.mean(e[even]**2))*1e3:.2f} | {np.median(R['sig'])*1e3:.3f} | {np.max(R['sig'])*1e3:.3f} | "
              f"{np.median(R['sig'][z0])*1e3:.3f} | {int(np.sum(R['nloc'] > 1))} |")
    w("")

    # ---- zero gaze
    w("## 5. Identifiability at ADE=BDE=0 (barrel only) vs large gaze")
    w("")
    w("At ADE=BDE=0 every keystone family equals its identity value (all monomials have total order >= 1), "
      "so both models reduce to the barrel B_A(r0) alone (barrel-only check in report.md: 0.256 um coord RMSE). "
      "The 21 zero-gaze states are in the training set.")
    w("")
    w("| A (D) | offset error (mD) | center-exact error (mD) | sigma_A all9 (mD) | sigma_A corners4 (mD) | sigma_A field(-1,-1) (mD) |")
    w("|---:|---:|---:|---:|---:|---:|")
    for i in np.flatnonzero(z0):
        w(f"| {A_true[i]:.1f} | {res[('main_even','all9')]['err'][i]*1e3:.2f} | "
          f"{res[('control_no_T','all9')]['err'][i]*1e3:.2f} | {res[('main_even','all9')]['sig'][i]*1e3:.3f} | "
          f"{res[('main_even','corners4')]['sig'][i]*1e3:.3f} | {res[('main_even','field(-1,-1)')]['sig'][i]*1e3:.3f} |")
    w("")
    big = (np.abs(ADE) >= 16) & (np.abs(BDE) >= 16)
    w("| subset | offset RMS / max error (mD) | center-exact RMS / max error (mD) | offset sigma_A median (mD) | center-exact sigma_A median (mD) |")
    w("|---|---:|---:|---:|---:|")
    for nm, s in (("ADE=BDE=0 (21 states)", z0), ("|ADE|>=16 and |BDE|>=16", big),
                  ("|ADE|=|BDE|=20", np.isclose(np.abs(ADE), 20) & np.isclose(np.abs(BDE), 20))):
        e1, e2 = res[("main_even", "all9")]["err"][s], res[("control_no_T", "all9")]["err"][s]
        w(f"| {nm} | {np.sqrt(np.mean(e1**2))*1e3:.2f} / {np.max(np.abs(e1))*1e3:.1f} | "
          f"{np.sqrt(np.mean(e2**2))*1e3:.2f} / {np.max(np.abs(e2))*1e3:.1f} | "
          f"{np.median(res[('main_even','all9')]['sig'][s])*1e3:.3f} | {np.median(res[('control_no_T','all9')]['sig'][s])*1e3:.3f} |")
    w("")
    w("## Runtime")
    w("")
    w(f"{time.perf_counter()-T0:.0f} s wall (numpy limited to 6 threads).")
    w("")
    REPORT.write_text("\n".join(Ld), encoding="utf-8")

    np.savez_compressed(
        RESULTS, A_true=A_true, ADE=ADE, BDE=BDE, even_even=even, ray_failure=rayfail,
        **{f"{k}__{fs}__Ahat": r["Ahat"] for (k, fs), r in res.items()},
        **{f"{k}__{fs}__sigmaA_1um": r["sig"] for (k, fs), r in res.items()},
        **{f"{k}__{fs}__linbias": r["linb"] for (k, fs), r in res.items()},
        acc_holdout_state_index=idx_ho, acc_holdout_Ahat=Ah_ah, acc_holdout_sigmaA_1um=sig_ah,
        mc_states=mc_states)
    log(f"Done in {time.perf_counter()-T0:.0f} s -> {REPORT}")


if __name__ == "__main__":
    main()
