"""On-axis (ADE = BDE = 0) P4 accommodation model study.

Uses only the ADE=BDE=0 slice of data/distortion_grid_p4_biaxial_macro/distortion_grid.pkl
(21 accommodations 0..4 D x 3x3 fields).  Gaze terms, keystone and decentering are ignored.

Model family (center-relative real coordinates, mm):
    u0      = D_{k(0)}^-1(r0)              r0 = real grid at A = 0, D_k(u) = (1 + k|u|^2) u
    real(A) = D_{k(A)}( s(A) * u0 )
so A = 0 is reproduced exactly whenever s(0) = 1.

Tasks: (1) symmetry / radial-map structure, (2) per-accommodation free (s, k) fit and the origin of
the residual of the current model, (3) compact smooth models of A fitted directly to coordinates with
leave-one-accommodation-out CV, (4) accommodation estimation (noise-free, CRB, Monte Carlo),
(5) interpretation.  Writes data/p4_onaxis/report.md and data/p4_onaxis/p4_onaxis_results.npz.
Self-contained (no project imports) and no bytecode is written.
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "6"
sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import itertools  # noqa: E402
import pickle  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GRID = ROOT / "data" / "distortion_grid_p4_biaxial_macro" / "distortion_grid.pkl"
OUT = ROOT / "data" / "p4_onaxis"

# current refit constants (fit_p4_biaxial_joint.refit_barrel / report.md)
C0, C1, NPOW = -0.0185004967, 0.000205903596, 1.43363145
CORNERS, EDGES, CENTER = [0, 2, 6, 8], [1, 3, 5, 7], [4]
SIGMA_MM = 1e-3            # 1 um iid per coordinate
XS = 4.0                   # polynomial variable x = A / 4
RNG_SEED = 20261106
MC_DRAWS = 200


# ----------------------------------------------------------------------------- helpers
def D(u, k):
    k = np.asarray(k, float)
    return u * (1.0 + k[..., None, None] * np.sum(u * u, -1, keepdims=True)) if k.ndim else \
        u * (1.0 + k * np.sum(u * u, -1, keepdims=True))


def D_inv(xy, k, tol=1e-15):
    r = np.linalg.norm(xy, axis=-1)
    q = r.copy()
    for _ in range(60):
        step = (q * (1 + k * q * q) - r) / (1 + 3 * k * q * q)
        q = q - step
        if np.max(np.abs(step)) < tol:
            break
    g = np.divide(q, r, out=np.ones_like(r), where=r > 0)
    return xy * g[..., None]


def fit_group(para, real):
    """Direct per-accommodation k1 (fit_p4_radial_distortion.fit_group), paraxial -> real."""
    r2 = np.sum(para ** 2, -1)
    des = (para * r2[:, None]).T.reshape(-1)
    disp = (real - para).T.reshape(-1)
    k = float(des @ disp / (des @ des))
    res = disp - k * des
    return k, float(np.sqrt(np.mean(res ** 2)))


def metrics(err_mm):
    """err (..., 9, 2) mm -> vertex RMS, coord RMSE, max point (um)."""
    e = np.asarray(err_mm) * 1e3
    n = np.linalg.norm(e, axis=-1)
    return float(np.sqrt(np.mean(n ** 2))), float(np.sqrt(np.mean(e ** 2))), float(np.max(n))


def per_A_metrics(err_mm):
    return np.array([metrics(e) for e in err_mm])          # (nA, 3)


def fmt(x, d=3):
    return f"{x:.{d}f}"


# ------------------------------------------------------------------------------- data
def load():
    raw = GRID.read_bytes()
    d = pickle.loads(raw)
    m = np.isclose(d["eye_rotation_deg"], 0, atol=1e-9, rtol=0) & \
        np.isclose(d["eye_rotation_x_deg"], 0, atol=1e-9, rtol=0)
    s = d[m]
    s = s[np.lexsort((s["field_y_relative"], s["field_x_relative"], s["accommodation_d"]))]
    acc = np.unique(s["accommodation_d"])
    assert len(acc) == 21 and np.allclose(acc, np.linspace(0, 4, 21), atol=1e-9) and len(s) == 189
    sh = (21, 9)
    f = np.stack((s["field_x_relative"], s["field_y_relative"]), -1).reshape(sh + (2,))
    exp = np.array([[x, y] for x in (-1., 0., 1.) for y in (-1., 0., 1.)])
    assert np.array_equal(f, np.broadcast_to(exp, f.shape))
    real = np.stack((s["real_x_mm"], s["real_y_mm"]), -1).reshape(sh + (2,))
    para = np.stack((s["paraxial_x_mm"], s["paraxial_y_mm"]), -1).reshape(sh + (2,))
    assert np.isfinite(real).all() and np.isfinite(para).all()
    return {"A": np.linspace(0, 4, 21), "acc_raw": acc, "fields": exp, "real": real, "para": para,
            "sha": hashlib.sha256(raw).hexdigest()}


# ------------------------------------------------------------------------- task 1
def symmetry(G):
    F = G["fields"]
    idx = {tuple(v): i for i, v in enumerate(F)}
    out = {}
    for name in ("real", "para"):
        X = G[name]
        swap = np.array([idx[(fy, fx)] for fx, fy in F])
        mx = np.array([idx[(-fx, fy)] for fx, fy in F])
        my = np.array([idx[(fx, -fy)] for fx, fy in F])
        out[name] = {
            "x<->y transpose": np.max(np.abs(X[:, swap][..., ::-1] - X)) * 1e3,
            "x mirror (fx->-fx)": np.max(np.abs(X[:, mx] * [-1, 1] - X)) * 1e3,
            "y mirror (fy->-fy)": np.max(np.abs(X[:, my] * [1, -1] - X)) * 1e3,
            "center |coord|": np.max(np.abs(X[:, 4])) * 1e3,
            "edge tangential comp": max(np.max(np.abs(X[:, [1, 7], 1])), np.max(np.abs(X[:, [3, 5], 0]))) * 1e3,
            "corner |x|-|y|": np.max(np.abs(np.abs(X[:, CORNERS, 0]) - np.abs(X[:, CORNERS, 1]))) * 1e3,
            "edge radius spread": np.max(np.ptp(np.linalg.norm(X[:, EDGES], axis=-1), axis=1)) * 1e3,
            "corner radius spread": np.max(np.ptp(np.linalg.norm(X[:, CORNERS], axis=-1), axis=1)) * 1e3,
        }
    # direction of real vs paraxial (angle, urad)
    R, P = G["real"][:, CORNERS + EDGES], G["para"][:, CORNERS + EDGES]
    cross = R[..., 0] * P[..., 1] - R[..., 1] * P[..., 0]
    out["angle_urad"] = float(np.max(np.abs(cross / (np.linalg.norm(R, axis=-1) * np.linalg.norm(P, axis=-1)))) * 1e6)
    re = np.linalg.norm(G["real"][:, EDGES], axis=-1).mean(1)
    rc = np.linalg.norm(G["real"][:, CORNERS], axis=-1).mean(1)
    pe = np.linalg.norm(G["para"][:, EDGES], axis=-1).mean(1)
    pc = np.linalg.norm(G["para"][:, CORNERS], axis=-1).mean(1)
    out.update(re=re, rc=rc, pe=pe, pc=pc,
               pc_over_pe_dev=float(np.max(np.abs(pc / pe - np.sqrt(2)))))
    return out


# ------------------------------------------------------------------------- task 2
def per_A_free(G, k0):
    """LS fit of real = a*u0 + b*|u0|^2 u0 per accommodation (exact 2x2 in radii); s=a, k=b/a^3."""
    u0 = D_inv(G["real"][0], k0)
    c1 = u0.reshape(-1)
    c2 = (u0 * np.sum(u0 ** 2, -1, keepdims=True)).reshape(-1)
    M = np.column_stack((c1, c2))
    s_fit, k_fit, res = [], [], []
    for ia in range(21):
        y = G["real"][ia].reshape(-1)
        (a, b), *_ = np.linalg.lstsq(M, y, rcond=None)
        s_fit.append(a); k_fit.append(b / a ** 3)
        pred = D(a * u0, b / a ** 3)
        res.append(metrics(pred - G["real"][ia]))
    return np.array(s_fit), np.array(k_fit), np.array(res), u0


def best_k_given_s(G, u0, s):
    out = []
    for ia in range(21):
        v = s[ia] * u0
        des = (v * np.sum(v ** 2, -1, keepdims=True)).reshape(-1)
        k = float(des @ (G["real"][ia] - v).reshape(-1) / (des @ des))
        out.append((k, *metrics(D(v, k) - G["real"][ia])))
    return np.array(out)


def best_s_given_k(G, u0, k):
    out = []
    for ia in range(21):
        def f(sv):
            return ((D(sv[0] * u0, k[ia]) - G["real"][ia]) * 1e3).reshape(-1)
        r = least_squares(f, [1.0], xtol=1e-15, ftol=1e-15, gtol=1e-15)
        out.append((r.x[0], *metrics(D(r.x[0] * u0, k[ia]) - G["real"][ia])))
    return np.array(out)


def powerlaw_fit(A, k):
    p0 = np.array([k[0], (k[-1] - k[0]) / A[-1], 1.0])
    r = least_squares(lambda p: p[0] + p[1] * np.power(A, p[2]) - k, p0,
                      bounds=([-np.inf, -np.inf, 1e-6], [np.inf, np.inf, 10.0]),
                      x_scale=np.array([0.02, 0.001, 1.0]), ftol=1e-14, xtol=1e-14, gtol=1e-14, max_nfev=10000)
    return r.x


def s_quad_law(A, s, train):
    m = train & (A > 0)
    X = np.column_stack((A[m], A[m] ** 2))
    c = np.linalg.lstsq(X, s[m] - 1, rcond=None)[0]
    return lambda a: 1 + c[0] * a + c[1] * a ** 2


# ------------------------------------------------------------------------- task 3 models
class Model:
    """s(A) = 1 + sum_{j=1..p} a_j x^j,  x = A/4.
       k(A): 'pow' -> c0 + c1 A^n ; ('poly', q) -> sum_{j=0..q} b_j x^j.
       base u0 = D_{k(0)}^-1(r0)."""

    def __init__(self, p, ktype, q=None, fix_k0=False):
        self.p, self.ktype, self.q, self.fix = p, ktype, q, fix_k0
        self.nk = (3 if ktype == "pow" else q + 1) - (1 if fix_k0 else 0)
        self.npar = p + self.nk
        base = f"s poly{p} + k power-law" if ktype == "pow" else f"s poly{p} + k poly{q}"
        self.name = base + (" [k0=c0]" if fix_k0 else " [k0 free]")

    def kfull(self, th):
        kt = np.asarray(th[self.p:], float)
        return np.concatenate(([C0], kt)) if self.fix else kt

    def s(self, th, A):
        x = np.asarray(A, float) / XS
        return 1 + sum(th[j - 1] * x ** j for j in range(1, self.p + 1))

    def k(self, th, A):
        A = np.asarray(A, float)
        kt = self.kfull(th)
        if self.ktype == "pow":
            return kt[0] + kt[1] * np.power(np.maximum(A, 0.0), kt[2])
        x = A / XS
        return sum(kt[j] * x ** j for j in range(self.q + 1))

    def predict(self, th, A, r0):
        u0 = D_inv(r0, float(self.k(th, 0.0)))
        A = np.atleast_1d(np.asarray(A, float))
        v = self.s(th, A)[:, None, None] * u0[None]
        return D(v, self.k(th, A))

    def init(self, A, s_fit, k_fit, train):
        x = A[train] / XS
        Xs = np.column_stack([x ** j for j in range(1, self.p + 1)])
        a = np.linalg.lstsq(Xs, s_fit[train] - 1, rcond=None)[0]
        if self.ktype == "pow":
            kt = powerlaw_fit(A[train], k_fit[train])
            if self.fix:
                kt = kt[1:]
        else:
            j0 = 1 if self.fix else 0
            Xk = np.column_stack([x ** j for j in range(j0, self.q + 1)])
            kt = np.linalg.lstsq(Xk, k_fit[train] - (C0 if self.fix else 0.0), rcond=None)[0]
        return np.concatenate((a, kt))

    def fit(self, G, train, s_fit, k_fit):
        A, real, r0 = G["A"], G["real"], G["real"][0]

        def res(th):
            return ((self.predict(th, A[train], r0) - real[train]) * 1e3).reshape(-1)
        starts = [self.init(A, s_fit, k_fit, train)]
        if self.ktype == "pow":
            for n in (0.7, 1.0, 1.5, 2.0, 3.0):
                t = starts[0].copy(); t[-1] = n; starts.append(t)
        lo = np.full(self.npar, -np.inf); hi = np.full(self.npar, np.inf)
        if self.ktype == "pow":
            lo[-1], hi[-1] = 0.2, 10.0
        best = None
        for t0 in starts:
            t0 = np.clip(t0, lo + 1e-9, hi - 1e-9)
            try:
                r = least_squares(res, t0, bounds=(lo, hi), method="trf", x_scale="jac",
                                  xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=4000)
            except ValueError:
                continue
            if best is None or r.cost < best.cost:
                best = r
        return best.x


class LinearAB:
    """real = a(A) u0 + b(A) |u0|^2 u0 with u0 = D_c0^-1(r0) (k0 fixed = C0),
       a = 1 + sum_{j=1..P} a_j x^j,  b = C0 + sum_{j=1..Q} b_j x^j  (s = a, k = b / a^3).
       Linear least squares; with P = Q equivalent to independent degree-P polynomials for r_e(A), r_c(A)."""

    def __init__(self, P, Q, k0=C0):
        self.P, self.Q, self.k0 = P, Q, k0
        self.npar = P + Q
        self.name = f"linear a poly{P}, b poly{Q} (k0 fixed)"

    def basis(self, A, u0):
        x = np.atleast_1d(np.asarray(A, float)) / XS
        g1 = u0.reshape(-1); g2 = (u0 * np.sum(u0 ** 2, -1, keepdims=True)).reshape(-1)
        cols = [np.outer(x ** j, g1) for j in range(1, self.P + 1)] + \
               [np.outer(x ** j, g2) for j in range(1, self.Q + 1)]
        return np.stack(cols, -1), g1 + self.k0 * g2          # (nA, 18, npar), offset (18,)

    def fit(self, G, train, *_):
        u0 = D_inv(G["real"][0], self.k0)
        B, off = self.basis(G["A"][train], u0)
        y = G["real"][train].reshape(train.sum(), -1) - off
        th = np.linalg.lstsq(B.reshape(-1, self.npar) * 1e3, y.reshape(-1) * 1e3, rcond=None)[0]
        return th

    def predict(self, th, A, r0):
        u0 = D_inv(r0, self.k0)
        B, off = self.basis(A, u0)
        return (B @ th + off).reshape(-1, 9, 2)


class Current:
    """Reference: k = c0 + c1 A^n (power law fitted to direct k1), s = measured paraxial ratio.
       Continuous s for estimation: cubic spline of measured s; LOO held-out s: refit_barrel's quadratic law."""

    def __init__(self, G, kcoef, s_meas, s_mode="measured", train=None):
        self.kc, self.s_meas, self.name = np.asarray(kcoef), s_meas, "current"
        self.npar = 3 + 20
        A = G["A"]
        train = np.ones(21, bool) if train is None else train
        self.spline = CubicSpline(A[train], s_meas[train])
        self.quad = s_quad_law(A, s_meas, train)
        self.train, self.s_mode = train, s_mode

    def s(self, A):
        A = np.atleast_1d(A)
        if self.s_mode == "spline":
            return self.spline(A)
        out = self.quad(A)
        g = np.rint(A / 0.2).astype(int)
        on = np.isclose(A, g * 0.2, atol=1e-9) & (g >= 0) & (g <= 20)
        on[on] &= self.train[g[on]]
        out[on] = self.s_meas[g[on]]
        return out

    def predict(self, _th, A, r0):
        A = np.atleast_1d(np.asarray(A, float))
        u0 = D_inv(r0, self.kc[0])
        k = self.kc[0] + self.kc[1] * np.power(np.maximum(A, 0.0), self.kc[2])
        return D(self.s(A)[:, None, None] * u0[None], k)


# ------------------------------------------------------------------------- task 4 estimation
def estimate(pred_fn, obs, ids=None, step=5e-4, gn_iter=6):
    """obs (N, 9, 2) mm -> A_hat (N,) in [0, 4]; scan + parabolic + Gauss-Newton refinement."""
    ids = list(range(9)) if ids is None else ids
    grid = np.linspace(0, 4, int(round(4 / step)) + 1)
    P = pred_fn(grid)[:, ids].reshape(len(grid), -1)
    O = obs[:, ids].reshape(len(obs), -1)
    sse = np.sum(O ** 2, 1)[:, None] - 2 * O @ P.T + np.sum(P ** 2, 1)[None]
    j = np.argmin(sse, 1)
    jj = np.clip(j, 1, len(grid) - 2)
    f0, f1, f2 = sse[np.arange(len(O)), jj - 1], sse[np.arange(len(O)), jj], sse[np.arange(len(O)), jj + 1]
    den = f0 - 2 * f1 + f2
    off = np.where(den > 0, 0.5 * (f0 - f2) / np.where(den > 0, den, 1), 0.0)
    a = np.clip(grid[jj] + np.clip(off, -1, 1) * step, 0, 4)
    a = np.where((j == 0) | (j == len(grid) - 1), grid[j], a)
    h = 1e-5
    for _ in range(gn_iter):
        p = pred_fn(a)[:, ids].reshape(len(a), -1)
        dp = (pred_fn(np.clip(a + h, 0, 4 + 1e-3))[:, ids].reshape(len(a), -1) -
              pred_fn(np.clip(a - h, -1e-3, 4))[:, ids].reshape(len(a), -1)) / (2 * h)
        a_new = a + np.sum(dp * (O - p), 1) / np.sum(dp * dp, 1)
        a = np.clip(a_new, 0, 4)
    return a


def crb(pred_fn, A, ids, h=1e-4):
    Ap = np.clip(A + h, None, 4.0 + h); Am = np.clip(A - h, 0, None)
    J = (pred_fn(Ap) - pred_fn(Am)) / (Ap - Am)[:, None, None]           # (nA, 9, 2) mm/D
    fi = np.sum(J[:, ids] ** 2, (1, 2)) / SIGMA_MM ** 2
    return 1 / np.sqrt(fi), J


# ------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    G = load()
    A, real, para = G["A"], G["real"], G["para"]
    r0 = real[0]
    L = []                       # report lines

    def w(s=""):
        L.append(s)

    # ---------------- task 1
    S = symmetry(G)
    re, rc, pe, pc = S["re"], S["rc"], S["pe"], S["pc"]
    dre, drc = np.gradient(re, A), np.gradient(rc, A)

    # ---------------- task 2
    kd = np.array([fit_group(para[i], real[i]) for i in range(21)])   # direct k1, rmse (mm)
    k_direct, fg_rmse = kd[:, 0], kd[:, 1]
    kcoef_mine = powerlaw_fit(A, k_direct)
    p0 = para[0].reshape(-1)
    s_meas = np.array([p0 @ para[i].reshape(-1) / (p0 @ p0) for i in range(21)])
    cur = Current(G, [C0, C1, NPOW], s_meas)
    err_cur = cur.predict(None, A, r0) - real
    m_cur = per_A_metrics(err_cur)
    u0_d = D_inv(r0, k_direct[0])
    err_exact = np.stack([D(s_meas[i] * u0_d, k_direct[i]) for i in range(21)]) - real
    m_exact = per_A_metrics(err_exact)
    s_fit, k_fit, m_free, u0 = per_A_free(G, C0)
    s_fit_d, k_fit_d, m_free_d, _ = per_A_free(G, k_direct[0])
    bk = best_k_given_s(G, u0, s_meas)                  # s measured, k free (base c0)
    kpow = C0 + C1 * A ** NPOW
    bs = best_s_given_k(G, u0, kpow)                    # k power law, s free
    bk_d = best_k_given_s(G, u0_d, s_meas)
    bs_d = best_s_given_k(G, u0_d, k_direct)
    # pseudo-paraxial base vs true paraxial
    u_e, u_c = np.linalg.norm(u0[EDGES], axis=-1).mean(), np.linalg.norm(u0[CORNERS], axis=-1).mean()

    # ---------------- task 3
    models = [Model(p, "pow", fix_k0=f) for f in (True, False) for p in range(1, 5)] + \
             [Model(p, "poly", q, fix_k0=f) for f in (True, False) for p in range(1, 5) for q in range(1, 5)] + \
             [LinearAB(P, Q) for P in range(1, 6) for Q in range(1, 6)]
    allA = np.ones(21, bool)
    results = []
    for M in models:
        th = M.fit(G, allA, s_fit, k_fit)
        err = M.predict(th, A, r0) - real
        mA = per_A_metrics(err)
        loo_err = np.zeros_like(real)
        loo_th = []
        for i in range(21):
            tr = allA.copy(); tr[i] = False
            thi = M.fit(G, tr, s_fit, k_fit)
            loo_th.append(thi)
            loo_err[i] = M.predict(thi, A[i:i + 1], r0)[0] - real[i]
        results.append({"model": M, "th": th, "err": err, "mA": mA, "pool": metrics(err),
                        "pool_pos": metrics(err[1:]), "loo_err": loo_err, "loo_th": loo_th,
                        "loo_mA": per_A_metrics(loo_err), "cv_int": metrics(loo_err[1:20]),
                        "cv_A4": metrics(loo_err[20:21]), "cv_A0": metrics(loo_err[0:1])})
    # current model LOO (k power law refit on direct k1 without A_i, held-out s from quadratic law)
    cur_loo_err = np.zeros_like(real)
    cur_loo_err_sp = np.zeros_like(real)
    cur_loo_models = []
    for i in range(21):
        tr = allA.copy(); tr[i] = False
        kc = powerlaw_fit(A[tr], k_direct[tr])
        Ci = Current(G, kc, s_meas, train=tr)
        cur_loo_models.append(Ci)
        cur_loo_err[i] = Ci.predict(None, A[i:i + 1], r0)[0] - real[i]
        Ci.s_mode = "spline"
        cur_loo_err_sp[i] = Ci.predict(None, A[i:i + 1], r0)[0] - real[i]
        Ci.s_mode = "measured"
    cur_res = {"pool": metrics(err_cur), "pool_pos": metrics(err_cur[1:]), "mA": m_cur,
               "cv_int": metrics(cur_loo_err[1:20]), "cv_A4": metrics(cur_loo_err[20:21]),
               "cv_int_sp": metrics(cur_loo_err_sp[1:20]), "cv_A4_sp": metrics(cur_loo_err_sp[20:21])}

    order = sorted(range(len(results)), key=lambda j: results[j]["cv_int"][0])
    best = results[order[0]]
    # parsimonious pick: fewest params within 10 % of best CV (and below 0.05 um)
    thr = max(best["cv_int"][0] * 1.10, best["cv_int"][0] + 0.005)
    pars = min((r for r in results if r["cv_int"][0] <= thr), key=lambda r: (r["model"].npar, r["cv_int"][0]))
    # parsimonious pick among the nonlinear (s, k) family too
    nl = [r for r in results if isinstance(r["model"], Model) and r["model"].fix]
    nlf = [r for r in results if isinstance(r["model"], Model) and not r["model"].fix]
    best_free = min(nlf, key=lambda r: r["cv_int"][0])
    # ray-trace jitter floor: residual of degree-8 polynomials in A fitted to r_e(A), r_c(A)
    jit = []
    for rr in (re, rc):
        cf = np.polynomial.polynomial.polyfit(A / XS, rr, 8)
        jit.append(rr - np.polynomial.polynomial.polyval(A / XS, cf))
    jit = np.array(jit) * 1e3                    # um, (2, 21)
    jit_vrms = float(np.sqrt(np.mean(4 * jit[0] ** 2 + 4 * jit[1] ** 2) / 9))
    best_nl = min(nl, key=lambda r: r["cv_int"][0])
    thr_nl = max(best_nl["cv_int"][0] * 1.10, best_nl["cv_int"][0] + 0.005)
    pars_nl = min((r for r in nl if r["cv_int"][0] <= thr_nl), key=lambda r: (r["model"].npar, r["cv_int"][0]))
    USE = pars_nl            # model used for the recommended formula / estimation

    # ---------------- task 4
    rng = np.random.default_rng(RNG_SEED)
    cur_sp = Current(G, [C0, C1, NPOW], s_meas, s_mode="spline")
    est_models = {"current (spline s)": (lambda a: cur_sp.predict(None, a, r0)),
                  f"best-CV {best['model'].name}": (lambda a, R=best: R["model"].predict(R["th"], a, r0)),
                  f"recommended {USE['model'].name}": (lambda a, R=USE: R["model"].predict(R["th"], a, r0))}
    est = {}
    for name, fn in est_models.items():
        ahat = estimate(fn, real)
        sig_all, J = crb(fn, A, list(range(9)))
        sig_e, _ = crb(fn, A, EDGES)
        sig_c, _ = crb(fn, A, CORNERS)
        noise = rng.normal(0, SIGMA_MM, (21, MC_DRAWS, 9, 2))
        obs = (real[:, None] + noise).reshape(-1, 9, 2)
        amc = estimate(fn, obs).reshape(21, MC_DRAWS)
        mc_rms = np.sqrt(np.mean((amc - A[:, None]) ** 2, 1))
        mc_bias = np.mean(amc - A[:, None], 1)
        mc_e = estimate(fn, obs, EDGES).reshape(21, MC_DRAWS)
        mc_c = estimate(fn, obs, CORNERS).reshape(21, MC_DRAWS)
        est[name] = {"ahat": ahat, "sig_all": sig_all, "sig_e": sig_e, "sig_c": sig_c, "J": J,
                     "mc_rms": mc_rms, "mc_bias": mc_bias,
                     "mc_rms_e": np.sqrt(np.mean((mc_e - A[:, None]) ** 2, 1)),
                     "mc_rms_c": np.sqrt(np.mean((mc_c - A[:, None]) ** 2, 1))}
    # LOO estimation: model fit without A_i, estimate A_i from true coords
    loo_est = {}
    for C in cur_loo_models:
        C.s_mode = "spline"
    loo_est["current (LOO spline s)"] = np.array(
        [estimate(lambda a, C=cur_loo_models[i]: C.predict(None, a, r0), real[i:i + 1])[0] for i in range(21)])
    for R, tag in ((best, "best-CV"), (USE, "recommended")):
        loo_est[f"{tag} {R['model'].name}"] = np.array(
            [estimate(lambda a, th=R["loo_th"][i]: R["model"].predict(th, a, r0), real[i:i + 1])[0]
             for i in range(21)])
    # model-free sensitivity from data
    sig_data_all = SIGMA_MM / np.sqrt(4 * dre ** 2 + 4 * drc ** 2)
    sig_data_e = SIGMA_MM / np.sqrt(4 * dre ** 2)
    sig_data_c = SIGMA_MM / np.sqrt(4 * drc ** 2)

    # s linearity
    ps1 = np.polyfit(A, s_meas, 1); ps2 = np.polyfit(A, s_meas, 2)
    lin_res_um = np.max(np.abs(np.polyval(ps1, A) - s_meas)) * pe[0] * 1e3
    lin_res_um_c = np.max(np.abs(np.polyval(ps1, A) - s_meas)) * pc[0] * 1e3
    xA = A / XS
    th1 = np.linalg.lstsq(xA[1:, None], s_fit[1:] - 1, rcond=None)[0]
    lin_fit_res = np.max(np.abs(1 + th1[0] * xA - s_fit))
    s_poly_res = []           # (deg, rms ppm, max ppm, argmax A, highest coef) for s_A (c0 base) and s_meas
    for name_s, sv in (("s_A (coordinate fit, c0 base)", s_fit), ("s_meas (paraxial)", s_meas)):
        for dg in range(1, 5):
            Xd = np.column_stack([xA[1:] ** j for j in range(1, dg + 1)])
            cd = np.linalg.lstsq(Xd, sv[1:] - 1, rcond=None)[0]
            rr = sv[1:] - 1 - Xd @ cd
            s_poly_res.append((name_s, dg, np.sqrt(np.mean(rr ** 2)) * 1e6, np.max(np.abs(rr)) * 1e6,
                               A[1:][np.argmax(np.abs(rr))], cd))

    # ======================================================================= report
    w("# P4 on-axis (ADE = BDE = 0) accommodation model")
    w()
    w(f"Generated by `Script/p4_onaxis_accommodation_model.py` ({time.strftime('%Y-%m-%d %H:%M')}). Input "
      f"`data/distortion_grid_p4_biaxial_macro/distortion_grid.pkl` (sha256 {G['sha'][:16]}...), slice "
      "ADE = BDE = 0 selected with `np.isclose(atol=1e-9)`: 189 rows = 21 accommodations (0..4 D, step 0.2) x "
      "9 fields, sorted by (A, field_x, field_y). Field index order: 0..8 = (fx, fy) in "
      "(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1); corners 0,2,6,8; edge midpoints 1,3,5,7; "
      "center 4. All coordinates are center-relative real (or paraxial) image coordinates in mm; all errors "
      "in um. Metrics: vertex RMS = sqrt(mean over 9 points of |e|^2), coord RMSE = sqrt(mean of 18 "
      "coordinate errors^2), max point = max |e| over points. Noise-free in-simulation data; noise studies "
      "assume 1 um iid Gaussian per coordinate only.")
    w()
    w("## 1. Data structure: symmetry and radial reduction")
    w()
    w("Maximum symmetry residual over all 21 accommodations (um):")
    w()
    w("| test | real | paraxial |")
    w("|---|---|---|")
    for k in S["real"]:
        w(f"| {k} | {S['real'][k]:.2e} | {S['para'][k]:.2e} |")
    w()
    w(f"Real points lie on the paraxial ray direction to within {S['angle_urad']:.2e} urad; paraxial corner/edge "
      f"radius ratio deviates from sqrt(2) by {S['pc_over_pe_dev']:.1e}. So the on-axis data are exactly "
      "(to print precision) D4-symmetric and purely radial: each accommodation is described by the paraxial "
      "edge radius p(A) (corner paraxial radius = sqrt(2) p) and two real radii, the edge-midpoint radius "
      "r_e(A) and the corner radius r_c(A) (corner coords = (r_c/sqrt2)(+-1, +-1)). The center is 0 by "
      "construction (center-relative). **Truly independent numbers per accommodation: 2 real (r_e, r_c) "
      "+ 1 paraxial (p)**; the 18 observed coordinates contain only 2 independent numbers, and A = 0 is the "
      "reference r0. Any 2-parameter-per-A radial model fits the real data exactly. In the residual sums: "
      "sum_9 |e|^2 = 4 de_r^2 + 4 dc_r^2, vertex RMS = sqrt(that/9), coord RMSE = sqrt(that/18).")
    w()
    w("Radial map r_real = f(r_paraxial, A), sampled at r_p = p (edge) and sqrt(2) p (corner):")
    w()
    w("| A (D) | p (mm) | r_e (mm) | r_e/p - 1 (%) | sqrt2 p (mm) | r_c (mm) | r_c/(sqrt2 p) - 1 (%) | dr_e/dA (um/D) | dr_c/dA (um/D) |")
    w("|---|---|---|---|---|---|---|---|---|")
    for i in range(21):
        w(f"| {A[i]:.1f} | {pe[i]:.9f} | {re[i]:.9f} | {100 * (re[i] / pe[i] - 1):.4f} | {pc[i]:.9f} | "
          f"{rc[i]:.9f} | {100 * (rc[i] / pc[i] - 1):.4f} | {dre[i] * 1e3:.3f} | {drc[i] * 1e3:.3f} |")
    w()
    w("## 2. Why the current structure leaves ~0.6 um, and the per-accommodation free (s, k) fit")
    w()
    w(f"Direct k1 (fit_group: paraxial -> real, cubic only) at A = 0 is {k_direct[0]:.10f} /mm^2, while the "
      f"power-law refit constant is c0 = {C0} (my own power-law refit of the 21 direct k1: c0 = "
      f"{kcoef_mine[0]:.10f}, c1 = {kcoef_mine[1]:.9e}, n = {kcoef_mine[2]:.8f}, i.e. the stated constants are "
      "reproduced). The single-k1 paraxial->real fit itself is not exact: its RMS coordinate residual is "
      f"{fg_rmse[0] * 1e3:.3f} um at 0 D and {fg_rmse[-1] * 1e3:.3f} um at 4 D (real distortion has higher-order "
      "radial terms that one cubic term cannot match at both radii).")
    w()
    w(f"The base u0 = D_c0^-1(r0) is therefore a pseudo-paraxial grid, not the paraxial grid: u0 edge radius "
      f"{u_e:.9f} vs paraxial {pe[0]:.9f} mm ({(u_e / pe[0] - 1) * 1e6:+.1f} ppm), corner {u_c:.9f} vs "
      f"{pc[0]:.9f} mm ({(u_c / pc[0] - 1) * 1e6:+.1f} ppm). The A = 0 higher-order mismatch is absorbed into "
      "u0 (A = 0 is exact by construction), but at A > 0 the measured paraxial scale s(A) = p(A)/p(0) and the "
      "direct k1(A) describe the paraxial->real map, which is not the map u0 -> real the model actually "
      "applies. The higher-order part of the distortion changes with A and nothing in (s_meas, k1) can carry "
      "it. Decomposition of the error, per A (vertex RMS um; base c0 unless noted):")
    w()
    w("| A | current (pow k, s_meas) | exact k1_direct + s_meas (base k_direct(0)) | s_meas, best k (base c0) | s_meas, best k (base kd0) | pow k, best s | k_direct, best s (base kd0) | free (s,k) | fit_group rmse (um) |")
    w("|---|---|---|---|---|---|---|---|---|")
    for i in range(21):
        w(f"| {A[i]:.1f} | {m_cur[i, 0]:.3f} | {m_exact[i, 0]:.3f} | {bk[i, 1]:.3f} | {bk_d[i, 1]:.3f} | "
          f"{bs[i, 1]:.3f} | {bs_d[i, 1]:.3f} | {m_free[i, 0]:.1e} | {fg_rmse[i] * 1e3:.3f} |")
    w()
    w(f"Current model, pooled over 21 A: vertex RMS {metrics(err_cur)[0]:.3f}, coord RMSE "
      f"{metrics(err_cur)[1]:.3f}, max point {metrics(err_cur)[2]:.3f} um; at 4 D vertex RMS {m_cur[-1, 0]:.3f} um, "
      f"max point (all A) {np.max(m_cur[:, 2]):.3f} um (stated: 0.594 / 0.89). Exact per-A k1 + measured s "
      f"(base D_kdirect(0)^-1): 4 D vertex RMS {m_exact[-1, 0]:.3f}, max point {np.max(m_exact[:, 2]):.3f} um "
      "(stated: 0.568 / 0.796).")
    w()
    w(f"**Per-A free (s_A, k_A), least squares on the 18 coordinates with fixed base D_c0^-1(r0):** the model "
      "real = a u0 + b |u0|^2 u0 (a = s, b = k s^3) is linear in (a, b) and 2 parameters meet 2 radii, so it is "
      f"exactly determined: max residual over all A = {np.max(m_free[:, 2]):.1e} um (machine precision). The "
      "same holds for any base k0 (gauge): the fitted (s_A, k_A) depend on k0 (columns with base k_direct(0) "
      "given for comparison). The residual of the current model is therefore entirely due to using "
      "s_meas / k1_direct (paraxial quantities) instead of the values the real coordinates want; the "
      "model *structure* D_k(A)(s(A) u0) is not limiting on-axis.")
    w()
    w("| A | s_meas | s_A (fit, c0 base) | (s_A - s_meas) ppm | x p0 (um) | k_direct | k power-law | k_A (fit, c0 base) | s_A (kd0 base) | k_A (kd0 base) |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for i in range(21):
        w(f"| {A[i]:.1f} | {s_meas[i]:.10f} | {s_fit[i]:.10f} | {(s_fit[i] - s_meas[i]) * 1e6:+.2f} | "
          f"{(s_fit[i] - s_meas[i]) * pe[0] * 1e3:+.3f} | {k_direct[i]:.9f} | {kpow[i]:.9f} | {k_fit[i]:.9f} | "
          f"{s_fit_d[i]:.10f} | {k_fit_d[i]:.9f} |")
    w()
    w("## 3. Compact smooth models of A fitted directly to coordinates")
    w()
    w("Objective: all-nine real-coordinate error (um), least squares over all 21 accommodations. "
      "x = A/4. Nonlinear family: s(A) = 1 + sum_{j=1..p} a_j x^j; k(A) = c0 + c1 A^n (power law, 3 params) "
      "or k(A) = sum_{j=0..q} b_j x^j (q+1 params); base u0 = D_{k(0)}^-1(r0), so A = 0 is exact for every "
      "model. Each is run twice: [k0=c0] fixes k(0) = c0 = -0.0185004967 (one parameter fewer, physically "
      "interpretable s and k), [k0 free] lets k(0) float. Because the per-A fit is exact for ANY base k0 "
      "(section 2), k(0) is close to a gauge freedom: freed, the optimiser may move it far from the physical "
      "value (see the k(0) column) and s(A), k(A) lose their paraxial meaning while the coordinate fit "
      "changes only at the 0.01 um level. Linear family ('both polynomial' in the linear (a, b) form): "
      "a(A) = 1 + sum_{j<=P} a_j x^j, b(A) = c0 + sum_{j<=Q} b_j x^j with base fixed at k0 = c0 (s = a, "
      "k = b/a^3); with P = Q this is identical to independent degree-P polynomials for r_e(A) and r_c(A). "
      "A Chebyshev basis spans the same polynomial space as the monomials, so it changes conditioning only, "
      "not the least-squares fit or CV (x in [0,1] is already well conditioned; not run separately). "
      "Current model: 3 k constants + 20 measured s values (= 23 params); its LOO refits the power law on "
      "direct k1 without A_i and takes the held-out s from refit_barrel's quadratic-through-1 law "
      "(alternative: cubic spline of the other 20 s values).")
    w()
    w("CV: leave one accommodation out, refit, predict it. 'CV interior' pools A = 0.2..3.8 (19 states); A = 4 is "
      "extrapolation; A = 0 is exact by construction for every model (trivially 0) and is not reported. "
      "In-sample pooled metrics are over all 21 A (A = 0 contributes exact zeros); 'A>0' excludes it.")
    w()
    w(f"Data floor: degree-8 polynomials in A fitted to r_e(A), r_c(A) leave a vertex RMS of {jit_vrms:.4f} um "
      f"(max |radius residual| {np.max(np.abs(jit)):.4f} um, largest at A = {A[np.argmax(np.max(np.abs(jit), 0))]:.1f} D). "
      "This non-smooth component (ray-trace / sampling jitter at the ~10 nm level, visible e.g. at 0.2 and 0.8 D) "
      "cannot be fit by any smooth model, so differences between models below ~0.01 um are not meaningful.")
    w()
    w("| model | params | k(0) (/mm^2) | in-sample vertex RMS (21 A) | in-sample coord RMSE | in-sample max pt | vertex RMS A>0 | CV interior vertex RMS | CV interior coord RMSE | CV interior max pt | A=4 extrap vertex RMS | A=4 extrap max pt |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    w(f"| current (k pow, s measured; LOO s quad-law) | 23 | {C0} | {cur_res['pool'][0]:.4f} | {cur_res['pool'][1]:.4f} | "
      f"{cur_res['pool'][2]:.4f} | {cur_res['pool_pos'][0]:.4f} | {cur_res['cv_int'][0]:.4f} | {cur_res['cv_int'][1]:.4f} | "
      f"{cur_res['cv_int'][2]:.4f} | {cur_res['cv_A4'][0]:.4f} | {cur_res['cv_A4'][2]:.4f} |")
    w(f"| current (LOO s spline) | 23 | {C0} | - | - | - | - | {cur_res['cv_int_sp'][0]:.4f} | {cur_res['cv_int_sp'][1]:.4f} | "
      f"{cur_res['cv_int_sp'][2]:.4f} | {cur_res['cv_A4_sp'][0]:.4f} | {cur_res['cv_A4_sp'][2]:.4f} |")
    w(f"| per-A free (s_A, k_A) ceiling | 40 | {C0} | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a |")
    for j in order:
        R = results[j]
        k0v = float(R["model"].k(R["th"], 0.0)) if isinstance(R["model"], Model) else R["model"].k0
        w(f"| {R['model'].name} | {R['model'].npar} | {k0v:.6g} | {R['pool'][0]:.4f} | {R['pool'][1]:.4f} | {R['pool'][2]:.4f} | "
          f"{R['pool_pos'][0]:.4f} | {R['cv_int'][0]:.4f} | {R['cv_int'][1]:.4f} | {R['cv_int'][2]:.4f} | "
          f"{R['cv_A4'][0]:.4f} | {R['cv_A4'][2]:.4f} |")
    w()
    w(f"Best by interior CV vertex RMS: **{best['model'].name}** ({best['model'].npar} params, CV "
      f"{best['cv_int'][0]:.4f} um). Fewest-parameter model within max(10 %, 0.005 um) of it: "
      f"**{pars['model'].name}** ({pars['model'].npar} params, CV {pars['cv_int'][0]:.4f} um). Within the "
      f"physical-gauge nonlinear family [k0=c0]: best {best_nl['model'].name} (CV {best_nl['cv_int'][0]:.4f} um), "
      f"parsimonious pick (fewest params within max(10 %, 0.005 um)) **{pars_nl['model'].name}** "
      f"({pars_nl['model'].npar} params, CV {pars_nl['cv_int'][0]:.4f} um, in-sample max point "
      f"{pars_nl['pool'][2]:.4f} um) -- this is the recommended model. Best [k0 free]: {best_free['model'].name} "
      f"(CV {best_free['cv_int'][0]:.4f} um, k(0) = {float(best_free['model'].k(best_free['th'], 0.0)):.6g}).")
    w()
    for R, tag in ((USE, "recommended"), (best, "best-CV")):
        w(f"Per-accommodation errors, {tag} model {R['model'].name} (vertex RMS / coord RMSE / max point, um; "
          "in-sample and LOO):")
        w()
        w("| A | in vRMS | in cRMSE | in max | LOO vRMS | LOO cRMSE | LOO max | current vRMS | current max |")
        w("|---|---|---|---|---|---|---|---|---|")
        for i in range(21):
            a, b = R["mA"][i], R["loo_mA"][i]
            w(f"| {A[i]:.1f} | {a[0]:.4f} | {a[1]:.4f} | {a[2]:.4f} | {b[0]:.4f} | {b[1]:.4f} | {b[2]:.4f} | "
              f"{m_cur[i, 0]:.4f} | {m_cur[i, 2]:.4f} |")
        w()

    # recommended formula
    def formula(R):
        M, th = R["model"], R["th"]
        lines = []
        if isinstance(M, Model):
            sa = ", ".join(f"a{j} = {th[j - 1]:.10e}" for j in range(1, M.p + 1))
            sA = ", ".join(f"{th[j - 1] / XS ** j:.10e}" for j in range(1, M.p + 1))
            lines.append(f"s(A) = 1 + sum_j a_j (A/4)^j, {sa}  (per-D^j coefficients: {sA})")
            if M.ktype == "pow":
                c0, c1, n = M.kfull(th)
                lines.append(f"k(A) = c0 + c1 A^n, c0 = {c0:.10e} /mm^2, c1 = {c1:.10e}, n = {n:.8f}")
            else:
                kb = M.kfull(th)
                lines.append("k(A) = sum_j b_j (A/4)^j, " + ", ".join(f"b{j} = {kb[j]:.10e}" for j in range(M.q + 1)) +
                             "  (per-D^j: " + ", ".join(f"{kb[j] / XS ** j:.10e}" for j in range(M.q + 1)) + ")")
            lines.append("u0 = D_{k(0)}^-1(r0) (radial Newton inverse), real(A) = (1 + k(A) s(A)^2 |u0|^2) s(A) u0")
        else:
            P, Q = M.P, M.Q
            lines.append("u0 = D_c0^-1(r0) with c0 = %.10f; real(A) = a(A) u0 + b(A) |u0|^2 u0" % M.k0)
            lines.append("a(A) = 1 + " + " + ".join(f"({th[j - 1]:.10e}) (A/4)^{j}" for j in range(1, P + 1)))
            lines.append(f"b(A) = {M.k0:.10f} + " + " + ".join(f"({th[P + j - 1]:.10e}) (A/4)^{j}" for j in range(1, Q + 1)))
        return lines

    w("### Model formulas")
    w()
    for R, tag in ((USE, "Recommended (parsimonious, k0 = c0)"), (best_nl, "Best-CV with k0 = c0"),
                   (best, "Best-CV overall"), (pars, "Parsimonious overall")):
        w(f"**{tag}: {R['model'].name}**")
        w()
        w("```")
        for ln in formula(R):
            w(ln)
        w("```")
        w()
    w("## 4. On-axis accommodation estimation")
    w()
    w("A is estimated from the 18 observed coordinates by minimising the coordinate SSE over A in [0, 4] "
      "(vectorised scan, step 0.0005 D, parabolic refinement, 6 Gauss-Newton steps, clipped to [0, 4]). "
      "The current model needs s at non-grid A: for in-sample estimation a cubic spline through the 21 "
      "measured s; for LOO a cubic spline through the other 20 s values (continuous in A; the piecewise quad-law rule is discontinuous between grid points).")
    w()
    w("Noise-free error A_hat - A (mD):")
    w()
    names = list(est.keys())
    lnames = list(loo_est.keys())
    w("| A | " + " | ".join(f"in-sample {n}" for n in names) + " | " + " | ".join(f"LOO {n}" for n in lnames) + " |")
    w("|---" * (1 + len(names) + len(lnames)) + "|")
    for i in range(21):
        w(f"| {A[i]:.1f} | " + " | ".join(f"{(est[n]['ahat'][i] - A[i]) * 1e3:+.3f}" for n in names) + " | " +
          " | ".join(f"{(loo_est[n][i] - A[i]) * 1e3:+.3f}" for n in lnames) + " |")
    w("| RMS (all 21) | " + " | ".join(f"{np.sqrt(np.mean((est[n]['ahat'] - A) ** 2)) * 1e3:.3f}" for n in names) +
      " | " + " | ".join(f"{np.sqrt(np.mean((loo_est[n] - A) ** 2)) * 1e3:.3f}" for n in lnames) + " |")
    w("| RMS interior (0.2..3.8) | " + " | ".join(f"{np.sqrt(np.mean((est[n]['ahat'][1:20] - A[1:20]) ** 2)) * 1e3:.3f}" for n in names) +
      " | " + " | ".join(f"{np.sqrt(np.mean((loo_est[n][1:20] - A[1:20]) ** 2)) * 1e3:.3f}" for n in lnames) + " |")
    w()
    w(f"With 1 um iid noise per coordinate: Cramer-Rao sigma_A = sigma / sqrt(sum (dX/dA)^2) (model Jacobian) "
      f"and Monte Carlo RMS of A_hat - A ({MC_DRAWS} draws per A, seed {RNG_SEED}, same draws for all fields "
      "subsets). The estimate is clipped to [0, 4], which lowers the MC RMS and biases the mean at the "
      "endpoints (A = 0 and 4 D), so MC < CRB there is expected. Model-free CRB from data: sigma / "
      "sqrt(4 (dr_e/dA)^2 + 4 (dr_c/dA)^2) with np.gradient of r_e, r_c.")
    w()
    for n in names:
        E = est[n]
        w(f"**{n}** (mD):")
        w()
        w("| A | CRB all 9 | CRB edges only | CRB corners only | data CRB all | MC RMS all 9 | MC bias all 9 | MC RMS edges | MC RMS corners |")
        w("|---|---|---|---|---|---|---|---|---|")
        for i in range(21):
            w(f"| {A[i]:.1f} | {E['sig_all'][i] * 1e3:.2f} | {E['sig_e'][i] * 1e3:.2f} | {E['sig_c'][i] * 1e3:.2f} | "
              f"{sig_data_all[i] * 1e3:.2f} | {E['mc_rms'][i] * 1e3:.2f} | {E['mc_bias'][i] * 1e3:+.2f} | "
              f"{E['mc_rms_e'][i] * 1e3:.2f} | {E['mc_rms_c'][i] * 1e3:.2f} |")
        w(f"| pooled RMS | {np.sqrt(np.mean(E['sig_all'] ** 2)) * 1e3:.2f} | {np.sqrt(np.mean(E['sig_e'] ** 2)) * 1e3:.2f} | "
          f"{np.sqrt(np.mean(E['sig_c'] ** 2)) * 1e3:.2f} | {np.sqrt(np.mean(sig_data_all ** 2)) * 1e3:.2f} | "
          f"{np.sqrt(np.mean(E['mc_rms'] ** 2)) * 1e3:.2f} | | {np.sqrt(np.mean(E['mc_rms_e'] ** 2)) * 1e3:.2f} | "
          f"{np.sqrt(np.mean(E['mc_rms_c'] ** 2)) * 1e3:.2f} |")
        w()
    Ju = est[names[-1]]["J"]
    w("Information content: the center carries none (0 by construction). Each edge point contributes one "
      "nonzero coordinate (dr_e/dA), each corner two (dr_c/dA / sqrt2 each), so the Fisher information is "
      "(4 (dr_e/dA)^2 + 4 (dr_c/dA)^2) / sigma^2: the four corners carry "
      f"{100 * np.mean(drc ** 2 / (dre ** 2 + drc ** 2)):.1f} % of the information on average (|dr_c/dA| "
      f"{np.mean(np.abs(drc)) * 1e3:.2f} vs |dr_e/dA| {np.mean(np.abs(dre)) * 1e3:.2f} um/D, mean over A). "
      f"Model Jacobian max |dX/dA| at the center: {np.max(np.abs(Ju[:, 4])) * 1e3:.1e} um/D.")
    w()
    # ---------------- task 5
    w("## 5. Interpretation")
    w()
    chi_cur = np.sum((err_cur * 1e3) ** 2, (1, 2))
    chi_use = np.sum((USE["err"] * 1e3) ** 2, (1, 2))
    bias_cur = est[names[0]]["ahat"] - A
    sig = est[names[-1]]["sig_all"]
    w(f"- Significance vs 1 um noise: the current model's on-axis error (max point {np.max(m_cur[:, 2]):.3f} um, "
      f"4 D vertex RMS {m_cur[-1, 0]:.3f} um) is below the 1 um per-coordinate noise. Its chi^2 = sum e^2 / "
      f"sigma^2 over 18 coordinates is at most {np.max(chi_cur):.2f} (at {A[np.argmax(chi_cur)]:.1f} D) against "
      f"an expected 17-18 for pure noise, so a single on-axis frame cannot detect it. As A bias it is "
      f"{np.max(np.abs(bias_cur)) * 1e3:.2f} mD max (noise-free, in-sample), vs CRB sigma_A ~ "
      f"{np.mean(sig) * 1e3:.1f} mD per frame: about {np.max(np.abs(bias_cur)) / np.mean(sig):.2f} sigma, i.e. "
      f"measurable only after averaging ~{(np.mean(sig) / max(np.max(np.abs(bias_cur)), 1e-12)) ** 2 * 4:.0f}+ "
      "frames (for a 2-sigma bias detection). The recommended model reduces chi^2 to at most "
      f"{np.max(chi_use):.2e}, i.e. negligible.")
    w(f"- s(A) linearity: measured paraxial s(A) deviates from its best free straight line by at most "
      f"{np.max(np.abs(np.polyval(ps1, A) - s_meas)) * 1e6:.1f} ppm = {lin_res_um:.3f} um at the edge radius "
      f"({lin_res_um_c:.3f} um at the corner radius). Polynomials through s(0) = 1 fitted to the per-A values "
      "(1 ppm of scale = 0.0013 um at the edge, 0.0019 um at the corner):")
    w()
    w("| series | degree | RMS residual (ppm) | max residual (ppm) | at A (D) | coefficients a_j of (A/4)^j |")
    w("|---|---|---|---|---|---|")
    for nm_, dg, rms_, mx_, am_, cd in s_poly_res:
        w(f"| {nm_} | {dg} | {rms_:.2f} | {mx_:.2f} | {am_:.1f} | " + ", ".join(f"{c:.6e}" for c in cd) + " |")
    w()
    w("  Note: s_A and k_A trade off against each other (and against the k0 gauge), so the linearity of s alone "
      "is not decisive; the coordinate-level table in section 3 (p = 1 vs p >= 2 at equal k order) is the "
      "relevant test.")
    w(f"- Measured paraxial s quadratic fit: s = {ps2[2]:.10f} + {ps2[1]:.10e} A + {ps2[0]:.10e} A^2.")
    w("- Structural conclusion: on-axis, D_k(A)(s(A) D_k(0)^-1(r0)) is exact per A with 2 free numbers, so "
      "the only modelling question is the smoothness of (s, k) in A; fitting them to coordinates rather than "
      "to paraxial quantities removes the 0.6 um error. The 9-point on-axis grid has only 2 independent "
      "radii, so these fits cannot validate the radial profile between/above r_e and r_c.")
    w()
    w(f"Runtime {time.time() - t0:.1f} s.")
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    np.savez(OUT / "p4_onaxis_results.npz", A=A, real_mm=real, para_mm=para, r_edge_mm=re, r_corner_mm=rc,
             p_edge_mm=pe, dre_dA=dre, drc_dA=drc, k_direct=k_direct, fit_group_rmse_mm=fg_rmse, s_meas=s_meas,
             s_fit_c0base=s_fit, k_fit_c0base=k_fit, s_fit_kd0base=s_fit_d, k_fit_kd0base=k_fit_d,
             current_err_mm=err_cur, current_metrics=m_cur, exact_k_err_mm=err_exact,
             model_names=np.array([R["model"].name for R in results]),
             model_npar=np.array([R["model"].npar for R in results]),
             model_cv_interior=np.array([R["cv_int"] for R in results]),
             model_cv_A4=np.array([R["cv_A4"] for R in results]),
             model_insample=np.array([R["pool"] for R in results]),
             recommended_name=USE["model"].name, recommended_theta=USE["th"],
             recommended_err_mm=USE["err"], recommended_loo_err_mm=USE["loo_err"],
             best_cv_name=best["model"].name, best_cv_theta=best["th"],
             **{f"est_{i}_ahat": est[n]["ahat"] for i, n in enumerate(names)},
             **{f"est_{i}_crb_all": est[n]["sig_all"] for i, n in enumerate(names)},
             **{f"est_{i}_crb_edge": est[n]["sig_e"] for i, n in enumerate(names)},
             **{f"est_{i}_crb_corner": est[n]["sig_c"] for i, n in enumerate(names)},
             **{f"est_{i}_mc_rms": est[n]["mc_rms"] for i, n in enumerate(names)},
             est_names=np.array(names),
             **{f"loo_est_{i}": loo_est[n] for i, n in enumerate(lnames)}, loo_est_names=np.array(lnames))
    print("\n".join(L))


if __name__ == "__main__":
    main()
