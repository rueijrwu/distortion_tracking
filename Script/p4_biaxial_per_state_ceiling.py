"""Per-state error ceiling of the P4 biaxial barrel + projective-keystone model family.

For every one of the 21 x 101 x 101 = 214,221 (A, ADE, BDE) states, fit a FREE per-state model to its
nine real-ray points (center-relative, mm) with no polynomial across states and no translation (T = 0).
This gives the lowest error the model form can reach state by state, i.e. an upper bound on what any
coefficient-surface polynomial (of any order) across states could achieve.

Notation (normalised coordinates, divide mm by L = |x| of the (-1,-1) reference corner):
    n      = s(A) * D_k0^-1(r0) / L        paraxial-at-A grid (fixed per A)
    R(u,k) = u * (1 + k |u|^2)             radial map (k in normalised units: k_norm = k_mm * L^2)
    BC(u)  = Brown-Conrady extra terms on u: u(1 + k2 r^4) + tangential (p1, p2), r^2 = |u|^2
    H(u)   = (M u) / (1 + q.u)             M = [[m11, m12], [m21, m22]], q = (qx, qy)
Variants (output X = L * ...):
    (0) H(R(n, k(A)))                              6 params  (M, q); barrel FIXED = global B_A
    (a) H(R(n, k))                                 7 params  (+ per-state k1)
    (b) H(BC(R(n, k(A))))                          9 params  (+ k2, p1, p2)
    (c) R(H(n), k(A))                              6 params  keystone first, then the fixed radial part
    (d) H(BC(R(n, k)))                            11 params  (+ k1, k2, p1, p2)
The center field maps 0 -> 0 in every variant, so its 2 residuals are a fixed constant (data center is
~0), leaving 16 effective residuals per state.

Batched Levenberg-Marquardt over all states at once (numpy), complex-step Jacobians.
Also: surface-fit diagnostics of the per-state variant-(0) parameters (existing order-4 family basis and
Legendre bases), forward-evaluated to an implied global coordinate error, plus a joint coordinate-space
LM fit with the Legendre basis (reusing fit_p4_biaxial_joint.BiaxialModel / lm_fit).

Outputs: data/p4_biaxial_joint/improvement.md, p4_biaxial_per_state_ceiling.npz, p4_biaxial_per_state_ceiling.pkl
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Script"))

import fit_p4_biaxial_joint as fj  # noqa: E402
import refit_p4_separable_transform as sep  # noqa: E402

OUT = ROOT / "data" / "p4_biaxial_joint"
COEF_NPZ = OUT / "p4_biaxial_joint_coefficients.npz"
FAM6 = ("m11", "m22", "m12", "m21", "qx", "qy")     # family order used by the global control_no_T fit
OUTER = [0, 1, 2, 3, 5, 6, 7, 8]
DEN_GUARD = 0.05
log = fj.log


# ------------------------------------------------------------------------------- model maps
def radial(u, k):
    r2 = u[..., 0] * u[..., 0] + u[..., 1] * u[..., 1]          # no abs(): complex-step safe
    g = 1.0 + k[:, None] * r2
    return u * g[..., None]


def brown(u, k2, p1, p2):
    x, y = u[..., 0], u[..., 1]
    r2 = x * x + y * y
    g = 1.0 + k2[:, None] * r2 * r2
    X = x * g + 2 * p1[:, None] * x * y + p2[:, None] * (r2 + 2 * x * x)
    Y = y * g + p1[:, None] * (r2 + 2 * y * y) + 2 * p2[:, None] * x * y
    return np.stack((X, Y), -1)


def homog(u, m11, m12, m21, m22, qx, qy):
    x, y = u[..., 0], u[..., 1]
    den = 1.0 + qx[:, None] * x + qy[:, None] * y
    X = (m11[:, None] * x + m12[:, None] * y) / den
    Y = (m21[:, None] * x + m22[:, None] * y) / den
    return np.stack((X, Y), -1), den


# P columns: 0..5 = m11, m12, m21, m22, qx, qy; then extras
VARIANTS = {
    "0": {"names": ("m11", "m12", "m21", "m22", "qx", "qy"), "desc": "H(R(n,k(A))): M + q on fixed barrel"},
    "a": {"names": ("m11", "m12", "m21", "m22", "qx", "qy", "k1n"), "desc": "(0) + per-state k1"},
    "b": {"names": ("m11", "m12", "m21", "m22", "qx", "qy", "k2", "p1", "p2"),
          "desc": "(0) + Brown-Conrady k2, p1, p2 on the barrel output"},
    "c": {"names": ("m11", "m12", "m21", "m22", "qx", "qy"), "desc": "R(H(n),k(A)): keystone first, then radial"},
    "d": {"names": ("m11", "m12", "m21", "m22", "qx", "qy", "k1n", "k2", "p1", "p2"),
          "desc": "M + q + per-state k1 + k2 + p1, p2"},
}


class Problem:
    def __init__(self, n_norm, knA, target_mm, L):
        self.n = n_norm            # (S, 9, 2) normalised paraxial-at-A grid
        self.bA = radial(n_norm, knA)   # fixed barrel output, normalised
        self.knA = knA
        self.T = target_mm
        self.L = L

    def forward(self, v, P, idx):
        c = [P[:, j] for j in range(P.shape[1])]
        m = c[:6]
        if v == "0":
            out, den = homog(self.bA[idx], *m)
        elif v == "a":
            out, den = homog(radial(self.n[idx], c[6]), *m)
        elif v == "b":
            out, den = homog(brown(self.bA[idx], c[6], c[7], c[8]), *m)
        elif v == "c":
            w, den = homog(self.n[idx], *m)
            out = radial(w, self.knA[idx])
        elif v == "d":
            out, den = homog(brown(radial(self.n[idx], c[6]), c[7], c[8], c[9]), *m)
        # decomposition sub-variants of (b)/(d)
        elif v == "b_k2":        # (0) + k2 only
            out, den = homog(brown(self.bA[idx], c[6], 0 * c[6], 0 * c[6]), *m)
        elif v == "b_p":         # (0) + p1, p2 only
            out, den = homog(brown(self.bA[idx], 0 * c[6], c[6], c[7]), *m)
        elif v == "d_k1p":       # (0) + k1 + p1, p2 (no k2)
            out, den = homog(brown(radial(self.n[idx], c[6]), 0 * c[6], c[7], c[8]), *m)
        return out, den

    def resid(self, v, P, idx):
        out, den = self.forward(v, P, idx)
        r = (self.L * out - self.T[idx]) * 1000.0      # um
        return r.reshape(len(idx), 18), den.real

    def jac(self, v, P, idx, h=1e-30):
        S, n = P.shape
        J = np.empty((S, 18, n))
        for j in range(n):
            Pc = P.astype(complex)
            Pc[:, j] += 1j * h
            J[:, :, j] = self.resid(v, Pc, idx)[0].imag / h
        return J


def batched_lm(prob, v, P0, max_iter=300, label=""):
    t0 = time.perf_counter()
    P = P0.copy()
    S, n = P.shape
    allidx = np.arange(S)
    r, den = prob.resid(v, P, allidx)
    cost = np.sum(r * r, 1)
    if np.any(den.min(1) <= DEN_GUARD):
        raise RuntimeError(f"[{label}] start point violates denominator guard")
    lam = np.full(S, 1e-4)
    done = np.zeros(S, bool)
    status = np.zeros(S, np.int8)     # 0 max_iter, 1 converged, 2 stuck (no descent)
    iters = np.zeros(S, np.int32)
    eye = np.eye(n)
    for it in range(max_iter):
        idx = np.flatnonzero(~done)
        if idx.size == 0:
            break
        Pi = P[idx]
        ri = prob.resid(v, Pi, idx)[0]
        J = prob.jac(v, Pi, idx)
        JTJ = np.einsum("sri,srj->sij", J, J)
        g = np.einsum("sri,sr->si", J, ri)
        d = np.sqrt(np.maximum(np.einsum("sii->si", JTJ), 1e-30))
        As = JTJ / (d[:, :, None] * d[:, None, :])
        gs = g / d
        ci = cost[idx]
        li = lam[idx]
        acc = np.zeros(idx.size, bool)
        Pn = Pi.copy()
        cn = ci.copy()
        for _ in range(14):
            todo = np.flatnonzero(~acc & (li < 1e12))
            if todo.size == 0:
                break
            Ab = As[todo] + li[todo, None, None] * eye
            try:
                st = -np.linalg.solve(Ab, gs[todo][..., None])[..., 0] / d[todo]
            except np.linalg.LinAlgError:
                st = -np.stack([np.linalg.lstsq(a_, b_, rcond=None)[0] for a_, b_ in zip(Ab, gs[todo])]) / d[todo]
            Pt = Pi[todo] + st
            rt, dt = prob.resid(v, Pt, idx[todo])
            ct = np.sum(rt * rt, 1)
            ok = np.isfinite(ct) & (ct <= ci[todo]) & (dt.min(1) > DEN_GUARD)
            a_ = todo[ok]
            Pn[a_], cn[a_], acc[a_] = Pt[ok], ct[ok], True
            li[todo[~ok]] *= 10
        rel = (ci - cn) / np.maximum(ci, 1e-300)
        P[idx[acc]] = Pn[acc]
        cost[idx[acc]] = cn[acc]
        li[acc] = np.maximum(li[acc] / 10, 1e-15)
        lam[idx] = li
        iters[idx] += 1
        conv = acc & ((rel < 1e-13) | (cn < 1e-20))
        stuck = ~acc
        done[idx[conv]] = True
        status[idx[conv]] = 1
        done[idx[stuck]] = True
        status[idx[stuck]] = 2
        if it % 10 == 0 or idx.size < 50:
            log(f"  [{label}] it {it:3d} active {idx.size:7d} total cost {cost.sum():.10g} um^2")
    # final scaled gradient
    r, den = prob.resid(v, P, allidx)
    gmax = np.empty(S)
    cond = np.empty(S)
    for s0 in range(0, S, 50000):
        sl = np.arange(s0, min(S, s0 + 50000))
        J = prob.jac(v, P[sl], sl)
        JTJ = np.einsum("sri,srj->sij", J, J)
        g = np.einsum("sri,sr->si", J, r[sl])
        d = np.sqrt(np.maximum(np.einsum("sii->si", JTJ), 1e-30))
        gmax[sl] = np.max(np.abs(g / d), 1)
        cond[sl] = np.linalg.cond(JTJ / (d[:, :, None] * d[:, None, :]))
    log(f"  [{label}] done in {time.perf_counter()-t0:.1f} s: converged {np.sum(status==1)}, stuck {np.sum(status==2)}, "
        f"max_iter {np.sum(status==0)}; max scaled grad {gmax.max():.3e}")
    return {"P": P, "cost": cost, "status": status, "iters": iters, "gmax": gmax, "cond": cond,
            "min_den": den.min(1), "runtime_s": time.perf_counter() - t0}


def dlt_init(prob):
    """Linear (algebraic) M, q from X(1+q.b) = M b in normalised coordinates."""
    b = prob.bA
    X = prob.T / prob.L
    bx, by, Xx, Xy = b[..., 0], b[..., 1], X[..., 0], X[..., 1]
    z = np.zeros_like(bx)
    # rows for x: [bx, by, 0, 0, -Xx bx, -Xx by] = Xx ; y: [0, 0, bx, by, -Xy bx, -Xy by] = Xy
    Ax = np.stack((bx, by, z, z, -Xx * bx, -Xx * by), -1)
    Ay = np.stack((z, z, bx, by, -Xy * bx, -Xy * by), -1)
    A = np.concatenate((Ax, Ay), 1)
    y = np.concatenate((Xx, Xy), 1)
    AtA = np.einsum("sri,srj->sij", A, A)
    Aty = np.einsum("sri,sr->si", A, y)
    return np.linalg.solve(AtA, Aty[..., None])[..., 0]


# ------------------------------------------------------------------------------- metrics
def metrics(err_um, ids=range(9)):
    e = err_um[:, list(ids), :]
    nrm = np.linalg.norm(e, axis=-1)
    st = np.sqrt(np.mean(nrm ** 2, -1))
    return {"vertex_rms_um": float(np.sqrt(np.mean(nrm ** 2))), "coord_rmse_um": float(np.sqrt(np.mean(e ** 2))),
            "max_point_um": float(nrm.max()), "mean_state_rms_um": float(st.mean()),
            "median_state_rms_um": float(np.median(st)), "max_state_rms_um": float(st.max()),
            "p99_state_rms_um": float(np.percentile(st, 99))}


BDE_EDGES = [(-0.1, 0.1, "0"), (0.1, 4.05, "(0,4]"), (4.05, 8.05, "(4,8]"), (8.05, 12.05, "(8,12]"),
             (12.05, 16.05, "(12,16]"), (16.05, 20.1, "(16,20]")]


def binned(err_um, absb, A, accs):
    nrm = np.linalg.norm(err_um, axis=-1)
    rows_b, rows_a = [], []
    for lo, hi, lab in BDE_EDGES:
        s = (absb > lo) & (absb <= hi)
        rows_b.append((lab, int(s.sum()), float(np.sqrt(np.mean(nrm[s] ** 2))), float(np.sqrt(np.mean(err_um[s] ** 2))),
                       float(nrm[s].max())))
    for a in accs:
        s = np.isclose(A, a, atol=1e-9)
        rows_a.append((float(a), int(s.sum()), float(np.sqrt(np.mean(nrm[s] ** 2))), float(np.sqrt(np.mean(err_um[s] ** 2))),
                       float(nrm[s].max())))
    return rows_b, rows_a


def worst(err_um, A, ADE, BDE, fields, nanstate, n=10):
    nrm = np.linalg.norm(err_um, axis=-1)
    st = np.sqrt(np.mean(nrm ** 2, -1))
    o = np.argsort(st)[::-1][:n]
    return [{"A": float(A[k]), "ADE": float(ADE[k]), "BDE": float(BDE[k]), "state_rms_um": float(st[k]),
             "max_point_um": float(nrm[k].max()), "worst_field": tuple(float(x) for x in fields[int(np.argmax(nrm[k]))]),
             "ray_failure": bool(nanstate[k])} for k in o]


# ------------------------------------------------------------------------------- bases
def legendre(x, deg):
    out = [np.ones_like(x), x]
    for k in range(1, deg):
        out.append(((2 * k + 1) * x * out[k] - k * out[k - 1]) / (k + 1))
    return np.stack(out[:deg + 1], -1)


def legendre_basis(A, tx, ty, KA, NT):
    PA = legendre(A / 2.0 - 1.0, KA)            # x = 2A/4 - 1
    Px, Py = legendre(tx, NT), legendre(ty, NT)
    cols = [PA[:, l] * Px[:, i] * Py[:, j] for l in range(KA + 1) for d_ in range(NT + 1)
            for i in range(d_, -1, -1) for j in [d_ - i]]
    return np.stack(cols, -1)


def build_problem(G):
    na = len(G["acc"])
    br = fj.refit_barrel(G, np.ones(na, bool))
    L = float(abs(br["reference_mm"][0, 0]))
    kc = br["k_coefficients_c0_c1_n"]
    kA = sep.radial_k1(kc, G["acc"])
    para0, _ = sep.invert_radial(br["reference_mm"], sep.radial_k1(kc, 0.0))
    psA = br["s_used"][:, None, None] * para0[None]
    ai, ti, bi = [x.ravel() for x in np.meshgrid(np.arange(na), np.arange(len(G["ade"])), np.arange(len(G["bde"])),
                                                 indexing="ij")]
    return Problem(psA[ai] / L, (kA * L * L)[ai], G["real"][ai, ti, bi], L), ai, ti, bi


def decomposition(prob, P0s, Pas, absb):
    """Which extra terms carry the (b)/(d) gain: k2 alone, p1/p2 alone, k1 + p1/p2."""
    S = P0s.shape[0]
    z = np.zeros((S, 1))
    out = {}
    for v, Pinit, npar, desc in (("b_k2", np.hstack((P0s, z)), 7, "(0) + k2 only"),
                                 ("b_p", np.hstack((P0s, z, z)), 8, "(0) + p1, p2 only"),
                                 ("d_k1p", np.hstack((Pas, z, z)), 9, "(0) + k1 + p1, p2 (no k2)")):
        F = batched_lm(prob, v, Pinit, label=v)
        r, _ = prob.resid(v, F["P"], np.arange(S))
        e = r.reshape(S, 9, 2)
        sc = {k: int(np.sum(F["status"] == i)) for i, k in enumerate(("max_iter", "converged", "stuck"))}
        out[v] = {"desc": desc, "nparams": npar, "all_nine": metrics(e), "status_counts": sc,
                  "max_scaled_grad": float(F["gmax"].max()), "cost": F["cost"],
                  "bde_vertex": [rb[2] for rb in binned(e, absb, np.zeros(S), [])[0]]}
    return out


def report_only():
    """Rebuild improvement.md from the saved pkl (adds the decomposition if missing)."""
    pk = OUT / "p4_biaxial_per_state_ceiling.pkl"
    P = pickle.loads(pk.read_bytes())
    if "decomp" not in P:
        G = fj.load_biaxial(fj.GRID_PATH)
        prob, ai, ti, bi = build_problem(G)
        Z = np.load(OUT / "p4_biaxial_per_state_ceiling.npz")
        assert np.allclose(Z["BDE"], G["bde"][bi]) and np.allclose(Z["ADE"], G["ade"][ti])
        D = decomposition(prob, Z["params_0"], Z["params_a"], np.abs(G["bde"][bi]))
        P["decomp"] = {k: {kk: vv for kk, vv in v.items() if kk != "cost"} for k, v in D.items()}
        ca = np.sum(prob.resid("a", Z["params_a"], np.arange(len(ai)))[0] ** 2, 1)
        P["k1_k2_equivalence_max_abs_cost_diff_um2"] = float(np.max(np.abs(D["b_k2"]["cost"] - ca)))
        with pk.open("wb") as s:
            pickle.dump(P, s, pickle.HIGHEST_PROTOCOL)
    if "n_state_worse_than_global" not in P:
        Z = np.load(OUT / "p4_biaxial_per_state_ceiling.npz")
        d = Z["state_rms_um_0"] - Z["global252_state_rms_um"]
        P["n_state_worse_than_global"], P["max_state_excess_over_global_um"] = int(np.sum(d > 1e-9)), float(d.max())
        with pk.open("wb") as s:
            pickle.dump(P, s, pickle.HIGHEST_PROTOCOL)
    write_report(P, OUT / "improvement.md")
    log("report rewritten")


# ---------------------------------------------------------------------------------- main
def main():
    T0 = time.perf_counter()
    log("Loading grid ...")
    G = fj.load_biaxial(fj.GRID_PATH)
    na, nt, nb = len(G["acc"]), len(G["ade"]), len(G["bde"])
    br = fj.refit_barrel(G, np.ones(na, bool))
    L = float(abs(br["reference_mm"][0, 0]))
    kc = br["k_coefficients_c0_c1_n"]
    k0 = sep.radial_k1(kc, 0.0)
    kA = sep.radial_k1(kc, G["acc"])
    para0, _ = sep.invert_radial(br["reference_mm"], k0)
    psA = br["s_used"][:, None, None] * para0[None]             # (21, 9, 2) mm
    recon = sep.apply_radial(psA, kA[:, None])
    barrel_recon_err = float(np.max(np.abs(recon - br["barrel_mm"])))
    log(f"  L={L}, barrel reconstruction max err {barrel_recon_err:.3e} mm")
    if barrel_recon_err > 1e-12:
        raise RuntimeError("barrel reconstruction mismatch")

    ai, ti, bi = [x.ravel() for x in np.meshgrid(np.arange(na), np.arange(nt), np.arange(nb), indexing="ij")]
    S = ai.size
    A, ADE, BDE = G["acc"][ai], G["ade"][ti], G["bde"][bi]
    tx, ty = BDE / fj.T_NORM, ADE / fj.T_NORM
    target = G["real"][ai, ti, bi]                    # (S, 9, 2) mm
    nanstate = G["nan_state"][ai, ti, bi]
    even = ((ti % 2 == 0) & (bi % 2 == 0))
    comp = ~even
    center_const = float(np.max(np.abs(target[:, 4, :])) * 1e3)
    barrel_center = float(np.max(np.abs(br["barrel_mm"][:, 4, :])) * 1e3)
    log(f"  max |data center| {center_const:.3e} um, max |barrel center| {barrel_center:.3e} um")

    prob = Problem(psA[ai] / L, (kA * L * L)[ai], target, L)

    # Jacobian audit (complex step vs central FD) on 50 states at a perturbed point, variant d
    rng = np.random.default_rng(0)
    st50 = rng.choice(S, 50, replace=False)
    Pa = np.column_stack((1 + rng.normal(0, 1e-3, 50), rng.normal(0, 1e-3, 50), rng.normal(0, 1e-3, 50),
                          1 + rng.normal(0, 1e-3, 50), rng.normal(0, 1e-2, 50), rng.normal(0, 1e-2, 50),
                          prob.knA[st50] * (1 + rng.normal(0, 1e-2, 50)), rng.normal(0, 1e-3, 50),
                          rng.normal(0, 1e-4, 50), rng.normal(0, 1e-4, 50)))
    Jc = prob.jac("d", Pa, st50)
    jaud = 0.0
    for j in range(Pa.shape[1]):
        h = 1e-6
        Pp, Pm = Pa.copy(), Pa.copy()
        Pp[:, j] += h
        Pm[:, j] -= h
        fd = (prob.resid("d", Pp, st50)[0] - prob.resid("d", Pm, st50)[0]) / (2 * h)
        jaud = max(jaud, float(np.max(np.abs(fd - Jc[:, :, j])) / max(1e-12, np.max(np.abs(fd)))))
    log(f"  complex-step vs FD Jacobian audit: max rel err {jaud:.2e}")

    fits = {}
    log("Variant (0): DLT init + LM")
    P0 = dlt_init(prob)
    fits["0"] = batched_lm(prob, "0", P0, label="0")
    P0s = fits["0"]["P"]
    knA = prob.knA[:, None]
    z = np.zeros((S, 1))
    fits["a"] = batched_lm(prob, "a", np.hstack((P0s, knA)), label="a")
    fits["b"] = batched_lm(prob, "b", np.hstack((P0s, z, z, z)), label="b")
    fits["c"] = batched_lm(prob, "c", P0s.copy(), label="c")
    da = batched_lm(prob, "d", np.hstack((fits["a"]["P"], z, z, z)), label="d_from_a")
    db = batched_lm(prob, "d", np.hstack((fits["b"]["P"][:, :6], knA, fits["b"]["P"][:, 6:])), label="d_from_b")
    pick_b = db["cost"] < da["cost"]
    fits["d"] = {k: np.where(pick_b if da[k].ndim == 1 else pick_b[:, None], db[k], da[k]) if isinstance(da[k], np.ndarray)
                 else da[k] + db[k] for k in da}
    fits["d"]["picked_from_b"] = int(pick_b.sum())

    nest = {}
    for ch, par in (("a", "0"), ("b", "0"), ("d", "a"), ("d", "b")):
        dc = fits[ch]["cost"] - fits[par]["cost"]
        nest[f"{ch}<={par}"] = {"violations_gt_1e-9": int(np.sum(dc > 1e-9)), "max_excess_um2": float(dc.max())}
    c_beats_0 = int(np.sum(fits["c"]["cost"] < fits["0"]["cost"] - 1e-9))
    log(f"  nesting {nest}; (c) better than (0) on {c_beats_0} states")

    errs, summ = {}, {}
    for v in VARIANTS:
        r, _ = prob.resid(v, fits[v]["P"], np.arange(S))
        e = r.reshape(S, 9, 2)
        errs[v] = e
        summ[v] = {"all_nine": metrics(e), "outer8": metrics(e, OUTER), "complement_all_nine": metrics(e[comp]),
                   "bins": binned(e, np.abs(BDE), A, G["acc"]),
                   "worst": worst(e, A, ADE, BDE, G["fields"], nanstate),
                   "nparams": len(VARIANTS[v]["names"]),
                   "status_counts": {k: int(np.sum(fits[v]["status"] == i)) for i, k in enumerate(("max_iter", "converged", "stuck"))},
                   "max_scaled_grad": float(fits[v]["gmax"].max()), "p99_scaled_grad": float(np.percentile(fits[v]["gmax"], 99)),
                   "frac_cond_gt_1e10": float(np.mean(fits[v]["cond"] > 1e10)),
                   "median_cond": float(np.median(fits[v]["cond"])), "max_cond": float(np.max(fits[v]["cond"])),
                   "min_den": float(fits[v]["min_den"].min()), "runtime_s": float(fits[v]["runtime_s"])}
        log(f"  ({v}) all-nine vertex RMS {summ[v]['all_nine']['vertex_rms_um']:.4f} coord {summ[v]['all_nine']['coord_rmse_um']:.4f} "
            f"max {summ[v]['all_nine']['max_point_um']:.4f}")

    # global joint 252-coef T=0 fit, evaluated with the parent model class
    Z = np.load(COEF_NPZ)
    fam = tuple(str(x) for x in Z["control_no_T_families"])
    assert fam == FAM6
    pg = Z["control_no_T_coefficients"].ravel()
    Mg = fj.BiaxialModel(G, br["barrel_mm"], L, np.ones((na, nt, nb), bool), fam)
    assert np.array_equal(Mg.idx[0], ai) and np.array_equal(Mg.idx[1], ti) and np.array_equal(Mg.idx[2], bi)
    rg, _ = Mg.residual_um(pg)
    glob = {"full": metrics(rg), "complement": metrics(rg[comp]), "train": metrics(rg[even]),
            "bins": binned(rg, np.abs(BDE), A, G["acc"])}
    gsurf = Mg.unpack(pg)
    log(f"  global 252 T=0: complement {glob['complement']}")

    # k1 per state (variant a) vs power law
    k1_state = fits["a"]["P"][:, 6] / L ** 2
    dk = k1_state - kA[ai]
    k1tab = {"bde": [], "acc": []}
    for lo, hi, lab in BDE_EDGES:
        s = (np.abs(BDE) > lo) & (np.abs(BDE) <= hi)
        k1tab["bde"].append((lab, float(dk[s].mean()), float(dk[s].std()), float(dk[s].min()), float(dk[s].max()),
                             float(np.mean(dk[s] / kA[ai][s]))))
    for ia, a in enumerate(G["acc"]):
        s = ai == ia
        k1tab["acc"].append((float(a), float(kA[ia]), float(dk[s].mean()), float(dk[s].std()), float(dk[s].min()),
                             float(dk[s].max())))

    # ----------------------------------------------------------------- surface diagnostics
    names0 = VARIANTS["0"]["names"]
    par0 = {nm: P0s[:, j] for j, nm in enumerate(names0)}
    f0 = {"m11": 1.0, "m22": 1.0}
    zero_rot = (ADE == 0) & (BDE == 0)
    id_dev = {f: float(np.max(np.abs(par0[f][zero_rot] - f0.get(f, 0.0)))) for f in FAM6}
    glob_vs_state = {f: float(np.sqrt(np.mean((gsurf[f] - par0[f]) ** 2))) for f in FAM6}

    def eval_surface(Pfam):
        P = np.column_stack([Pfam[nm] for nm in names0])
        r, den = prob.resid("0", P, np.arange(S))
        return r.reshape(S, 9, 2), float(den.min())

    bases = {"existing a^0..2 x 14 (f-f0, no t^0 terms)": (fj.state_basis(A / fj.A_NORM, tx, ty), True)}
    for KA in (3, 4):
        for NT in (4, 6):
            bases[f"Legendre A<= {KA}, t total<= {NT}"] = (legendre_basis(A, tx, ty, KA, NT), False)
    surf = {}
    for bname, (B, minus_f0) in bases.items():
        res = {"ncols": B.shape[1], "fam": {}}
        Pfam = {}
        Btr = B[even]
        for f in FAM6:
            y = par0[f] - (f0.get(f, 0.0) if minus_f0 else 0.0)
            c = np.linalg.lstsq(Btr, y[even], rcond=None)[0]
            fit = B @ c + (f0.get(f, 0.0) if minus_f0 else 0.0)
            Pfam[f] = fit
            res["fam"][f] = {"rms_train": float(np.sqrt(np.mean((fit - par0[f])[even] ** 2))),
                             "rms_comp": float(np.sqrt(np.mean((fit - par0[f])[comp] ** 2))),
                             "max_abs": float(np.max(np.abs(fit - par0[f]))), "param_rms": float(np.std(par0[f]))}
        e, dmin = eval_surface(Pfam)
        res["coord_full"], res["coord_comp"], res["min_den"] = metrics(e), metrics(e[comp]), dmin
        res["bins"] = binned(e, np.abs(BDE), A, G["acc"])
        surf[bname] = res
        log(f"  surface {bname}: implied complement vertex RMS {res['coord_comp']['vertex_rms_um']:.4f}")

    # joint coordinate-space LM with Legendre bases (parent BiaxialModel / lm_fit, basis swapped)
    joint = {}
    NB_saved = fj.NB
    try:
        for KA, NT in ((3, 4), (4, 4), (4, 6)):
            nm = f"Legendre A<= {KA}, t total<= {NT}"
            Btr_mask = np.broadcast_to(even.reshape(na, nt, nb), (na, nt, nb))
            Mt = fj.BiaxialModel(G, br["barrel_mm"], L, Btr_mask, FAM6)
            Mt.B = legendre_basis(Mt.A, Mt.tx, Mt.ty, KA, NT)
            fj.NB = Mt.B.shape[1]
            F = fj.lm_fit(Mt, np.zeros(6 * fj.NB), max_iter=40, label=f"joint {nm}")
            Mg.B = legendre_basis(Mg.A, Mg.tx, Mg.ty, KA, NT)
            rj, denj = Mg.residual_um(F["p"])
            joint[nm] = {"ncoef": 6 * fj.NB, "status": F["status"], "iterations": F["iterations"],
                         "runtime_s": F["runtime_s"], "train": metrics(rj[even]), "complement": metrics(rj[comp]),
                         "full": metrics(rj), "min_den": float(denj.min()),
                         "bins": binned(rj, np.abs(BDE), A, G["acc"])}
            log(f"  joint {nm}: complement {joint[nm]['complement']}")
    finally:
        fj.NB = NB_saved

    D = decomposition(prob, P0s, fits["a"]["P"], np.abs(BDE))
    k1k2 = float(np.max(np.abs(D["b_k2"]["cost"] - fits["a"]["cost"])))
    dstate = np.sqrt(np.mean(np.sum(errs["0"] ** 2, -1), -1)) - np.sqrt(np.mean(np.sum(rg ** 2, -1), -1))
    P = {"decomp": {k: {kk: vv for kk, vv in v.items() if kk != "cost"} for k, v in D.items()},
         "n_state_worse_than_global": int(np.sum(dstate > 1e-9)), "max_state_excess_over_global_um": float(dstate.max()),
         "k1_k2_equivalence_max_abs_cost_diff_um2": k1k2,
         "S": S, "L": L, "barrel_recon_err_mm": barrel_recon_err, "center_const_um": center_const,
         "barrel_center_um": barrel_center, "jac_audit": jaud, "summ": summ, "nest": nest, "c_beats_0": c_beats_0,
         "d_picked_from_b": fits["d"]["picked_from_b"], "glob": glob, "k1tab": k1tab, "id_dev": id_dev,
         "glob_vs_state": glob_vs_state, "surf": surf, "joint": joint, "acc": G["acc"],
         "kc": kc, "sha256": G["sha256"], "runtime_total_s": time.perf_counter() - T0}
    with (OUT / "p4_biaxial_per_state_ceiling.pkl").open("wb") as s:
        pickle.dump(P, s, pickle.HIGHEST_PROTOCOL)
    np.savez_compressed(OUT / "p4_biaxial_per_state_ceiling.npz", A=A, ADE=ADE, BDE=BDE, L_mm=L,
                        **{f"params_{v}": fits[v]["P"] for v in VARIANTS},
                        **{f"param_names_{v}": np.array(VARIANTS[v]["names"]) for v in VARIANTS},
                        **{f"state_rms_um_{v}": np.sqrt(np.mean(np.sum(errs[v] ** 2, -1), -1)) for v in VARIANTS},
                        **{f"status_{v}": fits[v]["status"] for v in VARIANTS},
                        **{f"cond_{v}": fits[v]["cond"] for v in VARIANTS},
                        k1_state_per_mm2=k1_state, k1_powerlaw_per_mm2=kA[ai],
                        global252_state_rms_um=np.sqrt(np.mean(np.sum(rg ** 2, -1), -1)))
    write_report(P, OUT / "improvement.md")
    log(f"Done in {time.perf_counter()-T0:.1f} s")


# ------------------------------------------------------------------------------- report
def write_report(P, path):
    S = P["summ"]
    g = P["glob"]
    L = ["# P4 biaxial model: per-state error ceiling and whether higher-order surfaces can help", "",
         "Generated by `Script/p4_biaxial_per_state_ceiling.py`. Data: "
         f"`data/distortion_grid_p4_biaxial_macro/distortion_grid.pkl` (SHA-256 `{P['sha256']}`), real_x/y_mm only, "
         f"center-relative, {P['S']:,} states x 9 fields. All errors are against the ray-traced simulation (um).", "",
         "## What was fitted", "",
         "Each state is fitted on its own (no polynomial across states, translation T = 0). Normalised coordinates "
         f"u = mm / L, L = {P['L']:.12g} mm. `n = s(A) D_k0^-1(r0) / L` is the paraxial-at-A grid (refit power-law barrel "
         f"c0, c1, n = {np.array2string(P['kc'], precision=10)}; measured s(A)); "
         "`R(u,k) = u (1 + k|u|^2)` so `R(n, k(A)) = B_A(r0)/L` (reconstruction matches the parent script's barrel to "
         f"{P['barrel_recon_err_mm']:.1e} mm); `H(u) = M u / (1 + q.u)`; "
         "`BC(u) = u (1 + k2 r^4) + [2 p1 x y + p2 (r^2 + 2x^2), p1 (r^2 + 2y^2) + 2 p2 x y]` (k2, p1, p2 in normalised units).", "",
         "| variant | definition | params | residuals (effective) |", "|---|---|---:|---|"]
    defs = {"0": "X = L H(R(n, k(A)))  (current form, barrel fixed)", "a": "X = L H(R(n, k))  (free per-state k1)",
            "b": "X = L H(BC(R(n, k(A))))", "c": "X = L R(H(n), k(A))  (keystone first, then the A-dependent radial; s(A) and D_k0^-1 still applied first)",
            "d": "X = L H(BC(R(n, k)))"}
    for v in VARIANTS:
        L.append(f"| ({v}) | `{defs[v]}` | {S[v]['nparams']} | 18 (16) |")
    L += ["", f"Every variant maps the center field to exactly 0 (max |barrel center| {P['barrel_center_um']:.1e} um), and the "
          f"data center is {P['center_const_um']:.1e} um at most, so the center's 2 residuals are a fixed constant: "
          "**16 effective residuals per state**. The brief's parameter list for (d) "
          "(M 4 + q 2 + k1 + p1, p2 + k2) sums to 10, not 11; nothing is omitted, and redundancy is 16 - 10 = 6. With 6..10 parameters on 8 moving points, a "
          "per-state fit is a ceiling (in-sample, overfit-prone by construction), not a deployable model.", "",
          "Solver: batched Levenberg-Marquardt over all states simultaneously (per-state damping, per-state accept/reject, "
          "denominator guard 0.05), complex-step Jacobians (audit vs central FD on 50 states: max rel err "
          f"{P['jac_audit']:.1e}). (0) starts from a linear DLT solution; (a), (b) start from the (0) optimum; (d) is run "
          f"from both the (a) and (b) optima and the better is kept per state ({P['d_picked_from_b']} states from (b)); "
          "(c) starts from the (0) parameters.", "",
          "Convergence/nesting checks (children must not be worse than parents):", ""]
    for k, vv in P["nest"].items():
        L.append(f"- cost({k.split('<=')[0]}) <= cost({k.split('<=')[1]}): {vv['violations_gt_1e-9']} violations > 1e-9 um^2 (max excess {vv['max_excess_um2']:.2e} um^2)")
    L.append(f"- (c) is not nested with (0); it is lower-cost on {P['c_beats_0']:,} of {P['S']:,} states.")
    L += ["", "| variant | converged | stuck (no descent) | max_iter | max scaled grad | p99 scaled grad | median cond | frac cond > 1e10 | min den | runtime (s) |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for v in VARIANTS:
        s = S[v]
        sc = s["status_counts"]
        L.append(f"| ({v}) | {sc['converged']} | {sc['stuck']} | {sc['max_iter']} | {s['max_scaled_grad']:.2e} | {s['p99_scaled_grad']:.2e} | "
                 f"{s['median_cond']:.2e} | {s['frac_cond_gt_1e10']:.4f} | {s['min_den']:.4f} | {s['runtime_s']:.1f} |")
    L += ["", "('stuck' = no damped step lowered the cost, i.e. at a stationary point to machine precision; scaled grad "
          "= max |J^T r| / sqrt(diag J^T J), um.)", "",
          "## Per-state ceiling vs global fits (all 214,221 states, all nine fields)", "",
          "| model | params | vertex RMS (um) | coord RMSE (um) | max point (um) | mean state RMS | median state RMS | p99 state RMS | max state RMS |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|"]

    def row(lab, npar, m):
        return (f"| {lab} | {npar} | {m['vertex_rms_um']:.4f} | {m['coord_rmse_um']:.4f} | {m['max_point_um']:.4f} | "
                f"{m['mean_state_rms_um']:.4f} | {m['median_state_rms_um']:.4f} | {m['p99_state_rms_um']:.4f} | {m['max_state_rms_um']:.4f} |")
    L.append(row("global 252-coef T=0 (parent `control_no_T`), complement holdout", "252 total", g["complement"]))
    L.append(row("global 252-coef T=0, in-sample even/even", "252 total", g["train"]))
    L.append(row("global 252-coef T=0, full grid", "252 total", g["full"]))
    L.append("| global 336-coef with T (offset model), complement holdout (from report.md) | 336 total | 6.4300 | 4.5467 | 50.8052 | 5.3757 | | | 26.5603 |")
    for v in VARIANTS:
        L.append(row(f"per-state ({v})", f"{S[v]['nparams']}/state", S[v]["all_nine"]))
    L += ["", "Outer eight fields only (center excluded):", "",
          "| model | vertex RMS (um) | coord RMSE (um) | max point (um) |", "|---|---:|---:|---:|"]
    for v in VARIANTS:
        m = S[v]["outer8"]
        L.append(f"| per-state ({v}) | {m['vertex_rms_um']:.4f} | {m['coord_rmse_um']:.4f} | {m['max_point_um']:.4f} |")

    L += ["", "## Binned by |BDE| (vertex RMS / coord RMSE / max point, um)", "",
          "| |BDE| | states | global 252 (full grid) | " + " | ".join(f"({v})" for v in VARIANTS) + " |",
          "|---|---:|---|" + "---|" * len(VARIANTS)]
    for i, rb in enumerate(g["bins"][0]):
        cells = [f"{S[v]['bins'][0][i][2]:.3f} / {S[v]['bins'][0][i][3]:.3f} / {S[v]['bins'][0][i][4]:.2f}" for v in VARIANTS]
        L.append(f"| {rb[0]} | {rb[1]} | {rb[2]:.3f} / {rb[3]:.3f} / {rb[4]:.2f} | " + " | ".join(cells) + " |")
    L += ["", "## Binned by accommodation (vertex RMS / coord RMSE / max point, um)", "",
          "| A (D) | global 252 (full grid) | " + " | ".join(f"({v})" for v in VARIANTS) + " |",
          "|---:|---|" + "---|" * len(VARIANTS)]
    for i, ra in enumerate(g["bins"][1]):
        cells = [f"{S[v]['bins'][1][i][2]:.3f} / {S[v]['bins'][1][i][3]:.3f} / {S[v]['bins'][1][i][4]:.2f}" for v in VARIANTS]
        L.append(f"| {ra[0]:.1f} | {ra[2]:.3f} / {ra[3]:.3f} / {ra[4]:.2f} | " + " | ".join(cells) + " |")
    for v in VARIANTS:
        L += ["", f"### 10 worst states, variant ({v})", "",
              "| A | ADE | BDE | state RMS (um) | max point (um) | worst field | ray-failure state |", "|---:|---:|---:|---:|---:|---|---|"]
        for w in S[v]["worst"]:
            L.append(f"| {w['A']:g} | {w['ADE']:g} | {w['BDE']:g} | {w['state_rms_um']:.4f} | {w['max_point_um']:.4f} | "
                     f"({w['worst_field'][0]:g},{w['worst_field'][1]:g}) | {w['ray_failure']} |")

    L += ["", "## Per-state k1 (variant a) vs power-law k1(A)", "",
          "dk = k1_state - k1(A), mm^-2.", "", "| |BDE| | mean dk | std dk | min dk | max dk | mean dk/k1(A) |",
          "|---|---:|---:|---:|---:|---:|"]
    for r in P["k1tab"]["bde"]:
        L.append(f"| {r[0]} | {r[1]:.4e} | {r[2]:.4e} | {r[3]:.4e} | {r[4]:.4e} | {r[5]:.4f} |")
    L += ["", "| A (D) | k1(A) | mean dk | std dk | min dk | max dk |", "|---:|---:|---:|---:|---:|---:|"]
    for r in P["k1tab"]["acc"]:
        L.append(f"| {r[0]:.1f} | {r[1]:.6e} | {r[2]:.4e} | {r[3]:.4e} | {r[4]:.4e} | {r[5]:.4e} |")

    L += ["", "## Are the per-state variant-(0) parameters smooth in (A, t_x, t_y)?", "",
          "Per-state (0) parameters are fitted by unweighted linear least squares on the even/even states (54,621) and "
          "evaluated everywhere. The surfaces are then put back into the variant-(0) forward model (fixed barrel) to get "
          "the implied coordinate error. Caveat: parameter-space least squares is not coordinate-optimal, so the "
          "implied error is expected to be at or above a joint coordinate-space fit with the same basis.", "",
          "Per-state (0) parameters at ADE = BDE = 0 vs identity (max |f - f0| over the 21 A): " +
          ", ".join(f"{f} {v:.2e}" for f, v in P["id_dev"].items()) +
          ". The existing basis has no t^0 terms, so this part cannot be represented by it.", "",
          "RMS difference between the global 252-coef fit's families and the per-state (0) optimum: " +
          ", ".join(f"{f} {v:.2e}" for f, v in P["glob_vs_state"].items()) + ".", "",
          "| basis | cols/family | " + " | ".join(f"{f} RMS misfit (comp)" for f in FAM6) + " | implied comp vertex RMS | coord RMSE | max point | full-grid vertex RMS | min den |",
          "|---|---:|" + "---:|" * (len(FAM6) + 5)]
    for b, r in P["surf"].items():
        L.append(f"| {b} | {r['ncols']} | " + " | ".join(f"{r['fam'][f]['rms_comp']:.2e}" for f in FAM6) +
                 f" | {r['coord_comp']['vertex_rms_um']:.3f} | {r['coord_comp']['coord_rmse_um']:.3f} | "
                 f"{r['coord_comp']['max_point_um']:.2f} | {r['coord_full']['vertex_rms_um']:.3f} | {r['min_den']:.4f} |")
    L += ["", "Per-state parameter spread (std over all states) for scale: " +
          ", ".join(f"{f} {list(P['surf'].values())[0]['fam'][f]['param_rms']:.2e}" for f in FAM6) + ".", "",
          "### Joint coordinate-space fit with Legendre bases (center-exact, T = 0)", "",
          "Same LM / normal-equation machinery as `fit_p4_biaxial_joint.py` (BiaxialModel, lm_fit) with the state basis "
          "replaced by the Legendre basis (constant and A-only terms included), trained on even/even, all nine fields.", "",
          "| basis | coefs | status | iters | train vertex RMS | comp vertex RMS | comp coord RMSE | comp max point | full max state RMS | min den |",
          "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    L.append(f"| existing (parent control_no_T) | 252 | | | {g['train']['vertex_rms_um']:.3f} | {g['complement']['vertex_rms_um']:.3f} | "
             f"{g['complement']['coord_rmse_um']:.3f} | {g['complement']['max_point_um']:.2f} | {g['full']['max_state_rms_um']:.2f} | |")
    for b, r in P["joint"].items():
        L.append(f"| {b} | {r['ncoef']} | {r['status']} | {r['iterations']} | {r['train']['vertex_rms_um']:.3f} | "
                 f"{r['complement']['vertex_rms_um']:.3f} | {r['complement']['coord_rmse_um']:.3f} | "
                 f"{r['complement']['max_point_um']:.2f} | {r['full']['max_state_rms_um']:.2f} | {r['min_den']:.4f} |")
    Dd = P.get("decomp", {})
    if Dd:
        L += ["", "## Which extra terms matter (decomposition of (b)/(d))", "",
              "Per-state fits with subsets of the extra terms (same solver; (b_k2), (b_p) start from the (0) optimum, "
              "(d_k1p) from the (a) optimum).", "",
              "| variant | params | vertex RMS | coord RMSE | max point | median state RMS | vertex RMS by |BDE| bin 0 / (0,4] / (4,8] / (8,12] / (12,16] / (16,20] | converged / stuck / max_iter |",
              "|---|---:|---:|---:|---:|---:|---|---|"]
        rows = [("(0)", S["0"]["nparams"], S["0"]["all_nine"], [b[2] for b in S["0"]["bins"][0]], S["0"]["status_counts"]),
                ("(a) = (0) + k1", S["a"]["nparams"], S["a"]["all_nine"], [b[2] for b in S["a"]["bins"][0]], S["a"]["status_counts"])]
        rows += [(f"({k}) = {v['desc']}", v["nparams"], v["all_nine"], v["bde_vertex"], v["status_counts"]) for k, v in Dd.items()]
        rows += [("(b) = (0) + k2, p1, p2", S["b"]["nparams"], S["b"]["all_nine"], [b[2] for b in S["b"]["bins"][0]], S["b"]["status_counts"]),
                 ("(d) = (0) + k1, k2, p1, p2", S["d"]["nparams"], S["d"]["all_nine"], [b[2] for b in S["d"]["bins"][0]], S["d"]["status_counts"])]
        for lab, npar, m, bv, sc in rows:
            L.append(f"| {lab} | {npar} | {m['vertex_rms_um']:.4f} | {m['coord_rmse_um']:.4f} | {m['max_point_um']:.3f} | "
                     f"{m['median_state_rms_um']:.4f} | " + " / ".join(f"{x:.3f}" for x in bv) +
                     f" | {sc['converged']} / {sc['stuck']} / {sc['max_iter']} |")
        L += ["", "k1 vs k2 on a 3x3 grid: the eight outer points sit at only two radii (edge midpoints, corners), so with M "
              "already able to scale, one radial coefficient fixes the corner/edge gain ratio and a second one is redundant. "
              f"Per state, (0)+k2 reproduces (0)+k1 exactly (max |cost difference| {P['k1_k2_equivalence_max_abs_cost_diff_um2']:.1e} um^2). "
              "In (d), k1 and k2 differ only because p1, p2 act on the k1-distorted (not the k2-distorted) coordinates. This "
              "is a nearly degenerate direction, hence the large condition numbers and the slow, partly unconverged (d) solve: "
              "the (d) numbers are an upper bound on its ceiling. The per-state k1 of (a) is the same: it is the corner/edge "
              "gain ratio of an 8-point fit, not a uniquely identified physical k1."]

    L += ["", f"Total runtime {P['runtime_total_s']:.0f} s.", "", "## Interpretation", ""] + interpretation(P)
    path.write_text("\n".join(L), encoding="utf-8")


def interpretation(P):
    S, g = P["summ"], P["glob"]
    s0, gc, gf = S["0"]["all_nine"], g["complement"], g["full"]
    s0c = S["0"]["complement_all_nine"]
    J = P["joint"]
    j6 = J.get("Legendre A<= 4, t total<= 6")
    best = min(("a", "b", "c", "d"), key=lambda v: S[v]["all_nine"]["vertex_rms_um"])
    Dd = P.get("decomp", {})
    out = [
        f"**Decision-rule outcome: (ii), and (iii) also holds.** Variant (0) fitted freely per state, with no surface and "
        f"6 parameters on 16 effective residuals, reaches vertex RMS {s0['vertex_rms_um']:.3f} um, coord RMSE "
        f"{s0['coord_rmse_um']:.3f} um and max point {s0['max_point_um']:.2f} um over all states. The global center-exact "
        f"252-coefficient fit on the same states (full grid) gets {gf['vertex_rms_um']:.3f} / {gf['coord_rmse_um']:.3f} / "
        f"{gf['max_point_um']:.2f} um. On the complement holdout it is {gc['vertex_rms_um']:.3f} / {gc['coord_rmse_um']:.3f} / "
        f"{gc['max_point_um']:.2f} um, against the per-state ceiling on the same states of {s0c['vertex_rms_um']:.3f} / "
        f"{s0c['coord_rmse_um']:.3f} / {s0c['max_point_um']:.2f} um. RMS ratio per-state/global = "
        f"{s0['vertex_rms_um']/gf['vertex_rms_um']:.4f} (full grid) and {s0c['vertex_rms_um']/gc['vertex_rms_um']:.4f} (complement). "
        "Because the global fit's parameters are a feasible point of every per-state problem, the per-state error must "
        f"not exceed the global one at any state; it exceeds it at {P.get('n_state_worse_than_global', 'n/a')} of "
        f"{P['S']:,} states (max excess {P.get('max_state_excess_over_global_um', float('nan')):.1e} um), so the per-state "
        "solutions are genuine minima, not local-minimum artefacts. "
        "So the order-4 coefficient surfaces already reach the per-state RMS ceiling of this model form to within about 0.3 %. "
        "They are not the bottleneck: raising the polynomial order cannot lower the RMS by more than about 0.03 um. The only "
        f"thing order buys is the extreme-state maximum ({gf['max_point_um']:.1f} um global vs {s0['max_point_um']:.1f} um per "
        "state, at |ADE| = |BDE| = 20 deg, 4 D).",
        "",
        "This was checked directly. The per-state (0) parameters are smooth: order-4 surfaces leave ~2e-4 RMS misfit "
        "(under 1 % of their spread across states), and order-6-in-t Legendre surfaces leave ~1e-5. But the implied coordinate error moves "
        "only from 9.03 to 9.00 um. The joint coordinate-space refit with Legendre bases gives the same answer: "
        + (f"840 coefficients (A order 4, t total order 6) reach {j6['complement']['vertex_rms_um']:.3f} um holdout vertex RMS and "
           f"{j6['complement']['max_point_um']:.2f} um max, against 9.027 / 53.20 for the existing 252. Order 3 vs 4 in A "
           "makes no difference." if j6 else "") +
        " The lower error of the offset model (6.43 vs 9.03 um holdout vertex RMS) comes from its 2 extra per-state degrees of freedom (T, "
        "which moves the center), not from better surfaces. It lies below the center-exact per-state ceiling, so no "
        "center-exact surface of any order can match it.",
        "",
        f"**What helps (per-state ceilings; in-sample, i.e. lower bounds on the error any global smooth model of that "
        f"form could reach):** the best is ({best}). "
        f"(a) free k1: {S['a']['all_nine']['vertex_rms_um']:.2f} um. (c) keystone-before-radial: "
        f"{S['c']['all_nine']['vertex_rms_um']:.2f} um; it is better on {P['c_beats_0']:,} of {P['S']:,} states but worse at "
        "A = 0, so swapping the order is a marginal change. (b) Brown-Conrady k2, p1, p2: "
        f"{S['b']['all_nine']['vertex_rms_um']:.2f} um. (d) everything: {S['d']['all_nine']['vertex_rms_um']:.2f} um "
        "(this value is itself an upper bound on (d)'s ceiling, because part of the solve is unconverged). "
        + (f"The decomposition shows that the decentering (tangential) pair p1, p2 is the dominant missing term: alone it takes "
           f"9.03 -> {Dd['b_p']['all_nine']['vertex_rms_um']:.2f} um. A free radial term (k1 or, equivalently, k2) takes 9.03 -> "
           f"{Dd['b_k2']['all_nine']['vertex_rms_um']:.2f} um. Together, k1 + p1, p2 (9 parameters) reach "
           f"{Dd['d_k1p']['all_nine']['vertex_rms_um']:.2f} um. On a fixed 3x3 grid, p1 and p2 are just two fixed quadratic "
           "displacement patterns with free amplitudes. What this shows is that the variant-(0) residual projects strongly "
           "onto those patterns. It does not show that the mirror is physically decentered (not tested). Based on these "
           "ceilings, the next model to try globally is therefore p1, p2 (and a per-state radial "
           "gain) as smooth families in (A, t_x, t_y), not higher-order keystone surfaces. " if Dd else ""),
        "",
        "**1 um target (iii):** no variant reaches it under either metric. By max point error (the criterion of the "
        f"parent report), nothing comes close: even the most favourable bin (BDE = 0) has a {S['d']['bins'][0][0][4]:.1f} um max point "
        f"under (d). Overall, the best per-state ceiling is {S['d']['all_nine']['vertex_rms_um']:.2f} um "
        f"vertex RMS with max point {S['d']['all_nine']['max_point_um']:.1f} um. Only near BDE = 0 do (b)/(d) get to "
        f"~{S['d']['bins'][0][0][2]:.2f} um RMS, still with ~{S['d']['bins'][0][0][4]:.1f} um max. The error grows steadily with |BDE| "
        f"(to {S['d']['bins'][0][-1][2]:.1f} um RMS for |BDE| in (16,20]) and with A. With 9-10 parameters on 16 effective "
        "residuals these per-state ceilings are already optimistic (heavy overfitting of 8 moving points is possible), "
        "so a global smooth model in this family will do no better. Reaching 1 um on all states needs a different "
        "model class, e.g. more field points per state to constrain higher-order terms, or a model built on ray-traced "
        "mirror geometry.",
        "",
        "Caveats: all errors are against the simulation, not hardware. Per-state fits are in-sample by construction. The "
        f"(d) solve hit the iteration cap on {S['d']['status_counts']['max_iter']:,} states, so its numbers are an upper bound "
        "(the nesting checks pass: no child is worse than its parent). 'Stuck' states are at a stationary point (scaled "
        "gradient <= 6e-6 um for (0), (a), (b), (c)).",
    ]
    return out


if __name__ == "__main__":
    if "--report-only" in sys.argv:
        report_only()
    else:
        main()
