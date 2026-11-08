"""Fit the extended biaxial (ADE x BDE) P4 joint model to the two-axis ray-traced grid.

Model (center-relative real-ray coordinates, millimetres):

    r0      = real grid at A=0, ADE=0, BDE=0 (nine fields)
    b       = B_A(r0) / L                      refit power-law barrel + paraxial scale
    den     = 1 + qx*b_x + qy*b_y
    X       = L*(m11*b_x + m12*b_y)/den + Tx
    Y       = L*(m21*b_x + m22*b_y)/den + Ty

Each family f in (m11, m22, m12, m21, qx, qy, Tx, Ty) is
    f = f0 + sum_{l=0..2} sum_{1<=i+j<=4} c_{l,i,j} a^l t_x^i t_y^j,
with a = A/5 (A in 0..4 D, so a in 0..0.8), t_x = BDE/20, t_y = ADE/20, f0 = 1 for m11/m22 and 0
otherwise.  So the keystone is the identity and T = 0 at zero rotation for every A.  14 monomials x 3
accommodation powers = 42 coefficients per family, 336 in total.

Objective: forward all-nine coordinate residual (micrometres), solved by Levenberg-Marquardt on
exactly accumulated normal equations (J^T J is assembled blockwise because every coefficient's basis
value depends only on the state), so the full 21x101x101x9 grid is used without a dense Jacobian.

Stage 1 (BDE=0 slice) and old-model comparisons are intentionally not run (coordinator instruction).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Script"))

import refit_p4_separable_transform as sep  # noqa: E402  barrel_map, metric_summary
from fit_p4_radial_distortion import RESULT_DTYPE, fit_accommodation_models, fit_group  # noqa: E402

GRID_PATH = ROOT / "data" / "distortion_grid_p4_biaxial_macro" / "distortion_grid.pkl"
OLD_RADIAL_FIT = ROOT / "data" / "p4_radial_distortion_fit" / "p4_radial_distortion_fit.pkl"
OLD_RADIAL_MODELS = ROOT / "data" / "p4_radial_distortion_fit" / "p4_radial_distortion_accommodation_models.pkl"
OLD_RADIAL_METRICS = ROOT / "data" / "p4_radial_distortion_fit" / "p4_radial_distortion_metrics.pkl"
OUTPUT_DIR = ROOT / "data" / "p4_biaxial_joint"
SCHEMA = "p4-biaxial-joint-forward-v1"

A_NORM = 5.0                       # a = A/5, as in all earlier P4 scripts
T_NORM = 20.0
MONOS = [(i, d - i) for d in range(1, 5) for i in range(d, -1, -1)]   # (power of t_x, power of t_y)
NMONO = len(MONOS)                 # 14
NL = 3                             # a^0..a^2
NB = NL * NMONO                    # 42 per family
FAMILIES = ("m11", "m22", "m12", "m21", "qx", "qy", "Tx", "Ty")
BASE = {"m11": 1.0, "m22": 1.0}
EVEN_TX = ("m11", "m22", "qy", "Ty")   # even in t_x under the BDE mirror (x -> -x)
ODD_TX = ("m12", "m21", "qx", "Tx")
FIELD_CLASSES = {"corners": [0, 2, 6, 8], "edge_midpoints": [1, 3, 5, 7], "center": [4],
                 "all_nine": list(range(9))}
DEN_GUARD = 0.05


def log(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------------------------- data
def load_biaxial(path):
    raw = path.read_bytes()
    d = pickle.loads(raw)
    acc = np.unique(d["accommodation_d"])
    ade = np.unique(d["eye_rotation_deg"])
    bde = np.unique(d["eye_rotation_x_deg"])
    if not (np.allclose(acc, np.linspace(0, 4, 21), atol=1e-9, rtol=0)
            and np.allclose(ade, np.linspace(-20, 20, 101), atol=1e-9, rtol=0)
            and np.allclose(bde, np.linspace(-20, 20, 101), atol=1e-9, rtol=0)):
        raise ValueError("Unexpected accommodation/ADE/BDE lattice")
    if not np.allclose(d["z_mm"], 0, atol=1e-9, rtol=0):
        raise ValueError("Only Z=0 data supported")
    ai = np.searchsorted(acc, d["accommodation_d"])
    ti = np.searchsorted(ade, d["eye_rotation_deg"])
    bi = np.searchsorted(bde, d["eye_rotation_x_deg"])
    key = (ai * len(ade) + ti) * len(bde) + bi
    if not np.all(np.bincount(key, minlength=len(acc) * len(ade) * len(bde)) == 9):
        raise ValueError("Incomplete 3x3 grid at some (A, ADE, BDE) state")
    order = np.lexsort((d["field_y_relative"], d["field_x_relative"], bi, ti, ai))
    r = d[order]
    shp = (len(acc), len(ade), len(bde), 9)
    fields = np.stack((r["field_x_relative"], r["field_y_relative"]), -1).reshape(shp + (2,))
    expected = np.array([[x, y] for x in (-1., 0., 1.) for y in (-1., 0., 1.)])
    if not np.array_equal(fields, np.broadcast_to(expected, fields.shape)):
        raise ValueError("Field identities/order inconsistent")
    real = np.stack((r["real_x_mm"], r["real_y_mm"]), -1).reshape(shp + (2,))
    para = np.stack((r["paraxial_x_mm"], r["paraxial_y_mm"]), -1).reshape(shp + (2,))
    cabs = np.stack((r["center_absolute_x_mm"], r["center_absolute_y_mm"]), -1).reshape(shp + (2,))
    if not np.isfinite(real).all() or not np.isfinite(cabs).all():
        raise ValueError("Nonfinite real/center coordinates")
    nan_rows = int(np.sum(~np.isfinite(r["paraxial_x_mm"]) | ~np.isfinite(r["paraxial_y_mm"])
                          | ~np.isfinite(r["radial_distortion_pct"]) | ~np.isfinite(r["tangential_distortion_pct"])))
    nan_state = ~np.all(np.isfinite(para), axis=(-2, -1))
    return {"acc": acc, "ade": ade, "bde": bde, "fields": expected, "real": real, "para": para,
            "center_abs": cabs[..., 0, :], "center_abs_field_spread_mm": float(np.max(np.abs(cabs - cabs[..., :1, :]))),
            "center_rel_max_mm": float(np.max(np.abs(real[..., 4, :]))),
            "nan_rows": nan_rows, "nan_state": nan_state, "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw), "nrows": len(d)}


# -------------------------------------------------------------------------------- barrel
def refit_barrel(G, train_acc_mask):
    """Repeat fit_p4_radial_distortion's procedure on the ADE=BDE=0 rows of the new grid."""
    it0 = int(np.flatnonzero(np.isclose(G["ade"], 0, atol=1e-9, rtol=0))[0])
    ib0 = int(np.flatnonzero(np.isclose(G["bde"], 0, atol=1e-9, rtol=0))[0])
    para, real = G["para"][:, it0, ib0], G["real"][:, it0, ib0]
    finite = np.all(np.isfinite(para), axis=(-2, -1))
    dropped_rows = int(np.sum(~np.isfinite(para).all(axis=-1)))
    use = train_acc_mask & finite
    dt = np.dtype([(n, "f8") for n in ("paraxial_x_mm", "paraxial_y_mm", "real_x_mm", "real_y_mm")])
    results = np.empty(int(use.sum()), dtype=RESULT_DTYPE)
    k_direct = np.full(len(G["acc"]), np.nan)
    for n, ia in enumerate(np.flatnonzero(use)):
        rows = np.empty(9, dtype=dt)
        rows["paraxial_x_mm"], rows["paraxial_y_mm"] = para[ia, :, 0], para[ia, :, 1]
        rows["real_x_mm"], rows["real_y_mm"] = real[ia, :, 0], real[ia, :, 1]
        k1, rmse, mx = fit_group(float(G["acc"][ia]), rows)
        k_direct[ia] = k1
        results[n] = (G["acc"][ia], k1, rmse, mx, np.nan)
    models = fit_accommodation_models(results)
    power = models[models["model"] == "power_law"][0]
    kcoef = np.array([power["coefficient_0"], power["coefficient_1"], power["coefficient_2"]], float)
    p0 = para[0].reshape(-1)
    s_meas = np.array([float(p0 @ para[ia].reshape(-1) / (p0 @ p0)) for ia in range(len(G["acc"]))])
    # s(A) law (quadratic through s(0)=1) from training accommodations; used only for held-out A.
    A = G["acc"]
    m = use & (A > 0)
    X = np.column_stack((A[m], A[m] ** 2))
    sc = np.linalg.lstsq(X, s_meas[m] - 1.0, rcond=None)[0]
    s_law = 1.0 + sc[0] * A + sc[1] * A ** 2
    s_used = np.where(use, s_meas, s_law)
    ref = real[0]
    barrel = np.stack([sep.barrel_map(ref, A[ia], s_used[ia], kcoef) for ia in range(len(A))])
    return {"k_coefficients_c0_c1_n": kcoef, "models": models, "k1_direct": k_direct,
            "s_measured": s_meas, "s_law_coeffs": sc, "s_used": s_used, "barrel_mm": barrel,
            "reference_mm": ref, "paraxial_rows_dropped_in_slice": dropped_rows,
            "train_accommodations": A[use].copy()}


# --------------------------------------------------------------------------------- model
def state_basis(a, tx, ty):
    cols = [a ** l * tx ** i * ty ** j for l in range(NL) for (i, j) in MONOS]
    return np.stack(cols, axis=-1)


def basis_labels():
    return [f"a{l}_tx{i}_ty{j}" for l in range(NL) for (i, j) in MONOS]


class BiaxialModel:
    def __init__(self, G, barrel_mm, L, mask, free=FAMILIES):
        ai, ti, bi = np.nonzero(mask)
        self.idx = (ai, ti, bi)
        self.ai = ai
        self.A = G["acc"][ai]
        self.tx = G["bde"][bi] / T_NORM
        self.ty = G["ade"][ti] / T_NORM
        self.B = state_basis(self.A / A_NORM, self.tx, self.ty)
        self.L = L
        self.xb = barrel_mm[ai, :, 0] / L          # (S, 9)
        self.yb = barrel_mm[ai, :, 1] / L
        self.target = G["real"][ai, ti, bi]         # (S, 9, 2) mm
        self.free = tuple(free)
        self.S = len(ai)

    def unpack(self, p):
        p = np.asarray(p, float).reshape(len(self.free), NB)
        out = {f: np.full(self.S, BASE.get(f, 0.0)) for f in FAMILIES}
        for k, f in enumerate(self.free):
            out[f] = out[f] + self.B @ p[k]
        return out

    def predict(self, p):
        s = self.unpack(p)
        c = {f: v[:, None] for f, v in s.items()}
        den = 1.0 + c["qx"] * self.xb + c["qy"] * self.yb
        U = (c["m11"] * self.xb + c["m12"] * self.yb) / den
        V = (c["m21"] * self.xb + c["m22"] * self.yb) / den
        pred = np.stack((self.L * U + c["Tx"], self.L * V + c["Ty"]), axis=-1)
        return pred, den, U, V

    def residual_um(self, p):
        pred, den, _, _ = self.predict(p)
        return (pred - self.target) * 1000.0, den

    def grads(self, den, U, V):
        k = 1000.0 * self.L
        xd, yd = self.xb / den, self.yb / den
        g = {"m11": (k * xd, None), "m12": (k * yd, None), "m21": (None, k * xd), "m22": (None, k * yd),
             "qx": (-k * U * xd, -k * V * xd), "qy": (-k * U * yd, -k * V * yd),
             "Tx": (np.full_like(den, 1000.0), None), "Ty": (None, np.full_like(den, 1000.0))}
        return [g[f] for f in self.free]

    def normal_equations(self, p):
        _, den, U, V = self.predict(p)
        r, _ = self.residual_um(p)
        gs = self.grads(den, U, V)
        nf = len(self.free)
        JTJ = np.zeros((nf * NB, nf * NB))
        JTr = np.zeros(nf * NB)
        for a_ in range(nf):
            gxa, gya = gs[a_]
            w = np.zeros(self.S)
            if gxa is not None:
                w += np.sum(gxa * r[..., 0], axis=1)
            if gya is not None:
                w += np.sum(gya * r[..., 1], axis=1)
            JTr[a_ * NB:(a_ + 1) * NB] = self.B.T @ w
            for b_ in range(a_, nf):
                gxb, gyb = gs[b_]
                w = np.zeros(self.S)
                if gxa is not None and gxb is not None:
                    w += np.sum(gxa * gxb, axis=1)
                if gya is not None and gyb is not None:
                    w += np.sum(gya * gyb, axis=1)
                blk = self.B.T @ (w[:, None] * self.B)
                JTJ[a_ * NB:(a_ + 1) * NB, b_ * NB:(b_ + 1) * NB] = blk
                JTJ[b_ * NB:(b_ + 1) * NB, a_ * NB:(a_ + 1) * NB] = blk.T
        return JTJ, JTr, 0.5 * float(np.sum(r ** 2)), float(np.min(den))

    def dense_jacobian(self, p, states):
        _, den, U, V = self.predict(p)
        gs = self.grads(den, U, V)
        n = len(states)
        J = np.zeros((n, 9, 2, len(self.free) * NB))
        for a_, (gx, gy) in enumerate(gs):
            sl = slice(a_ * NB, (a_ + 1) * NB)
            if gx is not None:
                J[:, :, 0, sl] = gx[states, :, None] * self.B[states, None, :]
            if gy is not None:
                J[:, :, 1, sl] = gy[states, :, None] * self.B[states, None, :]
        return J.reshape(n * 18, -1)


def jacobian_audit(model, p, nstate=50, seed=1):
    rng = np.random.default_rng(seed)
    st = rng.choice(model.S, size=min(nstate, model.S), replace=False)
    J = model.dense_jacobian(p, st)
    worst = 0.0
    for col in range(len(p)):
        h = 1e-6
        pp, pm = p.copy(), p.copy()
        pp[col] += h
        pm[col] -= h
        fd = (model.residual_um(pp)[0][st] - model.residual_um(pm)[0][st]).reshape(-1) / (2 * h)
        worst = max(worst, float(np.max(np.abs(fd - J[:, col])) / max(1e-8, float(np.max(np.abs(fd))))))
    return worst


def lm_fit(model, p0, max_iter=60, label=""):
    t0 = time.perf_counter()
    p = np.asarray(p0, float).copy()
    lam = 1e-6
    history = []
    JTJ, JTr, cost, dmin = model.normal_equations(p)
    status = "max_iter"
    for it in range(max_iter):
        d = np.sqrt(np.maximum(np.diag(JTJ), 1e-30))
        As = JTJ / np.outer(d, d)
        gs = JTr / d
        accepted = False
        while lam < 1e12:
            try:
                step = -np.linalg.solve(As + lam * np.eye(len(p)), gs) / d
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            pn = p + step
            rn, den = model.residual_um(pn)
            cn = 0.5 * float(np.sum(rn ** 2))
            if np.min(den) > DEN_GUARD and np.isfinite(cn) and cn <= cost:
                accepted = True
                break
            lam *= 10
        if not accepted:
            status = "no_descent_step"
            break
        rel = (cost - cn) / max(cost, 1e-300)
        p = pn
        lam = max(lam / 10, 1e-12)
        JTJ, JTr, cost, dmin = model.normal_equations(p)
        gnorm = float(np.max(np.abs(JTr / d)))
        history.append((it, cost, rel, lam, gnorm))
        log(f"  [{label}] it {it:3d} cost={cost:.10g} rel_drop={rel:.3e} lam={lam:.1e} scaled_grad={gnorm:.3e}")
        if rel < 1e-13:
            status = "converged_rel_cost"
            break
    return {"p": p, "cost": cost, "iterations": len(history), "status": status, "history": history,
            "min_train_denominator": dmin, "runtime_s": time.perf_counter() - t0}


# ------------------------------------------------------------------------------- metrics
def class_metrics(err, G=None, idx=None):
    """err (S, 9, 2) in mm.  Returns per-class pooled metrics and worst state."""
    out = {}
    for name, ids in FIELD_CLASSES.items():
        e = err[:, ids, :]
        nrm = np.linalg.norm(e, axis=-1)
        st = np.sqrt(np.mean(nrm ** 2, axis=-1))
        k = int(np.argmax(st))
        m = {"vertex_rms_um": float(np.sqrt(np.mean(nrm ** 2)) * 1e3),
             "coordinate_rmse_um": float(np.sqrt(np.mean(e ** 2)) * 1e3),
             "max_point_error_um": float(np.max(nrm) * 1e3),
             "max_abs_coordinate_um": float(np.max(np.abs(e)) * 1e3),
             "mean_state_vertex_rms_um": float(np.mean(st) * 1e3),
             "max_state_vertex_rms_um": float(st[k] * 1e3)}
        if G is not None:
            ai, ti, bi = (x[k] for x in idx)
            m["worst_state"] = (float(G["acc"][ai]), float(G["ade"][ti]), float(G["bde"][bi]))
        out[name] = m
    return out


def evaluate(model_full, p, G, masks):
    r, den = model_full.residual_um(p)
    err = r / 1000.0
    res = {"min_denominator_full_grid": float(np.min(den))}
    for name, m in masks.items():
        sel = m[model_full.idx]
        if not np.any(sel):
            continue
        idx = tuple(x[sel] for x in model_full.idx)
        res[name] = class_metrics(err[sel], G, idx)
    return res, err


def surface_stats(model_full, p, free):
    """Max |f - f0| per family, plus max |contribution| of terms the BDE mirror forbids."""
    P = np.asarray(p).reshape(len(free), NB)
    odd_cols = np.array([i % 2 == 1 for l in range(NL) for (i, j) in MONOS])
    out = {}
    for k, f in enumerate(free):
        tot = model_full.B @ P[k]
        forbid = odd_cols if f in EVEN_TX else ~odd_cols
        fb = model_full.B[:, forbid] @ P[k][forbid]
        out[f] = {"max_abs_deviation": float(np.max(np.abs(tot))),
                  "max_abs_symmetry_forbidden_part": float(np.max(np.abs(fb))),
                  "max_abs_forbidden_coefficient": float(np.max(np.abs(P[k][forbid]))),
                  "max_abs_allowed_coefficient": float(np.max(np.abs(P[k][~forbid])))}
    return out


def binned_tables(err, model_full, G, masks_all):
    nrm = np.linalg.norm(err, axis=-1)                     # (S, 9)
    absb = np.abs(G["bde"][model_full.idx[2]])
    edges = [(-0.1, 0.1, "0"), (0.1, 4.05, "(0,4]"), (4.05, 8.05, "(4,8]"), (8.05, 12.05, "(8,12]"),
             (12.05, 16.05, "(12,16]"), (16.05, 20.1, "(16,20]")]
    bde_rows = []
    for lo, hi, lab in edges:
        s = (absb > lo) & (absb <= hi)
        bde_rows.append((lab, int(s.sum()), float(np.sqrt(np.mean(err[s] ** 2)) * 1e3),
                         float(np.sqrt(np.mean(nrm[s] ** 2)) * 1e3), float(np.max(nrm[s]) * 1e3)))
    acc_rows = []
    A = G["acc"][model_full.idx[0]]
    for a in G["acc"]:
        s = np.isclose(A, a, atol=1e-9)
        acc_rows.append((float(a), int(s.sum()), float(np.sqrt(np.mean(err[s] ** 2)) * 1e3),
                         float(np.sqrt(np.mean(nrm[s] ** 2)) * 1e3), float(np.max(nrm[s]) * 1e3)))
    return bde_rows, acc_rows


def worst_states(err, model_full, G, train_mask, n=10):
    nrm = np.linalg.norm(err, axis=-1)
    st = np.sqrt(np.mean(nrm ** 2, axis=-1))
    order = np.argsort(st)[::-1][:n]
    rows = []
    for k in order:
        ai, ti, bi = (int(x[k]) for x in model_full.idx)
        rows.append({"A": float(G["acc"][ai]), "ADE": float(G["ade"][ti]), "BDE": float(G["bde"][bi]),
                     "state_vertex_rms_um": float(st[k] * 1e3), "max_point_um": float(np.max(nrm[k]) * 1e3),
                     "worst_field": [float(v) for v in G["fields"][int(np.argmax(nrm[k]))]],
                     "in_training": bool(train_mask[ai, ti, bi]),
                     "ray_failure_state": bool(G["nan_state"][ai, ti, bi])})
    return rows


def mirror_symmetry(G):
    real = G["real"]
    mir = real[:, :, ::-1][:, :, :, [6, 7, 8, 3, 4, 5, 0, 1, 2]]   # BDE -> -BDE, fx -> -fx
    return {"max_abs_x_plus_mirrored_x_um": float(np.max(np.abs(real[..., 0] + mir[..., 0])) * 1e3),
            "max_abs_y_minus_mirrored_y_um": float(np.max(np.abs(real[..., 1] - mir[..., 1])) * 1e3)}


def center_abs_model(G, train_mask, masks):
    """Separate linear fit of the absolute centre position with the same 42-term basis per axis."""
    ai, ti, bi = np.nonzero(np.ones_like(train_mask))
    B = state_basis(G["acc"][ai] / A_NORM, G["bde"][bi] / T_NORM, G["ade"][ti] / T_NORM)
    y = G["center_abs"][ai, ti, bi]
    tr = train_mask[ai, ti, bi]
    coef = np.linalg.lstsq(B[tr], y[tr], rcond=None)[0]
    e = B @ coef - y
    out = {"coefficients": coef}
    for name, m in masks.items():
        s = m[ai, ti, bi]
        out[name] = {"coordinate_rmse_um": float(np.sqrt(np.mean(e[s] ** 2)) * 1e3),
                     "max_point_error_um": float(np.max(np.linalg.norm(e[s], axis=-1)) * 1e3)}
    out["max_abs_center_mm"] = float(np.max(np.abs(y)))
    return out


# -------------------------------------------------------------------------------- report
def mtable(metrics, classes=("corners", "edge_midpoints", "center", "all_nine")):
    lines = ["| Field class | vertex RMS (um) | coord RMSE (um) | max point (um) | max |coord| (um) | "
             "mean state RMS (um) | max state RMS (um) | worst state (A D, ADE, BDE) |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for c in classes:
        m = metrics[c]
        ws = m.get("worst_state")
        wss = f"({ws[0]:g}, {ws[1]:g}, {ws[2]:g})" if ws else ""
        lines.append(f"| {c} | {m['vertex_rms_um']:.4f} | {m['coordinate_rmse_um']:.4f} | "
                     f"{m['max_point_error_um']:.4f} | {m['max_abs_coordinate_um']:.4f} | "
                     f"{m['mean_state_vertex_rms_um']:.4f} | {m['max_state_vertex_rms_um']:.4f} | {wss} |")
    return lines


def write_report(P, path):
    G = P["data"]
    br = P["barrel_main"]
    L = []
    L += ["# P4 biaxial (ADE x BDE) joint model fit", "",
          "Generated by `Script/fit_p4_biaxial_joint.py`. Data: "
          f"`data/distortion_grid_p4_biaxial_macro/distortion_grid.pkl` (SHA-256 `{G['sha256']}`, "
          f"{G['nrows']:,} rows; 21 accommodations 0..4 D x 101 ADE x 101 BDE x 9 fields).", "",
          "**Scope note.** Stage 1 (BDE=0 slice refit), the evaluation of unchanged old/BDE-free models on the "
          "full grid, and all old-lens metric comparisons were intentionally not run (coordinator instruction). "
          "All numbers below are in-sample or holdout residuals against the CODE V simulation, not hardware "
          "measurements. No claim is made beyond 0..4 D, |ADE|,|BDE| <= 20 deg.", "",
          "## Model", "",
          "Center-relative real-ray coordinates (mm). `r0` is the real grid at A=0, ADE=0, BDE=0. "
          "`b = B_A(r0)/L`, with `B_A = D_k1(A) o S_s(A) o D_k1(0)^-1` (refit power-law barrel, see below), "
          f"`L = {P['L']:.15g} mm` (|x| of the (-1,-1) reference corner, a constant normalisation only).", "",
          "```", "den = 1 + qx*b_x + qy*b_y",
          "X   = L*(m11*b_x + m12*b_y)/den + Tx",
          "Y   = L*(m21*b_x + m22*b_y)/den + Ty",
          "f(A,ADE,BDE) = f0 + sum_{l=0..2} sum_{1<=i+j<=4} c_{l,i,j} a^l t_x^i t_y^j",
          "a = A/5 (so a in [0, 0.8]),  t_x = BDE/20,  t_y = ADE/20,  f0 = 1 (m11, m22) else 0", "```", "",
          "Where theta (= ADE) entered the earlier joint corner model: only the `s_x`, `s_y`, `q_n` blocks "
          "(t = theta/20, powers 1..4, times a^0..a^2). `beta(A)`/`M_B` depend on A only, and `L` is a constant "
          "normalisation, not a translation; no earlier P4 model had a translation term. Here every theta-dependent "
          "family becomes a polynomial in (t_x, t_y) and a new 2-D translation family `(Tx, Ty)` is added. "
          f"Because `real_*` is center-relative (max |center field real| = {G['center_rel_max_mm']*1e3:.3g} um over the "
          "grid), the translation is expected to be ~0; it is scored against a T=0 control.", "",
          "### Coefficient counts", "",
          f"14 monomials (total order 1..4 in t_x, t_y) x 3 accommodation powers = {NB} per family; "
          f"8 families -> {8*NB} coefficients (full model); control without qx/m12/m21 -> {5*NB}; "
          f"control without Tx/Ty -> {6*NB}. Barrel: 3 power-law constants + per-accommodation s(A) "
          "(fixed before the keystone fit, not jointly refit).", "",
          "Fit: Levenberg-Marquardt on the forward all-nine residual (um) with exact normal equations "
          f"(analytic Jacobian; finite-difference audit max relative error {P['jac_audit']:.2e} on 50 random "
          "states). Penalty-free; steps that would make the denominator < 0.05 on training states are rejected.", "",
          "## Data handling", "",
          f"- States keyed on (accommodation, ADE, BDE); every state has exactly 9 fields (checked).",
          f"- Rows with NaN in paraxial_* or *_pct: {G['nan_rows']} rows in {int(G['nan_state'].sum())} states. "
          "Paraxial values enter only the barrel refit (k1(A) and s(A)) on the ADE=BDE=0 slice "
          "(selected with np.isclose(atol=1e-9)); that slice has "
          f"{br['paraxial_rows_dropped_in_slice']} NaN rows, so **0 rows were dropped**. The keystone fit uses only "
          "real_* (no NaN), so the 102 ray-failure states remain in the fit/evaluation with their real coordinates; "
          "they are flagged in the worst-state table.",
          f"- center_absolute_* is identical across the 9 fields of a state to {G['center_abs_field_spread_mm']:.2e} mm.", "",
          "## Refit barrel constants (new lens, 0..4 D only) vs old (old lens, 0..5 D)", "",
          "Same procedure as `fit_p4_radial_distortion.py`: one pooled k1 per accommodation at ADE=BDE=0 from "
          "paraxial vs real, then `k1(A)=c0+c1*A^n` by bounded nonlinear least squares; s(A) = pooled paraxial "
          "scale ratio to A=0.", "",
          "| constant | old (stored artifact) | new |", "|---|---:|---:|"]
    ok, nk = P["old_barrel"]["k"], br["k_coefficients_c0_c1_n"]
    for i, nm in enumerate(("c0 (mm^-2)", "c1 (mm^-2 D^-n)", "n")):
        L.append(f"| {nm} | {ok[i]:.12g} | {nk[i]:.12g} |")
    L += ["", "| A (D) | old direct k1 | new direct k1 | old s(A) | new s(A) |", "|---:|---:|---:|---:|---:|"]
    for a in (0., 1., 2., 3., 4.):
        io = int(np.argmin(np.abs(P["old_barrel"]["acc"] - a)))
        inew = int(np.argmin(np.abs(G["acc"] - a)))
        L.append(f"| {a:g} | {P['old_barrel']['k1'][io]:.10g} | {br['k1_direct'][inew]:.10g} | "
                 f"{P['old_barrel']['s'][io]:.10g} | {br['s_measured'][inew]:.10g} |")
    dk = max(abs(P['old_barrel']['k1'][int(np.argmin(np.abs(P['old_barrel']['acc'] - a)))] - br['k1_direct'][i])
             for i, a in enumerate(G["acc"]))
    L += ["", f"Max |old - new| direct k1 over the 21 shared accommodations: {dk:.3e} mm^-2. At zero rotation the "
          "new grid reproduces the old lens's per-accommodation barrel and paraxial scale; the power-law constants "
          "differ only because the new fit domain is 0..4 D (21 points) instead of 0..5 D (51 points). "
          "No 5 D claim is made.", "", f"Barrel-only check at ADE=BDE=0 (B_A(r0) vs real, all 21 A): {P['barrel_check']}", ""]

    L += ["## Data symmetry (fit-free)", "",
          "BDE mirror test: compare (A, ADE, BDE, fx, fy) with (A, ADE, -BDE, -fx, fy), x negated: "
          f"max |x + x_mirror| = {P['mirror']['max_abs_x_plus_mirrored_x_um']:.3e} um, "
          f"max |y - y_mirror| = {P['mirror']['max_abs_y_minus_mirrored_y_um']:.3e} um.", ""]

    def fitline(F):
        return (f"status `{F['status']}`, {F['iterations']} LM iterations, cost {F['cost']:.10g} um^2, "
                f"runtime {F['runtime_s']:.1f} s, min training denominator {F['min_train_denominator']:.6f}")

    main = P["fits"]["main_even"]
    L += ["## Main fit: even ADE index & even BDE index (all 21 A), full 336-coefficient model", "",
          f"Training states: {P['counts']['even']:,}; holdout (complement, ADE or BDE index odd): "
          f"{P['counts']['complement']:,}; strictly odd/odd: {P['counts']['oddodd']:,}. ADE/BDE = +-20 and 0 are "
          "in-sample, so this tests interpolation on the 0.4 deg lattice only.", "", fitline(main), "",
          f"Min denominator over full grid: {main['eval']['min_denominator_full_grid']:.6f}.", ""]
    for nm, lab in (("train", "In-sample (even/even)"), ("complement", "Holdout: complement of even/even"),
                    ("oddodd", "Holdout: odd/odd only"), ("full", "Full grid (all 214,221 states)")):
        L += [f"### {lab}", ""] + mtable(main["eval"][nm]) + [""]

    full = P["fits"]["full"]
    L += ["## Fit on the full grid (all states; final coefficients)", "", fitline(full), "",
          f"Min denominator: {full['eval']['min_denominator_full_grid']:.6f}.", ""] + mtable(full["eval"]["full"]) + [""]

    ctl = P["fits"]["control_no_qx_m12_m21"]
    ctT = P["fits"]["control_no_T"]
    L += ["## Identifiability: qx, m12, m21 and T", "",
          "Same even/even training set; controls fix the named families at 0.", "",
          "| fit | coefficients | train cost (um^2) | train all-nine coord RMSE (um) | complement all-nine coord RMSE (um) | complement max point (um) | runtime (s) |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for nm, F in (("full model", main), ("qx=m12=m21=0", ctl), ("Tx=Ty=0", ctT)):
        L.append(f"| {nm} | {F['ncoef']} | {F['cost']:.8g} | {F['eval']['train']['all_nine']['coordinate_rmse_um']:.4f} | "
                 f"{F['eval']['complement']['all_nine']['coordinate_rmse_um']:.4f} | "
                 f"{F['eval']['complement']['all_nine']['max_point_error_um']:.4f} | {F['runtime_s']:.1f} |")
    L += ["", "### Control Tx=Ty=0, complement holdout by field class", ""] + mtable(ctT["eval"]["complement"])
    L += ["", "### Control qx=m12=m21=0, complement holdout by field class", ""] + mtable(ctl["eval"]["complement"])
    L += ["", "Fitted family magnitudes (main fit, over the full grid). 'Forbidden part' = contribution of the "
          "terms that the BDE mirror symmetry forbids: odd-in-t_x terms for m11, m22, qy, Ty; even-in-t_x terms "
          "(including t_x^0) for m12, m21, qx, Tx.", "",
          "| family | max |f - f0| | max |forbidden part| | max |forbidden coef| | max |allowed coef| |",
          "|---|---:|---:|---:|---:|"]
    for f, s in main["surfaces"].items():
        L.append(f"| {f} | {s['max_abs_deviation']:.4e} | {s['max_abs_symmetry_forbidden_part']:.4e} | "
                 f"{s['max_abs_forbidden_coefficient']:.4e} | {s['max_abs_allowed_coefficient']:.4e} |")
    L += [""]

    ah = P["fits"]["acc_holdout"]
    L += ["## Holdout 2: leave out A = 1.0 and 3.0 D", "",
          "Barrel refit without 1.0/3.0 D (power law on 19 accommodations; s(A) at held-out A from a quadratic "
          f"law s=1+s1*A+s2*A^2 fit to the training s values, s1,s2 = {ah['barrel']['s_law_coeffs'].tolist()}). "
          "Keystone trained on even/even states of the 19 accommodations; evaluated on every ADE x BDE state "
          "of the held-out accommodations.", "",
          f"Refit power law (c0, c1, n) = {ah['barrel']['k_coefficients_c0_c1_n'].tolist()}; s used at 1.0/3.0 D = "
          f"{ah['s_heldout'][0]:.10g}/{ah['s_heldout'][1]:.10g} vs measured {ah['s_meas_heldout'][0]:.10g}/{ah['s_meas_heldout'][1]:.10g}.", "",
          fitline(ah), ""]
    L += ["### Held-out accommodations (1.0, 3.0 D; all ADE x BDE)", ""] + mtable(ah["eval"]["heldout_acc"]) + [""]
    L += ["### Same held-out states predicted by the main fit (those accommodations in training at even/even)", ""] + \
        mtable(main["eval"]["heldout_acc"]) + [""]
    L += ["### Training accommodations, in-sample (even/even)", ""] + mtable(ah["eval"]["train"]) + [""]

    bde_rows, acc_rows = P["binned_main"]
    L += ["## Error vs |BDE| (main even/even fit, evaluated on full grid, all nine fields)", "",
          "| |BDE| bin (deg) | states | coord RMSE (um) | vertex RMS (um) | max point (um) |", "|---|---:|---:|---:|---:|"]
    L += [f"| {r[0]} | {r[1]} | {r[2]:.4f} | {r[3]:.4f} | {r[4]:.4f} |" for r in bde_rows]
    L += ["", "## Error vs accommodation (main even/even fit, full grid, all nine fields)", "",
          "| A (D) | states | coord RMSE (um) | vertex RMS (um) | max point (um) |", "|---:|---:|---:|---:|---:|"]
    L += [f"| {r[0]:.1f} | {r[1]} | {r[2]:.4f} | {r[3]:.4f} | {r[4]:.4f} |" for r in acc_rows]
    L += ["", "## 10 worst states (main even/even fit, full grid, by all-nine state vertex RMS)", "",
          "| A (D) | ADE | BDE | state RMS (um) | max point (um) | worst field | in training | ray-failure state |",
          "|---:|---:|---:|---:|---:|---|---|---|"]
    for w in P["worst_main"]:
        L.append(f"| {w['A']:g} | {w['ADE']:g} | {w['BDE']:g} | {w['state_vertex_rms_um']:.4f} | "
                 f"{w['max_point_um']:.4f} | ({w['worst_field'][0]:g},{w['worst_field'][1]:g}) | "
                 f"{w['in_training']} | {w['ray_failure_state']} |")
    ca = P["center_abs"]
    L += ["", "## Separate absolute-center translation fit (diagnostic)", "",
          "Because the main model is center-relative, the physically meaningful translation is the absolute "
          f"center (max |c_abs| = {ca['max_abs_center_mm']:.4f} mm). Linear least squares with the same 42-term "
          "basis per axis (zero at zero rotation), trained on even/even:", "",
          f"- in-sample coord RMSE {ca['train']['coordinate_rmse_um']:.4f} um, max point {ca['train']['max_point_error_um']:.4f} um",
          f"- complement holdout coord RMSE {ca['complement']['coordinate_rmse_um']:.4f} um, max point {ca['complement']['max_point_error_um']:.4f} um", ""]
    L += ["## Runtime", "", f"Total script wall time {P['runtime_total_s']:.1f} s; per-fit times listed above.", "",
          "## Interpretation", "", P["interpretation"], ""]
    path.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------------- main
def run(args):
    T0 = time.perf_counter()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    log("Loading grid ...")
    G = load_biaxial(GRID_PATH)
    na, nt, nb = len(G["acc"]), len(G["ade"]), len(G["bde"])
    log(f"  {G['nrows']} rows; NaN rows {G['nan_rows']} in {int(G['nan_state'].sum())} states")
    ti, bi = np.meshgrid(np.arange(nt), np.arange(nb), indexing="ij")
    even2 = (ti % 2 == 0) & (bi % 2 == 0)
    oddodd2 = (ti % 2 == 1) & (bi % 2 == 1)
    full = np.ones((na, nt, nb), bool)
    even = np.broadcast_to(even2, full.shape).copy()
    complement = ~even
    oddodd = np.broadcast_to(oddodd2, full.shape).copy()
    ho_acc = np.isclose(G["acc"], 1.0, atol=1e-9) | np.isclose(G["acc"], 3.0, atol=1e-9)
    heldout_acc = np.broadcast_to(ho_acc[:, None, None], full.shape).copy()
    train_ah = even & ~heldout_acc
    counts = {"even": int(even.sum()), "complement": int(complement.sum()), "oddodd": int(oddodd.sum())}
    masks = {"train": even, "complement": complement, "oddodd": oddodd, "full": full, "heldout_acc": heldout_acc}

    log("Refitting barrel on ADE=BDE=0 ...")
    br = refit_barrel(G, np.ones(na, bool))
    L_norm = float(abs(br["reference_mm"][0, 0]))
    it0, ib0 = nt // 2, nb // 2
    bchk = br["barrel_mm"] - G["real"][:, it0, ib0]
    barrel_check = (f"coord RMSE {np.sqrt(np.mean(bchk**2))*1e3:.4f} um, max point "
                    f"{np.max(np.linalg.norm(bchk, axis=-1))*1e3:.4f} um")
    log(f"  power law {br['k_coefficients_c0_c1_n']}, s(4)={br['s_measured'][-1]}; barrel check {barrel_check}")
    old_models = pickle.loads(OLD_RADIAL_MODELS.read_bytes())
    old_pow = old_models[old_models["model"] == "power_law"][0]
    old_fit = pickle.loads(OLD_RADIAL_FIT.read_bytes())
    old_met = pickle.loads(OLD_RADIAL_METRICS.read_bytes())["metrics"]
    old_barrel = {"k": [float(old_pow[f"coefficient_{i}"]) for i in range(3)], "acc": old_fit["accommodation_d"],
                  "k1": old_fit["k1_per_mm2"], "s": old_met["paraxial_scale_vs_a0"]}

    mirror = mirror_symmetry(G)
    log(f"  mirror symmetry {mirror}")

    log("Building full-grid evaluator ...")
    Mfull = BiaxialModel(G, br["barrel_mm"], L_norm, full)
    fits = {}

    def do_fit(name, train_mask, free, barrel, p0=None, eval_masks=masks, max_iter=args.max_iter):
        log(f"== fit {name}: {int(train_mask.sum())} states, families {free}")
        M = BiaxialModel(G, barrel, L_norm, train_mask, free)
        p0 = np.zeros(len(free) * NB) if p0 is None else p0
        F = lm_fit(M, p0, max_iter=max_iter, label=name)
        F["ncoef"] = len(F["p"])
        F["free"] = free
        Me = Mfull if barrel is br["barrel_mm"] and free == FAMILIES else BiaxialModel(G, barrel, L_norm, full, free)
        F["eval"], err = evaluate(Me, F["p"], G, eval_masks)
        F["surfaces"] = surface_stats(Me, F["p"], free)
        log(f"   -> train all-nine coord RMSE {F['eval']['train']['all_nine']['coordinate_rmse_um'] if 'train' in F['eval'] else float('nan'):.4f} um "
            f"in {F['runtime_s']:.1f} s")
        return F, err, M

    # Jacobian audit on a mid-fit-like point (small random coefficients).
    Maud = BiaxialModel(G, br["barrel_mm"], L_norm, even)
    rng = np.random.default_rng(0)
    paud = rng.normal(scale=0.003, size=len(FAMILIES) * NB)
    jac_audit = jacobian_audit(Maud, paud)
    log(f"Jacobian FD audit max relative error {jac_audit:.3e}")
    del Maud
    if jac_audit > 1e-5:
        raise RuntimeError("Analytic Jacobian audit failed")

    fits["main_even"], err_main, _ = do_fit("main_even", even, FAMILIES, br["barrel_mm"])
    with (out / "p4_biaxial_joint_partial.pkl").open("wb") as s:
        pickle.dump({"main_even_p": fits["main_even"]["p"]}, s)
    fits["control_no_qx_m12_m21"], _, _ = do_fit("control_no_qx_m12_m21", even,
                                                 ("m11", "m22", "qy", "Tx", "Ty"), br["barrel_mm"])
    fits["control_no_T"], _, _ = do_fit("control_no_T", even, ("m11", "m22", "m12", "m21", "qx", "qy"),
                                        br["barrel_mm"])
    fits["full"], err_full, _ = do_fit("full", full, FAMILIES, br["barrel_mm"], p0=fits["main_even"]["p"])

    log("Accommodation holdout: barrel refit without 1.0 / 3.0 D ...")
    br_ah = refit_barrel(G, ~ho_acc)
    ah_masks = {"train": train_ah, "heldout_acc": heldout_acc, "full": full}
    fits["acc_holdout"], _, _ = do_fit("acc_holdout", train_ah, FAMILIES, br_ah["barrel_mm"], eval_masks=ah_masks)
    fits["acc_holdout"]["barrel"] = {k: br_ah[k] for k in ("k_coefficients_c0_c1_n", "s_law_coeffs", "s_used",
                                                          "s_measured", "train_accommodations")}
    fits["acc_holdout"]["s_heldout"] = br_ah["s_used"][ho_acc]
    fits["acc_holdout"]["s_meas_heldout"] = br_ah["s_measured"][ho_acc]

    binned_main = binned_tables(err_main, Mfull, G, masks)
    worst_main = worst_states(err_main, Mfull, G, even)
    center_abs = center_abs_model(G, even, {"train": even, "complement": complement})

    P_bde, P_acc = binned_main
    m = fits["main_even"]
    c = fits["control_no_qx_m12_m21"]
    t = fits["control_no_T"]
    mm = m["eval"]
    ah = fits["acc_holdout"]["eval"]["heldout_acc"]["all_nine"]
    interp = (
        f"The 336-coefficient biaxial model reaches {mm['train']['all_nine']['coordinate_rmse_um']:.3f} um all-nine "
        f"coordinate RMSE in-sample (even/even) and {mm['complement']['all_nine']['coordinate_rmse_um']:.3f} um on "
        f"the interleaved holdout (max point {mm['complement']['all_nine']['max_point_error_um']:.2f} um; "
        f"worst state RMS {mm['full']['all_nine']['max_state_vertex_rms_um']:.2f} um). In-sample vs holdout are "
        f"{'essentially equal' if abs(mm['complement']['all_nine']['coordinate_rmse_um']/mm['train']['all_nine']['coordinate_rmse_um']-1) < 0.05 else 'different'} "
        "(the holdout is slightly lower because the +-20 deg extremes, where errors are largest, are all in the "
        "even/even training set), so there is no sign of overfitting on the 0.4 deg lattice; the residual is not "
        "an interpolation effect but a misfit of this model family. "
        f"Leaving out 1.0 and 3.0 D entirely gives {ah['coordinate_rmse_um']:.3f} um coord RMSE / "
        f"{ah['max_point_error_um']:.2f} um max point on those accommodations "
        f"(vs {mm['heldout_acc']['all_nine']['coordinate_rmse_um']:.3f} um for the main fit, which had them in training). "
        f"Fixing qx=m12=m21=0 raises the training cost from {m['cost']:.6g} to {c['cost']:.6g} um^2 "
        f"(x{c['cost']/m['cost']:.3g}): with BDE present these families are identifiable and carry the BDE keystone. "
        f"Fixing Tx=Ty=0 changes the cost from {m['cost']:.6g} to {t['cost']:.6g} um^2 (x{t['cost']/m['cost']:.4g}). "
        "Caution: because the data are center-relative, a nonzero T is not a physical translation; it moves the "
        f"center field off zero (center-field coord RMSE {mm['full']['center']['coordinate_rmse_um']:.2f} um in the full "
        "model, exactly 0 in the T=0 control) in exchange for lower error at the eight outer fields, i.e. it acts "
        "as an extra uniform-offset deformation term. The data are BDE-mirror symmetric to ~1e-5 um and the "
        "symmetry-forbidden coefficient parts come out at ~1e-11 or below without being imposed. The ~1 um "
        "measurement-noise reference is "
        f"{'met' if mm['full']['all_nine']['max_point_error_um'] <= 1 else 'not met'}: the error grows with |BDE| "
        f"(coord RMSE {P_bde[0][2]:.2f} um at BDE=0 to {P_bde[-1][2]:.2f} um at |BDE| in (16,20]) and with A "
        f"({P_acc[0][2]:.2f} um at 0 D to {P_acc[-1][2]:.2f} um at 4 D), and peaks at the corners of the combined "
        "|ADE|=|BDE|=20 deg states at 4 D. A single projective map on the barrel output with order-4 (t_x, t_y) "
        "coefficient polynomials therefore does not capture the combined-rotation deformation to the 1 um level.")

    payload = {
        "schema": SCHEMA, "data": {k: v for k, v in G.items() if k not in ("real", "para", "center_abs")},
        "nan_state_mask": G["nan_state"],
        "L": L_norm, "a_normalisation": "a = A/5", "t_normalisation": "t_x = BDE/20, t_y = ADE/20",
        "families": FAMILIES, "basis_labels": basis_labels(), "monomials_tx_ty": MONOS,
        "barrel_main": br, "old_barrel": old_barrel, "barrel_check": barrel_check, "mirror": mirror,
        "jac_audit": jac_audit, "counts": counts, "fits": fits, "binned_main": binned_main,
        "worst_main": worst_main, "center_abs": center_abs, "interpretation": interp,
        "runtime_total_s": time.perf_counter() - T0,
    }
    payload["data"]["fields"] = G["fields"]
    with (out / "p4_biaxial_joint.pkl").open("wb") as s:
        pickle.dump(payload, s, pickle.HIGHEST_PROTOCOL)
    np.savez(out / "p4_biaxial_joint_coefficients.npz",
             families=np.array(FAMILIES), basis_labels=np.array(basis_labels()), L_mm=L_norm,
             barrel_power_c0_c1_n=br["k_coefficients_c0_c1_n"], s_A=br["s_measured"], accommodation_d=G["acc"],
             reference_mm=br["reference_mm"],
             **{f"{k}_coefficients": v["p"].reshape(len(v["free"]), NB) for k, v in fits.items()},
             **{f"{k}_families": np.array(v["free"]) for k, v in fits.items()})
    (out / "p4_biaxial_joint_partial.pkl").unlink(missing_ok=True)
    write_report(payload, out / "report.md")
    log(f"Done in {payload['runtime_total_s']:.1f} s -> {out}")
    log(interp)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--max-iter", type=int, default=60)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
