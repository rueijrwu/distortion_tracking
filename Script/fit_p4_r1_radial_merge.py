"""Fit the requested P4 radial-merge accommodation and two-axis gaze model.

The script only consumes the stored P4 grid and writes artifacts under
data/p4_r1_radial_merge. It never starts CODE V or recomputes reference models.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "4"

import numpy as np
from scipy.optimize import least_squares

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "data/distortion_grid_p4_biaxial_macro/distortion_grid.pkl"
OUT = ROOT / "data/p4_r1_radial_merge"
L = 1.235429592627
C0 = -0.0185004967
K = np.array([C0, 2.3989235714e-4, 1.5756063948e-3, -1.4256378340e-3, 6.6200789981e-4])
S1 = -0.081595476571
FIELDS = np.array([(x, y) for x in (-1., 0., 1.) for y in (-1., 0., 1.)])
OUTER = np.array([0, 1, 2, 3, 5, 6, 7, 8])


def load_grid():
    raw = GRID.read_bytes()
    d = pickle.loads(raw)
    A, ADE, BDE = (np.unique(d[k]) for k in ("accommodation_d", "eye_rotation_deg", "eye_rotation_x_deg"))
    assert len(d) == 21 * 101 * 101 * 9
    assert np.allclose(A, np.linspace(0, 4, 21), atol=1e-9, rtol=0)
    assert np.allclose(ADE, np.linspace(-20, 20, 101), atol=1e-9, rtol=0)
    assert np.allclose(BDE, np.linspace(-20, 20, 101), atol=1e-9, rtol=0)
    ai = np.searchsorted(A, d["accommodation_d"])
    di = np.searchsorted(ADE, d["eye_rotation_deg"])
    bi = np.searchsorted(BDE, d["eye_rotation_x_deg"])
    order = np.lexsort((d["field_y_relative"], d["field_x_relative"], di, bi, ai))
    r = d[order]
    shape = (21, 101, 101, 9)
    fields = np.stack((r["field_x_relative"], r["field_y_relative"]), -1).reshape(shape + (2,))
    assert np.array_equal(fields, np.broadcast_to(FIELDS, fields.shape))
    for key, vals in (("accommodation_d", A[:, None, None, None]),
                      ("eye_rotation_x_deg", BDE[None, :, None, None]),
                      ("eye_rotation_deg", ADE[None, None, :, None])):
        assert np.array_equal(r[key].reshape(shape), np.broadcast_to(vals, shape))
    real = np.stack((r["real_x_mm"], r["real_y_mm"]), -1).reshape(shape + (2,))
    centered = real - real[..., 4:5, :]
    return A, ADE, BDE, real, centered, hashlib.sha256(raw).hexdigest()


def symmetry(real, centered):
    idx = {tuple(v): i for i, v in enumerate(FIELDS)}
    fy = np.array([idx[(x, -y)] for x, y in FIELDS])
    fx = np.array([idx[(-x, y)] for x, y in FIELDS])
    rows = {}
    for name, x in (("raw", real), ("centered", centered)):
        ey = (x[:, :, ::-1, :, :][:, :, :, fy, :] * [1., -1.] - x) * 1000.
        ex = (x[:, ::-1, :, :, :][:, :, :, fx, :] * [-1., 1.] - x) * 1000.
        rows[name] = {"ADE_flip_y_max_abs_um": float(np.max(np.abs(ey))),
                      "ADE_flip_y_rms_component_um": float(np.sqrt(np.mean(ey**2))),
                      "BDE_flip_x_max_abs_um": float(np.max(np.abs(ex))),
                      "BDE_flip_x_rms_component_um": float(np.sqrt(np.mean(ex**2)))}
    return rows


def radial_inverse(points):
    """Invert rho -> rho*(1+C0*rho^2) on its continuous positive central branch."""
    rad = np.linalg.norm(points, axis=-1)
    rho = rad.copy()
    for _ in range(16):
        rho -= (rho + C0*rho**3-rad)/(1+3*C0*rho**2)
    assert np.all(rho >= 0) and np.all(1+3*C0*rho**2 > 0)
    u = points * np.divide(rho, rad, out=np.ones_like(rad), where=rad > 0)[..., None]
    recon = u * (1+C0*np.sum(u*u, axis=-1, keepdims=True))
    assert np.max(np.abs(recon-points)) < 5e-13
    return u


def barrel(points, A):
    v = A / 4.
    s = 1. + S1*v
    k = sum(K[j]*v**j for j in range(5))
    u0 = radial_inverse(points)
    q = s[..., None, None] * u0[None, ...]
    return q*(1+k[..., None, None]*np.sum(q*q, axis=-1, keepdims=True))


def gaze_basis(ADE, BDE):
    tx, ty = BDE/20., ADE/20.
    terms = []
    labels = []
    groups = [
        ("m11", [(2,0),(0,2),(4,0),(2,2),(0,4)]),
        ("m22", [(2,0),(0,2),(4,0),(2,2),(0,4)]),
        ("g", [(2,0),(0,2),(4,0),(2,2),(0,4)]),
        ("m12", [(1,1),(3,1),(1,3)]),
        ("m21", [(1,1),(3,1),(1,3)]),
        ("qx", [(1,0),(3,0),(1,2)]),
        ("p2", [(1,0),(3,0),(1,2)]),
        ("qy", [(0,1),(2,1),(0,3)]),
        ("p1", [(0,1),(2,1),(0,3)]),
    ]
    for p, powers in groups:
        for i,j in powers:
            terms.append(tx**i * ty**j)
            labels.append(f"{p}:tx^{i}ty^{j}")
    assert len(labels) == 33
    return np.stack(terms, -1), labels


def predict(v, basis, coeff, base_kappa):
    # v is states x 9 x 2: scaled inverse-radial coordinates, normalized by L.
    # Basis is states x 33; coeff follows the explicit 33-entry labels.
    c = np.asarray(coeff).reshape(33)
    n = len(basis)
    pars = np.zeros((n, 9))
    spans=((0,5),(5,10),(10,15),(15,18),(18,21),(21,24),(24,27),(27,30),(30,33))
    for j,(lo,hi) in enumerate(spans):
        pars[:,j]=basis[:,lo:hi] @ c[lo:hi]
    pars[:,0]+=1.; pars[:,1]+=1.
    b = np.broadcast_to(np.asarray(v,float), (n, 9, 2))
    x,y=b[...,0],b[...,1]
    den=1+pars[:,None,5]*x+pars[:,None,7]*y
    ux=(pars[:,None,0]*x+pars[:,None,3]*y)/den
    uy=(pars[:,None,4]*x+pars[:,None,1]*y)/den
    p1,p2=pars[:,None,8],pars[:,None,6]
    wx=ux+2*p1*ux*uy+p2*(3*ux*ux+uy*uy)
    wy=uy+p1*(ux*ux+3*uy*uy)+2*p2*ux*uy
    w=np.stack((wx,wy),-1)
    kap=(np.asarray(base_kappa,float).reshape(n,1,1)+pars[:,None,2,None])
    w=w*(1.+kap*np.sum(w*w,axis=-1,keepdims=True))
    return w*L, den


def fit_step0(ADE, BDE, base_mm, base_kappa, target):
    bde, ade=np.meshgrid(BDE,ADE,indexing="ij")
    basis, labels = gaze_basis(ade.ravel(),bde.ravel())
    basis=basis.reshape(101,101,33)
    train=(np.arange(101)[:,None]%2==0)&(np.arange(101)[None,:]%2==0)
    test=~train
    outer=OUTER
    def residual(c, mask):
        bb=basis[mask]
        pred,_=predict(base_mm[None,:,:]/L,bb,c.reshape(33,1),np.full(len(bb),base_kappa))
        return ((pred[:,outer]-target[0,mask][:,outer])*1000.).ravel()
    res=least_squares(lambda c: residual(c,train), np.zeros(33), jac="2-point", x_scale="jac",
                      ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=300, verbose=0)
    if not res.success:
        raise RuntimeError(f"Step 1 optimizer failed: {res.message}")
    pred,_=predict(base_mm[None,:,:]/L,basis.reshape(-1,33),res.x.reshape(33,1),np.full(101*101,base_kappa))
    pred=pred.reshape(101,101,9,2)
    err=(pred-target[0])*1000.
    def metric(mask, fields):
        e=err[mask][:,fields]
        return [float(np.sqrt(np.mean(np.sum(e*e,axis=-1)))),float(np.max(np.linalg.norm(e,axis=-1)))]
    return res,labels,pred,err,{"train8":metric(train,outer),"test8":metric(test,outer),"test9":metric(test,np.arange(9))},train,test


def report_text(ADE,BDE,sha,sym,labels,res,metrics,zero_error):
    lines=["# P4 R1 radial-merge Step 1", "", f"Source grid SHA-256: `{sha}`. Grid axes: 21 accommodations (0–4 D, 0.2 D steps), BDE-major then ADE (101 each, −20° to +20°, 0.4° steps), nine fields. Center-relative real coordinates are formed by subtracting field 4 separately at each state, consistent with the stored combined-model convention.", "", "## Symmetry audit", "", "Signed residuals are reflected-state coordinate minus target-state coordinate, in µm: ADE maps `(ADE, fx, fy, x, y)` to `(-ADE, fx, -fy, x, -y)`; BDE maps `(BDE, fx, fy, x, y)` to `(-BDE, -fx, fy, -x, y)`. Values shown are max absolute component and component RMS across all states, fields and coordinates.", "", "| Coordinates | ADE max | ADE RMS | BDE max | BDE RMS |", "|---|---:|---:|---:|---:|"]
    for key,title in (("raw","Raw"),("centered","Center-relative")):
        q=sym[key]; lines.append(f"| {title} | {q['ADE_flip_y_max_abs_um']:.9g} | {q['ADE_flip_y_rms_component_um']:.9g} | {q['BDE_flip_x_max_abs_um']:.9g} | {q['BDE_flip_x_rms_component_um']:.9g} |")
    lines += ["", "## Model and Step 1 fit", "", f"`u0 = D_c0^-1(r0)`, `D_c(u)=(1+c|u|²)u`, `c0={C0:.10g} mm^-2` on the positive central branch. `v=s(A)u0/L`, `s(A)=1−0.081595476571(A/4)`, `L={L:.13f} mm`. `h=M(t)v/(1+qx vx+qy vy)`, `rho²=hx²+hy²`, `wx=hx+2p1 hx hy+p2(rho²+2hx²)`, `wy=hy+p1(rho²+2hy²)+2p2 hx hy`, then `r=L(1+kappa|w|²)w`, `kappa=k(A)L²+g(t)`. `k(A)=sum b_j(A/4)^j`; `b=[{', '.join(f'{v:.11g}' for v in K)}] mm^-2`. `tx=BDE/20`, `ty=ADE/20`.", "", "The coefficient functions are grouped dot products with these monomial lists: m11,m22,g use [tx²,ty²,tx⁴,tx²ty²,ty⁴]; m12,m21 use [tx ty,tx³ty,tx ty³]; qx,p2 use [tx,tx³,tx ty²]; qy,p1 use [ty,tx²ty,ty³]. Identity constants are 1 for m11/m22 and 0 otherwise.", "", "Gaze coefficient labels (33 total): `"+", ".join(labels)+"`.", "", f"A=0 training states: 2601; held-out: 7600. Fit status `{res.message}`, nfev={res.nfev}, cost={res.cost:.9g} µm²/2. Optimization used SciPy least_squares, numerical two-point Jacobian, x_scale='jac', tight tolerances. Eight noncenter fields are the objective.", "", "| Metric | Value (µm) |", "|---|---:|"]
    for name,val in metrics.items():
        lines.append(f"| {name} vertex RMS | {val[0]:.8f} |")
        lines.append(f"| {name} max point | {val[1]:.8f} |")
    lines += ["", f"Zero-gaze agreement with the prescribed `B_A(r0)` at every A: maximum absolute component error {zero_error:.3g} µm.", "", "A=0 gate: held-out eight-field RMS must be ≤2.2052 µm (stored reference 2.1552 µm plus 0.05 µm). If the gate fails, a one-time same-basis current-chain control is run and Step 2 stops. If it passes, this script records the gate only and waits for Step 2 orchestration; it does not claim or perform Step 2.", "", "Reference context, not recomputed: C1 order 2, 378 coefficients, 3.33 µm eight-field held-out RMS / 26.29 µm max; joint model, 252 coefficients, 9.03 µm all-nine held-out RMS / 53.2 µm max; frozen C0, 66.05 µm eight-field RMS. Step 2 R1-A0/R1-all tables are not run unless separately orchestrated after passing this gate. The 3.28 µm current-chain floor is not an R1 result.", "", "## All 38 independent model values", "", f"Fixed on-axis terms: s1={S1:.12g}; k1..k4=[{', '.join(f'{v:.12g}' for v in K[1:])}] mm^-2 in powers of A/4. Prescribed k0={C0:.12g} mm^-2 and L={L:.13f} mm. These five plus 33 gaze coefficients are the 38 model values.", "", "| # | coefficient | value |", "|---:|---|---:|"]
    for i,(label,value) in enumerate(zip(labels,res.x),1):
        lines.append(f"| {i} | {label} | {value:.12g} |")
    lines.append("")
    return "\n".join(lines)


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError("Refusing to overwrite existing output")
    OUT.mkdir(parents=True,exist_ok=True)
    A,ADE,BDE,raw,real,sha=load_grid()
    sym=symmetry(raw,real)
    r0=real[0,50,50]
    u0=radial_inverse(r0)
    # Zero gaze through the same predictor must reproduce analytic fixed B_A.
    zbasis,_=gaze_basis(np.zeros(1),np.zeros(1))
    zpred=[]
    for av in A:
        v=(1+S1*av/4)*u0/L
        k=sum(K[j]*(av/4)**j for j in range(5))
        p0,_=predict(v[None,:,:],zbasis,np.zeros((33,1)),np.array([k*L*L]))
        zpred.append(p0[0])
    zero=np.max(np.abs(np.asarray(zpred)-barrel(r0,A))*1000.)
    if zero>1e-8:
        raise RuntimeError(f"zero-gaze analytic composition check failed: {zero} um")
    if sym["centered"]["ADE_flip_y_max_abs_um"] > 1e-8 or sym["centered"]["BDE_flip_x_max_abs_um"] > 1e-8:
        raise RuntimeError("Centered source grid fails required reflection symmetry gate")
    res,labels,pred,err,metrics,train,test=fit_step0(ADE,BDE,u0,C0*L*L,real)
    control=None
    if metrics["test8"][0] > 2.2052:
        cres,_,cpred,cerr,cm,_,_=fit_step0(ADE,BDE,r0,0.,real)
        control={"metrics":cm,"prediction":cpred,"residual_um":cerr,"coefficients":cres.x}
    # Repeat zero-gaze verification with fitted coefficients through the same predictor.
    zpred_fit=[]
    for av in A:
        v=(1+S1*av/4)*u0/L
        k=sum(K[j]*(av/4)**j for j in range(5))
        p0,_=predict(v[None,:,:],zbasis,res.x.reshape(33,1),np.array([k*L*L]))
        zpred_fit.append(p0[0])
    zero_fit=np.max(np.abs(np.asarray(zpred_fit)-barrel(r0,A))*1000.)
    if zero_fit>1e-8:
        raise RuntimeError(f"fitted zero-gaze analytic check failed: {zero_fit} um")
    np.savez_compressed(OUT/"step1_a0.npz",A=A,ADE=ADE,BDE=BDE,fields=FIELDS,r0_mm=r0,u0_mm=u0,
        target_a0_mm=real[0],prediction_a0_mm=pred,residual_a0_um=err,train_mask=train,test_mask=test,
        coefficient_labels=np.array(labels),coefficients=res.x,metrics_json=np.array(json.dumps(metrics)),
        symmetry_json=np.array(json.dumps(sym)),grid_sha256=np.array(sha),fixed_k_coefficients=K,
        fixed_s_coefficient=np.array([S1]),L_mm=np.array(L),c0=np.array(C0),
        control_metrics_json=np.array(json.dumps(control["metrics"] if control else {})),
        control_coefficients=(control["coefficients"] if control else np.empty((0,))),
        control_prediction_mm=(control["prediction"] if control else np.empty((0,))),
        control_residual_um=(control["residual_um"] if control else np.empty((0,))))
    report=report_text(ADE,BDE,sha,sym,labels,res,metrics,max(zero,zero_fit))
    if control:
        report += "\n\n## Same-basis current-chain control\n\n"
        report += f"Control fitted at A=0 on r0/L with zero base radial coefficient: held-out eight-field vertex RMS {control['metrics']['test8'][0]:.8f} µm, max point {control['metrics']['test8'][1]:.8f} µm. Step 2 stopped because primary R1 gate exceeded 2.2052 µm.\n"
    (OUT/"report.md").write_text(report,encoding="utf-8")
    print(report)


if __name__=="__main__":
    main()
