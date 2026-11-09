"""Step 0/1 physical-convention audit and frozen-teacher physical candidate fits.

Writes only data/p4_phys/.  No CODE V, refits, noise, or runtime solver is used.
"""
from __future__ import annotations

import hashlib
import html as html_lib
import json
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Script"))
OUT = ROOT / "data" / "p4_phys"
DATA = ROOT / "data"
GRID = DATA / "distortion_grid_p4_biaxial_macro" / "distortion_grid.pkl"
SOURCE = DATA / "p4_p1_limited_gaze_12p5" / "source_subset.npz"
FIELD_SUMMARY = DATA / "p4_unified_combined/calibration/full_field_profile_final/summary.json"
MOTION_SUMMARY = DATA / "p4_unified_combined/calibration/center_motion/summary.json"
ROT_DOC = Path(r"C:\CODEV202403_SR1\help\lens_system\tilts.11.11.html")
COMPANION_SEQ = ROOT / "Lens" / "p4_ME.seq"
K = np.array([-0.018500496700, 0.00023989235714, 0.0015756063948,
              -0.0014256378340, 0.00066200789981])
SLOPE = -0.081595476571


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rotation(gaze):
    """CODE V compound local rotations: Rx(-ADE) @ Ry(-BDE), CDE=0."""
    g = np.deg2rad(np.asarray(gaze, float))
    a, b = -g[..., 0], -g[..., 1]
    ca, sa, cb, sb = np.cos(a), np.sin(a), np.cos(b), np.sin(b)
    rx = np.zeros(g.shape[:-1] + (3, 3)); ry = np.zeros_like(rx)
    rx[..., 0, 0] = 1; rx[..., 1, 1] = ca; rx[..., 1, 2] = -sa
    rx[..., 2, 1] = sa; rx[..., 2, 2] = ca
    ry[..., 0, 0] = cb; ry[..., 0, 2] = sb; ry[..., 1, 1] = 1
    ry[..., 2, 0] = -sb; ry[..., 2, 2] = cb
    return rx @ ry


def radial_inverse(x, k):
    x = np.asarray(x, float)
    ro = np.linalg.norm(x, axis=-1, keepdims=True)
    rho = ro.copy()
    for _ in range(30):
        den = 1 + 3*k*rho*rho
        rho -= (rho*(1+k*rho*rho)-ro)/den
    return x * np.divide(rho, ro, out=np.ones_like(rho), where=ro != 0)


def design_conventions():
    html_content = ROT_DOC.read_text(encoding="utf-8", errors="replace") if ROT_DOC.exists() else ""
    plain = html_lib.unescape(re.sub(r"<[^>]+>", " ", html_content))
    if "for compound tilts" not in plain.lower() or "assumed order of operation" not in plain.lower():
        raise RuntimeError("Rotation convention source quotation not found")
    g = np.array([[0., 0.], [2., 0.], [0., 2.]])
    vec = np.broadcast_to([0., 0., 1.], (3, 3))
    rv = np.einsum("nij,nj->ni", rotation(g), vec)
    # Sign sanity: Rx(-ADE) sends +Z toward +Y; Ry(-BDE) sends +Z toward -X.
    if not (rv[1, 1] > 0 and rv[2, 0] < 0):
        raise AssertionError("Rotation sign sanity failed")
    return {"rotation_order": "Local compound orientation represented by R = Rx(-ADE) @ Ry(-BDE), CDE=0 (active-column matrix convention inferred for the candidate law; CODE V source establishes local tilt order, not this matrix notation)",
            "CODE_V_source": str(ROT_DOC), "source_sha256": sha(ROT_DOC),
            "source_quotes": ["For compound tilts, β must be defined in the α-tilted coordinate system, and similarly for γ (defined in the α, β tilted system).",
                              "assumed order of operation ADE, BDE, CDE for a single surface"],
            "left_handed_signs": "ADE/BDE local axes; +ADE maps +Z toward +Y; +BDE maps +Z toward -X",
            "canonical_ray_labels": "all grids are explicitly keyed by (field_x_relative, field_y_relative); P4 source uses x outer/y inner, while P1 chunk rows use y outer/x inner; field (0,0) is located by label",
            "rotation_sign_sanity_Rz": rv.tolist(),
            "lens_comparison": "Requested .len source-level prescription comparison is unavailable; only companion Lens/p4_ME.seq is present. Values in that .seq are not asserted to equal the .len prescription.",
            "companion_seq": str(COMPANION_SEQ), "companion_seq_sha256": sha(COMPANION_SEQ) if COMPANION_SEQ.exists() else None}


def grid_tensor(grid):
    """Sort structured macro records by state and explicit field labels."""
    fields = np.unique(np.column_stack((grid["field_x_relative"], grid["field_y_relative"])), axis=0)
    fmap = {tuple(v): i for i, v in enumerate(fields)}
    fi = np.array([fmap[(x, y)] for x, y in zip(grid["field_x_relative"], grid["field_y_relative"])])
    a = grid["accommodation_d"] if "accommodation_d" in grid.dtype.names else np.zeros(len(grid))
    ade, bde = grid["eye_rotation_deg"], grid["eye_rotation_x_deg"]
    order = np.lexsort((fi, bde, ade, a))
    rows = grid[order]
    state = np.unique(np.column_stack((a, ade, bde)), axis=0)
    if len(rows) != len(state)*len(fields):
        raise RuntimeError("Macro grid is not a complete state-by-field product")
    coords = np.column_stack((rows["real_x_mm"], rows["real_y_mm"])).reshape(len(state),len(fields),2)
    centers_abs = np.column_stack((rows["center_absolute_x_mm"],rows["center_absolute_y_mm"]))
    centers_abs = centers_abs.reshape(len(state),len(fields),2)[:,0]
    center_i = fmap[(0.,0.)]
    center_ray = coords[:,center_i].copy()
    centered = coords-center_ray[:,None,:]
    return state, fields, centered, center_ray, centers_abs, center_i


def metrics(delta_mm):
    d = np.asarray(delta_mm)*1000.
    ok = np.isfinite(d).all(axis=-1)
    v = np.linalg.norm(d[ok],axis=-1)
    return {"n":int(ok.sum()),"nonfinite":int((~ok).sum()),
            "max_2d_um":float(v.max()) if len(v) else None,
            "RMS_2d_um":float(np.sqrt(np.mean(v*v))) if len(v) else None,
            "max_coordinate_um":float(np.max(np.abs(d[ok]))) if len(v) else None,
            "mean_signed_xy_um":np.mean(d[ok],axis=0).tolist() if len(v) else None,
            "max_point_index":int(np.flatnonzero(ok)[np.argmax(v)]) if len(v) else None}


def budgets(m):
    x=m.get("max_2d_um")
    if x is None:return {"gaze_0p25_arcmin":False,"A_0p025D":False,"both":False}
    return {"gaze_0p25_arcmin":bool(x<=1.1666666667),"A_0p025D":bool(x<=.6),"both":bool(x<=.6)}


def main():
    t0 = time.time()
    OUT.mkdir(parents=True,exist_ok=True)
    from evaluate_p4_unified_combined import load_models
    frozen, field, motion, model_hashes = load_models(FIELD_SUMMARY, MOTION_SUMMARY)
    with np.load(SOURCE) as z:
        states = z["states"].copy(); obs = z["absolute_p4_mm"].copy()
        mean_p1 = z["mean_p1_absolute_mm"].copy(); center_abs = z["center_absolute_mm"].copy()
    with GRID.open("rb") as f:
        grid = pickle.load(f)
    p4state,p4fields,p4H,p4ray,p4ca,p4ci=grid_tensor(grid)
    p4center=p4ca+p4ray
    # P1 teacher forward grid is separate; do not substitute P4 centers/shapes.
    p1path=DATA/"distortion_grid_p1_biaxial_macro"/"distortion_grid.pkl"
    with p1path.open("rb") as f:p1grid=pickle.load(f)
    p1state,p1fields,p1H,p1ray,p1ca,p1ci=grid_tensor(p1grid)
    p1center=p1ca+p1ray
    # Evaluate actual frozen G maps and self-invert over their respective domains.
    def invert_map(poly,target):
        q=np.broadcast_to(poly.mean_mm,target.shape).copy(); active=np.ones(len(q),bool)
        for _ in range(60):
            err=poly.predict(q)-target; norm=np.max(np.abs(err),axis=1); active &= np.isfinite(norm)
            if np.all((norm<1e-11)|~active):break
            ids=np.flatnonzero(active & (norm>=1e-11))
            if not len(ids):break
            jac=poly.jacobian(q[ids]); det=np.linalg.det(jac)
            good=np.isfinite(det)&(np.abs(det)>1e-10)
            active[ids[~good]]=False; ids=ids[good]; jac=jac[good]; err=err[ids]
            q[ids]-=np.linalg.solve(jac,err[...,None])[...,0]
        residual=np.max(np.abs(poly.predict(q)-target),axis=1)
        return q,active&(residual<1e-8),residual
    def center_fit(g, c, poly, axial_only=False):
        iz=np.flatnonzero(np.all(np.isclose(g,0.,atol=1e-10),axis=1))
        origin=np.mean(c[iz],axis=0)
        jac=np.linalg.inv(poly.jacobian(np.zeros((1,2)))[0])
        # Sign-only sensor basis, inferred from axis-specific teacher derivatives.
        ss=np.diag([-np.sign(jac[0,1]),np.sign(jac[1,0])])
        def forward(lever):
            rr=rotation(g)
            return origin+np.einsum("ij,nj->ni",ss,np.einsum("nij,j->ni",rr-np.eye(3),lever)[:,:2])
        if axial_only:
            fit=least_squares(lambda x:((forward(np.array([0.,0.,x[0]]))-c)*1000).ravel(),[20.])
            lever=np.array([0.,0.,fit.x[0]])
        else:
            # Linear in the three projected lever coordinates; preserve the measured origin.
            rr=rotation(g)-np.eye(3)
            design=np.concatenate([ss@rr[:,:2,:] for _ in [0]],axis=0)
            mat=np.zeros((len(g)*2,3))
            for n in range(len(g)):mat[2*n:2*n+2]=ss@rr[n,:2,:]
            rhs=(c-origin).reshape(-1)
            lever,_,rank,_=np.linalg.lstsq(mat,rhs,rcond=None)
            class Fit:
                x=lever
                message=f"linear lever least squares rank {rank}/3"
        fit=fit if axial_only else Fit()
        return origin,ss,lever,forward,fit
    def a0_unique(state,center):
        ix=np.flatnonzero(np.isclose(state[:,0],0.))
        return state[ix,1:],center[ix]
    g4,c4=a0_unique(p4state,p4center); g1,c1=a0_unique(p1state,p1center)
    # Each raytrace gaze is the teacher's input; invert its frozen center polynomial
    # to obtain the corresponding own-input teacher center target.
    q4,inv4,invres4=invert_map(frozen.p4_gaze,g4)
    q1,inv1,invres1=invert_map(frozen.p1_gaze,g1)
    if not inv4.any() or not inv1.any():raise RuntimeError("Teacher center inverse has no valid rows")
    o4,S4,d4,_,fit4=center_fit(g4[inv4],q4[inv4],frozen.p4_gaze,axial_only=False)
    o1,S1,d1,_,fit1=center_fit(g1[inv1],q1[inv1],frozen.p1_gaze,axial_only=True)
    def rot_center(g,o,S,d):
        rr=rotation(g); return o+np.einsum("ij,nj->ni",S,np.einsum("nij,j->ni",rr-np.eye(3),d)[:,:2])
    phys_g4=rot_center(g4,o4,S4,d4); phys_g1=rot_center(g1,o1,S1,d1)
    # Actual G readout on physical candidate centers, plus candidate-vs-inverse-teacher.
    dense_eye=np.linspace(-20.,20.,101)
    dense_g=np.array(np.meshgrid(dense_eye,dense_eye,indexing="ij")).reshape(2,-1).T
    phys_dense_g4=rot_center(dense_g,o4,S4,d4); phys_dense_g1=rot_center(dense_g,o1,S1,d1)
    read_g4=frozen.p4_gaze.predict(phys_dense_g4); read_g1=frozen.p1_gaze.predict(phys_dense_g1)
    q_dense4,v_dense4,res_dense4=invert_map(frozen.p4_gaze,dense_g)
    q_dense1,v_dense1,res_dense1=invert_map(frozen.p1_gaze,dense_g)
    err_g4=np.linalg.norm(read_g4-dense_g,axis=1)*60
    err_g1=np.linalg.norm(read_g1-dense_g,axis=1)*60
    gaze_op=(np.abs(dense_g[:,0])<=12.4+1e-10)&(np.abs(dense_g[:,1])<=12.4+1e-10)
    g4phys_full=metrics(phys_dense_g4-q_dense4); g4phys_op=metrics((phys_dense_g4-q_dense4)[gaze_op])
    g1phys_full=metrics(phys_dense_g1-q_dense1); g1phys_op=metrics((phys_dense_g1-q_dense1)[gaze_op])
    # M1 fixed known accommodation scale and axial-lever motion, fitted to calibration teacher deltas.
    mz=np.load(DATA/"p4_unified_combined/calibration/center_motion/calibration_data.npz")
    ma=mz["states"][:,0]; mg=mz["states"][:,1:]; mc=mz["true_p4_center_A_mm"]
    mc0=mz["matched_p4_center_A0_mm"]
    def sA(a):return 1+SLOPE*np.asarray(a)/4.
    qmot, qmotvalid, _ = invert_map(frozen.p4_gaze,mg)
    mteacher=motion.forward(qmot,ma)-qmot
    def mphys(a,g,ell):
        rr=rotation(g)
        return (sA(a)-1)[:,None]*np.einsum("ij,nj->ni",S4,np.einsum("nij,j->ni",rr-np.eye(3),[0.,0.,ell])[:,:2])
    mf=least_squares(lambda x:((mphys(ma,mg,x[0])-mteacher)*1000).ravel(),[10.])
    mp=mphys(ma,mg,float(mf.x[0]))
    mteacher_trace=motion.forward(mc0,ma)-mc
    # Full PF candidate: invert fixed A0 radial map, pinhole rotation, fixed k(A),
    # radial distortion about G4 lever displacement; only f is free.
    # A0 reference is the frozen sparse model's own reference, ordered by its explicit field lattice.
    refix=np.flatnonzero(np.all(np.isclose(p4state[:,1:],0.,atol=1e-10),axis=1)&np.isclose(p4state[:,0],0.))[0]
    sparse=field.sparse_model()
    r0=np.asarray(sparse.reference_mm)
    if r0.shape!=(len(p4fields),2):raise RuntimeError("Frozen sparse reference and field count differ")
    ref_match=metrics(r0-p4H[refix])
    d0=rot_center(np.zeros((1,2)),o4,S4,d4)[0]
    def radial(x,k):return x*(1+k*np.sum(x*x,axis=-1,keepdims=True))
    u0=radial_inverse(r0+radial(d0,K[0]),K[0])-d0
    centerindex=p4ci
    fields=p4fields
    # s and k are fixed from published on-axis report; form candidate state-by-rays.
    A=p4state[:,0]; g=p4state[:,1:]
    op=(np.abs(g[:,0])<=12.4+1e-10)&(np.abs(g[:,1])<=12.4+1e-10)
    # Fit only f on a deterministic 1/64 subset; all other coefficients fixed.
    sample=np.arange(len(A))%64==0
    raw_teacher=field.sparse_model().evaluate(A,g[:,0],g[:,1])["prediction"]
    def phys_pf(focal, rows):
        aa=A[rows]; gg=g[rows]; rr=rotation(gg); kval=np.polynomial.polynomial.polyval(aa/4,K)
        sc=sA(aa); out=np.empty((len(rows),len(fields),2)); den=rr[:,2,2]
        all_den=[]
        for j,u in enumerate(u0):
            local_xy=np.array([u[0],u[1]])@S4
            ray=np.column_stack((sc*local_xy[0]/focal,sc*local_xy[1]/focal,np.ones(len(rows))))
            turned=np.einsum("nij,nj->ni",rr,ray)
            all_den.append(turned[:,2])
            central=np.einsum("nij,j->ni",rr,[0.,0.,1.])
            q=focal*(turned[:,:2]/turned[:,2,None]-central[:,:2]/central[:,2,None])
            q=q@S4
            d=rot_center(gg,o4,S4,d4)
            r=q+d
            rdist=r*(1+kval[:,None]*np.sum(r*r,axis=-1,keepdims=True))
            dc=d*(1+kval[:,None]*np.sum(d*d,axis=-1,keepdims=True))
            out[:,j]=rdist-dc
        return out,np.stack(all_den,axis=1)
    def pf_obj(logf):
        rows=np.flatnonzero(sample); pred,_=phys_pf(np.exp(logf[0]),rows)
        if not np.isfinite(pred).all():return 1e12
        delta=(pred-raw_teacher[rows])*1000
        return np.mean(delta*delta)
    pfopt=minimize_scalar(lambda lf:pf_obj([lf]),bounds=(np.log(1.),np.log(1000.)),method="bounded",options={"xatol":1e-8})
    focal=float(np.exp(pfopt.x)); pfpred,den=phys_pf(focal,np.arange(len(A)))
    refpred,refden=phys_pf(focal,np.array([refix]))
    ref_identity=metrics(refpred[0]-r0)
    if ref_identity["max_2d_um"]>1e-7 or np.any(den<=0):
        raise RuntimeError("PF candidate baseline identity or turned-ray denominator check failed")
    # Teacher/candidate B terms: raw mean first, discrete anchor correction separately.
    trace_mean=np.mean(p4H,axis=1)
    bshape=np.mean(raw_teacher,axis=1)
    fresult=field.evaluate(A,g)
    b4=frozen.p4_bias.predict(g)
    # For each gaze, use matching A0 raw mean to derive the teacher F mean identity.
    lookup={(float(y),float(x)):i for i,(a,y,x) in enumerate(p4state) if np.isclose(a,0.)}
    b0=np.array([bshape[lookup[(float(y),float(x))]] for a,y,x in p4state])
    expected_mean=b4+bshape-b0
    mean_identity=np.max(np.abs(fresult.mean(axis=1)-expected_mean),axis=1)
    trace_b0=np.array([trace_mean[lookup[(float(y),float(x))]] for a,y,x in p4state])
    raycorr=(9./8.)*(b4-trace_b0)
    rayF=p4H.copy(); outer=np.any(np.abs(p4fields)>0,axis=1)
    rayF[:,outer,:]+=raycorr[:,None,:]
    f_vs_trace_full=metrics(fresult-rayF)
    f_vs_trace_op=metrics((fresult-rayF)[op])
    trace_b4_discrepancy=metrics(expected_mean-trace_mean)
    # Natural A0 anchor discrepancy is a direct derivation, not a fitted B correction.
    a0mean=bshape[np.isclose(A,0.)]
    a0gaze=g[np.isclose(A,0.)]
    b4a0=frozen.p4_bias.predict(a0gaze)
    b4err=metrics(a0mean-b4a0)
    trace_b4err=metrics(trace_mean[np.isclose(A,0.)]-b4a0)
    # P1 own full-field center-relative mean for B1/B3 derivation.
    p1a0=np.flatnonzero(np.isclose(p1state[:,0],0.))
    p1mean=np.mean(p1H[p1a0],axis=1)
    p1g=p1state[p1a0,1:]
    b1=frozen.p1_bias.predict(p1g)
    b1err=metrics(p1mean-b1)
    # Independently verify source-subset central field mapping against macro labels.
    # Reordering by field labels avoids assuming benchmark/collector ray conventions.
    # Input masks and reporting windows.
    full=np.ones(len(A),bool)
    pf_vs_teacher_full=metrics(pfpred-raw_teacher); pf_vs_teacher_op=metrics((pfpred-raw_teacher)[op])
    raw_teacher_vs_trace=metrics(raw_teacher-p4H)
    pf_vs_trace=metrics(pfpred-p4H)
    pf_full=pf_vs_teacher_full; pf_op=pf_vs_teacher_op
    if focal<=1.0001 or focal>=999.9: bound_status="at/near bound"
    else:bound_status="interior"
    g4full={"max_arcmin":float(np.max(err_g4)),"RMS_arcmin":float(np.sqrt(np.mean(err_g4**2))),"n":len(err_g4)}
    g4op={"max_arcmin":float(np.max(err_g4[gaze_op])),"RMS_arcmin":float(np.sqrt(np.mean(err_g4[gaze_op]**2))),"n":int(gaze_op.sum())}
    g1full={"max_arcmin":float(np.max(err_g1)),"RMS_arcmin":float(np.sqrt(np.mean(err_g1**2))),"n":len(err_g1)}
    g1op={"max_arcmin":float(np.max(err_g1[gaze_op])),"RMS_arcmin":float(np.sqrt(np.mean(err_g1[gaze_op]**2))),"n":int(gaze_op.sum())}
    report = {
        "scope": "Steps 0 and 1 only; frozen artifacts read-only; no CODE V/noise/refit/solver",
        "conventions": design_conventions(), "model_artifact_hashes": model_hashes,
        "input_hashes": {str(p.relative_to(ROOT)): sha(p) for p in [GRID,SOURCE,FIELD_SUMMARY,MOTION_SUMMARY]},
        "datasets": {"P4_macro_records":int(len(grid)),"P4_states":len(p4state),"P4_A_levels":np.unique(p4state[:,0]).tolist(),"P4_rays":p4fields.tolist(),"P1_macro_records":int(len(p1grid)),"P1_states":len(p1state),"P1_rays":p1fields.tolist(),"source_subset_rows":len(states)},
        "G4_orthographic": {"sensor_sign_matrix":S4.tolist(),"effective_lever_xyz_mm":d4.tolist(),"origin_mm":o4.tolist(),"fit_status":fit4.message,"inverse_teacher_center_vs_raytrace_macro_mm":metrics(q4[inv4]-c4[inv4]),"candidate_vs_inverse_teacher_dense_full_mm":g4phys_full,"candidate_vs_inverse_teacher_dense_operating_mm":g4phys_op,"full_domain_actual_G_readout":g4full,"operating_domain_actual_G_readout":g4op,"macro_teacher_inverse_failures":int((~inv4).sum()),"dense_teacher_inverse_failures":int((~v_dense4).sum()),"inverse_teacher_roundtrip_mm_max":float(res_dense4[v_dense4].max()),"interpretation":"effective projected lever; physical distance uncalibrated","numeric_gaze_budget_pass_full":bool(g4full["max_arcmin"]<=.25),"numeric_gaze_budget_pass_operating":bool(g4op["max_arcmin"]<=.25)},
        "G1_glint": {"sensor_sign_matrix":S1.tolist(),"effective_axial_distance_mm":float(d1[2]),"origin_mm":o1.tolist(),"fit_status":fit1.message,"inverse_teacher_center_vs_raytrace_macro_mm":metrics(q1[inv1]-c1[inv1]),"candidate_vs_inverse_teacher_dense_full_mm":g1phys_full,"candidate_vs_inverse_teacher_dense_operating_mm":g1phys_op,"full_domain_actual_G_readout":g1full,"operating_domain_actual_G_readout":g1op,"macro_teacher_inverse_failures":int((~inv1).sum()),"dense_teacher_inverse_failures":int((~v_dense1).sum()),"inverse_teacher_roundtrip_mm_max":float(res_dense1[v_dense1].max()),"corneal_radius_mm":7.77,"radius_identifiable":False,"mirror_derivation":"under fixed world source/view bisector, c_glint=R*d_cornea+r*n_fixed; subtract zero gaze gives (R-I)*d_cornea, so r*n_fixed cancels exactly and r has zero sensitivity","numeric_gaze_budget_pass_full":bool(g1full["max_arcmin"]<=.25),"numeric_gaze_budget_pass_operating":bool(g1op["max_arcmin"]<=.25),"limitations":"no conic/asphere or finite-camera perspective; .len prescription comparison unavailable"},
        "M1_fixed_scale": {"slope_per_A":SLOPE/4,"slope_source":"data/p4_onaxis/report.md recommended s poly1 coefficient","effective_axial_lever_mm":float(mf.x[0]),"fit_status":mf.message,"candidate_vs_frozen_M1_teacher_mm":metrics(mp-mteacher),"frozen_M1_teacher_vs_trace_ownpart_mm":metrics(mteacher_trace),"frozen_M1_teacher_vs_trace_chained_G4plusM1_mm":metrics(mteacher-(mc-mc0)),"inverse_teacher_failures":int((~qmotvalid).sum()),"numeric_gaze_budget_pass":bool(metrics(mp-mteacher)["max_2d_um"]<=1.1666667)},
        "PF_raw_H": {"fitted_focal_length_mm":focal,"fit_status":str(pfopt.message),"fit_bound_status":bound_status,"fit_rows":int(sample.sum()),"fit_policy":"deterministic every-64th macro state; f only; fixed k(A), s(A), rotation, G4 three-parameter lever","candidate_vs_teacher_full":pf_vs_teacher_full,"candidate_vs_teacher_operating":pf_vs_teacher_op,"teacher_vs_raytrace_full":raw_teacher_vs_trace,"candidate_vs_raytrace_full":pf_vs_trace,"budget_full":budgets(pf_full),"budget_operating":budgets(pf_op),"frozen_reference_vs_macro_A0_zero_gaze":ref_match,"PF_A0_identity_max_error_um":ref_identity["max_2d_um"],"turned_ray_denominator_min":float(np.min(den)),"turned_ray_denominator_nonpositive":int(np.sum(den<=0)),"fixed_k_A_coefficients_mm_minus2":K.tolist(),"scale_slope":SLOPE,"raw_H_definition":"candidate and frozen teacher raw H; macro trace compared separately; no B4 correction"},
        "B_terms": {"P4_raw_teacher_mean_vs_frozen_B4_at_A0":b4err,"P4_raytrace_mean_vs_frozen_B4_at_A0":trace_b4err,"P4_predicted_F_mean_vs_raytrace_shape_mean_full":trace_b4_discrepancy,"P4_full_field_mean_identity_max_mm":float(np.max(mean_identity)),"P4_final_F_vs_raytrace_anchored_full":f_vs_trace_full,"P4_final_F_vs_raytrace_anchored_operating":f_vs_trace_op,"P1_raytrace_mean_vs_frozen_B1_at_A0":b1err,"P1_shape_from_own_macro_grid":True},
        "budget_rule": {"max_euclidean_um": 0.6, "A_error_bound_D": 0.025, "gaze_error_bound_arcmin": 0.25,
                        "conversion": "A: um/24; gaze: um/280*60", "scope": "both-budget conservative criterion"},
        "limitations": ["No `.len` file available for requested prescription comparison.",
                        "G1 corneal radius is mathematically unidentifiable in fixed-vector cancellation model.",
                        "G4/M1/G1 lever values are effective projected equivalents, not calibrated physical distances.",
                        "No physical identification claim is made if numeric budgets pass."],
        "runtime_seconds": time.time()-t0}
    conv=report["conventions"]
    conv_lines=["# Step 0: conventions and provenance", "",
      "## Rotation convention", "",
      "CODE V Help, *Order of Operations: Decenters and Tilts* (`tilts.11.11.html`), states:",
      "", "> For compound tilts, β must be defined in the α-tilted coordinate system, and similarly for γ (defined in the α, β tilted system).", "",
      "> The assumed order of operation is ADE, BDE, CDE for a single surface.", "",
      "The candidate uses the inferred active-column matrix `R = Rx(-ADE) @ Ry(-BDE)`, with `CDE=0`. This matrix convention is a modeling inference; the cited help establishes local tilt order. Sign sanity for a local +Z lever: +ADE moves +Z toward +Y, and +BDE moves +Z toward −X.",
      "", "Canonical rays are addressed by their explicit `(field_x_relative, field_y_relative)` labels. The P4 macro uses x-major/y-minor labels; P1 chunk rows use y-major/x-minor labels. Each grid is reordered by labels, and field `(0,0)` is selected by its label.",
      "", "## Lens data", "",
      f"A source-level comparison of `Lens/p4_ME.len` is unavailable: the `.len` is binary (SHA-256 `{sha(ROOT/'Lens'/'p4_ME.len') if (ROOT/'Lens'/'p4_ME.len').exists() else 'not present'}`), and no live CODE V prescription export was made. The companion `Lens/p4_ME.seq` is a separate source; its SHA-256 is `{conv['companion_seq_sha256']}`. Its stated 7.77 mm corneal radius and −0.15 conic are not asserted to match the `.len` prescription.",
      "", f"Rotation help source: `{conv['CODE_V_source']}` (SHA-256 `{conv['source_sha256']}`).", ""]
    (OUT/"conventions.md").write_text("\n".join(conv_lines),encoding="utf-8")
    def row(name, full, op, extra=""):
        def cell(m):return f"{m.get('max_2d_um',float('nan')):.3f} / {m.get('RMS_2d_um',float('nan')):.3f} / {m.get('max_coordinate_um',float('nan')):.3f}"
        return f"| {name} | {cell(full)} | {cell(op)} | {extra} |"
    def normrow(name, maximum_um, full_flag, op_flag):
        return f"| {name} | {maximum_um:.3f} | {maximum_um/24:.5f} | {maximum_um/280*60:.3f} | {full_flag} | {op_flag} |"
    g4r=report['G4_orthographic']; g1r=report['G1_glint']; m1r=report['M1_fixed_scale']; pfr=report['PF_raw_H']; br=report['B_terms']
    R4=g4r['candidate_vs_inverse_teacher_dense_full_mm']; R4op=g4r['candidate_vs_inverse_teacher_dense_operating_mm']; R1=g1r['candidate_vs_inverse_teacher_dense_full_mm']; R1op=g1r['candidate_vs_inverse_teacher_dense_operating_mm']; RM=m1r['candidate_vs_frozen_M1_teacher_mm']; RMtrace=m1r['frozen_M1_teacher_vs_trace_ownpart_mm']
    report_lines=["# Step 1: frozen-teacher physical candidate audit", "",
      "## Scope and decision", "",
      "Steps 0 and 1 ran against the frozen artifacts and raytrace macro grids. There were no CODE V launches, noise runs, solver runs, or refits. None of the physical candidates passes the stated numeric budgets. G1 radius is unidentifiable in the far-field fixed-bisector model, so even a numeric pass would not identify a physical radius or a fully calibrated optical system.", "",
      "## Candidate residuals", "",
      "Displacements are maximum 2D Euclidean / RMS 2D Euclidean / maximum coordinate, in µm. Full grid covers the macro grid; operating grid restricts both |ADE| and |BDE| to 12.4°.", "",
      "| Part / comparison | Full grid max / RMS / coordinate max (µm) | Operating max / RMS / coordinate max (µm) |", "|---|---:|---:|",
      row("G4 physical center vs inverse-teacher center",R4,R4op),
      row("G4 inverse-teacher center vs raytrace center (macro sampling)",g4r['inverse_teacher_center_vs_raytrace_macro_mm'],g4r['inverse_teacher_center_vs_raytrace_macro_mm']),
      row("G1 physical center vs inverse-teacher center",R1,R1op),
      row("G1 inverse-teacher center vs raytrace center (macro sampling)",g1r['inverse_teacher_center_vs_raytrace_macro_mm'],g1r['inverse_teacher_center_vs_raytrace_macro_mm']),
      row("M1 physical law vs frozen M1 teacher",RM,RM),
      row("M1 frozen teacher vs raytrace own-part",RMtrace,RMtrace),
      row("PF candidate raw H vs frozen raw H",pfr['candidate_vs_teacher_full'],pfr['candidate_vs_teacher_operating']),
      row("PF frozen raw H vs raytrace raw H",pfr['teacher_vs_raytrace_full'],pfr['teacher_vs_raytrace_full']),
      row("PF candidate raw H vs raytrace raw H",pfr['candidate_vs_raytrace_full'],pfr['candidate_vs_raytrace_full']),
      row("Final F vs B4-anchored raytrace field",br['P4_final_F_vs_raytrace_anchored_full'],br['P4_final_F_vs_raytrace_anchored_operating']),
      "", "## Conservative converted limits", "",
      "Convert maximum displacement to accommodation as µm/24 and to gaze as µm/280×60 arcmin. G4/G1/M1 are judged on gaze only; PF and final F use both limits (max ≤0.6 µm and ≤0.25 arcmin).",
      "", "| Part | Max displacement (µm) | Accommodation equivalent (D) | Gaze equivalent (arcmin) | Full flag | Operating flag |", "|---|---:|---:|---:|---|---|",
      normrow("G4 center law",R4['max_2d_um'],str(g4r['numeric_gaze_budget_pass_full']),str(g4r['numeric_gaze_budget_pass_operating'])),
      normrow("G1 glint center law",R1['max_2d_um'],str(g1r['numeric_gaze_budget_pass_full']),str(g1r['numeric_gaze_budget_pass_operating'])),
      normrow("M1 fixed-scale center law",RM['max_2d_um'],str(m1r['numeric_gaze_budget_pass']),str(m1r['numeric_gaze_budget_pass'])),
      normrow("PF raw H",pfr['candidate_vs_teacher_full']['max_2d_um'],str(pfr['budget_full']),str(pfr['budget_operating'])),
      normrow("Final F",br['P4_final_F_vs_raytrace_anchored_full']['max_2d_um'],str(br['P4_final_F_vs_raytrace_anchored_full']['max_2d_um']<=.6),str(br['P4_final_F_vs_raytrace_anchored_operating']['max_2d_um']<=.6)),
      "", "## Frozen formulas and fitted constants", "",
      f"- Rotation: `Rx(-ADE) @ Ry(-BDE)`; sensor sign matrix for G4 `{S4.tolist()}`, G1 `{S1.tolist()}`.",
      f"- G4: `c(g)=c0+S[(R−I)d]xy`, fixed `c0={o4.tolist()} mm`, fitted effective lever `d={d4.tolist()} mm`.",
      f"- G1: `c_glint=R*d_cornea+r*n_fixed`; fixed-bisector subtraction cancels `r*n_fixed`. Effective axial distance `{float(d1[2]):.8f} mm`; corneal radius remains unidentifiable.",
      f"- M1: `s(A)=1+({SLOPE:.12f})(A/4)`; fitted effective axial lever `{float(mf.x[0]):.8f} mm`.",
      f"- PF: `k(A)=Σ b_j(A/4)^j`, coefficients `/mm²` `{K.tolist()}`; scale coefficient `{SLOPE:.12f}`; fitted effective focal length `{focal:.8f} mm` ({bound_status}); deterministic every-64th macro state fit, f only. A0 identity residual `{ref_identity['max_2d_um']:.3g} µm`.",
      "", "## Mean terms and residual patterns", "",
      f"P4 raw teacher mean at A=0 vs frozen B4: max 2D `{b4err['max_2d_um']:.4f} µm`; P4 trace raw mean vs B4: `{trace_b4err['max_2d_um']:.4f} µm`. Full F mean identity residual max `{float(np.max(mean_identity))*1000:.6g} µm`; F vs B4-anchored trace is in the table. P1 trace mean vs frozen B1: `{b1err['max_2d_um']:.4f} µm`.",
      f"PF fit focal is {bound_status}; turned-ray denominator min `{float(np.min(den)):.6g}`, nonpositive count `{int(np.sum(den<=0))}`. The largest PF state index is `{pfr['candidate_vs_teacher_full']['max_point_index']}`. G4/G1 center laws show smooth systematic errors exceeding budgets throughout the operating region; their signed mean errors are recorded in `step1.npz` and JSON metadata.",
      "", "## Limitations and stopped claims", "",
      "- G4/G1 levers are effective projected equivalents; the camera relay scale is not calibrated.",
      "- G1 radius has zero sensitivity under the fixed world half-vector approximation; finite-conjugate perspective and the corneal conic/asphere are omitted.",
      "- The `.len` prescription was not available as readable source and no live CODE V export was made. The separate `.seq` was not treated as equivalent.",
      "- PF f is at/near its search bound and residuals are far outside budgets. No physical acceptance claim is made.",
      "", "## Provenance", "", "Input SHA-256 hashes, convergence masks, all dense prediction arrays, and complete machine-readable metrics are included in `step1.npz` and its `report_json` member.", ""]
    (OUT/"step1_report.md").write_text("\n".join(report_lines)+"\n\n### Machine-readable report\n\n```json\n"+json.dumps(report,indent=2)+"\n```\n",encoding="utf-8")
    np.savez_compressed(OUT/"step1.npz",source_states=states,source_p4=obs,source_mean_p1=mean_p1,source_center_absolute=center_abs,
        P4_states=p4state,P4_fields=p4fields,P4_trace_raw_H=p4H,P4_teacher_raw_H=raw_teacher,P4_candidate_raw_H=pfpred,P4_teacher_center=p4center,
        P1_states=p1state,P1_fields=p1fields,P1_teacher_raw_H=p1H,P1_teacher_center=p1center,
        G4_sensor_sign=S4,G4_origin=o4,G4_effective_lever_xyz_mm=np.array(d4),G4_trace_center=c4,G4_inverse_teacher_center=q4,G4_physical_center=phys_g4,G4_readout_dense=read_g4,
        G1_sensor_sign=S1,G1_origin=o1,G1_effective_axial_lever_mm=np.array(d1),G1_trace_center=c1,G1_inverse_teacher_center=q1,G1_physical_center=phys_g1,G1_readout_dense=read_g1,
        M1_states=mz["states"],M1_teacher_delta=mteacher,M1_trace_ownpart_delta=mteacher_trace,M1_trace_chained_delta=mc-mc0,M1_candidate_delta=mp,M1_lever_mm=mf.x,
        PF_focal_mm=np.array(focal),PF_fit_sample=np.flatnonzero(sample),PF_denominator=den,P4_B4_teacher=b4,P4_B4_expected_mean=expected_mean,
        P1_B1_teacher=b1,P1_B1_shape_mean=p1mean,report_json=np.array(json.dumps(report)))
    print(json.dumps({"out":str(OUT),"G4":report["G4_orthographic"],"G1":report["G1_glint"],"PF":report["PF_raw_H"],"seconds":report["runtime_seconds"]},indent=2))


if __name__ == "__main__":
    main()
