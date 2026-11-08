"""Local one-frame Cramer-Rao floors from the P4 biaxial macro grid."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/distortion_grid_p4_biaxial_macro/distortion_grid.pkl"
OUT = ROOT / "data/p4_cr_floor"
SIGMAS = np.array([0.0010, 0.0011])
STEPS = np.array([0.2, 0.4, 0.4])
RANK_REL_TOL = 18 * np.finfo(float).eps


def stats(x):
    if np.isfinite(x).all():
        return (*np.percentile(x, [50, 95]), np.max(x))
    z = np.sort(x); out=[]
    for q in (0.5, 0.95, 1.0):
        t=q*(len(z)-1); a=int(np.floor(t)); b=int(np.ceil(t))
        out.append(z[a] if a==b else (np.inf if not np.isfinite(z[b]) else z[a]+(t-a)*(z[b]-z[a])))
    return tuple(out)


def difference(xyz, axis, h, offset):
    """Vectorized central/one-sided two-point derivative and 0/1/2 mask."""
    naxis=xyz.shape[axis]
    pos=np.arange(naxis)
    central=(pos>=offset)&(pos+offset<naxis)
    mode=np.where(central,0,np.where(pos+offset<naxis,1,2)).astype(np.uint8)
    lo=np.where(mode==1,pos,np.where(mode==2,pos-offset,np.maximum(pos-offset,0)))
    hi=np.where(mode==1,pos+offset,np.where(mode==2,pos,np.minimum(pos+offset,naxis-1)))
    lv=np.take(xyz,lo,axis=axis); hv=np.take(xyz,hi,axis=axis)
    den=(hi-lo)*h
    shape=[1]*xyz.ndim; shape[axis]=naxis
    deriv=(hv-lv)/den.reshape(shape)
    deriv=deriv.reshape(-1,9,2).reshape(-1,18)
    # Central denominator is already 2*offset*h; one-sided denominator is offset*h.
    mask=np.broadcast_to(mode.reshape(shape[:3]),xyz.shape[:3]).reshape(-1)
    return deriv,mask


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with DATA.open("rb") as f: rows=pickle.load(f)
    A,ai=np.unique(rows["accommodation_d"],return_inverse=True)
    ade,di=np.unique(rows["eye_rotation_deg"],return_inverse=True)
    bde,bi=np.unique(rows["eye_rotation_x_deg"],return_inverse=True)
    fx,xi=np.unique(rows["field_x_relative"],return_inverse=True)
    fy,yi=np.unique(rows["field_y_relative"],return_inverse=True)
    dims=(len(A),len(ade),len(bde),len(fx),len(fy))
    if dims!=(21,101,101,3,3) or len(rows)!=1_927_989: raise ValueError(f"Unexpected grid: {dims}, rows={len(rows)}")
    if not (np.allclose(A,np.linspace(0,4,21)) and np.allclose(ade,np.linspace(-20,20,101)) and np.allclose(bde,np.linspace(-20,20,101))): raise ValueError("Grid axes mismatch")
    key=(((ai*101+di)*101+bi)*9+xi*3+yi)
    if np.unique(key).size!=len(rows): raise ValueError("Duplicate or missing state-field entries")
    absolute=np.stack((rows["center_absolute_x_mm"]+rows["real_x_mm"],rows["center_absolute_y_mm"]+rows["real_y_mm"]),axis=1)
    nonfinite=int((~np.isfinite(absolute)).sum())
    xyz=np.empty((len(rows),2)); xyz[key]=absolute; xyz=xyz.reshape(21,101,101,9,2)
    if not np.isfinite(xyz).all(): raise ValueError("Nonfinite or missing absolute positions")
    n=21*101*101
    J=np.empty((n,18,3)); J2=np.empty_like(J); masks=np.empty((n,3),np.uint8); masks2=np.empty_like(masks)
    for axis,h in enumerate(STEPS):
        J[:,:,axis],masks[:,axis]=difference(xyz,axis,h,1)
        J2[:,:,axis],masks2[:,axis]=difference(xyz,axis,h,2)
    # Fisher parameters are (D, radians, radians); derivatives above use degrees.
    J[:,:,1:]*=180/np.pi; J2[:,:,1:]*=180/np.pi
    G=np.einsum("nri,nrj->nij",J,J); G2=np.einsum("nri,nrj->nij",J2,J2)
    ev=np.linalg.eigvalsh(G); ev2=np.linalg.eigvalsh(G2)
    rank=(ev>np.maximum(ev[:,-1],0)[:,None]*RANK_REL_TOL).sum(axis=1).astype(np.uint8)
    rank2=(ev2>np.maximum(ev2[:,-1],0)[:,None]*RANK_REL_TOL).sum(axis=1).astype(np.uint8)
    cond=np.full(n,np.inf); cond2=np.full(n,np.inf)
    cp=ev[:,0]>0; cp2=ev2[:,0]>0
    cond[cp]=ev[cp,-1]/ev[cp,0]; cond2[cp2]=ev2[cp2,-1]/ev2[cp2,0]
    valid=(rank==3)&cp; valid2=(rank2==3)&cp2
    inv=np.linalg.inv(G[valid]); inv2=np.linalg.inv(G2[valid2])
    d=np.diagonal(inv,axis1=1,axis2=2); d2=np.diagonal(inv2,axis1=1,axis2=2)
    if not (np.isfinite(d).all() and np.isfinite(d2).all() and (d>=0).all() and (d2>=0).all()): raise ValueError("Invalid covariance diagonal")
    unit=np.full((n,3),np.inf); unit2=np.full((n,3),np.inf)
    unit[valid]=np.sqrt(d); unit2[valid2]=np.sqrt(d2)
    unit[:,1:]*=180/np.pi; unit2[:,1:]*=180/np.pi
    sig=SIGMAS[:,None,None]*unit[None,:,:]; sig2=SIGMAS[:,None,None]*unit2[None,:,:]
    if not (np.allclose(sig[1],1.1*sig[0],equal_nan=True) and np.allclose(sig2[1],1.1*sig2[0],equal_nan=True)): raise ValueError("Noise scaling failed")
    gaze=np.sqrt((sig[:,:,1]**2+sig[:,:,2]**2)/2); gaze2=np.sqrt((sig2[:,:,1]**2+sig2[:,:,2]**2)/2)
    coords=np.array(np.meshgrid(A,ade,bde,indexing="ij")).reshape(3,-1).T
    edge=np.isclose(coords[:,0],0)|np.isclose(coords[:,0],4)|np.isclose(coords[:,1],-20)|np.isclose(coords[:,1],20)|np.isclose(coords[:,2],-20)|np.isclose(coords[:,2],20)
    interior=(np.round(coords[:,0],8)>=.2)&(np.round(coords[:,0],8)<=3.8)&(np.round(abs(coords[:,1]),8)<=12)&(np.round(abs(coords[:,2]),8)<=12)
    sym2=np.all(masks2==0,axis=1)
    if (edge.sum(),interior.sum(),sym2.sum())!=(28002,70699,159953): raise ValueError("Region or stencil mask count mismatch")
    aa=np.round(abs(coords[:,1]),8); bb=np.round(abs(coords[:,2]),8)
    edges=np.array([0,4,8,12,16,20.0000001]); abin=np.clip(np.digitize(aa,edges)-1,0,4); bbin=np.clip(np.digitize(bb,edges)-1,0,4)
    residual=max(np.max(abs(G[valid]@inv-np.eye(3))) if valid.any() else np.nan,np.max(abs(G2[valid2]@inv2-np.eye(3))) if valid2.any() else np.nan)
    np.savez_compressed(OUT/"p4_cr_floor.npz",accommodation_d=coords[:,0],ADE_deg=coords[:,1],BDE_deg=coords[:,2],sigma_A_D=sig[:,:,0],sigma_ADE_deg=sig[:,:,1],sigma_BDE_deg=sig[:,:,2],sigma_gaze_deg=gaze,sigma_A_2h_D=sig2[:,:,0],sigma_ADE_2h_deg=sig2[:,:,1],sigma_BDE_2h_deg=sig2[:,:,2],sigma_gaze_2h_deg=gaze2,condition_radian_parameters=np.broadcast_to(cond,(2,n)),condition_2h_radian_parameters=np.broadcast_to(cond2,(2,n)),rank=rank,rank_2h=rank2,edge_mask=edge,interior_mask=interior,symmetric_2h_mask=sym2,ADE_abs_bin=abin,BDE_abs_bin=bbin,stencil_mask_h=masks,stencil_mask_2h=masks2,noise_sigma_cases_mm=SIGMAS,noise_case_order=np.array(["0.0010 mm","0.0011 mm"]),rank_relative_tolerance=RANK_REL_TOL,finite_difference_steps=STEPS,region_counts=np.array([n,edge.sum(),(~edge).sum(),interior.sum(),sym2.sum()]),inverse_max_abs_residual=residual,raw_nonfinite_absolute_coordinate_entries=nonfinite,units=np.array("A D; ADE/BDE Fisher parameters radians, reported degrees; position mm; sigmas D/degrees; first dimension noise case; condition replicated across noise cases"))
    write_report(OUT/"report.md",coords,sig,sig2,gaze,gaze2,cond,cond2,rank,rank2,edge,interior,sym2,masks,masks2,abin,bbin,nonfinite,residual)
    print(f"states={n} edge={edge.sum()} nonedge={(~edge).sum()} interior={interior.sum()} symmetric2h={sym2.sum()} raw_nonfinite={nonfinite}")
    print(f"rankdef h={(rank<3).sum()} 2h={(rank2<3).sum()} cond>1e8 h={(cond>1e8).sum()} 2h={(cond2>1e8).sum()} inverse_residual={residual:.4g}")


def write_report(path,coords,sig,sig2,gaze,gaze2,cond,cond2,rank,rank2,edge,interior,sym2,masks,masks2,abin,bbin,nonfinite,resid):
    lines=["# Per-frame Cramer-Rao floor","","## Method","",f"Local one-frame lower bound for unbiased joint estimation of accommodation and both gaze angles, using the true grid labels without label error and assuming grid positions are the true model with no model error. Nonlinear finite-sample attainability is not guaranteed. Nine field points provide 18 total x/y coordinates. Noise is independent Gaussian per coordinate; two isotropic cases are evaluated: sigma=0.0010 mm and sigma=0.0011 mm. Absolute position is center_absolute + real (mm). For each case G=J^T J, I=G/sigma^2, Cov=sigma^2 G^-1; use the diagonal after joint inversion. Nonfinite raw absolute coordinate entries: {nonfinite}. Dataset has 214221 states; boundary states=28002; requested interior states=70699.","","Angular derivatives are computed per degree then converted to per radian for Fisher information in (D,rad,rad). Covariance angular standard deviations are converted to degrees once. sigma_gaze=sqrt((sigma_ADE^2+sigma_BDE^2)/2). Numerical-rank-deficient states receive infinite sigmas and remain in summaries.","","Stencil masks: 0 central, 1 forward, 2 backward. Steps: 0.2 D and 0.4 deg; comparison offsets: 0.4 D and 0.8 deg. One-sided differences use two points.","","## Overall and central interior","","| Region | A 1.0um med/P95/max mD | A 1.1um med/P95/max mD | ADE 1.0um med/P95/max deg | ADE 1.1um med/P95/max deg | BDE 1.0um med/P95/max deg | BDE 1.1um med/P95/max deg |","|---|---:|---:|---:|---:|---:|---:|"]
    for label,m in (("All",np.ones(len(edge),bool)),("Interior A=.2..3.8 and abs angles<=12 deg",interior),("Any boundary",edge),("Non-boundary",~edge)):
        cell=[label]
        for k,scale in ((0,1000),(1,1),(2,1)):
            for c in range(2): cell.append(" / ".join(f"{v*scale:.6g}" for v in stats(sig[c,m,k])))
        lines.append("| "+" | ".join(cell)+" |")
    lines += ["","## By accommodation","","| A D | 1.0um median sigma_A mD | median ADE deg | median BDE deg | max A mD | 1.1um median sigma_A mD | median ADE deg | median BDE deg | max A mD |","|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for a in np.unique(coords[:,0]):
        m=coords[:,0]==a; cell=[]
        for c in range(2):
            av,ad,bd=(stats(sig[c,m,k]) for k in range(3)); cell += [f"{av[0]*1000:.6g}",f"{ad[0]:.6g}",f"{bd[0]:.6g}",f"{av[2]*1000:.6g}"]
        lines.append(f"| {a:g} | "+" | ".join(cell)+" |")
    lines += ["","## By absolute angle bins","","Bins [0,4), [4,8), [8,12), [12,16), [16,20]; exact boundaries go to the bin starting there, except 20 in the last bin.",""]
    for name,bins in (("abs ADE",abin),("abs BDE",bbin)):
        lines += [f"### {name}","","| Bin deg | N | sigma_A med/max 1.0um mD | sigma_A med/max 1.1um mD | sigma_gaze med/max 1.0um deg | sigma_gaze med/max 1.1um deg |","|---|---:|---:|---:|---:|---:|"]
        for j,label in enumerate(("[0,4)","[4,8)","[8,12)","[12,16)","[16,20]")):
            m=bins==j; cell=[label,str(m.sum())]
            av0,av1=stats(sig[0,m,0]),stats(sig[1,m,0]); gv0,gv1=stats(gaze[0,m]),stats(gaze[1,m])
            cell += [f"{av0[0]*1000:.6g} / {av0[2]*1000:.6g}",f"{av1[0]*1000:.6g} / {av1[2]*1000:.6g}",f"{gv0[0]:.6g} / {gv0[2]:.6g}",f"{gv1[0]:.6g} / {gv1[2]:.6g}"]
            lines.append("| "+" | ".join(cell)+" |")
    lines += ["","## Worst ten by sigma_A","","| A D | ADE deg | BDE deg | sigma_A 1.0um mD | sigma_A 1.1um mD |","|---:|---:|---:|---:|---:|"]
    for i in np.argsort(sig[0,:,0])[-10:][::-1]: lines.append(f"| {coords[i,0]:g} | {coords[i,1]:g} | {coords[i,2]:g} | {sig[0,i,0]*1000:.6g} | {sig[1,i,0]*1000:.6g} |")
    rel=np.full(len(edge),np.nan); good=np.isfinite(sig2[0,:,0])&np.isfinite(sig[0,:,0]); rel[good]=abs(sig2[0,good,0]-sig[0,good,0])/sig[0,good,0]
    lines += ["","## Step-size sensitivity","","Relative change abs(sigma_A(2h)-sigma_A(h))/sigma_A(h), independent of isotropic noise.","","| Region | Maximum | State A,ADE,BDE | Valid count |","|---|---:|---|---:|"]
    for label,m in (("All",np.ones(len(edge),bool)),("Requested interior",interior),("Boundary",edge),("Symmetric 2h interior",sym2)):
        valid=m&np.isfinite(rel)
        if valid.any():
            ix=np.flatnonzero(valid)[np.argmax(rel[valid])]; value=f"{rel[ix]:.8g}"; state=coords[ix].tolist()
        else: value="undefined"; state="undefined"
        lines.append(f"| {label} | {value} | {state} | {valid.sum()}/{m.sum()} |")
    lines += ["","## Rank and conditioning","",f"Rank from eigvalsh of unweighted G=J^T J in (D,rad,rad); threshold lambda_max*{RANK_REL_TOL:.6g} (18 machine eps). Condition is lambda_max/lambda_min when lambda_min>0, infinity otherwise. Rank and condition are noise-independent.","","| Step | Rank deficient | Condition >1e8 | Median / P95 / max condition |","|---|---:|---:|---:|"]
    for label,c,r in (("h",cond,rank),("2h",cond2,rank2)):
        q=stats(c); lines.append(f"| {label} | {(r<3).sum()} | {(c>1e8).sum()} | {q[0]:.6g} / {q[1]:.6g} / {q[2]:.6g} |")
    for j,name in enumerate(("A","ADE","BDE")): lines.append(f"{name} stencil counts h central/forward/backward={np.bincount(masks[:,j],minlength=3).tolist()}, 2h={np.bincount(masks2[:,j],minlength=3).tolist()}")
    lines += [f"Max inverse residual max|G G^-1-I| across valid states: {resid:.6g}.",""]
    path.write_text("\n".join(lines),encoding="utf-8")


if __name__=="__main__": main()
