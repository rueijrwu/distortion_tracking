#!/usr/bin/env python3
"""Fit the user-selected E1 camera model to the verified P1 retrace."""
from __future__ import annotations
import argparse, base64, hashlib, importlib.util, json, math, sys
from datetime import datetime, timezone
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/"data/p1_raytrace/p1_raytrace_251x251_12p5.pkl"
LENS=ROOT/"Lens/p1_ME_no_DM.seq"
MODEL=ROOT/"Script/p1_camera_fit.py"
N=251; ROWS=567009; RAYS=np.arange(9,dtype=np.int64)
FIELDS=[[-1.,1.],[0.,1.],[1.,1.],[-1.,0.],[0.,0.],[1.,0.],[-1.,-1.],[0.,-1.],[1.,-1.]]

def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()

def load_model():
 spec=importlib.util.spec_from_file_location("p1_camera_fit_reused",MODEL)
 if spec is None or spec.loader is None: raise ImportError(MODEL)
 mod=importlib.util.module_from_spec(spec); flag=sys.dont_write_bytecode
 try:
  sys.dont_write_bytecode=True; spec.loader.exec_module(mod)
 finally: sys.dont_write_bytecode=flag
 return mod

def validate(d,digest,expected_sha256=None):
 if not isinstance(d,np.ndarray) or d.dtype.names is None: raise ValueError("pickle is not a structured array")
 schema=(d.dtype.itemsize==224 and len(d.dtype.names)==28 and all(d.dtype.fields[n][0]==np.dtype("float64") for n in d.dtype.names))
 rows=d.shape==(ROWS,)
 if not schema or not rows: raise ValueError(f"schema/row gate failed: schema={schema}, shape={d.shape}")
 finite={n:bool(np.all(np.isfinite(d[n]))) for n in d.dtype.names}
 if not all(finite.values()): raise ValueError(f"nonfinite fields: {[k for k,v in finite.items() if not v]}")
 meta=(d.dtype.metadata or {}).get("p1_retrace")
 if not isinstance(meta,dict): raise ValueError("missing p1_retrace metadata")
 axes=[round(-12.5+i*.1,10) for i in range(N)]
 gates={"schema_valid":schema,"rowcount_valid":rows,
        "all_input_fields_finite":all(finite.values()),"metadata_axes":meta.get("axes_deg")==axes,
        "metadata_lattice_step":len(meta.get("axes_deg",[]))==N and all(abs(meta["axes_deg"][i+1]-meta["axes_deg"][i]-.1)<1e-12 for i in range(N-1)),
        "metadata_state_order":meta.get("state_order")=="BDE outer, ADE inner; state_index=iBDE*nADE+iADE",
        "metadata_ray_order":meta.get("ray_order")=="Y=1,0,-1; X=-1,0,1 within each Y",
        "metadata_counts":meta.get("validation",{}).get("states")==N*N and meta.get("validation",{}).get("rays")==ROWS,
        "metadata_status_counts":meta.get("validation",{}).get("status_counts")=={"0":ROWS}}
 ai=np.rint((d["ade_deg"]+12.5)/.1).astype(np.int64); bi=np.rint((d["bde_deg"]+12.5)/.1).astype(np.int64)
 expected_state=np.repeat(np.arange(N*N,dtype=np.int64),9); expected_rays=np.tile(RAYS,N*N)
 expected_ade=np.repeat(np.tile(np.asarray(axes),N),9); expected_bde=np.repeat(np.repeat(np.asarray(axes),N),9)
 checks={"raw_state_order":bool(np.array_equal(d["state_index"],expected_state)),"raw_ray_order":bool(np.array_equal(d["ray_index"],expected_rays)),
         "raw_lattice_axes_order":bool(np.array_equal(d["ade_deg"],expected_ade) and np.array_equal(d["bde_deg"],expected_bde)),
         "unique_ade_bde_axes":bool(len(np.unique(d["ade_deg"]))==N and len(np.unique(d["bde_deg"]))==N),
         "lattice_bounds_step":bool(np.allclose(d["ade_deg"],ai*.1-12.5,rtol=0,atol=1e-8) and np.allclose(d["bde_deg"],bi*.1-12.5,rtol=0,atol=1e-8)),
         "metadata_state_index_actual":bool(np.array_equal(d["state_index"],(bi*N+ai).astype(float))),
         "status_all_zero":bool(np.all(d["rayrsi_status"]==0)),
         "readbacks_match":bool(np.allclose(d["ade_deg"],d["rc_ade_readback_deg"],rtol=0,atol=1e-9) and np.allclose(d["bde_deg"],d["rc_bde_readback_deg"],rtol=0,atol=1e-9)),
         "stop_origin":bool(max(np.max(np.abs(d[k])) for k in ("x8","y8","z8"))<=1e-9),"sensor_z_constant":bool(np.ptp(d["zsi"])==0),
         "tangent_matches_angles":bool(np.allclose(np.tan(np.deg2rad(d["theta_x_deg"])),d["l8"]/d["n8"],rtol=1e-12,atol=1e-15) and np.allclose(np.tan(np.deg2rad(d["theta_y_deg"])),d["m8"]/d["n8"],rtol=1e-12,atol=1e-15)),
         "n8_negative_magnitude":bool(np.all(d["n8"]<0) and np.all(np.abs(d["n8"])>=.9999))}
 if expected_sha256 is not None: gates["pickle_sha256_case_insensitive"]=digest.lower()==expected_sha256.lower()
 if not all(gates.values()) or not all(checks.values()): raise ValueError(f"validation gates failed: {gates}; {checks}")
 grouped=d.reshape(N*N,9)
 fields={str(i):[float(grouped[0,i]["field_x_normalized"]),float(grouped[0,i]["field_y_normalized"])] for i in range(9)}
 fields_ok=all(np.all(grouped[:,i]["field_x_normalized"]==fields[str(i)][0]) and np.all(grouped[:,i]["field_y_normalized"]==fields[str(i)][1]) for i in range(9))
 checks["field_mapping_constant"]=fields_ok
 checks["field_mapping_matches_documented"]=fields=={str(i):FIELDS[i] for i in range(9)} and meta.get("field_xy")==FIELDS
 if not fields_ok or not checks["field_mapping_matches_documented"]: raise ValueError("field mapping gate failed")
 info={"checks":{**gates,**checks},"input_field_finiteness":finite,"rows":int(d.size),"states":N*N,"rays_per_state":9,
       "lattice":{"ADE_BDE_min_deg":float(min(meta["axes_deg"])),"ADE_BDE_max_deg":float(max(meta["axes_deg"])),"step_deg":float(meta["axes_deg"][1]-meta["axes_deg"][0]),"dimensions":[len(np.unique(d["ade_deg"])),len(np.unique(d["bde_deg"]))],"metadata_axes_deg":meta["axes_deg"],"state_order":meta["state_order"],"ray_order":meta["ray_order"]},
       "field_by_ray":fields,"status_counts":{"0":int(np.count_nonzero(d["rayrsi_status"]==0))},
       "n8":{"negative_count":int(np.count_nonzero(d["n8"]<0)),"nonnegative_count":int(np.count_nonzero(d["n8"]>=0)),"min":float(d["n8"].min()),"max":float(d["n8"].max()),"abs_min":float(np.abs(d["n8"]).min())},
       "sensor_z_mm":float(d["zsi"][0]),"metadata_lens_sha256":meta.get("lens_sha256"),"metadata_profile":meta.get("profile")}
 return meta,info,ai,bi

def plot(path,grids,smoke):
 fig,axs=plt.subplots(1,2,figsize=(10,5),constrained_layout=True); suffix=" (smoke sample)" if smoke else ""
 for ax,key,title in zip(axs,("rms","bias"),("Per-state nine-ray RMS residual (µm)","Mean nine-ray minus center-ray bias error (µm)")):
  im=ax.imshow(grids[key],origin="lower",extent=(-12.55,12.55,-12.55,12.55),aspect="auto")
  ax.set_xlabel("BDE (deg)");ax.set_ylabel("ADE (deg)");ax.set_title(title+suffix);fig.colorbar(im,ax=ax,label="µm")
 fig.savefig(path,dpi=150);plt.close(fig)

def class_errors(px,py,x,y,rays,mask):
 point=np.hypot(px-x,py-y)*1000
 groups={"ray_"+str(k):np.flatnonzero(mask&(rays==k)) for k in range(9)}
 groups["center"]=np.flatnonzero(mask&(rays==4));groups["edge"]=np.flatnonzero(mask&np.isin(rays,[1,3,5,7]));groups["corner"]=np.flatnonzero(mask&np.isin(rays,[0,2,6,8]))
 return {k:{"rms_um":float(np.sqrt(np.mean(point[ix]**2))),"max_um":float(np.max(point[ix]))} for k,ix in groups.items() if len(ix)}

def run(smoke, output_dir=None, expected_sha256=None):
 logpath=ROOT/"runs/subagent_status.log"
 logpath.parent.mkdir(parents=True,exist_ok=True)
 def log(s):
  with logpath.open("a",encoding="utf-8") as f:f.write(f"{datetime.now(timezone.utc).isoformat()} p1_camera_refit {s}\n")
 source_path=Path(__file__).resolve();source_bytes=source_path.read_bytes();source_hash=hashlib.sha256(source_bytes).hexdigest()
 out=Path(output_dir).resolve() if output_dir is not None else ROOT/"runs"/("p1_camera_refit_smoke" if smoke else "p1_camera_refit")
 if smoke and not out.name.endswith("_smoke"):out=out.with_name(out.name+"_smoke")
 out.mkdir(parents=True,exist_ok=True)
 digest=sha(INPUT)
 print(f"Input SHA-256: {digest}; check {'enabled' if expected_sha256 is not None else 'skipped (no --expected-sha256)'}",flush=True)
 if expected_sha256 is not None and digest.lower()!=expected_sha256.lower():raise ValueError(f"input SHA256 mismatch: {digest}")
 log(f"input SHA256 {digest}; check {'verified' if expected_sha256 is not None else 'skipped'}")
 data=np.load(INPUT,allow_pickle=True);meta,gates,ai,bi=validate(data,digest,expected_sha256);mod=load_model();log("full input validation gates passed")
 if smoke:
  sample=np.unique(np.floor(np.linspace(0,250,21)).astype(np.int64));keep=np.isin(ai,sample)&np.isin(bi,sample);data=data[keep];ai=ai[keep];bi=bi[keep]
 else:sample=np.arange(N)
 rays=data["ray_index"].astype(int);state=bi*N+ai;train=(ai%2==0)&(bi%2==0);counts=(int(train.sum()//9),int((~train).sum()//9));expected=(121,320) if smoke else (15876,47125)
 if counts!=expected:raise ValueError(f"split counts mismatch {counts}")
 tx,ty=data["l8"]/data["n8"],data["m8"]/data["n8"];x,y=data["xsi"],data["ysi"]
 T=float(max(np.max(np.abs(tx[train])),np.max(np.abs(ty[train]))))
 if not math.isfinite(T) or T<=0:raise ValueError(f"invalid training tangent scale {T}")
 u,v=tx/T,ty/T
 f0=float(np.median(np.hypot(x[train],y[train])/np.maximum(np.hypot(u[train],v[train]),1e-15)))
 p,solver=mod._fit_level(1,np.array([f0,0.,0.,0.]),u[train],v[train],x[train],y[train]);px,py=mod._project(p,u,v,1)
 finite_model=bool(np.all(np.isfinite(p)) and np.all(np.isfinite(px)) and np.all(np.isfinite(py)) and np.all(np.isfinite(px-x)) and np.all(np.isfinite(py-y)))
 if not finite_model:raise ValueError("fitted arrays are not finite")
 log(f"E1 fit complete smoke={smoke} converged={solver.status>0}")
 metrics={};state_metrics={}
 for name,mask in (("train",train),("holdout",~train)):
  m=mod._metrics(px[mask],py[mask],x[mask],y[mask],rays[mask],state[mask],np.hypot(u[mask],v[mask]));state_metrics[name]=m
  metrics[name]={k:v for k,v in m.items() if not k.startswith("_")};metrics[name]["per_ray_class_errors"]=class_errors(px,py,x,y,rays,mask)
 grids={}
 for key,field in (("rms","_per_state_rms_mm"),("bias","_per_state_bias_mm")):
  ids=np.concatenate([state_metrics[s]["_state_ids"] for s in ("train","holdout")]);vals=np.concatenate([state_metrics[s][field] for s in ("train","holdout")])*1000
  grid=np.full((N,N),np.nan);grid[ids%N,ids//N]=vals;grids[key]=grid
 plotname="E1_residual_maps.png";plot(out/plotname,grids,smoke)
 tangents={"description":"One global fit uses traced stop-direction tangents tx=l8/n8 and ty=m8/n8. ADE, BDE, and normalized field coordinates are not independent regressors; their influence is represented only through the traced tangents.","tx":{"residual_x":[],"residual_y":[]},"ty":{"residual_x":[],"residual_y":[]}}
 for tn,tv in (("tx",tx),("ty",ty)):
  edges=np.linspace(float(tv.min()),float(tv.max()),21)
  for rn,res in (("residual_x",px-x),("residual_y",py-y)):
   for j,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
    sel=(tv>=lo)&(tv<(hi if j<19 else hi+np.finfo(float).eps))
    if np.any(sel):tangents[tn][rn].append({"tangent_range":[float(lo),float(hi)],"count":int(sel.sum()),"residual_rms_um":float(np.sqrt(np.mean(res[sel]**2))*1000)})
 np.savez_compressed(out/"residuals.npz",predicted_minus_traced_mm=np.column_stack((px-x,py-y)),ray_index=rays,ade_index=ai,bde_index=bi,state_index=data["state_index"],is_train=train,tangent_x_physical=tx,tangent_y_physical=ty,radius_normalized=np.hypot(u,v),target_x_mm=x,target_y_mm=y,predicted_x_mm=px,predicted_y_mm=py)
 ser=mod._serial_params(p,T,1)
 for key in ("normalized","physical"):ser[key]={k:v for k,v in ser[key].items() if k in ("f","cx","cy","k1")}
 ser["normalization"]["conversion"]="f=f_norm/T; k1=k1_norm/T^2; cx and cy unchanged"
 summary={"model":"E1","model_definition":"pinhole f,cx,cy and radial coefficient k1","input_pickle":str(INPUT),"input_pickle_sha256":digest,"dataset_gates":gates,
  "expected_input_sha256":expected_sha256,"input_sha256_check":"verified" if expected_sha256 is not None else "skipped",
  "current_lens_informational":{"path":str(LENS),"sha256":sha(LENS)},"imported_model_implementation":{"path":str(MODEL),"sha256":sha(MODEL)},
  "fit_scope":"deterministic 21x21 smoke subset" if smoke else "full 251x251 lattice","smoke_subset_global_indices":sample.tolist() if smoke else None,
  "holdout_definition":"train iff global i_ade and i_bde are both even; holdout otherwise","training_split":{"train_states":counts[0],"holdout_states":counts[1],"train_rays":int(train.sum()),"holdout_rays":int((~train).sum())},
  "normalization":{"T":T,"definition":"max(abs(tx),abs(ty)) over active training rays","inputs":"u=tx/T,v=ty/T; tx=l8/n8, ty=m8/n8"},"parameters":ser,
  "fit_flags":{"converged":bool(solver.status>0),"status":int(solver.status),"finite_model_arrays":finite_model,"nfev":int(solver.nfev),"message":str(solver.message)},
  "metrics":metrics,"gaze_field_dependence_traced_tangents_only":tangents,"historical_comparison":{"old_E1_holdout_max_um":0.0015829611284427967,"context":"earlier data and lens; reference only"},
  "residual_sign":"predicted minus traced","outputs":{"summary":"summary.json","residuals":"residuals.npz","plot":plotname},"script_sha256":source_hash,
  "script_source_snapshot":{"encoding":"UTF-8 exact bytes, base64","sha256":source_hash,"base64":base64.b64encode(source_bytes).decode("ascii")}}
 if source_path.read_bytes()!=source_bytes:raise RuntimeError("script changed during fit; refusing to publish provenance mismatch")
 (out/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8");log(f"summary published {out}")
 return out

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument("--expected-sha256",help="Expected input SHA-256; default: skip hash check and print the hash found")
 ap.add_argument("--smoke",action="store_true")
 ap.add_argument("--output-dir",type=Path,help="Output folder; default: runs/p1_camera_refit (smoke: runs/p1_camera_refit_smoke)")
 args=ap.parse_args();out=run(args.smoke,args.output_dir,args.expected_sha256);print(out);return 0
if __name__=="__main__":raise SystemExit(main())
