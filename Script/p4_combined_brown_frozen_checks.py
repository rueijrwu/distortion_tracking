"""Evaluate stored P4 Brown coefficients on held-out gaze states; no fitting."""
import csv, json, importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/p4_combined_brown/final_checks'
CORE = ROOT / 'data/p4_combined_brown/baseline_check/combined_core.py'
MODEL = ROOT / 'data/p4_combined_brown/baseline_check/combined_model_order2_1.npz'
BENCH = ROOT / 'data/p4_combined_brown/baseline_check/benchmark.npz'

def metrics(err):
    e = err[:, [0,1,2,3,5,6,7,8]]
    n = np.linalg.norm(e, axis=-1)
    return {'states': int(len(e)), 'vertex_rms_um': float(np.sqrt(np.mean(n*n))),
            'max_point_um': float(n.max())}

def main():
    spec = importlib.util.spec_from_file_location('combined_core', CORE)
    C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
    G = C.load_data()
    assert G['real'].shape == (21,101,101,9,2)
    mask = ~((np.arange(101)[:,None] % 2 == 0) & (np.arange(101)[None,:] % 2 == 0))
    assert mask.sum() == 7600
    with np.load(MODEL, allow_pickle=False) as z:
        c1, c0 = np.asarray(z['c1_coef'],float), np.asarray(z['c0_coef'],float)
    with np.load(BENCH, allow_pickle=False) as z: bp = np.asarray(z['params'],float)
    assert c1.shape == (42,9) and c0.shape == (14,9) and bp.shape == (21,101,101,9)
    tx, ty = np.meshgrid(G['BDE']/20, G['ADE']/20, indexing='xy')
    mono = [(i,d-i) for d in range(1,5) for i in range(d,-1,-1)]
    tv = np.stack([tx**i * ty**j for i,j in mono],-1).reshape(-1,14)
    frozen_basis, frozen_mono = C.gaze_basis(np.zeros(10201), np.repeat(G['ADE'],101),
                                               np.tile(G['BDE'],101), 2)
    assert frozen_mono == mono
    frozen_params = C.IDENTITY + frozen_basis @ c1
    manual_params = C.IDENTITY + np.concatenate([tv*v for v in
        np.polynomial.legendre.legvander(np.array([-1.0]),2).reshape(-1)],axis=1) @ c1
    assert np.allclose(frozen_params,manual_params,atol=1e-13,rtol=0)
    mask = mask.ravel(); L = G['L']; rows = {k:[] for k in ('BA only','C1 order 2','benchmark frozen','stored C0')}
    for ai,A in enumerate(G['A']):
        BA = C.barrel(G['r0'], A); truth = G['real'][ai].reshape(-1,9,2)
        e = (np.broadcast_to(BA,(10201,9,2))-truth)*1000
        rows['BA only'].append({'A_D':float(A), **metrics(e[mask])})
        # Freeze the shared C1 surface at A=0 for every evaluation A.
        leg = np.polynomial.legendre.legvander(np.array([-1.0]),2).reshape(-1)
        basis = np.concatenate([tv * v for v in leg], axis=1)
        par = C.IDENTITY + basis @ c1
        if ai == 0:
            assert np.allclose(par, frozen_params, atol=1e-13, rtol=0)
        pred,_,_ = C.forward_jac(BA[None]/L, par, 'BC + radial r2')
        rows['C1 order 2'].append({'A_D':float(A), **metrics((pred*L-truth)[mask]*1000)})
        par = bp[0].reshape(-1,9)
        pred,_,_ = C.forward_jac(BA[None]/L, par, 'BC + radial r2')
        rows['benchmark frozen'].append({'A_D':float(A), **metrics((pred*L-truth)[mask]*1000)})
        par = C.IDENTITY + tv @ c0
        pred,_,_ = C.forward_jac(BA[None]/L, par, 'BC + radial r2')
        rows['stored C0'].append({'A_D':float(A), **metrics((pred*L-truth)[mask]*1000)})
    OUT.mkdir(parents=True,exist_ok=True)
    files = {'BA only':'step1_a_baseline.csv','C1 order 2':'step1_b_c1_frozen_A0.csv',
             'benchmark frozen':'step1_c_benchmark_frozen_A0.csv','stored C0':'step1_c0_fault_assessment.csv'}
    (OUT/'step1_frozen_metrics.json').write_text(json.dumps({'grid_sha256':G['sha256'],'states_per_A':7600,'fields':'8 non-center','mask':'complement of even/even ADE/BDE indices','table_files':files,'tables':rows},indent=2)+'\n',encoding='utf8')
    for name,rr in rows.items():
        with (OUT/files[name]).open('w',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=list(rr[0])); w.writeheader(); w.writerows(rr)
    print(json.dumps({k:{'A0':v[0],'A4':v[-1]} for k,v in rows.items()},indent=2))

if __name__ == '__main__': main()
