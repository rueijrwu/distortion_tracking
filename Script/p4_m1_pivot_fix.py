"""Noise-free input-(b) M1 convention diagnostic and conditional pivot job.

Reads frozen models and saved evaluations. Writes only its own report/NPZ files.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

import p4_closed_loop_readout as base
import p4_eval_ablation as ab
from p4_phys_step1 import rotation

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/p4_m1_pivot_fix'
REPORT = OUT / 'report.md'
STEP = ROOT / 'data/p4_phys/step1.npz'
SAVED = ROOT / 'data/p4_closed_loop_readout/closed_loop_readout.npz'
EVAL = ROOT / 'data/p4_eval_ablation/evaluation.npz'
SLOPE = -.081595476571


def read(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def triple(x):
    return ab.fmt(ab.stats(x))


class JobReport:
    def __init__(self):
        self.sections = {}

    def put(self, key, text):
        self.sections[key] = text
        content = [
            '# P4 M1 convention and conditional pivot evaluation', '',
            'Noise-free; input (b), hybrid-corrected centres only. Parts (a)–(e) only. '
            'Existing sources, frozen models, saved reference and ablation outputs are read-only. '
            'No CODE V, baseline rerun, sensor noise, or commit. Each completed part is flushed here immediately.', '',
            'All gaze triples are radial RMS / P95 / maximum in arcmin; accommodation triples are '
            'RMS / absolute-error P95 / maximum in D. Signed gaze components are estimated minus true '
            '[ADE, BDE]. The primary population remains all 83,229 attempted states, including failures; '
            'the same 21 diagnostic states are excluded from primary scores. Gaze bins use '
            'max(|ADE|,|BDE|), exactly as the frozen hybrid reports.', '',
            'The reference is 57 fitted values: 18 B3/B4 + 12 PF12 + 15 A terms + 2 M1 + '
            '10 readout values. The readout has exactly 6 ADE and 4 BDE coefficients in total. '
            'Frozen B/PF12/A models, readout, search bounds, observed-only seed and solver mechanics '
            'remain fixed for the convention substitution in (a).', '',
            '**All results are conditional on the unverified shared sensor frame between P1 and P4:** '
            'common origin, axis signs, scale and physical registration have not been independently verified.', '',
        ]
        for k in sorted(self.sections):
            content.extend([self.sections[k], ''])
        REPORT.write_text('\n'.join(content), encoding='utf-8')


def diagnose(step, motion):
    states = step['M1_states']
    teacher = step['M1_frozen_teacher_delta_mm']
    physical = step['M1_candidate_delta']
    lever = float(step['M1_lever_mm'][0])
    signs = step['G4_sensor_sign']
    rebuilt = physical_delta(states[:, 0], states[:, 1:], lever, signs)
    if np.max(np.abs(rebuilt-physical)) > 1e-12:
        raise RuntimeError('Saved Step-1 physical increment and rebuilt formula disagree')
    # The archived Step-1 M1 teacher was made from inverse-G4 calibration centres;
    # the saved paired arrays are already in the same detector frame.
    with np.load(base.MOTION_CAL, allow_pickle=False) as z:
        trace_states = z['states'].copy()
        trace_c0 = z['matched_p4_center_A0_mm'].copy()
    if not np.array_equal(trace_states, states):
        raise RuntimeError('Step-1 and frozen M1 calibration states differ')
    trace_teacher = motion.forward(trace_c0, states[:, 0])-trace_c0
    output = []
    arrays = dict(calibration_states=states, teacher_delta_mm=teacher,
                  physical_delta_mm=physical, true_center_teacher_delta_mm=trace_teacher,
                  physical_rebuild_max_abs_mm=np.array(np.max(np.abs(rebuilt-physical))))
    for label, t in [('Step-1 paired teacher', teacher), ('M1 calibration trace-centre teacher', trace_teacher)]:
        for a in (2., 4.):
            mask = np.isclose(states[:, 0], a) & (np.linalg.norm(t, axis=1)>1e-10)
            p, tv = physical[mask], t[mask]
            length = np.linalg.norm(p, axis=1)/np.linalg.norm(tv, axis=1)
            signed = np.sum(p*tv, axis=1)/np.sum(tv*tv, axis=1)
            angle = np.degrees(np.arccos(np.clip(np.sum(p*tv, axis=1)/
                (np.linalg.norm(p, axis=1)*np.linalg.norm(tv, axis=1)), -1., 1.)))
            output.append(f'| {label} | {a:g} | {mask.sum()} | '+
                ' | '.join(' / '.join(f'{v:.9g}' for v in (x.min(),np.median(x),x.max()))
                           for x in (length,signed,angle))+' |')
    return output, arrays


def physical_delta(a, gaze, lever, signs):
    rv = np.einsum('nij,j->ni', rotation(gaze)-np.eye(3), np.array([0.,0.,lever]))
    # sensor signs are archived as a diagonal matrix.
    shifted = (rv[:, :2] @ signs.T) if signs.shape == (2,2) else rv[:, :2]*signs
    return (SLOPE*np.asarray(a)/4.)[:,None]*shifted


class PhysicalMotion:
    def __init__(self, lever, signs):
        self.lever = lever
        self.signs = signs


def corrected_displacement(p4_mean, p1_mean, a, g, frozen, field, motion):
    if not isinstance(motion, PhysicalMotion):
        return base.corrected_displacement(p4_mean,p1_mean,a,g,frozen,field,motion)
    forward = field.evaluate(a,g,diagnostics=True)
    c1 = p1_mean-frozen.p1_bias.predict(g)
    c4 = p4_mean-forward['prediction'].mean(axis=1)
    c4ref = c4-physical_delta(a,g,motion.lever,motion.signs)
    return c4ref-c1,c1,c4,c4ref,forward['valid']


def signed_bias(saved, reference):
    truth = saved['truth_state_scoring_only']
    primary = saved['primary_mask']
    radius = np.max(np.abs(truth[:,1:]),axis=1)
    error = (reference['estimated_state'][:,1:]-truth[:,1:])*60
    grid = np.arange(21)/5
    rows, fits, data = [], [], []
    confirmed = True
    for lo,hi in ((0,4),(4,8)):
        means = []
        for a in grid:
            mask = primary & (radius>=lo) & (radius<hi) & np.isclose(truth[:,0],a,atol=1e-10,rtol=0)
            mu = error[mask].mean(axis=0)
            means.append(mu)
            data.append([a,lo,hi,mask.sum(),*mu,np.linalg.norm(mu)])
            rows.append(f'| {a:.1f} | [{lo},{hi}) | {mask.sum()} | {mu[0]:.10g} | {mu[1]:.10g} | {np.linalg.norm(mu):.10g} |')
        means = np.array(means)
        coef = np.polyfit(grid,means[:,0],1)
        fit = np.polyval(coef,grid)
        r2 = 1-np.sum((means[:,0]-fit)**2)/np.sum((means[:,0]-means[:,0].mean())**2)
        zero = -coef[1]/coef[0]
        crossing = bool(means[:,0].min() <= 0 <= means[:,0].max())
        confirmed &= crossing and (0<zero<4) and r2>=.9
        fits.append(f'| [{lo},{hi}) | {coef[1]:.10g} | {coef[0]:.10g} | {r2:.9g} | {zero:.9g} | {crossing} |')
    return rows,fits,np.array(data),bool(confirmed)


def metrics_row(name,count,result,saved):
    p=saved['primary_mask']; truth=saved['truth_state_scoring_only']
    ge,ae=base.errors(result,truth)
    return f'| {name} | {count} | {triple(ge[p])} | {triple(ae[p])} | {int(result["converged"][p].sum())} | {int(result["invalid"][p].sum())} |'


def bins(result,saved):
    truth=saved['truth_state_scoring_only']; p=saved['primary_mask']
    r=np.max(np.abs(truth[:,1:]),axis=1); ge,ae=base.errors(result,truth)
    rows=['| True A D | Gaze bin ° | N | Gaze arcmin | Accommodation D |','|---:|---|---:|---|---|']
    for a in np.arange(21)/5:
        for lo,hi in ((0,4),(4,8)):
            mask=p & np.isclose(truth[:,0],a,atol=1e-10,rtol=0) & (r>=lo) & (r<hi)
            rows.append(f'| {a:.1f} | [{lo},{hi}) | {mask.sum()} | {triple(ge[mask])} | {triple(ae[mask])} |')
    return rows


def consolidated(results,saved):
    counts={'reference_saved':'57','identity_M1':'55','identity_B4':'48','identity_B3':'48',
        'identity_field_shape':'45','identity_A_terms':'42','physical_M1':'57 declared (invalid scalar-centre law)',
        'physical_B4':'48','oracle_true_A':'57 (oracle truth supplied)','A_linear_gain':'59',
        'physical_M1_corrected':'57 (55 + fixed slope + archived lever)'}
    text=['## (e) Consolidated results','',
        'Identity rows count the nonzero fitted values left after removal. Substitution rows count '
        'the numerical law retained: the old scalar physical-M1 row uses one fixed slope '
        '(56 stored values, 55 active fitted values; 57 if its duplicate x/y slots are retained); '
        'the corrected physical law uses one archived fitted lever plus the fixed slope '
        '(55 + 2 = 57 stored scalar law values; 56 fitted values if the fixed slope is excluded). Natural B4 '
        'uses the existing raw field and removes nine separate B4 values. The oracle uses true A '
        'at runtime and is not a deployable 57-value estimator. Two added gain values give 59.', '',
        'Quoted reference: 57 values, gaze 0.92 / 1.77 / 4.27 arcmin; A 0.028 / 0.055 / 0.185 D. '
        'Quoted oracle maximum: 3.11 arcmin. The table uses the unrounded saved numerical arrays.', '',
        '| Variant | Value count | Gaze RMS / P95 / max arcmin | A RMS / P95 / max D | Converged | Invalid |',
        '|---|---|---|---|---:|---:|']
    for name,result in results.items():
        text.append(metrics_row(name,counts.get(name,'57'),result,saved))
    text.extend(['| physical_B3 | unavailable | not run in ablation (continuous physical B3 law unavailable) | — | — | — |',
        '| pivot_P1_centre | not deployed (57 planned) | not run: (b) gate failed | — | — | — |',
        '| pivot_fitted | not deployed (59 planned) | not run: (b) gate failed | — | — | — |',
        '| physical_PF12 | not deployed | not run: deviation exceeds budget | — | — | — |','',
        '### Endpoint accommodation with and without clipped estimates','',
        'A clipped estimate is numerical contact with either search bound 0 or 4 D at tolerance '
        '1e-7 D. These are bound contacts, not evidence of an unconstrained optimum beyond the bound. '
        'Both endpoint populations are shown with all estimates and with bound contacts omitted; '
        'the primary headline metrics always retain them. Oracle bound contacts are imposed truth.', '',
        '| Variant | True A D | N all | N clipped | All A RMS / P95 / max D | N unclipped | Unclipped A RMS / P95 / max D |',
        '|---|---:|---:|---:|---|---:|---|'])
    t=saved['truth_state_scoring_only']; p=saved['primary_mask']
    for name,result in results.items():
        est=result['estimated_state'][:,0]; ae=est-t[:,0]
        clipped=np.isclose(est,0,atol=1e-7,rtol=0)|np.isclose(est,4,atol=1e-7,rtol=0)
        for a in (0.,4.):
            m=p & np.isclose(t[:,0],a,atol=1e-10,rtol=0); u=m&~clipped
            text.append(f'| {name} | {a:g} | {m.sum()} | {(m&clipped).sum()} | {triple(ae[m])} | {u.sum()} | {triple(ae[u])} |')
    return '\n'.join(text)


def worst(result,saved):
    g,a=base.errors(result,saved['truth_state_scoring_only']); p=saved['primary_mask']
    rows=['### Ten worst primary states: corrected physical M1','',
        '| Source ID | Source row | True A / ADE / BDE | Estimated A / ADE / BDE | Gaze arcmin | A error D | Converged | Invalid |',
        '|---:|---:|---|---|---:|---:|---|---|']
    for k in np.flatnonzero(p)[np.argsort(g[p])[-10:][::-1]]:
        ts=' / '.join(f'{x:.9g}' for x in saved['truth_state_scoring_only'][k])
        es=' / '.join(f'{x:.9g}' for x in result['estimated_state'][k])
        rows.append(f'| {saved["source_indices"][k]} | {saved["source_subset_rows"][k]} | {ts} | {es} | {g[k]:.9g} | {a[k]:.9g} | {bool(result["converged"][k])} | {bool(result["invalid"][k])} |')
    return '\n'.join(rows)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--saved-only',action='store_true')
    args=parser.parse_args(); start=time.perf_counter()
    OUT.mkdir(parents=True,exist_ok=True)
    report=JobReport()
    report.put('00','## Design and progress\n\n(a) Compare same-frame calibration increments, '
        'then substitute the source-backed physical M1 correction. (b) Compute signed gaze bias '
        'from saved reference arrays, without a loop. (c) A P1-centred 57-value pivot and then a '
        '59-value fitted pivot are conditional on (b) confirming a roughly linear signed offset '
        'that crosses zero between calibration levels; no held-out fit is permitted. (d) Check '
        'the PF12 handoff conversion and budget before any substitution. (e) Consolidate saved '
        'and newly run variants, including endpoint clipping. This section is updated as work finishes.')
    frozen,field,motion,hashes=base.load_models(base.FIELD,base.MOTION)
    saved=read(SAVED); step=read(STEP)
    source_hashes_before={str(path.relative_to(ROOT)):base.digest(path) for path in
        [SAVED,STEP,EVAL,base.SOURCE,ROOT/'Script/p4_eval_ablation.py',ROOT/'Script/p4_closed_loop_readout.py',
         ROOT/'data/p4_unified_combined/calibration/center_motion/central_motion_M1.npz']}
    if json.loads(str(saved['model_hashes_json'])) != hashes:
        raise RuntimeError('Frozen artifact hash mismatch')
    if saved['primary_mask'].sum()!=83229 or saved['excluded_mask'].sum()!=21:
        raise RuntimeError('Saved population changed')
    results={'reference_saved':{k[len('degree3_'):]:v for k,v in saved.items() if k.startswith('degree3_')}}
    for path in sorted((ROOT/'data/p4_eval_ablation').glob('variant_*.npz')):
        results[path.stem[len('variant_'):]]=read(path)
        source_hashes_before[str(path.relative_to(ROOT))]=base.digest(path)
    evaluation=read(EVAL)
    if not np.array_equal(evaluation['reference_saved_estimated_state'],results['reference_saved']['estimated_state']):
        raise RuntimeError('Ablation reference and original saved estimates disagree')
    model={'terms':[tuple(x) for x in saved['degree3_powers']],
           'coefficients':evaluation['parity_coefficients']}
    del evaluation
    cal=saved['calibration_states_teacher']; cald=saved['calibration_d_teacher_mm']
    unrestricted={'terms':model['terms'],'coefficients':saved['degree3_coefficients']}
    parity_error=np.max(np.linalg.norm(base.predict(model,saved['degree3_d_mm'])-
        base.predict(unrestricted,saved['degree3_d_mm']),axis=1))*60
    if parity_error>1e-6:
        raise RuntimeError('Saved 6+4 parity map is not equivalent to saved reference readout')
    diag,arrays=diagnose(step,motion)
    report.put('10','## (a) M1 convention diagnostic\n\n'
        'The teacher and Step-1 vectors below are forward increments c(A)−c(0), in the same '
        'nominal detector frame, at calibration poses only. Zero-length increments at A=0 and '
        'zero gaze are omitted from ratios/angles (48 nonzero poses). A length ratio is nonnegative; '
        'the signed projection ratio (physical·teacher)/(teacher·teacher) detects a reversal. '
        'Columns show minimum / median / maximum across poses; angles are in degrees.\n\n'
        '| Teacher construction | A D | N | Length ratio | Signed projection ratio | Angle ° |\n'
        '|---|---:|---:|---|---|---|\n'+'\n'.join(diag)+'\n\n'
        'The archived Step-1 law is δ_phys=(s(A)−1) S[(R(g)−I)(0,0,ell)]xy, '
        's=1−0.081595476571(A/4), R=Rx(−ADE)Ry(−BDE). Its saved lever and sensor signs are '
        'retained. The law is an additive central-motion increment driven by an eye-fixed lever; '
        'it is not scalar magnification of the absolute P4 centre. The old ablation applied s(A) '
        'directly to c4 about zero, changing the physical meaning of the slope. The corrected '
        'runtime inverse is c4_ref=c4−δ_phys(estimated A,estimated gaze). No sign flip, '
        'normalizer factor, fitted scale, or evaluation-label fit is introduced.\n\n'
        'Acceptance is tested only after running the complete loop: each gaze summary must be '
        'at most 1.05 times 0.92 / 1.77 / 4.27 arcmin. The loop result is written in the '
        'following subsection when complete.\n\n'
        f'The evaluation reference state array is exactly identical to the original saved '
        f'closed-loop state array. Maximum 6+4 parity-map prediction difference on all saved '
        f'runtime d is {parity_error:.9g} arcmin (tolerance 1e-6). This establishes the '
        '57-value reference without rerunning it.')
    biasrows,fitrows,bias,confirmed=signed_bias(saved,results['reference_saved'])
    arrays['signed_bias_by_A_bin']=bias
    arrays['offset_hypothesis_confirmed']=np.array(confirmed)
    report.put('20','## (b) Signed-bias table (saved arrays; no loop)\n\n'
        'For each true A and frozen max-axis gaze bin, average the signed gaze error vector '
        'before taking its Euclidean magnitude. This is not the mean radial error. '
        'A=1.5 D is absent; the actual 1.4 and 1.6 D rows are reported.\n\n'
        '| True A D | Gaze bin ° | N | Mean ADE error arcmin | Mean BDE error arcmin | Magnitude arcmin |\n'
        '|---:|---|---:|---:|---:|---:|\n'+'\n'.join(biasrows)+'\n\n'
        'Least-squares line of the dominant signed ADE mean versus A (diagnostic only; not a fit '
        'used by inference):\n\n| Gaze bin ° | Intercept arcmin | Slope arcmin/D | R² | Extrapolated zero D | Observed zero crossing |\n'
        '|---|---:|---:|---:|---:|---|\n'+'\n'.join(fitrows)+'\n\n'
        f'Offset hypothesis confirmed: **{confirmed}**. Direction is nearly fixed, but the mean '
        'signed error stays positive over 0–4 D in both bins; it does not cross zero between '
        'the calibration levels 0, 2 and 4 D. A U-shaped radial-error pattern alone does not '
        'establish the stated signed-offset mechanism.')
    if confirmed:
        raise RuntimeError('Unexpected offset gate passed; pivot fit requires implementation audit')
    report.put('30','## (c) Conditional pivot fix\n\n'
        '**Not run: the explicit (b) prerequisite failed.** Neither c1 nor c2 is fitted or evaluated. '
        'The planned c1 has 57 values, with M1 magnification about the current corrected P1 '
        'centre; c2 would fit a constant two-coordinate pivot on calibration poses only, '
        'giving **59 values**, and would run only if c1 failed. A changed pivot would require '
        'teacher-consistent recalibration of the existing 10 readout coefficients, without '
        'adding readout values. The frozen M1 scales, B/PF12 and A terms would remain fixed. '
        'There is no claim that a pivot removes the [0,4°) U-shape or reaches 3 arcmin.\n\n'
        'Saved reference [0,4°) and [4,8°) gaze and accommodation by A:\n\n'+'\n'.join(bins(results['reference_saved'],saved)))
    pf=ab.pf12_check(frozen,field)
    j=base.jacobian(model,saved['degree3_d_mm'][saved['primary_mask']])
    gain=float(np.linalg.svd(j,compute_uv=False)[:,0].max()*60/1000)
    zero=np.isclose(cal[:,0],0)&np.all(np.isclose(cal[:,1:],0),axis=1)
    gain0=float(np.linalg.svd(base.jacobian(model,cald[zero]),compute_uv=False)[:,0].max()*60/1000)
    pfs=['## (d) PF12 corrected Step-1 deviation','',
        'Recompute the frozen teacher with c_new=c_old·(20/12.4)^(i+j), leaving the physical '
        'candidate, its fitted focal length, reference rays, transform order, and 15 A terms '
        'unchanged. These are radial per-ray deviations of the saved Step-1 physical candidate '
        'from the converted teacher, not a refit. The saved Step-1 teacher already uses this '
        'conversion. The equivalent gaze error is a local sensitivity scale, not a loop result.', '',
        f'Readout gain: central A=0 largest Jacobian singular value {gain0:.9g} arcmin/µm; '
        f'maximum over saved primary runtime d {gain:.9g} arcmin/µm. The latter gives a '
        f'conservative 3-arcmin displacement budget {3/gain:.9g} µm (approximately 7.2 µm).','',
        '| Step-1 domain | Corrected deviation RMS / P95 / max µm | Equivalent arcmin at central gain | Equivalent arcmin at maximum saved gain |',
        '|---|---|---|---|']
    for label,key in [('Full Step-1 domain','after_full'),('Operating max-axis gaze ≤12.4°','after_operating')]:
        pfs.append(f'| {label} | {ab.fmt(pf[key])} | {ab.fmt(pf[key]*gain0)} | {ab.fmt(pf[key]*gain)} |')
    rawteacher=field.sparse_model().evaluate(step['P4_states'][:,0],step['P4_states'][:,1],step['P4_states'][:,2])['prediction']
    delta_mean=(step['P4_candidate_raw_H']-rawteacher).mean(axis=1)
    delta_d=delta_mean/motion.scales(step['P4_states'][:,0])
    operating=np.max(np.abs(step['P4_states'][:,1:]),axis=1)<=12.4+1e-10
    centre_stats=ab.stats(np.linalg.norm(delta_mean[operating],axis=1)*1000)
    d_stats=ab.stats(np.linalg.norm(delta_d[operating],axis=1)*1000)
    pfs.extend(['','The 7.2-µm budget applies to displacement d, whereas raw-H deviations are per ray. '
        'For the centre effect, average the candidate-minus-teacher raw-H vector over the nine '
        'rays, then apply the unchanged frozen M1 inverse scale. The sign of this centre '
        'correction does not affect its radial size. This gives:', '',
        '| Operating-domain centre metric | RMS / P95 / max µm | Equivalent arcmin at maximum saved gain |',
        '|---|---|---|',
        f'| Raw-H nine-ray mean deviation | {ab.fmt(centre_stats)} | {ab.fmt(centre_stats*gain)} |',
        f'| Resulting d-centre deviation after frozen M1 inverse | {ab.fmt(d_stats)} | {ab.fmt(d_stats*gain)} |'])
    pf['operating_mean_delta_um']=centre_stats
    pf['operating_d_delta_um']=d_stats
    pfs.extend(['',f'Saved Step-1 teacher versus recomputed converted teacher maximum: '
        f'{float(pf["saved_teacher_vs_after_max_um"]):.9g} µm. PF focal length remains '
        f'{float(pf["saved_physical_focal_mm"]):.9g} mm (inherited upper bound).', '',
        '**PF12 substitution not run:** the corrected discrepancy greatly exceeds the displacement '
        'budget, so the conditional runtime gate fails.'])
    report.put('40','\n'.join(pfs))
    arrays.update({f'PF12_{k}':v for k,v in pf.items()})
    arrays.update(readout_gain_central_arcmin_per_um=np.array(gain0),readout_gain_max_arcmin_per_um=np.array(gain),
        parity_coefficients=model['coefficients'],parity_powers=np.array(model['terms']),
        frozen_hashes_json=np.array(json.dumps(hashes,sort_keys=True)))
    np.savez_compressed(OUT/'diagnostics.npz',**arrays)
    report.put('50',consolidated(results,saved))
    if args.saved_only:
        print('Saved diagnostic parts (a diagnostic), (b), (c gate), (d), and provisional (e).',flush=True)
        return
    with np.load(base.SOURCE,allow_pickle=False) as z:
        rows=saved['source_subset_rows']
        if not np.array_equal(z['source_indices'][rows],saved['source_indices']):
            raise RuntimeError('Source identity join changed')
        if not np.array_equal(z['states'][rows],saved['truth_state_scoring_only']):
            raise RuntimeError('Source state join changed')
        observed_p4=z['absolute_p4_mm'][rows].copy()
        observed_p1=z['mean_p1_absolute_mm'][rows].copy()
    seed=dict(estimated_A_d=saved['initial_observed_only_A_d'],converged=saved['initial_profile_valid'],
        valid_map=saved['initial_profile_valid'],reference401_fallback=saved['initial_profile_fallback'])
    motion_phys=PhysicalMotion(float(step['M1_lever_mm'][0]),step['G4_sensor_sign'])
    checkpoint=OUT/'variant_physical_M1_corrected.npz'
    if checkpoint.exists():
        result=read(checkpoint)
        if str(result['script_sha256'])!=base.digest(__file__):
            raise RuntimeError('Own checkpoint source mismatch')
    else:
        ab.corrected_displacement=corrected_displacement
        result=ab.solve(observed_p4,observed_p1,model,frozen,field,motion_phys,seed,label='physical_M1_corrected')
        np.savez_compressed(checkpoint,script_sha256=np.array(base.digest(__file__)),**result)
    results['physical_M1_corrected']=result
    ge,ae=base.errors(result,saved['truth_state_scoring_only']); p=saved['primary_mask']
    accepted=bool(np.all(ab.stats(ge[p])<=1.05*np.array([.92,1.77,4.27])))
    report.put('11','### (a) Corrected physical M1 closed loop\n\n'
        '| Variant | Value count | Gaze RMS / P95 / max arcmin | A RMS / P95 / max D | Converged | Invalid |\n'
        '|---|---|---|---|---:|---:|\n'+metrics_row('physical_M1_corrected','57 (55 + fixed slope + archived lever)',result,saved)+'\n\n'
        f'Within the requested approximately 5% gaze envelope: **{accepted}**. '
        'Acceptance requires all three gaze summaries to pass; convergence alone does not imply accuracy. '
        'The 3-arcmin maximum target is evaluated separately.\n\n'+'\n'.join(bins(result,saved)))
    report.put('50',consolidated(results,saved))
    report.put('60',worst(result,saved))
    source_hashes_after={path:base.digest(ROOT/path) for path in source_hashes_before}
    if source_hashes_before != source_hashes_after:
        raise RuntimeError('Read-only input hashes changed during this job')
    _,_,_,hashes_after=base.load_models(base.FIELD,base.MOTION)
    if hashes_after != hashes:
        raise RuntimeError('Frozen model hashes changed during this job')
    arrays['read_only_input_hashes_json']=np.array(json.dumps(source_hashes_after,sort_keys=True))
    report.put('70','## Problems, decision, and provenance\n\n'
        'The earlier scalar-centre physical M1 substitution used a different law from the '
        'archived Step-1 increment and is retained only as an invalid comparison row. '
        'The signed-bias zero-crossing criterion failed, so no pivot variant or fit was run. '
        'PF12 remained over budget after conversion, so no PF12 loop was run. No physical '
        'identification, independent sensor registration, or noise robustness is established.\n\n'
        f'Corrected physical M1 maximum {ab.stats(ge[p])[2]:.9g} arcmin versus quoted reference '
        '4.27 and oracle 3.11 arcmin. The 3-arcmin target is '
        f'{"met" if ab.stats(ge[p])[2]<=3 else "not met"}.\n\n'
        'Source-backed conventions: Script/p4_phys_step1.py (rotation and candidate increment); '
        'Script/p4_closed_loop_readout.py (hybrid centre correction and observed-only solver); '
        'Script/p4_eval_ablation.py (frozen 6+4 readout and saved ablations); '
        'Script/p4_unified_fullshape.py (PF12 basis conversion).\n\n'
        f'Source hashes: script `{base.digest(__file__)}`; Step-1 `{base.digest(STEP)}`; '
        f'reference `{base.digest(SAVED)}`; ablation `{base.digest(EVAL)}`. '
        f'Frozen model hashes: `{json.dumps(hashes,sort_keys=True)}`.\n\n'
        f'Elapsed wall time {time.perf_counter()-start:.3f} seconds. Reproduce with '
        "`& 'C:\\Users\\rueijrwu\\.pyenv-win-venv\\envs\\venv\\Scripts\\python.exe' -B -u Script/p4_m1_pivot_fix.py`. "
        'Before/after hashes confirm all referenced frozen models and the saved input/source '
        'files checked by this script are unchanged. Outputs are only this new script and '
        'data/p4_m1_pivot_fix/report.md / NPZ artifacts. '
        'All results remain conditional on the unverified shared P1/P4 sensor frame.')
    output=dict(source_indices=saved['source_indices'],source_subset_rows=saved['source_subset_rows'],
        truth_state_scoring_only=saved['truth_state_scoring_only'],primary_mask=p,excluded_mask=saved['excluded_mask'],
        physical_M1_corrected_gaze_error_arcmin_scoring_only=ge,physical_M1_corrected_A_error_D_scoring_only=ae,
        accepted_within_5percent=np.array(accepted),**arrays)
    output.update({f'physical_M1_corrected_{k}':v for k,v in result.items()})
    np.savez_compressed(OUT/'evaluation.npz',**output)
    print(f'Completed: gaze {triple(ge[p])}; A {triple(ae[p])}; accepted={accepted}',flush=True)


if __name__=='__main__':
    main()
