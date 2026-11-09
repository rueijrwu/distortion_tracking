"""Finish Step 0/1 from saved constants. Evaluation only: no fitting or optimization.

Run with the project-mandated interpreter and -B. Writes only data/p4_phys/.
The initial fitting driver is NOT rerun by this program.
"""
from pathlib import Path
import hashlib
import json
import pickle
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/p4_phys'
sys.path.insert(0, str(ROOT / 'Script'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metric(delta):
    d = np.asarray(delta).reshape(-1, 2) * 1000
    if not np.isfinite(d).all():
        raise ValueError('Nonfinite comparison; do not silently drop rows')
    v = np.linalg.norm(d, axis=1)
    return dict(n=len(v), max_um=float(v.max()), rms_um=float(np.sqrt(np.mean(v*v))),
                coordinate_max_um=float(np.abs(d).max()))


def rot(g):
    a, b = np.deg2rad(g).T
    ca, sa, cb, sb = np.cos(a), np.sin(a), np.cos(b), np.sin(b)
    r = np.zeros((len(g), 3, 3))
    r[:, 0, 0] = cb; r[:, 0, 2] = -sb
    r[:, 1, 0] = sa*sb; r[:, 1, 1] = ca; r[:, 1, 2] = sa*cb
    r[:, 2, 0] = ca*sb; r[:, 2, 1] = -sa; r[:, 2, 2] = ca*cb
    return r


def physical_inverse(q, origin, sign, lever):
    # Central branch of the existing rotated-lever law; no parameter estimation.
    x, y = ((q-origin) @ sign).T
    dx, dy, dz = lever
    argx = (x+dx)/np.hypot(dx, dz)
    if np.any(np.abs(argx) > 1):
        raise ValueError('Physical inverse outside central branch')
    b = np.arctan2(dx, dz)-np.arcsin(argx)
    zp = dx*np.sin(b)+dz*np.cos(b)
    argy = (y+dy)/np.hypot(dy, zp)
    if np.any(np.abs(argy) > 1):
        raise ValueError('Physical inverse outside central branch')
    a = np.arcsin(argy)-np.arctan2(dy, zp)
    return np.rad2deg(np.column_stack((a, b)))


def key(g):
    return tuple(np.rint(np.asarray(g)*1000000).astype(int))


def main():
    from evaluate_p4_unified_combined import load_models
    from p4_phys_step1 import FIELD_SUMMARY, MOTION_SUMMARY
    with np.load(OUT/'step1.npz') as z:
        z = {k: z[k].copy() for k in z.files}
    previous = json.loads(str(z['report_json']))
    old = previous.get('initial_run', previous)
    # Allow the same evaluation-only command to be rerun on its corrected output.
    for legacy, corrected in [('P4_teacher_center', 'P4_trace_center_mm'),
                              ('P1_teacher_raw_H', 'P1_trace_raw_H_mm'),
                              ('P1_teacher_center', 'P1_trace_center_mm')]:
        if legacy not in z:
            z[legacy] = z[corrected]
    frozen, field, motion, model_hashes = load_models(FIELD_SUMMARY, MOTION_SUMMARY)
    fixed_keys = ['G4_effective_lever_xyz_mm', 'G1_effective_axial_lever_mm',
                  'M1_lever_mm', 'PF_focal_mm', 'G4_origin', 'G1_origin',
                  'G4_sensor_sign', 'G1_sensor_sign']
    constants_before = {k: z[k].copy() for k in fixed_keys}
    s4, s1 = z['P4_states'], z['P1_states']
    g0 = s4[np.isclose(s4[:, 0], 0), 1:]
    op0 = np.max(np.abs(g0), axis=1) <= 12.4+1e-9
    op = np.max(np.abs(s4[:, 1:]), axis=1) <= 12.4+1e-9
    assert np.allclose(g0, s1[:, 1:], atol=1e-10)

    # Blocking fix 2: independently check labels, provenance and canonical masks.
    rawpath = ROOT/'data/distortion_grid_p1_biaxial_macro/distortion_grid.pkl'
    with rawpath.open('rb') as f:
        raw = pickle.load(f)
    labels = np.unique(np.column_stack((raw['field_x_relative'], raw['field_y_relative'])), axis=0)
    assert np.array_equal(labels, z['P1_fields'])
    order = np.lexsort((raw['field_y_relative'], raw['field_x_relative'],
                        raw['eye_rotation_x_deg'], raw['eye_rotation_deg']))
    rows = raw[order].reshape(len(g0), 9)
    assert np.allclose(rows['eye_rotation_deg'], g0[:, 0, None])
    assert np.allclose(rows['eye_rotation_x_deg'], g0[:, 1, None])
    assert np.array_equal(rows['field_x_relative'], np.broadcast_to(labels[:, 0], rows.shape))
    assert np.array_equal(rows['field_y_relative'], np.broadcast_to(labels[:, 1], rows.shape))
    absolute = np.stack((rows['center_absolute_x_mm']+rows['real_x_mm'],
                         rows['center_absolute_y_mm']+rows['real_y_mm']), axis=-1)
    ci = int(np.flatnonzero(np.all(labels == 0, axis=1))[0])
    p1c = absolute[:, ci]
    p1mean = absolute.mean(axis=1)
    assert np.max(np.abs(p1c-z['G1_trace_center'])) < 1e-10
    assert np.max(np.abs(absolute-p1c[:, None]-z['P1_teacher_raw_H'])) < 1e-10
    ix = {key(g): i for i, g in enumerate(g0)}
    srci = np.array([ix[key(g)] for g in z['source_states'][:, 1:]])
    paired_mean_delta = float(np.max(np.abs(p1mean[srci]-z['source_mean_p1'])))
    assert paired_mean_delta < 1e-10
    calpath = ROOT/'data/p1_p4_a0_independent_gaze/calibration/summary.json'
    canonpath = ROOT/'data/p1_p4_a0_independent_gaze/runtime/summary.json'
    runtimepath = canonpath.parent/'runtime_independent_P1_P4.npz'
    sourcepath = ROOT/'data/p4_p1_limited_gaze_12p5/source_subset.npz'
    pairedpath = ROOT/'data/p4_p1_both_centers/calibration/paired_data.npz'
    cal = json.loads(calpath.read_text(encoding='utf-8-sig'))
    canon = json.loads(canonpath.read_text(encoding='utf-8-sig'))
    assert digest(rawpath) == cal['P1_raw_sha256']
    assert digest(sourcepath) == cal['raw_paired_source_sha256']
    assert digest(pairedpath) == cal['P1_paired_data_sha256'] == canon['P1_paired_data_sha256']
    assert digest(calpath) == canon['calibration_summary_sha256']
    assert digest(runtimepath) == canon['runtime_artifact_sha256']
    with np.load(pairedpath) as p:
        pi = np.array([ix[key(g)] for g in p['p1_gaze_deg']])
        assert np.max(np.abs(p1mean[pi]-p['p1_mean_mm'])) < 1e-10
        paired_central_delta = float(np.max(np.abs(p1c[pi]-p['p1_labeled_center_mm'])))
        assert paired_central_delta < 1e-6  # native chief-ray retrace tolerance
    with np.load(runtimepath) as rt:
        ri = np.array([ix[key(g)] for g in rt['truth_gaze_deg']])
        assert np.max(np.abs(p1mean[ri]-rt['p1_raw_center_mm'])) < 1e-10
        assert np.max(np.abs(p1c[ri]-rt['p1_true_labeled_center_mm'])) < 1e-6
        primary = rt['primary_user_exclusion_mask'].astype(bool)
        e = np.linalg.norm(rt['p1_gaze_deg']-rt['truth_gaze_deg'], axis=1)*60
        canonical_rms = float(np.sqrt(np.mean(e[primary]**2)))
        canonical_max = float(e[primary].max())
        assert abs(canonical_rms-60*canon['channels']['P1']['scores']['primary_user_exclusion_3943']['final_primary']['radial']['RMS_deg']) < 1e-10
    excluded = np.all(np.isclose(g0, canon['excluded_gaze_deg'], atol=1e-9), axis=1)
    assert excluded.sum() == 1
    ei = int(np.flatnonzero(excluded)[0])
    provenance = dict(row_order_error=False, label_error=False, status='unresolved physical cause; not teacher error',
        raw_sha256=digest(rawpath), source_subset_sha256=digest(sourcepath),
        canonical_summary_sha256=digest(canonpath), paired_data_sha256=digest(pairedpath),
        max_paired_mean_difference_mm=paired_mean_delta,
        max_paired_chief_retrace_difference_mm=paired_central_delta,
        canonical_primary_count=int(primary.sum()), canonical_P1_rms_arcmin=canonical_rms,
        canonical_P1_max_arcmin=canonical_max, excluded_gaze=canon['excluded_gaze_deg'],
        excluded_absolute_center_mm=p1c[ei].tolist(), excluded_absolute_centroid_mm=p1mean[ei].tolist(),
        excluded_repetitions_in_paired_subset=int(np.sum(srci == ei)))

    # Blocking fix 1: preserve the original 75-pose RAY-TRACE DIFFERENCES by name.
    motioncal = ROOT/'data/p4_unified_combined/calibration/center_motion/calibration_data.npz'
    with np.load(motioncal) as mc:
        raydiff = mc['true_p4_center_A_mm']-mc['matched_p4_center_A0_mm']
        assert np.allclose(z['M1_states'], mc['states'])
    z['M1_raytrace_center_difference_mm'] = raydiff
    z.pop('M1_trace_chained_delta', None)
    if 'M1_trace_ownpart_delta' in z:
        z['M1_teacher_to_raytrace_residual_calibration_mm'] = z.pop('M1_trace_ownpart_delta')
    qcal = np.array([z['G4_inverse_teacher_center'][ix[key(g)]] for g in z['M1_states'][:, 1:]])
    teacher_cal = motion.forward(qcal, z['M1_states'][:, 0])-qcal
    z['M1_frozen_teacher_delta_mm'] = teacher_cal
    z.pop('M1_teacher_delta', None)
    # Evaluate the already fitted lever densely, without fitting again.
    ai = np.array([ix[key(g)] for g in s4[:, 1:]])
    q = z['G4_inverse_teacher_center'][ai]
    teacher = motion.forward(q, s4[:, 0])-q
    lever = np.array([0., 0., float(z['M1_lever_mm'][0])])
    shift = np.einsum('nij,j->ni', rot(s4[:, 1:])-np.eye(3), lever)[:, :2] @ z['G4_sensor_sign']
    candidate = (-0.081595476571*s4[:, 0]/4)[:, None]*shift
    ctrace = z['P4_teacher_center']
    c0trace = ctrace[np.isclose(s4[:, 0], 0)][ai]
    own_trace_residual = motion.forward(c0trace, s4[:, 0])-ctrace
    z.update(M1_dense_states=s4, M1_dense_frozen_teacher_delta_mm=teacher,
             M1_dense_candidate_delta_mm=candidate,
             M1_dense_raytrace_center_difference_mm=ctrace-c0trace,
             M1_dense_teacher_to_raytrace_residual_mm=own_trace_residual)

    # Actual inverse of candidate law on teacher inputs, not a self-roundtrip.
    inverse_stats = {}
    for channel in ['G4', 'G1']:
        lever_key = 'G4_effective_lever_xyz_mm' if channel == 'G4' else 'G1_effective_axial_lever_mm'
        qi = z[channel+'_inverse_teacher_center']
        gi = physical_inverse(qi, z[channel+'_origin'], z[channel+'_sensor_sign'], z[lever_key])
        pred = z[channel+'_origin'] + np.einsum('nij,j->ni', rot(gi)-np.eye(3), z[lever_key])[:, :2] @ z[channel+'_sensor_sign']
        assert np.max(np.abs(pred-qi)) < 1e-10
        e = np.linalg.norm(gi-g0, axis=1)*60
        inverse_stats[channel] = dict(full_max_arcmin=float(e.max()), operating_max_arcmin=float(e[op0].max()),
            reconstruction_max_mm=float(np.max(np.abs(pred-qi))), failures=0)
        z[channel+'_physical_inverse_of_teacher_deg'] = gi

    # Direct final F-to-trace comparison. Do not anchor the trace to the teacher.
    final = field.evaluate(s4[:, 0], s4[:, 1:])
    rawH, traceH = z['P4_teacher_raw_H'], z['P4_trace_raw_H']
    b4 = z['P4_B4_teacher']
    h0mean = rawH[np.isclose(s4[:, 0], 0)].mean(axis=1)[ai]
    expected = b4+rawH.mean(axis=1)-h0mean
    assert np.max(np.abs(final.mean(axis=1)-expected)) < 1e-12
    z['P4_final_teacher_F_mm'] = final
    z['P4_B4_expected_mean'] = expected
    z['P4_trace_center_mm'] = z.pop('P4_teacher_center')
    z['P1_trace_raw_H_mm'] = z.pop('P1_teacher_raw_H')
    z['P1_trace_center_mm'] = z.pop('P1_teacher_center')
    z['paired_labelled_center_difference_P4_minus_P1_mm'] = z['source_p4'][:, 4]-p1c[srci]
    z['paired_centroid_difference_P4_minus_P1_mm'] = z['source_p4'].mean(axis=1)-z['source_mean_p1']
    z['P1_predeclared_excluded_gaze_mask'] = excluded
    z['operating_gaze_mask'] = op0
    z['operating_state_mask'] = op

    comparisons = {}
    def add(name, delta, mask, own=None):
        comparisons[name] = dict(full=metric(delta), operating=metric(delta[mask]))
        if own is not None:
            comparisons[name]['teacher_to_trace_full'] = metric(own)
            comparisons[name]['teacher_to_trace_operating'] = metric(own[mask])
    add('G4', z['G4_physical_center']-z['G4_inverse_teacher_center'], op0,
        z['G4_inverse_teacher_center']-z['G4_trace_center'])
    add('G1', z['G1_physical_center']-z['G1_inverse_teacher_center'], op0)
    add('M1', candidate-teacher, op, own_trace_residual)
    add('PF12/A-terms', z['P4_candidate_raw_H']-rawH, op, rawH-traceH)
    add('B4 natural shape derivation', rawH[np.isclose(s4[:, 0], 0)].mean(axis=1)-b4[np.isclose(s4[:, 0], 0)], op0,
        b4[np.isclose(s4[:, 0], 0)]-traceH[np.isclose(s4[:, 0], 0)].mean(axis=1))
    add('Final F (diagnostic)', final-traceH, op)
    # Keep excluded P1 anomaly as raw observations, not teacher-error scores.
    old['G1_glint'].pop('inverse_teacher_center_vs_raytrace_macro_mm', None)
    old.get('B_terms', {}).pop('P1_raytrace_mean_vs_frozen_B1_at_A0', None)
    old['conventions'].pop('source_quotes', None)  # replace malformed initial encoding below
    old['limitations'] = ['Lens/p4_ME.len exists but is binary; its prescription was not extracted.',
        'P1 anomalous excluded state has unresolved physical cause, not a teacher-error interpretation.',
        'G1 radius has zero sensitivity in the fixed-bisector approximation.',
        'Sensor-space levers are not identified anatomical distances.']
    old.pop('B_terms', None)  # corrected comparisons and identity supersede old anchored diagnostics
    fixed = dict(scope='Steps 0/1 only. Finalization evaluates saved constants; no new fitting.',
        comparisons=comparisons, physical_inverse=inverse_stats, P1_provenance_check=provenance,
        M1_array_definitions={
          'M1_raytrace_center_difference_mm':'75 calibration poses: true_p4_center_A_mm - matched_p4_center_A0_mm; ray-trace differences, NOT teacher output',
          'M1_frozen_teacher_delta_mm':'75 poses: frozen M1.forward(G4_inverse(g),A)-G4_inverse(g)',
          'M1_dense_raytrace_center_difference_mm':'All 214221 states: observed central ray at A minus observed central ray at A0, same gaze',
          'M1_dense_frozen_teacher_delta_mm':'All 214221 states: frozen M1.forward(G4_inverse(g),A)-G4_inverse(g)'},
        final_F_mean_identity_max_mm=float(np.max(np.abs(final.mean(axis=1)-expected))),
        original_fitted_constants={k:z[k].tolist() for k in fixed_keys},
        fixed_model_hashes=model_hashes, initial_run=old)
    for k, v in constants_before.items():
        assert np.array_equal(z[k], v), 'A fitted constant changed'

    conventions = '''# Step 0: frozen conventions

Coordinates are the recorded terminal-image x/y sensor-frame values in **mm**, with their stored origin, handedness and scale preserved. For each labelled field `(field_x_relative, field_y_relative)` in `{-1,0,1}²`, absolute position is **`center_absolute_x/y_mm + real_x/y_mm`** from `data/distortion_grid_p4_biaxial_macro/distortion_grid.pkl`. The point `(0,0)` is the **labelled central ray**; the **nine-ray centroid** is the arithmetic mean of all nine absolute positions. These points are distinct. Select and join rays by field labels, never flattened order; P1 collector order and P4/benchmark order differ. The paraxial reference is a separate near-axis ray, not the absolute origin.

Gaze is `[ADE,BDE]=[Ey,Ex]` in **degrees**, applied on surface `RC`; accommodation **A is in D**. Keep source `SRC_ROT ADE -20` unchanged. CODE V Help, `C:/CODEV202403_SR1/help/lens_system/tilts.11.11.html` (*Order of Operations: Decenters and Tilts*), states: “For compound tilts, β must be defined in the α-tilted coordinate system, and similarly for γ (defined in the α, β tilted system).” Its quoted order is “assumed order of operation ADE, BDE, CDE for a single surface”. `tilts.11.08.html` defines ADE and BDE as left-handed about local +X and +Y. Thus the candidate local-frame orientation is represented by **`R=Rx(-ADE) Ry(-BDE)`**, CDE=0, with standard active column-vector matrices. The matrix is an inference under that convention, rather than a matrix quoted by CODE V; +ADE tips a +Z lever toward +Y, +BDE toward -X. Later mirror BEN tilts are separate from RC gaze.

The **distortion pivot is fixed at sensor `[0,0] mm`** as a stated candidate-model assumption, not an identified optical principal point. The translation gauge fixes each channel's centre at zero gaze to its frozen teacher-inverse reference; no additional fitted translation or free pivot is allowed. Step 1 records the fixed sensor-axis sign maps and distinguishes these references from measured ray positions. **`L=1.235429592627 mm`** is fixed. There is no fitting in this convention declaration.

The paired subset supports calculating **nominal P4-minus-P1 positions at matched `(A,ADE,BDE)`**. The primary difference here uses the **labelled `(0,0)` centre in BOTH images**; a separately named centroid difference uses nine-ray centroids in BOTH images. Do not mix definitions. P1 positions are joined by gaze to the pinned P1 source and repeated over A, not new independent P1 measurements. The canonical runtime uses each image's own centroid and independent G/B maps, not an established difference readout. A standalone P4-minus-P1 gaze/accommodation readout and physical inter-camera registration are **not validated**: the P1 manifest calls the shared rear-camera frame provisional. No difference readout is fitted here.

`Lens/p4_ME.len` exists as a binary prescription; it was not exported or decoded. The readable `Lens/p4_ME.seq` is a companion, not verified equivalent: lines 51–53 place RC with THI -13.8 and zero decenter; lines 56/60 give CorneaF radius 7.77 mm and conic -0.15. Companion values are not claimed to be values extracted from the `.len` or an identified sensor relay scale.
'''
    (OUT/'conventions.md').write_text(conventions, encoding='utf-8')

    def tab(name):
        c = comparisons[name]
        lines = ['| Quantity | Full ±20° | Operating ±12.4° |', '|---|---:|---:|']
        for title, fn in [('Max displacement (µm)', lambda m:m['max_um']),
                          ('RMS displacement (µm)', lambda m:m['rms_um']),
                          ('A equivalent (D), 30–24 µm/D', None),
                          ('Gaze equivalent (arcmin), 280 µm/deg', lambda m:m['max_um']/280*60)]:
            if fn is None:
                values = [f"{c[k]['max_um']/30:.6g}–{c[k]['max_um']/24:.6g}" for k in ['full','operating']]
            else:
                values = [f'{fn(c[k]):.6g}' for k in ['full','operating']]
            lines.append(f'| {title} | {values[0]} | {values[1]} |')
        if 'teacher_to_trace_full' in c:
            lines.append('| Teacher’s own ray-trace max (µm) | '+f"{c['teacher_to_trace_full']['max_um']:.6g} | {c['teacher_to_trace_operating']['max_um']:.6g} |")
        return '\n'.join(lines)
    m = comparisons['M1']
    mpass = m['operating']['max_um'] <= 280*.25/60
    summary = []
    for name in ['G4','G1','M1','PF12/A-terms','B4 natural shape derivation']:
        c=comparisons[name]
        status = 'Failed / failed'
        if name == 'M1' and mpass:
            status = 'Failed full / accepted operating, numerical gaze budget only'
        if name == 'G1': status += '; radius and P1 trace unresolved'
        summary.append(f"| {name} | {c['full']['max_um']:.6g} | {c['operating']['max_um']:.6g} | {status} |")
    summary.append('| B3 (P1) shape derivation | — | — | Unresolved; no frozen P1 nine-ray forward shape |')
    def invtext(ch):
        i=inverse_stats[ch]
        return (f"Actual physical inverse on frozen-teacher centre inputs: max gaze deviation "
            f"{i['full_max_arcmin']:.6g} arcmin full / {i['operating_max_arcmin']:.6g} operating; "
            f"{i['failures']} inverse failures; forward reconstruction {i['reconstruction_max_mm']:.3g} mm. "
            'This is an inverse-to-teacher comparison, not merely self-inversion.')
    original = old
    lines = ['# Step 1: physical-law screening of the frozen 87-value teacher', '',
      'Only Steps 0 and 1 ran. The initial candidate constants are retained unchanged. The finalization command performs evaluation, provenance checks and reporting only: no new fitting, CODE V, R0–R3 ladder refit, noise, Steps 2–4, or joint physical-model calibration. The 87-value model is the teacher only. No total parameter count or target-level physical-model success is claimed.', '',
      'Full grid: 21 A levels 0–4 D × 101 ADE × 101 BDE = 214,221 P4 states, nine rays each. G1/G4 use 10,201 unique gazes; operating means both components within ±12.4°. Gaze readout and B centroid checks use the quarter-target 0.25 arcmin budget (about 1.1667 µm); accommodation field checks use 0.025 D (conservatively 0.6 µm). Both rough conversions are shown; they are not end-to-end accommodation/gaze accuracy proofs.', '',
      '| Part | Max law–teacher deviation, full (µm) | Operating (µm) | Result |', '|---|---:|---:|---|',
      *summary, '',
      'Accepted, failed and unresolved are separate: M1 can pass the operating **numerical gaze-equivalent** budget while its anatomical lever remains unidentified; all full-domain candidates fail. G1 radius and P1 anomalous-trace attribution are unresolved. B3 derivation stops because no deployed P1 nine-ray forward shape exists.', '',
      '## G4 — rotation-centre lever', '', tab('G4'), '',
      f"Existing fit: `c(g)=c0+S[(R−I)d]xy`, `d={z['G4_effective_lever_xyz_mm'].tolist()} mm`, `c0={z['G4_origin'].tolist()} mm`, `S={z['G4_sensor_sign'].tolist()}`. Three lever coordinates were fit to inverse-teacher centres, not ray tracing. All 10,201 frozen G4 inversions converged in the initial run.", '',
      invtext('G4'), '',
      'Result: **failed**, including the operating region. Largest forward residual is at combined extreme gaze; the systematic cross-axis/corner growth is inconsistent with a single orthographic rotated lever. One missing term: finite-conjugate relay projection. The effective 16.12 mm axial lever cannot be compared directly with an anatomical distance without relay magnification.', '',
      'Lens comparison: requested `.len` counterpart is unresolved (binary, no export). Companion `.seq` RC thickness -13.8 mm at line 51 is a coordinate-break thickness, not a verified sensor-space lever; no equality is asserted.', '',
      '## G1 — convex corneal mirror', '', tab('G1'), '',
      f"Existing spherical-mirror approximation: `c_glint=R*d_cornea+r*n_fixed`. Under fixed collimated illumination and orthographic observation, the world bisector `n_fixed` is constant. Zero-gaze subtraction cancels `r*n_fixed` exactly: radius sensitivity is zero. The fitted effective axial distance is `{float(z['G1_effective_axial_lever_mm'][2]):.9g} mm`; the 7.77 mm companion radius is a fixed prior, not an identified fitted radius.", '',
      invtext('G1'), '',
      'Result: **failed numerical law**, plus **unresolved radius identification and P1 trace attribution**. Combined-gaze corner residuals dominate; one missing term is finite-conjugate specular observation. The omitted conic is another model limitation, not a term added here. P1 teacher-to-trace maxima from the earlier draft are withdrawn: the anomalous state has no row/label error, but its physical cause remains unresolved. Do not interpret it as teacher error.', '',
      'Lens comparison: companion `.seq` line 56 gives CorneaF radius 7.77 mm and line 60 conic -0.15. Neither is verified as the binary `.len` prescription. The fitted projected 7.44 mm lever is not a corneal radius or an independently calibrated RC distance.', '',
      '## M1 — fixed on-axis scale times a lever', '', tab('M1'), '',
      f"Existing lever `{float(z['M1_lever_mm'][0]):.9g} mm`, fitted previously on 75 known calibration poses, is evaluated unchanged on all 214,221 states. Candidate increment is `(s(A)−1) S[(R−I)(0,0,ell)]xy`, with `s(A)=1−0.081595476571(A/4)` fixed from `data/p4_onaxis/report.md:244–270`. Frozen M1 has two diagonal scale coefficients `{motion.coefficients.tolist()}` and pivot `{motion.pivot_mm.tolist()}`. Its own trace error is `M1(c_trace(A0),A)−c_trace(A)`, which isolates M1 from G4 error.", '',
      ('Result: **accepted operating numerical gaze-equivalent approximation; failed full grid**.' if mpass else 'Result: **failed numerical approximation**.'),
      'The accommodation-equivalent flag is separate; passing a gaze budget does not establish 0.025 D. Residuals grow with A and combined gaze. One missing term for full-domain use: anisotropic relay magnification. No extra term was fitted. Lens comparison is unresolved: this is an effective projected lever; no directly readable `.len` counterpart or calibrated relay scale is available.', '',
      'Blocking array fix: `M1_raytrace_center_difference_mm` is **ray-trace A minus matched ray-trace A0 central positions on the original 75 poses**. It is not teacher output and not a model residual. `M1_frozen_teacher_delta_mm` is separately recomputed from frozen `M1.forward(G4_inverse(g),A)−G4_inverse(g)`. Dense versions have explicit names. The ambiguous `M1_teacher_delta` and `M1_trace_chained_delta` keys are removed. The old 75-pose and corrected dense comparisons are not conflated.', '',
      '## PF12 / 15 A-terms — tilt, pivot-offset Brown terms and radial scale', '', tab('PF12/A-terms'), '',
      f"One candidate only: undistort the frozen zero-gaze reference with on-axis k(0), scale by s(A), rotate the ray directions in the Step 0 order, pinhole-project relative to the central direction, then apply fixed-pivot radial distortion. Existing effective focal fit `{float(z['PF_focal_mm']):.9g} mm` is at the 1000 mm upper search bound. It used 3,348 deterministic every-64th states; evaluation uses the entire grid. No new fit was run during finalization.", '',
      'For a single ADE tilt, the pinhole Jacobian at the chief direction gives perpendicular `sec(theta)` and parallel `sec²(theta)`. For a cubic radial map `D(x)=x+k|x|²x`, expansion of `D(d+q)−D(d)` gives Brown terms `p1=k*d_y`, `p2=k*d_x` in mm⁻¹ (normalized coefficients `L*k*d_y`, `L*k*d_x`). They are derived, not independently fitted. Linear and radial terms from the same expansion are retained. `k(A)` is fixed from the on-axis report, coefficients in powers of A/4: `[-0.018500496700,0.00023989235714,0.0015756063948,-0.0014256378340,0.00066200789981] mm⁻²`.', '',
      'Result: **failed**. Strong combined-gaze edge/corner residuals grow rapidly with gaze; the largest is at A=0, ADE=BDE=20°. A0 zero-gaze identity and positive projective denominators pass. The large failure and bound-hitting fit reject this direct transfer of the on-axis radial law and sensor-centre displacement into a common distortion plane. They do not refute the sec/sec² tilt relation. One missing term: relay mapping from sensor-centre displacement to distortion-plane pivot offset. The on-axis radial law was validated at only two nonzero sampled radii; extrapolation to large absolute offsets is not established.', '',
      'The PF comparison uses raw teacher H before its empirical mean anchor. Final F versus **unmodified** ray tracing is additionally stored in the artifact. Lens comparison: focal length is an effective pinhole parameter, not an extracted camera-lens focal length; no verified `.len` counterpart is available.', '',
      '## B3 / B4 — derived means, no fitting', '', tab('B4 natural shape derivation'), '',
      'B4 result: **failed shape-derived replacement**. With `H_center=0`, the natural centroid bias is `mean(H)`. It differs from frozen B4 even at A=0. The implemented teacher instead adds `(9/8)*(B4−mean(H(0,g)))` to the eight outer rays. Algebraically, `mean(F)=B4+mean(H(A,g))−mean(H(0,g))`. This identity is checked, not fitted.', '',
      f"Full-grid identity discrepancy is `{fixed['final_F_mean_identity_max_mm']:.3g} mm`. The residual reveals a missing continuous off-axis centroid displacement in the raw shape law; the discrete outer-ray anchor is not a physical derivation. Lens comparison: B polynomials/centroid anchors have no direct prescription counterpart; binary `.len` extraction is also unavailable.", '',
      'B3 (P1) result: **unresolved / stopped for shape derivation**. The teacher contains P1 B3 and G20 but no continuous P1 nine-ray forward shape. Measured P1 means are observations, not a physical shape law, so they cannot be used to claim a derived B3 replacement. The anomalous P1 state is handled in the blocking-fix section below, not scored as a teacher error.', '',
      '## P1 blocking provenance check and unresolved observation', '',
      'Checked the dense raw pickle, explicit `(ADE,BDE,field_x,field_y)` labels, complete nine-field groups, pinned paired means, paired central rays, canonical runtime observation rows and hashes. No row-order or label error was found; no P1 labels were corrected and no affected comparison was refit or rerun as an ordering correction.', '',
      f"Dense reconstructed P1 means versus all 83,349 paired-source rows: maximum coordinate difference `{paired_mean_delta:.3g} mm`. Pinned raw P1 SHA-256 `{provenance['raw_sha256']}` matches calibration metadata; paired source `{provenance['source_subset_sha256']}` and paired-data hashes match. Canonical summary and runtime artifact hashes also match their pins.", '',
      f"The canonical summary explicitly excludes gaze `{canon['excluded_gaze_deg']}` (source id `{canon['excluded_source_id']}`) from its 3,943 primary holdout rows and keeps it in the 3,944-row diagnostic. Its 21 paired repetitions are unchanged. At this state, raw labelled centre is `{p1c[ei].tolist()} mm`, whereas raw nine-ray centroid is `{p1mean[ei].tolist()} mm`. The canonical primary P1 runtime score is reproduced from saved arrays: RMS `{canonical_rms:.6g}` / max `{canonical_max:.6g}` arcmin. That primary score and the full-grid anomaly use different masks and are not contradictory.", '',
      'The observation’s physical cause remains **unresolved**. A zero ray-status code is insufficient to diagnose it. No new quality mask is invented, no point is silently dropped, and no large full-grid P1 discrepancy is interpreted as teacher error. The predeclared exclusion is stored explicitly for traceability.', '',
      '## Sources, checks and limitations', '',
      '`data/p4_biaxial_joint/` is a historical joint transform; `data/p4_ade_a0_decenter_fine401/` is an A=0 specialist with three incomplete states; `data/p4_combined_brown/baseline_check/benchmark.npz` is a per-state nine-parameter ceiling on the macro grid. These were located/inspected but are not substituted for the deployed hybrid teacher. Canonical loader: `evaluate_p4_unified_combined.load_models`, frozen G/B loader `p4_unified_a_gaze_model.load_frozen_a0`, field `UnifiedP4Field`, motion `CentralMotion`.', '',
      'The required `.len` comparison remains unresolved. Current file exists and is binary; no live optical export was made. Companion `.seq` constants are quoted only as companion evidence, not verified model equivalence. P1 collector provenance is its pinned `p1_ME.seq` hash in its manifest, not the current P4 binary. The paired data use nominal shared sensor coordinates with provisional registration.', '',
      'All final arrays are finite; physical inverse reconstruction, raw-reference matching, field mean identity, source hash pins and unchanged fitted constants are checked. The initial syntax check briefly created a bytecode cache under Script/__pycache__; that generated file was removed. Finalization uses -B, creates no cache and saves only under data/p4_phys/. No pre-task existing file was modified and no commit was made.', '',
      'Reproduce finalization only: `& C:\\Users\\rueijrwu\\.pyenv-win-venv\\envs\\venv\\Scripts\\python.exe -B Script/p4_phys_finalize.py`. This command performs no fitting. `step1.npz` stores full arrays, corrected definitions, masks, constants, checks and `report_json`.', '']
    (OUT/'step1_report.md').write_text('\n'.join(lines), encoding='utf-8')
    z['report_json'] = np.array(json.dumps(fixed, ensure_ascii=False))
    # Write atomically within the authorized output directory.
    temporary = OUT/'step1_finalizing.npz'
    np.savez_compressed(temporary, **z)
    temporary.replace(OUT/'step1.npz')
    print(json.dumps(dict(comparisons=comparisons, physical_inverse=inverse_stats,
          P1_provenance=provenance, M1_operating_numerical_gaze_pass=mpass), indent=2))


if __name__ == '__main__':
    main()
