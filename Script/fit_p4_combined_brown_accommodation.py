"""Run the mandatory direct Step-1 gate; stop before fitting if it fails.

This run records the failed prerequisite. It is not a fitted combined model.
No CODE V dependencies. Only data/p4_combined_brown is written.
"""
import argparse
import hashlib
import os
import pickle
from pathlib import Path

for _name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_name] = '6'
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/p4_combined_brown'
GRID = ROOT / 'data/distortion_grid_p4_biaxial_macro/distortion_grid.pkl'
COARSE = ROOT / 'data/p4_ade_a0_decenter/p4_ade_a0_biaxial.npz'
FINE = ROOT / 'data/p4_ade_a0_decenter_fine401/p4_ade_a0_biaxial.npz'
FAMILIES = ('projective', 'projective + BC', 'BC + radial r2')
OUTER = np.array([0, 1, 2, 3, 5, 6, 7, 8])
FIELDS = np.array([(x, y) for x in (-1., 0., 1.) for y in (-1., 0., 1.)])
IDENTITY = np.array([1., 0., 0., 1., 0., 0., 0., 0., 0.])
EXPECTED = np.array([[5.94, 28.6], [3.48, 16.9], [2.13, 11.8]])
TOL = np.array([.01, .10])  # Wider than requested rounding, not a fit tolerance.


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def forward(b, p, family):
    """Section-1k H -> Brown p1,p2 -> optional radial r2, normalized xy."""
    x, y = b[..., 0], b[..., 1]
    den = 1 + p[..., 4] * x + p[..., 5] * y
    ux = (p[..., 0] * x + p[..., 1] * y) / den
    uy = (p[..., 2] * x + p[..., 3] * y) / den
    if family == 'projective':
        wx, wy = ux, uy
    else:
        p1, p2 = p[..., 6], p[..., 7]
        wx = ux + 2 * p1 * ux * uy + p2 * (3 * ux * ux + uy * uy)
        wy = uy + p1 * (ux * ux + 3 * uy * uy) + 2 * p2 * ux * uy
    if family == 'BC + radial r2':
        gain = 1 + p[..., 8] * (wx * wx + wy * wy)
        wx, wy = gain * wx, gain * wy
    return np.stack((wx, wy), -1), den


def metrics(e, mask, count):
    e = e[mask][:, OUTER if count == 8 else np.arange(9)]
    return np.array([np.sqrt(np.mean(e[..., 0] ** 2)),
                     np.sqrt(np.mean(e[..., 1] ** 2)),
                     np.sqrt(np.mean(np.sum(e ** 2, -1))),
                     np.sqrt(np.mean(e ** 2)), np.linalg.norm(e, axis=-1).max()])


def evaluate(real, ade, bde, valid, saved, L):
    """Compute predictions from coefficients; never use saved residuals."""
    assert real.shape == (len(bde), len(ade), 9, 2)
    real = real - real[:, :, 4:5]
    ia = np.flatnonzero(np.isclose(ade, 0, atol=1e-9, rtol=0)).item()
    ib = np.flatnonzero(np.isclose(bde, 0, atol=1e-9, rtol=0)).item()
    reference = real[ib, ia]
    b = reference[None] / L
    tx, ty = np.meshgrid(bde / 20, ade / 20, indexing='ij')
    powers = saved['polynomial_powers']
    basis = np.stack([tx.ravel() ** i * ty.ravel() ** j for i, j in powers], -1)
    assert np.array_equal(powers[0], [0, 0])
    even = ((np.arange(len(bde))[:, None] % 2 == 0) &
            (np.arange(len(ade))[None, :] % 2 == 0)).ravel()
    train, test = even & valid.ravel(), ~even & valid.ravel()
    result = dict(train=train, test=test, reference=reference, target=real.reshape(-1, 9, 2))
    for family in FAMILIES:
        c = saved['coefficients_' + family]
        pred, den = forward(b, (basis @ c)[:, None], family)
        e = (pred * L - real.reshape(-1, 9, 2)) * 1000
        assert np.isfinite(e[valid.ravel()]).all()
        assert den[valid.ravel()].min() > .05
        assert np.abs(e[valid.ravel(), 4]).max() == 0
        result[family] = dict(error=e, test8=metrics(e, test, 8), test9=metrics(e, test, 9),
                              constant=c[0], coefficient=c, min_den=float(den[valid.ravel()].min()))
    return result


def table(lines, name, result):
    for family in FAMILIES:
        for count in (8, 9):
            m = result[family][f'test{count}']
            lines.append(f'| {name} | {family} | {count} | ' + ' | '.join(f'{v:.6f}' for v in m) + ' |')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pipeline-only', action='store_true', help='run the prerequisite gate (always first)')
    ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with GRID.open('rb') as stream:
        d = pickle.load(stream)
    acc = np.unique(d['accommodation_d'])
    ade, bde = np.unique(d['eye_rotation_deg']), np.unique(d['eye_rotation_x_deg'])
    assert len(d) == 21 * 101 * 101 * 9
    assert np.allclose(acc, np.linspace(0, 4, 21), atol=1e-9, rtol=0)
    assert np.allclose(ade, np.linspace(-20, 20, 101), atol=1e-9, rtol=0)
    assert np.allclose(bde, np.linspace(-20, 20, 101), atol=1e-9, rtol=0)
    assert np.isfinite(d['real_x_mm']).all() and np.isfinite(d['real_y_mm']).all()
    s = d[np.isclose(d['accommodation_d'], 0, atol=1e-9, rtol=0)]
    s = s[np.lexsort((s['field_y_relative'], s['field_x_relative'],
                      s['eye_rotation_deg'], s['eye_rotation_x_deg']))]
    fields = np.stack((s['field_x_relative'], s['field_y_relative']), -1).reshape(-1, 9, 2)
    assert np.array_equal(fields, np.broadcast_to(FIELDS, fields.shape))
    assert np.allclose(s['eye_rotation_x_deg'].reshape(101, 101, 9), bde[:, None, None])
    assert np.allclose(s['eye_rotation_deg'].reshape(101, 101, 9), ade[None, :, None])
    real = np.stack((s['real_x_mm'], s['real_y_mm']), -1).reshape(101, 101, 9, 2)
    del d, s
    with np.load(COARSE, allow_pickle=False) as coarse, np.load(FINE, allow_pickle=False) as fine:
        L = float(coarse['L_mm'])
        assert np.isclose(L, float(fine['L_mm']), atol=1e-12, rtol=0)
        primary = evaluate(real, ade, bde, np.ones((101, 101), bool), coarse, L)
        # Diagnostic only, never substituted for the primary 101x101 gate.
        fc = evaluate(fine['real_mm'], fine['ADE'], fine['BDE'], fine['valid_state'], coarse, L)
        ff = evaluate(fine['real_mm'], fine['ADE'], fine['BDE'], fine['valid_state'], fine, L)
        actual = np.array([primary[f]['test8'][[2, 4]] for f in FAMILIES])
        passed = bool(np.all(np.abs(actual - EXPECTED) <= TOL))
        payload = dict(pipeline_passed=np.array(passed), families=np.array(FAMILIES),
                       expected_vertex_max_um=EXPECTED, actual_vertex_max_um=actual, tolerance_vertex_max_um=TOL,
                       accommodation_d=acc, ADE=ade, BDE=bde, fields=FIELDS, L_mm=np.array(L),
                       polynomial_powers=coarse['polynomial_powers'], reference_mm=primary['reference'],
                       A0_target_mm=primary['target'], train_mask=primary['train'], test_mask=primary['test'],
                       metric_columns=np.array(['x_RMSE', 'y_RMSE', 'vertex_RMS', 'coordinate_RMSE', 'max_point']),
                       source_paths=np.array([str(p.relative_to(ROOT)) for p in (GRID, COARSE, FINE)]),
                       source_sha256=np.array([sha256(p) for p in (GRID, COARSE, FINE)]),
                       barrel_s_coefficients=np.array([1., -.081595476571]),
                       barrel_k_coefficients=np.array([-.0185004967, .00023989235714, .0015756063948,
                                                      -.0014256378340, .00066200789981]))
        zero_lines = []
        for k, f in enumerate(FAMILIES):
            for key in ('error', 'test8', 'test9', 'constant', 'coefficient', 'min_den'):
                payload[f'family{k}_{key}'] = np.asarray(primary[f][key])
            for label, result in (('fine_coarse', fc), ('fine_refit', ff)):
                for count in (8, 9):
                    payload[f'family{k}_{label}_test{count}'] = result[f][f'test{count}']
            c = fine['coefficients_' + f][0]
            pred, _ = forward(primary['reference'] / L, c, f)
            zmax = float(np.linalg.norm(pred * L - primary['reference'], axis=-1).max() * 1000)
            payload[f'family{k}_fine_constant'] = c
            payload[f'family{k}_fine_zero_gaze_max_um'] = np.array(zmax)
            zero_lines.append(f'- {f}: coarse intercept minus identity `{(primary[f]["constant"]-IDENTITY[:len(c)]).tolist()}`; fine intercept minus identity `{(c-IDENTITY[:len(c)]).tolist()}`. Fine zero-gaze max displacement {zmax:.6f} um.')
        np.savez_compressed(OUT / 'p4_combined_brown_coefficients.npz', **payload)
        lines = ['# Combined radial-Brown accommodation: Step-1 stop', '',
                 'Reading: "radial Brown" = section-1k H+Brown+r2; "combine" = one transform for 0..4 D and +-20 deg both axes.', '',
                 f'Pipeline check: **{"PASS" if passed else "FAIL"}**. Direct saved-coefficient predictions on the requested A=0, 101x101 slice failed the requested baseline. No cached residuals were used to evaluate the check.', '',
                 '**Steps 2-5 were NOT RUN**, following the explicit STOP instruction. No combined model was fitted. The NPZ contains prerequisite evidence, not a validated combined transform.', '',
                 '## Direct pipeline evaluation', '',
                 f'The source has 214,221 states and finite real x/y throughout. A=0: 10,201 states; {primary["train"].sum():,} even/even training, {primary["test"].sum():,} held-out. BDE-major/ADE-minor ordering and every field label are checked against source records. Physical x/y are unchanged; t_x=BDE/20, t_y=ADE/20. Explicit center subtraction; fixed measured zero-gaze reference; no translation. L={L:.15g} mm.', '',
                 'Errors in um. Metric A uses eight non-center fields; metric B uses all nine. Center error is exactly zero. All rows below are held-out.', '',
                 '| grid / coefficients | family | fields | x RMSE | y RMSE | vertex RMS | coordinate RMSE | max point |',
                 '|---|---|---:|---:|---:|---:|---:|---:|']
        table(lines, '101x101 / coarse (PRIMARY)', primary)
        lines += ['', '| family | requested vertex / max | actual vertex / max | difference vertex / max | gate |',
                  '|---|---:|---:|---:|---|']
        for k, f in enumerate(FAMILIES):
            pairs = (EXPECTED[k], actual[k], actual[k] - EXPECTED[k])
            lines.append(f'| {f} | ' + ' | '.join(f'{v[0]:.6f} / {v[1]:.6f}' for v in pairs) +
                         f' | {"PASS" if np.all(np.abs(actual[k]-EXPECTED[k]) <= TOL) else "FAIL"} |')
        lines += ['', 'Tolerances are 0.01 um vertex RMS and 0.10 um max point, both wider than the displayed rounding. All three vertex RMS values fail independently of the max-point comparison.', '',
                  '## Diagnostic: the quoted baseline belongs to the fine grid', '',
                  'The fine401 report explicitly says its saved-coarse comparison uses fine held-out states, not coarse-grid test metrics. The following values were independently recomputed from coefficients and measured fine coordinates, excluding the same three incomplete-real states. They do not replace the failed primary check.', '',
                  '| grid / coefficients | family | fields | x RMSE | y RMSE | vertex RMS | coordinate RMSE | max point |',
                  '|---|---|---:|---:|---:|---:|---:|---:|']
        table(lines, '401x401 / coarse', fc)
        table(lines, '401x401 / fine refit', ff)
        lines += ['', 'The requested 5.94 / 3.48 / 2.13 um vertex RMS values belong to the fine grid. The 11.8 um radial maximum corresponds to the fine refit (~11.7598); saved coarse coefficients on the fine grid give ~11.6340. The 101x101 holdout instead reproduces the coarse report (~6.0418 / 3.5227 / 2.1552). Sampling and holdout populations change both RMS and sampled maxima. This is a prerequisite grid/baseline mismatch, not evidence of a failed accommodation model.', '',
                  '## Step 0: gauge', '',
                  'In fit_p4_ade_a0_decenter.py, bi_poly_basis includes total degree zero and the least-squares coefficient fit leaves its intercept free. The section-1k map therefore is not identity at zero gaze. Its constants are retained ONLY to reproduce that model in Step 1. Parameter order: m11,m12,m21,m22,qx,qy,p1,p2,g_r2 (truncated for smaller families).', '']
        lines += zero_lines
        lines += ['', 'For combined candidates the required gauge is p(A,t)=p_identity + sum_{i+j=1..4,l=0..d} c_ijl t_x^i t_y^j P_l(A/2-1), d=1..3. Thus the gaze deviations have NO t^0 terms: M=I and qx=qy=p1=p2=g=0 at zero gaze for all A. B_A owns zero gaze. C0 must also replace the fine-fit intercept with identity before freezing gaze coefficients. No constrained candidate was fitted after the gate failed.', '',
                  'Recommended physical-gauge B_A from data/p4_onaxis/report.md: v=A/4; s=1-0.081595476571 v; k=-0.0185004967+0.00023989235714 v+0.0015756063948 v^2-0.0014256378340 v^3+0.00066200789981 v^4. With D_k(u)=(1+k||u||^2)u, use u0=D_k(0)^-1(r0) on the small continuous radial branch and B_A(r0)=D_k(A)(s(A)u0). This fixes k(0)=-0.0185004967 /mm^2 and B_0=identity. Formula recorded only; not refitted or used to bypass Step 1.', '',
                  '## Decision-rule outcome and outstanding work', '',
                  'Undetermined. C0 versus C1, A-order selection, C1-rev, C1-ref, analytic-Jacobian verification, the 214,221-state benchmark, iteration-cap counts, A-smoothness misfit, accommodation/gaze-bin tables, worst-state and high-A-corner checks, and accommodation inversion were NOT RUN. No new claim about the previous -146 mD corner bias is justified.', '',
                  'To proceed, reconcile the prerequisite: retain the requested 101x101 dataset and adopt its coarse held-out baseline, or explicitly validate the quoted fine401 baseline separately before fitting the multi-accommodation data. This run makes neither protocol change.', '',
                  '## Reproduction and artifacts', '', '```powershell', "$env:PYTHONDONTWRITEBYTECODE = '1'",
                  "& 'C:\\Users\\rueijrwu\\.pyenv-win-venv\\envs\\venv\\Scripts\\python.exe' -B -u Script\\fit_p4_combined_brown_accommodation.py --pipeline-only", '```', '',
                  'Exit code 2 indicates the failed prerequisite. No CODE V was imported or started; no existing scripts, reports, or datasets were modified; no commit was made. The new script and this directory are the only deliverables. p4_combined_brown_coefficients.npz contains input hashes, axes, masks, direct residuals, metrics, constants and saved coefficients; it contains no fitted C model.', '',
                  '| read-only input | SHA-256 |', '|---|---|']
        for path, digest in zip(payload['source_paths'], payload['source_sha256']):
            lines.append(f'| `{path}` | `{digest}` |')
        (OUT / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('Pipeline', 'PASS' if passed else 'FAIL', flush=True)
    for f, m in zip(FAMILIES, actual):
        print(f'{f}: vertex RMS {m[0]:.6f}, max point {m[1]:.6f} um', flush=True)
    print('Steps 2-5 NOT RUN; see data/p4_combined_brown/report.md', flush=True)
    return 0 if passed else 2


if __name__ == '__main__':
    raise SystemExit(main())
