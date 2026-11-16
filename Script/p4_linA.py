r"""F versus symmetry-constrained FL; CURRENT_PLAN.md Section 16 revision 4. --selftest is synthetic only."""
import sys
sys.dont_write_bytecode = True
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import stat
import p4_final as fin
import numpy as np

ac, tw = fin.ac, fin.tw
OUT = Path(__file__).absolute().parents[1]/'runs'/'p4_linA'
FILES = ('summary.txt', 'constants.json', 'progress.log')
LABEL = ('noise-free design traces; ground-truth gaze used for calibration targets and scoring only; '
         'development data, non-anchor A previously inspected, not fresh validation')
POOLS = dict(fin.POOLS, anchors=(0, 5, 10, 15, 20), nonanchors=tuple(i for i in range(21) if i % 5))
COUNTS = dict(FL=[24, 30, 35], F=[80, 86, 91])
MASK = np.zeros((10, 2), bool)
for term, axis in tw.ALLOWED.values():
    MASK[tw.TERMS3.index(term), axis] = True

def guard():
    for path in [OUT, *OUT.parents, *(OUT/name for name in FILES)]:
        ac.require(path.resolve() == path.absolute(), 'Redirected output: '+str(path))
        if path.exists() or path.is_symlink():
            info = path.lstat()
            ac.require(not path.is_symlink() and not (getattr(info, 'st_file_attributes', 0) & 0x400), 'Output reparse point: '+str(path))
            ac.require((stat.S_ISREG(info.st_mode) and info.st_nlink == 1) if path.name in FILES else path.is_dir(), 'Invalid output target: '+str(path))

def encoded(value):
    return json.dumps(fin.plain(value), separators=(',', ':'), allow_nan=False)

def log(message, heartbeat=False):
    if not heartbeat:
        line = datetime.now(timezone.utc).isoformat(timespec='seconds')+' '+message
        print(line, flush=True)
        guard()
        with (OUT/'progress.log').open('a', encoding='utf-8') as stream:
            stream.write(line+'\n')

def calibrate(states, cal, anchors):
    a = np.array([states[i]['A'] for i in anchors]); matrix = np.column_stack((np.ones(5), a))
    observed = np.stack([states[i]['fit']['coef'] for i in anchors]).copy()
    dropped = np.max(abs(observed[:, ~MASK]), axis=1); observed[:, ~MASK] = 0.
    c, cc = ac.linear_fit(matrix, observed.reshape(5, 20), 'FL linear gaze laws')
    c = c.reshape(2, 10, 2)
    assert np.all(c[:, ~MASK] == 0.), 'Forbidden FL gaze laws must remain exact zero'
    model = dict(coefficient=c, scale=states[0]['scale'], bounds=(0., 4.), record=dict(**cc, degree=1, anchor_indices=anchors, anchor_dioptres=a, parameter_units='normalised gaze deg / D**power'))
    u, uc = ac.linear_fit(matrix, cal['mu'][cal['anchors']], 'FL linear size law')
    runtime = dict(anchor_indices=anchors, accommodation_d=a, mu=u[0]+u[1]*a, law='u0+u1*A', coefficients=u, interpolation='samples of one fitted line')
    cal_fl = dict(cal, mu=u[0]+u[1]*cal['A'], record=dict(runtime=runtime, measured_anchor_mu=cal['mu'][cal['anchors']]))
    values, checks = [], {}
    for i in anchors:
        state = states[i]; train = state['groups']['train']; tau = tw.tangent_coordinates(state['w'][train])['primary']
        g, check = ac.linear_fit(ac.s1_size_design(tau, 'E4'), state['mu_raw'][train]/cal_fl['mu'][i]-1, f'FL gamma anchor A{i:02d}')
        values.append(g); checks[f'A{i:02d}'] = dict(**check, gamma=g)
    gamma, gc = ac.linear_fit(matrix, np.stack(values), 'FL linear gamma laws')
    record = dict(basis='E4', anchor_indices=anchors, anchors=checks, gamma_anchors=np.stack(values), gamma_trend=gamma, trend_check=dict(**gc, degree=1))
    surface = dict(basis='E4', gamma_A00=values[0], trend=np.vstack((gamma, np.zeros(5))), grid=np.arange(401)*.01, record=record)
    laws = dict(gaze=c, gamma=gamma, mu=u, dropped_normalised_deg=dict(zip(map(str, anchors), dropped)), gaze_fit=model['record'], mu_fit=uc, gamma_fit=record,
                runtime_padding='third gamma row is exact structural zero; not fitted or independent', mu_runtime=cal_fl['record'])
    return model, cal_fl, surface, laws

def scored(view, loop, rc):
    result = fin.score(view, loop, rc)
    error = 60*np.linalg.norm(loop['gaze']-view['w'], axis=1); valid = loop['reason'] == 0
    result.update(failed=int((~valid).sum()), gaze_counts_gt_arcmin={str(t): int(np.sum(valid & (error > t))) for t in (1., .1)})
    return result

def pooled(entries, rc):
    return scored(*[{k: np.concatenate([entry[j][k] for entry in entries]) for k in entries[0][j]} for j in (0, 1)], rc)

def verdict(l0, l1, maximum):
    return 'NOT CLOSED: INVALID RUN' if not l0 else 'CLOSE: FL' if l1 and maximum < 1. else 'CLOSE: F'

def selftest():
    anchors = [0, 5, 10, 15, 20]; a = np.arange(5.); grid = np.linspace(-3., 3., 5)
    m = np.array([(x, y) for x in grid for y in grid]); n = len(m); design = ac.gp.design(m, tw.TERMS3)
    c = np.zeros((2, 10, 2)); c[:, MASK] = np.array([[1., 1.1, .001, .002, .003, .004], [.01, .02, .0001, .0002, .0003, .0004]])
    g = np.array([[.1, .15, .02, .03, .04], [.01, .02, .003, .004, .005]]); u = np.array([1., -.02])
    cal = dict(A=np.arange(21)*.2, mu=1-.02*np.arange(21)*.2, anchors=np.array(anchors)); states = {}
    for i, ai in zip(anchors, a):
        w = design@(c[0]+ai*c[1]); tau = tw.tangent_coordinates(w)['primary']; coefficient = c[0]+ai*c[1]; coefficient[~MASK] = .123
        states[i] = dict(A=ai, fit=dict(coef=coefficient), scale=np.ones(2), w=w, groups=dict(train=np.ones(n, bool)), mu_raw=(u[0]+u[1]*ai)*(1+ac.s1_size_design(tau, 'E4')@(g[0]+ai*g[1])))
    model, cal_fl, surface, laws = calibrate(states, cal, anchors)
    assert np.max(abs(model['coefficient']-c)) < 1e-12 and np.all(model['coefficient'][:, ~MASK] == 0.) and np.all(surface['trend'][2] == 0.)
    assert laws['gamma'].shape == (2, 5) and laws['gaze_fit']['rank'] == 2 and laws['gaze_fit']['columns'] == 2 and np.allclose(laws['mu'], u, atol=1e-12)
    assert all(value == .123 for value in laws['dropped_normalised_deg'].values()) and laws['gamma_fit']['trend_check']['columns'] == 2
    ai = np.full(n, 2.375); w = design@(c[0]+ai[0]*c[1]); tau = tw.tangent_coordinates(w)['primary']; e = ac.s1_size_design(tau, 'E4')
    assert np.max(abs(ac.evaluate_gaze(model, m, ai)-w)) < 1e-12
    expected = (u[0]+u[1]*ai)*(1+e@(g[0]+ai[0]*g[1]))
    assert np.max(abs(fin.size_value(cal_fl, tau, ai, surface=surface)-expected)) < 1e-12 and np.any(abs(u[1]*(e@g[1])) > 1e-10)
    table_expected = (u[0]+u[1]*surface['grid'])[None, :]*(1+e@(g[0]+surface['grid'][:, None]*g[1]).T)
    assert np.max(abs(ac.s1_surface_table(surface, cal_fl, w)-table_expected)) < 1e-12
    exact = dict(m=m, mu_raw=expected); loop = fin.run_loop(exact, cal_fl, model, surface)
    assert not np.any(loop['reason']) and np.max(abs(loop['estimate']-ai)) < 1e-5
    view = dict(w=w, A=ai, shard=np.full(n, 'synthetic'), state_ids=np.arange(n)); entry = (view, {k: loop[k] for k in ('estimate', 'reason', 'gaze', 'passes', 'clipped', 'size_residual')})
    assert pooled([entry, entry], 1.)['states'] == 2*n and scored(view, loop, 1.)['gaze_counts_gt_arcmin']['1.0'] == 0
    bad = fin.run_loop(dict(m=np.array([[0., 0.], [np.nan, 0.]]), mu_raw=np.array([u.sum()-1., 1.])), cal_fl, model, surface)
    bad_view = dict(w=np.zeros((2, 2)), A=np.zeros(2), shard=np.full(2, 'synthetic'), state_ids=np.arange(2)); metric = scored(bad_view, bad, 1.)
    assert bad['clipped'][0] and bad['estimate'][0] == 4. and metric['failed'] == 1 and np.isinf(metric['gaze']['rms']) and np.isinf(metric['gaze']['max'])
    perturbed = np.stack([states[i]['fit']['coef'] for i in anchors]).reshape(5, 20).copy(); perturbed[2, 2] += .1
    matrix = np.column_stack((np.ones(5), a)); fit, check = ac.linear_fit(matrix, perturbed, 'synthetic non-collinear trend')
    assert fit.shape == (2, 20) and check['rank'] == 2 and np.max(abs(matrix@fit-perturbed)) > .01
    try:
        ac.linear_fit(np.ones((5, 2)), a, 'synthetic deficient rank')
    except ac.NumericalFailure:
        pass
    else:
        raise AssertionError('Rank-deficient fit accepted')
    assert COUNTS['FL'] == [12+10+2, 30, 35] and verdict(False, True, 0.) == 'NOT CLOSED: INVALID RUN' and verdict(True, True, .99) == 'CLOSE: FL' and verdict(True, True, 1.) == verdict(True, False, 0.) == 'CLOSE: F'
    return 'PASS: linear laws, structural zeros, induced product, production loop, clipping/failures, pooled counts, gates and verdicts'

def real_run(lines, constants):
    args = ac.arguments(['--stages', 'S2']); report = ac.new_report(args, 'linA'); numerical = []
    def loader(*arguments):
        try:
            return ac.five.load_shard(*arguments)
        except Exception as exc:
            if arguments[1].get('calibration') or isinstance(exc, (ac.NumericalFailure, FloatingPointError, np.linalg.LinAlgError)):
                numerical.append(str(exc))
            raise
    states = ac.prepare_all(report, log, loader)
    if states is None:
        constants.update(status='invalid_data', errors=report['errors']); lines.append('data errors='+encoded(report['errors'])); return (3 if numerical else 2), False, False, float('inf')
    if any(len(s['w']) != 10201 or not np.isclose(s['A'], .2*i, rtol=0., atol=1e-12) for i, s in states.items()):
        constants.update(status='invalid_data'); lines.append('Invalid published shard sizes/accommodations'); return 2, False, False, float('inf')
    log('preparation: fixed camera, shared reference, own-A cubic checks')
    ac.shared_preparation(states, report, stage_s1=True); cal = ac.size_calibration(states, args); projections = {}
    for i, state in states.items():
        state['groups']['field_lt12'] = np.logical_or.reduce([state['groups'][name] for name in ac.gp.BIN_NAMES[:3]])
        state['mu_raw'], _, _, check = ac.two_radius_projection(state['Zshape'], cal['H0'], cal['q0']); projections[f'A{i:02d}'] = check
    log('calibration: anchor-only F and FL')
    model = ac.coefficient_model(states, args); surface = ac.s1_fit_surface(states, cal, args, 'E4')
    model_fl, cal_fl, surface_fl, laws = calibrate(states, cal, args.anchors)
    constants.update(FL=laws, F=dict(gaze=model['coefficient'], gamma=surface['trend'], measured_anchor_mu=cal['mu'][cal['anchors']]), terms=tw.TERMS3, axes=['wx', 'wy'],
        scale_mm=model['scale'], origin_mm=states[0]['origin'], H0_mm=cal['H0'], q0=cal['q0'], camera=report['camera'], anchor_indices=args.anchors, anchor_A=cal['A'][cal['anchors']],
        grid=dict(start=0., stop=4., step=.01, count=401), loop=dict(start=0., tolerance_D=ac.S2_TOLERANCE, max_passes=ac.S2_MAX_PASSES, endpoint='clip'), independent_counts=COUNTS,
        units=dict(gaze_law='intercept deg; slope deg/D, normalized sensor basis', gamma_mu_laws='intercept dimensionless; slope 1/D', score_gaze='arcmin', score_A='D', size_residual='um'),
        checks=dict(preparation=report['preparation'], projections=projections, F_gaze=model['record'], F_gamma=surface['record']))
    lines.extend(['settings: A=0..4 D; field +/-12.5 deg; 21x10201 states; anchor training gaze only; gaze arcmin; A D; residual um', 'constants and fit records='+encoded(constants)])
    rc = 1000*np.sqrt(2)*cal['H0']*(1+2*cal['q0']); rows = {}
    for name, mod, calibration, surf in (('F', model, cal, surface), ('FL', model_fl, cal_fl, surface_fl)):
        entries = {}; rows[name] = {}
        for i, state in states.items():
            log(f'{name} A{i:02d}'); loop = fin.run_loop(state, calibration, mod, surf); n = len(state['w'])
            view = dict(w=state['w'], A=np.full(n, state['A']), shard=np.full(n, f'A{i:02d}'), state_ids=state['state_ids'])
            entries[i] = (view, {k: loop[k] for k in ('estimate', 'reason', 'gaze', 'passes', 'clipped', 'size_residual')})
            lines.append(f'{name} A{i:02d} '+('anchor ' if i in args.anchors else 'nonanchor ')+encoded(scored(view, loop, rc)))
        for pool, indices in POOLS.items():
            rows[name][pool] = pooled([entries[i] for i in indices], rc); lines.append(f'{name} pool={pool} '+encoded(rows[name][pool]))
    for pool in POOLS:
        delta = {metric: {key: rows['FL'][pool][metric][key]-rows['F'][pool][metric][key] for key in ('rms', 'max')} for metric in ('gaze', 'A')}
        delta.update({key: rows['FL'][pool][key]-rows['F'][pool][key] for key in ('valid', 'failed', 'coverage', 'clipped')})
        delta['gaze_counts_gt_arcmin'] = {key: rows['FL'][pool]['gaze_counts_gt_arcmin'][key]-rows['F'][pool]['gaze_counts_gt_arcmin'][key] for key in ('1.0', '0.1')}
        delta['passes_median_max'] = np.subtract(rows['FL'][pool]['passes_median_max'], rows['F'][pool]['passes_median_max'])
        delta['size_equation_max_um'] = {key: (rows['FL'][pool][key]['max_um']-rows['F'][pool][key]['max_um'] if rows['FL'][pool][key] and rows['F'][pool][key] else None) for key in ('size_equation_unclipped', 'size_equation_clipped')}
        lines.append('FL-minus-F pool='+pool+' '+encoded(delta))
    f = rows['F']; checks = {pool+'_gaze_max': abs(f[pool]['gaze']['max']-v) <= 1e-6 for pool, v in fin.AUDITED.items()}
    checks.update(overall_gaze_max=abs(f['all21']['gaze']['max']-.1903982096) <= 1e-6, interior_gaze_rms=abs(f['A01-A19']['gaze']['rms']-.01926065) <= 1e-6,
        interior_A_rms=abs(f['A01-A19']['A']['rms']-.00197210) <= 1e-8, interior_A_max=abs(f['A01-A19']['A']['max']-.03299789) <= 1e-8,
        passes=f['all21']['passes_median_max'] == [6., 12], F_coverage=f['all21']['valid'] == f['all21']['states'] == 214221)
    l0 = all(checks.values()); l1 = rows['FL']['all21']['valid'] == rows['FL']['all21']['states'] == 214221; maximum = rows['FL']['all21']['gaze']['max']
    constants.update(status='completed' if l0 else 'reproduction_failed', reproduction=checks, pooled=rows, L0=l0, L1=l1, L2=maximum < 1.)
    lines.extend(['F reproduction (gaze tolerance 1e-6 arcmin; A tolerance 1e-8 D; passes exact)='+encoded(checks), f'L0={l0}; L1={l1}; L2={maximum < 1.}; A accuracy has no acceptance threshold', 'independent counts FL/F='+encoded(COUNTS), 'limits: development only; no noise, real-eye, off-grid, contraction or unique-fixed-point claim; CLOSE: F retains quadratic F and does not certify FL'])
    return (0 if l0 else 3), l0, l1, maximum

def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--selftest', action='store_true'); options = parser.parse_args()
    guard(); OUT.mkdir(parents=True, exist_ok=True); guard(); (OUT/'progress.log').write_text(LABEL+'\n', encoding='utf-8')
    lines = [LABEL]; constants = dict(label=LABEL, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), status='starting'); code, l0, l1, maximum = 4, False, False, float('inf')
    try:
        tested = selftest(); lines.append('selftest='+tested); log('selftest '+tested)
        if options.selftest:
            code = 0; constants.update(status='synthetic_only'); lines.append('Synthetic only; no trace I/O or completed production comparison')
        else:
            code = 3; code, l0, l1, maximum = real_run(lines, constants)
    except Exception as exc:
        constants.update(status='failed', error=str(exc), numbers=getattr(exc, 'numbers', {})); lines.append(f'exit={code} {type(exc).__name__}: {exc}; numbers='+encoded(getattr(exc, 'numbers', {})))
    outcome = verdict(l0, l1, maximum); constants.update(exit_code=code, verdict=outcome); lines.extend(['status='+constants['status']+'; script_sha256='+constants['script_sha256'], outcome]); log('terminal status='+constants['status']+'; exit='+str(code))
    guard(); (OUT/'constants.json').write_text(json.dumps(fin.plain(constants), indent=2, allow_nan=False)+'\n', encoding='utf-8'); (OUT/'summary.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return code

if __name__ == '__main__':
    raise SystemExit(main())
