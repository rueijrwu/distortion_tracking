r"""Closing F / LF-L2 / LF-L3 fit/test, CURRENT_PLAN.md 14 and P4_MODEL.md 9.

Run once from the project root (selftests precede all trace I/O):
& 'C:\Users\rueijrwu\.pyenv-win-venv\envs\venv\Scripts\python.exe' Script\p4_final.py
Add --selftest for synthetic checks alone. Only runs/p4_final is writable.
"""
import sys
sys.dont_write_bytecode = True
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import stat
import p4_accom_test as ac
import numpy as np

tw, joint = ac.tw, ac.five.joint
ROOT = Path(__file__).absolute().parents[1]
OUT = ROOT/'runs'/'p4_final'
FILES = ('summary.txt', 'constants.json', 'progress.log')
LABEL = 'noise-free design traces; ground-truth gaze used for calibration targets and scoring only'
POOLS = {'A00': (0,), 'A01-A19': tuple(range(1, 20)), 'A20': (20,), 'all21': tuple(range(21))}
AUDITED = {'A00': .05193063, 'A01-A19': .12489784, 'A20': .19039821}


def guard():
    for path in [OUT, *OUT.parents, *(OUT/name for name in FILES)]:
        ac.require(path.resolve() == path.absolute(), 'Redirected output: '+str(path))
        if path.exists() or path.is_symlink():
            info = path.lstat()
            ac.require(not path.is_symlink() and not (getattr(info, 'st_file_attributes', 0) & 0x400),
                       'Output reparse point: '+str(path))
            ac.require((stat.S_ISREG(info.st_mode) and info.st_nlink == 1) if path.name in FILES
                       else path.is_dir(), 'Invalid output file/parent: '+str(path))


def log(message, heartbeat=False):
    if heartbeat:
        return
    line = datetime.now(timezone.utc).isoformat(timespec='seconds')+' '+message
    print(line, flush=True)
    guard()
    with (OUT/'progress.log').open('a', encoding='utf-8') as stream:
        stream.write(line+'\n')


def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return 'NaN' if np.isnan(value) else 'Infinity' if value > 0 else '-Infinity'
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def fit_check(record):
    return {key: record.get(key) for key in ('rank', 'columns', 'condition')}


def lf_calibrate(A, tau, m, q, gamma, order):
    """Anchor training arrays only; every fit uses the unchanged linear gate."""
    A, q = np.asarray(A, float), np.asarray(q, float)
    values, checks = [], {}

    def fit(matrix, target, name):
        try:
            coefficient, checked = ac.linear_fit(matrix, target, f'LF L{order} '+name)
        except ac.NumericalFailure as exc:
            exc.numbers.update(completed_fit_checks=checks.copy(), failed_fit=name)
            raise
        checks[name] = fit_check(checked)
        return coefficient
    for j, (t, observed) in enumerate(zip(tau, m)):
        r2 = np.sum(t*t, axis=1)
        matrix = np.stack([t*r2[:, None]**k for k in range(order)], axis=-1).reshape(-1, order)
        g = fit(matrix, np.asarray(observed).reshape(-1), f'mean_anchor_{j}')
        values.append(np.pad(g, (0, 3-order)))
    values = np.asarray(values)
    powers = np.column_stack((np.ones(len(A)), A, A*A))
    gtrend = fit(powers, values, 'g_laws')
    qtrend = fit(powers, q, 'q_law')
    eps = np.divide(values[:, 1], values[:, 0], out=np.full(len(A), np.nan), where=values[:, 0] != 0)
    offsets = np.column_stack((np.ones(len(A)), q-q[0], eps-eps[0]))
    coupling = fit(offsets, gamma, 'gamma_coupling')
    return dict(order=order, g_trend=gtrend, q_trend=qtrend, gamma=coupling,
                q0=float(q[0]), eps0=float(eps[0]), fit_checks=checks)


def lf_constants(law, a):
    a = np.atleast_1d(np.asarray(a, float))
    c, d = law['g_trend'], law['q_trend']
    g = c[0]+a[:, None]*(c[1]+a[:, None]*c[2])
    q = d[0]+a*(d[1]+a*d[2])
    good = np.isfinite(a) & np.isfinite(g).all(axis=1) & (g[:, 0] != 0)
    eps = np.divide(g[:, 1], g[:, 0], out=np.full(len(a), np.nan), where=good)
    eps5 = np.divide(g[:, 2], g[:, 0], out=np.full(len(a), np.nan), where=good)
    coupling = law['gamma']
    gamma = coupling[0]+(q-law['q0'])[:, None]*coupling[1]+(eps-law['eps0'])[:, None]*coupling[2]
    return g[:, 0], eps, eps5, q, gamma


def lf_radial_inverse(m, a, G, eps, eps5):
    """Split failing batches to single states; never discard valid neighbours."""
    tau = np.full_like(m, np.nan, dtype=float)
    valid = (np.isfinite(a) & np.isfinite(m).all(axis=1) & np.isfinite(G) & (G != 0)
             & np.isfinite(eps) & np.isfinite(eps5))
    ids = np.flatnonzero(valid)
    v = np.full_like(tau, np.nan)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        v[ids] = m[ids]/G[ids, None]
    valid &= np.isfinite(v).all(axis=1)

    def retry(indices):
        if not len(indices):
            return
        try:
            result, info = joint.radial_inverse(v[indices], eps[indices], eps5[indices])
        except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
            if len(indices) == 1:
                valid[indices] = False
            else:
                middle = len(indices)//2
                retry(indices[:middle])
                retry(indices[middle:])
            return
        gaze = tw.tangent_inverse(result, 1., 'primary')
        good = (np.asarray(info['converged']) & (np.asarray(info['min_tangential']) > 0)
                & (np.asarray(info['min_radial']) > 0) & np.isfinite(result).all(axis=1)
                & np.isfinite(gaze).all(axis=1))
        valid[indices] = good
        tau[indices[good]] = result[good]
    retry(np.flatnonzero(valid))
    tau[~valid] = np.nan
    return tau, valid


def lf_gaze(law, m, a):
    m = np.asarray(m, float)
    a = np.broadcast_to(np.asarray(a, float), (len(m),))
    constants = [np.full(len(m), np.nan) for _ in range(3)]
    ids = np.flatnonzero(np.isfinite(a) & np.isfinite(m).all(axis=1))
    if len(ids):
        for target, value in zip(constants, lf_constants(law, a[ids])[:3]):
            target[ids] = value
    tau, valid = lf_radial_inverse(m, a, *constants)
    gaze = tw.tangent_inverse(tau, 1., 'primary')
    valid &= np.isfinite(gaze).all(axis=1)
    gaze[~valid], tau[~valid] = np.nan, np.nan
    return gaze, tau, valid


def lf_mean(law, gaze, a):
    G, eps, eps5 = lf_constants(law, a)[:3]
    tau = tw.tangent_coordinates(gaze)['primary']
    return G[:, None]*joint.radial(tau, eps, eps5, check=False)


def size_value(cal, tau, a, law=None, surface=None):
    a = np.broadcast_to(np.asarray(a, float), (len(tau),))
    if law is None:
        c = surface['trend']
        gamma = c[0]+a[:, None]*(c[1]+a[:, None]*c[2])
    else:
        gamma = lf_constants(law, a)[4]
    mu = np.interp(a, cal['A'][cal['anchors']], cal['mu'][cal['anchors']])
    return mu*(1+np.sum(ac.s1_size_design(tau, 'E4')*gamma, axis=1))


def lf_table(law, cal, tau):
    grid = np.arange(401, dtype=float)*.01
    gamma = lf_constants(law, grid)[4]
    table = np.ones((len(tau), len(grid)))
    for column, coefficient in zip(ac.s1_size_design(tau, 'E4').T, gamma.T):
        table += column[:, None]*coefficient[None, :]
    table *= np.interp(grid, cal['A'][cal['anchors']], cal['mu'][cal['anchors']])[None, :]
    return grid, table


def lf_map(state, law, cal, a, ids):
    _, tau, valid = lf_gaze(law, state['m'][ids], a)
    value, reason, clipped = np.full(len(ids), np.nan), np.ones(len(ids), np.int8), np.zeros(len(ids), bool)
    if np.any(valid):
        grid, table = lf_table(law, cal, tau[valid])
        mu = state['mu_raw'][ids][valid]
        value[valid], reason[valid], _ = ac.s1_table_inverse(grid, table, mu, clip=True)
        clipped[valid] = (mu < np.min(table, axis=1)) | (mu > np.max(table, axis=1))
    return value, reason, clipped


def run_loop(state, cal, model=None, surface=None, law=None):
    count = len(state['m'])
    clipped, captured = np.zeros(count, bool), np.zeros(count, bool)

    def operation(a, ids):
        result = (ac.s2_map(state, model, cal, surface, 'clip', a, ids) if law is None
                  else lf_map(state, law, cal, a, ids))
        value, reason = result[:2]
        stopped = (reason == 0) & (abs(value-a) <= ac.S2_TOLERANCE) & ~captured[ids]
        if np.any(stopped):
            chosen = ids[stopped]
            if law is None:
                gaze = ac.evaluate_gaze(model, state['m'][chosen], a[stopped])
                table = ac.s1_surface_table(surface, cal, gaze)
                mu = state['mu_raw'][chosen]
                clipped[chosen] = (mu < np.min(table, axis=1)) | (mu > np.max(table, axis=1))
            else:
                clipped[chosen] = result[2][stopped]
            captured[chosen] = True
        return value, reason
    loop = ac.s2_loop(operation, count, 0.)
    if law is None:
        gaze = ac.evaluate_gaze(model, state['m'], loop['estimate'])
        valid = np.isfinite(gaze).all(axis=1)
    else:
        gaze, _, valid = lf_gaze(law, state['m'], loop['estimate'])
    loop['reason'][(loop['reason'] == 0) & ~valid] = 1
    bad = loop['reason'] != 0
    loop['estimate'][bad], gaze[bad], clipped[bad] = np.nan, np.nan, False
    tau = tw.tangent_coordinates(gaze)['primary']
    residual = abs(size_value(cal, tau, loop['estimate'], law, surface)-state['mu_raw'])
    loop.update(gaze=gaze, clipped=clipped, size_residual=residual)
    return loop


def score(state, loop, Rc, mean=None, ring=None):
    def location(index):
        return dict(shard=str(state['shard'][index]), state_id=int(state['state_ids'][index]),
                    gaze_deg=state['w'][index].tolist())
    valid = loop['reason'] == 0
    errors = dict(gaze=60*np.linalg.norm(loop['gaze']-state['w'], axis=1),
                  A=abs(loop['estimate']-state['A']))
    if mean is not None:
        errors.update(mean=mean, ring=ring)
    result = dict(states=len(valid), valid=int(valid.sum()), coverage=float(valid.mean()),
                  reasons={int(code): int(np.sum(loop['reason'] == code)) for code, _ in ac.S2_REASONS},
                  clipped=int(loop['clipped'].sum()))
    passes = loop['passes']
    result['passes_median_max'] = [float(np.median(passes)), int(np.max(passes))] if len(passes) else [None, None]
    result['passes_max_location'] = location(int(np.argmax(passes))) if len(passes) else None
    for name, values in errors.items():
        values = np.where(valid, values, np.nan)
        finite = np.isfinite(values)
        index = int(np.argmax(values)) if np.all(finite) else int(np.flatnonzero(~finite)[0])
        rms, maximum = ((float(np.sqrt(np.mean(values*values))), float(np.max(values)))
                        if np.all(finite) else (float('inf'), float('inf')))
        result[name] = dict(rms=rms, max=maximum, location=location(index))
    for label, mask in (('unclipped', valid & ~loop['clipped']), ('clipped', valid & loop['clipped'])):
        values = loop['size_residual'][mask]*Rc
        result['size_equation_'+label] = dict(max_um=float(np.max(values)),
            location=location(np.flatnonzero(mask)[np.argmax(values)])) if len(values) else None
    return result


def selftest():
    A = np.arange(5, dtype=float)
    law = dict(order=3, g_trend=np.array([[-31.5, 9., 6.], [.2, -1., .2], [.01, .1, .01]]),
               q_trend=np.array([.02, .001, .0005]), q0=.02, eps0=9./-31.5,
               gamma=np.array([[.6, .65, .05, .08, .06], [1., 2., .2, -.1, .2], [.2, .3, .05, .02, .04]]))
    cal = dict(A=A, mu=1-.02*A, anchors=np.arange(5))
    w = np.array([(x, y) for x in np.linspace(-12.5, 12.5, 5) for y in np.linspace(-12.5, 12.5, 5)])
    tau = tw.tangent_coordinates(w)['primary']
    corner = np.tile([[12.5, -12.5]], (3, 1))
    recovered, _, good = lf_gaze(law, lf_mean(law, corner, [0., 2., 4.]), [0., 2., 4.])
    assert np.all(good) and np.max(abs(recovered-corner)) <= 1e-9, 'ST1 mean inverse'
    observed = [lf_mean(law, w, np.full(len(w), a)) for a in A]
    fitted = lf_calibrate(A, [tau]*5, observed, lf_constants(law, A)[3], lf_constants(law, A)[4], 3)
    for key in ('g_trend', 'q_trend', 'gamma'):
        assert np.allclose(fitted[key], law[key], rtol=1e-8, atol=1e-12), 'ST2 '+key
    assert np.allclose([fitted['q0'], fitted['eps0']], [law['q0'], law['eps0']], rtol=1e-8), 'ST2 raw offsets'
    smaller = lf_calibrate(A, [tau]*5, observed, lf_constants(law, A)[3], lf_constants(law, A)[4], 2)
    assert np.max(abs(lf_mean(smaller, w, np.zeros(len(w)))-observed[0])) > 1e-8, 'ST2 L2 residual'
    truth_a, gaze = np.repeat([.6, 2.2, 3.4], len(w)), np.tile(w, (3, 1))
    t = tw.tangent_coordinates(gaze)['primary']
    state = dict(m=lf_mean(law, gaze, truth_a), mu_raw=size_value(cal, t, truth_a, law))
    loop = run_loop(state, cal, law=law)
    assert not np.any(loop['reason']) and np.max(abs(loop['estimate']-truth_a)) <= .002, 'ST3 A'
    assert np.max(60*np.linalg.norm(loop['gaze']-gaze, axis=1)) <= .02, 'ST3 gaze'
    clip = dict(m=np.array([[0., 0.], [np.nan, 0.]]), mu_raw=np.array([cal['mu'][-1]-.1, 1.]))
    loop = run_loop(clip, cal, law=law)
    assert loop['estimate'][0] == 4. and loop['reason'][0] == 0 and loop['clipped'][0], 'ST4 clip'
    assert loop['reason'][1] == 1 and np.isnan(loop['estimate'][1]), 'ST4 nonfinite'
    # ST5 uses an outside-range zero of G: an in-table zero would poison every
    # neighbour's size table. Only this synthetic operation injects assumed A=5.
    branch = dict(law, g_trend=np.array([[-1., 1., 0.], [.2, 0., 0.], [0., 0., 0.]]),
                  eps0=-1., gamma=np.array([[.1, .12, .01, .02, .03], [.4, .3, .02, .01, .02],
                                           [.005, .006, .001, .002, .001]]))
    mixed_w = np.array([[2., -3.], [-12.5, 12.5], [0., 0.], [0., 0.], [0., 0.], [6., 0.]])
    mixed_a = np.array([0., 0., 0., 0., 5., 0.])
    mixed = dict(m=lf_mean(branch, mixed_w, np.zeros(6)),
                 mu_raw=size_value(cal, tw.tangent_coordinates(mixed_w)['primary'], np.zeros(6), branch))
    mixed['m'][2], mixed['m'][3] = [np.nan, 0.], [1., 0.]
    valid_ids = np.array([0, 1, 5])
    direct, direct_tau, good = lf_gaze(branch, mixed['m'], mixed_a)
    alone, alone_tau, alone_good = lf_gaze(branch, mixed['m'][valid_ids], mixed_a[valid_ids])
    assert np.array_equal(np.flatnonzero(~good), [2, 3, 4]) and np.all(alone_good), 'ST5 direct mask'
    assert np.isnan(direct[~good]).all() and np.isnan(direct_tau[~good]).all(), 'ST5 direct NaN'
    assert np.array_equal(direct[valid_ids], alone) and np.array_equal(direct_tau[valid_ids], alone_tau), 'ST5 direct isolation'

    def synthetic_loop(selected):
        state = {key: value[selected] for key, value in mixed.items()}
        def operation(a, ids):
            assumed = a.copy()
            assumed[selected[ids] == 4] = 5.
            return lf_map(state, branch, cal, assumed, ids)
        result = ac.s2_loop(operation, len(selected), 0.)
        gaze, _, valid = lf_gaze(branch, state['m'], result['estimate'])
        result['reason'][(result['reason'] == 0) & ~valid] = 1
        bad = result['reason'] != 0
        result['estimate'][bad], gaze[bad] = np.nan, np.nan
        result.update(gaze=gaze, clipped=np.zeros(len(selected), bool), size_residual=np.zeros(len(selected)))
        return result
    mixed_loop, valid_loop = synthetic_loop(np.arange(6)), synthetic_loop(valid_ids)
    assert np.array_equal(mixed_loop['reason'], [0, 0, 1, 1, 1, 0]), 'ST5 loop reasons'
    assert np.isnan(mixed_loop['estimate'][[2, 3, 4]]).all() and np.isnan(mixed_loop['gaze'][[2, 3, 4]]).all(), 'ST5 loop NaN'
    for key in ('estimate', 'gaze', 'passes'):
        assert np.array_equal(mixed_loop[key][valid_ids], valid_loop[key]), 'ST5 loop isolation '+key
    view = dict(w=mixed_w, A=np.zeros(6), shard=np.full(6, 'synthetic'), state_ids=np.arange(6))
    pooled = score(view, mixed_loop, 1.)
    valid_only = score({k: v[valid_ids] for k, v in view.items()}, valid_loop, 1.)
    for metric in ('gaze', 'A'):
        assert np.isinf(pooled[metric]['rms']) and np.isinf(pooled[metric]['max']), 'ST5 full denominator'
        assert np.isfinite(valid_only[metric]['rms']) and np.isfinite(valid_only[metric]['max']), 'ST5 valid-only finite'
    return 'ST1-ST5 PASS'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selftest', action='store_true')
    options = parser.parse_args()
    guard()
    OUT.mkdir(parents=True, exist_ok=True)
    guard()
    (OUT/'progress.log').write_text(LABEL+'\n', encoding='utf-8')
    lines, constants = [LABEL, ac.DEVELOPMENT], {}
    code = 4
    try:
        tested = selftest()
        lines.append('selftest = '+tested)
        log(tested)
        if options.selftest:
            lines.append('synthetic only; no trace I/O')
            code = 0
        else:
            code = 3
            code = real_run(lines, constants)
    except Exception as exc:
        lines.append(f'exit={code} {type(exc).__name__}: {exc}; numbers={plain(getattr(exc, "numbers", {}))}')
        lines.append('NOT CLOSED: INVESTIGATE')
        log(lines[-2])
    guard()
    (OUT/'constants.json').write_text(json.dumps(plain(constants), indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (OUT/'summary.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return code


def real_run(lines, constants):
    args = ac.arguments(['--stages', 'S2'])
    report = ac.new_report(args, 'final')
    numerical_loading = []

    def loader(*arguments):
        try:
            return ac.five.load_shard(*arguments)
        except Exception as exc:
            if arguments[1]['calibration'] or isinstance(exc, (ac.NumericalFailure, FloatingPointError, np.linalg.LinAlgError)):
                numerical_loading.append(str(exc))
            raise
    states = ac.prepare_all(report, log, loader)
    if states is None:
        code = 3 if numerical_loading else 2
        lines.extend([f'data/preparation stop; exit={code}; errors={plain(report["errors"])}', 'NOT CLOSED: INVESTIGATE'])
        return code
    try:
        ac.shared_preparation(states, report, stage_s1=True)
        model, cal = ac.coefficient_model(states, args), ac.size_calibration(states, args)
        projection_checks = {}
        for index, state in states.items():
            state['groups']['field_lt12'] = np.logical_or.reduce([state['groups'][name] for name in ac.gp.BIN_NAMES[:3]])
            state['mu_raw'], _, state['radial_residual_um'], check = ac.two_radius_projection(state['Zshape'], cal['H0'], cal['q0'])
            projection_checks[f'A{index:02d}'] = fit_check(check)
        surface = ac.s1_fit_surface(states, cal, args, 'E4')
    except Exception as exc:
        lines.extend([f'F preparation/calibration failure: {exc}; numbers={plain(getattr(exc, "numbers", {}))}', 'NOT CLOSED: INVESTIGATE'])
        return 3
    Rc = 1000*np.sqrt(2)*cal['H0']*(1+2*cal['q0'])
    checks = dict(coefficient_trend=fit_check(model['record']), gamma_trend=fit_check(surface['record']['trend_check']),
                  cubic_fits={k: fit_check(v['actual_polynomial']) for k, v in report['preparation'].items()},
                  cubic_designs={k: fit_check(v['scaled_design']) for k, v in report['preparation'].items()},
                  projections=projection_checks, gamma_anchors={k: fit_check(v) for k, v in surface['record']['anchors'].items()})
    constants.update(polynomial=dict(coefficient=model['coefficient'], shape=list(model['coefficient'].shape),
        monomial_order=tw.TERMS3, scale_mm=model['scale'], origin_mm=states[0]['origin']), camera=report['camera'],
        H0_mm=cal['H0'], q0=cal['q0'], mu_cal=dict(A=cal['A'][cal['anchors']], mu=cal['mu'][cal['anchors']]),
        gamma_trend=surface['trend'], table_grid=dict(start=0., stop=4., step=.01, count=401),
        loop=dict(start=0., tolerance_D=ac.S2_TOLERANCE, max_passes=ac.S2_MAX_PASSES, endpoint='clip'),
        fit_checks=checks, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    lines.extend([f'configuration: anchors 0,1,2,3,4 D; training gaze; all 21 x 10201 states scored; Rc_um={Rc:.17g}',
                  'F: full coefficient array 3x10x2; start 0 D; 20 passes; tolerance 1e-6 D; table 0:.01:4 D; clip',
                  f'constants digest: scale={model["scale"]}; origin={states[0]["origin"]}; H0={cal["H0"]:.17g}; q0={cal["q0"]:.17g}',
                  'camera/mu_cal/gamma_trend digest: '+json.dumps(plain({k: constants[k] for k in ('camera', 'mu_cal', 'gamma_trend')}), separators=(',', ':'), allow_nan=False),
                  'F fit rank/condition checks: '+json.dumps(plain(checks), separators=(',', ':'), allow_nan=False)])
    laws = {'F': None}
    anchors = [states[i] for i in args.anchors]
    for order in (2, 3):
        name = f'LF-L{order}'
        try:
            laws[name] = lf_calibrate(np.array([s['A'] for s in anchors]),
                [tw.tangent_coordinates(s['w'][s['groups']['train']])['primary'] for s in anchors],
                [s['m'][s['groups']['train']] for s in anchors], np.array([s['q'] for s in anchors]),
                surface['record']['gamma_anchors'], order)
            constants[name] = laws[name]
            lines.append(name+' constants/fit checks: '+json.dumps(plain(laws[name]), separators=(',', ':'), allow_nan=False))
        except ac.NumericalFailure as exc:
            failure = {'rank': None, 'condition': None, **getattr(exc, 'numbers', {})}
            constants[name] = dict(status='unavailable', error=str(exc), fit_failure=failure)
            lines.append(name+' unavailable: '+json.dumps(plain(constants[name]), separators=(',', ':'), allow_nan=False))
    rows, all_readout = {}, []
    for name in ('F', 'LF-L2', 'LF-L3'):
        rows[name], collected = {}, []
        for index, state in states.items():
            log(f'{name} A{index:02d}')
            n = len(state['m'])
            loop = (run_loop(state, cal, model, surface, laws[name]) if name in laws else
                    dict(estimate=np.full(n, np.nan), reason=np.full(n, 5, np.int8), gaze=np.full((n, 2), np.nan),
                         passes=np.zeros(n), clipped=np.zeros(n, bool), size_residual=np.full(n, np.nan)))
            view = dict(w=state['w'], A=np.full(n, state['A']), state_ids=state['state_ids'], shard=np.full(n, f'A{index:02d}'))
            mean = ring = None
            if name != 'F':
                mean = 1000*np.linalg.norm(lf_mean(laws[name], state['w'], view['A'])-state['m'], axis=1) if name in laws else np.full(n, np.nan)
                ring = abs(size_value(cal, tw.tangent_coordinates(state['w'])['primary'], view['A'], laws[name])-state['mu_raw'])*Rc if name in laws else np.full(n, np.nan)
            record = score(view, loop, Rc, mean, ring)
            lines.append(f'{name} A{index:02d} '+compact(record, shard=True))
            collected.append((view, {k: loop[k] for k in ('estimate', 'reason', 'gaze', 'passes', 'clipped', 'size_residual')}, mean, ring))
            if name == 'F':
                all_readout.extend(state['radial_residual_um'])
        for pool, indices in POOLS.items():
            subset = [collected[i] for i in indices]
            view = {k: np.concatenate([entry[0][k] for entry in subset]) for k in subset[0][0]}
            loop = {k: np.concatenate([entry[1][k] for entry in subset]) for k in subset[0][1]}
            mean, ring = ((None, None) if name == 'F' else
                          tuple(np.concatenate([entry[j] for entry in subset]) for j in (2, 3)))
            rows[name][pool] = score(view, loop, Rc, mean, ring)
            lines.append(f'{name} pool={pool} '+compact(rows[name][pool]))
    cross = {pool: rows['F'][pool]['gaze']['max']-value for pool, value in AUDITED.items()}
    c0 = all(abs(delta) <= 1e-6 for delta in cross.values())
    c1 = rows['F']['all21']['coverage'] == 1. and rows['F']['all21']['gaze']['max'] < 1.
    lines.append('cross-check F minus audited S2 maxima, arcmin: '+str(cross))
    lines.extend([f'C0 {"PASS" if c0 else "FAIL"}: exit=0; selftest passed; max delta={max(abs(v) for v in cross.values())}',
                  f'C1 {"PASS" if c1 else "FAIL"}: coverage={rows["F"]["all21"]["coverage"]}; gaze_max={rows["F"]["all21"]["gaze"]["max"]}'])
    passes = {}
    for name in ('LF-L2', 'LF-L3'):
        overall = rows[name]['all21']
        flags = [overall['coverage'] == 1. and overall['gaze']['max'] < 1., overall['mean']['max'] < 1., overall['ring']['max'] < 1.]
        passes[name] = all(flags)
        for criterion, passed, metric in zip(('C2', 'C3', 'C4'), flags, ('gaze', 'mean', 'ring')):
            lines.append(f'{criterion} {name} {"PASS" if passed else "FAIL"}: max={overall[metric]["max"]}; coverage={overall["coverage"]}')
    selected = next((name[3:] for name in ('LF-L2', 'LF-L3') if passes[name]), 'none')
    residuals = np.asarray(all_readout)
    shard, offset = divmod(int(np.argmax(residuals)), len(states[0]['m']))
    readout_location = f'A{shard:02d}:{states[shard]["state_ids"][offset]}:{states[shard]["w"][offset].tolist()}'
    lines.extend([f'readout only: two-radius 16-coordinate residual RMS={np.sqrt(np.mean(residuals**2)):.9g} max={np.max(residuals):.9g} um@{readout_location} (third projection output)',
                  'limits: noise-free development traces; A 0..4 D clipped; field +/-12.5 deg; ring-size narrowing pending user confirmation; full-ring points/beta/shape accuracy unresolved',
                  'selected_form = '+selected,
                  'NOT CLOSED: INVESTIGATE' if not (c0 and c1) else 'CLOSE: ALGORITHM ONLY' if selected == 'none' else 'CLOSE: FULL (ring goal narrowed to ring size)'])
    return 0


def compact(record, shard=False):
    values = [f'coverage={record["valid"]}/{record["states"]}', f'passes_med/max={record["passes_median_max"]}',
              f'passes_max_location={record["passes_max_location"]}', f'failed_by_reason={record["reasons"]}', f'clipped={record["clipped"]}']
    for metric in ('gaze', 'A', 'mean', 'ring'):
        if metric in record:
            item = record[metric]
            location = item['location']
            values.append(f'{metric}_{"max" if shard else "rms/max"}='+(str(item['max']) if shard else f'{item["rms"]}/{item["max"]}')
                          +f'@{location["shard"]}:{location["state_id"]}:{location["gaze_deg"]}')
    values.append('size_eq_max_um/location(unclipped/clipped)='+str([record['size_equation_unclipped'], record['size_equation_clipped']]))
    return ' '.join(values)


if __name__ == '__main__':
    raise SystemExit(main())
