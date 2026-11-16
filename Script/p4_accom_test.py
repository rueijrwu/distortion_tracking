r"""Accommodation ladder AE0--AE2 and development stage S1 (Section 13.10).

noise-free design traces, ground-truth gaze, unaudited

PowerShell, from the project root, AFTER the independent static audit:
& 'C:\Users\rueijrwu\.pyenv-win-venv\envs\venv\Scripts\python.exe' Script\p4_accom_test.py --selftest
& 'C:\Users\rueijrwu\.pyenv-win-venv\envs\venv\Scripts\python.exe' Script\p4_accom_test.py

The default production run requires all 21 published full5 shards. No prior
output, cache, legacy input or substitute trace is read. Its own synthetic
selftest runs in process before production loading. Every write is confined
to runs/p4_accom; imported helpers are called unmodified, never their mains.
Stage S1 adds AE2b, oracle-gaze AE3e, one-pass AE4e and descriptive SEP.
Stage S2 adds an optional fixed-point development loop. No nonlinear fit,
reversed lens map or centre estimator is implemented. All scientific errors
are provisional readouts, never gates.
"""
import sys
sys.dont_write_bytecode = True

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat
import time
from types import SimpleNamespace

for _thread_name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                     'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_thread_name] = '1'

import numpy as np
import p4_a0_data as data_json
import p4_a0_gazepoly as gp
import p4_a0_ninept5 as five
import p4_termwise_test as tw

ROOT = Path(__file__).absolute().parents[1]
OUT = ROOT/'runs'/'p4_accom'
LABEL = 'noise-free design traces, ground-truth gaze, unaudited'
PROVISIONAL = 'provisional diagnostics: reported, not gated'
INDICES = tuple(range(21))
TAGS = ('selftest', 'all21')
S1_TAGS = ('selftest_s1', 'all21_s1')
DEVELOPMENT = 'development: held-out A previously inspected; not validation'
LOCAL_SENSITIVITY = ('finite-difference local sensitivity of the one-pass map near the true A; '
    'not a contraction constant over the iteration domain, not evidence of a fixed point or of convergence')
SURFACE_DERIVATIVE = ('differences of the same 0.01 D table: central across neighbouring nodes, '
    'mean of left/right slopes at integer-D knots; forward at 0 D, backward at 4 D')
S1_REASONS = ((0, 'valid'), (1, 'nonfinite'), (2, 'out_of_calibration_range'),
              (3, 'secondary_calibration_unavailable'), (4, 'nonmonotone_in_A'),
              (5, 'estimator_unavailable'))
S2_TAGS = ('selftest_s2', 'all21_s2')
S2_LOOP_LABEL = ('measured on development data; counts of states meeting the stopping rule; '
    'not a proof of convergence, of contraction over the domain, or of a unique fixed point')
S2_AE6_LABEL = 'descriptive only; all 21 shards; not calibration; no estimator reads AE6-lite'
S2_DIOPTRE_LABEL = ('zero-gaze-slope-equivalent dioptre diagnostic; common scale only; '
    'not an estimator A error and not the corner-local inverse error')
S2_MAX_PASSES, S2_TOLERANCE = 20, 1e-6
S2_PROFILE_STEP, S2_EXTENSION_MARGIN = .05, .2
S2_REASONS = (*S1_REASONS, (6, 'not_stopped_within_max_passes'))
S2_ROWS = (('AE5e-joint', 'joint', 0.), ('AE5e-joint-start2', 'joint', 2.),
    ('AE5e-joint-start4', 'joint', 4.), ('AE5e-joint-clip', 'clip', 0.),
    ('AE5e-joint-ext02', 'ext02', 0.), ('AE5e-sep', 'sep', 0.),
    ('AE5e-sep-start2', 'sep', 2.), ('AE5e-sep-start4', 'sep', 4.))
S2_CONTEXT = ('AE4e-joint-P0', 'AE4e-joint-pass2')
S2_HEADLINE_CELLS = ('overall', 'field_lt12', 'four_corner_extremes',
                     'odd_odd', 'odd_odd/field_lt12')
COND = five.joint.COND                 # Existing, unchanged 1e8 gate.
require = five.require
NumericalFailure = tw.NumericalFailure


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--selftest', action='store_true', help='Synthetic only; no trace I/O')
    # D1--D6: scope decisions are explicit even when this increment has one option.
    parser.add_argument('--goal', choices=('accommodation',), default='accommodation')
    parser.add_argument('--centre-calibration', choices=('none',), default='none',
                        help='D2 deferred to AE3; no E1 or D0 used in AE0--AE2')
    parser.add_argument('--gaze-readouts', nargs='+', type=float, default=[1., .1],
                        help='D3 provisional arcmin levels, both reported by default')
    parser.add_argument('--anchors', nargs='+', type=int, default=[0, 5, 10, 15, 20],
                        help='D4 calibration shard indices; all other A held out')
    parser.add_argument('--empirical-bar', choices=('include',), default='include')
    parser.add_argument('--stages', choices=('AE0-AE2', 'S1', 'S2'), default='AE0-AE2')
    parser.add_argument('--size-basis', choices=('E4',), default=argparse.SUPPRESS)
    parser.add_argument('--a-grid-step', type=float, choices=(.01,), default=argparse.SUPPRESS)
    parser.add_argument('--sensitivity-step', type=float, choices=(.2,), default=argparse.SUPPRESS)
    parser.add_argument('--loop-max-passes', type=int, choices=(S2_MAX_PASSES,), default=argparse.SUPPRESS)
    parser.add_argument('--loop-tolerance', type=float, choices=(S2_TOLERANCE,), default=argparse.SUPPRESS)
    parser.add_argument('--profile-step', type=float, choices=(S2_PROFILE_STEP,), default=argparse.SUPPRESS)
    parser.add_argument('--extension-margin', type=float, choices=(S2_EXTENSION_MARGIN,), default=argparse.SUPPRESS)
    # Unspecified modelling choices: fixed before looking at held-out errors.
    parser.add_argument('--a-poly-degree', type=int, choices=(1, 2), default=2,
                        help='Anchor-only trend of full-cubic gaze coefficients')
    parser.add_argument('--a-poly-interpolation', choices=('polynomial',), default='polynomial')
    parser.add_argument('--size-interpolation', choices=('piecewise-linear',), default='piecewise-linear')
    parser.add_argument('--size-inversion', choices=('monotone-linear',), default='monotone-linear')
    parser.add_argument('--range-policy', choices=('fail-state',), default='fail-state',
                        help='Never clamp or extrapolate beyond calibrated A/size range')
    parser.add_argument('--empirical-degree', type=int, choices=(1, 2), default=2,
                        help='1: [1,sigma]; 2: [1,sigma,sigma^2,mx^2,mx*my,my^2]')
    parser.add_argument('--sigma-definition', choices=('pre-camera-trace-covariance',),
                        default='pre-camera-trace-covariance')
    parser.add_argument('--sigma-normalization', choices=('A00-zero-ratio-minus-one',),
                        default='A00-zero-ratio-minus-one')
    parser.add_argument('--mean-origin', choices=('fixed-A00',), default='fixed-A00')
    parser.add_argument('--raw-size-statistic', choices=('two-radius-projection',),
                        default='two-radius-projection')
    parser.add_argument('--proxy-degree', type=int, choices=(1,), default=1,
                        help='Effective isotropic size slope versus rho^2; not a physical m1 fit')
    args = parser.parse_args(argv)
    if args.stages in ('S1', 'S2'):
        if (args.anchors != [0, 5, 10, 15, 20] or args.a_poly_degree != 2 or
                args.empirical_degree != 2 or args.gaze_readouts != [1., .1]):
            parser.error(args.stages+' fixes --anchors 0 5 10 15 20, --a-poly-degree 2, '
                         '--empirical-degree 2 and --gaze-readouts 1 0.1')
        args.size_basis = 'E4'
        args.a_grid_step = .01
        args.sensitivity_step = .2
    if args.stages == 'S2':
        args.loop_max_passes = S2_MAX_PASSES
        args.loop_tolerance = S2_TOLERANCE
        args.profile_step = S2_PROFILE_STEP
        args.extension_margin = S2_EXTENSION_MARGIN
    if (len(set(args.anchors)) != len(args.anchors) or args.anchors != sorted(args.anchors)
            or not set(args.anchors) <= set(INDICES) or not {0, 20} <= set(args.anchors)
            or len(args.anchors) <= args.a_poly_degree or len(args.anchors) == len(INDICES)):
        parser.error('--anchors must be sorted, distinct, include 0 and 20, leave held-out A, '
                     'and outnumber --a-poly-degree')
    if (not args.gaze_readouts or not all(np.isfinite(v) and v > 0 for v in args.gaze_readouts)
            or len(set(args.gaze_readouts)) != len(args.gaze_readouts)):
        parser.error('--gaze-readouts must be distinct finite positive arcmin levels')
    return args


def artifacts(tag):
    require(tag in (*TAGS, *S1_TAGS), 'Unknown output tag')
    return (f'ladder_{tag}.txt', f'report_{tag}.txt', f'results_{tag}.json',
            f'maps_{tag}.npz', f'progress_{tag}.log')


def guard_outputs(tag):
    """Same fixed-folder alias/reparse/hard-link protections as the reused scripts."""
    require(ROOT.resolve() == ROOT, 'Redirected project root')
    targets = [OUT/name for name in artifacts(tag)]
    for path in [OUT, *OUT.parents, *targets]:
        require(path.resolve() == path.absolute(), 'Redirected output: '+str(path))
        if path.exists() or path.is_symlink():
            info = path.lstat()
            require(not path.is_symlink() and not
                    (getattr(info, 'st_file_attributes', 0) & 0x400), 'Output reparse point: '+str(path))
            if path in targets:
                require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
                        'Output must be a regular file with one link: '+str(path))
            else:
                require(path.is_dir(), 'Output parent is not a directory: '+str(path))
    sources = [five.MANIFEST, *(gp.accommodation_input(i) for i in INDICES),
               ROOT/'CURRENT_PLAN.md', ROOT/'P4_MODEL.md', ROOT/'P4_SPEC.md',
               *list((ROOT/'Script').glob('*.py')),
               *(OUT/name for other in (*TAGS, *S1_TAGS) if other != tag for name in artifacts(other))]
    for i, target in enumerate(targets):
        require(target.parent == OUT, 'Output outside runs/p4_accom')
        for source in [*sources, *targets[i+1:]]:
            require(target.resolve() != source.resolve() and not
                    (target.exists() and source.exists() and target.samefile(source)),
                    'Output aliases an input/source/target')


class Progress:
    def __init__(self, tag):
        self.tag = tag
        self.started = self.last = time.perf_counter()
        guard_outputs(tag)
        OUT.mkdir(parents=True, exist_ok=True)
        guard_outputs(tag)
        (OUT/artifacts(tag)[-1]).write_text(LABEL+'\n', encoding='utf-8')

    def __call__(self, message, heartbeat=False):
        now = time.perf_counter()
        if heartbeat and now-self.last < 30:
            return
        self.last = now
        line = datetime.now(timezone.utc).isoformat(timespec='seconds')
        line += f' elapsed_s={now-self.started:.3f} '+message
        print(line, flush=True)
        guard_outputs(self.tag)
        with (OUT/artifacts(self.tag)[-1]).open('a', encoding='utf-8') as stream:
            stream.write(line+'\n')
            stream.flush()


def new_report(args, tag, synthetic=False):
    heldout = [i for i in INDICES if i not in args.anchors]
    return {LABEL: True, **dict(diagnostic_label=PROVISIONAL, tag=tag, synthetic=synthetic,
        status='preparing', exit_code=None, indices=list(INDICES), errors=[], results={}, ladder=[],
        configuration=vars(args).copy(),
        decisions=dict(D1='Estimate A; gaze only from the M8 polynomial; centre is not a target',
            D2='Deferred/unused: no E1 or D0 in AE0--AE2; loader validates its full published schema',
            D3=dict(gaze_levels_arcmin=args.gaze_readouts, scientific_gates=False),
            D4=dict(anchor_indices=args.anchors, heldout_indices=heldout,
                    calibration='Anchor training gaze only; zero-gaze references are calibration'),
            D5='Include fixed empirical bar Q; no held-out model selection',
            D6='AE0--AE2 only; independent of the OPEN Section 12 nonlinear solver decision'),
        definitions=dict(M8='Mean of the eight outer original-3x3 labels; sensor xsi/ysi in mm',
            w_deg='(BDE,-ADE); primary tangent convention from p4_termwise_test',
            gaze_origin='One measured A00 zero-gaze M8 origin, fixed at runtime and in every fit',
            polynomial='Full inverse cubic, 10 terms per axis, same training/rcond as existing fits',
            oracle='Own-A cubic fit on training gaze only; oracle/scoring reference, not runtime calibration',
            interpolation='Anchor-only polynomial trend of coefficients in one A00-normalised basis',
            sensitivity='Analytic dP_A(m)/dA of that anchor-only coefficient trend, at fixed measured m',
            raw_size='Camera-undistorted, mean-free ring; two-radius linear projection, no gaze correction',
            mu='Edge-midpoint radial scale / A00 edge-midpoint radial scale',
            beta='(corner radial scale / A00 corner radial scale) / mu - 1',
            calibration='mu_cal=H_A*(1+q_A)/(H_0*(1+q_0)); beta_cal from corner/midpoint ratio',
            Sigma='Scalar trace of pre-camera ring covariance: mean_i |z_i-zbar|^2 in mm^2',
            sigma='Sigma/Sigma(A00,zero)-1; normalisation is fixed before Q training',
            Q='Even under simultaneous m -> -m; declared fixed basis, no lens map or iteration',
            splits='2600 nonzero even/even training; three parity holdouts; zero is calibration only',
            A_split='Anchor and held-out A scored separately, crossed with gaze splits and bins',
            numerical_gates=dict(full_rank=True, column_scaled_condition_max=COND, rcond=gp.RCOND,
                nonlinear_gradient_gate='unchanged 1e-10; no nonlinear fit called',
                accept_fit='unchanged; not called'),
            failures='No failed state dropped; any failure makes full-denominator group RMS/p95/max Infinity',
            range_policy='Fail and count state outside calibrated range; no clipping/extrapolation',
            beta_inverse='Secondary diagnostic only; unavailable if anchor beta is not strictly monotone',
            proxies='Raw-size effective isotropic slope and |slope|*rho^2 are UNVERIFIED proxies; '
                    'neither physical m1 nor the measured AE4/AE5 contraction'),
        inputs=[], preparation={}, calibration={}, pooled={}, caveats=[
            'Noise-free design traces and ground-truth gaze only; no real-eye/noise claim.',
            'All 21 own-A cubics and zero-gaze curves are descriptive oracle/readout data. '
            'Runtime trends, size inversions and Q use only the declared anchors.',
            'Pre-camera raw size is a fixed-camera baseline; it has no gaze-dependent lens correction.',
            'Sigma is scalar trace covariance, not the full covariance matrix.',
            'A00 camera centre and A00 zero-gaze M8 origin remain fixed; per-A origins are diagnostics.',
            'AE3--AE6, reversed maps, E1/D0, nonlinear fits and the gaze/A loop are outside this script.'])}


def linear_fit(matrix, target, context):
    """Column-normalised linear LS, with the unchanged rank/condition gates."""
    matrix, target = np.asarray(matrix, float), np.asarray(target, float)
    if (matrix.ndim != 2 or len(matrix) != len(target) or not np.isfinite(matrix).all()
            or not np.isfinite(target).all()):
        raise NumericalFailure(context+': invalid/nonfinite linear input')
    scales = np.linalg.norm(matrix, axis=0)
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise NumericalFailure(context+': zero/nonfinite design column', column_scales=scales)
    normal = matrix/scales
    coefficient, _, rank, singular = np.linalg.lstsq(normal, target, rcond=gp.RCOND)
    condition = float(singular[0]/singular[-1]) if rank == matrix.shape[1] else float('inf')
    if rank != matrix.shape[1] or condition > COND or not np.isfinite(coefficient).all():
        raise NumericalFailure(context+': rank/condition/finite failure', rank=int(rank),
                               condition=condition, singular_values=singular, column_scales=scales)
    divisor = scales.reshape((len(scales),)+(1,)*(coefficient.ndim-1))
    coefficient = coefficient/divisor
    return coefficient, dict(rank=int(rank), columns=matrix.shape[1], condition=condition,
        singular_values=singular, column_scales=scales, rcond=gp.RCOND,
        source='column-normalised linear least squares', status='complete')


def polynomial_fit(m, w, train, scale, context):
    fit, prediction = tw.checked_cubic(m[train], w[train], scale, m, context)
    # checked_cubic's unmodified gate is retained; Section 13 also requires
    # the scaled-condition gate used by linear shift fits.
    matrix = gp.design(m[train]/scale, tw.TERMS3)
    _, checked = linear_fit(matrix, w[train], context+' design')
    return fit, prediction, checked


def scalar_metric(values, mask, unit='D'):
    """Scalar full-denominator errors; no finite-only/converged-only scores."""
    values = np.asarray(values)[mask]
    n = len(values)
    failed = int((~np.isfinite(values)).sum())
    out = dict(states=n, nonfinite_states=failed)
    if not n:
        metrics = (None, None, None, None)
    elif failed:
        metrics = (float('inf'),)*4
    else:
        metrics = (float(np.sqrt(np.mean(values*values))), float(np.percentile(abs(values), 95)),
                   float(np.max(abs(values))), float(np.mean(values)))
    out.update(zip(('rms_'+unit, 'p95_abs_'+unit, 'max_abs_'+unit, 'bias_'+unit), metrics))
    return out


def scalar_summaries(values, groups, unit='D'):
    extremes = [name for name in groups if name.startswith('extreme_')]
    return {name: dict(metrics=scalar_metric(values, groups[name], unit),
        by_magnitude={b: scalar_metric(values, groups[name] & groups[b], unit) for b in gp.BIN_NAMES},
        corner_extremes={c: scalar_metric(values, groups[name] & groups[c], unit) for c in extremes},
        four_corner_extremes=scalar_metric(values, groups[name] & groups['four_corner_extremes'], unit))
        for name in gp.SPLITS}


def monotonic_record(a, values):
    a, values = np.asarray(a, float), np.asarray(values, float)
    slopes = np.diff(values)/np.diff(a)
    finite = bool(np.isfinite(values).all() and np.isfinite(slopes).all())
    increasing = finite and bool(np.all(slopes > 0))
    decreasing = finite and bool(np.all(slopes < 0))
    return dict(strictly_monotone=bool(increasing or decreasing), increasing=increasing,
                decreasing=decreasing, adjacent_slopes_per_D=slopes,
                minimum_abs_slope_per_D=float(np.min(abs(slopes))) if finite else None,
                zero_slope_intervals=int(np.sum(slopes == 0)),
                sign_change_intervals=int(np.sum(slopes[:-1]*slopes[1:] < 0)))


def monotone_inverse(a, values, targets, required=True):
    """No extrapolation, endpoint clipping, tolerance band or finite-only metric."""
    a, values, targets = np.asarray(a, float), np.asarray(values, float), np.asarray(targets, float)
    record = monotonic_record(a, values)
    result = np.full(targets.shape, np.nan)
    reason = np.full(targets.shape, 3, dtype=np.int8)
    if not record['strictly_monotone']:
        if required:
            raise NumericalFailure('Primary size calibration is not strictly monotone', **record)
        return result, reason, dict(record, status='diagnostic_unavailable',
                                    reason='Nonmonotone/flat/nonfinite secondary calibration')
    if record['decreasing']:
        values, a = values[::-1], a[::-1]
    finite = np.isfinite(targets)
    valid = finite & (targets >= values[0]) & (targets <= values[-1])
    reason[~finite] = 1
    reason[finite & ~valid] = 2
    reason[valid] = 0
    result[valid] = np.interp(targets[valid], values, a)
    return result, reason, dict(record, status='complete', calibrated_size_range=[values[0], values[-1]])


def reason_record(reason, state=None, inputs=None, method=None):
    answer = dict(total_states=int(reason.size), failed_states=int(np.sum(reason != 0)),
        by_reason={name: int(np.sum(reason == code)) for code, name in
                   ((0, 'valid'), (1, 'nonfinite'), (2, 'out_of_calibration_range'),
                    (3, 'secondary_calibration_unavailable'))})
    if state is not None:
        selected = np.flatnonzero(reason != 0)[:50]
        answer.update(first_failed_states=[dict(state_id=int(state['state_ids'][i]),
            w_deg=state['w'][i], reason_code=int(reason[i]),
            runtime_input=None if inputs is None else float(inputs[i]),
            iterations=None, residual=None,
            iteration_residual_note='Not applicable: '+str(method)) for i in selected],
            failure_examples_selected=len(selected),
            failure_examples_omitted=int(np.sum(reason != 0))-len(selected))
    return answer


def prepare_all(report, log, loader):
    """Every shard must pass every loader/data gate before any AE rung exists."""
    states = {}
    for index in INDICES:
        key = f'A{index:02d}'
        entry = dict(status='loading', calibration={}, timings={})
        report['inputs'].append(dict(index=index, shard=gp.accommodation_input(index).name, loader=entry))
        metadata = {}
        try:
            log(f'validating {key}; all 21 inputs precede AE0--AE2')
            observed, c, H, q, inherited = loader(SimpleNamespace(quick=False, shard_index=index,
                input_path=gp.accommodation_input(index)), entry, metadata, log)
            state = gp.accommodation_states(observed, inherited, metadata)
            dioptres = entry['input']['manifest_entry']['accommodation_d']
            require(isinstance(dioptres, (int, float)) and np.isfinite(dioptres),
                    'Nonfinite/missing manifest accommodation_d')
            require((index != 0 or dioptres == 0) and (index != 20 or dioptres == 4),
                    'Published accommodation endpoints must be A00=0 D and A20=4 D')
            if states:
                base = states.get(0, states[min(states)])
                require(dioptres > states[max(states)]['A'],
                        'Manifest A must increase strictly with shard index')
                require(np.array_equal(state['state_ids'], base['state_ids']) and
                        np.array_equal(state['w'], base['w']) and
                        state['groups'].keys() == base['groups'].keys() and all(
                        np.array_equal(state['groups'][name], base['groups'][name])
                        for name in state['groups']), 'Cross-shard gaze/state/group mismatch')
            require(np.isfinite(c).all() and np.isfinite(H) and H > 0 and np.isfinite(q),
                    'Invalid loader calibration')
            state.update(A=float(dioptres), H=float(H), q=float(q), c=np.asarray(c).copy(),
                         ring_sensor=observed[:, state['node_ids']].copy())
            del observed
            states[index] = state
            entry.update(status='validated', accommodation_d=float(dioptres),
                         reported_calibration=dict(c_mm=c, H_mm=H, q=q))
        except Exception as exc:
            entry['status'] = 'failed'
            report['errors'].append(dict(shard=key, exception=type(exc).__name__, message=str(exc)))
            log(f'{key} invalid: {type(exc).__name__}: {exc}')
    if len(states) != len(INDICES):
        report.update(status='stopped_missing_or_invalid_data', exit_code=2, results={}, ladder=[])
        return None
    report['split_counts'] = {name: int(mask.sum()) for name, mask in states[0]['groups'].items()}
    return states


def shared_preparation(states, report, stage_s1=False):
    """Fixed camera inverse and own-A cubics; failures stop before all rungs."""
    base = states[0]
    report['shared_A00_origin_mm'] = base['origin']
    scale = np.max(abs(base['centered'][base['groups']['train']]), axis=0)
    report['shared_A00_scale_mm'] = scale
    report['camera'] = dict(**tw.camera_constants(), c_cam_mm=base['c'])
    for index, state in states.items():
        key, item = f'A{index:02d}', 'camera_inverse'
        record = report['preparation'][key] = dict(status='preparing')
        try:
            state['m'] = state['sensor']-base['origin']
            state['scale'] = scale
            z, inverse = five.joint.camera_inverse(state['ring_sensor']-base['c'])
            if (not np.isfinite(z).all() or not np.all(inverse['converged']) or
                    not np.all(inverse['min_tangential'] > 0) or not np.all(inverse['min_radial'] > 0)):
                raise NumericalFailure('Camera inverse invalid/nonfinite')
            closure_um = 1000*np.linalg.norm(
                five.joint.camera(z)-(state['ring_sensor']-base['c']), axis=2)
            # Exactly the existing five.inverse closure gate, with all scalar
            # inverse diagnostics retained instead of discarding their record.
            if not np.all(closure_um < 1e-9):
                raise NumericalFailure('Camera closure >=1e-9 um',
                    maximum_closure_um=float(np.max(closure_um)),
                    failed_states=int(np.sum(np.any(closure_um >= 1e-9, axis=1))))
            state['Zshape'] = z-z.mean(axis=1, keepdims=True)
            if stage_s1:
                state['Zbar'] = z.mean(axis=1)
                state['Zring'] = state['Zshape']+state['Zbar'][:, None, :]
            state['Sigma'] = np.mean(np.sum(state['Zshape']**2, axis=2), axis=1)
            record['camera_inverse'] = dict(iterations=inverse['iterations'],
                max_abs_residual_mm=float(np.max(abs(inverse['residual']))),
                min_radial=float(np.min(inverse['min_radial'])),
                min_tangential=float(np.min(inverse['min_tangential'])),
                closure_max_um=float(np.max(closure_um)))
            item = 'own_A_full_cubic'
            state['fit'], state['oracle'], checked = polynomial_fit(state['m'], state['w'],
                state['groups']['train'], scale, key+' M8 cubic')
            record.update(status='complete', actual_polynomial=tw.fit_record(state['fit'], tw.TERMS3),
                scaled_design=checked, origin_delta_vs_A00_mm=state['origin']-base['origin'],
                loader_camera_centre_delta_vs_A00_mm=state['c']-base['c'])
        except Exception as exc:
            record.update(status='failed', item=item)
            raise NumericalFailure(key+' '+item+': '+str(exc),
                                   **getattr(exc, 'numbers', {})) from exc


def coefficient_model(states, args):
    anchors = args.anchors
    a = np.array([states[i]['A'] for i in anchors])
    # The A00 shared basis is fixed; only anchor own-A cubics feed the trend.
    coefficient, checked = linear_fit(np.column_stack([a**j for j in range(args.a_poly_degree+1)]),
        np.stack([states[i]['fit']['coef'] for i in anchors]).reshape(len(anchors), -1),
        'Anchor-only polynomial coefficient trend')
    return dict(coefficient=coefficient.reshape(args.a_poly_degree+1, len(tw.TERMS3), 2),
        scale=states[0]['scale'], bounds=(a[0], a[-1]),
        record=dict(**checked, degree=args.a_poly_degree, anchor_indices=anchors,
                    anchor_dioptres=a, parameter_units='normalised gaze deg / D**power'))


def evaluate_gaze(model, m, accommodation, derivative=False):
    accommodation = np.broadcast_to(np.asarray(accommodation, float), (len(m),))
    valid = (np.isfinite(accommodation) & (accommodation >= model['bounds'][0]) &
             (accommodation <= model['bounds'][1]) & np.isfinite(m).all(axis=1))
    prediction = np.full((len(m), 2), np.nan)
    if np.any(valid):
        a = accommodation[valid]
        degree = len(model['coefficient'])-1
        weights = np.column_stack([j*a**(j-1) if j else np.zeros_like(a)
            for j in range(degree+1)]) if derivative else np.column_stack(
            [a**j for j in range(degree+1)])
        coefficients = np.einsum('np,ptc->ntc', weights, model['coefficient'])
        prediction[valid] = np.einsum('nt,ntc->nc',
            gp.design(m[valid]/model['scale'], tw.TERMS3), coefficients)
    return prediction


def two_radius_projection(shape, H0, q0):
    """Two linear radial coefficients at the fixed eight labelled nodes."""
    nodes = -five.LABELS[gp.estimator_nodes()['E2'][1]]
    radius2 = np.sum(nodes*nodes, axis=1)
    baseline = H0*(1+q0*radius2)
    if not np.isfinite(baseline).all() or np.any(baseline <= 0):
        raise NumericalFailure('Nonpositive/nonfinite A00 two-radius baseline')
    matrix = np.stack((nodes, nodes*(radius2-1)[:, None]), axis=-1).reshape(16, 2)
    rhs = (shape/baseline[None, :, None]).reshape(len(shape), 16).T
    coefficient, checked = linear_fit(matrix, rhs, 'Per-state two-radius radial projection')
    mu, mubeta = coefficient
    beta = np.divide(mubeta, mu, out=np.full_like(mu, np.nan), where=mu != 0)
    residual = (rhs.T-(matrix@coefficient).T).reshape(-1, 8, 2)*baseline[None, :, None]
    return mu, beta, np.sqrt(np.mean(np.sum(residual*residual, axis=2), axis=1))*1000, checked


def size_calibration(states, args):
    H0, q0 = states[0]['H'], states[0]['q']
    a = np.array([states[i]['A'] for i in INDICES])
    edge = np.array([states[i]['H']*(1+states[i]['q']) for i in INDICES])
    corner = np.array([states[i]['H']*(1+2*states[i]['q']) for i in INDICES])
    if not np.isfinite(edge).all() or not np.isfinite(corner).all() or np.any(edge <= 0) or np.any(corner <= 0):
        raise NumericalFailure('Invalid two-point zero-gaze radial calibration')
    mu = edge/edge[0]
    beta = (corner/corner[0])/mu-1
    anchors = np.asarray(args.anchors)
    # Fail a bad PRIMARY calibration. The weaker beta inverse is diagnostic.
    _, _, mu_inverse = monotone_inverse(a[anchors], mu[anchors], mu[anchors])
    _, _, beta_inverse = monotone_inverse(a[anchors], beta[anchors], beta[anchors], required=False)
    zero = int(np.flatnonzero(states[0]['groups']['zero'])[0])
    covariance0 = float(states[0]['Sigma'][zero])
    if not np.isfinite(covariance0) or covariance0 <= 0:
        raise NumericalFailure('Nonpositive/nonfinite A00 zero-gaze Sigma')
    return dict(A=a, mu=mu, beta=beta, anchors=anchors, H0=H0, q0=q0, Sigma0=covariance0,
        record=dict(all21=dict(accommodation_d=a, mu=mu, beta=beta,
            edge_factor_mm=edge, corner_component_factor_mm=corner,
            mu_monotonicity=monotonic_record(a, mu), beta_monotonicity=monotonic_record(a, beta)),
            runtime=dict(anchor_indices=anchors, accommodation_d=a[anchors], mu=mu[anchors],
                beta=beta[anchors], mu_inverse=mu_inverse, beta_inverse=beta_inverse,
                interpolation=args.size_interpolation, inversion=args.size_inversion),
            Sigma_A00_zero_mm2=covariance0,
            H_A00_mm=H0, q_A00=q0))


def empirical_design(m, sigma, scale, degree):
    x, y = (m/scale).T
    if degree == 1:
        return np.column_stack((np.ones(len(m)), sigma)), ['1', 'sigma']
    return (np.column_stack((np.ones(len(m)), sigma, sigma*sigma, x*x, x*y, y*y)),
            ['1', 'sigma', 'sigma^2', 'mx^2', 'mx*my', 'my^2'])


def fit_empirical(states, cal, args):
    matrices, targets = [], []
    for index in args.anchors:
        state = states[index]
        train = state['groups']['train']
        sigma = state['Sigma']/cal['Sigma0']-1
        matrix, terms = empirical_design(state['m'][train], sigma[train], state['scale'], args.empirical_degree)
        matrices.append(matrix)
        targets.append(np.full(int(train.sum()), state['A']))
    coefficient, checked = linear_fit(np.concatenate(matrices), np.concatenate(targets),
                                      'Anchor training-only empirical A polynomial')
    return dict(coefficient=coefficient, record=dict(**checked, terms=terms,
        coefficients_D=coefficient, empirical_degree=args.empirical_degree,
        anchor_indices=args.anchors, training_states_per_anchor=int(states[0]['groups']['train'].sum()),
        Sigma_definition=args.sigma_definition, sigma_normalization=args.sigma_normalization,
        simultaneous_mean_oddness='Q(-m,sigma)=Q(m,sigma) exactly by construction',
        selection='Declared CLI basis; no hyperparameter/degree selection on held-out A or gaze'))


def estimator_scores(predicted_A, reason, state, gaze_model, runtime_input=None, method=None):
    valid_A = np.where(reason == 0, predicted_A, np.nan)
    gaze = evaluate_gaze(gaze_model, state['m'], valid_A)
    return dict(failures=reason_record(reason, state, runtime_input, method),
        A_error_D=scalar_summaries(valid_A-state['A'], state['groups']),
        gaze_vs_truth_arcmin=tw.gaze_summaries(gaze-state['w'], state['groups']),
        gaze_vs_oracle_A_arcmin=tw.gaze_summaries(gaze-state['oracle'], state['groups'])), gaze


def analyse(states, report, args, log, maps):
    model = coefficient_model(states, args)
    cal = size_calibration(states, args)
    empirical = fit_empirical(states, cal, args)
    report['calibration'] = dict(gaze_coefficient_trend=model['record'], raw_size=cal['record'],
                                 empirical_A=empirical['record'])
    report['calibration']['gaze_coefficient_trend']['coefficients'] = model['coefficient']
    primary_failure_total = 0
    beta_failure_total = 0
    pooled = {name: [] for name in ('anchor_A', 'heldout_A')}
    p0 = states[0]['fit']
    for index, state in states.items():
        key, groups, a = f'A{index:02d}', state['groups'], state['A']
        log(f'{key}: AE0 bars, AE1 raw size, AE2 fixed empirical bar')
        interpolated = evaluate_gaze(model, state['m'], a)
        derivative = evaluate_gaze(model, state['m'], a, derivative=True)*60
        sensitivity = np.linalg.norm(derivative, axis=1)
        pzero = gp.predict(p0, state['m'], tw.TERMS3)
        tolerance_maps = {format(level, '.17g'): np.divide(level, sensitivity,
            out=np.full_like(sensitivity, np.inf), where=sensitivity != 0) for level in args.gaze_readouts}
        # Infinity here means unconstrained A at an exactly zero derivative;
        # it is not a failed sensitivity diagnostic.
        sensitivity_record = dict(vector_arcmin_per_D=scalar_summaries(sensitivity, groups, 'arcmin_per_D'),
            zero_sensitivity_states=int(np.sum(sensitivity == 0)),
            required_A_accuracy_D={level: dict(gaze_level_arcmin=float(level),
                conservative_all_state_bound_D=(float('inf') if np.max(sensitivity) == 0 else
                                                float(level)/float(np.max(sensitivity))),
                definition='level / |dP/dA|; maps Infinity for exactly zero sensitivity')
                for level in tolerance_maps})
        ae0 = dict(oracle_A_gaze=tw.gaze_summaries(state['oracle']-state['w'], groups),
            P0_gaze=tw.gaze_summaries(pzero-state['w'], groups),
            interpolated_A_gaze=tw.gaze_summaries(interpolated-state['w'], groups),
            interpolation_vs_oracle_A=tw.gaze_summaries(interpolated-state['oracle'], groups),
            sensitivity=sensitivity_record)
        mu, beta, residual_um, projection = two_radius_projection(state['Zshape'], cal['H0'], cal['q0'])
        raw_A, raw_reason, _ = monotone_inverse(cal['A'][cal['anchors']], cal['mu'][cal['anchors']], mu)
        beta_A, beta_reason, _ = monotone_inverse(cal['A'][cal['anchors']],
            cal['beta'][cal['anchors']], beta, required=False)
        raw_scores, raw_gaze = estimator_scores(raw_A, raw_reason, state, model, mu,
            'direct piecewise-linear monotone inverse; failed states have no valid in-range root')
        tau = tw.tangent_coordinates(state['w'])['primary']
        rho2 = np.sum(tau*tau, axis=1)
        effective, effective_check = linear_fit(rho2[groups['train'], None],
            mu[groups['train']]/cal['mu'][index]-1, key+' effective raw-size rho2 slope')
        effective_slope = float(effective[0])
        zero = int(np.flatnonzero(groups['zero'])[0])
        ae1 = dict(raw_size_estimator=raw_scores, two_radius_projection=projection,
            raw_mu_minus_own_zero_gaze_mu=scalar_summaries(mu-cal['mu'][index], groups, 'relative_scale'),
            radial_projection_residual_um=scalar_summaries(residual_um, groups, 'um'),
            beta_A_diagnostic=dict(failures=reason_record(beta_reason, state, beta,
                'direct secondary monotone inverse or unavailable nonmonotone calibration'),
                A_error_D=scalar_summaries(beta_A-a, groups),
                A_mu_minus_A_beta_D=scalar_summaries(raw_A-beta_A, groups)),
            zero_gaze_calibration_defects=dict(mu=float(mu[zero]-cal['mu'][index]),
                                               beta=float(beta[zero]-cal['beta'][index])),
            effective_isotropic_size_proxy=dict(**effective_check, slope=effective_slope,
                maximum_abs_slope_times_rho2=float(np.max(abs(effective_slope)*rho2)),
                label='UNVERIFIED effective isotropic proxy; m1/n2/projective effects confounded; '
                      'not a measured contraction derivative'))
        sigma = state['Sigma']/cal['Sigma0']-1
        matrix, _ = empirical_design(state['m'], sigma, state['scale'], args.empirical_degree)
        predicted_A = matrix@empirical['coefficient']
        reason = np.zeros(len(predicted_A), dtype=np.int8)
        reason[~np.isfinite(predicted_A)] = 1
        reason[np.isfinite(predicted_A) & ((predicted_A < model['bounds'][0]) |
                                         (predicted_A > model['bounds'][1]))] = 2
        empirical_scores, empirical_gaze = estimator_scores(predicted_A, reason, state, model,
            predicted_A, 'direct empirical polynomial evaluation; no iterative solver')
        ae2 = dict(**empirical_scores,
            unbounded_Q_A_error_D_diagnostic=scalar_summaries(predicted_A-a, groups),
            unbounded_note='Finite Q outputs outside calibrated A are visible only here; '
                           'they fail the primary estimator and never extrapolate P_A')
        report['results'][key] = dict(accommodation_d=a,
            A_role='anchor' if index in args.anchors else 'heldout', AE0=ae0, AE1=ae1, AE2=ae2)
        primary_failure_total += (raw_scores['failures']['failed_states']+
                                  empirical_scores['failures']['failed_states'])
        beta_failure_total += int(np.sum(beta_reason != 0))
        maps.update({key+'_m_mm': state['m'], key+'_oracle_gaze_deg': state['oracle'],
            key+'_AE0_P0_gaze_deg': pzero, key+'_AE0_interpolated_gaze_deg': interpolated,
            key+'_AE0_dgaze_dA_arcmin_per_D': derivative,
            key+'_AE1_raw_mu': mu, key+'_AE1_raw_beta': beta, key+'_AE1_raw_A_D': raw_A,
            key+'_AE1_failure_reason': raw_reason, key+'_AE1_beta_A_D': beta_A,
            key+'_AE1_beta_failure_reason': beta_reason, key+'_AE1_gaze_deg': raw_gaze,
            key+'_AE1_radial_residual_um': residual_um,
            key+'_AE1_contraction_proxy_UNVERIFIED': abs(effective_slope)*rho2,
            key+'_AE2_Sigma_mm2': state['Sigma'], key+'_AE2_sigma': sigma,
            key+'_AE2_unbounded_Q_A_D': predicted_A, key+'_AE2_failure_reason': reason,
            key+'_AE2_A_D': np.where(reason == 0, predicted_A, np.nan),
            key+'_AE2_gaze_deg': empirical_gaze})
        for level, bound in tolerance_maps.items():
            maps[key+'_AE0_required_A_D_level_'+level.replace('.', 'p')] = bound
        pooled['anchor_A' if index in args.anchors else 'heldout_A'].append(dict(
            groups=groups, raw_A_error=raw_A-a, empirical_A_error=np.where(reason == 0, predicted_A-a, np.nan),
            oracle_error=state['oracle']-state['w'], P0_error=pzero-state['w'],
            interpolated_error=interpolated-state['w'], raw_gaze_error=raw_gaze-state['w'],
            empirical_gaze_error=empirical_gaze-state['w'],
            raw_oracle_delta=raw_gaze-state['oracle'], empirical_oracle_delta=empirical_gaze-state['oracle']))
        report['ladder'].append(dict(shard=key, accommodation_d=a,
            A_role=report['results'][key]['A_role'],
            oracle_gaze=ae0['oracle_A_gaze']['overall']['metrics'],
            P0_gaze=ae0['P0_gaze']['overall']['metrics'],
            interpolated_gaze=ae0['interpolated_A_gaze']['overall']['metrics'],
            raw_A=raw_scores['A_error_D']['overall']['metrics'],
            empirical_A=ae2['A_error_D']['overall']['metrics'],
            raw_gaze=raw_scores['gaze_vs_truth_arcmin']['overall']['metrics'],
            empirical_gaze=ae2['gaze_vs_truth_arcmin']['overall']['metrics'],
            raw_failed_states=raw_scores['failures']['failed_states'],
            empirical_failed_states=ae2['failures']['failed_states']))
    for role, entries in pooled.items():
        groups = {name: np.concatenate([item['groups'][name] for item in entries])
                  for name in entries[0]['groups']}
        record = dict(accommodation_count=len(entries),
            state_count=sum(len(item['raw_A_error']) for item in entries))
        for name in entries[0]:
            if name == 'groups':
                continue
            values = np.concatenate([item[name] for item in entries])
            record[name] = (scalar_summaries(values, groups) if values.ndim == 1 else
                            tw.gaze_summaries(values, groups))
        report['pooled'][role] = record
    report.update(status='completed_with_failures' if primary_failure_total+beta_failure_total else 'complete',
                  exit_code=0, primary_estimator_failed_state_routes=primary_failure_total,
                  secondary_beta_failed_state_routes=beta_failure_total)


def s1_metadata(report):
    report['development_label'] = DEVELOPMENT
    report['decisions'].update(N1='AE2b + AE3e + AE4e + SEP only',
        N2='Joint surface primary; strict separable route always retained',
        N3=DEVELOPMENT, N4='1 arcmin headline; 0.1 arcmin companion; neither is a gate',
        N5='Full field headline; field_lt12 companion', N6='No E1 or D; deferred')
    report['definitions'].update(AE3e='oracle-gaze diagnostic, not a runtime estimator',
        AE4e_true='oracle-input one-pass consistency/bias diagnostic; not a fixed-point result',
        AE4e_runtime='P0, mid and pass2 use measured inputs and frozen calibration only',
        AE4e_sensitivity=LOCAL_SENSITIVITY, surface_derivative=SURFACE_DERIVATIVE,
        SEP='Data only, descriptive; not calibration and not an estimator',
        headline_cell='heldout_A / odd_odd / full field',
        companion_cell='heldout_A / odd_odd / field_lt12')
    report['caveats'].append(DEVELOPMENT)
    report['caveats'] = [text for text in report['caveats']
        if not text.startswith('AE3--AE6, reversed maps')]


def s1_cells(groups):
    """Every crossing retains its full denominator, including zero failures."""
    cells = {name: groups[name] for name in (*gp.SPLITS, 'field_lt12')}
    for split in gp.SPLITS:
        cells[split+'/field_lt12'] = groups[split] & groups['field_lt12']
        for name in gp.BIN_NAMES:
            cells[split+'/'+name] = groups[split] & groups[name]
    return cells


def s1_location(state, index, A_error=None, gaze_error=None):
    if index is None:
        return None
    return dict(state_id=int(state['state_ids'][index]), gaze_deg=state['w'][index],
        accommodation_D=float(np.broadcast_to(state['A'], (len(state['w']),))[index]),
        A_error_D=None if A_error is None else float(A_error[index]),
        gaze_error_arcmin=None if gaze_error is None else float(gaze_error[index]))


def s1_extreme(values, mask, state, minimum=False, A_error=None, gaze_error=None):
    ids = np.flatnonzero(mask)
    finite = ids[np.isfinite(values[ids])]
    if not len(ids):
        return dict(value=None, location=None, undefined_states=0)
    invalid = ids[~np.isfinite(values[ids])]
    pick = None if not len(finite) else int(finite[
        np.argmin(abs(values[finite])) if minimum else np.argmax(abs(values[finite]))])
    record = dict(value=None if pick is None else float(abs(values[pick])),
        location=s1_location(state, pick, A_error, gaze_error), undefined_states=int(len(invalid)))
    if len(invalid):
        record.update(valid_states_only=dict(value=record['value'], location=record['location']),
            value=float('nan') if minimum else float('inf'),
            location=s1_location(state, int(invalid[0]), A_error, gaze_error))
    return record


def s1_reason_counts(reason, mask):
    return {name: int(np.sum(reason[mask] == code)) for code, name in S1_REASONS}


def s1_scores(predicted_A, reason, gaze, state, label):
    # Sanitize BOTH coordinates before any score or saved estimator output.
    valid = (reason == 0) & np.isfinite(predicted_A) & np.isfinite(gaze).all(axis=1)
    reason = np.asarray(reason, np.int8).copy()
    reason[(reason == 0) & ~valid] = 1
    predicted_A = np.where(reason == 0, predicted_A, np.nan)
    gaze = np.where((reason == 0)[:, None], gaze, np.nan)
    A_error = predicted_A-state['A']
    gaze_error = 60*np.linalg.norm(gaze-state['w'], axis=1)
    oracle_error = 60*np.linalg.norm(gaze-state['oracle'], axis=1)
    cells = {}
    for name, mask in s1_cells(state['groups']).items():
        count, good = int(mask.sum()), int(np.sum(mask & (reason == 0)))
        cells[name] = dict(states=count, valid_states=good,
            coverage=None if count == 0 else good/count,
            failures_by_reason=s1_reason_counts(reason, mask),
            A_error_D=scalar_metric(A_error, mask),
            gaze_vs_truth_arcmin=scalar_metric(gaze_error, mask, 'arcmin'),
            gaze_vs_oracle_A_arcmin=scalar_metric(oracle_error, mask, 'arcmin'),
            readout_fractions={str(level): None if count == 0 else
                float(np.sum(mask & np.isfinite(gaze_error) & (gaze_error <= level))/count)
                for level in (1., .1)},
            maximum_A_error=s1_extreme(A_error, mask, state, A_error=A_error, gaze_error=gaze_error),
            maximum_gaze_error=s1_extreme(gaze_error, mask, state, A_error=A_error, gaze_error=gaze_error),
            maximum_gaze_vs_oracle_A_error=s1_extreme(oracle_error, mask, state,
                A_error=A_error, gaze_error=gaze_error))
    return dict(development_label=DEVELOPMENT, input_label=label, cells=cells), predicted_A, reason, gaze


def s1_diagnostic(values, state, unit, label, errors=None):
    """Descriptive extrema include undefined counts rather than dropping states."""
    A_error, gaze_error = (None, None) if errors is None else errors
    cells = {}
    for name, mask in s1_cells(state['groups']).items():
        finite = mask & np.isfinite(values)
        cells[name] = dict(metrics=scalar_metric(values, mask, unit),
            minimum_abs=s1_extreme(values, mask, state, True, A_error, gaze_error),
            maximum_abs=s1_extreme(values, mask, state, False, A_error, gaze_error),
            median_abs_valid_states_only=(None if not np.any(finite) else
                                           float(np.median(abs(values[finite])))))
    return dict(development_label=DEVELOPMENT, label=label, cells=cells)


def s1_q3_design(m, sigma, scale):
    x, y = (m/scale).T
    return np.column_stack((np.ones(len(m)), sigma, sigma*sigma, x*x, x*y, y*y,
        sigma*x*x, sigma*x*y, sigma*y*y, x**4, x**3*y, x*x*y*y, x*y**3, y**4))


def s1_fit_q3(states, cal, args):
    matrices, targets = [], []
    for index in args.anchors:
        state, train = states[index], states[index]['groups']['train']
        matrices.append(s1_q3_design(state['m'][train],
            state['Sigma'][train]/cal['Sigma0']-1, state['scale']))
        targets.append(np.full(int(train.sum()), state['A']))
    coefficient, checked = linear_fit(np.concatenate(matrices), np.concatenate(targets),
                                      'S1 Q3 anchor training only')
    return dict(coefficient=coefficient, record=dict(**checked, anchor_indices=args.anchors,
        training_states_per_anchor=int(states[0]['groups']['train'].sum()),
        terms=['1', 'sigma', 'sigma^2', 'x^2', 'xy', 'y^2', 'sigma*x^2', 'sigma*xy',
               'sigma*y^2', 'x^4', 'x^3*y', 'x^2*y^2', 'x*y^3', 'y^4'],
        coefficients_D=coefficient, development_label=DEVELOPMENT))


def s1_empirical_prediction(state, cal, fit, degree=2, clip=False):
    sigma = state['Sigma']/cal['Sigma0']-1
    matrix = (s1_q3_design(state['m'], sigma, state['scale']) if degree == 3 else
              empirical_design(state['m'], sigma, state['scale'], 2)[0])
    unbounded = matrix@fit['coefficient']
    reason = np.zeros(len(unbounded), np.int8)
    reason[~np.isfinite(unbounded)] = 1
    reason[np.isfinite(unbounded) & ((unbounded < 0) | (unbounded > 4))] = 2
    predicted = unbounded.copy()
    if clip:
        outside = reason == 2
        predicted[outside] = np.clip(predicted[outside], 0., 4.)
        reason[outside] = 0
    predicted[reason != 0] = np.nan
    return predicted, reason


def s1_size_design(tau, basis):
    x, y = np.asarray(tau).T
    columns = [x*x, y*y, x**4, x*x*y*y, y**4]
    if basis == 'E6':
        columns.extend((x**6, x**4*y*y, x*x*y**4, y**6))
    return np.column_stack(columns)


def s1_fit_surface(states, cal, args, basis):
    anchors, records, gamma = args.anchors, {}, []
    for index in anchors:
        state, train = states[index], states[index]['groups']['train']
        tau = tw.tangent_coordinates(state['w'])['primary']
        target = state['mu_raw']/cal['mu'][index]-1
        coefficient, checked = linear_fit(s1_size_design(tau[train], basis), target[train],
                                           f'S1 {basis} A{index:02d} gaze correction')
        gamma.append(coefficient)
        residual = s1_size_design(tau, basis)@coefficient-target
        records[f'A{index:02d}'] = dict(**checked, gamma=coefficient,
            fit_residual=s1_diagnostic(residual, state, 'relative_size',
                'anchor own-coefficient fit residual; training gaze only calibrates'))
    a = np.array([states[index]['A'] for index in anchors])
    gamma = np.stack(gamma)
    trend, checked = linear_fit(np.column_stack((np.ones(len(a)), a, a*a)), gamma,
                               'S1 '+basis+' anchor-only quadratic gamma trend')
    return dict(basis=basis, gamma_A00=gamma[0].copy(), trend=trend,
        grid=np.arange(401, dtype=float)*.01,
        record=dict(basis=basis, anchor_indices=anchors, anchors=records,
            gamma_anchors=gamma, gamma_trend=trend, trend_check=checked,
            development_label=DEVELOPMENT))


def s1_surface_table(surface, cal, gaze):
    a = surface['grid']
    gamma = np.column_stack((np.ones(len(a)), a, a*a))@surface['trend']
    mu_cal = np.interp(a, cal['A'][cal['anchors']], cal['mu'][cal['anchors']])
    tau = tw.tangent_coordinates(gaze)['primary']
    return (1+s1_size_design(tau, surface['basis'])@gamma.T)*mu_cal[None, :]


def s1_table_inverse(grid, table, targets, clip=False):
    """Strict per-state table inverse; clipping exists only for the named row."""
    table, targets = np.asarray(table, float), np.asarray(targets, float)
    result = np.full(len(targets), np.nan)
    reason = np.zeros(len(targets), np.int8)
    finite = np.isfinite(targets) & np.isfinite(table).all(axis=1)
    delta = np.diff(table, axis=1)
    increasing, decreasing = np.all(delta > 0, axis=1), np.all(delta < 0, axis=1)
    monotone = increasing | decreasing
    reason[~finite] = 1
    reason[finite & ~monotone] = 4
    sign_changes = np.sum((table[:, :-1]-targets[:, None])*
                          (table[:, 1:]-targets[:, None]) < 0, axis=1)
    for index in np.flatnonzero(finite & monotone):
        values = table[index] if increasing[index] else table[index, ::-1]
        nodes = grid if increasing[index] else grid[::-1]
        if targets[index] < values[0] or targets[index] > values[-1]:
            reason[index] = 2
            if clip:
                endpoint = 0 if abs(targets[index]-table[index, 0]) <= abs(
                    targets[index]-table[index, -1]) else -1
                result[index], reason[index] = grid[endpoint], 0
        else:
            result[index] = np.interp(targets[index], values, nodes)
    result[reason != 0] = np.nan
    return result, reason, sign_changes


def s1_surface_prediction(state, surface, cal, gaze, route, clip=False):
    if route == 'joint':
        return s1_table_inverse(surface['grid'], s1_surface_table(surface, cal, gaze),
                                state['mu_raw'], clip)
    factor = 1+s1_size_design(tw.tangent_coordinates(gaze)['primary'],
                             surface['basis'])@surface['gamma_A00']
    target = np.divide(state['mu_raw'], factor, out=np.full(len(factor), np.nan), where=factor != 0)
    predicted, reason, _ = monotone_inverse(cal['A'][cal['anchors']],
        cal['mu'][cal['anchors']], target)
    return predicted, reason, np.zeros(len(factor), np.int64)


def s1_pass2(inner, operation):
    predicted, reason = inner[0].copy(), inner[1].copy()
    valid = reason == 0
    predicted[~valid] = np.nan
    if np.any(valid):
        outer_A, outer_reason = operation(np.where(valid, inner[0], np.nan))[:2]
        predicted[valid], reason[valid] = outer_A[valid], outer_reason[valid]
    predicted[reason != 0] = np.nan
    return predicted, reason


def s1_local_sensitivity(true, lower, upper, pzero, accommodation, h):
    """Arguments are (estimate, reason); no failed endpoint enters a difference."""
    if accommodation == 0:
        left, right, denominator, convention = true, upper, h, 'forward one-sided'
    elif accommodation == 4:
        left, right, denominator, convention = lower, true, h, 'backward one-sided'
    else:
        left, right, denominator, convention = lower, upper, 2*h, 'central'
    valid = (left[1] == 0) & (right[1] == 0)
    c = np.where(valid, (right[0]-left[0])/denominator, np.nan)
    c0 = np.full_like(c, np.nan)
    if accommodation > 0:
        valid0 = (pzero[1] == 0) & (true[1] == 0)
        c0[valid0] = (pzero[0][valid0]-true[0][valid0])/(-accommodation)
    return c, c0, convention


def s1_runtime_predictions(state, model, cal, q2, q3=None, surface=None):
    """Frozen-calibration runtime path: never reads truth A, w, or oracle."""
    predictions = {}
    for name, fit, degree in (('Q2', q2, 2), ('Q3', q3, 3)):
        if fit is not None:
            for clip in (False, True):
                predicted, reason = s1_empirical_prediction(state, cal, fit, degree, clip)
                predictions[name+('-clip' if clip else '')] = (
                    predicted, reason, evaluate_gaze(model, state['m'], predicted))
    if surface is not None:
        for route in ('joint', 'sep'):
            def operation(assumed, route=route):
                return s1_surface_prediction(state, surface, cal,
                    evaluate_gaze(model, state['m'], assumed), route)
            first = operation(0.)
            for name, output in (('P0', first), ('mid', operation(2.)),
                                 ('pass2', s1_pass2(first, operation))):
                predictions[f'AE4e-{route}-{name}'] = (
                    output[0], output[1], evaluate_gaze(model, state['m'], output[0]))
    return predictions


def s1_jacobian_measures(jacobian):
    """Batch 16x3 geometry; singular/undefined states remain NaN, never stop."""
    jacobian = np.asarray(jacobian, float)
    count = len(jacobian)
    norms = np.linalg.norm(jacobian, axis=1)
    answer = {name: np.full(count, np.nan) for name in
              ('condition', 'sin_theta_A', 'leverage_um_per_D', 'inverse_leverage_D_per_um')}
    answer.update(column_norms=norms, compensating_gaze=np.full((count, 2), np.nan),
                  independent_A=np.full(jacobian.shape[:2], np.nan))
    candidate = (np.isfinite(jacobian).all(axis=(1, 2)) & np.isfinite(norms).all(axis=1) &
                 np.all(norms > 0, axis=1))
    ids = np.flatnonzero(candidate)
    if not len(ids):
        return answer
    J = jacobian[ids]
    W, A = J[:, :, :2], J[:, :, 2]
    left_W, singular_W, right_W = np.linalg.svd(W, full_matrices=False)
    rank2 = singular_W[:, -1] > gp.RCOND*singular_W[:, 0]
    ids, J, W, A = ids[rank2], J[rank2], W[rank2], A[rank2]
    left_W, singular_W, right_W = left_W[rank2], singular_W[rank2], right_W[rank2]
    if not len(ids):
        return answer
    # Full-rank SVD evaluation of (W.T W)^-1 W.T avoids squaring its condition.
    coefficient = np.einsum('nij,nj->ni', np.swapaxes(right_W, 1, 2),
        np.einsum('nki,nk->ni', left_W, A)/singular_W)
    independent = A-np.einsum('nki,ni->nk', W, coefficient)
    leverage = np.linalg.norm(independent, axis=1)
    singular = np.linalg.svd(J/norms[ids, None, :], compute_uv=False)
    rank3 = singular[:, -1] > gp.RCOND*singular[:, 0]
    answer['condition'][ids[rank3]] = singular[rank3, 0]/singular[rank3, -1]
    answer['sin_theta_A'][ids] = leverage/norms[ids, 2]
    answer['leverage_um_per_D'][ids] = leverage
    answer['inverse_leverage_D_per_um'][ids] = np.divide(1., leverage,
        out=np.full_like(leverage, np.nan), where=leverage > 0)
    answer['compensating_gaze'][ids] = -coefficient
    answer['independent_A'][ids] = independent
    return answer


def s1_data_derivatives(states, quantity):
    """Checked 101*i_ADE+i_BDE ordering; derivatives per deg and per D."""
    base = states[0]
    coordinate = np.linspace(-12.5, 12.5, 101)
    ade, bde = np.indices((101, 101)).reshape(2, -1)
    expected_w = np.column_stack((coordinate[bde], -coordinate[ade]))
    a = np.array([states[index]['A'] for index in INDICES])
    require(np.array_equal(base['state_ids'], np.arange(10201)) and
            np.array_equal(base['w'], expected_w) and
            all(np.array_equal(state['w'], expected_w) for state in states.values()) and
            np.allclose(np.diff(a), .2, rtol=0, atol=1e-14) and
            np.allclose(a, .2*np.arange(21), rtol=0, atol=1e-14),
            'SEP requires ordered 101x101 uniform 0.25 deg grid and 0.2 D shards')
    values = np.stack([states[index][quantity] for index in INDICES])
    shaped = values.reshape((21, 101, 101)+values.shape[2:])
    dA = np.gradient(shaped, .2, axis=0, edge_order=2)
    dWx = np.gradient(shaped, .25, axis=2, edge_order=2)
    dWy = -np.gradient(shaped, .25, axis=1, edge_order=2)
    return tuple(item.reshape(values.shape) for item in (dWx, dWy, dA))


def s1_mean_compensation(states):
    dx, dy, da = s1_data_derivatives(states, 'm')
    blocks = np.stack((dx, dy), axis=-1)
    result = np.full(da.shape, np.nan)
    finite = np.isfinite(blocks).all(axis=(-1, -2)) & np.isfinite(da).all(axis=-1)
    ids = np.argwhere(finite)
    if len(ids):
        matrices = blocks[finite]
        singular = np.linalg.svd(matrices, compute_uv=False)
        valid = singular[:, -1] > gp.RCOND*singular[:, 0]
        selected = tuple(ids[valid].T)
        result[selected] = -60*np.linalg.solve(matrices[valid], da[finite][valid, :, None])[..., 0]
    return result


def s1_additive_remainder(mu):
    logarithm = np.full_like(np.asarray(mu, float), np.nan)
    np.log(mu, out=logarithm, where=np.asarray(mu) > 0)
    return logarithm-logarithm.mean(axis=1, keepdims=True)-logarithm.mean(
        axis=0, keepdims=True)+logarithm.mean()


def s1_save_estimator(bucket, states, model, index, name, prediction, label, status='complete'):
    state, key = states[index], f'A{index:02d}'
    predicted, reason = prediction[:2]
    gaze = (prediction[2] if len(prediction) == 3 and np.asarray(prediction[2]).shape == (len(predicted), 2)
            else evaluate_gaze(model, state['m'], predicted))
    record, predicted, reason, gaze = s1_scores(predicted, reason, gaze, state, label)
    record['status'] = status
    bucket['estimators'].setdefault(key, {})[name] = record
    prefix = key+'_'+name.replace('-', '_')
    bucket['maps'].update({prefix+'_A_D': predicted, prefix+'_failure_reason': reason,
                           prefix+'_gaze_deg': gaze})


def s1_commit(report, maps, bucket):
    maps.update(bucket['maps'])
    report['calibration'].update(bucket['calibration'])
    for key, records in bucket['estimators'].items():
        report['results'][key]['S1']['estimators'].update(records)
    for kind in ('sensitivity', 'SEP', 'surface_diagnostics'):
        for key, records in bucket[kind].items():
            report['results'][key]['S1'][kind].update(records)


def s1_block(report, maps, states, model, name, operation, unavailable=()):
    bucket = {key: {} for key in ('maps', 'calibration', 'estimators',
                                  'sensitivity', 'SEP', 'surface_diagnostics')}
    try:
        operation(bucket)
    except Exception as exc:
        error = dict(block=name, exception=type(exc).__name__, message=str(exc),
                     numbers=getattr(exc, 'numbers', {}))
        report['errors'].append(error)
        report['S1_blocks'][name] = dict(status='failed', error=error)
        # Discard only this uncommitted block, and write explicit unavailable rows.
        bucket = {key: {} for key in bucket}
        bucket['calibration'][name] = dict(status='estimator_unavailable', error=error)
        for index, state in states.items():
            for estimator in unavailable:
                s1_save_estimator(bucket, states, model, index, estimator,
                    (np.full(len(state['w']), np.nan), np.full(len(state['w']), 5, np.int8)),
                    ('oracle-gaze diagnostic, not a runtime estimator' if estimator.startswith('AE3e') else
                     'oracle-input diagnostic' if estimator.endswith('-true') else
                     'runtime-legal measured inputs with frozen calibration'), 'estimator_unavailable')
            if name.startswith('SEP_'):
                bucket['SEP'][f'A{index:02d}'] = {name: dict(status='unavailable', error=error)}
        s1_commit(report, maps, bucket)
        return False
    report['S1_blocks'][name] = dict(status='complete')
    s1_commit(report, maps, bucket)
    return True


def s1_surface_block(bucket, states, model, cal, args, basis):
    surface = s1_fit_surface(states, cal, args, basis)
    bucket['calibration']['size_surface_'+basis] = surface['record']
    for index, state in states.items():
        key, a = f'A{index:02d}', state['A']
        tau = tw.tangent_coordinates(state['w'])['primary']
        gamma = np.array([1., a, a*a])@surface['trend']
        residual = s1_size_design(tau, basis)@gamma-(state['mu_raw']/cal['mu'][index]-1)
        table = s1_surface_table(surface, cal, state['w'])
        node = int(round(a/.01))
        left, right = max(0, node-1), min(400, node+1)
        dM = (table[:, right]-table[:, left])/(surface['grid'][right]-surface['grid'][left])
        mu_grid = np.interp(surface['grid'], cal['A'][cal['anchors']], cal['mu'][cal['anchors']])
        dmu = (mu_grid[right]-mu_grid[left])/(surface['grid'][right]-surface['grid'][left])
        ratio = np.divide(dM, dmu, out=np.full_like(dM, np.nan), where=dmu != 0)
        bucket['surface_diagnostics'][key] = {basis: dict(development_label=DEVELOPMENT,
            input_label='oracle-gaze diagnostic, not a runtime estimator',
            interpolated_coefficient_fit_residual=s1_diagnostic(residual, state, 'relative_size',
                'anchor-only quadratic gamma trend; held-out shards score only'),
            dM_dA=s1_diagnostic(dM, state, 'relative_size_per_D', SURFACE_DERIVATIVE),
            dM_dA_over_dmu_cal_dA=s1_diagnostic(ratio, state, 'ratio', SURFACE_DERIVATIVE),
            dmu_cal_dA=float(dmu), derivative_convention=SURFACE_DERIVATIVE)}
        bucket['maps'].update({key+'_AE3e_'+basis+'_fit_residual': residual,
            key+'_AE3e_'+basis+'_dM_dA': dM, key+'_AE3e_'+basis+'_dM_dA_ratio': ratio})
        suffix = '' if basis == 'E4' else '-E6'
        for route in ('joint', 'sep'):
            output = s1_surface_prediction(state, surface, cal, state['w'], route)
            name = 'AE3e-'+route+suffix
            s1_save_estimator(bucket, states, model, index, name, output,
                              'oracle-gaze diagnostic, not a runtime estimator')
            if route == 'joint':
                bucket['estimators'][key][name]['sign_change_count'] = dict(
                    definition='strict adjacent sign crossings of M-mu_raw; exact node roots counted separately',
                    by_count={str(int(value)): int(np.sum(output[2] == value)) for value in np.unique(output[2])})
                bucket['maps'][key+'_'+name.replace('-', '_')+'_sign_changes'] = output[2]
                exact_roots = np.sum(table == state['mu_raw'][:, None], axis=1)
                bucket['estimators'][key][name]['sign_change_count']['exact_node_roots_by_count'] = {
                    str(int(value)): int(np.sum(exact_roots == value)) for value in np.unique(exact_roots)}
                bucket['maps'][key+'_'+name.replace('-', '_')+'_exact_node_roots'] = exact_roots
        if basis != 'E4':
            continue
        clipped = s1_surface_prediction(state, surface, cal, state['w'], 'joint', True)
        s1_save_estimator(bucket, states, model, index, 'AE3e-joint-clip', clipped,
                          'oracle-gaze diagnostic, not a runtime estimator; named clipping variant')
        for route in ('joint', 'sep'):
            def operation(assumed, route=route):
                return s1_surface_prediction(state, surface, cal,
                    evaluate_gaze(model, state['m'], assumed), route)
            true, pzero, mid = operation(a), operation(0.), operation(2.)
            lower = operation(a-args.sensitivity_step) if a > 0 else true
            upper = operation(a+args.sensitivity_step) if a < 4 else true
            second = s1_pass2(pzero, operation)
            for variant, output in (('true', true), ('P0', pzero), ('mid', mid), ('pass2', second)):
                s1_save_estimator(bucket, states, model, index, f'AE4e-{route}-{variant}', output,
                    ('oracle-input one-pass consistency/bias; not a fixed-point or converged result'
                     if variant == 'true' else 'runtime-legal measured inputs with frozen calibration'))
                if route == 'joint':
                    assumed = {'true': a, 'P0': 0., 'mid': 2., 'pass2':
                        np.where(pzero[1] == 0, pzero[0], np.nan)}[variant]
                    evaluated_table = s1_surface_table(surface, cal,
                        evaluate_gaze(model, state['m'], assumed))
                    changes = np.sum((evaluated_table[:, :-1]-state['mu_raw'][:, None])*
                        (evaluated_table[:, 1:]-state['mu_raw'][:, None]) < 0, axis=1)
                    roots = np.sum(evaluated_table == state['mu_raw'][:, None], axis=1)
                    diagnostic_name = f'AE4e-{route}-{variant}'
                    bucket['estimators'][key][diagnostic_name]['sign_change_count'] = dict(
                        definition='strict adjacent sign crossings of M-mu_raw; exact table-node roots separate',
                        by_count={str(int(v)): int(np.sum(changes == v)) for v in np.unique(changes)},
                        exact_node_roots_by_count={str(int(v)): int(np.sum(roots == v)) for v in np.unique(roots)})
                    diagnostic_prefix = key+'_'+diagnostic_name.replace('-', '_')
                    bucket['maps'][diagnostic_prefix+'_sign_changes'] = changes
                    bucket['maps'][diagnostic_prefix+'_exact_node_roots'] = roots
            c, c0, convention = s1_local_sensitivity(true, lower, upper, pzero, a, args.sensitivity_step)
            true_valid = np.where(true[1] == 0, true[0], np.nan)
            errors = (true_valid-a, 60*np.linalg.norm(evaluate_gaze(model, state['m'], true_valid)-state['w'], axis=1))
            bucket['sensitivity'].setdefault(key, {})[route] = dict(
                development_label=DEVELOPMENT, label=LOCAL_SENSITIVITY,
                input_label='oracle-input diagnostic', difference_convention=convention,
                c=s1_diagnostic(c, state, 'dimensionless', LOCAL_SENSITIVITY, errors),
                c0=s1_diagnostic(c0, state, 'dimensionless', LOCAL_SENSITIVITY, errors),
                extrema_error_reference='A and gaze errors of this route true-input one-pass row at the extremum state',
                c0_A00='undefined: secant denominator is zero',
                count_abs_c_ge1={name: int(np.sum(mask & np.isfinite(c) & (abs(c) >= 1)))
                                for name, mask in s1_cells(state['groups']).items()})
            bucket['maps'].update({key+f'_AE4e_{route}_c': c, key+f'_AE4e_{route}_c0': c0})


def s1_sep_joint(bucket, states, model):
    dx, dy, da = s1_data_derivatives(states, 'Zring')
    mean = s1_mean_compensation(states)
    for index, state in states.items():
        key = f'A{index:02d}'
        J = np.stack((dx[index].reshape(-1, 16)*1000/60,
            dy[index].reshape(-1, 16)*1000/60, da[index].reshape(-1, 16)*1000), axis=-1)
        measured = s1_jacobian_measures(J)
        analytic = evaluate_gaze(model, state['m'], state['A'], derivative=True)*60
        values = {name: measured[name] for name in
                  ('condition', 'sin_theta_A', 'leverage_um_per_D', 'inverse_leverage_D_per_um')}
        for axis, name in enumerate(('wx_um_per_arcmin', 'wy_um_per_arcmin', 'A_um_per_D')):
            values['column_norm_'+name] = measured['column_norms'][:, axis]
        for axis in range(2):
            values['compensating_gaze_'+str(axis)+'_arcmin_per_D'] = measured['compensating_gaze'][:, axis]
            values['mean_only_'+str(axis)+'_arcmin_per_D'] = mean[index, :, axis]
            values['analytic_dP_dA_'+str(axis)+'_arcmin_per_D'] = analytic[:, axis]
            values['mean_minus_analytic_'+str(axis)+'_arcmin_per_D'] = mean[index, :, axis]-analytic[:, axis]
        bucket['SEP'][key] = {'joint': {name: s1_diagnostic(value, state, name,
            'SEP data only, descriptive; not calibration and not an estimator') for name, value in values.items()}}
        for name, value in values.items():
            bucket['maps'][key+'_SEP_'+name] = value


def s1_sep_shape(bucket, states, cal):
    dx, dy, da = s1_data_derivatives(states, 'Zshape')
    d = -five.LABELS[gp.estimator_nodes()['E2'][1]]
    r2 = np.sum(d*d, axis=1)
    baseline = cal['H0']*(1+cal['q0']*r2)
    # Physical mm-coordinate column space of the existing two-radius projection.
    radial = np.stack((baseline[:, None]*d, baseline[:, None]*d*(r2-1)[:, None]), axis=-1).reshape(16, 2)
    radial_Q = np.linalg.qr(radial, mode='reduced')[0]
    for index, state in states.items():
        key = f'A{index:02d}'
        J = np.stack((dx[index].reshape(-1, 16)*1000/60,
            dy[index].reshape(-1, 16)*1000/60, da[index].reshape(-1, 16)*1000), axis=-1)
        measured = s1_jacobian_measures(J)
        independent = measured['independent_A']
        energy = np.sum(independent*independent, axis=1)
        radial_energy = np.sum((independent@radial_Q)**2, axis=1)
        fraction = np.divide(radial_energy, energy, out=np.full_like(energy, np.nan), where=energy > 0)
        values = dict(shape_condition=measured['condition'], shape_sin_theta_A=measured['sin_theta_A'],
            shape_leverage_um_per_D=measured['leverage_um_per_D'],
            shape_gaze_um_per_arcmin=np.linalg.norm(J[:, :, :2], axis=(1, 2)), radial_fraction=fraction)
        bucket['SEP'][key] = {'shape': {name: s1_diagnostic(value, state, name,
            'SEP data only, descriptive; not calibration and not an estimator') for name, value in values.items()}}
        bucket['maps'].update({key+'_SEP_'+name: value for name, value in values.items()})


def s1_sep_cJ(bucket, states, maps):
    dx, dy, da = s1_data_derivatives(states, 'mu_raw')
    compensation = s1_mean_compensation(states)
    for index, state in states.items():
        gradient = np.column_stack((dx[index], dy[index]))/60
        factor = np.divide(-gradient, da[index, :, None],
            out=np.full_like(gradient, np.nan), where=da[index, :, None] != 0)
        cJ = np.sum(factor*compensation[index], axis=1)
        key = f'A{index:02d}'
        values = dict(cJ=cJ)
        comparison = maps.get(key+'_AE4e_joint_c')
        if comparison is not None:
            values['cJ_minus_c_joint'] = cJ-comparison
        bucket['SEP'][key] = {'linearised_sensitivity': {name: s1_diagnostic(value, state, 'dimensionless',
            'SEP exact-data local size sensitivity; '+LOCAL_SENSITIVITY) for name, value in values.items()}}
        bucket['maps'].update({key+'_SEP_'+name: value for name, value in values.items()})


def s1_sep_additive(bucket, states, cal):
    remainder = s1_additive_remainder(np.stack([states[index]['mu_raw'] for index in INDICES]))
    log_slope = np.diff(np.log(cal['mu']))/np.diff(cal['A'])
    conversion = float(np.mean(abs(log_slope)))
    if not np.isfinite(conversion) or conversion == 0:
        raise NumericalFailure('SEP additive size-to-D conversion undefined')
    for index, state in states.items():
        key = f'A{index:02d}'
        bucket['SEP'][key] = dict(additive_remainder=dict(
            log_size=s1_diagnostic(remainder[index], state, 'log_relative_size',
                'SEP two-way row/column mean fit; all A and gaze, descriptive, not calibration'),
            equivalent_D=s1_diagnostic(remainder[index]/conversion, state, 'D',
                'SEP additive remainder / mean absolute adjacent dln(mu_cal)/dA; descriptive'),
            conversion_mean_abs_dln_mu_cal_per_D=conversion))
        bucket['maps'].update({key+'_SEP_additive_remainder': remainder[index],
                               key+'_SEP_additive_remainder_D': remainder[index]/conversion})


def s1_pooled(report, maps, states, model, args):
    report['S1_pooled'] = {}
    names = sorted({name for state in report['results'].values() for name in state['S1']['estimators']})
    for role, indices in (('anchor_A', args.anchors),
                          ('heldout_A', [index for index in INDICES if index not in args.anchors])):
        entries = [states[index] for index in indices]
        pooled_state = dict(w=np.concatenate([s['w'] for s in entries]),
            A=np.concatenate([np.full(len(s['w']), s['A']) for s in entries]),
            oracle=np.concatenate([s['oracle'] for s in entries]),
            state_ids=np.concatenate([s['state_ids'] for s in entries]),
            groups={name: np.concatenate([s['groups'][name] for s in entries]) for name in entries[0]['groups']})
        rows = report['S1_pooled'][role] = {}
        for name in names:
            prefix = '_'+name.replace('-', '_')
            predicted = np.concatenate([maps[f'A{index:02d}'+prefix+'_A_D'] for index in indices])
            reason = np.concatenate([maps[f'A{index:02d}'+prefix+'_failure_reason'] for index in indices])
            gaze = np.concatenate([maps[f'A{index:02d}'+prefix+'_gaze_deg'] for index in indices])
            rows[name] = s1_scores(predicted, reason, gaze, pooled_state,
                report['results'][f'A{indices[0]:02d}']['S1']['estimators'][name]['input_label'])[0]


def analyse_s1(states, report, args, log, maps):
    for index, state in states.items():
        state['groups']['field_lt12'] = np.logical_or.reduce(
            [state['groups'][name] for name in gp.BIN_NAMES[:3]])
        state['mu_raw'] = maps[f'A{index:02d}_AE1_raw_mu']
        report['results'][f'A{index:02d}']['S1'] = {key: {} for key in
            ('estimators', 'sensitivity', 'SEP', 'surface_diagnostics')}
    maps['group_field_lt12'] = states[0]['groups']['field_lt12']
    report['split_counts']['field_lt12'] = int(states[0]['groups']['field_lt12'].sum())
    report['S1_blocks'] = {}
    model = coefficient_model(states, args)
    cal = size_calibration(states, args)
    q2 = fit_empirical(states, cal, args)
    a, mu, H = cal['A'], cal['mu'], np.array([states[i]['H'] for i in INDICES])
    report['calibration']['zero_gaze_adjacent_slopes'] = dict(
        interval_midpoint_D=(a[:-1]+a[1:])/2, mu_prime=np.diff(mu)/np.diff(a),
        mu_prime_over_mu=np.diff(mu)/np.diff(a)/((mu[:-1]+mu[1:])/2),
        H_prime_over_H=np.diff(H)/np.diff(a)/((H[:-1]+H[1:])/2),
        convention='adjacent differences / interval-midpoint value for logarithmic slopes')
    def q2_block(bucket):
        for index, state in states.items():
            for name, prediction in s1_runtime_predictions(state, model, cal, q2).items():
                s1_save_estimator(bucket, states, model, index, name, prediction,
                                  'runtime-legal measured inputs with frozen calibration')
    s1_block(report, maps, states, model, 'AE2b_Q2', q2_block, ('Q2', 'Q2-clip'))
    def q3_block(bucket):
        q3 = s1_fit_q3(states, cal, args)
        bucket['calibration']['Q3'] = q3['record']
        for index, state in states.items():
            for name, prediction in s1_runtime_predictions(state, model, cal, q2, q3).items():
                if name.startswith('Q3'):
                    s1_save_estimator(bucket, states, model, index, name, prediction,
                                      'runtime-legal measured inputs with frozen calibration')
    s1_block(report, maps, states, model, 'AE2b_Q3', q3_block, ('Q3', 'Q3-clip'))
    e4_names = ('AE3e-joint', 'AE3e-joint-clip', 'AE3e-sep',
                *(f'AE4e-{route}-{variant}' for route in ('joint', 'sep')
                  for variant in ('true', 'P0', 'mid', 'pass2')))
    for basis, names in (('E4', e4_names), ('E6', ('AE3e-joint-E6', 'AE3e-sep-E6'))):
        log('S1 fitting and scoring '+basis+'; '+DEVELOPMENT)
        s1_block(report, maps, states, model, 'surface_'+basis,
            lambda bucket, basis=basis: s1_surface_block(bucket, states, model, cal, args, basis), names)
    for name, operation in (
            ('SEP_joint', lambda bucket: s1_sep_joint(bucket, states, model)),
            ('SEP_shape', lambda bucket: s1_sep_shape(bucket, states, cal)),
            ('SEP_linearised_sensitivity', lambda bucket: s1_sep_cJ(bucket, states, maps)),
            ('SEP_additive', lambda bucket: s1_sep_additive(bucket, states, cal))):
        log('S1 '+name+'; descriptive, not calibration and not an estimator')
        s1_block(report, maps, states, model, name, operation)
    s1_pooled(report, maps, states, model, args)
    failures = sum(record['cells']['overall']['states']-record['cells']['overall']['valid_states']
        for shard in report['results'].values() for record in shard['S1']['estimators'].values())
    failed_block = any(block['status'] != 'complete' for block in report['S1_blocks'].values())
    report.update(status='NUMERICAL_FAIL' if failed_block else
        'completed_with_failures' if failures else 'complete', exit_code=3 if failed_block else 0,
        S1_estimator_failed_state_routes=failures)


def compute_run(args, tag, log, loader=five.load_shard, synthetic=False):
    report, maps = new_report(args, tag, synthetic), {}
    if args.stages == 'S1':
        s1_metadata(report)
    states = prepare_all(report, log, loader)
    if states is None:
        return report, maps
    base = states[0]
    maps.update(global_state_id=base['state_ids'], w_deg=base['w'],
                accommodation_d=np.array([states[i]['A'] for i in INDICES]))
    maps.update({'group_'+name: mask for name, mask in base['groups'].items()})
    try:
        shared_preparation(states, report, stage_s1=args.stages == 'S1')
    except Exception as exc:
        report.update(status='stopped_preparation_numerical_failure', exit_code=3)
        report['errors'].append(dict(exception=type(exc).__name__, message=str(exc),
                                    numbers=getattr(exc, 'numbers', {})))
        return report, maps
    try:
        analyse(states, report, args, log, maps)
    except Exception as exc:
        report.update(status='NUMERICAL_FAIL', exit_code=3,
                      incomplete_stage_results=report['results'], results={}, ladder=[], pooled={})
        report['errors'].append(dict(exception=type(exc).__name__, message=str(exc),
                                    numbers=getattr(exc, 'numbers', {})))
        maps = {name: value for name, value in maps.items() if not name.startswith('A')}
    if args.stages == 'S1' and report['exit_code'] == 0:
        try:
            analyse_s1(states, report, args, log, maps)
        except Exception as exc:
            # S1 never sends a new-block exception through the legacy discard-all handler.
            report.update(status='NUMERICAL_FAIL', exit_code=3)
            report['errors'].append(dict(block='S1_assembly', exception=type(exc).__name__,
                message=str(exc), numbers=getattr(exc, 'numbers', {})))
    return report, maps


def synthetic_loader(fail_index=None):
    """Published loader contract on a synthetic full grid; no input file reads."""
    ga, gb = np.indices((101, 101)).reshape(2, -1)
    w = np.column_stack((np.linspace(-12.5, 12.5, 101)[gb],
                         -np.linspace(-12.5, 12.5, 101)[ga]))
    zero = (ga == 50) & (gb == 50)
    groups = dict(train=(ga % 2 == 0) & (gb % 2 == 0) & ~zero,
        even_odd=(ga % 2 == 0) & (gb % 2 == 1), odd_even=(ga % 2 == 1) & (gb % 2 == 0),
        odd_odd=(ga % 2 == 1) & (gb % 2 == 1), zero=zero, overall=np.ones(len(w), bool))
    for a, b in ((0, 0), (0, 100), (100, 0), (100, 100)):
        groups[f'extreme_{a}_{b}'] = (ga == a) & (gb == b)
    def load(args, entry, metadata, log):
        index = args.shard_index
        if index == fail_index:
            raise ValueError('Injected unavailable shard')
        a, H, q = index/5., 2.6*(1-.02*index/5.), -.03+.001*index/5.
        d = -five.LABELS
        z = H*five.joint.radial(d, q)[None, :, :]+(-.5*(1-.01*a)*w)[:, None, :]
        observed = five.joint.camera(z)
        entry.update(input=dict(manifest_entry=dict(accommodation_d=a)), status='validated')
        metadata.update(global_state_id=np.arange(len(w)), global_ade_index=ga,
                        global_bde_index=gb, w_deg=w, input_failed=np.zeros(len(w), bool))
        return observed, np.zeros(2), H, q, groups.copy()
    return load


def synthetic_s1_loader():
    """Separate exactly-linear zero-gaze calibration fixture; old fixture unchanged."""
    original = synthetic_loader()
    def load(args, entry, metadata, log):
        _, c, _, _, groups = original(args, entry, metadata, log)
        a, q = args.shard_index/5., -.03
        H = 2.6*(1-.02*a)
        w = metadata['w_deg']
        tau = tw.tangent_coordinates(w)['primary']
        rho2 = np.sum(tau*tau, axis=1)
        factor = 1+(.8+.15*a)*rho2
        z = factor[:, None, None]*H*five.joint.radial(-five.LABELS, q)[None]
        z += (-.5*(1-.01*a)*w)[:, None, :]
        return five.joint.camera(z), c, H, q, groups
    return load


def selftests(args, log):
    report = new_report(args, 'selftest', synthetic=True)
    checks = []
    def check(name, operation):
        try:
            operation()
            checks.append(dict(name=name, passed=True))
            log('selftest '+name+' complete')
        except Exception as exc:
            checks.append(dict(name=name, passed=False, exception=type(exc).__name__,
                               message=str(exc), numbers=getattr(exc, 'numbers', {})))
            log('selftest '+name+' FAILED: '+str(exc))
    def close(left, right, tolerance, message):
        require(np.all(np.isfinite(left)) and np.all(np.isfinite(right)) and
                np.max(abs(np.asarray(left)-right)) <= tolerance, message)
    def inverse_check():
        answer, reason, _ = monotone_inverse([0, 2, 4], [1, .96, .92], [1, .98, .92, 1.01, np.nan])
        close(answer[:3], [0, 1, 4], 1e-13, 'Inverse interpolation/endpoints')
        require(np.array_equal(reason, [0, 0, 0, 2, 1]) and np.isnan(answer[3:]).all(), 'Range/nonfinite policy')
        try:
            monotone_inverse([0, 2, 4], [1, .9, 1], [1])
        except NumericalFailure:
            pass
        else:
            raise ValueError('Nonmonotone primary calibration accepted')
        _, reason, record = monotone_inverse([0, 2, 4], [0, 0, 0], [0], required=False)
        require(record['status'] == 'diagnostic_unavailable' and reason[0] == 3, 'Secondary unavailable')
    check('S1_monotone_inversion_and_range_failures', inverse_check)
    def projection_check():
        d = -five.LABELS[gp.estimator_nodes()['E2'][1]]
        t = np.sum(d*d, axis=1)
        mu, beta, H0, q0 = .96, .003, 2.6, -.03
        shape = (H0*(1+q0*t)*mu*(1+beta*(t-1)))[:, None]*d
        recovered_mu, recovered_beta, residual, _ = two_radius_projection(shape[None], H0, q0)
        close(recovered_mu, [mu], 1e-13, 'Mu recovery')
        close(recovered_beta, [beta], 1e-13, 'Beta recovery')
        close(residual, [0], 1e-9, 'Radial projection closure')
    check('S2_two_radius_radial_recovery', projection_check)
    def linear_check():
        for matrix in (np.ones((10, 2)), np.column_stack((np.ones(10), 1+1e-9*np.arange(10)))):
            try:
                linear_fit(matrix, np.arange(10), 'Expected bad design')
            except NumericalFailure:
                continue
            raise ValueError('Rank/condition failure accepted')
    check('S3_rank_and_unchanged_condition_gate', linear_check)
    def derivative_check():
        coefficient = np.zeros((3, len(tw.TERMS3), 2))
        coefficient[0, tw.TERMS3.index((1, 0)), 0] = 2
        coefficient[1, tw.TERMS3.index((1, 0)), 0] = 3
        coefficient[2, tw.TERMS3.index((0, 1)), 1] = 4
        model = dict(coefficient=coefficient, scale=np.ones(2), bounds=(0, 4))
        m = np.array([[2., 3.], [1., 4.]])
        close(evaluate_gaze(model, m, [1, 2]), [[10, 12], [8, 64]], 1e-13, 'Polynomial evaluation')
        close(evaluate_gaze(model, m, [1, 2], derivative=True), [[6, 24], [3, 64]], 1e-13,
              'Analytic accommodation derivative')
        require(np.isnan(evaluate_gaze(model, m, [-1, 5])).all(), 'Gaze extrapolation allowed')
    check('S4_gaze_trend_analytic_derivative_and_range', derivative_check)
    def parity_check():
        m = np.array([[1., 2.], [-2., 3.], [.5, -.2]])
        sigma = np.array([-.2, .1, .4])
        for degree in (1, 2):
            left, _ = empirical_design(m, sigma, np.ones(2), degree)
            right, _ = empirical_design(-m, sigma, np.ones(2), degree)
            close(left, right, 0, 'Empirical even parity')
        metric = scalar_metric(np.array([0., np.nan]), np.ones(2, bool))
        require(metric['states'] == 2 and metric['nonfinite_states'] == 1 and
                np.isinf(metric['rms_D']) and np.isinf(metric['max_abs_D']), 'Failure denominator')
        failures = reason_record(np.array([0, 2], dtype=np.int8),
            dict(state_ids=np.array([3, 7]), w=np.array([[0., 0.], [1., 2.]])),
            np.array([1., 1.1]), 'direct noniterative test inverse')
        require(failures['first_failed_states'][0]['state_id'] == 7 and
                failures['first_failed_states'][0]['iterations'] is None and
                failures['first_failed_states'][0]['residual'] is None and
                failures['failure_examples_omitted'] == 0, 'Failure detail/iteration n-a record')
    check('S5_empirical_parity_and_failure_denominators', parity_check)
    fixture = {}
    def pipeline_check():
        production, maps = compute_run(args, 'selftest', log, synthetic_loader(), synthetic=True)
        fixture.update(report=production, maps=maps)
        require(production['exit_code'] == 0 and len(production['results']) == 21, 'All21 pipeline incomplete')
        require(production['pooled']['heldout_A']['accommodation_count'] == 21-len(args.anchors),
                'Held-out accommodation denominator')
        require(production['calibration']['empirical_A']['anchor_indices'] == args.anchors and
                production['calibration']['gaze_coefficient_trend']['anchor_indices'] == args.anchors,
                'Anchor calibration provenance')
        require(production['preparation']['A20']['actual_polynomial']['rank'] == 10,
                'Own-A cubic full rank')
        require(next(iter(production)) == LABEL, 'First JSON label key')
        require(all(np.asarray(array).dtype.kind in 'biufc' for array in maps.values()), 'Non-numeric map')
    check('S6_all21_production_pipeline', pipeline_check)
    def data_failure_check():
        failed, maps = compute_run(args, 'selftest', log, synthetic_loader(fail_index=2), synthetic=True)
        require(failed['status'] == 'stopped_missing_or_invalid_data' and failed['exit_code'] == 2
                and not failed['results'] and not failed['ladder'] and not maps
                and any(error['shard'] == 'A02' for error in failed['errors']), 'Data failure did not stop run')
    check('S7_missing_shard_stops_before_rungs', data_failure_check)
    def isolation_check():
        # Exercise actual calibration functions, poisoning every held-out A.
        # Oracle/readout data may change; runtime calibrators must remain equal.
        groups = {'train': np.array([True, True, True, True, True, True])}
        m = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.], [2., 1.], [1., 2.]])
        states = {}
        for index in INDICES:
            a = index/5
            coef = np.ones((len(tw.TERMS3), 2))*(1+.1*a+.01*a*a)
            states[index] = dict(A=a, scale=np.ones(2), fit=dict(coef=coef), groups=groups,
                m=m, Sigma=np.full(6, 1-.02*a)+.001*np.sum(m*m, axis=1))
        cal = dict(Sigma0=1.)
        first = coefficient_model(states, args)
        bar = fit_empirical(states, cal, args)
        for index in INDICES:
            if index not in args.anchors:
                states[index].update(A=999., m=m*999, Sigma=np.full(6, np.nan),
                                     fit=dict(coef=np.full((len(tw.TERMS3), 2), np.nan)))
        second = coefficient_model(states, args)
        other = fit_empirical(states, cal, args)
        close(first['coefficient'], second['coefficient'], 0, 'Held-out A leaked into gaze trend')
        close(bar['coefficient'], other['coefficient'], 0, 'Held-out A leaked into Q')
    check('S8_heldout_A_never_calibrates_trend_or_Q', isolation_check)
    def preparation_failure_check():
        base_loader = synthetic_loader()
        def invalid_camera(args, entry, metadata, logger):
            observed, c, H, q, groups = base_loader(args, entry, metadata, logger)
            # Finite inputs pass the data adapter but cannot lie on the fixed
            # camera's origin-connected monotone inverse branch.
            if args.shard_index == 20:
                observed = observed*1e6
            return observed, c, H, q, groups
        failed, maps = compute_run(args, 'selftest', log, invalid_camera, synthetic=True)
        require(failed['status'] == 'stopped_preparation_numerical_failure' and failed['exit_code'] == 3
                and not failed['results'] and not failed['ladder'] and
                not any(name.startswith('A') for name in maps), 'Preparation numerical stop')
    check('S9_invalid_camera_branch_stops_before_rungs', preparation_failure_check)
    report = fixture.get('report', report)
    report['selftest_checks'] = checks
    report['status'] = 'complete' if all(item['passed'] for item in checks) else 'selftest_failed'
    report['exit_code'] = 0 if report['status'] == 'complete' else 4
    report['selftest_scope'] = 'Synthetic code checks only; not scientific acceptance of any rung'
    return report, fixture.get('maps', {})


def selftests_s1(args, log):
    """S10--S17 are synthetic code checks, never scientific acceptance gates."""
    report, maps = new_report(args, 'selftest_s1', synthetic=True), {}
    s1_metadata(report)
    checks, fixture = [], {}
    def check(name, operation):
        try:
            operation()
            checks.append(dict(name=name, passed=True))
            log('selftest '+name+' complete')
        except Exception as exc:
            checks.append(dict(name=name, passed=False, exception=type(exc).__name__,
                message=str(exc), numbers=getattr(exc, 'numbers', {})))
            log('selftest '+name+' FAILED: '+str(exc))
    def close(left, right, tolerance, message):
        left, right = np.asarray(left), np.asarray(right)
        require(np.isfinite(left).all() and np.isfinite(right).all() and
                np.max(abs(left-right)) <= tolerance, message)
    def prepared():
        if fixture:
            return fixture
        scratch = new_report(args, 'selftest_s1', synthetic=True)
        states = prepare_all(scratch, log, synthetic_s1_loader())
        require(states is not None, 'S1 fixture preparation failed')
        shared_preparation(states, scratch, stage_s1=True)
        cal = size_calibration(states, args)
        for state in states.values():
            state['groups']['field_lt12'] = np.logical_or.reduce(
                [state['groups'][name] for name in gp.BIN_NAMES[:3]])
            state['mu_raw'] = two_radius_projection(state['Zshape'], cal['H0'], cal['q0'])[0]
        fixture.update(states=states, cal=cal, model=coefficient_model(states, args),
            q2=fit_empirical(states, cal, args), q3=s1_fit_q3(states, cal, args),
            E4=s1_fit_surface(states, cal, args, 'E4'), E6=s1_fit_surface(states, cal, args, 'E6'))
        return fixture
    def clip_agreement(primary, clipped, message):
        A, reason = primary[:2]
        other, other_reason = clipped[:2]
        unchanged = reason != 2
        require(np.array_equal(A[unchanged], other[unchanged], equal_nan=True) and
                np.array_equal(reason[unchanged], other_reason[unchanged]), message)
        require(np.all(other_reason[reason == 2] == 0) and
                np.all(np.isfinite(other[reason == 2])), message+' outside')
    def q3_check():
        m = np.array([[1., 2.], [-2., 3.], [.5, -.2], [0., 0.]])
        sigma = np.array([-.2, .1, .4, 0.])
        matrix = s1_q3_design(m, sigma, np.array([2., 3.]))
        require(matrix.shape == (4, 14), 'Q3 column count')
        close(matrix, s1_q3_design(-m, sigma, np.array([2., 3.])), 0, 'Q3 exact evenness')
        state = dict(m=np.zeros((4, 2)), scale=np.ones(2), Sigma=np.array([-1., 1., 6., np.nan]))
        for degree, width in ((2, 6), (3, 14)):
            coefficient = np.zeros(width)
            coefficient[0:2] = [1., 1.]
            fit = dict(coefficient=coefficient)
            primary = s1_empirical_prediction(state, dict(Sigma0=1.), fit, degree)
            clipped = s1_empirical_prediction(state, dict(Sigma0=1.), fit, degree, True)
            clip_agreement(primary, clipped, 'Q'+str(degree)+' clip changes only reason 2')
            require(set(primary[1]) == {0, 1, 2}, 'Q clip fixture must exercise all reasons')
    check('S10_Q3_basis_parity_and_named_clip_rules', q3_check)
    def endpoint_check(output, true_A, mu, endpoint, label):
        A, reason = output[:2]
        valid = reason == 0
        boundary = reason == 2
        require(np.all(valid | boundary), label+' unexpected failure')
        require(np.all(abs(A[valid]-true_A) <= 1e-5), label+' valid A error')
        require(np.all(abs(mu[boundary]-endpoint[boundary]) <= 1e-12*abs(endpoint[boundary])),
                label+' endpoint round-off exceeds test-side rule')
        log(label+f' valid={int(valid.sum())} endpoint_roundoff_reason2={int(boundary.sum())}')
        return dict(valid=int(valid.sum()), endpoint_roundoff_reason2=int(boundary.sum()))
    def surface_check():
        f = prepared()
        counts = {}
        for index, state in f['states'].items():
            output = s1_surface_prediction(state, f['E4'], f['cal'], state['w'], 'joint')
            if index in (0, 20):
                table = s1_surface_table(f['E4'], f['cal'], state['w'])
                counts[f'A{index:02d}_joint'] = endpoint_check(output, state['A'], state['mu_raw'],
                    table[:, 0 if index == 0 else -1], f'S11 A{index:02d} joint')
            else:
                require(np.all(output[1] == 0), f'S11 interior A{index:02d} failure')
                close(output[0], state['A'], 1e-5, f'S11 A{index:02d} exact joint recovery')
        zero, end = f['states'][0], f['states'][20]
        sep0 = s1_surface_prediction(zero, f['E4'], f['cal'], zero['w'], 'sep')
        factor = 1+s1_size_design(tw.tangent_coordinates(zero['w'])['primary'], 'E4')@f['E4']['gamma_A00']
        counts['A00_sep'] = endpoint_check(sep0, 0., zero['mu_raw'], factor, 'S11 A00 sep')
        sep4 = s1_surface_prediction(end, f['E4'], f['cal'], end['w'], 'sep')
        require(sep4[1][0] == 0 and abs(sep4[0][0]-4.) > 1e-3, 'S11 A20 corner separable bias absent')
        report['selftest_endpoint_counts'] = counts
    check('S11_separate_linear_fixture_joint_recovery_and_sep_bias', surface_check)
    def inversion_check():
        grid = np.array([0., 2., 4.])
        table = np.array([[1., .9, 1.], [1., .96, .92], [1., .96, .92], [1., np.nan, .92]])
        target = np.array([.95, 1.01, .98, 1.])
        primary = s1_table_inverse(grid, table, target)
        clipped = s1_table_inverse(grid, table, target, True)
        require(np.array_equal(primary[1], [4, 2, 0, 1]) and
                np.isnan(primary[0][[0, 1, 3]]).all() and primary[2][0] == 2,
                'S12 strict inverse reason/sign-change policy')
        clip_agreement(primary, clipped, 'Joint clip changes only reason 2')
        metric = scalar_metric(primary[0], np.ones(4, bool))
        require(metric['states'] == 4 and metric['nonfinite_states'] == 3 and
                np.isinf(metric['rms_D']) and np.isinf(metric['max_abs_D']), 'S12 failed-state denominator')
    check('S12_nonmonotone_range_and_clip_inversion_rules', inversion_check)
    def sensitivity_check():
        saved = {}
        for route, slope in (('joint', .25), ('sep', -.5)):
            def operation(assumed):
                assumed = np.broadcast_to(np.asarray(assumed, float), (3,))
                output = 1+slope*assumed
                reason = np.where(np.isfinite(assumed), 0, 1).astype(np.int8)
                return output, reason
            for a in (0., 2., 4.):
                true, lower, upper, zero = operation(a), operation(a-.2), operation(a+.2), operation(0.)
                c, c0, convention = s1_local_sensitivity(true, lower, upper, zero, a, .2)
                close(c, slope, 1e-12, 'S13 '+route+' '+convention)
                if a:
                    close(c0, slope, 1e-12, 'S13 '+route+' secant')
                else:
                    require(np.isnan(c0).all(), 'S13 undefined A00 secant')
                saved[f'A{int(a*5):02d}_AE4e_{route}_c'] = c
                saved[f'A{int(a*5):02d}_AE4e_{route}_c0'] = c0
            true, lower, upper, zero = operation(2.), operation(1.8), operation(2.2), operation(0.)
            upper[1][0] = 2
            c, _, _ = s1_local_sensitivity(true, lower, upper, zero, 2., .2)
            require(np.isnan(c[0]) and scalar_metric(c, np.ones(3, bool))['nonfinite_states'] == 1,
                    'S13 failed sensitivity value not counted')
            inner = (np.array([np.nan, 1., 2.]), np.array([4, 0, 0], np.int8))
            second = s1_pass2(inner, operation)
            require(second[1][0] == 4 and np.isnan(second[0][0]), 'S13 pass2 inner reason lost')
            close(second[0][1:], 1+slope*np.array([1., 2.]), 1e-12, 'S13 pass2 composition')
        require(all(('_joint_' in key or '_sep_' in key) for key in saved), 'S13 route-qualified map names')
        maps.update(saved)
        report['selftest_sensitivity_map_keys'] = sorted(saved)
    check('S13_route_qualified_central_one_sided_secant_and_pass2', sensitivity_check)
    def sep_check():
        f = prepared()
        template = f['states']
        J = np.zeros((16, 3))
        J[0, 0], J[1, 1], J[0, 2], J[2, 2] = 2., 3., 1., 4.
        synthetic = {}
        for index, state in template.items():
            w = state['w']
            value = w[:, 0, None]*60/1000*J[:, 0]+w[:, 1, None]*60/1000*J[:, 1]
            value += state['A']/1000*J[:, 2]
            synthetic[index] = dict(state_ids=state['state_ids'], w=w, A=state['A'],
                                    linear=value.reshape(-1, 8, 2))
        dx, dy, da = s1_data_derivatives(synthetic, 'linear')
        found = np.stack((dx.reshape(21, 10201, 16)*1000/60,
            dy.reshape(21, 10201, 16)*1000/60, da.reshape(21, 10201, 16)*1000), axis=-1)
        close(found, J[None, None], 1e-9, 'S14 finite-difference Jacobian and ADE sign')
        measured = s1_jacobian_measures(found[10])
        correlation = 1/np.sqrt(17.)
        known_condition = np.sqrt((1+correlation)/(1-correlation))
        close(measured['condition'], known_condition, 1e-9, 'S14 scaled condition')
        close(measured['sin_theta_A'], 4/np.sqrt(17.), 1e-9, 'S14 angle')
        close(measured['leverage_um_per_D'], 4., 1e-9, 'S14 leverage')
        r, c = np.array([-.2, .1, .4]), np.array([-.3, -.1, .2, .5])
        separable = np.exp(r[:, None]+c[None])
        close(s1_additive_remainder(separable), 0., 1e-12, 'S14 zero additive remainder')
        interaction = .02*np.array([-1., 0., 1.])[:, None]*np.array([-3., -1., 1., 3.])[None]
        variant = np.exp(r[:, None]+c[None]+interaction)
        close(s1_additive_remainder(variant), interaction, 1e-12, 'S14 known additive interaction')
        undefined = s1_jacobian_measures(np.zeros((2, 16, 3)))
        require(np.isnan(undefined['condition']).all() and np.isnan(undefined['leverage_um_per_D']).all(),
                'S14 zero-column undefined state')
    check('S14_SEP_linear_observable_geometry_and_additive_remainder', sep_check)
    def leakage_check():
        f = prepared()
        poisoned = {}
        for index, state in f['states'].items():
            other = state.copy()
            if index not in args.anchors:
                other.update(A=999., w=np.full_like(state['w'], np.nan),
                    m=np.full_like(state['m'], np.nan), Sigma=np.full_like(state['Sigma'], np.nan),
                    mu_raw=np.full_like(state['mu_raw'], np.nan))
            poisoned[index] = other
        q3 = s1_fit_q3(poisoned, f['cal'], args)
        require(np.array_equal(q3['coefficient'], f['q3']['coefficient']), 'S15 held-out Q3 leakage')
        for basis in ('E4', 'E6'):
            other = s1_fit_surface(poisoned, f['cal'], args, basis)
            for name in ('gamma_anchors', 'gamma_trend'):
                require(np.array_equal(other['record'][name], f[basis]['record'][name]),
                        'S15 held-out '+basis+' '+name+' leakage')
        score_changes = 0
        for index, state in f['states'].items():
            runtime = s1_runtime_predictions(state, f['model'], f['cal'], f['q2'], f['q3'], f['E4'])
            changed = dict(state, A=state['A']+7., w=state['w']+.125, oracle=state['oracle']-.25)
            repeated = s1_runtime_predictions(changed, f['model'], f['cal'], f['q2'], f['q3'], f['E4'])
            require(runtime.keys() == repeated.keys(), 'S15 runtime row names changed')
            for name, output in runtime.items():
                require(all(np.array_equal(left, right, equal_nan=True)
                    for left, right in zip(output, repeated[name])), 'S15 truth leaked into '+name)
            # Score change is checked on finite, in-range runtime estimates.
            output = runtime['Q2-clip']
            before = s1_scores(*output, state, 'test')[0]['cells']['overall']
            after = s1_scores(*output, changed, 'test')[0]['cells']['overall']
            score_changes += before['A_error_D']['bias_D'] != after['A_error_D']['bias_D']
        require(score_changes > 0, 'S15 replacing truth failed to change scores')
    check('S15_anchor_isolation_and_frozen_runtime_calibration', leakage_check)
    def regression_check():
        legacy, legacy_maps = selftests(arguments([]), log)
        require(legacy['exit_code'] == 0 and len(legacy['selftest_checks']) == 9 and
                all(item['passed'] for item in legacy['selftest_checks']), 'S16 unchanged S1-S9 failed')
        require(set(legacy['results']) == {f'A{i:02d}' for i in INDICES} and
                all(set(value) == {'accommodation_d', 'A_role', 'AE0', 'AE1', 'AE2'}
                    for value in legacy['results'].values()) and
                all(not name.endswith('_s1') for name in artifacts('selftest')) and
                'S1_blocks' not in legacy and not any('_SEP_' in name or '_Q3_' in name for name in legacy_maps),
                'S16 default result keys changed')
        report['default_stage_regression_checks'] = legacy['selftest_checks']
    check('S16_unchanged_default_fixture_checks_and_result_keys', regression_check)
    def configuration_check():
        # Parser rejection is tested directly: no output object/path is created.
        import contextlib
        import io
        for changed in (['--anchors', '0', '4', '10', '15', '20'],
                        ['--a-poly-degree', '1'], ['--empirical-degree', '1'],
                        ['--gaze-readouts', '1'], ['--gaze-readouts', '.1', '1']):
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    arguments(['--stages', 'S1', *changed])
            except SystemExit as exc:
                require(exc.code == 2, 'S17 wrong parser exit')
            else:
                raise ValueError('S17 S1 override accepted: '+str(changed))
        fixed = arguments(['--stages', 'S1'])
        require(fixed.anchors == [0, 5, 10, 15, 20] and fixed.size_basis == 'E4' and
                fixed.a_grid_step == .01 and fixed.sensitivity_step == .2, 'S17 fixed configuration')
    check('S17_fixed_S1_configuration_rejected_before_output', configuration_check)
    report['selftest_checks'] = checks
    report['status'] = 'complete' if all(item['passed'] for item in checks) else 'selftest_failed'
    report['exit_code'] = 0 if report['status'] == 'complete' else 4
    report['selftest_scope'] = 'Synthetic code checks only; development, not scientific acceptance'
    return report, maps


def ladder_text(report):
    lines = [LABEL, PROVISIONAL, 'AE0--AE2; all errors use every state in their group',
             'status='+report['status']+' exit_code='+str(report['exit_code']),
             'A role | oracle gaze RMS/max arcmin | P0 gaze RMS/max arcmin | '
             'interpolated gaze RMS/max arcmin | raw A RMS/max D | Q A RMS/max D | '
             'raw gaze RMS/max arcmin | Q gaze RMS/max arcmin | raw/Q failed states']
    def pair(value, unit, vector=False):
        value = value['vector'] if vector else value
        return '/'.join(format(value[name+'_'+unit], '.9g') for name in
                        (('rms', 'max') if vector else ('rms', 'max_abs')))
    for row in report['ladder']:
        values = [row['shard']+' '+row['A_role'],
            *(pair(row[name], 'arcmin', True) for name in ('oracle_gaze', 'P0_gaze', 'interpolated_gaze')),
            *(pair(row[name], 'D') for name in ('raw_A', 'empirical_A')),
            *(pair(row[name], 'arcmin', True) for name in ('raw_gaze', 'empirical_gaze')),
            str(row['raw_failed_states'])+'/'+str(row['empirical_failed_states'])]
        lines.append(' | '.join(values))
    legacy = '\n'.join(lines)+'\n'
    return s1_ladder_text(report, legacy) if report['configuration']['stages'] == 'S1' else legacy


def s1_ladder_text(report, legacy):
    legacy = legacy.replace(LABEL+'\n', LABEL+'\n'+DEVELOPMENT+'\n', 1)
    lines = [legacy.rstrip(), '', DEVELOPMENT,
        'S1 estimators; full field; every state in denominator; carried AE0--AE2 context above',
        'shard | estimator | A RMS/max D | gaze RMS/max arcmin | coverage | failures by reason | input label']
    def number(value):
        return 'n/a' if value is None else format(value, '.9g')
    def estimator_line(shard, name, cell, input_label):
        a, g = cell['A_error_D'], cell['gaze_vs_truth_arcmin']
        return ' | '.join((shard, name,
            number(a['rms_D'])+'/'+number(a['max_abs_D']),
            number(g['rms_arcmin'])+'/'+number(g['max_abs_arcmin']), number(cell['coverage']),
            json.dumps(cell['failures_by_reason'], sort_keys=True), input_label, DEVELOPMENT))
    for key, shard in report['results'].items():
        for name, record in shard.get('S1', {}).get('estimators', {}).items():
            lines.append(estimator_line(key, name, record['cells']['overall'], record['input_label']))
    lines.extend(('', DEVELOPMENT, 'S1 per-route local sensitivities; '+LOCAL_SENSITIVITY,
        'shard | route | max abs c | argmax c | max abs c0 | argmax c0 | count abs c >= 1'))
    for key, shard in report['results'].items():
        for route in ('joint', 'sep'):
            record = shard.get('S1', {}).get('sensitivity', {}).get(route)
            if record is None:
                if 'S1_blocks' in report:
                    lines.append(key+' | '+route+' | unavailable | '+LOCAL_SENSITIVITY+' | '+DEVELOPMENT)
                continue
            c = record['c']['cells']['overall']['maximum_abs']
            c0 = record['c0']['cells']['overall']['maximum_abs']
            lines.append(' | '.join((key, route, number(c['value']),
                json.dumps(data_json.safe(c['location'])), number(c0['value']),
                json.dumps(data_json.safe(c0['location'])), str(record['count_abs_c_ge1']['overall']),
                record['difference_convention'], LOCAL_SENSITIVITY, DEVELOPMENT)))
    lines.extend(('', DEVELOPMENT,
        'Pooled crossed cells: heldout_A odd_odd full field headline; field_lt12 companion'))
    for role, records in report.get('S1_pooled', {}).items():
        for name, record in records.items():
            for cell in ('odd_odd', 'odd_odd/field_lt12'):
                lines.append(estimator_line(role+'/'+cell, name, record['cells'][cell], record['input_label']))
    lines.extend(('', DEVELOPMENT,
        'SEP data only, descriptive; not calibration and not an estimator',
        'shard | median valid states only/worst scaled condition | minimum sin theta_A | minimum leverage um/D | '
        'maximum additive remainder D | undefined counts'))
    for key, shard in report['results'].items():
        sep = shard.get('S1', {}).get('SEP', {})
        joint, additive = sep.get('joint', {}), sep.get('additive_remainder', {})
        if not joint or not additive:
            lines.append(key+' | SEP item unavailable; see errors | '+DEVELOPMENT)
            continue
        condition = joint['condition']['cells']['overall']
        angle = joint['sin_theta_A']['cells']['overall']
        leverage = joint['leverage_um_per_D']['cells']['overall']
        remainder = additive['equivalent_D']['cells']['overall']
        lines.append(' | '.join((key,
            number(condition['median_abs_valid_states_only'])+'/'+number(condition['maximum_abs']['value']),
            number(angle['minimum_abs']['value']), number(leverage['minimum_abs']['value']),
            number(remainder['maximum_abs']['value']),
            json.dumps({name: cell['metrics']['nonfinite_states'] for name, cell in
                (('condition', condition), ('sin_theta_A', angle), ('leverage', leverage), ('additive', remainder))}),
            DEVELOPMENT)))
    if 'zero_gaze_adjacent_slopes' in report['calibration']:
        lines.extend(('', DEVELOPMENT, 'Zero-gaze adjacent slopes (interval midpoint D, mu prime, mu prime/mu, H prime/H)',
            json.dumps(data_json.safe(report['calibration']['zero_gaze_adjacent_slopes'])),
            'AE3e derivative convention: '+SURFACE_DERIVATIVE))
    if report.get('selftest_endpoint_counts'):
        lines.append('S11 test-side endpoint counts: '+json.dumps(report['selftest_endpoint_counts']))
    return '\n'.join(lines)+'\n'


def write_outputs(tag, report, maps, log):
    for key, value in maps.items():
        require('/' not in key and '\\' not in key and np.asarray(value).dtype.kind in 'biufc',
                'Unsafe/non-numeric NPZ key: '+key)
    safe = data_json.safe(report)
    encoded = json.dumps(safe, indent=2, allow_nan=False)+'\n'
    guard_outputs(tag)
    names = artifacts(tag)
    (OUT/names[0]).write_text(ladder_text(report), encoding='utf-8')
    guard_outputs(tag)
    stage_label = DEVELOPMENT+'\n' if report['configuration']['stages'] == 'S1' else ''
    (OUT/names[1]).write_text(LABEL+'\n'+PROVISIONAL+'\n'+stage_label+'\n'+encoded, encoding='utf-8')
    guard_outputs(tag)
    (OUT/names[2]).write_text(encoded, encoding='utf-8')
    guard_outputs(tag)
    np.savez_compressed(OUT/names[3], **maps)
    log('final status='+report['status']+' saved exit_code='+str(report['exit_code']))


def s2_artifacts(tag):
    require(tag in S2_TAGS, 'Unknown S2 output tag')
    return tuple(name.replace('_selftest.', '_'+tag+'.') for name in artifacts('selftest'))


def s2_guard_outputs(tag):
    """Reuse legacy protections read-only; extend them to the separate S2 targets."""
    guard_outputs('selftest_s1')
    targets = [OUT/name for name in s2_artifacts(tag)]
    for path in [OUT, *OUT.parents, *targets]:
        require(path.resolve() == path.absolute(), 'Redirected S2 output: '+str(path))
        if path.exists() or path.is_symlink():
            info = path.lstat()
            require(not path.is_symlink() and not
                (getattr(info, 'st_file_attributes', 0) & 0x400), 'S2 output reparse point: '+str(path))
            if path in targets:
                require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
                    'S2 output must be a regular file with one link: '+str(path))
            else:
                require(path.is_dir(), 'S2 output parent is not a directory: '+str(path))
    sources = [five.MANIFEST, *(gp.accommodation_input(i) for i in INDICES),
        ROOT/'CURRENT_PLAN.md', ROOT/'P4_MODEL.md', ROOT/'P4_SPEC.md',
        *list((ROOT/'Script').glob('*.py')),
        *(OUT/name for other in (*TAGS, *S1_TAGS) for name in artifacts(other)),
        *(OUT/name for other in S2_TAGS if other != tag for name in s2_artifacts(other))]
    for i, target in enumerate(targets):
        require(target.parent == OUT, 'S2 output outside runs/p4_accom')
        for source in [*sources, *targets[i+1:]]:
            require(target.resolve() != source.resolve() and not
                (target.exists() and source.exists() and target.samefile(source)),
                'S2 output aliases an input/source/target')


def s2_progress(tag):
    s2_guard_outputs(tag)
    OUT.mkdir(parents=True, exist_ok=True)
    s2_guard_outputs(tag)
    (OUT/s2_artifacts(tag)[-1]).write_text(LABEL+'\n'+DEVELOPMENT+'\n'+S2_LOOP_LABEL+'\n',
                                          encoding='utf-8')
    started, last = time.perf_counter(), [time.perf_counter()]
    def log(message, heartbeat=False):
        now = time.perf_counter()
        if heartbeat and now-last[0] < 30:
            return
        last[0] = now
        line = datetime.now(timezone.utc).isoformat(timespec='seconds')
        line += f' elapsed_s={now-started:.3f} '+message
        print(line, flush=True)
        s2_guard_outputs(tag)
        with (OUT/s2_artifacts(tag)[-1]).open('a', encoding='utf-8') as stream:
            stream.write(line+'\n')
            stream.flush()
    return log


def s2_metadata(report):
    s1_metadata(report)
    report['S2_label'] = S2_LOOP_LABEL
    report['decisions'].update(M1='Joint E4 primary; strict sep diagnostic baseline',
        M2='Strict primary; named joint -clip and -ext02; unclipped domain A01-A19',
        M3='Full field A01-A19 headline, worst case at full coverage; below-12 and endpoints beside',
        M4='S2 = AE5e + AE6-lite', M5='Fresh traces approved in principle only; none authorised',
        M6='Noise study deferred; no noise in S2', M7='E4 fixed; no basis selection')
    report['definitions'].update(headline_cell='domain_A01_A19 / overall; literal corners included',
        companion_cell='domain_A01_A19 / field_lt12', S2= S2_LOOP_LABEL,
        failed_pass='1-based map pass that failed; 0 means no map-pass failure, including reason 6',
        max_pass_failure='reason 6: estimate NaN, passes=20; last iterate and step diagnostic only',
        fixed_point_residual='One extra diagnostic map evaluation for stopped states; never a pass/update',
        profile_roots='Strict adjacent sign changes plus exact node zeros; holes counted separately',
        AE6=S2_AE6_LABEL, AE6_dioptre=S2_DIOPTRE_LABEL,
        S2_pool_note='domain_A01_A19 contains three anchors and all gaze splits; partly in-sample. '
            'heldout_A/odd_odd is out-of-calibration. Both are development.',
        range_policy='Strict [0,4] primary; clipping only -clip; bounded extrapolation only -ext02')
    report['caveats'].extend((S2_LOOP_LABEL, S2_AE6_LABEL, S2_DIOPTRE_LABEL,
                              report['definitions']['S2_pool_note']))


def s2_new_report(args, tag, synthetic=False):
    report = new_report(args, tag, synthetic)
    s2_metadata(report)
    return report


def s2_cells(groups):
    return dict(s1_cells(groups), four_corner_extremes=groups['four_corner_extremes'])


def s2_scores(predicted_A, reason, gaze, state, label, cell_names=None):
    """S1 score definitions, with reason 6 and the literal-corner cell added."""
    reason = np.asarray(reason, np.int8).copy()
    valid = (reason == 0) & np.isfinite(predicted_A) & np.isfinite(gaze).all(axis=1)
    reason[(reason == 0) & ~valid] = 1
    predicted_A = np.where(reason == 0, predicted_A, np.nan)
    gaze = np.where((reason == 0)[:, None], gaze, np.nan)
    A_error = predicted_A-state['A']
    gaze_error = 60*np.linalg.norm(gaze-state['w'], axis=1)
    oracle_error = 60*np.linalg.norm(gaze-state['oracle'], axis=1)
    cells = {}
    for name, mask in s2_cells(state['groups']).items():
        if cell_names is not None and name not in cell_names:
            continue
        count, good = int(mask.sum()), int(np.sum(mask & (reason == 0)))
        cells[name] = dict(development_label=DEVELOPMENT, loop_profile_label=S2_LOOP_LABEL,
            states=count, valid_states=good, coverage=None if count == 0 else good/count,
            failures_by_reason={text: int(np.sum(mask & (reason == code))) for code, text in S2_REASONS},
            A_error_D=scalar_metric(A_error, mask),
            gaze_vs_truth_arcmin=scalar_metric(gaze_error, mask, 'arcmin'),
            gaze_vs_oracle_A_arcmin=scalar_metric(oracle_error, mask, 'arcmin'),
            readout_fractions={str(level): None if count == 0 else
                float(np.sum(mask & np.isfinite(gaze_error) & (gaze_error <= level))/count)
                for level in (1., .1)},
            maximum_A_error=s1_extreme(A_error, mask, state, A_error=A_error, gaze_error=gaze_error),
            maximum_gaze_error=s1_extreme(gaze_error, mask, state, A_error=A_error, gaze_error=gaze_error),
            maximum_gaze_vs_oracle_A_error=s1_extreme(oracle_error, mask, state,
                A_error=A_error, gaze_error=gaze_error))
    return dict(development_label=DEVELOPMENT, loop_profile_label=S2_LOOP_LABEL,
                input_label=label, cells=cells), predicted_A, reason, gaze


def s2_extension(model, cal, surface):
    anchors = cal['anchors']
    a, mu = cal['A'][anchors], cal['mu'][anchors]
    slopes = np.array([(mu[1]-mu[0])/(a[1]-a[0]),
                       (mu[-1]-mu[-2])/(a[-1]-a[-2])])
    # Integer ticks give the identical S1 nodes on the common 0..4 interval.
    grid = np.arange(-20, 421, dtype=float)*.01
    extended_mu = np.interp(grid, a, mu)
    low, high = grid < a[0], grid > a[-1]
    extended_mu[low] = mu[0]+slopes[0]*(grid[low]-a[0])
    extended_mu[high] = mu[-1]+slopes[1]*(grid[high]-a[-1])
    model_ext = dict(model, bounds=(-S2_EXTENSION_MARGIN, 4+S2_EXTENSION_MARGIN))
    return dict(model=model_ext, grid=grid, mu=extended_mu, endpoint_slopes=slopes,
                surface=surface, extrapolation_label='bounded quadratic/linear extrapolation, -ext02 only')


def s2_extended_table(extension, gaze):
    grid, surface = extension['grid'], extension['surface']
    gamma = np.column_stack((np.ones(len(grid)), grid, grid*grid))@surface['trend']
    tau = tw.tangent_coordinates(gaze)['primary']
    return (1+s1_size_design(tau, 'E4')@gamma.T)*extension['mu'][None, :]


def s2_map(state, model, cal, surface, route, assumed, indices=None, extension=None):
    """Measured m/mu_raw plus frozen calibration only; no scoring truth is read."""
    m = state['m'] if indices is None else state['m'][indices]
    mu = state['mu_raw'] if indices is None else state['mu_raw'][indices]
    runtime = dict(mu_raw=mu)
    if route == 'ext02':
        gaze = evaluate_gaze(extension['model'], m, assumed)
        return s1_table_inverse(extension['grid'], s2_extended_table(extension, gaze), mu)[:2]
    gaze = evaluate_gaze(model, m, assumed)
    return s1_surface_prediction(runtime, surface, cal, gaze,
                                 'sep' if route == 'sep' else 'joint', route == 'clip')[:2]


def s2_loop(operation, count, start):
    """Exactly 20 passes / 1e-6 D; stopped states get only one residual call."""
    current = np.full(count, start, float)
    estimate, step = np.full(count, np.nan), np.full(count, np.nan)
    reason, passes = np.zeros(count, np.int8), np.zeros(count, np.int16)
    failed_pass = np.zeros(count, np.int16)
    active, stopped = np.ones(count, bool), np.zeros(count, bool)
    history = np.full((S2_MAX_PASSES, count), np.nan)
    before = np.zeros((S2_MAX_PASSES, count), bool)
    for k in range(S2_MAX_PASSES):
        before[k] = active
        ids = np.flatnonzero(active)
        if len(ids):
            value, code = operation(current[ids], ids)[:2]
            value, code = np.asarray(value, float), np.asarray(code, np.int8).copy()
            code[(code == 0) & ~np.isfinite(value)] = 1
            passes[ids] = k+1
            bad = ids[code != 0]
            reason[bad], failed_pass[bad], active[bad] = code[code != 0], k+1, False
            good, next_value = ids[code == 0], value[code == 0]
            step[good] = abs(next_value-current[good])
            current[good] = next_value
            done = good[step[good] <= S2_TOLERANCE]
            estimate[done], stopped[done], active[done] = current[done], True, False
        history[k] = np.where(reason == 0, current, np.nan)
    reason[active] = 6
    # The pass-20 iterate is retained above; the failed final estimator is NaN.
    estimate[reason != 0] = np.nan
    residual, residual_reason = np.full(count, np.nan), np.full(count, 5, np.int8)
    ids = np.flatnonzero(stopped)
    if len(ids):
        value, code = operation(estimate[ids], ids)[:2]
        value, code = np.asarray(value, float), np.asarray(code, np.int8).copy()
        code[(code == 0) & ~np.isfinite(value)] = 1
        residual_reason[ids] = code
        good = code == 0
        residual[ids[good]] = abs(value[good]-estimate[ids[good]])
    return dict(estimate=estimate, reason=reason, passes=passes, failed_pass=failed_pass,
        final_step=step, last_iterate=current, stopped=stopped, history=history,
        active_before=before, residual=residual, residual_reason=residual_reason,
        start=float(start), label=S2_LOOP_LABEL)


def s2_profile(operation, count, route):
    grid = (np.arange(-4, 85, dtype=float)*S2_PROFILE_STEP if route == 'ext02' else
            np.arange(81, dtype=float)*S2_PROFILE_STEP)
    values, reasons = [], []
    ids = np.arange(count)
    for assumed in grid:
        value, reason = operation(np.full(count, assumed), ids)[:2]
        code = np.asarray(reason, np.int8).copy()
        value = np.asarray(value, float)
        code[(code == 0) & ~np.isfinite(value)] = 1
        values.append(np.where(code == 0, value, np.nan))
        reasons.append(code)
    values, reasons = np.stack(values), np.stack(reasons)
    phi = values-grid[:, None]
    valid = reasons == 0
    pairs = valid[:-1] & valid[1:]
    changes = pairs & (((phi[:-1] < 0) & (phi[1:] > 0)) |
                       ((phi[:-1] > 0) & (phi[1:] < 0)))
    exact = valid & (phi == 0)
    sign_changes, exact_zeros = changes.sum(axis=0), exact.sum(axis=0)
    roots = sign_changes+exact_zeros
    root = np.full(count, np.nan)
    ids = np.flatnonzero((sign_changes == 1) & (exact_zeros == 0))
    if len(ids):
        j = np.argmax(changes[:, ids], axis=0)
        root[ids] = grid[j]-phi[j, ids]*S2_PROFILE_STEP/(phi[j+1, ids]-phi[j, ids])
    ids = np.flatnonzero((exact_zeros == 1) & (sign_changes == 0))
    if len(ids):
        root[ids] = grid[np.argmax(exact[:, ids], axis=0)]
    slopes = np.where(pairs, abs(np.diff(values, axis=0))/S2_PROFILE_STEP, -np.inf)
    max_slope = np.max(slopes, axis=0)
    max_slope[~np.any(pairs, axis=0)] = np.nan
    holes = np.sum(~valid, axis=0)
    return dict(label=S2_LOOP_LABEL, grid=grid, map_values=values, failure_reason=reasons,
        phi=phi, holes=holes, holes_by_reason={name: np.sum(reasons == code, axis=0)
            for code, name in S2_REASONS if code}, roots=roots, sign_changes=sign_changes,
        exact_zeros=exact_zeros, root_A_D=root, max_slope=max_slope,
        root_class=np.where(roots == 1, 1, np.where(roots == 0, 0, 2)), has_holes=holes > 0,
        class_codes={0: 'no_root', 1: 'unique_root', 2: 'multiple_roots'})


def s2_valid_summary(values, mask, unit):
    finite = mask & np.isfinite(values)
    return dict(label=S2_LOOP_LABEL, full_denominator=scalar_metric(values, mask, unit),
        valid_states_only=dict(label='valid states only', metrics=scalar_metric(values, finite, unit)))


def s2_loop_record(loop, state, model):
    cells = {}
    for name, mask in s2_cells(state['groups']).items():
        used = loop['passes'][mask]
        cells[name] = dict(label=S2_LOOP_LABEL, states=int(mask.sum()),
            median_passes=None if not len(used) else float(np.median(used)),
            max_passes=None if not len(used) else int(np.max(used)),
            stopped_states=int(np.sum(mask & loop['stopped'])),
            failed_states=int(np.sum(mask & (loop['reason'] != 0))),
            map_failed_states=int(np.sum(mask & (loop['reason'] != 0) & (loop['reason'] != 6))),
            not_stopped_states=int(np.sum(mask & (loop['reason'] == 6))),
            final_step_D=s2_valid_summary(loop['final_step'], mask, 'D'),
            fixed_point_residual_D=s2_valid_summary(loop['residual'], mask, 'D'),
            residual_failed_stopped_states=int(np.sum(mask & loop['stopped'] &
                                                     (loop['residual_reason'] != 0))))
    iterates = np.vstack((np.full(len(state['m']), loop['start']), loop['history']))
    steps = abs(np.diff(iterates, axis=0))
    ratios = {}
    for k in range(5):
        valid = np.isfinite(steps[k]) & np.isfinite(steps[k+1]) & (steps[k] > 1e-9)
        value = np.full(len(state['m']), np.nan)
        value[valid] = steps[k+1, valid]/steps[k, valid]
        ratios[str(k+1)] = dict(label=S2_LOOP_LABEL, denominator_rule='step exceeds 1e-9 D',
            full_denominator=scalar_metric(value, state['groups']['overall'], 'ratio'),
            valid_states_only=dict(label='valid states only', states=int(valid.sum()),
                median=None if not np.any(valid) else float(np.median(value[valid])),
                p95=None if not np.any(valid) else float(np.percentile(value[valid], 95)),
                max=None if not np.any(valid) else float(np.max(value[valid]))))
    per_pass = {}
    for k, value in enumerate(loop['history']):
        # Before max-pass exhaustion the pass-20 iterate remains diagnostic,
        # but reason-6 states are already failed in its full-denominator score.
        reason = np.where(np.isfinite(value), 0, loop['reason']).astype(np.int8)
        if k == S2_MAX_PASSES-1:
            reason[loop['reason'] == 6] = 6
        per_pass[str(k+1)] = s2_scores(value, reason, evaluate_gaze(model, state['m'], value),
            state, 'current iterate; stopped states carry; failed states NaN', S2_HEADLINE_CELLS)[0]
    return dict(label=S2_LOOP_LABEL, start_D=loop['start'], cells=cells,
        passes=loop['passes'], failed_pass=loop['failed_pass'], final_step_D=loop['final_step'],
        fixed_point_residual_D=loop['residual'], residual_failure_reason=loop['residual_reason'],
        last_iterate_D_diagnostic_only=loop['last_iterate'],
        iterate_after_pass_D={str(k+1): value for k, value in enumerate(loop['history'])},
        per_pass_errors=per_pass, measured_step_ratios=ratios)


def s2_profile_record(profile, loop, state):
    cells = {}
    unique = profile['roots'] == 1
    finite_slope = np.isfinite(profile['max_slope'])
    checked = loop['stopped'] & unique & ~profile['has_holes'] & finite_slope & (profile['max_slope'] < 1)
    difference = np.where(loop['stopped'] & np.isfinite(profile['root_A_D']),
                          loop['estimate']-profile['root_A_D'], np.nan)
    for name, mask in s2_cells(state['groups']).items():
        # Mutually exclusive complement, in this declared order; raw flags also saved.
        remaining, complement = mask & ~checked, {}
        for why, condition in (('not_stopped', ~loop['stopped']), ('nonunique_root', ~unique),
                ('has_holes', profile['has_holes']), ('undefined_slope', ~finite_slope),
                ('max_slope_ge1', profile['max_slope'] >= 1)):
            picked = remaining & condition
            complement[why] = int(picked.sum())
            remaining &= ~picked
        cells[name] = dict(label=S2_LOOP_LABEL, states=int(mask.sum()),
            class_counts={text: int(np.sum(mask & (profile['root_class'] == code)))
                          for code, text in profile['class_codes'].items()},
            states_with_holes=int(np.sum(mask & profile['has_holes'])),
            hole_nodes=int(np.sum(profile['holes'][mask])),
            hole_nodes_by_reason={why: int(np.sum(value[mask]))
                                  for why, value in profile['holes_by_reason'].items()},
            sign_change_roots=int(np.sum(profile['sign_changes'][mask])),
            exact_node_zeros=int(np.sum(profile['exact_zeros'][mask])),
            max_slope=s2_valid_summary(profile['max_slope'], mask, 'slope'),
            count_max_slope_ge1=int(np.sum(mask & (profile['max_slope'] >= 1))),
            loop_minus_profile_root_D=s2_valid_summary(difference, mask, 'D'),
            maximum_loop_minus_profile_root=s1_extreme(difference, mask, state),
            checked_states=int(np.sum(mask & checked)), complement_states=int(np.sum(mask & ~checked)),
            complement_partition=complement)
    return dict(profile, cells=cells, checked_state=checked, loop_minus_profile_root_D=difference)


def s2_exception(report, block, exc):
    error = dict(block=block, exception=type(exc).__name__, message=str(exc),
                 numbers=getattr(exc, 'numbers', {}))
    report['errors'].append(error)
    report['S2_blocks'][block] = dict(status='unavailable', error=error,
                                      development_label=DEVELOPMENT, label=S2_LOOP_LABEL)
    report.update(status='NUMERICAL_FAIL', exit_code=3)


def s2_block(report, name, operation):
    try:
        result = operation()
    except Exception as exc:
        s2_exception(report, name, exc)
        return None
    report['S2_blocks'][name] = dict(status='complete', development_label=DEVELOPMENT,
                                   label=S2_LOOP_LABEL)
    return result


def s2_unavailable_loop(count, start):
    answer = s2_loop(lambda a, ids: (np.full(len(a), np.nan), np.full(len(a), 5, np.int8)),
                     count, start)
    answer['passes'][:] = 0
    answer['failed_pass'][:] = 0
    answer['last_iterate'][:] = np.nan
    return answer


def s2_save_loop(report, maps, state, index, name, loop, model, status='complete'):
    key = f'A{index:02d}'
    prediction = s2_scores(loop['estimate'], loop['reason'],
        evaluate_gaze(model, state['m'], loop['estimate']), state,
        'runtime-legal measured inputs with frozen five-anchor E4 calibration')
    score, estimate, reason, gaze = prediction
    score['status'] = status
    report['results'][key]['S2']['estimators'][name] = score
    report['results'][key]['S2']['loop'][name] = s2_loop_record(loop, state, model)
    prefix = key+'_'+name.replace('-', '_')+'_'
    maps.update({prefix+'A_D': estimate, prefix+'failure_reason': reason, prefix+'gaze_deg': gaze,
        prefix+'passes': loop['passes'], prefix+'failed_pass': loop['failed_pass'],
        prefix+'final_step_D': loop['final_step'], prefix+'fixed_point_residual_D': loop['residual']})
    for k in range(4):
        maps[prefix+f'pass{k+1}_A_D'] = loop['history'][k]


def s2_unavailable_profile(count, route):
    return s2_profile(lambda a, ids: (np.full(len(a), np.nan), np.full(len(a), 5, np.int8)),
                      count, route)


def s2_save_profile(report, maps, state, index, route, profile, loop, status='complete'):
    record = s2_profile_record(profile, loop, state)
    record['status'] = status
    report['results'][f'A{index:02d}']['S2']['profile'][route] = record
    for name in ('roots', 'exact_zeros', 'holes', 'root_A_D', 'max_slope'):
        maps[f'A{index:02d}_profile_{route}_{name}'] = profile[name]


def s2_pair_comparison(left, right, state, label, mask=None):
    if mask is None:
        mask = state['groups']['overall']
    both = mask & (left[1] == 0) & (right[1] == 0)
    difference = np.where(both, left[0]-right[0], np.nan)
    return dict(label=S2_LOOP_LABEL, comparison=label, states_compared=int(mask.sum()),
        both_valid_states=int(both.sum()),
        estimates_bit_identical=bool(np.array_equal(left[0][mask], right[0][mask], equal_nan=True)),
        reasons_bit_identical=bool(np.array_equal(left[1][mask], right[1][mask])),
        difference_D=s2_valid_summary(difference, mask, 'D'),
        maximum_difference=s1_extreme(difference, mask, state))


def s2_consistency(report, maps, states, model, cal, surface, loops):
    for index, state in states.items():
        key = f'A{index:02d}'
        def operation(assumed):
            return s2_map(state, model, cal, surface, 'joint', assumed)
        first = operation(0.)
        second = s1_pass2(first, operation)
        context = {}
        for name in S2_CONTEXT:
            prefix = key+'_'+name.replace('-', '_')+'_'
            context[name] = (maps[prefix+'A_D'], maps[prefix+'failure_reason'])
        output = dict(label=S2_LOOP_LABEL,
            raw_pass1_vs_S1=s2_pair_comparison(first, context[S2_CONTEXT[0]], state, 'T_joint(0) vs S1 P0'),
            raw_pass2_vs_S1=s2_pair_comparison(second, context[S2_CONTEXT[1]], state,
                                             'unmodified S1 composition T_joint(T_joint(0)) vs S1 pass2'))
        loop = loops[index]['AE5e-joint']
        mask = loop['active_before'][1]
        history_reason = np.where(np.isfinite(loop['history'][1]), 0, loop['reason']).astype(np.int8)
        output['loop_pass2_vs_S1_active_before_pass2_only'] = s2_pair_comparison(
            (loop['history'][1], history_reason), context[S2_CONTEXT[1]], state,
            'loop pass2 vs S1 pass2; stopped/failed at pass1 excluded', mask)
        for route in ('joint', 'sep'):
            primary = loops[index]['AE5e-'+route]
            differences = {}
            for start in (2, 4):
                other = loops[index][f'AE5e-{route}-start{start}']
                both = primary['stopped'] & other['stopped']
                values = np.where(both, abs(primary['estimate']-other['estimate']), np.nan)
                differences[str(start)] = dict(label=S2_LOOP_LABEL, cells={name: dict(
                    label=S2_LOOP_LABEL, both_stopped_states=int(np.sum(mask & both)),
                    exactly_one_stopped_states=int(np.sum(mask & (primary['stopped'] ^ other['stopped']))),
                    difference_D=s2_valid_summary(values, mask, 'D'),
                    maximum_difference=s1_extreme(values, mask, state))
                    for name, mask in s2_cells(state['groups']).items()})
            report['results'][key]['S2']['loop']['AE5e-'+route]['start_differences'] = differences
        report['results'][key]['S2']['consistency'] = output


def s2_pool_state(states, indices):
    entries = [states[i] for i in indices]
    return dict(w=np.concatenate([s['w'] for s in entries]),
        A=np.concatenate([np.full(len(s['w']), s['A']) for s in entries]),
        oracle=np.concatenate([s['oracle'] for s in entries]),
        state_ids=np.concatenate([s['state_ids'] for s in entries]),
        groups={name: np.concatenate([s['groups'][name] for s in entries])
                for name in entries[0]['groups']})


def s2_pooled(report, maps, states, args, loops, model, extension):
    pools = (('domain_A01_A19', list(range(1, 20))),
        ('heldout_A', [i for i in INDICES if i not in args.anchors]),
        ('anchor_interior', [5, 10, 15]), ('endpoint_A00', [0]), ('endpoint_A20', [20]))
    report['S2_pooled'], report['S2_per_pass_pooled'] = {}, {}
    for pool, indices in pools:
        state = s2_pool_state(states, indices)
        rows, per_pass = {}, {}
        for name in (*[row[0] for row in S2_ROWS], *S2_CONTEXT):
            prefixes = [f'A{i:02d}_'+name.replace('-', '_')+'_' for i in indices]
            prediction = [np.concatenate([maps[prefix+kind] for prefix in prefixes])
                          for kind in ('A_D', 'failure_reason', 'gaze_deg')]
            rows[name] = s2_scores(*prediction, state,
                'S1 context re-scored' if name in S2_CONTEXT else
                'runtime-legal measured inputs with frozen five-anchor E4 calibration')[0]
            rows[name]['headline'] = rows[name]['cells']['overall']
            if name in S2_CONTEXT:
                continue
            gaze_model = extension['model'] if name.endswith('-ext02') and extension else model
            pass_rows = {}
            for k in range(S2_MAX_PASSES):
                values = np.concatenate([loops[i][name]['history'][k] for i in indices])
                reasons = np.concatenate([np.where(np.isfinite(loops[i][name]['history'][k]),
                    0, loops[i][name]['reason']) for i in indices]).astype(np.int8)
                if k == S2_MAX_PASSES-1:
                    final = np.concatenate([loops[i][name]['reason'] for i in indices])
                    reasons[final == 6] = 6
                gazes = np.concatenate([evaluate_gaze(gaze_model, states[i]['m'],
                    loops[i][name]['history'][k]) for i in indices])
                pass_rows[str(k+1)] = s2_scores(values, reasons, gazes, state,
                    'per-pass development errors; stopped states carry; failures stay in denominator',
                    S2_HEADLINE_CELLS)[0]
            per_pass[name] = dict(label=S2_LOOP_LABEL, passes=pass_rows)
        report['S2_pooled'][pool] = dict(development_label=DEVELOPMENT,
            loop_profile_label=S2_LOOP_LABEL, indices=indices,
            note=report['definitions']['S2_pool_note'], estimators=rows)
        report['S2_per_pass_pooled'][pool] = per_pass


def s2_corner_effect(residual, states, cal):
    effects = []
    for index in INDICES:
        state = states[index]
        corners = state['groups']['four_corner_extremes']
        design = s1_size_design(tw.tangent_coordinates(state['w'][corners])['primary'], 'E4')
        require(int(corners.sum()) == 4, 'AE6 requires the four literal corner states')
        effects.append(float(np.max(abs(design@residual[index]))))
    fraction = np.asarray(effects)
    size = cal['mu']*fraction
    slopes = abs(np.diff(cal['mu'])/np.diff(cal['A']))
    mean_slope = float(np.mean(slopes))
    defined = np.isfinite(mean_slope) and mean_slope != 0
    dioptres = size/mean_slope if defined else np.full(len(size), np.nan)
    def summary(value, unit):
        finite = np.isfinite(value)
        pick = None if not np.any(finite) else int(np.flatnonzero(finite)[np.argmax(abs(value[finite]))])
        return dict(unit=unit, per_shard={f'A{i:02d}': float(value[i]) for i in INDICES},
            full_denominator=scalar_metric(value, np.ones(len(value), bool), unit),
            maximum_shard=None if pick is None else f'A{pick:02d}',
            valid_states_only=dict(label='valid states only',
                metrics=scalar_metric(value, finite, unit)))
    return dict(label=S2_AE6_LABEL, fractional_size=summary(fraction, 'fraction'),
        A00_scale_size=summary(size, 'A00_size'),
        zero_gaze_slope_equivalent_D=dict(summary(dioptres, 'D'), diagnostic_label=S2_DIOPTRE_LABEL,
            defined=bool(defined), undefined_shards=int(np.sum(~np.isfinite(dioptres)))),
        mean_abs_adjacent_zero_gaze_slope_per_D=mean_slope,
        slope_definition='mean of all 20 absolute adjacent all-21 mu_cal slopes; no floor')


def s2_ae6_own(states, cal):
    gamma, regressors, records = [], [], {}
    for index in INDICES:
        state = states[index]
        tau = tw.tangent_coordinates(state['w'])['primary']
        design = s1_size_design(tau, 'E4')
        target = state['mu_raw']/cal['mu'][index]-1
        train = state['groups']['train']
        coefficient, checked = linear_fit(design[train], target[train], f'S2 AE6 own A{index:02d}')
        residual = design@coefficient-target
        six = tw.six_coefficients(state['fit'])
        epsilon = six['c21']/six['c_s']
        gain = 180/(np.pi*six['a_bar'])
        if not np.isfinite([epsilon, gain]).all():
            raise NumericalFailure('AE6 nonfinite epsilon or G', shard=index)
        gamma.append(coefficient)
        regressors.append((state['A'], state['q'], epsilon, state['H'], cal['mu'][index], gain))
        records[f'A{index:02d}'] = dict(label=S2_AE6_LABEL, status='complete', gamma=coefficient,
            fit=checked, regressors=dict(zip(('A_D', 'q', 'epsilon', 'H_mm', 'mu_cal', 'G_mm'),
                                           regressors[-1])),
            fit_residual={name: dict(label=S2_AE6_LABEL,
                metrics=scalar_metric(residual, mask, 'fractional_size'),
                maximum=s1_extreme(residual, mask, state))
                for name, mask in s2_cells(state['groups']).items()})
    return dict(gamma=np.stack(gamma), regressors=np.asarray(regressors), shards=records)


def s2_ae6_regression(name, regressors, gamma):
    a, q, epsilon = np.asarray(regressors)[:, :3].T
    ones = np.ones(len(a))
    matrices = dict(R1=np.column_stack((ones, a)), R2=np.column_stack((ones, a, a*a)),
        R3=np.column_stack((ones, q)), R4=np.column_stack((ones, epsilon)),
        R5=np.column_stack((ones, q, epsilon)))
    try:
        coefficient, checked = linear_fit(matrices[name], gamma, 'S2 AE6 '+name)
    except Exception as exc:
        return dict(status='unavailable', label=S2_AE6_LABEL, exception=type(exc).__name__,
            message=str(exc), numbers=getattr(exc, 'numbers', {}))
    residual = matrices[name]@coefficient-gamma
    rms = np.sqrt(np.mean(residual*residual, axis=0))
    ranges = np.max(gamma, axis=0)-np.min(gamma, axis=0)
    fractions = [float(rms[j]/ranges[j]) if np.isfinite(ranges[j]) and ranges[j] != 0 else
                 None for j in range(gamma.shape[1])]
    return dict(status='complete', label=S2_AE6_LABEL, fit=checked, coefficients=coefficient,
        residual=residual, residual_rms=rms, residual_max=np.max(abs(residual), axis=0),
        target_range=ranges, unexplained_fraction=fractions,
        undefined_unexplained_fraction_targets=sum(value is None for value in fractions),
        unexplained_definition='residual RMS / target range; exactly zero/nonfinite range undefined; no floor')


def s2_ae6(report, maps, states, cal, surface):
    own = s2_block(report, 'AE6_own', lambda: s2_ae6_own(states, cal))
    report['AE6_regressions'] = dict(label=S2_AE6_LABEL, development_label=DEVELOPMENT)
    if own is None:
        for state in report['results'].values():
            state['S2']['AE6']['own'] = dict(status='unavailable', label=S2_AE6_LABEL)
        for name in ('R1', 'R2', 'R3', 'R4', 'R5'):
            error = NumericalFailure('AE6 own coefficients unavailable')
            s2_exception(report, 'AE6_'+name, error)
            report['AE6_regressions'][name] = dict(status='unavailable', label=S2_AE6_LABEL)
        return
    for index in INDICES:
        report['results'][f'A{index:02d}']['S2']['AE6']['own'] = own['shards'][f'A{index:02d}']
        maps[f'A{index:02d}_AE6_own_gamma'] = own['gamma'][index]
    for name in ('R1', 'R2', 'R3', 'R4', 'R5'):
        def regression(name=name):
            result = s2_ae6_regression(name, own['regressors'], own['gamma'])
            if result['status'] == 'unavailable':
                raise NumericalFailure(result['message'], **result.get('numbers', {}))
            result['corner_effect'] = s2_corner_effect(result['residual'], states, cal)
            return result
        result = s2_block(report, 'AE6_'+name, regression)
        if result is None:
            result = dict(status='unavailable', label=S2_AE6_LABEL,
                          error=report['S2_blocks']['AE6_'+name]['error'])
        report['AE6_regressions'][name] = result
    if surface is None:
        report['AE6_regressions']['frozen_trend_minus_own'] = dict(status='unavailable', label=S2_AE6_LABEL)
    else:
        a = cal['A']
        difference = np.column_stack((np.ones(len(a)), a, a*a))@surface['trend']-own['gamma']
        report['AE6_regressions']['frozen_trend_minus_own'] = dict(status='complete', label=S2_AE6_LABEL,
            residual=difference, corner_effect=s2_corner_effect(difference, states, cal))


def s2_analyse(states, report, args, log, maps, context_error=None):
    report['S2_blocks'] = {}
    for index, state in states.items():
        state['mu_raw'] = maps[f'A{index:02d}_AE1_raw_mu']
        report['results'][f'A{index:02d}']['S2'] = {name: {} for name in ('estimators', 'loop', 'profile', 'AE6')}
    loops = {index: {} for index in INDICES}
    model = cal = surface = extension = None
    def calibration():
        if context_error is not None:
            raise NumericalFailure('Legacy calibration/context unavailable: '+str(context_error))
        model = coefficient_model(states, args)
        cal = size_calibration(states, args)
        surface = s1_fit_surface(states, cal, args, 'E4')
        return model, cal, surface, s2_extension(model, cal, surface)
    frozen = s2_block(report, 'frozen_calibration', calibration)
    if frozen is not None:
        model, cal, surface, extension = frozen
        report['S2_calibration'] = dict(label=S2_LOOP_LABEL, model=model, size=cal['record']['runtime'],
            surface=surface, extension=dict(grid=extension['grid'], mu=extension['mu'],
                endpoint_slopes=extension['endpoint_slopes'], bounds=extension['model']['bounds'],
                label=extension['extrapolation_label']))
    else:
        # Scoring unavailable rows needs only a nonfinite gaze; no fallback estimator.
        model = dict(bounds=(0., 4.), coefficient=np.full((3, len(tw.TERMS3), 2), np.nan),
                     scale=np.ones(2))
    for block, names in (
        ('AE5e_joint', ('AE5e-joint',)),
        ('AE5e_joint_starts', ('AE5e-joint-start2', 'AE5e-joint-start4')),
        ('AE5e_joint_clip', ('AE5e-joint-clip',)), ('AE5e_joint_ext02', ('AE5e-joint-ext02',)),
        ('AE5e_sep', ('AE5e-sep', 'AE5e-sep-start2', 'AE5e-sep-start4'))):
        log('S2 '+block+'; '+S2_LOOP_LABEL)
        def compute(names=names):
            if frozen is None:
                raise NumericalFailure('Frozen S2 calibration unavailable')
            result = {}
            for index, state in states.items():
                result[index] = {}
                for name, route, start in S2_ROWS:
                    if name in names:
                        operation = lambda a, ids, state=state, route=route: s2_map(
                            state, model, cal, surface, route, a, ids, extension)
                        result[index][name] = s2_loop(operation, len(state['m']), start)
            return result
        result = s2_block(report, block, compute)
        for index, state in states.items():
            for name, route, start in S2_ROWS:
                if name not in names:
                    continue
                loop = (s2_unavailable_loop(len(state['m']), start) if result is None else result[index][name])
                loops[index][name] = loop
                gaze_model = extension['model'] if route == 'ext02' and extension else model
                s2_save_loop(report, maps, state, index, name, loop, gaze_model,
                             'estimator_unavailable' if result is None else 'complete')
    for route, name in (('joint', 'AE5e-joint'), ('sep', 'AE5e-sep'), ('ext02', 'AE5e-joint-ext02')):
        log('S2 profile_'+route+'; '+S2_LOOP_LABEL)
        def compute(route=route):
            if frozen is None:
                raise NumericalFailure('Frozen S2 calibration unavailable')
            return {index: s2_profile(lambda a, ids, state=state: s2_map(
                state, model, cal, surface, route, a, ids, extension), len(state['m']), route)
                for index, state in states.items()}
        result = s2_block(report, 'profile_'+route, compute)
        for index, state in states.items():
            profile = (s2_unavailable_profile(len(state['m']), route) if result is None else result[index])
            s2_save_profile(report, maps, state, index, route, profile, loops[index][name],
                            'unavailable' if result is None else 'complete')
    if frozen is not None:
        s2_block(report, 'S1_consistency', lambda: s2_consistency(report, maps, states, model, cal, surface, loops))
    s2_block(report, 'pooled', lambda: s2_pooled(report, maps, states, args, loops, model, extension))
    # Own-A coefficients remain independent of the runtime surface block.
    if cal is None:
        cal = s2_block(report, 'AE6_calibration', lambda: size_calibration(states, args))
    if cal is not None:
        s2_block(report, 'AE6_assembly', lambda: s2_ae6(report, maps, states, cal, surface))
    else:
        for block in ('AE6_own', 'AE6_R1', 'AE6_R2', 'AE6_R3', 'AE6_R4', 'AE6_R5'):
            s2_exception(report, block, NumericalFailure('Zero-gaze descriptive calibration unavailable'))
        for record in report['results'].values():
            record['S2']['AE6']['own'] = dict(status='unavailable', label=S2_AE6_LABEL)
        report['AE6_regressions'] = {name: dict(status='unavailable', label=S2_AE6_LABEL)
                                     for name in ('R1', 'R2', 'R3', 'R4', 'R5')}
    failures = sum(int(np.sum(loop['reason'] != 0)) for shard in loops.values() for loop in shard.values())
    failed = report['exit_code'] == 3 or any(item['status'] != 'complete' for item in report['S2_blocks'].values())
    report.update(status='NUMERICAL_FAIL' if failed else 'completed_with_failures' if failures else 'complete',
                  exit_code=3 if failed else 0, S2_estimator_failed_state_routes=failures)
    return loops


def s2_unavailable_context(states, report, maps):
    """Populate explicit S2 fail-state rows even when legacy calibration raises first."""
    for index, state in states.items():
        state['groups']['field_lt12'] = np.logical_or.reduce(
            [state['groups'][name] for name in gp.BIN_NAMES[:3]])
        key, count = f'A{index:02d}', len(state['m'])
        try:
            raw = two_radius_projection(state['Zshape'], states[0]['H'], states[0]['q'])[0]
        except Exception:
            raw = np.full(count, np.nan)
        maps[key+'_AE1_raw_mu'] = raw
        report['results'].setdefault(key, {})
        report['results'][key]['S1'] = dict(status='unavailable', estimators={})
        for name in S2_CONTEXT:
            prefix = key+'_'+name.replace('-', '_')+'_'
            maps.update({prefix+'A_D': np.full(count, np.nan),
                prefix+'failure_reason': np.full(count, 5, np.int8),
                prefix+'gaze_deg': np.full((count, 2), np.nan)})


def s2_compute_run(args, tag, log, loader=five.load_shard, synthetic=False):
    """Same in-memory AE0--AE2/S1 call sequence; then independent S2 blocks."""
    report, maps = s2_new_report(args, tag, synthetic), {}
    states = prepare_all(report, log, loader)
    if states is None:
        return report, maps
    base = states[0]
    maps.update(global_state_id=base['state_ids'], w_deg=base['w'],
                accommodation_d=np.array([states[i]['A'] for i in INDICES]))
    maps.update({'group_'+name: mask for name, mask in base['groups'].items()})
    try:
        shared_preparation(states, report, stage_s1=True)
    except Exception as exc:
        report.update(status='stopped_preparation_numerical_failure', exit_code=3)
        report['errors'].append(dict(exception=type(exc).__name__, message=str(exc),
                                    numbers=getattr(exc, 'numbers', {})))
        return report, maps
    try:
        analyse(states, report, args, log, maps)
    except Exception as exc:
        report.update(status='NUMERICAL_FAIL', exit_code=3,
            incomplete_stage_results=report['results'], results={}, ladder=[], pooled={})
        report['errors'].append(dict(exception=type(exc).__name__, message=str(exc),
                                    numbers=getattr(exc, 'numbers', {})))
        maps = {name: value for name, value in maps.items() if not name.startswith('A')}
        s2_unavailable_context(states, report, maps)
        try:
            s2_analyse(states, report, args, log, maps, context_error=exc)
        except Exception as assembly:
            report['errors'].append(dict(block='S2_unavailable_assembly',
                exception=type(assembly).__name__, message=str(assembly)))
        return report, maps
    try:
        analyse_s1(states, report, args, log, maps)
    except Exception as exc:
        report.update(status='NUMERICAL_FAIL', exit_code=3)
        report['errors'].append(dict(block='S1_assembly', exception=type(exc).__name__,
            message=str(exc), numbers=getattr(exc, 'numbers', {})))
    # Restore S2 metadata after the unmodified S1 report writer sets its headline.
    s2_metadata(report)
    try:
        s2_analyse(states, report, args, log, maps)
    except Exception as exc:
        report.update(status='NUMERICAL_FAIL', exit_code=3)
        report['errors'].append(dict(block='S2_assembly', exception=type(exc).__name__,
            message=str(exc), numbers=getattr(exc, 'numbers', {})))
    return report, maps


def s2_identical(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(s2_identical(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(s2_identical(a, b) for a, b in zip(left, right))
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return bool(np.array_equal(left, right, equal_nan=True))
    return left == right


def s2_selftests(args, log):
    """Unchanged S1--S17 first; S18--S25 use production rules and synthetic data only."""
    legacy_args = arguments(['--stages', 'S1'])
    inherited, _ = selftests_s1(legacy_args, log)
    report, maps = s2_new_report(args, 'selftest_s2', synthetic=True), {}
    checks = [*inherited.get('default_stage_regression_checks', []), *inherited['selftest_checks']]
    report['S1_selftest_context'] = inherited
    fixture = {}
    def check(name, operation):
        try:
            operation()
            checks.append(dict(name=name, passed=True))
            log('selftest '+name+' complete')
        except Exception as exc:
            checks.append(dict(name=name, passed=False, exception=type(exc).__name__,
                message=str(exc), numbers=getattr(exc, 'numbers', {})))
            log('selftest '+name+' FAILED: '+str(exc))
    def close(left, right, tolerance, message):
        left, right = np.asarray(left), np.asarray(right)
        require(np.isfinite(left).all() and np.isfinite(right).all() and
                np.max(abs(left-right)) <= tolerance, message)
    def prepared():
        if fixture:
            return fixture
        scratch = s2_new_report(args, 'selftest_s2', synthetic=True)
        states = prepare_all(scratch, log, synthetic_s1_loader())
        require(states is not None, 'S2 synthetic fixture failed to load')
        shared_preparation(states, scratch, stage_s1=True)
        cal = size_calibration(states, args)
        for state in states.values():
            state['groups']['field_lt12'] = np.logical_or.reduce(
                [state['groups'][name] for name in gp.BIN_NAMES[:3]])
            state['mu_raw'] = two_radius_projection(state['Zshape'], cal['H0'], cal['q0'])[0]
        model, surface = coefficient_model(states, args), s1_fit_surface(states, cal, args, 'E4')
        fixture.update(states=states, cal=cal, model=model, surface=surface,
            extension=s2_extension(model, cal, surface), loops={}, profiles={})
        return fixture
    def operation(f, state, route):
        return lambda a, ids: s2_map(state, f['model'], f['cal'], f['surface'],
                                    route, a, ids, f['extension'])
    def loop(index, name):
        f = prepared()
        key = (index, name)
        if key not in f['loops']:
            _, route, start = next(row for row in S2_ROWS if row[0] == name)
            f['loops'][key] = s2_loop(operation(f, f['states'][index], route),
                                     len(f['states'][index]['m']), start)
        return f['loops'][key]
    def profile(index, route='joint'):
        f = prepared()
        key = (index, route)
        if key not in f['profiles']:
            f['profiles'][key] = s2_profile(operation(f, f['states'][index], route),
                                            len(f['states'][index]['m']), route)
        return f['profiles'][key]
    def mechanics_check():
        def half(a, ids):
            return .5*a+1, np.zeros(len(a), np.int8)
        stopped = s2_loop(half, 1, 1.)
        require(stopped['reason'][0] == 0 and stopped['passes'][0] == 20, 'S18 start1 stop pass')
        close(stopped['history'][:, 0], 2.-.5**np.arange(1, 21), 0, 'S18 half-map history')
        close(stopped['final_step'], [.5**20], 0, 'S18 last step')
        close(stopped['estimate'], [2.-.5**20], 0, 'S18 final estimate')
        close(stopped['residual'], [.5**21], 0, 'S18 diagnostic residual')
        exhausted = s2_loop(half, 1, 0.)
        require(exhausted['reason'][0] == 6 and exhausted['passes'][0] == 20 and
            exhausted['failed_pass'][0] == 0 and np.isnan(exhausted['estimate'][0]) and
            np.isinf(scalar_metric(exhausted['estimate']-2, np.ones(1, bool))['rms_D']),
            'S18 reason6 and full-denominator score')
        close(exhausted['last_iterate'], [2.-2*.5**20], 0, 'S18 last iterate is diagnostic only')
        calls = np.zeros(2, int)
        def mixed(a, ids):
            calls[ids] += 1
            return np.where(ids == 0, 0., .5*a+1), np.zeros(len(a), np.int8)
        mixed_result = s2_loop(mixed, 2, 0.)
        require(mixed_result['passes'][0] == 1 and calls[0] == 2 and calls[1] == 20,
                'S18 stopped state updated again or residual call missing')
        calls = np.zeros(1, int)
        def fails(a, ids):
            calls[ids] += 1
            return np.full(len(a), 1.), np.where(calls[ids] == 2, 2, 0).astype(np.int8)
        failed = s2_loop(fails, 1, 0.)
        require(failed['reason'][0] == 2 and failed['failed_pass'][0] == 2 and
                np.isnan(failed['estimate'][0]) and calls[0] == 2, 'S18 failure propagation/restart')
        # A calibration failure before legacy analyse has any results still gets
        # every named unavailable row. This uses a tiny state subset, not I/O.
        f = prepared()
        tiny = {i: dict(s) for i, s in f['states'].items()}
        for state in tiny.values():
            count = len(state['m'])
            for name, value in list(state.items()):
                if isinstance(value, np.ndarray) and value.ndim and len(value) == count:
                    state[name] = value[:4].copy()
            state['groups'] = {name: mask[:4].copy() for name, mask in state['groups'].items()}
        scratch, tiny_maps = s2_new_report(args, 'selftest_s2', True), {}
        s2_unavailable_context(tiny, scratch, tiny_maps)
        s2_analyse(tiny, scratch, args, lambda *a, **k: None, tiny_maps,
                   context_error=NumericalFailure('injected pre-context calibration failure'))
        require(scratch['exit_code'] == 3 and all(
            set(shard['S2']['estimators']) == {row[0] for row in S2_ROWS} and
            set(shard['S2']['profile']) == {'joint', 'sep', 'ext02'} and
            all(record['status'] == 'estimator_unavailable' for record in shard['S2']['estimators'].values()) and
            all(record['status'] == 'unavailable' for record in shard['S2']['profile'].values())
            for shard in scratch['results'].values()), 'S18 calibration-failure unavailable rows')
    check('S18_exact_loop_stopping_failure_and_calibration_unavailability', mechanics_check)
    def profile_check():
        def hand(a, ids):
            values = np.empty(len(a))
            values[ids == 0] = 1.23
            values[ids == 1] = a[ids == 1]+1
            selected = ids == 2
            values[selected] = a[selected]+np.interp(a[selected], [0, 1, 2, 3, 4], [-1, 1, -1, 1, 1])
            values[(ids == 3) | (ids == 4)] = 2.
            reasons = np.zeros(len(a), np.int8)
            reasons[(ids == 3) & ((a == 1.) | (a == 1.05))] = 2
            return values, reasons
        p = s2_profile(hand, 5, 'joint')
        require(np.array_equal(p['roots'], [1, 0, 3, 1, 1]) and
                np.array_equal(p['root_class'], [1, 0, 2, 1, 1]) and
                np.array_equal(p['holes'], [0, 0, 0, 2, 0]), 'S19 roots/classes/holes')
        require(p['sign_changes'][0] == 1 and p['exact_zeros'][0] == 0 and
                p['exact_zeros'][4] == 1 and p['holes_by_reason']['out_of_calibration_range'][3] == 2,
                'S19 sign roots, exact zeros and hole reasons')
        close(p['root_A_D'][[0, 3, 4]], [1.23, 2., 2.], 1e-12, 'S19 interpolated/exact roots')
        close(p['max_slope'], [0., 1., 3., 0., 0.], 1e-12, 'S19 hand slopes')
    check('S19_profiles_roots_exact_zeros_holes_and_hand_slopes', profile_check)
    def pipeline_check():
        f = prepared()
        endpoints, compared = {}, 0
        for index, state in f['states'].items():
            primary = loop(index, 'AE5e-joint')
            p = profile(index)
            for start in (0, 2, 4):
                name = 'AE5e-joint'+('' if start == 0 else '-start'+str(start))
                result = loop(index, name)
                if 0 < index < 20:
                    require(np.all(result['reason'] == 0), 'S20 interior loop failed '+name+f' A{index:02d}')
                    close(result['estimate'], state['A'], 2e-3, 'S20 interior A')
                    require(np.max(result['residual']) <= 1e-5, 'S20 interior residual')
                    close(result['estimate'], primary['estimate'], 1e-5, 'S20 independence of start')
                    valid = result['stopped'] & np.isfinite(p['root_A_D']) & (p['root_class'] == 1)
                    require(np.all(valid), 'S20 missing or nonunique interior profile root '+name+f' A{index:02d}')
                    close(result['estimate'], p['root_A_D'], 1e-3, 'S20 loop/profile difference '+name+f' A{index:02d}')
                else:
                    stopped = result['reason'] == 0
                    require(np.all(stopped | (result['reason'] == 2)), 'S20 endpoint failure reason')
                    if np.any(stopped):
                        close(result['estimate'][stopped], state['A'], 2e-3, 'S20 endpoint stopped error')
                    for k in np.unique(result['failed_pass'][~stopped]):
                        require(k > 0, 'S20 missing endpoint failing pass')
                        ids = np.flatnonzero((result['failed_pass'] == k) & ~stopped)
                        previous = np.full(len(ids), start, float) if k == 1 else result['history'][k-2, ids]
                        table = s1_surface_table(f['surface'], f['cal'],
                            evaluate_gaze(f['model'], state['m'][ids], previous))
                        target = state['mu_raw'][ids]
                        require(np.all((target < np.min(table, axis=1)) | (target > np.max(table, axis=1))),
                                'S20 endpoint failure is not outside its runtime table')
                    endpoints[f'A{index:02d}_start{start}'] = dict(stopped=int(stopped.sum()),
                        runtime_out_of_range_failures=int((~stopped).sum()))
            runtime = s1_runtime_predictions(state, f['model'], f['cal'],
                fit_empirical(f['states'], f['cal'], args), surface=f['surface'])
            first = s2_map(state, f['model'], f['cal'], f['surface'], 'joint', 0.)
            second = s1_pass2(first, lambda a: s2_map(state, f['model'], f['cal'], f['surface'], 'joint', a))
            require(s2_identical(first, runtime[S2_CONTEXT[0]][:2]) and
                    s2_identical(second, runtime[S2_CONTEXT[1]][:2]), 'S20 raw/composed S1 bit identity')
            active = primary['active_before'][1]
            reference = runtime[S2_CONTEXT[1]]
            reason = np.where(np.isfinite(primary['history'][1]), 0, primary['reason']).astype(np.int8)
            require(np.array_equal(primary['history'][1, active], reference[0][active], equal_nan=True) and
                    np.array_equal(reason[active], reference[1][active]), 'S20 active pass2 identity')
            compared += int(active.sum())
        report['S20_endpoint_runtime_counts'] = endpoints
        report['S20_pass2_states_compared'] = compared
        log('S20 endpoint runtime counts '+json.dumps(endpoints)+'; '+S2_LOOP_LABEL)
        log('S20 loop-pass2 states compared='+str(compared)+'; '+S2_LOOP_LABEL)
    check('S20_joint_fixture_pipeline_endpoints_and_S1_pass_consistency', pipeline_check)
    def extension_check():
        f = prepared()
        gaze = np.array([[0., 0.], [12.5, 12.5], [-12.5, 12.5], [7., -4.]])
        table = s2_extended_table(f['extension'], gaze)
        close(table[:, 20:421], s1_surface_table(f['surface'], f['cal'], gaze), 1e-15,
              'S21 extended/common table equality')
        for index in (0, 20):
            result = loop(index, 'AE5e-joint-ext02')
            require(np.all(result['reason'] == 0), 'S21 extended endpoint failed')
            close(result['estimate'], f['states'][index]['A'], 2e-3, 'S21 extended endpoint error')
        outside = s2_map(f['states'][0], f['model'], f['cal'], f['surface'], 'ext02',
            np.array([-.200001, 4.200001]), np.array([0, 1]), f['extension'])
        require(np.all(outside[1] == 1) and np.isnan(outside[0]).all(), 'S21 assumed A outside extension')
    check('S21_bounded_extension_common_nodes_and_endpoints', extension_check)
    def clip_check():
        for index in INDICES:
            strict, clipped = loop(index, 'AE5e-joint'), loop(index, 'AE5e-joint-clip')
            unchanged = strict['reason'] != 2
            for name in ('estimate', 'reason', 'passes', 'failed_pass', 'final_step'):
                require(np.array_equal(strict[name][unchanged], clipped[name][unchanged], equal_nan=True),
                        'S22 clip changed state without strict reason2: '+name)
    check('S22_named_clip_changes_only_runtime_range_failures', clip_check)
    def ae6_check():
        f = prepared()
        own = s2_ae6_own(f['states'], f['cal'])
        expected = (.8+.15*f['cal']['A'])[:, None]*np.array([1., 1., 0., 0., 0.])[None]
        close(own['gamma'], expected, 1e-10, 'S23 known own-A gamma')
        anchors = np.asarray(args.anchors)
        require(np.array_equal(own['gamma'][anchors], f['surface']['record']['gamma_anchors']),
                'S23 S1 anchor gamma not bit-identical')
        outcomes = {name: s2_ae6_regression(name, own['regressors'], own['gamma'])
                    for name in ('R1', 'R2', 'R3', 'R4', 'R5')}
        require(outcomes['R1']['status'] == outcomes['R2']['status'] == 'complete' and
                np.max(abs(outcomes['R1']['residual'])) <= 1e-10 and
                outcomes['R3']['status'] == outcomes['R5']['status'] == 'unavailable',
                'S23 expected regression independence/rank failures')
        report['S23_fixture_R4_outcome'] = outcomes['R4']
        constant, varying = own['regressors'].copy(), own['regressors'].copy()
        constant[:, 2], varying[:, 2] = 1., np.linspace(.2, .7, 21)
        require(s2_ae6_regression('R4', constant, own['gamma'])['status'] == 'unavailable' and
                s2_ae6_regression('R4', varying, own['gamma'])['status'] == 'complete',
                'S23 constructed constant/varying epsilon rank gate')
        scratch = s2_new_report(args, 'selftest_s2', True)
        scratch['S2_blocks'] = {}
        scratch['results'] = {f'A{i:02d}': dict(S2=dict(AE6={})) for i in INDICES}
        s2_ae6(scratch, {}, f['states'], f['cal'], f['surface'])
        require(scratch['exit_code'] == 3 and
            scratch['AE6_regressions']['R1']['status'] == scratch['AE6_regressions']['R2']['status'] == 'complete' and
            scratch['AE6_regressions']['R3']['status'] == scratch['AE6_regressions']['R5']['status'] == 'unavailable',
            'S23 block-wise regression failures did not stand independently')
        # Unit conversion is independent of the fixture's near-zero R1 residual.
        residual = np.tile(np.array([.01, -.02, .003, .004, -.005]), (21, 1))
        effect = s2_corner_effect(residual, f['states'], f['cal'])
        slope = np.mean(abs(np.diff(f['cal']['mu'])/np.diff(f['cal']['A'])))
        for i in INDICES:
            key = f'A{i:02d}'
            fraction = effect['fractional_size']['per_shard'][key]
            close(effect['A00_scale_size']['per_shard'][key], f['cal']['mu'][i]*fraction, 0,
                  'S23 fractional-to-A00 conversion')
            close(effect['zero_gaze_slope_equivalent_D']['per_shard'][key],
                  f['cal']['mu'][i]*fraction/slope, 0, 'S23 average-slope-equivalent conversion')
    check('S23_AE6_coefficients_rank_failures_and_fractional_size_conversion', ae6_check)
    def frozen_check():
        f = prepared()
        def subset(states):
            cal = size_calibration(states, args)
            model = coefficient_model(states, args)
            surface = s1_fit_surface(states, cal, args, 'E4')
            ext = s2_extension(model, cal, surface)
            gaze = np.array([[0., 0.], [3., -2.], [12.5, 12.5]])
            return dict(cal=dict(anchors=cal['anchors'], A=cal['A'][cal['anchors']],
                mu=cal['mu'][cal['anchors']], H0=cal['H0'], q0=cal['q0'], Sigma0=cal['Sigma0']),
                model={name: model[name] for name in ('coefficient', 'scale', 'bounds')},
                surface={name: surface[name] for name in ('basis', 'trend', 'gamma_A00', 'grid')},
                extension=dict(slopes=ext['endpoint_slopes'], grid=ext['grid'], mu=ext['mu'],
                               table=s2_extended_table(ext, gaze)))
        original = subset(f['states'])
        for perturbation in ('valid_H_q', 'fitted_and_scoring_data'):
            changed = {i: dict(s) for i, s in f['states'].items()}
            for index, state in changed.items():
                if index in args.anchors:
                    continue
                if perturbation == 'valid_H_q':
                    state['H'], state['q'] = 1.01*state['H'], .01+.001*index
                    require(state['H'] > 0 and 1+state['q'] > 0 and 1+2*state['q'] > 0,
                            'S24 perturbation not physically valid')
                else:
                    for name in ('m', 'sensor', 'centered', 'Zshape', 'mu_raw', 'Sigma', 'w', 'oracle'):
                        state[name] = state[name]+.123
                    state['fit'] = dict(state['fit'], coef=state['fit']['coef']+.123)
            require(s2_identical(original, subset(changed)), 'S24 runtime subset changed: '+perturbation)
        for index, state in f['states'].items():
            poisoned = dict(state, A=state['A']+7., w=state['w']+7., oracle=state['oracle']-3.)
            for name, route, start in S2_ROWS:
                baseline = loop(index, name)
                repeated = s2_loop(operation(f, poisoned, route), len(state['m']), start)
                for field in ('estimate', 'reason', 'passes'):
                    require(np.array_equal(baseline[field], repeated[field], equal_nan=True),
                            'S24 scoring truth leaked into '+name+' '+field)
                model = f['extension']['model'] if route == 'ext02' else f['model']
                require(np.array_equal(evaluate_gaze(model, state['m'], baseline['estimate']),
                    evaluate_gaze(model, poisoned['m'], repeated['estimate']), equal_nan=True),
                    'S24 scoring truth leaked into runtime gaze')
        report['S24_AE6_exemption'] = S2_AE6_LABEL
    check('S24_runtime_truth_isolation_and_two_frozen_calibration_perturbations', frozen_check)
    def regression_check():
        baseline, old_maps = compute_run(arguments([]), 'selftest', log, synthetic_s1_loader(), True)
        previous, s1_maps = compute_run(legacy_args, 'selftest_s1', log, synthetic_s1_loader(), True)
        extended, s2_maps = s2_compute_run(args, 'selftest_s2', log, synthetic_s1_loader(), True)
        require(baseline['exit_code'] == previous['exit_code'] == 0, 'S25 legacy fixture failed')
        for name, original_maps in (('AE0-AE2', old_maps), ('S1', s1_maps)):
            require(set(original_maps) <= set(s2_maps) and all(
                np.array_equal(value, s2_maps[key], equal_nan=True) for key, value in original_maps.items()),
                'S25 '+name+' context map keys/numbers changed')
        for key in baseline['results']:
            candidate = {name: value for name, value in extended['results'][key].items() if name not in ('S1', 'S2')}
            require(s2_identical(data_json.safe(baseline['results'][key]), data_json.safe(candidate)),
                    'S25 default result keys/numbers changed')
            require(s2_identical(data_json.safe(previous['results'][key]['S1']),
                                 data_json.safe(extended['results'][key]['S1'])), 'S25 S1 results changed')
        import contextlib
        import io
        overrides = (['--anchors', '0', '10', '20'], ['--a-poly-degree', '1'],
            ['--empirical-degree', '1'], ['--gaze-readouts', '.1', '1'], ['--size-basis', 'E6'],
            ['--a-grid-step', '.02'], ['--sensitivity-step', '.1'],
            ['--loop-max-passes', '21'], ['--loop-tolerance', '1e-5'],
            ['--profile-step', '.1'], ['--extension-margin', '.3'])
        for changed in overrides:
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    arguments(['--stages', 'S2', *changed])
            except SystemExit as exc:
                require(exc.code == 2, 'S25 wrong rejected-option exit')
            else:
                raise ValueError('S25 S2 changed option accepted '+str(changed))
        fixed = arguments(['--stages', 'S2'])
        require((fixed.loop_max_passes, fixed.loop_tolerance, fixed.profile_step, fixed.extension_margin) ==
                (20, 1e-6, .05, .2), 'S25 fixed S2 constants')
        require(all(not hasattr(arguments([]), name) and not hasattr(legacy_args, name) for name in
            ('loop_max_passes', 'loop_tolerance', 'profile_step', 'extension_margin')),
            'S25 S2 options changed legacy default configuration')
        report['S25_context_map_counts'] = dict(default=len(old_maps), S1=len(s1_maps), S2=len(s2_maps))
    check('S25_unchanged_legacy_context_and_fixed_configuration', regression_check)
    report['selftest_checks'] = checks
    report['status'] = 'complete' if all(item['passed'] for item in checks) else 'selftest_failed'
    report['exit_code'] = 0 if report['status'] == 'complete' else 4
    report['selftest_scope'] = 'Synthetic code checks only; development, not scientific acceptance'
    return report, maps


def s2_ladder_text(report):
    lines = [LABEL, PROVISIONAL, DEVELOPMENT, S2_LOOP_LABEL,
        'status='+report['status']+' exit_code='+str(report['exit_code']),
        report['definitions']['S2_pool_note'],
        'S2 headline: coverage | fraction <=1 / <=0.1 arcmin | worst gaze / A with locations | failures']
    def number(value):
        return 'n/a' if value is None else format(value, '.9g')
    def encoded(value):
        return json.dumps(data_json.safe(value), sort_keys=True)
    for pool, cells in (('domain_A01_A19', ('overall', 'field_lt12', 'four_corner_extremes')),
            ('endpoint_A00', ('overall', 'field_lt12')), ('endpoint_A20', ('overall', 'field_lt12')),
            ('heldout_A', ('odd_odd', 'odd_odd/field_lt12')),
            ('anchor_interior', ('overall', 'field_lt12', 'four_corner_extremes'))):
        rows = report.get('S2_pooled', {}).get(pool, {}).get('estimators', {})
        for cell in cells:
            for name, record in rows.items():
                row = record['cells'][cell]
                lines.append(' | '.join((pool+'/'+cell, name, number(row['coverage']),
                    number(row['readout_fractions']['1.0'])+'/'+number(row['readout_fractions']['0.1']),
                    'worst gaze='+encoded(row['maximum_gaze_error']),
                    'worst A='+encoded(row['maximum_A_error']), encoded(row['failures_by_reason']),
                    DEVELOPMENT, S2_LOOP_LABEL)))
    lines.extend(('', DEVELOPMENT, S2_LOOP_LABEL,
        'S2 per-shard loop: median/max passes | stopped / failed / not-stopped | fixed-point residual'))
    for shard, data in report['results'].items():
        for name, record in data.get('S2', {}).get('loop', {}).items():
            row = record['cells']['overall']
            residual = row['fixed_point_residual_D']
            lines.append(' | '.join((shard, name,
                number(row['median_passes'])+'/'+number(row['max_passes']),
                str(row['stopped_states'])+'/'+str(row['failed_states'])+'/'+str(row['not_stopped_states']),
                'max residual D='+number(residual['full_denominator']['max_abs_D']),
                'valid states only max residual D='+number(residual['valid_states_only']['metrics']['max_abs_D']),
                'failed residual evaluations of stopped states='+str(row['residual_failed_stopped_states']),
                DEVELOPMENT, S2_LOOP_LABEL)))
    lines.extend(('', DEVELOPMENT, S2_LOOP_LABEL,
        'S2 profile per shard/route: classes | holes | slope | checked / complement'))
    for shard, data in report['results'].items():
        for route, record in data.get('S2', {}).get('profile', {}).items():
            row = record['cells']['overall']
            lines.append(' | '.join((shard, route, encoded(row['class_counts']),
                'states with holes='+str(row['states_with_holes']),
                'slope RMS/max='+number(row['max_slope']['full_denominator']['rms_slope'])+'/'+
                    number(row['max_slope']['full_denominator']['max_abs_slope']),
                'slope >=1='+str(row['count_max_slope_ge1']),
                'checked='+str(row['checked_states'])+' complement='+encoded(row['complement_partition']),
                DEVELOPMENT, S2_LOOP_LABEL)))
    lines.extend(('', DEVELOPMENT, S2_AE6_LABEL, S2_DIOPTRE_LABEL,
        'AE6 regression | status | target residual RMS / max | unexplained fraction | corner-effect summaries'))
    for name in ('R1', 'R2', 'R3', 'R4', 'R5', 'frozen_trend_minus_own'):
        record = report.get('AE6_regressions', {}).get(name)
        if record is None:
            continue
        lines.append(' | '.join((name, record['status'],
            encoded(dict(rms=record.get('residual_rms'), max=record.get('residual_max'))),
            encoded(record.get('unexplained_fraction')),
            encoded(record.get('corner_effect')), DEVELOPMENT, S2_AE6_LABEL, S2_DIOPTRE_LABEL)))
    if report.get('selftest_checks'):
        lines.extend(('', DEVELOPMENT, 'Synthetic code checks only; not scientific acceptance',
            *[item['name']+' '+('PASS' if item['passed'] else 'FAIL') for item in report['selftest_checks']]))
    for key in ('S20_endpoint_runtime_counts', 'S20_pass2_states_compared'):
        if key in report:
            lines.append(key+'='+encoded(report[key])+'; '+DEVELOPMENT+'; '+S2_LOOP_LABEL)
    # The existing renderers stay untouched and carry the complete AE0--AE2/S1 context.
    lines.extend(('', 'AE0--AE2 and S1 context follows',
                  s1_ladder_text(report, ladder_text(report)).rstrip()))
    return '\n'.join(lines)+'\n'


def s2_write_outputs(tag, report, maps, log):
    for key, value in maps.items():
        require('/' not in key and '\\' not in key and np.asarray(value).dtype.kind in 'biufc',
                'Unsafe/non-numeric S2 NPZ key: '+key)
    encoded = json.dumps(data_json.safe(report), indent=2, allow_nan=False)+'\n'
    names = s2_artifacts(tag)
    s2_guard_outputs(tag)
    (OUT/names[0]).write_text(s2_ladder_text(report), encoding='utf-8')
    s2_guard_outputs(tag)
    (OUT/names[1]).write_text(LABEL+'\n'+PROVISIONAL+'\n'+DEVELOPMENT+'\n'+S2_LOOP_LABEL+'\n\n'+encoded,
                              encoding='utf-8')
    s2_guard_outputs(tag)
    (OUT/names[2]).write_text(encoded, encoding='utf-8')
    s2_guard_outputs(tag)
    np.savez_compressed(OUT/names[3], **maps)
    log('final status='+report['status']+' saved exit_code='+str(report['exit_code']))


def s2_main(args):
    tag = 'selftest_s2' if args.selftest else 'all21_s2'
    print(LABEL, flush=True)
    log = s2_progress(tag)
    started = time.perf_counter()
    synthetic, synthetic_maps = s2_selftests(args, log)
    if args.selftest:
        report, maps = synthetic, synthetic_maps
    elif synthetic['exit_code']:
        report, maps = s2_new_report(args, tag), {}
        report.update(status='selftest_failed', exit_code=4, selftest_checks=synthetic['selftest_checks'])
    else:
        log('Own unchanged S1--S17 and S18--S25 satisfied; starting strict full5 shard loading; '+S2_LOOP_LABEL)
        report, maps = s2_compute_run(args, tag, log)
        report['selftest_precondition'] = dict(satisfied=True,
            method='Own unchanged S1--S17 plus S18--S25 in process before production input; no saved artifacts read',
            checks=synthetic['selftest_checks'])
    report['elapsed_seconds'] = time.perf_counter()-started
    s2_write_outputs(tag, report, maps, log)
    return report['exit_code']


def main(argv=None):
    args = arguments(argv)
    if args.stages == 'S2':
        return s2_main(args)
    tag = 'selftest' if args.selftest else 'all21'
    if args.stages == 'S1':
        tag += '_s1'
    print(LABEL, flush=True)
    log = Progress(tag)
    started = time.perf_counter()
    synthetic, synthetic_maps = (selftests_s1(args, log) if args.stages == 'S1' else selftests(args, log))
    if args.selftest:
        report, maps = synthetic, synthetic_maps
    elif synthetic['exit_code']:
        report = new_report(args, tag)
        if args.stages == 'S1':
            s1_metadata(report)
        report.update(status='selftest_failed', exit_code=4,
                      selftest_checks=synthetic['selftest_checks'])
        maps = {}
    else:
        log(('Own S10--S17 and default S1--S9 precondition satisfied; starting strict full5 shard loading'
             if args.stages == 'S1' else 'Own S1--S9 precondition satisfied; starting strict full5 shard loading'))
        report, maps = compute_run(args, tag, log)
        report['selftest_precondition'] = dict(satisfied=True,
            method=('Own S10--S17 plus default S1--S9 in process before production input; no saved artifacts read'
                    if args.stages == 'S1' else 'Own S1--S9 in process before production input; no saved artifacts read'),
            checks=synthetic['selftest_checks'])
    report['elapsed_seconds'] = time.perf_counter()-started
    write_outputs(tag, report, maps, log)
    return report['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
