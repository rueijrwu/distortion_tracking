r"""Termwise T0/T1, or Increment 2 T0-T3 with --stages T3, of CURRENT_PLAN.md.

noise-free design traces, ground-truth gaze, unaudited

PowerShell (from the project root; execution is a separate workflow step):
& 'C:\Users\rueijrwu\.pyenv-win-venv\envs\venv\Scripts\python.exe' Script\p4_termwise_test.py --selftest
& 'C:\Users\rueijrwu\.pyenv-win-venv\envs\venv\Scripts\python.exe' Script\p4_termwise_test.py --indices 0
& 'C:\Users\rueijrwu\.pyenv-win-venv\envs\venv\Scripts\python.exe' Script\p4_termwise_test.py --indices 0 20

The two-accommodation run is scheduled only after review of the A00 run.
No earlier result, output, cache, legacy pickle, or substitute trace is read.
All requested shards are validated before any T0/T1 result is computed.
The synthetic checks run in process before real loading, without reading or
writing a prior selftest artifact. Explicit --selftest writes its own tag.
Only report/results/maps/progress files of the selected tag are replaced,
under runs/p4_termwise. The default T1 path retains Increment 1 behaviour.
--stages T3 writes only its separate eight-file Increment 2 set; forward
fits remain diagnostics and never determine a stage constant.
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
import p4_a0_data
import p4_a0_gazepoly as gp
import p4_a0_ninept5 as five

ROOT = Path(__file__).absolute().parents[1]
OUT = ROOT / 'runs' / 'p4_termwise'
LABEL = 'noise-free design traces, ground-truth gaze, unaudited'
PROVISIONAL = 'provisional diagnostics: reported, not gated'
E1_LABEL = 'diagnostic reference, not a runtime measurement'
TAGS = ('selftest', 'a00', 'a00_a20')
TERMS3 = gp.powers(3, 'full')
TERMS1 = gp.powers(1, 'full')
ODD3 = gp.powers(3, 'odd')
ALLOWED = dict(a_x=((1, 0), 0), a_y=((0, 1), 1),
               c30=((3, 0), 0), c03=((0, 3), 1),
               c12=((1, 2), 0), c21=((2, 1), 1))
EPS_NAMES = ('c21', 'c12', 'c30', 'c03')
CONVENTIONS = ('primary', 'alternative')
require = five.require
joint = five.joint
CAMERA_F = 425.137557657
CAMERA_K1 = 2.943551420
CAMERA_K2 = -106.482286730
KAPPA1 = CAMERA_K1/CAMERA_F**2
KAPPA2 = CAMERA_K2/CAMERA_F**4


def artifacts(tag, stages='T1'):
    require(tag in TAGS, 'Unknown output tag')
    require(stages in ('T1', 'T3'), 'Unknown stage set')
    if stages == 'T3':
        return (f'ladder_{tag}.txt', f'ladder_{tag}.json',
                f'report_{tag}_T2.txt', f'results_{tag}_T2.json',
                f'report_{tag}_T3.txt', f'results_{tag}_T3.json',
                f'maps_{tag}_T2T3.npz', f'progress_{tag}_T2T3.log')
    return (f'report_{tag}.txt', f'results_{tag}.json',
            f'maps_{tag}.npz', f'progress_{tag}.log')


def guard_outputs(tag, stages='T1'):
    """Guard this script's fixed folder/files, without invoking gp's guard."""
    require(ROOT.resolve() == ROOT, 'Redirected project root')
    targets = [OUT / name for name in artifacts(tag, stages)]
    for path in [OUT, *OUT.parents, *targets]:
        require(path.resolve() == path.absolute(), 'Redirected output: '+str(path))
        if path.exists() or path.is_symlink():
            info = path.lstat()
            require(not path.is_symlink() and not
                    (getattr(info, 'st_file_attributes', 0) & 0x400),
                    'Output reparse point: '+str(path))
            if path in targets:
                require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
                        'Output must be a regular file with one link: '+str(path))
            else:
                require(path.is_dir(), 'Output parent is not a directory: '+str(path))
    sources = [five.MANIFEST, *(gp.accommodation_input(i) for i in (0, 20)),
               ROOT/'CURRENT_PLAN.md', ROOT/'P4_MODEL.md', ROOT/'P4_SPEC.md',
               *list((ROOT/'Script').glob('*.py'))]
    other_targets = [OUT/name for other_tag in TAGS for other_stages in ('T1', 'T3')
                     if (other_tag, other_stages) != (tag, stages)
                     for name in artifacts(other_tag, other_stages)]
    for i, target in enumerate(targets):
        require(target.parent == OUT, 'Output outside runs/p4_termwise')
        for source in [*sources, *other_targets, *targets[i+1:]]:
            require(target.resolve() != source.resolve() and not
                    (target.exists() and source.exists() and target.samefile(source)),
                    'Output aliases an input/source/target')


class Progress:
    def __init__(self, tag, stages='T1'):
        self.tag = tag
        self.stages = stages
        self.started = self.last = time.perf_counter()
        guard_outputs(tag, stages)
        OUT.mkdir(parents=True, exist_ok=True)
        guard_outputs(tag, stages)
        (OUT/artifacts(tag, stages)[-1]).write_text(LABEL+'\n', encoding='utf-8')

    def __call__(self, message, heartbeat=False):
        now = time.perf_counter()
        if heartbeat and now-self.last < 30:
            return
        self.last = now
        line = (datetime.now(timezone.utc).isoformat(timespec='seconds')+
                f' elapsed_s={now-self.started:.3f} '+message)
        print(line, flush=True)
        guard_outputs(self.tag, self.stages)
        with (OUT/artifacts(self.tag, self.stages)[-1]).open('a', encoding='utf-8') as stream:
            stream.write(line+'\n')
            stream.flush()


class NumericalFailure(RuntimeError):
    def __init__(self, message, **numbers):
        super().__init__(message)
        self.numbers = numbers


def clean_diagnostics(value):
    """Retain helper metrics without their retired scientific verdict flags."""
    retired = {'target_max_le_1_arcmin', 'below_tolerance'}
    if isinstance(value, dict):
        return {k: clean_diagnostics(v) for k, v in value.items() if k not in retired}
    if isinstance(value, (tuple, list)):
        return [clean_diagnostics(v) for v in value]
    return value


def gaze_summaries(error_deg, groups, valid=True):
    return clean_diagnostics(gp.summaries(error_deg, groups, valid))


def unit_summaries(field, groups, unit, valid=True):
    """Use gp.summaries' failure/denominator semantics for any vector unit."""
    result = gaze_summaries(field/60, groups, valid)

    def rename(value):
        if isinstance(value, dict):
            return {k.replace('_arcmin', '_'+unit): rename(v) for k, v in value.items()}
        return value
    return rename(result)


def ratio(numerator, denominator):
    return None if denominator == 0 else float(numerator/denominator)


def tangent_coordinates(w):
    bde = np.deg2rad(w[:, 0])
    ade = -np.deg2rad(w[:, 1])
    primary = np.column_stack((np.tan(bde)/np.cos(ade), -np.tan(ade)))
    alternative = np.column_stack((np.tan(bde), -np.tan(ade)/np.cos(bde)))
    return dict(primary=primary, alternative=alternative)


def tangent_inverse(m, gain, convention):
    tau = m/gain
    if convention == 'primary':
        ade = -np.arctan(tau[:, 1])
        bde = np.arctan(tau[:, 0]*np.cos(ade))
    else:
        bde = np.arctan(tau[:, 0])
        ade = -np.arctan(tau[:, 1]*np.cos(bde))
    return np.rad2deg(np.column_stack((bde, -ade)))


def raw_coefficients(fit, terms):
    divisors = np.array([fit['scale'][0]**i*fit['scale'][1]**j for i, j in terms])
    return fit['coef']/divisors[:, None]


def six_coefficients(fit):
    raw = raw_coefficients(fit, TERMS3)
    six = {name: float(raw[TERMS3.index(term), axis])
           for name, (term, axis) in ALLOWED.items()}
    a_bar = (six['a_x']+six['a_y'])/2
    six.update(a_bar=a_bar, c_p_bar=(six['c30']+six['c03'])/2,
               a_x_minus_a_y=six['a_x']-six['a_y'],
               c30_minus_c03=six['c30']-six['c03'],
               D12=six['c12']-six['c21'], D30=six['c30']-six['c21'],
               D03=six['c03']-six['c21'],
               c_s=-a_bar**3*(np.pi/180)**2)
    return six


def coefficient_unit(name):
    return 'deg/mm' if name.startswith('a_') else 'deg/mm^3'


def fit_record(fit, terms):
    raw = raw_coefficients(fit, terms)
    allowed_positions = set(ALLOWED.values())
    entries = []
    for t, (i, j) in enumerate(terms):
        for axis in range(2):
            entries.append(dict(powers=[i, j], axis=('wx', 'wy')[axis],
                                allowed=((i, j), axis) in allowed_positions,
                                normalised_deg=float(fit['coef'][t, axis]),
                                raw=float(raw[t, axis]), raw_unit=f'deg/mm^{i+j}'))
    return dict(terms=terms, axis_order=['wx', 'wy'],
                coefficients_normalised_deg=fit['coef'], coefficients_raw=raw,
                coefficient_entries=entries, rank=fit['rank'],
                condition=fit['condition'], singular_values=fit['singular'],
                scale_mm=fit['scale'], valid=fit['valid'], rcond=gp.RCOND)


def checked_cubic(m_train, w_train, scale, evaluation_m, context):
    fit = gp.fit_polynomial(m_train, w_train, TERMS3, scale_mm=scale)
    if fit['rank'] != 10 or not fit['valid']:
        raise NumericalFailure(context+': full cubic rank/coefficients',
                               rank=fit['rank'], condition=fit['condition'],
                               scale_mm=fit['scale'], coefficients=fit['coef'])
    prediction = gp.predict(fit, evaluation_m, TERMS3)
    if not np.all(np.isfinite(prediction)):
        raise NumericalFailure(context+': nonfinite polynomial predictions',
                               rank=fit['rank'], nonfinite_states=int(
                                   (~np.all(np.isfinite(prediction), axis=1)).sum()))
    return fit, prediction


def new_report(indices, tag, synthetic=False):
    return {LABEL: True, 'status': 'preparing', 'tag': tag,
            'synthetic': synthetic, 'indices': list(indices), 'stages': ['T0', 'T1'],
            'frame': 'sensor xsi,ysi in mm; estimator minus its own zero-gaze value',
            'estimators': {'M8': 'primary; mean of eight outer original-3x3 labels',
                           'E1': E1_LABEL},
            'diagnostic_label': PROVISIONAL,
            'definitions': dict(
                w_deg='(BDE,-ADE); angles converted to radians for tangent coordinates',
                primary_tau='(tan(BDE)/cos(ADE), -tan(ADE))',
                alternative_tau='(tan(BDE), -tan(ADE)/cos(BDE))',
                training='2600 nonzero even/even states; zero is calibration only',
                scale='A00 training max(abs(m_mm)), separately per estimator; shared thereafter',
                coefficient_operator='Full cubic inverse fit of G*tau on training gaze only',
                relative_gap='(actual-LS_prediction)/actual; null for actual=0',
                implied_epsilon='(actual cubic coefficient-T1 LS coefficient)/actual c_s',
                spread_relative='(max-min)/abs(mean); null for mean=0',
                arcmin_equivalent='r_um @ gaze_jacobian_deg_per_mm.T * 60/1000',
                radial='dot(field,w/|w|); zero gaze excluded',
                tangential='dot(field,(-wy,wx)/|w|); zero gaze excluded',
                preview='Independent training-only forward cubic, diagnostic; never feeds T1',
                fraction_removed='1-SSE_preview/SSE_T1 per group; null if T1 SSE=0',
                quantities='Coefficients/constants: double precision; errors in declared units',
                numerical_gates='full cubic rank 10, finite polynomial predictions, solve residual <=1e-10',
                nonfinite='No state dropped; any required nonfinite makes group RMS/max Infinity'),
            'inputs': [], 'errors': [], 'results': {}, 'comparisons': {}}


def prepare_all(indices, report, log, loader, keep_points=False):
    """Data exceptions stop the whole run before any stage is computed."""
    prepared = {}
    reference = None
    for index in indices:
        entry = dict(status='loading', calibration={}, timings={})
        metadata = {}
        path = gp.accommodation_input(index)
        report['inputs'].append(dict(index=index, shard=path.name, loader=entry))
        log(f'preparing A{index:02d} (all shards precede T0/T1)')
        try:
            observed, c, H, q, groups = loader(
                SimpleNamespace(quick=False, shard_index=index, input_path=path),
                entry, metadata, log)
            state = gp.accommodation_states(observed, groups, metadata)
            dioptres = entry['input']['manifest_entry']['accommodation_d']
            require(dioptres == (0 if index == 0 else 4),
                    f'A{index:02d} manifest dioptres must be exactly '+('0' if index == 0 else '4'))
            if reference is not None:
                require(np.array_equal(state['w'], reference['w']), 'Cross-shard gaze mismatch')
                require(np.array_equal(state['state_ids'], reference['state_ids']),
                        'Cross-shard state-ID mismatch')
                require(state['groups'].keys() == reference['groups'].keys() and all(
                    np.array_equal(mask, reference['groups'][name])
                    for name, mask in state['groups'].items()), 'Cross-shard group mask mismatch')
            elif index == 0:
                reference = state
            zero = int(np.flatnonzero(state['groups']['zero'])[0])
            sensors = dict(M8=state['sensor'],
                           E1=observed[:, gp.estimator_nodes()['E1'][1]].mean(axis=1))
            means = {name: dict(origin_mm=sensor[zero].copy(),
                                m_mm=sensor-sensor[zero]) for name, sensor in sensors.items()}
            entry.update(status='validated', accommodation_d=dioptres,
                         reported_calibration=dict(c_mm=c, H=H, q=q,
                             use='report only; not used in Increment 1'))
            prepared[index] = dict(state=state, metadata=metadata, means=means,
                                   accommodation_d=dioptres,
                                   tangent=tangent_coordinates(state['w']))
            if keep_points:
                prepared[index].update(points_mm=observed, loader_c_mm=np.asarray(c).copy(),
                                       loader_H=H, loader_q=q)
        except Exception as exc:
            entry['status'] = 'stopped_missing_or_invalid_data'
            report['errors'].append(dict(shard=path.name, index=index,
                exception=type(exc).__name__, message=str(exc)))
            log(f'data stop A{index:02d}: {type(exc).__name__}: {exc}')
    if report['errors']:
        report['status'] = 'stopped_missing_or_invalid_data'
        report['results'] = {}
        return None
    return prepared


def reproduction(index, six, summaries):
    rounded = ({'a_x': (-1.8208, 4), 'a_y': (-1.8208, 4),
                'c30': (3.55e-5, 7), 'c03': (3.63e-5, 7),
                'c12': (3.41e-4, 6), 'c21': (-5.85e-4, 6),
                'own_cubic_max_arcmin': (.093, 3)} if index == 0 else
               {'a_x': (-1.8952, 4), 'a_y': (-1.8955, 4),
                'c30': (3.21e-4, 6), 'c03': (3.29e-4, 6),
                'c12': (6.56e-4, 6), 'c21': (-3.57e-4, 6),
                'own_cubic_max_arcmin': (.183, 3)})
    actual = dict(six, own_cubic_max_arcmin=summaries['overall']['metrics']['vector']['max_arcmin'])
    return {name: dict(actual=actual[name], rounded_reference=value,
                       decimal_places=digits, unit=('arcmin' if name.endswith('arcmin') else coefficient_unit(name)),
                       matches_rounded_table=bool(round(actual[name], digits) == value))
            for name, (value, digits) in rounded.items()}


def t0(index, estimator, m, w, groups, scale):
    train = groups['train']
    fit, prediction = checked_cubic(m[train], w[train], scale, m, 'T0 actual')
    six = six_coefficients(fit)
    fit1 = gp.fit_polynomial(m[train], w[train], TERMS1, scale_mm=scale)
    fit_odd = gp.fit_polynomial(m[train], w[train], ODD3, scale_mm=scale)
    linear_raw = raw_coefficients(fit1, TERMS1)
    linear_coefficients = dict(a_x=float(linear_raw[TERMS1.index((1, 0)), 0]),
                               a_y=float(linear_raw[TERMS1.index((0, 1)), 1]))
    six_prediction = np.column_stack((
        six['a_x']*m[:, 0]+six['c30']*m[:, 0]**3+six['c12']*m[:, 0]*m[:, 1]**2,
        six['a_y']*m[:, 1]+six['c03']*m[:, 1]**3+six['c21']*m[:, 0]**2*m[:, 1]))
    truncated_linear = m*np.array([six['a_x'], six['a_y']])
    baselines = dict(
        own_full_cubic=gaze_summaries(prediction-w, groups, fit['valid']),
        own_linear=gaze_summaries(gp.predict(fit1, m, TERMS1)-w, groups, fit1['valid']),
        cubic_linear_terms=gaze_summaries(truncated_linear-w, groups, fit['valid']),
        cubic_six_allowed_terms=gaze_summaries(six_prediction-w, groups, fit['valid']))
    record = fit_record(fit, TERMS3)
    forbidden = [entry for entry in record['coefficient_entries'] if not entry['allowed']]
    sensitivity = gp.sensitivity(fit, TERMS3)
    quotient = six['D12']/six['c_s']
    result = dict(coefficients=six, units=dict(linear='deg/mm', cubic='deg/mm^3',
                  shared_normalised='deg', c_s='deg/mm^3'),
                  full_cubic=record, forbidden=forbidden,
                  forbidden_max_abs_normalised_deg=max(abs(x['normalised_deg']) for x in forbidden),
                  forbidden_max_abs_raw=max(abs(x['raw']) for x in forbidden),
                  forbidden_raw_max_note='Maximum of raw numerical entries; degrees have different units; see per-degree maxima',
                  forbidden_max_abs_raw_by_degree={str(degree): max(
                      abs(x['raw']) for x in forbidden if sum(x['powers']) == degree)
                      for degree in range(4)},
                  odd_cubic=fit_record(fit_odd, ODD3),
                  oddness=clean_diagnostics(gp.oddness(fit_odd, ODD3, m[train])),
                  own_linear_fit=fit_record(fit1, TERMS1),
                  own_linear_coefficients=linear_coefficients,
                  baselines=baselines, sensitivity=sensitivity,
                  D12_over_c_s=quotient,
                  convention_indicator='primary' if quotient > 0 else 'alternative')
    if estimator == 'M8':
        result['reproduction'] = reproduction(index, six, baselines['own_full_cubic'])
    return result, fit, prediction


def operator(gain, tau, w, train, scale):
    mm = gain*tau
    fit, _ = checked_cubic(mm[train], w[train], scale, mm, 'T1 coefficient operator')
    return six_coefficients(fit), fit


def solve_gain(target, quantity, tau, w, train, scale):
    if not np.isfinite(target) or target == 0:
        raise NumericalFailure('T1 gain target must be finite and nonzero',
                               target=target, quantity=quantity)
    seed = 180/(np.pi*target)
    gain = seed
    model, fit = operator(gain, tau, w, train, scale)
    history = [dict(iteration=0, gain_mm=gain, model_target=model[quantity],
                    relative_residual=abs(model[quantity]-target)/abs(target))]
    for iteration in range(1, 31):
        gain = gain*model[quantity]/target
        model, fit = operator(gain, tau, w, train, scale)
        residual = abs(model[quantity]-target)/abs(target)
        history.append(dict(iteration=iteration, gain_mm=gain,
                            model_target=model[quantity], relative_residual=residual))
        if np.isfinite(residual) and residual <= 1e-10:
            return dict(G0_mm=seed, G1_mm=gain, G1_over_G0_minus_1=gain/seed-1,
                        iterations=iteration, solve_residual=residual,
                        matched_quantity=quantity, target=target,
                        target_unit='deg/mm', history=history), model, fit
    raise NumericalFailure('T1 LS-matched gain residual exceeds 1e-10',
                           quantity=quantity, target=target, solver_history=history,
                           last_gain_mm=gain, solve_residual=residual)


def taylor_coefficients(actual, convention):
    cs = actual['c_s']
    c12, c21 = ((cs/2, 0.) if convention == 'primary' else (0., cs/2))
    return dict(a_x=actual['a_bar'], a_y=actual['a_bar'], a_bar=actual['a_bar'],
                c30=cs/3, c03=cs/3, c12=c12, c21=c21,
                c_p_bar=cs/3, D12=c12-c21, D30=cs/3-c21, D03=cs/3-c21,
                a_x_minus_a_y=0., c30_minus_c03=0.)


def projected_components(field, w):
    radius = np.linalg.norm(w, axis=1)
    nonzero = radius > 0
    radial = np.full(len(w), np.nan)
    tangential = np.full(len(w), np.nan)
    unit = w[nonzero]/radius[nonzero, None]
    radial[nonzero] = np.sum(field[nonzero]*unit, axis=1)
    tangential[nonzero] = np.sum(field[nonzero]*np.column_stack((-unit[:, 1], unit[:, 0])), axis=1)
    return dict(x=field[:, 0], y=field[:, 1], radial=radial, tangential=tangential), nonzero


def component_statistics(components, mask, nonzero):
    result = {}
    for name, values in components.items():
        required = mask & nonzero if name in ('radial', 'tangential') else mask
        v = values[required]
        n = len(v)
        if not n:
            stats = dict(states=0, signed_mean=None, rms=None, max_abs=None)
        elif not np.all(np.isfinite(v)):
            stats = dict(states=n, signed_mean=float('inf'), rms=float('inf'), max_abs=float('inf'))
        else:
            stats = dict(states=n, signed_mean=float(np.mean(v)),
                         rms=float(np.sqrt(np.mean(v*v))), max_abs=float(np.max(abs(v))))
        result[name] = stats
    return result


def location_of_maximum(field, mask, w, metadata):
    ids = np.flatnonzero(mask)
    if not len(ids):
        return None
    vectors = np.linalg.norm(field[ids], axis=1)
    failed = ~np.isfinite(vectors)
    local = int(np.flatnonzero(failed)[0]) if np.any(failed) else int(np.argmax(vectors))
    pos = int(ids[local])
    return dict(state_id=int(metadata['global_state_id'][pos]),
                ade_index=int(metadata['global_ade_index'][pos]),
                bde_index=int(metadata['global_bde_index'][pos]),
                ADE_deg=float(-w[pos, 1]), BDE_deg=float(w[pos, 0]),
                components=field[pos], vector_magnitude=float('inf') if np.any(failed) else float(vectors[local]),
                nonfinite_required_state=bool(np.any(failed)))


def residual_structure(field, w, groups, metadata, unit):
    components, nonzero = projected_components(field, w)
    radius = np.linalg.norm(w, axis=1)
    phi = np.rad2deg(np.arctan2(abs(w[:, 1]), abs(w[:, 0])))
    sectors = dict(near_x=phi < 22.5, diagonal=(phi >= 22.5) & (phi <= 67.5),
                   near_y=phi > 67.5)
    by_bin = {name: component_statistics(components, groups[name], nonzero) for name in gp.BIN_NAMES}
    sector_bins = {sector: {name: component_statistics(components, mask & groups[name], nonzero)
                           for name in gp.BIN_NAMES} for sector, mask in sectors.items()}
    ga, gb = metadata['global_ade_index'], metadata['global_bde_index']
    ba, bb = np.minimum(ga//20, 4), np.minimum(gb//20, 4)
    coarse = {name+'_signed_mean': np.full((5, 5), np.nan) for name in components}
    coarse['vector_rms'] = np.full((5, 5), np.nan)
    coarse['states'] = np.zeros((5, 5), dtype=int)
    for a in range(5):
        for b in range(5):
            mask = (ba == a) & (bb == b)
            stats = component_statistics(components, mask, nonzero)
            coarse['states'][a, b] = int(mask.sum())
            for name in components:
                coarse[name+'_signed_mean'][a, b] = stats[name]['signed_mean']
            vectors = field[mask]
            coarse['vector_rms'][a, b] = (float(np.sqrt(np.mean(np.sum(vectors*vectors, axis=1))))
                if np.all(np.isfinite(vectors)) else float('inf'))
    profiles = {}
    lines = dict(BDE_axis=ga == 50, ADE_axis=gb == 50,
                 diagonal_equal_indices=ga == gb, diagonal_opposite_indices=ga+gb == 100)
    for name, mask in lines.items():
        ids = np.flatnonzero(mask)
        profiles[name] = dict(state_id=metadata['global_state_id'][ids],
            ade_index=ga[ids], bde_index=gb[ids], w_deg=w[ids], radius_deg=radius[ids],
            x=components['x'][ids], y=components['y'][ids],
            radial=components['radial'][ids], tangential=components['tangential'][ids])
    return dict(unit=unit, components=component_statistics(components, groups['overall'], nonzero),
                by_magnitude=by_bin, by_sector_and_magnitude=sector_bins,
                coarse_blocks=coarse,
                coarse_definition='Rows ADE block, columns BDE block; min(index//20,4)',
                line_profiles=profiles,
                maxima={split: location_of_maximum(field, groups[split], w, metadata)
                        for split in gp.SPLITS})


def preview_fit(m, tau, gain, groups, jacobian):
    train = groups['train']
    columns = (np.column_stack((tau[:, 0], tau[:, 0]**3, tau[:, 0]*tau[:, 1]**2)),
               np.column_stack((tau[:, 1], tau[:, 1]**3, tau[:, 1]*tau[:, 0]**2)))
    parameters = []
    predicted = np.zeros_like(m)
    ranks, conditions = [], []
    for axis in range(2):
        coef, _, rank, singular = np.linalg.lstsq(columns[axis][train], m[train, axis], rcond=gp.RCOND)
        parameters.append(coef)
        ranks.append(int(rank))
        conditions.append(float(singular[0]/singular[-1]) if singular[-1] else float('inf'))
        predicted[:, axis] = columns[axis] @ coef
    params = np.asarray(parameters)
    gx, gy = params[:, 0]
    px, cx = params[0, 1:]/gx
    py, cy = params[1, 1:]/gy
    remainder_um = 1000*(m-predicted)
    t1_residual = m-gain*tau
    fraction = {}
    for split in gp.SPLITS:
        mask = groups[split]
        before = float(np.sum(t1_residual[mask]**2))
        after = float(np.sum((m-predicted)[mask]**2))
        fraction[split] = dict(T1_sum_squares_mm2=before, preview_sum_squares_mm2=after,
                               fraction_removed=None if before == 0 else 1-after/before)
    parameters_report = dict(G_x_mm=gx, G_y_mm=gy,
        G_x_relative_to_G1=gx/gain-1, G_y_relative_to_G1=gy/gain-1,
        epsilon_px=px, epsilon_cx=cx, epsilon_py=py, epsilon_cy=cy,
        epsilon_cx_minus_cy=cx-cy, epsilon_px_minus_py=px-py,
        epsilon_p_bar=(px+py)/2, epsilon_c_bar=(cx+cy)/2,
        epsilon_p_bar_minus_c_bar=(px+py-cx-cy)/2)
    result = dict(use='next-term preview only; not a stage calibration',
                parameters=parameters_report, coefficients_mm=params,
                column_order=['axis tangent', 'pure cubic', 'cross cubic'],
                rank_by_axis=ranks, condition_by_axis=conditions, rcond=gp.RCOND,
                fraction_of_T1_SSE_removed=fraction,
                remainder_um=unit_summaries(remainder_um, groups, 'um'),
                remainder_arcmin_equivalent=unit_summaries(
                    remainder_um @ jacobian.T*60/1000, groups, 'arcmin_equivalent'))
    return result, remainder_um


def t1(convention, actual, fit3, actual_prediction, linear, m, w, groups, scale, tau, metadata):
    train = groups['train']
    solver, predicted, model_fit = solve_gain(actual['a_bar'], 'a_bar', tau, w, train, scale)
    gain = solver['G1_mm']
    axis_gains = {axis: solve_gain(actual[axis], axis, tau, w, train, scale)[0]
                  for axis in ('a_x', 'a_y')}
    tay = taylor_coefficients(actual, convention)
    table = []
    for name in ('a_x', 'a_y', 'c30', 'c03', 'c12', 'c21', 'D12', 'D30', 'D03',
                 'a_x_minus_a_y', 'c30_minus_c03'):
        gap = actual[name]-predicted[name]
        table.append(dict(quantity=name, unit=coefficient_unit(name), actual=actual[name],
                          LS_prediction=predicted[name], Taylor_prediction=tay[name],
                          gap=gap, relative_gap=ratio(gap, actual[name]),
                          gap_over_c_s=gap/actual['c_s'],
                          gap_over_c_s_unit='mm^2' if name.startswith('a_') else 'dimensionless'))
    epsilon = {name: (actual[name]-predicted[name])/actual['c_s'] for name in EPS_NAMES}
    average = float(np.mean(list(epsilon.values())))
    spread = max(epsilon.values())-min(epsilon.values())
    model_m = gain*tau
    gaze = tangent_inverse(m, gain, convention)
    gaze_error = (gaze-w)*60
    forward_um = (m-model_m)*1000
    jac = gp.sensitivity(fit3, TERMS3)['gaze_jacobian_deg_per_mm']
    model_poly_prediction = gp.predict(fit3, model_m, TERMS3)
    if not np.all(np.isfinite(model_poly_prediction)):
        raise NumericalFailure('T1 actual-polynomial model predictions nonfinite', G1_mm=gain)
    preview, remainder = preview_fit(m, tau, gain, groups, jac)
    result = dict(stage_role='stage result' if convention == 'primary' else 'reported alternative only',
                  convention=convention, solver=solver,
                  diagnostic_axis_gain_solvers=axis_gains,
                  diagnostic_linear_only_gains_mm={axis: 180/(np.pi*linear[axis])
                                                   for axis in ('a_x', 'a_y')},
                  coefficient_operator=fit_record(model_fit, TERMS3),
                  LS_predicted_coefficients=predicted, Taylor_coefficients=tay,
                  Taylor_definition='Actual a_bar/c_s; alternative shifts geometric 1/2 from c12 to c21',
                  predicted_vs_actual=table,
                  implied_epsilon=dict(values=epsilon, cross_terms=['c21', 'c12'],
                      pure_terms=['c30', 'c03'], mean=average, spread=spread,
                      spread_relative_to_abs_mean=ratio(spread, abs(average)), unit='dimensionless'),
                  gaze=gaze_summaries(gaze-w, groups),
                  forward_um=unit_summaries(forward_um, groups, 'um'),
                  forward_arcmin_equivalent=unit_summaries(
                      forward_um @ jac.T*60/1000, groups, 'arcmin_equivalent'),
                  E_poly=gaze_summaries(model_poly_prediction-actual_prediction, groups),
                  E_close=gaze_summaries(model_poly_prediction-w, groups),
                  preview=preview)
    structure = dict(gaze=residual_structure(gaze_error, w, groups, metadata, 'arcmin'),
                     forward=residual_structure(forward_um, w, groups, metadata, 'um'))
    maps = dict(t1_model_mm=model_m, t1_gaze_deg=gaze, t1_gaze_err_arcmin=gaze_error,
                t1_fwd_res_um=forward_um, preview_remainder_um=remainder)
    return result, structure, maps


def comparison_for_estimators(items):
    comparisons = {}
    for convention in CONVENTIONS:
        m8 = items['M8']['T1'][convention]
        e1 = items['E1']['T1'][convention]
        comparisons[convention] = dict(
            gain_relative_difference=(m8['solver']['G1_mm']-e1['solver']['G1_mm'])/e1['solver']['G1_mm'],
            definition='(G1_M8-G1_E1)/G1_E1; reported only',
            implied_epsilon_M8_minus_E1={name: m8['implied_epsilon']['values'][name]-
                                       e1['implied_epsilon']['values'][name] for name in EPS_NAMES})
    return comparisons


def comparison_for_accommodation(results):
    comparison = {}
    for estimator in ('M8', 'E1'):
        first, last = results['A00'][estimator], results['A20'][estimator]
        comparison[estimator] = {}
        for convention in CONVENTIONS:
            low, high = first['T1'][convention], last['T1'][convention]
            a, b = first['T0']['coefficients'], last['T0']['coefficients']
            changes = dict(G1_mm=high['solver']['G1_mm']-low['solver']['G1_mm'])
            changes['actual_coefficients'] = {key: b[key]-a[key] for key in a}
            gaps0 = {r['quantity']: r['gap'] for r in low['predicted_vs_actual']}
            gaps4 = {r['quantity']: r['gap'] for r in high['predicted_vs_actual']}
            changes['coefficient_gaps'] = {key: gaps4[key]-gaps0[key] for key in gaps0}
            changes['implied_epsilon'] = {key: high['implied_epsilon']['values'][key]-
                                         low['implied_epsilon']['values'][key] for key in EPS_NAMES}
            changes['implied_epsilon_spread'] = high['implied_epsilon']['spread']-low['implied_epsilon']['spread']
            changes['preview_remainder_rms_um'] = (high['preview']['remainder_um']['overall']['metrics']['vector']['rms_um']-
                                                   low['preview']['remainder_um']['overall']['metrics']['vector']['rms_um'])

            def per_d(v):
                return {key: per_d(val) for key, val in v.items()} if isinstance(v, dict) else v/4
            comparison[estimator][convention] = dict(delta_D=4, signed_change=changes,
                                                     change_per_dioptre=per_d(changes))
    return comparison


def what_to_look_at(report):
    """Repeat the eleven reading prompts with this run's own numbers."""
    notes = []
    for shard, estimators in report['results'].items():
        for name, item in estimators.items():
            zero, one = item['T0'], item['T1']['primary']
            prefix = f'{shard} {name}: '
            coeff = zero['coefficients']
            overall = lambda summaries: summaries['overall']['metrics']['vector']
            by_gap = {row['quantity']: row['gap'] for row in one['predicted_vs_actual']}
            profiles = item['residual_structure']['primary']['gaze']
            notes.extend([
                prefix+'1. Fourteen forbidden terms / six-term truncation: '+
                    f"max normalised={zero['forbidden_max_abs_normalised_deg']:.17g} deg; six-term max={overall(zero['baselines']['cubic_six_allowed_terms'])['max_arcmin']:.6g} arcmin.",
                prefix+f"2. Convention: D12/c_s={zero['D12_over_c_s']:.17g}; indicator={zero['convention_indicator']}; compare both conventions below.",
                prefix+'3. Linear-only baselines / T1: '+
                    f"own linear RMS={overall(zero['baselines']['own_linear'])['rms_arcmin']:.6g}; truncated RMS={overall(zero['baselines']['cubic_linear_terms'])['rms_arcmin']:.6g}; T1 RMS={overall(one['gaze'])['rms_arcmin']:.6g} arcmin.",
                prefix+'4. Geometry differences and missing c21: '+
                    ', '.join(f'{key} gap={by_gap[key]:.17g}' for key in ('D12', 'D30', 'D03', 'c21'))+' deg/mm^3.',
                prefix+'5. Implied epsilons: '+', '.join(f'{key}={value:.17g}' for key, value in one['implied_epsilon']['values'].items())+
                    f"; spread={one['implied_epsilon']['spread']:.17g}.",
                prefix+'6. Radial/tangential T1 gaze residual: '+
                    f"RMS={profiles['components']['radial']['rms']:.6g}/{profiles['components']['tangential']['rms']:.6g} arcmin; compare axis/diagonal sectors.",
                prefix+'7. Profiles / outer magnitude bin: all 101 points per line retained; '+
                    f"outer radial RMS={profiles['by_magnitude']['bin_12_plus']['radial']['rms']:.6g} arcmin; inspect curvature in the sampled profile tables.",
                prefix+'8. Preview remainder after any forward cubic: '+
                    f"RMS={overall(one['preview']['remainder_um'])['rms_um']:.6g}, max={overall(one['preview']['remainder_um'])['max_um']:.6g} um; magnitude bins and locations are reported.",
                prefix+'9. x/y splits actual versus T1: '+
                    f"a_x-a_y={coeff['a_x_minus_a_y']:.17g}/{one['LS_predicted_coefficients']['a_x_minus_a_y']:.17g} deg/mm; "+
                    f"c30-c03={coeff['c30_minus_c03']:.17g}/{one['LS_predicted_coefficients']['c30_minus_c03']:.17g} deg/mm^3."])
        comp = report['comparisons']['E1_versus_M8'][shard]['primary']
        notes.append(f"{shard}: 10. E1 versus M8 gain ratio difference={comp['gain_relative_difference']:.17g}; "+
                     'implied epsilon M8-E1: '+', '.join(f'{key}={val:.17g}' for key, val in comp['implied_epsilon_M8_minus_E1'].items())+'.')
    if 'A4_versus_A0' in report['comparisons']:
        for name, comparisons in report['comparisons']['A4_versus_A0'].items():
            values = comparisons['primary']['signed_change']
            notes.append(f"{name}: 11. A4 versus A0: delta G={values['G1_mm']:.17g} mm; "+
                         'delta implied epsilon: '+', '.join(f'{k}={v:.17g}' for k, v in values['implied_epsilon'].items())+
                         f"; delta spread={values['implied_epsilon_spread']:.17g}; delta remainder RMS={values['preview_remainder_rms_um']:.6g} um; all per-D changes above.")
    else:
        notes.append('11. A4 versus A0: not present in this A00-only run; scheduled after A00 audit/review.')
    return notes


def analyse_prepared(prepared, report, log):
    maps = {}
    scales = {}
    ref = prepared[0]
    state = ref['state']
    maps.update(ade_index=ref['metadata']['global_ade_index'],
                bde_index=ref['metadata']['global_bde_index'], state_id=state['state_ids'])
    maps.update({'mask_'+name: mask for name, mask in state['groups'].items()})
    report['split_counts'] = {key: int(mask.sum()) for key, mask in state['groups'].items()}
    report['comparisons']['E1_versus_M8'] = {}
    for index in report['indices']:
        shard = f'A{index:02d}'
        data = prepared[index]
        w, groups = data['state']['w'], data['state']['groups']
        shard_results = report['results'][shard] = {}
        for estimator in ('M8', 'E1'):
            log(f'computing {shard} {estimator} T0/T1')
            mean = data['means'][estimator]
            m = mean['m_mm']
            if index == 0:
                scales[estimator] = np.max(abs(m[groups['train']]), axis=0)
            scale = scales[estimator]
            zero, fit3, prediction = t0(index, estimator, m, w, groups, scale)
            item = shard_results[estimator] = dict(accommodation_d=data['accommodation_d'],
                role='primary' if estimator == 'M8' else E1_LABEL,
                origin_mm=mean['origin_mm'], shared_A00_scale_mm=scale, T0=zero,
                T1={}, residual_structure={})
            prefix = shard+'_'+estimator+'_'
            maps.update({prefix+'w_deg': w, prefix+'m_mm': m,
                         prefix+'tau': data['tangent']['primary'],
                         prefix+'tau_alt': data['tangent']['alternative'],
                         prefix+'cubic_err_arcmin': (prediction-w)*60})
            for convention in CONVENTIONS:
                one, structure, fields = t1(convention, zero['coefficients'], fit3, prediction,
                    zero['own_linear_coefficients'], m, w, groups, scale,
                    data['tangent'][convention], data['metadata'])
                item['T1'][convention] = one
                item['residual_structure'][convention] = structure
                suffix = '' if convention == 'primary' else 'alt_'
                maps.update({prefix+suffix+key: val for key, val in fields.items()})
            item['convention_comparison'] = {convention: dict(
                convention_indicator=zero['convention_indicator'],
                gaze_overall=item['T1'][convention]['gaze']['overall']['metrics'],
                preview_cross_asymmetry=item['T1'][convention]['preview']['parameters']['epsilon_cx_minus_cy'],
                preview_remainder_overall_um=item['T1'][convention]['preview']['remainder_um']['overall']['metrics'])
                for convention in CONVENTIONS}
        report['comparisons']['E1_versus_M8'][shard] = comparison_for_estimators(shard_results)
    if 20 in report['indices']:
        report['comparisons']['A4_versus_A0'] = comparison_for_accommodation(report['results'])
    report['status'] = 'complete'
    report['what_to_look_at'] = what_to_look_at(report)
    return maps


def compute_run(indices, tag, log, loader=five.load_shard, synthetic=False):
    report = new_report(indices, tag, synthetic)
    prepared = prepare_all(indices, report, log, loader)
    if prepared is None:
        return report, {}, 2
    try:
        maps = analyse_prepared(prepared, report, log)
    except NumericalFailure as exc:
        report['status'] = 'NUMERICAL_FAIL'
        report['errors'].append(dict(message=str(exc), numbers=exc.numbers))
        report['incomplete_stage_results'] = report['results']
        report['results'] = {}
        log('NUMERICAL_FAIL: '+str(exc))
        return report, {}, 3
    except (np.linalg.LinAlgError, FloatingPointError) as exc:
        report['status'] = 'NUMERICAL_FAIL'
        report['errors'].append(dict(message=str(exc), exception=type(exc).__name__))
        report['incomplete_stage_results'] = report['results']
        report['results'] = {}
        log('NUMERICAL_FAIL: '+str(exc))
        return report, {}, 3
    return report, maps, 0


def synthetic_loader(case, fail_index=None):
    """Same loader signature; creates the true grid without any trace I/O."""
    def load(args, entry, metadata, log):
        if args.shard_index == fail_index:
            raise FileNotFoundError('Synthetic missing shard '+args.input_path.name)
        ga, gb = np.indices((101, 101)).reshape(2, -1)
        w = np.column_stack((-12.5+.25*gb, 12.5-.25*ga))
        zero = (ga == 50) & (gb == 50)
        groups = dict(overall=np.ones(10201, bool),
            train=(ga % 2 == 0) & (gb % 2 == 0) & ~zero,
            even_odd=(ga % 2 == 0) & (gb % 2 == 1),
            odd_even=(ga % 2 == 1) & (gb % 2 == 0),
            odd_odd=(ga % 2 == 1) & (gb % 2 == 1), zero=zero)
        for a, b in ((0, 0), (0, 100), (100, 0), (100, 100)):
            groups[f'extreme_{a}_{b}'] = (ga == a) & (gb == b)
        tau = tangent_coordinates(w)['alternative' if case == 'S4' else 'primary']
        gain = -31.5 if args.shard_index == 0 else -30.5
        m = gain*tau
        if case in ('S2', 'S3', 'S8', 'S10'):
            ep, ec = (-.3, -.3) if case in ('S2', 'S8') else (-.30, -.33)
            m = gain*tau*(1+ep*tau*tau+ec*tau[:, ::-1]**2)
        if case == 'S5':
            m[:, 0] += 1e-4*tau[:, 1]**2
        centre = np.zeros(2)
        if case in ('S11', 'S12'):
            m = stage_forward(tau, gain, -.33, 'K1' if case == 'S11' else 'K1K2')
        if case == 'S12':
            centre = np.array([.01, -.02])
            m = m+centre
        # All 25 labels receive the same sensor law, including M8 and E1.
        observed = np.repeat(m[:, None, :], 25, axis=1)
        metadata.update(w_deg=w, global_state_id=np.arange(10201),
                        global_ade_index=ga, global_bde_index=gb, input_failed=np.zeros(10201, bool))
        entry['input'] = dict(path='synthetic:'+case, synthetic=True, bytes=0, sha256=None,
            states=10201, complete=True,
            manifest_entry=dict(accommodation_d=0 if args.shard_index == 0 else 4,
                                row_count=255025, expected_row_count=255025,
                                completion_status='complete'))
        if case == 'S12' and args.shard_index == 20:
            centre = centre+np.array([.003, 0.])
        return observed, centre, 1., 0., groups
    return load


def synthetic_selftests(log):
    tests = {}
    reports, maps = {}, {}

    def add(case, name, outcome, numbers):
        tests.setdefault(case, []).append(dict(check=name, satisfied=bool(outcome), numbers=numbers))

    for case in ('S1', 'S2', 'S3', 'S4', 'S5', 'S6'):
        indices = [0, 20] if case in ('S1', 'S6') else [0]
        log('synthetic '+case)
        report, fields, exit_code = compute_run(indices, 'selftest', lambda *a, **k: None,
            loader=synthetic_loader(case, fail_index=20 if case == 'S6' else None), synthetic=True)
        reports[case] = report
        maps.update({case+'_'+key: value for key, value in fields.items()})
        if case == 'S6':
            add(case, 'missing loader path stops whole run, names A20, computes no T0/T1',
                report['status'] == 'stopped_missing_or_invalid_data' and exit_code != 0 and
                not report['results'] and any(e.get('shard', '').endswith('A20.pkl') for e in report['errors']),
                dict(status=report['status'], exit_code=exit_code, errors=report['errors'], results=report['results']))
            continue
        add(case, 'production analysis completes', exit_code == 0 and report['status'] == 'complete',
            dict(exit_code=exit_code, status=report['status'], errors=report['errors']))
        if exit_code:
            continue
        for shard, estimators in report['results'].items():
            expected_gain = -31.5 if shard == 'A00' else -30.5
            for estimator, item in estimators.items():
                prefix = shard+' '+estimator+' '
                primary = item['T1']['primary']
                preview = primary['preview']['parameters']
                if case == 'S1':
                    relative_gain = abs(primary['solver']['G1_mm']/expected_gain-1)
                    add(case, prefix+'gain recovery', relative_gain <= 1e-10, relative_gain)
                    cs = item['T0']['coefficients']['c_s']
                    abar = abs(item['T0']['coefficients']['a_bar'])
                    differences = {r['quantity']: abs(r['gap'])/(abar if r['quantity'].startswith('a_') else abs(cs))
                                   for r in primary['predicted_vs_actual']}
                    add(case, prefix+'LS predictions', all(v <= 1e-9 for v in differences.values()), differences)
                    epsilon = primary['implied_epsilon']['values']
                    add(case, prefix+'implied epsilon zero', max(abs(v) for v in epsilon.values()) <= 1e-9, epsilon)
                    maxima = {name: values['metrics']['vector']['max_arcmin'] for name, values in primary['gaze'].items()}
                    add(case, prefix+'gaze inverse every split', max(maxima.values()) <= 1e-9, maxima)
                    eps = {key: preview[key] for key in ('epsilon_px', 'epsilon_cx', 'epsilon_py', 'epsilon_cy')}
                    add(case, prefix+'preview epsilon zero', max(abs(v) for v in eps.values()) <= 1e-9, eps)
                    add(case, prefix+'primary convention indicator', item['T0']['convention_indicator'] == 'primary', item['T0']['convention_indicator'])
                    add('S7', prefix+'one-step scaling identity', primary['solver']['iterations'] == 1 and
                        primary['solver']['solve_residual'] <= 1e-10, primary['solver'])
                elif case in ('S2', 'S3'):
                    ep, ec = (-.3, -.3) if case == 'S2' else (-.30, -.33)
                    gain_errors = {key: abs(preview[key]-expected_gain) for key in ('G_x_mm', 'G_y_mm')}
                    eps_errors = {key: abs(preview[key]-(ep if '_p' in key else ec))
                                  for key in ('epsilon_px', 'epsilon_cx', 'epsilon_py', 'epsilon_cy')}
                    add(case, prefix+'forward gain / epsilon recovery',
                        max(gain_errors.values()) <= 1e-9 and max(eps_errors.values()) <= 1e-9,
                        dict(parameters=preview, expected_gain=expected_gain, expected_pure=ep, expected_cross=ec))
                    if case == 'S2':
                        remainder = primary['preview']['remainder_um']['overall']['metrics']['vector']['max_um']
                        add(case, prefix+'preview remainder', remainder <= 1e-9, remainder)
                        implied = primary['implied_epsilon']['values']
                        add(case, prefix+'LS implied epsilon finite and negative',
                            all(np.isfinite(v) and v < 0 for v in implied.values()),
                            dict(LS_implied=implied, synthetic_epsilon=-.3, tolerance='none; LS values'))
                        gaze_max = primary['gaze']['overall']['metrics']['vector']['max_arcmin']
                        add(case, prefix+'T1 gaze error nonzero', gaze_max > 0, gaze_max)
                        c = fields[shard+'_'+estimator+'_t1_gaze_err_arcmin']
                        w = fields[shard+'_'+estimator+'_w_deg']
                        phi = np.rad2deg(np.arctan2(abs(w[:, 1]), abs(w[:, 0])))
                        components, nonzero = projected_components(c, w)
                        mask = nonzero & ((phi < 22.5) | (phi > 67.5))
                        signed = float(np.mean(components['radial'][mask]))
                        add(case, prefix+'near-axis radial sign recorded', np.isfinite(signed),
                            dict(mean_arcmin=signed, sign=float(np.sign(signed))))
                    else:
                        difference = preview['epsilon_p_bar_minus_c_bar']
                        add(case, prefix+'pure-cross difference', abs(difference-.03) <= 1e-9, difference)
                elif case == 'S4':
                    alternative = item['T1']['alternative']
                    alt_max = alternative['gaze']['overall']['metrics']['vector']['max_arcmin']
                    primary_max = primary['gaze']['overall']['metrics']['vector']['max_arcmin']
                    add(case, prefix+'alternative convention and inverse', item['T0']['convention_indicator'] == 'alternative' and
                        alt_max <= 1e-9 and primary_max > 0,
                        dict(indicator=item['T0']['convention_indicator'], alternative_max_arcmin=alt_max, primary_max_arcmin=primary_max))
                elif case == 'S5':
                    forbidden = item['T0']['forbidden']
                    corresponding = [entry for entry in forbidden if entry['powers'] == [0, 2] and entry['axis'] == 'wx']
                    maximum = item['T0']['forbidden_max_abs_normalised_deg']
                    add(case, prefix+'quadratic forbidden coefficient listed and nonzero',
                        len(corresponding) == 1 and corresponding[0]['normalised_deg'] != 0 and maximum > 1e-9,
                        dict(corresponding=corresponding, max_normalised_deg=maximum))
    for case in ('S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7'):
        if case not in tests:
            add(case, 'required case exercised', False, 'Earlier numerical failure prevented this case')
    satisfied = all(check['satisfied'] for checks in tests.values() for check in checks)
    result = new_report([], 'selftest', True)
    result.update(status='complete' if satisfied else 'selftest_failed',
                  selftest_checks=tests, synthetic_cases=reports,
                  errors=[] if satisfied else [dict(message='One or more S1-S7 correctness checks not satisfied')],
                  selftest_all_satisfied=satisfied)
    return result, maps, 0 if satisfied else 4


def number(value, digits=17):
    if value is None:
        return 'undefined'
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value)).lower()
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return format(float(value), f'.{digits}g')
    return str(value)


def tree_lines(value, prefix='', digits=17):
    """Human-readable complete diagnostics, preserving dictionary ordering."""
    if isinstance(value, dict):
        lines = []
        for key, child in value.items():
            if isinstance(child, (dict, list, tuple, np.ndarray)):
                lines.append(prefix+str(key)+':')
                lines.extend(tree_lines(child, prefix+'  ', digits))
            else:
                lines.append(prefix+str(key)+' = '+number(child, digits))
        return lines
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        lines = []
        for index, child in enumerate(value):
            if isinstance(child, (dict, list, tuple, np.ndarray)):
                lines.append(prefix+str(index)+':')
                lines.extend(tree_lines(child, prefix+'  ', digits))
            else:
                lines.append(prefix+str(index)+' = '+number(child, digits))
        return lines
    return [prefix+number(value, digits)]


def metric_lines(title, value):
    return [title, PROVISIONAL, *tree_lines(value, '  ', 6)]


def coefficient_lines(record):
    lines = ['term axis allowed normalised_deg raw raw_unit']
    for row in record['coefficient_entries']:
        lines.append(f"{tuple(row['powers'])} {row['axis']} {row['allowed']} "+
                     f"{row['normalised_deg']:.17g} {row['raw']:.17g} {row['raw_unit']}")
    return lines


def structure_lines(structure):
    lines = []
    for field, data in structure.items():
        lines.extend(metric_lines(field+' residual structure ['+data['unit']+']',
            {key: value for key, value in data.items() if key not in ('line_profiles', 'coarse_blocks')}))
        lines.extend(['5x5 coarse blocks: rows ADE, columns BDE', PROVISIONAL])
        for name, matrix in data['coarse_blocks'].items():
            lines.append(name+' ['+data['unit']+']' if name != 'states' else name)
            for row in matrix:
                lines.append('  '+' '.join(number(value, 6) for value in row))
        for name, profile in data['line_profiles'].items():
            lines.extend(['Line profile '+name+'; all 101 points saved, every tenth printed',
                          PROVISIONAL,
                          'state_id ADE_index BDE_index BDE_deg minus_ADE_deg x y radial tangential'])
            for index in range(0, 101, 10):
                values = [profile['state_id'][index], profile['ade_index'][index],
                          profile['bde_index'][index], *profile['w_deg'][index],
                          profile['x'][index], profile['y'][index],
                          profile['radial'][index], profile['tangential'][index]]
                lines.append(' '.join(number(value, 6) for value in values))
    return lines


def text_report(report):
    lines = [LABEL, 'run status: '+report['status'], PROVISIONAL,
             'M8: primary; E1: '+E1_LABEL]
    lines.extend(tree_lines({key: report[key] for key in (
        'tag', 'synthetic', 'indices', 'stages', 'frame', 'definitions', 'inputs', 'errors')}))
    if 'incomplete_stage_results' in report:
        lines.extend(['Incomplete numerical-stage diagnostics (run did not complete)',
                      PROVISIONAL, *tree_lines(report['incomplete_stage_results'])])
    if 'selftest_checks' in report:
        lines.extend(['S1-S7 synthetic correctness checks (code checks only)',
                      *tree_lines(report['selftest_checks'])])
        for case, case_report in report['synthetic_cases'].items():
            lines.extend(['', 'Synthetic case '+case, *text_report(case_report).splitlines()[1:]])
        return '\n'.join(lines)+'\n'
    if 'split_counts' in report:
        lines.extend(tree_lines(dict(split_counts=report['split_counts'])))
    for shard, estimators in report['results'].items():
        for estimator, item in estimators.items():
            lines.extend(['', shard+' '+estimator+' '+item['role'],
                          *tree_lines({key: item[key] for key in ('accommodation_d', 'origin_mm', 'shared_A00_scale_mm')}),
                          'T0: six allowed coefficients and derived quantities', PROVISIONAL,
                          *tree_lines(item['T0']['coefficients']),
                          'T0: full 20 coefficients', *coefficient_lines(item['T0']['full_cubic']),
                          'T0: fourteen forbidden coefficients',
                          *tree_lines(item['T0']['forbidden'])])
            zero = item['T0']
            lines.extend(tree_lines({key: zero[key] for key in (
                'units', 'forbidden_max_abs_normalised_deg', 'forbidden_max_abs_raw',
                'forbidden_raw_max_note', 'forbidden_max_abs_raw_by_degree',
                'oddness', 'own_linear_coefficients', 'D12_over_c_s', 'convention_indicator')}))
            lines.extend(['T0 fit rank/condition/scale diagnostics', PROVISIONAL])
            for name in ('full_cubic', 'odd_cubic', 'own_linear_fit'):
                lines.extend([name, *tree_lines({key: zero[name][key] for key in (
                    'rank', 'condition', 'singular_values', 'scale_mm', 'valid', 'rcond')})])
            lines.extend(tree_lines(dict(sensitivity=zero['sensitivity'])))
            lines.extend(metric_lines('T0 polynomial-side baselines [arcmin]', zero['baselines']))
            if 'reproduction' in zero:
                lines.extend(['T0 rounded-table reproduction (reported only)', PROVISIONAL])
                for name, row in zero['reproduction'].items():
                    if not row['matches_rounded_table']:
                        lines.append('*** ROUNDED TABLE MISMATCH: '+name+'; reported, not gated ***')
                    lines.extend([name, *tree_lines(row)])
            for convention in CONVENTIONS:
                one = item['T1'][convention]
                lines.extend(['T1 '+convention+' ('+one['stage_role']+')', PROVISIONAL,
                              *tree_lines({key: one[key] for key in ('solver', 'diagnostic_axis_gain_solvers',
                                  'diagnostic_linear_only_gains_mm', 'Taylor_definition')}),
                              'quantity unit actual LS_prediction Taylor_prediction gap relative_gap gap/c_s gap/c_s_unit'])
                for row in one['predicted_vs_actual']:
                    lines.append(' '.join(number(row[key]) for key in (
                        'quantity', 'unit', 'actual', 'LS_prediction', 'Taylor_prediction',
                        'gap', 'relative_gap', 'gap_over_c_s', 'gap_over_c_s_unit')))
                lines.extend(['T1 implied epsilons (cross c21,c12; pure c30,c03)',
                              *tree_lines(one['implied_epsilon']),
                              'T1 coefficient-operator full 20 coefficients',
                              *coefficient_lines(one['coefficient_operator'])])
                for metric in ('gaze', 'forward_um', 'forward_arcmin_equivalent', 'E_poly', 'E_close'):
                    lines.extend(metric_lines('T1 '+metric, one[metric]))
            # Primary residual detail is followed by the reported alternative.
            for convention in CONVENTIONS:
                one = item['T1'][convention]
                lines.extend(['Residual structure '+convention,
                              *structure_lines(item['residual_structure'][convention]),
                              'Next-term preview '+convention+' (never sets a stage constant)', PROVISIONAL,
                              *tree_lines({key: val for key, val in one['preview'].items()
                                           if key not in ('remainder_um', 'remainder_arcmin_equivalent')})])
                for metric in ('remainder_um', 'remainder_arcmin_equivalent'):
                    lines.extend(metric_lines('Preview '+metric, one['preview'][metric]))
            lines.extend(metric_lines('Convention comparison', item['convention_comparison']))
    lines.extend(['E1 versus M8; A4 versus A0 when present', PROVISIONAL,
                  *tree_lines(report['comparisons'])])
    lines.extend(['What to look at', *report.get('what_to_look_at', [])])
    return '\n'.join(lines)+'\n'


def write_outputs(tag, report, maps, log):
    """Only these own-tag artifacts may be replaced; outputs are never read."""
    guard_outputs(tag)
    text_path, json_path, map_path, _ = [OUT/name for name in artifacts(tag)]
    text_path.write_text(text_report(report), encoding='utf-8')
    guard_outputs(tag)
    with json_path.open('w', encoding='utf-8') as stream:
        json.dump(p4_a0_data.safe(report), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    guard_outputs(tag)
    require(all(isinstance(array, np.ndarray) and array.dtype.kind in 'biufc'
                for array in maps.values()), 'NPZ must contain numeric arrays only')
    with map_path.open('wb') as stream:
        np.savez_compressed(stream, **maps)
    log('artifacts written; status='+report['status'])


def camera_constants():
    values = (CAMERA_F, CAMERA_K1, CAMERA_K2)
    if values != (joint.F, joint.K1, joint.K2):
        raise NumericalFailure('Fixed camera constants differ from joint',
                               declared=values, joint=(joint.F, joint.K1, joint.K2))
    return dict(F_mm=CAMERA_F, K1=CAMERA_K1, K2=CAMERA_K2,
                kappa1_per_mm2=KAPPA1, kappa2_per_mm4=KAPPA2,
                centre_definition='A00 loader c in mm, fixed for every A; no H scaling')


def radial_inverse_checked(v, a, b, context):
    try:
        recovered, status = joint.radial_inverse(v, a, b)
        if not (np.all(status['converged']) and np.all(status['min_tangential'] > 0)
                and np.all(status['min_radial'] > 0) and np.all(np.isfinite(recovered))):
            raise NumericalFailure(context+': inverse status/branch', status=status)
        return recovered, status
    except ValueError as exc:
        raise NumericalFailure(context+': '+str(exc), a=a, b=b) from exc


def stage_forward(tau, gain, eps, camera):
    if camera not in (None, 'K1', 'K1K2'):
        raise NumericalFailure('Unknown stage camera', camera=camera)
    if not np.isfinite(gain) or gain == 0 or not np.isfinite(eps):
        raise NumericalFailure('Nonfinite/zero stage constant', gain=gain, epsilon=eps)
    try:
        z = gain*joint.radial(tau, eps, 0.)
        model = z if camera is None else joint.radial(z, KAPPA1, 0. if camera == 'K1' else KAPPA2)
        if not np.all(np.isfinite(model)):
            raise NumericalFailure('Nonfinite stage forward', gain=gain, epsilon=eps, camera=camera)
        return model
    except ValueError as exc:
        raise NumericalFailure('Stage forward: '+str(exc), gain=gain, epsilon=eps,
                               camera=camera) from exc


def stage_inverse(m, gain, eps, camera):
    if camera not in (None, 'K1', 'K1K2'):
        raise NumericalFailure('Unknown stage camera', camera=camera)
    if not np.isfinite(gain) or gain == 0 or not np.isfinite(eps):
        raise NumericalFailure('Nonfinite/zero stage inverse constant', gain=gain, epsilon=eps)
    z = m if camera is None else radial_inverse_checked(
        m, KAPPA1, 0. if camera == 'K1' else KAPPA2, 'Stage camera')[0]
    tau, _ = radial_inverse_checked(z/gain, eps, 0., 'Stage lens')
    return tangent_inverse(tau, 1., 'primary')


def stage_branch_record(m, tau, gain, eps, camera):
    """Report the reused inverse's status and both forward branch minima."""
    mt, mr = joint.radial_minima(eps, 0., np.linalg.norm(tau, axis=1))
    z = m
    camera_record = None
    if camera is not None:
        b = 0. if camera == 'K1' else KAPPA2
        z, status = radial_inverse_checked(m, KAPPA1, b, 'Stage camera diagnostics')
        closure = 1000*np.linalg.norm(joint.radial(z, KAPPA1, b)-m, axis=1)
        camera_record = dict(converged_all=bool(np.all(status['converged'])),
            min_tangential=float(np.min(status['min_tangential'])),
            min_radial=float(np.min(status['min_radial'])),
            inverse_iterations=status['iterations'], closure_max_um=float(np.max(closure)))
    recovered, status = radial_inverse_checked(z/gain, eps, 0., 'Stage lens diagnostics')
    return dict(forward_min_tangential=float(np.min(mt)), forward_min_radial=float(np.min(mr)),
        inverse_converged_all=bool(np.all(status['converged'])),
        inverse_min_tangential=float(np.min(status['min_tangential'])),
        inverse_min_radial=float(np.min(status['min_radial'])),
        inverse_iterations=status['iterations'],
        lens_closure_max_tangent=float(np.max(np.linalg.norm(
            joint.radial(recovered, eps, 0.)-z/gain, axis=1))), camera=camera_record)


def operator2(gain, eps, camera, tau, w, train, scale):
    mm = stage_forward(tau, gain, eps, camera)
    fit, _ = checked_cubic(mm[train], w[train], scale, mm, 'T2/T3 coefficient operator')
    return six_coefficients(fit), fit


def solve_stage(actual, camera, tau, w, train, scale, seed_gain, seed_eps):
    """Only a_bar and c21 calibrate a stage; no holdout enters this solve."""
    target, cross, cs = actual['a_bar'], actual['c21'], actual['c_s']
    if not all(np.isfinite(x) for x in (target, cross, cs, seed_gain, seed_eps)) or target == 0 or cs == 0:
        raise NumericalFailure('Invalid T2/T3 solver target/seed', actual=actual,
                               seed_gain=seed_gain, seed_eps=seed_eps)
    gain, epsilon = seed_gain, seed_eps
    history, evaluations = [], 0
    previous = None
    for outer in range(30):
        inner_history = []
        for inner in range(30):
            predicted, fit = operator2(gain, epsilon, camera, tau, w, train, scale)
            evaluations += 1
            residual_a = abs(predicted['a_bar']-target)/abs(target)
            inner_history.append(dict(evaluation=inner+1, G_mm=gain,
                model_a_bar=predicted['a_bar'], relative_residual=residual_a))
            if np.isfinite(residual_a) and residual_a <= 1e-12:
                break
            gain = gain*predicted['a_bar']/target
        else:
            raise NumericalFailure('T2/T3 inner gain solve did not converge', camera=camera,
                outer_history=history, inner_history=inner_history, epsilon=epsilon,
                total_operator_evaluations=evaluations)
        h = (predicted['c21']-cross)/cs
        history.append(dict(outer_step=outer+1, epsilon=epsilon, G_mm=gain,
            c21_residual_over_c_s=h, a_bar_relative_residual=residual_a,
            inner_evaluations=len(inner_history), inner_history=inner_history))
        if not np.isfinite(h):
            raise NumericalFailure('T2/T3 outer residual nonfinite', history=history)
        if abs(h) <= 1e-10:
            final, final_fit = operator2(gain, epsilon, camera, tau, w, train, scale)
            evaluations += 1
            ra = abs(final['a_bar']-target)/abs(target)
            rh = (final['c21']-cross)/cs
            if not np.isfinite(ra+rh) or ra > 1e-12 or abs(rh) > 1e-10:
                raise NumericalFailure('T2/T3 fresh final operator residual', history=history,
                                       a_bar_relative_residual=ra, c21_residual_over_c_s=rh)
            return dict(seed_G_mm=seed_gain, seed_epsilon=seed_eps, G_mm=gain,
                epsilon=epsilon, camera=camera, matched_quantities=['a_bar', 'c21'],
                targets=dict(a_bar=target, c21=cross, c_s=cs),
                outer_iterations=outer+1, total_operator_evaluations=evaluations,
                a_bar_relative_residual=ra, c21_residual_over_c_s=rh,
                inner_tolerance=1e-12, outer_tolerance=1e-10,
                max_inner_evaluations=30, max_outer_steps=30, history=history), final, final_fit
        if previous is None:
            next_epsilon = epsilon-h
        else:
            old_epsilon, old_h = previous
            if h == old_h:
                raise NumericalFailure('T2/T3 secant equal residuals', history=history)
            next_epsilon = epsilon-h*(epsilon-old_epsilon)/(h-old_h)
        previous, epsilon = (epsilon, h), next_epsilon
    raise NumericalFailure('T2/T3 outer solve did not converge', camera=camera,
                           history=history, total_operator_evaluations=evaluations)


def frame_z(prepared, c_cam):
    """Invert all 25 camera points in mm, then select and centre each estimator."""
    observed = prepared['points_mm']
    try:
        z, valid = five.inverse(observed, c_cam, 1.0)
        if not np.all(valid) or not np.all(np.isfinite(z)):
            raise NumericalFailure('Frame Z inverse invalid required point/state',
                                   invalid_count=int(np.count_nonzero(~valid)))
        closure = 1000*np.linalg.norm(joint.camera(z)-(observed-c_cam), axis=-1)
        # five.inverse enforces its existing <1e-9 um camera closure gate.
        if not np.all(np.isfinite(closure)):
            raise NumericalFailure('Frame Z camera closure nonfinite')
    except ValueError as exc:
        raise NumericalFailure('Frame Z: '+str(exc), c_cam_mm=c_cam) from exc
    zero = int(np.flatnonzero(prepared['state']['groups']['zero'])[0])
    means = {}
    for name, key in (('M8', 'E2'), ('E1', 'E1')):
        sensor = z[:, gp.estimator_nodes()[key][1]].mean(axis=1)
        means[name] = dict(m_mm=sensor-sensor[zero], origin_mm=sensor[zero].copy(),
            sensor_origin_relative_to_c_cam_mm=prepared['means'][name]['origin_mm']-c_cam)
    return dict(means=means, closure_max_um=float(np.max(closure)),
                camera_centre_mm=np.asarray(c_cam).copy(), valid_all=True,
                point_order='inverse full camera per point, then original-3x3 estimator mean')


def bin_vector_stats(field, groups):
    answer = {}
    for name in gp.BIN_NAMES:
        values = field[groups[name]]
        if not len(values):
            rms = maximum = None
        elif not np.all(np.isfinite(values)):
            rms = maximum = float('inf')
        else:
            norms = np.linalg.norm(values, axis=1)
            rms, maximum = float(np.sqrt(np.mean(norms**2))), float(np.max(norms))
        answer[name] = dict(states=len(values), vector_rms=rms, vector_max=maximum)
    return answer


def fraction_removed(before_mm, after_mm, groups):
    return {split: dict(T1_sum_squares_mm2=float(np.sum(before_mm[mask]**2)),
        stage_sum_squares_mm2=float(np.sum(after_mm[mask]**2)),
        fraction_removed=ratio(float(np.sum(before_mm[mask]**2)-np.sum(after_mm[mask]**2)),
                               float(np.sum(before_mm[mask]**2))))
        for split in gp.SPLITS for mask in (groups[split],)}


def forward_fit(m, tau, groups, jacobian, order):
    if order not in (2, 3):
        raise NumericalFailure('Forward diagnostic order must be 2 or 3', order=order)
    rho2 = np.sum(tau*tau, axis=1)
    columns = np.stack([tau*rho2[:, None]**k for k in range(order)], axis=-1)
    train = groups['train']
    coefficients, _, rank, singular = np.linalg.lstsq(
        columns[train].reshape(-1, order), m[train].reshape(-1), rcond=gp.RCOND)
    if rank != order or not np.all(np.isfinite(coefficients)) or coefficients[0] == 0:
        raise NumericalFailure('Forward diagnostic rank/finite failure', order=order,
                               rank=int(rank), coefficients=coefficients)
    gain, epsilon = coefficients[0], coefficients[1]/coefficients[0]
    predicted = np.einsum('nik,k->ni', columns, coefficients)
    remainder = 1000*(m-predicted)
    result = dict(use='forward-fit diagnostic only; never calibrates a stage',
        G_mm=gain, epsilon=epsilon, epsilon5=0. if order == 2 else coefficients[2]/gain,
        rank=int(rank), condition=float(singular[0]/singular[-1]), singular_values=singular,
        coefficients_mm=coefficients, rcond=gp.RCOND, training_states=int(train.sum()),
        axes='both axes stacked in one training-only least-squares problem',
        remainder_um=unit_summaries(remainder, groups, 'um'),
        remainder_arcmin_equivalent=unit_summaries(remainder @ jacobian.T*60/1000,
                                                 groups, 'arcmin_equivalent'),
        remainder_by_magnitude_um=bin_vector_stats(remainder, groups),
        remainder_by_magnitude_arcmin_equivalent=bin_vector_stats(
            remainder @ jacobian.T*60/1000, groups))
    if order == 2:
        result['T2_forward_fit_constants'] = gaze_summaries(
            stage_inverse(m, gain, epsilon, None)-tangent_inverse(tau, 1., 'primary'), groups)
    return result, remainder


def t2(actual, fit3, actual_prediction, t1_primary, m, w, groups, scale, tau, metadata, camera=None):
    gain1 = t1_primary['solver']['G1_mm']
    baseline = metadata.get('T2_sensor') if camera is not None else None
    seed_gain = gain1 if baseline is None else baseline['solver']['G_mm']
    seed_eps = actual['c21']/actual['c_s'] if baseline is None else (
        baseline['solver']['epsilon']-KAPPA1*seed_gain**2)
    solver, predicted, model_fit = solve_stage(actual, camera, tau, w, groups['train'],
                                              scale, seed_gain, seed_eps)
    gain, epsilon = solver['G_mm'], solver['epsilon']
    effective_eps = epsilon if camera is None else epsilon+KAPPA1*gain**2
    tay = taylor_coefficients(actual, 'primary')
    for name in ('c30', 'c03', 'c12', 'c21', 'c_p_bar'):
        tay[name] += actual['c_s']*effective_eps
    table = []
    for name in ('a_x', 'a_y', 'c30', 'c03', 'c12', 'c21', 'D12', 'D30', 'D03',
                 'a_x_minus_a_y', 'c30_minus_c03', 'c_p_bar'):
        gap = actual[name]-predicted[name]
        gap1 = actual[name]-t1_primary['LS_predicted_coefficients'][name]
        row = dict(quantity=name, unit=coefficient_unit(name), actual=actual[name],
            T1_LS_prediction=t1_primary['LS_predicted_coefficients'][name],
            LS_prediction=predicted[name], Taylor_prediction=tay[name], gap=gap,
            relative_gap=ratio(gap, actual[name]), gap_over_c_s=gap/actual['c_s'],
            gap_over_c_s_unit='mm^2' if name.startswith('a_') else 'dimensionless',
            gap_closed=None if gap1 == 0 else 1-abs(gap)/abs(gap1), matched=name == 'c21')
        if baseline is not None:
            gap2 = actual[name]-baseline['LS_predicted_coefficients'][name]
            row.update(T2_LS_prediction=baseline['LS_predicted_coefficients'][name],
                       gap_closed_relative_to_T2=None if gap2 == 0 else 1-abs(gap)/abs(gap2))
        table.append(row)
    residual_eps = {name: (actual[name]-predicted[name])/actual['c_s'] for name in EPS_NAMES}
    pure = (residual_eps['c30']+residual_eps['c03'])/2
    residual_eps.update(pure_mean=pure, cross=residual_eps['c12'],
        pure_minus_cross=pure-residual_eps['c12'], pure_split=residual_eps['c30']-residual_eps['c03'])
    model = stage_forward(tau, gain, epsilon, camera)
    gaze = stage_inverse(m, gain, epsilon, camera)
    gaze_error, forward = (gaze-w)*60, (m-model)*1000
    jac = gp.sensitivity(fit3, TERMS3)['gaze_jacobian_deg_per_mm']
    polynomial = gp.predict(fit3, model, TERMS3)
    if not np.all(np.isfinite(polynomial)):
        raise NumericalFailure('T2/T3 actual-polynomial model prediction nonfinite', solver=solver)
    before = m-gain1*tau
    fractions = fraction_removed(before, forward/1000, groups)
    for split in gp.SPLITS:
        preview_fraction = t1_primary['preview']['fraction_of_T1_SSE_removed'][split]['fraction_removed']
        stage_fraction = fractions[split]['fraction_removed']
        fractions[split].update(preview_fraction_removed=preview_fraction,
            share_of_cubic_ceiling=None if stage_fraction is None or preview_fraction in (None, 0)
            else stage_fraction/preview_fraction)
    prior_bins = bin_vector_stats((tangent_inverse(m, gain1, 'primary')-w)*60, groups)
    current_bins = bin_vector_stats(gaze_error, groups)
    bin_comparison = {name: dict(T1_arcmin=prior_bins[name], stage_arcmin=current_bins[name],
        rms_ratio=ratio(current_bins[name]['vector_rms'], prior_bins[name]['vector_rms']),
        max_ratio=ratio(current_bins[name]['vector_max'], prior_bins[name]['vector_max']))
        for name in gp.BIN_NAMES}
    reference_eps = dict(t1_primary['implied_epsilon']['values'])
    reference_eps['T1_implied_mean'] = t1_primary['implied_epsilon']['mean']
    reference_eps.update({key: t1_primary['preview']['parameters'][key]
                         for key in ('epsilon_p_bar', 'epsilon_c_bar')})
    result = dict(stage_role='stage result' if camera != 'K1K2' else 'reported full-camera variant',
        convention='primary', camera=camera, solver=solver, G_over_G1_minus_1=gain/gain1-1,
        epsilon_theta=epsilon+1/3, effective_sensor_epsilon=effective_eps,
        Taylor_seed_c21_over_c_s=actual['c21']/actual['c_s'],
        reference_epsilons=reference_eps,
        epsilon_minus_reference={key: epsilon-value for key, value in reference_eps.items()},
        coefficient_operator=fit_record(model_fit, TERMS3), LS_predicted_coefficients=predicted,
        Taylor_coefficients=tay, Taylor_definition='actual a_bar/c_s and effective sensor epsilon',
        predicted_vs_actual=table, residual_implied_epsilon=residual_eps,
        gaze=gaze_summaries(gaze-w, groups), forward_um=unit_summaries(forward, groups, 'um'),
        forward_arcmin_equivalent=unit_summaries(forward @ jac.T*60/1000, groups, 'arcmin_equivalent'),
        E_poly=gaze_summaries(polynomial-actual_prediction, groups),
        E_close=gaze_summaries(polynomial-w, groups),
        fraction_of_T1_SSE_removed=fractions, gaze_by_magnitude_comparison=bin_comparison,
        branch_and_closure=stage_branch_record(m, tau, gain, epsilon, camera))
    if baseline is not None:
        formula_pre = baseline['solver']['epsilon']-KAPPA1*baseline['solver']['G_mm']**2
        result['sensor_identities'] = dict(
            epsilon2_minus_pre_minus_camera=baseline['solver']['epsilon']-epsilon-KAPPA1*gain**2,
            Gp_over_G2_minus_1=gain/baseline['solver']['G_mm']-1,
            epsilon_pre_minus_formula=epsilon-formula_pre)
    structure = {} if camera == 'K1K2' else dict(
        gaze=residual_structure(gaze_error, w, groups, metadata, 'arcmin'),
        forward=residual_structure(forward, w, groups, metadata, 'um'))
    fields = dict(model_mm=model, gaze_deg=gaze, gaze_err_arcmin=gaze_error, fwd_res_um=forward)
    if camera is None:
        fits = {}
        for name, order in (('F2', 2), ('F3', 3)):
            diagnostic, remainder = forward_fit(m, tau, groups, jac, order)
            diagnostic['fraction_of_T1_SSE_removed'] = fraction_removed(before, remainder/1000, groups)
            fits[name] = diagnostic
            fields[name.lower()+'_remainder_um'] = remainder
        f2 = fits['F2']
        fits['epsilon2_minus_epsilonF2'] = epsilon-f2['epsilon']
        fits['G2_over_GF2_minus_1'] = gain/f2['G_mm']-1
        fits['remainders_side_by_side_um'] = dict(F2=f2['remainder_um'],
            anisotropic_cubic_preview=t1_primary['preview']['remainder_um'], F3=fits['F3']['remainder_um'])
        result['forward_fit_diagnostics'] = fits
    return result, structure, fields


def t3_split(two, actual):
    gain, eps = two['solver']['G_mm'], two['solver']['epsilon']
    eps_cam = KAPPA1*gain**2
    pre = eps-eps_cam
    delta = actual['c_s']*eps_cam
    return dict(epsilon_cam=eps_cam, epsilon_pre_formula=pre, epsilon_theta_pre=pre+1/3,
        camera_share_of_epsilon=ratio(eps_cam, eps), delta_c_cam=delta,
        delta_c_cam_unit='deg/mm^3', Taylor_minus_a_bar_kappa1=-actual['a_bar']*KAPPA1,
        delta_c_cam_over_actual={name: ratio(delta, actual[name]) for name in EPS_NAMES},
        quintic_kappa2_G4=KAPPA2*gain**4, quintic_3_kappa1_G2_epsilon_pre=3*eps_cam*pre,
        use='formula split only; no fit and no new polynomial term')


def camera_identities(two, three, full, ztwo, estimator, H, q):
    g, eps = two['solver']['G_mm'], two['solver']['epsilon']
    gz, ez = ztwo['solver']['G_mm'], ztwo['solver']['epsilon']
    return dict(I1=eps-ez-KAPPA1*gz**2, I2=g/gz-1,
        I2_hand=KAPPA1*H**2*(3+10*q) if estimator == 'M8' else 0.,
        I3={name: item['solver']['epsilon']-ez for name, item in (('K1', three), ('K1K2', full))},
        I4={name: item['solver']['G_mm']/gz-1 for name, item in (('K1', three), ('K1K2', full))},
        meaning='M8 carries ring-camera coupling; E1 is the single-point diagnostic',
        diagnostic_label=PROVISIONAL)


def comparison_for_estimators2(items):
    m8, e1 = items['M8'], items['E1']
    comparison = {}
    for frame, key in (('sensor', 'T2'), ('frame_Z', 'T3Z')):
        a = m8[key] if key == 'T2' else m8[key]['T2']
        b = e1[key] if key == 'T2' else e1[key]['T2']
        comparison[frame] = dict(
            G2_relative_difference=(a['solver']['G_mm']-b['solver']['G_mm'])/b['solver']['G_mm'],
            epsilon2_M8_minus_E1=a['solver']['epsilon']-b['solver']['epsilon'],
            residual_implied_epsilon_M8_minus_E1={name:
                a['residual_implied_epsilon'][name]-b['residual_implied_epsilon'][name]
                for name in ('c12', 'c30', 'c03')})
    comparison['epsilon_cam_M8_minus_E1'] = m8['T3a']['epsilon_cam']-e1['T3a']['epsilon_cam']
    comparison['epsilon_pre_formula_M8_minus_E1'] = (
        m8['T3a']['epsilon_pre_formula']-e1['T3a']['epsilon_pre_formula'])
    comparison['epsilon_pre_M8_minus_E1'] = {camera:
        m8[key]['solver']['epsilon']-e1[key]['solver']['epsilon']
        for camera, key in (('K1', 'T3'), ('K1K2', 'T3_K1K2'))}
    comparison['epsilon_Z_M8_minus_E1'] = (
        m8['T3Z']['T2']['solver']['epsilon']-e1['T3Z']['T2']['solver']['epsilon'])
    return comparison


def accommodation_quantities2(item):
    """The numeric tree whose A differences are reported, with explicit units."""
    two, ztwo = item['T2'], item['T3Z']['T2']
    values = dict(G2_mm=two['solver']['G_mm'], epsilon2=two['solver']['epsilon'],
        epsilon_cam=item['T3a']['epsilon_cam'], epsilon_pre_formula=item['T3a']['epsilon_pre_formula'],
        epsilon_pre=item['T3']['solver']['epsilon'],
        epsilon_pre_K1K2=item['T3_K1K2']['solver']['epsilon'],
        G2_Z_mm=ztwo['solver']['G_mm'], epsilon_Z=ztwo['solver']['epsilon'],
        camera_identities={key: item['camera_identities'][key] for key in ('I1', 'I2', 'I3', 'I4')},
        stages={}, forward_fit_remainders={})
    stages = dict(T1=item['inputs_from_T0_T1']['T1'], T2=two, T3_K1=item['T3'],
                  T3_K1K2=item['T3_K1K2'], T1_Z=item['T3Z']['T1'], T2_Z=ztwo)
    for key, stage in stages.items():
        metrics = stage['gaze']['overall']['metrics']['vector']
        values['stages'][key] = dict(gaze_rms_arcmin=metrics['rms_arcmin'],
            gaze_max_arcmin=metrics['max_arcmin'])
        if 'residual_implied_epsilon' in stage:
            values['stages'][key]['residual_implied_epsilon'] = stage['residual_implied_epsilon']
            values['stages'][key]['coefficient_gaps'] = {
                row['quantity']: row['gap'] for row in stage['predicted_vs_actual']}
    for frame, stage in (('sensor', two), ('frame_Z', ztwo)):
        values['forward_fit_remainders'][frame] = {name: {
            split: {metric: fit['remainder_um'][split]['metrics']['vector'][metric]
                    for metric in ('rms_um', 'max_um')} for split in gp.SPLITS}
            for name, fit in stage['forward_fit_diagnostics'].items() if name in ('F2', 'F3')}
        f2 = stage['forward_fit_diagnostics']['F2']['T2_forward_fit_constants']['overall']['metrics']['vector']
        values['stages']['F2_'+frame] = {key: f2[key] for key in ('rms_arcmin', 'max_arcmin')}
    return values


def comparison_for_accommodation2(results, delta_d=4):
    def subtract(high, low):
        return {key: subtract(high[key], value) for key, value in low.items()} if isinstance(low, dict) else high-low

    def per_d(value):
        return {key: per_d(child) for key, child in value.items()} if isinstance(value, dict) else value/delta_d

    comparison = {}
    for name in ('M8', 'E1'):
        low, high = (accommodation_quantities2(results[shard][name]) for shard in ('A00', 'A20'))
        changes = subtract(high, low)
        comparison[name] = dict(delta_D=delta_d, signed_change=changes, change_per_dioptre=per_d(changes))
    return comparison


def compact_inputs(zero, one):
    return dict(coefficients=zero['coefficients'], c_s=zero['coefficients']['c_s'],
        G1_mm=one['solver']['G1_mm'], forbidden_max_abs_normalised_deg=zero['forbidden_max_abs_normalised_deg'],
        own_polynomial_gaze=zero['baselines'], sensitivity=zero['sensitivity'],
        T1={key: one[key] for key in ('solver', 'LS_predicted_coefficients', 'Taylor_coefficients',
            'predicted_vs_actual', 'implied_epsilon', 'gaze', 'forward_um', 'forward_arcmin_equivalent', 'preview')})


def ladder_rows(item):
    inputs = item['inputs_from_T0_T1']
    actual = inputs['coefficients']
    rows = []

    def append(name, matched, stage, constants=None, polynomial=False, predicted=None):
        gaze = stage['gaze']
        vector = gaze['overall']['metrics']['vector']
        prediction = predicted if predicted is not None else stage.get('LS_predicted_coefficients')
        rows.append(dict(stage=name, polynomial_terms_matched=matched, constants=constants,
            gaze_overall_arcmin=vector,
            gaze_vector_rms_by_magnitude_arcmin={b: gaze['overall']['by_magnitude'][b]['vector']['rms_arcmin']
                                               for b in gp.BIN_NAMES},
            forward_residual_overall_um=None if polynomial else stage['forward_um']['overall']['metrics']['vector'],
            gap_over_c_s=None if polynomial or prediction is None else {
                key: (actual[key]-prediction[key])/actual['c_s'] for key in ('c12', 'c30', 'c03')},
            splits=None if polynomial or prediction is None else {key:
                dict(actual=actual[key], predicted=prediction[key]) for key in ('a_x_minus_a_y', 'c30_minus_c03')}))

    for name, baseline in (('own linear polynomial', 'own_linear'), ('own cubic polynomial', 'own_full_cubic')):
        append(name, 'own polynomial training least squares',
               dict(gaze=inputs['own_polynomial_gaze'][baseline]), polynomial=True)
    one, two, three, full = inputs['T1'], item['T2'], item['T3'], item['T3_K1K2']
    append('T1 sensor', ['a_bar'], one, dict(G_mm=one['solver']['G1_mm']))
    append('T2 sensor', ['a_bar', 'c21'], two,
           dict(G_mm=two['solver']['G_mm'], epsilon=two['solver']['epsilon']))
    f2 = two['forward_fit_diagnostics']['F2']
    append('T2 with F2 constants (diagnostic)', 'forward fit; no inverse coefficients matched',
           dict(gaze=f2['T2_forward_fit_constants'], forward_um=f2['remainder_um']),
           dict(G_mm=f2['G_mm'], epsilon=f2['epsilon']),
           predicted=item['F2_LS_predicted_coefficients'])
    for name, stage in (('T3 K1', three), ('T3 K1K2 (diagnostic)', full)):
        append(name, ['a_bar', 'c21'], stage,
               dict(G_mm=stage['solver']['G_mm'], epsilon_pre=stage['solver']['epsilon']))
    # Z rows use Z coefficients and scale, not the sensor coefficients above.
    actual = item['T3Z']['inputs_from_T0_T1']['coefficients']
    zone, ztwo = item['T3Z']['T1'], item['T3Z']['T2']
    append('T1 frame Z', ['a_bar'], zone, dict(G_mm=zone['solver']['G1_mm']))
    append('T2 frame Z', ['a_bar', 'c21'], ztwo,
           dict(G_mm=ztwo['solver']['G_mm'], epsilon=ztwo['solver']['epsilon']))
    return rows


def what_to_look_at2(document):
    notes = []
    for shard, estimators in document['results'].items():
        for estimator, item in estimators.items():
            two, ztwo = item['T2'], item['T3Z']['T2']
            prefix = shard+' '+estimator+': '
            gaps = {r['quantity']: r['gap_closed'] for r in two['predicted_vs_actual']}
            residual = two['residual_implied_epsilon']
            fits = two['forward_fit_diagnostics']
            maximum = lambda summary: summary['overall']['metrics']['vector']['max_um']
            notes.extend([
                prefix+'T2 gap_closed and predicted splits (two constants carry the cubic?): '+
                    json.dumps(p4_a0_data.safe(dict(gap_closed=gaps,
                        splits={key: two['LS_predicted_coefficients'][key] for key in ('a_x_minus_a_y', 'c30_minus_c03')}))),
                prefix+'Pure/cross residual implied epsilons; compare M8 with E1 for ring anisotropy: '+
                    json.dumps(p4_a0_data.safe(residual)),
                prefix+'Quintic aliasing / mixed terms: F2, anisotropic preview, F3 max remainders [um] = '+
                    ', '.join(number(maximum(summary)) for summary in
                        (fits['F2']['remainder_um'], fits['remainders_side_by_side_um']['anisotropic_cubic_preview'],
                         fits['F3']['remainder_um']))+'; consider T4b or T5 only after review.',
                prefix+'Calibration loss versus form: epsilon2-epsilonF2='+number(fits['epsilon2_minus_epsilonF2'])+
                    '; G2/GF2-1='+number(fits['G2_over_GF2_minus_1'])+'; compare F2-constant gaze row.',
                prefix+'T2 versus T1 and own polynomials: inspect all four bin RMS columns and the outer-bin profiles.',
                prefix+'Camera share, I1, and ring-camera I2 beside hand value: '+
                    json.dumps(p4_a0_data.safe(dict(camera_share=item['T3a']['camera_share_of_epsilon'],
                                                   identities=item['camera_identities']))),
                prefix+'Across-frame gaze errors (camera higher-order content?): '+
                    json.dumps(p4_a0_data.safe(item['gaze_across_frames'])),
                prefix+'Pre-camera M8-E1 offsets: epsilonZ='+number(ztwo['solver']['epsilon'])+
                    '; read estimator comparison in both frames; shape fits stay out of scope.'])
    notes.append('A4 versus A0: '+(json.dumps(p4_a0_data.safe(document['comparisons']['A4_versus_A0']))
        if 'A4_versus_A0' in document['comparisons'] else 'not present in this A00-only run.')+
        ' Compare pre-camera versus camera epsilon changes and residual growth before selecting intermediate A.')
    notes.append('Resolve numerical failures, closure issues and nonzero centre variants before interpretation; '+PROVISIONAL+'.')
    return notes


def analyse_increment2(prepared, report, log):
    constants = camera_constants()
    c_cam = prepared[0]['loader_c_mm']
    document = {LABEL: True, 'status': 'preparing', 'tag': report['tag'],
        'synthetic': report['synthetic'], 'indices': report['indices'], 'stages': ['T0', 'T1', 'T2', 'T3'],
        'diagnostic_label': PROVISIONAL, 'estimators': report['estimators'],
        'inputs': report['inputs'], 'split_counts': report['split_counts'], 'errors': [],
        'camera': dict(**constants, c_cam_mm=c_cam), 'results': {}, 'comparisons': {}, 'ladder': {},
        'definitions': dict(matched=['a_bar', 'c21'], convention='primary only in T2/T3',
            stage='T3 means K1; K1K2, F2/F3, per-shard centre are diagnostics only',
            frame_Z='inverse camera per point in mm about A00 c; then mean and own zero origin',
            scale='A00 training max(abs(m)) per estimator and frame; reused at A20',
            solver='inner relative a_bar <=1e-12, outer abs(c21 gap/c_s)<=1e-10; 30 each',
            gaze='arcmin; vector RMS/P95/max; every state retained',
            forward='um; sensor frame unless explicitly frame Z (pre-camera)',
            F2_F3='training-only both-axis-stacked forward diagnostics; never set stage constants'),
        '_increment1_results': report['results']}
    # Retain partial Increment 2 numbers if a later required stage fails.
    report['_increment2_document'] = document
    maps = dict(ade_index=prepared[0]['metadata']['global_ade_index'],
                bde_index=prepared[0]['metadata']['global_bde_index'], state_id=prepared[0]['state']['state_ids'])
    maps.update({'mask_'+key: value for key, value in prepared[0]['state']['groups'].items()})
    z_scales = {}
    for index in report['indices']:
        shard, data = f'A{index:02d}', prepared[index]
        w, groups = data['state']['w'], data['state']['groups']
        tau = data['tangent']['primary']
        z = frame_z(data, c_cam)
        variant = frame_z(data, data['loader_c_mm']) if np.any(data['loader_c_mm'] != c_cam) else None
        estimators = document['results'][shard] = {}
        document['ladder'][shard] = {}
        for estimator in ('M8', 'E1'):
            log(f'computing {shard} {estimator} T2/T3 and frame Z')
            previous = report['results'][shard][estimator]
            zero, one = previous['T0'], previous['T1']['primary']
            m, scale = data['means'][estimator]['m_mm'], previous['shared_A00_scale_mm']
            item = estimators[estimator] = dict(accommodation_d=data['accommodation_d'],
                role=previous['role'], inputs_from_T0_T1=compact_inputs(zero, one),
                origin_sensor_mm=previous['origin_mm'], shared_A00_scale_mm=scale)
            fit3, prediction = checked_cubic(m[groups['train']], w[groups['train']], scale, m, 'T2 actual')
            two, structure, fields = t2(zero['coefficients'], fit3, prediction, one,
                m, w, groups, scale, tau, data['metadata'])
            item.update(T2=two, T2_residual_structure=structure)
            camera_metadata = dict(data['metadata'], T2_sensor=two)
            three, t3structure, t3fields = t2(zero['coefficients'], fit3, prediction, one,
                m, w, groups, scale, tau, camera_metadata, 'K1')
            item.update(T3=three, T3_residual_structure=t3structure)
            full, _, _ = t2(zero['coefficients'], fit3, prediction, one,
                m, w, groups, scale, tau, camera_metadata, 'K1K2')
            item['T3_K1K2'] = full
            mz = z['means'][estimator]['m_mm']
            if index == 0:
                z_scales[estimator] = np.max(abs(mz[groups['train']]), axis=0)
            scalez = z_scales[estimator]
            zzero, zfit, zprediction = t0(index, estimator+'_Z', mz, w, groups, scalez)
            zone, _, z1fields = t1('primary', zzero['coefficients'], zfit, zprediction,
                zzero['own_linear_coefficients'], mz, w, groups, scalez, tau, data['metadata'])
            ztwo, zstructure, zfields = t2(zzero['coefficients'], zfit, zprediction, zone,
                mz, w, groups, scalez, tau, data['metadata'])
            f2 = two['forward_fit_diagnostics']['F2']
            f2coeff, _ = operator2(f2['G_mm'], f2['epsilon'], None, tau, w, groups['train'], scale)
            item.update(accommodation_d=data['accommodation_d'],
                role=previous['role'], inputs_from_T0_T1=compact_inputs(zero, one),
                origin_sensor_mm=previous['origin_mm'], shared_A00_scale_mm=scale,
                loader_calibration=dict(c_mm=data['loader_c_mm'], H=data['loader_H'], q=data['loader_q'],
                    c_minus_c_cam_mm=data['loader_c_mm']-c_cam),
                T2=two, T2_residual_structure=structure, T3a=t3_split(two, zero['coefficients']),
                T3=three, T3_residual_structure=t3structure, T3_K1K2=full,
                T3Z=dict(inputs_from_T0_T1=compact_inputs(zzero, zone),
                    origin_pre_camera_mm=z['means'][estimator]['origin_mm'],
                    origin_sensor_relative_to_c_cam_mm=z['means'][estimator]['sensor_origin_relative_to_c_cam_mm'],
                    camera_closure_max_um=z['closure_max_um'], shared_A00_scale_mm=scalez,
                    T1=compact_inputs(zzero, zone)['T1'], T2=ztwo, residual_structure=zstructure,
                    forward_unit='um, pre-camera'), F2_LS_predicted_coefficients=f2coeff,
                camera_identities=camera_identities(two, three, full, ztwo, estimator,
                                                    data['loader_H'], data['loader_q']))
            item['gaze_across_frames'] = {name: dict(
                overall=stage['gaze']['overall']['metrics']['vector'],
                by_magnitude={b: stage['gaze']['overall']['by_magnitude'][b]['vector'] for b in gp.BIN_NAMES})
                for name, stage in (('T2_sensor', two), ('T3_K1', three), ('T3_K1K2', full), ('T2_frame_Z', ztwo))}
            if variant is not None:
                vm = variant['means'][estimator]['m_mm']
                vz, vf, vp = t0(index, estimator+'_Z_variant', vm, w, groups, scalez)
                vs, _, _ = solve_stage(vz['coefficients'], None, tau, w, groups['train'], scalez,
                    solve_gain(vz['coefficients']['a_bar'], 'a_bar', tau, w, groups['train'], scalez)[0]['G1_mm'],
                    vz['coefficients']['c21']/vz['coefficients']['c_s'])
                item['centre_variant_per_shard_c'] = dict(
                    use='reported centre diagnostic; never replaces fixed-A00-centre stage numbers',
                    c_mm=data['loader_c_mm'], c_minus_c_cam_mm=data['loader_c_mm']-c_cam,
                    coefficients=vz['coefficients'],
                    forbidden_max_abs_normalised_deg=vz['forbidden_max_abs_normalised_deg'],
                    G2_Z_mm=vs['G_mm'], epsilon_Z=vs['epsilon'])
            document['ladder'][shard][estimator] = ladder_rows(item)
            prefix = shard+'_'+estimator+'_'
            saved = {**{'t2_'+key: fields[key] for key in ('model_mm', 'gaze_deg', 'gaze_err_arcmin', 'fwd_res_um')},
                'f2_remainder_um': fields['f2_remainder_um'], 'f3_remainder_um': fields['f3_remainder_um'],
                **{'t3_'+key: t3fields[key] for key in ('model_mm', 'gaze_err_arcmin', 'fwd_res_um')},
                'z_m_mm': mz, 'z_t1_gaze_err_arcmin': z1fields['t1_gaze_err_arcmin'],
                **{'z_t2_'+key: zfields[key] for key in ('model_mm', 'gaze_err_arcmin', 'fwd_res_um')}}
            maps.update({prefix+key: value for key, value in saved.items()})
        document['comparisons'].setdefault('E1_versus_M8', {})[shard] = comparison_for_estimators2(estimators)
    if 20 in report['indices']:
        delta_d = prepared[20]['accommodation_d']-prepared[0]['accommodation_d']
        document['comparisons']['A4_versus_A0'] = comparison_for_accommodation2(document['results'], delta_d)
    document['status'] = 'complete'
    document['what_to_look_at'] = what_to_look_at2(document)
    return document, maps


def compute_run2(indices, tag, log, loader=five.load_shard, synthetic=False):
    report = new_report(indices, tag, synthetic)
    prepared = prepare_all(indices, report, log, loader, keep_points=True)
    if prepared is None:
        report.update(stages=['T0', 'T1', 'T2', 'T3'], ladder={})
        return report, {}, 2
    try:
        camera_constants()
        analyse_prepared(prepared, report, log)
        document, maps = analyse_increment2(prepared, report, log)
    except (NumericalFailure, np.linalg.LinAlgError, FloatingPointError, ValueError) as exc:
        partial = report.get('_increment2_document', {})
        report['incomplete_stage_results'] = partial.get('results', report['results'])
        report.update(status='NUMERICAL_FAIL', stages=['T0', 'T1', 'T2', 'T3'], ladder={})
        report['errors'].append(dict(message=str(exc), exception=type(exc).__name__,
                                    numbers=exc.numbers if isinstance(exc, NumericalFailure) else {}))
        report['results'] = {}
        log('NUMERICAL_FAIL: '+str(exc))
        return report, {}, 3
    return document, maps, 0


def synthetic_selftests2(log):
    """S1-S7 stay on their unchanged path; S8-S16 are additional code checks."""
    original, _, original_code = synthetic_selftests(log)
    tests = dict(original['selftest_checks'])
    reports, maps = {}, {}
    quiet = lambda *a, **k: None

    def add(case, check, satisfied, numbers):
        tests.setdefault(case, []).append(dict(check=check, satisfied=bool(satisfied), numbers=numbers))

    def max_gaze(stage):
        return max(value['metrics']['vector']['max_arcmin'] for value in stage['gaze'].values())

    def prediction_errors(stage, actual, linear_scale=True):
        return {row['quantity']: abs(row['gap'])/(
            abs(actual['a_bar']) if linear_scale and row['quantity'].startswith('a_') else abs(actual['c_s']))
            for row in stage['predicted_vs_actual']}

    for case in ('S8', 'S9', 'S10', 'S11', 'S12', 'S15'):
        indices = [0, 20] if case in ('S8', 'S9', 'S12', 'S15') else [0]
        law = 'S1' if case in ('S9', 'S15') else case
        log('synthetic '+case)
        document, fields, code = compute_run2(indices, 'selftest', quiet,
            loader=synthetic_loader(law, fail_index=20 if case == 'S15' else None), synthetic=True)
        reports[case] = document
        maps.update({case+'_'+key: value for key, value in fields.items()})
        if case == 'S15':
            add(case, 'missing A20 stops whole run, names shard and computes no T2/T3',
                document['status'] == 'stopped_missing_or_invalid_data' and code == 2 and
                not document['results'] and any(error.get('shard', '').endswith('A20.pkl')
                                               for error in document['errors']),
                dict(status=document['status'], exit_code=code, errors=document['errors']))
            continue
        add(case, 'production Increment 2 analysis completes', code == 0 and document['status'] == 'complete',
            dict(status=document['status'], exit_code=code, errors=document['errors']))
        if code:
            continue
        for shard, estimators in document['results'].items():
            gain = -31.5 if shard == 'A00' else -30.5
            for estimator, item in estimators.items():
                prefix = shard+' '+estimator+' '
                two = item['T2']
                actual = item['inputs_from_T0_T1']['coefficients']
                if case == 'S8':
                    solver, fits = two['solver'], two['forward_fit_diagnostics']
                    add(case, prefix+'T2 gain and epsilon recovery',
                        abs(solver['G_mm']/gain-1) <= 1e-9 and abs(solver['epsilon']+.3) <= 1e-8, solver)
                    errors = prediction_errors(two, actual)
                    add(case, prefix+'every T2 LS prediction', max(errors.values()) <= 1e-8, errors)
                    residual = two['residual_implied_epsilon']
                    add(case, prefix+'residual implied epsilons',
                        max(abs(v) for v in residual.values()) <= 1e-8, residual)
                    add(case, prefix+'gaze inverse every split', max_gaze(two) <= 1e-6, two['gaze'])
                    forward = two['forward_um']['overall']['metrics']['vector']['max_um']
                    add(case, prefix+'forward residual', forward <= 1e-5, forward)
                    add(case, prefix+'F2 constants and F3 quintic zero',
                        abs(fits['F2']['G_mm']-gain) <= 1e-9 and abs(fits['F2']['epsilon']+.3) <= 1e-9 and
                        abs(fits['F3']['epsilon5']) <= 1e-7, fits)
                    add(case, prefix+'both solver residuals within tolerance',
                        solver['a_bar_relative_residual'] <= 1e-12 and abs(solver['c21_residual_over_c_s']) <= 1e-10,
                        dict(a_bar=solver['a_bar_relative_residual'], c21=solver['c21_residual_over_c_s']))
                elif case == 'S9':
                    relative = two['solver']['G_mm']/item['inputs_from_T0_T1']['G1_mm']-1
                    add(case, prefix+'tangent-law epsilon zero and unchanged gain',
                        abs(two['solver']['epsilon']) <= 1e-8 and abs(relative) <= 1e-9,
                        dict(epsilon=two['solver']['epsilon'], G2_over_G1_minus_1=relative))
                elif case == 'S10':
                    solver = two['solver']
                    residual = two['residual_implied_epsilon']
                    add(case, prefix+'anisotropic matched set only',
                        solver['a_bar_relative_residual'] <= 1e-12 and abs(solver['c21_residual_over_c_s']) <= 1e-10,
                        solver)
                    add(case, prefix+'pure residual implied epsilons positive',
                        residual['c30'] > 0 and residual['c03'] > 0,
                        dict(c30=residual['c30'], c03=residual['c03'], hand_pure_minus_cross=.03,
                             tolerance='none on comparison with +0.03'))
                elif case == 'S11':
                    three = item['T3']
                    solver = three['solver']
                    add(case, prefix+'K1 pre-camera constants recovery',
                        abs(solver['G_mm']/gain-1) <= 1e-9 and abs(solver['epsilon']+.33) <= 1e-8, solver)
                    errors = prediction_errors(three, actual, linear_scale=False)
                    add(case, prefix+'K1 coefficient recovery', max(errors.values()) <= 1e-8, errors)
                    add(case, prefix+'K1 gaze inverse', max_gaze(three) <= 1e-6, three['gaze'])
                    share = (two['solver']['epsilon']-solver['epsilon'])/(KAPPA1*solver['G_mm']**2)
                    add(case, prefix+'camera unit and LS-dependent split ratio', .5 <= share <= 1.5, share)
                elif case == 'S12':
                    ztwo, full = item['T3Z']['T2'], item['T3_K1K2']
                    # Recover true commanded w through the injected loader, never through a saved map.
                    metadata = {}
                    synthetic_loader('S12')(SimpleNamespace(shard_index=0 if shard == 'A00' else 20,
                        input_path=gp.accommodation_input(0 if shard == 'A00' else 20)), {}, metadata, quiet)
                    expected = gain*joint.radial(tangent_coordinates(metadata['w_deg'])['primary'], -.33, 0.)
                    mean_error = float(np.max(abs(fields[shard+'_'+estimator+'_z_m_mm']-expected)))
                    add(case, prefix+'per-point frame Z uses fixed A00 centre', mean_error <= 1e-9, mean_error)
                    add(case, prefix+'Z constants and inverse recovery',
                        abs(ztwo['solver']['G_mm'] - gain) <= 1e-9 and
                        abs(ztwo['solver']['epsilon']+.33) <= 1e-8 and max_gaze(ztwo) <= 1e-6, ztwo['solver'])
                    ids = item['camera_identities']
                    add(case, prefix+'full-camera constants and I3/I4 recovery',
                        abs(full['solver']['G_mm'] - gain) <= 1e-8 and
                        abs(full['solver']['epsilon']+.33) <= 1e-8 and
                        abs(ids['I3']['K1K2']) <= 2e-8 and abs(ids['I4']['K1K2']) <= 2e-8,
                        dict(solver=full['solver'], identities=ids))
                    add(case, prefix+'I1 and I2 finite, reported only',
                        np.isfinite(ids['I1']) and np.isfinite(ids['I2']), ids)
                    if shard == 'A20':
                        variant = item.get('centre_variant_per_shard_c')
                        delta = item['loader_calibration']['c_minus_c_cam_mm']
                        add(case, prefix+'A20 centre variant retained and forbidden term exposed',
                            np.allclose(delta, [.003, 0.], rtol=0., atol=1e-15) and variant is not None and
                            variant['forbidden_max_abs_normalised_deg'] > 1e-9,
                            dict(c_minus_c_cam_mm=delta, variant=variant))
    log('synthetic S13')
    grid = np.linspace(-12.5, 12.5, 11)
    x, y = np.meshgrid(grid, grid)
    w = np.column_stack((x.ravel(), y.ravel()))
    tau = tangent_coordinates(w)['primary']
    for camera in (None, 'K1', 'K1K2'):
        try:
            recovered = stage_inverse(stage_forward(tau, -31.5, -.3, camera), -31.5, -.3, camera)
            error = float(np.max(np.linalg.norm((recovered-w)*60, axis=1)))
            add('S13', 'forward/inverse '+str(camera), error <= 1e-9, error)
        except NumericalFailure as exc:
            add('S13', 'forward/inverse '+str(camera), False, dict(message=str(exc), numbers=exc.numbers))
        try:
            stage_inverse(stage_forward(tau, -31.5, -4., camera), -31.5, -4., camera)
        except NumericalFailure as exc:
            add('S13', 'non-monotone branch rejected '+str(camera), True, str(exc))
        else:
            add('S13', 'non-monotone branch rejected '+str(camera), False, 'No NumericalFailure raised')
    log('synthetic S14')
    for law in ('S1', 'S2'):
        baseline, _, code1 = compute_run([0], 'selftest', quiet, loader=synthetic_loader(law), synthetic=True)
        extended, _, code2 = compute_run2([0], 'selftest', quiet, loader=synthetic_loader(law), synthetic=True)
        equal = code1 == code2 == 0 and json.dumps(p4_a0_data.safe(baseline['results'])) == json.dumps(
            p4_a0_data.safe(extended.get('_increment1_results', {})))
        add('S14', law+' identical serialised T0/T1 result trees', equal,
            dict(Increment1_exit=code1, Increment2_exit=code2, identical=equal))
    log('synthetic S16')
    z = np.array([[0., 0.], [.1, -.2], [4., -5.]])
    add('S16', 'kappa1 units, fixed constants, exact joint camera equivalence',
        KAPPA1 == CAMERA_K1/CAMERA_F**2 and abs(KAPPA1/1.62859e-5-1) <= 1e-4 and
        (CAMERA_F, CAMERA_K1, CAMERA_K2) == (joint.F, joint.K1, joint.K2) and
        np.array_equal(joint.radial(z, KAPPA1, KAPPA2), joint.camera(z)),
        dict(F=CAMERA_F, K1=CAMERA_K1, K2=CAMERA_K2, kappa1=KAPPA1, kappa2=KAPPA2))
    for number_ in range(8, 17):
        case = 'S'+str(number_)
        if case not in tests:
            add(case, 'required case exercised', False, 'Earlier numerical failure prevented case')
    satisfied = original_code == 0 and all(check['satisfied'] for checks in tests.values() for check in checks)
    result = {LABEL: True, 'status': 'complete' if satisfied else 'selftest_failed',
        'tag': 'selftest', 'synthetic': True, 'indices': [], 'stages': ['T0', 'T1', 'T2', 'T3'],
        'estimators': {'M8': 'primary', 'E1': E1_LABEL}, 'diagnostic_label': PROVISIONAL,
        'inputs': [], 'errors': [] if satisfied else [dict(message='One or more S1-S16 correctness checks not satisfied')],
        'results': {}, 'comparisons': {}, 'ladder': {},
        'camera': dict(F_mm=CAMERA_F, K1=CAMERA_K1, K2=CAMERA_K2,
                       kappa1_per_mm2=KAPPA1, kappa2_per_mm4=KAPPA2),
        'selftest_checks': tests, 'synthetic_cases': reports, 'selftest_all_satisfied': satisfied,
        'what_to_look_at': ['Synthetic code-correctness checks; scientific quantities remain provisional.']}
    return result, maps, 0 if satisfied else 4


def stage_document2(document, stage):
    """Self-contained projections; the full Increment 1 tree stays in memory."""
    shared = ('status', 'tag', 'synthetic', 'indices', 'stages', 'diagnostic_label', 'estimators',
              'inputs', 'errors', 'split_counts', 'definitions', 'camera', 'selftest_checks',
              'selftest_all_satisfied', 'selftest_precondition', 'what_to_look_at', 'incomplete_stage_results')
    result = {LABEL: True, **{key: document[key] for key in shared if key in document}}
    result['document_stage'] = stage
    result['comparisons'] = document.get('comparisons', {})
    result['results'] = {}
    for shard, estimators in document.get('results', {}).items():
        result['results'][shard] = {}
        for estimator, item in estimators.items():
            if 'inputs_from_T0_T1' not in item:
                continue
            keep = ('accommodation_d', 'role', 'inputs_from_T0_T1', 'origin_sensor_mm',
                    'shared_A00_scale_mm')
            if stage == 'T2':
                keep += ('T2', 'T2_residual_structure')
            elif stage == 'T3':
                keep += ('loader_calibration', 'T2', 'T3a', 'T3', 'T3_residual_structure', 'T3_K1K2',
                         'T3Z', 'camera_identities', 'gaze_across_frames', 'centre_variant_per_shard_c')
            result['results'][shard][estimator] = {key: item[key] for key in keep if key in item}
            if stage == 'ladder' and 'T3Z' in item:
                result['results'][shard][estimator]['inputs_from_T0_T1_frame_Z'] = item['T3Z']['inputs_from_T0_T1']
    if stage == 'ladder':
        result['ladder'] = document.get('ladder', {})
    if 'synthetic_cases' in document:
        result['synthetic_cases'] = {case: stage_document2(case_document, stage)
                                     for case, case_document in document['synthetic_cases'].items()}
    return result


def tree_lines2(value, prefix='', digits=17):
    """Increment 2 text only: six-digit errors, 17-digit constants/coefficients."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (dict, list, tuple)):
        return [prefix+number(value, digits)]
    lines = []
    children = value.items() if isinstance(value, dict) else enumerate(value)
    for key, child in children:
        child_digits = 6 if key in (
            'rms_arcmin', 'p95_arcmin', 'max_arcmin', 'rms_um', 'p95_um', 'max_um',
            'rms_arcmin_equivalent', 'p95_arcmin_equivalent', 'max_arcmin_equivalent',
            'gaze_rms_arcmin', 'gaze_max_arcmin', 'vector_rms', 'vector_max',
            'closure_max_um', 'camera_closure_max_um', 'lens_closure_max_tangent',
            'gaze_vector_rms_by_magnitude_arcmin') else digits
        if isinstance(child, (dict, list, tuple, np.ndarray)):
            lines.append(prefix+str(key)+':')
            lines.extend(tree_lines2(child, prefix+'  ', child_digits))
        else:
            lines.append(prefix+str(key)+' = '+number(child, child_digits))
    return lines


def compact_text2(value, digits=17):
    """Compact text rendering of safe values, without changing JSON data."""
    if isinstance(value, dict):
        return '{'+', '.join(json.dumps(key)+': '+compact_text2(child, digits)
                             for key, child in value.items())+'}'
    if isinstance(value, (list, tuple)):
        return '['+', '.join(compact_text2(child, digits) for child in value)+']'
    return number(value, digits) if isinstance(value, float) else json.dumps(value)


def text_report2(document, stage):
    selected = stage_document2(document, stage)
    lines = [LABEL, 'Increment 2 '+stage, 'status = '+selected['status'], PROVISIONAL,
             'E1: '+E1_LABEL, 'Constants/coefficient precision: 17 significant digits; error metrics: 6 significant digits in declared units.']
    for key, value in selected.items():
        if key in (LABEL, 'results', 'synthetic_cases'):
            continue
        lines.extend([key, PROVISIONAL, *tree_lines2(value, '  ')])
    for shard, estimators in selected['results'].items():
        for estimator, item in estimators.items():
            lines.extend([shard+' '+estimator, PROVISIONAL])
            for key, value in item.items():
                if key.endswith('residual_structure'):
                    lines.extend([key, *structure_lines(value)])
                elif key == 'T3Z':
                    lines.extend([key, PROVISIONAL, *tree_lines2(
                        {k: v for k, v in value.items() if k != 'residual_structure'}, '  ')])
                    lines.extend(structure_lines(value['residual_structure']))
                else:
                    lines.extend([key, PROVISIONAL, *tree_lines2(value, '  ')])
    for case, case_document in document.get('synthetic_cases', {}).items():
        lines.extend(['Synthetic '+case, text_report2(case_document, stage)])
    return '\n'.join(lines)+'\n'


def ladder_text(document):
    selected = stage_document2(document, 'ladder')
    lines = [LABEL, 'Increment 2 ladder; status = '+selected['status'], PROVISIONAL, 'E1: '+E1_LABEL]
    for key in ('inputs', 'camera', 'errors'):
        lines.extend([key, *tree_lines2(selected.get(key, {}), '  ')])
    for shard, estimators in selected.get('ladder', {}).items():
        for estimator, rows in estimators.items():
            lines.extend([shard+' '+estimator, PROVISIONAL,
                'stage | polynomial terms matched | constants [mm, dimensionless] | gaze RMS/P95/max [arcmin] | '+
                'four bin vector RMS [arcmin] | forward RMS/max [um] | c12,c30,c03 gap/c_s | predicted/actual splits'])
            for row in rows:
                lines.append(' | '.join((row['stage'], json.dumps(row['polynomial_terms_matched']),
                    compact_text2(p4_a0_data.safe(row['constants']), 17),
                    compact_text2(p4_a0_data.safe(row['gaze_overall_arcmin']), 6),
                    compact_text2(p4_a0_data.safe(row['gaze_vector_rms_by_magnitude_arcmin']), 6),
                    compact_text2(p4_a0_data.safe(row['forward_residual_overall_um']), 6),
                    compact_text2(p4_a0_data.safe(row['gap_over_c_s']), 17),
                    compact_text2(p4_a0_data.safe(row['splits']), 17))))
    for key in ('selftest_checks', 'synthetic_cases', 'what_to_look_at'):
        if key in selected:
            lines.extend([key, PROVISIONAL, *tree_lines2(selected[key], '  ')])
    return '\n'.join(lines)+'\n'


def write_outputs2(tag, document, maps, log):
    guard_outputs(tag, 'T3')
    paths = [OUT/name for name in artifacts(tag, 'T3')]
    for stage, text_index, json_index in (('ladder', 0, 1), ('T2', 2, 3), ('T3', 4, 5)):
        guard_outputs(tag, 'T3')
        paths[text_index].write_text(ladder_text(document) if stage == 'ladder' else text_report2(document, stage),
                                     encoding='utf-8')
        guard_outputs(tag, 'T3')
        with paths[json_index].open('w', encoding='utf-8') as stream:
            json.dump(p4_a0_data.safe(stage_document2(document, stage)), stream,
                      indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
    guard_outputs(tag, 'T3')
    require(all(isinstance(array, np.ndarray) and array.dtype.kind in 'biufc'
                for array in maps.values()), 'NPZ must contain numeric arrays only')
    with paths[6].open('wb') as stream:
        np.savez_compressed(stream, **maps)
    log('Increment 2 artifacts written; status='+document['status'])


def main_increment2(args, indices, tag):
    log = Progress(tag, 'T3')
    log('starting Increment 2; '+PROVISIONAL)
    synthetic_report, synthetic_maps, synthetic_code = synthetic_selftests2(log)
    if args.selftest:
        write_outputs2(tag, synthetic_report, synthetic_maps, log)
        return synthetic_code
    if synthetic_code:
        report = new_report(indices, tag)
        report.update(status='selftest_failed', stages=['T0', 'T1', 'T2', 'T3'],
            errors=synthetic_report['errors'], selftest_checks=synthetic_report['selftest_checks'],
            selftest_all_satisfied=False, ladder={}, synthetic_cases={})
        write_outputs2(tag, report, {}, log)
        return synthetic_code
    log('S1-S16 synthetic code checks satisfied; beginning strict trace loading')
    document, maps, code = compute_run2(indices, tag, log)
    document['selftest_precondition'] = dict(satisfied=True,
        method='S1-S16 run in process before trace loading; no saved outputs read',
        checks=synthetic_report['selftest_checks'])
    write_outputs2(tag, document, maps, log)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--selftest', action='store_true', help='S1-S7 synthetic only; no trace reads')
    parser.add_argument('--indices', type=int, nargs='+', default=None,
                        help='Increment 1 accepts exactly 0 or 0 20, in that order')
    parser.add_argument('--stages', choices=('T1', 'T3'), default='T1',
                        help='T1: unchanged Increment 1; T3: Increment 2 T0-T3 and S1-S16')
    args = parser.parse_args(argv)
    if args.selftest and args.indices is not None:
        parser.error('--selftest cannot be combined with --indices')
    indices = [0] if args.indices is None else args.indices
    if indices not in ([0], [0, 20]):
        parser.error('Increment 1 accepts only --indices 0 or --indices 0 20')
    tag = 'selftest' if args.selftest else ('a00' if indices == [0] else 'a00_a20')
    print(LABEL, flush=True)
    if args.stages == 'T3':
        return main_increment2(args, indices, tag)
    log = Progress(tag)
    log('starting Increment 1; '+PROVISIONAL)
    synthetic_report, synthetic_maps, synthetic_code = synthetic_selftests(log)
    if args.selftest:
        write_outputs(tag, synthetic_report, synthetic_maps, log)
        return synthetic_code
    if synthetic_code:
        report = new_report(indices, tag)
        report.update(status='selftest_failed', errors=synthetic_report['errors'],
                      selftest_checks=synthetic_report['selftest_checks'], synthetic_cases={})
        write_outputs(tag, report, {}, log)
        return synthetic_code
    log('S1-S7 synthetic code checks satisfied; beginning strict trace loading')
    report, maps, code = compute_run(indices, tag, log)
    report['selftest_precondition'] = dict(satisfied=True,
        method='S1-S7 run in process before trace loading; no saved outputs read',
        checks=synthetic_report['selftest_checks'])
    write_outputs(tag, report, maps, log)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
