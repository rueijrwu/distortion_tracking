r"""P4 A=0 direct sensor-to-gaze polynomials; noise-free design traces; unaudited.

Usage in PowerShell from the project root (execution requires authorisation):
& 'C:\Users\rueijrwu\.pyenv-win-venv\envs\venv\Scripts\python.exe' Script\p4_a0_gazepoly.py --selftest
& 'C:\Users\rueijrwu\.pyenv-win-venv\envs\venv\Scripts\python.exe' Script\p4_a0_gazepoly.py --quick
& 'C:\Users\rueijrwu\.pyenv-win-venv\envs\venv\Scripts\python.exe' Script\p4_a0_gazepoly.py
& 'C:\Users\rueijrwu\.pyenv-win-venv\envs\venv\Scripts\python.exe' Script\p4_a0_gazepoly.py --accommodation-indices 0 5 10 15 20

Default: all 10,201 A00 states, training on the loader's 2,600 even/even
nonzero states. --quick still loads/validates the full A00 shard, then selects
both original gaze indices divisible by 5 (441 states, 120 training). It changes
coverage only, never degree sweep, CV, parity, rank or metric rules. No pilot,
legacy 3x3 pickle, old results or data cache is read. Other accommodation is
read only when --accommodation-indices is explicitly supplied.
--selftest reads no trace data. The default path writes only report.txt,
results.json, maps.npz and progress.log under runs/p4_gazepoly; reruns replace
those artifacts.

Explicit --accommodation-indices selects a separate full-grid E2-only cubic
experiment: full degree 3 is primary (10 terms/axis), odd degree 3 secondary
(6 terms/axis); no CV or degree selection. Indices are unique integers 0..20.
A00 is loaded first, automatically added if omitted, and fixes both sensor-axis
scales for every fit. Each accommodation retains its own zero-gaze origin.
The explicit experiment writes only results_accom.json, results_accom.txt and
progress_accom.log in the same fixed output folder; legacy artifacts are kept.
It cannot be combined with --quick. Synthetic selftests also cover this loop.

Inputs are xsi/ysi sensor mm, selected by field LABEL, never d=-label or local
coordinates. E1 is the unmeasurable traced centre reference; E2/E3/E4 are the
eight-point / four edge-midpoint / four corner means, not ring estimators.
Truth w=(BDE,-ADE) deg comes from the validated loader metadata w_deg.
On the default A00 path, subtract each estimator's zero-gaze position and scale
each sensor axis by its maximum absolute TRAINING displacement. Thus default
training inputs are in [-1,1]; excursions are reported and never clipped. The
explicit sweep uses the A00 scale also for other accommodations and reports
their training/holdout excursions. Within default-path CV the scale is fitted
afresh on that fold's training states. Zero gaze is a calibration reference
only, never a regression or CV target.

For the default path, five deterministic CV folds group entire paired +/-ADE slices (including both
BDE signs) to prevent antipodal pairs crossing folds. Sorted unique absolute
ADE-index offsets from 50 are assigned round-robin to folds. Select minimum
pooled out-of-fold vector RMS, breaking exact ties by lower degree, separately
for each estimator and odd/full family. Holdouts never select a degree/model.
"Same polynomial" means the same basis and degree, with coefficients trained
separately for each estimator; matched-degree E2/E1 comparisons are included.
All degrees are also refitted on all training states. Rank-deficient fits retain
their diagnostic predictions but cannot pass compliance or select a CV degree.
No failed state is removed. Quick's unsampled states remain in loader metadata,
with analysis_index and analyzed_mask explicitly identifying the sampled set.

One process, one BLAS thread, scaled monomials, numpy.linalg.lstsq, fixed
rcond=1e-12, no ridge. Runtime and accuracy are unmeasured until authorised runs.
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

# Set before numpy/project imports; avoid thread-pool overhead for small fits.
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
import p4_a0_data
import p4_a0_ninept5 as five

ROOT = Path(__file__).absolute().parents[1]
OUT = ROOT / 'runs' / 'p4_gazepoly'
INPUT = five.DATA / 'p4_raytrace_5x5_101x101_12p5_A00.pkl'
ARTIFACTS = ('report.txt', 'results.json', 'maps.npz', 'progress.log')
ACCOM_ARTIFACTS = ('results_accom.json', 'results_accom.txt', 'progress_accom.log')
DEGREES = (1, 3, 5, 7, 9, 11, 13)
RCOND = 1e-12
TARGET = 1.0
SPLITS = ('train', 'even_odd', 'odd_even', 'odd_odd', 'zero', 'overall')
BIN_NAMES = ('bin_0_4', 'bin_4_8', 'bin_8_12', 'bin_12_plus')
require = five.require


def accommodation_input(index):
    return five.DATA / f'p4_raytrace_5x5_101x101_12p5_A{index:02d}.pkl'


def output_files(accommodation_sweep):
    return ACCOM_ARTIFACTS[:2] if accommodation_sweep else ('maps.npz', 'results.json', 'report.txt')


class AccommodationIndices(argparse.Action):
    """Distinguish an explicit one-A sweep from the unchanged default A00 run."""
    def __call__(self, parser, namespace, values, option_string=None):
        if any(index < 0 or index > 20 for index in values):
            parser.error('--accommodation-indices must be in 0..20')
        if len(set(values)) != len(values):
            parser.error('--accommodation-indices must not contain duplicates')
        if getattr(namespace, 'accommodation_sweep', False):
            parser.error('--accommodation-indices may be supplied only once')
        setattr(namespace, self.dest, values)
        namespace.accommodation_sweep = True


def guard_outputs():
    """No redirects, reparse points, hard links or paths outside fixed OUT."""
    require(ROOT.resolve() == ROOT, 'Redirected project root')
    targets = [OUT / name for name in (*ARTIFACTS, *ACCOM_ARTIFACTS)]
    for path in [OUT, *OUT.parents, *targets]:
        require(path.resolve() == path.absolute(), 'Redirected output: '+str(path))
        if path.exists() or path.is_symlink():
            info = path.lstat()
            require(not path.is_symlink() and not
                    (getattr(info, 'st_file_attributes', 0) & 0x400), 'Output reparse point')
            if path in targets:
                require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
                        'Output must be a regular file with one link')
            else:
                require(path.is_dir(), 'Output parent is not a directory')
    sources = [INPUT, *(accommodation_input(i) for i in range(21)),
               five.MANIFEST, ROOT/'P4_9PT_PLAN.md',
               *list((ROOT/'Script').glob('*.py'))]
    for i, target in enumerate(targets):
        for source in [*sources, *targets[i+1:]]:
            require(target.resolve() != source.resolve() and not
                    (target.exists() and source.exists() and target.samefile(source)),
                    'Output aliases an input/source/target')


class Progress:
    def __init__(self, accommodation_sweep=False):
        self.filename = 'progress_accom.log' if accommodation_sweep else 'progress.log'
        guard_outputs()
        OUT.mkdir(parents=True, exist_ok=True)
        guard_outputs()
        (OUT/self.filename).write_text('', encoding='utf-8')
        self.started = self.last = time.perf_counter()

    def __call__(self, message, heartbeat=False):
        now = time.perf_counter()
        if heartbeat and now-self.last < 30:
            return
        self.last = now
        line = (datetime.now(timezone.utc).isoformat(timespec='seconds')+
                f' elapsed_s={now-self.started:.3f} '+message)
        print(line, flush=True)
        guard_outputs()
        with (OUT/self.filename).open('a', encoding='utf-8') as stream:
            stream.write(line+'\n')
            stream.flush()


def powers(degree, parity):
    """Graded order, x power descending within each total degree."""
    return tuple((total-j, j) for total in range(degree+1)
                 if parity == 'full' or total % 2 == (1 if parity == 'odd' else 0)
                 for j in range(total+1))


def design(m, terms):
    return np.column_stack([m[:, 0]**i*m[:, 1]**j for i, j in terms])


def fit_polynomial(m_mm, w_deg, terms, scale_mm=None):
    """Arguments contain regression training states ONLY, already zero-centred."""
    require(len(m_mm) > 0 and np.all(np.isfinite(m_mm)) and
            np.all(np.isfinite(w_deg)), 'Invalid training input; no state may be dropped')
    scale = (np.max(abs(m_mm), axis=0) if scale_mm is None else
             np.asarray(scale_mm, dtype=float).copy())
    require(scale.shape == (2,) and np.all(np.isfinite(scale)) and np.all(scale > 0),
            'Degenerate sensor scale')
    a = design(m_mm/scale, terms)
    coef, _, rank, singular = np.linalg.lstsq(a, w_deg, rcond=RCOND)
    valid = int(rank) == len(terms) and np.all(np.isfinite(coef))
    condition = (float(singular[0]/singular[-1])
                 if len(singular) == len(terms) and singular[-1] > 0 else float('inf'))
    return dict(coef=coef, scale=scale, rank=int(rank), condition=condition,
                singular=singular, valid=bool(valid))


def predict(fit, m_mm, terms):
    # Nonfinite holdout inputs propagate to failure metrics, never to a fit.
    return design(m_mm/fit['scale'], terms) @ fit['coef']


def metric(error_deg, mask, model_valid=True):
    e = error_deg[mask]*60
    n = len(e)
    failed = ~np.all(np.isfinite(e), axis=1)
    answer = dict(states=n, nonfinite_states=int(failed.sum()),
                  model_valid=bool(model_valid), target_max_le_1_arcmin=None)
    # Do not produce converged-only statistics: any failure poisons the group.
    if not n:
        answer.update(vector=None, wx=None, wy=None)
        return answer
    if np.any(failed) or not model_valid:
        values = np.full((3, 3), np.inf)
    else:
        v = np.column_stack((np.linalg.norm(e, axis=1), abs(e)))
        values = np.vstack((np.sqrt(np.mean(v*v, axis=0)),
                            np.percentile(v, 95, axis=0), np.max(v, axis=0)))
    for k, name in enumerate(('vector', 'wx', 'wy')):
        answer[name] = dict(zip(('rms_arcmin', 'p95_arcmin', 'max_arcmin'),
                                map(float, values[:, k])))
    answer['target_max_le_1_arcmin'] = bool(values[2, 0] <= TARGET)
    return answer


def summaries(error_deg, groups, valid):
    """Every split, and every split crossed with magnitude bins/corner extremes."""
    corners = [name for name in groups if name.startswith('extreme_')]
    return {name: dict(metrics=metric(error_deg, groups[name], valid),
                       by_magnitude={b: metric(error_deg, groups[name] & groups[b], valid)
                                     for b in BIN_NAMES},
                       corner_extremes={c: metric(error_deg, groups[name] & groups[c], valid)
                                        for c in corners},
                       four_corner_extremes=metric(error_deg,
                           groups[name] & groups['four_corner_extremes'], valid))
            for name in SPLITS}


def cv_folds(ade_index, train):
    """Group paired ADE slices; no antipodal or same-ADE states cross folds."""
    offsets = abs(ade_index[train]-50)
    unique = np.unique(offsets)
    require(len(unique) >= 5, 'Insufficient ADE groups for five-fold CV')
    fold = np.searchsorted(unique, offsets) % 5
    require(all(np.any(fold == k) and np.any(fold != k) for k in range(5)),
            'Empty CV training or validation fold')
    return fold, unique


def cross_validate(m_train, w_train, terms, fold):
    out_of_fold = np.full_like(w_train, np.nan)
    trials = []
    for k in range(5):
        validation = fold == k
        fit = fit_polynomial(m_train[~validation], w_train[~validation], terms)
        prediction = predict(fit, m_train[validation], terms)
        out_of_fold[validation] = prediction
        error = prediction-w_train[validation]
        trials.append(dict(fold=k, fit_states=int((~validation).sum()),
                           validation_states=int(validation.sum()), rank=fit['rank'],
                           condition_number=fit['condition'], scale_mm=fit['scale'],
                           valid=fit['valid'], metrics=metric(error, np.ones(len(error), bool),
                                                           fit['valid'])))
    valid = all(trial['valid'] for trial in trials)
    pooled = metric(out_of_fold-w_train, np.ones(len(w_train), bool), valid)
    return dict(folds=trials, pooled=pooled,
                selection_rms_arcmin=pooled['vector']['rms_arcmin'], valid=valid), out_of_fold


def sensitivity(fit, terms):
    # Higher monomials have zero derivative at the zero-gaze origin.
    jac = np.stack([fit['coef'][terms.index(term)]/fit['scale'][axis]
                    for axis, term in enumerate(((1, 0), (0, 1)))], axis=1)
    answer = dict(gaze_jacobian_deg_per_mm=jac,
                  definition='Inverse fitted linear term at zero gaze; sensor um per arcmin',
                  reference_only='Centre approximately 9 um/arcmin; not a fitted constraint')
    if not fit['valid'] or not np.all(np.isfinite(jac)) or np.linalg.matrix_rank(jac) < 2:
        answer.update(valid=False, sensor_um_per_arcmin_matrix=None,
                      axis_um_per_arcmin=None, principal_um_per_arcmin=None)
    else:
        inverse = np.linalg.inv(jac)*1000/60
        answer.update(valid=True, sensor_um_per_arcmin_matrix=inverse,
                      axis_um_per_arcmin=np.linalg.norm(inverse, axis=0),
                      principal_um_per_arcmin=np.linalg.svd(inverse, compute_uv=False))
    return answer


def oddness(fit, terms, m_train):
    defect = predict(fit, -m_train, terms)+predict(fit, m_train, terms)
    maximum = float(np.max(abs(defect)))
    return dict(definition='max absolute component of f(-m)+f(m), m=training sensor minus zero',
                max_abs_deg=maximum, max_abs_arcmin=60*maximum,
                per_axis_max_abs_arcmin=60*np.max(abs(defect), axis=0),
                tolerance_deg=1e-9, below_tolerance=bool(maximum <= 1e-9))


def estimator_nodes():
    """Reuse loader label order, locate 3x3 nodes by label, never by position."""
    labels = five.LABELS[five.NINE]
    require(np.array_equal(labels, five.nine.LABELS) and len(np.unique(labels, axis=0)) == 9,
            'Nine-label adapter disagrees with loader')
    centre = np.flatnonzero(np.all(labels == (0, 0), axis=1))
    require(len(centre) == 1, 'No unique centre label')
    outer = np.flatnonzero(np.any(labels != 0, axis=1))
    edges = np.flatnonzero(np.sum(abs(labels), axis=1) == 1)
    corners = np.flatnonzero(np.all(abs(labels) == 1, axis=1))
    require((len(outer), len(edges), len(corners)) == (8, 4, 4) and
            np.array_equal(outer, np.flatnonzero(five.nine.OUTER)), '3x3 subset mismatch')
    return {
        'E1': ('centre point (reference only; not measurable)', five.NINE[centre]),
        'E2': ('mean of eight outer points (measurable)', five.NINE[outer]),
        'E3': ('mean of four edge midpoints (measurable; different mean)', five.NINE[edges]),
        'E4': ('mean of four corners (measurable; different mean)', five.NINE[corners])}


def analyze(observed, inherited, metadata, report, maps, log, quick):
    require(len(observed) == 10201 and metadata['w_deg'].shape == (10201, 2),
            'Full A00 loader coverage required')
    expected = dict(train=2600, even_odd=2550, odd_even=2550, odd_odd=2500, zero=1)
    require(all(int(inherited[k].sum()) == n for k, n in expected.items()), 'Loader split counts')
    ga, gb = metadata['global_ade_index'], metadata['global_bde_index']
    selected = np.flatnonzero((ga % 5 == 0) & (gb % 5 == 0)) if quick else np.arange(10201)
    stride = 5 if quick else 1
    wanted = np.array([a*101+b for a in range(0, 101, stride) for b in range(0, 101, stride)])
    require(np.array_equal(metadata['global_state_id'][selected], wanted), 'Stride lattice coverage')
    # Reuse loader splits/zero/extreme masks, replacing only its historical bin edges.
    groups = {name: mask[selected].copy() for name, mask in inherited.items()
              if not name.startswith('bin_')}
    w = metadata['w_deg'][selected]
    radius = np.linalg.norm(w, axis=1)
    groups.update(bin_0_4=radius < 4, bin_4_8=(radius >= 4) & (radius < 8),
                  bin_8_12=(radius >= 8) & (radius < 12), bin_12_plus=radius >= 12)
    extremes = [groups[k] for k in groups if k.startswith('extreme_')]
    require(len(extremes) == 4 and all(int(m.sum()) == 1 for m in extremes), 'Four exact corners')
    groups['four_corner_extremes'] = np.logical_or.reduce(extremes)
    train = groups['train']
    zero = np.flatnonzero(groups['zero'])
    require(len(zero) == 1 and int(train.sum()) == (120 if quick else 2600), 'Training/reference count')
    zero = int(zero[0])
    maps.update(metadata)
    maps['analysis_index'] = selected
    maps['analyzed_mask'] = np.isin(np.arange(10201), selected)
    maps.update({'analysis_group_'+k: v for k, v in groups.items()})
    maps['analysis_w_deg'] = w
    report['coverage'] = dict(loaded_states=10201, analyzed_states=len(selected), stride=stride,
        unsampled_by_design=10201-len(selected), full_grid=not quick,
        group_counts={name: int(mask.sum()) for name, mask in groups.items()},
        loaded_input_failures=int(metadata['input_failed'].sum()),
        analyzed_input_failures=int(metadata['input_failed'][selected].sum()),
        array_convention='Loader maps have 10201 rows; analysis_* and model maps use analysis_index order')
    if np.any(metadata['input_failed'][selected]):
        raise ValueError('Invalid required input states retained in maps; analysis refused without dropping them')
    fold, offsets = cv_folds(ga[selected], train)
    maps['analysis_cv_fold'] = np.full(len(selected), -1, dtype=np.int16)
    maps['analysis_cv_fold'][train] = fold
    report['cv'] = dict(grouping='entire paired +/-ADE slices, sorted unique |ade_index-50| mod 5',
        absolute_ade_index_offsets=offsets,
        validation_states=[int((fold == k).sum()) for k in range(5)],
        selection='minimum pooled OOF vector RMS arcmin, exact ties by lower degree; training only',
        scale='recomputed from each CV fit subset; zero-gaze origin fixed calibration reference')
    report['estimators'] = {}
    for name, (description, node_ids) in estimator_nodes().items():
        sensor = observed[selected][:, node_ids].mean(axis=1)
        origin = sensor[zero].copy()
        m = sensor-origin
        require(np.all(np.isfinite(m)), 'Invalid sensor state; no state may be dropped')
        opposite = np.searchsorted(metadata['global_state_id'][selected],
                                   10200-metadata['global_state_id'][selected])
        require(np.array_equal(metadata['global_state_id'][selected][opposite],
                               10200-metadata['global_state_id'][selected]), 'Antipodal lookup')
        result = dict(description=description, labels=five.LABELS[node_ids],
                      node_indices=node_ids, origin_mm=origin,
                      training_sensor_antisymmetry_max_um=float(1000*np.max(abs((m+m[opposite])[train]))),
                      families={})
        report['estimators'][name] = result
        maps[name+'_sensor_mm'] = sensor
        for parity in ('odd', 'full'):
            family = dict(degree_sweep={}, selected_degree=None)
            result['families'][parity] = family
            for degree in DEGREES:
                start = time.perf_counter()
                terms = powers(degree, parity)
                cv, oof = cross_validate(m[train], w[train], terms, fold)
                fit = fit_polynomial(m[train], w[train], terms)
                prediction = predict(fit, m, terms)
                error = prediction-w
                parity_check = oddness(fit, terms, m[train])
                valid = fit['valid'] and (parity != 'odd' or parity_check['below_tolerance'])
                entry = dict(degree=degree, coefficient_count_per_axis=len(terms),
                    coefficient_count_total=2*len(terms), rank=fit['rank'],
                    condition_number=fit['condition'], rcond=RCOND,
                    valid=bool(valid), status='full_rank' if valid else 'rank_or_parity_failure',
                    prediction_nonfinite_states=int((~np.all(np.isfinite(prediction), axis=1)).sum()),
                    scale_mm=fit['scale'], cv=cv, oddness=parity_check,
                    input_max_abs_scaled_by_axis=np.max(abs(m/fit['scale']), axis=0),
                    states_outside_training_scale=int(np.any(abs(m/fit['scale']) > 1+1e-12, axis=1).sum()),
                    sensitivity=sensitivity(fit, terms), groups=summaries(error, groups, valid),
                    training_paired_gaze_prediction_sum_max_arcmin=float(
                        60*np.max(abs((prediction+prediction[opposite])[train]))),
                    elapsed_s=time.perf_counter()-start)
                family['degree_sweep'][str(degree)] = entry
                prefix = f'{name}_{parity}_d{degree}'
                maps[prefix+'_powers'] = np.asarray(terms, dtype=np.int16)
                maps[prefix+'_coef_deg'] = fit['coef']
                maps[prefix+'_scale_mm'] = fit['scale']
                maps[prefix+'_singular_values'] = fit['singular']
                maps[prefix+'_prediction_deg'] = prediction
                maps[prefix+'_error_arcmin'] = 60*error
                maps[prefix+'_oof_prediction_deg'] = oof
                log(f'{name} {parity} d={degree} rank={fit["rank"]}/{len(terms)} '
                    f'cv_rms_arcmin={cv["selection_rms_arcmin"]:.8g} seconds={entry["elapsed_s"]:.3f}')
            eligible = [d for d in DEGREES if family['degree_sweep'][str(d)]['valid']
                        and family['degree_sweep'][str(d)]['cv']['valid']
                        and np.isfinite(family['degree_sweep'][str(d)]['cv']['selection_rms_arcmin'])]
            if eligible:
                chosen = min(eligible, key=lambda d: (
                    family['degree_sweep'][str(d)]['cv']['selection_rms_arcmin'], d))
                family['selected_degree'] = chosen
                chosen_result = family['degree_sweep'][str(chosen)]
                # A holdout numerical failure cannot reopen training-only selection.
                family['selected_numerical_gates_passed'] = bool(
                    chosen_result['prediction_nonfinite_states'] == 0)
                family['selected_target_pass_on_sampled_states'] = chosen_result['groups']['overall']['metrics'][
                    'target_max_le_1_arcmin']
                family['full_grid_target_pass'] = bool(not quick and
                    family['selected_target_pass_on_sampled_states'])
            else:
                family.update(selection_failure='No degree with five valid full-rank CV fits',
                              selected_numerical_gates_passed=False,
                              selected_target_pass_on_sampled_states=False, full_grid_target_pass=False)
    comparisons(report)
    families = [f for e in report['estimators'].values() for f in e['families'].values()]
    missing = any(f['selected_degree'] is None for f in families)
    numerical = all(f['selected_numerical_gates_passed'] for f in families)
    report['status'] = ('incomplete_model_selection' if missing else
                        ('completed' if numerical else 'failed_selected_predictions'))
    report['numerical_gates_passed'] = numerical


def comparisons(report):
    """Descriptive comparisons only; never choose an estimator using holdouts."""
    result = {}
    for parity in ('odd', 'full'):
        families = {k: e['families'][parity] for k, e in report['estimators'].items()}
        selected = {k: f['selected_degree'] for k, f in families.items()}
        item = dict(training_cv_selected_degrees=selected, selected_metrics={},
                    same_degree_eight_mean_vs_centre={})
        for k, d in selected.items():
            if d is not None:
                item['selected_metrics'][k] = {
                    split: families[k]['degree_sweep'][str(d)]['groups'][split]['metrics']
                    for split in SPLITS}
        for d in DEGREES:
            item['same_degree_eight_mean_vs_centre'][str(d)] = {
                split: {k: families[k]['degree_sweep'][str(d)]['groups'][split]['metrics']
                        for k in ('E1', 'E2')} for split in SPLITS}
        result[parity] = item
    report['comparisons'] = dict(
        scope='Descriptive sampled-state errors; no holdout-based estimator or degree selection',
        target='Euclidean gaze max <= 1 arcmin, including zero and all required holdout states',
        results=result)


def accommodation_states(observed, inherited, metadata):
    """Require every full-grid state; use the existing E2 field-label adapter."""
    require(observed.shape == (10201, 25, 2) and
            metadata['w_deg'].shape == (10201, 2), 'Full accommodation coverage required')
    expected = dict(train=2600, even_odd=2550, odd_even=2550, odd_odd=2500, zero=1,
                    overall=10201)
    require(all(inherited[k].shape == (10201,) and inherited[k].dtype.kind == 'b' and
                int(inherited[k].sum()) == n for k, n in expected.items()),
            'Accommodation loader split counts')
    ga, gb = np.indices((101, 101)).reshape(2, -1)
    require(np.array_equal(metadata['global_state_id'], np.arange(10201)) and
            np.array_equal(metadata['global_ade_index'], ga) and
            np.array_equal(metadata['global_bde_index'], gb), 'Full accommodation state ordering')
    zero_mask = (ga == 50) & (gb == 50)
    canonical = dict(train=(ga % 2 == 0) & (gb % 2 == 0) & ~zero_mask,
        even_odd=(ga % 2 == 0) & (gb % 2 == 1),
        odd_even=(ga % 2 == 1) & (gb % 2 == 0),
        odd_odd=(ga % 2 == 1) & (gb % 2 == 1), zero=zero_mask,
        overall=np.ones(10201, dtype=bool))
    require(all(np.array_equal(inherited[k], mask) for k, mask in canonical.items()),
            'Full accommodation split membership')
    require(metadata['input_failed'].shape == (10201,) and
            not np.any(metadata['input_failed']) and np.all(np.isfinite(observed)) and
            np.all(np.isfinite(metadata['w_deg'])),
            'Invalid required accommodation input; no state may be dropped')
    groups = {name: mask.copy() for name, mask in inherited.items()
              if not name.startswith('bin_')}
    w = metadata['w_deg']
    radius = np.linalg.norm(w, axis=1)
    groups.update(bin_0_4=radius < 4, bin_4_8=(radius >= 4) & (radius < 8),
                  bin_8_12=(radius >= 8) & (radius < 12), bin_12_plus=radius >= 12)
    extremes = [groups[k] for k in groups if k.startswith('extreme_')]
    require(len(extremes) == 4 and all(int(mask.sum()) == 1 for mask in extremes),
            'Accommodation four exact corners')
    groups['four_corner_extremes'] = np.logical_or.reduce(extremes)
    description, node_ids = estimator_nodes()['E2']
    sensor = observed[:, node_ids].mean(axis=1)
    zero = int(np.flatnonzero(groups['zero'])[0])
    require(np.array_equal(w[zero], np.zeros(2)) and not groups['train'][zero],
            'Zero gaze must be calibration only')
    origin = sensor[zero].copy()
    return dict(sensor=sensor, origin=origin, centered=sensor-origin, w=w, groups=groups,
                state_ids=metadata['global_state_id'].copy(), node_ids=node_ids,
                description=description)


def coefficient_trend(diopters, values, degree, models_valid=True):
    """Fit actual manifest diopters, with explicit unavailable diagnostics."""
    a = np.asarray(diopters, dtype=float)
    v = np.asarray(values, dtype=float)
    matrix = np.column_stack([a**power for power in range(degree+1)])
    rank = int(np.linalg.matrix_rank(matrix))
    result = dict(degree=degree, input='actual manifest accommodation_d (diopters)',
                  powers_d=list(range(degree+1)), states=len(a),
                  distinct_diopters=int(len(np.unique(a))), rank=rank, rcond=RCOND,
                  available=False, parameters=None, fitted_values=None, residuals=None,
                  rms_residual=None, max_abs_residual=None,
                  residual_definition='observed coefficient minus fitted coefficient')
    if not models_valid:
        result['reason'] = 'At least one contributing per-accommodation model is invalid'
    elif len(np.unique(a)) < degree+1:
        result['reason'] = f'Need at least {degree+1} distinct accommodation diopters'
    elif rank < degree+1:
        result['reason'] = 'Accommodation trend design is rank deficient'
    elif not np.all(np.isfinite(v)):
        result['reason'] = 'Nonfinite coefficient values'
    else:
        parameters, _, fitted_rank, _ = np.linalg.lstsq(matrix, v, rcond=RCOND)
        result['rank'] = int(fitted_rank)
        if fitted_rank != degree+1 or not np.all(np.isfinite(parameters)):
            result['reason'] = 'Accommodation trend least-squares rank or numerical failure'
            return result
        fitted = matrix @ parameters
        residuals = v-fitted
        result.update(available=True, reason=None, parameters=parameters, fitted_values=fitted,
                      residuals=residuals, rms_residual=float(np.sqrt(np.mean(residuals**2))),
                      max_abs_residual=float(np.max(abs(residuals))))
    return result


def coefficient_series(diopters, values, models_valid):
    values = np.asarray(values, dtype=float)
    delta = values-values[0]
    return dict(values=values, signed_delta_vs_A00=delta, absolute_delta_vs_A00=abs(delta),
        relative_absolute_delta_vs_A00=(None if values[0] == 0 else abs(delta)/abs(values[0])),
        relative_delta_reason=('Baseline coefficient is exactly zero; ratio unavailable'
                               if values[0] == 0 else None),
        trends={name: coefficient_trend(diopters, values, degree, models_valid)
                for name, degree in (('linear', 1), ('quadratic', 2))})


def accommodation_coefficients(states, report):
    indices = report['accommodation_selection']['analyzed_indices']
    diopters = [states[index]['diopters'] for index in indices]
    result = dict(accommodation_indices=indices, accommodation_d=diopters,
        axis_order=['wx', 'wy'],
        relative_delta_definition='abs(c(A)-c(A00))/abs(c(A00)); null only when baseline is exactly zero; no epsilon',
        trend_definition='c(A)=p0+p1*A (linear) or p0+p1*A+p2*A**2 (quadratic); A is actual manifest diopters',
        interpretation='Descriptive coefficient variation; trends do not choose a model or establish a physical cause',
        families={})
    for parity in ('full', 'odd'):
        terms = powers(3, parity)
        valid = all(report['accommodations'][f'A{index:02d}']['families'][parity]['valid']
                    for index in indices)
        families = {}
        for representation, key in (('raw_mm', 'raw_coefficients_deg_per_mm_power'),
                                     ('shared_normalized', 'normalized_coefficients_deg')):
            rows = []
            for term_index, (i, j) in enumerate(terms):
                for axis_index, axis in enumerate(('wx', 'wy')):
                    values = [report['accommodations'][f'A{index:02d}']['families'][parity][key][
                              term_index, axis_index] for index in indices]
                    rows.append(dict(power=[i, j], axis=axis,
                        coefficient_units=(f'deg/mm**{i+j}' if representation == 'raw_mm' else 'deg'),
                        trend_parameter_units='coefficient units / diopter**power',
                        **coefficient_series(diopters, values, valid)))
            families[representation] = rows
        result['families'][parity] = families
    report['coefficient_comparison'] = result


def accommodation_transfer(states, report):
    """Hold A00 coefficients and scale fixed; distinguish the origin policies."""
    baseline = states[0]
    result = dict(source_accommodation_index=0, source_accommodation_d=baseline['diopters'],
        shared_scale_mm=report['shared_scale_mm'],
        primary='target_recentered: target E2 zero-gaze origin, A00 coefficients and scale',
        secondary='frozen_A00_origin: A00 E2 origin, coefficients and scale',
        interpretation='Compare with target own fit at the same degree and parity. Nonzero transfer error alone does not establish that accommodation must be a model input.',
        target='vector gaze max <= 1 arcmin on all 10201 states, including zero', targets={})
    for index in report['accommodation_selection']['analyzed_indices']:
        state = states[index]
        target = dict(accommodation_d=state['diopters'], origin_mm=state['origin'],
                      origin_delta_vs_A00_mm=state['origin']-baseline['origin'], families={})
        for parity in ('full', 'odd'):
            fit, terms = baseline['fits'][parity], powers(3, parity)
            valid = report['accommodations']['A00']['families'][parity]['valid']
            entry = {}
            for label, origin in (('target_recentered', state['origin']),
                                  ('frozen_A00_origin', baseline['origin'])):
                prediction = predict(fit, state['sensor']-origin, terms)
                finite = bool(np.all(np.isfinite(prediction)))
                entry[label] = dict(origin_mm=origin,
                    prediction_nonfinite_states=int((~np.all(np.isfinite(prediction), axis=1)).sum()),
                    groups=summaries(prediction-state['w'], state['groups'], valid and finite))
            own = report['accommodations'][f'A{index:02d}']['families'][parity]
            entry['matched_target_own_fit_groups'] = own['groups']
            entry['overall_target_comparison'] = dict(
                target_recentered_A00_pass=entry['target_recentered']['groups']['overall']['metrics'][
                    'target_max_le_1_arcmin'],
                frozen_A00_origin_pass=entry['frozen_A00_origin']['groups']['overall']['metrics'][
                    'target_max_le_1_arcmin'],
                matched_target_own_fit_pass=own['groups']['overall']['metrics']['target_max_le_1_arcmin'])
            target['families'][parity] = entry
        result['targets'][f'A{index:02d}'] = target
    report['A00_cross_application'] = result


def run_accommodation(report, requested_indices, log, loader=None):
    """The production multi-shard loop; injectable loader is synthetic-test only."""
    requested = list(requested_indices)
    require(requested and all(type(index) is int and 0 <= index <= 20 for index in requested)
            and len(set(requested)) == len(requested), 'Invalid accommodation index selection')
    indices = [0]+[index for index in requested if index != 0]
    report['accommodation_selection'] = dict(requested_indices=requested, analyzed_indices=indices,
        baseline_added=0 not in requested,
        baseline_added_reason=('A00 is required to fix the shared scale and coefficient/transfer baseline'
                               if 0 not in requested else None),
        order='A00 first, then remaining requested indices in supplied order',
        diopters_source='Validated loader input.manifest_entry.accommodation_d; not inferred from index')
    report['accommodations'] = {}
    report['coefficient_definitions'] = dict(axis_order=['wx', 'wy'], sensor_axis_order=['xsi', 'ysi'],
        raw='w_axis=sum c_raw[i,j,axis]*(sensor_x-origin_x)**i*(sensor_y-origin_y)**j; output deg',
        normalized='w_axis=sum c_norm[i,j,axis]*((sensor_x-origin_x)/sx)**i*((sensor_y-origin_y)/sy)**j; output deg',
        conversion='c_raw[i,j,axis]=c_norm[i,j,axis]/(sx**i*sy**j)',
        raw_units='deg/mm**(i+j)', normalized_units='deg',
        scale='sx,sy are max absolute A00 TRAINING displacements, reused unchanged in all fits',
        origins='Per-accommodation E2 zero-gaze mean; zero state excluded from all fits',
        primary='full degree 3: 10 terms/axis, 20 total',
        secondary='odd degree 3: 6 terms/axis, 12 total', selection='No CV or degree selection')
    load = five.load_shard if loader is None else loader
    states = {}
    for index in indices:
        key = f'A{index:02d}'
        entry = dict(status='loading', calibration={}, timings={})
        report['accommodations'][key] = entry
        metadata = {}
        loader_args = SimpleNamespace(quick=False, shard_index=index,
                                      input_path=accommodation_input(index))
        log(f'accommodation {key}: loading full published shard')
        observed, _, _, _, inherited = load(loader_args, entry, metadata, log)
        diopters = entry['input']['manifest_entry']['accommodation_d']
        require(isinstance(diopters, (int, float)) and np.isfinite(diopters),
                'Invalid manifest accommodation diopters')
        if index == 0:
            require(diopters == 0, 'A00 baseline must have zero manifest diopters')
        state = accommodation_states(observed, inherited, metadata)
        del observed
        if states:
            require(np.array_equal(state['state_ids'], states[0]['state_ids']) and
                    np.array_equal(state['w'], states[0]['w']) and
                    all(np.array_equal(state['groups'][name], states[0]['groups'][name])
                        for name in state['groups']) and
                    set(state['groups']) == set(states[0]['groups']),
                    'Accommodation gaze coordinates, state IDs or split masks disagree with A00')
        else:
            scale = np.max(abs(state['centered'][state['groups']['train']]), axis=0)
            require(np.all(np.isfinite(scale)) and np.all(scale > 0), 'Invalid A00 shared scale')
            report['shared_scale_mm'] = scale.copy()
        state.update(diopters=float(diopters), fits={})
        states[index] = state
        entry.update(accommodation_index=index, accommodation_d=float(diopters), estimator='E2',
            description=state['description'], labels=five.LABELS[state['node_ids']],
            node_indices=state['node_ids'], origin_mm=state['origin'],
            origin_delta_vs_A00_mm=state['origin']-states[0]['origin'],
            coverage=dict(loaded_states=10201, analyzed_states=10201, input_failures=0,
                          full_grid=True, zero_is_fit_target=False,
                          group_counts={name: int(mask.sum()) for name, mask in state['groups'].items()}),
            families={})
        start = time.perf_counter()
        for parity in ('full', 'odd'):
            terms = powers(3, parity)
            m, w, groups = state['centered'], state['w'], state['groups']
            fit = fit_polynomial(m[groups['train']], w[groups['train']], terms,
                                 scale_mm=report['shared_scale_mm'])
            state['fits'][parity] = fit
            prediction = predict(fit, m, terms)
            parity_check = oddness(fit, terms, m[groups['train']])
            finite = bool(np.all(np.isfinite(prediction)))
            valid = bool(fit['valid'] and finite and
                         (parity != 'odd' or parity_check['below_tolerance']))
            denominators = np.array([fit['scale'][0]**i*fit['scale'][1]**j for i, j in terms])
            entry['families'][parity] = dict(degree=3,
                role='primary' if parity == 'full' else 'secondary',
                powers=terms, axis_order=['wx', 'wy'], coefficient_count_per_axis=len(terms),
                coefficient_count_total=2*len(terms), normalized_coefficients_deg=fit['coef'],
                raw_coefficients_deg_per_mm_power=fit['coef']/denominators[:, None],
                scale_mm=fit['scale'], rank=fit['rank'], condition_number=fit['condition'],
                singular_values=fit['singular'], rcond=RCOND, valid=valid,
                status='full_rank' if valid else 'rank_parity_or_prediction_failure',
                prediction_nonfinite_states=int((~np.all(np.isfinite(prediction), axis=1)).sum()),
                input_max_abs_scaled_by_axis=np.max(abs(m/fit['scale']), axis=0),
                states_outside_A00_training_scale=int(np.any(abs(m/fit['scale']) > 1+1e-12, axis=1).sum()),
                oddness=parity_check, sensitivity=sensitivity(fit, terms),
                groups=summaries(prediction-w, groups, valid))
            log(f'accommodation {key} E2 {parity} d=3 rank={fit["rank"]}/{len(terms)} '
                f'overall_max_arcmin={entry["families"][parity]["groups"]["overall"]["metrics"]["vector"]["max_arcmin"]:.8g}')
        entry['timings']['fit_and_metrics_s'] = time.perf_counter()-start
        entry['status'] = ('completed' if all(f['valid'] for f in entry['families'].values())
                           else 'failed_model_gates')
    accommodation_coefficients(states, report)
    accommodation_transfer(states, report)
    report['numerical_gates_passed'] = all(
        entry['status'] == 'completed' for entry in report['accommodations'].values())
    report['status'] = 'completed' if report['numerical_gates_passed'] else 'failed_model_gates'


def accommodation_selftest(log):
    """Exercise the production loop without file loading or artifact writes."""
    loaded = []
    ga, gb = np.indices((101, 101)).reshape(2, -1)
    w = np.column_stack(((gb-50)/4, -(ga-50)/4))
    zero = (ga == 50) & (gb == 50)
    groups = dict(overall=np.ones(10201, dtype=bool),
        train=(ga % 2 == 0) & (gb % 2 == 0) & ~zero,
        even_odd=(ga % 2 == 0) & (gb % 2 == 1),
        odd_even=(ga % 2 == 1) & (gb % 2 == 0),
        odd_odd=(ga % 2 == 1) & (gb % 2 == 1), zero=zero)
    for a, b in ((0, 0), (0, 100), (100, 0), (100, 100)):
        groups[f'extreme_{a}_{b}'] = (ga == a) & (gb == b)
    e2 = estimator_nodes()['E2'][1]
    offsets = five.LABELS*np.array([.013, -.017])
    # The eight E2 offsets average to zero; other nodes deliberately do not.
    offsets[~np.isin(np.arange(25), e2)] += np.array([.19, -.23])

    def synthetic_state(index):
        a = index/5
        sx, sy, cubic = 1.8+.12*a, 2.1-.08*a+.025*a*a, .002+.0007*a
        mx = w[:, 0]/sx
        my = (w[:, 1]-cubic*mx**3)/sy
        m = np.column_stack((mx, my))
        origin = np.array([.3+.04*a, -.25+.03*a*a])
        return a, sx, sy, cubic, m, origin

    def synthetic_loader(args, entry, metadata, unused_log):
        require(args.quick is False and args.input_path == accommodation_input(args.shard_index),
                'Synthetic loop did not request the published full shard')
        loaded.append(args.shard_index)
        a, _, _, _, m, origin = synthetic_state(args.shard_index)
        observed = (m+origin)[:, None, :]+offsets[None, :, :]
        metadata.update(global_state_id=np.arange(10201), global_ade_index=ga.copy(),
                        global_bde_index=gb.copy(), w_deg=w.copy(),
                        input_failed=np.zeros(10201, dtype=bool))
        entry['input'] = dict(path=str(args.input_path), profile='synthetic_full5',
            manifest_entry=dict(accommodation_index=args.shard_index, accommodation_d=a),
            states=10201, input_failures=0, complete=True,
            note='Analytic synthetic loader; no trace data or manifest read')
        entry['calibration']['synthetic_origin_mm'] = origin
        entry['timings']['synthetic_loader_s'] = 0.
        return observed, None, None, None, {name: mask.copy() for name, mask in groups.items()}

    result = dict(calibration={}, timings={}, errors=[])
    run_accommodation(result, [5, 10, 15, 20], log, loader=synthetic_loader)
    require(loaded == [0, 5, 10, 15, 20] and result['numerical_gates_passed'] and
            result['accommodation_selection']['baseline_added'] and
            bool(result['accommodation_selection']['baseline_added_reason']) and
            result['accommodation_selection']['requested_indices'] == [5, 10, 15, 20],
            'Synthetic multi-shard order or automatic baseline failed')
    expected_scale = np.max(abs(synthetic_state(0)[4][groups['train']]), axis=0)
    require(np.max(abs(result['shared_scale_mm']-expected_scale)) < 1e-12,
            'Synthetic A00 training scale failed')
    maximum_raw_error = maximum_equivalence_error = maximum_delta_error = 0.
    for index in loaded:
        a, sx, sy, cubic, m, origin = synthetic_state(index)
        entry = result['accommodations'][f'A{index:02d}']
        require(entry['coverage']['group_counts']['train'] == 2600 and
                entry['coverage']['group_counts']['zero'] == 1 and
                not entry['coverage']['zero_is_fit_target'] and
                np.max(abs(entry['origin_mm']-origin)) < 1e-12 and
                np.array_equal(entry['node_indices'], e2), 'Synthetic E2/origin/zero policy failed')
        for parity, count in (('full', 20), ('odd', 12)):
            terms = powers(3, parity)
            fit = entry['families'][parity]
            expected = np.zeros((len(terms), 2))
            expected[terms.index((1, 0)), 0] = sx
            expected[terms.index((0, 1)), 1] = sy
            expected[terms.index((3, 0)), 1] = cubic
            raw = fit['raw_coefficients_deg_per_mm_power']
            raw_error = float(np.max(abs(raw-expected)))
            maximum_raw_error = max(maximum_raw_error, raw_error)
            require(fit['coefficient_count_total'] == count and
                    np.array_equal(fit['scale_mm'], result['shared_scale_mm']) and
                    fit['valid'] and raw_error < 1e-9 and
                    fit['groups']['overall']['metrics']['vector']['max_arcmin'] < 1e-7,
                    'Synthetic cubic recovery or shared scale failed')
            # Independent probes check both coefficient representations, not just fits.
            probe = np.column_stack((2.7*np.sin(np.arange(83)*1.7),
                                     3.1*np.cos(np.arange(83)*2.3)))
            raw_prediction = design(probe, terms) @ raw
            scaled_prediction = design(probe/result['shared_scale_mm'], terms) @ fit[
                'normalized_coefficients_deg']
            equivalence = float(np.max(abs(raw_prediction-scaled_prediction)))
            maximum_equivalence_error = max(maximum_equivalence_error, equivalence)
            require(equivalence < 1e-11 and
                    np.max(abs(design(m, terms) @ raw-w)) < 1e-9,
                    'Synthetic raw/shared-normalized predictions disagree')
            comparisons = result['coefficient_comparison']['families'][parity]
            expected_changes = {(1, 0, 'wx'): .12*a,
                                (0, 1, 'wy'): -.08*a+.025*a*a,
                                (3, 0, 'wy'): .0007*a}
            position = loaded.index(index)
            for row in comparisons['raw_mm']:
                wanted = expected_changes.get((*row['power'], row['axis']), 0.)
                delta_error = abs(row['signed_delta_vs_A00'][position]-wanted)
                maximum_delta_error = max(maximum_delta_error, float(delta_error))
                require(delta_error < 1e-9 and
                        abs(row['absolute_delta_vs_A00'][position]-abs(wanted)) < 1e-9,
                        'Synthetic coefficient deltas failed')
                if row['values'][0] != 0:
                    require(row['relative_absolute_delta_vs_A00'][position] ==
                            abs(row['signed_delta_vs_A00'][position])/abs(row['values'][0]),
                            'Synthetic relative coefficient delta failed')
            for representation in comparisons.values():
                for row in representation:
                    trend = row['trends']['quadratic']
                    require(trend['available'] and trend['rank'] == 3 and
                            trend['max_abs_residual'] < 1e-8,
                            'Synthetic quadratic accommodation trend failed')
    for parity in ('full', 'odd'):
        rows = result['coefficient_comparison']['families'][parity]['raw_mm']
        for term, axis, name, expected in (
                ((1, 0), 'wx', 'linear', [1.8, .12]),
                ((0, 1), 'wy', 'quadratic', [2.1, -.08, .025]),
                ((3, 0), 'wy', 'linear', [.002, .0007])):
            row = next(row for row in rows if row['power'] == list(term) and row['axis'] == axis)
            trend = row['trends'][name]
            require(trend['available'] and np.max(abs(trend['parameters']-expected)) < 1e-9 and
                    trend['max_abs_residual'] < 1e-9, 'Synthetic known trend parameters failed')
        transfer = result['A00_cross_application']['targets']['A20']['families'][parity]
        require(transfer['target_recentered']['groups']['overall']['metrics']['vector']['max_arcmin'] > 1 and
                transfer['target_recentered']['groups']['zero']['metrics']['vector']['max_arcmin'] < 1e-7 and
                transfer['frozen_A00_origin']['groups']['zero']['metrics']['vector']['max_arcmin'] > 1 and
                transfer['matched_target_own_fit_groups']['overall']['metrics']['vector']['max_arcmin'] < 1e-7,
                'Synthetic nontrivial transfer or origin-drift control failed')
    require(coefficient_series([0., 1.], [0., 1.], True)[
            'relative_absolute_delta_vs_A00'] is None, 'Exactly zero baseline needs null ratio')
    one = dict(calibration={}, timings={}, errors=[])
    run_accommodation(one, [0], log, loader=synthetic_loader)
    require(loaded == [0, 5, 10, 15, 20, 0] and one['numerical_gates_passed'] and
            not one['accommodation_selection']['baseline_added'], 'Synthetic one-A sweep failed')
    for family in one['coefficient_comparison']['families'].values():
        for representation in family.values():
            for row in representation:
                require(all(not trend['available'] and trend['rank'] == 1 and
                            trend['parameters'] is None and bool(trend['reason'])
                            for trend in row['trends'].values()),
                        'Synthetic insufficient-A trend must be explicitly unavailable')
    require(set(ACCOM_ARTIFACTS).isdisjoint(ARTIFACTS) and
            output_files(True) == ('results_accom.json', 'results_accom.txt'),
            'Accommodation artifact overlap')
    log('selftest accommodation loop: five shards, baseline, coefficients, trends and transfer passed')
    return dict(passed=True, synthetic_only=True, no_trace_loads=True,
        requested_indices=[5, 10, 15, 20], analyzed_indices=[0, 5, 10, 15, 20],
        loader_call_indices=loaded, shared_scale_mm=result['shared_scale_mm'],
        maximum_raw_coefficient_error=maximum_raw_error,
        maximum_raw_normalized_prediction_difference_deg=maximum_equivalence_error,
        maximum_coefficient_delta_error=maximum_delta_error,
        one_accommodation_trends='unavailable with rank and reason; valid per-A fits retained',
        artifact_policy='Analysis loop performs no writes; explicit sweep writer uses only accommodation artifacts',
        transfer_checks='Nonzero A00 transfer, own-fit recovery, target zero recentering and frozen-origin drift')


def run_selftest(report, maps, log):
    """Nonzero known coefficients, independent evaluation points, adversarial parity."""
    axis = np.linspace(-1, 1, 25)
    m = np.array([(x, y) for y in axis for x in axis])
    checks = []
    for degree in DEGREES:
        terms = powers(degree, 'odd')
        require(len(terms) == (degree+1)*(degree+3)//4 and
                all((i+j) % 2 == 1 and i+j <= degree for i, j in terms),
                'Odd basis count/parity mismatch')
        require(len(powers(degree, 'full')) == (degree+1)*(degree+2)//2,
                'Full basis coefficient count mismatch')
        index = np.arange(2*len(terms)).reshape(-1, 2)
        truth = (0.1+0.003*index)*np.where(index % 3 == 0, -1., 1.)
        w = design(m, terms) @ truth
        for parity in ('odd', 'full'):
            fitted_terms = powers(degree, parity)
            expected = np.zeros((len(fitted_terms), 2))
            for j, term in enumerate(terms):
                expected[fitted_terms.index(term)] = truth[j]
            fit = fit_polynomial(m, w, fitted_terms)
            # Scale is exactly (1,1); coefficients have the same known units.
            relative = float(np.max(abs(fit['coef']-expected))/np.max(abs(truth)))
            nonzero = expected != 0
            element_relative = float(np.max(abs((fit['coef']-expected)[nonzero]/expected[nonzero])))
            probe = np.column_stack((0.97*np.sin(np.arange(83)*1.7),
                                     0.93*np.cos(np.arange(83)*2.3)))
            defect = float(np.max(abs(predict(fit, probe, fitted_terms)-design(probe, terms) @ truth)))
            check = dict(degree=degree, parity=parity, valid=fit['valid'],
                         coefficient_relative_error=relative,
                         nonzero_coefficient_max_relative_error=element_relative,
                         independent_prediction_max_deg=defect,
                         condition_number=fit['condition'])
            checks.append(check)
            require(fit['valid'] and relative <= 1e-9 and element_relative <= 1e-9 and
                    defect < 1e-9, 'Synthetic coefficient recovery failed: '+str(check))
            if parity == 'odd':
                require(oddness(fit, fitted_terms, m)['below_tolerance'], 'Synthetic oddness failed')
        wrong_terms = powers(degree, 'even')
        wrong = fit_polynomial(m, w, wrong_terms)
        wrong_rms = float(60*np.sqrt(np.mean(np.sum((predict(wrong, m, wrong_terms)-w)**2, axis=1))))
        signal_rms = float(60*np.sqrt(np.mean(np.sum(w*w, axis=1))))
        require(signal_rms > 1 and wrong_rms > 1 and wrong_rms > .9*signal_rms,
                'Wrong-parity control was not demonstrably wrong')
        checks.append(dict(degree=degree, control='deliberately even basis on nonzero odd truth',
                           wrong_rms_arcmin=wrong_rms, signal_rms_arcmin=signal_rms))
        maps[f'selftest_d{degree}_truth_coef'] = truth
        log(f'selftest degree={degree} coefficient recovery and wrong-parity control passed')
    # Exercise actual CV code on paired gaze slices, including fold-local scaling.
    ga, gb = np.indices((21, 21)).reshape(2, -1)*5
    w = np.column_stack(((gb-50)/4, -(ga-50)/4))
    train = (ga % 2 == 0) & (gb % 2 == 0) & np.any(w != 0, axis=1)
    fold, offsets = cv_folds(ga, train)
    mt = w[train]/12.5
    terms = powers(3, 'odd')
    truth = np.arange(1, 2*len(terms)+1, dtype=float).reshape(-1, 2)/10
    synthetic_w = design(mt, terms) @ truth
    cv, oof = cross_validate(mt, synthetic_w, terms, fold)
    require(cv['valid'] and cv['selection_rms_arcmin'] < 1e-8 and np.all(np.isfinite(oof)),
            'Grouped CV recovery failed')
    for offset in offsets:
        require(len(np.unique(fold[abs(ga[train]-50) == offset])) == 1, 'CV group split')
    rank_limited = fit_polynomial(mt[:4], synthetic_w[:4], powers(13, 'full'))
    require(not rank_limited['valid'], 'Underdetermined fit incorrectly passed')
    e = np.array([[.05, 1/15], [np.nan, 0.]])
    require(abs(metric(e, np.array([True, False]))['vector']['max_arcmin']-5) < 1e-12,
            'Vector error units wrong')
    require(np.isinf(metric(e, np.ones(2, bool))['vector']['rms_arcmin']), 'Failure was dropped')
    require(metric(e, np.zeros(2, bool))['vector'] is None, 'Empty group incorrectly passed')
    linear = fit_polynomial(m, m/.54, powers(1, 'odd'))
    require(np.max(abs(sensitivity(linear, powers(1, 'odd'))['axis_um_per_arcmin']-9)) < 1e-10,
            'Sensitivity units wrong')
    report['selftest'] = dict(passed=True, checks=checks, grouped_cv=cv,
        additional_checks=['CV group integrity', 'rank failure cannot pass',
                           'vector arcmin units', 'failed state retained as Infinity',
                           'empty group unavailable', '9 um/arcmin sensitivity units'])
    report['selftest']['passed'] = False
    report['selftest']['accommodation_loop'] = accommodation_selftest(log)
    report['selftest']['passed'] = True
    report['status'] = 'selftest_passed'
    report['numerical_gates_passed'] = True


def initial_report(args):
    report = dict(status='started', honesty='noise-free design traces; unaudited',
        mode='selftest' if args.selftest else ('quick_stride5' if args.quick else 'full'),
        scope='P4, A=0 only; ground-truth gaze; direct inverse polynomials; no missing-centre estimation',
        definitions=dict(gaze='w=(BDE,-ADE) deg from loader metadata w_deg',
            sensor='xsi,ysi mm, zero-gaze estimator position subtracted; scaled monomials',
            polynomial='x^i y^j, i+j<=d; odd: i+j odd, full control: all totals including constant',
            same_polynomial='Same basis/degree, independently trained coefficients; matched-degree E2/E1 reported',
            errors='60*(prediction-truth); vector Euclidean norm, per-axis absolute error; equal state weights',
            rms='sqrt(mean squared vector norm), axes sqrt(mean squared axis error)',
            bins='[0,4), [4,8), [8,12), [12,infinity) degrees |w|; exact four gaze corners separately',
            failures='Never remove failed states; any required failure makes group metrics Infinity',
            oddness='Algebraic f(-m)+f(m) check on training inputs, plus actual antipodal-state diagnostic',
            target='gaze vector max <= 1 arcmin; quick cannot establish full-grid compliance'),
        settings=dict(degrees=DEGREES, rcond=RCOND, ridge=False, process_count=1,
                      blas_threads=1, output=str(OUT), numpy_version=np.__version__),
        references=dict(scope='Supplied previous run, DIFFERENT estimator: eight points + model; '
                              'training states of joint run; not recomputed, not matched holdout evidence',
                        G2=dict(rms_arcmin=.035, max_arcmin=.154),
                        G3=dict(rms_arcmin=.021, max_arcmin=.162),
                        G4=dict(rms_arcmin=.022, max_arcmin=.058)),
        calibration={}, timings={}, errors=[], numerical_gates_passed=False)
    if args.accommodation_sweep:
        report['mode'] = 'selftest_accommodation' if args.selftest else 'accommodation_sweep'
        report['scope'] = ('E2 eight-point mean only; full-grid accommodation sweep; '
                           'full cubic primary, odd cubic secondary; no CV or degree selection')
        report['definitions']['same_polynomial'] = 'Same cubic basis/degree and fixed A00 sensor scale; independently fitted at each accommodation'
        report['definitions']['oddness'] = 'Algebraic f(-m)+f(m) check on training inputs; diagnostic only for the primary full cubic'
        report['definitions']['target'] = 'gaze vector max <= 1 arcmin on all 10201 states; evidence for coefficient transfer, not a unique physical explanation'
        report['settings'].update(degrees=[3], estimator='E2',
            families=['full', 'odd'], degree_selection=False, cross_validation=False,
            accommodation_indices=args.accommodation_indices,
            artifacts=ACCOM_ARTIFACTS)
        # Previous A00 model-family references do not describe this experiment.
        report.pop('references')
    return report


def text_report(report):
    lines = ['P4 A=0 direct gaze polynomials', 'noise-free design traces; unaudited',
             'E1: centre reference only, not measurable. E2: eight-point mean.',
             'E3: four edge-midpoint mean. E4: four corner mean. Different means, not ring estimators.',
             'Supplied reference, DIFFERENT estimator, 8 points + model; joint-run training states:',
             'G2 0.035/0.154 arcmin RMS/max; G3 0.021/0.162; G4 0.022/0.058 (not recomputed).',
             'Target: vector gaze max <= 1 arcmin. Ground truth: w=(BDE,-ADE) degrees.',
             'Degrees selected separately by training-only grouped CV; full sweep follows.']
    for name, estimator in report.get('estimators', {}).items():
        for parity, family in estimator['families'].items():
            d = family['selected_degree']
            lines.append(f'{name} {parity}: CV-selected degree={d}; '+estimator['description'])
            if d is not None:
                chosen = family['degree_sweep'][str(d)]
                for split in SPLITS:
                    m = chosen['groups'][split]['metrics']
                    lines.append(f'  {split} n={m["states"]} vector arcmin RMS/P95/max: '+
                                 str(m['vector']))
                lines.append('  local sensitivity axis um/arcmin: '+
                             str(chosen['sensitivity']['axis_um_per_arcmin']))
                lines.append('  sampled target pass='+str(family['selected_target_pass_on_sampled_states'])+
                             '; full-grid target pass='+str(family['full_grid_target_pass']))
    # A complete readable record includes every sweep, CV fold, axis, bin and corner.
    lines.append('\nComplete definitions and results (JSON; nonfinite values encoded as strings):')
    lines.append(json.dumps(p4_a0_data.safe(report), indent=2, allow_nan=False))
    return '\n'.join(lines)+'\n'


def accommodation_text_report(report):
    """Compact tables; exhaustive split/axis/bin diagnostics remain in JSON."""
    def numbers(values):
        return '['+', '.join(f'{value:.7g}' for value in values)+']'

    def trend_text(trend):
        if not trend['available']:
            return f'unavailable(rank={trend["rank"]}; {trend["reason"]})'
        return (numbers(trend['parameters'])+
                f' residual_RMS/max={trend["rms_residual"]:.3g}/{trend["max_abs_residual"]:.3g}')

    lines = ['P4 E2 accommodation cubic comparison',
             'noise-free design traces; unaudited; status='+report['status'],
             'Full d3 primary: 10 terms/axis (20 total). Odd d3 secondary: 6/axis (12 total).',
             'No CV or degree selection. Zero gaze calibrates each E2 mean; it is excluded from fitting.',
             'Truth w=(BDE,-ADE) deg; errors in arcmin. Existing target: vector max <= 1 arcmin.']
    if 'selftest' in report:
        test = report['selftest']
        lines.extend(['Synthetic selftest passed='+str(test['passed'])+'; no trace data read.',
            'Legacy coefficient/CV/parity/units checks retained.',
            'Accommodation loop: '+str(test.get('accommodation_loop', 'incomplete; see errors'))])
    selection = report.get('accommodation_selection')
    if selection:
        lines.extend(['Requested indices: '+str(selection['requested_indices']),
                      'Analyzed indices (A00 first): '+str(selection['analyzed_indices']),
                      'A00 automatically added: '+str(selection['baseline_added'])+
                      ('; '+selection['baseline_added_reason'] if selection['baseline_added'] else '')])
    if 'shared_scale_mm' in report:
        lines.append('Fixed A00 training scale [xsi,ysi] mm: '+numbers(report['shared_scale_mm']))
    entries = report.get('accommodations', {})
    if entries:
        lines.extend(['', 'Calibration / coverage',
            'A | manifest D | E2 origin xsi,ysi mm | loaded/analyzed | train/even_odd/odd_even/odd_odd/zero'])
    for key, entry in entries.items():
        if 'coverage' not in entry:
            lines.append(key+' | '+entry['status'])
            continue
        counts = entry['coverage']['group_counts']
        lines.append(f'{key} | {entry["accommodation_d"]:.7g} | '+numbers(entry['origin_mm'])+
            ' | 10201/10201 | '+ '/'.join(str(counts[name]) for name in SPLITS[:-1]))
    if entries:
        lines.extend(['', 'Target own-fit errors (all 10201 states)',
            'A family | rank/terms cond | vector RMS/P95/max | wx RMS/P95/max | wy RMS/P95/max | sensitivity x,y um/arcmin | oddness max'])
    for key, entry in entries.items():
        for parity, fit in entry.get('families', {}).items():
            metrics = fit['groups']['overall']['metrics']
            triples = ['/'.join(f'{metrics[axis][measure+"_arcmin"]:.6g}'
                        for measure in ('rms', 'p95', 'max')) for axis in ('vector', 'wx', 'wy')]
            sens = fit['sensitivity']['axis_um_per_arcmin']
            lines.append(f'{key} {parity} | {fit["rank"]}/{fit["coefficient_count_per_axis"]} '
                f'{fit["condition_number"]:.5g} | '+' | '.join(triples)+' | '+
                ('unavailable' if sens is None else numbers(sens))+
                f' | {fit["oddness"]["max_abs_arcmin"]:.4g}')
    transfer = report.get('A00_cross_application')
    if transfer:
        lines.extend(['', 'A00 coefficient transfer: target zero recentered (primary), frozen A00 origin (secondary)',
            'A family | recentered vector RMS/P95/max | frozen vector RMS/P95/max | matched own max | pass recentered/frozen/own'])
        for key, target in transfer['targets'].items():
            for parity, entry in target['families'].items():
                triples = []
                for origin in ('target_recentered', 'frozen_A00_origin'):
                    metrics = entry[origin]['groups']['overall']['metrics']['vector']
                    triples.append('/'.join(f'{metrics[measure+"_arcmin"]:.6g}'
                                           for measure in ('rms', 'p95', 'max')))
                own = entry['matched_target_own_fit_groups']['overall']['metrics']['vector']['max_arcmin']
                comparison = entry['overall_target_comparison']
                passes = '/'.join(str(comparison[name]) for name in (
                    'target_recentered_A00_pass', 'frozen_A00_origin_pass', 'matched_target_own_fit_pass'))
                lines.append(f'{key} {parity} | '+' | '.join(triples)+f' | {own:.6g} | '+passes)
        lines.append('A nonzero error alone does not establish that accommodation must be an explicit model input.')
    comparison = report.get('coefficient_comparison')
    if comparison:
        lines.extend(['', 'Raw centered-mm coefficients, axis order wx,wy; units deg/mm**(i+j).',
            'Values follow analyzed index order '+str(comparison['accommodation_indices'])+'.',
            'Trend parameters: linear [p0,p1], quadratic [p0,p1,p2], actual manifest diopters.',
            'family axis (i,j) | coefficient values | linear fit | quadratic fit'])
        for parity, family in comparison['families'].items():
            for row in family['raw_mm']:
                lines.append(f'{parity} {row["axis"]} {tuple(row["power"])} | '+numbers(row['values'])+
                    ' | '+trend_text(row['trends']['linear'])+' | '+trend_text(row['trends']['quadratic']))
        lines.append('Relative delta = abs(c-c_A00)/abs(c_A00), unavailable only for an exactly zero baseline; no epsilon.')
    if report['errors']:
        lines.extend(['', 'Errors:', *report['errors']])
    lines.append('results_accom.json includes both coefficient representations, deltas, trends/residuals, all split/axis errors, sensitivity and provenance.')
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--selftest', action='store_true', help='Synthetic only; never read trace data')
    mode.add_argument('--quick', action='store_true', help='Full A00 validation, then stride-5 gaze lattice')
    parser.set_defaults(accommodation_sweep=False)
    parser.add_argument('--accommodation-indices', type=int, nargs='+', default=[0],
        action=AccommodationIndices,
        help='Explicit E2 cubic sweep over full5 shard indices 0..20; adds A00 baseline if needed')
    args = parser.parse_args()
    if args.accommodation_sweep and args.quick:
        parser.error('--quick cannot be combined with --accommodation-indices')
    report, maps = initial_report(args), {}
    try:
        log = Progress(args.accommodation_sweep)
    except Exception as exc:
        print('Unsafe output paths: '+str(exc), file=sys.stderr)
        return 1
    started = time.perf_counter()
    try:
        log('start '+report['mode']+'; noise-free design traces; unaudited')
        if args.selftest:
            run_selftest(report, maps, log)
        elif args.accommodation_sweep:
            run_accommodation(report, args.accommodation_indices, log)
        else:
            metadata = {}
            loader_args = SimpleNamespace(quick=False, shard_index=0, input_path=INPUT)
            # Only this loader: preserves its schema, metadata, manifest, provenance,
            # double hash, ray/geometry, calibration, ordering and partition checks.
            observed, _, _, _, groups = five.load_shard(loader_args, report, metadata, log)
            maps.update(metadata)  # Retain loaded states even if downstream validation refuses.
            require(report['input']['manifest_entry']['accommodation_d'] == 0,
                    'Only A00 / A=0 is allowed')
            analyze(observed, groups, metadata, report, maps, log, args.quick)
    except Exception as exc:
        report['status'] = 'failed'
        report['numerical_gates_passed'] = False
        report['errors'].append(type(exc).__name__+': '+str(exc))
        log('failed: '+str(exc))
    report['timings']['total_analysis_s'] = time.perf_counter()-started
    log('final status='+report['status'])
    # Fixed artifacts, no temporary/cache files; attempt each independently on failure.
    for name in output_files(args.accommodation_sweep):
        try:
            guard_outputs()
            if name == 'maps.npz':
                require(all(isinstance(v, np.ndarray) and v.dtype.kind != 'O'
                            for v in maps.values()), 'Object/non-numeric map forbidden')
                np.savez_compressed(OUT/name, **maps)
            elif name in ('results.json', 'results_accom.json'):
                (OUT/name).write_text(json.dumps(p4_a0_data.safe(report), indent=2,
                                                allow_nan=False)+'\n', encoding='utf-8')
            else:
                writer = accommodation_text_report if args.accommodation_sweep else text_report
                (OUT/name).write_text(writer(report), encoding='utf-8')
        except Exception as exc:
            report['status'] = 'failed'
            report['numerical_gates_passed'] = False
            report['errors'].append('write '+name+': '+str(exc))
            print(report['errors'][-1], file=sys.stderr)
    return int(not report['numerical_gates_passed'] or bool(report['errors']))


if __name__ == '__main__':
    raise SystemExit(main())
