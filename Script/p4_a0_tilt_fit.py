r"""Outer-field A=0 tilt fits: read-only on data, noise-free, no tangents as inputs,
centre truth for evaluation only. Run in PowerShell from any directory:
& 'C:\Users\rueijrwu\.pyenv-win-venv\envs\venv\Scripts\python.exe' 'C:\Users\rueijrwu\OneDrive\Project\DistortionTracking\Script\p4_a0_tilt_fit.py'
"""
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path

import numpy as np
from p4_a0_data import DEFAULT, safe
from p4_a0_gain import load_a0, require  # Import-safe; reuse validation only.

ROOT = Path(__file__).resolve().parents[1]
EPS = np.finfo(float).eps
TARGETS = dict(outer_rms_um=0.1, outer_max_um=1., centre_rms_um=0.1,
               centre_max_um=0.3, unknown_max_arcmin=1.)
# Primitive columns: component, node-x/y powers, gaze-ux/uy powers.
TERMS = [(0, 0, 0, 1, 0), (0, 0, 0, 3, 0), (0, 0, 0, 1, 2),
         (0, 1, 0, 2, 0), (0, 1, 0, 0, 2), (0, 0, 1, 1, 1),
         (0, 2, 0, 1, 0), (0, 0, 2, 1, 0), (0, 1, 1, 0, 1),
         (1, 0, 0, 0, 1), (1, 0, 0, 0, 3), (1, 0, 0, 2, 1),
         (1, 0, 1, 0, 2), (1, 0, 1, 2, 0), (1, 1, 0, 1, 1),
         (1, 0, 2, 0, 1), (1, 2, 0, 0, 1), (1, 1, 1, 1, 0)]
PRIMITIVES = ('sx_gain sx_cubic sx_cross_cubic mx_self mx_cross shear_x '
              'qx_x2_ux qx_y2_ux qx_xy_uy sy_gain sy_cubic sy_cross_cubic '
              'my_self my_cross shear_y qy_y2_uy qy_x2_uy qy_xy_ux').split()
POWERS = np.array([t[3:] for t in TERMS])


def definitions():
    eye = np.eye(18)
    shift, mag, shear = [0, 1, 2, 9, 10, 11], [3, 4, 12, 13], [5, 14]
    columns = {n: eye[:, i] for i, n in enumerate(PRIMITIVES)}
    # Keystone columns couple components; thin prism adds, rather than replaces.
    columns.update(k_x=-eye[:, 6]-eye[:, 17], k_y=-eye[:, 8]-eye[:, 15],
                   alpha_x=eye[:, 6]+eye[:, 7], alpha_y=eye[:, 15]+eye[:, 16])
    names = lambda indices: [PRIMITIVES[i] for i in indices]
    s, m, h = names(shift), names(mag), names(shear)
    tilt = s+m+h+['k_x', 'k_y', 'alpha_x', 'alpha_y']
    sets = dict(M6=s, M10=s+m, M12=s+m+h, M14=s+m+h+['k_x', 'k_y'], M16=tilt,
                N18=PRIMITIVES.copy(),
                N3T_no_shear=[n for n in tilt if n not in h],
                N3T_no_cross_mag=[n for n in tilt if n not in ('mx_cross', 'my_cross')],
                N3T_neither=[n for n in tilt if n not in (*h, 'mx_cross', 'my_cross')])
    require([len(v) for v in sets.values()] == [6, 10, 12, 14, 16, 18, 14, 14, 12],
            'Model parameter counts inconsistent')
    return {name: (ns, np.column_stack([columns[n] for n in ns])) for name, ns in sets.items()}


def node_basis(g):
    phi = np.zeros((len(g), 2, 18))
    for k, (j, px, py, _, _) in enumerate(TERMS):
        phi[:, j, k] = g[:, 0]**px * g[:, 1]**py
    return phi.reshape(-1, 18)


def monomials(u, derivative=None):
    powers = POWERS.copy()
    factor = np.ones(18)
    if derivative is not None:
        factor = powers[:, derivative].copy()
        powers[:, derivative] = np.maximum(0, powers[:, derivative]-1)
    u = np.atleast_2d(u)
    return factor * u[:, 0, None]**powers[:, 0] * u[:, 1, None]**powers[:, 1]


def calibrate(cube, fields):
    require(set(map(tuple, fields)) == {(x, y) for x in (-1, 0, 1) for y in (-1, 0, 1)},
            'Expected exactly the {-1,0,1} squared field labels')
    outer = np.any(fields != 0, axis=1)
    labels, g = fields[outer], -fields[outer]
    points = np.stack((cube['xsi'][50, 50, outer], cube['ysi'][50, 50, outer]), -1)
    point = lambda label: points[np.flatnonzero(np.all(labels == label, axis=1))[0]]
    c0 = (point((-1, 0))+point((1, 0)))/2
    check = (point((0, -1))+point((0, 1)))/2
    template = points-c0
    e = np.linalg.norm(point((-1, 0))-point((1, 0)))/2
    c = point((-1, -1))[0]-c0[0]  # Fixed (+1,+1) node's x component; no averaging.
    H = 2*e-c
    reproduced = np.where((np.sum(g*g, axis=1) == 1)[:, None], e*g, c*g)
    defect = np.max(np.linalg.norm(template-reproduced, axis=1))
    require(e > 0 and c > 0 and H > 0 and defect <= 1e-6,
            f'S=-I symmetric zero-gaze reproduction failed: {defect} mm')
    calibration = dict(c0_mm=c0, c0_definition='midpoint of labels (-1,0),(+1,0)',
        other_axial_midpoint_mm=check, midpoint_difference_mm=check-c0,
        e_mm=e, e_definition='half distance between the fixed horizontal axial pair',
        c_mm=c, c_definition='x component at label (-1,-1) relative to c0', H_mm=H,
        H_definition='2*e-c; expected approximately 2.591699 mm', fitted_parameters=0,
        gauge=-np.eye(2), template_mm=template, template_mean_mm=template.mean(0),
        zero_pattern_max_defect_mm=defect, zero_pattern_tolerance_mm=1e-6,
        mapping=[dict(label=l, g=n, sensor_zero_mm=p, reproduced_zero_mm=c0+t)
                 for l, n, p, t in zip(labels, g, points, reproduced)])
    return outer, g, c0, template, H, calibration


def partition(cube):
    w = np.stack((cube['bde_deg'][..., 0], -cube['ade_deg'][..., 0]), -1)
    zero = np.all(w == 0, axis=-1)
    require(zero.sum() == 1 and zero[50, 50], 'Zero gaze must be reference index (50,50)')
    a, b = np.indices((101, 101))
    ae, be = a % 2 == 0, b % 2 == 0
    masks = dict(train=ae & be & ~zero, even_odd=ae & ~be, odd_even=~ae & be, odd_odd=~ae & ~be)
    require([int(m.sum()) for m in masks.values()] == [2600, 2550, 2550, 2500], 'Split counts')
    require(np.all(sum(m.astype(int) for m in masks.values()) == ~zero), 'Split coverage')
    for m in masks.values():
        require(np.array_equal(m, m[::-1, :]) and np.array_equal(m, m[:, ::-1]),
                'Reflection-related gaze states crossed a split')
    return w.reshape(-1, 2), {n: np.flatnonzero(m.ravel()) for n, m in masks.items()}


def fit(design, target, transform):
    a = design @ transform
    scales = np.linalg.norm(a, axis=0)
    require(np.all(scales > 0), 'Zero design column')
    left, singular, right = np.linalg.svd(a/scales, full_matrices=False)
    tolerance = singular[0]*max(a.shape)*EPS
    keep = singular > tolerance
    coef = (right[keep].T @ ((left[:, keep].T @ target)/singular[keep]))/scales
    return coef, dict(rank=int(keep.sum()), columns=a.shape[1], full_rank=bool(keep.all()),
        rank_tolerance=tolerance, rank_rule='sigma_max*max(scaled_design.shape)*machine_epsilon',
        singular_values=singular, column_scales=scales,
        condition_number=singular[0]/singular[-1] if keep.all() else float('inf'),
        parameter_count_claim_supported=bool(keep.all()), design_shape=a.shape)


def location(index, w):
    return dict(ade_index=int(index//101), bde_index=int(index % 101),
                ADE_deg=float(-w[index, 1]), BDE_deg=float(w[index, 0]), w_deg=w[index])


def error_metrics(error, ids, w):
    # State axis first, optional node axis second; every norm is a 2D vector norm.
    norms = np.linalg.norm(error, axis=-1)
    loc = np.unravel_index(np.argmax(norms), norms.shape)
    result = dict(rms=float(np.sqrt(np.mean(norms**2))), max=float(norms[loc]),
                  signed_bias_xy=error.reshape(-1, 2).mean(0), worst=location(ids[loc[0]], w))
    result['worst']['signed_error_xy'] = error[loc]
    if len(loc) == 2:
        result['worst']['outer_node_index'] = int(loc[1])
    return result


def nearest_training(y, train):
    # Exact measured-configuration Euclidean search, shared across all models.
    reference, chosen = y[train], np.empty(len(y), dtype=int)
    squared = np.sum(reference**2, axis=1)
    for start in range(0, len(y), 128):
        ids = np.arange(start, min(start+128, len(y)))
        distance = np.sum(y[ids]**2, axis=1)[:, None]+squared-2*y[ids] @ reference.T
        distance[ids[:, None] == train[None, :]] = np.inf  # Leave self out for training queries.
        chosen[ids] = train[np.argmin(distance, axis=1)]
    return chosen


def invert(target, initial, weights):
    u, converged, reason, iterations = initial.copy(), False, 'iteration_limit', 0
    predict = lambda v: (monomials(v) @ weights.T)[0]
    for iteration in range(50):
        iterations = iteration+1
        residual = predict(u)-target
        jac = np.column_stack([(monomials(u, j) @ weights.T)[0] for j in (0, 1)])
        left, singular, right = np.linalg.svd(jac, full_matrices=False)
        cutoff = singular[0]*max(jac.shape)*EPS
        if np.sum(singular > cutoff) != 2:
            reason = 'rank_deficient'
            break
        gradient = jac.T @ residual
        pg = np.linalg.norm(u-np.clip(u-gradient, -1, 1), np.inf)
        pg /= max(1., singular[0]*np.linalg.norm(residual))
        if pg <= 1e-10:
            converged, reason = True, 'projected_gradient'
            break
        cost = np.dot(residual, residual)/2
        damping, accepted = 1e-10*singular[0]**2, False
        for _ in range(18):
            step = -right.T @ ((singular/(singular**2+damping))*(left.T @ residual))
            trial = np.clip(u+step, -1, 1)
            trial_residual = predict(trial)-target
            trial_cost = np.dot(trial_residual, trial_residual)/2
            if np.isfinite(trial_cost) and trial_cost <= cost:
                small_step = np.linalg.norm(trial-u, np.inf) <= 1e-10
                relative_change = abs(cost-trial_cost)/max(cost, np.finfo(float).tiny)
                u, accepted = trial, True
                if small_step and relative_change <= 1e-12:
                    converged, reason = True, 'step_and_relative_objective'
                break
            damping *= 10
        if not accepted:
            reason = 'damping_stalled'
            break
        if converged:
            break
    residual = predict(u)-target
    jac = np.column_stack([(monomials(u, j) @ weights.T)[0] for j in (0, 1)])
    singular = np.linalg.svd(jac, compute_uv=False)
    cutoff = singular[0]*max(jac.shape)*EPS
    rank = int(np.sum(singular > cutoff))
    pg = np.linalg.norm(u-np.clip(u-jac.T @ residual, -1, 1), np.inf)
    pg /= max(1., singular[0]*np.linalg.norm(residual))
    if pg <= 1e-10 and rank == 2:
        converged, reason = True, 'projected_gradient'
    boundary = bool(np.any(np.abs(u) >= 1-1e-10))
    return u, dict(converged=converged, success=converged and rank == 2,
        reason=reason, boundary=boundary, at_boundary=boundary, rank=rank, rank_tolerance=cutoff,
        singular_values=singular, iterations=iterations, normalized_projected_gradient=pg,
        objective=float(np.dot(residual, residual)/2), u_hat=u)


def inverse_sample_ids(ids, full, training_ids, training_group):
    ids = ids[ids != 50*101+50]
    if not training_group:
        ids = ids[~np.isin(ids, training_ids)]
    if not full:
        ids = ids[((ids//101-50) % 5 == 0) & ((ids % 101-50) % 5 == 0)]
    return ids


def inverse_sampling(ids, full):
    ai, bi = np.unique(ids//101), np.unique(ids % 101)
    missing = np.setdiff1d((ai[:, None]*101+bi).ravel(), ids)
    return dict(states=len(ids), ade_bde_indices=np.column_stack((ids//101, ids % 101)),
        ade_indices=ai, bde_indices=bi,
        excluded_cartesian_pairs=np.column_stack((missing//101, missing % 101)),
        sampling='full group' if full else 'absolute ADE/BDE index lattice (i-50)%5==0, intersected with group; zero excluded; training states excluded from holdouts',
        full_group=full)


def inverse_metrics(ids, full, y, nearest, u, weights, w, record):
    recovered, statuses = [], []
    for index in ids:
        initial = u[nearest[index]]
        try:
            hat, status = invert(y[index], initial, weights)
            require(np.all(np.isfinite(hat)), 'Nonfinite inverse solution')
        except Exception as exc:
            stage_error(record, 'inverse_solver', exc, [(index//101, index % 101)])
            hat = initial.copy()
            status = dict(converged=False, success=False, reason=f'{type(exc).__name__}: {exc}',
                          boundary=bool(np.any(np.abs(hat) >= 1-1e-10)),
                          at_boundary=bool(np.any(np.abs(hat) >= 1-1e-10)), rank=0,
                          rank_tolerance=None, singular_values=[], iterations=None,
                          normalized_projected_gradient=None, objective=None, u_hat=hat)
        status.update(state=location(index, w), initial_training_state=location(nearest[index], w))
        recovered.append(hat)
        statuses.append(status)
    result = (error_metrics((np.array(recovered)-u[ids])*12.5*60, ids, w) if len(ids)
              else dict(rms=None, max=None, signed_bias_xy=None, worst=None))
    result.update(tested_states=len(ids), failed_inversions=sum(not s['success'] for s in statuses),
        boundary_solutions=sum(s['at_boundary'] for s in statuses),
        at_boundary=sum(s['at_boundary'] for s in statuses),
        rank_deficient_solutions=sum(s['rank'] < 2 for s in statuses),
        unconverged=sum(not s['converged'] for s in statuses), statuses=statuses,
        empty=not len(ids), **inverse_sampling(ids, full))
    return result


def stage_error(record, stage, exc, ade_bde_indices=None):
    error = exc if isinstance(exc, str) else f'{type(exc).__name__}: {exc}'
    errors = record.setdefault('errors', [])
    if ade_bde_indices is None:
        errors.append(dict(stage=stage, error=error))
        return
    affected = [[int(a), int(b)] for a, b in ade_bde_indices]
    for entry in errors:
        if (entry.get('stage') == stage and entry.get('error') == error
                and 'ade_bde_indices' in entry):
            entry['ade_bde_indices'].extend(affected)
            return
    errors.append(dict(stage=stage, error=error, ade_bde_indices=affected))


def analyze(report):
    cube, fields = load_a0((ROOT/DEFAULT).resolve(), report)
    outer, g, c0, template, H, report['calibration'] = calibrate(cube, fields)
    w, groups = partition(cube)
    u, phi = w/12.5, node_basis(g)
    observed = np.stack((cube['xsi'][..., outer], cube['ysi'][..., outer]), -1).reshape(-1, 8, 2)
    y = ((observed-c0-template)/H).reshape(-1, 16)
    train, sets = groups['train'], definitions()
    report['splits'] = {n: dict(states=len(ids), ade_bde_indices=np.column_stack((ids//101, ids % 101)))
                        for n, ids in groups.items()}
    report['zero_reference'] = dict(ade_index=50, bde_index=50, states=1, used_for_fit=False)
    design_error = None
    try:
        design = (monomials(u[train])[:, None, :]*phi[None, :, :]).reshape(-1, 18)
    except Exception as exc:
        design_error = exc
    models, eligible = report['models'], []
    for order, (name, (names, transform)) in enumerate(sets.items()):
        result = models[name] = dict(parameter_count=len(names), columns=names, primitive_transform=transform,
            coefficients=None, primitive_coefficients=None, conditioning=None, training_qualifies=None,
            errors=[], groups={group: dict(outer_um=None, centre_um=None, unknown_arcmin=None,
                                          target_checks=None, errors=[]) for group in groups})
        try:
            if design_error is not None:
                raise design_error
            coef, conditioning = fit(design, y[train].ravel(), transform)
            beta = transform @ coef
            weights = phi*beta
            result.update(coefficients=dict(zip(names, coef)), primitive_coefficients=beta,
                          conditioning=conditioning)
        except Exception as exc:
            stage = 'training_design' if design_error is not None else 'fit'
            stage_error(result, stage, exc)
            for metrics in result['groups'].values():
                stage_error(metrics, stage, exc)
            continue
        try:
            training = error_metrics((monomials(u[train]) @ weights.T-y[train]).reshape(-1, 8, 2)*H*1000, train, w)
            result['groups']['train']['outer_um'] = training
            qualifies = (conditioning['full_rank'] and training['rms'] <= TARGETS['outer_rms_um']
                         and training['max'] <= TARGETS['outer_max_um'])
            result['training_qualifies'] = qualifies
        except Exception as exc:
            stage_error(result['groups']['train'], 'training_outer', exc)
            continue
        if qualifies:
            eligible.append((len(names), training['rms'], order, name))
    selected = min(eligible)[-1] if eligible else None  # Freeze selection before accessing centre truth.
    report['selection'].update(selected=selected, status='selected' if selected else 'none qualifies')
    centre_setup_error = nearest_error = None
    try:
        centre_index = int(np.flatnonzero(np.all(fields == 0, axis=1))[0])
        truth = np.stack((cube['xsi'][..., centre_index], cube['ysi'][..., centre_index]), -1).reshape(-1, 2)
        centre_phi = node_basis(np.zeros((1, 2)))
    except Exception as exc:
        centre_setup_error = exc
    try:
        nearest = nearest_training(y, train)
    except Exception as exc:
        nearest_error = exc
    for name, result in models.items():
        full = name == selected
        for group, ids in groups.items():
            metrics = result['groups'][group]
            metrics['inverse_sampling'] = None
            try:
                sample_ids = inverse_sample_ids(ids, full, train, group == 'train')
                metrics['inverse_sampling'] = inverse_sampling(sample_ids, full)
            except Exception as exc:
                stage_error(metrics, 'inverse_sampling', exc)
        if result['primitive_coefficients'] is None:
            continue
        beta, weights = result['primitive_coefficients'], phi*result['primitive_coefficients']
        for group, ids in groups.items():
            metrics = result['groups'][group]
            if group != 'train':
                try:
                    prediction = (monomials(u[ids]) @ weights.T).reshape(-1, 8, 2)
                    metrics['outer_um'] = error_metrics((prediction-y[ids].reshape(-1, 8, 2))*H*1000, ids, w)
                except Exception as exc:
                    stage_error(metrics, 'outer', exc)
            try:
                if centre_setup_error is not None:
                    raise centre_setup_error
                shift = monomials(u[ids]) @ (centre_phi*beta).T
                metrics['centre_um'] = error_metrics((c0+H*shift-truth[ids])*1000, ids, w)
            except Exception as exc:
                stage_error(metrics, 'centre', exc)
            try:
                if nearest_error is not None:
                    raise nearest_error
                require(metrics['inverse_sampling'] is not None, 'Inverse sampling not computed')
                sample_ids = inverse_sample_ids(ids, full, train, group == 'train')
                metrics['unknown_arcmin'] = inverse_metrics(sample_ids, full, y, nearest, u, weights, w, metrics)
            except Exception as exc:
                stage_error(metrics, 'inverse', exc,
                            (metrics.get('inverse_sampling') or {}).get('ade_bde_indices'))
            try:
                a, b, inv = (metrics[k] for k in ('outer_um', 'centre_um', 'unknown_arcmin'))
                outer_check = None if a is None else (a['rms'] <= TARGETS['outer_rms_um'] and a['max'] <= TARGETS['outer_max_um'])
                centre_check = None if b is None else (b['rms'] <= TARGETS['centre_rms_um'] and b['max'] <= TARGETS['centre_max_um'])
                tested_check = (None if inv is None or not inv['tested_states'] else
                                inv['max'] <= TARGETS['unknown_max_arcmin'] and inv['failed_inversions'] == 0)
                full_group = inv is not None and full and inv['tested_states'] == len(ids)
                checks = dict(outer=outer_check, centre=centre_check, unknown=tested_check if full_group else None)
                metrics['target_checks'] = dict(**checks,
                    all=all(checks.values()) if all(v is not None for v in checks.values()) else None,
                    full_rank=result['conditioning']['full_rank'], unknown_tested_states=tested_check,
                    unknown_full_group_evaluated=full_group,
                    unknown_scope=(metrics['inverse_sampling'] or {}).get('sampling'))
            except Exception as exc:
                stage_error(metrics, 'target_checks', exc)
        if 'alpha_x' in result['columns']:
            result['mean_to_centre_bias'] = None
            try:
                if centre_setup_error is not None:
                    raise centre_setup_error
                p = result['coefficients']
                expected = .75*H*u*np.array((2*p['alpha_x']-p['k_x'], 2*p['alpha_y']-p['k_y']))
                prediction = (monomials(u) @ weights.T).reshape(-1, 8, 2)
                shift = monomials(u) @ (centre_phi*beta).T
                actual = H*(prediction.mean(1)-shift)
                result['mean_to_centre_bias'] = dict(formula='(3/4)*H*((2*alpha_x-k_x)*ux,(2*alpha_y-k_y)*uy)',
                    max_identity_defect_um=float(np.max(np.linalg.norm(actual-expected, axis=1))*1000),
                    max_bias_um=float(np.max(np.linalg.norm(expected, axis=1))*1000),
                    template_mean_mm=template.mean(0), magnification_and_shear_mean=0,
                    note='Diagnostic of deformation mean only; no mean is used to estimate the centre')
            except Exception as exc:
                stage_error(result, 'mean_to_centre_bias', exc)
    report['pareto'] = {}
    for group in groups:
        entries = [(n, r['parameter_count'], r['groups'][group]['centre_um']['rms'])
                   for n, r in models.items() if (r.get('conditioning') or {}).get('full_rank')
                   and (r['groups'][group].get('centre_um') or {}).get('rms') is not None]
        report['pareto'][group] = [dict(model=n, parameters=p, centre_rms_um=e) for n, p, e in entries
            if not any(p2 <= p and e2 <= e and (p2 < p or e2 < e) for _, p2, e2 in entries)]
    report['status'] = ('completed_with_errors' if any(r.get('errors') or
        any(m.get('errors') for m in r['groups'].values()) for r in models.values()) else 'completed')


def compact_text(report):
    fmt = lambda v: ('NOT COMPUTED' if v is None else
                     np.array2string(np.asarray(v), precision=10, threshold=np.inf, max_line_width=10000).replace('\n', ''))
    number = lambda v, precision=6: 'NOT COMPUTED' if v is None else f'{v:.{precision}g}'
    flag = lambda v: 'NOT COMPUTED' if v is None else str(v)
    def error_text(error):
        text = f"{error.get('stage', 'unknown stage')}: {error.get('error', 'unknown error')}"
        if 'ade_bde_indices' in error:
            pairs = error['ade_bde_indices']
            text += f"; affected ADE/BDE index pairs={pairs[:10]} (total={len(pairs)})"
        return text
    errors = lambda r: '; '.join(error_text(e) for e in (r.get('errors') or []))
    lines = ['A=0 outer-field tilt ladder; sensor mm; residuals prediction minus observation.',
             report.get('model') or 'Model: NOT COMPUTED',
             (report.get('selection') or {}).get('rule') or 'Selection rule: NOT COMPUTED',
             f"Selection: {(report.get('selection') or {}).get('selected') or 'none'}; status: {report.get('status', 'NOT COMPUTED')}",
             'Targets: outer RMS/max <=0.1/1 um; centre RMS/max <=0.1/0.3 um; unknown max <=1 arcmin, zero unconverged or rank-deficient inversions.',
             (report.get('solver') or {}).get('sampling') or 'Inverse sampling: NOT COMPUTED',
             'Inverse success requires a convergence stopping criterion and numerical Jacobian rank 2. at_boundary is diagnostic only.',
             'Sampled inverse maxima and sampled target checks cannot establish full-group compliance; an empty sample is NOT COMPUTED.']
    if report.get('calibration'):
        c = report['calibration']
        lines += [f"c0={fmt(c.get('c0_mm'))} mm; vertical midpoint={fmt(c.get('other_axial_midpoint_mm'))} mm; e={number(c.get('e_mm'), 12)}; c={number(c.get('c_mm'), 12)}; H={number(c.get('H_mm'), 12)} mm",
                  f"S=-I; zero-pattern max defect={number(c.get('zero_pattern_max_defect_mm'))} mm; calibration parameter count=0."]
        lines += [f"label={fmt(m.get('label'))} -> g={fmt(m.get('g'))}; zero={fmt(m.get('sensor_zero_mm'))}; reproduced={fmt(m.get('reproduced_zero_mm'))}" for m in (c.get('mapping') or [])]
    lines += ['Splits (ADE/BDE parity; zero 50/50 reference only): '+', '.join(f"{n}={number((d or {}).get('states'))}" for n, d in (report.get('splits') or {}).items()),
              'Columns: primitive = component * x^px*y^py*ux^a*uy^b; x,y are g coordinates.']
    lines += [f"{n}: component={'xy'[j]}, powers=({px},{py},{a},{b})" for n, (j, px, py, a, b) in zip(PRIMITIVES, TERMS)]
    lines += ['k_x=(-x^2*ux,-x*y*ux); k_y=(-x*y*uy,-y^2*uy); alpha_x=((x^2+y^2)*ux,0); alpha_y=(0,(x^2+y^2)*uy).',
              'Each split: outer RMS/max um; centre bias xy, RMS/max um, worst w deg; unknown RMS/max arcmin, failures/tested, at_boundary count, worst w deg; pass outer/centre/full-group unknown.']
    for name, r in (report.get('models') or {}).items():
        r = r or {}
        cond = r.get('conditioning') or {}
        rank_text = ('NOT COMPUTED' if cond.get('full_rank') is None else
                     'FULL RANK' if cond['full_rank'] else 'RANK DEFICIENT: parameter-count claim unsupported')
        lines += [f"{name} | {number(r.get('parameter_count'))}/{number(cond.get('rank'))} {rank_text}",
                  f"{name} columns: {', '.join(r.get('columns') or [])}; SVD tol={number(cond.get('rank_tolerance'))}; singular={fmt(cond.get('singular_values'))}; scaled condition={number(cond.get('condition_number'))}"]
        if r.get('errors'):
            lines.append(f"{name} NOT COMPUTED stage(s): {errors(r)}")
        for group, metrics in (r.get('groups') or {}).items():
            metrics = metrics or {}
            a, b, v, checks = (metrics.get(k) or {} for k in ('outer_um', 'centre_um', 'unknown_arcmin', 'target_checks'))
            unknown_check = flag(checks.get('unknown'))
            if v.get('empty'):
                unknown_check = 'NOT COMPUTED (EMPTY SAMPLE)'
            elif v and (metrics.get('inverse_sampling') or {}).get('full_group') is False:
                unknown_check = f"NOT ESTABLISHED (sampled tested-state pass={flag(checks.get('unknown_tested_states'))})"
            lines.append(f"{name}/{group}: outer {number(a.get('rms'))}/{number(a.get('max'))}; centre {fmt(b.get('signed_bias_xy'))},{number(b.get('rms'))}/{number(b.get('max'))}@{fmt((b.get('worst') or {}).get('w_deg'))}; unknown {number(v.get('rms'))}/{number(v.get('max'))},{number(v.get('failed_inversions'))}/{number(v.get('tested_states'))},at_boundary={number(v.get('at_boundary'))}@{fmt((v.get('worst') or {}).get('w_deg'))}; pass {flag(checks.get('outer'))}/{flag(checks.get('centre'))}/{unknown_check}" + (f"; errors: {errors(metrics)}" if metrics.get('errors') else ''))
            sample = metrics.get('inverse_sampling') or {}
            empty = ' EMPTY GROUP/SAMPLE' if sample.get('states') == 0 else ''
            lines.append(f"{name}/{group} inverse states: tested={number(v.get('tested_states'))}, planned={number(sample.get('states'))}{empty}; {sample.get('sampling') or 'NOT COMPUTED'}; exact Cartesian ADE indices={fmt(sample.get('ade_indices'))}, BDE indices={fmt(sample.get('bde_indices'))}, excluding pairs={fmt(sample.get('excluded_cartesian_pairs'))}; exact ADE/BDE pairs also in JSON.")
        if r.get('mean_to_centre_bias'):
            b = r['mean_to_centre_bias']
            lines += [f"{name} mean-to-centre bias: {b.get('formula') or 'NOT COMPUTED'}; max={number(b.get('max_bias_um'))} um; identity defect={number(b.get('max_identity_defect_um'))} um; magnification/shear mean=0."]
    lines += [f"Pareto {group} (full-rank parameters vs centre RMS, descriptive only): "+('; '.join(f"{r.get('model', 'NOT COMPUTED')} ({number(r.get('parameters'))}, {number(r.get('centre_rms_um'))} um)" for r in (entries or [])) or 'NOT COMPUTED: no eligible computed centre metrics') for group, entries in (report.get('pareto') or {}).items()]
    if report.get('failure'):
        lines.append(f"NOT COMPUTED: {report['failure']}")
    require(len(lines) < 300, 'Report must be under 300 lines')
    return '\n'.join(lines)+'\n'


def main():
    path, out = (ROOT/DEFAULT).resolve(), ROOT/'runs'/'p4_a0_tilt'
    targets = [out/'report.txt', out/'results.json']
    for target in targets:
        require(target.resolve() != path and not (target.exists() and path.exists() and target.samefile(path)),
                'Output aliases read-only input; refusing to run')
    report = dict(input=str(DEFAULT), numpy_version=np.__version__, models={}, targets=TARGETS,
        model='X_i=c0+H*s(u)+T_i+H*d(g_i,u); w=(BDE,-ADE) deg; u=w/12.5; polynomial keystone only.',
        primitive_definitions=[dict(name=n, component='xy'[t[0]], node_powers=t[1:3], gaze_powers=t[3:]) for n, t in zip(PRIMITIVES, TERMS)],
        selection=dict(rule='Frozen: full-rank training outer RMS<=0.1 um and max<=1 um; smallest parameter count, then training RMS, then fixed model order; no holdouts or centre truth.', fixed_order=list(definitions())),
        solver=dict(method='bounded damped Gauss-Newton, analytic Jacobian, 16 normalized equations',
            bounds=[-1., 1.], max_iterations=50, projected_gradient_tolerance=1e-10,
            projected_gradient='||u-clip(u-J.T*r,-1,1)||inf/max(1,||J||2*||r||2)',
            step_tolerance=1e-10, relative_objective_tolerance=1e-12, boundary_tolerance=1e-10,
            initialization='nearest measured TRAINING configuration by exact Euclidean search; no true-gaze initialization; training queries leave themselves out',
            sampling='Selected model: full training and all holdouts. Others: each absolute ADE/BDE index satisfies (i-50)%5==0 (0,5,...,100), intersected with split parity; exclude zero and training states from holdouts; retain the training split for training evaluation. Sampled maxima cannot establish full-group compliance.',
            failure_rule='convergence stopping criteria not met or numerical Jacobian rank<2; at_boundary is counted as a diagnostic only'))
    try:
        analyze(report)
    except Exception as exc:
        report.update(status='failed', failure=f'{type(exc).__name__}: {exc}')
    text = compact_text(report)
    out.mkdir(parents=True, exist_ok=True)
    targets[0].write_text(text, encoding='utf-8')
    targets[1].write_text(json.dumps(safe(report), indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(text, end='')
    return int(report['status'] != 'completed')


if __name__ == '__main__':
    raise SystemExit(main())
