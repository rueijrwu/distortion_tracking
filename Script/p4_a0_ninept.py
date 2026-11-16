"""Usage: p4_a0_ninept.py [--quick | --selftest] [--workers N]; default: full D0/D1."""
import sys
sys.dont_write_bytecode = True

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import stat
import time

import numpy as np
import p4_a0_data
import p4_a0_joint_fit as joint
from p4_a0_joint_fit import F, K1, K2, EPS, FD, COND


ROOT = Path(__file__).absolute().parents[1]
OUT = ROOT / 'runs' / 'p4_ninept'
ARTIFACTS = ('report.txt', 'results.json', 'maps.npz', 'progress.log')
LABELS = np.array([(x, y) for y in (1, 0, -1) for x in (-1, 0, 1)], float)
NODES = -LABELS
OUTER = np.arange(9) != 4
PARAMETERS = ('sx', 'sy', 'a', 'b', 'c_L', 'e', 'kx', 'ky')
DIAGNOSTICS = ('prism_x', 'prism_y', 'key_x', 'key_y', 'barrel_tie',
               'barrel_level', 'cubic_x', 'cubic_y', 'quart_x', 'quart_y')
CODEBOOK = {0: 'active', 1: 'converged_gradient', 2: 'converged_step_and_gradient',
            3: 'invalid_forward_or_branch', 4: 'invalid_jacobian',
            5: 'rank_or_condition', 6: 'rejected_step_stall', 7: 'iteration_cap',
            8: 'invalid_observation', 9: 'invalid_postfit_inverse', 10: 'svd_failure',
            11: 'invalid_affine_start'}
START_PATHS = {1: 'affine_warm_converged', 2: 'continuation_converged',
               3: 'affine_warm_failed', 4: 'continuation_failed',
               5: 'identity_converged', 6: 'identity_failed'}
CONFIG = joint.spec('G2')
# Correction (Section 14): the FD Jacobian floor requires the frozen joint limit.
GRADIENT_LIMIT = 1e-10
SELFTEST_GRADIENT_LIMIT = 1e-13


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def guard_paths():
    """Reject redirected directories, links and aliases before every write."""
    require(ROOT == joint.ROOT and ROOT.resolve() == ROOT, 'Redirected project root')
    targets = [OUT / name for name in ARTIFACTS]
    for path in [OUT, *OUT.parents, *targets]:
        require(path.resolve() == path.absolute(), 'Redirected output path: ' + str(path))
        if path.exists() or path.is_symlink():
            info = path.lstat()
            require(not path.is_symlink() and not
                    (getattr(info, 'st_file_attributes', 0) & 0x400),
                    'Symlink/junction/reparse output path: ' + str(path))
            if path in targets:
                require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
                        'Output is not an unaliased regular file: ' + str(path))
        require(path != OUT or not path.exists() or path.is_dir(), 'Output is not a directory')
    sources = [ROOT / p4_a0_data.DEFAULT, ROOT / 'P4_9PT_PLAN.md',
               *list((ROOT / 'Script').glob('*.py'))]
    for i, target in enumerate(targets):
        for other in [*sources, *targets[i+1:]]:
            require(target.resolve() != other.resolve() and not
                    (target.exists() and other.exists() and target.samefile(other)),
                    'Output aliases an input/source/other output')
    return targets


class Progress:
    def __init__(self):
        guard_paths()
        OUT.mkdir(parents=True, exist_ok=True)
        guard_paths()
        (OUT / 'progress.log').write_text('', encoding='utf-8')
        self.started = time.perf_counter()

    def __call__(self, message):
        line = (datetime.now(timezone.utc).isoformat(timespec='seconds') +
                f' elapsed_s={time.perf_counter()-self.started:.3f} ' + message)
        print(line, flush=True)
        guard_paths()
        with (OUT / 'progress.log').open('a', encoding='utf-8') as stream:
            stream.write(line + '\n')
            stream.flush()


def json_values(value):
    # data.safe spells positive infinity "inf"; compliance requires "Infinity".
    if isinstance(value, dict):
        return {str(k): json_values(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return json_values(value.tolist())
    if isinstance(value, (list, tuple)):
        return [json_values(v) for v in value]
    if isinstance(value, np.generic):
        return json_values(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return 'Infinity' if value == np.inf else '-Infinity' if value == -np.inf else None
    return value


def checked_batch(function, values, tail):
    """Isolate exceptions in checked imported batches, down to single states."""
    n = len(values)
    result = np.full((n,) + tuple(tail), np.nan)
    valid = np.zeros(n, bool)

    def visit(ids):
        try:
            with np.errstate(over='raise', divide='raise', invalid='raise'):
                candidate = np.asarray(function(values[ids]), float)
                require(candidate.shape == (len(ids),) + tuple(tail), 'Batch output shape')
                good = np.all(np.isfinite(candidate).reshape(len(ids), -1), axis=1)
                result[ids] = candidate
                valid[ids] = good
        except Exception:
            if len(ids) > 1:
                mid = len(ids) // 2
                visit(ids[:mid])
                visit(ids[mid:])

    if n:
        visit(np.arange(n))
    return result, valid


def denominator_bound(order, H, q):
    mb = 1 + 2*abs(q)
    mc = mb*(1 + 2*abs(K1)*H**2*mb**2/F**2 +
             4*abs(K2)*H**4*mb**4/F**4)
    return {'A': 1., 'B': mb, 'C': mc}[order]


def state_arrays(p, order, H, q):
    require(np.all(np.isfinite(p)), 'Nonfinite parameters')
    L = np.empty((len(p), 2, 2))
    L[:, 0, 0], L[:, 0, 1] = 1+p[:, 2], p[:, 3]
    L[:, 1, 0], L[:, 1, 1] = p[:, 4], 1+p[:, 5]
    singular = np.linalg.svd(L, compute_uv=False)
    require(np.all(np.linalg.det(L) > 0) and np.all(singular[:, -1] > 1e-10)
            and np.all(singular[:, 0]/singular[:, -1] <= COND), 'Invalid state L')
    require(np.all(1-denominator_bound(order, H, q)*np.sum(abs(p[:, 6:]), axis=1)
                   > 1e-10), 'Projective denominator bound')
    return p[:, :2], L, p[:, 6:], np.full(len(p), q)


def forward_unchecked(p, order, c, H, q, nodes=NODES):
    # Actual gaze never enters these independent state parameters.
    s, L, k, qs = state_arrays(p, order, H, q)
    u = np.ones((len(p), 2))
    d = np.broadcast_to(nodes, (len(p), len(nodes), 2))
    if order == 'A':
        t = joint.tilt(d, u, L, k, {}, CONFIG)
        z = joint.barrel_stage(t, u, s, qs, {}, ()) + s[:, None, :]
        return c + joint.camera(H*z)
    if order == 'B':
        r = joint.barrel_stage(d, u, s, qs, {}, ())
        z = joint.tilt(r, u, L, k, {}, CONFIG) + s[:, None, :]
        return c + joint.camera(H*z)
    require(order == 'C', 'Unknown order')
    r = joint.barrel_stage(d, u, s, qs, {}, ())
    v = joint.camera(H*r)/H
    return c + H*(s[:, None, :] + joint.tilt(v, u, L, k, {}, CONFIG))


def forward(p, order, c, H, q):
    return checked_batch(lambda x: forward_unchecked(x, order, c, H, q), p, (9, 2))


def inverse_points(points, c, H):
    def invert(x):
        value, status = joint.camera_inverse(x-c)
        require(np.all(status['converged']) and np.all(status['min_tangential'] > 0)
                and np.all(status['min_radial'] > 0), 'Camera inverse branch/status')
        closure = 1000*np.linalg.norm(joint.camera(value)-(x-c), axis=-1)
        require(np.all(closure < 1e-9), 'Camera forward/inverse closure')
        return value/H
    return checked_batch(invert, points, (9, 2))


def design_matrix():
    x, y = NODES.T
    return np.column_stack((np.ones(9), x, y, x*x, x*y, y*y,
                            x*x*y, x*y*y, x*x*y*y))


V = design_matrix()
ROWS = np.zeros((10, 18))
ROWS[0, 5] = ROWS[1, 12] = 1
ROWS[2, 3], ROWS[2, 13] = 1, -1
ROWS[3, 4], ROWS[3, 14] = 1, -1
ROWS[4, 7], ROWS[4, 15] = 1, -1
ROWS[5, 7] = ROWS[5, 15] = .5
ROWS[6, 6] = ROWS[7, 16] = ROWS[8, 8] = ROWS[9, 17] = 1
# Minimum-norm component-major coefficient allocation, one column per diagnostic.
ALLOC = ROWS.T / np.sum(ROWS*ROWS, axis=1)
NODE_ALLOC = np.stack((V @ ALLOC[:9], V @ ALLOC[9:]), axis=1)  # 9,2,10


def coefficients(z):
    n = len(z)
    C = np.full((n, 18), np.nan)
    valid = np.all(np.isfinite(z).reshape(n, -1), axis=1)
    if np.any(valid):
        rhs = z[valid].transpose(1, 0, 2).reshape(9, -1)
        solved = np.linalg.solve(V, rhs).reshape(9, -1, 2).transpose(1, 0, 2)
        C[valid] = solved.transpose(0, 2, 1).reshape(-1, 18)
    reconstructed = np.einsum('ij,scj->sic', V, C.reshape(n, 2, 9))
    defect = np.max(abs(reconstructed-z), axis=(1, 2))
    valid &= np.isfinite(defect) & (defect < 1e-12)
    return C, valid, defect


def diagnostic_values(C, q):
    result = C @ ROWS.T
    result[:, 5] -= q
    return result


def ratios(C):
    den = C[:, [3, 14]]
    mask = np.isfinite(den) & (abs(den) >= 1e-8)
    result = np.full(den.shape, np.nan)
    np.divide(C[:, [8, 17]], den, out=result, where=mask)
    return result, mask


def camera_jacobian(z, H):
    def calculate(x):
        columns = []
        for j in range(2):
            h = FD*np.maximum(1., abs(H*x[..., j]))
            delta = np.zeros_like(x)
            delta[..., j] = h
            columns.append((joint.camera(H*x+delta)-joint.camera(H*x-delta)) /
                           (2*h[..., None]))
        return np.stack(columns, axis=-1)
    return checked_batch(calculate, z, (9, 2, 2))


def diagnostic_scales(values, z, H):
    jac, ok = camera_jacobian(z, H)
    # Each allocated delta-z is NODE_ALLOC[:,:,j] * values[:,j].
    vectors = 1000*H*np.einsum('snab,nbj,sj->snaj', jac, NODE_ALLOC, values)
    norm = np.linalg.norm(vectors, axis=2)
    return 1000*H*values, np.sqrt(np.mean(norm**2, axis=1)), np.max(norm, axis=1), ok


def jacobian(p, order, c, H, q, nodes, nfree=8):
    coarse, fine = [], []
    valid = np.ones(len(p), bool)
    for j in range(nfree):
        h = FD*np.maximum(1., abs(p[:, j]))
        delta = np.zeros_like(p)
        delta[:, j] = h
        fp, vp = forward(p+delta, order, c, H, q)
        fm, vm = forward(p-delta, order, c, H, q)
        hp, vh = forward(p+delta/2, order, c, H, q)
        hm, vl = forward(p-delta/2, order, c, H, q)
        valid &= vp & vm & vh & vl
        coarse.append(((fp[:, nodes]-fm[:, nodes])/(2*h[:, None, None])).reshape(len(p), -1))
        fine.append(((hp[:, nodes]-hm[:, nodes])/h[:, None, None]).reshape(len(p), -1))
    a, b = np.stack(coarse, -1), np.stack(fine, -1)
    na, nb = np.linalg.norm(a, axis=(1, 2)), np.linalg.norm(b, axis=(1, 2))
    disagreement = np.linalg.norm(a-b, axis=(1, 2))/np.maximum(np.maximum(na, nb), EPS)
    valid &= np.isfinite(disagreement) & (disagreement <= 5e-5)
    return b, valid, disagreement


def affine_start(observed, order, scope, c, H, q):
    """Use only this fit's observations; ring inputs contain no traced centre."""
    chosen = np.ones(9, bool) if scope == 'nine' else OUTER
    d = NODES[chosen]

    def least_squares(template, values):
        design = np.column_stack((np.ones(len(template)), template))
        rhs = values.transpose(1, 0, 2).reshape(len(template), -1)
        coef, _, rank, _ = np.linalg.lstsq(design, rhs, rcond=None)
        require(rank == design.shape[1], 'Affine start design rank')
        return coef.reshape(design.shape[1], len(values), 2).transpose(1, 0, 2)

    def initialize(points):
        value, status = joint.camera_inverse(points-c)
        require(np.all(status['converged']) and np.all(status['min_tangential'] > 0)
                and np.all(status['min_radial'] > 0), 'Affine start camera inverse branch/status')
        require(np.all(1000*np.linalg.norm(joint.camera(value)-(points-c), axis=-1) < 1e-9),
                'Affine start camera closure')
        z = value/H
        if order == 'C':
            # Equivalent sensor-side affine homography: (X-c)/H = s+L*v,
            # v=Cam(H*R_q(d))/H. No pre-camera translation is inferred for C.
            template, target = joint.camera(H*joint.radial(d, q))/H, (points-c)/H
        else:
            template = d if order == 'A' else joint.radial(d, q)
            target = z
        coef = least_squares(template, target)
        s, L = coef[:, 0], coef[:, 1:].transpose(0, 2, 1)
        p = np.zeros((len(points), 8))
        p[:, :2] = s
        p[:, 2], p[:, 3] = L[:, 0, 0]-1, L[:, 0, 1]
        p[:, 4], p[:, 5] = L[:, 1, 0], L[:, 1, 1]-1
        state_arrays(p, order, H, q)  # Same geometry gates as the nonlinear solve.
        return p

    return checked_batch(initialize, observed, (8,))


def solve_pass(observed, order, scope, c, H, q, initial, budget, nfree=8,
               gradient_limit=GRADIENT_LIMIT):
    """One damped pass; a restricted pass freezes the two k entries at zero."""
    n = len(observed)
    chosen = np.ones(9, bool) if scope == 'nine' else OUTER
    require(observed.shape == (n, int(chosen.sum()), 2), 'Solver accepts only its fitted nodes')
    p = np.array(initial, float, copy=True)
    budget = np.broadcast_to(np.asarray(budget, int), (n,))
    require(p.shape == (n, 8) and np.all(budget >= 0), 'Solver start/budget shape')
    require(nfree in (6, 8) and (nfree == 8 or np.all(p[:, 6:] == 0)), 'Frozen k must be zero')
    reason = np.zeros(n, np.int16)
    iterations = np.zeros(n, np.int32)
    objective = np.full(n, np.inf)
    gradient = np.full(n, np.nan)
    ranks = np.zeros(n, np.int16)
    conditions = np.full(n, np.inf)
    disagreement = np.full(n, np.nan)
    damping = np.full(n, np.nan)
    last_step = np.full(n, np.inf)
    last_change = np.full(n, np.inf)
    reason[~np.all(np.isfinite(observed).reshape(n, -1), axis=1)] = 8
    for iteration in range(int(np.max(budget, initial=0))+1):
        ids = np.flatnonzero(reason == 0)
        if not len(ids):
            break
        iterations[ids] = iteration
        pred, good = forward(p[ids], order, c, H, q)
        reason[ids[~good]] = 3
        ids, pred = ids[good], pred[good]
        if not len(ids):
            continue
        residual = (pred[:, chosen]-observed[ids]).reshape(len(ids), -1)
        objective[ids] = np.sum(residual**2, axis=1)
        J, good, diff = jacobian(p[ids], order, c, H, q, chosen, nfree)
        disagreement[ids] = diff
        reason[ids[~good]] = 4
        ids, residual, J = ids[good], residual[good], J[good]
        if not len(ids):
            continue
        scales = np.linalg.norm(J, axis=1)
        good = np.all(np.isfinite(scales) & (scales > 0), axis=1)
        reason[ids[~good]] = 5
        ids, residual, J, scales = ids[good], residual[good], J[good], scales[good]
        if not len(ids):
            continue
        # A bad LAPACK batch must not discard unrelated valid states.
        packed, good = checked_batch(
            lambda x: pack_svd(x), J/scales[:, None, :], (J.shape[1]+nfree+1, nfree))
        reason[ids[~good]] = 10
        ids, residual, scales, packed = ids[good], residual[good], scales[good], packed[good]
        if not len(ids):
            continue
        m = residual.shape[1]
        U, singular, vt = packed[:, :m], packed[:, m], packed[:, m+1:]
        rank = np.sum(singular > singular[:, :1]*max(m, nfree)*EPS, axis=1)
        condition = singular[:, 0]/np.maximum(singular[:, -1], np.finfo(float).tiny)
        ranks[ids], conditions[ids] = rank, condition
        good = (rank == nfree) & (condition <= COND)
        reason[ids[~good]] = 5
        ids, residual, scales = ids[good], residual[good], scales[good]
        U, singular, vt = U[good], singular[good], vt[good]
        if not len(ids):
            continue
        utr = np.einsum('smi,sm->si', U, residual)
        g = np.max(abs(np.einsum('sij,si->sj', vt, singular*utr)), axis=1)
        g /= np.maximum(1., singular[:, 0]*np.linalg.norm(residual, axis=1))
        gradient[ids] = g
        converged = g <= gradient_limit
        step_stop = (last_step[ids] <= 1e-12) & (last_change[ids] <= 1e-14)
        reason[ids[converged]] = np.where(step_stop[converged], 2, 1)
        capped = iteration >= budget[ids]
        reason[ids[~converged & capped]] = 7
        stepping = ~converged & ~capped
        ids, scales, singular, vt, utr = (x[stepping] for x in (ids, scales, singular, vt, utr))
        if not len(ids):
            continue
        # A rejected-step attempt consumes budget too; convergence-only checks do not.
        iterations[ids] = iteration+1
        smax2 = singular[:, 0]**2
        lam = np.maximum(np.where(np.isfinite(damping[ids]), damping[ids], 1e-10*smax2),
                         EPS*smax2)
        pending = np.ones(len(ids), bool)
        for retry in range(18):
            loc = np.flatnonzero(pending)
            if not len(loc):
                break
            gain = singular[loc]/(singular[loc]**2+lam[loc, None])
            delta = -np.einsum('sij,si->sj', vt[loc], gain*utr[loc])/scales[loc]
            trial = p[ids[loc]].copy()
            trial[:, :nfree] += delta
            candidate, valid = forward(trial, order, c, H, q)
            rtrial = (candidate[:, chosen]-observed[ids[loc]]).reshape(len(loc), -1)
            obj = np.sum(rtrial*rtrial, axis=1)
            accept = valid & np.isfinite(obj) & (obj < objective[ids[loc]])
            accepted, sid = loc[accept], ids[loc[accept]]
            if len(accepted):
                old = objective[sid].copy()
                scaled_step = np.linalg.norm(scales[accepted]*delta[accept], axis=1)
                scaled_p = np.linalg.norm(scales[accepted]*p[sid, :nfree], axis=1)
                last_step[sid] = scaled_step/np.maximum(1., scaled_p)
                last_change[sid] = abs(old-obj[accept])/np.maximum(old, EPS)
                p[sid] = trial[accept]
                objective[sid] = obj[accept]
                damping[sid] = np.maximum(lam[accepted]/10, EPS*smax2[accepted])
                pending[accepted] = False
            lam[loc[~accept]] *= 10
        damping[ids[pending]] = lam[pending]
        reason[ids[pending]] = 6
    failed = ~np.isin(reason, (1, 2))
    return dict(parameters=p, failed=failed, reason=reason, iterations=iterations,
                objective_mm2=objective, gradient=gradient, rank=ranks,
                condition=conditions, jacobian_disagreement=disagreement, damping=damping,
                relative_step=last_step, relative_objective_change=last_change)


def solve_chunk(observed, order, scope, c, H, q, start='warm',
                gradient_limit=GRADIENT_LIMIT):
    """Warm attempt and short k=0 continuation share 200 iterations per state."""
    require(start in ('warm', 'identity'), 'Unknown solver start')
    n = len(observed)
    if start == 'identity':
        initial, valid = np.zeros((n, 8)), np.ones(n, bool)
    else:
        initial, valid = affine_start(observed, order, scope, c, H, q)
    result = solve_pass(observed, order, scope, c, H, q, initial, 200,
                        gradient_limit=gradient_limit)
    finite_observed = np.all(np.isfinite(observed).reshape(n, -1), axis=1)
    result['reason'][~valid & finite_observed] = 11
    result['failed'] = ~np.isin(result['reason'], (1, 2))
    result['start_path'] = np.where(result['failed'], 6 if start == 'identity' else 3,
                                    5 if start == 'identity' else 1).astype(np.int16)
    result['warm_reason'] = result['reason'].copy()
    result['warm_iterations'] = result['iterations'].copy()
    result['continuation_affine_reason'] = np.zeros(n, np.int16)
    result['continuation_affine_iterations'] = np.zeros(n, np.int32)
    result['continuation_release_reason'] = np.zeros(n, np.int16)
    result['continuation_release_iterations'] = np.zeros(n, np.int32)
    result['best_iterate_stage'] = np.ones(n, np.int16)
    # Identity is a labelled selftest comparison of the old path, without fallback.
    ids = np.flatnonzero(result['reason'] == 6) if start == 'warm' else []
    if not len(ids):
        return result
    remaining = 200-result['iterations'][ids]
    restricted = solve_pass(observed[ids], order, scope, c, H, q, initial[ids],
                            np.minimum(20, np.maximum(0, remaining-1)), nfree=6,
                            gradient_limit=gradient_limit)
    released = solve_pass(observed[ids], order, scope, c, H, q, restricted['parameters'],
                          remaining-restricted['iterations'], gradient_limit=gradient_limit)
    total = result['iterations'][ids]+restricted['iterations']+released['iterations']
    # A failed restart cannot discard the best finite diagnostic iterate. Only a
    # converged eight-parameter pass can turn a required fit into a success.
    pass_keys = tuple(released)
    for stage, candidate in ((2, restricted), (3, released)):
        better = (np.isfinite(candidate['objective_mm2']) &
                  (candidate['objective_mm2'] < result['objective_mm2'][ids]))
        if candidate is released:
            better |= ~released['failed']
        for key in pass_keys:
            if key not in ('reason', 'failed', 'iterations'):
                result[key][ids[better]] = candidate[key][better]
        result['best_iterate_stage'][ids[better]] = stage
    result['reason'][ids] = released['reason']
    result['failed'][ids] = released['failed']
    result['iterations'][ids] = total
    result['start_path'][ids] = np.where(released['failed'], 4, 2)
    result['continuation_affine_reason'][ids] = restricted['reason']
    result['continuation_affine_iterations'][ids] = restricted['iterations']
    result['continuation_release_reason'][ids] = released['reason']
    result['continuation_release_iterations'][ids] = released['iterations']
    return result


def pack_svd(J):
    U, singular, vt = np.linalg.svd(J, full_matrices=False)
    return np.concatenate((U, singular[:, None, :], vt), axis=1)


def solve(observed, order, scope, c, H, q, log, name=None, start='warm',
          gradient_limit=GRADIENT_LIMIT):
    started = time.perf_counter()
    pieces = []
    label = name or order+'.'+scope
    log(label+' start states='+str(len(observed)))
    failed = 0
    for start_row in range(0, len(observed), 1000):
        part = solve_chunk(observed[start_row:start_row+1000], order, scope, c, H, q, start,
                           gradient_limit=gradient_limit)
        pieces.append(part)
        failed += int(part['failed'].sum())
        log(f'{label} completed={min(start_row+1000, len(observed))}/{len(observed)} '
            f'failures={failed} solve_s={time.perf_counter()-started:.3f}')
    result = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    return result, time.perf_counter()-started


def aggregate(values, failed):
    """All required states remain counted; any failure forces compliance infinity."""
    values, failed = np.asarray(values), np.asarray(failed, bool)
    n = len(values)
    bad = failed | ~np.all(np.isfinite(values).reshape(n, -1), axis=1) if n else failed
    answer = dict(total_count=n, failure_count=int(bad.sum()))
    if not n:
        return dict(answer, rms=None, max=None, signed_mean=None, min=None)
    if np.any(bad):
        return dict(answer, rms=np.inf, max=np.inf, signed_mean=None, min=None)
    return dict(answer, rms=np.sqrt(np.mean(values**2, axis=0)),
                max=np.max(abs(values), axis=0), signed_mean=np.mean(values, axis=0),
                min=np.min(values, axis=0), range_max=np.max(values, axis=0))


def vector_summary(errors, failed):
    n = len(errors)
    good = aggregate(errors.reshape(n, -1) if n else np.empty((0, 1)), failed)
    if not n or good['failure_count']:
        return dict(total_count=n, failure_count=good['failure_count'],
                    rms_um=None if not n else np.inf, max_um=None if not n else np.inf,
                    signed_bias_um=None)
    norm = np.linalg.norm(errors, axis=-1)
    return dict(total_count=n, failure_count=0, rms_um=float(np.sqrt(np.mean(norm**2))),
                max_um=float(np.max(norm)),
                signed_bias_um=np.mean(errors.reshape(-1, 2), axis=0))


def parameter_summary(parameters, failed):
    summary = aggregate(parameters, failed)
    if not len(parameters) or summary['failure_count']:
        return dict(total_count=len(parameters), failure_count=summary['failure_count'],
                    min=None, max=None, mean=None, rms=None)
    return dict(total_count=len(parameters), failure_count=0, min=np.min(parameters, axis=0),
                max=np.max(parameters, axis=0), mean=np.mean(parameters, axis=0),
                rms=np.sqrt(np.mean(parameters**2, axis=0)))


def start_path_counts(paths):
    return {name: int(np.count_nonzero(paths == code)) for code, name in START_PATHS.items()}


def groups_for(ids, w, splits):
    groups = {'overall': np.ones(len(ids), bool)}
    groups.update({name: np.isin(ids, members) for name, members in splits.items()})
    groups['zero_gaze'] = np.all(w == 0, axis=1)
    r = np.linalg.norm(w, axis=1)
    groups.update(radial_0_4=(r < 4), radial_4_8=(r >= 4) & (r < 8),
                  radial_8_12p5=(r >= 8) & (r <= 12.5), corners=(r > 12.5))
    for ai, bi in ((0, 0), (0, 100), (100, 0), (100, 100)):
        groups[f'extreme_{ai}_{bi}'] = ids == ai*101+bi
    return groups


def summarize_ratios(value, defined, failed, groups, q, reference=None):
    predictions = dict(A=3*q/(1+3*q), B=(2*q+3*q*q)/(1+q)**2,
                       B_first_order_q=2*q/(1+2*q))
    result = {}
    for group, mask in groups.items():
        entry = {}
        for j, axis in enumerate(('X', 'Y')):
            m = mask & defined[:, j]
            item = dict(total_count=int(mask.sum()), defined_count=int(m.sum()),
                        undefined_count=int((mask & ~defined[:, j]).sum()),
                        failure_count=int((mask & failed).sum()),
                        values=aggregate(value[m, j:j+1], failed[m]))
            item['analytic_errors_derived_unaudited'] = {
                key: aggregate(value[m, j:j+1]-v, failed[m]) for key, v in predictions.items()}
            if reference is not None:
                ref, ref_defined = reference
                both = m & ref_defined[:, j]
                item['observed_minus_fitted'] = aggregate(
                    ref[both, j:j+1]-value[both, j:j+1], failed[both])
                item['both_defined_count'] = int(both.sum())
            # Undefined denominators are not failures; failed required fits are.
            if item['failure_count']:
                item['values'] = dict(total_count=int(mask.sum()),
                    failure_count=item['failure_count'], rms=np.inf, max=np.inf, signed_mean=None)
                for key in item['analytic_errors_derived_unaudited']:
                    item['analytic_errors_derived_unaudited'][key] = dict(item['values'])
                if reference is not None:
                    item['observed_minus_fitted'] = dict(item['values'])
            entry[axis] = item
        result[group] = entry
    return result


def save_diagnostics(prefix, values, z, H, failed, groups, maps):
    signed, rms, maximum, valid = diagnostic_scales(values, z, H)
    bad = failed | ~valid | ~np.all(np.isfinite(values), axis=1)
    maps[prefix+'_normalized'] = values
    maps[prefix+'_signed_coefficient_um'] = signed
    maps[prefix+'_sensor_rms_um'] = rms
    maps[prefix+'_sensor_max_um'] = maximum
    maps[prefix+'_failed'] = bad
    return {g: dict(signed_coefficient_um=aggregate(signed[m], bad[m]),
                    sensor_rms_um=aggregate(rms[m], bad[m]),
                    sensor_max_um=aggregate(maximum[m], bad[m])) for g, m in groups.items()}


def analyze_dataset(observed, c, H, q, groups, report, maps, log, orders=('A', 'B', 'C')):
    """D0 is completed before any state fit; predictions are retained only in memory."""
    log('D0 raw start')
    z, inverse_ok = inverse_points(observed, c, H)
    C, reconstructed, defect = coefficients(z)
    raw_bad = ~inverse_ok | ~reconstructed
    _, scale_ok = camera_jacobian(z, H)
    raw_bad |= ~scale_ok
    maps.update(raw_C=C, raw_failed=raw_bad, reconstruction_error_normalized=defect,
                raw_inverse_failed=~inverse_ok, raw_reconstruction_failed=~reconstructed,
                raw_camera_jacobian_failed=~scale_ok, true_shift_normalized=z[:, 4])
    raw = diagnostic_values(C, q)
    raw_ratio, raw_defined = ratios(C)
    maps.update(raw_ratios=raw_ratio, raw_ratio_defined=raw_defined)
    report['D0'] = dict(raw=save_diagnostics('D0_raw', raw, z, H, raw_bad, groups, maps),
                       reconstruction={g: aggregate(defect[m, None], raw_bad[m]) for g, m in groups.items()},
                       observed_ratios=summarize_ratios(raw_ratio, raw_defined, raw_bad, groups, q),
                       corrected={})
    log(f'D0 raw complete failures={int(raw_bad.sum())}')
    results, summaries, timings = {}, {}, {}
    report['fits'] = summaries
    report['raw_failure_count'] = int(raw_bad.sum())
    report['fit_failure_count'] = 0
    for order in orders:
        summaries[order] = {}
        for scope in ('nine', 'ring'):
            name = order+'.'+scope
            # The ring solver receives exactly eight outer observations, no centre truth.
            targets = observed if scope == 'nine' else observed[:, OUTER]
            fit, elapsed = solve(targets, order, scope, c, H, q, log)
            timings[name+'_solve_s'] = elapsed
            pred, forward_ok = forward(fit['parameters'], order, c, H, q)
            zp, invert_ok = inverse_points(pred, c, H)
            fit['failed'] |= ~forward_ok | ~invert_ok | raw_bad
            post_bad = (~forward_ok | ~invert_ok | raw_bad) & np.isin(fit['reason'], (1, 2))
            fit['reason'][post_bad] = 9
            if scope == 'nine':
                Cp, cp_ok, cp_defect = coefficients(zp)
                post_bad = ~cp_ok & np.isin(fit['reason'], (1, 2))
                fit['reason'][post_bad] = 9
                fit['failed'] |= ~cp_ok
            for converged_path, failed_path in ((1, 3), (2, 4), (5, 6)):
                fit['start_path'][fit['failed'] & (fit['start_path'] == converged_path)] = failed_path
            error = 1000*(pred-observed)
            null = zp[:, 4]-z[:, 4]
            key = order+'_'+scope
            for k, value in fit.items():
                maps[key+'_'+k] = value
            nine_norm = np.linalg.norm(error, axis=2)
            maps.update({key+'_nine_rms_um': np.sqrt(np.mean(nine_norm**2, axis=1)),
                         key+'_nine_max_um': np.max(nine_norm, axis=1),
                         key+'_outer_rms_um': np.sqrt(np.mean(nine_norm[:, OUTER]**2, axis=1)),
                         key+'_outer_max_um': np.max(nine_norm[:, OUTER], axis=1),
                         key+'_centre_error_um': error[:, 4],
                         key+'_sensor_residual_um': error.reshape(len(error), 18),
                         key+'_centre_norm_um': nine_norm[:, 4], key+'_centre_mm': pred[:, 4],
                         key+'_common_shift': zp[:, 4], key+'_null_normalized': null,
                         key+'_null_coefficient_um': 1000*H*null})
            summary = {}
            for g, m in groups.items():
                bad = fit['failed'][m]
                gradient = fit['gradient'][m]
                failed_gradient = gradient[bad]
                finite_failed_gradient = failed_gradient[np.isfinite(failed_gradient)]
                summary[g] = dict(nine=vector_summary(error[m], bad),
                    outer=vector_summary(error[m][:, OUTER], bad),
                    centre=vector_summary(error[m, 4], bad),
                    null_normalized=aggregate(null[m], bad),
                    null_coefficient_um=aggregate(1000*H*null[m], bad),
                    parameters=parameter_summary(fit['parameters'][m], bad),
                    converged_fd_floor_count=int(np.count_nonzero(
                        ~bad & (gradient > 1e-13) & (gradient <= GRADIENT_LIMIT))),
                    failed_final_gradient=dict(total_count=int(bad.sum()),
                        finite_count=int(finite_failed_gradient.size),
                        nonfinite_count=int(failed_gradient.size-finite_failed_gradient.size),
                        median=float(np.median(finite_failed_gradient))
                               if finite_failed_gradient.size else None,
                        max=float(np.max(finite_failed_gradient))
                            if finite_failed_gradient.size else None),
                    start_path_counts=start_path_counts(fit['start_path'][m]),
                    reason_counts={CODEBOOK[int(k)]: int(v) for k, v in
                                   zip(*np.unique(fit['reason'][m], return_counts=True))})
            summaries[order][scope] = summary
            if scope == 'nine':
                corrected_bad = fit['failed'] | ~cp_ok
                fitted_ratio, defined = ratios(Cp)
                maps[key+'_C'] = Cp
                maps[key+'_ratios'] = fitted_ratio
                maps[key+'_ratio_defined'] = defined
                maps[key+'_C_reconstruction_error'] = cp_defect
                report['D0']['corrected'][order] = dict(
                    maps=save_diagnostics('D0_'+order+'_corrected', raw-diagnostic_values(Cp, q),
                                          z, H, corrected_bad, groups, maps),
                    ratios=summarize_ratios(fitted_ratio, defined, corrected_bad, groups, q,
                                           (raw_ratio, raw_defined)))
            results[name] = dict(fit=fit, centre=pred[:, 4], error=error)
            report['fit_failure_count'] += int(fit['failed'].sum())
            report['timings'][name+'_solve_s'] = elapsed
    report['fits'] = summaries
    report.setdefault('timings', {}).update(timings)
    report['raw_failure_count'] = int(raw_bad.sum())
    report['fit_failure_count'] = sum(int(v['fit']['failed'].sum()) for v in results.values())
    pair_summary = {}
    for scope in ('nine', 'ring'):
        pair_summary[scope] = {}
        for a, b in (('A', 'B'), ('A', 'C'), ('B', 'C')):
            if a not in orders or b not in orders:
                continue
            left, right = results[a+'.'+scope], results[b+'.'+scope]
            delta = 1000*(left['centre']-right['centre'])
            bad = left['fit']['failed'] | right['fit']['failed']
            maps[f'{scope}_{a}_{b}_centre_difference_um'] = delta
            maps[f'{scope}_{a}_{b}_centre_difference_failed'] = bad
            pair_summary[scope][a+'-'+b] = {g: vector_summary(delta[m], bad[m]) for g, m in groups.items()}
    report['pair_centres'] = pair_summary
    return results


def decision(report, complete):
    fits = report.get('fits', {})
    all_orders = all(o in fits for o in ('A', 'B', 'C'))
    no_failures = (report.get('raw_failure_count', 1) == 0 and
                   report.get('fit_failure_count', 1) == 0 and all_orders)
    outer, centre = {}, {}
    for o in ('A', 'B', 'C'):
        s = fits.get(o, {}).get('ring', {}).get('overall', {})
        outer[o] = bool(s and s['outer']['max_um'] <= 1.)
        centre[o] = bool(s and s['centre']['max_um'] <= .3)
    pair = report.get('pair_centres', {}).get('ring', {})
    agree = {key: bool(v['overall']['max_um'] <= .3) for key, v in pair.items()}
    all_agree = len(agree) == 3 and all(agree.values())
    outer_centre_conflict = any(outer[o] and not centre[o] for o in outer)
    order_conflict = any(outer[a] and outer[b] and not agree.get(a+'-'+b, True)
                         for a, b in (('A', 'B'), ('A', 'C'), ('B', 'C')))
    gates = dict(full_coverage=bool(complete), no_failures=bool(no_failures),
                 all_three_orders=all_orders, ring_outer_pass=outer, ring_centre_pass=centre,
                 ring_pair_agreement=agree, all_ring_pairs_agree=all_agree,
                 outer_pass_centre_fail=outer_centre_conflict,
                 outer_passing_order_centre_conflict=order_conflict)
    if not complete or not no_failures:
        result, why = 'ambiguous', 'Incomplete full-grid coverage or a required diagnostic/fit failed.'
    elif all(outer.values()) and all(centre.values()) and all_agree:
        result, why = 'not needed', 'All three ring fits and all pairwise ring centres pass.'
    elif outer_centre_conflict or order_conflict:
        result, why = 'needed: null mode', 'Ring outer fits demonstrate a centre/null-mode conflict.'
    else:
        result, why = 'ambiguous', 'D2/denser sampling needed to resolve aliasing; no D2 tie was tested.'
    return dict(value=result, reason=why, gates=gates,
                sampled_gates_only=not complete,
                limitation='No re-trace indicated means only at the sampled field nodes; '
                'it proves neither between-node correctness nor that only gaze laws remain. '
                'No result authorizes a CODE V trace.')


def initial_report(args):
    return dict(status='started', qualification='noise-free design traces; unaudited',
        provenance=dict(script=str(Path(__file__).absolute()),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            data_module=str(Path(p4_a0_data.__file__).absolute()),
            joint_module=str(Path(joint.__file__).absolute()), numpy_version=np.__version__),
        mode='selftest' if args.selftest else 'quick' if args.quick else 'full',
        settings=dict(workers_requested=args.workers, workers_effective=1,
            execution='Serial NumPy batches; --workers is accepted but no process pool is used.',
            chunk_max_states=1000, iteration_cap=200, damping_retries=18,
            finite_difference=FD, jacobian_disagreement_limit=5e-5, gradient_limit=GRADIENT_LIMIT,
            relative_step_limit=1e-12, relative_objective_limit=1e-14, condition_limit=COND,
            start='Per-state observed affine warm start; k=0 and q=q0.',
            continuation='On rejected-step stall: restart affine seed, fit s+L with k=0 for at most '
                         '20 remaining iterations, then release k; all stages together <=200 '
                         'step attempts per state. Convergence-only checks consume no step budget.',
            camera=dict(F=F, K1=K1, K2=K2), fixed_q=True),
        definitions=dict(parameters=PARAMETERS, diagnostics=DIAGNOSTICS,
            C_order='component-major X then Y; 1,x,y,x2,xy,y2,x2y,xy2,x2y2',
            nodes='-metadata labels, validated y-major order; centre index 4',
            protocol='Fixed centre-assisted diagnostic protocol, not an image-only fitted-model freeze.',
            D0_raw='Exact model-free interpolation before D1; barrel is not inverted.',
            D0_corrected='Model-relative: raw D(C) minus exact nine-fit D(C_hat), not a first-order subtraction.',
            coefficient_scale='1000 H D, undistorted signed coefficient um; not a sensor point error.',
            allocation='delta_C = row.T * violation / squared_row_norm; constant -q has zero derivative.',
            sensor_scale='1000 H J_camera(H z) V delta_C; local sensor-equivalent node vector RMS/max.',
            D1='Independent 8-parameter fits at ground-truth gaze; fixed camera and calibration; no gaze recovery.',
            gradient_convergence='||J_s.T*r||inf/max(1,sigma_max*||r||2) <=1e-10, matching '
                                 'the frozen joint solver; the former 1e-13 limit is below the '
                                 'finite-difference Jacobian floor (disagreement limit 5e-5). '
                                 'Selftest recovery retains parameters <1e-10 and predictions <1e-9 um.',
            converged_fd_floor_count='Successful states with final stored normalized gradient '
                                     'strictly >1e-13 and <=1e-10; counted per order/scope/group. '
                                     'Gradients >1e-10 remain failures.',
            failed_final_gradient='Median/max of finite final stored normalized gradients of failed '
                                  'states at retained best iterates; total/finite/nonfinite counts '
                                  'are explicit, and no finite values gives unavailable statistics. '
                                  'Best-iterate nine-point and ring (outer) residuals remain in maps '
                                  'for every failed state; these diagnostics do not affect compliance.',
            residual_dof=dict(nine=10, ring=8),
            A='c+Cam(H*(R_q(T(d))+s)); tilt then barrel, G2',
            B='c+Cam(H*(T(R_q(d))+s)); barrel then tilt, order_control',
            C='c+H*(s+T(Cam(H*R_q(d))/H)); full free sensor homography, translation after division',
            C_interpretation='Term-form analogy, not an eye-physics claim or a rigid sensor rotation; no forced m=-ka relation.',
            shift_units='A/B: normalized before-camera coordinates; C: normalized sensor coordinates.',
            affine_start='Only fitted observations, no gaze or traced centre in ring fits. '
                         'A: affine z=s+L*d in pre-camera coordinates (the full solve retains '
                         'barrel nonlinearity). B: affine z=s+L*R_q(d). '
                         'C: equivalent sensor-side affine (X-c)/H=s+L*v with '
                         'v=Cam(H*R_q(d))/H; observed z is camera-inverted and closure-checked.',
            null_map='Cam_inverse(F_ring(0)-c)/H - traced z_c; model-dependent common-frame shift, never ring mean.',
            metrics='Sensor vector RMS=sqrt(mean(point Euclidean norm squared)); max=maximum point norm.',
            aggregates='Any failed required state forces compliance RMS/max Infinity and unavailable signed/parameter summaries.',
            holdouts='Per-state holdout fits are diagnostics, not predictions from a trained gaze law.',
            ratio_mask='Undefined for abs(normalized quadratic denominator)<1e-8; retain mask/count.',
            analytic_checks='Derived, not numerically audited; ratios only at L=I, first order in k.',
            ratio_comparisons='A/B analytic values are comparison limits, not arbitrary-L,k order falsifiers; C has no asserted scalar law.',
            status_encoding='Numeric per-state reason codes in maps; textual codebook and per-group reason counts in summaries.',
            start_path_encoding='Numeric start_path map with outcome-labelled codebook; per-group '
                                'start_path_counts retain failed states. warm_reason/iterations and '
                                'continuation_affine/release reason/iterations retain stage history; '
                                'zero continuation reason means not attempted. Failed fits retain '
                                'the lowest finite objective across stages, with the terminal failure reason. '
                                'best_iterate_stage: 1=initial full solve, 2=k-zero restricted, 3=k released.',
            interpretation='D0 names do not identify unique aberrations on 3x3; no D2, parity inference, or missing-shape claim.',
            source_io='Whole pickle temporarily deserialized by imported loaders; only A=0 retained/analysed; no cache.'),
        solver_codebook=CODEBOOK, start_path_codebook=START_PATHS, errors=[], timings={})


def load_real(args, report, maps, log):
    started = time.perf_counter()
    path = (joint.ROOT / p4_a0_data.DEFAULT).resolve()
    report['input'] = str(path)
    report['schema'] = p4_a0_data.EXPECTED
    log('load outer start')
    cube, fields = joint.load_a0_outer(path, report)
    require(np.array_equal(fields, LABELS) and len(np.unique(fields, axis=0)) == 9
            and np.array_equal(np.flatnonzero(np.all(fields == 0, axis=1)), [4]),
            'Metadata labels must be y-major and match ray indices, centre index 4')
    calibration = {}
    outer, _, c, H, q = joint.calibrate(cube, fields, calibration)
    require(np.array_equal(outer, OUTER), 'Outer/centre mask mismatch')
    report['calibration'] = calibration
    w_all, splits = joint.additive.partition(cube)
    report['split_counts_full'] = {k: len(v) for k, v in splits.items()}
    report['selection'] = dict(frozen=True,
        protocol=report['definitions']['protocol'], fixed_diagnostics=['A', 'B', 'C'],
        fixed_scopes=['nine', 'ring'], fixed_q=True)
    log('fixed centre-assisted protocol frozen; load traced centre')
    centre = joint.load_a0_centre(path, cube, fields, report)
    report['selection']['centre_truth_first_access'] = (
        'load_a0_centre after fixed centre-assisted diagnostic protocol was frozen; '
        'not an image-only fitted-model freeze')
    report['validation']['scope'] = (
        'Full schema/metadata; all A=0 values, labels, statuses and geometry; '
        'centre geometry checked after fixed centre-assisted diagnostic definitions were frozen')
    observed = np.empty((10201, 9, 2))
    observed[:, OUTER] = np.stack((cube['xsi'], cube['ysi']), -1).reshape(10201, 9, 2)[:, OUTER]
    observed[:, 4] = centre
    ids = np.arange(10201)
    if args.quick:
        axis = (ids//101 == 50) | (ids % 101 == 50)
        interior = ids[(ids//101 > 0) & (ids//101 < 100) &
                       (ids % 101 > 0) & (ids % 101 < 100) & ~axis]
        sample = np.random.default_rng(0).choice(interior, 32, replace=False)
        ids = np.unique(np.r_[ids[axis], [0, 100, 10100, 10200], sample])
        require(len(ids) == 237 and int(axis.sum()) == 201, 'Quick sample count')
    w = w_all[ids]
    maps.update(state_id=ids, ade_index=ids//101, bde_index=ids % 101,
                ade_deg=-w[:, 1], bde_deg=w[:, 0], gaze_w_deg=w, gaze_u=w/12.5)
    groups = groups_for(ids, w, splits)
    for name, mask in groups.items():
        maps['group_'+name] = mask
    report['coverage'] = dict(states=len(ids), full_states=10201, full=not args.quick,
        includes_zero=bool(5100 in ids), group_counts={k: int(v.sum()) for k, v in groups.items()})
    report['calibration']['denominator_coordinate_bounds'] = {
        order: denominator_bound(order, H, q) for order in ('A', 'B', 'C')}
    report['analytic_ratios_derived_unaudited'] = dict(A=3*q/(1+3*q),
        B=(2*q+3*q*q)/(1+q)**2, B_first_order_q=2*q/(1+2*q), C=None)
    report['timings']['load_s'] = time.perf_counter()-started
    log(f'load complete states={len(ids)} load_s={report["timings"]["load_s"]:.3f}')
    return observed[ids], c, H, q, groups


def imported_single(p, order, c, H, q):
    adapter = dict(a_x=p[0], a_y=p[1], m_xx=p[2], h_x=p[3], h_y=p[4],
                   m_yy=p[5], k_x=p[6], k_y=p[7])
    s, L, k, qs = joint.state_parts(np.ones((1, 2)), adapter, q)
    require(np.max(abs(s-p[:2])) < 1e-15 and
            np.max(abs(L[0]-[[1+p[2], p[3]], [p[4], 1+p[5]]])) < 1e-15,
            'Single-state adapter mismatch')
    configuration = joint.spec('G2', () if order == 'A' else ('barrel_before_tilt',))
    theta = np.array([adapter.get(name, 0.) for name in configuration['names']])
    if order in ('A', 'B'):
        z = joint.normalized_map(configuration, theta, [[1., 1.]], NODES, {'q0': q})
        return c+joint.camera(H*z)[0]
    # C uses the imported stages but no normalized_map analogue exists for a sensor homography.
    d = np.broadcast_to(NODES, (1, 9, 2))
    r = joint.radial(d, q)
    return (c+H*(s[:, None]+joint.tilt(joint.camera(H*r)/H,
                                     np.ones((1, 2)), L, k, {}, CONFIG)))[0]


def first_order(p, q, order):
    a, b, cl, e, kx, ky = p[2:]
    t = 3*q if order == 'A' else 2*q+3*q*q
    return np.array([0., 0., t*kx, -t*ky,
        q*(e-a) if order == 'A' else q*(a-e),
        (1.5 if order == 'A' else .5)*q*(a+e),
        q*(3*b+2*cl) if order == 'A' else q*b,
        q*(2*b+3*cl) if order == 'A' else q*cl, -t*kx, -t*ky])


def run_selftest(report, maps, log):
    c, H, q = np.zeros(2), 2.59, -.03
    report['coverage'] = dict(states=9, full_states=10201, full=False,
                              synthetic=True, real_input_loaded=False)
    report['calibration'] = dict(c_mm=c, H_mm=H, q0=q, synthetic=True,
        denominator_coordinate_bounds={o: denominator_bound(o, H, q) for o in ('A', 'B', 'C')})
    checks = []

    def check(name, condition, value=None, *, gradient_limit=None, optional=False):
        checks.append(dict(name=name, passed=bool(condition), measured=value,
                           gradient_limit=gradient_limit, optional=optional))
        limit = 'not_applicable' if gradient_limit is None else f'{gradient_limit:g}'
        log('selftest '+name+' '+('ok' if condition else 'failed')+
            ' gradient_limit='+limit+' '+('optional' if optional else 'mandatory'))

    known = np.zeros((9, 8))
    for j in range(1, 9):
        known[j] = [.01*j, -.007*j, .001*j, .0003*j, -.0002*j,
                    -.0005*j, .0004*j, -.0003*j]
    maps.update(state_id=np.arange(9), selftest_known_parameters=known)
    report['selftest'] = dict(checks=checks, starts={}, identity_start_failure_counts={},
        identity_start_policy='Old independent identity path, without continuation; failures are '
                              'optional, labelled and counted separately. Recovery checks apply '
                              'to its converged states and are also optional. '
                              'Every warm-start state must converge and recover known parameters.')
    for order in ('A', 'B', 'C'):
        truth = np.stack([imported_single(p, order, c, H, q) for p in known])
        batch, valid = forward(known, order, c, H, q)
        wiring = float(np.max(1000*np.linalg.norm(batch-truth, axis=2)))
        check(order+'_import_wiring', valid.all() and wiring < 1e-9, wiring)
        z, ok = inverse_points(truth, c, H)
        closure = float(np.max(1000*np.linalg.norm(c+joint.camera(H*z)-truth, axis=2)))
        check(order+'_camera_closure', ok.all() and closure < 1e-9, closure)
        C, reconstructed, defect = coefficients(z)
        maps[order+'_selftest_raw_C'] = C
        check(order+'_V_reconstruction', reconstructed.all() and np.max(defect) < 1e-12,
              float(np.max(defect)))
        for scope in ('nine', 'ring'):
            targets = truth if scope == 'nine' else truth[:, OUTER]
            label = order+'.'+scope
            report['selftest']['starts'][label] = {}
            for start in ('warm', 'identity'):
                check_options = dict(gradient_limit=SELFTEST_GRADIENT_LIMIT,
                                     optional=start == 'identity')
                fit, elapsed = solve(targets, order, scope, c, H, q, log,
                                     'selftest.'+label+'.'+start, start=start,
                                     gradient_limit=SELFTEST_GRADIENT_LIMIT)
                pred, valid = forward(fit['parameters'], order, c, H, q)
                assessed = np.ones(len(known), bool) if start == 'warm' else ~fit['failed']
                max_um = float(np.max(1000*np.linalg.norm(pred[assessed]-truth[assessed], axis=2), initial=0.))
                param_error = float(np.max(abs(fit['parameters'][assessed]-known[assessed]), initial=0.))
                parameter_sensor_um = float(np.max(1000*np.linalg.norm(pred[assessed]-batch[assessed], axis=2), initial=0.))
                key = order+'_'+scope+('' if start == 'warm' else '_identity')
                for k, value in fit.items():
                    maps[key+'_'+k] = value
                maps[key+'_centre_error_um'] = 1000*(pred[:, 4]-truth[:, 4])
                failures = int(fit['failed'].sum())
                report['selftest']['starts'][label][start] = dict(
                    total_count=len(known), failure_count=failures,
                    gradient_limit=SELFTEST_GRADIENT_LIMIT, optional=start == 'identity',
                    assessed_recovery_count=int(assessed.sum()),
                    start_path_counts=start_path_counts(fit['start_path']),
                    reason_counts={CODEBOOK[int(k)]: int(v) for k, v in
                                   zip(*np.unique(fit['reason'], return_counts=True))},
                    max_prediction_um=max_um, max_parameter_error=param_error,
                    max_parameter_sensor_displacement_um=parameter_sensor_um)
                if start == 'warm':
                    check(key+'_no_failures', failures == 0 and valid.all(), failures,
                          **check_options)
                else:
                    report['selftest']['identity_start_failure_counts'][label] = failures
                    log('selftest '+label+' identity_start_failures_allowed='+str(failures))
                    check(key+'_converged_forward_valid', valid[assessed].all(), **check_options)
                check(key+'_iteration_budget', np.all(fit['iterations'] <= 200) and
                      np.array_equal(fit['iterations'], fit['warm_iterations']+
                                     fit['continuation_affine_iterations']+
                                     fit['continuation_release_iterations']),
                      int(np.max(fit['iterations'])), **check_options)
                check(key+'_predictions_including_centre', max_um < 1e-9, max_um,
                      **check_options)
                check(key+'_parameter_recovery', param_error < 1e-10, param_error,
                      **check_options)
                check(key+'_parameter_sensor_displacement', parameter_sensor_um < 1e-9,
                      parameter_sensor_um, **check_options)
                zp, inv_ok = inverse_points(pred, c, H)
                Cp, cp_ok, _ = coefficients(zp)
                corrected = diagnostic_values(C, q)-diagnostic_values(Cp, q)
                maps[key+'_corrected_coefficient_um'] = 1000*H*corrected
                if order == 'A':
                    err = float(np.max(abs(1000*H*corrected[assessed]), initial=0.))
                    check(key+'_corrected_D0', inv_ok[assessed].all() and cp_ok[assessed].all()
                          and err < 1e-9, err, **check_options)
                report['timings'][key+'_solve_s'] = elapsed
    # Analytic checks are deliberately independent of fitting.
    h = 1e-6
    for order in ('A', 'B'):
        for axis in (0, 1):
            p = np.zeros((2, 8))
            p[0, 6+axis], p[1, 6+axis] = h, -h
            sensor, ok = forward(p, order, c, H, q)
            z, inv = inverse_points(sensor, c, H)
            C, rec, _ = coefficients(z)
            odd = (C[0]-C[1])/(2*h)
            den, num = (3, 8) if axis == 0 else (14, 17)
            expected = 3*q/(1+3*q) if order == 'A' else (2*q+3*q*q)/(1+q)**2
            observed_ratio = odd[num]/odd[den]
            check(f'{order}_ratio_limit_{axis}', ok.all() and inv.all() and rec.all()
                  and abs(observed_ratio-expected) < 1e-6, observed_ratio-expected)
        for j in range(2, 8):
            p = np.zeros((2, 8))
            p[0, j], p[1, j] = h, -h
            sensor, ok = forward(p, order, c, H, q)
            z, inv = inverse_points(sensor, c, H)
            C, rec, _ = coefficients(z)
            actual = np.diff(diagnostic_values(C, q), axis=0)[0]/(-2*h)
            expected = (first_order(p[0], q, order)-first_order(p[1], q, order))/(2*h)
            error = float(np.max(abs(actual-expected)))
            check(f'{order}_table_derivative_{PARAMETERS[j]}', ok.all() and inv.all()
                  and rec.all() and error < 1e-6, error)
    for order in ('A', 'B', 'C'):
        invalid = np.zeros((2, 8))
        invalid[1, 6] = 2.
        pred, ok = forward(invalid, order, c, H, q)
        check(order+'_invalid_denominator_isolation', ok[0] and not ok[1])
        combined = vector_summary(np.zeros((2, 9, 2)), ~ok)
        check(order+'_failure_infinity_rule', combined['total_count'] == 2 and
              combined['failure_count'] == 1 and combined['max_um'] == np.inf and
              combined['rms_um'] == np.inf and combined['signed_bias_um'] is None)
    # Near-zero ratio denominators are explicitly masked, not divided through.
    r, defined = ratios(np.zeros((1, 18)))
    check('undefined_ratio_mask', not defined.any() and np.isnan(r).all())
    check('empty_group_unavailable', vector_summary(np.empty((0, 9, 2)), np.zeros(0, bool))['max_um'] is None)
    report['selftest']['failure_count'] = sum(not c['passed'] for c in checks if not c['optional'])
    report['selftest']['optional_failure_count'] = sum(not c['passed'] for c in checks if c['optional'])
    report['selftest']['passed'] = report['selftest']['failure_count'] == 0
    report['decision'] = dict(value='ambiguous', reason='Synthetic selftest has no full-grid coverage.',
                              gates=dict(full_coverage=False), sampled_gates_only=True)
    if not report['selftest']['passed']:
        report['errors'].append('Selftest assertions failed; see selftest.checks (tolerances unchanged).')
    log('selftest '+('PASS' if report['selftest']['passed'] else 'FAIL')+
        f" mandatory_failure_count={report['selftest']['failure_count']}"+
        f" optional_failure_count={report['selftest']['optional_failure_count']}"+
        ' optional_identity_start_solver_failure_count='+
        str(sum(report['selftest']['identity_start_failure_counts'].values())))


def text_report(report):
    lines = ['qualification = noise-free design traces; unaudited']
    for order, scopes in report.get('fits', {}).items():
        for scope, grouped in scopes.items():
            overall = grouped.get('overall', {})
            for metric in ('nine', 'outer', 'centre'):
                for name in ('rms_um', 'max_um'):
                    value = overall.get(metric, {}).get(name)
                    lines.append(f'{order}.{scope}.{metric}.{name} = {json_values(value)}')
            for path, count in overall.get('start_path_counts', {}).items():
                lines.append(f'{order}.{scope}.start_path_counts.{path} = {count}')

    def visit(value, path):
        if isinstance(value, dict):
            for k, v in value.items():
                visit(v, path+'.'+str(k) if path else str(k))
        elif isinstance(value, list):
            # Short vectors/codebooks remain on a single metric line.
            if any(isinstance(v, dict) for v in value):
                for i, v in enumerate(value):
                    visit(v, path+'.'+str(i))
            else:
                lines.append(path+' = '+json.dumps(value, allow_nan=False))
        else:
            lines.append(path+' = '+str(value))
    visit(json_values(report), '')
    return '\n'.join(lines)+'\n'


def finalize(report, maps, log):
    # Write available artifacts independently, even after an analysis failure.
    for key, value in list(maps.items()):
        value = np.asarray(value)
        if not (value.dtype.kind in 'biuf' and value.ndim in (1, 2)
                and len(value) <= 10201 and (value.ndim == 1 or value.shape[1] <= 20)):
            report['errors'].append('Rejected invalid map shape/type: '+key)
            del maps[key]
    report['status'] = 'completed_with_failures' if report['errors'] else 'completed'
    try:
        guard_paths()
        with (OUT / 'maps.npz').open('wb') as stream:
            np.savez_compressed(stream, **maps)
    except Exception as exc:
        report['errors'].append('maps.npz: '+repr(exc))
    for attempt in range(2):
        report['status'] = 'completed_with_failures' if report['errors'] else 'completed'
        count = len(report['errors'])
        for name, render in (
            ('report.txt', lambda: text_report(report)),
            ('results.json', lambda: json.dumps(p4_a0_data.safe(json_values(report)),
                                               indent=2, allow_nan=False)+'\n')):
            try:
                payload = render()
                guard_paths()
                (OUT / name).write_text(payload, encoding='utf-8')
            except Exception as exc:
                report['errors'].append(name+': '+repr(exc))
        if len(report['errors']) == count:
            break
    print('status='+report['status'], flush=True)
    return int(bool(report['errors']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--quick', action='store_true')
    mode.add_argument('--selftest', action='store_true')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be a positive integer')
    report, maps = initial_report(args), {}
    try:
        log = Progress()
    except Exception as exc:
        print('Unsafe/unavailable output paths: '+repr(exc), file=sys.stderr, flush=True)
        return 1
    started = time.perf_counter()
    log('mode='+report['mode']+'; noise-free design traces; unaudited')
    try:
        if args.selftest:
            run_selftest(report, maps, log)
        else:
            observed, c, H, q, groups = load_real(args, report, maps, log)
            analyze_dataset(observed, c, H, q, groups, report, maps, log)
            report['decision'] = decision(report, report['coverage']['full'])
            for key, value in report['decision']['gates'].items():
                if isinstance(value, dict):
                    for part, passed in value.items():
                        log('gate.'+key+'.'+part+'='+str(passed))
                else:
                    log('gate.'+key+'='+str(value))
            log('decision='+report['decision']['value']+'; '+report['decision']['reason'])
            if args.quick:
                solve_s = sum(v for k, v in report['timings'].items() if k.endswith('_solve_s'))
                report['timings']['full_estimate_s'] = report['timings']['load_s']+solve_s*10201/237
                report['timings']['estimate_warning'] = 'Measured load plus solve time * 10201/237; harder states can take longer.'
            if report['raw_failure_count'] or report['fit_failure_count']:
                report['errors'].append('Required D0/D1 failures; retained in maps, counts and compliance summaries.')
    except Exception as exc:
        report['errors'].append(type(exc).__name__+': '+str(exc))
        report.setdefault('decision', dict(value='ambiguous', reason='Required analysis stage failed.',
                                          gates=dict(full_coverage=False, no_failures=False)))
        if args.selftest:
            report.setdefault('selftest', {})['passed'] = False
            log('selftest FAIL '+repr(exc))
        else:
            log('analysis failure '+repr(exc))
    report['timings']['total_analysis_s'] = time.perf_counter()-started
    log('analysis complete; failures='+str(len(report['errors'])))
    return finalize(report, maps, log)


if __name__ == '__main__':
    raise SystemExit(main())
