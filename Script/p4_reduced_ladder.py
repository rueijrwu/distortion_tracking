"""Noise-free, staged, sparse-calibration physical P4 ladder. No CODE V calls.

Run with the project's explicitly named Python interpreter and -B. All artifacts
go to data/p4_reduced_ladder; existing inputs and programs are read-only.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys
sys.dont_write_bytecode = True
import hashlib
import pickle
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import RegularGridInterpolator
from p4_cr_floor import difference

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/distortion_grid_p4_biaxial_macro/distortion_grid.pkl'
OUT = ROOT/'data/p4_reduced_ladder'
BENCH = ROOT/'data/p4_combined_brown/baseline_check/benchmark.npz'
K = np.array([-.0185004967, .00023989235714, .0015756063948,
              -.0014256378340, .00066200789981])
S1 = -.081595476571
RAYS = np.array([(x, y) for x in (-1., 0., 1.) for y in (-1., 0., 1.)])
POSES = np.array([-12.4, -6.4, 0., 6.4, 12.4])
NAMES = ['edge_mm', 'corner_coordinate_mm', 'Rx_mm', 'Ry_mm',
         'cxx', 'cxy', 'cyx', 'cyy', 'qx_per_mm', 'qy_per_mm',
         'Rx_A_mm', 'Ry_A_mm', 'cxx_A', 'cxy_A', 'cyx_A', 'cyy_A',
         'qx_A_per_mm', 'qy_A_per_mm', 'pivot_hx_mm', 'pivot_hy_mm']
NPAR = (4, 10, 18, 20)
TARGET_A, TARGET_G = .1, 1/60  # D, radial degrees


def log(s):
    print(s, flush=True)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def load():
    with DATA.open('rb') as f:
        rows = pickle.load(f)
    names = ('accommodation_d', 'eye_rotation_deg', 'eye_rotation_x_deg',
             'field_x_relative', 'field_y_relative')
    axes = [np.unique(rows[n]) for n in names]
    assert tuple(map(len, axes)) == (21, 101, 101, 3, 3)
    assert len(rows) == 21*101*101*9
    for ax, expected in zip(axes[:3], (np.linspace(0, 4, 21),
                                     np.linspace(-20, 20, 101), np.linspace(-20, 20, 101))):
        assert np.allclose(ax, expected)
    assert np.array_equal(axes[3], [-1, 0, 1]) and np.array_equal(axes[4], [-1, 0, 1])
    ai, di, bi, xi, yi = [np.searchsorted(ax, rows[n]) for ax, n in zip(axes, names)]
    key = (((ai*101+di)*101+bi)*9+xi*3+yi)
    assert np.unique(key).size == len(rows)
    flat = np.empty((len(rows), 2))
    flat[key] = np.stack((rows['center_absolute_x_mm']+rows['real_x_mm'],
                          rows['center_absolute_y_mm']+rows['real_y_mm']), axis=1)
    cube = flat.reshape(21, 101, 101, 9, 2)
    assert np.isfinite(cube).all(), 'Missing/nonfinite positions; do not silently omit states'
    coords = np.stack(np.meshgrid(*axes[:3], indexing='ij'), axis=-1).reshape(-1, 3)
    return axes[:3], coords, cube


def engine(cube, offset=1):
    j = np.stack([difference(cube, a, h, offset)[0]
                  for a, h in enumerate((.2, .4, .4))], axis=-1)
    g = np.einsum('nri,nrj->nij', j, j)
    ev = np.linalg.eigvalsh(g)
    assert (ev[:, 0] > ev[:, -1]*18*np.finfo(float).eps).all(), 'Rank deficient truth Jacobian'
    projection = np.linalg.solve(g, j.swapaxes(1, 2))
    return j, projection, ev[:, -1]/ev[:, 0]


def forward(c, p, rung):
    """Fixed image frame: x is opposite field_x; y is opposite field_y.

    p has 20 slots; only the first NPAR[rung] are active. q is in mm^-1.
    Gaze law is predeclared tan, avoiding held-out or CV-based law selection.
    """
    A, ade, bde = np.asarray(c).T
    a = A/4
    kval = sum(K[j]*a**j for j in range(5))
    scale = 1+S1*a
    radii = np.where(np.sum(abs(RAYS), axis=1) == 1, p[0], p[1])
    r0 = -RAYS*radii[:, None]
    radius = np.linalg.norm(r0, axis=1)
    rho = radius.copy()
    for _ in range(12):
        rho -= (rho+K[0]*rho**3-radius)/(1+3*K[0]*rho**2)
    u0 = r0*np.divide(rho, radius, out=np.ones_like(rho), where=radius > 0)[:, None]
    z = scale[:, None, None]*u0
    b = z*(1+kval[:, None, None]*np.sum(z*z, axis=-1, keepdims=True))
    ang_a, ang_b = np.deg2rad(ade), np.deg2rad(bde)
    ta, tb = np.tan(ang_a), np.tan(ang_b)
    # Each pre-existing group receives exactly one linear A/4 correction in R2.
    groups = np.broadcast_to(p[2:10], (len(c), 8)).copy()
    if rung >= 2:
        groups += a[:, None]*p[10:18]
    rx, ry, cxx, cxy, cyx, cyy, qx, qy = groups.T
    trans = np.stack((rx*tb, ry*ta), axis=-1)
    if rung >= 1:
        sa, sb = 1/np.cos(ang_a), 1/np.cos(ang_b)
        sx = 1+cxx*(sb**2-1)+cxy*(sa-1)
        sy = 1+cyx*(sb-1)+cyy*(sa**2-1)
        den = 1+qx[:, None]*tb[:, None]*b[..., 0]+qy[:, None]*ta[:, None]*b[..., 1]
        w = np.stack((sx[:, None]*b[..., 0], sy[:, None]*b[..., 1]), -1)/den[..., None]
    else:
        w = b
    if rung >= 3:
        # Rotating pivot relative to the labeled central ray. First-order
        # expansion of D_k(v-d)-D_k(-d) predicts p1=-k*d_y, p2=-k*d_x.
        dx, dy = -p[18]*np.sin(ang_b), -p[19]*np.sin(ang_a)
        p1, p2 = -kval*dy, -kval*dx
        x, y = w[..., 0], w[..., 1]
        w = np.stack((x+2*p1[:, None]*x*y+p2[:, None]*(3*x*x+y*y),
                      y+p1[:, None]*(x*x+3*y*y)+2*p2[:, None]*x*y), -1)
    return (w+trans[:, None, :]).reshape(-1, 18)


def initial():
    p = np.zeros(20)
    p[:4] = [1.2753, 1.2354, -15.5, 15.5]
    p[4:8] = 1
    return p


def fit(c, y, ids, rung, start=None, removed=()):
    p = initial() if start is None else start.copy()
    p[NPAR[rung]:] = 0
    p[list(removed)] = 0
    free = np.array([j for j in range(NPAR[rung]) if j not in removed], dtype=int)
    lo = np.array([.1, .1, -100, -100]+[-10]*4+[-2]*2+[-100]*2+[-20]*4+[-4]*2+[-100]*2)
    hi = -lo
    hi[:2] = 2
    def residual(x):
        pp = p.copy(); pp[free] = x
        return (forward(c[ids], pp, rung)-y[ids]).ravel()
    f = least_squares(residual, p[free], bounds=(lo[free], hi[free]),
                      x_scale='jac', max_nfev=500, ftol=1e-11, xtol=1e-11, gtol=1e-10)
    if not f.success or not np.isfinite(f.fun).all():
        raise RuntimeError(f'R{rung} optimizer failure: {f.message}')
    p[free] = f.x
    sv = np.linalg.svd(f.jac, compute_uv=False)
    rank = np.linalg.matrix_rank(f.jac)
    if rank != len(free):
        raise RuntimeError(f'R{rung} calibration rank {rank}/{len(free)}')
    normj = f.jac/np.linalg.norm(f.jac, axis=0)
    return p, dict(cost=float(f.cost), optimality=float(f.optimality),
                   nfev=f.nfev, rank=int(rank), condition=float(sv[0]/sv[-1]),
                   scaled_condition=float(np.linalg.cond(normj)))


def project(pred, truth, P):
    # User's r = candidate - truth. The actual candidate inverse bias has
    # the opposite first-order sign. Absolute/radial summary metrics agree.
    return np.einsum('nij,nj->ni', P, pred-truth)


def triple(x):
    assert np.isfinite(x).all() and len(x)
    return np.array([np.sqrt(np.mean(x*x)), np.percentile(x, 95), np.max(x)])


def metrics(d, ids):
    return np.r_[triple(abs(d[ids, 0])), triple(np.linalg.norm(d[ids, 1:], axis=1)*60)]


def passes(m):
    return bool(m[2] <= TARGET_A and m[5] <= TARGET_G*60)


def loo(c, y, P, ids, rung, start, removed=()):
    d = np.empty((len(ids), 3))
    for i, held in enumerate(ids):
        train = np.delete(ids, i)
        p, _ = fit(c, y, train, rung, start, removed)
        d[i] = project(forward(c[held:held+1], p, rung), y[held:held+1], P[held:held+1])[0]
    return d, metrics(d, np.arange(len(ids)))


def numerical_jac(fun, x):
    columns = []
    for j, eps in enumerate((1e-4, 1e-4, 1e-4)):
        h = np.zeros_like(x); h[:, j] = eps
        columns.append((fun(x+h)-fun(x-h))/(2*eps))
    return np.stack(columns, axis=-1)


def invert(fun, obs, start, maxiter=60):
    """Batched local least-squares inverse, no clipping of final A or gaze.

    Trust limits apply to steps only. Armijo-free monotone line search and
    stationarity checks allow nonzero model residual at a valid minimum.
    """
    x = start.copy(); converged = np.zeros(len(x), bool)
    iterations = np.zeros(len(x), int)
    for it in range(maxiter):
        active = np.flatnonzero(~converged)
        if not len(active):
            break
        xx = x[active]; pred = fun(xx); residual = obs[active]-pred
        j = numerical_jac(fun, xx)
        g = np.einsum('nri,nrj->nij', j, j)
        rhs = np.einsum('nri,nr->ni', j, residual)
        grad = np.max(abs(rhs)/np.maximum(np.linalg.norm(j, axis=1), 1e-30), axis=1)
        step = np.linalg.solve(g+1e-14*np.eye(3), rhs[..., None])[..., 0]
        done = (grad < 1e-9) | (np.max(abs(step), axis=1) < 1e-9)
        converged[active[done]] = True
        iterations[active] = it+1
        keep = ~done
        if not keep.any():
            continue
        active, xx, step, residual = active[keep], xx[keep], step[keep], residual[keep]
        factor = np.maximum(1, np.max(abs(step)/np.array([.5, 2, 2]), axis=1))
        step /= factor[:, None]
        old = np.sum(residual*residual, axis=1)
        alpha = np.ones(len(xx)); accepted = np.zeros(len(xx), bool)
        for _ in range(14):
            trial = xx+alpha[:, None]*step
            cost = np.sum((obs[active]-fun(trial))**2, axis=1)
            good = (~accepted) & np.isfinite(cost) & (cost <= old)
            x[active[good]] = trial[good]
            accepted |= good
            if accepted.all():
                break
            alpha[~accepted] *= .5
    j = numerical_jac(fun, x)
    er = obs-fun(x)
    grad = np.max(abs(np.einsum('nri,nr->ni', j, er))/np.maximum(np.linalg.norm(j, axis=1), 1e-30), axis=1)
    converged |= grad < 1e-8
    return x, converged, iterations, grad


def outside(x):
    return (x[:, 0] < -1e-8) | (x[:, 0] > 4+1e-8) | (np.max(abs(x[:, 1:]), axis=1) > 20+1e-8)


def fmt(m):
    return ' / '.join(f'{v:.6g}' for v in m)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log('Loading noise-free absolute rays and building truth Jacobians')
    axes, c, cube = load(); y = cube.reshape(-1, 18)
    J, P, cond = engine(cube)
    _, P2, cond2 = engine(cube, 2)
    pose = np.isin(np.round(c[:, 1], 8), POSES) & np.isin(np.round(c[:, 2], 8), POSES)
    cal25 = np.flatnonzero(pose & np.isclose(c[:, 0], 0))
    cal75 = np.flatnonzero(pose & np.isin(np.round(c[:, 0], 8), [0, 2, 4]))
    assert (len(cal25), len(cal75)) == (25, 75)
    core = np.max(abs(c[:, 1:]), axis=1) <= 12.4+1e-8
    assert core.sum() == 83349
    saved = dict(coords=c, ray_labels=RAYS, calibration25=cal25, calibration75=cal75,
                 operating_mask=core, truth_jacobian_condition=cond,
                 truth_jacobian_2h_condition=cond2, fixed_barrel_K=K, fixed_scale_S1=S1,
                 parameter_names=np.array(NAMES))
    interpolation = (c[:, 0] >= .2-1e-9) & (c[:, 0] <= 3.8+1e-9) & ~np.isclose(c[:, 0], 2.)
    records = []; winner = None; start = None
    for rung in range(4):
        cal = cal25 if rung < 2 else cal75
        held = np.ones(len(c), bool); held[cal] = False
        op, ex = np.flatnonzero(held & core), np.flatnonzero(held & ~core)
        p, diag = fit(c, y, cal, rung, start)
        pred = forward(c, p, rung); d = project(pred, y, P)
        cv, cvm = loo(c, y, P, cal, rung, p)
        m, em = metrics(d, op), metrics(d, ex)
        ai = op[interpolation[op]]; ae = ex[interpolation[ex]]
        im, iem = metrics(d, ai), metrics(d, ae)
        passed = passes(m) and passes(cvm)
        rec = dict(rung=rung, p=p, diag=diag, m=m, em=em, cv=cvm,
                   im=im, iem=iem, interp_n=(len(ai), len(ae)),
                   op_n=len(op), ex_n=len(ex), cal=cal, pass_gate=passed)
        records.append(rec); start = p
        for key, val in dict(coefficients=p, projection_bias=d, loo_bias=cv,
                             heldout_metrics=m, extrap_metrics=em, loo_metrics=cvm,
                             interpolation_metrics=im, interpolation_extrap_metrics=iem,
                             interpolation_ids=ai, interpolation_extrap_ids=ae,
                             position_coord_rmse_mm=np.sqrt(np.mean((pred-y)**2, axis=1))).items():
            saved[f'R{rung}_{key}'] = val
        log(f'R{rung}: {NPAR[rung]} fitted + 6 fixed; heldout A {fmt(m[:3])} D; gaze {fmt(m[3:])} arcmin; LOO max {cvm[2]:.5g} D/{cvm[5]:.5g} arcmin; {"PASS" if passed else "FAIL"}')
        if passed:
            winner = rung
            break
    final = records[-1]; rung = final['rung']; p = final['p'].copy(); cal = final['cal']
    held = np.ones(len(c), bool); held[cal] = False
    op, ex = np.flatnonzero(held & core), np.flatnonzero(held & ~core)
    # Sequential deletion with refitting of all surviving coefficients. Decisions
    # compare the change in the requested operating-region maxima.
    removed = []; prune_rows = []; base_d = project(forward(c, p, rung), y, P)
    for term in range(NPAR[rung]):
        trial_removed = removed+[term]
        try:
            pp, dd = fit(c, y, cal, rung, p, trial_removed)
            trial = project(forward(c, pp, rung), y, P)
            delta = trial[op]-base_d[op]
            da = np.max(abs(delta[:, 0])); dg = np.max(np.linalg.norm(delta[:, 1:], axis=1))*60
            bm, tm = metrics(base_d, op), metrics(trial, op)
            drop = abs(tm[2]-bm[2]) < .01 and abs(tm[5]-bm[5]) < .1
            prune_rows.append((term, da, dg, tm[2], tm[5], drop, ''))
            if drop:
                p, base_d = pp, trial
                removed = trial_removed
        except RuntimeError as exc:
            prune_rows.append((term, np.inf, np.inf, np.nan, np.nan, False, str(exc)))
        log(f'Pruning {NAMES[term]}: {"DROP" if prune_rows[-1][5] else "KEEP"}')
    pcv, pcvm = loo(c, y, P, cal, rung, p, removed)
    pm, pem = metrics(base_d, op), metrics(base_d, ex)
    saved.update(pruned_coefficients=p, pruned_removed=np.array(removed, int),
                 pruned_projection_bias=base_d, pruned_loo_bias=pcv,
                 pruning_numeric=np.array([r[:6] for r in prune_rows], float),
                 pruned_metrics=pm, pruned_extrap_metrics=pem, pruned_loo_metrics=pcvm)
    # Exactly one random-state engine comparison, with both inverse directions
    # made explicit. No measurement or label perturbations are generated.
    rng = np.random.default_rng(20261108)
    sample = np.r_[rng.choice(op, 1000, replace=False), rng.choice(ex, 1000, replace=False)]
    trace = RegularGridInterpolator(tuple(axes), cube.reshape(21, 101, 101, 18),
                                   bounds_error=False, fill_value=None)
    model = lambda x: forward(x, p, rung)
    sample_pred = model(c[sample]); linear = project(sample_pred, y[sample], P[sample])
    log('Checking projection and actual candidate inversion on 2000 held-out states')
    tx, tok, tit, tg = invert(trace, sample_pred, c[sample])
    mx, mok, mit, mg = invert(model, y[sample], c[sample])
    true_error = tx-c[sample]-linear
    model_error = mx-c[sample]+linear
    saved.update(check_state_ids=sample, check_projection=linear,
                 check_truth_inverse=tx-c[sample], check_model_inverse=mx-c[sample],
                 check_truth_converged=tok, check_model_converged=mok,
                 check_truth_iterations=tit, check_model_iterations=mit,
                 check_truth_gradient=tg, check_model_gradient=mg,
                 check_truth_linear_difference=true_error, check_model_linear_difference=model_error)
    check = dict(truth=metrics(true_error, np.arange(2000)),
                 model=metrics(model_error, np.arange(2000)),
                 truth_conv=int(tok.sum()), model_conv=int(mok.sum()),
                 truth_out=int(outside(tx).sum()), model_out=int(outside(mx).sum()),
                 sample_core_model=metrics(mx-c[sample], np.arange(1000)),
                 sample_core_truth_agreement=metrics(true_error, np.arange(1000)),
                 sample_core_model_agreement=metrics(model_error, np.arange(1000)))
    full_nl = None
    if winner is not None:
        log('Confirming proposed winner by nonlinear inversion on every held-out state')
        ids = np.flatnonzero(held); nhat = np.empty_like(c[ids]); nok = np.empty(len(ids), bool)
        for lo in range(0, len(ids), 4096):
            batch = ids[lo:lo+4096]
            nhat[lo:lo+len(batch)], nok[lo:lo+len(batch)], _, _ = invert(model, y[batch], c[batch])
        nd = np.full_like(c, np.nan); nd[ids] = nhat-c[ids]
        full_nl = dict(op=metrics(nd, op), ex=metrics(nd, ex), converged=int(nok.sum()),
                       total=len(ids), outside=int(outside(nhat).sum()))
        saved.update(winner_nonlinear_ids=ids, winner_nonlinear_bias=nd[ids], winner_nonlinear_converged=nok)
        if not (passes(full_nl['op']) and nok.all() and passes(pm) and passes(pcvm)):
            winner = None
    # Benchmark is a center-relative, per-state fit: compare shape errors only,
    # not absolute centers or coefficients in mismatched gauges.
    bench = None
    if BENCH.exists():
        pred = model(c).reshape(-1, 9, 2); obs = y.reshape(-1, 9, 2)
        shape_error = ((pred-pred[:, 4:5])-(obs-obs[:, 4:5]))*1000
        outer = [0, 1, 2, 3, 5, 6, 7, 8]
        with np.load(BENCH, allow_pickle=False) as z:
            assert np.allclose(z['fields'], RAYS)
            assert np.allclose(z['accommodation_d'], axes[0])
            assert np.allclose(z['ADE'], axes[1]) and np.allclose(z['BDE'], axes[2])
            be = z['errors_um'].reshape(-1, 9, 2)
            status, counts = np.unique(z['status'], return_counts=True)
            bm = [float(np.sqrt(np.mean(be[op][:, outer]**2))),
                  float(np.max(np.linalg.norm(be[op][:, outer], axis=-1)))]
            cm = [float(np.sqrt(np.mean(shape_error[op][:, outer]**2))),
                  float(np.max(np.linalg.norm(shape_error[op][:, outer], axis=-1)))]
            bench = dict(benchmark_coord_RMS_maxpoint_um=bm, candidate_coord_RMS_maxpoint_um=cm,
                         statuses=dict(zip(status.tolist(), counts.tolist())))
            saved['benchmark_shape_check_um'] = np.array([bm, cm])
    step_change = project(model(c), y, P2)-base_d
    saved['step_2h_bias_difference'] = step_change
    saved['winning_rung'] = np.array(-1 if winner is None else winner)
    np.savez_compressed(OUT/'p4_reduced_ladder.npz', **saved)
    report(records, final, winner, removed, prune_rows, p, pm, pem, pcvm,
           check, full_nl, bench, cond, cond2, metrics(step_change, op))
    log(f'Complete: verified winner={winner}; saved report and NPZ')


def report(records, final, winner, removed, pruning, p, pm, pem, pcvm, check,
           full_nl, bench, cond, cond2, step_metric):
    rung = final['rung']
    L = ['# Noise-free P4 reduced physical ladder', '',
         'Noise-free model-bias evaluation. No CODE V run, hybrid rerun, or later workflow step is executed.', '',
         'Noise assumption: 1.0 um per image coordinate, with calibration and runtime averaged over 3,000-5,000 frames at 1 kHz. The expected noise contribution is negligible (about 0.3 mD and 3e-5 deg); no noise Monte Carlo is run. The whole budget here is model bias.', '',
         '## Data and protocol', '',
         f'Input: `{DATA.relative_to(ROOT).as_posix()}`; SHA-256 `{sha(DATA)}`. All 214,221 states and 1,927,989 state-field records are finite. Absolute position is `center_absolute + real`, in mm. Parameters are `(A D, ADE deg, BDE deg)`.', '',
         'The recorded sensor frame and field labels are fixed. Fields are sorted by `(fx,fy)` in {-1,0,1} x {-1,0,1}; field 4 is the labeled central ray. At zero gaze the image reverses both field signs. The fixed image origin is (0,0) mm; no fitted registration, free center, or coordinate rotation is introduced. Central-ray absolute position includes its stored `real` contribution.', '',
         'R0/R1 use only the 25 A=0 poses with each gaze axis in {-12.4,-6.4,0,6.4,12.4} deg. R2/R3 use the 75 poses at A=0,2,4 D on that gaze lattice, refitting their full active parameter sets. Every other state is scored. LOPO leaves out all nine rays at one labeled (A,ADE,BDE) pose, refitting the reference radii as well. Optimizer warm starts do not enter the calibration objective. The tan gaze law is declared before scoring; no law is selected using held-out data.', '',
         'Operating region: both absolute gaze components <=12.4 deg, all 21 A levels (83,349 states before calibration exclusion). Extrapolation is the rest of the square through +/-20 deg, including mixed inner/outer-axis states. The quoted hybrid has a different stored state count and is not recomputed. Errors use |A bias| and radial sqrt(ADE_bias^2+BDE_bias^2), with gaze reported in arcmin. Pass requires maximum <=0.1 D and <=1 arcmin in both operating holdout and LOPO.', '',
         '## Explicit nested physical hypotheses', '',
         'The requested on-axis report has no subsection labeled "1g". This implementation uses its recommended, physical-gauge `s poly1 + k poly4 [k0=c0]` formula, also present in the existing combined-Brown helper; it does not use the older power-law barrel variant.', '',
         '`a=A/4; s=1-0.081595476571*a; k=sum(K[j]*a**j); K=[-0.0185004967,0.00023989235714,0.0015756063948,-0.0014256378340,0.00066200789981] mm^-2; D_k(u)=(1+k*|u|^2)*u; B_A(r0)=D_k(A)(s*D_k(0)^-1(r0))`.', '',
         'These six prior constants are fixed, not refitted on the staged calibration poses. They originated in the prior 21-accommodation on-axis study; thus this experiment tests transfer of a supplied accommodation prior, not learning A dependence from A=0 alone. The near-axis radial inverse uses Newton iterations. Two fitted reference values describe edge radius e and each corner coordinate c: `r0=-e*(fx,fy)` at edges and `-c*(fx,fy)` at corners.', '',
         'R0: `T=(Rx*tan(BDE),Ry*tan(ADE)); prediction=T+B_A(r0)`. Rx is signed in the recorded frame. Four fitted scalars, including e,c.', '',
         'R1: let `b=B_A(r0)`, angles be radians inside trigonometric functions. `sx=1+cxx*(sec(BDE)^2-1)+cxy*(sec(ADE)-1)`; `sy=1+cyx*(sec(BDE)-1)+cyy*(sec(ADE)^2-1)`; `den=1+qx*tan(BDE)*b_x+qy*tan(ADE)*b_y`; `w=(sx*b_x,sy*b_y)/den`; prediction=T+w. Six additional scalars. For pure ADE at 20 deg, sec-1=0.0641778, sec^2-1=0.1324743; the quadratic Taylor coefficients for normalized theta/20 are 0.0609235 and 0.1218470, comparable to the supplied stored 0.0606/0.1250 hints. These are hints, not fixed calibrated scale values.', '',
         'R2: one additive linear `A/4` coefficient for each of eight existing groups `(Rx,Ry,cxx,cxy,cyx,cyy,qx,qy)`, adding eight scalars. The fixed barrel law and two reference radii remain unchanged in form.', '',
         'R3: add two effective pivot radii hx,hy. Relative distortion-pivot offset `d=(-hx*sin(BDE),-hy*sin(ADE))` predicts `p1=-k(A)*d_y`, `p2=-k(A)*d_x`. Apply `wx -> wx+2*p1*wx*wy+p2*(3*wx^2+wy^2)`, `wy -> wy+p1*(wx^2+3*wy^2)+2*p2*wx*wy` after keystone and before translation. This is a first-order shifted-barrel hypothesis; higher-order offset terms are outside this rung. No independent per-state p1/p2 or arbitrary gaze polynomial is fitted. The offset law is a calibrated hypothesis, not independent evidence of an anatomical pivot.', '',
         '## Ladder', '',
         'Each cell is RMS / P95 / max. A is D; gaze is radial arcmin. Counts include all finite held-out states. The six fixed barrel constants are in addition to fitted counts.', '',
         '| Rung | Fitted (+ fixed) | Holdout N op / extrap | Operating A | Operating gaze | Extrap A | Extrap gaze | LOPO A | LOPO gaze | Both gates |',
         '|---|---:|---:|---|---|---|---|---|---|---|']
    for r in records:
        L.append(f"| R{r['rung']} | {NPAR[r['rung']]} (+6) | {r['op_n']} / {r['ex_n']} | {fmt(r['m'][:3])} | {fmt(r['m'][3:])} | {fmt(r['em'][:3])} | {fmt(r['em'][3:])} | {fmt(r['cv'][:3])} | {fmt(r['cv'][3:])} | {'PASS' if r['pass_gate'] else 'FAIL'} |")
    L += ['', '## Accommodation interpolation', '',
          'A=0.2 to 3.8 D, excluding 2 D; these 18 levels are absent from calibration. RMS / P95 / max; A in D and radial gaze in arcmin. Extrapolation remains outside the pass criterion.', '',
          '| Rung | Region | N | A RMS / P95 / max D | Gaze RMS / P95 / max arcmin |',
          '|---|---|---:|---|---|']
    for r in records:
        for label, metric, n in [('Operating', r['im'], r['interp_n'][0]), ('Extrapolation', r['iem'], r['interp_n'][1])]:
            L.append(f"| R{r['rung']} | {label} | {n} | {fmt(metric[:3])} | {fmt(metric[3:])} |")
    L += ['', f"Verified winning rung: **{'none' if winner is None else 'R'+str(winner)}**. Final assessed rung R{rung}: {NPAR[rung]} fitted before pruning, {NPAR[rung]-len(removed)} fitted after pruning, plus six fixed constants ({NPAR[rung]-len(removed)+6} retained numerical scalars total). No anatomical identification is inferred from a numerical fit.", '',
          'Stored hybrid reference (quoted, not rerun): **87 values**, noise-free, **83,229 states**, +/-12.4 deg: **A RMS/P95/max 0.026/0.051/0.180 D (99.35% within 0.1 D); P4 gaze RMS/P95/max 0.90/1.73/2.06 arcmin; P1 gaze RMS 0.18 arcmin**.', '',
          '## Calibration diagnostics', '', '| Rung | Rank / fitted | Raw Jacobian condition | Column-normalized condition | Calibration SSE mm^2 | Optimality |', '|---|---:|---:|---:|---:|---:|']
    for r in records:
        d = r['diag']
        L.append(f"| R{r['rung']} | {d['rank']} / {NPAR[r['rung']]} | {d['condition']:.6g} | {d['scaled_condition']:.6g} | {2*d['cost']:.6g} | {d['optimality']:.6g} |")
    L += ['', '## Pruning', '',
          f'Sequential single-term removal and calibration-only refitting on R{rung}, including tests of the two reference radii. A term is dropped only if refitting it out changes the operating holdout maximum absolute A bias by <0.01 D AND the operating holdout maximum radial gaze bias by <0.1 arcmin. The table also reports the maximum pointwise change as a diagnostic; it does not determine pruning. Fixed barrel prior constants are protected by the requested fixed-law protocol, and are counted separately. Decisions are held-out relevance selection; LOPO below evaluates the resulting fixed retained structure, not an outer nested selection CV.', '',
          '| Term | Max change A D | Max change gaze arcmin | New max A D | New max gaze arcmin | Decision |', '|---|---:|---:|---:|---:|---|']
    for term, da, dg, ma, mg, drop, error in pruning:
        L.append(f"| {NAMES[term]} | {da:.6g} | {dg:.6g} | {ma:.6g} | {mg:.6g} | {'DROP' if drop else 'KEEP'}{' (refit rank/convergence failure)' if error else ''} |")
    L += ['', '| Pruned result | A RMS/P95/max D | Gaze RMS/P95/max arcmin |', '|---|---|---|',
          f'| Operating holdout | {fmt(pm[:3])} | {fmt(pm[3:])} |',
          f'| Extrapolation holdout | {fmt(pem[:3])} | {fmt(pem[3:])} |',
          f'| LOPO | {fmt(pcvm[:3])} | {fmt(pcvm[3:])} |', '',
          '## Bias engine and nonlinear checks', '',
          'The 18x3 truth Jacobian uses the imported difference helper only, with central two-point/one-sided stencils of 0.2 D and 0.4 deg. Units remain per D and per degree. All truth Jacobians have rank 3. `r=candidate-truth; d=(J^T J)^-1 J^T r` is evaluated with a linear solve. The candidate inverse has first-order bias `-d`. Absolute/radial scores do not depend on that sign convention.', '',
          f'Truth J^T J condition median/P95/max (D,deg,deg units): {fmt(np.percentile(cond,[50,95,100]))}; at 2h: {fmt(np.percentile(cond2,[50,95,100]))}. Final candidate 2h-minus-h bias agreement A: {fmt(step_metric[:3])} D; gaze: {fmt(step_metric[3:])} arcmin.', '',
          'One seeded check samples 1,000 random operating and 1,000 random extrapolation held-out states. The true-grid inverse uses trilinear interpolation of the noise-free rays, with linear extrapolation when necessary; this is a diagnostic surrogate, not fresh ray tracing. Separately, the actual continuous candidate model is inverted against the true image positions. Solvers start at exact labels to test local bias and stationarity, not initialization/global uniqueness. No A/gaze boundary clipping is used. Convergence is a residual-gradient test; nonzero image SSE can remain at a least-squares optimum.', '',
          f"Truth inverse least-squares `F_truth(theta_hat)~=F_candidate(theta)` versus +d: {check['truth_conv']}/2000 converged; {check['truth_out']} solutions outside the stored truth domain. The 18 observations need not lie on the three-parameter interpolated truth manifold, so residuals can remain nonzero at the least-squares solution. A difference RMS/P95/max {fmt(check['truth'][:3])} D; radial gaze difference {fmt(check['truth'][3:])} arcmin.", '',
          f"Actual candidate inverse `F_candidate(theta_hat)~=F_truth(theta)` versus -d: {check['model_conv']}/2000 converged; {check['model_out']} solutions outside the stored truth domain. A difference RMS/P95/max {fmt(check['model'][:3])} D; radial gaze difference {fmt(check['model'][3:])} arcmin. The true-grid projection is only a local linear diagnostic and does not certify nonlinear bias: actual candidate-inverse operating agreement reaches {check['sample_core_model_agreement'][2]:.6g} D maximum and {check['sample_core_model_agreement'][5]:.6g} arcmin maximum. Nonconverged inverses remain counted and reported; they are not treated as successful.", '',
          f"Operating-only 1,000-state agreement: truth inverse A {fmt(check['sample_core_truth_agreement'][:3])} D, gaze {fmt(check['sample_core_truth_agreement'][3:])} arcmin; actual candidate inverse A {fmt(check['sample_core_model_agreement'][:3])} D, gaze {fmt(check['sample_core_model_agreement'][3:])} arcmin. Actual candidate inverse operating bias A {fmt(check['sample_core_model'][:3])} D and gaze {fmt(check['sample_core_model'][3:])} arcmin.", '']
    if full_nl is None:
        L += ['No rung passed both linear operating and LOPO gates; there is no winning rung to confirm with full held-out nonlinear inversion. The 2,000-state inverse test is diagnostic only.', '']
    else:
        L += [f"Full held-out nonlinear confirmation: {full_nl['converged']}/{full_nl['total']} converged, {full_nl['outside']} estimates outside the truth domain. Operating A {fmt(full_nl['op'][:3])} D, gaze {fmt(full_nl['op'][3:])} arcmin; extrap A {fmt(full_nl['ex'][:3])} D, gaze {fmt(full_nl['ex'][3:])} arcmin.", '']
    L += ['## Independent benchmark check', '',
          f'Post-fit only: `{BENCH.relative_to(ROOT).as_posix()}`. Ray labels and axes are verified. The benchmark is a per-state center-relative outer-ray fit with zero central residual by construction; it is not the reduced model. We compare outer-eight-ray center-relative shape errors in the operating holdout, coordinate RMS / max point norm in um: `{bench}`. No benchmark positions, parameters, or residuals enter fitting or pruning.', '',
          '## Retained constants and lens counterparts', '',
          f"`Lens/p4_ME.len` is binary; SHA-256 `{sha(ROOT/'Lens/p4_ME.len')}`. It was neither decoded nor opened in CODE V. Its textual companion `Lens/p4_ME.seq` identifies RC and a -13.8 mm thickness and nine field angles with components 0,+/-20 deg. Companion-to-binary identity is unverified. These do not identify an effective sensor-plane rotation radius; no equality to the fitted radii is asserted.", '',
          'The six fixed barrel constants above are image-model calibration constants from `data/p4_onaxis/report.md`; no direct lens-file surface-parameter counterparts are established. R0-R3 coefficients are effective image-space geometry; the following table gives all retained fitted constants. There is no verified one-to-one lens-file counterpart for these coefficients.', '',
          '| Constant | Value | Lens-file counterpart |', '|---|---:|---|']
    for j in range(NPAR[rung]):
        if j not in removed:
            L.append(f'| {NAMES[j]} | {p[j]:.12g} | Not established |')
    L += ['', '## Problems and scope', '',
          'The ladder tests explicit restricted hypotheses; failing these rungs does not prove that every reduced physical model fails. The Brown pivot law is not an independently measured pivot, and a sensor origin must not be equated with a distortion pivot. The absent "1g" label and binary-only lens counterparts limit provenance. Linear projections can differ from nonlinear inversions, especially when model residuals or extrapolation are large; the agreement numbers above must accompany the ladder. No success is claimed when either target fails.', '',
          'Pruning preserves the requested maxima-based projection criterion, but pointwise projection changes and nonlinear bias can change substantially; it does not establish nonlinear irrelevance of the removed term.', '',
          'Artifacts: `Script/p4_reduced_ladder.py`, this report, and `p4_reduced_ladder.npz`. NPZ stores labels, calibration indices, per-rung and pruned signed projection biases, whole-pose CV biases, fitted coefficients, pruning tests, finite-difference sensitivity, and all 2,000 nonlinear check states/results. No existing input or program is intentionally modified.', '']
    (OUT/'report.md').write_text('\n'.join(L), encoding='utf-8')


class BoundedTruthGrid:
    """Exact trilinear value/analytic cell derivative, in D and degrees.

    This inverse is for noise-free TRUE-grid images, not off-manifold images.
    Convergence requires image recovery, not just gradient stationarity.
    """
    def __init__(self, axes, cube):
        self.axes = axes
        self.low = np.array([a[0] for a in axes])
        self.high = np.array([a[-1] for a in axes])
        self.h = np.array([a[1]-a[0] for a in axes])
        self.maxcell = np.array([len(a)-2 for a in axes])
        self.values = cube.reshape(21, 101, 101, 18)

    def evaluate(self, x, jacobian=False):
        u = (np.asarray(x)-self.low)/self.h
        cell = np.minimum(np.maximum(np.floor(u).astype(int), 0), self.maxcell)
        t = u-cell
        out = np.zeros((len(x), 18))
        jac = np.zeros((len(x), 18, 3)) if jacobian else None
        for b0 in (0, 1):
            for b1 in (0, 1):
                for b2 in (0, 1):
                    bit = np.array([b0, b1, b2])
                    w = np.where(bit, t, 1-t)
                    val = self.values[cell[:, 0]+b0, cell[:, 1]+b1, cell[:, 2]+b2]
                    out += np.prod(w, axis=1)[:, None]*val
                    if jacobian:
                        for j in range(3):
                            others = [k for k in range(3) if k != j]
                            dw = (2*bit[j]-1)/self.h[j]*np.prod(w[:, others], axis=1)
                            jac[:, :, j] += dw[:, None]*val
        return (out, jac) if jacobian else out

    def inverse(self, obs, start):
        x = np.clip(np.asarray(start).copy(), self.low, self.high)
        ok = np.zeros(len(x), bool)
        nit = np.zeros(len(x), int)
        for iteration in range(60):
            active = np.flatnonzero(~ok)
            if not len(active):
                break
            xx = x[active]
            pred, jac = self.evaluate(xx, True)
            er = obs[active]-pred
            done = np.max(abs(er), axis=1) <= 1e-10
            ok[active[done]] = True
            nit[active] = iteration+1
            active, xx, er, jac = active[~done], xx[~done], er[~done], jac[~done]
            if not len(active):
                continue
            g = np.einsum('nri,nrj->nij', jac, jac)
            rhs = np.einsum('nri,nr->ni', jac, er)
            step = np.linalg.solve(g+1e-15*np.eye(3), rhs[..., None])[..., 0]
            step /= np.maximum(1, np.max(abs(step)/[.5, 3, 3], axis=1))[:, None]
            oldcost = np.sum(er*er, axis=1)
            alpha = np.ones(len(xx)); accepted = np.zeros(len(xx), bool)
            for _ in range(16):
                trial = np.clip(xx+alpha[:, None]*step, self.low, self.high)
                cost = np.sum((obs[active]-self.evaluate(trial))**2, axis=1)
                good = (~accepted) & (cost <= oldcost)
                x[active[good]] = trial[good]
                accepted |= good
                if accepted.all():
                    break
                alpha[~accepted] *= .5
        residual = np.max(abs(obs-self.evaluate(x)), axis=1)
        ok |= residual <= 1e-10
        return x, ok, nit, residual


def decomposition_metrics(center_error, shape_error, ids):
    if not len(ids):
        return np.full(4, np.nan)
    cn = np.linalg.norm(center_error[ids], axis=-1)*1000
    sn = np.linalg.norm(shape_error[ids], axis=-1)*1000
    return np.array([np.sqrt(np.mean(cn*cn)), np.max(cn),
                     np.sqrt(np.mean(sn*sn)), np.max(sn)])


def frozen_hybrid_prediction(c, ids):
    """A forward realization of frozen G4 + M1 + F4, with no refitting.

    The blind runtime is not rerun. G4 inversion chooses its central local
    branch and must round-trip exactly at all gaze nodes before use.
    """
    from evaluate_p4_unified_combined import load_models, FIELD_SUMMARY, MOTION_SUMMARY
    frozen, field, motion, hashes = load_models(FIELD_SUMMARY, MOTION_SUMMARY)
    gaze, inverse = np.unique(c[ids, 1:], axis=0, return_inverse=True)
    q = np.broadcast_to(frozen.p4_gaze.mean_mm, gaze.shape).copy()
    for _ in range(40):
        er = frozen.p4_gaze.predict(q)-gaze
        if np.max(abs(er)) < 1e-11:
            break
        step = np.linalg.solve(frozen.p4_gaze.jacobian(q), er[..., None])[..., 0]
        q -= step
    roundtrip = float(np.max(abs(frozen.p4_gaze.predict(q)-gaze)))
    assert roundtrip < 1e-9, 'Frozen G4 inverse branch/roundtrip failure'
    pc = motion.forward(q[inverse], c[ids, 0])
    shape = field.evaluate(c[ids, 0], c[ids, 1:])
    assert np.max(abs(shape[:, 4])) < 1e-12
    return pc, shape, roundtrip, hashes, motion.coefficients


def continuation():
    """Only decomposition, self-inversion audit and the conditional gate.

    This entry point never calls fit(), loo(), main(), or an optical program.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    log('Continuation: frozen coefficients only; no rung refitting')
    axes, c, cube = load(); y = cube.reshape(-1, 18)
    with DATA.open('rb') as f:
        raw = pickle.load(f)
    names = ('accommodation_d', 'eye_rotation_deg', 'eye_rotation_x_deg',
             'field_x_relative', 'field_y_relative')
    full_axes = axes+[np.array([-1., 0., 1.]), np.array([-1., 0., 1.])]
    ai, di, bi, xi, yi = [np.searchsorted(ax, raw[n]) for ax, n in zip(full_axes, names)]
    key = (((ai*101+di)*101+bi)*9+xi*3+yi)
    detector_centers = np.empty((len(raw), 2))
    detector_centers[key] = np.c_[raw['center_absolute_x_mm'], raw['center_absolute_y_mm']]
    dc = detector_centers.reshape(-1, 9, 2)
    assert np.max(abs(dc-dc[:, 4:5])) == 0
    measured_center = dc[:, 4].copy()
    measured_shape = cube.reshape(-1, 9, 2)-measured_center[:, None, :]
    del raw, detector_centers, dc
    source = OUT/'p4_reduced_ladder.npz'
    with np.load(source, allow_pickle=False) as z:
        frozen = {k: z[k].copy() for k in z.files}
    assert np.allclose(frozen['coords'], c)
    sample = frozen['check_state_ids']
    trace = RegularGridInterpolator(tuple(axes), cube.reshape(21, 101, 101, 18),
                                   bounds_error=False, fill_value=None)
    truth = BoundedTruthGrid(axes, cube)
    lower, upper = truth.low, truth.high
    before = []
    saved = dict(coords=c, measured_center_absolute_mm=measured_center,
                 measured_shape_field4_mm=measured_shape[:, 4],
                 original_fit_npz_sha256=np.array(sha(source)), inverse_sample_ids=sample)
    log('True-grid self-inversion before: exact, offset and common numerical starts')
    starts = [c[sample], np.clip(c[sample]+[.35, .8, -.6], lower, upper),
              np.broadcast_to([2., 0., 0.], (len(sample), 3))]
    for name, start in zip(('exact', 'offset', 'common'), starts):
        result, conv, nit, grad = invert(trace, y[sample], start)
        err = result-c[sample]
        recovered = (abs(err[:, 0]) <= 1e-6) & (np.linalg.norm(err[:, 1:], axis=1) <= 1e-6)
        success = conv & recovered
        maxpos = float(np.max(abs(trace(result)-y[sample])))
        before.append((name, int((~success).sum()), int((~conv).sum()),
                       float(np.max(abs(err[:, 0]))),
                       float(np.max(np.linalg.norm(err[:, 1:], axis=1))*60), maxpos))
        saved[f'before_{name}_delta'] = err
        saved[f'before_{name}_success'] = success
    log('Hardened bounded analytic-Jacobian inverse: all 214221 true-grid images')
    inverse = np.empty_like(c); conv = np.empty(len(c), bool)
    residual = np.empty(len(c)); nit = np.empty(len(c), int)
    for lo in range(0, len(c), 4096):
        hi = min(lo+4096, len(c))
        inverse[lo:hi], conv[lo:hi], nit[lo:hi], residual[lo:hi] = truth.inverse(
            y[lo:hi], np.broadcast_to([2., 0., 0.], (hi-lo, 3)))
    err = inverse-c
    recovered = (abs(err[:, 0]) <= 1e-6) & (np.linalg.norm(err[:, 1:], axis=1) <= 1e-6)
    success = conv & recovered
    gate_pass = bool(success.all())
    historical_failure_ids = sample[~frozen['check_truth_converged']]
    saved.update(after_truth_delta=err, after_truth_success=success,
                 after_truth_converged=conv, after_truth_residual_maxcoord_mm=residual,
                 after_truth_iterations=nit, historical_off_manifold_failure_ids=historical_failure_ids)
    log(f'Self-inversion gate: {int((~success).sum())}/{len(c)} failures')
    # Do not produce any new model-inverse numbers if this gate fails.
    assert gate_pass, 'TRUE-grid self-inversion did not pass; nonlinear reporting blocked'
    gaze_abs = np.round(np.max(abs(c[:, 1:]), axis=1), 8)
    operating = gaze_abs <= 12.4
    bin_id = np.searchsorted([4, 8, 12, 16], gaze_abs, side='right')
    models = []
    all_summary = []
    for rung in range(4):
        pred = forward(c, frozen[f'R{rung}_coefficients'], rung).reshape(-1, 9, 2)
        # T is the explicit center law. All current relative models fix field4=0.
        predicted_center = pred[:, 4]
        center_error = predicted_center-measured_center
        shape_error = (pred-predicted_center[:, None, :])-measured_shape
        assert np.allclose(pred-cube.reshape(-1, 9, 2), center_error[:, None, :]+shape_error,
                           rtol=0, atol=2e-15)
        mask = operating.copy()
        mask[frozen['calibration25' if rung < 2 else 'calibration75']] = False
        models.append((f'R{rung}', mask, center_error, shape_error))
    runtime = ROOT/'data/p4_unified_combined/runtime/full_fast41/runtime.npz'
    with np.load(runtime, allow_pickle=False) as z:
        hc = z['truth_state_scoring_only'][z['primary_mask']]
    hidx = np.rint((hc-np.array([0., -20., -20.]))/[.2, .4, .4]).astype(int)
    hybrid_ids = (hidx[:, 0]*101+hidx[:, 1])*101+hidx[:, 2]
    assert len(hybrid_ids) == 83229 and len(np.unique(hybrid_ids)) == 83229
    assert np.max(abs(c[hybrid_ids]-hc)) < 1e-8 and operating[hybrid_ids].all()
    log('Decomposing frozen hybrid forward realization; no blind runtime rerun')
    pc, ps, g4_roundtrip, hybrid_hashes, m1 = frozen_hybrid_prediction(c, hybrid_ids)
    hcerror = np.full((len(c), 2), np.nan); hserror = np.full((len(c), 9, 2), np.nan)
    hcerror[hybrid_ids] = pc-measured_center[hybrid_ids]
    hserror[hybrid_ids] = ps-measured_shape[hybrid_ids]
    hmask = np.zeros(len(c), bool); hmask[hybrid_ids] = True
    models.append(('Hybrid87', hmask, hcerror, hserror))
    for name, mask, ce, se in models:
        summary = decomposition_metrics(ce, se, np.flatnonzero(mask))
        by_a = np.stack([decomposition_metrics(ce, se, np.flatnonzero(mask & np.isclose(c[:, 0], a))) for a in axes[0]])
        by_bin = np.stack([decomposition_metrics(ce, se, np.flatnonzero(mask & (bin_id == j))) for j in range(5)])
        na = np.array([np.sum(mask & np.isclose(c[:, 0], a)) for a in axes[0]])
        nb = np.array([np.sum(mask & (bin_id == j)) for j in range(5)])
        saved.update({f'{name}_mask': mask, f'{name}_center_error_mm': ce,
                      f'{name}_shape_error_mm': se, f'{name}_summary_um': summary,
                      f'{name}_by_A_um': by_a, f'{name}_by_gaze_bin_um': by_bin,
                      f'{name}_by_A_count': na, f'{name}_by_gaze_bin_count': nb})
        all_summary.append((name, int(mask.sum()), summary, by_a, by_bin, na, nb))
        log(f'{name}: centre RMS/max {fmt(summary[:2])} um; shape RMS/max {fmt(summary[2:])} um')
    # Gate is intentionally qualitative in the report: neither the R0 shape
    # nor the R3 shape is a few um, and comparable component sizes reject a
    # center-only explanation. No A-scaling parameter can be learned at A=0.
    r0_summary = all_summary[0][2]
    r3_summary = all_summary[3][2]
    center_only = bool(r0_summary[2] <= 5 and r0_summary[0] >= 3*r0_summary[2]
                       and r3_summary[2] <= 5 and r3_summary[0] >= 3*r3_summary[2])
    saved.update(decomposition_columns=np.array(['center_RMSnorm_um', 'center_maxnorm_um',
                                                'shape_RMSnorm_um', 'shape_maxnorm_um']),
                 hybrid_G4_roundtrip_max_deg=np.array(g4_roundtrip),
                 hybrid_M1_coefficients=m1, r0_prime_gate=np.array(center_only))
    if center_only:
        raise RuntimeError('Center-only gate unexpectedly passed: R0-prime needs an explicit A-scaling prior; do not silently fit it at A=0')
    np.savez_compressed(OUT/'continuation.npz', **saved)
    continuation_report(c, before, err, residual, conv, success, sample,
                        historical_failure_ids, all_summary, g4_roundtrip, hybrid_hashes)
    # Supersede the old report's model-error paragraph rather than presenting
    # the previously unvalidated nonlinear numbers as an acceptance result.
    old_path = OUT/'report.md'
    old_text = old_path.read_text(encoding='utf-8')
    old_lines = old_text.splitlines()
    old_lines = [('The prior operating-sample nonlinear model-error figures are withdrawn from acceptance use. '
                  'See `continuation_report.md` for genuine true-grid self-inversion validation and the centre/shape decomposition.')
                 if line.startswith('Operating-only 1,000-state agreement:') else line for line in old_lines]
    note = '\n\nContinuation audit: the earlier 16 unconverged true-grid fits used candidate-model images, not true-grid self-images. '
    note += 'They were off-manifold fitting diagnostics and must not be described as failed true-grid self-inversions. See `continuation_report.md`.\n'
    if 'Continuation audit:' not in old_text:
        old_path.write_text('\n'.join(old_lines)+note, encoding='utf-8')
    log('Continuation complete: no rung refit; R0-prime not run because its condition failed')


def continuation_report(c, before, err, residual, conv, success, sample,
                        historical_failures, summaries, g4_roundtrip, hybrid_hashes):
    bnames = ['[0,4)', '[4,8)', '[8,12)', '[12,16), truncated at 12.4', '[16,20] (empty)']
    L = ['# Noise-free reduced-ladder continuation', '',
         '## Scope and definitions', '',
         'R0-R3 coefficients are loaded from the original fitted NPZ and evaluated without any refitting. All images and labels are noise-free. No noise case, CODE V run, hybrid blind-solver rerun, or other workflow step is executed.', '',
         'Measured centre is exactly the stored `center_absolute_x/y_mm`, not the absolute labeled central ray. Measured absolute rays are `center_absolute + real`. Predicted centre for R0-R3 is their explicit translation T (also their predicted field 4). Define `e_c=predicted_center-center_absolute` and `e_s=(predicted_rays-predicted_center)-(measured_rays-center_absolute)`. Thus absolute ray residual equals `e_c+e_s` exactly; this identity is checked. The measured `real[4]` offset is retained, not subtracted away.', '',
         'RMS means radial RMS: centre sqrt(mean_states(dx^2+dy^2)); shape sqrt(mean_states,fields(dx^2+dy^2)), using all nine rays. Max means maximum vector norm, over states for centre and states/rays for shape. These are not per-coordinate RMS. All decomposition numbers are um.', '',
         'Operating holdout uses both gaze components within +/-12.4 deg, all 21 A levels. R0/R1 exclude their 25 calibration poses; R2/R3 exclude their 75. The hybrid uses its original 83,229 primary held-out paired states, joined to the macro grid by exact (A,ADE,BDE) labels. Its mask differs and is saved. Bins use max(|ADE|,|BDE|); intervals are left-closed, right-open, except the last includes 20. The requested 16-20 bin is empty on the operating region, and the 12-16 bin ends at 12.4.', '',
         'Hybrid reference convention: construct a frozen forward realization by solving `G4(c0)=true gaze`, then `c_pred=M1(c0,A)` and `rays_pred=c_pred+F4(A,gaze)`. This inverts only the frozen calibration polynomial, not the blind runtime. The existing hybrid has no separately saved forward centre law; its observation-conditioned runtime centre is not used as a prediction. M1/G4 describe the labeled central ray, whereas the requested centre is `center_absolute`; that convention difference remains explicitly in the residual.', '',
         f'Frozen G4 roundtrip maximum: {g4_roundtrip:.6g} deg. Frozen provenance hashes: `{hybrid_hashes}`.', '',
         '## Whole operating holdout decomposition', '',
         '| Model | N | Centre RMS | Centre max | Shape RMS | Shape max |', '|---|---:|---:|---:|---:|---:|']
    for name, n, m, *_ in summaries:
        L.append(f'| {name} | {n} | '+' | '.join(f'{v:.6g}' for v in m)+' |')
    for name, n, m, aa, bb, na, nb in summaries:
        L += ['', f'## {name}: by gaze bin', '',
              '| max absolute gaze deg | N | Centre RMS | Centre max | Shape RMS | Shape max |',
              '|---|---:|---:|---:|---:|---:|']
        for label, count, row in zip(bnames, nb, bb):
            values = ['--' if not np.isfinite(v) else f'{v:.6g}' for v in row]
            L.append(f'| {label} | {count} | '+' | '.join(values)+' |')
        L += ['', f'## {name}: by A (21 levels)', '',
              '| A D | N | Centre RMS | Centre max | Shape RMS | Shape max |',
              '|---:|---:|---:|---:|---:|---:|']
        for a, count, row in zip(np.linspace(0, 4, 21), na, aa):
            L.append(f'| {a:.1f} | {count} | '+' | '.join(f'{v:.6g}' for v in row)+' |')
    L += ['', '## Inversion audit and correction', '',
          'The previous 16 nonconverged true-grid fits did NOT invert true-grid images. Their targets were candidate-model predictions, generally off the true-grid image manifold. The previous description as a true-inversion check conflated inverse directions. They are not a valid before count for a self-inversion fix, and previously quoted nonlinear model-error numbers are not used for acceptance in this report.', '',
          'A genuine before test uses the old solver and true-grid observations at the same 2,000 saved state IDs. Only numerical initial guesses change; observations and labels do not. A success requires solver convergence AND recovered A within 1e-6 D and radial gaze within 1e-6 deg. The offset start is truth+[0.35 D,0.8 deg,-0.6 deg], clipped only to initialize inside the domain. The common start is (2 D,0,0), with no truth initialization.', '',
          '| Old solver start | Failed recovery / 2000 | Solver nonconverged | Max A recovery D | Max radial recovery arcmin | Max coordinate residual mm |',
          '|---|---:|---:|---:|---:|---:|']
    for name, failed, nonconv, amax, gmax, pmax in before:
        L.append(f'| {name} | {failed} | {nonconv} | {amax:.6g} | {gmax:.6g} | {pmax:.6g} |')
    L += ['', 'Hardened solver: analytic trilinear cell Jacobian, physical bounds A=[0,4] D and ADE/BDE=[-20,20] deg, monotone line search, and a per-coordinate image residual <=1e-10 mm required for convergence. Stationary nonzero SSE alone cannot count as self-inversion success. It uses the same exact grid interpolant; no spline smoothing, new ray tracing, or model fitting changes the observations.', '',
          f'After: **{int((~success).sum())} failed recoveries / {len(c)} states**, **{int((~conv).sum())} solver nonconvergences**. Every state starts at (2 D,0,0). Max |A recovery error|={np.max(abs(err[:,0])):.6g} D; max radial gaze recovery error={np.max(np.linalg.norm(err[:,1:],axis=1))*60:.6g} arcmin; max coordinate residual={np.max(residual):.6g} mm. On the original 2,000-ID subset, after failures={int((~success[sample]).sum())}/2000. Genuine before/after common-start count is therefore 0/2000 versus 0/2000; the old solver already passed self-inversion, and the stronger validation now covers the entire grid.', '',
          '| True-grid subgroup | N | After failed recovery |', '|---|---:|---:|']
    masks = [('A=0', np.isclose(c[:, 0], 0)), ('A=4', np.isclose(c[:, 0], 4)),
             ('Any gaze face at +/-20', np.any(np.isclose(abs(c[:, 1:]), 20), axis=1)),
             ('Any A or gaze domain boundary', np.isclose(c[:, 0], 0) | np.isclose(c[:, 0], 4) | np.any(np.isclose(abs(c[:, 1:]), 20), axis=1))]
    ga = np.round(np.max(abs(c[:, 1:]), axis=1), 8)
    for j, (low, high) in enumerate(zip((0, 4, 8, 12, 16), (4, 8, 12, 16, 20.000001))):
        label = f'Max gaze [{low},{min(high,20):g}]' if j == 4 else f'Max gaze [{low},{high})'
        masks.append((label, (ga >= low) & (ga < high)))
    for name, mask in masks:
        L.append(f'| {name} | {mask.sum()} | {np.sum(mask & ~success)} |')
    L += ['', 'Historical off-manifold nonconvergence locations (not self-inversion failures):', '',
          '| Macro state ID | A D | ADE deg | BDE deg |', '|---:|---:|---:|---:|']
    for i in historical_failures:
        L.append(f'| {i} | {c[i,0]:g} | {c[i,1]:g} | {c[i,2]:g} |')
    L += ['', '## R0-prime gate and CODE V composition', '',
          'R0-prime is **not run**. The requested centre-dominance condition is not met: R0 shape residual is tens of um RMS with a large maximum, and R3 shape RMS is comparable to or larger than its centre RMS. This is not the proposed few-um shape / tens-to-hundreds-um centre split. A centre-only replacement would leave substantial shape error. No other rung is refitted.', '',
          'CODE V order was verified even though the conditional fit is skipped. `C:/CODEV202403_SR1/help/lens_system/tilts.11.11.html`, source line 1 (the HTML is one physical line), "Order of Operations: Decenters and Tilts", lists ADE before BDE before CDE. It states: "For compound tilts, beta must be defined in the alpha-tilted coordinate system" (Greek alpha/beta in the original). `tilts.11.08.html`, source line 1, defines ADE as left-handed about +X and BDE left-handed about +Y.', '',
          'Consequently, with ordinary right-handed matrix notation, the local-to-prior basis map is `R=Rx(-ADE) Ry(-BDE)`, an intrinsic sequence, and the local z-axis in the prior frame is `(-sin(BDE), sin(ADE)*cos(BDE), cos(ADE)*cos(BDE))`. The transpose maps old coordinates into the rotated frame. A rigid-point transverse lever law and a fixed-screen ray intersection law are different hypotheses; downstream refractions/reflections prevent identifying either directly with the final image centre without additional optical assumptions. No guessed physical centre law is fitted here.', '',
          'An A-scaling coefficient is also unidentifiable from 25 A=0 poses because its design column vanishes there. It would need a fixed prior, such as the existing M1, if a future authorized conditional fit were warranted. This does not block the present audit because the decomposition gate already fails.', '',
          '## Problems and limits', '',
          'The hybrid decomposition is a stated forward realization of frozen inverse-calibration components, not the saved blind-solver reconstruction. Masks differ across models and are given explicitly; no calibration rows are silently added to R0/R1. The measured labelled-central-ray offset is not interchangeable with center_absolute. Full-grid true-image self-inversion validates this numerical inverse on the stored grid, not off-manifold inversion, global uniqueness outside the grid, or the accuracy of any reduced physical model. No accommodation/gaze model-error pass is claimed.', '',
          'Files: `continuation.npz` stores frozen-model centre/shape residuals and masks, all table arrays, original fit hash, full-grid recovery/convergence data, historical off-manifold failure IDs, and hybrid G4 roundtrip. `continuation_report.md` is this report. No noise or further workflow step was run.', '']
    (OUT/'continuation_report.md').write_text('\n'.join(L), encoding='utf-8')


if __name__ == '__main__':
    if sys.argv[1:] == ['--continuation']:
        continuation()
    elif not sys.argv[1:]:
        main()
    else:
        raise SystemExit('Usage: p4_reduced_ladder.py [--continuation]')
