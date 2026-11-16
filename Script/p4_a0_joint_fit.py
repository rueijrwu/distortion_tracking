r"""Deterministic A=0 joint tilt/barrel/camera experiment; no data are modified.

Run in PowerShell from any directory (typically 10--60 minutes):
& 'C:\Users\rueijrwu\.pyenv-win-venv\envs\venv\Scripts\python.exe' 'C:\Users\rueijrwu\OneDrive\Project\DistortionTracking\Script\p4_a0_joint_fit.py'

Only report.txt and results.json in the dedicated output directory are written.
Inputs are sensor positions, fixed field labels and normalized commanded gaze.
Centre truth is accessed only after the image-only selection has been frozen.
"""
import sys
sys.dont_write_bytecode = True
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from p4_a0_data import DEFAULT, EXPECTED, safe
from p4_a0_gain import require
import p4_a0_tilt_fit as additive

ROOT = Path(__file__).resolve().parents[1]
EPS = np.finfo(float).eps
FD = EPS**(1./3.)
F, K1, K2 = 425.137557657, 2.943551420, -106.482286730
COND = 1e8
SHIFT = ['a_x', 'b_x', 'c_x', 'a_y', 'b_y', 'c_y']
LINEAR = ['m_xx', 'm_xy', 'm_yy', 'm_yx', 'h_x', 'h_y']
KEY = ['k_x', 'k_y']
QUAD = ['A_x', 'B_x', 'C_x', 'A_y', 'B_y', 'C_y']
BLOCKS = ['q_variation', 'prism_before', 'prism_after', 'lag', 'anisotropic',
          'barrel_before_tilt', 'truncated_keystone']
EXTRA = dict(q_variation=['q2x', 'q2y'], prism_before=['alpha_x', 'alpha_y'],
             prism_after=['alpha_x', 'alpha_y'], lag=['eta'],
             anisotropic=['beta_x', 'beta_y'], barrel_before_tilt=[], truncated_keystone=[])
TARGETS = dict(outer_max_um=1., older_outer_rms_um=0.1, centre_rms_um=0.1,
               centre_max_um=0.3, gaze_max_arcmin=1.)


def read_a0_source(path):
    # The pickle necessarily deserializes whole records. Keep the raw array
    # loader-local; never return it (or a view of it) to fitting/selection.
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
        size = stream.tell()
        stream.seek(0)
        data = pickle.load(stream)
    return data, digest.hexdigest(), size


def validate_a0_geometry(rays):
    for key in EXPECTED[EXPECTED.index('x13'):]:
        if key not in ('center_xsi_local', 'center_ysi_local'):
            require(np.all(np.isfinite(rays[key])), f'A=0 {key} contains nonfinite values')
    norm_error = np.max(np.abs(sum(rays[k]**2 for k in ('l16', 'm16', 'n16'))-1))
    require(norm_error <= 2e-8 and np.all(rays['n16'] < 0), 'S16 LMN norm/sign gate failed')
    require(max(np.max(np.abs(rays[k])) for k in ('x16', 'y16', 'z16')) <= 1e-6
            and np.ptp(rays['zsi']) <= 1e-9, 'Stop origin/sensor plane gate failed')
    return norm_error


def load_a0_outer(path, report):
    """Gain-loader gates, with all centre geometry deferred until the freeze."""
    data, digest, size = read_a0_source(path)
    report.update(input_sha256=digest, input_bytes=size)
    require(isinstance(data, np.ndarray) and data.shape == (21*101*101*9,),
            'Expected the full flat P4 structured ndarray')
    require(data.dtype.names == tuple(EXPECTED) and data.dtype.itemsize == 8*len(EXPECTED),
            'P4 retrace schema mismatch')
    require(all(data.dtype.fields[k][0] == np.dtype('float64') for k in EXPECTED),
            'Every P4 field must be float64')
    meta = (data.dtype.metadata or {}).get('p4_retrace', {})
    require(meta.get('record_count') == len(data) and meta.get('profile') == 'full',
            'Missing/inconsistent full-profile metadata')
    axes = meta.get('axes', {})
    angle = np.arange(101)*.25-12.5
    for key, expected in (('accommodation_d', np.arange(21)*.2),
                          ('ade_deg', angle), ('bde_deg', angle)):
        observed = np.asarray(axes.get(key, []))
        require(observed.shape == expected.shape and np.allclose(observed, expected, rtol=0, atol=1e-8),
                f'Metadata {key} axis mismatch')
    fields = np.array(axes.get('field_xy_y_major', []), dtype=float, copy=True)
    require(fields.shape == (9, 2) and len(np.unique(fields, axis=0)) == 9
            and set(map(tuple, fields)) == {(x, y) for x in (-1, 0, 1) for y in (-1, 0, 1)},
            'Expected exactly the nine fixed square labels')
    outer = np.any(fields != 0, axis=1)
    require(np.all(np.isfinite(data['accommodation_d'])), 'Nonfinite accommodation values')
    selected = np.flatnonzero(data['accommodation_d'] == 0)
    require(len(selected) == 101*101*9, 'A=0 must contain 10,201 complete nine-ray states')
    order = np.lexsort(tuple(data[k][selected] for k in ('ray_index', 'bde_index', 'ade_index')))
    rows = selected[order].reshape(101, 101, 9)
    # A fresh allocation has no base/view path back to the raw pickle. Only
    # labels, gaze, status and index metadata are copied for all nine rays.
    # Centre geometry and the centre-coordinate companion fields stay NaN.
    cube = np.empty(rows.shape, dtype=[(key, 'float64') for key in EXPECTED])
    metadata_keys = EXPECTED[:EXPECTED.index('x13')]
    for key in EXPECTED:
        cube[key] = np.nan
        if key in metadata_keys:
            cube[key] = data[key][rows]
            require(np.all(np.isfinite(cube[key])), f'A=0 {key} contains nonfinite values')
    ai, bi, ri = np.arange(101)[:, None, None], np.arange(101)[None, :, None], np.arange(9)
    aindex = np.unique(cube['accommodation_index'])
    require(len(aindex) == 1 and aindex[0] == 0, 'A=0 accommodation index inconsistent with metadata')
    for key, expected in (('ade_index', ai), ('bde_index', bi), ('ray_index', ri),
                          ('state_index', (aindex[0]*101+ai)*101+bi),
                          ('ade_deg', angle[:, None, None]), ('bde_deg', angle[None, :, None])):
        require(np.max(np.abs(cube[key]-expected)) <= (1e-8 if key.endswith('deg') else 0),
                f'A=0 {key} lattice/state grouping mismatch')
    for key, j in (('field_x_relative', 0), ('field_y_relative', 1)):
        require(np.max(np.abs(cube[key]-fields[:, j])) <= 1e-12, f'{key} metadata mismatch')
    for key in ('rayrsi_status', 'center_rayrsi_status'):
        require(np.all(cube[key] == 0), f'A=0 {key} contains nonzero status')
    for axis in ('ade', 'bde'):
        require(np.max(np.abs(cube[f'rc_{axis}_readback_deg']-cube[f'{axis}_deg'])) <= 1e-7,
                f'{axis} readback mismatch')
    require(np.max(np.abs(cube['cornea_ent_d_thi_readback_mm'])) <= 1e-8,
            'Expected zero Cornea_ENT_D thickness')
    # Establish ray ordering and labels before selecting any positions.
    for key in EXPECTED[EXPECTED.index('x13'):]:
        if key not in ('center_xsi_local', 'center_ysi_local'):
            cube[key][..., outer] = data[key][rows[..., outer]]
    del data
    norm_error = validate_a0_geometry(cube[..., outer])
    cube.flags.writeable = False
    report['validation'] = dict(scope='Full schema/metadata, all A=0 labels/gaze/status/indices; '
        'eight outer rays only for geometry; centre coordinates deferred',
        A_selected_by='accommodation_d == 0', states=10201, rays=91809, geometry_rays=81608,
        max_lmn_squared_norm_error=norm_error, centre_validation_complete=False,
        source_handling='Whole pickle deserialized only within loader scope; returned cube is a detached '
                        'allocation with NaN centre geometry and NaN centre-coordinate companion fields')
    return cube, fields


def load_a0_centre(path, cube, fields, report):
    """First centre-coordinate access, strictly after image-only selection."""
    require(report['selection']['frozen'], 'Centre truth requires frozen image-only selection')
    data, digest, size = read_a0_source(path)
    require(digest == report['input_sha256'] and size == report['input_bytes'],
            'Input changed after image-only selection')
    selected = np.flatnonzero(data['accommodation_d'] == 0)
    order = np.lexsort(tuple(data[k][selected] for k in ('ray_index', 'bde_index', 'ade_index')))
    rows = selected[order].reshape(101, 101, 9)
    ci = int(np.flatnonzero(np.all(fields == 0, axis=1))[0])
    report['selection']['centre_truth_first_access'] = (
        'load_a0_centre: centre = data[rows[..., ci]].copy(); '
        'called from analyze after selection.frozen=True, before G5 and evaluation')
    centre = data[rows[..., ci]].copy()
    for key in EXPECTED:
        require(np.all(np.isfinite(centre[key])), f'A=0 centre {key} contains nonfinite values')
    # These companions encode centre coordinates even on outer-ray records.
    for key in ('center_xsi_local', 'center_ysi_local'):
        require(np.all(np.isfinite(data[key][rows])), f'A=0 {key} contains nonfinite values')
    del data
    norm_error = validate_a0_geometry(centre)
    outer_zsi = cube['zsi'][..., np.any(fields != 0, axis=1)]
    require(max(np.max(outer_zsi), np.max(centre['zsi']))
            - min(np.min(outer_zsi), np.min(centre['zsi'])) <= 1e-9,
            'Stop origin/sensor plane gate failed across outer and centre rays')
    report['validation'].update(centre_validation_complete=True, geometry_rays=91809,
        scope='Full schema/metadata; all A=0 values, states, labels, statuses and geometry; '
              'centre geometry validated only after image-only freeze',
        max_lmn_squared_norm_error=max(report['validation']['max_lmn_squared_norm_error'], norm_error))
    return np.stack((centre['xsi'], centre['ysi']), -1).reshape(-1, 2)


def stage_error(record, stage, exc, ids=None):
    if isinstance(exc, OSError):
        print(f'{stage}: {type(exc).__name__}: {exc}', file=sys.stderr)
    additive.stage_error(record, stage, exc, None if ids is None else
                         np.column_stack((np.asarray(ids)//101, np.asarray(ids) % 101)))


def radial_minima(a, b, radius):
    """Exact minima of both eigenvalues on [0,radius], also for array a,b."""
    a, b, t = np.broadcast_arrays(a, b, np.asarray(radius)**2)
    mt, mr = np.ones_like(t), np.ones_like(t)
    for v in (np.zeros_like(t), t,
              np.clip(np.divide(-a, 2*b, out=np.zeros_like(t), where=b != 0), 0, t),
              np.clip(np.divide(-3*a, 10*b, out=np.zeros_like(t), where=b != 0), 0, t)):
        mt = np.minimum(mt, 1+a*v+b*v*v)
        mr = np.minimum(mr, 1+3*a*v+5*b*v*v)
    return mt, mr


def radial(v, a, b=0., check=True):
    r2 = np.sum(v*v, axis=-1)
    if check:
        mt, mr = radial_minima(a, b, np.sqrt(r2))
        require(np.all(mt > 0) and np.all(mr > 0), 'Radial map leaves origin-connected monotone branch')
    return v*(1+np.asarray(a)*r2+np.asarray(b)*r2*r2)[..., None]


def radial_inverse(v, a, b=0.):
    """Bracketed Newton, positive radial AND tangential eigenvalues to each root."""
    target = np.linalg.norm(v, axis=-1)
    a, b, target = np.broadcast_arrays(a, b, target)
    # First positive radial derivative zero. Stable quadratic roots in r^2.
    turn = np.full(target.shape, np.inf)
    linear = (b == 0) & (a < 0)
    turn[linear] = np.sqrt(-1/(3*a[linear]))
    disc = 9*a*a-20*b
    real = (b != 0) & (disc >= 0)
    qroot = -.5*(3*a+np.copysign(np.sqrt(np.maximum(disc, 0)), 3*a))
    for root in (np.divide(qroot, 5*b, out=np.full_like(target, -1.), where=real),
                 np.divide(1., qroot, out=np.full_like(target, -1.), where=real & (qroot != 0))):
        turn = np.minimum(turn, np.where(real & (root > 0), np.sqrt(np.maximum(root, 0)), np.inf))
    limited = np.isfinite(turn)
    hi = np.where(limited, turn*(1-1e-12), np.maximum(1., 2*target))
    fun = lambda r: r*(1+a*r*r+b*r**4)
    for _ in range(80):
        expand = ~limited & (fun(hi) < target)
        if not np.any(expand):
            break
        hi = np.where(expand, 2*hi, hi)
        require(np.all(np.isfinite(hi)) and np.all(hi < 1e12), 'Cannot bracket radial inverse')
    require(np.all(fun(hi) >= target), 'Target outside origin-connected radial branch')
    bracket = hi.copy()
    lo, r = np.zeros_like(target), np.minimum(target, hi/2)
    tolerance = 2e-14*np.maximum(1., target)
    for iteration in range(101):
        error = fun(r)-target
        converged = np.abs(error) <= tolerance
        if np.all(converged) or iteration == 100:
            break
        lo, hi = np.where(error < 0, r, lo), np.where(error >= 0, r, hi)
        derivative = 1+3*a*r*r+5*b*r**4
        trial = r-error/np.where(derivative > 0, derivative, 1.)
        r = np.where(converged, r, np.where((trial > lo) & (trial < hi) &
                     (derivative > 0), trial, (lo+hi)/2))
    mt, mr = radial_minima(a, b, r)
    require(np.all(converged) and np.all(mt > 0) and np.all(mr > 0),
            'Radial Newton failed convergence/branch/eigenvalue checks')
    out = v*np.divide(r, target, out=np.ones_like(r), where=target != 0)[..., None]
    return out, dict(iterations=iteration, converged=converged, target_radius=target,
                     recovered_radius=r, residual=error, tolerance=tolerance,
                     bracket_high=bracket, min_tangential=mt, min_radial=mr)


def camera(z):
    return radial(z, K1/F**2, K2/F**4)


def camera_inverse(z):
    return radial_inverse(z, K1/F**2, K2/F**4)


def calibrate(cube, fields, record):
    require(set(map(tuple, fields)) == {(x, y) for x in (-1, 0, 1) for y in (-1, 0, 1)},
            'Expected exactly the nine fixed square labels')
    outer = np.any(fields != 0, axis=1)
    labels, g = fields[outer], -fields[outer]
    points = np.stack((cube['xsi'][50, 50, outer], cube['ysi'][50, 50, outer]), -1)
    at = lambda label: int(np.flatnonzero(np.all(labels == label, axis=1))[0])
    c = (points[at((-1, 0))]+points[at((1, 0))])/2
    vertical = (points[at((0, -1))]+points[at((0, 1))])/2
    record.update(c_mm=c, vertical_midpoint_mm=vertical, gauge=-np.eye(2),
                  tolerance_mm=1e-6, fitted_parameters=0,
                  definitions='c: horizontal midpoint; a: undistorted horizontal half-distance; '
                  'b: undistorted x at label (-1,-1); H=2a-b; J=b-a; q0=J/H', defects={})
    defects = record['defects']
    defects.update(c_vs_origin_mm=float(np.linalg.norm(c)),
                   vertical_vs_horizontal_mm=float(np.linalg.norm(vertical-c)),
                   vertical_vs_origin_mm=float(np.linalg.norm(vertical)))
    z, inverse_status = camera_inverse(points-c)
    record['camera_inverse'] = inverse_status
    a = np.linalg.norm(z[at((-1, 0))]-z[at((1, 0))])/2
    b = z[at((-1, -1)), 0]
    H, J = 2*a-b, b-a
    record.update(a_mm=a, b_mm=b, H_mm=H, J_mm=J, q0=J/H if H > 0 else None,
                  mapping=[dict(label=l, g=n, sensor_zero_mm=p, camera_undistorted_mm=v)
                           for l, n, p, v in zip(labels, g, points, z)])
    require(a > 0 and b > 0 and H > 0, 'Nonpositive axial/corner/H calibration')
    q = J/H
    reproduced = H*radial(g, q)
    sensor = c+camera(reproduced)
    recovered, barrel_status = radial_inverse(z/H, q)
    record['barrel_inverse'] = barrel_status
    defects.update(symmetry_mm=float(np.max(np.linalg.norm(z-reproduced, axis=1))),
        eight_point_reproduction_mm=float(np.max(np.linalg.norm(sensor-points, axis=1))),
        camera_forward_inverse_mm=float(np.max(np.linalg.norm(camera(z)-(points-c), axis=1))),
        phi0_inverse_node_mm=float(np.max(np.linalg.norm(recovered-g, axis=1))*H))
    record.update(zero_forward_mm=sensor, phi0_inverse_nodes=recovered,
                  branch_check=dict(camera=inverse_status, barrel=barrel_status),
                  defects_pass=all(value <= 1e-6 for value in defects.values()))
    require(record['defects_pass'], f'Calibration disagreement exceeds 1e-6 mm: {defects}')
    return outer, g, c, H, q


def spec(kind='G2', blocks=()):
    blocks = tuple(b for b in BLOCKS if b in blocks)
    require(not ('prism_before' in blocks and 'prism_after' in blocks), 'Mutually exclusive prism placements')
    names = SHIFT.copy()
    if kind != 'G0':
        names += LINEAR
    if kind == 'G4':
        require(not blocks, 'G4 replaces G2 tilt without selected extensions')
        names += QUAD
    elif kind not in ('G0', 'G1'):
        names += KEY
    for block in blocks:
        names += EXTRA[block]
    require(len(names) == len(set(names)), 'Redundant parameter combination')
    return dict(kind=kind, blocks=blocks, names=names, parameter_count=len(names))


def shift_law(u, p):
    x, y = u[:, 0], u[:, 1]
    return np.column_stack((p.get('a_x', 0)*x+p.get('b_x', 0)*x**3+p.get('c_x', 0)*x*y*y,
                            p.get('a_y', 0)*y+p.get('b_y', 0)*y**3+p.get('c_y', 0)*y*x*x))


def state_parts(u, p, q0):
    u = np.atleast_2d(u)
    x, y = u[:, 0], u[:, 1]
    L = np.zeros((len(u), 2, 2))
    L[:, 0, 0] = 1+p.get('m_xx', 0)*x*x+p.get('m_xy', 0)*y*y
    L[:, 1, 1] = 1+p.get('m_yy', 0)*y*y+p.get('m_yx', 0)*x*x
    L[:, 0, 1], L[:, 1, 0] = p.get('h_x', 0)*x*y, p.get('h_y', 0)*x*y
    singular = np.linalg.svd(L, compute_uv=False)
    require(np.all(np.linalg.det(L) > 0) and np.all(singular[:, -1] > 1e-10)
            and np.all(singular[:, 0]/singular[:, -1] <= COND), 'Invalid L')
    k = u*np.array((p.get('k_x', 0), p.get('k_y', 0)))
    require(np.all(1-np.sum(np.abs(k), axis=1) > 1e-10), 'Rational tilt pole in pattern square')
    q = q0+p.get('q2x', 0)*x*x+p.get('q2y', 0)*y*y
    return shift_law(u, p), L, k, q


def tilt(d, u, L, k, p, configuration):
    v = np.einsum('sij,snj->sni', L, d)
    dot = np.sum(k[:, None, :]*d, axis=-1)
    if configuration['kind'] == 'G4':
        x, y = d[..., 0], d[..., 1]
        ux, uy = u[:, 0, None], u[:, 1, None]
        v += np.stack((p['A_x']*ux*x*x+p['B_x']*ux*y*y+p['C_x']*uy*x*y,
                       p['A_y']*uy*y*y+p['B_y']*uy*x*x+p['C_y']*ux*x*y), -1)
    elif 'truncated_keystone' in configuration['blocks']:
        require(np.all(1-dot > 0), 'Invalid truncated tilt factor')
        v *= (1-dot)[..., None]
    else:
        require(np.all(1+dot > 1e-10), 'Nonpositive rational denominator')
        v /= (1+dot)[..., None]
    return v


def barrel_stage(d, u, s, q, p, blocks):
    if 'anisotropic' in blocks:
        x, y = d[..., 0].copy(), d[..., 1].copy()
        d = d+np.stack((p['beta_x']*u[:, 0, None]*(x*x-y*y),
                        p['beta_y']*u[:, 1, None]*(y*y-x*x)), -1)
    alpha = u*np.array((p.get('alpha_x', 0), p.get('alpha_y', 0)))
    if 'prism_before' in blocks:
        d = d+alpha[:, None, :]*np.sum(d*d, axis=-1)[..., None]
    lag = p.get('eta', 0)*s[:, None, :]
    d = lag+radial(d-lag, q[:, None])
    if 'prism_after' in blocks:
        d = d+alpha[:, None, :]*np.sum(d*d, axis=-1)[..., None]
    return d


def normalized_map(configuration, theta, u, nodes, calibration, include_shift=True):
    u = np.atleast_2d(u)
    d = np.asarray(nodes, dtype=float)
    if d.ndim == 2:
        d = np.broadcast_to(d, (len(u),)+d.shape)
    p = dict(zip(configuration['names'], theta))
    s, L, k, q = state_parts(u, p, calibration['q0'])
    blocks = configuration['blocks']
    if 'barrel_before_tilt' in blocks:
        d = barrel_stage(d, u, s, q, p, blocks)
        d = tilt(d, u, L, k, p, configuration)
    else:
        d = tilt(d, u, L, k, p, configuration)
        d = barrel_stage(d, u, s, q, p, blocks)
    return d+s[:, None, :] if include_shift else d


def t0_baseline(model, nodes):
    """Interpolate only measured zero-template roundoff, preserving exact M14.

    The original eight zero points are constants, not fitted coefficients. A
    tensor cardinal interpolation extends their tiny symmetry defects only so
    the continuous inverse agrees exactly at the original calibration nodes.
    The central zero-template value is fixed at zero.
    """
    value = radial(nodes, model['old_q'])
    for node, defect in zip(model['zero_nodes'], model['zero_template_defects']):
        factor = np.ones(np.asarray(nodes).shape[:-1])
        for j in (0, 1):
            x = np.asarray(nodes)[..., j]
            factor *= (1-x*x) if node[j] == 0 else x*(x+node[j])/2
        value = value+factor[..., None]*defect
    return value


def forward(model, u, nodes, calibration):
    if model['spec']['kind'] == 'T0':
        nodes = np.asarray(nodes)
        phi = additive.node_basis(nodes)
        deformation = (additive.monomials(u) @ (phi*model['beta']).T).reshape(-1, len(nodes), 2)
        old = model['old_calibration']
        return old['c0_mm']+old['H_mm']*(t0_baseline(model, nodes)+deformation)
    if model['spec']['kind'] == 'G5':
        base = forward(model['base'], u, nodes, calibration)
        centre = forward(model['base'], u, np.zeros((1, 2)), calibration)[:, 0]
        law = calibration['c_mm']+shift_law(np.atleast_2d(u), model['law'])
        n = law-centre
        factor = np.prod(1-np.asarray(nodes)**2, axis=-1)
        return base+n[:, None, :]*factor[None, :, None]
    v = normalized_map(model['spec'], model['theta'], u, nodes, calibration)
    return calibration['c_mm']+camera(calibration['H_mm']*v)


def fd_jacobian(fun, p):
    columns, discrepancies = [], []
    for j in range(len(p)):
        step = FD*max(1., abs(p[j]))
        delta = np.zeros_like(p)
        delta[j] = step
        coarse = (fun(p+delta)-fun(p-delta))/(2*step)
        fine = (fun(p+delta/2)-fun(p-delta/2))/step
        difference = np.linalg.norm(coarse-fine)/max(1., np.linalg.norm(fine))
        require(np.all(np.isfinite(fine)) and difference <= 5e-5,
                f'Central finite differences failed half-step check in column {j}: {difference}')
        columns.append(fine)
        discrepancies.append(difference)
    return np.column_stack(columns), max(discrepancies, default=0.)


def scaled_svd(jac):
    scales = np.linalg.norm(jac, axis=0)
    require(np.all(np.isfinite(scales)) and np.all(scales > 0), 'Zero/nonfinite Jacobian column')
    left, singular, right = np.linalg.svd(jac/scales, full_matrices=False)
    cutoff = singular[0]*max(jac.shape)*EPS
    rank = int(np.sum(singular > cutoff))
    condition = singular[0]/singular[-1] if rank == len(scales) else float('inf')
    info = dict(rank=rank, columns=len(scales), full_rank=rank == len(scales),
                singular_values=singular, column_scales=scales, rank_tolerance=cutoff,
                rank_rule='sigma_max*max(J.shape)*eps on column-scaled Jacobian',
                condition_number=condition, design_shape=jac.shape)
    return left, singular, right, scales, info


def solver_svd(jac, bounded):
    if not bounded:
        return scaled_svd(jac)
    # Preserve the accepted 50-iteration unknown-gaze solver's unscaled SVD
    # and damping. Also check the column-scaled Jacobian for acceptance.
    left, singular, right = np.linalg.svd(jac, full_matrices=False)
    cutoff = singular[0]*max(jac.shape)*EPS
    rank = int(np.sum(singular > cutoff))
    scales = np.ones(jac.shape[1])
    conditioning = dict(rank=rank, columns=jac.shape[1], full_rank=rank == jac.shape[1],
        singular_values=singular, column_scales=scales, rank_tolerance=cutoff,
        rank_rule='sigma_max*max(J.shape)*eps on inverse residual Jacobian (accepted gaze solver)',
        condition_number=singular[0]/singular[-1] if rank == jac.shape[1] else float('inf'),
        design_shape=jac.shape)
    conditioning['scaled_jacobian'] = scaled_svd(jac)[4]
    return left, singular, right, scales, conditioning


def solve(fun, initial, bounded=False, max_iterations=200):
    """No Jacobian history retained. A failed/stalled start cannot win a fit."""
    p = np.array(initial, dtype=float)
    info = dict(success=False, converged=False, reason='iteration_limit', iterations=0,
                initial=p.copy(), rejected_trials=0, half_step_max_discrepancy=0.)
    try:
        for iteration in range(max_iterations):
            r = fun(p)
            require(np.all(np.isfinite(r)), 'Nonfinite residual')
            jac, discrepancy = fd_jacobian(fun, p)
            left, sv, right, scales, conditioning = solver_svd(jac, bounded)
            info.update(iterations=iteration+1, conditioning=conditioning)
            info['half_step_max_discrepancy'] = max(info['half_step_max_discrepancy'], discrepancy)
            gradient = (jac/scales).T @ r
            if bounded:
                # Accepted inverse stopping convention, including constrained boundaries.
                gradient_norm = np.linalg.norm(p-np.clip(p-jac.T @ r, -1, 1), np.inf)
                gradient_norm /= max(1., np.linalg.norm(jac, 2)*np.linalg.norm(r))
            else:
                gradient_norm = np.linalg.norm(gradient, np.inf)/max(1., sv[0]*np.linalg.norm(r))
            info['normalized_scaled_gradient'] = gradient_norm
            require(conditioning['full_rank'], 'Rank-deficient scaled Jacobian')
            if gradient_norm <= 1e-10:
                info.update(converged=True, reason='projected_gradient' if bounded else 'scaled_gradient')
                break
            objective = float(r @ r/2)
            damping, accepted = 1e-10*sv[0]**2, False
            for _ in range(18):
                delta_scaled = -right.T @ ((sv/(sv*sv+damping))*(left.T @ r))
                trial = p+delta_scaled/scales
                if bounded:
                    trial = np.clip(trial, -1, 1)
                try:
                    trial_r = fun(trial)
                    trial_cost = float(trial_r @ trial_r/2)
                except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                    trial_cost = float('inf')
                if np.isfinite(trial_cost) and trial_cost <= objective:
                    relative_step = np.linalg.norm(scales*(trial-p), np.inf)/max(1., np.linalg.norm(scales*p, np.inf))
                    if bounded:
                        relative_step = np.linalg.norm(trial-p, np.inf)/max(1., np.linalg.norm(p, np.inf))
                    change = abs(objective-trial_cost)/max(objective, np.finfo(float).tiny)
                    tiny_step = relative_step <= 1e-10 and change <= 1e-12
                    if tiny_step and bounded:
                        # Every bounded tiny step needs stationarity, including
                        # steps clipped to zero (or rounded to zero at a bound).
                        trial_jac, discrepancy = fd_jacobian(fun, trial)
                        info['half_step_max_discrepancy'] = max(info['half_step_max_discrepancy'], discrepancy)
                        projected_gradient = np.linalg.norm(
                            trial-np.clip(trial-trial_jac.T @ trial_r, -1, 1), np.inf)
                        projected_gradient /= max(1., np.linalg.norm(trial_jac, 2)*np.linalg.norm(trial_r))
                        info['projected_step_gradient_norm'] = projected_gradient
                        if not projected_gradient <= 1e-10:
                            info['rejected_trials'] += 1
                            damping *= 10
                            continue
                    info.update(relative_scaled_step=relative_step, relative_objective_change=change,
                                last_accepted_damping=damping)
                    p, accepted = trial, True
                    if tiny_step:
                        info.update(converged=True, reason='step_and_relative_objective')
                    break
                info['rejected_trials'] += 1
                damping *= 10
            if not accepted:
                info['reason'] = 'damping_stalled'
                break
            if info['converged']:
                break
        r = fun(p)
        jac, discrepancy = fd_jacobian(fun, p)
        _, sv, _, scales, conditioning = solver_svd(jac, bounded)
        info['half_step_max_discrepancy'] = max(info['half_step_max_discrepancy'], discrepancy)
        info.update(conditioning=conditioning, objective=float(r @ r/2), final=p.copy())
        # Do not turn a stall or an iteration limit into a successful fit.
        scaled_conditioning = conditioning['scaled_jacobian'] if bounded else conditioning
        info['success'] = bool(info['converged'] and conditioning['full_rank']
                               and scaled_conditioning['full_rank']
                               and scaled_conditioning['condition_number'] <= COND)
        if info['converged'] and not info['success']:
            info['reason'] = 'final_rank_or_condition_failure'
    except Exception as exc:
        info.update(reason=f'{type(exc).__name__}: {exc}', final=p.copy())
    info.update(boundary=bool(np.any(np.abs(p) >= 1-1e-10)) if bounded else False)
    return p, info


def additive_fit(cube, fields, observed, u, train, g, record):
    outer, old_g, c, template, H, old = additive.calibrate(cube, fields)
    require(np.array_equal(old_g, g), 'Additive field ordering changed')
    names, transform = additive.definitions()['M14']
    phi = additive.node_basis(g)
    design = (additive.monomials(u[train])[:, None, :]*phi[None, :, :]).reshape(-1, 18)
    target = ((observed[train]-c-template)/H).ravel()
    coef, conditioning = additive.fit(design, target, transform)
    require(conditioning['full_rank'] and conditioning['condition_number'] <= COND,
            'Additive M14 rank/conditioning failed')
    model = dict(spec=dict(kind='T0', names=names, blocks=(), parameter_count=14),
        theta=coef, beta=transform @ coef, old_calibration=old,
        old_q=(old['c_mm']-old['e_mm'])/H, conditioning=conditioning,
        coefficients=dict(zip(names, coef)), solver=dict(success=True, method='exact additive scaled SVD'),
        errors=[], groups={})
    model.update(zero_nodes=g, zero_template_defects=template/H-radial(g, model['old_q']))
    residual = (forward(model, u[train], g, {})-observed[train])*1000
    model['training_rms_um'] = float(np.sqrt(np.mean(np.sum(residual*residual, axis=-1))))
    model['reproduction_reference'] = dict(training_outer_rms_um=3.62,
        centre_rms_um=4.89, centre_max_um=6.04, interpretation='Approximate accepted reference; independently recomputed')
    record.update(old_calibration=old, training_rms_um=model['training_rms_um'],
                  expected_training_rms_um=3.62, delta_um=model['training_rms_um']-3.62)
    return model


def additive_start(t0, calibration):
    old = t0['coefficients']
    ratio = t0['old_calibration']['H_mm']/calibration['H_mm']
    mapping = dict(a_x='sx_gain', b_x='sx_cubic', c_x='sx_cross_cubic',
                   a_y='sy_gain', b_y='sy_cubic', c_y='sy_cross_cubic',
                   m_xx='mx_self', m_xy='mx_cross', m_yy='my_self', m_yx='my_cross',
                   h_x='shear_x', h_y='shear_y', k_x='k_x', k_y='k_y')
    p = {name: ratio*old[source] for name, source in mapping.items()}
    # Additive columns are already NEGATIVE squared/xy terms. Rational expansion
    # is -g*(k.g); therefore k_new=+ratio*k_old, not its negative.
    p.update(A_x=-p['k_x'], B_x=0., C_x=-p['k_y'],
             A_y=-p['k_y'], B_y=0., C_y=-p['k_x'])
    return p, dict(old_over_new_H=ratio, mapping=mapping,
        keystone_sign='old M14=(-kx*ux*x^2-ky*uy*x*y, -ky*uy*y^2-kx*ux*x*y); '
        'rational expansion -g*(k.g), so k_new=+H_old/H_new*k_old; initialization only',
        camera_derivative_conversion='H ratio only; sensor/camera mismatch is left to nonlinear optimization')


def model_parameters(model):
    return dict(zip(model['spec']['names'], model['theta']))


def start_vector(configuration, values):
    return np.array([values.get(name, 0.) for name in configuration['names']], dtype=float)


def fit_model(configuration, predecessor, derived, u, observed, g, calibration):
    values = (derived.copy() if predecessor is not None and predecessor['spec']['kind'] == 'T0'
              else model_parameters(predecessor) if predecessor is not None else {})
    # G4 replaces kappa, transferring its local quadratic expansion once.
    if configuration['kind'] == 'G4' and predecessor is not None:
        values.update(A_x=-values.get('k_x', 0), B_x=0., C_x=-values.get('k_y', 0),
                      A_y=-values.get('k_y', 0), B_y=0., C_y=-values.get('k_x', 0))
    starts = [('predecessor', start_vector(configuration, values)),
              ('additive_derived', start_vector(configuration, derived))]
    trial_model = dict(spec=configuration)
    def residual(theta):
        trial_model['theta'] = theta
        return ((forward(trial_model, u, g, calibration)-observed)/calibration['H_mm']).ravel()
    attempts, valid = [], []
    for label, initial in starts:
        theta, status = solve(residual, initial)
        status['start'] = label
        attempts.append(status)
        if status['success']:
            # Final branch/local orientation checks include interior points, not only nodes.
            try:
                validate_model(configuration, theta, u, calibration)
                error = residual(theta).reshape(-1, 8, 2)
                rms = float(np.sqrt(np.mean(np.sum(error*error, axis=-1)))*calibration['H_mm']*1000)
                valid.append((rms, label, theta.copy(), status))
                status['training_rms_um'] = rms
            except Exception as exc:
                status.update(success=False, validity_failure=f'{type(exc).__name__}: {exc}')
    result = dict(spec=configuration, parameter_count=configuration['parameter_count'],
                  attempts=attempts, errors=[], groups={}, valid=bool(valid))
    if valid:
        rms, label, theta, status = min(valid, key=lambda v: (v[0], v[1]))
        result.update(theta=theta, coefficients=dict(zip(configuration['names'], theta)),
                      training_rms_um=rms, chosen_start=label, solver=status,
                      conditioning=status['conditioning'])
    return result


def validate_model(configuration, theta, u, calibration):
    axis = np.linspace(-1, 1, 9)
    x, y = np.meshgrid(axis, axis)
    nodes = np.column_stack((x.ravel(), y.ravel()))
    # Chunk only for final validation. Trial forward residuals already check all
    # node denominators, L, barrel and camera eigenvalues during optimization.
    for begin in range(0, len(u), 128):
        gaze = u[begin:begin+128]
        mapping = lambda d: normalized_map(configuration, theta, gaze, d, calibration)
        v = mapping(nodes)
        camera(calibration['H_mm']*v)
        derivatives = []
        for j in (0, 1):
            delta = np.zeros(2)
            delta[j] = FD
            derivatives.append((mapping(nodes+delta)-mapping(nodes-delta))/(2*FD))
        jac = np.stack(derivatives, -1)
        sv = np.linalg.svd(jac, compute_uv=False)
        require(np.all(np.linalg.det(jac) > 1e-10) and np.all(sv[..., -1] > 1e-10)
                and np.all(sv[..., 0]/sv[..., -1] <= COND), 'Fold/singularity in interior validation lattice')


def choose(entries):
    """Relative 1e-8 RMS tie, then parameter count, then explicit listed order."""
    best = min(entry[1]['training_rms_um'] for entry in entries)
    tied = [entry for entry in entries if entry[1]['training_rms_um'] <= best*(1+1e-8)]
    return min(tied, key=lambda entry: (entry[1]['spec']['parameter_count'], entry[0]))


def select_extensions(base, derived, u, observed, g, calibration, selection):
    current, accepted = base, []
    tested = {tuple(base['spec']['blocks'])}
    cache = {tuple(base['spec']['blocks']): base}
    history = selection['history']
    round_number = 0
    while True:
        round_number += 1
        candidates = []
        blocks = current['spec']['blocks']
        for order, block in enumerate(BLOCKS):
            event = dict(phase='forward', round=round_number, block=block,
                         predecessor_blocks=blocks, predecessor_rms_um=current['training_rms_um'])
            if block in blocks:
                event.update(decision='inadmissible', reason='block already present')
                history.append(event)
                continue
            if block in ('prism_before', 'prism_after') and any(b.startswith('prism_') for b in blocks):
                event.update(decision='inadmissible', reason='mutually exclusive prism placements')
                history.append(event)
                continue
            config = spec('G2', blocks+(block,))
            key = config['blocks']
            event['structural_configuration'] = key
            if key in tested:
                event.update(decision='not_revisited', reason='structural configuration previously tested')
                history.append(event)
                continue
            tested.add(key)
            trial = fit_model(config, current, derived, u, observed, g, calibration)
            cache[key] = trial
            event.update(parameter_count=config['parameter_count'], parameter_names=config['names'],
                         coefficients=trial.get('coefficients'), conditioning=trial.get('conditioning'), attempts=trial['attempts'],
                         valid=trial['valid'], training_rms_um=trial.get('training_rms_um'))
            if not trial['valid']:
                event.update(decision='rejected', reason='no converged full-rank well-conditioned branch-valid start')
            else:
                event['decision'] = 'evaluated'
                candidates.append((order, trial, event, block))
            history.append(event)
        if not candidates:
            break
        order, trial = choose([(v[0], v[1]) for v in candidates])
        improvement = 1-trial['training_rms_um']/max(current['training_rms_um'], np.finfo(float).tiny)
        for _, candidate, event, block in candidates:
            if candidate is trial and improvement >= .1:
                event.update(decision='accepted', relative_improvement=improvement)
                accepted.append(block)
            else:
                event.update(decision='rejected', reason='less than 10 percent improvement' if candidate is trial
                             else 'another admissible candidate wins training RMS tie rule')
        if improvement < .1:
            break
        current = trial
    # One backward pass. A previously tested configuration is reused, never
    # re-optimized; its stored refit result includes both required starts.
    for block in reversed(accepted):
        config = spec('G2', tuple(b for b in current['spec']['blocks'] if b != block))
        key = config['blocks']
        event = dict(phase='backward', block=block, structural_configuration=key,
                     predecessor_rms_um=current['training_rms_um'], reused=key in cache)
        if key in cache:
            trial = cache[key]
        else:
            tested.add(key)
            trial = fit_model(config, current, derived, u, observed, g, calibration)
            cache[key] = trial
        event.update(parameter_names=config['names'], coefficients=trial.get('coefficients'),
                     conditioning=trial.get('conditioning'), attempts=trial['attempts'],
                     valid=trial['valid'], training_rms_um=trial.get('training_rms_um'))
        if trial['valid']:
            rise = trial['training_rms_um']/max(current['training_rms_um'], np.finfo(float).tiny)-1
            event['relative_rms_rise'] = rise
            if rise < .1:
                current = trial
                event['decision'] = 'pruned_or_reverted'
            else:
                event['decision'] = 'retained'
        else:
            event.update(decision='retained', reason='removal has no valid fit')
        history.append(event)
    selection.update(final_blocks=current['spec']['blocks'],
                     tested_structural_configurations=list(tested), backward_order=list(reversed(accepted)))
    return current


def base_inverse(model, u, X, calibration):
    """Analytic-stage G0/G1/G2 inverse; also the initializer for other stages."""
    p = model_parameters(model)
    s, L, k, q = state_parts(np.atleast_2d(u), p, calibration['q0'])
    z, _ = camera_inverse(X-calibration['c_mm'])
    d = z/calibration['H_mm']-s[:, None, :]
    d, _ = radial_inverse(d, q[:, None])
    e = np.linalg.solve(L[:, None, :, :], d[..., None])[..., 0]
    denominator = 1-np.sum(k[:, None, :]*e, axis=-1)
    require(np.all(denominator > 1e-10), 'Inverse rational denominator invalid')
    return e/denominator[..., None]


def newton2(fun, target, initial):
    """Vectorized safeguarded 2D Newton with positive Jacobian orientation."""
    nodes = np.array(initial, dtype=float, copy=True)
    for iteration in range(101):
        residual = fun(nodes)-target
        norm = np.linalg.norm(residual, axis=-1)
        done = norm <= 2e-14*np.maximum(1., np.linalg.norm(target, axis=-1))
        columns = []
        for j in (0, 1):
            delta = np.zeros_like(nodes)
            delta[..., j] = FD*np.maximum(1., np.abs(nodes[..., j]))
            columns.append((fun(nodes+delta)-fun(nodes-delta))/(2*delta[..., j, None]))
        jac = np.stack(columns, -1)
        sv = np.linalg.svd(jac, compute_uv=False)
        require(np.all(np.linalg.det(jac) > 1e-12) and np.all(sv[..., -1] > 1e-10)
                and np.all(sv[..., 0]/sv[..., -1] <= COND), 'Singular/folded nonanalytic inverse')
        if np.all(done):
            return nodes
        require(iteration < 100, 'Two-dimensional Newton iteration limit')
        step = np.linalg.solve(jac, residual[..., None])[..., 0]
        accepted = done.copy()
        next_nodes = nodes.copy()
        for fraction in .5**np.arange(24):
            trial = nodes-fraction*step
            try:
                trial_norm = np.linalg.norm(fun(trial)-target, axis=-1)
                take = ~accepted & np.isfinite(trial_norm) & (trial_norm < norm)
                next_nodes[take] = trial[take]
                accepted |= take
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                continue
            if np.all(accepted):
                break
        require(np.all(accepted), 'Two-dimensional Newton stalled')
        nodes = next_nodes
    raise ValueError('Unreachable Newton termination')


def inverse_nodes(model, u, X, calibration):
    if model['spec']['kind'] == 'G5':
        return inverse_nodes(model['base'], u, X, calibration)
    u = np.atleast_2d(u)
    if np.asarray(X).ndim == 2:
        X = np.asarray(X)[None, ...]
    config = model['spec']
    if config['kind'] == 'T0':
        old = model['old_calibration']
        target = (X-old['c0_mm'])/old['H_mm']
        beta = model['beta']
        def mapping(d):
            # Node basis accepts shared nodes; here each gaze can have its own nodes.
            deformation = np.zeros_like(d)
            weights = additive.monomials(u)*beta
            for j, (component, px, py, _, _) in enumerate(additive.TERMS):
                deformation[..., component] += weights[:, j, None]*d[..., 0]**px*d[..., 1]**py
            return t0_baseline(model, d)+deformation
        guess, _ = radial_inverse(target, model['old_q'])
    else:
        guess = base_inverse(model, u, X, calibration)
        if config['kind'] in ('G0', 'G1', 'G2') and not config['blocks']:
            return guess
        z, _ = camera_inverse(X-calibration['c_mm'])
        target = z/calibration['H_mm']
        mapping = lambda d: normalized_map(config, model['theta'], u, d, calibration)
    root = newton2(mapping, target, guess)
    # Deterministic alternate basins expose multiple regular roots. All starts
    # must converge to the same root; inability to establish this is a failure.
    for offset in (np.array((.35, 0.)), np.array((0., .35)), np.array((-.35, -.35))):
        alternate = newton2(mapping, target, guess+offset)
        require(np.max(np.linalg.norm(alternate-root, axis=-1)) <= 1e-9,
                'Ambiguous nonanalytic inverse: distinct regular roots')
    return root


def recover_gaze(model, observed, initial, nodes, calibration):
    residual = lambda u: (inverse_nodes(model, u, observed, calibration)[0]-nodes).ravel()
    hat, status = solve(residual, initial, bounded=True, max_iterations=50)
    status['u_hat'] = hat
    status['at_boundary'] = status['boundary']
    status['rank'] = (status.get('conditioning') or {}).get('rank', 0)
    return hat, status


def metric(error, ids, w):
    error = np.asarray(error)
    norms = np.linalg.norm(error, axis=-1)
    finite = np.isfinite(norms)
    result = dict(expected_vectors=int(norms.size), computed_vectors=int(finite.sum()),
                  complete=bool(finite.all()), rms=None, max=None, signed_bias_xy=None, worst=None)
    if np.any(finite):
        loc = np.unravel_index(np.nanargmax(np.where(finite, norms, np.nan)), norms.shape)
        result.update(rms=float(np.sqrt(np.mean(norms[finite]**2))), max=float(norms[loc]),
                      signed_bias_xy=error[finite].reshape(-1, 2).mean(0),
                      worst=additive.location(int(ids[loc[0]]), w))
        result['worst']['signed_error_xy'] = error[loc]
        if norms.ndim == 2:
            result['worst']['outer_node_index'] = int(loc[1])
    return result


def predict_states(model, u, nodes, calibration, record, stage, ids):
    output = np.full((len(ids), len(nodes), 2), np.nan)
    for begin in range(0, len(ids), 128):
        chosen = ids[begin:begin+128]
        try:
            values = forward(model, u[chosen], nodes, calibration)
            require(np.all(np.isfinite(values)), 'Nonfinite forward prediction')
            output[begin:begin+len(chosen)] = values
        except Exception:
            for k, index in enumerate(chosen):
                try:
                    values = forward(model, u[index:index+1], nodes, calibration)
                    require(np.all(np.isfinite(values)), 'Nonfinite forward prediction')
                    output[begin+k] = values[0]
                except Exception as exc:
                    stage_error(record, stage, exc, [index])
    return output


def inverse_evaluation(model, ids, full, all_group_ids, observed, nearest, u, g,
                       calibration, truth, w, record):
    hats, statuses = [], []
    grid_known = np.full((len(ids), 8, 2), np.nan)
    grid_recovered = np.full_like(grid_known, np.nan)
    centres = np.full((len(ids), 2), np.nan)
    for j, index in enumerate(ids):
        initial = u[nearest[index]].copy()
        try:
            hat, status = recover_gaze(model, observed[index], initial, g, calibration)
            require(np.all(np.isfinite(hat)), 'Nonfinite recovered gaze')
        except Exception as exc:
            hat = initial
            status = dict(success=False, converged=False, reason=f'{type(exc).__name__}: {exc}',
                          rank=0, iterations=0, boundary=bool(np.any(np.abs(hat) >= 1-1e-10)))
        if not status['success']:
            stage_error(record, 'unknown_gaze', status['reason'], [index])
        status.update(state=additive.location(index, w), initial_training_state=additive.location(nearest[index], w),
                      u_hat=hat, at_boundary=status['boundary'])
        hats.append(hat)
        statuses.append(status)
        try:
            grid_known[j] = 1000*calibration['H_mm']*(inverse_nodes(model, u[index], observed[index], calibration)[0]-g)
            grid_recovered[j] = 1000*calibration['H_mm']*(inverse_nodes(model, hat, observed[index], calibration)[0]-g)
        except Exception as exc:
            stage_error(record, 'grid_equivalent_inverse', exc, [index])
        try:
            centres[j] = forward(model, hat[None, :], np.zeros((1, 2)), calibration)[0, 0]
        except Exception as exc:
            stage_error(record, 'centre_at_recovered_gaze', exc, [index])
    hats = np.asarray(hats).reshape(-1, 2)
    gaze_errors = (hats-u[ids])*12.5*60
    centre_errors = (centres-truth[ids])*1000
    result = dict(**metric(gaze_errors, ids, w), tested_states=len(ids),
                  full_group=bool(full and len(ids) == len(all_group_ids)),
                  failed_inversions=sum(not s['success'] for s in statuses),
                  boundary_solutions=sum(s['boundary'] for s in statuses),
                  rank_deficient_solutions=sum(s['rank'] < 2 for s in statuses),
                  unconverged=sum(not s['converged'] for s in statuses),
                  statuses=statuses, u_hat=hats, error_arcmin=gaze_errors,
                  failure_estimate='Failed states retain their solver estimate (or nearest-training start); never dropped')
    record.update(unknown_arcmin=result,
        centre_recovered_um=metric(centre_errors, ids, w),
        grid_equivalent_known_um=metric(grid_known, ids, w),
        grid_equivalent_recovered_um=metric(grid_recovered, ids, w),
        inverse_per_state=dict(ids=ids, centre_recovered_error_um=centre_errors,
            centre_recovered_sensor_mm=centres, grid_known_error_um=grid_known,
            grid_recovered_error_um=grid_recovered))
    record['grid_equivalent_scale'] = dict(H_mm=calibration['H_mm'],
        convention='Common joint-calibration H, as specified; not a sensor residual.')
    if model['spec']['kind'] == 'T0':
        old_H = model['old_calibration']['H_mm']
        ratio = old_H/calibration['H_mm']
        record['original_T0_grid_scale_extra'] = dict(H_mm=old_H,
            known_um=metric(grid_known*ratio, ids, w), recovered_um=metric(grid_recovered*ratio, ids, w))
    if model['spec']['kind'] == 'G5':
        record['inverse_definition'] = ('Explicitly reused G4 outer-node inverse and gaze solution; '
            'G5 sensor-space imposed centre law evaluated at that recovered gaze. '
            'Grid-equivalent inverse diagnostic belongs to G4, not to the corrected G5 interior map.')


def target_checks(metrics, group_size):
    outer, centre = metrics.get('outer_um') or {}, metrics.get('centre_known_um') or {}
    inv, recovered = metrics.get('unknown_arcmin') or {}, metrics.get('centre_recovered_um') or {}
    outer_ok = bool(outer.get('complete') and outer.get('max') is not None and outer['max'] <= 1.)
    known_ok = bool(centre.get('complete') and centre.get('rms') is not None and
                    centre['rms'] <= .1 and centre['max'] <= .3)
    sampled_centre_ok = bool(recovered.get('complete') and recovered.get('rms') is not None and
                            recovered['rms'] <= .1 and recovered['max'] <= .3)
    sampled_gaze_ok = bool(inv.get('complete') and inv.get('tested_states', 0) > 0 and
                          inv.get('max') is not None and inv['max'] <= 1. and inv['failed_inversions'] == 0)
    full = bool(inv.get('full_group') and inv.get('tested_states') == group_size)
    return dict(outer_max=outer_ok,
        older_outer_rms_extra=bool(outer.get('complete') and outer.get('rms') is not None and outer['rms'] <= .1),
        centre_known=known_ok, centre_recovered_sampled=sampled_centre_ok,
        centre_recovered_full=sampled_centre_ok if full else None,
        gaze_sampled=sampled_gaze_ok, gaze_full=sampled_gaze_ok if full else None,
        full_coverage=full, all_stage_checks=not bool(metrics.get('errors')),
        all_full=bool(full and not metrics.get('errors') and outer_ok and known_ok and sampled_centre_ok and sampled_gaze_ok),
        scope='full split' if full else 'sampled inversion only; full compliance not established')


def evaluate_model(name, model, full, observed, nearest, u, g, calibration, truth, w, groups):
    model['evaluation_scope'] = 'full inversion' if full else 'sampled inversion only'
    ids_all = np.arange(len(u))
    prediction = predict_states(model, u, g, calibration, model, 'outer_forward', ids_all)
    centres = predict_states(model, u, np.zeros((1, 2)), calibration, model, 'centre_known', ids_all)[:, 0]
    for group, ids in groups.items():
        result = model['groups'][group] = dict(errors=[])
        outer_error = 1000*(prediction[ids]-observed[ids])
        centre_error = 1000*(centres[ids]-truth[ids])
        result.update(outer_um=metric(outer_error, ids, w), centre_known_um=metric(centre_error, ids, w),
            per_state=dict(ids=ids, outer_error_um=outer_error, centre_known_error_um=centre_error))
        try:
            sample = additive.inverse_sample_ids(ids, full, groups['train'], group == 'train')
            result['inverse_sampling'] = additive.inverse_sampling(sample, full)
            require(np.array_equal(np.sort(10200-sample), np.sort(sample)), 'Inverse lattice not reflection closed')
            inverse_evaluation(model, sample, full, ids, observed, nearest, u, g, calibration, truth, w, result)
        except Exception as exc:
            stage_error(result, 'inverse_evaluation', exc, ids)
        result['target_checks'] = target_checks(result, len(ids))
    try:
        diagnostics(model, u, g, observed, prediction, centres, truth, calibration, w)
    except Exception as exc:
        stage_error(model, 'cross_checks', exc)
    if name == 'T0':
        combined = np.concatenate(list(groups.values()))
        model['reproduction_reference'].update(centre_recomputed_um=metric(1000*(centres[combined]-truth[combined]), combined, w))


def undo_barrel(d, u, s, q, p, blocks):
    alpha = u*np.array((p.get('alpha_x', 0), p.get('alpha_y', 0)))
    if 'prism_after' in blocks:
        fun = lambda v: v+alpha[:, None, :]*np.sum(v*v, axis=-1)[..., None]
        d = newton2(fun, d, d)
    lag = p.get('eta', 0)*s[:, None, :]
    d = lag+radial_inverse(d-lag, q[:, None])[0]
    # Return the actual pre-barrel frame; optional pre-prism/anisotropic
    # changes remain visible to the homography closure diagnostic.
    return d


def homography_fit(nodes, values):
    nstate, nnode, _ = values.shape
    x, y = np.broadcast_to(nodes, values.shape).transpose(2, 0, 1)
    X, Y = values[..., 0], values[..., 1]
    z, one = np.zeros_like(x), np.ones_like(x)
    rowx = np.stack((x, y, one, z, z, z, -X*x, -X*y), -1)
    rowy = np.stack((z, z, z, x, y, one, -Y*x, -Y*y), -1)
    design = np.stack((rowx, rowy), 2).reshape(nstate, 2*nnode, 8)
    target = values.reshape(nstate, 2*nnode)
    scales = np.linalg.norm(design, axis=1)
    require(np.all(scales > 0), 'Zero homography column')
    left, singular, right = np.linalg.svd(design/scales[:, None, :], full_matrices=False)
    cutoff = singular[:, :1]*max(design.shape[1:])*EPS
    ranks = np.sum(singular > cutoff, axis=1)
    require(np.all(ranks == 8), 'Rank-deficient homography diagnostic')
    coef = np.einsum('sji,sj->si', right, np.einsum('sij,si->sj', left, target)/singular)/scales
    matrix = np.concatenate((coef, np.ones((nstate, 1))), axis=1).reshape(-1, 3, 3)
    homogeneous = np.column_stack((nodes, np.ones(len(nodes))))
    v = np.einsum('sij,nj->sni', matrix, homogeneous)
    predicted = v[..., :2]/v[..., 2, None]
    return matrix, predicted-values, dict(rank=ranks, singular_values=singular,
                                         rank_tolerance=cutoff[:, 0])


def diagnostics(model, u, g, observed, prediction, centres, truth, calibration, w):
    allids = np.arange(len(u))
    b = prediction.mean(1)-centres
    identity = centres-prediction.mean(1)+b
    traced_minus_mean = truth-observed.mean(1)
    discrepancy = traced_minus_mean+b
    model['mean_centre_cross_check'] = dict(definition='b=mean_i F_u(g_i)-F_u(0), complete sensor composition',
        bias_b_mm=b, identity_F0_minus_mean_plus_b_um=identity*1000,
        identity_max_um=float(np.nanmax(np.linalg.norm(identity, axis=-1))*1000),
        traced_centre_minus_observed_mean_mm=traced_minus_mean,
        discrepancy_traced_minus_mean_vs_negative_b_um=discrepancy*1000,
        discrepancy=metric(discrepancy*1000, allids, w),
        interpretation='Composition identity is exact algebraically; traced-centre comparison is a measured discrepancy, not an exact data identity. No mean estimates a centre.')
    if model['spec']['kind'] == 'G5':
        model['cross_checks'] = dict(inherited_from='G4',
            outer_null_correction_max_um=float(np.max(np.linalg.norm(prediction-forward(model['base'], u, g, calibration), axis=-1))*1000))
        return
    if model['spec']['kind'] == 'T0':
        # Additive T0 has a different baseline; its quadratic frame is the
        # original sensor frame divided by its original H.
        coefficients = (additive.monomials(u)*model['beta'])
        lookup = {name: coefficients[:, j] for j, name in enumerate(additive.PRIMITIVES)}
        local = np.stack((np.column_stack((lookup['qx_x2_ux'], lookup['qx_y2_ux'], lookup['qx_xy_uy'])),
                          np.column_stack((lookup['qy_y2_uy'], lookup['qy_x2_uy'], lookup['qy_xy_ux']))), 1)
        frame = 'T0 original normalized sensor displacement; no camera/barrel undo applies'
        pre = None
    else:
        p = model_parameters(model)
        s, L, k, q = state_parts(u, p, calibration['q0'])
        config = model['spec']
        # Local quadratics immediately before barrel (or before tilt for the
        # order replacement). Finite differences are diagnostics only.
        def before_barrel(nodes):
            d = np.broadcast_to(nodes, (len(u),)+nodes.shape).copy()
            if 'barrel_before_tilt' not in config['blocks']:
                d = tilt(d, u, L, k, p, config)
            if 'anisotropic' in config['blocks']:
                x, y = d[..., 0].copy(), d[..., 1].copy()
                d += np.stack((p['beta_x']*u[:, 0, None]*(x*x-y*y),
                               p['beta_y']*u[:, 1, None]*(y*y-x*x)), -1)
            if 'prism_before' in config['blocks']:
                alpha = u*np.array((p['alpha_x'], p['alpha_y']))
                d += alpha[:, None, :]*np.sum(d*d, axis=-1)[..., None]
            return d
        step = 2e-4
        stencil = step*np.array(((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)))
        v = before_barrel(stencil)
        xx = (v[:, 1]+v[:, 2]-2*v[:, 0])/(2*step**2)
        yy = (v[:, 3]+v[:, 4]-2*v[:, 0])/(2*step**2)
        xy = (v[:, 5]-v[:, 6]-v[:, 7]+v[:, 8])/(4*step**2)
        # x component: x^2,y^2,xy; y component: y^2,x^2,xy.
        local = np.stack((np.column_stack((xx[:, 0], yy[:, 0], xy[:, 0])),
                          np.column_stack((yy[:, 1], xx[:, 1], xy[:, 1]))), 1)
        frame = 'actual immediate pre-barrel frame; order replacement has identity tilt before barrel'
        if 'barrel_before_tilt' in config['blocks']:
            frame += '; tilt quadratic separately reported after barrel'
        # Undo camera and shift on OBSERVATIONS, then barrel. Homography fit
        # closure is reported separately from forward sensor error.
        z = camera_inverse(observed-calibration['c_mm'])[0]/calibration['H_mm']-s[:, None, :]
        if 'barrel_before_tilt' in config['blocks']:
            e = np.linalg.solve(L[:, None, :, :], z[..., None])[..., 0]
            if 'truncated_keystone' in config['blocks']:
                mapping = lambda d: tilt(d, u, L, k, p, config)
                z = newton2(mapping, z, e)
            else:
                denom = 1-np.sum(k[:, None, :]*e, axis=-1)
                require(np.all(denom > 0), 'Order-control diagnostic inverse denominator')
                z = e/denom[..., None]
        pre = undo_barrel(z, u, s, q, p, config['blocks'])
    # Factor the local linear map before comparing keystone estimates; this
    # removes L mixing and makes the sign convention explicit.
    if model['spec']['kind'] not in ('T0',):
        linear = (before_barrel(np.array(((step, 0.), (-step, 0.), (0., step), (0., -step)))))
        J = np.stack(((linear[:, 0]-linear[:, 1])/(2*step),
                      (linear[:, 2]-linear[:, 3])/(2*step)), -1)
        raw_quad = np.stack((xx, yy, xy), -1)
        canonical = np.linalg.solve(J[:, None, :, :], raw_quad.transpose(0, 2, 1)[..., None])[..., 0]
        cx, cy = canonical[:, :, 0], canonical[:, :, 1]
        same = -np.column_stack((cx[:, 0], cy[:, 1]))
        other = -np.column_stack((cy[:, 2], cx[:, 2]))
    else:
        same = -np.column_stack((local[:, 0, 0], local[:, 1, 0]))
        other = -np.column_stack((local[:, 1, 2], local[:, 0, 2]))
    cross = dict(local_quadratic_coefficients=local,
        local_columns=['same_squared', 'other_squared', 'xy'], frame=frame,
        keystone_same_squared=same, keystone_other_component_xy=other,
        keystone_inference_difference=same-other,
        sign_convention='T=Lg/(1+k.g); after undoing local L: squared coefficient=-k, other-component xy=-k.',
        decentering_convention='D=(p_x*(3*x^2+y^2)+2*p_y*x*y, p_y*(x^2+3*y^2)+2*p_x*x*y). '
        'Writing D=alpha*r^2-g*(k.g) gives k=-2*alpha, p=alpha. Thus k+2*alpha is the unconstrained 1:2 check.',
        decentering_relation_imposed=False)
    p = model_parameters(model)
    cross['fitted_keystone_plus_twice_prism'] = [p.get('k_x', 0)+2*p.get('alpha_x', 0),
                                               p.get('k_y', 0)+2*p.get('alpha_y', 0)]
    cross['decentering_comparison_frame'] = ('Fitted coefficients use u; prism-after is post-barrel, '
        'so its raw coefficient comparison is descriptive, not an equality in the pre-barrel frame.')
    if pre is not None:
        matrix, error, status = homography_fit(g, pre)
        cross['homography_after_undo'] = dict(matrix=matrix, closure_normalized=error,
            grid_equivalent_closure_um=metric(error*calibration['H_mm']*1000, allids, w),
            conditioning=status, note='Observed points; undo camera/shift/barrel; for barrel-before-tilt also undo tilt first. Optional pre-barrel blocks remain. Closure is not sensor fit residual.')
    model['cross_checks'] = cross


def design_assisted(g4, u, truth, train, calibration):
    # Separate stage: measured sensor centre positions, never outer residuals.
    ux, uy = u[train, 0], u[train, 1]
    design = np.zeros((len(train), 2, 6))
    design[:, 0, :3] = np.column_stack((ux, ux**3, ux*uy*uy))
    design[:, 1, 3:] = np.column_stack((uy, uy**3, uy*ux*ux))
    coef, conditioning = additive.fit(design.reshape(-1, 6),
        (truth[train]-calibration['c_mm']).ravel(), np.eye(6))
    require(conditioning['full_rank'] and conditioning['condition_number'] <= COND,
            'Centre law rank/conditioning failure')
    law = dict(zip(SHIFT, coef))
    error = calibration['c_mm']+shift_law(u[train], law)-truth[train]
    return dict(spec=dict(kind='G5', names=g4['spec']['names']+['centre_'+n for n in SHIFT],
                         blocks=(), parameter_count=24), base=g4, base_model='G4',
        parameter_count=24, image_only=False, design_assisted=True, independently_fitted_null_coefficients=0,
        theta=np.r_[g4['theta'], coef], law=law,
        coefficients=dict(image_stage=g4['coefficients'], sensor_centre_law_mm=law),
        conditioning=dict(image_stage=g4['conditioning'], centre_stage=conditioning),
        solver=dict(success=True, method='G4 image fit retained; independent six-column sensor centre SVD'),
        centre_stage_training_rms_um=float(np.sqrt(np.mean(np.sum(error*error, axis=-1)))*1000),
        training_rms_um=g4['training_rms_um'], valid=True, groups={}, errors=[],
        definition='F5=F4+[C_law(u)-F4(0)]*(1-x^2)*(1-y^2); outer predictions fixed; gaze inversion reuses G4')


def analyze(report):
    try:
        cube, fields = load_a0_outer((ROOT/DEFAULT).resolve(), report)
        w, groups = additive.partition(cube)
        u = w/12.5
        outer = np.any(fields != 0, axis=1)
        g = -fields[outer]
        observed = np.stack((cube['xsi'][..., outer], cube['ysi'][..., outer]), -1).reshape(-1, 8, 2)
        train = groups['train']
        report['splits'] = {name: dict(states=len(ids), ids=ids,
            ade_bde_indices=np.column_stack((ids//101, ids % 101))) for name, ids in groups.items()}
        report['field_ordering'] = dict(labels=fields[outer], g=g)
        report['zero_reference'] = dict(ade_index=50, bde_index=50, used_for_fit=False)
    except Exception as exc:
        stage_error(report, 'load_partition', exc)
        return
    models = report['models']
    try:
        t0 = additive_fit(cube, fields, observed, u, train, g, report['additive_reproduction'])
        models['T0'] = t0
    except Exception as exc:
        stage_error(report, 'additive_reproduction', exc)
        return
    calibration = report['calibration']
    try:
        _, calibrated_g, c, H, q0 = calibrate(cube, fields, calibration)
        require(np.array_equal(calibrated_g, g), 'Joint calibration changed field order')
        derived, report['initialization'] = additive_start(t0, calibration)
        # At u=0 all fitted coefficients vanish; this is Phi_0^-1(g).
        reference = dict(spec=spec('G2'), theta=np.zeros(14))
        zero = forward(reference, np.zeros((1, 2)), g, calibration)[0]
        zero_observed = observed[50*101+50]
        calibration['F0_vs_Phi0_inverse_max_mm'] = float(np.max(np.linalg.norm(zero-zero_observed, axis=1)))
        require(calibration['F0_vs_Phi0_inverse_max_mm'] <= 1e-6, 'F0 differs from Phi0 inverse nodes')
    except Exception as exc:
        stage_error(report, 'joint_calibration', exc)
        return
    predecessor = t0
    for name in ('G0', 'G1', 'G2'):
        try:
            model = fit_model(spec(name), predecessor, derived, u[train], observed[train], g, calibration)
            models[name] = model
            require(model['valid'], 'Both required training starts failed')
            predecessor = model
        except Exception as exc:
            stage_error(models.setdefault(name, dict(errors=[], groups={})), 'training_fit', exc)
    if (models.get('G2') or {}).get('valid'):
        try:
            models['G3'] = select_extensions(models['G2'], derived, u[train], observed[train], g,
                                             calibration, report['selection'])
            # G3 can share a cached G2 object. Give its report fields independent ownership.
            models['G3'] = dict(models['G3'], groups={}, errors=list(models['G3']['errors']))
        except Exception as exc:
            stage_error(report, 'training_selection', exc)
        for name, configuration in (('order_control', spec('G2', ('barrel_before_tilt',))), ('G4', spec('G4'))):
            try:
                models[name] = fit_model(configuration, models['G2'], derived, u[train], observed[train], g, calibration)
                require(models[name]['valid'], 'Both required training starts failed')
            except Exception as exc:
                stage_error(models.setdefault(name, dict(errors=[], groups={})), 'training_fit', exc)
    # Freeze before any centre positions are selected or accessed. T0 is a
    # separately calibrated baseline and participates as an image-only diagnostic.
    order = ['T0', 'G0', 'G1', 'G2', 'G3', 'order_control', 'G4']
    eligible = [(i, models[name]) for i, name in enumerate(order)
                if name in models and models[name].get('theta') is not None and not models[name].get('errors')]
    if not eligible:
        stage_error(report, 'winner_freeze', 'No valid image-only model')
        return
    chosen_order, winner = choose(eligible)
    selected = order[chosen_order]
    report['selection'].update(selected_image_only=selected, frozen=True,
        frozen_training_rms_um=winner['training_rms_um'],
        frozen_parameter_count=winner['spec']['parameter_count'], image_model_order=order,
        centre_truth_accessed_at_freeze=False)
    try:
        truth = load_a0_centre((ROOT/DEFAULT).resolve(), cube, fields, report)
    except Exception as exc:
        stage_error(report, 'centre_truth_setup', exc)
        return
    try:
        nearest = additive.nearest_training(observed.reshape(-1, 16), train)
        report['nearest_training_ids'] = nearest
    except Exception as exc:
        stage_error(report, 'nearest_training_setup', exc)
        return
    if (models.get('G4') or {}).get('valid'):
        try:
            models['G5_design_assisted'] = design_assisted(models['G4'], u, truth, train, calibration)
        except Exception as exc:
            stage_error(report, 'design_assisted_centre_stage', exc)
    for name, model in models.items():
        if model.get('theta') is None:
            continue
        try:
            evaluate_model(name, model, name == selected, observed, nearest, u, g, calibration,
                           truth, w, groups)
        except Exception as exc:
            stage_error(model, 'evaluation', exc)


def initial_report():
    return dict(status='initialized', input=str(DEFAULT), numpy_version=np.__version__, errors=[],
        camera=dict(f_mm=F, k1=K1, k2=K2, principal_point_mm=[0., 0.], fitted_parameters=0),
        conventions=dict(g='-fixed field label', w='(BDE,-ADE) degrees', u='w/12.5',
            forward='c+Cam(H*(s+B_q(T(g)))); tilt then pattern-centred barrel then shift then camera',
            residual='prediction minus observation; equal weight per sensor component; vector RMS',
            fit_data='training outer nodes only; zero gaze calibration reference excluded from fitting',
            centre_truth='evaluation after image-only freeze; G5 centre law training stage explicitly design-assisted'),
        targets=TARGETS, calibration={}, additive_reproduction={}, initialization={}, splits={}, models={},
        selection=dict(history=[], frozen=False, selected_image_only=None, centre_truth_first_access=None,
            rule='Training vector RMS only; relative 1e-8 ties favor fewer coefficients then listed order; '
                 'grouped additions require >=10 percent improvement; reverse backward pass reverts if rise <10 percent; '
                 'structural configurations are never reoptimized; rank and scaled condition <=1e8 required.',
            candidate_order=BLOCKS, forbidden=['higher radial powers', 'four-fold cubic terms', 'free null mode']),
        solver_settings=dict(method='NumPy column-scaled central-finite-difference damped SVD Gauss-Newton',
            gaze_step='Accepted unscaled-Jacobian SVD damped step; central finite differences through the inverse',
            fit_iterations_per_start=200, gaze_iterations=50, damping_initial='1e-10*sigma_max^2',
            damping_increases=18, damping_multiplier=10., difference_step='eps^(1/3)*max(1,abs(theta_j))',
            half_step_check_relative_limit=5e-5, difference_used='half-step central difference',
            rank_cutoff='sigma_max*max(J.shape)*eps after column scaling', condition_limit=COND,
            normalized_scaled_gradient_tolerance=1e-10, relative_scaled_step_tolerance=1e-10,
            fit_gradient='||(J/column_scales).T*r||inf/max(1,sigma_max*||r||2)',
            relative_objective_tolerance=1e-12, gaze_bounds=[-1., 1.],
            gaze_gradient='||u-clip(u-J.T*r,-1,1)||inf/max(1,||J||2*||r||2)',
            projected_step_stop='Bounded tiny step and unchanged objective also require projected gradient <=1e-10; '
                                'otherwise increase damping or fail as stalled',
            gaze_acceptance='Converged, full unscaled and column-scaled rank two, scaled condition <=1e8; '
                            'violations count as failed inversions with state locations',
            starts=['predecessor', 'independent additive-derived approximation'],
            inverse_initialization='nearest measured TRAINING configuration, exact chunked Euclidean search; self excluded',
            radial_iterations=100, radial_tolerance='2e-14*max(1,target_radius), per point',
            nonanalytic_inverse='safeguarded 2D Newton from base inverse; three deterministic alternate-basin checks; '
                                'positive orientation and full rank required; nonconvergence/ambiguity rejected',
            sampling='Frozen image-only winner: all 10200 nonzero states. Others: reflection-closed absolute '
                     'five-index lattice intersected with splits. Failed states remain counted; sampled compliance only.',
            validation='Branch/eigenvalue checks at all predicted and inverse points; final 9x9 interior '
                       'pattern lattice at every training gaze; no claim of global polynomial injectivity certification.',
            runtime_expectation_minutes=[10, 60]),
        definitions=dict(T0='Exact original additive M14, original sensor calibration and template; 14 coefficients',
            G0='Six parity-cubic shifts; L=I, kappa=0, q=q0',
            G1='G0 + four quadratic magnifications and two shears; 12 coefficients',
            G2='G1 + rational kappa=(kx*ux,ky*uy); 14 coefficients',
            G3='Training-selected additions/replacements starting at G2; explicit fitted coefficient count',
            order_control='G2 with pattern-centred barrel before rational tilt; 14 coefficients',
            G4='G1 linear/shift plus six quadratic Q coefficients replacing rational tilt; 18 coefficients',
            G5_design_assisted='Image-only G4 retained plus six sensor-centre law coefficients; '
                               '24 across stages, zero independently fitted null-mode coefficients'),
        formulas=dict(shift='sx=ax*ux+bx*ux^3+cx*ux*uy^2; sy=ay*uy+by*uy^3+cy*uy*ux^2',
            L='[[1+mxx*ux^2+mxy*uy^2,hx*ux*uy],[hy*ux*uy,1+myy*uy^2+myx*ux^2]]',
            Q='(Ax*ux*x^2+Bx*ux*y^2+Cx*uy*x*y, Ay*uy*y^2+By*uy*x^2+Cy*ux*x*y)',
            q_variation='q0+q2x*ux^2+q2y*uy^2',
            prism='(alpha_x*ux,alpha_y*uy)*|d|^2, before OR after barrel',
            lag='ell=eta*s; ell+B(d-ell)',
            anisotropic='(beta_x*ux*(dx^2-dy^2),beta_y*uy*(dy^2-dx^2)), immediately before barrel',
            truncated='L*g*(1-kappa.g)', G5='F4(g)+(C_law-F4(0))*(1-x^2)*(1-y^2)'))


def has_errors(value):
    if isinstance(value, dict):
        return bool(value.get('errors')) or any(has_errors(v) for k, v in value.items() if k != 'base')
    if isinstance(value, (list, tuple)):
        return any(has_errors(v) for v in value)
    return False


def exportable(value):
    if isinstance(value, dict):
        # G5 retains a runtime G4 reference, but JSON references the separately
        # complete G4 record by name rather than duplicating it.
        return {k: exportable(v) for k, v in value.items() if k != 'base'}
    if isinstance(value, (tuple, list)):
        return [exportable(v) for v in value]
    return value


def compact_text(report):
    def number(value):
        return 'NOT COMPUTED' if value is None else f'{value:.9g}'
    def vector(value):
        return 'NOT COMPUTED' if value is None else np.array2string(np.asarray(value), precision=8,
                       threshold=30, max_line_width=10000).replace('\n', '')
    lines = ['A=0 joint fit; fixed camera; sensor coordinates in mm; errors prediction minus observation.',
        'Status: '+report['status'],
        'g=-label; w=(BDE,-ADE) deg; u=w/12.5. Equal forward sensor weights; training outer nodes only.',
        'Order: tilt, pattern-centred barrel, shift, camera. T0 preserves its original additive calibration.',
        'No random perturbations; no centre estimate from an outer mean; no directional ray ratios as model inputs.',
        'Targets: outer max <=1 um; centre RMS/max <=0.1/0.3 um at known AND recovered gaze; gaze max <=1 arcmin.',
        'Older outer RMS <=0.1 um is an additional labelled diagnostic. Missing/failed coverage cannot pass.',
        'Only frozen image-only winner has full inversion coverage; all others have sampled inversion compliance only.',
        'Boundary inversions require convergence, rank two and scaled condition <=1e8; all failed states remain counted.',
        'Camera f/k1/k2: '+vector([F, K1, K2])+'; principal point fixed at zero.',
        'Winner frozen before centre truth: '+str(report['selection'].get('selected_image_only')),
        'Centre truth first accessed: '+str(report['selection'].get('centre_truth_first_access')),
        'Selection: >=10% grouped improvement; relative 1e-8 tie: fewer coefficients, listed order; reverse pruning <10% rise.',
        'Forward-fit solver: column-scaled central finite differences with half-step checks; damped SVD; full rank; condition <=1e8.',
        'Unknown-gaze solver: unscaled damped SVD, 50 iterations; rank two and column-scaled condition <=1e8 required.',
        'Stops: normalized gradient <=1e-10, or scaled relative step <=1e-10 AND objective change <=1e-12; stalls/limits fail.',
        'Bounded tiny-step stop also requires projected gradient <=1e-10; otherwise increase damping or fail as stalled.',
        'Radial inverses: bracketed Newton, 100 steps, per-point tolerance 2e-14*max(1,target radius), positive eigenvalues.',
        'Nonanalytic inverses: 2D Newton with deterministic alternate-basin checks; finite checks do not certify global injectivity.']
    c = report['calibration']
    lines += ['Calibration c='+vector(c.get('c_mm'))+'; vertical='+vector(c.get('vertical_midpoint_mm')),
              'Undistorted a/b/H/J/q0: '+vector([c.get(k) for k in ('a_mm', 'b_mm', 'H_mm', 'J_mm', 'q0')]),
              'Calibration defects (mm): '+str(c.get('defects', {})),
              'Splits: '+', '.join(f'{n}={v["states"]}' for n, v in report['splits'].items())]
    for name, definition in report['definitions'].items():
        lines.append(name+': '+definition)
    for name, model in report['models'].items():
        cond = model.get('conditioning') or {}
        if name == 'G5_design_assisted':
            cond = cond.get('image_stage', {})
        lines.append(f'{name}: coefficients={model.get("parameter_count", model.get("spec", {}).get("parameter_count"))}; '
                     f'train RMS={number(model.get("training_rms_um"))} um; rank={cond.get("rank")}; '
                     f'condition={number(cond.get("condition_number"))}; scope={model.get("evaluation_scope", "NOT COMPUTED")}')
        lines.append(f'{name} coefficients: '+str(safe(model.get('coefficients'))))
        lines.append(f'{name} singular values: '+vector(cond.get('singular_values')))
        for group, m in model.get('groups', {}).items():
            a, b, d, v = (m.get(k) or {} for k in ('outer_um', 'centre_known_um', 'centre_recovered_um', 'unknown_arcmin'))
            checks = m.get('target_checks') or {}
            lines.append(f'{name}/{group}: outer RMS/max {number(a.get("rms"))}/{number(a.get("max"))} um; '
                f'known-centre bias {vector(b.get("signed_bias_xy"))}, RMS/max {number(b.get("rms"))}/{number(b.get("max"))} um '
                f'worst w={vector((b.get("worst") or {}).get("w_deg"))}; recovered-centre bias {vector(d.get("signed_bias_xy"))}, RMS/max '
                f'{number(d.get("rms"))}/{number(d.get("max"))} um worst w={vector((d.get("worst") or {}).get("w_deg"))}')
            lines.append(f'{name}/{group}: gaze RMS/max {number(v.get("rms"))}/{number(v.get("max"))} arcmin; '
                f'failures/tested={v.get("failed_inversions")}/{v.get("tested_states")}, boundary={v.get("boundary_solutions")}; '
                f'worst w={vector((v.get("worst") or {}).get("w_deg"))}; checks={checks}')
            kg, rg = (m.get(k) or {} for k in ('grid_equivalent_known_um', 'grid_equivalent_recovered_um'))
            lines.append(f'{name}/{group}: grid-equivalent inverse 1000*H*(g_hat-g), known-gaze RMS/max '
                f'{number(kg.get("rms"))}/{number(kg.get("max"))} um; recovered-gaze RMS/max '
                f'{number(rg.get("rms"))}/{number(rg.get("max"))} um; distinct from sensor residuals.')
        cross = model.get('mean_centre_cross_check') or {}
        discrepancy = cross.get('discrepancy') or {}
        lines.append(f'{name}: mean/centre composition identity defect={number(cross.get("identity_max_um"))} um; '
                     f'traced-minus-mean vs -b discrepancy RMS/max={number(discrepancy.get("rms"))}/{number(discrepancy.get("max"))} um.')
        diagnostic = model.get('cross_checks') or {}
        homography = (diagnostic.get('homography_after_undo') or {}).get('grid_equivalent_closure_um') or {}
        lines.append(f'{name}: homography closure after undo (grid-equivalent um) RMS/max '
            f'{number(homography.get("rms"))}/{number(homography.get("max"))}; '
            f'keystone+2*prism={vector(diagnostic.get("fitted_keystone_plus_twice_prism"))}; coefficients/checks per state in JSON.')
        if name == 'T0':
            reference = model.get('reproduction_reference') or {}
            measured = reference.get('centre_recomputed_um') or {}
            lines.append('T0 independent reproduction: accepted approximate centre RMS/max=4.89/6.04 um, outer train RMS=3.62 um; '
                f'recomputed centre {number(measured.get("rms"))}/{number(measured.get("max"))} um, outer {number(model.get("training_rms_um"))} um.')
    history = report['selection']['history']
    # JSON contains every decision and both solver attempts, without truncation.
    lines.append(f'Selection trials/decisions: {len(history)}; final blocks={report["selection"].get("final_blocks")}; complete history in JSON.')
    for event in history:
        if event.get('decision') in ('accepted', 'pruned_or_reverted', 'retained'):
            lines.append(f'Selection {event["phase"]}: {event["block"]}: {event["decision"]}; RMS={number(event.get("training_rms_um"))} um.')
    error_summary = []
    def errors(value, prefix=''):
        if isinstance(value, dict):
            for error in value.get('errors', []):
                error_summary.append(prefix+': '+error['stage']+': '+error['error']+
                    (f'; affected states={len(error["ade_bde_indices"])}' if 'ade_bde_indices' in error else ''))
            for key, child in value.items():
                if key not in ('errors', 'base'):
                    errors(child, prefix+'/'+str(key))
        elif isinstance(value, (list, tuple)):
            for child in value:
                errors(child, prefix)
    errors(report)
    lines.append(f'Recorded stage errors: {len(error_summary)}; full details and locations in JSON.')
    lines += error_summary[:max(0, 295-len(lines))]
    lines.append('JSON contains all coefficients, calibration defects, splits, trials, solver statuses, per-state errors and diagnostics.')
    text = '\n'.join(lines)+'\n'
    require(len(text.splitlines()) < 300, 'Report must have fewer than 300 lines')
    return text


def guarded_paths(out, targets):
    expected = ROOT/'runs'/'p4_a0_joint'
    require(out.absolute() == expected and out.resolve() == expected,
            'Unsafe resolved output directory (redirected path)')
    inputs = [(ROOT/DEFAULT).resolve(), Path(__file__).resolve(),
              Path(additive.__file__).resolve(), ROOT/'Script'/'p4_a0_gain.py',
              ROOT/'Script'/'p4_a0_data.py', ROOT/'Script'/'p4_unwarp_accom.py']
    for target in targets:
        require(target.parent == out and target.resolve() == target.absolute(), 'Unsafe resolved output target')
        for source in inputs:
            require(target.resolve() != source.resolve() and not
                    (target.exists() and source.exists() and target.samefile(source)),
                    'Output aliases a read-only input or source')
    for i, left in enumerate(targets):
        for right in targets[i+1:]:
            require(left.resolve() != right.resolve() and not
                    (left.exists() and right.exists() and left.samefile(right)), 'Output-to-output alias')


def minimal_text(report):
    return ('A=0 joint fit\nStatus: '+report['status']+
            '\nReport rendering or a stage failed. Complete available details are in results.json.\n')


def finalize(report, out, targets):
    ok = [False, False]
    # Independent writes; a failed renderer/serializer must not suppress the
    # other artifact. A second pass incorporates first-pass filesystem errors.
    for attempt in range(2):
        report['status'] = 'completed_with_errors' if has_errors(report) else 'completed'
        try:
            text = compact_text(report)
        except Exception as exc:
            stage_error(report, 'render_report', exc)
            report['status'] = 'completed_with_errors'
            text = minimal_text(report)
        try:
            encoded = json.dumps(safe(exportable(report)), indent=2, allow_nan=False)+'\n'
        except Exception as exc:
            stage_error(report, 'serialize_json', exc)
            report['status'] = 'completed_with_errors'
            try:
                text = compact_text(report)
            except Exception:
                text = minimal_text(report)
            encoded = json.dumps(safe(dict(status=report['status'], errors=report['errors'],
                failure='Complete serialization failed')), indent=2, allow_nan=False)+'\n'
        prepared_error_state = has_errors(report)
        for j, payload in enumerate((text, encoded)):
            try:
                guarded_paths(out, targets)
                out.mkdir(parents=True, exist_ok=True)
                guarded_paths(out, targets)
                targets[j].write_text(payload, encoding='utf-8')
                ok[j] = True
            except Exception as exc:
                ok[j] = False
                stage_error(report, 'write_'+targets[j].name, exc)
                print(f'{targets[j].name}: {type(exc).__name__}: {exc}', file=sys.stderr)
        if all(ok) and has_errors(report) == prepared_error_state:
            break
    report['status'] = 'completed_with_errors' if has_errors(report) else 'completed'
    print('A=0 joint fit: '+report['status'])
    return int(not all(ok) or has_errors(report))


def main():
    report = initial_report()  # Exists even when validation/calibration fails.
    out = ROOT/'runs'/'p4_a0_joint'
    targets = [out/'report.txt', out/'results.json']
    try:
        guarded_paths(out, targets)
    except Exception as exc:
        stage_error(report, 'output_safety', exc)
        print(f'Unsafe/denied output paths: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            analyze(report)
    except Exception as exc:
        stage_error(report, 'analysis', exc)
    return finalize(report, out, targets)


if __name__ == '__main__':
    raise SystemExit(main())
