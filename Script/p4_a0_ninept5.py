r"""Usage (PowerShell, from the project root; run only after audit):

& 'C:\Users\rueijrwu\.pyenv-win-venv\envs\venv\Scripts\python.exe' Script\p4_a0_ninept5.py --selftest
& 'C:\Users\rueijrwu\.pyenv-win-venv\envs\venv\Scripts\python.exe' Script\p4_a0_ninept5.py --quick
& 'C:\Users\rueijrwu\.pyenv-win-venv\envs\venv\Scripts\python.exe' Script\p4_a0_ninept5.py --pilot-gates-passed

--input PATH selects one manifest-complete shard; --shard-index N defaults to 0.
Full mode requires --pilot-gates-passed, an operator attestation of the audited
pilot's input/numerical gates (no old results are read). Residual size is not a
pilot numerical gate. One process; outputs replace only the four fixed artifacts
under runs/p4_ninept5. No CODE V, cache, source or input writes.
Noise-free design traces; unaudited. Section 15 of P4_9PT_PLAN.md is the protocol.
"""
import sys
sys.dont_write_bytecode = True

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import stat
import time
import zipfile

import numpy as np
import p4_a0_ninept as nine
import p4_a0_data
import p4_a0_joint_fit as joint
from p4_a0_joint_fit import EPS, FD, COND

ROOT = Path(__file__).absolute().parents[1]
OUT = ROOT / 'runs' / 'p4_ninept5'
DATA = ROOT / 'data' / 'p4_raytrace'
MANIFEST = DATA / 'p4_raytrace_5x5_manifest.json'
ARTIFACTS = ('report.txt', 'results.json', 'maps.npz', 'progress.log')
LABELS = np.array([(x, y) for y in (1, .5, 0, -.5, -1)
                   for x in (-1, -.5, 0, .5, 1)], float)
NODES = -LABELS
NINE = np.array([np.flatnonzero(np.all(LABELS == l, axis=1))[0] for l in nine.LABELS])
EIGHT = NINE[nine.OUTER]
NONCENTRE = np.flatnonzero(np.arange(25) != 12)
SCOPES = dict(all=np.arange(25), noncentre=NONCENTRE, eight=EIGHT)
POWERS = tuple((i, j) for j in range(5) for i in range(5))
V = np.column_stack([NODES[:, 0]**i*NODES[:, 1]**j for i, j in POWERS])
LATTICE = np.array([(x, y) for y in np.linspace(-1, 1, 11)
                    for x in np.linspace(-1, 1, 11)])
BLOCKS = dict(Q=('qxx', 'qxy', 'qyy'), prism=('alpha_x', 'alpha_y'),
              trefoil=('t1', 't2'), beta=('beta',), quartic=('gamma_x', 'gamma_y'),
              lag=('eta',))
RADIUS = np.linalg.norm(NODES, axis=1)
RINGS = dict(inner_axial=np.flatnonzero(RADIUS == .5),
             inner_square=np.flatnonzero(np.max(abs(NODES), axis=1) == .5),
             inner_diagonal=np.flatnonzero(np.all(abs(NODES) == .5, axis=1)),
             outer_boundary=np.flatnonzero(np.max(abs(NODES), axis=1) == 1),
             original_outer=EIGHT)
CODEBOOK = dict(nine.CODEBOOK)
CODEBOOK[12] = 'structurally_unidentifiable'
CODEBOOK[13] = 'sampled_fold_or_postfit_branch'
require = nine.require
REQUIRED_CONFIGURATION_KEYS = (
    'profile', 'workers', 'lens', 'lens_sha256', 'lens_seq_sha256', 'source',
    'source_sha256', 'ray_command', 'cornea_ent_d_thi_mode',
    'cornea_ent_d_thi_target_mm', 'field_setup', 'pilot_axis_choice', 'axes',
    'axes.accommodation_d', 'axes.ade_deg', 'axes.bde_deg', 'axes.field_xy_y_major',
    'state_index_formula', 'ray_order', 'units', 'units.coordinates', 'units.angles',
    'units.accommodation', 'units.direction_cosines', 'units.field_xy',
    'global_reference', 'surface_labels', 'surface_labels.S13', 'surface_labels.S16',
    'raw_direction_signs_preserved', 'failed_ray_policy', 'thresholds',
    'thresholds.readback_deg', 'thresholds.readback_mm', 'thresholds.direct_position_mm',
    'thresholds.direct_direction', 'thresholds.lmn_norm', 'thresholds.stop_origin_mm',
    'thresholds.s13_s16_lmn', 'thresholds.segment_alignment', 'record_count',
)
REQUIRED_SHARD_METADATA_KEYS = REQUIRED_CONFIGURATION_KEYS + (
    'accommodation_index', 'accommodation_d', 'shard_filename', 'shard_count',
)
IMPORTED_HELPERS = (
    'archive/distortion_grid.py', 'archive/distortion_grid_p4.py',
    'archive/distortion_grid_p4_biaxial.py', 'archive/distortion_grid_p4_biaxial_macro.py',
)


def require_manifest_keys(record, keys, context, nullable=()):
    """Check presence before comparison, including nested scientific fields."""
    for key in keys:
        value = record
        for part in key.split('.'):
            require(isinstance(value, dict) and part in value,
                    f'Manifest missing required key {key} ({context})')
            value = value[part]
        allow_null = key in nullable and (key != 'source_sha256' or
                                          ('source' in record and record['source'] is None))
        require(value is not None or allow_null,
                f'Manifest null required key {key} ({context})')


def recorded_provenance_status(recorded, current, context):
    """Mirror resume helper gates using records only; never read helper files."""
    helpers = recorded.get('helper_sha256', {})
    current_helpers = current.get('helper_sha256', {})
    require(isinstance(helpers, dict) and isinstance(current_helpers, dict),
            'Invalid helper_sha256 provenance')
    matched, notes = True, []
    for name in IMPORTED_HELPERS:
        for hashes in (helpers, current_helpers):
            if name in hashes:
                require(hashes[name] is not None, 'Manifest null required helper hash '+name)
        if name not in helpers:
            status = 'lacks imported-helper hash; accepting legacy omission (informational)'
        elif name not in current_helpers:
            status = 'manifest current hash absent (informational)'
        elif helpers[name] != current_helpers[name]:
            status, matched = 'MISMATCH', False
        else:
            status = 'match'
        notes.append(f'{name}: {status}')
    references = {n: h for n, h in helpers.items() if n not in IMPORTED_HELPERS}
    current_references = {n: h for n, h in current_helpers.items() if n not in IMPORTED_HELPERS}
    for record, hashes in ((recorded, references), (current, current_references)):
        extra = record.get('reference_sha256', {})
        require(isinstance(extra, dict), 'Invalid reference_sha256 provenance')
        hashes.update(extra)
    differences = [f'{n}: recorded={references.get(n)}, manifest={current_references.get(n)}'
                   for n in sorted(set(references) | set(current_references))
                   if references.get(n) != current_references.get(n)]
    note = (f"{context} script_sha256={recorded.get('script_sha256')}, "
            f"manifest current script_sha256={current.get('script_sha256')} (informational); "
            f"helper hash status: {'; '.join(notes)}; "
            f"reference hash differences (informational): {'; '.join(differences) or 'none'}")
    return matched, note


def family(blocks=(), placement='post'):
    blocks = tuple(blocks)
    return dict(blocks=blocks, placement=placement,
                names=tuple(nine.PARAMETERS)+tuple(n for b in blocks for n in BLOCKS[b]))


BASE = family()
PASS_DIAGNOSTICS = ('objective_mm2', 'gradient', 'rank', 'condition',
                    'jacobian_disagreement', 'damping', 'relative_step',
                    'relative_objective_change', 'effective_p', 'eta_frozen',
                    'frozen_gradient')


def allow_eight(spec):
    # Full column rank is additionally checked before any solver step.
    return len(spec['names']) < 16


def guard_paths(input_path=None):
    """Reject links, reparse points, hard links and aliases before every write."""
    require(ROOT == joint.ROOT and ROOT.resolve() == ROOT, 'Redirected project root')
    targets = [OUT / n for n in ARTIFACTS]
    for path in [OUT, *OUT.parents, *targets]:
        require(path.absolute() == path.resolve(), 'Redirected output: '+str(path))
        if path.exists() or path.is_symlink():
            info = path.lstat()
            require(not path.is_symlink() and not
                    (getattr(info, 'st_file_attributes', 0) & 0x400), 'Output reparse point')
            if path in targets:
                require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
                        'Output must be a regular file with one link')
            elif path == OUT:
                require(path.is_dir(), 'Output directory is a file')
    sources = [MANIFEST, ROOT/'P4_9PT_PLAN.md', *list((ROOT/'Script').glob('*.py'))]
    if input_path is not None:
        sources.append(input_path)
    for i, target in enumerate(targets):
        for source in [*sources, *targets[i+1:]]:
            require(target.resolve() != source.resolve() and not
                    (target.exists() and source.exists() and target.samefile(source)),
                    'Output aliases source/input/target')
    return targets


class Progress:
    def __init__(self, input_path):
        self.input = input_path
        guard_paths(input_path)
        OUT.mkdir(parents=True, exist_ok=True)
        guard_paths(input_path)
        (OUT/'progress.log').write_text('', encoding='utf-8')
        self.started = self.last = time.perf_counter()

    def __call__(self, message, heartbeat=False):
        now = time.perf_counter()
        if heartbeat and now-self.last < 55:
            return
        self.last = now
        line = datetime.now(timezone.utc).isoformat(timespec='seconds')
        line += f' elapsed_s={now-self.started:.3f} '+message
        print(line, flush=True)
        guard_paths(self.input)
        with (OUT/'progress.log').open('a', encoding='utf-8') as stream:
            stream.write(line+'\n')
            stream.flush()


class NumericMaps(dict):
    """Stream maps directly into the sole NPZ artifact; retain only small IDs.

    Each key is written once. No observation cache, per-state Jacobian, object
    array, temporary archive or second output file is created. Finalization
    closes the ZIP central directory even after an analysis exception.
    """
    KEEP = {'global_state_id', 'global_ade_index', 'global_bde_index', 'w_deg', 'u'}

    def __init__(self, input_path, log):
        super().__init__()
        self.input, self.log = input_path, log
        self.seen, self.archive, self.stream = set(), None, None

    def __setitem__(self, key, value):
        array = np.asarray(value)
        require(array.dtype.kind in 'biufc', 'NPZ numeric arrays only: '+key)
        require(key not in self.seen and '/' not in key and '\\' not in key, 'Repeated/unsafe NPZ key: '+key)
        self.open()
        guard_paths(self.input)
        with self.archive.open(key+'.npy', 'w', force_zip64=True) as member:
            np.lib.format.write_array(member, array, allow_pickle=False)
        self.stream.flush()
        self.seen.add(key)
        if key in self.KEEP:
            super().__setitem__(key, array)
        self.log('map written '+key, heartbeat=True)

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def open(self):
        if self.archive is None:
            guard_paths(self.input)
            self.stream = (OUT/'maps.npz').open('wb')
            self.archive = zipfile.ZipFile(self.stream, 'w', compression=zipfile.ZIP_DEFLATED, allowZip64=True)

    def close(self):
        self.open()  # Empty failure runs still replace the previous maps artifact.
        guard_paths(self.input)
        try:
            self.archive.close()
        finally:
            self.stream.close()


def read_manifest_snapshot():
    """Accept two consecutive equal parsed reads, at most five reads total."""
    previous = None
    for attempt in range(5):
        if attempt:
            time.sleep(.5)
        try:
            current = json.loads(MANIFEST.read_text(encoding='utf-8-sig'))
        except (OSError, UnicodeError, json.JSONDecodeError):
            previous = None
            continue
        if previous is not None and current == previous:
            return current
        previous = current
    raise ValueError('Incomplete: manifest did not stabilize after five reads')


def manifest_entry(document, profile, filename, index):
    for e in document['shards']:
        require_manifest_keys(e, ('profile', 'filename', 'accommodation_index'), 'shard entry')
    entries = [e for e in document['shards'] if e['profile'] == profile
               and e['filename'] == filename and e['accommodation_index'] == index]
    require(len(entries) == 1, 'Incomplete: no unique manifest entry')
    e = entries[0]
    require_manifest_keys(e, ('profile', 'filename', 'accommodation_index', 'accommodation_d',
                             'completion_status', 'expected_row_count', 'row_count', 'sha256'),
                          'selected shard entry')
    expected = 625 if profile == 'pilot5' else 255025
    require(e['completion_status'] == 'complete' and
            e['expected_row_count'] == e['row_count'] == expected and
            isinstance(e['sha256'], str) and len(e['sha256']) == 64,
            'Incomplete: manifest completion/count/hash gate')
    return e


def digest_stream(stream, log):
    digest = hashlib.sha256()
    stream.seek(0)
    for block in iter(lambda: stream.read(1 << 20), b''):
        digest.update(block)
        log('hash streaming', heartbeat=True)
    return digest.hexdigest(), stream.tell()


def inverse(points, c, H):
    def one_batch(x):
        z, status = joint.camera_inverse(x-c)
        require(np.all(status['converged']) and np.all(status['min_tangential'] > 0)
                and np.all(status['min_radial'] > 0), 'Inverse branch/status')
        require(np.all(1000*np.linalg.norm(joint.camera(z)-(x-c), axis=-1) < 1e-9),
                'Camera closure >=1e-9 um')
        return z/H
    return nine.checked_batch(one_batch, points, points.shape[1:])


def load_shard(args, report, maps, log):
    profile = 'pilot5' if args.quick else 'full5'
    path = args.input_path
    # Only the exact published filename in the manifest directory is permissible.
    # This rejects legacy, temporary and aliased inputs before pickle.load.
    require(path.parent == DATA and path.resolve() == path and not path.is_symlink(),
            'Input must be an unredirected published shard in data/p4_raytrace')
    document = read_manifest_snapshot()
    require_manifest_keys(document, ('schema_version', 'profiles', 'shards'), 'manifest')
    require(document['schema_version'] == 1 and
            isinstance(document['profiles'], dict) and isinstance(document['shards'], list),
            'Unrecognized shard manifest schema')
    entry = manifest_entry(document, profile, path.name, args.shard_index)
    require_manifest_keys(document, (f'profiles.{profile}.configuration',
                                     f'profiles.{profile}.provenance'), 'manifest')
    configuration = document['profiles'][profile]['configuration']
    provenance = document['profiles'][profile]['provenance']
    require(isinstance(configuration, dict) and isinstance(provenance, dict),
            'Invalid manifest configuration/provenance')
    # These fields explicitly represent absent optional inputs in published records.
    nullable = ('source', 'source_sha256') + (() if args.quick else ('pilot_axis_choice',))
    require_manifest_keys(configuration, REQUIRED_CONFIGURATION_KEYS, 'configuration', nullable)
    ignored = {'run_id', 'created_utc', 'worker_summaries', 'script_sha256', 'helper_sha256',
               'reference_sha256', 'accommodation_index', 'accommodation_d', 'shard_filename',
               'shard_count', 'source_comparison', 'record_count'}
    configuration_keys = tuple(k for k in configuration if k not in ignored)
    provenance_keys = tuple(k for k in provenance if k not in ignored)
    require_manifest_keys(configuration, configuration_keys, 'configuration', nullable)
    require_manifest_keys(provenance, provenance_keys, 'provenance', nullable)
    provenance_policy = dict(
        gated=dict(configuration=tuple(dict.fromkeys(REQUIRED_CONFIGURATION_KEYS + configuration_keys)),
                   provenance=provenance_keys, shard_metadata=REQUIRED_SHARD_METADATA_KEYS,
                   manifest=('schema_version', 'completion_status', 'expected_row_count',
                             'row_count', 'sha256', 'profile', 'filename',
                             'accommodation_index', 'accommodation_d'),
                   imported_helper_sha256=IMPORTED_HELPERS,
                   helper_rule='Configuration and shard hashes must equal manifest provenance when both recorded'),
        informational=('script_sha256', 'reference_sha256',
                       'helper_sha256 entries outside the four imported helpers',
                       'absent imported-helper hashes (legacy omission)',
                       'run_id', 'created_utc', 'worker_summaries', 'source_comparison'),
        nullable=nullable,
        source_hash_rule='source_sha256 may be null only when source is null',
        record_count_rule='Configuration total equals shard_count * manifest row_count; shard record_count equals manifest row_count')
    report['input'] = dict(path=str(path), manifest_entry=entry, profile=profile,
                           provenance=provenance, provenance_policy=provenance_policy, complete=False)
    t = time.perf_counter()
    before = path.stat()
    require(not (getattr(before, 'st_file_attributes', 0) & 0x400), 'Input reparse point')
    with path.open('rb') as stream:
        hashed, size = digest_stream(stream, log)
        require(hashed == entry['sha256'], 'Shard hash mismatch before deserialization')
        report['input'].update(sha256=hashed, bytes=size)
        report['timings']['hash_s'] = time.perf_counter()-t
        log(f'load hash complete bytes={size}')
        t = time.perf_counter()
        stream.seek(0)
        rows = pickle.load(stream)  # Exactly one accommodation, exactly once.
        require(stream.read(1) == b'', 'Trailing shard payload')
        report['timings']['deserialize_s'] = time.perf_counter()-t
        after_hash, after_size = digest_stream(stream, log)
    after = path.stat()
    # Keep the accepted manifest snapshot; other shards may publish during load.
    require(after_hash == hashed and after_size == size and
            (before.st_size, before.st_mtime_ns, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_ino), 'Input changed during load')
    require(isinstance(rows, np.ndarray) and rows.ndim == 1 and
            rows.dtype.names == tuple(p4_a0_data.EXPECTED) and
            rows.dtype.itemsize == 8*len(p4_a0_data.EXPECTED) and
            all(rows.dtype[n] == np.dtype('float64') and rows.dtype.fields[n][1] == 8*j
                for j, n in enumerate(rows.dtype.names)),
            'Flat structured float64 schema required')
    meta = (rows.dtype.metadata or {}).get('p4_retrace', {})
    require_manifest_keys(meta, REQUIRED_SHARD_METADATA_KEYS, 'shard metadata', nullable)
    require_manifest_keys(meta, configuration_keys + provenance_keys, 'shard metadata', nullable)
    configuration_helpers_match, configuration_note = recorded_provenance_status(
        configuration, provenance, 'manifest configuration')
    shard_helpers_match, shard_note = recorded_provenance_status(meta, provenance, 'shard metadata')
    provenance_note = 'provenance note: '+configuration_note+'; '+shard_note
    report['input']['provenance_note'] = provenance_note
    log(provenance_note)
    require(configuration_helpers_match and shard_helpers_match, 'Imported-helper hash mismatch')
    require(len(rows) == entry['row_count'] == meta['record_count'] and
            meta['shard_filename'] == path.name and meta['profile'] == profile and
            meta['accommodation_index'] == args.shard_index and
            meta['accommodation_d'] == entry['accommodation_d'], 'Shard metadata/count')
    # Scientific values compare only after explicit presence/null checks above.
    for key in configuration_keys:  # All-accommodation count is checked separately below.
        require(meta[key] == configuration[key], 'Manifest configuration mismatch: '+key)
    for key in provenance_keys:
        require(meta[key] == provenance[key], 'Manifest provenance mismatch: '+key)
    axes = meta['axes']
    aa, bb = np.asarray(axes['ade_deg']), np.asarray(axes['bde_deg'])
    ac = np.asarray(axes['accommodation_d'])
    require(np.array_equal(axes['field_xy_y_major'], LABELS) and
            0 <= args.shard_index < len(ac) and ac[args.shard_index] == entry['accommodation_d'],
            'Axes/accommodation mismatch')
    require(meta['shard_count'] == len(ac) and
            configuration['record_count'] == len(ac)*entry['row_count'], 'Manifest all-accommodation count')
    expected_axis = np.array([-12., -6., 0., 6., 12.]) if args.quick else np.linspace(-12.5, 12.5, 101)
    require(np.array_equal(aa, expected_axis) and np.array_equal(bb, expected_axis), 'Command axes')
    na, nb = len(aa), len(bb)
    require(rows.shape == (na*nb*25,), 'State coverage/count')
    rays = rows.reshape(na*nb, 25)
    ai, bi = np.indices((na, nb)).reshape(2, -1)
    source = ((args.shard_index*na)+ai)*nb+bi
    expected = dict(ray_index=np.broadcast_to(np.arange(25), rays.shape),
                    state_index=source[:, None], accommodation_index=args.shard_index,
                    accommodation_d=entry['accommodation_d'], ade_index=ai[:, None],
                    bde_index=bi[:, None], ade_deg=aa[ai, None], bde_deg=bb[bi, None],
                    field_x_relative=LABELS[None, :, 0], field_y_relative=LABELS[None, :, 1])
    for key, value in expected.items():
        require(np.all(rays[key] == value), 'Schema/order/command mismatch: '+key)
    bad = np.zeros(na*nb, bool)
    geometry_reasons = {}
    for i, state in enumerate(rays):
        try:
            require(np.all(state['rayrsi_status'] == 0) and
                    np.all(state['center_rayrsi_status'] == 0), 'Nonzero ray status')
            joint.validate_a0_geometry(state)
            require(np.all(abs(state['rc_ade_readback_deg']-state['ade_deg']) <= 1e-7) and
                    np.all(abs(state['rc_bde_readback_deg']-state['bde_deg']) <= 1e-7), 'Angle readback')
            require(np.all(abs(state['cornea_ent_d_thi_readback_mm']) <= 1e-8), 'Nonzero thickness')
        except Exception as exc:
            bad[i] = True
            reason = str(exc)
            geometry_reasons[reason] = geometry_reasons.get(reason, 0)+1
        log(f'geometry states={i+1}/{len(rays)} failures={bad[:i+1].sum()}', heartbeat=True)
    zero = (aa[ai] == 0) & (bb[bi] == 0)
    require(zero.sum() == 1 and not bad[zero][0], 'Failed/missing calibration reference')
    plane = rays['zsi'][zero, 0][0]
    bad |= ~np.all(np.isfinite(rays['zsi']) & (abs(rays['zsi']-plane) <= 1e-9), axis=1)
    require(not bad[zero][0], 'Failed calibration sensor-plane reference')
    nine_rows = rays[:, NINE]
    if args.quick:
        adapter = np.empty((101, 101, 9), dtype=rows.dtype)
        for key in adapter.dtype.names:
            adapter[key] = np.nan
        adapter[50, 50] = nine_rows[zero][0]
    else:
        adapter = nine_rows.reshape(101, 101, 9)
    _, _, c, H, q = joint.calibrate(adapter, nine.LABELS, report['calibration'])
    if not args.quick:
        w, splits = joint.additive.partition(adapter)
    del adapter
    w = np.column_stack((bb[bi], -aa[ai]))
    ga = (aa[ai]+12.5)/.25
    gb = (bb[bi]+12.5)/.25
    require(np.array_equal(ga, np.rint(ga)) and np.array_equal(gb, np.rint(gb)), 'Global angle indices')
    ga, gb = ga.astype(int), gb.astype(int)
    ids = ga*101+gb
    groups = dict(overall=np.ones(len(rays), bool), train=(ga % 2 == 0) & (gb % 2 == 0) & ~zero,
                  even_odd=(ga % 2 == 0) & (gb % 2 == 1),
                  odd_even=(ga % 2 == 1) & (gb % 2 == 0), odd_odd=(ga % 2 == 1) & (gb % 2 == 1),
                  zero=zero)
    if not args.quick:
        for name, members in splits.items():
            require(np.array_equal(np.flatnonzero(groups[name]), members), 'Split adapter disagreement')
    radius = np.linalg.norm(w, axis=1)
    groups.update(bin_0_4=radius < 4, bin_4_8=(radius >= 4) & (radius < 8),
                  bin_8_12p5=(radius >= 8) & (radius <= 12.5), bin_corners=radius > 12.5)
    for a, b in ((0, 0), (0, 100), (100, 0), (100, 100)):
        groups[f'extreme_{a}_{b}'] = (ga == a) & (gb == b)
    observed = np.stack((rays['xsi'], rays['ysi']), axis=-1)
    observed[bad] = np.nan  # A copy; the structured input is never changed.
    maps.update(source_state_id=source, local_row_id=np.arange(len(rays)), global_state_id=ids,
                global_ade_index=ga, global_bde_index=gb, w_deg=w, u=w/12.5, input_failed=bad)
    maps.update({'group_'+k: v for k, v in groups.items()})
    report['input'] = dict(path=str(path), sha256=hashed, bytes=size, manifest_entry=entry,
                           profile=profile, provenance=provenance, provenance_note=provenance_note,
                           provenance_policy=provenance_policy,
                           geometry_reasons=geometry_reasons,
                           input_failures=int(bad.sum()), states=len(rays), complete=not args.quick)
    log(f'load complete states={len(rays)} input_failures={bad.sum()}')
    return observed, c, H, q, groups


def coefficients(z):
    n = len(z)
    C = np.full((n, 50), np.nan)
    good = np.all(np.isfinite(z), axis=(1, 2))
    if np.any(good):
        rhs = z[good].transpose(1, 0, 2).reshape(25, -1)
        solved = np.linalg.solve(V, rhs).reshape(25, -1, 2).transpose(1, 2, 0)
        C[good] = solved.reshape(-1, 50)
    defect = np.max(abs(np.einsum('nj,scj->snc', V, C.reshape(n, 2, 25))-z), axis=(1, 2))
    return C, good & (defect < 1e-12), defect


def extras(p, spec):
    return {name: p[:, j, None] for j, name in enumerate(spec['names']) if j >= 8}


def barrel(v, p, spec, q):
    e = extras(p, spec)
    alpha = np.stack((e.get('alpha_x', np.zeros((len(p), 1))),
                      e.get('alpha_y', np.zeros((len(p), 1)))), axis=-1)
    lag = e.get('eta', 0)*p[:, :2]
    if np.isscalar(lag):
        lag = np.zeros((len(p), 2))
    lag = lag[:, None, :]

    def prism(x):
        return alpha*np.sum(x*x, axis=-1)[..., None]

    def core(x):
        z = joint.radial(x, q, e.get('beta', 0))
        xx, yy = x[..., 0], x[..., 1]
        tensor = e.get('qxx', 0)*xx*xx+2*e.get('qxy', 0)*xx*yy+e.get('qyy', 0)*yy*yy
        return z+x*np.asarray(tensor)[..., None]

    if spec['placement'] == 'pre':
        z = core(v+prism(v)-lag)-core(-lag)
    elif spec['placement'] == 'middle':
        z = core(v-lag+prism(v-lag))-core(-lag+prism(-lag))
    else:
        z = core(v-lag)-core(-lag)+prism(v)
    x, y = v[..., 0], v[..., 1]
    hx, hy = x**3-3*x*y*y, 3*x*x*y-y**3
    z += np.stack((e.get('t1', 0)*hx-e.get('t2', 0)*hy,
                   e.get('t1', 0)*hy+e.get('t2', 0)*hx), axis=-1)
    z += np.stack((e.get('gamma_x', 0)*x*x*y*y, e.get('gamma_y', 0)*x*x*y*y), axis=-1)
    return z


def template_bound(p, spec, q, H, order):
    """Coefficient-l1 majorants under interval polynomial composition on [-1,1]^2.

    Addition sums absolute coefficient bounds; multiplication multiplies them.
    Cancellation is deliberately never credited. This covers lag, both prism
    compositions, Q, beta, trefoil, W and the fixed polynomial camera.
    """
    e = {k: abs(v[:, 0]) for k, v in extras(p, spec).items()}
    zeros = np.zeros(len(p))
    alpha = np.maximum(e.get('alpha_x', zeros), e.get('alpha_y', zeros))
    lag = e.get('eta', zeros)*np.max(abs(p[:, :2]), axis=1)
    def prism_bound(t):
        return 2*alpha*t*t
    def core_bound(t):
        r2 = 2*t*t
        return t*(1+abs(q)*r2+e.get('beta', zeros)*r2*r2+
                  (e.get('qxx', zeros)+2*e.get('qxy', zeros)+e.get('qyy', zeros))*t*t)
    if spec['placement'] == 'pre':
        bound = core_bound(1+prism_bound(1)+lag)+core_bound(lag)
    elif spec['placement'] == 'middle':
        bound = core_bound(1+lag+prism_bound(1+lag))+core_bound(lag+prism_bound(lag))
    else:
        bound = core_bound(1+lag)+core_bound(lag)+prism_bound(1)
    bound += 4*(e.get('t1', zeros)+e.get('t2', zeros))
    bound += np.maximum(e.get('gamma_x', zeros), e.get('gamma_y', zeros))
    if order == 'C':
        r2 = 2*(H*bound)**2
        bound *= 1+abs(joint.K1)/joint.F**2*r2+abs(joint.K2)/joint.F**4*r2*r2
    return bound


def forward_unchecked(p, order, c, H, q, spec=BASE, nodes=NODES):
    if not spec['blocks']:
        return nine.forward_unchecked(p, order, c, H, q, nodes=nodes)
    require(np.all(np.isfinite(p)), 'Nonfinite parameters')
    L = np.stack((1+p[:, 2], p[:, 3], p[:, 4], 1+p[:, 5]), -1).reshape(-1, 2, 2)
    singular = np.linalg.svd(L, compute_uv=False)
    require(np.all(np.linalg.det(L) > 0) and np.all(singular[:, -1] > 1e-10)
            and np.all(singular[:, 0]/singular[:, -1] <= COND), 'Invalid L')
    bound = np.ones(len(p)) if order == 'A' else template_bound(p, spec, q, H, order)
    require(np.all(1-bound*np.sum(abs(p[:, 6:8]), axis=1) > 1e-10), 'Extended denominator bound')
    d = np.broadcast_to(nodes, (len(p), len(nodes), 2))
    u = np.ones((len(p), 2))
    tilt = lambda v: joint.tilt(v, u, L, p[:, 6:8], {}, nine.CONFIG)
    if order == 'A':
        return c+joint.camera(H*(barrel(tilt(d), p, spec, q)+p[:, None, :2]))
    b = barrel(d, p, spec, q)
    if order == 'B':
        return c+joint.camera(H*(tilt(b)+p[:, None, :2]))
    require(order == 'C', 'Unknown order')
    return c+H*(tilt(joint.camera(H*b)/H)+p[:, None, :2])


def predict(p, order, c, H, q, spec, nodes):
    return nine.checked_batch(lambda x: forward_unchecked(x, order, c, H, q, spec, nodes),
                              p, (len(nodes), 2))


def affine_seed(observed, nodes, order, c, H, q, spec):
    """The caller has already removed every excluded target."""
    def calculate(points):
        z, valid = inverse(points, c, H)
        require(np.all(valid), 'Affine inverse invalid')
        template = nodes if order == 'A' else joint.radial(nodes, q)
        target = z
        if order == 'C':
            template, target = joint.camera(H*template)/H, (points-c)/H
        design = np.column_stack((np.ones(len(nodes)), template))
        coef, _, rank, _ = np.linalg.lstsq(design, target.transpose(1, 0, 2).reshape(len(nodes), -1), rcond=None)
        require(rank == 3, 'Affine design rank')
        coef = coef.reshape(3, len(points), 2).transpose(1, 0, 2)
        p = np.zeros((len(points), len(spec['names'])))
        p[:, :2] = coef[:, 0]
        L = coef[:, 1:].transpose(0, 2, 1)
        p[:, 2], p[:, 3], p[:, 4], p[:, 5] = L[:, 0, 0]-1, L[:, 0, 1], L[:, 1, 0], L[:, 1, 1]-1
        forward_unchecked(p, order, c, H, q, spec, nodes)
        return p
    return nine.checked_batch(calculate, observed, (len(spec['names']),))


def derivative(p, function, active):
    h = FD*np.maximum(1., abs(p[active]))
    shifts = np.zeros((4*len(active), len(p)))
    for j, col in enumerate(active):
        shifts[4*j:4*j+4, col] = h[j]*np.array([1, -1, .5, -.5])
    values, good = nine.checked_batch(function, p[None]+shifts, function.tail)
    require(np.all(good), 'Invalid finite-difference branch')
    a, b = [], []
    for j in range(len(active)):
        f = values[4*j:4*j+4].reshape(4, -1)
        a.append((f[0]-f[1])/(2*h[j]))
        b.append((f[2]-f[3])/h[j])
    a, b = np.stack(a, -1), np.stack(b, -1)
    diff = np.linalg.norm(a-b)/max(np.linalg.norm(a), np.linalg.norm(b), EPS)
    require(np.isfinite(diff) and diff <= 5e-5, 'Jacobian half-step disagreement')
    return b, diff


def fold_valid(p, order, H, q, spec):
    if not spec['blocks']:
        return True
    # A's barrel input is T(d), so sample the composed stage on the square.
    def stage(nodes):
        v = np.broadcast_to(nodes, (1, len(nodes), 2))
        if order == 'A':
            L = np.array([[[1+p[2], p[3]], [p[4], 1+p[5]]]])
            v = joint.tilt(v, np.ones((1, 2)), L, p[None, 6:8], {}, nine.CONFIG)
        return barrel(v, p[None], spec, q)[0]
    h = FD/2
    dx = (stage(LATTICE+[h, 0])-stage(LATTICE-[h, 0]))/(2*h)
    dy = (stage(LATTICE+[0, h])-stage(LATTICE-[0, h]))/(2*h)
    det = dx[:, 0]*dy[:, 1]-dx[:, 1]*dy[:, 0]
    return bool(np.all(np.isfinite(det)) and np.all(det > 0))


def solve_pass(target, nodes, order, c, H, q, spec, seed, budget, limit, log, label,
               restricted=False, condition_limit=COND):
    """Generic per-state driver; all imported exceptions remain state-local."""
    p = np.array(seed, float, copy=True)
    r = dict(parameters=p, reason=0, failed=True, iterations=0, objective_mm2=np.inf,
             gradient=np.nan, rank=0, condition=np.inf, jacobian_disagreement=np.nan,
             damping=np.nan, relative_step=np.inf, relative_objective_change=np.inf,
             effective_p=len(p), eta_frozen=False, frozen_gradient=np.nan)
    if not np.all(np.isfinite(target)):
        r['reason'] = 8
        return r
    eta = spec['names'].index('eta') if 'eta' in spec['names'] else None
    if restricted:
        p[6:8] = 0
    def function(x):
        return forward_unchecked(x, order, c, H, q, spec, nodes)
    function.tail = (len(nodes), 2)
    last_step = last_change = np.inf
    damping = None
    for iteration in range(budget+1):
        r['iterations'] = iteration
        log(label+f' iteration={iteration}', heartbeat=True)
        # Structural degeneracy belongs to the current iterate, not its seed.
        r['eta_frozen'] = eta is not None and np.max(abs(p[:2])) <= 1e-14
        if r['eta_frozen']:
            p[eta] = 0
        active = np.array([j for j in range(len(p)) if not
                           (restricted and j in (6, 7)) and not (r['eta_frozen'] and j == eta)])
        r.update(effective_p=len(active), gradient=np.nan, rank=0, condition=np.inf,
                 jacobian_disagreement=np.nan, frozen_gradient=np.nan)
        try:
            pred = function(p[None])[0]
            residual = (pred-target).ravel()
            obj = float(residual@residual)
            require(np.isfinite(obj), 'Nonfinite objective')
            r['objective_mm2'] = obj
        except Exception:
            r['reason'] = 3
            break
        try:
            J, diff = derivative(p, function, active)
            r['jacobian_disagreement'] = diff
        except Exception:
            r['reason'] = 4
            break
        try:
            U, s, vt, scales, info = joint.scaled_svd(J)
            r.update(rank=info['rank'], condition=info['condition_number'])
            r['gradient'] = float(np.max(abs((J/scales).T@residual))/
                                  max(1., s[0]*np.linalg.norm(residual)))
            if not info['full_rank'] or info['condition_number'] > condition_limit:
                r['reason'] = 5
                break
        except (ValueError, np.linalg.LinAlgError):
            # Retain the actual rank even when scaled_svd refuses a zero
            # column. This is diagnostic only; no reduced-basis step is taken.
            try:
                norms = np.linalg.norm(J, axis=0)
                normalized = J/np.where(norms > 0, norms, 1.)
                singular = np.linalg.svd(normalized, compute_uv=False)
                rank = int(np.sum(singular > singular[0]*max(J.shape)*EPS)) if len(singular) else 0
                r['rank'] = rank
                r['condition'] = singular[0]/singular[-1] if rank == len(active) else np.inf
                r['gradient'] = float(np.max(abs(normalized.T@residual))/
                                      max(1., singular[0]*np.linalg.norm(residual)))
            except Exception:
                pass
            r['reason'] = 5
            break
        utr = U.T@residual
        gradient = float(np.max(abs(vt.T@(s*utr)))/max(1., s[0]*np.linalg.norm(residual)))
        r['gradient'] = gradient
        if gradient <= limit:
            r['reason'] = 2 if last_step <= 1e-12 and last_change <= 1e-14 else 1
            break
        if iteration == budget:
            r['reason'] = 7
            break
        lam = max(1e-10*s[0]**2 if damping is None else damping, EPS*s[0]**2)
        accepted = False
        r['iterations'] = iteration+1
        for _ in range(18):
            delta = -(vt.T@((s/(s*s+lam))*utr))/scales
            trial = p.copy()
            trial[active] += delta
            try:
                residual_trial = (function(trial[None])[0]-target).ravel()
                trial_obj = float(residual_trial@residual_trial)
                accepted = np.isfinite(trial_obj) and trial_obj < obj
            except Exception:
                accepted = False
            if accepted:
                last_step = np.linalg.norm(scales*delta)/max(1., np.linalg.norm(scales*p[active]))
                last_change = abs(obj-trial_obj)/max(obj, EPS)
                p = trial
                r['objective_mm2'] = trial_obj
                damping = max(lam/10, EPS*s[0]**2)
                break
            lam *= 10
        r['damping'] = damping if accepted else lam
        if not accepted:
            r['reason'] = 6
            break
    # A reduced-basis gradient cannot certify an unconstrained solution.
    if r['reason'] in (1, 2):
        frozen = np.setdiff1d(np.arange(len(p)), active)
        r['frozen_gradient'] = 0.
        if len(frozen):
            try:
                Jf, diff = derivative(p, function, frozen)
                norms = np.linalg.norm(Jf, axis=0)
                scaled = Jf/np.where(norms > 0, norms, 1.)
                full_scaled = np.column_stack((J/scales, scaled))
                sigma_max = np.linalg.svd(full_scaled, compute_uv=False)[0]
                frozen_gradient = float(np.max(abs(scaled.T@residual))/
                                        max(1., sigma_max*np.linalg.norm(residual)))
                r['frozen_gradient'] = frozen_gradient
                r['jacobian_disagreement'] = max(r['jacobian_disagreement'], diff)
                r['gradient'] = max(r['gradient'], frozen_gradient)
                if not np.isfinite(frozen_gradient) or frozen_gradient > limit:
                    r['reason'] = 6
            except Exception:
                r.update(reason=4, frozen_gradient=np.nan)
    r.update(parameters=p, failed=r['reason'] not in (1, 2),
             relative_step=last_step, relative_objective_change=last_change)
    return r


def attempt(target, nodes, order, c, H, q, spec, seed, budget, limit, log, label,
            continuation=True, condition_limit=COND):
    first = solve_pass(target, nodes, order, c, H, q, spec, seed, budget, limit, log, label,
                       condition_limit=condition_limit)
    first.update(warm_reason=first['reason'], warm_iterations=first['iterations'],
                 restricted_reason=0, restricted_iterations=0, release_reason=0,
                 release_iterations=0, best_iterate_stage=1,
                 start_path=3 if first['failed'] else 1)
    # With no continuation, the first pass is also the last unrestricted pass.
    for key in PASS_DIAGNOSTICS:
        first['best_iterate_'+key] = first[key]
        first['final_release_'+key] = first[key]
    first['final_release_reason'] = first['reason']
    first['final_release_failed'] = first['failed']
    if not continuation or first['reason'] != 6 or budget-first['iterations'] < 1:
        return first
    remaining = budget-first['iterations']
    restricted_seed = seed.copy()
    restricted_seed[6:8] = 0
    short = solve_pass(target, nodes, order, c, H, q, spec, restricted_seed,
                       min(20, max(0, remaining-1)), limit, log, label, restricted=True,
                       condition_limit=condition_limit)
    release = solve_pass(target, nodes, order, c, H, q, spec, short['parameters'],
                         remaining-short['iterations'], limit, log, label,
                         condition_limit=condition_limit)
    best = first.copy()
    for stage, value in ((2, short), (3, release)):
        if value['objective_mm2'] < best['objective_mm2'] or (stage == 3 and not value['failed']):
            for key in value:
                if key not in ('reason', 'failed', 'iterations'):
                    best[key] = value[key]
            best['best_iterate_stage'] = stage
    for key in PASS_DIAGNOSTICS:
        best['best_iterate_'+key] = best[key]
        best['final_release_'+key] = release[key]
    best['final_release_reason'] = release['reason']
    best['final_release_failed'] = release['failed']
    best.update(reason=release['reason'], failed=release['failed'],
                iterations=first['iterations']+short['iterations']+release['iterations'],
                restricted_reason=short['reason'], restricted_iterations=short['iterations'],
                release_reason=release['reason'], release_iterations=release['iterations'],
                start_path=4 if release['failed'] else 2)
    return best


def expand_parent(parameters, parent, spec):
    result = np.zeros((len(parameters), len(spec['names'])))
    for j, name in enumerate(parent['names']):
        result[:, spec['names'].index(name)] = parameters[:, j]
    return result


def failure_detail(state):
    """Describe the retained iterate and the last unrestricted solver pass."""
    reason = int(state['reason'])
    path = int(state.get('start_path', 0))
    fields = [f'reason={reason}({CODEBOOK[reason]})',
              f'iterations={int(state["iterations"])}',
              f'gradient={state["gradient"]:.17g}',
              f'rank={int(state["rank"])}', f'condition={state["condition"]:.17g}',
              f'start_path={path}({nine.START_PATHS.get(path, "unrecorded")})']
    for prefix in ('best_iterate_', 'final_release_'):
        for key in ('gradient', 'rank', 'condition'):
            fields.append(f'{prefix}{key}={state.get(prefix+key, state[key]):.17g}')
    release = int(state.get('final_release_reason', reason))
    fields.append(f'final_release_reason={release}({CODEBOOK[release]})')
    if path in (5, 6):
        fields.append('attempt_selected=identity')
    elif 'attempt_selected' in state:
        fields.append('attempt_selected='+{1: 'parent', 2: 'affine'}[int(state['attempt_selected'])])
    for stage in ('warm', 'restricted', 'release'):
        if stage+'_reason' in state:
            code = int(state[stage+'_reason'])
            fields.append(f'{stage}_reason={code}({CODEBOOK[code]})')
            fields.append(f'{stage}_iterations={int(state[stage+"_iterations"])}')
    return ' '.join(fields)


def solve(target, nodes, order, c, H, q, spec, log, label, parent=None, limit=1e-10,
          identity=False, condition_limit=COND):
    started = time.perf_counter()
    pieces = []
    n = len(target)
    log(f'{label} start states={n} nodes={len(nodes)} p={len(spec["names"])}')
    for start in range(0, n, 256):
        stop = min(n, start+256)
        seed_started = time.perf_counter()
        seeds, valid = affine_seed(target[start:stop], nodes, order, c, H, q, spec)
        seed_seconds = (time.perf_counter()-seed_started)/(stop-start)
        for loc, i in enumerate(range(start, stop)):
            state_started = time.perf_counter()
            if identity:
                seeds[loc] = 0
                valid[loc] = True
            if parent is None:
                value = attempt(target[i], nodes, order, c, H, q, spec, seeds[loc], 200,
                                limit, log, label, continuation=not identity,
                                condition_limit=condition_limit)
                value.update(parent_reason=0, affine_reason=value['reason'],
                             parent_iterations=0, affine_iterations=value['iterations'], attempt_selected=2)
                if not valid[loc] and np.all(np.isfinite(target[i])):
                    value.update(reason=11, failed=True)
                if identity:
                    value['start_path'] = 6 if value['failed'] else 5
            else:
                initial = parent[i].copy()
                a = attempt(target[i], nodes, order, c, H, q, spec, initial, 100, limit, log, label,
                            condition_limit=condition_limit)
                b = attempt(target[i], nodes, order, c, H, q, spec, seeds[loc],
                            200-a['iterations'], limit, log, label, condition_limit=condition_limit)
                if not valid[loc]:
                    b.update(reason=11, failed=True)
                # A successful full-rank attempt wins; otherwise keep the best finite iterate.
                value = min((a, b), key=lambda v: (v['failed'], v['objective_mm2'])).copy()
                value.update(parent_reason=a['reason'], affine_reason=b['reason'],
                             parent_iterations=a['iterations'], affine_iterations=b['iterations'],
                             attempt_selected=1 if value['parameters'] is a['parameters'] else 2,
                             iterations=a['iterations']+b['iterations'])
                for start_name, candidate in (('parent', a), ('affine', b)):
                    for key in ('parameters', 'objective_mm2', 'gradient', 'rank', 'condition',
                                'jacobian_disagreement', 'damping', 'start_path', 'warm_reason',
                                'restricted_reason', 'restricted_iterations', 'release_reason',
                                'release_iterations', 'best_iterate_stage', 'failed',
                                'final_release_reason', 'final_release_failed',
                                *('best_iterate_'+k for k in PASS_DIAGNOSTICS),
                                *('final_release_'+k for k in PASS_DIAGNOSTICS)):
                        value[start_name+'_attempt_'+key] = candidate[key]
            try:
                if not value['failed'] and not fold_valid(value['parameters'], order, H, q, spec):
                    value.update(reason=13, failed=True)
            except Exception:
                value.update(reason=13, failed=True)
            if value['failed'] and limit == 1e-13:
                # Include expected degeneracies too: the recovery table, not
                # the solver reason, decides which failures are permissible.
                log(f'{label} failed state={i} '+failure_detail(value))
                if parent is not None:
                    for start_name, candidate in (('parent', a), ('affine', b)):
                        log(f'{label} failed state={i} attempt={start_name} '+failure_detail(candidate))
            value['seconds'] = time.perf_counter()-state_started+seed_seconds
            pieces.append(value)
            log(f'{label} states={i+1}/{n} failures={sum(v["failed"] for v in pieces)}', heartbeat=True)
        log(f'{label} chunk={stop}/{n} failures={sum(v["failed"] for v in pieces)} '
            f'solve_s={time.perf_counter()-started:.3f}')
    require(bool(pieces), 'Empty required fit')
    result = {k: np.asarray([v[k] for v in pieces]) for k in pieces[0]}
    elapsed = time.perf_counter()-started
    log(f'{label} complete seconds_per_state={elapsed/n:.6g} '
        f'seconds_per_iteration={elapsed/max(1,int(result["iterations"].sum())):.6g}')
    return result, elapsed


def split_coefficients(maps, prefix, C):
    maps[prefix+'_X'] = C[:, :25]
    maps[prefix+'_Y'] = C[:, 25:]


def tangent(order, c, H, q):
    def fun(p):
        points = forward_unchecked(p, order, c, H, q, BASE, NODES)
        z, valid = inverse(points, c, H)
        require(np.all(valid), 'Tangent inverse')
        C, good, _ = coefficients(z)
        require(np.all(good), 'Tangent interpolation')
        return C
    fun.tail = (50,)
    J, diff = derivative(np.zeros(8), fun, np.arange(8))
    U, s, _ = np.linalg.svd(J, full_matrices=False)
    rank = int(np.sum(s > s[0]*max(J.shape)*EPS))
    require(rank == 8, 'Coefficient tangent rank')
    projector = np.eye(50)-U[:, :rank]@U[:, :rank].T
    return fun(np.zeros((1, 8)))[0], J, projector, dict(rank=rank, independent_constraints=50-rank,
                                                     half_step_disagreement=diff)


def camera_jacobian(z, H):
    def calculate(x):
        cols = []
        for j in range(2):
            h = FD*np.maximum(1., abs(H*x[..., j]))
            delta = np.zeros_like(x)
            delta[..., j] = h/2
            cols.append((joint.camera(H*x+delta)-joint.camera(H*x-delta))/h[..., None])
        return np.stack(cols, axis=-1)
    return nine.checked_batch(calculate, z, (25, 2, 2))


def coefficient_diagnostic(prefix, C, z, H, failed, groups, maps):
    """Full coefficient-vector allocation, evaluated through V and local camera J."""
    split_coefficients(maps, prefix+'_normalized', C)
    split_coefficients(maps, prefix+'_coefficient_um', 1000*H*C)
    jac, valid = camera_jacobian(z, H)
    dz = np.einsum('nj,scj->snc', V, C.reshape(-1, 2, 25))
    sensor = 1000*H*np.einsum('snab,snb->sna', jac, dz)
    bad = failed | ~valid
    maps[prefix+'_failed'] = bad
    maps[prefix+'_local_sensor_rms_um'] = np.sqrt(np.mean(np.sum(sensor**2, axis=-1), axis=1))
    maps[prefix+'_local_sensor_max_um'] = np.max(np.linalg.norm(sensor, axis=-1), axis=1)
    # Each individual coordinate has an unambiguous allocation: one component,
    # one monomial, all other coefficients zero. Keep its sensor scale distinct
    # from the combined diagnostic above (cancellation matters in the latter).
    individual_rms, individual_max = np.empty_like(C), np.empty_like(C)
    for component in range(2):
        column_norm = np.linalg.norm(jac[..., component], axis=-1)
        for j in range(25):
            scale = 1000*H*abs(C[:, component*25+j, None])*abs(V[None, :, j])*column_norm
            individual_rms[:, component*25+j] = np.sqrt(np.mean(scale**2, axis=1))
            individual_max[:, component*25+j] = np.max(scale, axis=1)
    split_coefficients(maps, prefix+'_individual_local_sensor_rms_um', individual_rms)
    split_coefficients(maps, prefix+'_individual_local_sensor_max_um', individual_max)
    return {g: dict(coefficient_um=nine.aggregate(1000*H*C[m], bad[m]),
                    individual_local_sensor_rms_um=nine.aggregate(individual_rms[m], bad[m]),
                    individual_local_sensor_max_um=nine.aggregate(individual_max[m], bad[m]),
                    local_sensor_equivalent=nine.vector_summary(sensor[m], bad[m])) for g, m in groups.items()}


def solver_summary(fit, mask):
    failure = fit['failed'][mask]
    grad = fit['final_release_gradient'][mask]
    badgrad = grad[failure]
    finite = badgrad[np.isfinite(badgrad)]
    seconds = float(fit['seconds'][mask].sum())
    iterations = int(fit['iterations'][mask].sum())
    summary = dict(states=int(mask.sum()), failures=int(failure.sum()),
                solve_seconds=seconds, total_iterations=iterations,
                seconds_per_state=seconds/int(mask.sum()) if mask.any() else None,
                seconds_per_iteration=seconds/iterations if iterations else None,
                reason_counts={name: int(np.count_nonzero(fit['reason'][mask] == code))
                               for code, name in CODEBOOK.items()},
                start_paths={name: int(np.count_nonzero(fit['start_path'][mask] == code))
                             for code, name in nine.START_PATHS.items()},
                successful_gradient_1e13_to_1e10=int(np.count_nonzero(
                    ~failure & (grad > 1e-13) & (grad <= 1e-10))),
                failed_gradient_finite_count=len(finite),
                failed_gradient_nonfinite_count=int(np.count_nonzero(~np.isfinite(badgrad))),
                failed_gradient_median=float(np.median(finite)) if len(finite) else None,
                failed_gradient_max=float(np.max(finite)) if len(finite) else None,
                actual_rank_counts={str(k): int(np.count_nonzero(fit['rank'][mask] == k))
                                    for k in np.unique(fit['rank'][mask])},
                effective_p_counts={str(k): int(np.count_nonzero(fit['effective_p'][mask] == k))
                                    for k in np.unique(fit['effective_p'][mask])},
                eta_frozen=int(fit['eta_frozen'][mask].sum()),
                mean_iterations=float(np.mean(fit['iterations'][mask])) if mask.any() else None)
    for prefix in ('best_iterate_', 'final_release_'):
        for key in PASS_DIAGNOSTICS:
            values = fit[prefix+key][mask]
            finite_values = values[np.isfinite(values)]
            summary[prefix+key] = dict(
                finite_count=len(finite_values),
                nonfinite_count=int(np.count_nonzero(~np.isfinite(values))),
                median=float(np.median(finite_values)) if len(finite_values) else None,
                max=float(np.max(finite_values)) if len(finite_values) else None)
        summary[prefix+'rank_counts'] = {
            str(k): int(np.count_nonzero(fit[prefix+'rank'][mask] == k))
            for k in np.unique(fit[prefix+'rank'][mask])}
    summary['final_release_reason_counts'] = {
        name: int(np.count_nonzero(fit['final_release_reason'][mask] == code))
        for code, name in CODEBOOK.items()}
    return summary


def evaluation(fit, observed, fitted_nodes, order, c, H, q, spec):
    pred, valid = predict(fit['parameters'], order, c, H, q, spec, NODES)
    # A fit using 24/8 nodes does not inherit failures of an excluded target.
    fit['failed'] |= ~valid
    fit['reason'][~valid & np.isin(fit['reason'], (1, 2))] = 9
    errors = 1000*(pred-observed)
    z, inv_ok = inverse(pred, c, H)
    C, coeff_ok, defect = coefficients(z)
    fit['failed'] |= ~inv_ok | ~coeff_ok
    fit['reason'][(~inv_ok | ~coeff_ok) & np.isin(fit['reason'], (1, 2))] = 9
    return dict(fit=fit, pred=pred, errors=errors, coefficients=C, coefficient_defect=defect,
                nodes=np.asarray(fitted_nodes), spec=spec)


def metric(result, mask, nodes):
    return nine.vector_summary(result['errors'][mask][:, nodes], result['fit']['failed'][mask])


def retain(prefix, result, observed_z, raw_C, raw_bad, projector, c, H, groups, maps):
    fit, errors, spec = result['fit'], result['errors'], result['spec']
    for key, value in fit.items():
        maps[prefix+'_solver_'+key] = value
    maps[prefix+'_residual_um'] = errors
    for label, nodes in dict(all=np.arange(25), nine=NINE, fitted=result['nodes'], **RINGS).items():
        norms = np.linalg.norm(errors[:, nodes], axis=-1)
        maps[prefix+'_'+label+'_rms_um'] = np.sqrt(np.mean(norms**2, axis=1))
        maps[prefix+'_'+label+'_max_um'] = np.max(norms, axis=1)
    maps[prefix+'_centre_error_um'] = errors[:, 12]
    maps[prefix+'_centre_norm_um'] = np.linalg.norm(errors[:, 12], axis=-1)
    shift, shift_valid = inverse(result['pred'][:, 12:13], c, H)
    null = shift[:, 0]-observed_z[:, 12]
    maps[prefix+'_ring_implied_shift'] = shift[:, 0]
    maps[prefix+'_null_normalized'] = null
    maps[prefix+'_null_coefficient_um'] = 1000*H*null
    maps[prefix+'_null_failed'] = fit['failed'] | ~shift_valid | raw_bad
    split_coefficients(maps, prefix+'_fitted_C', result['coefficients'])
    corrected = raw_C-result['coefficients']
    bad = fit['failed'] | raw_bad
    corrected_summary = coefficient_diagnostic(prefix+'_corrected', corrected, observed_z, H, bad, groups, maps)
    projected_summary = coefficient_diagnostic(prefix+'_projected', corrected@projector.T,
                                               observed_z, H, bad, groups, maps)
    summaries = {}
    for group, m in groups.items():
        entry = dict(solver=solver_summary(fit, m),
                     parameters=nine.parameter_summary(fit['parameters'][m], fit['failed'][m]),
                     all25=metric(result, m, np.arange(25)), nine=metric(result, m, NINE),
                     fitted=metric(result, m, result['nodes']),
                     centre=nine.vector_summary(errors[m, 12], fit['failed'][m] | raw_bad[m]),
                     rings={name: dict(node_count=len(nodes), **metric(result, m, nodes))
                            for name, nodes in RINGS.items()},
                     corrected=corrected_summary[group], projected=projected_summary[group])
        failed = m & fit['failed']
        entry['failed_best_iterate_diagnostic_only'] = dict(
            all25=nine.vector_summary(errors[failed], np.zeros(int(failed.sum()), bool)),
            centre=nine.vector_summary(errors[failed, 12], np.zeros(int(failed.sum()), bool)))
        summaries[group] = entry
    return dict(family=spec, nominal_p=len(spec['names']), nominal_dof=2*len(result['nodes'])-len(spec['names']),
                shift_units='normalized sensor' if prefix.startswith('C') else 'normalized before camera',
                corrected_label='model-relative interpolation coefficients; not point errors',
                null_label='model-dependent camera-inverted centre; not a gaze-law prediction', groups=summaries)


def training_bins(groups):
    return {name: m & groups['train'] for name, m in groups.items() if name.startswith('bin_')}


def gain_test(parent, trial, bins):
    details = {}
    passed = True
    available = 0
    for name, mask in bins.items():
        if not mask.any():
            details[name] = dict(count=0, available=False, passed=None)
            continue
        available += 1
        entry = dict(count=int(mask.sum()), available=True, metrics={})
        checks = []
        for label, nodes in (('nine', NINE), ('all25', np.arange(25))):
            a, b = metric(parent, mask, nodes), metric(trial, mask, nodes)
            for key in ('rms_um', 'max_um'):
                x, y = a[key], b[key]
                good = (a['failure_count'] == b['failure_count'] == 0 and
                        np.isfinite(x) and np.isfinite(y) and
                        (y <= 1e-9 if x <= 1e-9 else y <= .9*x))
                checks.append(good)
                entry['metrics'][label+'_'+key] = dict(parent=x, trial=y,
                    ratio=y/x if x > 1e-9 and np.isfinite(x) else None, floor=x <= 1e-9, passed=good)
        entry['passed'] = all(checks)
        details[name] = entry
        passed &= entry['passed']
    return bool(passed and available), details


def training_target(result, train):
    m = metric(result, train, np.arange(25))
    return bool(train.any() and m['failure_count'] == 0 and m['max_um'] < 1)


def fit_evaluate(observed, nodes, order, c, H, q, spec, log, label, report,
                 parent=None, limit=1e-10):
    fit, elapsed = solve(observed[:, nodes], NODES[nodes], order, c, H, q, spec,
                          log, label, parent, limit)
    report['timings'][label+'_s'] = elapsed
    return evaluation(fit, observed, nodes, order, c, H, q, spec)


def ladder(observed, base, order, c, H, q, groups, log, report, save):
    # Only training observations are passed to candidates. Global masks are not
    # allowed to turn a holdout result into an acceptance/stop signal.
    train_ids = np.flatnonzero(groups['train'])
    local_groups = {k: v[train_ids] for k, v in groups.items()}
    bins = training_bins(local_groups)
    parent = {k: v for k, v in base.items()}
    parent['fit'] = {k: v[train_ids].copy() for k, v in base['fit'].items()}
    for key in ('pred', 'errors', 'coefficients', 'coefficient_defect'):
        parent[key] = base[key][train_ids]
    spec = BASE
    accepted = []
    trials = []
    consecutive = 0
    reason = 'end_of_ladder'
    candidates = list(BLOCKS)+['placement_pre', 'placement_middle']
    for block in candidates:
        if training_target(parent, local_groups['train']):
            reason = 'training_target'
            break
        if consecutive >= 2:
            reason = 'two_consecutive_failed_gain_tests'
            break
        if block.startswith('placement_'):
            if 'prism' not in spec['blocks']:
                trials.append(dict(block=block, decision='skipped_prism_not_accepted'))
                continue
            candidate = family(spec['blocks'], block.split('_')[1])
        else:
            candidate = family((*spec['blocks'], block), spec['placement'])
        if not len(train_ids):
            trials.append(dict(block=block, decision='unavailable_no_training_states'))
            reason = 'unavailable_training'
            break
        initial = expand_parent(parent['fit']['parameters'], spec, candidate)
        label = f'{order}.D2.{len(trials)+1}.{block}.train'
        trial = fit_evaluate(observed[train_ids], np.arange(25), order, c, H, q,
                             candidate, log, label, report, initial)
        gain, details = gain_test(parent, trial, bins)
        decision = dict(block=block, family=candidate, gain_by_bin=details,
                        decision='accepted' if gain else 'rejected',
                        solver=solver_summary(trial['fit'], np.ones(len(train_ids), bool)))
        trials.append(decision)
        # Trial maps include starts, failure reasons and best iterates, never Jacobians.
        save(f'{order}_trial{len(trials)}', trial, train_ids)
        if gain:
            spec, parent = candidate, trial
            accepted.append(spec)
            consecutive = 0
        else:
            consecutive += 1
        log(f'{label} decision={decision["decision"]} consecutive_failed_gains={consecutive}')
    if consecutive >= 2:
        reason = 'two_consecutive_failed_gain_tests'
    elif training_target(parent, local_groups['train']):
        reason = 'training_target'
    return spec, accepted, dict(trials=trials, stop=reason, consecutive_failed_gains=consecutive,
                                frozen=True, frozen_before_holdout_evaluation=True,
                                condition_limit=COND,
                                condition_policy='Extension steps whose per-state condition exceeds the '
                                    'production limit are rejected by the solver (counted as failures), never relaxed',
                                selection='25-node centre-assisted training only; zero excluded')


def polynomial_design(u, degree):
    return np.column_stack([u[:, 0]**i*u[:, 1]**(d-i)
                            for d in range(degree+1) for i in range(d+1)])


def svd_regression(A, y):
    U, s, vt, scales, info = joint.scaled_svd(A)
    require(info['full_rank'], 'Polynomial design lacks full column rank')
    return (vt.T@((U.T@y)/s[:, None]))/scales[:, None]


def compression(result, order, c, H, q, groups, maps, prefix, quick, log):
    u = maps['u']
    train = groups['train']
    folds = (maps['global_ade_index']//2+2*(maps['global_bde_index']//2)) % 4
    fit = result['fit']
    history = []
    selected = None
    for degree in range(9):
        log(f'{order} compression degree={degree}')
        A = polynomial_design(u, degree)
        params = np.full_like(fit['parameters'], np.nan)
        predicted = np.full_like(result['pred'], np.nan)
        bad = np.ones(len(u), bool)
        rank_ok = True
        try:
            require(train.any() and not np.any(fit['failed'][train]), 'Failed training parameter maps')
            for fold in range(4):
                validation = train & (folds == fold)
                training = train & (folds != fold)
                require(validation.any(), 'Empty deterministic validation fold')
                coef = svd_regression(A[training], fit['parameters'][training])
                params[validation] = A[validation]@coef
                predicted[validation], good = predict(params[validation], order, c, H, q,
                                                      result['spec'], NODES)
                bad[validation] = ~good
        except Exception as exc:
            rank_ok = False
            history.append(dict(degree=degree, full_rank_folds=False, reason=str(exc), passed=False))
        if not rank_ok:
            continue
        error = 1000*(predicted-result['pred'])
        metrics = {name: nine.vector_summary(error[m], bad[m]) for name, m in training_bins(groups).items()}
        nonempty = [v for v in metrics.values() if v['total_count']]
        passed = bool(nonempty and all(v['failure_count'] == 0 and v['rms_um'] <= .1
                                       and v['max_um'] <= .3 for v in nonempty))
        history.append(dict(degree=degree, full_rank_folds=True, bins=metrics, passed=passed))
        if passed:
            selected = degree
            maps[prefix+'_compression_cv_parameters'] = params
            break
    answer = dict(training_folds='(floor(global_ade/2)+2*floor(global_bde/2)) mod 4',
                  errors='compression relative to free-state predictions, not trace residuals',
                  trials=history, selected_degree=selected,
                  conclusion='pilot insufficient for a global degree claim' if quick else
                  ('degree >8 or unresolved' if selected is None else f'total degree {selected}'))
    if selected is not None:
        A = polynomial_design(u, selected)
        coef = svd_regression(A[train], fit['parameters'][train])
        params = A@coef
        prediction, good = predict(params, order, c, H, q, result['spec'], NODES)
        maps[prefix+'_compression_polynomial_coefficients'] = coef
        maps[prefix+'_compression_parameters'] = params
        maps[prefix+'_compression_failed'] = ~good
        error = 1000*(prediction-result['pred'])
        maps[prefix+'_compression_residual_um'] = error
        answer['refit_forward_error_once'] = {
            name: nine.vector_summary(error[m], ~good[m] | fit['failed'][m])
            for name, m in groups.items()}
    return answer


def smoothness(result, maps, prefix):
    p, bad = result['fit']['parameters'], result['fit']['failed']
    ai, bi, w = maps['global_ade_index'], maps['global_bde_index'], maps['w_deg']
    lookup = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(ai, bi))}
    answer = dict(parameters=nine.parameter_summary(p, bad), axes={})
    # Adjacency means consecutive available commanded angles, including the
    # pilot's six-degree spacing. Difference units are per degree/per degree^2.
    for axis in (0, 1):
        first, second, masks1, masks2, pairs, triples = [], [], [], [], [], []
        avalues, bvalues = np.unique(ai), np.unique(bi)
        along, across = (avalues, bvalues) if axis == 0 else (bvalues, avalues)
        for fixed in across:
            line = [lookup.get((int(x), int(fixed)) if axis == 0 else (int(fixed), int(x))) for x in along]
            for k in range(len(line)-1):
                i, j = line[k:k+2]
                if i is None or j is None:
                    continue
                spacing = abs(along[k+1]-along[k])*.25
                first.append((p[j]-p[i])/spacing)
                masks1.append(bad[i] | bad[j])
                pairs.append((i, j))
            for k in range(len(line)-2):
                i, j, l = line[k:k+3]
                if any(v is None for v in (i, j, l)):
                    continue
                h1, h2 = np.diff(along[k:k+3])*.25
                second.append(2*((p[l]-p[j])/h2-(p[j]-p[i])/h1)/(h1+h2))
                masks2.append(bad[i] | bad[j] | bad[l])
                triples.append((i, j, l))
        entry = {}
        for label, values, failed, indices in (('first', first, masks1, pairs), ('second', second, masks2, triples)):
            values = np.asarray(values).reshape(-1, p.shape[1])
            failed = np.asarray(failed, bool)
            maps[f'{prefix}_axis{axis}_{label}_differences'] = values
            maps[f'{prefix}_axis{axis}_{label}_failed'] = failed
            maps[f'{prefix}_axis{axis}_{label}_indices'] = np.asarray(indices, int)
            entry[label] = nine.aggregate(values, failed)
        answer['axes']['ADE' if axis == 0 else 'BDE'] = entry
    return answer


def centre_comparison(a, b, groups, prefix, maps):
    # a=24-node, b=eight-node, matched D1 capacity.
    delta = 1000*(a['pred'][:, 12]-b['pred'][:, 12])
    bad = a['fit']['failed'] | b['fit']['failed']
    maps[prefix+'_24_minus_eight_centre_um'] = delta
    maps[prefix+'_24_minus_eight_failed'] = bad
    bins = {}
    for name, m in groups.items():
        if not name.startswith('bin_'):
            continue
        x = nine.vector_summary(a['errors'][m, 12], a['fit']['failed'][m])
        y = nine.vector_summary(b['errors'][m, 12], b['fit']['failed'][m])
        passed = None if not m.any() else bool(x['failure_count'] == y['failure_count'] == 0 and
                    x['rms_um'] <= .9*y['rms_um'] and x['max_um'] <= .9*y['max_um'])
        bins[name] = dict(noncentre24=x, outer8=y, improvement_10percent=passed,
                          centre_max_le_0p3=None if not m.any() else x['max_um'] <= .3,
                          centre_prediction_difference=nine.vector_summary(delta[m], bad[m]))
    populated = [v for v in bins.values() if v['improvement_10percent'] is not None]
    return dict(bins=bins, interior_benefit_established=bool(populated and all(
        v['improvement_10percent'] for v in populated)),
        scope='matched eight-parameter D1; fixed outer-only calibration; not a gaze-law prediction')


def ambiguity(noncentre_fits, maps):
    affected = np.zeros(len(maps['global_state_id']), bool)
    for name, result in noncentre_fits.items():
        valid = ~result['failed'] & (result['node_max'] < 1)
        flags = valid & (result['centre_norm'] > .3)
        maps[name+'_null_ambiguity'] = flags
        affected |= flags
    names = list(noncentre_fits)
    counts = {}
    for i, left in enumerate(names):
        a = noncentre_fits[left]
        for right in names[i+1:]:
            b = noncentre_fits[right]
            avalid = ~a['failed'] & (a['node_max'] < 1)
            bvalid = ~b['failed'] & (b['node_max'] < 1)
            delta = 1000*(a['centre']-b['centre'])
            flags = avalid & bvalid & (np.linalg.norm(delta, axis=-1) > .3)
            key = left+'_vs_'+right
            maps[key+'_centre_difference_um'] = delta
            maps[key+'_ambiguity'] = flags
            counts[key] = int(flags.sum())
            affected |= flags
    maps['remaining_null_order_ambiguity'] = affected
    return dict(affected_state_count=int(affected.sum()), pair_counts=counts,
                state_locations='maps.npz masks with global/source state IDs')


def analyze(observed, c, H, q, groups, report, maps, log, quick):
    z, valid_inverse = inverse(observed, c, H)
    C, valid_coeff, defect = coefficients(z)
    raw_bad = ~valid_inverse | ~valid_coeff
    maps.update(raw_failed=raw_bad, raw_inverse_failed=~valid_inverse,
                raw_interpolation_failed=~valid_coeff, raw_reconstruction_defect=defect,
                true_centre_normalized=z[:, 12], monomial_powers=np.asarray(POWERS), nodes=NODES,
                original_nine_indices=NINE, original_outer_indices=EIGHT)
    split_coefficients(maps, 'raw_C', C)
    report['D0'] = dict(zero_residual_dof=0, coefficient_count=50, V_condition=float(np.linalg.cond(V)),
        interpretation='Exact node interpolation coordinates, not Taylor coefficients. Raw D0 is model-free; '
                       'corrected D0 is model-relative. Aliases x^5=1.25x^3-.25x, x^6=1.25x^4-.25x^2 persist.',
        reconstruction={g: nine.aggregate(defect[m, None], raw_bad[m]) for g, m in groups.items()},
        shape_hypotheses=dict(prism='wrong-component X02/Y20; independent quadratic/keystone ties',
            cubic='anisotropic barrel, isotropic q variation or trefoil; independent tied cubic combinations',
            quartic='X04/Y40 and independent 22; quartic shift/null partner',
            higher='unconstrained fifth/higher mixed powers; radial r^4 populates mixed degree five',
            absent_AB_both=['14', '23', '32', '41', '34', '43', '44'],
            absent_A_tied_B=['24', '33', '42'],
            caution='First-order identity-L/zero-k hypotheses only; coefficients do not identify physical origin'),
        tangent={}, raw_local_constraints={})
    report['coverage'] = {g: int(m.sum()) for g, m in groups.items()}
    report['fits'], report['selection'], report['smoothness'], report['compression'] = {}, {}, {}, {}
    report['centre_comparison'] = {}
    noncentre = {}
    baselines, finals = {}, {}
    required_failures = int(raw_bad.sum())
    bin_scales = training_bins(groups)
    for order in ('A', 'B', 'C'):
        C0, J, projector, info = tangent(order, c, H, q)
        report['D0']['tangent'][order] = info
        maps[order+'_identity_C'] = C0
        maps[order+'_identity_coefficient_tangent'] = J
        maps[order+'_identity_projector'] = projector
        # This map is a first-order local diagnostic, never an exact-family residual.
        report['D0']['raw_local_constraints'][order] = coefficient_diagnostic(
            order+'_raw_local_constraint', (C-C0)@projector.T, z, H, raw_bad, groups, maps)
        base = fit_evaluate(observed, SCOPES['all'], order, c, H, q, BASE, log,
                            order+'.D1.all', report)
        baselines[order] = base
        report['fits'][order+'_D1_all'] = retain(order+'_D1_all', base, z, C, raw_bad,
                                                projector, c, H, groups, maps)
        required_failures += int(base['fit']['failed'].sum())

        def save_trial(prefix, result, ids):
            maps[prefix+'_global_state_id'] = maps['global_state_id'][ids]
            local = {k: v[ids] for k, v in groups.items()}
            report['fits'][prefix] = retain(prefix, result, z[ids], C[ids], raw_bad[ids],
                                            projector, c, H, local, maps)

        selected, accepted, selection = ladder(observed, base, order, c, H, q, groups,
                                                log, report, save_trial)
        report['selection'][order] = selection
        log(f'{order} frozen family={selected["names"]} placement={selected["placement"]}')
        previous, previous_spec = base, BASE
        for j, spec in enumerate(accepted, 1):
            initial = expand_parent(previous['fit']['parameters'], previous_spec, spec)
            label = f'{order}_D2_accepted{j}_all'
            value = fit_evaluate(observed, SCOPES['all'], order, c, H, q, spec, log, label, report, initial)
            report['fits'][label] = retain(label, value, z, C, raw_bad, projector, c, H, groups, maps)
            required_failures += int(value['fit']['failed'].sum())
            previous, previous_spec = value, spec
        final = previous
        finals[order] = final
        report['selection'][order]['selected_family'] = selected
        report['selection'][order]['full_target_pass'] = bool(not quick and
            metric(final, groups['overall'], np.arange(25))['max_um'] < 1 and not final['fit']['failed'].any())
        holdout = groups['even_odd'] | groups['odd_even'] | groups['odd_odd']
        # Pilots may have no original holdout states. Their transfer is unavailable.
        train_metric, test_metric = metric(final, groups['train'], np.arange(25)), metric(final, holdout, np.arange(25))
        ratio = None
        if train_metric['total_count'] and test_metric['total_count']:
            a, b = train_metric['rms_um'], test_metric['rms_um']
            if np.isfinite(a) and np.isfinite(b) and not (a <= 1e-9 and b <= 1e-9):
                ratio = b/a if a > 0 else np.inf
        report['selection'][order].update(training=train_metric, frozen_holdout=test_metric,
            holdout_train_rms_ratio=ratio, poor_transfer=None if ratio is None else ratio > 1.2)
        report['smoothness'][order] = smoothness(final, maps, order+'_frozen')
        report['smoothness'][order]['shift_units'] = 'normalized sensor' if order == 'C' else 'normalized before camera'
        report['compression'][order] = compression(final, order, c, H, q, groups, maps,
                                                   order+'_frozen', quick, log)

        # Independently restart each subset's entire accepted-parent chain.
        subset_results = {}
        for scope in ('noncentre', 'eight'):
            nodes = SCOPES[scope]
            prefix = f'{order}_D1_{scope}'
            parent = fit_evaluate(observed, nodes, order, c, H, q, BASE, log, prefix, report)
            subset_results[scope] = parent
            noncentre[prefix] = parent
            report['fits'][prefix] = retain(prefix, parent, z, C, raw_bad, projector, c, H, groups, maps)
            required_failures += int(parent['fit']['failed'].sum())
            pspec = BASE
            for j, spec in enumerate(accepted, 1):
                prefix = f'{order}_D2_accepted{j}_{scope}'
                if scope == 'eight' and not allow_eight(spec):
                    report['fits'][prefix] = dict(status='structurally_unidentifiable', nominal_p=len(spec['names']),
                        reason='Eight-node extension requires p<16 and full column rank; no minimum-norm fit used')
                    break
                initial = expand_parent(parent['fit']['parameters'], pspec, spec)
                value = fit_evaluate(observed, nodes, order, c, H, q, spec, log, prefix, report, initial)
                report['fits'][prefix] = retain(prefix, value, z, C, raw_bad, projector, c, H, groups, maps)
                report['fits'][prefix]['selection_leakage_label'] = (
                    'Family selected using centre-assisted training data; holdout centres unused by selection. '
                    'Subset starts and refits never use centre targets or centre-assisted parameter fits.')
                noncentre[prefix] = value
                if scope == 'noncentre':
                    required_failures += int(value['fit']['failed'].sum())
                else:
                    rank_failed = value['fit']['reason'] == 5
                    report['fits'][prefix]['unidentifiable_states'] = int(rank_failed.sum())
                    required_failures += int((value['fit']['failed'] & ~rank_failed).sum())
                parent, pspec = value, spec
        report['centre_comparison'][order] = centre_comparison(subset_results['noncentre'], subset_results['eight'],
                                                              groups, order, maps)
        # Scope/candidate/bin timings are reported without per-state JSON.
        for name, entry in list(report['fits'].items()):
            if name.startswith(order) and 'groups' in entry:
                for bin_name in bin_scales:
                    summary = entry['groups'][bin_name]['solver']
                    log(f'{name}/{bin_name} states={summary["states"]} failures={summary["failures"]} '
                        f'seconds_per_state={summary["seconds_per_state"]} '
                        f'seconds_per_iteration={summary["seconds_per_iteration"]}')
        # Only centres and fitted-node maxima are needed for inter-order tests;
        # the full diagnostic arrays have already been written to maps.npz.
        for name, entry in list(noncentre.items()):
            if name.startswith(order):
                norms = np.linalg.norm(entry['errors'], axis=-1)
                noncentre[name] = dict(centre=entry['pred'][:, 12].copy(), centre_norm=norms[:, 12],
                    node_max=np.max(norms[:, entry['nodes']], axis=1), failed=entry['fit']['failed'].copy())
        log(f'{order} complete required_failures_so_far={required_failures}')
    report['remaining_ambiguity'] = ambiguity(noncentre, maps)
    complete = bool(not quick and len(observed) == 10201)
    limited = not complete or required_failures != 0
    insufficient = bool(not limited and all(metric(v, groups['overall'], np.arange(25))['max_um'] >= 1
                                          for v in baselines.values()))
    extension = {}
    for order, value in finals.items():
        selection = report['selection'][order]
        rank_deficient = any(t.get('solver', {}).get('reason_counts', {}).get('rank_or_condition', 0)
                             for t in selection['trials'])
        hold = selection['frozen_holdout']
        supported = bool(not limited and value['spec']['blocks'] and selection['full_target_pass'])
        unresolved = bool(selection['consecutive_failed_gains'] >= 2 or rank_deficient or
                          (hold['max_um'] is not None and hold['max_um'] >= 1))
        extension[order] = dict(supported_as_shape=supported, falsified_or_unresolved=unresolved,
                               physical_origin='unproved hypothesis', rank_deficient_trial=bool(rank_deficient))
    winner = min(finals, key=lambda o: (
        metric(finals[o], groups['train'], np.arange(25))['max_um']
            if groups['train'].any() else np.inf,
        metric(finals[o], groups['train'], np.arange(25))['rms_um']
            if groups['train'].any() else np.inf,
        len(finals[o]['spec']['names']), o))
    report['decision'] = dict(incomplete_or_solver_limited=limited, full_coverage=complete,
        required_failure_count=required_failures, eight_parameter_family_insufficient=insufficient,
        D1_failed_bins={o: [b for b in bin_scales if
            (metric(v, groups[b], np.arange(25))['max_um'] or 0) >= 1] for o, v in baselines.items()},
        extension=extension, overall_winner_training_only=winner,
        winner_rule='Training worst 25-node error, RMS, parameter count, then A/B/C',
        remaining_null_order_ambiguity=report['remaining_ambiguity']['affected_state_count'] > 0,
        note='No full-grid compliance claim' if limited else
             'Sampled field-node diagnostic; isolated optimization is not proof of a global optimum')
    if quick:
        # Use pass-specific state counts; training-only passes scale to 2600,
        # all-state passes to 10201. Hash/load cost is explicitly separate.
        estimate = 0.
        for name, seconds in report['timings'].items():
            if name in ('hash_s', 'deserialize_s', 'load_total_s'):
                continue
            estimate += seconds*(2600/max(1, groups['train'].sum()) if '.train' in name else 10201/len(observed))
        load_estimate = report['timings'].get('load_total_s', 0)*255025/625
        report['planning_estimate'] = dict(full_load_estimate_s=load_estimate,
            full_compute_estimate_s=estimate, total_range_s=[load_estimate+estimate, load_estimate+3*estimate],
            all_state_factor=10201/25, measured_guarantee=False,
            note='Pilot extrapolation only; harder states and ladder paths may differ; not a deadline')
    report['numerical_gates_passed'] = required_failures == 0
    report['status'] = 'completed' if not required_failures else 'completed_with_failures'


def synthetic_parameters(spec, j):
    p = np.zeros(len(spec['names']))
    if j:
        p[:8] = [.01*j, -.007*j, .001*j, .0003*j, -.0002*j, -.0005*j, .0004*j, -.0003*j]
        for k, name in enumerate(spec['names'][8:]):
            p[k+8] = .02 if name == 'eta' else 1e-4*(-1)**k
    return p


# Fixed recovery expectations, independent of derivative/scaled_svd results.
# Rows follow specifications below; columns are all/noncentre/eight. Named
# sets include their counts so deleting a case cannot silently shrink a gate.
SELFTEST_CASE_NAMES = ('identity', 'nonsymmetric_1', 'nonsymmetric_2',
                       'nonsymmetric_3', 'nonsymmetric_4', 'nonsymmetric_5',
                       'nonsymmetric_6', 'nonsymmetric_7', 'nonsymmetric_8')
SELFTEST_CASE_SETS = {
    'all9': dict(count=9, names=SELFTEST_CASE_NAMES, classification='recoverable'),
    'nonzero8': dict(count=8, names=SELFTEST_CASE_NAMES[1:], classification='recoverable'),
    'degenerate0': dict(count=0, names=(), classification='structurally_unidentifiable'),
    'production_rejected_ill_conditioned': dict(count=8, names=SELFTEST_CASE_NAMES[1:],
        classification='production_rejected_ill_conditioned',
        justification='rank = p, truth-point condition exceeds production limit COND; '
            'production rejects it; no recovery claim is made'),
}
# Declared accounting for EVERY row/order/scope below (not solver outcomes).
# p = 4 L + 2 shift + 2 k + listed extras; equations = 50/48/16 for
# all/noncentre/eight. I=all9, N=nonzero8, D=degenerate0,
# K=production_rejected_ill_conditioned. Each triple is A/B/C.
# O/E count odd/even tangent columns at identity, before freezing eta, on the
# four opposite pairs of the eight-node ring (capacity 8 odd + 8 even).
#                              p   O/E    all:50  noncentre:48  eight:16
#  0 baseline/post             8   4/4    I/I/I      I/I/I       I/I/I
#  1 Q/post                   11   7/4    I/I/I      I/I/I       I/I/I
#  2 prism/post               10   4/6    I/I/I      I/I/I       I/I/I
#  3 trefoil/post             10   6/4    I/I/I      I/I/I       I/I/I
#  4 beta/post                 9   5/4    I/I/I      I/I/I       I/I/I
#  5 quartic/post             10   4/6    I/I/I      I/I/I       I/I/I
#  6 lag/post                  9   4/4+0  I/I/I      I/I/I       I/I/I
#  7 Q+prism/post             13   7/6    I/I/I      I/I/I       I/I/I
#  8 Q+prism+trefoil/post     15   9/6    I/I/I      I/I/I       N/D/K
#  9 preceding+beta/post      16  10/6    I/I/I      I/I/I       D/D/D
# 10 preceding+quartic/post   18  10/8    I/I/I      I/I/I       D/D/D
# 11 preceding+lag/post       19  10/8+0  I/I/I      I/I/I       D/D/D
# 12 same full family/pre     19  10/8+0  I/I/I      I/I/I       D/D/D
# 13 same full family/middle  19  10/8+0  I/I/I      I/I/I       D/D/D
# For all/noncentre the opposite-pair capacities are 24 odd/26 even and
# 24 odd/24 even. All listed column counts fit, with no ring-only aliases:
# interior |x|,|y|=.5 distinguish x from x^3 and x^2 from x^4, and the
# additional radii distinguish radial cubic/quintic columns. I retains a
# full-rank recovery REQUIREMENT, not a claim proved merely by p<equations.
# Lag's eta column is exactly zero at s=0: freeze it, effective p=p-1 there
# (8 or 18), and require recovery only of the other parameters. Placement
# changes do not alter these identity counts. For eight nodes rows 9..13
# violate the stipulated nominal p<16 gate, even if eta is frozen.
# On eight nodes x^3=x, y^3=y; odd scalar functions have dimension four.
# The counts above are column COUNTS, not ranks. With Q and beta together,
# the isotropic L/Q/beta directions also sample only two radii (r^2=1,2).
# For post placement, 1-r^2+x^2*y^2=0 makes shift/prism/quartic dependent
# componentwise at identity. Rows 9..13 already fail the p budget regardless;
# pre/middle prism derivatives need not have this particular post alias.
# At identity row 8 has L(4)+Q(3)+trefoil(2)=9 odd columns in dimension eight,
# so rank <= 8+6=14<15 for A/B/C. Shift(2), k(2), prism(2) are even there.
# For B this deficiency persists: write S(d)=U(d)+alpha*r^2, U odd.
# Eight odd-component equations for delta U=M U have nine unknowns
# (delta Q,delta t:5; M:4), hence a nonzero null direction. Set
# delta alpha=M alpha, delta L=-L M, delta k=-M.T k, delta s=0; then
# delta S=M S is absorbed by T, before the invertible outer camera. Thus
# B has rank <=14 for every case, irrespective of k, s, or the free prism.
# A instead has S(T(d))+s before camera: nonzero k destroys antipodal
# pairing of its barrel inputs; nonzero prism also mixes L's parity.
# C has Cam(H*S(d))/H INSIDE T and s outside camera: at nonzero prism,
# D Cam(H*S) delta S is not a fixed linear M times Cam(H*S); B's gauge
# cannot be carried through a nonlinear radial camera. A/C therefore have
# no deficiency established by this counting away from identity. N retains
# strict production-limit recovery. K separately requires full truth rank,
# truth condition above COND and production rejection, with no recovery claim;
# neither class may be reclassified from a solver failure.
# In particular, C's Q+prism PARENT is not structurally deficient at its
# affine/zero-extra seed. Put b_r=1+q*r^2, psi_r=Cam(H*b_r*d)/(H*b_r*d)
# (the scalar radial factor), f_r=b_r*psi_r and
# a_r=psi_r+2*b_r^2*r^2*psi'_r, for r^2=1,2. An odd-column null would
# require delta Q isotropic and 2*a_2=f_2*a_1/f_1. The fixed camera and
# q=-.03 give the two sides near 2 and .94/.97, respectively, so the
# seven L/Q columns are independent. For the even columns, eliminating
# shift and k using the axial pairs, then comparing diagonal components,
# forces (2*psi_2-psi_1)*delta alpha=0. This factor is near 1, so shift/k/
# prism have rank six. Thus this seed has rank 7+6=13, not <=12.
# This does NOT establish child conditioning: C's outer homography is an
# invertible change of output coordinates, so nonzero s or k cannot lift
# a template gauge. At alpha=0 the child still has nine odd columns in
# eight components; with an identity camera B's gauge also persists at
# nonzero alpha. Lifting it can therefore be weak in BOTH alpha and the
# fixed camera nonlinearity (K1*H^2/F^2 is about 1.1e-4 here). C's measured
# truth rank is 15 with condition about 1.25e8: K, not a structural null.
# Production COND and all parameter/prediction gates remain unchanged.
SELFTEST_RECOVERY_TABLE = {
    'A': {
        'baseline/post': ('all9', 'all9', 'all9'),
        'Q/post': ('all9', 'all9', 'all9'),
        'prism/post': ('all9', 'all9', 'all9'),
        'trefoil/post': ('all9', 'all9', 'all9'),
        'beta/post': ('all9', 'all9', 'all9'),
        'quartic/post': ('all9', 'all9', 'all9'),
        'lag/post': ('all9', 'all9', 'all9'),
        'Q+prism/post': ('all9', 'all9', 'all9'),
        'Q+prism+trefoil/post': ('all9', 'all9', 'nonzero8'),
        'Q+prism+trefoil+beta/post': ('all9', 'all9', 'degenerate0'),
        'Q+prism+trefoil+beta+quartic/post': ('all9', 'all9', 'degenerate0'),
        'Q+prism+trefoil+beta+quartic+lag/post': ('all9', 'all9', 'degenerate0'),
        'Q+prism+trefoil+beta+quartic+lag/pre': ('all9', 'all9', 'degenerate0'),
        'Q+prism+trefoil+beta+quartic+lag/middle': ('all9', 'all9', 'degenerate0'),
    },
    'B': {
        'baseline/post': ('all9', 'all9', 'all9'),
        'Q/post': ('all9', 'all9', 'all9'),
        'prism/post': ('all9', 'all9', 'all9'),
        'trefoil/post': ('all9', 'all9', 'all9'),
        'beta/post': ('all9', 'all9', 'all9'),
        'quartic/post': ('all9', 'all9', 'all9'),
        'lag/post': ('all9', 'all9', 'all9'),
        'Q+prism/post': ('all9', 'all9', 'all9'),
        'Q+prism+trefoil/post': ('all9', 'all9', 'degenerate0'),
        'Q+prism+trefoil+beta/post': ('all9', 'all9', 'degenerate0'),
        'Q+prism+trefoil+beta+quartic/post': ('all9', 'all9', 'degenerate0'),
        'Q+prism+trefoil+beta+quartic+lag/post': ('all9', 'all9', 'degenerate0'),
        'Q+prism+trefoil+beta+quartic+lag/pre': ('all9', 'all9', 'degenerate0'),
        'Q+prism+trefoil+beta+quartic+lag/middle': ('all9', 'all9', 'degenerate0'),
    },
    'C': {
        'baseline/post': ('all9', 'all9', 'all9'),
        'Q/post': ('all9', 'all9', 'all9'),
        'prism/post': ('all9', 'all9', 'all9'),
        'trefoil/post': ('all9', 'all9', 'all9'),
        'beta/post': ('all9', 'all9', 'all9'),
        'quartic/post': ('all9', 'all9', 'all9'),
        'lag/post': ('all9', 'all9', 'all9'),
        'Q+prism/post': ('all9', 'all9', 'all9'),
        'Q+prism+trefoil/post': ('all9', 'all9', 'production_rejected_ill_conditioned'),
        'Q+prism+trefoil+beta/post': ('all9', 'all9', 'degenerate0'),
        'Q+prism+trefoil+beta+quartic/post': ('all9', 'all9', 'degenerate0'),
        'Q+prism+trefoil+beta+quartic+lag/post': ('all9', 'all9', 'degenerate0'),
        'Q+prism+trefoil+beta+quartic+lag/pre': ('all9', 'all9', 'degenerate0'),
        'Q+prism+trefoil+beta+quartic+lag/middle': ('all9', 'all9', 'degenerate0'),
    },
}


def run_selftest(report, maps, log):
    """No input reads; declared production rejections make no recovery claim."""
    c, H, q = np.zeros(2), 2.59, -.03
    failures, checks, recovery_checks, ill_conditioned_checks = [], [], [], []
    def check(name, function):
        log('selftest '+name)
        try:
            function()
            checks.append(dict(name=name, passed=True))
        except Exception as exc:
            failures.append(name+': '+type(exc).__name__+': '+str(exc))
            checks.append(dict(name=name, passed=False, reason=str(exc)))
            log('selftest failure '+failures[-1])

    def require_synthetic_fit(fit, label, ids=None, truth_context=None):
        """Failure details must survive the enclosing check's exception handler.

        Truth Jacobians are diagnostics only, never starts, steps, rank
        exceptions or evidence for removing a required recovery case.
        """
        failed = np.atleast_1d(fit['failed'])
        ids = np.arange(len(failed)) if ids is None else ids
        bad = [int(i) for i in ids if failed[i]]
        if not bad:
            return
        first = None
        for i in bad:
            state = {k: np.atleast_1d(v)[i] for k, v in fit.items()}
            detail = f'{label} state={i} '+failure_detail(state)
            log('selftest required failure '+detail)
            if first is None:
                first = detail
            if truth_context is not None:
                truth, nodes, order, spec = truth_context
                p = truth[i].copy()
                active = np.arange(len(p))
                if 'eta' in spec['names'] and np.max(abs(p[:2])) <= 1e-14:
                    eta = spec['names'].index('eta')
                    p[eta] = 0
                    active = active[active != eta]
                def function(x):
                    return forward_unchecked(x, order, c, H, q, spec, nodes)
                function.tail = (len(nodes), 2)
                try:
                    J, diff = derivative(p, function, active)
                    _, _, _, _, info = joint.scaled_svd(J)
                    log(f'{label} failed state={i} diagnostic_only_truth_rank={info["rank"]} '
                        f'truth_effective_p={len(active)} truth_condition={info["condition_number"]:.17g} '
                        f'truth_jacobian_disagreement={diff:.17g}')
                except Exception as exc:
                    log(f'{label} failed state={i} diagnostic_only_truth_jacobian_error='
                        +type(exc).__name__+': '+str(exc))
        require(False, 'Identifiable synthetic solver failure; first failing '+first)

    def algebra():
        require(np.max(abs(NODES[:, 0]**5-(1.25*NODES[:, 0]**3-.25*NODES[:, 0]))) < EPS,
                'x^5 alias')
        require(np.max(abs(NODES[:, 0]**6-(1.25*NODES[:, 0]**4-.25*NODES[:, 0]**2))) < EPS,
                'x^6 alias')
        require(len(RINGS['inner_axial']) == 4 and len(RINGS['inner_square']) == 8 and
                len(RINGS['inner_diagonal']) == 4 and len(RINGS['outer_boundary']) == 16, 'Ring definitions')
        samples = np.arange(50, dtype=float).reshape(1, 2, 25)/100
        values = np.einsum('nj,scj->snc', V, samples)
        recovered, good, defect = coefficients(values)
        require(good.all() and defect.max() < 1e-12 and np.max(abs(recovered-samples.reshape(1, 50))) < 1e-12,
                'Exact square coefficient solve')
        fake = dict(shards=[dict(profile='full5', filename='blocked.pkl', accommodation_index=0,
                    completion_status='pending', expected_row_count=255025, row_count=255025, sha256='0'*64)])
        refused = False
        try:
            manifest_entry(fake, 'full5', 'blocked.pkl', 0)
        except ValueError:
            refused = True
        require(refused, 'Incomplete manifest must refuse before opening')
        agg = nine.vector_summary(np.zeros((2, 25, 2)), np.array([False, True]))
        require(agg['rms_um'] == agg['max_um'] == np.inf and agg['failure_count'] == 1, 'Infinity accounting')
        require(p4_a0_data.safe(nine.json_values(agg))['max_um'] == 'Infinity', 'Strict Infinity spelling')
    check('alias_interpolation_manifest_infinity', algebra)

    def invalid_cases():
        p = np.zeros((3, 8))
        p[1, 6] = 2
        values, good = predict(p, 'A', c, H, q, BASE, NODES)
        require(np.array_equal(good, [True, False, True]), 'Per-state exception isolation')
        require(np.isfinite(values[[0, 2]]).all(), 'Unrelated states survive')
        raised = False
        try:
            joint.radial(np.array([[2., 0.]]), -1.)
        except ValueError:
            raised = True
        require(raised, 'Invalid radial branch refusal')
        spec = family(tuple(BLOCKS))
        require(not allow_eight(spec) and allow_eight(BASE), 'Excessive-p eight-node rejection')
        p = np.zeros((1, len(spec['names'])))
        target = forward_unchecked(p, 'A', c, H, q, spec, NODES)
        r = solve_pass(target[0], NODES, 'A', c, H, q, spec, p[0], 0, 1e-13, log, 'eta-degeneracy')
        require(r['eta_frozen'] and r['effective_p'] == len(p[0])-1 and r['parameters'][-1] == 0,
                'Eta frozen at zero shift')
        # Nonidentifiability must be explicit, never evidence of eta recovery.
        report['selftest_eta'] = dict(frozen=bool(r['eta_frozen']), rank=int(r['rank']),
            effective_p=int(r['effective_p']), claimed_recovery=False,
            justification='At current s=0, B(v-eta*s)-B(-eta*s) is independent of eta')
        log('selftest permitted degeneracy: '+report['selftest_eta']['justification'])
        require(r['frozen_gradient'] == 0., 'Zero-shift eta final gradient check')
        lag_spec = family(('lag',))
        truth = synthetic_parameters(lag_spec, 1)
        target = forward_unchecked(truth[None], 'A', c, H, q, lag_spec, NODES)[0]
        moving = attempt(target, NODES, 'A', c, H, q, lag_spec, np.zeros(len(truth)),
                         200, 1e-13, log, 'eta-zero-start', continuation=False)
        require_synthetic_fit(moving, 'eta-zero-start')
        require(not moving['failed'] and not moving['eta_frozen'] and
                np.max(abs(moving['parameters']-truth)) < 1e-10,
                'Eta must reactivate after a zero-shift start moves')
    check('branches_isolation_eta_and_parameter_budget', invalid_cases)

    def selection_rules():
        groups = dict(train=np.array([True, True, False]),
                      bin_low=np.array([True, False, True]), bin_high=np.array([False, True, False]),
                      bin_absent=np.zeros(3, bool))
        def result(norms, failed=None):
            errors = np.zeros((3, 25, 2))
            errors[..., 0] = np.asarray(norms)[:, None]
            return dict(errors=errors, fit=dict(failed=np.zeros(3, bool) if failed is None else failed))
        parent, better = result([2., 3., 1000.]), result([1.7, 2.5, 1e9])
        good, detail = gain_test(parent, better, training_bins(groups))
        require(good and detail['bin_absent']['passed'] is None, 'Training-only gain and absent bins')
        failed, _ = gain_test(parent, result([1.7, 2.9, 0.]), training_bins(groups))
        require(not failed, 'Every populated bin must pass')
        failed, _ = gain_test(parent, result([1.7, 2.5, 0.], np.array([True, False, False])), training_bins(groups))
        require(not failed, 'Failed state cannot pass gain')
        floor_parent, floor_trial = result([0., 5e-10, 0.]), result([1e-10, 1e-9, 1e9])
        passed, _ = gain_test(floor_parent, floor_trial, training_bins(groups))
        require(passed, 'Numerical-floor gain rule')
        require(training_target(result([.9, .9, 1e9]), groups['train']), 'Training-only stop')
        require(not training_target(result([.9, 1., 0.]), groups['train']), 'Strict <1 training stop')
    check('gain_floor_missing_bins_failures_training_only_stop', selection_rules)

    def tangent_checks():
        for order in ('A', 'B', 'C'):
            C0, J, P, info = tangent(order, c, H, q)
            require(info['rank'] == 8 and np.linalg.norm(P@J) < 1e-10 and
                    np.linalg.norm(P@P-P) < 1e-10, 'Tangent projector')
            maps[order+'_selftest_tangent'] = J
            maps[order+'_selftest_projector'] = P
            if order == 'C':
                continue
            x, y = NODES.T
            zero = np.zeros(25)
            r2 = x*x+y*y
            factor = (1+3*q*r2) if order == 'A' else (1+q*r2)**2
            expected_kx = np.stack((-x*x*factor, -x*y*factor), -1)
            expected_ky = np.stack((-x*y*factor, -y*y*factor), -1)
            expected, valid, _ = coefficients(np.stack((expected_kx, expected_ky)))
            require(valid.all() and np.max(abs(J[:, 6:8]-expected.T)) < 1e-6, 'Keystone supports/ties')
            for col, E in enumerate(([[1, 0], [0, 0]], [[0, 1], [0, 0]],
                                     [[0, 0], [1, 0]], [[0, 0], [0, 1]]), 2):
                Ed = NODES@np.asarray(E).T
                val = (1+q*r2)[:, None]*Ed
                if order == 'A':
                    val += 2*q*NODES*np.sum(NODES*Ed, axis=1)[:, None]
                expect, good, _ = coefficients(val[None])
                require(good.all() and np.max(abs(J[:, col]-expect[0])) < 1e-6, 'L supports/ties')
            absent = [(1, 4), (2, 3), (3, 2), (4, 1), (3, 4), (4, 3), (4, 4)]
            for i, j in absent:
                require(np.max(abs(J[[5*j+i, 25+5*j+i]])) < 1e-6, 'Absent tangent support')
    check('exact_first_order_tangent_support_and_ties', tangent_checks)

    specifications = [BASE]+[family((b,)) for b in BLOCKS]
    keys = tuple(BLOCKS)
    specifications += [family(keys[:i]) for i in range(2, len(keys)+1)]
    specifications += [family(keys, place) for place in ('pre', 'middle')]
    def expectation_table():
        names = {('+'.join(s['blocks']) or 'baseline')+'/'+s['placement'] for s in specifications}
        for order in ('A', 'B', 'C'):
            require(set(SELFTEST_RECOVERY_TABLE[order]) == names, 'Missing/extra recovery family')
            for row in SELFTEST_RECOVERY_TABLE[order].values():
                require(len(row) == 3 and all(k in SELFTEST_CASE_SETS for k in row),
                        'Missing recovery scope')
                require(all(SELFTEST_CASE_SETS[k]['count'] > 0 for k in row[:2]),
                        'Empty expected identifiable set')
        for cases in SELFTEST_CASE_SETS.values():
            require(cases['count'] == len(set(cases['names'])) and
                    set(cases['names']) <= set(SELFTEST_CASE_NAMES), 'Recovery names/count mismatch')
    check('fixed_recovery_expectation_table', expectation_table)
    # Identity plus all eight nonsymmetric states, for every own-order model.
    for order in ('A', 'B', 'C'):
        def zero_equivalence(order=order):
            parameters = np.stack([synthetic_parameters(BASE, j) for j in range(9)])
            expected = nine.forward_unchecked(parameters, order, c, H, q, nodes=NODES)
            if order in ('A', 'B'):
                # Independent single-state gaze-law adapter, evaluated on all
                # 25 nodes without using a nine-node hard-coded wrapper.
                for p, reference in zip(parameters, expected):
                    adapter = dict(zip(('a_x', 'a_y', 'm_xx', 'h_x', 'h_y', 'm_yy', 'k_x', 'k_y'), p))
                    config = joint.spec('G2', () if order == 'A' else ('barrel_before_tilt',))
                    theta = np.array([adapter.get(n, 0.) for n in config['names']])
                    normalized = joint.normalized_map(config, theta, [[1., 1.]], NODES, {'q0': q})
                    require(1000*np.max(np.linalg.norm(c+joint.camera(H*normalized[0])-reference, axis=-1)) < 1e-9,
                            'Imported normalized_map single-state equivalence')
            for spec in specifications:
                p = np.zeros((9, len(spec['names'])))
                p[:, :8] = parameters
                actual = forward_unchecked(p, order, c, H, q, spec, NODES)
                require(1000*np.max(np.linalg.norm(actual-expected, axis=-1)) < 1e-9,
                        'Zero-extra equivalence to imported baseline')
            old = nine.solve_chunk(expected[:, NINE], order, 'nine', c, H, q,
                                   gradient_limit=1e-13)
            new, _ = solve(expected[:, NINE], NODES[NINE], order, c, H, q, BASE,
                           log, 'nine-subset-'+order, limit=1e-13)
            require_synthetic_fit(old, 'imported-nine-subset-'+order)
            require_synthetic_fit(new, 'nine-subset-'+order)
            require(not old['failed'].any() and not new['failed'].any(), 'Nine-subset solver failures')
            require(np.max(abs(old['parameters']-new['parameters'])) < 1e-10,
                    'Imported old solver equivalence')
            # Labelled identity comparison, never used as a production rescue.
            identity, _ = solve(expected[:2], NODES, order, c, H, q, BASE, log,
                                'identity-comparison-'+order, limit=1e-13, identity=True)
            require_synthetic_fit(identity, 'identity-comparison-'+order)
            require(not identity['failed'].any() and np.max(abs(identity['parameters']-parameters[:2])) < 1e-10,
                    'Identity comparison recovery')
        check(order+'_zero_extra_and_old_solver_equivalence', zero_equivalence)

        def two_start_recovery(order=order):
            spec = family(('Q',))
            truth = synthetic_parameters(spec, 1)[None]
            observed = forward_unchecked(truth, order, c, H, q, spec, NODES)
            # Both starts and their continuations consume a single 200 cap.
            parent, _ = solve(observed, NODES, order, c, H, q, BASE, log,
                              'synthetic-parent-'+order, limit=1e-13)
            seed = expand_parent(parent['parameters'], BASE, spec)
            fitted, _ = solve(observed, NODES, order, c, H, q, spec, log,
                              'synthetic-two-start-'+order, parent=seed, limit=1e-13)
            require_synthetic_fit(fitted, 'synthetic-two-start-'+order,
                                  truth_context=(truth, NODES, order, spec))
            require(not fitted['failed'].any() and fitted['iterations'].max() <= 200,
                    'Parent/affine shared-budget fit')
            require('affine_attempt_gradient' in fitted and 'parent_attempt_parameters' in fitted,
                    'Both attempts must be recorded')
            require(np.max(abs(fitted['parameters']-truth)) < 1e-10, 'Two-start parameter recovery')
        check(order+'_D2_parent_affine_shared_budget', two_start_recovery)

        for si, spec in enumerate(specifications):
            def recovery(order=order, spec=spec, si=si):
                family_name = ('+'.join(spec['blocks']) or 'baseline')+'/'+spec['placement']
                expectations = SELFTEST_RECOVERY_TABLE[order][family_name]
                required_cases = {(scope, name)
                    for scope, key in zip(('all', 'noncentre', 'eight'), expectations)
                    if SELFTEST_CASE_SETS[key]['classification'] == 'recoverable'
                    for name in SELFTEST_CASE_SETS[key]['names']}
                required_rejections = {(scope, name)
                    for scope, key in zip(('all', 'noncentre', 'eight'), expectations)
                    if SELFTEST_CASE_SETS[key]['classification'] == 'production_rejected_ill_conditioned'
                    for name in SELFTEST_CASE_SETS[key]['names']}
                require(bool(required_cases), 'Empty identifiable recovery set')
                recovered_cases, rejected_cases = set(), set()
                truth = np.stack([synthetic_parameters(spec, j) for j in range(9)])
                observed = forward_unchecked(truth, order, c, H, q, spec, NODES)
                z, good = inverse(observed, c, H)
                C, reconstructed, defect = coefficients(z)
                require(good.all() and reconstructed.all() and defect.max() < 1e-12, 'Camera/D0 closure')
                for scope, expected_key in zip(('all', 'noncentre', 'eight'), expectations):
                    nodes = SCOPES[scope]
                    expected = SELFTEST_CASE_SETS[expected_key]
                    expected_names = expected['names']
                    require(expected['count'] == len(expected_names), 'Recovery table count')
                    detail = dict(order=order, family=family_name, scope=scope,
                                  expected_cases=expected_names, expected_count=expected['count'],
                                  classification=expected['classification'],
                                  recovered_cases=[], recovered_count=0, permitted_degeneracies=[])
                    recovery_checks.append(detail)
                    def degeneracy(case, justification):
                        detail['permitted_degeneracies'].append(dict(case=case, justification=justification))
                        log(f'selftest permitted degeneracy {order} {family_name} {scope} {case}: '+justification)
                    if scope == 'eight' and not allow_eight(spec):
                        require(expected_key == 'degenerate0', 'Required recovery case skipped by p budget')
                        for name in SELFTEST_CASE_NAMES:
                            degeneracy(name, 'Eight-node extension requires p<16; this family has p='+str(len(spec['names'])))
                        continue
                    if scope == 'eight' and spec['blocks'] == ('Q', 'prism', 'trefoil'):
                        if order == 'B':
                            require(expected_key == 'degenerate0', 'B eight-node cubic gauge expectation')
                            for name in SELFTEST_CASE_NAMES:
                                degeneracy(name, 'On eight nodes the odd cubic template has eight components but '
                                    'L/Q/trefoil have nine parameters; a linear template change is absorbed by '
                                    'L, k and the free prism vector in B')
                        else:
                            require(expected_key == ('production_rejected_ill_conditioned' if order == 'C' else 'nonzero8'),
                                    'Eight-node identity rank/condition expectation')
                            degeneracy('identity', 'At identity the four opposite node pairs have eight odd '
                                'components but L/Q/trefoil supply nine odd tangent columns')
                    else:
                        require(expected['count'] > 0, 'Empty identifiable recovery set')
                    if 'eta' in spec['names']:
                        degeneracy('identity:eta', 'At current s=0 eta has an exactly zero column; '
                            'freeze eta=0 and recover only the remaining parameters')
                    # Class membership is declared, never inferred from a failed
                    # solve. Evaluate the truth Jacobian before either start is
                    # solved; truth parameters are never used for initialization.
                    ill_records = []
                    is_ill = expected['classification'] == 'production_rejected_ill_conditioned'
                    detail['production_rejected_cases'] = []
                    detail['production_rejected_count'] = 0
                    if is_ill:
                        require(expected['count'] == 8 and expected_names == SELFTEST_CASE_NAMES[1:],
                                'Production rejection class must contain all eight non-identity cases')
                    def truth_function(x):
                        return forward_unchecked(x, order, c, H, q, spec, NODES[nodes])
                    truth_function.tail = (len(nodes), 2)
                    for name in expected_names:
                        j = SELFTEST_CASE_NAMES.index(name)
                        p = truth[j].copy()
                        active = np.arange(len(p))
                        if 'eta' in spec['names'] and np.max(abs(p[:2])) <= 1e-14:
                            eta = spec['names'].index('eta')
                            p[eta] = 0
                            active = active[active != eta]
                        J, diff = derivative(p, truth_function, active)
                        _, _, _, _, info = joint.scaled_svd(J)
                        log(f'selftest truth {order} {family_name} {scope} {name}: '
                            f'rank={info["rank"]} p={len(active)} '
                            f'condition={info["condition_number"]:.17g} production_limit={COND:.17g} '
                            f'jacobian_disagreement={diff:.17g}')
                        if is_ill:
                            # This declaration must be true independently of the
                            # production solver outcome; no recovery is asserted.
                            record = dict(order=order, family=family_name, scope=scope, case=name,
                                classification=expected['classification'],
                                justification=f'rank = p, truth-point condition {info["condition_number"]:.17g} '
                                    f'exceeds production limit COND ({COND:.17g}); production rejects it; '
                                    'no recovery claim is made', truth_rank=info['rank'],
                                p=len(p), truth_point_condition=info['condition_number'],
                                truth_jacobian_disagreement=diff,
                                truth_condition_definition='joint.scaled_svd: column-scaled model Jacobian at truth',
                                production_condition_limit=COND,
                                production_rejects=None, production_recovery_claimed=None)
                            ill_records.append(record)
                            ill_conditioned_checks.append(record)
                            log(f'selftest production_rejected_ill_conditioned {order} {family_name} '
                                f'{scope} {name}: '+record['justification'])
                            require(info['full_rank'] and info['rank'] == len(p),
                                    'Declared ill-conditioned case must have truth rank = p')
                            require(np.isfinite(info['condition_number']) and info['condition_number'] > COND,
                                    'Declared ill-conditioned case must exceed production condition limit at truth')
                        else:
                            require(info['full_rank'] and info['condition_number'] <= COND,
                                    'Undeclared rank-deficient or ill-conditioned truth in recovery table')
                    target = observed[:, nodes]
                    seed, seed_valid = affine_seed(target, NODES[nodes], order, c, H, q, spec)
                    require(seed_valid.all(), 'Synthetic affine seed')
                    # No numerical rank/condition result may remove an expected case.
                    good_ids = np.array([SELFTEST_CASE_NAMES.index(name) for name in expected_names], dtype=int)
                    ill_ids = good_ids if is_ill else np.array([], dtype=int)
                    production_ids = np.array([], dtype=int) if is_ill else good_ids
                    degenerate_ids = np.array([j for j, name in enumerate(SELFTEST_CASE_NAMES)
                                               if name not in expected_names], dtype=int)
                    parent_seed = None
                    if (scope == 'eight' and order in ('A', 'C') and
                            spec['blocks'] == ('Q', 'prism', 'trefoil')):
                        # The affine/zero-extra seed has k=alpha=0. Even with
                        # nonidentity L and nonzero s, its nine L/Q/trefoil
                        # columns are odd on four opposite pairs: rank <=14.
                        # It cannot take a full-rank first step. Use D2's
                        # same-target, same-subset parent fit (p=13), adding
                        # only zero trefoil amplitudes. No truth/centre seed,
                        # reduced basis, rank exception or identity rescue.
                        # solve records the affine attempt too and shares
                        # the existing 200 cap between the two child attempts.
                        parent_spec = family(('Q', 'prism'))
                        parent_fit, _ = solve(target, NODES[nodes], order, c, H, q,
                            parent_spec, log, f'synthetic-parent-{order}-{si}-{scope}',
                            limit=1e-13)
                        require_synthetic_fit(parent_fit, f'synthetic-parent-{order}-{si}-{scope}')
                        parent_seed = expand_parent(parent_fit['parameters'], parent_spec, spec)
                    answer, _ = solve(target, NODES[nodes], order, c, H, q, spec, log,
                                      f'synthetic-{order}-{si}-{scope}', parent=parent_seed, limit=1e-13,
                                      condition_limit=COND)
                    require_synthetic_fit(answer, f'synthetic-{order}-{si}-{scope}', production_ids,
                                          (truth, NODES[nodes], order, spec))
                    if len(ill_ids):
                        for j, record in zip(ill_ids, ill_records):
                            record.update(production_rejects=bool(answer['failed'][j] and answer['reason'][j] == 5),
                                production_reason=int(answer['reason'][j]),
                                production_reason_name=CODEBOOK[int(answer['reason'][j])],
                                production_recovery_claimed=bool(not answer['failed'][j]),
                                production_rank=int(answer['rank'][j]),
                                production_condition=float(answer['condition'][j]))
                            record['production_attempts'] = {
                                start_name: dict(failed=bool(answer[start_name+'_attempt_failed'][j]),
                                    reason=int(answer[start_name+'_reason'][j]),
                                    reason_name=CODEBOOK[int(answer[start_name+'_reason'][j])],
                                    recovery_claimed=bool(not answer[start_name+'_attempt_failed'][j]),
                                    rank=int(answer[start_name+'_attempt_rank'][j]),
                                    condition=float(answer[start_name+'_attempt_condition'][j]))
                                for start_name in ('parent', 'affine')}
                            log(f'selftest production_rejected_ill_conditioned {order} {family_name} {scope} '
                                f'{record["case"]} truth_point_condition={record["truth_point_condition"]:.17g} '
                                f'production_rejects={record["production_rejects"]} '
                                f'production_reason={record["production_reason"]}({record["production_reason_name"]})')
                        require(np.all(answer['failed'][ill_ids]) and np.all(answer['reason'][ill_ids] == 5),
                                'Declared ill-conditioned cases must be rejected at production limit with rank_or_condition')
                        for start_name in ('parent', 'affine'):
                            require(np.all(answer[start_name+'_attempt_failed'][ill_ids]) and
                                    np.all(answer[start_name+'_reason'][ill_ids] == 5),
                                    'Each production attempt must reject with rank_or_condition and claim no recovery')
                        detail.update(production_rejected_cases=list(expected_names),
                                      production_rejected_count=len(ill_ids))
                        rejected_cases.update((scope, name) for name in expected_names)
                        require(detail['production_rejected_count'] == expected['count'] == 8,
                                'Production rejection count must equal all eight declared cases')
                    if len(production_ids):
                        recovered, valid = predict(answer['parameters'][production_ids], order, c, H, q, spec, NODES)
                        require(valid.all() and 1000*np.max(np.linalg.norm(recovered-observed[production_ids], axis=-1)) < 1e-9,
                                'Synthetic prediction/withheld-centre recovery >=1e-9 um')
                        require(np.max(abs(answer['parameters'][production_ids]-truth[production_ids])) < 1e-10,
                                'Synthetic parameter recovery >=1e-10')
                        zr, ok = inverse(recovered, c, H)
                        cr, okc, _ = coefficients(zr)
                        require(ok.all() and okc.all() and 1000*H*np.max(abs(cr-C[production_ids])) < 1e-9,
                                'Corrected coefficient recovery >=1e-9 um')
                        detail.update(recovered_cases=list(expected_names), recovered_count=len(production_ids))
                        detail['recovery_condition_limit'] = COND
                        recovered_cases.update((scope, name) for name in expected_names)
                        require(detail['recovered_count'] == expected['count'] > 0,
                                'Recovery count must be nonzero and equal the table')
                    if len(degenerate_ids):
                        require(np.all(answer['failed'][degenerate_ids]),
                                'Structurally deficient synthetic fit must not be claimed recovered')
                    if scope != 'all':
                        corrupted = observed.copy()
                        corrupted[:, 12] += [1e6, -1e6]
                        other_seed, other_valid = affine_seed(corrupted[:, nodes], NODES[nodes], order, c, H, q, spec)
                        require(np.array_equal(seed, other_seed) and np.array_equal(seed_valid, other_valid),
                                'Excluded centre changes start')
                        other_parent_seed = None
                        if parent_seed is not None:
                            # Refit the parent from corrupted subset targets,
                            # rather than reusing a seed that might hide leakage.
                            other_parent_fit, _ = solve(corrupted[:, nodes], NODES[nodes], order,
                                c, H, q, parent_spec, log, 'corrupted-centre-parent', limit=1e-13)
                            other_parent_seed = expand_parent(other_parent_fit['parameters'], parent_spec, spec)
                            require(np.array_equal(parent_seed, other_parent_seed, equal_nan=True) and
                                    np.array_equal(parent_fit['reason'], other_parent_fit['reason']),
                                    'Excluded centre changes parent fit/start')
                        repeated, _ = solve(corrupted[:, nodes], NODES[nodes], order, c, H, q, spec,
                                              log, 'corrupted-centre', parent=other_parent_seed, limit=1e-13)
                        require(np.array_equal(answer['parameters'], repeated['parameters'], equal_nan=True) and
                                np.array_equal(answer['reason'], repeated['reason']), 'Excluded centre changes fit')
                require(bool(required_cases) and recovered_cases == required_cases,
                        'Empty identifiable set or a declared recovery case was skipped')
                require(rejected_cases == required_rejections,
                        'A declared production rejection case was skipped')
                log(f'selftest recovery {order} {family_name}: recovered={len(recovered_cases)} '
                    f'expected={len(required_cases)}')
                maps[f'{order}_selftest_case{si}_truth_parameters'] = truth
            check(f'{order}_synthetic_family_{si}_{spec["placement"]}', recovery)
    report['selftest'] = dict(checks=checks, failures=failures, passed=not failures,
        recovery_case_sets=SELFTEST_CASE_SETS, recovery_expectations=SELFTEST_RECOVERY_TABLE,
        recovery_checks=recovery_checks,
        production_rejected_ill_conditioned=len(ill_conditioned_checks),
        ill_conditioned_cases=ill_conditioned_checks,
        production_condition_limit=COND,
        condition_policy='Extension steps whose per-state condition exceeds the production limit '
            'are rejected by the solver (counted as failures), never relaxed',
        gradient_limit=1e-13, parameter_tolerance=1e-10, prediction_um_tolerance=1e-9,
        corrected_coefficient_um_tolerance=1e-9,
        fixed_tolerance_scope='All recoverable classes; production_rejected_ill_conditioned asserts '
            'truth rank/condition and production rejection only, with no recovery claim',
        inputs_read=False, note='A/B/C use their own extended forward-generated observations')
    report['status'] = 'PASS' if not failures else 'FAIL'
    report['numerical_gates_passed'] = not failures
    log(report['status']+' selftest failures='+str(len(failures))+
        (' first_failure='+failures[0] if failures else ''))


def initial_report(args):
    return dict(status='initialized', mode='selftest' if args.selftest else 'quick' if args.quick else 'full',
        scope='noise-free design traces; unaudited', numpy_version=np.__version__, errors=[],
        calibration={}, timings={}, camera=dict(F_mm=joint.F, K1=joint.K1, K2=joint.K2,
                                               fitted_parameters=0, principal_point_mm=[0., 0.]),
        settings=dict(processes=1, solver_chunk_max=256, iterations_total=200,
            candidate_starts='parent attempt up to 100, affine attempt consumes remaining budget; both recorded',
            production_gradient=1e-10, selftest_gradient=1e-13, FD=FD,
            jacobian_half_step_limit=5e-5, condition_limit=COND,
            damped_step='-D^-1 V diag(sigma/(sigma^2+lambda)) U^T r',
            damping_initial='1e-10*sigma_max^2', retries=18, damping_multiplier=10,
            damping_floor='EPS*sigma_max^2', relative_step=1e-12, relative_objective=1e-14,
            rank_cutoff='sigma_max*max(J.shape)*EPS', extended_fold_lattice='11x11; sampled, not global proof',
            eta_zero_shift_threshold=1e-14, eta_scope='Recheck current normalized shift each iteration; '
                'freeze eta=0 at zero shift, check frozen-column gradient before convergence; no eta recovery claim there',
            full_pilot_attestation=bool(args.pilot_gates_passed)),
        definitions=dict(nodes='-field labels; y-major 25 nodes; centre index 12',
            gaze='w=(BDE,-ADE) deg, u=w/12.5; independent state parameters; no gaze recovery',
            baseline_orders=dict(A='c+Cam(H*(R(T(d))+s))', B='c+Cam(H*(T(R(d))+s))',
                                 C='c+H*(s+T(Cam(H*R(d))/H))'),
            extras='Q symmetric correction; P=alpha*r^2; h=(x^3-3xy^2,3x^2y-y^3); '
                   'trefoil=t1*h+t2*Jh; beta*v*r^4; W=gamma*x^2*y^2; '
                   'B_eta(v)=B(v-eta*s)-B(-eta*s); prism post/pre/middle exclusive',
            metrics='1000*(prediction-observation); RMS=sqrt(mean squared Euclidean node norm); '
                    'max=worst Euclidean node norm; equal state/node weights',
            failures='Required failures force compliance RMS/max Infinity; retain best finite diagnostic iterates',
            coefficient_units='1000*H*delta_C; not point error; local sensor equivalents use V and camera Jacobian',
            rings={k: dict(node_count=len(v), node_indices=v) for k, v in RINGS.items()},
            ring_note='Overlapping field diagnostics, not a partition; inner axial Euclidean radius .5 has FOUR nodes',
            selection='Training-only centre-assisted D2 family selection; holdout centre truth unused by selection',
            physical_interpretation='Shape hypotheses only; no unique physical aberration identification',
            polynomial_compression='Training CV in parameter space, judged in frozen forward sensor space; not trace residual'),
        codebooks=dict(status=CODEBOOK, start_paths=nine.START_PATHS,
                       best_iterate_stage={1: 'first', 2: 'restricted', 3: 'released'},
                       attempt_selected={1: 'parent', 2: 'affine'}))


def text_report(report):
    lines = ['noise-free design traces; unaudited', 'P4 5x5 D0/D1/D2 diagnostic']
    def visit(value, prefix=''):
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, prefix+'.'+str(key) if prefix else str(key))
        elif isinstance(value, (list, tuple)):
            # Summaries/trials only: no per-state JSON or observations.
            if all(not isinstance(x, (dict, list, tuple)) for x in value):
                lines.append(prefix+' = '+json.dumps(value, ensure_ascii=True))
            else:
                for i, child in enumerate(value):
                    visit(child, prefix+'.'+str(i))
        else:
            lines.append(prefix+' = '+str(value))
    visit(p4_a0_data.safe(nine.json_values(report)))
    for case in report.get('selftest', {}).get('ill_conditioned_cases', []):
        lines.append('production_rejected_ill_conditioned=1 '+case['order']+' '+case['family']+' '+case['scope']
            +' '+case['case']+' truth_point_condition='+str(case['truth_point_condition'])
            +' production_rejects='+str(case['production_rejects'])
            +' production_recovery_claimed='+str(case['production_recovery_claimed']))
    return '\n'.join(lines)+'\n'


def store_codebooks(report, maps):
    """Numeric UTF-8 bytes/offsets keep NPZ codebooks readable without pickle."""
    for name, book in report['codebooks'].items():
        codes = sorted(book)
        words = [book[code].encode('utf-8') for code in codes]
        maps['codebook_'+name+'_codes'] = np.asarray(codes, dtype=np.int16)
        maps['codebook_'+name+'_utf8'] = np.frombuffer(b''.join(words), dtype=np.uint8)
        maps['codebook_'+name+'_offsets'] = np.cumsum([0]+[len(w) for w in words], dtype=np.int32)


def finalize(report, maps, log, input_path):
    # Fixed artifacts only, direct writes, no temporary file or cache. Each
    # available artifact is attempted independently after a stage failure.
    errors = []
    for name in ('maps.npz', 'results.json', 'report.txt'):
        try:
            guard_paths(input_path)
            path = OUT/name
            if name == 'maps.npz':
                maps.close()
            elif name == 'results.json':
                path.write_text(json.dumps(p4_a0_data.safe(nine.json_values(report)),
                                           indent=2, allow_nan=False)+'\n', encoding='utf-8')
            else:
                path.write_text(text_report(report), encoding='utf-8')
        except Exception as exc:
            error = dict(stage='write_'+name, error=type(exc).__name__+': '+str(exc))
            errors.append(error)
            report['errors'].append(error)
            report['status'] = 'failed'
            print(str(error), file=sys.stderr, flush=True)
    log('final status='+report['status'])
    return int(bool(errors or report['errors'] or not report.get('numerical_gates_passed', False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--selftest', action='store_true')
    mode.add_argument('--quick', action='store_true')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--pilot-gates-passed', action='store_true',
                        help='Attest the audited pilot passed input/numerical gates; required for full mode')
    args = parser.parse_args()
    require(args.shard_index >= 0, 'Negative shard index')
    if args.selftest and args.input is not None:
        parser.error('--selftest reads no input; do not supply --input')
    default_name = (f'p4_pilot5_detail_A{args.shard_index:02d}.pkl' if args.quick else
                    f'p4_raytrace_5x5_101x101_12p5_A{args.shard_index:02d}.pkl')
    args.input_path = None if args.selftest else (args.input or DATA/default_name).absolute()
    report = initial_report(args)
    try:
        log = Progress(args.input_path)
    except Exception as exc:
        print('Unsafe output paths: '+str(exc), file=sys.stderr)
        return 1
    maps = NumericMaps(args.input_path, log)
    try:
        store_codebooks(report, maps)
        if args.selftest:
            run_selftest(report, maps, log)
        else:
            require(args.quick or args.pilot_gates_passed,
                    'Full mode needs --pilot-gates-passed after audit and successful pilot input/numerical gates')
            t = time.perf_counter()
            observed, c, H, q, groups = load_shard(args, report, maps, log)
            report['timings']['load_total_s'] = time.perf_counter()-t
            analyze(observed, c, H, q, groups, report, maps, log, args.quick)
    except Exception as exc:
        report['errors'].append(dict(stage='analysis', error=type(exc).__name__+': '+str(exc)))
        report['status'] = 'FAIL' if args.selftest else 'incomplete_or_solver_limited'
        report['numerical_gates_passed'] = False
        report['decision'] = dict(incomplete_or_solver_limited=True, full_grid_compliance=False)
        log(report['status']+': '+str(exc))
    return finalize(report, maps, log, args.input_path)


if __name__ == '__main__':
    raise SystemExit(main())
