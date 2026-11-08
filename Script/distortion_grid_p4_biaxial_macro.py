"""Collect P4 two-axis (ADE x BDE) distortion grids with one CODE V macro per accommodation.

Production replacement for the per-state COM loop in distortion_grid_p4_biaxial.py
(kept as the slow reference). Python loops over accommodations; for each one it
reloads the P4 lens, applies the GRIN accommodation model, computes the angular
field of view, regenerates a Macro-PLUS sequence for the whole ADE x BDE sweep,
runs it in a single ``cv.Command`` call and parses the CSV it writes.

Per state the macro repeats exactly the setup of
distortion_grid.calculate_grid_at_biaxial_rotation (ADE/BDE on RC, THI Cornea_ENT_D
0, SET VIG, XAN/YAN restore, XOB/YOB write-back of field 2) and traces the same
rays in the same order: center chief ray (field 1), near-axis 0.05/0.05 ray
(field 0), then the 3x3 grid (rows relative_y = 1..-1, columns relative_x = -1..1).
Every rayrsi status and the ADE/BDE/THI readbacks are written out; Python rejects
any readback mismatch and derives paraxial/radial/tangential columns with the
same expressions as the slow path.

Failed rays (nonzero rayrsi status = failed surface) are real optical limits of
the model (the slow path fails at the same states). They do not abort the sweep;
the affected values become NaN and the x/y CODE V reports for a failed ray (stale
values of an earlier ray) are never used:
  center ray fails     -> all value columns of the state's 9 rows NaN
                          (center_absolute, paraxial, real, radial, tangential);
  near-axis ray fails  -> paraxial_x/y, radial, tangential NaN for all 9 rows;
  grid ray k fails     -> row k's real_x/y, radial, tangential NaN.
ADE/BDE/accommodation/z/field_*_relative are always valid. Every value not
masked by these rules must be finite, and every masked value must be NaN.

Per shard, group_<index>.failures.npz holds the int16 status array
(n_ade, n_bde, 11; columns STATUS_COLUMNS); its sha256 and failure counts are
recorded in metadata.json and re-checked on --resume. On publication
<output>/failure_map.npz (accommodation, ADE, BDE axes + status of shape
n_acc x n_ade x n_bde x 11) is written next to distortion_grid.pkl and a
failure summary is printed.

A failed COM-level RAYRSI (slow path, not the macro) leaves a deferred
"Error: Ray missed surface ..." message that CODE V prints in the response of
the NEXT cv.Command (e.g. the next ``res``). Before each lens reload the session
is drained with a no-op command that tolerates only those known ray-failure
messages; load_lens itself stays strict.

Output schema, sharding (one shard per accommodation), checkpoint metadata and
the published pickle mirror distortion_grid_p4_biaxial.py.

Parallel sessions (--workers N, default 1)
------------------------------------------
N=1 runs everything in this process with one CODE V session (original behavior).
N>1 starts min(N, shards to run) worker processes (multiprocessing "spawn"); each
does its own CoInitialize / Dispatch("CodeV.Application") / StartCodeV with
CommandTimeout=-1 and StopCodeV in a finally block, and uses its own work
directory <checkpoint>/work/worker_<k>/ for the generated .seq and CSV. The
coordinator (main process) hands out accommodation indices one at a time from a
shared queue (ascending order); a worker sweeps the accommodation, validates it
and writes its shard group_<index>.pkl atomically. Only the coordinator writes
metadata.json, once per completed shard (completed_groups sorted by index, plus
completed_accommodation_indices).

If a worker fails, no further shards are dispatched, the other workers finish
their current shard (recorded in metadata), the error is reported and the
checkpoint is left for --resume; nothing is published. A shard file written
without a metadata entry (e.g. coordinator killed) is simply recomputed.

--resume accepts any set of completed accommodation indices (not only a
prefix), re-validates every recorded shard (hash, contents, accommodation) and
runs only the missing ones. The final pickle is always concatenated in
accommodation order, so results do not depend on the worker count.

Usage:
  python distortion_grid_p4_biaxial_macro.py [sweep options] [--workers 4]
  python distortion_grid_p4_biaxial_macro.py [same options] --resume [--workers 4]
  --stop-after-accommodations K dispatches at most K new shards, then stops.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import pickle
import queue
import re
import shutil
import stat
import time
import traceback
from pathlib import Path
from typing import Sequence

import numpy as np

from distortion_grid import angular_fov
from distortion_grid_p4 import (
    LENS_FILE,
    apply_accommodation,
    load_lens,
    read_accommodation_baseline,
)
from distortion_grid_p4_biaxial import (
    GRID_FIELDS,
    Z_MM,
    atomic_json,
    atomic_pickle,
    file_sha256,
    make_metadata,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "data" / "distortion_grid_p4_biaxial_macro"

# v2: failed rays -> NaN + failure sidecars (v1 checkpoints have no sidecars).
BACKEND = "codev_macro_v2"
GRID_LINES = 3
# CODE V COM API Reference Guide, Property CommandTimeout (p. 36):
# "CommandTimeout As Long ... Default Value -1 (Commands do not time out)".
COMMAND_TIMEOUT_UNLIMITED = -1

READBACK_TOL_DEG = 1e-7
READBACK_TOL_Z_MM = 1e-8
LINSPACE_TOL_DEG = 1e-9

NEAR_AXIS = (0.05, 0.05)


def grid_fields(grid_lines: int) -> list[tuple[float, float]]:
    """(relative_x, relative_y) in the slow path's row order (i over y, j over x)."""
    fields = []
    for i in range(grid_lines):
        relative_y = 1.0 - 2.0 * i / (grid_lines - 1)
        for j in range(grid_lines):
            relative_x = -1.0 + 2.0 * j / (grid_lines - 1)
            fields.append((relative_x, relative_y))
    return fields


N_GRID = GRID_LINES**2
# CSV columns written by the macro, one row per (ADE, BDE) state.
CSV_COLUMNS = (
    ["ia", "ib", "ade", "bde", "ade_rb", "bde_rb", "thi_rb",
     "st_c", "cx", "cy", "st_n", "nx", "ny"]
    + [name for k in range(1, N_GRID + 1) for name in (f"st_{k}", f"gx_{k}", f"gy_{k}")]
)
COL = {name: index for index, name in enumerate(CSV_COLUMNS)}
# Ray statuses in trace order: center, near-axis, grid rays 1..9 (failure-map last axis).
STATUS_COLUMNS = ("st_c", "st_n") + tuple(f"st_{k}" for k in range(1, N_GRID + 1))
# (status column, x column, y column) for each traced ray.
RAY_COLUMNS = (("st_c", "cx", "cy"), ("st_n", "nx", "ny")) + tuple(
    (f"st_{k}", f"gx_{k}", f"gy_{k}") for k in range(1, N_GRID + 1)
)
ALWAYS_VALID_FIELDS = (
    "eye_rotation_deg", "eye_rotation_x_deg", "accommodation_d", "z_mm",
    "field_x_relative", "field_y_relative",
)
STATUS_DTYPE = np.int16
# Deferred diagnostics of a failed COM-level RAYRSI, printed by the next cv.Command.
DEFERRED_RAY_ERROR = re.compile(
    r"^Error: (Ray missed surface|Total reflection at surface)\s+\d+ for zoom position\s+\d+$"
)
DRAIN_MAX_COMMANDS = 5


def remove_tree(path: Path) -> None:
    """shutil.rmtree that clears the ReadOnly attribute OneDrive sets on new folders."""

    def clear_readonly_and_retry(func, failed_path, _exc):
        os.chmod(failed_path, stat.S_IWRITE)
        func(failed_path)

    shutil.rmtree(path, onexc=clear_readonly_and_retry)


def configure_session(cv) -> int:
    """Apply COM properties allowed before StartCodeV; returns the timeout readback."""
    cv.CommandTimeout = COMMAND_TIMEOUT_UNLIMITED
    timeout = int(cv.CommandTimeout)
    if timeout != COMMAND_TIMEOUT_UNLIMITED:
        raise RuntimeError(f"CommandTimeout readback {timeout} != {COMMAND_TIMEOUT_UNLIMITED}")
    cv.MaxTextBufferSize = 1000000
    cv.StartingDirectory = str(LENS_FILE.parent)
    return timeout


def _linspace_literals(values: Sequence[float]) -> tuple[str, str]:
    """Start and step literals reproducing np.linspace as start + i*step."""
    values = np.asarray(values, dtype=np.float64)
    start = float(values[0])
    step = 0.0 if values.size == 1 else float((values[-1] - values[0]) / (values.size - 1))
    return repr(start), repr(step)


def write_macro(
    seq_path: Path,
    csv_path: Path,
    ade_values: Sequence[float],
    bde_values: Sequence[float],
    fov_x: float,
    fov_y: float,
) -> None:
    for path in (seq_path, csv_path):
        if '"' in str(path):
            raise ValueError(f"Path cannot contain a double quote for CODE V: {path}")
    n_ade, n_bde = len(ade_values), len(bde_values)
    ade0, ade_step = _linspace_literals(ade_values)
    bde0, bde_step = _linspace_literals(bde_values)
    grid_vars = [f"^{name}" for name in CSV_COLUMNS[13:]]
    lines = [
        "! Generated by distortion_grid_p4_biaxial_macro.py -- do not edit.",
        "! ADE x BDE sweep at one accommodation; per-state setup mirrors",
        "! distortion_grid.calculate_grid_at_biaxial_rotation.",
        "LCL num ^ia ^ib ^ade ^bde ^ade_rb ^bde_rb ^thi_rb ^input(4) ^xob2 ^yob2",
        "LCL num ^st_c ^cx ^cy ^st_n ^nx ^ny",
        "LCL num " + " ".join(grid_vars),
        f'OPE NEW U^1 "{csv_path}"',
        "^input(1) == 0",
        "^input(2) == 0",
        "OUT N",
        f"FOR ^ia 0 {n_ade - 1} 1",
        f"^ade == {ade0} + ^ia*{ade_step}",
        f"FOR ^ib 0 {n_bde - 1} 1",
        f"^bde == {bde0} + ^ib*{bde_step}",
        'ADE S"RC" ^ade',
        'BDE S"RC" ^bde',
        f'THI S"Cornea_ENT_D" {Z_MM!r}',
        "SET VIG",
        '^ade_rb == (ade s"RC")',
        '^bde_rb == (bde s"RC")',
        '^thi_rb == (thi s"Cornea_ENT_D")',
        f"XAN 0 {fov_x:.12g}",
        f"YAN 0 {fov_y:.12g}",
        "^xob2 == (xob f2 z1)",
        "^yob2 == (yob f2 z1)",
        "yob 0 ^yob2",
        "xob 0 ^xob2",
        # Center chief ray: trace_rsi(cv, 0, 0, field_number=1)
        "^input(3) == 0",
        "^input(4) == 0",
        "^st_c == rayrsi(1,1,1,0,^input)",
        "^cx == (x si)",
        "^cy == (y si)",
        # Near-axis 5% reference ray: trace_rsi(cv, 0.05, 0.05)
        f"^input(3) == {NEAR_AXIS[0]!r}",
        f"^input(4) == {NEAR_AXIS[1]!r}",
        "^st_n == rayrsi(1,1,0,0,^input)",
        "^nx == (x si)",
        "^ny == (y si)",
    ]
    for k, (relative_x, relative_y) in enumerate(grid_fields(GRID_LINES), start=1):
        lines += [
            f"^input(3) == {relative_x!r}",
            f"^input(4) == {relative_y!r}",
            f"^st_{k} == rayrsi(1,1,0,0,^input)",
            f"^gx_{k} == (x si)",
            f"^gy_{k} == (y si)",
        ]
    wri_args = " ".join(f"^{name}" for name in CSV_COLUMNS)
    qfmt = " ".join(["'3e.17e'"] * len(CSV_COLUMNS))
    lines += [
        f'WRI U^1 Q"{qfmt}" {wri_args}',
        "END FOR",
        "END FOR",
        "OUT Y",
        "CLO U^1",
        "RTN",
    ]
    seq_path.parent.mkdir(parents=True, exist_ok=True)
    seq_path.write_text("\n".join(lines) + "\n", encoding="ascii")


def run_macro(cv, seq_path: Path, csv_path: Path) -> str:
    if csv_path.exists():
        csv_path.unlink()
    response = cv.Command(f'run "{seq_path}";go')
    bad = [
        line for line in response.splitlines()
        if line.lstrip().lower().startswith(("error", "*** error"))
    ]
    if bad:
        raise RuntimeError(f"CODE V macro reported errors: {bad!r}\nFull response: {response!r}")
    if not csv_path.is_file():
        raise RuntimeError(f"Macro produced no CSV ({csv_path}); response: {response!r}")
    return response


def parse_csv(csv_path: Path, n_ade: int, n_bde: int) -> np.ndarray:
    n_cols = len(CSV_COLUMNS)
    lines = [line for line in csv_path.read_text(encoding="ascii").splitlines() if line.strip()]
    if len(lines) != n_ade * n_bde:
        raise RuntimeError(
            f"Macro CSV {csv_path} has {len(lines)} rows; expected {n_ade * n_bde} "
            f"({n_ade} ADE x {n_bde} BDE) -- truncated or duplicated output"
        )
    table = np.empty((len(lines), n_cols), dtype=np.float64)
    for row, line in enumerate(lines):
        parts = line.split()
        if len(parts) != n_cols:
            raise RuntimeError(
                f"Macro CSV row {row} has {len(parts)} columns; expected {n_cols}: {line[:200]!r}"
            )
        values = []
        for part in parts:
            try:
                values.append(float(part))
            except ValueError:
                # Unparseable token -> NaN; accepted below only for x/y of a failed ray.
                values.append(math.nan)
        table[row] = values
    # Index, readback and status columns must always be finite...
    required = np.zeros(table.shape, dtype=bool)
    required[:, :COL["thi_rb"] + 1] = True
    for st_name, x_name, y_name in RAY_COLUMNS:
        required[:, COL[st_name]] = True
        # ...and the x/y of every ray whose status is 0 (a failed ray's x/y are stale).
        ok = table[:, COL[st_name]] == 0
        required[:, COL[x_name]] = ok
        required[:, COL[y_name]] = ok
    bad = np.argwhere(required & ~np.isfinite(table))
    if bad.size:
        row, col = bad[0]
        raise RuntimeError(
            f"Macro CSV has {len(bad)} nonfinite required value(s); first: row {row} "
            f"column {CSV_COLUMNS[col]}"
        )
    expected_ia = np.repeat(np.arange(n_ade), n_bde)
    expected_ib = np.tile(np.arange(n_bde), n_ade)
    if not (np.array_equal(table[:, COL["ia"]], expected_ia)
            and np.array_equal(table[:, COL["ib"]], expected_ib)):
        raise RuntimeError("Macro CSV (ADE, BDE) index sequence is not the expected nested order")
    return table


def check_table(table: np.ndarray, ade_values: Sequence[float], bde_values: Sequence[float]) -> np.ndarray:
    """Hard-fail on sweep/readback mismatches; return ray statuses (n_states, 11) as int16."""
    ade = np.asarray(ade_values, dtype=np.float64)[table[:, COL["ia"]].astype(int)]
    bde = np.asarray(bde_values, dtype=np.float64)[table[:, COL["ib"]].astype(int)]
    checks = (
        ("macro ADE vs linspace", table[:, COL["ade"]], ade, LINSPACE_TOL_DEG),
        ("macro BDE vs linspace", table[:, COL["bde"]], bde, LINSPACE_TOL_DEG),
        ("ADE readback", table[:, COL["ade_rb"]], ade, READBACK_TOL_DEG),
        ("BDE readback", table[:, COL["bde_rb"]], bde, READBACK_TOL_DEG),
        ("Cornea_ENT_D readback", table[:, COL["thi_rb"]], np.full(len(table), Z_MM), READBACK_TOL_Z_MM),
    )
    for label, actual, expected, tol in checks:
        err = np.abs(actual - expected)
        if np.any(err > tol):
            row = int(np.argmax(err))
            raise RuntimeError(
                f"{label} mismatch {err[row]:.3g} > {tol:g} at ADE={ade[row]!r}, BDE={bde[row]!r}"
            )
    status = table[:, [COL[name] for name in STATUS_COLUMNS]]
    info = np.iinfo(STATUS_DTYPE)
    if not np.all((status == np.round(status)) & (status >= info.min) & (status <= info.max)):
        raise RuntimeError("Macro CSV ray status values are not integers in the int16 range")
    return status.astype(STATUS_DTYPE)


def nan_masks(status: np.ndarray) -> dict[str, np.ndarray]:
    """Expected NaN mask per GRID_FIELDS column, flat in record order.

    status: (..., 11) ray statuses per state (STATUS_COLUMNS order); each state
    expands to its N_GRID rows.
    """
    status = np.asarray(status).reshape(-1, len(STATUS_COLUMNS))
    center_fail = np.repeat(status[:, 0] != 0, N_GRID)
    near_fail = np.repeat(status[:, 1] != 0, N_GRID)
    grid_fail = (status[:, 2:] != 0).reshape(-1)
    paraxial = center_fail | near_fail
    real = center_fail | grid_fail
    distortion = paraxial | grid_fail
    masks = {field: np.zeros(status.shape[0] * N_GRID, dtype=bool) for field in ALWAYS_VALID_FIELDS}
    masks.update({
        "paraxial_x_mm": paraxial, "paraxial_y_mm": paraxial,
        "center_absolute_x_mm": center_fail, "center_absolute_y_mm": center_fail,
        "real_x_mm": real, "real_y_mm": real,
        "radial_distortion_pct": distortion, "tangential_distortion_pct": distortion,
    })
    assert set(masks) == set(GRID_FIELDS)
    return masks


def check_nan_pattern(data: np.ndarray, status: np.ndarray, label: str) -> None:
    """NaN exactly where the failure mask says, finite everywhere else."""
    masks = nan_masks(status)
    for field in GRID_FIELDS:
        values = data[field]
        mask = masks[field]
        if values.shape != mask.shape:
            raise ValueError(f"{label}: failure mask size {mask.size} does not match {values.size} records")
        if not np.array_equal(np.isnan(values), mask) or not np.all(np.isfinite(values[~mask])):
            n_bad = int(np.count_nonzero((np.isnan(values) != mask) | (~mask & ~np.isfinite(values))))
            raise ValueError(
                f"{label}: {field} has {n_bad} value(s) whose NaN/nonfinite pattern does not "
                f"match the ray-failure mask"
            )


def derive_rows(
    table: np.ndarray,
    ade_values: Sequence[float],
    bde_values: Sequence[float],
    accommodation_d: float,
    status: np.ndarray,
) -> np.ndarray:
    """Build GRID_FIELDS records with calculate_grid_at_biaxial_rotation's arithmetic.

    x/y of failed rays are replaced by NaN before any arithmetic, and the failure
    mask is then applied explicitly (the center row's radius_squared branch would
    otherwise give 0.0 instead of NaN).
    """
    grid_lines = GRID_LINES
    fields = grid_fields(grid_lines)
    table = table.copy()
    for st_name, x_name, y_name in RAY_COLUMNS:
        failed = table[:, COL[st_name]] != 0
        table[failed, COL[x_name]] = math.nan
        table[failed, COL[y_name]] = math.nan
    records = []
    for raw in table.tolist():
        ade_deg = float(ade_values[int(raw[COL["ia"]])])
        bde_deg = float(bde_values[int(raw[COL["ib"]])])
        center = (raw[COL["cx"]], raw[COL["cy"]])
        near_axis = (raw[COL["nx"]], raw[COL["ny"]])
        x_corner = near_axis[0] - center[0]
        y_corner = near_axis[1] - center[1]
        delta_x = 40.0 * x_corner / (grid_lines - 1)
        delta_y = 40.0 * y_corner / (grid_lines - 1)
        k = 0
        for i in range(grid_lines):
            relative_y = 1.0 - 2.0 * i / (grid_lines - 1)
            paraxial_y = ((grid_lines - 1) / 2.0 - i) * delta_y
            for j in range(grid_lines):
                relative_x = -1.0 + 2.0 * j / (grid_lines - 1)
                paraxial_x = (j - (grid_lines - 1) / 2.0) * delta_x
                k += 1
                assert fields[k - 1] == (relative_x, relative_y)
                real = (raw[COL[f"gx_{k}"]], raw[COL[f"gy_{k}"]])
                real_x = real[0] - center[0]
                real_y = real[1] - center[1]
                residual_x = real_x - paraxial_x
                residual_y = real_y - paraxial_y
                radius_squared = paraxial_x**2 + paraxial_y**2
                if radius_squared < 1e-20:
                    radial = tangential = 0.0
                else:
                    radial = (
                        residual_x * paraxial_x + residual_y * paraxial_y
                    ) / radius_squared * 100.0
                    tangential = (
                        residual_x * paraxial_y - residual_y * paraxial_x
                    ) / radius_squared * 100.0
                records.append((
                    ade_deg, bde_deg, float(accommodation_d), Z_MM,
                    relative_x, relative_y, paraxial_x, paraxial_y,
                    center[0], center[1], real_x, real_y, radial, tangential,
                ))
    data = np.array(records, dtype=[(name, np.float64) for name in GRID_FIELDS])
    for field, mask in nan_masks(status).items():
        data[field][mask] = math.nan
    return data


def drain_deferred_messages(cv) -> list[str]:
    """Flush deferred ray-failure diagnostics before a lens reload.

    A failed COM-level cv.RAYRSI (slow path) leaves e.g. "Error: Ray missed
    surface 5 for zoom position 1" that CODE V prints in the response of the
    next cv.Command; without this, load_lens would see it on ``res``. Only
    DEFERRED_RAY_ERROR lines are tolerated here; anything else raises.
    """
    drained: list[str] = []
    for _ in range(DRAIN_MAX_COMMANDS):
        response = cv.Command("OUT Y")
        lines = [line.strip() for line in response.splitlines() if line.strip()]
        if "Command End:" not in lines:
            raise RuntimeError(f"CODE V drain command did not complete: {response!r}")
        other = [line for line in lines if line != "Command End:"]
        unknown = [line for line in other if not DEFERRED_RAY_ERROR.match(line)]
        if unknown:
            raise RuntimeError(f"Unexpected CODE V output before lens reload: {response!r}")
        if not other:
            return drained
        drained.extend(other)
    raise RuntimeError(
        f"CODE V still reports deferred ray errors after {DRAIN_MAX_COMMANDS} drain commands: {drained!r}"
    )


def sweep_accommodation(
    cv,
    accommodation_d: float,
    ade_values: Sequence[float],
    bde_values: Sequence[float],
    work_dir: Path,
    tag: str,
    keep_csv: bool = False,
) -> tuple[np.ndarray, dict]:
    """Load lens, apply accommodation, run one macro over ADE x BDE, return records.

    info["ray_status"] is the (n_ade, n_bde, 11) int16 status array (not JSON;
    run_shard moves it to the failure sidecar).
    """
    drained = drain_deferred_messages(cv)
    if drained:
        print(f"  drained deferred CODE V ray diagnostics before reload: {drained!r}", flush=True)
    load_lens(cv)
    if drained:
        # Prove the restored lens is clean: the baseline chief ray (field 1) traces.
        st = int(cv.RAYRSI(1, 1, 1, 0, [0.0, 0.0, 0.0, 0.0]))
        if st != 0:
            raise RuntimeError(f"Baseline chief ray fails (surface {st}) after reloading {LENS_FILE}")
    baseline = read_accommodation_baseline(cv)
    targets = apply_accommodation(cv, baseline, accommodation_d)
    fov_x, fov_y = angular_fov(cv)
    seq_path = work_dir / f"{tag}.seq"
    csv_path = work_dir / f"{tag}.csv"
    write_macro(seq_path, csv_path, ade_values, bde_values, fov_x, fov_y)
    n_states = len(ade_values) * len(bde_values)
    t0 = time.perf_counter()
    run_macro(cv, seq_path, csv_path)
    elapsed = time.perf_counter() - t0
    table = parse_csv(csv_path, len(ade_values), len(bde_values))
    status = check_table(table, ade_values, bde_values)
    data = derive_rows(table, ade_values, bde_values, accommodation_d, status)
    if not keep_csv:
        csv_path.unlink()
    info = {
        "ray_status": status.reshape(len(ade_values), len(bde_values), len(STATUS_COLUMNS)),
        "fov_x_deg": fov_x,
        "fov_y_deg": fov_y,
        "geometry_baseline": baseline,
        "geometry_target": targets,
        "macro_seconds": elapsed,
        "ms_per_state": 1000.0 * elapsed / n_states,
    }
    return data, info


def validate_group(
    data: np.ndarray,
    ade_values: Sequence[float],
    bde_values: Sequence[float],
    accommodation_d: float,
    grid_lines: int,
    status: np.ndarray | None = None,
) -> None:
    """Accommodation-shard analogue of distortion_grid_p4_biaxial.validate_group.

    status: (n_ade, n_bde, 11) ray statuses; None means no ray may have failed.
    NaN must appear exactly where nan_masks(status) says, finite values elsewhere.
    """
    n_ade, n_bde, n_grid = len(ade_values), len(bde_values), grid_lines**2
    if data.ndim != 1 or data.size != n_ade * n_bde * n_grid or data.dtype.names != GRID_FIELDS:
        raise ValueError(
            f"Shard shape/schema mismatch: got {data.size} records with {data.dtype.names}; "
            f"expected {n_ade * n_bde * n_grid} and {GRID_FIELDS}"
        )
    if n_grid != N_GRID:
        raise ValueError(f"The macro backend supports grid_lines={GRID_LINES} only")
    expected_status_shape = (n_ade, n_bde, len(STATUS_COLUMNS))
    if status is None:
        status = np.zeros(expected_status_shape, dtype=STATUS_DTYPE)
    if not isinstance(status, np.ndarray) or status.shape != expected_status_shape or status.dtype != STATUS_DTYPE:
        raise ValueError(
            f"Ray status array must be {STATUS_DTYPE.__name__} of shape {expected_status_shape}; got "
            f"{getattr(status, 'dtype', type(status))} {getattr(status, 'shape', None)}"
        )
    check_nan_pattern(data, status, "Shard")
    shaped = data.reshape(n_ade, n_bde, n_grid)
    ade = np.asarray(ade_values, dtype=np.float64)
    bde = np.asarray(bde_values, dtype=np.float64)
    if not np.all(shaped["eye_rotation_deg"] == ade[:, None, None]):
        raise ValueError("Shard ADE values do not exactly match the requested sweep")
    if not np.all(shaped["eye_rotation_x_deg"] == bde[None, :, None]):
        raise ValueError("Shard BDE values do not exactly match the requested sweep")
    if not np.all(data["accommodation_d"] == accommodation_d):
        raise ValueError("Shard accommodation values do not match the requested value")
    if not np.all(data["z_mm"] == Z_MM):
        raise ValueError("Shard Z values are not held at the fixed constant")
    fields = np.asarray(grid_fields(grid_lines), dtype=np.float64)
    if not (np.all(shaped["field_x_relative"] == fields[None, None, :, 0])
            and np.all(shaped["field_y_relative"] == fields[None, None, :, 1])):
        raise ValueError("Shard field coordinates are not the expected grid order")
    if np.unique(fields, axis=0).shape[0] != n_grid:
        raise ValueError("Grid field pairs are not unique")
    for center_field in ("center_absolute_x_mm", "center_absolute_y_mm"):
        values = shaped[center_field]
        first = values[:, :, :1]
        same = (values == first) | (np.isnan(values) & np.isnan(first))
        if not np.all(same):
            raise ValueError(f"Shard has varying {center_field} within a state's grid")


def validate_complete_sweep(
    data: np.ndarray,
    status_map: np.ndarray,
    ade_values: Sequence[float],
    bde_values: Sequence[float],
    accommodations: Sequence[float],
    expected_count: int,
) -> None:
    """Mask-aware analogue of distortion_grid_p4_biaxial.validate_complete_sweep.

    status_map: (n_acc, n_ade, n_bde, 11); NaN only where it explains it.
    """
    if data.ndim != 1 or data.size != expected_count or data.dtype.names != GRID_FIELDS:
        raise ValueError(
            f"Final dataset has {data.size} records with schema {data.dtype.names}; "
            f"expected {expected_count} records with {GRID_FIELDS}"
        )
    expected_shape = (len(accommodations), len(ade_values), len(bde_values), len(STATUS_COLUMNS))
    if status_map.shape != expected_shape or status_map.dtype != STATUS_DTYPE:
        raise ValueError(f"Failure map has shape {status_map.shape}/{status_map.dtype}; expected {expected_shape}")
    expected_dimensions = {
        "accommodation_d": np.asarray(accommodations, dtype=np.float64),
        "eye_rotation_deg": np.asarray(ade_values, dtype=np.float64),
        "eye_rotation_x_deg": np.asarray(bde_values, dtype=np.float64),
    }
    for field, expected_values in expected_dimensions.items():
        actual_values = np.unique(data[field])
        if not np.array_equal(actual_values, np.unique(expected_values)):
            raise ValueError(
                f"Final {field} dimension mismatch: got {actual_values.size} unique values, "
                f"expected {expected_values.size}"
            )
    check_nan_pattern(data, status_map, "Final dataset")
    if not np.all(data["z_mm"] == Z_MM):
        raise ValueError("Final dataset Z values are not held at the fixed constant")


def shard_name(ai: int) -> str:
    return f"group_{ai:06d}.pkl"


def failures_name(ai: int) -> str:
    return f"group_{ai:06d}.failures.npz"


def atomic_npz(path: Path, **arrays) -> None:
    """np.savez through an open handle (no '.npz' suffix appended to the tmp name)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    tmp_path.replace(path)


def failure_counts(status: np.ndarray) -> dict:
    """JSON-ready failure counts of a (..., 11) status array."""
    failed = np.asarray(status) != 0
    return {
        "failed_states": int(np.count_nonzero(np.any(failed, axis=-1))),
        "failed_rays": int(np.count_nonzero(failed)),
        "failed_rays_by_column": {
            name: int(np.count_nonzero(failed[..., k])) for k, name in enumerate(STATUS_COLUMNS)
        },
    }


def load_failures(
    path: Path,
    ade_values: Sequence[float],
    bde_values: Sequence[float],
    accommodation_d: float,
) -> np.ndarray:
    """Load and check a shard's failure sidecar; returns its status array."""
    with np.load(path, allow_pickle=False) as npz:
        if set(npz.files) != {"status", "status_columns", "accommodation_d", "ade_deg", "bde_deg"}:
            raise ValueError(f"Failure sidecar {path} has unexpected arrays {npz.files}")
        status = npz["status"]
        if tuple(str(name) for name in npz["status_columns"]) != STATUS_COLUMNS:
            raise ValueError(f"Failure sidecar {path} has unexpected status columns")
        if float(npz["accommodation_d"]) != float(accommodation_d):
            raise ValueError(f"Failure sidecar {path} accommodation does not match")
        if not (np.array_equal(npz["ade_deg"], np.asarray(ade_values, dtype=np.float64))
                and np.array_equal(npz["bde_deg"], np.asarray(bde_values, dtype=np.float64))):
            raise ValueError(f"Failure sidecar {path} ADE/BDE axes do not match the sweep")
    expected_shape = (len(ade_values), len(bde_values), len(STATUS_COLUMNS))
    if status.shape != expected_shape or status.dtype != STATUS_DTYPE:
        raise ValueError(f"Failure sidecar {path} status is {status.dtype} {status.shape}; expected {expected_shape}")
    return status


def print_failure_summary(
    status_map: np.ndarray,
    ade_values: Sequence[float],
    bde_values: Sequence[float],
    accommodations: Sequence[float],
    max_listed: int = 10,
) -> None:
    ade = np.asarray(ade_values, dtype=np.float64)
    bde = np.asarray(bde_values, dtype=np.float64)
    total = failure_counts(status_map)
    n_states = status_map.shape[0] * status_map.shape[1] * status_map.shape[2]
    print(
        f"Ray-failure summary: {total['failed_states']:,}/{n_states:,} states with a failed ray, "
        f"{total['failed_rays']:,} failed rays; by ray {total['failed_rays_by_column']}",
        flush=True,
    )
    for ai, accommodation_d in enumerate(accommodations):
        counts = failure_counts(status_map[ai])
        if not counts["failed_states"]:
            print(f"  {accommodation_d:.6g} D: 0 failed states", flush=True)
            continue
        ia, ib = np.nonzero(np.any(status_map[ai] != 0, axis=-1))
        by_ray = {k: v for k, v in counts["failed_rays_by_column"].items() if v}
        print(
            f"  {accommodation_d:.6g} D: {counts['failed_states']} failed states, "
            f"{counts['failed_rays']} failed rays {by_ray}; ADE [{ade[ia].min():g}, {ade[ia].max():g}], "
            f"BDE [{bde[ib].min():g}, {bde[ib].max():g}]",
            flush=True,
        )
    listed = 0
    for ai, ia, ib in zip(*np.nonzero(np.any(status_map != 0, axis=-1))):
        if listed >= max_listed:
            print(f"  ... ({total['failed_states'] - listed} more failed states)", flush=True)
            break
        rays = {
            STATUS_COLUMNS[k]: int(status_map[ai, ia, ib, k])
            for k in np.nonzero(status_map[ai, ia, ib])[0]
        }
        print(
            f"  failed state: acc={accommodations[ai]:.6g} D, ADE={ade[ia]:g}, BDE={bde[ib]:g}: "
            f"{rays} (ray: failed surface)",
            flush=True,
        )
        listed += 1


def run_shard(
    cv,
    ai: int,
    accommodation_d: float,
    ade_values: Sequence[float],
    bde_values: Sequence[float],
    grid_lines: int,
    checkpoint_dir: Path,
    work_dir: Path,
) -> dict:
    """Sweep one accommodation, validate it and write its shard atomically; returns its metadata entry."""
    group_data, info = sweep_accommodation(
        cv, accommodation_d, ade_values, bde_values, work_dir, f"acc_{ai:04d}"
    )
    status = info.pop("ray_status")
    validate_group(group_data, ade_values, bde_values, accommodation_d, grid_lines, status)
    # Sidecar first: a shard is only recorded once both files and their hashes exist.
    failures_path = checkpoint_dir / failures_name(ai)
    atomic_npz(
        failures_path,
        status=status,
        status_columns=np.array(STATUS_COLUMNS),
        accommodation_d=np.float64(accommodation_d),
        ade_deg=np.asarray(ade_values, dtype=np.float64),
        bde_deg=np.asarray(bde_values, dtype=np.float64),
    )
    shard_path = checkpoint_dir / shard_name(ai)
    atomic_pickle(group_data, shard_path)
    return {
        "accommodation_index": ai,
        "accommodation_d": accommodation_d,
        "record_count": int(group_data.size),
        "shard": shard_name(ai),
        "sha256": file_sha256(shard_path),
        "failures": failures_name(ai),
        "failures_sha256": file_sha256(failures_path),
        **failure_counts(status),
        **info,
    }


def checkpoint_metadata(config: dict, config_hash: str, groups_by_index: dict[int, dict]) -> dict:
    """Checkpoint metadata with completed groups sorted by accommodation index."""
    groups = [groups_by_index[ai] for ai in sorted(groups_by_index)]
    total = sum(int(group["record_count"]) for group in groups)
    metadata = make_metadata(config, config_hash, groups, total)
    metadata["completed_accommodation_indices"] = sorted(groups_by_index)
    metadata["failed_states_checkpointed"] = sum(int(group["failed_states"]) for group in groups)
    metadata["failed_rays_checkpointed"] = sum(int(group["failed_rays"]) for group in groups)
    return metadata


def _worker_main(
    worker_id: int,
    ade_values: list[float],
    bde_values: list[float],
    accommodations: list[float],
    grid_lines: int,
    checkpoint_dir: str,
    task_queue,
    result_queue,
    stop_event,
) -> None:
    """Worker process: one CODE V session, shards taken from task_queue until a None sentinel."""
    import pythoncom
    import win32com.client

    checkpoint_path = Path(checkpoint_dir)
    work_dir = checkpoint_path / "work" / f"worker_{worker_id}"
    prefix = f"[worker {worker_id}]"
    current = None
    pythoncom.CoInitialize()
    cv = None
    started = False
    try:
        try:
            cv = win32com.client.Dispatch("CodeV.Application")
            timeout = configure_session(cv)
            t0 = time.perf_counter()
            cv.StartCodeV()
            started = True
            print(
                f"{prefix} CODE V started (pid {os.getpid()}, CommandTimeout = {timeout}, "
                f"startup {time.perf_counter() - t0:.1f} s)",
                flush=True,
            )
            result_queue.put(("started", worker_id, None, None))
            while True:
                ai = task_queue.get()
                if ai is None:
                    break
                if stop_event.is_set():
                    result_queue.put(("skipped", worker_id, ai, None))
                    continue
                current = ai
                t0 = time.perf_counter()
                group = run_shard(
                    cv, ai, float(accommodations[ai]), ade_values, bde_values, grid_lines,
                    checkpoint_path, work_dir,
                )
                group["worker"] = worker_id
                group["shard_wall_seconds"] = time.perf_counter() - t0
                result_queue.put(("done", worker_id, ai, group))
                current = None
        except BaseException:
            result_queue.put(("error", worker_id, current, traceback.format_exc()))
    finally:
        try:
            if cv is not None and started:
                cv.StopCodeV()
        except BaseException:
            result_queue.put(("error", worker_id, None, "StopCodeV failed:\n" + traceback.format_exc()))
        finally:
            pythoncom.CoUninitialize()
            result_queue.put(("exit", worker_id, None, None))


def _run_parallel(
    n_workers: int,
    todo: list[int],
    ade_values: Sequence[float],
    bde_values: Sequence[float],
    accommodations: Sequence[float],
    grid_lines: int,
    checkpoint_dir: Path,
    on_done,
) -> None:
    """Coordinator: dispatch accommodation indices to worker processes; on_done(group) per shard."""
    ctx = multiprocessing.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    stop_event = ctx.Event()
    args = (
        [float(v) for v in ade_values], [float(v) for v in bde_values],
        [float(v) for v in accommodations], int(grid_lines), str(checkpoint_dir),
        task_queue, result_queue, stop_event,
    )
    procs = [
        ctx.Process(target=_worker_main, args=(k, *args), name=f"codev_worker_{k}")
        for k in range(n_workers)
    ]
    pending = list(todo)
    in_flight: set[int] = set()
    alive = set(range(n_workers))
    dead_polls = {k: 0 for k in range(n_workers)}
    failures: list[str] = []
    sentinels_sent = False
    stopping = False

    def dispatch() -> None:
        nonlocal sentinels_sent
        while not stopping and pending and len(in_flight) < len(alive):
            ai = pending.pop(0)
            in_flight.add(ai)
            task_queue.put(ai)
        if (stopping or not pending) and not sentinels_sent:
            for _ in range(n_workers):
                task_queue.put(None)
            sentinels_sent = True

    print(f"Starting {n_workers} CODE V worker processes for {len(todo)} accommodation shards...", flush=True)
    for proc in procs:
        proc.start()
    try:
        dispatch()
        while alive:
            try:
                kind, worker_id, ai, payload = result_queue.get(timeout=1.0)
            except queue.Empty:
                for k in list(alive):
                    if not procs[k].is_alive():
                        dead_polls[k] += 1
                        if dead_polls[k] >= 3:  # exited without its final "exit" message
                            alive.discard(k)
                            failures.append(
                                f"worker {k} died unexpectedly (exit code {procs[k].exitcode})"
                            )
                            stopping = True
                            stop_event.set()
                dispatch()
                continue
            if kind == "done":
                in_flight.discard(ai)
                on_done(payload)
            elif kind == "error":
                if ai is not None:
                    in_flight.discard(ai)
                where = f"accommodation index {ai}" if ai is not None else "session setup/teardown"
                failures.append(f"worker {worker_id} failed at {where}:\n{payload}")
                print(f"[worker {worker_id}] ERROR at {where}; stopping dispatch.", flush=True)
                stopping = True
                stop_event.set()
            elif kind == "skipped":
                in_flight.discard(ai)
            elif kind == "exit":
                alive.discard(worker_id)
            dispatch()
    except BaseException:
        stopping = True
        stop_event.set()
        dispatch()
        raise
    finally:
        task_queue.cancel_join_thread()
        for proc in procs:
            proc.join(timeout=120)
        for proc in procs:
            if proc.is_alive():
                print(
                    f"WARNING: terminating {proc.name}; its CODE V session may need manual cleanup",
                    flush=True,
                )
                proc.terminate()
                proc.join(timeout=10)
    if failures:
        raise RuntimeError(
            f"{len(failures)} worker failure(s); completed shards are checkpointed, "
            f"rerun with --resume:\n" + "\n".join(failures)
        )


def collect(
    ade_values: Sequence[float],
    bde_values: Sequence[float],
    accommodations: Sequence[float],
    grid_lines: int,
    output_dir: Path,
    resume: bool = False,
    stop_after: int | None = None,
    workers: int = 1,
) -> bool:
    """Run/resume the sweep. Returns True when the canonical pickle was published."""
    if grid_lines != GRID_LINES:
        raise ValueError(f"The macro backend supports grid_lines={GRID_LINES} only")
    if not ade_values or not bde_values or not accommodations:
        raise ValueError("ADE, BDE, and accommodation sweeps must be nonempty")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if not LENS_FILE.is_file():
        raise FileNotFoundError(f"P4 CODE V lens file not found: {LENS_FILE}")

    checkpoint_dir = output_dir.with_name(output_dir.name + ".checkpoint")
    checkpoint_meta = checkpoint_dir / "metadata.json"
    rows_per_group = len(ade_values) * len(bde_values) * grid_lines**2
    config = {
        "backend": BACKEND,
        "model": str(LENS_FILE.relative_to(PROJECT_DIR)),
        "model_sha256": hashlib.sha256(LENS_FILE.read_bytes()).hexdigest(),
        "zoom": "z1 (P4)",
        "ade_deg": [float(value) for value in ade_values],
        "bde_deg": [float(value) for value in bde_values],
        "accommodation_d": [float(value) for value in accommodations],
        "z_mm_fixed": Z_MM,
        "grid_lines": int(grid_lines),
        "fields": list(GRID_FIELDS),
        "expected_record_count": rows_per_group * len(accommodations),
    }
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    expected_group_count = len(accommodations)
    groups_by_index: dict[int, dict] = {}

    def check_group_failures(group: dict, ai: int, error=ValueError) -> np.ndarray:
        """Verify a group's failure sidecar (name, sha256, contents, counts); return its status."""
        if group.get("failures") != failures_name(ai):
            raise error(f"Group {ai} has unexpected failure sidecar name {group.get('failures')!r}")
        path = checkpoint_dir / failures_name(ai)
        if not path.is_file() or file_sha256(path) != group.get("failures_sha256"):
            raise error(f"Failure sidecar is missing or corrupt: {path}")
        try:
            status = load_failures(path, ade_values, bde_values, float(accommodations[ai]))
        except ValueError as exc:
            raise error(str(exc)) from exc
        counts = failure_counts(status)
        if any(group.get(key) != value for key, value in counts.items()):
            raise error(f"Group {ai} failure counts in metadata do not match {path}")
        return status

    if resume:
        if not checkpoint_meta.is_file():
            raise FileNotFoundError(f"--resume requires checkpoint metadata under {checkpoint_dir}")
        metadata = json.loads(checkpoint_meta.read_text(encoding="utf-8"))
        if any(metadata.get(key) != value for key, value in config.items()) or metadata.get(
            "configuration_sha256"
        ) != config_hash:
            raise ValueError("Checkpoint model or sweep settings do not match this run")
        prior_groups = metadata.get("completed_groups", [])
        if len(prior_groups) > expected_group_count:
            raise ValueError("Checkpoint contains more groups than requested")
        total_records = 0
        for group in prior_groups:
            ai = group.get("accommodation_index")
            if type(ai) is not int or not 0 <= ai < expected_group_count:
                raise ValueError(f"Checkpoint group has invalid accommodation index {ai!r}")
            if ai in groups_by_index:
                raise ValueError(f"Checkpoint lists accommodation index {ai} more than once")
            if not math.isclose(
                float(group.get("accommodation_d", math.nan)), accommodations[ai],
                rel_tol=0.0, abs_tol=1e-8,
            ):
                raise ValueError(f"Checkpoint group {ai} accommodation does not match the requested sweep")
            if group.get("shard") != shard_name(ai):
                raise ValueError(f"Checkpoint group {ai} has unexpected shard name {group.get('shard')!r}")
            shard_path = checkpoint_dir / group["shard"]
            if not shard_path.is_file() or file_sha256(shard_path) != group.get("sha256"):
                raise ValueError(f"Checkpoint shard is missing or corrupt: {shard_path}")
            status = check_group_failures(group, ai, ValueError)
            with shard_path.open("rb") as stream:
                shard = pickle.load(stream)
            if (
                not isinstance(shard, np.ndarray)
                or group.get("record_count") != rows_per_group
            ):
                raise ValueError(f"Checkpoint shard has unexpected contents: {shard_path}")
            validate_group(shard, ade_values, bde_values, float(accommodations[ai]), grid_lines, status)
            groups_by_index[ai] = group
            total_records += int(shard.size)
        if metadata.get("total_records_checkpointed") != total_records:
            raise ValueError("Checkpoint record total does not match its groups")
        recorded = metadata.get("completed_accommodation_indices")
        if recorded is not None and sorted(recorded) != sorted(groups_by_index):
            raise ValueError("Checkpoint completed_accommodation_indices do not match its groups")
        print(
            f"Resuming with {len(groups_by_index)}/{expected_group_count} accommodation shards "
            f"complete (indices {sorted(groups_by_index)}).",
            flush=True,
        )
    elif checkpoint_dir.exists():
        raise FileExistsError(
            f"Checkpoint exists under {checkpoint_dir}; pass --resume or choose a new output directory"
        )
    else:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(checkpoint_metadata(config, config_hash, {}), checkpoint_meta)

    todo = [ai for ai in range(expected_group_count) if ai not in groups_by_index]
    if stop_after is not None:
        todo = todo[:max(0, stop_after)]
    sweep_t0 = time.perf_counter()

    def record_shard(group: dict) -> None:
        """Coordinator-only: verify a finished shard and checkpoint metadata."""
        ai = group["accommodation_index"]
        shard_path = checkpoint_dir / shard_name(ai)
        if ai in groups_by_index:
            raise RuntimeError(f"Accommodation index {ai} completed twice")
        if group.get("record_count") != rows_per_group:
            raise RuntimeError(f"Shard {ai} has {group.get('record_count')} records; expected {rows_per_group}")
        if not shard_path.is_file() or file_sha256(shard_path) != group["sha256"]:
            raise RuntimeError(f"Shard {shard_path} is missing or does not match its reported hash")
        check_group_failures(group, ai, RuntimeError)
        groups_by_index[ai] = group
        metadata = checkpoint_metadata(config, config_hash, groups_by_index)
        atomic_json(metadata, checkpoint_meta)
        worker = f" by worker {group['worker']}" if "worker" in group else ""
        print(
            f"Accommodation {group['accommodation_d']:g} D (index {ai}){worker}: fields "
            f"X={group['fov_x_deg']:g}°, Y={group['fov_y_deg']:g}°; macro "
            f"{group['macro_seconds']:.2f} s ({group['ms_per_state']:.2f} ms/state); "
            f"{group['failed_states']} failed states / {group['failed_rays']} failed rays (-> NaN); "
            f"checkpointed {len(groups_by_index)}/{expected_group_count} shards, "
            f"{metadata['total_records_checkpointed']:,} records "
            f"(elapsed {time.perf_counter() - sweep_t0:.1f} s)",
            flush=True,
        )

    n_workers = min(workers, len(todo))
    if n_workers == 1:
        import pythoncom
        import win32com.client

        work_dir = checkpoint_dir / "work"
        pythoncom.CoUninitialize()
        pythoncom.CoInitialize()
        cv = None
        started = False
        try:
            print("Connecting to CODE V COM...", flush=True)
            cv = win32com.client.Dispatch("CodeV.Application")
            timeout = configure_session(cv)
            print(f"CommandTimeout = {timeout} (unlimited)", flush=True)
            cv.StartCodeV()
            started = True
            sweep_t0 = time.perf_counter()
            for ai in todo:
                accommodation_d = float(accommodations[ai])
                print(
                    f"Accommodation {accommodation_d:g} D ({ai + 1}/{expected_group_count}): "
                    f"{len(ade_values)} ADE x {len(bde_values)} BDE in one macro",
                    flush=True,
                )
                record_shard(run_shard(
                    cv, ai, accommodation_d, ade_values, bde_values, grid_lines,
                    checkpoint_dir, work_dir,
                ))
        finally:
            try:
                if cv is not None and started:
                    cv.StopCodeV()
            finally:
                pythoncom.CoUninitialize()
    elif n_workers > 1:
        _run_parallel(
            n_workers, todo, ade_values, bde_values, accommodations, grid_lines,
            checkpoint_dir, record_shard,
        )

    if len(groups_by_index) < expected_group_count:
        print(
            f"Stopped with {len(groups_by_index)}/{expected_group_count} accommodation shards "
            f"complete; rerun with --resume to continue.",
            flush=True,
        )
        return False

    final_parts = []
    status_parts = []
    for ai in range(expected_group_count):
        group = groups_by_index[ai]
        shard_path = checkpoint_dir / group["shard"]
        if file_sha256(shard_path) != group["sha256"]:
            raise RuntimeError(f"Shard {shard_path} changed after it was checkpointed")
        status_parts.append(check_group_failures(group, ai, RuntimeError))
        with shard_path.open("rb") as stream:
            final_parts.append(pickle.load(stream))
    final_data = np.concatenate(final_parts)
    status_map = np.stack(status_parts)
    try:
        validate_complete_sweep(
            final_data, status_map, ade_values, bde_values, accommodations,
            config["expected_record_count"],
        )
    except ValueError as exc:
        raise RuntimeError(
            f"Final P4 biaxial sweep validation failed; checkpoints remain available for --resume: {exc}"
        ) from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = output_dir / "distortion_grid.pkl"
    failure_map_path = output_dir / "failure_map.npz"
    atomic_npz(
        failure_map_path,
        status=status_map,
        status_columns=np.array(STATUS_COLUMNS),
        accommodation_d=np.asarray(accommodations, dtype=np.float64),
        ade_deg=np.asarray(ade_values, dtype=np.float64),
        bde_deg=np.asarray(bde_values, dtype=np.float64),
    )
    atomic_pickle(final_data, canonical_path)
    stale_metadata = output_dir / "metadata.json"
    if stale_metadata.exists():
        stale_metadata.unlink()
    remove_tree(checkpoint_dir)
    print_failure_summary(status_map, ade_values, bde_values, accommodations)
    print(
        f"Published {final_data.size:,} records to {canonical_path} "
        f"(ray-failure map {status_map.shape} int16 in {failure_map_path})",
        flush=True,
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ade-min", type=float, default=-20.0)
    parser.add_argument("--ade-max", type=float, default=20.0)
    parser.add_argument("--ade-count", type=int, default=401)
    parser.add_argument("--bde-min", type=float, default=-20.0)
    parser.add_argument("--bde-max", type=float, default=20.0)
    parser.add_argument("--bde-count", type=int, default=401)
    parser.add_argument("--accommodation-min", type=float, default=0.0)
    parser.add_argument("--accommodation-max", type=float, default=4.0)
    parser.add_argument("--accommodation-count", type=int, default=41)
    parser.add_argument("--grid-lines", type=int, default=GRID_LINES)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--resume", action="store_true", help="resume matching accommodation checkpoints")
    parser.add_argument(
        "--stop-after-accommodations", type=int, default=None, metavar="N",
        help="dispatch at most N new accommodation shards, then exit without publishing",
    )
    parser.add_argument(
        "--workers", type=int, default=1, metavar="N",
        help="parallel CODE V sessions in separate processes (default 1 = in-process, original behavior)",
    )
    args = parser.parse_args()
    if args.ade_count < 1 or args.bde_count < 1 or args.accommodation_count < 1:
        parser.error("ADE, BDE, and accommodation counts must be at least 1")
    if args.grid_lines != GRID_LINES:
        parser.error(f"the macro backend supports --grid-lines {GRID_LINES} only")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.stop_after_accommodations is not None and args.stop_after_accommodations < 0:
        parser.error("--stop-after-accommodations must be nonnegative")
    output_dir = args.output_dir.resolve()
    if PROJECT_DIR not in output_dir.parents:
        parser.error(f"--output-dir must be inside the project ({PROJECT_DIR})")
    ade_values = np.linspace(args.ade_min, args.ade_max, args.ade_count)
    bde_values = np.linspace(args.bde_min, args.bde_max, args.bde_count)
    accommodations = np.linspace(args.accommodation_min, args.accommodation_max, args.accommodation_count)
    collect(
        ade_values.tolist(), bde_values.tolist(), accommodations.tolist(),
        args.grid_lines, output_dir, args.resume, args.stop_after_accommodations, args.workers,
    )


if __name__ == "__main__":
    main()
