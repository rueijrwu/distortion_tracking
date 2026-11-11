"""Trace P4 rays at S13, the S16 stop, and SI in the S16 global frame.

The pilot uses two accommodations and a 5 by 5 gaze grid. The full profile is
21 by 101 by 101 states. Each worker owns one CODE V session and reloads the
requested P4 .len before each accommodation. Verified full zero-thickness
traces publish to data/p4_raytrace/p4_raytrace_251x251_12p5_A4.pkl. Pilots and
file-thickness traces use distinct names there; temporary files stay in runs/.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import math
import multiprocessing as mp
import os
import pickle
import queue
import re
import time
import traceback
from pathlib import Path

import numpy as np

from distortion_grid import angular_fov
from distortion_grid_p4 import (
    apply_accommodation,
    command_checked,
    cv_eval,
    read_accommodation_baseline,
)
from distortion_grid_p4_biaxial_macro import drain_deferred_messages


ROOT = Path(__file__).resolve().parent.parent
LENS = ROOT / "Lens" / "p4_ME_no_DM.len"
LENS_SEQ = ROOT / "Lens" / "p4_ME_no_DM.seq"
RUNS = ROOT / "runs"
OUTPUT_DIR = ROOT / "data" / "p4_raytrace"
FIELDS = tuple((float(x), float(y)) for y in (1, 0, -1) for x in (-1, 0, 1))
SURFACE_FIELDS = tuple(f"{axis}{surface}" for surface in (13, 16) for axis in "xyzlmn")
SENSOR_FIELDS = ("xsi", "ysi", "zsi")
LOCAL_FIELDS = ("xsi_local", "ysi_local", "center_xsi_local", "center_ysi_local")
OPTICAL_FIELDS = SURFACE_FIELDS + SENSOR_FIELDS + LOCAL_FIELDS
COLS = (
    "state_index", "ray_index", "accommodation_index", "ade_index", "bde_index",
    "accommodation_d", "ade_deg", "bde_deg", "field_x_relative",
    "field_y_relative", "rc_ade_readback_deg", "rc_bde_readback_deg",
    "cornea_ent_d_thi_readback_mm", "center_rayrsi_status", "rayrsi_status",
) + OPTICAL_FIELDS
DTYPE = np.dtype([(name, np.float64) for name in COLS])
RAW_COLS = (
    "ade_index", "bde_index", "ray_index", "ade_deg", "bde_deg",
    "rc_ade_readback_deg", "rc_bde_readback_deg",
    "cornea_ent_d_thi_readback_mm", "center_rayrsi_status", "rayrsi_status",
) + OPTICAL_FIELDS
READBACK_TOL_DEG = 1e-7
READBACK_TOL_MM = 1e-8
DIRECT_TOL_MM = 1e-8
DIRECT_TOL_DIRECTION = 1e-9
LMN_NORM_TOL = 1e-8
STOP_ORIGIN_TOL_MM = 1e-6
S13_S16_LMN_TOL = 1e-8
SEGMENT_ALIGNMENT_TOL = 1e-7


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def axes(profile: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if profile == "pilot":
        return np.array([0.0, 4.0]), np.linspace(-12.0, 12.0, 5), np.linspace(-12.0, 12.0, 5)
    return np.linspace(0.0, 4.0, 21), np.linspace(-12.5, 12.5, 101), np.linspace(-12.5, 12.5, 101)


def verify_sequence_topology() -> float:
    """Check the readable companion before relying on the numbered .len surfaces."""
    lines = LENS_SEQ.read_text(encoding="ascii", errors="replace").splitlines()
    labels = {}
    surface = None
    for line in lines:
        if re.match(r"^S\s+", line):
            surface = 1 if surface is None else surface + 1
        elif line.startswith("SO "):
            surface = 0
        match = re.search(r'SLB\s+"([^"]+)"', line)
        if match:
            labels[match.group(1)] = surface
    if labels.get("CorneaF_D") != 13 or labels.get("STP_ENT") != 16:
        raise RuntimeError(f"Companion sequence has unexpected S13/S16 labels: {labels!r}")
    if labels.get("Cornea_ENT_D") != 14:
        raise RuntimeError(f"Companion sequence has unexpected Cornea_ENT_D surface: {labels!r}")
    thi = None
    for i, line in enumerate(lines):
        if 'SLB "Cornea_ENT_D"' in line:
            for previous in reversed(lines[:i]):
                match = re.match(r"^S\s+[-+0-9.eE]+\s+([-+0-9.eE]+)", previous)
                if match:
                    thi = float(match.group(1))
                    break
            break
    if thi is None or not math.isfinite(thi):
        raise RuntimeError("Could not parse Cornea_ENT_D THI from companion sequence")
    if not any(re.match(r"^GLO\s+S16\s", line) for line in lines):
        raise RuntimeError("Companion sequence does not declare GLO S16")
    if not any(re.match(r"^\s*STO\s*$", lines[i]) and i + 1 < len(lines) and
               'SLB "STP_ENT"' in lines[i + 1] for i in range(len(lines))):
        raise RuntimeError("Companion sequence does not mark STP_ENT as stop")
    return thi


def wait_ready(cv: object, timeout_s: float = 180.0) -> str:
    start = time.monotonic()
    last = ""
    while time.monotonic() - start < timeout_s:
        try:
            response = command_checked(cv, "VER", "checking CODE V readiness")
            if response.strip():
                return response
        except Exception as exc:
            last = repr(exc)
        time.sleep(0.5)
    raise RuntimeError(f"CODE V was not ready within {timeout_s:g} s: {last}")


def setup_state(cv: object, ade: float, bde: float, fov_x: float, fov_y: float,
                target_thi: float = 0.0) -> tuple[float, float, float]:
    command_checked(
        cv,
        f'ADE S"RC" {ade:.12g};BDE S"RC" {bde:.12g};' +
        (f'THI S"Cornea_ENT_D" {target_thi:.16g};' if target_thi == 0.0 else '') + 'SET VIG',
        "setting RC and field geometry",
    )
    ade_rb = cv_eval(cv, "ade", 's"RC"')
    bde_rb = cv_eval(cv, "bde", 's"RC"')
    thi_rb = cv_eval(cv, "thi", 's"Cornea_ENT_D"')
    if (not all(math.isfinite(value) for value in (ade_rb, bde_rb, thi_rb))
            or abs(ade_rb - ade) > READBACK_TOL_DEG
            or abs(bde_rb - bde) > READBACK_TOL_DEG
            or abs(thi_rb - target_thi) > READBACK_TOL_MM):
        raise RuntimeError(f"RC/thickness readback mismatch: {(ade_rb, bde_rb, thi_rb)} vs {(ade, bde, target_thi)}")
    command_checked(cv, f"XAN 0 {fov_x:.12g};YAN 0 {fov_y:.12g}", "restoring angular field")
    xob2, yob2 = cv_eval(cv, "xob", "f2", "z1"), cv_eval(cv, "yob", "f2", "z1")
    command_checked(cv, f"YOB 0 {yob2:.12g};XOB 0 {xob2:.12g}", "writing object-height fields")
    return ade_rb, bde_rb, thi_rb


def query_ray(cv: object) -> dict[str, float]:
    result = {}
    for surface in (13, 16):
        for axis in "xyzlmn":
            result[f"{axis}{surface}"] = cv_eval(cv, axis, f"s{surface}", "g16")
    for axis in "xyz":
        result[f"{axis}si"] = cv_eval(cv, axis, "si", "g16")
    result["xsi_local"] = cv_eval(cv, "x", "si")
    result["ysi_local"] = cv_eval(cv, "y", "si")
    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError(f"Successful direct ray has nonfinite optical data: {result}")
    return result


def probe_loaded_lens(cv: object, fov_x: float, fov_y: float, ade: float, bde: float,
                      target_thi: float = 0.0) -> dict:
    """Check numeric/label surfaces after RES and record the actual S16 frame."""
    setup_state(cv, ade, bde, fov_x, fov_y, target_thi)
    status = int(cv.RAYRSI(1, 1, 0, 0, [0.0, 0.0, 0.0, 0.0]))
    if status:
        raise RuntimeError(f"Loaded-lens center probe failed at surface {status}")
    ray = query_ray(cv)
    max_label_delta = 0.0
    for number, label in ((13, "CorneaF_D"), (16, "STP_ENT")):
        for axis in "xyzlmn":
            named = cv_eval(cv, axis, f's"{label}"', "g16")
            if not math.isfinite(named):
                raise RuntimeError(f"Loaded .len label {label} has nonfinite {axis} readback")
            max_label_delta = max(max_label_delta, abs(named - ray[f"{axis}{number}"]))
    if max_label_delta > 1e-9:
        raise RuntimeError(f"Loaded .len numeric/label surface mismatch: {max_label_delta}")
    # A chief ray uses pupil input (0,0), so its stop intersection should be
    # at the origin in the selected global frame. SI z is observed, not assumed.
    if math.hypot(ray["x16"], ray["y16"]) > 1e-6 or abs(ray["z16"]) > 1e-6:
        raise RuntimeError(f"S16 stop chief intersection is not at global origin: {ray}")
    return {"status": status, "s16_xyz_mm": [ray[f"{a}16"] for a in "xyz"],
            "si_xyz_mm": [ray[f"{a}si"] for a in "xyz"],
            "numeric_label_max_difference": max_label_delta}


def write_macro(path: Path, raw_path: Path, ade_values: np.ndarray, bde_values: np.ndarray,
                fov_x: float, fov_y: float, target_thi: float = 0.0) -> None:
    if '"' in str(raw_path) or '"' in str(path):
        raise ValueError("CODE V paths cannot contain double quotes")
    ade_step = float(ade_values[1] - ade_values[0])
    bde_step = float(bde_values[1] - bde_values[0])
    # Some names in RAW_COLS are copied from ^variables into the output; the
    # macro uses shorter names for the fixed state and center-ray values.
    base = ["^ia", "^ib", "^ri", "^ade", "^bde", "^ade_rb", "^bde_rb", "^thi_rb",
            "^st_c", "^st", "^input(4)", "^xob2", "^yob2", "^cx", "^cy"]
    optics = [f"^{name}" for name in SURFACE_FIELDS + SENSOR_FIELDS] + ["^xl", "^yl"]
    lines = [
        "! Generated by p4_retrace_full_detail.py; current-ray G16 reads.",
        "LCL NUM " + " ".join(base),
        "LCL NUM " + " ".join(optics),
        f'OPE NEW U^1 "{raw_path}"',
        "^input(1) == 0", "^input(2) == 0", "OUT N",
        f"FOR ^ia 0 {len(ade_values) - 1} 1",
        f"^ade == {float(ade_values[0])!r} + ^ia*{ade_step!r}",
        f"FOR ^ib 0 {len(bde_values) - 1} 1",
        f"^bde == {float(bde_values[0])!r} + ^ib*{bde_step!r}",
        'ADE S"RC" ^ade', 'BDE S"RC" ^bde',
        *([f'THI S"Cornea_ENT_D" {target_thi:.16g}'] if target_thi == 0.0 else []),
        "SET VIG", '^ade_rb == (ade s"RC")', '^bde_rb == (bde s"RC")',
        '^thi_rb == (thi s"Cornea_ENT_D")',
        f"XAN 0 {fov_x:.12g}", f"YAN 0 {fov_y:.12g}",
        "^xob2 == (xob f2 z1)", "^yob2 == (yob f2 z1)",
        "YOB 0 ^yob2", "XOB 0 ^xob2",
        "^input(3) == 0", "^input(4) == 0",
        "^st_c == rayrsi(1,1,1,0,^input)",
        "^cx == (x si)", "^cy == (y si)",
    ]
    fmt = " ".join(["'3e.17e'"] * len(RAW_COLS))
    prefix = "^ia ^ib ^ri ^ade ^bde ^ade_rb ^bde_rb ^thi_rb ^st_c ^st"
    optical_args = " ".join(f"^{name}" for name in SURFACE_FIELDS + SENSOR_FIELDS)
    for ri, (fx, fy) in enumerate(FIELDS):
        lines += [f"^ri == {ri}", f"^input(3) == {fx:g}", f"^input(4) == {fy:g}",
                  "^st == rayrsi(1,1,0,0,^input)"]
        for surface in (13, 16):
            for axis in "xyzlmn":
                lines.append(f"^{axis}{surface} == ({axis} s{surface} g16)")
        for axis in "xyz":
            lines.append(f"^{axis}si == ({axis} si g16)")
        lines += ["^xl == (x si)", "^yl == (y si)",
                  f'WRI U^1 Q"{fmt}" {prefix} {optical_args} ^xl ^yl ^cx ^cy']
    lines += ["END FOR", "END FOR", "OUT Y", "CLO U^1", "RTN", ""]
    path.write_text("\n".join(lines), encoding="ascii")


def parse_macro(raw_path: Path, ai: int, accommodation: float, ade_values: np.ndarray,
                bde_values: np.ndarray, target_thi: float = 0.0) -> tuple[np.ndarray, dict]:
    expected = len(ade_values) * len(bde_values) * len(FIELDS)
    table = np.loadtxt(raw_path, dtype=np.float64, ndmin=2)
    if table.shape != (expected, len(RAW_COLS)):
        raise ValueError(f"CODE V macro table shape {table.shape}, expected {(expected, len(RAW_COLS))}")
    raw = {name: table[:, k] for k, name in enumerate(RAW_COLS)}
    for name in RAW_COLS[:10]:
        if not np.all(np.isfinite(raw[name])):
            raise ValueError(f"Macro metadata/readback column {name} contains nonfinite values")
    ia = np.repeat(np.arange(len(ade_values)), len(bde_values) * len(FIELDS))
    ib = np.tile(np.repeat(np.arange(len(bde_values)), len(FIELDS)), len(ade_values))
    ri = np.tile(np.arange(len(FIELDS)), len(ade_values) * len(bde_values))
    for name, wanted in (("ade_index", ia), ("bde_index", ib), ("ray_index", ri)):
        if not np.array_equal(raw[name], wanted):
            raise ValueError(f"Macro row order mismatch in {name}")
    ade_req, bde_req = ade_values[ia], bde_values[ib]
    if np.max(np.abs(raw["ade_deg"] - ade_req)) > READBACK_TOL_DEG or np.max(np.abs(raw["bde_deg"] - bde_req)) > READBACK_TOL_DEG:
        raise ValueError("Macro requested angle values differ from the axis")
    if np.max(np.abs(raw["rc_ade_readback_deg"] - ade_req)) > READBACK_TOL_DEG or np.max(np.abs(raw["rc_bde_readback_deg"] - bde_req)) > READBACK_TOL_DEG:
        raise ValueError("RC ADE/BDE readback mismatch")
    if np.max(np.abs(raw["cornea_ent_d_thi_readback_mm"] - target_thi)) > READBACK_TOL_MM:
        raise ValueError("Cornea_ENT_D thickness readback mismatch")
    status = raw["rayrsi_status"]
    center_status = raw["center_rayrsi_status"]
    if not np.all(np.isfinite(status)) or not np.all(np.isfinite(center_status)) or not np.all(status == np.rint(status)) or not np.all(center_status == np.rint(center_status)):
        raise ValueError("Noninteger or nonfinite RAYRSI status")
    if target_thi != 0.0 and (np.any(status != 0) or np.any(center_status != 0)):
        raise ValueError("File-thickness pilot has a nonzero ray or center RAYRSI status")
    rows = np.empty(expected, dtype=DTYPE)
    rows["state_index"] = (ai * len(ade_values) + ia) * len(bde_values) + ib
    rows["ray_index"] = ri
    rows["accommodation_index"] = ai
    rows["ade_index"] = ia
    rows["bde_index"] = ib
    rows["accommodation_d"] = accommodation
    rows["ade_deg"], rows["bde_deg"] = ade_req, bde_req
    rows["field_x_relative"] = np.asarray(FIELDS)[ri, 0]
    rows["field_y_relative"] = np.asarray(FIELDS)[ri, 1]
    for name in ("rc_ade_readback_deg", "rc_bde_readback_deg", "cornea_ent_d_thi_readback_mm", "center_rayrsi_status", "rayrsi_status"):
        rows[name] = raw[name]
    for name in OPTICAL_FIELDS:
        rows[name] = raw[name]
    failed = status != 0
    for name in SURFACE_FIELDS + SENSOR_FIELDS + ("xsi_local", "ysi_local"):
        rows[name][failed] = math.nan
        if not np.all(np.isfinite(rows[name][~failed])):
            raise ValueError(f"Successful ray has nonfinite {name}")
    for name in ("center_xsi_local", "center_ysi_local"):
        rows[name][center_status != 0] = math.nan
        if not np.all(np.isfinite(rows[name][center_status == 0])):
            raise ValueError(f"Successful center ray has nonfinite {name}")
    if not np.any(~failed):
        raise ValueError("No successful rays in accommodation shard")
    d13 = np.stack([rows[f"{axis}13"][~failed] for axis in "lmn"], axis=1)
    d16 = np.stack([rows[f"{axis}16"][~failed] for axis in "lmn"], axis=1)
    p13 = np.stack([rows[f"{axis}13"][~failed] for axis in "xyz"], axis=1)
    p16 = np.stack([rows[f"{axis}16"][~failed] for axis in "xyz"], axis=1)
    segment = p16 - p13
    lengths = np.linalg.norm(segment, axis=1)
    if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
        raise ValueError("S13-to-S16 segment is zero or nonfinite")
    unit_segment = segment / lengths[:, None]
    norm_error = max(float(np.max(np.abs(np.linalg.norm(direction, axis=1) - 1))) for direction in (d13, d16))
    stop_origin = float(np.max(np.linalg.norm(p16, axis=1)))
    lmn_difference = float(np.max(np.abs(d13 - d16)))
    signed_dot = np.sum(d16 * unit_segment, axis=1)
    alignment_error = float(np.max(np.abs(np.abs(signed_dot) - 1)))
    if norm_error > LMN_NORM_TOL or stop_origin > STOP_ORIGIN_TOL_MM or lmn_difference > S13_S16_LMN_TOL or alignment_error > SEGMENT_ALIGNMENT_TOL:
        raise ValueError(f"S13/S16 geometry gate failed: norm={norm_error}, stop_origin={stop_origin}, "
                         f"lmn_difference={lmn_difference}, alignment={alignment_error}")
    summary = {
        "accommodation_index": ai, "accommodation_d": accommodation,
        "rows": expected, "success_rows": int(np.count_nonzero(~failed)),
        "failed_rows": int(np.count_nonzero(failed)),
        "center_failed_states": int(np.count_nonzero(center_status[::len(FIELDS)])),
        "status_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(status, return_counts=True))},
        "max_lmn_norm_error": norm_error, "s16_max_origin_radius_mm": stop_origin,
        "max_s13_s16_lmn_difference": lmn_difference,
        "max_abs_segment_alignment_error": alignment_error,
        "segment_direction_dot_min": float(np.min(signed_dot)),
        "segment_direction_dot_max": float(np.max(signed_dot)),
    }
    return rows, summary


def direct_comparison(cv: object, rows: np.ndarray, fov_x: float, fov_y: float,
                      ade: float, bde: float, target_thi: float = 0.0) -> dict:
    setup_state(cv, ade, bde, fov_x, fov_y, target_thi)
    status = int(cv.RAYRSI(1, 1, 0, 0, [0.0, 0.0, 0.0, 0.0]))
    macro_row = rows[4]  # First state, center field of y-major 3x3.
    if status != int(macro_row["rayrsi_status"]):
        raise RuntimeError(f"Direct COM status {status} != macro status {macro_row['rayrsi_status']}")
    if status:
        return {"status": status, "compared": False}
    direct = query_ray(cv)
    differences = {name: abs(direct[name] - float(macro_row[name])) for name in direct}
    max_position = max(differences[name] for name in differences if name[0] in "xyz")
    max_direction = max(differences[name] for name in differences if name[0] in "lmn")
    if max_position > DIRECT_TOL_MM or max_direction > DIRECT_TOL_DIRECTION:
        raise RuntimeError(f"Direct COM and macro ray differ: {differences}")
    return {"status": status, "compared": True,
            "max_position_difference_mm": max_position,
            "max_direction_difference": max_direction}


def worker(worker_id: int, tasks: object, results: object, stop_event: object,
           work_dir: str, accommodation_values: list[float], ade_values: list[float],
           bde_values: list[float], expected_lens_sha: str, target_thi: float) -> None:
    import pythoncom
    from win32com.client import DispatchEx

    cv = None
    initialized = started = False
    try:
        pythoncom.CoInitialize()
        initialized = True
        cv = DispatchEx("CodeV.Application")
        cv.CommandTimeout = -1
        cv.MaxTextBufferSize = 1000000
        cv.StartingDirectory = str(LENS.parent)
        cv.StartCodeV()
        started = True
        version = wait_ready(cv)
        ade_axis, bde_axis = np.asarray(ade_values), np.asarray(bde_values)
        while not stop_event.is_set():
            ai = tasks.get()
            if ai is None:
                break
            try:
                if sha256(LENS) != expected_lens_sha:
                    raise RuntimeError("P4 .len changed during collection")
                drain_deferred_messages(cv)
                command_checked(cv, f'RES "{LENS}"', "loading no-DM P4 lens")
                baseline = read_accommodation_baseline(cv)
                accommodation = accommodation_values[ai]
                targets = apply_accommodation(cv, baseline, accommodation)
                fov_x, fov_y = angular_fov(cv)
                probe = probe_loaded_lens(cv, fov_x, fov_y, float(ade_axis[0]), float(bde_axis[0]), target_thi)
                seq_path = Path(work_dir) / f"worker_{worker_id}_a{ai:02d}.seq"
                raw_path = Path(work_dir) / f"worker_{worker_id}_a{ai:02d}.txt"
                if raw_path.exists():
                    raw_path.unlink()  # Replace an unfinished run's raw output at the same path.
                write_macro(seq_path, raw_path, ade_axis, bde_axis, fov_x, fov_y, target_thi)
                response = str(cv.Command(f'RUN "{seq_path}";GO'))
                bad = [line for line in response.splitlines() if line.lstrip().lower().startswith(("error:", "*** error"))]
                if bad or not raw_path.is_file():
                    raise RuntimeError(f"CODE V macro failed for A={accommodation:g}: {bad!r}; {response[-2000:]!r}")
                rows, summary = parse_macro(raw_path, ai, accommodation, ade_axis, bde_axis, target_thi)
                summary.update({"worker_id": worker_id, "fov_deg": [fov_x, fov_y],
                                "baseline_geometry": baseline, "target_geometry": targets,
                                "loaded_lens_probe": probe, "direct_comparison": direct_comparison(
                                    cv, rows, fov_x, fov_y, float(ade_axis[0]), float(bde_axis[0]), target_thi
                                ), "codev_version": None,
                                "codev_readiness_response": version.strip()[:300]})
                shard = Path(work_dir) / f"a{ai:02d}.npy"
                np.save(shard, rows, allow_pickle=False)
                seq_path.unlink()
                raw_path.unlink()
                results.put(("done", worker_id, ai, str(shard), summary))
            except Exception:
                stop_event.set()
                results.put(("error", worker_id, ai, None, traceback.format_exc()))
                break
    except Exception:
        stop_event.set()
        results.put(("error", worker_id, None, None, traceback.format_exc()))
    finally:
        if started:
            try:
                cv.StopCodeV()
            except Exception:
                results.put(("error", worker_id, None, None, "StopCodeV failed:\n" + traceback.format_exc()))
        if initialized:
            try:
                pythoncom.CoUninitialize()
            except Exception:
                results.put(("error", worker_id, None, None, "CoUninitialize failed:\n" + traceback.format_exc()))
        results.put(("exit", worker_id, None, None, None))


def collect(work_dir: Path, accommodations: np.ndarray, ade_values: np.ndarray,
            bde_values: np.ndarray, workers: int, lens_sha: str, target_thi: float) -> tuple[list[Path], list[dict]]:
    ctx = mp.get_context("spawn")
    dedicated_pilot = len(accommodations) == workers == 2
    task_queues = [ctx.Queue() for _ in range(workers)] if dedicated_pilot else [ctx.Queue()]
    results, stop_event = ctx.Queue(), ctx.Event()
    processes = [ctx.Process(target=worker, name=f"p4-retrace-{i}", args=(
        i, task_queues[i] if dedicated_pilot else task_queues[0], results, stop_event,
        str(work_dir), accommodations.tolist(),
        ade_values.tolist(), bde_values.tolist(), lens_sha, target_thi,
    )) for i in range(workers)]
    for ai in range(len(accommodations)):
        task_queues[ai if dedicated_pilot else 0].put(ai)
    for i in range(workers):
        task_queues[i if dedicated_pilot else 0].put(None)
    for process in processes:
        process.start()
    shards: dict[int, Path] = {}
    summaries: dict[int, dict] = {}
    errors: list[str] = []
    exited: set[int] = set()
    while len(exited) < len(processes):
        try:
            kind, worker_id, ai, shard, payload = results.get(timeout=5)
        except queue.Empty:
            for i, process in enumerate(processes):
                if not process.is_alive() and i not in exited:
                    exited.add(i)
                    if process.exitcode:
                        errors.append(f"Worker {i} exited with code {process.exitcode}")
                        stop_event.set()
            continue
        if kind == "done":
            if ai in shards:
                errors.append(f"Duplicate shard {ai}")
                stop_event.set()
            else:
                shards[ai], summaries[ai] = Path(shard), payload
                print(f"A={accommodations[ai]:g} D: {payload['rows']:,} rays, {payload['failed_rows']} failed", flush=True)
        elif kind == "error":
            errors.append(f"Worker {worker_id}, A index {ai}:\n{payload}")
            stop_event.set()
        elif kind == "exit":
            exited.add(worker_id)
    for process in processes:
        process.join()
    if errors or len(shards) != len(accommodations):
        raise RuntimeError("Collection failed:\n" + "\n".join(errors) +
                           f"\nCompleted {len(shards)}/{len(accommodations)} accommodations")
    return [shards[i] for i in range(len(accommodations))], [summaries[i] for i in range(len(accommodations))]


def source_comparison(rows: np.ndarray, source_path: Path | None) -> dict:
    """Diagnostic only: source is a different lens and a different gaze axis."""
    if source_path is None:
        return {"skipped": True, "note": "No legacy comparison pickle supplied; diagnostic only"}
    with source_path.open("rb") as stream:
        source = pickle.load(stream)
    if source.size != 21 * 101 * 101 * 9:
        raise ValueError("Legacy source shape changed")
    expected_names = (
        "eye_rotation_deg", "eye_rotation_x_deg", "accommodation_d", "z_mm",
        "field_x_relative", "field_y_relative", "paraxial_x_mm", "paraxial_y_mm",
        "center_absolute_x_mm", "center_absolute_y_mm", "real_x_mm", "real_y_mm",
        "radial_distortion_pct", "tangential_distortion_pct",
    )
    if source.dtype.names != expected_names:
        raise ValueError("Legacy source schema changed")
    source = source.reshape(21, 101, 101, 9)
    expected_source_axes = (
        ("accommodation_d", np.linspace(0, 4, 21)[:, None, None, None]),
        ("eye_rotation_deg", np.linspace(-20, 20, 101)[None, :, None, None]),
        ("eye_rotation_x_deg", np.linspace(-20, 20, 101)[None, None, :, None]),
        ("field_x_relative", np.asarray(FIELDS)[None, None, None, :, 0]),
        ("field_y_relative", np.asarray(FIELDS)[None, None, None, :, 1]),
    )
    for name, expected in expected_source_axes:
        if not np.all(np.isfinite(source[name])) or not np.allclose(source[name], expected, rtol=0, atol=1e-9):
            raise ValueError(f"Legacy source axis/order changed for {name}")
    errors_x, errors_y = [], []
    for row in rows:
        if row["rayrsi_status"] != 0 or row["center_rayrsi_status"] != 0:
            continue
        ia = (row["ade_deg"] + 20.0) / 0.4
        ib = (row["bde_deg"] + 20.0) / 0.4
        aa = row["accommodation_d"] / 0.2
        if not all(abs(v - round(v)) <= 1e-8 for v in (ia, ib, aa)):
            continue
        old = source[round(aa), round(ia), round(ib), int(row["ray_index"])]
        if not math.isfinite(float(old["real_x_mm"])) or not math.isfinite(float(old["real_y_mm"])):
            continue
        errors_x.append(float(row["xsi_local"] - row["center_xsi_local"] - old["real_x_mm"]))
        errors_y.append(float(row["ysi_local"] - row["center_ysi_local"] - old["real_y_mm"]))
    if not errors_x:
        return {"matched_rows": 0, "note": "No shared successful state/ray keys"}
    both = np.hypot(errors_x, errors_y)
    return {"matched_rows": len(errors_x), "rms_vector_mm": float(np.sqrt(np.mean(both**2))),
            "max_vector_mm": float(np.max(both)),
            "note": "Diagnostic residual only: source lens p4_ME.len differs from p4_ME_no_DM.len"}


def publish(run_dir: Path, shards: list[Path], summaries: list[dict],
            accommodations: np.ndarray, ade_values: np.ndarray, bde_values: np.ndarray,
            provenance: dict, source_path: Path | None) -> Path:
    count = len(accommodations) * len(ade_values) * len(bde_values) * len(FIELDS)
    metadata = dict(provenance)
    metadata.update({
        "axes": {"accommodation_d": accommodations.tolist(), "ade_deg": ade_values.tolist(),
                 "bde_deg": bde_values.tolist(), "field_xy_y_major": list(FIELDS)},
        "state_index_formula": "((accommodation_index*n_ade)+ade_index)*n_bde+bde_index",
        "ray_order": "Y 1,0,-1; within each Y, X -1,0,1",
        "units": {"coordinates": "mm", "angles": "deg", "accommodation": "diopters",
                  "direction_cosines": "unitless", "field_xy": "normalized relative input"},
        "global_reference": "G16 for S13, S16, and SI; G16 denotes surface 16",
        "surface_labels": {"S13": "CorneaF_D", "S16": "STP_ENT (physical stop)"},
        "raw_direction_signs_preserved": True,
        "failed_ray_policy": "nonzero rayrsi_status retained; S13/S16/SI and local ray values NaN; center local values masked only when center status is nonzero",
        "thresholds": {"readback_deg": READBACK_TOL_DEG, "readback_mm": READBACK_TOL_MM,
                       "direct_position_mm": DIRECT_TOL_MM, "direct_direction": DIRECT_TOL_DIRECTION,
                       "lmn_norm": LMN_NORM_TOL, "stop_origin_mm": STOP_ORIGIN_TOL_MM,
                       "s13_s16_lmn": S13_S16_LMN_TOL,
                       "segment_alignment": SEGMENT_ALIGNMENT_TOL},
        "record_count": count, "worker_summaries": summaries,
    })
    final_dtype = np.dtype(DTYPE.descr, metadata={"p4_retrace": metadata})
    result = np.empty(count, dtype=final_dtype)
    offset = 0
    for ai, shard in enumerate(shards):
        part = np.load(shard, allow_pickle=False)
        wanted = len(ade_values) * len(bde_values) * len(FIELDS)
        if part.shape != (wanted,) or part.dtype.names != COLS:
            raise ValueError(f"Accommodation shard {ai} has wrong shape or schema")
        if not np.all(part["accommodation_index"] == ai) or not np.array_equal(
            part["state_index"], np.repeat(np.arange(ai * len(ade_values) * len(bde_values),
                                                       (ai + 1) * len(ade_values) * len(bde_values)), len(FIELDS))
        ):
            raise ValueError(f"Accommodation shard {ai} has wrong state order")
        for name in COLS:
            result[name][offset:offset + wanted] = part[name]
        offset += wanted
    if offset != count:
        raise ValueError("Final ray count mismatch")
    metadata["source_comparison"] = source_comparison(result, source_path)
    if "source_comparison" not in result.dtype.metadata["p4_retrace"]:
        raise ValueError("Final dtype did not retain source comparison metadata")
    profile = provenance["profile"]
    suffix = "_file_thickness" if provenance["cornea_ent_d_thi_mode"] == "file" else ""
    filename = "p4_raytrace_251x251_12p5_A4.pkl" if profile == "full" and not suffix else f"p4_{profile}_detail{suffix}.pkl"
    output = OUTPUT_DIR / filename
    temporary = run_dir / (output.name + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    with temporary.open("rb") as stream:
        reopened = pickle.load(stream)
    if not isinstance(reopened, np.ndarray) or reopened.shape != (count,) or reopened.dtype.names != COLS:
        raise ValueError("Temporary pickle did not reopen with the expected count/schema")
    reopened_meta = reopened.dtype.metadata or {}
    if reopened_meta.get("p4_retrace", {}).get("lens_sha256") != provenance["lens_sha256"] or reopened_meta.get("p4_retrace", {}).get("record_count") != count:
        raise ValueError("Temporary pickle lost provenance metadata")
    del reopened
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("pilot", "full"), required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--cornea-ent-d-thi", choices=("zero", "file"), default="zero")
    parser.add_argument("--comparison-pickle", type=Path,
                        help="optional legacy biaxial pickle for diagnostic comparison")
    parser.add_argument("--output-dir", type=Path, help="Run folder; default: runs/p4_retrace_full_detail (pilot: runs/p4_retrace_full_detail_smoke); pickles publish under data/p4_raytrace")
    args = parser.parse_args()
    expected_workers = 2 if args.profile == "pilot" else 4
    if args.workers != expected_workers:
        parser.error(f"{args.profile} profile requires --workers {expected_workers}")
    source_path = args.comparison_pickle.resolve() if args.comparison_pickle else None
    for path in (LENS, LENS_SEQ) + ((source_path,) if source_path is not None else ()):
        if not path.is_file():
            raise FileNotFoundError(path)
    file_thi = verify_sequence_topology()
    target_thi = 0.0 if args.cornea_ent_d_thi == "zero" else file_thi
    accommodations, ade_values, bde_values = axes(args.profile)
    run_dir = (args.output_dir.resolve() if args.output_dir is not None else
               RUNS / ("p4_retrace_full_detail_smoke" if args.profile == "pilot" else "p4_retrace_full_detail"))
    if args.profile == "pilot" and not run_dir.name.endswith("_smoke"):
        run_dir = run_dir.with_name(run_dir.name + "_smoke")
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_dir.name
    provenance = {
        "run_id": run_id, "profile": args.profile, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "workers": args.workers, "lens": str(LENS.relative_to(ROOT)), "lens_sha256": sha256(LENS),
        "lens_seq_sha256": sha256(LENS_SEQ), "source": str(source_path) if source_path else None,
        "source_sha256": sha256(source_path) if source_path else None,
        "script_sha256": sha256(Path(__file__).resolve()),
        "helper_sha256": {name: sha256(ROOT / name) for name in (
            "Script/distortion_grid.py", "Script/distortion_grid_p4.py",
            "Script/distortion_grid_p4_biaxial_macro.py", "Script/distortion_grid_p4_biaxial.py",
            "archive/2026-11-11/Script/raytracing.py",
            "archive/2026-11-11/Script/p1_retrace_full_detail.py")},
        "ray_command": "rayrsi(1,1,0,0,^input)",
        "cornea_ent_d_thi_mode": args.cornea_ent_d_thi, "cornea_ent_d_thi_target_mm": target_thi,
        "field_setup": 'ADE/BDE S"RC"; THI S"Cornea_ENT_D" 0 only in zero mode; SET VIG; XAN/YAN restore; XOB/YOB field-2 writeback',
        "pilot_axis_choice": "five source-overlap angles -12,-6,0,6,12 deg at 0 and 4 D" if args.profile == "pilot" else None,
    }
    try:
        work_dir = run_dir / "_work"
        work_dir.mkdir(parents=True, exist_ok=True)
        shards, summaries = collect(work_dir, accommodations, ade_values,
                                    bde_values, args.workers, provenance["lens_sha256"], target_thi)
        if (sha256(LENS) != provenance["lens_sha256"]
                or sha256(LENS_SEQ) != provenance["lens_seq_sha256"]
                or (source_path is not None and sha256(source_path) != provenance["source_sha256"])):
            raise RuntimeError("Lens, companion sequence, or source changed during run")
        for name, digest in provenance["helper_sha256"].items():
            if sha256(ROOT / name) != digest:
                raise RuntimeError(f"Helper/reference {name} changed during run")
        if sha256(Path(__file__).resolve()) != provenance["script_sha256"]:
            raise RuntimeError("Retriever script changed during run")
        output = publish(run_dir, shards, summaries, accommodations, ade_values, bde_values, provenance, source_path)
        print(f"Published {output} with {len(accommodations) * len(ade_values) * len(bde_values) * 9:,} ray rows", flush=True)
        return 0
    except Exception:
        print(f"Run {run_dir} failed; no passing pickle published:\n{traceback.format_exc()}", flush=True)
        return 1


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
