"""One-session P1 no-DM S4/S8/SI retrace with an audited pilot gate.

The only optical mutations are ADE and BDE on RC. The loaded lens field
settings and SRC_ROT are left untouched. Full traces publish to
data/p1_raytrace/p1_raytrace_251x251_12p5.pkl; pilots use p1_retrace_pilot.pkl.
Chunk checkpoints and temporary files reuse runs/p1_retrace_no_dm_full_detail
(pilot: runs/p1_retrace_no_dm_full_detail_smoke).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import pickle
import re
import shutil
import stat
import time
import traceback
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
LENS = ROOT / "Lens" / "p1_ME_no_DM.seq"
REFERENCE = ROOT / "archive" / "2026-11-11" / "Script" / "p1_retrace_full_detail.py"
RUNS = ROOT / "runs"
OUTPUT_DIR = ROOT / "data" / "p1_raytrace"
FIELDS = tuple((float(x), float(y)) for y in (1, 0, -1) for x in (-1, 0, 1))
OPTICAL = tuple(f"{axis}{surface}" for surface in (4, 8) for axis in "xyzlmn") + ("xsi", "ysi", "zsi")
ANGLES = ("theta_x_deg", "theta_y_deg", "geometric_theta_x_deg", "geometric_theta_y_deg")
COLS = (
    "state_index", "ray_index", "ade_deg", "bde_deg", "field_x_normalized",
    "field_y_normalized", "rc_ade_readback_deg", "rc_bde_readback_deg", "rayrsi_status",
) + OPTICAL + ANGLES
RAW_COLS = COLS[:9] + OPTICAL
DTYPE = np.dtype([(name, np.float64) for name in COLS])
CHUNK_STATES = 250
THRESHOLDS = {
    "readback_deg": 1e-8, "stop_origin_mm": 1e-7, "lmn_norm": 1e-9,
    "s4_s8_lmn": 1e-8, "segment_alignment": 1e-8,
    "angle_deg": 1e-8, "sensor_plane_mm": 1e-9,
    "direct_position_mm": 1e-8, "direct_direction": 1e-9,
    "pilot_position_mm": 1e-9, "pilot_direction": 1e-10,
    "pilot_angle_deg": 1e-8,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def axes(profile: str) -> tuple[float, ...]:
    if profile == "pilot":
        return (-12.5, -6.2, 0.0, 6.2, 12.5)
    return tuple(round(-12.5 + i * 0.1, 10) for i in range(251))


def verify_lens_text() -> None:
    lines = LENS.read_text(encoding="ascii", errors="replace").splitlines()
    labels = {}
    surface = None
    for line in lines:
        if line.startswith("SO "):
            surface = 0
        elif re.match(r"^S\s+", line):
            surface = 1 if surface is None else surface + 1
        match = re.search(r'SLB\s+"([^"]+)"', line)
        if match:
            labels[match.group(1)] = surface
    if labels.get("CorneaF") != 4 or labels.get("STP_ENT") != 8:
        raise RuntimeError(f"Lens companion labels do not put CorneaF/STP_ENT at S4/S8: {labels!r}")
    if not any(re.match(r"^GLO\s+S8\s", line) for line in lines):
        raise RuntimeError("Current P1 lens does not declare GLO S8")
    if not any(re.match(r"^\s*STO\s*$", lines[i]) and i + 1 < len(lines) and
               'SLB "STP_ENT"' in lines[i + 1] for i in range(len(lines))):
        raise RuntimeError("STP_ENT is not marked as the stop")


def cv_eval(cv: object, command: str, *args: str) -> float:
    return float(cv.EvaluateExpression(f"({command} {' '.join(args)})"))


def checked_command(cv: object, command: str, context: str) -> str:
    response = str(cv.Command(command))
    bad = [line for line in response.splitlines() if line.lstrip().lower().startswith(("error:", "fatal:", "*** error"))]
    if "Command End:" not in response or bad:
        raise RuntimeError(f"CODE V failed {context}: {response[-2000:]!r}")
    return response


def wait_ready(cv: object, timeout_s: float = 180.0) -> str:
    start = time.monotonic()
    last = ""
    while time.monotonic() - start < timeout_s:
        try:
            return checked_command(cv, "VER", "readiness")
        except Exception as exc:
            last = repr(exc)
        time.sleep(0.5)
    raise RuntimeError(f"CODE V not ready after {timeout_s:g} s: {last}")


def query_ray(cv: object) -> dict[str, float]:
    result = {}
    for surface in (4, 8):
        for axis in "xyzlmn":
            result[f"{axis}{surface}"] = cv_eval(cv, axis, f"s{surface}", "g8")
    for axis in "xyz":
        result[f"{axis}si"] = cv_eval(cv, axis, "si", "g8")
    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError(f"Direct successful ray has nonfinite output: {result}")
    return result


def probe_loaded_lens(cv: object) -> dict:
    checked_command(cv, 'ADE S"RC" 0;BDE S"RC" 0', "setting zero gaze probe")
    ade_rb, bde_rb = cv_eval(cv, "ade", 's"RC"'), cv_eval(cv, "bde", 's"RC"')
    if not all(math.isfinite(value) and abs(value) <= THRESHOLDS["readback_deg"] for value in (ade_rb, bde_rb)):
        raise RuntimeError(f"Zero-gaze RC readback mismatch: {(ade_rb, bde_rb)}")
    status = int(cv.RAYRSI(1, 1, 0, 0, [0.0, 0.0, 0.0, 0.0]))
    if status:
        raise RuntimeError(f"Zero-gaze center ray failed at surface {status}")
    ray = query_ray(cv)
    label_delta = 0.0
    for number, label in ((4, "CorneaF"), (8, "STP_ENT")):
        for axis in "xyzlmn":
            named = cv_eval(cv, axis, f's"{label}"', "g8")
            if not math.isfinite(named):
                raise RuntimeError(f"Named {label} {axis} query was nonfinite")
            label_delta = max(label_delta, abs(named - ray[f"{axis}{number}"]))
    if label_delta > 1e-9:
        raise RuntimeError(f"Loaded lens numbered/labelled surface mismatch: {label_delta}")
    stop_origin = math.sqrt(sum(ray[f"{a}8"] ** 2 for a in "xyz"))
    if stop_origin > THRESHOLDS["stop_origin_mm"]:
        raise RuntimeError(f"Zero-gaze stop is not at G8 origin: {stop_origin}")
    segment = np.array([ray[f"{a}8"] - ray[f"{a}4"] for a in "xyz"])
    direction = np.array([ray[f"{a}8"] for a in "lmn"])
    length = float(np.linalg.norm(segment))
    if length <= 0 or not math.isfinite(length):
        raise RuntimeError("Zero-gaze S4-to-S8 segment is invalid")
    dot = float(np.dot(direction, segment / length))
    if abs(abs(dot) - 1.0) > THRESHOLDS["segment_alignment"]:
        raise RuntimeError(f"Zero-gaze S8 direction/segment mismatch: {dot}")
    return {
        "status": status, "rc_readback_deg": [ade_rb, bde_rb],
        "s4_xyz_mm": [ray[f"{a}4"] for a in "xyz"],
        "s8_xyz_mm": [ray[f"{a}8"] for a in "xyz"],
        "si_xyz_mm": [ray[f"{a}si"] for a in "xyz"],
        "numeric_label_max_difference": label_delta,
        "s8_direction_segment_dot": dot,
        "s8_direction_to_segment_sign": 1 if dot > 0 else -1,
        "empirical_sensor_z_mm": ray["zsi"],
    }


def macro_text(state_start: int, state_stop: int, values: tuple[float, ...], raw_path: Path) -> str:
    if '"' in str(raw_path):
        raise ValueError("CODE V output path cannot contain a double quote")
    variables = ["^input(4)", "^si", "^ri", "^ade_req", "^bde_req", "^fx", "^fy", "^ade_rb", "^bde_rb", "^st"]
    variables += [f"^{name}" for name in OPTICAL]
    format_string = " ".join(["'3e.17e'"] * len(RAW_COLS))
    lines = ["! P1 no-DM: only RC ADE/BDE mutations; current-ray G8 reads.",
             "LCL NUM " + " ".join(variables), f'OPE NEW U^1 "{raw_path}"',
             "^input(1) == 0", "^input(2) == 0", "OUT N"]
    for state_index in range(state_start, state_stop):
        ade, bde = values[state_index % len(values)], values[state_index // len(values)]
        lines += [f"^si == {state_index}", f"^ade_req == {ade:.10g}",
                  f"^bde_req == {bde:.10g}", 'ADE S"RC" ^ade_req',
                  'BDE S"RC" ^bde_req', '^ade_rb == (ade s"RC")',
                  '^bde_rb == (bde s"RC")']
        for ray_index, (fx, fy) in enumerate(FIELDS):
            lines += [f"^ri == {ray_index}", f"^fx == {fx:g}", f"^fy == {fy:g}",
                      "^input(3) == ^fx", "^input(4) == ^fy",
                      "^st == rayrsi(1,(ref),0,0,^input)"]
            for surface in (4, 8):
                for axis in "xyzlmn":
                    lines.append(f"^{axis}{surface} == ({axis} s{surface} g8)")
            for axis in "xyz":
                lines.append(f"^{axis}si == ({axis} si g8)")
            values_line = "^si ^ri ^ade_req ^bde_req ^fx ^fy ^ade_rb ^bde_rb ^st " + " ".join(f"^{name}" for name in OPTICAL)
            lines.append(f'WRI U^1 Q"{format_string}" {values_line}')
    lines += ["OUT Y", "CLO U^1", "RTN", ""]
    return "\n".join(lines)


def parse_chunk(path: Path, state_start: int, state_stop: int, values: tuple[float, ...],
                orientation_sign: int, sensor_z_mm: float) -> tuple[np.ndarray, dict]:
    expected = (state_stop - state_start) * len(FIELDS)
    table = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if table.shape != (expected, len(RAW_COLS)):
        raise ValueError(f"Chunk table shape {table.shape}, expected {(expected, len(RAW_COLS))}")
    raw = {name: table[:, i] for i, name in enumerate(RAW_COLS)}
    for name in RAW_COLS[:9]:
        if not np.all(np.isfinite(raw[name])):
            raise ValueError(f"Nonfinite key/readback/status in {name}")
    state_indices = np.repeat(np.arange(state_start, state_stop), len(FIELDS))
    ray_indices = np.tile(np.arange(len(FIELDS)), state_stop - state_start)
    ade = np.asarray(values)[state_indices % len(values)]
    bde = np.asarray(values)[state_indices // len(values)]
    fields = np.asarray(FIELDS)[ray_indices]
    wanted = {"state_index": state_indices, "ray_index": ray_indices,
              "ade_deg": ade, "bde_deg": bde,
              "field_x_normalized": fields[:, 0], "field_y_normalized": fields[:, 1]}
    for name, expected_values in wanted.items():
        if np.max(np.abs(raw[name] - expected_values)) > 1e-9:
            raise ValueError(f"Chunk row keys mismatch in {name}")
    if (np.max(np.abs(raw["rc_ade_readback_deg"] - ade)) > THRESHOLDS["readback_deg"]
            or np.max(np.abs(raw["rc_bde_readback_deg"] - bde)) > THRESHOLDS["readback_deg"]):
        raise ValueError("RC ADE/BDE readback mismatch")
    status = raw["rayrsi_status"]
    if not np.all(status == np.rint(status)):
        raise ValueError("Noninteger ray status")
    good = status == 0
    if not np.any(good):
        raise ValueError("Chunk contains no successful ray")
    rows = np.empty(expected, dtype=DTYPE)
    for name in RAW_COLS:
        rows[name] = raw[name]
    for name in OPTICAL:
        rows[name][~good] = math.nan
        if not np.all(np.isfinite(rows[name][good])):
            raise ValueError(f"Successful ray has nonfinite {name}")
    p4 = np.stack([rows[f"{a}4"][good] for a in "xyz"], axis=1)
    p8 = np.stack([rows[f"{a}8"][good] for a in "xyz"], axis=1)
    d4 = np.stack([rows[f"{a}4"][good] for a in "lmn"], axis=1)
    d8 = np.stack([rows[f"{a}8"][good] for a in "lmn"], axis=1)
    segment = p8 - p4
    lengths = np.linalg.norm(segment, axis=1)
    if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
        raise ValueError("Invalid S4-to-S8 segment")
    physical = segment / lengths[:, None]
    signed_dot = np.sum(d8 * physical, axis=1)
    stop_origin = float(np.max(np.linalg.norm(p8, axis=1)))
    norm_error = max(float(np.max(np.abs(np.linalg.norm(d, axis=1) - 1))) for d in (d4, d8))
    direction_difference = float(np.max(np.abs(d4 - d8)))
    alignment_error = float(np.max(np.abs(orientation_sign * signed_dot - 1)))
    sensor_z_error = float(np.max(np.abs(rows["zsi"][good] - sensor_z_mm)))
    theta = np.rad2deg(np.arctan2(orientation_sign * d8[:, :2], orientation_sign * d8[:, 2, None]))
    geometric = np.rad2deg(np.arctan2(physical[:, :2], physical[:, 2, None]))
    angle_error = float(np.max(np.abs((theta - geometric + 180) % 360 - 180)))
    for axis_index, name in enumerate(ANGLES[:2]):
        rows[name] = math.nan
        rows[name][good] = theta[:, axis_index]
    for axis_index, name in enumerate(ANGLES[2:]):
        rows[name] = math.nan
        rows[name][good] = geometric[:, axis_index]
    metrics = {
        "state_start": state_start, "state_stop": state_stop,
        "rows": expected, "successful_rows": int(np.count_nonzero(good)),
        "failed_rows": int(np.count_nonzero(~good)),
        "status_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(status, return_counts=True))},
        "max_stop_origin_mm": stop_origin, "max_lmn_norm_error": norm_error,
        "max_s4_s8_lmn_difference": direction_difference,
        "max_segment_alignment_error": alignment_error,
        "segment_dot_min": float(np.min(signed_dot)), "segment_dot_max": float(np.max(signed_dot)),
        "max_angle_difference_deg": angle_error,
        "max_sensor_z_difference_mm": sensor_z_error,
    }
    gates = (("max_stop_origin_mm", "stop_origin_mm"), ("max_lmn_norm_error", "lmn_norm"),
             ("max_s4_s8_lmn_difference", "s4_s8_lmn"),
             ("max_segment_alignment_error", "segment_alignment"),
             ("max_angle_difference_deg", "angle_deg"),
             ("max_sensor_z_difference_mm", "sensor_plane_mm"))
    for metric, threshold in gates:
        if metrics[metric] > THRESHOLDS[threshold]:
            raise ValueError(f"Chunk {state_start}:{state_stop} {metric}={metrics[metric]} exceeds {THRESHOLDS[threshold]}")
    return rows, metrics


def direct_comparison(cv: object, rows: np.ndarray, first_ade: float, first_bde: float) -> dict:
    checked_command(cv, f'ADE S"RC" {first_ade:.10g};BDE S"RC" {first_bde:.10g}', "direct comparison gaze")
    ade_rb, bde_rb = cv_eval(cv, "ade", 's"RC"'), cv_eval(cv, "bde", 's"RC"')
    if (not all(math.isfinite(value) for value in (ade_rb, bde_rb))
            or abs(ade_rb - first_ade) > THRESHOLDS["readback_deg"]
            or abs(bde_rb - first_bde) > THRESHOLDS["readback_deg"]):
        raise RuntimeError("Direct comparison RC readback mismatch")
    status = int(cv.RAYRSI(1, 1, 0, 0, [0.0, 0.0, 0.0, 0.0]))
    macro = rows[4]
    if status != int(macro["rayrsi_status"]):
        raise RuntimeError(f"Direct status {status} != macro status {macro['rayrsi_status']}")
    if status:
        raise RuntimeError("Direct center comparison ray failed")
    direct = query_ray(cv)
    differences = {name: abs(direct[name] - float(macro[name])) for name in OPTICAL}
    max_position = max(differences[name] for name in OPTICAL if name[0] in "xyz")
    max_direction = max(differences[name] for name in OPTICAL if name[0] in "lmn")
    if max_position > THRESHOLDS["direct_position_mm"] or max_direction > THRESHOLDS["direct_direction"]:
        raise RuntimeError(f"Direct COM/macro mismatch: {differences}")
    return {"status": status, "max_position_difference_mm": max_position,
            "max_direction_difference": max_direction}


def chunk_path(work: Path, chunk_index: int) -> Path:
    return work / f"chunk_{chunk_index:04d}.npy"


def save_shard(path: Path, rows: np.ndarray) -> None:
    temporary = path.with_suffix(".npy.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, rows, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def verify_shard(path: Path, entry: dict, expected_start: int, expected_stop: int) -> None:
    if not path.is_file() or sha256(path) != entry.get("sha256"):
        raise ValueError(f"Checkpoint missing or changed: {path}")
    rows = np.load(path, allow_pickle=False)
    if rows.shape != ((expected_stop - expected_start) * 9,) or rows.dtype.names != COLS:
        raise ValueError(f"Checkpoint shape/schema mismatch: {path}")
    if int(rows["state_index"][0]) != expected_start or int(rows["state_index"][-1]) != expected_stop - 1:
        raise ValueError(f"Checkpoint state range mismatch: {path}")


def compare_pilot(full: np.ndarray, pilot_path: Path, lens_sha: str, script_sha: str) -> dict:
    with pilot_path.open("rb") as stream:
        pilot = pickle.load(stream)
    if not isinstance(pilot, np.ndarray) or pilot.dtype.names != COLS or pilot.size != 225:
        raise ValueError("Pilot pickle schema/count mismatch")
    meta = (pilot.dtype.metadata or {}).get("p1_retrace", {})
    if meta.get("profile") != "pilot" or meta.get("lens_sha256") != lens_sha or meta.get("script_sha256") != script_sha:
        raise ValueError("Pilot profile, lens, or collector hash differs")
    if meta.get("axes_deg") != list(axes("pilot")):
        raise ValueError("Pilot gaze axes differ")
    max_position = max_direction = max_angle = 0.0
    for row in pilot:
        ia = round((float(row["ade_deg"]) + 12.5) / 0.1)
        ib = round((float(row["bde_deg"]) + 12.5) / 0.1)
        if not (0 <= ia < 251 and 0 <= ib < 251):
            raise ValueError("Pilot key lies outside full grid")
        full_row = full[(ib * 251 + ia) * 9 + int(row["ray_index"])]
        for name in ("ade_deg", "bde_deg", "field_x_normalized", "field_y_normalized", "ray_index"):
            if abs(float(full_row[name]) - float(row[name])) > 1e-9:
                raise ValueError(f"Pilot/full key mismatch in {name}")
        if full_row["rayrsi_status"] != row["rayrsi_status"]:
            raise ValueError("Pilot/full status mismatch")
        if row["rayrsi_status"]:
            continue
        for name in OPTICAL:
            difference = abs(float(full_row[name]) - float(row[name]))
            if name[0] in "xyz":
                max_position = max(max_position, difference)
            else:
                max_direction = max(max_direction, difference)
        for name in ANGLES:
            difference = (float(full_row[name]) - float(row[name]) + 180.0) % 360.0 - 180.0
            max_angle = max(max_angle, abs(difference))
    if (max_position > THRESHOLDS["pilot_position_mm"]
            or max_direction > THRESHOLDS["pilot_direction"]
            or max_angle > THRESHOLDS["pilot_angle_deg"]):
        raise ValueError(f"Pilot/full optical mismatch: {(max_position, max_direction, max_angle)}")
    return {"matched_keys": 225, "pilot_sha256": sha256(pilot_path),
            "max_position_difference_mm": max_position,
            "max_direction_difference": max_direction,
            "max_angle_difference_deg": max_angle}


def validate_pilot_preflight(pilot_path: Path, lens_sha: str, script_sha: str) -> None:
    """Reject a stale or incompatible pilot before starting full CODE V work."""
    with pilot_path.open("rb") as stream:
        pilot = pickle.load(stream)
    if not isinstance(pilot, np.ndarray) or pilot.shape != (225,) or pilot.dtype.names != COLS:
        raise ValueError("Pilot pickle schema/count mismatch")
    meta = (pilot.dtype.metadata or {}).get("p1_retrace", {})
    if (meta.get("profile") != "pilot" or meta.get("lens_sha256") != lens_sha
            or meta.get("script_sha256") != script_sha or meta.get("axes_deg") != list(axes("pilot"))
            or meta.get("field_xy") != [list(field) for field in FIELDS]):
        raise ValueError("Pilot profile, lens, script, axes, or fields differ")
    if meta.get("validation", {}).get("rays") != 225:
        raise ValueError("Pilot validation metadata is incomplete")
    if not np.all(np.isfinite(pilot["rayrsi_status"])):
        raise ValueError("Pilot has nonfinite ray statuses")


def remove_work(run_dir: Path, work: Path) -> None:
    """Delete only this run's work directory, including OneDrive readonly files."""
    resolved_run = run_dir.resolve(strict=True)
    resolved_work = work.resolve(strict=True)
    if work.name != "_work" or resolved_work.parent != resolved_run:
        raise ValueError(f"Refusing cleanup outside this run: {resolved_work}")

    def clear_readonly_and_retry(func, failed_path, _exc):
        os.chmod(failed_path, stat.S_IWRITE)
        func(failed_path)

    shutil.rmtree(work, onexc=clear_readonly_and_retry)


def assemble(run_dir: Path, work: Path, manifest: dict, values: tuple[float, ...],
             pilot_path: Path | None) -> Path:
    total_states = len(values) ** 2
    chunks = math.ceil(total_states / CHUNK_STATES)
    count = total_states * 9
    rows = np.empty(count, dtype=DTYPE)
    offset = 0
    summaries = []
    for chunk_index in range(chunks):
        start, stop = chunk_index * CHUNK_STATES, min((chunk_index + 1) * CHUNK_STATES, total_states)
        entry = manifest["completed_chunks"].get(str(chunk_index))
        if not entry:
            raise ValueError(f"Missing chunk {chunk_index}")
        shard = chunk_path(work, chunk_index)
        verify_shard(shard, entry, start, stop)
        part = np.load(shard, allow_pickle=False)
        for name in COLS:
            rows[name][offset:offset + part.size] = part[name]
        offset += part.size
        summaries.append(entry["summary"])
    if offset != count or not np.array_equal(rows["state_index"], np.repeat(np.arange(total_states), 9)):
        raise ValueError("Merged count/state ordering mismatch")
    all_status = {key: sum(summary["status_counts"].get(key, 0) for summary in summaries)
                  for key in sorted({key for summary in summaries for key in summary["status_counts"]})}
    validation = {"states": total_states, "rays": count, "chunks": chunks,
                  "status_counts": all_status,
                  "max_stop_origin_mm": max(s["max_stop_origin_mm"] for s in summaries),
                  "max_lmn_norm_error": max(s["max_lmn_norm_error"] for s in summaries),
                  "max_s4_s8_lmn_difference": max(s["max_s4_s8_lmn_difference"] for s in summaries),
                  "max_segment_alignment_error": max(s["max_segment_alignment_error"] for s in summaries),
                  "max_angle_difference_deg": max(s["max_angle_difference_deg"] for s in summaries),
                  "max_sensor_z_difference_mm": max(s["max_sensor_z_difference_mm"] for s in summaries),
                  "segment_dot_min": min(s["segment_dot_min"] for s in summaries),
                  "segment_dot_max": max(s["segment_dot_max"] for s in summaries),
                  "direct_comparison": manifest["direct_comparison"]}
    if pilot_path is not None:
        validation["pilot_comparison"] = compare_pilot(rows, pilot_path, manifest["config"]["lens_sha256"], manifest["config"]["script_sha256"])
        with pilot_path.open("rb") as stream:
            pilot_meta = pickle.load(stream).dtype.metadata["p1_retrace"]
        if abs(float(pilot_meta["empirical_sensor_z_mm"]) - float(manifest["probe"]["empirical_sensor_z_mm"])) > THRESHOLDS["sensor_plane_mm"]:
            raise ValueError("Pilot/full sensor plane differs")
    metadata = {**manifest["config"],
                "run_id": manifest["run_id"], "created_utc": manifest["created_utc"],
                "ray_order": "Y=1,0,-1; X=-1,0,1 within each Y",
                "state_order": "BDE outer, ADE inner; state_index=iBDE*nADE+iADE",
                "frame": "S4/S8 XYZLMN and SI XYZ in G8 global reference; S8 physical stop at origin",
                "units": {"coordinates": "mm", "angles": "degrees", "field_xy": "normalized relative",
                          "direction_cosines": "unitless"},
                "source_rotation_policy": "Loaded lens source settings untouched; collector changes only RC ADE/BDE",
                "raw_direction_signs_preserved": True,
                "derived_angles": "physical global atan2: theta from empirically sign-aligned S8 LMN; geometric theta from S4-to-S8 segment",
                "failed_ray_policy": "nonzero status retained, all optical and derived-angle columns NaN",
                "thresholds": THRESHOLDS, "probe": manifest["probe"],
                "empirical_sensor_z_mm": manifest["probe"]["empirical_sensor_z_mm"],
                "codev_version": None,
                "codev_readiness_response": manifest.get("readiness_response"),
                "validation": validation}
    result = np.empty(count, dtype=np.dtype(DTYPE.descr, metadata={"p1_retrace": metadata}))
    for name in COLS:
        result[name] = rows[name]
    filename = "p1_raytrace_251x251_12p5.pkl" if manifest["config"]["profile"] == "full" else "p1_retrace_pilot.pkl"
    output = OUTPUT_DIR / filename
    temporary = run_dir / (filename + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    with temporary.open("rb") as stream:
        reopened = pickle.load(stream)
    if (not isinstance(reopened, np.ndarray) or reopened.shape != result.shape
            or reopened.dtype.names != COLS
            or reopened.dtype.metadata["p1_retrace"]["validation"] != validation
            or not np.array_equal(reopened.view(np.uint8), result.view(np.uint8))):
        raise ValueError("Temporary pickle reload did not match validated rows and metadata")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, output)
    return output


def collect(run_dir: Path, work: Path, manifest: dict, values: tuple[float, ...]) -> None:
    import pythoncom
    from win32com.client import DispatchEx

    total_states = len(values) ** 2
    chunks = math.ceil(total_states / CHUNK_STATES)
    pending = [i for i in range(chunks) if str(i) not in manifest["completed_chunks"]]
    if not pending:
        return
    initialized = started = False
    cv = None
    try:
        pythoncom.CoInitialize()
        initialized = True
        cv = DispatchEx("CodeV.Application")
        cv.CommandTimeout = -1
        cv.MaxTextBufferSize = 1000000
        cv.StartingDirectory = str(LENS.parent)
        cv.StartCodeV()
        started = True
        manifest["readiness_response"] = wait_ready(cv).strip()[:300]
        checked_command(cv, f'RUN "{LENS}";GO', "loading current P1 sequence")
        probe = probe_loaded_lens(cv)
        if manifest.get("probe") is not None:
            old = manifest["probe"]
            if (old["s8_direction_to_segment_sign"] != probe["s8_direction_to_segment_sign"]
                    or abs(old["empirical_sensor_z_mm"] - probe["empirical_sensor_z_mm"]) > THRESHOLDS["sensor_plane_mm"]):
                raise RuntimeError("Resume lens probe differs from original")
        manifest["probe"] = probe
        atomic_json(work / "manifest.json", manifest)
        for chunk_index in pending:
            start, stop = chunk_index * CHUNK_STATES, min((chunk_index + 1) * CHUNK_STATES, total_states)
            macro_path, raw_path = work / f"chunk_{chunk_index:04d}.seq", work / f"chunk_{chunk_index:04d}.txt"
            if raw_path.exists():
                raw_path.unlink()  # Replace an unfinished run's raw output at the same path.
            macro_path.write_text(macro_text(start, stop, values, raw_path), encoding="ascii")
            response = str(cv.Command(f'RUN "{macro_path}";GO'))
            bad = [line for line in response.splitlines() if line.lstrip().lower().startswith(("error:", "fatal:", "*** error"))]
            if bad or not raw_path.is_file():
                raise RuntimeError(f"Chunk {chunk_index} macro failed: {bad!r}; {response[-2000:]!r}")
            part, summary = parse_chunk(raw_path, start, stop, values,
                                        probe["s8_direction_to_segment_sign"],
                                        probe["empirical_sensor_z_mm"])
            if "direct_comparison" not in manifest:
                manifest["direct_comparison"] = direct_comparison(cv, part, values[0], values[0])
            shard = chunk_path(work, chunk_index)
            save_shard(shard, part)
            manifest["completed_chunks"][str(chunk_index)] = {"sha256": sha256(shard), "summary": summary}
            atomic_json(work / "manifest.json", manifest)
            macro_path.unlink()
            raw_path.unlink()
            print(f"Chunk {chunk_index + 1}/{chunks}: states {stop}/{total_states}, successful rays {summary['successful_rows']}/{summary['rows']}", flush=True)
    finally:
        cleanup_errors = []
        if started:
            try:
                cv.StopCodeV()
            except Exception:
                cleanup_errors.append("StopCodeV failed: " + traceback.format_exc())
        if initialized:
            try:
                pythoncom.CoUninitialize()
            except Exception:
                cleanup_errors.append("CoUninitialize failed: " + traceback.format_exc())
        if cleanup_errors:
            raise RuntimeError("\n".join(cleanup_errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("pilot", "full"), required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--pilot-pickle", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output-dir", type=Path, help="Run folder; default: runs/p1_retrace_no_dm_full_detail (pilot: runs/p1_retrace_no_dm_full_detail_smoke); pickles publish under data/p1_raytrace")
    args = parser.parse_args()
    if args.resume is not None and args.output_dir is not None:
        parser.error("use --resume or --output-dir, not both")
    if args.workers != 1:
        parser.error("P1 retrace requires exactly one CODE V worker")
    if args.profile == "full" and args.pilot_pickle is None:
        parser.error("full profile requires the audited --pilot-pickle")
    if args.profile == "pilot" and args.pilot_pickle is not None:
        parser.error("pilot profile does not take --pilot-pickle")
    if not LENS.is_file() or not REFERENCE.is_file():
        raise FileNotFoundError("Current P1 lens or reference collector is missing")
    verify_lens_text()
    values = axes(args.profile)
    pilot_path = args.pilot_pickle.resolve() if args.pilot_pickle else None
    if pilot_path is not None and not pilot_path.is_file():
        raise FileNotFoundError(pilot_path)
    current_config = {
        "profile": args.profile, "workers": 1, "lens": str(LENS.relative_to(ROOT)),
        "lens_sha256": sha256(LENS), "script_sha256": sha256(Path(__file__).resolve()),
        "reference_script_sha256": sha256(REFERENCE),
        "axes_deg": list(values), "field_xy": [list(field) for field in FIELDS],
        "pilot_pickle": str(pilot_path) if pilot_path else None,
        "pilot_sha256": sha256(pilot_path) if pilot_path else None,
        "ray_command": "rayrsi(1,(ref),0,0,^input)",
        "optical_mutations": ['ADE S"RC"', 'BDE S"RC"'],
    }
    if pilot_path is not None:
        validate_pilot_preflight(pilot_path, current_config["lens_sha256"], current_config["script_sha256"])
    if args.resume:
        run_dir = args.resume.resolve()
        work = run_dir / "_work"
        manifest = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
        if manifest["config"] != current_config:
            raise ValueError("Resume configuration, lens, script, or pilot changed")
        for key, entry in manifest["completed_chunks"].items():
            i = int(key)
            verify_shard(chunk_path(work, i), entry, i * CHUNK_STATES,
                         min((i + 1) * CHUNK_STATES, len(values) ** 2))
    else:
        run_dir = (args.output_dir.resolve() if args.output_dir is not None else
                   RUNS / ("p1_retrace_no_dm_full_detail_smoke" if args.profile == "pilot" else "p1_retrace_no_dm_full_detail"))
        if args.profile == "pilot" and not run_dir.name.endswith("_smoke"):
            run_dir = run_dir.with_name(run_dir.name + "_smoke")
        run_dir.mkdir(parents=True, exist_ok=True)
        run_id = run_dir.name
        work = run_dir / "_work"
        work.mkdir(parents=True, exist_ok=True)
        manifest = {"run_id": run_id, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "config": current_config, "probe": None, "completed_chunks": {}}
        atomic_json(work / "manifest.json", manifest)
    try:
        collect(run_dir, work, manifest, values)
        if (sha256(LENS) != current_config["lens_sha256"]
                or sha256(Path(__file__).resolve()) != current_config["script_sha256"]
                or sha256(REFERENCE) != current_config["reference_script_sha256"]
                or (pilot_path is not None and sha256(pilot_path) != current_config["pilot_sha256"])):
            raise RuntimeError("Lens, script, reference, or pilot changed during collection")
        output = assemble(run_dir, work, manifest, values, pilot_path)
        remove_work(run_dir, work)
        print(f"Published {output}: {len(values) ** 2:,} states, {len(values) ** 2 * 9:,} rays", flush=True)
        return 0
    except Exception:
        print(f"P1 run {run_dir} stopped; _work retained for resume:\n{traceback.format_exc()}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
