"""Resumable P1 RC sweep with global S4, S8, and sensor SI ray data."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import pickle
import time
import uuid
from pathlib import Path

import numpy as np
import psutil

from p1_retrace_gaze_11x11 import AXES, FIELDS, LENS, ROOT, norm, wrap_angle_degrees
from p1_retrace_smoke import codev_path, wait_ready


OUTPUT_ROOT = ROOT / "data" / "p1_retrace_full_detail_251x251"
BASELINE_CSV = ROOT / "data" / "p1_retrace_full_251x251" / "20261109T201039Z_e76ad3b8" / "p1_retrace_251x251.csv"
SENSOR_Z_MM = 45.564056516201
ANGLES = tuple(round(-12.5 + index * 0.1, 10) for index in range(251))
STATE_COUNT = len(ANGLES) ** 2
RAYS_PER_STATE = len(FIELDS)
ROW_COUNT = STATE_COUNT * RAYS_PER_STATE
CHUNK_STATES = 250
CHUNK_COUNT = math.ceil(STATE_COUNT / CHUNK_STATES)
COLS = ["state_index", "ray_index", "ade_deg", "bde_deg", "field_x_normalized", "field_y_normalized", "rc_ade_readback_deg", "rc_bde_readback_deg", "rayrsi_status"]
COLS += [f"{axis}{surface}" for surface in (4, 8) for axis in AXES]
COLS += ["xsi", "ysi", "zsi"]
COLS += ["theta_x_deg", "theta_y_deg", "geometric_theta_x_deg", "geometric_theta_y_deg"]
THRESHOLDS = {"readback_deg": 1e-8, "stop_radius_mm": 1e-7, "lmn_unit": 1e-9, "direction_component": 1e-8, "angle_deg": 1e-8, "sensor_z_mm": 1e-9}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def state_values(state_index: int) -> tuple[float, float]:
    return ANGLES[state_index % len(ANGLES)], ANGLES[state_index // len(ANGLES)]


def chunk_states(chunk_index: int) -> range:
    start = chunk_index * CHUNK_STATES
    return range(start, min(start + CHUNK_STATES, STATE_COUNT))


def macro_text(chunk_index: int, raw_path: Path) -> str:
    variables = ["^input(4)", "^si", "^ri", "^ade_req", "^bde_req", "^fx", "^fy", "^ade_rb", "^bde_rb", "^st"]
    variables += [f"^{axis}{surface}" for surface in (4, 8) for axis in AXES]
    variables += ["^xsi", "^ysi", "^zsi"]
    metadata_format = " ".join(["'3e.17e'"] * 9)
    ray_format = " ".join(["'3e.17e'"] * 15)
    lines = [
        f"! P1 full sweep chunk {chunk_index}; only RC ADE/BDE mutations.",
        "LCL NUM " + " ".join(variables),
        f'OPE NEW U^1 "{codev_path(raw_path)}"',
        "^input(1) == 0", "^input(2) == 0",
    ]
    for state_index in chunk_states(chunk_index):
        ade, bde = state_values(state_index)
        lines.extend((
            f"^si == {state_index}",
            f"^ade_req == {ade:.10g}", f"^bde_req == {bde:.10g}",
            'ADE S"RC" ^ade_req', 'BDE S"RC" ^bde_req',
            '^ade_rb == (ade s"RC")', '^bde_rb == (bde s"RC")',
        ))
        for ray_index, (fx, fy) in enumerate(FIELDS):
            lines.extend((
                f"^ri == {ray_index}", f"^fx == {fx:g}", f"^fy == {fy:g}",
                "^input(3) == ^fx", "^input(4) == ^fy",
                "^st == rayrsi(1,(ref),0,0,^input)",
            ))
            for surface in (4, 8):
                for axis in AXES:
                    lines.append(f"^{axis}{surface} == ({axis} s{surface} g8)")
            for axis in ("x", "y", "z"):
                lines.append(f"^{axis}si == ({axis} si g8)")
            lines.append(f'WRI U^1 Q"{metadata_format}" ^si ^ri ^ade_req ^bde_req ^fx ^fy ^ade_rb ^bde_rb ^st')
            ray_values = " ".join(f"^{axis}{surface}" for surface in (4, 8) for axis in AXES) + " ^xsi ^ysi ^zsi"
            lines.append(f'WRI U^1 Q"{ray_format}" {ray_values}')
    lines.extend(("CLO U^1", "RTN", ""))
    return "\n".join(lines)


def finite_metrics(values: list[float]) -> tuple[list[float], float, float, float, float, float, float, float, float, float]:
    p4, p8, sensor = values[:6], values[6:12], values[12:15]
    d4, d8 = p4[3:], p8[3:]
    segment = [p8[index] - p4[index] for index in range(3)]
    segment_length = norm(segment)
    if segment_length <= 0:
        raise ValueError("Zero-length S4-to-S8 segment")
    physical = [value / segment_length for value in segment]
    stop_direction = [-value for value in d8]
    theta_x = math.degrees(math.atan2(stop_direction[0], stop_direction[2]))
    theta_y = math.degrees(math.atan2(stop_direction[1], stop_direction[2]))
    geom_x = math.degrees(math.atan2(physical[0], physical[2]))
    geom_y = math.degrees(math.atan2(physical[1], physical[2]))
    stop_radius = math.hypot(p8[0], p8[1])
    norm_error = max(abs(norm(d4) - 1), abs(norm(d8) - 1))
    direction_error = max(abs(a - b) for a, b in zip(d4, d8))
    sign_error = max(abs(a + b) for a, b in zip(d8, physical))
    angle_error = max(abs(wrap_angle_degrees(theta_x - geom_x)), abs(wrap_angle_degrees(theta_y - geom_y)))
    sensor_z_error = abs(sensor[2] - SENSOR_Z_MM)
    return [theta_x, theta_y, geom_x, geom_y], stop_radius, norm_error, direction_error, sign_error, angle_error, theta_x, theta_y, segment_length, sensor_z_error


def process_chunk(cv: object, chunk_index: int, attempt_dir: Path, config_hash: str) -> dict[str, object]:
    attempt_dir.mkdir(parents=True, exist_ok=False)
    raw_path = attempt_dir / "codev_raw.txt"
    macro_path = attempt_dir / "issued_macro.seq"
    macro_path.write_text(macro_text(chunk_index, raw_path), encoding="ascii")
    response = str(cv.Command(f'run "{codev_path(macro_path)}";go'))
    (attempt_dir / "codev_response.txt").write_text(response, encoding="utf-8")
    if not raw_path.is_file():
        raise RuntimeError(f"Chunk {chunk_index}: CODE V did not write raw output")
    expected = len(chunk_states(chunk_index)) * RAYS_PER_STATE
    csv_path = attempt_dir / "rays.csv"
    csv_tmp = csv_path.with_suffix(".csv.tmp")
    valid_count = failure_count = 0
    max_stop = max_norm = max_direction = max_sign = max_angle = max_sensor_z = 0.0
    sensor_x_min = sensor_y_min = math.inf
    sensor_x_max = sensor_y_max = -math.inf
    theta_x_min = theta_y_min = math.inf
    theta_x_max = theta_y_max = -math.inf
    readback_errors = 0
    status_counts: dict[str, int] = {}
    with raw_path.open("r", encoding="utf-8", errors="replace") as raw, csv_tmp.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(COLS)
        for state_index in chunk_states(chunk_index):
            ade, bde = state_values(state_index)
            for ray_index, (fx, fy) in enumerate(FIELDS):
                metadata_line, ray_line = raw.readline(), raw.readline()
                if not metadata_line or not ray_line:
                    raise ValueError(f"Chunk {chunk_index}: missing raw row for state {state_index}, ray {ray_index}")
                metadata = [float(token.replace("D", "E")) for token in metadata_line.split()]
                values = [float(token.replace("D", "E")) for token in ray_line.split()]
                if len(metadata) != 9 or len(values) != 15:
                    raise ValueError(f"Chunk {chunk_index}: malformed row at state {state_index}, ray {ray_index}")
                wanted = (state_index, ray_index, ade, bde, fx, fy)
                if any(abs(got - want) > 1e-9 for got, want in zip(metadata[:6], wanted)):
                    raise ValueError(f"Chunk {chunk_index}: key mismatch at state {state_index}, ray {ray_index}")
                if abs(metadata[6] - ade) > THRESHOLDS["readback_deg"] or abs(metadata[7] - bde) > THRESHOLDS["readback_deg"]:
                    readback_errors += 1
                status = metadata[8]
                status_counts[str(status)] = status_counts.get(str(status), 0) + 1
                valid = math.isfinite(status) and abs(status) < 1e-12 and all(math.isfinite(value) for value in values)
                if valid:
                    try:
                        angles, stop, norm_error, direction, sign, angle_error, theta_x, theta_y, _, sensor_z_error = finite_metrics(values)
                    except ValueError:
                        valid = False
                if valid:
                    valid_count += 1
                    max_stop = max(max_stop, stop)
                    max_norm = max(max_norm, norm_error)
                    max_direction = max(max_direction, direction)
                    max_sign = max(max_sign, sign)
                    max_angle = max(max_angle, angle_error)
                    max_sensor_z = max(max_sensor_z, sensor_z_error)
                    theta_x_min, theta_x_max = min(theta_x_min, theta_x), max(theta_x_max, theta_x)
                    theta_y_min, theta_y_max = min(theta_y_min, theta_y), max(theta_y_max, theta_y)
                    sensor_x_min, sensor_x_max = min(sensor_x_min, values[12]), max(sensor_x_max, values[12])
                    sensor_y_min, sensor_y_max = min(sensor_y_min, values[13]), max(sensor_y_max, values[13])
                    writer.writerow(metadata + values + angles)
                else:
                    failure_count += 1
                    writer.writerow(metadata + ["nan"] * 19)
        if raw.readline().strip():
            raise ValueError(f"Chunk {chunk_index}: unexpected extra raw output")
    os.replace(csv_tmp, csv_path)
    source_after = float(cv.EvaluateExpression('(ade s"SRC_ROT")'))
    quality = (
        readback_errors == 0 and abs(source_after + 20) <= THRESHOLDS["readback_deg"]
        and max_stop <= THRESHOLDS["stop_radius_mm"] and max_norm <= THRESHOLDS["lmn_unit"]
        and max_direction <= THRESHOLDS["direction_component"] and max_sign <= THRESHOLDS["direction_component"]
        and max_angle <= THRESHOLDS["angle_deg"] and max_sensor_z <= THRESHOLDS["sensor_z_mm"]
    )
    summary: dict[str, object] = {
        "chunk_index": chunk_index, "state_start": chunk_states(chunk_index).start,
        "state_end_exclusive": chunk_states(chunk_index).stop,
        "rows": expected, "valid_rows": valid_count, "failed_rows": failure_count,
        "readback_errors": readback_errors, "source_ade_after": source_after,
        "max_s8_xy_radius_mm": max_stop, "max_lmn_norm_error": max_norm,
        "max_s4_s8_lmn_diff": max_direction, "max_raw_lmn_plus_segment_diff": max_sign,
        "max_stop_angle_diff_deg": max_angle,
        "max_sensor_z_error_mm": max_sensor_z,
        "sensor_x_range_mm": [sensor_x_min, sensor_x_max] if valid_count else None,
        "sensor_y_range_mm": [sensor_y_min, sensor_y_max] if valid_count else None,
        "theta_x_range_deg": [theta_x_min, theta_x_max] if valid_count else None,
        "theta_y_range_deg": [theta_y_min, theta_y_max] if valid_count else None,
        "status_counts": status_counts, "quality_pass": quality,
        "config_hash": config_hash, "raw_sha256": digest(raw_path),
        "macro_sha256": digest(macro_path), "csv_sha256": digest(csv_path),
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    atomic_json(attempt_dir / "summary.json", summary)
    return summary


def backend_snapshot() -> dict[int, dict[str, object]]:
    found = {}
    for process in psutil.process_iter(["pid", "name", "exe", "create_time"]):
        try:
            info = process.info
            if str(info.get("name", "")).lower() == "codevm.exe":
                found[int(info["pid"])] = {"pid": int(info["pid"]), "exe": info.get("exe"), "create_time": info.get("create_time")}
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found


def config() -> dict[str, object]:
    value = {
        "lens": str(LENS), "lens_sha256": digest(LENS),
        "baseline_csv": str(BASELINE_CSV), "baseline_csv_sha256": digest(BASELINE_CSV),
        "collector_sha256": digest(Path(__file__).resolve()),
        "angles_deg": ANGLES, "fields_xy": FIELDS,
        "state_order": "BDE major, ADE minor", "ray_order": "Y 1,0,-1; X -1,0,1",
        "state_count": STATE_COUNT, "row_count": ROW_COUNT,
        "chunk_states": CHUNK_STATES, "chunk_count": CHUNK_COUNT,
        "workers": 1, "source_ade_deg": -20,
        "ray_command": "rayrsi(1,(ref),0,0,^input)",
        "input_pupil_xy": [0, 0], "query": "unqualified XYZLMN S4/S8 G8",
        "only_optical_mutations": ['ADE S"SRC_ROT" -20', 'ADE S"RC"', 'BDE S"RC"'],
        "csv_columns": COLS, "thresholds": THRESHOLDS,
        "sensor_z_mm": SENSOR_Z_MM, "sensor_query": "unqualified XYZ SI G8 after each RAYRSI",
        "pickle_format": "flat NumPy structured ndarray, all CSV columns float64, pickle.HIGHEST_PROTOCOL",
    }
    value["config_sha256"] = hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()
    return value


def verify_completed(run_dir: Path, manifest: dict[str, object], chunk_index: int) -> bool:
    entry = manifest["completed_chunks"].get(str(chunk_index))
    if not entry:
        return False
    folder = run_dir / entry["path"]
    try:
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        return (
            summary["config_hash"] == manifest["config"]["config_sha256"]
            and summary["quality_pass"] and summary["rows"] == len(chunk_states(chunk_index)) * RAYS_PER_STATE
            and digest(folder / "codev_raw.txt") == summary["raw_sha256"]
            and digest(folder / "issued_macro.seq") == summary["macro_sha256"]
            and digest(folder / "rays.csv") == summary["csv_sha256"]
        )
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False


def compare_baseline(csv_path: Path) -> dict[str, object]:
    """Compare all overlapping rays against the prior 25-column full run."""
    full = len(ANGLES) == 251
    old_names = [name for name in COLS if name not in ("xsi", "ysi", "zsi")]
    coordinates = [f"{axis}{surface}" for surface in (4, 8) for axis in ("x", "y", "z")]
    directions = [f"{axis}{surface}" for surface in (4, 8) for axis in ("l", "m", "n")]
    angles = ["theta_x_deg", "theta_y_deg", "geometric_theta_x_deg", "geometric_theta_y_deg"]

    def key(row: dict[str, str]) -> tuple[float, float, float, float]:
        return tuple(round(float(row[name]), 10) for name in ("ade_deg", "bde_deg", "field_x_normalized", "field_y_normalized"))

    metrics = {"compared_rows": 0, "max_coordinate_diff_mm": 0.0, "max_direction_diff": 0.0, "max_angle_diff_deg": 0.0, "status_mismatches": 0, "key_mismatches": 0}

    def compare(new: dict[str, str], old: dict[str, str]) -> None:
        metrics["compared_rows"] += 1
        if key(new) != key(old) or int(float(new["rayrsi_status"])) != int(float(old["rayrsi_status"])):
            metrics["key_mismatches"] += key(new) != key(old)
            metrics["status_mismatches"] += int(float(new["rayrsi_status"])) != int(float(old["rayrsi_status"]))
        for name in coordinates:
            metrics["max_coordinate_diff_mm"] = max(metrics["max_coordinate_diff_mm"], abs(float(new[name]) - float(old[name])))
        for name in directions:
            metrics["max_direction_diff"] = max(metrics["max_direction_diff"], abs(float(new[name]) - float(old[name])))
        for name in angles:
            metrics["max_angle_diff_deg"] = max(metrics["max_angle_diff_deg"], abs(float(new[name]) - float(old[name])))

    with csv_path.open("r", newline="", encoding="utf-8") as new_stream, BASELINE_CSV.open("r", newline="", encoding="utf-8") as old_stream:
        new_reader, old_reader = csv.DictReader(new_stream), csv.DictReader(old_stream)
        if list(old_reader.fieldnames or []) != old_names or list(new_reader.fieldnames or []) != COLS:
            raise ValueError("Baseline or new CSV schema mismatch")
        if full:
            for new, old in zip(new_reader, old_reader):
                compare(new, old)
        else:
            overlap = {-12.5, 0.0, 12.5}
            wanted = {key(row): row for row in new_reader if float(row["ade_deg"]) in overlap and float(row["bde_deg"]) in overlap}
            for old in old_reader:
                old_key = key(old)
                if old_key in wanted:
                    compare(wanted.pop(old_key), old)
            metrics["missing_overlap_keys"] = len(wanted)
    expected = ROW_COUNT if full else 81
    metrics["expected_overlap_rows"] = expected
    metrics["pass"] = (
        metrics["compared_rows"] == expected and metrics.get("missing_overlap_keys", 0) == 0
        and metrics["key_mismatches"] == 0 and metrics["status_mismatches"] == 0
        and metrics["max_coordinate_diff_mm"] <= 1e-9
        and metrics["max_direction_diff"] <= 1e-10
        and metrics["max_angle_diff_deg"] <= 1e-8
    )
    return metrics


def merge_and_pickle(run_dir: Path, manifest: dict[str, object]) -> dict[str, object]:
    csv_path = run_dir / f"p1_retrace_detail_{len(ANGLES)}x{len(ANGLES)}.csv"
    csv_tmp = csv_path.with_suffix(".csv.tmp")
    count = 0
    with csv_tmp.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(COLS)
        for chunk_index in range(CHUNK_COUNT):
            entry = manifest["completed_chunks"][str(chunk_index)]
            folder = run_dir / entry["path"]
            with (folder / "rays.csv").open("r", newline="", encoding="utf-8") as input_stream:
                reader = csv.reader(input_stream)
                if next(reader) != COLS:
                    raise ValueError(f"Chunk {chunk_index} CSV schema mismatch")
                for row in reader:
                    if len(row) != len(COLS):
                        raise ValueError(f"Chunk {chunk_index} short CSV row")
                    if int(float(row[0])) != count // RAYS_PER_STATE or int(float(row[1])) != count % RAYS_PER_STATE:
                        raise ValueError(f"Unexpected key order at merged row {count}")
                    writer.writerow(row)
                    count += 1
    if count != ROW_COUNT:
        raise ValueError(f"Merged CSV has {count} rows, expected {ROW_COUNT}")
    os.replace(csv_tmp, csv_path)
    dtype = np.dtype([(name, np.float64) for name in COLS])

    def tuples():
        with csv_path.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.reader(stream)
            if next(reader) != COLS:
                raise ValueError("Final CSV header mismatch")
            for row in reader:
                yield tuple(float(value) for value in row)

    array = np.fromiter(tuples(), dtype=dtype, count=ROW_COUNT)
    if array.shape != (ROW_COUNT,) or array.dtype.names != tuple(COLS):
        raise ValueError("Structured-array conversion failed")
    pickle_path = run_dir / f"p1_retrace_detail_{len(ANGLES)}x{len(ANGLES)}.pkl"
    pickle_tmp = pickle_path.with_suffix(".pkl.tmp")
    with pickle_tmp.open("wb") as stream:
        pickle.dump(array, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(pickle_tmp, pickle_path)
    with pickle_path.open("rb") as stream:
        loaded = pickle.load(stream)
    if loaded.shape != array.shape or loaded.dtype != array.dtype or not all(np.array_equal(loaded[name], array[name], equal_nan=True) for name in COLS):
        raise ValueError("Pickle reload does not equal merged CSV structured array")
    summaries = [json.loads(((run_dir / manifest["completed_chunks"][str(index)]["path"]) / "summary.json").read_text(encoding="utf-8")) for index in range(CHUNK_COUNT)]
    failed_rows = sum(summary["failed_rows"] for summary in summaries)
    readback_errors = sum(summary["readback_errors"] for summary in summaries)
    aggregate = {
        "csv": csv_path.name, "csv_sha256": digest(csv_path),
        "pickle": pickle_path.name, "pickle_sha256": digest(pickle_path),
        "rows": ROW_COUNT, "states": STATE_COUNT, "chunks": CHUNK_COUNT,
        "failed_rows": failed_rows, "readback_errors": readback_errors,
        "max_s8_xy_radius_mm": max(s["max_s8_xy_radius_mm"] for s in summaries),
        "max_lmn_norm_error": max(s["max_lmn_norm_error"] for s in summaries),
        "max_s4_s8_lmn_diff": max(s["max_s4_s8_lmn_diff"] for s in summaries),
        "max_raw_lmn_plus_segment_diff": max(s["max_raw_lmn_plus_segment_diff"] for s in summaries),
        "max_stop_angle_diff_deg": max(s["max_stop_angle_diff_deg"] for s in summaries),
        "max_sensor_z_error_mm": max(s["max_sensor_z_error_mm"] for s in summaries),
        "sensor_x_range_mm": [min(s["sensor_x_range_mm"][0] for s in summaries if s["sensor_x_range_mm"]), max(s["sensor_x_range_mm"][1] for s in summaries if s["sensor_x_range_mm"])],
        "sensor_y_range_mm": [min(s["sensor_y_range_mm"][0] for s in summaries if s["sensor_y_range_mm"]), max(s["sensor_y_range_mm"][1] for s in summaries if s["sensor_y_range_mm"])],
        "theta_x_range_deg": [min(s["theta_x_range_deg"][0] for s in summaries if s["theta_x_range_deg"]), max(s["theta_x_range_deg"][1] for s in summaries if s["theta_x_range_deg"])],
        "theta_y_range_deg": [min(s["theta_y_range_deg"][0] for s in summaries if s["theta_y_range_deg"]), max(s["theta_y_range_deg"][1] for s in summaries if s["theta_y_range_deg"])],
        "pickle_shape": list(array.shape), "pickle_dtype": str(array.dtype),
    }
    aggregate["validation_pass"] = (
        failed_rows == 0 and readback_errors == 0
        and aggregate["max_s8_xy_radius_mm"] <= THRESHOLDS["stop_radius_mm"]
        and aggregate["max_lmn_norm_error"] <= THRESHOLDS["lmn_unit"]
        and aggregate["max_s4_s8_lmn_diff"] <= THRESHOLDS["direction_component"]
        and aggregate["max_raw_lmn_plus_segment_diff"] <= THRESHOLDS["direction_component"]
        and aggregate["max_stop_angle_diff_deg"] <= THRESHOLDS["angle_deg"]
        and aggregate["max_sensor_z_error_mm"] <= THRESHOLDS["sensor_z_mm"]
    )
    aggregate["baseline_comparison"] = compare_baseline(csv_path)
    aggregate["validation_pass"] = aggregate["validation_pass"] and aggregate["baseline_comparison"]["pass"]
    return aggregate


def report_text(manifest: dict[str, object]) -> str:
    final = manifest["final"]
    return "\n".join((
        f"# P1 {len(ANGLES)}×{len(ANGLES)} RC detail sweep",
        "",
        f"Run `{manifest['run_id']}`: {final['states']} ADE/BDE states, {final['rows']} rays, {final['chunks']} resumable chunks, one CODE V session. Final validation: {'passed' if final['validation_pass'] else 'FAILED'}.",
        "",
        f"RC ADE and BDE each use {list(ANGLES)} degrees. Each state has nine normalized field rays with pupil input (0,0). Source ADE was set to −20° in memory. The lens file and XAN/YAN/XOB/YOB/VIG settings were unchanged.",
        "",
        "Each ray uses `rayrsi(1,(ref),0,0,^input)` followed by unqualified global `(x/y/z/l/m/n s4 g8)`, `s8 g8`, and `(x/y/z si g8)` reads. Stop angles are atan2 of the sign-corrected S8 LMN components, independently checked against geometric S4→S8 propagation for this reflected air segment. Raw LMN remains in the CSV and pickle.",
        "",
        f"Failed ray rows: {final['failed_rows']}; RC readback errors: {final['readback_errors']}. Maximum S8 XY radius: {final['max_s8_xy_radius_mm']} mm; LMN norm error: {final['max_lmn_norm_error']}; direction component difference: {final['max_raw_lmn_plus_segment_diff']}; stop-angle difference: {final['max_stop_angle_diff_deg']}°.",
        f"Sensor SI Z plane: {SENSOR_Z_MM} mm; maximum Z error: {final['max_sensor_z_error_mm']} mm. SI X range: {final['sensor_x_range_mm']} mm; SI Y range: {final['sensor_y_range_mm']} mm.",
        f"Prior 25-column baseline comparison: {'passed' if final['baseline_comparison']['pass'] else 'FAILED'} across {final['baseline_comparison']['compared_rows']} overlapping ray keys. Maximum coordinate difference {final['baseline_comparison']['max_coordinate_diff_mm']} mm, direction difference {final['baseline_comparison']['max_direction_diff']}, angle difference {final['baseline_comparison']['max_angle_diff_deg']}°.",
        f"Stop θx range: {final['theta_x_range_deg']}°. Stop θy range: {final['theta_y_range_deg']}°.",
        "",
        f"Canonical CSV: `{final['csv']}` (SHA-256 `{final['csv_sha256']}`). Pickle: `{final['pickle']}` (SHA-256 `{final['pickle_sha256']}`). The pickle is a flat NumPy structured array with one float64 field per CSV column; it was reloaded and compared field by field against the CSV-derived array.",
        "",
        "The `manifest.json` records the config and source hashes, chunk checksums, startup/backend evidence, and progress. Each chunk retains its issued macro, raw CODE V output, response, CSV, and summary.",
        "",
    ))


def main() -> int:
    global ANGLES, STATE_COUNT, ROW_COUNT, CHUNK_STATES, CHUNK_COUNT, OUTPUT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis-size", type=int, choices=(5, 251), default=251)
    parser.add_argument("--resume", type=Path, help="Existing run directory to resume")
    args = parser.parse_args()
    if args.axis_size == 5:
        ANGLES = (-12.5, -6.25, 0.0, 6.25, 12.5)
        STATE_COUNT = len(ANGLES) ** 2
        ROW_COUNT = STATE_COUNT * RAYS_PER_STATE
        CHUNK_STATES = 25
        CHUNK_COUNT = 1
        OUTPUT_ROOT = ROOT / "data" / "p1_retrace_detail_5x5"
    current_config = config()
    if args.resume:
        run_dir = args.resume.resolve()
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest["config"]["config_sha256"] != current_config["config_sha256"]:
            raise ValueError("Resume configuration, lens, or collector hash differs")
    else:
        run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        run_dir = OUTPUT_ROOT / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        manifest = {"run_id": run_id, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "config": current_config, "completed_chunks": {}, "attempts": [], "status": "created"}
        atomic_json(run_dir / "manifest.json", manifest)
    pending = [index for index in range(CHUNK_COUNT) if not verify_completed(run_dir, manifest, index)]
    print(f"Run {run_dir}; {len(pending)}/{CHUNK_COUNT} chunks pending", flush=True)
    if not pending:
        if "final" not in manifest:
            manifest["final"] = merge_and_pickle(run_dir, manifest)
            manifest["status"] = "complete" if manifest["final"]["validation_pass"] else "complete_with_failures"
            atomic_json(run_dir / "manifest.json", manifest)
        (run_dir / "report.md").write_text(report_text(manifest), encoding="utf-8")
        return 0 if manifest["final"]["validation_pass"] else 1
    baseline = backend_snapshot()
    manifest["codev_backend_baseline"] = list(baseline.values())
    manifest["status"] = "starting_codev"
    atomic_json(run_dir / "manifest.json", manifest)
    started = initialized = owned_backend = False
    cv = None
    initial_rc = None
    code = 2
    started_time = time.monotonic()
    try:
        import pythoncom
        from win32com.client import DispatchEx
        pythoncom.CoInitialize()
        initialized = True
        cv = DispatchEx("CodeV.Application")
        cv.CommandTimeout = 1800000
        cv.MaxTextBufferSize = 1000000
        cv.StartingDirectory = str(LENS.parent)
        cv.StartCodeV()
        started = True
        manifest["version_response"] = wait_ready(cv)
        newly_created = {pid: info for pid, info in backend_snapshot().items() if pid not in baseline}
        if not newly_created:
            raise RuntimeError("No distinct new codevm.exe backend was observed; refusing to use an existing session")
        owned_backend = True
        manifest["owned_codev_backends"] = list(newly_created.values())
        manifest["lens_load_response"] = str(cv.Command(f'run "{codev_path(LENS)}";go'))
        initial_rc = (float(cv.EvaluateExpression('(ade s"RC")')), float(cv.EvaluateExpression('(bde s"RC")')))
        manifest["initial_state"] = {"rc_ade": initial_rc[0], "rc_bde": initial_rc[1], "source_ade": float(cv.EvaluateExpression('(ade s"SRC_ROT")'))}
        manifest["source_set_response"] = str(cv.Command('ADE S"SRC_ROT" -20'))
        if abs(float(cv.EvaluateExpression('(ade s"SRC_ROT")')) + 20) > THRESHOLDS["readback_deg"]:
            raise RuntimeError("SRC_ROT ADE readback did not match -20")
        manifest["status"] = "collecting"
        atomic_json(run_dir / "manifest.json", manifest)
        for sequence, chunk_index in enumerate(pending):
            attempts = [entry for entry in manifest["attempts"] if entry["chunk_index"] == chunk_index]
            attempt_dir = run_dir / "chunks" / f"chunk_{chunk_index:04d}_attempt_{len(attempts) + 1}"
            attempt = {"chunk_index": chunk_index, "path": str(attempt_dir.relative_to(run_dir)), "started_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
            manifest["attempts"].append(attempt)
            atomic_json(run_dir / "manifest.json", manifest)
            chunk_start = time.monotonic()
            try:
                summary = process_chunk(cv, chunk_index, attempt_dir, current_config["config_sha256"])
            except Exception as exc:
                attempt["error"] = repr(exc)
                manifest["status"] = "interrupted"
                atomic_json(run_dir / "manifest.json", manifest)
                raise
            attempt["summary"] = str((attempt_dir / "summary.json").relative_to(run_dir))
            attempt["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            attempt["elapsed_seconds"] = time.monotonic() - chunk_start
            if not summary["quality_pass"]:
                attempt["error"] = "Chunk quality thresholds or state readbacks failed"
                manifest["status"] = "quality_failed"
                atomic_json(run_dir / "manifest.json", manifest)
                raise RuntimeError(f"Chunk {chunk_index} failed quality gate; raw and CSV retained")
            if sequence == 0 and not manifest.get("initial_chunk_gate_passed"):
                if summary["failed_rows"] or summary["readback_errors"]:
                    raise RuntimeError("First production chunk failed the continuation gate")
                manifest["initial_chunk_gate_passed"] = {"chunk_index": chunk_index, "backend_pids": list(newly_created), "time_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
            manifest["completed_chunks"][str(chunk_index)] = {"path": str(attempt_dir.relative_to(run_dir)), "rows": summary["rows"], "csv_sha256": summary["csv_sha256"]}
            atomic_json(run_dir / "manifest.json", manifest)
            completed_states = sum(len(chunk_states(int(index))) for index in manifest["completed_chunks"])
            elapsed = time.monotonic() - started_time
            print(f"Chunk {chunk_index + 1}/{CHUNK_COUNT}: {completed_states}/{STATE_COUNT} states, {completed_states * RAYS_PER_STATE}/{ROW_COUNT} rays; {elapsed:.1f}s elapsed", flush=True)
        manifest["final"] = merge_and_pickle(run_dir, manifest)
        manifest["status"] = "complete" if manifest["final"]["validation_pass"] else "complete_with_failures"
        manifest["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        atomic_json(run_dir / "manifest.json", manifest)
        (run_dir / "report.md").write_text(report_text(manifest), encoding="utf-8")
        code = 0 if manifest["final"]["validation_pass"] else 1
    except Exception as exc:
        manifest["error"] = repr(exc)
        if manifest["status"] not in ("quality_failed", "interrupted"):
            manifest["status"] = "interrupted"
        atomic_json(run_dir / "manifest.json", manifest)
        print(f"Run stopped: {exc!r}", flush=True)
    finally:
        if owned_backend and initial_rc is not None:
            try:
                cv.Command(f'ADE S"RC" {initial_rc[0]:.17g}')
                cv.Command(f'BDE S"RC" {initial_rc[1]:.17g}')
                manifest["rc_restored"] = [float(cv.EvaluateExpression('(ade s"RC")')), float(cv.EvaluateExpression('(bde s"RC")'))]
            except Exception as exc:
                manifest["restore_error"] = repr(exc)
                code = 2
        if started and owned_backend:
            try:
                cv.StopCodeV()
            except Exception as exc:
                manifest["stop_error"] = repr(exc)
                code = 2
        if initialized:
            try:
                pythoncom.CoUninitialize()
            except Exception as exc:
                manifest["com_cleanup_error"] = repr(exc)
                code = 2
        atomic_json(run_dir / "manifest.json", manifest)
        print(json.dumps({"run_dir": str(run_dir), "status": manifest["status"], "completed_chunks": len(manifest["completed_chunks"]), "error": manifest.get("error")}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
