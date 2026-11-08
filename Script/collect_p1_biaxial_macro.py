"""Collect the P1 101x101 biaxial grid in one owned CODE V session.

There is no accommodation loop or accommodation command. A retained macro
pilot is compared with the independent slow P1 helper before the full sweep.
Raw CSV/SEQ chunks and validated checkpoints make interrupted work resumable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LENS = ROOT / "Lens/p1_ME.seq"
DEFAULT_OUT = ROOT / "data/distortion_grid_p1_biaxial_macro"
S_RC = 's"RC"'
S_SOURCE = 's"SRC_ROT"'
SOURCE_ADE = -20.
ADE = np.linspace(-20., 20., 101)
BDE = np.linspace(-20., 20., 101)
NEAR = (.05, .05)
FIELDS = [(float(x), float(y)) for y in (1., 0., -1.) for x in (-1., 0., 1.)]
GRID_FIELDS = ("eye_rotation_deg", "eye_rotation_x_deg", "z_mm", "field_x_relative", "field_y_relative",
               "paraxial_x_mm", "paraxial_y_mm", "center_absolute_x_mm", "center_absolute_y_mm",
               "real_x_mm", "real_y_mm", "radial_distortion_pct", "tangential_distortion_pct")
DTYPE = np.dtype([(name, np.float64) for name in GRID_FIELDS])
CSV_COLUMNS = (["ia", "ib", "ade", "bde", "ade_rb", "bde_rb", "source_ade_rb", "xob_edge", "yob_edge",
                "st_c", "cx", "cy", "st_n", "nx", "ny"] +
               [name for k in range(1, 10) for name in (f"st_{k}", f"gx_{k}", f"gy_{k}")])
COL = {name: index for index, name in enumerate(CSV_COLUMNS)}
RAYS = [("st_c", "cx", "cy"), ("st_n", "nx", "ny")] + [(f"st_{k}", f"gx_{k}", f"gy_{k}") for k in range(1, 10)]
STATUS_NAMES = [names[0] for names in RAYS]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def check_command_result(result, context):
    text = str(result)
    errors = [line for line in text.splitlines() if line.lstrip().upper().startswith(("ERROR", "FATAL", "*** ERROR"))]
    if errors:
        raise RuntimeError(f"{context}: " + " | ".join(errors))
    return text


def evaluate(cv, command, *args):
    return float(cv.EvaluateExpression(f'({command} {" ".join(args)})'))


def linspace_literals(values):
    values = np.asarray(values, float)
    step = float(values[1] - values[0]) if len(values) > 1 else 0.
    if not np.allclose(values, values[0] + np.arange(len(values))*step, atol=1e-10, rtol=0):
        raise ValueError("Macro axes must be regular")
    return f"{values[0]:.17g}", f"{step:.17g}"


def write_macro(seq_path, csv_path, ade, bde, fov_x, fov_y):
    """P1-specific macro: RC sweep, FOV conversion, rays, SOURCE readback."""
    for path in (seq_path, csv_path):
        if '"' in str(path):
            raise ValueError("Quotes are not supported in CODE V paths")
    a0, da = linspace_literals(ade)
    b0, db = linspace_literals(bde)
    lines = [
        "LCL num ^ia ^ib ^ade ^bde ^ade_rb ^bde_rb ^source_ade_rb ^input(4) ^xob_edge ^yob_edge",
        "LCL num ^st_c ^cx ^cy ^st_n ^nx ^ny",
        "LCL num " + " ".join(f"^{name}" for name in CSV_COLUMNS[15:]),
        f'OPE NEW U^1 "{csv_path}"', "^input(1) == 0", "^input(2) == 0", "OUT N",
        f"FOR ^ia 0 {len(ade)-1} 1", f"^ade == {a0} + ^ia*{da}",
        f"FOR ^ib 0 {len(bde)-1} 1", f"^bde == {b0} + ^ib*{db}",
        'ADE S"RC" ^ade', 'BDE S"RC" ^bde', "SET VIG",
        '^ade_rb == (ade s"RC")', '^bde_rb == (bde s"RC")',
        '^source_ade_rb == (ade s"SRC_ROT")',
        f"XAN 0 {fov_x:.17g}", f"YAN 0 {fov_y:.17g}",
        "^xob_edge == (xob f2 z1)", "^yob_edge == (yob f2 z1)",
        "yob 0 ^yob_edge", "xob 0 ^xob_edge",
        "^input(3) == 0", "^input(4) == 0", "^st_c == rayrsi(1,1,1,0,^input)",
        "^cx == (x si)", "^cy == (y si)",
        f"^input(3) == {NEAR[0]:.17g}", f"^input(4) == {NEAR[1]:.17g}",
        "^st_n == rayrsi(1,1,0,0,^input)", "^nx == (x si)", "^ny == (y si)"]
    for k, (x, y) in enumerate(FIELDS, 1):
        lines += [f"^input(3) == {x:.17g}", f"^input(4) == {y:.17g}",
                  f"^st_{k} == rayrsi(1,1,0,0,^input)", f"^gx_{k} == (x si)", f"^gy_{k} == (y si)"]
    args = " ".join(f"^{name}" for name in CSV_COLUMNS)
    formats = " ".join(["'3e.17e'"] * len(CSV_COLUMNS))
    lines += [f'WRI U^1 Q"{formats}" {args}', "END FOR", "END FOR", "OUT Y", "CLO U^1", "RTN"]
    # These source-command restrictions also guard future accidental edits.
    assert not any(line.upper().startswith(("THI ", "CUY ", "RDY ")) for line in lines)
    seq_path.parent.mkdir(parents=True, exist_ok=True)
    seq_path.write_text("\n".join(lines) + "\n", encoding="ascii")


def parse_csv(path, ade, bde):
    lines = [line for line in Path(path).read_text(encoding="ascii").splitlines() if line.strip()]
    if len(lines) != len(ade)*len(bde):
        raise RuntimeError(f"{path}: {len(lines)} records, expected {len(ade)*len(bde)}")
    table = np.empty((len(lines), len(CSV_COLUMNS)))
    for row, line in enumerate(lines):
        parts = line.split()
        if len(parts) != len(CSV_COLUMNS):
            raise RuntimeError(f"{path}: row{row} has{len(parts)} columns, expected{len(CSV_COLUMNS)}")
        for column, value in enumerate(parts):
            try:
                table[row, column] = float(value)
            except ValueError:
                table[row, column] = np.nan
    required = np.zeros(table.shape, bool)
    required[:, :9] = True
    for status, x, y in RAYS:
        required[:, COL[status]] = True
        ok = table[:, COL[status]] == 0
        required[:, COL[x]], required[:, COL[y]] = ok, ok
    if np.any(required & ~np.isfinite(table)):
        row, column = np.argwhere(required & ~np.isfinite(table))[0]
        raise RuntimeError(f"Nonfinite required CSV value at row{row}, {CSV_COLUMNS[column]}")
    ai, bi = np.repeat(np.arange(len(ade)), len(bde)), np.tile(np.arange(len(bde)), len(ade))
    if not (np.array_equal(table[:, COL["ia"]], ai) and np.array_equal(table[:, COL["ib"]], bi)):
        raise RuntimeError("CSV does not have exact ADE-major/BDE-minor index order")
    expected_a, expected_b = np.asarray(ade)[ai], np.asarray(bde)[bi]
    for key, expected in (("ade", expected_a), ("bde", expected_b),
                          ("ade_rb", expected_a), ("bde_rb", expected_b),
                          ("source_ade_rb", np.full(len(table), SOURCE_ADE))):
        if np.max(np.abs(table[:, COL[key]] - expected)) > 1e-8:
            raise RuntimeError(f"{key} readback/request mismatch")
    status = table[:, [COL[name] for name in STATUS_NAMES]]
    if not np.all((status == np.round(status)) & (status >= -32768) & (status <= 32767)):
        raise RuntimeError("Ray status is not an int16 integer")
    return table, status.astype(np.int16)


def derive_rows(table, ade=None, bde=None):
    clean = np.asarray(table).copy()
    for status, x, y in RAYS:
        bad = clean[:, COL[status]] != 0
        clean[bad, COL[x]], clean[bad, COL[y]] = np.nan, np.nan
    data = np.empty(len(clean)*9, dtype=DTYPE)
    for row, raw in enumerate(clean):
        center = np.array([raw[COL["cx"]], raw[COL["cy"]]])
        near = np.array([raw[COL["nx"]], raw[COL["ny"]]]) - center
        for k, (x, y) in enumerate(FIELDS, 1):
            paraxial = near * np.array([x, y]) / np.array(NEAR)
            real = np.array([raw[COL[f"gx_{k}"]], raw[COL[f"gy_{k}"]]]) - center
            residual = real - paraxial
            r2 = float(np.sum(paraxial**2))
            radial = tangential = 0.
            if r2 >= 1e-20 or not np.isfinite(r2):
                radial = float(np.dot(residual, paraxial) / r2 * 100.)
                tangential = float((residual[0]*paraxial[1] - residual[1]*paraxial[0]) / r2 * 100.)
            angle_a = raw[COL["ade"]] if ade is None else ade[int(raw[COL["ia"]])]
            angle_b = raw[COL["bde"]] if bde is None else bde[int(raw[COL["ib"]])]
            data[row*9 + k-1] = (angle_a, angle_b, 0., x, y,
                *paraxial, *center, *real, radial, tangential)
    return data


def run_macro(cv, seq_path, csv_path, ade, bde, fov):
    # Retain completed raw outputs; never remove a completed chunk.
    if csv_path.exists():
        raise FileExistsError(csv_path)
    write_macro(seq_path, csv_path, ade, bde, *fov)
    started = time.perf_counter()
    response = check_command_result(cv.Command(f'run "{seq_path}";go'), "P1 macro")
    seq_path.with_suffix(".response.txt").write_text(response, encoding="utf-8")
    table, status = parse_csv(csv_path, ade, bde)
    return table, status, time.perf_counter()-started


def pilot(cv, out, fov, legacy):
    axes = np.array([-20., 0., 20.])
    cases = [("pilot_3x3", axes, axes),
             ("pilot_asymmetric", np.array([-6.8, 4.4]), np.array([-11.2, 7.6]))]
    report = {"comparisons": [], "tolerance_mm": 1e-8, "initial_grid_states": 9}
    for name, av, bv in cases:
        table, status, macro_seconds = run_macro(cv, out/f"{name}.seq", out/f"{name}.csv", av, bv, fov)
        if np.any(status):
            raise RuntimeError(f"{name} has failed rays; inspect retained CSV before choosing a validation pose")
        fast = derive_rows(table, av, bv)
        started = time.perf_counter()
        slow_rows = []
        for ade in av:
            for bde in bv:
                slow_rows.extend(legacy.calculate_grid_at_biaxial_rotation(cv, float(ade), float(bde), 3, *fov, z_mm=None))
                if abs(evaluate(cv, "ade", S_SOURCE)-SOURCE_ADE) > 1e-8:
                    raise RuntimeError("SOURCE setting changed during slow validation")
        slow_seconds = time.perf_counter()-started
        errors = {}
        for field in ("eye_rotation_deg", "eye_rotation_x_deg", "field_x_relative", "field_y_relative",
                      "center_absolute_x_mm", "center_absolute_y_mm", "real_x_mm", "real_y_mm",
                      "paraxial_x_mm", "paraxial_y_mm"):
            slow = np.array([row[field] for row in slow_rows])
            delta = np.max(np.abs(slow-fast[field]))
            errors[field] = float(delta)
            if delta > (1e-8 if field.endswith("mm") else 1e-8):
                raise RuntimeError(f"Macro/slow mismatch in {field}: {delta}")
        count = len(av)*len(bv)
        row = {"case": name, "states": count, "macro_trace_seconds": macro_seconds,
               "slow_trace_seconds": slow_seconds, "max_abs_differences": errors,
               "projected_full_macro_trace_seconds": macro_seconds*10201/count,
               "projected_full_slow_trace_seconds": slow_seconds*10201/count}
        report["comparisons"].append(row)
        log(f"{name} PASS: macro{macro_seconds:.3f}s / slow{slow_seconds:.3f}s; full serial macro projection{row['projected_full_macro_trace_seconds']:.1f}s (startup excluded; block-overhead uncertainty)")
    write_json(out/"pilot_validation.json", report)
    return report


def wait_ready(cv, timeout=180.):
    started = time.perf_counter()
    last = ""
    while time.perf_counter()-started < timeout:
        try:
            text = check_command_result(cv.Command("VER"), "CODE V readiness")
            if text.strip():
                return text
        except Exception as error:
            last = str(error)
        time.sleep(.5)
    raise RuntimeError(f"CODE V startup did not finish within{timeout}s: {last}")


def collect(args):
    import pythoncom
    import win32com.client
    import distortion_grid as legacy
    out = args.output_dir
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError(f"Use a new output directory or --resume: {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out/"chunks").mkdir(exist_ok=True)
    immutable = {"lens_sequence": str(LENS), "lens_sha256": digest(LENS),
        "collector_sha256": digest(__file__), "legacy_helper_sha256": digest(Path(legacy.__file__)),
        "ADE_deg": ADE.tolist(), "BDE_deg": BDE.tolist(), "source_ADE_deg": SOURCE_ADE,
        "fields_relative_xy": FIELDS, "near_axis_relative_xy": NEAR,
        "workers": 1, "accommodation_commands": False, "accommodation_loop": False,
        "Z_mm_convention": 0., "z_or_thickness_command_sent": False,
        "field_convention": "angular XAN/YAN restored before per-state XOB/YOB conversion; normalized rayrsi inputs recorded",
        "detector_frame_status": "provisional shared rear-camera frame; registration requires separate audit",
        "schema": list(GRID_FIELDS), "CSV_columns": CSV_COLUMNS, "bde_block": args.bde_block}
    config_hash = hashlib.sha256(json.dumps(immutable, sort_keys=True).encode()).hexdigest()
    manifest_path = out/"manifest.json"
    if args.resume:
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved["configuration_sha256"] != config_hash:
            raise RuntimeError("Resume configuration/source hashes differ")
        manifest = saved
        resume_fov = saved.get("angular_FOV_semi_deg_xy")
    else:
        resume_fov = None
        manifest = {"configuration": immutable, "configuration_sha256": config_hash,
                    "complete": False, "chunks": [], "started_local": time.strftime("%Y-%m-%dT%H:%M:%S")}
        (out/"collector_source.py").write_bytes(Path(__file__).read_bytes())
        (out/"p1_source.seq").write_bytes(LENS.read_bytes())
    cv, started_cv = None, False
    pythoncom.CoInitialize()
    try:
        log("Starting one owned CODE V session; waiting for readiness")
        cv = win32com.client.Dispatch("CodeV.Application")
        cv.CommandTimeout = 600000
        cv.MaxTextBufferSize = 1000000
        cv.StartingDirectory = str(LENS.parent)
        startup = time.perf_counter()
        cv.StartCodeV()
        started_cv = True
        manifest["CODEV_version_response"] = wait_ready(cv)
        manifest["startup_seconds"] = time.perf_counter()-startup
        response = check_command_result(cv.Command(f'run "{LENS}";go'), "Load P1 source")
        (out/"load_response.txt").write_text(response, encoding="utf-8")
        loaded_source = evaluate(cv, "ade", S_SOURCE)
        manifest["loaded_SOURCE_ADE_deg"] = loaded_source
        if abs(loaded_source-SOURCE_ADE) > 1e-8:
            log(f"Loaded SOURCE ADE is{loaded_source:.12g}deg; applying required{SOURCE_ADE:g}deg in owned session")
            check_command_result(cv.Command(f"ADE {S_SOURCE} {SOURCE_ADE:.17g}"), "Set required SOURCE")
        applied_source = evaluate(cv, "ade", S_SOURCE)
        if abs(applied_source-SOURCE_ADE) > 1e-8:
            raise RuntimeError("SOURCE ADE verification failed")
        manifest["applied_SOURCE_ADE_deg"] = applied_source
        fov = tuple(float(value) for value in legacy.angular_fov(cv))
        if resume_fov is not None and not np.allclose(resume_fov, fov, atol=1e-12, rtol=0):
            raise RuntimeError("Loaded-lens FOV differs from retained resume configuration")
        manifest["angular_FOV_semi_deg_xy"] = list(fov)
        geometry = {}
        for label, expression in (("image_thickness", "(thi si)"), ("image_yde", "(yde si)"),
                                   ("image_xde", "(xde si)"), ("image_ADE", "(ade si)"),
                                   ("image_BDE", "(bde si)")):
            try:
                value = float(cv.EvaluateExpression(expression))
                geometry[label] = value if math.isfinite(value) else None
            except Exception:
                geometry[label] = None
        manifest["optional_image_geometry_CODEV_native_units"] = geometry
        write_json(manifest_path, manifest)
        # A resume reuses retained pilot outputs after exact hash validation.
        if (out/"pilot_validation.json").exists():
            validation = json.loads((out/"pilot_validation.json").read_text(encoding="utf-8"))
            if len(validation.get("comparisons", [])) != 2:
                raise RuntimeError("Incomplete retained pilot validation")
        else:
            validation = pilot(cv, out, fov, legacy)
        manifest["pilot_validation"] = validation
        write_json(manifest_path, manifest)
        if args.pilot_only:
            return
        sweep_start = time.perf_counter()
        chunks, statuses, geometries = [], [], []
        for lower in range(0, len(BDE), args.bde_block):
            upper = min(lower+args.bde_block, len(BDE))
            tag = f"bde_{lower:03d}_{upper-1:03d}"
            checkpoint = out/"chunks"/f"{tag}.npz"
            csv_path, seq_path = out/"chunks"/f"{tag}.csv", out/"chunks"/f"{tag}.seq"
            bde = BDE[lower:upper]
            if checkpoint.exists():
                with np.load(checkpoint, allow_pickle=False) as cache:
                    if str(cache["configuration_sha256"]) != config_hash:
                        raise RuntimeError(f"Checkpoint configuration mismatch: {checkpoint}")
                    if not np.array_equal(cache["ADE"], ADE) or not np.array_equal(cache["BDE"], bde):
                        raise RuntimeError(f"Checkpoint axes mismatch: {checkpoint}")
                    if str(cache["csv_sha256"]) != digest(csv_path) or str(cache["seq_sha256"]) != digest(seq_path):
                        raise RuntimeError(f"Checkpoint retained raw-source hash mismatch: {checkpoint}")
                table, status = parse_csv(csv_path, ADE, bde)
                rows, seconds = derive_rows(table, ADE, bde), 0.
                log(f"validated resumed{tag}")
            else:
                if csv_path.exists():
                    # Retain an uncheckpointed raw attempt and rerun this one
                    # chunk. Completed, hashed checkpoints are never replaced.
                    suffix = f"uncheckpointed_{time.time_ns()}"
                    for raw_path in (csv_path, seq_path, seq_path.with_suffix(".response.txt")):
                        if raw_path.exists():
                            raw_path.rename(raw_path.with_name(f"{raw_path.stem}_{suffix}{raw_path.suffix}"))
                    log(f"retained uncheckpointed{tag} raw attempt; retrying chunk")
                table, status, seconds = run_macro(cv, seq_path, csv_path, ADE, bde, fov)
                rows = derive_rows(table, ADE, bde)
                np.savez_compressed(checkpoint, ADE=ADE, BDE=bde,
                    configuration_sha256=np.array(config_hash), csv_sha256=np.array(digest(csv_path)),
                    seq_sha256=np.array(digest(seq_path)), rows=rows, status=status)
            chunks.append(rows.reshape(len(ADE), len(bde), 9))
            statuses.append(status.reshape(len(ADE), len(bde), 11))
            geometries.append(table[:, [COL["xob_edge"], COL["yob_edge"]]].reshape(len(ADE), len(bde), 2))
            manifest["chunks"] = [{"tag": f"bde_{lo:03d}_{min(lo+args.bde_block,len(BDE))-1:03d}"}
                                  for lo in range(0, upper, args.bde_block)]
            manifest["completed_BDE_values"] = upper
            write_json(manifest_path, manifest)
            log(f"P1{upper}/{len(BDE)}BDE columns complete ({upper*len(ADE):,}/10,201states); latest macro{seconds:.2f}s; sweep elapsed{time.perf_counter()-sweep_start:.1f}s")
        grid = np.concatenate(chunks, axis=1)
        failure_map = np.concatenate(statuses, axis=1)
        object_height_edges = np.concatenate(geometries, axis=1)
        assert grid.shape == (101, 101, 9) and grid.size == 91809
        assert np.allclose(grid["eye_rotation_deg"], ADE[:, None, None], atol=1e-8, rtol=0)
        assert np.allclose(grid["eye_rotation_x_deg"], BDE[None, :, None], atol=1e-8, rtol=0)
        final = out/"distortion_grid.pkl"
        temporary = out/"distortion_grid.pkl.tmp"
        with temporary.open("wb") as stream:
            pickle.dump(grid.reshape(-1), stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(final)
        np.savez_compressed(out/"failure_map.npz", ADE=ADE, BDE=BDE, status=failure_map,
                            status_names=np.array(STATUS_NAMES), object_height_edges_mm=object_height_edges,
                            field_relative_xy=np.array(FIELDS))
        manifest.update(complete=True, records=int(grid.size), states=10201,
            sweep_seconds=time.perf_counter()-sweep_start, pickle_sha256=digest(final),
            failed_ray_counts=np.sum(failure_map != 0, axis=(0, 1)).tolist(),
            nonfinite_real_records=int(np.sum(~np.isfinite(grid["real_x_mm"]) | ~np.isfinite(grid["real_y_mm"]))),
            absolute_coordinate_reconstruction="absolute_ray_mm = center_absolute_mm + real_mm; real_mm is center-relative",
            no_accommodation_commands_sent=True)
        write_json(manifest_path, manifest)
        log(f"P1 complete:{grid.size:,}records; pickle{final}")
    except BaseException:
        manifest["last_failure_traceback"] = traceback.format_exc()
        write_json(manifest_path, manifest)
        raise
    finally:
        if cv is not None and started_cv:
            try:
                cv.StopCodeV()
                log("Owned CODE V session stopped")
            except Exception as error:
                log(f"StopCodeV reported:{error}")
        pythoncom.CoUninitialize()


def self_check():
    """No CODE V: macro restrictions, indexing, field arithmetic and failures."""
    import tempfile
    with tempfile.TemporaryDirectory(prefix="p1_macro_check_") as folder:
        root = Path(folder)
        av, bv = np.array([-20., 0., 20.]), np.array([-11.2, 7.6])
        seq, csv = root/"check.seq", root/"check.csv"
        write_macro(seq, csv, av, bv, 1., 1.)
        text = seq.read_text(encoding="ascii").upper()
        assert "CORNEA_ENT_D" not in text and "THI " not in text and "ACCOMMODATION" not in text
        records = []
        for ia, a in enumerate(av):
            for ib, b in enumerate(bv):
                row = [ia, ib, a, b, a, b, SOURCE_ADE, 1., 1., 0., .3, -.2, 0., .35, -.15]
                for x, y in FIELDS:
                    row += [0., .3+x, -.2+y]
                records.append(row)
        np.savetxt(csv, records, fmt="%.17g")
        table, status = parse_csv(csv, av, bv)
        data = derive_rows(table, av, bv)
        assert len(data) == 54 and not np.any(status)
        assert np.max(np.abs(data["real_x_mm"]-np.tile(np.array(FIELDS)[:, 0], 6))) < 1e-14
        assert np.max(np.abs(data["real_y_mm"]-np.tile(np.array(FIELDS)[:, 1], 6))) < 1e-14
        table[0, COL["st_n"]] = 1
        failed = derive_rows(table, av, bv)
        assert np.all(np.isfinite(failed["real_x_mm"][:9]))
        assert np.all(~np.isfinite(failed["paraxial_x_mm"][:9]))
    log("offline macro/schema/index/failure checks PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bde-block", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.bde_block <= 101:
        parser.error("--bde-block must be in1..101")
    self_check()
    if not args.check_only:
        collect(args)


if __name__ == "__main__":
    main()
