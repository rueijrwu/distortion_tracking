"""Durable A=0, 401x401 P4 ADE/BDE macro sweep using persistent CODE V workers.

This is a focused collector for one accommodation. It reuses the production
macro writer/parser/validation and ray-failure handling, but splits BDE into
small blocks so four long-lived CODE V sessions can checkpoint independently.
Raw state CSVs and generated .seq macros are retained beside the final pickle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import queue
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Script"))
import distortion_grid_p4_biaxial_macro as macro  # noqa: E402
from distortion_grid import angular_fov  # noqa: E402
from distortion_grid_p4 import (  # noqa: E402
    LENS_FILE, apply_accommodation, load_lens, read_accommodation_baseline,
)

OUT_DEFAULT = ROOT / "data" / "distortion_grid_p4_biaxial_macro_a0_401x401"
OLD_GRID = ROOT / "data" / "distortion_grid_p4_biaxial_macro" / "distortion_grid.pkl"
ACCOMMODATION_D = 0.0
GRID_LINES = 3
DEFAULT_BLOCK_BDE = 5
COORD_TOL_UM = 0.01
PCT_TOL = 1e-4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def digest_config(config: dict) -> str:
    raw = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def chunk_plan(bde_values: np.ndarray, workers: int, block_size: int) -> list[dict]:
    """Partition contiguous BDE indices among workers, then into durable blocks."""
    assignments = np.array_split(np.arange(len(bde_values)), workers)
    plan = []
    chunk_id = 0
    for worker_id, indices in enumerate(assignments):
        for start in range(0, len(indices), block_size):
            part = indices[start:start + block_size]
            plan.append({
                "chunk_id": chunk_id,
                "worker_id": worker_id,
                "bde_indices": part.astype(int).tolist(),
                "bde_values": bde_values[part].astype(float).tolist(),
            })
            chunk_id += 1
    return plan


def validate_plan(plan: list[dict], bde_values: np.ndarray, workers: int, block_size: int) -> None:
    expected = chunk_plan(bde_values, workers, block_size)
    if plan != expected:
        raise ValueError("Chunk plan is not the expected contiguous, nonoverlapping BDE partition")


def _load_chunk(root: Path, item: dict, ade: np.ndarray, bde: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate a completed block from its checkpoint manifest entry."""
    cid = int(item["chunk_id"])
    files = item["files"]
    paths = {key: root / files[key] for key in ("csv", "seq", "pickle", "failures")}
    for key, path in paths.items():
        if root.resolve() not in path.resolve().parents:
            raise ValueError(f"Chunk {cid}: {key} path escapes output directory: {path}")
        if not path.is_file() or macro.file_sha256(path) != item[f"{key}_sha256"]:
            raise ValueError(f"Chunk {cid}: missing or changed {key} file {path}")
    with paths["pickle"].open("rb") as f:
        data = pickle.load(f)
    with np.load(paths["failures"], allow_pickle=False) as z:
        status = z["status"]
        if not np.array_equal(z["ade_deg"], ade) or not np.array_equal(z["bde_deg"], bde[item["bde_indices"]]):
            raise ValueError(f"Chunk {cid}: failure sidecar gaze labels changed")
        if not np.array_equal(z["status_columns"], np.array(macro.STATUS_COLUMNS)):
            raise ValueError(f"Chunk {cid}: failure sidecar status schema changed")
    bchunk = bde[item["bde_indices"]]
    macro.validate_group(data, ade, bchunk, ACCOMMODATION_D, GRID_LINES, status)
    if data.size != len(ade) * len(bchunk) * GRID_LINES**2:
        raise ValueError(f"Chunk {cid}: unexpected record count {data.size}")
    return data.reshape(len(ade), len(bchunk), GRID_LINES**2), status


def _worker_main(worker_id: int, tasks: list[dict], ade: list[float], bde: list[float],
                 out_root: str, result_queue) -> None:
    """One COM session per worker; load the lens once and execute all its blocks."""
    import pythoncom
    import win32com.client

    root = Path(out_root)
    cv = None
    started = False
    pythoncom.CoInitialize()
    try:
        cv = win32com.client.Dispatch("CodeV.Application")
        timeout = macro.configure_session(cv)
        t0 = time.perf_counter()
        cv.StartCodeV()
        started = True
        startup = time.perf_counter() - t0
        load_lens(cv)
        baseline = read_accommodation_baseline(cv)
        geometry = apply_accommodation(cv, baseline, ACCOMMODATION_D)
        fov_x, fov_y = angular_fov(cv)
        result_queue.put(("started", worker_id, {
            "pid": os.getpid(), "startup_seconds": startup, "command_timeout": timeout,
            "fov_x_deg": fov_x, "fov_y_deg": fov_y,
            "geometry_target": geometry,
        }))
        for task in tasks:
            cid = int(task["chunk_id"])
            bchunk = np.asarray(task["bde_values"], dtype=np.float64)
            tag = f"chunk_{cid:04d}_worker_{worker_id}"
            macro_dir = root / "macros" / f"worker_{worker_id}"
            csv_dir = root / "csv" / f"worker_{worker_id}"
            checkpoint_dir = root / "checkpoints"
            seq_path = macro_dir / f"{tag}.seq"
            csv_path = csv_dir / f"{tag}.csv"
            seq_path.parent.mkdir(parents=True, exist_ok=True)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            macro.write_macro(seq_path, csv_path, ade, bchunk, fov_x, fov_y)
            run_t0 = time.perf_counter()
            macro.run_macro(cv, seq_path, csv_path)
            macro_seconds = time.perf_counter() - run_t0
            table = macro.parse_csv(csv_path, len(ade), len(bchunk))
            status_flat = macro.check_table(table, ade, bchunk)
            status = status_flat.reshape(len(ade), len(bchunk), len(macro.STATUS_COLUMNS))
            data = macro.derive_rows(table, ade, bchunk, ACCOMMODATION_D, status_flat)
            macro.validate_group(data, ade, bchunk, ACCOMMODATION_D, GRID_LINES, status)
            pkl_path = checkpoint_dir / f"chunk_{cid:04d}.pkl"
            fail_path = checkpoint_dir / f"chunk_{cid:04d}.failures.npz"
            macro.atomic_pickle(data, pkl_path)
            macro.atomic_npz(
                fail_path, status=status, status_columns=np.array(macro.STATUS_COLUMNS),
                accommodation_d=np.float64(ACCOMMODATION_D), ade_deg=np.asarray(ade),
                bde_deg=bchunk,
            )
            result_queue.put(("done", worker_id, {
                **task,
                "record_count": int(data.size),
                "state_count": int(len(ade) * len(bchunk)),
                "macro_seconds": macro_seconds,
                "ms_per_state": 1000.0 * macro_seconds / (len(ade) * len(bchunk)),
                "failed_states": int(np.count_nonzero(np.any(status != 0, axis=-1))),
                "failed_rays": int(np.count_nonzero(status)),
                "files": {
                    "csv": str(csv_path.relative_to(root)), "seq": str(seq_path.relative_to(root)),
                    "pickle": str(pkl_path.relative_to(root)), "failures": str(fail_path.relative_to(root)),
                },
                "csv_sha256": macro.file_sha256(csv_path),
                "seq_sha256": macro.file_sha256(seq_path),
                "pickle_sha256": macro.file_sha256(pkl_path),
                "failures_sha256": macro.file_sha256(fail_path),
            }))
        result_queue.put(("worker_done", worker_id, None))
    except BaseException:
        result_queue.put(("error", worker_id, traceback.format_exc()))
    finally:
        try:
            if cv is not None and started:
                cv.StopCodeV()
        except BaseException:
            result_queue.put(("error", worker_id, "StopCodeV failed:\n" + traceback.format_exc()))
        finally:
            pythoncom.CoUninitialize()
            result_queue.put(("exit", worker_id, None))


def compare_overlap(data: np.ndarray, ade: np.ndarray, bde: np.ndarray) -> dict:
    """Compare every state/field shared with the published 101x101 A=0 grid."""
    with OLD_GRID.open("rb") as f:
        old = pickle.load(f)
    old = old[np.isclose(old["accommodation_d"], 0.0, atol=1e-9, rtol=0)]
    old_ade, old_bde = np.unique(old["eye_rotation_deg"]), np.unique(old["eye_rotation_x_deg"])
    old_cube = old.reshape(len(old_ade), len(old_bde), GRID_LINES**2)
    new_cube = data.reshape(len(ade), len(bde), GRID_LINES**2)
    ai = [(i, int(np.argmin(np.abs(old_ade - value)))) for i, value in enumerate(ade)
          if np.min(np.abs(old_ade - value)) <= 1e-9]
    bi = [(i, int(np.argmin(np.abs(old_bde - value)))) for i, value in enumerate(bde)
          if np.min(np.abs(old_bde - value)) <= 1e-9]
    if not ai or not bi:
        raise ValueError("No matching gaze states with the published A=0 101x101 grid")
    exact = {"eye_rotation_deg", "eye_rotation_x_deg", "accommodation_d", "z_mm",
             "field_x_relative", "field_y_relative"}
    coord = {"paraxial_x_mm", "paraxial_y_mm", "center_absolute_x_mm", "center_absolute_y_mm",
             "real_x_mm", "real_y_mm"}
    metrics = {"overlap_states": len(ai) * len(bi), "overlap_records": len(ai) * len(bi) * GRID_LINES**2,
               "max_coordinate_diff_um": 0.0, "max_pct_diff": 0.0}
    failures = []
    for new_a, old_a in ai:
        for new_b, old_b in bi:
            nrow, orow = new_cube[new_a, new_b], old_cube[old_a, old_b]
            for field in macro.GRID_FIELDS:
                nval, oval = nrow[field], orow[field]
                if not np.array_equal(np.isnan(nval), np.isnan(oval)):
                    failures.append(f"NaN mismatch {field} at ADE={ade[new_a]}, BDE={bde[new_b]}")
                    continue
                finite = np.isfinite(nval) & np.isfinite(oval)
                diff = np.abs(nval[finite] - oval[finite])
                maxdiff = float(diff.max()) if diff.size else 0.0
                if field in exact and maxdiff != 0.0:
                    failures.append(f"exact-column mismatch {field}: {maxdiff:g}")
                elif field in coord:
                    metrics["max_coordinate_diff_um"] = max(metrics["max_coordinate_diff_um"], maxdiff * 1000)
                    if maxdiff * 1000 >= COORD_TOL_UM:
                        failures.append(f"coordinate mismatch {field}: {maxdiff*1000:g} um")
                elif field in ("radial_distortion_pct", "tangential_distortion_pct"):
                    metrics["max_pct_diff"] = max(metrics["max_pct_diff"], maxdiff)
                    if maxdiff >= PCT_TOL:
                        failures.append(f"percentage mismatch {field}: {maxdiff:g} points")
    metrics["status"] = "PASS" if not failures else "FAIL"
    metrics["failures"] = failures[:20]
    if failures:
        raise ValueError(f"Overlapping-grid comparison failed: {failures[:5]}")
    return metrics


def _read_chunk_sidecar(root: Path, item: dict, ade: np.ndarray, bde: np.ndarray) -> np.ndarray:
    path = root / item["files"]["failures"]
    with np.load(path, allow_pickle=False) as z:
        expected_bde = bde[item["bde_indices"]]
        if not np.array_equal(z["ade_deg"], ade) or not np.array_equal(z["bde_deg"], expected_bde):
            raise ValueError(f"Chunk {item['chunk_id']} failure labels do not match global axes")
        return z["status"]


def merge_and_publish(root: Path, ade: np.ndarray, bde: np.ndarray,
                      chunks: list[dict], manifest: dict) -> dict:
    data_cube = np.empty((len(ade), len(bde), GRID_LINES**2), dtype=[(n, np.float64) for n in macro.GRID_FIELDS])
    status_cube = np.zeros((len(ade), len(bde), len(macro.STATUS_COLUMNS)), dtype=macro.STATUS_DTYPE)
    filled = np.zeros(len(bde), dtype=bool)
    by_id = {int(item["chunk_id"]): item for item in chunks}
    for chunk in manifest["chunk_plan"]:
        cid = int(chunk["chunk_id"])
        if cid not in by_id:
            raise ValueError(f"Cannot merge: chunk {cid} is not checkpointed")
        item = by_id[cid]
        vals = chunk["bde_indices"]
        if any(filled[vals]):
            raise ValueError(f"BDE overlap while merging chunk {cid}")
        data_part, status_part = _load_chunk(root, item, ade, bde)
        if data_part.shape != (len(ade), len(vals), GRID_LINES**2):
            raise ValueError(f"Chunk {cid} grid shape {data_part.shape} is inconsistent")
        # Explicit indexed merge is essential: CSV chunks are ADE-major with a local BDE axis.
        data_cube[:, vals, :] = data_part
        status_cube[:, vals, :] = status_part
        filled[vals] = True
    if not filled.all():
        raise ValueError(f"BDE axis has unfilled indices {np.flatnonzero(~filled).tolist()}")
    data = data_cube.reshape(-1)
    status = status_cube[None, ...]
    macro.validate_complete_sweep(
        data, status, ade, bde, [ACCOMMODATION_D], len(ade) * len(bde) * GRID_LINES**2,
    )
    overlap = compare_overlap(data, ade, bde)
    pkl_path = root / "distortion_grid.pkl"
    failure_path = root / "failure_map.npz"
    macro.atomic_pickle(data, pkl_path)
    macro.atomic_npz(
        failure_path, status=status,
        status_columns=np.array(macro.STATUS_COLUMNS),
        accommodation_d=np.array([ACCOMMODATION_D], dtype=np.float64),
        ade_deg=ade.astype(np.float64), bde_deg=bde.astype(np.float64),
    )
    manifest["overlap_validation"] = overlap
    manifest["published"] = {
        "pickle": pkl_path.name, "pickle_sha256": macro.file_sha256(pkl_path),
        "record_count": int(data.size), "failure_map": failure_path.name,
        "failure_map_sha256": macro.file_sha256(failure_path),
        "failed_states": int(np.count_nonzero(np.any(status != 0, axis=-1))),
        "failed_rays": int(np.count_nonzero(status)), "completed_at_utc": utc_now(),
    }
    manifest["state"] = "complete"
    manifest["updated_at_utc"] = utc_now()
    macro.atomic_json(manifest, root / "manifest.json")
    macro.atomic_json({
        "state": "complete", "completed_at_utc": manifest["updated_at_utc"],
        "completed_chunks": len(chunks), "total_chunks": len(manifest["chunk_plan"]),
        "states_completed": len(ade) * len(bde), "states_total": len(ade) * len(bde),
        "overlap_validation": overlap, "published": manifest["published"],
    }, root / "status.json")
    return manifest["published"]


def validate_existing_chunks(root: Path, manifest: dict, ade: np.ndarray, bde: np.ndarray) -> list[dict]:
    plan = manifest["chunk_plan"]
    expected_by_id = {int(x["chunk_id"]): x for x in plan}
    items = manifest.get("completed_chunks", [])
    ids = [int(item.get("chunk_id", -1)) for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("Manifest repeats a completed chunk ID")
    if any(cid not in expected_by_id for cid in ids):
        raise ValueError("Manifest lists a completed chunk absent from the configured plan")
    valid = []
    for item in items:
        expected = expected_by_id[int(item["chunk_id"])]
        for key in ("worker_id", "bde_indices", "bde_values"):
            if item.get(key) != expected[key]:
                raise ValueError(f"Chunk {item['chunk_id']}: manifest {key} does not match configured plan")
        _load_chunk(root, item, ade, bde)
        valid.append(item)
    return valid


def run(args) -> None:
    root = args.output_dir.resolve()
    if ROOT not in root.parents:
        raise ValueError(f"Output directory must be inside {ROOT}")
    ade = np.linspace(args.ade_min, args.ade_max, args.ade_count, dtype=np.float64)
    bde = np.linspace(args.bde_min, args.bde_max, args.bde_count, dtype=np.float64)
    plan = chunk_plan(bde, args.workers, args.block_bde)
    config = {
        "collector": "distortion_grid_p4_a0_biaxial_macro", "backend": macro.BACKEND,
        "accommodation_d": ACCOMMODATION_D, "z_mm": macro.Z_MM,
        "ade_deg": ade.tolist(), "bde_deg": bde.tolist(),
        "grid_lines": GRID_LINES, "csv_columns": list(macro.CSV_COLUMNS),
        "grid_fields": list(macro.GRID_FIELDS), "status_columns": list(macro.STATUS_COLUMNS),
        "block_bde_rows": args.block_bde, "lens_file": str(LENS_FILE.relative_to(ROOT)),
        "worker_count": args.workers,
        "lens_sha256": macro.file_sha256(LENS_FILE),
        "source_sequence_sha256": macro.file_sha256(ROOT / "Lens" / "p4_ME.seq"),
        "coarse_source_pickle_sha256": macro.file_sha256(OLD_GRID),
        "runner_sha256": macro.file_sha256(Path(__file__).resolve()),
        "macro_source_sha256": macro.file_sha256(Path(macro.__file__).resolve()),
        "rotation_policy": "load p4_ME.len unchanged; each gaze ADE/BDE assignment explicitly selects RC",
    }
    config_hash = digest_config(config)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if args.resume:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"--resume requires {manifest_path}")
        status_path = root / "status.json"
        if status_path.is_file():
            prior_status = json.loads(status_path.read_text(encoding="utf-8"))
            if prior_status.get("state") == "running":
                active_pids = []
                candidates = [prior_status.get("owner_pid")]
                candidates.extend(v.get("pid") for v in prior_status.get("workers_detail", {}).values())
                for pid in candidates:
                    if not pid or int(pid) == os.getpid():
                        continue
                    try:
                        os.kill(int(pid), 0)
                    except OSError:
                        continue
                    active_pids.append(int(pid))
                if active_pids:
                    raise RuntimeError(f"Run still has active owner/worker PIDs {active_pids}; do not resume concurrently")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("configuration_sha256") != config_hash or manifest.get("configuration") != config:
            raise ValueError("Existing manifest does not match lens or sweep configuration")
        validate_plan(manifest.get("chunk_plan", []), bde, args.workers, args.block_bde)
        completed = validate_existing_chunks(root, manifest, ade, bde)
        completed_ids = {int(x["chunk_id"]) for x in completed}
        manifest["completed_chunks"] = completed
    else:
        if len(list(root.iterdir())):
            raise FileExistsError(f"Output directory is not empty: {root}; use --resume or a new path")
        completed_ids = set()
        manifest = {
            "state": "running", "created_at_utc": utc_now(), "updated_at_utc": utc_now(),
            "configuration": config, "configuration_sha256": config_hash,
            "chunk_plan": plan, "completed_chunks": [],
        }
        macro.atomic_json(manifest, manifest_path)
    todo_by_worker = {i: [] for i in range(args.workers)}
    for chunk in plan:
        if int(chunk["chunk_id"]) not in completed_ids:
            todo_by_worker[int(chunk["worker_id"])].append(chunk)
    todo_by_worker = {k: v for k, v in todo_by_worker.items() if v}
    if not todo_by_worker:
        merge_and_publish(root, ade, bde, manifest["completed_chunks"], manifest)
        return
    start_utc = time.time()
    status = {
        "state": "running", "owner_pid": os.getpid(), "started_at_utc": utc_now(),
        "completed_chunks": len(completed_ids), "total_chunks": len(plan),
        "states_completed": sum(int(x["state_count"]) for x in manifest["completed_chunks"]),
        "states_total": len(ade) * len(bde), "workers": args.workers,
        "chunk_bde_rows": args.block_bde,
    }
    macro.atomic_json(status, root / "status.json")
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    workers = [ctx.Process(target=_worker_main,
                           args=(wid, tasks, ade.tolist(), bde.tolist(), str(root), result_queue),
                           name=f"p4_a0_codev_{wid}")
               for wid, tasks in todo_by_worker.items()]
    for proc in workers:
        proc.start()
    active = {int(proc.name.rsplit("_", 1)[-1]): proc for proc in workers}
    exited = set()
    errors = []
    try:
        while len(exited) < len(workers):
            try:
                kind, wid, payload = result_queue.get(timeout=2)
            except queue.Empty:
                for worker_id, proc in active.items():
                    if worker_id not in exited and not proc.is_alive():
                        exited.add(worker_id)
                        errors.append(f"worker {worker_id} exited without cleanup event (code {proc.exitcode})")
                continue
            if kind == "started":
                status.setdefault("workers_detail", {})[str(wid)] = payload
                status["updated_at_utc"] = utc_now()
                macro.atomic_json(status, root / "status.json")
                print(f"[{utc_now()}] worker {wid} pid={payload['pid']} CODE V ready in {payload['startup_seconds']:.1f}s; "
                      f"lens loaded; timeout={payload['command_timeout']}; "
                      f"FOV=({payload['fov_x_deg']:.4g},{payload['fov_y_deg']:.4g})", flush=True)
            elif kind == "done":
                payload["completed_at_utc"] = utc_now()
                manifest["completed_chunks"].append(payload)
                manifest["completed_chunks"].sort(key=lambda x: int(x["chunk_id"]))
                manifest["updated_at_utc"] = utc_now()
                macro.atomic_json(manifest, manifest_path)
                done_states = sum(int(x["state_count"]) for x in manifest["completed_chunks"])
                avg_ms = 1000.0 * sum(float(x["macro_seconds"]) for x in manifest["completed_chunks"]) / max(done_states, 1)
                remaining = len(ade) * len(bde) - done_states
                eta = remaining * avg_ms / 1000.0 / max(len(workers), 1)
                status.update({
                    "updated_at_utc": utc_now(), "completed_chunks": len(manifest["completed_chunks"]),
                    "states_completed": done_states, "last_chunk_id": payload["chunk_id"],
                    "last_chunk_ms_per_state": payload["ms_per_state"], "mean_ms_per_state": avg_ms,
                    "eta_seconds": eta, "elapsed_seconds": time.time() - start_utc,
                    "failed_states": sum(int(x["failed_states"]) for x in manifest["completed_chunks"]),
                    "failed_rays": sum(int(x["failed_rays"]) for x in manifest["completed_chunks"]),
                })
                macro.atomic_json(status, root / "status.json")
                print(f"[{utc_now()}] checkpoint chunk {payload['chunk_id']:04d} worker {wid}: "
                      f"{payload['state_count']} states, {payload['ms_per_state']:.2f} ms/state, "
                      f"{status['completed_chunks']}/{len(plan)} chunks, ETA {eta/3600:.2f} h", flush=True)
            elif kind == "error":
                errors.append(f"worker {wid}:\n{payload}")
                print(f"[{utc_now()}] ERROR worker {wid}; checkpoints remain resumable", flush=True)
            elif kind == "exit":
                exited.add(wid)
    finally:
        for proc in workers:
            proc.join(timeout=30)
            if proc.is_alive():
                errors.append(f"{proc.name} failed to exit after StopCodeV; process remains alive")
    if errors:
        status.update({"state": "failed", "updated_at_utc": utc_now(), "errors": errors})
        manifest["state"] = "failed"
        manifest["updated_at_utc"] = utc_now()
        macro.atomic_json(manifest, manifest_path)
        macro.atomic_json(status, root / "status.json")
        raise RuntimeError("A0 macro run failed; use --resume after workers are stopped.\n" + "\n".join(errors))
    manifest["completed_chunks"].sort(key=lambda x: int(x["chunk_id"]))
    merge_and_publish(root, ade, bde, manifest["completed_chunks"], manifest)
    print(f"[{utc_now()}] COMPLETE: {root / 'distortion_grid.pkl'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--ade-min", type=float, default=-20.0)
    parser.add_argument("--ade-max", type=float, default=20.0)
    parser.add_argument("--ade-count", type=int, default=401)
    parser.add_argument("--bde-min", type=float, default=-20.0)
    parser.add_argument("--bde-max", type=float, default=20.0)
    parser.add_argument("--bde-count", type=int, default=401)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--block-bde", type=int, default=DEFAULT_BLOCK_BDE)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.block_bde < 1 or min(args.ade_count, args.bde_count) < 1:
        parser.error("workers, block size, and angle counts must be positive")
    run(args)


if __name__ == "__main__":
    main()
