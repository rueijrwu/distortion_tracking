"""Collect P4 distortion grids over accommodation and two-axis (ADE/BDE) eye rotation.

ADE (Ey, vertical gaze) and BDE (Ex, horizontal gaze) are both swept on the RC
surface, forming a 3-D grid: accommodation x ADE x BDE. Z is not swept; it is
held fixed at the P4 lens file's default (0 mm) and recorded as a constant
column for schema compatibility with the existing P4 loaders.

Completed accommodation/ADE groups are stored separately as resumable shards;
the canonical pickle is published only when the full sweep is complete. This
mirrors distortion_grid_p4.py's checkpoint structure, but shards by
(accommodation, ADE) rather than by (accommodation, Z), since the inner BDE
sweep at fixed accommodation/ADE is itself large enough to need its own
resumable unit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
import pythoncom
import win32com.client

from distortion_grid import (
    angular_fov,
    calculate_grid_at_biaxial_rotation,
)

# Reuse the P4 optical-model setup (lens file, GRIN accommodation model) verbatim
# from the single-axis P4 collector so both scripts stay consistent with the
# same lens revision and accommodation law.
from distortion_grid_p4 import (
    LENS_FILE,
    load_lens,
    read_accommodation_baseline,
    apply_accommodation,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "data" / "distortion_grid_p4_biaxial"

GRID_FIELDS = (
    "eye_rotation_deg",       # ADE, Ey (vertical gaze)
    "eye_rotation_x_deg",     # BDE, Ex (horizontal gaze)
    "accommodation_d",
    "z_mm",
    "field_x_relative",
    "field_y_relative",
    "paraxial_x_mm",
    "paraxial_y_mm",
    "center_absolute_x_mm",
    "center_absolute_y_mm",
    "real_x_mm",
    "real_y_mm",
    "radial_distortion_pct",
    "tangential_distortion_pct",
)

Z_MM = 0.0


def structured_rows(rows: list[dict[str, float]], accommodation_d: float) -> np.ndarray:
    dtype = [(name, np.float64) for name in GRID_FIELDS]
    values = [
        (
            float(row["eye_rotation_deg"]), float(row["eye_rotation_x_deg"]),
            accommodation_d, Z_MM,
            float(row["field_x_relative"]), float(row["field_y_relative"]),
            float(row["paraxial_x_mm"]), float(row["paraxial_y_mm"]),
            float(row["center_absolute_x_mm"]), float(row["center_absolute_y_mm"]),
            float(row["real_x_mm"]), float(row["real_y_mm"]),
            float(row["radial_distortion_pct"]), float(row["tangential_distortion_pct"]),
        )
        for row in rows
    ]
    return np.array(values, dtype=dtype)


def atomic_pickle(data, path: Path) -> None:
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    tmp_path.replace(path)


def atomic_json(metadata: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_metadata(config: dict, config_hash: str, groups: list[dict], total: int) -> dict:
    return {
        **config,
        "configuration_sha256": config_hash,
        "completed_groups": groups,
        "total_records_checkpointed": int(total),
        "complete": False,
    }


def validate_group(
    data: np.ndarray,
    bde_values: Sequence[float],
    accommodation_d: float,
    ade_deg: float,
    grid_lines: int,
) -> None:
    expected_size = len(bde_values) * grid_lines**2
    if data.ndim != 1 or data.size != expected_size or data.dtype.names != GRID_FIELDS:
        raise ValueError(
            f"Group shape/schema mismatch: got {data.size} records with {data.dtype.names}; "
            f"expected {expected_size} and {GRID_FIELDS}"
        )
    for field in GRID_FIELDS:
        if not np.all(np.isfinite(data[field])):
            raise ValueError(f"Group contains nonfinite values in {field}")
    if not np.array_equal(
        np.unique(data["eye_rotation_x_deg"]),
        np.unique(np.asarray(bde_values, dtype=np.float64)),
    ):
        raise ValueError("Group BDE values do not exactly match the requested sweep")
    if not np.all(data["eye_rotation_deg"] == ade_deg):
        raise ValueError("Group ADE values do not match the requested value")
    if not np.all(data["accommodation_d"] == accommodation_d):
        raise ValueError("Group accommodation values do not match the requested value")
    if not np.all(data["z_mm"] == Z_MM):
        raise ValueError("Group Z values are not held at the fixed constant")

    for bde in bde_values:
        bde_rows = data[data["eye_rotation_x_deg"] == float(bde)]
        pairs = np.unique(
            np.column_stack((bde_rows["field_x_relative"], bde_rows["field_y_relative"])),
            axis=0,
        )
        if bde_rows.size != grid_lines**2 or pairs.shape[0] != grid_lines**2:
            raise ValueError(
                f"BDE {bde:g}° has {pairs.shape[0]} unique field coordinate pairs; "
                f"expected {grid_lines**2}"
            )
        for center_field in ("center_absolute_x_mm", "center_absolute_y_mm"):
            if not np.all(bde_rows[center_field] == bde_rows[center_field][0]):
                raise ValueError(f"BDE {bde:g}° has varying {center_field} across its grid")


def validate_complete_sweep(
    data: np.ndarray,
    ade_values: Sequence[float],
    bde_values: Sequence[float],
    accommodations: Sequence[float],
    expected_count: int,
) -> None:
    if data.ndim != 1 or data.size != expected_count or data.dtype.names != GRID_FIELDS:
        raise ValueError(
            f"Final dataset has {data.size} records with schema {data.dtype.names}; "
            f"expected {expected_count} records with {GRID_FIELDS}"
        )
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
    if not np.all(np.isfinite(data.view(np.float64).reshape(data.size, len(GRID_FIELDS)))):
        raise ValueError("Final dataset contains nonfinite numeric values")
    if not np.all(data["z_mm"] == Z_MM):
        raise ValueError("Final dataset Z values are not held at the fixed constant")


def collect(
    ade_values: Sequence[float],
    bde_values: Sequence[float],
    accommodations: Sequence[float],
    grid_lines: int,
    output_dir: Path,
    resume: bool = False,
) -> None:
    if grid_lines < 3 or grid_lines > 21:
        raise ValueError("Grid line count must be between 3 and 21.")
    if not ade_values or not bde_values or not accommodations:
        raise ValueError("ADE, BDE, and accommodation sweeps must be nonempty")
    if not LENS_FILE.is_file():
        raise FileNotFoundError(f"P4 CODE V lens file not found: {LENS_FILE}")

    checkpoint_dir = output_dir.with_name(output_dir.name + ".checkpoint")
    checkpoint_meta = checkpoint_dir / "metadata.json"
    config = {
        "model": str(LENS_FILE.relative_to(PROJECT_DIR)),
        "model_sha256": hashlib.sha256(LENS_FILE.read_bytes()).hexdigest(),
        "zoom": "z1 (P4)",
        "ade_deg": [float(value) for value in ade_values],
        "bde_deg": [float(value) for value in bde_values],
        "accommodation_d": [float(value) for value in accommodations],
        "z_mm_fixed": Z_MM,
        "grid_lines": int(grid_lines),
        "fields": list(GRID_FIELDS),
        "expected_record_count": (
            len(ade_values) * len(bde_values) * len(accommodations) * grid_lines**2
        ),
    }
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    expected_group_count = len(accommodations) * len(ade_values)
    expected_rows_per_group = len(bde_values) * grid_lines**2
    groups: list[dict] = []
    total_records = 0

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
        for number, group in enumerate(prior_groups):
            ai, ei = divmod(number, len(ade_values))
            if (
                group.get("accommodation_index") != ai
                or group.get("ade_index") != ei
                or not math.isclose(float(group.get("accommodation_d", math.nan)), accommodations[ai], rel_tol=0.0, abs_tol=1e-8)
                or not math.isclose(float(group.get("ade_deg", math.nan)), ade_values[ei], rel_tol=0.0, abs_tol=1e-8)
            ):
                raise ValueError("Checkpoint groups are not a prefix of the requested sweep")
            shard_path = checkpoint_dir / group["shard"]
            if not shard_path.is_file() or file_sha256(shard_path) != group.get("sha256"):
                raise ValueError(f"Checkpoint shard is missing or corrupt: {shard_path}")
            with shard_path.open("rb") as stream:
                shard = pickle.load(stream)
            if (
                not isinstance(shard, np.ndarray) or shard.ndim != 1
                or shard.dtype.names != GRID_FIELDS or shard.size != expected_rows_per_group
                or group.get("record_count") != expected_rows_per_group
                or not np.allclose(np.unique(shard["accommodation_d"]), [accommodations[ai]], rtol=0.0, atol=1e-8)
                or not np.allclose(np.unique(shard["eye_rotation_deg"]), [ade_values[ei]], rtol=0.0, atol=1e-8)
            ):
                raise ValueError(f"Checkpoint shard has unexpected contents: {shard_path}")
            groups.append(group)
            total_records += int(shard.size)
        if metadata.get("total_records_checkpointed") != total_records:
            raise ValueError("Checkpoint record total does not match its groups")
        print(f"Resuming after {len(groups)} accommodation/ADE groups.", flush=True)
    elif checkpoint_dir.exists():
        raise FileExistsError(
            f"Checkpoint exists under {checkpoint_dir}; pass --resume or choose a new output directory"
        )
    else:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(make_metadata(config, config_hash, [], 0), checkpoint_meta)

    if len(groups) < expected_group_count:
        pythoncom.CoUninitialize()
        pythoncom.CoInitialize()
        cv = None
        started = False
        try:
            print("Connecting to CODE V COM...", flush=True)
            cv = win32com.client.Dispatch("CodeV.Application")
            cv.CommandTimeout = 600000
            cv.MaxTextBufferSize = 1000000
            cv.StartingDirectory = str(LENS_FILE.parent)
            cv.StartCodeV()
            started = True
            current_accommodation = None
            for number in range(len(groups), expected_group_count):
                ai, ei = divmod(number, len(ade_values))
                accommodation_d, ade_deg = float(accommodations[ai]), float(ade_values[ei])
                if current_accommodation != ai:
                    print(f"Reloading P4 z1 for accommodation {accommodation_d:g} D", flush=True)
                    load_lens(cv)
                    baseline = read_accommodation_baseline(cv)
                    targets = apply_accommodation(cv, baseline, accommodation_d)
                    fov_x, fov_y = angular_fov(cv)
                    current_accommodation = ai
                print(
                    f"Accommodation {accommodation_d:g} D, ADE={ade_deg:g}° "
                    f"({ei + 1}/{len(ade_values)}), fields X={fov_x:g}°, Y={fov_y:g}°",
                    flush=True,
                )
                group_rows: list[dict[str, float]] = []
                for bi, bde in enumerate(bde_values, start=1):
                    if bi == 1 or bi % 50 == 0 or bi == len(bde_values):
                        print(f"    BDE {bde:g}° ({bi}/{len(bde_values)})", flush=True)
                    group_rows.extend(
                        calculate_grid_at_biaxial_rotation(
                            cv, ade_deg, float(bde), grid_lines, fov_x, fov_y, z_mm=Z_MM
                        )
                    )
                group_data = structured_rows(group_rows, accommodation_d)
                validate_group(group_data, bde_values, accommodation_d, ade_deg, grid_lines)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                shard_name = f"group_{number:06d}.pkl"
                shard_path = checkpoint_dir / shard_name
                atomic_pickle(group_data, shard_path)
                group = {
                    "accommodation_index": ai,
                    "accommodation_d": accommodation_d,
                    "ade_index": ei,
                    "ade_deg": ade_deg,
                    "record_count": int(group_data.size),
                    "shard": shard_name,
                    "sha256": file_sha256(shard_path),
                    "geometry_baseline": baseline,
                    "geometry_target": targets,
                }
                groups.append(group)
                total_records += int(group_data.size)
                atomic_json(make_metadata(config, config_hash, groups, total_records), checkpoint_meta)
                print(f"Checkpointed {total_records:,} records after this group.", flush=True)
        finally:
            try:
                if cv is not None and started:
                    cv.StopCodeV()
            finally:
                pythoncom.CoUninitialize()

    # Reassemble only after every group is checkpointed; existing canonical output remains intact until then.
    final_parts = []
    for group in groups:
        with (checkpoint_dir / group["shard"]).open("rb") as stream:
            final_parts.append(pickle.load(stream))
    final_data = np.concatenate(final_parts)
    try:
        validate_complete_sweep(
            final_data, ade_values, bde_values, accommodations, config["expected_record_count"]
        )
    except ValueError as exc:
        raise RuntimeError(
            f"Final P4 biaxial sweep validation failed; checkpoints remain available for --resume: {exc}"
        ) from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = output_dir / "distortion_grid.pkl"
    atomic_pickle(final_data, canonical_path)
    import shutil
    stale_metadata = output_dir / "metadata.json"
    if stale_metadata.exists():
        stale_metadata.unlink()
    shutil.rmtree(checkpoint_dir)
    print(f"Published {final_data.size:,} records to {canonical_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ade-min", type=float, default=-20.0)
    parser.add_argument("--ade-max", type=float, default=20.0)
    parser.add_argument("--ade-count", type=int, default=401)
    parser.add_argument("--bde-min", type=float, default=-20.0)
    parser.add_argument("--bde-max", type=float, default=20.0)
    parser.add_argument("--bde-count", type=int, default=401)
    parser.add_argument("--accommodation-min", type=float, default=0.0)
    parser.add_argument("--accommodation-max", type=float, default=4.0)
    parser.add_argument("--accommodation-count", type=int, default=41)
    parser.add_argument("--grid-lines", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--resume", action="store_true", help="resume matching accommodation/ADE checkpoints")
    args = parser.parse_args()
    if args.ade_count < 1 or args.bde_count < 1 or args.accommodation_count < 1:
        parser.error("ADE, BDE, and accommodation counts must be at least 1")
    ade_values = np.linspace(args.ade_min, args.ade_max, args.ade_count)
    bde_values = np.linspace(args.bde_min, args.bde_max, args.bde_count)
    accommodations = np.linspace(args.accommodation_min, args.accommodation_max, args.accommodation_count)
    collect(
        ade_values.tolist(), bde_values.tolist(), accommodations.tolist(),
        args.grid_lines, args.output_dir, args.resume,
    )


if __name__ == "__main__":
    main()
