"""Collect P4 distortion grids over accommodation, cornea Z, and RC rotation.

Completed accommodation/Z groups are stored separately as resumable shards;
the canonical pickle is published only when the full sweep is complete.
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
    calculate_grid_at_rotation,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
LENS_FILE = PROJECT_DIR / "Lens" / "p4_ME.len"
OUTPUT_DIR = PROJECT_DIR / "data" / "distortion_grid_p4"

WAVELENGTH = 850.0
COEFF_850 = np.array([
    [1.3623935136e+00, -1.8001392205e-05, 3.6735803323e-06],
    [6.7394986702e-02, -9.7774753376e-04, -6.4222012087e-05],
    [-3.9461873882e-02, -6.4768189371e-04, 4.6215222695e-04],
    [4.4424787588e-02, 2.2422863661e-03, -1.0416671225e-03],
    [-5.5515479042e-02, -8.6352875872e-04, 1.0665069364e-03],
    [3.5725367937e-02, -4.4653401125e-04, -5.7214066466e-04],
    [-1.2033695765e-02, 4.1750060378e-04, 1.6559364161e-04],
    [2.0486551193e-03, -1.1051053515e-04, -2.4340126390e-05],
    [-1.3968088637e-04, 1.0004847342e-05, 1.4125818106e-06],
], dtype=np.float64)
CAUCHY_LENS_CORE = [1.401105, 6.576875e3, -6.162814e8, 5.958617e13]
CAUCHY_LENS_EDGE = [1.354665, 6.358883e3, -5.958546e8, 5.761117e13]

AQUEOUS_THICKNESS_SLP = -0.04
LENS_THICKNESS_SLP = 0.06
LENS_F_RADIUS_SLP = -0.65
LENS_F_CONIC_SLP = -0.4
LENS_B_RADIUS_SLP = 0.17
LENS_B_CONIC_SLP = -0.25
LENS_SEMI_AP_SLP = -0.259 / 2

S_RC = 's"RC"'
S_CORNEA_B = 's"CorneaB"'
S_LENS_F = 's"LensF"'
S_LENS_B = 's"LensB"'
S_Z_MOV = 's"Cornea_ENT_D"'
ZOOM = "z1"

GRID_FIELDS = (
    "eye_rotation_deg",
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


def cv_eval(cv, command: str, *args: str) -> float:
    return float(cv.EvaluateExpression(f"({command} {' '.join(args)})"))


def cauchy(wavelength_nm: float, na: float, nb: float, nc: float, nd: float) -> float:
    return na + nb / wavelength_nm**2 + nc / wavelength_nm**4 + nd / wavelength_nm**6


def grin_glass_command(accommodation_d: float, lens_semi_ap_mm: float) -> str:
    ridx_core = cauchy(WAVELENGTH, *CAUCHY_LENS_CORE)
    ridx_edge = cauchy(WAVELENGTH, *CAUCHY_LENS_EDGE)
    delta_ridx = ridx_core - ridx_edge
    powers = accommodation_d ** np.arange(COEFF_850.shape[1])
    urn_c = np.append(COEFF_850 @ powers, -delta_ridx / lens_semi_ap_mm**2)
    coefficient_commands = ";\n".join(
        f"udg c{index} {value:.6e}"
        for index, value in enumerate(urn_c[1:], start=1)
    )
    return (
        f"prv;\npwl {WAVELENGTH:.2f};\n'Lens' {urn_c[0]:.6f};\n"
        f"udg 0.1;\n{coefficient_commands};\nend"
    )


def command_checked(cv, command: str, context: str) -> str:
    response = cv.Command(command)
    if "Command End:" not in response or any(
        line.lstrip().lower().startswith("error:") for line in response.splitlines()
    ):
        raise RuntimeError(f"CODE V failed {context}: {response!r}")
    return response


def load_lens(cv) -> str:
    if not LENS_FILE.is_file():
        raise FileNotFoundError(f"P4 CODE V lens file not found: {LENS_FILE}")
    response = command_checked(cv, f'res "{LENS_FILE}"', f"loading {LENS_FILE}")
    return response


def read_accommodation_baseline(cv) -> dict[str, float]:
    baseline = {
        "aqueous_thickness_mm": cv_eval(cv, "thi", ZOOM, S_CORNEA_B),
        "lens_thickness_mm": cv_eval(cv, "thi", ZOOM, S_LENS_F),
        "lens_f_radius_mm": cv_eval(cv, "rdy", ZOOM, S_LENS_F),
        "lens_b_radius_mm": cv_eval(cv, "rdy", ZOOM, S_LENS_B),
        "lens_f_conic": cv_eval(cv, "k", ZOOM, S_LENS_F),
        "lens_b_conic": cv_eval(cv, "k", ZOOM, S_LENS_B),
        "lens_semi_ap_mm": cv_eval(cv, "cir", S_LENS_F),
    }
    if not all(math.isfinite(value) for value in baseline.values()):
        raise RuntimeError(f"P4 z1 baseline geometry contains nonfinite values: {baseline}")
    return baseline


def accommodation_targets(baseline: dict[str, float], accommodation_d: float) -> dict[str, float]:
    targets = {
        "aqueous_thickness_mm": baseline["aqueous_thickness_mm"] + AQUEOUS_THICKNESS_SLP * accommodation_d,
        "lens_thickness_mm": baseline["lens_thickness_mm"] + LENS_THICKNESS_SLP * accommodation_d,
        "lens_f_radius_mm": baseline["lens_f_radius_mm"] + LENS_F_RADIUS_SLP * accommodation_d,
        "lens_b_radius_mm": baseline["lens_b_radius_mm"] + LENS_B_RADIUS_SLP * accommodation_d,
        "lens_f_conic": baseline["lens_f_conic"] + LENS_F_CONIC_SLP * accommodation_d,
        "lens_b_conic": baseline["lens_b_conic"] + LENS_B_CONIC_SLP * accommodation_d,
        "lens_semi_ap_mm": baseline["lens_semi_ap_mm"] + LENS_SEMI_AP_SLP * accommodation_d**0.81,
    }
    if not all(math.isfinite(value) for value in targets.values()):
        raise RuntimeError(f"P4 accommodation {accommodation_d:g} D gives nonfinite geometry")
    return targets


def apply_accommodation(cv, baseline: dict[str, float], accommodation_d: float) -> dict[str, float]:
    targets = accommodation_targets(baseline, accommodation_d)
    commands = (
        f"thi {S_CORNEA_B} {ZOOM} {targets['aqueous_thickness_mm']:.12g};"
        f"thi {S_LENS_F} {ZOOM} {targets['lens_thickness_mm']:.12g};"
        f"rdy {S_LENS_F} {ZOOM} {targets['lens_f_radius_mm']:.12g};"
        f"rdy {S_LENS_B} {ZOOM} {targets['lens_b_radius_mm']:.12g};"
        f"k {S_LENS_F} {ZOOM} {targets['lens_f_conic']:.12g};"
        f"k {S_LENS_B} {ZOOM} {targets['lens_b_conic']:.12g}"
    )
    command_checked(cv, commands, f"applying accommodation {accommodation_d:g} D geometry")
    checks = {
        "aqueous_thickness_mm": cv_eval(cv, "thi", ZOOM, S_CORNEA_B),
        "lens_thickness_mm": cv_eval(cv, "thi", ZOOM, S_LENS_F),
        "lens_f_radius_mm": cv_eval(cv, "rdy", ZOOM, S_LENS_F),
        "lens_b_radius_mm": cv_eval(cv, "rdy", ZOOM, S_LENS_B),
        "lens_f_conic": cv_eval(cv, "k", ZOOM, S_LENS_F),
        "lens_b_conic": cv_eval(cv, "k", ZOOM, S_LENS_B),
    }
    for key, expected in checks.items():
        if not math.isclose(expected, targets[key], rel_tol=0.0, abs_tol=1e-8):
            raise RuntimeError(
                f"Accommodation {accommodation_d:g} D {key} readback {expected:g} "
                f"does not match target {targets[key]:g}"
            )

    command_checked(
        cv,
        grin_glass_command(accommodation_d, targets["lens_semi_ap_mm"]),
        f"updating GRIN glass for accommodation {accommodation_d:g} D",
    )
    return targets


def structured_rows(rows: list[dict[str, float]], accommodation_d: float, z_mm: float) -> np.ndarray:
    dtype = [(name, np.float64) for name in GRID_FIELDS]
    values = [
        (
            float(row["eye_rotation_deg"]), accommodation_d, z_mm,
            float(row["field_x_relative"]), float(row["field_y_relative"]),
            float(row["paraxial_x_mm"]), float(row["paraxial_y_mm"]),
            float(row["center_absolute_x_mm"]), float(row["center_absolute_y_mm"]),
            float(row["real_x_mm"]), float(row["real_y_mm"]),
            float(row["radial_distortion_pct"]), float(row["tangential_distortion_pct"]),
        )
        for row in rows
    ]
    return np.array(values, dtype=dtype)


def atomic_pickle(data: np.ndarray, path: Path) -> None:
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
    rotations: Sequence[float],
    accommodation_d: float,
    z_mm: float,
    grid_lines: int,
) -> None:
    expected_size = len(rotations) * grid_lines**2
    if data.ndim != 1 or data.size != expected_size or data.dtype.names != GRID_FIELDS:
        raise ValueError(
            f"Group shape/schema mismatch: got {data.size} records with {data.dtype.names}; "
            f"expected {expected_size} and {GRID_FIELDS}"
        )
    for field in GRID_FIELDS:
        if not np.all(np.isfinite(data[field])):
            raise ValueError(f"Group contains nonfinite values in {field}")
    if not np.array_equal(
        np.unique(data["eye_rotation_deg"]),
        np.unique(np.asarray(rotations, dtype=np.float64)),
    ):
        raise ValueError("Group rotation values do not exactly match the requested rotations")
    if not np.all(data["accommodation_d"] == accommodation_d):
        raise ValueError("Group accommodation values do not match the requested value")
    if not np.all(data["z_mm"] == z_mm):
        raise ValueError("Group Z values do not match the requested value")

    for rotation in rotations:
        rotation_rows = data[data["eye_rotation_deg"] == float(rotation)]
        pairs = np.unique(
            np.column_stack((rotation_rows["field_x_relative"], rotation_rows["field_y_relative"])),
            axis=0,
        )
        if rotation_rows.size != grid_lines**2 or pairs.shape[0] != grid_lines**2:
            raise ValueError(
                f"Rotation {rotation:g}° has {pairs.shape[0]} unique field coordinate pairs; "
                f"expected {grid_lines**2}"
            )
        for center_field in ("center_absolute_x_mm", "center_absolute_y_mm"):
            if not np.all(rotation_rows[center_field] == rotation_rows[center_field][0]):
                raise ValueError(
                    f"Rotation {rotation:g}° has varying {center_field} across its grid"
                )


def validate_complete_sweep(
    data: np.ndarray,
    rotations: Sequence[float],
    accommodations: Sequence[float],
    z_values: Sequence[float],
    expected_count: int,
) -> None:
    if data.ndim != 1 or data.size != expected_count or data.dtype.names != GRID_FIELDS:
        raise ValueError(
            f"Final dataset has {data.size} records with schema {data.dtype.names}; "
            f"expected {expected_count} records with {GRID_FIELDS}"
        )
    expected_dimensions = {
        "accommodation_d": np.asarray(accommodations, dtype=np.float64),
        "z_mm": np.asarray(z_values, dtype=np.float64),
        "eye_rotation_deg": np.asarray(rotations, dtype=np.float64),
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


def collect(
    rotations: Sequence[float],
    accommodations: Sequence[float],
    z_values: Sequence[float],
    grid_lines: int,
    output_dir: Path,
    resume: bool = False,
) -> None:
    if grid_lines < 3 or grid_lines > 21:
        raise ValueError("Grid line count must be between 3 and 21.")
    if not rotations or not accommodations or not z_values:
        raise ValueError("Rotation, accommodation, and Z sweeps must be nonempty")
    if not LENS_FILE.is_file():
        raise FileNotFoundError(f"P4 CODE V lens file not found: {LENS_FILE}")

    checkpoint_dir = output_dir.with_name(output_dir.name + ".checkpoint")
    checkpoint_meta = checkpoint_dir / "metadata.json"
    config = {
        "model": str(LENS_FILE.relative_to(PROJECT_DIR)),
        "model_sha256": hashlib.sha256(LENS_FILE.read_bytes()).hexdigest(),
        "zoom": "z1 (P4)",
        "rotation_deg": [float(value) for value in rotations],
        "accommodation_d": [float(value) for value in accommodations],
        "z_mm": [float(value) for value in z_values],
        "z_surface": S_Z_MOV,
        "grid_lines": int(grid_lines),
        "fields": list(GRID_FIELDS),
        "expected_record_count": len(rotations) * len(accommodations) * len(z_values) * grid_lines**2,
    }
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    expected_group_count = len(accommodations) * len(z_values)
    expected_rows_per_group = len(rotations) * grid_lines**2
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
            ai, zi = divmod(number, len(z_values))
            if (
                group.get("accommodation_index") != ai
                or group.get("z_index") != zi
                or not math.isclose(float(group.get("accommodation_d", math.nan)), accommodations[ai], rel_tol=0.0, abs_tol=1e-8)
                or not math.isclose(float(group.get("z_mm", math.nan)), z_values[zi], rel_tol=0.0, abs_tol=1e-8)
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
                or not np.allclose(np.unique(shard["z_mm"]), [z_values[zi]], rtol=0.0, atol=1e-8)
            ):
                raise ValueError(f"Checkpoint shard has unexpected contents: {shard_path}")
            groups.append(group)
            total_records += int(shard.size)
        if metadata.get("total_records_checkpointed") != total_records:
            raise ValueError("Checkpoint record total does not match its groups")
        print(f"Resuming after {len(groups)} accommodation/Z groups.", flush=True)
    elif checkpoint_dir.exists():
        raise FileExistsError(
            f"Checkpoint exists under {checkpoint_dir}; pass --resume or choose a new output directory"
        )
    else:
        # Make an empty, valid checkpoint before CODE V starts. If interruption
        # occurs between a group's shard and metadata replacement, resume can
        # safely continue from the last metadata-confirmed group.
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
                ai, zi = divmod(number, len(z_values))
                accommodation_d, z_mm = float(accommodations[ai]), float(z_values[zi])
                if current_accommodation != ai:
                    print(f"Reloading P4 z1 for accommodation {accommodation_d:g} D", flush=True)
                    load_lens(cv)
                    baseline = read_accommodation_baseline(cv)
                    targets = apply_accommodation(cv, baseline, accommodation_d)
                    fov_x, fov_y = angular_fov(cv)
                    current_accommodation = ai
                print(
                    f"Accommodation {accommodation_d:g} D, Z={z_mm:g} mm "
                    f"({zi + 1}/{len(z_values)}), fields X={fov_x:g}°, Y={fov_y:g}°",
                    flush=True,
                )
                group_rows: list[dict[str, float]] = []
                for ri, rotation in enumerate(rotations, start=1):
                    print(f"  RC {rotation:g}° ({ri}/{len(rotations)})", flush=True)
                    group_rows.extend(
                        calculate_grid_at_rotation(
                            cv, float(rotation), grid_lines, fov_x, fov_y, z_mm=z_mm
                        )
                    )
                group_data = structured_rows(group_rows, accommodation_d, z_mm)
                validate_group(group_data, rotations, accommodation_d, z_mm, grid_lines)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                shard_name = f"group_{number:04d}.pkl"
                shard_path = checkpoint_dir / shard_name
                atomic_pickle(group_data, shard_path)
                group = {
                    "accommodation_index": ai,
                    "accommodation_d": accommodation_d,
                    "z_index": zi,
                    "z_mm": z_mm,
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
            final_data, rotations, accommodations, z_values, config["expected_record_count"]
        )
    except ValueError as exc:
        raise RuntimeError(f"Final P4 sweep validation failed; checkpoints remain available for --resume: {exc}") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = output_dir / "distortion_grid.pkl"
    atomic_pickle(final_data, canonical_path)
    # Checkpoint metadata lives only in the staging directory; the final data
    # folder contains the structured NumPy pickle alone.
    import shutil
    stale_metadata = output_dir / "metadata.json"
    if stale_metadata.exists():
        stale_metadata.unlink()
    shutil.rmtree(checkpoint_dir)
    print(f"Published {final_data.size:,} records to {canonical_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rotation-min", type=float, default=-20.0)
    parser.add_argument("--rotation-max", type=float, default=20.0)
    parser.add_argument("--rotation-count", type=int, default=401)
    parser.add_argument("--accommodation-min", type=float, default=0.0)
    parser.add_argument("--accommodation-max", type=float, default=5.0)
    parser.add_argument("--accommodation-count", type=int, default=51)
    parser.add_argument("--z-min", type=float, default=0.0)
    parser.add_argument("--z-max", type=float, default=0.0)
    parser.add_argument("--z-count", type=int, default=1)
    parser.add_argument("--grid-lines", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--resume", action="store_true", help="resume matching accommodation/Z checkpoints")
    args = parser.parse_args()
    if args.rotation_count < 1 or args.accommodation_count < 1 or args.z_count < 1:
        parser.error("rotation, accommodation, and Z counts must be at least 1")
    rotations = np.linspace(args.rotation_min, args.rotation_max, args.rotation_count)
    accommodations = np.linspace(args.accommodation_min, args.accommodation_max, args.accommodation_count)
    z_values = np.linspace(args.z_min, args.z_max, args.z_count)
    collect(
        rotations.tolist(), accommodations.tolist(), z_values.tolist(),
        args.grid_lines, args.output_dir, args.resume,
    )


if __name__ == "__main__":
    main()
