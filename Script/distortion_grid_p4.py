"""Collect P4 distortion grids over accommodation and RC rotation with CODE V.

Each completed accommodation group is checkpointed to the P4-specific output
directory. The P1 collector and its data remain independent.
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
ZOOM = "z1"

GRID_FIELDS = (
    "eye_rotation_deg",
    "accommodation_d",
    "field_x_relative",
    "field_y_relative",
    "paraxial_x_mm",
    "paraxial_y_mm",
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


def structured_rows(rows: list[dict[str, float]], accommodation_d: float) -> np.ndarray:
    dtype = [(name, np.float64) for name in GRID_FIELDS]
    values = [
        (float(row["eye_rotation_deg"]), accommodation_d)
        + tuple(float(row[name]) for name in GRID_FIELDS[2:])
        for row in rows
    ]
    return np.array(values, dtype=dtype)


def save_checkpoint(data: np.ndarray, output_dir: Path, metadata: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = output_dir / "distortion_grid.pkl"
    tmp_path = pkl_path.with_suffix(".pkl.tmp")
    with tmp_path.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
    tmp_path.replace(pkl_path)
    metadata_path = output_dir / "metadata.json"
    meta_tmp = metadata_path.with_suffix(".json.tmp")
    meta_tmp.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    meta_tmp.replace(metadata_path)


def collect(
    rotations: Sequence[float],
    accommodations: Sequence[float],
    grid_lines: int,
    output_dir: Path,
    resume: bool = False,
) -> None:
    if grid_lines < 3 or grid_lines > 21:
        raise ValueError("Grid line count must be between 3 and 21.")
    if not LENS_FILE.is_file():
        raise FileNotFoundError(f"P4 CODE V lens file not found: {LENS_FILE}")

    digest = hashlib.sha256(LENS_FILE.read_bytes()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    accumulated: list[np.ndarray] = []
    completed_groups: list[dict] = []
    pkl_path = output_dir / "distortion_grid.pkl"
    metadata_path = output_dir / "metadata.json"
    if resume:
        if not pkl_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError("--resume requires both checkpoint files in the output directory")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_config = {
            "model": str(LENS_FILE.relative_to(PROJECT_DIR)),
            "model_sha256": digest,
            "zoom": "z1 (P4)",
            "rotation_deg": [float(value) for value in rotations],
            "accommodation_d": [float(value) for value in accommodations],
            "grid_lines": grid_lines,
            "fields": list(GRID_FIELDS),
        }
        if any(metadata.get(key) != value for key, value in expected_config.items()):
            raise ValueError("Checkpoint model or sweep settings do not match this run")
        with pkl_path.open("rb") as stream:
            prior_data = pickle.load(stream)
        if (
            not isinstance(prior_data, np.ndarray)
            or prior_data.ndim != 1
            or prior_data.dtype.names != GRID_FIELDS
        ):
            raise ValueError("Checkpoint has an unexpected structured-array format")
        expected_rows_per_group = len(rotations) * grid_lines**2
        prior_groups = metadata.get("completed_groups", [])
        if len(prior_data) != len(prior_groups) * expected_rows_per_group:
            raise ValueError("Checkpoint record count does not match its completed groups")
        for index, group in enumerate(prior_groups):
            if index >= len(accommodations) or not math.isclose(
                float(group.get("accommodation_d", math.nan)),
                float(accommodations[index]),
                rel_tol=0.0,
                abs_tol=1e-8,
            ):
                raise ValueError("Checkpoint groups are not a prefix of requested accommodations")
            start = index * expected_rows_per_group
            end = start + expected_rows_per_group
            group_data = prior_data[start:end]
            if group_data.size != expected_rows_per_group or not np.allclose(
                np.unique(group_data["accommodation_d"]),
                [accommodations[index]],
                rtol=0.0,
                atol=1e-8,
            ):
                raise ValueError("Checkpoint accommodation data does not match its metadata")
            accumulated.append(group_data)
            completed_groups.append(group)
        if len(prior_groups) > len(accommodations):
            raise ValueError("Checkpoint contains more groups than requested")
        if len(completed_groups) == len(accommodations):
            print("All requested P4 accommodation groups are already checkpointed.", flush=True)
            return
        print(f"Resuming after {len(completed_groups)} completed accommodation groups.", flush=True)
    elif pkl_path.exists() or metadata_path.exists():
        raise FileExistsError(
            f"P4 checkpoint exists under {output_dir}; pass --resume to continue it "
            "or choose a new --output-dir."
        )

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

        for group_offset in range(len(completed_groups), len(accommodations)):
            group_index = group_offset + 1
            accommodation_d = accommodations[group_offset]
            print(
                f"Loading P4 z1 for accommodation {accommodation_d:g} D "
                f"({group_index}/{len(accommodations)})...",
                flush=True,
            )
            load_lens(cv)
            baseline = read_accommodation_baseline(cv)
            targets = apply_accommodation(cv, baseline, float(accommodation_d))
            fov_x, fov_y = angular_fov(cv)
            print(
                f"P4 z1 fields: X={fov_x:g}°, Y={fov_y:g}°; "
                f"geometry readback verified.",
                flush=True,
            )
            group_rows: list[dict[str, float]] = []
            for rotation_index, rotation in enumerate(rotations, start=1):
                print(
                    f"  RC {rotation:g}° ({rotation_index}/{len(rotations)}) at "
                    f"{accommodation_d:g} D",
                    flush=True,
                )
                rows = calculate_grid_at_rotation(
                    cv, float(rotation), grid_lines, fov_x, fov_y
                )
                group_rows.extend(rows)

            group_data = structured_rows(group_rows, float(accommodation_d))
            accumulated.append(group_data)
            completed_groups.append(
                {
                    "accommodation_d": float(accommodation_d),
                    "record_count": int(group_data.size),
                    "geometry_baseline": baseline,
                    "geometry_target": targets,
                }
            )
            data = np.concatenate(accumulated)
            metadata = {
                "model": str(LENS_FILE.relative_to(PROJECT_DIR)),
                "model_sha256": digest,
                "zoom": "z1 (P4)",
                "rotation_deg": [float(value) for value in rotations],
                "accommodation_d": [float(value) for value in accommodations],
                "grid_lines": grid_lines,
                "fields": list(GRID_FIELDS),
                "completed_groups": completed_groups,
                "total_records_checkpointed": int(data.size),
            }
            save_checkpoint(data, output_dir, metadata)
            print(
                f"Checkpointed {data.size:,} records after {accommodation_d:g} D.",
                flush=True,
            )
    finally:
        try:
            if cv is not None and started:
                cv.StopCodeV()
        finally:
            pythoncom.CoUninitialize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rotation-min", type=float, default=-20.0)
    parser.add_argument("--rotation-max", type=float, default=20.0)
    parser.add_argument("--rotation-count", type=int, default=401)
    parser.add_argument("--accommodation-min", type=float, default=0.0)
    parser.add_argument("--accommodation-max", type=float, default=5.0)
    parser.add_argument("--accommodation-count", type=int, default=51)
    parser.add_argument("--grid-lines", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--resume", action="store_true", help="resume matching completed accommodation checkpoints")
    args = parser.parse_args()
    if args.rotation_count < 1 or args.accommodation_count < 1:
        parser.error("rotation and accommodation counts must be at least 1")
    rotations = np.linspace(args.rotation_min, args.rotation_max, args.rotation_count)
    accommodations = np.linspace(
        args.accommodation_min, args.accommodation_max, args.accommodation_count
    )
    collect(rotations.tolist(), accommodations.tolist(), args.grid_lines, args.output_dir, args.resume)
    print(f"P4 grid saved to {(args.output_dir / 'distortion_grid.pkl').resolve()}")


if __name__ == "__main__":
    main()
