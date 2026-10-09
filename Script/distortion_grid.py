"""Calculate eye-rotation distortion grids with CODE V COM and save a pickle.

This follows the ray tracing and central 5% reference-grid calculation in
CODE V's supplied DIST macro, using the COM RAYRSI method directly. Plot the
saved data later with plot_distortion_grid.py without starting CODE V.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
import pythoncom
import win32com.client


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
LENS_DIR = PROJECT_DIR / "Lens"
SEQ_FILE = LENS_DIR / "p1_ME.seq"
OUTPUT_DIR = PROJECT_DIR / "data" / "distortion_grid"

EYE_ROTATION_MIN_DEG = -20.0
EYE_ROTATION_MAX_DEG = 20.0
EYE_ROTATION_COUNT = 401

Z_MAX_MM = -5
Z_MIN_MM = 5
Z_COUNT = 51

GRID_LINES = 3
ZOOM_POSITION = 1
S_RC = "s\"RC\""
S_Z_MOV = "s\"Cornea_ENT_D\""

def cv_eval(cv, command: str, *args: str) -> float:
    """Evaluate a CODE V expression using raytracing.py's helper pattern."""
    joined_args = " ".join(args)
    return float(cv.EvaluateExpression(f"({command} {joined_args})"))


def trace_rsi(
    cv, field_x: float, field_y: float, field_number: int = 0
) -> tuple[float, float] | None:
    """Trace a chief ray through COM RAYRSI and return image-surface X/Y."""
    ray_input = [0.0, 0.0, float(field_x), float(field_y)]
    failed_surface = int(cv.RAYRSI(ZOOM_POSITION, 1, field_number, 0, ray_input))
    if failed_surface != 0:
        return None
    return cv_eval(cv, "x", "si"), cv_eval(cv, "y", "si")


def angular_fov(cv) -> tuple[float, float]:
    """Match DIST's automatic X/Y semi-field calculation for angular fields."""
    n_fields = int(cv.FieldCount)
    max_x = max(abs(cv_eval(cv, "xan", f"f{i}", "z1")) for i in range(1, n_fields + 1))
    max_y = max(abs(cv_eval(cv, "yan", f"f{i}", "z1")) for i in range(1, n_fields + 1))
    if max_x * max_y == 0:
        semi_field = max(max_x, max_y) / math.sqrt(2.0)
        if semi_field == 0:
            semi_field = 1.0
        return semi_field, semi_field
    return max_x, max_y


def calculate_grid_at_rotation(
    cv, rotation_deg: float, grid_lines: int, fov_x: float, fov_y: float,
    z_mm: float | None = None,
) -> list[dict[str, float]]:
    """Mirror DIST's field setup, central reference, and RAYRSI grid tracing."""
    # Match raytracing.py's explicit surface-label selector for RC.
    commands = [f"ade {S_RC} {rotation_deg:.8f}"]
    if z_mm is not None:
        commands.append(f"thi {S_Z_MOV} {float(z_mm):.8f}")
    command = ";".join(commands + ["set vig"])
    response = cv.Command(command)
    actual_rotation = cv_eval(cv, "ade", S_RC)
    if "Command End:" not in response or any(
        line.lstrip().lower().startswith("error:") for line in response.splitlines()
    ):
        raise RuntimeError(
            f"CODE V reported an error for RC rotation {rotation_deg:g}°: {response!r}"
        )
    if not math.isclose(actual_rotation, rotation_deg, rel_tol=0.0, abs_tol=1e-7):
        raise RuntimeError(
            f"CODE V did not apply requested RC rotation {rotation_deg:g}° "
            f"(read back {actual_rotation:g}°; command response: {response!r})"
        )
    if z_mm is not None:
        actual_z = cv_eval(cv, "thi", S_Z_MOV)
        if not math.isclose(actual_z, float(z_mm), rel_tol=0.0, abs_tol=1e-8):
            raise RuntimeError(
                f"CODE V did not apply requested cornea movement {z_mm:g} mm "
                f"(read back {actual_z:g} mm; command response: {response!r})"
            )
        print(f"  Cornea movement readback verified: {actual_z:g} mm", flush=True)
    print(f"  RC readback verified: {actual_rotation:g}°", flush=True)
    # Restore the original angular field specification before each rotation;
    # the DIST reference calculation below changes the active specification to XOB/YOB.
    cv.Command(f"xan 0 {fov_x:.12g};yan 0 {fov_y:.12g}")

    # DIST switches to equivalent object-height fields for its reference grid.
    xob_f2 = cv_eval(cv, "xob", "f2", "z1")
    yob_f2 = cv_eval(cv, "yob", "f2", "z1")
    cv.Command(f"yob 0 {yob_f2:.12g};xob 0 {xob_f2:.12g}")

    center = trace_rsi(cv, 0.0, 0.0, field_number=1)
    if center is None:
        raise RuntimeError(f"Center chief ray failed at RC rotation {rotation_deg:g}°")
    near_axis = trace_rsi(cv, 0.05, 0.05)
    if near_axis is None:
        raise RuntimeError(f"Central 5% reference ray failed at RC rotation {rotation_deg:g}°")
    x_corner = near_axis[0] - center[0]
    y_corner = near_axis[1] - center[1]
    delta_x = 40.0 * x_corner / (grid_lines - 1)
    delta_y = 40.0 * y_corner / (grid_lines - 1)

    rows: list[dict[str, float]] = []
    for i in range(grid_lines):
        relative_y = 1.0 - 2.0 * i / (grid_lines - 1)
        paraxial_y = ((grid_lines - 1) / 2.0 - i) * delta_y
        for j in range(grid_lines):
            relative_x = -1.0 + 2.0 * j / (grid_lines - 1)
            paraxial_x = (j - (grid_lines - 1) / 2.0) * delta_x
            real = trace_rsi(cv, relative_x, relative_y)
            if real is None:
                if i in (0, grid_lines - 1) or j in (0, grid_lines - 1):
                    raise RuntimeError(
                        f"Edge chief ray failed at RC={rotation_deg:g}°, "
                        f"relative field=({relative_x:g}, {relative_y:g})"
                    )
                real_x = real_y = radial = tangential = math.nan
            else:
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

            rows.append(
                {
                    "field_x_relative": relative_x,
                    "field_y_relative": relative_y,
                    "paraxial_x_mm": paraxial_x,
                    "paraxial_y_mm": paraxial_y,
                    "real_x_mm": real_x,
                    "real_y_mm": real_y,
                    "radial_distortion_pct": radial,
                    "tangential_distortion_pct": tangential,
                    "eye_rotation_deg": float(rotation_deg),
                    "z_mm": float(z_mm) if z_mm is not None else math.nan,
                }
            )
    return rows


def write_pickle(rows: list[dict[str, float]], path: Path) -> None:
    columns = [
        "eye_rotation_deg",
        "z_mm",
        "field_x_relative",
        "field_y_relative",
        "paraxial_x_mm",
        "paraxial_y_mm",
        "real_x_mm",
        "real_y_mm",
        "radial_distortion_pct",
        "tangential_distortion_pct",
    ]
    dtype = [(column, np.float64) for column in columns]
    data = np.array(
        [tuple(row[column] for column in columns) for row in rows],
        dtype=dtype,
    )
    with path.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)


def atomic_pickle(value, path: Path) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp_path, path)


def save_rows_checkpoint(rows: list[dict[str, float]], path: Path, config: dict) -> None:
    atomic_pickle({"config": config, "rows": rows}, path)


def calculate_distortion_grid(
    rotations: Sequence[float], grid_lines: int, output_dir: Path,
    z_values: Sequence[float | None] | None = None, resume: bool = False,
) -> None:
    if not SEQ_FILE.is_file():
        raise FileNotFoundError(f"Target sequence not found: {SEQ_FILE}")
    if grid_lines < 3 or grid_lines > 21:
        raise ValueError("Grid line count must be between 3 and 21.")

    output_dir.mkdir(parents=True, exist_ok=True)
    if z_values is None:
        z_values = [None]
    config = {
        "rotations": [float(value) for value in rotations],
        "z_values_mm": [None if value is None else float(value) for value in z_values],
        "grid_lines": int(grid_lines),
        "lens_sha256": hashlib.sha256(SEQ_FILE.read_bytes()).hexdigest(),
    }
    checkpoint_path = output_dir / "distortion_grid.checkpoint.pkl"
    if resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"No P1 checkpoint to resume: {checkpoint_path}")
        with checkpoint_path.open("rb") as stream:
            checkpoint = pickle.load(stream)
        if checkpoint.get("config") != config:
            raise ValueError("P1 checkpoint settings do not match the requested sweep.")
        all_rows = list(checkpoint["rows"])
    else:
        all_rows = []
        if checkpoint_path.exists():
            checkpoint_path.unlink()
    completed = {
        (None if math.isnan(row.get("z_mm", math.nan)) else row["z_mm"], row["eye_rotation_deg"])
        for row in all_rows
    }
    # Match raytracing.py's COM apartment reset before creating CODE V.
    pythoncom.CoUninitialize()
    pythoncom.CoInitialize()
    cv = None
    codev_started = False
    try:
        print("Connecting to CODE V COM...", flush=True)
        cv = win32com.client.Dispatch("CodeV.Application")
        cv.CommandTimeout = 600000
        cv.MaxTextBufferSize = 1000000
        cv.StartingDirectory = str(LENS_DIR)
        cv.StartCodeV()
        codev_started = True

        print(f"Loading {SEQ_FILE}...", flush=True)
        cv.Command(f'run "{SEQ_FILE}";go')
        fov_x, fov_y = angular_fov(cv)
        print(f"Angular semi-field: X={fov_x:g}°, Y={fov_y:g}°", flush=True)

        for z_index, z_value in enumerate(z_values, start=1):
            z_mm = None if z_value is None else float(z_value)
            key_z = None if z_value is None else float(z_value)
            if all((key_z, float(angle)) in completed for angle in rotations):
                continue
            z_rows: list[dict[str, float]] = []
            for index, angle in enumerate(rotations, start=1):
                print(
                    f"z {z_value if z_value is not None else 'baseline'} mm "
                    f"({z_index}/{len(z_values)}), "
                    f"RC rotation {angle:g}° ({index}/{len(rotations)})", flush=True
                )
                grid_rows = calculate_grid_at_rotation(
                    cv, float(angle), grid_lines, fov_x, fov_y, z_mm=z_mm
                )
                z_rows.extend(grid_rows)
            all_rows.extend(z_rows)
            completed.update((key_z, float(angle)) for angle in rotations)
            save_rows_checkpoint(all_rows, checkpoint_path, config)

        canonical = output_dir / "distortion_grid.pkl"
        tmp_path = canonical.with_name(canonical.name + ".tmp")
        write_pickle(all_rows, tmp_path)
        os.replace(tmp_path, canonical)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
    finally:
        try:
            if cv is not None and codev_started:
                cv.StopCodeV()
        finally:
            pythoncom.CoUninitialize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rotation-min", type=float, default=EYE_ROTATION_MIN_DEG)
    parser.add_argument("--rotation-max", type=float, default=EYE_ROTATION_MAX_DEG)
    parser.add_argument("--rotation-count", type=int, default=EYE_ROTATION_COUNT)
    parser.add_argument("--z-min", type=float, default=Z_MIN_MM)
    parser.add_argument("--z-max", type=float, default=Z_MAX_MM)
    parser.add_argument("--z-count", type=int, default=Z_COUNT)
    parser.add_argument("--grid-lines", type=int, default=GRID_LINES)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.rotation_count < 1:
        parser.error("--rotation-count must be at least 1")
    if args.z_count < 1:
        parser.error("--z-count must be at least 1")

    if args.rotation_count == 1:
        rotations = [args.rotation_min]
    else:
        step = (args.rotation_max - args.rotation_min) / (args.rotation_count - 1)
        rotations = [args.rotation_min + i * step for i in range(args.rotation_count)]
    z_values = np.linspace(args.z_min, args.z_max, args.z_count).tolist()
    calculate_distortion_grid(rotations, args.grid_lines, args.output_dir, z_values, args.resume)
    print(f"Saved distortion pickle under: {(args.output_dir / 'distortion_grid.pkl').resolve()}")


if __name__ == "__main__":
    main()
