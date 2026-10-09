"""Measure P4 image magnification versus absolute Cornea_ENT_D thickness.

The collector samples five accommodation settings, five RC rotations, and 51
absolute THI values. CODE V is used only while collecting; subsequent runs
reuse the saved structured pickle and regenerate the fit/report/figure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pythoncom
import win32com.client

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
LENS_FILE = PROJECT_DIR / "Lens" / "p4_ME.len"
OUTPUT_DIR = PROJECT_DIR / "data" / "p4_z_magnification"
sys.path.insert(0, str(SCRIPT_DIR))
import distortion_grid as grid  # noqa: E402
import distortion_grid_p4 as p4  # noqa: E402

ACCOMMODATIONS_D = (0.0, 1.0, 2.0, 3.0, 4.0)
ROTATIONS_DEG = (-10.0, -5.0, 0.0, 5.0, 10.0)
Z_DISTANCE_MM = tuple(float(v) for v in np.linspace(-5.0, 5.0, 51))
GRID_LINES = 3
FIELDS = (
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
    "z_distance_mm",
    "z_thickness_mm",
)
CHECKPOINT_VERSION = 1


def model_hash() -> str:
    return hashlib.sha256(LENS_FILE.read_bytes()).hexdigest()


def config_for(accommodations: Sequence[float], rotations: Sequence[float],
               z_values: Sequence[float], grid_lines: int) -> dict:
    return {
        "model": str(LENS_FILE.relative_to(PROJECT_DIR)),
        "model_sha256": model_hash(),
        "zoom": "z1 (P4)",
        "z_selector": 's"Cornea_ENT_D"',
        "z_distance_mm": [float(v) for v in z_values],
        "z_thickness_mm": [float(v) for v in z_values],
        "accommodation_d": [float(v) for v in accommodations],
        "rotation_deg": [float(v) for v in rotations],
        "grid_lines": int(grid_lines),
        "fields": list(FIELDS),
    }


def structured_group(rows: list[dict[str, float]], accommodation: float,
                     z_distance: float) -> np.ndarray:
    dtype = [(name, np.float64) for name in FIELDS]
    values = []
    for row in rows:
        values.append((
            float(row["eye_rotation_deg"]), float(accommodation),
            float(row["field_x_relative"]), float(row["field_y_relative"]),
            float(row["paraxial_x_mm"]), float(row["paraxial_y_mm"]),
            float(row["real_x_mm"]), float(row["real_y_mm"]),
            float(row["radial_distortion_pct"]),
            float(row["tangential_distortion_pct"]), float(z_distance), float(z_distance),
        ))
    return np.asarray(values, dtype=dtype)


def atomic_pickle(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(value: dict, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate_data(data: np.ndarray, config: dict, *, complete: bool) -> None:
    if not isinstance(data, np.ndarray) or data.ndim != 1 or data.dtype.names != FIELDS:
        raise ValueError("Saved P4 Z magnification data has an unexpected structured dtype")
    if not np.isfinite(np.column_stack([data[name] for name in FIELDS])).all():
        raise ValueError("Saved P4 Z magnification data contains nonfinite values")
    per_group = len(config["rotation_deg"]) * config["grid_lines"] ** 2
    expected = (len(config["accommodation_d"]) * len(config["z_distance_mm"])
                * per_group)
    if complete and data.size != expected:
        raise ValueError(f"Complete dataset has {data.size} rows, expected {expected}")
    if not complete and (data.size > expected or data.size % per_group != 0):
        raise ValueError("Partial dataset row count does not contain complete grid groups")
    for name, values in (("accommodation_d", config["accommodation_d"]),
                         ("eye_rotation_deg", config["rotation_deg"]),
                         ("z_distance_mm", config["z_distance_mm"]),
                         ("z_thickness_mm", config["z_thickness_mm"])):
        observed = np.unique(data[name])
        if complete:
            valid = np.allclose(observed, np.sort(values), rtol=0, atol=1e-8)
        else:
            valid = all(any(math.isclose(float(value), float(expected_value), rel_tol=0.0, abs_tol=1e-8)
                            for expected_value in values) for value in observed)
        if not valid:
            raise ValueError(f"Saved {name} values do not match the requested configuration")


def collect(accommodations: Sequence[float], rotations: Sequence[float],
            z_values: Sequence[float], grid_lines: int, output_dir: Path,
            resume: bool) -> tuple[np.ndarray, dict]:
    if grid_lines != 3:
        raise ValueError("This magnification study requires the established 3 x 3 field grid")
    if not LENS_FILE.is_file():
        raise FileNotFoundError(LENS_FILE)
    config = config_for(accommodations, rotations, z_values, grid_lines)
    checkpoint_path = output_dir / "p4_z_magnification.checkpoint.pkl"
    final_path = output_dir / "p4_z_magnification.pkl"
    completed: list[tuple[float, float]] = []
    accumulated: list[np.ndarray] = []
    loaded_baselines: dict[str, dict[str, float]] = {}
    geometry_targets: dict[str, dict[str, float]] = {}
    if resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"No checkpoint to resume: {checkpoint_path}")
        with checkpoint_path.open("rb") as stream:
            checkpoint = pickle.load(stream)
        if checkpoint.get("version") != CHECKPOINT_VERSION or checkpoint.get("config") != config:
            raise ValueError("Checkpoint model hash or sweep settings do not match")
        data = checkpoint["data"]
        completed = [(float(a), float(z)) for a, z in checkpoint["completed_groups"]]
        loaded_baselines = checkpoint.get("loaded_z1_geometry_baselines", {})
        geometry_targets = checkpoint.get("accommodation_geometry_targets", {})
        per_group = len(rotations) * grid_lines**2
        if data.size != len(completed) * per_group:
            raise ValueError("Checkpoint row count does not match completed groups")
        expected_order = [(float(a), float(z)) for a in accommodations for z in z_values]
        if completed != expected_order[:len(completed)]:
            raise ValueError("Checkpoint is not a prefix of the requested accommodation/Z sweep")
        if data.size:
            validate_data(data, config, complete=False)
            accumulated.append(data)
        print(f"Resuming after {len(completed)} of {len(expected_order)} accommodation/Z groups.", flush=True)
    elif checkpoint_path.exists():
        raise FileExistsError(f"Checkpoint exists at {checkpoint_path}; pass --resume")

    pythoncom.CoUninitialize()
    pythoncom.CoInitialize()
    cv = None
    started = False
    try:
        cv = win32com.client.Dispatch("CodeV.Application")
        cv.CommandTimeout = 600000
        cv.MaxTextBufferSize = 1000000
        cv.StartingDirectory = str(LENS_FILE.parent)
        cv.StartCodeV()
        started = True
        per_group = len(rotations) * grid_lines**2
        group_index = len(completed)
        for accommodation in accommodations:
            a_key = f"{float(accommodation):g}"
            print(f"Loading P4 z1 for accommodation {accommodation:g} D...", flush=True)
            p4.load_lens(cv)
            baseline = p4.read_accommodation_baseline(cv)
            baseline["cornea_ent_d_thickness_mm"] = grid.cv_eval(
                cv, "thi", "z1", 's"Cornea_ENT_D"'
            )
            if a_key not in loaded_baselines:
                loaded_baselines[a_key] = baseline
            geometry = p4.apply_accommodation(cv, baseline, float(accommodation))
            geometry_targets[a_key] = geometry
            fov_x, fov_y = grid.angular_fov(cv)
            for z_value in z_values:
                group_key = (float(accommodation), float(z_value))
                if group_index < len(completed) and completed[group_index] == group_key:
                    group_index += 1
                    continue
                print(f"  z1 accommodation={accommodation:g} D, physical Z/absolute THI={z_value:+g} mm", flush=True)
                group_rows: list[dict[str, float]] = []
                for rotation in rotations:
                    rows = grid.calculate_grid_at_rotation(
                        cv, float(rotation), grid_lines, fov_x, fov_y,
                        z_mm=float(z_value),
                    )
                    group_rows.extend(rows)
                group_data = structured_group(group_rows, float(accommodation), float(z_value))
                if group_data.size != per_group or not np.isfinite(
                    np.column_stack([group_data[name] for name in FIELDS])
                ).all():
                    raise RuntimeError(f"Incomplete/nonfinite group at A={accommodation:g}, Z={z_value:g}")
                accumulated.append(group_data)
                completed.append(group_key)
                combined = np.concatenate(accumulated)
                checkpoint = {
                    "version": CHECKPOINT_VERSION,
                    "config": config,
                    "completed_groups": completed,
                    "loaded_z1_geometry_baselines": loaded_baselines,
                    "accommodation_geometry_targets": geometry_targets,
                    "data": combined,
                }
                atomic_pickle(checkpoint, checkpoint_path)
                group_index += 1
                print(f"    verified ADE on all {len(rotations)} rotations; checkpoint rows={combined.size:,}", flush=True)
        data = np.concatenate(accumulated) if accumulated else np.empty(0, dtype=[(n, np.float64) for n in FIELDS])
        validate_data(data, config, complete=True)
        final = {
            "data": data,
            "config": config,
            "loaded_z1_geometry_baselines": loaded_baselines,
            "accommodation_geometry_targets": geometry_targets,
            "completed_groups": [[float(a), float(z)] for a, z in completed],
            "records": int(data.size),
            "state_count": int(len(accommodations) * len(rotations) * len(z_values)),
        }
        atomic_pickle(data, final_path)
        metadata = {k: v for k, v in final.items() if k != "data"}
        metadata.update({
            "loaded_z1_THI_baseline_mm": loaded_baselines.get("0", {}).get("cornea_ent_d_thickness_mm", 0.0),
            "thickness_command_rule": "THI = z_distance_mm (same sign); absolute value reset for every state",
            "finite_numeric_records": int(sum(np.isfinite(data[name]).sum() for name in FIELDS)),
        })
        atomic_json(metadata, output_dir / "metadata.json")
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        return data, metadata
    finally:
        try:
            if cv is not None and started:
                cv.StopCodeV()
        finally:
            pythoncom.CoUninitialize()


def _point_coordinates(group: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.lexsort((group["field_x_relative"], group["field_y_relative"]))
    fields = np.column_stack((group["field_x_relative"][order], group["field_y_relative"][order]))
    coords = np.column_stack((group["real_x_mm"][order], group["real_y_mm"][order]))
    if np.unique(fields, axis=0).shape[0] != 9:
        raise ValueError("Each fitted grid must contain nine distinct field identifiers")
    return fields, coords


def analyze(data: np.ndarray, output_dir: Path) -> dict:
    accommodations = np.unique(data["accommodation_d"])
    rotations = np.unique(data["eye_rotation_deg"])
    z_values = np.unique(data["z_distance_mm"])
    zero_z = int(np.argmin(np.abs(z_values)))
    if abs(z_values[zero_z]) > 1e-9:
        raise ValueError("Study needs an actual physical Z=0 plane")
    zero_theta = int(np.argmin(np.abs(rotations)))
    if abs(rotations[zero_theta]) > 1e-9:
        raise ValueError("Study needs an actual theta=0 grid")
    ratios = np.empty((len(accommodations), len(rotations), len(z_values)), dtype=float)
    scaling_residuals: list[float] = []
    scaling_maxima: list[float] = []
    linear_residuals: list[float] = []
    linear_maxima: list[float] = []
    for ai, accommodation in enumerate(accommodations):
        for ti, rotation in enumerate(rotations):
            grids = []
            for zi, z_value in enumerate(z_values):
                mask = ((data["accommodation_d"] == accommodation)
                        & (data["eye_rotation_deg"] == rotation)
                        & (data["z_distance_mm"] == z_value))
                group = data[mask]
                if group.size != 9:
                    raise ValueError(f"Expected nine rows for A={accommodation:g}, theta={rotation:g}, Z={z_value:g}")
                fields, coords = _point_coordinates(group)
                if zi == 0:
                    field_ids = fields
                elif not np.array_equal(fields, field_ids):
                    raise ValueError("Field identifiers differ between Z planes")
                grids.append(coords)
            baseline = grids[zero_z]
            denom = float(np.sum(baseline * baseline))
            if not math.isfinite(denom) or denom <= 1e-20:
                raise ValueError("Physical Z=0 grid has no magnification leverage")
            for zi, coords in enumerate(grids):
                ratios[ai, ti, zi] = float(np.sum(baseline * coords) / denom)
            if not math.isclose(float(ratios[ai, ti, zero_z]), 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(
                    f"Z=0 magnification invariant failed at A={accommodation:g} D, "
                    f"theta={rotation:g}: m={ratios[ai, ti, zero_z]:.16g}"
                )
            # Actual scalar similarity residual uses the directly measured factor per angle.
            direct_prediction = ratios[ai, ti, :, None, None] * baseline[None, :, :]
            direct_errors = np.linalg.norm(direct_prediction - np.stack(grids), axis=2)
            scaling_residuals.extend(direct_errors.ravel().tolist())
            scaling_maxima.extend(direct_errors.max(axis=1).tolist())
    alpha = np.empty(len(accommodations), dtype=float)
    linear_rmse, linear_max = np.empty(len(accommodations)), np.empty(len(accommodations))
    ratio_rmse, ratio_max = np.empty(len(accommodations)), np.empty(len(accommodations))
    independence_max = np.empty(len(accommodations), dtype=float)
    ratio_by_theta_rmse = np.empty(len(accommodations), dtype=float)
    linear_worst_state_rmse = np.empty(len(accommodations), dtype=float)
    linear_max_euclidean_point = np.empty(len(accommodations), dtype=float)
    direct_max_euclidean_point = np.empty(len(accommodations), dtype=float)
    measured_endpoints = np.empty((len(accommodations), 2), dtype=float)
    for ai, accommodation in enumerate(accommodations):
        measured = ratios[ai, zero_theta]
        alpha[ai] = float(np.dot(z_values, measured - 1.0) / np.dot(z_values, z_values))
        common_linear = 1.0 + alpha[ai] * z_values
        theta_delta = np.abs(ratios[ai] - measured[None, :]) / np.maximum(np.abs(measured[None, :]), 1e-12) * 100.0
        independence_max[ai] = float(theta_delta.max())
        ratio_by_theta_rmse[ai] = float(np.sqrt(np.mean((ratios[ai] - measured[None, :]) ** 2)))
        direct_errors, line_errors = [], []
        linear_state_rmse, linear_point_max, direct_point_max = [], [], []
        for ti, rotation in enumerate(rotations):
            baselines = []
            actuals = []
            for z_value in z_values:
                subset = data[(data["accommodation_d"] == accommodation)
                              & (data["eye_rotation_deg"] == rotation)
                              & (data["z_distance_mm"] == z_value)]
                _, coords = _point_coordinates(subset)
                actuals.append(coords)
                if z_value == 0:
                    base = coords
            for zi, coords in enumerate(actuals):
                direct_error = ratios[ai, ti, zi] * base - coords
                line_error = common_linear[zi] * base - coords
                direct_errors.extend(direct_error.ravel().tolist())
                line_errors.extend(line_error.ravel().tolist())
                direct_point_max.append(float(np.max(np.linalg.norm(direct_error, axis=1))))
                linear_point_max.append(float(np.max(np.linalg.norm(line_error, axis=1))))
                linear_state_rmse.append(float(np.sqrt(np.mean(line_error**2))))
        direct_errors = np.asarray(direct_errors)
        line_errors = np.asarray(line_errors)
        ratio_rmse[ai] = float(np.sqrt(np.mean(direct_errors**2)))
        ratio_max[ai] = float(np.max(np.abs(direct_errors)))
        linear_rmse[ai] = float(np.sqrt(np.mean(line_errors**2)))
        linear_max[ai] = float(np.max(np.abs(line_errors)))
        linear_worst_state_rmse[ai] = max(linear_state_rmse)
        linear_max_euclidean_point[ai] = max(linear_point_max)
        direct_max_euclidean_point[ai] = max(direct_point_max)
        measured_endpoints[ai] = [
            measured[int(np.argmin(np.abs(z_values - endpoint)))]
            for endpoint in (-5.0, 5.0)
        ]

    factors = 1.0 + alpha[:, None] * np.asarray([-5.0, -2.6, 0.0, 2.6, 5.0])[None, :]
    samples = np.asarray([-5.0, -2.6, 0.0, 2.6, 5.0])
    sample_rows = [[float(a), float(s), float(factors[i, j])] for i, a in enumerate(accommodations) for j, s in enumerate(samples)]
    slopes_mean = float(np.mean(alpha))
    slope_span_pct = float((np.max(alpha) - np.min(alpha)) / max(abs(slopes_mean), 1e-15) * 100.0)
    direct_rmse = float(np.sqrt(np.mean(np.asarray(scaling_residuals) ** 2)))
    direct_max = float(np.max(scaling_maxima))

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(accommodations)))
    z_fine = np.linspace(float(z_values.min()), float(z_values.max()), 301)
    for ai, (accommodation, color) in enumerate(zip(accommodations, colors)):
        axes[0].plot(z_values, ratios[ai, zero_theta], "o", color=color, ms=3.5,
                     label=f"{accommodation:g} D measured")
        axes[0].plot(z_fine, 1.0 + alpha[ai] * z_fine, color=color, lw=1.4,
                     label=f"{accommodation:g} D linear")
    axes[0].set_xlabel("Physical Z / absolute Cornea_ENT_D THI (mm)")
    axes[0].set_ylabel("Magnification ratio, m")
    axes[0].set_title("P4 z1 at 0° RC rotation")
    axes[0].grid(True, alpha=0.28)
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].plot(accommodations, alpha, "o-", color="tab:purple")
    axes[1].set_xlabel("Accommodation (D)")
    axes[1].set_ylabel("Linear slope α (mm⁻¹)")
    axes[1].set_title(f"Slope varies {slope_span_pct:.3g}% across sampled accommodation")
    axes[1].grid(True, alpha=0.28)
    figure_path = output_dir / "p4_z_magnification.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    report = {
        "accommodation_d": accommodations.tolist(),
        "rotation_deg": rotations.tolist(),
        "z_distance_mm": z_values.tolist(),
        "magnification_ratio": ratios,
        "alpha_per_accommodation_per_mm": alpha,
        "ratio_theta_independence_max_relative_percent": independence_max,
        "ratio_theta_independence_rmse": ratio_by_theta_rmse,
        "direct_factor_scaling_coordinate_rmse_mm": ratio_rmse,
        "direct_factor_scaling_coordinate_max_abs_mm": ratio_max,
        "theta0_linear_coordinate_rmse_mm": linear_rmse,
        "theta0_linear_coordinate_max_abs_mm": linear_max,
        "theta0_linear_worst_state_coordinate_rmse_mm": linear_worst_state_rmse,
        "theta0_linear_max_euclidean_point_mm": linear_max_euclidean_point,
        "direct_max_euclidean_point_mm": direct_max_euclidean_point,
        "measured_theta0_m_at_minus5_plus5_mm": measured_endpoints,
        "selected_linear_magnification_ratios": sample_rows,
        "alpha_range_percent_of_mean": slope_span_pct,
        "direct_similarity_grid_point_rmse_mm": direct_rmse,
        "direct_similarity_grid_point_max_abs_mm": direct_max,
    }
    with (output_dir / "p4_z_magnification_fit.pkl").open("wb") as stream:
        pickle.dump(report, stream, protocol=pickle.HIGHEST_PROTOCOL)
    lines = [
        "# P4 z1 linear magnification versus Z", "",
        "The collector uses `Lens/p4_ME.len`, zoom z1 (P4), the established 3 × 3 grid, accommodations 0–4 D in 1 D steps, RC rotations -10°, -5°, 0°, +5°, +10°, and 51 absolute THI values from -5 to +5 mm. `z_distance_mm` is the physical Z coordinate, and `z_thickness_mm` records the commanded absolute THI with the same sign. The study resets `Cornea_ENT_D` to each requested value; a fresh loaded LEN readback is recorded in `metadata.json` (the active z1 baseline readback is 0 mm).",
        "",
        "For each accommodation and rotation, the measured real-coordinate grid at Z=0 is the fixed source. At each Z, `m = dot(R0, RZ) / dot(R0, R0)` over both coordinates of all nine field points; this least-squares scalar is defined even for the center point. The linear model is `m = 1 + alpha*Z`, with intercept fixed at one, fit to the theta=0 ratios. The same slope curve is compared with all five actual rotation grids.", "",
        f"Collected {len(accommodations)*len(rotations)*len(z_values):,} states and {len(data):,} finite records. Maximum change in m relative to theta=0 across rotations: {independence_max.max():.6g}%. Direct measured-factor coordinate RMSE/max absolute coordinate residual are listed per accommodation below. Pooling Euclidean errors over every accommodation/rotation/Z/field gives direct-factor point RMSE/max {direct_rmse:.6g}/{direct_max:.6g} mm. Linear theta=0 factor prediction applied at all rotations has pooled coordinate RMSE/max absolute coordinate residual ranges {linear_rmse.min():.6g}–{linear_rmse.max():.6g} / {linear_max.min():.6g}–{linear_max.max():.6g} mm across accommodation. The per-accommodation slope span is {slope_span_pct:.6g}% of its mean.", "",
        "| Accommodation (D) | alpha (mm^-1) | ratio-angle max variation (%) | Direct factor RMSE (mm) | Direct factor max abs (mm) | Linear RMSE (mm) | Linear max abs (mm) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, a in enumerate(accommodations):
        lines.append(f"| {a:g} | {alpha[i]:.9g} | {independence_max[i]:.6g} | {ratio_rmse[i]:.6g} | {ratio_max[i]:.6g} | {linear_rmse[i]:.6g} | {linear_max[i]:.6g} |")
    lines.extend(["", "## Measured endpoints and worst single-grid errors", "", "Endpoint m values are measured theta=0 ratios. Worst-state coordinate RMSE is calculated over nine points and two coordinates in one theta/Z grid, then maximized over all rotations and Z planes. Euclidean point error is the norm across x/y for each grid point.", "", "| Accommodation (D) | Measured m(-5 mm) | Measured m(+5 mm) | Linear worst-state RMSE (mm) | Linear max Euclidean point (mm) | Direct max Euclidean point (mm) |", "|---:|---:|---:|---:|---:|---:|"])
    for i, a in enumerate(accommodations):
        lines.append(f"| {a:g} | {measured_endpoints[i,0]:.9g} | {measured_endpoints[i,1]:.9g} | {linear_worst_state_rmse[i]:.6g} | {linear_max_euclidean_point[i]:.6g} | {direct_max_euclidean_point[i]:.6g} |")
    lines.extend(["", "## Linear factors at selected physical Z", "", "| Accommodation (D) | Z (mm) | m = 1 + alpha Z |", "|---:|---:|---:|"])
    for a, z, factor in sample_rows:
        lines.append(f"| {a:g} | {z:+g} | {factor:.9g} |")
    lines.extend(["", "Center point is included in the grid residual but contributes zero to the scalar magnification fit because its baseline coordinate is zero. The directly fitted factors and the constrained linear approximation have separate residual columns above.", ""])
    (output_dir / "p4_z_magnification.md").write_text("\n".join(lines), encoding="utf-8")
    report["figure"] = str(figure_path)
    report["report"] = str(output_dir / "p4_z_magnification.md")
    return report


def parse_values(text: str | None, defaults: Sequence[float]) -> list[float]:
    if text is None:
        return [float(v) for v in defaults]
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def load_saved(path: Path, config: dict) -> tuple[np.ndarray, dict]:
    with path.open("rb") as stream:
        saved = pickle.load(stream)
    if isinstance(saved, np.ndarray):
        metadata_path = path.parent / "metadata.json"
        if not metadata_path.is_file():
            raise ValueError("Existing structured grid pickle has no metadata sidecar")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        data = saved
    elif isinstance(saved, dict) and isinstance(saved.get("data"), np.ndarray):
        # Upgrade an early dict-wrapped output from this script to the canonical
        # standalone structured NumPy array while preserving its JSON metadata.
        data = saved["data"]
        metadata = saved
        atomic_pickle(data, path)
    else:
        raise ValueError("Existing output is not a structured P4 NumPy grid")
    if metadata.get("config") != config:
        raise ValueError("Existing output does not match requested model hash and sweep settings")
    validate_data(data, config, complete=True)
    return data, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--accommodations", help="comma-separated D values; default 0,1,2,3,4")
    parser.add_argument("--rotations", help="comma-separated RC degrees; default -10,-5,0,5,10")
    parser.add_argument("--z-values-mm", help="comma-separated absolute THI mm; default 51 values -5..+5")
    parser.add_argument("--grid-lines", type=int, default=GRID_LINES)
    parser.add_argument("--resume", action="store_true", help="continue a matching atomic checkpoint")
    parser.add_argument("--collect", action="store_true", help="collect if the validated final pickle is absent")
    args = parser.parse_args()
    accommodations = parse_values(args.accommodations, ACCOMMODATIONS_D)
    rotations = parse_values(args.rotations, ROTATIONS_DEG)
    z_values = parse_values(args.z_values_mm, Z_DISTANCE_MM)
    config = config_for(accommodations, rotations, z_values, args.grid_lines)
    final_path = args.output_dir / "p4_z_magnification.pkl"
    if final_path.is_file() and not args.resume:
        data, saved = load_saved(final_path, config)
        metadata_path = args.output_dir / "metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("config") != config:
                raise ValueError("Existing metadata does not match requested sweep")
        print(f"Using validated cached P4 Z grid ({data.size:,} rows); CODE V will not be started.", flush=True)
    else:
        if final_path.exists() and args.resume:
            raise ValueError("A complete final dataset already exists; rerun without --resume")
        data, saved = collect(accommodations, rotations, z_values, args.grid_lines,
                              args.output_dir, args.resume)
    result = analyze(data, args.output_dir)
    print(f"States: {len(accommodations)*len(rotations)*len(z_values):,}; rows: {data.size:,}", flush=True)
    print("Accommodation slopes (mm^-1): " + ", ".join(
        f"{a:g} D={s:.9g}" for a, s in zip(result["accommodation_d"], result["alpha_per_accommodation_per_mm"])
    ), flush=True)
    print(f"Max rotation dependence: {np.max(result['ratio_theta_independence_max_relative_percent']):.6g}%", flush=True)
    print(f"Wrote {args.output_dir / 'p4_z_magnification.md'} and {args.output_dir / 'p4_z_magnification.png'}", flush=True)


if __name__ == "__main__":
    main()
