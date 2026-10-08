"""Measure distortion changes at each eye rotation relative to RC=0."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_INPUT = PROJECT_DIR / "data" / "distortion_grid" / "distortion_grid.pkl"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "distortion_grid"
BASELINE_DEG = 0.0
ANGLE_FIELD = "eye_rotation_deg"
X_FIELD = "field_x_relative"
Y_FIELD = "field_y_relative"
VALUE_FIELDS = ("radial_distortion_pct", "tangential_distortion_pct")
REQUIRED_FIELDS = (ANGLE_FIELD, X_FIELD, Y_FIELD, *VALUE_FIELDS)


def load_data(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Distortion pickle not found: {path}")
    with path.open("rb") as stream:
        data = pickle.load(stream)
    if not isinstance(data, np.ndarray) or data.ndim != 1 or data.dtype.names is None:
        raise ValueError("Input must contain a one-dimensional structured NumPy array.")
    missing = [field for field in REQUIRED_FIELDS if field not in data.dtype.names]
    if missing:
        raise ValueError(f"Input is missing fields: {', '.join(missing)}")
    if not np.all(np.isfinite(data[ANGLE_FIELD])):
        raise ValueError("Eye rotations contain non-finite values.")
    return data


def compare_to_baseline(data: np.ndarray) -> np.ndarray:
    angles = np.unique(data[ANGLE_FIELD])
    baseline_angles = angles[angles == BASELINE_DEG]
    if baseline_angles.size != 1:
        raise ValueError("Input must contain exactly one stored 0 degree rotation.")

    baseline = data[data[ANGLE_FIELD] == BASELINE_DEG]
    keys = np.rec.fromarrays((baseline[X_FIELD], baseline[Y_FIELD]), names="x,y")
    if np.unique(keys).size != baseline.size:
        raise ValueError("The 0 degree grid has duplicate relative field points.")
    baseline_order = np.lexsort((baseline[Y_FIELD], baseline[X_FIELD]))
    baseline = baseline[baseline_order]
    baseline_keys = np.column_stack((baseline[X_FIELD], baseline[Y_FIELD]))

    output_dtype = [
        (ANGLE_FIELD, np.float64),
        (X_FIELD, np.float64),
        (Y_FIELD, np.float64),
        ("radial_delta_percentage_points", np.float64),
        ("tangential_delta_percentage_points", np.float64),
    ]
    output = []
    for angle in angles:
        rows = data[data[ANGLE_FIELD] == angle]
        if rows.size != baseline.size:
            raise ValueError(
                f"RC={angle:g} has {rows.size} points; expected {baseline.size}."
            )
        order = np.lexsort((rows[Y_FIELD], rows[X_FIELD]))
        rows = rows[order]
        row_keys = np.column_stack((rows[X_FIELD], rows[Y_FIELD]))
        if not np.array_equal(row_keys, baseline_keys):
            raise ValueError(f"RC={angle:g} field points do not match the 0 degree grid.")
        radial = rows[VALUE_FIELDS[0]] - baseline[VALUE_FIELDS[0]]
        tangential = rows[VALUE_FIELDS[1]] - baseline[VALUE_FIELDS[1]]
        output.append(
            np.array(
                list(zip(
                    np.full(rows.size, angle), rows[X_FIELD], rows[Y_FIELD], radial, tangential
                )),
                dtype=output_dtype,
            )
        )
    return np.concatenate(output)


def summarize(comparison: np.ndarray) -> np.ndarray:
    summary_dtype = [
        (ANGLE_FIELD, np.float64),
        ("radial_max_abs_delta_percentage_points", np.float64),
        ("radial_rms_delta_percentage_points", np.float64),
        ("tangential_max_abs_delta_percentage_points", np.float64),
        ("tangential_rms_delta_percentage_points", np.float64),
    ]
    records = []
    for angle in np.unique(comparison[ANGLE_FIELD]):
        rows = comparison[comparison[ANGLE_FIELD] == angle]
        values = []
        for field in ("radial_delta_percentage_points", "tangential_delta_percentage_points"):
            delta = rows[field]
            finite = delta[np.isfinite(delta)]
            if finite.size == 0:
                raise ValueError(f"RC={angle:g} has no finite {field} comparisons.")
            values.extend((np.max(np.abs(finite)), np.sqrt(np.mean(finite**2))))
        records.append((angle, *values))
    return np.array(records, dtype=summary_dtype)


def save_pickle(value: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)


def plot_summary(summary: np.ndarray, path: Path) -> None:
    angles = summary[ANGLE_FIELD]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    for ax, component, title in zip(axes, ("radial", "tangential"), ("Radial", "Tangential")):
        maximum = summary[f"{component}_max_abs_delta_percentage_points"]
        rms = summary[f"{component}_rms_delta_percentage_points"]
        ax.plot(angles, maximum, label="Maximum absolute change", linewidth=1.5)
        ax.plot(angles, rms, label="RMS change", linewidth=1.5)
        ax.axhline(0.0, color="0.5", linewidth=0.7)
        ax.axvline(BASELINE_DEG, color="0.5", linestyle=":", linewidth=0.8)
        ax.set_title(f"{title} distortion")
        ax.set_xlabel("RC eye rotation (degrees)")
        ax.set_ylabel("Change from RC 0 (percentage points)")
        ax.grid(True, color="0.9", linewidth=0.6)
        ax.legend(frameon=False)
    fig.suptitle("Distortion change relative to the same field point at RC 0°")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    data = load_data(args.input)
    comparison = compare_to_baseline(data)
    summary = summarize(comparison)
    output_path = args.output_dir / "distortion_change_from_zero.pkl"
    plot_path = args.output_dir / "distortion_change_from_zero.png"
    save_pickle({"pointwise_changes": comparison, "per_angle_summary": summary}, output_path)
    plot_summary(summary, plot_path)
    print(f"Saved pointwise changes and summary: {output_path.resolve()}")
    print(f"Saved summary chart: {plot_path.resolve()}")
    for row in (summary[0], summary[-1]):
        print(
            f"RC={row[ANGLE_FIELD]:+g}: radial max/RMS="
            f"{row['radial_max_abs_delta_percentage_points']:.6g}/"
            f"{row['radial_rms_delta_percentage_points']:.6g} pp; tangential max/RMS="
            f"{row['tangential_max_abs_delta_percentage_points']:.6g}/"
            f"{row['tangential_rms_delta_percentage_points']:.6g} pp"
        )


if __name__ == "__main__":
    main()
