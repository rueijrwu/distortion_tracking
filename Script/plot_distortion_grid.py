"""Plot selected distortion grids from a saved pickle without starting CODE V."""

from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_INPUT = PROJECT_DIR / "data" / "distortion_grid" / "distortion_grid.pkl"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "distortion_grid" / "distortion_grid.png"
DEFAULT_ROTATIONS = (-10.0, -5.0, 0.0, 5.0, 10.0)
REQUIRED_FIELDS = (
    "eye_rotation_deg",
    "field_x_relative",
    "field_y_relative",
    "paraxial_x_mm",
    "paraxial_y_mm",
    "real_x_mm",
    "real_y_mm",
    "radial_distortion_pct",
    "tangential_distortion_pct",
)


def validate_grid_rows(rows: np.ndarray, rotation_deg: float) -> int:
    point_count = len(rows)
    grid_lines = math.isqrt(point_count)
    if grid_lines < 3 or grid_lines * grid_lines != point_count:
        raise ValueError(
            f"RC={rotation_deg:g} has {point_count} records; expected a square grid."
        )

    expected_x = np.tile(np.linspace(-1.0, 1.0, grid_lines), grid_lines)
    expected_y = np.repeat(np.linspace(1.0, -1.0, grid_lines), grid_lines)
    if not np.allclose(rows["field_x_relative"], expected_x, rtol=0.0, atol=1e-8):
        raise ValueError(f"RC={rotation_deg:g} grid X fields are incomplete or unordered.")
    if not np.allclose(rows["field_y_relative"], expected_y, rtol=0.0, atol=1e-8):
        raise ValueError(f"RC={rotation_deg:g} grid Y fields are incomplete or unordered.")
    return grid_lines


def load_selected_grids(path: Path, requested_rotations: Sequence[float]):
    if not path.is_file():
        raise FileNotFoundError(f"Distortion pickle not found: {path}")
    with path.open("rb") as stream:
        data = pickle.load(stream)

    if not isinstance(data, np.ndarray) or data.ndim != 1 or data.dtype.names is None:
        raise ValueError("Pickle must contain a one-dimensional structured NumPy array.")
    missing = [field for field in REQUIRED_FIELDS if field not in data.dtype.names]
    if missing:
        raise ValueError(f"Pickle is missing required fields: {', '.join(missing)}")
    available = np.unique(data["eye_rotation_deg"])
    grids = []
    for requested in requested_rotations:
        mask = np.isclose(data["eye_rotation_deg"], requested, rtol=0.0, atol=1e-8)
        rows = data[mask]
        if rows.size == 0:
            available_text = ", ".join(f"{value:g}" for value in available)
            raise ValueError(
                f"Requested RC={requested:g} is absent from {path}. "
                f"Available rotations: {available_text or '(none)'}."
            )
        matched_angles = np.unique(rows["eye_rotation_deg"])
        if matched_angles.size != 1:
            raise ValueError(f"RC={requested:g} ambiguously matches multiple stored rotations.")
        validate_grid_rows(rows, requested)
        grids.append((float(requested), rows))
    return grids


def plot_combined_grids(
    grids: Sequence[tuple[float, np.ndarray]], output_path: Path
) -> None:
    finite_points = [
        (float(row[x]), float(row[y]))
        for _, rows in grids
        for x, y in (("paraxial_x_mm", "paraxial_y_mm"), ("real_x_mm", "real_y_mm"))
        for row in rows
        if math.isfinite(float(row[x])) and math.isfinite(float(row[y]))
    ]
    if not finite_points:
        raise RuntimeError("No finite image coordinates in the selected rotations.")
    x_min, x_max = min(x for x, _ in finite_points), max(x for x, _ in finite_points)
    y_min, y_max = min(y for _, y in finite_points), max(y for _, y in finite_points)
    x_pad = max((x_max - x_min) * 0.04, 1e-6)
    y_pad = max((y_max - y_min) * 0.04, 1e-6)

    fig, axes_grid = plt.subplots(
        1,
        len(grids),
        figsize=(5.2 * len(grids), 5.8),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes_grid[0]
    for ax, (rotation, rows) in zip(axes, grids):
        grid_lines = math.isqrt(len(rows))
        px = rows["paraxial_x_mm"].reshape(grid_lines, grid_lines)
        py = rows["paraxial_y_mm"].reshape(grid_lines, grid_lines)
        rx = rows["real_x_mm"].reshape(grid_lines, grid_lines)
        ry = rows["real_y_mm"].reshape(grid_lines, grid_lines)
        for i in range(grid_lines):
            ax.plot(px[i], py[i], color="0.55", linestyle="--", linewidth=0.8,
                    label="Reference" if i == 0 else None)
            ax.plot(px[:, i], py[:, i], color="0.55", linestyle="--", linewidth=0.8)
            ax.plot(rx[i], ry[i], color="#1769aa", linewidth=1.1,
                    label="Real image" if i == 0 else None)
            ax.plot(rx[:, i], ry[:, i], color="#1769aa", linewidth=1.1)
        valid = np.isfinite(px) & np.isfinite(py) & np.isfinite(rx) & np.isfinite(ry)
        if np.any(valid):
            ax.quiver(
                px[valid], py[valid], rx[valid] - px[valid], ry[valid] - py[valid],
                angles="xy", scale_units="xy", scale=1, color="#c43b36",
                alpha=0.55, width=0.0025, label="Distortion vector",
            )
        ax.set_title(f"RC {rotation:+g}°")
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="0.9", linewidth=0.5)
        ax.set_xlabel("Image X (mm)")
    axes[0].set_ylabel("Image Y (mm)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955),
               ncol=3, frameon=False)
    fig.suptitle("Distortion grids by RC eye rotation", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--rotations",
        type=float,
        nargs="+",
        default=DEFAULT_ROTATIONS,
        metavar="DEG",
        help="RC rotations to plot (default: -10 -5 0 5 10)",
    )
    args = parser.parse_args()
    if not args.rotations:
        parser.error("--rotations must include at least one angle")

    grids = load_selected_grids(args.input, args.rotations)
    plot_combined_grids(grids, args.output)
    print(f"Saved {len(grids)} distortion grids to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
