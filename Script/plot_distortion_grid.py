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
DEFAULT_Z_VALUES = (-5.0, -2.6, 0.0, 2.6, 5.0)
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


def load_selected_grids(
    path: Path, requested_rotations: Sequence[float], z_mm: float | None = None
):
    if not path.is_file():
        raise FileNotFoundError(f"Distortion pickle not found: {path}")
    with path.open("rb") as stream:
        data = pickle.load(stream)

    if not isinstance(data, np.ndarray) or data.ndim != 1 or data.dtype.names is None:
        raise ValueError("Pickle must contain a one-dimensional structured NumPy array.")
    missing = [field for field in REQUIRED_FIELDS if field not in data.dtype.names]
    if missing:
        raise ValueError(f"Pickle is missing required fields: {', '.join(missing)}")
    if "z_mm" in data.dtype.names:
        available_z = np.unique(data["z_mm"])
        selected_z = float(available_z[np.argmin(np.abs(available_z))]) if z_mm is None else z_mm
        z_mask = np.isclose(data["z_mm"], selected_z, rtol=0.0, atol=1e-8)
        if not np.any(z_mask):
            available_text = ", ".join(f"{value:g}" for value in available_z)
            raise ValueError(
                f"Requested z={selected_z:g} mm is absent from {path}. "
                f"Available z values: {available_text or '(none)'}."
            )
        data = data[z_mask]
    elif z_mm is not None:
        raise ValueError(f"Requested z={z_mm:g} mm, but this pickle has no z_mm field.")
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


def load_rotation_z_grids(path: Path, requested_rotations: Sequence[float], requested_z: Sequence[float]):
    """Load exact stored image grids for selected rotation and z values."""
    if not path.is_file():
        raise FileNotFoundError(f"Distortion pickle not found: {path}")
    with path.open("rb") as stream:
        data = pickle.load(stream)
    if not isinstance(data, np.ndarray) or data.ndim != 1 or data.dtype.names is None:
        raise ValueError("Pickle must contain a one-dimensional structured NumPy array.")
    missing = [field for field in REQUIRED_FIELDS if field not in data.dtype.names]
    if missing or "z_mm" not in data.dtype.names:
        detail = f" Missing fields: {', '.join(missing)}." if missing else ""
        raise ValueError("P1 comparison requires structured fields including z_mm." + detail)

    available_z = np.unique(data["z_mm"])
    selected_z = []
    for z in requested_z:
        exact = available_z[np.isclose(available_z, z, rtol=0.0, atol=1e-8)]
        if exact.size != 1:
            available_text = ", ".join(f"{value:g}" for value in available_z)
            raise ValueError(
                f"Requested z={z:g} mm is not an exact stored plane. "
                f"Available z values: {available_text}."
            )
        selected_z.append(float(exact[0]))
    available_rotations = np.unique(data["eye_rotation_deg"])
    result = []
    for requested in requested_rotations:
        matches = available_rotations[np.isclose(available_rotations, requested, rtol=0.0, atol=1e-8)]
        if matches.size != 1:
            raise ValueError(f"Requested rotation {requested:g}° is not an exact stored angle.")
        rotation = float(matches[0])
        grids = []
        for z in selected_z:
            rows = data[
                np.isclose(data["eye_rotation_deg"], rotation, rtol=0.0, atol=1e-8)
                & np.isclose(data["z_mm"], z, rtol=0.0, atol=1e-8)
            ]
            validate_grid_rows(rows, rotation)
            grids.append((z, rows))
        result.append((rotation, grids))
    return result


def plot_combined_grids(rotation_grids, output_path: Path) -> None:
    finite_points = [
        (float(row[x]), float(row[y]))
        for _, grids in rotation_grids
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
        len(rotation_grids),
        figsize=(5.2 * len(rotation_grids), 5.8),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes_grid[0]
    z_values = [z for z, _ in rotation_grids[0][1]]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    z_colors = {z: colors[i % len(colors)] for i, z in enumerate(z_values)}
    for ax, (rotation, grids) in zip(axes, rotation_grids):
        for z_value, rows in grids:
            grid_lines = math.isqrt(len(rows))
            px = rows["paraxial_x_mm"].reshape(grid_lines, grid_lines)
            py = rows["paraxial_y_mm"].reshape(grid_lines, grid_lines)
            rx = rows["real_x_mm"].reshape(grid_lines, grid_lines)
            ry = rows["real_y_mm"].reshape(grid_lines, grid_lines)
            color = z_colors[z_value]
            for i in range(grid_lines):
                ax.plot(px[i], py[i], color=color, linestyle="--", linewidth=0.8, alpha=0.8)
                ax.plot(px[:, i], py[:, i], color=color, linestyle="--", linewidth=0.8, alpha=0.8)
                ax.plot(rx[i], ry[i], color=color, linewidth=1.05)
                ax.plot(rx[:, i], ry[:, i], color=color, linewidth=1.05)
            valid = np.isfinite(px) & np.isfinite(py) & np.isfinite(rx) & np.isfinite(ry)
            if np.any(valid):
                ax.quiver(
                    px[valid], py[valid], rx[valid] - px[valid], ry[valid] - py[valid],
                    angles="xy", scale_units="xy", scale=1, color=color,
                    alpha=0.32, width=0.0022,
                )
        ax.set_title(f"RC {rotation:+g}°")
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="0.9", linewidth=0.5)
        ax.set_xlabel("Image X (mm)")
    axes[0].set_ylabel("Image Y (mm)")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=z_colors[z], lw=1.4, label=f"Z {z:+g} mm") for z in z_values]
    handles.extend([
        Line2D([0], [0], color="0.2", lw=1.2, linestyle="--", label="Paraxial reference"),
        Line2D([0], [0], color="0.2", lw=1.2, linestyle="-", label="Real image"),
    ])
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.96),
               ncol=7, frameon=False)
    fig.suptitle("P1 distortion grids by eye rotation", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rotations", type=float, nargs="+", default=DEFAULT_ROTATIONS,
                        metavar="DEG", help="exact stored RC rotations (default: -10 -5 0 5 10)")
    parser.add_argument(
        "--z-values", type=float, nargs="+", default=DEFAULT_Z_VALUES, metavar="MM",
        help="z planes to compare (default: -5 -2.5 0 2.5 5 mm)",
    )
    args = parser.parse_args()
    if not args.z_values or not args.rotations:
        parser.error("--z-values and --rotations must each include at least one value")

    rotation_grids = load_rotation_z_grids(args.input, args.rotations, args.z_values)
    plot_combined_grids(rotation_grids, args.output)
    print(f"Saved {len(rotation_grids)} rotation panels across {len(args.z_values)} z planes to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
