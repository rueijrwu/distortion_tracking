"""Plot P4 distortion grids across accommodation and RC rotation."""

from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plot_distortion_grid import validate_grid_rows


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_INPUT = PROJECT_DIR / "data" / "distortion_grid_p4" / "distortion_grid.pkl"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "distortion_grid_p4" / "distortion_grid.png"
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


def load_grids(path: Path, accommodations: list[float], rotations: list[float]):
    if not path.is_file():
        raise FileNotFoundError(f"P4 distortion pickle not found: {path}")
    with path.open("rb") as stream:
        data = pickle.load(stream)
    if not isinstance(data, np.ndarray) or data.ndim != 1 or data.dtype.names is None:
        raise ValueError("Pickle must contain a one-dimensional structured NumPy array.")
    missing = [name for name in GRID_FIELDS if name not in data.dtype.names]
    if missing:
        raise ValueError(f"P4 pickle is missing fields: {', '.join(missing)}")

    selected = []
    for accommodation in accommodations:
        available = np.unique(data["accommodation_d"])
        matched = available[
            np.isclose(available, accommodation, rtol=0.0, atol=1e-8)
        ]
        if matched.size != 1:
            available_text = ", ".join(f"{value:g}" for value in available)
            raise ValueError(
                f"Accommodation {accommodation:g} D is absent from {path}. "
                f"Available groups: {available_text or '(none)'}."
            )
        group = data[
            np.isclose(data["accommodation_d"], matched[0], rtol=0.0, atol=1e-8)
        ]
        grids = []
        for rotation in rotations:
            rows = group[
                np.isclose(
                    group["eye_rotation_deg"], rotation, rtol=0.0, atol=1e-8
                )
            ]
            if rows.size == 0:
                raise ValueError(
                    f"RC={rotation:g} is absent at accommodation {accommodation:g} D."
                )
            validate_grid_rows(rows, rotation)
            grids.append((float(rotation), rows))
        selected.append((float(accommodation), grids))
    return selected


def plot_accommodation_grids(
    accommodation_grids: list[tuple[float, list[tuple[float, np.ndarray]]]],
    output_path: Path,
) -> None:
    finite_points = [
        (float(row[x]), float(row[y]))
        for _, grids in accommodation_grids
        for _, rows in grids
        for x, y in (
            ("paraxial_x_mm", "paraxial_y_mm"),
            ("real_x_mm", "real_y_mm"),
        )
        for row in rows
        if math.isfinite(float(row[x])) and math.isfinite(float(row[y]))
    ]
    if not finite_points:
        raise RuntimeError("No finite P4 image coordinates in the selected grids.")
    x_min, x_max = min(x for x, _ in finite_points), max(x for x, _ in finite_points)
    y_min, y_max = min(y for _, y in finite_points), max(y for _, y in finite_points)
    x_pad = max((x_max - x_min) * 0.04, 1e-6)
    y_pad = max((y_max - y_min) * 0.04, 1e-6)

    ncols = len(accommodation_grids[0][1])
    fig, axes_grid = plt.subplots(
        1, ncols, figsize=(4.1 * ncols, 4.5), sharex=True, sharey=True, squeeze=False
    )
    axes = axes_grid[0]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = [colors[index % len(colors)] for index in range(len(accommodation_grids))]
    rotations = [rotation for rotation, _ in accommodation_grids[0][1]]
    for col_index, rotation in enumerate(rotations):
        ax = axes[col_index]
        for (_, grids), color in zip(accommodation_grids, colors):
            rows = grids[col_index][1]
            grid_lines = math.isqrt(len(rows))
            px = rows["paraxial_x_mm"].reshape(grid_lines, grid_lines)
            py = rows["paraxial_y_mm"].reshape(grid_lines, grid_lines)
            rx = rows["real_x_mm"].reshape(grid_lines, grid_lines)
            ry = rows["real_y_mm"].reshape(grid_lines, grid_lines)
            for i in range(grid_lines):
                ax.plot(px[i], py[i], color=color, linestyle="--", linewidth=0.85, alpha=0.42)
                ax.plot(px[:, i], py[:, i], color=color, linestyle="--", linewidth=0.85, alpha=0.42)
                ax.plot(rx[i], ry[i], color=color, linewidth=1.15)
                ax.plot(rx[:, i], ry[:, i], color=color, linewidth=1.15)
        ax.set_title(f"RC {rotation:+g}°")
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="0.9", linewidth=0.5)
        ax.set_xlabel("Image X (mm)")
    axes[0].set_ylabel("Image Y (mm)")
    handles = [
        Line2D([0], [0], color=color, linewidth=1.5, label=f"Accommodation {acc:g} D")
        for (acc, _), color in zip(accommodation_grids, colors)
    ]
    handles.extend([
        Line2D([0], [0], color="0.25", linestyle="--", linewidth=1.0, label="Reference grid"),
        Line2D([0], [0], color="0.25", linestyle="-", linewidth=1.0, label="Real image grid"),
    ])
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=len(handles), frameon=False)
    fig.suptitle("P4 distortion grids by accommodation", y=1.06)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--accommodations", type=float, nargs="+", default=[0.0, 2.0, 4.0], metavar="D",
        help="accommodation rows to plot (default: 0 2 4 D)",
    )
    parser.add_argument(
        "--accommodation", type=float, metavar="D",
        help="plot a single accommodation group",
    )
    parser.add_argument(
        "--rotations", type=float, nargs="+", default=[-10, -5, 0, 5, 10], metavar="DEG",
        help="RC columns to plot (default: -10 -5 0 5 10)",
    )
    args = parser.parse_args()
    accommodations = [args.accommodation] if args.accommodation is not None else args.accommodations
    if not accommodations or not args.rotations:
        parser.error("at least one accommodation and one RC rotation are required")
    selected = load_grids(args.input, accommodations, args.rotations)
    plot_accommodation_grids(selected, args.output)
    print(
        f"Saved {len(args.rotations)} rotations across {len(accommodations)} "
        f"P4 accommodation groups to: {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
