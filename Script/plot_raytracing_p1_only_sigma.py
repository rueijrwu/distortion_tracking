"""Plot absolute P1-only sigma image Y and zoom differences across RC pose."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "p1_only_sigma_raytracing" / "p1_only_sigma_raytracing_cleaned.pkl"
DEFAULT_DIR = DEFAULT_INPUT.parent
ZOOM_LABELS = {1: "z1 Inf (YOB 0)", 2: "z2 fin+ (YOB +5)", 3: "z3 fin- (YOB -5)"}
TRANSLATIONS = (-1.0, 0.0, 1.0)
TRANSLATION_STYLES = {-1.0: "--", 0.0: "-", 1.0: ":"}


def load_data(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        data = pickle.load(stream)
    required = {"h_translation_mm", "rotation_deg", "zoom_number", "field_yob_mm",
                "real_y_si_mm", "trace_success"}
    if not isinstance(data, np.ndarray) or data.ndim != 1 or data.dtype.names is None:
        raise ValueError("Input pickle must contain a one-dimensional structured NumPy array")
    missing = required.difference(data.dtype.names)
    if missing:
        raise ValueError(f"Input pickle is missing fields: {', '.join(sorted(missing))}")
    if not np.all(data["trace_success"]):
        raise ValueError("Input contains failed rays; refusing to plot incomplete curves")
    if (not np.isfinite(data["rotation_deg"]).all()
            or np.isinf(data["real_y_si_mm"]).any()):
        raise ValueError("Input contains nonfinite rotations or infinite SI Y values")
    if set(np.unique(data["zoom_number"]).tolist()) != set(ZOOM_LABELS):
        raise ValueError("Expected zoom numbers 1, 2, and 3")
    expected_yobs = {1: 0.0, 2: 5.0, 3: -5.0}
    for zoom, expected_yob in expected_yobs.items():
        actual_yobs = np.unique(data["field_yob_mm"][data["zoom_number"] == zoom])
        if not np.array_equal(actual_yobs, np.array([expected_yob])):
            raise ValueError(f"Unexpected F1 YOB metadata for z{zoom}: {actual_yobs}")
    return data


def render_figure(data: np.ndarray, output_dir: Path,
                  rotation_window: tuple[float, float], output_name: str) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 10.5))
    ax, diff_ax, cross_ax, baseline_ax = axes.flat
    colors = {1: "#1f77b4", 2: "#d95f02", 3: "#2b8a3e"}
    baseline_color = "#6f42c1"
    reference_angles = None
    cross_by_h = {}
    h_handles = [Line2D([0], [0], color="#444444", lw=1.8,
                        linestyle=TRANSLATION_STYLES[h], label=f"h = {h:+g} mm")
                 for h in TRANSLATIONS]

    for requested_h in TRANSLATIONS:
        matched_h = np.unique(data["h_translation_mm"][np.isclose(
            data["h_translation_mm"], requested_h, rtol=0.0, atol=1e-8
        )])
        if matched_h.size != 1:
            available = ", ".join(f"{value:g}" for value in np.unique(data["h_translation_mm"]))
            raise ValueError(f"Expected one h translation at {requested_h:g} mm; found {available}")
        h = float(matched_h[0])
        per_zoom = {}
        for zoom in ZOOM_LABELS:
            rows = data[(np.isclose(data["h_translation_mm"], h, rtol=0.0, atol=1e-8))
                        & (data["zoom_number"] == zoom)]
            angles = np.unique(rows["rotation_deg"])
            if rows.size != angles.size:
                raise ValueError(f"Duplicate rows for h={h:g} mm, z{zoom}")
            if reference_angles is None:
                reference_angles = angles
            if not np.array_equal(angles, reference_angles):
                raise ValueError(f"Rotation samples differ for h={h:g} mm, z{zoom}")
            order = np.argsort(rows["rotation_deg"])
            sorted_angles = rows["rotation_deg"][order]
            values = rows["real_y_si_mm"][order]
            in_window = ((sorted_angles >= rotation_window[0])
                         & (sorted_angles <= rotation_window[1]))
            sorted_angles = sorted_angles[in_window]
            values = values[in_window]
            per_zoom[zoom] = (sorted_angles, values)
            ax.plot(*per_zoom[zoom], color=colors[zoom],
                    linestyle=TRANSLATION_STYLES[requested_h], linewidth=1.55)

        base_angles, base_y = per_zoom[1]
        for zoom in (2, 3):
            angles, values = per_zoom[zoom]
            if not np.array_equal(angles, base_angles):
                raise ValueError(f"Zoom z{zoom} rotations do not align with z1 at h={h:g} mm")
            diff_ax.plot(angles, values - base_y, color=colors[zoom],
                         linestyle=TRANSLATION_STYLES[requested_h], linewidth=1.55)

        z2_angles, z2_y = per_zoom[2]
        z3_angles, z3_y = per_zoom[3]
        if not np.array_equal(z2_angles, z3_angles):
            raise ValueError(f"z2 and z3 rotations do not align at h={h:g} mm")
        cross = z2_y - z3_y
        cross_ax.plot(z2_angles, cross, color=colors[2],
                      linestyle=TRANSLATION_STYLES[requested_h], linewidth=1.55)
        cross_by_h[requested_h] = (z2_angles, cross)

    zero_h_angles, zero_h_cross = cross_by_h[0.0]
    for requested_h in TRANSLATIONS:
        angles, cross = cross_by_h[requested_h]
        if not np.array_equal(angles, zero_h_angles):
            raise ValueError(f"Rotation samples do not align with h=0 for h={requested_h:g} mm")
        baseline_ax.plot(angles, cross - zero_h_cross, color=baseline_color,
                         linestyle=TRANSLATION_STYLES[requested_h], linewidth=1.55)

    for current_ax in (ax, diff_ax, cross_ax, baseline_ax):
        current_ax.set_xlim(*rotation_window)

    ax.axhline(0.0, color="#777777", linewidth=0.8, alpha=0.65)
    ax.axvline(0.0, color="#aaaaaa", linewidth=0.7, alpha=0.6)
    ax.set_xlabel("RC rotation (deg)")
    ax.set_ylabel("Real-ray Y at image surface SI (mm)")
    ax.set_title("Absolute SI real-ray Y (P1-only sigma)")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    zoom_handles = [Line2D([0], [0], color=colors[z], lw=2, label=label)
                    for z, label in ZOOM_LABELS.items()]
    zoom_legend = ax.legend(handles=zoom_handles, title="Zoom", fontsize=8,
                            title_fontsize=8, loc="upper left", frameon=True)
    ax.add_artist(zoom_legend)

    diff_ax.axhline(0.0, color="#777777", linewidth=0.8, alpha=0.65)
    diff_ax.axvline(0.0, color="#aaaaaa", linewidth=0.7, alpha=0.6)
    diff_ax.set_xlabel("RC rotation (deg)")
    diff_ax.set_ylabel("SI real-ray Y difference from z1 (mm)")
    diff_ax.set_title("Finite-conjugate shifts relative to z1")
    diff_ax.grid(True, color="#dddddd", linewidth=0.7)
    diff_handles = [Line2D([0], [0], color=colors[z], lw=2,
                           label=f"z{z} - z1 ({'fin+' if z == 2 else 'fin-'} - Inf)")
                    for z in (2, 3)]
    diff_legend = diff_ax.legend(handles=diff_handles, title="Difference", fontsize=8,
                                 title_fontsize=8, loc="upper left", frameon=True)
    diff_ax.add_artist(diff_legend)

    for current_ax in (cross_ax, baseline_ax):
        current_ax.axvline(0.0, color="#aaaaaa", linewidth=0.7, alpha=0.6)
        current_ax.set_xlabel("RC rotation (deg)")
        current_ax.grid(True, color="#dddddd", linewidth=0.7)
    cross_ax.set_ylabel("SI real-ray Y difference (mm)")
    cross_ax.set_title("z2 - z3")
    baseline_ax.axhline(0.0, color="#777777", linewidth=0.8, alpha=0.65)
    baseline_ax.set_ylabel("Change in z2 - z3 from h = 0 (mm)")
    baseline_ax.set_title("z2 - z3 shift from zero head translation")

    output_dir.mkdir(parents=True, exist_ok=True)
    svg = output_dir / output_name
    fig.tight_layout(rect=(0.0, 0.11, 1.0, 1.0))
    fig.legend(handles=h_handles, title="RC Y translation (line style)",
               loc="lower center", ncol=3, fontsize=8, title_fontsize=8, frameon=True)
    fig.savefig(svg, facecolor="white")
    plt.close(fig)
    return svg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()

    data = load_data(args.input)
    outputs = (
        ((-20.0, 20.0), "p1_only_sigma_real_y_vs_rotation.svg"),
        ((-1.0, 1.0), "p1_only_sigma_real_y_vs_rotation_pm1deg.svg"),
    )
    for window, filename in outputs:
        svg = render_figure(data, args.output_dir, window, filename)
        sample_count = np.unique(data["rotation_deg"][(data["rotation_deg"] >= window[0])
                                                        & (data["rotation_deg"] <= window[1])]).size
        print(
            f"Plotted 4 panels from {len(TRANSLATIONS)} h slices and "
            f"{sample_count} rotations each ({window[0]:g}..{window[1]:g} deg), from {args.input}"
        )
        print(f"SVG: {svg}")


if __name__ == "__main__":
    main()
