"""Plot signed absolute image-surface Y from the P1-only ray sweep."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "p1_only_raytracing" / "p1_only_raytracing.pkl"
DEFAULT_DIR = DEFAULT_INPUT.parent
ZOOM_LABELS = {1: "z1 Inf (YOB 0)", 2: "z2 fin+ (YOB +10)", 3: "z3 fin− (YOB −10)"}
TRANSLATIONS = (-1.0, 0.0, 1.0)
TRANSLATION_STYLES = {-1.0: "--", 0.0: "-", 1.0: ":"}


def load_data(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        data = pickle.load(stream)
    required = {"h_translation_mm", "rotation_deg", "zoom_number", "real_y_si_mm", "trace_success"}
    if not isinstance(data, np.ndarray) or data.ndim != 1 or data.dtype.names is None:
        raise ValueError("Input pickle must contain a one-dimensional structured NumPy array")
    missing = required.difference(data.dtype.names)
    if missing:
        raise ValueError(f"Input pickle is missing fields: {', '.join(sorted(missing))}")
    if not np.all(data["trace_success"]):
        raise ValueError("Input contains failed rays; refusing to plot incomplete curves")
    if not np.isfinite(data["rotation_deg"]).all() or not np.isfinite(data["real_y_si_mm"]).all():
        raise ValueError("Input contains nonfinite rotation or SI Y values")
    if set(np.unique(data["zoom_number"]).tolist()) != set(ZOOM_LABELS):
        raise ValueError("Expected zoom numbers 1, 2, and 3")
    angles = np.unique(data["rotation_deg"])
    if angles.size < 2:
        raise ValueError("At least two rotation samples are required")
    for zoom in ZOOM_LABELS:
        zdata = data[data["zoom_number"] == zoom]
        if not np.array_equal(np.unique(zdata["rotation_deg"]), angles):
            raise ValueError(f"Zoom z{zoom} does not contain the same rotation samples")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()

    data = load_data(args.input)
    fig, (ax, diff_ax, cross_ax) = plt.subplots(1, 3, figsize=(17.2, 5.8))
    colors = {1: "#1f77b4", 2: "#d95f02", 3: "#2b8a3e"}
    slices = {}
    reference_angles = None
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
            per_zoom[zoom] = (rows["rotation_deg"][order], rows["real_y_si_mm"][order])
            ax.plot(*per_zoom[zoom], color=colors[zoom], linestyle=TRANSLATION_STYLES[requested_h],
                    linewidth=1.55, alpha=0.95)
        slices[requested_h] = per_zoom

        base_angles, base_y = per_zoom[1]
        for zoom in (2, 3):
            angles, values = per_zoom[zoom]
            if not np.array_equal(angles, base_angles):
                raise ValueError(f"Zoom z{zoom} rotations do not align with z1 at h={h:g} mm")
            diff_ax.plot(angles, values - base_y, color=colors[zoom],
                         linestyle=TRANSLATION_STYLES[requested_h], linewidth=1.55, alpha=0.95)
        z2_angles, z2_y = per_zoom[2]
        z3_angles, z3_y = per_zoom[3]
        if not np.array_equal(z2_angles, z3_angles):
            raise ValueError(f"z2 and z3 rotations do not align at h={h:g} mm")
        cross_ax.plot(z2_angles, z2_y - z3_y, color=colors[2],
                      linestyle=TRANSLATION_STYLES[requested_h], linewidth=1.55, alpha=0.95)
    ax.axhline(0.0, color="#777777", linewidth=0.8, alpha=0.65)
    ax.axvline(0.0, color="#aaaaaa", linewidth=0.7, alpha=0.6)
    ax.set_xlabel("RC rotation (deg)")
    ax.set_ylabel("Real-ray Y at image surface SI (mm)")
    ax.set_title("Absolute SI real-ray Y")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    from matplotlib.lines import Line2D
    zoom_handles = [Line2D([0], [0], color=colors[z], lw=2, label=label)
                    for z, label in ZOOM_LABELS.items()]
    h_handles = [Line2D([0], [0], color="#444444", lw=1.8, linestyle=TRANSLATION_STYLES[h],
                        label=f"{h:+g} mm") for h in TRANSLATIONS]
    zoom_legend = ax.legend(handles=zoom_handles, title="Zoom", fontsize=8, title_fontsize=8,
                            loc="upper left", frameon=True)
    ax.add_artist(zoom_legend)
    ax.legend(handles=h_handles, title="RC Y translation h", fontsize=8,
              title_fontsize=8, loc="lower right", frameon=True)
    diff_ax.axhline(0.0, color="#777777", linewidth=0.8, alpha=0.65)
    diff_ax.axvline(0.0, color="#aaaaaa", linewidth=0.7, alpha=0.6)
    diff_ax.set_xlabel("RC rotation (deg)")
    diff_ax.set_ylabel("SI real-ray Y difference from z1 (mm)")
    diff_ax.set_title("Finite-conjugate shift relative to z1")
    diff_ax.grid(True, color="#dddddd", linewidth=0.7)
    diff_handles = [Line2D([0], [0], color=colors[z], lw=2,
                           label=f"z{z} − z1 ({'fin+' if z == 2 else 'fin−'} − Inf)")
                    for z in (2, 3)]
    diff_legend = diff_ax.legend(handles=diff_handles, title="Difference", fontsize=8,
                                 title_fontsize=8, loc="upper left", frameon=True)
    diff_ax.add_artist(diff_legend)
    diff_ax.legend(handles=h_handles, title="RC Y translation h", fontsize=8,
                   title_fontsize=8, loc="lower right", frameon=True)
    cross_ax.axvline(0.0, color="#aaaaaa", linewidth=0.7, alpha=0.6)
    cross_ax.set_xlabel("RC rotation (deg)")
    cross_ax.set_ylabel("SI real-ray Y difference (mm)")
    cross_ax.set_title("z2 − z3")
    cross_ax.grid(True, color="#dddddd", linewidth=0.7)
    cross_handles = [Line2D([0], [0], color=colors[2], lw=2,
                            label="z2 fin+ − z3 fin−")]
    cross_legend = cross_ax.legend(handles=cross_handles, title="Difference", fontsize=8,
                                   title_fontsize=8, loc="upper left", frameon=True)
    cross_ax.add_artist(cross_legend)
    cross_ax.legend(handles=h_handles, title="RC Y translation h", fontsize=8,
                    title_fontsize=8, loc="lower right", frameon=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    svg = args.output_dir / "p1_only_real_y_vs_rotation.svg"
    fig.tight_layout()
    fig.savefig(svg, facecolor="white")
    plt.close(fig)
    print(f"Plotted {len(TRANSLATIONS)} translation slices, {len(reference_angles)} rotations each, from {args.input}")
    print(f"SVG: {svg}")


if __name__ == "__main__":
    main()
