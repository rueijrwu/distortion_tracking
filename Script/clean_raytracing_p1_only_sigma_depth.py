"""Mask robust jumps in every plotted sigma curve in a separate pickle."""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
RAW_INPUT = ROOT / "data" / "p1_only_sigma_depth_raytracing" / "p1_only_sigma_depth_raytracing.pkl"
CLEAN_OUTPUT = ROOT / "data" / "p1_only_sigma_depth_raytracing" / "p1_only_sigma_depth_raytracing_cleaned.pkl"
TRANSLATIONS = (-1.0, 0.0, 1.0)
ZOOMS = (1, 2, 3)
EXPECTED_METADATA = {
    1: (0.0, 1.0e13, 50.0, -15.0, 0.0),
    2: (0.0, 0.0, 50.0, 0.0, 5.0),
    3: (0.0, 0.0, 25.0, 0.0, -5.0),
}
METADATA_FIELDS = (
    "field_yob_mm", "object_thi_s0_mm", "s1_thickness_mm", "s1_ade_deg", "s1_yde_mm",
)


def find_jumps(values: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    differences = np.abs(np.diff(values))
    median = float(np.median(differences))
    mad = float(np.median(np.abs(differences - median)))
    cutoff = max(1.0, median + 10.0 * 1.4826 * mad)
    return np.flatnonzero(differences > cutoff), median, mad, cutoff


def main() -> None:
    with RAW_INPUT.open("rb") as stream:
        raw = pickle.load(stream)
    required = {"h_translation_mm", "rotation_deg", "zoom_number", "real_y_si_mm", "trace_success",
                *METADATA_FIELDS}
    if not isinstance(raw, np.ndarray) or raw.ndim != 1 or raw.dtype.names is None:
        raise ValueError("Raw pickle must contain a one-dimensional structured NumPy array")
    missing = required.difference(raw.dtype.names)
    if missing:
        raise ValueError(f"Raw pickle is missing fields: {', '.join(sorted(missing))}")
    if not np.all(raw["trace_success"]):
        raise ValueError("Raw pickle contains failed traces")
    if not np.isfinite(raw["real_y_si_mm"]).all():
        raise ValueError("Raw SI Y values must all be finite before cleaning")
    for zoom, expected_values in EXPECTED_METADATA.items():
        rows = raw["zoom_number"] == zoom
        for field, expected_value in zip(METADATA_FIELDS, expected_values):
            actual = np.unique(raw[field][rows])
            if not np.array_equal(actual, np.array([expected_value])):
                raise ValueError(
                    f"Unexpected current-sequence metadata {field} for z{zoom}: "
                    f"{actual}; expected {expected_value}"
                )

    # Build aligned, rotation-sorted raw arrays and row-index arrays. All flags
    # below are calculated from these raw finite values before any masking.
    rows_by_key: dict[tuple[float, int], np.ndarray] = {}
    y_by_key: dict[tuple[float, int], np.ndarray] = {}
    rotation_ref: np.ndarray | None = None
    for h in TRANSLATIONS:
        for zoom in ZOOMS:
            indices = np.flatnonzero(
                np.isclose(raw["h_translation_mm"], h, rtol=0.0, atol=1e-8)
                & (raw["zoom_number"] == zoom)
            )
            order = np.argsort(raw["rotation_deg"][indices])
            indices = indices[order]
            angles = raw["rotation_deg"][indices]
            if np.unique(angles).size != angles.size:
                raise ValueError(f"Duplicate rotation samples for h={h:g}, z{zoom}")
            if rotation_ref is None:
                rotation_ref = angles
            elif not np.array_equal(angles, rotation_ref):
                raise ValueError(f"Rotation samples differ for h={h:g}, z{zoom}")
            rows_by_key[(h, zoom)] = indices
            y_by_key[(h, zoom)] = raw["real_y_si_mm"][indices].copy()

    assert rotation_ref is not None
    masked_rows: set[int] = set()
    total_curve_jumps = 0

    def scan_curve(label: str, values: np.ndarray,
                   sources: tuple[tuple[float, int], ...]) -> None:
        nonlocal total_curve_jumps
        jumps, median, mad, cutoff = find_jumps(values)
        total_curve_jumps += int(jumps.size)
        for i in jumps:
            # Mask both angle endpoints in every source curve contributing to
            # this plotted curve, including h=0 contributors to baseline shifts.
            for endpoint in (int(i), int(i + 1)):
                for key in sources:
                    masked_rows.add(int(rows_by_key[key][endpoint]))
        intervals = [(float(rotation_ref[i]), float(rotation_ref[i + 1])) for i in jumps]
        print(
            f"{label}: median={median:.6g}, MAD={mad:.6g}, cutoff={cutoff:.6g} mm, "
            f"flagged={len(jumps)}, intervals={intervals}", flush=True
        )

    cross_by_h: dict[float, np.ndarray] = {}
    for h in TRANSLATIONS:
        y1 = y_by_key[(h, 1)]
        y2 = y_by_key[(h, 2)]
        y3 = y_by_key[(h, 3)]
        scan_curve(f"absolute h={h:+g} z1", y1, ((h, 1),))
        scan_curve(f"absolute h={h:+g} z2", y2, ((h, 2),))
        scan_curve(f"absolute h={h:+g} z3", y3, ((h, 3),))
        scan_curve(f"z2-z1 h={h:+g}", y2 - y1, ((h, 2), (h, 1)))
        scan_curve(f"z3-z1 h={h:+g}", y3 - y1, ((h, 3), (h, 1)))
        cross_by_h[h] = y2 - y3
        scan_curve(f"z2-z3 h={h:+g}", cross_by_h[h], ((h, 2), (h, 3)))

    for h in (-1.0, 1.0):
        baseline_delta = cross_by_h[h] - cross_by_h[0.0]
        scan_curve(
            f"baseline delta h={h:+g}", baseline_delta,
            ((h, 2), (h, 3), (0.0, 2), (0.0, 3)),
        )

    # Panels 3 and 4 are mostly level curves. Adjacent-step detection breaks
    # branch transitions but can leave interior samples on a shifted branch.
    # Detect those samples against each raw derived curve's own median/MAD.
    adjacent_mask_count = len(masked_rows)
    level_outlier_counts: dict[str, int] = {}

    def mask_level_outliers(label: str, values: np.ndarray,
                            sources: tuple[tuple[float, int], ...]) -> None:
        median = float(np.median(values))
        scale = 1.4826 * float(np.median(np.abs(values - median)))
        cutoff = 10.0 * scale
        flagged = np.flatnonzero(np.abs(values - median) > cutoff)
        level_outlier_counts[label] = int(flagged.size)
        for angle_index in flagged:
            for key in sources:
                masked_rows.add(int(rows_by_key[key][int(angle_index)]))
        angles = [float(rotation_ref[i]) for i in flagged]
        print(
            f"level outliers {label}: median={median:.6g}, cutoff={cutoff:.6g} mm, "
            f"flagged samples={len(flagged)}, angles={angles}",
            flush=True,
        )

    for h in TRANSLATIONS:
        mask_level_outliers(f"z2-z3 h={h:+g}", cross_by_h[h], ((h, 2), (h, 3)))
    for h in (-1.0, 1.0):
        mask_level_outliers(
            f"baseline delta h={h:+g}", cross_by_h[h] - cross_by_h[0.0],
            ((h, 2), (h, 3), (0.0, 2), (0.0, 3)),
        )

    cleaned = raw.copy()
    masked_array = np.fromiter(sorted(masked_rows), dtype=np.int64)
    cleaned["real_y_si_mm"][masked_array] = np.nan
    if np.isinf(cleaned["real_y_si_mm"]).any():
        raise ValueError("Cleaning produced infinite SI Y values")
    CLEAN_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = CLEAN_OUTPUT.with_name(CLEAN_OUTPUT.name + ".tmp")
    with tmp.open("wb") as stream:
        pickle.dump(cleaned, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, CLEAN_OUTPUT)
    print(
        f"Saved {len(cleaned)} rows to {CLEAN_OUTPUT}; flagged curve jumps="
        f"{total_curve_jumps}; unique contributing source rows masked={len(masked_rows)}; "
        f"additional source rows from level masks={len(masked_rows) - adjacent_mask_count}; "
        f"level outlier samples={sum(level_outlier_counts.values())}; "
        f"NaN rows={int(np.isnan(cleaned['real_y_si_mm']).sum())}",
        flush=True,
    )


if __name__ == "__main__":
    main()
