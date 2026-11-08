"""Validate the macro P4 biaxial collector against the slow per-state COM reference.

In ONE CODE V session, for accommodations {0, 4} D x ADE {-20, 0, 20} x BDE
{-20, 0, 20}: run distortion_grid.calculate_grid_at_biaxial_rotation state by
state (the slow path used by distortion_grid_p4_biaxial.py), then reload the
lens, reapply accommodation and run the macro path
(distortion_grid_p4_biaxial_macro.sweep_accommodation). All 14 GRID_FIELDS are
compared row by row. Work files go under data/validate_p4_biaxial_macro.work/
and are removed afterwards unless --keep-work.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pythoncom
import win32com.client

from distortion_grid import angular_fov, calculate_grid_at_biaxial_rotation
from distortion_grid_p4 import apply_accommodation, load_lens, read_accommodation_baseline
from distortion_grid_p4_biaxial import GRID_FIELDS, Z_MM, structured_rows
import distortion_grid_p4_biaxial_macro as macro

ACCOMMODATIONS = [0.0, 4.0]
ADE_VALUES = [-20.0, 0.0, 20.0]
BDE_VALUES = [-20.0, 0.0, 20.0]
WORK_DIR = macro.PROJECT_DIR / "data" / "validate_p4_biaxial_macro.work"

EXACT_FIELDS = (
    "eye_rotation_deg", "eye_rotation_x_deg", "accommodation_d", "z_mm",
    "field_x_relative", "field_y_relative",
)
COORD_FIELDS = (
    "paraxial_x_mm", "paraxial_y_mm", "center_absolute_x_mm", "center_absolute_y_mm",
    "real_x_mm", "real_y_mm",
)
PCT_FIELDS = ("radial_distortion_pct", "tangential_distortion_pct")
COORD_TOL_UM = 0.01
PCT_TOL = 1e-4  # percentage points


def slow_path(cv, accommodation_d: float) -> tuple[np.ndarray, float, float, float]:
    load_lens(cv)
    baseline = read_accommodation_baseline(cv)
    apply_accommodation(cv, baseline, accommodation_d)
    fov_x, fov_y = angular_fov(cv)
    rows: list[dict[str, float]] = []
    t0 = time.perf_counter()
    for ade in ADE_VALUES:
        for bde in BDE_VALUES:
            rows.extend(
                calculate_grid_at_biaxial_rotation(
                    cv, ade, bde, macro.GRID_LINES, fov_x, fov_y, z_mm=Z_MM
                )
            )
    elapsed = time.perf_counter() - t0
    return structured_rows(rows, accommodation_d), fov_x, fov_y, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args()
    if WORK_DIR.exists():
        macro.remove_tree(WORK_DIR)

    slow_parts, macro_parts = [], []
    pythoncom.CoUninitialize()
    pythoncom.CoInitialize()
    cv = None
    started = False
    try:
        cv = win32com.client.Dispatch("CodeV.Application")
        print(f"CommandTimeout = {macro.configure_session(cv)}", flush=True)
        cv.StartCodeV()
        started = True
        for ai, accommodation_d in enumerate(ACCOMMODATIONS):
            slow, fov_x, fov_y, slow_s = slow_path(cv, accommodation_d)
            fast, info = macro.sweep_accommodation(
                cv, accommodation_d, ADE_VALUES, BDE_VALUES, WORK_DIR, f"validate_{ai}",
                keep_csv=args.keep_work,
            )
            if (info["fov_x_deg"], info["fov_y_deg"]) != (fov_x, fov_y):
                raise RuntimeError(
                    f"FOV differs between paths: slow {(fov_x, fov_y)}, "
                    f"macro {(info['fov_x_deg'], info['fov_y_deg'])}"
                )
            macro.validate_group(fast, ADE_VALUES, BDE_VALUES, accommodation_d, macro.GRID_LINES)
            n_states = len(ADE_VALUES) * len(BDE_VALUES)
            print(
                f"Accommodation {accommodation_d:g} D: fov=({fov_x:g}, {fov_y:g}) deg; "
                f"slow {slow_s:.2f} s ({1000 * slow_s / n_states:.1f} ms/state), "
                f"macro {info['macro_seconds']:.3f} s ({info['ms_per_state']:.1f} ms/state)",
                flush=True,
            )
            slow_parts.append(slow)
            macro_parts.append(fast)
    finally:
        try:
            if cv is not None and started:
                cv.StopCodeV()
        finally:
            pythoncom.CoUninitialize()

    slow = np.concatenate(slow_parts)
    fast = np.concatenate(macro_parts)
    if slow.dtype.names != GRID_FIELDS or fast.dtype.names != GRID_FIELDS or slow.size != fast.size:
        print(f"FAIL: schema/size mismatch slow={slow.size} macro={fast.size}")
        return 1
    print(f"\nCompared {slow.size} rows x {len(GRID_FIELDS)} columns "
          f"({len(ACCOMMODATIONS)} acc x {len(ADE_VALUES)} ADE x {len(BDE_VALUES)} BDE x 9)")
    ok = True
    print(f"{'column':<28} {'max |diff|':>16}  unit   result")
    for field in GRID_FIELDS:
        diff = float(np.max(np.abs(slow[field] - fast[field])))
        if field in EXACT_FIELDS:
            passed, shown, unit = diff == 0.0, diff, "exact"
        elif field in COORD_FIELDS:
            shown, unit = diff * 1000.0, "um"
            passed = shown < COORD_TOL_UM
        else:
            shown, unit = diff, "pct"
            passed = diff < PCT_TOL
        ok &= passed
        print(f"{field:<28} {shown:16.6e}  {unit:<5}  {'ok' if passed else 'FAIL'}")
    # Anti-false-positive: the compared data must actually vary with accommodation and gaze.
    n_acc = fast.size // len(ACCOMMODATIONS)
    acc_spread = float(np.max(np.abs(fast["real_x_mm"][:n_acc] - fast["real_x_mm"][n_acc:]))) * 1000.0
    gaze_spread = float(np.ptp(fast["center_absolute_x_mm"])) * 1000.0
    print(f"{'(spread) real_x 0 D vs 4 D':<28} {acc_spread:16.6e}  um")
    print(f"{'(spread) center_abs_x gaze':<28} {gaze_spread:16.6e}  um")
    if acc_spread < 1.0 or gaze_spread < 1.0:
        print("FAIL: compared data does not vary with accommodation/gaze (test would be vacuous)")
        ok = False
    if not args.keep_work and WORK_DIR.exists():
        macro.remove_tree(WORK_DIR)
    print(f"\nVALIDATION {'PASS' if ok else 'FAIL'} "
          f"(coordinates < {COORD_TOL_UM} um, pct < {PCT_TOL:g}, exact columns identical)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
