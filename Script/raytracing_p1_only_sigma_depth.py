"""Trace the P1-only sigma lens chief ray over RC translation and rotation."""

from __future__ import annotations

import argparse
import math
import os
import pickle
from pathlib import Path

import numpy as np
import pythoncom
import win32com.client


ROOT = Path(__file__).resolve().parent.parent
SEQ = ROOT / "Lens" / "p1_only_sigma_depth.seq"
OUTPUT = ROOT / "data" / "p1_only_sigma_depth_raytracing" / "p1_only_sigma_depth_raytracing.pkl"
S_RC = 's"RC"'
ZOOMS = (
    # zoom, title, expected F1 YOB, S0 THI, S1 THI, S1 ADE, S1 YDE
    (1, "Inf_S1ADE-15", 0.0, 1.0e13, 50.0, -15.0, 0.0),
    (2, "finite_S1YDE+5", 0.0, 0.0, 50.0, 0.0, 5.0),
    (3, "finite_S1YDE-5", 0.0, 0.0, 25.0, 0.0, -5.0),
)


def cv_eval(cv, command: str, *args: str) -> float:
    return float(cv.EvaluateExpression(f"({command} {' '.join(args)})"))


def checked(cv, command: str) -> str:
    response = cv.Command(command)
    if "Command End:" not in response or any(
        line.lstrip().lower().startswith("error:") for line in response.splitlines()
    ):
        raise RuntimeError(f"CODE V failed command {command!r}: {response!r}")
    return response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rotation-min", type=float, default=-20.0)
    parser.add_argument("--rotation-max", type=float, default=20.0)
    parser.add_argument("--rotation-count", type=int, default=401)
    parser.add_argument("--translation-min", type=float, default=-1.0)
    parser.add_argument("--translation-max", type=float, default=1.0)
    parser.add_argument("--translation-count", type=int, default=201)
    args = parser.parse_args()
    if (args.rotation_count < 1 or args.rotation_max < args.rotation_min
            or args.translation_count < 1 or args.translation_max < args.translation_min):
        parser.error("counts must be positive and max values must be >= min values")
    if not SEQ.is_file():
        raise FileNotFoundError(SEQ)

    rotations = np.linspace(args.rotation_min, args.rotation_max, args.rotation_count)
    translations = np.linspace(args.translation_min, args.translation_max, args.translation_count)
    rows = []
    cv = None
    started = False
    pythoncom.CoUninitialize()
    pythoncom.CoInitialize()
    try:
        print("Creating CODE V COM object...", flush=True)
        cv = win32com.client.Dispatch("CodeV.Application")
        cv.StartingDirectory = str(SEQ.parent)
        print("Starting CODE V...", flush=True)
        cv.StartCodeV()
        started = True
        # Startup is synchronous in the established project COM workflow.
        checked(cv, f'pth seq app {SEQ.parent}\\')
        checked(cv, f'run "{SEQ}"')

        # Validate all required zoom-specific properties directly; this also
        # catches an unloaded lens without relying on an undocumented NZO token.
        field_count = int(cv.FieldCount)
        if field_count != 1:
            raise RuntimeError(f"Expected only F1 in p1_only_sigma_depth.seq, CODE V reports {field_count} fields")

        # Confirm the sequence's intended per-zoom field and object conjugates.
        zoom_readbacks = {}
        for zoom_no, title, expected_yob, expected_s0_thickness, expected_s1_thickness, expected_s1_ade, expected_s1_yde in ZOOMS:
            z = f"z{zoom_no}"
            yob = cv_eval(cv, "yob", "f1", z)
            actual_s0_thickness = cv_eval(cv, "thi", "s0", z)
            actual_s1_thickness = cv_eval(cv, "thi", "s1", z)
            actual_s1_ade = cv_eval(cv, "ade", "s1", z)
            actual_s1_yde = cv_eval(cv, "yde", "s1", z)
            if not math.isclose(yob, expected_yob, abs_tol=1e-8, rel_tol=0.0):
                raise RuntimeError(f"F1 YOB readback at {z}: expected {expected_yob}, got {yob}")
            if not math.isclose(
                actual_s0_thickness, expected_s0_thickness, abs_tol=1e-8, rel_tol=0.0
            ):
                raise RuntimeError(f"S0 THI readback at {z}: expected {expected_s0_thickness}, got {actual_s0_thickness}")
            if not math.isclose(
                actual_s1_thickness, expected_s1_thickness, abs_tol=1e-8, rel_tol=0.0
            ):
                raise RuntimeError(f"S1 THI readback at {z}: expected {expected_s1_thickness}, got {actual_s1_thickness}")
            if not math.isclose(actual_s1_ade, expected_s1_ade, abs_tol=1e-8, rel_tol=0.0):
                raise RuntimeError(f"S1 ADE readback at {z}: expected {expected_s1_ade}, got {actual_s1_ade}")
            if not math.isclose(actual_s1_yde, expected_s1_yde, abs_tol=1e-8, rel_tol=0.0):
                raise RuntimeError(f"S1 YDE readback at {z}: expected {expected_s1_yde}, got {actual_s1_yde}")
            zoom_readbacks[zoom_no] = (yob, actual_s0_thickness, actual_s1_thickness,
                                       actual_s1_ade, actual_s1_yde)

        for ti, translation in enumerate(translations, 1):
            checked(cv, f"yde {S_RC} {translation:.8f}")
            actual_translation = cv_eval(cv, "yde", S_RC)
            if not math.isclose(actual_translation, float(translation), abs_tol=1e-7, rel_tol=0.0):
                raise RuntimeError(
                    f"RC YDE readback mismatch: requested {translation}, got {actual_translation}"
                )
            for i, angle in enumerate(rotations, 1):
                checked(cv, f"ade {S_RC} {angle:.8f};set vig")
                actual_rotation = cv_eval(cv, "ade", S_RC)
                actual_translation = cv_eval(cv, "yde", S_RC)
                if not math.isclose(actual_rotation, float(angle), abs_tol=1e-7, rel_tol=0.0):
                    raise RuntimeError(f"RC ADE readback mismatch: requested {angle}, got {actual_rotation}")
                if not math.isclose(actual_translation, float(translation), abs_tol=1e-7, rel_tol=0.0):
                    raise RuntimeError(
                        f"RC YDE changed during trace: expected {translation}, got {actual_translation}"
                    )
                for zoom_no, title, expected_yob, expected_s0, expected_s1, expected_ade, expected_yde in ZOOMS:
                    z = f"z{zoom_no}"
                    actual_yob, object_thi_s0, s1_thickness, s1_ade, s1_yde = zoom_readbacks[zoom_no]
                    response = cv.Command(f"rsi r1 fl {z} si")
                    failed = "error" in response.lower()
                    err = ""
                    real_y = math.nan
                    if failed:
                        err = response.strip()
                    else:
                        try:
                            real_y = cv_eval(cv, "y", "r1", "fl", z, "si")
                            if not math.isfinite(real_y):
                                failed = True
                                err = "nonfinite SI y"
                        except Exception as exc:
                            failed = True
                            err = repr(exc)
                    rows.append((float(translation), float(angle), zoom_no, 1, title,
                                 actual_yob, object_thi_s0, s1_thickness, s1_ade, s1_yde,
                                 real_y, not failed, err))
                if i == 1 or i == len(rotations) or i % max(1, len(rotations) // 10) == 0:
                    print(
                        f"Translation {ti}/{len(translations)} ({translation:.2f} mm), "
                        f"rotation {i}/{len(rotations)} ({angle:.2f} deg)", flush=True
                    )

        dtype = np.dtype([
            ("h_translation_mm", "f8"), ("rotation_deg", "f8"),
            ("zoom_number", "i4"), ("field_number", "i4"),
            ("zoom_title", "U24"), ("field_yob_mm", "f8"),
            ("object_thi_s0_mm", "f8"), ("s1_thickness_mm", "f8"),
            ("s1_ade_deg", "f8"), ("s1_yde_mm", "f8"),
            ("real_y_si_mm", "f8"), ("trace_success", "?"), ("error", "U512"),
        ])
        data = np.array(rows, dtype=dtype)
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        tmp = OUTPUT.with_name(OUTPUT.name + ".tmp")
        with tmp.open("wb") as stream:
            pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, OUTPUT)
        failures = int(np.count_nonzero(~data["trace_success"]))
        print(f"Saved {len(data)} rows to {OUTPUT}; failed traces={failures}", flush=True)
    finally:
        try:
            if started and cv is not None:
                cv.StopCodeV()
        finally:
            pythoncom.CoUninitialize()


if __name__ == "__main__":
    main()
