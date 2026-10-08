"""Show the COM creation and CODE V startup stages separately."""

from time import monotonic

import pythoncom
import win32com.client


pythoncom.CoUninitialize()
pythoncom.CoInitialize()
cv = None
started = False
try:
    start = monotonic()
    print("Creating CODE V COM object...", flush=True)
    cv = win32com.client.Dispatch("CodeV.Application")
    print(f"COM object created in {monotonic() - start:.1f} s", flush=True)

    start = monotonic()
    print("Calling StartCodeV()...", flush=True)
    status = cv.StartCodeV()
    started = True
    print(f"StartCodeV() returned {status!r} in {monotonic() - start:.1f} s", flush=True)
    print(f"CODE V version: {cv.CodeVVersion}", flush=True)
finally:
    try:
        if started:
            cv.StopCodeV()
    finally:
        pythoncom.CoUninitialize()
