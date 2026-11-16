r"""Inspect P4 pickle structure only; no fitting, CODE V, or data changes.

Run in PowerShell (from any directory):
& 'C:\Users\rueijrwu\.pyenv-win-venv\envs\venv\Scripts\python.exe' 'C:\Users\rueijrwu\OneDrive\Project\DistortionTracking\Script\p4_a0_data.py'
"""
import argparse
import hashlib
import json
import math
import pickle
import platform
from collections.abc import Mapping
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = Path("data/p4_raytrace/p4_raytrace_251x251_12p5_A4.pkl")
MISSING = "MISSING"
# Names copied from p4_retrace_full_detail.py; no imports of tracing/fit modules.
EXPECTED = (
    "state_index ray_index accommodation_index ade_index bde_index accommodation_d "
    "ade_deg bde_deg field_x_relative field_y_relative rc_ade_readback_deg "
    "rc_bde_readback_deg cornea_ent_d_thi_readback_mm center_rayrsi_status rayrsi_status "
    "x13 y13 z13 l13 m13 n13 x16 y16 z16 l16 m16 n16 xsi ysi zsi "
    "xsi_local ysi_local center_xsi_local center_ysi_local"
).split()


def safe(value):
    if isinstance(value, Mapping):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, (tuple, list)):
        return [safe(v) for v in value]
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)  # Preserve NaN/Infinity explicitly in valid JSON.
    return value if isinstance(value, (str, int, float, bool, type(None))) else repr(value)


def inventory(value, name, found):
    if isinstance(value, Mapping):
        for key, child in value.items():
            inventory(child, f"{name}.{key}", found)
        return
    if not isinstance(value, (np.ndarray, list, tuple)):
        return
    try:
        array = np.asarray(value)  # Existing ndarrays and field views are not copied.
    except (ValueError, TypeError):
        for i, child in enumerate(value):
            inventory(child, f"{name}[{i}]", found)
        return
    stats = dict(shape=list(array.shape), dtype=str(array.dtype), min="N/A", max="N/A", nan_count="N/A")
    found[name] = stats
    if array.dtype.names:
        for key in array.dtype.names:
            inventory(array[key], f"{name}.{key}", found)
    elif array.dtype.kind in "biufc":
        lo = hi = None
        nan_count = 0
        for chunk in np.nditer(array, flags=["external_loop", "buffered", "zerosize_ok"],
                               op_flags=["readonly"], buffersize=65536):
            missing = int(np.count_nonzero(np.isnan(chunk)))
            nan_count += missing
            if missing < chunk.size:
                a, b = np.nanmin(chunk), np.nanmax(chunk)
                lo, hi = a if lo is None else min(lo, a), b if hi is None else max(hi, b)
        stats.update(min=lo, max=hi, nan_count=nan_count)
    elif array.dtype.kind == "O":
        for i, child in enumerate(array.flat):
            inventory(child, f"{name}[{i}]", found)
    elif array.size and array.dtype.kind in "SU":
        stats.update(min=min(array.flat), max=max(array.flat), nan_count=0)


def state_patterns(rows, names):
    if "state_index" not in names:
        return MISSING
    # Group by the stored state key, never by an assumed grid or ray count.
    ids = rows["state_index"]
    order = np.argsort(ids, kind="stable") if np.any(ids[1:] < ids[:-1]) else None
    if order is not None:
        ids = ids[order]
    starts = np.r_[0, 1 + np.flatnonzero(ids[1:] != ids[:-1]), len(ids)]
    labels = ("ray_index", "field_x_relative", "field_y_relative")
    patterns = {}
    for start, stop in zip(starts[:-1], starts[1:]):
        if start == stop:
            continue
        block = rows[start:stop] if order is None else rows[order[start:stop]]
        columns = [safe(block[k]) if k in names else [MISSING] * len(block) for k in labels]
        signature = tuple(zip(*columns))
        if signature not in patterns:
            patterns[signature] = {"rays_per_state": len(block), "state_count": 0,
                "field_labels": [dict(zip(labels, values)) for values in signature], "state_index_ranges": []}
        record = patterns[signature]
        record["state_count"] += 1
        ranges, state = record["state_index_ranges"], safe(ids[start])
        if ranges and isinstance(state, (int, float)) and state == ranges[-1][1] + 1:
            ranges[-1][1] = state
        else:
            ranges.append([state, state])
    return {"note": "Inclusive state_index ranges; labels checked for every state; no grid reshape.",
            "count": sum(p["state_count"] for p in patterns.values()), "patterns": list(patterns.values())}


def inspect(data, report):
    names = data.dtype.names or () if isinstance(data, np.ndarray) else ()
    report["top_level"] = {"type": f"{type(data).__module__}.{type(data).__qualname__}",
        "keys": list(data) if isinstance(data, Mapping) else list(names) if names else MISSING}
    report["arrays"] = {}
    inventory(data, "data", report["arrays"])
    metadata = dict(data.dtype.metadata or {}) if isinstance(data, np.ndarray) else {}
    report["metadata"] = metadata or MISSING
    inventory(metadata, "metadata", report["arrays"])
    meta = metadata.get("p4_retrace", {})
    axes = meta.get("axes", {})
    report["expected_metadata"] = {k: meta.get(k, MISSING) for k in ("lens", "profile", "record_count")}
    report["expected_metadata"]["axes"] = {k: axes.get(k, MISSING) for k in
        ("accommodation_d", "ade_deg", "bde_deg", "field_xy_y_major")}
    report["schema_fields"] = {key: "PRESENT" if key in names else MISSING for key in EXPECTED}
    if not isinstance(data, np.ndarray) or not names or data.ndim != 1:
        raise ValueError("Expected a flat structured ndarray; structure reported, no reshape attempted")
    report["ray_count"] = len(data)
    report["axes_observed"] = {}
    for key, index in (("accommodation_d", "accommodation_index"), ("ade_deg", "ade_index"), ("bde_deg", "bde_index")):
        if key not in names:
            report["axes_observed"][key] = MISSING
            continue
        values = np.unique(data[key])
        steps = np.unique(np.round(np.diff(values), 12))
        report["axes_observed"][key] = {"values": values, "count": len(values),
            "start": values[0] if values.size else None, "stop": values[-1] if values.size else None,
            "step": steps[0] if len(steps) == 1 else "N/A (nonuniform or fewer than two values)",
            "steps_observed_rounded_12dp": steps, "zero_index_in_sorted_values": np.flatnonzero(values == 0),
            "stored_indices_at_zero": np.unique(data[index][data[key] == 0]) if index in names else MISSING}
    has_a0 = bool(np.any(data["accommodation_d"] == 0)) if "accommodation_d" in names else False
    report["A0_exists"] = has_a0 if "accommodation_d" in names else MISSING
    report["per_state"] = state_patterns(data, names)
    report["ray_status_counts"] = {key: dict(zip(*np.unique(data[key], return_counts=True)))
        if key in names else MISSING for key in ("rayrsi_status", "center_rayrsi_status")}
    diagnostic = {"note": "DIAGNOSTIC ONLY: sensor xsi/ysi/zsi in S16 frame (mm); tx=-l16/n16, ty=-m16/n16.", "rows": []}
    report["zero_gaze_A0_diagnostic"] = diagnostic
    if all(k in names for k in ("accommodation_d", "ade_deg", "bde_deg")):
        indices = np.flatnonzero((data["accommodation_d"] == 0) & (data["ade_deg"] == 0) & (data["bde_deg"] == 0))
        diagnostic["row_count"] = len(indices)
        nodes = (-1.0, 0.0, 1.0) if len(indices) == 9 else (-1.0, -0.5, 0.0, 0.5, 1.0)
        expected_labels = {(x, y) for y in nodes for x in nodes}
        labelled_complete = (len(indices) in (9, 25)
            and all(k in names for k in ("field_x_relative", "field_y_relative"))
            and set(zip(data["field_x_relative"][indices], data["field_y_relative"][indices])) == expected_labels)
        # Retain the historical report key for 3x3 consumers; 5x5 is also complete.
        diagnostic["nine_rows_present"] = labelled_complete
        diagnostic["complete_labelled_rows_present"] = labelled_complete
        diagnostic["accepted_rows_per_state"] = [9, 25]
        for i in indices:
            row = {k: data[k][i] if k in names else MISSING for k in
                ("ray_index", "field_x_relative", "field_y_relative", "xsi", "ysi", "zsi", "l16", "m16", "n16")}
            row["field_label"] = [row["field_x_relative"], row["field_y_relative"]]
            for target, numerator in (("tx", "l16"), ("ty", "m16")):
                row[target] = (-row[numerator] / row["n16"] if row["n16"] != 0 else "UNDEFINED (n16=0)") if numerator in names and "n16" in names else MISSING
            diagnostic["rows"].append(row)
    else:
        diagnostic["row_count"] = MISSING
    report["verdict"] = "READABLE_A0_FOUND" if has_a0 else "READABLE_A0_MISSING"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT)
    args = parser.parse_args()
    path = (ROOT / args.input).resolve()
    out = ROOT / "runs" / "p4_data"
    targets = [out / "report.txt", out / "report.json"]
    if any(path == p.resolve() or (p.exists() and path.exists() and path.samefile(p)) for p in targets):
        parser.error("Input aliases a report output; refusing to overwrite the read-only input")
    report = {"input": str(path), "python_version": platform.python_version(), "numpy_version": np.__version__,
              "file_size_bytes": MISSING, "sha256": MISSING}
    try:
        report["file_size_bytes"] = path.stat().st_size
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
            report["sha256"] = digest.hexdigest()
            stream.seek(0)
            data = pickle.load(stream)  # Exactly one load; never write to the input.
        inspect(data, report)
    except Exception as exc:
        report["verdict"] = f"UNREADABLE: {type(exc).__name__}: {exc}"
    report = safe(report)
    text = "\n\n".join(f"{k}:\n{json.dumps(v, indent=2, allow_nan=False)}" for k, v in report.items() if k != "verdict") + "\n\n" + report["verdict"] + "\n"
    out.mkdir(parents=True, exist_ok=True)
    targets[0].write_text(text, encoding="utf-8")
    targets[1].write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(text, end="")
    return int(report["verdict"].startswith("UNREADABLE"))


if __name__ == "__main__":
    raise SystemExit(main())
