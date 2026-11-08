"""Pin numerical reference/calibration arrays without modifying runtime files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from fit_p4_limited_gaze import DEFAULT_OUT
from fit_p4_sparse_calibration import write_json


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(array):
    array = np.ascontiguousarray(array)
    header = json.dumps({"dtype": str(array.dtype), "shape": list(array.shape)}, sort_keys=True).encode()
    return hashlib.sha256(header+b"\n"+array.tobytes()).hexdigest()


def numeric_arrays(path):
    with np.load(path) as saved:
        return {key: {"sha256": array_hash(saved[key]), "shape": list(saved[key].shape), "dtype": str(saved[key].dtype)}
            for key in saved.files if saved[key].dtype.kind in "biufc"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--same-map-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    runtime = args.runtime_dir or args.calibration_dir/"joint_runtime"
    out = args.output_dir or args.calibration_dir/"provenance_audit"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((runtime/"summary.json").read_text(encoding="utf-8"))
    cases = dict(summary["cases"])
    if args.same_map_dir:
        extra = json.loads((args.same_map_dir/"summary.json").read_text(encoding="utf-8"))
        cases[extra["case"]] = extra
    result = {"source_subset_sha256": file_hash(args.calibration_dir/"source_subset.npz"),
        "fit_summary_sha256": file_hash(args.calibration_dir/"summary.json"),
        "runtime_summary_sha256": file_hash(runtime/"summary.json"), "cases": {}}
    for name, case in cases.items():
        paths = {"shape": Path(case["shape_path"]), "map": Path(case["map_path"])}
        row = {}
        for kind, path in paths.items():
            actual = file_hash(path)
            if actual != case[f"{kind}_sha256"]:
                raise RuntimeError(f"Frozen input changed since inference: {name}:{kind}")
            row[kind] = {"path": str(path), "file_sha256": actual, "numerical_arrays": numeric_arrays(path)}
        result["cases"][name] = row
    write_json(out/"reference_calibration_hashes.json", result)


if __name__ == "__main__":
    main()
