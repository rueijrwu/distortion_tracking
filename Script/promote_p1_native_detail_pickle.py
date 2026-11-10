"""Atomically promote the validated native-source P1 pickle to its user alias."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data" / "p1_retrace_native_detail_251x251" / "20261109T221348Z_1f596ed9"
SOURCE = RUN / "p1_retrace_detail_251x251.pkl"
DEST = ROOT / "data" / "distortion_grid" / "p1_retrace_detail_251x251.pkl"
HISTORICAL = ROOT / "data" / "p1_retrace_full_detail_251x251" / "20261109T210803Z_88259c44" / "p1_retrace_detail_251x251.pkl"
LEGACY_ALIAS = ROOT / "data" / "distortion_grid" / "p1_full_101_125.pkl"
EXPECTED_NEW = "2249c93a9b8ad59e254f4941015fb9108b8a22459ab361320ad3cf1d64273736"
EXPECTED_OLD = "61a89af4e30f7ba51fb8de4525bd62f0103ec34caabad5335c54f946229095dc"


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def main() -> None:
    manifest = json.loads((RUN / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or not manifest.get("final", {}).get("validation_pass"):
        raise RuntimeError("Native-source run is not complete and validated")
    if len(manifest["completed_chunks"]) != 253 or manifest["final"]["rows"] != 567009:
        raise RuntimeError("Run coverage does not match the expected full grid")
    if digest(SOURCE) != EXPECTED_NEW:
        raise RuntimeError("New source pickle hash mismatch")
    if digest(HISTORICAL) != EXPECTED_OLD or digest(DEST) != EXPECTED_OLD:
        raise RuntimeError("Historical source or current alias hash mismatch")
    temporary = DEST.with_name(DEST.name + ".tmp_" + uuid.uuid4().hex[:8])
    try:
        shutil.copyfile(SOURCE, temporary)
        if digest(temporary) != EXPECTED_NEW:
            raise RuntimeError("Copied temporary pickle hash mismatch")
        for index, delay in enumerate((0.5, 1.0, 2.0, 4.0, None)):
            try:
                os.replace(temporary, DEST)
                break
            except PermissionError:
                if delay is None:
                    raise
                time.sleep(delay)
        if digest(DEST) != EXPECTED_NEW:
            raise RuntimeError("Promoted alias hash mismatch")
    finally:
        if temporary.exists():
            temporary.unlink()
    print(json.dumps({
        "destination": str(DEST), "sha256": digest(DEST), "bytes": DEST.stat().st_size,
        "historical_source_preserved": digest(HISTORICAL) == EXPECTED_OLD,
        "legacy_alias_preserved": LEGACY_ALIAS.is_file(),
    }))


if __name__ == "__main__":
    main()
