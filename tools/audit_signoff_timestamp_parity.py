"""Parity audit: reviewer page `validTime` versus importer `_valid_timestamp`.

The offline reviewer page must never accept a timestamp that the import gate
rejects, otherwise a complete 81-row human signoff becomes unimportable.
This audit runs the *real* page JavaScript (extracted from a synthetic build)
in a Node VM and compares it with the Python importer on a shared vector set.
Synthetic data only; no network, no state writes.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.build_cvat_human_signoff_reviewer import _page_html  # noqa: E402
from tools.validate_cvat_human_signoff_import import _valid_timestamp  # noqa: E402

VECTORS: dict[str, bool] = {
    # expected = importer verdict (the gate of record)
    "2026-11-22T10:00:00+08:00": True,
    "2026-11-22T10:00:00Z": True,
    "2026-11-22T10:00:00.123456Z": True,
    "2026-12-31T23:59:59-11:30": True,
    "2026-02-30T12:00:00+08:00": False,   # impossible date
    "2026-13-01T00:00:00Z": False,        # month 13
    "2026-11-22T24:00:00Z": False,        # hour 24
    "2026-11-22T12:60:00Z": False,        # minute 60
    "2026-11-22T12:00:60Z": False,        # second 60
    "2026-11-22T12:00:00+24:00": False,   # offset hour 24
    "2026-11-22T12:00:00+08:60": False,   # offset minute 60
    "2026-11-22T12:00:00.1234567Z": False,  # 7 fractional digits
    "2026-11-22T12:00:00": False,         # no timezone
    "2026-11-22 12:00:00Z": False,        # space separator
    " 2026-11-22T12:00:00Z": False,       # leading whitespace
    "2026-11-22T12:00:00Z ": False,       # trailing whitespace
    "": False,
}

NODE_DRIVER = r"""
const vm = require("node:vm");
const [src, vectorsJson] = [process.argv[2], process.argv[3]];
const ctx = { Date, Number };
vm.runInNewContext(require("node:fs").readFileSync(src, "utf8"), ctx);
const out = {};
for (const v of JSON.parse(require("node:fs").readFileSync(vectorsJson, "utf8"))) out[v] = Boolean(ctx.validTime(v));
process.stdout.write(JSON.stringify(out));
"""


def extract_valid_time(page: str) -> str:
    script = next(s for s in re.findall(r"<script>([\s\S]*?)</script>", page) if '"use strict"' in s)
    match = re.search(r"function validTime\(value\)\{.*?\}\n", script, re.S)
    if not match:
        raise RuntimeError("validTime not found in page script")
    return match.group(0)


def main() -> int:
    node = shutil.which("node")
    if not node:
        print("node is required", file=sys.stderr)
        return 2
    rows = [{
        "candidate_id": f"synthetic-{i:03d}", "frame_id": i,
        "source_image_sha256": hashlib.sha256(f"s{i}".encode()).hexdigest(),
        "canonical_geometry_sha256": hashlib.sha256(f"g{i}".encode()).hexdigest(),
        "focused_image_sha256": hashlib.sha256(f"f{i}".encode()).hexdigest(),
    } for i in range(81)]
    page = _page_html(rows, ["data:image/jpeg;base64,AA=="] * 81, "a" * 64, "b" * 64)
    with tempfile.TemporaryDirectory() as directory:
        temp = Path(directory)
        (temp / "validTime.js").write_text(extract_valid_time(page), encoding="utf-8")
        (temp / "driver.js").write_text(NODE_DRIVER, encoding="utf-8")
        (temp / "vectors.json").write_text(json.dumps(list(VECTORS)), encoding="utf-8")
        completed = subprocess.run(
            [node, str(temp / "driver.js"), str(temp / "validTime.js"), str(temp / "vectors.json")],
            capture_output=True, text=True, timeout=60, check=True,
        )
    page_verdicts = json.loads(completed.stdout)

    rows_out = []
    unsafe = 0
    for vector, expected in VECTORS.items():
        importer = _valid_timestamp(vector)
        page_ok = page_verdicts[vector]
        if importer != expected:
            raise AssertionError(f"importer verdict drifted from expectation: {vector!r}")
        # Unsafe = page accepts what the importer rejects (human work becomes unimportable).
        if page_ok and not importer:
            unsafe += 1
        rows_out.append({"vector": vector, "importer": importer, "page": page_ok,
                         "parity": importer == page_ok})
    report = {
        "schema_version": "labelagent.signoff-timestamp-parity-audit.v1",
        "vector_count": len(rows_out),
        "mismatch_count": sum(not r["parity"] for r in rows_out),
        "page_accepts_importer_rejects": unsafe,
        "verdict": "PARITY" if unsafe == 0 else "UNSAFE_PAGE_LOOSER_THAN_IMPORTER",
        "rows": rows_out,
        "synthetic_only": True,
        "training_authorized": False,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if unsafe == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
