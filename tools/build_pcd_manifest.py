#!/usr/bin/env python3
"""Bind held-out frame identities to local read-only PCD files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def build(manifest_path: Path, pcd_root: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != "labelagent.heldout_evaluation_manifest.v1"
            or manifest.get("evaluation_only") is not True
            or manifest.get("training_return_allowed") is not False):
        raise ValueError("requires a frozen evaluation-only image manifest")
    pcd_root = pcd_root.resolve()
    if not pcd_root.is_dir():
        raise ValueError("PCD root is unavailable")
    identities = sorted({(str(row["sequence"]), str(row["frame_id"]))
                         for row in manifest["files"]})
    if not identities:
        raise ValueError("manifest contains no frame identities")
    rows = []
    for sequence, frame in identities:
        path = pcd_root / f"{frame}.pcd"
        if not path.is_file():
            raise ValueError(f"paired PCD is missing: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"paired PCD is empty: {path}")
        rows.append({
            "sequence": sequence,
            "frame_id": frame,
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": digest(path),
        })
    report = {
        "schema_version": "labelagent.pcd_manifest.v1",
        "source": "read_only_local_frozen_copy",
        "image_manifest_sha256": digest(manifest_path),
        "frame_count": len(rows),
        "files": rows,
        "source_writes": 0,
        "workbench_writes": 0,
        "oss_writes": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--pcd-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = build(args.manifest, args.pcd_root, args.output)
    print(json.dumps({
        "frame_count": report["frame_count"],
        "image_manifest_sha256": report["image_manifest_sha256"],
        "source_writes": report["source_writes"],
        "workbench_writes": report["workbench_writes"],
        "oss_writes": report["oss_writes"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
