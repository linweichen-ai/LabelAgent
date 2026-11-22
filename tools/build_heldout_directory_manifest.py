#!/usr/bin/env python3
"""Freeze a local image/label directory as a read-only held-out manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


SCHEMA = "labelagent.heldout_evaluation_manifest.v1"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def build(source_root: Path, output: Path, *, dataset: str, sequence: str,
          cameras: tuple[str, ...] = ("camera_0", "camera_1", "camera_2", "camera_3"),
          limit: int | None = None) -> dict:
    source_root = source_root.resolve()
    output = output.resolve()
    if not source_root.is_dir():
        raise ValueError(f"source root is unavailable: {source_root}")
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    if not sequence or any(part in sequence for part in ("/", "\\", "..")):
        raise ValueError("sequence must be one safe path component")
    if not cameras or len(set(cameras)) != len(cameras):
        raise ValueError("camera list is empty or duplicated")
    labels = source_root / "labels"
    if not labels.is_dir():
        raise ValueError("source labels directory is unavailable")
    label_paths = sorted(labels.glob("*.json"))
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        label_paths = label_paths[:limit]
    if not label_paths:
        raise ValueError("no label files selected")

    files: list[dict] = []
    label_rows: list[dict] = []
    checked: list[tuple[Path, Path]] = []
    for label_path in label_paths:
        frame = label_path.stem
        try:
            payload = json.loads(label_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"label is not valid UTF-8 JSON: {label_path}") from error
        if str(payload.get("frame_id")) != frame:
            raise ValueError(f"label frame_id mismatch: {label_path}")
        staged_label = output / "labels" / sequence / label_path.name
        checked.append((label_path, staged_label))
        label_rows.append({
            "sequence": sequence,
            "frame_id": frame,
            "source_relative_path": label_path.relative_to(source_root).as_posix(),
            "staged_relative_path": staged_label.relative_to(output).as_posix(),
            "sha256": digest(label_path),
        })
        for camera in cameras:
            image = source_root / "cameras" / camera / f"{frame}.png"
            if not image.is_file():
                raise ValueError(f"paired image is missing: {image}")
            files.append({
                "sequence": sequence,
                "frame_id": frame,
                "camera": camera,
                "relative_path": image.relative_to(source_root).as_posix(),
                "sha256": digest(image),
            })

    output.mkdir(parents=True)
    try:
        for source, target in checked:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if digest(target) != digest(source):
                raise RuntimeError(f"staged label failed hash readback: {source.name}")
        manifest = {
            "schema_version": SCHEMA,
            "created_date": "2026-11-22",
            "dataset": dataset,
            "split": "test",
            "evaluation_only": True,
            "training_return_allowed": False,
            "candidate_write_allowed": False,
            "human_truth_write_allowed": False,
            "selection": (
                f"all {len(label_paths)} frozen local frames; "
                f"{len(cameras)} paired cameras per frame"
            ),
            "local_root": str(source_root),
            "frame_count": len(label_paths),
            "camera_count": len(cameras),
            "image_count": len(files),
            "label_files": label_rows,
            "files": files,
        }
        manifest_path = output / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        receipt = {
            "schema_version": "labelagent.heldout_directory_manifest_receipt.v1",
            "manifest_sha256": digest(manifest_path),
            "frame_count": len(label_paths),
            "camera_count": len(cameras),
            "image_count": len(files),
            "label_count": len(label_rows),
            "source_writes": 0,
            "workbench_writes": 0,
            "human_truth_writes": 0,
            "training_data_writes": 0,
            "production_deployed": False,
        }
        (output / "receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        return receipt
    except Exception:
        # The output is new and contains only this tool's generated files.
        shutil.rmtree(output, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    receipt = build(args.source_root, args.output, dataset=args.dataset,
                    sequence=args.sequence, limit=args.limit)
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
