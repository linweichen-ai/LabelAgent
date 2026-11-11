"""Build a deterministic, candidate-only batch manifest from a COCO pack.

The tool never interprets ``_preannotations.coco.json`` as truth.  It hashes
every source image and emits the identity fields required by the closed-loop
contract.  Output is stdout only so callers choose an explicit destination.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


NAME = re.compile(
    r"^(?P<sequence>[0-9]{3})_(?P<frame_id>[0-9]+)_"
    r"(?P<camera>camera_[0-3])\.png$"
)
FIXED_HOLDOUT = ("1789897506782450197", "camera_0")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value):
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def frame_time(frame_id):
    value = datetime.fromtimestamp(int(frame_id) / 1_000_000_000, tz=timezone.utc)
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def build_manifest(pack_root, split, dataset_id, source_batch, limit=0):
    split_dir = Path(pack_root) / split
    coco_path = split_dir / "_preannotations.coco.json"
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    categories = {item["id"]: item["name"] for item in coco["categories"]}
    counts = Counter(item["image_id"] for item in coco["annotations"])
    images = sorted(coco["images"], key=lambda item: item["file_name"])
    entries = []
    excluded = []
    for image in images:
        match = NAME.fullmatch(image["file_name"])
        if not match:
            raise ValueError(f"unsupported image file name: {image['file_name']}")
        identity = match.groupdict()
        if (identity["frame_id"], identity["camera"]) == FIXED_HOLDOUT:
            excluded.append(image["file_name"])
            continue
        path = split_dir / image["file_name"]
        if not path.is_file():
            raise FileNotFoundError(path)
        image_sha256 = sha256(path)
        input_identity = {
            "dataset_id": dataset_id,
            "source_batch": source_batch,
            "sequence": identity["sequence"],
            "frame_id": identity["frame_id"],
            "camera": identity["camera"],
            "image_sha256": image_sha256,
        }
        entries.append({
            **input_identity,
            "input_id": canonical_sha256(input_identity),
            "capture_time": frame_time(identity["frame_id"]),
            "source_kind": "real",
            "relative_image_path": f"{split}/{image['file_name']}",
            "width": image["width"],
            "height": image["height"],
            "machine_preannotation_count": counts[image["id"]],
            "candidate_only": True,
            "truth_eligible": False,
            "approved_for_training": False,
        })
    if limit:
        entries = entries[:limit]
    body = {
        "schema_version": "labelagent.candidate-batch-manifest.v1",
        "dataset_id": dataset_id,
        "source_batch": source_batch,
        "split": split,
        "source_kind": "real",
        "preannotation_source": {
            "path": f"{split}/_preannotations.coco.json",
            "sha256": sha256(coco_path),
            "role": "machine_candidate_only",
            "truth_eligible": False,
            "categories": [categories[key] for key in sorted(categories)],
        },
        "fixed_holdout_exclusion": {
            "frame_id": FIXED_HOLDOUT[0],
            "camera": FIXED_HOLDOUT[1],
            "excluded_matches": excluded,
        },
        "entry_count": len(entries),
        "entries": entries,
        "human_truth_available": False,
        "formal_training_authorized": False,
    }
    return {**body, "manifest_content_sha256": canonical_sha256(body)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--source-batch", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.limit < 0:
        raise SystemExit("--limit must be zero or positive")
    result = build_manifest(
        args.pack_root.resolve(), args.split, args.dataset_id,
        args.source_batch, args.limit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
