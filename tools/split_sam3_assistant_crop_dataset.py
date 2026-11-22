"""Create an anti-leak temporal-group split for assistant-reviewed pallet crops."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TRAIN_NAME = "_assistant_delegated_pallet_crops_train56.coco.json"
HOLDOUT_NAME = "_assistant_delegated_pallet_crops_holdout15.coco.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               sort_keys=True) + "\n", encoding="utf-8")


def split(coco, manifest):
    if manifest.get("schema_version") != "labelagent.sam3-assistant-instance-crop-dataset.v1":
        raise ValueError("unsupported source manifest")
    if manifest.get("crop_count") != 71 or len(coco.get("images", [])) != 71:
        raise ValueError("source must contain exactly 71 crops")
    if manifest.get("human_truth") is not False or manifest.get("test24_included") is not False:
        raise ValueError("source boundary drift")
    by_image = {row["id"]: row for row in coco["images"]}
    if len(by_image) != 71 or len(coco.get("annotations", [])) != 71:
        raise ValueError("one-image/one-annotation contract differs")
    holdout_ids = {
        image_id for image_id, row in by_image.items()
        if int(str(row["candidate_id"]).split(":", 1)[0]) <= 15
    }
    if len(holdout_ids) != 15:
        raise ValueError(f"expected temporal group of 15, got {len(holdout_ids)}")
    train_ids = set(by_image) - holdout_ids
    if len(train_ids) != 56 or train_ids & holdout_ids:
        raise ValueError("56/15 split contract differs")

    def subset(ids):
        return {
            "info": dict(coco.get("info") or {}),
            "licenses": list(coco.get("licenses") or []),
            "categories": list(coco["categories"]),
            "images": [row for row in coco["images"] if row["id"] in ids],
            "annotations": [row for row in coco["annotations"] if row["image_id"] in ids],
        }

    return subset(train_ids), subset(holdout_ids), {
        "split_method": "whole earliest temporal group candidate_index_le_15",
        "train_images": 56, "holdout_images": 15,
        "train_candidate_ids": [by_image[index]["candidate_id"] for index in sorted(train_ids)],
        "holdout_candidate_ids": [by_image[index]["candidate_id"] for index in sorted(holdout_ids)],
        "human_truth": False, "assistant_reviewed": True,
        "test24_included": False, "training_authorized": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    coco = json.loads(args.coco.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if sha256(args.coco) != manifest.get("output_coco_sha256"):
        raise ValueError("source COCO hash mismatch")
    train, holdout, receipt = split(coco, manifest)
    args.output.mkdir(parents=True)
    train_path, holdout_path = args.output / TRAIN_NAME, args.output / HOLDOUT_NAME
    _canonical_write(train_path, train)
    _canonical_write(holdout_path, holdout)
    receipt.update({
        "schema_version": "labelagent.sam3-assistant-crop-split.v1",
        "source_coco_sha256": sha256(args.coco),
        "source_manifest_sha256": sha256(args.manifest),
        "train_coco": TRAIN_NAME, "train_coco_sha256": sha256(train_path),
        "holdout_coco": HOLDOUT_NAME, "holdout_coco_sha256": sha256(holdout_path),
    })
    _canonical_write(args.output / "split_manifest.json", receipt)
    print(json.dumps({key: receipt[key] for key in
                      ("train_images", "holdout_images", "train_coco_sha256",
                       "holdout_coco_sha256")}, sort_keys=True))


if __name__ == "__main__":
    main()
