"""Build a versioned, append-only, candidate-only LabelAgent batch manifest.

This static builder performs file hashing only.  It never calls SAM/VLM/GPU,
writes candidate outputs, or promotes machine data to truth.
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
VERSION = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
HASH = re.compile(r"^[0-9a-f]{64}$")
FIXED_HOLDOUT = ("1789897506782450197", "camera_0")
REVIEW_TRIGGERS = ("new_label_or_unknown", "low_confidence", "vlm_sam_conflict")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
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


def _require_hash(name, value):
    if not HASH.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def build_manifest(
    pack_root,
    split,
    dataset_id,
    source_batch,
    manifest_version,
    pipeline_config_sha256,
    sam3_model_sha256,
    vlm_config_sha256,
):
    if not VERSION.fullmatch(manifest_version):
        raise ValueError("manifest_version is not a safe append-only namespace")
    for name, value in (
        ("pipeline_config_sha256", pipeline_config_sha256),
        ("sam3_model_sha256", sam3_model_sha256),
        ("vlm_config_sha256", vlm_config_sha256),
    ):
        _require_hash(name, value)

    split_dir = Path(pack_root) / split
    coco_path = split_dir / "_preannotations.coco.json"
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    preannotation_sha256 = sha256(coco_path)
    categories = {item["id"]: item["name"] for item in coco["categories"]}
    counts = Counter(item["image_id"] for item in coco["annotations"])
    split_id = "train72" if split == "train" else "test24"
    evaluation_only = split == "test"
    entries = []
    excluded = []
    for image in sorted(coco["images"], key=lambda item: item["file_name"]):
        match = NAME.fullmatch(image["file_name"])
        if not match:
            raise ValueError(f"unsupported image file name: {image['file_name']}")
        identity = match.groupdict()
        if (identity["frame_id"], identity["camera"]) == FIXED_HOLDOUT:
            excluded.append(image["file_name"])
            continue
        image_path = split_dir / image["file_name"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        image_sha256 = sha256(image_path)
        input_identity = {
            "dataset_id": dataset_id,
            "source_batch": source_batch,
            "split": split_id,
            "sequence": identity["sequence"],
            "frame_id": identity["frame_id"],
            "camera": identity["camera"],
            "image_sha256": image_sha256,
        }
        input_id = canonical_sha256(input_identity)
        output_candidate_path = (
            f"candidate_outputs/{manifest_version}/{split_id}/"
            f"{identity['sequence']}/{identity['frame_id']}_"
            f"{identity['camera']}.candidate.json"
        )
        idempotency_payload = {
            "input_id": input_id,
            "preannotation_sha256": preannotation_sha256,
            "sam3_model_sha256": sam3_model_sha256,
            "vlm_config_sha256": vlm_config_sha256,
            "pipeline_config_sha256": pipeline_config_sha256,
            "manifest_version": manifest_version,
            "output_candidate_path": output_candidate_path,
        }
        entries.append({
            **input_identity,
            "input_id": input_id,
            "idempotency_key": canonical_sha256(idempotency_payload),
            "capture_time": frame_time(identity["frame_id"]),
            "source_kind": "real",
            "relative_image_path": f"{split}/{image['file_name']}",
            "width": image["width"],
            "height": image["height"],
            "preannotation_sha256": preannotation_sha256,
            "sam3_model_sha256": sam3_model_sha256,
            "vlm_config_sha256": vlm_config_sha256,
            "pipeline_config_sha256": pipeline_config_sha256,
            "machine_preannotation_count": counts[image["id"]],
            "output_candidate_path": output_candidate_path,
            "output_write_mode": "append_only_explicit_new_version",
            "evaluation_only": evaluation_only,
            "training_return_allowed": False,
            "candidate_only": True,
            "truth_eligible": False,
            "approved_for_training": False,
            "human_review_required_if": list(REVIEW_TRIGGERS),
            "automatic_truth_promotion": False,
        })

    body = {
        "schema_version": "labelagent.candidate-batch-manifest.v2",
        "manifest_version": manifest_version,
        "dataset_id": dataset_id,
        "source_batch": source_batch,
        "split": split_id,
        "source_kind": "real",
        "pipeline_config_sha256": pipeline_config_sha256,
        "model_bindings": {
            "sam3_model_sha256": sam3_model_sha256,
            "vlm_config_sha256": vlm_config_sha256,
        },
        "preannotation_source": {
            "path": f"{split}/_preannotations.coco.json",
            "sha256": preannotation_sha256,
            "role": "machine_candidate_only",
            "truth_eligible": False,
            "categories": [categories[key] for key in sorted(categories)],
        },
        "write_policy": {
            "mode": "append_only_explicit_new_version",
            "candidate_root": f"candidate_outputs/{manifest_version}/{split_id}",
            "overwrite_existing_candidate": False,
            "overwrite_human_truth": False,
        },
        "split_policy": {
            "evaluation_only": evaluation_only,
            "test24_training_return_allowed": False,
            "machine_candidate_truth_eligible": False,
        },
        "review_routing": {
            trigger: "human_review_required" for trigger in REVIEW_TRIGGERS
        } | {"automatic_truth_promotion": False},
        "fixed_holdout_exclusion": {
            "frame_id": FIXED_HOLDOUT[0],
            "camera": FIXED_HOLDOUT[1],
            "excluded_matches": excluded,
        },
        "entry_count": len(entries),
        "entries": entries,
        "human_truth_available": False,
        "formal_training_authorized": False,
        "gpu_calls": 0,
        "model_calls": 0,
    }
    return {**body, "manifest_content_sha256": canonical_sha256(body)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--source-batch", required=True)
    parser.add_argument("--manifest-version", required=True)
    parser.add_argument("--pipeline-config-sha256", required=True)
    parser.add_argument("--sam3-model-sha256", required=True)
    parser.add_argument("--vlm-config-sha256", required=True)
    args = parser.parse_args()
    result = build_manifest(
        args.pack_root.resolve(), args.split, args.dataset_id,
        args.source_batch, args.manifest_version,
        args.pipeline_config_sha256, args.sam3_model_sha256,
        args.vlm_config_sha256,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
