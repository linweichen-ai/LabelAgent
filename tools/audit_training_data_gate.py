"""Fail-closed audit for the SAM3 train72/test24 human-review gate.

The tool never treats machine preannotations as ground truth.  A training-ready
result requires two independently exported, human-verified COCO files and an
approved per-image checklist.  The held-out test split remains evaluation-only.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_LABELS = {"pallet", "person"}
EXPECTED_SPLITS = {"train": 72, "test": 24}
EXPECTED_SEQUENCES = {
    "train": {f"{index:03d}" for index in range(9)},
    "test": {"010", "011", "012"},
}


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_checklist(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _split_manifest_images(manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result = {split: [] for split in EXPECTED_SPLITS}
    for image in manifest.get("images", []):
        split = image.get("split")
        if split not in result:
            raise ValueError(f"unsupported manifest split: {split!r}")
        result[split].append(image)
    return result


def _validate_human_coco(
    split: str,
    coco: dict[str, Any],
    expected_images: list[dict[str, Any]],
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    info = coco.get("info") if isinstance(coco.get("info"), dict) else {}
    reviewer = str(info.get("reviewer", "")).strip()
    images = coco.get("images") if isinstance(coco.get("images"), list) else []
    categories = coco.get("categories") if isinstance(coco.get("categories"), list) else []
    annotations = coco.get("annotations") if isinstance(coco.get("annotations"), list) else []

    _require(info.get("annotation_status") == "human_verified", f"{split}: annotation_status is not human_verified", errors)
    _require(bool(reviewer), f"{split}: reviewer is missing", errors)

    image_ids = [image.get("id") for image in images]
    file_names = [image.get("file_name") for image in images]
    expected_names = [image["file_name"] for image in expected_images]
    _require(len(images) == EXPECTED_SPLITS[split], f"{split}: unexpected image count", errors)
    _require(all(image_id is not None for image_id in image_ids), f"{split}: image id is missing", errors)
    _require(len(set(image_ids)) == len(image_ids), f"{split}: duplicate image ids", errors)
    _require(len(set(file_names)) == len(file_names), f"{split}: duplicate file names", errors)
    _require(set(file_names) == set(expected_names), f"{split}: COCO image names differ from frozen manifest", errors)

    reviewed = info.get("reviewed_image_ids")
    _require(isinstance(reviewed, list), f"{split}: reviewed_image_ids is missing", errors)
    if isinstance(reviewed, list):
        _require(set(reviewed) == set(image_ids), f"{split}: reviewed_image_ids does not cover every image", errors)

    category_ids = [category.get("id") for category in categories]
    category_names = {category.get("name") for category in categories}
    _require(category_names == EXPECTED_LABELS, f"{split}: categories must be exactly pallet and person", errors)
    _require(len(categories) == len(EXPECTED_LABELS), f"{split}: duplicate or extra categories", errors)
    _require(all(category_id is not None for category_id in category_ids), f"{split}: category id is missing", errors)
    _require(len(set(category_ids)) == len(category_ids), f"{split}: duplicate category ids", errors)

    annotation_ids: list[Any] = []
    valid_image_ids = set(image_ids)
    valid_category_ids = set(category_ids)
    for annotation in annotations:
        annotation_ids.append(annotation.get("id"))
        _require(annotation.get("id") is not None, f"{split}: annotation id is missing", errors)
        _require(annotation.get("image_id") in valid_image_ids, f"{split}: annotation references an unknown image", errors)
        _require(annotation.get("category_id") in valid_category_ids, f"{split}: annotation references an unknown category", errors)
        _require(bool(annotation.get("segmentation")), f"{split}: annotation has no segmentation", errors)
    _require(len(set(annotation_ids)) == len(annotation_ids), f"{split}: duplicate annotation ids", errors)

    return errors, {
        "reviewer": reviewer or None,
        "images": len(images),
        "annotations": len(annotations),
        "categories": sorted(name for name in category_names if isinstance(name, str)),
    }


def audit_training_gate(
    manifest: dict[str, Any],
    validation: dict[str, Any],
    creation_receipt: dict[str, Any],
    checklist: list[dict[str, str]],
    human_coco: dict[str, dict[str, Any] | None],
) -> dict[str, Any]:
    errors: list[str] = []
    images = _split_manifest_images(manifest)

    _require(manifest.get("image_count") == 96, "manifest image_count must be 96", errors)
    _require(manifest.get("train_images") == 72, "manifest train_images must be 72", errors)
    _require(manifest.get("test_images") == 24, "manifest test_images must be 24", errors)
    _require(manifest.get("preannotations_are_ground_truth") is False, "preannotations must not be ground truth", errors)
    _require(validation.get("verified_image_hashes") == 96, "96 image hashes were not verified", errors)
    _require(validation.get("human_annotations_present") is False, "frozen pack validation unexpectedly claims human truth", errors)
    _require(validation.get("training_authorized") is False, "frozen pack must not pre-authorize training", errors)
    _require(creation_receipt.get("status") == "complete", "CVAT creation receipt is incomplete", errors)
    _require(creation_receipt.get("train_test_isolation") is True, "CVAT receipt does not affirm split isolation", errors)
    _require(creation_receipt.get("machine_preannotations_only") is True, "CVAT receipt does not identify machine preannotations", errors)

    all_names: dict[str, set[str]] = {}
    all_hashes: dict[str, set[str]] = {}
    split_summary: dict[str, Any] = {}
    for split, expected_count in EXPECTED_SPLITS.items():
        rows = images[split]
        names = {str(row.get("file_name")) for row in rows}
        hashes = {str(row.get("source_sha256")) for row in rows}
        sequences = {str(row.get("sequence")) for row in rows}
        all_names[split], all_hashes[split] = names, hashes
        _require(len(rows) == expected_count, f"{split}: manifest row count differs", errors)
        _require(len(names) == expected_count, f"{split}: duplicate file names in manifest", errors)
        _require(len(hashes) == expected_count, f"{split}: duplicate image hashes in manifest", errors)
        _require(sequences == EXPECTED_SEQUENCES[split], f"{split}: sequence membership differs", errors)

        receipt = creation_receipt.get("tasks", {}).get(split, {}).get("readback", {})
        labels = {row.get("name") for row in receipt.get("labels", [])}
        label_types = {row.get("type") for row in receipt.get("labels", [])}
        _require(creation_receipt.get("tasks", {}).get(split, {}).get("creation_completed") is True, f"{split}: CVAT task creation is incomplete", errors)
        _require(isinstance(creation_receipt.get("tasks", {}).get(split, {}).get("task_id"), int), f"{split}: CVAT task id is missing", errors)
        _require(receipt.get("subset") == split, f"{split}: CVAT subset differs", errors)
        _require(receipt.get("size") == expected_count, f"{split}: CVAT task size differs", errors)
        _require(receipt.get("frame_count") == expected_count, f"{split}: CVAT frame count differs", errors)
        _require(labels == EXPECTED_LABELS, f"{split}: CVAT labels differ", errors)
        _require(label_types == {"mask"}, f"{split}: CVAT labels must be mask labels", errors)
        _require(receipt.get("preannotations_are_human_truth") is False, f"{split}: CVAT preannotations mislabeled as truth", errors)
        _require(not receipt.get("validation_errors"), f"{split}: CVAT creation validation errors exist", errors)
        split_summary[split] = {
            "manifest_images": len(rows),
            "sequences": sorted(sequences),
            "cvat_task_id": creation_receipt.get("tasks", {}).get(split, {}).get("task_id"),
            "cvat_job_ids": receipt.get("job_ids", []),
            "machine_preannotations": receipt.get("shape_count"),
        }

    _require(not (all_names["train"] & all_names["test"]), "train/test file-name leakage", errors)
    _require(not (all_hashes["train"] & all_hashes["test"]), "train/test image-content leakage", errors)

    checklist_names = {row.get("file_name", "") for row in checklist}
    expected_checklist_names = all_names["train"] | all_names["test"]
    _require(len(checklist) == 96, "review checklist must contain 96 rows", errors)
    _require(checklist_names == expected_checklist_names, "review checklist does not match frozen images", errors)
    approved_rows = [row for row in checklist if row.get("review_status", "").strip().lower() == "approved"]
    approved_with_reviewer = [row for row in approved_rows if row.get("reviewer", "").strip()]

    human_summaries: dict[str, Any] = {}
    human_errors: list[str] = []
    present = {split: human_coco.get(split) is not None for split in EXPECTED_SPLITS}
    for split in EXPECTED_SPLITS:
        coco = human_coco.get(split)
        if coco is not None:
            split_errors, summary = _validate_human_coco(split, coco, images[split])
            human_errors.extend(split_errors)
            human_summaries[split] = summary

    checklist_ready = len(approved_rows) == 96 and len(approved_with_reviewer) == 96
    human_exports_ready = all(present.values()) and not human_errors
    training_authorized = not errors and checklist_ready and human_exports_ready
    blockers: list[str] = []
    if not checklist_ready:
        blockers.append("all 96 checklist rows require approved status and a reviewer")
    if not all(present.values()):
        blockers.append("both human-verified train and test COCO exports are required")
    blockers.extend(human_errors)
    blockers.extend(errors)

    return {
        "schema_version": "labelagent.training-data-gate.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ready_for_isolated_training" if training_authorized else "blocked_waiting_for_human_review",
        "training_authorized": training_authorized,
        "test_split_policy": "evaluation_only_never_train",
        "machine_preannotations_are_ground_truth": False,
        "split_isolation": {
            "file_name_overlap": sorted(all_names["train"] & all_names["test"]),
            "source_sha256_overlap": sorted(all_hashes["train"] & all_hashes["test"]),
        },
        "splits": split_summary,
        "checklist": {
            "rows": len(checklist),
            "approved": len(approved_rows),
            "approved_with_reviewer": len(approved_with_reviewer),
            "pending_or_invalid": len(checklist) - len(approved_with_reviewer),
        },
        "human_exports": {
            "present": present,
            "summaries": human_summaries,
        },
        "blockers": blockers,
        "integrity_errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--creation-receipt", required=True)
    parser.add_argument("--checklist", required=True)
    parser.add_argument("--train-human-coco")
    parser.add_argument("--test-human-coco")
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()

    paths = {
        "manifest": Path(args.manifest).resolve(),
        "validation": Path(args.validation).resolve(),
        "creation_receipt": Path(args.creation_receipt).resolve(),
        "checklist": Path(args.checklist).resolve(),
    }
    report = audit_training_gate(
        read_json(paths["manifest"]),
        read_json(paths["validation"]),
        read_json(paths["creation_receipt"]),
        read_checklist(paths["checklist"]),
        {
            "train": read_json(args.train_human_coco) if args.train_human_coco else None,
            "test": read_json(args.test_human_coco) if args.test_human_coco else None,
        },
    )
    report["input_sha256"] = {name: sha256_file(path) for name, path in paths.items()}
    if args.train_human_coco:
        report["input_sha256"]["train_human_coco"] = sha256_file(args.train_human_coco)
    if args.test_human_coco:
        report["input_sha256"]["test_human_coco"] = sha256_file(args.test_human_coco)
    report["audit_code_sha256"] = sha256_file(__file__)

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 2 if args.require_ready and not report["training_authorized"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
