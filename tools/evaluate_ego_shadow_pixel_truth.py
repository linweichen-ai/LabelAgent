#!/usr/bin/env python3
"""Read-only pixel-truth gate for camera-aware ego shadow diagnostics.

The evaluator accepts only explicitly human-confirmed binary truth masks.  It
never writes LabelAgent state and never promotes a shadow result to a training
or mergeable candidate.  Reports contain hashes and metrics, not image paths or
pixel geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


MANIFEST_SCHEMA_VERSION = "labelagent.ego_pixel_truth_manifest.v2"
REPORT_SCHEMA_VERSION = "labelagent.ego_pixel_truth_report.v2"
REVIEW_PACKET_SCHEMA_VERSION = "labelagent.ego_pixel_truth_review_packet.v1"
RECEIPT_SCHEMA_VERSION = "labelagent.ego_human_pixel_truth_receipt.v2"
CAMERAS = {f"camera{index}" for index in range(4)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_sha256(value) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdefABCDEF" for character in value))


def _bounded_relative_path(root: Path, value, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"{field} must be relative to the manifest")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{field} escapes the manifest directory") from error
    if not resolved.is_file():
        raise ValueError(f"{field} does not exist")
    return resolved


def _binary_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"))
    values = set(int(value) for value in np.unique(array))
    if not values <= {0, 1, 255}:
        raise ValueError(f"mask is not binary: {path.name}")
    return array > 0


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _require_human_truth(manifest: dict, root: Path) -> tuple[dict, str]:
    provenance = manifest.get("truth_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("truth_provenance is required")
    receipt = provenance.get("confirmation_receipt_sha256")
    if (provenance.get("kind") != "human_pixel_truth"
            or provenance.get("human_confirmed") is not True
            or provenance.get("assistant_generated") is not False
            or provenance.get("machine_generated") is not False
            or not _valid_sha256(receipt)):
        raise ValueError("pixel truth must have an explicit human confirmation receipt")
    receipt_path = _bounded_relative_path(
        root, provenance.get("confirmation_receipt"), "confirmation_receipt")
    actual_sha256 = _sha256(receipt_path)
    if actual_sha256.casefold() != receipt.casefold():
        raise ValueError("human confirmation receipt hash mismatch")
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (not isinstance(payload, dict)
            or payload.get("schema_version") != RECEIPT_SCHEMA_VERSION
            or payload.get("human_confirmed") is not True
            or payload.get("assistant_generated") is not False
            or payload.get("machine_generated") is not False
            or not isinstance(payload.get("reviewed_masks"), list)):
        raise ValueError("human confirmation receipt content is invalid")
    return payload, actual_sha256


def _require_candidate_packet(manifest: dict, root: Path) -> tuple[dict, str, Path]:
    provenance = manifest.get("candidate_provenance")
    required = {"kind", "review_queue", "review_queue_sha256"}
    if not isinstance(provenance, dict) or set(provenance) != required:
        raise ValueError("candidate_provenance must bind exactly one review queue")
    expected_sha256 = provenance.get("review_queue_sha256")
    if (provenance.get("kind") != "machine_candidate_review_packet"
            or not _valid_sha256(expected_sha256)):
        raise ValueError("candidate review queue provenance is invalid")
    queue_path = _bounded_relative_path(
        root, provenance.get("review_queue"), "candidate review_queue")
    actual_sha256 = _sha256(queue_path)
    if actual_sha256.casefold() != expected_sha256.casefold():
        raise ValueError("candidate review queue hash mismatch")
    packet = json.loads(queue_path.read_text(encoding="utf-8"))
    if (not isinstance(packet, dict)
            or packet.get("schema_version") != REVIEW_PACKET_SCHEMA_VERSION
            or packet.get("candidate_only") is not True
            or packet.get("human_truth") is not False
            or packet.get("human_confirmed") is not False
            or packet.get("approved_for_training") is not False
            or packet.get("production_deployed") is not False
            or packet.get("formal_state_modified") is not False
            or not isinstance(packet.get("samples"), list)):
        raise ValueError("candidate review queue content is invalid")
    mapping = {}
    for item in packet["samples"]:
        sample_id = item.get("sample_id") if isinstance(item, dict) else None
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError("candidate review queue sample entry is invalid")
        sample_id = sample_id.strip()
        if sample_id in mapping:
            raise ValueError("candidate review queue sample IDs must be unique")
        mapping[sample_id] = item
    return mapping, actual_sha256, queue_path.parent


def _verify_receipt(rows: list[dict], receipt: dict,
                    candidate_queue_sha256: str) -> None:
    if receipt.get("candidate_review_queue_sha256") != candidate_queue_sha256:
        raise ValueError("human confirmation receipt candidate queue hash mismatch")
    reviewed = receipt["reviewed_masks"]
    mapping = {}
    for item in reviewed:
        if not isinstance(item, dict) or not isinstance(item.get("sample_id"), str):
            raise ValueError("human confirmation receipt mask entry is invalid")
        if item["sample_id"] in mapping:
            raise ValueError("human confirmation receipt sample IDs must be unique")
        mapping[item["sample_id"]] = item
    if set(mapping) != {row["sample_id"] for row in rows}:
        raise ValueError("human confirmation receipt sample scope mismatch")
    for row in rows:
        item = mapping[row["sample_id"]]
        expected = {
            "candidate_mask_sha256": row["candidate_sha256"],
            "ego_truth_sha256": row["ego_truth_sha256"],
            "person_truth_sha256": row["person_truth_sha256"],
            "floor_truth_sha256": row["floor_truth_sha256"],
        }
        if any(item.get(key) != value for key, value in expected.items()):
            raise ValueError(
                f"human confirmation receipt mask hash mismatch: {row['sample_id']}")


def _thresholds(manifest: dict) -> dict:
    values = manifest.get("gates")
    if not isinstance(values, dict):
        raise ValueError("gates are required")
    required = {
        "minimum_mean_iou", "minimum_sample_iou",
        "maximum_person_overlap", "maximum_floor_overlap",
        "minimum_samples_per_camera", "required_cameras",
    }
    if set(values) != required:
        raise ValueError("gates must contain exactly the documented fields")
    for name in ("minimum_mean_iou", "minimum_sample_iou",
                 "maximum_person_overlap", "maximum_floor_overlap"):
        value = values[name]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not np.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"invalid gate threshold: {name}")
    count = values["minimum_samples_per_camera"]
    if type(count) is not int or count <= 0:
        raise ValueError("minimum_samples_per_camera must be a positive integer")
    cameras = values["required_cameras"]
    if (not isinstance(cameras, list) or not cameras
            or len(cameras) != len(set(cameras))
            or any(camera not in CAMERAS for camera in cameras)):
        raise ValueError("required_cameras must be unique supported cameras")
    return values


def _evaluate_sample(root: Path, row: dict, packet_row: dict,
                     packet_root: Path) -> dict:
    if not isinstance(row, dict):
        raise ValueError("each sample must be an object")
    sample_id, camera = row.get("sample_id"), row.get("camera")
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError("sample_id is required")
    if camera not in CAMERAS:
        raise ValueError(f"unsupported camera for sample {sample_id}")
    candidate_path = _bounded_relative_path(root, row.get("candidate_mask"),
                                            "candidate_mask")
    truth_path = _bounded_relative_path(root, row.get("ego_truth_mask"),
                                        "ego_truth_mask")
    person_path = _bounded_relative_path(root, row.get("person_truth_mask"),
                                         "person_truth_mask")
    floor_path = _bounded_relative_path(root, row.get("floor_truth_mask"),
                                        "floor_truth_mask")
    paths = (candidate_path, truth_path, person_path, floor_path)
    if len(set(paths)) != len(paths):
        raise ValueError(f"candidate and truth mask paths must be distinct for {sample_id}")
    if (packet_row.get("camera") != camera
            or packet_row.get("candidate_only") is not True
            or packet_row.get("human_truth") is not False
            or packet_row.get("approved_for_training") is not False
            or packet_row.get("eligible_for_candidate_merge") is not False
            or packet_row.get("state_written") is not False
            or packet_row.get("candidate_observed") is not True
            or type(packet_row.get("candidate_available")) is not bool
            or not isinstance(packet_row.get("status"), str)
            or not packet_row["status"].strip()
            or not _valid_sha256(packet_row.get("candidate_mask_sha256"))):
        raise ValueError(f"candidate review queue binding is invalid for {sample_id}")
    packet_candidate_path = _bounded_relative_path(
        packet_root, packet_row.get("candidate_mask"), "packet candidate_mask")
    if packet_candidate_path != candidate_path:
        raise ValueError(f"candidate review queue path mismatch for {sample_id}")
    candidate_sha256 = _sha256(candidate_path)
    if candidate_sha256.casefold() != packet_row["candidate_mask_sha256"].casefold():
        raise ValueError(f"candidate review queue mask hash mismatch for {sample_id}")
    candidate = _binary_mask(candidate_path)
    truth = _binary_mask(truth_path)
    person = _binary_mask(person_path)
    floor = _binary_mask(floor_path)
    shapes = {mask.shape for mask in (candidate, truth, person, floor)}
    if len(shapes) != 1:
        raise ValueError(f"mask dimensions differ for sample {sample_id}")
    candidate_area, truth_area = int(candidate.sum()), int(truth.sum())
    if not candidate_area or not truth_area:
        raise ValueError(f"candidate and ego truth must be non-empty for {sample_id}")
    intersection = int(np.logical_and(candidate, truth).sum())
    union = int(np.logical_or(candidate, truth).sum())
    return {
        "sample_id": sample_id.strip(),
        "camera": camera,
        "width": int(candidate.shape[1]),
        "height": int(candidate.shape[0]),
        "candidate_sha256": candidate_sha256,
        "candidate_status": packet_row["status"],
        "candidate_available": packet_row["candidate_available"],
        "ego_truth_sha256": _sha256(truth_path),
        "person_truth_sha256": _sha256(person_path),
        "floor_truth_sha256": _sha256(floor_path),
        "iou": round(_ratio(intersection, union), 6),
        "precision": round(_ratio(intersection, candidate_area), 6),
        "recall": round(_ratio(intersection, truth_area), 6),
        "person_overlap": round(
            _ratio(int(np.logical_and(candidate, person).sum()), candidate_area), 6),
        "floor_overlap": round(
            _ratio(int(np.logical_and(candidate, floor).sum()), candidate_area), 6),
    }


def evaluate_manifest(path: Path) -> dict:
    manifest_path = path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported ego pixel-truth manifest schema")
    if manifest.get("dataset_split") not in {"validation", "test"}:
        raise ValueError("pixel-truth gate is evaluation-only")
    receipt, receipt_sha256 = _require_human_truth(manifest, manifest_path.parent)
    packet_rows, candidate_queue_sha256, packet_root = _require_candidate_packet(
        manifest, manifest_path.parent)
    gates = _thresholds(manifest)
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("samples must be a non-empty list")
    rows = []
    for sample in samples:
        sample_id = sample.get("sample_id") if isinstance(sample, dict) else None
        normalized_id = sample_id.strip() if isinstance(sample_id, str) else None
        packet_row = packet_rows.get(normalized_id)
        if packet_row is None:
            raise ValueError("candidate review queue sample scope mismatch")
        rows.append(_evaluate_sample(
            manifest_path.parent, sample, packet_row, packet_root))
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("sample_id values must be unique")
    if set(packet_rows) != set(ids):
        raise ValueError("candidate review queue sample scope mismatch")
    _verify_receipt(rows, receipt, candidate_queue_sha256)

    per_camera = {}
    gate_reasons = []
    for camera in gates["required_cameras"]:
        current = [row for row in rows if row["camera"] == camera]
        if len(current) < gates["minimum_samples_per_camera"]:
            gate_reasons.append(f"{camera}:insufficient_samples")
        if current:
            per_camera[camera] = {
                "samples": len(current),
                "mean_iou": round(float(np.mean([row["iou"] for row in current])), 6),
                "minimum_iou": min(row["iou"] for row in current),
                "maximum_person_overlap": max(row["person_overlap"] for row in current),
                "maximum_floor_overlap": max(row["floor_overlap"] for row in current),
            }
    mean_iou = round(float(np.mean([row["iou"] for row in rows])), 6)
    minimum_iou = min(row["iou"] for row in rows)
    maximum_person = max(row["person_overlap"] for row in rows)
    maximum_floor = max(row["floor_overlap"] for row in rows)
    if mean_iou < gates["minimum_mean_iou"]:
        gate_reasons.append("mean_iou_below_gate")
    if minimum_iou < gates["minimum_sample_iou"]:
        gate_reasons.append("sample_iou_below_gate")
    if maximum_person > gates["maximum_person_overlap"]:
        gate_reasons.append("person_overlap_above_gate")
    if maximum_floor > gates["maximum_floor_overlap"]:
        gate_reasons.append("floor_overlap_above_gate")
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "manifest_sha256": _sha256(manifest_path),
        "candidate_review_queue_sha256": candidate_queue_sha256,
        "human_confirmation_receipt_sha256": receipt_sha256,
        "dataset_split": manifest["dataset_split"],
        "evaluation_only": True,
        "human_truth": True,
        "assistant_generated_truth": False,
        "machine_generated_truth": False,
        "approved_for_training": False,
        "eligible_for_candidate_merge": False,
        "samples": rows,
        "summary": {
            "sample_count": len(rows),
            "mean_iou": mean_iou,
            "minimum_iou": minimum_iou,
            "maximum_person_overlap": maximum_person,
            "maximum_floor_overlap": maximum_floor,
            "per_camera": per_camera,
        },
        "gates": gates,
        "gate": {
            "passed": not gate_reasons,
            "reasons": gate_reasons,
            "production_deployed": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = evaluate_manifest(arguments.manifest)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0 if report["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
