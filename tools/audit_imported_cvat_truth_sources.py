"""Audit CVAT-imported masks without confusing machine shapes with human truth."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any


KNOWN_SOURCES = {"manual", "auto"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _frame_items(value: Any) -> list[tuple[int, Any]]:
    if isinstance(value, dict):
        items: list[tuple[int, Any]] = []
        for raw_key, frame in value.items():
            try:
                frame_id = int(raw_key)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"frame key is not an integer: {raw_key!r}") from exc
            if frame_id < 0:
                raise ValueError(f"frame key is negative: {frame_id}")
            items.append((frame_id, frame))
        items.sort(key=lambda item: item[0])
        if len({item[0] for item in items}) != len(items):
            raise ValueError("duplicate frame ids after integer normalization")
        return items
    if isinstance(value, list):
        return list(enumerate(value))
    raise ValueError("top-level JSON must be a frame-id object or frame list")


def _valid_point(point: Any) -> bool:
    return (
        isinstance(point, list)
        and len(point) == 2
        and all(isinstance(value, (int, float)) and math.isfinite(value) for value in point)
    )


def _validate_shape(shape: Any, frame_id: int, shape_index: int) -> list[str]:
    prefix = f"frame {frame_id} shape {shape_index}"
    if not isinstance(shape, dict):
        return [f"{prefix}: shape is not an object"]
    errors: list[str] = []
    label = shape.get("label")
    if not isinstance(label, str) or not label.strip():
        errors.append(f"{prefix}: label is missing")
    points = shape.get("points")
    if not isinstance(points, list) or len(points) < 3 or not all(
        _valid_point(point) for point in points
    ):
        errors.append(f"{prefix}: points are not a valid polygon")
    parts = shape.get("parts", [])
    if not isinstance(parts, list):
        errors.append(f"{prefix}: parts is not a list")
    else:
        for part_index, part in enumerate(parts):
            if not isinstance(part, dict):
                errors.append(f"{prefix}: part {part_index} is not an object")
                continue
            part_points = part.get("points")
            if not isinstance(part_points, list) or len(part_points) < 3 or not all(
                _valid_point(point) for point in part_points
            ):
                errors.append(f"{prefix}: part {part_index} has invalid points")
    return errors


def _reviewer_approval(
    receipt_path: Path | None,
    source_digest: str,
    candidate_frames: list[int],
) -> tuple[bool, list[int], list[str]]:
    if receipt_path is None:
        return False, [], ["reviewer signoff receipt is absent"]
    errors: list[str] = []
    value = read_json(receipt_path)
    if not isinstance(value, dict):
        return False, [], ["reviewer signoff receipt must be an object"]
    if value.get("schema_version") != "labelagent.cvat-human-truth-signoff.v1":
        errors.append("reviewer signoff schema differs")
    if value.get("source_sha256") != source_digest:
        errors.append("reviewer signoff source hash differs")
    if value.get("approved") is not True:
        errors.append("reviewer signoff is not approved")
    reviewer = value.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer.strip():
        errors.append("reviewer identity is absent")
    approved = value.get("approved_frame_ids")
    if not isinstance(approved, list) or not all(
        isinstance(frame_id, int) and not isinstance(frame_id, bool) for frame_id in approved
    ):
        errors.append("approved_frame_ids is invalid")
        approved_ids: list[int] = []
    else:
        approved_ids = sorted(set(approved))
        if len(approved_ids) != len(approved):
            errors.append("approved_frame_ids contains duplicates")
        outside = sorted(set(approved_ids) - set(candidate_frames))
        if outside:
            errors.append(f"reviewer approved non-human-only frames: {outside}")
    return not errors and bool(approved_ids), approved_ids, errors


def audit(source_path: Path, reviewer_receipt: Path | None = None) -> dict[str, Any]:
    source_path = source_path.resolve()
    source_digest = sha256(source_path)
    value = read_json(source_path)
    try:
        frames = _frame_items(value)
    except ValueError as exc:
        return {
            "schema_version": "labelagent.imported-cvat-truth-source-audit.v1",
            "status": "failed_closed",
            "errors": [str(exc)],
            "source_sha256": source_digest,
        }

    errors: list[str] = []
    source_counts: Counter[str] = Counter()
    label_source_counts: Counter[tuple[str, str]] = Counter()
    frame_classes: Counter[str] = Counter()
    frame_details: list[dict[str, Any]] = []
    all_manual_frames: list[int] = []
    mixed_frames: list[int] = []
    all_auto_frames: list[int] = []
    unknown_source_frames: list[int] = []
    empty_frames: list[int] = []
    manual_instance_frames: list[int] = []

    for frame_id, raw_shapes in frames:
        if not isinstance(raw_shapes, list):
            errors.append(f"frame {frame_id}: shapes is not a list")
            shapes: list[Any] = []
        else:
            shapes = raw_shapes
        sources: list[str] = []
        labels: Counter[str] = Counter()
        manual_shapes = 0
        for shape_index, shape in enumerate(shapes):
            errors.extend(_validate_shape(shape, frame_id, shape_index))
            if not isinstance(shape, dict):
                continue
            label = shape.get("label") if isinstance(shape.get("label"), str) else "<invalid>"
            metadata = shape.get("cvat_metadata")
            source = metadata.get("source") if isinstance(metadata, dict) else None
            if not isinstance(source, str) or not source.strip():
                source = "<missing>"
            else:
                source = source.strip().lower()
            sources.append(source)
            labels[label] += 1
            source_counts[source] += 1
            label_source_counts[(label, source)] += 1
            if source == "manual":
                manual_shapes += 1

        source_set = set(sources)
        if not shapes:
            frame_class = "empty"
            empty_frames.append(frame_id)
        elif source_set == {"manual"}:
            frame_class = "all_manual"
            all_manual_frames.append(frame_id)
        elif source_set == {"auto"}:
            frame_class = "all_auto"
            all_auto_frames.append(frame_id)
        elif source_set <= KNOWN_SOURCES:
            frame_class = "mixed_manual_auto"
            mixed_frames.append(frame_id)
        else:
            frame_class = "contains_unknown_source"
            unknown_source_frames.append(frame_id)
            errors.append(
                f"frame {frame_id}: unknown sources {sorted(source_set - KNOWN_SOURCES)}"
            )
        if manual_shapes:
            manual_instance_frames.append(frame_id)
        frame_classes[frame_class] += 1
        frame_details.append(
            {
                "frame_id": frame_id,
                "shape_count": len(shapes),
                "manual_shape_count": manual_shapes,
                "auto_shape_count": sources.count("auto"),
                "unknown_shape_count": len(sources) - manual_shapes - sources.count("auto"),
                "source_class": frame_class,
                "labels": dict(sorted(labels.items())),
            }
        )

    approved, approved_ids, reviewer_errors = _reviewer_approval(
        reviewer_receipt.resolve() if reviewer_receipt else None,
        source_digest,
        all_manual_frames,
    )
    training_eligible = approved and bool(all_manual_frames) and not errors
    return {
        "schema_version": "labelagent.imported-cvat-truth-source-audit.v1",
        "status": "passed_source_inventory" if not errors else "failed_closed",
        "errors": errors,
        "source": {
            "name": source_path.name,
            "sha256": source_digest,
            "top_level_kind": "frame_id_object" if isinstance(value, dict) else "frame_list",
        },
        "summary": {
            "frame_count": len(frames),
            "shape_count": sum(source_counts.values()),
            "source_counts": dict(sorted(source_counts.items())),
            "frame_class_counts": dict(sorted(frame_classes.items())),
            "label_source_counts": {
                f"{label}|{source}": count
                for (label, source), count in sorted(label_source_counts.items())
            },
        },
        "frame_sets": {
            "all_manual": all_manual_frames,
            "mixed_manual_auto": mixed_frames,
            "all_auto": all_auto_frames,
            "contains_unknown_source": unknown_source_frames,
            "empty": empty_frames,
            "contains_manual_instances": manual_instance_frames,
        },
        "review_gate": {
            "receipt_present": reviewer_receipt is not None,
            "receipt_errors": reviewer_errors,
            "approved_frame_ids": approved_ids,
        },
        "decision": {
            "whole_frame_human_truth_candidates": len(all_manual_frames),
            "manual_instance_candidates": source_counts.get("manual", 0),
            "whole_frame_training_eligible": training_eligible,
            "manual_instance_training_eligible": False,
            "reason": (
                "No non-empty frame contains only manual shapes; every manual shape shares its frame "
                "with auto shapes. The manual shapes are instance-level review candidates, not complete "
                "whole-frame semantic truth, and no separate reviewer signoff was supplied."
                if not all_manual_frames
                else "Human-only frames remain blocked until a hash-bound reviewer signoff approves them."
            ),
            "safe_next_step": (
                "Export or review train-only masks with explicit completion/signoff. Keep auto shapes as "
                "candidates and keep test24 evaluation-only."
            ),
        },
        "frames": frame_details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--reviewer-receipt", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ValueError("choose a fresh output path")
    result = audit(args.source, args.reviewer_receipt)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))
    print(json.dumps(result["decision"], ensure_ascii=False))
    return 0 if result["status"] == "passed_source_inventory" else 1


if __name__ == "__main__":
    raise SystemExit(main())
