"""Build a local-only positive-instance candidate manifest from manual CVAT shapes."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

try:
    from tools.audit_imported_cvat_truth_sources import (
        _frame_items,
        _validate_shape,
        read_json,
        sha256,
    )
except ModuleNotFoundError:  # Direct ``python tools/...py`` execution.
    from audit_imported_cvat_truth_sources import (  # type: ignore[no-redef]
        _frame_items,
        _validate_shape,
        read_json,
        sha256,
    )


def _frame_manifest(value: Any) -> dict[int, dict[str, Any]]:
    if not isinstance(value, dict) or not isinstance(value.get("frames"), list):
        raise ValueError("import manifest must contain a frames list")
    result: dict[int, dict[str, Any]] = {}
    for row in value["frames"]:
        if not isinstance(row, dict) or not isinstance(row.get("frame"), int):
            raise ValueError("import manifest frame row is invalid")
        frame_id = row["frame"]
        if frame_id in result:
            raise ValueError(f"duplicate import manifest frame: {frame_id}")
        result[frame_id] = row
    return result


def build(
    truth_path: Path,
    import_manifest_path: Path,
    image_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    truth_path = truth_path.resolve()
    import_manifest_path = import_manifest_path.resolve()
    image_root = image_root.resolve()
    truth = read_json(truth_path)
    manifest = read_json(import_manifest_path)
    frames = _frame_items(truth)
    frame_rows = _frame_manifest(manifest)
    errors: list[str] = []
    candidates: list[dict[str, Any]] = []
    labels: Counter[str] = Counter()
    verified_images: dict[int, str] = {}

    for frame_id, shapes in frames:
        if not isinstance(shapes, list):
            errors.append(f"frame {frame_id}: shapes is not a list")
            continue
        manual_shapes = []
        for shape_index, shape in enumerate(shapes):
            if not isinstance(shape, dict):
                continue
            metadata = shape.get("cvat_metadata")
            if isinstance(metadata, dict) and metadata.get("source") == "manual":
                shape_errors = _validate_shape(shape, frame_id, shape_index)
                errors.extend(shape_errors)
                if not shape_errors:
                    manual_shapes.append((shape_index, shape))
        if not manual_shapes:
            continue
        row = frame_rows.get(frame_id)
        if row is None:
            errors.append(f"frame {frame_id}: absent from import manifest")
            continue
        filename = row.get("file")
        expected_digest = row.get("sha256")
        if not isinstance(filename, str) or Path(filename).name != filename:
            errors.append(f"frame {frame_id}: unsafe image filename")
            continue
        if not isinstance(expected_digest, str) or len(expected_digest) != 64:
            errors.append(f"frame {frame_id}: invalid image digest")
            continue
        image_path = image_root / filename
        if not image_path.is_file():
            errors.append(f"frame {frame_id}: image is absent")
            continue
        actual_digest = sha256(image_path)
        if actual_digest != expected_digest:
            errors.append(f"frame {frame_id}: image digest differs")
            continue
        verified_images[frame_id] = actual_digest
        for shape_index, shape in manual_shapes:
            label = shape["label"]
            shape_id = shape.get("id")
            if not isinstance(shape_id, int):
                errors.append(f"frame {frame_id} shape {shape_index}: id is invalid")
                continue
            labels[label] += 1
            candidates.append(
                {
                    "candidate_id": f"{frame_id}:{shape_id}",
                    "frame_id": frame_id,
                    "image": {
                        "file": filename,
                        "sha256": actual_digest,
                        "source_name": row.get("source_name"),
                        "source_timestamp_ns": row.get("source_timestamp_ns"),
                    },
                    "shape": {
                        "id": shape_id,
                        "label": label,
                        "points": shape["points"],
                        "holes": shape.get("holes", []),
                        "parts": shape.get("parts", []),
                        "cvat_metadata": shape.get("cvat_metadata", {}),
                    },
                    "supervision": {
                        "positive_instance_only": True,
                        "outside_instance": "ignore",
                        "full_frame_semantic_loss_allowed": False,
                        "negative_sampling_allowed": False,
                        "review_status": "pending_independent_signoff",
                    },
                }
            )

    candidate_ids = [item["candidate_id"] for item in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        errors.append("candidate ids are not unique")
    private = {
        "schema_version": "labelagent.cvat-manual-instance-candidates.v1",
        "status": "prepared_not_training_authorized" if not errors else "failed_closed",
        "errors": errors,
        "source": {
            "truth_name": truth_path.name,
            "truth_sha256": sha256(truth_path),
            "import_manifest_name": import_manifest_path.name,
            "import_manifest_sha256": sha256(import_manifest_path),
        },
        "policy": {
            "human_truth_claimed": False,
            "training_authorized": False,
            "test24_included": False,
            "auto_shapes_included": False,
            "outside_manual_instance": "ignore",
            "full_frame_semantic_loss_allowed": False,
            "negative_sampling_allowed": False,
            "required_next_gate": "hash_bound_independent_instance_review",
        },
        "summary": {
            "candidate_count": len(candidates),
            "frame_count": len(verified_images),
            "label_counts": dict(sorted(labels.items())),
            "verified_image_count": len(verified_images),
        },
        "candidates": candidates,
    }
    return private, {
        "schema_version": "labelagent.cvat-manual-instance-candidate-receipt.v1",
        "status": private["status"],
        "errors": errors,
        "source": private["source"],
        "policy": private["policy"],
        "summary": private["summary"],
        "private_manifest_sha256": None,
        "private_geometry_published": False,
        "raw_images_published": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--import-manifest", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--private-output", required=True, type=Path)
    parser.add_argument("--receipt-output", required=True, type=Path)
    args = parser.parse_args()
    private_output = args.private_output.resolve()
    receipt_output = args.receipt_output.resolve()
    if private_output.exists() or receipt_output.exists():
        raise ValueError("choose fresh output paths")
    private, receipt = build(args.truth, args.import_manifest, args.image_root)
    private_output.parent.mkdir(parents=True, exist_ok=True)
    private_output.write_text(
        json.dumps(private, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    receipt["private_manifest_sha256"] = sha256(private_output)
    receipt_output.parent.mkdir(parents=True, exist_ok=True)
    receipt_output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False))
    return 0 if private["status"] == "prepared_not_training_authorized" else 1


if __name__ == "__main__":
    raise SystemExit(main())
