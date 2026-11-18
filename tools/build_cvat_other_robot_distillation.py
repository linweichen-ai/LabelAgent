"""Build a frame-isolated, candidate-only other_robot distillation dataset."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath


INSTRUCTION = (
    "Inspect this factory image and assess the project label other_robot exactly once. "
    "other_robot means a physically visible robot or mobile platform distinct from the "
    "camera-carrying ego robot and any person. Project convention may include a visible "
    "robot body, chassis, fork tip, or attachment when it clearly belongs to a separate "
    "industrial robot. Do not decide from color alone, and do not call a pallet, chair, "
    "wall, floor patch, manual tool, or manned vehicle other_robot. Return only compact "
    "JSON with assessments.other_robot containing evidence, confidence, and a short visual "
    "reason, plus unknown_candidates. evidence must be present, absent, or uncertain."
)


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "assessments": {
            "type": "object",
            "properties": {
                "other_robot": {
                    "type": "object",
                    "properties": {
                        "evidence": {"type": "string",
                                     "enum": ["present", "absent", "uncertain"]},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "reason": {"type": "string", "minLength": 1, "maxLength": 96},
                    },
                    "required": ["evidence", "confidence", "reason"],
                    "additionalProperties": False,
                },
            },
            "required": ["other_robot"],
            "additionalProperties": False,
        },
        "unknown_candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate_name": {"type": "string", "minLength": 1,
                                       "maxLength": 80},
                    "evidence": {"type": "string", "minLength": 1,
                                 "maxLength": 96},
                    "confidence": {"type": "number", "minimum": 0,
                                   "maximum": 1},
                },
                "required": ["candidate_name", "evidence", "confidence"],
                "additionalProperties": False,
            },
            "maxItems": 5,
        },
    },
    "required": ["assessments", "unknown_candidates"],
    "additionalProperties": False,
}


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_sha256(value: dict) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def generation_contract() -> dict:
    record = {
        "schema_version": "labelagent.vlm-teacher-contract.v1",
        "model": "qwen3vl2b-other-robot-candidate",
        "prompt_profile": "other_robot_binary_v1",
        "max_tokens": 128,
        "labels": ["other_robot"],
        "instruction": INSTRUCTION,
        "response_schema": RESPONSE_SCHEMA,
    }
    return record | {"contract_sha256": json_sha256(record)}


def frame_split(frame: int) -> str | None:
    """Use disjoint temporal blocks with boundary buffers and both label states."""
    if 19 <= frame <= 31 or 101 <= frame <= 109:
        return "dev"
    if 48 <= frame <= 66 or 126 <= frame <= 134:
        return "holdout"
    if frame in {18, 32, 47, 67, 100, 110, 125, 135}:
        return None
    return "train" if 0 <= frame <= 149 else None


def target_for_count(count: int) -> str:
    if count < 0:
        raise ValueError("instance count cannot be negative")
    if count:
        assessment = {
            "evidence": "present", "confidence": 1.0,
            "reason": "visible project-labeled robot body or attachment",
        }
    else:
        assessment = {
            "evidence": "absent", "confidence": 1.0,
            "reason": "no visible project-labeled robot region",
        }
    return json.dumps({"assessments": {"other_robot": assessment},
                       "unknown_candidates": []},
                      ensure_ascii=False, separators=(",", ":"))


def build_rows(source: Path, remote_image_root: str) -> tuple[dict[str, list[dict]], dict]:
    truth_path = source / "imported_truth.json"
    image_dir = source / "images"
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    if set(truth) != {str(index) for index in range(150)}:
        raise ValueError("expected exactly frames 0..149")
    rows = {"train": [], "dev": [], "holdout": []}
    sources = Counter()
    total_instances = 0
    for frame in range(150):
        split = frame_split(frame)
        if split is None:
            continue
        image_path = image_dir / f"frame_api_{frame:03d}.jpg"
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        shapes = [shape for shape in truth[str(frame)]
                  if shape.get("label") == "other_robot"]
        total_instances += len(shapes)
        sources.update(shape.get("cvat_metadata", {}).get("source")
                       for shape in shapes)
        target = target_for_count(len(shapes))
        rows[split].append({
            "schema_version": "labelagent.qwen3vl-other-robot-distillation-sample.v1",
            "sample_id": f"cvat-task3-frame-{frame:03d}",
            "split": "train" if split == "train" else split,
            "track": "teacher_distillation",
            "candidate_kind": "machine_teacher_output",
            "human_truth": False,
            "production_training_approved": False,
            "requires_acceptance_gate": True,
            "image_path": str(PurePosixPath(remote_image_root)
                              / image_path.name),
            "image_sha256": file_sha256(image_path),
            "instruction": INSTRUCTION,
            "target": target,
            "target_sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
            "source_sequence": "CVAT_Task3_20261026_snapshot",
            "source_frame_id": frame,
            "source_annotation_kind": "legacy_auto_snapshot",
            "source_other_robot_instances": len(shapes),
        })
    audit = {
        "source_truth_sha256": file_sha256(truth_path),
        "included_other_robot_instances": total_instances,
        "included_other_robot_sources": dict(sorted(sources.items())),
    }
    return rows, audit


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--remote-image-root", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows, audit = build_rows(args.source, args.remote_image_root)
    args.output.mkdir(parents=True)
    paths = {}
    for split, values in rows.items():
        path = args.output / f"{split}.jsonl"
        write_jsonl(path, values)
        paths[split] = path
    counts = {
        split: {
            "frames": len(values),
            "positive_frames": sum(
                row["source_other_robot_instances"] > 0 for row in values),
            "instances": sum(
                row["source_other_robot_instances"] for row in values),
        } for split, values in rows.items()
    }
    contract = generation_contract()
    (args.output / "contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    manifest = {
        "schema_version": "labelagent.qwen3vl-other-robot-distillation.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ready_for_isolated_lora_smoke",
        "track": "teacher_distillation",
        "human_truth": False,
        "production_training_approved": False,
        "requires_acceptance_gate": True,
        "test24_included": False,
        "source_remote_cvat_current_state_checked": False,
        "source_labels_are_legacy_auto": True,
        "split_policy": (
            "dev 19-31 and 101-109; holdout 48-66 and 126-134; one-frame "
            "buffers at every block boundary; all remaining frames train"),
        "counts": counts,
        "train_samples": len(rows["train"]),
        "dev_samples": len(rows["dev"]),
        "holdout_samples": len(rows["holdout"]),
        "positive_label": "other_robot",
        "dev_source_sequences": ["CVAT_Task3_20261026_snapshot"],
        "holdout_source_sequences": ["CVAT_Task3_20261026_snapshot"],
        "teacher_contract_sha256": contract["contract_sha256"],
        "generation_contract_schema_version": contract["schema_version"],
        "generation_max_tokens": contract["max_tokens"],
        "remote_image_root": args.remote_image_root,
        "train_jsonl_sha256": file_sha256(paths["train"]),
        "dev_jsonl_sha256": file_sha256(paths["dev"]),
        "holdout_jsonl_sha256": file_sha256(paths["holdout"]),
        **audit,
        "boundary": (
            "Legacy auto-source snapshot labels are machine-teacher candidates, not "
            "human truth or production accuracy evidence."),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
