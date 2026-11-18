"""Build a balanced assistant-reviewed candidate set for isolated micro-tuning."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_spec(spec: dict) -> list[dict]:
    samples = spec.get("samples")
    if (spec.get("schema_version") != "labelagent.other-robot-contrastive-spec.v1"
            or spec.get("human_truth") is not False
            or spec.get("candidate_only") is not True
            or spec.get("production_training_approved") is not False
            or not isinstance(samples, list) or not samples):
        raise ValueError("contrastive candidate spec boundary differs")
    ids = []
    evidence_values = set()
    for item in samples:
        review = item.get("assistant_review_candidate")
        evidence = (review or {}).get("expected_evidence")
        if (not item.get("sample_id") or not item.get("local_image_path")
                or not item.get("remote_image_path") or not item.get("image_sha256")
                or not isinstance(review, dict)
                or review.get("review_kind") != "assistant_visual_candidate"
                or review.get("human_truth") is not False
                or evidence not in {"present", "absent"}
                or not review.get("reason")
                or not isinstance(item.get("training_reason"), str)
                or not 1 <= len(item["training_reason"]) <= 96):
            raise ValueError("contrastive candidate sample boundary differs")
        ids.append(item["sample_id"])
        evidence_values.add(evidence)
    if len(ids) != len(set(ids)):
        raise ValueError("contrastive sample IDs must be unique")
    if evidence_values != {"present", "absent"}:
        raise ValueError("contrastive candidates require both evidence states")
    return samples


def build_rows(samples: list[dict], repeat: int, instruction: str) -> list[dict]:
    if not 1 <= repeat <= 16:
        raise ValueError("contrastive repeat is out of bounds")
    rows = []
    for item in samples:
        local_path = Path(item["local_image_path"])
        if not local_path.is_file() or file_sha256(local_path) != item["image_sha256"]:
            raise ValueError(f"contrastive image hash differs: {item['sample_id']}")
        evidence = item["assistant_review_candidate"]["expected_evidence"]
        target = json.dumps({
            "assessments": {"other_robot": {
                "evidence": evidence, "confidence": 1.0,
                "reason": item["training_reason"],
            }}, "unknown_candidates": [],
        }, ensure_ascii=False, separators=(",", ":"))
        for repetition in range(repeat):
            rows.append({
                "schema_version": "labelagent.qwen3vl-other-robot-distillation-sample.v1",
                "sample_id": f"{item['sample_id']}-repeat-{repetition:02d}",
                "split": "train", "track": "teacher_distillation",
                "candidate_kind": "assistant_review_candidate",
                "human_truth": False, "production_training_approved": False,
                "requires_acceptance_gate": True,
                "image_path": item["remote_image_path"],
                "image_sha256": item["image_sha256"],
                "instruction": instruction, "target": target,
                "target_sha256": hashlib.sha256(target.encode()).hexdigest(),
                "source_sequence": item.get("source_sequence"),
                "source_frame_id": item.get("source_frame_id"),
                "source_annotation_kind": "assistant_review_candidate",
                "source_other_robot_instances": 1 if evidence == "present" else 0,
                "assistant_review_candidate": item["assistant_review_candidate"],
                "contrastive_repeat_index": repetition,
            })
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False,
                                        separators=(",", ":")) + "\n"
                            for row in rows), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dataset", required=True, type=Path)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeat", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    base_manifest_path = args.base_dataset / "manifest.json"
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    contract_path = args.base_dataset / "contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if (base_manifest.get("human_truth") is not False
            or base_manifest.get("production_training_approved") is not False
            or base_manifest.get("test24_included") is not False):
        raise ValueError("base dataset boundary differs")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    samples = validate_spec(spec)
    rows = build_rows(samples, args.repeat, contract["instruction"])
    args.output.mkdir(parents=True)
    train_path = args.output / "train.jsonl"
    write_jsonl(train_path, rows)
    shutil.copy2(contract_path, args.output / "contract.json")
    evidence = [item["assistant_review_candidate"]["expected_evidence"]
                for item in samples]
    manifest = {
        "schema_version": "labelagent.qwen3vl-other-robot-contrastive.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ready_for_isolated_lora_smoke",
        "track": "teacher_distillation",
        "human_truth": False,
        "production_training_approved": False,
        "requires_acceptance_gate": True,
        "test24_included": False,
        "source_labels_are_legacy_auto": False,
        "teacher_contract_sha256": contract["contract_sha256"],
        "generation_contract_schema_version": contract["schema_version"],
        "generation_max_tokens": contract["max_tokens"],
        "derived_from_manifest_sha256": file_sha256(base_manifest_path),
        "contrastive_spec_sha256": file_sha256(args.spec),
        "assistant_candidate_sources": len(samples),
        "positive_sources": evidence.count("present"),
        "negative_sources": evidence.count("absent"),
        "candidate_repeat": args.repeat,
        "train_samples": len(rows),
        "train_jsonl_sha256": file_sha256(train_path),
        "boundary": (
            "Balanced assistant-reviewed examples are candidate-only micro-tuning "
            "signals, not human truth or production-approved data."),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "train_samples": len(rows), "positive_sources": evidence.count("present"),
        "negative_sources": evidence.count("absent"), "human_truth": False,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
