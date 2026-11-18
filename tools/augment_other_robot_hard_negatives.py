"""Add assistant-reviewed hard negatives to an isolated candidate dataset."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def validate_spec(spec: dict) -> list[dict]:
    samples = spec.get("samples")
    if (spec.get("schema_version") != "labelagent.other-robot-hard-negative-spec.v1"
            or spec.get("human_truth") is not False
            or spec.get("candidate_only") is not True
            or spec.get("production_training_approved") is not False
            or not isinstance(samples, list) or not samples):
        raise ValueError("hard-negative spec boundary differs")
    sample_ids = []
    for item in samples:
        review = item.get("assistant_review_candidate")
        if (not item.get("sample_id") or not item.get("local_image_path")
                or not item.get("remote_image_path") or not item.get("image_sha256")
                or not isinstance(review, dict)
                or review.get("review_kind") != "assistant_visual_candidate"
                or review.get("human_truth") is not False
                or review.get("expected_evidence") != "absent"
                or not review.get("reason")
                or not isinstance(item.get("training_reason"), str)
                or not 1 <= len(item["training_reason"]) <= 96):
            raise ValueError("hard-negative sample boundary differs")
        sample_ids.append(item["sample_id"])
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("hard-negative sample IDs must be unique")
    return samples


def augment_rows(base_rows: list[dict], samples: list[dict], repeat: int,
                 instruction: str) -> list[dict]:
    if not 1 <= repeat <= 16:
        raise ValueError("hard-negative repeat is out of bounds")
    rows = list(base_rows)
    for item in samples:
        local_path = Path(item["local_image_path"])
        if not local_path.is_file() or file_sha256(local_path) != item["image_sha256"]:
            raise ValueError(f"hard-negative image hash differs: {item['sample_id']}")
        target = json.dumps({
            "assessments": {"other_robot": {
                "evidence": "absent", "confidence": 1.0,
                "reason": item["training_reason"],
            }},
            "unknown_candidates": [],
        }, ensure_ascii=False, separators=(",", ":"))
        target_sha = hashlib.sha256(target.encode("utf-8")).hexdigest()
        for repetition in range(repeat):
            rows.append({
                "schema_version": "labelagent.qwen3vl-other-robot-distillation-sample.v1",
                "sample_id": f"{item['sample_id']}-repeat-{repetition:02d}",
                "split": "train",
                "track": "teacher_distillation",
                "candidate_kind": "assistant_review_candidate",
                "human_truth": False,
                "production_training_approved": False,
                "requires_acceptance_gate": True,
                "image_path": item["remote_image_path"],
                "image_sha256": item["image_sha256"],
                "instruction": instruction,
                "target": target,
                "target_sha256": target_sha,
                "source_sequence": item.get("source_sequence"),
                "source_frame_id": item.get("source_frame_id"),
                "source_annotation_kind": "assistant_review_candidate",
                "source_other_robot_instances": 0,
                "assistant_review_candidate": item["assistant_review_candidate"],
                "hard_negative_repeat_index": repetition,
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
    manifest_path = args.base_dataset / "manifest.json"
    contract_path = args.base_dataset / "contract.json"
    train_path = args.base_dataset / "train.jsonl"
    base_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if (base_manifest.get("human_truth") is not False
            or base_manifest.get("production_training_approved") is not False
            or base_manifest.get("test24_included") is not False
            or base_manifest.get("train_jsonl_sha256") != file_sha256(train_path)):
        raise ValueError("base candidate dataset boundary differs")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    samples = validate_spec(spec)
    rows = augment_rows(read_jsonl(train_path), samples, args.repeat,
                        contract["instruction"])
    args.output.mkdir(parents=True)
    new_train = args.output / "train.jsonl"
    write_jsonl(new_train, rows)
    for name in ("dev.jsonl", "holdout.jsonl", "contract.json"):
        shutil.copy2(args.base_dataset / name, args.output / name)
    manifest = dict(base_manifest)
    manifest.update({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "derived_from_manifest_sha256": file_sha256(manifest_path),
        "hard_negative_spec_sha256": file_sha256(args.spec),
        "assistant_candidate_hard_negative_sources": len(samples),
        "assistant_candidate_hard_negative_rows": len(samples) * args.repeat,
        "hard_negative_repeat": args.repeat,
        "train_samples": len(rows),
        "train_jsonl_sha256": file_sha256(new_train),
        "boundary": (
            "Legacy machine labels plus assistant-reviewed hard negatives remain "
            "candidate-only; none are human truth or production-approved training data."),
    })
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "train_samples": len(rows),
        "hard_negative_rows": len(samples) * args.repeat,
        "train_jsonl_sha256": manifest["train_jsonl_sha256"],
        "human_truth": False,
        "production_training_approved": False,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
