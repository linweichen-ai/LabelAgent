"""Build a schema-preserving, short-target Qwen3-VL distillation dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def compact_reason(label: str, evidence: str) -> str:
    if evidence == "present":
        return f"visible {label} pixels"
    if evidence == "absent":
        return f"no visible {label}"
    return f"unclear {label} pixels"


def compact_target(target_text: str) -> str:
    target = json.loads(target_text)
    assessments = target.get("assessments")
    if not isinstance(assessments, dict) or not assessments:
        raise ValueError("target assessments must be a non-empty object")
    compacted: dict[str, dict[str, object]] = {}
    for label, value in assessments.items():
        if not isinstance(label, str) or not isinstance(value, dict):
            raise ValueError("invalid assessment")
        evidence = value.get("evidence")
        if evidence not in {"present", "absent", "uncertain"}:
            raise ValueError(f"invalid evidence for {label}")
        confidence = value.get("confidence")
        if not isinstance(confidence, (int, float)):
            raise ValueError(f"invalid confidence for {label}")
        compacted[label] = {
            "evidence": evidence,
            "confidence": round(float(confidence), 2),
            "reason": compact_reason(label, evidence),
        }
    result = {
        "assessments": compacted,
        "unknown_candidates": target.get("unknown_candidates") or [],
    }
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


def compact_instruction(value: str) -> str:
    marker = "Each reason must be one visual clause of at most 10 words;"
    replacement = (
        "Each reason must be exactly one short visual phrase of at most 3 words;"
    )
    if marker not in value:
        raise ValueError("instruction contract marker not found")
    value = value.replace(marker, replacement)
    return value.replace(
        "Return only compact schema JSON with no indentation or extra whitespace.",
        "Return one-line compact schema JSON with no indentation, prose, or extra whitespace.",
    )


def convert_split(source: Path, destination: Path) -> tuple[int, str]:
    rows: list[str] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        target = compact_target(row["target"])
        row["instruction"] = compact_instruction(row["instruction"])
        row["target"] = target
        row["target_sha256"] = sha256_bytes(target.encode("utf-8"))
        row["target_compaction"] = "deterministic_schema_preserving_v1"
        rows.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
    payload = ("\n".join(rows) + "\n").encode("utf-8")
    destination.write_bytes(payload)
    return len(rows), sha256_bytes(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("output already exists")
    args.output.mkdir(parents=True)
    source_manifest = json.loads((args.source / "manifest.json").read_text(encoding="utf-8"))
    split_receipts = {}
    for split in ("train", "dev"):
        count, digest = convert_split(
            args.source / f"{split}.jsonl", args.output / f"{split}.jsonl"
        )
        split_receipts[split] = {"samples": count, "sha256": digest}
    manifest = dict(source_manifest)
    manifest.update(
        {
            "schema_version": "labelagent.qwen3vl-distillation-dataset.compact-targets.v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "target_compaction": "deterministic_schema_preserving_v1",
            "source_manifest_sha256": sha256_bytes(
                (args.source / "manifest.json").read_bytes()
            ),
            "splits": split_receipts,
            "train_jsonl_sha256": split_receipts["train"]["sha256"],
            "dev_jsonl_sha256": split_receipts["dev"]["sha256"],
            "human_truth": False,
            "production_training_approved": False,
            "test24_included": False,
        }
    )
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
