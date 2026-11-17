"""Build an audited Qwen3-VL teacher-distillation JSONL without image copies."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


DEV_SEQUENCE = "008"


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=False,
                      separators=(",", ":"), allow_nan=False)


def canonical_hash_json(value: object) -> str:
    """Match the teacher collector's stable, key-sorted hash contract."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_records(collection: Path, audit: dict) -> tuple[list[dict], list[dict], dict]:
    if audit.get("status") != "passed" or audit.get("train72_exact_coverage") is not True:
        raise ValueError("teacher collection did not pass independent audit")
    if audit.get("test24_included") is not False:
        raise ValueError("test24 may not enter distillation data")
    contract = read_json(collection / "contract.json")
    instruction = contract.get("instruction")
    contract_sha = contract.get("contract_sha256")
    if not isinstance(instruction, str) or not instruction or not isinstance(contract_sha, str):
        raise ValueError("teacher contract is incomplete")

    train, dev = [], []
    excluded = []
    seen_names, seen_hashes = set(), set()
    for path in sorted((collection / "rows").glob("*.json")):
        row = read_json(path)
        image = row.get("image") if isinstance(row.get("image"), dict) else {}
        if row.get("split") != "train" or row.get("human_truth") is not False:
            raise ValueError("teacher row split or truth boundary differs")
        if row.get("training_authorized") is not False or row.get("requires_acceptance_gate") is not True:
            raise ValueError("teacher row training boundary differs")
        name, source_sha = image.get("file_name"), image.get("source_sha256")
        if name in seen_names or source_sha in seen_hashes:
            raise ValueError("duplicate teacher image binding")
        seen_names.add(name)
        seen_hashes.add(source_sha)
        if row.get("distillation_candidate") is not True:
            excluded.append({
                "file_name": name,
                "sequence": image.get("sequence"),
                "reason_codes": [flag.get("reason_code") for flag in
                                 row.get("review", {}).get("consistency_flags", [])],
            })
            continue
        target = row.get("target")
        target_sha = row.get("target_sha256")
        if not isinstance(target, dict) or not isinstance(target_sha, str):
            raise ValueError("teacher target is incomplete")
        target_text = canonical_json(target)
        hash_text = canonical_hash_json(target)
        if hashlib.sha256(hash_text.encode("utf-8")).hexdigest() != target_sha:
            raise ValueError("teacher target canonical hash differs")
        split = "dev" if str(image.get("sequence")) == DEV_SEQUENCE else "train"
        record = {
            "schema_version": "labelagent.qwen3vl-distillation-sample.v1",
            "sample_id": str(name).removesuffix(".png"),
            "split": split,
            "track": "teacher_distillation",
            "candidate_kind": "machine_teacher_output",
            "human_truth": False,
            "production_training_approved": False,
            "requires_acceptance_gate": True,
            "image_path": image.get("source_path"),
            "image_sha256": source_sha,
            "production_png_sha256": image.get("production_png_sha256"),
            "instruction": instruction,
            "target": target_text,
            "target_sha256": target_sha,
            "teacher_model": row.get("teacher", {}).get("model"),
            "teacher_model_config_sha256": row.get("teacher", {}).get("model_config_sha256"),
            "teacher_raw_parsed_sha256": row.get("teacher", {}).get("raw_parsed_sha256"),
            "teacher_contract_sha256": contract_sha,
            "source_sequence": image.get("sequence"),
            "source_frame_id": image.get("frame_id"),
            "source_camera": image.get("camera"),
        }
        (dev if split == "dev" else train).append(record)
    if len(train) != 62 or len(dev) != 8 or len(excluded) != 2:
        raise ValueError("expected 62 train, 8 dev and 2 review-only rows")
    return train, dev, {"excluded": excluded, "contract": contract}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                      for row in rows)
    path.write_text(payload, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    collection = args.collection.resolve()
    output = args.output.resolve()
    artifact_root = (Path(__file__).resolve().parents[1] / "artifacts").resolve()
    if not output.is_relative_to(artifact_root):
        raise ValueError("output must stay under repository artifacts")
    if output.exists():
        raise FileExistsError(output)
    audit = read_json(args.audit)
    train, dev, context = build_records(collection, audit)
    output.mkdir(parents=True)
    train_path, dev_path = output / "train.jsonl", output / "dev.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(dev_path, dev)
    manifest = {
        "schema_version": "labelagent.qwen3vl-distillation-dataset.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ready_for_isolated_lora_smoke",
        "track": "teacher_distillation",
        "human_truth": False,
        "production_training_approved": False,
        "requires_acceptance_gate": True,
        "test24_included": False,
        "split_policy": "train sequences 000-007; internal dev sequence 008; test24 010-012 remains external evaluation only",
        "train_samples": len(train),
        "dev_samples": len(dev),
        "review_only_samples": len(context["excluded"]),
        "review_only": context["excluded"],
        "train_jsonl_sha256": file_sha256(train_path),
        "dev_jsonl_sha256": file_sha256(dev_path),
        "teacher_collection_audit_sha256": file_sha256(args.audit),
        "teacher_contract_sha256": context["contract"]["contract_sha256"],
        "teacher_model_config_sha256": audit["teacher_model_config_sha256"],
        "source_image_bytes_copied": 0,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
