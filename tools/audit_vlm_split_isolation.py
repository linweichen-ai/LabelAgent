"""Audit exact train/dev/test isolation for the compact VLM distillation set.

This audit deliberately does not claim perceptual near-duplicate detection:
the frozen JSONL files bind image hashes but do not contain the image bytes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


SHA256 = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_SEQUENCES = {
    "train": {f"{index:03d}" for index in range(8)},
    "dev": {"008"},
    "test24": {"010", "011", "012"},
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL object required at {path}:{line_number}")
        rows.append(value)
    return rows


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def row_identity(row: dict) -> tuple[str, str, str]:
    return (str(row.get("source_sequence") or ""),
            str(row.get("source_frame_id") or ""),
            str(row.get("source_camera") or ""))


def audit_split_isolation(
    train_path: Path,
    dev_path: Path,
    compact_manifest_path: Path,
    test_path: Path,
    test_manifest_path: Path,
) -> dict:
    compact_manifest = read_json(compact_manifest_path)
    test_manifest = read_json(test_manifest_path)
    paths = {"train": train_path, "dev": dev_path, "test24": test_path}
    rows = {name: read_jsonl(path) for name, path in paths.items()}
    errors: list[str] = []

    require(file_sha256(train_path) == compact_manifest.get("train_jsonl_sha256"),
            "train JSONL hash differs from compact manifest", errors)
    require(file_sha256(dev_path) == compact_manifest.get("dev_jsonl_sha256"),
            "dev JSONL hash differs from compact manifest", errors)
    require(file_sha256(test_path) == test_manifest.get("test24_jsonl_sha256"),
            "test24 JSONL hash differs from external manifest", errors)
    require(compact_manifest.get("test24_included") is False,
            "compact manifest does not exclude test24", errors)
    require(test_manifest.get("evaluation_only") is True
            and test_manifest.get("training_return_allowed") is False,
            "test24 manifest is not permanently evaluation-only", errors)
    require(compact_manifest.get("train_samples") == len(rows["train"]),
            "train row count differs from compact manifest", errors)
    require(compact_manifest.get("dev_samples") == len(rows["dev"]),
            "dev row count differs from compact manifest", errors)
    require(test_manifest.get("samples") == len(rows["test24"]),
            "test24 row count differs from external manifest", errors)

    split_summary = {}
    sets: dict[str, dict[str, set]] = {}
    for split, split_rows in rows.items():
        sample_ids, image_hashes, identities, targets = set(), set(), set(), set()
        sequences = set()
        for index, row in enumerate(split_rows, 1):
            prefix = f"{split} row {index}"
            sample_id = row.get("sample_id")
            image_hash = str(row.get("image_sha256") or "").lower()
            identity = row_identity(row)
            require(isinstance(sample_id, str) and bool(sample_id),
                    f"{prefix}: sample_id missing", errors)
            require(sample_id not in sample_ids,
                    f"{prefix}: duplicate sample_id", errors)
            require(bool(SHA256.fullmatch(image_hash)),
                    f"{prefix}: image_sha256 invalid", errors)
            require(image_hash not in image_hashes,
                    f"{prefix}: duplicate image_sha256 within split", errors)
            require(all(identity), f"{prefix}: source identity incomplete", errors)
            require(identity not in identities,
                    f"{prefix}: duplicate source identity within split", errors)
            require(row.get("split") == split,
                    f"{prefix}: split marker differs", errors)
            require(row.get("human_truth") is False
                    and row.get("production_training_approved") is False,
                    f"{prefix}: machine candidate promoted to truth/training", errors)
            if split == "test24":
                require(row.get("evaluation_only") is True
                        and row.get("training_return_allowed") is False
                        and row.get("test24") is True,
                        f"{prefix}: external evaluation boundary differs", errors)
            else:
                require(row.get("requires_acceptance_gate") is True,
                        f"{prefix}: acceptance gate missing", errors)
            sample_ids.add(sample_id)
            image_hashes.add(image_hash)
            identities.add(identity)
            sequences.add(identity[0])
            if isinstance(row.get("target_sha256"), str):
                targets.add(row["target_sha256"].lower())
        require(sequences == EXPECTED_SEQUENCES[split],
                f"{split}: sequence membership differs", errors)
        sets[split] = {
            "sample_ids": sample_ids, "image_hashes": image_hashes,
            "identities": identities, "targets": targets,
        }
        split_summary[split] = {
            "rows": len(split_rows),
            "sequences": sorted(sequences),
            "unique_sample_ids": len(sample_ids),
            "unique_image_sha256": len(image_hashes),
            "unique_source_identities": len(identities),
            "jsonl_sha256": file_sha256(paths[split]),
        }

    overlaps = {}
    for left, right in (("train", "dev"), ("train", "test24"),
                        ("dev", "test24")):
        key = f"{left}_vs_{right}"
        overlaps[key] = {
            "sample_ids": sorted(sets[left]["sample_ids"] & sets[right]["sample_ids"]),
            "image_sha256": sorted(sets[left]["image_hashes"] & sets[right]["image_hashes"]),
            "source_identities": [list(value) for value in sorted(
                sets[left]["identities"] & sets[right]["identities"])],
            "sequences": sorted(set(split_summary[left]["sequences"])
                                & set(split_summary[right]["sequences"])),
            "target_sha256_count": len(sets[left]["targets"] & sets[right]["targets"]),
        }
        require(not overlaps[key]["sample_ids"], f"{key}: sample-id leakage", errors)
        require(not overlaps[key]["image_sha256"], f"{key}: exact image leakage", errors)
        require(not overlaps[key]["source_identities"],
                f"{key}: source-identity leakage", errors)
        require(not overlaps[key]["sequences"], f"{key}: sequence leakage", errors)

    return {
        "schema_version": "labelagent.vlm-split-isolation-audit.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed_exact_isolation" if not errors else "failed",
        "scope": "exact manifest/hash/source isolation only",
        "splits": split_summary,
        "overlaps": overlaps,
        "near_duplicate_audit": {
            "status": "not_evaluated_without_bound_image_bytes",
            "passed": None,
            "reason": "JSONL manifests bind hashes but do not include pixels for perceptual comparison",
        },
        "human_truth": False,
        "production_training_approved": False,
        "test24_training_return_allowed": False,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--dev", required=True, type=Path)
    parser.add_argument("--compact-manifest", required=True, type=Path)
    parser.add_argument("--test24", required=True, type=Path)
    parser.add_argument("--test24-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit_split_isolation(
        args.train.resolve(), args.dev.resolve(), args.compact_manifest.resolve(),
        args.test24.resolve(), args.test24_manifest.resolve())
    output = args.output.resolve()
    if output.exists():
        raise ValueError("choose a fresh audit output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "passed_exact_isolation" else 2


if __name__ == "__main__":
    raise SystemExit(main())
