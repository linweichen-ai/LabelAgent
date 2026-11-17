"""Independently read back a frozen VLM teacher-candidate collection."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def audit_collection(
    collection: Path,
    manifest: dict,
    expected_state_sha256: str,
    expected_model_config_sha256: str,
) -> dict:
    contract_file = collection / "contract.json"
    summary_file = collection / "summary.json"
    contract = read_json(contract_file)
    summary = read_json(summary_file)
    contract_body = {key: value for key, value in contract.items()
                     if key != "contract_sha256"}
    contract_sha = json_sha256(contract_body)
    errors: list[str] = []
    if contract.get("contract_sha256") != contract_sha:
        errors.append("contract hash differs")

    rows = [read_json(path) for path in sorted((collection / "rows").glob("*.json"))]
    train = {image["file_name"]: image for image in manifest.get("images", [])
             if image.get("split") == "train"}
    test_names = {image["file_name"] for image in manifest.get("images", [])
                  if image.get("split") == "test"}
    test_hashes = {image["source_sha256"] for image in manifest.get("images", [])
                   if image.get("split") == "test"}
    names, hashes, targets = [], [], []
    candidates, review_required = 0, 0
    cache_hits, coalesced_requests = 0, 0
    reason_codes: dict[str, int] = {}
    for index, row in enumerate(rows, 1):
        prefix = f"row {index}"
        image = row.get("image") if isinstance(row.get("image"), dict) else {}
        name = image.get("file_name")
        source_hash = image.get("source_sha256")
        target = row.get("target")
        names.append(name)
        hashes.append(source_hash)
        targets.append(row.get("target_sha256"))
        if name not in train:
            errors.append(f"{prefix}: file is not in train72")
        elif source_hash != train[name].get("source_sha256"):
            errors.append(f"{prefix}: source hash differs from manifest")
        if name in test_names or source_hash in test_hashes:
            errors.append(f"{prefix}: test24 leakage")
        if row.get("split") != "train":
            errors.append(f"{prefix}: split is not train")
        if row.get("candidate_kind") != "machine_teacher_output":
            errors.append(f"{prefix}: candidate kind differs")
        if row.get("human_truth") is not False or row.get("training_authorized") is not False:
            errors.append(f"{prefix}: candidate is mislabeled as truth or authorized training")
        if row.get("requires_acceptance_gate") is not True:
            errors.append(f"{prefix}: acceptance gate is missing")
        if row.get("bindings", {}).get("state_sha256") != expected_state_sha256:
            errors.append(f"{prefix}: state binding differs")
        if row.get("bindings", {}).get("contract_sha256") != contract_sha:
            errors.append(f"{prefix}: contract binding differs")
        if row.get("teacher", {}).get("model_config_sha256") != expected_model_config_sha256:
            errors.append(f"{prefix}: teacher model configuration differs")
        raw_parsed = row.get("teacher", {}).get("raw_parsed")
        if row.get("teacher", {}).get("raw_parsed_sha256") != json_sha256(raw_parsed):
            errors.append(f"{prefix}: raw parsed teacher hash differs")
        if not isinstance(target, dict) or row.get("target_sha256") != json_sha256(target):
            errors.append(f"{prefix}: target hash differs")
        flags = row.get("review", {}).get("consistency_flags") or []
        conflicts = row.get("review", {}).get("registry_conflicts") or []
        open_set_complete = row.get("review", {}).get("open_set_scan_complete") is True
        expected_candidate = open_set_complete and not flags and not conflicts
        if row.get("distillation_candidate") is not expected_candidate:
            errors.append(f"{prefix}: candidate decision differs from review evidence")
        if expected_candidate:
            candidates += 1
        else:
            review_required += 1
        cache_hits += row.get("teacher", {}).get("cache_hit") is True
        coalesced_requests += row.get("teacher", {}).get("coalesced") is True
        for flag in flags:
            reason = str(flag.get("reason_code") or "missing")
            reason_codes[reason] = reason_codes.get(reason, 0) + 1

    if len(rows) != 72:
        errors.append("collection does not contain 72 rows")
    if set(names) != set(train) or len(set(names)) != len(names):
        errors.append("collection file names do not exactly cover train72")
    if len(set(hashes)) != len(hashes):
        errors.append("collection source hashes are not unique")
    if summary.get("test24_included") is not False:
        errors.append("summary does not exclude test24")
    if summary.get("human_truth") is not False or summary.get("training_authorized") is not False:
        errors.append("summary is mislabeled as truth or authorized training")
    if summary.get("images") != len(rows):
        errors.append("summary image count differs")
    if summary.get("distillation_candidates") != candidates:
        errors.append("summary candidate count differs")
    if summary.get("review_required") != review_required:
        errors.append("summary review count differs")
    if summary.get("cache_hits") != cache_hits:
        errors.append("summary cache-hit count differs")
    if summary.get("coalesced_requests") != coalesced_requests:
        errors.append("summary coalesced count differs")
    if summary.get("contract_sha256") != contract_sha:
        errors.append("summary contract binding differs")
    if summary.get("state_sha256") != expected_state_sha256:
        errors.append("summary state binding differs")
    if summary.get("teacher_model_config_sha256") != expected_model_config_sha256:
        errors.append("summary model binding differs")

    return {
        "schema_version": "labelagent.vlm-teacher-collection-audit.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not errors else "failed",
        "rows": len(rows),
        "train72_exact_coverage": set(names) == set(train) and len(set(names)) == 72,
        "test24_included": False if not any(name in test_names for name in names) else True,
        "unique_source_hashes": len(set(hashes)),
        "unique_target_hashes": len(set(targets)),
        "distillation_candidates": candidates,
        "review_required": review_required,
        "cache_hits": cache_hits,
        "coalesced_requests": coalesced_requests,
        "review_reason_counts": reason_codes,
        "human_truth": False,
        "training_authorized": False,
        "contract_sha256": contract_sha,
        "state_sha256": expected_state_sha256,
        "teacher_model_config_sha256": expected_model_config_sha256,
        "collection_summary_sha256": file_sha256(summary_file),
        "collection_contract_file_sha256": file_sha256(contract_file),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expected-state-sha256", required=True)
    parser.add_argument("--expected-model-config-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    collection = args.collection.resolve()
    output = args.output.resolve()
    if not output.is_relative_to(collection):
        raise ValueError("audit output must stay inside the collection")
    report = audit_collection(
        collection,
        read_json(args.manifest),
        args.expected_state_sha256.lower(),
        args.expected_model_config_sha256.lower(),
    )
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
