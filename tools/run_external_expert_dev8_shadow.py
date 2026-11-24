"""Run a bounded external-expert shadow review on the frozen dev8 set.

The external model receives image bytes and the frozen instruction only.  Local
teacher outputs are loaded strictly after each response for offline comparison.
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.agent.workflow import chat_external_expert


LABELS = ("车辆", "pallet", "person", "floor", "slope", "ego",
          "other_robot", "cargo_floor")
SCHEMA_VERSION = "labelagent.external-expert-dev8-shadow.v1"


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_dev8(dataset_path, image_dir, contract_path):
    rows = [json.loads(line) for line in Path(dataset_path).read_text(
        encoding="utf-8").splitlines() if line.strip()]
    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    if len(rows) != 8 or len({row.get("sample_id") for row in rows}) != 8:
        raise ValueError("frozen dev8 must contain eight unique rows")
    if any(row.get("split") != "dev" for row in rows):
        raise ValueError("only the dev split may enter this shadow review")
    if any(row.get("test24") is True or "test24" in row.get("track", "").lower()
           for row in rows):
        raise ValueError("test24 is forbidden in this shadow review")
    if contract.get("labels") != list(LABELS):
        raise ValueError("frozen contract labels differ")
    for row in rows:
        image_path = Path(image_dir) / f"{row['sample_id']}.png"
        if not image_path.is_file():
            raise ValueError(f"missing dev image: {row['sample_id']}")
        if file_sha256(image_path) != row.get("image_sha256"):
            raise ValueError(f"dev image hash mismatch: {row['sample_id']}")
        row["_image_path"] = image_path
    return rows, contract


def validate_result(result):
    assessments = result.get("assessments") if isinstance(result, dict) else None
    if not isinstance(assessments, dict) or set(assessments) != set(LABELS):
        raise ValueError("expert response does not contain the exact label set")
    for label in LABELS:
        item = assessments[label]
        if (not isinstance(item, dict)
                or item.get("evidence") not in {"present", "absent", "uncertain"}
                or not isinstance(item.get("confidence"), (int, float))
                or not 0 <= item["confidence"] <= 1
                or not isinstance(item.get("reason"), str)
                or not item["reason"]):
            raise ValueError(f"invalid assessment for {label}")
    if not isinstance(result.get("unknown_candidates"), list):
        raise ValueError("unknown_candidates must be a list")


def compare_with_machine_teacher(expert, teacher):
    expert_assessments = expert["assessments"]
    teacher_assessments = teacher["assessments"]
    matches = {
        label: expert_assessments[label]["evidence"]
        == teacher_assessments[label]["evidence"] for label in LABELS
    }
    return {
        "evidence_matches": matches,
        "matched_labels": sum(matches.values()),
        "total_labels": len(LABELS),
        "slope_match": matches["slope"],
    }


def run_shadow(*, rows, contract, output_dir, config, chat_fn=chat_external_expert,
               assistant_slope_review=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "results.jsonl"
    completed = {}
    if jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                if item.get("status") == "passed":
                    completed[item["sample_id"]] = item

    records = []
    for row in rows:
        if row["sample_id"] in completed:
            records.append(completed[row["sample_id"]])
            continue
        image = row["_image_path"].read_bytes()
        started = time.perf_counter()
        try:
            expert = chat_fn(
                config, row["instruction"], image,
                response_schema=contract["response_schema"])
            validate_result(expert)
            latency_ms = round((time.perf_counter() - started) * 1000, 3)
            teacher = json.loads(row["target"])
            comparison = compare_with_machine_teacher(expert, teacher)
            record = {
                "schema_version": SCHEMA_VERSION,
                "sample_id": row["sample_id"],
                "status": "passed",
                "latency_ms": latency_ms,
                "image_sha256": row["image_sha256"],
                "expert_result": expert,
                "machine_teacher_comparison": comparison,
                "candidate_only": True,
                "human_truth": False,
                "approved_for_training": False,
                "test24_used": False,
            }
        except Exception as exc:
            record = {
                "schema_version": SCHEMA_VERSION,
                "sample_id": row["sample_id"],
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
                "candidate_only": True,
                "human_truth": False,
                "approved_for_training": False,
                "test24_used": False,
            }
        with jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        records.append(record)

    passed = [record for record in records if record["status"] == "passed"]
    latencies = [record["latency_ms"] for record in passed]
    matches = sum(record["machine_teacher_comparison"]["matched_labels"]
                  for record in passed)
    slope_matches = sum(record["machine_teacher_comparison"]["slope_match"]
                        for record in passed)
    per_label = {}
    for label in LABELS:
        label_matches = sum(
            record["machine_teacher_comparison"]["evidence_matches"][label]
            for record in passed)
        expert_present = sum(
            record["expert_result"]["assessments"][label]["evidence"] == "present"
            for record in passed)
        per_label[label] = {
            "machine_teacher_matches": label_matches,
            "total": len(passed),
            "expert_present": expert_present,
        }
    assistant_slope_matches = None
    if assistant_slope_review is not None:
        decisions = {
            item["sample_id"]: bool(item["assistant_visible_slope_candidate"])
            for item in assistant_slope_review.get("decisions", [])
        }
        if set(decisions) != {record["sample_id"] for record in passed}:
            raise ValueError("assistant slope review does not match completed dev8")
        assistant_slope_matches = sum(
            (record["expert_result"]["assessments"]["slope"]["evidence"]
             == "present") == decisions[record["sample_id"]]
            for record in passed)
    summary = {
        "schema_version": "labelagent.external-expert-dev8-shadow-summary.v1",
        "status": "completed" if len(passed) == len(rows) else "partial",
        "samples_total": len(rows),
        "samples_passed": len(passed),
        "samples_failed": len(rows) - len(passed),
        "machine_teacher_evidence_matches": matches,
        "machine_teacher_evidence_total": len(passed) * len(LABELS),
        "slope_matches": slope_matches,
        "slope_total": len(passed),
        "assistant_slope_candidate_matches": assistant_slope_matches,
        "assistant_slope_candidate_total": (
            len(passed) if assistant_slope_matches is not None else None),
        "per_label_machine_teacher_comparison": per_label,
        "unknown_candidate_proposals": sum(
            len(record["expert_result"]["unknown_candidates"])
            for record in passed),
        "latency_median_ms": round(statistics.median(latencies), 3) if latencies else None,
        "latency_max_ms": max(latencies) if latencies else None,
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "test24_used": False,
        "uploaded_content": "eight_frozen_640x512_dev_images_only",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--key-env", required=True)
    parser.add_argument("--assistant-slope-review", type=Path)
    args = parser.parse_args(argv)
    rows, contract = load_dev8(args.dataset, args.images, args.contract)
    config = {
        "llm_expert_enabled": True,
        "llm_expert_provider": "openai_chat",
        "llm_expert_url": args.url,
        "llm_expert_model": args.model,
        "llm_expert_key_env": args.key_env,
        "llm_expert_timeout_seconds": 120,
    }
    assistant_review = None
    if args.assistant_slope_review:
        assistant_review = json.loads(args.assistant_slope_review.read_text(
            encoding="utf-8"))
    summary = run_shadow(
        rows=rows, contract=contract, output_dir=args.output, config=config,
        assistant_slope_review=assistant_review)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
