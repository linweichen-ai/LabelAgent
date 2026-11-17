"""Bind frozen test24 images to frozen 8B teacher evidence for evaluation only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


LABELS = ["车辆", "pallet", "person", "floor", "slope", "ego",
          "other_robot", "cargo_floor"]
TEST_SEQUENCES = {"010", "011", "012"}


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def target_sha256(target: dict) -> str:
    payload = json.dumps(target, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_instruction(path: Path) -> tuple[str, str]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    instructions = {row.get("instruction") for row in rows}
    contracts = {row.get("teacher_contract_sha256") for row in rows}
    if len(instructions) != 1 or len(contracts) != 1:
        raise ValueError("instruction source is not a single frozen contract")
    return instructions.pop(), contracts.pop()


def build_rows(pack: dict, teacher: dict, instruction: str,
               contract_sha256: str) -> list[dict]:
    images = [row for row in pack.get("images", []) if row.get("split") == "test"]
    runs = teacher.get("runs")
    if (len(images) != 24 or not isinstance(runs, list) or len(runs) != 24
            or teacher.get("success_count") != 24 or teacher.get("failure_count") != 0):
        raise ValueError("frozen test24 sources are incomplete")
    if teacher.get("labels") != LABELS:
        raise ValueError("teacher label registry differs")
    by_sha = {row.get("image_sha256"): row for row in runs}
    if len(by_sha) != 24:
        raise ValueError("teacher image hashes are not unique")
    rows = []
    for image in sorted(images, key=lambda row: row["image_id"]):
        if image.get("sequence") not in TEST_SEQUENCES:
            raise ValueError("non-test sequence entered external evaluation")
        run = by_sha.get(image.get("source_sha256"))
        if run is None or run.get("failure") is not None:
            raise ValueError("teacher result is missing for a test image")
        assessments = run.get("assessments")
        if (not isinstance(assessments, list)
                or [item.get("label") for item in assessments] != LABELS):
            raise ValueError("teacher assessments differ from the frozen registry")
        target = {
            "assessments": {
                item["label"]: {
                    "evidence": item.get("evidence"),
                    "confidence": item.get("confidence"),
                } for item in assessments
            },
            "unknown_candidates": run.get("unknown_candidates") or [],
        }
        if any(item["evidence"] not in {"present", "absent", "uncertain"}
               for item in target["assessments"].values()):
            raise ValueError("teacher evidence is invalid")
        rows.append({
            "schema_version": "labelagent.qwen3vl-external-eval-sample.v1",
            "sample_id": Path(image["file_name"]).stem,
            "split": "test24",
            "track": "teacher_distillation_external_evaluation",
            "evaluation_only": True,
            "training_return_allowed": False,
            "candidate_kind": "frozen_machine_teacher_output",
            "human_truth": False,
            "production_training_approved": False,
            "test24": True,
            "image_path": image["source_path"],
            "image_sha256": image["source_sha256"],
            "instruction": instruction,
            "target": json.dumps(target, ensure_ascii=False,
                                  separators=(",", ":")),
            "target_sha256": target_sha256(target),
            "teacher_model": teacher.get("model"),
            "teacher_contract_sha256": contract_sha256,
            "source_sequence": image["sequence"],
            "source_frame_id": image["frame_id"],
            "source_camera": image["camera"],
        })
    if len({row["image_sha256"] for row in rows}) != 24:
        raise ValueError("external evaluation images are not unique")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-manifest", required=True, type=Path)
    parser.add_argument("--teacher-results", required=True, type=Path)
    parser.add_argument("--instruction-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    instruction, contract_sha256 = load_instruction(args.instruction_source)
    pack = json.loads(args.pack_manifest.read_text(encoding="utf-8"))
    teacher = json.loads(args.teacher_results.read_text(encoding="utf-8"))
    rows = build_rows(pack, teacher, instruction, contract_sha256)
    args.output.mkdir(parents=True)
    jsonl = args.output / "test24.jsonl"
    jsonl.write_text("".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows), encoding="utf-8")
    manifest = {
        "schema_version": "labelagent.qwen3vl-external-eval-manifest.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_external_evaluation_only",
        "split": "test24",
        "evaluation_only": True,
        "training_return_allowed": False,
        "human_truth": False,
        "production_training_approved": False,
        "test24_included": True,
        "samples": len(rows),
        "sequences": sorted({row["source_sequence"] for row in rows}),
        "unique_image_sha256": len({row["image_sha256"] for row in rows}),
        "teacher_model": teacher.get("model"),
        "teacher_contract_sha256": contract_sha256,
        "pack_manifest_sha256": file_sha256(args.pack_manifest),
        "teacher_results_sha256": file_sha256(args.teacher_results),
        "instruction_source_sha256": file_sha256(args.instruction_source),
        "test24_jsonl_sha256": file_sha256(jsonl),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
