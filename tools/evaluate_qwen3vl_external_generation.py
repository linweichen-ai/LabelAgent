"""Run a frozen evaluation-only test24 set through a Qwen3-VL LoRA candidate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

try:
    from tools.evaluate_qwen3vl_lora_generation import evaluate_mode, file_sha256
except ModuleNotFoundError:  # Flat, isolated remote bundle.
    from evaluate_qwen3vl_lora_generation import evaluate_mode, file_sha256


LABELS = ["车辆", "pallet", "person", "floor", "slope", "ego",
          "other_robot", "cargo_floor"]


def load_external_rows(dataset_dir: Path) -> tuple[list[dict], dict]:
    manifest_path = dataset_dir / "manifest.json"
    rows_path = dataset_dir / "test24.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version")
            != "labelagent.qwen3vl-external-eval-manifest.v1"
            or manifest.get("status") != "frozen_external_evaluation_only"
            or manifest.get("split") != "test24"
            or manifest.get("evaluation_only") is not True
            or manifest.get("training_return_allowed") is not False
            or manifest.get("human_truth") is not False
            or manifest.get("production_training_approved") is not False
            or manifest.get("test24_included") is not True
            or manifest.get("samples") != 24
            or manifest.get("sequences") != ["010", "011", "012"]
            or manifest.get("test24_jsonl_sha256") != file_sha256(rows_path)):
        raise ValueError("external evaluation manifest violates its boundary")
    rows = [json.loads(line) for line in rows_path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 24 or len({row.get("image_sha256") for row in rows}) != 24:
        raise ValueError("external evaluation rows are incomplete or duplicated")
    for row in rows:
        if (row.get("split") != "test24" or row.get("test24") is not True
                or row.get("evaluation_only") is not True
                or row.get("training_return_allowed") is not False
                or row.get("human_truth") is not False
                or row.get("production_training_approved") is not False
                or row.get("source_sequence") not in {"010", "011", "012"}):
            raise ValueError("external row violates evaluation-only isolation")
    return rows, manifest


def label_summary(rows: list[dict]) -> dict:
    result = {}
    for label in LABELS:
        matches = 0
        teacher_present = 0
        candidate_present = 0
        for row in rows:
            if row.get("status") != "ok":
                continue
            target = row["score"]["label_matches"]
            matches += bool(target[label])
            teacher = json.loads(row["target"])["assessments"][label]["evidence"]
            candidate = row["content"]["assessments"][label]["evidence"]
            teacher_present += teacher == "present"
            candidate_present += candidate == "present"
        result[label] = {
            "teacher_matches": matches,
            "total": len(rows),
            "teacher_present": teacher_present,
            "candidate_present": candidate_present,
        }
    return result


def verify_external_images(rows: list[dict]) -> dict:
    mismatches = []
    total_bytes = 0
    for row in rows:
        path = Path(row["image_path"])
        if not path.is_file():
            mismatches.append({"sample_id": row["sample_id"], "reason": "missing"})
            continue
        total_bytes += path.stat().st_size
        if file_sha256(path) != row["image_sha256"]:
            mismatches.append({"sample_id": row["sample_id"], "reason": "sha256"})
    if mismatches:
        raise ValueError(f"external image verification failed: {mismatches}")
    return {"images": len(rows), "bytes": total_bytes, "sha256_matches": len(rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--server-module-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows, manifest = load_external_rows(args.dataset_dir)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if (contract.get("schema_version") != "labelagent.vlm-teacher-contract.v1"
            or contract.get("contract_sha256")
            != manifest.get("teacher_contract_sha256")
            or contract.get("max_tokens") != 384
            or not isinstance(contract.get("response_schema"), dict)):
        raise ValueError("external generation contract differs")
    image_verification = verify_external_images(rows)
    if args.preflight_only:
        print(json.dumps({
            "status": "external_preflight_passed",
            "split": "test24",
            "evaluation_only": True,
            "training_return_allowed": False,
            **image_verification,
        }, ensure_ascii=False, indent=2))
        return 0
    sys.path.insert(0, str(args.server_module_dir.resolve()))

    import peft
    import torch
    import transformers
    from peft import PeftModel
    from local_vlm_server import TransformersBackend

    backend = TransformersBackend(
        args.model_dir, "qwen3vl2b-external-candidate",
        max_pixels=256 * 32 * 32, context_budget=16384, quantization="none",
        generation_cache_implementation="static", startup_prewarm=True)
    backend.model = PeftModel.from_pretrained(
        backend.model, args.adapter_dir, is_trainable=False)
    backend.model.eval()
    prewarm = backend._run_startup_prewarm()
    backend.startup_prewarm = prewarm
    backend.model_loaded = True
    backend.last_inference_stats = None
    summary, result_rows = evaluate_mode(
        backend, rows, contract, "adapter_static_test24")
    # Retain the frozen target alongside each result so per-label aggregation
    # remains independently auditable without rejoining mutable files.
    by_id = {row["sample_id"]: row for row in rows}
    for result_row in result_rows:
        result_row["target"] = by_id[result_row["sample_id"]]["target"]
    labels = label_summary(result_rows)
    result = {
        "schema_version": "labelagent.qwen3vl2b-external-generation-eval.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "external_evaluated_not_deployed",
        "track": "teacher_distillation_external_evaluation",
        "split": "test24",
        "evaluation_only": True,
        "training_return_allowed": False,
        "human_truth": False,
        "production_deployed": False,
        "generation_cache_implementation": "static",
        "dataset_manifest_sha256": file_sha256(args.dataset_dir / "manifest.json"),
        "image_verification": image_verification,
        "contract_sha256": contract["contract_sha256"],
        "adapter_sha256": file_sha256(args.adapter_dir / "adapter_model.safetensors"),
        "startup_prewarm": prewarm,
        "summary": summary,
        "per_label": labels,
        "per_sample": result_rows,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
    }
    args.output.mkdir(parents=True)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    (args.output / "result.json").write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
