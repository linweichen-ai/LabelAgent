"""Compare base and LoRA Qwen3-VL structured generations on internal dev8."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

from PIL import Image, ImageOps

try:
    from tools.evaluate_qwen3vl_lora_teacher_forcing import load_dev_rows
except ModuleNotFoundError:  # Flat, isolated remote bundle.
    from evaluate_qwen3vl_lora_teacher_forcing import load_dev_rows


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score_output(row: dict, parsed: dict) -> dict:
    target = json.loads(row["target"])
    target_assessments = target.get("assessments")
    generated_assessments = parsed.get("assessments") if isinstance(parsed, dict) else None
    if not isinstance(target_assessments, dict) or not isinstance(generated_assessments, dict):
        raise ValueError("assessment object is missing")
    if set(generated_assessments) != set(target_assessments):
        raise ValueError("generated label registry differs")
    matches = {}
    for label, expected in target_assessments.items():
        actual = generated_assessments.get(label)
        if not isinstance(actual, dict) or actual.get("evidence") not in {
                "present", "absent", "uncertain"}:
            raise ValueError("generated evidence is invalid")
        matches[label] = actual["evidence"] == expected.get("evidence")
    unknown = parsed.get("unknown_candidates")
    if not isinstance(unknown, list):
        raise ValueError("unknown candidate list is invalid")
    return {
        "label_matches": matches,
        "matched_labels": sum(matches.values()),
        "label_count": len(matches),
        "unknown_count": len(unknown),
        "teacher_unknown_count": len(target.get("unknown_candidates") or []),
        "unknown_count_matches": len(unknown) == len(target.get("unknown_candidates") or []),
    }


def summarize(rows: list[dict]) -> dict:
    successful = [row for row in rows if row["status"] == "ok"]
    label_total = sum(row["score"]["label_count"] for row in successful)
    label_matches = sum(row["score"]["matched_labels"] for row in successful)
    generations = [row["generation_seconds"] for row in successful]
    completions = [row["completion_tokens"] for row in successful]
    return {
        "samples": len(rows),
        "one_pass_successes": len(successful),
        "one_pass_success_rate": len(successful) / len(rows),
        "teacher_evidence_matches": label_matches,
        "teacher_evidence_total": label_total,
        "teacher_evidence_accuracy": label_matches / label_total if label_total else 0.0,
        "unknown_count_matches": sum(
            row["score"]["unknown_count_matches"] for row in successful),
        "generation_seconds_median": statistics.median(generations) if generations else None,
        "generation_seconds_max": max(generations) if generations else None,
        "completion_tokens_median": statistics.median(completions) if completions else None,
        "completion_tokens_max": max(completions) if completions else None,
    }


def evaluate_mode(backend, rows: list[dict], contract: dict, mode: str) -> tuple[dict, list[dict]]:
    results = []
    for row in rows:
        started = time.perf_counter()
        content = None
        prompt_tokens = None
        completion_tokens = None
        stats = {}
        try:
            image_path = Path(row["image_path"])
            if file_sha256(image_path) != row["image_sha256"]:
                raise ValueError("source image hash differs")
            with Image.open(image_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "text", "text": row["instruction"]}, {"type": "image"}]}]
            content, prompt_tokens, completion_tokens = backend.generate(
                messages, [image], contract["max_tokens"],
                response_schema=contract["response_schema"])
            stats = backend.last_inference_stats or {}
            parsed = json.loads(content)
            score = score_output(row, parsed)
            result = {
                "sample_id": row["sample_id"], "mode": mode, "status": "ok",
                "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                "generation_seconds": stats.get("generation_seconds"),
                "wall_seconds": round(time.perf_counter() - started, 6),
                "actual_image_token_counts": stats.get("actual_image_token_counts"),
                "content": parsed, "content_sha256": hashlib.sha256(
                    json.dumps(parsed, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":")).encode("utf-8")).hexdigest(),
                "score": score,
            }
        except Exception as error:
            result = {
                "sample_id": row.get("sample_id"), "mode": mode, "status": "failed",
                "error_type": type(error).__name__, "error": str(error)[:1000],
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "generation_seconds": stats.get("generation_seconds"),
                "content_sha256": (hashlib.sha256(content.encode("utf-8")).hexdigest()
                                   if isinstance(content, str) else None),
                "wall_seconds": round(time.perf_counter() - started, 6),
            }
        results.append(result)
        print(json.dumps({key: result.get(key) for key in (
            "sample_id", "mode", "status", "completion_tokens",
            "generation_seconds", "score", "error_type", "error")},
            ensure_ascii=False), flush=True)
    return summarize(results), results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--server-module-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows, manifest = load_dev_rows(args.dataset_dir)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if (contract.get("schema_version") != "labelagent.vlm-teacher-contract.v1"
            or contract.get("contract_sha256") != manifest.get("teacher_contract_sha256")
            or contract.get("max_tokens") != 384
            or not isinstance(contract.get("response_schema"), dict)):
        raise ValueError("generation contract differs")
    sys.path.insert(0, str(args.server_module_dir.resolve()))

    import peft
    import torch
    import transformers
    from peft import PeftModel
    from local_vlm_server import TransformersBackend

    backend = TransformersBackend(
        args.model_dir, "qwen3vl2b-base", max_pixels=256 * 32 * 32,
        context_budget=16384, quantization="none",
        generation_cache_implementation="dynamic", startup_prewarm=False)
    base_summary, base_rows = evaluate_mode(backend, rows, contract, "base")
    backend.model = PeftModel.from_pretrained(
        backend.model, args.adapter_dir, is_trainable=False)
    backend.model.eval()
    adapter_summary, adapter_rows = evaluate_mode(
        backend, rows, contract, "adapter")
    comparison = {
        "one_pass_success_delta": (
            adapter_summary["one_pass_success_rate"] - base_summary["one_pass_success_rate"]),
        "teacher_evidence_accuracy_delta": (
            adapter_summary["teacher_evidence_accuracy"]
            - base_summary["teacher_evidence_accuracy"]),
        "generation_seconds_median_delta": (
            adapter_summary["generation_seconds_median"]
            - base_summary["generation_seconds_median"]
            if adapter_summary["generation_seconds_median"] is not None
            and base_summary["generation_seconds_median"] is not None else None),
        "adapter_all_samples_valid": adapter_summary["one_pass_successes"] == len(rows),
        "adapter_teacher_evidence_better": (
            adapter_summary["teacher_evidence_accuracy"]
            > base_summary["teacher_evidence_accuracy"]),
    }
    result = {
        "schema_version": "labelagent.qwen3vl2b-lora-generation-eval.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "evaluated",
        "track": "teacher_distillation",
        "prompt_order": "text_then_image",
        "human_truth": False,
        "production_deployed": False,
        "test24_used": False,
        "evaluation_sequence": "008",
        "generation_cache_implementation": "dynamic",
        "dataset_manifest_sha256": file_sha256(args.dataset_dir / "manifest.json"),
        "contract_sha256": contract["contract_sha256"],
        "adapter_sha256": file_sha256(args.adapter_dir / "adapter_model.safetensors"),
        "base": base_summary,
        "adapter": adapter_summary,
        "comparison": comparison,
        "per_sample": {"base": base_rows, "adapter": adapter_rows},
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "peft": peft.__version__,
    }
    args.output.mkdir(parents=True)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    (args.output / "result.json").write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
