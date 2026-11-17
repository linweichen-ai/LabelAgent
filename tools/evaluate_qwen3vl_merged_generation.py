"""Evaluate an isolated merged Qwen3-VL checkpoint on frozen internal dev8."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

try:
    from tools.evaluate_qwen3vl_lora_generation import evaluate_mode, file_sha256
    from tools.evaluate_qwen3vl_lora_teacher_forcing import load_dev_rows
except ModuleNotFoundError:  # Flat, isolated remote bundle.
    from evaluate_qwen3vl_lora_generation import evaluate_mode, file_sha256
    from evaluate_qwen3vl_lora_teacher_forcing import load_dev_rows


def semantic_signature(content: dict) -> dict:
    assessments = content.get("assessments") if isinstance(content, dict) else None
    unknown = content.get("unknown_candidates") if isinstance(content, dict) else None
    if not isinstance(assessments, dict) or not isinstance(unknown, list):
        raise ValueError("structured content has no semantic signature")
    return {
        "assessments": {
            label: value.get("evidence") if isinstance(value, dict) else None
            for label, value in sorted(assessments.items())
        },
        "unknown_candidates": unknown,
    }


def compare_reference(rows: list[dict], reference_path: Path) -> dict:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    reference_rows = reference.get("per_sample", {}).get("adapter")
    if not isinstance(reference_rows, list):
        raise ValueError("reference result has no adapter rows")
    expected = {row.get("sample_id"): row for row in reference_rows}
    actual = {row.get("sample_id"): row for row in rows}
    if set(expected) != set(actual):
        raise ValueError("reference and merged sample sets differ")
    details = []
    for sample_id in sorted(actual):
        left, right = expected[sample_id], actual[sample_id]
        both_ok = left.get("status") == right.get("status") == "ok"
        exact = both_ok and left.get("content_sha256") == right.get("content_sha256")
        semantic = False
        if both_ok:
            semantic = semantic_signature(left["content"]) == semantic_signature(right["content"])
        details.append({
            "sample_id": sample_id,
            "reference_status": left.get("status"),
            "merged_status": right.get("status"),
            "exact_content_sha256_match": exact,
            "semantic_signature_match": semantic,
        })
    return {
        "reference_result_sha256": file_sha256(reference_path),
        "samples": len(details),
        "exact_content_sha256_matches": sum(
            row["exact_content_sha256_match"] for row in details),
        "semantic_signature_matches": sum(
            row["semantic_signature_match"] for row in details),
        "all_exact": all(row["exact_content_sha256_match"] for row in details),
        "all_semantic": all(row["semantic_signature_match"] for row in details),
        "details": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--server-module-dir", required=True, type=Path)
    parser.add_argument("--reference-result", required=True, type=Path)
    parser.add_argument("--cache-implementation", choices=("dynamic", "static"),
                        required=True)
    parser.add_argument("--require-reference-semantic-parity", action="store_true")
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
    config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_vl":
        raise ValueError("merged checkpoint is not dense Qwen3-VL")
    sys.path.insert(0, str(args.server_module_dir.resolve()))

    import torch
    import transformers
    from local_vlm_server import TransformersBackend

    static = args.cache_implementation == "static"
    backend = TransformersBackend(
        args.model_dir, "qwen3vl2b-merged", max_pixels=256 * 32 * 32,
        context_budget=16384, quantization="none",
        generation_cache_implementation=args.cache_implementation,
        startup_prewarm=static)
    adapter_sha256 = None
    if args.adapter_dir is not None:
        from peft import PeftModel
        adapter_weights = args.adapter_dir / "adapter_model.safetensors"
        if not adapter_weights.is_file():
            raise FileNotFoundError(adapter_weights)
        adapter_sha256 = file_sha256(adapter_weights)
        backend.model = PeftModel.from_pretrained(
            backend.model, args.adapter_dir, is_trainable=False)
        backend.model.eval()
    prewarm = None
    if static:
        prewarm = backend._run_startup_prewarm()
        backend.startup_prewarm = prewarm
        backend.model_loaded = True
        backend.last_inference_stats = None
    summary, result_rows = evaluate_mode(
        backend, rows, contract,
        f"{'adapter' if args.adapter_dir is not None else 'merged'}_"
        f"{args.cache_implementation}")
    parity = compare_reference(result_rows, args.reference_result)
    result = {
        "schema_version": "labelagent.qwen3vl2b-merged-generation-eval.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "evaluated_not_deployed",
        "track": "teacher_distillation",
        "prompt_order": "text_then_image",
        "human_truth": False,
        "production_deployed": False,
        "test24_used": False,
        "evaluation_sequence": "008",
        "generation_cache_implementation": args.cache_implementation,
        "candidate_kind": "unmerged_adapter" if args.adapter_dir is not None else "merged",
        "adapter_sha256": adapter_sha256,
        "dataset_manifest_sha256": file_sha256(args.dataset_dir / "manifest.json"),
        "contract_sha256": contract["contract_sha256"],
        "model_config_sha256": file_sha256(args.model_dir / "config.json"),
        "startup_prewarm": prewarm,
        "summary": summary,
        "reference_parity": parity,
        "reference_semantic_parity_required": args.require_reference_semantic_parity,
        "per_sample": result_rows,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    args.output.mkdir(parents=True)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    (args.output / "result.json").write_text(payload, encoding="utf-8")
    print(payload, end="")
    return (2 if args.require_reference_semantic_parity
            and not parity["all_semantic"] else 0)


if __name__ == "__main__":
    raise SystemExit(main())
