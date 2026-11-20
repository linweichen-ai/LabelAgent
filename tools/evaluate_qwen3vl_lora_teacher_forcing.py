"""Compare base and LoRA Qwen3-VL on held-out teacher targets without generation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

try:
    from tools.train_qwen3vl_lora import OneImageCollator
except ModuleNotFoundError:  # Flat, isolated remote bundle.
    from train_qwen3vl_lora import OneImageCollator


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_eval_rows(dataset_dir: Path, split: str = "dev") -> tuple[list[dict], dict]:
    if split not in {"dev", "holdout"}:
        raise ValueError("evaluation split is invalid")
    manifest_path = dataset_dir / "manifest.json"
    dev_path = dataset_dir / f"{split}.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("status") != "ready_for_isolated_lora_smoke"
            or manifest.get("track") != "teacher_distillation"
            or manifest.get("human_truth") is not False
            or manifest.get("production_training_approved") is not False
            or manifest.get("requires_acceptance_gate") is not True
            or manifest.get("test24_included") is not False):
        raise ValueError("distillation manifest boundary differs")
    if manifest.get(f"{split}_jsonl_sha256") != file_sha256(dev_path):
        raise ValueError(f"{split} JSONL hash differs")
    rows = [json.loads(line) for line in dev_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if len(rows) != manifest.get(f"{split}_samples") or not rows:
        raise ValueError(f"{split} sample count differs")
    seen = set()
    allowed_sequences = manifest.get(f"{split}_source_sequences") \
        or manifest.get("dev_source_sequences") or ["008"]
    if (not isinstance(allowed_sequences, list) or not allowed_sequences
            or any(not isinstance(value, str) or not value for value in allowed_sequences)):
        raise ValueError("development source sequence boundary differs")
    for row in rows:
        if (row.get("split") != split
                or row.get("source_sequence") not in allowed_sequences
                or row.get("track") != "teacher_distillation"
                or row.get("candidate_kind") != "machine_teacher_output"
                or row.get("human_truth") is not False
                or row.get("production_training_approved") is not False
                or row.get("requires_acceptance_gate") is not True):
            raise ValueError("development row boundary differs")
        if row.get("sample_id") in seen:
            raise ValueError("duplicate development sample")
        seen.add(row.get("sample_id"))
    return rows, manifest


def load_dev_rows(dataset_dir: Path) -> tuple[list[dict], dict]:
    return load_eval_rows(dataset_dir, "dev")


def summarize(per_sample: list[dict]) -> dict:
    tokens = sum(item["supervised_tokens"] for item in per_sample)
    if not tokens:
        raise ValueError("no supervised development tokens")
    nll = sum(item["nll_sum"] for item in per_sample) / tokens
    correct = sum(item["correct_tokens"] for item in per_sample)
    return {
        "samples": len(per_sample),
        "supervised_tokens": tokens,
        "mean_nll": nll,
        "perplexity": math.exp(min(nll, 20.0)),
        "token_accuracy": correct / tokens,
        "correct_tokens": correct,
        "elapsed_seconds": sum(item["elapsed_seconds"] for item in per_sample),
    }


def evaluate(model, processor, rows: list[dict], label: str) -> tuple[dict, list[dict]]:
    import torch

    collator = OneImageCollator(processor)
    model.eval()
    per_sample = []
    for row in rows:
        batch = collator([row])
        batch = {key: value.to(model.device) if torch.is_tensor(value) else value
                 for key, value in batch.items()}
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = model(**batch, use_cache=False)
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
        elapsed = time.perf_counter() - started
        shifted_labels = batch["labels"][:, 1:]
        predicted = outputs.logits[:, :-1].argmax(dim=-1)
        mask = shifted_labels != -100
        supervised = int(mask.sum())
        correct = int(((predicted == shifted_labels) & mask).sum())
        loss = float(outputs.loss)
        per_sample.append({
            "sample_id": row["sample_id"],
            "mode": label,
            "supervised_tokens": supervised,
            "correct_tokens": correct,
            "nll": loss,
            "nll_sum": loss * supervised,
            "elapsed_seconds": round(elapsed, 6),
        })
        print(json.dumps({
            "mode": label, "sample_id": row["sample_id"],
            "nll": loss, "supervised_tokens": supervised,
            "correct_tokens": correct, "elapsed_seconds": round(elapsed, 6),
        }, ensure_ascii=False), flush=True)
        del outputs, predicted, shifted_labels, mask, batch
    return summarize(per_sample), per_sample


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--quantization", default="none", choices=("none", "nf4"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows, manifest = load_dev_rows(args.dataset_dir)

    import peft
    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for teacher-forcing evaluation")
    processor = AutoProcessor.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=False)
    model_kwargs = {
        "local_files_only": True,
        "trust_remote_code": False,
        "dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
    }
    if args.quantization == "nf4":
        from transformers import BitsAndBytesConfig
        model_kwargs.update({
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            "device_map": {"": 0},
        })
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_dir, **model_kwargs)
    if args.quantization == "none":
        model.to(torch.device("cuda"))
    model.config.use_cache = False
    torch.cuda.reset_peak_memory_stats()
    base_summary, base_rows = evaluate(model, processor, rows, "base")
    adapter = PeftModel.from_pretrained(model, args.adapter_dir, is_trainable=False)
    adapter_summary, adapter_rows = evaluate(adapter, processor, rows, "adapter")
    comparison = {
        "mean_nll_delta": adapter_summary["mean_nll"] - base_summary["mean_nll"],
        "mean_nll_relative_change": (
            adapter_summary["mean_nll"] / base_summary["mean_nll"] - 1.0),
        "token_accuracy_delta": (
            adapter_summary["token_accuracy"] - base_summary["token_accuracy"]),
        "nll_improved": adapter_summary["mean_nll"] < base_summary["mean_nll"],
        "token_accuracy_not_worse": (
            adapter_summary["token_accuracy"] >= base_summary["token_accuracy"]),
    }
    evaluation_sequences = manifest.get("dev_source_sequences") or ["008"]
    result = {
        "schema_version": "labelagent.qwen3vl-lora-teacher-forcing.v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "evaluated",
        "track": "teacher_distillation",
        "prompt_order": "text_then_image",
        "human_truth": False,
        "production_deployed": False,
        "test24_used": False,
        "quantization": args.quantization,
        "evaluation_sequence": (evaluation_sequences[0]
                                if len(evaluation_sequences) == 1 else None),
        "evaluation_sequences": evaluation_sequences,
        "dataset_manifest_sha256": file_sha256(args.dataset_dir / "manifest.json"),
        "dev_jsonl_sha256": manifest["dev_jsonl_sha256"],
        "adapter_sha256": file_sha256(args.adapter_dir / "adapter_model.safetensors"),
        "base": base_summary,
        "adapter": adapter_summary,
        "comparison": comparison,
        "per_sample": {"base": base_rows, "adapter": adapter_rows},
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
