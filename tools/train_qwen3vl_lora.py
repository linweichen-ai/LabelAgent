"""Isolated Qwen3-VL-2B LoRA smoke/full training on audited teacher JSONL."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("training JSONL is empty")
    for row in rows:
        if (row.get("split") != "train" or row.get("track") != "teacher_distillation"
                or row.get("candidate_kind") != "machine_teacher_output"
                or row.get("human_truth") is not False
                or row.get("production_training_approved") is not False
                or row.get("requires_acceptance_gate") is not True):
            raise ValueError("training row boundary differs")
    return rows


def verify_manifest(manifest_path: Path, train_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("status") != "ready_for_isolated_lora_smoke"
            or manifest.get("track") != "teacher_distillation"
            or manifest.get("human_truth") is not False
            or manifest.get("production_training_approved") is not False
            or manifest.get("requires_acceptance_gate") is not True
            or manifest.get("test24_included") is not False):
        raise ValueError("distillation manifest boundary differs")
    if manifest.get("train_jsonl_sha256") != file_sha256(train_path):
        raise ValueError("training JSONL hash differs")
    return manifest


def slope_balanced_rows(rows: list[dict], repeat: int) -> tuple[list[dict], dict]:
    """Repeat train-only slope-positive teacher rows without changing truth status.

    This is deliberately a narrow distillation aid.  It does not invent labels,
    does not read the external test split, and keeps every duplicated row bound
    to the original image/target hashes.
    """
    if not 1 <= repeat <= 8:
        raise ValueError("slope-positive repeat must be between 1 and 8")
    positives = []
    for row in rows:
        try:
            target = json.loads(row["target"])
            evidence = target["assessments"]["slope"]["evidence"]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("training row has no valid slope evidence") from error
        if evidence not in {"present", "absent", "uncertain"}:
            raise ValueError("training row has invalid slope evidence")
        if evidence == "present":
            positives.append(row)
    balanced = list(rows)
    for _ in range(repeat - 1):
        balanced.extend(positives)
    return balanced, {
        "original_samples": len(rows),
        "original_slope_positive_samples": len(positives),
        "slope_positive_repeat": repeat,
        "effective_samples": len(balanced),
        "effective_slope_positive_samples": len(positives) * repeat,
    }


class OneImageCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, rows):
        if len(rows) != 1:
            raise ValueError("this audited collator requires batch size 1")
        import torch
        from PIL import Image, ImageOps
        row = rows[0]
        image_path = Path(row["image_path"])
        if file_sha256(image_path) != row["image_sha256"]:
            raise ValueError("training image hash differs")
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        # Match source/agent/workflow.py::_chat exactly: production sends the
        # instruction first and appends image inputs after it.
        user = {"role": "user", "content": [
            {"type": "text", "text": row["instruction"]}, {"type": "image"}]}
        assistant = {"role": "assistant", "content": [
            {"type": "text", "text": row["target"]}]}
        prompt_text = self.processor.apply_chat_template(
            [user], tokenize=False, add_generation_prompt=True)
        full_text = self.processor.apply_chat_template(
            [user, assistant], tokenize=False, add_generation_prompt=False)
        prompt = self.processor(text=[prompt_text], images=[image],
                                padding=False, return_tensors="pt")
        full = self.processor(text=[full_text], images=[image],
                              padding=False, return_tensors="pt")
        prompt_ids, full_ids = prompt["input_ids"][0], full["input_ids"][0]
        if len(prompt_ids) >= len(full_ids) or not torch.equal(full_ids[:len(prompt_ids)], prompt_ids):
            raise ValueError("assistant target is not an exact prompt suffix")
        labels = full["input_ids"].clone()
        labels[:, :len(prompt_ids)] = -100
        full["labels"] = labels
        return full


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--slope-positive-repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    train_path = args.dataset_dir / "train.jsonl"
    manifest_path = args.dataset_dir / "manifest.json"
    manifest = verify_manifest(manifest_path, train_path)
    source_rows = load_jsonl(train_path)
    rows, sampling = slope_balanced_rows(source_rows, args.slope_positive_repeat)
    random.Random(args.seed).shuffle(rows)
    rows = rows[:args.max_samples]
    if not 1 <= args.max_steps <= 100 or not 1 <= args.max_samples <= len(rows):
        raise ValueError("unsafe smoke/full training bounds")

    import torch
    import transformers
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=False)
    collator = OneImageCollator(processor)
    first_batch = collator([rows[0]])
    if args.dry_run:
        result = {
            "status": "dry_run_passed", "samples": len(rows),
            "prompt_order": "text_then_image",
            "input_shape": list(first_batch["input_ids"].shape),
            "supervised_tokens": int((first_batch["labels"] != -100).sum()),
            "image_grid_thw": first_batch.get("image_grid_thw").tolist(),
            "dataset_manifest_sha256": file_sha256(manifest_path),
            "sampling": sampling,
            "training_started": False,
        }
        args.output.mkdir(parents=True)
        (args.output / "run_receipt.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0

    import peft
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for LoRA training")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation="sdpa")
    model.config.use_cache = False
    model.enable_input_require_grads()
    lora = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    args.output.mkdir(parents=True)
    training_args = TrainingArguments(
        output_dir=str(args.output), max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, lr_scheduler_type="constant",
        warmup_steps=0, logging_steps=1, save_strategy="no",
        bf16=True, tf32=True, gradient_checkpointing=True,
        remove_unused_columns=False, report_to=[], dataloader_num_workers=0,
        seed=args.seed, data_seed=args.seed, optim="adamw_torch",
    )
    class Rows(torch.utils.data.Dataset):
        def __len__(self):
            return len(rows)
        def __getitem__(self, index):
            return rows[index]
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trainer = Trainer(model=model, args=training_args,
                      train_dataset=Rows(), data_collator=collator)
    output = trainer.train()
    model.save_pretrained(args.output / "adapter")
    processor.save_pretrained(args.output / "adapter")
    adapter_path = args.output / "adapter" / "adapter_model.safetensors"
    receipt = {
        "schema_version": "labelagent.qwen3vl2b-lora-run.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "smoke_complete" if args.max_steps <= 2 else "training_complete",
        "track": "teacher_distillation",
        "prompt_order": "text_then_image",
        "human_truth": False,
        "production_deployed": False,
        "test24_used_for_training": False,
        "model_dir": str(args.model_dir),
        "model_config_sha256": file_sha256(args.model_dir / "config.json"),
        "dataset_manifest_sha256": file_sha256(manifest_path),
        "train_jsonl_sha256": manifest["train_jsonl_sha256"],
        "samples_available": manifest["train_samples"],
        "samples_loaded": len(rows),
        "sampling": sampling,
        "steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_percent": trainable / total * 100,
        "train_loss": output.training_loss,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "adapter_sha256": file_sha256(adapter_path),
        "step_metrics": [item for item in trainer.state.log_history if "loss" in item],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
    }
    (args.output / "run_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
