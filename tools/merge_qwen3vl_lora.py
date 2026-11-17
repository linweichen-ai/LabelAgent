"""Merge an audited Qwen3-VL LoRA into an isolated CPU checkpoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_manifest(root: str | Path) -> list[dict]:
    root = Path(root)
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == "merge_receipt.json":
            continue
        rows.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        })
    return rows


def validate_inputs(model_dir: Path, adapter_dir: Path, output: Path,
                    expected_adapter_sha256: str | None) -> str:
    if output.exists():
        raise FileExistsError(output)
    model_config = model_dir / "config.json"
    adapter_config = adapter_dir / "adapter_config.json"
    adapter_weights = adapter_dir / "adapter_model.safetensors"
    for required in (model_config, adapter_config, adapter_weights):
        if not required.is_file():
            raise FileNotFoundError(required)
    config = json.loads(model_config.read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_vl":
        raise ValueError("base checkpoint is not dense Qwen3-VL")
    adapter_sha256 = file_sha256(adapter_weights)
    if (expected_adapter_sha256 is not None
            and adapter_sha256 != expected_adapter_sha256.lower()):
        raise ValueError("adapter SHA-256 differs from the audited input")
    return adapter_sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-adapter-sha256")
    args = parser.parse_args()
    model_dir = args.model_dir.resolve()
    adapter_dir = args.adapter_dir.resolve()
    output = args.output.resolve()
    adapter_sha256 = validate_inputs(
        model_dir, adapter_dir, output, args.expected_adapter_sha256)
    partial = output.with_name(f".{output.name}.partial-{uuid.uuid4().hex}")
    if partial.exists():
        raise FileExistsError(partial)

    import peft
    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        raise RuntimeError("merge must be explicitly isolated from CUDA")
    started = time.monotonic()
    partial.mkdir(parents=False)
    try:
        merged_dir = partial / "model"
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_dir), local_files_only=True, dtype=torch.bfloat16,
            device_map={"": "cpu"}, low_cpu_mem_usage=True)
        model = PeftModel.from_pretrained(
            model, str(adapter_dir), is_trainable=False)
        merged = model.merge_and_unload(safe_merge=True)
        merged.save_pretrained(
            str(merged_dir), safe_serialization=True, max_shard_size="4GB")
        processor = AutoProcessor.from_pretrained(
            str(model_dir), local_files_only=True)
        processor.save_pretrained(str(merged_dir))
        manifest = checkpoint_manifest(merged_dir)
        if not any(row["path"].endswith(".safetensors") for row in manifest):
            raise RuntimeError("merged checkpoint has no safetensors weights")
        receipt = {
            "schema_version": "labelagent.qwen3vl-lora-merge.v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "merged_isolated_not_deployed",
            "base_model_dir": str(model_dir),
            "base_config_sha256": file_sha256(model_dir / "config.json"),
            "adapter_dir": str(adapter_dir),
            "adapter_sha256": adapter_sha256,
            "human_truth": False,
            "production_deployed": False,
            "test24_used": False,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "checkpoint_files": manifest,
            "checkpoint_bytes": sum(row["bytes"] for row in manifest),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        }
        (partial / "merge_receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        partial.rename(output)
    except BaseException:
        if partial.exists():
            shutil.rmtree(partial)
        raise
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
