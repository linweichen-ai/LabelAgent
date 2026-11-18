"""Run a frozen LoRA candidate on unlabelled images without creating truth."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_shadow_manifest(manifest: dict, contract: dict) -> list[dict]:
    """Keep unlabelled shadow inputs outside truth and production paths."""
    images = manifest.get("images")
    if (manifest.get("schema_version") != "labelagent.qwen3vl-shadow-manifest.v1"
            or manifest.get("human_truth") is not False
            or manifest.get("candidate_only") is not True
            or manifest.get("production_deployed") is not False
            or manifest.get("contract_sha256") != contract.get("contract_sha256")
            or not isinstance(images, list) or not images):
        raise ValueError("shadow manifest boundary differs")
    sample_ids = []
    for item in images:
        if (not isinstance(item, dict) or not item.get("sample_id")
                or not item.get("image_path") or not item.get("image_sha256")):
            raise ValueError("shadow image entry is incomplete")
        review = item.get("assistant_review_candidate")
        if review is not None and (not isinstance(review, dict)
                or review.get("review_kind") != "assistant_visual_candidate"
                or review.get("human_truth") is not False
                or review.get("expected_evidence") not in {"present", "absent"}
                or not review.get("reason")):
            raise ValueError("assistant shadow review boundary differs")
        sample_ids.append(item["sample_id"])
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("shadow sample_id values must be unique")
    return images


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--server-module-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    images = verify_shadow_manifest(manifest, contract)
    sys.path.insert(0, str(args.server_module_dir.resolve()))

    import peft
    import torch
    import transformers
    from peft import PeftModel
    from PIL import Image, ImageOps
    from local_vlm_server import TransformersBackend

    backend = TransformersBackend(
        args.model_dir, "qwen3vl2b-other-robot-shadow", max_pixels=256 * 32 * 32,
        context_budget=16384, quantization="none",
        generation_cache_implementation="dynamic", startup_prewarm=False)
    backend.model = PeftModel.from_pretrained(
        backend.model, args.adapter_dir, is_trainable=False)
    backend.model.eval()
    rows = []
    for item in images:
        started = time.perf_counter()
        image_path = Path(item["image_path"])
        if file_sha256(image_path) != item["image_sha256"]:
            raise ValueError(f"shadow image hash differs: {item.get('sample_id')}")
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "text", "text": contract["instruction"]},
            {"type": "image"},
        ]}]
        content, prompt_tokens, completion_tokens = backend.generate(
            messages, [image], contract["max_tokens"],
            response_schema=contract["response_schema"])
        parsed = json.loads(content)
        review = item.get("assistant_review_candidate")
        predicted_evidence = (parsed.get("assessments", {})
                              .get("other_robot", {}).get("evidence"))
        rows.append({
            "sample_id": item["sample_id"],
            "source_kind": item.get("source_kind"),
            "source_ref": item.get("source_ref"),
            "status": "candidate_generated",
            "candidate_only": True,
            "human_truth": False,
            "assistant_review_candidate": review,
            "assistant_candidate_agreement": (
                predicted_evidence == review["expected_evidence"]
                if review is not None else None),
            "prediction": parsed,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "generation_seconds": (backend.last_inference_stats or {}).get(
                "generation_seconds"),
            "wall_seconds": round(time.perf_counter() - started, 6),
        })
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    reviewed_rows = [row for row in rows
                     if row["assistant_review_candidate"] is not None]
    result = {
        "schema_version": "labelagent.qwen3vl-shadow-result.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "shadow_candidates_generated",
        "human_truth": False,
        "candidate_only": True,
        "production_deployed": False,
        "accuracy_computed": False,
        "assistant_candidate_review": {
            "reviewed": len(reviewed_rows),
            "agreements": sum(row["assistant_candidate_agreement"] is True
                              for row in reviewed_rows),
            "disagreements": sum(row["assistant_candidate_agreement"] is False
                                 for row in reviewed_rows),
            "is_accuracy": False,
            "boundary": "Assistant visual review is triage only, not human truth.",
        },
        "manifest_sha256": file_sha256(args.manifest),
        "contract_sha256": contract["contract_sha256"],
        "adapter_sha256": file_sha256(
            args.adapter_dir / "adapter_model.safetensors"),
        "rows": rows,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "boundary": "Unlabelled shadow outputs are candidates, not accuracy evidence or truth.",
    }
    args.output.mkdir(parents=True)
    (args.output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
