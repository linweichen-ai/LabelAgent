"""Build a fail-closed release manifest for an offline Qwen3-VL LoRA candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED = (
    "dataset/train.jsonl",
    "dataset/dev.jsonl",
    "dataset/manifest.json",
    "training/run_receipt.json",
    "training/audit.json",
    "training/adapter/adapter_config.json",
    "training/adapter/adapter_model.safetensors",
)
SENSITIVE_KEYS = {
    "access_key", "access_key_id", "access_key_secret", "api_key", "password",
    "secret", "secret_key", "token", "authorization",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def sensitive_paths(value: Any, prefix: str = "$") -> list[str]:
    hits: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            child_path = f"{prefix}.{key}"
            if normalized in SENSITIVE_KEYS and child not in (None, "", False):
                hits.append(child_path)
            hits.extend(sensitive_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            hits.extend(sensitive_paths(child, f"{prefix}[{index}]"))
    return hits


def audit(root: Path, summary_path: Path, source_paths: list[Path]) -> dict[str, Any]:
    root = root.resolve()
    errors: list[str] = []
    for name in REQUIRED:
        if not (root / name).is_file():
            errors.append(f"missing required file: {name}")

    if errors:
        return {"schema_version": "labelagent.qwen3vl-release-manifest.v1",
                "status": "failed", "errors": errors}

    dataset = read_json(root / "dataset/manifest.json")
    receipt = read_json(root / "training/run_receipt.json")
    tensor_audit = read_json(root / "training/audit.json")
    config = read_json(root / "training/adapter/adapter_config.json")
    summary = read_json(summary_path.resolve())

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(receipt.get("status") == "training_complete", "training is not complete")
    require(receipt.get("track") == "teacher_distillation", "track differs")
    require(receipt.get("human_truth") is False, "receipt claims human truth")
    require(receipt.get("production_deployed") is False, "receipt claims deployment")
    require(receipt.get("test24_used_for_training") is False, "test24 entered training")
    require(dataset.get("test24_included") is False, "dataset includes test24")
    require(dataset.get("human_truth") is False, "dataset claims human truth")
    require(dataset.get("production_training_approved") is False,
            "dataset claims production approval")
    require(tensor_audit.get("status") == "passed", "tensor audit failed")
    require(config.get("r") == 8, "LoRA rank differs")

    train_hash = sha256(root / "dataset/train.jsonl")
    dev_hash = sha256(root / "dataset/dev.jsonl")
    adapter_hash = sha256(root / "training/adapter/adapter_model.safetensors")
    require(dataset.get("train_jsonl_sha256") == train_hash, "train hash differs")
    require(dataset.get("dev_jsonl_sha256") == dev_hash, "dev hash differs")
    require(receipt.get("train_jsonl_sha256") == train_hash, "receipt train hash differs")
    require(receipt.get("adapter_sha256") == adapter_hash, "receipt adapter hash differs")
    require(tensor_audit.get("adapter_sha256") == adapter_hash,
            "tensor-audit adapter hash differs")
    require(summary.get("training", {}).get("adapter_sha256") == adapter_hash,
            "summary adapter hash differs")
    require(summary.get("human_truth") is False, "summary claims human truth")
    require(summary.get("production_deployed") is False, "summary claims deployment")
    require(summary.get("test24_used_for_training") is False,
            "summary claims test24 training")

    for name, value in (("dataset manifest", dataset), ("training receipt", receipt),
                        ("tensor audit", tensor_audit), ("adapter config", config),
                        ("summary", summary)):
        for path in sensitive_paths(value):
            errors.append(f"sensitive value in {name}: {path}")

    files = []
    for path in sorted((root / "training/adapter").iterdir(), key=lambda p: p.name):
        if path.is_file():
            files.append({"path": path.relative_to(root).as_posix(),
                          "bytes": path.stat().st_size, "sha256": sha256(path)})
    sources = []
    for path in source_paths:
        resolved = path.resolve()
        require(resolved.is_file(), f"missing source: {path}")
        if resolved.is_file():
            sources.append({"path": path.as_posix(), "sha256": sha256(resolved)})

    return {
        "schema_version": "labelagent.qwen3vl-release-manifest.v1",
        "status": "passed_local_release_audit" if not errors else "failed",
        "errors": errors,
        "candidate": {
            "kind": "offline_machine_teacher_lora",
            "adapter_sha256": adapter_hash,
            "adapter_bytes": (root / "training/adapter/adapter_model.safetensors").stat().st_size,
            "lora_rank": config.get("r"),
            "training_steps": receipt.get("steps"),
            "train_samples": dataset.get("train_samples"),
            "dev_samples": dataset.get("dev_samples"),
        },
        "boundaries": {
            "human_truth": False,
            "production_training_approved": False,
            "production_deployed": False,
            "test24_used_for_training": False,
            "weights_published": False,
            "distribution_approved": False,
        },
        "dataset": {"train_sha256": train_hash, "dev_sha256": dev_hash},
        "adapter_files": files,
        "source_files": sources,
        "next_gate": "explicit controlled distribution decision and independent human evaluation",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--source", action="append", default=[], type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    value = audit(args.root, args.summary, args.source)
    output = args.output.resolve()
    if output.exists():
        raise ValueError("choose a fresh output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps(value, ensure_ascii=False))
    return 0 if value["status"] == "passed_local_release_audit" else 1


if __name__ == "__main__":
    raise SystemExit(main())
