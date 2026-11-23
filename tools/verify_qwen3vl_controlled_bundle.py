"""Independently verify a controlled LabelAgent Qwen3-VL LoRA ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import zipfile


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, expected_sha256: str | None = None,
           tensor_scan: bool = False) -> dict:
    path = path.resolve(strict=True)
    bundle_hash = file_sha256(path)
    if expected_sha256 and bundle_hash != expected_sha256.lower():
        raise ValueError("bundle SHA-256 differs")
    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate ZIP member")
        for name in names:
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts or "\\" in name:
                raise ValueError("unsafe ZIP member")
        required = {"README.txt", "bundle-manifest.json",
                    "adapter/adapter_config.json",
                    "adapter/adapter_model.safetensors"}
        if not required.issubset(names):
            raise ValueError("required ZIP member missing")
        manifest = json.loads(archive.read("bundle-manifest.json"))
        if manifest.get("schema_version") != "labelagent.qwen3vl-controlled-bundle.v1":
            raise ValueError("bundle schema differs")
        if manifest.get("scope") != "local_controlled_candidate":
            raise ValueError("bundle scope differs")
        boundaries = manifest.get("boundaries") or {}
        if (boundaries.get("human_truth") is not False
                or boundaries.get("production_deployed") is not False
                or boundaries.get("test24_used_for_training") is not False
                or boundaries.get("bundle_local_only") is not True):
            raise ValueError("bundle safety boundary differs")
        rows = manifest.get("files") or []
        expected_names = {str(row.get("path")) for row in rows}
        if set(names) != expected_names | {"README.txt", "bundle-manifest.json"}:
            raise ValueError("ZIP members differ from manifest")
        for row in rows:
            payload = archive.read(row["path"])
            if len(payload) != row.get("bytes") or sha256(payload) != row.get("sha256"):
                raise ValueError(f"member drift: {row['path']}")
        config = json.loads(archive.read("adapter/adapter_config.json"))
        if Path(str(config.get("base_model_name_or_path") or "")).name != manifest.get("base_model"):
            raise ValueError("base model identity differs")
        adapter_payload = archive.read("adapter/adapter_model.safetensors")

    tensor_result = {"performed": False}
    if tensor_scan:
        from safetensors import safe_open

        with tempfile.TemporaryDirectory() as directory:
            adapter_path = Path(directory) / "adapter_model.safetensors"
            adapter_path.write_bytes(adapter_payload)
            all_finite = True
            nonzero_a = nonzero_b = 0
            with safe_open(adapter_path, framework="pt", device="cpu") as tensors:
                keys = list(tensors.keys())
                lora_a = [key for key in keys if "lora_A" in key]
                lora_b = [key for key in keys if "lora_B" in key]
                for key in keys:
                    tensor = tensors.get_tensor(key)
                    all_finite = all_finite and bool(tensor.isfinite().all())
                    if key in lora_a:
                        nonzero_a += int(bool((tensor != 0).any()))
                    if key in lora_b:
                        nonzero_b += int(bool((tensor != 0).any()))
            tensor_result = {
                "performed": True,
                "tensor_count": len(keys),
                "all_finite": all_finite,
                "lora_a_tensors": len(lora_a),
                "nonzero_lora_a_tensors": nonzero_a,
                "lora_b_tensors": len(lora_b),
                "nonzero_lora_b_tensors": nonzero_b,
                "passed": (all_finite and bool(lora_a) and bool(lora_b)
                           and nonzero_a == len(lora_a) and nonzero_b == len(lora_b)),
            }
            if not tensor_result["passed"]:
                raise ValueError("adapter tensor scan failed")

    return {
        "schema_version": "labelagent.qwen3vl-controlled-bundle-verification.v1",
        "status": "passed",
        "bundle_name": path.name,
        "bundle_bytes": path.stat().st_size,
        "bundle_sha256": bundle_hash,
        "member_count": len(names),
        "adapter_sha256": sha256(adapter_payload),
        "base_model": manifest["base_model"],
        "tensor_scan": tensor_result,
        "remote_uploaded": False,
        "production_deployed": False,
        "human_truth": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--tensor-scan", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.bundle, args.expected_sha256, args.tensor_scan)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = args.output.resolve()
        if output.exists():
            raise ValueError("choose a fresh output path")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
