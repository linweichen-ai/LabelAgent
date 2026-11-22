"""Build and verify a local-only Qwen3-VL LoRA adapter bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def build(candidate_root: Path, manifest_path: Path, output: Path) -> dict:
    candidate_root = candidate_root.resolve()
    manifest_path = manifest_path.resolve()
    output = output.resolve()
    if output.exists():
        raise ValueError("choose a fresh output path")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "passed_local_release_audit":
        raise ValueError("release audit did not pass")
    boundaries = manifest.get("boundaries") or {}
    if boundaries.get("weights_published") is not False:
        raise ValueError("unexpected weight publication state")
    if boundaries.get("distribution_approved") is not False:
        raise ValueError("this builder is local-only")

    files: list[tuple[str, bytes, str]] = []
    for row in manifest.get("adapter_files") or []:
        relative = PurePosixPath(str(row.get("path") or ""))
        if (relative.is_absolute() or ".." in relative.parts
                or relative.parts[:2] != ("training", "adapter")):
            raise ValueError("unsafe adapter path")
        source = candidate_root.joinpath(*relative.parts).resolve(strict=True)
        if candidate_root not in source.parents or not source.is_file():
            raise ValueError("adapter file outside candidate root")
        payload = source.read_bytes()
        actual = sha256_bytes(payload)
        if actual != row.get("sha256") or len(payload) != row.get("bytes"):
            raise ValueError(f"adapter file drift: {relative.as_posix()}")
        name = "adapter/" + relative.name
        files.append((name, payload, actual))
    if not files or not any(name == "adapter/adapter_model.safetensors"
                            for name, _, _ in files):
        raise ValueError("adapter weights missing")

    config_payload = next(payload for name, payload, _ in files
                          if name == "adapter/adapter_config.json")
    config = json.loads(config_payload)
    base_model = Path(str(config.get("base_model_name_or_path") or "")).name
    if base_model != "Qwen3-VL-2B-Instruct":
        raise ValueError("unexpected base model identity")

    public_manifest = {
        "schema_version": "labelagent.qwen3vl-controlled-bundle.v1",
        "scope": "local_controlled_candidate",
        "base_model": base_model,
        "candidate": manifest["candidate"],
        "boundaries": {**boundaries, "bundle_local_only": True},
        "files": [{"path": name, "bytes": len(payload), "sha256": digest}
                  for name, payload, digest in files],
    }
    readme = (
        "LabelAgent Qwen3-VL-2B LoRA offline candidate\n"
        "\n"
        "This bundle is a local controlled artifact, not a production model or human truth.\n"
        "Base model: Qwen3-VL-2B-Instruct\n"
        "Verify every file against bundle-manifest.json before loading.\n"
        "test24 is evaluation-only and must never return to training.\n"
    ).encode("utf-8")
    manifest_bytes = (json.dumps(public_manifest, ensure_ascii=False, indent=2)
                      + "\n").encode("utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", allowZip64=True) as archive:
        for name, payload, _ in sorted(files):
            archive.writestr(_zip_info(name), payload)
        archive.writestr(_zip_info("bundle-manifest.json"), manifest_bytes)
        archive.writestr(_zip_info("README.txt"), readme)

    expected = {name: digest for name, _, digest in files}
    expected["bundle-manifest.json"] = sha256_bytes(manifest_bytes)
    expected["README.txt"] = sha256_bytes(readme)
    with zipfile.ZipFile(output, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError("bundle member set differs")
        for name in names:
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts:
                raise ValueError("unsafe bundle member")
            if sha256_bytes(archive.read(name)) != expected[name]:
                raise ValueError(f"bundle member hash differs: {name}")

    return {
        "schema_version": "labelagent.qwen3vl-controlled-bundle-receipt.v1",
        "status": "built_and_readback_verified",
        "scope": "local_controlled_candidate",
        "bundle_name": output.name,
        "bundle_bytes": output.stat().st_size,
        "bundle_sha256": sha256_file(output),
        "member_count": len(expected),
        "adapter_sha256": manifest["candidate"]["adapter_sha256"],
        "base_model": base_model,
        "remote_uploaded": False,
        "production_deployed": False,
        "human_truth": False,
        "test24_used_for_training": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--release-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    receipt = build(args.candidate_root, args.release_manifest, args.output)
    receipt_path = args.receipt.resolve()
    if receipt_path.exists():
        raise ValueError("choose a fresh receipt path")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
