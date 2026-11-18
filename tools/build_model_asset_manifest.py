"""Build the one-time full-content manifest used by the VLM response cache.

The output must live outside the model directory.  Runtime startup validates
the complete selected file set, sizes and nanosecond mtimes against this
manifest, while the expensive full SHA-256 pass is performed only here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import uuid


ASSET_SUFFIXES = {".json", ".txt", ".safetensors"}
REQUIRED_ASSETS = {
    "config.json",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
}


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(model_dir):
    root = Path(model_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("Model directory does not exist")
    assets = sorted(
        (path for path in root.iterdir()
         if path.is_file() and path.suffix.lower() in ASSET_SUFFIXES),
        key=lambda path: path.name)
    names = {path.name for path in assets}
    if not REQUIRED_ASSETS.issubset(names):
        raise ValueError("Model directory lacks required runtime files")
    if not any(name.endswith(".safetensors") for name in names):
        raise ValueError("Model directory has no safetensors weights")
    records = []
    for path in assets:
        before = path.stat()
        digest = sha256_file(path)
        after = path.stat()
        if (before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns):
            raise RuntimeError(
                f"Model asset changed while it was hashed: {path.name}")
        records.append({"name": path.name, "size": after.st_size,
                        "mtime_ns": after.st_mtime_ns, "sha256": digest})
    return {"schema_version": "labelagent.model-assets.v1",
            "files": records}


def write_manifest(model_dir, output_path, force=False):
    root = Path(model_dir).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if output == root or output.is_relative_to(root):
        raise ValueError("Manifest output must be outside the model directory")
    if output.exists() and not force:
        raise FileExistsError("Manifest already exists; use --force to replace it")
    payload = build_manifest(root)
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":")) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as target:
            target.write(encoded)
            target.flush()
            os.fsync(target.fileno())
        try:
            os.chmod(temporary, 0o600)
        except OSError:
            pass
        os.replace(temporary, output)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return {"path": str(output), "files": len(payload["files"]),
            "bytes": sum(item["size"] for item in payload["files"]),
            "manifest_sha256": hashlib.sha256(encoded).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(write_manifest(
        args.model_dir, args.output, force=args.force), sort_keys=True))


if __name__ == "__main__":
    main()
