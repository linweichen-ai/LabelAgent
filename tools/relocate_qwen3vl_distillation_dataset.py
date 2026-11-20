"""Relocate audited Qwen3-VL JSONL image paths without changing labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--old-prefix", required=True)
    parser.add_argument("--new-prefix", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    manifest = json.loads((args.input / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("test24_included") is not False:
        raise ValueError("test24 boundary differs")
    args.output.mkdir(parents=True)
    relocated = 0
    missing: list[str] = []
    for name in ("train.jsonl", "dev.jsonl"):
        source = args.input / name
        if not source.is_file():
            continue
        rows = []
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            old_path = str(row["image_path"])
            if not old_path.startswith(args.old_prefix):
                raise ValueError(f"unexpected image prefix: {old_path}")
            row["image_path"] = args.new_prefix + old_path[len(args.old_prefix):]
            image = Path(row["image_path"])
            if not image.is_file() or sha256_file(image) != row["image_sha256"]:
                missing.append(str(image))
            rows.append(row)
            relocated += 1
        target = args.output / name
        target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                          encoding="utf-8")
    if missing:
        raise ValueError(f"missing or hash-mismatched images: {len(missing)}")

    manifest.update({
        "relocated_only": True,
        "relocated_rows": relocated,
        "relocated_from_prefix": args.old_prefix,
        "relocated_to_prefix": args.new_prefix,
        "train_jsonl_sha256": sha256_file(args.output / "train.jsonl"),
    })
    if (args.output / "dev.jsonl").is_file():
        manifest["dev_jsonl_sha256"] = sha256_file(args.output / "dev.jsonl")
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "relocated",
        "rows": relocated,
        "test24_included": manifest["test24_included"],
        "train_jsonl_sha256": manifest["train_jsonl_sha256"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
