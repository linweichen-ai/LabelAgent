"""Build an internal dev8 prompt variant for partial/edge slope discovery."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


SLOPE_GUIDANCE = (
    "Slope visibility rule: judge slope from surface geometry, not color. "
    "Report slope as present when any visible part of an inclined ramp or "
    "transition surface appears, including a partial fragment at an image "
    "edge or behind an occluder. Do not relabel ordinary level floor as "
    "slope. Slope is traversable like floor downstream, but report its "
    "presence independently in this discovery response."
)


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_dev_rows(source: Path) -> tuple[list[dict], dict]:
    manifest_path = source / "manifest.json"
    dev_path = source / "dev.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("status") != "ready_for_isolated_lora_smoke"
            or manifest.get("track") != "teacher_distillation"
            or manifest.get("human_truth") is not False
            or manifest.get("production_training_approved") is not False
            or manifest.get("requires_acceptance_gate") is not True
            or manifest.get("test24_included") is not False
            or manifest.get("dev_samples") != 8
            or manifest.get("dev_jsonl_sha256") != file_sha256(dev_path)):
        raise ValueError("source dev dataset boundary differs")
    rows = [json.loads(line) for line in dev_path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 8 or len({row.get("sample_id") for row in rows}) != 8:
        raise ValueError("source dev8 is incomplete or duplicated")
    for row in rows:
        if (row.get("split") != "dev"
                or row.get("source_sequence") != "008"
                or row.get("track") != "teacher_distillation"
                or row.get("candidate_kind") != "machine_teacher_output"
                or row.get("human_truth") is not False
                or row.get("production_training_approved") is not False
                or row.get("requires_acceptance_gate") is not True
                or not isinstance(row.get("instruction"), str)):
            raise ValueError("source dev row boundary differs")
    return rows, manifest


def build_rows(rows: list[dict]) -> list[dict]:
    result = []
    for source in rows:
        row = dict(source)
        if SLOPE_GUIDANCE in row["instruction"]:
            raise ValueError("slope guidance is already present")
        row["instruction"] = f"{row['instruction'].rstrip()}\n\n{SLOPE_GUIDANCE}"
        row["prompt_variant"] = "geometry_partial_edge_slope_v1"
        row["prompt_variant_human_truth"] = False
        result.append(row)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows, source_manifest = load_dev_rows(args.source)
    if not args.contract.is_file():
        raise FileNotFoundError(args.contract)
    variant = build_rows(rows)
    args.output.mkdir(parents=True)
    dev_path = args.output / "dev.jsonl"
    dev_path.write_text("".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in variant), encoding="utf-8")
    shutil.copyfile(args.contract, args.output / "contract.json")
    manifest = dict(source_manifest)
    manifest.update({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "prompt_variant": "geometry_partial_edge_slope_v1",
        "prompt_variant_human_truth": False,
        "source_manifest_sha256": file_sha256(args.source / "manifest.json"),
        "source_dev_jsonl_sha256": file_sha256(args.source / "dev.jsonl"),
        "slope_guidance_sha256": hashlib.sha256(
            SLOPE_GUIDANCE.encode("utf-8")).hexdigest(),
        "dev_jsonl_sha256": file_sha256(dev_path),
        "contract_file_sha256": file_sha256(args.output / "contract.json"),
        "source_image_bytes_copied": 0,
    })
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
