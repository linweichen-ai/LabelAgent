#!/usr/bin/env python3
"""Build a read-only workbench pack from pallet ownership evidence sheets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

from PIL import Image


SOURCE_SCHEMA = "labelagent.pallet-roi-shadow-review.v1"
OUTPUT_SCHEMA = "labelagent.component-review-pack.v2"
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_source(path: Path) -> tuple[dict, str]:
    path = path.resolve(strict=True)
    raw = path.read_bytes()
    if len(raw) > 16 * 1024 * 1024:
        raise ValueError("source review manifest is too large")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("source review manifest is not valid UTF-8 JSON") from error
    if not isinstance(value, dict) or value.get("schema_version") != SOURCE_SCHEMA:
        raise ValueError("unsupported source review schema")
    if (value.get("candidate_only") is not True
            or value.get("human_truth") is not False
            or value.get("training_authorized") is not False
            or value.get("deployment_authorized") is not False
            or value.get("workbench_write") is not False
            or value.get("component_context_ownership_included") is not True):
        raise ValueError("source review safety gates are inconsistent")
    rows = value.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("source review has no candidate rows")
    return value, hashlib.sha256(raw).hexdigest()


def _verified_sheet(manifest_path: Path, component: dict) -> tuple[Path, bytes]:
    name = component.get("context_detail_sheet")
    expected_sha = component.get("context_detail_sheet_sha256")
    if (component.get("context_ownership") is not True
            or component.get("context_locator") is not True
            or not isinstance(name, str) or Path(name).name != name
            or Path(name).suffix.lower() not in {".png", ".jpg", ".jpeg"}
            or not isinstance(expected_sha, str)
            or SHA256_RE.fullmatch(expected_sha) is None):
        raise ValueError("ownership evidence binding is invalid")
    path = (manifest_path.parent / name).resolve(strict=True)
    if path.parent != manifest_path.parent:
        raise ValueError("ownership evidence path escapes source folder")
    payload = path.read_bytes()
    if len(payload) > 4 * 1024 * 1024:
        raise ValueError("ownership evidence image is too large")
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        raise ValueError("ownership evidence image hash mismatch")
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        if (image.format not in {"PNG", "JPEG"} or image.width <= 0
                or image.height <= 0 or image.width > 4096
                or image.height > 4096
                or image.width * image.height > 16_000_000):
            raise ValueError("ownership evidence image format or size is invalid")
    return path, payload


def build_pack(review_manifests: list[Path], output: Path) -> dict:
    if not review_manifests:
        raise ValueError("at least one source review manifest is required")
    if output.exists():
        raise FileExistsError(output)
    sources = []
    pending = []
    seen_ids = set()
    for source_index, manifest_path in enumerate(review_manifests, 1):
        manifest_path = manifest_path.resolve(strict=True)
        value, manifest_sha = _read_source(manifest_path)
        sources.append({
            "source_index": source_index,
            "manifest_sha256": manifest_sha,
            "candidate_row_count": len(value["rows"]),
        })
        for row in sorted(value["rows"], key=lambda item: (
                int(item.get("frame", -1)), int(item.get("candidate_id", -1)))):
            frame = row.get("frame")
            candidate_id = row.get("candidate_id")
            components = row.get("component_sheets")
            if (isinstance(frame, bool) or not isinstance(frame, int) or frame < 0
                    or isinstance(candidate_id, bool)
                    or not isinstance(candidate_id, int) or candidate_id < 0
                    or not isinstance(components, list) or not components):
                raise ValueError("source candidate component metadata is invalid")
            for component in sorted(components, key=lambda item: int(
                    item.get("component_index", -1))):
                component_index = component.get("component_index")
                area = component.get("area")
                if (isinstance(component_index, bool)
                        or not isinstance(component_index, int)
                        or component_index <= 0 or isinstance(area, bool)
                        or not isinstance(area, int) or area <= 0):
                    raise ValueError("source component identity is invalid")
                component_id = (
                    f"source-{source_index:02d}-frame-{frame:06d}-"
                    f"candidate-{candidate_id}-component-{component_index:02d}")
                if component_id in seen_ids:
                    raise ValueError("duplicate component identity")
                seen_ids.add(component_id)
                _, payload = _verified_sheet(manifest_path, component)
                pending.append({
                    "component_id": component_id,
                    "area": area,
                    "payload": payload,
                    "suffix": Path(component["context_detail_sheet"]).suffix.lower(),
                })
    if not pending or len(pending) > 500:
        raise ValueError("ownership component count is outside workbench limits")

    staging = output.with_name(f".{output.name}.staging-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        rows = []
        for rank, item in enumerate(pending, 1):
            suffix = ".jpg" if item["suffix"] in {".jpg", ".jpeg"} else ".png"
            image_name = f"ownership-evidence-{rank:04d}{suffix}"
            target = staging / image_name
            target.write_bytes(item["payload"])
            rows.append({
                "component_id": item["component_id"],
                "review_rank": rank,
                "policy_queue": "priority_review",
                "policy_reason": (
                    f"候选补点 {item['area']} px · 所有权上下文人工复核"),
                "focused_image": image_name,
                "focused_image_sha256": hashlib.sha256(
                    item["payload"]).hexdigest(),
                "candidate_only": True,
                "human_truth": False,
                "approved_for_training": False,
                "training_return_allowed": False,
                "automatic_decision": None,
                "human_decision": None,
                "reviewer_declared_id": None,
            })
        packet = {
            "schema_version": OUTPUT_SCHEMA,
            "display_name": "Pallet 边界所有权复核",
            "review_kind": "pallet_boundary_ownership",
            "evidence_mode": "ownership_overlay",
            "candidate_only": True,
            "assistant_review_only": True,
            "human_truth": False,
            "approved_for_training": False,
            "production_deployed": False,
            "default_enabled": False,
            "automatic_decisions_enabled": False,
            "human_decision": None,
            "reviewer_declared_id": None,
            "training_return_allowed": False,
            "review_count": len(rows),
            "priority_review": len(rows),
            "deferred_manual_review": 0,
            "source_bindings": sources,
            "rows": rows,
        }
        manifest_path = staging / "focused_review_manifest.json"
        manifest_path.write_text(
            json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        manifest_sha = file_sha256(manifest_path)
        staging.replace(output)
        return {
            "schema_version": OUTPUT_SCHEMA,
            "review_count": len(rows),
            "priority_review": len(rows),
            "manifest_sha256": manifest_sha,
            "output": str(output),
        }
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-manifest", action="append", required=True,
                        type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build_pack(args.review_manifest, args.output),
                     ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
