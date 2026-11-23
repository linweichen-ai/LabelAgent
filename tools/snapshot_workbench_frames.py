#!/usr/bin/env python3
"""Freeze source images from one idle workbench scope using GET requests only."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.build_pallet_boundary_comparison_pack import (
    _get_json,
    _read_response,
    _safe_base_url,
    _validate_bootstrap,
)


SCHEMA_VERSION = "labelagent.workbench-frame-snapshot.v1"


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def parse_frames(value: str) -> list[int]:
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("frames must be comma-separated integers") from error


def validate_idle_status(value: dict, scope: dict) -> None:
    if (value.get("status") != "idle"
            or value.get("selected_file") != scope["path"]
            or value.get("selected_topic") != scope["topic"]
            or value.get("selected_file_activation_generation") !=
            scope["file_activation_generation"]):
        raise ValueError("workbench is not idle in the requested data scope")


def image_extension(payload: bytes, frame: int) -> str:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
            image_format = image.format
    except OSError as error:
        raise ValueError(f"frame {frame} is not a decodable image") from error
    extensions = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
    if image_format not in extensions:
        raise ValueError(f"frame {frame} uses unsupported image format {image_format!r}")
    return extensions[image_format]


def snapshot(base_url, frames, output, *, opener=urlopen):
    base_url = _safe_base_url(base_url)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    normalized = []
    for frame in frames:
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            raise ValueError("frames must contain non-negative integers")
        if frame in normalized:
            raise ValueError("frames must not contain duplicates")
        normalized.append(frame)
    if not normalized:
        raise ValueError("frames must not be empty")

    bootstrap = _get_json(opener, base_url, "/api/bootstrap")
    source_path, topic, generation, frame_count = _validate_bootstrap(bootstrap)
    if any(frame >= frame_count for frame in normalized):
        raise ValueError(f"frame exceeds current topic range of {frame_count}")
    scope = {"path": source_path, "topic": topic,
             "file_activation_generation": generation}
    validate_idle_status(
        _get_json(opener, base_url, "/api/run/status", scope), scope)

    output.parent.mkdir(parents=True, exist_ok=True)
    wrapper = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=output.parent))
    try:
        staging = wrapper / output.name
        staging.mkdir()
        records = []
        query = urlencode(scope)
        for frame in normalized:
            payload = _read_response(
                opener, f"{base_url}/api/frame/{frame}?{query}")
            extension = image_extension(payload, frame)
            name = f"frame-{frame:06d}{extension}"
            (staging / name).write_bytes(payload)
            records.append({"frame": frame, "image": name,
                            "sha256": sha256(payload), "bytes": len(payload)})

        bootstrap_after = _get_json(opener, base_url, "/api/bootstrap")
        if _validate_bootstrap(bootstrap_after)[:3] != (
                source_path, topic, generation):
            raise ValueError("workbench data scope changed during snapshot")
        validate_idle_status(
            _get_json(opener, base_url, "/api/run/status", scope), scope)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "source": scope,
            "frame_count": frame_count,
            "requested_frames": normalized,
            "images": records,
            "server_methods": ["GET"],
            "truth_read": False,
            "workbench_write": False,
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2,
                       sort_keys=True) + "\n", encoding="utf-8")
        for record in records:
            frozen_payload = (staging / record["image"]).read_bytes()
            if (len(frozen_payload) != record["bytes"]
                    or sha256(frozen_payload) != record["sha256"]):
                raise ValueError("staged snapshot image readback mismatch")
        manifest_sha256 = sha256(manifest_path.read_bytes())
        staging.replace(output)
        return {
            "schema_version": SCHEMA_VERSION,
            "output": str(output),
            "image_count": len(records),
            "manifest_sha256": manifest_sha256,
            "server_methods": ["GET"],
            "workbench_write": False,
        }
    finally:
        shutil.rmtree(wrapper, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--frames", required=True, type=parse_frames)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(snapshot(args.base_url, args.frames, args.output),
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
