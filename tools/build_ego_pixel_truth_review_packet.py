#!/usr/bin/env python3
"""Build a local-only ego pixel-truth review packet from fixed camera frames.

The packet contains machine candidate masks and visual overlays only.  It never
creates human truth, never signs a review receipt, and never writes workbench
state.  Real images remain local and are not copied into tracked directories.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from source.agent.edit_agent import _rasterize, valid_shape
from source.agent.ego_shadow import materialize_camera_template
from source.agent.workflow import _sam_request, candidate_overlay


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _find_image(root: Path, camera: int, frame: int) -> Path:
    matches = [path for path in root.glob(f"camera{camera}-frame{frame}.*")
               if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one image for camera{camera} frame{frame}")
    return matches[0]


def _write_prompt_overlay(image: bytes, shape: dict, points: list[dict],
                          mask: np.ndarray, target: Path) -> None:
    """Write a local review overlay with prompt satisfaction made visible."""
    rendered = Image.open(io.BytesIO(candidate_overlay(image, [shape]))).convert("RGB")
    draw = ImageDraw.Draw(rendered)
    radius = max(4, min(rendered.size) // 90)
    for point in points:
        x, y, label = point["x"], point["y"], point["label"]
        inside = bool(mask[y, x])
        # green: covered positive; red: missed positive; cyan: excluded negative;
        # magenta: violated negative.  A black outer ring keeps points visible on
        # both bright factory floors and dark vehicle surfaces.
        color = ((58, 216, 132) if inside else (255, 82, 82)) if label == 1 else (
            (255, 70, 210) if inside else (72, 205, 255))
        box = (x - radius, y - radius, x + radius, y + radius)
        draw.ellipse(box, outline=(8, 12, 18), width=5)
        draw.ellipse(box, outline=color, width=3)
        if (label == 1 and not inside) or (label == 0 and inside):
            draw.line((x - radius, y - radius, x + radius, y + radius),
                      fill=color, width=3)
            draw.line((x - radius, y + radius, x + radius, y - radius),
                      fill=color, width=3)
    rendered.save(target, "PNG")


def _validate_scope(cameras, frames) -> tuple[list[int], list[int]]:
    if (not isinstance(cameras, (list, tuple)) or not cameras
            or any(type(value) is not int or value not in range(4) for value in cameras)
            or len(cameras) != len(set(cameras))):
        raise ValueError("cameras must be unique integers from 0 to 3")
    if (not isinstance(frames, (list, tuple)) or not frames
            or any(type(value) is not int or value < 0 for value in frames)
            or len(frames) != len(set(frames))):
        raise ValueError("frames must be unique non-negative integers")
    return list(cameras), list(frames)


def _write_contact_sheet(output: Path, rows: list[dict], cameras: list[int],
                         frames: list[int]) -> Path:
    cell_width, image_height, label_height = 320, 256, 28
    canvas = Image.new("RGB", (cell_width * len(cameras),
                               (image_height + label_height) * len(frames)),
                       (15, 23, 31))
    draw, font = ImageDraw.Draw(canvas), ImageFont.load_default()
    camera_columns = {camera: index for index, camera in enumerate(cameras)}
    frame_rows = {frame: index for index, frame in enumerate(frames)}
    for row in rows:
        if "overlay" not in row:
            continue
        camera = int(row["camera"].removeprefix("camera"))
        left = camera_columns[camera] * cell_width
        top = frame_rows[row["frame"]] * (image_height + label_height)
        with Image.open(output / row["overlay"]) as image:
            preview = image.convert("RGB")
            preview.thumbnail((cell_width, image_height))
        x = left + (cell_width - preview.width) // 2
        y = top + (image_height - preview.height) // 2
        canvas.paste(preview, (x, y))
        blocked = row["status"] != "human_review_required"
        color = (244, 130, 90) if blocked else (83, 210, 161)
        draw.rectangle((left, top + image_height,
                        left + cell_width, top + image_height + label_height),
                       fill=(25, 35, 45))
        draw.text((left + 6, top + image_height + 7),
                  f"{row['sample_id']} | {row['status']}", fill=color, font=font)
    path = output / "contact_sheet.jpg"
    canvas.save(path, "JPEG", quality=88)
    return path


def build_packet(image_root: Path, output: Path, request_points, *,
                 cameras=(0, 1, 2, 3), frames=(0, 38, 76, 114, 151)) -> dict:
    cameras, frames = _validate_scope(cameras, frames)
    if output.exists():
        raise FileExistsError(output)
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)
    mask_root, overlay_root = output / "candidate_masks", output / "overlays"
    mask_root.mkdir(parents=True)
    overlay_root.mkdir(parents=True)
    rows = []
    for camera in cameras:
        for frame in frames:
            source = _find_image(image_root, camera, frame)
            image = source.read_bytes()
            with Image.open(source) as decoded:
                width, height = decoded.size
            topic = f"/capture/camera_{camera}/image"
            template_name, points = materialize_camera_template(topic, width, height)
            sample_id = f"camera{camera}-frame{frame:03d}"
            row = {
                "sample_id": sample_id,
                "camera": f"camera{camera}",
                "frame": frame,
                "source_image": str(source.resolve()),
                "source_image_sha256": hashlib.sha256(image).hexdigest(),
                "template_name": template_name,
                "point_count": len(points),
                "candidate_only": True,
                "human_truth": False,
                "human_confirmed": False,
                "approved_for_training": False,
                "eligible_for_candidate_merge": False,
                "state_written": False,
            }
            try:
                response = request_points(image, points)
                if not isinstance(response, dict):
                    raise ValueError("point response must be an object")
                parts, score = response.get("parts"), response.get("score")
                warnings = response.get("warnings") or []
                if (not isinstance(parts, list) or not parts
                        or isinstance(score, bool) or not isinstance(score, (int, float))
                        or not np.isfinite(score) or not 0 <= score <= 1
                        or not isinstance(warnings, list)
                        or any(not isinstance(item, str) for item in warnings)):
                    raise ValueError("point response is incomplete")
                shape = {"id": 1, "label": "ego", "parts": parts, "score": score}
                if not valid_shape(shape, width, height):
                    raise ValueError("point response geometry is invalid")
                mask = _rasterize(shape, width, height).astype(bool)
                if not mask.any():
                    raise ValueError("point response mask is empty")
                positive_points = [point for point in points if point["label"] == 1]
                negative_points = [point for point in points if point["label"] == 0]
                positive_covered = sum(
                    bool(mask[point["y"], point["x"]]) for point in positive_points)
                negative_violated = sum(
                    bool(mask[point["y"], point["x"]]) for point in negative_points)
                blocked_prompt = (positive_covered != len(positive_points)
                                  or negative_violated > 0)
                mask_path = mask_root / f"{sample_id}.png"
                Image.fromarray(mask.astype(np.uint8) * 255, "L").save(mask_path)
                overlay_path = overlay_root / f"{sample_id}.png"
                _write_prompt_overlay(image, shape, points, mask, overlay_path)
                prompt_points = [{
                    "x": point["x"],
                    "y": point["y"],
                    "label": point["label"],
                    "satisfied": (bool(mask[point["y"], point["x"]])
                                  if point["label"] == 1 else
                                  not bool(mask[point["y"], point["x"]])),
                } for point in points]
                row.update({
                    "status": ("blocked_prompt_constraint" if blocked_prompt
                               else "blocked_sam_warning" if warnings
                               else "human_review_required"),
                    "candidate_observed": True,
                    "candidate_available": not blocked_prompt and not warnings,
                    "score": round(float(score), 6),
                    "area_ratio": round(float(mask.mean()), 6),
                    "sam_warning_count": len(warnings),
                    "positive_point_count": len(positive_points),
                    "positive_points_covered": positive_covered,
                    "negative_point_count": len(negative_points),
                    "negative_points_violated": negative_violated,
                    "prompt_points": prompt_points,
                    "sam_warning_sha256": (hashlib.sha256(json.dumps(
                        warnings, ensure_ascii=False, separators=(",", ":")).encode(
                            "utf-8")).hexdigest() if warnings else None),
                    "candidate_mask": str(mask_path.relative_to(output)).replace("\\", "/"),
                    "candidate_mask_sha256": _sha256(mask_path),
                    "overlay": str(overlay_path.relative_to(output)).replace("\\", "/"),
                    "overlay_sha256": _sha256(overlay_path),
                })
            except Exception as error:
                row.update({
                    "status": "failed",
                    "candidate_observed": False,
                    "candidate_available": False,
                    "error": str(error)[:500],
                })
            rows.append(row)
    contact_sheet = _write_contact_sheet(output, rows, cameras, frames)
    packet = {
        "schema_version": "labelagent.ego_pixel_truth_review_packet.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_only": True,
        "human_truth": False,
        "human_confirmed": False,
        "approved_for_training": False,
        "production_deployed": False,
        "formal_state_modified": False,
        "sample_count": len(rows),
        "candidate_observed_count": sum(row["candidate_observed"] for row in rows),
        "candidate_available_count": sum(row["candidate_available"] for row in rows),
        "blocked_warning_count": sum(
            int(row.get("sam_warning_count") or 0) > 0 for row in rows),
        "blocked_prompt_constraint_count": sum(
            row["status"] == "blocked_prompt_constraint" for row in rows),
        "failed_count": sum(row["status"] == "failed" for row in rows),
        "prompt_overlay_legend": {
            "positive_covered": "green_circle",
            "positive_missed": "red_circle_cross",
            "negative_excluded": "cyan_circle",
            "negative_violated": "magenta_circle_cross",
        },
        "contact_sheet": contact_sheet.name,
        "contact_sheet_sha256": _sha256(contact_sheet),
        "samples": rows,
        "next_step": (
            "A human must create ego/person/floor truth masks and a hash-bound review "
            "receipt before the pixel-truth evaluator can run."),
    }
    packet_path = output / "review_queue.json"
    packet_path.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    return packet


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--point-sam-url", default="http://127.0.0.1:28766")
    parser.add_argument("--frames", default="0,38,76,114,151")
    arguments = parser.parse_args()
    if not arguments.point_sam_url.startswith("http://127.0.0.1:"):
        raise ValueError("point SAM URL must be loopback")
    frames = tuple(int(value) for value in arguments.frames.split(","))
    config = {"sam3_url": arguments.point_sam_url}
    packet = build_packet(
        arguments.image_root, arguments.output,
        lambda image, points: _sam_request(
            config, "/segment_points", image, timeout_seconds=5.0, points=points),
        frames=frames)
    print(json.dumps({
        "output": str((arguments.output / "review_queue.json").resolve()),
        "sample_count": packet["sample_count"],
        "candidate_observed_count": packet["candidate_observed_count"],
        "candidate_available_count": packet["candidate_available_count"],
        "blocked_warning_count": packet["blocked_warning_count"],
        "blocked_prompt_constraint_count": packet[
            "blocked_prompt_constraint_count"],
        "failed_count": packet["failed_count"],
    }, ensure_ascii=False))
    return 0 if packet["failed_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
