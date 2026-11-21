"""Evaluate explicit positive/negative ego point sets without state writes.

This is a shadow-only diagnostic for hard, partly occluded camera-host robot
bodies.  Point sets are human-authored development hints.  The output is not
human truth, is never merged into workbench candidates, and cannot approve
training or production deployment.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from source.agent.workflow import candidate_overlay, sam_point_groups
from tools.diagnose_ego_bottom_point_fallback import (
    bounded_candidate_summary, mask_record)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_image_dimensions(width: int, height: int) -> None:
    if (type(width) is not int or type(height) is not int
            or width <= 0 or height <= 0):
        raise ValueError("image dimensions must be positive integers")


def validate_point_sets(payload, width: int, height: int) -> list[dict]:
    validate_image_dimensions(width, height)
    if not isinstance(payload, list) or not 1 <= len(payload) <= 8:
        raise ValueError("point-set file must contain 1 to 8 experiments")
    names = set()
    validated = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("each point set must be an object")
        name, points = item.get("name"), item.get("points")
        clean_name = name.strip() if isinstance(name, str) else ""
        if (not clean_name or clean_name in names or not isinstance(points, list)
                or not 1 <= len(points) <= 32):
            raise ValueError("point-set names must be unique and contain 1 to 32 points")
        normalized = []
        coordinates = set()
        for point in points:
            if not isinstance(point, dict):
                raise ValueError("every point must be an object")
            x, y, label = point.get("x"), point.get("y"), point.get("label")
            if (type(x) is not int or type(y) is not int or type(label) is not int
                    or label not in (0, 1)
                    or not 0 <= x < width or not 0 <= y < height):
                raise ValueError("point coordinates or labels are invalid")
            if (x, y) in coordinates:
                raise ValueError("point coordinates must be unique within a point set")
            coordinates.add((x, y))
            normalized.append({"x": x, "y": y, "label": label})
        if not any(point["label"] == 1 for point in normalized):
            raise ValueError("each point set requires at least one positive point")
        names.add(clean_name)
        validated.append({"name": clean_name, "points": normalized})
    return validated


def materialize_normalized_point_sets(payload, width: int, height: int) -> list[dict]:
    """Convert bounded image-relative hints into validated pixel coordinates."""
    validate_image_dimensions(width, height)
    if not isinstance(payload, list) or not 1 <= len(payload) <= 8:
        raise ValueError("normalized point-set file must contain 1 to 8 experiments")
    materialized = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("each normalized point set must be an object")
        points = item.get("points")
        if not isinstance(points, list):
            raise ValueError("normalized point sets require a points list")
        pixel_points = []
        for point in points:
            if not isinstance(point, dict):
                raise ValueError("every normalized point must be an object")
            x_ratio = point.get("x_ratio")
            y_ratio = point.get("y_ratio")
            label = point.get("label")
            if (isinstance(x_ratio, bool) or isinstance(y_ratio, bool)
                    or not isinstance(x_ratio, (int, float))
                    or not isinstance(y_ratio, (int, float))
                    or type(label) is not int or label not in (0, 1)
                    or not math.isfinite(x_ratio) or not math.isfinite(y_ratio)
                    or not 0 <= x_ratio <= 1 or not 0 <= y_ratio <= 1):
                raise ValueError("normalized point ratios or labels are invalid")
            pixel_points.append({
                "x": round(float(x_ratio) * (width - 1)),
                "y": round(float(y_ratio) * (height - 1)),
                "label": label,
            })
        materialized.append({"name": item.get("name"), "points": pixel_points})
    return validate_point_sets(materialized, width, height)


def prompt_constraint_summary(record: dict, points: list[dict]) -> dict:
    """Report whether a returned mask obeys every supplied point constraint.

    The diagnostic intentionally keeps the mask itself out of ``results.json``.
    ``mask_record`` already retains a bounded crop in memory, which is enough to
    check prompt coverage before that crop is discarded.
    """
    x0, y0, x1, y1 = record["bbox"]
    mask = record["mask"]

    def covered(point: dict) -> bool:
        x, y = point["x"], point["y"]
        return (x0 <= x < x1 and y0 <= y < y1
                and bool(mask[y - y0, x - x0]))

    positives = [point for point in points if point["label"] == 1]
    negatives = [point for point in points if point["label"] == 0]
    uncovered_positives = [point.copy() for point in positives if not covered(point)]
    violated_negatives = [point.copy() for point in negatives if covered(point)]
    positive_points_covered = len(positives) - len(uncovered_positives)
    negative_points_violated = len(violated_negatives)
    return {
        "positive_point_count": len(positives),
        "positive_points_covered": positive_points_covered,
        "uncovered_positive_points": uncovered_positives,
        "negative_point_count": len(negatives),
        "negative_points_violated": negative_points_violated,
        "violated_negative_points": violated_negatives,
        "passed": (positive_points_covered == len(positives)
                   and negative_points_violated == 0),
    }


def write_candidate_mask(record: dict, width: int, height: int,
                         target: Path) -> None:
    """Persist a full-frame review mask from the bounded in-memory crop."""
    validate_image_dimensions(width, height)
    x0, y0, x1, y1 = record["bbox"]
    mask = record["mask"]
    if (not 0 <= x0 < x1 <= width or not 0 <= y0 < y1 <= height
            or getattr(mask, "shape", None) != (y1 - y0, x1 - x0)):
        raise ValueError("candidate mask crop does not match its bounded bbox")
    canvas = Image.new("L", (width, height), 0)
    crop = Image.fromarray(mask.astype("uint8") * 255, mode="L")
    canvas.paste(crop, (x0, y0))
    buffer = io.BytesIO()
    canvas.save(buffer, "PNG", optimize=True)
    target.write_bytes(buffer.getvalue())


def request_points(endpoint: str, image: bytes, points: list[dict]) -> dict:
    body = {"image": base64.b64encode(image).decode("ascii"), "points": points}
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/segment_points",
        data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=1200) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read(4096).decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"point SAM returned HTTP {error.code}: {detail or error.reason}") from error
    if not isinstance(payload, dict):
        raise ValueError("point SAM response must be an object")
    parts = payload.get("parts", [])
    if not isinstance(parts, list) or len(parts) > 64:
        raise ValueError("point SAM response parts must be a bounded list")
    for part in parts:
        if (not isinstance(part, dict)
                or not isinstance(part.get("points"), list)
                or not isinstance(part.get("holes", []), list)):
            raise ValueError("point SAM response contains invalid mask geometry")
    score = payload.get("score")
    if (score is not None and (isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score))):
        raise ValueError("point SAM response score must be finite")
    return payload


def request_point_groups(endpoint: str, image: bytes,
                         point_sets: list[dict]) -> dict:
    """Evaluate every point set with one encoding when the service supports it.

    ``sam_point_groups`` validates the response contract and only falls back to
    the legacy per-group route when the grouped route is genuinely absent
    (HTTP 404).  Busy, timeout and malformed responses remain visible.
    """
    return sam_point_groups(
        image,
        [experiment["points"] for experiment in point_sets],
        {"sam3_point_url": endpoint, "sam3_url": endpoint},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    point_source = parser.add_mutually_exclusive_group(required=True)
    point_source.add_argument("--point-sets", type=Path)
    point_source.add_argument("--normalized-point-sets", type=Path)
    parser.add_argument("--point-sam-url", default="http://127.0.0.1:28766")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not args.point_sam_url.startswith("http://127.0.0.1:"):
        raise ValueError("point SAM URL must be a loopback endpoint")
    point_sets_path = args.point_sets or args.normalized_point_sets
    if not args.image.is_file() or not point_sets_path.is_file():
        raise FileNotFoundError("image and point-set file are required")

    image = args.image.read_bytes()
    with Image.open(args.image) as decoded:
        width, height = decoded.size
    point_payload = json.loads(point_sets_path.read_text(encoding="utf-8"))
    if args.normalized_point_sets:
        point_sets = materialize_normalized_point_sets(point_payload, width, height)
        point_coordinate_mode = "normalized_image_ratio"
    else:
        point_sets = validate_point_sets(point_payload, width, height)
        point_coordinate_mode = "absolute_pixel"
    args.output_dir.mkdir(parents=True)
    batch_started = time.monotonic()
    try:
        grouped = request_point_groups(args.point_sam_url, image, point_sets)
        grouped_items = grouped["items"]
        request_diagnostics = grouped["diagnostics"]
        grouped_failure = None
    except Exception as error:
        grouped_items = [None] * len(point_sets)
        request_diagnostics = None
        grouped_failure = str(error)[:500]
    batch_elapsed = round(time.monotonic() - batch_started, 4)

    rows = []
    for index, (experiment, item) in enumerate(zip(point_sets, grouped_items)):
        failure = None
        shape = None
        geometry = None
        candidate_mask = None
        try:
            if grouped_failure is not None:
                raise RuntimeError(grouped_failure)
            if not isinstance(item, dict):
                raise ValueError("point SAM group item is invalid")
            if "error" in item:
                raise RuntimeError(str(item["error"])[:500])
            result = item.get("result")
            if not isinstance(result, dict):
                raise ValueError("point SAM group result is invalid")
            parts = result.get("parts") or []
            if parts:
                shape = {
                    "id": index + 1, "label": "ego",
                    "points": parts[0].get("points", []),
                    "holes": parts[0].get("holes", []),
                    "parts": parts, "score": float(result.get("score") or 0),
                    "source": "shadow_guided_points",
                }
                shape["_record"] = mask_record(shape, width, height)
                shape["point"] = experiment["points"]
                geometry = bounded_candidate_summary(shape, width, height)
                geometry["prompt_constraints"] = prompt_constraint_summary(
                    shape["_record"], experiment["points"])
                candidate_mask = (args.output_dir
                                  / f"{index:02d}-{experiment['name']}-mask.png")
                write_candidate_mask(shape["_record"], width, height,
                                     candidate_mask)
                shape.pop("_record", None)
                shape.pop("point", None)
        except Exception as error:  # Shadow diagnostics must preserve the other rows.
            failure = str(error)[:500]
            shape = None
            geometry = None
        overlay = args.output_dir / f"{index:02d}-{experiment['name']}.jpg"
        overlay.write_bytes(candidate_overlay(image, [shape] if shape else []))
        prompt_constraints_passed = bool(
            geometry
            and geometry.get("prompt_constraints", {}).get("passed") is True)
        rows.append({
            "name": experiment["name"],
            "points": experiment["points"],
            "elapsed_seconds": batch_elapsed,
            "elapsed_seconds_shared_batch": True,
            "candidate_available": shape is not None,
            "prompt_constraints_passed": prompt_constraints_passed,
            "error": failure,
            "geometry": geometry,
            "candidate_mask": candidate_mask.name if candidate_mask else None,
            "candidate_mask_sha256": (
                file_sha256(candidate_mask) if candidate_mask else None),
            "overlay": overlay.name,
            "overlay_sha256": file_sha256(overlay),
        })

    report = {
        "schema_version": "labelagent.ego_guided_point_shadow.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_only": True,
        "human_truth": False,
        "human_confirmed": False,
        "approved_for_training": False,
        "production_deployed": False,
        "source_state_modified": False,
        "image": str(args.image.resolve()),
        "image_sha256": hashlib.sha256(image).hexdigest(),
        "point_sets_sha256": file_sha256(point_sets_path),
        "point_coordinate_mode": point_coordinate_mode,
        "request_diagnostics": request_diagnostics,
        "client_batch_elapsed_seconds": batch_elapsed,
        "rows": rows,
        "summary": {
            "experiment_count": len(rows),
            "candidate_count": sum(row["candidate_available"] for row in rows),
            "prompt_constraint_pass_count": sum(
                row["prompt_constraints_passed"] for row in rows),
            "failed_closed_count": sum(
                not row["candidate_available"] for row in rows),
        },
        "evidence_note": (
            "Point sets are bounded assistant-authored development hints. "
            "Every mask requires independent human review and negative controls."),
    }
    output = args.output_dir / "results.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "output": str(output.resolve()),
        "candidate_count": sum(row["candidate_available"] for row in rows),
        "prompt_constraint_pass_count": sum(
            row["prompt_constraints_passed"] for row in rows),
        "experiment_count": len(rows), "sha256": file_sha256(output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
