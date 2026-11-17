"""Send label-free LiDAR geometry hints through the formal SAM point endpoint."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

import cv2
import numpy as np


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def request_json(url: str, payload: dict | None = None, timeout: float = 60) -> dict:
    body = None if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read())
    if not isinstance(result, dict):
        raise ValueError("Service response is not an object")
    return result


def prompt_points(hints: list[dict]) -> list[dict]:
    """Flatten component points while preserving deterministic order and uniqueness."""
    points = []
    seen = set()
    for hint in hints:
        for point in hint.get("points", []):
            value = (point.get("x"), point.get("y"), point.get("label"))
            if (
                type(value[0]) is not int
                or type(value[1]) is not int
                or value[2] != 1
                or not 0 <= value[0] < 640
                or not 0 <= value[1] < 512
            ):
                raise ValueError("Invalid geometry prompt point")
            if value not in seen:
                points.append({"x": value[0], "y": value[1], "label": 1})
                seen.add(value)
    if not points or len(points) > 32:
        raise ValueError("Geometry prompt must contain 1-32 unique positive points")
    return points


def ring_constrained_points(hints: list[dict], width: int = 640, height: int = 512, margin: int = 40) -> list[dict]:
    """Use one positive per component and geometry-only negatives outside their union."""
    if not hints or margin < 8 or margin > 128:
        raise ValueError("Invalid ring-constrained geometry hints")
    positives = []
    boxes = []
    for hint in hints:
        x, y = hint.get("x"), hint.get("y")
        box = hint.get("pixel_bbox")
        if type(x) is not int or type(y) is not int or not 0 <= x < width or not 0 <= y < height:
            raise ValueError("Invalid component center")
        if not isinstance(box, list) or len(box) != 4 or any(type(value) is not int for value in box):
            raise ValueError("Invalid component pixel bounds")
        positives.append({"x": x, "y": y, "label": 1})
        boxes.append(box)
    x0, y0 = min(box[0] for box in boxes), min(box[1] for box in boxes)
    x1, y1 = max(box[2] for box in boxes), max(box[3] for box in boxes)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    candidates = [(x0 - margin, cy), (x1 + margin, cy), (cx, y0 - margin), (cx, y1 + margin)]
    negatives = [
        {"x": x, "y": y, "label": 0}
        for x, y in candidates
        if 0 <= x < width and 0 <= y < height and all(not (box[0] <= x <= box[2] and box[1] <= y <= box[3]) for box in boxes)
    ]
    if len(negatives) < 2:
        raise ValueError("Geometry bounds do not provide two safe in-image negative ring points")
    return positives + negatives


def geometry_mask_metrics(mask: np.ndarray, hints: list[dict], padding: int = 40) -> dict:
    """Score a SAM mask using geometry only; assistant points are unavailable here."""
    points = prompt_points(hints)
    covered = sum(bool(mask[point["y"], point["x"]]) for point in points)
    boxes = [hint["pixel_bbox"] for hint in hints]
    x0 = max(0, min(box[0] for box in boxes) - padding)
    y0 = max(0, min(box[1] for box in boxes) - padding)
    x1 = min(mask.shape[1] - 1, max(box[2] for box in boxes) + padding)
    y1 = min(mask.shape[0] - 1, max(box[3] for box in boxes) + padding)
    inside = int(mask[y0:y1 + 1, x0:x1 + 1].sum())
    total = int(mask.sum())
    coverage = covered / len(points)
    spill = 1 - inside / total
    fraction = total / mask.size
    return {
        "geometry_point_coverage": coverage,
        "geometry_points_covered": covered,
        "geometry_points_total": len(points),
        "padded_geometry_bbox": [x0, y0, x1, y1],
        "spill_fraction_outside_padded_geometry_bbox": spill,
        "mask_fraction": fraction,
        "geometry_gate_passed": coverage >= 0.2 and spill <= 0.35 and fraction <= 0.12,
    }


def geometry_attempt_rank(attempt: dict):
    metrics = attempt["metrics"]
    return (
        metrics["geometry_gate_passed"],
        metrics["geometry_point_coverage"],
        -metrics["spill_fraction_outside_padded_geometry_bbox"],
        -metrics["mask_fraction"],
        not bool(attempt["warnings"]),
    )


def polygon_mask(result: dict, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    polygons: list[list] = []
    holes: list[list] = []
    if result.get("points"):
        polygons.append(result["points"])
    holes.extend(result.get("holes") or [])
    for part in result.get("parts") or []:
        if isinstance(part, dict):
            if part.get("points"):
                polygons.append(part["points"])
            holes.extend(part.get("holes") or [])
        else:
            polygons.append(part)
    for polygon in polygons:
        array = np.asarray(polygon, dtype=np.int32)
        if array.ndim != 2 or array.shape[1] != 2 or len(array) < 3:
            raise ValueError("SAM returned an invalid polygon")
        cv2.fillPoly(mask, [array], 1)
    for hole in holes:
        array = np.asarray(hole, dtype=np.int32)
        if array.ndim != 2 or array.shape[1] != 2 or len(array) < 3:
            raise ValueError("SAM returned an invalid hole")
        cv2.fillPoly(mask, [array], 0)
    if not mask.any():
        raise ValueError("SAM returned an empty candidate")
    return mask


def render(image_path: Path, output: Path, mask: np.ndarray, points: list[dict], review: list[dict]) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != mask.shape:
        raise ValueError("Image and SAM mask dimensions differ")
    color = np.zeros_like(image)
    color[mask != 0] = (0, 180, 0)
    image = cv2.addWeighted(image, 1.0, color, 0.42, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 255), 2)
    for point in points:
        cv2.circle(image, (point["x"], point["y"]), 4, (0, 0, 255), -1)
    for point in review:
        color_value = (0, 255, 0) if point["label"] == 1 else (255, 0, 0)
        cv2.drawMarker(image, (point["x"], point["y"]), color_value, cv2.MARKER_CROSS, 18, 2)
    if not cv2.imwrite(str(output), image):
        raise ValueError("Cannot save SAM overlay")


def evaluate(args) -> dict:
    geometry_path = Path(args.geometry_report).resolve()
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    summary = geometry.get("summary") or {}
    if (
        geometry.get("status") != "candidate_only_not_deployed"
        or geometry.get("human_truth") is not False
        or geometry.get("labels_read") is not False
        or geometry.get("prediction_reads_assistant_points") is not False
        or summary.get("assistant_candidate_positives") != 4
        or summary.get("assistant_candidate_negatives") != 4
        or summary.get("negative_views_with_any_geometry_hint") != 0
    ):
        raise ValueError("Geometry report has not passed the bounded candidate gate")
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("Output directory already exists")
    output.mkdir(parents=True)
    health_before = request_json(args.sam_url.rstrip("/") + "/health")
    if health_before.get("status") != "ok" or health_before.get("model_loaded") is not True:
        raise ValueError("Formal SAM service is not loaded")

    rows = []
    for item in geometry["images"]:
        if not item["assistant_visible_slope_candidate"]:
            if item["hints"]:
                raise ValueError("Negative diagnostic view unexpectedly has geometry hints")
            continue
        image_path = Path(item["image_path"])
        image = image_path.read_bytes()
        decoded = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("Cannot decode source image")
        if args.point_policy == "geometry_cascade":
            policies = ["component_center_ring_negatives", "all_positive"]
        else:
            policies = [args.point_policy]
        attempts = []
        failed_attempts = []
        for policy in policies:
            points = (
                prompt_points(item["hints"])
                if policy == "all_positive"
                else ring_constrained_points(
                    item["hints"], decoded.shape[1], decoded.shape[0], margin=args.negative_margin
                )
            )
            payload = {"image": base64.b64encode(image).decode("ascii"), "points": points}
            started = time.monotonic()
            try:
                result = request_json(args.sam_url.rstrip("/") + "/segment_points", payload, timeout=args.timeout)
                elapsed = time.monotonic() - started
                mask = polygon_mask(result, decoded.shape[1], decoded.shape[0])
                metrics = geometry_mask_metrics(mask, item["hints"])
                attempts.append({
                    "policy": policy,
                    "points": points,
                    "result": result,
                    "mask": mask,
                    "warnings": result.get("warnings") or [],
                    "elapsed_seconds": elapsed,
                    "metrics": metrics,
                })
            except urllib.error.HTTPError as error:
                failed_attempts.append({
                    "policy": policy,
                    "points": points,
                    "http_status": error.code,
                    "detail": error.read().decode("utf-8", "replace")[:1000],
                    "elapsed_seconds": round(time.monotonic() - started, 4),
                })
        if not attempts:
            raise ValueError(f"All SAM geometry policies failed for {item['sample_id']}: {failed_attempts}")
        selected = max(attempts, key=geometry_attempt_rank)
        points, result, mask = selected["points"], selected["result"], selected["mask"]
        elapsed = sum(row["elapsed_seconds"] for row in attempts) + sum(row["elapsed_seconds"] for row in failed_attempts)
        review = item.get("diagnostic_review_points") or []
        # Geometry reports prior to this evaluator do not copy review points;
        # load them separately while preserving the post-generation boundary.
        review_by_id = args.review_by_id
        review = review_by_id.get(item["sample_id"], review)
        positive = [point for point in review if point["label"] == 1]
        negative = [point for point in review if point["label"] == 0]
        positive_covered = all(bool(mask[point["y"], point["x"]]) for point in positive)
        negative_clear = all(not bool(mask[point["y"], point["x"]]) for point in negative)
        overlay = output / f"{item['sample_id']}-geometry-sam.jpg"
        render(image_path, overlay, mask, points, review)
        row = {
            "sample_id": item["sample_id"],
            "image_sha256": digest(image_path),
            "geometry_points": points,
            "geometry_component_hints": len(item["hints"]),
            "selected_policy": selected["policy"],
            "geometry_selection_metrics": selected["metrics"],
            "attempts": [
                {
                    "policy": attempt["policy"],
                    "points": attempt["points"],
                    "warnings": attempt["warnings"],
                    "elapsed_seconds": round(attempt["elapsed_seconds"], 4),
                    "metrics": attempt["metrics"],
                }
                for attempt in attempts
            ] + failed_attempts,
            "sam_score": result.get("score"),
            "sam_warnings": result.get("warnings") or [],
            "mask_pixels": int(mask.sum()),
            "mask_fraction": float(mask.mean()),
            "bbox": [
                int(np.flatnonzero(mask.any(axis=0))[0]),
                int(np.flatnonzero(mask.any(axis=1))[0]),
                int(np.flatnonzero(mask.any(axis=0))[-1]),
                int(np.flatnonzero(mask.any(axis=1))[-1]),
            ],
            "assistant_positive_point_covered": positive_covered,
            "assistant_negative_points_clear": negative_clear,
            "elapsed_seconds": round(elapsed, 4),
            "overlay": overlay.name,
        }
        rows.append(row)
        (output / f"{item['sample_id']}.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(row, ensure_ascii=False, separators=(",", ":")), flush=True)

    health_after = request_json(args.sam_url.rstrip("/") + "/health")
    report = {
        "schema_version": "labelagent.slope-lidar-sam.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "candidate_only_visual_review_required",
        "geometry_report_sha256": digest(geometry_path),
        "assistant_review_sha256": digest(Path(args.assistant_points)),
        "human_truth": False,
        "human_confirmed": False,
        "candidate_only": True,
        "production_deployed": False,
        "training_authorized": False,
        "color_rule_used": False,
        "negative_views_sent_to_sam": 0,
        "point_policy": args.point_policy,
        "negative_ring_margin_pixels": args.negative_margin if args.point_policy != "all_positive" else None,
        "health_before": health_before,
        "health_after": health_after,
        "summary": {
            "images": len(rows),
            "sam_candidates_returned": len(rows),
            "assistant_positive_points_covered": sum(row["assistant_positive_point_covered"] for row in rows),
            "assistant_negative_sets_clear": sum(row["assistant_negative_points_clear"] for row in rows),
            "warnings": sum(bool(row["sam_warnings"]) for row in rows),
            "elapsed_seconds": round(sum(row["elapsed_seconds"] for row in rows), 4),
        },
        "rows": rows,
        "evidence_note": "Assistant points are diagnostic candidates only. Mask quality still requires visual review and independent human truth before production.",
    }
    (output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, separators=(",", ":")), flush=True)
    return report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-report", required=True)
    parser.add_argument("--assistant-points", required=True)
    parser.add_argument("--sam-url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument(
        "--point-policy",
        choices=("all_positive", "component_center_ring_negatives", "geometry_cascade"),
        default="geometry_cascade",
    )
    parser.add_argument("--negative-margin", type=int, default=40)
    args = parser.parse_args()
    if not args.sam_url.startswith("http://127.0.0.1:"):
        parser.error("SAM URL must be loopback")
    if not 8 <= args.negative_margin <= 128:
        parser.error("--negative-margin must be between 8 and 128")
    review = json.loads(Path(args.assistant_points).read_text(encoding="utf-8-sig"))
    if review.get("human_truth") is not False or review.get("human_confirmed") is not False:
        parser.error("Assistant points must remain explicitly non-human")
    args.review_by_id = {row["sample_id"]: row["points"] for row in review["items"]}
    return args


if __name__ == "__main__":
    evaluate(parse_args())
