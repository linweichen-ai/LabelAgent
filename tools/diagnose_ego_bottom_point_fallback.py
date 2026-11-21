"""Evaluate a guarded bottom-point SAM fallback for partially visible ego bodies.

Inputs are explicitly split into assistant-reviewed positive and negative
development images.  The tool is diagnostic only: it never changes workbench
state, truth, training approval, or the production prompt.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import urllib.request

from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from source.agent.edit_agent import _rasterize
from source.agent.workflow import (candidate_overlay, ego_bottom_geometry_profile,
                                   ego_bottom_points)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def point_batch(image: bytes, points: list[dict], endpoint: str) -> dict:
    body = {"image": base64.b64encode(image).decode("ascii"), "points": points}
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/segment_points_batch",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=1200) as response:
        payload = json.load(response)
    diagnostics = payload.get("diagnostics", {}) if isinstance(payload, dict) else {}
    if (not isinstance(payload, dict) or not isinstance(payload.get("items"), list)
            or diagnostics.get("image_encoded_once") is not True
            or diagnostics.get("point_count") != len(points)):
        raise ValueError("point batch did not prove one-pass complete coverage")
    return payload


def mask_record(shape: dict, width: int, height: int) -> dict:
    mask = _rasterize(shape, width, height)
    rows, columns = mask.nonzero()
    if not len(rows):
        raise ValueError("empty SAM candidate")
    x0, x1 = int(columns.min()), int(columns.max()) + 1
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    return {"bbox": [x0, y0, x1, y1], "area": int(len(rows)),
            "mask": mask[y0:y1, x0:x1].copy()}


def mask_iou(first: dict, second: dict) -> float:
    ax0, ay0, ax1, ay1 = first["bbox"]
    bx0, by0, bx1, by1 = second["bbox"]
    x0, y0, x1, y1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    a = first["mask"][y0 - ay0:y1 - ay0, x0 - ax0:x1 - ax0]
    b = second["mask"][y0 - by0:y1 - by0, x0 - bx0:x1 - bx0]
    intersection = int((a & b).sum())
    return intersection / max(first["area"] + second["area"] - intersection, 1)


def bounded_candidate_summary(shape: dict, width: int, height: int) -> dict:
    """Keep enough geometry to explain a rejected shadow candidate."""
    record = shape["_record"]
    x0, y0, x1, y1 = record["bbox"]
    return {
        "point": shape["point"],
        "score": shape["score"],
        "guard_profile": shape.get("guard_profile"),
        "bbox": record["bbox"],
        "area": record["area"],
        "area_ratio": round(record["area"] / (width * height), 6),
        "width_ratio": round((x1 - x0) / width, 6),
        "height_ratio": round((y1 - y0) / height, 6),
        "bottom_gap": height - y1,
        "center_ratio": round(((x0 + x1) / 2) / width, 6),
    }


def evaluate(image_path: Path, expected: bool, endpoint: str,
             output_dir: Path, index: int) -> dict:
    image = image_path.read_bytes()
    with Image.open(io.BytesIO(image)) as decoded:
        width, height = decoded.size
    points = ego_bottom_points(width, height)
    started = time.monotonic()
    batch = point_batch(image, points, endpoint)
    raw, guarded = [], []
    for item in batch["items"]:
        result = item.get("result")
        if not isinstance(result, dict) or not result.get("parts"):
            continue
        parts = result["parts"]
        shape = {
            "id": len(raw) + 1, "label": "ego",
            "points": parts[0].get("points", []),
            "holes": parts[0].get("holes", []),
            "parts": parts, "score": float(result.get("score") or 0),
            "point": item["point"],
        }
        record = mask_record(shape, width, height)
        shape["_record"] = record
        raw.append(shape)
        profile = ego_bottom_geometry_profile(
            record["bbox"], record["area"], shape["score"], width, height)
        if profile:
            shape["guard_profile"] = profile
            guarded.append(shape)
    selected = []
    for shape in sorted(guarded, key=lambda item: item["score"], reverse=True):
        if any(mask_iou(shape["_record"], prior["_record"]) >= 0.85
               for prior in selected):
            continue
        selected.append(shape)
    selected = selected[:1]
    overlay = output_dir / f"{index:02d}-{image_path.stem}-ego-bottom-point.jpg"
    overlay.write_bytes(candidate_overlay(
        image, [{key: value for key, value in shape.items() if key != "_record"}
                for shape in selected]))

    return {
        "image": str(image_path.resolve()),
        "image_sha256": hashlib.sha256(image).hexdigest(),
        "expected_assistant_visible_ego": expected,
        "width": width, "height": height,
        "point_count": len(points),
        "elapsed_seconds": round(time.monotonic() - started, 4),
        "service_diagnostics": batch["diagnostics"],
        "raw_candidate_count": len(raw),
        "guarded_candidate_count": len(guarded),
        "raw_candidates": [bounded_candidate_summary(shape, width, height)
                           for shape in sorted(
                               raw, key=lambda item: item["score"], reverse=True)],
        "selected": [bounded_candidate_summary(shape, width, height)
                     for shape in selected],
        "predicted_visible_ego": bool(selected),
        "matches_assistant_review": bool(selected) is expected,
        "overlay": str(overlay.resolve()),
        "overlay_sha256": file_sha256(overlay),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive", action="append", default=[], type=Path)
    parser.add_argument("--negative", action="append", default=[], type=Path)
    parser.add_argument("--point-sam-url", default="http://127.0.0.1:28766")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not args.point_sam_url.startswith("http://127.0.0.1:"):
        raise ValueError("point SAM URL must be a loopback endpoint")
    samples = [(path, True) for path in args.positive]
    samples += [(path, False) for path in args.negative]
    if not samples or any(not path.is_file() for path, _ in samples):
        raise FileNotFoundError("Every diagnostic image must exist")
    args.output_dir.mkdir(parents=True)
    rows = [evaluate(path, expected, args.point_sam_url,
                     args.output_dir, index)
            for index, (path, expected) in enumerate(samples)]
    report = {
        "schema_version": "labelagent.ego_bottom_point_fallback_diagnostic.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "assistant_review_candidate_only",
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "production_deployed": False,
        "positive_count": len(args.positive),
        "negative_count": len(args.negative),
        "assistant_review_matches": sum(row["matches_assistant_review"] for row in rows),
        "rows": rows,
        "evidence_note": (
            "The expected labels are bounded assistant visual review, not human "
            "truth. Geometry performance on these development images cannot by "
            "itself authorize production deployment."),
    }
    result = args.output_dir / "results.json"
    result.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({"results": str(result.resolve()),
                      "sha256": file_sha256(result),
                      "matches": report["assistant_review_matches"],
                      "total": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
