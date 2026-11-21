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
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from source.agent.workflow import candidate_overlay
from tools.diagnose_ego_bottom_point_fallback import (
    bounded_candidate_summary, mask_record)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_point_sets(payload, width: int, height: int) -> list[dict]:
    if not isinstance(payload, list) or not 1 <= len(payload) <= 8:
        raise ValueError("point-set file must contain 1 to 8 experiments")
    names = set()
    validated = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("each point set must be an object")
        name, points = item.get("name"), item.get("points")
        if (not isinstance(name, str) or not name.strip()
                or name in names or not isinstance(points, list)
                or not 1 <= len(points) <= 32):
            raise ValueError("point-set names must be unique and contain 1 to 32 points")
        normalized = []
        for point in points:
            if not isinstance(point, dict):
                raise ValueError("every point must be an object")
            x, y, label = point.get("x"), point.get("y"), point.get("label")
            if (type(x) is not int or type(y) is not int or label not in (0, 1)
                    or not 0 <= x < width or not 0 <= y < height):
                raise ValueError("point coordinates or labels are invalid")
            normalized.append({"x": x, "y": y, "label": label})
        if not any(point["label"] == 1 for point in normalized):
            raise ValueError("each point set requires at least one positive point")
        names.add(name)
        validated.append({"name": name.strip(), "points": normalized})
    return validated


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
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--point-sets", required=True, type=Path)
    parser.add_argument("--point-sam-url", default="http://127.0.0.1:28766")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not args.point_sam_url.startswith("http://127.0.0.1:"):
        raise ValueError("point SAM URL must be a loopback endpoint")
    if not args.image.is_file() or not args.point_sets.is_file():
        raise FileNotFoundError("image and point-set file are required")

    image = args.image.read_bytes()
    with Image.open(args.image) as decoded:
        width, height = decoded.size
    point_sets = validate_point_sets(
        json.loads(args.point_sets.read_text(encoding="utf-8")), width, height)
    args.output_dir.mkdir(parents=True)
    rows = []
    for index, experiment in enumerate(point_sets):
        started = time.monotonic()
        result = {}
        failure = None
        try:
            result = request_points(args.point_sam_url, image, experiment["points"])
        except Exception as error:  # Shadow diagnostics must preserve the other rows.
            failure = str(error)[:500]
        parts = result.get("parts") or []
        shape = None
        geometry = None
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
            shape.pop("_record", None)
            shape.pop("point", None)
        overlay = args.output_dir / f"{index:02d}-{experiment['name']}.jpg"
        overlay.write_bytes(candidate_overlay(image, [shape] if shape else []))
        rows.append({
            "name": experiment["name"],
            "points": experiment["points"],
            "elapsed_seconds": round(time.monotonic() - started, 4),
            "candidate_available": shape is not None,
            "error": failure,
            "geometry": geometry,
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
        "point_sets_sha256": file_sha256(args.point_sets),
        "rows": rows,
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
        "experiment_count": len(rows), "sha256": file_sha256(output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
