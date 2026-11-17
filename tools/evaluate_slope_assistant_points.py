"""Evaluate assistant-candidate slope points through the production SAM endpoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from source.agent.workflow import _shape_summary, candidate_overlay, sam_request
from tools.evaluate_slope_static_images import file_sha256, load_inputs


def load_point_candidates(path: Path, rows: list[dict]) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (payload.get("schema_version") !=
            "labelagent.slope-assistant-point-candidates.v1"
            or payload.get("status") != "assistant_candidate_only"
            or payload.get("split") != "dev8"
            or payload.get("source_sequence") != "008"
            or payload.get("human_truth") is not False
            or payload.get("human_confirmed") is not False
            or payload.get("training_authorized") is not False
            or payload.get("color_rule_used") is not False):
        raise ValueError("assistant point candidate boundary differs")
    by_id = {row["sample_id"]: row for row in rows}
    items = payload.get("items") or []
    if len(items) != 4 or len({item.get("sample_id") for item in items}) != 4:
        raise ValueError("assistant point candidates must cover four unique images")
    output = []
    for item in items:
        source = by_id.get(item.get("sample_id"))
        if source is None or source["assistant_visible_slope_candidate"] is not True:
            raise ValueError("assistant point candidate is outside reviewed positives")
        points = item.get("points")
        if (not isinstance(points, list) or not 1 <= len(points) <= 32
                or not any(point.get("label") == 1 for point in points)
                or any(set(point) != {"x", "y", "label"}
                       or point["label"] not in (0, 1)
                       or type(point["x"]) is not int or type(point["y"]) is not int
                       or not 0 <= point["x"] < 640 or not 0 <= point["y"] < 512
                       for point in points)):
            raise ValueError("assistant point candidate points are invalid")
        output.append(source | {"points": points})
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--points", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--sam-url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.sam_url.startswith("http://127.0.0.1:"):
        raise ValueError("SAM URL must be a loopback endpoint")
    rows, _ = load_inputs(args.dataset_dir, args.review, args.images_dir)
    items = load_point_candidates(args.points, rows)
    state = json.loads(args.state.read_text(encoding="utf-8"))
    config = dict(state.get("config") or {})
    config["sam3_url"] = args.sam_url.rstrip("/")
    args.output.mkdir(parents=True)
    results = []
    for item in items:
        image = item["image_path"].read_bytes()
        started = time.monotonic()
        result = sam_request(config, "/segment_points", image, points=item["points"])
        shape = {"id": 1, "label": "slope", "points": result.get("points") or [],
                 "holes": result.get("holes") or [],
                 "parts": result.get("parts") or [], "score": result.get("score")}
        overlay = args.output / f"{item['sample_id']}-assistant-points.jpg"
        overlay.write_bytes(candidate_overlay(image, [shape]))
        row = {
            "sample_id": item["sample_id"],
            "image_sha256": item["image_sha256"],
            "points": item["points"],
            "candidate": _shape_summary(shape) | {"score": shape.get("score")},
            "warnings": result.get("warnings") or [],
            "elapsed_seconds": round(time.monotonic() - started, 4),
            "overlay": overlay.name,
        }
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    report = {
        "schema_version": "labelagent.slope-assistant-point-eval.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "assistant_candidate_only",
        "split": "dev8", "source_sequence": "008",
        "human_truth": False, "human_confirmed": False,
        "candidate_only": True, "production_deployed": False,
        "state_sha256": file_sha256(args.state),
        "review_sha256": file_sha256(args.review),
        "points_sha256": file_sha256(args.points),
        "summary": {"images": len(results),
                    "sam_candidates_returned": sum(bool(row["candidate"]["vertices"])
                                                   for row in results),
                    "warnings": sum(bool(row["warnings"]) for row in results),
                    "elapsed_seconds": round(sum(row["elapsed_seconds"]
                                                 for row in results), 3)},
        "rows": results,
        "evidence_note": (
            "Assistant-selected diagnostic points are not human truth. This run only "
            "tests SAM point-prompt segmentability; visual mask quality requires review."),
    }
    (args.output / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()),
                      "summary": report["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
