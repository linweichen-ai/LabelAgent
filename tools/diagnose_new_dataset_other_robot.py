#!/usr/bin/env python3
"""Read-only point-scan diagnostic for missed other_robot candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from source.agent.mcap_video import Video
from tools.evaluate_reference_point_scan import evaluate_frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--frame", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--point-sam-url", default="http://127.0.0.1:28766")
    parser.add_argument("--top-k", default=12, type=int)
    parser.add_argument("--exemplar-sheet", type=Path)
    parser.add_argument("--adaptive-exemplar-review", action="store_true")
    parser.add_argument("--isolated-semantic-triage", action="store_true")
    args = parser.parse_args()

    parsed = urlparse(args.point_sam_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        parser.error("--point-sam-url must use a loopback endpoint")
    if args.frame < 0 or not 1 <= args.top_k <= 24:
        parser.error("invalid frame or top-k")
    if args.output.exists():
        raise FileExistsError(args.output)

    state_bytes = args.state.read_bytes()
    state = json.loads(state_bytes.decode("utf-8"))
    path = state.get("video_path") or state.get("path")
    topic = state.get("topic")
    config = dict(state.get("config") or {})
    if not path or not topic or not config.get("llm_url"):
        raise ValueError("state must contain a loaded video, topic and model config")

    video = Video(path)
    image = video.frame(topic, args.frame)
    existing = list((state.get("candidates") or {}).get(str(args.frame)) or [])
    exemplar_sheet = (args.exemplar_sheet.read_bytes()
                      if args.exemplar_sheet else None)
    if args.adaptive_exemplar_review and exemplar_sheet is None:
        parser.error("--adaptive-exemplar-review requires --exemplar-sheet")
    args.output.mkdir(parents=True)
    frame_report = evaluate_frame(
        args.frame,
        image,
        existing,
        config,
        args.point_sam_url,
        args.output,
        args.top_k,
        exemplar_sheet=exemplar_sheet,
        adaptive_exemplar_review=args.adaptive_exemplar_review,
        isolated_semantic_triage=args.isolated_semantic_triage,
    )
    report = {
        "schema_version": "labelagent.new-dataset-other-robot-point-diagnostic.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "state_sha256": hashlib.sha256(state_bytes).hexdigest(),
        "dataset": str(path),
        "topic": topic,
        "frame": args.frame,
        "point_sam_url": args.point_sam_url,
        "top_k": args.top_k,
        "exemplar_sheet": str(args.exemplar_sheet) if args.exemplar_sheet else None,
        "adaptive_exemplar_review": args.adaptive_exemplar_review,
        "isolated_semantic_triage": args.isolated_semantic_triage,
        "result": frame_report,
    }
    result_path = args.output / "results.json"
    result_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(result_path.resolve()),
        "raw_candidates": frame_report["raw_candidates"],
        "classified_count": frame_report["classified_count"],
        "selected_ids": frame_report["selected_ids"],
        "uncertain_review_ids": frame_report["uncertain_review_ids"],
        "elapsed_seconds": round(frame_report["frame_elapsed_seconds"], 3),
    }, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
