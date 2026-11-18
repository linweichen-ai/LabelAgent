"""Run the production slope path on provenance-audited saved cohorts."""
import argparse
import io
import json
from pathlib import Path
import time

from PIL import Image

from source.agent.evaluation_provenance import classify_target_provenance
from source.agent.mcap_video import Video
from source.agent.workflow import AnnotationAgent, _shape_summary, candidate_overlay
from tools.evaluate_slope_sam import metrics, union_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcap", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--sessions")
    parser.add_argument("--topic", help="Image topic for unlabelled candidate mode")
    parser.add_argument("--frames", required=True, help="Comma-separated frame IDs")
    parser.add_argument("--mode", choices=("negative", "positive", "candidate"),
                        default="negative")
    parser.add_argument("--output", required=True)
    parser.add_argument("--quiet", action="store_true",
                        help="Write per-frame evidence without printing every row")
    parser.add_argument(
        "--require-human-truth", action="store_true",
        help="Fail before model calls unless every saved cohort frame has a current review receipt.")
    args = parser.parse_args()

    mcap = str(Path(args.mcap).resolve())
    state = json.loads(Path(args.state).read_text(encoding="utf-8"))
    if args.mode == "candidate":
        session = {"topic": args.topic or state.get("topic"), "truth": {}}
        if not session["topic"]:
            raise ValueError("Candidate mode requires --topic or a topic in state")
    else:
        if not args.sessions:
            raise ValueError("Positive and negative modes require --sessions")
        sessions = json.loads(Path(args.sessions).read_text(encoding="utf-8"))
        session = sessions[mcap]
    frames = [int(value.strip()) for value in args.frames.split(",") if value.strip()]
    if not frames or len(set(frames)) != len(frames):
        raise ValueError("Evaluation frames must be unique and nonempty")
    truth = session.get("truth") or {}
    for frame in frames:
        shapes = truth.get(str(frame))
        if args.mode == "candidate":
            continue
        if not isinstance(shapes, list):
            raise ValueError(f"Frame {frame} has no saved evaluation target")
        has_slope = any(item.get("label") == "slope" for item in shapes)
        if args.mode == "negative" and has_slope:
            raise ValueError(f"Frame {frame} is not a no-slope negative")
        if args.mode == "positive" and not has_slope:
            raise ValueError(f"Frame {frame} has no saved slope target")

    if args.mode == "candidate":
        target_provenance = {
            "kind": "unlabelled_candidate_mode",
            "human_truth_verified": False,
            "frame_count": len(frames),
            "verified_frames": [],
            "unverified_frames": frames,
            "evidence_note": (
                "Candidate mode has no evaluation targets and produces review-only outputs."),
        }
    else:
        cohort_labels = sorted({
            shape.get("label")
            for frame in frames
            for shape in truth.get(str(frame), [])
            if isinstance(shape, dict) and shape.get("label")
        })
        target_provenance = classify_target_provenance(
            session, frames, cohort_labels if args.mode == "negative" else ["slope"])
        if args.require_human_truth and not target_provenance["human_truth_verified"]:
            raise ValueError(
                "Human truth is required but target provenance is "
                f"{target_provenance['kind']}; unverified frames: "
                f"{target_provenance['unverified_frames']}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    video = Video(mcap)
    config = state.get("config") or {}
    rows = []
    for frame in frames:
        image = video.frame(session["topic"], frame)
        events = []
        started = time.monotonic()
        agent = AnnotationAgent(config, image, {
            "description": "slope negative evaluation",
            "prompts": [{"label": "pallet", "text": "pallet"},
                        {"label": "slope", "text": "traversable surface"}],
        }, ["pallet", "slope"], 1, on_progress=events.append)
        agent.generate()
        slope_shapes = [shape for shape in agent.shapes if shape.get("label") == "slope"]
        pallet_shapes = [shape for shape in agent.shapes if shape.get("label") == "pallet"]
        row = {
            "frame": frame, "seconds": round(time.monotonic() - started, 4),
            "candidate_count": len(slope_shapes),
            "crop_fallback_used": any(
                event.get("operation") == "slope_overlap_crop_fallback" for event in events),
            "filtered_count": len(agent.candidate_filters),
            "candidates": [_shape_summary(shape) | {
                "source": shape.get("source"), "crop_index": shape.get("crop_index")}
                for shape in slope_shapes],
            "pallet_candidates": [_shape_summary(shape) for shape in pallet_shapes],
            "filter_reason_codes": [item.get("reason_code")
                                    for item in agent.candidate_filters],
        }
        if args.mode == "positive":
            width, height = Image.open(io.BytesIO(image)).size
            prediction = union_mask(slope_shapes, width, height)
            expected = union_mask(
                [shape for shape in truth[str(frame)] if shape.get("label") == "slope"],
                width, height)
            row.update(metrics(prediction, expected))
        rows.append(row)
        if slope_shapes or args.mode == "candidate":
            suffix = "review-candidates" if args.mode == "candidate" else "false-candidates"
            (output / f"frame-{frame}-{suffix}.jpg").write_bytes(
                candidate_overlay(image, slope_shapes))
        if not args.quiet:
            print(json.dumps(row, ensure_ascii=False), flush=True)

    summary = {
        "frames": len(rows),
        "frames_with_candidate": sum(row["candidate_count"] > 0 for row in rows),
        "candidates": sum(row["candidate_count"] for row in rows),
        "crop_fallback_frames": sum(row["crop_fallback_used"] for row in rows),
        "mean_seconds": sum(row["seconds"] for row in rows) / len(rows),
    }
    if args.mode == "negative":
        summary.update({
            "negative_cohort_human_truth_verified": target_provenance["human_truth_verified"],
            "frames_with_slope_candidate": summary["frames_with_candidate"],
            "slope_candidates": summary["candidates"],
            "frames_with_false_candidate": (summary["frames_with_candidate"]
                                             if target_provenance["human_truth_verified"]
                                             else None),
            "false_candidates": (summary["candidates"]
                                 if target_provenance["human_truth_verified"]
                                 else None),
        })
    if args.mode == "positive":
        summary.update({
            "frames_iou_ge_0_5": sum(row["iou"] >= 0.5 for row in rows),
            "mean_iou": sum(row["iou"] for row in rows) / len(rows),
            "mean_precision": sum(row["precision"] for row in rows) / len(rows),
            "mean_recall": sum(row["recall"] for row in rows) / len(rows),
        })
    report = {
        "mcap": mcap, "topic": session["topic"], "mode": args.mode,
        "target_provenance": target_provenance,
        "summary": summary, "rows": rows,
        "evidence_note": (
            target_provenance["evidence_note"]
            + (" No accuracy claim or truth mutation is allowed."
               if args.mode == "candidate" else
               " Saved targets were used only for cohort membership and post-generation scoring; "
               "generation used image/model data only.")),
    }
    (output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
