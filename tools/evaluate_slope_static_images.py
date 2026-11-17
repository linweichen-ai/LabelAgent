"""Run the production slope SAM path on a hash-bound static dev8 image set."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from source.agent.workflow import AnnotationAgent, _shape_summary, candidate_overlay


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_inputs(dataset_dir: Path, review_path: Path,
                images_dir: Path) -> tuple[list[dict], dict]:
    manifest_path = dataset_dir / "manifest.json"
    dev_path = dataset_dir / "dev.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("status") != "ready_for_isolated_lora_smoke"
            or manifest.get("human_truth") is not False
            or manifest.get("production_training_approved") is not False
            or manifest.get("test24_included") is not False
            or manifest.get("dev_samples") != 8
            or manifest.get("dev_jsonl_sha256") != file_sha256(dev_path)):
        raise ValueError("dev8 dataset boundary differs")
    rows = [json.loads(line) for line in dev_path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if (review.get("schema_version") != "labelagent.slope-visual-review.v1"
            or review.get("split") != "dev8"
            or review.get("source_sequence") != "008"
            or review.get("human_truth") is not False
            or review.get("human_confirmed") is not False
            or review.get("training_authorized") is not False):
        raise ValueError("review boundary differs")
    decisions = {row.get("sample_id"): row for row in review.get("decisions", [])}
    if len(rows) != 8 or len(decisions) != 8:
        raise ValueError("dev8 rows or decisions are incomplete")
    result = []
    seen = set()
    for row in rows:
        sample_id = row.get("sample_id")
        if (row.get("split") != "dev" or row.get("source_sequence") != "008"
                or row.get("human_truth") is not False
                or row.get("production_training_approved") is not False
                or sample_id in seen or sample_id not in decisions):
            raise ValueError("dev8 row isolation differs")
        image_path = images_dir / f"{sample_id}.png"
        if not image_path.is_file() or file_sha256(image_path) != row.get("image_sha256"):
            raise ValueError(f"image binding differs: {sample_id}")
        decision = decisions[sample_id]
        visible = decision.get("assistant_visible_slope_candidate")
        if not isinstance(visible, bool):
            raise ValueError("review decision is invalid")
        seen.add(sample_id)
        result.append({
            "sample_id": sample_id,
            "image_path": image_path,
            "image_sha256": row["image_sha256"],
            "assistant_visible_slope_candidate": visible,
            "assistant_view": decision.get("view"),
        })
    if set(seen) != set(decisions):
        raise ValueError("review contains extra samples")
    return result, review


def summarize_presence(rows: list[dict]) -> dict:
    tp = sum(row["candidate_positive"] and row["assistant_visible_slope_candidate"]
             for row in rows)
    fp = sum(row["candidate_positive"] and not row["assistant_visible_slope_candidate"]
             for row in rows)
    fn = sum(not row["candidate_positive"] and row["assistant_visible_slope_candidate"]
             for row in rows)
    tn = sum(not row["candidate_positive"] and not row["assistant_visible_slope_candidate"]
             for row in rows)
    return {
        "samples": len(rows),
        "assistant_visible_slope_candidates": sum(
            row["assistant_visible_slope_candidate"] for row in rows),
        "candidate_positive_images": sum(row["candidate_positive"] for row in rows),
        "true_positive_candidates": tp,
        "false_positive_candidates": fp,
        "false_negative_candidates": fn,
        "true_negative_candidates": tn,
        "precision_on_assistant_review": tp / (tp + fp) if tp + fp else None,
        "recall_on_assistant_review": tp / (tp + fn) if tp + fn else None,
        "total_slope_candidates": sum(row["slope_candidate_count"] for row in rows),
        "crop_fallback_images": sum(row["crop_fallback_used"] for row in rows),
        "elapsed_seconds": round(sum(row["elapsed_seconds"] for row in rows), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--sam-url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    inputs, review = load_inputs(args.dataset_dir, args.review, args.images_dir)
    state = json.loads(args.state.read_text(encoding="utf-8"))
    config = dict(state.get("config") or {})
    if not args.sam_url.startswith("http://127.0.0.1:"):
        raise ValueError("SAM URL must be a loopback endpoint")
    config["sam3_url"] = args.sam_url.rstrip("/")
    args.output.mkdir(parents=True)
    result_rows = []
    for source in inputs:
        image = source["image_path"].read_bytes()
        events = []
        started = time.monotonic()
        agent = AnnotationAgent(config, image, {
            "description": "bounded static dev8 slope candidate evaluation",
            "prompts": [
                {"label": "pallet", "text": "pallet"},
                {"label": "slope", "text": "traversable surface"},
            ],
        }, ["pallet", "slope"], 1, on_progress=events.append)
        agent.generate()
        slope_shapes = [shape for shape in agent.shapes
                        if shape.get("label") == "slope"]
        pallet_shapes = [shape for shape in agent.shapes
                         if shape.get("label") == "pallet"]
        elapsed = time.monotonic() - started
        output_image = args.output / f"{source['sample_id']}-slope-candidates.jpg"
        output_image.write_bytes(candidate_overlay(image, slope_shapes))
        result_rows.append({
            "sample_id": source["sample_id"],
            "image_sha256": source["image_sha256"],
            "assistant_visible_slope_candidate": source[
                "assistant_visible_slope_candidate"],
            "assistant_view": source["assistant_view"],
            "candidate_positive": bool(slope_shapes),
            "slope_candidate_count": len(slope_shapes),
            "pallet_candidate_count": len(pallet_shapes),
            "crop_fallback_used": any(
                event.get("operation") == "slope_overlap_crop_fallback"
                for event in events),
            "filter_reason_codes": [item.get("reason_code")
                                    for item in agent.candidate_filters],
            "slope_candidates": [_shape_summary(shape) | {
                "source": shape.get("source"),
                "crop_index": shape.get("crop_index"),
            } for shape in slope_shapes],
            "elapsed_seconds": round(elapsed, 4),
            "overlay": output_image.name,
        })
        print(json.dumps(result_rows[-1], ensure_ascii=False), flush=True)
    report = {
        "schema_version": "labelagent.slope-static-image-eval.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "assistant_review_candidate_only",
        "split": "dev8",
        "source_sequence": "008",
        "human_truth": False,
        "human_confirmed": False,
        "candidate_only": True,
        "production_deployed": False,
        "state_sha256": file_sha256(args.state),
        "review_sha256": file_sha256(args.review),
        "sam_url_scope": "loopback",
        "prompts": ["pallet", "traversable surface"],
        "summary": summarize_presence(result_rows),
        "rows": result_rows,
        "evidence_note": (
            "The production SAM candidate path was run read-only. Metrics compare "
            "candidate presence with bounded assistant visual review, not human truth "
            "or pixel segmentation IoU."),
    }
    (args.output / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()),
                      "summary": report["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
