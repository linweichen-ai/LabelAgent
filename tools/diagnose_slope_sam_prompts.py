"""Compare color-independent SAM slope prompts on the hash-bound dev8 set."""

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

from source.agent.workflow import AnnotationAgent, _shape_summary, candidate_overlay, sam_text_multi
from tools.evaluate_slope_static_images import file_sha256, load_inputs


PROMPTS = {
    "baseline": "traversable surface",
    "ramp": "ramp",
    "loading_ramp": "loading ramp",
    "inclined_floor": "inclined floor surface",
    "sloped_floor": "sloped floor",
    "traversable_incline": "traversable incline",
}


def summarize(rows: list[dict]) -> dict:
    output = {}
    for key, text_prompt in PROMPTS.items():
        tp = fp = fn = tn = 0
        for row in rows:
            expected = row["assistant_visible_slope_candidate"]
            predicted = bool(row["prompts"][key]["candidate_count"])
            tp += expected and predicted
            fp += (not expected) and predicted
            fn += expected and not predicted
            tn += (not expected) and not predicted
        output[key] = {
            "text_prompt": text_prompt,
            "true_positive_images": tp,
            "false_positive_images": fp,
            "false_negative_images": fn,
            "true_negative_images": tn,
            "precision_on_assistant_review": tp / (tp + fp) if tp + fp else None,
            "recall_on_assistant_review": tp / (tp + fn) if tp + fn else None,
        }
    return output


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
    if not args.sam_url.startswith("http://127.0.0.1:"):
        raise ValueError("SAM URL must be a loopback endpoint")
    inputs, _ = load_inputs(args.dataset_dir, args.review, args.images_dir)
    state = json.loads(args.state.read_text(encoding="utf-8"))
    config = dict(state.get("config") or {})
    config["sam3_url"] = args.sam_url.rstrip("/")
    args.output.mkdir(parents=True)
    rows = []
    for source in inputs:
        image = source["image_path"].read_bytes()
        started = time.monotonic()
        outputs, diagnostics = sam_text_multi(image, PROMPTS, config)
        agent = AnnotationAgent(config, image, {"prompts": []}, ["slope"], 1)
        prompt_results = {}
        for key, text_prompt in PROMPTS.items():
            shapes = agent.candidates(outputs[key]["results"], "slope")
            overlay = args.output / f"{source['sample_id']}-{key}.jpg"
            overlay.write_bytes(candidate_overlay(image, shapes))
            prompt_results[key] = {
                "text_prompt": text_prompt,
                "candidate_count": len(shapes),
                "candidates": [_shape_summary(shape) for shape in shapes],
                "overlay": overlay.name,
            }
        row = {
            "sample_id": source["sample_id"],
            "image_sha256": source["image_sha256"],
            "assistant_visible_slope_candidate": source[
                "assistant_visible_slope_candidate"],
            "assistant_view": source["assistant_view"],
            "elapsed_seconds": round(time.monotonic() - started, 4),
            "service_diagnostics": diagnostics,
            "prompts": prompt_results,
        }
        rows.append(row)
        print(json.dumps({"sample_id": row["sample_id"],
                          "expected": row["assistant_visible_slope_candidate"],
                          "counts": {key: value["candidate_count"]
                                     for key, value in prompt_results.items()},
                          "elapsed_seconds": row["elapsed_seconds"]},
                         ensure_ascii=False), flush=True)
    report = {
        "schema_version": "labelagent.slope-sam-prompt-diagnostic.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "assistant_review_candidate_only",
        "split": "dev8",
        "source_sequence": "008",
        "human_truth": False,
        "human_confirmed": False,
        "candidate_only": True,
        "production_deployed": False,
        "color_dependent_prompts": False,
        "state_sha256": file_sha256(args.state),
        "review_sha256": file_sha256(args.review),
        "prompts": PROMPTS,
        "summary": summarize(rows),
        "rows": rows,
        "evidence_note": (
            "Raw SAM image-level candidate presence is compared with bounded "
            "assistant visual review. These are not human-truth or pixel-IoU metrics."),
    }
    (args.output / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()),
                      "summary": report["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
