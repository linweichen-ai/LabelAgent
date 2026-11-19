"""Compare color-independent SAM prompts for distinct non-ego robots.

The output is candidate-only diagnostic evidence. It never modifies the
workbench state, human truth, training approval, or the production prompt.
"""

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

from source.agent.workflow import (AnnotationAgent, _shape_summary,
                                   candidate_overlay, sam_text_multi)


PROMPTS = {
    "baseline_robot": "robot",
    "mobile_platform": "mobile platform",
    "warehouse_robot": "warehouse robot",
    "autonomous_mobile_robot": "autonomous mobile robot",
    "robotic_pallet_truck": "robotic pallet truck",
    "forklift_robot": "forklift robot",
    "industrial_mobile_robot": "industrial mobile robot",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--sam-url", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not args.sam_url.startswith("http://127.0.0.1:"):
        raise ValueError("SAM URL must be a loopback endpoint")
    if not args.images or any(not path.is_file() for path in args.images):
        raise FileNotFoundError("Every diagnostic image must exist")

    state_bytes = args.state.read_bytes()
    state = json.loads(state_bytes.decode("utf-8"))
    config = dict(state.get("config") or {})
    config["sam3_url"] = args.sam_url.rstrip("/")
    args.output_dir.mkdir(parents=True)
    rows = []
    for index, image_path in enumerate(args.images):
        image = image_path.read_bytes()
        started = time.monotonic()
        outputs, diagnostics = sam_text_multi(image, PROMPTS, config)
        agent = AnnotationAgent(config, image, {"prompts": []},
                                ["other_robot"], 1)
        prompt_results = {}
        for key, text_prompt in PROMPTS.items():
            shapes = agent.candidates(outputs[key]["results"], "other_robot")
            overlay = args.output_dir / f"{index:02d}-{image_path.stem}-{key}.jpg"
            overlay.write_bytes(candidate_overlay(image, shapes))
            prompt_results[key] = {
                "text_prompt": text_prompt,
                "candidate_count": len(shapes),
                "candidates": [_shape_summary(shape) for shape in shapes],
                "overlay": overlay.name,
                "overlay_sha256": file_sha256(overlay),
            }
        row = {
            "image": str(image_path.resolve()),
            "image_sha256": hashlib.sha256(image).hexdigest(),
            "elapsed_seconds": round(time.monotonic() - started, 4),
            "service_diagnostics": diagnostics,
            "prompts": prompt_results,
        }
        rows.append(row)
        print(json.dumps({
            "image": image_path.name,
            "counts": {key: value["candidate_count"]
                       for key, value in prompt_results.items()},
            "elapsed_seconds": row["elapsed_seconds"],
        }, ensure_ascii=False), flush=True)

    report = {
        "schema_version": "labelagent.other_robot_sam_prompt_diagnostic.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "assistant_review_candidate_only",
        "candidate_only": True,
        "human_truth": False,
        "human_confirmed": False,
        "approved_for_training": False,
        "production_deployed": False,
        "color_dependent_prompts": False,
        "state_sha256": hashlib.sha256(state_bytes).hexdigest(),
        "prompts": PROMPTS,
        "rows": rows,
        "evidence_note": (
            "Raw SAM prompt candidates are localization proposals only. A prompt "
            "must pass representative positives, hard negatives, and the existing "
            "blind VLM class gate before production use."),
    }
    result_path = args.output_dir / "results.json"
    result_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({"output": str(result_path.resolve()),
                      "sha256": file_sha256(result_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
