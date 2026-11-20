"""Read-only A/B for a downscaled overview plus native candidate crops.

The benchmark reuses the production proposal geometry and semantic contract,
but never updates workbench candidates, truth, references, or training data.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import statistics
import sys
import time

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.agent import candidate_class_audit, workflow
from source.agent.media import open_media
from tools.evaluate_visible_vehicle_recovery_shadow import load_samples


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def compact_overview(image_bytes: bytes, maximum=(640, 512)) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as opened:
        image = ImageOps.contain(opened.convert("RGB"), maximum)
    output = io.BytesIO()
    image.save(output, "JPEG", quality=92, optimize=False, progressive=False)
    return output.getvalue()


def semantic_route(decision: dict, allowed_labels: list[str]) -> str:
    confidence = decision.get("confidence")
    if (type(confidence) not in (int, float) or isinstance(confidence, bool)
            or confidence < 0.90):
        return "rejected"
    classification = decision.get("classification")
    if classification == "conventional_vehicle" and "车辆" in allowed_labels:
        return "车辆"
    if (classification == "autonomous_robot_or_agv"
            and "other_robot" in allowed_labels):
        return "other_robot"
    return "rejected"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--media", required=True, type=Path)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overview-max-width", type=int, default=640)
    parser.add_argument("--overview-max-height", type=int, default=512)
    parser.add_argument("--preserve-overview", action="store_true",
                        help="Send the original overview bytes without resizing")
    parser.add_argument("--force-primary", action="store_true",
                        help="Keep the compact overview but bypass the fast model")
    parser.add_argument("--max-tokens", type=int,
                        help="Experiment-local structured response token cap")
    args = parser.parse_args()

    state_path = args.state.resolve()
    media_path = args.media.resolve()
    sample_path = args.samples.resolve()
    baseline_path = args.baseline.resolve()
    output_path = args.output.resolve()
    artifact_root = (ROOT / "artifacts").resolve()
    if not output_path.is_relative_to(artifact_root):
        raise ValueError("output must be below repository artifacts")
    if output_path.exists():
        raise FileExistsError(output_path)
    for path in (state_path, media_path, sample_path, baseline_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    state_sha256_before = sha256_file(state_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    config = dict(state.get("config") or {})
    if not 1 <= args.overview_max_width <= 1280:
        raise ValueError("overview-max-width must be in [1, 1280]")
    if not 1 <= args.overview_max_height <= 1024:
        raise ValueError("overview-max-height must be in [1, 1024]")
    overview_maximum = (args.overview_max_width, args.overview_max_height)
    # Experiment-local only: the persisted state and production thresholds
    # stay unchanged until both semantic parity and latency improve.
    config["llm_fast_max_pixels"] = args.overview_max_width * args.overview_max_height
    config["llm_fast_max_long_edge"] = max(overview_maximum)
    if args.force_primary:
        config["llm_fast_url"] = ""
        config["llm_fast_model"] = ""
    if args.max_tokens is not None:
        if not 128 <= args.max_tokens <= 4096:
            raise ValueError("max-tokens must be in [128, 4096]")
        config["llm_max_tokens"] = args.max_tokens
    labels = list(state.get("labels") or [])
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_rows = {(row["topic"], row["frame"]): row
                     for row in baseline.get("rows") or []}
    samples = load_samples(sample_path)
    samples = [sample for sample in samples
               if isinstance((baseline_rows.get(
                   (sample["topic"], sample["frame"])) or {}).get(
                       "recovery", {}).get("semantic_guard"), dict)]
    media = open_media(media_path, output_path.parent / (output_path.stem + "-index"))
    instruction, schema = workflow.visible_vehicle_semantic_contract()

    rows = []
    for position, sample in enumerate(samples, 1):
        topic, frame = sample["topic"], sample["frame"]
        baseline_row = baseline_rows.get((topic, frame)) or {}
        baseline_guard = (baseline_row.get("recovery") or {}).get("semantic_guard")
        if not isinstance(baseline_guard, dict):
            continue
        image = media.frame(topic, frame)
        proposals, proposal_evidence = workflow.visible_vehicle_point_proposal(
            config, image, next_id=1)
        if len(proposals) != 1:
            raise RuntimeError(
                f"proposal drift for {topic} frame {frame}: "
                f"{proposal_evidence.get('status')}")
        candidate = proposals[0]
        evidence = candidate_class_audit.prepare_candidate_evidence(image, candidate)
        overview = image if args.preserve_overview else compact_overview(
            image, overview_maximum)
        traces = {}

        def capture(event):
            traces[event["id"]] = event

        started = time.perf_counter()
        with workflow.trace_calls(capture):
            decision = workflow.chat(
                config, instruction, overview,
                extra_images=(evidence["marked"], evidence["isolated"]),
                response_schema=schema)
        elapsed = time.perf_counter() - started
        call = next((event for event in traces.values()
                     if event.get("service") == "GPT"), {})
        output_record = call.get("output") or {}
        local = output_record.get("local_vlm") or {}
        row = {
            "topic": topic, "frame": frame,
            "baseline": baseline_guard,
            "candidate": decision,
            "baseline_route": semantic_route(baseline_guard, labels),
            "candidate_route": semantic_route(decision, labels),
            "route_match": (semantic_route(baseline_guard, labels)
                            == semantic_route(decision, labels)),
            "classification_match": (
                baseline_guard.get("classification")
                == decision.get("classification")),
            "elapsed_seconds": round(elapsed, 3),
            "model_route": output_record.get("model_route"),
            "cache_hit": local.get("cache_hit"),
            "image_count": local.get("image_count"),
            "actual_image_token_counts": local.get("actual_image_token_counts"),
            "call_elapsed_ms": call.get("elapsed_ms"),
            "candidate_only": True,
            "human_truth": False,
            "approved_for_training": False,
        }
        rows.append(row)
        print(json.dumps({"progress": f"{position}/{len(samples)}",
                          "frame": frame, "route": row["model_route"],
                          "route_match": row["route_match"],
                          "elapsed_seconds": row["elapsed_seconds"]},
                         ensure_ascii=False, separators=(",", ":")), flush=True)

    if not rows:
        raise RuntimeError("no baseline semantic rows were evaluated")
    result = {
        "schema_version": "labelagent.visible_machine_semantic_fast_ab.v2",
        "read_only": True,
        "state": {"path": str(state_path), "sha256": state_sha256_before,
                  "modified": sha256_file(state_path) != state_sha256_before},
        "media": str(media_path),
        "samples_sha256": sha256_file(sample_path),
        "baseline_sha256": sha256_file(baseline_path),
        "experiment": {
            "overview_maximum": list(overview_maximum),
            "overview_original_bytes_preserved": args.preserve_overview,
            "fast_primary_max_pixels": config["llm_fast_max_pixels"],
            "fast_primary_max_long_edge": config["llm_fast_max_long_edge"],
            "force_primary": args.force_primary,
            "max_tokens": config.get("llm_max_tokens"),
            "state_or_production_config_modified": False,
        },
        "rows": rows,
        "summary": {
            "evaluated_count": len(rows),
            "route_match_count": sum(row["route_match"] for row in rows),
            "classification_match_count": sum(
                row["classification_match"] for row in rows),
            "fast_route_count": sum(
                row["model_route"] == "small_image_fast" for row in rows),
            "cache_hit_count": sum(row["cache_hit"] is True for row in rows),
            "elapsed_p50_seconds": round(statistics.median(
                row["elapsed_seconds"] for row in rows), 3),
            "candidate_only": True,
            "human_truth_modified": False,
            "training_data_modified": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print("RESULT=" + json.dumps(result["summary"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
