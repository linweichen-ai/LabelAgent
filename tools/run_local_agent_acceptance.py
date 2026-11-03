"""Run the production Agent on imported images, writing an isolated local receipt.

Requires a real VLM service and an explicitly supplied 2D reference. Never writes
to the workbench, source annotations, or its truth store. No simulated backend.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import time

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from connect_local_vlm import loopback_url, require_real_service


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_image(dataset, record):
    dataset = Path(dataset).resolve()
    path = (dataset / record["source_file"]).resolve()
    if not path.is_relative_to(dataset):
        raise ValueError("Manifest image path escapes the dataset")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError("Source image changed since import")
    return raw


def evaluate_run(result, traces):
    """Keep completion, observed editing and accuracy as separate claims."""
    completed = [item for item in traces if item.get("status") == "succeeded"]
    vlm_calls = sum(item.get("service") == "GPT" for item in completed)
    sam_calls = sum(item.get("service") == "SAM 3" for item in completed)
    agent = result.get("agent", {})
    edits = sum(item.get("status") == "applied"
                for entry in agent.get("audit", []) for item in entry.get("tool_results", []))
    loop_completed = bool(vlm_calls >= 1 and sam_calls >= 1 and agent.get("rounds", 0) >= 1
                          and agent.get("converged") and agent.get("stop_reason") == "done")
    return {"loop_completed": loop_completed, "successful_vlm_calls": vlm_calls,
            "successful_sam_calls": sam_calls, "rounds": agent.get("rounds", 0),
            "converged": bool(agent.get("converged")), "stop_reason": agent.get("stop_reason"),
            "generated_candidates": result.get("candidates_added", 0),
            "remaining_candidates": len(result.get("shapes", [])), "applied_edits": edits,
            "multi_round_correction_observed": bool(loop_completed and edits and agent.get("rounds", 0) >= 2),
            "pixel_accuracy_evaluated": False, "candidates_confirmed_as_truth": False}


def acceptance(args):
    dataset = Path(args.dataset).resolve()
    manifest = json.loads((dataset / "import_manifest.json").read_text(encoding="utf-8"))
    records = {row["frame"]: row for row in manifest["records"] if row["topic"] == args.topic}
    targets = [int(value) for value in args.frames.split(",")]
    if not targets or len(set(targets)) != len(targets) or args.reference_frame in targets:
        raise ValueError("Target frames must be distinct and exclude the reference frame")
    if any(frame not in records for frame in [args.reference_frame, *targets]):
        raise ValueError("Requested image frame is not in the import manifest")
    if not 1 <= args.max_rounds <= 20:
        raise ValueError("Acceptance budget must be 1 to 20 review rounds")
    labels = [value.strip() for value in args.labels.split(",") if value.strip()]
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("Provide distinct labels")
    reference_bytes = Path(args.reference_json).read_bytes()
    annotations = json.loads(reference_bytes)
    if not isinstance(annotations, list) or not 1 <= len(annotations) <= 200:
        raise ValueError("Reference JSON must be a nonempty 2D shape list")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
    from agent import workflow
    reference_image = load_image(dataset, records[args.reference_frame])
    with Image.open(io.BytesIO(reference_image)) as image:
        width, height = image.size
    ids = []
    for shape in annotations:
        if (not isinstance(shape, dict) or shape.get("label") not in labels
                or type(shape.get("id")) is not int or not workflow.valid_shape(shape, width, height)):
            raise ValueError("Reference contains an invalid 2D shape, integer ID or label")
        ids.append(shape["id"])
    if len(set(ids)) != len(ids) or {shape["label"] for shape in annotations} != set(labels):
        raise ValueError("Reference must have unique IDs and cover all requested labels")
    # Read and verify all inputs before making inference calls or creating output.
    images = {frame: load_image(dataset, records[frame]) for frame in targets}
    base, sam = loopback_url(args.base), loopback_url(args.sam)
    require_real_service(base, args.model)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = {"llm_url": base, "llm_model": args.model, "llm_key": "local", "sam3_url": sam}
    summary = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
               "archive_sha256": manifest["archive_sha256"], "topic": args.topic,
               "reference_frame": args.reference_frame, "reference_kind": args.reference_kind,
               "reference_json_sha256": hashlib.sha256(reference_bytes).hexdigest(),
               "full_agent_verified": False, "pixel_accuracy_evaluated": False,
               "candidates_confirmed_as_truth": False, "workbench_modified": False, "frames": []}
    save(output / "public_summary.json", summary)
    traces = {}

    def record(event):
        traces[event["id"]] = event
        save(output / "model_trace.json", list(traces.values()))

    try:
        marked = workflow.candidate_overlay(reference_image, annotations)
        with workflow.trace_calls(record):
            description = workflow.describe(config, reference_image, annotations, labels, marked_image=marked)
            save(output / "description.json", description)
            for frame in targets:
                before = set(traces)
                started = time.monotonic()
                events = []

                def progress(event):
                    events.append({"phase": event["phase"], "round": event["round"],
                                   "shape_count": len(event["shapes"])})
                    save(output / f"frame_{frame}_events.json", events)

                result = workflow.run(config, images[frame], description, labels, max_rounds=args.max_rounds,
                    on_progress=progress, reference={"image": reference_image, "marked_image": marked,
                                                    "annotations": annotations})
                save(output / f"frame_{frame}_result.json", result)
                (output / f"frame_{frame}_candidates.jpg").write_bytes(workflow.candidate_overlay(images[frame], result["shapes"]))
                evidence = evaluate_run(result, [value for key, value in traces.items() if key not in before])
                summary["frames"].append({"frame": frame, "seconds": round(time.monotonic()-started, 3), **evidence})
                save(output / "public_summary.json", summary)
        reference_ok = any(item.get("service") == "GPT" and item.get("status") == "succeeded" for item in traces.values())
        summary["full_agent_verified"] = bool(reference_ok and summary["frames"] and all(row["loop_completed"] for row in summary["frames"]))
        summary["status"] = "complete" if summary["full_agent_verified"] else "incomplete"
    except Exception as error:
        # Full diagnostics stay local; the public summary omits paths and URLs.
        (output / "local_error.txt").write_text(f"{type(error).__name__}: {error}", encoding="utf-8")
        summary.update(status="failed", error_type=type(error).__name__)
    finally:
        try:
            for frame in [args.reference_frame, *targets]:
                load_image(dataset, records[frame])
            summary["source_images_unchanged"] = True
        except Exception:
            summary.update(source_images_unchanged=False, full_agent_verified=False, status="failed")
        summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save(output / "public_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["full_agent_verified"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--topic", default="/capture/camera_1/image")
    parser.add_argument("--reference-frame", type=int, default=0)
    parser.add_argument("--reference-json", required=True)
    parser.add_argument("--reference-kind", choices=["human_reviewed", "unconfirmed_functional_fixture"], required=True)
    parser.add_argument("--frames", default="76,151")
    parser.add_argument("--labels", default="pallet,person")
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--base", default="http://127.0.0.1:18868/v1")
    parser.add_argument("--model", default="labelagent-vlm")
    parser.add_argument("--sam", default="http://127.0.0.1:8766")
    parser.add_argument("--output", required=True)
    raise SystemExit(acceptance(parser.parse_args()))
