"""Generate resumable, hash-bound Qwen8B suggestions for mixed-hard review.

The output is deliberately separate from the human adjudication template.  It
may help a reviewer, but it cannot populate truth or approve training data.
"""
import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.agent.workflow import chat
from source.agent.expert_routing import evaluate_expert_shadow_route


LABELS = ("车辆", "pallet", "person", "floor", "slope", "ego",
          "other_robot", "cargo_floor")
EVIDENCE = {"present", "absent", "uncertain"}
PALLET_QUALITY = {"no_pallet", "clean", "holes_or_notches", "fragmented",
                  "merged_instances", "multiple_issues", "uncertain"}
SLOPE_GEOMETRY = {"no_visible_slope", "full_slope", "partial_edge_slope",
                  "occluded_slope", "uncertain"}
CONFLICTS = {"vehicle_other_robot", "vehicle_ego", "floor_slope",
             "pallet_cargo_floor", "none", "uncertain"}
DIFFICULTIES = {"edge_target", "small_target", "far_target", "occlusion",
                "dense_texture", "instance_merge", "boundary_holes",
                "class_conflict", "unknown_class", "none"}
SCHEMA_VERSION = "labelagent.local-qwen-mixed-hard-suggestions.v1"

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "assessments": {
            "type": "object",
            "properties": {
                label: {"type": "string", "enum": sorted(EVIDENCE)}
                for label in LABELS
            },
            "required": list(LABELS),
            "additionalProperties": False,
        },
        "overall_confidence": {"type": "number", "minimum": 0,
                               "maximum": 1},
        "pallet_instance_count": {"type": "integer", "minimum": 0,
                                  "maximum": 50},
        "pallet_quality": {"type": "string", "enum": sorted(PALLET_QUALITY)},
        "slope_geometry": {"type": "string", "enum": sorted(SLOPE_GEOMETRY)},
        "class_conflicts": {
            "type": "array", "maxItems": 4, "uniqueItems": True,
            "items": {"type": "string", "enum": sorted(CONFLICTS)},
        },
        "difficulty_flags": {
            "type": "array", "maxItems": 9, "uniqueItems": True,
            "items": {"type": "string", "enum": sorted(DIFFICULTIES)},
        },
        "unknown_candidates": {
            "type": "array", "maxItems": 5,
            "items": {"type": "string"},
        },
        "review_note": {"type": "string"},
    },
    "required": ["assessments", "overall_confidence", "pallet_instance_count", "pallet_quality",
                 "slope_geometry", "class_conflicts", "difficulty_flags",
                 "unknown_candidates", "review_note"],
    "additionalProperties": False,
}

INSTRUCTION = """You are independently reviewing one factory mobile-robot camera image.
Do not assume a class from color or from any prior candidate. Inspect visible evidence only.
The exact project labels are: 车辆, pallet, person, floor, slope, ego, other_robot, cargo_floor.
Treat slope as a physically inclined traversable surface, not merely orange/yellow paint.
Distinguish ego robot body/forks from other robots and vehicles. Distinguish pallet from a
raised cargo/loading floor. Count visible pallet instances conservatively. Report uncertain
when an edge crop, occlusion, distance, or texture prevents a reliable decision. Keep the
review_note under 60 characters. Return only the requested compact JSON. This is an assistant
suggestion, never human truth."""


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def _validate_machine_result(result):
    if not isinstance(result, dict) or set(result) != set(RESPONSE_SCHEMA["required"]):
        raise ValueError("local Qwen result fields differ from the frozen schema")
    assessments = result.get("assessments")
    if not isinstance(assessments, dict) or set(assessments) != set(LABELS):
        raise ValueError("local Qwen result does not contain the exact label set")
    for label in LABELS:
        item = assessments[label]
        if item not in EVIDENCE:
            raise ValueError(f"invalid local Qwen assessment: {label}")
    confidence = result.get("overall_confidence")
    if type(confidence) not in (int, float) or not 0 <= confidence <= 1:
        raise ValueError("invalid overall confidence")
    count = result.get("pallet_instance_count")
    if type(count) is not int or not 0 <= count <= 50:
        raise ValueError("invalid pallet instance count")
    if result.get("pallet_quality") not in PALLET_QUALITY:
        raise ValueError("invalid pallet quality")
    if result.get("slope_geometry") not in SLOPE_GEOMETRY:
        raise ValueError("invalid slope geometry")
    for key, allowed, maximum in (("class_conflicts", CONFLICTS, 4),
                                  ("difficulty_flags", DIFFICULTIES, 9)):
        values = result.get(key)
        if (not isinstance(values, list) or len(values) > maximum
                or len(values) != len(set(values))
                or any(value not in allowed for value in values)):
            raise ValueError(f"invalid {key}")
    unknown = result.get("unknown_candidates")
    if (not isinstance(unknown, list) or len(unknown) > 5
            or any(not isinstance(value, str) for value in unknown)):
        raise ValueError("invalid unknown candidates")
    if not isinstance(result.get("review_note"), str):
        raise ValueError("invalid review note")


def load_inputs(manifest_path, receipt_path, image_root):
    manifest_path, receipt_path = Path(manifest_path), Path(receipt_path)
    image_root = Path(image_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    rows = manifest.get("rows") if isinstance(manifest, dict) else None
    if (manifest.get("schema_version") != "labelagent.mixed-hard-review-pack.v1"
            or not isinstance(rows, list) or len(rows) != 34
            or len({row.get("sample_id") for row in rows}) != 34):
        raise ValueError("mixed-hard manifest must contain 34 unique rows")
    if manifest.get("summary", {}).get("test24_intersection_count") != 0:
        raise ValueError("test24 is forbidden")
    if any(row.get("human_truth") is not False
           or row.get("approved_for_training") is not False for row in rows):
        raise ValueError("manifest safety boundary differs")
    files = receipt.get("files") if isinstance(receipt, dict) else None
    if (receipt.get("candidate_only") is not True
            or receipt.get("human_truth") is not False
            or receipt.get("approved_for_training") is not False
            or not isinstance(files, dict)):
        raise ValueError("workspace receipt safety boundary differs")
    loaded = []
    for row in rows:
        relative = str(row.get("raw_image") or "")
        if not relative.startswith("raw/") or ".." in Path(relative).parts:
            raise ValueError(f"unsafe image path: {row.get('sample_id')}")
        path = image_root / Path(relative)
        if not path.is_file():
            raise ValueError(f"missing image: {row.get('sample_id')}")
        expected = files.get(relative)
        actual = sha256_file(path)
        if expected != actual:
            raise ValueError(f"image hash mismatch: {row.get('sample_id')}")
        loaded.append({**row, "_image_path": path, "_copied_image_sha256": actual})
    return loaded, {
        "manifest_sha256": sha256_file(manifest_path),
        "receipt_sha256": sha256_file(receipt_path),
    }


def run_suggestions(*, rows, bindings, output_path, config, resume=True,
                    limit=None, chat_fn=chat):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if resume and output_path.is_file():
        prior = json.loads(output_path.read_text(encoding="utf-8"))
        if (prior.get("manifest_sha256") != bindings["manifest_sha256"]
                or prior.get("receipt_sha256") != bindings["receipt_sha256"]):
            raise ValueError("existing suggestions belong to a different review pack")
        existing = {row["sample_id"]: row for row in prior.get("rows", [])
                    if row.get("status") == "passed"}
    records = []
    executed = 0
    for row in rows:
        if row["sample_id"] in existing:
            records.append(existing[row["sample_id"]])
            continue
        if limit is not None and executed >= limit:
            continue
        executed += 1
        image = row["_image_path"].read_bytes()
        started = time.perf_counter()
        try:
            result = chat_fn(config, INSTRUCTION, image,
                             response_schema=RESPONSE_SCHEMA)
            _validate_machine_result(result)
            record = {
                "sample_id": row["sample_id"], "cohort": row["cohort"],
                "status": "passed",
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "copied_image_sha256": row["_copied_image_sha256"],
                "suggestion": result,
                "candidate_only": True, "human_truth": False,
                "approved_for_training": False, "test24_used": False,
            }
        except Exception as error:
            record = {
                "sample_id": row["sample_id"], "cohort": row["cohort"],
                "status": "failed", "error_type": type(error).__name__,
                "error": str(error)[:500],
                "copied_image_sha256": row["_copied_image_sha256"],
                "candidate_only": True, "human_truth": False,
                "approved_for_training": False, "test24_used": False,
            }
        records.append(record)
        _write_output(output_path, rows, records, bindings, config)
    return _write_output(output_path, rows, records, bindings, config)


def _write_output(output_path, source_rows, records, bindings, config):
    records = [_with_shadow_route(record) for record in records]
    passed = [row for row in records if row.get("status") == "passed"]
    failed = [row for row in records if row.get("status") == "failed"]
    latencies = [row["latency_ms"] for row in passed]
    payload = {
        "schema_version": SCHEMA_VERSION,
        **bindings,
        "model": str(config.get("llm_model") or ""),
        "status": "completed" if len(passed) == len(source_rows) else "partial",
        "samples_total": len(source_rows), "samples_attempted": len(records),
        "samples_passed": len(passed), "samples_failed": len(failed),
        "latency_median_ms": (round(statistics.median(latencies), 3)
                              if latencies else None),
        "latency_p95_ms": (_percentile(latencies, 0.95) if latencies else None),
        "rows": records,
        "candidate_only": True, "human_truth": False,
        "approved_for_training": False, "automatic_truth_prefill": False,
        "test24_used": False,
    }
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(output_path)
    return payload


def _with_shadow_route(record):
    if record.get("status") != "passed" or not isinstance(record.get("suggestion"), dict):
        return record
    updated = dict(record)
    suggestion = record["suggestion"]
    assessments = suggestion.get("assessments") or {}
    uncertain = [label for label, evidence in assessments.items()
                 if evidence == "uncertain"]
    conflicts = [value for value in suggestion.get("class_conflicts") or []
                 if value not in {"none", "uncertain"}]
    updated["expert_shadow_route"] = evaluate_expert_shadow_route(
        {"llm_expert_shadow_routing_enabled": True},
        discovery={
            "uncertain_labels": uncertain,
            "unknown_candidates": suggestion.get("unknown_candidates") or [],
            "registry_conflicts": conflicts,
        },
        candidate_labels=[label for label, evidence in assessments.items()
                          if evidence != "absent"],
        local_model_latency_ms=record.get("latency_ms"),
        sam_result_reused=True)
    return updated


def _percentile(values, fraction):
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1,
                       int((len(ordered) - 1) * fraction + 0.999999)))
    return round(ordered[index], 3)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="labelagent-vlm8b")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    rows, bindings = load_inputs(args.manifest, args.receipt, args.image_root)
    result = run_suggestions(
        rows=rows, bindings=bindings, output_path=args.output,
        config={"llm_url": args.url.rstrip("/"), "llm_model": args.model,
                "llm_prompt_profile": "compact", "llm_max_tokens": 768,
                "llm_timeout_seconds": 90, "llm_busy_retries": 2},
        resume=not args.no_resume, limit=args.limit)
    print(json.dumps({key: result[key] for key in (
        "status", "samples_total", "samples_attempted", "samples_passed",
        "samples_failed", "latency_median_ms", "latency_p95_ms",
        "candidate_only", "human_truth", "approved_for_training",
        "test24_used")}, ensure_ascii=False, indent=2))
    return 0 if result["samples_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
