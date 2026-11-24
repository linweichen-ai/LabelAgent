"""Run an authorized, resumable external-expert shadow on a frozen queue."""
import argparse
import hashlib
import io
import json
import statistics
import sys
import time
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.agent.workflow import chat_external_expert
from tools.run_local_qwen_mixed_hard_suggestions import (
    INSTRUCTION, RESPONSE_SCHEMA, _validate_machine_result,
)


QUEUE_SCHEMA = "labelagent.mixed-hard-expert-shadow-queue.v1"
AUTH_SCHEMA = "labelagent.external-upload-authorization.v1"
AUTH_SCHEMA_V2 = "labelagent.external-upload-authorization.v2"
AUTH_SCHEMA_V3 = "labelagent.external-upload-authorization.v3"
RESULT_SCHEMA = "labelagent.mixed-hard-external-shadow.v1"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize_image_variant(value):
    if not isinstance(value, dict):
        raise ValueError("image variant must be an object")
    name, mode = value.get("name"), value.get("mode")
    if not isinstance(name, str) or not name or len(name) > 64:
        raise ValueError("image variant name is invalid")
    if mode == "source":
        expected_width = value.get("expected_width")
        expected_height = value.get("expected_height")
        if expected_width is None and expected_height is None:
            return {"name": name, "mode": "source"}
        if (type(expected_width) is not int or type(expected_height) is not int
                or not 128 <= expected_width <= 2048
                or not 128 <= expected_height <= 2048):
            raise ValueError("source image variant dimensions are invalid")
        return {"name": name, "mode": "source",
                "expected_width": expected_width,
                "expected_height": expected_height}
    if mode != "resize_lanczos_jpeg":
        raise ValueError("image variant mode is invalid")
    width, height, quality = (value.get("width"), value.get("height"),
                              value.get("jpeg_quality"))
    if (type(width) is not int or type(height) is not int
            or not 128 <= width <= 2048 or not 128 <= height <= 2048
            or type(quality) is not int or not 70 <= quality <= 100):
        raise ValueError("resized image variant parameters are invalid")
    return {"name": name, "mode": mode, "width": width, "height": height,
            "jpeg_quality": quality}


def prepare_upload_image(source_bytes, variant):
    variant = normalize_image_variant(variant)
    if variant["mode"] == "source":
        metadata = {
            "name": variant["name"], "mode": "source",
            "upload_image_sha256": hashlib.sha256(source_bytes).hexdigest(),
        }
        if "expected_width" in variant:
            with Image.open(io.BytesIO(source_bytes)) as source:
                source_size = source.size
            if source_size != (variant["expected_width"],
                               variant["expected_height"]):
                raise ValueError("source image dimensions differ from authorization")
            metadata.update(source_width=source_size[0],
                            source_height=source_size[1])
        return source_bytes, metadata
    with Image.open(io.BytesIO(source_bytes)) as source:
        source_size = list(source.size)
        resized = source.convert("RGB").resize(
            (variant["width"], variant["height"]), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    resized.save(buffer, "JPEG", quality=variant["jpeg_quality"],
                 optimize=False, progressive=False)
    upload = buffer.getvalue()
    return upload, {
        **variant, "source_width": source_size[0],
        "source_height": source_size[1],
        "upload_image_sha256": hashlib.sha256(upload).hexdigest(),
    }


def load_authorized_inputs(queue_path, manifest_path, receipt_path, image_root,
                           authorization_path, image_variant_name="source",
                           authorization_run_name=None):
    queue_path, manifest_path = Path(queue_path), Path(manifest_path)
    receipt_path, authorization_path = Path(receipt_path), Path(authorization_path)
    image_root = Path(image_root)
    if not authorization_path.is_file():
        raise ValueError("matching explicit external-upload authorization is required")
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    queue_sha = sha256_file(queue_path)
    samples = queue.get("samples") if isinstance(queue, dict) else None
    if (queue.get("schema_version") != QUEUE_SCHEMA
            or not isinstance(samples, list) or not samples
            or queue.get("sample_count") != len(samples)
            or queue.get("candidate_only") is not True
            or queue.get("human_truth") is not False
            or queue.get("approved_for_training") is not False
            or queue.get("test24_used") is not False
            or queue.get("images_copied") is not False
            or queue.get("image_paths_included") is not False):
        raise ValueError("expert queue identity or safety boundary differs")
    auth_schema = authorization.get("schema_version")
    if (auth_schema not in {AUTH_SCHEMA, AUTH_SCHEMA_V2, AUTH_SCHEMA_V3}
            or authorization.get("authorized") is not True
            or not isinstance(authorization.get("authorized_by"), str)
            or not authorization["authorized_by"].strip()
            or authorization.get("candidate_only") is not True
            or authorization.get("human_truth") is not False
            or authorization.get("approved_for_training") is not False
            or authorization.get("test24_used") is not False):
        raise ValueError("matching explicit external-upload authorization is required")
    if auth_schema == AUTH_SCHEMA:
        if (authorization.get("queue_sha256") != queue_sha
                or authorization.get("authorized_sample_count") != len(samples)
                or authorization.get("scope") != "mixed_hard_external_shadow"
                or image_variant_name != "source"):
            raise ValueError("matching explicit external-upload authorization is required")
        image_variant = {"name": "source", "mode": "source"}
        authorization_run = None
    elif auth_schema == AUTH_SCHEMA_V2:
        if (authorization.get("queue_sha256") != queue_sha
                or authorization.get("authorized_sample_count") != len(samples)
                or authorization.get("scope")
                != "mixed_hard_external_resolution_ab_shadow"):
            raise ValueError("matching explicit external-upload authorization is required")
        variants = authorization.get("authorized_image_variants")
        if (not isinstance(variants, list) or not 2 <= len(variants) <= 4):
            raise ValueError("matching explicit external-upload authorization is required")
        try:
            normalized_variants = [normalize_image_variant(item) for item in variants]
        except ValueError as error:
            raise ValueError(
                "matching explicit external-upload authorization is required") from error
        if len({item["name"] for item in normalized_variants}) != len(normalized_variants):
            raise ValueError("matching explicit external-upload authorization is required")
        if (authorization.get("authorized_upload_count")
                != len(samples) * len(normalized_variants)
                or authorization.get("max_external_calls")
                != len(samples) * len(normalized_variants)):
            raise ValueError("matching explicit external-upload authorization is required")
        matching = [item for item in normalized_variants
                    if item["name"] == image_variant_name]
        if len(matching) != 1:
            raise ValueError("matching explicit external-upload authorization is required")
        image_variant = matching[0]
        authorization_run = None
    else:
        runs = authorization.get("authorized_runs")
        if (authorization.get("scope") != "mixed_hard_combined_external_shadow"
                or authorization.get("authorized_unique_source_samples") != 34
                or authorization.get("authorized_upload_count") != 51
                or authorization.get("max_external_calls") != 51
                or not isinstance(runs, list) or len(runs) != 2
                or not isinstance(authorization_run_name, str)):
            raise ValueError("matching explicit external-upload authorization is required")
        normalized_runs = []
        try:
            for run in runs:
                variant = normalize_image_variant(run.get("image_variant"))
                if (not isinstance(run.get("name"), str)
                        or not isinstance(run.get("queue_sha256"), str)
                        or type(run.get("sample_count")) is not int
                        or run.get("max_external_calls") != run.get("sample_count")):
                    raise ValueError("invalid combined run")
                normalized_runs.append({**run, "image_variant": variant})
        except (AttributeError, ValueError) as error:
            raise ValueError(
                "matching explicit external-upload authorization is required") from error
        if (len({run["name"] for run in normalized_runs}) != len(normalized_runs)
                or sum(run["max_external_calls"] for run in normalized_runs) != 51):
            raise ValueError("matching explicit external-upload authorization is required")
        runs_by_name = {run["name"]: run for run in normalized_runs}
        if set(runs_by_name) != {"all34_source_640x512",
                                "hybrid17_resize_512x410"}:
            raise ValueError("matching explicit external-upload authorization is required")
        source_run = runs_by_name["all34_source_640x512"]
        resized_run = runs_by_name["hybrid17_resize_512x410"]
        if (source_run["sample_count"] != 34
                or source_run["max_external_calls"] != 34
                or source_run["image_variant"] != {
                    "name": "source_640x512", "mode": "source",
                    "expected_width": 640, "expected_height": 512}
                or resized_run["sample_count"] != 17
                or resized_run["max_external_calls"] != 17
                or resized_run["image_variant"] != {
                    "name": "resize_512x410", "mode": "resize_lanczos_jpeg",
                    "width": 512, "height": 410, "jpeg_quality": 90}):
            raise ValueError("matching explicit external-upload authorization is required")
        matching_runs = [run for run in normalized_runs
                         if run["name"] == authorization_run_name]
        if len(matching_runs) != 1:
            raise ValueError("matching explicit external-upload authorization is required")
        authorization_run = matching_runs[0]
        if (authorization_run["queue_sha256"] != queue_sha
                or authorization_run["sample_count"] != len(samples)
                or (image_variant_name != "source"
                    and image_variant_name
                    != authorization_run["image_variant"]["name"])):
            raise ValueError("matching explicit external-upload authorization is required")
        image_variant = authorization_run["image_variant"]
    rows = manifest.get("rows") if isinstance(manifest, dict) else None
    files = receipt.get("files") if isinstance(receipt, dict) else None
    if (manifest.get("summary", {}).get("test24_intersection_count") != 0
            or not isinstance(rows, list) or not isinstance(files, dict)
            or receipt.get("candidate_only") is not True
            or receipt.get("human_truth") is not False
            or receipt.get("approved_for_training") is not False):
        raise ValueError("review workspace safety boundary differs")
    manifest_rows = {row.get("sample_id"): row for row in rows}
    if len(manifest_rows) != len(rows):
        raise ValueError("review manifest contains duplicate samples")
    loaded, seen = [], set()
    for sample in samples:
        sample_id = sample.get("sample_id")
        if sample_id in seen or sample_id not in manifest_rows:
            raise ValueError("queue sample is duplicated or missing from the manifest")
        seen.add(sample_id)
        manifest_row = manifest_rows[sample_id]
        relative = str(manifest_row.get("raw_image") or "")
        if not relative.startswith("raw/") or ".." in Path(relative).parts:
            raise ValueError("unsafe review image path")
        path = image_root / Path(relative)
        actual = sha256_file(path) if path.is_file() else None
        if (actual != sample.get("source_image_sha256")
                or files.get(relative) != actual):
            raise ValueError(f"authorized image hash mismatch: {sample_id}")
        loaded.append({**sample, "_image_path": path})
    return loaded, {
        "queue_sha256": queue_sha,
        "authorization_sha256": sha256_file(authorization_path),
        "manifest_sha256": sha256_file(manifest_path),
        "receipt_sha256": sha256_file(receipt_path),
        "authorized_by": authorization["authorized_by"].strip(),
        "authorization_schema": auth_schema,
        "image_variant": image_variant,
        "authorized_upload_count": authorization.get(
            "authorized_upload_count", len(samples)),
        "authorization_run": authorization_run,
    }


def run_shadow(*, samples, bindings, output_path, config,
               chat_fn=chat_external_expert):
    if config.get("llm_expert_retry_count", 0) != 0:
        raise ValueError("authorized shadow forbids automatic API retries")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = {}
    if output_path.is_file():
        prior = json.loads(output_path.read_text(encoding="utf-8"))
        if any(prior.get(key) != value for key, value in bindings.items()
               if key != "authorized_by"):
            raise ValueError("existing external results belong to a different authorization")
        prior_rows = prior.get("rows")
        expected_ids = {sample["sample_id"] for sample in samples}
        if (not isinstance(prior_rows, list)
                or len({row.get("sample_id") for row in prior_rows})
                != len(prior_rows)
                or any(row.get("sample_id") not in expected_ids
                       or row.get("status") not in {"passed", "failed"}
                       or row.get("external_attempts") != 1
                       for row in prior_rows)):
            raise ValueError("existing external result attempt ledger differs")
        completed = {row["sample_id"]: row for row in prior_rows}
    records = []
    for sample in samples:
        if sample["sample_id"] in completed:
            records.append(completed[sample["sample_id"]])
            continue
        source_image = sample["_image_path"].read_bytes()
        image, upload_image = prepare_upload_image(
            source_image, bindings.get("image_variant") or
            {"name": "source", "mode": "source"})
        started = time.perf_counter()
        expert_metadata = {}
        try:
            result = chat_fn(config, INSTRUCTION, image,
                             response_schema=RESPONSE_SCHEMA,
                             metadata_out=expert_metadata)
            _validate_machine_result(result)
            record = {
                "sample_id": sample["sample_id"], "cohort": sample["cohort"],
                "source_image_sha256": sample["source_image_sha256"],
                "upload_image": upload_image,
                "status": "passed",
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "expert_result": result,
                "expert_metadata": expert_metadata,
                "external_attempts": 1,
                "candidate_only": True, "human_truth": False,
                "approved_for_training": False, "test24_used": False,
            }
        except Exception as error:
            record = {
                "sample_id": sample["sample_id"], "cohort": sample["cohort"],
                "source_image_sha256": sample["source_image_sha256"],
                "status": "failed", "error_type": type(error).__name__,
                "error": str(error)[:500],
                "external_attempts": 1,
                "candidate_only": True, "human_truth": False,
                "approved_for_training": False, "test24_used": False,
            }
        records.append(record)
        _write_output(output_path, samples, records, bindings, config)
    return _write_output(output_path, samples, records, bindings, config)


def _write_output(output_path, samples, records, bindings, config):
    passed = [row for row in records if row.get("status") == "passed"]
    failed = [row for row in records if row.get("status") == "failed"]
    latencies = [row["latency_ms"] for row in passed]
    usage_rows = []
    for row in passed:
        usage = (row.get("expert_metadata", {}).get("external_vlm", {})
                 .get("usage", {}))
        if all(type(usage.get(key)) is int and usage[key] >= 0 for key in
               ("prompt_tokens", "completion_tokens", "total_tokens")):
            usage_rows.append(usage)
    usage_summary = {
        "samples_with_usage": len(usage_rows),
        "samples_missing_usage": len(passed) - len(usage_rows),
        "complete": len(usage_rows) == len(passed),
        "prompt_tokens": sum(row["prompt_tokens"] for row in usage_rows),
        "completion_tokens": sum(row["completion_tokens"] for row in usage_rows),
        "total_tokens": sum(row["total_tokens"] for row in usage_rows),
    }
    result = {
        "schema_version": RESULT_SCHEMA, **bindings,
        "model": config.get("llm_expert_model"),
        "status": "completed" if len(passed) == len(samples) else "partial",
        "samples_total": len(samples), "samples_attempted": len(records),
        "samples_passed": len(passed), "samples_failed": len(failed),
        "external_attempts": sum(
            row.get("external_attempts", 0) for row in records),
        "automatic_retry_enabled": False,
        "latency_median_ms": (round(statistics.median(latencies), 3)
                              if latencies else None),
        "external_usage": usage_summary,
        "rows": records, "candidate_only": True, "human_truth": False,
        "approved_for_training": False, "automatic_truth_write": False,
        "test24_used": False,
    }
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(output_path)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--image-variant", default="source")
    parser.add_argument("--authorization-run")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--key-env", required=True)
    args = parser.parse_args(argv)
    samples, bindings = load_authorized_inputs(
        args.queue, args.manifest, args.receipt, args.image_root,
        args.authorization, args.image_variant, args.authorization_run)
    result = run_shadow(
        samples=samples, bindings=bindings, output_path=args.output,
        config={"llm_expert_enabled": True,
                "llm_expert_provider": "openai_chat",
                "llm_expert_url": args.url, "llm_expert_model": args.model,
                "llm_expert_key_env": args.key_env,
                "llm_expert_timeout_seconds": 120,
                "llm_expert_retry_count": 0})
    print(json.dumps({key: result[key] for key in (
        "status", "samples_total", "samples_attempted", "samples_passed",
        "samples_failed", "latency_median_ms", "candidate_only",
        "human_truth", "approved_for_training", "test24_used")},
        ensure_ascii=False, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
