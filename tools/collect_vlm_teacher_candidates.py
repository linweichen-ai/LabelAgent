"""Collect immutable 8B teacher candidates for isolated 2B distillation.

Images may be streamed read-only over SSH so the Windows host does not need a
second dataset copy.  Outputs stay under ``artifacts`` and are always marked as
machine teacher candidates, never human truth or production-approved training
data.  Only the frozen train split is accepted; test24 is rejected by design.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import urllib.request

from PIL import Image, ImageOps, UnidentifiedImageError


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.agent import workflow
from tools.evaluate_visible_discovery_format_ablation import capture_contract


EXPECTED_TRAIN_SEQUENCES = {f"{index:03d}" for index in range(9)}
SAFE_REMOTE_PATH = re.compile(r"^/[A-Za-z0-9_./-]+$")


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_train_images(manifest: dict, limit: int | None = None) -> list[dict]:
    images = manifest.get("images")
    if not isinstance(images, list):
        raise ValueError("manifest images are missing")
    train = [dict(image) for image in images if image.get("split") == "train"]
    test = [image for image in images if image.get("split") == "test"]
    if len(train) != 72 or len(test) != 24:
        raise ValueError("frozen manifest must contain train72 and test24")
    if manifest.get("preannotations_are_ground_truth") is not False:
        raise ValueError("machine preannotations may not be treated as truth")
    names = [image.get("file_name") for image in train]
    hashes = [image.get("source_sha256") for image in train]
    sequences = {str(image.get("sequence")) for image in train}
    if len(set(names)) != 72 or len(set(hashes)) != 72:
        raise ValueError("train72 contains duplicate names or image hashes")
    if sequences != EXPECTED_TRAIN_SEQUENCES:
        raise ValueError("train72 sequence membership differs")
    test_names = {image.get("file_name") for image in test}
    test_hashes = {image.get("source_sha256") for image in test}
    if set(names) & test_names or set(hashes) & test_hashes:
        raise ValueError("test24 leakage detected")

    grouped = {sequence: [] for sequence in sorted(EXPECTED_TRAIN_SEQUENCES)}
    for image in train:
        grouped[str(image["sequence"])].append(image)
    for rows in grouped.values():
        rows.sort(key=lambda row: (str(row.get("frame_id")), str(row.get("camera")),
                                   str(row.get("file_name"))))
    ordered: list[dict] = []
    for position in range(max(len(rows) for rows in grouped.values())):
        for sequence in sorted(grouped):
            if position < len(grouped[sequence]):
                ordered.append(grouped[sequence][position])
    if limit is None:
        return ordered
    if not 1 <= limit <= 72:
        raise ValueError("limit must be between 1 and 72")
    return ordered[:limit]


def read_remote_image(host: str, path: str) -> bytes:
    if not host or not SAFE_REMOTE_PATH.fullmatch(path):
        raise ValueError("unsafe remote host or image path")
    ssh = shutil.which("ssh")
    if not ssh:
        raise RuntimeError("ssh executable is unavailable")
    result = subprocess.run(
        [ssh, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host,
         "cat", "--", path],
        stdin=subprocess.DEVNULL, capture_output=True, timeout=60,
        check=False,
    )
    if result.returncode:
        error = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"remote image read failed: {error[:300]}")
    return result.stdout


def production_png(source: bytes) -> bytes:
    try:
        with Image.open(io.BytesIO(source)) as original:
            if original.format not in {"JPEG", "PNG"}:
                raise ValueError("teacher source must be JPEG or PNG")
            image = ImageOps.exif_transpose(original).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("teacher source image is invalid") from error
    output = io.BytesIO()
    image.save(output, "PNG")
    return output.getvalue()


def canonical_teacher_target(discovery: dict, labels: list[str]) -> dict:
    assessments = discovery.get("assessments")
    if not isinstance(assessments, list) or len(assessments) != len(labels):
        raise ValueError("normalized teacher assessments are incomplete")
    by_label = {}
    for item in assessments:
        if not isinstance(item, dict) or item.get("label") not in labels:
            raise ValueError("normalized teacher assessment label differs")
        label = item["label"]
        if label in by_label:
            raise ValueError("normalized teacher assessment is duplicated")
        by_label[label] = {key: item[key] for key in ("evidence", "confidence", "reason")}
    if set(by_label) != set(labels):
        raise ValueError("normalized teacher labels differ from the registry")
    unknown = []
    for item in discovery.get("unknown_candidates") or []:
        unknown.append({key: item[key] for key in ("candidate_name", "evidence", "confidence")})
    return {
        "assessments": {label: by_label[label] for label in labels},
        "unknown_candidates": unknown,
    }


def teacher_candidate_record(
    manifest_image: dict,
    source: bytes,
    normalized_image: bytes,
    discovery: dict,
    trace: dict,
    labels: list[str],
    state_sha256: str,
    contract_sha256: str,
    model_config_sha256: str | None,
) -> dict:
    if trace.get("status") != "succeeded" or trace.get("service") != "GPT":
        raise ValueError("teacher trace did not succeed")
    output = trace.get("output") if isinstance(trace.get("output"), dict) else {}
    raw_text = output.get("raw_text")
    if not isinstance(raw_text, str) or not raw_text:
        raise ValueError("teacher trace has no raw structured response")
    if hashlib.sha256(source).hexdigest() != manifest_image.get("source_sha256"):
        raise ValueError("teacher source hash differs from frozen manifest")
    target = canonical_teacher_target(discovery, labels)
    flags = discovery.get("consistency_flags") or []
    conflicts = discovery.get("registry_conflicts") or []
    open_set_complete = discovery.get("open_set_scan_complete") is True
    raw_parsed = output.get("parsed")
    local = output.get("local_vlm") if isinstance(output.get("local_vlm"), dict) else {}
    stats = local.get("inference_stats") if isinstance(local.get("inference_stats"), dict) else {}
    return {
        "schema_version": "labelagent.vlm-teacher-candidate.v1",
        "candidate_kind": "machine_teacher_output",
        "human_truth": False,
        "training_authorized": False,
        "requires_acceptance_gate": True,
        "distillation_candidate": open_set_complete and not flags and not conflicts,
        "split": "train",
        "image": {
            "image_id": manifest_image.get("image_id"),
            "sequence": manifest_image.get("sequence"),
            "frame_id": manifest_image.get("frame_id"),
            "camera": manifest_image.get("camera"),
            "file_name": manifest_image.get("file_name"),
            "source_path": manifest_image.get("source_path"),
            "source_sha256": hashlib.sha256(source).hexdigest(),
            "production_png_sha256": hashlib.sha256(normalized_image).hexdigest(),
        },
        "teacher": {
            "model": trace.get("input", {}).get("model"),
            "model_route": output.get("model_route"),
            "model_config_sha256": model_config_sha256,
            "finish_reason": output.get("finish_reason"),
            "busy_retry_count": output.get("busy_retry_count", 0),
            "raw_text_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
            "raw_parsed": raw_parsed,
            "raw_parsed_sha256": json_sha256(raw_parsed),
            "normalized_discovery_sha256": json_sha256(discovery),
            "cache_hit": local.get("cache_hit"),
            "coalesced": local.get("coalesced"),
            "prompt_tokens": stats.get("prompt_tokens"),
            "completion_tokens": stats.get("completion_tokens"),
            "generation_seconds": stats.get("generation_seconds"),
        },
        "target": target,
        "target_sha256": json_sha256(target),
        "review": {
            "open_set_scan_complete": open_set_complete,
            "consistency_flags": flags,
            "registry_conflicts": conflicts,
        },
        "bindings": {
            "state_sha256": state_sha256,
            "contract_sha256": contract_sha256,
        },
    }


def read_health(url: str | None) -> dict | None:
    if not url:
        return None
    with urllib.request.urlopen(url, timeout=10) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError("VLM health is not an object")
    return value


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


@contextmanager
def trace_buffer():
    events: dict[str, dict] = {}

    def capture(event):
        events[event["id"]] = dict(event)

    with workflow.trace_calls(capture):
        yield events


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--health-url")
    parser.add_argument("--expected-model-config-sha256")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve()
    artifact_root = (ROOT / "artifacts").resolve()
    if not output.is_relative_to(artifact_root):
        raise ValueError("output must be inside repository artifacts")
    if output.exists() and not args.resume:
        raise FileExistsError(output)
    manifest_path = args.manifest.resolve()
    state_path = args.state.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    state_sha = file_sha256(state_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    labels = list(state.get("labels") or [])
    config = dict(state.get("config") or {})
    if not labels or not config.get("llm_url") or not config.get("llm_model"):
        raise ValueError("state lacks labels or primary VLM settings")
    for key in tuple(config):
        if key.startswith("llm_fast_"):
            config.pop(key, None)
    images = frozen_train_images(manifest, args.limit)
    health_before = read_health(args.health_url)
    model_config_sha = (health_before or {}).get("model_config_sha256")
    if health_before is not None:
        if health_before.get("model_loaded") is not True:
            raise RuntimeError("teacher VLM is not loaded")
        if (args.expected_model_config_sha256
                and model_config_sha != args.expected_model_config_sha256):
            raise RuntimeError("teacher model configuration differs")

    first_source = read_remote_image(args.remote_host, images[0]["source_path"])
    if hashlib.sha256(first_source).hexdigest() != images[0]["source_sha256"]:
        raise ValueError("first teacher source differs from frozen manifest")
    first_image = production_png(first_source)
    contract = capture_contract(config, first_image, labels)
    contract_record = {
        "schema_version": "labelagent.vlm-teacher-contract.v1",
        "model": contract["config"].get("llm_model"),
        "prompt_profile": contract["config"].get("llm_prompt_profile"),
        "max_tokens": contract["config"].get("llm_max_tokens"),
        "labels": labels,
        "instruction": contract["instruction"],
        "response_schema": contract["response_schema"],
    }
    contract_sha = json_sha256(contract_record)
    contract_path = output / "contract.json"
    if contract_path.exists():
        existing_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        existing_sha = existing_contract.pop("contract_sha256", None)
        if existing_sha != contract_sha or json_sha256(existing_contract) != contract_sha:
            raise RuntimeError("existing collection contract differs")
    else:
        write_json_atomic(contract_path, contract_record | {"contract_sha256": contract_sha})

    rows_by_position: dict[int, dict] = {}
    resumed_rows = 0
    new_rows = 0
    started = time.perf_counter()
    work_items = list(enumerate(images, 1))
    if args.reverse:
        work_items.reverse()
    for position, manifest_image in work_items:
        row_path = output / "rows" / f"{position:03d}-{manifest_image['file_name']}.json"
        if args.resume and row_path.exists():
            record = json.loads(row_path.read_text(encoding="utf-8"))
            if (record.get("image", {}).get("file_name") != manifest_image["file_name"]
                    or record.get("image", {}).get("source_sha256") != manifest_image["source_sha256"]
                    or record.get("bindings", {}).get("state_sha256") != state_sha
                    or record.get("bindings", {}).get("contract_sha256") != contract_sha
                    or record.get("teacher", {}).get("model_config_sha256") != model_config_sha):
                raise RuntimeError("existing teacher row binding differs")
            rows_by_position[position] = record
            resumed_rows += 1
            print(json.dumps({
                "progress": f"{len(rows_by_position)}/{len(images)}",
                "file_name": manifest_image["file_name"],
                "resumed": True,
                "distillation_candidate": record["distillation_candidate"],
            }, ensure_ascii=False), flush=True)
            continue
        source = first_source if position == 1 else read_remote_image(
            args.remote_host, manifest_image["source_path"])
        normalized = first_image if position == 1 else production_png(source)
        current_contract = capture_contract(config, normalized, labels)
        current_record = {
            "schema_version": contract_record["schema_version"],
            "model": current_contract["config"].get("llm_model"),
            "prompt_profile": current_contract["config"].get("llm_prompt_profile"),
            "max_tokens": current_contract["config"].get("llm_max_tokens"),
            "labels": labels,
            "instruction": current_contract["instruction"],
            "response_schema": current_contract["response_schema"],
        }
        if json_sha256(current_record) != contract_sha:
            raise RuntimeError("teacher request contract changed within collection")
        with trace_buffer() as events:
            discovery = workflow.discover_visible_labels(config, normalized, labels)
        traces = [event for event in events.values() if event.get("service") == "GPT"]
        if not traces:
            raise RuntimeError("teacher inference emitted no GPT trace")
        record = teacher_candidate_record(
            manifest_image, source, normalized, discovery, traces[-1], labels,
            state_sha, contract_sha, model_config_sha)
        write_json_atomic(row_path, record)
        rows_by_position[position] = record
        new_rows += 1
        print(json.dumps({
            "progress": f"{len(rows_by_position)}/{len(images)}",
            "file_name": manifest_image["file_name"],
            "distillation_candidate": record["distillation_candidate"],
            "completion_tokens": record["teacher"]["completion_tokens"],
            "generation_seconds": record["teacher"]["generation_seconds"],
        }, ensure_ascii=False), flush=True)

    rows = [rows_by_position[position] for position in sorted(rows_by_position)]
    if len(rows) != len(images):
        raise RuntimeError("teacher collection did not cover every selected image")
    health_after = read_health(args.health_url)
    summary = {
        "schema_version": "labelagent.vlm-teacher-collection.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "teacher_candidates_collected",
        "candidate_kind": "machine_teacher_output",
        "human_truth": False,
        "training_authorized": False,
        "requires_acceptance_gate": True,
        "test24_included": False,
        "images": len(rows),
        "sequences": sorted({row["image"]["sequence"] for row in rows}),
        "distillation_candidates": sum(row["distillation_candidate"] for row in rows),
        "review_required": sum(not row["distillation_candidate"] for row in rows),
        "cache_hits": sum(row["teacher"].get("cache_hit") is True for row in rows),
        "coalesced_requests": sum(row["teacher"].get("coalesced") is True for row in rows),
        "source_hashes_verified": len(rows),
        "contract_sha256": contract_sha,
        "state_sha256": state_sha,
        "state_unchanged": file_sha256(state_path) == state_sha,
        "manifest_sha256": file_sha256(manifest_path),
        "teacher_model_config_sha256": model_config_sha,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "resumed_rows": resumed_rows,
        "new_rows": new_rows,
        "processing_order": "reverse" if args.reverse else "forward",
        "health_before": {
            "model_loaded": (health_before or {}).get("model_loaded"),
            "exact_requests_inflight": (health_before or {}).get("exact_requests_inflight"),
        },
        "health_after": {
            "model_loaded": (health_after or {}).get("model_loaded"),
            "exact_requests_inflight": (health_after or {}).get("exact_requests_inflight"),
        },
    }
    write_json_atomic(output / "summary.json", summary)
    print("RESULT=" + json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
