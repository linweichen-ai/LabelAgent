"""Compare production visible-label discovery with and without JSON constraints.

The request image, instruction, model, token budget and sampling settings stay
identical.  Results are diagnostic only and are written under ``artifacts``;
they never enter the workbench cache, candidates, truth or training data.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.agent import workflow


class CapturedRequest(Exception):
    def __init__(self, config, instruction, response_schema):
        super().__init__("captured visible-label discovery request")
        self.config = dict(config)
        self.instruction = instruction
        self.response_schema = response_schema


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture_contract(config: dict, image: bytes, labels: list[str]) -> dict:
    original = workflow.chat

    def capture(request_config, instruction, request_image, extra_images=(),
                response_schema=None):
        if request_image != image or extra_images:
            raise AssertionError("discovery image binding changed")
        raise CapturedRequest(request_config, instruction, response_schema)

    workflow.chat = capture
    try:
        workflow.discover_visible_labels(config, image, labels)
    except CapturedRequest as captured:
        return {
            "config": captured.config,
            "instruction": captured.instruction,
            "response_schema": captured.response_schema,
        }
    finally:
        workflow.chat = original
    raise AssertionError("production discovery did not issue a model request")


def request_json(url: str, body: dict, key: str) -> tuple[dict, float]:
    request = urllib.request.Request(
        url, data=json.dumps(body, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + key})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=1200) as response:
        result = json.load(response)
    return result, round((time.perf_counter() - started) * 1000, 3)


def parse_content(response: dict) -> tuple[dict | None, bool]:
    content = response["choices"][0]["message"]["content"]
    try:
        value = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return None, False
    return value, isinstance(value, dict)


def known_states(value: dict | None, labels: list[str]) -> dict | None:
    if not isinstance(value, dict):
        return None
    assessments = value.get("assessments")
    if not isinstance(assessments, dict) or set(assessments) != set(labels):
        return None
    result = {}
    for label in labels:
        item = assessments[label]
        if not isinstance(item, dict) or item.get("evidence") not in {
                "present", "absent", "uncertain"}:
            return None
        result[label] = item["evidence"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    output = args.output.resolve()
    artifact_root = (ROOT / "artifacts").resolve()
    if not output.is_relative_to(artifact_root):
        raise ValueError("output must be inside repository artifacts")
    if output.exists():
        raise FileExistsError(output)
    state_path = args.state.resolve()
    images = [path.resolve() for path in args.images]
    for path in [state_path, *images]:
        if not path.is_file():
            raise FileNotFoundError(path)

    state_before = file_sha256(state_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    config = dict(state.get("config") or {})
    labels = list(state.get("labels") or [])
    if not config.get("llm_url") or not config.get("llm_model") or not labels:
        raise ValueError("state lacks primary model settings or labels")
    for key in tuple(config):
        if key.startswith("llm_fast_"):
            config.pop(key, None)

    payloads = [path.read_bytes() for path in images]
    contracts = [capture_contract(config, payload, labels)
                 for payload in payloads]
    endpoint = config["llm_url"].rstrip("/") + "/chat/completions"
    rows = []
    for index, (path, image, contract) in enumerate(
            zip(images, payloads, contracts)):
        request_config = contract["config"]
        body = {
            "model": request_config["llm_model"],
            "temperature": 0,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": contract["instruction"]},
                {"type": "image_url", "image_url": {"url":
                    "data:image/jpeg;base64," + base64.b64encode(image).decode()}},
            ]}],
        }
        if request_config.get("llm_prompt_profile") == "compact":
            body["max_tokens"] = int(
                request_config.get("llm_max_tokens") or 768)
        structured = dict(body)
        structured["response_format"] = {
            "type": "json_schema", "json_schema": {
                "name": "labelagent",
                "schema": contract["response_schema"], "strict": True}}
        variants = [("structured", structured), ("unconstrained", body)]
        if index % 2:
            variants.reverse()
        outputs = {}
        for name, request_body in variants:
            response, wall_ms = request_json(
                endpoint, request_body, request_config.get("llm_key", ""))
            value, json_object = parse_content(response)
            local = response.get("local_vlm") or {}
            stats = local.get("inference_stats") or {}
            outputs[name] = {
                "wall_ms": wall_ms,
                "finish_reason": response["choices"][0].get("finish_reason"),
                "json_object": json_object,
                "protocol_known_states": known_states(value, labels),
                "value": value,
                "content_sha256": hashlib.sha256(
                    response["choices"][0]["message"]["content"].encode()).hexdigest(),
                "cache_hit": local.get("cache_hit"),
                "structured_output": stats.get("structured_output"),
                "generation_seconds": stats.get("generation_seconds"),
                "completion_tokens": stats.get("completion_tokens"),
                "prompt_tokens": stats.get("prompt_tokens"),
            }
        baseline = outputs["structured"]
        candidate = outputs["unconstrained"]
        rows.append({
            "image": str(path),
            "image_sha256": hashlib.sha256(image).hexdigest(),
            "order": [name for name, _ in variants],
            "structured": baseline,
            "unconstrained": candidate,
            "full_json_equal": (baseline["value"] is not None
                                and baseline["value"] == candidate["value"]),
            "known_states_equal": (baseline["protocol_known_states"] is not None
                                   and baseline["protocol_known_states"]
                                   == candidate["protocol_known_states"]),
        })
        print(json.dumps({
            "progress": f"{index + 1}/{len(images)}",
            "image": str(path),
            "full_json_equal": rows[-1]["full_json_equal"],
            "known_states_equal": rows[-1]["known_states_equal"],
            "structured_ms": baseline["wall_ms"],
            "unconstrained_ms": candidate["wall_ms"],
        }, ensure_ascii=False), flush=True)

    structured_ms = [row["structured"]["wall_ms"] for row in rows]
    unconstrained_ms = [row["unconstrained"]["wall_ms"] for row in rows]
    result = {
        "schema_version": "labelagent.visible-discovery-format-ablation.v1",
        "read_only": True,
        "difference": "response_format omitted only",
        "labels": labels,
        "rows": rows,
        "summary": {
            "image_count": len(rows),
            "structured_total_ms": round(sum(structured_ms), 3),
            "unconstrained_total_ms": round(sum(unconstrained_ms), 3),
            "structured_median_ms": round(statistics.median(structured_ms), 3),
            "unconstrained_median_ms": round(statistics.median(unconstrained_ms), 3),
            "strict_json_count": sum(
                row["unconstrained"]["json_object"] for row in rows),
            "full_json_equal_count": sum(row["full_json_equal"] for row in rows),
            "known_states_equal_count": sum(
                row["known_states_equal"] for row in rows),
            "cache_hit_count": sum(
                item[variant]["cache_hit"] is True
                for item in rows for variant in ("structured", "unconstrained")),
            "candidate_written": False,
            "human_truth_modified": False,
            "training_data_modified": False,
            "source_state_modified": file_sha256(state_path) != state_before,
        },
        "state": str(state_path),
        "state_sha256": state_before,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print("RESULT=" + json.dumps(result["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
