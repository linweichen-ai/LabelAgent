"""Read-only true-batch probe for exact visible-label discovery requests.

The tool captures the production instruction, schema and token budget without
calling the model, then submits two or three independent images to the bounded
batch endpoint.  It never loads the workbench or writes candidates, truth,
reference frames, training approval, or source state.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from source.agent import workflow


class _CapturedRequest(Exception):
    def __init__(self, config, instruction, response_schema):
        super().__init__("captured production discovery request")
        self.config = config
        self.instruction = instruction
        self.response_schema = response_schema


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture_contract(config: dict, image: bytes, labels: list[str]) -> dict:
    original = workflow.chat

    def capture(request_config, instruction, request_image, extra_images=(),
                response_schema=None):
        if request_image != image or extra_images:
            raise AssertionError("discovery request image binding changed")
        raise _CapturedRequest(
            dict(request_config), instruction, response_schema)

    workflow.chat = capture
    try:
        workflow.discover_visible_labels(config, image, labels)
    except _CapturedRequest as captured:
        return {
            "config": captured.config,
            "instruction": captured.instruction,
            "response_schema": captured.response_schema,
        }
    finally:
        workflow.chat = original
    raise AssertionError("production discovery did not issue a model request")


def data_url(payload: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(payload).decode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not 2 <= len(args.images) <= 3:
        raise ValueError("true-batch probe requires two or three images")

    artifact_root = (REPOSITORY_ROOT / "artifacts").resolve()
    output = args.output.resolve()
    state_path = args.state.resolve()
    images = [path.resolve() for path in args.images]
    if not output.is_relative_to(artifact_root):
        raise ValueError("output must be inside the repository artifacts directory")
    if output.exists():
        raise FileExistsError(output)
    for path in (state_path, *images):
        if not path.is_file():
            raise FileNotFoundError(path)

    state_hash = sha256(state_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    config = dict(state.get("config") or {})
    labels = list(state.get("labels") or [])
    if not config.get("llm_url") or not config.get("llm_model") or not labels:
        raise ValueError("state lacks primary VLM settings or labels")
    # This probe intentionally measures the already validated primary route.
    for key in tuple(config):
        if key.startswith("llm_fast_"):
            config.pop(key, None)

    payloads = [path.read_bytes() for path in images]
    contracts = [capture_contract(config, payload, labels)
                 for payload in payloads]
    schemas = [item["response_schema"] for item in contracts]
    max_tokens = [int(item["config"].get("llm_max_tokens") or 2048)
                  for item in contracts]
    if any(schema != schemas[0] for schema in schemas[1:]):
        raise AssertionError("same registry produced different discovery schemas")
    if len(set(max_tokens)) != 1:
        raise AssertionError("same registry produced different token budgets")

    requests = []
    for contract, image in zip(contracts, payloads):
        requests.append({
            "model": config["llm_model"],
            "temperature": 0,
            "max_tokens": max_tokens[0],
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": contract["instruction"]},
                {"type": "image_url", "image_url": {"url": data_url(image)}},
            ]}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "labelagent", "schema": schemas[0], "strict": True}},
        })

    endpoint = config["llm_url"].rstrip("/") + "/chat/batch_completions"
    request = urllib.request.Request(
        endpoint, data=json.dumps({"requests": requests}).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + config.get("llm_key", "")})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=1200) as response:
        response_body = json.load(response)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    outputs = response_body.get("results")
    if not isinstance(outputs, list) or len(outputs) != len(images):
        raise RuntimeError("batch endpoint returned a mismatched result count")

    rows = []
    for path, payload, item in zip(images, payloads, outputs):
        content = item["choices"][0]["message"]["content"]
        value = json.loads(content)
        assessments = value.get("assessments")
        protocol_valid = (
            isinstance(assessments, dict)
            and set(assessments) == set(labels)
            and isinstance(value.get("unknown_candidates"), list))
        rows.append({
            "image": str(path),
            "image_sha256": hashlib.sha256(payload).hexdigest(),
            "protocol_valid": protocol_valid,
            "assessments": assessments,
            "unknown_candidates": value.get("unknown_candidates"),
            "usage": item.get("usage"),
            "finish_reason": item["choices"][0].get("finish_reason"),
        })

    result = {
        "schema_version": "labelagent.batched-visible-discovery-probe.v1",
        "read_only": True,
        "batch_size": len(rows),
        "elapsed_ms": elapsed_ms,
        "endpoint": endpoint,
        "labels": labels,
        "max_tokens": max_tokens[0],
        "local_vlm": response_body.get("local_vlm"),
        "rows": rows,
        "summary": {
            "success_count": len(rows),
            "protocol_valid_count": sum(row["protocol_valid"] for row in rows),
            "candidate_written": False,
            "human_truth_modified": False,
            "training_data_modified": False,
            "source_state_modified": sha256(state_path) != state_hash,
        },
        "state": str(state_path),
        "state_sha256": state_hash,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "batch_size": len(rows),
        "elapsed_ms": elapsed_ms,
        "protocol_valid_count": result["summary"]["protocol_valid_count"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
