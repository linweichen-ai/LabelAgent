"""Verify one cold production discovery and its exact cached replay."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluate_visible_discovery_format_ablation import (
    capture_contract, file_sha256, request_json)


def health_url(endpoint: str) -> str:
    marker = "/v1/chat/completions"
    if not endpoint.endswith(marker):
        raise ValueError("expected an OpenAI-compatible chat completion endpoint")
    return endpoint[:-len(marker)] + "/health"


def record(response: dict, wall_ms: float) -> dict:
    local = response.get("local_vlm") or {}
    content = response["choices"][0]["message"]["content"]
    return {
        "wall_ms": wall_ms,
        "cache_hit": local.get("cache_hit"),
        "coalesced": local.get("coalesced"),
        "cache_entries": local.get("response_cache_entries"),
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "usage": response.get("usage"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "artifacts").resolve()):
        raise ValueError("output must be inside repository artifacts")
    if output.exists():
        raise FileExistsError(output)
    image_path = args.image.resolve()
    state_path = args.state.resolve()
    for path in (image_path, state_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    state_before = file_sha256(state_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    config = dict(state.get("config") or {})
    for key in tuple(config):
        if key.startswith("llm_fast_"):
            config.pop(key, None)
    labels = list(state.get("labels") or [])
    image = image_path.read_bytes()
    contract = capture_contract(config, image, labels)
    request_config = contract["config"]
    endpoint = request_config["llm_url"].rstrip("/") + "/chat/completions"
    body = {
        "model": request_config["llm_model"],
        "temperature": 0,
        "max_tokens": int(request_config.get("llm_max_tokens") or 768),
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": contract["instruction"]},
            {"type": "image_url", "image_url": {"url":
                "data:image/jpeg;base64," + base64.b64encode(image).decode()}},
        ]}],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "labelagent", "schema": contract["response_schema"],
            "strict": True}},
    }
    first_response, first_ms = request_json(
        endpoint, body, request_config.get("llm_key", ""))
    second_response, second_ms = request_json(
        endpoint, body, request_config.get("llm_key", ""))
    with urllib.request.urlopen(health_url(endpoint), timeout=10) as response:
        health = json.load(response)
    first = record(first_response, first_ms)
    second = record(second_response, second_ms)
    result = {
        "schema_version": "labelagent.exact-replay-verification.v1",
        "read_only": True,
        "image": str(image_path),
        "image_sha256": hashlib.sha256(image).hexdigest(),
        "first": first,
        "second": second,
        "acceptance": {
            "first_was_cold": first["cache_hit"] is False,
            "second_was_exact_hit": second["cache_hit"] is True,
            "content_sha256_equal": (
                first["content_sha256"] == second["content_sha256"]),
            "usage_equal": first["usage"] == second["usage"],
            "capacity": health.get("response_cache_capacity"),
            "source_state_modified": file_sha256(state_path) != state_before,
            "candidate_written": False,
            "human_truth_modified": False,
            "training_data_modified": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print("RESULT=" + json.dumps(result["acceptance"], ensure_ascii=False))
    return 0 if all([
        result["acceptance"]["first_was_cold"],
        result["acceptance"]["second_was_exact_hit"],
        result["acceptance"]["content_sha256_equal"],
        result["acceptance"]["usage_equal"],
        result["acceptance"]["capacity"] == 64,
        not result["acceptance"]["source_state_modified"],
    ]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
