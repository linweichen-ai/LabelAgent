"""Verify two concurrent exact production requests share one GPU inference."""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import threading


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluate_visible_discovery_format_ablation import (
    capture_contract, file_sha256, request_json)
from tools.verify_vlm_exact_replay import (
    cache_stat_delta, read_health, record)


def coalescing_acceptance(records: list[dict], stats_delta: dict) -> dict:
    return {
        "two_successful_responses": len(records) == 2,
        "one_leader": sum(item.get("cache_hit") is False
                          for item in records) == 1,
        "one_coalesced_follower": sum(
            item.get("cache_hit") is True and item.get("coalesced") is True
            for item in records) == 1,
        "content_sha256_equal": (len(records) == 2 and
            records[0].get("content_sha256") == records[1].get("content_sha256")),
        "usage_equal": (len(records) == 2 and
            records[0].get("usage") == records[1].get("usage")),
        "expected_cache_stat_delta": all([
            stats_delta.get("misses") == 2,
            stats_delta.get("stores") == 1,
            stats_delta.get("coalesced") == 1,
            stats_delta.get("hits") == 0,
            stats_delta.get("busy_rejections") == 0,
            stats_delta.get("coalesce_timeouts") == 0,
            stats_delta.get("coalesce_reuse_failures") == 0,
        ]),
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
    image = image_path.read_bytes()
    contract = capture_contract(config, image, list(state.get("labels") or []))
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
    health_before = read_health(endpoint)
    barrier = threading.Barrier(2)

    def send():
        barrier.wait(timeout=5)
        return request_json(endpoint, body, request_config.get("llm_key", ""))

    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(executor.map(lambda _: send(), range(2)))
    health_after = read_health(endpoint)
    records = [record(response, wall_ms) for response, wall_ms in responses]
    records.sort(key=lambda item: bool(item.get("cache_hit")))
    stats_delta = cache_stat_delta(health_before, health_after)
    core_acceptance = coalescing_acceptance(records, stats_delta)
    acceptance = {**core_acceptance,
        "source_state_modified": file_sha256(state_path) != state_before,
        "candidate_written": False,
        "human_truth_modified": False,
        "training_data_modified": False,
    }
    result = {
        "schema_version": "labelagent.concurrent-exact-coalescing.v1",
        "read_only": True,
        "image": str(image_path),
        "image_sha256": hashlib.sha256(image).hexdigest(),
        "records": records,
        "health_before": health_before.get("response_cache_stats"),
        "health_after": health_after.get("response_cache_stats"),
        "response_cache_stats_delta": stats_delta,
        "acceptance": acceptance,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print("RESULT=" + json.dumps(acceptance, ensure_ascii=False))
    return 0 if (all(core_acceptance.values())
                 and not acceptance["source_state_modified"]
                 and not acceptance["candidate_written"]
                 and not acceptance["human_truth_modified"]
                 and not acceptance["training_data_modified"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
