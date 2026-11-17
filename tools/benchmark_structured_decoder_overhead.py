"""Isolate structured-decoder overhead with identical long JSON output.

Both variants receive the same image, prompt, target JSON, model, sampling and
token budget.  Only ``response_format`` differs.  A row is valid only when both
variants reproduce the exact same target value and completion-token count.
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

from tools.evaluate_visible_discovery_format_ablation import capture_contract


def request_json(url: str, body: dict, key: str) -> tuple[dict, float]:
    request = urllib.request.Request(
        url, data=json.dumps(body, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + key})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=1200) as response:
        result = json.load(response)
    return result, round((time.perf_counter() - started) * 1000, 3)


def response_record(response: dict, target: dict, wall_ms: float) -> dict:
    choice = response["choices"][0]
    content = choice["message"]["content"]
    try:
        parsed = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        parsed = None
    local = response.get("local_vlm") or {}
    stats = local.get("inference_stats") or {}
    return {
        "wall_ms": wall_ms,
        "finish_reason": choice.get("finish_reason"),
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "exact_target_value": parsed == target,
        "cache_hit": local.get("cache_hit"),
        "structured_output": stats.get("structured_output"),
        "generation_seconds": stats.get("generation_seconds"),
        "constraint_setup_seconds": stats.get("constraint_setup_seconds"),
        "completion_tokens": stats.get("completion_tokens"),
        "prompt_tokens": stats.get("prompt_tokens"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trials", type=int, default=2)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "artifacts").resolve()):
        raise ValueError("output must be inside repository artifacts")
    if output.exists():
        raise FileExistsError(output)
    if not 1 <= args.trials <= 4:
        raise ValueError("trials must be between one and four")

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    source = baseline["rows"][0]
    image_path = Path(source["image"]).resolve()
    image = image_path.read_bytes()
    if hashlib.sha256(image).hexdigest() != source["image_sha256"]:
        raise ValueError("baseline image binding changed")
    target = source["structured"]["value"]
    state_path = args.state.resolve()
    state_before = hashlib.sha256(state_path.read_bytes()).hexdigest()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    config = dict(state.get("config") or {})
    labels = list(state.get("labels") or [])
    for key in tuple(config):
        if key.startswith("llm_fast_"):
            config.pop(key, None)
    contract = capture_contract(config, image, labels)
    request_config = contract["config"]
    target_text = json.dumps(target, ensure_ascii=False, separators=(",", ":"))
    endpoint = request_config["llm_url"].rstrip("/") + "/chat/completions"
    rows = []
    for trial in range(args.trials):
        prompt = (
            f"Decoder benchmark trial {trial + 1}. Return exactly the following JSON value "
            "with every field and value preserved. Return JSON only, without markdown or commentary:\n"
            + target_text)
        common = {
            "model": request_config["llm_model"],
            "temperature": 0,
            "max_tokens": int(request_config.get("llm_max_tokens") or 768),
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url":
                    "data:image/jpeg;base64," + base64.b64encode(image).decode()}},
            ]}],
        }
        structured = dict(common)
        structured["response_format"] = {
            "type": "json_schema", "json_schema": {
                "name": "labelagent", "schema": contract["response_schema"],
                "strict": True}}
        variants = [("structured", structured), ("unconstrained", common)]
        if trial % 2:
            variants.reverse()
        records = {}
        for name, body in variants:
            response, wall_ms = request_json(
                endpoint, body, request_config.get("llm_key", ""))
            records[name] = response_record(response, target, wall_ms)
        row = {
            "trial": trial + 1,
            "order": [name for name, _ in variants],
            **records,
            "valid_identical_output": (
                records["structured"]["exact_target_value"]
                and records["unconstrained"]["exact_target_value"]
                and records["structured"]["completion_tokens"]
                == records["unconstrained"]["completion_tokens"]),
        }
        rows.append(row)
        print(json.dumps({
            "trial": row["trial"], "order": row["order"],
            "valid_identical_output": row["valid_identical_output"],
            "structured_ms": row["structured"]["wall_ms"],
            "unconstrained_ms": row["unconstrained"]["wall_ms"],
            "structured_tokens": row["structured"]["completion_tokens"],
            "unconstrained_tokens": row["unconstrained"]["completion_tokens"],
        }, ensure_ascii=False), flush=True)

    valid = [row for row in rows if row["valid_identical_output"]
             and row["structured"]["cache_hit"] is not True
             and row["unconstrained"]["cache_hit"] is not True]
    structured_median_ms = (round(statistics.median(
        row["structured"]["wall_ms"] for row in valid), 3)
        if valid else None)
    unconstrained_median_ms = (round(statistics.median(
        row["unconstrained"]["wall_ms"] for row in valid), 3)
        if valid else None)
    overhead_ms = (round(structured_median_ms - unconstrained_median_ms, 3)
                   if valid else None)
    overhead_pct = (round(overhead_ms / unconstrained_median_ms * 100, 3)
                    if valid and unconstrained_median_ms else None)
    result = {
        "schema_version": "labelagent.structured-decoder-overhead.v1",
        "read_only": True,
        "image": str(image_path),
        "image_sha256": hashlib.sha256(image).hexdigest(),
        "target_sha256": hashlib.sha256(target_text.encode()).hexdigest(),
        "rows": rows,
        "summary": {
            "trial_count": len(rows),
            "valid_identical_output_count": len(valid),
            "structured_median_ms": structured_median_ms,
            "unconstrained_median_ms": unconstrained_median_ms,
            "structured_overhead_ms": overhead_ms,
            "structured_overhead_pct": overhead_pct,
            "candidate_written": False,
            "human_truth_modified": False,
            "training_data_modified": False,
            "source_state_modified": (
                hashlib.sha256(state_path.read_bytes()).hexdigest() != state_before),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print("RESULT=" + json.dumps(result["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
