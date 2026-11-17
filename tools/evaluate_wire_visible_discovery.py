"""Probe a lossless short-key wire schema for visible-label discovery.

All production semantic fields remain mandatory.  Only repeated JSON field
names and registry-label keys are shortened, then deterministically expanded
before comparison with a frozen full-protocol baseline.
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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluate_visible_discovery_format_ablation import capture_contract


def wire_schema(labels: list[str]) -> tuple[dict, dict[str, str]]:
    mapping = {f"l{index}": label for index, label in enumerate(labels)}
    assessment = {
        "type": "object",
        "properties": {
            "e": {"type": "string",
                  "enum": ["present", "absent", "uncertain"]},
            "c": {"type": "number", "minimum": 0, "maximum": 1},
            "r": {"type": "string", "minLength": 1, "maxLength": 120},
        },
        "required": ["e", "c", "r"],
        "additionalProperties": False,
    }
    schema = {
        "type": "object",
        "properties": {
            "assessments": {
                "type": "object",
                "properties": {wire: assessment for wire in mapping},
                "required": list(mapping),
                "additionalProperties": False,
            },
            "unknown_candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate_name": {"type": "string", "minLength": 1,
                                           "maxLength": 48},
                        "evidence": {"type": "string", "minLength": 1,
                                     "maxLength": 120},
                        "confidence": {"type": "number", "minimum": 0,
                                       "maximum": 1},
                    },
                    "required": ["candidate_name", "evidence", "confidence"],
                    "additionalProperties": False,
                },
                "maxItems": 5,
            },
        },
        "required": ["assessments", "unknown_candidates"],
        "additionalProperties": False,
    }
    return schema, mapping


def wire_instruction(production: str, mapping: dict[str, str]) -> str:
    old = (
        "Return assessments as one JSON object keyed by the exact known label strings; "
        "every key is required and must occur once. "
        "Use evidence='present' only when at least one instance is visibly supported in the image; "
        "use 'absent' when it is not visible and 'uncertain' when occlusion or image quality prevents a reliable decision. "
        "Do not infer an object merely because its label is listed. confidence is a number from 0 to 1. "
        "Each reason must be one visual clause of at most 10 words; do not list multiple exclusions or repeat the definition. "
        "A reason must name visible pixels or a visible physical part, not scene context, camera viewpoint, or an assumed function. "
    )
    new = (
        "Return assessments as one JSON object keyed by the required wire IDs below; "
        "every wire ID is required and must occur once. Each value uses e for evidence, c for confidence, and r for reason. "
        "Use e='present' only when at least one mapped label instance is visibly supported in the image; "
        "use 'absent' when it is not visible and 'uncertain' when occlusion or image quality prevents a reliable decision. "
        "Do not infer an object merely because its mapped label is listed. c is a number from 0 to 1. "
        "Each r must be one visual clause of at most 10 words; do not list multiple exclusions or repeat the definition. "
        "An r must name visible pixels or a visible physical part, not scene context, camera viewpoint, or an assumed function. "
        "Wire ID mapping: " + json.dumps(mapping, ensure_ascii=False) + ". "
    )
    if production.count(old) != 1:
        raise ValueError("production discovery instruction no longer matches the audited contract")
    return production.replace(old, new)


def expand_wire(value: dict, mapping: dict[str, str]) -> dict:
    assessments = value.get("assessments")
    if not isinstance(assessments, dict) or set(assessments) != set(mapping):
        raise ValueError("wire response omitted or added an assessment")
    expanded = {}
    for wire, label in mapping.items():
        item = assessments[wire]
        expanded[label] = {
            "evidence": item["e"],
            "confidence": item["c"],
            "reason": item["r"],
        }
    return {"assessments": expanded,
            "unknown_candidates": value.get("unknown_candidates")}


def request_json(url: str, body: dict, key: str) -> tuple[dict, float]:
    request = urllib.request.Request(
        url, data=json.dumps(body, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + key})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=1200) as response:
        result = json.load(response)
    return result, round((time.perf_counter() - started) * 1000, 3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "artifacts").resolve()):
        raise ValueError("output must be inside repository artifacts")
    if output.exists():
        raise FileExistsError(output)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    state_path = args.state.resolve()
    state_before = hashlib.sha256(state_path.read_bytes()).hexdigest()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    config = dict(state.get("config") or {})
    labels = list(state.get("labels") or [])
    for key in tuple(config):
        if key.startswith("llm_fast_"):
            config.pop(key, None)
    schema, mapping = wire_schema(labels)
    rows = []
    for position, source in enumerate(baseline["rows"], 1):
        path = Path(source["image"]).resolve()
        image = path.read_bytes()
        digest = hashlib.sha256(image).hexdigest()
        if digest != source["image_sha256"]:
            raise ValueError(f"baseline image binding changed: {path}")
        contract = capture_contract(config, image, labels)
        request_config = contract["config"]
        body = {
            "model": request_config["llm_model"],
            "temperature": 0,
            "max_tokens": int(request_config.get("llm_max_tokens") or 768),
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": wire_instruction(
                    contract["instruction"], mapping)},
                {"type": "image_url", "image_url": {"url":
                    "data:image/jpeg;base64," + base64.b64encode(image).decode()}},
            ]}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "labelagent_wire_discovery",
                "schema": schema, "strict": True}},
        }
        response, wall_ms = request_json(
            request_config["llm_url"].rstrip("/") + "/chat/completions",
            body, request_config.get("llm_key", ""))
        value = json.loads(response["choices"][0]["message"]["content"])
        expanded = expand_wire(value, mapping)
        expected = source["structured"]["value"]
        expected_states = {label: expected["assessments"][label]["evidence"]
                           for label in labels}
        actual_states = {label: expanded["assessments"][label]["evidence"]
                         for label in labels}
        local = response.get("local_vlm") or {}
        stats = local.get("inference_stats") or {}
        row = {
            "image": str(path),
            "image_sha256": digest,
            "wall_ms": wall_ms,
            "cache_hit": local.get("cache_hit"),
            "finish_reason": response["choices"][0].get("finish_reason"),
            "completion_tokens": stats.get("completion_tokens"),
            "structured_output": stats.get("structured_output"),
            "expanded": expanded,
            "full_json_equal": expanded == expected,
            "known_states_equal": actual_states == expected_states,
            "unknown_candidates_equal": (expanded["unknown_candidates"]
                                         == expected.get("unknown_candidates")),
            "state_differences": {label: {
                "baseline": expected_states[label], "candidate": actual_states[label],
            } for label in labels if actual_states[label] != expected_states[label]},
        }
        rows.append(row)
        print(json.dumps({"progress": f"{position}/{len(baseline['rows'])}",
            "wall_ms": wall_ms, "completion_tokens": row["completion_tokens"],
            "full_json_equal": row["full_json_equal"],
            "known_states_equal": row["known_states_equal"],
            "unknown_candidates_equal": row["unknown_candidates_equal"],
            "differences": row["state_differences"]},
            ensure_ascii=False), flush=True)
    result = {
        "schema_version": "labelagent.wire-visible-discovery-probe.v1",
        "read_only": True,
        "baseline": str(args.baseline.resolve()),
        "mapping": mapping,
        "rows": rows,
        "summary": {
            "sample_count": len(rows),
            "full_json_equal_count": sum(row["full_json_equal"] for row in rows),
            "known_states_equal_count": sum(
                row["known_states_equal"] for row in rows),
            "unknown_candidates_equal_count": sum(
                row["unknown_candidates_equal"] for row in rows),
            "cache_hit_count": sum(row["cache_hit"] is True for row in rows),
            "candidate_total_ms": round(sum(row["wall_ms"] for row in rows), 3),
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
