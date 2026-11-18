#!/usr/bin/env python3
"""Replay bounded VLM trace inputs without copying prompts or images to output."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import time
import urllib.request
from pathlib import Path


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def load_case(spec):
    parts = spec.split("|", 2)
    if len(parts) != 3:
        raise ValueError("Each --case must be DEBUG_JSON|CALL_ID|INSPECTION_DIR")
    debug_path, call_id, inspection_dir = Path(parts[0]), parts[1], Path(parts[2])
    payload = json.loads(debug_path.read_text(encoding="utf-8"))
    matches = [record for record in payload.get("trace", [])
               if isinstance(record, dict) and record.get("id") == call_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one trace call {call_id}, found {len(matches)}")
    record = matches[0]
    bounded = record.get("input")
    if not isinstance(bounded, dict):
        raise ValueError(f"Trace call {call_id} has no bounded input")
    instruction = bounded.get("instruction")
    response_schema = bounded.get("response_schema")
    max_tokens = bounded.get("max_tokens")
    image_records = bounded.get("images")
    if (not isinstance(instruction, str) or not isinstance(response_schema, dict)
            or type(max_tokens) is not int or not isinstance(image_records, list)):
        raise ValueError(f"Trace call {call_id} has an incomplete generation contract")

    content = [{"type": "text", "text": instruction}]
    image_hashes = []
    for item in image_records:
        expected = item.get("sha256") if isinstance(item, dict) else None
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"Trace call {call_id} has an invalid image digest")
        matches = [path for path in inspection_dir.glob(expected + ".*") if path.is_file()]
        if len(matches) != 1:
            raise ValueError(f"Expected one image for {expected}, found {len(matches)}")
        data = matches[0].read_bytes()
        if sha256(data) != expected:
            raise ValueError(f"Image digest mismatch for {matches[0]}")
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            mime = "image/png"
        elif data.startswith(b"\xff\xd8"):
            mime = "image/jpeg"
        else:
            raise ValueError(f"Unsupported image type for {matches[0]}")
        content.append({"type": "image_url", "image_url": {
            "url": f"data:{mime};base64," + base64.b64encode(data).decode("ascii")}})
        image_hashes.append(expected)

    request_body = {
        "model": "labelagent-vlm8b",
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": content}],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "labelagent", "schema": response_schema, "strict": True}},
    }
    return request_body, {
        "call_id": call_id,
        "frame": record.get("frame"),
        "purpose": record.get("purpose"),
        "image_count": len(image_hashes),
        "image_sha256": image_hashes,
        "generation_contract_sha256": sha256(json.dumps({
            "instruction": instruction,
            "response_schema": response_schema,
            "max_tokens": max_tokens,
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")),
    }


def replay(endpoint, request_body, metadata, timeout):
    started = time.monotonic()
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    elapsed = time.monotonic() - started
    content = result["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(content)
        parsed_sha = sha256(json.dumps(
            parsed, ensure_ascii=False, sort_keys=True,
            separators=(",", ":")).encode("utf-8"))
        parse_error = None
    except (TypeError, ValueError) as error:
        parsed, parsed_sha, parse_error = None, None, str(error)
    return {
        "source": metadata,
        "elapsed_seconds": round(elapsed, 3),
        "content_sha256": sha256(content.encode("utf-8")),
        "content_chars": len(content),
        "parsed_sha256": parsed_sha,
        "parse_error": parse_error,
        "parsed": parsed,
        "usage": result.get("usage"),
        "local_vlm": result.get("local_vlm"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args(argv)
    rows = []
    for spec in args.case:
        request_body, metadata = load_case(spec)
        rows.append(replay(args.endpoint, request_body, metadata, args.timeout))
    report = {
        "schema_version": "labelagent.vlm_trace_replay.v1",
        "endpoint": args.endpoint,
        "rows": rows,
        "content_boundary": "No prompts, image bytes or credentials are stored in this report."
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps([{
        "call_id": row["source"]["call_id"],
        "elapsed_seconds": row["elapsed_seconds"],
        "generation_seconds": row.get("local_vlm", {}).get(
            "inference_stats", {}).get("generation_seconds"),
        "cache": row.get("local_vlm", {}).get(
            "inference_stats", {}).get("generation_cache_implementation"),
        "parsed_sha256": row["parsed_sha256"],
    } for row in rows], ensure_ascii=False))


if __name__ == "__main__":
    main()
