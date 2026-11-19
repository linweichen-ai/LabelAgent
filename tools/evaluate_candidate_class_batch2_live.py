"""Compare live sequential and two-item candidate class audits read-only.

The evaluator reads one existing workbench frame and its frozen candidate state,
then sends the same two independent blind-audit prompts through the established
single-request path and the local VLM's native two-item batch endpoint.  It
writes only a receipt under ``artifacts`` and never updates candidates, truth,
reference frames, training approval, model weights, or the workbench state.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.agent import candidate_class_audit as audit  # noqa: E402
from source.agent.workflow import VISIBLE_LABEL_DEFINITIONS  # noqa: E402


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_sha256(value) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(payload)


def data_url(payload: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(payload).decode("ascii")


def request_json(url: str, body: dict, api_key: str) -> tuple[dict, float]:
    request = urllib.request.Request(
        url,
        json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key,
        },
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=1200) as response:
        payload = json.load(response)
    return payload, round(time.perf_counter() - started, 3)


def fetch_frame(base_url: str, frame: int, *, topic: str, path: str) -> bytes:
    query = urllib.parse.urlencode({"topic": topic, "path": path})
    url = f"{base_url.rstrip('/')}/api/frame/{frame}?{query}"
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read()


def frame_candidates(state: dict, topic: str, frame: int) -> list[dict]:
    key = str(frame)
    by_topic = state.get("candidates_by_topic") or {}
    scoped = by_topic.get(topic) if isinstance(by_topic, dict) else None
    candidates = scoped if isinstance(scoped, dict) else state.get("candidates")
    rows = (candidates or {}).get(key)
    if not isinstance(rows, list):
        raise ValueError(f"frame {frame} has no candidate list")
    return rows


def body_for(config: dict, prompt: str, images: tuple[bytes, ...], schema: dict,
             max_tokens: int) -> dict:
    content = [{"type": "text", "text": prompt}]
    content.extend(
        {"type": "image_url", "image_url": {"url": data_url(image)}}
        for image in images
    )
    return {
        "model": config["llm_model"],
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": content}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "labelagent", "schema": schema, "strict": True},
        },
    }


def parse_output(output: dict, labels: list[str]) -> tuple[dict, dict | None, str | None, str | None]:
    content = output["choices"][0]["message"]["content"]
    raw = json.loads(content)
    repair = None
    error_text = None
    try:
        decision = audit.validate_blind_candidate_class_decision(raw, labels)
    except audit.CandidateClassProtocolError as error:
        repaired = audit._repair_non_known_reason_code(raw)
        if repaired is None:
            decision = None
            error_text = str(error)
        else:
            decision = audit.validate_blind_candidate_class_decision(repaired, labels)
            repair = "non_known_reason_code_normalized"
    return raw, decision, repair, error_text


def semantic_core(decision: dict, current_label: str) -> dict:
    preserved = (
        decision["classification"] == "known_label"
        and decision["observed_label"] == current_label
        and decision["candidate_supported"]
        and decision["confidence"] >= 0.90
    )
    return {
        "classification": decision["classification"],
        "observed_label": decision["observed_label"],
        "candidate_supported": decision["candidate_supported"],
        "reason_code": decision["reason_code"],
        "route": "candidate_preserved" if preserved else "human_review_required",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--frame", required=True, type=int)
    parser.add_argument("--candidate-ids", nargs=2)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sequential-max-tokens", default=768, type=int)
    parser.add_argument("--batch-max-tokens", default=384, type=int)
    args = parser.parse_args()

    state_path = args.state.resolve()
    output_path = args.output.resolve()
    artifact_root = (ROOT / "artifacts").resolve()
    if not output_path.is_relative_to(artifact_root):
        raise ValueError("output must be inside this repository's artifacts directory")
    if output_path.exists():
        raise FileExistsError(output_path)
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    if not 1 <= args.sequential_max_tokens <= 4096:
        raise ValueError("sequential-max-tokens is out of range")
    if not 1 <= args.batch_max_tokens <= 4096:
        raise ValueError("batch-max-tokens is out of range")

    state_before = state_path.read_bytes()
    state = json.loads(state_before)
    config = dict(state.get("config") or {})
    labels = list(state.get("labels") or [])
    topic = state.get("topic")
    source_path = state.get("path") or state.get("video_path")
    if not isinstance(topic, str) or not isinstance(source_path, str):
        raise ValueError("state lacks the selected dataset path or topic")
    if not config.get("llm_url") or not config.get("llm_model") or not labels:
        raise ValueError("state lacks model settings or labels")
    parsed = urllib.parse.urlsplit(config["llm_url"])
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("the evaluator only permits a loopback VLM endpoint")

    candidates = frame_candidates(state, topic, args.frame)
    requested_ids = set(args.candidate_ids or [])
    targets = [
        candidate for candidate in candidates
        if audit.candidate_requires_class_audit(candidate)
        and (not requested_ids or str(candidate.get("id")) in requested_ids)
    ]
    if len(targets) != 2:
        raise ValueError(
            f"exactly two current audit targets are required; observed {len(targets)}"
        )
    if requested_ids and {str(item.get("id")) for item in targets} != requested_ids:
        raise ValueError("one or more requested candidate IDs are missing")

    image_bytes = fetch_frame(
        args.base_url, args.frame, topic=topic, path=source_path
    )
    schema = audit.blind_candidate_class_schema(labels)
    prepared = []
    for candidate in targets:
        evidence = audit.prepare_candidate_evidence(image_bytes, candidate)
        prompt = audit.build_blind_candidate_class_prompt(
            candidate["id"],
            labels,
            VISIBLE_LABEL_DEFINITIONS,
            isolated_only=str(candidate.get("label", "")).casefold() == "pallet",
        )
        primary, extra = audit.candidate_audit_images(candidate, evidence)
        prepared.append({
            "candidate": candidate,
            "evidence": evidence,
            "prompt": prompt,
            "images": (primary, *extra),
        })

    single_url = config["llm_url"].rstrip("/") + "/chat/completions"
    batch_url = config["llm_url"].rstrip("/") + "/chat/batch_completions"
    api_key = config.get("llm_key", "")
    sequential_outputs = []
    sequential_wall = 0.0
    for item in prepared:
        payload, elapsed = request_json(
            single_url,
            body_for(
                config, item["prompt"], item["images"], schema,
                args.sequential_max_tokens,
            ),
            api_key,
        )
        sequential_outputs.append(payload)
        sequential_wall += elapsed

    batch_bodies = [
        body_for(
            config, item["prompt"], item["images"], schema,
            args.batch_max_tokens,
        )
        for item in prepared
    ]
    batch_payload, batch_wall = request_json(
        batch_url, {"requests": batch_bodies}, api_key
    )
    batch_outputs = batch_payload.get("results")
    if not isinstance(batch_outputs, list) or len(batch_outputs) != 2:
        raise RuntimeError("batch endpoint returned a mismatched result count")

    rows = []
    for item, single, batched in zip(prepared, sequential_outputs, batch_outputs):
        single_raw, single_decision, single_repair, single_error = parse_output(
            single, labels
        )
        batch_raw, batch_decision, batch_repair, batch_error = parse_output(
            batched, labels
        )
        candidate = item["candidate"]
        single_core = (
            semantic_core(single_decision, candidate["label"])
            if single_decision is not None else None
        )
        batch_core = (
            semantic_core(batch_decision, candidate["label"])
            if batch_decision is not None else None
        )
        rows.append({
            "candidate_id": candidate["id"],
            "bound_current_label": candidate["label"],
            "candidate_sha256": audit.candidate_fingerprint(candidate),
            "evidence": {
                "crop_box_xyxy": item["evidence"]["crop_box_xyxy"],
                "isolated_sha256": item["evidence"]["isolated_sha256"],
                "current_label_shown_to_model": False,
            },
            "sequential": {
                "semantic_core": single_core,
                "confidence": single_decision["confidence"],
                "cache_hit": bool((single.get("local_vlm") or {}).get("cache_hit")),
                "seconds": (single.get("local_vlm") or {}).get("seconds"),
                "raw_decision": single_raw,
                "protocol_valid": single_decision is not None,
                "protocol_repair": single_repair,
                "protocol_error": single_error,
            },
            "batch2": {
                "semantic_core": batch_core,
                "confidence": batch_decision["confidence"],
                "cache_hit": bool((batched.get("local_vlm") or {}).get("cache_hit")),
                "seconds": (batched.get("local_vlm") or {}).get("seconds"),
                "raw_decision": batch_raw,
                "protocol_valid": batch_decision is not None,
                "protocol_repair": batch_repair,
                "protocol_error": batch_error,
            },
            "semantic_core_match": single_core == batch_core,
            "route_match": (
                single_core is not None and batch_core is not None
                and single_core["route"] == batch_core["route"]
            ),
            "candidate_only": True,
            "human_truth": False,
            "approved_for_training": False,
        })

    all_uncached = all(
        not row[mode]["cache_hit"]
        for row in rows for mode in ("sequential", "batch2")
    )
    semantic_match = all(row["semantic_core_match"] for row in rows)
    route_match = all(row["route_match"] for row in rows)
    protocol_valid = all(
        row[mode]["protocol_valid"]
        for row in rows for mode in ("sequential", "batch2")
    )
    latency_gain = sequential_wall - batch_wall
    result = {
        "schema_version": "labelagent.candidate-class-batch2-live-evaluation.v1",
        "frame": args.frame,
        "topic": topic,
        "source_path_sha256": sha256_bytes(source_path.encode("utf-8")),
        "state_sha256": sha256_bytes(state_before),
        "candidate_artifact_sha256": canonical_sha256(candidates),
        "source_image_sha256": sha256_bytes(image_bytes),
        "candidate_ids": [item["candidate"]["id"] for item in prepared],
        "sequential": {
            "max_tokens": args.sequential_max_tokens,
            "wall_seconds": round(sequential_wall, 3),
        },
        "batch2": {
            "max_tokens": args.batch_max_tokens,
            "wall_seconds": batch_wall,
            "local_vlm": batch_payload.get("local_vlm"),
        },
        "rows": rows,
        "summary": {
            "all_uncached": all_uncached,
            "all_protocol_valid": protocol_valid,
            "all_semantic_cores_match": semantic_match,
            "all_routes_match": route_match,
            "latency_gain_seconds": round(latency_gain, 3),
            "latency_reduction_fraction": (
                round(latency_gain / sequential_wall, 4) if sequential_wall else None
            ),
            "safe_to_consider_batch2": (
                all_uncached and protocol_valid and semantic_match
                and route_match and latency_gain > 0
            ),
        },
        "inputs_unchanged": {
            "state": state_path.read_bytes() == state_before,
            "candidates": canonical_sha256(
                frame_candidates(json.loads(state_path.read_bytes()), topic, args.frame)
            ) == canonical_sha256(candidates),
        },
        "formal_workbench_writes": False,
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "approved_for_deployment": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
