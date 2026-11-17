"""Evaluate true independent GPU batching for frozen candidate-class audits.

The tool reads frozen state, image and candidate artifacts and calls only the
bounded experimental batch endpoint.  It never updates workbench state,
candidates, truth, reference frames, or training approval.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import time
import urllib.request
import urllib.parse
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from source.agent import candidate_class_audit as audit
from source.agent import workflow


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def semantic_core(decision, route):
    return {
        "classification": decision.get("classification"),
        "observed_label": decision.get("observed_label"),
        "candidate_supported": decision.get("candidate_supported"),
        "reason_code": decision.get("reason_code"),
        "route": route,
    }


def data_url(payload: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(payload).decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=Path)
    image_group = parser.add_mutually_exclusive_group(required=True)
    image_group.add_argument("--image", type=Path)
    image_group.add_argument("--image-url")
    parser.add_argument("--candidate-json", type=Path)
    parser.add_argument("--frame", type=int)
    parser.add_argument("--candidate-ids", required=True, nargs="+", type=int)
    parser.add_argument("--max-tokens", default=384, type=int)
    parser.add_argument("--llm-url")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if not 2 <= len(args.candidate_ids) <= 3:
        raise ValueError("independent batch evaluation requires two or three candidates")
    if not 1 <= args.max_tokens <= 4096:
        raise ValueError("max-tokens must be between 1 and 4096")
    artifact_root = (REPOSITORY_ROOT / "artifacts").resolve()
    state_path = args.state.resolve()
    image_path = args.image.resolve() if args.image else None
    candidate_path = args.candidate_json.resolve() if args.candidate_json else None
    output_path = args.output.resolve()
    if not output_path.is_relative_to(artifact_root):
        raise ValueError("output must be inside this repository's artifacts directory")
    if output_path.exists():
        raise FileExistsError(output_path)
    if (candidate_path is None) == (args.frame is None):
        raise ValueError("provide exactly one of --candidate-json or --frame")
    for path in (state_path, *([image_path] if image_path else []),
                 *([candidate_path] if candidate_path else [])):
        if not path.is_file():
            raise FileNotFoundError(path)

    def read_image_source():
        if image_path:
            return image_path.read_bytes()
        parsed = urllib.parse.urlsplit(args.image_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
            raise ValueError("image-url must use the loopback workbench")
        with urllib.request.urlopen(args.image_url, timeout=30) as response:
            return response.read()

    image_bytes = read_image_source()

    hashes_before = {"state": sha256(state_path),
                     "image": hashlib.sha256(image_bytes).hexdigest(),
                     "candidate_json": (sha256(candidate_path)
                                        if candidate_path else None)}
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if candidate_path:
        frozen = json.loads(candidate_path.read_text(encoding="utf-8"))
    else:
        key = str(args.frame)
        frozen = {"candidates": (state.get("candidates") or {}).get(key),
                  "agent": (state.get("agent_runs") or {}).get(key)}
    config = dict(state.get("config") or {})
    if args.llm_url:
        config["llm_url"] = args.llm_url
    labels = list(state.get("labels") or [])
    if not config.get("llm_url") or not config.get("llm_model") or not labels:
        raise ValueError("state lacks model settings or labels")
    candidates = {item.get("id"): item
                  for item in frozen.get("candidates") or []}
    baselines = {item.get("candidate_id"): item for item in
                 (frozen.get("agent") or {}).get("candidate_class_audits") or []}
    requested = []
    schema = audit.blind_candidate_class_schema(labels)
    for candidate_id in args.candidate_ids:
        candidate = candidates.get(candidate_id)
        if candidate is None or candidate_id not in baselines:
            raise ValueError("requested candidate or production baseline is missing")
        if candidate.get("label") not in audit.HIGH_RISK_LABELS:
            raise ValueError("evaluation is restricted to production high-risk labels")
        evidence = audit.prepare_candidate_evidence(image_bytes, candidate)
        prompt = audit.build_blind_candidate_class_prompt(
            candidate_id, labels, workflow.VISIBLE_LABEL_DEFINITIONS)
        requested.append({
            "candidate": candidate,
            "evidence": evidence,
            "body": {
                "model": config["llm_model"], "temperature": 0,
                "max_tokens": args.max_tokens,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": data_url(evidence["original"])}},
                    {"type": "image_url", "image_url": {
                        "url": data_url(evidence["marked"])}},
                ]}],
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": "labelagent", "schema": schema, "strict": True}},
            },
        })

    endpoint = config["llm_url"].rstrip("/") + "/chat/batch_completions"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps({"requests": [item["body"] for item in requested]}).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + config.get("llm_key", "")})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=1200) as response:
        response_body = json.load(response)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    outputs = response_body.get("results")
    if not isinstance(outputs, list) or len(outputs) != len(requested):
        raise RuntimeError("batch endpoint returned a mismatched result count")

    rows = []
    for item, output in zip(requested, outputs):
        candidate = item["candidate"]
        content = output["choices"][0]["message"]["content"]
        raw_decision = json.loads(content)
        protocol_error = None
        try:
            decision = audit.validate_blind_candidate_class_decision(
                raw_decision, labels)
        except Exception as error:
            decision = None
            protocol_error = str(error)
        route = ("candidate_preserved" if decision is not None
                 and decision["classification"] == "known_label"
                 and decision["observed_label"] == candidate["label"]
                 and decision["confidence"] >= 0.90
                 else "human_review_required")
        baseline = baselines[candidate["id"]]
        baseline_core = semantic_core(
            baseline.get("decision") or {}, baseline.get("route"))
        candidate_core = (semantic_core(decision, route)
                          if decision is not None else None)
        rows.append({
            "candidate_id": candidate["id"],
            "bound_current_label": candidate["label"],
            "candidate_fingerprint": audit.candidate_fingerprint(candidate),
            "baseline_semantic_core": baseline_core,
            "candidate_semantic_core": candidate_core,
            "semantic_core_match": (decision is not None
                                    and candidate_core == baseline_core),
            "protocol_valid": decision is not None,
            "protocol_error": protocol_error,
            "raw_model_decision": raw_decision,
            "usage": output.get("usage"),
            "finish_reason": output["choices"][0].get("finish_reason"),
            "original_sha256": item["evidence"]["original_sha256"],
            "marked_sha256": item["evidence"]["marked_sha256"],
            "candidate_only": True,
            "human_truth": False,
            "approved_for_training": False,
        })
    local = response_body.get("local_vlm") or {}
    result = {
        "schema_version": "labelagent.independent-batch-class-audit.v1",
        "endpoint": endpoint,
        "batch_size": len(rows),
        "max_tokens": args.max_tokens,
        "elapsed_ms": elapsed_ms,
        "local_vlm": local,
        "rows": rows,
        "summary": {
            "protocol_valid_count": sum(row["protocol_valid"] for row in rows),
            "semantic_core_matches": sum(row["semantic_core_match"] for row in rows),
            "all_protocol_valid": all(row["protocol_valid"] for row in rows),
            "all_semantic_cores_match": all(row["semantic_core_match"] for row in rows),
            "all_candidate_only": all(row["candidate_only"] for row in rows),
            "all_human_truth_false": all(not row["human_truth"] for row in rows),
            "all_training_approval_false": all(
                not row["approved_for_training"] for row in rows),
        },
        "inputs": {
            "state": str(state_path), "state_sha256": hashes_before["state"],
            "image": str(image_path) if image_path else args.image_url,
            "image_sha256": hashes_before["image"],
            "candidate_json": str(candidate_path) if candidate_path else None,
            "candidate_json_sha256": hashes_before["candidate_json"],
            "state_frame": args.frame,
        },
        "inputs_unchanged": {
            "state": sha256(state_path) == hashes_before["state"],
            "image": hashlib.sha256(read_image_source()).hexdigest()
                     == hashes_before["image"],
            "candidate_json": (sha256(candidate_path) == hashes_before["candidate_json"]
                               if candidate_path else True),
        },
        "formal_workbench_writes": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
