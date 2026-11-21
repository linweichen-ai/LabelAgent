#!/usr/bin/env python3
"""Compare one frozen class audit on primary and fast VLM routes.

The evaluator reads one candidate and one loopback-served frame, performs no
workbench writes, and stores only an experiment receipt under ``artifacts``.
"""
from __future__ import annotations

import argparse
import copy
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
from source.agent import workflow  # noqa: E402


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_loopback(url: str) -> bytes:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("image-url must use the loopback workbench")
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read()


def semantic_core(result: dict) -> dict:
    decision = result["decision"]
    return {
        "classification": decision["classification"],
        "observed_label": decision["observed_label"],
        "candidate_supported": decision["candidate_supported"],
        "reason_code": decision["reason_code"],
        "route": result["route"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--image-url", required=True)
    parser.add_argument("--frame", required=True, type=int)
    parser.add_argument("--candidate-id", required=True, type=int)
    parser.add_argument("--synthetic-id", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    state_path = args.state.resolve()
    output_path = args.output.resolve()
    artifact_root = (ROOT / "artifacts").resolve()
    if not output_path.is_relative_to(artifact_root):
        raise ValueError("output must be inside repository artifacts")
    if output_path.exists():
        raise FileExistsError(output_path)
    state_bytes = state_path.read_bytes()
    state = json.loads(state_bytes.decode("utf-8"))
    candidates = (state.get("candidates") or {}).get(str(args.frame)) or []
    source = next((item for item in candidates
                   if item.get("id") == args.candidate_id), None)
    if source is None:
        raise ValueError("candidate is missing from the frozen frame")
    candidate = copy.deepcopy(source)
    candidate["id"] = args.synthetic_id
    image_bytes = read_loopback(args.image_url)
    labels = list(state.get("labels") or [])
    config = dict(state.get("config") or {})
    if not labels or not config.get("llm_url") or not config.get("llm_fast_url"):
        raise ValueError("state lacks labels or primary/fast VLM routes")

    primary_config = dict(config)
    primary_config["llm_fast_url"] = ""
    primary_config["llm_fast_model"] = ""

    def evaluate(route: str, call, route_config: dict) -> dict:
        started = time.perf_counter()
        result = audit.audit_candidate_class_blind(
            route_config, image_bytes, candidate, labels,
            workflow.VISIBLE_LABEL_DEFINITIONS, call)
        return {
            "route": route,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "semantic_core": semantic_core(result),
            "decision": result["decision"],
            "candidate_sha256": result["candidate_sha256"],
            "audit_sha256": result["audit_sha256"],
        }

    primary = evaluate("primary", workflow.chat, primary_config)
    fast = evaluate("fast_only", workflow.chat_fast_only, config)
    final_state_bytes = state_path.read_bytes()
    receipt = {
        "schema_version": "labelagent.single-candidate-fast-audit-ab.v1",
        "state": str(state_path),
        "state_sha256": sha256_bytes(state_bytes),
        "state_unchanged": final_state_bytes == state_bytes,
        "image_url": args.image_url,
        "image_sha256": sha256_bytes(image_bytes),
        "frame": args.frame,
        "source_candidate_id": args.candidate_id,
        "synthetic_candidate_id": args.synthetic_id,
        "candidate_label": candidate.get("label"),
        "primary": primary,
        "fast": fast,
        "semantic_core_match": primary["semantic_core"] == fast["semantic_core"],
        "speedup_percent": round(
            (primary["elapsed_seconds"] - fast["elapsed_seconds"])
            / max(primary["elapsed_seconds"], 0.001) * 100, 2),
        "candidate_only": True,
        "candidate_mutated": False,
        "human_truth": False,
        "approved_for_training": False,
        "formal_workbench_writes": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, separators=(",", ":")))
    return 0 if receipt["semantic_core_match"] and receipt["state_unchanged"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
