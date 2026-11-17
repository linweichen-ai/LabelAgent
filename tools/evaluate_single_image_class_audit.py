"""Evaluate bounded blind-audit latency variants against frozen baselines.

This is an isolated experiment.  It reads frozen candidates and model settings,
performs VLM calls, and writes only an explicit artifact receipt.  It never
updates workbench state, candidates, truth, or training approval.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from pathlib import Path

from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from source.agent import candidate_class_audit as audit
from source.agent import workflow


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence_card(original_bytes: bytes, marked_bytes: bytes) -> bytes:
    with Image.open(io.BytesIO(original_bytes)) as opened:
        original = opened.convert("RGB")
    with Image.open(io.BytesIO(marked_bytes)) as opened:
        marked = opened.convert("RGB")
    if original.size != marked.size:
        raise ValueError("production audit crops must have identical sizes")
    card = Image.new("RGB", (original.width * 2, original.height), (127, 127, 127))
    card.paste(original, (0, 0))
    card.paste(marked, (original.width, 0))
    output = io.BytesIO()
    card.save(output, "JPEG", quality=92, optimize=False, progressive=False)
    return output.getvalue()


def card_prompt(candidate_id, labels, definitions):
    prompt = audit.build_blind_candidate_class_prompt(
        candidate_id, labels, definitions)
    original = (
        "Image 1 is an uncolored local crop. Image 2 is the same crop with only "
        "the candidate mask highlighted in cyan; holes are excluded."
    )
    replacement = (
        "The single image is a side-by-side evidence card. Its left panel is the "
        "uncolored local crop and its right panel is the same crop with only the "
        "candidate mask highlighted in cyan; holes are excluded."
    )
    if original not in prompt:
        raise RuntimeError("production blind-audit prompt contract changed")
    return prompt.replace(original, replacement, 1)


def semantic_core(decision, route):
    return {
        "classification": decision.get("classification"),
        "observed_label": decision.get("observed_label"),
        "candidate_supported": decision.get("candidate_supported"),
        "reason_code": decision.get("reason_code"),
        "route": route,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--candidate-json", required=True, type=Path)
    parser.add_argument("--candidate-ids", required=True, nargs="+", type=int)
    parser.add_argument(
        "--variant", choices=("lossless_side_by_side_card", "production_two_image_384"),
        default="lossless_side_by_side_card")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    artifact_root = (REPOSITORY_ROOT / "artifacts").resolve()
    state_path = args.state.resolve()
    image_path = args.image.resolve()
    candidate_path = args.candidate_json.resolve()
    output_path = args.output.resolve()
    if not output_path.is_relative_to(artifact_root):
        raise ValueError("output must be inside this repository's artifacts directory")
    if output_path.exists():
        raise FileExistsError(output_path)
    for path in (state_path, image_path, candidate_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    state_sha_before = sha256(state_path)
    image_sha_before = sha256(image_path)
    candidate_sha_before = sha256(candidate_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    frozen = json.loads(candidate_path.read_text(encoding="utf-8"))
    config = dict(state.get("config") or {})
    labels = list(state.get("labels") or [])
    if not config.get("llm_url") or not config.get("llm_model") or not labels:
        raise ValueError("state lacks model settings or labels")
    image_bytes = image_path.read_bytes()
    candidates = {item.get("id"): item for item in frozen.get("candidates") or []}
    baselines = {
        item.get("candidate_id"): item
        for item in (frozen.get("agent") or {}).get("candidate_class_audits") or []
    }
    if set(args.candidate_ids) - candidates.keys():
        raise ValueError("requested candidate is missing")
    if set(args.candidate_ids) - baselines.keys():
        raise ValueError("requested candidate lacks a production baseline audit")

    rows = []
    for candidate_id in args.candidate_ids:
        candidate = candidates[candidate_id]
        if candidate.get("label") not in audit.HIGH_RISK_LABELS:
            raise ValueError("experiment is restricted to production high-risk labels")
        evidence = audit.prepare_candidate_evidence(image_bytes, candidate)
        if args.variant == "lossless_side_by_side_card":
            request_image = evidence_card(evidence["original"], evidence["marked"])
            extra_images = ()
            prompt = card_prompt(candidate_id, labels, workflow.VISIBLE_LABEL_DEFINITIONS)
            request_config = config
        else:
            request_image = evidence["original"]
            extra_images = (evidence["marked"],)
            prompt = audit.build_blind_candidate_class_prompt(
                candidate_id, labels, workflow.VISIBLE_LABEL_DEFINITIONS)
            request_config = dict(config)
            request_config["llm_max_tokens"] = 384
        raw = {}
        started = time.perf_counter()
        decision = audit.validate_blind_candidate_class_decision(
            workflow._chat(
                request_config, prompt, request_image, extra_images=extra_images, raw=raw,
                response_schema=audit.blind_candidate_class_schema(labels)),
            labels,
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        route = (
            "candidate_preserved"
            if decision["classification"] == "known_label"
            and decision["observed_label"] == candidate["label"]
            and decision["confidence"] >= 0.90
            else "human_review_required"
        )
        baseline = baselines[candidate_id]
        baseline_core = semantic_core(baseline.get("decision") or {}, baseline.get("route"))
        candidate_core = semantic_core(decision, route)
        local = raw.get("local_vlm") or {}
        stats = local.get("inference_stats") or {}
        rows.append({
            "candidate_id": candidate_id,
            "bound_current_label": candidate.get("label"),
            "candidate_fingerprint": audit.candidate_fingerprint(candidate),
            "baseline_semantic_core": baseline_core,
            "candidate_semantic_core": candidate_core,
            "semantic_core_match": baseline_core == candidate_core,
            "elapsed_ms": elapsed_ms,
            "finish_reason": raw.get("finish_reason"),
            "model_route": raw.get("model_route"),
            "cache_hit": local.get("cache_hit"),
            "image_count": local.get("image_count"),
            "actual_image_token_counts": local.get("actual_image_token_counts"),
            "prompt_tokens": stats.get("prompt_tokens"),
            "completion_tokens": stats.get("completion_tokens"),
            "generation_seconds": stats.get("generation_seconds"),
            "generation_cache_implementation": stats.get(
                "generation_cache_implementation"),
            "request_image_sha256": hashlib.sha256(request_image).hexdigest(),
            "extra_image_sha256": [hashlib.sha256(item).hexdigest()
                                   for item in extra_images],
            "max_tokens": request_config.get("llm_max_tokens") or 768,
            "candidate_only": True,
            "human_truth": False,
            "approved_for_training": False,
        })

    expected_image_count = 1 if args.variant == "lossless_side_by_side_card" else 2
    result = {
        "schema_version": "labelagent.class-audit-latency-experiment.v1",
        "variant": args.variant,
        "state": str(state_path),
        "state_sha256": state_sha_before,
        "image": str(image_path),
        "image_sha256": image_sha_before,
        "candidate_json": str(candidate_path),
        "candidate_json_sha256": candidate_sha_before,
        "rows": rows,
        "summary": {
            "candidate_count": len(rows),
            "semantic_core_matches": sum(row["semantic_core_match"] for row in rows),
            "all_semantic_cores_match": all(row["semantic_core_match"] for row in rows),
            "expected_image_count": expected_image_count,
            "all_image_counts_match_variant": all(
                row["image_count"] == expected_image_count for row in rows),
            "total_elapsed_ms": round(sum(row["elapsed_ms"] for row in rows), 3),
            "all_candidate_only": all(row["candidate_only"] for row in rows),
            "all_human_truth_false": all(not row["human_truth"] for row in rows),
            "all_training_approval_false": all(
                not row["approved_for_training"] for row in rows),
        },
        "inputs_unchanged": {
            "state": sha256(state_path) == state_sha_before,
            "image": sha256(image_path) == image_sha_before,
            "candidate_json": sha256(candidate_path) == candidate_sha_before,
        },
        "formal_workbench_writes": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
