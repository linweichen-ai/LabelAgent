"""Run the authorized 34-source + 17-resized expert shadows safely."""
import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_mixed_hard_external_shadow import (
    load_authorized_inputs, normalize_expert_target, run_shadow, sha256_file,
)
from tools.external_authorization_ledger import (
    AuthorizationLedger, default_ledger_dir, inspect_ledger_session,
)
from source.agent.workflow import external_expert_config


SCHEMA_VERSION = "labelagent.combined-expert-shadow-run.v1"


def validate_execution_credential(config):
    """Validate the configured credential locally without returning its value."""
    prepared = external_expert_config(config)
    target = normalize_expert_target({
        "provider": prepared.get("llm_expert_provider"),
        "url": prepared.get("llm_url"),
        "model": prepared.get("llm_model"),
    })
    return {"credential_available": True, "expert_target": target}


def _attempts(path):
    path = Path(path)
    if not path.is_file():
        return 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("external_attempts")
    if type(value) is not int or value < 0:
        raise ValueError("existing external attempt count is invalid")
    return value


def _write_receipt(path, *, status, source_result, resized_result,
                   source_path, resized_path, prior_attempts,
                   canary_guard=None, authorization_ledger=None,
                   credential_preflight=None):
    source_attempts = source_result.get("external_attempts", 0)
    resized_attempts = (resized_result or {}).get("external_attempts", 0)
    total = source_attempts + resized_attempts
    if total > 51:
        raise ValueError("authorized external attempt budget exceeded")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "source_status": source_result.get("status"),
        "resized_status": ((resized_result or {}).get("status")
                           if resized_result is not None else "not_started"),
        "source_result_sha256": (sha256_file(source_path)
                                 if Path(source_path).is_file() else None),
        "resized_result_sha256": (sha256_file(resized_path)
                                  if Path(resized_path).is_file() else None),
        "external_attempts": total,
        "new_external_attempts": total - prior_attempts,
        "maximum_authorized_external_attempts": 51,
        "automatic_retry_enabled": False,
        "source_canary_guard": canary_guard,
        "authorization_ledger": authorization_ledger,
        "credential_preflight": credential_preflight,
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "automatic_truth_write": False,
        "test24_used": False,
        "production_deployed": False,
    }
    if payload["new_external_attempts"] < 0:
        raise ValueError("external attempt ledger regressed")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)
    return payload


def stratified_canary_order(samples, count=8):
    if type(count) is not int or not 1 <= count <= len(samples):
        raise ValueError("canary count is invalid")
    first_by_cohort, selected_ids = [], set()
    for sample in samples:
        cohort = sample.get("cohort")
        if cohort not in {item.get("cohort") for item in first_by_cohort}:
            first_by_cohort.append(sample)
            selected_ids.add(sample["sample_id"])
        if len(first_by_cohort) == count:
            break
    selected = first_by_cohort[:count]
    if len(selected) < count:
        for sample in samples:
            if sample["sample_id"] not in selected_ids:
                selected.append(sample)
                selected_ids.add(sample["sample_id"])
                if len(selected) == count:
                    break
    return selected + [sample for sample in samples
                       if sample["sample_id"] not in selected_ids]


def canary_guard(result, *, expected_attempts, max_median_latency_ms,
                 max_average_tokens):
    usage = result.get("external_usage") or {}
    passed = result.get("samples_passed")
    reasons = []
    if result.get("status") not in {"in_progress", "completed"}:
        reasons.append("canary_has_failed_request")
    if result.get("external_attempts") != expected_attempts or passed != expected_attempts:
        reasons.append("canary_attempt_count_differs")
    if usage.get("complete") is not True or usage.get("samples_with_usage") != passed:
        reasons.append("canary_usage_incomplete")
    median = result.get("latency_median_ms")
    if not isinstance(median, (int, float)) or median > max_median_latency_ms:
        reasons.append("canary_latency_exceeds_guard")
    total = usage.get("total_tokens")
    average = (total / passed if type(total) is int and type(passed) is int
               and passed > 0 else None)
    if average is None or average > max_average_tokens:
        reasons.append("canary_tokens_exceed_guard")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "attempts": result.get("external_attempts"),
        "latency_median_ms": median,
        "average_tokens_per_call": round(average, 3) if average is not None else None,
        "maximum_median_latency_ms": max_median_latency_ms,
        "maximum_average_tokens_per_call": max_average_tokens,
    }


def preflight_combined(*, all_queue, hybrid_queue, manifest, receipt,
                       image_root, authorization, output_dir, config,
                       ledger_dir=None, load_fn=load_authorized_inputs,
                       credential_preflight_fn=validate_execution_credential):
    """Validate the complete combined run without claims or network activity."""
    source_samples, source_bindings = load_fn(
        all_queue, manifest, receipt, image_root, authorization,
        authorization_run_name="all34_source_640x512")
    resized_samples, resized_bindings = load_fn(
        hybrid_queue, manifest, receipt, image_root, authorization,
        "resize_512x410", "hybrid17_resize_512x410")
    if (len(source_samples) != 34 or len(resized_samples) != 17
            or source_bindings.get("authorization_schema")
            != "labelagent.external-upload-authorization.v4"
            or source_bindings.get("authorization_sha256")
            != resized_bindings.get("authorization_sha256")
            or source_bindings.get("authorized_upload_count") != 51
            or resized_bindings.get("authorized_upload_count") != 51):
        raise ValueError("combined authorization binding differs")
    try:
        configured_target = normalize_expert_target({
            "provider": config.get("llm_expert_provider"),
            "url": config.get("llm_expert_url"),
            "model": config.get("llm_expert_model"),
        })
    except ValueError as error:
        raise ValueError("combined authorization target differs") from error
    if (source_bindings.get("authorized_expert_target") != configured_target
            or resized_bindings.get("authorized_expert_target")
            != configured_target):
        raise ValueError("combined authorization target differs")
    if config.get("llm_expert_retry_count", 0) != 0:
        raise ValueError("combined shadow forbids automatic API retries")
    credential_preflight = credential_preflight_fn(config)
    if (not isinstance(credential_preflight, dict)
            or credential_preflight.get("credential_available") is not True
            or credential_preflight.get("expert_target") != configured_target):
        raise ValueError("external expert credential preflight differs")
    authorization_sha = source_bindings["authorization_sha256"]
    ledger_status = inspect_ledger_session(
        ledger_dir or default_ledger_dir(),
        authorization_sha256=authorization_sha, output_dir=output_dir)
    return {
        "source_samples": source_samples,
        "source_bindings": source_bindings,
        "resized_samples": resized_samples,
        "resized_bindings": resized_bindings,
        "credential_preflight": credential_preflight,
        "ledger_status": ledger_status,
        "summary": {
            "status": "ready",
            "authorization_schema": source_bindings["authorization_schema"],
            "authorization_sha256": authorization_sha,
            "source_sample_count": len(source_samples),
            "resized_sample_count": len(resized_samples),
            "maximum_external_calls": 51,
            "expert_target": configured_target,
            "credential_available": True,
            "ledger_state": ledger_status["state"],
            "existing_claimed_attempts": ledger_status["claimed_attempts"],
            "ledger_claim_created": False,
            "external_call_performed": False,
            "candidate_only": True,
            "human_truth": False,
            "approved_for_training": False,
            "test24_used": False,
            "production_deployed": False,
        },
    }


def run_combined(*, all_queue, hybrid_queue, manifest, receipt, image_root,
                 authorization, output_dir, config,
                 ledger_dir=None,
                 source_canary_attempts=8,
                 canary_max_median_latency_ms=60000,
                 canary_max_average_tokens=30000,
                 load_fn=load_authorized_inputs, run_fn=run_shadow,
                 credential_preflight_fn=validate_execution_credential):
    output_dir = Path(output_dir)
    source_path = output_dir / "all34-source-results.json"
    resized_path = output_dir / "hybrid17-resized-results.json"
    run_receipt_path = output_dir / "combined-run-receipt.json"
    preflight = preflight_combined(
        all_queue=all_queue, hybrid_queue=hybrid_queue,
        manifest=manifest, receipt=receipt, image_root=image_root,
        authorization=authorization, output_dir=output_dir, config=config,
        ledger_dir=ledger_dir, load_fn=load_fn,
        credential_preflight_fn=credential_preflight_fn)
    source_samples = preflight["source_samples"]
    source_bindings = preflight["source_bindings"]
    resized_samples = preflight["resized_samples"]
    resized_bindings = preflight["resized_bindings"]
    credential_preflight = preflight["credential_preflight"]
    authorization_sha = source_bindings["authorization_sha256"]
    ledger = AuthorizationLedger(
        ledger_dir or default_ledger_dir(),
        authorization_sha256=authorization_sha, output_dir=output_dir)
    source_claim = lambda sample, upload: ledger.claim_attempt(
        "all34_source_640x512", sample, upload)
    resized_claim = lambda sample, upload: ledger.claim_attempt(
        "hybrid17_resize_512x410", sample, upload)
    prior_source_attempts = _attempts(source_path)
    prior_attempts = prior_source_attempts + _attempts(resized_path)
    source_samples = stratified_canary_order(
        source_samples, source_canary_attempts)
    source = run_fn(
        samples=source_samples, bindings=source_bindings,
        output_path=source_path, config=config,
        claim_attempt_fn=source_claim,
        max_new_attempts=(source_canary_attempts
                          if prior_source_attempts == 0 else 0))
    if source.get("status") == "partial":
        return _write_receipt(
            run_receipt_path, status="source_canary_failed_resized_not_started",
            source_result=source, resized_result=None,
            source_path=source_path, resized_path=resized_path,
            prior_attempts=prior_attempts,
            canary_guard={"passed": False,
                          "reasons": ["canary_has_failed_request"]},
            authorization_ledger=ledger.snapshot(),
            credential_preflight=credential_preflight)
    expected_canary = (source_canary_attempts if prior_source_attempts == 0
                       else prior_source_attempts)
    guard = canary_guard(
        source, expected_attempts=expected_canary,
        max_median_latency_ms=canary_max_median_latency_ms,
        max_average_tokens=canary_max_average_tokens)
    if not guard["passed"]:
        return _write_receipt(
            run_receipt_path, status="source_canary_guard_failed",
            source_result=source, resized_result=None,
            source_path=source_path, resized_path=resized_path,
            prior_attempts=prior_attempts, canary_guard=guard,
            authorization_ledger=ledger.snapshot(),
            credential_preflight=credential_preflight)
    if source.get("status") == "in_progress":
        source = run_fn(samples=source_samples, bindings=source_bindings,
                        output_path=source_path, config=config,
                        claim_attempt_fn=source_claim)
    if source.get("status") != "completed":
        return _write_receipt(
            run_receipt_path, status="source_incomplete_resized_not_started",
            source_result=source, resized_result=None,
            source_path=source_path, resized_path=resized_path,
            prior_attempts=prior_attempts, canary_guard=guard,
            authorization_ledger=ledger.snapshot(),
            credential_preflight=credential_preflight)
    resized = run_fn(samples=resized_samples, bindings=resized_bindings,
                     output_path=resized_path, config=config,
                     claim_attempt_fn=resized_claim)
    status = ("completed" if resized.get("status") == "completed"
              else "resized_incomplete")
    return _write_receipt(
        run_receipt_path, status=status, source_result=source,
        resized_result=resized, source_path=source_path,
        resized_path=resized_path, prior_attempts=prior_attempts,
        canary_guard=guard, authorization_ledger=ledger.snapshot(),
        credential_preflight=credential_preflight)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-queue", type=Path, required=True)
    parser.add_argument("--hybrid-queue", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ledger-dir", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--key-env", required=True)
    parser.add_argument("--source-canary-attempts", type=int, default=8)
    parser.add_argument("--canary-max-median-latency-ms", type=float,
                        default=60000)
    parser.add_argument("--canary-max-average-tokens", type=float,
                        default=30000)
    args = parser.parse_args(argv)
    config = {"llm_expert_enabled": True,
              "llm_expert_provider": "openai_chat",
              "llm_expert_url": args.url,
              "llm_expert_model": args.model,
              "llm_expert_key_env": args.key_env,
              "llm_expert_timeout_seconds": 120,
              "llm_expert_retry_count": 0}
    common = dict(
        all_queue=args.all_queue, hybrid_queue=args.hybrid_queue,
        manifest=args.manifest, receipt=args.receipt,
        image_root=args.image_root, authorization=args.authorization,
        output_dir=args.output_dir,
        ledger_dir=args.ledger_dir,
        config=config)
    try:
        if args.preflight_only:
            result = preflight_combined(**common)["summary"]
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        result = run_combined(
            **common,
            source_canary_attempts=args.source_canary_attempts,
            canary_max_median_latency_ms=args.canary_max_median_latency_ms,
            canary_max_average_tokens=args.canary_max_average_tokens)
    except (ValueError, OSError, json.JSONDecodeError) as error:
        print(json.dumps({
            "status": "refused",
            "error_type": type(error).__name__,
            "error": str(error)[:500],
            "external_call_performed": False,
            "ledger_claim_created": False,
            "candidate_only": True,
            "human_truth": False,
            "approved_for_training": False,
            "test24_used": False,
            "production_deployed": False,
        }, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
