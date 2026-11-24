"""Run the authorized 34-source + 17-resized expert shadows safely."""
import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_mixed_hard_external_shadow import (
    load_authorized_inputs, run_shadow, sha256_file,
)


SCHEMA_VERSION = "labelagent.combined-expert-shadow-run.v1"


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
                   source_path, resized_path, prior_attempts):
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


def run_combined(*, all_queue, hybrid_queue, manifest, receipt, image_root,
                 authorization, output_dir, config,
                 load_fn=load_authorized_inputs, run_fn=run_shadow):
    output_dir = Path(output_dir)
    source_path = output_dir / "all34-source-results.json"
    resized_path = output_dir / "hybrid17-resized-results.json"
    run_receipt_path = output_dir / "combined-run-receipt.json"

    # Validate both complete upload sets before the first network request.
    source_samples, source_bindings = load_fn(
        all_queue, manifest, receipt, image_root, authorization,
        authorization_run_name="all34_source_640x512")
    resized_samples, resized_bindings = load_fn(
        hybrid_queue, manifest, receipt, image_root, authorization,
        "resize_512x410", "hybrid17_resize_512x410")
    if (len(source_samples) != 34 or len(resized_samples) != 17
            or source_bindings.get("authorization_schema")
            != "labelagent.external-upload-authorization.v3"
            or source_bindings.get("authorization_sha256")
            != resized_bindings.get("authorization_sha256")
            or source_bindings.get("authorized_upload_count") != 51
            or resized_bindings.get("authorized_upload_count") != 51):
        raise ValueError("combined authorization binding differs")
    if config.get("llm_expert_retry_count", 0) != 0:
        raise ValueError("combined shadow forbids automatic API retries")
    prior_attempts = _attempts(source_path) + _attempts(resized_path)
    source = run_fn(samples=source_samples, bindings=source_bindings,
                    output_path=source_path, config=config)
    if source.get("status") != "completed":
        return _write_receipt(
            run_receipt_path, status="source_incomplete_resized_not_started",
            source_result=source, resized_result=None,
            source_path=source_path, resized_path=resized_path,
            prior_attempts=prior_attempts)
    resized = run_fn(samples=resized_samples, bindings=resized_bindings,
                     output_path=resized_path, config=config)
    status = ("completed" if resized.get("status") == "completed"
              else "resized_incomplete")
    return _write_receipt(
        run_receipt_path, status=status, source_result=source,
        resized_result=resized, source_path=source_path,
        resized_path=resized_path, prior_attempts=prior_attempts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-queue", type=Path, required=True)
    parser.add_argument("--hybrid-queue", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--key-env", required=True)
    args = parser.parse_args(argv)
    result = run_combined(
        all_queue=args.all_queue, hybrid_queue=args.hybrid_queue,
        manifest=args.manifest, receipt=args.receipt,
        image_root=args.image_root, authorization=args.authorization,
        output_dir=args.output_dir,
        config={"llm_expert_enabled": True,
                "llm_expert_provider": "openai_chat",
                "llm_expert_url": args.url,
                "llm_expert_model": args.model,
                "llm_expert_key_env": args.key_env,
                "llm_expert_timeout_seconds": 120,
                "llm_expert_retry_count": 0})
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
