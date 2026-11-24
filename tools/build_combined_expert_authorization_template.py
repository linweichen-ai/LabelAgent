"""Build a non-authorizing template that reuses all34 source results for A/B."""
import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_mixed_hard_external_shadow import (
    QUEUE_SCHEMA, normalize_expert_target, sha256_file,
)


SCHEMA_VERSION = "labelagent.external-upload-authorization.v4"
SOURCE_VARIANT = {"name": "source_640x512", "mode": "source",
                  "expected_width": 640, "expected_height": 512}
RESIZED_VARIANT = {"name": "resize_512x410",
                   "mode": "resize_lanczos_jpeg", "width": 512,
                   "height": 410, "jpeg_quality": 90}


def _queue(path, expected_count):
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples") if isinstance(payload, dict) else None
    if (payload.get("schema_version") != QUEUE_SCHEMA
            or not isinstance(samples, list) or len(samples) != expected_count
            or payload.get("sample_count") != expected_count
            or payload.get("candidate_only") is not True
            or payload.get("human_truth") is not False
            or payload.get("approved_for_training") is not False
            or payload.get("test24_used") is not False
            or payload.get("images_copied") is not False
            or payload.get("image_paths_included") is not False):
        raise ValueError("expert queue identity or safety boundary differs")
    ids = {row.get("sample_id") for row in samples}
    if len(ids) != expected_count or None in ids:
        raise ValueError("expert queue sample identities differ")
    return payload, ids, sha256_file(path)


def build_template(all_queue_path, hybrid_queue_path, manifest_path,
                   receipt_path, *, provider, url, model):
    _, all_ids, all_sha = _queue(all_queue_path, 34)
    _, hybrid_ids, hybrid_sha = _queue(hybrid_queue_path, 17)
    if not hybrid_ids.issubset(all_ids):
        raise ValueError("hybrid17 queue is not a subset of all34 queue")
    target = normalize_expert_target({
        "provider": provider, "url": url, "model": model})
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "awaiting_explicit_user_authorization",
        "scope": "mixed_hard_combined_external_shadow",
        "authorized": False, "authorized_by": "",
        "authorized_unique_source_samples": 34,
        "authorized_upload_count": 51,
        "max_external_calls": 51,
        "review_manifest_sha256": sha256_file(manifest_path),
        "review_receipt_sha256": sha256_file(receipt_path),
        "expert_target": target,
        "authorized_runs": [
            {"name": "all34_source_640x512", "queue_sha256": all_sha,
             "sample_count": 34, "image_variant": SOURCE_VARIANT,
             "max_external_calls": 34},
            {"name": "hybrid17_resize_512x410", "queue_sha256": hybrid_sha,
             "sample_count": 17, "image_variant": RESIZED_VARIANT,
             "max_external_calls": 17},
        ],
        "reuse_contract": {
            "source17_for_resolution_ab": "subset of all34_source_640x512",
            "duplicate_source_calls_avoided": 17,
            "separate_plan_calls": 68,
            "combined_plan_calls": 51,
        },
        "candidate_only": True, "human_truth": False,
        "approved_for_training": False, "test24_used": False,
        "automatic_truth_write": False, "production_deployed": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-queue", type=Path, required=True)
    parser.add_argument("--hybrid-queue", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--provider", default="openai_chat")
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_template(
        args.all_queue, args.hybrid_queue, args.manifest, args.receipt,
        provider=args.provider, url=args.url, model=args.model)
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(args.output)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

