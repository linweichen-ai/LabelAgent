"""Freeze a path-free, candidate-only pre-A/B queue from a review packet."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


SCHEMA_VERSION = "labelagent.external-expert-preab-manifest.v1"


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def build_preab_manifest(source, *, source_bytes=None):
    if source_bytes is None:
        if not isinstance(source, (str, Path)):
            raise ValueError("source path is required")
        path = Path(source).resolve()
        if not path.is_file():
            raise ValueError("source review manifest does not exist")
        source_bytes = path.read_bytes()
    if not isinstance(source_bytes, bytes):
        raise ValueError("source manifest bytes are required")
    payload = json.loads(source_bytes.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("source review manifest must be an object")
    serialized = json.dumps(payload, ensure_ascii=False).casefold()
    if "test24" in serialized:
        raise ValueError("test24 is evaluation-only and cannot enter pre-A/B queue")
    dataset_sha = payload.get("mcap_sha256")
    topic, rows = payload.get("topic"), payload.get("rows")
    if (not isinstance(dataset_sha, str) or len(dataset_sha) != 64
            or not isinstance(topic, str) or not topic.strip()
            or not isinstance(rows, list) or not rows or len(rows) > 500):
        raise ValueError("source review manifest identity is invalid")
    samples, seen, strata = [], set(), Counter()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("source review row must be an object")
        frame, frame_sha, stratum = (
            row.get("frame"), row.get("source_frame_sha256"), row.get("cohort"))
        if (type(frame) is not int or frame < 0 or not isinstance(frame_sha, str)
                or len(frame_sha) != 64 or not isinstance(stratum, str)
                or not stratum.strip() or frame in seen):
            raise ValueError("source review row identity is invalid")
        if (row.get("candidate_only") is not True
                or row.get("human_truth") is not False
                or row.get("approved_for_training") is not False
                or row.get("training_return_allowed") is not False):
            raise ValueError("pre-A/B row violates candidate-only boundaries")
        if row.get("review_status") != "pending_human_review":
            raise ValueError("pre-A/B queue accepts pending human review only")
        seen.add(frame)
        stratum = stratum.strip()
        strata[stratum] += 1
        sample_id = _sha256_bytes(
            f"{dataset_sha}\n{topic.strip()}\n{frame}\n{frame_sha}".encode())
        samples.append({
            "sample_id": sample_id,
            "frame": frame,
            "stratum": stratum,
            "source_frame_sha256": frame_sha,
            "candidate_count": int(row.get("candidate_count") or 0),
            "target_provenance": str(row.get("target_provenance") or "unknown")[:80],
            "review_status": "pending_human_review",
            "human_reference_available": False,
            "external_upload_authorized": False,
        })
    samples.sort(key=lambda item: (item["stratum"], item["frame"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "source_manifest_sha256": _sha256_bytes(source_bytes),
        "source_dataset_sha256": dataset_sha,
        "topic": topic.strip(),
        "sample_count": len(samples),
        "strata": dict(sorted(strata.items())),
        "pending_human_review": len(samples),
        "human_reference_count": 0,
        "ready_for_quality_ab": False,
        "external_upload_authorized": False,
        "images_copied": False,
        "image_paths_included": False,
        "test24_used": False,
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "samples": samples,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build a path-free candidate-only external expert pre-A/B queue")
    parser.add_argument("--source", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build_preab_manifest(args.source), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
