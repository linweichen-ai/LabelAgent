"""Prepare a reviewer-confirmed three-arm input skeleton without running models."""
import argparse
import json
from pathlib import Path

from source.agent.three_arm_benchmark import INPUT_SCHEMA, LABELS
from tools.build_mixed_hard_adjudication_workspace import (
    read_json, validate_packet, validate_submission,
)


def build_reference(packet_dir, submission):
    manifest, digest = validate_packet(packet_dir)
    result = validate_submission(manifest, digest, submission)
    if not result["benchmark_reference_ready"]:
        raise ValueError("review contains uncertain or legacy fields and is not benchmark-ready")
    rows = {row["sample_id"]: row for row in submission["rows"]}
    samples = []
    for source in manifest["rows"]:
        row = rows[source["sample_id"]]
        presence = row["class_presence"]
        if set(presence) != set(LABELS):
            raise ValueError(f"class presence coverage differs: {source['sample_id']}")
        samples.append({
            "sample_id": source["sample_id"],
            "source_image_sha256": source["source_image_sha256"],
            "class_presence": {label: presence[label] for label in LABELS},
            "pallet_instance_count": row["pallet_instance_count"],
        })
    return {
        "schema_version": INPUT_SCHEMA,
        "partition": "mixed_hard_reviewer_confirmed_dev",
        "test24": False,
        "synthetic_fixture": False,
        "human_reference": True,
        "reviewer_confirmed": True,
        "reference_scope": "declared_human_class_presence_and_pallet_instance_count",
        "pixel_truth": False,
        "review_manifest_sha256": digest,
        "reviewer_declared_id": submission["reviewer_declared_id"],
        "samples": samples,
        "runs": [],
        "candidate_only": True,
        "human_truth_written": False,
        "approved_for_training": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = build_reference(args.packet, read_json(args.submission))
    Path(args.output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ready_for_runs": True, "sample_count": len(payload["samples"]),
                      "output": str(Path(args.output).resolve()),
                      "approved_for_training": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
