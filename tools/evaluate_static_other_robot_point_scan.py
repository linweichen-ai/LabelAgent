"""Run candidate-only reference point scanning on static cross-sequence images."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from tools.evaluate_reference_point_scan import evaluate_frame


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_manifest(manifest: dict) -> list[dict]:
    samples = manifest.get("samples")
    if (manifest.get("schema_version")
            != "labelagent.static-other-robot-point-scan.v1"
            or manifest.get("human_truth") is not False
            or manifest.get("candidate_only") is not True
            or manifest.get("production_deployed") is not False
            or manifest.get("approved_for_training") is not False
            or not isinstance(samples, list) or not samples):
        raise ValueError("static point-scan manifest boundary differs")
    identifiers = []
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("static point-scan sample boundary differs")
        review = sample.get("assistant_review_candidate")
        image_sha256 = sample.get("image_sha256")
        if (not sample.get("sample_id")
                or not sample.get("image_path") or not sample.get("image_sha256")
                or len(image_sha256) != 64
                or any(character not in "0123456789abcdefABCDEF"
                       for character in image_sha256)
                or not isinstance(review, dict)
                or review.get("review_kind") != "assistant_visual_candidate"
                or review.get("human_truth") is not False
                or review.get("expected_evidence") not in {"present", "absent"}
                or not review.get("reason")):
            raise ValueError("static point-scan sample boundary differs")
        identifiers.append(sample["sample_id"])
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("static point-scan sample IDs must be unique")
    return samples


def safe_sample_dir(sample_id: str) -> str:
    if not sample_id or any(not (character.isalnum() or character in "-_")
                            for character in sample_id):
        raise ValueError("sample ID is unsafe for an output directory")
    return sample_id


def summarize_frame(sample: dict, report: dict) -> dict:
    selected_ids = report["selected_ids"]
    review_ids = report["uncertain_review_ids"]
    effective_evidence = "present" if selected_ids else "absent"
    triage_status = ("auto_candidate_present" if selected_ids
                     else "review_required" if review_ids
                     else "no_candidate")
    review = sample["assistant_review_candidate"]
    return {
        "sample_id": sample["sample_id"],
        "source_ref": sample.get("source_ref"),
        "human_truth": False,
        "candidate_only": True,
        "production_deployed": False,
        "approved_for_training": False,
        "assistant_review_candidate": review,
        "effective_evidence": effective_evidence,
        "assistant_candidate_agreement": (
            effective_evidence == review["expected_evidence"]),
        "triage_status": triage_status,
        "assistant_candidate_review_covered": (
            bool(selected_ids or review_ids)
            if review["expected_evidence"] == "present" else None),
        "selected_ids": selected_ids,
        "uncertain_review_ids": review_ids,
        "foreground_ego_pair_veto_ids": report["foreground_ego_pair_veto_ids"],
        "preclassification_foreground_ego_pair_veto": (
            report["preclassification_foreground_ego_pair_veto"]),
        "classification_disagreement_review_ids": (
            report["classification_disagreement_review_ids"]),
        "point_count": report["point_count"],
        "scan_seconds": report["scan_seconds"],
        "raw_candidates": report["raw_candidates"],
        "geometric_candidates": report["geometric_candidates"],
        "nms_candidates": report["nms_candidates"],
        "known_exclusion_passed": report["known_exclusion_passed"],
        "classified_count": report["classified_count"],
        "classification_batch_seconds": report["classification_batch_seconds"],
        "surface_veto_batch_seconds": report["surface_veto_batch_seconds"],
        "rows": report["rows"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--point-sam-url", default="http://127.0.0.1:28766")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--exemplar-sheet", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 1 <= args.top_k <= 24:
        parser.error("--top-k must be within 1..24")
    state = json.loads(args.state.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    samples = verify_manifest(manifest)
    config = copy.deepcopy(state.get("config") or {})
    if not config.get("llm_url") or not config.get("sam3_url"):
        raise ValueError("state config lacks the formal model endpoints")
    exemplar_sheet = (args.exemplar_sheet.read_bytes()
                      if args.exemplar_sheet else None)
    verified_samples = []
    for sample in samples:
        image_path = Path(sample["image_path"])
        if (not image_path.is_file()
                or file_sha256(image_path) != sample["image_sha256"].lower()):
            raise ValueError(f"static image hash differs: {sample['sample_id']}")
        verified_samples.append((sample, image_path))
    args.output.mkdir(parents=True)
    rows = []
    for sample, image_path in verified_samples:
        sample_output = args.output / safe_sample_dir(sample["sample_id"])
        sample_output.mkdir()
        report = evaluate_frame(
            sample["sample_id"], image_path.read_bytes(), [], config,
            args.point_sam_url, sample_output, args.top_k,
            exemplar_sheet=exemplar_sheet)
        row = summarize_frame(sample, report)
        rows.append(row)
        print(json.dumps({key: row[key] for key in (
            "sample_id", "effective_evidence", "assistant_candidate_agreement",
            "selected_ids", "uncertain_review_ids", "scan_seconds")},
            ensure_ascii=False), flush=True)
    reviewed = len(rows)
    positive_rows = [row for row in rows
                     if row["assistant_review_candidate"]["expected_evidence"] == "present"]
    negative_rows = [row for row in rows
                     if row["assistant_review_candidate"]["expected_evidence"] == "absent"]
    result = {
        "schema_version": "labelagent.static-other-robot-point-scan-result.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "candidate_shadow_evaluated",
        "human_truth": False,
        "candidate_only": True,
        "production_deployed": False,
        "approved_for_training": False,
        "accuracy_computed": False,
        "state_written": False,
        "manifest_sha256": file_sha256(args.manifest),
        "exemplar_sheet_sha256": (
            file_sha256(args.exemplar_sheet) if args.exemplar_sheet else None),
        "top_k": args.top_k,
        "assistant_candidate_review": {
            "reviewed": reviewed,
            "agreements": sum(row["assistant_candidate_agreement"] for row in rows),
            "disagreements": sum(not row["assistant_candidate_agreement"] for row in rows),
            "is_accuracy": False,
        },
        "assistant_candidate_triage": {
            "positive_reviewed": len(positive_rows),
            "positive_covered_by_auto_or_review": sum(
                bool(row["selected_ids"] or row["uncertain_review_ids"])
                for row in positive_rows),
            "negative_reviewed": len(negative_rows),
            "negative_with_auto_candidate": sum(
                bool(row["selected_ids"]) for row in negative_rows),
            "negative_with_review_candidate": sum(
                bool(row["uncertain_review_ids"]) for row in negative_rows),
            "is_accuracy": False,
        },
        "rows": rows,
        "boundary": (
            "Point-scan outputs and assistant review are candidate-only triage, not "
            "human truth, accuracy, or production acceptance."),
    }
    (args.output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
