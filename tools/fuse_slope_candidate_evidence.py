"""Fuse bounded VLM and LiDAR-to-SAM slope evidence for offline review.

The output is a machine candidate receipt, never human truth or deployment
approval.  Geometry may rescue a VLM miss only when a hash-bound SAM result
exists for the same image and contains a non-empty mask.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def fuse(vlm_score: dict, geometry: dict, sam: dict) -> dict:
    for payload, schema in (
            (vlm_score, "labelagent.qwen3vl-slope-review-score.v1"),
            (geometry, "labelagent.slope-lidar-geometry.v1"),
            (sam, "labelagent.slope-lidar-sam.v1")):
        if (payload.get("schema_version") != schema
                or payload.get("human_truth") is not False
                or payload.get("human_confirmed") is not False
                or payload.get("production_deployed") is not False):
            raise ValueError(f"unsafe or unsupported evidence: {schema}")

    vlm_rows = vlm_score.get("summary", {}).get("details")
    geometry_rows = geometry.get("images")
    sam_rows = sam.get("rows")
    if not all(isinstance(rows, list) for rows in (vlm_rows, geometry_rows, sam_rows)):
        raise ValueError("evidence rows are missing")

    def unique(rows, name):
        result = {}
        for row in rows:
            sample_id = row.get("sample_id")
            if not sample_id or sample_id in result:
                raise ValueError(f"{name} sample ids are missing or duplicated")
            result[sample_id] = row
        return result

    vlm_by_id = unique(vlm_rows, "VLM")
    geometry_by_id = unique(geometry_rows, "geometry")
    sam_by_id = unique(sam_rows, "SAM")
    if set(vlm_by_id) != set(geometry_by_id) or len(vlm_by_id) != 8:
        raise ValueError("VLM and geometry sample sets differ")

    details = []
    tp = fp = fn = tn = 0
    for sample_id in sorted(vlm_by_id):
        vlm_row = vlm_by_id[sample_id]
        geometry_row = geometry_by_id[sample_id]
        visible = vlm_row.get("assistant_visible_slope_candidate")
        evidence = vlm_row.get("candidate_evidence")
        if not isinstance(visible, bool) or evidence not in {
                "present", "absent", "uncertain"}:
            raise ValueError("invalid bounded VLM review row")
        hint_count = geometry_row.get("generated_hint_count")
        if not isinstance(hint_count, int) or hint_count < 0:
            raise ValueError("invalid geometry hint count")
        geometry_supported = False
        if hint_count:
            sam_row = sam_by_id.get(sample_id)
            if sam_row is None:
                raise ValueError("geometry hint has no matching SAM result")
            if sam_row.get("image_sha256") != geometry_row.get("image_sha256"):
                raise ValueError("geometry and SAM image hashes differ")
            geometry_supported = (
                isinstance(sam_row.get("mask_pixels"), int)
                and sam_row["mask_pixels"] > 0)
        elif sample_id in sam_by_id:
            raise ValueError("SAM result exists without a geometry hint")

        fused_present = evidence == "present" or geometry_supported
        if fused_present and visible:
            tp += 1
        elif fused_present:
            fp += 1
        elif visible:
            fn += 1
        else:
            tn += 1
        details.append({
            "sample_id": sample_id,
            "assistant_visible_slope_candidate": visible,
            "vlm_evidence": evidence,
            "geometry_hint_count": hint_count,
            "geometry_sam_supported": geometry_supported,
            "fused_candidate_present": fused_present,
        })
    return {
        "samples": len(details),
        "true_positive_candidates": tp,
        "false_positive_candidates": fp,
        "false_negative_candidates": fn,
        "true_negative_candidates": tn,
        "precision_on_assistant_review": tp / (tp + fp) if tp + fp else None,
        "recall_on_assistant_review": tp / (tp + fn) if tp + fn else None,
        "details": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vlm-score", required=True, type=Path)
    parser.add_argument("--geometry", required=True, type=Path)
    parser.add_argument("--sam", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    summary = fuse(*(
        json.loads(path.read_text(encoding="utf-8"))
        for path in (args.vlm_score, args.geometry, args.sam)))
    payload = {
        "schema_version": "labelagent.slope-fused-candidate.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "assistant_candidate_only_not_human_truth",
        "human_truth": False,
        "human_confirmed": False,
        "training_authorized": False,
        "production_deployed": False,
        "production_approved": False,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
