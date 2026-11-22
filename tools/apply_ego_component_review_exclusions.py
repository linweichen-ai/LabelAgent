#!/usr/bin/env python3
"""Revert uncertain/failed ego micro-fill components in a shadow candidate."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

try:
    from tools.build_ego_component_union_review import _verified_bytes, file_sha256
except ModuleNotFoundError:  # Direct script execution.
    from build_ego_component_union_review import (  # type: ignore
        _verified_bytes, file_sha256)


EXCLUSION_DECISIONS = {"uncertain", "assistant_fail"}
ALLOWED_DECISIONS = EXCLUSION_DECISIONS | {"assistant_pass"}


def _decode_mask(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as decoded:
        return np.asarray(decoded.convert("L")) > 0


def _save_mask(mask: np.ndarray, target: Path) -> None:
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(
        target, "PNG", optimize=True)


def apply_exclusions(shadow_packet_path: Path, component_review_path: Path,
                     decision_receipt_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    shadow_sha = file_sha256(shadow_packet_path)
    review_sha = file_sha256(component_review_path)
    receipt_sha = file_sha256(decision_receipt_path)
    shadow = json.loads(shadow_packet_path.read_text(encoding="utf-8"))
    review = json.loads(component_review_path.read_text(encoding="utf-8"))
    receipt = json.loads(decision_receipt_path.read_text(encoding="utf-8"))
    for name, packet in (("shadow", shadow), ("component review", review),
                         ("decision receipt", receipt)):
        if (packet.get("candidate_only") is not True
                or packet.get("human_truth") is not False
                or packet.get("approved_for_training") is not False
                or packet.get("production_deployed") is not False):
            raise ValueError(f"{name} safety flags are inconsistent")
    if (shadow.get("source_masks_modified") is not False
            or review.get("source_masks_modified") is not False
            or review.get("default_enabled") is not False
            or receipt.get("default_enabled") is not False):
        raise ValueError("candidate safety flags are inconsistent")
    if (review.get("shadow_packet_sha256") != shadow_sha
            or receipt.get("source_component_review_packet_sha256") !=
            review_sha):
        raise ValueError("packet hash binding mismatch")

    review_rows = review.get("rows", [])
    rows_by_id = {row.get("component_id"): row for row in review_rows}
    if (len(rows_by_id) != len(review_rows)
            or len(rows_by_id) != int(review.get("component_count", -1))):
        raise ValueError("component review identities are incomplete or duplicated")
    decisions = receipt.get("decisions", [])
    decisions_by_id = {row.get("component_id"): row for row in decisions}
    if len(decisions_by_id) != len(decisions):
        raise ValueError("decision identities are duplicated")
    for component_id, decision in decisions_by_id.items():
        source = rows_by_id.get(component_id)
        if source is None:
            raise ValueError(f"decision references unknown component: {component_id}")
        if (decision.get("decision") not in ALLOWED_DECISIONS
                or decision.get("review_image_sha256") !=
                source.get("review_image_sha256")
                or int(decision.get("area_pixels", -1)) !=
                int(source.get("area_pixels", -2))):
            raise ValueError(f"decision binding mismatch: {component_id}")

    shadow_rows = shadow.get("samples", [])
    shadow_by_sample = {row.get("sample_id"): row for row in shadow_rows}
    if len(shadow_by_sample) != len(shadow_rows):
        raise ValueError("shadow sample identities are duplicated")
    components_by_sample: dict[str, list[dict]] = {}
    for row in review_rows:
        sample_id = row.get("sample_id")
        if sample_id not in shadow_by_sample:
            raise ValueError(f"component references unknown sample: {sample_id}")
        components_by_sample.setdefault(sample_id, []).append(row)

    prepared = []
    excluded_ids = []
    excluded_pixels = 0
    reconstructed_component_count = 0
    reconstructed_added_pixels = 0
    for sample_id, shadow_row in sorted(shadow_by_sample.items()):
        shadow_mask = _decode_mask(_verified_bytes(
            shadow_packet_path.parent / shadow_row.get("mask", ""),
            shadow_row.get("mask_sha256"), "shadow mask"))
        added_union = np.zeros_like(shadow_mask)
        exclusion_mask = np.zeros_like(shadow_mask)
        sample_components = components_by_sample.get(sample_id, [])
        for row in sample_components:
            bbox = row.get("bbox_xyxy")
            if (not isinstance(bbox, list) or len(bbox) != 4
                    or not all(isinstance(value, int) for value in bbox)):
                raise ValueError(f"invalid component bbox: {row.get('component_id')}")
            x0, y0, x1, y1 = bbox
            if not (0 <= x0 < x1 <= shadow_mask.shape[1]
                    and 0 <= y0 < y1 <= shadow_mask.shape[0]):
                raise ValueError(f"component bbox is out of bounds: {row.get('component_id')}")
            cropped = _decode_mask(_verified_bytes(
                component_review_path.parent / row.get("component_mask", ""),
                row.get("component_mask_sha256"), "component mask"))
            if (cropped.shape != (y1 - y0, x1 - x0)
                    or int(cropped.sum()) != int(row.get("area_pixels", -1))):
                raise ValueError(f"component mask accounting failed: {row.get('component_id')}")
            full = np.zeros_like(shadow_mask)
            full[y0:y1, x0:x1] = cropped
            if np.logical_and(added_union, full).any():
                raise ValueError(f"component masks overlap: {sample_id}")
            if np.logical_and(full, ~shadow_mask).any():
                raise ValueError(f"component is outside shadow mask: {row.get('component_id')}")
            added_union |= full
            reconstructed_component_count += 1
            reconstructed_added_pixels += int(cropped.sum())
            decision = decisions_by_id.get(row.get("component_id"), {})
            if decision.get("decision") in EXCLUSION_DECISIONS:
                exclusion_mask |= full
                excluded_ids.append(row["component_id"])
                excluded_pixels += int(cropped.sum())
        refined = shadow_mask & ~exclusion_mask
        if np.logical_and(refined, ~shadow_mask).any():
            raise ValueError(f"refined mask adds pixels: {sample_id}")
        prepared.append({
            "sample_id": sample_id,
            "source_image_sha256": shadow_row.get("source_image_sha256"),
            "source_union_mask_sha256": shadow_row.get(
                "source_union_mask_sha256"),
            "source_shadow_mask_sha256": shadow_row.get("mask_sha256"),
            "mask_name": f"{sample_id}-review-refined-mask.png",
            "mask": refined,
            "shadow_pixels": int(shadow_mask.sum()),
            "refined_pixels": int(refined.sum()),
            "reverted_pixels": int(exclusion_mask.sum()),
        })
    if (reconstructed_component_count != int(review.get("component_count", -1))
            or reconstructed_added_pixels !=
            int(review.get("added_pixel_total", -1))):
        raise ValueError("component review does not reconstruct global additions")
    if excluded_pixels != sum(
            int(decisions_by_id[component_id]["area_pixels"])
            for component_id in excluded_ids):
        raise ValueError("excluded pixel accounting failed")

    staging = output.with_name(f".{output.name}.staging-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        output_rows = []
        for row in prepared:
            target = staging / row["mask_name"]
            _save_mask(row["mask"], target)
            output_rows.append({
                "sample_id": row["sample_id"],
                "source_image_sha256": row["source_image_sha256"],
                "source_union_mask_sha256": row["source_union_mask_sha256"],
                "source_shadow_mask_sha256": row["source_shadow_mask_sha256"],
                "mask": row["mask_name"],
                "mask_sha256": file_sha256(target),
                "shadow_pixels": row["shadow_pixels"],
                "refined_pixels": row["refined_pixels"],
                "reverted_pixels": row["reverted_pixels"],
            })
        packet = {
            "schema_version": "labelagent.ego_review_exclusion_shadow.v1",
            "candidate_only": True, "human_truth": False,
            "approved_for_training": False, "production_deployed": False,
            "default_enabled": False, "source_masks_modified": False,
            "review_status": "partial_review_exclusions_applied",
            "identity_manifest_sha256": shadow.get(
                "identity_manifest_sha256"),
            "shadow_packet_sha256": shadow_sha,
            "component_review_sha256": review_sha,
            "decision_receipt_sha256": receipt_sha,
            "source_component_count": reconstructed_component_count,
            "source_added_pixel_total": reconstructed_added_pixels,
            "decision_count": len(decisions),
            "assistant_pass_count": sum(
                row.get("decision") == "assistant_pass" for row in decisions),
            "excluded_component_count": len(excluded_ids),
            "excluded_component_ids": excluded_ids,
            "excluded_pixel_total": excluded_pixels,
            "remaining_candidate_component_count": (
                reconstructed_component_count - len(excluded_ids)),
            "remaining_candidate_added_pixel_total": (
                reconstructed_added_pixels - excluded_pixels),
            "samples": output_rows,
            "next_step": "Re-run frozen-reference metrics. This partial review shadow cannot approve truth, training or deployment.",
        }
        target = staging / "review_exclusion_shadow.json"
        target.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
        readback = json.loads(target.read_text(encoding="utf-8"))
        if (readback["excluded_pixel_total"] != excluded_pixels
                or any(file_sha256(staging / row["mask"]) != row["mask_sha256"]
                       for row in output_rows)):
            raise ValueError("review exclusion readback verification failed")
        staging.replace(output)
    except Exception:
        if staging.exists() and staging.parent == output.parent:
            shutil.rmtree(staging)
        raise
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-packet", type=Path, required=True)
    parser.add_argument("--component-review", type=Path, required=True)
    parser.add_argument("--decision-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    packet = apply_exclusions(
        arguments.shadow_packet, arguments.component_review,
        arguments.decision_receipt, arguments.output)
    print(json.dumps({
        "excluded_component_count": packet["excluded_component_count"],
        "excluded_pixel_total": packet["excluded_pixel_total"],
        "remaining_candidate_added_pixel_total": packet[
            "remaining_candidate_added_pixel_total"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
