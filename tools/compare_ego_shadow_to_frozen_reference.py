#!/usr/bin/env python3
"""Compare shadow masks against references built only from original masks."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from tools.audit_ego_union_pixels import (
        DEFAULT_THRESHOLDS, mask_metrics, package_balanced_reference,
        threshold_alerts)
    from tools.build_ego_component_union_review import (
        _verified_bytes, file_sha256, load_identity_manifest)
except ModuleNotFoundError:  # Direct script execution.
    from audit_ego_union_pixels import (  # type: ignore
        DEFAULT_THRESHOLDS, mask_metrics, package_balanced_reference,
        threshold_alerts)
    from build_ego_component_union_review import (  # type: ignore
        _verified_bytes, file_sha256, load_identity_manifest)


def _decode_mask(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as decoded:
        return np.asarray(decoded.convert("L")) > 0


def compare(review_queues: list[Path], identity_manifest_path: Path,
            shadow_packet_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    identities = load_identity_manifest(identity_manifest_path)
    identity_sha = file_sha256(identity_manifest_path)
    originals, seen = [], set()
    for queue_path in review_queues:
        packet = json.loads(queue_path.read_text(encoding="utf-8"))
        gates = packet.get("safety_gates") or {}
        if (packet.get("candidate_only") is not True
                or packet.get("human_truth") is not False
                or packet.get("approved_for_training") is not False
                or packet.get("production_deployed") is not False
                or gates.get("identity_manifest_sha256") != identity_sha):
            raise ValueError("review queue safety or identity gate mismatch")
        for row in packet.get("samples", []):
            sample_id = row.get("sample_id")
            if sample_id in seen:
                raise ValueError(f"duplicate sample identity: {sample_id}")
            seen.add(sample_id)
            if row.get("union_available") is not True:
                continue
            identity = identities.get(sample_id)
            if identity is None or identity["image_sha256"] != row.get(
                    "source_image_sha256"):
                raise ValueError(f"identity binding mismatch: {sample_id}")
            mask_bytes = _verified_bytes(
                queue_path.parent / row.get("union_mask", ""),
                row.get("union_mask_sha256"), "original union mask")
            originals.append({
                "sample_id": sample_id, "package": identity["package"],
                "camera": identity["camera"], "mask": _decode_mask(mask_bytes),
                "mask_sha256": row["union_mask_sha256"],
            })
    if seen != set(identities):
        raise ValueError("review queues and identity manifest differ")
    shadow = json.loads(shadow_packet_path.read_text(encoding="utf-8"))
    if (shadow.get("candidate_only") is not True
            or shadow.get("human_truth") is not False
            or shadow.get("approved_for_training") is not False
            or shadow.get("production_deployed") is not False
            or shadow.get("source_masks_modified") is not False
            or shadow.get("identity_manifest_sha256") != identity_sha):
        raise ValueError("shadow packet safety or identity binding mismatch")
    shadow_rows = {row.get("sample_id"): row for row in shadow.get("samples", [])}
    if set(shadow_rows) != {item["sample_id"] for item in originals}:
        raise ValueError("shadow and original sample sets differ")
    rows = []
    for sample in sorted(originals, key=lambda item: item["sample_id"]):
        external = {}
        for other in originals:
            if (other["camera"] == sample["camera"]
                    and other["package"] != sample["package"]):
                external.setdefault(other["package"], []).append(other["mask"])
        reference, _, _ = package_balanced_reference(external)
        shadow_row = shadow_rows[sample["sample_id"]]
        if shadow_row.get("source_union_mask_sha256") != sample["mask_sha256"]:
            raise ValueError(f"shadow source hash mismatch: {sample['sample_id']}")
        shadow_mask = _decode_mask(_verified_bytes(
            shadow_packet_path.parent / shadow_row.get("mask", ""),
            shadow_row.get("mask_sha256"), "shadow mask"))
        if np.logical_and(sample["mask"], ~shadow_mask).any():
            raise ValueError(f"shadow removed original pixels: {sample['sample_id']}")
        before = mask_metrics(sample["mask"], reference)
        after = mask_metrics(shadow_mask, reference)
        before_alerts = threshold_alerts(before, sample["camera"], DEFAULT_THRESHOLDS)
        after_alerts = threshold_alerts(after, sample["camera"], DEFAULT_THRESHOLDS)
        rows.append({
            "sample_id": sample["sample_id"], "camera": sample["camera"],
            "package": sample["package"], "before": before, "after": after,
            "before_alerts": before_alerts, "after_alerts": after_alerts,
            "new_alerts": sorted(set(after_alerts) - set(before_alerts)),
            "resolved_alerts": sorted(set(before_alerts) - set(after_alerts)),
            "iou_delta": round(after["reference_iou"] - before["reference_iou"], 6),
            "recall_delta": round(after["reference_recall"] - before["reference_recall"], 6),
            "excess_delta": round(after["candidate_excess"] - before["candidate_excess"], 6),
            "hole_ratio_delta": round(after["hole_area_ratio"] - before["hole_area_ratio"], 6),
        })
    packet = {
        "schema_version": "labelagent.ego_shadow_frozen_reference_comparison.v1",
        "candidate_only": True, "human_truth": False,
        "approved_for_training": False, "production_deployed": False,
        "reference_uses_original_masks_only": True,
        "identity_manifest_sha256": identity_sha,
        "shadow_packet_sha256": file_sha256(shadow_packet_path),
        "sample_count": len(rows),
        "before_alert_count": sum(bool(row["before_alerts"]) for row in rows),
        "after_alert_count": sum(bool(row["after_alerts"]) for row in rows),
        "new_alert_sample_count": sum(bool(row["new_alerts"]) for row in rows),
        "resolved_alert_sample_count": sum(bool(row["resolved_alerts"]) for row in rows),
        "mean_iou_delta": round(float(np.mean([row["iou_delta"] for row in rows])), 6),
        "mean_recall_delta": round(float(np.mean([row["recall_delta"] for row in rows])), 6),
        "mean_excess_delta": round(float(np.mean([row["excess_delta"] for row in rows])), 6),
        "mean_hole_ratio_delta": round(float(np.mean([row["hole_ratio_delta"] for row in rows])), 6),
        "rows": rows,
        "next_step": "Accept only after visual review; frozen-reference metrics do not create human truth or training approval.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        temporary.write_text(
            json.dumps(packet, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        json.loads(temporary.read_text(encoding="utf-8"))
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-queue", action="append", type=Path, required=True)
    parser.add_argument("--identity-manifest", type=Path, required=True)
    parser.add_argument("--shadow-packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    packet = compare(arguments.review_queue, arguments.identity_manifest,
                     arguments.shadow_packet, arguments.output)
    print(json.dumps({key: packet[key] for key in (
        "sample_count", "before_alert_count", "after_alert_count",
        "new_alert_sample_count", "resolved_alert_sample_count",
        "mean_iou_delta", "mean_hole_ratio_delta")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
