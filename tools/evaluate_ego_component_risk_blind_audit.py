#!/usr/bin/env python3
"""Unblind frozen assistant decisions and measure review-risk enrichment."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

try:
    from tools.build_ego_component_union_review import file_sha256
except ModuleNotFoundError:  # Direct script execution.
    from build_ego_component_union_review import file_sha256  # type: ignore


DECISIONS = {"likely_fill_ok", "uncertain", "likely_reject"}
RISK_DECISIONS = {"uncertain", "likely_reject"}


def wilson_interval(risky: int, total: int, z: float = 1.96
                    ) -> tuple[float, float]:
    if total <= 0 or not (0 <= risky <= total):
        raise ValueError("invalid Wilson interval counts")
    proportion = risky / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return round(max(0.0, center - margin), 6), round(
        min(1.0, center + margin), 6)


def _summary(rows: list[dict]) -> dict:
    total = len(rows)
    risky = sum(row["decision"] in RISK_DECISIONS for row in rows)
    uncertain = sum(row["decision"] == "uncertain" for row in rows)
    rejected = sum(row["decision"] == "likely_reject" for row in rows)
    interval = wilson_interval(risky, total)
    return {
        "total": total,
        "likely_fill_ok": total - risky,
        "uncertain": uncertain,
        "likely_reject": rejected,
        "review_risk_count": risky,
        "review_risk_rate": round(risky / total, 6),
        "review_risk_rate_wilson95": list(interval),
    }


def evaluate(audit_path: Path, decision_receipt_path: Path,
             output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    audit_sha = file_sha256(audit_path)
    receipt_sha = file_sha256(decision_receipt_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    receipt = json.loads(decision_receipt_path.read_text(encoding="utf-8"))
    for name, packet in (("blind audit", audit), ("decision receipt", receipt)):
        if (packet.get("candidate_only") is not True
                or packet.get("human_truth") is not False
                or packet.get("approved_for_training") is not False
                or packet.get("production_deployed") is not False
                or packet.get("default_enabled") is not False):
            raise ValueError(f"{name} safety flags are inconsistent")
    if (audit.get("blinded_for_risk_band") is not True
            or receipt.get("blind_audit_sha256") != audit_sha
            or receipt.get("reviewer_blinded_to_risk_band") is not True):
        raise ValueError("blind audit decision binding mismatch")
    mapping = audit.get("mapping", [])
    mapping_by_id = {row.get("audit_id"): row for row in mapping}
    decisions = receipt.get("decisions", [])
    decisions_by_id = {row.get("audit_id"): row for row in decisions}
    if (len(mapping_by_id) != len(mapping)
            or len(decisions_by_id) != len(decisions)
            or set(decisions_by_id) != set(mapping_by_id)
            or len(decisions) != int(audit.get("selected_count", -1))):
        raise ValueError("blind audit decisions are incomplete or duplicated")
    revealed = []
    for audit_id in sorted(mapping_by_id):
        source, decision = mapping_by_id[audit_id], decisions_by_id[audit_id]
        if decision.get("decision") not in DECISIONS:
            raise ValueError(f"invalid blind decision: {audit_id}")
        revealed.append({
            "audit_id": audit_id,
            "component_id": source["component_id"],
            "camera": source["camera"],
            "hidden_band": source["hidden_band"],
            "global_rank": source["global_rank"],
            "camera_rank": source["camera_rank"],
            "score": source["score"],
            "decision": decision["decision"],
            "reason": decision.get("reason", ""),
        })
    by_band = {band: _summary(
        [row for row in revealed if row["hidden_band"] == band])
        for band in ("top", "middle", "bottom")}
    by_camera_band = {
        camera: {band: _summary([
            row for row in revealed
            if row["camera"] == camera and row["hidden_band"] == band])
            for band in ("top", "middle", "bottom")}
        for camera in ("camera1", "camera3")
    }
    top_rate = by_band["top"]["review_risk_rate"]
    middle_rate = by_band["middle"]["review_risk_rate"]
    bottom_rate = by_band["bottom"]["review_risk_rate"]
    total_rejects = sum(row["decision"] == "likely_reject" for row in revealed)
    top_rejects = by_band["top"]["likely_reject"]
    middle_rejects = by_band["middle"]["likely_reject"]
    bottom_rejects = by_band["bottom"]["likely_reject"]
    packet = {
        "schema_version": "labelagent.ego_component_risk_blind_evaluation.v1",
        "candidate_only": True, "human_truth": False,
        "approved_for_training": False, "production_deployed": False,
        "default_enabled": False, "assistant_review_only": True,
        "blind_audit_sha256": audit_sha,
        "decision_receipt_sha256": receipt_sha,
        "reviewer_blinded_to_risk_band": True,
        "selected_count": len(revealed),
        "by_band": by_band,
        "by_camera_band": by_camera_band,
        "top_minus_bottom_review_risk_rate": round(top_rate - bottom_rate, 6),
        "top_minus_middle_review_risk_rate": round(top_rate - middle_rate, 6),
        "likely_reject_total": total_rejects,
        "top_likely_reject_capture_rate": round(
            top_rejects / total_rejects, 6) if total_rejects else None,
        "top_plus_middle_likely_reject_capture_rate": round(
            (top_rejects + middle_rejects) / total_rejects, 6)
            if total_rejects else None,
        "bottom_likely_reject_count": bottom_rejects,
        "top_plus_middle_review_share": round(
            (by_band["top"]["total"] + by_band["middle"]["total"])
            / len(revealed), 6),
        "hard_reject_enrichment_observed": (
            total_rejects > 0 and bottom_rejects == 0
            and top_rejects + middle_rejects == total_rejects),
        "monotonic_enrichment_observed": (
            top_rate >= middle_rate >= bottom_rate and top_rate > bottom_rate),
        "revealed_rows": revealed,
        "conclusion_boundary": "Assistant blind-review enrichment is not human truth, model quality, or new-data generalization.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        temporary.write_text(
            json.dumps(packet, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        readback = json.loads(temporary.read_text(encoding="utf-8"))
        if readback["selected_count"] != len(revealed):
            raise ValueError("blind evaluation readback verification failed")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind-audit", type=Path, required=True)
    parser.add_argument("--decision-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    packet = evaluate(arguments.blind_audit, arguments.decision_receipt,
                      arguments.output)
    print(json.dumps({
        "selected_count": packet["selected_count"],
        "by_band": packet["by_band"],
        "top_minus_bottom_review_risk_rate": packet[
            "top_minus_bottom_review_risk_rate"],
        "monotonic_enrichment_observed": packet[
            "monotonic_enrichment_observed"],
        "hard_reject_enrichment_observed": packet[
            "hard_reject_enrichment_observed"],
        "top_plus_middle_likely_reject_capture_rate": packet[
            "top_plus_middle_likely_reject_capture_rate"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
