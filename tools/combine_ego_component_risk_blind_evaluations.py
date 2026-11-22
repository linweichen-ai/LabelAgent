#!/usr/bin/env python3
"""Combine disjoint assistant-blind ego risk evaluations reproducibly."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from tools.build_ego_component_union_review import file_sha256
    from tools.evaluate_ego_component_risk_blind_audit import _summary
except ModuleNotFoundError:  # Direct script execution.
    from build_ego_component_union_review import file_sha256  # type: ignore
    from evaluate_ego_component_risk_blind_audit import _summary  # type: ignore


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def combine(evaluation_paths: list[Path], output: Path) -> dict:
    _require(len(evaluation_paths) >= 2, "at least two evaluations are required")
    _require(not output.exists(), "combined output already exists")
    rows = []
    inputs = []
    seen_components = set()
    for index, path in enumerate(evaluation_paths, 1):
        packet = json.loads(path.read_text(encoding="utf-8"))
        for field, expected in (
                ("candidate_only", True), ("human_truth", False),
                ("approved_for_training", False),
                ("production_deployed", False), ("default_enabled", False),
                ("assistant_review_only", True),
                ("reviewer_blinded_to_risk_band", True)):
            _require(packet.get(field) is expected,
                     f"evaluation {index} safety flag mismatch: {field}")
        revealed = packet.get("revealed_rows")
        _require(isinstance(revealed, list) and revealed,
                 f"evaluation {index} revealed rows are missing")
        _require(packet.get("selected_count") == len(revealed),
                 f"evaluation {index} selected count mismatch")
        for row in revealed:
            component_id = row.get("component_id")
            _require(isinstance(component_id, str) and component_id,
                     f"evaluation {index} component id is missing")
            _require(component_id not in seen_components,
                     f"component appears in multiple evaluations: {component_id}")
            seen_components.add(component_id)
            copied = dict(row)
            copied["source_evaluation_index"] = index
            rows.append(copied)
        inputs.append({
            "index": index,
            "sha256": file_sha256(path),
            "selected_count": len(revealed),
        })

    by_band = {band: _summary(
        [row for row in rows if row.get("hidden_band") == band])
        for band in ("top", "middle", "bottom")}
    by_camera_band = {
        camera: {band: _summary([
            row for row in rows
            if row.get("camera") == camera and row.get("hidden_band") == band])
            for band in ("top", "middle", "bottom")}
        for camera in ("camera1", "camera3")
    }
    _require(all(summary["total"] > 0 for summary in by_band.values()),
             "combined evaluation has an empty risk band")
    top_rate = by_band["top"]["review_risk_rate"]
    middle_rate = by_band["middle"]["review_risk_rate"]
    bottom_rate = by_band["bottom"]["review_risk_rate"]
    total_rejects = sum(row.get("decision") == "likely_reject" for row in rows)
    top_rejects = by_band["top"]["likely_reject"]
    middle_rejects = by_band["middle"]["likely_reject"]
    bottom_rejects = by_band["bottom"]["likely_reject"]
    packet = {
        "schema_version": "labelagent.ego_component_risk_blind_combined.v1",
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "production_deployed": False,
        "default_enabled": False,
        "assistant_review_only": True,
        "reviewer_blinded_to_risk_band": True,
        "evaluation_count": len(inputs),
        "input_evaluations": inputs,
        "selected_count": len(rows),
        "disjoint_component_ids": True,
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
            / len(rows), 6),
        "hard_reject_enrichment_observed": (
            total_rejects > 0 and bottom_rejects == 0
            and top_rejects + middle_rejects == total_rejects),
        "monotonic_enrichment_observed": (
            top_rate >= middle_rate >= bottom_rate and top_rate > bottom_rate),
        "revealed_rows": rows,
        "conclusion_boundary": (
            "Two disjoint assistant-blind samples from the same exposed dataset; "
            "not human truth, model quality, or new-data generalization."),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    _require(not temporary.exists(), "combined staging output already exists")
    try:
        temporary.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
        readback = json.loads(temporary.read_text(encoding="utf-8"))
        _require(readback["selected_count"] == len(rows),
                 "combined evaluation readback mismatch")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    packet = combine(arguments.evaluation, arguments.output)
    print(json.dumps({
        "evaluation_count": packet["evaluation_count"],
        "selected_count": packet["selected_count"],
        "by_band": packet["by_band"],
        "likely_reject_total": packet["likely_reject_total"],
        "top_plus_middle_likely_reject_capture_rate": packet[
            "top_plus_middle_likely_reject_capture_rate"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
