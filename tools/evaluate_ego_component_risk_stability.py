#!/usr/bin/env python3
"""Evaluate review-priority stability under mild grayscale perturbations."""

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
    from tools.build_ego_component_union_review import file_sha256
    from tools.rank_ego_component_review_risk import build_ranking
except ModuleNotFoundError:  # Direct script execution.
    from build_ego_component_union_review import file_sha256  # type: ignore
    from rank_ego_component_review_risk import build_ranking  # type: ignore


def gamma_090(gray: np.ndarray) -> np.ndarray:
    return np.clip(np.power(gray / 255.0, 0.90) * 255.0, 0, 255).astype(
        np.float32)


def jpeg_q95(gray: np.ndarray) -> np.ndarray:
    buffer = io.BytesIO()
    Image.fromarray(np.rint(gray).astype(np.uint8), mode="L").save(
        buffer, "JPEG", quality=95, subsampling=0)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return np.asarray(decoded.convert("L"), dtype=np.float32)


def evaluate(review_queues: list[Path], identity_manifest: Path,
             component_review: Path, known_review_receipt: Path,
             output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    staging = output.with_name(f".{output.name}.staging-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        runs = {}
        for name, transform in (("baseline", None), ("gamma_090", gamma_090),
                                ("jpeg_q95", jpeg_q95)):
            target = staging / f"ranking_{name}.json"
            packet = build_ranking(
                review_queues, identity_manifest, component_review, target,
                known_review_receipt, transform, name)
            runs[name] = packet
        baseline = {row["component_id"]: row for row in runs["baseline"]["rows"]}
        if any({row["component_id"] for row in packet["rows"]}
               != set(baseline) for packet in runs.values()):
            raise ValueError("stability runs have different component sets")
        comparisons = []
        for name in ("gamma_090", "jpeg_q95"):
            altered = {row["component_id"]: row for row in runs[name]["rows"]}
            exact_band = sum(
                baseline[key]["review_priority_band"] ==
                altered[key]["review_priority_band"] for key in baseline)
            top_30_agreement = sum(
                (baseline[key]["review_priority_rank"] <= 74) ==
                (altered[key]["review_priority_rank"] <= 74)
                for key in baseline)
            rank_shifts = [abs(baseline[key]["review_priority_rank"]
                               - altered[key]["review_priority_rank"])
                           for key in baseline]
            comparisons.append({
                "transform": name,
                "exact_band_stability": round(exact_band / len(baseline), 6),
                "top_30_percent_membership_stability": round(
                    top_30_agreement / len(baseline), 6),
                "median_absolute_rank_shift": float(np.median(rank_shifts)),
                "maximum_absolute_rank_shift": max(rank_shifts),
            })
        uncertain_id = "000-camera3-1793758297542631987-c001"
        uncertain = [{
            "transform": name,
            "rank": next(row["review_priority_rank"] for row in packet["rows"]
                         if row["component_id"] == uncertain_id),
            "band": next(row["review_priority_band"] for row in packet["rows"]
                         if row["component_id"] == uncertain_id),
        } for name, packet in runs.items()]
        packet = {
            "schema_version": "labelagent.ego_component_risk_stability.v1",
            "candidate_only": True, "human_truth": False,
            "approved_for_training": False, "production_deployed": False,
            "default_enabled": False, "automatic_decisions_enabled": False,
            "component_review_sha256": file_sha256(component_review),
            "known_review_receipt_sha256": file_sha256(known_review_receipt),
            "component_count": len(baseline),
            "comparisons": comparisons,
            "known_uncertain_component": uncertain,
            "all_exact_band_stability_at_least_95_percent": all(
                row["exact_band_stability"] >= 0.95 for row in comparisons),
            "known_uncertain_stays_top_10_percent": all(
                row["rank"] <= 25 for row in uncertain),
            "run_files": [],
            "next_step": "This measures queue stability only and cannot make component decisions.",
        }
        for name in runs:
            path = staging / f"ranking_{name}.json"
            packet["run_files"].append({
                "transform": name, "path": path.name,
                "sha256": file_sha256(path)})
        target = staging / "stability.json"
        target.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
        readback = json.loads(target.read_text(encoding="utf-8"))
        if (readback["component_count"] != len(baseline)
                or any(file_sha256(staging / row["path"]) != row["sha256"]
                       for row in readback["run_files"])):
            raise ValueError("stability readback verification failed")
        staging.replace(output)
    except Exception:
        if staging.exists() and staging.parent == output.parent:
            shutil.rmtree(staging)
        raise
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-queue", action="append", type=Path,
                        required=True)
    parser.add_argument("--identity-manifest", type=Path, required=True)
    parser.add_argument("--component-review", type=Path, required=True)
    parser.add_argument("--known-review-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    packet = evaluate(
        arguments.review_queue, arguments.identity_manifest,
        arguments.component_review, arguments.known_review_receipt,
        arguments.output)
    print(json.dumps({
        "component_count": packet["component_count"],
        "comparisons": packet["comparisons"],
        "known_uncertain_component": packet["known_uncertain_component"],
        "all_exact_band_stability_at_least_95_percent": packet[
            "all_exact_band_stability_at_least_95_percent"],
        "known_uncertain_stays_top_10_percent": packet[
            "known_uncertain_stays_top_10_percent"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
