#!/usr/bin/env python3
"""Evaluate frozen 3D-instance LiDAR prompts with point-SAM, offline only."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.error

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from source.agent.edit_agent import _rasterize, valid_shape
from tools.evaluate_pallet_lidar_point_prompts import (
    _render_panel,
    _shape_from_result,
    point_policies,
    summarize_sam_attempts,
)
from tools.evaluate_slope_lidar_sam import polygon_mask, request_json
from tools.review_distant_pallet_recovery import (
    _mapping,
    _public_metrics,
    match_expansion,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(audit_path: Path, baseline_path: Path, output: Path,
             sam_url: str, *, timeout: float = 120) -> dict:
    if output.exists():
        raise ValueError("output directory already exists")
    audit_path, baseline_path = audit_path.resolve(), baseline_path.resolve()
    inputs = {str(path): _sha256(path) for path in (audit_path, baseline_path)}
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("schema_version") != "labelagent.pallet_lidar_instance_prompt_audit.v1"
        or audit.get("sam_calls") != 0
        or audit.get("candidate_only") is not True
        or audit.get("human_truth") is not False
    ):
        raise ValueError("requires a frozen candidate-only instance-prompt audit")
    baseline = _mapping(json.loads(baseline_path.read_text(encoding="utf-8")))
    state_path = Path(r"D:\Apps\LabelAgent\work\state.json")
    state_before = _sha256(state_path)
    health_before = request_json(sam_url.rstrip("/") + "/health", timeout=15)
    if health_before.get("status") != "ok" or health_before.get("model_loaded") is not True:
        raise ValueError("point SAM service is not ready")
    output.mkdir(parents=True)
    rows, panels = [], []
    for source in audit["rows"]:
        topic, frame = str(source["topic"]), str(source["frame"])
        shapes = baseline[topic][frame]
        matches = [shape for shape in shapes if str(shape.get("id")) == str(source["shape_id"])]
        if len(matches) != 1:
            raise ValueError(f"frozen candidate identity changed: {topic} {frame}")
        current_shape = matches[0]
        current = _rasterize(current_shape, 1920, 1536).astype(np.uint8)
        prompt = source["instance_prompt"]
        image_path = Path(source["image_path"])
        if _sha256(image_path) != source["image_sha256"]:
            raise ValueError(f"source image changed: {image_path}")
        row = {
            "topic": topic,
            "frame": frame,
            "shape_id": source["shape_id"],
            "prompt_ready": bool(prompt["ready"]),
            "sam_called": False,
            "safe_expansion": False,
            "candidate_only": True,
            "human_truth": False,
            "approved_for_training": False,
        }
        proposed = current
        points_with_sources = prompt.get("points") or []
        if prompt["ready"]:
            points = [
                {"x": int(point["x"]), "y": int(point["y"]), "label": int(point["label"])}
                for point in points_with_sources
            ]
            source_by_point = {
                (int(point["x"]), int(point["y"]), int(point["label"])): point.get("source")
                for point in points_with_sources
            }
            encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
            attempts, successful = [], []
            for attempt_index, (policy, policy_points) in enumerate(point_policies(points), start=1):
                started = time.monotonic()
                try:
                    result = request_json(
                        sam_url.rstrip("/") + "/segment_points",
                        {"image": encoded, "points": policy_points},
                        timeout=timeout,
                    )
                    elapsed = time.monotonic() - started
                    mask = polygon_mask(result, 1920, 1536)
                    proposal = _shape_from_result(
                        result, f"lidar-instance-{topic.split('/')[2]}-{frame}-{attempt_index}"
                    )
                    if not valid_shape({**proposal, "id": 1}, 1920, 1536):
                        raise ValueError("SAM returned invalid pallet geometry")
                    matched, metrics = match_expansion(
                        current_shape, [proposal], 1920, 1536,
                        minimum_baseline_coverage=.5,
                        minimum_area_ratio=1.25,
                        maximum_minor_axis_ratio=1.5,
                        minimum_axis_elongation=2.0,
                        other_baselines=[shape for shape in shapes
                                         if str(shape.get("id")) != str(source["shape_id"])],
                        maximum_other_baseline_fraction=.1,
                    )
                    positive_covered = all(mask[p["y"], p["x"]] != 0
                                           for p in policy_points if p["label"] == 1)
                    negative_clear = all(mask[p["y"], p["x"]] == 0
                                         for p in policy_points if p["label"] == 0)
                    tall_points = [
                        point for point in policy_points
                        if source_by_point.get((point["x"], point["y"], point["label"]))
                        == "tall_background"
                    ]
                    tall_clear = all(mask[p["y"], p["x"]] == 0 for p in tall_points)
                    safe = matched is not None and positive_covered and negative_clear and tall_clear
                    attempt = {
                        "policy": policy,
                        "points": policy_points,
                        "elapsed_seconds": round(elapsed, 4),
                        "sam_score": result.get("score"),
                        "sam_warnings": result.get("warnings") or [],
                        "sam_mask_pixels": int(mask.sum()),
                        "positive_points_covered": positive_covered,
                        "negative_points_clear": negative_clear,
                        "tall_background_points": len(tall_points),
                        "tall_background_clear": tall_clear,
                        "safe_expansion": safe,
                        "expansion_metrics": _public_metrics(metrics),
                    }
                    attempts.append(attempt)
                    successful.append((
                        bool(safe), bool(positive_covered), bool(negative_clear), bool(tall_clear),
                        -int(mask.sum()), float(result.get("score", 0) or 0), attempt, mask,
                    ))
                except urllib.error.HTTPError as error:
                    attempts.append({
                        "policy": policy,
                        "elapsed_seconds": round(time.monotonic() - started, 4),
                        "http_status": error.code,
                        "error": error.read().decode("utf-8", "replace")[:1000],
                        "safe_expansion": False,
                    })
                except Exception as error:
                    attempts.append({
                        "policy": policy,
                        "elapsed_seconds": round(time.monotonic() - started, 4),
                        "error_type": type(error).__name__,
                        "error": str(error)[:1000],
                        "safe_expansion": False,
                    })
            row["sam_called"] = True
            row["sam_attempts"] = attempts
            if successful:
                successful.sort(key=lambda item: item[:6], reverse=True)
                selected = successful[0]
                attempt, proposed = selected[6], selected[7]
                row.update({key: attempt[key] for key in (
                    "policy", "sam_score", "sam_warnings", "sam_mask_pixels",
                    "positive_points_covered", "negative_points_clear",
                    "tall_background_points", "tall_background_clear",
                    "safe_expansion", "expansion_metrics",
                )})
                row["sam_elapsed_seconds"] = round(sum(
                    attempt.get("elapsed_seconds", 0) for attempt in attempts
                ), 4)
            else:
                row["sam_all_attempts_failed"] = True
        panels.append(_render_panel(
            Image.open(image_path).convert("RGB"), current.astype(bool), proposed.astype(bool),
            [{"x": p["x"], "y": p["y"], "label": p["label"]}
             for p in points_with_sources],
            f"{topic} frame {frame} safe={row['safe_expansion']}",
        ))
        rows.append(row)
        (output / f"{topic.split('/')[2]}-frame-{frame}.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    sheet = Image.new("RGB", (1440, sum(panel.height for panel in panels)), "white")
    top = 0
    for panel in panels:
        sheet.paste(panel, (0, top))
        top += panel.height
    sheet_path = output / "instance-lidar-sam-review.jpg"
    sheet.save(sheet_path, quality=90, optimize=True)
    health_after = request_json(sam_url.rstrip("/") + "/health", timeout=15)
    if any(_sha256(Path(path)) != digest for path, digest in inputs.items()):
        raise RuntimeError("an input changed during point-SAM evaluation")
    summary = summarize_sam_attempts(rows)
    report = {
        "schema_version": "labelagent.pallet_lidar_instance_sam.v1",
        "status": "offline_candidate_not_deployed",
        "input_sha256": inputs,
        "selected_count": len(rows),
        "prompt_ready_count": sum(row["prompt_ready"] for row in rows),
        **summary,
        "safe_expansion_count": sum(row["safe_expansion"] for row in rows),
        "health_before": health_before,
        "health_after": health_after,
        "reference_labels_used": False,
        "workbench_writes": 0,
        "production_state_before_sha256": state_before,
        "production_state_after_sha256": _sha256(state_path),
        "contact_sheet": str(sheet_path.resolve()),
        "contact_sheet_sha256": _sha256(sheet_path),
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "production_deployed": False,
        "rows": rows,
    }
    report_path = output / "results.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path.resolve())
    report["report_sha256"] = _sha256(report_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sam-url", default="http://127.0.0.1:28766")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    report = evaluate(args.audit, args.baseline, args.output, args.sam_url, timeout=args.timeout)
    print(json.dumps({key: report[key] for key in (
        "selected_count", "prompt_ready_count", "sam_frame_count", "sam_attempt_count",
        "sam_successful_attempt_count", "sam_http_error_count", "sam_other_error_count",
        "safe_expansion_count", "reference_labels_used", "workbench_writes",
        "production_state_before_sha256", "production_state_after_sha256",
        "contact_sheet", "contact_sheet_sha256", "report_path", "report_sha256",
    )}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
