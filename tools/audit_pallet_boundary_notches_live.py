#!/usr/bin/env python3
"""Read-only live audit for image-guided pallet boundary cleanup.

The tool reads the current workbench bootstrap, frame images, and candidate
state through GET endpoints.  It never calls a write endpoint.  For every
requested frame it replays the current local cleanup implementation against
the largest pallet candidate and emits compact JSON plus an optional visual
comparison sheet.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import cv2
import numpy as np

from source.agent.edit_agent import _rasterize
from source.agent.workflow import (
    _significant_component_count,
    _structural_hole_masks,
    refine_pallet_candidate_geometry,
)
from tools.audit_pallet_geometry import enclosed_holes


def _get_json(base_url: str, path: str, query: dict | None = None) -> dict:
    url = f"{base_url.rstrip('/')}{path}"
    if query:
        url = f"{url}?{urlencode(query)}"
    with urlopen(url, timeout=15) as response:  # noqa: S310 - explicit local URL
        return json.loads(response.read().decode("utf-8"))


def _get_image(base_url: str, frame: int, query: dict) -> np.ndarray:
    url = f"{base_url.rstrip('/')}/api/frame/{frame}?{urlencode(query)}"
    with urlopen(url, timeout=15) as response:  # noqa: S310 - explicit local URL
        encoded = np.frombuffer(response.read(), dtype=np.uint8)
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"frame {frame}: JPEG decode failed")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _largest_pallet(candidates: list[dict], width: int, height: int) -> tuple[dict, np.ndarray]:
    ranked = []
    for shape in candidates:
        if str(shape.get("label", "")).casefold() != "pallet":
            continue
        mask = _rasterize(shape, width, height).astype(np.uint8)
        ranked.append((int(mask.sum()), str(shape.get("id", "")), shape, mask))
    if not ranked:
        raise ValueError("frame contains no pallet candidate")
    _, _, shape, mask = max(ranked, key=lambda item: (item[0], item[1]))
    return shape, mask


def _frame_shapes(payload: object, frame: int) -> list[dict]:
    if isinstance(payload, dict):
        payload = payload.get(str(frame), payload.get(frame, []))
    return payload if isinstance(payload, list) else []


def _minimum_structural_retention(before: np.ndarray, after: np.ndarray,
                                  image_rgb: np.ndarray) -> float:
    structural = _structural_hole_masks(before, image_rgb=image_rgb)
    if not structural:
        return 1.0
    return min(
        int(np.count_nonzero(hole_mask & (after == 0))) / hole_area
        for hole_mask, hole_area in structural
    )


def _panel(image_rgb: np.ndarray, mask: np.ndarray, title: str) -> np.ndarray:
    output = image_rgb.copy()
    color = np.zeros_like(output)
    color[..., 1] = 220
    color[..., 2] = 70
    selected = mask.astype(bool)
    output[selected] = cv2.addWeighted(
        output[selected], 0.58, color[selected], 0.42, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, (255, 90, 20), 2)
    cv2.rectangle(output, (0, 0), (output.shape[1], 34), (10, 18, 26), -1)
    cv2.putText(output, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                .62, (255, 255, 255), 2, cv2.LINE_AA)
    return output


def audit(base_url: str, frames: list[int]) -> tuple[dict, np.ndarray]:
    bootstrap = _get_json(base_url, "/api/bootstrap")
    scope = {
        "path": bootstrap["path"],
        "topic": bootstrap["topic"],
        "file_activation_generation": bootstrap["file_activation_generation"],
    }
    results = []
    panels = []
    for frame in frames:
        state = _get_json(base_url, f"/api/frame_state/{frame}", scope)
        image_rgb = _get_image(base_url, frame, scope)
        height, width = image_rgb.shape[:2]
        candidates = _frame_shapes(state.get("candidates", {}), frame)
        truth = _frame_shapes(state.get("truth", {}), frame)
        shape, before = _largest_pallet(candidates, width, height)
        refined, adjustment = refine_pallet_candidate_geometry(
            copy.deepcopy(shape), width, height, image_rgb=image_rgb)
        after = _rasterize(refined, width, height).astype(np.uint8)
        changed = int(np.count_nonzero(before != after))
        before_area = int(before.sum())
        holes_before = enclosed_holes(before)
        holes_after = enclosed_holes(after)
        components_before = _significant_component_count(before)
        components_after = _significant_component_count(after)
        adjustment = adjustment or {}
        result = {
            "frame": frame,
            "shape_id": shape.get("id"),
            "candidate_count": len(candidates),
            "truth_count": len(truth),
            "geometry_cleanup_version": adjustment.get("geometry_cleanup_version"),
            "changed_pixels": changed,
            "change_ratio": round(changed / max(before_area, 1), 6),
            "holes_before": len(holes_before),
            "holes_after": len(holes_after),
            "filled_image_boundary_notch_count": adjustment.get(
                "filled_image_boundary_notch_count", 0),
            "filled_image_boundary_notch_pixels": adjustment.get(
                "filled_image_boundary_notch_pixels", 0),
            "components_before": components_before,
            "components_after": components_after,
            "component_merge_count": components_before - components_after,
            "minimum_structural_hole_retention": round(
                _minimum_structural_retention(before, after, image_rgb), 6),
            "candidate_only": True,
            "human_truth": False,
            "approved_for_training": False,
        }
        results.append(result)
        panels.extend([
            _panel(image_rgb, before, f"frame {frame} before"),
            _panel(image_rgb, after, (
                f"frame {frame} v5 | change {result['change_ratio']:.2%} | "
                f"merge {result['component_merge_count']}")),
        ])
    sheet = np.vstack([
        np.hstack(panels[index:index + 2])
        for index in range(0, len(panels), 2)
    ])
    report = {
        "schema_version": "labelagent.pallet_boundary_notch_live_audit.v1",
        "workbench": base_url,
        "path": bootstrap["path"],
        "topic": bootstrap["topic"],
        "file_activation_generation": bootstrap["file_activation_generation"],
        "frames": frames,
        "result_count": len(results),
        "maximum_change_ratio": max((row["change_ratio"] for row in results), default=0),
        "minimum_structural_hole_retention": min(
            (row["minimum_structural_hole_retention"] for row in results), default=1),
        "maximum_component_merge_count": max(
            (row["component_merge_count"] for row in results), default=0),
        "gate_passed": all(
            row["change_ratio"] <= .06
            and row["minimum_structural_hole_retention"] >= .75
            and 0 <= row["component_merge_count"] <= 2
            for row in results),
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "results": results,
    }
    return report, sheet


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--frames", default="110,112,118")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--image", type=Path)
    args = parser.parse_args()
    frames = [int(value) for value in args.frames.split(",") if value.strip()]
    report, sheet_rgb = audit(args.base_url, frames)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        report["output_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    if args.image:
        args.image.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.image), cv2.cvtColor(sheet_rgb, cv2.COLOR_RGB2BGR))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
