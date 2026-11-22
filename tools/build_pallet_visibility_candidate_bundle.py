#!/usr/bin/env python3
"""Build a review-only pallet mask bundle from frozen visibility routes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.evaluate_pallet_box3d_roi_sam import digest


COLORS = [(35, 210, 125), (255, 160, 35), (90, 125, 255), (220, 75, 180)]


def validate_mask(path: Path, expected_pixels: int) -> np.ndarray:
    value = np.asarray(Image.open(path).convert("L"))
    if value.shape != (512, 640) or not set(np.unique(value)).issubset({0, 255}):
        raise ValueError("candidate mask is not a 640x512 binary PNG")
    mask = value > 0
    if int(mask.sum()) != int(expected_pixels):
        raise ValueError("candidate mask pixel count differs from split report")
    return mask


def render_preview(image_path: Path, candidates: list[dict], suppressed: list[dict],
                   output: Path) -> None:
    canvas = np.asarray(Image.open(image_path).convert("RGB")).copy()
    for index, row in enumerate(candidates):
        mask = np.asarray(Image.open(row["absolute_mask_path"]).convert("L")) > 0
        color = np.asarray(COLORS[index % len(COLORS)], dtype=np.uint8)
        canvas[mask] = ((canvas[mask].astype(np.uint16) * 3 + color * 2) // 5).astype(np.uint8)
        contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[0]
        cv2.drawContours(canvas, contours, -1, tuple(int(v) for v in color), 1,
                         cv2.LINE_AA)
    legend = [(COLORS[index % len(COLORS)],
               f"obj {row['obj_id']}: review mask candidate")
              for index, row in enumerate(candidates)]
    legend.extend([((70, 70, 230),
                    f"obj {row['obj_id']}: 2D suppressed ({row['route']})")
                   for row in suppressed])
    overlay = canvas.copy(); height = 12 + len(legend) * 21
    cv2.rectangle(overlay, (8, 8), (390, height), (12, 20, 27), -1)
    canvas = cv2.addWeighted(overlay, .8, canvas, .2, 0)
    for index, (color, text) in enumerate(legend):
        y = 26 + index * 21
        cv2.circle(canvas, (20, y - 5), 4, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, text, (32, y), cv2.FONT_HERSHEY_SIMPLEX, .42,
                    (242, 246, 248), 1, cv2.LINE_AA)
    Image.fromarray(canvas).save(output)


def build(route_path: Path, split_path: Path, split_mask_root: Path,
          image_manifest_path: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("output directory already exists")
    routes = json.loads(route_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    manifest = json.loads(image_manifest_path.read_text(encoding="utf-8"))
    if (routes.get("schema_version") != "labelagent.pallet_visibility_candidate_routes.v1"
            or routes.get("production_deployed") is not False
            or split.get("schema_version") != "labelagent.pallet_box3d_overlap_pairs.v1"
            or split.get("production_deployed") is not False):
        raise ValueError("requires frozen non-deployed route and split reports")
    image_files = {(str(row["sequence"]), int(str(row["camera"]).split("_")[-1])): row
                   for row in manifest["files"]}
    split_rows = {(str(row["sequence"]), int(row["camera"]), str(row["frame_id"])): row
                  for row in split["rows"]}
    output.mkdir(parents=True); masks_dir = output / "masks"; masks_dir.mkdir()
    previews = output / "previews"; previews.mkdir()
    rows = []
    for route in routes["rows"]:
        key = (str(route["sequence"]), int(route["camera"]), str(route["frame_id"]))
        split_row = split_rows.get(key)
        if split_row is None:
            raise ValueError(f"route has no matching split row: {key}")
        metric = next((row for row in split_row["depth_split_metrics"]
                       if str(row["obj_id"]) == str(route["obj_id"])), None)
        if metric is None:
            raise ValueError("route object has no split metric")
        base = {"sequence": key[0], "camera": key[1], "frame_id": key[2],
                "obj_id": route["obj_id"], "route": route["route"],
                "review_required": True, "write_human_truth": False,
                "write_training_data": False, "production_deploy": False}
        if route["allow_current_2d_candidate"]:
            if not metric.get("candidate_pass"):
                raise ValueError("allowed route lacks a passing split metric")
            name = f"{key[0]}-camera{key[1]}-frame{key[2]}-obj{route['obj_id']}.png"
            source = split_mask_root / (
                f"{key[0]}-camera{key[1]}-obj{route['obj_id']}-depth-split.png")
            expected = split_row["depth_split_masks"][str(route["obj_id"])]["sha256"]
            if digest(source) != expected:
                raise ValueError("source split mask hash mismatch")
            validate_mask(source, metric["split_pixels"])
            target = masks_dir / name; shutil.copyfile(source, target)
            if digest(target) != expected:
                raise ValueError("copied candidate mask hash mismatch")
            rows.append({**base, "action": "emit_review_mask_candidate",
                         "mask_path": f"masks/{name}", "mask_sha256": expected,
                         "mask_pixels": metric["split_pixels"],
                         "propagation_source_frame_ids": []})
        else:
            rows.append({**base, "action": "suppress_current_2d_candidate",
                         "mask_path": None, "mask_sha256": None, "mask_pixels": 0,
                         "propagation_source_frame_ids": route[
                             "propagation_source_frame_ids"]})
    preview_rows = []
    for key in sorted(set((row["sequence"], row["camera"], row["frame_id"])
                          for row in rows)):
        item = image_files[(key[0], key[1])]
        if str(item["frame_id"]) != key[2]:
            raise ValueError("image manifest frame differs from route")
        image_path = Path(manifest["local_root"]) / item["relative_path"]
        if digest(image_path) != item["sha256"]:
            raise ValueError("held-out image changed")
        current = [row for row in rows if (row["sequence"], row["camera"],
                                           row["frame_id"]) == key]
        candidates = []
        for row in current:
            if row["mask_path"]:
                candidates.append({**row,
                                   "absolute_mask_path": str(output / row["mask_path"])})
        suppressed = [row for row in current if not row["mask_path"]]
        name = f"{key[0]}-camera{key[1]}-candidate-route.png"
        preview = previews / name
        render_preview(image_path, candidates, suppressed, preview)
        preview_rows.append({"sequence": key[0], "camera": key[1],
                             "frame_id": key[2], "path": f"previews/{name}",
                             "sha256": digest(preview)})
    report = {
        "schema_version": "labelagent.pallet_visibility_candidate_bundle.v1",
        "status": "review_only_candidate_bundle_not_deployed",
        "route_report_sha256": digest(route_path),
        "split_report_sha256": digest(split_path),
        "image_manifest_sha256": digest(image_manifest_path),
        "candidate_count": len(rows),
        "mask_candidates_emitted": sum(row["mask_path"] is not None for row in rows),
        "current_2d_candidates_suppressed": sum(row["mask_path"] is None for row in rows),
        "preview_count": len(preview_rows),
        "human_truth_writes": 0, "training_data_writes": 0,
        "workbench_writes": 0, "production_deployed": False,
        "rows": rows, "previews": preview_rows,
    }
    bundle = output / "bundle.json"
    bundle.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    report["bundle_path"] = str(bundle.resolve()); report["bundle_sha256"] = digest(bundle)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-report", required=True, type=Path)
    parser.add_argument("--split-report", required=True, type=Path)
    parser.add_argument("--split-mask-root", required=True, type=Path)
    parser.add_argument("--image-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = build(args.route_report, args.split_report, args.split_mask_root,
                   args.image_manifest, args.output)
    print(json.dumps({key: report[key] for key in (
        "candidate_count", "mask_candidates_emitted",
        "current_2d_candidates_suppressed", "preview_count",
        "human_truth_writes", "training_data_writes", "workbench_writes",
        "bundle_path", "bundle_sha256")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
