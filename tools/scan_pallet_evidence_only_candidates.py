#!/usr/bin/env python3
"""Scan isolated candidate JSON files with the v7 pallet shadow policy offline."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from source.agent.edit_agent import _rasterize
from source.agent.workflow import refine_pallet_candidate_geometry
from source.agent.workflow import PALLET_COMPACT_TEXTURE_MINIMUM_CANDIDATE_AREA_RATIO
from source.agent.workflow import _pallet_image_evidence_scale_gate
from source.agent.workflow import _image_guided_pallet_texture_holes
from source.agent.workflow import _image_guided_pallet_boundary_notches
from source.agent.workflow import _structural_hole_masks
from source.agent.workflow import PALLET_IMAGE_BOUNDARY_CLOSE_KERNEL


SCHEMA_VERSION = "labelagent.pallet-evidence-only-offline-scan.v1"
ROI_SCHEMA_VERSION = "labelagent.pallet-evidence-roi-offline-scan.v2"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def png_bytes(mask: np.ndarray) -> bytes:
    target = io.BytesIO()
    Image.fromarray((mask != 0).astype(np.uint8) * 255).save(
        target, format="PNG", optimize=False, compress_level=9)
    return target.getvalue()


def scan(snapshot_dir, candidate_dir, output_dir, *,
         geometry_policy="appearance_evidence_only"):
    if geometry_policy not in {
            "appearance_evidence_only", "appearance_evidence_roi"}:
        raise ValueError("unsupported evidence-only geometry policy")
    roi_policy = geometry_policy == "appearance_evidence_roi"
    schema_version = ROI_SCHEMA_VERSION if roi_policy else SCHEMA_VERSION
    snapshot_dir = Path(snapshot_dir).resolve()
    candidate_dir = Path(candidate_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    snapshot_manifest_path = snapshot_dir / "manifest.json"
    snapshot_payload = snapshot_manifest_path.read_bytes()
    snapshot = json.loads(snapshot_payload.decode("utf-8"))
    if snapshot.get("schema_version") != "labelagent.workbench-frame-snapshot.v1":
        raise ValueError("unsupported snapshot manifest")
    summary_path = candidate_dir / "summary.json"
    summary_payload = summary_path.read_bytes()
    summary = json.loads(summary_payload.decode("utf-8"))
    if (summary.get("schema_version") != "labelagent.new_dataset_smoke.v1"
            or summary.get("candidate_only") is not True
            or summary.get("human_truth") is not False
            or summary.get("approved_for_training") is not False
            or not isinstance(summary.get("runs"), list)):
        raise ValueError("isolated smoke summary safety flags invalid")
    snapshot_images = snapshot.get("images")
    if not isinstance(snapshot_images, list) or not snapshot_images:
        raise ValueError("snapshot image records invalid")
    snapshot_names = set()
    snapshot_frames = set()
    for item in snapshot_images:
        if not isinstance(item, dict):
            raise ValueError("snapshot image record invalid")
        frame = item.get("frame")
        name = item.get("image")
        if (isinstance(frame, bool) or not isinstance(frame, int) or frame < 0
                or not isinstance(name, str) or Path(name).name != name
                or not name or frame in snapshot_frames or name in snapshot_names):
            raise ValueError("snapshot image identity invalid")
        snapshot_frames.add(frame)
        snapshot_names.add(name)

    runs_by_input = {}
    for run in summary["runs"]:
        name = Path(str(run.get("input", ""))).name
        if not name or name in runs_by_input:
            raise ValueError("isolated smoke summary input binding invalid")
        runs_by_input[name] = run
    if set(runs_by_input) != snapshot_names:
        raise ValueError("snapshot and smoke run inputs differ")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    wrapper = Path(tempfile.mkdtemp(prefix=".pallet-v7-", dir=output_dir.parent))
    try:
        staging = wrapper / output_dir.name
        staging.mkdir()
        rows = []
        for item in snapshot_images:
            frame = item.get("frame")
            image_path = snapshot_dir / str(item.get("image"))
            image_payload = image_path.read_bytes()
            if digest(image_payload) != item.get("sha256"):
                raise ValueError(f"frame {frame} snapshot image SHA-256 drift")
            with Image.open(io.BytesIO(image_payload)) as decoded:
                image_rgb = np.asarray(decoded.convert("RGB"))
            height, width = image_rgb.shape[:2]
            run = runs_by_input.get(image_path.name)
            if (not isinstance(run, dict)
                    or run.get("input_sha256") != item.get("sha256")
                    or run.get("candidate_only") is not True
                    or run.get("human_truth") is not False
                    or run.get("approved_for_training") is not False):
                raise ValueError(f"frame {frame} smoke run binding invalid")
            candidate_path = candidate_dir / f"frame-{frame:06d}-candidate.json"
            candidate_payload = candidate_path.read_bytes()
            result = json.loads(candidate_payload.decode("utf-8"))
            if (Path(str(run.get("result_json", ""))).name != candidate_path.name
                    or run.get("result_sha256") != digest(candidate_payload)
                    or not isinstance(result.get("input_sha256"), str)
                    or not SHA256_PATTERN.fullmatch(result["input_sha256"])
                    or result.get("candidate_only") is not True
                    or result.get("human_truth") is not False
                    or result.get("approved_for_training") is not False
                    or not isinstance(result.get("candidates"), list)):
                raise ValueError(f"frame {frame} candidate binding or safety flags invalid")
            identifiers = set()
            for candidate in result["candidates"]:
                if str(candidate.get("label", "")).casefold() != "pallet":
                    continue
                identifier = candidate.get("id")
                if (isinstance(identifier, bool) or not isinstance(identifier, int)
                        or identifier < 0 or identifier in identifiers):
                    raise ValueError(f"frame {frame} pallet candidate ID invalid")
                identifiers.add(identifier)
                before = _rasterize(candidate, width, height).astype(np.uint8)
                refined, audit = refine_pallet_candidate_geometry(
                    candidate, width, height, image_rgb=image_rgb,
                    geometry_policy=geometry_policy)
                after = _rasterize(refined, width, height).astype(np.uint8)
                before_png = png_bytes(before)
                after_png = png_bytes(after)
                changed = int(np.count_nonzero(before != after))
                ys, xs = np.where(before != 0)
                if not len(xs):
                    raise ValueError(f"frame {frame} pallet candidate mask empty")
                bbox_width = int(xs.max() - xs.min() + 1)
                bbox_height = int(ys.max() - ys.min() + 1)
                bbox_area = bbox_width * bbox_height
                row = {
                    "frame": frame,
                    "candidate_id": identifier,
                    "source_image_sha256": item["sha256"],
                    # The isolated workflow hashes its canonical loaded frame;
                    # this is intentionally distinct from the source-file SHA.
                    "workflow_input_sha256": result["input_sha256"],
                    "candidate_file_sha256": digest(candidate_payload),
                    "before_mask_sha256": digest(before_png),
                    "after_mask_sha256": digest(after_png),
                    "changed_pixels": changed,
                    "pixel_count_before": int(before.sum()),
                    "pixel_count_after": int(after.sum()),
                    "candidate_area_ratio": round(
                        int(before.sum()) / max(before.size, 1), 8),
                    "bbox_width": bbox_width,
                    "bbox_height": bbox_height,
                    "bbox_area_ratio": round(
                        bbox_area / max(before.size, 1), 8),
                    "bbox_fill_ratio": round(
                        int(before.sum()) / max(bbox_area, 1), 8),
                    "image_texture_scale_gate_passed": bool(
                        int(before.sum()) / max(before.size, 1)
                        >= PALLET_COMPACT_TEXTURE_MINIMUM_CANDIDATE_AREA_RATIO),
                    "audit": audit,
                }
                if roi_policy:
                    scale_passed, scale_audit = _pallet_image_evidence_scale_gate(
                        before, allow_roi_scale=True)
                    background = (before == 0).astype(np.uint8)
                    _, components, stats, _ = cv2.connectedComponentsWithStats(
                        background, connectivity=8)
                    border_labels = set(np.unique(np.concatenate((
                        components[0, :], components[-1, :],
                        components[:, 0], components[:, -1]))).tolist())
                    texture_holes, _ = _image_guided_pallet_texture_holes(
                        before, image_rgb, components, stats, border_labels,
                        allow_roi_scale=True)
                    structural_holes = _structural_hole_masks(
                        before, image_rgb=image_rgb, allow_roi_scale=True)
                    boundary_notches = _image_guided_pallet_boundary_notches(
                        before, image_rgb, structural_holes,
                        allow_roi_scale=True)
                    enclosed_components = [
                        index for index in range(1, len(stats))
                        if index not in border_labels]
                    close_kernel = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (PALLET_IMAGE_BOUNDARY_CLOSE_KERNEL,
                         PALLET_IMAGE_BOUNDARY_CLOSE_KERNEL))
                    closed = cv2.morphologyEx(
                        before.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel)
                    raw_boundary_additions = (
                        (closed > 0) & (before == 0)).astype(np.uint8)
                    raw_count, _, raw_stats, _ = cv2.connectedComponentsWithStats(
                        raw_boundary_additions, connectivity=8)
                    row.update({
                        "roi_evidence_scale_gate_passed": scale_passed,
                        "roi_evidence_scale": scale_audit,
                        "roi_image_texture_hole_count": len(texture_holes),
                        "roi_image_texture_hole_pixels": sum(
                            int(stats[index, cv2.CC_STAT_AREA])
                            for index in texture_holes),
                        "roi_image_boundary_notch_count": len(boundary_notches),
                        "roi_image_boundary_notch_pixels": sum(
                            int(item["area"]) for item in boundary_notches),
                        "roi_raw_enclosed_hole_count": len(enclosed_components),
                        "roi_raw_enclosed_hole_pixels": sum(
                            int(stats[index, cv2.CC_STAT_AREA])
                            for index in enclosed_components),
                        "roi_raw_boundary_addition_count": max(raw_count - 1, 0),
                        "roi_raw_boundary_addition_pixels": sum(
                            int(raw_stats[index, cv2.CC_STAT_AREA])
                            for index in range(1, raw_count)),
                    })
                if changed:
                    prefix = f"frame-{frame:06d}-candidate-{identifier}"
                    before_name = f"{prefix}-before.png"
                    after_name = f"{prefix}-after.png"
                    (staging / before_name).write_bytes(before_png)
                    (staging / after_name).write_bytes(after_png)
                    row.update({"before_mask": before_name,
                                "after_mask": after_name})
                rows.append(row)

        manifest = {
            "schema_version": schema_version,
            "snapshot_manifest_sha256": digest(snapshot_payload),
            "smoke_summary_sha256": digest(summary_payload),
            "frame_count": len(snapshot_images),
            "pallet_candidate_count": len(rows),
            "changed_candidate_count": sum(row["changed_pixels"] > 0 for row in rows),
            "image_texture_scale_gate_passed_count": sum(
                row["image_texture_scale_gate_passed"] for row in rows),
            "candidate_only": True,
            "human_truth": False,
            "training_authorized": False,
            "deployment_authorized": False,
            "workbench_write": False,
            "rows": rows,
        }
        if roi_policy:
            manifest.update({
                "geometry_policy": geometry_policy,
                "roi_evidence_scale_gate_passed_count": sum(
                    row["roi_evidence_scale_gate_passed"] for row in rows),
                "roi_image_texture_hole_candidate_count": sum(
                    bool(row["roi_image_texture_hole_count"]) for row in rows),
                "roi_image_boundary_notch_candidate_count": sum(
                    bool(row["roi_image_boundary_notch_count"]) for row in rows),
            })
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2,
                       sort_keys=True) + "\n", encoding="utf-8")
        for row in rows:
            if not row["changed_pixels"]:
                continue
            if (digest((staging / row["before_mask"]).read_bytes()) !=
                    row["before_mask_sha256"]
                    or digest((staging / row["after_mask"]).read_bytes()) !=
                    row["after_mask_sha256"]):
                raise ValueError("staged changed-mask readback mismatch")
        manifest_sha256 = digest(manifest_path.read_bytes())
        staging.replace(output_dir)
        return {
            "schema_version": schema_version,
            "frame_count": manifest["frame_count"],
            "pallet_candidate_count": len(rows),
            "changed_candidate_count": manifest["changed_candidate_count"],
            "image_texture_scale_gate_passed_count": manifest[
                "image_texture_scale_gate_passed_count"],
            "manifest_sha256": manifest_sha256,
            "workbench_write": False,
        } | ({
            "geometry_policy": geometry_policy,
            "roi_evidence_scale_gate_passed_count": manifest[
                "roi_evidence_scale_gate_passed_count"],
        } if roi_policy else {})
    finally:
        shutil.rmtree(wrapper, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--geometry-policy",
        choices=("appearance_evidence_only", "appearance_evidence_roi"),
        default="appearance_evidence_only")
    args = parser.parse_args()
    print(json.dumps(scan(
        args.snapshot_dir, args.candidate_dir, args.output,
        geometry_policy=args.geometry_policy),
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
