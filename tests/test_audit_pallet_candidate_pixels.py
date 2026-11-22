import json
from pathlib import Path

import numpy as np
from PIL import Image

from tools.audit_pallet_candidate_pixels import (
    build, digest, structural_alerts, structure_metrics,
)


def test_structure_metrics_detect_fragments_and_holes():
    mask = np.zeros((40, 50), dtype=bool)
    mask[5:30, 5:35] = True
    mask[12:18, 12:20] = False
    mask[35:38, 45:48] = True
    metrics = structure_metrics(mask, small_island_pixels=20)
    assert metrics["component_count"] == 2
    assert metrics["hole_count"] == 1
    assert metrics["small_island_count"] == 1
    assert "fragmented_components" in structural_alerts(metrics)


def test_build_preserves_frozen_inputs_and_suppressed_rows(tmp_path: Path):
    source = tmp_path / "source"
    image_path = source / "cameras" / "camera_0" / "100.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (80, 60), "gray").save(image_path)
    mask = np.zeros((60, 80), dtype=np.uint8)
    mask[20:40, 20:60] = 255
    bundle_root = tmp_path / "bundle"
    masks = bundle_root / "masks"
    masks.mkdir(parents=True)
    mask_path = masks / "candidate.png"
    Image.fromarray(mask).save(mask_path)
    manifest = {
        "schema_version": "labelagent.heldout_evaluation_manifest.v1",
        "evaluation_only": True, "local_root": str(source),
        "files": [{"sequence": "010", "camera": "camera_0",
                   "frame_id": "100", "relative_path":
                   "cameras/camera_0/100.png", "sha256": digest(image_path)}],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    split = {
        "schema_version": "labelagent.pallet_box3d_overlap_pairs.v1",
        "rows": [{"sequence": "010", "camera": 0, "frame_id": "100",
                  "depth_split_metrics": [
                      {"obj_id": 1, "candidate_pass": True,
                       "source_mask_retention": 1.0,
                       "projected_box_coverage": .8},
                      {"obj_id": 2, "candidate_pass": False,
                       "source_mask_retention": 0.0,
                       "projected_box_coverage": 0.0}]}],
    }
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    bundle = {
        "schema_version": "labelagent.pallet_visibility_candidate_bundle.v1",
        "production_deployed": False, "human_truth_writes": 0,
        "training_data_writes": 0,
        "image_manifest_sha256": digest(manifest_path),
        "split_report_sha256": digest(split_path),
        "rows": [
            {"sequence": "010", "camera": 0, "frame_id": "100",
             "obj_id": 1, "route": "depth_split_candidate",
             "mask_path": "masks/candidate.png", "mask_sha256": digest(mask_path),
             "mask_pixels": int((mask > 0).sum())},
            {"sequence": "010", "camera": 0, "frame_id": "100",
             "obj_id": 2, "route": "split_candidate_failed_review",
             "mask_path": None, "mask_sha256": None, "mask_pixels": 0},
        ],
    }
    bundle_path = bundle_root / "bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    before = {path: path.read_bytes() for path in (
        bundle_path, manifest_path, split_path, image_path, mask_path)}

    report = build(
        bundle_path, manifest_path, split_path, tmp_path / "output")

    assert report["candidate_count"] == 2
    assert report["mask_count"] == 1
    assert report["suppressed_count"] == 1
    assert report["human_truth"] is False
    assert report["approved_for_training"] is False
    assert all(path.read_bytes() == value for path, value in before.items())
