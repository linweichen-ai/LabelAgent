import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from tools import build_pallet_dense_hard_review_pack as pack


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_mask(path: Path, value: int = 255) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", (640, 512), 0)
    if value:
        for x in range(250, 265):
            for y in range(140, 155):
                image.putpixel((x, y), value)
    image.save(path)
    return digest(path)


def fixtures(root: Path):
    r674 = root / "round674" / "manifest.json"
    r676 = root / "round676" / "results.json"
    r714 = root / "round714" / "results.json"
    r706 = root / "round706" / "results.json"
    files = []
    rows676, rows714, rows706 = [], [], []
    for sequence, spec in pack.FRAME_SPECS.items():
        image_path = r674.parent / "images" / sequence / "camera_1" / f"{spec['frame_id']}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (640, 512), (80, 90, 100)).save(image_path)
        files.append({"sequence": sequence, "frame_id": spec["frame_id"],
                      "camera": "camera_1",
                      "relative_path": str(image_path.relative_to(r674.parent)).replace("\\", "/"),
                      "sha256": digest(image_path)})
        all_ids = set(spec["expected_component"])
        next_id = 100 if sequence == "006" else 200
        while len(all_ids) < 20:
            all_ids.add(str(next_id)); next_id += 1
        label = {"objects": [{"obj_type": "Pallet", "obj_id": obj_id}
                             for obj_id in sorted(all_ids, key=int)]}
        write_json(r674.parent / "labels" / sequence / f"{spec['frame_id']}.json", label)
        instances, metrics, masks = [], [], {}
        for index, obj_id in enumerate(sorted(spec["expected_component"], key=int)):
            is_warning = obj_id in spec["warning_676"]
            is_redraw = obj_id in spec["redraw_676"]
            instances.append({"obj_id": int(obj_id), "localization_pass": not is_warning and not is_redraw,
                              "projected_bbox": [240 + index, 135 + index, 290 + index, 170 + index]})
            metrics.append({"obj_id": int(obj_id), "candidate_pass": not is_redraw,
                            "handled_candidate": not is_redraw, "split_pixels": 225})
            mask_path = r676.parent / "masks" / f"{sequence}-obj{obj_id}-depth-split.png"
            mask_sha = make_mask(mask_path, 0 if obj_id == "79" else 255)
            masks[obj_id] = {"path": str(mask_path), "sha256": mask_sha}
        rows676.append({"sequence": sequence, "frame_id": spec["frame_id"], "camera": 1,
                        "instances": instances, "depth_split_metrics": metrics,
                        "depth_split_masks": masks})
        for obj_id, policy in spec["primary_714"].items():
            mask_path = r714.parent / "masks" / f"{sequence}-obj{obj_id}-{policy}.png"
            mask_sha = make_mask(mask_path)
            rows714.append({"sequence": sequence, "frame_id": spec["frame_id"], "camera": 1,
                            "target_obj_id": obj_id, "attempts": [{
                                "policy": policy, "positive_coverage_all_temporal_points": 1.0,
                                "negative_clear_all_center_conflict_points": .9,
                                "split_metrics": [{"obj_id": int(obj_id), "candidate_pass": True}],
                                "accepted": False, "mask_path": str(mask_path),
                                "mask_sha256": mask_sha}]})
        rows706.append({"sequence": sequence,
                        "component_obj_ids": sorted(spec["expected_component"], key=int),
                        "best_failed_combination": {"failed_member_obj_ids": [spec["expected_blocker"]]}})
    write_json(r674, {"evaluation_only": True, "training_return_allowed": False,
                      "candidate_write_allowed": False, "human_truth_write_allowed": False,
                      "files": files})
    write_json(r676, {"status": "offline_instance_diagnostic_not_deployed",
                      "human_pixel_truth": False, "approved_for_training": False,
                      "production_deployed": False, "workbench_writes": 0, "rows": rows676})
    write_json(r714, {"status": "offline_temporal_local_point_sam_candidate_not_deployed",
                      "candidate_only": True, "human_pixel_truth": False,
                      "approved_for_training": False, "production_deployed": False,
                      "human_truth_writes": 0, "training_data_writes": 0,
                      "workbench_writes": 0, "oss_writes": 0,
                      "recovered_candidate_count": 0, "rows": rows714})
    write_json(r706, {"status": "offline_joint_conflict_candidate_diagnostic_not_deployed",
                      "candidate_only": True, "human_pixel_truth": False,
                      "approved_for_training": False, "production_deployed": False,
                      "human_truth_writes": 0, "training_data_writes": 0,
                      "workbench_writes": 0, "oss_writes": 0, "rows": rows706})
    return r674, r676, r714, r706


def test_builds_expected_candidate_only_pack(tmp_path):
    r674, r676, r714, r706 = fixtures(tmp_path)
    output = tmp_path / "pack"
    result = pack.build_pack(r674, r676, r714, output, r706)
    manifest = json.loads((output / "manifest.json").read_text())
    assert result["primary_count"] == 9
    assert result["warning_count"] == 1
    assert result["redraw_count"] == 5
    assert result["out_of_scope_pallet_count"] == 25
    assert len(list(output.glob("overview_*.jpg"))) == 2
    assert len(list((output / "cards").glob("*.jpg"))) == 15
    assert manifest["candidate_only"] is True
    assert manifest["human_pixel_truth"] is False
    assert manifest["approved_for_training"] is False
    assert manifest["training_authorized"] is False
    assert manifest["automatic_approval"] is False
    assert manifest["workbench_writes"] == manifest["training_data_writes"] == 0
    assert all(item["outside_instance"] == "ignore" for item in manifest["items"])
    assert sum(len(frame["out_of_scope_obj_ids"]) for frame in manifest["frames"]) == 25
    assert not any(item["candidate_source"].startswith("round700")
                   or item["candidate_source"].startswith("round706")
                   for item in manifest["items"])


def test_fails_closed_on_selected_mask_hash_drift(tmp_path):
    r674, r676, r714, r706 = fixtures(tmp_path)
    data = json.loads(r714.read_text())
    data["rows"][0]["attempts"][0]["mask_sha256"] = "0" * 64
    write_json(r714, data)
    with pytest.raises(ValueError, match="hash mismatch"):
        pack.build_pack(r674, r676, r714, tmp_path / "pack", r706)
    assert not (tmp_path / "pack").exists()


def test_fails_closed_on_training_boundary_drift(tmp_path):
    r674, r676, r714, r706 = fixtures(tmp_path)
    data = json.loads(r714.read_text())
    data["approved_for_training"] = True
    write_json(r714, data)
    with pytest.raises(ValueError, match="boundary is unsafe"):
        pack.build_pack(r674, r676, r714, tmp_path / "pack", r706)
    assert not (tmp_path / "pack").exists()


def test_fails_closed_if_out_of_scope_count_changes(tmp_path):
    r674, r676, r714, r706 = fixtures(tmp_path)
    label_path = r674.parent / "labels" / "006" / f"{pack.FRAME_SPECS['006']['frame_id']}.json"
    data = json.loads(label_path.read_text())
    data["objects"].pop()
    write_json(label_path, data)
    with pytest.raises(ValueError, match="expected 20 pallet"):
        pack.build_pack(r674, r676, r714, tmp_path / "pack", r706)
    assert not (tmp_path / "pack").exists()


def test_fails_closed_on_duplicate_manifest_identity(tmp_path):
    r674, r676, r714, r706 = fixtures(tmp_path)
    data = json.loads(r674.read_text())
    data["files"].append(dict(data["files"][0]))
    write_json(r674, data)
    with pytest.raises(ValueError, match="duplicate Round674 file identity"):
        pack.build_pack(r674, r676, r714, tmp_path / "pack", r706)
    assert not (tmp_path / "pack").exists()
