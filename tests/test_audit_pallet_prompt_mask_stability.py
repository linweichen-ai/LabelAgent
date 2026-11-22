import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from tools import audit_pallet_prompt_mask_stability as audit


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_fixture(root: Path, *, empty_first: bool = False):
    results_path = root / "round714" / "results.json"
    frozen_path = root / "round674" / "manifest.json"
    mask_root = results_path.parent / "masks"
    mask_root.mkdir(parents=True)
    rows, files = [], []
    for target_index, (sequence, frame_id, camera, obj_id) in enumerate(
            sorted(audit.TARGETS, key=lambda value: (value[0], int(value[3])))):
        image_path = frozen_path.parent / "images" / sequence / f"camera_{camera}" / f"{frame_id}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        if not image_path.exists():
            Image.new("RGB", (640, 512), (70 + target_index * 10, 90, 110)).save(image_path)
        identity = (sequence, frame_id, camera)
        if not any((entry["sequence"], entry["frame_id"],
                    int(entry["camera"].split("_")[-1])) == identity for entry in files):
            files.append({"sequence": sequence, "frame_id": frame_id,
                          "camera": f"camera_{camera}",
                          "relative_path": image_path.relative_to(frozen_path.parent).as_posix(),
                          "sha256": digest(image_path)})
        attempts = []
        for policy_index, policy in enumerate(["policy_a", "policy_b", "policy_c"]):
            mask_path = mask_root / f"{sequence}-obj{obj_id}-{policy}.png"
            mask = Image.new("L", (640, 512), 0)
            if not (empty_first and target_index == 0 and policy_index == 0):
                draw = ImageDraw.Draw(mask)
                draw.rectangle((240 + policy_index, 150, 270 + policy_index, 168), fill=255)
            mask.save(mask_path)
            attempts.append({"policy": policy, "mask_path": str(mask_path),
                             "mask_sha256": digest(mask_path),
                             "split_metrics": [{"obj_id": int(obj_id),
                                                "candidate_pass": True}],
                             "accepted": False})
        attempts.append({"policy": "failed_no_mask", "error": "expected", "accepted": False})
        rows.append({"sequence": sequence, "frame_id": frame_id, "camera": camera,
                     "target_obj_id": obj_id, "attempts": attempts})
    write_json(frozen_path, {"evaluation_only": True, "training_return_allowed": False,
                             "candidate_write_allowed": False,
                             "human_truth_write_allowed": False,
                             "workbench_writes": 0, "human_truth_writes": 0,
                             "oss_writes": 0, "files": files})
    write_json(results_path, {
        "status": "offline_temporal_local_point_sam_candidate_not_deployed",
        "candidate_only": True, "human_pixel_truth": False,
        "approved_for_training": False, "production_deployed": False,
        "human_truth_writes": 0, "training_data_writes": 0,
        "workbench_writes": 0, "oss_writes": 0, "rows": rows})
    return results_path, frozen_path, mask_root


def authorize(monkeypatch, results: Path) -> None:
    monkeypatch.setattr(audit, "EXPECTED_ROUND714_SHA", digest(results))


def test_builds_consensus_masks_and_stable_medoid(tmp_path, monkeypatch):
    results, frozen, masks = make_fixture(tmp_path)
    authorize(monkeypatch, results)
    output = tmp_path / "output"
    result = audit.build_audit(results, frozen, [masks], output)
    assert result["target_count"] == 3
    assert result["included_policy_mask_count"] == 9
    report = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert report["independent_validation"] is False
    assert report["medoid_is_assistant_candidate_only"] is True
    assert report["candidate_only"] is True
    assert report["human_pixel_truth"] is False
    assert report["approved_for_training"] is False
    assert report["training_authorized"] is False
    assert report["human_truth_writes"] == report["training_data_writes"] == 0
    assert report["workbench_writes"] == report["oss_writes"] == 0
    assert len(list(output.glob("*-consensus50.png"))) == 3
    assert len(list(output.glob("*-consensus75.png"))) == 3
    assert len(list(output.glob("*-overlay.png"))) == 6
    for row in report["rows"]:
        assert len(row["pairwise_iou_matrix"]) == 3
        assert row["assistant_medoid_candidate"]["candidate_only"] is True


def test_rejects_empty_included_mask(tmp_path, monkeypatch):
    results, frozen, masks = make_fixture(tmp_path, empty_first=True)
    authorize(monkeypatch, results)
    with pytest.raises(ValueError, match="empty included mask"):
        audit.build_audit(results, frozen, [masks], tmp_path / "output")


def test_rejects_duplicate_target_row(tmp_path, monkeypatch):
    results, frozen, masks = make_fixture(tmp_path)
    value = json.loads(results.read_text(encoding="utf-8"))
    value["rows"].append(dict(value["rows"][0]))
    write_json(results, value)
    authorize(monkeypatch, results)
    with pytest.raises(ValueError, match="duplicate Round714 target"):
        audit.build_audit(results, frozen, [masks], tmp_path / "output")


def test_rejects_mask_hash_drift(tmp_path, monkeypatch):
    results, frozen, masks = make_fixture(tmp_path)
    authorize(monkeypatch, results)
    target = next(masks.glob("*-policy_a.png"))
    Image.new("L", (640, 512), 255).save(target)
    with pytest.raises(ValueError, match="hash mismatch"):
        audit.build_audit(results, frozen, [masks], tmp_path / "output")


def test_rejects_unexpected_target(tmp_path, monkeypatch):
    results, frozen, masks = make_fixture(tmp_path)
    value = json.loads(results.read_text(encoding="utf-8"))
    value["rows"][0]["target_obj_id"] = "999"
    write_json(results, value)
    authorize(monkeypatch, results)
    with pytest.raises(ValueError, match="unexpected Round714 target"):
        audit.build_audit(results, frozen, [masks], tmp_path / "output")


def test_rejects_existing_output(tmp_path, monkeypatch):
    results, frozen, masks = make_fixture(tmp_path)
    authorize(monkeypatch, results)
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(ValueError, match="output directory already exists"):
        audit.build_audit(results, frozen, [masks], output)


def test_rejects_duplicate_frozen_image_identity(tmp_path, monkeypatch):
    results, frozen, masks = make_fixture(tmp_path)
    authorize(monkeypatch, results)
    value = json.loads(frozen.read_text(encoding="utf-8"))
    value["files"].append(dict(value["files"][0]))
    write_json(frozen, value)
    with pytest.raises(ValueError, match="duplicate Round674 image identity"):
        audit.build_audit(results, frozen, [masks], tmp_path / "output")
