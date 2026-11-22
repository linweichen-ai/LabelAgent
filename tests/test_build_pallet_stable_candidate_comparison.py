import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from tools import build_pallet_stable_candidate_comparison as compare


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_mask(path: Path, offset: int = 0, *, empty: bool = False) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", (640, 512), 0)
    if not empty:
        ImageDraw.Draw(image).rectangle((245 + offset, 150, 270 + offset, 168), fill=255)
    image.save(path)
    return digest(path), 0 if empty else 26 * 19


def fixture(root: Path, *, empty_medoid: bool = False):
    review_path = root / "round721" / "manifest.json"
    stability_path = root / "round728" / "manifest.json"
    frozen_path = root / "round674" / "manifest.json"
    mask_root = root / "masks"
    review_items, stable_rows, files = [], [], []
    made_images = set()
    for index, (sequence, frame_id, camera, obj_id) in enumerate(
            sorted(compare.TARGETS, key=lambda value: (value[0], int(value[3])))):
        frame_key = (sequence, frame_id, camera)
        if frame_key not in made_images:
            image_path = frozen_path.parent / "images" / sequence / f"camera_{camera}" / f"{frame_id}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (640, 512), (80, 95, 110)).save(image_path)
            files.append({"sequence": sequence, "frame_id": frame_id,
                          "camera": f"camera_{camera}",
                          "relative_path": image_path.relative_to(frozen_path.parent).as_posix(),
                          "sha256": digest(image_path)})
            made_images.add(frame_key)
        current_path = mask_root / f"{sequence}-obj{obj_id}-current.png"
        medoid_path = mask_root / f"{sequence}-obj{obj_id}-medoid.png"
        current_sha, current_pixels = make_mask(current_path, 0)
        medoid_sha, medoid_pixels = make_mask(
            medoid_path, 1, empty=empty_medoid and index == 0)
        c50_path = stability_path.parent / f"{sequence}-obj{obj_id}-c50.png"
        c75_path = stability_path.parent / f"{sequence}-obj{obj_id}-c75.png"
        c50_sha, c50_pixels = make_mask(c50_path, 1)
        c75_sha, c75_pixels = make_mask(c75_path, 2)
        review_items.append({
            "candidate_id": f"{sequence}:{frame_id}:camera{camera}:obj{obj_id}",
            "sequence": sequence, "frame_id": frame_id, "camera": camera,
            "obj_id": obj_id, "policy": "old_policy",
            "source_mask_name": current_path.name,
            "source_mask_sha256": current_sha, "mask_pixels": current_pixels,
            "projected_bbox": [240, 145, 285, 172],
            "candidate_only": True, "human_pixel_truth": False,
            "approved_for_training": False})
        stable_rows.append({
            "sequence": sequence, "frame_id": frame_id, "camera": camera,
            "target_obj_id": obj_id,
            "assistant_medoid_candidate": {
                "policy": "medoid_policy", "mask_name": medoid_path.name,
                "mask_sha256": medoid_sha, "candidate_only": True,
                "human_pixel_truth": False, "approved_for_training": False},
            "consensus50_mask": {"relative_path": c50_path.name,
                                 "sha256": c50_sha, "bytes": c50_path.stat().st_size,
                                 "pixels": c50_pixels},
            "consensus75_mask": {"relative_path": c75_path.name,
                                 "sha256": c75_sha, "bytes": c75_path.stat().st_size,
                                 "pixels": c75_pixels}})
    write_json(frozen_path, {"evaluation_only": True, "training_return_allowed": False,
                             "candidate_write_allowed": False,
                             "human_truth_write_allowed": False,
                             "workbench_writes": 0, "human_truth_writes": 0,
                             "oss_writes": 0, "files": files})
    common = {"candidate_only": True, "human_pixel_truth": False,
              "approved_for_training": False, "training_authorized": False,
              "automatic_approval": False, "human_truth_writes": 0,
              "training_data_writes": 0, "workbench_writes": 0,
              "oss_writes": 0, "production_deployed": False}
    write_json(review_path, {**common,
                             "status": "candidate_only_human_review_pack_not_deployed",
                             "items": review_items})
    write_json(stability_path, {**common,
                                "status": "offline_same_source_prompt_mask_stability_audit_not_deployed",
                                "independent_validation": False, "rows": stable_rows})
    return review_path, stability_path, frozen_path, mask_root


def authorize(monkeypatch, review: Path, stability: Path) -> None:
    monkeypatch.setattr(compare, "ROUND721_SHA", digest(review))
    monkeypatch.setattr(compare, "ROUND728_SHA", digest(stability))


def test_builds_three_same_roi_comparisons(tmp_path, monkeypatch):
    review, stability, frozen, masks = fixture(tmp_path)
    authorize(monkeypatch, review, stability)
    output = tmp_path / "output"
    result = compare.build_pack(review, stability, frozen, [masks], output)
    assert result["target_count"] == 3
    report = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert report["summary"] == {"target_count": 3, "comparison_image_count": 3}
    assert report["same_source_same_image"] is True
    assert report["independent_validation"] is False
    assert report["candidate_only"] is True
    assert report["human_pixel_truth"] is False
    assert report["training_authorized"] is False
    assert report["human_truth_writes"] == report["training_data_writes"] == 0
    assert report["workbench_writes"] == report["oss_writes"] == 0
    assert len(list(output.glob("*-comparison.png"))) == 3
    assert all(item["assistant_decision"] is None and item["human_decision"] is None
               for item in report["items"])


def test_rejects_duplicate_round721_identity(tmp_path, monkeypatch):
    review, stability, frozen, masks = fixture(tmp_path)
    value = json.loads(review.read_text(encoding="utf-8"))
    value["items"].append(dict(value["items"][0]))
    write_json(review, value)
    authorize(monkeypatch, review, stability)
    with pytest.raises(ValueError, match="duplicate Round721 identity"):
        compare.build_pack(review, stability, frozen, [masks], tmp_path / "output")


def test_rejects_hash_drift(tmp_path, monkeypatch):
    review, stability, frozen, masks = fixture(tmp_path)
    authorize(monkeypatch, review, stability)
    target = next(masks.glob("*-current.png"))
    Image.new("L", (640, 512), 255).save(target)
    with pytest.raises(ValueError, match="hash mismatch"):
        compare.build_pack(review, stability, frozen, [masks], tmp_path / "output")


def test_rejects_empty_mask(tmp_path, monkeypatch):
    review, stability, frozen, masks = fixture(tmp_path, empty_medoid=True)
    authorize(monkeypatch, review, stability)
    with pytest.raises(ValueError, match="empty mask"):
        compare.build_pack(review, stability, frozen, [masks], tmp_path / "output")


def test_rejects_existing_output(tmp_path, monkeypatch):
    review, stability, frozen, masks = fixture(tmp_path)
    authorize(monkeypatch, review, stability)
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(ValueError, match="output directory already exists"):
        compare.build_pack(review, stability, frozen, [masks], output)


def test_rejects_duplicate_frozen_image_identity(tmp_path, monkeypatch):
    review, stability, frozen, masks = fixture(tmp_path)
    authorize(monkeypatch, review, stability)
    value = json.loads(frozen.read_text(encoding="utf-8"))
    value["files"].append(dict(value["files"][0]))
    write_json(frozen, value)
    with pytest.raises(ValueError, match="duplicate Round674 image identity"):
        compare.build_pack(review, stability, frozen, [masks], tmp_path / "output")
