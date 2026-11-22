import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from tools import build_pallet_uncertain_zoom_review as zoom


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_fixture(root: Path):
    review_path = root / "round721" / "manifest.json"
    frozen_path = root / "round674" / "manifest.json"
    mask_root = root / "masks"
    groups = {
        "006": ("1789899601982497364", ["10", "11", "12", "13", "14", "15", "16", "23"]),
        "008": ("1789899778882397075", ["67", "69", "75", "76", "77", "78", "79"]),
    }
    files, frames, items = [], [], []
    selected = {(seq, obj) for seq, _, _, obj in zoom.EXPECTED}
    for sequence, (frame_id, members) in groups.items():
        image_path = frozen_path.parent / "images" / sequence / "camera_1" / f"{frame_id}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (640, 512), (85, 100, 115)).save(image_path)
        label_path = frozen_path.parent / "labels" / sequence / f"{frame_id}.json"
        pallet_ids = list(members)
        value = 300 if sequence == "006" else 400
        while len(pallet_ids) < 20:
            pallet_ids.append(str(value)); value += 1
        write_json(label_path, {"frame_id": frame_id,
                                "objects": [{"obj_type": "Pallet", "obj_id": obj_id}
                                            for obj_id in pallet_ids]})
        files.append({"sequence": sequence, "frame_id": frame_id,
                      "camera": "camera_1",
                      "relative_path": image_path.relative_to(frozen_path.parent).as_posix(),
                      "sha256": digest(image_path)})
        frames.append({"sequence": sequence, "frame_id": frame_id, "camera": 1,
                       "source_image_name": image_path.name,
                       "source_image_sha256": digest(image_path),
                       "source_label_name": label_path.name,
                       "source_label_sha256": digest(label_path),
                       "review_scope_obj_ids": members})
        for index, obj_id in enumerate(members):
            item = {"candidate_id": f"{sequence}:{frame_id}:camera1:obj{obj_id}",
                    "sequence": sequence, "frame_id": frame_id, "camera": 1,
                    "obj_id": obj_id,
                    "projected_bbox": [240 + index, 145 + index, 286 + index, 165 + index],
                    "candidate_only": True, "human_pixel_truth": False,
                    "approved_for_training": False}
            if (sequence, obj_id) in selected:
                mask_path = mask_root / f"{sequence}-obj{obj_id}.png"
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                mask = Image.new("L", (640, 512), 0)
                for x in range(252 + index, 258 + index):
                    for y in range(151 + index, 155 + index):
                        mask.putpixel((x, y), 255)
                mask.save(mask_path)
                item.update({"source_mask_name": mask_path.name,
                             "source_mask_sha256": digest(mask_path)})
            items.append(item)
    frozen = {"evaluation_only": True, "training_return_allowed": False,
              "candidate_write_allowed": False, "human_truth_write_allowed": False,
              "workbench_writes": 0, "human_truth_writes": 0, "oss_writes": 0,
              "files": files}
    review = {"status": "candidate_only_human_review_pack_not_deployed",
              "frames": frames, "items": items, "candidate_only": True,
              "human_pixel_truth": False, "approved_for_training": False,
              "training_authorized": False, "automatic_approval": False,
              "human_truth_writes": 0, "training_data_writes": 0,
              "workbench_writes": 0, "oss_writes": 0,
              "production_deployed": False}
    write_json(frozen_path, frozen)
    write_json(review_path, review)
    return review_path, frozen_path, mask_root


def authorize_fixture(monkeypatch, review_path: Path) -> None:
    monkeypatch.setattr(zoom, "EXPECTED_REVIEW_MANIFEST_SHA", digest(review_path))


def test_builds_six_pixel_faithful_zoom_cards(tmp_path, monkeypatch):
    review, frozen, masks = make_fixture(tmp_path)
    authorize_fixture(monkeypatch, review)
    output = tmp_path / "output"
    result = zoom.build_pack(review, frozen, [masks], output)
    assert result["zoom_item_count"] == 6
    report = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert report["summary"] == {"zoom_item_count": 6, "source_frame_count": 2}
    assert report["source_resolution"] == [640, 512]
    assert report["no_new_visual_information"] is True
    assert report["candidate_only"] is True
    assert report["human_pixel_truth"] is False
    assert report["approved_for_training"] is False
    assert report["training_authorized"] is False
    assert report["automatic_approval"] is False
    assert len(list(output.glob("*-zoom.png"))) == 6
    with Image.open(output / report["items"][0]["relative_path"]) as rendered:
        assert rendered.size == (1480, 430)


def test_rejects_wrong_round721_identity(tmp_path):
    review, frozen, masks = make_fixture(tmp_path)
    with pytest.raises(ValueError, match="Round721 manifest identity mismatch"):
        zoom.build_pack(review, frozen, [masks], tmp_path / "output")


def test_rejects_duplicate_item_identity(tmp_path, monkeypatch):
    review, frozen, masks = make_fixture(tmp_path)
    value = json.loads(review.read_text(encoding="utf-8"))
    value["items"].append(dict(value["items"][0]))
    write_json(review, value)
    authorize_fixture(monkeypatch, review)
    with pytest.raises(ValueError, match="duplicate Round721 item"):
        zoom.build_pack(review, frozen, [masks], tmp_path / "output")


def test_rejects_candidate_mask_hash_drift(tmp_path, monkeypatch):
    review, frozen, masks = make_fixture(tmp_path)
    authorize_fixture(monkeypatch, review)
    target = masks / "006-obj11.png"
    Image.new("L", (640, 512), 255).save(target)
    with pytest.raises(ValueError, match="candidate_mask_006_obj11 hash mismatch"):
        zoom.build_pack(review, frozen, [masks], tmp_path / "output")


def test_rejects_existing_output(tmp_path, monkeypatch):
    review, frozen, masks = make_fixture(tmp_path)
    authorize_fixture(monkeypatch, review)
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(ValueError, match="output directory already exists"):
        zoom.build_pack(review, frozen, [masks], output)


def test_rejects_duplicate_frozen_image_identity(tmp_path, monkeypatch):
    review, frozen, masks = make_fixture(tmp_path)
    authorize_fixture(monkeypatch, review)
    value = json.loads(frozen.read_text(encoding="utf-8"))
    value["files"].append(dict(value["files"][0]))
    write_json(frozen, value)
    with pytest.raises(ValueError, match="duplicate Round674 image identity"):
        zoom.build_pack(review, frozen, [masks], tmp_path / "output")
