import hashlib
import json
from pathlib import Path

import pytest

from tools.build_heldout_directory_manifest import build


def _write_sample(root: Path, frame: str = "123") -> None:
    labels = root / "labels"
    labels.mkdir(parents=True)
    (labels / f"{frame}.json").write_text(
        json.dumps({"frame_id": frame, "objects": []}), encoding="utf-8")
    for camera in range(4):
        path = root / "cameras" / f"camera_{camera}" / f"{frame}.png"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"png fixture " + str(camera).encode())


def test_build_freezes_images_and_stages_labels(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_sample(source)

    receipt = build(source, output, dataset="fixture", sequence="dense")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    assert receipt["image_count"] == 4
    assert manifest["evaluation_only"] is True
    assert manifest["training_return_allowed"] is False
    assert manifest["human_truth_write_allowed"] is False
    assert manifest["frame_count"] == 1
    assert len(manifest["files"]) == 4
    staged = output / "labels" / "dense" / "123.json"
    assert staged.is_file()
    assert hashlib.sha256(staged.read_bytes()).hexdigest() == manifest["label_files"][0]["sha256"]


def test_build_rejects_missing_camera_pair_without_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_sample(source)
    (source / "cameras" / "camera_3" / "123.png").unlink()

    with pytest.raises(ValueError, match="paired image is missing"):
        build(source, output, dataset="fixture", sequence="dense")
    assert not output.exists()


def test_build_rejects_frame_identity_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_sample(source)
    (source / "labels" / "123.json").write_text(
        json.dumps({"frame_id": "456", "objects": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="frame_id mismatch"):
        build(source, output, dataset="fixture", sequence="dense")
    assert not output.exists()
