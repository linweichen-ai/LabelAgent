import hashlib
import json
from pathlib import Path

import pytest

from tools.build_pcd_manifest import build


def _manifest(path: Path) -> None:
    path.write_text(json.dumps({
        "schema_version": "labelagent.heldout_evaluation_manifest.v1",
        "evaluation_only": True,
        "training_return_allowed": False,
        "files": [
            {"sequence": "010", "frame_id": "100", "camera": "camera_0"},
            {"sequence": "010", "frame_id": "100", "camera": "camera_1"},
            {"sequence": "010", "frame_id": "200", "camera": "camera_0"},
        ],
    }), encoding="utf-8")


def test_build_binds_unique_frames(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    pcd_root = tmp_path / "pcd"
    output = tmp_path / "pcd-manifest.json"
    pcd_root.mkdir()
    _manifest(manifest)
    (pcd_root / "100.pcd").write_bytes(b"pcd-100")
    (pcd_root / "200.pcd").write_bytes(b"pcd-200")

    report = build(manifest, pcd_root, output)

    assert report["frame_count"] == 2
    assert [row["frame_id"] for row in report["files"]] == ["100", "200"]
    assert report["files"][0]["sha256"] == hashlib.sha256(b"pcd-100").hexdigest()
    assert report["oss_writes"] == 0


def test_build_rejects_missing_pair_without_output(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    pcd_root = tmp_path / "pcd"
    output = tmp_path / "pcd-manifest.json"
    pcd_root.mkdir()
    _manifest(manifest)
    (pcd_root / "100.pcd").write_bytes(b"pcd-100")

    with pytest.raises(ValueError, match="paired PCD is missing"):
        build(manifest, pcd_root, output)
    assert not output.exists()
