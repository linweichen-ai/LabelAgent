"""Freeze an unseen multi-camera/LiDAR slice before any visual or model inspection."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def evenly_spaced(values: list[str], count: int) -> list[str]:
    if count < 2 or len(values) < count or values != sorted(values) or len(set(values)) != len(values):
        raise ValueError("Need a sorted unique population at least as large as the requested frozen slice")
    indices = [index * (len(values) - 1) // (count - 1) for index in range(count)]
    selected = [values[index] for index in indices]
    if len(set(selected)) != count or selected[0] != values[0] or selected[-1] != values[-1]:
        raise ValueError("Even spacing failed to preserve unique endpoints")
    return selected


def file_record(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"Missing frozen source: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def freeze(data_root: Path, dataset: str, sequence: str, count: int) -> dict:
    data_root = data_root.resolve()
    cameras = {
        camera: {path.stem for path in (data_root / "cameras_640x512" / f"camera_{camera}").glob("*.png")}
        for camera in range(4)
    }
    lidar = {path.stem for path in (data_root / "lidar").glob("*.pcd")}
    common = sorted(set.intersection(lidar, *cameras.values()))
    selected = evenly_spaced(common, count)
    calibration = data_root / "calib_640x512"
    expected_calibration = [calibration / f"camera{camera}_calib.yaml" for camera in range(4)] + [calibration / "lidar2camera.yaml"]
    records = []
    for frame in selected:
        records.append({
            "frame_id": frame,
            "lidar": file_record(data_root / "lidar" / f"{frame}.pcd"),
            "images": {
                f"camera_{camera}": file_record(
                    data_root / "cameras_640x512" / f"camera_{camera}" / f"{frame}.png"
                )
                for camera in range(4)
            },
        })
    return {
        "schema_version": "labelagent.slope-geometry-external-freeze.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_visual_or_model_inspection",
        "dataset": dataset,
        "sequence": sequence,
        "selection": {
            "method": "first nonempty sequence chosen by directory order; exact four-camera/LiDAR intersection; evenly spaced including endpoints",
            "population_frames": len(common),
            "selected_frames": count,
            "indices": [index * (len(common) - 1) // (count - 1) for index in range(count)],
            "frame_ids": selected,
        },
        "boundaries": {
            "images_inspected_before_freeze": False,
            "model_outputs_inspected_before_freeze": False,
            "labels_read": False,
            "human_truth": False,
            "training_authorized": False,
            "production_deployed": False,
        },
        "calibration": [file_record(path) for path in expected_calibration],
        "frames": records,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not 4 <= args.count <= 64:
        parser.error("--count must be between 4 and 64")
    if args.output.exists():
        parser.error("--output must not already exist")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    manifest = freeze(arguments.data_root, arguments.dataset, arguments.sequence, arguments.count)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest["selection"], ensure_ascii=False, separators=(",", ":")), flush=True)
