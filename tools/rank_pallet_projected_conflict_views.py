#!/usr/bin/env python3
"""Rank eval views by projected pallet conflicts without treating predictions as truth.

The local ``ali_label`` files are used only to choose difficult views.  Any
downstream evaluation must replace them with the frozen human 3-D labels named
by the sample manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.evaluate_pallet_box3d_overlap_pairs import (
    projected_conflict_component, projected_conflict_edges,
)
from tools.evaluate_pallet_box3d_roi_sam import load_camera, project_box


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(sequence: str, frame_id: str, camera: int | str) -> tuple[str, str, int]:
    return str(sequence), str(frame_id), int(str(camera).split("_")[-1])


def excluded_identities(paths: list[Path]) -> tuple[set[tuple[str, str, int]], set[str]]:
    excluded, sequences = set(), set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("files") or []:
            key = identity(row["sequence"], row["frame_id"], row["camera"])
            excluded.add(key)
            sequences.add(key[0])
    return excluded, sequences


def conflict_metrics(projected: list[dict]) -> dict:
    seen: set[str] = set()
    components = []
    for row in projected:
        obj_id = str(row["obj_id"])
        if obj_id in seen:
            continue
        component = projected_conflict_component(row, projected)
        ids = {str(item["obj_id"]) for item in component}
        seen.update(ids)
        edges = projected_conflict_edges(component)
        if not edges:
            continue
        components.append({
            "object_ids": sorted(ids, key=lambda value: (len(value), value)),
            "object_count": len(component),
            "edge_count": len(edges),
            "projected_pixels": sum(int(item["mask"].sum()) for item in component),
        })
    components.sort(key=lambda row: (
        -row["object_count"], -row["edge_count"], -row["projected_pixels"],
        row["object_ids"],
    ))
    return {
        "visible_pallet_count": len(projected),
        "conflict_component_count": len(components),
        "conflicted_object_count": sum(row["object_count"] for row in components),
        "conflict_edge_count": sum(row["edge_count"] for row in components),
        "maximum_component_size": max((row["object_count"] for row in components), default=0),
        "maximum_component_edge_count": max((row["edge_count"] for row in components), default=0),
        "components": components,
    }


def rank_key(row: dict) -> tuple:
    return (
        -row["maximum_component_size"],
        -row["maximum_component_edge_count"],
        -row["conflict_edge_count"],
        -row["conflicted_object_count"],
        -row["visible_pallet_count"],
        row["sequence"], row["frame_id"], row["camera"],
    )


def select_views(rows: list[dict], count: int, previously_used_sequences: set[str]) -> list[dict]:
    eligible = sorted((row for row in rows if row["conflict_edge_count"]), key=rank_key)
    best_by_sequence: dict[str, dict] = {}
    for row in eligible:
        best_by_sequence.setdefault(row["sequence"], row)
    sequence_best = list(best_by_sequence.values())
    sequence_best.sort(key=lambda row: (
        row["sequence"] in previously_used_sequences, *rank_key(row),
    ))
    selected = sequence_best[:count]
    if len(selected) < count:
        selected_ids = {identity(row["sequence"], row["frame_id"], row["camera"])
                        for row in selected}
        selected.extend(row for row in eligible
                        if identity(row["sequence"], row["frame_id"], row["camera"])
                        not in selected_ids)
        selected = selected[:count]
    return selected


def run(sample_manifest: Path, metadata_root: Path, calibration: Path,
        exclude_manifests: list[Path], count: int) -> dict:
    manifest = json.loads(sample_manifest.read_text(encoding="utf-8"))
    excluded, used_sequences = excluded_identities(exclude_manifests)
    frozen_paths = [sample_manifest, *exclude_manifests,
                    calibration / "lidar2camera.yaml",
                    *(calibration / f"camera{index}_calib.yaml"
                      for index in range(4))]
    inputs = {str(path.resolve()): digest(path) for path in frozen_paths}
    cameras = {index: load_camera(calibration, index) for index in range(4)}
    rows = []
    missing_prediction_count = 0
    for frame in manifest["frames"]:
        sequence, frame_id = str(frame["sequence"]), str(frame["frame_id"])
        source = frame.get("sources", {}).get("ali_label") or {}
        if not source.get("exists"):
            missing_prediction_count += 1
            continue
        label_path = metadata_root / source["relative_path"]
        label_sha = digest(label_path)
        if label_sha != source.get("sha256"):
            raise ValueError(f"selection label changed: {label_path}")
        inputs[str(label_path.resolve())] = label_sha
        label = json.loads(label_path.read_text(encoding="utf-8"))
        pallets = [obj for obj in label.get("objects") or []
                   if obj.get("obj_type") == "Pallet" and not obj.get("ignore")]
        for camera_index, camera in cameras.items():
            key = identity(sequence, frame_id, camera_index)
            if key in excluded:
                continue
            projected = [value for obj in pallets
                         if (value := project_box(obj, camera)) is not None]
            rows.append({
                "sequence": sequence,
                "frame_id": frame_id,
                "camera": camera_index,
                "selection_label_relative_path": source["relative_path"],
                **conflict_metrics(projected),
            })
    rows.sort(key=rank_key)
    selected = select_views(rows, count, used_sequences)
    if any(digest(Path(path)) != value for path, value in inputs.items()):
        raise RuntimeError("frozen ranking input changed during evaluation")
    return {
        "schema_version": "labelagent.pallet_projected_conflict_view_ranking.v1",
        "status": "selection_only_not_evaluation",
        "selection_source": "ali_label_preannotation_3d_boxes",
        "selection_source_is_truth": False,
        "downstream_evaluation_requires_manifest_gt": True,
        "sample_manifest": str(sample_manifest),
        "sample_manifest_sha256": digest(sample_manifest),
        "calibration": str(calibration),
        "exclude_manifests": [str(path) for path in exclude_manifests],
        "exclude_manifest_sha256": {str(path): digest(path) for path in exclude_manifests},
        "input_sha256": inputs,
        "excluded_view_count": len(excluded),
        "previously_used_sequences": sorted(used_sequences),
        "missing_prediction_frame_count": missing_prediction_count,
        "ranked_view_count": len(rows),
        "conflict_view_count": sum(bool(row["conflict_edge_count"]) for row in rows),
        "selected_count": len(selected),
        "selection_policy": "best_conflict_view_per_sequence_unseen_sequences_first_then_rank_fill",
        "selected": selected,
        "rows": rows,
        "human_truth_writes": 0,
        "training_writes": 0,
        "workbench_writes": 0,
        "oss_writes": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.sample_manifest, arguments.metadata_root,
                 arguments.calibration, arguments.exclude_manifest, arguments.count)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "ranked_view_count", "conflict_view_count", "selected_count", "selected")},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
