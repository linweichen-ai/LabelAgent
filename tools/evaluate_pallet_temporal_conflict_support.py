#!/usr/bin/env python3
"""Batch frozen temporal support checks for unresolved pallet conflicts.

This is an offline candidate diagnostic.  It never promotes temporal evidence
to pixel truth, training data, a workbench write, or a deployed decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.evaluate_pallet_temporal_visibility import evaluate as evaluate_temporal


OVERLAP_SCHEMA = "labelagent.pallet_box3d_overlap_pairs.v1"
MANIFEST_SCHEMA = "labelagent.pallet_temporal_conflict_manifest.v1"
REPORT_SCHEMA = "labelagent.pallet_temporal_conflict_support.v1"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _finite(value: object, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {name}") from error
    if not math.isfinite(number):
        raise ValueError(f"invalid {name}")
    return number


def validate_overlap_report(report: dict) -> None:
    if (report.get("schema_version") != OVERLAP_SCHEMA
            or report.get("status") != "offline_instance_diagnostic_not_deployed"
            or report.get("human_pixel_truth") is not False
            or report.get("approved_for_training") is not False
            or report.get("production_deployed") is not False
            or report.get("workbench_writes") != 0
            or not isinstance(report.get("input_sha256"), dict)
            or not report["input_sha256"]):
        raise ValueError("requires a frozen non-deployed overlap split report")
    if not isinstance(report.get("rows"), list):
        raise ValueError("overlap split rows are absent")


def select_nearer_occluder(row: dict, target_metric: dict
                            ) -> tuple[dict | None, str | None]:
    """Select the most-overlapping conflicting instance strictly nearer in depth."""
    target_id = str(target_metric["obj_id"])
    target_depth = _finite(target_metric.get("camera_depth"), "target camera depth")
    if target_depth <= 0:
        raise ValueError("target camera depth must be positive")
    metrics = row.get("depth_split_metrics")
    pairs = row.get("pairs")
    if not isinstance(metrics, list) or not isinstance(pairs, list):
        raise ValueError("overlap row is missing metrics or conflict pairs")
    depth_by_id: dict[str, float] = {}
    for metric in metrics:
        obj_id = str(metric["obj_id"])
        if obj_id in depth_by_id:
            raise ValueError(f"duplicate depth metric for obj {obj_id}")
        depth = _finite(metric.get("camera_depth"), "camera depth")
        if depth <= 0:
            raise ValueError("camera depth must be positive")
        depth_by_id[obj_id] = depth
    if (target_id not in depth_by_id
            or depth_by_id[target_id] != target_depth):
        raise ValueError("target depth metric differs from overlap row")

    relevant = []
    for pair in pairs:
        first = str(pair.get("target_obj_id"))
        second = str(pair.get("other_obj_id"))
        if target_id == first:
            other_id = second
            overlap = _finite(pair.get("projected_target_overlap"),
                              "projected target overlap")
        elif target_id == second:
            other_id = first
            overlap = _finite(pair.get("projected_other_overlap"),
                              "projected other overlap")
        else:
            continue
        if not 0 <= overlap <= 1:
            raise ValueError("projected overlap must be in [0, 1]")
        if other_id not in depth_by_id:
            raise ValueError(f"conflict peer has no depth metric: {other_id}")
        relevant.append((other_id, depth_by_id[other_id], overlap))
    if not relevant:
        return None, "no_conflict_pair_for_target"
    nearer = [item for item in relevant if item[1] < target_depth]
    if not nearer:
        return None, "no_nearer_conflicting_occluder"
    nearer.sort(key=lambda item: (-item[2], item[1], item[0]))
    other_id, other_depth, overlap = nearer[0]
    return {
        "target_obj_id": target_id,
        "target_camera_depth": target_depth,
        "occluder_obj_id": other_id,
        "occluder_camera_depth": other_depth,
        "target_projected_overlap": overlap,
    }, None


def plan_targets(report: dict) -> tuple[list[dict], list[dict]]:
    """Return unresolved jobs and explicit fail-closed skips."""
    validate_overlap_report(report)
    jobs, skipped = [], []
    identities = set()
    for row in report["rows"]:
        sequence = str(row["sequence"])
        frame_id = str(row["frame_id"])
        camera = int(row["camera"])
        metrics = row.get("depth_split_metrics")
        if not isinstance(metrics, list):
            raise ValueError("overlap row has no depth split metrics")
        for metric in metrics:
            if not isinstance(metric.get("handled_candidate"), bool):
                raise ValueError("handled_candidate must be an explicit boolean")
            if metric["handled_candidate"] is not False:
                continue
            target_id = str(metric["obj_id"])
            identity = (sequence, frame_id, camera, target_id)
            if identity in identities:
                raise ValueError(f"duplicate unresolved target: {identity}")
            identities.add(identity)
            selected, reason = select_nearer_occluder(row, metric)
            base = {
                "sequence": sequence,
                "center_frame_id": frame_id,
                "camera": camera,
                "target_obj_id": target_id,
            }
            if selected is None:
                skipped.append({**base, "status": "skipped", "reason": reason})
            else:
                jobs.append({**base, **selected})
    return jobs, skipped


def classify_temporal_evidence(temporal: dict) -> tuple[str, str]:
    if temporal.get("temporal_occlusion_supported_for_center") is True:
        return "occlusion_support", "defer_2d"
    if temporal.get("temporal_presence_supported_for_center") is True:
        return "label_conditioned_presence_support_only", "review_support_only"
    return "insufficient", "insufficient"


def _frame_sort_key(frame_id: str) -> tuple[int, int | str]:
    return (0, int(frame_id)) if frame_id.isdecimal() else (1, frame_id)


def _sequence_root(window_root: Path, sequence: str) -> Path:
    root = window_root.resolve()
    candidate = (root / sequence).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"sequence escapes frozen window root: {sequence}") from error
    return candidate


def freeze_window(window_root: Path, sequence: str) -> tuple[Path, list[dict], dict[str, str]]:
    sequence_root = _sequence_root(window_root, sequence)
    labels_root = sequence_root / "labels"
    pcd_root = sequence_root / "pcd"
    if not labels_root.is_dir() or not pcd_root.is_dir():
        raise ValueError(f"frozen sequence window is incomplete: {sequence}")
    labels = {path.stem: path for path in labels_root.glob("*.json") if path.is_file()}
    pcds = {path.stem: path for path in pcd_root.glob("*.pcd") if path.is_file()}
    if not labels or set(labels) != set(pcds):
        raise ValueError(f"label/PCD frame identities differ: {sequence}")
    frames, frozen = [], {}
    for frame_id in sorted(labels, key=_frame_sort_key):
        label_path, pcd_path = labels[frame_id], pcds[frame_id]
        payload = json.loads(label_path.read_text(encoding="utf-8"))
        if str(payload.get("frame_id")) != frame_id:
            raise ValueError(f"label frame identity differs from filename: {label_path}")
        label_sha, pcd_sha = digest(label_path), digest(pcd_path)
        frozen[str(label_path.resolve())] = label_sha
        frozen[str(pcd_path.resolve())] = pcd_sha
        frames.append({
            "frame_id": frame_id,
            "label_size": label_path.stat().st_size,
            "label_sha256": label_sha,
            "pcd_size": pcd_path.stat().st_size,
            "pcd_sha256": pcd_sha,
        })
    return sequence_root, frames, frozen


def _calibration_inputs(calibration: Path, cameras: set[int]) -> dict[str, str]:
    paths = [calibration / "lidar2camera.yaml"]
    paths.extend(calibration / f"camera{camera}_calib.yaml"
                 for camera in sorted(cameras))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"calibration inputs missing: {missing}")
    return {str(path.resolve()): digest(path) for path in paths}


def _verify_frozen(frozen: dict[str, str]) -> None:
    drifted = [path for path, expected in frozen.items()
               if not Path(path).is_file() or digest(Path(path)) != expected]
    if drifted:
        raise RuntimeError(f"frozen temporal conflict input changed: {drifted}")


def validate_window_centers(jobs: list[dict],
                            windows: dict[str, tuple[Path, list[dict]]]) -> None:
    for job in jobs:
        frame_ids = {item["frame_id"] for item in windows[job["sequence"]][1]}
        if job["center_frame_id"] not in frame_ids:
            raise ValueError(f"center frame is absent from frozen window: {job}")


def evaluate(overlap_report: Path, window_root: Path, calibration: Path,
             output: Path) -> dict:
    if output.exists():
        raise ValueError("output already exists")
    overlap_bytes = overlap_report.read_bytes()
    overlap = json.loads(overlap_bytes.decode("utf-8"))
    jobs, skipped = plan_targets(overlap)
    frozen = {str(overlap_report.resolve()): hashlib.sha256(overlap_bytes).hexdigest()}
    frozen.update(_calibration_inputs(calibration,
                                      {int(job["camera"]) for job in jobs}))
    windows: dict[str, tuple[Path, list[dict]]] = {}
    for sequence in sorted({str(job["sequence"]) for job in jobs}):
        root, frames, hashes = freeze_window(window_root, sequence)
        windows[sequence] = (root, frames)
        frozen.update(hashes)
    validate_window_centers(jobs, windows)

    output.mkdir(parents=True)
    manifests = output / "manifests"
    evaluations = output / "evaluations"
    manifests.mkdir(); evaluations.mkdir()
    rows = []
    try:
        for index, job in enumerate(jobs, start=1):
            sequence_root, frames = windows[job["sequence"]]
            manifest = {
                "schema_version": MANIFEST_SCHEMA,
                "status": "frozen_temporal_candidate_input_not_truth",
                "sequence": job["sequence"],
                "camera": job["camera"],
                "target_obj_id": job["target_obj_id"],
                "occluder_obj_id": job["occluder_obj_id"],
                "center_frame_id": job["center_frame_id"],
                "frames": frames,
                "evaluation_only": True,
                "training_return_allowed": False,
                "candidate_write_allowed": False,
                "human_truth_write_allowed": False,
                "human_pixel_truth": False,
                "approved_for_training": False,
                "production_deployed": False,
            }
            manifest_path = manifests / f"case-{index:03d}.json"
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
            evaluation_dir = evaluations / f"case-{index:03d}"
            temporal = evaluate_temporal(
                manifest_path, sequence_root, calibration, evaluation_dir)
            evidence_status, decision = classify_temporal_evidence(temporal)
            rows.append({
                **job,
                "evidence_status": evidence_status,
                "decision": decision,
                "temporal_manifest_path": str(manifest_path.resolve()),
                "temporal_manifest_sha256": digest(manifest_path),
                "temporal_report_path": temporal["report_path"],
                "temporal_report_sha256": temporal["report_sha256"],
                "target_present_frames": temporal["target_present_frames"],
                "frame_count": temporal["frame_count"],
                "target_supported_before_center": temporal[
                    "target_supported_before_center"],
                "target_supported_after_center": temporal[
                    "target_supported_after_center"],
                "human_pixel_truth": False,
                "approved_for_training": False,
                "candidate_only": True,
                "requires_review": True,
                "evidence_source": "human_3d_track_identity_plus_lidar_visibility",
            })
        _verify_frozen(frozen)
        report = {
            "schema_version": REPORT_SCHEMA,
            "status": "offline_temporal_conflict_support_not_deployed",
            "overlap_report_sha256": hashlib.sha256(overlap_bytes).hexdigest(),
            "input_sha256": frozen,
            "unresolved_target_count": len(jobs) + len(skipped),
            "evaluated_target_count": len(rows),
            "any_review_support_count": sum(
                row["evidence_status"] != "insufficient" for row in rows),
            "occlusion_support_count": sum(
                row["evidence_status"] == "occlusion_support" for row in rows),
            "presence_support_only_count": sum(
                row["evidence_status"] ==
                "label_conditioned_presence_support_only" for row in rows),
            "defer_count": sum(row["decision"] == "defer_2d" for row in rows),
            "insufficient_count": sum(
                row["evidence_status"] == "insufficient" for row in rows),
            "skipped_count": len(skipped),
            "skipped_no_nearer_occluder_count": sum(
                row["reason"] == "no_nearer_conflicting_occluder"
                for row in skipped),
            "rows": rows,
            "skipped": skipped,
            "candidate_only": True,
            "conditioned_on_human_3d_labels": True,
            "not_end_to_end_detection": True,
            "human_pixel_truth": False,
            "approved_for_training": False,
            "human_truth_writes": 0,
            "training_data_writes": 0,
            "workbench_writes": 0,
            "oss_writes": 0,
            "production_deployed": False,
        }
        report_path = output / "results.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        report["report_path"] = str(report_path.resolve())
        report["report_sha256"] = digest(report_path)
        return report
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlap-report", required=True, type=Path)
    parser.add_argument("--window-root", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    report = evaluate(arguments.overlap_report, arguments.window_root,
                      arguments.calibration, arguments.output)
    print(json.dumps({key: report[key] for key in (
        "unresolved_target_count", "evaluated_target_count",
        "any_review_support_count",
        "occlusion_support_count", "presence_support_only_count",
        "defer_count", "insufficient_count", "skipped_count",
        "skipped_no_nearer_occluder_count", "human_truth_writes",
        "training_data_writes", "workbench_writes", "oss_writes",
        "production_deployed", "report_path", "report_sha256")},
        ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
