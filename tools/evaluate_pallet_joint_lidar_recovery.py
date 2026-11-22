#!/usr/bin/env python3
"""Jointly re-evaluate failed pallet masks from frozen LiDAR prompt reports."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.audit_pallet_candidate_pixels import structure_metrics, structural_alerts
from tools.evaluate_pallet_box3d_overlap_pairs import (
    depth_split_masks, mask_separation, projected_conflict_component,
    projected_conflict_edges, projected_conflicts,
)
from tools.evaluate_pallet_box3d_roi_sam import (
    association_metrics, digest, load_camera, localization_gate, project_box,
)


def load_mask(path: Path) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if value is None:
        raise ValueError(f"cannot read mask: {path}")
    return value > 0


def point_evidence(mask: np.ndarray, positives, negatives) -> dict:
    positive = (sum(bool(mask[y, x]) for x, y in positives) / len(positives)
                if positives else 0.0)
    negative = (sum(not bool(mask[y, x]) for x, y in negatives) / len(negatives)
                if negatives else 0.0)
    return {"positive_coverage": positive,
            "negative_clear_fraction": negative,
            "point_gate_pass": bool(positive >= .8 and negative >= .8)}


def locally_supported(mask, target, component, positives, negatives) -> dict:
    direct_conflicts = [row for row, _, _ in projected_conflicts(target, component)]
    metrics = association_metrics(
        mask, target["mask"], [row["mask"] for row in direct_conflicts],
        target["center"])
    evidence = point_evidence(mask, positives, negatives)
    return {**evidence, "localization_metrics": metrics,
            "localization_pass": localization_gate(metrics),
            "local_support_pass": bool(
                evidence["point_gate_pass"] and localization_gate(metrics))}


def mask_sha(mask: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()


def recovery_index(reports: list[dict]) -> dict[tuple[str, str, int, str], list[dict]]:
    result = {}
    for report_index, report in enumerate(reports):
        for row in report["rows"]:
            key = (str(row["sequence"]), str(row["frame_id"]),
                   int(row["camera"]), str(row["target_obj_id"]))
            result.setdefault(key, []).append({**row, "report_index": report_index})
    return result


def evaluate(split_report_path: Path, recovery_report_paths: list[Path],
             labels_root: Path, calibration: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("output directory already exists")
    split_report = json.loads(split_report_path.read_text(encoding="utf-8"))
    recovery_reports = [json.loads(path.read_text(encoding="utf-8"))
                        for path in recovery_report_paths]
    recoveries = recovery_index(recovery_reports)
    output.mkdir(parents=True); masks_dir = output / "masks"; masks_dir.mkdir()
    rows = []
    for split_row in split_report["rows"]:
        baseline_metrics = {str(row["obj_id"]): row
                            for row in split_row["depth_split_metrics"]}
        if sum(not row["handled_candidate"] for row in baseline_metrics.values()) < 2:
            continue
        sequence, frame, camera_number = (
            str(split_row["sequence"]), str(split_row["frame_id"]),
            int(split_row["camera"]))
        label = json.loads((labels_root / sequence / f"{frame}.json").read_text(
            encoding="utf-8"))
        camera = load_camera(calibration, camera_number)
        projected = [value for obj in label["objects"]
                     if obj.get("obj_type") == "Pallet" and not obj.get("ignore")
                     if (value := project_box(obj, camera)) is not None]
        target = next(row for row in projected
                      if str(row["obj_id"]) == str(split_row["target_obj_id"]))
        component = projected_conflict_component(target, projected)
        baseline_masks = {
            str(obj_id): load_mask(Path(value["path"]))
            for obj_id, value in split_row["depth_split_masks"].items()}
        variant_sets, variant_audit = [], {}
        for projected_row in component:
            obj_id = str(projected_row["obj_id"])
            baseline = baseline_masks[obj_id]
            variants = []
            baseline_metric = baseline_metrics[obj_id]
            if baseline_metric["handled_candidate"]:
                variants.append({"name": "baseline_handled", "mask": baseline,
                                 "local_support_pass": True,
                                 "source_report_index": None})
            for recovery_row in recoveries.get(
                    (sequence, frame, camera_number, obj_id), []):
                positives = recovery_row["selected_positive_points"]
                negatives = recovery_row["selected_negative_points"]
                for attempt_index, attempt in enumerate(recovery_row["attempts"]):
                    if not attempt.get("mask_path"):
                        continue
                    attempt_mask = load_mask(Path(
                        attempt.get("raw_mask_path") or attempt["mask_path"]))
                    for kind, candidate in (
                            ("attempt", attempt_mask),
                            ("baseline_union_attempt", baseline | attempt_mask)):
                        # The saved mask is the target's already depth-split
                        # result, while the frozen report's localization
                        # metrics were measured on the pre-split proposal.
                        # Preserve that original localization receipt and
                        # re-check point evidence on the exact mask used here.
                        if kind == "attempt":
                            evidence = point_evidence(candidate, positives, negatives)
                            localization_pass = localization_gate(attempt["metrics"])
                            support = {
                                **evidence,
                                "localization_metrics": attempt["metrics"],
                                "localization_pass": localization_pass,
                                "local_support_pass": bool(
                                    evidence["point_gate_pass"] and localization_pass),
                            }
                        else:
                            support = locally_supported(
                                candidate, projected_row, component,
                                positives, negatives)
                        if not support["local_support_pass"]:
                            continue
                        variants.append({
                            "name": f"r{recovery_row['report_index']}-a{attempt_index}-{kind}",
                            "mask": candidate,
                            "source_report_index": recovery_row["report_index"],
                            "source_policy": attempt.get("policy"), **support})
            unique = {}
            for variant in variants:
                identity = mask_sha(variant["mask"])
                unique.setdefault(identity, variant)
            variants = list(unique.values())
            variant_sets.append(variants)
            variant_audit[obj_id] = [{key: value for key, value in item.items()
                                      if key != "mask"} | {
                                          "mask_pixels": int(item["mask"].sum()),
                                          "mask_array_sha256": mask_sha(item["mask"])}
                                     for item in variants]
        if any(not values for values in variant_sets):
            rows.append({"sequence": sequence, "frame_id": frame,
                         "camera": camera_number, "joint_recovered": False,
                         "reason": "at_least_one_unhandled_instance_has_no_locally_supported_variant",
                         "variant_audit": variant_audit})
            continue
        accepted = []
        edges = projected_conflict_edges(component)
        for combination in itertools.product(*variant_sets):
            candidate_masks = {row["obj_id"]: variant["mask"]
                               for row, variant in zip(component, combination)}
            split_masks, split_metrics = depth_split_masks(component, candidate_masks)
            separations = [{"first_obj_id": first["obj_id"],
                            "second_obj_id": second["obj_id"],
                            **mask_separation(split_masks[first["obj_id"]],
                                              split_masks[second["obj_id"]])}
                           for first, second, _, _ in edges]
            handled = all(row["handled_candidate"] for row in split_metrics)
            separated = all(row["separated_at_0_15"] for row in separations)
            if handled and separated:
                alert_count = sum(
                    bool(structural_alerts(structure_metrics(
                        split_masks[row["obj_id"]])))
                    for row in split_metrics if row["candidate_pass"])
                accepted.append((
                    -sum(row["candidate_pass"] for row in split_metrics),
                    alert_count,
                    sum(int(mask.sum()) for mask in split_masks.values()),
                    combination, split_masks, split_metrics, separations))
        accepted.sort(key=lambda item: item[:3])
        if not accepted:
            rows.append({"sequence": sequence, "frame_id": frame,
                         "camera": camera_number, "joint_recovered": False,
                         "reason": "no_joint_combination_passed_handled_and_separation_gates",
                         "variant_audit": variant_audit})
            continue
        _, selected_alert_count, _, selected, split_masks, split_metrics, separations = accepted[0]
        emitted = {}
        for projected_row, variant in zip(component, selected):
            obj_id = str(projected_row["obj_id"])
            metric = next(row for row in split_metrics
                          if str(row["obj_id"]) == obj_id)
            if not metric["candidate_pass"]:
                continue
            structure = structure_metrics(split_masks[projected_row["obj_id"]])
            path = masks_dir / f"{sequence}-frame{frame}-camera{camera_number}-obj{obj_id}.png"
            Image.fromarray(split_masks[projected_row["obj_id"]].astype(np.uint8) * 255).save(path)
            emitted[obj_id] = {"path": str(path.resolve()),
                               "sha256": digest(path),
                               "pixels": int(split_masks[projected_row["obj_id"]].sum()),
                               "variant": variant["name"],
                               "structure_metrics": structure,
                               "structural_alerts": structural_alerts(structure)}
        rows.append({"sequence": sequence, "frame_id": frame,
                     "camera": camera_number, "joint_recovered": True,
                     "component_obj_ids": [row["obj_id"] for row in component],
                     "variant_audit": variant_audit,
                     "selected_variants": {str(row["obj_id"]): variant["name"]
                                           for row, variant in zip(component, selected)},
                     "selected_structural_alert_count": selected_alert_count,
                     "split_metrics": split_metrics,
                     "separations": separations, "emitted_masks": emitted})
    report = {
        "schema_version": "labelagent.pallet_joint_lidar_recovery.v1",
        "status": "offline_candidate_diagnostic_not_deployed",
        "split_report_sha256": digest(split_report_path),
        "recovery_reports": [{"path": str(path.resolve()), "sha256": digest(path)}
                             for path in recovery_report_paths],
        "joint_group_count": len(rows),
        "joint_recovered_group_count": sum(row["joint_recovered"] for row in rows),
        "structural_alert_count": sum(
            bool(mask.get("structural_alerts")) for row in rows
            for mask in row.get("emitted_masks", {}).values()),
        "rows": rows, "human_truth_writes": 0, "training_data_writes": 0,
        "workbench_writes": 0, "production_deployed": False,
    }
    path = output / "results.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    report["report_path"] = str(path.resolve()); report["report_sha256"] = digest(path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-report", required=True, type=Path)
    parser.add_argument("--recovery-report", required=True, type=Path,
                        action="append")
    parser.add_argument("--labels-root", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = evaluate(args.split_report, args.recovery_report, args.labels_root,
                      args.calibration, args.output)
    print(json.dumps({key: report[key] for key in (
        "joint_group_count", "joint_recovered_group_count", "workbench_writes",
        "report_path", "report_sha256")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

