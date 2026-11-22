#!/usr/bin/env python3
"""Offline joint selection of frozen pallet masks within conflict components.

This diagnostic never invokes SAM.  It combines already-frozen Round 700 raw
temporal candidates with Round 676 incumbent raw masks, then applies the
unchanged depth-split and pair-separation gates to every complete component.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.evaluate_pallet_box3d_overlap_pairs import (
    depth_split_masks, mask_separation, projected_conflict_component,
)
from tools.evaluate_pallet_box3d_roi_sam import (
    digest, load_camera, localization_gate, project_box,
)


REPORT_SCHEMA = "labelagent.pallet_joint_conflict_candidates.v1"
TEMPORAL_SCHEMA = "labelagent.pallet_temporal_local_point_sam.v1"
OVERLAP_SCHEMA = "labelagent.pallet_box3d_overlap_pairs.v1"


def bound_json(path: Path) -> tuple[dict, str]:
    before = digest(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if digest(path) != before:
        raise RuntimeError(f"JSON input changed while being parsed: {path}")
    return payload, before


def bind_file(path: Path, inputs: dict[str, str], expected: str | None = None) -> str:
    if not path.is_file():
        raise ValueError(f"frozen input is missing: {path}")
    value = digest(path)
    if expected is not None and value != expected:
        raise ValueError(f"frozen input hash differs: {path}")
    key = str(path.resolve())
    prior = inputs.setdefault(key, value)
    if prior != value:
        raise ValueError(f"conflicting frozen input identity: {path}")
    return value


def verify_frozen(inputs: dict[str, str]) -> None:
    drifted = [path for path, expected in inputs.items()
               if not Path(path).is_file() or digest(Path(path)) != expected]
    if drifted:
        raise RuntimeError(f"joint candidate input changed: {drifted}")


def read_mask(path: Path, expected: str, inputs: dict[str, str],
              shape: tuple[int, int] | None = None) -> np.ndarray:
    bind_file(path, inputs, expected)
    value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if digest(path) != expected:
        raise RuntimeError(f"frozen mask changed while being decoded: {path}")
    if value is None or (shape is not None and value.shape != shape):
        raise ValueError(f"frozen mask is unreadable or has wrong dimensions: {path}")
    return value > 0


def validate_reports(temporal: dict, overlap: dict) -> None:
    if (temporal.get("schema_version") != TEMPORAL_SCHEMA
            or temporal.get("status") !=
            "offline_temporal_local_point_sam_candidate_not_deployed"
            or temporal.get("candidate_only") is not True
            or temporal.get("human_pixel_truth") is not False
            or temporal.get("approved_for_training") is not False
            or temporal.get("production_deployed") is not False
            or temporal.get("human_truth_writes") != 0
            or temporal.get("training_data_writes") != 0
            or temporal.get("workbench_writes") != 0
            or temporal.get("oss_writes") != 0
            or not isinstance(temporal.get("input_sha256"), dict)):
        raise ValueError("requires a frozen candidate-only temporal SAM report")
    if (overlap.get("schema_version") != OVERLAP_SCHEMA
            or overlap.get("status") != "offline_instance_diagnostic_not_deployed"
            or overlap.get("human_pixel_truth") is not False
            or overlap.get("approved_for_training") is not False
            or overlap.get("production_deployed") is not False
            or overlap.get("workbench_writes") != 0
            or not isinstance(overlap.get("input_sha256"), dict)):
        raise ValueError("requires a frozen non-deployed overlap report")


def index_temporal_rows(rows: list[dict]) -> dict[tuple[str, str, int, str], dict]:
    result = {}
    for row in rows:
        key = (str(row["sequence"]), str(row["frame_id"]), int(row["camera"]),
               str(row["target_obj_id"]))
        if key in result:
            raise ValueError(f"duplicate temporal identity: {key}")
        result[key] = row
    return result


def index_overlap_rows(rows: list[dict]) -> dict[tuple[str, str, int], dict]:
    result = {}
    for row in rows:
        key = (str(row["sequence"]), str(row["frame_id"]), int(row["camera"]))
        if key in result:
            raise ValueError(f"duplicate overlap frame identity: {key}")
        result[key] = row
    return result


def discover_bound_path(input_map: dict[str, str], predicate, description: str,
                        allow_same_hash_duplicates: bool = False
                        ) -> tuple[Path, str, list[tuple[Path, str]]]:
    matches = sorted((Path(path), value) for path, value in input_map.items()
                     if predicate(Path(path)))
    if not matches:
        raise ValueError(f"bound {description} is missing")
    if len(matches) > 1 and not allow_same_hash_duplicates:
        raise ValueError(f"bound {description} is ambiguous")
    if len({value for _, value in matches}) != 1:
        raise ValueError(f"bound {description} copies disagree")
    return matches[0][0], matches[0][1], matches


def intrinsic_candidate_pass(attempt: dict) -> bool:
    try:
        positive = float(attempt["positive_coverage_all_temporal_points"])
        negative = float(attempt["negative_clear_all_center_conflict_points"])
        metrics = attempt["metrics"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("temporal attempt lacks frozen intrinsic metrics") from error
    return bool(np.isfinite(positive) and np.isfinite(negative)
                and localization_gate(metrics) and positive >= .8 and negative >= .8)


def temporal_options(row: dict, inputs: dict[str, str], shape: tuple[int, int]
                     ) -> tuple[list[dict], dict]:
    policies, accepted, rejected = set(), [], 0
    for attempt in row.get("attempts", []):
        policy = str(attempt.get("policy", ""))
        if not policy or policy in policies:
            raise ValueError("temporal attempt policy is missing or duplicated")
        policies.add(policy)
        if "error" in attempt:
            rejected += 1
            continue
        path = Path(attempt.get("raw_mask_path", ""))
        sha = str(attempt.get("raw_mask_sha256", ""))
        if len(sha) != 64:
            raise ValueError("temporal raw mask receipt is incomplete")
        mask = read_mask(path, sha, inputs, shape)
        if not intrinsic_candidate_pass(attempt):
            rejected += 1
            continue
        accepted.append({
            "source": "round700_temporal_raw", "policy": policy,
            "option_id": f"temporal:{policy}:{sha}", "path": str(path.resolve()),
            "sha256": sha, "mask": mask,
        })
    # Equal masks do not create distinct joint combinations.  Preserve the
    # lexicographically first policy as the deterministic representative.
    accepted.sort(key=lambda item: item["option_id"])
    unique = {}
    for option in accepted:
        unique.setdefault(option["sha256"], option)
    return list(unique.values()), {
        "attempt_count": len(row.get("attempts", [])),
        "intrinsic_rejected_count": rejected,
        "duplicate_mask_pruned_count": len(accepted) - len(unique),
        "intrinsic_candidate_count": len(unique),
    }


def conflict_components(projected: list[dict]) -> list[list[dict]]:
    remaining = {str(row["obj_id"]): row for row in projected}
    if len(remaining) != len(projected):
        raise ValueError("duplicate projected object identity")
    result = []
    while remaining:
        first_id = sorted(remaining)[0]
        component = projected_conflict_component(remaining[first_id], projected)
        identities = {str(row["obj_id"]) for row in component}
        if not identities or first_id not in identities:
            raise RuntimeError("projected conflict component is invalid")
        result.append(sorted(component, key=lambda row: str(row["obj_id"])))
        for identity in identities:
            remaining.pop(identity, None)
    return sorted(result, key=lambda rows: tuple(str(row["obj_id"]) for row in rows))


def evaluate_combinations(component: list[dict], options: dict[str, list[dict]]) -> dict:
    identities = [str(row["obj_id"]) for row in component]
    raw_id_by_identity = {str(row["obj_id"]): row["obj_id"] for row in component}
    if len(raw_id_by_identity) != len(component):
        raise ValueError("component identities collide after string normalization")
    if set(options) != set(identities) or any(not options[obj_id] for obj_id in identities):
        raise ValueError("every conflict member requires at least one mask option")
    ordered_options = [sorted(options[obj_id], key=lambda item: item["option_id"])
                       for obj_id in identities]
    total = int(np.prod([len(rows) for rows in ordered_options], dtype=np.int64))
    evaluated, passing = 0, []
    incumbent_pass = False
    member_failure_histogram: dict[str, int] = {}
    separation_failure_histogram: dict[str, int] = {}
    best_failed = None
    for choice in itertools.product(*ordered_options):
        evaluated += 1
        masks = {raw_id_by_identity[obj_id]: option["mask"]
                 for obj_id, option in zip(identities, choice)}
        split_masks, split_metrics = depth_split_masks(component, masks)
        separations = [{
            "first_obj_id": first, "second_obj_id": second,
            **mask_separation(split_masks[raw_id_by_identity[first]],
                              split_masks[raw_id_by_identity[second]]),
        } for first, second in itertools.combinations(identities, 2)]
        passed = bool(split_metrics
                      and all(row["candidate_pass"] for row in split_metrics)
                      and separations
                      and all(row["separated_at_0_15"] for row in separations))
        failed_members = sorted(str(row["obj_id"]) for row in split_metrics
                                if not row["candidate_pass"])
        failed_pairs = sorted(
            f"{row['first_obj_id']}:{row['second_obj_id']}"
            for row in separations if not row["separated_at_0_15"])
        for identity in failed_members:
            member_failure_histogram[identity] = (
                member_failure_histogram.get(identity, 0) + 1)
        for identity in failed_pairs:
            separation_failure_histogram[identity] = (
                separation_failure_histogram.get(identity, 0) + 1)
        temporal_count = sum(option["source"] == "round700_temporal_raw"
                             for option in choice)
        incumbent_only = temporal_count == 0
        incumbent_pass = incumbent_pass or (incumbent_only and passed)
        if passed:
            passing.append({
                "temporal_choice_count": temporal_count,
                "choice_ids": [option["option_id"] for option in choice],
                "choices": [{key: option[key] for key in
                             ("source", "policy", "option_id", "path", "sha256")}
                            for option in choice],
                "split_metrics": split_metrics, "separations": separations,
            })
        else:
            diagnostic = {
                "passed_member_count": len(split_metrics) - len(failed_members),
                "separated_pair_count": len(separations) - len(failed_pairs),
                "temporal_choice_count": temporal_count,
                "choice_ids": [option["option_id"] for option in choice],
                "failed_member_obj_ids": failed_members,
                "failed_separation_pairs": failed_pairs,
                "split_metrics": split_metrics,
                "separations": separations,
            }
            rank = (-diagnostic["passed_member_count"],
                    -diagnostic["separated_pair_count"],
                    -diagnostic["temporal_choice_count"],
                    diagnostic["choice_ids"])
            if best_failed is None or rank < best_failed[0]:
                best_failed = (rank, diagnostic)
    passing.sort(key=lambda row: (-row["temporal_choice_count"], row["choice_ids"]))
    selected = passing[0] if passing else None
    return {
        "combination_count": total, "evaluated_combination_count": evaluated,
        "search_pruned_combination_count": 0,
        "passing_combination_count": len(passing),
        "incumbent_combination_pass": incumbent_pass,
        "joint_temporal_recovered": bool(
            selected and selected["temporal_choice_count"] > 0),
        "selected": selected,
        "member_failure_histogram": dict(sorted(member_failure_histogram.items())),
        "separation_failure_histogram": dict(
            sorted(separation_failure_histogram.items())),
        "best_failed_combination": best_failed[1] if best_failed else None,
    }


def evaluate(temporal_report_path: Path, overlap_report_path: Path,
             output_path: Path) -> dict:
    if output_path.exists():
        raise ValueError("output already exists")
    temporal, temporal_sha = bound_json(temporal_report_path)
    overlap, overlap_sha = bound_json(overlap_report_path)
    validate_reports(temporal, overlap)
    temporal_inputs = temporal["input_sha256"]
    if temporal_inputs.get(str(overlap_report_path.resolve())) != overlap_sha:
        raise ValueError("temporal report is not bound to this overlap path and hash")
    inputs = {
        str(temporal_report_path.resolve()): temporal_sha,
        str(overlap_report_path.resolve()): overlap_sha,
    }
    bind_file(Path(__file__), inputs)
    temporal_rows = index_temporal_rows(temporal.get("rows", []))
    overlap_rows = index_overlap_rows(overlap.get("rows", []))
    frames = sorted({key[:3] for key in temporal_rows})
    rows = []
    for sequence, frame_id, camera_number in frames:
        overlap_row = overlap_rows.get((sequence, frame_id, camera_number))
        if overlap_row is None:
            raise ValueError("temporal frame has no overlap component")
        label_path, label_sha, label_copies = discover_bound_path(
            temporal_inputs,
            lambda path: (path.name == f"{frame_id}.json"
                          and path.parent.name == "labels"
                          and path.parent.parent.name == sequence),
            "center label", allow_same_hash_duplicates=True)
        for path, sha in label_copies:
            bind_file(path, inputs, sha)
        label, parsed_label_sha = bound_json(label_path)
        if parsed_label_sha != label_sha or str(label.get("frame_id")) != frame_id:
            raise ValueError("center label identity differs")
        camera_path, camera_sha, _ = discover_bound_path(
            temporal_inputs,
            lambda path: path.name == f"camera{camera_number}_calib.yaml",
            "camera calibration")
        lidar_path = camera_path.parent / "lidar2camera.yaml"
        lidar_sha = temporal_inputs.get(str(lidar_path.resolve()))
        if lidar_sha is None:
            raise ValueError("bound lidar calibration is missing")
        bind_file(camera_path, inputs, camera_sha)
        bind_file(lidar_path, inputs, lidar_sha)
        camera = load_camera(camera_path.parent, camera_number)
        if digest(camera_path) != camera_sha or digest(lidar_path) != lidar_sha:
            raise RuntimeError("calibration changed while being parsed")
        objects = label.get("objects")
        if not isinstance(objects, list):
            raise ValueError("center label objects are absent")
        projected = [value for obj in objects
                     if obj.get("obj_type") == "Pallet" and not obj.get("ignore")
                     if (value := project_box(obj, camera)) is not None]
        components = conflict_components(projected)
        serialized_ids = [str(item["obj_id"])
                          for item in overlap_row.get("instances", [])]
        if len(serialized_ids) != len(set(serialized_ids)):
            raise ValueError("overlap component has duplicate member identity")
        matching = [component for component in components
                    if set(serialized_ids) == {str(item["obj_id"])
                                               for item in component}]
        if len(matching) != 1:
            raise ValueError("recomputed projected conflict component differs")
        component = matching[0]
        shape = component[0]["mask"].shape
        instance_receipts = {}
        for instance in overlap_row.get("instances", []):
            obj_id = str(instance.get("obj_id"))
            if obj_id in instance_receipts:
                raise ValueError("overlap component has duplicate raw mask receipt")
            instance_receipts[obj_id] = instance
        temporal_for_frame = {key[3]: value for key, value in temporal_rows.items()
                              if key[:3] == (sequence, frame_id, camera_number)}
        options, pruning = {}, {}
        for member in component:
            obj_id = str(member["obj_id"])
            receipt = instance_receipts.get(obj_id)
            if not isinstance(receipt, dict):
                raise ValueError("incumbent raw mask receipt is missing")
            incumbent_path = Path(receipt.get("mask_path", ""))
            incumbent_sha = str(receipt.get("mask_sha256", ""))
            incumbent = read_mask(incumbent_path, incumbent_sha, inputs, shape)
            options[obj_id] = [{
                "source": "round676_incumbent_raw", "policy": "incumbent",
                "option_id": f"incumbent:{incumbent_sha}",
                "path": str(incumbent_path.resolve()), "sha256": incumbent_sha,
                "mask": incumbent,
            }]
            if obj_id in temporal_for_frame:
                candidates, stats = temporal_options(
                    temporal_for_frame[obj_id], inputs, shape)
                options[obj_id].extend(candidates)
                pruning[obj_id] = stats
            else:
                pruning[obj_id] = {
                    "attempt_count": 0, "intrinsic_rejected_count": 0,
                    "duplicate_mask_pruned_count": 0,
                    "intrinsic_candidate_count": 0,
                }
        result = evaluate_combinations(component, options)
        rows.append({
            "sequence": sequence, "frame_id": frame_id,
            "camera": camera_number,
            "component_obj_ids": [str(item["obj_id"]) for item in component],
            "temporal_target_obj_ids": sorted(temporal_for_frame),
            "candidate_pruning": pruning,
            **result,
        })
    verify_frozen(inputs)
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "offline_joint_conflict_candidate_diagnostic_not_deployed",
        "temporal_report_sha256": temporal_sha,
        "overlap_report_sha256": overlap_sha,
        "tool_sha256": digest(Path(__file__)),
        "input_sha256": inputs,
        "component_count": len(rows),
        "joint_temporal_recovered_component_count": sum(
            row["joint_temporal_recovered"] for row in rows),
        "rows": rows,
        "candidate_only": True,
        "conditioned_on_human_3d_labels": True,
        "same_source_comparison_is_independent_validation": False,
        "human_pixel_truth": False, "approved_for_training": False,
        "human_truth_writes": 0, "training_data_writes": 0,
        "workbench_writes": 0, "oss_writes": 0,
        "production_deployed": False, "readback_verified": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    raw = output_path.read_bytes()
    if json.loads(raw.decode("utf-8")) != report:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("joint candidate report readback differs")
    try:
        verify_frozen(inputs)
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    report["report_path"] = str(output_path.resolve())
    report["report_sha256"] = hashlib.sha256(raw).hexdigest()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal-report", required=True, type=Path)
    parser.add_argument("--overlap-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = evaluate(args.temporal_report, args.overlap_report, args.output)
    keys = ("component_count", "joint_temporal_recovered_component_count",
            "human_truth_writes", "training_data_writes", "workbench_writes",
            "oss_writes", "report_path", "report_sha256")
    print(json.dumps({key: result[key] for key in keys}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
