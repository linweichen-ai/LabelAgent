#!/usr/bin/env python3
"""Audit same-image prompt-policy mask stability for three frozen pallet targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


SCHEMA = "labelagent.pallet_prompt_mask_stability.v1"
EXPECTED_ROUND714_SHA = "51af15865358c0d7d1c6919cf246539902f5cbf3c2fc17bcfc764a82bafb3bf4"
TARGETS = {
    ("006", "1789899601982497364", 1, "12"),
    ("006", "1789899601982497364", 1, "13"),
    ("008", "1789899778882397075", 1, "78"),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json_bound(path: Path) -> tuple[dict[str, Any], str]:
    require(path.is_file(), f"missing JSON input: {path}")
    before = sha256(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if sha256(path) != before:
        raise RuntimeError(f"input changed while reading: {path.name}")
    require(isinstance(value, dict), f"expected JSON object: {path.name}")
    return value, before


def bind(path: Path, expected: str, role: str,
         receipts: list[dict[str, Any]], frozen: dict[Path, str]) -> None:
    require(path.is_file(), f"missing {role}: {path}")
    observed = sha256(path)
    require(observed == expected, f"{role} hash mismatch: {path.name}")
    receipts.append({"role": role, "name": path.name,
                     "bytes": path.stat().st_size, "sha256": observed})
    frozen[path.resolve()] = observed


def verify_frozen(frozen: dict[Path, str]) -> None:
    changed = [str(path) for path, expected in frozen.items()
               if not path.is_file() or sha256(path) != expected]
    if changed:
        raise RuntimeError(f"frozen stability-audit input changed: {changed}")


def resolve_under(root: Path, relative: str) -> Path:
    root = root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"path escapes frozen root: {relative}") from error
    return candidate


def locate_mask(name: str, roots: list[Path]) -> Path:
    matches: list[Path] = []
    for root in roots:
        root = root.resolve()
        require(root.is_dir(), f"mask root is not a directory: {root}")
        candidate = (root / name).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ValueError(f"mask path escapes root: {name}") from error
        if candidate.is_file():
            matches.append(candidate)
    require(len(matches) == 1, f"mask lookup must resolve exactly once: {name}")
    return matches[0]


def validate_boundaries(results: dict[str, Any], frozen_manifest: dict[str, Any]) -> None:
    require(results.get("status") == "offline_temporal_local_point_sam_candidate_not_deployed",
            "Round714 status drift")
    require(results.get("candidate_only") is True
            and results.get("human_pixel_truth") is False
            and results.get("approved_for_training") is False
            and results.get("production_deployed") is False
            and results.get("human_truth_writes") == 0
            and results.get("training_data_writes") == 0
            and results.get("workbench_writes") == 0
            and results.get("oss_writes") == 0,
            "Round714 safety boundary drift")
    require(frozen_manifest.get("evaluation_only") is True
            and frozen_manifest.get("training_return_allowed") is False
            and frozen_manifest.get("candidate_write_allowed") is False
            and frozen_manifest.get("human_truth_write_allowed") is False
            and frozen_manifest.get("workbench_writes") == 0
            and frozen_manifest.get("human_truth_writes") == 0
            and frozen_manifest.get("oss_writes") == 0,
            "Round674 safety boundary drift")


def image_index(manifest: dict[str, Any]) -> dict[tuple[str, str, int], dict[str, Any]]:
    result: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in manifest.get("files", []):
        key = (str(row.get("sequence")), str(row.get("frame_id")),
               int(str(row.get("camera", "camera_-1")).split("_")[-1]))
        require(key not in result, f"duplicate Round674 image identity: {key}")
        result[key] = row
    return result


def iou(left: np.ndarray, right: np.ndarray) -> float:
    union = int(np.count_nonzero(left | right))
    require(union > 0, "cannot calculate IoU for two empty masks")
    return float(np.count_nonzero(left & right) / union)


def save_mask(mask: np.ndarray, path: Path) -> dict[str, Any]:
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path, compress_level=9)
    return {"relative_path": path.name, "bytes": path.stat().st_size,
            "sha256": sha256(path), "pixels": int(mask.sum())}


def save_overlay(source: Image.Image, mask: np.ndarray, path: Path,
                 color: tuple[int, int, int]) -> dict[str, Any]:
    base = np.asarray(source.convert("RGB"), dtype=np.float32).copy()
    selected = mask.astype(bool)
    base[selected] = base[selected] * .52 + np.asarray(color, dtype=np.float32) * .48
    Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB").save(
        path, compress_level=9)
    return {"relative_path": path.name, "bytes": path.stat().st_size,
            "sha256": sha256(path)}


def target_metric_pass(attempt: dict[str, Any], target_obj_id: str) -> bool:
    metrics = [metric for metric in attempt.get("split_metrics", [])
               if str(metric.get("obj_id")) == target_obj_id]
    require(len(metrics) <= 1, "duplicate target split metric")
    return bool(metrics and metrics[0].get("candidate_pass") is True)


def build_audit(results_path: Path, frozen_manifest_path: Path,
                mask_roots: list[Path], output_dir: Path) -> dict[str, Any]:
    results, results_sha = read_json_bound(results_path)
    frozen_manifest, frozen_sha = read_json_bound(frozen_manifest_path)
    require(results_sha == EXPECTED_ROUND714_SHA, "Round714 results identity mismatch")
    validate_boundaries(results, frozen_manifest)
    require(not output_dir.exists(), "output directory already exists")

    rows: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for row in results.get("rows", []):
        key = (str(row.get("sequence")), str(row.get("frame_id")),
               int(row.get("camera", -1)), str(row.get("target_obj_id")))
        require(key in TARGETS, f"unexpected Round714 target: {key}")
        require(key not in rows, f"duplicate Round714 target: {key}")
        rows[key] = row
    require(set(rows) == TARGETS, "Round714 target identity set drift")

    images = image_index(frozen_manifest)
    receipts: list[dict[str, Any]] = []
    frozen: dict[Path, str] = {}
    bind(results_path, results_sha, "round714_results", receipts, frozen)
    bind(frozen_manifest_path, frozen_sha, "round674_manifest", receipts, frozen)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output_dir.name + ".tmp-", dir=output_dir.parent))
    try:
        output_rows: list[dict[str, Any]] = []
        for key in sorted(TARGETS, key=lambda value: (value[0], int(value[3]))):
            sequence, frame_id, camera, obj_id = key
            image_row = images.get(key[:3])
            require(image_row is not None, f"Round674 image identity missing: {key[:3]}")
            image_path = resolve_under(frozen_manifest_path.parent,
                                       str(image_row["relative_path"]))
            bind(image_path, str(image_row["sha256"]),
                 f"source_image_{sequence}_obj{obj_id}", receipts, frozen)
            with Image.open(image_path) as opened:
                source = opened.convert("RGB")
            require(source.size == (640, 512), "source image resolution drift")

            included: list[dict[str, Any]] = []
            seen_policies: set[str] = set()
            for attempt in rows[key].get("attempts", []):
                policy = str(attempt.get("policy", ""))
                require(policy and policy not in seen_policies, "empty or duplicate prompt policy")
                seen_policies.add(policy)
                if not attempt.get("mask_path") or not attempt.get("mask_sha256"):
                    continue
                if not target_metric_pass(attempt, obj_id):
                    continue
                mask_path = locate_mask(Path(str(attempt["mask_path"])).name, mask_roots)
                bind(mask_path, str(attempt["mask_sha256"]),
                     f"mask_{sequence}_obj{obj_id}_{policy}", receipts, frozen)
                with Image.open(mask_path) as opened:
                    array = np.asarray(opened.convert("L"), dtype=np.uint8) > 0
                require(array.shape == (512, 640), "mask resolution drift")
                require(bool(array.any()), f"empty included mask: {sequence}/{obj_id}/{policy}")
                included.append({"policy": policy, "mask_name": mask_path.name,
                                 "mask_sha256": attempt["mask_sha256"], "mask": array})
            require(included, f"no successful masks for target {sequence}/{obj_id}")
            included.sort(key=lambda value: value["policy"])
            n = len(included)
            matrix = [[iou(left["mask"], right["mask"]) for right in included]
                      for left in included]
            pair_values = [matrix[i][j] for i in range(n) for j in range(i + 1, n)]
            require(pair_values, f"need at least two successful policies for {sequence}/{obj_id}")
            mean_by_policy = [sum(matrix[index][other] for other in range(n) if other != index)
                              / (n - 1) for index in range(n)]
            medoid_index = sorted(range(n), key=lambda index: (-mean_by_policy[index],
                                                                included[index]["policy"]))[0]
            votes = np.sum(np.stack([entry["mask"] for entry in included], axis=0), axis=0)
            required50, required75 = math.ceil(n * .50), math.ceil(n * .75)
            consensus50, consensus75 = votes >= required50, votes >= required75
            require(bool(consensus50.any()) and bool(consensus75.any()),
                    f"consensus mask is empty for {sequence}/{obj_id}")
            prefix = f"{sequence}-frame{frame_id}-camera{camera}-obj{obj_id}"
            mask50 = save_mask(consensus50, temporary / f"{prefix}-consensus50.png")
            mask75 = save_mask(consensus75, temporary / f"{prefix}-consensus75.png")
            overlay50 = save_overlay(source, consensus50,
                                     temporary / f"{prefix}-consensus50-overlay.png",
                                     (35, 220, 150))
            overlay75 = save_overlay(source, consensus75,
                                     temporary / f"{prefix}-consensus75-overlay.png",
                                     (255, 205, 35))
            policy_rows = []
            for index, entry in enumerate(included):
                policy_rows.append({
                    "policy": entry["policy"], "mask_name": entry["mask_name"],
                    "mask_sha256": entry["mask_sha256"],
                    "pixels": int(entry["mask"].sum()),
                    "mean_iou_to_other_policies": mean_by_policy[index],
                    "iou_to_consensus50": iou(entry["mask"], consensus50),
                    "iou_to_consensus75": iou(entry["mask"], consensus75),
                })
            output_rows.append({
                "sequence": sequence, "frame_id": frame_id, "camera": camera,
                "target_obj_id": obj_id, "included_policy_count": n,
                "policies": policy_rows,
                "pairwise_policy_order": [entry["policy"] for entry in included],
                "pairwise_iou_matrix": matrix,
                "pairwise_iou_mean": sum(pair_values) / len(pair_values),
                "pairwise_iou_min": min(pair_values),
                "consensus50_required_votes": required50,
                "consensus75_required_votes": required75,
                "consensus50_mask": mask50, "consensus75_mask": mask75,
                "consensus50_overlay": overlay50, "consensus75_overlay": overlay75,
                "assistant_medoid_candidate": {
                    "policy": included[medoid_index]["policy"],
                    "mask_name": included[medoid_index]["mask_name"],
                    "mask_sha256": included[medoid_index]["mask_sha256"],
                    "mean_iou_to_other_policies": mean_by_policy[medoid_index],
                    "candidate_only": True, "human_pixel_truth": False,
                    "approved_for_training": False,
                },
            })

        report = {
            "schema_version": SCHEMA,
            "status": "offline_same_source_prompt_mask_stability_audit_not_deployed",
            "summary": {"target_count": 3,
                        "included_policy_mask_count": sum(row["included_policy_count"]
                                                          for row in output_rows)},
            "rows": output_rows,
            "source_inputs": sorted(receipts, key=lambda row: (row["role"], row["name"])),
            "evidence_scope": "same_source_same_image_prompt_policy_stability_only",
            "independent_validation": False,
            "medoid_is_assistant_candidate_only": True,
            "candidate_only": True, "human_pixel_truth": False,
            "approved_for_training": False, "training_authorized": False,
            "automatic_approval": False, "human_truth_writes": 0,
            "training_data_writes": 0, "workbench_writes": 0, "oss_writes": 0,
            "production_deployed": False, "readback_verified": True,
        }
        manifest = temporary / "manifest.json"
        manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        require(json.loads(manifest.read_text(encoding="utf-8")) == report,
                "output manifest readback mismatch")
        verify_frozen(frozen)
        temporary.replace(output_dir)
        return {"output": str(output_dir), "manifest": str(output_dir / "manifest.json"),
                "manifest_sha256": sha256(output_dir / "manifest.json"),
                **report["summary"]}
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round714-results", type=Path, required=True)
    parser.add_argument("--round674-manifest", type=Path, required=True)
    parser.add_argument("--mask-root", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_audit(args.round714_results, args.round674_manifest,
                                 args.mask_root, args.output_dir), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
