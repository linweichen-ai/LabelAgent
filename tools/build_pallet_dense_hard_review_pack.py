#!/usr/bin/env python3
"""Build a local, candidate-only review pack for the frozen 006/008 hard frames.

The tool never writes LabelAgent state, truth, training data, model weights, or OSS.
It renders already-frozen masks for human review and deliberately treats every
pixel outside a reviewed instance as ignore.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


SCHEMA = "labelagent.pallet_dense_hard_review_pack.v1"
FRAME_SPECS = {
    "006": {
        "frame_id": "1789899601982497364",
        "camera": 1,
        "primary_676": {"11", "15", "16", "23"},
        "primary_714": {
            "12": "three_positive_all_negative",
            "13": "three_positive_all_negative",
        },
        "warning_676": {"14"},
        "redraw_676": {"10"},
        "expected_component": {"10", "11", "12", "13", "14", "15", "16", "23"},
        "expected_blocker": "10",
    },
    "008": {
        "frame_id": "1789899778882397075",
        "camera": 1,
        "primary_676": {"67", "69"},
        "primary_714": {"78": "three_positive_two_negative"},
        "warning_676": set(),
        "redraw_676": {"75", "76", "77", "79"},
        "expected_component": {"67", "69", "75", "76", "77", "78", "79"},
        "expected_blocker": "75",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_bound_json(path: Path) -> tuple[dict[str, Any], str]:
    before = sha256(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if sha256(path) != before:
        raise RuntimeError(f"input changed while reading: {path.name}")
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return value, before


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def resolve_under(root: Path, relative: str) -> Path:
    require(bool(relative), "relative path is empty")
    root = root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"path escapes frozen root: {relative}") from error
    return candidate


def bind_file(path: Path, inputs: dict[str, dict[str, Any]],
              *, expected_sha: str | None = None, role: str) -> str:
    require(path.is_file(), f"missing {role}: {path}")
    observed = sha256(path)
    if expected_sha is not None:
        require(observed == expected_sha, f"{role} hash mismatch: {path.name}")
    key = f"{role}:{path.name}"
    if key in inputs and inputs[key]["sha256"] != observed:
        raise ValueError(f"duplicate bound identity disagrees: {key}")
    inputs[key] = {"name": path.name, "bytes": path.stat().st_size,
                   "sha256": observed, "role": role}
    return observed


def verify_inputs(paths: dict[Path, str]) -> None:
    changed = [str(path) for path, expected in paths.items()
               if not path.is_file() or sha256(path) != expected]
    if changed:
        raise RuntimeError(f"frozen review-pack input changed: {changed}")


def validate_boundaries(manifest: dict[str, Any], overlap: dict[str, Any],
                        corrected: dict[str, Any], joint: dict[str, Any] | None) -> None:
    require(manifest.get("evaluation_only") is True, "Round674 must be evaluation-only")
    require(manifest.get("training_return_allowed") is False,
            "Round674 must forbid training return")
    require(manifest.get("candidate_write_allowed") is False,
            "Round674 must forbid candidate writes")
    require(manifest.get("human_truth_write_allowed") is False,
            "Round674 must forbid human-truth writes")
    require(overlap.get("status") == "offline_instance_diagnostic_not_deployed",
            "Round676 status is not frozen diagnostic")
    require(overlap.get("human_pixel_truth") is False
            and overlap.get("approved_for_training") is False
            and overlap.get("production_deployed") is False
            and overlap.get("workbench_writes") == 0,
            "Round676 boundary is unsafe")
    require(corrected.get("status") == "offline_temporal_local_point_sam_candidate_not_deployed",
            "Round714 status is not frozen candidate")
    require(corrected.get("candidate_only") is True
            and corrected.get("human_pixel_truth") is False
            and corrected.get("approved_for_training") is False
            and corrected.get("production_deployed") is False
            and corrected.get("human_truth_writes") == 0
            and corrected.get("training_data_writes") == 0
            and corrected.get("workbench_writes") == 0
            and corrected.get("oss_writes") == 0,
            "Round714 boundary is unsafe")
    require(corrected.get("recovered_candidate_count") == 0,
            "Round714 unexpectedly claims a recovered group")
    if joint is not None:
        require(joint.get("status") == "offline_joint_conflict_candidate_diagnostic_not_deployed",
                "Round706 status is not frozen diagnostic")
        require(joint.get("candidate_only") is True
                and joint.get("human_pixel_truth") is False
                and joint.get("approved_for_training") is False
                and joint.get("production_deployed") is False
                and joint.get("human_truth_writes") == 0
                and joint.get("training_data_writes") == 0
                and joint.get("workbench_writes") == 0
                and joint.get("oss_writes") == 0,
                "Round706 boundary is unsafe")


def row_index(report: dict[str, Any]) -> dict[tuple[str, str, int], dict[str, Any]]:
    result: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in report.get("rows", []):
        key = (str(row.get("sequence")), str(row.get("frame_id")), int(row.get("camera", -1)))
        if key in result:
            raise ValueError(f"duplicate report row: {key}")
        result[key] = row
    expected = {(sequence, spec["frame_id"], spec["camera"])
                for sequence, spec in FRAME_SPECS.items()}
    require(expected.issubset(result), "Round676 does not contain both hard frames")
    return result


def corrected_row_index(report: dict[str, Any]) -> dict[tuple[str, str, int, str], dict[str, Any]]:
    """Index the target-specific Round714 rows without collapsing targets in one frame."""
    result: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    expected_frames = {(sequence, spec["frame_id"], spec["camera"])
                       for sequence, spec in FRAME_SPECS.items()}
    expected_targets = {(sequence, spec["frame_id"], spec["camera"], obj_id)
                        for sequence, spec in FRAME_SPECS.items()
                        for obj_id in spec["primary_714"]}
    for row in report.get("rows", []):
        frame_key = (str(row.get("sequence")), str(row.get("frame_id")),
                     int(row.get("camera", -1)))
        require(frame_key in expected_frames, "Round714 contains an unexpected frame")
        key = frame_key + (str(row.get("target_obj_id", "")),)
        if key in result:
            raise ValueError(f"duplicate Round714 target row: {key}")
        result[key] = row
    require(set(result) == expected_targets, "Round714 target identity set drift")
    return result


def manifest_file_index(manifest: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in manifest.get("files", []):
        key = (str(row.get("sequence")), str(row.get("frame_id")), str(row.get("camera")))
        require(key not in result, f"duplicate Round674 file identity: {key}")
        result[key] = row
    return result


def label_pallet_ids(label: dict[str, Any]) -> set[str]:
    result: set[str] = set()
    for obj in label.get("objects", []):
        if obj.get("obj_type") != "Pallet":
            continue
        obj_id = str(obj.get("obj_id", ""))
        require(obj_id and obj_id not in result, "label has missing or duplicate pallet obj_id")
        result.add(obj_id)
    return result


def mask_pixels(mask: Image.Image) -> int:
    histogram = mask.convert("L").histogram()
    return int(sum(histogram[1:]))


def crop_box(bbox: list[float], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    require(isinstance(bbox, list) and len(bbox) == 4, "projected bbox is invalid")
    x1, y1, x2, y2 = (float(value) for value in bbox)
    require(x2 > x1 and y2 > y1, "projected bbox is empty")
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(176.0, (x2 - x1) * 5.0, (y2 - y1) * 5.0)
    side = min(side, float(max(image_size)))
    left = max(0, min(image_size[0] - side, cx - side / 2))
    top = max(0, min(image_size[1] - side, cy - side / 2))
    return (int(round(left)), int(round(top)),
            int(round(left + side)), int(round(top + side)))


def fit_square(image: Image.Image, *, nearest: bool = False, size: int = 320) -> Image.Image:
    resampling = Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
    copy = image.convert("RGB")
    copy.thumbnail((size, size), resampling)
    canvas = Image.new("RGB", (size, size), (18, 27, 34))
    canvas.paste(copy, ((size - copy.width) // 2, (size - copy.height) // 2))
    return canvas


def overlay(image: Image.Image, mask: Image.Image, status: str) -> Image.Image:
    colors = {
        "primary_review_candidate": (30, 205, 145, 100),
        "localization_warning_review_candidate": (245, 166, 35, 110),
        "manual_redraw_required": (235, 70, 70, 110),
    }
    base = image.convert("RGBA")
    layer = Image.new("RGBA", base.size, colors[status][:-1] + (0,))
    alpha = mask.convert("L").point(lambda value: colors[status][3] if value else 0)
    layer.putalpha(alpha)
    return Image.alpha_composite(base, layer).convert("RGB")


def draw_roi(image: Image.Image, bbox: list[float], status: str) -> Image.Image:
    result = image.convert("RGB").copy()
    colors = {
        "primary_review_candidate": (20, 180, 120),
        "localization_warning_review_candidate": (230, 145, 20),
        "manual_redraw_required": (220, 55, 55),
    }
    draw = ImageDraw.Draw(result)
    draw.rectangle(tuple(float(value) for value in bbox), outline=colors[status], width=2)
    return result


def save_jpeg(image: Image.Image, path: Path) -> str:
    image.save(path, format="JPEG", quality=94, subsampling=0, optimize=False)
    return sha256(path)


def render_card(image: Image.Image, mask: Image.Image, bbox: list[float],
                *, sequence: str, obj_id: str, status: str, source: str,
                policy: str | None, pixels: int, output: Path) -> str:
    box = crop_box(bbox, image.size)
    raw_crop = image.crop(box)
    roi_crop = draw_roi(image, bbox, status).crop(box)
    mask_crop = mask.crop(box)
    overlay_crop = overlay(image, mask, status).crop(box)
    card = Image.new("RGB", (1280, 372), "white")
    panels = [fit_square(raw_crop), fit_square(roi_crop),
              fit_square(Image.merge("RGB", (mask_crop, mask_crop, mask_crop)), nearest=True),
              fit_square(overlay_crop)]
    for index, panel in enumerate(panels):
        card.paste(panel, (index * 320, 52))
    draw = ImageDraw.Draw(card)
    font = ImageFont.load_default()
    title = f"{sequence} obj{obj_id} | {status} | source={source} | pixels={pixels}"
    if policy:
        title += f" | policy={policy}"
    draw.text((8, 8), title, fill="black", font=font)
    for index, text in enumerate(("RAW", "PROJECTED ROI", "MASK ONLY", "OVERLAY - REVIEW ONLY")):
        draw.text((8 + index * 320, 30), text, fill="black", font=font)
    return save_jpeg(card, output)


def render_overview(image: Image.Image, items: list[dict[str, Any]],
                    out_of_scope: list[str], *, sequence: str, output: Path) -> str:
    canvas = Image.new("RGB", (980, 512), (245, 247, 248))
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    colors = {
        "primary_review_candidate": (20, 180, 120),
        "localization_warning_review_candidate": (230, 145, 20),
        "manual_redraw_required": (220, 55, 55),
    }
    for item in items:
        bbox = item["projected_bbox"]
        color = colors[item["review_status"]]
        draw.rectangle(tuple(float(value) for value in bbox), outline=color, width=2)
        draw.text((int(bbox[0]), max(0, int(bbox[1]) - 11)), f"o{item['obj_id']}", fill=color,
                  font=ImageFont.load_default())
    canvas.paste(annotated, (0, 0))
    side = ImageDraw.Draw(canvas)
    side.text((650, 12), f"Sequence {sequence} | candidate-only", fill="black")
    side.text((650, 32), "green=primary  amber=warning  red=redraw", fill="black")
    y = 58
    for status in ("primary_review_candidate", "localization_warning_review_candidate",
                   "manual_redraw_required"):
        ids = [item["obj_id"] for item in items if item["review_status"] == status]
        side.text((650, y), f"{status}:", fill="black")
        side.text((650, y + 16), ", ".join(ids) or "none", fill="black")
        y += 44
    side.text((650, y), f"out-of-scope pallet ids ({len(out_of_scope)}):", fill="black")
    y += 16
    chunks = [out_of_scope[index:index + 7] for index in range(0, len(out_of_scope), 7)]
    for chunk in chunks:
        side.text((650, y), ", ".join(chunk), fill=(80, 80, 80))
        y += 15
    side.text((650, y + 8), "ALL out-of-scope and outside-mask pixels = IGNORE", fill=(150, 40, 40))
    side.text((650, y + 26), "Not human truth. No automatic approval.", fill=(150, 40, 40))
    return save_jpeg(canvas, output)


def build_pack(manifest_path: Path, round676_path: Path, round714_path: Path,
               output_dir: Path, round706_path: Path | None = None) -> dict[str, Any]:
    manifest, manifest_sha = read_bound_json(manifest_path)
    overlap, overlap_sha = read_bound_json(round676_path)
    corrected, corrected_sha = read_bound_json(round714_path)
    joint = joint_sha = None
    if round706_path is not None:
        joint, joint_sha = read_bound_json(round706_path)
    validate_boundaries(manifest, overlap, corrected, joint)
    require(not output_dir.exists(), "output directory already exists")

    frozen_paths = {manifest_path.resolve(): manifest_sha, round676_path.resolve(): overlap_sha,
                    round714_path.resolve(): corrected_sha}
    if round706_path is not None and joint_sha is not None:
        frozen_paths[round706_path.resolve()] = joint_sha
    input_receipts: dict[str, dict[str, Any]] = {}
    bind_file(manifest_path, input_receipts, expected_sha=manifest_sha, role="round674_manifest")
    bind_file(round676_path, input_receipts, expected_sha=overlap_sha, role="round676_results")
    bind_file(round714_path, input_receipts, expected_sha=corrected_sha, role="round714_results")
    if round706_path is not None and joint_sha is not None:
        bind_file(round706_path, input_receipts, expected_sha=joint_sha, role="round706_explanation_only")

    overlap_rows = row_index(overlap)
    corrected_rows = corrected_row_index(corrected)
    manifest_files = manifest_file_index(manifest)
    expected_manifest_keys = {(seq, spec["frame_id"], f"camera_{spec['camera']}")
                              for seq, spec in FRAME_SPECS.items()}
    require(expected_manifest_keys.issubset(manifest_files), "Round674 manifest lacks a hard frame")

    if joint is not None:
        joint_rows = {str(row.get("sequence")): row for row in joint.get("rows", [])}
        for sequence, spec in FRAME_SPECS.items():
            row = joint_rows.get(sequence)
            require(row is not None, f"Round706 lacks sequence {sequence}")
            require(set(map(str, row.get("component_obj_ids", []))) == spec["expected_component"],
                    f"Round706 component drift for {sequence}")
            best = row.get("best_failed_combination") or {}
            require(set(map(str, best.get("failed_member_obj_ids", []))) == {spec["expected_blocker"]},
                    f"Round706 blocker drift for {sequence}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output_dir.name + ".tmp-", dir=output_dir.parent))
    try:
        (temporary / "cards").mkdir()
        frame_receipts: list[dict[str, Any]] = []
        item_receipts: list[dict[str, Any]] = []
        for sequence, spec in FRAME_SPECS.items():
            key = (sequence, spec["frame_id"], spec["camera"])
            row = overlap_rows[key]
            instances = {str(instance["obj_id"]): instance for instance in row.get("instances", [])}
            metrics = {str(metric["obj_id"]): metric for metric in row.get("depth_split_metrics", [])}
            require(set(instances) == spec["expected_component"], f"Round676 component drift for {sequence}")
            require(set(metrics) == spec["expected_component"], f"Round676 metric drift for {sequence}")

            mf = manifest_files[(sequence, spec["frame_id"], f"camera_{spec['camera']}")]
            image_path = resolve_under(manifest_path.parent, str(mf["relative_path"]))
            image_sha = bind_file(image_path, input_receipts, expected_sha=str(mf["sha256"]),
                                  role=f"source_image_{sequence}")
            frozen_paths[image_path] = image_sha
            label_path = resolve_under(manifest_path.parent,
                                       f"labels/{sequence}/{spec['frame_id']}.json")
            label, label_sha = read_bound_json(label_path)
            bind_file(label_path, input_receipts, expected_sha=label_sha, role=f"source_label_{sequence}")
            frozen_paths[label_path] = label_sha
            pallet_ids = label_pallet_ids(label)
            require(len(pallet_ids) == 20, f"expected 20 pallet labels for {sequence}")
            require(spec["expected_component"].issubset(pallet_ids),
                    f"review component is not present in label for {sequence}")
            out_of_scope = sorted(pallet_ids - spec["expected_component"], key=int)
            require(len(out_of_scope) == (12 if sequence == "006" else 13),
                    f"out-of-scope pallet count drift for {sequence}")

            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            frame_items = []
            for obj_id in sorted(spec["expected_component"], key=int):
                instance = instances[obj_id]
                metric = metrics[obj_id]
                source = policy = None
                if obj_id in spec["primary_714"]:
                    review_status = "primary_review_candidate"
                    source = "round714_corrected_temporal_target_local"
                    policy = spec["primary_714"][obj_id]
                    temporal_row = corrected_rows[(sequence, spec["frame_id"], spec["camera"], obj_id)]
                    attempts = [attempt for attempt in temporal_row.get("attempts", [])
                                if attempt.get("policy") == policy]
                    require(len(attempts) == 1 and attempts[0].get("mask_path"),
                            f"Round714 policy mismatch for {sequence}/{obj_id}")
                    attempt = attempts[0]
                    require(float(attempt.get("positive_coverage_all_temporal_points", -1)) >= .8
                            and float(attempt.get("negative_clear_all_center_conflict_points", -1)) >= .8,
                            f"Round714 point gate failed for {sequence}/{obj_id}")
                    target_metric = [entry for entry in attempt.get("split_metrics", [])
                                     if str(entry.get("obj_id")) == obj_id]
                    require(len(target_metric) == 1 and target_metric[0].get("candidate_pass") is True,
                            f"Round714 target split gate failed for {sequence}/{obj_id}")
                    require(attempt.get("accepted") is False,
                            "Round714 candidate must remain group-blocked")
                    receipt = {"path": attempt["mask_path"], "sha256": attempt["mask_sha256"]}
                    blocker = "target-local pass; complete conflict group did not pass"
                else:
                    receipt = row.get("depth_split_masks", {}).get(obj_id)
                    require(isinstance(receipt, dict), f"Round676 mask receipt missing for {sequence}/{obj_id}")
                    if obj_id in spec["primary_676"]:
                        review_status = "primary_review_candidate"
                        source = "round676_depth_split"
                        require(instance.get("localization_pass") is True
                                and metric.get("candidate_pass") is True
                                and metric.get("handled_candidate") is True,
                                f"Round676 primary gate failed for {sequence}/{obj_id}")
                        blocker = "individual candidate only; complete conflict group did not pass"
                    elif obj_id in spec["warning_676"]:
                        review_status = "localization_warning_review_candidate"
                        source = "round676_depth_split"
                        require(instance.get("localization_pass") is False
                                and metric.get("candidate_pass") is True,
                                f"Round676 warning gate drift for {sequence}/{obj_id}")
                        blocker = "depth split passed but source localization failed"
                    else:
                        review_status = "manual_redraw_required"
                        source = "round676_failed_depth_split_evidence"
                        require(obj_id in spec["redraw_676"] and metric.get("candidate_pass") is False,
                                f"Round676 redraw gate drift for {sequence}/{obj_id}")
                        blocker = "no corrected safe candidate; failed mask is evidence only"
                mask_path = round676_path.parent / "masks" / Path(str(receipt["path"])).name
                if source.startswith("round714"):
                    mask_path = round714_path.parent / "masks" / Path(str(receipt["path"])).name
                mask_sha = bind_file(mask_path, input_receipts,
                                     expected_sha=str(receipt["sha256"]),
                                     role=f"candidate_mask_{sequence}_obj{obj_id}")
                frozen_paths[mask_path.resolve()] = mask_sha
                with Image.open(mask_path) as opened:
                    mask = opened.convert("L")
                require(mask.size == image.size, f"mask/image size mismatch for {sequence}/{obj_id}")
                pixels = mask_pixels(mask)
                card_rel = f"cards/{sequence}-frame{spec['frame_id']}-camera1-obj{obj_id}.jpg"
                card_path = temporary / card_rel
                card_sha = render_card(image, mask, instance["projected_bbox"],
                                       sequence=sequence, obj_id=obj_id, status=review_status,
                                       source=source, policy=policy, pixels=pixels, output=card_path)
                item = {
                    "candidate_id": f"{sequence}:{spec['frame_id']}:camera1:obj{obj_id}",
                    "sequence": sequence, "frame_id": spec["frame_id"], "camera": 1,
                    "obj_id": obj_id, "review_status": review_status,
                    "candidate_source": source, "policy": policy,
                    "source_mask_name": mask_path.name, "source_mask_sha256": mask_sha,
                    "mask_pixels": pixels, "projected_bbox": instance["projected_bbox"],
                    "blocker_or_caveat": blocker, "card": card_rel,
                    "card_sha256": card_sha, "assistant_decision": None,
                    "human_decision": None, "outside_instance": "ignore",
                    "candidate_only": True, "human_pixel_truth": False,
                    "approved_for_training": False,
                }
                frame_items.append(item)
                item_receipts.append(item)

            overview_rel = f"overview_{sequence}.jpg"
            overview_sha = render_overview(image, frame_items, out_of_scope,
                                           sequence=sequence, output=temporary / overview_rel)
            frame_receipts.append({
                "sequence": sequence, "frame_id": spec["frame_id"], "camera": 1,
                "source_image_name": image_path.name, "source_image_sha256": image_sha,
                "source_label_name": label_path.name, "source_label_sha256": label_sha,
                "pallet_label_count": len(pallet_ids),
                "review_scope_obj_ids": sorted(spec["expected_component"], key=int),
                "out_of_scope_obj_ids": out_of_scope,
                "out_of_scope_policy": "ignore",
                "overview": overview_rel, "overview_sha256": overview_sha,
            })

        summary = {
            "frame_count": len(frame_receipts),
            "overview_count": len(frame_receipts),
            "instance_card_count": len(item_receipts),
            "primary_count": sum(item["review_status"] == "primary_review_candidate"
                                 for item in item_receipts),
            "warning_count": sum(item["review_status"] ==
                                 "localization_warning_review_candidate" for item in item_receipts),
            "redraw_count": sum(item["review_status"] == "manual_redraw_required"
                                for item in item_receipts),
            "out_of_scope_pallet_count": sum(len(frame["out_of_scope_obj_ids"])
                                              for frame in frame_receipts),
        }
        require(summary == {"frame_count": 2, "overview_count": 2,
                            "instance_card_count": 15, "primary_count": 9,
                            "warning_count": 1, "redraw_count": 5,
                            "out_of_scope_pallet_count": 25},
                f"review-pack summary drift: {summary}")
        report = {
            "schema_version": SCHEMA,
            "status": "candidate_only_human_review_pack_not_deployed",
            "summary": summary,
            "source_inputs": sorted(input_receipts.values(),
                                    key=lambda item: (item["role"], item["name"])),
            "frames": frame_receipts,
            "items": item_receipts,
            "superseded_sources": [
                {"round": 700, "role": "historical_only_not_primary",
                 "reason": "temporal visibility protocol was corrected after Round700"},
                {"round": 706, "role": "explanation_only_not_primary",
                 "reason": "best combinations depend on superseded Round700 candidates"},
            ],
            "supervision_policy": {
                "outside_reviewed_instance": "ignore",
                "out_of_scope_objects": "ignore",
                "full_frame_semantic_loss_allowed": False,
                "negative_sampling_allowed": False,
            },
            "candidate_only": True,
            "human_pixel_truth": False,
            "approved_for_training": False,
            "training_authorized": False,
            "automatic_approval": False,
            "human_truth_writes": 0,
            "training_data_writes": 0,
            "workbench_writes": 0,
            "oss_writes": 0,
            "production_deployed": False,
            "readback_verified": True,
        }
        manifest_output = temporary / "manifest.json"
        manifest_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8")
        reread = json.loads(manifest_output.read_text(encoding="utf-8"))
        require(reread == report, "review-pack manifest readback mismatch")
        verify_inputs(frozen_paths)
        temporary.replace(output_dir)
        return {"output": str(output_dir), "manifest": str(output_dir / "manifest.json"),
                "manifest_sha256": sha256(output_dir / "manifest.json"), **summary}
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round674-manifest", type=Path, required=True)
    parser.add_argument("--round676-results", type=Path, required=True)
    parser.add_argument("--round714-results", type=Path, required=True)
    parser.add_argument("--round706-results", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_pack(args.round674_manifest, args.round676_results,
                        args.round714_results, args.output_dir, args.round706_results)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
