#!/usr/bin/env python3
"""Build a local, fail-closed review packet from ego component masks.

The input is produced by ``diagnose_ego_guided_points.py``.  A union is only
materialized when the source image, frozen point-set identity, two required
components, prompt constraints and mask hashes all validate.  The packet is
assistant-review material, never human truth or a production candidate.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont


EXPECTED_COMPONENTS = {
    "camera1": (
        "camera1-left-seat-console-v2",
        "camera1-right-hood-v2",
    ),
    "camera3": (
        "camera3-crossbar-core-v2",
        "camera3-lower-assembly-v2",
    ),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_bytes(path: Path, expected_sha256: str, label: str) -> bytes:
    """Read once so hashing and decoding use the same immutable bytes."""
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"{label} is missing or hash-mismatched")
    return data


def _camera_from_sample(sample_id: str) -> str:
    match = re.search(r"-(camera[13])-", sample_id)
    if match is None:
        raise ValueError("sample identity does not contain camera1 or camera3")
    return match.group(1)


def _mask_geometry(mask: np.ndarray) -> dict:
    rows, columns = mask.nonzero()
    if not len(rows):
        raise ValueError("candidate mask is empty")
    return {
        "area_pixels": int(mask.sum()),
        "area_ratio": round(float(mask.mean()), 6),
        "bbox": [int(columns.min()), int(rows.min()),
                 int(columns.max()) + 1, int(rows.max()) + 1],
    }


def load_component_guard(path: Path) -> dict[str, dict[str, float]]:
    """Load and validate fail-closed per-component area-ratio bounds."""
    data = json.loads(path.read_text(encoding="utf-8"))
    components = data.get("components") if isinstance(data, dict) else None
    if not isinstance(components, dict) or not components:
        raise ValueError("component guard must contain a non-empty components object")
    validated = {}
    for name, rule in components.items():
        if not isinstance(name, str) or not isinstance(rule, dict):
            raise ValueError("component guard names and rules must be objects")
        minimum = rule.get("min_area_ratio")
        maximum = rule.get("max_area_ratio")
        if (isinstance(minimum, bool) or isinstance(maximum, bool)
                or not isinstance(minimum, (int, float))
                or not isinstance(maximum, (int, float))
                or not 0 <= float(minimum) <= float(maximum) <= 1):
            raise ValueError(f"invalid component area guard: {name}")
        validated[name] = {
            "min_area_ratio": float(minimum),
            "max_area_ratio": float(maximum),
        }
    return validated


def load_identity_manifest(path: Path) -> dict[str, dict[str, str]]:
    """Load a portable sample-to-package/camera/frame/hash binding."""
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = data.get("samples") if isinstance(data, dict) else None
    if not isinstance(samples, dict) or not samples:
        raise ValueError("identity manifest must contain a non-empty samples object")
    validated = {}
    for sample_id, item in samples.items():
        if not isinstance(sample_id, str) or not isinstance(item, dict):
            raise ValueError("identity manifest sample entries must be objects")
        package = item.get("package")
        camera = item.get("camera")
        frame_id = item.get("frame_id")
        image_sha256 = item.get("image_sha256")
        expected_sample_id = f"{package}-{camera}-{frame_id}"
        if (sample_id != expected_sample_id
                or not re.fullmatch(r"\d{3}", str(package))
                or camera not in EXPECTED_COMPONENTS
                or not re.fullmatch(r"\d+", str(frame_id))
                or not re.fullmatch(r"[0-9a-f]{64}", str(image_sha256))):
            raise ValueError(f"invalid identity manifest entry: {sample_id}")
        validated[sample_id] = {
            "package": str(package),
            "camera": str(camera),
            "frame_id": str(frame_id),
            "image_sha256": str(image_sha256),
        }
    return validated


def _write_overlay(image_bytes: bytes, mask: np.ndarray, target: Path) -> None:
    with Image.open(io.BytesIO(image_bytes)) as source:
        base = source.convert("RGBA")
    if base.size != (mask.shape[1], mask.shape[0]):
        raise ValueError("source image and candidate mask dimensions differ")
    color = Image.new("RGBA", base.size, (24, 206, 138, 0))
    color.putalpha(Image.fromarray(mask.astype(np.uint8) * 104, mode="L"))
    rendered = Image.alpha_composite(base, color).convert("RGB")
    rendered.save(target, "JPEG", quality=90)


def evaluate_sample(sample_dir: Path, output: Path,
                    expected_spec_hashes: dict[str, str],
                    allowed_source_root: Path | None = None,
                    component_guard: dict[str, dict[str, float]] | None = None,
                    identity_manifest: dict[str, dict[str, str]] | None = None) -> dict:
    sample_id = sample_dir.name
    row = {
        "sample_id": sample_id,
        "candidate_only": True,
        "human_truth": False,
        "human_confirmed": False,
        "approved_for_training": False,
        "production_deployed": False,
        "union_available": False,
        "status": "blocked",
        "reasons": [],
    }
    try:
        camera = _camera_from_sample(sample_id)
        row["camera"] = camera
        report_path = sample_dir / "results.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (report.get("candidate_only") is not True
                or report.get("human_truth") is not False
                or report.get("approved_for_training") is not False
                or report.get("production_deployed") is not False
                or report.get("source_state_modified") is not False):
            raise ValueError("input safety flags are missing or inconsistent")
        if report.get("point_sets_sha256") != expected_spec_hashes[camera]:
            raise ValueError("frozen point-set identity mismatch")
        image_path = Path(report.get("image", ""))
        image_bytes = _verified_bytes(
            image_path, report.get("image_sha256"), "source image")
        image_resolved = image_path.resolve(strict=True)
        if allowed_source_root is not None:
            allowed_resolved = allowed_source_root.resolve(strict=True)
            try:
                image_resolved.relative_to(allowed_resolved)
            except ValueError as error:
                raise ValueError(
                    "source image is outside allowed source root") from error
        if identity_manifest is not None:
            identity = identity_manifest.get(sample_id)
            if identity is None:
                raise ValueError("sample is missing from identity manifest")
            expected_parent = f"camera_{camera[-1]}"
            if (identity["camera"] != camera
                    or identity["image_sha256"] != report.get("image_sha256")
                    or image_resolved.name != f"{identity['frame_id']}.png"
                    or image_resolved.parent.name != expected_parent
                    or image_resolved.parent.parent.name != identity["package"]):
                raise ValueError("sample identity manifest binding mismatch")
        with Image.open(io.BytesIO(image_bytes)) as decoded:
            width, height = decoded.size

        rows = report.get("rows")
        if not isinstance(rows, list):
            raise ValueError("component rows are missing")
        by_name = {item.get("name"): item for item in rows
                   if isinstance(item, dict) and isinstance(item.get("name"), str)}
        expected = EXPECTED_COMPONENTS[camera]
        if len(by_name) != len(rows) or set(by_name) != set(expected):
            raise ValueError("exact required component rows are missing")

        masks = []
        component_rows = []
        for name in expected:
            component = by_name[name]
            if (component.get("candidate_available") is not True
                    or component.get("prompt_constraints_passed") is not True):
                raise ValueError(f"required component failed closed: {name}")
            mask_name = component.get("candidate_mask")
            if not isinstance(mask_name, str) or Path(mask_name).name != mask_name:
                raise ValueError(f"candidate mask path is invalid: {name}")
            mask_path = sample_dir / mask_name
            mask_bytes = _verified_bytes(
                mask_path, component.get("candidate_mask_sha256"),
                f"candidate mask: {name}")
            with Image.open(io.BytesIO(mask_bytes)) as decoded:
                if decoded.size != (width, height):
                    raise ValueError(f"candidate mask dimensions mismatch: {name}")
                mask = np.asarray(decoded.convert("L")) > 0
            geometry = _mask_geometry(mask)
            reported = component.get("geometry") or {}
            if (geometry["area_pixels"] != reported.get("area")
                    or geometry["bbox"] != reported.get("bbox")):
                raise ValueError(f"candidate mask geometry mismatch: {name}")
            guard_rule = None
            if component_guard is not None:
                guard_rule = component_guard.get(name)
                if guard_rule is None:
                    raise ValueError(f"component area guard missing: {name}")
                if not (guard_rule["min_area_ratio"]
                        <= geometry["area_ratio"]
                        <= guard_rule["max_area_ratio"]):
                    raise ValueError(
                        "component area ratio outside guard: "
                        f"{name}={geometry['area_ratio']:.6f} not in "
                        f"[{guard_rule['min_area_ratio']:.6f}, "
                        f"{guard_rule['max_area_ratio']:.6f}]")
            masks.append(mask)
            component_row = {
                "name": name,
                "mask_sha256": component["candidate_mask_sha256"],
                **geometry,
            }
            if guard_rule is not None:
                component_row["area_guard"] = guard_rule
            component_rows.append(component_row)

        union = np.logical_or.reduce(masks)
        union_geometry = _mask_geometry(union)
        intersection = np.logical_and(masks[0], masks[1]).sum()
        overlap_union = np.logical_or(masks[0], masks[1]).sum()
        union_path = output / f"{sample_id}-union-mask.png"
        Image.fromarray(union.astype(np.uint8) * 255, mode="L").save(
            union_path, "PNG", optimize=True)
        overlay_path = output / f"{sample_id}-union-overlay.jpg"
        _write_overlay(image_bytes, union, overlay_path)
        row.update({
            "status": "assistant_review_required",
            "union_available": True,
            "source_image": str(image_resolved),
            "source_image_sha256": report["image_sha256"],
            "point_sets_sha256": report["point_sets_sha256"],
            "components": component_rows,
            "component_overlap_iou": round(float(intersection) / overlap_union, 6),
            "union_geometry": union_geometry,
            "union_mask": union_path.name,
            "union_mask_sha256": file_sha256(union_path),
            "union_overlay": overlay_path.name,
            "union_overlay_sha256": file_sha256(overlay_path),
        })
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        row["reasons"] = [str(error)[:500]]
    return row


def _write_contact_sheet(output: Path, rows: list[dict]) -> Path:
    cell_width, image_height, label_height, columns = 320, 256, 30, 4
    count = max(1, len(rows))
    row_count = (count + columns - 1) // columns
    canvas = Image.new("RGB", (cell_width * columns,
                               (image_height + label_height) * row_count),
                       (15, 23, 31))
    draw, font = ImageDraw.Draw(canvas), ImageFont.load_default()
    for index, row in enumerate(rows):
        left = (index % columns) * cell_width
        top = (index // columns) * (image_height + label_height)
        if row.get("union_overlay"):
            with Image.open(output / row["union_overlay"]) as image:
                preview = image.convert("RGB")
                preview.thumbnail((cell_width, image_height))
            canvas.paste(preview, (left + (cell_width - preview.width) // 2,
                                   top + (image_height - preview.height) // 2))
        draw.rectangle((left, top + image_height,
                        left + cell_width, top + image_height + label_height),
                       fill=(25, 35, 45))
        color = (83, 210, 161) if row["union_available"] else (244, 130, 90)
        draw.text((left + 6, top + image_height + 8),
                  f"{row['sample_id']} | {row['status']}", fill=color, font=font)
    path = output / "contact_sheet.jpg"
    canvas.save(path, "JPEG", quality=88)
    return path


def build_packet(input_root: Path, output: Path,
                 camera1_spec: Path, camera3_spec: Path,
                 allowed_source_root: Path | None = None,
                 component_guard_path: Path | None = None,
                 identity_manifest_path: Path | None = None) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    expected_spec_hashes = {
        "camera1": file_sha256(camera1_spec),
        "camera3": file_sha256(camera3_spec),
    }
    allowed_resolved = (allowed_source_root.resolve(strict=True)
                        if allowed_source_root is not None else None)
    component_guard = (load_component_guard(component_guard_path)
                       if component_guard_path is not None else None)
    component_guard_sha256 = (file_sha256(component_guard_path)
                              if component_guard_path is not None else None)
    identity_manifest = (load_identity_manifest(identity_manifest_path)
                         if identity_manifest_path is not None else None)
    identity_manifest_sha256 = (file_sha256(identity_manifest_path)
                                if identity_manifest_path is not None else None)
    output.mkdir(parents=True)
    sample_dirs = sorted(path for path in input_root.iterdir()
                         if path.is_dir() and (path / "results.json").is_file())
    if not sample_dirs:
        raise ValueError("no component result directories found")
    rows = [evaluate_sample(path, output, expected_spec_hashes,
                            allowed_resolved, component_guard,
                            identity_manifest)
            for path in sample_dirs]
    contact_sheet = _write_contact_sheet(output, rows)
    packet = {
        "schema_version": "labelagent.ego_component_union_review.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_only": True,
        "human_truth": False,
        "human_confirmed": False,
        "approved_for_training": False,
        "production_deployed": False,
        "formal_state_modified": False,
        "sample_count": len(rows),
        "union_available_count": sum(row["union_available"] for row in rows),
        "failed_closed_count": sum(not row["union_available"] for row in rows),
        "point_set_sha256": expected_spec_hashes,
        "safety_gates": {
            "allowed_source_root": (
                str(allowed_resolved) if allowed_resolved is not None else None),
            "component_area_guard_enabled": component_guard is not None,
            "component_area_guard_sha256": component_guard_sha256,
            "identity_manifest_enabled": identity_manifest is not None,
            "identity_manifest_sha256": identity_manifest_sha256,
        },
        "contact_sheet": contact_sheet.name,
        "contact_sheet_sha256": file_sha256(contact_sheet),
        "samples": rows,
        "next_step": (
            "Assistant or human pixel review must compare each union against the source. "
            "No union is training truth or production-approved."),
    }
    target = output / "review_queue.json"
    target.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera1-spec", type=Path, required=True)
    parser.add_argument("--camera3-spec", type=Path, required=True)
    parser.add_argument("--allowed-source-root", type=Path)
    parser.add_argument("--component-guard", type=Path)
    parser.add_argument("--identity-manifest", type=Path)
    parser.add_argument(
        "--allow-unguarded", action="store_true",
        help="Explicitly allow legacy research runs without all safety gates")
    arguments = parser.parse_args()
    if (not arguments.allow_unguarded
            and (arguments.allowed_source_root is None
                 or arguments.component_guard is None
                 or arguments.identity_manifest is None)):
        parser.error(
            "guarded mode requires --allowed-source-root, --component-guard "
            "and --identity-manifest; use --allow-unguarded only for isolated research")
    packet = build_packet(arguments.input_root, arguments.output,
                          arguments.camera1_spec, arguments.camera3_spec,
                          arguments.allowed_source_root,
                          arguments.component_guard,
                          arguments.identity_manifest)
    print(json.dumps({
        "output": str((arguments.output / "review_queue.json").resolve()),
        "sample_count": packet["sample_count"],
        "union_available_count": packet["union_available_count"],
        "failed_closed_count": packet["failed_closed_count"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
