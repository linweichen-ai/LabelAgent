#!/usr/bin/env python3
"""Compare bounded pallet edge policies without changing any candidate mask."""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import re
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


INPUT_SCHEMA = "labelagent.pallet-gradient-policy-input-map.v1"
SCAN_SCHEMA = "labelagent.pallet-evidence-roi-offline-scan.v2"
OUTPUT_SCHEMA = "labelagent.pallet-gradient-policy-comparison.v1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
INPUT_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_bound(path: Path, expected_sha256: str, label: str) -> bytes:
    if not SHA256_PATTERN.fullmatch(str(expected_sha256)):
        raise ValueError(f"{label} SHA-256 invalid")
    first = path.read_bytes()
    if _digest(first) != expected_sha256:
        raise ValueError(f"{label} SHA-256 drift")
    second = path.read_bytes()
    if second != first:
        raise ValueError(f"{label} changed during read")
    return first


def _safe_relative(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"{label} path invalid")
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} path escapes input root") from exc
    return candidate


def _safe_basename(root: Path, value: object, label: str) -> Path:
    if (not isinstance(value, str) or not value
            or Path(value).name != value or "/" in value or "\\" in value):
        raise ValueError(f"{label} basename invalid")
    return _safe_relative(root, value, label)


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} invalid")
    return value


def _nonnegative_decimal(value: object, label: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{label} invalid")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{label} invalid") from exc
    if not result.is_finite() or result < 0:
        raise ValueError(f"{label} invalid")
    return result


def _mask(payload: bytes, label: str) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(payload)) as decoded:
            array = np.asarray(decoded.convert("L"))
    except Exception as exc:
        raise ValueError(f"{label} mask decode failed") from exc
    if array.ndim != 2 or not array.size:
        raise ValueError(f"{label} mask invalid")
    return array != 0


def _foreground_component_count(mask: np.ndarray) -> int:
    count, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return max(int(count) - 1, 0)


def _validate_scan(entry: dict, input_root: Path, identities: set) -> tuple:
    input_id = entry.get("input_id")
    if not isinstance(input_id, str) or not INPUT_ID_PATTERN.fullmatch(input_id):
        raise ValueError("input_id invalid")
    manifest_path = _safe_relative(
        input_root, entry.get("manifest_path"), f"{input_id} manifest")
    expected_manifest_sha = entry.get("manifest_sha256")
    payload = _read_bound(
        manifest_path, expected_manifest_sha, f"{input_id} manifest")
    try:
        manifest = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{input_id} manifest JSON invalid") from exc
    required_flags = {
        "candidate_only": True,
        "human_truth": False,
        "training_authorized": False,
        "deployment_authorized": False,
        "workbench_write": False,
        "geometry_policy": "appearance_evidence_roi",
    }
    if manifest.get("schema_version") != SCAN_SCHEMA:
        raise ValueError(f"{input_id} scan schema invalid")
    for key, expected in required_flags.items():
        if manifest.get(key) != expected:
            raise ValueError(f"{input_id} scan safety flag {key} invalid")
    snapshot_sha = manifest.get("snapshot_manifest_sha256")
    smoke_sha = manifest.get("smoke_summary_sha256")
    if (not SHA256_PATTERN.fullmatch(str(snapshot_sha))
            or not SHA256_PATTERN.fullmatch(str(smoke_sha))):
        raise ValueError(f"{input_id} source binding invalid")
    rows = manifest.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{input_id} rows invalid")
    pallet_count = _nonnegative_int(
        manifest.get("pallet_candidate_count"), "pallet candidate count")
    roi_count = _nonnegative_int(
        manifest.get("roi_evidence_scale_gate_passed_count"),
        "ROI scale count")
    changed_count = _nonnegative_int(
        manifest.get("changed_candidate_count"), "changed candidate count")
    if len(rows) != pallet_count:
        raise ValueError(f"{input_id} pallet count mismatch")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{input_id} row invalid")
    if any(type(row.get("roi_evidence_scale_gate_passed")) is not bool
           for row in rows):
        raise ValueError(f"{input_id} ROI scale flag invalid")
    if (sum(row["roi_evidence_scale_gate_passed"] for row in rows)
            != roi_count):
        raise ValueError(f"{input_id} ROI scale count mismatch")
    if (sum(isinstance(row.get("changed_pixels"), int)
            and not isinstance(row.get("changed_pixels"), bool)
            and row["changed_pixels"] > 0 for row in rows)
            != changed_count):
        raise ValueError(f"{input_id} changed count mismatch")

    output_rows = []
    bound_files = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{input_id} row invalid")
        frame = row.get("frame")
        candidate_id = row.get("candidate_id")
        changed = row.get("changed_pixels")
        if (isinstance(frame, bool) or not isinstance(frame, int) or frame < 0
                or isinstance(candidate_id, bool)
                or not isinstance(candidate_id, int) or candidate_id < 0
                or isinstance(changed, bool) or not isinstance(changed, int)
                or changed < 0):
            raise ValueError(f"{input_id} row identity invalid")
        for key in ("source_image_sha256", "candidate_file_sha256",
                    "before_mask_sha256", "after_mask_sha256"):
            if not SHA256_PATTERN.fullmatch(str(row.get(key))):
                raise ValueError(f"{input_id} row {key} invalid")
        # The source image plus candidate ID remains stable across a smoke
        # replay; smoke-summary SHA must not make the same candidate count twice.
        identity = (row["source_image_sha256"], candidate_id)
        if identity in identities:
            raise ValueError("duplicate candidate identity")
        identities.add(identity)
        if changed == 0:
            continue

        before_path = _safe_basename(
            manifest_path.parent, row.get("before_mask"), "before mask")
        after_path = _safe_basename(
            manifest_path.parent, row.get("after_mask"), "after mask")
        before_payload = _read_bound(
            before_path, row["before_mask_sha256"], "before mask")
        after_payload = _read_bound(
            after_path, row["after_mask_sha256"], "after mask")
        bound_files.extend(((before_path, row["before_mask_sha256"]),
                            (after_path, row["after_mask_sha256"])))
        before = _mask(before_payload, "before")
        after = _mask(after_payload, "after")
        if before.shape != after.shape:
            raise ValueError("mask dimensions differ")
        if np.any(before & ~after):
            raise ValueError("candidate mask is not add-only")
        if int(np.count_nonzero(after & ~before)) != changed:
            raise ValueError("candidate changed pixel count mismatch")
        if _foreground_component_count(before) != _foreground_component_count(after):
            raise ValueError("candidate foreground component count changed")

        audit = row.get("audit")
        if (not isinstance(audit, dict)
                or audit.get("geometry_cleanup_version") != 8
                or audit.get("geometry_cleanup_policy")
                != "appearance_evidence_roi"
                or audit.get("add_only") is not True
                or audit.get("component_merge_count") != 0
                or audit.get("required_structural_hole_retention") != 1.0
                or audit.get("changed_pixels") != changed):
            raise ValueError("candidate audit safety fields invalid")
        retention = audit.get("minimum_structural_hole_retention")
        if (retention is not None
                and _nonnegative_decimal(
                    retention, "minimum structural hole retention")
                < Decimal("1")):
            raise ValueError("candidate structural hole retention invalid")
        evidence = audit.get("filled_image_boundary_notch_evidence")
        if not isinstance(evidence, list):
            raise ValueError("candidate gradient evidence invalid")
        if (len(evidence) != audit.get("filled_image_boundary_notch_count")
                or sum(item.get("area", 0) for item in evidence
                       if isinstance(item, dict))
                != audit.get("filled_image_boundary_notch_pixels")):
            raise ValueError("candidate notch evidence count mismatch")
        if not evidence:
            output_rows.append({
                "input_id": input_id,
                "frame": frame,
                "candidate_id": candidate_id,
                "changed_pixels": changed,
                "notch_count": 0,
                "status": "insufficient",
                "policies": None,
            })
            continue

        pairs = []
        for item in evidence:
            if (not isinstance(item, dict)
                    or isinstance(item.get("area"), bool)
                    or not isinstance(item.get("area"), int)
                    or item["area"] <= 0
                    or item.get("scale_evidence_route") != "roi_absolute"):
                raise ValueError("candidate notch evidence invalid")
            notch = _nonnegative_decimal(
                item.get("notch_gradient_mean"), "notch gradient")
            ring = _nonnegative_decimal(
                item.get("ring_gradient_mean"), "ring gradient")
            pairs.append((notch, ring))
        relative0 = all(notch <= ring for notch, ring in pairs)
        relative8 = all(notch <= ring + Decimal("8") for notch, ring in pairs)
        absolute64 = all(notch <= Decimal("64") for notch, _ in pairs)
        output_rows.append({
            "input_id": input_id,
            "frame": frame,
            "candidate_id": candidate_id,
            "changed_pixels": changed,
            "notch_count": len(pairs),
            "maximum_notch_gradient": float(max(notch for notch, _ in pairs)),
            "maximum_gradient_excess": float(max(
                notch - ring for notch, ring in pairs)),
            "status": "evaluated",
            "policies": {
                "absolute64": absolute64,
                "combined": absolute64 and relative8,
                "relative0": relative0,
                "relative_plus_8": relative8,
            },
        })
    return ({"input_id": input_id,
             "manifest_sha256": expected_manifest_sha},
            output_rows, bound_files, manifest_path)


def compare(input_map_path, input_map_sha256, output_dir):
    input_map_path = Path(input_map_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    input_payload = _read_bound(input_map_path, input_map_sha256, "input map")
    try:
        input_map = json.loads(input_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("input map JSON invalid") from exc
    entries = input_map.get("inputs")
    if (input_map.get("schema_version") != INPUT_SCHEMA
            or not isinstance(entries, list) or not entries):
        raise ValueError("input map invalid")
    input_ids = [entry.get("input_id") for entry in entries
                 if isinstance(entry, dict)]
    input_hashes = [entry.get("manifest_sha256") for entry in entries
                    if isinstance(entry, dict)]
    if (len(input_ids) != len(entries) or len(set(input_ids)) != len(entries)
            or len(set(input_hashes)) != len(entries)):
        raise ValueError("input map identities invalid")

    identities = set()
    sources = []
    rows = []
    bound_files = []
    manifests = []
    for entry in sorted(entries, key=lambda item: item["input_id"]):
        source, new_rows, new_files, manifest_path = _validate_scan(
            entry, input_map_path.parent, identities)
        sources.append(source)
        rows.extend(new_rows)
        bound_files.extend(new_files)
        manifests.append((manifest_path, entry["manifest_sha256"]))
    rows.sort(key=lambda row: (
        row["input_id"], row["frame"], row["candidate_id"]))
    evaluated = [row for row in rows if row["status"] == "evaluated"]
    policy_names = ("absolute64", "combined", "relative0", "relative_plus_8")
    policies = {
        name: {
            "pass": sum(row["policies"][name] for row in evaluated),
            "reject": sum(not row["policies"][name] for row in evaluated),
            "insufficient": len(rows) - len(evaluated),
        }
        for name in policy_names
    }
    report = {
        "schema_version": OUTPUT_SCHEMA,
        "status": "diagnostic_only_threshold_not_frozen",
        "candidate_only": True,
        "human_truth": False,
        "training_authorized": False,
        "deployment_authorized": False,
        "workbench_write": False,
        "mask_write": False,
        "input_map_sha256": input_map_sha256,
        "sources": sources,
        "changed_candidate_count": len(rows),
        "evaluated_candidate_count": len(evaluated),
        "insufficient_candidate_count": len(rows) - len(evaluated),
        "policies": policies,
        "rows": rows,
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        report_payload = (json.dumps(
            report, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
        (staging / "report.json").write_bytes(report_payload)
        if json.loads((staging / "report.json").read_text(encoding="utf-8")) != report:
            raise ValueError("report readback mismatch")
        _read_bound(input_map_path, input_map_sha256, "input map")
        for path, expected in manifests + bound_files:
            _read_bound(path, expected, "bound input")
        staging.rename(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Compare read-only pallet boundary gradient policies.")
    parser.add_argument("--input-map", required=True, type=Path)
    parser.add_argument("--input-map-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = compare(args.input_map, args.input_map_sha256, args.output)
    print(json.dumps({
        "schema_version": report["schema_version"],
        "changed_candidate_count": report["changed_candidate_count"],
        "evaluated_candidate_count": report["evaluated_candidate_count"],
        "policies": report["policies"],
        "workbench_write": report["workbench_write"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
