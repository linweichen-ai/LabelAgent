"""Diversify a candidate-only hard-review packet without creating truth."""
import argparse
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from tools.build_mixed_hard_adjudication_workspace import validate_packet
from tools.build_mixed_hard_review_pack import (
    add_review_banner,
    file_sha256,
    review_row,
    safe_name,
    test24_hashes,
    write_html,
    write_image_bytes,
)


SCHEMA = "labelagent.mixed-hard-review-pack.v1"
SELECTION_SCHEMA = "labelagent.diverse-hard-selection.v1"
ALLOWED_NEW_COHORTS = {
    "batchb_flat_negative_candidate": (65, 160, 255),
    "batchb_ramp_candidate": (255, 168, 35),
    "batchb_class_conflict_candidate": (180, 115, 255),
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _safe_boundaries(payload, label):
    if payload.get("candidate_only") is not True:
        raise ValueError(f"{label} must remain candidate_only")
    for key in ("human_truth", "approved_for_training"):
        if payload.get(key) is not False:
            raise ValueError(f"unsafe {label} boundary: {key}")


def _copy_parent_row(parent, output, row):
    copied = json.loads(json.dumps(row))
    for key in ("raw_image", "review_image"):
        source = parent / row[key]
        destination = output / row[key]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return copied


def _contact_sheet(rows, root, destination):
    columns, cell_width, cell_height = 4, 320, 300
    height = ((len(rows) + columns - 1) // columns) * cell_height
    sheet = Image.new("RGB", (columns * cell_width, height), (14, 18, 23))
    draw, font = ImageDraw.Draw(sheet), ImageFont.load_default()
    legacy = {
        "slope_positive_candidate": (255, 168, 35),
        "slope_negative_candidate": (65, 160, 255),
        "slope_hard_negative_candidate": (88, 125, 235),
        "dense_pallet_candidate": (65, 205, 135),
    }
    colors = {**legacy, **ALLOWED_NEW_COHORTS}
    for index, row in enumerate(rows):
        x, y = index % columns * cell_width, index // columns * cell_height
        image = Image.open(root / row["review_image"]).convert("RGB")
        image.thumbnail((cell_width - 16, cell_height - 48))
        sheet.paste(image, (x + 8, y + 38))
        color = colors.get(row["cohort"], (190, 190, 190))
        draw.rectangle((x + 2, y + 2, x + cell_width - 3, y + cell_height - 3),
                       outline=color, width=3)
        draw.text((x + 10, y + 10), row["sample_id"][:42],
                  fill=(244, 246, 249), font=font)
        draw.text((x + 10, y + 24), row["cohort"], fill=color, font=font)
    sheet.save(destination, "JPEG", quality=90)


def build(*, parent_packet, selection_manifest, test24_jsonl, output):
    parent = Path(parent_packet)
    output = Path(output)
    parent_manifest, parent_digest = validate_packet(parent)
    _safe_boundaries(parent_manifest, "parent packet")
    selection = read_json(selection_manifest)
    if selection.get("schema_version") != SELECTION_SCHEMA:
        raise ValueError("unsupported selection manifest schema")
    _safe_boundaries(selection, "selection manifest")
    if selection.get("test24_used") is not False:
        raise ValueError("selection manifest must declare test24_used=false")
    declared_test_digest = selection.get("test24_jsonl_sha256")
    actual_test_digest = file_sha256(test24_jsonl)
    if declared_test_digest != actual_test_digest:
        raise ValueError("test24 JSONL binding differs")
    frozen_test_hashes = test24_hashes(test24_jsonl)

    parent_rows = {row["sample_id"]: row for row in parent_manifest["rows"]}
    replace_ids = selection.get("replace_sample_ids")
    samples = selection.get("samples")
    if not isinstance(replace_ids, list) or not isinstance(samples, list):
        raise ValueError("selection manifest must contain replacement lists")
    if len(replace_ids) != len(samples) or not replace_ids:
        raise ValueError("replacement and new sample counts must match")
    if len(set(replace_ids)) != len(replace_ids):
        raise ValueError("duplicate replacement sample id")
    for sample_id in replace_ids:
        row = parent_rows.get(sample_id)
        if not row or row.get("cohort") != "slope_hard_negative_candidate":
            raise ValueError(f"replacement is not a hard-negative candidate: {sample_id}")

    output.mkdir(parents=True, exist_ok=False)
    rows, seen = [], set()
    for row in parent_manifest["rows"]:
        if row["sample_id"] in replace_ids:
            continue
        digest = row["source_image_sha256"]
        if digest in frozen_test_hashes:
            raise ValueError(f"parent row intersects test24: {row['sample_id']}")
        if digest in seen:
            raise ValueError(f"duplicate parent source image: {row['sample_id']}")
        seen.add(digest)
        rows.append(_copy_parent_row(parent, output, row))

    sample_ids = {row["sample_id"] for row in rows}
    for sample in samples:
        sample_id = sample.get("sample_id")
        cohort = sample.get("cohort")
        if not isinstance(sample_id, str) or not sample_id or sample_id in sample_ids:
            raise ValueError("missing or duplicate new sample id")
        if cohort not in ALLOWED_NEW_COHORTS:
            raise ValueError(f"unsupported diverse cohort: {cohort}")
        source = Path(sample.get("source_path") or "")
        if not source.is_file():
            raise ValueError(f"new sample source is missing: {sample_id}")
        digest = file_sha256(source)
        if digest != sample.get("source_image_sha256"):
            raise ValueError(f"new sample hash differs: {sample_id}")
        if digest in frozen_test_hashes:
            raise ValueError(f"test24 image intersection: {sample_id}")
        if digest in seen:
            raise ValueError(f"duplicate source image: {sample_id}")
        metadata = sample.get("source_metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"source metadata is required: {sample_id}")
        if any(key.lower().endswith("path") for key in metadata):
            raise ValueError(f"source metadata must not publish paths: {sample_id}")
        data = source.read_bytes()
        name = safe_name(sample_id) + ".jpg"
        raw_path, review_path = output / "raw" / name, output / "review" / name
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        review_path.parent.mkdir(parents=True, exist_ok=True)
        write_image_bytes(data, raw_path)
        note = str(sample.get("review_note") or "human review required")
        add_review_banner(raw_path, review_path, f"{sample_id} | {cohort}", note,
                          ALLOWED_NEW_COHORTS[cohort])
        rows.append(review_row(
            sample_id=sample_id,
            cohort=cohort,
            source="declared_local_diversity_pool",
            raw_path=raw_path.relative_to(output).as_posix(),
            review_path=review_path.relative_to(output).as_posix(),
            digest=digest,
            note=note,
            source_metadata=metadata,
        ))
        seen.add(digest)
        sample_ids.add(sample_id)

    if len(rows) != len(parent_manifest["rows"]) or not 24 <= len(rows) <= 40:
        raise ValueError("diversification must preserve a 24-40 frame packet")
    counts = dict(sorted(Counter(row["cohort"] for row in rows).items()))
    datasets = sorted({str(row.get("source_metadata", {}).get("dataset", ""))
                       for row in rows if row.get("source_metadata", {}).get("dataset")})
    sequences = sorted({str(row.get("source_metadata", {}).get("sequence", ""))
                        for row in rows if row.get("source_metadata", {}).get("sequence")})
    cameras = sorted({str(row.get("source_metadata", {}).get("camera", ""))
                      for row in rows if row.get("source_metadata", {}).get("camera")})
    manifest = {
        "schema_version": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "candidate_only_pending_human_review_not_deployed",
        "summary": {
            "frames": len(rows),
            "unique_source_hashes": len(seen),
            "replaced_parent_frames": len(replace_ids),
            "added_diverse_frames": len(samples),
            "test24_intersection_count": 0,
            "human_truth_frames": 0,
            "training_approved_frames": 0,
            "cohort_counts": counts,
            "declared_datasets": datasets,
            "declared_sequences": sequences,
            "declared_cameras": cameras,
        },
        "source_receipts": {
            "parent_manifest_sha256": parent_digest,
            "selection_manifest_sha256": file_sha256(selection_manifest),
            "test24_jsonl_sha256": actual_test_digest,
        },
        "rows": rows,
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "training_authorized": False,
        "test24_used": False,
        "production_deployed": False,
        "human_truth_writes": 0,
        "workbench_writes": 0,
        "external_api_calls": 0,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    _contact_sheet(rows, output, output / "contact-sheet.jpg")
    write_html(manifest, output / "index.html")
    files = sorted(path for path in output.rglob("*") if path.is_file())
    receipt = {
        "schema_version": "labelagent.diverse-mixed-hard-review-pack-receipt.v1",
        "manifest_sha256": file_sha256(manifest_path),
        "files": [{"path": path.relative_to(output).as_posix(),
                   "sha256": file_sha256(path), "bytes": path.stat().st_size}
                  for path in files],
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "test24_used": False,
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-packet", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--test24-jsonl", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = build(**vars(args))
    print(json.dumps({"output": str(Path(args.output).resolve()),
                      "summary": manifest["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
