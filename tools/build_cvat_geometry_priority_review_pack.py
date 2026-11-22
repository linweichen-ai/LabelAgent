"""Build a local-only review pack grouped by geometry review priority."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _priority(score: int) -> str:
    if score >= 4:
        return "high"
    if score >= 2:
        return "medium"
    return "routine"


def build(review_manifest: Path, priority_audit: Path,
          review_root: Path, output_dir: Path, *, per_page: int = 4
          ) -> tuple[dict[str, Any], dict[str, Any]]:
    review_manifest = review_manifest.resolve()
    priority_audit = priority_audit.resolve()
    review_root = review_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ValueError("choose a fresh output directory")
    if per_page < 1 or per_page > 8:
        raise ValueError("per_page must be between 1 and 8")
    review, priority = _read(review_manifest), _read(priority_audit)
    if review.get("schema_version") != "labelagent.cvat-manual-instance-review-pack.v1":
        raise ValueError("review manifest schema differs")
    if review.get("status") != "pending_independent_review":
        raise ValueError("review manifest status differs")
    if priority.get("schema_version") != "labelagent.cvat-manual-instance-geometry-priority.v1":
        raise ValueError("priority audit schema differs")
    if priority.get("candidate_manifest_sha256") != review.get("candidate_manifest_sha256"):
        raise ValueError("priority and review candidate manifests differ")
    if priority.get("automatic_decisions") is not False or priority.get("training_authorized") is not False:
        raise ValueError("priority audit crosses the decision boundary")
    review_items = review.get("review_items")
    priority_rows = priority.get("rows")
    if not isinstance(review_items, list) or not isinstance(priority_rows, list):
        raise ValueError("review items or priority rows are absent")
    by_id = {item.get("candidate_id"): item for item in review_items
             if isinstance(item, dict)}
    if len(by_id) != len(review_items):
        raise ValueError("review item ids are missing or repeated")

    grouped: dict[str, list[dict[str, Any]]] = {"high": [], "medium": [], "routine": []}
    for row in priority_rows:
        if not isinstance(row, dict) or not isinstance(row.get("candidate_id"), str):
            raise ValueError("priority row is invalid")
        candidate_id = row["candidate_id"]
        item = by_id.get(candidate_id)
        if item is None:
            raise ValueError(f"priority candidate absent from review pack: {candidate_id}")
        score = row.get("risk_score")
        if not isinstance(score, int):
            raise ValueError(f"priority score is invalid: {candidate_id}")
        rel = item.get("focused_image_file")
        if not isinstance(rel, str) or Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise ValueError(f"unsafe focused image path: {candidate_id}")
        path = review_root / rel
        if not path.is_file() or sha256(path) != item.get("focused_image_sha256"):
            raise ValueError(f"focused image hash differs: {candidate_id}")
        grouped[_priority(score)].append({"candidate_id": candidate_id,
                                          "score": score,
                                          "flags": row.get("risk_flags") or [],
                                          "path": path,
                                          "focused_sha256": item["focused_image_sha256"]})
    for rows in grouped.values():
        rows.sort(key=lambda row: (-row["score"], row["candidate_id"]))

    output_dir.mkdir(parents=True)
    pages = []
    font = ImageFont.load_default()
    colors = {"high": (168, 47, 42), "medium": (184, 112, 20), "routine": (39, 111, 91)}
    for level in ("high", "medium", "routine"):
        rows = grouped[level]
        for offset in range(0, len(rows), per_page):
            batch = rows[offset:offset + per_page]
            page = Image.new("RGB", (1080, len(batch) * 348), (236, 241, 243))
            ids = []
            for index, row in enumerate(batch):
                with Image.open(row["path"]) as opened:
                    card = opened.convert("RGB")
                y = index * 348
                page.paste(card, (0, y + 36))
                draw = ImageDraw.Draw(page)
                draw.rectangle((0, y, 1080, y + 35), fill=colors[level])
                text = f"{level.upper()} score={row['score']} flags={','.join(row['flags']) or 'none'}"
                draw.text((10, y + 10), text, fill="white", font=font)
                ids.append(row["candidate_id"])
            path = output_dir / f"{level}_review_{offset // per_page + 1:02d}.jpg"
            page.save(path, quality=92, optimize=True)
            pages.append({"priority": level, "file": path.name,
                          "sha256": sha256(path), "candidate_ids": ids})
    manifest = {
        "schema_version": "labelagent.cvat-geometry-priority-review-pack.v1",
        "status": "assistant_triage_pending",
        "review_manifest_sha256": sha256(review_manifest),
        "priority_audit_sha256": sha256(priority_audit),
        "priority_counts": {key: len(value) for key, value in grouped.items()},
        "page_count": len(pages),
        "pages": pages,
        "assistant_triage": [],
        "automatic_decisions": False,
        "human_truth": False,
        "training_authorized": False,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    receipt = {
        "schema_version": "labelagent.cvat-geometry-priority-review-receipt.v1",
        "status": "assistant_triage_pending",
        "review_manifest_sha256": sha256(review_manifest),
        "priority_audit_sha256": sha256(priority_audit),
        "private_pack_manifest_sha256": sha256(manifest_path),
        "priority_counts": manifest["priority_counts"],
        "page_count": len(pages),
        "automatic_decisions": False,
        "human_truth": False,
        "training_authorized": False,
        "private_candidate_ids_published": False,
        "private_review_images_published": False,
    }
    return manifest, receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-manifest", required=True, type=Path)
    parser.add_argument("--priority-audit", required=True, type=Path)
    parser.add_argument("--review-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--receipt-output", required=True, type=Path)
    parser.add_argument("--per-page", type=int, default=4)
    args = parser.parse_args()
    if args.receipt_output.exists():
        raise ValueError("choose a fresh receipt output")
    _, receipt = build(args.review_manifest, args.priority_audit,
                       args.review_root, args.output_dir, per_page=args.per_page)
    args.receipt_output.parent.mkdir(parents=True, exist_ok=True)
    args.receipt_output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
