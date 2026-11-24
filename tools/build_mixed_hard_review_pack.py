"""Build a hash-bound mixed hard-frame review packet without signing truth."""
import argparse
import hashlib
import html
import io
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from source.agent.mcap_video import Video


SCHEMA = "labelagent.mixed-hard-review-pack.v1"


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test24_hashes(path):
    hashes = set()
    with Path(path).open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "test24" or row.get("evaluation_only") is not True:
                raise ValueError(f"test24 row {number} is not frozen evaluation-only")
            digest = row.get("image_sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"test24 row {number} has no valid image hash")
            hashes.add(digest)
    if not hashes:
        raise ValueError("test24 hash set is empty")
    return hashes


def safe_name(value):
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in str(value))


def write_image_bytes(data, destination):
    image = Image.open(io.BytesIO(data)).convert("RGB")
    image.save(destination, "JPEG", quality=95)


def add_review_banner(source, destination, title, note, color):
    image = Image.open(source).convert("RGB")
    banner = 74
    output = Image.new("RGB", (image.width, image.height + banner), (17, 21, 27))
    output.paste(image, (0, banner))
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, 10, output.height), fill=color)
    draw.text((22, 14), title, fill=(255, 255, 255), font=font)
    draw.text((22, 42), note, fill=(199, 208, 219), font=font)
    output.save(destination, "JPEG", quality=92)


def review_row(*, sample_id, cohort, source, raw_path, review_path, digest, note,
               source_metadata):
    return {
        "sample_id": sample_id,
        "cohort": cohort,
        "source": source,
        "source_image_sha256": digest,
        "raw_image": raw_path,
        "review_image": review_path,
        "review_note": note,
        "source_metadata": source_metadata,
        "review_status": "pending_human_review",
        "allowed_human_decisions": [
            "accept_candidate", "correct_candidate", "reject_candidate", "uncertain"
        ],
        "human_decision": None,
        "reviewer_declared_id": None,
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "training_return_allowed": False,
    }


def contact_sheet(rows, root, destination):
    columns, cell_width, cell_height = 4, 320, 300
    height = ((len(rows) + columns - 1) // columns) * cell_height
    sheet = Image.new("RGB", (columns * cell_width, height), (14, 18, 23))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    colors = {
        "slope_positive_candidate": (255, 168, 35),
        "slope_negative_candidate": (65, 160, 255),
        "slope_hard_negative_candidate": (88, 125, 235),
        "dense_pallet_candidate": (65, 205, 135),
    }
    for index, row in enumerate(rows):
        x = index % columns * cell_width
        y = index // columns * cell_height
        image = Image.open(root / row["review_image"]).convert("RGB")
        image.thumbnail((cell_width - 16, cell_height - 48))
        sheet.paste(image, (x + 8, y + 38))
        color = colors[row["cohort"]]
        draw.rectangle((x + 2, y + 2, x + cell_width - 3, y + cell_height - 3),
                       outline=color, width=3)
        draw.text((x + 10, y + 10), row["sample_id"][:42], fill=(244, 246, 249), font=font)
        draw.text((x + 10, y + 24), row["cohort"], fill=color, font=font)
    sheet.save(destination, "JPEG", quality=90)


def write_html(manifest, destination):
    cards = []
    for row in manifest["rows"]:
        cards.append(
            '<article class="card">'
            f'<h2>{html.escape(row["sample_id"])}</h2>'
            f'<p class="cohort">{html.escape(row["cohort"])}</p>'
            f'<img src="{html.escape(row["review_image"])}" alt="review image">'
            f'<p>{html.escape(row["review_note"])}</p>'
            '<p class="pending">待人工复核 · 不是人工真值 · 禁止训练</p>'
            f'<a href="{html.escape(row["raw_image"])}">打开原图副本</a>'
            '</article>')
    destination.write_text(f"""<!doctype html><html lang="zh-CN"><head>
<meta charset="utf-8"><title>LabelAgent 混合困难帧复核包</title><style>
body{{font-family:system-ui,sans-serif;background:#0f1318;color:#eef2f7;margin:24px}}
.notice{{padding:18px;background:#2a2112;border:1px solid #d99b2b;border-radius:12px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:18px;margin-top:22px}}
.card{{background:#181e25;border:1px solid #34404e;border-radius:12px;padding:14px}}
.card img{{width:100%;height:auto;border-radius:8px;background:#000}}h2{{font-size:17px}}
.cohort{{color:#86c9ff}}.pending{{color:#ffca65;font-weight:700}}a{{color:#78b7ff}}
</style></head><body><h1>LabelAgent 混合困难帧复核包</h1>
<div class="notice"><b>安全边界：</b>本包只整理机器/助手候选，不保存人工决定，
不写工作台，不授权训练。test24 图像交集必须为0。</div>
<p>共 {manifest['summary']['frames']} 张，全部待人工复核。</p>
<div class="grid">{''.join(cards)}</div></body></html>""", encoding="utf-8")


def build(*, dev_dir, dev_review, slope_results, slope_mcap, pallet_manifest,
          pallet_image_root, test24_jsonl, output, video_factory=Video):
    output = Path(output)
    raw_dir, review_dir = output / "raw", output / "review"
    raw_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    frozen_test_hashes = test24_hashes(test24_jsonl)
    rows, seen = [], set()

    def register(*, sample_id, cohort, source, data, note, metadata, color):
        digest = sha256_bytes(data)
        if digest in frozen_test_hashes:
            raise ValueError(f"test24 image intersection: {sample_id}")
        if digest in seen:
            raise ValueError(f"duplicate source image: {sample_id}")
        seen.add(digest)
        name = safe_name(sample_id) + ".jpg"
        raw_path, review_path = raw_dir / name, review_dir / name
        write_image_bytes(data, raw_path)
        add_review_banner(raw_path, review_path, f"{sample_id} | {cohort}", note, color)
        rows.append(review_row(
            sample_id=sample_id, cohort=cohort, source=source,
            raw_path=raw_path.relative_to(output).as_posix(),
            review_path=review_path.relative_to(output).as_posix(), digest=digest,
            note=note, source_metadata=metadata))

    dev_payload = read_json(dev_review)
    if (dev_payload.get("split") != "dev8" or dev_payload.get("human_truth") is not False
            or dev_payload.get("training_authorized") is not False):
        raise ValueError("dev review does not preserve candidate-only boundaries")
    for decision in dev_payload.get("decisions") or []:
        sample_id = decision["sample_id"]
        path = Path(dev_dir) / f"{sample_id}.png"
        cohort = ("slope_positive_candidate" if decision["assistant_visible_slope_candidate"]
                  else "slope_negative_candidate")
        register(sample_id=sample_id, cohort=cohort, source="frozen_dev8_assistant_review",
                 data=path.read_bytes(), note=decision.get("view") or "assistant candidate",
                 metadata={"split": "dev8", "assistant_candidate": True},
                 color=(255, 168, 35) if "positive" in cohort else (65, 160, 255))

    slope_payload = read_json(slope_results)
    if slope_payload.get("mode") != "candidate" or not slope_payload.get("rows"):
        raise ValueError("slope result packet is not a candidate packet")
    if Path(slope_payload.get("mcap", "")).resolve() != Path(slope_mcap).resolve():
        raise ValueError("slope result packet is not bound to requested MCAP")
    video = video_factory(str(Path(slope_mcap).resolve()))
    topic = slope_payload["topic"]
    for result in slope_payload["rows"]:
        frame = int(result["frame"])
        register(
            sample_id=f"heldout-slope-negative-frame-{frame}",
            cohort="slope_hard_negative_candidate", source="heldout_slope_candidate_run",
            data=video.frame(topic, frame),
            note=(f"candidate_count={result.get('candidate_count', 0)}; "
                  f"filtered={result.get('filtered_count', 0)}; human review required"),
            metadata={"frame": frame, "topic": topic, "candidate_run_only": True},
            color=(88, 125, 235))

    pallet_payload = read_json(pallet_manifest)
    if (pallet_payload.get("candidate_only") is not True
            or pallet_payload.get("human_pixel_truth") is not False
            or pallet_payload.get("approved_for_training") is not False):
        raise ValueError("pallet packet does not preserve candidate-only boundaries")
    image_root = Path(pallet_image_root)
    for frame in pallet_payload.get("frames") or []:
        sequence, frame_id, camera = frame["sequence"], frame["frame_id"], int(frame["camera"])
        path = image_root / sequence / f"camera_{camera}" / f"{frame_id}.png"
        register(
            sample_id=f"dense-pallet-{sequence}-{frame_id}-camera-{camera}",
            cohort="dense_pallet_candidate", source="dense_pallet_review_packet",
            data=path.read_bytes(),
            note=(f"pallet labels={frame.get('pallet_label_count')}; "
                  f"review objects={len(frame.get('review_scope_obj_ids') or [])}"),
            metadata={"sequence": sequence, "frame_id": frame_id, "camera": camera,
                      "review_scope_obj_ids": frame.get("review_scope_obj_ids") or []},
            color=(65, 205, 135))

    if not 24 <= len(rows) <= 40:
        raise ValueError(f"review packet must contain 24-40 unique frames, got {len(rows)}")
    counts = {name: sum(row["cohort"] == name for row in rows) for name in {
        "slope_positive_candidate", "slope_negative_candidate",
        "slope_hard_negative_candidate", "dense_pallet_candidate"}}
    manifest = {
        "schema_version": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "candidate_only_pending_human_review_not_deployed",
        "summary": {"frames": len(rows), "unique_source_hashes": len(seen),
                    "pending_human_review": len(rows), "test24_intersection_count": 0,
                    "human_truth_frames": 0, "training_approved_frames": 0,
                    "cohort_counts": counts},
        "source_receipts": {
            "dev_review_sha256": file_sha256(dev_review),
            "slope_results_sha256": file_sha256(slope_results),
            "slope_mcap_sha256": file_sha256(slope_mcap),
            "pallet_manifest_sha256": file_sha256(pallet_manifest),
            "test24_jsonl_sha256": file_sha256(test24_jsonl),
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
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    contact_sheet(rows, output, output / "contact-sheet.jpg")
    write_html(manifest, output / "index.html")
    files = sorted(path for path in output.rglob("*") if path.is_file())
    receipt = {
        "schema_version": "labelagent.mixed-hard-review-pack-receipt.v1",
        "manifest_sha256": file_sha256(manifest_path),
        "files": [{"path": path.relative_to(output).as_posix(),
                   "sha256": file_sha256(path), "bytes": path.stat().st_size}
                  for path in files],
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-dir", required=True)
    parser.add_argument("--dev-review", required=True)
    parser.add_argument("--slope-results", required=True)
    parser.add_argument("--slope-mcap", required=True)
    parser.add_argument("--pallet-manifest", required=True)
    parser.add_argument("--pallet-image-root", required=True)
    parser.add_argument("--test24-jsonl", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = build(**vars(args))
    print(json.dumps({"output": str(Path(args.output).resolve()),
                      "summary": manifest["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
