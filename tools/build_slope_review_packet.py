"""Build a read-only, hash-bound slope review packet without signing truth."""
import argparse
import hashlib
import html
import io
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from source.agent.mcap_video import Video


SCHEMA = "labelagent.slope-review-packet.v1"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_row(*, frame, cohort, source_frame_sha256, raw_path, review_path,
                  route, candidate_count, target_provenance):
    """Create a candidate-only row; no caller can promote it through arguments."""
    return {
        "frame": int(frame),
        "cohort": cohort,
        "source_frame_sha256": source_frame_sha256,
        "raw_image": raw_path,
        "review_image": review_path,
        "route": route,
        "candidate_count": int(candidate_count),
        "target_provenance": target_provenance,
        "review_status": "pending_human_review",
        "allowed_human_decisions": [
            "slope_present_and_mask_acceptable",
            "slope_present_mask_needs_correction",
            "no_slope_present",
            "uncertain",
        ],
        "human_decision": None,
        "reviewer_declared_id": None,
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "training_return_allowed": False,
    }


def add_banner(source, destination, title, subtitle, color):
    image = Image.open(source).convert("RGB")
    banner = 66
    output = Image.new("RGB", (image.width, image.height + banner), (18, 22, 28))
    output.paste(image, (0, banner))
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, 10, output.height), fill=color)
    draw.text((22, 13), title, fill=(255, 255, 255), font=font)
    draw.text((22, 37), subtitle, fill=(196, 205, 216), font=font)
    output.save(destination, quality=92)


def write_contact_sheet(rows, root, destination):
    columns, cell_width, cell_height = 4, 320, 300
    sheet = Image.new("RGB", (columns * cell_width,
                              ((len(rows) + columns - 1) // columns) * cell_height),
                      (15, 18, 23))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, row in enumerate(rows):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        image = Image.open(root / row["review_image"]).convert("RGB")
        image.thumbnail((cell_width - 16, cell_height - 42))
        sheet.paste(image, (x + 8, y + 30))
        color = (255, 170, 40) if row["cohort"] == "positive_proxy" else (80, 170, 255)
        draw.rectangle((x + 2, y + 2, x + cell_width - 3, y + cell_height - 3),
                       outline=color, width=3)
        draw.text((x + 10, y + 9),
                  f"frame {row['frame']} | {row['cohort']} | pending",
                  fill=(240, 240, 240), font=font)
    sheet.save(destination, quality=90)


def write_html(manifest, destination):
    cards = []
    for row in manifest["rows"]:
        cards.append(
            '<article class="card">'
            f'<h2>Frame {row["frame"]} · {html.escape(row["cohort"])}</h2>'
            f'<img src="{html.escape(row["review_image"])}" alt="frame {row["frame"]}">'
            f'<p>Route: {html.escape(row["route"])} · candidates: {row["candidate_count"]}</p>'
            '<p class="pending">Pending human review · candidate only · training disabled</p>'
            f'<a href="{html.escape(row["raw_image"])}">Open raw image</a>'
            '</article>')
    body = "\n".join(cards)
    destination.write_text(f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>斜坡人工复核包</title>
<style>
body{{font-family:system-ui,sans-serif;background:#0f1217;color:#eef2f7;margin:24px}}
.notice{{max-width:1100px;padding:18px;background:#282113;border:1px solid #d99b2b;border-radius:12px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:18px;margin-top:22px}}
.card{{background:#171c23;border:1px solid #303846;border-radius:12px;padding:14px}}
.card img{{width:100%;height:auto;border-radius:8px;background:#000}}
.card h2{{font-size:17px;margin:0 0 10px}}.card p{{color:#bbc4d0}}
.pending{{color:#ffca65!important;font-weight:700}}a{{color:#78b7ff}}
</style></head><body>
<h1>斜坡人工复核包</h1>
<div class="notice"><b>重要：</b>本页只用于集中目视复核。绿色旧轮廓和机器候选都不是人工真值；
本页不会保存决定。请在 LabelAgent 工作台逐帧修改并通过人工确认门禁提交，禁止直接回流训练。</div>
<p>共 {len(manifest['rows'])} 帧；正候选代理 {manifest['summary']['positive_proxy_frames']}，
困难负候选 {manifest['summary']['negative_cohort_frames']}。</p>
<div class="grid">{body}</div></body></html>""", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcap", required=True)
    parser.add_argument("--positive-results", required=True)
    parser.add_argument("--negative-results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    mcap = Path(args.mcap).resolve()
    positive_path = Path(args.positive_results).resolve()
    negative_path = Path(args.negative_results).resolve()
    positive = json.loads(positive_path.read_text(encoding="utf-8"))
    negative = json.loads(negative_path.read_text(encoding="utf-8"))
    if positive.get("mcap") != str(mcap) or negative.get("mcap") != str(mcap):
        raise ValueError("Result packets are not bound to the requested MCAP")
    if positive.get("target_provenance", {}).get("human_truth_verified") is not False:
        raise ValueError("Positive packet must explicitly remain non-human truth")
    if negative.get("target_provenance", {}).get("human_truth_verified") is not False:
        raise ValueError("Negative packet must explicitly remain non-human truth")
    if positive.get("topic") != negative.get("topic"):
        raise ValueError("Positive and negative packets use different topics")

    output = Path(args.output)
    raw_dir, review_dir = output / "raw", output / "review"
    raw_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    video = Video(str(mcap))
    topic = positive["topic"]
    positive_root = positive_path.parent
    rows = []

    final_rows = positive.get("final_slope_summary", {}).get("rows") or []
    for result in final_rows:
        frame = int(result["frame"])
        frame_bytes = video.frame(topic, frame)
        raw_path = raw_dir / f"frame-{frame}.jpg"
        Image.open(io.BytesIO(frame_bytes)).convert("RGB").save(raw_path, quality=95)
        source_name = (f"frame-{frame}-crop-fallback.jpg"
                       if result.get("route") == "overlap_crop_fallback"
                       else f"frame-{frame}-filtered-slope.jpg")
        source = positive_root / source_name
        if not source.exists():
            raise FileNotFoundError(source)
        review_path = review_dir / f"frame-{frame}-positive-proxy.jpg"
        add_banner(source, review_path,
                   f"FRAME {frame} | POSITIVE PROXY | {result.get('route')}",
                   "GREEN=legacy auto target; MAGENTA/CYAN=machine output; NOT HUMAN TRUTH",
                   (255, 164, 32))
        rows.append(candidate_row(
            frame=frame, cohort="positive_proxy",
            source_frame_sha256=hashlib.sha256(frame_bytes).hexdigest(),
            raw_path=raw_path.relative_to(output).as_posix(),
            review_path=review_path.relative_to(output).as_posix(),
            route=result.get("route", "unknown"),
            candidate_count=int(result.get("retained") or 0),
            target_provenance=positive["target_provenance"]["kind"]))

    for result in negative.get("rows") or []:
        frame = int(result["frame"])
        frame_bytes = video.frame(topic, frame)
        raw_path = raw_dir / f"frame-{frame}.jpg"
        Image.open(io.BytesIO(frame_bytes)).convert("RGB").save(raw_path, quality=95)
        review_path = review_dir / f"frame-{frame}-negative-cohort.jpg"
        add_banner(raw_path, review_path,
                   f"FRAME {frame} | UNVERIFIED NO-SLOPE COHORT",
                   f"retained slope candidates={result.get('candidate_count', 0)}; HUMAN REVIEW REQUIRED",
                   (60, 150, 255))
        rows.append(candidate_row(
            frame=frame, cohort="negative_cohort",
            source_frame_sha256=hashlib.sha256(frame_bytes).hexdigest(),
            raw_path=raw_path.relative_to(output).as_posix(),
            review_path=review_path.relative_to(output).as_posix(),
            route="production_path",
            candidate_count=int(result.get("candidate_count") or 0),
            target_provenance=negative["target_provenance"]["kind"]))

    if len({row["frame"] for row in rows}) != len(rows):
        raise ValueError("Positive and negative review cohorts overlap")
    manifest = {
        "schema_version": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "mcap": str(mcap),
        "mcap_sha256": file_sha256(mcap),
        "topic": topic,
        "source_results": {
            "positive": {"path": str(positive_path), "sha256": file_sha256(positive_path)},
            "negative": {"path": str(negative_path), "sha256": file_sha256(negative_path)},
        },
        "summary": {
            "frames": len(rows),
            "positive_proxy_frames": sum(row["cohort"] == "positive_proxy" for row in rows),
            "negative_cohort_frames": sum(row["cohort"] == "negative_cohort" for row in rows),
            "pending_human_review": len(rows),
            "human_truth_frames": 0,
            "training_approved_frames": 0,
        },
        "rows": rows,
        "boundary": (
            "This packet is a read-only review aid. Decisions must be submitted through the "
            "LabelAgent human-review gate; this packet never signs truth or training approval."),
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_contact_sheet(rows, output, output / "contact-sheet.jpg")
    write_html(manifest, output / "index.html")
    files = sorted(path for path in output.rglob("*") if path.is_file())
    receipt = {
        "schema_version": "labelagent.slope-review-packet-receipt.v1",
        "manifest_sha256": file_sha256(manifest_path),
        "files": [{"path": path.relative_to(output).as_posix(),
                   "sha256": file_sha256(path), "bytes": path.stat().st_size}
                  for path in files],
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "summary": manifest["summary"],
                      "manifest_sha256": receipt["manifest_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
