"""Read-only shadow evaluation for the guarded visible-vehicle recovery path.

The tool reads selected frames from one local recording, calls the exact VLM
point locator, point-SAM geometry guard, and independent blind class audit used
by LabelAgent, and writes only a fresh directory below ``artifacts``.  It never
loads or updates workbench candidates, human truth, reference frames, or
training data.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import statistics
import sys
import time

from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.agent import candidate_class_audit, workflow
from source.agent.media import open_media


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def load_samples(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(payload, dict)
            or payload.get("schema_version")
            != "labelagent.visible_vehicle_shadow_samples.v1"):
        raise ValueError("sample manifest schema is invalid")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("sample manifest must contain samples")
    normalized = []
    seen = set()
    for item in samples:
        if not isinstance(item, dict) or set(item) != {"topic", "frame"}:
            raise ValueError("every sample needs exactly topic and frame")
        topic, frame = item["topic"], item["frame"]
        if not isinstance(topic, str) or not topic or type(frame) is not int or frame < 0:
            raise ValueError("sample topic or frame is invalid")
        key = (topic, frame)
        if key in seen:
            raise ValueError("duplicate sample")
        seen.add(key)
        normalized.append({"topic": topic, "frame": frame})
    return normalized


def _fit_panel(image: Image.Image, size=(300, 240)) -> Image.Image:
    panel = Image.new("RGB", size, (24, 30, 38))
    fitted = ImageOps.contain(image.convert("RGB"), size)
    panel.paste(fitted, ((size[0] - fitted.width) // 2,
                         (size[1] - fitted.height) // 2))
    return panel


def render_tile(image_bytes: bytes, marked_bytes: bytes | None,
                heading: str) -> Image.Image:
    tile = Image.new("RGB", (640, 300), (245, 247, 250))
    draw = ImageDraw.Draw(tile)
    draw.text((12, 10), heading[:100], fill=(20, 28, 38))
    with Image.open(io.BytesIO(image_bytes)) as original:
        tile.paste(_fit_panel(original), (10, 50))
    if marked_bytes is None:
        draw.rectangle((330, 50, 630, 290), fill=(232, 236, 240))
        draw.text((420, 160), "NO CANDIDATE", fill=(90, 100, 112))
    else:
        with Image.open(io.BytesIO(marked_bytes)) as marked:
            tile.paste(_fit_panel(marked), (330, 50))
    return tile


def render_contact_sheet(tiles: list[Image.Image], columns=4) -> Image.Image:
    if not tiles:
        raise ValueError("contact sheet needs tiles")
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (640 * columns, 300 * rows), "white")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * 640,
                           (index // columns) * 300))
    return sheet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--media", required=True, type=Path)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    state_path = args.state.resolve()
    media_path = args.media.resolve()
    sample_path = args.samples.resolve()
    output = args.output_dir.resolve()
    artifact_root = (ROOT / "artifacts").resolve()
    if not output.is_relative_to(artifact_root):
        raise ValueError("output directory must be below repository artifacts")
    if output.exists():
        raise FileExistsError(output)
    for path in (state_path, media_path, sample_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    state_sha256_before = sha256_file(state_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    config = dict(state.get("config") or {})
    labels = list(state.get("labels") or [])
    if (not config.get("llm_url") or not config.get("llm_model")
            or not config.get("sam3_point_url") or "车辆" not in labels):
        raise ValueError("state lacks production model settings or 车辆 label")
    samples = load_samples(sample_path)
    output.mkdir(parents=True)
    media = open_media(media_path, output / "media_index")

    rows = []
    tiles = []
    for position, sample in enumerate(samples, 1):
        topic, frame = sample["topic"], sample["frame"]
        if topic not in media.topics or frame >= media.topics[topic]["count"]:
            raise ValueError(f"sample is outside media: {topic} frame {frame}")
        image = media.frame(topic, frame)
        started = time.perf_counter()
        row = {
            "topic": topic, "frame": frame,
            "image_sha256": sha256_bytes(image),
            "candidate_only": True, "human_truth": False,
            "approved_for_training": False,
        }
        marked = None
        trace_by_id = {}

        def capture(event):
            trace_by_id[event["id"]] = event

        try:
            with workflow.trace_calls(capture):
                candidates, evidence = workflow.visible_vehicle_point_candidate(
                    config, image, next_id=1)
                row["recovery"] = evidence
                row["candidate_count"] = len(candidates)
                if candidates:
                    candidate = candidates[0]
                    audit = candidate_class_audit.audit_candidate_class_blind(
                        config, image, candidate, labels,
                        workflow.VISIBLE_LABEL_DEFINITIONS, workflow.chat)
                    row["audit"] = audit
                    marked = candidate_class_audit.prepare_candidate_evidence(
                        image, candidate)["marked"]
        except Exception as error:
            row.update(candidate_count=0, error={
                "type": type(error).__name__, "message": str(error)[:500]})
        row["model_calls"] = [{
            "service": event.get("service"),
            "status": event.get("status"),
            "elapsed_ms": event.get("elapsed_ms"),
            "cache_hit": ((event.get("output") or {}).get("local_vlm") or {}).get(
                "cache_hit"),
            "coalesced": ((event.get("output") or {}).get("local_vlm") or {}).get(
                "coalesced"),
        } for event in trace_by_id.values()]
        row["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        status = (row.get("recovery") or {}).get("status", "error")
        route = (row.get("audit") or {}).get("route", "not_audited")
        heading = (f"{position:02d} {topic.rsplit('/', 2)[-2]} f{frame} "
                   f"{status} {route}")
        tiles.append(render_tile(image, marked, heading))
        rows.append(row)
        print(json.dumps({"progress": f"{position}/{len(samples)}",
                          "topic": topic, "frame": frame,
                          "status": status, "route": route,
                          "elapsed_seconds": row["elapsed_seconds"]},
                         ensure_ascii=False, separators=(",", ":")), flush=True)

    sheet_path = output / "contact-sheet.jpg"
    render_contact_sheet(tiles).save(sheet_path, "JPEG", quality=90,
                                     optimize=False, progressive=False)
    elapsed = [row["elapsed_seconds"] for row in rows]
    candidates = [row for row in rows if row.get("candidate_count") == 1]
    preserved = [row for row in candidates
                 if (row.get("audit") or {}).get("route") == "candidate_preserved"]
    review = [row for row in candidates
              if (row.get("audit") or {}).get("route") == "human_review_required"]
    errors = [row for row in rows if "error" in row]
    model_calls = [call for row in rows for call in row["model_calls"]]
    result = {
        "schema_version": "labelagent.visible_vehicle_shadow_evaluation.v1",
        "read_only": True,
        "assistant_visual_proxy_only": True,
        "candidate_only": True,
        "human_truth_modified": False,
        "training_data_modified": False,
        "media": {"path": str(media_path), "size_bytes": media_path.stat().st_size,
                  "mtime_ns": media_path.stat().st_mtime_ns},
        "sample_manifest": {"path": str(sample_path),
                            "sha256": sha256_file(sample_path)},
        "rows": rows,
        "summary": {
            "sample_count": len(rows),
            "candidate_count": len(candidates),
            "candidate_preserved_count": len(preserved),
            "human_review_required_count": len(review),
            "no_candidate_count": len(rows) - len(candidates) - len(errors),
            "error_count": len(errors),
            "model_call_count": len(model_calls),
            "model_cache_hit_count": sum(
                call["cache_hit"] is True for call in model_calls),
            "model_coalesced_count": sum(
                call["coalesced"] is True for call in model_calls),
            "elapsed_p50_seconds": round(statistics.median(elapsed), 3),
            "elapsed_p95_seconds": round(percentile(elapsed, 95), 3),
            "source_state_modified": sha256_file(state_path) != state_sha256_before,
        },
        "state": {"path": str(state_path), "sha256": state_sha256_before},
        "contact_sheet": {"path": str(sheet_path),
                          "sha256": sha256_file(sheet_path)},
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print("RESULT=" + json.dumps(result["summary"], ensure_ascii=False),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
