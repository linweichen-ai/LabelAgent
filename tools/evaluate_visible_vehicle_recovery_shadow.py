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

from PIL import Image, ImageChops, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.agent import candidate_class_audit, workflow
from source.agent.evaluation_provenance import classify_target_provenance
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


def load_reference_session(path: Path, media_path: Path) -> dict:
    """Load one saved session without treating its ``truth`` key as human truth."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    media_key = str(media_path)
    candidates = []
    if isinstance(payload, dict):
        candidates.append(payload.get(media_key))
        file_sessions = payload.get("file_sessions")
        if isinstance(file_sessions, dict):
            candidates.append(file_sessions.get(media_key))
        if isinstance(payload.get("truth"), dict):
            candidates.append(payload)
    for session in candidates:
        if isinstance(session, dict) and isinstance(session.get("truth"), dict):
            return session
    raise ValueError("reference session does not contain this media path")


def without_fast_route(config: dict) -> dict:
    primary = dict(config)
    for key in tuple(primary):
        if key.startswith("llm_fast_"):
            primary.pop(key)
    return primary


def polygon_mask(points: list, size: tuple[int, int]) -> Image.Image | None:
    normalized = []
    for point in points or []:
        if (isinstance(point, (list, tuple)) and len(point) >= 2
                and all(isinstance(value, (int, float)) for value in point[:2])):
            normalized.append((round(point[0]), round(point[1])))
    if len(normalized) < 3:
        return None
    mask = Image.new("1", size, 0)
    ImageDraw.Draw(mask).polygon(normalized, fill=1)
    return mask


def polygon_iou(left: list, right: list, size: tuple[int, int]) -> float:
    left_mask = polygon_mask(left, size)
    right_mask = polygon_mask(right, size)
    if left_mask is None or right_mask is None:
        return 0.0
    intersection = ImageChops.logical_and(left_mask, right_mask).histogram()[255]
    union = ImageChops.logical_or(left_mask, right_mask).histogram()[255]
    return intersection / union if union else 0.0


def reference_alignment(candidate: dict | None, truth_shapes: list,
                        image_size: tuple[int, int]) -> dict:
    """Measure candidate overlap while keeping provenance separate from metrics."""
    targets = [shape for shape in truth_shapes
               if isinstance(shape, dict)
               and shape.get("label") in {"车辆", "other_robot"}]
    result = {
        "target_count": len(targets),
        "target_labels": sorted({shape.get("label") for shape in targets}),
        "candidate_present": candidate is not None,
        "candidate_label": candidate.get("label") if candidate else None,
        "same_label_target_count": 0,
        "best_same_label_iou": 0.0,
        "best_any_label_iou": 0.0,
        "best_any_label": None,
    }
    if candidate is None:
        return result
    candidate_points = candidate.get("points") or []
    scored = [(shape, polygon_iou(candidate_points, shape.get("points") or [],
                                  image_size)) for shape in targets]
    same_label = [(shape, score) for shape, score in scored
                  if shape.get("label") == candidate.get("label")]
    result["same_label_target_count"] = len(same_label)
    if same_label:
        result["best_same_label_iou"] = round(max(score for _, score in same_label), 6)
    if scored:
        best_shape, best_score = max(scored, key=lambda item: item[1])
        result["best_any_label_iou"] = round(best_score, 6)
        result["best_any_label"] = best_shape.get("label")
    return result


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
    parser.add_argument("--reference-session", type=Path)
    parser.add_argument("--disable-fast-route", action="store_true")
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
    required_paths = [state_path, media_path, sample_path]
    if args.reference_session:
        required_paths.append(args.reference_session.resolve())
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    state_sha256_before = sha256_file(state_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    config = dict(state.get("config") or {})
    if args.disable_fast_route:
        config = without_fast_route(config)
    labels = list(state.get("labels") or [])
    if (not config.get("llm_url") or not config.get("llm_model")
            or not config.get("sam3_point_url") or "车辆" not in labels):
        raise ValueError("state lacks production model settings or 车辆 label")
    samples = load_samples(sample_path)
    reference_path = args.reference_session.resolve() if args.reference_session else None
    reference_sha256_before = sha256_file(reference_path) if reference_path else None
    reference_session = (load_reference_session(reference_path, media_path)
                         if reference_path else None)
    reference_provenance = (classify_target_provenance(
        reference_session, [sample["frame"] for sample in samples],
        ["车辆", "other_robot"]) if reference_session else None)
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
        candidate = None
        trace_by_id = {}

        def capture(event):
            trace_by_id[event["id"]] = event

        try:
            with workflow.trace_calls(capture):
                candidates, evidence = workflow.visible_vehicle_point_candidate(
                    config, image, next_id=1, allowed_labels=labels)
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
        row["model_calls"] = []
        for event in trace_by_id.values():
            output_record = event.get("output") or {}
            local_vlm = output_record.get("local_vlm") or {}
            row["model_calls"].append({
                "service": event.get("service"),
                "status": event.get("status"),
                "elapsed_ms": event.get("elapsed_ms"),
                "model_route": output_record.get("model_route"),
                "fast_fallback_reason": output_record.get("fast_fallback_reason"),
                "cache_hit": local_vlm.get("cache_hit"),
                "coalesced": local_vlm.get("coalesced"),
                "image_count": local_vlm.get("image_count"),
                "actual_image_token_counts": local_vlm.get(
                    "actual_image_token_counts"),
                "preprocess_seconds": local_vlm.get("preprocess_seconds"),
                "generation_seconds": local_vlm.get("generation_seconds"),
                "total_seconds": local_vlm.get("seconds"),
            })
        row["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        if reference_session is not None:
            with Image.open(io.BytesIO(image)) as source_image:
                image_size = source_image.size
            row["reference_alignment"] = reference_alignment(
                candidate, reference_session.get("truth", {}).get(str(frame), []),
                image_size)
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
    alignments = [row["reference_alignment"] for row in rows
                  if "reference_alignment" in row]
    result = {
        "schema_version": "labelagent.visible_vehicle_shadow_evaluation.v1",
        "read_only": True,
        "assistant_visual_proxy_only": True,
        "candidate_only": True,
        "fast_route_disabled": args.disable_fast_route,
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
    if reference_path:
        result["reference"] = {
            "path": str(reference_path),
            "sha256": reference_sha256_before,
            "modified": sha256_file(reference_path) != reference_sha256_before,
            "provenance": reference_provenance,
            "metric_scope": "recovery_candidate_agreement_only",
        }
        result["summary"].update({
            "reference_target_shape_count": sum(
                item["target_count"] for item in alignments),
            "reference_positive_frame_count": sum(
                item["target_count"] > 0 for item in alignments),
            "no_candidate_on_reference_positive_count": sum(
                item["target_count"] > 0 and not item["candidate_present"]
                for item in alignments),
            "candidate_on_reference_negative_count": sum(
                item["target_count"] == 0 and item["candidate_present"]
                for item in alignments),
            "same_label_iou_ge_0_10_count": sum(
                item["best_same_label_iou"] >= 0.10 for item in alignments),
            "same_label_iou_ge_0_25_count": sum(
                item["best_same_label_iou"] >= 0.25 for item in alignments),
            "same_label_iou_ge_0_50_count": sum(
                item["best_same_label_iou"] >= 0.50 for item in alignments),
            "wrong_label_overlap_ge_0_10_count": sum(
                item["best_any_label_iou"] >= 0.10
                and item["best_any_label"] != item["candidate_label"]
                for item in alignments),
        })
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print("RESULT=" + json.dumps(result["summary"], ensure_ascii=False),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
