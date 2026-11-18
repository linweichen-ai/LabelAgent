"""Evaluate text-prompted SAM slope masks against saved targets, read-only.

Saved annotations are not assumed to be human truth.  The report carries an
explicit provenance audit, and ``--require-human-truth`` fails before any SAM
request unless every evaluated frame has a current, committed review receipt.
"""
import argparse
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from source.agent.edit_agent import _rasterize
from source.agent.evaluation_provenance import classify_target_provenance
from source.agent.mcap_video import Video
from source.agent.workflow import filter_slope_candidate_geometry, sam_request, sam_text


DEFAULT_PROMPTS = [
    "ramp",
    "inclined ramp surface",
    "warehouse loading ramp",
    "sloped traversable surface",
]


def union_mask(shapes, width, height):
    mask = np.zeros((height, width), dtype=np.uint8)
    for shape in shapes:
        mask |= _rasterize(shape, width, height)
    return mask.astype(bool)


def metrics(prediction, truth):
    intersection = int(np.logical_and(prediction, truth).sum())
    predicted = int(prediction.sum())
    expected = int(truth.sum())
    union = predicted + expected - intersection
    return {
        "iou": intersection / union if union else 1.0,
        "precision": intersection / predicted if predicted else 0.0,
        "recall": intersection / expected if expected else 0.0,
        "intersection_pixels": intersection,
        "prediction_pixels": predicted,
        "truth_pixels": expected,
    }


def sam_shapes(results):
    shapes = []
    for index, result in enumerate(results, 1):
        parts = result.get("parts") or []
        if not parts:
            continue
        shapes.append({
            "id": index,
            "label": "slope",
            "points": parts[0].get("points", []),
            "holes": parts[0].get("holes", []),
            "parts": parts,
            "score": result.get("score"),
        })
    return shapes


def oracle_point_sets(truth):
    """Build localization-oracle clicks from truth to isolate SAM mask quality."""
    binary = truth.astype(np.uint8)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    positives = []
    work = distance.copy()
    for _ in range(3):
        _, maximum, _, location = cv2.minMaxLoc(work)
        if maximum <= 0:
            break
        x, y = location
        positives.append({"x": int(x), "y": int(y), "label": 1})
        cv2.circle(work, (x, y), max(8, round(maximum * 1.5)), 0, -1)
    ys, xs = np.where(binary)
    x1, x2, y1, y2 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    margin = max(8, round(max(x2 - x1, y2 - y1) * 0.08))
    height, width = binary.shape
    negatives = []
    for x, y in ((cx, y1 - margin), (cx, y2 + margin),
                 (x1 - margin, cy), (x2 + margin, cy)):
        x, y = min(width - 1, max(0, x)), min(height - 1, max(0, y))
        if not binary[y, x]:
            negatives.append({"x": int(x), "y": int(y), "label": 0})
    return {"one_positive": positives[:1],
            "three_positive_four_negative": positives + negatives}


def render_overlay(image_bytes, truth, prediction, destination):
    image = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"), dtype=np.float32)
    output = image.copy()
    output[prediction] = output[prediction] * 0.55 + np.array([0, 220, 255]) * 0.45
    truth_edge = cv2.morphologyEx(truth.astype(np.uint8), cv2.MORPH_GRADIENT,
                                  np.ones((5, 5), np.uint8)).astype(bool)
    pred_edge = cv2.morphologyEx(prediction.astype(np.uint8), cv2.MORPH_GRADIENT,
                                 np.ones((3, 3), np.uint8)).astype(bool)
    output[pred_edge] = (255, 40, 200)
    output[truth_edge] = (40, 255, 40)
    Image.fromarray(np.clip(output, 0, 255).astype(np.uint8)).save(destination, quality=92)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcap", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--sessions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    parser.add_argument("--frame-selector-labels", nargs="+",
                        default=["slope", "斜坡", "ramp"])
    parser.add_argument("--truth-labels", nargs="+", default=["slope", "斜坡", "ramp"])
    parser.add_argument("--crop-fallback", action="store_true")
    parser.add_argument(
        "--require-human-truth", action="store_true",
        help="Fail before model calls unless every selected frame has a current review receipt.")
    args = parser.parse_args()

    mcap = str(Path(args.mcap).resolve())
    state = json.loads(Path(args.state).read_text(encoding="utf-8"))
    sessions = json.loads(Path(args.sessions).read_text(encoding="utf-8"))
    session = sessions.get(mcap)
    if not isinstance(session, dict):
        raise ValueError(f"No saved session for {mcap}")
    topic = session.get("topic")
    truth_by_frame = session.get("truth") or {}
    frames = sorted(int(frame) for frame, shapes in truth_by_frame.items()
                    if any(item.get("label") in set(args.frame_selector_labels)
                           for item in shapes))
    if not frames:
        raise ValueError("No slope evaluation-target frames found")
    target_provenance = classify_target_provenance(session, frames, args.truth_labels)
    if args.require_human_truth and not target_provenance["human_truth_verified"]:
        raise ValueError(
            "Human truth is required but target provenance is "
            f"{target_provenance['kind']}; unverified frames: "
            f"{target_provenance['unverified_frames']}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    video = Video(mcap)
    rows = []
    oracle_rows = []
    frame_cache = {}
    prediction_cache = {}
    candidate_cache = {}
    for frame in frames:
        image_bytes = video.frame(topic, frame)
        frame_cache[frame] = image_bytes
        picture = Image.open(io.BytesIO(image_bytes))
        width, height = picture.size
        truth_shapes = [item for item in truth_by_frame[str(frame)]
                        if item.get("label") in set(args.truth_labels)]
        truth_mask = union_mask(truth_shapes, width, height)
        for prompt in args.prompts:
            try:
                candidates = sam_shapes(sam_text(image_bytes, prompt, state.get("config") or {}))
                prediction = union_mask(candidates, width, height)
                prediction_cache[(frame, prompt)] = prediction
                candidate_cache[(frame, prompt)] = candidates
                candidate_metrics = [
                    {"candidate_index": index, "score": candidate.get("score"),
                     **metrics(union_mask([candidate], width, height), truth_mask)}
                    for index, candidate in enumerate(candidates)
                ]
                row = {"frame": frame, "prompt": prompt, "candidate_count": len(candidates),
                       "scores": [item.get("score") for item in candidates],
                       "candidate_metrics": candidate_metrics,
                       **metrics(prediction, truth_mask)}
            except Exception as error:
                row = {"frame": frame, "prompt": prompt, "candidate_count": 0,
                       "scores": [], "candidate_metrics": [], "error": str(error)[:500], "iou": 0.0,
                       "precision": 0.0, "recall": 0.0, "intersection_pixels": 0,
                       "prediction_pixels": 0, "truth_pixels": int(truth_mask.sum())}
            rows.append(row)
        for point_profile, points in oracle_point_sets(truth_mask).items():
            try:
                result = sam_request(state.get("config") or {}, "/segment_points",
                                     image_bytes, points=points)
                candidates = sam_shapes([result])
                prediction = union_mask(candidates, width, height)
                row = {"frame": frame, "point_profile": point_profile,
                       "points": points, "candidate_count": len(candidates),
                       "score": result.get("score"), **metrics(prediction, truth_mask)}
            except Exception as error:
                row = {"frame": frame, "point_profile": point_profile, "points": points,
                       "candidate_count": 0, "error": str(error)[:500], "iou": 0.0,
                       "precision": 0.0, "recall": 0.0, "intersection_pixels": 0,
                       "prediction_pixels": 0, "truth_pixels": int(truth_mask.sum())}
            oracle_rows.append(row)

    if "floor" in args.prompts and "traversable surface" in args.prompts:
        for row in rows:
            if row["prompt"] != "traversable surface":
                continue
            frame = row["frame"]
            image_bytes = frame_cache[frame]
            width, height = Image.open(io.BytesIO(image_bytes)).size
            floor_prediction = prediction_cache.get(
                (frame, "floor"), np.zeros((height, width), dtype=bool))
            distance = cv2.distanceTransform((~floor_prediction).astype(np.uint8),
                                             cv2.DIST_L2, 5)
            for details, candidate in zip(
                    row.get("candidate_metrics") or [],
                    candidate_cache.get((frame, "traversable surface"), [])):
                mask = union_mask([candidate], width, height)
                ys, xs = np.where(mask)
                if not len(xs):
                    continue
                details.update({
                    "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                    "centroid": [float(xs.mean()), float(ys.mean())],
                    "floor_overlap_fraction": float(np.logical_and(mask, floor_prediction).sum()
                                                    / mask.sum()),
                    "minimum_distance_to_floor": float(distance[mask].min()),
                    "median_distance_to_floor": float(np.median(distance[mask])),
                    "fraction_within_32px_of_floor": float((distance[mask] <= 32).mean()),
                })

    filtered_slope_rows = []
    crop_fallback_rows = []
    if "traversable surface" in args.prompts:
        for frame in frames:
            image_bytes = frame_cache[frame]
            width, height = Image.open(io.BytesIO(image_bytes)).size
            candidates = candidate_cache.get((frame, "traversable surface"), [])
            retained, suppressed = filter_slope_candidate_geometry(candidates, height)
            prediction = union_mask(retained, width, height)
            slope_shapes = [item for item in truth_by_frame[str(frame)]
                            if item.get("label") in {"slope", "斜坡", "ramp"}]
            slope_truth = union_mask(slope_shapes, width, height)
            render_overlay(image_bytes, slope_truth, prediction,
                           output / f"frame-{frame}-filtered-slope.jpg")
            row = {"frame": frame, "retained": len(retained),
                   "suppressed": len(suppressed),
                   "suppression_reasons": [item["reason_code"] for item in suppressed],
                   **metrics(prediction, slope_truth)}
            if "floor" in args.prompts:
                floor_prediction = prediction_cache.get(
                    (frame, "floor"), np.zeros((height, width), dtype=bool))
                traversable_prediction = floor_prediction | prediction
                traversable_truth = union_mask(
                    [item for item in truth_by_frame[str(frame)]
                     if item.get("label") in {"floor", "slope"}], width, height)
                row["traversable_metrics"] = metrics(traversable_prediction,
                                                       traversable_truth)
            filtered_slope_rows.append(row)
            if args.crop_fallback and not retained:
                source = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                crop_prediction = np.zeros((height, width), dtype=bool)
                crop_audit = []
                bounds = ((0.0, 0.0, 0.55, 0.55), (0.45, 0.0, 1.0, 0.55),
                          (0.0, 0.45, 0.55, 1.0), (0.45, 0.45, 1.0, 1.0))
                for crop_index, (left_f, top_f, right_f, bottom_f) in enumerate(bounds):
                    left, top = round(width * left_f), round(height * top_f)
                    right, bottom = round(width * right_f), round(height * bottom_f)
                    crop = source.crop((left, top, right, bottom))
                    encoded = io.BytesIO()
                    crop.save(encoded, "JPEG", quality=95)
                    crop_candidates = sam_shapes(sam_text(
                        encoded.getvalue(), "traversable surface", state.get("config") or {}))
                    crop_retained, crop_suppressed = filter_slope_candidate_geometry(
                        crop_candidates, crop.height)
                    crop_mask = union_mask(crop_retained, crop.width, crop.height)
                    crop_prediction[top:bottom, left:right] |= crop_mask
                    crop_audit.append({
                        "crop_index": crop_index, "bbox": [left, top, right, bottom],
                        "retained": len(crop_retained), "suppressed": len(crop_suppressed),
                    })
                crop_fallback_rows.append({
                    "frame": frame, "crop_audit": crop_audit,
                    "retained": sum(item["retained"] for item in crop_audit),
                    "suppressed": sum(item["suppressed"] for item in crop_audit),
                    **metrics(crop_prediction, slope_truth),
                })
                if "floor" in args.prompts:
                    floor_prediction = prediction_cache.get(
                        (frame, "floor"), np.zeros((height, width), dtype=bool))
                    traversable_truth = union_mask(
                        [item for item in truth_by_frame[str(frame)]
                         if item.get("label") in {"floor", "slope"}], width, height)
                    crop_fallback_rows[-1]["traversable_metrics"] = metrics(
                        floor_prediction | crop_prediction, traversable_truth)
                render_overlay(image_bytes, slope_truth, crop_prediction,
                               output / f"frame-{frame}-crop-fallback.jpg")
    accepted_filtered = [row for row in filtered_slope_rows if row["retained"]]
    filtered_slope_summary = None
    if filtered_slope_rows:
        filtered_slope_summary = {
            "frames": len(filtered_slope_rows),
            "accepted_frames": len(accepted_filtered),
            "review_required_frames": len(filtered_slope_rows) - len(accepted_filtered),
            "frames_iou_ge_0_5": sum(row["iou"] >= 0.5 for row in filtered_slope_rows),
            "mean_iou_accepted": (float(np.mean([row["iou"] for row in accepted_filtered]))
                                  if accepted_filtered else 0.0),
            "mean_slope_recall_all": float(np.mean(
                [row["recall"] for row in filtered_slope_rows])),
            "false_positive_accepted": sum(row["retained"] and row["intersection_pixels"] == 0
                                           for row in filtered_slope_rows),
            "rows": filtered_slope_rows,
        }
        if "floor" in args.prompts:
            filtered_slope_summary.update({
                "traversable_mean_iou": float(np.mean(
                    [row["traversable_metrics"]["iou"] for row in filtered_slope_rows])),
                "traversable_mean_precision": float(np.mean(
                    [row["traversable_metrics"]["precision"] for row in filtered_slope_rows])),
                "traversable_mean_recall": float(np.mean(
                    [row["traversable_metrics"]["recall"] for row in filtered_slope_rows])),
            })
    crop_fallback_summary = None
    if crop_fallback_rows:
        crop_fallback_summary = {
            "attempted_frames": len(crop_fallback_rows),
            "frames_iou_ge_0_5": sum(row["iou"] >= 0.5 for row in crop_fallback_rows),
            "mean_iou": float(np.mean([row["iou"] for row in crop_fallback_rows])),
            "rows": crop_fallback_rows,
        }
    final_slope_summary = None
    if filtered_slope_rows:
        crop_by_frame = {row["frame"]: row for row in crop_fallback_rows}
        final_rows = []
        for row in filtered_slope_rows:
            selected = row
            route = "full_frame"
            if not row["retained"] and crop_by_frame.get(row["frame"], {}).get("retained"):
                selected = crop_by_frame[row["frame"]]
                route = "overlap_crop_fallback"
            final_rows.append({"frame": row["frame"], "route": route,
                               **{key: selected.get(key) for key in
                                  ("retained", "suppressed", "iou", "precision", "recall",
                                   "intersection_pixels", "prediction_pixels", "truth_pixels",
                                   "traversable_metrics")}})
        accepted = [row for row in final_rows if row["retained"]]
        final_slope_summary = {
            "frames": len(final_rows), "accepted_frames": len(accepted),
            "review_required_frames": len(final_rows) - len(accepted),
            "frames_iou_ge_0_5": sum(row["iou"] >= 0.5 for row in final_rows),
            "mean_iou_accepted": float(np.mean([row["iou"] for row in accepted])),
            "mean_slope_precision_accepted": float(np.mean(
                [row["precision"] for row in accepted])),
            "mean_slope_recall_all": float(np.mean([row["recall"] for row in final_rows])),
            "false_positive_accepted": sum(row["retained"] and row["intersection_pixels"] == 0
                                           for row in final_rows),
            "rows": final_rows,
        }
        if all(row.get("traversable_metrics") for row in final_rows):
            final_slope_summary.update({
                "traversable_mean_iou": float(np.mean(
                    [row["traversable_metrics"]["iou"] for row in final_rows])),
                "traversable_mean_precision": float(np.mean(
                    [row["traversable_metrics"]["precision"] for row in final_rows])),
                "traversable_mean_recall": float(np.mean(
                    [row["traversable_metrics"]["recall"] for row in final_rows])),
            })

    summaries = []
    for prompt in args.prompts:
        selected = [row for row in rows if row["prompt"] == prompt]
        summaries.append({
            "prompt": prompt,
            "frames": len(selected),
            "frames_with_candidate": sum(row["candidate_count"] > 0 for row in selected),
            "frames_iou_ge_0_5": sum(row["iou"] >= 0.5 for row in selected),
            "mean_iou": float(np.mean([row["iou"] for row in selected])),
            "mean_precision": float(np.mean([row["precision"] for row in selected])),
            "mean_recall": float(np.mean([row["recall"] for row in selected])),
        })
    summaries.sort(key=lambda item: (item["mean_iou"], item["mean_recall"]), reverse=True)
    oracle_summaries = []
    for point_profile in ("one_positive", "three_positive_four_negative"):
        selected = [row for row in oracle_rows if row["point_profile"] == point_profile]
        oracle_summaries.append({
            "point_profile": point_profile,
            "frames": len(selected),
            "frames_with_candidate": sum(row["candidate_count"] > 0 for row in selected),
            "frames_iou_ge_0_5": sum(row["iou"] >= 0.5 for row in selected),
            "mean_iou": float(np.mean([row["iou"] for row in selected])),
            "mean_precision": float(np.mean([row["precision"] for row in selected])),
            "mean_recall": float(np.mean([row["recall"] for row in selected])),
        })
    oracle_summaries.sort(key=lambda item: (item["mean_iou"], item["mean_recall"]), reverse=True)
    combined_rows = []
    if len(args.prompts) > 1:
        for frame in frames:
            image_bytes = frame_cache[frame]
            width, height = Image.open(io.BytesIO(image_bytes)).size
            prediction = np.zeros((height, width), dtype=bool)
            for prompt in args.prompts:
                prediction |= prediction_cache.get((frame, prompt), np.zeros_like(prediction))
            truth_shapes = [item for item in truth_by_frame[str(frame)]
                            if item.get("label") in set(args.truth_labels)]
            truth = union_mask(truth_shapes, width, height)
            slope_shapes = [item for item in truth_by_frame[str(frame)]
                            if item.get("label") in {"slope", "斜坡", "ramp"}]
            slope = union_mask(slope_shapes, width, height)
            slope_pixels = int(slope.sum())
            slope_recall = (int(np.logical_and(prediction, slope).sum()) / slope_pixels
                            if slope_pixels else None)
            combined_rows.append({"frame": frame, "slope_recall": slope_recall,
                                  **metrics(prediction, truth)})
            render_overlay(image_bytes, truth, prediction,
                           output / f"frame-{frame}-combined.jpg")
    combined_prompt_union = None
    guided_prompt_unions = []
    if combined_rows:
        slope_recalls = [row["slope_recall"] for row in combined_rows
                         if row["slope_recall"] is not None]
        combined_prompt_union = {
            "prompts": args.prompts, "frames": len(combined_rows),
            "frames_iou_ge_0_5": sum(row["iou"] >= 0.5 for row in combined_rows),
            "mean_iou": float(np.mean([row["iou"] for row in combined_rows])),
            "mean_precision": float(np.mean([row["precision"] for row in combined_rows])),
            "mean_recall": float(np.mean([row["recall"] for row in combined_rows])),
            "mean_slope_recall": float(np.mean(slope_recalls)) if slope_recalls else None,
            "rows": combined_rows,
        }
        if "floor" in args.prompts and "traversable surface" in args.prompts:
            for radius in (8, 16, 32, 48, 64, 96, 128):
                guided_rows = []
                for frame in frames:
                    image_bytes = frame_cache[frame]
                    width, height = Image.open(io.BytesIO(image_bytes)).size
                    floor_prediction = prediction_cache.get(
                        (frame, "floor"), np.zeros((height, width), dtype=bool))
                    auxiliary = prediction_cache.get(
                        (frame, "traversable surface"), np.zeros_like(floor_prediction))
                    distance = cv2.distanceTransform((~floor_prediction).astype(np.uint8),
                                                     cv2.DIST_L2, 5)
                    prediction = floor_prediction | (auxiliary & (distance <= radius))
                    truth_shapes = [item for item in truth_by_frame[str(frame)]
                                    if item.get("label") in set(args.truth_labels)]
                    truth = union_mask(truth_shapes, width, height)
                    slope_shapes = [item for item in truth_by_frame[str(frame)]
                                    if item.get("label") in {"slope", "斜坡", "ramp"}]
                    slope = union_mask(slope_shapes, width, height)
                    slope_pixels = int(slope.sum())
                    guided_rows.append({
                        "frame": frame,
                        "slope_recall": (int(np.logical_and(prediction, slope).sum())
                                         / slope_pixels if slope_pixels else None),
                        **metrics(prediction, truth),
                    })
                slope_recalls = [row["slope_recall"] for row in guided_rows
                                 if row["slope_recall"] is not None]
                guided_prompt_unions.append({
                    "radius_pixels": radius,
                    "mean_iou": float(np.mean([row["iou"] for row in guided_rows])),
                    "mean_precision": float(np.mean([row["precision"] for row in guided_rows])),
                    "mean_recall": float(np.mean([row["recall"] for row in guided_rows])),
                    "mean_slope_recall": float(np.mean(slope_recalls)) if slope_recalls else None,
                    "rows": guided_rows,
                })
            guided_prompt_unions.sort(key=lambda item: item["mean_iou"], reverse=True)
    best_prompt = summaries[0]["prompt"]
    for row in rows:
        if row["prompt"] != best_prompt:
            continue
        frame = row["frame"]
        image_bytes = frame_cache[frame]
        picture = Image.open(io.BytesIO(image_bytes))
        width, height = picture.size
        truth_shapes = [item for item in truth_by_frame[str(frame)]
                        if item.get("label") in set(args.truth_labels)]
        render_overlay(image_bytes, union_mask(truth_shapes, width, height),
                       prediction_cache.get((frame, best_prompt),
                                            np.zeros((height, width), dtype=bool)),
                       output / f"frame-{frame}.jpg")

    report = {"mcap": mcap, "topic": topic, "frames": frames,
              "prompts": args.prompts, "best_prompt_on_this_development_set": best_prompt,
              "frame_selector_labels": args.frame_selector_labels,
              "truth_labels": args.truth_labels,
              "target_provenance": target_provenance,
              "summaries": summaries, "rows": rows,
              "combined_prompt_union": combined_prompt_union,
              "guided_prompt_unions": guided_prompt_unions,
              "filtered_slope_summary": filtered_slope_summary,
              "crop_fallback_summary": crop_fallback_summary,
              "final_slope_summary": final_slope_summary,
              "oracle_point_summaries": oracle_summaries, "oracle_point_rows": oracle_rows,
              "evidence_note": (target_provenance["evidence_note"]
                                + " Prompt selection is development-set evidence, not "
                                  "generalization. Point profiles use target-derived oracle "
                                  "clicks and only isolate segmentation from localization.")}
    (output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "best_prompt": best_prompt,
                      "summaries": summaries, "combined_prompt_union": combined_prompt_union,
                      "guided_prompt_unions": guided_prompt_unions,
                      "target_provenance": target_provenance,
                      "filtered_slope_summary": filtered_slope_summary,
                      "crop_fallback_summary": crop_fallback_summary,
                      "final_slope_summary": final_slope_summary,
                      "oracle_point_summaries": oracle_summaries},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
