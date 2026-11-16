"""Evaluate image-only temporal transport from model-localized ramp anchors."""
import argparse
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from source.agent.mcap_video import Video
from source.agent.workflow import sam_request, sam_text
from tools.evaluate_slope_sam import metrics, sam_shapes, union_mask
from tools.temporal_candidate_utils import estimate_bidirectional_flow, prepare_flow_warp


FLOW_CONFIG = {
    "width": 320, "height": 240, "pyr_scale": 0.5, "levels": 4,
    "winsize": 31, "iterations": 5, "poly_n": 7, "poly_sigma": 1.5,
    "gaussian": True,
}


def parse_anchors(value):
    anchors = {}
    for item in value.split(","):
        frame, x, y = (int(part) for part in item.split(":"))
        if frame in anchors:
            raise ValueError(f"Duplicate anchor frame {frame}")
        anchors[frame] = (x, y)
    if not anchors:
        raise ValueError("At least one anchor is required")
    return anchors


def overlay(image_bytes, prediction, truth):
    image = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB")).copy()
    prediction, truth = prediction.astype(bool), truth.astype(bool)
    image[truth] = (0.55 * image[truth] + 0.45 * np.array([30, 255, 30])).astype(np.uint8)
    image[prediction] = (0.45 * image[prediction] + 0.55 * np.array([255, 30, 210])).astype(np.uint8)
    image[prediction & truth] = (255, 220, 30)
    return Image.fromarray(image)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcap", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--sessions", required=True)
    parser.add_argument("--anchors", required=True, help="frame:x:y comma-separated")
    parser.add_argument("--max-frame-gap", type=int, default=3)
    parser.add_argument("--adjacent-chain", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.max_frame_gap < 1:
        raise ValueError("--max-frame-gap must be positive")
    anchors = parse_anchors(args.anchors)
    mcap = str(Path(args.mcap).resolve())
    state = json.loads(Path(args.state).read_text(encoding="utf-8"))
    sessions = json.loads(Path(args.sessions).read_text(encoding="utf-8"))
    session = sessions[mcap]
    topic, truth_by_frame = session["topic"], session["truth"]
    frames = sorted(int(frame) for frame, shapes in truth_by_frame.items()
                    if any(item.get("label") == "slope" for item in shapes))
    if not set(anchors) <= set(frames):
        raise ValueError("Every anchor must be one of the evaluation frames")

    video = Video(mcap)
    image_bytes = {frame: video.frame(topic, frame) for frame in frames}
    rgb = {frame: np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
           for frame, data in image_bytes.items()}
    def frame_rgb(frame):
        if frame not in rgb:
            data = video.frame(topic, frame)
            image_bytes[frame] = data
            rgb[frame] = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
        return rgb[frame]
    sizes = {(image.shape[1], image.shape[0]) for image in rgb.values()}
    if len(sizes) != 1:
        raise ValueError("Evaluation frames have different sizes")
    width, height = sizes.pop()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    anchor_masks = {}
    anchor_scores = {}
    for frame, (x, y) in anchors.items():
        if not 0 <= x < width or not 0 <= y < height:
            raise ValueError(f"Anchor point for frame {frame} lies outside the image")
        result = sam_request(state.get("config") or {}, "/segment_points", image_bytes[frame],
                             points=[{"x": x, "y": y, "label": 1}])
        anchor_masks[frame] = union_mask(sam_shapes([result]), width, height).astype(np.uint8)
        anchor_scores[frame] = result.get("score")

    floor_masks = {}
    for frame in frames:
        floor_masks[frame] = union_mask(
            sam_shapes(sam_text(image_bytes[frame], "floor", state.get("config") or {})),
            width, height).astype(np.uint8)

    rows = []
    for frame in frames:
        truth_shapes = [item for item in truth_by_frame[str(frame)] if item.get("label") == "slope"]
        truth = union_mask(truth_shapes, width, height).astype(np.uint8)
        source = min(anchors, key=lambda candidate: (abs(candidate - frame), candidate))
        gap = abs(source - frame)
        accepted = gap <= args.max_frame_gap
        prediction = np.zeros_like(truth)
        consistent_fraction = None
        area_ratio = None
        raw_transport_metrics = None
        flow_steps = []
        reason = None
        if accepted and source == frame:
            prediction = anchor_masks[source].copy()
            consistent_fraction = 1.0
            area_ratio = 1.0
            reason = "model_anchor"
        elif accepted:
            current_frame = source
            transported = anchor_masks[source].copy()
            targets = ([*range(source + (1 if frame > source else -1),
                              frame + (1 if frame > source else -1),
                              1 if frame > source else -1)]
                       if args.adjacent_chain else [frame])
            step_consistencies = []
            for next_frame in targets:
                forward, backward = estimate_bidirectional_flow(
                    frame_rgb(current_frame), frame_rgb(next_frame), FLOW_CONFIG)
                warp = prepare_flow_warp(forward, backward, width, height,
                                         error_threshold=2.0)
                transported = cv2.remap(transported, warp["map_x"], warp["map_y"],
                                        cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=0)
                step_area = int(transported.sum())
                step_consistency = (float(np.count_nonzero((transported != 0)
                                                            & warp["consistent"])) / step_area
                                    if step_area else 0.0)
                step_consistencies.append(step_consistency)
                flow_steps.append({"from": current_frame, "to": next_frame,
                                   "consistent_fraction": step_consistency,
                                   "pixels": step_area})
                current_frame = next_frame
            source_area = int(anchor_masks[source].sum())
            target_area = int(transported.sum())
            area_ratio = target_area / source_area if source_area else 0.0
            consistent_fraction = min(step_consistencies) if step_consistencies else 0.0
            raw_transport_metrics = metrics(transported, truth)
            accepted = (0.45 <= area_ratio <= 1.8 and consistent_fraction >= 0.65)
            if accepted:
                prediction = transported
                reason = "accepted_bidirectional_flow"
            else:
                reason = "rejected_flow_guard"
        else:
            reason = "no_anchor_within_gap"
        row = {"frame": frame, "source_frame": source, "frame_gap": gap,
               "accepted": accepted, "reason": reason,
               "source_sam_score": anchor_scores[source],
               "flow_consistent_fraction": consistent_fraction, "area_ratio": area_ratio,
               "flow_steps": flow_steps,
               "raw_transport_metrics": raw_transport_metrics,
               **metrics(prediction, truth)}
        traversable_truth = union_mask(
            [item for item in truth_by_frame[str(frame)]
             if item.get("label") in {"floor", "slope"}], width, height).astype(np.uint8)
        traversable_prediction = (floor_masks[frame] | prediction).astype(np.uint8)
        slope_pixels = int(truth.sum())
        row["traversable_metrics"] = metrics(traversable_prediction, traversable_truth)
        row["traversable_slope_recall"] = (
            int(np.logical_and(traversable_prediction, truth).sum()) / slope_pixels
            if slope_pixels else None)
        rows.append(row)
        overlay(image_bytes[frame], prediction, truth).save(
            output / f"frame-{frame}-temporal-overlay.jpg", quality=92)

    accepted_rows = [row for row in rows if row["accepted"]]
    summary = {
        "frames": len(rows), "anchors": len(anchors),
        "accepted_predictions": len(accepted_rows),
        "frames_iou_ge_0_5": sum(row["iou"] >= 0.5 for row in rows),
        "mean_iou_all_frames": float(np.mean([row["iou"] for row in rows])),
        "mean_iou_accepted": (float(np.mean([row["iou"] for row in accepted_rows]))
                              if accepted_rows else 0.0),
        "false_positive_accepted": sum(row["accepted"] and row["intersection_pixels"] == 0
                                       for row in rows),
        "traversable_mean_iou": float(np.mean(
            [row["traversable_metrics"]["iou"] for row in rows])),
        "traversable_mean_precision": float(np.mean(
            [row["traversable_metrics"]["precision"] for row in rows])),
        "traversable_mean_recall": float(np.mean(
            [row["traversable_metrics"]["recall"] for row in rows])),
        "traversable_mean_slope_recall": float(np.mean(
            [row["traversable_slope_recall"] for row in rows])),
    }
    report = {
        "mcap": mcap, "topic": topic, "anchors": anchors,
        "max_frame_gap": args.max_frame_gap, "adjacent_chain": args.adjacent_chain,
        "flow_config": FLOW_CONFIG,
        "summary": summary, "rows": rows,
        "evidence_note": ("Predictions used only model-localized anchors, SAM and image optical flow. "
                          "Human slope truth was read only after prediction for scoring/overlays."),
    }
    (output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "summary": summary, "rows": rows},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
