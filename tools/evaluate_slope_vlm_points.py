"""Evaluate VLM ramp localization followed by point-prompted SAM, read-only."""
import argparse
import copy
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from source.agent.mcap_video import Video
from source.agent.workflow import chat, sam_request
from tools.evaluate_slope_sam import metrics, sam_shapes, union_mask


GRID_COLUMNS = 16
GRID_ROWS = 12


def response_schema(frames, width, height):
    detection = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "frame": {"type": "integer", "enum": frames},
            "present": {"type": "boolean"},
            "x": {"type": "integer", "minimum": -1, "maximum": width - 1},
            "y": {"type": "integer", "minimum": -1, "maximum": height - 1},
            "reason": {"type": "string", "minLength": 1, "maxLength": 100},
        },
        "required": ["frame", "present", "x", "y", "reason"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {"detections": {"type": "array", "minItems": len(frames),
                                        "maxItems": len(frames), "items": detection}},
        "required": ["detections"],
    }


def grid_schema(frames):
    detection = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "frame": {"type": "integer", "enum": frames},
            "present": {"type": "boolean"},
            "column": {"type": "integer", "minimum": -1, "maximum": GRID_COLUMNS - 1},
            "row": {"type": "integer", "minimum": -1, "maximum": GRID_ROWS - 1},
            "reason": {"type": "string", "minLength": 1, "maxLength": 100},
        },
        "required": ["frame", "present", "column", "row", "reason"],
    }
    return {"type": "object", "additionalProperties": False,
            "properties": {"detections": {"type": "array", "minItems": len(frames),
                                             "maxItems": len(frames), "items": detection}},
            "required": ["detections"]}


def valid_grid_detection(detection, frame):
    if not isinstance(detection, dict) or detection.get("frame") != frame:
        return False
    present = detection.get("present")
    column, row = detection.get("column"), detection.get("row")
    if present is False:
        return column == -1 and row == -1
    return (present is True and type(column) is int and type(row) is int
            and 0 <= column < GRID_COLUMNS and 0 <= row < GRID_ROWS)


def grid_image(image_bytes, top_fraction=1.0, x_start=0.0, x_end=1.0):
    picture = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be within (0, 1]")
    if not 0 <= x_start < x_end <= 1:
        raise ValueError("x crop bounds must satisfy 0 <= start < end <= 1")
    if top_fraction < 1 or x_start > 0 or x_end < 1:
        crop_height = max(1, round(picture.height * top_fraction))
        crop_left = round(picture.width * x_start)
        crop_right = max(crop_left + 1, round(picture.width * x_end))
        picture = picture.crop((crop_left, 0, crop_right, crop_height)).resize(
            (picture.width, picture.height), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(picture)
    cell_width = picture.width / GRID_COLUMNS
    cell_height = picture.height / GRID_ROWS
    for column in range(GRID_COLUMNS + 1):
        x = round(column * cell_width)
        draw.line((x, 0, x, picture.height - 1), fill=(255, 0, 255), width=1)
    for row in range(GRID_ROWS + 1):
        y = round(row * cell_height)
        draw.line((0, y, picture.width - 1, y), fill=(255, 0, 255), width=1)
    for row in range(GRID_ROWS):
        for column in range(GRID_COLUMNS):
            draw.text((round(column * cell_width) + 2, round(row * cell_height) + 1),
                      f"{column},{row}", fill=(255, 255, 255), stroke_width=1,
                      stroke_fill=(0, 0, 0))
    output = io.BytesIO()
    picture.save(output, "JPEG", quality=92)
    return output.getvalue()


def reference_composite(reference_bytes, reference_point, target_grid_bytes):
    reference = Image.open(io.BytesIO(reference_bytes)).convert("RGB")
    target = Image.open(io.BytesIO(target_grid_bytes)).convert("RGB")
    if reference.size != target.size:
        raise ValueError("Reference and target grid images must have equal dimensions")
    draw = ImageDraw.Draw(reference)
    x, y = reference_point
    draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=(40, 255, 40),
                 outline=(0, 0, 0), width=3)
    draw.text((8, 8), "REFERENCE RAMP", fill=(40, 255, 40), stroke_width=2,
              stroke_fill=(0, 0, 0))
    canvas = Image.new("RGB", (reference.width * 2, reference.height))
    canvas.paste(reference, (0, 0))
    canvas.paste(target, (reference.width, 0))
    canvas_draw = ImageDraw.Draw(canvas)
    canvas_draw.text((reference.width + 8, 8), "TARGET GRID", fill=(255, 255, 255),
                     stroke_width=2, stroke_fill=(0, 0, 0))
    output = io.BytesIO()
    canvas.save(output, "JPEG", quality=92)
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcap", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--sessions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--grid-only", action="store_true")
    parser.add_argument("--grid-top-fraction", type=float, default=1.0)
    parser.add_argument("--grid-x-start", type=float, default=0.0)
    parser.add_argument("--grid-x-end", type=float, default=1.0)
    parser.add_argument("--grid-seed-results")
    parser.add_argument("--frames", help="Optional comma-separated evaluation frame IDs")
    parser.add_argument("--reference-frame", type=int)
    parser.add_argument("--reference-x", type=int)
    parser.add_argument("--reference-y", type=int)
    args = parser.parse_args()

    mcap = str(Path(args.mcap).resolve())
    state = json.loads(Path(args.state).read_text(encoding="utf-8"))
    sessions = json.loads(Path(args.sessions).read_text(encoding="utf-8"))
    session = sessions[mcap]
    topic, truth_by_frame = session["topic"], session["truth"]
    frames = sorted(int(frame) for frame, shapes in truth_by_frame.items()
                    if any(item.get("label") == "slope" for item in shapes))
    if args.frames:
        requested = {int(value.strip()) for value in args.frames.split(",") if value.strip()}
        unknown = requested - set(frames)
        if unknown:
            raise ValueError(f"Requested frames have no slope truth: {sorted(unknown)}")
        frames = [frame for frame in frames if frame in requested]
    video = Video(mcap)
    images = {frame: video.frame(topic, frame) for frame in frames}
    sizes = {Image.open(io.BytesIO(data)).size for data in images.values()}
    if len(sizes) != 1:
        raise ValueError("Evaluation frames have different sizes")
    width, height = sizes.pop()
    mapping = ", ".join(f"image {index}=frame {frame}"
                        for index, frame in enumerate(frames, 1))
    instruction = (
        "Inspect each unmodified warehouse camera image and localize a physical ramp. "
        "A ramp is an inclined planar transition surface leading from the floor into a raised "
        "container, dock, platform, or doorway. Color is not reliable evidence: do not detect "
        "an area merely because it is yellow, green, orange, painted, or rectangular. Use the "
        "visible height transition, inclined plane, raised destination, and scene geometry. "
        "For every frame return present and one pixel point clearly inside the ramp surface. "
        "When no ramp is visible, return present=false and x=y=-1. Do not place the point on a "
        "fork, floor marking, pallet, wall, container side, or flat warehouse floor. "
        f"Image mapping: {mapping}. Reasons are one short English sentence."
    )
    ordered = [images[frame] for frame in frames]
    model_config = copy.deepcopy(state.get("config") or {})
    model_config["llm_max_tokens"] = max(1536, int(model_config.get("llm_max_tokens") or 0))
    direct_error = None
    by_frame = {}
    if not args.grid_only:
        result = chat(model_config, instruction, ordered[0], extra_images=tuple(ordered[1:]),
                      response_schema=response_schema(frames, width, height))
        detections = result.get("detections") if isinstance(result, dict) else None
        if not isinstance(detections, list) or len(detections) != len(frames):
            direct_error = "VLM did not return one detection per frame"
        else:
            by_frame = {item.get("frame"): item for item in detections if isinstance(item, dict)}
            if set(by_frame) != set(frames):
                direct_error = "VLM returned duplicate or missing frame IDs"
                by_frame = {}
    if not by_frame:
        by_frame = {frame: {"frame": frame, "present": False, "x": -1, "y": -1,
                            "reason": direct_error or "Direct localization skipped"}
                    for frame in frames}

    if not 0 < args.grid_top_fraction <= 1:
        raise ValueError("--grid-top-fraction must be within (0, 1]")
    if not 0 <= args.grid_x_start < args.grid_x_end <= 1:
        raise ValueError("--grid-x-start/--grid-x-end must satisfy 0 <= start < end <= 1")
    grid_images = {frame: grid_image(images[frame], args.grid_top_fraction,
                                     args.grid_x_start, args.grid_x_end) for frame in frames}
    reference_enabled = args.reference_frame is not None
    if reference_enabled:
        if args.reference_x is None or args.reference_y is None:
            raise ValueError("Reference x/y are required with --reference-frame")
        reference_bytes = video.frame(topic, args.reference_frame)
        reference_size = Image.open(io.BytesIO(reference_bytes)).size
        if reference_size != (width, height):
            raise ValueError("Reference frame size differs from evaluation frames")
        if not 0 <= args.reference_x < width or not 0 <= args.reference_y < height:
            raise ValueError("Reference point lies outside the reference frame")
        grid_images = {frame: reference_composite(
            reference_bytes, (args.reference_x, args.reference_y), grid_images[frame])
            for frame in frames}
    grid_instruction = (
        f"Each warehouse image has a {GRID_COLUMNS} by {GRID_ROWS} coordinate grid. Columns run "
        f"0..{GRID_COLUMNS - 1} left to right; rows run 0..{GRID_ROWS - 1} top to bottom. "
        + (f"The image is an enlarged view of the top {args.grid_top_fraction:.0%} of the original frame. "
           if args.grid_top_fraction < 1 else "") +
        (f"It is also the horizontal {args.grid_x_start:.0%}..{args.grid_x_end:.0%} crop of the original frame. "
         if args.grid_x_start > 0 or args.grid_x_end < 1 else "") +
        ("The left panel is a prior frame with a green point on a model-confirmed ramp. "
         "Localize the corresponding physical ramp only in the right TARGET GRID panel. "
         if reference_enabled else "") +
        "Find a physical inclined ramp using height transition and scene geometry, not its color. "
        "Choose one grid cell whose CENTER lies clearly inside the ramp surface, away from edges. "
        "Do not choose flat floor, a fork, pallet, wall, container side, or painted marking. "
        "If no ramp is visible return present=false and column=row=-1. "
        "Reasons are one short English sentence."
    )
    # Evaluate one frame per request. The deployed VLM may not preserve image-to-frame
    # identity reliably when eight near-identical video frames share one multimodal prompt.
    grid_by_frame = {}
    if args.grid_seed_results:
        seed_report = json.loads(Path(args.grid_seed_results).read_text(encoding="utf-8"))
        if (float(seed_report.get("grid_top_fraction", 1.0)) != args.grid_top_fraction
                or float(seed_report.get("grid_x_start", 0.0)) != args.grid_x_start
                or float(seed_report.get("grid_x_end", 1.0)) != args.grid_x_end):
            raise ValueError("Seed results use different grid crop bounds")
        for item in seed_report.get("grid_rows") or []:
            frame = item.get("frame")
            if frame in frames and valid_grid_detection(item, frame):
                grid_by_frame[frame] = {key: item.get(key) for key in
                                        ("frame", "present", "column", "row", "reason")}
    for frame in frames:
        if frame in grid_by_frame:
            print(json.dumps({"stage": "grid_localization", "frame": frame,
                              "reused": True, "detection": grid_by_frame[frame]},
                             ensure_ascii=False), flush=True)
            continue
        frame_instruction = f"{grid_instruction} This image is frame {frame}."
        grid_result = chat(model_config, frame_instruction, grid_images[frame],
                           response_schema=grid_schema([frame]))
        detections = grid_result.get("detections") if isinstance(grid_result, dict) else None
        if not isinstance(detections, list) or len(detections) != 1:
            raise ValueError(f"VLM did not return exactly one grid detection for frame {frame}")
        detection = detections[0]
        if not valid_grid_detection(detection, frame):
            correction = (f"{frame_instruction} Your prior answer was invalid. present=true requires "
                          f"column 0..{GRID_COLUMNS - 1} and row 0..{GRID_ROWS - 1}; "
                          "present=false requires column=row=-1. Return a corrected answer.")
            grid_result = chat(model_config, correction, grid_images[frame],
                               response_schema=grid_schema([frame]))
            detections = grid_result.get("detections") if isinstance(grid_result, dict) else None
            detection = detections[0] if isinstance(detections, list) and len(detections) == 1 else None
        if not valid_grid_detection(detection, frame):
            raise ValueError(f"VLM returned an invalid grid detection for frame {frame}")
        grid_by_frame[frame] = detection
        print(json.dumps({"stage": "grid_localization", "frame": frame,
                          "detection": detection}, ensure_ascii=False), flush=True)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    grid_rows = []
    for frame in frames:
        image_bytes = images[frame]
        truth_shapes = [item for item in truth_by_frame[str(frame)] if item.get("label") == "slope"]
        truth = union_mask(truth_shapes, width, height)
        detection = by_frame[frame]
        x, y = detection.get("x"), detection.get("y")
        valid_point = (detection.get("present") is True and type(x) is int and type(y) is int
                       and 0 <= x < width and 0 <= y < height)
        point_in_truth = bool(valid_point and truth[y, x])
        error = None
        prediction = np.zeros_like(truth)
        score = None
        if valid_point:
            try:
                sam = sam_request(state.get("config") or {}, "/segment_points", image_bytes,
                                  points=[{"x": x, "y": y, "label": 1}])
                candidates = sam_shapes([sam])
                prediction = union_mask(candidates, width, height)
                score = sam.get("score")
            except Exception as exception:
                error = str(exception)[:500]
        row = {"frame": frame, "present": detection.get("present"), "x": x, "y": y,
               "reason": detection.get("reason"), "valid_point": valid_point,
               "point_in_truth": point_in_truth, "sam_score": score, "error": error,
               **metrics(prediction, truth)}
        rows.append(row)

        picture = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        draw = ImageDraw.Draw(picture)
        if valid_point:
            color = (40, 255, 40) if point_in_truth else (255, 40, 40)
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color, outline=(0, 0, 0), width=2)
        picture.save(output / f"frame-{frame}-vlm-point.jpg", quality=92)

        grid_detection = grid_by_frame[frame]
        column, grid_row = grid_detection.get("column"), grid_detection.get("row")
        valid_cell = (grid_detection.get("present") is True and type(column) is int
                      and type(grid_row) is int and 0 <= column < GRID_COLUMNS
                      and 0 <= grid_row < GRID_ROWS)
        crop_width_fraction = args.grid_x_end - args.grid_x_start
        grid_x = min(width - 1, round((args.grid_x_start +
                                      (column + 0.5) * crop_width_fraction / GRID_COLUMNS)
                                     * width)) if valid_cell else -1
        grid_y = min(height - 1, round((grid_row + 0.5) * height
                                       * args.grid_top_fraction / GRID_ROWS)) if valid_cell else -1
        grid_in_truth = bool(valid_cell and truth[grid_y, grid_x])
        grid_prediction = np.zeros_like(truth)
        grid_error = None
        grid_score = None
        if valid_cell:
            try:
                sam = sam_request(state.get("config") or {}, "/segment_points", image_bytes,
                                  points=[{"x": grid_x, "y": grid_y, "label": 1}])
                grid_prediction = union_mask(sam_shapes([sam]), width, height)
                grid_score = sam.get("score")
            except Exception as exception:
                grid_error = str(exception)[:500]
        grid_rows.append({"frame": frame, "present": grid_detection.get("present"),
                          "column": column, "row": grid_row, "x": grid_x, "y": grid_y,
                          "reason": grid_detection.get("reason"), "valid_cell": valid_cell,
                          "point_in_truth": grid_in_truth, "sam_score": grid_score,
                          "error": grid_error, **metrics(grid_prediction, truth)})
        grid_picture = Image.open(io.BytesIO(grid_images[frame])).convert("RGB")
        if valid_cell:
            grid_draw = ImageDraw.Draw(grid_picture)
            color = (40, 255, 40) if grid_in_truth else (255, 40, 40)
            display_x = min(width - 1, round((grid_x / width - args.grid_x_start)
                                             / crop_width_fraction * width))
            display_y = min(height - 1, round(grid_y / args.grid_top_fraction))
            if reference_enabled:
                display_x += width
            grid_draw.ellipse((display_x - 7, display_y - 7,
                               display_x + 7, display_y + 7),
                              fill=color, outline=(0, 0, 0), width=2)
        grid_picture.save(output / f"frame-{frame}-vlm-grid.jpg", quality=92)

    summary = {
        "frames": len(rows),
        "vlm_present": sum(row["present"] is True for row in rows),
        "point_in_truth": sum(row["point_in_truth"] for row in rows),
        "frames_iou_ge_0_5": sum(row["iou"] >= 0.5 for row in rows),
        "mean_iou": float(np.mean([row["iou"] for row in rows])),
        "mean_precision": float(np.mean([row["precision"] for row in rows])),
        "mean_recall": float(np.mean([row["recall"] for row in rows])),
    }
    grid_summary = {
        "frames": len(grid_rows),
        "vlm_present": sum(row["present"] is True for row in grid_rows),
        "point_in_truth": sum(row["point_in_truth"] for row in grid_rows),
        "frames_iou_ge_0_5": sum(row["iou"] >= 0.5 for row in grid_rows),
        "mean_iou": float(np.mean([row["iou"] for row in grid_rows])),
        "mean_precision": float(np.mean([row["precision"] for row in grid_rows])),
        "mean_recall": float(np.mean([row["recall"] for row in grid_rows])),
    }
    report = {"mcap": mcap, "topic": topic, "frames": frames, "summary": summary,
              "rows": rows, "grid_summary": grid_summary, "grid_rows": grid_rows,
              "direct_error": direct_error, "grid_top_fraction": args.grid_top_fraction,
              "grid_x_start": args.grid_x_start, "grid_x_end": args.grid_x_end,
              "reference_frame": args.reference_frame,
              "reference_point": ([args.reference_x, args.reference_y]
                                    if reference_enabled else None),
              "evidence_note": "VLM saw original images or coordinate-grid overlays only. Human slope truth was read-only and used only after inference for scoring."}
    (output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "summary": summary,
                      "grid_summary": grid_summary, "rows": rows,
                      "grid_rows": grid_rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
