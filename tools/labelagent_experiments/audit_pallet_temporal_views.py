"""Frozen RGB-only adjacent-frame diagnostics; never generates annotation masks.

Run --freeze with stdlib Python, then --run with the existing image runtime.
Only the import manifest, native RGB photographs, fixed image-only seed masks,
and existing frame76 RGB copies are inputs. No label, point cloud, pose,
calibration, area-difference image, or model service is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/labelagent_point_ids_20261107/temporal_audit"
DATA = Path("D:/Apps/LabelAgent/work/datasets/Capture_20261103")
COHORT = ROOT / "outputs/labelagent_visible_base_20261107/native_inputs/cohort.json"
BASELINE = ROOT / "outputs/labelagent_context_comparison_20261105/inputs"

PARAMETERS = {
    "frames": [74, 76, 78],
    "source_frame": 76,
    "corner_detector": {"maxCorners": 64, "qualityLevel": 0.01,
                        "minDistance": 5, "blockSize": 7,
                        "useHarrisDetector": False},
    "include_original_image_anchor": True,
    "anchor_duplicate_distance_px": 1.0,
    "lk": {"winSize": [31, 31], "maxLevel": 4,
           "criteria_count": 30, "criteria_epsilon": 0.01,
           "minEigThreshold": 0.0001, "flags": 0},
    "valid_correspondence": {"max_forward_backward_error_px": 1.5,
                             "both_status_true": True,
                             "both_endpoints_within_native_photo": True},
    "diagnostic_support": {"min_valid_points": 6,
                           "min_valid_fraction": 0.6,
                           "meaning": "image correspondence only; not object identity"},
    "display": {"crop": "original source_box translated by median valid RGB flow",
                "fallback": "fixed original source_box when no valid RGB flow",
                "translation_used_for_annotation": False,
                "mask_warping": False},
    "cv2_threads": 2,
    "reference_points_read": False,
    "labels_read": False,
    "calibration_or_pose_read": False,
    "area_difference_read": False,
    "model_requests": 0,
    "annotation_writes": 0,
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def freeze():
    if (OUT / "frozen_plan.json").exists():
        raise FileExistsError("Frozen plan already exists; do not silently revise it")
    manifest_path = DATA / "import_manifest.json"
    manifest = read(manifest_path)
    seeds = read(COHORT)
    assert len(seeds) == 8
    protected = {str(COHORT): digest(COHORT),
                 str(manifest_path): digest(manifest_path),
                 str(Path(__file__).resolve()): digest(__file__)}
    sources = []
    for camera in sorted({int(row["case"].split("_")[0][6:]) for row in seeds}):
        for frame in PARAMETERS["frames"]:
            stamp = str(manifest["timestamps_ns"][frame])
            relative = f"cameras/camera_{camera}/{stamp}.png"
            path = (DATA / "source" / relative).resolve()
            assert path.is_relative_to((DATA / "source/cameras").resolve())
            expected = manifest["extracted_sha256"][relative]
            assert digest(path) == expected, (camera, frame, "native source changed")
            protected[str(path)] = expected
            sources.append({"camera": camera, "frame": frame, "path": str(path),
                            "manifest_sha256": expected,
                            "timestamp_role": "file identity only; physical clock not verified"})
        baseline = BASELINE / f"camera{camera}_frame76.png"
        protected[str(baseline)] = digest(baseline)
    frozen_seeds = []
    for seed in seeds:
        mask_path = Path(seed["folder"]) / "seed_mask.png"
        protected[str(mask_path)] = digest(mask_path)
        frozen_seeds.append({"index": seed["index"], "case": seed["case"],
                             "key": seed["key"], "source": seed["source"],
                             "mask_path": str(mask_path),
                             "binding": seed["binding"]})
    plan = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "parameters": PARAMETERS, "seeds": frozen_seeds,
            "native_sources": sources, "protected_sha256": protected,
            "limitations": [
                "The model mask only chooses initial image texture; it is not ground truth.",
                "Forward-backward consistency does not establish physical object identity.",
                "Poor texture, occlusion, blur, or repeated patterns can invalidate tracking.",
                "A translated display crop is not a predicted or accepted mask.",
                "No acceptance threshold, original mask, or Agent round count changes.",
            ]}
    save(OUT / "frozen_plan.json", plan)
    print(json.dumps({"status": "frozen_before_image_tracking", "seeds": 8,
                      "native_photos": len(sources), "plan": str(OUT / "frozen_plan.json")}), flush=True)


def as_point(value):
    if value is None or not all(math.isfinite(float(x)) for x in value):
        return None
    return [float(x) for x in value]


def safe_number(value):
    return float(value) if value is not None and math.isfinite(float(value)) else None


def track(cv2, np, source, target, points):
    options = PARAMETERS["lk"]
    kwargs = {"winSize": tuple(options["winSize"]), "maxLevel": options["maxLevel"],
              "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                           options["criteria_count"], options["criteria_epsilon"]),
              "minEigThreshold": options["minEigThreshold"], "flags": options["flags"]}
    p0 = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    q, sf, ef = cv2.calcOpticalFlowPyrLK(source, target, p0, None, **kwargs)
    count = len(p0)
    q = np.full_like(p0, np.nan) if q is None else q
    sf = np.zeros(count, dtype=np.uint8) if sf is None else sf.ravel()
    ef = np.full(count, np.nan) if ef is None else ef.ravel()
    back = np.full_like(p0, np.nan)
    sb = np.zeros(count, dtype=np.uint8)
    eb = np.full(count, np.nan)
    h, w = target.shape
    inside = np.isfinite(q).all(axis=(1, 2)) & (q[:, 0, 0] >= 0) & (q[:, 0, 0] < w)
    inside &= (q[:, 0, 1] >= 0) & (q[:, 0, 1] < h)
    eligible = np.where((sf == 1) & inside)[0]
    if len(eligible):
        b, bs, be = cv2.calcOpticalFlowPyrLK(target, source, q[eligible], None, **kwargs)
        if b is not None:
            back[eligible] = b
            sb[eligible] = bs.ravel()
            eb[eligible] = be.ravel()
    fb = np.linalg.norm(back[:, 0] - p0[:, 0], axis=1)
    sh, sw = source.shape
    back_inside = np.isfinite(back).all(axis=(1, 2)) & (back[:, 0, 0] >= 0) & (back[:, 0, 0] < sw)
    back_inside &= (back[:, 0, 1] >= 0) & (back[:, 0, 1] < sh)
    valid = ((sf == 1) & (sb == 1) & inside & back_inside
             & (fb <= PARAMETERS["valid_correspondence"]["max_forward_backward_error_px"]))
    rows = []
    for index in range(count):
        rows.append({"point_id": index, "kind": "original_anchor" if index == 0 else "RGB_corner",
                     "source_xy": as_point(p0[index, 0]), "target_xy": as_point(q[index, 0]),
                     "back_to_source_xy": as_point(back[index, 0]),
                     "forward_status": bool(sf[index]), "backward_status": bool(sb[index]),
                     "forward_patch_error": safe_number(ef[index]),
                     "backward_patch_error": safe_number(eb[index]),
                     "forward_backward_error_px": safe_number(fb[index]),
                     "valid_image_correspondence": bool(valid[index])})
    good = int(valid.sum())
    median_delta = (np.median(q[valid, 0] - p0[valid, 0], axis=0).tolist()
                    if good else None)
    supported = (good >= PARAMETERS["diagnostic_support"]["min_valid_points"]
                 and good / count >= PARAMETERS["diagnostic_support"]["min_valid_fraction"])
    finite_fb = fb[np.isfinite(fb)]
    summary = {"initial_points": count, "initial_corners": count - 1,
               "valid_points": good, "valid_fraction": good / count,
               "anchor_valid": bool(valid[0]),
               "all_finite_fb_median_px": safe_number(np.median(finite_fb)) if len(finite_fb) else None,
               "valid_fb_median_px": safe_number(np.median(fb[valid])) if good else None,
               "median_valid_translation_xy": median_delta,
               "diagnostic_image_correspondence_available": supported,
               "object_identity_confirmed": False,
               "more_complete_visible_base_confirmed": False,
               "points": rows}
    return summary


def crop_box(original, delta, size):
    left, top, right, bottom = map(int, original)
    width, height = right - left, bottom - top
    dx, dy = delta if delta is not None else (0, 0)
    requested = [left + round(dx), top + round(dy)]
    actual = [max(0, min(size[0] - width, requested[0])),
              max(0, min(size[1] - height, requested[1]))]
    return [*actual, actual[0] + width, actual[1] + height], actual != requested


def make_panels(Image, ImageDraw, ImageFont, ImageOps, np, folder, seed, photos, points, results):
    title_font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 25)
    small_font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 20)
    point_font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 11)
    board = Image.new("RGB", (1920, 1080), "#152436")
    draw = ImageDraw.Draw(board)
    draw.text((22, 18), f"种子{seed['index']:02d} {seed['case']} / 同相机原图与双向RGB跟踪", font=title_font, fill="white")
    draw.text((22, 57), "上排无标记原照片裁块；下排实际跟踪点。编号0是旧图像anchor；绿为FB≤1.5px，红为未通过。", font=small_font, fill="#cfe1ec")
    display_rows = []
    for column, frame in enumerate(PARAMETERS["frames"]):
        result = results.get(frame)
        delta = None if frame == 76 else result["median_valid_translation_xy"]
        box, clipped_display = crop_box(seed["binding"]["source_box"], delta, photos[frame].size)
        crop = photos[frame].crop(tuple(box))
        crop.save(folder / f"frame{frame}_source.png")
        overlay = crop.copy()
        od = ImageDraw.Draw(overlay)
        if frame == 76:
            rows = [{"point_id": j, "xy": xy, "valid": None} for j, xy in enumerate(points)]
        else:
            rows = [{"point_id": r["point_id"], "xy": r["target_xy"],
                     "valid": r["valid_image_correspondence"]} for r in result["points"]]
        for row in rows:
            if row["xy"] is None:
                continue
            x, y = row["xy"][0] - box[0], row["xy"][1] - box[1]
            if not (0 <= x < crop.width and 0 <= y < crop.height):
                continue
            color = "#ffa31a" if row["valid"] is None else ("#00ff8c" if row["valid"] else "#ff5d5d")
            radius = 4 if row["point_id"] == 0 else 2
            od.ellipse((x-radius, y-radius, x+radius, y+radius), outline=color, width=2)
            od.text((x+3, y-11), str(row["point_id"]), font=point_font, fill=color,
                    stroke_width=1, stroke_fill="black")
        overlay.save(folder / f"frame{frame}_tracks.png")
        x0 = 22 + column * 636
        label = f"帧 {frame} / " + ("起始原图" if frame == 76 else f"有效 {result['valid_points']}/{result['initial_points']}")
        draw.text((x0, 105), label, font=title_font, fill="white")
        for row_number, panel in enumerate([crop, overlay]):
            tile = ImageOps.contain(panel, (604, 365), Image.Resampling.LANCZOS)
            board.paste(tile, (x0 + (604-tile.width)//2, 145 + row_number*397 + (365-tile.height)//2))
        if result:
            fb = result["valid_fb_median_px"]
            text = f"有效点FB中位数 {fb:.3f}px" if fb is not None else "没有通过双向检查的点"
            draw.text((x0, 936), text, font=small_font, fill="#d9e6eb")
        display_rows.append({"frame": frame, "source_crop_box": box,
                             "display_crop_repositioned_to_image_edge": clipped_display,
                             "delta_source": "median_valid_RGB_flow" if frame != 76 and delta is not None else "original_box",
                             "mask_warped": False})
    draw.text((22, 987), "图像对应只用于查看邻帧；不是身份确认、完整基座证明或新标注。未读取3D参考点/标签/面积差/外参。", font=small_font, fill="#e0cf9d")
    draw.text((22, 1023), "纹理不足、遮挡、重复结构或模糊均可能使跟踪失效；旧mask仅界定起始角点区域，所有原始像素保留。", font=small_font, fill="#cfe1ec")
    board.save(folder / "three_frame_comparison.jpg", quality=94)
    save(folder / "display_crop_binding.json", display_rows)


def run():
    if (OUT / "completed.json").exists():
        raise FileExistsError("Completed audit is immutable")
    plan = read(OUT / "frozen_plan.json")
    assert plan["parameters"] == PARAMETERS
    for path, expected in plan["protected_sha256"].items():
        assert digest(path) == expected, path
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    cv2.setNumThreads(PARAMETERS["cv2_threads"])
    images = {}
    source_pixel_equality = []
    for row in plan["native_sources"]:
        image = Image.open(row["path"]).convert("RGB")
        images[(row["camera"], row["frame"])] = image
        if row["frame"] == 76:
            baseline = Image.open(BASELINE / f"camera{row['camera']}_frame76.png").convert("RGB")
            equal = np.array_equal(np.asarray(image), np.asarray(baseline))
            source_pixel_equality.append({"camera": row["camera"], "native76_matches_current_RGB": equal})
            assert equal, "Baseline pixels differ from immutable native frame76"
    results = []
    for seed in plan["seeds"]:
        folder = OUT / f"seed{seed['index']:02d}"
        folder.mkdir()
        camera = int(seed["case"].split("_")[0][6:])
        photos = {frame: images[(camera, frame)] for frame in PARAMETERS["frames"]}
        grays = {frame: cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY) for frame, image in photos.items()}
        mask = (np.asarray(Image.open(seed["mask_path"]).convert("L")) > 0).astype(np.uint8)
        assert mask.shape == grays[76].shape
        anchor = list(map(float, seed["binding"]["anchor_source"]))
        ax, ay = map(int, anchor)
        assert mask[ay, ax] == 1
        points = [anchor]
        corners = cv2.goodFeaturesToTrack(grays[76], mask=mask*255, **PARAMETERS["corner_detector"])
        if corners is not None:
            for point in corners[:, 0]:
                if np.linalg.norm(point-np.asarray(anchor)) <= PARAMETERS["anchor_duplicate_distance_px"]:
                    continue
                assert mask[int(round(point[1])), int(round(point[0]))] == 1
                points.append([float(x) for x in point])
        tracking = {frame: track(cv2, np, grays[76], grays[frame], points) for frame in [74, 78]}
        row = {"index": seed["index"], "case": seed["case"], "seed_key": seed["key"],
               "model_mask_role": "initial RGB feature region only; no truth implied",
               "seed_mask_pixels": int(mask.sum()), "initial_points": points,
               "tracking": {str(frame): result for frame, result in tracking.items()}}
        save(folder / "tracking.json", row)
        make_panels(Image, ImageDraw, ImageFont, ImageOps, np, folder, seed, photos, points, tracking)
        results.append({"index": seed["index"], "case": seed["case"], "seed_key": seed["key"],
                        "initial_points": len(points),
                        "targets": [{"frame": frame, **{k:v for k,v in result.items() if k != "points"}}
                                    for frame, result in tracking.items()]})
        save(OUT / "progress.json", results)
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    image_rows = []
    for path in sorted(OUT.rglob("*")):
        if path.suffix.lower() in [".png", ".jpg"]:
            with Image.open(path) as check:
                size = list(check.size)
                check.verify()
            image_rows.append({"path": path.relative_to(OUT).as_posix(),
                               "sha256": digest(path), "size": size})
    protected_rows = [{"path": path, "sha256_before": expected,
                       "sha256_after": digest(path), "unchanged": digest(path) == expected}
                      for path, expected in plan["protected_sha256"].items()]
    assert all(row["unchanged"] for row in protected_rows)
    save(OUT / "protected_inputs_readback.json", {"passed": True,
          "rows": protected_rows, "frame76_pixel_binding": source_pixel_equality})
    save(OUT / "image_readback.json", {"passed": True, "count": len(image_rows), "rows": image_rows})
    completed = {"status": "complete_RGB_diagnostic_only", "seeds": len(results),
                 "pairs": 2*len(results), "pictures": len(image_rows),
                 "diagnostic_supported_pairs": sum(t["diagnostic_image_correspondence_available"] for r in results for t in r["targets"]),
                 "anchor_valid_pairs": sum(t["anchor_valid"] for r in results for t in r["targets"]),
                 "object_identity_confirmations": 0, "complete_base_confirmations": 0,
                 "new_masks": 0, "model_requests": 0, "annotation_writes": 0,
                 "reference_points_labels_areas_calibration_read": False,
                 "original_inputs_unchanged": True,
                 "runtime": {"opencv": cv2.__version__, "numpy": np.__version__},
                 "frozen_plan_sha256": digest(OUT / "frozen_plan.json"),
                 "rows": results, "limitations": plan["limitations"]}
    save(OUT / "completed.json", completed)
    print(json.dumps({k:v for k,v in completed.items() if k != "rows"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    freeze() if args.freeze else run()
