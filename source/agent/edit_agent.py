"""Validated OpenCV operations on local polygon annotations."""
import cv2
import numpy as np
import math


def _rasterize(shape, width, height):
    canvas = np.zeros((height, width), dtype=np.uint8)
    for part in shape.get("parts") or [shape]:
        points = np.asarray(part.get("points", []), dtype=np.int32)
        if len(points) >= 3:
            cv2.fillPoly(canvas, [points], 1)
        for hole in part.get("holes", []):
            vertices = np.asarray(hole, dtype=np.int32)
            if len(vertices) >= 3:
                cv2.fillPoly(canvas, [vertices], 0)
    return canvas


def _shape_from_mask(mask, label, shape_id):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return None
    parts = []
    for index, node in enumerate(hierarchy[0]):
        if node[3] != -1 or cv2.contourArea(contours[index]) < 9:
            continue
        points = contours[index].reshape(-1, 2).tolist()
        holes = []
        child = node[2]
        while child != -1:
            # OpenCV's child contour follows foreground pixels around the
            # hole. Store the actual background contour so rasterizing an
            # edited shape does not enlarge every untouched hole by a pixel.
            region = np.zeros_like(mask)
            cv2.drawContours(region, contours, child, 1, cv2.FILLED)
            region[mask != 0] = 0
            inner, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hole = max(inner, key=cv2.contourArea).reshape(-1, 2).tolist() if inner else []
            if len(hole) >= 3:
                holes.append(hole)
            child = hierarchy[0][child][0]
        if len(points) >= 3:
            parts.append({"points": points, "holes": holes})
    if not parts:
        return None
    return {"id": shape_id, "label": label, "points": parts[0]["points"],
            "holes": parts[0]["holes"], "parts": parts}


def _hole_regions(shape, width, height):
    """Return internal raster holes with stable IDs for one candidate shape."""
    mask = _rasterize(shape, width, height)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    holes = []
    for index, node in enumerate(hierarchy[0]):
        if node[3] == -1:
            continue
        area = abs(float(cv2.contourArea(contours[index])))
        if area < 9:
            continue
        x, y, w, h = cv2.boundingRect(contours[index])
        region = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(region, contours, index, 1, thickness=cv2.FILLED)
        region[mask != 0] = 0
        moments = cv2.moments(contours[index])
        cx = (moments["m10"] / moments["m00"]) if moments["m00"] else x + w / 2
        cy = (moments["m01"] / moments["m00"]) if moments["m00"] else y + h / 2
        holes.append({"shape_id": shape["id"], "kind": "hole", "area": int(region.sum()),
                      "bbox": [x, y, x + w - 1, y + h - 1],
                      "centroid": [round(cx, 1), round(cy, 1)], "mask": region})
    holes.sort(key=lambda item: (item["bbox"][1], item["bbox"][0], -item["area"]))
    for item in holes:
        x1, y1, x2, y2 = item["bbox"]
        item["id"] = f"{shape['id']}:hole:{x1}:{y1}:{x2}:{y2}"
    return holes


def _open_gap_regions(shape, width, height):
    """Return narrow mask gaps that are connected to another excluded region.

    SAM often leaves a thin strip outside a floor mask without creating a
    contour hole.  A small closing operation proposes those strips; it does
    not decide whether they are ground.  The caller must have the model
    inspect the source image before applying a polygon to one of these
    regions.
    """
    mask = _rasterize(shape, width, height)
    kernel_size = max(7, min(61, int(min(width, height) * .02) | 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    gap = ((closed != 0) & (mask == 0)).astype(np.uint8)
    # Enclosed holes have their own stable IDs and must not be reported twice.
    for hole in _hole_regions(shape, width, height):
        gap[hole["mask"] != 0] = 0
    count, components, stats, _ = cv2.connectedComponentsWithStats(gap, connectivity=8)
    regions = []
    for index in range(1, count):
        x, y, w, h, area = [int(value) for value in stats[index]]
        # Tiny morphology noise is not actionable.  Cap the size so a large
        # object/background opening cannot become a blanket fill proposal.
        if area < max(9, width * height * .00001) or w >= width - 2 or h >= height - 2:
            continue
        region = np.asarray(components == index, dtype=np.uint8)
        moments = cv2.moments(region, binaryImage=True)
        cx = (moments["m10"] / moments["m00"]) if moments["m00"] else x + w / 2
        cy = (moments["m01"] / moments["m00"]) if moments["m00"] else y + h / 2
        regions.append({"shape_id": shape["id"], "area": area,
                        "bbox": [x, y, x + w - 1, y + h - 1],
                        "centroid": [round(cx, 1), round(cy, 1)],
                        "mask": region, "kind": "open_gap"})
    regions.sort(key=lambda item: (item["bbox"][1], item["bbox"][0], -item["area"]))
    for item in regions:
        x1, y1, x2, y2 = item["bbox"]
        item["id"] = f"{shape['id']}:gap:{x1}:{y1}:{x2}:{y2}"
    return regions


def _candidate_gap_regions(shape, width, height):
    """Return enclosed holes and open narrow gaps for one candidate."""
    return _hole_regions(shape, width, height) + _open_gap_regions(shape, width, height)


def _valid_region(points, width, height):
    if not isinstance(points, list) or not 3 <= len(points) <= 2000:
        return False
    return all(isinstance(point, list) and len(point) == 2 and
               all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in point) and
               0 <= point[0] < width and 0 <= point[1] < height for point in points)


def valid_shape(shape, width, height):
    """Validate an annotation returned by SAM or an LLM tool."""
    if not isinstance(shape, dict) or type(shape.get("id")) is not int:
        return False
    if not isinstance(shape.get("label"), str) or not shape["label"].strip():
        return False
    parts = shape.get("parts") or [shape]
    if not isinstance(parts, list) or not parts or len(parts) > 200:
        return False
    vertices = 0
    for part in parts:
        if not isinstance(part, dict) or not _valid_region(part.get("points"), width, height):
            return False
        vertices += len(part["points"])
        holes = part.get("holes", [])
        if not isinstance(holes, list) or len(holes) > 200:
            return False
        if any(not _valid_region(hole, width, height) for hole in holes):
            return False
        vertices += sum(len(hole) for hole in holes)
    return vertices <= 10000


def _next_id(current, next_id):
    used = [shape_id for shape_id in current if type(shape_id) is int]
    return max([next_id or 1, *(value + 1 for value in used)])


def apply_operations(shapes, operations, labels, width, height, next_id=None, gap_regions=None):
    """Apply a small allowlist of LLM actions; return edited shapes and errors."""
    current = {shape["id"]: shape for shape in shapes}
    # All polygons in a decision address the same observation. Keep those
    # regions stable even if an earlier polygon splits or shrinks a gap.
    if gap_regions is None:
        gap_regions = [region for shape in shapes
                       for region in _candidate_gap_regions(shape, width, height)]
    applied, review_queue = [], []
    for item in operations:
        if not isinstance(item, dict):
            review_queue.append({"reason": "操作格式无效", "operation": item})
            continue
        kind = item.get("operation")
        ids = item.get("shape_ids")
        if kind in ("merge",):
            if (not isinstance(ids, list) or len(ids) < 2 or len(set(ids)) != len(ids) or
                    any(type(shape_id) is not int or shape_id not in current for shape_id in ids)):
                review_queue.append({**item, "reason": "合并操作需要至少两个有效 shape ID"})
                continue
            label = item.get("label")
            if label not in labels:
                review_queue.append({**item, "reason": "目标类别不存在"})
                continue
            mask = np.zeros((height, width), dtype=np.uint8)
            for shape_id in ids:
                mask = cv2.bitwise_or(mask, _rasterize(current[shape_id], width, height))
            merged = _shape_from_mask(mask, label, _next_id(current, next_id))
            if merged is None:
                review_queue.append({**item, "reason": "合并后区域为空或过小"})
                continue
            for shape_id in ids:
                del current[shape_id]
            current[merged["id"]] = merged
            if next_id is not None:
                next_id = merged["id"] + 1
            applied.append({"operation": kind, "shape_ids": ids, "result_ids": [merged["id"]]})
            continue
        if kind == "split":
            if (not isinstance(ids, list) or len(ids) != 1 or type(ids[0]) is not int or
                    ids[0] not in current):
                review_queue.append({**item, "reason": "拆分操作需要一个有效 shape ID"})
                continue
            polygon = item.get("polygon")
            if not _valid_region(polygon, width, height):
                review_queue.append({**item, "reason": "拆分区域坐标无效"})
                continue
            source_id = ids[0]
            source = current[source_id]
            source_mask = _rasterize(source, width, height)
            region = np.zeros_like(source_mask)
            cv2.fillPoly(region, [np.asarray(polygon, dtype=np.int32)], 1)
            inside = cv2.bitwise_and(source_mask, region)
            outside = source_mask.copy()
            outside[region != 0] = 0
            first = _shape_from_mask(inside, source["label"], source_id)
            second = _shape_from_mask(outside, source["label"], _next_id(current, next_id))
            if first is None or second is None:
                review_queue.append({**item, "reason": "拆分必须产生两个有效区域"})
                continue
            del current[source_id]
            current[first["id"]] = first
            current[second["id"]] = second
            if next_id is not None:
                next_id = second["id"] + 1
            applied.append({"operation": kind, "shape_ids": ids,
                            "result_ids": [first["id"], second["id"]]})
            continue
        if kind == "fill_gap":
            if (not isinstance(ids, list) or len(ids) != 1 or type(ids[0]) is not int or
                    ids[0] not in current or not isinstance(item.get("hole_id"), str)):
                review_queue.append({**item, "reason": "填洞操作需要一个有效 shape ID 和 hole_id"})
                continue
            polygon = item.get("polygon")
            if not _valid_region(polygon, width, height):
                review_queue.append({**item, "reason": "填洞区域坐标无效"})
                continue
            hole = next((candidate for candidate in gap_regions
                         if candidate["id"] == item["hole_id"] and candidate["shape_id"] == ids[0]), None)
            if hole is None:
                review_queue.append({**item, "reason": "指定的空洞不存在或已在此前操作中改变"})
                continue
            region = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(region, [np.asarray(polygon, dtype=np.int32)], 1)
            # A fill_gap can only add pixels that are inside this exact
            # model-reviewed hole or open-gap proposal.
            patch = cv2.bitwise_and(region, hole["mask"])
            mask = _rasterize(current[ids[0]], width, height)
            patch[mask != 0] = 0
            if not np.any(patch):
                review_queue.append({**item, "reason": "多边形与指定空洞没有交集"})
                continue
            mask = cv2.bitwise_or(mask, patch)
            updated = _shape_from_mask(mask, current[ids[0]]["label"], ids[0])
            if updated is None:
                review_queue.append({**item, "reason": "填洞后区域无效"})
                continue
            current[ids[0]] = updated
            applied.append({"operation": kind, "shape_id": ids[0],
                            "hole_id": item["hole_id"],
                            "filled_pixels": int(np.count_nonzero(patch))})
            continue
        if kind not in ("remove", "reclassify", "add_region", "erase_region") or not isinstance(ids, list) or len(ids) != 1 or type(ids[0]) is not int or ids[0] not in current:
            review_queue.append({**item, "reason": "操作类型或 shape ID 无效"})
            continue
        shape_id = ids[0]
        if kind == "remove":
            del current[shape_id]
        elif kind == "reclassify":
            if item.get("label") not in labels:
                review_queue.append({**item, "reason": "目标类别不存在"})
                continue
            current[shape_id] = {**current[shape_id], "label": item["label"]}
        else:
            polygon = item.get("polygon")
            if not _valid_region(polygon, width, height):
                review_queue.append({**item, "reason": "区域坐标无效"})
                continue
            mask = _rasterize(current[shape_id], width, height)
            region = np.zeros_like(mask)
            cv2.fillPoly(region, [np.asarray(polygon, dtype=np.int32)], 1)
            if kind == "add_region":
                mask = cv2.bitwise_or(mask, region)
            else:
                mask[region != 0] = 0
            updated = _shape_from_mask(mask, current[shape_id]["label"], shape_id)
            if updated is None:
                review_queue.append({**item, "reason": "操作后区域为空或过小"})
                continue
            current[shape_id] = updated
        applied.append({"operation": kind, "shape_id": shape_id})
    return list(current.values()), applied, review_queue
