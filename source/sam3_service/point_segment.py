"""Point prompted SAM 3 segmentation for the reference frame editor."""
import io
import sys
import threading
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image

SAM3_SOURCE = Path(__file__).resolve().parents[1] / "sam3"
WEIGHTS = SAM3_SOURCE / "weights"
_model = None
_model_lock = threading.Lock()


def model_loaded():
    return _model is not None


def _predictor():
    global _model
    if _model is not None:
        return _model
    weights = sorted(WEIGHTS.glob("*.pt"))
    if not weights:
        raise RuntimeError(f"缺少 SAM 3 权重，请将 sam3.pt 放入 {WEIGHTS}")
    try:
        import torch
        if str(SAM3_SOURCE) not in sys.path:
            sys.path.insert(0, str(SAM3_SOURCE))
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError as error:
        raise RuntimeError(f"SAM 3 推理依赖尚未安装到 autolabel/.venv：{error}") from error
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_image_model(checkpoint_path=str(weights[0]),
        load_from_HF=False, enable_inst_interactivity=True, device=device)
    _model = (model, Sam3Processor(model))
    return _model


def segment(image_bytes, points, prior_mask=None):
    """Return a simplified polygon and quality score for positive/negative clicks."""
    import cv2

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size
    previous = None
    if prior_mask is not None:
        mask_image = Image.open(io.BytesIO(prior_mask)).convert("L")
        if mask_image.size != image.size:
            raise ValueError("输入 mask 尺寸必须与原图一致")
        previous = np.asarray(mask_image) > 127
    if not points or not any(point["label"] == 1 for point in points):
        raise ValueError("至少需要一个目标内的正点")
    if len(points) > 32:
        raise ValueError("最多支持 32 个提示点")
    coords = []
    labels = []
    for point in points:
        x, y, label = point["x"], point["y"], point["label"]
        if label not in (0, 1) or not (0 <= x < width and 0 <= y < height):
            raise ValueError("提示点超出图像范围或类型无效")
        coords.append([x, y])
        labels.append(label)
    with _model_lock:
        model, processor = _predictor()
        import torch
        precision = torch.autocast("cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else nullcontext()
        with torch.inference_mode(), precision:
            state = processor.set_image(image)
            # With several clicks the prompt is no longer ambiguous.  SAM3's
            # single-mask path uses all positive/negative clicks together;
            # asking for multimask candidates can otherwise return a high
            # scoring mask that ignores a corrective negative click.
            mask_prompt = {}
            if previous is not None:
                prompt_encoder = model.inst_interactive_predictor.model.sam_prompt_encoder
                mask_height, mask_width = prompt_encoder.mask_input_size
                dense = np.where(previous, 10.0, -10.0).astype(np.float32)
                dense = cv2.resize(dense, (mask_width, mask_height), interpolation=cv2.INTER_LINEAR)
                mask_prompt["mask_input"] = dense[None, :, :]
            masks, scores, _ = model.predict_inst(state,
                point_coords=np.asarray(coords, dtype=np.float32),
                point_labels=np.asarray(labels, dtype=np.int32),
                multimask_output=previous is None and len(coords) == 1, **mask_prompt)
    # SAM 3 may return several candidates.  Its quality score alone can pick
    # a mask that still covers a user supplied negative point, so rank masks
    # by prompt consistency first and use the model score as the tie breaker.
    masks_array = np.asarray(masks)
    positive = [(int(y), int(x)) for (x, y), label in zip(coords, labels) if label == 1]
    negative = [(int(y), int(x)) for (x, y), label in zip(coords, labels) if label == 0]
    # Check a small neighbourhood as well as the exact click pixel.  This
    # avoids accepting a mask whose boundary merely passes through a negative
    # click because of image scaling/interpolation.
    def covered(binary, y, x, radius=3):
        y0, y1 = max(0, y-radius), min(binary.shape[0], y+radius+1)
        x0, x1 = max(0, x-radius), min(binary.shape[1], x+radius+1)
        return bool(binary[y0:y1, x0:x1].any())
    candidates = []
    for index, candidate in enumerate(masks_array):
        binary = np.asarray(candidate, dtype=bool)
        positive_ok = all(binary[y, x] for y, x in positive)
        negative_ok = all(not covered(binary, y, x) for y, x in negative)
        violations = sum(not binary[y, x] for y, x in positive) + sum(covered(binary, y, x) for y, x in negative)
        candidates.append((positive_ok and negative_ok, -violations, float(scores[index]), index))
    # Exact prompt agreement wins; if the model has no exact candidate, pick
    # the least violating one rather than silently ignoring negative clicks.
    best = max(candidates)[-1]
    mask = np.asarray(masks_array[best], dtype=np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("SAM 3 没有找到目标，请补充提示点")
    outer = [index for index, node in enumerate(hierarchy[0]) if node[3] == -1 and cv2.contourArea(contours[index]) >= 9]
    if not outer:
        raise ValueError("生成区域过小，请补充提示点")
    prompted_outer = [index for index in outer if any(
        mask[y, x] and cv2.pointPolygonTest(contours[index], (x, y), False) >= 0
        for y, x in positive)]
    if previous is not None:
        # Refinement must keep all predicted components, including regions
        # disconnected by occluders that were part of the previous mask.
        pass
    elif prompted_outer:
        outer = prompted_outer
    else:
        outer = [max(outer, key=lambda index: cv2.contourArea(contours[index]))]

    def polygon(contour, simplify=True):
        if simplify:
            contour = cv2.approxPolyDP(contour, max(0.5, cv2.arcLength(contour, True) * 0.001), True)
        return [[int(x), int(y)] for [[x, y]] in contour]

    def extract_parts(simplify=True):
        parts = []
        for index in outer:
            vertices = polygon(contours[index], simplify)
            if len(vertices) < 3:
                continue
            holes = []
            child = hierarchy[0][index][2]
            while child != -1:
                hole = polygon(contours[child], simplify)
                if len(hole) >= 3:
                    holes.append(hole)
                child = hierarchy[0][child][0]
            parts.append({"points": vertices, "holes": holes})
        return parts

    def rasterize(parts):
        rendered = np.zeros(mask.shape, dtype=np.uint8)
        for part in parts:
            cv2.fillPoly(rendered, [np.asarray(part["points"], dtype=np.int32)], 1)
            for hole in part["holes"]:
                cv2.fillPoly(rendered, [np.asarray(hole, dtype=np.int32)], 0)
        return rendered

    parts = extract_parts()
    rendered = rasterize(parts)
    # Simplification may move a boundary across a click. Keep exact contour
    # vertices when that happens; do not blame the model for conversion loss.
    if any(bool(mask[y, x]) != bool(rendered[y, x]) for y, x in positive + negative):
        parts = extract_parts(simplify=False)
        rendered = rasterize(parts)
    if any(rendered[y, x] for y, x in negative):
        raise ValueError("SAM 3 无法排除当前负点区域，请补充或调整提示点")
    warnings = []
    if any(not rendered[y, x] for y, x in positive):
        warnings.append("SAM 3 未覆盖全部正点；已保留候选轮廓，请继续调整提示点")
    return {"points": parts[0]["points"], "holes": parts[0]["holes"],
            "parts": parts, "score": float(scores[best]), "warnings": warnings}
