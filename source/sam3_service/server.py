"""HTTP service for SAM 3 image segmentation."""
import os

from flask import Flask, jsonify, request

from .point_segment import model_loaded, segment

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024


@app.get("/health")
def health():
    return jsonify(status="ok", model_loaded=model_loaded())


@app.post("/segment_points")
def segment_points():
    body = request.get_json(silent=True) or {}
    image = body.get("image")
    points = body.get("points")
    if not isinstance(image, str) or not isinstance(points, list):
        return jsonify(error="image (base64) and points are required"), 400
    try:
        import base64

        image_bytes = base64.b64decode(image, validate=True)
        mask = body.get("mask")
        if mask is not None and not isinstance(mask, str):
            raise ValueError("mask must be a base64 PNG image")
        prior_mask = base64.b64decode(mask, validate=True) if mask is not None else None
        return jsonify(segment(image_bytes, points, prior_mask=prior_mask))
    except (ValueError, TypeError) as error:
        return jsonify(error=str(error)), 400


@app.post("/segment_text")
def segment_text():
    body = request.get_json(silent=True) or {}
    image = body.get("image")
    prompt = body.get("prompt")
    if not isinstance(image, str) or not isinstance(prompt, str) or not prompt.strip():
        return jsonify(error="image (base64) and prompt are required"), 400
    try:
        import base64
        import cv2
        import numpy as np
        import torch
        from contextlib import nullcontext
        from PIL import Image
        import io
        from .point_segment import _model_lock, _predictor

        image_bytes = base64.b64decode(image, validate=True)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        with _model_lock:
            _, processor = _predictor()
            precision = torch.autocast("cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else nullcontext()
            with torch.inference_mode(), precision:
                state = processor.set_image(pil_image)
                state = processor.set_text_prompt(prompt.strip(), state)
                masks = state["masks"].detach().cpu().numpy()
                scores = state["scores"].detach().float().cpu().numpy()
        results = []
        discarded_contours = 0
        for mask, score in zip(masks, scores):
            binary = np.asarray(mask.squeeze(), dtype=np.uint8)
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if hierarchy is None:
                continue
            parts = []
            for i, node in enumerate(hierarchy[0]):
                if node[3] != -1 or cv2.contourArea(contours[i]) < 9:
                    discarded_contours += 1
                    continue
                contour = cv2.approxPolyDP(contours[i], max(0.5, cv2.arcLength(contours[i], True) * 0.001), True)
                points = [[int(x), int(y)] for [[x, y]] in contour]
                holes = []
                child = node[2]
                while child != -1:
                    hole = contours[child]
                    hole = cv2.approxPolyDP(hole, max(0.5, cv2.arcLength(hole, True) * 0.001), True)
                    vertices = [[int(x), int(y)] for [[x, y]] in hole]
                    if len(vertices) >= 3:
                        holes.append(vertices)
                    child = hierarchy[0][child][0]
                if len(points) >= 3:
                    parts.append({"points": points, "holes": holes})
            if parts:
                results.append({"parts": parts, "score": float(score)})
        return jsonify(results=results, diagnostics={
            "confidence_threshold": processor.confidence_threshold,
            "model_mask_count": len(masks), "returned_count": len(results),
            "scores": [float(score) for score in scores],
            "min_contour_area": 9, "discarded_contours": discarded_contours,
            "empty_reason": ("模型在置信度过滤后没有返回 mask" if not len(masks)
                             else "mask 没有可用轮廓" if not results else None)})
    except (ValueError, TypeError) as error:
        return jsonify(error=str(error)), 400


def main():
    host = os.environ.get("SAM3_HOST", "127.0.0.1")
    port = int(os.environ.get("SAM3_PORT", "8766"))
    print(f"SAM 3 service: http://{host}:{port}", flush=True)
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    main()
