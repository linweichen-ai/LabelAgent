"""Pure helpers used by the deployed FastAPI SAM multi-prompt service."""

from __future__ import annotations

from time import perf_counter

import cv2
import numpy as np


def polygons(mask):
    binary = np.asarray(mask.squeeze(), dtype=np.uint8)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP,
                                            cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return [], 0
    parts, discarded = [], 0
    for index, node in enumerate(hierarchy[0]):
        if node[3] != -1 or cv2.contourArea(contours[index]) < 9:
            discarded += 1
            continue
        contour = cv2.approxPolyDP(
            contours[index],
            max(0.5, cv2.arcLength(contours[index], True) * 0.001), True)
        points = [[int(x), int(y)] for [[x, y]] in contour]
        holes, child = [], node[2]
        while child != -1:
            hole = cv2.approxPolyDP(
                contours[child],
                max(0.5, cv2.arcLength(contours[child], True) * 0.001), True)
            vertices = [[int(x), int(y)] for [[x, y]] in hole]
            if len(vertices) >= 3:
                holes.append(vertices)
            child = hierarchy[0][child][0]
        if len(points) >= 3:
            parts.append({"points": points, "holes": holes})
    return parts, discarded


def validate_prompt_map(value):
    if not isinstance(value, dict) or not value or len(value) > 16:
        raise ValueError("prompts must be a non-empty object with at most 16 entries")
    result = {}
    for label, prompt in value.items():
        if not isinstance(label, str) or not label.strip() or len(label) > 64:
            raise ValueError("prompt labels must be non-empty strings of at most 64 characters")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 256:
            raise ValueError("prompt text must be non-empty and at most 256 characters")
        result[label.strip()] = prompt.strip()
    if len(result) != len(value):
        raise ValueError("prompt labels must remain unique after trimming")
    return result


def prediction_payload(masks, scores, confidence_threshold):
    results, discarded = [], 0
    for mask, score in zip(masks, scores):
        parts, count = polygons(mask)
        discarded += count
        if parts:
            results.append({"parts": parts, "score": float(score)})
    return {
        "results": results,
        "diagnostics": {
            "confidence_threshold": confidence_threshold,
            "model_mask_count": len(masks),
            "returned_count": len(results),
            "scores": [float(score) for score in scores],
            "min_contour_area": 9,
            "discarded_contours": discarded,
            "empty_reason": (
                "model returned no mask after confidence filtering"
                if not len(masks) else "mask has no usable contour"
                if not results else None),
        },
    }


def evaluate_prompt_states(processor, state, prompts, clock=perf_counter):
    outputs, prompt_seconds = {}, {}
    for label, prompt in prompts.items():
        started = clock()
        state = processor.set_text_prompt(prompt, state)
        masks = state["masks"].detach().cpu().numpy()
        scores = state["scores"].detach().float().cpu().numpy()
        outputs[label] = prediction_payload(
            masks, scores, processor.confidence_threshold)
        prompt_seconds[label] = clock() - started
    return state, outputs, prompt_seconds
