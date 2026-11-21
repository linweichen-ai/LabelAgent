"""Read-only camera-aware ego recovery diagnostics.

The evaluator never returns a mergeable shape.  It reports only bounded
geometry and overlap summaries so callers can collect shadow evidence without
turning assistant-authored point templates into annotation truth.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re

import numpy as np
from PIL import Image

from .edit_agent import _rasterize, valid_shape


EGO_CAMERA_SHADOW_SCHEMA_VERSION = "labelagent.ego_camera_shadow.v1"
EGO_CAMERA_SHADOW_TEMPLATE_VERSION = "sfl_cpd15y_camera_templates.round385.v1"
EGO_CAMERA_SHADOW_SOURCE_SCOPE = "capture_20261103"
EGO_CAMERA_SHADOW_PERSON_OVERLAP_LIMIT = 0.08
EGO_CAMERA_SHADOW_FLOOR_OVERLAP_LIMIT = 0.20

# Development-only primary templates from the read-only Round385 study.
# They are deliberately camera-specific and never imply cross-factory truth.
EGO_CAMERA_NORMALIZED_TEMPLATES = {
    "camera0": {
        "name": "camera0-ego-broad",
        "points": ((0.20, 0.90, 1), (0.45, 0.94, 1), (0.60, 0.88, 1),
                   (0.78, 0.88, 0), (0.50, 0.68, 0), (0.30, 0.75, 0)),
    },
    "camera1": {
        "name": "camera1-ego-broad",
        "points": ((0.5625, 0.8333, 1), (0.6250, 0.7227, 1),
                   (0.7240, 0.8920, 1), (0.6410, 0.9700, 1),
                   (0.4270, 0.7290, 0), (0.4580, 0.5340, 0),
                   (0.8230, 0.6050, 0)),
    },
    "camera2": {
        "name": "camera2-ego-broad",
        "points": ((0.30, 0.92, 1), (0.55, 0.90, 1), (0.72, 0.94, 1),
                   (0.20, 0.75, 0), (0.50, 0.65, 0), (0.85, 0.75, 0)),
    },
    "camera3": {
        "name": "camera3-ego-broad",
        "points": ((0.36, 0.78, 1), (0.50, 0.88, 1), (0.64, 0.78, 1),
                   (0.50, 0.96, 1), (0.50, 0.55, 0), (0.20, 0.75, 0),
                   (0.82, 0.75, 0)),
    },
}


def camera_template_key(topic: str) -> str | None:
    if not isinstance(topic, str):
        return None
    lowered = topic.strip().lower()
    if lowered in EGO_CAMERA_NORMALIZED_TEMPLATES:
        return lowered
    # Accept only an entire camera token or path segment.  Names such as
    # ``front_camera_1`` are different camera identities and must not inherit
    # an SFL camera1 template merely because they contain a similar suffix.
    matched = re.search(r"(?:^|/)camera[_-]?([0-3])(?:/|$)", lowered)
    return f"camera{matched.group(1)}" if matched else None


def _base_status(enabled: bool) -> dict:
    return {
        "schema_version": EGO_CAMERA_SHADOW_SCHEMA_VERSION,
        "template_version": EGO_CAMERA_SHADOW_TEMPLATE_VERSION,
        "template_source_scope": EGO_CAMERA_SHADOW_SOURCE_SCOPE,
        "enabled": enabled,
        "triggered": False,
        "status": "disabled" if not enabled else "not_triggered",
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "production_deployed": False,
        "eligible_for_candidate_merge": False,
        "state_written": False,
    }


def materialize_camera_template(topic: str, width: int, height: int) -> tuple[str, list[dict]]:
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError("ego shadow image dimensions must be positive integers")
    template = EGO_CAMERA_NORMALIZED_TEMPLATES.get(camera_template_key(topic))
    if template is None:
        raise ValueError("ego shadow has no template for this camera")
    points = [{
        "x": round(x_ratio * (width - 1)),
        "y": round(y_ratio * (height - 1)),
        "label": label,
    } for x_ratio, y_ratio, label in template["points"]]
    return template["name"], points


def _eligible_discovery(discovery) -> tuple[bool, float | None]:
    assessments = discovery.get("assessments") if isinstance(discovery, dict) else None
    item = next((entry for entry in assessments or []
                 if isinstance(entry, dict) and entry.get("label") == "ego"), None)
    confidence = item.get("confidence") if isinstance(item, dict) else None
    eligible = (isinstance(item, dict) and item.get("evidence") == "present"
                and type(confidence) in (int, float)
                and not isinstance(confidence, bool) and math.isfinite(confidence)
                and confidence >= 0.90)
    return eligible, float(confidence) if eligible else None


def _maximum_overlap(mask, shapes, label, width, height) -> float:
    area = int(mask.sum())
    if area <= 0:
        return 0.0
    union = np.zeros((height, width), dtype=bool)
    for item in shapes:
        if not isinstance(item, dict) or item.get("label") != label:
            continue
        if not valid_shape(item, width, height):
            continue
        union |= _rasterize(item, width, height).astype(bool)
    return float(np.logical_and(mask, union).sum()) / area


def evaluate_camera_ego_shadow(config, image: bytes, topic: str, discovery,
                               existing_shapes, request_points,
                               *, source_identity=None, check_cancelled=None,
                               cancelled_error_type=()) -> dict:
    """Evaluate one camera template without returning or mutating a candidate."""
    enabled = isinstance(config, dict) and config.get("ego_camera_shadow_evaluator") is True
    status = _base_status(enabled)
    if not enabled:
        return status
    if source_identity is not None:
        normalized_source = (source_identity.strip().lower()
                             if isinstance(source_identity, str) else "")
        source_parts = [part for part in re.split(r"[\\/]+", normalized_source)
                        if part]
        if EGO_CAMERA_SHADOW_SOURCE_SCOPE not in source_parts:
            identity_digest = (hashlib.sha256(
                normalized_source.encode("utf-8")).hexdigest()
                if normalized_source else None)
            return status | {
                "status": "source_scope_mismatch",
                "source_identity_sha256": identity_digest,
            }
    if not isinstance(existing_shapes, list):
        return status | {"status": "invalid_existing_candidates"}
    if any(isinstance(shape, dict) and shape.get("label") == "ego"
           for shape in existing_shapes):
        return status | {"status": "existing_candidate"}
    eligible, confidence = _eligible_discovery(discovery)
    if not eligible:
        return status | {"status": "discovery_not_eligible"}
    camera_key = camera_template_key(topic)
    if camera_key is None:
        return status | {"status": "unsupported_camera", "camera": topic}
    if check_cancelled is not None:
        check_cancelled()
    try:
        with Image.open(io.BytesIO(image)) as decoded:
            width, height = decoded.size
        template_name, points = materialize_camera_template(camera_key, width, height)
        response = request_points(points)
        if not isinstance(response, dict):
            raise ValueError("ego shadow point response must be an object")
        parts = response.get("parts")
        score = response.get("score")
        warnings = response.get("warnings", [])
        if warnings is None:
            warnings = []
        if (not isinstance(parts, list) or not parts or len(parts) > 200
                or isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError("ego shadow point response is incomplete")
        if (not isinstance(warnings, list) or len(warnings) > 20
                or any(not isinstance(item, str) or not item.strip()
                       or len(item) > 500 for item in warnings)):
            raise ValueError("ego shadow point response has malformed warnings")
        shadow_shape = {"id": 1, "label": "ego", "parts": parts, "score": score}
        if not valid_shape(shadow_shape, width, height):
            raise ValueError("ego shadow point response contains invalid geometry")
        mask = _rasterize(shadow_shape, width, height).astype(bool)
        rows, columns = mask.nonzero()
        if not len(rows):
            raise ValueError("ego shadow mask is empty")
        area = int(mask.sum())
        area_ratio = area / (width * height)
        positive_points = [point for point in points if point["label"] == 1]
        negative_points = [point for point in points if point["label"] == 0]
        positive_points_covered = sum(
            bool(mask[point["y"], point["x"]]) for point in positive_points)
        negative_points_violated = sum(
            bool(mask[point["y"], point["x"]]) for point in negative_points)
        blocked_prompt = (positive_points_covered != len(positive_points)
                          or negative_points_violated > 0)
        bbox = [int(columns.min()), int(rows.min()),
                int(columns.max()) + 1, int(rows.max()) + 1]
        person_overlap = _maximum_overlap(mask, existing_shapes, "person", width, height)
        floor_overlap = _maximum_overlap(mask, existing_shapes, "floor", width, height)
        blocked_overlap = (person_overlap > EGO_CAMERA_SHADOW_PERSON_OVERLAP_LIMIT
                           or floor_overlap > EGO_CAMERA_SHADOW_FLOOR_OVERLAP_LIMIT)
        blocked_warning = bool(warnings)
        coverage_band = ("camera1_low_coverage" if camera_key == "camera1" and area_ratio < 0.06
                         else "camera1_broad_coverage" if camera_key == "camera1"
                         else "camera_specific_observation")
        guard_status = ("blocked_prompt_constraint" if blocked_prompt
                        else "blocked_sam_warning" if blocked_warning
                        else "blocked_person_floor_overlap" if blocked_overlap
                        else "human_review_required")
        response_sha256 = hashlib.sha256(json.dumps(
            response, ensure_ascii=False, sort_keys=True,
            separators=(",", ":")).encode("utf-8")).hexdigest()
        return status | {
            "triggered": True,
            "status": "shadow_observed",
            "camera": camera_key,
            "camera_topic": topic,
            "template_name": template_name,
            "point_count": len(points),
            "discovery_confidence": confidence,
            "geometry_observed": True,
            "candidate_available": (
                not blocked_prompt and not blocked_warning and not blocked_overlap),
            "score": round(float(score), 6),
            "geometry": {
                "bbox": bbox,
                "area_pixels": area,
                "area_ratio": round(area_ratio, 6),
                "touches_bottom": bbox[3] >= height - 1,
            },
            "coverage_band": coverage_band,
            "person_overlap_ratio": round(person_overlap, 6),
            "floor_overlap_ratio": round(floor_overlap, 6),
            "guard_status": guard_status,
            "positive_point_count": len(positive_points),
            "positive_points_covered": positive_points_covered,
            "negative_point_count": len(negative_points),
            "negative_points_violated": negative_points_violated,
            "sam_warning_count": len(warnings),
            "sam_warning_sha256": (hashlib.sha256(json.dumps(
                warnings, ensure_ascii=False, sort_keys=True,
                separators=(",", ":")).encode("utf-8")).hexdigest()
                if warnings else None),
            "response_sha256": response_sha256,
            "human_review_required": True,
        }
    except Exception as error:
        if cancelled_error_type and isinstance(error, cancelled_error_type):
            raise
        return status | {
            "triggered": True,
            "status": "failed",
            "camera": topic,
            "error": str(error)[:500],
            "candidate_available": False,
            "human_review_required": True,
        }
