"""Bounded annotation agent coordinating a vision model, SAM 3 and OpenCV."""
import base64
import copy
import hashlib
import io
import json
import math
import os
import queue
import re
import socket
import threading
import time
import urllib.error
import urllib.request
import uuid
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from PIL import Image, ImageDraw
import cv2
import numpy as np
from .edit_agent import apply_operations, valid_shape, _candidate_gap_regions, _valid_region


_trace_callback = ContextVar("model_trace_callback", default=None)
_trace_asset = ContextVar("model_trace_asset", default=None)
_review_call = ContextVar("review_call", default=None)
_cancel_event = ContextVar("workflow_cancel_event", default=None)
_fast_route_lock = threading.Lock()
_fast_route_circuits = {}
_sam_multi_protocol_lock = threading.Lock()
_sam_multi_protocols = {}
_sam_point_group_protocol_lock = threading.Lock()
_sam_point_group_protocols = {}
MAX_REVIEW_ROUNDS = 100
MAX_PROTOCOL_RETRIES = 2
EGO_BOTTOM_POINT_FALLBACK_VERSION = "labelagent.ego_bottom_point_fallback.v1"
VISIBLE_VEHICLE_POINT_RECOVERY_VERSION = (
    "labelagent.visible_vehicle_point_recovery.v1")
VISIBLE_VEHICLE_POINT_MIN_SCORE = 0.55
VISIBLE_VEHICLE_POINT_HIGH_SCORE = 0.75
VISIBLE_VEHICLE_POINT_MAX_AREA_RATIO = 0.03


class ModelOutputError(ValueError):
    """A model response that can be corrected without repeating tool execution."""

    def __init__(self, message, raw_text, finish_reason=None):
        super().__init__(message)
        self.raw_text = raw_text
        self.finish_reason = finish_reason


class ReviewProtocolError(ValueError):
    """The reviewer exhausted correction attempts without a safe tool action."""

    def __init__(self, message, last_error=None, last_result=None):
        super().__init__(message)
        self.last_error = str(last_error) if last_error is not None else None
        self.last_result = copy.deepcopy(last_result)


class RunCancelled(Exception):
    """The user terminated this annotation run."""


class SamHttpError(ValueError):
    """A SAM HTTP failure with a machine-readable status for safe fallback."""

    def __init__(self, status_code, message):
        super().__init__(f"SAM 3 请求失败：{message}")
        self.status_code = status_code
        self.service_message = str(message)


def check_cancelled():
    event = _cancel_event.get()
    if event is not None and event.is_set():
        raise RunCancelled("用户已停止自动标注")


def _wait_cancellable(seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        check_cancelled()
        time.sleep(min(0.1, max(0, deadline - time.monotonic())))
    check_cancelled()


@contextmanager
def cancellation(event):
    token = _cancel_event.set(event)
    try:
        yield
    finally:
        _cancel_event.reset(token)


def _cancellable_call(function, *args, **kwargs):
    """Discard an in-flight model response promptly when this run is stopped."""
    check_cancelled()
    event = _cancel_event.get()
    if event is None:
        return function(*args, **kwargs)
    result = queue.Queue(maxsize=1)
    context = copy_context()

    def worker():
        try:
            result.put((True, context.run(function, *args, **kwargs)))
        except Exception as error:
            result.put((False, error))

    threading.Thread(target=worker, daemon=True).start()
    while True:
        check_cancelled()
        try:
            succeeded, value = result.get(timeout=0.1)
        except queue.Empty:
            continue
        check_cancelled()
        if succeeded:
            return value
        raise value


@contextmanager
def trace_calls(callback, save_image=None):
    token = _trace_callback.set(callback)
    asset_token = _trace_asset.set(save_image)
    try:
        yield
    finally:
        _trace_asset.reset(asset_token)
        _trace_callback.reset(token)


def _image_summary(data):
    with Image.open(io.BytesIO(data)) as picture:
        summary = {"width": picture.width, "height": picture.height,
                   "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    save_image = _trace_asset.get()
    if save_image:
        summary["url"] = save_image(data)
    return summary


def _sam_summary(response):
    if isinstance(response, dict) and isinstance(response.get("prompts"), dict):
        return {
            "prompt_count": len(response["prompts"]),
            "prompts": {label: _sam_summary(payload)
                        for label, payload in response["prompts"].items()},
            "diagnostics": response.get("diagnostics", {}),
        }
    results = (response.get("results", []) if "results" in response else [response]) \
        if isinstance(response, dict) else response
    if not isinstance(results, list):
        return {"response_type": type(results).__name__}
    candidates = []
    for result in results:
        if not isinstance(result, dict):
            candidates.append({"response_type": type(result).__name__})
            continue
        parts = result.get("parts") or []
        points = [point for part in parts if isinstance(part, dict)
                  for point in part.get("points", [])]
        item = {"score": result.get("score"), "parts": len(parts),
                "vertices": len(points)}
        if points:
            item["bbox"] = [min(p[0] for p in points), min(p[1] for p in points),
                            max(p[0] for p in points), max(p[1] for p in points)]
        candidates.append(item)
    summary = {"returned_count": len(results), "candidates": candidates[:200]}
    if isinstance(response, dict) and response.get("warnings"):
        summary["warnings"] = response["warnings"]
    if isinstance(response, dict) and "diagnostics" in response:
        summary["diagnostics"] = response["diagnostics"]
    return summary


def _emit_trace(event):
    callback = _trace_callback.get()
    if callback:
        callback(event.copy())


def sam_request(config, endpoint, image, *, timeout_seconds=None, **prompt):
    trace_prompt = dict(prompt)
    if "mask" in trace_prompt:
        trace_prompt["mask"] = _image_summary(base64.b64decode(trace_prompt["mask"]))
    record = {"id": uuid.uuid4().hex, "service": "SAM 3", "status": "running",
              "started_at": time.time(), "input": {"endpoint": endpoint,
              "image": _image_summary(image), **trace_prompt}}
    _emit_trace(record)
    try:
        response = _cancellable_call(
            _sam_request, config, endpoint, image,
            timeout_seconds=timeout_seconds, **prompt)
        record.update(status="succeeded", output=_sam_summary(response))
        if _trace_asset.get():
            try:
                results = response.get("results", [response]) if isinstance(response, dict) else response
                preview_shapes = [{"id": i + 1, "points": item["parts"][0]["points"],
                                   "parts": item["parts"]}
                                  for i, item in enumerate(results)
                                  if isinstance(item, dict) and item.get("parts")]
                record["output"]["segmentation"] = _image_summary(candidate_overlay(image, preview_shapes))
            except Exception as preview_error:
                record["output"]["preview_error"] = str(preview_error)[:200]
        return response
    except Exception as error:
        record.update(status="failed", error=str(error)[:500])
        raise
    finally:
        record["elapsed_ms"] = round((time.time() - record["started_at"]) * 1000)
        _emit_trace(record)


def _sam_request(config, endpoint, image, *, timeout_seconds=None, **prompt):
    point_endpoint = endpoint in {
        "/segment_points", "/segment_points_batch",
        "/segment_point_groups_batch"}
    url = ((config.get("sam3_point_url") if point_endpoint else None)
           or config.get("sam3_url")
           or os.environ.get("SAM3_URL", "http://127.0.0.1:8766"))
    payload = json.dumps({"image": base64.b64encode(image).decode("ascii"), **prompt}).encode()
    request = urllib.request.Request(url.rstrip("/") + endpoint, data=payload,
        headers={"Content-Type": "application/json"})
    if timeout_seconds is None:
        timeout_seconds = 1200
    if (isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or not 0.1 <= timeout_seconds <= 1200):
        raise ValueError("SAM 3 请求超时设置无效")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.load(response)
    except (TimeoutError, socket.timeout) as error:
        raise ValueError(f"SAM 3 请求超时（{endpoint}）") from error
    except urllib.error.HTTPError as error:
        try:
            payload = json.load(error)
            message = payload.get("error") or payload.get("detail")
        except (ValueError, AttributeError):
            message = None
        raise SamHttpError(error.code, message or error.reason) from error
    except urllib.error.URLError as error:
        raise ValueError(f"SAM 3 请求失败：{error.reason}") from error


def sam_text(image, prompt, config):
    return sam_request(config, "/segment_text", image, prompt=prompt).get("results", [])


def _point_group_endpoint_key(config):
    return ((config.get("sam3_point_url") or config.get("sam3_url")
             or os.environ.get("SAM3_URL", "http://127.0.0.1:8766"))
            .rstrip("/"))


def _validate_point_group_response(response, prompt_groups):
    items = response.get("items") if isinstance(response, dict) else None
    diagnostics = response.get("diagnostics") if isinstance(response, dict) else None
    if (not isinstance(items, list) or len(items) != len(prompt_groups)
            or not isinstance(diagnostics, dict)
            or diagnostics.get("image_encoded_once") is not True
            or diagnostics.get("prompt_group_count") != len(prompt_groups)):
        raise ValueError("SAM 3 正负提示组响应未证明单次编码和完整组覆盖")
    for index, (item, points) in enumerate(zip(items, prompt_groups)):
        if (not isinstance(item, dict) or item.get("index") != index
                or item.get("points") != points
                or (("result" in item) == ("error" in item))):
            raise ValueError("SAM 3 正负提示组响应顺序或结果格式无效")
        if "result" in item and not isinstance(item["result"], dict):
            raise ValueError("SAM 3 正负提示组成功结果格式无效")
        if "error" in item and not isinstance(item["error"], str):
            raise ValueError("SAM 3 正负提示组失败结果格式无效")
    return response


def _sam_point_groups_sequential(
        image, prompt_groups, config, timeout_seconds=None):
    """Preserve the batch item contract on a service without the new route."""
    started = time.perf_counter()
    items = []
    for index, points in enumerate(prompt_groups):
        try:
            request_options = {"points": points}
            if timeout_seconds is not None:
                request_options["timeout_seconds"] = timeout_seconds
            result = sam_request(
                config, "/segment_points", image, **request_options)
            items.append({"index": index, "points": points, "result": result})
        except SamHttpError as error:
            if error.status_code != 400:
                raise
            items.append({"index": index, "points": points,
                          "error": error.service_message[:200]})
    return {"items": items, "diagnostics": {
        "image_encoded_once": False,
        "prompt_group_count": len(prompt_groups),
        "successful_count": sum("result" in item for item in items),
        "mode": "sequential_endpoint_fallback",
        "fallback_reason": "point_group_endpoint_not_found",
        "total_seconds": time.perf_counter() - started,
    }}


def sam_point_groups(image, prompt_groups, config, *, timeout_seconds=None):
    """Use one-encoding point groups and fall back only for a missing route."""
    if (not isinstance(prompt_groups, list) or not 1 <= len(prompt_groups) <= 64
            or any(not isinstance(group, list) or not 1 <= len(group) <= 32
                   for group in prompt_groups)):
        raise ValueError("SAM 3 正负提示组须包含 1～64 组、每组 1～32 个点")
    endpoint_key = _point_group_endpoint_key(config)
    with _sam_point_group_protocol_lock:
        protocol = _sam_point_group_protocols.get(endpoint_key)
    if protocol == "sequential":
        return _sam_point_groups_sequential(
            image, prompt_groups, config, timeout_seconds=timeout_seconds)
    try:
        request_options = {"prompt_groups": prompt_groups}
        if timeout_seconds is not None:
            request_options["timeout_seconds"] = timeout_seconds
        response = sam_request(
            config, "/segment_point_groups_batch", image,
            **request_options)
    except SamHttpError as error:
        if error.status_code != 404:
            raise
        with _sam_point_group_protocol_lock:
            _sam_point_group_protocols[endpoint_key] = "sequential"
        return _sam_point_groups_sequential(
            image, prompt_groups, config, timeout_seconds=timeout_seconds)
    validated = _validate_point_group_response(response, prompt_groups)
    with _sam_point_group_protocol_lock:
        _sam_point_group_protocols[endpoint_key] = "batch"
    return validated


def ego_bottom_points(width, height):
    """Return one deterministic, resolution-relative scan of the image bottom."""
    if type(width) is not int or type(height) is not int or width < 2 or height < 2:
        raise ValueError("ego 底部点扫描需要有效图像尺寸")
    x_fractions = (0.094, 0.25, 0.406, 0.562, 0.719, 0.875)
    y_fractions = (0.742, 0.859, 0.957)
    return [{"x": min(width - 1, round(width * x)),
             "y": min(height - 1, round(height * y))}
            for y in y_fractions for x in x_fractions]


def ego_bottom_geometry_profile(bbox, area, score, width, height):
    """Fail closed unless a mask matches one reviewed host-robot geometry."""
    if (not isinstance(bbox, (list, tuple)) or len(bbox) != 4
            or any(type(value) is not int for value in bbox)
            or type(area) is not int or area <= 0
            or type(score) not in (int, float) or isinstance(score, bool)
            or not math.isfinite(score)):
        return None
    x0, y0, x1, y1 = bbox
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        return None
    area_ratio = area / (width * height)
    width_ratio = (x1 - x0) / width
    height_ratio = (y1 - y0) / height
    center_ratio = ((x0 + x1) / 2) / width
    common = score >= 0.70 and y1 >= height - 2
    if common and (
        y0 >= height * 0.70
        and width_ratio >= 0.60
        and 0.12 <= height_ratio <= 0.35
        and 0.08 <= area_ratio <= 0.30
    ):
        return "wide_bottom_chassis"
    if common and (
        y0 >= height * 0.60
        and 0.35 <= width_ratio <= 0.60
        and 0.30 <= height_ratio <= 0.45
        and 0.08 <= area_ratio <= 0.18
        and 0.35 <= center_ratio <= 0.65
    ):
        return "centered_fork_carriage"
    return None


def ego_bottom_point_candidates(config, image, next_id=1):
    """Recover at most one review-only ego proposal from a single SAM feature pass."""
    if type(next_id) is not int or next_id < 1:
        raise ValueError("ego 补救候选 ID 无效")
    with Image.open(io.BytesIO(image)) as decoded:
        width, height = decoded.size
    points = ego_bottom_points(width, height)
    response = sam_request(config, "/segment_points_batch", image, points=points)
    items = response.get("items") if isinstance(response, dict) else None
    diagnostics = response.get("diagnostics") if isinstance(response, dict) else None
    expected_points = {(point["x"], point["y"]) for point in points}
    returned_points = {
        (item.get("point", {}).get("x"), item.get("point", {}).get("y"))
        for item in items or [] if isinstance(item, dict)
    }
    if (not isinstance(items, list) or len(items) != len(points)
            or not isinstance(diagnostics, dict)
            or diagnostics.get("image_encoded_once") is not True
            or diagnostics.get("point_count") != len(points)
            or returned_points != expected_points):
        raise ValueError("SAM 3 ego 批量点扫描未证明单次编码和完整点覆盖")

    from .edit_agent import _rasterize
    guarded = []
    raw_count = 0
    for item in items:
        result = item.get("result")
        if not isinstance(result, dict) or not result.get("parts"):
            continue
        raw_count += 1
        score = result.get("score")
        shape = {
            "id": next_id,
            "label": "ego",
            "points": result["parts"][0].get("points", []),
            "holes": result["parts"][0].get("holes", []),
            "parts": result["parts"],
            "score": score,
            "source": "sam_point_ego_fallback",
            "source_point": copy.deepcopy(item["point"]),
        }
        if not valid_shape(shape, width, height):
            continue
        mask = _rasterize(shape, width, height)
        rows, columns = mask.nonzero()
        if not len(rows):
            continue
        bbox = [int(columns.min()), int(rows.min()),
                int(columns.max()) + 1, int(rows.max()) + 1]
        profile = ego_bottom_geometry_profile(
            bbox, int(mask.sum()), score, width, height)
        if profile is None:
            continue
        shape["ego_fallback_profile"] = profile
        shape["ego_fallback_version"] = EGO_BOTTOM_POINT_FALLBACK_VERSION
        shape["ego_fallback_bbox"] = bbox
        shape["ego_fallback_area"] = int(mask.sum())
        guarded.append(shape)
    selected = max(guarded, key=lambda shape: shape["score"], default=None)
    evidence = {
        "version": EGO_BOTTOM_POINT_FALLBACK_VERSION,
        "point_count": len(points),
        "raw_candidate_count": raw_count,
        "guarded_candidate_count": len(guarded),
        "selected_profile": (selected.get("ego_fallback_profile") if selected else None),
        "selected_score": (selected.get("score") if selected else None),
        "service_diagnostics": copy.deepcopy(diagnostics),
    }
    return ([selected] if selected is not None else []), evidence


def visible_vehicle_point_proposal(config, image, next_id=1):
    """Recover one geometry-guarded proposal before independent class checks."""
    if type(next_id) is not int or next_id < 1:
        raise ValueError("车辆补救候选 ID 无效")
    with Image.open(io.BytesIO(image)) as decoded:
        width, height = decoded.size
    schema = {
        "type": "object",
        "properties": {
            "decision": {"type": "string",
                         "enum": ["located", "ambiguous", "absent"]},
            "entity_type": {"type": "string", "enum": [
                "conventional_vehicle_or_forklift",
                "autonomous_robot_or_agv", "other"]},
            "x": {"type": "integer", "minimum": 0, "maximum": 1000},
            "y": {"type": "integer", "minimum": 0, "maximum": 1000},
            "evidence": {"type": "string", "minLength": 1,
                         "maxLength": 100},
        },
        "required": ["decision", "entity_type", "x", "y", "evidence"],
        "additionalProperties": False,
    }
    instruction = (
        "Locate one conventional industrial vehicle or forklift physically "
        "separate from the camera-carrying ego machine. Ignore every foreground "
        "black fork or chassis part attached to the camera. Search the distant "
        "or background image, especially near walls. A parked, unoccupied, or "
        "partly occluded forklift is still a conventional vehicle; a clear solid "
        "body, cabin, mast, or base is enough even when the whole vehicle is not "
        "visible. For decision='located', choose one point strictly inside a "
        "visible solid vehicle part. The normalized point must have y <= 550; "
        "if only foreground vehicle pixels exist, return decision='ambiguous'. "
        "Never select a pallet, person, floor, wall, rack, window, or gap. "
        "Coordinates are normalized integers 0..1000 from left/top to "
        "right/bottom. If only an autonomous AGV is visible, return "
        "entity_type='autonomous_robot_or_agv' and decision='ambiguous'. "
        "Return compact JSON only."
    )
    located = chat(config, instruction, image, response_schema=schema)
    evidence = {
        "schema_version": VISIBLE_VEHICLE_POINT_RECOVERY_VERSION,
        "locator": copy.deepcopy(located),
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
    }
    if (not isinstance(located, dict)
            or located.get("decision") != "located"
            or located.get("entity_type")
            != "conventional_vehicle_or_forklift"):
        return [], evidence | {"status": "locator_not_eligible"}
    normalized_x, normalized_y = located.get("x"), located.get("y")
    if (type(normalized_x) is not int or type(normalized_y) is not int
            or not 0 <= normalized_x <= 1000
            or not 0 <= normalized_y <= 1000):
        raise ValueError("车辆补救定位点格式无效")
    point = {
        "x": round(normalized_x * (width - 1) / 1000),
        "y": round(normalized_y * (height - 1) / 1000),
    }
    evidence["point"] = copy.deepcopy(point)
    if normalized_y > 550:
        return [], evidence | {"status": "foreground_ego_point_rejected"}
    response = sam_request(
        config, "/segment_points_batch", image, points=[point])
    items = response.get("items") if isinstance(response, dict) else None
    diagnostics = response.get("diagnostics") if isinstance(response, dict) else None
    if (not isinstance(items, list) or len(items) != 1
            or not isinstance(diagnostics, dict)
            or diagnostics.get("image_encoded_once") is not True
            or diagnostics.get("point_count") != 1
            or items[0].get("point") != point):
        raise ValueError("SAM 3 车辆点补救未证明单次编码和完整点覆盖")
    result = items[0].get("result")
    if not isinstance(result, dict) or not result.get("parts"):
        return [], evidence | {"status": "sam_empty"}
    shape = {
        "id": next_id,
        "label": "车辆",
        "points": result["parts"][0].get("points", []),
        "holes": result["parts"][0].get("holes", []),
        "parts": result["parts"],
        "score": result.get("score"),
        "source": "sam_vlm_visible_vehicle_point_recovery",
        "source_point": copy.deepcopy(point),
    }
    if not valid_shape(shape, width, height):
        return [], evidence | {"status": "invalid_shape"}
    from .edit_agent import _rasterize
    mask = _rasterize(shape, width, height)
    rows, columns = mask.nonzero()
    if not len(rows):
        return [], evidence | {"status": "empty_mask"}
    bbox = [int(columns.min()), int(rows.min()),
            int(columns.max()) + 1, int(rows.max()) + 1]
    area = int(len(rows))
    area_ratio = area / (width * height)
    score = result.get("score")
    evidence.update(score=score, bbox=bbox, area=area,
                    area_ratio=area_ratio)
    if (type(score) not in (int, float) or isinstance(score, bool)
            or not math.isfinite(score) or score < VISIBLE_VEHICLE_POINT_MIN_SCORE
            or not 0.0001 <= area_ratio <= VISIBLE_VEHICLE_POINT_MAX_AREA_RATIO
            or bbox[2] - bbox[0] > width * 0.5
            or bbox[3] - bbox[1] > height * 0.5
            or bbox[1] >= height * 0.75):
        return [], evidence | {"status": "geometry_rejected"}
    confidence_band = (
        "high" if score >= VISIBLE_VEHICLE_POINT_HIGH_SCORE
        else "guarded_low")
    evidence["confidence_band"] = confidence_band
    shape.update(candidate_only=True, human_truth=False,
                 approved_for_training=False,
                 recovery_confidence_band=confidence_band,
                 recovery_version=VISIBLE_VEHICLE_POINT_RECOVERY_VERSION)
    return [shape], evidence | {"status": "proposal_generated"}


def visible_vehicle_point_candidate(config, image, next_id=1,
                                    allowed_labels=None):
    """Recover and semantically route one review-only visible machine proposal."""
    recovered, evidence = visible_vehicle_point_proposal(
        config, image, next_id=next_id)
    if not recovered:
        return [], evidence
    shape = recovered[0]
    semantic_guard = classify_visible_vehicle_candidate(
        config, image, shape)
    evidence["semantic_guard"] = semantic_guard
    classification = semantic_guard.get("classification")
    confidence = semantic_guard.get("confidence")
    allowed = set(allowed_labels or ("车辆", "other_robot"))
    if (type(confidence) not in (int, float)
            or isinstance(semantic_guard.get("confidence"), bool)
            or confidence < 0.90):
        return [], evidence | {"status": "semantic_guard_rejected"}
    if classification == "conventional_vehicle" and "车辆" in allowed:
        routed_label = "车辆"
    elif (classification == "autonomous_robot_or_agv"
          and "other_robot" in allowed):
        routed_label = "other_robot"
    else:
        return [], evidence | {"status": "semantic_guard_rejected"}
    shape["label"] = routed_label
    shape["semantic_route"] = classification
    evidence["routed_label"] = routed_label
    return [shape], evidence | {"status": "candidate_generated"}


def visible_vehicle_semantic_contract():
    """Return the shared fail-closed class/completeness review contract."""
    schema = {
        "type": "object",
        "properties": {
            "classification": {"type": "string", "enum": [
                "conventional_vehicle", "autonomous_robot_or_agv",
                "partial_fragment", "ambiguous"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string", "minLength": 1, "maxLength": 180},
        },
        "required": ["classification", "confidence", "reason"],
        "additionalProperties": False,
    }
    instruction = (
        "Decide whether the highlighted candidate is a conventional operated "
        "industrial vehicle or an autonomous mobile robot/AGV. Image 1 is the "
        "full original for context. Image 2 highlights the exact candidate in "
        "cyan. Image 3 keeps only candidate pixels. A conventional forklift "
        "needs visible forklift-truck evidence such as an operator cab/seat, "
        "steering area, counterweight, or a mast-and-fork assembly attached to "
        "a truck chassis. A compact low mobile base carrying a tall upright "
        "orange/black cabinet, SEER branding, lidar/safety scanners, or control "
        "cabinet without an operator compartment is autonomous_robot_or_agv, "
        "never a conventional vehicle, even if it transports pallets or is "
        "described as a forklift. A generic body, wheels, or industrial scene "
        "alone is insufficient. The highlighted mask must cover a recognizable "
        "core of one complete machine, not merely a wheel, caster, fork tip, "
        "bumper, cabinet edge, or small chassis fragment. Use partial_fragment "
        "whenever the full original contains a machine but the cyan/isolated "
        "candidate contains only one such part. Use ambiguous when decisive "
        "pixels are blurred or outside the mask. Return strict compact JSON only."
    )
    return instruction, schema


def classify_visible_vehicle_candidate(config, image, candidate):
    """Separate conventional vehicles from cabinet-style AGV/AMR robots."""
    from .candidate_class_audit import prepare_candidate_evidence

    candidate_evidence = prepare_candidate_evidence(image, candidate)
    instruction, schema = visible_vehicle_semantic_contract()
    return chat(
        config, instruction, image,
        extra_images=(candidate_evidence["marked"],
                      candidate_evidence["isolated"]),
        response_schema=schema)


def sam_text_prefetch(image, prompt, config):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("SAM 3 单提示预热需要有效关键词")
    response = sam_request(config, "/prefetch_text", image, prompt=prompt.strip())
    if (isinstance(response, dict) and response.get("prefetched") is False
            and response.get("code") == "sam_busy"):
        return response
    if not isinstance(response, dict) or not isinstance(response.get("results"), list):
        raise ValueError("SAM 3 单提示预热响应格式无效")
    return response | {"prefetched": True}


def _sam_multi_endpoint_key(config):
    return (config.get("sam3_url") or os.environ.get(
        "SAM3_URL", "http://127.0.0.1:8766")).rstrip("/")


def _sam_text_multi_legacy(image, prompts, config, timeout_seconds=None):
    """Adapt the original list/items contract used by the 5090 SAM worker."""
    request_options = {"prompts": list(prompts.values())}
    if timeout_seconds is not None:
        request_options["timeout_seconds"] = timeout_seconds
    response = sam_request(
        config, "/segment_text_multi", image, **request_options)
    items = response.get("items") if isinstance(response, dict) else None
    if not isinstance(items, list) or len(items) != len(prompts):
        raise ValueError("SAM 3 旧版多提示响应缺少完整类别结果")
    outputs = {}
    for (label, expected_prompt), item in zip(prompts.items(), items):
        if (not isinstance(item, dict)
                or item.get("prompt") != expected_prompt
                or not isinstance(item.get("results"), list)):
            raise ValueError(f"SAM 3 旧版多提示响应中的类别 {label} 格式无效")
        payload = {"results": item["results"]}
        if isinstance(item.get("diagnostics"), dict):
            payload["diagnostics"] = item["diagnostics"]
        outputs[label] = payload
    diagnostics = (response.get("diagnostics", {}).copy()
                   if isinstance(response.get("diagnostics"), dict) else {})
    diagnostics["legacy_protocol"] = True
    diagnostics["image_encoded_once"] = bool(
        diagnostics.get("image_encoded_once")
        or diagnostics.get("feature_reuse")
        or diagnostics.get("image_feature_passes") == 1)
    return outputs, diagnostics


def sam_text_multi(image, prompts, config, *, timeout_seconds=None):
    """Evaluate several label prompts while the service reuses one image encoding."""
    if (not isinstance(prompts, dict) or not 2 <= len(prompts) <= 16
            or any(not isinstance(label, str) or not label.strip()
                   or not isinstance(prompt, str) or not prompt.strip()
                   for label, prompt in prompts.items())):
        raise ValueError("SAM 3 多提示请求须包含 2～16 个有效类别与关键词")
    endpoint_key = _sam_multi_endpoint_key(config)
    with _sam_multi_protocol_lock:
        protocol = _sam_multi_protocols.get(endpoint_key)
    if protocol == "legacy_list":
        return _sam_text_multi_legacy(
            image, prompts, config, timeout_seconds=timeout_seconds)
    try:
        request_options = {"prompts": prompts}
        if timeout_seconds is not None:
            request_options["timeout_seconds"] = timeout_seconds
        response = sam_request(
            config, "/segment_text_multi", image, **request_options)
    except ValueError as error:
        if "image (base64) and prompts are required" not in str(error):
            raise
        outputs, diagnostics = _sam_text_multi_legacy(
            image, prompts, config, timeout_seconds=timeout_seconds)
        with _sam_multi_protocol_lock:
            _sam_multi_protocols[endpoint_key] = "legacy_list"
        return outputs, diagnostics
    outputs = response.get("prompts") if isinstance(response, dict) else None
    if not isinstance(outputs, dict) or set(outputs) != set(prompts):
        raise ValueError("SAM 3 多提示响应缺少类别结果")
    for label, payload in outputs.items():
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise ValueError(f"SAM 3 多提示响应中的类别 {label} 格式无效")
    with _sam_multi_protocol_lock:
        _sam_multi_protocols[endpoint_key] = "mapping"
    return outputs, response.get("diagnostics", {})


def sam_text_multi_prefetch(image, prompts, config):
    """Warm the exact multi-prompt response without falling back or queuing.

    The remote service owns the busy policy. A 202-style ``sam_busy`` payload
    is returned to the caller as an ordinary optional-prewarm result; malformed
    successful responses still fail closed and are never presented as warm.
    """
    if (not isinstance(prompts, dict) or not 2 <= len(prompts) <= 16
            or any(not isinstance(label, str) or not label.strip()
                   or not isinstance(prompt, str) or not prompt.strip()
                   for label, prompt in prompts.items())):
        raise ValueError("SAM 3 预热请求须包含 2～16 个有效类别与关键词")
    response = sam_request(config, "/prefetch_text_multi", image, prompts=prompts)
    if (isinstance(response, dict) and response.get("prefetched") is False
            and response.get("code") == "sam_busy"):
        return response
    outputs = response.get("prompts") if isinstance(response, dict) else None
    if not isinstance(outputs, dict) or set(outputs) != set(prompts):
        raise ValueError("SAM 3 预热响应缺少类别结果")
    for label, payload in outputs.items():
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise ValueError(f"SAM 3 预热响应中的类别 {label} 格式无效")
    return response | {"prefetched": True}


def chat(config, instruction, image, extra_images=(), response_schema=None):
    record = {"id": uuid.uuid4().hex, "service": "GPT", "status": "running",
              "started_at": time.time(), "input": {"model": config.get("llm_model"),
              "instruction": instruction, "images": [_image_summary(data)
              for data in (image, *extra_images)]}}
    if config.get("llm_prompt_profile") == "compact":
        record["input"]["max_tokens"] = int(config.get("llm_max_tokens") or 768)
    if response_schema is not None:
        record["input"]["response_schema"] = response_schema
    review_call = _review_call.get()
    if review_call is not None:
        record.update(review_attempt=review_call["attempt"],
                      review_max_attempts=MAX_PROTOCOL_RETRIES + 1)
        if review_call.get("feedback"):
            record["retry_feedback"] = review_call["feedback"]
        review_call["record"] = record
    raw = {}
    _emit_trace(record)
    try:
        options = {"response_schema": response_schema} if response_schema is not None else {}
        response = _cancellable_call(_chat, config, instruction, image, extra_images, raw, **options)
        record.update(status="succeeded", output={"raw_text": raw.get("text"),
                                                   "finish_reason": raw.get("finish_reason"),
                                                   "busy_retry_count": raw.get("busy_retry_count", 0),
                                                   "model_route": raw.get("model_route"),
                                                   "fast_fallback_reason": raw.get("fast_fallback_reason"),
                                                   "structured_fallback_reason": raw.get("structured_fallback_reason"),
                                                   "local_vlm": raw.get("local_vlm"),
                                                   "parsed": response})
        return response
    except Exception as error:
        record.update(status="failed", error=str(error)[:500])
        if raw:
            record["output"] = {"raw_text": raw.get("text"),
                                "finish_reason": raw.get("finish_reason"),
                                "busy_retry_count": raw.get("busy_retry_count", 0),
                                "model_route": raw.get("model_route"),
                                "fast_fallback_reason": raw.get("fast_fallback_reason")}
        raise
    finally:
        record["elapsed_ms"] = round((time.time() - record["started_at"]) * 1000)
        _emit_trace(record)


def chat_batch(config, requests, response_schema=None, max_tokens=384):
    """Run a bounded batch of independent prompts through the local VLM."""
    return _cancellable_call(
        _chat_batch, config, requests, response_schema, max_tokens)


def chat_fast_only(config, instruction, image, extra_images=(), response_schema=None):
    """Use the independent small-image VLM lane without occupying the primary.

    This helper is intentionally fail-closed: it never falls back to the
    primary route by itself.  Callers that overlap it with a primary batch can
    therefore wait for that batch to finish and then perform the established
    sequential fallback without creating two simultaneous primary requests.
    """
    _, route = select_vlm_route(config, image, extra_images=extra_images)
    if route != "small_image_fast":
        raise ValueError("快速复核通道不适用于当前图像或尚未配置")
    isolated = dict(config)
    isolated["_llm_fast_only"] = True
    return chat(
        isolated, instruction, image, extra_images=extra_images,
        response_schema=response_schema)


def _positive_int(value, default):
    if type(value) is not int or value < 1:
        return default
    return value


def _fast_route_key(config):
    return (str(config.get("llm_fast_url") or "").rstrip("/"),
            str(config.get("llm_fast_model") or "").strip())


def _fast_route_circuit_reason(config):
    key = _fast_route_key(config)
    if not all(key):
        return None
    now = time.monotonic()
    with _fast_route_lock:
        state = _fast_route_circuits.get(key)
        if not state:
            return None
        if state["disabled_until"] <= now:
            _fast_route_circuits.pop(key, None)
            return None
        return "fast_circuit_open"


def _open_fast_route_circuit(config, reason):
    key = _fast_route_key(config)
    if not all(key):
        return
    configured = max(10, min(300, _positive_int(
        config.get("llm_fast_cooldown_seconds"), 60)))
    cooldown = min(15, configured) if reason == "fast_busy" else configured
    with _fast_route_lock:
        _fast_route_circuits[key] = {
            "disabled_until": time.monotonic() + cooldown,
            "trigger": reason,
        }


def _close_fast_route_circuit(config):
    with _fast_route_lock:
        _fast_route_circuits.pop(_fast_route_key(config), None)


def _reset_fast_route_circuits():
    """Clear process-local fast-route health state for isolated tests."""
    with _fast_route_lock:
        _fast_route_circuits.clear()


def select_vlm_route(config, image, extra_images=()):
    """Select the optional small-image accelerator without changing source state.

    The accelerator is deliberately opt-in and eligible only when every image
    in the request is within both configured bounds.  Any undecodable image is
    sent to the primary route so the established transport keeps ownership of
    the resulting validation error.
    """
    fast_url = str(config.get("llm_fast_url") or "").strip()
    fast_model = str(config.get("llm_fast_model") or "").strip()
    if not fast_url or not fast_model:
        return config, "primary"
    max_pixels = _positive_int(config.get("llm_fast_max_pixels"), 640 * 512)
    max_long_edge = _positive_int(config.get("llm_fast_max_long_edge"), 640)
    max_extra_pixels = _positive_int(
        config.get("llm_fast_max_extra_pixels"), max_pixels * 4)
    max_extra_long_edge = _positive_int(
        config.get("llm_fast_max_extra_long_edge"), max_long_edge * 2)
    for position, data in enumerate((image, *extra_images)):
        try:
            with Image.open(io.BytesIO(data)) as decoded:
                width, height = decoded.size
        except (OSError, TypeError, ValueError):
            return config, "primary"
        pixel_limit = max_pixels if position == 0 else max_extra_pixels
        edge_limit = max_long_edge if position == 0 else max_extra_long_edge
        if width * height > pixel_limit or max(width, height) > edge_limit:
            return config, "primary"
    fast = dict(config)
    fast["llm_url"] = fast_url
    fast["llm_model"] = fast_model
    fast["llm_key"] = config.get("llm_fast_key", "")
    fast["llm_busy_retries"] = 0
    return fast, "small_image_fast"


def _chat(config, instruction, image, extra_images=(), raw=None, response_schema=None):
    if not config.get("llm_url") or not config.get("llm_model"):
        raise ValueError("请先配置大模型地址和模型名")
    content = [{"type": "text", "text": instruction}]
    for data in (image, *extra_images):
        content.append({"type": "image_url", "image_url": {"url":
            "data:image/jpeg;base64," + base64.b64encode(data).decode()}})
    fast_only = config.get("_llm_fast_only") is True
    fast_fallback_reason = _fast_route_circuit_reason(config)
    if fast_only and fast_fallback_reason is not None:
        raise ValueError(f"快速复核通道暂不可用：{fast_fallback_reason}")
    if fast_fallback_reason is None:
        selected_config, selected_route = select_vlm_route(
            config, image, extra_images=extra_images)
    else:
        selected_config, selected_route = config, "primary"
    routes = [(selected_config, selected_route)]
    if fast_only and selected_route != "small_image_fast":
        raise ValueError("快速复核通道不适用于当前图像或尚未配置")
    if selected_route == "small_image_fast" and not fast_only:
        routes.append((config, "primary_fallback"))
    busy_retry_count = 0
    result = None
    final_route = selected_route
    for route_config, route_name in routes:
        if raw is not None:
            raw["model_route"] = route_name
            if fast_fallback_reason is not None:
                raw["fast_fallback_reason"] = fast_fallback_reason
        payload = {"model": route_config["llm_model"], "temperature": 0,
                   "messages": [{"role": "user", "content": content}]}
        if route_config.get("llm_prompt_profile") == "compact":
            payload["max_tokens"] = int(route_config.get("llm_max_tokens") or 768)
        if response_schema is not None:
            payload["response_format"] = {"type": "json_schema", "json_schema": {"name": "labelagent", "schema": response_schema, "strict": True}}
        request = urllib.request.Request(
            route_config["llm_url"].rstrip("/") + "/chat/completions",
            json.dumps(payload).encode(), headers={"Content-Type": "application/json",
            "Authorization": "Bearer " + route_config.get("llm_key", "")})
        busy_retry_limit = max(0, min(8, int(route_config.get("llm_busy_retries", 5))))
        timeout_seconds = (max(10, min(300, _positive_int(
            config.get("llm_fast_timeout_seconds"), 60)))
            if route_name == "small_image_fast" else 1200)
        route_failed = False
        for busy_attempt in range(busy_retry_limit + 1):
            try:
                with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                    result = json.load(response)
                final_route = route_name
                if route_name == "small_image_fast":
                    _close_fast_route_circuit(config)
                break
            except (TimeoutError, socket.timeout) as error:
                if route_name == "small_image_fast":
                    fast_fallback_reason = "fast_timeout"
                    if raw is not None:
                        raw["fast_fallback_reason"] = fast_fallback_reason
                    _open_fast_route_circuit(config, fast_fallback_reason)
                    route_failed = True
                    break
                raise ValueError("llm_timeout: 大模型请求排队或推理超过 1200 秒") from error
            except urllib.error.HTTPError as error:
                try:
                    detail = error.read().decode("utf-8", "replace")[:300]
                except Exception:
                    detail = str(error.reason)
                local_busy = (error.code == 503 and
                              ("local vlm is busy" in detail.lower()
                               or '"busy"' in detail.lower()))
                structured_constraint_failure = (
                    error.code == 400 and response_schema is not None
                    and ("prefix_allowed_tokens_fn" in detail
                         or "constraint is unsatisfiable" in detail.lower()))
                if structured_constraint_failure:
                    # The local decoder can occasionally reach an impossible
                    # token prefix even though the same prompt can still emit
                    # valid JSON. Retry exactly once without the transport-level
                    # grammar; the normal JSON parser and Agent protocol
                    # validator remain mandatory afterwards.
                    fallback_payload = copy.deepcopy(payload)
                    fallback_payload.pop("response_format", None)
                    fallback_request = urllib.request.Request(
                        route_config["llm_url"].rstrip("/") + "/chat/completions",
                        json.dumps(fallback_payload).encode(),
                        headers={"Content-Type": "application/json",
                                 "Authorization": "Bearer " + route_config.get("llm_key", "")})
                    if raw is not None:
                        raw["structured_fallback_reason"] = "decoder_constraint_unsatisfiable"
                    try:
                        with urllib.request.urlopen(
                                fallback_request, timeout=timeout_seconds) as response:
                            result = json.load(response)
                    except (TimeoutError, socket.timeout) as fallback_error:
                        raise ValueError(
                            "llm_timeout: 结构化解码恢复请求超过等待上限") from fallback_error
                    except urllib.error.HTTPError as fallback_error:
                        try:
                            fallback_detail = fallback_error.read().decode(
                                "utf-8", "replace")[:300]
                        except Exception:
                            fallback_detail = str(fallback_error.reason)
                        raise ValueError(
                            f"大模型结构化解码恢复失败（HTTP {fallback_error.code}）："
                            f"{fallback_detail}") from fallback_error
                    except urllib.error.URLError as fallback_error:
                        raise ValueError(
                            f"大模型结构化解码恢复失败：{fallback_error.reason}") from fallback_error
                    final_route = route_name
                    if route_name == "small_image_fast":
                        _close_fast_route_circuit(config)
                    break
                if route_name == "small_image_fast" and (
                        local_busy or error.code in (408, 429)
                        or 500 <= error.code <= 599):
                    fast_fallback_reason = ("fast_busy" if local_busy or error.code == 429
                                            else "fast_http_unavailable")
                    if raw is not None:
                        raw["fast_fallback_reason"] = fast_fallback_reason
                    _open_fast_route_circuit(config, fast_fallback_reason)
                    route_failed = True
                    break
                if local_busy and busy_attempt < busy_retry_limit:
                    busy_retry_count += 1
                    # Newer VLM servers queue requests internally. Keep this
                    # short fallback only for an older server during rollout;
                    # long exponential sleeps leave the model idle and make a
                    # harmless overlap look like a stalled demo.
                    retry_seconds = 1
                    review_call = _review_call.get()
                    trace_record = review_call.get("record") if review_call else None
                    if trace_record is not None:
                        trace_record.pop("error", None)
                        trace_record.update(status="waiting", busy_retry_count=busy_retry_count,
                                            retry_in_seconds=retry_seconds,
                                            waiting_reason="共享视觉模型正在处理上一项请求，当前任务已排队")
                        _emit_trace(trace_record)
                    _wait_cancellable(retry_seconds)
                    continue
                raise ValueError(f"大模型请求失败（HTTP {error.code}）：{detail}") from error
            except urllib.error.URLError as error:
                if route_name == "small_image_fast":
                    fast_fallback_reason = "fast_unavailable"
                    if raw is not None:
                        raw["fast_fallback_reason"] = fast_fallback_reason
                    _open_fast_route_circuit(config, fast_fallback_reason)
                    route_failed = True
                    break
                raise ValueError(f"大模型请求失败：{error.reason}") from error
        if result is not None:
            break
        if not route_failed:
            break
    if result is None:
        raise ValueError("大模型请求失败：快速通道和主通道均未返回结果")
    if raw is not None:
        raw["busy_retry_count"] = busy_retry_count
        raw["model_route"] = final_route
        if fast_fallback_reason is not None:
            raw["fast_fallback_reason"] = fast_fallback_reason
        metadata = result.get("local_vlm")
        if isinstance(metadata, dict):
            # The local service already returns request-bound timing and image
            # encoding attestations. Preserve that bounded metadata in the
            # trace so latency experiments do not need a racy /health read.
            allowed = {
                "backend", "quantization", "model_config_sha256", "image_count",
                "requested_image_token_budget", "effective_image_token_budget",
                "actual_image_token_counts", "image_grid_thw", "prompt_tokens",
                "completion_tokens", "preprocess_seconds",
                "constraint_setup_seconds", "generation_seconds", "seconds",
                "cache_hit", "cache_key_sha256", "response_cache_entries",
                "coalesced", "coalesced_wait_seconds",
            }
            raw["local_vlm"] = {key: copy.deepcopy(value)
                                for key, value in metadata.items() if key in allowed}
            if type(raw["local_vlm"].get("cache_hit")) is not bool:
                raw["local_vlm"].pop("cache_hit", None)
            cache_key = raw["local_vlm"].get("cache_key_sha256")
            if (not isinstance(cache_key, str) or len(cache_key) != 64
                    or any(character not in "0123456789abcdef" for character in cache_key)):
                raw["local_vlm"].pop("cache_key_sha256", None)
            cache_entries = raw["local_vlm"].get("response_cache_entries")
            if (type(cache_entries) is not int or not 0 <= cache_entries <= 256):
                raw["local_vlm"].pop("response_cache_entries", None)
            if type(raw["local_vlm"].get("coalesced")) is not bool:
                raw["local_vlm"].pop("coalesced", None)
            coalesced_wait = raw["local_vlm"].get("coalesced_wait_seconds")
            if (type(coalesced_wait) not in (int, float)
                    or isinstance(coalesced_wait, bool) or not 0 <= coalesced_wait <= 300):
                raw["local_vlm"].pop("coalesced_wait_seconds", None)
            stats = metadata.get("inference_stats")
            if isinstance(stats, dict):
                stat_keys = {
                    "prompt_tokens", "completion_tokens", "preprocess_seconds",
                    "constraint_setup_seconds", "generation_seconds",
                    "actual_image_token_counts", "image_grid_thw",
                    "effective_image_token_budget", "structured_output",
                    "generation_cache_implementation",
                    "configured_generation_cache_implementation",
                    "generation_cache_fallback_reason",
                }
                raw["local_vlm"]["inference_stats"] = {
                    key: copy.deepcopy(value) for key, value in stats.items()
                    if key in stat_keys
                }
            usage = result.get("usage")
            if isinstance(usage, dict):
                raw["local_vlm"]["usage"] = {
                    key: usage.get(key) for key in
                    ("prompt_tokens", "completion_tokens", "total_tokens")
                    if type(usage.get(key)) is int
                }
    choice = result["choices"][0]
    content = choice["message"]["content"]
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content)
    if raw is not None:
        raw["text"] = content[:20000]
        raw["finish_reason"] = choice.get("finish_reason")
    content = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        result = json.loads(content)
    except json.JSONDecodeError as error:
        raise ModelOutputError(
            f"大模型未返回有效 JSON：{error}", content[:20000],
            finish_reason=choice.get("finish_reason")) from error
    if not isinstance(result, dict):
        raise ModelOutputError(
            "大模型必须返回 JSON 对象", content[:20000],
            finish_reason=choice.get("finish_reason"))
    return result


def _chat_batch(config, requests, response_schema=None, max_tokens=384):
    if (not isinstance(requests, list) or not 2 <= len(requests) <= 3
            or type(max_tokens) is not int or not 1 <= max_tokens <= 4096):
        raise ValueError("批量盲审只接受 2～3 个独立请求")
    if not config.get("llm_url") or not config.get("llm_model"):
        raise ValueError("请先配置大模型地址和模型名")
    bodies = []
    for item in requests:
        if not isinstance(item, dict) or not isinstance(item.get("instruction"), str):
            raise ValueError("批量盲审请求格式无效")
        images = (item.get("image"), *tuple(item.get("extra_images") or ()))
        if any(not isinstance(data, bytes) or not data for data in images):
            raise ValueError("批量盲审图像无效")
        content = [{"type": "text", "text": item["instruction"]}]
        for data in images:
            content.append({"type": "image_url", "image_url": {"url":
                "data:image/jpeg;base64," + base64.b64encode(data).decode()}})
        body = {"model": config["llm_model"], "temperature": 0,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": content}]}
        if response_schema is not None:
            body["response_format"] = {
                "type": "json_schema", "json_schema": {
                    "name": "labelagent", "schema": response_schema,
                    "strict": True}}
        bodies.append(body)
    endpoint = (config["llm_url"].rstrip("/")
                + "/chat/batch_completions")
    request = urllib.request.Request(
        endpoint, json.dumps({"requests": bodies}).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + config.get("llm_key", "")})
    try:
        with urllib.request.urlopen(request, timeout=1200) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        try:
            detail = error.read().decode("utf-8", "replace")[:300]
        except Exception:
            detail = str(error.reason)
        raise ValueError(
            f"批量盲审请求失败（HTTP {error.code}）：{detail}") from error
    except urllib.error.URLError as error:
        raise ValueError(f"批量盲审请求失败：{error.reason}") from error
    outputs = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(outputs, list) or len(outputs) != len(requests):
        raise ValueError("批量盲审响应数量不匹配")
    parsed = []
    for output in outputs:
        try:
            choice = output["choices"][0]
            content = choice["message"]["content"]
            if isinstance(content, list):
                content = "".join(part.get("text", "") for part in content)
            content = content.strip().removeprefix("```json").removeprefix(
                "```").removesuffix("```").strip()
            value = json.loads(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise ModelOutputError(
                "批量盲审未返回有效 JSON 对象", str(output)[:20000]) from error
        if not isinstance(value, dict):
            raise ModelOutputError(
                "批量盲审必须逐项返回 JSON 对象", content[:20000])
        parsed.append(value)
    return parsed


def compact_review_token_budget(config, hole_count):
    """Budget structured review output from the bounded schema complexity."""
    if config.get("llm_prompt_profile") != "compact":
        return None
    if config.get("llm_max_tokens") is not None:
        return int(config["llm_max_tokens"])
    return min(1536, 768 + 192 * min(4, max(0, int(hole_count))))


def compact_discovery_token_budget(config, label_count):
    """Use the validated short budget only for small compact registries."""
    if config.get("llm_prompt_profile") != "compact":
        return None
    if config.get("llm_max_tokens") is not None:
        return int(config["llm_max_tokens"])
    label_count = min(20, max(1, int(label_count)))
    if label_count <= 8:
        return 384
    return min(1024, 384 + 32 * label_count)


def compact_discovery_fallback_budget(config, label_count):
    """Return the former full budget used after a short response truncates."""
    if config.get("llm_prompt_profile") != "compact":
        return None
    if config.get("llm_max_tokens") is not None:
        return int(config["llm_max_tokens"])
    return min(1024, 384 + 32 * min(20, max(1, int(label_count))))


def reference_labels(annotations, labels):
    """Only categories actually annotated on the reference frame may propagate."""
    present = {shape.get("label") for shape in annotations}
    selected = [label for label in labels if label in present]
    if not selected or len(selected) != len(present):
        raise ValueError("参考帧真值类别与类别列表不一致，请先核对参考帧标注")
    return selected


VISIBLE_LABEL_DEFINITIONS = {
    "车辆": "a physically visible generic vehicle instance; when ego or other_robot is also in the registry, do not reuse the same physical instance as evidence for more than one of these labels",
    "pallet": "a visible material-handling pallet; stacked cargo alone is not a pallet unless the pallet structure is visible",
    "person": "a visible human body; do not call a robot, mannequin, reflection, or vehicle a person",
    "floor": "the warehouse ground surface; exclude robot decks, vehicle bodies, pallets, ramps, and elevated cargo platforms",
    "slope": "a visibly inclined ramp or transition surface, kept separate from floor for annotation even though downstream policy may union floor and slope as traversable_area",
    "ego": "visible physical body, hood, chassis, forks, or structure belonging to the camera-carrying host robot; camera viewpoint or camera position alone is never evidence",
    "other_robot": "a physically visible autonomous unmanned robot or AGV chassis distinct from the camera-carrying ego robot and distinct from any human operator; a conventional forklift, tug, pallet jack, cart, or parked/unoccupied industrial vehicle is still 车辆, never other_robot, unless autonomous robot hardware is visibly supported",
    "cargo_floor": "a visible elevated cargo-carrying deck physically attached to or forming part of a robot/mobile platform; a pallet (including a blue plastic pallet), static raised platform, warehouse ground, or merely saying that a vehicle stands on a floor is never cargo_floor evidence",
}


def visible_label_definitions(labels):
    """Return only project semantics that apply to the active registry."""
    return [f"- {label}: {VISIBLE_LABEL_DEFINITIONS[label]}"
            for label in labels if label in VISIBLE_LABEL_DEFINITIONS]


def _complete_json_array(raw_text, key):
    """Recover one fully closed array without repairing or inventing values."""
    if not isinstance(raw_text, str):
        return None
    marker = json.dumps(str(key))
    marker_at = raw_text.find(marker)
    if marker_at < 0:
        return None
    colon_at = raw_text.find(":", marker_at + len(marker))
    start = raw_text.find("[", colon_at + 1) if colon_at >= 0 else -1
    if start < 0:
        return None
    depth, in_string, escaped = 0, False, False
    for position in range(start, len(raw_text)):
        character = raw_text[position]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(raw_text[start:position + 1])
                except json.JSONDecodeError:
                    return None
                return value if isinstance(value, list) else None
    return None


def _complete_json_object(raw_text, key):
    """Recover one fully closed object without repairing or inventing values."""
    if not isinstance(raw_text, str):
        return None
    marker = json.dumps(str(key))
    marker_at = raw_text.find(marker)
    if marker_at < 0:
        return None
    colon_at = raw_text.find(":", marker_at + len(marker))
    start = raw_text.find("{", colon_at + 1) if colon_at >= 0 else -1
    if start < 0:
        return None
    depth, in_string, escaped = 0, False, False
    for position in range(start, len(raw_text)):
        character = raw_text[position]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(raw_text[start:position + 1])
                except json.JSONDecodeError:
                    return None
                return value if isinstance(value, dict) else None
    return None


def discover_visible_labels(config, image, labels):
    """Discover visible registry labels without promoting open-set guesses.

    This is deliberately separate from ``describe``: the latter must produce a
    SAM prompt for every label it receives and therefore cannot truthfully be
    used as a visibility filter.  Unknown objects remain review-only evidence.
    """
    labels = list(dict.fromkeys(labels))
    if not labels or any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("自动识别需要非空的当前标签表")
    from .protocol_schema import visible_label_schema
    schema = visible_label_schema(labels)
    definitions = visible_label_definitions(labels)
    instruction = (
        "Inspect this factory image and assess every known label below exactly once. "
        "Return assessments as one JSON object keyed by the exact known label strings; "
        "every key is required and must occur once. "
        "Use evidence='present' only when at least one instance is visibly supported in the image; "
        "use 'absent' when it is not visible and 'uncertain' when occlusion or image quality prevents a reliable decision. "
        "Do not infer an object merely because its label is listed. confidence is a number from 0 to 1. "
        "Each reason must be one visual clause of at most 10 words; do not list multiple exclusions or repeat the definition. "
        "A reason must name visible pixels or a visible physical part, not scene context, camera viewpoint, or an assumed function. "
        "If 车辆, ego, and other_robot are all available, they are not synonyms: one physical instance must not make multiple labels present; use the most specific project label and identify distinct instances separately. "
        "pallet and cargo_floor are not synonyms: cargo_floor requires visible physical attachment to a robot/mobile platform, and a pallet or static raised platform must never support cargo_floor='present'. "
        "If a clearly visible object does not fit any known label, add only an unknown_candidates proposal; "
        "known labels may be Chinese or English, so do not report a translation or simple synonym of a known label as unknown; "
        "never rename, merge, or modify the known-label registry. Return only compact schema JSON with no indentation or extra whitespace.\n"
        + "Known labels: " + json.dumps(labels, ensure_ascii=False)
        + ("\nProject label definitions:\n" + "\n".join(definitions) if definitions else "")
    )
    open_set_scan_complete = True
    recovery_error = None
    request_config = config
    token_budget = compact_discovery_token_budget(config, len(labels))
    fallback_budget = compact_discovery_fallback_budget(config, len(labels))
    if token_budget is not None and token_budget != config.get("llm_max_tokens"):
        request_config = copy.deepcopy(config)
        request_config["llm_max_tokens"] = token_budget
    output_error = None
    try:
        result = chat(request_config, instruction, image, response_schema=schema)
    except ModelOutputError as error:
        output_error = error
        truncated = (error.finish_reason == "length"
                     or "unterminated string" in str(error).lower())
        if (truncated and fallback_budget is not None and token_budget is not None
                and fallback_budget > token_budget):
            retry_config = copy.deepcopy(config)
            retry_config["llm_max_tokens"] = fallback_budget
            try:
                result = chat(retry_config, instruction, image, response_schema=schema)
                output_error = None
            except ModelOutputError as retry_error:
                output_error = retry_error
    if output_error is not None:
        error = output_error
        recovered_assessments = _complete_json_object(error.raw_text, "assessments")
        if recovered_assessments is None:
            # Defensive compatibility for an in-flight response from the older
            # array contract during a rolling local upgrade.
            recovered_assessments = _complete_json_array(error.raw_text, "assessments")
        if recovered_assessments is None:
            raise error
        # Never guess or auto-complete a truncated open-set proposal.  A fully
        # closed known-label array is still useful for candidate generation,
        # while the missing open-set tail becomes an explicit ontology gate.
        result = {"assessments": recovered_assessments, "unknown_candidates": []}
        open_set_scan_complete = False
        recovery_error = str(error)
    if not isinstance(result, dict):
        raise ValueError("VLM 可见类别识别未返回 JSON 对象")
    assessments = result.get("assessments")
    if isinstance(assessments, dict):
        if set(assessments) != set(labels) or len(assessments) != len(labels):
            raise ValueError("VLM 必须逐项判断当前标签表中的每个类别")
        normalized = []
        for label in labels:
            value = assessments[label]
            if not isinstance(value, dict) or "label" in value:
                raise ValueError("VLM 类别判断必须是对象")
            normalized.append({"label": label, **value})
        assessments = normalized
    if not isinstance(assessments, list) or len(assessments) != len(labels):
        raise ValueError("VLM 必须逐项判断当前标签表中的每个类别")
    by_label, consistency_flags = {}, []
    if not open_set_scan_complete:
        consistency_flags.append({
            "normalized_evidence": "open_set_scan_incomplete",
            "reason_code": "vlm_open_set_output_truncated",
            "route": "ontology_review_required",
            "reason": ("Known-label assessments were complete, but the open-set tail was truncated; "
                       "unknown/new-label review remains required."),
            "model_error": recovery_error[:300] if recovery_error else None,
        })
    def positive_visibility_claim(text):
        normalized = " ".join(text.casefold().split())
        negative = ("not visible", "no visible", "isn't visible", "is not visible",
                    "are not visible", "cannot be seen", "can't be seen", "not seen",
                    "不可见", "未见", "没有看到", "无法看到")
        positive = ("is visible", "are visible", "visible in the image", "clearly visible",
                    "can be seen", "可见", "可以看到", "能看到")
        negated_visible = re.search(r"\b(?:no|not)\b.{0,80}\bvisible\b", normalized) is not None
        return (not negated_visible and not any(term in normalized for term in negative)
                and any(term in normalized for term in positive))

    def cargo_floor_consistency_issue(text):
        """Reject confident cargo-floor claims that lack robot attachment pixels."""
        normalized = " ".join(text.casefold().split())
        pallet_terms = ("pallet", "托盘", "栈板", "卡板")
        if any(term in normalized for term in pallet_terms):
            return "cargo_floor_pallet_confusion"
        attachment_terms = (
            "attached to a robot", "attached to the robot", "attached to robot",
            "mounted on a robot", "mounted on the robot", "mounted on robot",
            "part of a robot", "part of the robot", "part of robot",
            "integrated with a robot", "integrated with the robot",
            "robot chassis", "robot body", "robot cargo deck", "robot cargo platform",
            "mobile robot chassis", "mobile platform chassis", "mobile platform deck",
            "连接在机器人", "安装在机器人", "机器人底盘", "机器人车体",
            "机器人货台", "机器人载货平台", "移动平台的货台",
        )
        if not any(term in normalized for term in attachment_terms):
            return "cargo_floor_missing_robot_attachment_evidence"
        return None
    for item in assessments:
        if not isinstance(item, dict):
            raise ValueError("VLM 类别判断必须是对象")
        label, evidence, confidence, reason_text = (item.get("label"), item.get("evidence"),
                                                     item.get("confidence"), item.get("reason"))
        if label not in labels:
            consistency_flags.append({"label": str(label),
                "normalized_evidence": "uncertain",
                "reason_code": "vlm_unknown_registry_label",
                "route": "human_review_required",
                "reason": "VLM returned a label outside the current registry; it was not added to the registry."})
            continue
        if evidence not in {"present", "absent", "uncertain"}:
            raise ValueError(f"类别 {label} 的可见性状态无效")
        if type(confidence) not in (int, float) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
            raise ValueError(f"类别 {label} 的置信度无效")
        if not isinstance(reason_text, str) or not reason_text.strip():
            raise ValueError(f"类别 {label} 缺少视觉依据")
        registry_false_claim = ("not present in the list of known labels" in reason_text.casefold()
                                or "not in the list of known labels" in reason_text.casefold())
        if evidence == "absent" and (positive_visibility_claim(reason_text) or registry_false_claim):
            consistency_flags.append({"label": label, "original_evidence": "absent",
                "normalized_evidence": "uncertain", "reason_code": "vlm_visibility_contradiction",
                "route": "human_review_required", "reason": reason_text.strip()})
            evidence = "uncertain"
        if label == "cargo_floor" and evidence == "present":
            cargo_floor_issue = cargo_floor_consistency_issue(reason_text)
            if cargo_floor_issue is not None:
                consistency_flags.append({"label": label, "original_evidence": "present",
                    "normalized_evidence": "uncertain", "reason_code": cargo_floor_issue,
                    "route": "human_review_required", "reason": reason_text.strip()})
                evidence = "uncertain"
        assessment = {"label": label, "evidence": evidence,
                      "confidence": float(confidence), "reason": reason_text.strip()}
        if label in by_label:
            by_label[label] = {"label": label, "evidence": "uncertain",
                "confidence": min(by_label[label]["confidence"], float(confidence)),
                "reason": "VLM returned this known label more than once; human review is required."}
            consistency_flags.append({"label": label,
                "normalized_evidence": "uncertain",
                "reason_code": "vlm_duplicate_registry_assessment",
                "route": "human_review_required",
                "reason": "Duplicate known-label assessment was conservatively downgraded."})
        else:
            by_label[label] = assessment
    for missing_label in labels:
        if missing_label in by_label:
            continue
        by_label[missing_label] = {"label": missing_label, "evidence": "uncertain",
            "confidence": 0.0,
            "reason": "VLM did not return a unique assessment for this known label; human review is required."}
        consistency_flags.append({"label": missing_label,
            "normalized_evidence": "uncertain",
            "reason_code": "vlm_missing_registry_assessment",
            "route": "human_review_required",
            "reason": "Missing known-label assessment was conservatively downgraded."})
    # A conventional forklift/vehicle is not project ``other_robot`` merely
    # because it is parked or no driver is visible.  The local VLM can reuse
    # one visible forklift as both labels even after receiving the exclusivity
    # instruction.  Preserve the vehicle assessment, but route the unsupported
    # robot claim to review instead of presenting it as visible evidence.
    other_robot = by_label.get("other_robot")
    if other_robot and other_robot.get("evidence") == "present":
        robot_reason = " ".join(other_robot.get("reason", "").casefold().split())
        conventional_terms = (
            "forklift", "fork lift", "industrial vehicle", "pallet jack",
            "manual cart", "tug vehicle", "conventional vehicle",
            "叉车", "工业车辆", "托盘车", "牵引车", "手推车",
        )
        autonomous_terms = (
            "autonomous", "unmanned", "agv", "amr", "robot chassis",
            "mobile robot", "self-driving", "driverless",
            "自主", "无人", "机器人底盘", "移动机器人", "自动驾驶",
        )
        if (any(term in robot_reason for term in conventional_terms)
                and not any(term in robot_reason for term in autonomous_terms)):
            other_robot["evidence"] = "uncertain"
            consistency_flags.append({
                "label": "other_robot", "original_evidence": "present",
                "normalized_evidence": "uncertain",
                "reason_code": "other_robot_conventional_vehicle_conflict",
                "route": "human_review_required",
                "conflicting_labels": [label for label in ("车辆", "other_robot")
                                       if label in by_label],
                "reason": (
                    "The visible evidence names a conventional vehicle without "
                    "autonomous robot hardware; other_robot requires human review."),
                "original_reason": other_robot["reason"],
            })
    # A shared explicit point is suspicious evidence reuse, not proof of object
    # identity. Keep both SAM queries and require a human to resolve the class.
    exclusive_labels = [label for label in ("车辆", "ego", "other_robot")
                        if label in by_label and by_label[label]["evidence"] == "present"]
    point_pattern = re.compile(
        r"(?:\[\s*(\d+(?:\.\d+)?)\s*[,，]\s*(\d+(?:\.\d+)?)\s*\]"
        r"|\(\s*(\d+(?:\.\d+)?)\s*[,，]\s*(\d+(?:\.\d+)?)\s*\))")
    anchors = {}
    for label in exclusive_labels:
        reason = by_label[label]["reason"]
        matches = list(point_pattern.finditer(reason))
        # Natural-language comparisons may cite another object's location.
        # Only use a single, unqualified positive point as an entity anchor.
        if len(matches) != 1 or re.search(
                r"\b(?:not|no|without|unlike|except|left of|right of|behind|between|rather than)\b"
                r"|不是|并非|而非|没有|不在|左侧|右侧|后方|之间", reason, re.IGNORECASE):
            continue
        values = matches[0].groups()
        point = (float(values[0] or values[2]), float(values[1] or values[3]))
        anchors.setdefault(point, set()).add(label)
    for point, shared_labels in anchors.items():
        if len(shared_labels) < 2:
            continue
        participants = [label for label in exclusive_labels if label in shared_labels]
        for label in participants:
            by_label[label]["evidence"] = "uncertain"
            consistency_flags.append({
                "label": label, "original_evidence": "present",
                "normalized_evidence": "uncertain",
                "reason_code": "vlm_cross_label_entity_conflict",
                "route": "human_review_required", "conflicting_labels": participants,
                "shared_anchor": list(point),
                "reason": "Mutually exclusive labels cite the same explicit point; "
                          "this may be reused entity evidence and requires human review.",
                "original_reason": by_label[label]["reason"]})
    unknown = result.get("unknown_candidates", [])
    if not isinstance(unknown, list) or len(unknown) > 5:
        raise ValueError("VLM unknown_candidate 格式无效")
    normalized_unknown, registry_conflicts = [], []
    def alias_forms(value):
        normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", value.casefold())
        forms = {normalized}
        if normalized.isascii() and normalized.endswith("s") and len(normalized) > 3:
            forms.add(normalized[:-1])
        return forms
    known_aliases = {}
    for known_label in labels:
        known_aliases.setdefault(known_label, set()).update(alias_forms(known_label))
        fallback = SAM_PROMPT_FALLBACKS.get(known_label)
        if fallback:
            known_aliases[known_label].update(alias_forms(fallback))
    for item in unknown:
        if not isinstance(item, dict):
            raise ValueError("VLM unknown_candidate 必须是对象")
        name, evidence, confidence = item.get("candidate_name"), item.get("evidence"), item.get("confidence")
        if (not isinstance(name, str) or not name.strip() or not isinstance(evidence, str)
                or not evidence.strip() or type(confidence) not in (int, float)
                or isinstance(confidence, bool) or not 0 <= confidence <= 1):
            raise ValueError("VLM unknown_candidate 缺少名称、证据或有效置信度")
        candidate = {"candidate_name": name.strip(), "evidence": evidence.strip(),
                     "confidence": float(confidence), "route": "human_review_required",
                     "ontology_mutation_allowed": False}
        name_forms = alias_forms(name)
        alias_label = next((label for label, forms in known_aliases.items() if name_forms & forms), None)
        if alias_label is not None:
            registry_conflicts.append(candidate | {"matching_known_label": alias_label,
                "reason_code": "known_label_alias_conflict"})
        else:
            normalized_unknown.append(candidate)
    ordered = [by_label[label] for label in labels]
    visible_labels = [item["label"] for item in ordered if item["evidence"] == "present"]
    uncertain_labels = [item["label"] for item in ordered if item["evidence"] == "uncertain"]
    return {"assessments": ordered,
            "visible_labels": visible_labels,
            "uncertain_labels": uncertain_labels,
            "candidate_labels": [label for label in labels if label in set(visible_labels + uncertain_labels)],
            "consistency_flags": consistency_flags,
            "unknown_candidates": normalized_unknown,
            "registry_conflicts": registry_conflicts,
            "open_set_scan_complete": open_set_scan_complete,
            "registry_mutated": False}


def candidate_labels_with_recall_probes(discovery, labels,
                                        probe_missing_selected=False):
    """Add bounded high-value probes without promoting visibility truth.

    Low, distant ramps are often described as ordinary floor in a single
    camera image.  If floor is visible or uncertain, preserve one independent
    slope SAM probe so downstream geometry and conflict guards can decide
    whether a machine candidate exists.  Reference drafting may additionally
    probe every user-selected registry label that discovery did not cover.
    Probes never mutate discovery evidence and are not VLM truth, human truth,
    or training approval.
    """
    registry = list(dict.fromkeys(labels or []))
    visible = list((discovery or {}).get("visible_labels") or [])
    uncertain = list((discovery or {}).get("uncertain_labels") or [])
    selected = list((discovery or {}).get("candidate_labels")
                    or [label for label in registry
                        if label in set(visible + uncertain)])
    probes = []
    if ("slope" in registry and "slope" not in selected
            and "floor" in set(visible + uncertain)):
        probes.append("slope")
    if probe_missing_selected:
        selected_or_probed = set(selected + probes)
        probes.extend(label for label in registry
                      if label not in selected_or_probed)
    selected_set = set(selected + probes)
    return [label for label in registry if label in selected_set], probes


SAM_PROMPT_FALLBACKS = {
    "车辆": "vehicle",
    "地面": "floor",
    "pallet": "pallet",
    "person": "person",
    "floor": "floor",
    "slope": "traversable surface",
    "ego": "ego robot",
    "other_robot": "robot",
    "cargo_floor": "robot cargo platform",
}

# These overrides are evidence-backed canonical object queries.  In particular,
# a real factory frame showed that a compound VLM phrase containing "operator"
# returned only the near driver, while the canonical "person" query returned all
# four visible people.  Keep the semantic label unchanged; only stabilize the
# text query sent to SAM.
SAM_PROMPT_OVERRIDES = {
    "pallet": "pallet",
    "person": "person",
    "ego": "ego robot",
    "other_robot": "robot",
    "cargo_floor": "robot cargo platform",
    "slope": "traversable surface",
}

OTHER_ROBOT_SEMANTIC_FALLBACK_PROMPT = "autonomous pallet truck"
OTHER_ROBOT_SEMANTIC_FALLBACK_MAX_CANDIDATES = 3


def select_other_robot_semantic_fallback_candidates(
        candidates, maximum=OTHER_ROBOT_SEMANTIC_FALLBACK_MAX_CANDIDATES):
    """Bound a recall-only fallback without turning it into class truth.

    The fallback is invoked only after 8B discovery explicitly reports a
    visible other_robot and the canonical ``robot`` SAM query returns nothing.
    SAM confidence is used only to cap review cost; every retained proposal is
    still high-risk and must pass the existing label-blind class audit.
    """
    if type(maximum) is not int or not 1 <= maximum <= 8:
        raise ValueError("other_robot fallback candidate limit must be 1～8")
    ranked = sorted(
        candidates,
        key=lambda item: (-(item.get("score") if isinstance(
            item.get("score"), (int, float)) else -1), item["id"]))
    retained, suppressed = ranked[:maximum], ranked[maximum:]
    for candidate in retained:
        candidate["source"] = "sam_text_other_robot_semantic_fallback"
        candidate["source_prompt"] = OTHER_ROBOT_SEMANTIC_FALLBACK_PROMPT
        candidate["candidate_only"] = True
        candidate["human_truth"] = False
        candidate["approved_for_training"] = False
    filters = [{
        "candidate_id": candidate["id"],
        "label": candidate.get("label"),
        "score": candidate.get("score"),
        "reason_code": "other_robot_fallback_candidate_limit",
        "route": "human_review_required",
        "candidate_mutated": False,
        "human_truth": False,
        "approved_for_training": False,
    } for candidate in suppressed]
    return retained, filters


def fallback_sam_prompt(label):
    return SAM_PROMPT_FALLBACKS.get(label) or (
        label.replace("_", " ") if label.isascii() else "")


def restrict_description(result, labels):
    """Keep the detailed review rule separate from concise SAM keywords."""
    if not isinstance(result, dict) or not isinstance(result.get("description"), str) or \
            not result["description"].strip() or \
            not isinstance(result.get("prompts"), list):
        raise ValueError("大模型描述格式不正确")
    proposals = {item["label"]: item.get("text", "").strip()
                 for item in result["prompts"] if isinstance(item, dict)
                 and isinstance(item.get("label"), str)
                 and isinstance(item.get("text"), str)}
    prompts = []
    for label in labels:
        text = proposals.get(label, "").strip()
        # SAM text is an object query, not the detailed annotation rule.
        # If GPT returns a sentence, fall back to the known reference class.
        bad_words = {"exclude", "excluding", "include", "including", "without",
                     "around", "surrounding", "bounded", "boundary", "only", "not",
                     "visible", "the", "and", "or"}
        valid = (bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9 -]*", text))
                 and 1 <= len(text.split()) <= 4
                 and not any(word.lower() in bad_words for word in text.split()))
        if label in SAM_PROMPT_OVERRIDES:
            text, valid = SAM_PROMPT_OVERRIDES[label], True
        if label == "地面" and valid:
            valid = any(word in text.lower().split() for word in ("floor", "ground"))
        if not valid:
            text = fallback_sam_prompt(label)
        if not text:
            raise ValueError(f"类别 {label} 缺少简短英文 SAM 3 提示，请重新生成描述")
        prompts.append({"label": label, "text": text})
    return {"description": result["description"].strip(), "prompts": prompts}


def canonical_unreferenced_description(labels):
    """Safe prompt fallback for a machine draft with no reference contours.

    The visible-label VLM has already selected the classes in this path.  If
    its separate prose/prompt reply is truncated, canonical object queries are
    safer than guessing or pretending that a human reference rule exists.
    Unknown registry labels still fail closed because they have no fallback.
    """
    result = restrict_description({
        "description": (
            "Machine draft only: no human-confirmed reference contours were "
            "supplied. Canonical object prompts are used and every resulting "
            "mask requires human review."
        ),
        "prompts": [],
    }, labels)
    result.update({
        "prompt_source": "deterministic_canonical_fallback",
        "human_reference_available": False,
        "candidate_only": True,
    })
    return result


DESCRIPTION_INPUT_VERSION = 12
MAX_REFERENCE_DESCRIPTION_CHARS = 450


def require_complete_reference_description(description):
    """Reject a syntactically closed string that exhausted its schema bound."""
    if not isinstance(description, str) or not description.strip():
        raise ValueError("参考描述不能为空")
    text = description.strip()
    dangling = re.search(
        r"(?:\b(?:and|or|but|with|including|such as)|[,;:])$", text,
        flags=re.IGNORECASE)
    if (len(text) > MAX_REFERENCE_DESCRIPTION_CHARS
            or not re.search(r"[.!?。！？]$", text)
            or dangling):
        raise ValueError(
            "参考描述达到长度上限，可能被截断；请用不超过 450 个字符的完整句子重写")
    return text


def canonical_reference_geometry_description(annotations, labels):
    """Build a complete, non-visual fallback from supplied reference geometry.

    This path is used only after every VLM protocol retry failed.  It must not
    infer appearance, scene context, actions, or annotation policy from the
    image.  Unknown classes also fail closed instead of inventing a SAM query.
    """
    counts = {}
    part_count = 0
    hole_count = 0
    allowed = set(labels)
    for shape in annotations:
        if not isinstance(shape, dict) or shape.get("label") not in allowed:
            continue
        label = shape["label"]
        counts[label] = counts.get(label, 0) + 1
        parts = shape.get("parts")
        if not isinstance(parts, list) or not parts:
            parts = [shape]
        part_count += len(parts)
        for part in parts:
            holes = part.get("holes", []) if isinstance(part, dict) else []
            if isinstance(holes, list):
                hole_count += len(holes)
    if not counts:
        raise ValueError("参考轮廓中没有可用于确定性描述的已选类别")

    prompts = []
    for label in labels:
        text = SAM_PROMPT_OVERRIDES.get(label) or SAM_PROMPT_FALLBACKS.get(label)
        if not text:
            raise ValueError(f"类别 {label} 没有受控的 SAM 3 兜底提示")
        prompts.append({"label": label, "text": text})

    phrases = []
    for label in labels:
        count = counts.get(label)
        if count:
            phrases.append(f"{count} {label} instance{'s' if count != 1 else ''}")
    if len(phrases) == 1:
        count_text = phrases[0]
    else:
        count_text = ", ".join(phrases[:-1]) + ", and " + phrases[-1]
    unreferenced_count = len([label for label in labels if label not in counts])
    description = (
        f"Supplied reference contours contain {count_text}. "
        f"Their {part_count} contour parts contain {hole_count} excluded "
        f"hole{'s' if hole_count != 1 else ''}; {unreferenced_count} selected "
        "classes have no reference contour, and all resulting machine "
        "candidates require human review."
    )
    if len(description) > MAX_REFERENCE_DESCRIPTION_CHARS:
        description = (
            f"Supplied reference geometry contains {sum(counts.values())} "
            f"annotations across {len(counts)} selected classes. Its "
            f"{part_count} contour parts contain {hole_count} excluded "
            f"hole{'s' if hole_count != 1 else ''}; {unreferenced_count} "
            "selected classes have no reference contour, and all resulting "
            "machine candidates require human review."
        )
    description = require_complete_reference_description(description)
    return {
        "description": description,
        "prompts": prompts,
        "prompt_source": "deterministic_reference_geometry_fallback",
        "human_reference_available": True,
        "candidate_only": True,
        "description_fallback_reason": "vlm_protocol_retries_exhausted",
    }

COMPACT_REFERENCE_RULES = """Use the supplied reference contours as the task example. Their review status is not inferred.
Only annotate the listed classes. Inspect original images for appearance and outlined images for geometry.
Overlay colors are not object colors. Preserve correct visible regions and excluded holes; do not invent occluded regions.
Instances, positions and counts can differ between frames. Never copy reference coordinates to the target.
Reference bounding boxes summarize contours, not rectangular segmentation masks.
Make conservative local edits only when supported by target-image and reference evidence.
"""

COMPACT_UNREFERENCED_RULES = """No human-confirmed reference image or reference contour was supplied.
Use only the target original and target candidate overlay. Never claim, quote, compare with, or infer a reference outline, reference mask, reference class, or human truth.
This is a machine-candidate draft. Preserve every candidate unless a supported non-destructive local repair is possible; removal is forbidden and potential false positives must remain for human review.
Audit every candidate label against class_definitions before considering geometry edits. A visually unmistakable one-instance semantic mismatch may use reclassify only to another selected class; ambiguous, mixed, partly occluded, or low-confidence candidates must keep their original label for human review.
Use each candidate ID in at most one operation per round. Never both split and reclassify the same ID in one response. A candidate containing mixed semantics must not be reclassified wholesale; use only a precise local non-destructive repair when its boundary is visually clear, otherwise keep it unchanged for human review.
In particular, a visible motorized chassis, wheels, sensors, or autonomous mobile base is other_robot when it is distinct from the camera host, not pallet. A static load-bearing pallet remains pallet. Visible foreground structure belonging to the camera host is ego, not other_robot. cargo_floor requires a visibly attached cargo deck and is not a free-standing pallet.
Only annotate the listed classes. Overlay colors are not object colors. Do not invent occluded regions or missing evidence.
"""

COMPACT_REVIEW_TOOLS = """Return one JSON object with fields status, reason, hole_assessment, operations.
Use status "done" and empty operations when no supported edit remains; otherwise status "continue". reason explains visual evidence.
Every operation needs operation and reason. shape_ids are existing integer IDs; polygon is [[x,y],...] with at least 3 target-image pixel points.
Tools:
remove: shape_ids:[id], only for an entirely false-positive instance.
reclassify: shape_ids:[id], label:allowed class different from that candidate's current label; never submit a no-op reclassification.
add_region, erase_region: shape_ids:[id], polygon:local repair region.
merge: shape_ids:[id1,id2,...], label:allowed class, only one real instance.
split: shape_ids:[id], polygon:region separating instances.
fill_gap: shape_ids:[id], hole_id:listed ID, polygon:local repair region.
sam_points: shape_ids:[id], polygon:local region, points:list of objects with numeric x,y and integer label (1 positive, 0 negative), 1-32 points with a positive point.
Never use sam_text in review or replace an entire correct object to repair its boundary.
After an edit, choose continue and inspect again. When no supported edit remains choose done with no operations; never needs_review.
If pending_sam_shape is present, add sam_assessment with boolean improved and string reason explaining the comparison.
If not improved, the previous geometry is restored and only OpenCV local edits are allowed for that shape.
For every listed hole return an object with hole_id, semantic ("ground" or "non_ground"), and reason in hole_assessment.
Only a hole assessed as ground may use fill_gap. A non_ground hole must never use fill_gap.
Ground holes must be repaired with fill_gap before done. Never fill equipment or infer semantics from geometry alone.
For floor tasks, include visible floor markings regardless of their color, without filling occluded parts.
Image 1: target original. Image 2: target candidates with IDs. Image 3: target holes/binary-mask overview.
If reference_provided, images 4 and 5 are reference original and reference contours.
"""

# Shared by description generation and every subsequent candidate review.
REFERENCE_PRINCIPLES = """
参考帧真值原则：
1. 参考帧真值优先：经人工确认为参考帧的真值是判断标注归属的首要依据，
   无论最初来自人工标注还是自动标注。不得用常识或默认分割习惯推翻其明确标注。
2. 先比对，再描述或决策：结合参考帧原图与真值几何数据，核对外轮廓、内部孔洞
   （holes）及各组成区域（parts），再判断目标归属、边界和包含／排除规则。
   标注轮廓颜色只是界面标记，绝不是物体真实颜色；颜色和材质仅依据未着色原图。
3. 包含与排除规则必须有标注证据：逐项核对相关内容在参考真值区域内还是被明确排除。
   例如，若标线已包含在地面真值内，不能仅因颜色差异将其排除；不得预设所有标线都包含或排除。
   不确定的归属不得写成确定规则，不得凭物体名称、颜色或常识新增孔洞；证据不足时省略未经证实的规则。
4. 参考帧未出现的独立物体默认按非地面语义处理：不能因为当前帧看到了参考帧中未出现的
   物体，就把它并入“地面”。若它有明确的其他语义且存在对应类别，按该类别处理；若当前任务没有
   对应类别，则不创建新类别，也不把它标进地面。若它遮挡地面，只保留其可见物体范围，不把物体内部当作地面。
   地面上的标线、纹理或痕迹是否属于地面，先按参考帧同类内容的真值归属判断，不能仅凭颜色将其视为独立物体。
5. 地面框线归属规则：原图中能够确认是绘制或粘贴在地面上的框线、边线或地面标线，
   无论颜色，只要当前可见且没有被物体遮挡，其可见部分必须包含在“地面”语义 mask 内。
   不能因为线条颜色、亮度或与地面材质不同而扣除；被物体遮挡的部分不补全。若无法确认线条位于地面，
   或它实际属于设备、桌椅等物体结构，则不适用本规则，必须依据参考帧真值判断。
6. 限定标注类别，不擅自改变区域归属：只为参考真值已有类别生成任务。
   “不为某物体新增类别”不能推导为“必须从现有类别扣除该物体所在区域”；是否属于已有类别，
   必须沿用参考帧中同类物体的标注方法。
7. 谨慎解释未标注区域：结合原图与真值区分明确排除、遮挡、画面截断及可能的标注不完整。
   参考帧中明确纳入目标区域的同类内容，在其他帧仍按同一语义纳入；参考帧从未出现的独立物体默认不并入地面。
   例如参考真值将黄色地面标线纳入地面、将机器人排除在地面之外时，其他帧沿用两者的归属；
   初始候选漏掉应纳入的标线或覆盖应排除的机器人时，应据此进行局部修复。
8. 提炼可迁移规则：区分标注规范与参考帧事实，位置、数量和具体轮廓不能变成其他帧的固定要求。
9. 输出前检查一致性：逐条对照参考帧真值，检查是否错误排除真值内内容、要求包含明确排除的内容，
   或新增没有依据的规则。出现矛盾时依据参考帧真值修正；不能可靠判断时不要猜测。
"""


def describe(config, image, annotations, labels, marked_image=None, include_unreferenced=False):
    if include_unreferenced:
        labels = list(dict.fromkeys(labels))
        present = {shape.get("label") for shape in annotations}
        if not labels:
            raise ValueError("推理类别不能为空")
    else:
        labels = reference_labels(annotations, labels)
    unreferenced = [label for label in labels if label not in {shape.get("label") for shape in annotations}]
    extended_note = ("\n允许类别中以下类别没有参考轮廓："
                     + json.dumps(unreferenced, ensure_ascii=False)
                     + "。只为它们生成通用英文对象关键词；不得声称参考帧提供了其边界或包含规则。"
                     if unreferenced else "")
    prompt = (REFERENCE_PRINCIPLES
              + "\n图像 1 是未着色的原始参考帧；图像 2（若提供）只用洋红色轮廓和编号标出参考帧真值。"
              "标注轮廓颜色只是界面标记，绝不是目标物体本身的颜色。"
              "描述地面或其他目标的真实颜色、材质时，仅依据图像 1；无法确认则省略颜色。"
              "真值几何数据中的 points 是外轮廓，holes 是排除的内部孔洞，parts 是同一标注的组成区域。"
              "结合参考帧图像及真值几何数据，输出严格 JSON："
              '{"description":"参考帧真值标注规范","prompts":[{"label":"类别名","text":"SAM 3 英文关键词"}]}。'
              "\n8. 分别生成完整描述和 SAM 3 关键词："
              "description 给复核 Agent 使用，描述真值中的目标、实例归属、可见边界、遮挡及排除范围；"
              "description must contain exactly two complete English sentences, use at most 360 characters total, and end with a period;"
              "不得在连接词或未完成短语处截断；"
              "只描述已标注类别，不为图中未标注的物体新增任务。"
              "prompts 给 SAM 3 分割使用，每个类别一个 text，仅 1～4 个英文关键词（名词短语），"
              "例如 floor、wooden pallet；不得写完整句子、排除项或边界规则。"
              "每个允许类别必须有且只有一个 prompt。"
              f"\n允许类别：{json.dumps(labels, ensure_ascii=False)}"
              + extended_note
              + "\n密集轮廓的几何文本可能是边界范围摘要；精确边界以单独的参考轮廓图为准。"
              f"\n真值：{json.dumps(reference_geometry(annotations), ensure_ascii=False)}")
    if config.get("llm_prompt_profile") == "compact":
        prompt = (COMPACT_REFERENCE_RULES
            + '\n图片1是参考原图，图片2是洋红色参考轮廓。只返回JSON对象，包含description和prompts两个字段。'
              'description用英文描述每个类别在图中的实际对象、可见边界、实例归属和排除范围，不要复述提示原则。'
              'description must contain exactly two complete English sentences, use at most 360 characters total, and end with a period;'
              '不得在连接词或未完成短语处截断。'
              'prompts是对象数组，每项包含label和text。label必须逐字使用允许类别，每个类别各一项，不能合并类别。'
              'text为对应类别的1到4个英文关键词，供SAM3分割使用。'
            + extended_note
            + '\nInput: ' + json.dumps({"labels": labels, "unreferenced_labels": unreferenced,
                                        "reference_geometry": reference_geometry(annotations)},
                                       ensure_ascii=False, separators=(",", ":")))
    last_error, last_result = None, None
    for attempt in range(MAX_PROTOCOL_RETRIES + 1):
        instruction = prompt
        if attempt:
            instruction += ("\n上一份输出未通过格式检查，请修正后重新返回完整JSON。错误：" + str(last_error)
                            + "\n上次输出：" + json.dumps(last_result, ensure_ascii=False)[:4000])
        try:
            options = {}
            if config.get("llm_prompt_profile") == "compact":
                from .protocol_schema import description_schema
                options["response_schema"] = description_schema(labels)
            result = chat(config, instruction, image, extra_images=(marked_image,) if marked_image else (), **options)
        except ModelOutputError as error:
            last_error, last_result = error, {"raw_text": error.raw_text}
            if include_unreferenced and not annotations:
                try:
                    return canonical_unreferenced_description(labels)
                except ValueError:
                    pass
            continue
        try:
            restricted = restrict_description(result, labels)
            restricted["description"] = require_complete_reference_description(
                restricted["description"])
            return restricted
        except ValueError as error:
            last_error, last_result = error, result
            if include_unreferenced and not annotations:
                try:
                    return canonical_unreferenced_description(labels)
                except ValueError:
                    pass
    if annotations:
        try:
            return canonical_reference_geometry_description(annotations, labels)
        except ValueError as fallback_error:
            raise ValueError(
                f"参考规范多次未通过协议校验：{last_error}；"
                f"确定性兜底也不可用：{fallback_error}") from fallback_error
    raise ValueError(f"参考规范多次未通过协议校验：{last_error}")


REPAIR_PRINCIPLES = """
复查修复原则：
1. 复查阶段允许使用 SAM 3，但只能对现有候选的局部问题使用 sam_points，逐步提供当前帧坐标的正点和负点；历史提示点会累计，现有 mask 会作为输入，polygon 指定允许修改的局部区域。禁止使用 sam_text 对整幅图像重新分割，也不得用空 shape_ids 创建全新整图候选。
2. SAM 3 点提示结果必须由下一轮 GPT 与修改前候选比较后确认是否改进；有待评估结果时，必须输出 sam_assessment:{"improved":true或false,"reason":"比较依据"}。只有缺陷已改善且正确区域未受损才接受；无改进、退化或不确定时 improved:false，Agent 自动恢复上一版本，并使用本轮的 OpenCV 操作进行精细调整，不再为该候选重复 SAM 3。此时操作坐标针对恢复后的版本；不能仅因 SAM 3 返回成功或分数较高就接受结果。
3. 保留已正确标注的区域，只针对有证据的局部缺陷修复；不得为修复局部边界删除整个候选再重建。remove 仅用于整个实例均为误检；merge、split 仅用于确有参考规范依据的实例归属错误。
4. 每项操作的 reason 须说明待处理帧的具体缺陷、对应参考真值证据和预期变化；不得仅凭参考帧与当前帧坐标或面积不同认定错误，不得将标线、阴影或颜色变化自动视为排除区域。
5. 对地面框线逐段检查：如果原图中能确认存在未被物体遮挡的地面框线或标线，必须检查当前候选 mask 是否覆盖其可见部分；发现漏标时使用局部 sam_points 或 add_region 修复，不能返回 done。线条颜色不能作为扣除依据。
6. polygon 和点提示必须使用当前帧原始像素坐标，修改范围限于需要修复的区域；polygon 必须至少包含 3 个不同且能围成非零面积区域的顶点；不得重复提交同一个提示点，也不得在同一坐标给出相互冲突的正负标签；不得用大范围粗略区域破坏正确边界、孔洞或其他区域。
7. 复查没有“人工复核”出口：每轮只能选择修改候选，或在没有可靠修改依据时保持候选不变并结束。
   不得输出 needs_review 状态或 review 操作；无法可靠判断或准确定位时，不修改候选并返回 done。
"""


TOOLS = """
输出严格 JSON：{"status":"continue|done","reason":"判断依据","hole_assessment":[{"hole_id":"缺口ID","semantic":"ground|non_ground","reason":"原图与参考真值依据"}],"operations":[...]}。
operations 工具（每项必须有 operation 和 reason）：
- remove: shape_ids:[一个整数]，删除误检。
- reclassify: shape_ids:[一个整数], label:有效类别。
- add_region / erase_region: shape_ids:[一个整数], polygon:[[x,y],...]，OpenCV 补区域/擦区域。
- fill_gap: shape_ids:[一个整数], hole_id:"缺口ID", polygon:[[x,y],...]，用矢量多边形补指定封闭空洞或开放缺口；Agent 会将多边形与该缺口的实际像素区域求交，只能填该缺口内像素。
- merge: shape_ids:[至少两个整数], label:有效类别，OpenCV 合并同一实例的区域。
- split: shape_ids:[一个整数], polygon:[[x,y],...]，用区域将一个实例分成区域内和区域外两个实例。
- sam_points: shape_ids:[一个整数], polygon:[[x,y],...], points:[{"x":像素x,"y":像素y,"label":1或0},...]。每次 1～32 个新增或修正提示点，累计提示必须含正点；polygon 外保留原 mask。执行后本轮结束，下一轮 GPT 必须先评估是否改进。
- 不提供 review 或 needs_review 操作；无法确认时 operations 返回空数组并使用 status=done。
每轮最多 32 个操作。图片 1 是原图，图片 2 是当前带 ID 的标注，图片 3 是当前内部空洞编号图；坐标用原始像素。
对下方提供的每一个封闭空洞或开放缺口都必须逐一输出 hole_assessment，语义只能是 ground 或 non_ground，并说明原图证据及参考帧真值依据；不能跳过任何 ID。图像 3 中的编号对应 hole_id。
属于地面的空洞或开放缺口必须用 fill_gap 的矢量多边形逐块补上；属于机器人、设备或其他非地面物体的区域保持不填。不要用一个宽泛多边形覆盖多个语义不同区域。
只有在全部空洞均已逐一判断，且所有 ground 空洞均已修补后才可返回 done。不得因整体轮廓看似合理而忽略内部空洞。操作只修改候选，不改原图或参考帧。
只能使用上述工具；每次要么执行有依据的修改，要么保持候选不变并返回 done。
当前结果已符合规范，或没有可由参考真值支持的安全修改时，返回 done 且不做修改；任何编辑后必须等下一轮重新观察。
不能仅因工具执行成功或分数高而认为标注正确。工具失败时依据错误调整操作。
"""


def reference_geometry(annotations):
    """Bound dense coordinate text; exact contours remain in the reference image."""
    # Simple reference polygons are small enough to include in full.
    if len(json.dumps(annotations, ensure_ascii=False)) <= 10000:
        return annotations
    def bounds(points):
        values = np.asarray(points, dtype=np.int32)
        return {"bbox": [int(values[:, 0].min()), int(values[:, 1].min()),
                          int(values[:, 0].max()), int(values[:, 1].max())],
                "vertices": int(len(values))}

    # The marked reference image carries the exact boundary.  For very dense
    # hand-drawn contours, send compact geometry facts instead of tens of
    # thousands of repeated coordinates so the vision request remains usable.
    return [{"label": shape["label"], "parts": [
        {"boundary": bounds(part["points"]),
         "holes": [bounds(hole) for hole in part.get("holes", [])]}
        for part in shape.get("parts") or [shape]]} for shape in annotations]


def _gap_bbox_polygon(hole, width, height):
    """Return a valid rectangle covering one trusted raster gap bounding box."""
    bbox = hole.get("bbox") if isinstance(hole, dict) else None
    if (not isinstance(bbox, list) or len(bbox) != 4
            or any(type(value) not in (int, float) or not math.isfinite(value)
                   for value in bbox)):
        return None
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    left, right = max(0, min(x1, x2)), min(width - 1, max(x1, x2))
    top, bottom = max(0, min(y1, y2)), min(height - 1, max(y1, y2))
    # One-pixel-wide morphology gaps are still valid masks. Expand only the
    # rectangle used to address them; apply_operations clips the edit back to
    # the exact reviewed gap mask before changing any candidate pixel.
    if left == right:
        if right < width - 1:
            right += 1
        elif left > 0:
            left -= 1
    if top == bottom:
        if bottom < height - 1:
            bottom += 1
        elif top > 0:
            top -= 1
    polygon = [[left, top], [right, top], [right, bottom], [left, bottom]]
    return polygon if _valid_region(polygon, width, height) else None


def _synthesize_explicit_ground_fills(result, holes, explicit_ground, width, height):
    """Complete only exact gaps the model explicitly assessed as ground.

    The semantic decision remains the model's. Geometry is deterministic and
    bounded by the exact raster gap mask inside ``apply_operations``. Omitted,
    malformed, conflicting and non-ground assessments never reach this path.
    """
    operations = result.get("operations")
    if not isinstance(operations, list) or not explicit_ground:
        return
    filled = {item.get("hole_id") for item in operations
              if isinstance(item, dict) and item.get("operation") == "fill_gap"}
    pending = [hole for hole in holes
               if hole.get("id") in explicit_ground and hole.get("id") not in filled]
    # Preserve the global protocol ceiling. The normal review path selects at
    # most four gaps, but fail closed if a caller ever provides a larger set.
    if len(operations) + len(pending) > 32:
        return
    synthesized = []
    for hole in pending:
        polygon = _gap_bbox_polygon(hole, width, height)
        if polygon is None or type(hole.get("shape_id")) is not int:
            continue
        operations.append({
            "operation": "fill_gap",
            "shape_ids": [hole["shape_id"]],
            "hole_id": hole["id"],
            "polygon": polygon,
            "reason": "Explicit ground assessment; fill is clipped to the reviewed gap mask",
        })
        synthesized.append({
            "reason_code": "explicit_ground_fill_synthesized",
            "hole_id": hole["id"],
            "shape_id": hole["shape_id"],
            "source": "vlm_hole_assessment",
            "geometry_guard": "exact_gap_mask_intersection",
        })
    if synthesized:
        result.setdefault("protocol_normalizations", []).extend(synthesized)


def _validate_review_result(result, holes, shapes, labels, width, height, sam_comparison=None,
                            reference_provided=True, synthesize_ground_fills=False):
    if not isinstance(result, dict):
        raise ValueError("大模型必须返回 JSON 对象")
    # Validate a copy: traces and correction prompts must retain the actual reply.
    result = copy.deepcopy(result)
    operations = result.get("operations", [])
    if not isinstance(operations, list) or len(operations) > 32:
        raise ValueError("大模型 operations 必须是数组且不能超过每轮 32 项限制")
    # Keep compatibility with older local prompts/models that only returned
    # operations. An empty operation list is a completed review; edits require
    # another observation before the result can be considered done.
    if "status" not in result:
        result["status"] = "done" if not result.get("operations") else "continue"
    if result.get("status") == "needs_review":
        result["status"] = "continue" if any(
            isinstance(item, dict) and item.get("operation") != "review"
            for item in operations) else "done"
    result["operations"] = [item for item in operations
                             if not (isinstance(item, dict) and item.get("operation") == "review")]
    errors = []
    if result["status"] not in ("done", "continue"):
        errors.append("大模型未返回有效的 agent 状态")
    shapes_by_id = {item["id"]: item for item in shapes}
    shape_ids = set(shapes_by_id)
    hole_ids = {item["id"]: item for item in holes}
    retained_operations = []
    for operation in result["operations"]:
        ids = operation.get("shape_ids") if isinstance(operation, dict) else None
        if (isinstance(operation, dict) and operation.get("operation") == "reclassify"
                and isinstance(ids, list) and len(ids) == 1 and ids[0] in shapes_by_id
                and operation.get("label") == shapes_by_id[ids[0]].get("label")):
            result.setdefault("protocol_normalizations", []).append({
                "reason_code": "no_op_reclassify_dropped",
                "shape_id": ids[0],
                "label": operation.get("label"),
            })
            continue
        if (isinstance(operation, dict) and operation.get("operation") == "remove"
                and isinstance(ids, list) and ids
                and all(shape_id in shape_ids for shape_id in ids)):
            reason = str(operation.get("reason", "")).casefold()
            negative_evidence = (
                "false positive", "not visible", "not present", "no visible",
                "absent", "wrong class", "wrong label", "wrong instance",
                "misclassified", "duplicate", "background artifact",
                "误检", "不存在", "不可见", "类别错误", "重复",
            )
            source_labels = {
                str(shapes_by_id[shape_id].get("label", "")).casefold().strip()
                for shape_id in ids if shape_id in shapes_by_id
            }
            source_label_negative = any(
                label and (
                    re.search(
                        rf"(?:^|\b)not\s+(?:(?:a|an|the)\s+)?{re.escape(label)}(?:\b|$)",
                        reason,
                    ) is not None
                    or f"不是{label}" in reason
                    or f"并非{label}" in reason
                )
                for label in source_labels
            )
            if (not source_label_negative
                    and not any(marker in reason for marker in negative_evidence)):
                result.setdefault("protocol_normalizations", []).append({
                    "reason_code": "remove_without_negative_evidence_dropped",
                    "shape_ids": copy.deepcopy(ids),
                    "reason": operation.get("reason"),
                })
                continue
        retained_operations.append(operation)
    result["operations"] = retained_operations
    for index, operation in enumerate(result["operations"], 1):
        if not isinstance(operation, dict):
            errors.append(f"第 {index} 项操作必须是 JSON 对象")
            continue
        kind = operation.get("operation")
        if kind not in ("remove", "reclassify", "add_region", "erase_region",
                        "fill_gap", "merge", "split", "sam_points"):
            errors.append(f"第 {index} 项操作名无效：{kind}；只能使用工具清单中的操作，"
                          "填缺口应使用 fill_gap，并提供 shape_ids、完整 hole_id 和 polygon")
        if not isinstance(operation.get("reason"), str) or not operation["reason"].strip():
            errors.append(f"第 {index} 项操作缺少 reason")
        if not reference_provided and kind == "remove":
            errors.append(f"第 {index} 项 remove 操作被拒绝：本轮没有人工参考帧，"
                          "必须保留潜在候选并转人工复核，不能删除整个实例")
        ids = operation.get("shape_ids")
        valid_ids = (isinstance(ids, list) and (len(ids) >= 2 if kind == "merge" else len(ids) == 1)
                     and all(type(shape_id) is int and shape_id in shape_ids for shape_id in ids)
                     and len(set(ids)) == len(ids))
        if not valid_ids:
            errors.append(f"第 {index} 项 {kind} 操作必须提供有效的 shape_ids；可用 ID：{sorted(shape_ids)}")
        if kind in ("merge", "reclassify") and operation.get("label") not in labels:
            errors.append(f"第 {index} 项 {kind} 操作缺少有效 label；可用类别：{labels}")
        if kind in ("add_region", "erase_region", "split", "fill_gap", "sam_points"):
            if not _valid_region(operation.get("polygon"), width, height):
                errors.append(f"第 {index} 项 {kind} 操作缺少有效 polygon："
                              f"需要至少 3 个彼此不同且不共线、能围成非零面积的 [x,y] 顶点，"
                              f"不得重复同一坐标，0≤x<{width}，0≤y<{height}；"
                              "若无法可靠给出局部区域，请删除该操作；没有其他可靠修改时返回 done 和空 operations")
        if kind == "fill_gap":
            hole_id = operation.get("hole_id")
            if not isinstance(hole_id, str) or hole_id not in hole_ids:
                errors.append(f"第 {index} 项 fill_gap 操作缺少有效 hole_id；"
                              f"请使用清单中的完整 ID：{list(hole_ids)}")
            elif valid_ids and hole_ids[hole_id]["shape_id"] != ids[0]:
                errors.append(f"第 {index} 项 fill_gap 的 hole_id 不属于指定的 shape_ids")
        if kind == "sam_points":
            points = operation.get("points")
            if not isinstance(points, list) or not 1 <= len(points) <= 32 or any(
                not isinstance(point, dict) or type(point.get("label")) is not int
                or point["label"] not in (0, 1)
                or any(type(point.get(key)) not in (int, float) or not math.isfinite(point[key])
                       or not 0 <= point[key] < (width if key == "x" else height)
                       for key in ("x", "y")) for point in points):
                errors.append(f"第 {index} 项 sam_points 操作缺少有效 points")
            elif points:
                labels_by_coordinate = {}
                for point in points:
                    coordinate = (float(point["x"]), float(point["y"]))
                    labels_by_coordinate.setdefault(coordinate, set()).add(point["label"])
                if any(len(point_labels) > 1 for point_labels in labels_by_coordinate.values()):
                    errors.append(f"第 {index} 项 sam_points 在同一坐标给出了冲突的正负标签")
                if len(points) > 1 and len(labels_by_coordinate) == 1:
                    errors.append(f"第 {index} 项 sam_points 只有重复的同一个点，不能视为有效修正")
                if not any(point["label"] == 1 for point in points):
                    errors.append(f"第 {index} 项 sam_points 至少需要一个有效正点")
    actionable_ground = set()
    if holes:
        assessment = result.get("hole_assessment")
        expected = {item["id"] for item in holes}
        if not isinstance(assessment, list):
            errors.append("GPT 未逐一判断当前候选的内部空洞；必须为清单中的每个完整 hole_id "
                          "输出含 semantic 和 reason 的判断")
            assessment = []
        # Some local-model responses repeat one assessment byte-for-byte while
        # omitting another ID. An exact duplicate adds no semantic information,
        # so remove it before the existing conservative missing-ID repair. Any
        # disagreement for the same ID remains malformed and keeps the normal
        # protocol retry path.
        unique_assessment = []
        assessment_by_id = {}
        for item in assessment:
            hole_id = item.get("hole_id") if isinstance(item, dict) else None
            previous = assessment_by_id.get(hole_id) if isinstance(hole_id, str) else None
            if previous is not None and item == previous:
                result.setdefault("protocol_normalizations", []).append({
                    "reason_code": "duplicate_hole_assessment_dropped",
                    "hole_id": hole_id,
                })
                continue
            unique_assessment.append(item)
            if isinstance(hole_id, str) and previous is None:
                assessment_by_id[hole_id] = item
        assessment = unique_assessment
        observed_items = [item.get("hole_id") for item in assessment
                          if isinstance(item, dict) and isinstance(item.get("hole_id"), str)]
        observed = set(observed_items)
        malformed = (any(not isinstance(item, dict) for item in assessment)
                     or len(observed_items) != len(assessment)
                     or len(observed_items) != len(observed)
                     or any(item.get("semantic") not in ("ground", "non_ground")
                            or not isinstance(item.get("reason"), str)
                            or not item["reason"].strip()
                            for item in assessment if isinstance(item, dict)))
        explicit_ground = {item["hole_id"] for item in assessment
                           if isinstance(item, dict)
                           and isinstance(item.get("hole_id"), str)
                           and item.get("semantic") == "ground"}
        missing = expected - observed
        unknown = observed - expected
        # A local VLM can occasionally omit one region while returning an
        # otherwise valid structured decision.  Never guess that the omitted
        # area is ground: preserve it unchanged, make the repair auditable and
        # route it to human review.  Unknown/duplicate/malformed IDs remain a
        # hard protocol error, as do attempts to fill an omitted region.
        if (assessment and missing and not unknown and not malformed
                and observed.issubset(expected)):
            repairs = result.setdefault("protocol_repairs", [])
            for hole_id in sorted(missing):
                assessment.append({
                    "hole_id": hole_id,
                    "semantic": "non_ground",
                    "reason": "Model omitted this region; preserved for human review",
                })
                repairs.append({
                    "reason_code": "vlm_hole_assessment_omitted",
                    "hole_id": hole_id,
                    "route": "human_review_required",
                })
            observed = expected
            missing = set()
        if (len(assessment) != len(expected) or malformed or observed != expected):
            errors.append("GPT 空洞判断必须覆盖全部 hole_id，并包含语义和依据；"
                          f"缺失 ID：{sorted(missing)}；"
                          f"无效 ID：{sorted(unknown)}。"
                          "每个完整 ID 恰好一次，禁止用 #5 或图片编号替代完整 hole_id；"
                          "semantic 必须为 ground/non_ground，reason 必须为非空字符串")
        if (synthesize_ground_fills and reference_provided and not errors
                and not malformed and not unknown):
            _synthesize_explicit_ground_fills(
                result, holes, explicit_ground, width, height)
        filled = {item.get("hole_id") for item in result["operations"]
                  if isinstance(item, dict) and item.get("operation") == "fill_gap"
                  and isinstance(item.get("hole_id"), str)}
        ground = {item["hole_id"] for item in assessment if isinstance(item, dict)
                  and isinstance(item.get("hole_id"), str) and item.get("semantic") == "ground"}
        actionable_ground = ground - filled
        if filled - ground:
            errors.append("GPT 只能修补判定为 ground 的空洞；禁止对 non_ground 使用 fill_gap："
                          f"{sorted(filled - ground)}")
        if not ground.issubset(filled) and result["status"] == "done":
            errors.append(f"GPT 不能在仍有属于地面的未修补空洞时返回 done：{sorted(ground - filled)}")
        result["hole_assessment"] = assessment
    if sam_comparison is not None:
        assessment = result.get("sam_assessment")
        if (not isinstance(assessment, dict) or type(assessment.get("improved")) is not bool
                or not isinstance(assessment.get("reason"), str) or not assessment["reason"].strip()):
            errors.append('GPT 未评估 SAM 3 点提示结果；必须提供 sam_assessment:'
                          '{"improved":true或false,"reason":"比较依据"}')
    elif "sam_assessment" in result:
        errors.append("当前没有待比较的 SAM 3 点提示结果，禁止声明 improved")
    if errors:
        raise ValueError("\n".join(errors))
    # Some local models emit `continue` even after returning no operation and
    # no ground gap left to repair.  Treat that self-contradictory tail as a
    # completed review instead of producing a false no-progress failure.
    if (result["status"] == "continue" and not result["operations"]
            and not actionable_ground and sam_comparison is None):
        result["status"] = "done"
        result.setdefault("protocol_normalizations", []).append({
            "reason_code": "empty_continue_normalized_to_done",
            "from": "continue",
            "to": "done",
        })
    return result


_POLYGON_OPERATIONS = {"add_region", "erase_region", "split", "fill_gap", "sam_points"}


def _degenerate_polygon_only(result, width, height, holes, shapes, labels,
                             sam_comparison=None, reference_provided=True):
    """Return true only when every requested edit is a zero-area polygon.

    Missing fields, out-of-bounds coordinates, mixed tool requests and malformed
    JSON keep the full protocol retry budget.  The bounded early-stop path is
    reserved for a repeated, structurally valid list of degenerate polygons.
    """
    operations = result.get("operations") if isinstance(result, dict) else None
    if not isinstance(operations, list) or not operations:
        return False
    repaired = copy.deepcopy(result)
    for operation in repaired["operations"]:
        if not isinstance(operation, dict) or operation.get("operation") not in _POLYGON_OPERATIONS:
            return False
        points = operation.get("polygon")
        if not isinstance(points, list) or not 3 <= len(points) <= 2000:
            return False
        if not all(isinstance(point, list) and len(point) == 2 and all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) for value in point)
                and 0 <= point[0] < width and 0 <= point[1] < height
                for point in points):
            return False
        unique = {(float(point[0]), float(point[1])) for point in points}
        if len(unique) >= 3 and abs(float(cv2.contourArea(
                np.asarray(points, dtype=np.float32)))) >= 0.5:
            return False
        if width < 2 or height < 2:
            return False
        operation["polygon"] = [[0, 0], [1, 0], [0, 1]]
    # A repaired synthetic triangle must make the whole response valid. This
    # keeps missing IDs, reasons, assessments, point prompts and other protocol
    # errors on the normal three-attempt path instead of hiding them behind the
    # repeated-polygon shortcut.
    try:
        _validate_review_result(repaired, holes, shapes, labels, width, height,
                                sam_comparison,
                                reference_provided=reference_provided)
    except ValueError:
        return False
    return True


def _review_exact_overlap(config, image, overlay, group, shapes, labels):
    """Ask the VLM about one identical-mask group before the general review."""
    source = Image.open(io.BytesIO(image)).convert("RGB")
    marked = Image.open(io.BytesIO(overlay)).convert("RGB")
    by_id = {shape["id"]: shape for shape in shapes}
    boxes = [by_id[identifier]["bbox"] for identifier in group["shape_ids"]]
    left, top = min(box[0] for box in boxes), min(box[1] for box in boxes)
    right, bottom = max(box[2] for box in boxes), max(box[3] for box in boxes)
    padding = max(32, round(max(right-left, bottom-top)*.15))
    crop = (max(0,left-padding),max(0,top-padding),
            min(source.width,right+padding+1),min(source.height,bottom+padding+1))
    def encode(picture):
        output = io.BytesIO()
        picture.crop(crop).save(output,"JPEG",quality=92)
        return output.getvalue()
    original_crop = encode(source)
    presence_schema = {"type":"object","properties":{
        "evidence":{"type":"string","enum":["present","absent","uncertain"]},
        "reason":{"type":"string","minLength":1,"maxLength":120}},
        "required":["evidence","reason"],"additionalProperties":False}
    presence = chat(config,
        f"Observe this uncolored photograph crop. Is a {group['label']!r} visibly present?"
        " Return a JSON object with evidence and reason. evidence is present, absent, or uncertain."
        " Use only the actual photograph. A blank field or uniform color contains no visible object."
        " Do not infer an object from the class name in the question. reason is one short English sentence.",
        original_crop,response_schema=presence_schema)
    if (not isinstance(presence,dict) or presence.get("evidence") not in {"present","absent","uncertain"}
            or not isinstance(presence.get("reason"),str) or not presence["reason"].strip()):
        raise ValueError("重复候选原图检查未返回有效的 evidence 和 reason")
    if presence["evidence"] != "present":
        return None
    schema = {"type":"object","properties":{
        "decision":{"type":"string","enum":["merge","keep"]},
        "reason":{"type":"string","minLength":1,"maxLength":180}},
        "required":["decision","reason"],"additionalProperties":False}
    instruction = ("Inspect a possible duplicate annotation, using these two crops of the SAME target image."
        " Image 1 is uncolored; image 2 shows the candidate masks. Overlay colors are not object colors."
        f" Candidate IDs {group['shape_ids']} all have label {group['label']!r} and pixel-identical masks."
        " Decide whether these IDs annotate one visible object redundantly or represent separate objects."
        " Return JSON with decision and reason. decision='merge' only when the same single visible object"
        " has been annotated more than once. Otherwise, including uncertain cases, decision='keep'."
        " A merge preserves the entire mask and its class. Do not change any other object."
        " Give one short English sentence of visible evidence in reason. No coordinates or other fields.")
    result = chat(config,instruction,original_crop,extra_images=(encode(marked),),response_schema=schema)
    if not isinstance(result,dict) or result.get("decision") not in {"merge","keep"} or not isinstance(result.get("reason"),str) or not result["reason"].strip():
        raise ValueError("重复候选复核未返回有效的 decision 和 reason")
    if result["decision"] == "keep":
        return None
    decision = {"status":"continue","reason":result["reason"],"hole_assessment":[],
                "operations":[{"operation":"merge","shape_ids":group["shape_ids"],
                               "label":group["label"],"reason":result["reason"]}]}
    return _validate_review_result(decision,[],shapes,labels,source.width,source.height)


def review(config, image, overlay, description, shapes, labels, history, reference=None,
           sam_comparison=None, hole_image=None, holes=None, gap_images=None, duplicate_groups=None):
    # Enabled only by the isolated evaluation tool while quality tests continue.
    if not config.get("llm_duplicate_review", False):
        duplicate_groups = None
    if (config.get("llm_prompt_profile") == "compact" and duplicate_groups
            and not holes and sam_comparison is None):
        focused = _review_exact_overlap(config,image,overlay,duplicate_groups[0],shapes,labels)
        if focused is not None:
            return focused
    width, height = Image.open(io.BytesIO(image)).size
    class_scope = ("只能标注参考帧真值中已有的类别；即使图中可见其他物体，也不得为其创建标注。"
                   if reference is not None else
                   "本次没有人工参考帧，只能标注用户明确选择的类别；输出仍是机器草稿，"
                   "不得声称已核对人工真值，也不得创建未选择的新类别。")
    evidence_principles = REFERENCE_PRINCIPLES if reference is not None else (
        """
无人工参考帧候选原则：
1. 本轮只有当前帧原图与机器候选，不存在参考图、参考轮廓或人工真值证据。
2. 不得声称看到了参考轮廓、参考 mask、参考类别或人工真值；不得用虚构的参考证据修改候选。
3. 无参考帧时禁止 remove 整个候选；疑似误检也必须保留并路由人工复核，避免删除真实小目标或人员。
4. 允许有当前帧像素证据的局部边界修复；证据不足时保持候选并返回 done。
""")
    rule_name = "参考帧真值标注规范" if reference is not None else "无参考帧机器候选规则"
    prompt = (TOOLS + evidence_principles + REPAIR_PRINCIPLES
        + "\n" + class_scope
        + "\n你正在复核当前候选（首轮为 SAM 3 初步分割，后续轮次可能经过 SAM 3 点提示或 OpenCV 修改）。比较原始帧、带候选轮廓/ID 的图像及下列规则，"
        "找出漏标、误检、边界偏差和类别错误，返回具体的工具操作；必要时多轮重新观察。"
        + f"\n图像尺寸：{width} x {height}"
        f"\n{rule_name}：{description.get('description', '')}"
        f"\nSAM 3 类别关键词：{json.dumps(description['prompts'], ensure_ascii=False)}"
        f"\n类别：{json.dumps(labels, ensure_ascii=False)}"
        f"\n当前候选（ID、类别、框、顶点数）：{json.dumps(shapes, ensure_ascii=False)}"
        f"\n历史决策及工具结果（最近 8 轮）：{json.dumps(history[-8:], ensure_ascii=False)}")
    holes = holes or []
    synthesize_ground_fills = bool(
        config.get("agent_synthesize_ground_fills", True)
        and reference is not None)
    hole_report = [{**{key: value for key, value in item.items() if key != "mask"},
                    "number": index} for index, item in enumerate(holes, 1)]
    prompt += f"\n本轮从当前候选 mask 提取的封闭空洞和开放缺口清单：{json.dumps(hole_report, ensure_ascii=False)}"
    prompt += ("\n开放缺口只是几何候选，绝不自动等同于地面。即使内部空洞清单为空，"
               "也必须检查与设备排除区域相连的标线漏标。图像 3 左半是原图加缺口，"
               "右半是二值地面 mask：白=已包含，黑=未包含，粉色=待判断缺口（也尚未包含）。"
               "左右两半均对应整幅当前帧，编号与清单 number 对应；输出多边形仍用原始帧坐标。"
               "若同一缺口同时包含地面标线和设备边缘，只用局部多边形补地面部分，保留设备。"
               "不能仅因缺口连通机器人或设备，就把整条可见地面标线判成非地面。")
    extra_images = (overlay, hole_image or overlay)
    if reference is not None:
        prompt += ("\n图像 4 是未着色的原始参考帧，图像 5 只用洋红色轮廓和编号标出参考帧真值。"
                   "图像 1、2、3 是本轮待处理帧，与图像 4、5 可以有不同位置、形状和实例数量；"
                   "不得直接照搬参考帧坐标。真值几何数据使用参考帧自身的原始像素坐标。"
                   "文字描述是对参考真值的总结；如与参考真值证据冲突，以参考真值为准。"
                   "每个修改须在 reason 中说明对应的参考标注证据及待处理帧问题；"
                   "无法可靠判断归属时不要修改并返回 done，不得输出 review 或 needs_review，不得无依据地增加排除区域。"
                   "密集参考轮廓的数值数据仅摘要为 boundary/holes 的 bbox 与顶点数，bbox 不是标注多边形；"
                   "精确形状及包含/排除关系必须对照图像 4 原图和图像 5 的真值轮廓，不能按矩形框判断归属。"
                   f"\n参考帧真值几何数据：{json.dumps(reference_geometry(reference['annotations']), ensure_ascii=False)}")
        extra_images += (reference["image"], reference["marked_image"])
    else:
        prompt += "\n本次未提供参考帧图像，不能声称已核对参考图；涉及无法确认的归属时不要修改并返回 done。"
    if sam_comparison is not None:
        image_number = len(extra_images) + 2
        prompt += (f"\n待评估 SAM 3 点提示结果：shape ID {sam_comparison['shape_id']}。"
                   f"图像 2 是点提示修改后候选；图像 {image_number} 是修改前候选，用于比较是否改进。"
                   "必须输出 sam_assessment；若 improved:false，Agent 将恢复图像中修改前版本，"
                   "本轮使用 OpenCV 局部操作进行精细调整；若无法可靠定位则不要修改并返回 done。")
        extra_images += (sam_comparison["before_image"],)
    for sheet, reports in gap_images or []:
        number = len(extra_images) + 2
        prompt += (f"\n图像 {number} 是缺口局部证据页，每行从左到右为未着色原图、"
                   "二值 mask（白=已包含/黑=未包含）、同一 mask 加粉色待判断缺口。"
                   "每行顶部标出清单编号；这些裁剪图经过缩放，不能直接用图中显示坐标作为操作坐标。"
                   f"各行对应原始帧裁剪范围（左上右下）：{json.dumps(reports, ensure_ascii=False)}。"
                   "必须对照未着色原图识别标线，再检查同位置二值图是否为白色；"
                   "黑色或粉色都表示尚未纳入地面，不能声称已完整包含。")
        extra_images += (sheet,)
    if config.get("llm_prompt_profile") == "compact":
        floor_task = any(label.strip().lower() in {"地面", "floor", "ground"} for label in labels)
        if not floor_task and not holes:
            extra_images = (overlay,)
            if reference is not None:
                extra_images += (reference["image"], reference["marked_image"])
            if sam_comparison is not None:
                extra_images += (sam_comparison["before_image"],)
            extra_images += tuple(sheet for sheet, _ in gap_images or [])
        payload = {"target_size": [width, height], "labels": labels,
                   "class_definitions": {label: VISIBLE_LABEL_DEFINITIONS[label]
                                         for label in labels if label in VISIBLE_LABEL_DEFINITIONS},
                   "rules": description.get("description", ""),
                   "prompts": description["prompts"], "candidates": shapes, "recent_history": history[-3:],
                   "holes": hole_report, "reference_provided": reference is not None}
        if reference is not None and floor_task:
            payload["reference_geometry"] = reference_geometry(reference["annotations"])
        base_image_count = 1 + len(extra_images) - int(sam_comparison is not None) - len(gap_images or [])
        if sam_comparison is not None:
            payload["pending_sam_shape"] = sam_comparison["shape_id"]
            payload["sam_before_image"] = base_image_count + 1
        if gap_images:
            first = base_image_count + 1 + int(sam_comparison is not None)
            payload["gap_details"] = [{"image": first+i, "original_crop_ranges": reports}
                                      for i, (_, reports) in enumerate(gap_images)]
        compact_rules = COMPACT_REFERENCE_RULES if reference is not None else COMPACT_UNREFERENCED_RULES
        prompt = compact_rules + COMPACT_REVIEW_TOOLS + "\nInput: " + json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"))
        if not floor_task and not holes:
            prompt += "\n本轮只标注提供的物体类别，不标注地面。图片1是目标原图，图片2是目标候选；"
            if reference is not None:
                prompt += ("图片3、4分别是参考原图和参考轮廓，覆盖此前通用图号说明。"
                           "参考图只提供类别语义和标注边界规范；不得统计或照搬参考图实例数量，"
                           "目标图数量不同完全正常，不能因此删除、增加或修改候选。")
            else:
                prompt += ("没有图片3或图片4，也没有任何参考原图、参考轮廓或人工真值。"
                           "禁止提及不存在的参考轮廓；禁止 remove，疑似误检保留给人工复核。")
            prompt += "holes列表为空，hole_assessment必须是空数组，禁止编造孔洞编号。"
        prompt += ("\n每个reason最多一句话且不超过100个英文字符，每轮最多4个有视觉证据的局部操作。仅允许使用candidates中的ID。"
                   "所有reason使用英文。无法可靠定位缺陷时保持候选，返回done且operations为空。不要复述所有实例，不要输出额外解释。")
        if synthesize_ground_fills and holes:
            prompt += ("\n对holes只输出逐项semantic和简短reason，不要输出fill_gap操作或缺口polygon；"
                       "系统会把semantic=ground的精确缺口mask一次性填满，并保持non_ground不变。"
                       "operations仅用于缺口以外且有视觉证据的局部修正。")
    if duplicate_groups:
        prompt += ("\nExact same-class raster-mask overlap groups: "
                   + json.dumps(duplicate_groups, separators=(",", ":"))
                   + "\nEach listed group has pixel-identical masks and the same label. This is a geometry observation,"
                     " not a new reference annotation. Check whether its IDs describe one visible object twice."
                     " If so, remove one redundant candidate or merge the IDs, preserve the visible mask and other instances,"
                     " then return continue to inspect the result in the next round."
                     " Do not infer duplicates from equal boxes or different-frame counts."
                     " Return a concrete operation only when the target image supports the duplicate interpretation.")
    last_result = None
    last_error = None
    previous_degenerate_error = None
    repeated_degenerate_stop = False
    attempts_used = 0
    base_review_tokens = compact_review_token_budget(config, len(holes))
    retry_max_tokens = None
    for attempt in range(MAX_PROTOCOL_RETRIES + 1):
        if attempt:
            correction = ("\n上一份输出未通过 Agent 协议校验。请保留其中正确的视觉判断，只修正协议错误，"
                "重新输出完整 JSON，不要解释。\n校验错误：" + str(last_error)
                + "\n上一份输出：" + json.dumps(last_result, ensure_ascii=False)[:12000])
            if previous_degenerate_error is not None:
                correction += ("\n这是退化多边形的唯一一次协议纠错机会：每个 polygon 必须改成至少 3 个"
                               "不同且不共线的当前帧坐标；无法可靠定位时删除该操作，"
                               "没有其他可靠修改就返回 done 和空 operations。")
            request_prompt = prompt + correction
        else:
            request_prompt = prompt
        attempts_used = attempt + 1
        call_state = {"attempt": attempt + 1, "feedback": str(last_error) if attempt else None}
        token = _review_call.set(call_state)
        try:
            check_cancelled()
            try:
                options = {}
                if config.get("llm_prompt_profile") == "compact":
                    from .protocol_schema import review_schema
                    options["response_schema"] = review_schema(
                        labels, shapes, holes, sam_comparison is not None,
                        synthesize_ground_fills=synthesize_ground_fills)
                request_config = config
                request_tokens = retry_max_tokens or base_review_tokens
                if request_tokens is not None and request_tokens != config.get("llm_max_tokens"):
                    request_config = copy.deepcopy(config)
                    request_config["llm_max_tokens"] = request_tokens
                last_result = chat(request_config, request_prompt, image,
                                   extra_images=extra_images, **options)
            except ModelOutputError as error:
                last_result = {"raw_text": error.raw_text}
                last_error = error
            else:
                try:
                    return _validate_review_result(
                        last_result, holes, shapes, labels, width, height,
                        sam_comparison, reference_provided=reference is not None,
                        synthesize_ground_fills=synthesize_ground_fills)
                except ValueError as error:
                    last_error = error
            # Update the original call rather than creating a fictitious model
            # request. Transport/configuration failures bypass this retry path.
            record = call_state.get("record")
            current_degenerate_error = (str(last_error)
                if _degenerate_polygon_only(
                    last_result, width, height, holes, shapes, labels,
                    sam_comparison, reference_provided=reference is not None)
                else None)
            repeated_degenerate_stop = (attempt > 0
                and current_degenerate_error is not None
                and current_degenerate_error == previous_degenerate_error)
            if record is not None:
                record.update(status="rejected", error=str(last_error),
                              retrying=attempt < MAX_PROTOCOL_RETRIES and not repeated_degenerate_stop)
                if repeated_degenerate_stop:
                    record["protocol_retry_stop_reason"] = "repeated_degenerate_polygon"
                _emit_trace(record)
            if repeated_degenerate_stop:
                break
            previous_degenerate_error = current_degenerate_error
            finish_reason = (record or {}).get("output", {}).get("finish_reason")
            if (finish_reason == "length"
                    or isinstance(last_error, ModelOutputError)
                    and "unterminated string" in str(last_error).lower()):
                current_budget = retry_max_tokens or base_review_tokens or 768
                retry_max_tokens = min(2048, max(1536, current_budget * 2))
        finally:
            _review_call.reset(token)
    message = (f"GPT 连续 {attempts_used} 次返回相同的退化多边形，已提前停止无进展重试：{last_error}"
               if repeated_degenerate_stop else
               f"GPT 协议校验重试 {MAX_PROTOCOL_RETRIES} 次后仍失败：{last_error}")
    raise ReviewProtocolError(
        message,
        last_error=last_error, last_result=last_result)


def reference_overlay(image, shapes):
    """Outline the reference without recoloring its real appearance."""
    result = Image.open(io.BytesIO(image)).convert("RGB")
    draw = ImageDraw.Draw(result)
    for number, shape in enumerate(shapes, 1):
        for part in shape.get("parts") or [shape]:
            for ring in [part.get("points", []), *(part.get("holes") or [])]:
                points = [tuple(point) for point in ring]
                if len(points) >= 3:
                    draw.line(points+points[:1], fill=(255, 0, 255), width=3)
        points = shape.get("points", [])
        if points:
            draw.text(tuple(points[0]), f"{number} {shape['label']}", fill=(255, 0, 255),
                      stroke_width=2, stroke_fill=(0, 0, 0))
    output = io.BytesIO()
    result.save(output, "JPEG", quality=88)
    return output.getvalue()


def candidate_overlay(image, shapes):
    from .edit_agent import _rasterize
    canvas = Image.open(io.BytesIO(image)).convert("RGB")
    layer = Image.new("RGBA", canvas.size)
    colors = [(255, 65, 62), (0, 190, 255), (250, 205, 40), (160, 80, 255)]
    for index, shape in enumerate(shapes):
        mask = Image.fromarray(_rasterize(shape, *canvas.size)*255, "L")
        color = colors[index % len(colors)]
        region = Image.new("RGBA", canvas.size, (*color, 85))
        layer.alpha_composite(Image.composite(region, Image.new("RGBA", canvas.size), mask))
    canvas = Image.alpha_composite(canvas.convert("RGBA"), layer)
    draw = ImageDraw.Draw(canvas)
    for shape in shapes:
        if shape.get("points"):
            x, y = shape["points"][0]
            draw.text((x+4, y+4), str(shape["id"]), fill="white", stroke_width=2, stroke_fill="black")
    output = io.BytesIO()
    canvas.convert("RGB").save(output, "JPEG", quality=88)
    return output.getvalue()


def candidate_holes(shapes, labels, width, height):
    """Find enclosed holes and open narrow gaps in floor candidates."""
    floor_labels = {"地面", "floor", "ground"}
    holes = []
    for shape in shapes:
        if shape.get("label", "").strip().lower() in floor_labels:
            holes.extend(_candidate_gap_regions(shape, width, height))
    holes.sort(key=lambda item: (item["bbox"][1], item["bbox"][0], -item["area"]))
    return holes


def select_review_gaps(holes, maximum=4, minimum_area=None):
    """Keep the largest actionable gaps so one review cannot explode in size.

    Morphological closing naturally creates strips along image borders when a
    floor mask reaches beyond the camera view.  Those strips have no visible
    exterior context and are unsafe fill proposals, so never send them to the
    VLM as repairable open gaps.  Enclosed holes remain reviewable.
    """
    if type(maximum) is not int or not 1 <= maximum <= 20:
        raise ValueError("max_review_gaps must be an integer from 1 to 20")
    if minimum_area is None:
        image_area = next((int(mask.shape[0] * mask.shape[1])
                           for item in holes for mask in [item.get("mask")]
                           if isinstance(mask, np.ndarray) and mask.ndim == 2), 0)
        # Ignore sub-pixel-scale tail noise while keeping the same relative
        # threshold across 1920x1536 and 640x512 inputs.
        minimum_area = max(16, round(image_area * 0.00002))
    if type(minimum_area) is not int or minimum_area < 1:
        raise ValueError("minimum_review_gap_area must be a positive integer")
    actionable = []
    for item in holes:
        if item.get("area", 0) < minimum_area:
            continue
        mask = item.get("mask")
        if item.get("kind") == "open_gap" and isinstance(mask, np.ndarray) and mask.ndim == 2:
            height, width = mask.shape
            x1, y1, x2, y2 = item["bbox"]
            if x1 <= 0 or y1 <= 0 or x2 >= width - 1 or y2 >= height - 1:
                continue
        actionable.append(item)
    selected = sorted(actionable, key=lambda item: item["area"], reverse=True)[:maximum]
    return sorted(selected, key=lambda item: (item["bbox"][1], item["bbox"][0], -item["area"]))


def hole_overlay(image, holes, shapes=None):
    """Show source context beside a binary floor mask and numbered gaps."""
    source = Image.open(io.BytesIO(image)).convert("RGB")
    width, height = source.size
    pixels = np.zeros((height, width, 4), dtype=np.uint8)
    for hole in holes:
        pixels[hole["mask"] != 0] = (255, 36, 120, 190)
    left = Image.alpha_composite(source.convert("RGBA"), Image.fromarray(pixels, "RGBA"))
    right_pixels = np.zeros((height, width, 3), dtype=np.uint8)
    if shapes:
        from .edit_agent import _rasterize
        for shape in shapes:
            if shape.get("label", "").strip().lower() in {"地面", "floor", "ground"}:
                right_pixels[_rasterize(shape, width, height) != 0] = (235, 235, 235)
    for hole in holes:
        right_pixels[hole["mask"] != 0] = (255, 40, 120)
    composite = Image.new("RGB", (width * 2, height))
    composite.paste(left.convert("RGB"), (0, 0))
    composite.paste(Image.fromarray(right_pixels, "RGB"), (width, 0))
    draw = ImageDraw.Draw(composite)
    for index, hole in enumerate(holes, 1):
        x1, y1, x2, y2 = hole["bbox"]
        draw.rectangle((x1, y1, x2, y2), outline=(255, 240, 0), width=2)
        draw.rectangle((width + x1, y1, width + x2, y2), outline=(255, 240, 0), width=2)
        cx, cy = hole["centroid"]
        for offset in (0, width):
            draw.text((offset + cx, cy), str(index), anchor="mm", fill="white",
                      stroke_width=3, stroke_fill="black")
    output = io.BytesIO()
    composite.save(output, "JPEG", quality=90)
    return output.getvalue()


def gap_detail_images(image, shapes, holes):
    """Keep thin missed strips readable instead of relying on a full-frame overlay."""
    from .edit_agent import _rasterize
    if not holes:
        return []
    source = Image.open(io.BytesIO(image)).convert("RGB")
    mask = np.zeros((source.height, source.width), dtype=np.uint8)
    # review() normally receives compact shape summaries. execute() passes
    # actual geometry separately for evidence generation.
    for shape in shapes:
        if shape.get("label", "").strip().lower() in {"地面", "floor", "ground"}:
            mask |= _rasterize(shape, source.width, source.height)
    binary = Image.fromarray(mask * 255).convert("RGB")
    # The full-frame evidence already contains every ID. Two four-row pages
    # keep eight actionable regions readable without an unbounded request.
    ranked = sorted(enumerate(holes, 1), key=lambda item: item[1]["area"], reverse=True)[:8]
    pages = []
    for start in range(0, len(ranked), 4):
        sheet = Image.new("RGB", (1080, 4 * 320), (28, 28, 28))
        draw = ImageDraw.Draw(sheet)
        reports = []
        for row, (number, hole) in enumerate(ranked[start:start + 4]):
            x1, y1, x2, y2 = hole["bbox"]
            crop = [max(0, x1 - 28), max(0, y1 - 28),
                    min(source.width, x2 + 29), min(source.height, y2 + 29)]
            marked = np.asarray(binary).copy()
            marked[hole["mask"] != 0] = (255, 40, 120)
            panels = [source.crop(crop), binary.crop(crop), Image.fromarray(marked).crop(crop)]
            for column, panel in enumerate(panels):
                panel.thumbnail((356, 286), Image.Resampling.NEAREST if column else Image.Resampling.LANCZOS)
                sheet.paste(panel, (column * 360 + (360 - panel.width) // 2, row * 320 + 30))
            draw.text((5, row * 320 + 8), f"#{number}  original | binary mask | proposed gap", fill="white")
            reports.append({"number": number, "hole_id": hole["id"], "crop_bbox": crop})
        output = io.BytesIO()
        sheet.crop((0, 0, 1080, len(reports) * 320)).save(output, "JPEG", quality=95)
        pages.append((output.getvalue(), reports))
    return pages


def _shape_summary(shape):
    parts = shape.get("parts") or [shape]
    points = [point for part in parts for point in part.get("points", [])]
    xs, ys = zip(*points)
    return {"id": shape["id"], "label": shape["label"], "score": shape.get("score"),
            "bbox": [min(xs), min(ys), max(xs), max(ys)], "vertices": len(points)}


def _exact_duplicate_groups(shapes, width, height):
    """Expose pixel-identical candidate masks as evidence, never apply edits."""
    from .edit_agent import _rasterize
    groups = {}
    for shape in shapes:
        mask = _rasterize(shape, width, height)
        if not mask.any():
            continue
        key = (shape["label"], hashlib.sha256(mask.tobytes()).digest())
        group = groups.setdefault(key, {"label": shape["label"], "shape_ids": [], "pixels": int(mask.sum())})
        group["shape_ids"].append(shape["id"])
    return [group for group in groups.values() if len(group["shape_ids"]) > 1]


def suppress_near_duplicate_candidates(shapes, width, height,
                                       same_label_iou=0.95, floor_conflict_iou=0.90,
                                       preserve_class_conflicts=False):
    """Remove same-class duplicates; unreferenced drafts retain class ambiguity."""
    from .edit_agent import _rasterize
    masks = {shape["id"]: _rasterize(shape, width, height).astype(bool) for shape in shapes}
    retained, suppressed = [], []

    def iou(left, right):
        intersection = np.logical_and(left, right).sum()
        union = np.logical_or(left, right).sum()
        return float(intersection / union) if union else 0.0

    # Prefer the highest-confidence candidate among almost identical same-label masks.
    ordered = sorted(enumerate(shapes),
                     key=lambda item: (-(item[1].get("score") or 0.0), item[0]))
    for _, candidate in ordered:
        duplicate = next((kept for kept in retained
                          if kept.get("label") == candidate.get("label")
                          and iou(masks[kept["id"]], masks[candidate["id"]]) >= same_label_iou), None)
        if duplicate is not None:
            suppressed.append({"id": candidate["id"], "label": candidate.get("label"),
                               "reason": "near_duplicate_same_label",
                               "kept_id": duplicate["id"],
                               "iou": round(iou(masks[duplicate["id"]], masks[candidate["id"]]), 4)})
        else:
            retained.append(candidate)

    floors = [shape for shape in retained if shape.get("label") == "floor"]
    final = []
    for candidate in retained:
        conflict = next((floor for floor in floors
                         if candidate.get("label") == "cargo_floor"
                         and not preserve_class_conflicts
                         and iou(masks[floor["id"]], masks[candidate["id"]]) >= floor_conflict_iou), None)
        if conflict is not None:
            suppressed.append({"id": candidate["id"], "label": "cargo_floor",
                               "reason": "overlaps_warehouse_floor",
                               "kept_id": conflict["id"],
                               "iou": round(iou(masks[conflict["id"]], masks[candidate["id"]]), 4)})
        else:
            final.append(candidate)
    return final, suppressed


def filter_slope_candidate_geometry(shapes, image_height, minimum_width_height_ratio=1.75,
                                    minimum_height_fraction=0.07):
    """Retain broad planar slope proposals; route tall strip responses to review.

    The filter does not confirm a slope or use color. It only rejects the repeatable
    vertical and compact floor-patch distractors returned by the broader
    traversable-surface query. Borderline proposals are retained in the audit as
    human-review-required rather than being promoted as slope truth.
    """
    if not 1.0 <= minimum_width_height_ratio <= 10.0:
        raise ValueError("斜坡候选宽高比阈值无效")
    if type(image_height) is not int or image_height < 1 or not 0.0 < minimum_height_fraction <= 0.5:
        raise ValueError("斜坡候选高度阈值无效")
    minimum_height = max(4, round(image_height * minimum_height_fraction))
    retained, suppressed = [], []
    for shape in shapes:
        if shape.get("label") != "slope":
            retained.append(shape)
            continue
        summary = _shape_summary(shape)
        left, top, right, bottom = summary["bbox"]
        box_width = max(1.0, float(right - left + 1))
        box_height = max(1.0, float(bottom - top + 1))
        ratio = box_width / box_height
        if ratio >= minimum_width_height_ratio and box_height >= minimum_height:
            retained.append(shape)
        else:
            too_thin = box_height < minimum_height
            suppressed.append({
                "id": shape.get("id"), "label": "slope",
                "reason": ("slope_candidate_too_thin" if too_thin
                           else "non_planar_slope_candidate_geometry"),
                "reason_code": ("slope_candidate_below_minimum_height" if too_thin
                                else "slope_candidate_not_broad_planar"),
                "route": "human_review_required",
                "bbox": summary["bbox"], "width_height_ratio": round(ratio, 4),
                "minimum_width_height_ratio": minimum_width_height_ratio,
                "bbox_height": round(box_height, 2), "minimum_height": minimum_height,
            })
    return retained, suppressed


def slope_overlap_crop_images(image):
    """Build the exact deterministic overlap crops used by slope fallback."""
    source = Image.open(io.BytesIO(image)).convert("RGB")
    bounds = ((0.0, 0.0, 0.55, 0.55), (0.45, 0.0, 1.0, 0.55),
              (0.0, 0.45, 0.55, 1.0), (0.45, 0.45, 1.0, 1.0))
    crops = []
    for left_f, top_f, right_f, bottom_f in bounds:
        left, top = round(source.width * left_f), round(source.height * top_f)
        right, bottom = round(source.width * right_f), round(source.height * bottom_f)
        crop = source.crop((left, top, right, bottom))
        encoded = io.BytesIO()
        crop.save(encoded, "JPEG", quality=95)
        crops.append({"image": encoded.getvalue(),
                      "bbox": [left, top, right, bottom],
                      "width": crop.width, "height": crop.height})
    return crops


def filter_slope_pallet_conflicts(shapes, width, height, maximum_slope_covered=0.5,
                                  maximum_pallet_envelope_covered=0.1):
    """Reject a slope draft covered by a pallet mask or its occlusion envelope."""
    from .edit_agent import _rasterize
    if (not 0.0 < maximum_slope_covered <= 1.0
            or not 0.0 < maximum_pallet_envelope_covered <= 1.0):
        raise ValueError("斜坡与托盘冲突阈值无效")
    pallets = [shape for shape in shapes if shape.get("label") == "pallet"]
    pallet_masks = [(shape, _rasterize(shape, width, height).astype(bool),
                     _shape_summary(shape)["bbox"])
                    for shape in pallets]
    retained, suppressed = [], []
    for shape in shapes:
        if shape.get("label") != "slope":
            retained.append(shape)
            continue
        slope_mask = _rasterize(shape, width, height).astype(bool)
        slope_area = int(slope_mask.sum())
        slope_bbox = _shape_summary(shape)["bbox"]
        slope_bbox_area = ((slope_bbox[2] - slope_bbox[0] + 1)
                           * (slope_bbox[3] - slope_bbox[1] + 1))
        conflict = None
        for pallet, pallet_mask, pallet_bbox in pallet_masks:
            covered = (float(np.logical_and(slope_mask, pallet_mask).sum()) / slope_area
                       if slope_area else 0.0)
            overlap_width = max(0, min(slope_bbox[2], pallet_bbox[2])
                                - max(slope_bbox[0], pallet_bbox[0]) + 1)
            overlap_height = max(0, min(slope_bbox[3], pallet_bbox[3])
                                 - max(slope_bbox[1], pallet_bbox[1]) + 1)
            envelope_covered = (overlap_width * overlap_height / slope_bbox_area
                                if slope_bbox_area else 0.0)
            normalized_conflict = max(
                covered / maximum_slope_covered,
                envelope_covered / maximum_pallet_envelope_covered)
            if (normalized_conflict >= 1.0
                    and (conflict is None or normalized_conflict > conflict[1])):
                conflict = (pallet, normalized_conflict, covered, envelope_covered)
        if conflict is None:
            retained.append(shape)
        else:
            pallet, _, covered, envelope_covered = conflict
            suppressed.append({
                "id": shape.get("id"), "label": "slope",
                "reason": "slope_candidate_overlaps_pallet",
                "reason_code": "slope_candidate_pallet_conflict",
                "route": "human_review_required", "pallet_id": pallet.get("id"),
                "slope_covered_by_pallet": round(covered, 4),
                "slope_envelope_covered_by_pallet": round(envelope_covered, 4),
                "maximum_slope_covered": maximum_slope_covered,
                "maximum_pallet_envelope_covered": maximum_pallet_envelope_covered,
            })
    return retained, suppressed


def filter_slope_floor_conflicts(shapes, width, height, maximum_slope_covered=0.9):
    """Route a slope proposal almost entirely explained by a floor mask to review.

    This is a cross-class geometry check, not a color or texture heuristic.  The
    floor candidate is preserved and the ambiguous slope is recorded in the
    candidate-filter audit instead of being promoted as a second semantic mask.
    """
    from .edit_agent import _rasterize
    if not 0.0 < maximum_slope_covered <= 1.0:
        raise ValueError("斜坡与地面冲突阈值无效")
    floors = [(shape, _rasterize(shape, width, height).astype(bool))
              for shape in shapes if shape.get("label") == "floor"]
    retained, suppressed = [], []
    for shape in shapes:
        if shape.get("label") != "slope" or not floors:
            retained.append(shape)
            continue
        slope_mask = _rasterize(shape, width, height).astype(bool)
        slope_area = int(slope_mask.sum())
        conflict = None
        for floor, floor_mask in floors:
            covered = (float(np.logical_and(slope_mask, floor_mask).sum()) / slope_area
                       if slope_area else 0.0)
            if covered >= maximum_slope_covered and (
                    conflict is None or covered > conflict[1]):
                conflict = (floor, covered)
        if conflict is None:
            retained.append(shape)
        else:
            floor, covered = conflict
            suppressed.append({
                "id": shape.get("id"), "label": "slope",
                "reason": "slope_candidate_covered_by_floor",
                "reason_code": "slope_candidate_floor_conflict",
                "route": "human_review_required", "floor_id": floor.get("id"),
                "slope_covered_by_floor": round(covered, 4),
                "maximum_slope_covered": maximum_slope_covered,
            })
    return retained, suppressed


def filter_slope_reference_ego_conflicts(shapes, reference, width, height,
                                         maximum_slope_covered=0.8):
    """Use a same-camera reviewed ego mask to exclude fixed vehicle hardware.

    Ego structure is rigid relative to a mounted camera, so its human-confirmed
    reference pixels can be reused across frames of that topic.  This guard is
    disabled without a reference, without an ``ego`` annotation, or when the
    reference resolution differs; it never learns an exclusion zone from a
    machine candidate.
    """
    from .edit_agent import _rasterize
    if not 0.0 < maximum_slope_covered <= 1.0:
        raise ValueError("斜坡与自车参考冲突阈值无效")
    if not isinstance(reference, dict) or not isinstance(reference.get("annotations"), list):
        return list(shapes), []
    try:
        reference_size = Image.open(io.BytesIO(reference.get("image", b""))).size
    except (OSError, TypeError):
        return list(shapes), []
    if reference_size != (width, height):
        return list(shapes), []
    ego_shapes = [shape for shape in reference["annotations"] if shape.get("label") == "ego"]
    if not ego_shapes:
        return list(shapes), []
    ego_mask = np.zeros((height, width), dtype=bool)
    for ego in ego_shapes:
        ego_mask |= _rasterize(ego, width, height).astype(bool)
    if not ego_mask.any():
        return list(shapes), []
    retained, suppressed = [], []
    for shape in shapes:
        if shape.get("label") != "slope":
            retained.append(shape)
            continue
        slope_mask = _rasterize(shape, width, height).astype(bool)
        slope_area = int(slope_mask.sum())
        covered = (float(np.logical_and(slope_mask, ego_mask).sum()) / slope_area
                   if slope_area else 0.0)
        if covered < maximum_slope_covered:
            retained.append(shape)
            continue
        suppressed.append({
            "id": shape.get("id"), "label": "slope",
            "reason": "slope_candidate_covered_by_reviewed_reference_ego",
            "reason_code": "slope_candidate_reference_ego_conflict",
            "route": "human_review_required",
            "slope_covered_by_reference_ego": round(covered, 4),
            "maximum_slope_covered": maximum_slope_covered,
            "reference_ego_instances": len(ego_shapes),
        })
    return retained, suppressed


def filter_slope_current_ego_conflicts(shapes, width, height,
                                       maximum_slope_covered=0.8,
                                       minimum_ego_score=0.85):
    """Route slope drafts lying inside a strong same-frame ego draft to review.

    This is a candidate-only mutual-exclusion guard, not an assertion that the
    machine ``ego`` mask is truth.  It therefore keeps the ego draft and records
    the conflicting slope as human-review-required instead of deleting evidence
    or promoting either mask to training data.  Requiring both near-total slope
    coverage and a strong ego score preserves disjoint ramps ahead of the robot.
    """
    from .edit_agent import _rasterize
    if not 0.0 < maximum_slope_covered <= 1.0:
        raise ValueError("斜坡与当前自车候选冲突阈值无效")
    if not 0.0 <= minimum_ego_score <= 1.0:
        raise ValueError("当前自车候选最低分数无效")
    egos = []
    for shape in shapes:
        score = shape.get("score")
        if (shape.get("label") == "ego" and not isinstance(score, bool)
                and isinstance(score, (int, float)) and score >= minimum_ego_score):
            egos.append((shape, _rasterize(shape, width, height).astype(bool)))
    if not egos:
        return list(shapes), []
    retained, suppressed = [], []
    for shape in shapes:
        if shape.get("label") != "slope":
            retained.append(shape)
            continue
        slope_mask = _rasterize(shape, width, height).astype(bool)
        slope_area = int(slope_mask.sum())
        conflict = None
        for ego, ego_mask in egos:
            covered = (float(np.logical_and(slope_mask, ego_mask).sum()) / slope_area
                       if slope_area else 0.0)
            if covered >= maximum_slope_covered and (
                    conflict is None or covered > conflict[1]):
                conflict = (ego, covered)
        if conflict is None:
            retained.append(shape)
            continue
        ego, covered = conflict
        suppressed.append({
            "id": shape.get("id"), "label": "slope",
            "reason": "slope_candidate_covered_by_current_ego_candidate",
            "reason_code": "slope_candidate_current_ego_conflict",
            "route": "human_review_required", "ego_id": ego.get("id"),
            "slope_covered_by_current_ego": round(covered, 4),
            "maximum_slope_covered": maximum_slope_covered,
            "ego_score": float(ego.get("score")),
            "minimum_ego_score": minimum_ego_score,
            "machine_candidate_only": True, "human_truth": False,
            "approved_for_training": False,
        })
    return retained, suppressed


def candidate_class_overlap_conflicts(shapes, width, height, threshold=0.90):
    """Flag ambiguous class pairs without deleting or relabeling either mask."""
    from .edit_agent import _rasterize
    robot_labels = {"车辆", "ego", "other_robot"}
    relevant = [item for item in shapes
                if item.get("label") in robot_labels | {"floor", "cargo_floor"}]
    masks = {}
    conflicts = []
    for position, left in enumerate(relevant):
        for right in relevant[position + 1:]:
            left_label, right_label = left.get("label"), right.get("label")
            labels = {left_label, right_label}
            relevant_pair = (left_label != right_label and labels.issubset(robot_labels)) \
                or labels == {"floor", "cargo_floor"}
            if not relevant_pair:
                continue
            for item in (left, right):
                if item["id"] not in masks:
                    masks[item["id"]] = _rasterize(item, width, height).astype(bool)
            a, b = masks[left["id"]], masks[right["id"]]
            union = np.logical_or(a, b).sum()
            overlap = float(np.logical_and(a, b).sum() / union) if union else 0.0
            if overlap >= threshold:
                conflicts.append({
                    "shape_ids": [left["id"], right["id"]],
                    "labels": [left["label"], right["label"]],
                    "iou": round(overlap, 4),
                    "reason": "不同类别候选高度重叠，请核验实体归属；已保留两个候选",
                    "reason_code": "candidate_cross_class_overlap",
                    "route": "human_review_required"})
    return conflicts


def _fingerprint(shapes):
    geometry = [{key: shape.get(key) for key in ("id", "label", "points", "holes", "parts")}
                for shape in shapes]
    return hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()


def _pallet_geometry_vertex_count(shape):
    return sum(
        len(part.get("points", []))
        + sum(len(hole) for hole in part.get("holes", []))
        for part in (shape.get("parts") or [shape]))


def _significant_component_count(mask):
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    return sum(int(stats[index, cv2.CC_STAT_AREA]) >= 9
               for index in range(1, count))


def _foreground_component_count(mask):
    count, _ = cv2.connectedComponents(
        mask.astype(np.uint8), connectivity=8)
    return count - 1


PALLET_COMPACT_TEXTURE_MINIMUM_CANDIDATE_AREA_RATIO = 0.30
PALLET_IMAGE_TEXTURE_MAXIMUM_AREA = 2048
PALLET_IMAGE_TEXTURE_MAXIMUM_SPAN = 224
PALLET_IMAGE_TEXTURE_MAXIMUM_GRAY_DELTA = 45.0
PALLET_IMAGE_TEXTURE_MAXIMUM_COLOR_DISTANCE = 90.0
PALLET_IMAGE_BOUNDARY_CLOSE_KERNEL = 17
PALLET_IMAGE_BOUNDARY_MAXIMUM_AREA = 1024
PALLET_IMAGE_BOUNDARY_MAXIMUM_SPAN = 96
PALLET_IMAGE_BOUNDARY_MINIMUM_RING_GRAY = 110.0
PALLET_IMAGE_BOUNDARY_MAXIMUM_RING_COLOR_SPREAD = 80.0
PALLET_IMAGE_BOUNDARY_MAXIMUM_COMPONENT_MERGES = 2
PALLET_ROI_EVIDENCE_MINIMUM_CANDIDATE_AREA = 4096
PALLET_ROI_EVIDENCE_MINIMUM_BBOX_AREA = 8192
PALLET_ROI_EVIDENCE_MINIMUM_BBOX_WIDTH = 96
PALLET_ROI_EVIDENCE_MINIMUM_BBOX_HEIGHT = 24


def _pallet_image_evidence_scale_gate(mask, *, allow_roi_scale=False):
    """Authorize image evidence at legacy full-frame or bounded ROI scale."""
    candidate_area = int(mask.sum())
    full_image_ratio = candidate_area / max(mask.size, 1)
    ys, xs = np.where(mask != 0)
    bbox_width = int(xs.max() - xs.min() + 1) if len(xs) else 0
    bbox_height = int(ys.max() - ys.min() + 1) if len(ys) else 0
    bbox_area = bbox_width * bbox_height
    legacy_passed = (
        full_image_ratio >= PALLET_COMPACT_TEXTURE_MINIMUM_CANDIDATE_AREA_RATIO)
    roi_passed = bool(
        allow_roi_scale
        and candidate_area >= PALLET_ROI_EVIDENCE_MINIMUM_CANDIDATE_AREA
        and bbox_area >= PALLET_ROI_EVIDENCE_MINIMUM_BBOX_AREA
        and bbox_width >= PALLET_ROI_EVIDENCE_MINIMUM_BBOX_WIDTH
        and bbox_height >= PALLET_ROI_EVIDENCE_MINIMUM_BBOX_HEIGHT)
    return legacy_passed or roi_passed, {
        "route": (
            "full_image_ratio" if legacy_passed else
            "roi_absolute" if roi_passed else "rejected"),
        "candidate_area": candidate_area,
        "candidate_area_ratio": round(full_image_ratio, 8),
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "bbox_area": bbox_area,
        "roi_scale_allowed": bool(allow_roi_scale),
    }


def _compact_pallet_texture_hole(area, width, height, *, maximum_area=768,
                                 maximum_span=64, maximum_aspect_ratio=3.0):
    """Identify compact surface drop-outs without swallowing long pallet slots."""
    short = max(min(width, height), 1)
    aspect_ratio = max(width, height) / short
    return (area <= maximum_area and max(width, height) <= maximum_span
            and aspect_ratio <= maximum_aspect_ratio)


def _image_guided_pallet_texture_holes(mask, image_rgb, components, stats,
                                       border_labels, *,
                                       maximum_area=PALLET_IMAGE_TEXTURE_MAXIMUM_AREA,
                                       maximum_span=PALLET_IMAGE_TEXTURE_MAXIMUM_SPAN,
                                       maximum_gray_delta=(
                                           PALLET_IMAGE_TEXTURE_MAXIMUM_GRAY_DELTA),
                                       maximum_color_distance=(
                                           PALLET_IMAGE_TEXTURE_MAXIMUM_COLOR_DISTANCE),
                                       allow_roi_scale=False):
    """Classify dense wooden-surface drop-outs while retaining real openings.

    A close pallet can contain dozens of irregular SAM holes caused by paint,
    wood grain, glare and shallow seams. Geometry alone cannot safely tell
    these from the regular dark openings of a plastic pallet. This helper
    requires both a close-instance scale gate and local appearance continuity.
    Long, rectangular, dark holes establish structural rows; other dark holes
    in the same row are retained even when fragmented.
    """
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3:
        return set(), {}
    if image_rgb.shape[:2] != mask.shape or image_rgb.shape[2] < 3:
        return set(), {}
    scale_gate_passed, scale_gate_audit = _pallet_image_evidence_scale_gate(
        mask, allow_roi_scale=allow_roi_scale)
    if not scale_gate_passed:
        return set(), {}

    kernel = np.ones((5, 5), dtype=np.uint8)
    rows = []
    rgb = image_rgb[..., :3].astype(np.float32)
    for component in range(1, len(stats)):
        if component in border_labels:
            continue
        x, y, width, height, area = [int(value) for value in stats[component]]
        component_mask = components == component
        ring = (cv2.dilate(component_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
                & ~component_mask & (mask > 0))
        if not ring.any():
            continue
        hole_mean = rgb[component_mask].mean(axis=0)
        ring_mean = rgb[ring].mean(axis=0)
        gray_delta = float(ring_mean.mean() - hole_mean.mean())
        color_distance = float(np.linalg.norm(ring_mean - hole_mean))
        short = max(min(width, height), 1)
        rows.append({
            "component": component,
            "width": width, "height": height, "area": area,
            "aspect_ratio": max(width, height) / short,
            "fill_ratio": area / max(width * height, 1),
            "center_y": y + (height - 1) / 2,
            "gray_delta": gray_delta,
            "color_distance": color_distance,
        })

    anchors = [row for row in rows
               if row["area"] >= 512
               and row["aspect_ratio"] >= 6.0
               and row["fill_ratio"] >= .60
               and row["gray_delta"] >= 40.0]
    selected, audit = set(), {}
    for row in rows:
        in_structural_band = any(
            abs(row["center_y"] - anchor["center_y"])
            <= max(12.0, anchor["height"] * 1.5)
            and (row["gray_delta"] >= 20.0
                 or row["aspect_ratio"] >= 3.0
                 or row["fill_ratio"] >= .60)
            for anchor in anchors)
        appearance_continuous = (
            row["gray_delta"] <= maximum_gray_delta
            and row["color_distance"] <= maximum_color_distance)
        is_texture = (
            row["area"] <= maximum_area
            and max(row["width"], row["height"]) <= maximum_span
            and appearance_continuous
            and not in_structural_band)
        if is_texture:
            selected.add(row["component"])
        audit[row["component"]] = {
            "gray_delta": round(row["gray_delta"], 3),
            "color_distance": round(row["color_distance"], 3),
            "fill_ratio": round(row["fill_ratio"], 6),
            "structural_band": in_structural_band,
            "image_texture_hole": is_texture,
        }
    return selected, audit


def _image_guided_pallet_boundary_notches(
        mask, image_rgb, structural_holes, *, allow_roi_scale=False,
        collect_gradient_audit=False):
    """Find narrow, light neutral boundary notches without closing real gaps.

    Unlike enclosed holes, these SAM drop-outs remain connected to the outer
    background.  A small morphological closing only proposes locations; the
    proposal is accepted component-by-component when it is away from the image
    edge, does not overlap a protected structural hole, and its original pixels
    remain locally consistent with a bright, low-colour-spread pallet surface.
    """
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3:
        return []
    if image_rgb.shape[:2] != mask.shape or image_rgb.shape[2] < 3:
        return []
    scale_gate_passed, scale_gate_audit = _pallet_image_evidence_scale_gate(
        mask, allow_roi_scale=allow_roi_scale)
    if not scale_gate_passed:
        return []

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (PALLET_IMAGE_BOUNDARY_CLOSE_KERNEL,
         PALLET_IMAGE_BOUNDARY_CLOSE_KERNEL))
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    additions = ((closed > 0) & (mask == 0)).astype(np.uint8)
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        additions, connectivity=8)
    structural_union = np.zeros_like(mask, dtype=bool)
    for hole_mask, _ in structural_holes:
        structural_union |= hole_mask
    rgb = image_rgb[..., :3].astype(np.float32)
    gradient = None
    if collect_gradient_audit:
        gray = cv2.cvtColor(
            image_rgb[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = cv2.magnitude(gradient_x, gradient_y)
    ring_kernel = np.ones((5, 5), dtype=np.uint8)
    proposals = []
    image_height, image_width = mask.shape
    for component in range(1, count):
        x, y, width, height, area = [int(value) for value in stats[component]]
        if area < 9 or area > PALLET_IMAGE_BOUNDARY_MAXIMUM_AREA:
            continue
        if max(width, height) > PALLET_IMAGE_BOUNDARY_MAXIMUM_SPAN:
            continue
        if x == 0 or y == 0 or x + width >= image_width or y + height >= image_height:
            continue
        component_mask = components == component
        if np.any(component_mask & structural_union):
            continue
        ring = (cv2.dilate(component_mask.astype(np.uint8), ring_kernel,
                           iterations=1).astype(bool)
                & ~component_mask & (mask > 0))
        if not ring.any():
            continue
        notch_mean = rgb[component_mask].mean(axis=0)
        ring_mean = rgb[ring].mean(axis=0)
        ring_gray = float(ring_mean.mean())
        gray_delta = float(ring_gray - notch_mean.mean())
        color_distance = float(np.linalg.norm(ring_mean - notch_mean))
        ring_color_spread = float(ring_mean.max() - ring_mean.min())
        roi_absolute_evidence = (
            scale_gate_audit["route"] == "roi_absolute")
        if not (
                gray_delta <= PALLET_IMAGE_TEXTURE_MAXIMUM_GRAY_DELTA
                and color_distance <= PALLET_IMAGE_TEXTURE_MAXIMUM_COLOR_DISTANCE
                and (roi_absolute_evidence
                     or (ring_gray >= PALLET_IMAGE_BOUNDARY_MINIMUM_RING_GRAY
                         and ring_color_spread
                         <= PALLET_IMAGE_BOUNDARY_MAXIMUM_RING_COLOR_SPREAD))):
            continue
        proposal = {
            "mask": component_mask,
            "area": area,
            "bbox": [x, y, x + width - 1, y + height - 1],
            "gray_delta": round(gray_delta, 3),
            "color_distance": round(color_distance, 3),
            "ring_gray": round(ring_gray, 3),
            "ring_color_spread": round(ring_color_spread, 3),
            "scale_evidence_route": scale_gate_audit["route"],
        }
        if gradient is not None:
            proposal.update({
                "notch_gradient_mean": round(
                    float(gradient[component_mask].mean()), 3),
                "ring_gradient_mean": round(float(gradient[ring].mean()), 3),
            })
        proposals.append(proposal)
    return proposals


def _structural_hole_masks(
        mask, *, image_rgb=None,
        minimum_compact_texture_candidate_area_ratio=(
            PALLET_COMPACT_TEXTURE_MINIMUM_CANDIDATE_AREA_RATIO),
        allow_roi_scale=False):
    background = (mask == 0).astype(np.uint8)
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        background, connectivity=8)
    border_labels = set(np.unique(np.concatenate((
        components[0, :], components[-1, :],
        components[:, 0], components[:, -1]))).tolist())
    image_texture_holes, _ = _image_guided_pallet_texture_holes(
        mask, image_rgb, components, stats, border_labels,
        allow_roi_scale=allow_roi_scale)
    image_guidance_available = (
        isinstance(image_rgb, np.ndarray) and image_rgb.ndim == 3
        and image_rgb.shape[:2] == mask.shape and image_rgb.shape[2] >= 3)
    output = []
    candidate_area = int(mask.sum())
    for component in range(1, count):
        if component in border_labels:
            continue
        width = int(stats[component, cv2.CC_STAT_WIDTH])
        height = int(stats[component, cv2.CC_STAT_HEIGHT])
        area = int(stats[component, cv2.CC_STAT_AREA])
        micro_hole = area <= 96 and max(width, height) <= 32
        texture_hole = area <= 256 and max(width, height) <= 24
        compact_texture_hole = (
            candidate_area / max(mask.size, 1)
            >= minimum_compact_texture_candidate_area_ratio
            and _compact_pallet_texture_hole(area, width, height)
            and (not image_guidance_available
                 or component in image_texture_holes))
        if not (micro_hole or texture_hole or compact_texture_hole
                or component in image_texture_holes):
            output.append((components == component, area))
    return output


def _simplify_pallet_boundary(shape, original_mask, width, height, *,
                              maximum_change_ratio, minimum_hole_retention=.75,
                              image_rgb=None,
                              minimum_compact_texture_candidate_area_ratio=(
                                  PALLET_COMPACT_TEXTURE_MINIMUM_CANDIDATE_AREA_RATIO)):
    """Reduce pixel-staircase vertices under the existing geometry gates."""
    from .edit_agent import _rasterize
    before_vertices = _pallet_geometry_vertex_count(shape)
    if before_vertices < 8:
        return shape, None, _rasterize(shape, width, height).astype(np.uint8)
    original_area = int(original_mask.sum())
    original_components = _significant_component_count(original_mask)
    structural_holes = _structural_hole_masks(
        original_mask, image_rgb=image_rgb,
        minimum_compact_texture_candidate_area_ratio=(
            minimum_compact_texture_candidate_area_ratio))
    baseline_mask = _rasterize(shape, width, height).astype(np.uint8)
    # One sub-pixel epsilon was sufficient for small/distant masks, but it
    # leaves close pallets with visibly stepped outlines because their contour
    # can span hundreds of pixels.  Try a size-adaptive proposal first and keep
    # the existing raster-change, component and structural-hole gates below as
    # the final authority.  The descending order means the cleanest safe
    # contour wins; unsafe proposals still fall back to the legacy values.
    if original_area >= 50000:
        epsilon_candidates = (2.0, 1.5, 1.0, .75, .5)
        simplification_scale_band = "large"
    elif original_area >= 5000:
        epsilon_candidates = (1.5, 1.0, .75, .5)
        simplification_scale_band = "medium"
    else:
        epsilon_candidates = (1.0, .75, .5)
        simplification_scale_band = "small"
    for epsilon in epsilon_candidates:
        proposal = copy.deepcopy(shape)
        parts = []
        for part in (shape.get("parts") or [shape]):
            points = np.asarray(part.get("points", []), dtype=np.float32)
            if len(points) < 3:
                continue
            simplified_points = cv2.approxPolyDP(
                points.reshape(-1, 1, 2), epsilon, True
            ).reshape(-1, 2).round().astype(int).tolist()
            simplified_holes = []
            for hole in part.get("holes", []):
                vertices = np.asarray(hole, dtype=np.float32)
                if len(vertices) < 3:
                    simplified_holes.append(copy.deepcopy(hole))
                    continue
                simplified = cv2.approxPolyDP(
                    vertices.reshape(-1, 1, 2), epsilon, True
                ).reshape(-1, 2).round().astype(int).tolist()
                simplified_holes.append(
                    simplified if len(simplified) >= 3 else copy.deepcopy(hole))
            replacement_part = copy.deepcopy(part)
            replacement_part.update(
                points=simplified_points, holes=simplified_holes)
            parts.append(replacement_part)
        if not parts:
            continue
        proposal.update(
            parts=parts, points=parts[0]["points"], holes=parts[0]["holes"])
        after_vertices = _pallet_geometry_vertex_count(proposal)
        if (after_vertices >= before_vertices
                or not _valid_pallet_geometry_shape(proposal, width, height)):
            continue
        proposal_mask = _rasterize(proposal, width, height).astype(np.uint8)
        total_changed = int(np.count_nonzero(proposal_mask != original_mask))
        if total_changed / max(original_area, 1) > maximum_change_ratio:
            continue
        if _significant_component_count(proposal_mask) != original_components:
            continue
        retention = [
            int(np.count_nonzero(hole_mask & (proposal_mask == 0))) / hole_area
            for hole_mask, hole_area in structural_holes]
        minimum_retention = min(retention) if retention else 1.0
        if minimum_retention < minimum_hole_retention:
            continue
        return proposal, {
            "boundary_simplification_applied": True,
            "boundary_simplification_epsilon": epsilon,
            "boundary_simplification_scale_band": simplification_scale_band,
            "boundary_vertices_before": before_vertices,
            "boundary_vertices_after": after_vertices,
            "boundary_vertex_reduction_ratio": round(
                1 - after_vertices / before_vertices, 6),
            "boundary_simplification_changed_pixels": int(
                np.count_nonzero(proposal_mask != baseline_mask)),
            "minimum_structural_hole_retention": round(minimum_retention, 6),
        }, proposal_mask
    return shape, None, baseline_mask


def _valid_pallet_geometry_shape(shape, width, height):
    """Validate geometry while preserving imported machine-candidate IDs.

    Interactive edit operations intentionally require integer IDs, but imported
    SAM candidate collections use stable string IDs.  Geometry cleanup never
    addresses an edit operation by ID and preserves the identifier verbatim, so
    validate a shallow probe with a temporary integer rather than rejecting an
    otherwise valid replacement.
    """
    identifier = shape.get("id") if isinstance(shape, dict) else None
    if type(identifier) is int:
        return valid_shape(shape, width, height)
    if not isinstance(identifier, str) or not identifier.strip():
        return False
    return valid_shape({**shape, "id": 1}, width, height)


def refine_pallet_candidate_geometry(shape, width, height, *, maximum_hole_area=96,
                                     maximum_hole_span=32,
                                     maximum_texture_hole_area=256,
                                     maximum_texture_hole_span=24,
                                     maximum_compact_texture_hole_area=768,
                                     maximum_compact_texture_hole_span=64,
                                     maximum_compact_texture_aspect_ratio=3.0,
                                     minimum_compact_texture_candidate_area_ratio=(
                                         PALLET_COMPACT_TEXTURE_MINIMUM_CANDIDATE_AREA_RATIO),
                                     maximum_change_ratio=0.025,
                                     image_rgb=None,
                                     geometry_policy="full"):
    """Remove texture-scale pallet mask noise without erasing structural gaps.

    The cleanup is deliberately bounded to one already detected instance.  It
    fills enclosed micro holes plus slightly larger, short-span texture holes.
    Close pallets also receive a compactness gate that fills irregular surface
    drop-outs while keeping long horizontal slat and fork openings.
    It then applies a size-adaptive median boundary proposal when that proposal
    preserves the connected-component count and stays inside a 2.5 percent
    total-change budget.  Long openings (for example fork slots and inter-slat
    gaps) remain even when their area is modest.
    """
    if geometry_policy not in {
            "full", "appearance_evidence_only", "appearance_evidence_roi"}:
        raise ValueError("unknown pallet geometry cleanup policy")
    if not isinstance(shape, dict) or str(shape.get("label", "")).casefold() != "pallet":
        return shape, None
    from .edit_agent import _rasterize, _shape_from_mask
    mask = _rasterize(shape, width, height).astype(np.uint8)
    area = int(mask.sum())
    if area < 400:
        return shape, None
    original_component_count = _significant_component_count(mask)
    original_foreground_component_count = _foreground_component_count(mask)
    appearance_evidence_policy = geometry_policy in {
        "appearance_evidence_only", "appearance_evidence_roi"}
    allow_roi_scale = geometry_policy == "appearance_evidence_roi"
    image_evidence_scale_passed, image_evidence_scale = (
        _pallet_image_evidence_scale_gate(
            mask, allow_roi_scale=allow_roi_scale))

    refined = mask.copy()
    authorized_additions = np.zeros_like(mask, dtype=bool)
    background = (mask == 0).astype(np.uint8)
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        background, connectivity=8)
    border_labels = set(np.unique(np.concatenate((
        components[0, :], components[-1, :], components[:, 0], components[:, -1]))).tolist())
    filled_holes = []
    filled_texture_holes = []
    filled_compact_texture_holes = []
    filled_image_texture_holes = []
    filled_image_boundary_notches = []
    used_image_texture_evidence = False
    image_texture_holes, image_texture_audit = _image_guided_pallet_texture_holes(
        mask, image_rgb, components, stats, border_labels,
        allow_roi_scale=allow_roi_scale)
    image_guidance_available = (
        isinstance(image_rgb, np.ndarray) and image_rgb.ndim == 3
        and image_rgb.shape[:2] == mask.shape and image_rgb.shape[2] >= 3)
    protected_structural_holes = _structural_hole_masks(
        mask, image_rgb=image_rgb,
        minimum_compact_texture_candidate_area_ratio=(
            minimum_compact_texture_candidate_area_ratio),
        allow_roi_scale=allow_roi_scale)
    image_boundary_notches = _image_guided_pallet_boundary_notches(
        mask, image_rgb, protected_structural_holes,
        allow_roi_scale=allow_roi_scale,
        collect_gradient_audit=(geometry_policy == "appearance_evidence_roi"))
    if appearance_evidence_policy:
        # V7 is a local add-only shadow policy.  One supported component must
        # never unlock the broader five/six-percent v6 budget elsewhere.
        effective_maximum_change_ratio = min(maximum_change_ratio, .01)
    else:
        effective_maximum_change_ratio = (
            max(maximum_change_ratio, .06) if image_boundary_notches else
            max(maximum_change_ratio, .05) if image_texture_holes else
            maximum_change_ratio)
    hole_fill_plans = []
    for component in range(1, count):
        if component in border_labels:
            continue
        x, y, w, h, component_area = [int(value) for value in stats[component]]
        micro_hole = (component_area <= maximum_hole_area
                      and max(w, h) <= maximum_hole_span)
        texture_hole = (component_area <= maximum_texture_hole_area
                        and max(w, h) <= maximum_texture_hole_span)
        compact_texture_hole = (
            area / max(mask.size, 1)
            >= minimum_compact_texture_candidate_area_ratio
            and _compact_pallet_texture_hole(
                component_area, w, h,
                maximum_area=maximum_compact_texture_hole_area,
                maximum_span=maximum_compact_texture_hole_span,
                maximum_aspect_ratio=maximum_compact_texture_aspect_ratio)
            and (not image_guidance_available
                 or component in image_texture_holes))
        image_texture_hole = component in image_texture_holes
        eligible_hole = (
            image_texture_hole
            if appearance_evidence_policy else
            (micro_hole or texture_hole or compact_texture_hole
             or image_texture_hole))
        if eligible_hole:
            record = {"area": component_area, "bbox": [x, y, x + w - 1, y + h - 1]}
            image_metrics = image_texture_audit.get(component, {})
            hole_fill_plans.append({
                "component": component,
                "record": record,
                "micro_hole": micro_hole,
                "texture_hole": texture_hole,
                "compact_texture_hole": compact_texture_hole,
                "image_texture_hole": image_texture_hole,
                # Prefer the smallest and most appearance-consistent repairs
                # when a dense close pallet contains more safe-looking noise
                # than the hard per-instance change budget permits.
                "priority": (
                    0 if micro_hole else 1 if texture_hole else
                    2 if compact_texture_hole else 3,
                    float(image_metrics.get("color_distance", 0.0)),
                    float(image_metrics.get("gray_delta", 0.0)),
                    component_area),
                "image_metrics": image_metrics,
            })

    fill_budget = max(int(area * effective_maximum_change_ratio), 1)
    if appearance_evidence_policy:
        fill_budget = min(fill_budget, 256)
    used_fill_budget = 0
    for plan in sorted(hole_fill_plans, key=lambda item: item["priority"]):
        component_area = plan["record"]["area"]
        if (appearance_evidence_policy
                and (not plan["image_texture_hole"]
                     or component_area > 256)):
            continue
        if used_fill_budget + component_area > fill_budget:
            continue
        component = plan["component"]
        component_mask = components == component
        refined[component_mask] = 1
        if appearance_evidence_policy:
            authorized_additions |= component_mask
        used_fill_budget += component_area
        record = plan["record"]
        micro_hole = plan["micro_hole"]
        texture_hole = plan["texture_hole"]
        compact_texture_hole = plan["compact_texture_hole"]
        image_texture_hole = plan["image_texture_hole"]
        used_image_texture_evidence = (
            used_image_texture_evidence or image_texture_hole)
        filled_holes.append(record)
        if texture_hole and not micro_hole:
            filled_texture_holes.append(record)
        if compact_texture_hole and not (micro_hole or texture_hole):
            filled_compact_texture_holes.append(record)
        if image_texture_hole and not (
                micro_hole or texture_hole or compact_texture_hole):
            record.update(plan["image_metrics"])
            filled_image_texture_holes.append(record)

    for notch in sorted(
            image_boundary_notches,
            key=lambda item: (item["color_distance"], abs(item["gray_delta"]),
                              item["area"])):
        if (used_fill_budget + notch["area"] > fill_budget
                or (appearance_evidence_policy
                    and notch["area"] > 256)):
            continue
        proposal = refined.copy()
        proposal[notch["mask"]] = 1
        proposal_component_count = _significant_component_count(proposal)
        proposal_foreground_component_count = _foreground_component_count(
            proposal)
        if (proposal_component_count > original_component_count
                or (appearance_evidence_policy
                    and proposal_foreground_component_count
                    != original_foreground_component_count)
                or original_component_count - proposal_component_count
                > PALLET_IMAGE_BOUNDARY_MAXIMUM_COMPONENT_MERGES):
            continue
        refined = proposal
        if appearance_evidence_policy:
            authorized_additions |= notch["mask"]
        used_fill_budget += notch["area"]
        filled_image_boundary_notches.append({
            key: copy.deepcopy(value) for key, value in notch.items()
            if key != "mask"})

    foreground_count, foreground, foreground_stats, _ = cv2.connectedComponentsWithStats(
        refined, connectivity=8)
    removed_island_pixels = 0
    if geometry_policy == "full":
        for component in range(1, foreground_count):
            component_area = int(foreground_stats[component, cv2.CC_STAT_AREA])
            if component_area < 9:
                refined[foreground == component] = 0
                removed_island_pixels += component_area
    before_components = sum(
        int(foreground_stats[component, cv2.CC_STAT_AREA]) >= 9
        for component in range(1, foreground_count))
    # Keep the bounded hole/island cleanup as an independent fallback.  A
    # large textured pallet can have many individually safe micro holes while
    # median smoothing adds just enough boundary pixels to push the combined
    # proposal over the total-change budget.  Rejecting the whole proposal in
    # that case preserves the very texture noise this helper is meant to
    # remove.  Falling back to this snapshot keeps the same hard change cap
    # without accepting the optional smoothing step.
    cleanup_only = refined.copy()
    cleanup_only_changed = int(np.count_nonzero(cleanup_only != mask))
    cleanup_only_filled_holes = copy.deepcopy(filled_holes)
    smoothing_kernel = 7 if area >= 50000 else (5 if area >= 5000 else 3)
    smoothed = (cv2.medianBlur(
        (refined * 255).astype(np.uint8), smoothing_kernel) > 127).astype(np.uint8)
    smoothed_count, _, smoothed_stats, _ = cv2.connectedComponentsWithStats(
        smoothed, connectivity=8)
    after_components = sum(
        int(smoothed_stats[component, cv2.CC_STAT_AREA]) >= 9
        for component in range(1, smoothed_count))
    smoothing_pixels = int(np.count_nonzero(smoothed != refined))
    smoothing_structural_holes = protected_structural_holes
    smoothing_structural_retention = [
        int(np.count_nonzero(hole_mask & (smoothed == 0))) / hole_area
        for hole_mask, hole_area in smoothing_structural_holes]
    smoothing_minimum_structural_retention = (
        min(smoothing_structural_retention)
        if smoothing_structural_retention else 1.0)
    # Dense image-guided repair already targets the real surface drop-outs.
    # A second median pass can blur the dark structural slots that supplied
    # the appearance evidence, so keep it disabled for this route and retain
    # the independently gated polygon simplification below.
    appearance_evidence_used = bool(
        used_image_texture_evidence or filled_image_boundary_notches)
    smoothing_allowed = geometry_policy == "full"
    smoothing_applied = (smoothing_allowed
                         and not image_texture_holes
                         and not image_boundary_notches
                         and before_components == after_components
                         and smoothing_pixels / max(area, 1)
                         <= effective_maximum_change_ratio
                         and smoothing_minimum_structural_retention >= .75)
    if smoothing_applied:
        refined = smoothed

    # Median smoothing can divide an irregular texture gap into a few tiny
    # enclosed remnants. Run one conservative micro-hole pass afterwards, but
    # do not reuse the broader texture-hole allowance here: long structural
    # openings that were split by visible supports must remain reviewable.
    post_smoothing_filled_holes = []
    if smoothing_applied:
        post_background = (refined == 0).astype(np.uint8)
        post_count, post_components, post_stats, _ = cv2.connectedComponentsWithStats(
            post_background, connectivity=8)
        post_border_labels = set(np.unique(np.concatenate((
            post_components[0, :], post_components[-1, :],
            post_components[:, 0], post_components[:, -1]))).tolist())
        for component in range(1, post_count):
            if component in post_border_labels:
                continue
            x, y, w, h, component_area = [int(value) for value in post_stats[component]]
            if (component_area <= maximum_hole_area
                    and max(w, h) <= maximum_hole_span):
                refined[post_components == component] = 1
                record = {"area": component_area,
                          "bbox": [x, y, x + w - 1, y + h - 1]}
                post_smoothing_filled_holes.append(record)
        filled_holes.extend(post_smoothing_filled_holes)

    changed = int(np.count_nonzero(refined != mask))
    smoothing_rejected_by_budget = False
    if (changed / max(area, 1) > effective_maximum_change_ratio
            and cleanup_only_changed
            and cleanup_only_changed / max(area, 1)
            <= effective_maximum_change_ratio):
        refined = cleanup_only
        filled_holes = cleanup_only_filled_holes
        post_smoothing_filled_holes = []
        smoothing_applied = False
        smoothing_rejected_by_budget = True
        changed = cleanup_only_changed
    if changed / max(area, 1) > effective_maximum_change_ratio:
        return shape, None
    if changed:
        replacement = _shape_from_mask(refined, shape["label"], shape["id"])
        if (replacement is None
                or not _valid_pallet_geometry_shape(replacement, width, height)):
            return shape, None
        for key, value in shape.items():
            if key not in {"id", "label", "points", "holes", "parts"}:
                replacement[key] = copy.deepcopy(value)
    else:
        replacement = copy.deepcopy(shape)
    if geometry_policy == "full":
        replacement, boundary_audit, replacement_mask = _simplify_pallet_boundary(
            replacement, mask, width, height,
            maximum_change_ratio=effective_maximum_change_ratio,
            image_rgb=image_rgb,
            minimum_compact_texture_candidate_area_ratio=(
                minimum_compact_texture_candidate_area_ratio))
    else:
        # Candidate v7: image evidence only authorizes its own local add-only
        # component.  It never authorizes median smoothing, island removal or
        # whole-contour simplification elsewhere.
        boundary_audit = None
        replacement_mask = _rasterize(
            replacement, width, height).astype(np.uint8)
    changed = int(np.count_nonzero(replacement_mask != mask))
    replacement_component_count = _significant_component_count(replacement_mask)
    replacement_foreground_component_count = _foreground_component_count(
        replacement_mask)
    component_merge_count = original_component_count - replacement_component_count
    if appearance_evidence_policy:
        additions = (replacement_mask != 0) & (mask == 0)
        if (np.any((mask != 0) & (replacement_mask == 0))
                or np.any(additions & ~authorized_additions)
                or not np.array_equal(replacement_mask, refined)
                or replacement_foreground_component_count
                != original_foreground_component_count):
            return shape, None
    if (replacement_component_count > original_component_count
            or component_merge_count < 0
            or (appearance_evidence_policy
                and component_merge_count != 0)
            or (component_merge_count
                and (not filled_image_boundary_notches
                     or component_merge_count
                     > PALLET_IMAGE_BOUNDARY_MAXIMUM_COMPONENT_MERGES))):
        return shape, None
    structural_holes = protected_structural_holes
    structural_retention = [
        int(np.count_nonzero(hole_mask & (replacement_mask == 0))) / hole_area
        for hole_mask, hole_area in structural_holes]
    minimum_structural_retention = (
        min(structural_retention) if structural_retention else None)
    required_structural_retention = (
        1.0 if appearance_evidence_policy else .75)
    if (minimum_structural_retention is not None
            and minimum_structural_retention < required_structural_retention):
        return shape, None
    if not changed and boundary_audit is None:
        return shape, None
    adjustment = {
        "reason_code": "pallet_texture_geometry_cleanup",
        "shape_id": shape["id"],
        "label": shape["label"],
        "filled_hole_count": len(filled_holes),
        "filled_hole_pixels": sum(item["area"] for item in filled_holes),
        "filled_texture_hole_count": len(filled_texture_holes),
        "filled_texture_hole_pixels": sum(item["area"] for item in filled_texture_holes),
        "filled_compact_texture_hole_count": len(filled_compact_texture_holes),
        "filled_compact_texture_hole_pixels": sum(
            item["area"] for item in filled_compact_texture_holes),
        "filled_image_texture_hole_count": len(filled_image_texture_holes),
        "filled_image_texture_hole_pixels": sum(
            item["area"] for item in filled_image_texture_holes),
        "filled_image_boundary_notch_count": len(filled_image_boundary_notches),
        "filled_image_boundary_notch_pixels": sum(
            item["area"] for item in filled_image_boundary_notches),
        "post_smoothing_filled_hole_count": len(post_smoothing_filled_holes),
        "post_smoothing_filled_hole_pixels": sum(
            item["area"] for item in post_smoothing_filled_holes),
        "removed_island_pixels": removed_island_pixels,
        "smoothing_applied": smoothing_applied,
        "smoothing_rejected_by_budget": smoothing_rejected_by_budget,
        "smoothing_kernel": smoothing_kernel if smoothing_applied else None,
        "smoothing_changed_pixels": smoothing_pixels if smoothing_applied else 0,
        "smoothing_minimum_structural_hole_retention": round(
            smoothing_minimum_structural_retention, 6),
        "changed_pixels": changed,
        "change_ratio": round(changed / area, 6),
        "maximum_hole_area": maximum_hole_area,
        "maximum_hole_span": maximum_hole_span,
        "maximum_texture_hole_area": maximum_texture_hole_area,
        "maximum_texture_hole_span": maximum_texture_hole_span,
        "maximum_compact_texture_hole_area": maximum_compact_texture_hole_area,
        "maximum_compact_texture_hole_span": maximum_compact_texture_hole_span,
        "maximum_compact_texture_aspect_ratio": maximum_compact_texture_aspect_ratio,
        "maximum_image_texture_hole_area": PALLET_IMAGE_TEXTURE_MAXIMUM_AREA,
        "maximum_image_texture_hole_span": PALLET_IMAGE_TEXTURE_MAXIMUM_SPAN,
        "maximum_image_texture_gray_delta": PALLET_IMAGE_TEXTURE_MAXIMUM_GRAY_DELTA,
        "maximum_image_texture_color_distance": (
            PALLET_IMAGE_TEXTURE_MAXIMUM_COLOR_DISTANCE),
        "image_boundary_close_kernel": PALLET_IMAGE_BOUNDARY_CLOSE_KERNEL,
        "maximum_image_boundary_notch_area": PALLET_IMAGE_BOUNDARY_MAXIMUM_AREA,
        "maximum_image_boundary_notch_span": PALLET_IMAGE_BOUNDARY_MAXIMUM_SPAN,
        "minimum_image_boundary_ring_gray": (
            PALLET_IMAGE_BOUNDARY_MINIMUM_RING_GRAY),
        "maximum_image_boundary_ring_color_spread": (
            PALLET_IMAGE_BOUNDARY_MAXIMUM_RING_COLOR_SPREAD),
        "maximum_image_boundary_component_merges": (
            PALLET_IMAGE_BOUNDARY_MAXIMUM_COMPONENT_MERGES),
        "effective_maximum_change_ratio": effective_maximum_change_ratio,
        "absolute_change_budget_pixels": (
            256 if appearance_evidence_policy else None),
        "add_only": appearance_evidence_policy,
        "required_structural_hole_retention": required_structural_retention,
        "minimum_compact_texture_candidate_area_ratio": (
            minimum_compact_texture_candidate_area_ratio),
        "preserved_component_count": replacement_component_count,
        "component_count_before": original_component_count,
        "component_count_after": replacement_component_count,
        "foreground_component_count_before": original_foreground_component_count,
        "foreground_component_count_after": replacement_foreground_component_count,
        "component_merge_count": component_merge_count,
        "geometry_cleanup_version": (
            8 if geometry_policy == "appearance_evidence_roi" else
            7 if geometry_policy == "appearance_evidence_only" else 6),
        "geometry_cleanup_policy": geometry_policy,
        "image_evidence_scale_gate_passed": image_evidence_scale_passed,
        "image_evidence_scale": image_evidence_scale,
        "appearance_evidence_used": appearance_evidence_used,
        "route": "candidate_only",
    }
    adjustment.update(boundary_audit or {
        "boundary_simplification_applied": False,
        "boundary_simplification_epsilon": None,
        "boundary_simplification_scale_band": (
            "large" if area >= 50000 else "medium" if area >= 5000 else "small"),
        "boundary_vertices_before": _pallet_geometry_vertex_count(replacement),
        "boundary_vertices_after": _pallet_geometry_vertex_count(replacement),
        "boundary_vertex_reduction_ratio": 0.0,
        "boundary_simplification_changed_pixels": 0,
        "minimum_structural_hole_retention": None,
    })
    adjustment["minimum_structural_hole_retention"] = (
        round(minimum_structural_retention, 6)
        if minimum_structural_retention is not None else None)
    if geometry_policy == "appearance_evidence_roi":
        adjustment["filled_image_boundary_notch_evidence"] = copy.deepcopy(
            filled_image_boundary_notches)
    return replacement, adjustment


def repeated_no_change_after_repairs(audit, minimum_rounds=3):
    """Find an exact no-change edit repeated while useful gap repairs finish.

    This is deliberately narrow: the current round must contain only applied
    ``fill_gap`` repairs plus no-change operations, and the same no-change edit
    (ignoring prose reason only) must occur in each of the last three rounds.
    The caller separately verifies that no review gap remains before stopping.
    """
    if len(audit) < minimum_rounds:
        return None

    def results(entry):
        decision = entry.get("decision") if isinstance(entry, dict) else None
        operations = decision.get("operations") if isinstance(decision, dict) else None
        outcomes = entry.get("tool_results") if isinstance(entry, dict) else None
        if not isinstance(operations, list) or not isinstance(outcomes, list):
            return []
        return [(operations[index], outcome) for index, outcome in enumerate(outcomes)
                if index < len(operations) and isinstance(operations[index], dict)
                and isinstance(outcome, dict)]

    current = results(audit[-1])
    if not current:
        return None
    applied = [operation for operation, outcome in current
               if outcome.get("status") == "applied"]
    if (not applied or any(operation.get("operation") != "fill_gap" for operation in applied)
            or any(outcome.get("status") not in {"applied", "no_change"}
                   for _, outcome in current)):
        return None

    def key(operation):
        identity = {name: copy.deepcopy(value) for name, value in operation.items()
                    if name != "reason"}
        return json.dumps(identity, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))

    current_no_change = {key(operation): operation for operation, outcome in current
                         if outcome.get("status") == "no_change"}
    for operation_key, operation in current_no_change.items():
        if all(any(outcome.get("status") == "no_change"
                   and key(candidate) == operation_key
                   for candidate, outcome in results(entry))
               for entry in audit[-minimum_rounds:]):
            return {name: copy.deepcopy(value) for name, value in operation.items()
                    if name != "reason"}
    return None


class AnnotationAgent:
    """Observe, choose tools, execute, and inspect again within a fixed budget."""

    def __init__(self, config, image, description, labels, max_rounds=100, on_progress=None,
                 reference=None):
        if type(max_rounds) is not int or not 1 <= max_rounds <= MAX_REVIEW_ROUNDS:
            raise ValueError("Agent 最大轮数须为 1～100")
        self.config, self.image, self.description, self.labels = config, image, description, labels
        self.reference = reference
        with Image.open(io.BytesIO(image)) as decoded:
            self.width, self.height = decoded.size
            self.image_rgb = np.asarray(decoded.convert("RGB"))
        self.max_rounds = max_rounds
        self.on_progress = on_progress or (lambda event: None)
        self.shapes, self.audit, self.queue, self.applied = [], [], [], []
        self.next_id = 1
        self.generated = 0
        self.failed_operations = {}
        self.sam_pending = None
        self.sam_points_by_shape = {}
        self.sam_fallback = set()
        self.candidate_filters = []
        self.candidate_geometry_adjustments = []
        self.sam_generation = {"mode": "not_started"}

    def progress(self, phase, **details):
        check_cancelled()
        event = {"phase": phase, "round": len(self.audit),
            "shapes": copy.deepcopy(self.shapes), "audit": copy.deepcopy(self.audit),
            "review_queue": copy.deepcopy(self.queue)}
        event.update(details)
        self.on_progress(event)

    def candidates(self, results, label):
        if not isinstance(results, list) or len(results) > 200:
            raise ValueError("SAM 3 单次返回候选格式无效或数量过多")
        shapes = []
        for result in results:
            if not isinstance(result, dict):
                raise ValueError("SAM 3 返回无效候选")
            parts = result.get("parts") or []
            if not parts:
                continue
            shape = {"id": self.next_id, "label": label, "points": parts[0].get("points", []),
                "holes": parts[0].get("holes", []), "parts": parts, "score": result.get("score")}
            if not valid_shape(shape, self.width, self.height):
                raise ValueError("SAM 3 返回无效轮廓")
            if str(label).casefold() == "pallet":
                shape, adjustment = refine_pallet_candidate_geometry(
                    shape, self.width, self.height, image_rgb=self.image_rgb)
                if adjustment is not None:
                    self.candidate_geometry_adjustments.append(adjustment)
            shapes.append(shape)
            self.next_id += 1
        return shapes

    def slope_crop_fallback(self):
        """Run fixed overlapping crops only when full-frame slope proposals all fail guards."""
        from .edit_agent import _rasterize, _shape_from_mask
        retained, filtered = [], []
        for crop_index, crop in enumerate(slope_overlap_crop_images(self.image)):
            left, top, right, bottom = crop["bbox"]
            crop_shapes = self.candidates(
                sam_text(crop["image"], "traversable surface", self.config), "slope")
            if any(not valid_shape(shape, crop["width"], crop["height"])
                   for shape in crop_shapes):
                raise ValueError("SAM 3 裁剪斜坡候选超出裁剪图像范围")
            crop_shapes, crop_filtered = filter_slope_candidate_geometry(
                crop_shapes, crop["height"])
            crop_box = crop["bbox"]
            for item in crop_filtered:
                item["crop_index"] = crop_index
                item["crop_bbox"] = crop_box
            filtered.extend(crop_filtered)
            for shape in crop_shapes:
                crop_mask = _rasterize(shape, crop["width"], crop["height"])
                full_mask = np.zeros((self.height, self.width), dtype=np.uint8)
                full_mask[top:bottom, left:right] = crop_mask
                mapped = _shape_from_mask(full_mask, "slope", shape["id"])
                if mapped is None or not valid_shape(mapped, self.width, self.height):
                    raise ValueError("SAM 3 裁剪斜坡候选映射到原图失败")
                mapped.update(score=shape.get("score"),
                              source="sam_text_overlap_crop_fallback",
                              crop_index=crop_index, crop_bbox=crop_box)
                retained.append(mapped)
        return retained, filtered

    def generate(self):
        prompt_items = self.description.get("prompts", [])
        for item in prompt_items:
            if (not isinstance(item, dict) or item.get("label") not in self.labels or
                    not isinstance(item.get("text"), str) or not item["text"].strip()):
                raise ValueError("SAM 3 提示缺少有效类别或文本")
        batched = None
        if len(prompt_items) > 1 and self.config.get("sam_multi_prompt", True):
            prompt_map = {item["label"]: item["text"] for item in prompt_items}
            if len(prompt_map) != len(prompt_items):
                raise ValueError("SAM 3 多提示类别不能重复")
            self.progress("generating", operation="sam_multi_prompt",
                          outcome={"prompt_count": len(prompt_map)})
            try:
                batched, diagnostics = sam_text_multi(self.image, prompt_map, self.config)
                self.sam_generation = {"mode": "multi_prompt", "prompt_count": len(prompt_map),
                                       "diagnostics": copy.deepcopy(diagnostics)}
            except RunCancelled:
                raise
            except Exception as error:
                self.sam_generation = {"mode": "sequential_fallback",
                                       "prompt_count": len(prompt_map),
                                       "error": str(error)[:500]}
                self.progress("generating", operation="sam_multi_prompt_fallback",
                              outcome=copy.deepcopy(self.sam_generation))
        elif prompt_items:
            self.sam_generation = {"mode": "sequential", "prompt_count": len(prompt_items)}
        for item in prompt_items:
            self.progress("generating", prompt=item["text"], label=item["label"])
            results = (batched[item["label"]]["results"] if batched is not None else
                       sam_text(self.image, item["text"], self.config))
            shapes = self.candidates(results, item["label"])
            if (item["label"] == "other_robot" and not shapes
                    and self.config.get(
                        "other_robot_semantic_fallback_visible") is True):
                self.progress(
                    "generating", prompt=OTHER_ROBOT_SEMANTIC_FALLBACK_PROMPT,
                    label=item["label"],
                    operation="other_robot_semantic_fallback")
                raw_fallback = self.candidates(
                    sam_text(self.image, OTHER_ROBOT_SEMANTIC_FALLBACK_PROMPT,
                             self.config), item["label"])
                shapes, fallback_filters = (
                    select_other_robot_semantic_fallback_candidates(raw_fallback))
                self.candidate_filters.extend(fallback_filters)
                self.sam_generation.setdefault("semantic_fallbacks", []).append({
                    "label": item["label"],
                    "trigger": "visible_label_baseline_empty",
                    "prompt": OTHER_ROBOT_SEMANTIC_FALLBACK_PROMPT,
                    "raw_candidate_count": len(raw_fallback),
                    "retained_candidate_count": len(shapes),
                    "suppressed_candidate_count": len(fallback_filters),
                    "maximum_candidates":
                        OTHER_ROBOT_SEMANTIC_FALLBACK_MAX_CANDIDATES,
                    "candidate_only": True,
                    "human_truth": False,
                    "approved_for_training": False,
                })
            if item["label"] == "slope" and item["text"] == "traversable surface":
                shapes, filtered = filter_slope_candidate_geometry(shapes, self.height)
                self.candidate_filters.extend(filtered)
                if not shapes:
                    self.progress("generating", prompt=item["text"], label=item["label"],
                                  operation="slope_overlap_crop_fallback")
                    shapes, crop_filtered = self.slope_crop_fallback()
                    self.candidate_filters.extend(crop_filtered)
            self.shapes.extend(shapes)
            self.generated += len(shapes)
            self.progress("generated")
        self.shapes, slope_pallet_filters = filter_slope_pallet_conflicts(
            self.shapes, self.width, self.height)
        self.candidate_filters.extend(slope_pallet_filters)
        self.shapes, slope_floor_filters = filter_slope_floor_conflicts(
            self.shapes, self.width, self.height)
        self.candidate_filters.extend(slope_floor_filters)
        self.shapes, slope_ego_filters = filter_slope_reference_ego_conflicts(
            self.shapes, self.reference, self.width, self.height)
        self.candidate_filters.extend(slope_ego_filters)
        self.shapes, slope_current_ego_filters = filter_slope_current_ego_conflicts(
            self.shapes, self.width, self.height)
        self.candidate_filters.extend(slope_current_ego_filters)
        self.shapes, duplicate_filters = suppress_near_duplicate_candidates(
            self.shapes, self.width, self.height,
            preserve_class_conflicts=self.reference is None)
        self.candidate_filters.extend(duplicate_filters)
        self.generated = len(self.shapes)
        if self.candidate_filters:
            self.progress("generated", operation="filter_candidates",
                          outcome={"suppressed": copy.deepcopy(self.candidate_filters)})

    def execute(self, *, skip_generation=False):
        stop_reason = "max_rounds"
        converged = False
        generated_shapes = []
        try:
            if not skip_generation:
                self.generate()
            # Preserve the last fully validated SAM draft as a review fallback.
            # VLM edits may remove individual false positives, but an unreviewed
            # machine decision must never turn a visible draft into a blank frame.
            generated_shapes = copy.deepcopy(self.shapes)
            seen = {_fingerprint(self.shapes)}
            for number in range(1, self.max_rounds + 1):
                self.progress("observing", round=number)
                pending = self.sam_pending
                comparison = ({"shape_id": pending["shape_id"],
                    "before_image": candidate_overlay(self.image, pending["before"])} if pending else None)
                # Gap semantics require a human reference example. For a generic,
                # unreferenced class batch, review visible candidates conservatively
                # without inventing reference-backed floor-hole rules.
                floor_review = any(
                    str(label).strip().casefold() in {"地面", "floor", "ground"}
                    for label in self.labels)
                # Internal-gap semantics are meaningful only for traversable-ground
                # masks. Treating pallet/person/robot mask holes as floor gaps adds
                # irrelevant evidence pages and can make the VLM chase reference
                # instance counts instead of judging the target image.
                all_holes = (candidate_holes(self.shapes, self.labels, self.width, self.height)
                             if self.reference is not None and floor_review else [])
                holes = select_review_gaps(
                    all_holes, self.config.get("max_review_gaps", 8),
                    self.config.get("minimum_review_gap_area")) if all_holes else []
                if len(all_holes) > len(holes):
                    self.progress("observing", round=number, operation="limit_review_gaps",
                                  outcome={"reviewed": len(holes),
                                           "suppressed_noise": len(all_holes) - len(holes)})
                gap_images = gap_detail_images(self.image, self.shapes, holes)
                try:
                    decision = review(self.config, self.image, candidate_overlay(self.image, self.shapes),
                        self.description, [_shape_summary(shape) for shape in self.shapes],
                        self.labels, self.audit, reference=self.reference, sam_comparison=comparison,
                        hole_image=hole_overlay(self.image, holes, self.shapes), holes=holes,
                        gap_images=gap_images,
                        duplicate_groups=(_exact_duplicate_groups(self.shapes, self.width, self.height)
                                          if self.config.get("llm_duplicate_review", False) else None))
                except ReviewProtocolError as error:
                    stop_reason = "invalid_model_output"
                    self.queue.append({
                        "reason": "VLM/Agent 连续返回无效修正；已保留 SAM 候选并转人工复核",
                        "reason_code": "agent_invalid_model_output",
                        "route": "human_review_required",
                        "error": str(error)[:500],
                        "last_validation_error": error.last_error,
                    })
                    self.progress("review_required", round=number,
                                  outcome={"reason_code": "agent_invalid_model_output"})
                    break
                if pending:
                    assessment = decision.get("sam_assessment")
                    if not isinstance(assessment, dict) or type(assessment.get("improved")) is not bool:
                        raise ValueError("GPT 未评估 SAM 3 点提示结果")
                    if not assessment["improved"]:
                        self.shapes = copy.deepcopy(pending["before"])
                        self.sam_fallback.add(pending["shape_id"])
                        self.sam_points_by_shape.pop(pending["shape_id"], None)
                        decision["sam_rollback"] = {"shape_id": pending["shape_id"],
                            "reason": assessment.get("reason", "未改进")}
                    self.sam_pending = None
                for repair in decision.get("protocol_repairs", []):
                    self.queue.append({
                        "reason": "VLM 漏答该缺口；系统未修改此区域，已转人工复核",
                        **copy.deepcopy(repair),
                    })
                before = _fingerprint(self.shapes)
                entry = {"round": number, "decision": decision, "tool_results": [],
                    "before_shape_count": len(self.shapes), "before_fingerprint": before}
                if decision.get("sam_rollback"):
                    entry["sam_rollback"] = decision["sam_rollback"]
                self.audit.append(entry)
                sam_failed = False
                for item in decision["operations"]:
                    self.progress("editing", round=number, operation=item)
                    old = copy.deepcopy(self.shapes)
                    old_count = len(old)
                    operation_key = json.dumps(item, sort_keys=True, ensure_ascii=False,
                                               separators=(",", ":"))
                    if operation_key in self.failed_operations:
                        error_text = self.failed_operations[operation_key]
                        outcome = {"operation": item, "status": "skipped",
                                   "error": error_text,
                                   "reason": "相同操作此前已失败，停止重复重试"}
                        self.queue.append({"reason": outcome["reason"],
                                           "operation": item, "error": error_text})
                        outcome["before_shape_count"] = old_count
                        outcome["after_shape_count"] = len(self.shapes)
                        entry["tool_results"].append(outcome)
                        self.progress("edited", round=number, outcome=outcome)
                        continue
                    try:
                        if not isinstance(item, dict):
                            raise ValueError("操作必须为对象")
                        if item.get("operation") == "sam_text":
                            raise ValueError("复查阶段禁止使用 sam_text 整幅重新分割；请使用针对已有候选的 sam_points 或 OpenCV 局部操作")
                        if item.get("operation") == "sam_points":
                            if self.sam_pending is not None:
                                raise ValueError("SAM 3 点提示结果尚未经过 GPT 复核")
                            ids = item.get("shape_ids")
                            if not isinstance(ids, list) or len(ids) != 1 or type(ids[0]) is not int:
                                raise ValueError("复查阶段 sam_points 必须针对一个已有候选")
                            if ids[0] not in {shape["id"] for shape in self.shapes}:
                                raise ValueError("sam_points 的候选 ID 不存在")
                            if ids[0] in self.sam_fallback:
                                raise ValueError("该候选点提示未改进，请改用 OpenCV 局部调整")
                            polygon = item.get("polygon")
                            from .edit_agent import _valid_region, _rasterize, _shape_from_mask
                            if not _valid_region(polygon, self.width, self.height):
                                raise ValueError("SAM 点提示局部 polygon 无效")
                            points = item.get("points")
                            if not isinstance(points, list) or not 1 <= len(points) <= 32:
                                raise ValueError("SAM 点提示数量须为 1～32")
                            for point in points:
                                if (not isinstance(point, dict) or type(point.get("label")) is not int or
                                        point["label"] not in (0, 1) or
                                        any(type(point.get(key)) not in (int, float) or
                                            not math.isfinite(point[key]) for key in ("x", "y")) or
                                        not 0 <= point["x"] < self.width or not 0 <= point["y"] < self.height):
                                    raise ValueError("SAM 点提示坐标或类型无效")
                            coordinate_labels = {}
                            for point in points:
                                coordinate = (float(point["x"]), float(point["y"]))
                                coordinate_labels.setdefault(coordinate, set()).add(point["label"])
                            if any(len(point_labels) > 1 for point_labels in coordinate_labels.values()):
                                raise ValueError("SAM 点提示在同一坐标存在冲突的正负标签")
                            if len(points) > 1 and len(coordinate_labels) == 1:
                                raise ValueError("SAM 点提示只有重复的同一个点，已拒绝无效几何修正")
                            if not any(point["label"] == 1 for point in points):
                                raise ValueError("SAM 点提示至少需要一个有效正点")
                            previous_points = self.sam_points_by_shape.get(ids[0], [])
                            combined = {(point["x"], point["y"]): point for point in previous_points + points}
                            all_points = list(combined.values())
                            if previous_points and len(all_points) == len({(point["x"], point["y"])
                                                                         for point in previous_points}):
                                raise ValueError("SAM 点提示没有新增有效坐标，已拒绝重复修正")
                            if not any(point["label"] == 1 for point in all_points) or len(all_points) > 32:
                                raise ValueError("SAM 累计提示点须包含正点且不超过 32 个")
                            old_shape = next(shape for shape in self.shapes if shape["id"] == ids[0])
                            prior_mask = _rasterize(old_shape, self.width, self.height)
                            encoded = io.BytesIO()
                            Image.fromarray(prior_mask * 255).save(encoded, "PNG")
                            result = sam_request(self.config, "/segment_points", self.image, points=all_points,
                                mask=base64.b64encode(encoded.getvalue()).decode("ascii"), polygon=polygon)
                            parts = result.get("parts") if isinstance(result, dict) else None
                            if not parts:
                                raise ValueError("SAM 3 点提示未生成有效候选")
                            if result.get("warnings"):
                                raise ValueError("SAM 3 点提示不满足输入点：" + "; ".join(result["warnings"]))
                            sam_shape = {"id": ids[0], "label": old_shape["label"], "parts": parts}
                            if not valid_shape(sam_shape, self.width, self.height):
                                raise ValueError("SAM 3 返回无效轮廓")
                            before_mask = _rasterize(old_shape, self.width, self.height)
                            sam_mask = _rasterize(sam_shape, self.width, self.height)
                            region = np.zeros_like(before_mask)
                            cv2.fillPoly(region, [np.asarray(polygon, dtype=np.int32)], 1)
                            before_mask[region != 0] = sam_mask[region != 0]
                            new_shape = _shape_from_mask(before_mask, old_shape["label"], ids[0])
                            if new_shape is None or np.array_equal(before_mask, _rasterize(old_shape, self.width, self.height)):
                                raise ValueError("SAM 点提示局部修复后区域为空或没有变化")
                            self.sam_pending = {"shape_id": ids[0], "before": old}
                            self.sam_points_by_shape[ids[0]] = all_points
                            self.shapes = [shape for shape in self.shapes if shape["id"] != ids[0]] + [new_shape]
                            self.generated += 1
                            outcome = {"operation": "sam_points", "shape_ids": ids,
                                       "result_ids": [ids[0]], "status": "applied",
                                       "point_count": len(all_points)}
                        else:
                            updated, applied, queue = apply_operations(
                                self.shapes, [item], self.labels, self.width, self.height,
                                next_id=self.next_id, gap_regions=holes,
                                allow_high_risk_reclassify=self.reference is not None,
                                allow_large_global_surface_expansion=self.reference is not None)
                            self.shapes = updated
                            self.next_id = max([self.next_id-1] + [s["id"] for s in updated]) + 1
                            self.queue.extend(queue)
                            outcome = {"operation": item.get("operation"), "applied": applied,
                                "review_queue": queue, "status": "rejected" if queue else "applied"}
                        if _fingerprint(old) == _fingerprint(self.shapes) and outcome["status"] == "applied":
                            outcome["status"] = "no_change"
                        if outcome["status"] == "applied":
                            self.applied.append(outcome)
                            if item.get("operation") == "sam_points":
                                entry["tool_results"].append(outcome)
                                outcome["before_shape_count"] = old_count
                                outcome["after_shape_count"] = len(self.shapes)
                                self.progress("edited", round=number, outcome=outcome)
                                break
                    except RunCancelled:
                        raise
                    except Exception as error:
                        self.shapes = old
                        error_text = str(error)[:500]
                        self.failed_operations[operation_key] = error_text
                        outcome = {"operation": item, "status": "error", "error": error_text}
                        ids = item.get("shape_ids") if isinstance(item, dict) else None
                        if (isinstance(item, dict) and item.get("operation") == "sam_points"
                                and isinstance(ids, list) and len(ids) == 1
                                and type(ids[0]) is int and any(s["id"] == ids[0] for s in old)):
                            self.sam_fallback.add(ids[0])
                            self.sam_points_by_shape.pop(ids[0], None)
                            outcome["fallback"] = "已保留原候选，下一轮改用 OpenCV 局部调整"
                            sam_failed = True
                        else:
                            self.queue.append({"reason": "Agent 工具执行失败", "operation": item,
                                "error": error_text})
                    entry["tool_results"].append(outcome)
                    outcome["before_shape_count"] = old_count
                    outcome["after_shape_count"] = len(self.shapes)
                    self.progress("edited", round=number, outcome=outcome)
                after = _fingerprint(self.shapes)
                entry["after_shape_count"] = len(self.shapes)
                entry["after_fingerprint"] = after
                # A 'done' sent with edits cannot validate their result. Observe again.
                if decision["status"] == "done" and not decision["operations"]:
                    converged = not self.queue
                    stop_reason = "done" if converged else "review_required"
                    break
                # A proposed SAM edit must be assessed even if it revisits an
                # earlier geometry. Failed/no-change prompts get one fresh
                # observation so GPT can choose an OpenCV fallback.
                if self.sam_pending is not None or sam_failed:
                    continue
                if after == before:
                    stop_reason = "no_progress"
                    break
                repeated = repeated_no_change_after_repairs(self.audit)
                if repeated is not None and self.reference is not None:
                    remaining_holes = candidate_holes(
                        self.shapes, self.labels, self.width, self.height)
                    if not remaining_holes:
                        entry["early_stop"] = {
                            "reason_code": "repeated_no_change_after_repairs",
                            "operation": repeated,
                            "consecutive_rounds": 3,
                        }
                        stop_reason = "repeated_no_change_after_repairs"
                        break
                if after != before and after in seen:
                    stop_reason = "cycle"
                    break
                seen.add(after)
        except RunCancelled:
            raise
        except Exception as error:
            stop_reason = "error"
            self.queue.append({"reason": "Agent 调用失败", "error": str(error)[:500]})
        if not converged and stop_reason not in ("error", "invalid_model_output", "review_required"):
            self.queue.append({"reason": {"max_rounds": "达到最大轮数，最新候选尚未确认",
                "no_progress": "本轮未产生有效修改", "cycle": "检测到循环修改",
                "repeated_no_change_after_repairs":
                    "缺口修补已完成，但同一无效操作连续重复；已停止自动重试并转人工复核"}[stop_reason]})
        if self.sam_pending is not None:
            self.shapes = copy.deepcopy(self.sam_pending["before"])
            self.queue.append({"reason": "SAM 3 点提示结果尚未经过下一轮 GPT 复核，已恢复修改前候选"})
            converged = False
            self.sam_pending = None
        if generated_shapes and not self.shapes:
            self.shapes = copy.deepcopy(generated_shapes)
            self.queue.append({
                "reason": "Agent 将整帧候选删空；已恢复原始 SAM 候选并转人工复核",
                "reason_code": "agent_empty_result_rollback",
                "route": "human_review_required",
                "restored_candidate_count": len(generated_shapes),
            })
            converged = False
            stop_reason = "empty_result_rollback"
        if self.reference is None:
            self.queue.extend(candidate_class_overlap_conflicts(
                self.shapes, self.width, self.height))
        result = {"shapes": self.shapes, "candidates_added": self.generated, "applied": self.applied,
            "review_queue": self.queue, "agent": {"converged": converged,
            "rounds": len(self.audit), "max_rounds": self.max_rounds,
            "stop_reason": stop_reason, "audit": self.audit,
            "candidate_filters": self.candidate_filters,
            "candidate_geometry_adjustments": self.candidate_geometry_adjustments,
            "sam_generation": copy.deepcopy(self.sam_generation)}}
        self.progress("finished", result=result)
        return result


def run(config, image, description, labels, max_rounds=100, on_progress=None, reference=None):
    return AnnotationAgent(config, image, description, labels, max_rounds, on_progress,
                           reference=reference).execute()


def generate_only(config, image, description, labels, max_rounds=100, on_progress=None,
                  reference=None):
    """Generate guarded SAM candidates without waiting for the VLM review loop.

    The returned shapes are still machine candidates.  This entry point exists
    so the workbench can display useful contours while the normal Agent review
    continues through :func:`review_existing`; it never promotes truth or skips
    the review gate silently.
    """
    agent = AnnotationAgent(config, image, description, labels, max_rounds, on_progress,
                            reference=reference)
    agent.generate()
    result = {
        "shapes": agent.shapes,
        "candidates_added": agent.generated,
        "applied": [],
        "review_queue": copy.deepcopy(agent.queue),
        "agent": {
            "converged": False,
            "rounds": 0,
            "max_rounds": agent.max_rounds,
            "stop_reason": "review_deferred",
            "audit": [],
            "candidate_filters": copy.deepcopy(agent.candidate_filters),
            "sam_generation": copy.deepcopy(agent.sam_generation),
        },
    }
    agent.progress("review_deferred", result=result)
    return result


def review_existing(config, image, description, labels, shapes, max_rounds=100,
                    on_progress=None, reference=None, candidate_filters=None,
                    sam_generation=None):
    """Run the unchanged Agent review loop against an atomically bound draft."""
    agent = AnnotationAgent(config, image, description, labels, max_rounds, on_progress,
                            reference=reference)
    candidate_shapes = copy.deepcopy(shapes)
    if (not candidate_shapes
            or any(not isinstance(shape, dict)
                   or shape.get("label") not in labels
                   or type(shape.get("id")) is not int
                   or not valid_shape(shape, agent.width, agent.height)
                   for shape in candidate_shapes)):
        raise ValueError("待精修机器候选包含无效类别、ID 或轮廓")
    candidate_ids = [shape["id"] for shape in candidate_shapes]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("待精修机器候选 ID 重复")
    agent.shapes = candidate_shapes
    agent.next_id = max(candidate_ids) + 1
    agent.generated = len(candidate_shapes)
    agent.candidate_filters = copy.deepcopy(candidate_filters or [])
    agent.sam_generation = copy.deepcopy(
        sam_generation or {"mode": "persisted_machine_candidate"})
    return agent.execute(skip_generation=True)
