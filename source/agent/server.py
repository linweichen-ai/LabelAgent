"""Flask workbench for MCAP review and local annotation."""
import io
import copy
import json
import math
import os
import shutil
import threading
import time
import uuid
import urllib.request
import urllib.error
from urllib.parse import urlsplit
import base64
import hashlib
import marshal
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from datetime import datetime, timezone
from pathlib import Path
from werkzeug.utils import secure_filename
from flask import Flask, jsonify, request, send_file
from PIL import Image, ImageDraw
import numpy as np

from .media import IMAGE_SUFFIXES, SUPPORTED_SUFFIXES, open_media, read_image
from .sensors import cloud_snapshot
from .candidate_import import (prepare_candidate_import, merge_candidate_import,
                               merge_candidate_labels)
from .workflow import (DESCRIPTION_INPUT_VERSION, MAX_REVIEW_ROUNDS, RunCancelled,
                       cancellation, candidate_labels_with_recall_probes,
                       canonical_unreferenced_description, describe,
                       discover_visible_labels, reference_labels, restrict_description,
                       generate_only, review_existing, run, trace_calls,
                       candidate_overlay, reference_overlay, chat, chat_batch,
                       refine_pallet_candidate_geometry,
                       ego_bottom_point_candidates,
                       filter_slope_current_ego_conflicts,
                       sam_text_multi_prefetch, sam_text_prefetch,
                       slope_overlap_crop_images,
                       VISIBLE_LABEL_DEFINITIONS)
from .candidate_class_audit import (audit_candidate_class_blind, candidate_fingerprint,
                                    audit_high_risk_candidates,
                                    candidate_requires_class_audit,
                                    validate_blind_candidate_class_decision)

from . import candidate_class_audit as _candidate_audit_module

_CANDIDATE_AUDIT_LOADED_CODE_SHA256 = hashlib.sha256(marshal.dumps([
    (name, value.__code__) for name, value in sorted(vars(_candidate_audit_module).items())
    if getattr(value, "__module__", None) == _candidate_audit_module.__name__
    and hasattr(value, "__code__")
])).hexdigest()

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(os.environ.get("LABELAGENT_WORK_DIR") or ROOT / "work").resolve()
WEB = ROOT / "source" / "web"
WORK.mkdir(exist_ok=True)
app = Flask(__name__, static_folder=str(WEB), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024 * 1024
state = {"video": None, "topic": None, "labels": [], "truth": {}, "reference_frames": {}, "config": {}, "candidates": {}, "candidates_by_topic": {}, "agent_runs": {}, "agent_runs_by_topic": {}, "candidate_revisions": {}, "candidate_revisions_by_topic": {}, "candidate_review_status": {}, "candidate_review_proposals": [], "human_review_transactions": {}, "rejected_candidates": {}}
lock = threading.RLock()
file_activation_intent_lock = threading.RLock()
file_activation_generation = 0
run_status_lock = threading.Lock()
run_summary_lock = threading.Lock()
run_claim_lock = threading.Lock()
model_prewarm_lock = threading.Lock()
reference_discovery_cache_lock = threading.Lock()
reference_discovery_cache = {}
REFERENCE_DISCOVERY_CACHE_TTL_SECONDS = 300
REFERENCE_DISCOVERY_CACHE_MAX_ENTRIES = 16
PRIMARY_MODEL_PREWARM_TOKENS = 2048
run_status = {"status": "idle"}
run_stop_event = threading.Event()
run_task_id = None
refinement_cancel_lock = threading.Lock()
refinement_cancel_intents = {}
REFINEMENT_CANCEL_TTL_SECONDS = 300
file_sessions_dirty = False
batch_review_index_cache = None
DEFAULT_REVIEW_ROUNDS = 6
RUN_STORAGE_WARNING_BYTES = 2 * 1024 * 1024 * 1024
RUN_STORAGE_BASE_RESERVE_BYTES = 256 * 1024 * 1024
RUN_STORAGE_PER_FRAME_BYTES = 8 * 1024 * 1024
REVIEW_PACKET_SCHEMA = "labelagent.slope-review-packet.v1"
REVIEW_PACKET_ASSISTANT_SCHEMA = "labelagent.slope-assistant-review.v1"
REVIEW_PACKET_DECISION_SCHEMA = "labelagent.slope-review-decision.v1"
REVIEW_PACKET_ACCEPT_DECISIONS = {
    "slope_present_and_mask_acceptable",
    "slope_present_mask_needs_correction",
    "no_slope_present",
    "uncertain",
}


def claim_file_activation():
    """Give one incoming file-selection request an authoritative order.

    This lock is deliberately independent from the main state lock.  A newer
    browser tab can therefore supersede an older request while the older file
    is still opening or decoding its compatibility frame.
    """
    global file_activation_generation
    with file_activation_intent_lock:
        file_activation_generation += 1
        return file_activation_generation


def current_file_activation_generation():
    with file_activation_intent_lock:
        return file_activation_generation


def reference_discovery_cache_key(video, topic, frame, input_sha256, labels,
                                  config_sha256, activation_generation=None):
    """Bind one speculative discovery result to the exact active input.

    This is intentionally stricter than the VLM response cache.  It includes
    the file-load generation and decoded image digest so a dataset, camera,
    frame, label-registry or configuration change can never reuse stale
    machine evidence.
    """
    if activation_generation is None:
        activation_generation = current_file_activation_generation()
    return _sha256_json({
        "schema_version": "labelagent.reference-discovery-cache-key.v1",
        "activation_generation": activation_generation,
        "video_identity": id(video),
        "file": str(video.path),
        "topic": topic,
        "frame": frame,
        "input_sha256": input_sha256,
        "labels": list(labels),
        "config_sha256": config_sha256,
    })


def _prune_reference_discovery_cache(now=None):
    now = time.monotonic() if now is None else now
    expired = [key for key, value in reference_discovery_cache.items()
               if now - value["stored_at"] > REFERENCE_DISCOVERY_CACHE_TTL_SECONDS]
    for key in expired:
        reference_discovery_cache.pop(key, None)
    while len(reference_discovery_cache) > REFERENCE_DISCOVERY_CACHE_MAX_ENTRIES:
        oldest = min(reference_discovery_cache,
                     key=lambda key: reference_discovery_cache[key]["stored_at"])
        reference_discovery_cache.pop(oldest, None)


def store_reference_discovery_cache(key, discovery, retry_count, now=None):
    now = time.monotonic() if now is None else now
    with reference_discovery_cache_lock:
        reference_discovery_cache[key] = {
            "stored_at": now,
            "discovery": copy.deepcopy(discovery),
            "retry_count": int(retry_count),
        }
        _prune_reference_discovery_cache(now)


def get_reference_discovery_cache(key, now=None):
    now = time.monotonic() if now is None else now
    with reference_discovery_cache_lock:
        _prune_reference_discovery_cache(now)
        value = reference_discovery_cache.get(key)
        if value is None:
            return None
        return copy.deepcopy(value["discovery"]), value["retry_count"]


def clear_reference_discovery_cache():
    """Test/deployment helper; cached discovery is never authoritative state."""
    with reference_discovery_cache_lock:
        reference_discovery_cache.clear()


def assert_current_file_activation(generation):
    if generation != file_activation_generation:
        raise StaleHumanReviewError(
            "已有更新的文件选择，本次过期加载未应用；请以最后一次选择为准")


def assert_request_file_activation(payload, key="file_activation_generation"):
    """Bind a read or mutation to the exact browser file activation.

    A path and topic are not sufficient when the same file is reloaded.  The
    monotonically increasing activation generation prevents an older browser
    window from reading, promoting, rejecting, or referencing the newly loaded
    session merely because its path and topic still match.
    """
    generation = payload.get(key)
    if type(generation) is not int:
        raise ValueError(f"{key} 必须是当前文件载入代次整数")
    with file_activation_intent_lock:
        if generation != file_activation_generation:
            raise StaleHumanReviewError(
                "数据文件已被重新载入，旧窗口请求已拒绝；请刷新后重新操作")
    return generation


def assert_request_workspace_scope(payload, *, generation_key="file_activation_generation",
                                   path_key="path", topic_key="topic"):
    """Require one request to target the exact active workspace and load."""
    generation = assert_request_file_activation(payload, generation_key)
    expected_path = payload.get(path_key)
    expected_topic = payload.get(topic_key)
    if not isinstance(expected_path, str):
        raise ValueError(f"{path_key} 必须是当前工作区数据路径")
    if expected_topic is not None and not isinstance(expected_topic, str):
        raise ValueError(f"{topic_key} 必须是当前图像相机或 null")
    video = state.get("video")
    current_path = str(video.path) if video is not None else ""
    if (expected_path != current_path
            or expected_topic != state.get("topic")):
        raise StaleHumanReviewError(
            "数据文件或相机已变化，旧窗口请求已拒绝；请刷新后重新操作")
    return generation, expected_path, expected_topic


def assert_request_data_scope(payload, *, generation_key="file_activation_generation",
                              path_key="path", topic_key="topic"):
    """Require one request to target the exact active file, topic and load."""
    generation, expected_path, expected_topic = assert_request_workspace_scope(
        payload, generation_key=generation_key, path_key=path_key,
        topic_key=topic_key)
    if state.get("video") is None or not expected_path:
        raise ValueError(f"{path_key} 必须是当前数据文件路径")
    if not isinstance(expected_topic, str) or not expected_topic:
        raise ValueError(f"{topic_key} 必须是当前图像相机")
    return generation, expected_path, expected_topic


def _normalize_refinement_id(value):
    """Return a canonical UUID for one deferred refinement request."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("refinement_id 必须是 UUID 字符串")
    try:
        return uuid.UUID(value).hex
    except (ValueError, AttributeError, TypeError) as error:
        raise ValueError("refinement_id 必须是有效 UUID") from error


def _prune_refinement_cancel_intents(now=None):
    now = time.monotonic() if now is None else now
    expired = [key for key, recorded_at in refinement_cancel_intents.items()
               if now - recorded_at > REFINEMENT_CANCEL_TTL_SECONDS]
    for key in expired:
        refinement_cancel_intents.pop(key, None)
    while len(refinement_cancel_intents) > 256:
        oldest = min(refinement_cancel_intents, key=refinement_cancel_intents.get)
        refinement_cancel_intents.pop(oldest, None)


def _record_refinement_cancel(refinement_id):
    with refinement_cancel_lock:
        _prune_refinement_cancel_intents()
        refinement_cancel_intents[refinement_id] = time.monotonic()


def _consume_refinement_cancel(refinement_id):
    if refinement_id is None:
        return False
    with refinement_cancel_lock:
        _prune_refinement_cancel_intents()
        return refinement_cancel_intents.pop(refinement_id, None) is not None


def Video(path):
    """Open local media with a content-invalidated metadata index."""
    return open_media(path, WORK / "media_index")


def _update_run_status_locked(**changes):
    """Update the live job contract while run_status_lock is held."""
    now = time.time()
    previous_phase = run_status.get("phase")
    run_status.update(changes)
    if "phase" in changes and changes.get("phase") != previous_phase:
        run_status["phase_started_at"] = now
    run_status["updated_at"] = now
    run_status["status_sequence"] = int(run_status.get("status_sequence", 0)) + 1


def run_claimed(handler):
    # 非阻塞占用先于 state/status 锁，直到响应和异常收尾结束才释放。
    # 终态本身不释放占用，因此旧任务的回调和写盘不会进入下一任务代际。
    @wraps(handler)
    def invoke(*args, **kwargs):
        global run_task_id, run_stop_event
        if not run_claim_lock.acquire(blocking=False):
            raise RuntimeError("已有推理任务正在运行，请稍后重试")
        task_id = uuid.uuid4().hex
        try:
            with run_status_lock:
                if run_status.get("status") in {"running", "stopping"}:
                    raise RuntimeError("已有自动标注任务正在运行")
                previous_status = run_status.copy()
                run_task_id = task_id
                run_stop_event = threading.Event()
                started_at = time.time()
                _update_run_status_locked(status="running", phase="claiming", task_id=task_id,
                                          started_at=started_at, phase_started_at=started_at,
                                          events=[], shape_count=0)
            try:
                result = handler(*args, **kwargs)
                with run_status_lock:
                    if run_task_id == task_id and run_status.get("phase") == "claiming":
                        _update_run_status_locked(status="completed", phase="completed")
                return result
            except BaseException as error:
                with run_status_lock:
                    if run_task_id == task_id and run_status.get("phase") == "claiming":
                        run_status.clear()
                        run_status.update(previous_status)
                    elif run_task_id == task_id and run_status.get("status") in {"running", "stopping"}:
                        _update_run_status_locked(
                            status="stopped" if run_stop_event.is_set() else "failed",
                            phase="stopped" if run_stop_event.is_set() else "failed",
                            error=str(error)[:500], frame=None)
                raise
        finally:
            with run_status_lock:
                if run_task_id == task_id:
                    run_task_id = None
            run_claim_lock.release()
    return invoke
point_segment_lock = threading.Lock()
point_segment_flights_lock = threading.Lock()
point_segment_flights = {}
POINT_SEGMENT_TIMEOUT_SECONDS = 180
REFERENCE_DISCOVERY_RETRY_ERRORS = (
    "大模型未返回有效 JSON",
    "VLM 类别判断存在未知或重复标签",
    "VLM 类别判断与当前标签表不一致",
    "VLM 必须逐项判断当前标签表中的每个类别",
)
TRANSIENT_RUN_ERROR_MARKERS = (
    "timed out", "timeout", "connection reset", "connection refused",
    "connection aborted", "remote end closed", "temporarily unavailable",
    "service unavailable", "too many requests", "http error 429",
    "http error 500", "http error 502", "http error 503", "http error 504",
    "http 429", "http 500", "http 502", "http 503", "http 504",
    "local vlm is busy", "vlm is busy",
    "超时", "连接被重置", "连接被拒绝", "积极拒绝", "winerror 10061",
    "连接中断", "暂时不可用", "服务不可用",
)


def transient_run_failure(result):
    """Return a bounded retry reason for transport/service failures only.

    Model protocol and deterministic geometry failures must stay visible for
    review instead of being hidden behind repeated, expensive inference.
    """
    if not isinstance(result, dict) or result.get("agent", {}).get("stop_reason") != "error":
        return None
    queue = result.get("review_queue") or []
    text = "\n".join(
        str(entry.get(key, ""))
        for entry in queue if isinstance(entry, dict)
        for key in ("error", "reason")
    ).lower()
    return next((marker for marker in TRANSIENT_RUN_ERROR_MARKERS if marker in text), None)


def requested_review_rounds(body):
    value = int((body or {}).get("max_rounds", DEFAULT_REVIEW_ROUNDS))
    if not 1 <= value <= MAX_REVIEW_ROUNDS:
        raise ValueError(f"复查轮数须为 1～{MAX_REVIEW_ROUNDS}")
    return value


def discover_reference_labels(config, image, labels, on_retry=None):
    """Retry one structurally invalid VLM registry response, then fail closed."""
    try:
        return discover_visible_labels(config, image, labels), 0
    except ValueError as error:
        if not any(marker in str(error) for marker in REFERENCE_DISCOVERY_RETRY_ERRORS):
            raise
        if on_retry is not None:
            on_retry(error)
        return discover_visible_labels(config, image, labels), 1


def reference_visibility_review_queue(discovery, candidates):
    """Turn VLM visibility gaps into explicit, auditable human-review routes.

    A completed SAM/VLM/Agent request is not evidence that every VLM-visible
    label has a usable mask. Keep those gaps, open-set candidates and registry
    conflicts visible in the server ledger instead of relying only on a browser
    warning or a generic ``no_progress`` reason.
    """
    if not isinstance(discovery, dict):
        return []
    candidate_labels = {item.get("label") for item in candidates or []
                        if isinstance(item, dict) and item.get("label")}
    queue = []
    seen = set()

    def append(entry):
        key = (entry.get("reason_code"), entry.get("label"),
               entry.get("unknown_index"), entry.get("conflict_index"))
        if key not in seen:
            seen.add(key)
            queue.append(entry)

    detailed_review_labels = {flag.get("label") for flag in discovery.get("consistency_flags") or []
                              if isinstance(flag, dict) and flag.get("label")
                              and flag.get("route") == "human_review_required"}
    for label in discovery.get("uncertain_labels") or []:
        if label in detailed_review_labels:
            continue
        append({"label": label, "reason_code": "vlm_visibility_uncertain",
                "route": "human_review_required", "source": "visible_label_discovery",
                "candidate_available": label in candidate_labels})
    for label in discovery.get("visible_labels") or []:
        if label not in candidate_labels:
            append({"label": label, "reason_code": "vlm_visible_label_missing_candidate",
                    "route": "human_review_required", "source": "visible_label_discovery",
                    "candidate_available": False})
    for flag in discovery.get("consistency_flags") or []:
        if not isinstance(flag, dict):
            continue
        if flag.get("route") == "human_review_required" and flag.get("label"):
            append({"label": flag["label"], "reason_code": flag.get("reason_code"),
                    "route": "human_review_required", "source": "visible_label_discovery",
                    "candidate_available": flag["label"] in candidate_labels,
                    "evidence": copy.deepcopy(flag)})
        if flag.get("reason_code") != "vlm_open_set_output_truncated":
            continue
        append({"reason_code": "vlm_open_set_output_truncated",
                "route": "ontology_review_required", "source": "visible_label_discovery",
                "open_set_scan_complete": False, "evidence": copy.deepcopy(flag)})
    for index, item in enumerate(discovery.get("unknown_candidates") or []):
        proposal = copy.deepcopy(item) if isinstance(item, dict) else {"evidence": str(item)}
        append({"reason_code": "vlm_unknown_candidate",
                "route": "ontology_review_required", "source": "visible_label_discovery",
                "unknown_index": index, "proposal": proposal})
    for index, item in enumerate(discovery.get("registry_conflicts") or []):
        conflict = copy.deepcopy(item) if isinstance(item, dict) else {"evidence": str(item)}
        append({"reason_code": "vlm_registry_conflict",
                "route": "human_review_required", "source": "visible_label_discovery",
                "conflict_index": index, "conflict": conflict})
    return queue


def recover_reference_ego_candidate(config, image, discovery, result):
    """Add one guarded ego proposal only for a high-confidence discovery miss."""
    status = {"enabled": config.get("ego_bottom_point_fallback") is True,
              "triggered": False, "status": "disabled"}
    if not status["enabled"]:
        return status
    shapes = result.get("shapes") if isinstance(result, dict) else None
    if not isinstance(shapes, list):
        raise ValueError("ego 补救前候选格式无效")
    if any(isinstance(shape, dict) and shape.get("label") == "ego" for shape in shapes):
        return status | {"status": "existing_candidate"}
    assessments = discovery.get("assessments") if isinstance(discovery, dict) else None
    assessment = next((item for item in assessments or []
                       if isinstance(item, dict) and item.get("label") == "ego"), None)
    if (not isinstance(assessment, dict) or assessment.get("evidence") != "present"
            or type(assessment.get("confidence")) not in (int, float)
            or isinstance(assessment.get("confidence"), bool)
            or assessment["confidence"] < 0.90):
        return status | {"status": "discovery_not_eligible"}
    status.update(triggered=True, discovery_confidence=float(assessment["confidence"]))
    next_id = max([0] + [shape.get("id", 0) for shape in shapes
                         if isinstance(shape, dict) and type(shape.get("id")) is int]) + 1
    try:
        recovered, evidence = ego_bottom_point_candidates(config, image, next_id=next_id)
    except RunCancelled:
        raise
    except Exception as error:
        result.setdefault("review_queue", []).append({
            "label": "ego", "reason_code": "ego_point_fallback_failed",
            "route": "human_review_required", "source": "ego_bottom_point_fallback",
            "error": str(error)[:500],
        })
        return status | {"status": "failed", "error": str(error)[:500]}
    if not recovered:
        result.setdefault("review_queue", []).append({
            "label": "ego", "reason_code": "ego_point_fallback_no_guarded_candidate",
            "route": "human_review_required", "source": "ego_bottom_point_fallback",
            "evidence": evidence,
        })
        return status | {"status": "no_guarded_candidate", "evidence": evidence}
    shapes.extend(recovered)
    result["candidates_added"] = int(result.get("candidates_added") or 0) + len(recovered)
    result.setdefault("review_queue", []).append({
        "label": "ego", "reason_code": "ego_point_fallback_candidate",
        "route": "human_review_required", "source": "ego_bottom_point_fallback",
        "candidate_available": True, "candidate_id": recovered[0]["id"],
        "evidence": evidence,
    })
    return status | {"status": "candidate_generated", "candidate_count": len(recovered),
                     "candidate_id": recovered[0]["id"], "evidence": evidence}


def filter_post_recovery_slope_ego_conflicts(image, result):
    """Apply the same-frame slope/ego guard after late ego recovery.

    Reference-draft generation may add the guarded ego fallback only after the
    normal SAM generation filters have run.  Repeating this one conservative
    conflict check here closes that ordering gap without treating either
    machine mask as truth.  Suppressed slope evidence remains in both the
    candidate-filter ledger and the human-review queue.
    """
    shapes = result.get("shapes") if isinstance(result, dict) else None
    if not isinstance(shapes, list):
        raise ValueError("ego 补救后的候选格式无效")
    if not (any(item.get("label") == "slope" for item in shapes
                if isinstance(item, dict))
            and any(item.get("label") == "ego" for item in shapes
                    if isinstance(item, dict))):
        return {"status": "not_applicable", "suppressed_count": 0}
    with Image.open(io.BytesIO(image)) as decoded:
        width, height = decoded.size
    retained, suppressed = filter_slope_current_ego_conflicts(
        shapes, width, height)
    if not suppressed:
        return {"status": "no_conflict", "suppressed_count": 0}
    result["shapes"] = retained
    result["candidates_added"] = max(
        0, int(result.get("candidates_added") or 0) - len(suppressed))
    result.setdefault("agent", {}).setdefault("candidate_filters", []).extend(
        copy.deepcopy(suppressed))
    result.setdefault("review_queue", []).extend(copy.deepcopy(suppressed))
    return {
        "status": "human_review_required",
        "suppressed_count": len(suppressed),
        "reason_code": "slope_candidate_current_ego_conflict",
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
    }


@app.after_request
def disable_workbench_asset_cache(response):
    """Keep the local workbench UI in sync with the deployed source.

    LabelAgent is a local operator tool that is updated in place.  A browser
    keeping an older index/app bundle can hide newly added controls while the
    backend is already on a newer version, which looks like a model hang.  API
    and image responses keep their normal caching behavior; only UI documents
    and assets are made non-cacheable.
    """
    path = request.path.lower()
    if path == "/" or path.endswith((".html", ".js", ".css")):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


class StaleHumanReviewError(Exception):
    """The frame or candidate changed after the reviewer opened the gate."""


def _commit_work_file(temporary, destination):
    for attempt in range(10):
        try:
            temporary.replace(destination)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(min(.05 * 2**attempt, 1.0))


def state_path():
    return WORK / "state.json"


def file_sessions_path():
    return WORK / "file_sessions.json"


def _read_json(path):
    # json.loads(path.read_text()) briefly holds both the entire UTF-8 text and
    # the decoded object. Large multi-file workspaces can exceed available RAM.
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


FAST_JSON_WRITE_LIMIT = 64 * 1024 * 1024


def _write_json(path, value, *, pretty=False, sort_keys=False, fast=False):
    # Stream JSON directly to disk. State is deliberately compact because dense
    # polygon points make pretty-printed workspaces several times larger.  For
    # bounded workspaces, building one compact string avoids millions of tiny
    # TextIO writes; large/unknown workspaces retain the low-memory stream path.
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        options = dict(ensure_ascii=False, sort_keys=sort_keys,
                       indent=2 if pretty else None,
                       separators=None if pretty else (",", ":"))
        if fast:
            handle.write(json.dumps(value, **options))
        else:
            json.dump(value, handle, **options)


def _fast_json_write_allowed(destination):
    try:
        return destination.is_file() and destination.stat().st_size <= FAST_JSON_WRITE_LIMIT
    except OSError:
        return False


def save():
    global file_sessions_dirty, batch_review_index_cache
    timings = {}
    sessions = state.get("file_sessions", {})
    sessions_path = file_sessions_path()
    if file_sessions_dirty or (sessions and not sessions_path.exists()):
        sessions_temporary = sessions_path.with_suffix(".tmp")
        started = time.perf_counter()
        _write_json(sessions_temporary, sessions,
                    fast=_fast_json_write_allowed(sessions_path))
        timings["sessions_write"] = round((time.perf_counter() - started) * 1000, 3)
        started = time.perf_counter()
        _commit_work_file(sessions_temporary, sessions_path)
        timings["sessions_commit"] = round((time.perf_counter() - started) * 1000, 3)
        file_sessions_dirty = False
    temporary = state_path().with_suffix(".tmp")
    started = time.perf_counter()
    _write_json(temporary, {k: v for k, v in state.items()
                            if k not in {"video", "file_sessions"}},
                fast=_fast_json_write_allowed(state_path()))
    timings["state_write"] = round((time.perf_counter() - started) * 1000, 3)
    started = time.perf_counter()
    _commit_work_file(temporary, state_path())
    timings["state_commit"] = round((time.perf_counter() - started) * 1000, 3)
    batch_review_index_cache = None
    return timings


def _canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _sha256_json(value):
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _valid_sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _candidate_fingerprint_or_none(candidate):
    try:
        return candidate_fingerprint(candidate)
    except (ValueError, TypeError, KeyError, AttributeError):
        return None


def candidate_review_binding(candidates):
    if not isinstance(candidates, list) or not candidates:
        return None
    return {
        "candidate_artifact_sha256": _sha256_json(candidates),
        "candidates": [
            {
                "candidate_id": candidate.get("id"),
                "candidate_sha256": _sha256_json(candidate),
                "candidate_fingerprint": _candidate_fingerprint_or_none(candidate),
                "candidate_fingerprint_sha256": _candidate_fingerprint_or_none(candidate),
            }
            for candidate in candidates
        ],
    }


def candidate_review_bindings():
    """Return server-canonical hashes so browsers never reinterpret JSON numbers."""
    return {str(frame): binding for frame, candidates in state.get("candidates", {}).items()
            if (binding := candidate_review_binding(candidates)) is not None}


def auto_result_clear_binding(index):
    """Bind a destructive candidate clear to the exact visible machine result."""
    key = str(index)
    candidates = state.get("candidates", {})
    agent_runs = state.get("agent_runs", {})
    return {
        "candidate_artifact_sha256": (
            _sha256_json(candidates[key]) if key in candidates else None),
        "agent_run_sha256": (
            _sha256_json(agent_runs[key]) if key in agent_runs else None),
    }


TRUTH_ACCEPT_DECISIONS = {
    "manual_annotation", "accept_candidate", "correct_candidate", "confirmed_empty",
}


def _fetch_readiness_json(url, timeout=3):
    """Read a small JSON health response without exposing credentials."""
    request_value = urllib.request.Request(url, method="GET",
        headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request_value, timeout=timeout) as response:
            if response.status != 200:
                raise ValueError(f"HTTP {response.status}")
            payload = response.read(1024 * 1024 + 1)
            if len(payload) > 1024 * 1024:
                raise ValueError("响应超过 1 MiB")
            result = json.loads(payload)
            if not isinstance(result, dict):
                raise ValueError("响应不是 JSON 对象")
            return result
    except (OSError, ValueError, json.JSONDecodeError,
            urllib.error.HTTPError, urllib.error.URLError) as error:
        raise RuntimeError(str(error)[:240]) from error


def _normalize_dataset_split(value):
    normalized = str(value or "").strip().lower().replace("_", "").replace("-", "")
    if normalized in {"train", "train72", "training"}:
        return "train"
    if normalized in {"test", "test24", "evaluation", "eval"}:
        return "test"
    if normalized in {"val", "validation"}:
        return "val"
    if normalized in {"", "none", "unassigned", "unknown"}:
        return "unassigned"
    return "invalid"


def _frame_dataset_split(frame):
    values = []
    config_split = state.setdefault("config", {}).get("dataset_split")
    if config_split is not None:
        values.append(config_split)
    run = state.setdefault("agent_runs", {}).get(str(frame)) or {}
    for key in ("dataset_split", "split"):
        if run.get(key) is not None:
            values.append(run[key])
    for shape in state.setdefault("candidates", {}).get(str(frame), []) or []:
        for key in ("dataset_split", "split"):
            if shape.get(key) is not None:
                values.append(shape[key])
    normalized = {_normalize_dataset_split(value) for value in values}
    normalized.discard("unassigned")
    if "invalid" in normalized or len(normalized) > 1:
        return "mixed"
    return next(iter(normalized), "unassigned")


def _review_scope(frame_sha256, frame):
    config = state.setdefault("config", {})
    file_key = hashlib.sha256(str(state["video"].path).encode("utf-8")).hexdigest()[:16]
    dataset_split = _frame_dataset_split(frame)
    split_policy = ({"train": "immutable_truth_promotion_required",
                     "test": "evaluation_only",
                     "val": "evaluation_only",
                     "mixed": "blocked_mixed_split"}
                    .get(dataset_split, "blocked_unassigned"))
    return {
        "dataset_id": config.get("dataset_id") or f"local-workbench-{file_key}",
        "dataset_version": config.get("dataset_version") or f"frame-content-{frame_sha256[:16]}",
        "camera": config.get("camera") or state["topic"],
        "dataset_split": dataset_split,
        "split_policy": split_policy,
        "training_return_allowed": False,
    }


def _assert_review_binding(body, *, frame, frame_sha256, candidate_sha256):
    """Validate the UI snapshot against values recomputed while holding lock."""
    assert_request_file_activation(body, "expected_file_activation_generation")
    if body.get("topic") != state.get("topic"):
        raise StaleHumanReviewError("图像源已变化，请重新打开人工复核门禁")
    expected_frame = body.get("expected_frame_sha256")
    if not _valid_sha256(expected_frame):
        raise ValueError("人工复核必须包含有效的 expected_frame_sha256")
    if "expected_candidate_sha256" not in body:
        raise ValueError("人工复核必须包含 expected_candidate_sha256（无候选时为 null）")
    expected_candidate = body.get("expected_candidate_sha256")
    if expected_candidate is not None and not _valid_sha256(expected_candidate):
        raise ValueError("expected_candidate_sha256 必须是 64 位 SHA-256 或 null")
    scope = _review_scope(frame_sha256, frame)
    expected_scope = {
        "dataset_id": body.get("expected_dataset_id"),
        "dataset_version": body.get("expected_dataset_version"),
        "camera": body.get("expected_camera"),
        "dataset_split": body.get("expected_dataset_split"),
        "split_policy": body.get("expected_split_policy"),
        "training_return_allowed": body.get("expected_training_return_allowed"),
    }
    for key in ("dataset_id", "dataset_version", "camera", "dataset_split", "split_policy"):
        if not isinstance(expected_scope[key], str) or not expected_scope[key].strip():
            raise ValueError("人工复核必须绑定数据集、版本、相机、split 和回流策略")
    if expected_scope["training_return_allowed"] is not False:
        raise ValueError("工作台人工复核不能直接授权训练回流")
    if scope["dataset_split"] == "mixed":
        raise ValueError("检测到 train/test split 混入，已阻止人工真值提交")
    if (expected_frame != frame_sha256
            or expected_candidate != candidate_sha256 or expected_scope != scope):
        raise StaleHumanReviewError("数据集、版本、相机、帧或候选已在复核期间变化，请重新复核后提交")
    return expected_frame, expected_candidate, scope


def _resolve_review_candidate(body, frame):
    """Bind a browser-side SAM draft without promoting or persisting it."""
    stored = state.setdefault("candidates", {}).get(str(frame))
    raw = body.get("draft_candidate")
    source = body.get("draft_source")
    if raw is None:
        if source is not None:
            raise ValueError("draft_source 只能与 draft_candidate 同时提供")
        return stored, ("server_state" if stored is not None else "none"), None
    if source != "sam_point_prompt":
        raise ValueError("浏览器候选只允许声明为 sam_point_prompt")
    prepared = prepare_candidate_import(state["video"], {
        state["topic"]: {str(frame): raw}}, state.get("labels", []))
    candidate = prepared[state["topic"]][str(frame)]
    if not candidate:
        raise ValueError("SAM 点提示候选不能为空")
    for original, normalized in zip(raw, candidate):
        if "id" not in original:
            normalized.pop("id", None)
    parent_sha = _sha256_json(stored) if stored is not None else None
    return candidate, "browser_sam_point_draft", parent_sha


def _contains_slope(shapes):
    return bool(shapes) and any(isinstance(item, dict)
        and item.get("label") in {"slope", "slope_visible"} for item in shapes)


def _contains_unknown_label(shapes):
    blocked = {"new_label_candidate"}
    return bool(shapes) and any(isinstance(item, dict) and (
        str(item.get("label", "")).strip().lower().startswith("unknown")
        or str(item.get("label", "")).strip().lower() in blocked) for item in shapes)


def _slope_review_context(*, decision, candidate, truth, candidate_sha,
                          candidate_event_id, frame_sha256):
    candidate_has_slope = _contains_slope(candidate)
    truth_has_slope = _contains_slope(truth)
    if not candidate_has_slope and not truth_has_slope:
        return None
    human_truth = truth is not None and truth_has_slope and decision != "reject_candidate"
    if decision == "reject_candidate":
        visible_status = "human_rejected"
    elif human_truth:
        visible_status = "human_confirmed"
    else:
        visible_status = "candidate_supported"
    provenance_sha = candidate_sha or frame_sha256
    return {"label_type": "slope", "semantic_label": "slope_visible",
        "visible_status": visible_status,
        "human_pixel_truth_available": human_truth, "approved_for_training": False,
        "business_mapping": "traversable_area", "traversability_status": "passable",
        "downstream_mapping": {"operation": "union",
            "source_labels": ["floor", "slope"], "target_label": "traversable_area"},
        "policy_default": "passable",
        "policy_basis": "user_defined_project_semantics",
        "policy_override": {"status": "none", "rule_id": None, "reason": None,
            "decided_by": None, "evidence_artifact_ids": []},
        "candidate_sha256": candidate_sha,
        "evidence_provenance": [{"kind": "machine_candidate" if candidate_sha else "input_frame",
            "source": candidate_event_id or "human_manual_annotation",
            "sha256": provenance_sha}], "hard_set_strata": ["unassigned"]}


def _parse_review(body):
    review = body.get("review") if isinstance(body, dict) else None
    if not isinstance(review, dict):
        raise ValueError("保存或驳回前必须填写人工复核信息")
    unknown = set(review) - {"reviewer_declared_id", "started_at", "note",
                             "decision_idempotency_key", "remediation_of_review_id"}
    if unknown:
        raise ValueError(f"人工复核信息包含未知字段：{sorted(unknown)}")
    reviewer_declared_id = review.get("reviewer_declared_id")
    if (not isinstance(reviewer_declared_id, str) or not reviewer_declared_id.strip()
            or len(reviewer_declared_id.strip()) > 100):
        raise ValueError("复核人自报 ID 不能为空且不能超过 100 个字符")
    decision_key = review.get("decision_idempotency_key")
    if (not isinstance(decision_key, str) or not decision_key.strip()
            or len(decision_key.strip()) > 200):
        raise ValueError("人工决定必须包含不超过 200 个字符的幂等键")
    started_at = review.get("started_at")
    if not isinstance(started_at, str):
        raise ValueError("人工复核必须包含带时区的开始时间")
    try:
        started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("人工复核开始时间必须是 ISO-8601 格式") from error
    if started.tzinfo is None:
        raise ValueError("人工复核开始时间必须包含时区")
    note = review.get("note", "")
    if not isinstance(note, str) or len(note) > 1000:
        raise ValueError("复核备注不能超过 1000 个字符")
    remediation = review.get("remediation_of_review_id")
    if remediation is not None and not _valid_sha256(remediation):
        raise ValueError("remediation_of_review_id 必须是 64 位复核记录哈希")
    return {"reviewer_declared_id": reviewer_declared_id.strip(),
            "identity_assurance": "self_declared",
            "started_at": started.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "note": note.strip(), "decision_idempotency_key": decision_key.strip(),
            "remediation_of_review_id": remediation}


def human_review_records(*, video_file=None, topic=None, frame=None):
    folder = WORK / "human_reviews"
    records = []
    previous = None
    paths = sorted(folder.glob("*.json")) if folder.exists() else []
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        review_id = record.pop("review_id", None)
        if record.get("previous_review_id") != previous or _sha256_json(record) != review_id:
            raise RuntimeError(f"人工复核审计链损坏：{path.name}")
        record["review_id"] = review_id
        previous = review_id
        if video_file is not None and record.get("file") != video_file:
            continue
        if topic is not None and record.get("topic") != topic:
            continue
        if frame is not None and record.get("frame") != frame:
            continue
        records.append(record)
    return records


def _legacy_shape_source_counts(shapes):
    counts = {"manual": 0, "auto": 0, "unknown": 0}
    for shape in shapes or []:
        metadata = shape.get("cvat_metadata") if isinstance(shape, dict) else None
        source = metadata.get("source") if isinstance(metadata, dict) else None
        if source in ("manual", "auto"):
            counts[source] += 1
        else:
            counts["unknown"] += 1
    return counts


def truth_review_status_for(selected_topic, truth_by_frame, records=None):
    """Derive whether current truth is bound to the current human-review contract.

    Legacy imports remain useful as editable drafts, but only a committed v2 review
    whose exact truth hash still matches may be reused directly as a reference.
    """
    if state.get("video") is None or not selected_topic:
        return {}
    video_file = str(getattr(state["video"], "path", state.get("video_path", "")))
    transactions = state.get("human_review_transactions", {})
    reviews_by_frame = {}
    source_records = (records if records is not None
                      else human_review_records(video_file=video_file, topic=selected_topic))
    for record in source_records:
        if record.get("file") != video_file or record.get("topic") != selected_topic:
            continue
        if record.get("decision") in TRUTH_ACCEPT_DECISIONS:
            reviews_by_frame.setdefault(str(record.get("frame")), []).append(record)
    result = {}
    for frame_key, truth in (truth_by_frame or {}).items():
        truth_hash = _sha256_json(truth)
        records = reviews_by_frame.get(str(frame_key), [])
        record = records[-1] if records else None
        transaction = transactions.get(record.get("review_id"), {}) if record else {}
        checks = {
            "nonempty_truth": isinstance(truth, list) and len(truth) > 0,
            "current_contract": bool(record and record.get("schema_version") == "labelagent.human-review.v2"),
            "truth_sha256_matches": bool(record and record.get("truth_sha256") == truth_hash),
            "transaction_committed": bool(record and transaction.get("status") == "committed"),
        }
        verified = all(checks.values())
        if not checks["nonempty_truth"]:
            reason = "empty_truth_not_reference_eligible"
        elif record is None:
            reason = "missing_current_human_review_receipt"
        elif not checks["current_contract"]:
            reason = "legacy_human_review_contract"
        elif not checks["truth_sha256_matches"]:
            reason = "truth_changed_after_human_review"
        elif not checks["transaction_committed"]:
            reason = "human_review_transaction_not_committed"
        else:
            reason = "verified_current_human_truth"
        result[str(frame_key)] = {
            "reusable_as_reference": verified,
            "current_human_review_verified": verified,
            "reason_code": reason,
            "truth_sha256": truth_hash,
            "shape_count": len(truth) if isinstance(truth, list) else 0,
            "legacy_source_counts": _legacy_shape_source_counts(truth),
            "review_id": record.get("review_id") if record else None,
            "decision": record.get("decision") if record else None,
            "reviewed_at": record.get("reviewed_at") if record else None,
            "reviewer_declared_id": record.get("reviewer_declared_id") if record else None,
            "identity_assurance": record.get("identity_assurance") if record else None,
            "checks": checks,
        }
    return result


def current_truth_review_status():
    return truth_review_status_for(state.get("topic"), state.get("truth", {}))


def _review_status_key(frame):
    return _sha256_json({"file": str(state["video"].path), "topic": state["topic"],
                         "frame": int(frame)})


def _persistent_state_snapshot():
    return copy.deepcopy({key: value for key, value in state.items() if key != "video"})


def _restore_persistent_state(snapshot):
    video = state.get("video")
    state.clear()
    state.update(snapshot)
    state["video"] = video


def _review_by_idempotency_key(decision_key):
    matches = [item for item in human_review_records()
               if item.get("decision_idempotency_key") == decision_key]
    if len(matches) > 1:
        raise RuntimeError(f"人工决定幂等键重复：{decision_key}")
    return matches[0] if matches else None


def _assert_idempotent_review(existing, *, review, frame, frame_sha256,
                              allowed_decisions, review_scope, truth=None, candidate=None,
                              candidate_origin="none", parent_candidate_sha256=None):
    expected = {"file": str(state["video"].path), "topic": state["topic"],
                "frame": frame, "frame_sha256": frame_sha256,
                "reviewer_declared_id": review["reviewer_declared_id"],
                "identity_assurance": "self_declared",
                "started_at": review["started_at"], "note": review["note"],
                "remediation_of_review_id": review.get("remediation_of_review_id"),
                "dataset_id": review_scope["dataset_id"],
                "dataset_version": review_scope["dataset_version"],
                "camera": review_scope["camera"],
                "dataset_split": review_scope["dataset_split"],
                "split_policy": review_scope["split_policy"],
                "training_return_allowed": review_scope["training_return_allowed"],
                "candidate_origin": candidate_origin,
                "parent_candidate_sha256": parent_candidate_sha256}
    if existing.get("decision") not in allowed_decisions:
        raise ValueError("该幂等键已用于另一种人工决定")
    if any(existing.get(key) != value for key, value in expected.items()):
        raise ValueError("该幂等键对应的人工决定内容与本次请求不一致")
    if truth is not None and existing.get("truth_sha256") != _sha256_json(truth):
        raise ValueError("该幂等键对应的真值与本次请求不一致")
    if candidate is not None and existing.get("candidate_sha256") != _sha256_json(candidate):
        raise ValueError("该幂等键对应的候选与本次请求不一致")


def prepare_human_review(*, frame, decision, review, frame_sha256,
                         expected_frame_sha256, expected_candidate_sha256,
                         review_scope, candidate, truth=None, confirmed_empty=False,
                         candidate_origin="none", parent_candidate_sha256=None):
    history = human_review_records()
    finished = datetime.now(timezone.utc)
    started = datetime.fromisoformat(review["started_at"].replace("Z", "+00:00"))
    candidate_sha = _sha256_json(candidate) if candidate is not None else None
    candidate_event_id = _sha256_json({"event": "workbench_candidate",
        "file": str(state["video"].path), "topic": state["topic"], "frame": frame,
        "candidate_sha256": candidate_sha, "candidate_origin": candidate_origin,
        "parent_candidate_sha256": parent_candidate_sha256}) if candidate is not None else None
    slope_context = _slope_review_context(decision=decision, candidate=candidate,
        truth=truth, candidate_sha=candidate_sha, candidate_event_id=candidate_event_id,
        frame_sha256=frame_sha256)
    body = {"schema_version": "labelagent.human-review.v2",
            "sequence": len(history) + 1,
            "previous_review_id": history[-1]["review_id"] if history else None,
            "file": str(state["video"].path), "topic": state["topic"],
            "dataset_id": review_scope["dataset_id"],
            "dataset_version": review_scope["dataset_version"],
            "camera": review_scope["camera"],
            "dataset_split": review_scope["dataset_split"],
            "split_policy": review_scope["split_policy"],
            "training_return_allowed": review_scope["training_return_allowed"],
            "truth_role": ("evaluation_truth" if review_scope["dataset_split"] in {"test", "val"}
                           else "workbench_human_truth_candidate"),
            "frame": frame, "frame_sha256": frame_sha256,
            "expected_frame_sha256": expected_frame_sha256,
            "expected_candidate_sha256": expected_candidate_sha256,
            "decision": decision, **review,
            "reviewed_at": finished.isoformat().replace("+00:00", "Z"),
            "elapsed_seconds": round(max(0.0, (finished - started).total_seconds()), 3),
            "candidate_sha256": candidate_sha,
            "candidate_origin": candidate_origin,
            "parent_candidate_sha256": parent_candidate_sha256,
            "candidate_event_id": candidate_event_id,
            "candidate_shape_count": len(candidate) if candidate is not None else 0,
            "truth_sha256": _sha256_json(truth) if truth is not None else None,
            "truth_shape_count": len(truth) if truth is not None else None,
            "confirmed_empty": bool(confirmed_empty),
            "slope_context": slope_context,
            "closed_loop_linkage": {"status": "pending_sync" if truth is not None else "not_applicable",
                "candidate_event_id": candidate_event_id,
                "human_accepted_event_id": None, "truth_committed_event_id": None}}
    review_id = _sha256_json(body)
    return {**body, "review_id": review_id}


def _review_commit_readback(record):
    persisted = _read_json(state_path())
    audit_present = any(item.get("review_id") == record["review_id"]
                        for item in human_review_records())
    transaction = persisted.get("human_review_transactions", {}).get(record["review_id"], {})
    truth = persisted.get("truth", {}).get(str(record["frame"]))
    truth_matches = (record["truth_sha256"] is None
                     or (truth is not None and _sha256_json(truth) == record["truth_sha256"]))
    candidate_inactive = str(record["frame"]) not in persisted.get("candidates", {})
    rejected_archived = (record["decision"] != "reject_candidate"
        or record["review_id"] in persisted.get("rejected_candidates", {}))
    verified = (audit_present and transaction.get("status") == "committed"
                and truth_matches and candidate_inactive and rejected_archived)
    return {"verified": verified, "audit_present": audit_present,
            "transaction_committed": transaction.get("status") == "committed",
            "truth_sha256_matches": truth_matches,
            "candidate_inactive": candidate_inactive,
            "rejected_candidate_archived": rejected_archived,
            "state_file": str(state_path())}


def _write_pending_review(record):
    folder = WORK / "human_review_pending"
    folder.mkdir(exist_ok=True)
    destination = folder / f"{record['review_id']}.json"
    temporary = folder / (uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2,
                                    sort_keys=True), encoding="utf-8")
    _commit_work_file(temporary, destination)
    return destination


def _commit_review_record(record, pending_path=None):
    folder = WORK / "human_reviews"
    folder.mkdir(exist_ok=True)
    destination = folder / f"{record['sequence']:06d}-{record['review_id']}.json"
    if destination.exists():
        if json.loads(destination.read_text(encoding="utf-8")) != record:
            raise RuntimeError(f"人工复核提交文件冲突：{destination.name}")
        if pending_path is not None:
            pending_path.unlink(missing_ok=True)
        return destination
    history = human_review_records()
    expected_previous = history[-1]["review_id"] if history else None
    if (record["sequence"] != len(history) + 1
            or record["previous_review_id"] != expected_previous):
        raise RuntimeError("人工复核待提交记录与当前审计链不连续")
    if pending_path is None:
        pending_path = _write_pending_review(record)
    _commit_work_file(pending_path, destination)
    return destination


def commit_human_review(record, mutation):
    """Commit state first, then the immutable audit, with a recoverable link."""
    snapshot = _persistent_state_snapshot()
    pending_path = _write_pending_review(record)
    transaction = {"review_id": record["review_id"], "status": "state_saved_pending_audit",
        "decision": record["decision"], "file": record["file"], "topic": record["topic"],
        "frame": record["frame"], "candidate_sha256": record["candidate_sha256"],
        "candidate_event_id": record["candidate_event_id"],
        "truth_sha256": record["truth_sha256"]}
    try:
        mutation()
        state.setdefault("human_review_transactions", {})[record["review_id"]] = transaction
        save()
    except Exception:
        _restore_persistent_state(snapshot)
        pending_path.unlink(missing_ok=True)
        state_path().with_suffix(".tmp").unlink(missing_ok=True)
        raise
    destination = _commit_review_record(record, pending_path)
    transaction["status"] = "committed"
    transaction["audit_file"] = destination.name
    try:
        save()
    except Exception:
        transaction["status"] = "state_saved_pending_audit"
        transaction.pop("audit_file", None)
        raise
    return record


def recover_pending_human_reviews():
    changed = False
    transactions = state.setdefault("human_review_transactions", {})
    pending_folder = WORK / "human_review_pending"
    for review_id, transaction in list(transactions.items()):
        if transaction.get("status") == "committed":
            continue
        pending_path = pending_folder / f"{review_id}.json"
        final_matches = list((WORK / "human_reviews").glob(f"*-{review_id}.json"))
        if final_matches:
            destination = final_matches[0]
        elif pending_path.exists():
            record = json.loads(pending_path.read_text(encoding="utf-8"))
            destination = _commit_review_record(record, pending_path)
        else:
            raise RuntimeError(f"状态包含未留证的人工作业：{review_id}")
        transaction["status"] = "committed"
        transaction["audit_file"] = destination.name
        changed = True
    if changed:
        save()


def update_run_status(**changes):
    with run_status_lock:
        _update_run_status_locked(**changes)


RUN_SUMMARY_SCHEMA = "labelagent.run-summary.v1"


def run_summary(output):
    frames = []
    for item in output.get("frames", []):
        if not isinstance(item, dict):
            continue
        frames.append({key: copy.deepcopy(item.get(key)) for key in (
            "frame", "candidates_added", "remaining", "failed",
            "existing_candidates_preserved") if key in item})
    current = output.get("current")
    if isinstance(current, dict):
        current = {key: copy.deepcopy(current.get(key)) for key in (
            "frame", "phase", "round", "retry_attempt") if key in current}
    else:
        current = None
    failed = output.get("failed_frames")
    if not isinstance(failed, list):
        failed = [item.get("frame") for item in frames if item.get("failed")]
    targets = output.get("targets")
    if not isinstance(targets, list):
        targets = []
    targets = list(dict.fromkeys(
        item for item in targets
        if isinstance(item, int) and not isinstance(item, bool)
    ))[:100]
    completed_frames = {
        item.get("frame") for item in frames
        if isinstance(item.get("frame"), int) and not isinstance(item.get("frame"), bool)
    }
    pending_frames = [item for item in targets if item not in completed_frames]
    task_id = output.get("task_id")
    if not task_id:
        task_id = "legacy-" + _sha256_json({
            "file": output.get("file"), "topic": output.get("topic"),
            "reference": output.get("reference"), "frames": frames,
        })[:24]
    inferred_status = "running" if current else "completed" if frames else "idle"
    return {
        "schema_version": RUN_SUMMARY_SCHEMA,
        "task_id": task_id,
        "file": output.get("file"),
        "topic": output.get("topic"),
        "reference": output.get("reference"),
        "status": output.get("status", inferred_status),
        "total": int(output.get("total") or len(output.get("targets", [])) or len(frames)),
        "completed": len(frames),
        "current": current,
        "frames": frames,
        "pending_frames": pending_frames,
        "failed_frames": [int(item) for item in failed if isinstance(item, int)],
        "message": str(output.get("message") or "")[:500],
        "error": str(output.get("error") or "")[:500],
        "updated_at": time.time(),
    }


def load_run_summary():
    path = WORK / "last_run_summary.json"
    with run_summary_lock:
        try:
            if path.is_file() and path.stat().st_size <= 1024 * 1024:
                value = json.loads(path.read_text(encoding="utf-8"))
                if value.get("schema_version") == RUN_SUMMARY_SCHEMA:
                    return value
            legacy = WORK / "last_run.json"
            if not legacy.is_file() or legacy.stat().st_size > FAST_JSON_WRITE_LIMIT:
                return None
            value = run_summary(json.loads(legacy.read_text(encoding="utf-8")))
            temporary = path.with_suffix(f".{threading.get_ident()}.tmp")
            _write_json(temporary, value, fast=True)
            _commit_work_file(temporary, path)
            return value
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None


def save_run(output):
    path = WORK / "last_run.json"
    temporary = path.with_suffix(".tmp")
    _write_json(temporary, output, fast=_fast_json_write_allowed(path))
    _commit_work_file(temporary, path)
    summary_path = WORK / "last_run_summary.json"
    summary_temporary = summary_path.with_suffix(".tmp")
    _write_json(summary_temporary, run_summary(output), fast=True)
    _commit_work_file(summary_temporary, summary_path)


def save_debug(trace):
    path = WORK / "model_debug.json"
    temporary = path.with_suffix(".tmp")
    _write_json(temporary, {"trace": trace}, fast=_fast_json_write_allowed(path))
    _commit_work_file(temporary, path)


def inspection_image_names(value):
    names = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ("url", "image") and isinstance(item, str):
                name = item.removeprefix("/api/inspection/image/")
                if (item.startswith("/api/inspection/image/") and len(name) == 68
                        and name.endswith(".jpg") and all(c in "0123456789abcdef" for c in name[:-4])):
                    names.add(name)
            else:
                names.update(inspection_image_names(item))
    elif isinstance(value, list):
        for item in value:
            names.update(inspection_image_names(item))
    return names


def save_inspection_image(data):
    """Store immutable images separately so debug JSON stays small and readable."""
    folder = WORK / "inspection"
    folder.mkdir(exist_ok=True)
    name = hashlib.sha256(data).hexdigest() + ".jpg"
    path = folder / name
    if not path.exists():
        temporary = folder / (uuid.uuid4().hex + ".tmp")
        temporary.write_bytes(data)
        _commit_work_file(temporary, path)
    return "/api/inspection/image/" + name


@app.get("/api/inspection/image/<name>")
def inspection_image(name):
    if len(name) != 68 or not name.endswith(".jpg") or any(c not in "0123456789abcdef" for c in name[:-4]):
        raise ValueError("无效的过程图像名称")
    return send_file(WORK / "inspection" / name, mimetype="image/jpeg", max_age=31536000)


def save_description_trace(trace, description):
    path = WORK / "description_debug.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"trace": trace, "description": description}, ensure_ascii=False), encoding="utf-8")
    _commit_work_file(temporary, path)


def redact_debug(value, *, credential=None):
    key = state["config"].get("llm_key") if credential is None else credential
    if isinstance(value, str):
        return value.replace(key, "[REDACTED]") if key else value
    if isinstance(value, dict):
        return {name: redact_debug(item, credential=key or "") for name, item in value.items()}
    if isinstance(value, list):
        return [redact_debug(item, credential=key or "") for item in value]
    return value


def load():
    global file_sessions_dirty, batch_review_index_cache
    batch_review_index_cache = None
    file_sessions_dirty = False
    legacy_sessions = None
    if state_path().exists():
        stored = _read_json(state_path())
        legacy_sessions = stored.pop("file_sessions", None)
        state.update(stored)
    if file_sessions_path().exists():
        state["file_sessions"] = _read_json(file_sessions_path())
    else:
        state["file_sessions"] = legacy_sessions or {}
        file_sessions_dirty = legacy_sessions is not None
    for key in ("sam_url", "cvat_url", "cvat_token"):
        state["config"].pop(key, None)
    for key in ("job_id", "task_id", "cvat_start", "cvat_count"):
        state.pop(key, None)
    for session in state.get("file_sessions", {}).values():
        for key in ("job_id", "task_id", "cvat_start", "cvat_count"):
            if key in session:
                session.pop(key, None)
                file_sessions_dirty = True
    active_path = state.get("video_path")
    if active_path:
        active_keys = {active_path, str(Path(active_path).resolve())}
        for active_key in active_keys:
            if active_key in state["file_sessions"]:
                state["file_sessions"].pop(active_key)
                file_sessions_dirty = True
    state.setdefault("candidates", {})
    state.setdefault("candidates_by_topic", {})
    state.setdefault("agent_runs", {})
    state.setdefault("agent_runs_by_topic", {})
    state.setdefault("candidate_revisions", {})
    state.setdefault("candidate_revisions_by_topic", {})
    state.setdefault("candidate_review_status", {})
    state.setdefault("candidate_review_proposals", [])
    state.setdefault("human_review_transactions", {})
    state.setdefault("rejected_candidates", {})
    recover_pending_human_reviews()


def frame_image(index):
    return state["video"].frame(state["topic"], index)


def decoded_frame_metadata(video, topic, index):
    """Decode one frame through the production media path and verify pixels."""
    payload = video.frame(topic, index)
    with Image.open(io.BytesIO(payload)) as decoded:
        width, height = decoded.size
        decoded.verify()
    if width < 1 or height < 1:
        raise ValueError("图像尺寸为空")
    return {
        "frame": index, "width": width, "height": height,
        "encoded_bytes": len(payload), "cache_warmed": True,
    }


def selected_auto_labels_for_truth(reference_truth):
    available = state.get("labels") or list(dict.fromkeys(
        shape.get("label") for shape in reference_truth if shape.get("label")))
    reference_only = reference_labels(reference_truth, available)
    selected = state.get("auto_labels")
    if not selected:
        return reference_only
    if (not isinstance(selected, list) or len(selected) != len(set(selected))
            or any(label not in available for label in selected)):
        raise ValueError("自动标注类别与类别列表不一致，请重新选择")
    return [label for label in available if label in selected]


def reference_signature_for(index, topic, reference_truth):
    video_file = str(getattr(state["video"], "path", state.get("video_path", "")))
    reference = {"file": video_file, "topic": topic, "frame": index,
                 "description_input_version": DESCRIPTION_INPUT_VERSION,
                 "prompt_profile": state.get("config", {}).get("llm_prompt_profile", "full"),
                 "auto_labels": selected_auto_labels_for_truth(reference_truth),
                 "truth": reference_truth}
    payload = json.dumps(reference, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def reference_signature(index):
    return reference_signature_for(index, state["topic"], state["truth"][str(index)])


def description_cache_path(signature):
    """Return the content-addressed cache path for one exact reference contract."""
    if not _valid_sha256(signature):
        raise ValueError("参考帧描述签名无效")
    return WORK / "descriptions" / f"{signature}.json"


def load_reference_description(reference):
    """Load only a description bound to the active file/topic/truth contract.

    The legacy ``description.json`` remains readable for existing installations,
    while the content-addressed cache keeps descriptions for other cameras from
    being overwritten when an operator switches topics.
    """
    signature = reference_signature(reference)
    paths = (description_cache_path(signature), WORK / "description.json")
    for path in paths:
        if not path.exists():
            continue
        try:
            saved = _read_json(path)
        except (OSError, ValueError, TypeError):
            continue
        if (isinstance(saved, dict)
                and saved.get("reference_signature") == signature):
            if path == WORK / "description.json" and not paths[0].exists():
                # One-time migration for existing installations. Cache failure
                # must never make an otherwise valid legacy description unusable.
                temporary = paths[0].with_suffix(".tmp")
                try:
                    paths[0].parent.mkdir(parents=True, exist_ok=True)
                    _write_json(temporary, saved, pretty=True)
                    _commit_work_file(temporary, paths[0])
                except OSError:
                    temporary.unlink(missing_ok=True)
            return saved
    return None


def save_reference_description(description):
    """Atomically save a reusable per-reference copy plus the legacy active copy."""
    signature = description.get("reference_signature") if isinstance(description, dict) else None
    cache_path = description_cache_path(signature)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_temporary = cache_path.with_suffix(".tmp")
    _write_json(cache_temporary, description, pretty=True)
    _commit_work_file(cache_temporary, cache_path)
    legacy_path = WORK / "description.json"
    legacy_temporary = legacy_path.with_suffix(".tmp")
    _write_json(legacy_temporary, description, pretty=True)
    _commit_work_file(legacy_temporary, legacy_path)


def topic_truth(topic):
    if topic == state.get("topic"):
        return state.get("truth", {})
    return state.get("truths_by_topic", {}).get(topic, {})


def topic_description_status(topic, reference, truth_by_frame):
    reference_truth = truth_by_frame.get(str(reference))
    if not isinstance(reference_truth, list) or not reference_truth:
        return "description_missing"
    try:
        signature = reference_signature_for(reference, topic, reference_truth)
        if topic == state.get("topic"):
            if load_reference_description(reference) is not None:
                return "ready"
            stale_exists = (description_cache_path(signature).exists()
                            or (WORK / "description.json").exists())
            return "description_stale" if stale_exists else "description_missing"
        path = description_cache_path(signature)
        if not path.exists():
            return "description_missing"
        saved = _read_json(path)
        if (isinstance(saved, dict)
                and saved.get("reference_signature") == signature):
            return "ready"
        return "description_stale"
    except (OSError, ValueError, TypeError, KeyError):
        return "description_missing"


def topic_readiness_summary():
    if state.get("video") is None:
        return {}
    video_file = str(getattr(state["video"], "path", state.get("video_path", "")))
    records = human_review_records(video_file=video_file)
    output = {}
    for topic, metadata in state["video"].topics.items():
        catalog = getattr(state["video"], "catalog", {}) or {}
        kind = (catalog.get(topic) or metadata or {}).get("type")
        if kind not in (None, "image"):
            continue
        references = state.get("reference_frames", {}).get(topic, [])
        reference = references[0] if references else None
        truth_by_frame = topic_truth(topic)
        truth_present = (reference is not None
                         and bool(truth_by_frame.get(str(reference))))
        review_status = truth_review_status_for(topic, truth_by_frame, records)
        human_verified = bool(reference is not None and
            review_status.get(str(reference), {}).get("reusable_as_reference") is True)
        description_code = (topic_description_status(topic, reference, truth_by_frame)
                            if reference is not None and truth_present else None)
        description_ready = description_code == "ready"
        if reference is None:
            code = "reference_missing"
        elif not truth_present:
            code = "reference_truth_missing"
        elif not description_ready:
            code = description_code or "description_missing"
        else:
            code = "ready"
        output[topic] = {
            "count": int((metadata or {}).get("count", 0)),
            "reference_frame": reference,
            "truth_present": truth_present,
            "human_truth_verified": human_verified,
            "description_ready": description_ready,
            "ready": code == "ready",
            "code": code,
        }
    return output


def overlay(index, labels=None):
    shapes = state["truth"].get(str(index), [])
    if labels is not None:
        shapes = [shape for shape in shapes if shape.get("label") in labels]
    return reference_overlay(frame_image(index), shapes)


def selected_auto_labels(reference):
    """Return an explicit non-empty subset of configured inference labels."""
    return selected_auto_labels_for_truth(state["truth"][str(reference)])


def current_inference_label_state():
    """Return the exact label lists shown for the active workspace."""
    labels = copy.deepcopy(state.get("labels", []))
    topic = state.get("topic")
    references = state.get("reference_frames", {}).get(topic, []) if topic else []
    reference = references[0] if references else None
    auto_labels = (selected_auto_labels(reference)
                   if reference is not None else list(labels))
    reference_only_labels = (
        reference_labels(state.get("truth", {}).get(str(reference), []), labels)
        if reference is not None else [])
    return {
        "labels": labels,
        "auto_labels": auto_labels,
        "reference_only_labels": reference_only_labels,
    }


def label_state_sha256(label_state=None):
    """Hash the mutable project and inference label selections together."""
    current = label_state or current_inference_label_state()
    return _sha256_json({
        "labels": current["labels"],
        "auto_labels": current["auto_labels"],
    })


CONFIG_SECRET_KEYS = {"llm_key", "llm_fast_key"}


def public_model_config(config=None):
    """Return browser-safe model settings without either credential field."""
    current = state.get("config", {}) if config is None else config
    return {
        key: copy.deepcopy(value) for key, value in current.items()
        if key not in CONFIG_SECRET_KEYS
    }


def config_state_sha256(config=None):
    """Fingerprint the complete persisted model configuration, including keys.

    The digest lets writers detect a stale base without exposing the secret
    values that also participate in the mutable configuration state.
    """
    current = state.get("config", {}) if config is None else config
    return _sha256_json(current)


def config_write_binding():
    """Bind a configuration write to the exact workspace and prior config."""
    video = state.get("video")
    return {
        "path": str(video.path) if video is not None else "",
        "topic": state.get("topic"),
        "file_activation_generation": current_file_activation_generation(),
        "config_state_sha256": config_state_sha256(),
    }


def assert_config_write_binding(payload):
    """Reject configuration updates prepared against an older browser state."""
    generation, path, topic = assert_request_workspace_scope(
        payload,
        generation_key="expected_file_activation_generation",
        path_key="expected_path",
        topic_key="expected_topic")
    expected = payload.get("expected_config_state_sha256")
    if not _valid_sha256(expected):
        raise ValueError("expected_config_state_sha256 必须是当前配置 SHA-256")
    current = config_state_sha256()
    if expected != current:
        raise StaleHumanReviewError(
            "模型配置已被其他窗口或启动工具更新，旧配置保存请求已拒绝；请刷新后重试")
    return generation, path, topic, current


def _all_topic_state(current_key, by_topic_key):
    """Canonicalize one current-topic map with its parked topic maps."""
    values = copy.deepcopy(state.get(by_topic_key, {}))
    topic = state.get("topic")
    if topic:
        values[topic] = copy.deepcopy(state.get(current_key, {}))
    return values


def candidate_import_binding():
    """Bind an external candidate write to the exact mutable base state."""
    video = state.get("video")
    return {
        "path": str(video.path) if video is not None else "",
        "topic": state.get("topic"),
        "file_activation_generation": current_file_activation_generation(),
        "label_state_sha256": label_state_sha256(),
        "truth_state_sha256": _sha256_json(
            _all_topic_state("truth", "truths_by_topic")),
        "candidate_state_sha256": _sha256_json(
            _all_topic_state("candidates", "candidates_by_topic")),
    }


def assert_candidate_import_binding(payload):
    """Reject external candidate output produced for an older workspace."""
    generation, path, topic = assert_request_data_scope(
        payload, generation_key="expected_file_activation_generation")
    current = candidate_import_binding()
    checks = (
        ("expected_label_state_sha256", "label_state_sha256",
         "类别配置已变化，请重新生成或重新读取候选导入绑定"),
        ("expected_truth_state_sha256", "truth_state_sha256",
         "人工真值已变化，请重新检查候选后再导入"),
        ("expected_candidate_state_sha256", "candidate_state_sha256",
         "机器候选已变化，请重新合并当前候选"),
    )
    for request_key, current_key, message in checks:
        expected = payload.get(request_key)
        if not _valid_sha256(expected):
            raise ValueError(f"{request_key} 必须是当前工作区 SHA-256")
        if expected != current[current_key]:
            raise StaleHumanReviewError(message)
    return generation, path, topic, current


def assert_request_label_state(payload):
    """Reject label writes based on another file load or an older label list."""
    generation, path, topic = assert_request_workspace_scope(payload)
    expected = payload.get("expected_label_state_sha256")
    if not _valid_sha256(expected):
        raise ValueError("expected_label_state_sha256 必须是当前类别状态 SHA-256")
    current = current_inference_label_state()
    if expected != label_state_sha256(current):
        raise StaleHumanReviewError(
            "类别或推理类别已在另一窗口变化，请刷新后重新设置")
    return generation, path, topic, current


def merge_inference_candidates(existing, generated, selected_labels, *, return_id_map=False):
    """Refresh selected classes without deleting candidates from other classes."""
    selected = set(selected_labels)
    preserved = [shape for shape in (existing or [])
                 if (shape.get("label") not in selected
                     or shape.get("source") == "validated_point_preset")]
    combined = copy.deepcopy(preserved + list(generated))
    used = {shape.get("id") for shape in preserved if type(shape.get("id")) is int}
    next_id = max(used, default=0) + 1
    id_map = {}
    for shape in combined[len(preserved):]:
        if type(shape.get("id")) is not int:
            continue
        original_id = shape["id"]
        if shape["id"] in used:
            while next_id in used:
                next_id += 1
            shape["id"] = next_id
            next_id += 1
        id_map.setdefault(original_id, []).append(shape["id"])
        used.add(shape["id"])
    return (combined, id_map) if return_id_map else combined


def bind_merged_candidate_references(result, id_map):
    """保留工作流原始记录，为最终候选引用添加显式映射。"""
    def remap(value):
        if isinstance(value, list):
            return [remap(item) for item in value]
        if not isinstance(value, dict):
            return value
        output = {}
        for key, item in value.items():
            if key == "candidate_id" and type(item) is int:
                output[key] = id_map.get(item, [item])[0]
                if output[key] != item:
                    output["original_" + key] = item
            elif key in {"shape_ids", "result_ids"} and isinstance(item, list):
                output[key] = [id_map.get(identifier, [identifier])[0]
                               if type(identifier) is int else identifier for identifier in item]
                if output[key] != item:
                    output["original_" + key] = copy.deepcopy(item)
            else:
                output[key] = remap(item)
        return output
    original_agent = copy.deepcopy(result["agent"])
    result["agent"] = remap(result["agent"])
    result["review_queue"] = remap(result["review_queue"])
    result["applied"] = remap(result["applied"])
    result["agent"]["candidate_id_mapping"] = [
        {"original_id": original_id, "final_ids": final_ids}
        for original_id, final_ids in id_map.items()]
    if any(final_ids != [original_id] for original_id, final_ids in id_map.items()):
        result["agent"]["workflow_original_agent"] = original_agent


def archive_candidate_revision(frame, topic, existing_candidates, existing_run,
                               replacement_candidates, replacement_run, reason):
    """Preserve the exact prior machine layer before refreshing current candidates."""
    if not existing_candidates and not existing_run:
        return None
    prior_candidates = copy.deepcopy(existing_candidates or [])
    prior_run = copy.deepcopy(existing_run or {})
    replacement_candidates = copy.deepcopy(replacement_candidates or [])
    replacement_run = copy.deepcopy(replacement_run or {})
    prior_candidate_sha = _sha256_json(prior_candidates)
    prior_run_sha = _sha256_json(prior_run)
    replacement_candidate_sha = _sha256_json(replacement_candidates)
    replacement_run_sha = _sha256_json(replacement_run)
    if (prior_candidate_sha == replacement_candidate_sha
            and prior_run_sha == replacement_run_sha):
        return None
    identity = {
        "file": str(state["video"].path) if state.get("video") else None,
        "topic": topic, "frame": int(frame), "reason": reason,
        "prior_candidate_sha256": prior_candidate_sha,
        "prior_agent_run_sha256": prior_run_sha,
        "replacement_candidate_sha256": replacement_candidate_sha,
        "replacement_agent_run_sha256": replacement_run_sha,
    }
    revision_id = _sha256_json(identity)
    ledger = state.setdefault("candidate_revisions", {}).setdefault(str(frame), [])
    existing = next((item for item in ledger if item.get("revision_id") == revision_id), None)
    if existing is not None:
        return existing
    declared_prior_sha = prior_run.get("candidate_artifact_sha256")
    record = {
        "schema_version": "labelagent.candidate_revision.v1",
        "revision_id": revision_id,
        "sequence": len(ledger) + 1,
        "recorded_at_unix": time.time(),
        "reason": reason,
        "file": identity["file"], "topic": topic, "frame": int(frame),
        "candidate_only": True, "human_truth": False,
        "approved_for_training": False,
        "prior_candidate_event_id": prior_run.get("candidate_event_id"),
        "prior_candidate_request_id": prior_run.get("candidate_request_id"),
        "prior_candidate_output_sha256": prior_run.get("candidate_output_sha256"),
        "prior_candidate_artifact_sha256": prior_candidate_sha,
        "prior_declared_candidate_artifact_sha256": declared_prior_sha,
        "prior_artifact_integrity_match": (declared_prior_sha is None
                                             or declared_prior_sha == prior_candidate_sha),
        "prior_agent_run_sha256": prior_run_sha,
        "superseded_by_candidate_event_id": replacement_run.get("candidate_event_id"),
        "superseded_by_candidate_request_id": replacement_run.get("candidate_request_id"),
        "superseded_by_candidate_output_sha256": replacement_run.get("candidate_output_sha256"),
        "superseded_by_candidate_artifact_sha256": replacement_candidate_sha,
        "superseded_by_agent_run_sha256": replacement_run_sha,
        "prior_candidates": prior_candidates,
        "prior_agent_run": prior_run,
    }
    ledger.append(record)
    return record


def candidate_revision_summary():
    fields = ("revision_id", "sequence", "recorded_at_unix", "reason", "topic", "frame",
              "prior_candidate_event_id", "prior_candidate_request_id",
              "prior_candidate_output_sha256", "prior_candidate_artifact_sha256",
              "prior_artifact_integrity_match", "superseded_by_candidate_event_id",
              "superseded_by_candidate_request_id", "superseded_by_candidate_output_sha256",
              "superseded_by_candidate_artifact_sha256", "candidate_only", "human_truth",
              "approved_for_training")
    return {frame: [{key: revision.get(key) for key in fields} for revision in revisions]
            for frame, revisions in state.get("candidate_revisions", {}).items()}


def selected_frame_payload(index):
    """Return one frame's mutable layers with their server-canonical bindings."""
    key = str(index)
    truth = state.get("truth", {})
    candidates = state.get("candidates", {})
    agent_runs = state.get("agent_runs", {})
    revisions = candidate_revision_summary()
    review_status = truth_review_status_for(
        state.get("topic"), {key: truth[key]} if key in truth else {})
    binding = candidate_review_binding(candidates.get(key))
    proposals = [copy.deepcopy(item)
                 for item in state.get("candidate_review_proposals", [])
                 if (item.get("file") == str(state["video"].path)
                     and item.get("topic") == state.get("topic")
                     and item.get("frame") == index)]
    return {
        "truth": {key: copy.deepcopy(truth[key])} if key in truth else {},
        "truth_review_status": review_status,
        "candidates": {key: copy.deepcopy(candidates[key])} if key in candidates else {},
        "agent_runs": {key: copy.deepcopy(agent_runs[key])} if key in agent_runs else {},
        "candidate_revision_summary": {key: copy.deepcopy(revisions[key])}
            if key in revisions else {},
        "candidate_review_proposals": proposals,
        "candidate_review_bindings": {key: binding} if binding is not None else {},
        "candidate_class_audit_version": candidate_class_audit_version(),
    }


def compact_topic_payload(index, description):
    references = state.get("reference_frames", {}).get(state["topic"], [])
    reference_index = references[0] if references else None
    auto_labels = (selected_auto_labels(reference_index)
                   if reference_index is not None else list(state.get("labels", [])))
    reference_only_labels = (
        reference_labels(state.get("truth", {}).get(str(reference_index), []),
                         state.get("labels", []))
        if reference_index is not None else [])
    payload = selected_frame_payload(index)
    payload.update({
        "compact_geometry": True,
        "file_activation_generation": current_file_activation_generation(),
        "path": str(state["video"].path),
        "topic": state["topic"],
        "frame": index,
        "truth_frames": sorted(state.get("truth", {}), key=lambda value: int(value)),
        "candidate_counts": {
            str(frame): len(items) if isinstance(items, list) else 0
            for frame, items in state.get("candidates", {}).items()
        },
        "reference_frames": copy.deepcopy(state.get("reference_frames", {})),
        "labels": copy.deepcopy(state.get("labels", [])),
        "auto_labels": auto_labels,
        "reference_only_labels": reference_only_labels,
        "label_state_sha256": label_state_sha256({
            "labels": state.get("labels", []),
            "auto_labels": auto_labels,
        }),
        "description": copy.deepcopy(description),
        "description_cache_hit": description is not None,
    })
    return payload


def reference_candidate_event_id(candidate_request_id, candidate_output_sha256):
    """Bind a candidate event to both the stable request and actual model output."""
    return _sha256_json({"event": "reference_draft_candidate.v2",
                         "candidate_request_id": candidate_request_id,
                         "candidate_output_sha256": candidate_output_sha256})


@app.errorhandler(ValueError)
@app.errorhandler(KeyError)
def bad_request(error):
    return jsonify(error=str(error)), 400


@app.errorhandler(RuntimeError)
def runtime_error(error):
    return jsonify(error=str(error)), 503


@app.errorhandler(StaleHumanReviewError)
def stale_human_review(error):
    return jsonify(error=str(error), stale_review=True), 409


@app.get("/")
def index():
    return app.send_static_file("index.html")


def candidate_class_audit_version():
    # 配置和协议源码变化均使旧机器审核失效；不向浏览器暴露凭据。
    return {
        "schema_version": "labelagent.candidate_class_audit_version.v1",
        "config_sha256": _sha256_json({
            key: value for key, value in state.get("config", {}).items()
            if key not in {"llm_key", "llm_fast_key"}}),
        "protocol_sha256": _CANDIDATE_AUDIT_LOADED_CODE_SHA256,
        "labels_sha256": _sha256_json({"labels": state.get("labels", []),
                                       "definitions": VISIBLE_LABEL_DEFINITIONS}),
    }


def reusable_candidate_class_audit(run, review_status, candidates, labels,
                                   artifact_sha, input_sha, audit_version, *,
                                   candidate_ids=None, require_review_bindings=False):
    """Return True only for a complete, byte-bound persisted audit receipt.

    This is deliberately stricter than the model response cache: it survives a
    service restart, but any image, candidate, registry, model configuration,
    protocol, audit payload, or review-binding change forces a fresh audit.
    """
    if (not isinstance(run, dict)
            or run.get("candidate_class_audit_status") != "complete"
            or run.get("candidate_class_audit_version") != audit_version
            or run.get("candidate_artifact_sha256") != artifact_sha
            or run.get("input_sha256") != input_sha
            or not isinstance(review_status, dict)):
        return False
    targets = [candidate for candidate in candidates
               if candidate_requires_class_audit(candidate)
               and (candidate_ids is None or candidate.get("id") in candidate_ids)]
    audits = run.get("candidate_class_audits")
    if not isinstance(audits, list) or len(audits) != len(targets):
        return False
    by_id = {item.get("candidate_id"): item for item in audits
             if isinstance(item, dict) and type(item.get("candidate_id")) is int}
    if len(by_id) != len(audits):
        return False
    for candidate in targets:
        try:
            fingerprint = candidate_fingerprint(candidate)
        except (TypeError, ValueError):
            return False
        audit = by_id.get(candidate.get("id"))
        try:
            decision = validate_blind_candidate_class_decision(
                copy.deepcopy(audit.get("decision")) if isinstance(audit, dict) else None,
                labels)
        except (TypeError, ValueError):
            return False
        minimum_confidence = audit.get("minimum_confidence")
        expected_route = (
            "candidate_preserved"
            if decision["classification"] == "known_label"
            and decision["observed_label"] == candidate.get("label")
            and isinstance(minimum_confidence, (int, float))
            and not isinstance(minimum_confidence, bool)
            and decision["confidence"] >= minimum_confidence
            else "human_review_required")
        if (not isinstance(audit, dict)
                or audit.get("schema_version") != "labelagent.blind_candidate_class_audit.v1"
                or audit.get("candidate_sha256") != fingerprint
                or audit.get("candidate_artifact_sha256") != artifact_sha
                or audit.get("source_image_sha256") != input_sha
                or audit.get("bound_current_label") != candidate.get("label")
                or audit.get("audit_version") != audit_version
                or audit.get("route") != expected_route
                or audit.get("candidate_mutated") is not False
                or audit.get("human_truth") is not False
                or audit.get("approved_for_training") is not False
                or audit.get("approved_for_deployment") is not False
                or not _valid_sha256(audit.get("audit_sha256"))
                or audit["audit_sha256"] != _sha256_json({
                    key: value for key, value in audit.items()
                    if key != "audit_sha256"})):
            return False
        if require_review_bindings or review_status:
            binding = review_status.get(fingerprint)
            if (not isinstance(binding, dict)
                    or binding.get("candidate_fingerprint") != fingerprint
                    or binding.get("candidate_artifact_sha256") != artifact_sha
                    or binding.get("audit_sha256") != audit["audit_sha256"]
                    or binding.get("route") != audit["route"]):
                return False
    return set(by_id) == {candidate.get("id") for candidate in targets}


def candidate_class_review_queue(audits, artifact_sha):
    """Rebuild UI-only review rows from validated immutable audit receipts."""
    return [{
        "reason_code": ("candidate_class_blind_audit_failed" if audit.get("error")
                        else "candidate_class_blind_review_required"),
        "route": "human_review_required", "source": "candidate_class_audit",
        "candidate_id": audit["candidate_id"],
        "shape_ids": [audit["candidate_id"]],
        "candidate_sha256": audit["candidate_sha256"],
        "candidate_artifact_sha256": artifact_sha,
        "audit_sha256": audit["audit_sha256"],
        "classification": audit.get("decision", {}).get("classification"),
        "reason": audit.get("decision", {}).get(
            "reason", "候选类别盲审失败，请人工检查"),
        **({"error": audit["error"]} if audit.get("error") else {}),
    } for audit in audits if audit.get("route") == "human_review_required"]


@app.get("/api/state")
def get_state():
    with lock:
        return _get_state_locked()


@app.get("/api/state/summary")
def get_state_summary():
    """Return the small, geometry-free state needed by the always-on guide.

    Full candidate polygons and Agent traces remain available from ``/api/state``
    for explicit review actions.  Keeping them out of the background heartbeat
    avoids repeatedly parsing megabytes of unchanged JSON in the browser.
    """
    with lock:
        recover_pending_human_reviews()
        truth_review_status = current_truth_review_status()
        truth_frames = sorted(state.get("truth", {}), key=lambda value: int(value))
        candidate_counts = {
            str(frame): len(items) if isinstance(items, list) else 0
            for frame, items in state.get("candidates", {}).items()
        }
        return jsonify(
            schema_version="labelagent.state-summary.v1",
            file_activation_generation=current_file_activation_generation(),
            loaded=state.get("video") is not None,
            path=str(state["video"].path) if state.get("video") else "",
            topic=state.get("topic"),
            labels=state.get("labels", []),
            reference_frames=state.get("reference_frames", {}),
            truth_frames=truth_frames,
            verified_truth_frames=[
                frame for frame in truth_frames
                if truth_review_status.get(str(frame), {}).get("reusable_as_reference") is True
            ],
            candidate_counts=candidate_counts,
            topic_readiness=topic_readiness_summary(),
        )


def _reference_frame_index(record):
    """Summarize per-frame geometry without returning any polygon points."""
    result = {}
    for frame, items in record.items():
        if not isinstance(items, list) or not items:
            continue
        labels = []
        for item in items:
            label = item.get("label") if isinstance(item, dict) else None
            if label and label not in labels:
                labels.append(label)
        result[str(frame)] = {"count": len(items), "labels": labels}
    return result


def _js_json_value(value):
    """Normalize JSON numbers to the representation used by JSON.stringify."""
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return [_js_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _js_json_value(item) for key, item in value.items()}
    return value


def _candidate_browser_seed(candidate):
    """Reproduce the browser's FNV-1a review key without exposing geometry."""
    fields = ("id", "label", "score", "points", "parts", "holes")
    payload = {key: candidate[key] for key in fields if key in candidate}
    text = json.dumps(_js_json_value(payload), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))
    result = 2166136261
    encoded = text.encode("utf-16-le")
    for index in range(0, len(encoded), 2):
        result ^= encoded[index] | (encoded[index + 1] << 8)
        result = (result * 16777619) & 0xffffffff
    return f"{result:08x}"


def _batch_review_candidate(candidate):
    fields = ("id", "label", "score", "dataset_split", "split")
    result = {key: copy.deepcopy(candidate[key]) for key in fields if key in candidate}
    result["review_seed"] = _candidate_browser_seed(candidate)
    return result


def _batch_review_run(run):
    if not isinstance(run, dict):
        return None
    fields = ("dataset_split", "split", "failed", "error", "quality_failure",
              "qc_failed", "stop_reason", "converged")
    result = {key: copy.deepcopy(run[key]) for key in fields if key in run}
    result["review_queue"] = [{} for _ in (run.get("review_queue") or [])]
    return result


@app.get("/api/review/index")
def get_batch_review_index():
    """Return cross-frame risk metadata without any polygon geometry."""
    global batch_review_index_cache
    with lock:
        video = state.get("video")
        cache_key = (id(video), state.get("topic"), id(state.get("labels")),
                     id(state.get("config")), id(state.get("candidates")),
                     id(state.get("agent_runs")), id(state.get("candidate_review_status")))
        if not batch_review_index_cache or batch_review_index_cache["key"] != cache_key:
            candidates = {
                str(frame): [_batch_review_candidate(candidate) for candidate in items]
                for frame, items in state.get("candidates", {}).items()
                if isinstance(items, list) and items
            }
            runs = {
                frame: summary for frame in candidates
                if (summary := _batch_review_run(state.get("agent_runs", {}).get(frame))) is not None
            }
            statuses = {
                key: {
                    field: copy.deepcopy(value[field])
                    for field in ("frame", "topic", "remediation_required") if field in value
                }
                for key, value in state.get("candidate_review_status", {}).items()
                if isinstance(value, dict)
            }
            batch_review_index_cache = {"key": cache_key, "payload": {
                "schema_version": "labelagent.batch-review-index.v1",
                "compact_geometry": True,
                "loaded": video is not None,
                "path": str(video.path) if video is not None else "",
                "topic": state.get("topic"),
                "labels": copy.deepcopy(state.get("labels", [])),
                "config": {"dataset_split": state.get("config", {}).get("dataset_split")},
                "candidates": candidates,
                "agent_runs": runs,
                "candidate_review_status": statuses,
            }}
        return jsonify(batch_review_index_cache["payload"])


@app.get("/api/reference-preparation")
def get_reference_preparation():
    """Return the geometry-free index used by the reference preparation dialog.

    Selecting a frame still requires ``/api/frame_state/<frame>``.  This keeps
    opening the dialog cheap even when a dataset contains many dense masks.
    """
    with lock:
        generation_text = request.args.get("file_activation_generation")
        try:
            requested_generation = int(generation_text)
        except (TypeError, ValueError) as error:
            raise ValueError("file_activation_generation 必须是当前文件载入代次整数") from error
        assert_request_file_activation({
            "file_activation_generation": requested_generation})
        recover_pending_human_reviews()
        video = state.get("video")
        topic = state.get("topic")
        frame_count = (video.topics.get(topic, {}).get("count", 0)
                       if video is not None and topic else 0)
        references = state.get("reference_frames", {}).get(topic, []) if topic else []
        reference_index = references[0] if references else None
        auto_labels = (selected_auto_labels(reference_index)
                       if reference_index is not None else list(state.get("labels", [])))
        truth_status = current_truth_review_status()
        compact_status = {
            str(frame): {
                key: copy.deepcopy(status.get(key))
                for key in ("reusable_as_reference", "reason_code",
                            "legacy_source_counts", "truth_sha256")
                if key in status
            }
            for frame, status in truth_status.items()
        }
        return jsonify(
            schema_version="labelagent.reference-preparation.v1",
            compact_geometry=True,
            loaded=video is not None,
            path=str(video.path) if video is not None else "",
            topic=topic,
            file_activation_generation=requested_generation,
            frame_count=frame_count,
            labels=copy.deepcopy(state.get("labels", [])),
            auto_labels=auto_labels,
            label_state_sha256=label_state_sha256({
                "labels": state.get("labels", []),
                "auto_labels": auto_labels,
            }),
            reference_frames=copy.deepcopy(state.get("reference_frames", {})),
            truth_index=_reference_frame_index(state.get("truth", {})),
            truth_review_status=compact_status,
            candidate_index=_reference_frame_index(state.get("candidates", {})),
        )


def _get_bootstrap_locked():
    """Return enough metadata to open the workbench without all-frame geometry."""
    recover_pending_human_reviews()
    video = state.get("video")
    topic = state.get("topic")
    reference = state.get("reference_frames", {}).get(topic, []) if topic else []
    reference_index = reference[0] if reference else None
    description = (load_reference_description(reference_index)
                   if video is not None and reference_index is not None else None)
    auto_labels = (selected_auto_labels(reference_index)
                   if reference_index is not None else list(state.get("labels", [])))
    reference_only_labels = (
        reference_labels(state.get("truth", {}).get(str(reference_index), []),
                         state.get("labels", []))
        if reference_index is not None else [])
    return {
        "schema_version": "labelagent.workbench-bootstrap.v1",
        "file_activation_generation": current_file_activation_generation(),
        "compact_geometry": True,
        "loaded": video is not None,
        "path": str(video.path) if video is not None else "",
        "catalog": video.catalog if video is not None else {},
        "duration": video.duration if video is not None else 0,
        "topics": video.topics if video is not None else {},
        "topic": topic,
        "labels": copy.deepcopy(state.get("labels", [])),
        "auto_labels": auto_labels,
        "reference_only_labels": reference_only_labels,
        "label_state_sha256": label_state_sha256({
            "labels": state.get("labels", []),
            "auto_labels": auto_labels,
        }),
        "candidate_import_binding": candidate_import_binding(),
        "reference_frames": copy.deepcopy(state.get("reference_frames", {})),
        "truth": {},
        "truth_review_status": {},
        "truth_frames": sorted(state.get("truth", {}), key=lambda value: int(value)),
        "candidates": {},
        "candidate_counts": {
            str(frame): len(items) if isinstance(items, list) else 0
            for frame, items in state.get("candidates", {}).items()
        },
        "agent_runs": {},
        "candidate_review_proposals": [],
        "candidate_review_bindings": {},
        "config": public_model_config(),
        "config_state_sha256": config_state_sha256(),
        "config_write_binding": config_write_binding(),
        "description": description,
    }


@app.get("/api/bootstrap")
def get_bootstrap():
    with lock:
        return jsonify(_get_bootstrap_locked())


def _get_state_locked():
    recover_pending_human_reviews()
    description = None
    reference = state.get("reference_frames", {}).get(state["topic"], []) if state["topic"] else []
    if state["video"] and reference:
        description = load_reference_description(reference[0])
    reference_index = reference[0] if reference else None
    auto_labels = selected_auto_labels(reference_index) if reference_index is not None else list(state["labels"])
    reference_only_labels = (reference_labels(state["truth"][str(reference_index)], state["labels"])
                             if reference_index is not None else [])
    truth_review_status = current_truth_review_status()
    return jsonify(loaded=state["video"] is not None,
        file_activation_generation=current_file_activation_generation(),
        path=str(state["video"].path) if state["video"] else "",
        catalog=state["video"].catalog if state["video"] else {}, duration=state["video"].duration if state["video"] else 0,
        topics=state["video"].topics if state["video"] else {}, topic=state["topic"],
        labels=state["labels"], auto_labels=auto_labels,
        reference_only_labels=reference_only_labels,
        label_state_sha256=label_state_sha256({
            "labels": state.get("labels", []),
            "auto_labels": auto_labels,
        }),
        truth=state["truth"], truth_review_status=truth_review_status,
        reference_frames=state.get("reference_frames", {}),
        config=public_model_config(),
        config_state_sha256=config_state_sha256(),
        config_write_binding=config_write_binding(),
        candidates=state.get("candidates", {}), agent_runs=state.get("agent_runs", {}),
        candidate_revision_summary=candidate_revision_summary(),
        candidate_review_status=state.get("candidate_review_status", {}),
        candidate_review_proposals=state.get("candidate_review_proposals", []),
        candidate_review_bindings=candidate_review_bindings(),
        candidate_class_audit_version=candidate_class_audit_version(),
        description=description)


@app.get("/api/frame/<int:index>")
def get_frame(index):
    data = state["video"].frame(request.args.get("topic", state["topic"]), index)
    return send_file(io.BytesIO(data), mimetype="image/png" if data.startswith(b"\x89PNG") else "image/jpeg", max_age=0)


@app.get("/api/frame_state/<int:index>")
def get_frame_state(index):
    """Read one frame's geometry after a compact topic activation."""
    with lock:
        if state.get("video") is None or not state.get("topic"):
            raise ValueError("请先加载图像数据")
        requested_topic = request.args.get("topic")
        requested_path = request.args.get("path")
        generation_text = request.args.get("file_activation_generation")
        try:
            requested_generation = int(generation_text)
        except (TypeError, ValueError) as error:
            raise ValueError("file_activation_generation 必须是当前文件载入代次整数") from error
        assert_request_file_activation({
            "file_activation_generation": requested_generation})
        if requested_topic != state["topic"]:
            raise StaleHumanReviewError("相机已变化，拒绝返回旧帧标注")
        if requested_path != str(state["video"].path):
            raise StaleHumanReviewError("数据文件已变化，拒绝返回旧帧标注")
        count = state["video"].topics[state["topic"]]["count"]
        if not 0 <= index < count:
            raise ValueError("帧号无效")
        payload = selected_frame_payload(index)
        payload.update(path=str(state["video"].path), topic=state["topic"],
                       frame=index, compact_geometry=True,
                       file_activation_generation=requested_generation,
                       reference_frames=copy.deepcopy(state.get("reference_frames", {})))
        return jsonify(payload)


@app.get("/api/downstream_projection/<int:index>")
def get_downstream_projection(index):
    """Expose the project-defined floor/slope union without changing training labels."""
    with lock:
        if state["video"] is None or not state["topic"]:
            raise ValueError("请先加载图像消息")
        count = state["video"].topics[state["topic"]]["count"]
        if not 0 <= index < count:
            raise ValueError("帧号无效")
        key = str(index)
        has_truth = key in state.get("truth", {})
        source = copy.deepcopy((state.get("truth", {}) if has_truth
                                else state.get("candidates", {})).get(key, []))
    traversable, passthrough = [], []
    for shape in source:
        if shape.get("label") in ("floor", "slope"):
            projected = copy.deepcopy(shape)
            projected["source_semantic_label"] = shape["label"]
            projected["label"] = "traversable_area"
            traversable.append(projected)
        else:
            passthrough.append(shape)
    return jsonify(projection={
        "frame": index,
        "topic": state["topic"],
        "source_layer": "human_truth" if has_truth else "machine_candidate",
        "machine_candidate_only": not has_truth,
        "training_labels_preserved": True,
        "policy_basis": "user_defined_project_semantics",
        "mapping": {"operation": "union", "source_labels": ["floor", "slope"],
                    "target_label": "traversable_area"},
        "traversable_area_parts": traversable,
        "passthrough_shapes": passthrough,
    })


class _PointSegmentFlight:
    """One exact point request shared only while its owner is still running."""

    def __init__(self):
        self.event = threading.Event()
        self.payload = None
        self.error = None


def _point_segment_result(image, points, service_url):
    """Run one point request under the existing global and SAM safety gates."""
    if not point_segment_lock.acquire(blocking=False):
        raise RuntimeError("已有不同的点提示 SAM 请求正在计算，请稍后再试")
    payload = json.dumps({"image": base64.b64encode(image).decode("ascii"),
                          "points": points}).encode()
    service_request = urllib.request.Request(service_url + "/segment_points", data=payload,
        headers={"Content-Type": "application/json"})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(
                service_request, timeout=POINT_SEGMENT_TIMEOUT_SECONDS) as response:
            result = json.load(response)
        result["inference_elapsed_seconds"] = round(time.monotonic() - started, 2)
        return result
    except Exception as error:
        raise RuntimeError(f"SAM 3 服务不可用（{service_url}）：{error}") from error
    finally:
        point_segment_lock.release()


@app.post("/api/segment_points")
def get_point_segment():
    body = request.get_json() or {}
    if run_claim_lock.locked() and ("frame" not in body or "points" not in body):
        raise RuntimeError("已有推理任务正在运行，请稍后重试")
    with lock:
        if state["video"] is None or not state["topic"]:
            raise ValueError("请先加载图像消息")
        activation_generation, data_path, topic = assert_request_data_scope(body)
        index = int(body["frame"])
        if index in state.get("reference_frames", {}).get(state["topic"], []):
            raise ValueError("参考帧只能人工调整，不能调用 SAM 生成或修改")
        image = frame_image(index)
        service_url = (state["config"].get("sam3_url") or os.environ.get("SAM3_URL")
                       or "http://127.0.0.1:8766").rstrip("/")
    points = body["points"]
    request_key = _sha256_json({
        "schema_version": "labelagent.point-segment-flight.v1",
        "service_url": service_url,
        "path": data_path,
        "topic": topic,
        "file_activation_generation": activation_generation,
        "frame": index,
        "image_sha256": hashlib.sha256(image).hexdigest(),
        "points": points,
    })
    with point_segment_flights_lock:
        flight = point_segment_flights.get(request_key)
        owner = flight is None
        if owner:
            flight = _PointSegmentFlight()
            point_segment_flights[request_key] = flight
    if not owner:
        wait_started = time.monotonic()
        if not flight.event.wait(POINT_SEGMENT_TIMEOUT_SECONDS + 5):
            raise RuntimeError("等待相同点提示 SAM 请求超时，请稍后重试")
        if flight.error is not None:
            raise RuntimeError(flight.error)
        if flight.payload is None:
            raise RuntimeError("相同点提示 SAM 请求结束但没有返回结果")
        with lock:
            assert_request_data_scope(body)
        result = copy.deepcopy(flight.payload)
        result["request_coalescing"] = {
            "role": "coalesced_waiter", "coalesced": True,
            "wait_seconds": round(time.monotonic() - wait_started, 2),
            "request_sha256": request_key,
        }
        return jsonify(result)

    compute = run_claimed(lambda: _point_segment_result(
        image, points, service_url))
    try:
        result = compute()
        with lock:
            assert_request_data_scope(body)
        result.update(path=data_path, topic=topic, frame=index,
                      file_activation_generation=activation_generation)
        flight.payload = copy.deepcopy(result)
        result["request_coalescing"] = {
            "role": "owner", "coalesced": False, "wait_seconds": 0.0,
            "request_sha256": request_key,
        }
        return jsonify(result)
    except BaseException as error:
        flight.error = str(error)[:500]
        raise
    finally:
        flight.event.set()
        with point_segment_flights_lock:
            if point_segment_flights.get(request_key) is flight:
                point_segment_flights.pop(request_key, None)

@app.post("/api/edit_region")
def edit_region():
    """Union or subtract a user drawn polygon from an existing annotation."""
    body = request.get_json() or {}
    with lock:
        if state.get("video") is None or not state.get("topic"):
            raise ValueError("请先加载图像消息")
        activation_generation, data_path, topic = assert_request_data_scope(body)
        index = int(body.get("frame", -1))
        image_payload = frame_image(index)
    shape = body.get("shape") or {}
    polygon = body.get("polygon") or []
    operation = body.get("operation")
    if operation not in ("add", "erase") or len(polygon) < 3:
        raise ValueError("区域至少需要 3 个点")
    image = Image.open(io.BytesIO(image_payload))
    width, height = image.size
    import cv2
    canvas = np.zeros((height, width), np.uint8)
    def paint(item, value=1):
        parts = item.get("parts") or [item]
        for part in parts:
            pts = np.asarray(part.get("points", []), np.int32)
            if len(pts) >= 3: cv2.fillPoly(canvas, [pts], value)
            for hole in part.get("holes", []):
                hp = np.asarray(hole, np.int32)
                if len(hp) >= 3: cv2.fillPoly(canvas, [hp], 0)
    paint(shape)
    region = np.zeros_like(canvas)
    cv2.fillPoly(region, [np.asarray(polygon, np.int32)], 1)
    canvas = np.maximum(canvas, region) if operation == "add" else np.where(region, 0, canvas).astype(np.uint8)
    contours, hierarchy = cv2.findContours(canvas, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: raise ValueError("编辑后区域为空")
    parts = []
    for i, c in enumerate(contours):
        if hierarchy[0][i][3] != -1 or cv2.contourArea(c) < 2: continue
        pts = c[:, 0, :].astype(int).tolist(); holes=[]; child=hierarchy[0][i][2]
        while child != -1:
            holes.append(contours[child][:,0,:].astype(int).tolist()); child=hierarchy[0][child][0]
        parts.append({"points": pts, "holes": holes})
    if not parts: raise ValueError("编辑后区域为空")
    with lock:
        assert_request_data_scope(body)
    return jsonify(label=shape.get("label", ""), points=parts[0]["points"],
                   holes=parts[0]["holes"], parts=parts, path=data_path,
                   topic=topic, frame=index,
                   file_activation_generation=activation_generation)


@app.get("/api/playback")
def get_playback():
    if state["video"] is None:
        raise ValueError("请先打开 MCAP")
    topic = request.args.get("topic", state["topic"])
    data = state["video"].playback(topic, WORK / "playback")
    data['offset'] = state["video"].catalog[topic].get('offset', 0)
    return jsonify(data)


@app.get("/api/timeline")
def get_timeline():
    with lock:
        if state["video"] is None:
            raise ValueError("请先打开 MCAP")
        video = state["video"]
        topic = request.args.get("topic", state["topic"])
        source_path = str(video.path)
    data = video.timeline(topic, WORK / "playback")
    with lock:
        if state.get("video") is not video or str(state["video"].path) != source_path:
            raise StaleHumanReviewError("时间轴生成期间数据文件已变化")
        data["offset"] = video.catalog[topic].get("offset", 0)
        data["path"] = source_path
        data["topic"] = topic
    return jsonify(data)


@app.get("/api/cloud")
def get_cloud():
    if state["video"] is None:
        raise ValueError("请先打开 MCAP")
    seconds = float(request.args.get('time', 0))
    if not math.isfinite(seconds) or not 0 <= seconds <= state["video"].duration + 1:
        raise ValueError("时间超出记录范围")
    data, count, messages = cloud_snapshot(state["video"], request.args.get('topic'), seconds)
    response = app.response_class(data, mimetype='application/octet-stream')
    response.headers['X-Point-Count'] = str(count)
    response.headers['X-Message-Count'] = str(messages)
    return response


@app.get("/api/media/<name>")
def get_media(name):
    if len(name) != 68 or not name.endswith(".mp4") or any(c not in "0123456789abcdef" for c in name[:-4]):
        raise ValueError("无效的视频名称")
    return send_file(WORK / "playback" / name, mimetype="video/mp4", conditional=True)


def library_path():
    return WORK / "files.json"


def read_library():
    return json.loads(library_path().read_text(encoding="utf-8")) if library_path().exists() else []


def write_library(files):
    temporary = library_path().with_suffix('.tmp')
    temporary.write_text(json.dumps(files, ensure_ascii=False, indent=2), encoding="utf-8")
    _commit_work_file(temporary, library_path())


def register_file(path, name=None):
    path = Path(path).expanduser().resolve()
    if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError('请选择存在的 JPG、PNG 图片或 MCAP 文件')
    files = read_library()
    existing = next((item for item in files if item['path'] == str(path)), None)
    if existing:
        return existing
    if path.suffix.lower() in IMAGE_SUFFIXES:
        read_image(path)
    item = dict(id=uuid.uuid4().hex, path=str(path), name=name or path.name, size=path.stat().st_size)
    files.append(item)
    write_library(files)
    return item


def file_library_state_sha256(files=None):
    """Version the exact registered-file list shown to one browser window."""
    return _sha256_json(read_library() if files is None else files)


def file_list(files=None):
    files = read_library() if files is None else files
    current = str(state['video'].path.resolve()) if state['video'] else None
    return [dict(item, available=Path(item['path']).is_file(),
                 loaded=item['path'] == current,
                 entry_sha256=_sha256_json(item)) for item in files]


def _review_packet_id(value):
    value = str(value or "")
    if (not 1 <= len(value) <= 80
            or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in value)):
        raise ValueError("专项复核包标识无效")
    return value


def _review_packet_path(packet_id):
    packet_id = _review_packet_id(packet_id)
    root = (WORK / "review_packets").resolve()
    path = (root / packet_id / "manifest.json").resolve()
    if path.parent.parent != root:
        raise ValueError("专项复核包路径越界")
    if not path.is_file():
        raise ValueError("专项复核包不存在")
    return path


def _valid_sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value.lower()))


def _read_review_packet(packet_id):
    """Read and fail closed on a local, read-only review packet manifest."""
    path = _review_packet_path(packet_id)
    raw = path.read_bytes()
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError("专项复核包清单过大")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("专项复核包清单不是有效 UTF-8 JSON") from error
    if not isinstance(value, dict) or value.get("schema_version") != REVIEW_PACKET_SCHEMA:
        raise ValueError("专项复核包版本不受支持")
    mcap = value.get("mcap")
    topic = value.get("topic")
    rows = value.get("rows")
    if (not isinstance(mcap, str) or not Path(mcap).is_absolute()
            or Path(mcap).suffix.lower() != ".mcap"):
        raise ValueError("专项复核包没有绑定有效 MCAP")
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("专项复核包没有绑定图像 topic")
    if not _valid_sha256(value.get("mcap_sha256")):
        raise ValueError("专项复核包 MCAP 哈希无效")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 500:
        raise ValueError("专项复核包帧清单为空或过大")
    normalized_rows = []
    seen_frames = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("专项复核包帧记录无效")
        frame = row.get("frame")
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0 or frame in seen_frames:
            raise ValueError("专项复核包包含无效或重复帧号")
        seen_frames.add(frame)
        if not _valid_sha256(row.get("source_frame_sha256")):
            raise ValueError(f"专项复核包第 {frame} 帧原图哈希无效")
        if (row.get("candidate_only") is not True
                or row.get("human_truth") is not False
                or row.get("approved_for_training") is not False
                or row.get("training_return_allowed") is not False
                or row.get("review_status") != "pending_human_review"
                or row.get("human_decision") is not None
                or row.get("reviewer_declared_id") is not None):
            raise ValueError(f"专项复核包第 {frame} 帧越过候选/训练隔离门禁")
        decisions = row.get("allowed_human_decisions")
        if (not isinstance(decisions, list) or not decisions
                or any(decision not in REVIEW_PACKET_ACCEPT_DECISIONS for decision in decisions)):
            raise ValueError(f"专项复核包第 {frame} 帧人工决策集合无效")
        cohort = row.get("cohort")
        if cohort not in {"positive_proxy", "negative_cohort"}:
            raise ValueError(f"专项复核包第 {frame} 帧分组无效")
        candidate_count = row.get("candidate_count")
        if isinstance(candidate_count, bool) or not isinstance(candidate_count, int) or candidate_count < 0:
            raise ValueError(f"专项复核包第 {frame} 帧候选数无效")
        normalized_rows.append({
            "frame": frame,
            "cohort": cohort,
            "source_frame_sha256": row["source_frame_sha256"].lower(),
            "route": str(row.get("route") or "")[:100],
            "candidate_count": candidate_count,
            "target_provenance": str(row.get("target_provenance") or "")[:100],
            "review_status": "pending_human_review",
            "candidate_only": True,
            "human_truth": False,
            "approved_for_training": False,
            "training_return_allowed": False,
            "allowed_human_decisions": list(decisions),
        })
    summary = value.get("summary") or {}
    if summary.get("frames") != len(normalized_rows):
        raise ValueError("专项复核包汇总帧数与清单不一致")
    positives = sum(row["cohort"] == "positive_proxy" for row in normalized_rows)
    negatives = len(normalized_rows) - positives
    if (summary.get("positive_proxy_frames") != positives
            or summary.get("negative_cohort_frames") != negatives
            or summary.get("pending_human_review") != len(normalized_rows)
            or summary.get("human_truth_frames") != 0
            or summary.get("training_approved_frames") != 0):
        raise ValueError("专项复核包汇总越过来源或训练隔离门禁")
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    proposals = _read_review_packet_assistant_proposals(
        path.parent, _review_packet_id(packet_id), manifest_sha256, normalized_rows)
    for row in normalized_rows:
        row["assistant_proposal"] = proposals.get(row["frame"])
    return {
        "schema_version": REVIEW_PACKET_SCHEMA,
        "packet_id": _review_packet_id(packet_id),
        "manifest_sha256": manifest_sha256,
        "mcap_path": str(Path(mcap).resolve()),
        "mcap_sha256": value["mcap_sha256"].lower(),
        "mcap_available": Path(mcap).is_file(),
        "topic": topic,
        "summary": {
            "frames": len(normalized_rows),
            "positive_proxy_frames": positives,
            "negative_cohort_frames": negatives,
            "pending_human_review": len(normalized_rows),
            "human_truth_frames": 0,
            "training_approved_frames": 0,
        },
        "rows": normalized_rows,
        "read_only_packet": True,
        "submits_human_truth": False,
        "assistant_proposals_loaded": len(proposals),
    }


def _read_review_packet_assistant_proposals(folder, packet_id, manifest_sha256, rows):
    """Read optional assistant suggestions without allowing them to become decisions."""
    path = folder / "assistant-review.json"
    if not path.exists():
        return {}
    raw = path.read_bytes()
    if len(raw) > 1024 * 1024:
        raise ValueError("专项复核助手建议文件过大")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("专项复核助手建议不是有效 UTF-8 JSON") from error
    if (not isinstance(value, dict)
            or value.get("schema_version") != REVIEW_PACKET_ASSISTANT_SCHEMA
            or value.get("packet_id") != packet_id
            or value.get("manifest_sha256") != manifest_sha256):
        raise ValueError("专项复核助手建议未绑定当前清单")
    packet_rows = {row["frame"]: row for row in rows}
    proposals = {}
    for proposal in value.get("rows", []):
        if not isinstance(proposal, dict):
            raise ValueError("专项复核助手建议行无效")
        frame = proposal.get("frame")
        row = packet_rows.get(frame)
        decision = proposal.get("proposed_decision")
        rationale = proposal.get("rationale")
        confidence = proposal.get("confidence_band")
        if frame in proposals or row is None:
            raise ValueError("专项复核助手建议包含未知或重复帧")
        if decision not in row["allowed_human_decisions"]:
            raise ValueError("专项复核助手建议决定不在人工允许集合")
        if confidence not in {"high", "medium", "low"}:
            raise ValueError("专项复核助手建议置信分档无效")
        if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 500:
            raise ValueError("专项复核助手建议理由无效")
        if (proposal.get("assistant_only") is not True
                or proposal.get("human_truth") is not False
                or proposal.get("approved_for_training") is not False
                or proposal.get("auto_submit_allowed") is not False):
            raise ValueError("专项复核助手建议越过人工/训练隔离门禁")
        sanitized = {
            "proposed_decision": decision,
            "confidence_band": confidence,
            "rationale": rationale.strip(),
            "assistant_only": True,
            "human_truth": False,
            "approved_for_training": False,
            "auto_submit_allowed": False,
        }
        sanitized["proposal_sha256"] = _sha256_json(sanitized)
        proposals[frame] = sanitized
    return proposals


def _review_packet_decision_folder(packet_id):
    folder = (_review_packet_path(packet_id).parent / "decisions").resolve()
    root = (WORK / "review_packets" / _review_packet_id(packet_id)).resolve()
    if folder.parent != root:
        raise ValueError("专项复核决定路径越界")
    return folder


def _review_packet_decision_records(packet):
    """Read and verify the append-only, packet-bound scoped decision chain."""
    folder = _review_packet_decision_folder(packet["packet_id"])
    records = []
    previous = None
    for path in sorted(folder.glob("*.json")) if folder.exists() else []:
        value = json.loads(path.read_text(encoding="utf-8"))
        decision_id = value.get("decision_id") if isinstance(value, dict) else None
        body = {key: copy.deepcopy(item) for key, item in value.items()
                if key != "decision_id"} if isinstance(value, dict) else {}
        if (body.get("schema_version") != REVIEW_PACKET_DECISION_SCHEMA
                or body.get("packet_id") != packet["packet_id"]
                or body.get("manifest_sha256") != packet["manifest_sha256"]
                or body.get("previous_decision_id") != previous
                or not _valid_sha256(decision_id)
                or _sha256_json(body) != decision_id
                or body.get("human_truth") is not False
                or body.get("approved_for_training") is not False
                or body.get("training_return_allowed") is not False):
            raise RuntimeError(f"专项复核决定审计链损坏：{path.name}")
        value["decision_id"] = decision_id
        records.append(value)
        previous = decision_id
    return records


def _current_review_packet_decisions(packet):
    latest = {}
    for record in _review_packet_decision_records(packet):
        latest[record["frame"]] = record
    return [latest[row["frame"]] for row in packet["rows"] if row["frame"] in latest]


def _review_packet_summary(packet):
    current_path = str(state["video"].path.resolve()) if state.get("video") else ""
    return {
        key: copy.deepcopy(packet[key]) for key in (
            "schema_version", "packet_id", "manifest_sha256", "mcap_sha256",
            "mcap_available", "topic", "summary", "read_only_packet",
            "submits_human_truth", "assistant_proposals_loaded")
    } | {
        "title": f"斜坡集中复核 · {packet['summary']['frames']} 帧",
        "active_file": current_path == packet["mcap_path"],
        "active_topic": state.get("topic") == packet["topic"],
        "ready_to_navigate": (current_path == packet["mcap_path"]
                              and state.get("topic") == packet["topic"]),
    }


def _review_packet_file_sha256(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _activate_review_packet_topic(topic):
    if topic not in state["video"].topics:
        raise ValueError("专项复核包绑定的图像 topic 不存在")
    if topic == state["topic"]:
        return
    state.setdefault("truths_by_topic", {})[state["topic"]] = state["truth"]
    state.setdefault("candidates_by_topic", {})[state["topic"]] = state.get("candidates", {})
    state.setdefault("agent_runs_by_topic", {})[state["topic"]] = state.get("agent_runs", {})
    state.setdefault("candidate_revisions_by_topic", {})[state["topic"]] = state.get("candidate_revisions", {})
    state["topic"] = topic
    state["truth"] = state["truths_by_topic"].get(topic, {})
    state["candidates"] = state["candidates_by_topic"].get(topic, {})
    state["agent_runs"] = state["agent_runs_by_topic"].get(topic, {})
    state["candidate_revisions"] = state["candidate_revisions_by_topic"].get(topic, {})
    save()


def activate_file(item, activation_generation=None):
    global file_sessions_dirty
    started = time.perf_counter()
    video = Video(item['path'])
    media_opened = time.perf_counter()
    if not video.topics:
        raise ValueError('该文件没有可用的图像话题，当前数据保持不变')
    saved_sessions = state.get('file_sessions')
    saved_session = (saved_sessions.get(item['path'], {})
                     if isinstance(saved_sessions, dict) else {})
    probe_topic = (saved_session.get('topic')
                   if saved_session.get('topic') in video.topics
                   else next(iter(video.topics)))
    probe_started = time.perf_counter()
    try:
        compatibility = decoded_frame_metadata(video, probe_topic, 0)
    except Exception as error:
        raise ValueError(f'文件首帧兼容检查失败，当前数据保持不变：{str(error)[:300]}')
    probe_finished = time.perf_counter()
    guarded = activation_generation is not None
    if guarded:
        file_activation_intent_lock.acquire()
    try:
        if guarded:
            assert_current_file_activation(activation_generation)
        # Preserve each file's annotations and candidates when switching records.
        sessions = state.setdefault('file_sessions', {})
        fields = ('topic', 'truth', 'truths_by_topic', 'reference_frames', 'candidates', 'candidates_by_topic', 'agent_runs', 'agent_runs_by_topic', 'candidate_revisions', 'candidate_revisions_by_topic')
        if state['video']:
            sessions[str(state['video'].path.resolve())] = {key: state.get(key) for key in fields}
            file_sessions_dirty = True
        previous = sessions.pop(item['path'], {})
        if previous:
            file_sessions_dirty = True
        state.update(video=video, video_path=str(video.path), topic=next(iter(video.topics), None),
                     truth={}, truths_by_topic={}, reference_frames={}, candidates={}, candidates_by_topic={}, agent_runs={}, agent_runs_by_topic={}, candidate_revisions={}, candidate_revisions_by_topic={})
        state.update(previous)
        if state['topic'] not in video.topics:
            state['topic'] = next(iter(video.topics), None)
        session_swapped = time.perf_counter()
        save_timings = save()
        state_saved = time.perf_counter()
        payload = _get_bootstrap_locked()
        payload["data_compatibility"] = {
            **compatibility, "ready": True, "code": "ready",
            "topic": probe_topic,
            "elapsed_ms": round((probe_finished - probe_started) * 1000, 3),
            "scope": "selected_topic_first_frame",
        }
        payload["load_timings_ms"] = {
            "media_open": round((media_opened - started) * 1000, 3),
            "data_probe": round((probe_finished - probe_started) * 1000, 3),
            "session_swap": round((session_swapped - probe_finished) * 1000, 3),
            "state_save": round((state_saved - session_swapped) * 1000, 3),
            **save_timings,
            "bootstrap": round((time.perf_counter() - state_saved) * 1000, 3),
        }
        return payload
    finally:
        if guarded:
            file_activation_intent_lock.release()


@app.get('/api/review_packets')
def get_review_packets():
    """List verified local review packets without serving factory images."""
    packets = []
    rejected = 0
    root = WORK / "review_packets"
    if root.is_dir():
        for folder in sorted(root.iterdir(), key=lambda item: item.name):
            if not folder.is_dir():
                continue
            try:
                packets.append(_review_packet_summary(_read_review_packet(folder.name)))
            except (OSError, ValueError):
                rejected += 1
    return jsonify(schema_version="labelagent.review-packet-index.v1",
                   packets=packets, rejected_packets=rejected,
                   raw_images_served=False, auto_truth_submission=False)


@app.get('/api/review_packets/<packet_id>')
def get_review_packet(packet_id):
    packet = _read_review_packet(packet_id)
    packet["active_file"] = (state.get("video") is not None
                             and str(state["video"].path.resolve()) == packet["mcap_path"])
    packet["active_topic"] = state.get("topic") == packet["topic"]
    packet["ready_to_navigate"] = packet["active_file"] and packet["active_topic"]
    return jsonify(packet)


@app.post('/api/review_packets/<packet_id>/activate')
def activate_review_packet(packet_id):
    """Activate exactly the packet-bound MCAP/topic after a content hash check."""
    activation_generation = claim_file_activation()
    with lock:
        with run_status_lock:
            if run_status.get("status") in {"running", "stopping"}:
                raise ValueError("自动任务运行中，暂不能切换专项复核数据")
        packet = _read_review_packet(packet_id)
        if not packet["mcap_available"]:
            raise ValueError("专项复核包绑定的 MCAP 不存在")
        if _review_packet_file_sha256(packet["mcap_path"]) != packet["mcap_sha256"]:
            raise ValueError("专项复核包绑定的 MCAP 内容哈希不一致")
        current = (str(state["video"].path.resolve()) if state.get("video") else "")
        if current != packet["mcap_path"]:
            payload = activate_file(register_file(packet["mcap_path"]), activation_generation)
        else:
            with file_activation_intent_lock:
                assert_current_file_activation(activation_generation)
            payload = _get_bootstrap_locked()
        _activate_review_packet_topic(packet["topic"])
        payload = _get_bootstrap_locked()
        payload["review_packet"] = packet
        payload["review_packet_activation"] = {
            "mcap_sha256_verified": True,
            "topic_verified": True,
            "raw_images_served": False,
            "human_truth_written": False,
            "training_data_written": False,
        }
        return jsonify(**payload)


@app.get('/api/review_packets/<packet_id>/decisions')
def get_review_packet_decisions(packet_id):
    """Return current scoped decisions without promoting them to frame truth."""
    with lock:
        packet = _read_review_packet(packet_id)
        records = _review_packet_decision_records(packet)
        return jsonify(
            schema_version="labelagent.slope-review-decision-index.v1",
            packet_id=packet["packet_id"],
            manifest_sha256=packet["manifest_sha256"],
            decisions=_current_review_packet_decisions(packet),
            history_count=len(records),
            human_truth_written=False,
            training_data_written=False,
        )


@app.post('/api/review_packets/<packet_id>/decisions')
def submit_review_packet_decision(packet_id):
    """Append an explicit human slope decision bound to the exact source frame.

    This is deliberately narrower than full-frame human truth.  It advances the
    packet review queue but cannot approve training data or make the frame a
    reusable reference.
    """
    body = request.get_json() or {}
    review = _parse_review(body)
    with lock:
        packet = _read_review_packet(packet_id)
        if body.get("expected_manifest_sha256") != packet["manifest_sha256"]:
            raise StaleHumanReviewError("专项复核清单已变化，请刷新后重新确认")
        if (state.get("video") is None
                or str(state["video"].path.resolve()) != packet["mcap_path"]
                or state.get("topic") != packet["topic"]):
            raise StaleHumanReviewError("当前文件或相机不是专项复核包绑定的数据")
        frame = body.get("frame")
        if isinstance(frame, bool) or not isinstance(frame, int):
            raise ValueError("专项复核决定必须包含整数帧号")
        matches = [row for row in packet["rows"] if row["frame"] == frame]
        if len(matches) != 1:
            raise ValueError("专项复核帧不在当前清单中")
        row = matches[0]
        decision = body.get("decision")
        if decision not in row["allowed_human_decisions"]:
            raise ValueError("专项复核决定不在允许集合中")
        expected_frame = body.get("expected_frame_sha256")
        if expected_frame != row["source_frame_sha256"]:
            raise StaleHumanReviewError("专项复核原图绑定已变化")
        frame_sha256 = hashlib.sha256(frame_image(frame)).hexdigest()
        if frame_sha256 != expected_frame:
            raise StaleHumanReviewError("专项复核原图内容已变化")
        slope_geometry = [copy.deepcopy(shape)
                          for shape in state.get("truth", {}).get(str(frame), [])
                          if str(shape.get("label", "")).strip().lower() == "slope"]
        slope_geometry_sha256 = _sha256_json(slope_geometry)
        if body.get("expected_slope_geometry_sha256") != slope_geometry_sha256:
            raise StaleHumanReviewError("专项复核斜坡候选几何已变化")
        if decision.startswith("slope_present") and not slope_geometry:
            raise ValueError("当前帧没有可供复核的斜坡候选轮廓")
        proposal = row.get("assistant_proposal")
        expected_proposal = body.get("expected_assistant_proposal_sha256")
        current_proposal = proposal.get("proposal_sha256") if proposal else None
        if expected_proposal != current_proposal:
            raise StaleHumanReviewError("助手建议已变化，请刷新后重新确认")
        records = _review_packet_decision_records(packet)
        idempotent = [item for item in records if item.get("decision_idempotency_key")
                      == review["decision_idempotency_key"]]
        if len(idempotent) > 1:
            raise RuntimeError("专项复核决定幂等键重复")
        if idempotent:
            existing = idempotent[0]
            expected = {
                "packet_id": packet["packet_id"],
                "manifest_sha256": packet["manifest_sha256"],
                "file": packet["mcap_path"], "topic": packet["topic"],
                "frame": frame, "frame_sha256": frame_sha256,
                "decision": decision,
                "reviewer_declared_id": review["reviewer_declared_id"],
                "started_at": review["started_at"], "note": review["note"],
                "assistant_proposal_sha256": current_proposal,
                "slope_geometry_sha256": slope_geometry_sha256,
            }
            if any(existing.get(key) != value for key, value in expected.items()):
                raise ValueError("该幂等键对应另一项专项复核决定")
            return jsonify(decision_receipt=existing,
                           readback={"verified": True, "idempotent_replay": True},
                           human_truth_written=False, training_data_written=False)
        finished = datetime.now(timezone.utc)
        started = datetime.fromisoformat(review["started_at"].replace("Z", "+00:00"))
        record_body = {
            "schema_version": REVIEW_PACKET_DECISION_SCHEMA,
            "sequence": len(records) + 1,
            "previous_decision_id": records[-1]["decision_id"] if records else None,
            "packet_id": packet["packet_id"],
            "manifest_sha256": packet["manifest_sha256"],
            "file": packet["mcap_path"], "topic": packet["topic"],
            "frame": frame, "frame_sha256": frame_sha256,
            "decision": decision,
            "review_scope": "slope_presence_and_visible_mask_quality",
            "slope_geometry_sha256": slope_geometry_sha256,
            "slope_candidate_count": len(slope_geometry),
            **review,
            "decided_at": finished.isoformat().replace("+00:00", "Z"),
            "elapsed_seconds": round(max(0.0, (finished - started).total_seconds()), 3),
            "assistant_proposal_sha256": current_proposal,
            "assistant_proposal_matched": bool(
                proposal and proposal.get("proposed_decision") == decision),
            "identity_assurance": "self_declared",
            "human_decision": True,
            "human_truth": False,
            "approved_for_training": False,
            "training_return_allowed": False,
        }
        decision_id = _sha256_json(record_body)
        record = {**record_body, "decision_id": decision_id}
        folder = _review_packet_decision_folder(packet["packet_id"])
        folder.mkdir(parents=True, exist_ok=True)
        destination = folder / f"{record['sequence']:06d}-{decision_id}.json"
        temporary = folder / f".{uuid.uuid4().hex}.tmp"
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2,
                                        sort_keys=True), encoding="utf-8")
        _commit_work_file(temporary, destination)
        persisted = _review_packet_decision_records(packet)
        verified = bool(persisted and persisted[-1] == record)
        if not verified:
            raise RuntimeError("专项复核决定写入后读回校验失败")
        return jsonify(
            decision_receipt=record,
            readback={"verified": True, "idempotent_replay": False,
                      "audit_file": destination.name},
            human_truth_written=False,
            training_data_written=False,
        )


@app.get('/api/files')
def get_files():
    with lock:
        if state['video']:
            register_file(state['video'].path)
        files = read_library()
        video = state.get("video")
        return jsonify(files=file_list(files),
                       library_state_sha256=file_library_state_sha256(files),
                       path=str(video.path) if video is not None else "",
                       topic=state.get("topic"),
                       file_activation_generation=current_file_activation_generation())


@app.get('/api/files/annotation-sources')
def get_annotation_sources():
    """List non-current registered files that contain saved annotations.

    This endpoint is discovery-only.  Counts advertise drawing work that can
    be reused, never current human truth.  After switching files, the existing
    human-review gate still decides whether a frame can become a reference.
    """
    with lock:
        current = str(state['video'].path.resolve()) if state.get('video') else None
        sessions = state.get('file_sessions', {})
        sources = []
        for item in read_library():
            path = item.get('path', '')
            if path == current or not Path(path).is_file():
                continue
            session = sessions.get(path)
            if not isinstance(session, dict):
                continue
            by_topic = dict(session.get('truths_by_topic') or {})
            selected_topic = session.get('topic')
            if selected_topic and isinstance(session.get('truth'), dict):
                by_topic[selected_topic] = session['truth']
            frame_count = 0
            shape_count = 0
            topic_count = 0
            source_counts = {"manual": 0, "auto": 0, "unknown": 0}
            for frames in by_topic.values():
                if not isinstance(frames, dict):
                    continue
                valid = [shapes for shapes in frames.values()
                         if isinstance(shapes, list) and shapes]
                if valid:
                    topic_count += 1
                    frame_count += len(valid)
                    shape_count += sum(len(shapes) for shapes in valid)
                    for shapes in valid:
                        counts = _legacy_shape_source_counts(shapes)
                        for key in source_counts:
                            source_counts[key] += counts[key]
            if frame_count:
                sources.append({"id": item['id'],
                                "name": item.get('name') or Path(path).name,
                                "frame_count": frame_count,
                                "shape_count": shape_count,
                                "topic_count": topic_count,
                                "legacy_source_counts": source_counts,
                                "loaded": False,
                                "requires_current_human_review": True})
        sources.sort(key=lambda entry: (-entry['frame_count'], entry['name']))
        return jsonify(sources=sources,
                       current_file_has_separate_controls=True,
                       saved_as_truth=False)


@app.post('/api/files')
def add_file():
    with lock:
        item = register_file(request.get_json()['path'])
        return jsonify(file=item, files=file_list())


@app.post('/api/files/<file_id>/load')
def load_file(file_id):
    activation_generation = claim_file_activation()
    with lock:
        item = next((item for item in read_library() if item['id'] == file_id), None)
        if item is None:
            raise ValueError('文件不在列表中')
        return jsonify(**activate_file(item, activation_generation))


@app.post('/api/candidates/import')
def import_candidates():
    body = request.get_json() or {}
    with lock:
        with run_status_lock:
            if run_status.get("status") in {"running", "stopping"}:
                raise ValueError("自动任务运行中，暂不能导入候选")
        generation, data_path, topic, before = assert_candidate_import_binding(body)
        video = state["video"]
        prepared = prepare_candidate_import(video, body.get("candidates_by_topic"), state.get("labels", []))
        input_sha256 = _sha256_json(body.get("candidates_by_topic"))
        patch = merge_candidate_import(state, prepared)
        previous = {key: state.get(key) for key in patch}
        absent = [key for key in patch if key not in state]
        try:
            state.update(patch)
            save()
        except Exception:
            state.update(previous)
            for key in absent:
                state.pop(key, None)
            raise
        after = candidate_import_binding()
        return jsonify(imported_frames=sum(len(frames) for frames in prepared.values()),
            imported_shapes=sum(len(shapes) for frames in prepared.values() for shapes in frames.values()),
            saved_as_truth=False, input_sha256=input_sha256,
            path=data_path, topic=topic, file_activation_generation=generation,
            applied_binding=before, candidate_import_binding=after)


@app.post('/api/candidates/merge')
def merge_candidates():
    """Merge an explicitly authorized label subset without replacing other candidates."""
    body = request.get_json() or {}
    with lock:
        with run_status_lock:
            if run_status.get("status") in {"running", "stopping"}:
                raise ValueError("自动任务运行中，暂不能合并候选")
        generation, data_path, topic, before = assert_candidate_import_binding(body)
        video = state["video"]
        selected = [str(label).strip() for label in body.get("labels", [])
                    if str(label).strip()]
        if (not selected or len(selected) != len(set(selected))
                or any(label not in state.get("labels", []) for label in selected)):
            raise ValueError("候选合并类别无效")
        prepared = prepare_candidate_import(
            video, body.get("candidates_by_topic"), state.get("labels", []))
        input_sha256 = _sha256_json({
            "labels": selected,
            "candidates_by_topic": body.get("candidates_by_topic"),
        })
        patch = merge_candidate_labels(state, prepared, selected)
        previous = {key: state.get(key) for key in patch}
        absent = [key for key in patch if key not in state]
        try:
            state.update(patch)
            save()
        except Exception:
            state.update(previous)
            for key in absent:
                state.pop(key, None)
            raise
        after = candidate_import_binding()
        return jsonify(merged_frames=sum(len(frames) for frames in prepared.values()),
            merged_shapes=sum(len(shapes) for frames in prepared.values() for shapes in frames.values()),
            labels=selected, preserved_other_labels=True, saved_as_truth=False,
            input_sha256=input_sha256,
            path=data_path, topic=topic, file_activation_generation=generation,
            applied_binding=before, candidate_import_binding=after)


@app.post('/api/candidates/review-proposals')
def register_candidate_review_proposal():
    """Append a bound machine-review suggestion without mutating the candidate layer."""
    body = request.get_json() or {}
    try:
        frame = int(body.get('frame'))
        candidate_id = int(body.get('candidate_id'))
        confidence = float(body.get('confidence'))
    except (TypeError, ValueError):
        raise ValueError('候选复核建议必须包含有效的 frame、candidate_id 和 confidence')
    if isinstance(body.get('confidence'), bool) or not 0 <= confidence <= 1:
        raise ValueError('候选复核置信度必须在 0 到 1 之间')
    topic = body.get('topic')
    from_label = str(body.get('from_label') or '').strip()
    proposed_label = str(body.get('proposed_label') or '').strip()
    reason = str(body.get('reason') or '').strip()
    evidence_path = str(body.get('evidence_path') or '').strip()
    evidence_sha = body.get('evidence_sha256')
    expected_artifact_sha = body.get('expected_candidate_artifact_sha256')
    idempotency_key = body.get('idempotency_key')
    source = body.get('source', 'assistant_reviewed_isolated_mask')
    if not isinstance(topic, str) or not topic:
        raise ValueError('候选复核建议必须绑定 topic')
    if not from_label or not proposed_label or from_label == proposed_label:
        raise ValueError('候选复核建议必须包含不同的原类别和建议类别')
    if not reason or len(reason) > 2000:
        raise ValueError('候选复核建议必须包含不超过 2000 字的判断依据')
    if len(evidence_path) > 1000:
        raise ValueError('候选复核证据路径过长')
    if not _valid_sha256(evidence_sha):
        raise ValueError('候选复核证据必须绑定有效 SHA-256')
    if not _valid_sha256(expected_artifact_sha):
        raise ValueError('候选复核建议必须绑定当前候选 artifact SHA-256')
    if not _valid_sha256(idempotency_key):
        raise ValueError('候选复核建议必须包含有效幂等键')
    if source != 'assistant_reviewed_isolated_mask':
        raise ValueError('候选复核建议来源必须是 assistant_reviewed_isolated_mask')

    immutable_request = {
        'topic': topic, 'frame': frame, 'candidate_id': candidate_id,
        'from_label': from_label, 'proposed_label': proposed_label,
        'confidence': confidence, 'reason': reason,
        'evidence_path': evidence_path, 'evidence_sha256': evidence_sha,
        'expected_candidate_artifact_sha256': expected_artifact_sha,
        'source': source,
    }
    with lock:
        ledger = state.setdefault('candidate_review_proposals', [])
        existing = next((item for item in ledger
                         if item.get('idempotency_key') == idempotency_key), None)
        if existing is not None:
            recorded_request = {key: existing.get(key) for key in immutable_request}
            if recorded_request != immutable_request:
                raise ValueError('同一幂等键对应的候选复核建议内容不同，已拒绝覆盖')
            return jsonify(proposal=existing, idempotent_replay=True,
                           candidate_mutated=False, saved_as_truth=False)

        if state.get('video') is None or state.get('topic') is None:
            raise ValueError('请先加载与复核建议对应的数据')
        if topic != state['topic']:
            raise StaleHumanReviewError('图像源已变化，请重新生成候选复核建议')
        if not 0 <= frame < state['video'].topics[topic]['count']:
            raise ValueError('候选复核帧号无效')
        if str(frame) in state.setdefault('truth', {}):
            raise ValueError('该帧已经存在人工真值，机器建议不能覆盖或回写')
        candidates = state.setdefault('candidates', {}).get(str(frame))
        if not isinstance(candidates, list) or not candidates:
            raise StaleHumanReviewError('当前帧机器候选已不存在，请重新复核')
        actual_artifact_sha = _sha256_json(candidates)
        run = state.setdefault('agent_runs', {}).get(str(frame)) or {}
        declared_artifact_sha = run.get('candidate_artifact_sha256')
        if declared_artifact_sha is not None and not _valid_sha256(declared_artifact_sha):
            raise RuntimeError('候选运行回执中的 artifact SHA-256 无效，已停止登记')
        if expected_artifact_sha != actual_artifact_sha:
            raise StaleHumanReviewError('候选已在复核后变化，请重新读取当前候选')
        matches = [item for item in candidates if item.get('id') == candidate_id]
        if len(matches) != 1:
            raise ValueError('候选 ID 不存在或不唯一，已停止登记')
        candidate = matches[0]
        if candidate.get('label') != from_label:
            raise StaleHumanReviewError('候选类别已在复核后变化，请重新检查')
        if proposed_label not in state.get('labels', []):
            raise ValueError('建议类别不在当前标签注册表中，必须进入 ontology 人工门禁')
        candidate_sha = _sha256_json(candidate)
        identity = {
            'file': str(state['video'].path), 'topic': topic, 'frame': frame,
            'candidate_id': candidate_id, 'candidate_sha256': candidate_sha,
            'candidate_artifact_sha256': actual_artifact_sha,
            'from_label': from_label, 'proposed_label': proposed_label,
            'evidence_sha256': evidence_sha, 'idempotency_key': idempotency_key,
        }
        record = {
            'schema_version': 'labelagent.candidate_review_proposal.v1',
            'proposal_id': _sha256_json(identity),
            'idempotency_key': idempotency_key,
            'recorded_at_unix': time.time(),
            'file': identity['file'], 'topic': topic, 'frame': frame,
            'candidate_id': candidate_id, 'candidate_sha256': candidate_sha,
            'candidate_artifact_sha256': actual_artifact_sha,
            'declared_candidate_artifact_sha256': declared_artifact_sha,
            'declared_artifact_integrity_match': (declared_artifact_sha is None
                                                   or declared_artifact_sha == actual_artifact_sha),
            'expected_candidate_artifact_sha256': expected_artifact_sha,
            'from_label': from_label, 'proposed_label': proposed_label,
            'confidence': confidence, 'reason': reason,
            'evidence_path': evidence_path, 'evidence_sha256': evidence_sha,
            'source': source, 'route': 'human_review_required',
            'candidate_only': True, 'candidate_mutated': False,
            'human_truth': False, 'approved_for_training': False,
        }
        previous = copy.deepcopy(ledger)
        try:
            ledger.append(record)
            save()
        except Exception:
            state['candidate_review_proposals'] = previous
            raise
        return jsonify(proposal=record, idempotent_replay=False,
                       candidate_mutated=False, saved_as_truth=False)


@app.delete('/api/files/<file_id>')
def remove_file(file_id):
    global file_sessions_dirty
    body = request.get_json(silent=True) or {}
    with lock:
        activation_generation, expected_path, expected_topic = assert_request_workspace_scope(
            body, generation_key="expected_file_activation_generation",
            path_key="expected_path", topic_key="expected_topic")
        files = read_library()
        expected_library_sha = body.get("expected_library_state_sha256")
        if not _valid_sha256(expected_library_sha):
            raise ValueError("移除文件必须绑定当前文件列表 SHA-256")
        current_library_sha = file_library_state_sha256(files)
        if expected_library_sha != current_library_sha:
            raise StaleHumanReviewError("文件列表已在另一窗口变化，请刷新后重新操作")
        item = next((item for item in files if item['id'] == file_id), None)
        if item is None:
            raise ValueError('文件不在列表中')
        expected_entry_sha = body.get("expected_file_entry_sha256")
        if not _valid_sha256(expected_entry_sha):
            raise ValueError("移除文件必须绑定目标条目 SHA-256")
        current_entry_sha = _sha256_json(item)
        if expected_entry_sha != current_entry_sha:
            raise StaleHumanReviewError("目标文件条目已变化，请刷新后重新操作")
        expected_loaded = body.get("expected_loaded")
        if type(expected_loaded) is not bool:
            raise ValueError("expected_loaded 必须是文件列表中的布尔状态")
        current = str(state['video'].path.resolve()) if state.get('video') else None
        actually_loaded = item['path'] == current
        if expected_loaded != actually_loaded:
            raise StaleHumanReviewError("目标文件的加载状态已变化，请刷新后重新操作")
        remaining = [entry for entry in files if entry['id'] != file_id]
        write_library(remaining)
        unloaded = state['video'] is not None and str(state['video'].path.resolve()) == item['path']
        if unloaded:
            sessions = state.setdefault('file_sessions', {})
            sessions[item['path']] = {key: state.get(key) for key in ('topic', 'truth', 'truths_by_topic', 'reference_frames', 'candidates', 'candidates_by_topic', 'agent_runs', 'agent_runs_by_topic', 'candidate_revisions', 'candidate_revisions_by_topic')}
            file_sessions_dirty = True
            state.update(video=None, video_path=None, topic=None, truth={}, truths_by_topic={}, reference_frames={}, candidates={}, candidates_by_topic={}, agent_runs={}, agent_runs_by_topic={}, candidate_revisions={}, candidate_revisions_by_topic={})
            save()
        response_path = "" if unloaded else expected_path
        response_topic = None if unloaded else expected_topic
        return jsonify(files=file_list(remaining), unloaded=unloaded,
                       library_state_sha256=file_library_state_sha256(remaining),
                       removed_library_state_sha256=current_library_sha,
                       removed_file_entry_sha256=current_entry_sha,
                       removed_file_id=file_id, disk_file_deleted=False,
                       path=response_path, topic=response_topic,
                       previous_path=expected_path, previous_topic=expected_topic,
                       file_activation_generation=activation_generation)


@app.post("/api/open")
def open_video():
    activation_generation = claim_file_activation()
    with lock:
        return jsonify(**activate_file(
            register_file(request.get_json()['path']), activation_generation))


MAX_BATCH_UPLOADS = 100


def register_uploaded_files(uploaded_files):
    uploaded_files = list(uploaded_files)
    if not 1 <= len(uploaded_files) <= MAX_BATCH_UPLOADS:
        raise ValueError(f"每批请选择 1～{MAX_BATCH_UPLOADS} 个文件")
    destination = WORK / "uploads"
    destination.mkdir(exist_ok=True)
    prepared = []
    saved_paths = []
    try:
        for uploaded in uploaded_files:
            suffix = Path(uploaded.filename or "").suffix.lower()
            if not uploaded.filename or suffix not in SUPPORTED_SUFFIXES:
                raise ValueError("请选择 JPG、JPEG、PNG 图片或 MCAP 文件")
            filename = (secure_filename(Path(uploaded.filename).stem) or "image") + suffix
            path = (destination / f"{uuid.uuid4().hex}-{filename}").resolve()
            uploaded.save(path)
            saved_paths.append(path)
            if suffix in IMAGE_SUFFIXES:
                read_image(path)
            prepared.append((path, uploaded.filename))
        with lock:
            files = read_library()
            items = []
            for path, original_name in prepared:
                item = dict(id=uuid.uuid4().hex, path=str(path), name=original_name,
                            size=path.stat().st_size)
                files.append(item)
                items.append(item)
            write_library(files)
            return items, file_list()
    except Exception:
        for path in saved_paths:
            path.unlink(missing_ok=True)
        raise


@app.post("/api/upload")
def upload_video():
    uploaded = request.files.get("file")
    items, files = register_uploaded_files([uploaded] if uploaded is not None else [])
    return jsonify(file=items[0], files=files)


@app.post("/api/uploads")
def upload_videos():
    items, files = register_uploaded_files(request.files.getlist("files"))
    return jsonify(added=items, files=files)


@app.post("/api/topic")
def set_topic():
    body = request.get_json()
    with lock:
        if state["video"] is None:
            raise ValueError("请先打开 MCAP")
        if (body.get("compact") is True
                and body.get("path") != str(state["video"].path)):
            raise StaleHumanReviewError(
                "数据文件已在另一个窗口切换，旧相机请求已拒绝")
        if (body.get("compact") is True
                and body.get("file_activation_generation")
                != current_file_activation_generation()):
            raise StaleHumanReviewError(
                "数据文件已被重新载入，旧窗口相机请求已拒绝")
        if body["topic"] not in state["video"].topics:
            raise ValueError("话题不存在")
        if body["topic"] != state["topic"]:
            state.setdefault("truths_by_topic", {})[state["topic"]] = state["truth"]
            state.setdefault("candidates_by_topic", {})[state["topic"]] = state.get("candidates", {})
            state.setdefault("agent_runs_by_topic", {})[state["topic"]] = state.get("agent_runs", {})
            state.setdefault("candidate_revisions_by_topic", {})[state["topic"]] = state.get("candidate_revisions", {})
            state["topic"] = body["topic"]
            state["truth"] = state["truths_by_topic"].get(state["topic"], {})
            state["candidates"] = state["candidates_by_topic"].get(state["topic"], {})
            state["agent_runs"] = state["agent_runs_by_topic"].get(state["topic"], {})
            state["candidate_revisions"] = state["candidate_revisions_by_topic"].get(state["topic"], {})
            save()
        topic_references = state.get("reference_frames", {}).get(state["topic"], [])
        description = (load_reference_description(topic_references[0])
                       if topic_references and state.get("truth", {}).get(str(topic_references[0]))
                       else None)
        if body.get("compact") is True:
            if type(body.get("frame")) is not int:
                raise ValueError("按需切换相机必须包含整数 frame")
            count = state["video"].topics[state["topic"]]["count"]
            if not 0 <= body["frame"] < count:
                raise ValueError("帧号无效")
            return jsonify(compact_topic_payload(body["frame"], description))
        label_state = current_inference_label_state()
        return jsonify(topic=state["topic"], truth=state["truth"],
                       truth_review_status=current_truth_review_status(),
                       candidates=state["candidates"], agent_runs=state["agent_runs"],
                       candidate_revision_summary=candidate_revision_summary(),
                       candidate_review_proposals=state.get("candidate_review_proposals", []),
                       reference_frames=state.get("reference_frames", {}),
                       labels=label_state["labels"],
                       auto_labels=label_state["auto_labels"],
                       reference_only_labels=label_state["reference_only_labels"],
                       label_state_sha256=label_state_sha256(label_state),
                       description=description,
                       description_cache_hit=description is not None)


@app.post("/api/labels")
def set_labels():
    body = request.get_json() or {}
    with lock:
        generation, path, topic, _ = assert_request_label_state(body)
        labels = [x.strip() for x in body["labels"] if x.strip()]
        if len(set(labels)) != len(labels) or not labels:
            raise ValueError("类别不能为空或重复")
        state["labels"] = labels
        if state.get("auto_labels"):
            state["auto_labels"] = [label for label in labels if label in state["auto_labels"]]
        save()
        label_state = current_inference_label_state()
        return jsonify(**label_state,
                       label_state_sha256=label_state_sha256(label_state),
                       path=path, topic=topic,
                       file_activation_generation=generation,
                       description_invalidated=True)


@app.post("/api/auto_labels")
def set_auto_labels():
    body = request.get_json() or {}
    with lock:
        generation, path, topic, _ = assert_request_label_state(body)
        labels = [str(label).strip() for label in body.get("labels", []) if str(label).strip()]
        if not labels or len(labels) != len(set(labels)) or any(label not in state["labels"] for label in labels):
            raise ValueError("请选择有效且不重复的推理类别")
        state["auto_labels"] = [label for label in state["labels"] if label in labels]
        save()
        label_state = current_inference_label_state()
        return jsonify(labels=label_state["auto_labels"],
                       project_labels=label_state["labels"],
                       auto_labels=label_state["auto_labels"],
                       reference_only_labels=label_state["reference_only_labels"],
                       label_state_sha256=label_state_sha256(label_state),
                       path=path, topic=topic,
                       file_activation_generation=generation,
                       description_invalidated=True)


@app.post("/api/truth")
def set_truth():
    body = request.get_json() or {}
    with lock:
        assert_request_file_activation(
            body, "expected_file_activation_generation")
        recover_pending_human_reviews()
        review = _parse_review(body)
        index = int(body["frame"])
        frame_data = frame_image(index)
        frame_sha = hashlib.sha256(frame_data).hexdigest()
        width, height = Image.open(io.BytesIO(frame_data)).size
        shapes = body.get("shapes")
        if not isinstance(shapes, list):
            raise ValueError("真值 shapes 必须是数组")
        for shape in shapes:
            if shape["label"] not in state["labels"] or len(shape["points"]) < 3:
                raise ValueError("真值类别或多边形无效")
            if any(not (0 <= x < width and 0 <= y < height) for x, y in shape["points"]):
                raise ValueError("坐标超出图像范围")
        confirmed_empty = body.get("confirmed_empty") is True
        requested_decision = body.get("decision")
        if not shapes and (not confirmed_empty or requested_decision != "confirmed_empty"):
            raise ValueError("空真值必须显式设置 decision=confirmed_empty 且 confirmed_empty=true")
        if shapes and (body.get("confirmed_empty") is True or requested_decision is not None):
            raise ValueError("非空真值不能设置 confirmed_empty 或客户端 decision")
        candidate, candidate_origin, parent_candidate_sha = _resolve_review_candidate(body, index)
        candidate_sha = _sha256_json(candidate) if candidate is not None else None
        existing = _review_by_idempotency_key(review["decision_idempotency_key"])
        if existing is not None:
            replay_has_candidate = candidate is not None or "draft_candidate" in body
            replay_candidate_sha = candidate_sha if replay_has_candidate else existing.get("candidate_sha256")
            replay_origin = (candidate_origin if replay_has_candidate else
                existing.get("candidate_origin", "server_state" if replay_candidate_sha else "none"))
            replay_parent_sha = (parent_candidate_sha if replay_has_candidate else
                existing.get("parent_candidate_sha256"))
            expected_frame, expected_candidate, review_scope = _assert_review_binding(body, frame=index,
                frame_sha256=frame_sha,
                candidate_sha256=replay_candidate_sha)
            _assert_idempotent_review(existing, review=review, frame=index,
                frame_sha256=frame_sha,
                allowed_decisions={"manual_annotation", "accept_candidate",
                                   "correct_candidate", "confirmed_empty"},
                review_scope=review_scope, truth=shapes,
                candidate=candidate if replay_has_candidate else None,
                candidate_origin=replay_origin,
                parent_candidate_sha256=replay_parent_sha)
            saved_truth = state.setdefault("truth", {}).get(str(index))
            if saved_truth is None or _sha256_json(saved_truth) != existing.get("truth_sha256"):
                raise RuntimeError("幂等复放找到审计记录，但持久化真值不匹配")
            readback = _review_commit_readback(existing)
            if not readback["verified"]:
                raise RuntimeError("幂等复放的服务端读回验证失败")
            return jsonify(saved=len(shapes), review=existing, readback=readback,
                           idempotent_replay=True)
        expected_frame, expected_candidate, review_scope = _assert_review_binding(body, frame=index,
            frame_sha256=frame_sha, candidate_sha256=candidate_sha)
        if _contains_unknown_label(candidate) or _contains_unknown_label(shapes):
            raise ValueError("未知或新类别候选只能进入 ontology 人工复核，不能直接保存为生产真值")
        status_key = _review_status_key(index)
        review_status = state.setdefault("candidate_review_status", {}).get(status_key)
        if review_status and review_status.get("status") == "rejected":
            remediation = review.get("remediation_of_review_id")
            if remediation != review_status.get("review_id"):
                raise ValueError("该候选已被驳回；接受修正结果时必须绑定 remediation_of_review_id")
            if candidate is None:
                raise ValueError("已驳回候选已失活；必须先生成新的修正版候选再复核")
            if candidate_sha == review_status.get("candidate_sha256"):
                raise ValueError("已驳回的同一候选不能重新激活；请生成不同哈希的修正版候选")
        if confirmed_empty:
            decision = "confirmed_empty"
        elif candidate is None:
            decision = "manual_annotation"
        elif _sha256_json(candidate) == _sha256_json(shapes):
            decision = "accept_candidate"
        else:
            decision = "correct_candidate"
        audit = prepare_human_review(frame=index, decision=decision, review=review,
            frame_sha256=frame_sha, expected_frame_sha256=expected_frame,
            expected_candidate_sha256=expected_candidate,
            review_scope=review_scope, candidate=candidate, truth=shapes,
            confirmed_empty=confirmed_empty, candidate_origin=candidate_origin,
            parent_candidate_sha256=parent_candidate_sha)
        def mutation():
            state.setdefault("truth", {})[str(index)] = shapes
            state.setdefault("candidates", {}).pop(str(index), None)
            if candidate is not None:
                state.setdefault("candidate_review_status", {})[status_key] = {
                    "status": "accepted", "remediation_required": False,
                    "route": ("truth_committed_pending_closed_loop_sync"
                              if decision == "accept_candidate"
                              else "candidate_final_regeneration_required"),
                    "review_id": audit["review_id"],
                    "candidate_sha256": audit["candidate_sha256"],
                    "candidate_event_id": audit["candidate_event_id"],
                    "decision": decision, "file": audit["file"],
                    "topic": audit["topic"], "frame": index}
        commit_human_review(audit, mutation)
        readback = _review_commit_readback(audit)
        if not readback["verified"]:
            raise RuntimeError("真值提交后的服务端读回验证失败")
        return jsonify(saved=len(shapes), review=audit, readback=readback,
                       idempotent_replay=False)


@app.post("/api/human_review_binding")
def get_human_review_binding():
    body = request.get_json() or {}
    with lock:
        recover_pending_human_reviews()
        index = int(body.get("frame", -1))
        if state["video"] is None or state["topic"] is None:
            raise ValueError("请先加载图像消息")
        activation_generation = assert_request_file_activation(body)
        if body.get("topic") != state["topic"]:
            raise StaleHumanReviewError("图像源已变化，请刷新后重新复核")
        if not 0 <= index < state["video"].topics[state["topic"]]["count"]:
            raise ValueError("帧号无效")
        frame_sha = hashlib.sha256(frame_image(index)).hexdigest()
        candidate, candidate_origin, parent_candidate_sha = _resolve_review_candidate(body, index)
        candidate_sha = _sha256_json(candidate) if candidate is not None else None
        scope = _review_scope(frame_sha, index)
        review_status = state.setdefault("candidate_review_status", {}).get(
            _review_status_key(index))
        return jsonify(frame=index, topic=state["topic"],
                       file_activation_generation=activation_generation,
                       frame_sha256=frame_sha, candidate_sha256=candidate_sha,
                       dataset_id=scope["dataset_id"],
                       dataset_version=scope["dataset_version"], camera=scope["camera"],
                       dataset_split=scope["dataset_split"],
                       split_policy=scope["split_policy"],
                       training_return_allowed=scope["training_return_allowed"],
                       candidate_origin=candidate_origin,
                       parent_candidate_sha256=parent_candidate_sha,
                       review_status=review_status)


@app.post("/api/human_review")
def reject_candidate():
    body = request.get_json() or {}
    with lock:
        assert_request_file_activation(
            body, "expected_file_activation_generation")
        recover_pending_human_reviews()
        review = _parse_review(body)
        index = int(body.get("frame", -1))
        if state["video"] is None or state["topic"] is None:
            raise ValueError("请先加载图像消息")
        if not 0 <= index < state["video"].topics[state["topic"]]["count"]:
            raise ValueError("帧号无效")
        frame_data = frame_image(index)
        frame_sha = hashlib.sha256(frame_data).hexdigest()
        candidate, candidate_origin, parent_candidate_sha = _resolve_review_candidate(body, index)
        candidate_sha = _sha256_json(candidate) if candidate is not None else None
        existing = _review_by_idempotency_key(review["decision_idempotency_key"])
        if existing is not None:
            replay_has_candidate = candidate is not None or "draft_candidate" in body
            replay_candidate_sha = candidate_sha if replay_has_candidate else existing.get("candidate_sha256")
            replay_origin = (candidate_origin if replay_has_candidate else
                existing.get("candidate_origin", "server_state" if replay_candidate_sha else "none"))
            replay_parent_sha = (parent_candidate_sha if replay_has_candidate else
                existing.get("parent_candidate_sha256"))
            expected_frame, expected_candidate, review_scope = _assert_review_binding(body, frame=index,
                frame_sha256=frame_sha,
                candidate_sha256=replay_candidate_sha)
            _assert_idempotent_review(existing, review=review, frame=index,
                frame_sha256=frame_sha, allowed_decisions={"reject_candidate"},
                review_scope=review_scope,
                candidate=candidate if replay_has_candidate else None,
                candidate_origin=replay_origin,
                parent_candidate_sha256=replay_parent_sha)
            readback = _review_commit_readback(existing)
            if not readback["verified"]:
                raise RuntimeError("驳回幂等复放的服务端读回验证失败")
            return jsonify(rejected=True, review=existing, idempotent_replay=True,
                           remediation_required=True, readback=readback)
        if candidate is None:
            raise ValueError("当前帧没有可驳回的自动候选")
        status_key = _review_status_key(index)
        current_status = state.setdefault("candidate_review_status", {}).get(status_key)
        expected_frame, expected_candidate, review_scope = _assert_review_binding(body, frame=index,
            frame_sha256=frame_sha, candidate_sha256=candidate_sha)
        if (current_status and current_status.get("status") == "rejected"
                and current_status.get("candidate_sha256") == candidate_sha):
            raise ValueError("该候选已被驳回；请复用原幂等键，或先完成修正后再复核")
        audit = prepare_human_review(frame=index, decision="reject_candidate", review=review,
            frame_sha256=frame_sha, expected_frame_sha256=expected_frame,
            expected_candidate_sha256=expected_candidate,
            review_scope=review_scope, candidate=candidate,
            candidate_origin=candidate_origin,
            parent_candidate_sha256=parent_candidate_sha)
        def mutation():
            state.setdefault("rejected_candidates", {})[audit["review_id"]] = {
                "file": audit["file"], "topic": audit["topic"], "frame": index,
                "candidate_sha256": audit["candidate_sha256"], "candidate": candidate}
            state.setdefault("candidates", {}).pop(str(index), None)
            state.setdefault("candidate_review_status", {})[status_key] = {
                "status": "rejected", "remediation_required": True,
                "route": "agent_remediation_required", "review_id": audit["review_id"],
                "candidate_sha256": audit["candidate_sha256"],
                "candidate_event_id": audit["candidate_event_id"],
                "decision": "reject_candidate", "file": audit["file"],
                "topic": audit["topic"], "frame": index}
        commit_human_review(audit, mutation)
        readback = _review_commit_readback(audit)
        if not readback["verified"]:
            raise RuntimeError("候选驳回后的服务端读回验证失败")
        return jsonify(rejected=True, review=audit, idempotent_replay=False,
                       remediation_required=True, readback=readback)


@app.get("/api/human_reviews")
def get_human_reviews():
    with lock:
        recover_pending_human_reviews()
        if state["video"] is None or state["topic"] is None:
            return jsonify(reviews=[])
        frame_value = request.args.get("frame")
        frame_number = int(frame_value) if frame_value is not None else None
        records = human_review_records(video_file=str(state["video"].path),
            topic=request.args.get("topic", state["topic"]), frame=frame_number)
        return jsonify(reviews=records)


@app.delete("/api/auto_result/<int:index>")
def clear_auto_result(index):
    """Remove one frame's automatic candidate and audit; never touch truth."""
    body = request.get_json(silent=True) or {}
    with lock:
        if state["video"] is None or state["topic"] is None:
            raise ValueError("请先加载图像消息")
        activation_generation, data_path, topic = assert_request_data_scope(body)
        if not 0 <= index < state["video"].topics[state["topic"]]["count"]:
            raise ValueError("帧号无效")
        binding = auto_result_clear_binding(index)
        for field in ("candidate_artifact_sha256", "agent_run_sha256"):
            expected_key = "expected_" + field
            if expected_key not in body:
                raise ValueError(f"清除机器结果必须包含 {expected_key}")
            expected = body.get(expected_key)
            if expected is not None and not _valid_sha256(expected):
                raise ValueError(f"{expected_key} 必须是 SHA-256 或 null")
            if expected != binding[field]:
                raise StaleHumanReviewError(
                    "机器候选或过程记录已变化，旧窗口清除请求已拒绝")
        with run_status_lock:
            if run_status.get("status") in ("running", "stopping"):
                raise RuntimeError("自动标注仍在运行，请任务结束后再清除")
        key = str(index)
        removed_candidate = state.setdefault("candidates", {}).pop(key, None) is not None
        removed_audit = state.setdefault("agent_runs", {}).pop(key, None) is not None
        save()
        video_file, selected_topic = str(state["video"].path), state["topic"]
        def belongs(record):
            return record.get("file") == video_file and record.get("topic") == selected_topic
        def keep_call(call):
            return not (belongs(call) and call.get("frame") == index
                        and call.get("purpose") != "生成真值描述")
        removed_images = set()
        remaining_frames = None
        for filename in ("last_run.json", "model_debug.json"):
            path = WORK / filename
            if not path.exists():
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            scoped_run = belongs(record) or any(belongs(call) for call in record.get("trace", []))
            if filename == "last_run.json" and not scoped_run:
                continue
            removed_images.update(inspection_image_names(
                [call for call in record.get("trace", []) if not keep_call(call)]))
            record["trace"] = [call for call in record.get("trace", []) if keep_call(call)]
            if filename == "last_run.json":
                if scoped_run:
                    removed_images.update(inspection_image_names(
                        [step for step in record.get("steps", []) if step.get("frame") == index]))
                    record["steps"] = [step for step in record.get("steps", [])
                                       if step.get("frame") != index]
                    record["frames"] = [item for item in record.get("frames", [])
                                        if item.get("frame") != index]
                    remaining_frames = len(record["frames"])
                    if (record.get("current") or {}).get("frame") == index:
                        record["current"] = None
            if filename == "last_run.json":
                save_run(record)
            else:
                temporary = path.with_suffix(".tmp")
                _write_json(temporary, record, fast=_fast_json_write_allowed(path))
                _commit_work_file(temporary, path)
        with run_status_lock:
            if belongs(run_status):
                run_status["events"] = [event for event in run_status.get("events", [])
                                        if event.get("frame") != index]
                if remaining_frames is not None:
                    run_status["completed"] = remaining_frames
                    run_status["total"] = remaining_frames
                    if not remaining_frames:
                        _update_run_status_locked(status="idle", frame=None, phase=None, round=0,
                                                  shape_count=0, error=None)
        if removed_images:
            still_used = set()
            for filename in ("last_run.json", "model_debug.json", "description_debug.json"):
                path = WORK / filename
                if path.exists():
                    still_used.update(inspection_image_names(json.loads(path.read_text(encoding="utf-8"))))
            for name in removed_images - still_used:
                (WORK / "inspection" / name).unlink(missing_ok=True)
        return jsonify(frame=index, candidate_removed=removed_candidate,
                       audit_removed=removed_audit,
                       truth_preserved=key in state["truth"],
                       removed_candidate_artifact_sha256=(
                           binding["candidate_artifact_sha256"]),
                       removed_agent_run_sha256=binding["agent_run_sha256"],
                       path=data_path, topic=topic,
                       file_activation_generation=activation_generation)


@app.post("/api/candidates/pallet_cleanup/<int:index>")
def cleanup_pallet_candidates(index):
    """Refresh legacy pallet geometry without touching truth or training state."""
    body = request.get_json(silent=True) or {}
    with lock:
        if state["video"] is None or state["topic"] is None:
            raise ValueError("请先加载图像消息")
        activation_generation, data_path, topic = assert_request_data_scope(body)
        if not 0 <= index < state["video"].topics[state["topic"]]["count"]:
            raise ValueError("帧号无效")
        with run_status_lock:
            if run_status.get("status") in ("running", "stopping"):
                raise RuntimeError("自动标注仍在运行，请任务结束后再优化栈板边界")
        key = str(index)
        current_candidates = copy.deepcopy(state.get("candidates", {}).get(key) or [])
        current_run = copy.deepcopy(state.get("agent_runs", {}).get(key) or {})
        if not current_candidates:
            raise ValueError("当前帧没有机器候选")
        binding = auto_result_clear_binding(index)
        for field in ("candidate_artifact_sha256", "agent_run_sha256"):
            expected_key = "expected_" + field
            if expected_key not in body:
                raise ValueError(f"栈板边界优化必须包含 {expected_key}")
            expected = body.get(expected_key)
            if expected is not None and not _valid_sha256(expected):
                raise ValueError(f"{expected_key} 必须是 SHA-256 或 null")
            if expected != binding[field]:
                raise StaleHumanReviewError(
                    "机器候选或过程记录已变化，旧窗口栈板边界优化请求已拒绝")

        frame_image = state["video"].frame(state["topic"], index)
        with Image.open(io.BytesIO(frame_image)) as decoded:
            width, height = decoded.size
            frame_rgb = np.asarray(decoded.convert("RGB"))
        already_adjusted = {
            item.get("shape_id") for item in current_run.get(
                "candidate_geometry_adjustments", [])
            if isinstance(item, dict)
            and item.get("reason_code") == "pallet_texture_geometry_cleanup"
            and "boundary_simplification_applied" in item
            and int(item.get("geometry_cleanup_version") or 1) >= 5
            and type(item.get("shape_id")) is int
        }
        refreshed, adjustments = [], []
        for candidate in current_candidates:
            if candidate.get("id") in already_adjusted:
                refined, adjustment = candidate, None
            else:
                refined, adjustment = refine_pallet_candidate_geometry(
                    candidate, width, height, image_rgb=frame_rgb)
            refreshed.append(refined)
            if adjustment is not None:
                adjustment = copy.deepcopy(adjustment)
                adjustment.update(
                    source="legacy_candidate_geometry_refresh",
                    candidate_only=True, human_truth=False,
                    approved_for_training=False, approved_for_deployment=False)
                adjustments.append(adjustment)

        if not adjustments:
            return jsonify(
                frame=index, changed=False, adjusted_candidate_count=0,
                candidate_only=True, human_truth=False,
                approved_for_training=False, approved_for_deployment=False,
                truth_preserved=key in state.get("truth", {}),
                prior_candidate_artifact_sha256=binding["candidate_artifact_sha256"],
                prior_agent_run_sha256=binding["agent_run_sha256"],
                candidate_artifact_sha256=binding["candidate_artifact_sha256"],
                agent_run_sha256=binding["agent_run_sha256"],
                path=data_path, topic=topic,
                file_activation_generation=activation_generation)

        new_artifact_sha = _sha256_json(refreshed)
        high_risk_total = sum(candidate_requires_class_audit(item) for item in refreshed)
        replacement_run = copy.deepcopy(current_run)
        replacement_run.update({
            "candidate_artifact_sha256": new_artifact_sha,
            "input_sha256": hashlib.sha256(frame_image).hexdigest(),
            "candidate_geometry_adjustments": (
                copy.deepcopy(current_run.get("candidate_geometry_adjustments", []))
                + adjustments),
            "candidate_geometry_cleanup_refresh": {
                "schema_version": "labelagent.pallet_geometry_refresh.v5",
                "adjusted_candidate_count": len(adjustments),
                "prior_candidate_artifact_sha256": binding["candidate_artifact_sha256"],
                "candidate_artifact_sha256": new_artifact_sha,
                "candidate_only": True, "human_truth": False,
                "approved_for_training": False, "approved_for_deployment": False,
            },
            "candidate_class_audits": [],
            "candidate_class_audit_status": "pending" if high_risk_total else "complete",
            "candidate_class_audit_deferred": bool(high_risk_total),
            "candidate_class_audit_total": high_risk_total,
            "candidate_class_audit_cache_hit": False,
            "review_queue": [
                item for item in current_run.get("review_queue", [])
                if item.get("source") != "candidate_class_audit"],
            "candidate_only": True, "human_truth": False,
            "approved_for_training": False, "approved_for_deployment": False,
            "human_review_required": True,
        })
        previous_revisions = copy.deepcopy(state.get("candidate_revisions", {}).get(key))
        previous_review_status = copy.deepcopy(
            state.get("candidate_review_status", {}).get(key))
        try:
            archive_candidate_revision(
                index, topic, current_candidates, current_run,
                refreshed, replacement_run, reason="pallet_geometry_cleanup_refresh")
            state.setdefault("candidates", {})[key] = refreshed
            state.setdefault("agent_runs", {})[key] = replacement_run
            state.setdefault("candidate_review_status", {})[key] = {}
            save()
        except Exception:
            state.setdefault("candidates", {})[key] = current_candidates
            state.setdefault("agent_runs", {})[key] = current_run
            if previous_revisions is None:
                state.setdefault("candidate_revisions", {}).pop(key, None)
            else:
                state.setdefault("candidate_revisions", {})[key] = previous_revisions
            if previous_review_status is None:
                state.setdefault("candidate_review_status", {}).pop(key, None)
            else:
                state.setdefault("candidate_review_status", {})[key] = previous_review_status
            raise
        return jsonify(
            frame=index, changed=True,
            adjusted_candidate_count=len(adjustments),
            changed_pixels=sum(item["changed_pixels"] for item in adjustments),
            candidate_only=True, human_truth=False,
            approved_for_training=False, approved_for_deployment=False,
            truth_preserved=key in state.get("truth", {}),
            prior_candidate_artifact_sha256=binding["candidate_artifact_sha256"],
            prior_agent_run_sha256=binding["agent_run_sha256"],
            candidate_artifact_sha256=new_artifact_sha,
            agent_run_sha256=_sha256_json(replacement_run),
            candidate_class_audit_status=replacement_run["candidate_class_audit_status"],
            path=data_path, topic=topic,
            file_activation_generation=activation_generation)

@app.post("/api/reference_frame")
def set_reference_frame():
    body = request.get_json() or {}
    with lock:
        activation_generation = assert_request_file_activation(
            body, "expected_file_activation_generation")
        index = int(body.get("frame", -1))
        if state["video"] is None or state["topic"] is None or not 0 <= index < state["video"].topics[state["topic"]]["count"]:
            raise ValueError("帧号无效")
        expected_topic = body.get("expected_topic")
        expected_path = body.get("expected_path")
        if expected_topic is not None and expected_topic != state["topic"]:
            raise StaleHumanReviewError("相机已变化，拒绝设置旧参考帧")
        if expected_path is not None and expected_path != str(state["video"].path):
            raise StaleHumanReviewError("数据文件已变化，拒绝设置旧参考帧")
        if not state["truth"].get(str(index)):
            raise ValueError("请先保存该帧真值，再设为参考帧")
        review_status = current_truth_review_status().get(str(index), {})
        if not review_status.get("reusable_as_reference"):
            reasons = {
                "missing_current_human_review_receipt": "该帧只有历史标注，尚未通过当前人工复核门禁",
                "legacy_human_review_contract": "该帧只有旧版复核记录，需要按当前门禁重新确认",
                "truth_changed_after_human_review": "该帧轮廓已在人工确认后发生变化，需要重新确认",
                "human_review_transaction_not_committed": "该帧人工复核事务尚未完整提交",
                "empty_truth_not_reference_eligible": "空真值不能作为参考帧",
            }
            reason = reasons.get(review_status.get("reason_code"), "该帧缺少可核验的人工确认回执")
            raise ValueError(reason + "；请先载入为草稿、人工检查并保存真值")
        refs = state.setdefault("reference_frames", {})
        refs[state["topic"]] = [index]
        save()
        return jsonify(frame=index, topic=state["topic"],
                       file_activation_generation=activation_generation)


@app.post("/api/config")
def set_config():
    body = request.get_json() or {}
    if not isinstance(body, dict):
        raise ValueError("配置必须是 JSON 对象")
    if "llm_prompt_profile" in body and body["llm_prompt_profile"] not in ("full", "compact"):
        raise ValueError("Unknown model prompt profile")
    fast_string_keys = {"llm_fast_url", "llm_fast_model", "llm_fast_key"}
    fast_integer_limits = {
        "llm_fast_max_pixels": (1024, 16_777_216),
        "llm_fast_max_long_edge": (32, 8192),
        "llm_fast_max_extra_pixels": (1024, 33_554_432),
        "llm_fast_max_extra_long_edge": (32, 16384),
        "llm_fast_timeout_seconds": (10, 300),
        "llm_fast_cooldown_seconds": (10, 300),
    }
    fast_updates = {}
    for key in fast_string_keys:
        if key not in body:
            continue
        value = body[key]
        if value in (None, ""):
            fast_updates[key] = None
            continue
        if not isinstance(value, str) or len(value) > (4096 if key.endswith("key") else 500):
            raise ValueError(f"Invalid {key}")
        if key == "llm_fast_url":
            parsed = urlsplit(value)
            if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                    or parsed.username is not None or parsed.password is not None):
                raise ValueError("快速 VLM 地址必须是无内嵌凭据的 HTTP(S) 地址")
        fast_updates[key] = value
    for key, (minimum, maximum) in fast_integer_limits.items():
        if key not in body:
            continue
        value = body[key]
        if value in (None, ""):
            fast_updates[key] = None
        elif type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"Invalid {key}")
        else:
            fast_updates[key] = value
    with lock:
        with run_status_lock:
            if run_status.get("status") in {"running", "stopping"}:
                raise RuntimeError("自动标注仍在运行，请任务结束后再修改模型配置")
        generation, path, topic, previous_sha = assert_config_write_binding(body)
        previous_config = copy.deepcopy(state.setdefault("config", {}))
        updated_keys = []
        try:
            for key in ("llm_url", "llm_model", "llm_key",
                        "llm_prompt_profile", "sam3_url", "sam3_point_url"):
                if key in body and body[key]:
                    state["config"][key] = body[key]
                    updated_keys.append(key)
            if "ego_bottom_point_fallback" in body:
                if type(body["ego_bottom_point_fallback"]) is not bool:
                    raise ValueError("ego_bottom_point_fallback 必须为布尔值")
                state["config"]["ego_bottom_point_fallback"] = body[
                    "ego_bottom_point_fallback"]
                updated_keys.append("ego_bottom_point_fallback")
            for key, value in fast_updates.items():
                if value is None:
                    state["config"].pop(key, None)
                else:
                    state["config"][key] = value
                updated_keys.append(key)
            save()
        except Exception as error:
            state["config"] = previous_config
            raise RuntimeError("模型配置保存失败，已恢复原配置") from error
        current_sha = config_state_sha256()
        return jsonify(
            saved=True,
            config=public_model_config(),
            config_state_sha256=current_sha,
            previous_config_state_sha256=previous_sha,
            updated_keys=sorted(set(updated_keys)),
            path=path,
            topic=topic,
            file_activation_generation=generation,
        )


@app.get("/api/model_readiness")
def model_readiness():
    started = time.perf_counter()
    with lock:
        config = dict(state.setdefault("config", {}))
    sam_url = (config.get("sam3_url") or os.environ.get("SAM3_URL")
               or "http://127.0.0.1:8766").rstrip("/")
    llm_url = str(config.get("llm_url") or "").rstrip("/")
    configured_model = str(config.get("llm_model") or "").strip()
    fast_url = str(config.get("llm_fast_url") or "").rstrip("/")
    fast_model = str(config.get("llm_fast_model") or "").strip()
    def probe_sam():
        result = {"ready": False, "status": "unavailable", "model_loaded": False,
                  "on_demand": False}
        try:
            health = _fetch_readiness_json(sam_url + "/health")
            result.update(status=health.get("status", "unknown"),
                          model_loaded=bool(health.get("model_loaded")),
                          on_demand=bool(health.get("on_demand")))
            result["ready"] = health.get("status") == "ok"
        except RuntimeError as error:
            result["error"] = str(error)
        return result

    def probe_vlm(url, model):
        result = {"ready": False, "status": "unavailable", "model": model,
                  "registered_models": []}
        if not url:
            result["error"] = "尚未配置 VLM 地址"
            return result
        if not model:
            result["error"] = "尚未配置 VLM 模型名"
            return result
        try:
            listing = _fetch_readiness_json(url + "/models")
            entries = listing.get("data") if isinstance(listing.get("data"), list) else []
            models = sorted({str(item.get("id")) for item in entries
                             if isinstance(item, dict) and item.get("id")})
            result.update(status="ok", registered_models=models,
                          ready=model in models)
            if not result["ready"]:
                result["error"] = (f"配置模型 {model} 未在服务中登记"
                                   if models else "VLM 服务没有返回可用模型")
        except RuntimeError as error:
            result["error"] = str(error)
        return result

    fast_configured = bool(fast_url or fast_model)
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="readiness") as executor:
        sam_future = executor.submit(probe_sam)
        vlm_future = executor.submit(probe_vlm, llm_url, configured_model)
        fast_future = (executor.submit(probe_vlm, fast_url, fast_model)
                       if fast_configured else None)
        sam, vlm = sam_future.result(), vlm_future.result()
        fast_vlm = (fast_future.result() if fast_future is not None else {
            "configured": False, "ready": False, "status": "disabled",
            "model": "", "registered_models": []})
    fast_vlm["configured"] = fast_configured
    fast_vlm["acceleration_only"] = True
    def positive_config_int(key, default):
        value = config.get(key)
        return value if type(value) is int and value > 0 else default
    fast_vlm["max_pixels"] = positive_config_int("llm_fast_max_pixels", 640 * 512)
    fast_vlm["max_long_edge"] = positive_config_int("llm_fast_max_long_edge", 640)
    fast_vlm["max_extra_pixels"] = positive_config_int(
        "llm_fast_max_extra_pixels", fast_vlm["max_pixels"] * 4)
    fast_vlm["max_extra_long_edge"] = positive_config_int(
        "llm_fast_max_extra_long_edge", fast_vlm["max_long_edge"] * 2)
    fast_vlm["cooldown_seconds"] = positive_config_int(
        "llm_fast_cooldown_seconds", 60)
    return jsonify(ready=sam["ready"] and vlm["ready"], sam=sam, vlm=vlm,
                   fast_vlm=fast_vlm,
                   checked_at=datetime.now(timezone.utc).isoformat(),
                   elapsed_ms=round((time.perf_counter() - started) * 1000, 3))


def run_storage_readiness(frame_count=1):
    """Return a conservative, user-visible save-space contract for a run."""
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or not 1 <= frame_count <= 100:
        raise ValueError("处理帧数须为 1～100")
    required = RUN_STORAGE_BASE_RESERVE_BYTES + frame_count * RUN_STORAGE_PER_FRAME_BYTES
    try:
        usage = shutil.disk_usage(WORK)
    except OSError as error:
        return {
            "ready": False, "warning": True, "status": "unavailable",
            "frame_count": frame_count, "required_bytes": required,
            "required_mib": round(required / (1024 * 1024), 2),
            "message": f"无法检查工作目录保存空间：{str(error)[:180]}；尚未启动推理",
        }
    free = int(usage.free)
    ready = free >= required
    warning = free < RUN_STORAGE_WARNING_BYTES
    free_mib = round(free / (1024 * 1024), 2)
    required_mib = round(required / (1024 * 1024), 2)
    if not ready:
        message = (f"保存空间不足：工作目录所在磁盘可用 {free_mib:.2f} MiB，"
                   f"本次 {frame_count} 帧至少预留 {required_mib:.2f} MiB。"
                   "尚未启动推理，请先释放空间或减少处理帧数")
    elif warning:
        message = (f"保存空间偏低：可用 {free_mib:.2f} MiB；"
                   f"本次 {frame_count} 帧已通过 {required_mib:.2f} MiB 安全门槛，"
                   "建议只运行少量帧并尽快释放空间")
    else:
        message = f"保存空间充足：可用 {free_mib:.2f} MiB"
    return {
        "ready": ready, "warning": warning,
        "status": "blocked" if not ready else "low" if warning else "ok",
        "frame_count": frame_count, "free_bytes": free,
        "free_mib": free_mib, "required_bytes": required,
        "required_mib": required_mib,
        "warning_below_bytes": RUN_STORAGE_WARNING_BYTES,
        "message": message,
    }


@app.get("/api/run/readiness")
def run_readiness():
    """Validate local workflow prerequisites before a long model request starts."""
    try:
        requested_count = int(request.args.get("count", "1"))
        storage = run_storage_readiness(requested_count)
    except (TypeError, ValueError):
        return jsonify(ready=False, code="frame_request_invalid",
                       message="本次处理数量须为 1～100 个整数帧")
    with lock:
        if state.get("video") is None or not state.get("topic"):
            return jsonify(ready=False, code="data_missing",
                           message="请先加载图像或 MCAP 数据", storage=storage)
        references = state.get("reference_frames", {}).get(state["topic"], [])
        if not references:
            return jsonify(ready=False, code="reference_missing",
                           message="请先人工确认并锁定一个参考帧", storage=storage)
        reference = references[0]
        if not state.get("truth", {}).get(str(reference)):
            return jsonify(ready=False, code="reference_truth_missing",
                           reference_frame=reference,
                           message=f"参考帧 {reference} 尚无人工真值", storage=storage)
        try:
            allowed_labels = selected_auto_labels(reference)
        except (KeyError, ValueError) as error:
            return jsonify(ready=False, code="label_selection_invalid",
                           reference_frame=reference, message=str(error), storage=storage)
        expected_description_path = description_cache_path(reference_signature(reference))
        legacy_description_path = WORK / "description.json"
        description = load_reference_description(reference)
        if description is None:
            stale_description_exists = (expected_description_path.exists()
                                        or legacy_description_path.exists())
            return jsonify(ready=False,
                           code="description_stale" if stale_description_exists else "description_missing",
                           reference_frame=reference,
                           message=(f"参考帧 {reference} 的描述已失效，请重新生成"
                                    if stale_description_exists else
                                    f"请打开参考帧 {reference} 并生成真值描述"),
                           storage=storage)
        try:
            prompts = description.get("prompts", []) if isinstance(description, dict) else []
            prompt_labels = [item.get("label") for item in prompts if isinstance(item, dict)]
            valid = (isinstance(description, dict) and bool(description.get("description"))
                     and len(prompts) == len(allowed_labels)
                     and set(prompt_labels) == set(allowed_labels)
                     and description.get("reference_signature") == reference_signature(reference))
        except (OSError, ValueError, TypeError, KeyError):
            valid = False
        if not valid:
            return jsonify(ready=False, code="description_stale",
                           reference_frame=reference,
                           message=f"参考帧 {reference} 的描述已失效，请重新生成",
                           storage=storage)
        if not storage["ready"]:
            return jsonify(ready=False, code="storage_low",
                           reference_frame=reference, topic=state["topic"],
                           labels=allowed_labels, description_ready=True,
                           message=storage["message"], storage=storage)
        return jsonify(ready=True, code="ready", reference_frame=reference,
                       topic=state["topic"], labels=allowed_labels,
                       description_ready=True, storage=storage)


@app.post("/api/run/data-readiness")
def run_data_readiness():
    """Decode the first requested frame before any model work is submitted.

    Catalog metadata alone cannot prove that a camera payload is decodable.  A
    bounded one-frame probe catches unsupported encodings and damaged sources
    early, while also warming the normal frame cache for the upcoming run.
    """
    body = request.get_json(silent=True) or {}
    with lock:
        video = state.get("video")
        topic = state.get("topic")
        if video is None or not topic:
            return jsonify(ready=False, code="data_missing",
                           message="请先加载图像或 MCAP 数据")
        topic_metadata = video.topics.get(topic)
        if not isinstance(topic_metadata, dict):
            return jsonify(ready=False, code="topic_missing",
                           message="当前相机话题已不存在，请重新加载数据")
        frame_count = topic_metadata.get("count")
        if (isinstance(frame_count, bool) or not isinstance(frame_count, int)
                or frame_count < 1):
            return jsonify(ready=False, code="empty_topic",
                           message="当前相机话题没有可读取的图像帧")
        requested_frames = body.get("frames")
        if requested_frames is not None:
            if (not isinstance(requested_frames, list) or not requested_frames
                    or len(requested_frames) > 100
                    or any(isinstance(item, bool) or not isinstance(item, int)
                           for item in requested_frames)):
                return jsonify(ready=False, code="frame_request_invalid",
                               message="待检查帧须为 1～100 个整数帧号")
            probe_frame = requested_frames[0]
            requested_count = len(requested_frames)
        else:
            probe_frame = body.get("frame")
            count = body.get("count", 1)
            if (isinstance(probe_frame, bool) or not isinstance(probe_frame, int)
                    or isinstance(count, bool) or not isinstance(count, int)
                    or not 1 <= count <= 100):
                return jsonify(ready=False, code="frame_request_invalid",
                               message="当前帧和处理帧数无效")
            requested_count = count
        if not 0 <= probe_frame < frame_count:
            return jsonify(ready=False, code="frame_out_of_range",
                           message=f"帧 {probe_frame} 超出当前图像源范围（共 {frame_count} 帧）")
        source_path = str(video.path)

    storage = run_storage_readiness(requested_count)
    if not storage["ready"]:
        return jsonify(ready=False, code="storage_low",
                       message=storage["message"], storage=storage,
                       frame=probe_frame, frame_count=frame_count)

    started = time.perf_counter()
    try:
        metadata = decoded_frame_metadata(video, topic, probe_frame)
    except Exception as error:
        return jsonify(ready=False, code="frame_decode_failed",
                       frame=probe_frame,
                       message=f"帧 {probe_frame} 无法解码：{str(error)[:300]}")
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    with lock:
        current = state.get("video")
        if current is not video or state.get("topic") != topic:
            return jsonify(ready=False, code="data_changed",
                           message="检查期间图像源已变化，请重新开始")
    return jsonify(ready=True, code="ready", path=source_path, topic=topic,
                   frame=probe_frame, frame_count=frame_count,
                   width=metadata["width"], height=metadata["height"],
                   encoded_bytes=metadata["encoded_bytes"],
                   elapsed_ms=elapsed_ms, cache_warmed=True,
                   scope="first_requested_frame", storage=storage)


@app.post("/api/describe")
@run_claimed
def describe_truth():
    body = request.get_json() or {}
    with lock:
        index = int(body["frame"])
        if state.get("video") is None or state.get("topic") is None:
            raise ValueError("请先加载图像消息")
        if index not in state.get("reference_frames", {}).get(state["topic"], []):
            raise ValueError("请先将当前帧设为参考帧")
        if not state["truth"].get(str(index)):
            raise ValueError("请先保存该参考帧的真值")
        allowed_labels = selected_auto_labels(index)
        selected_annotations = [shape for shape in copy.deepcopy(state["truth"][str(index)])
                                if shape.get("label") in allowed_labels]
        selected_present = {shape.get("label") for shape in selected_annotations}
        video, topic = state["video"], state["topic"]
        file_name = str(video.path)
        config = copy.deepcopy(state["config"])
        labels = copy.deepcopy(state.get("labels", []))
        auto_labels = copy.deepcopy(state.get("auto_labels"))
        references = copy.deepcopy(state.get("reference_frames", {}))
        truth = copy.deepcopy(state["truth"][str(index)])
        signature = reference_signature(index)
        target_image = video.frame(topic, index)
        marked_image = reference_overlay(target_image, selected_annotations)
        with run_status_lock:
            _update_run_status_locked(status="running", phase="describing", frame=index,
                                      completed=0, total=1, round=0, shape_count=0,
                                      topic=topic, file=file_name, error=None)

    trace = []
    save_debug(trace)
    def on_trace(event):
        event = redact_debug(event | {"frame": index, "purpose": "生成真值描述",
            "topic": topic, "file": file_name}, credential=config.get("llm_key") or "")
        if trace:
            trace[0] = event
        else:
            trace.append(event)
        save_debug(trace)
    try:
        with cancellation(run_stop_event), trace_calls(on_trace, save_inspection_image):
            result = describe(config, target_image,
                selected_annotations, allowed_labels, marked_image,
                include_unreferenced=any(label not in selected_present for label in allowed_labels))
        if run_stop_event.is_set():
            raise RunCancelled("用户已停止生成真值描述")
        with lock:
            if (state.get("video") is not video
                    or str(state["video"].path) != file_name
                    or state.get("topic") != topic
                    or state.get("config") != config
                    or state.get("labels", []) != labels
                    or state.get("auto_labels") != auto_labels
                    or state.get("reference_frames", {}) != references
                    or state.get("truth", {}).get(str(index)) != truth
                    or reference_signature(index) != signature):
                raise StaleHumanReviewError("数据、配置、类别或参考真值在描述期间变化，未保存旧描述")
            result["reference_signature"] = signature
            save_reference_description(result)
        save_description_trace(trace, result)
        update_run_status(status="completed", phase="completed", frame=None,
                          completed=1, total=1)
        return jsonify(result)
    except RunCancelled:
        update_run_status(status="stopped", phase="stopped", frame=None,
                          completed=0, total=1, error=None)
        return jsonify(stopped=True, description_saved=False)


@app.post("/api/reference_candidates/refine")
@run_claimed
def refine_reference_candidates():
    """Finish deferred Agent review and replace an untouched machine draft."""
    body = request.get_json() or {}
    if type(body.get("frame")) is not int or not isinstance(body.get("topic"), str):
        raise ValueError("候选精修必须包含整数 frame 和 topic")
    index, topic = body["frame"], body["topic"]
    refinement_id = _normalize_refinement_id(body.get("refinement_id"))
    if _consume_refinement_cancel(refinement_id):
        update_run_status(status="stopped", phase="stopped", frame=None,
                          completed=0, total=1, topic=topic,
                          refinement_id=refinement_id,
                          cancellation_reason="manual_edit")
        return jsonify(stopped=True, cancellation_reason="manual_edit",
                       refinement_id=refinement_id, candidate_only=True,
                       candidate_mutated=False, human_truth=False)
    expected_artifact = body.get("expected_candidate_artifact_sha256")
    expected_frame = body.get("expected_frame_sha256")
    with lock:
        video = state.get("video")
        if video is None or state.get("topic") is None:
            raise ValueError("请先加载候选对应的数据")
        if topic != state["topic"]:
            raise StaleHumanReviewError("当前 topic 已变化，请重新读取候选")
        if not 0 <= index < video.topics[topic]["count"]:
            raise ValueError("候选精修帧号无效")
        if index in state.get("reference_frames", {}).get(topic, []):
            raise ValueError("参考帧不能进行机器候选精修")
        if str(index) in state.get("truth", {}):
            raise ValueError("已有真值的帧不能进行机器候选精修")
        key = str(index)
        candidates = copy.deepcopy(state.get("candidates", {}).get(key))
        if not isinstance(candidates, list) or not candidates:
            raise StaleHumanReviewError("当前帧没有持久候选")
        artifact_sha = _sha256_json(candidates)
        if not _valid_sha256(expected_artifact) or expected_artifact != artifact_sha:
            raise StaleHumanReviewError("候选已变化，请重新读取当前候选")
        if expected_frame is not None and not _valid_sha256(expected_frame):
            raise ValueError("expected_frame_sha256 必须是有效 SHA-256")
        image_bytes = frame_image(index)
        input_sha = hashlib.sha256(image_bytes).hexdigest()
        if expected_frame is not None and expected_frame != input_sha:
            raise StaleHumanReviewError("帧图像已变化，请重新读取后精修")
        previous_run = copy.deepcopy(state.get("agent_runs", {}).get(key, {}))
        if previous_run.get("agent_review_status") != "pending":
            raise StaleHumanReviewError("当前候选没有待完成的 Agent 精修")
        draft_labels = previous_run.get("draft_labels")
        registry = list(state.get("labels", []))
        if (not isinstance(draft_labels, list) or not draft_labels
                or any(label not in registry for label in draft_labels)
                or any(shape.get("label") not in draft_labels for shape in candidates)):
            raise StaleHumanReviewError("候选类别或标签表已变化，请重新生成草稿")
        config = copy.deepcopy(state.get("config", {}))
        config_sha = _sha256_json(config)
        truth_sha = _sha256_json(state.get("truth", {}))
        references_sha = _sha256_json(state.get("reference_frames", {}))
        run_sha = _sha256_json(previous_run)
        review_reference = previous_run.get("review_reference_frame")
        if review_reference is None:
            description = canonical_unreferenced_description(draft_labels)
            reference_context = None
        else:
            if type(review_reference) is not int:
                raise StaleHumanReviewError("候选参考帧记录无效，请重新生成草稿")
            if review_reference not in state.get("reference_frames", {}).get(topic, []):
                raise StaleHumanReviewError("候选绑定的参考帧已变化，请重新生成草稿")
            reference_truth = copy.deepcopy(state.get("truth", {}).get(str(review_reference)))
            if not isinstance(reference_truth, list) or not reference_truth:
                raise StaleHumanReviewError("候选绑定的参考真值已缺失，请重新生成草稿")
            if previous_run.get("review_reference_signature") != reference_signature(review_reference):
                raise StaleHumanReviewError("候选绑定的参考真值已变化，请重新生成草稿")
            current_labels = selected_auto_labels(review_reference)
            if current_labels != draft_labels:
                raise StaleHumanReviewError("候选绑定的自动标注类别已变化，请重新生成草稿")
            description = load_reference_description(review_reference)
            if (not isinstance(description, dict)
                    or description.get("reference_signature") != reference_signature(review_reference)
                    or _sha256_json(description) != previous_run.get("review_description_sha256")):
                raise StaleHumanReviewError("候选绑定的参考描述已变化，请重新生成草稿")
            description = restrict_description(description, draft_labels)
            reference_annotations = [shape for shape in reference_truth
                                     if shape.get("label") in draft_labels]
            reference_image = video.frame(topic, review_reference)
            reference_context = {
                "image": reference_image,
                "marked_image": reference_overlay(reference_image, reference_annotations),
                "annotations": reference_annotations,
            }
        max_rounds = previous_run.get("max_rounds", 1)
        if type(max_rounds) is not int or not 1 <= max_rounds <= MAX_REVIEW_ROUNDS:
            raise ValueError("候选精修轮数记录无效")
        update_run_status(status="running", phase="agent_refining", frame=index,
                          completed=0, total=1, topic=topic, shape_count=len(candidates),
                          round=0, error=None, refinement_id=refinement_id,
                          cancellation_reason=None)

    trace = []
    debug_key = config.get("llm_key")
    def on_trace(event):
        record = redact_debug(event | {"frame": index, "topic": topic,
            "file": str(video.path), "purpose": "后台 Agent 候选精修"},
            credential=debug_key or "")
        existing = next((position for position, item in enumerate(trace)
                         if item["id"] == record["id"]), None)
        if existing is None:
            trace.append(record)
        else:
            trace[existing] = record
        save_debug(trace)

    def on_progress(event):
        if run_stop_event.is_set():
            raise RunCancelled("用户已停止候选精修")
        update_run_status(phase=("agent_refining" if event["phase"] == "observing"
                                 else event["phase"]),
                          round=event["round"], shape_count=len(event["shapes"]),
                          error=None)

    try:
        with cancellation(run_stop_event), trace_calls(on_trace, save_inspection_image):
            result = review_existing(
                config, image_bytes, description, draft_labels, candidates,
                max_rounds=max_rounds, on_progress=on_progress,
                reference=reference_context,
                candidate_filters=previous_run.get("candidate_filters"),
                sam_generation=previous_run.get("sam_generation"))
        if result["agent"].get("stop_reason") == "error":
            raise RuntimeError("Agent 未能完成候选精修；已保留初始机器候选")
        if run_stop_event.is_set():
            raise RunCancelled("用户已停止候选精修")
        refined = result["shapes"]
        initial_geometry_sha = _sha256_json([
            {name: item.get(name) for name in ("id", "label", "points", "holes", "parts")}
            for item in candidates])
        refined_geometry_sha = _sha256_json([
            {name: item.get(name) for name in ("id", "label", "points", "holes", "parts")}
            for item in refined])
        fallback_provenance = copy.deepcopy(candidates[0].get("provenance", {}))
        for item in refined:
            item["candidate_only"] = True
            item["human_truth"] = False
            item["approved_for_training"] = False
            provenance = copy.deepcopy(item.get("provenance") or fallback_provenance)
            provenance["kind"] = "machine_candidate"
            provenance["pipeline"] = "sam_vlm_agent"
            provenance["agent_review_status"] = "complete"
            item["provenance"] = provenance
        refined_sha = _sha256_json(refined)
        geometry_mutated = refined_geometry_sha != initial_geometry_sha
        preserved_queue = [item for item in previous_run.get("review_queue", [])
                           if (item.get("source") == "visible_label_discovery"
                               or item.get("reason_code") ==
                               "slope_candidate_current_ego_conflict")]
        replacement_run = copy.deepcopy(previous_run)
        for name in ("converged", "rounds", "max_rounds", "stop_reason", "audit",
                     "candidate_filters", "sam_generation"):
            replacement_run[name] = copy.deepcopy(result["agent"].get(name))
        replacement_run["review_queue"] = result["review_queue"] + preserved_queue
        replacement_run["applied"] = result["applied"]
        replacement_run["agent_review_status"] = "complete"
        replacement_run["agent_review_deferred"] = False
        replacement_run["agent_refinement"] = {
            "initial_candidate_artifact_sha256": artifact_sha,
            "refined_candidate_artifact_sha256": refined_sha,
            "initial_geometry_sha256": initial_geometry_sha,
            "refined_geometry_sha256": refined_geometry_sha,
            "input_sha256": input_sha,
            "candidate_mutated": geometry_mutated,
        }
        replacement_run["candidate_artifact_sha256"] = refined_sha
        replacement_run["candidate_class_audit_status"] = (
            "pending" if any(candidate_requires_class_audit(item) for item in refined)
            else "complete")
        replacement_run["candidate_class_audit_deferred"] = (
            replacement_run["candidate_class_audit_status"] == "pending")
        replacement_run["candidate_class_audit_total"] = sum(
            candidate_requires_class_audit(item) for item in refined)
        replacement_run["candidate_class_audits"] = []
        replacement_run["human_review_required"] = True
        replacement_run["inspection_trace"] = copy.deepcopy(trace)

        with lock, run_status_lock:
            if (run_stop_event.is_set() or state.get("video") is not video
                    or state.get("topic") != topic
                    or _sha256_json(state.get("config", {})) != config_sha
                    or list(state.get("labels", [])) != registry
                    or _sha256_json(state.get("truth", {})) != truth_sha
                    or _sha256_json(state.get("reference_frames", {})) != references_sha
                    or hashlib.sha256(frame_image(index)).hexdigest() != input_sha
                    or _sha256_json(state.get("candidates", {}).get(key)) != artifact_sha
                    or _sha256_json(state.get("agent_runs", {}).get(key, {})) != run_sha):
                raise StaleHumanReviewError(
                    "候选、帧、标签或人工状态在精修期间变化，未采用后台结果")
            previous_revisions = copy.deepcopy(state.get("candidate_revisions", {}).get(key))
            previous_review_status = copy.deepcopy(
                state.get("candidate_review_status", {}).get(key))
            try:
                archive_candidate_revision(
                    index, topic, candidates, previous_run, refined, replacement_run,
                    reason="deferred_agent_refinement")
                state["candidates"][key] = refined
                state.setdefault("agent_runs", {})[key] = replacement_run
                state.setdefault("candidate_review_status", {})[key] = {}
                save()
            except Exception:
                state["candidates"][key] = candidates
                state.setdefault("agent_runs", {})[key] = previous_run
                if previous_revisions is None:
                    state.setdefault("candidate_revisions", {}).pop(key, None)
                else:
                    state.setdefault("candidate_revisions", {})[key] = previous_revisions
                if previous_review_status is None:
                    state.setdefault("candidate_review_status", {}).pop(key, None)
                else:
                    state.setdefault("candidate_review_status", {})[key] = previous_review_status
                raise
            _update_run_status_locked(status="completed", phase="completed", frame=None,
                                      completed=1, total=1, shape_count=len(refined))
            bindings = candidate_review_bindings()
            return jsonify(frame=index, topic=topic, candidates=refined,
                           agent=replacement_run, binding=bindings.get(key),
                           refinement_id=refinement_id,
                           candidate_review_bindings=bindings,
                           input_sha256=input_sha,
                           initial_candidate_artifact_sha256=artifact_sha,
                           candidate_artifact_sha256=refined_sha,
                           candidate_mutated=geometry_mutated,
                           candidate_only=True, human_truth=False,
                           approved_for_training=False)
    except RunCancelled:
        with run_status_lock:
            cancellation_reason = run_status.get("cancellation_reason") or "user_stop"
        update_run_status(status="stopped", phase="stopped", frame=None, completed=0,
                          cancellation_reason=cancellation_reason)
        return jsonify(stopped=True, candidate_only=True, candidate_mutated=False,
                       human_truth=False, refinement_id=refinement_id,
                       cancellation_reason=cancellation_reason)
    except Exception as error:
        update_run_status(status="failed", phase="failed", frame=None,
                          completed=0, error=str(error)[:300])
        raise


@app.post("/api/reference_candidates/refine/stop")
def stop_reference_candidate_refinement():
    """Cancel only the deferred refinement identified by the editing client."""
    body = request.get_json() or {}
    if type(body.get("frame")) is not int or not isinstance(body.get("topic"), str):
        raise ValueError("取消候选精修必须包含整数 frame 和 topic")
    index, topic = body["frame"], body["topic"]
    refinement_id = _normalize_refinement_id(body.get("refinement_id"))
    if refinement_id is None:
        raise ValueError("取消候选精修必须包含 refinement_id")
    _record_refinement_cancel(refinement_id)
    with run_status_lock:
        matched = (
            run_status.get("status") in {"running", "stopping"}
            and run_status.get("refinement_id") == refinement_id
            and run_status.get("frame") == index
            and run_status.get("topic") == topic
        )
        if matched:
            run_stop_event.set()
            _consume_refinement_cancel(refinement_id)
            _update_run_status_locked(status="stopping", phase="stopping",
                                      cancellation_reason="manual_edit")
    return jsonify(cancellation_requested=True, stopped=matched,
                   refinement_id=refinement_id, frame=index, topic=topic)


@app.post("/api/reference_candidates/audit")
@run_claimed
def audit_reference_candidates():
    """Atomically blind-audit every persisted high-risk candidate in a frame."""
    body = request.get_json() or {}
    if type(body.get("frame")) is not int or not isinstance(body.get("topic"), str):
        raise ValueError("候选盲审必须包含整数 frame 和 topic")
    index, topic = body["frame"], body["topic"]
    expected_artifact = body.get("expected_candidate_artifact_sha256")
    expected_frame = body.get("expected_frame_sha256")
    with lock:
        video = state.get("video")
        if video is None or state.get("topic") is None:
            raise ValueError("请先加载候选对应的数据")
        if topic != state["topic"]:
            raise StaleHumanReviewError("当前 topic 已变化，请重新读取候选")
        if not 0 <= index < video.topics[topic]["count"]:
            raise ValueError("候选盲审帧号无效")
        if index in state.get("reference_frames", {}).get(topic, []):
            raise ValueError("参考帧不能进行机器候选盲审")
        if str(index) in state.get("truth", {}):
            raise ValueError("已有真值的帧不能进行机器候选盲审")
        key = str(index)
        candidates = copy.deepcopy(state.get("candidates", {}).get(key))
        if not isinstance(candidates, list) or not candidates:
            raise StaleHumanReviewError("当前帧没有持久候选")
        artifact_sha = _sha256_json(candidates)
        if not _valid_sha256(expected_artifact) or expected_artifact != artifact_sha:
            raise StaleHumanReviewError("候选已变化，请重新读取当前候选")
        if expected_frame is not None and not _valid_sha256(expected_frame):
            raise ValueError("expected_frame_sha256 必须是有效 SHA-256")
        image_bytes = frame_image(index)
        input_sha = hashlib.sha256(image_bytes).hexdigest()
        if expected_frame is not None and expected_frame != input_sha:
            raise StaleHumanReviewError("帧图像已变化，请重新读取后复审")
        ids = [item.get("id") for item in candidates
               if candidate_requires_class_audit(item)]
        if len(ids) != len(set(ids)):
            raise ValueError("高风险候选 ID 重复，拒绝复审")
        audit_version = candidate_class_audit_version()
        existing_run = state.get("agent_runs", {}).get(key, {})
        existing_review_status = state.get("candidate_review_status", {}).get(key, {})
        if reusable_candidate_class_audit(
                existing_run, existing_review_status, candidates,
                list(state.get("labels", [])), artifact_sha, input_sha, audit_version):
            update_run_status(status="completed", phase="completed", frame=None,
                              completed=1, total=1, audit_cache_hit=True)
            bindings = candidate_review_bindings()
            return jsonify(
                frame=index, topic=topic, candidates=candidates,
                agent=copy.deepcopy(existing_run), binding=bindings.get(key),
                candidate_review_bindings=bindings, input_sha256=input_sha,
                candidate_artifact_sha256=artifact_sha,
                candidate_class_audit_cache_hit=True,
                candidate_only=True, human_truth=False,
                approved_for_training=False)
        update_run_status(status="stopping" if run_stop_event.is_set() else "running", phase="candidate_class_audit", frame=index,
                          completed=0, total=len(ids), topic=topic, error=None)
        def check_cancelled():
            if run_stop_event.is_set():
                raise RunCancelled("用户已停止候选盲审")
        try:
            with cancellation(run_stop_event):
                audits, queue = audit_high_risk_candidates(
                    state["config"], image_bytes, candidates, list(state.get("labels", [])),
                    VISIBLE_LABEL_DEFINITIONS, chat, check_cancelled=check_cancelled,
                    tolerate_errors=False, batch_chat=chat_batch)
            check_cancelled()
        except RunCancelled:
            update_run_status(status="stopped", phase="stopped", frame=None, completed=0)
            return jsonify(stopped=True, candidate_only=True, candidate_mutated=False,
                           human_truth=False)
        except Exception as error:
            update_run_status(status="failed", phase="failed", frame=None,
                              completed=0, error=str(error)[:300])
            raise

        current_candidates = state.get("candidates", {}).get(key)
        if (_sha256_json(current_candidates) != artifact_sha
                or hashlib.sha256(frame_image(index)).hexdigest() != input_sha
                or state.get("topic") != topic
                or candidate_class_audit_version() != audit_version):
            raise StaleHumanReviewError("候选、帧或审核配置在复审期间变化，未写入任何审核结果")
        previous_run_present = key in state.get("agent_runs", {})
        previous_run = copy.deepcopy(state.get("agent_runs", {}).get(key, {}))
        runs = state.setdefault("agent_runs", {})
        replacement_run = copy.deepcopy(previous_run)
        for audit in audits:
            audit["audit_version"] = audit_version
            audit["audit_sha256"] = _sha256_json({
                k: v for k, v in audit.items() if k != "audit_sha256"})
        for entry in queue:
            entry["audit_sha256"] = next(
                audit["audit_sha256"] for audit in audits
                if audit["candidate_id"] == entry["candidate_id"])
        replacement_run["candidate_class_audits"] = audits
        replacement_run["candidate_class_audit_version"] = audit_version
        old_queue = replacement_run.get("review_queue", [])
        replacement_run["review_queue"] = [
            item for item in old_queue
            if item.get("source") != "candidate_class_audit"] + queue
        replacement_run["candidate_artifact_sha256"] = artifact_sha
        replacement_run["input_sha256"] = input_sha
        replacement_run["candidate_class_audit_status"] = "complete"
        replacement_run["human_review_required"] = True
        review_status = state.setdefault("candidate_review_status", {})
        review_status_present = key in review_status
        previous_review_status = copy.deepcopy(review_status.get(key, {}))
        status_for_frame = {}
        for audit in audits:
            status_for_frame[audit["candidate_sha256"]] = {
                "candidate_fingerprint": audit["candidate_sha256"],
                "candidate_artifact_sha256": artifact_sha,
                "audit_sha256": audit["audit_sha256"], "route": audit["route"]}
        try:
            runs[key] = replacement_run
            review_status[key] = status_for_frame
            save()
        except Exception:
            if previous_run_present:
                runs[key] = previous_run
            else:
                runs.pop(key, None)
            if review_status_present:
                review_status[key] = previous_review_status
            else:
                review_status.pop(key, None)
            raise
        update_run_status(status="completed", phase="completed", frame=None,
                          completed=1, total=1)
        bindings = candidate_review_bindings()
        return jsonify(frame=index, topic=topic, candidates=candidates,
                       agent=replacement_run, binding=bindings.get(key),
                       candidate_review_bindings=bindings, input_sha256=input_sha,
                       candidate_artifact_sha256=artifact_sha,
                       candidate_class_audit_cache_hit=False,
                       candidate_only=True, human_truth=False,
                       approved_for_training=False)


@app.post("/api/reference_discovery/prefetch")
def prefetch_reference_discovery():
    """Warm the exact deterministic visible-label request without writing state.

    The local VLM owns an exact-request cache and coalesces an identical request
    already in flight.  This endpoint only starts that same validated request
    after the operator opens the reference-preparation dialog.  It never claims
    a workflow task, writes candidates, or waits behind an unrelated busy model.
    """
    body = request.get_json() or {}
    if run_claim_lock.locked():
        return jsonify(prefetched=False, code="active_workflow"), 202
    with run_status_lock:
        if run_status.get("status") in {"running", "stopping"}:
            return jsonify(prefetched=False, code="active_workflow"), 202
    with lock:
        if state.get("video") is None or state.get("topic") is None:
            raise ValueError("请先加载图像消息")
        expected_path = body.get("expected_path")
        expected_topic = body.get("expected_topic")
        video = state["video"]
        selected_topic = state["topic"]
        file_name = str(video.path)
        if expected_path != file_name or expected_topic != selected_topic:
            # This endpoint is speculative.  A dataset/topic switch may race an
            # already-issued browser prefetch even after the client aborts it.
            # Treat that expected cancellation as a successful no-op instead of
            # surfacing a red 409 in the operator workbench.
            return jsonify(
                schema_version="labelagent.reference-discovery-prefetch.v1",
                prefetched=False,
                code="scope_changed",
                stale_request_discarded=True,
                candidate_only=True,
                human_truth=False,
                approved_for_training=False,
            ), 200
        index = int(body.get("frame", -1))
        count = video.topics[selected_topic]["count"]
        if not 0 <= index < count:
            raise ValueError("帧号无效")
        key = str(index)
        if key in state.get("truth", {}) or state.get("candidates", {}).get(key):
            return jsonify(prefetched=False, code="existing_annotation"), 200
        labels = list(state.get("labels", []))
        if not labels:
            raise ValueError("当前标签表为空，无法预取可见类别")
        config = copy.deepcopy(state.get("config", {}))
        config_sha256 = _sha256_json(config)
        target_image = frame_image(index)
        input_sha256 = hashlib.sha256(target_image).hexdigest()
        discovery_cache_key = reference_discovery_cache_key(
            video, selected_topic, index, input_sha256, labels, config_sha256)
    # Speculation must yield immediately when the shared model is serving an
    # unrelated request.  This option only changes HTTP busy retry behaviour;
    # it is not part of the deterministic VLM request body or cache key.
    config["llm_busy_retries"] = 0
    started = time.perf_counter()
    try:
        discovery, retry_count = discover_reference_labels(config, target_image, labels)
    except ValueError as error:
        message = str(error)
        if "Local VLM is busy" in message or "HTTP 503" in message:
            return jsonify(prefetched=False, code="vlm_busy"), 202
        raise
    discovery_elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    selected, recall_probes = candidate_labels_with_recall_probes(discovery, labels)
    sam_prefetch = {
        "prefetched": False,
        "code": "not_applicable",
        "selected_labels": selected,
        "recall_probe_labels": recall_probes,
    }
    if len(selected) > 1 and config.get("sam_multi_prompt", True):
        description = canonical_unreferenced_description(selected)
        prompt_map = {item["label"]: item["text"]
                      for item in description.get("prompts", [])}
        sam_started = time.perf_counter()
        warmed = None
        try:
            warmed = sam_text_multi_prefetch(target_image, prompt_map, config)
            sam_elapsed_ms = round((time.perf_counter() - sam_started) * 1000, 3)
            cache = ((warmed.get("diagnostics") or {}).get("response_cache")
                     if isinstance(warmed, dict) else None) or {}
            if warmed.get("prefetched") is False:
                sam_prefetch.update(code=warmed.get("code", "not_prefetched"),
                                    elapsed_ms=sam_elapsed_ms)
            else:
                sam_prefetch.update(
                    prefetched=True,
                    code="ready",
                    elapsed_ms=sam_elapsed_ms,
                    prompt_sha256=_sha256_json(prompt_map),
                    response_sha256=_sha256_json(warmed.get("prompts")),
                    cache_hit=cache.get("hit"),
                    coalesced=cache.get("coalesced"),
                    cache_entries=cache.get("entries"),
                )
        except Exception as error:
            # VLM discovery is still useful when an older SAM service lacks
            # speculative prefetch or a transient transport error occurs.
            sam_prefetch.update(code="sam_prefetch_unavailable",
                                error=str(error)[:300],
                                elapsed_ms=round((time.perf_counter() - sam_started) * 1000, 3))
        slope_payload = ((warmed.get("prompts") or {}).get("slope")
                         if isinstance(warmed, dict) else None)
        if (sam_prefetch.get("prefetched") is True and "slope" in selected
                and isinstance(slope_payload, dict)
                and slope_payload.get("results") == []):
            roi_items = []
            roi_started = time.perf_counter()
            for crop_index, crop in enumerate(slope_overlap_crop_images(target_image)):
                try:
                    roi_warm = sam_text_prefetch(
                        crop["image"], "traversable surface", config)
                    roi_cache = ((roi_warm.get("diagnostics") or {}).get("response_cache")
                                 if isinstance(roi_warm, dict) else None) or {}
                    roi_items.append({
                        "crop_index": crop_index,
                        "bbox": crop["bbox"],
                        "prefetched": roi_warm.get("prefetched") is True,
                        "code": roi_warm.get("code", "ready"),
                        "cache_hit": roi_cache.get("hit"),
                        "response_sha256": (_sha256_json(roi_warm.get("results"))
                                            if isinstance(roi_warm.get("results"), list)
                                            else None),
                    })
                    if roi_warm.get("prefetched") is False:
                        break
                except Exception as error:
                    roi_items.append({"crop_index": crop_index,
                                      "bbox": crop["bbox"],
                                      "prefetched": False,
                                      "code": "sam_prefetch_unavailable",
                                      "error": str(error)[:300]})
                    break
            sam_prefetch["slope_roi_prefetch"] = {
                "reason": "full_frame_slope_empty",
                "requested": 4,
                "completed": sum(item.get("prefetched") is True for item in roi_items),
                "elapsed_ms": round((time.perf_counter() - roi_started) * 1000, 3),
                "items": roi_items,
            }
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    with lock:
        if (state.get("video") is not video or str(video.path) != file_name
                or state.get("topic") != selected_topic
                or state.get("labels", []) != labels
                or _sha256_json(state.get("config", {})) != config_sha256):
            return jsonify(
                schema_version="labelagent.reference-discovery-prefetch.v1",
                prefetched=False,
                code="scope_changed",
                stale_request_discarded=True,
                candidate_only=True,
                human_truth=False,
                approved_for_training=False,
            ), 200
    store_reference_discovery_cache(
        discovery_cache_key, discovery, retry_count)
    return jsonify(
        schema_version="labelagent.reference-discovery-prefetch.v1",
        prefetched=True,
        file=file_name,
        topic=selected_topic,
        frame=index,
        input_sha256=input_sha256,
        labels=labels,
        discovery_sha256=_sha256_json(discovery),
        visible_labels=discovery.get("visible_labels", []),
        uncertain_labels=discovery.get("uncertain_labels", []),
        retry_count=retry_count,
        discovery_elapsed_ms=discovery_elapsed_ms,
        server_discovery_cache_stored=True,
        sam_prefetch=sam_prefetch,
        elapsed_ms=elapsed_ms,
        candidate_only=True,
        human_truth=False,
        approved_for_training=False,
    )


def primary_model_prewarm_image():
    """Return a deterministic, data-free factory scene for the present path."""
    image = Image.new("RGB", (640, 512), (210, 215, 220))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 300, 639, 511), fill=(125, 125, 125))
    draw.polygon([(380, 300), (620, 300), (639, 430), (420, 430)],
                 fill=(185, 170, 130))
    draw.rectangle((70, 360, 270, 385), fill=(135, 80, 35))
    for x in (80, 125, 170, 215, 260):
        draw.rectangle((x, 330, x + 18, 410), fill=(150, 90, 40))
    draw.ellipse((300, 225, 335, 260), fill=(235, 190, 150))
    draw.rectangle((307, 258, 328, 350), fill=(40, 90, 180))
    draw.line((317, 280, 280, 330), fill=(30, 30, 30), width=8)
    draw.line((317, 280, 360, 330), fill=(30, 30, 30), width=8)
    draw.line((316, 350, 290, 420), fill=(30, 30, 30), width=9)
    draw.line((320, 350, 350, 420), fill=(30, 30, 30), width=9)
    draw.rectangle((430, 240, 555, 330), fill=(190, 45, 35))
    draw.rectangle((470, 190, 520, 250), fill=(60, 60, 60))
    draw.rectangle((520, 325, 630, 340), fill=(70, 70, 70))
    draw.rectangle((520, 345, 630, 358), fill=(70, 70, 70))
    output = io.BytesIO()
    image.save(output, "PNG")
    return output.getvalue()


@app.post("/api/model_prewarm")
def prewarm_primary_model():
    """Exercise the full visible-label decode path without annotation writes.

    The remote VLM receives a deterministic synthetic request with its complete
    generation budget.  Repeated workbench starts are therefore cheap, while
    the first start after a model restart pays the long structured-decode setup
    cost before the browser is opened.  The fast route is deliberately removed
    so this warms the authoritative 4090 path.
    """
    if run_claim_lock.locked():
        return jsonify(prewarmed=False, code="active_workflow"), 202
    with run_status_lock:
        if run_status.get("status") in {"running", "stopping"}:
            return jsonify(prewarmed=False, code="active_workflow"), 202
    with lock:
        labels = list(state.get("labels", []))
        config = copy.deepcopy(state.get("config", {}))
    if not labels:
        return jsonify(prewarmed=False, code="no_labels"), 200
    for key in tuple(config):
        if key.startswith("llm_fast_"):
            config.pop(key, None)
    config["llm_busy_retries"] = 0
    # Match the remote service's unbounded default explicitly.  The production
    # discovery path remains compact; only this data-free startup request warms
    # the longer static generation cache before a user submits a real image.
    config["llm_max_tokens"] = PRIMARY_MODEL_PREWARM_TOKENS
    started = time.perf_counter()
    with model_prewarm_lock:
        try:
            discovery, retry_count = discover_reference_labels(
                config, primary_model_prewarm_image(), labels)
        except ValueError as error:
            message = str(error)
            if "Local VLM is busy" in message or "HTTP 503" in message:
                return jsonify(prewarmed=False, code="vlm_busy"), 202
            raise
    return jsonify(
        schema_version="labelagent.primary-model-prewarm.v1",
        profile="synthetic_factory_present_v1",
        prewarmed=True,
        code="ready",
        elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
        discovery_sha256=_sha256_json(discovery),
        visible_labels=discovery.get("visible_labels", []),
        retry_count=retry_count,
        candidate_only=True,
        human_truth=False,
        approved_for_training=False,
        state_written=False,
    )


@app.post("/api/reference_draft")
@run_claimed
def generate_reference_draft():
    """Create a candidate-only SAM -> VLM -> Agent draft without reference truth."""
    body = request.get_json() or {}
    with lock:
        if state["video"] is None or state["topic"] is None:
            raise ValueError("请先加载图像消息")
        index = int(body.get("frame", -1))
        count = state["video"].topics[state["topic"]]["count"]
        if not 0 <= index < count:
            raise ValueError("帧号无效")
        if index in state.get("reference_frames", {}).get(state["topic"], []):
            raise ValueError("该帧已经锁定为参考帧，机器不能再修改")
        if str(index) in state.get("truth", {}):
            status = current_truth_review_status().get(str(index), {})
            if status.get("reusable_as_reference") is True:
                raise ValueError("该帧已有已核验人工真值，请直接复用或另选未标注帧")
            raise ValueError("该帧已有历史标注，请先载入历史标注并完成人工复核")
        auto_detect_visible = body.get("auto_detect_visible") is True
        probe_missing_selected = body.get("probe_missing_selected") is True
        if probe_missing_selected and not auto_detect_visible:
            raise ValueError("类别补漏只适用于自动识别参考帧模式")
        defer_class_audit = body.get("defer_class_audit") is True
        defer_agent_review = body.get("defer_agent_review") is True
        max_rounds = requested_review_rounds(body)
        requested = list(state.get("labels", [])) if auto_detect_visible else body.get("labels")
        if not isinstance(requested, list) or not requested:
            raise ValueError("请选择至少一个参考类别，或启用自动识别可见类别")
        selected = []
        for label in requested:
            if not isinstance(label, str) or label not in state.get("labels", []):
                raise ValueError(f"参考类别不在当前标签表中：{label}")
            if label not in selected:
                selected.append(label)
        video = state["video"]
        config = copy.deepcopy(state["config"])
        labels = list(state["labels"])
        target_image = frame_image(index)
        input_sha256 = hashlib.sha256(target_image).hexdigest()
        file_name, selected_topic = str(video.path), state["topic"]
        config_sha256 = _sha256_json(config)
        discovery_cache_key = reference_discovery_cache_key(
            video, selected_topic, index, input_sha256, labels, config_sha256)
        existing = copy.deepcopy(state.get("candidates", {}).get(str(index), []))
        existing_run = copy.deepcopy(state.get("agent_runs", {}).get(str(index), {}))
        if defer_agent_review and existing:
            raise ValueError("已有机器候选时不能启动后台 Agent 精修；请先直接载入现有候选")
        original_candidates_sha256 = _sha256_json(existing)
        original_run_sha256 = _sha256_json(existing_run)
        audit_version = candidate_class_audit_version()
        dataset_split = _frame_dataset_split(index)
        truth_sha256 = _sha256_json(state.get("truth", {}))
        references_sha256 = _sha256_json(state.get("reference_frames", {}))
        trace = []
        with run_status_lock:
            _update_run_status_locked(status="running", frame=index, completed=0, total=1,
                phase="discovering" if auto_detect_visible else "describing", round=0, error=None,
                topic=selected_topic, file=file_name, max_rounds=max_rounds)

    debug_key = config.get("llm_key")
    def redact_snapshot(value):
        if isinstance(value, str):
            return value.replace(debug_key, "[REDACTED]") if debug_key else value
        if isinstance(value, dict):
            return {name: redact_snapshot(item) for name, item in value.items()}
        if isinstance(value, list):
            return [redact_snapshot(item) for item in value]
        return value

    def on_trace(event):
        if event.get("service") == "GPT" and event.get("status") == "waiting":
            update_run_status(phase="vlm_waiting",
                              busy_retry_count=event.get("busy_retry_count", 1),
                              retry_in_seconds=event.get("retry_in_seconds"), error=None)
        record = redact_snapshot(event | {"frame": index,
            "topic": selected_topic, "file": file_name,
            "purpose": "参考帧机器草稿"})
        existing = next((position for position, item in enumerate(trace)
                         if item["id"] == record["id"]), None)
        if existing is None:
            trace.append(record)
        else:
            trace[existing] = record
        with lock:
            save_debug(trace)

    def on_progress(event):
        if run_stop_event.is_set():
            raise RunCancelled("用户已停止参考帧草稿生成")
        record = {"time": time.time(), "frame": index,
                  "phase": event["phase"], "round": event["round"],
                  "shape_count": len(event["shapes"])}
        with run_status_lock:
            _update_run_status_locked(phase=event["phase"], round=event["round"],
                                      shape_count=record["shape_count"])
            run_status["events"] = (run_status.get("events", []) + [record])[-150:]

    try:
        with cancellation(run_stop_event), trace_calls(on_trace, save_inspection_image):
            discovery = None
            discovery_retry_count = 0
            discovery_prefetch_cache_hit = False
            if auto_detect_visible:
                prefetched_discovery = get_reference_discovery_cache(
                    discovery_cache_key)
                if prefetched_discovery is not None:
                    discovery, discovery_retry_count = prefetched_discovery
                    discovery_prefetch_cache_hit = True
                else:
                    discovery, discovery_retry_count = discover_reference_labels(
                        config, target_image, selected,
                        lambda error: update_run_status(
                            phase="discovering_retry", discovery_attempt=2,
                            error=f"首次类别结构不合规，正在自动重试：{str(error)[:300]}"))
                selected, recall_probes = candidate_labels_with_recall_probes(
                    discovery, labels,
                    probe_missing_selected=probe_missing_selected)
                discovery["recall_probe_labels"] = recall_probes
                config["other_robot_semantic_fallback_visible"] = (
                    "other_robot" in discovery.get("visible_labels", []))
                if not selected:
                    raise ValueError("VLM 未发现可生成候选的已知类别；请改用单类别起稿或点提示 SAM")
                update_run_status(phase="describing", visible_labels=selected,
                                  uncertain_labels=discovery["uncertain_labels"],
                                  unknown_candidate_count=len(discovery["unknown_candidates"]),
                                  discovery_retry_count=discovery_retry_count,
                                  discovery_prefetch_cache_hit=discovery_prefetch_cache_hit,
                                  error=None)
            if auto_detect_visible:
                # Discovery already selected a bounded set of known registry
                # labels. With no human reference geometry, a second VLM call
                # cannot add trustworthy annotation rules; use the same
                # deterministic object prompts that are the fail-safe fallback
                # for malformed unreferenced descriptions.
                description = canonical_unreferenced_description(selected)
                description_source = "deterministic_after_visible_label_discovery"
            else:
                description = describe(config, target_image, [], selected,
                                       include_unreferenced=True)
                description_source = "vlm_unreferenced_description"
            prompt_payload = description.get("prompts", [])
            discovery_sha256 = _sha256_json(discovery) if discovery is not None else None
            pipeline_config_sha256 = _sha256_json({
                "description_input_version": DESCRIPTION_INPUT_VERSION,
                "llm_model": config.get("llm_model"),
                "llm_prompt_profile": config.get("llm_prompt_profile", "full"),
                "llm_fast_policy": {
                    "enabled": bool(config.get("llm_fast_url")
                                    and config.get("llm_fast_model")),
                    "model": config.get("llm_fast_model"),
                    "max_pixels": config.get("llm_fast_max_pixels", 640 * 512),
                    "max_long_edge": config.get("llm_fast_max_long_edge", 640),
                    "max_extra_pixels": config.get(
                        "llm_fast_max_extra_pixels", 4 * 640 * 512),
                    "max_extra_long_edge": config.get(
                        "llm_fast_max_extra_long_edge", 1280),
                    "timeout_seconds": config.get("llm_fast_timeout_seconds", 60),
                    "cooldown_seconds": config.get("llm_fast_cooldown_seconds", 60),
                },
                "max_review_rounds": max_rounds,
                "sam_prompts": prompt_payload,
                "ego_bottom_point_fallback": bool(
                    config.get("ego_bottom_point_fallback")),
                "other_robot_semantic_fallback_visible": bool(
                    config.get("other_robot_semantic_fallback_visible")),
                "sam3_point_url": config.get("sam3_point_url"),
                "auto_detect_visible": auto_detect_visible,
                "probe_missing_selected": probe_missing_selected,
                "defer_class_audit": defer_class_audit,
                "defer_agent_review": defer_agent_review,
                "discovery_sha256": discovery_sha256,
                "description_source": description_source,
            })
            candidate_request_id = _sha256_json({
                "request": "reference_draft_candidate",
                "file": file_name,
                "topic": selected_topic,
                "frame": index,
                "input_sha256": input_sha256,
                "labels": selected,
                "auto_detect_visible": auto_detect_visible,
                "probe_missing_selected": probe_missing_selected,
                "discovery_sha256": discovery_sha256,
                "pipeline_config_sha256": pipeline_config_sha256,
            })
            if run_stop_event.is_set():
                raise RunCancelled("用户已停止参考帧草稿生成")
            update_run_status(phase="generating")
            runner = generate_only if defer_agent_review else run
            result = runner(config, target_image, description, selected,
                            max_rounds=max_rounds,
                            on_progress=on_progress, reference=None)
            ego_recovery = recover_reference_ego_candidate(
                config, target_image, discovery, result)
            ego_recovery["post_recovery_slope_guard"] = (
                filter_post_recovery_slope_ego_conflicts(target_image, result))
            result.setdefault("agent", {})["ego_bottom_point_fallback"] = ego_recovery
        if result["agent"].get("stop_reason") == "error":
            raise RuntimeError("SAM/VLM/Agent 未能生成有效参考帧草稿")
        candidate_output_sha256 = _sha256_json(result["shapes"])
        candidate_event_id = reference_candidate_event_id(
            candidate_request_id, candidate_output_sha256)
        for generated_shape in result["shapes"]:
            generated_shape["candidate_only"] = True
            generated_shape["human_truth"] = False
            generated_shape["approved_for_training"] = False
            generated_shape["provenance"] = {
                "kind": "machine_candidate",
                "pipeline": "sam_vlm_agent",
                "agent_review_status": ("pending" if defer_agent_review else "complete"),
                "candidate_event_id": candidate_event_id,
                "candidate_request_id": candidate_request_id,
                "candidate_output_sha256": candidate_output_sha256,
                "input_sha256": input_sha256,
                "file": file_name,
                "topic": selected_topic,
                "frame": index,
                "labels": selected,
                "auto_detect_visible": auto_detect_visible,
                "probe_missing_selected": probe_missing_selected,
                "visible_label_discovery": discovery,
                "discovery_sha256": discovery_sha256,
                "sam_prompts": prompt_payload,
                "llm_model": config.get("llm_model"),
                "prompt_profile": config.get("llm_prompt_profile", "full"),
                "description_input_version": DESCRIPTION_INPUT_VERSION,
                "description_source": description_source,
                "pipeline_config_sha256": pipeline_config_sha256,
            }
        combined, merged_id_map = merge_inference_candidates(
            existing, result["shapes"], selected, return_id_map=True)
        bind_merged_candidate_references(result, merged_id_map)
        candidate_artifact_sha256 = _sha256_json(combined)
        visibility_review_queue = reference_visibility_review_queue(discovery, combined)
        audit_total = sum(candidate_requires_class_audit(candidate)
                          for candidate in combined)
        class_audits, class_review_queue = [], []
        if audit_total and not defer_class_audit and not defer_agent_review:
            update_run_status(phase="auditing", audit_completed=0,
                              audit_total=audit_total, audit_candidate_id=None,
                              shape_count=len(combined), error=None)
            def check_audit_cancelled():
                if run_stop_event.is_set():
                    raise RunCancelled("用户已停止参考帧草稿生成")
            def on_audit_progress(completed, total, candidate, audit):
                update_run_status(
                    phase="auditing", audit_completed=completed, audit_total=total,
                    audit_candidate_id=candidate.get("id"),
                    audit_route=audit.get("route"), shape_count=len(combined), error=None)
            with cancellation(run_stop_event), trace_calls(on_trace, save_inspection_image):
                class_audits, class_review_queue = audit_high_risk_candidates(
                    config, target_image, combined, labels,
                    VISIBLE_LABEL_DEFINITIONS, chat, check_cancelled=check_audit_cancelled,
                    tolerate_errors=True, on_progress=on_audit_progress,
                    batch_chat=chat_batch)
        for audit in class_audits:
            audit["audit_version"] = audit_version
            audit["audit_sha256"] = _sha256_json({
                key: value for key, value in audit.items() if key != "audit_sha256"})
        for entry in class_review_queue:
            entry["audit_sha256"] = next(
                audit["audit_sha256"] for audit in class_audits
                if audit["candidate_id"] == entry["candidate_id"])
        visibility_review_queue.extend(class_review_queue)
        replacement_run = result["agent"] | {
            "review_queue": result["review_queue"] + visibility_review_queue,
            "candidate_class_audits": class_audits,
            "candidate_class_audit_version": audit_version,
            "candidate_class_audit_status": (
                "blocked_on_agent_review" if defer_agent_review and audit_total else
                "pending" if defer_class_audit and audit_total else "complete"),
            "candidate_class_audit_deferred": bool(
                audit_total and (defer_class_audit or defer_agent_review)),
            "candidate_class_audit_total": audit_total,
            "agent_review_status": "pending" if defer_agent_review else "complete",
            "agent_review_deferred": defer_agent_review,
            "draft_labels": selected,
            "applied": result["applied"],
            "candidates_added": result["candidates_added"],
            "reference_draft": True, "candidate_only": True,
            "human_truth": False, "approved_for_training": False,
            "dataset_split": dataset_split,
            "existing_candidates_preserved": True,
            "candidate_event_id": candidate_event_id,
            "candidate_request_id": candidate_request_id,
            "candidate_output_sha256": candidate_output_sha256,
            "candidate_artifact_sha256": candidate_artifact_sha256,
            "input_sha256": input_sha256,
            "pipeline_config_sha256": pipeline_config_sha256,
            "sam_prompts": prompt_payload,
            "auto_detect_visible": auto_detect_visible,
            "probe_missing_selected": probe_missing_selected,
            "visible_label_discovery": discovery,
            "description_source": description_source,
            "discovery_sha256": discovery_sha256,
            "discovery_retry_count": discovery_retry_count,
            "discovery_prefetch_cache_hit": discovery_prefetch_cache_hit,
            "human_review_required": True,
            "inspection_trace": copy.deepcopy(trace)}
        update_run_status(phase="committing", audit_completed=len(class_audits),
                          audit_total=audit_total, audit_candidate_id=None,
                          shape_count=len(combined), error=None)
        with lock, run_status_lock:
            if run_stop_event.is_set():
                raise RunCancelled("用户已停止参考帧草稿生成")
            if (state.get("video") is not video
                    or str(video.path) != file_name or state.get("topic") != selected_topic
                    or _sha256_json(state.get("config", {})) != config_sha256
                    or state.get("labels", []) != labels
                    or candidate_class_audit_version() != audit_version
                    or _sha256_json(state.get("truth", {})) != truth_sha256
                    or _sha256_json(state.get("reference_frames", {})) != references_sha256
                    or _frame_dataset_split(index) != dataset_split
                    or _sha256_json(state.get("candidates", {}).get(str(index), [])) != original_candidates_sha256
                    or _sha256_json(state.get("agent_runs", {}).get(str(index), {})) != original_run_sha256):
                raise StaleHumanReviewError("数据作用域、候选、人工真值或审核配置已变化，未保存旧草稿")
            revision = archive_candidate_revision(
                index, selected_topic, existing, existing_run, combined, replacement_run,
                reason="reference_draft_rerun")
            state["candidates"][str(index)] = combined
            state.setdefault("agent_runs", {})[str(index)] = replacement_run
            save()
            _update_run_status_locked(status="completed", phase="completed", frame=None,
                                      completed=1, shape_count=len(combined))
            return jsonify(frame=index, labels=selected, candidates=combined,
                           candidates_added=result["candidates_added"],
                           agent=state["agent_runs"][str(index)],
                           candidate_only=True, human_truth=False,
                           approved_for_training=False,
                           candidate_event_id=candidate_event_id,
                           candidate_request_id=candidate_request_id,
                           candidate_output_sha256=candidate_output_sha256,
                           candidate_artifact_sha256=candidate_artifact_sha256,
                           input_sha256=input_sha256,
                           pipeline_config_sha256=pipeline_config_sha256,
                           prior_candidate_revision_id=(revision.get("revision_id") if revision else None),
                           candidate_revision_count=len(state.get("candidate_revisions", {}).get(str(index), [])),
                           auto_detect_visible=auto_detect_visible,
                            probe_missing_selected=probe_missing_selected,
                           description_source=description_source,
                           visible_label_discovery=discovery,
                           discovery_sha256=discovery_sha256,
                           discovery_prefetch_cache_hit=discovery_prefetch_cache_hit,
                           agent_review_deferred=defer_agent_review,
                           agent_review_status=("pending" if defer_agent_review else "complete"),
                           candidate_class_audit_deferred=bool(
                               audit_total and (defer_class_audit or defer_agent_review)),
                           candidate_class_audit_total=audit_total,
                           human_review_required=True,
                           message=("SAM 初始草稿已生成；Agent 精修与高风险复核将在后台继续"
                                    if defer_agent_review else
                                    "SAM/VLM/Agent 草稿已生成；高风险类别复核将在后台继续"
                                    if defer_class_audit and audit_total else
                                    "SAM/VLM/Agent 草稿已生成；请人工精修并确认"))
    except RunCancelled:
        update_run_status(status="stopped", phase="stopped", frame=None,
                          error=None, completed=0)
        return jsonify(stopped=True, candidate_only=True, human_truth=False)
    except Exception as error:
        update_run_status(status="failed", phase="failed", frame=None,
                          error=str(error)[:500], completed=0)
        raise


@app.post("/api/run")
@run_claimed
def run_workflow():
    body = request.get_json() or {}
    with lock:
        if state.get("video") is None or state.get("topic") is None:
            raise ValueError("请先加载图像消息")
        expected_file = body.get("expected_file")
        expected_topic = body.get("expected_topic")
        if expected_file is not None or expected_topic is not None:
            if not isinstance(expected_file, str) or not isinstance(expected_topic, str):
                raise ValueError("自动标注作用域必须同时包含数据文件和图像源")
            if (expected_file != str(state["video"].path)
                    or expected_topic != state["topic"]):
                raise StaleHumanReviewError(
                    "预检期间数据文件或图像源已经变化，本次自动标注未启动")
        topic_references = state.get("reference_frames", {}).get(state["topic"], [])
        if not topic_references:
            raise ValueError("请先设置参考帧并保存其真值")
        reference = topic_references[0]
        topic_count = state["video"].topics[state["topic"]]["count"]
        if not state["truth"].get(str(reference)):
            raise ValueError("请先保存参考帧真值")
        max_rounds = requested_review_rounds(body)
        defer_agent_review = body.get("defer_agent_review") is True
        compare_with_truth = body.get("compare_with_truth") is True
        results = []
        requested_frames = body.get("frames")
        if requested_frames is not None:
            if (not isinstance(requested_frames, list) or not requested_frames
                    or len(requested_frames) > 100
                    or any(isinstance(item, bool) or not isinstance(item, int)
                           for item in requested_frames)):
                raise ValueError("重试帧须为 1～100 个整数帧号")
            requested_frames = list(dict.fromkeys(requested_frames))
            if any(not 0 <= item < topic_count for item in requested_frames):
                raise ValueError("重试帧号超出当前图像源范围")
            expected_result_task_id = body.get("expected_result_task_id")
            if (not isinstance(expected_result_task_id, str)
                    or not expected_result_task_id.strip()):
                raise ValueError("重试请求缺少来源任务编号；请读取最新运行结果后重试")
            expected_result_task_id = expected_result_task_id.strip()
            retry_source = load_run_summary()
            if (not retry_source
                    or retry_source.get("task_id") != expected_result_task_id
                    or retry_source.get("file") != str(state["video"].path)
                    or retry_source.get("topic") != state["topic"]):
                raise StaleHumanReviewError(
                    "重试来源任务已被更新结果替代；未启动旧重试")
            retryable_frames = set(retry_source.get("pending_frames") or [])
            retryable_frames.update(retry_source.get("failed_frames") or [])
            if any(item not in retryable_frames for item in requested_frames):
                raise StaleHumanReviewError(
                    "重试帧清单已被更新结果替代；未启动旧重试")
            targets = [target for target in requested_frames
                       if target not in topic_references
                       and (compare_with_truth or str(target) not in state["truth"])]
            requested_range = "、".join(map(str, requested_frames))
        else:
            index = int(body["frame"])
            if not 0 <= index < topic_count:
                raise ValueError("当前帧号无效")
            count = int(body.get("count", 10))
            if not 1 <= count <= 100:
                raise ValueError("处理帧数须为 1～100")
            last = min(index + count, topic_count)
            targets = [target for target in range(index, last)
                       if target not in topic_references
                       and (compare_with_truth or str(target) not in state["truth"])]
            requested_frames = list(range(index, last))
            requested_range = f"{index}～{last - 1}"
        if not targets:
            raise ValueError(
                f"所选范围（帧 {requested_range}）没有待处理帧。参考帧始终会跳过；"
                + ("请选择其他帧。" if compare_with_truth else
                   "已保存真值默认不会覆盖；请增大处理帧数，或在评测模式生成独立候选。")
            )
        storage = run_storage_readiness(len(targets))
        if not storage["ready"]:
            raise RuntimeError(storage["message"])
        allowed_labels = selected_auto_labels(reference)
        description = body.get("description")
        if description is None:
            description = load_reference_description(reference)
        supplied_prompts = description.get("prompts", []) if isinstance(description, dict) else []
        supplied_labels = [item.get("label") for item in supplied_prompts if isinstance(item, dict)]
        if (not isinstance(description, dict) or not description.get("description")
                or len(supplied_prompts) != len(allowed_labels)
                or set(supplied_labels) != set(allowed_labels)
                or description.get("reference_signature") != reference_signature(reference)):
            raise ValueError("参考帧描述缺失或已失效，请打开参考帧并点击“生成真值描述”后再开始自动标注")
        description = restrict_description(description, allowed_labels)
        description["reference_signature"] = reference_signature(reference)
        output = {"description": description, "frames": results, "current": None,
                  "trace": [], "steps": [], "file": str(state["video"].path),
                  "topic": state["topic"], "reference": reference,
                   "task_id": run_task_id, "status": "running",
                   "total": len(targets), "targets": targets,
                   "requested_frames": requested_frames,
                   "agent_review_deferred": defer_agent_review,
                   "compare_with_truth": compare_with_truth,
                  "message": ("初始机器候选已保存；Agent 将在后台继续精修，请人工复核后保存真值"
                              if defer_agent_review else
                              "候选已保存到本地；请在工作台逐帧复核并保存真值")}
        video, topic = state["video"], state["topic"]
        file_name = str(video.path)
        config = copy.deepcopy(state.get("config", {}))
        labels = copy.deepcopy(state.get("labels", []))
        auto_labels = copy.deepcopy(state.get("auto_labels"))
        config_sha256 = _sha256_json(config)
        references = copy.deepcopy(state.get("reference_frames", {}))
        reference_truth = copy.deepcopy(state["truth"][str(reference)])
        reference_signature_snapshot = description["reference_signature"]
        candidates = {str(target): copy.deepcopy(state.get("candidates", {}).get(str(target)))
                      for target in targets}
        runs = {str(target): copy.deepcopy(state.get("agent_runs", {}).get(str(target)))
                for target in targets}
        review_statuses = {
            str(target): copy.deepcopy(
                state.get("candidate_review_status", {}).get(str(target)))
            for target in targets}
        target_truth = {str(target): copy.deepcopy(state.get("truth", {}).get(str(target)))
                        for target in targets}
        failed_frames = []
        with run_status_lock:
            _update_run_status_locked(status="running", frame=None, completed=0, total=len(targets),
                              phase="starting", round=0, error=None, topic=topic, file=file_name,
                              started_at=time.time(), events=[], shape_count=0,
                              max_rounds=max_rounds)
    trace_frame = reference
    def on_trace(event):
        if event.get("service") == "GPT" and event.get("status") == "waiting":
            update_run_status(phase="vlm_waiting",
                              busy_retry_count=event.get("busy_retry_count", 1),
                              retry_in_seconds=event.get("retry_in_seconds"), error=None)
        with run_status_lock:
            phase, round_number = run_status.get("phase"), run_status.get("round", 0)
        event = redact_debug(event | {"frame": trace_frame, "round": round_number,
            "phase": phase, "topic": topic, "file": file_name,
            "purpose": "自动标注"}, credential=config.get("llm_key") or "")
        trace = output["trace"]
        existing = next((i for i, item in enumerate(trace) if item["id"] == event["id"]), None)
        if existing is None:
            trace.append(event)
        else:
            trace[existing] = event
        save_debug(trace)
        save_run(output)

    def check_cancelled():
        if run_stop_event.is_set():
            raise RunCancelled("用户已停止自动标注")

    try:
        save_run(output)
        save_debug([])
        with cancellation(run_stop_event), trace_calls(on_trace, save_inspection_image):
            check_cancelled()
            selected_reference = [shape for shape in reference_truth
                                  if shape.get("label") in allowed_labels]
            reference_image = video.frame(topic, reference) if selected_reference else None
            reference_context = ({"image": reference_image,
                "marked_image": reference_overlay(reference_image, selected_reference),
                "annotations": selected_reference} if selected_reference else None)
            for target in targets:
                trace_frame = target
                with run_status_lock:
                    check_cancelled()
                    _update_run_status_locked(frame=target, phase="loading", round=0, shape_count=0)
                target_image = video.frame(topic, target)
                def on_progress(event, target=target):
                    record = {"time": time.time(), "frame": target,
                        "phase": event["phase"], "round": event["round"],
                        "shape_count": len(event["shapes"])}
                    for field in ("label", "operation", "outcome"):
                        if event.get(field):
                            record[field] = event[field]
                    if event["phase"] == "finished":
                        record["stop_reason"] = event["result"]["agent"]["stop_reason"]
                    with run_status_lock:
                        check_cancelled()
                        _update_run_status_locked(phase=event["phase"], round=event["round"],
                                                  shape_count=record["shape_count"])
                        run_status["events"] = (run_status.get("events", []) + [record])[-150:]
                    output["current"] = {"frame": target, "phase": event["phase"],
                        "round": event["round"], "audit": event["audit"],
                        "review_queue": event["review_queue"]}
                    if event["phase"] in ("generated", "edited", "finished"):
                        output["steps"].append({"id": uuid.uuid4().hex,
                            "frame": target, "phase": event["phase"], "round": event["round"],
                            "shape_count": len(event["shapes"]),
                            "image": save_inspection_image(candidate_overlay(target_image, event["shapes"])),
                            "outcome": event.get("outcome"),
                            "result": {"agent": event["result"]["agent"]} if event.get("result") else None})
                    save_run(output)
                check_cancelled()
                transient_retry_errors = []
                runner = generate_only if defer_agent_review else run
                while True:
                    result = runner(config, target_image, description, allowed_labels,
                        max_rounds=max_rounds, on_progress=on_progress, reference=reference_context)
                    retry_reason = transient_run_failure(result)
                    if retry_reason is None or transient_retry_errors:
                        break
                    retry_error = "\n".join(
                        str(item.get("error") or item.get("reason") or "")[:500]
                        for item in result.get("review_queue") or [] if isinstance(item, dict)
                    )[:1000]
                    transient_retry_errors.append(retry_error)
                    retry_record = {"time": time.time(), "frame": target,
                                    "phase": "retrying", "round": 0,
                                    "shape_count": 0, "retry_attempt": 1,
                                    "reason": retry_reason}
                    with run_status_lock:
                        check_cancelled()
                        _update_run_status_locked(
                            phase="retrying", round=0, shape_count=0,
                            retry_attempt=1, retry_reason=retry_reason)
                        run_status["events"] = (run_status.get("events", []) +
                                                [retry_record])[-150:]
                    output["current"] = {"frame": target, "phase": "retrying",
                                         "round": 0, "audit": [],
                                         "review_queue": result.get("review_queue", []),
                                         "retry_attempt": 1}
                    save_run(output)
                check_cancelled()
                result["agent"]["transient_retry_count"] = len(transient_retry_errors)
                if transient_retry_errors:
                    result["agent"]["transient_retry_errors"] = transient_retry_errors
                key = str(target)
                existing = candidates[key] or []
                failed = result["agent"].get("stop_reason") == "error"
                if failed:
                    combined = list(existing)
                else:
                    combined, merged_id_map = merge_inference_candidates(
                        existing, result["shapes"], allowed_labels, return_id_map=True)
                    bind_merged_candidate_references(result, merged_id_map)
                candidate_artifact_sha256 = _sha256_json(combined)
                input_sha256 = hashlib.sha256(target_image).hexdigest()
                class_audits, class_review_queue = [], []
                audit_version = candidate_class_audit_version()
                pallet_audit_ids = [
                    candidate.get("id") for candidate in combined
                    if str(candidate.get("label", "")).casefold() == "pallet"
                    and candidate_requires_class_audit(candidate)
                ] if not failed else []
                candidate_class_audit_cache_hit = False
                if pallet_audit_ids and not defer_agent_review:
                    update_run_status(
                        phase="candidate_class_audit", frame=target,
                        completed=len(results), audit_completed=0,
                        audit_total=len(pallet_audit_ids))
                    candidate_id_set = set(pallet_audit_ids)
                    previous_run = runs.get(key) or {}
                    previous_review_status = review_statuses.get(key) or {}
                    if reusable_candidate_class_audit(
                            previous_run, previous_review_status, combined, labels,
                            candidate_artifact_sha256, input_sha256, audit_version,
                            candidate_ids=candidate_id_set,
                            require_review_bindings=True):
                        class_audits = copy.deepcopy(
                            previous_run.get("candidate_class_audits", []))
                        class_review_queue = candidate_class_review_queue(
                            class_audits, candidate_artifact_sha256)
                        candidate_class_audit_cache_hit = True
                        update_run_status(
                            phase="candidate_class_audit", frame=target,
                            audit_completed=len(class_audits),
                            audit_total=len(pallet_audit_ids), audit_cache_hit=True)
                    else:
                        class_audits, class_review_queue = audit_high_risk_candidates(
                            config, target_image, combined, labels,
                            VISIBLE_LABEL_DEFINITIONS, chat,
                            check_cancelled=check_cancelled,
                            tolerate_errors=True,
                            candidate_ids=candidate_id_set,
                            on_progress=lambda completed, total, candidate, audit: update_run_status(
                                phase="candidate_class_audit", frame=target,
                                audit_completed=completed, audit_total=total),
                            batch_chat=chat_batch)
                        for audit in class_audits:
                            audit["audit_version"] = audit_version
                            audit["audit_sha256"] = _sha256_json({
                                field: value for field, value in audit.items()
                                if field != "audit_sha256"})
                        for entry in class_review_queue:
                            entry["audit_sha256"] = next(
                                audit["audit_sha256"] for audit in class_audits
                                if audit["candidate_id"] == entry["candidate_id"])
                final_review_queue = list(result["review_queue"]) + class_review_queue
                class_review_status = {
                    audit["candidate_sha256"]: {
                        "candidate_fingerprint": audit["candidate_sha256"],
                        "candidate_artifact_sha256": candidate_artifact_sha256,
                        "audit_sha256": audit["audit_sha256"],
                        "route": audit["route"],
                    }
                    for audit in class_audits
                }
                replacement_run = result["agent"] | {
                    "review_queue": final_review_queue, "applied": result["applied"],
                    "candidates_added": 0 if failed else result["candidates_added"],
                    "existing_candidates_preserved": failed,
                    "agent_review_status": ("pending" if defer_agent_review and not failed
                                            else "complete"),
                    "agent_review_deferred": bool(defer_agent_review and not failed),
                    "draft_labels": copy.deepcopy(allowed_labels),
                    "candidate_artifact_sha256": candidate_artifact_sha256,
                    "input_sha256": input_sha256,
                    "human_review_required": True,
                    "candidate_class_audits": class_audits,
                    "candidate_class_audit_version": audit_version,
                    "candidate_class_audit_status": (
                        "blocked_on_agent_review" if defer_agent_review and not failed
                        else "complete"),
                    "candidate_class_audit_deferred": bool(
                        defer_agent_review and not failed),
                    "candidate_class_audit_total": len(pallet_audit_ids),
                    "candidate_class_audit_scope": "solid_pallet_after_agent",
                    "candidate_class_audit_cache_hit": candidate_class_audit_cache_hit,
                    "inspection_trace": copy.deepcopy([
                        item for item in output["trace"]
                        if item.get("frame") == target
                        and item.get("purpose") != "生成真值描述"]),
                }
                if defer_agent_review and not failed:
                    replacement_run.update(
                        review_reference_frame=reference,
                        review_reference_signature=reference_signature_snapshot,
                        review_description_sha256=_sha256_json(description),
                        candidate_class_audit_status="blocked_on_agent_review",
                        candidate_class_audit_deferred=True,
                        candidate_class_audits=[],
                    )
                with lock, run_status_lock:
                    check_cancelled()
                    if (state.get("video") is not video or str(video.path) != file_name
                            or state.get("topic") != topic
                            or _sha256_json(state.get("config", {})) != config_sha256
                            or state.get("labels", []) != labels
                            or state.get("auto_labels") != auto_labels
                            or state.get("reference_frames", {}) != references
                            or state.get("truth", {}).get(str(reference)) != reference_truth
                            or state.get("truth", {}).get(key) != target_truth[key]
                            or state.get("candidates", {}).get(key) != candidates[key]
                            or state.get("agent_runs", {}).get(key) != runs[key]
                            or state.get("candidate_review_status", {}).get(key)
                            != review_statuses[key]
                            or reference_signature(reference) != reference_signature_snapshot):
                        raise StaleHumanReviewError("数据作用域、配置、标签、参考真值或目标帧已变化，未保存旧推理结果")
                    if failed:
                        failed_frames.append(target)
                    else:
                        state.setdefault("candidates", {})[key] = combined
                    state.setdefault("agent_runs", {})[key] = replacement_run
                    state.setdefault("candidate_review_status", {})[key] = class_review_status
                    save()
                    results.append({"frame": target,
                        "candidates_added": 0 if failed else result["candidates_added"],
                        "remaining": len(combined), "applied": result["applied"],
                        "review_queue": final_review_queue, "agent": result["agent"],
                        "failed": failed, "existing_candidates_preserved": failed,
                        "agent_review_status": replacement_run["agent_review_status"],
                        "agent_review_deferred": replacement_run["agent_review_deferred"],
                        "candidate_artifact_sha256": candidate_artifact_sha256,
                        "input_sha256": input_sha256,
                        "candidate_class_audit_cache_hit":
                            candidate_class_audit_cache_hit})
                    output["current"] = None
                    save_run(output)
                    _update_run_status_locked(completed=len(results), phase="saved")
        output["failed_frames"] = failed_frames
        with run_status_lock:
            check_cancelled()
            if failed_frames:
                output["message"] = (f"{len(failed_frames)}/{len(results)} 帧失败；失败帧旧候选已保留，"
                                     "请查看错误后重试")
                output["status"] = "failed"
                _update_run_status_locked(status="failed", phase="failed", frame=None,
                                          error=output["message"])
            else:
                output["status"] = "completed"
                _update_run_status_locked(status="completed", phase="completed", frame=None)
        save_run(output)
        return jsonify(output)
    except RunCancelled:
        output["current"] = None
        output["status"] = "stopped"
        output["message"] = "用户已停止自动标注；已完成帧的候选已保存"
        save_run(output)
        with run_status_lock:
            _update_run_status_locked(status="stopped", phase="stopped", frame=None,
                                      error=None, completed=len(results))
        return jsonify(output | {"stopped": True})
    except Exception as error:
        output["status"] = "failed"
        output["error"] = str(error)[:500]
        save_run(output)
        with run_status_lock:
            if run_stop_event.is_set():
                _update_run_status_locked(status="stopped", phase="stopped",
                                          error="用户已停止自动标注")
            else:
                _update_run_status_locked(status="failed", phase="failed",
                                          error=str(error)[:500])
        raise


@app.post("/api/run/stop")
def stop_workflow():
    body = request.get_json(silent=True) or {}
    expected_task_id = body.get("expected_task_id")
    if not isinstance(expected_task_id, str) or not expected_task_id.strip():
        raise ValueError("停止请求缺少任务编号；请刷新任务状态后重试")
    expected_task_id = expected_task_id.strip()
    with run_status_lock:
        active_task_id = run_status.get("task_id")
        if (run_status.get("status") in ("running", "stopping")
                and expected_task_id != active_task_id):
            raise StaleHumanReviewError("停止请求对应的任务已经更新；未停止当前新任务")
        if run_status.get("status") not in ("running", "stopping"):
            return jsonify(status=run_status.get("status", "idle"), stopped=False,
                           task_id=active_task_id)
        if run_status.get("status") == "stopping":
            return jsonify(status="stopping", stopped=True, task_id=active_task_id)
        run_stop_event.set()
        _update_run_status_locked(status="stopping", phase="stopping")
    return jsonify(status="stopping", stopped=True, task_id=active_task_id)


@app.get("/api/run/status")
def get_run_status():
    with run_status_lock:
        snapshot = copy.deepcopy(run_status)
    observed_at = time.time()
    terminal = snapshot.get("status") in ("completed", "failed", "stopped")
    elapsed_until = snapshot.get("updated_at") if terminal else observed_at
    if not isinstance(elapsed_until, (int, float)):
        elapsed_until = observed_at
    started_at = snapshot.get("started_at")
    phase_started_at = snapshot.get("phase_started_at") or started_at
    snapshot.update(
        observed_at=observed_at,
        elapsed_seconds=max(0.0, elapsed_until - started_at) if started_at else 0.0,
        phase_elapsed_seconds=max(0.0, elapsed_until - phase_started_at)
        if phase_started_at else 0.0,
        progress_contract_version=2,
    )
    with lock:
        selected_file = str(state["video"].path) if state.get("video") else ""
        selected_topic = state.get("topic")
    snapshot.update(selected_file=selected_file, selected_topic=selected_topic)
    if snapshot.get("status") not in ("running", "stopping"):
        summary = load_run_summary()
        same_scope = (summary and summary.get("file") == selected_file
                      and summary.get("topic") == selected_topic)
        same_task = (not snapshot.get("task_id")
                     or summary and summary.get("task_id") == snapshot.get("task_id"))
        resumable_interruption = (summary and summary.get("status") == "running"
                                  and summary.get("pending_frames"))
        if same_scope and same_task and (summary.get("frames") or resumable_interruption):
            snapshot.update(
                result_available=True,
                result_task_id=summary.get("task_id"),
                result_status=summary.get("status"),
                result_completed=summary.get("completed", 0),
                result_total=summary.get("total", 0),
                failed_frames=summary.get("failed_frames", []),
                pending_frames=summary.get("pending_frames", []),
                result_interrupted=bool(resumable_interruption),
            )
    return jsonify(snapshot)


@app.get("/api/run/result")
def get_run_result():
    summary = load_run_summary()
    if summary is None:
        return jsonify(error="没有可恢复的运行结果"), 404
    requested_task = request.args.get("task_id")
    if requested_task and requested_task != summary.get("task_id"):
        raise StaleHumanReviewError("请求的任务已被更新结果替代")
    with lock:
        selected_file = str(state["video"].path) if state.get("video") else ""
        selected_topic = state.get("topic")
    if summary.get("file") != selected_file or summary.get("topic") != selected_topic:
        raise StaleHumanReviewError("运行结果不属于当前数据文件和图像源")
    return jsonify(summary)


@app.get("/api/run/debug")
def get_run_debug():
    path = WORK / "model_debug.json"
    if not path.exists():
        path = WORK / "last_run.json"
    if not path.exists():
        return jsonify(trace=[])
    output = json.loads(path.read_text(encoding="utf-8"))
    return jsonify(trace=output.get("trace", []))


@app.get("/api/run/inspection")
def get_inspection():
    def read(name):
        path = WORK / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    output = read("last_run.json")
    description_record = read("description_debug.json")
    legacy_trace = not output.get("file") and not output.get("topic")
    model_trace = read("model_debug.json").get("trace", [])
    if not output.get("trace"):
        output["trace"] = model_trace
    selected_topic = request.args.get("topic", state.get("topic"))
    selected_file = str(state["video"].path) if state["video"] else ""
    def belongs(item):
        return item.get("topic") == selected_topic and item.get("file") == selected_file
    trace = [item for item in output.get("trace", []) if belongs(item)]
    description_trace = [item for item in description_record.get("trace", []) if belongs(item)]
    live_descriptions = [item for item in read("model_debug.json").get("trace", [])
                         if belongs(item) and item.get("purpose") == "生成真值描述"]
    if live_descriptions and max(item.get("started_at", 0) for item in live_descriptions) >= max(
            (item.get("started_at", 0) for item in description_trace), default=0):
        description_trace = live_descriptions
    description = output.get("description") if belongs(output) or legacy_trace and trace else None
    saved_description = description_record.get("description", {})
    reference = state.get("reference_frames", {}).get(selected_topic, [])
    if (state["video"] and selected_topic == state["topic"] and reference and
            saved_description.get("reference_signature") == reference_signature(reference[0])):
        description = saved_description
    elif not description and description_trace:
        description = saved_description
    trace = description_trace + [item for item in trace if item.get("purpose") != "生成真值描述"
                                 or not description_trace]
    steps = output.get("steps", []) if belongs(output) or legacy_trace and trace else []
    frames = sorted({item["frame"] for item in trace if item.get("purpose") != "生成真值描述"}
                    | {item["frame"] for item in steps})
    if request.args.get("frame") is not None:
        selected_frame = int(request.args["frame"])
        frame_calls = [item for item in trace if item.get("frame") == selected_frame
                       and item.get("purpose") != "生成真值描述"]
        has_frame_result = (belongs(output) or legacy_trace and trace) and any(
            item.get("frame") == selected_frame for item in output.get("frames", []))
        has_frame_state = selected_topic == state.get("topic") and (
            str(selected_frame) in state.get("candidates", {})
            or str(selected_frame) in state.get("agent_runs", {}))
        # Keep the newest scope-bound trace among the latest batch, the trace
        # persisted with this frame, and the live debug stream.  Persisted
        # per-frame traces prevent a later run from making older GPT tabs look
        # empty; requiring live machine state keeps deleted frames deleted.
        persisted_frame_calls = [
            item for item in state.get("agent_runs", {}).get(
                str(selected_frame), {}).get("inspection_trace", [])
            if belongs(item) and item.get("frame") == selected_frame
            and item.get("purpose") != "生成真值描述"]
        live_frame_calls = [item for item in model_trace if belongs(item)
                            and item.get("frame") == selected_frame
                            and item.get("purpose") != "生成真值描述"]
        if has_frame_state:
            frame_calls = max(
                (frame_calls, persisted_frame_calls, live_frame_calls),
                key=lambda items: max(
                    (item.get("started_at", 0) for item in items), default=0))
        show_description = (selected_frame in reference or bool(frame_calls)
                            or any(item.get("frame") == selected_frame for item in steps)
                            or has_frame_result or has_frame_state)
        trace = ([item for item in trace if item.get("purpose") == "生成真值描述"
                  and item.get("frame") == selected_frame]
                 if selected_frame in reference else []) + frame_calls
        if not show_description:
            description = None
        steps = [item for item in steps if item["frame"] == selected_frame]
    with run_status_lock:
        job = {key: run_status.get(key) for key in
               ("status", "phase", "frame", "round", "shape_count", "completed", "total")} if belongs(run_status) else {}
    clear_binding = None
    if request.args.get("frame") is not None:
        with lock:
            current_video = state.get("video")
            if (current_video is not None and selected_topic == state.get("topic")
                    and selected_file == str(current_video.path)):
                clear_binding = auto_result_clear_binding(selected_frame)
                clear_binding.update(
                    path=selected_file, topic=selected_topic, frame=selected_frame,
                    file_activation_generation=current_file_activation_generation())
    return jsonify(trace=trace, steps=steps, description=description,
                   topic=selected_topic, file=selected_file, reference=output.get("reference"), frames=frames,
                   job=job,
                   clear_auto_result_binding=clear_binding,
                   active_frame=output.get("current", {}).get("frame") if output.get("current") and
                       (belongs(output) or legacy_trace and trace) else None)


def main():
    load()
    if state.get("video_path") and Path(state["video_path"]).is_file():
        state["video"] = Video(state["video_path"])
    port = int(os.environ.get("AUTOLABEL_PORT", "8765"))
    print(f"Autolabel: http://127.0.0.1:{port}", flush=True)
    app.run(host="127.0.0.1", port=port, threaded=True)


if __name__ == "__main__":
    main()
