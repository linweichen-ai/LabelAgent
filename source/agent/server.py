"""Flask workbench for MCAP review and local annotation."""
import io
import copy
import json
import math
import os
import threading
import time
import uuid
import urllib.request
import urllib.error
import base64
import hashlib
import marshal
from functools import wraps
from datetime import datetime, timezone
from pathlib import Path
from werkzeug.utils import secure_filename
from flask import Flask, jsonify, request, send_file
from PIL import Image, ImageDraw
import numpy as np

from .media import IMAGE_SUFFIXES, SUPPORTED_SUFFIXES, open_media as Video, read_image
from .sensors import cloud_snapshot
from .candidate_import import (prepare_candidate_import, merge_candidate_import,
                               merge_candidate_labels)
from .workflow import (DESCRIPTION_INPUT_VERSION, MAX_REVIEW_ROUNDS, RunCancelled,
                       cancellation, describe, discover_visible_labels, reference_labels, restrict_description,
                       run, trace_calls, candidate_overlay, reference_overlay, chat,
                       VISIBLE_LABEL_DEFINITIONS)
from .candidate_class_audit import (audit_candidate_class_blind, candidate_fingerprint,
                                    audit_high_risk_candidates, HIGH_RISK_LABELS)

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
run_status_lock = threading.Lock()
run_claim_lock = threading.Lock()
run_status = {"status": "idle"}
run_stop_event = threading.Event()
run_task_id = None
file_sessions_dirty = False
DEFAULT_REVIEW_ROUNDS = 6


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
    "超时", "连接被重置", "连接被拒绝", "连接中断", "暂时不可用", "服务不可用",
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


def _write_json(path, value, *, pretty=False, sort_keys=False):
    # Stream JSON directly to disk. State is deliberately compact because dense
    # polygon points make pretty-printed workspaces several times larger.
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=sort_keys,
                  indent=2 if pretty else None,
                  separators=None if pretty else (",", ":"))


def save():
    global file_sessions_dirty
    sessions = state.get("file_sessions", {})
    sessions_path = file_sessions_path()
    if file_sessions_dirty or (sessions and not sessions_path.exists()):
        sessions_temporary = sessions_path.with_suffix(".tmp")
        _write_json(sessions_temporary, sessions)
        _commit_work_file(sessions_temporary, sessions_path)
        file_sessions_dirty = False
    temporary = state_path().with_suffix(".tmp")
    _write_json(temporary, {k: v for k, v in state.items()
                            if k not in {"video", "file_sessions"}})
    _commit_work_file(temporary, state_path())


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


def save_run(output):
    path = WORK / "last_run.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    _commit_work_file(temporary, path)


def save_debug(trace):
    path = WORK / "model_debug.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"trace": trace}, ensure_ascii=False, indent=2), encoding="utf-8")
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
    global file_sessions_dirty
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


def topic_description_ready(topic, reference, truth_by_frame):
    reference_truth = truth_by_frame.get(str(reference))
    if not isinstance(reference_truth, list) or not reference_truth:
        return False
    try:
        signature = reference_signature_for(reference, topic, reference_truth)
        if topic == state.get("topic"):
            return load_reference_description(reference) is not None
        path = description_cache_path(signature)
        if not path.exists():
            return False
        saved = _read_json(path)
        return (isinstance(saved, dict)
                and saved.get("reference_signature") == signature)
    except (OSError, ValueError, TypeError, KeyError):
        return False


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
        description_ready = bool(reference is not None and truth_present
            and topic_description_ready(topic, reference, truth_by_frame))
        if reference is None:
            code = "reference_missing"
        elif not truth_present:
            code = "reference_truth_missing"
        elif not description_ready:
            code = "description_missing"
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
    }


def compact_topic_payload(index, description):
    payload = selected_frame_payload(index)
    payload.update({
        "compact_geometry": True,
        "path": str(state["video"].path),
        "topic": state["topic"],
        "frame": index,
        "truth_frames": sorted(state.get("truth", {}), key=lambda value: int(value)),
        "candidate_counts": {
            str(frame): len(items) if isinstance(items, list) else 0
            for frame, items in state.get("candidates", {}).items()
        },
        "reference_frames": copy.deepcopy(state.get("reference_frames", {})),
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
            if key != "llm_key"}),
        "protocol_sha256": _CANDIDATE_AUDIT_LOADED_CODE_SHA256,
        "labels_sha256": _sha256_json({"labels": state.get("labels", []),
                                       "definitions": VISIBLE_LABEL_DEFINITIONS}),
    }


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
        path=str(state["video"].path) if state["video"] else "",
        catalog=state["video"].catalog if state["video"] else {}, duration=state["video"].duration if state["video"] else 0,
        topics=state["video"].topics if state["video"] else {}, topic=state["topic"],
        labels=state["labels"], auto_labels=auto_labels, reference_only_labels=reference_only_labels,
        truth=state["truth"], truth_review_status=truth_review_status,
        reference_frames=state.get("reference_frames", {}),
        config={k: v for k, v in state["config"].items() if k != "llm_key"},
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
        if requested_topic != state["topic"]:
            raise StaleHumanReviewError("相机已变化，拒绝返回旧帧标注")
        if requested_path != str(state["video"].path):
            raise StaleHumanReviewError("数据文件已变化，拒绝返回旧帧标注")
        count = state["video"].topics[state["topic"]]["count"]
        if not 0 <= index < count:
            raise ValueError("帧号无效")
        payload = selected_frame_payload(index)
        payload.update(path=str(state["video"].path), topic=state["topic"],
                       frame=index, compact_geometry=True)
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


@app.post("/api/segment_points")
@run_claimed
def get_point_segment():
    body = request.get_json()
    with lock:
        if state["video"] is None or not state["topic"]:
            raise ValueError("请先加载图像消息")
        index = int(body["frame"])
        if index in state.get("reference_frames", {}).get(state["topic"], []):
            raise ValueError("参考帧只能人工调整，不能调用 SAM 生成或修改")
        image = frame_image(index)
        service_url = (state["config"].get("sam3_url") or os.environ.get("SAM3_URL")
                       or "http://127.0.0.1:8766").rstrip("/")
    if not point_segment_lock.acquire(blocking=False):
        raise RuntimeError("已有点提示 SAM 请求正在计算，请勿重复点击")
    payload = json.dumps({"image": base64.b64encode(image).decode("ascii"),
                          "points": body["points"]}).encode()
    service_request = urllib.request.Request(service_url + "/segment_points", data=payload,
        headers={"Content-Type": "application/json"})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(service_request, timeout=POINT_SEGMENT_TIMEOUT_SECONDS) as response:
            result = json.load(response)
        result["inference_elapsed_seconds"] = round(time.monotonic() - started, 2)
        return jsonify(result)
    except Exception as error:
        raise RuntimeError(f"SAM 3 服务不可用（{service_url}）：{error}") from error
    finally:
        point_segment_lock.release()

@app.post("/api/edit_region")
def edit_region():
    """Union or subtract a user drawn polygon from an existing annotation."""
    body = request.get_json() or {}
    index = int(body.get("frame", -1))
    shape = body.get("shape") or {}
    polygon = body.get("polygon") or []
    operation = body.get("operation")
    if operation not in ("add", "erase") or len(polygon) < 3:
        raise ValueError("区域至少需要 3 个点")
    image = Image.open(io.BytesIO(frame_image(index)))
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
    return jsonify(label=shape.get("label", ""), points=parts[0]["points"], holes=parts[0]["holes"], parts=parts)


@app.get("/api/playback")
def get_playback():
    if state["video"] is None:
        raise ValueError("请先打开 MCAP")
    topic = request.args.get("topic", state["topic"])
    data = state["video"].playback(topic, WORK / "playback")
    data['offset'] = state["video"].catalog[topic].get('offset', 0)
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
    if path.suffix.lower() in IMAGE_SUFFIXES:
        read_image(path)
    files = read_library()
    existing = next((item for item in files if item['path'] == str(path)), None)
    if existing:
        return existing
    item = dict(id=uuid.uuid4().hex, path=str(path), name=name or path.name, size=path.stat().st_size)
    files.append(item)
    write_library(files)
    return item


def file_list():
    current = str(state['video'].path.resolve()) if state['video'] else None
    return [dict(item, available=Path(item['path']).is_file(), loaded=item['path'] == current) for item in read_library()]


def activate_file(item):
    global file_sessions_dirty
    video = Video(item['path'])
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
    save()
    return dict(path=str(video.path), topics=video.topics, catalog=video.catalog, duration=video.duration,
                topic=state['topic'], truth=state['truth'],
                truth_review_status=current_truth_review_status(),
                candidates=state.get('candidates', {}),
                agent_runs=state.get('agent_runs', {}),
                candidate_revision_summary=candidate_revision_summary(),
                reference_frames=state.get('reference_frames', {}))


@app.get('/api/files')
def get_files():
    with lock:
        if state['video']:
            register_file(state['video'].path)
        return jsonify(files=file_list())


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
    with lock:
        item = next((item for item in read_library() if item['id'] == file_id), None)
        if item is None:
            raise ValueError('文件不在列表中')
        return jsonify(**activate_file(item))


@app.post('/api/candidates/import')
def import_candidates():
    body = request.get_json()
    with lock:
        with run_status_lock:
            if run_status.get("status") in {"running", "stopping"}:
                raise ValueError("自动任务运行中，暂不能导入候选")
        video = state.get("video")
        if video is None or not body.get("path") or Path(body["path"]).resolve() != video.path.resolve():
            raise ValueError("请先加载与候选对应的文件")
        prepared = prepare_candidate_import(video, body.get("candidates_by_topic"), state.get("labels", []))
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
        return jsonify(imported_frames=sum(len(frames) for frames in prepared.values()),
            imported_shapes=sum(len(shapes) for frames in prepared.values() for shapes in frames.values()),
            saved_as_truth=False)


@app.post('/api/candidates/merge')
def merge_candidates():
    """Merge an explicitly authorized label subset without replacing other candidates."""
    body = request.get_json()
    with lock:
        with run_status_lock:
            if run_status.get("status") in {"running", "stopping"}:
                raise ValueError("自动任务运行中，暂不能合并候选")
        video = state.get("video")
        if video is None or not body.get("path") or Path(body["path"]).resolve() != video.path.resolve():
            raise ValueError("请先加载与候选对应的文件")
        selected = [str(label).strip() for label in body.get("labels", [])
                    if str(label).strip()]
        if (not selected or len(selected) != len(set(selected))
                or any(label not in state.get("labels", []) for label in selected)):
            raise ValueError("候选合并类别无效")
        prepared = prepare_candidate_import(
            video, body.get("candidates_by_topic"), state.get("labels", []))
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
        return jsonify(merged_frames=sum(len(frames) for frames in prepared.values()),
            merged_shapes=sum(len(shapes) for frames in prepared.values() for shapes in frames.values()),
            labels=selected, preserved_other_labels=True, saved_as_truth=False)


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
    with lock:
        files = read_library()
        item = next((item for item in files if item['id'] == file_id), None)
        if item is None:
            raise ValueError('文件不在列表中')
        write_library([entry for entry in files if entry['id'] != file_id])
        unloaded = state['video'] is not None and str(state['video'].path.resolve()) == item['path']
        if unloaded:
            sessions = state.setdefault('file_sessions', {})
            sessions[item['path']] = {key: state.get(key) for key in ('topic', 'truth', 'truths_by_topic', 'reference_frames', 'candidates', 'candidates_by_topic', 'agent_runs', 'agent_runs_by_topic', 'candidate_revisions', 'candidate_revisions_by_topic')}
            file_sessions_dirty = True
            state.update(video=None, video_path=None, topic=None, truth={}, truths_by_topic={}, reference_frames={}, candidates={}, candidates_by_topic={}, agent_runs={}, agent_runs_by_topic={}, candidate_revisions={}, candidate_revisions_by_topic={})
            save()
        return jsonify(files=file_list(), unloaded=unloaded)


@app.post("/api/open")
def open_video():
    with lock:
        return jsonify(**activate_file(register_file(request.get_json()['path'])))


@app.post("/api/upload")
def upload_video():
    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename or Path(uploaded.filename).suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError("请选择 JPG、JPEG、PNG 图片或 MCAP 文件")
    destination = WORK / "uploads"
    destination.mkdir(exist_ok=True)
    suffix = Path(uploaded.filename).suffix.lower()
    filename = (secure_filename(Path(uploaded.filename).stem) or "image") + suffix
    path = destination / f"{uuid.uuid4().hex}-{filename}"
    uploaded.save(path)
    try:
        with lock:
            item = register_file(path, uploaded.filename)
            return jsonify(file=item, files=file_list())
    except Exception:
        path.unlink(missing_ok=True)
        raise


@app.post("/api/topic")
def set_topic():
    body = request.get_json()
    with lock:
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
        return jsonify(topic=state["topic"], truth=state["truth"],
                       truth_review_status=current_truth_review_status(),
                       candidates=state["candidates"], agent_runs=state["agent_runs"],
                       candidate_revision_summary=candidate_revision_summary(),
                       candidate_review_proposals=state.get("candidate_review_proposals", []),
                       reference_frames=state.get("reference_frames", {}),
                       description=description,
                       description_cache_hit=description is not None)


@app.post("/api/labels")
def set_labels():
    body = request.get_json()
    with lock:
        labels = [x.strip() for x in body["labels"] if x.strip()]
        if len(set(labels)) != len(labels) or not labels:
            raise ValueError("类别不能为空或重复")
        state["labels"] = labels
        if state.get("auto_labels"):
            state["auto_labels"] = [label for label in labels if label in state["auto_labels"]]
        save()
        return jsonify(labels=labels)


@app.post("/api/auto_labels")
def set_auto_labels():
    body = request.get_json() or {}
    with lock:
        labels = [str(label).strip() for label in body.get("labels", []) if str(label).strip()]
        if not labels or len(labels) != len(set(labels)) or any(label not in state["labels"] for label in labels):
            raise ValueError("请选择有效且不重复的推理类别")
        state["auto_labels"] = [label for label in state["labels"] if label in labels]
        save()
        return jsonify(labels=state["auto_labels"], description_invalidated=True)


@app.post("/api/truth")
def set_truth():
    body = request.get_json() or {}
    with lock:
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
        if body.get("topic") != state["topic"]:
            raise ValueError("图像源已切换，请刷新页面后重试")
        if not 0 <= index < state["video"].topics[state["topic"]]["count"]:
            raise ValueError("帧号无效")
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
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
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
        return jsonify(frame=index, candidate_removed=removed_candidate, audit_removed=removed_audit,
                       truth_preserved=key in state["truth"])

@app.post("/api/reference_frame")
def set_reference_frame():
    body = request.get_json() or {}
    with lock:
        index = int(body.get("frame", -1))
        if state["video"] is None or state["topic"] is None or not 0 <= index < state["video"].topics[state["topic"]]["count"]:
            raise ValueError("帧号无效")
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
        return jsonify(frame=index, topic=state["topic"])


@app.post("/api/config")
def set_config():
    body = request.get_json()
    if "llm_prompt_profile" in body and body["llm_prompt_profile"] not in ("full", "compact"):
        raise ValueError("Unknown model prompt profile")
    with lock:
        for key in ("llm_url", "llm_model", "llm_key", "llm_prompt_profile", "sam3_url"):
            if key in body and body[key]:
                state["config"][key] = body[key]
        save()
        return jsonify(saved=True)


@app.get("/api/model_readiness")
def model_readiness():
    with lock:
        config = dict(state.setdefault("config", {}))
    sam_url = (config.get("sam3_url") or os.environ.get("SAM3_URL")
               or "http://127.0.0.1:8766").rstrip("/")
    llm_url = str(config.get("llm_url") or "").rstrip("/")
    configured_model = str(config.get("llm_model") or "").strip()
    sam = {"ready": False, "status": "unavailable", "model_loaded": False,
           "on_demand": False}
    vlm = {"ready": False, "status": "unavailable", "model": configured_model,
           "registered_models": []}
    try:
        health = _fetch_readiness_json(sam_url + "/health")
        sam.update(status=health.get("status", "unknown"),
                   model_loaded=bool(health.get("model_loaded")),
                   on_demand=bool(health.get("on_demand")))
        sam["ready"] = health.get("status") == "ok"
    except RuntimeError as error:
        sam["error"] = str(error)
    if not llm_url:
        vlm["error"] = "尚未配置 VLM 地址"
    elif not configured_model:
        vlm["error"] = "尚未配置 VLM 模型名"
    else:
        try:
            listing = _fetch_readiness_json(llm_url + "/models")
            entries = listing.get("data") if isinstance(listing.get("data"), list) else []
            models = sorted({str(item.get("id")) for item in entries
                             if isinstance(item, dict) and item.get("id")})
            vlm.update(status="ok", registered_models=models,
                       ready=configured_model in models)
            if not vlm["ready"]:
                vlm["error"] = (f"配置模型 {configured_model} 未在服务中登记"
                                if models else "VLM 服务没有返回可用模型")
        except RuntimeError as error:
            vlm["error"] = str(error)
    return jsonify(ready=sam["ready"] and vlm["ready"], sam=sam, vlm=vlm,
                   checked_at=datetime.now(timezone.utc).isoformat())


@app.get("/api/run/readiness")
def run_readiness():
    """Validate local workflow prerequisites before a long model request starts."""
    with lock:
        if state.get("video") is None or not state.get("topic"):
            return jsonify(ready=False, code="data_missing",
                           message="请先加载图像或 MCAP 数据")
        references = state.get("reference_frames", {}).get(state["topic"], [])
        if not references:
            return jsonify(ready=False, code="reference_missing",
                           message="请先人工确认并锁定一个参考帧")
        reference = references[0]
        if not state.get("truth", {}).get(str(reference)):
            return jsonify(ready=False, code="reference_truth_missing",
                           reference_frame=reference,
                           message=f"参考帧 {reference} 尚无人工真值")
        try:
            allowed_labels = selected_auto_labels(reference)
        except (KeyError, ValueError) as error:
            return jsonify(ready=False, code="label_selection_invalid",
                           reference_frame=reference, message=str(error))
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
                                    f"请打开参考帧 {reference} 并生成真值描述"))
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
                           message=f"参考帧 {reference} 的描述已失效，请重新生成")
        return jsonify(ready=True, code="ready", reference_frame=reference,
                       topic=state["topic"], labels=allowed_labels,
                       description_ready=True)


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
        ids = [item.get("id") for item in candidates if item.get("label") in HIGH_RISK_LABELS]
        if len(ids) != len(set(ids)):
            raise ValueError("高风险候选 ID 重复，拒绝复审")
        update_run_status(status="stopping" if run_stop_event.is_set() else "running", phase="candidate_class_audit", frame=index,
                          completed=0, total=len(ids), topic=topic, error=None)
        def check_cancelled():
            if run_stop_event.is_set():
                raise RunCancelled("用户已停止候选盲审")
        audit_version = candidate_class_audit_version()
        try:
            with cancellation(run_stop_event):
                audits, queue = audit_high_risk_candidates(
                    state["config"], image_bytes, candidates, list(state.get("labels", [])),
                    VISIBLE_LABEL_DEFINITIONS, chat, check_cancelled=check_cancelled,
                    tolerate_errors=False)
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
                       candidate_only=True, human_truth=False,
                       approved_for_training=False)


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
        existing = copy.deepcopy(state.get("candidates", {}).get(str(index), []))
        existing_run = copy.deepcopy(state.get("agent_runs", {}).get(str(index), {}))
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
            if auto_detect_visible:
                discovery, discovery_retry_count = discover_reference_labels(
                    config, target_image, selected,
                    lambda error: update_run_status(
                        phase="discovering_retry", discovery_attempt=2,
                        error=f"首次类别结构不合规，正在自动重试：{str(error)[:300]}"))
                selected = discovery.get("candidate_labels") or list(dict.fromkeys(
                    discovery["visible_labels"] + discovery["uncertain_labels"]))
                if not selected:
                    raise ValueError("VLM 未发现可生成候选的已知类别；请改用单类别起稿或点提示 SAM")
                update_run_status(phase="describing", visible_labels=selected,
                                  uncertain_labels=discovery["uncertain_labels"],
                                  unknown_candidate_count=len(discovery["unknown_candidates"]),
                                  discovery_retry_count=discovery_retry_count, error=None)
            description = describe(config, target_image, [], selected,
                                   include_unreferenced=True)
            prompt_payload = description.get("prompts", [])
            discovery_sha256 = _sha256_json(discovery) if discovery is not None else None
            pipeline_config_sha256 = _sha256_json({
                "description_input_version": DESCRIPTION_INPUT_VERSION,
                "llm_model": config.get("llm_model"),
                "llm_prompt_profile": config.get("llm_prompt_profile", "full"),
                "max_review_rounds": max_rounds,
                "sam_prompts": prompt_payload,
                "auto_detect_visible": auto_detect_visible,
                "discovery_sha256": discovery_sha256,
            })
            candidate_request_id = _sha256_json({
                "request": "reference_draft_candidate",
                "file": file_name,
                "topic": selected_topic,
                "frame": index,
                "input_sha256": input_sha256,
                "labels": selected,
                "auto_detect_visible": auto_detect_visible,
                "discovery_sha256": discovery_sha256,
                "pipeline_config_sha256": pipeline_config_sha256,
            })
            if run_stop_event.is_set():
                raise RunCancelled("用户已停止参考帧草稿生成")
            update_run_status(phase="generating")
            result = run(config, target_image, description, selected,
                         max_rounds=max_rounds,
                         on_progress=on_progress, reference=None)
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
                "candidate_event_id": candidate_event_id,
                "candidate_request_id": candidate_request_id,
                "candidate_output_sha256": candidate_output_sha256,
                "input_sha256": input_sha256,
                "file": file_name,
                "topic": selected_topic,
                "frame": index,
                "labels": selected,
                "auto_detect_visible": auto_detect_visible,
                "visible_label_discovery": discovery,
                "discovery_sha256": discovery_sha256,
                "sam_prompts": prompt_payload,
                "llm_model": config.get("llm_model"),
                "prompt_profile": config.get("llm_prompt_profile", "full"),
                "description_input_version": DESCRIPTION_INPUT_VERSION,
                "pipeline_config_sha256": pipeline_config_sha256,
            }
        combined, merged_id_map = merge_inference_candidates(
            existing, result["shapes"], selected, return_id_map=True)
        bind_merged_candidate_references(result, merged_id_map)
        candidate_artifact_sha256 = _sha256_json(combined)
        visibility_review_queue = reference_visibility_review_queue(discovery, combined)
        def check_audit_cancelled():
            if run_stop_event.is_set():
                raise RunCancelled("用户已停止参考帧草稿生成")
        with cancellation(run_stop_event), trace_calls(on_trace, save_inspection_image):
            class_audits, class_review_queue = audit_high_risk_candidates(
                config, target_image, combined, labels,
                VISIBLE_LABEL_DEFINITIONS, chat, check_cancelled=check_audit_cancelled,
                tolerate_errors=True)
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
            "candidate_class_audit_status": "complete", "applied": result["applied"],
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
            "visible_label_discovery": discovery,
            "discovery_sha256": discovery_sha256,
            "discovery_retry_count": discovery_retry_count,
            "human_review_required": True}
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
                           visible_label_discovery=discovery,
                           discovery_sha256=discovery_sha256,
                           human_review_required=True,
                           message="SAM/VLM/Agent 草稿已生成；请人工精修并确认")
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
        index = int(body["frame"])
        topic_references = state.get("reference_frames", {}).get(state["topic"], [])
        if not topic_references:
            raise ValueError("请先设置参考帧并保存其真值")
        reference = topic_references[0]
        topic_count = state["video"].topics[state["topic"]]["count"]
        if not 0 <= index < topic_count:
            raise ValueError("当前帧号无效")
        if not state["truth"].get(str(reference)):
            raise ValueError("请先保存参考帧真值")
        count = int(body.get("count", 10))
        max_rounds = requested_review_rounds(body)
        if not 1 <= count <= 100:
            raise ValueError("处理帧数须为 1～100")
        last = min(index + count, topic_count)
        results = []
        targets = [target for target in range(index, last)
                   if target not in topic_references
                   and str(target) not in state["truth"]]
        if not targets:
            raise ValueError(
                f"所选范围（帧 {index}～{last - 1}）没有待处理帧。参考帧和已保存真值会跳过；"
                "请增大处理帧数，或选择包含未标注帧的范围。"
            )
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
                  "message": "候选已保存到本地；请在工作台逐帧复核并保存真值"}
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
                while True:
                    result = run(config, target_image, description, allowed_labels,
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
                replacement_run = result["agent"] | {
                    "review_queue": result["review_queue"], "applied": result["applied"],
                    "candidates_added": 0 if failed else result["candidates_added"],
                    "existing_candidates_preserved": failed}
                with lock, run_status_lock:
                    check_cancelled()
                    if (state.get("video") is not video or str(video.path) != file_name
                            or state.get("topic") != topic
                            or _sha256_json(state.get("config", {})) != config_sha256
                            or state.get("labels", []) != labels
                            or state.get("auto_labels") != auto_labels
                            or state.get("reference_frames", {}) != references
                            or state.get("truth", {}).get(str(reference)) != reference_truth
                            or key in state.get("truth", {})
                            or state.get("candidates", {}).get(key) != candidates[key]
                            or state.get("agent_runs", {}).get(key) != runs[key]
                            or reference_signature(reference) != reference_signature_snapshot):
                        raise StaleHumanReviewError("数据作用域、配置、标签、参考真值或目标帧已变化，未保存旧推理结果")
                    if failed:
                        failed_frames.append(target)
                    else:
                        state.setdefault("candidates", {})[key] = combined
                    state.setdefault("agent_runs", {})[key] = replacement_run
                    save()
                    results.append({"frame": target,
                        "candidates_added": 0 if failed else result["candidates_added"],
                        "remaining": len(combined), "applied": result["applied"],
                        "review_queue": result["review_queue"], "agent": result["agent"],
                        "failed": failed, "existing_candidates_preserved": failed})
                    output["current"] = None
                    save_run(output)
                    _update_run_status_locked(completed=len(results), phase="saved")
        output["failed_frames"] = failed_frames
        with run_status_lock:
            check_cancelled()
            if failed_frames:
                output["message"] = (f"{len(failed_frames)}/{len(results)} 帧失败；失败帧旧候选已保留，"
                                     "请查看错误后重试")
                _update_run_status_locked(status="failed", phase="failed", frame=None,
                                          error=output["message"])
            else:
                _update_run_status_locked(status="completed", phase="completed", frame=None)
        return jsonify(output)
    except RunCancelled:
        output["current"] = None
        output["message"] = "用户已停止自动标注；已完成帧的候选已保存"
        save_run(output)
        with run_status_lock:
            _update_run_status_locked(status="stopped", phase="stopped", frame=None,
                                      error=None, completed=len(results))
        return jsonify(output | {"stopped": True})
    except Exception as error:
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
    with run_status_lock:
        if run_status.get("status") not in ("running", "stopping"):
            return jsonify(status=run_status.get("status", "idle"), stopped=False)
        if run_status.get("status") == "stopping":
            return jsonify(status="stopping", stopped=True)
        run_stop_event.set()
        _update_run_status_locked(status="stopping", phase="stopping")
    return jsonify(status="stopping", stopped=True)


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
    return jsonify(snapshot)


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
    if not output.get("trace"):
        output["trace"] = read("model_debug.json").get("trace", [])
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
    return jsonify(trace=trace, steps=steps, description=description,
                   topic=selected_topic, file=selected_file, reference=output.get("reference"), frames=frames,
                   job=job,
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
