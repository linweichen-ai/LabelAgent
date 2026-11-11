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
import base64
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from werkzeug.utils import secure_filename
from flask import Flask, jsonify, request, send_file
from PIL import Image, ImageDraw
import numpy as np

from .media import IMAGE_SUFFIXES, SUPPORTED_SUFFIXES, open_media as Video, read_image
from .sensors import cloud_snapshot
from .candidate_import import prepare_candidate_import, merge_candidate_import
from .downstream_mapping import mapping_contract, project_shapes
from .workflow import (DESCRIPTION_INPUT_VERSION, MAX_REVIEW_ROUNDS, RunCancelled,
                       cancellation, describe, reference_labels, restrict_description,
                       run, trace_calls, candidate_overlay, reference_overlay)

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(os.environ.get("LABELAGENT_WORK", ROOT / "work")).expanduser().resolve()
WEB = ROOT / "source" / "web"
WORK.mkdir(parents=True, exist_ok=True)
app = Flask(__name__, static_folder=str(WEB), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024 * 1024
state = {"video": None, "topic": None, "labels": [], "truth": {}, "reference_frames": {}, "config": {}, "candidates": {}, "candidates_by_topic": {}, "agent_runs": {}, "agent_runs_by_topic": {}, "candidate_review_status": {}, "human_review_transactions": {}}
lock = threading.RLock()
run_status_lock = threading.Lock()
run_status = {"status": "idle"}
run_stop_event = threading.Event()


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


def save():
    temporary = state_path().with_suffix(".tmp")
    temporary.write_text(json.dumps({k: v for k, v in state.items() if k != "video"}, ensure_ascii=False, indent=2), encoding="utf-8")
    _commit_work_file(temporary, state_path())


def _canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _sha256_json(value):
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _valid_sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _review_scope(frame_sha256):
    config = state.setdefault("config", {})
    file_key = hashlib.sha256(str(state["video"].path).encode("utf-8")).hexdigest()[:16]
    return {
        "dataset_id": config.get("dataset_id") or f"local-workbench-{file_key}",
        "dataset_version": config.get("dataset_version") or f"frame-content-{frame_sha256[:16]}",
        "camera": config.get("camera") or state["topic"],
    }


def _assert_review_binding(body, *, frame_sha256, candidate_sha256):
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
    scope = _review_scope(frame_sha256)
    expected_scope = {
        "dataset_id": body.get("expected_dataset_id"),
        "dataset_version": body.get("expected_dataset_version"),
        "camera": body.get("expected_camera"),
    }
    if any(not isinstance(value, str) or not value.strip()
           for value in expected_scope.values()):
        raise ValueError("人工复核必须绑定 expected_dataset_id、expected_dataset_version 和 expected_camera")
    if (expected_frame != frame_sha256
            or expected_candidate != candidate_sha256 or expected_scope != scope):
        raise StaleHumanReviewError("数据集、版本、相机、帧或候选已在复核期间变化，请重新复核后提交")
    return expected_frame, expected_candidate, scope


def _contains_slope(shapes):
    return bool(shapes) and any(isinstance(item, dict)
        and item.get("label") in {"slope", "slope_visible"} for item in shapes)


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
    return {"semantic_label": "slope_visible", "visible_status": visible_status,
        "human_pixel_truth_available": human_truth, "approved_for_training": False,
        "business_mapping": "traversable_area", "traversability_status": "passable",
        "downstream_mapping": mapping_contract(),
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
    if remediation is not None and (not isinstance(remediation, str) or len(remediation) != 64):
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
                              allowed_decisions, review_scope, truth=None, candidate=None):
    expected = {"file": str(state["video"].path), "topic": state["topic"],
                "frame": frame, "frame_sha256": frame_sha256,
                "reviewer_declared_id": review["reviewer_declared_id"],
                "identity_assurance": "self_declared",
                "started_at": review["started_at"], "note": review["note"],
                "remediation_of_review_id": review.get("remediation_of_review_id"),
                "dataset_id": review_scope["dataset_id"],
                "dataset_version": review_scope["dataset_version"],
                "camera": review_scope["camera"]}
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
                         review_scope, candidate, truth=None, confirmed_empty=False):
    history = human_review_records()
    finished = datetime.now(timezone.utc)
    started = datetime.fromisoformat(review["started_at"].replace("Z", "+00:00"))
    candidate_sha = _sha256_json(candidate) if candidate is not None else None
    candidate_event_id = _sha256_json({"event": "workbench_candidate",
        "file": str(state["video"].path), "topic": state["topic"], "frame": frame,
        "candidate_sha256": candidate_sha}) if candidate is not None else None
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
            "frame": frame, "frame_sha256": frame_sha256,
             "expected_frame_sha256": expected_frame_sha256,
             "expected_candidate_sha256": expected_candidate_sha256,
             "decision": decision, **review,
             "reviewed_at": finished.isoformat().replace("+00:00", "Z"),
             "elapsed_seconds": round(max(0.0, (finished - started).total_seconds()), 3),
             "candidate_sha256": candidate_sha,
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
    persisted = json.loads(state_path().read_text(encoding="utf-8"))
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
        run_status.update(changes)


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


def redact_debug(value):
    key = state["config"].get("llm_key")
    if isinstance(value, str):
        return value.replace(key, "[REDACTED]") if key else value
    if isinstance(value, dict):
        return {name: redact_debug(item) for name, item in value.items()}
    if isinstance(value, list):
        return [redact_debug(item) for item in value]
    return value


def load():
    if state_path().exists():
        stored = json.loads(state_path().read_text(encoding="utf-8"))
        state.update(stored)
    for key in ("sam_url", "cvat_url", "cvat_token"):
        state["config"].pop(key, None)
    for key in ("job_id", "task_id", "cvat_start", "cvat_count"):
        state.pop(key, None)
    for session in state.get("file_sessions", {}).values():
        for key in ("job_id", "task_id", "cvat_start", "cvat_count"):
            session.pop(key, None)
    state.setdefault("candidates", {})
    state.setdefault("candidates_by_topic", {})
    state.setdefault("agent_runs", {})
    state.setdefault("agent_runs_by_topic", {})
    state.setdefault("candidate_review_status", {})
    state.setdefault("human_review_transactions", {})
    recover_pending_human_reviews()


def frame_image(index):
    return state["video"].frame(state["topic"], index)


def reference_signature(index):
    reference = {"file": str(state["video"].path), "topic": state["topic"], "frame": index,
                 "description_input_version": DESCRIPTION_INPUT_VERSION,
                 "prompt_profile": state.get("config", {}).get("llm_prompt_profile", "full"),
                 "truth": state["truth"][str(index)]}
    payload = json.dumps(reference, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def overlay(index):
    return reference_overlay(frame_image(index), state["truth"].get(str(index), []))


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


@app.get("/api/state")
def get_state():
    with lock:
        recover_pending_human_reviews()
    description = None
    reference = state.get("reference_frames", {}).get(state["topic"], []) if state["topic"] else []
    path = WORK / "description.json"
    if state["video"] and reference and path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("reference_signature") == reference_signature(reference[0]):
            description = saved
    return jsonify(loaded=state["video"] is not None,
        path=str(state["video"].path) if state["video"] else "",
        catalog=state["video"].catalog if state["video"] else {}, duration=state["video"].duration if state["video"] else 0,
        topics=state["video"].topics if state["video"] else {}, topic=state["topic"],
        labels=state["labels"], truth=state["truth"], reference_frames=state.get("reference_frames", {}),
        config={k: v for k, v in state["config"].items() if k != "llm_key"},
        candidates=state.get("candidates", {}), agent_runs=state.get("agent_runs", {}),
        candidate_review_status=state.get("candidate_review_status", {}),
        description=description)


@app.get("/api/frame/<int:index>")
def get_frame(index):
    data = state["video"].frame(request.args.get("topic", state["topic"]), index)
    return send_file(io.BytesIO(data), mimetype="image/png" if data.startswith(b"\x89PNG") else "image/jpeg", max_age=0)


@app.post("/api/segment_points")
def get_point_segment():
    body = request.get_json()
    with lock:
        if state["video"] is None or not state["topic"]:
            raise ValueError("请先加载图像消息")
        index = int(body["frame"])
        if index in state.get("reference_frames", {}).get(state["topic"], []):
            raise ValueError("参考帧只能人工调整，不能调用 SAM 生成或修改")
        image = frame_image(index)
    service_url = os.environ.get("SAM3_URL", "http://127.0.0.1:8766").rstrip("/")
    payload = json.dumps({"image": base64.b64encode(image).decode("ascii"),
                          "points": body["points"]}).encode()
    service_request = urllib.request.Request(service_url + "/segment_points", data=payload,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(service_request, timeout=600) as response:
            return jsonify(json.load(response))
    except Exception as error:
        raise RuntimeError(f"SAM 3 服务不可用（{service_url}）：{error}") from error

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
    video = Video(item['path'])
    # Preserve each file's annotations and candidates when switching records.
    sessions = state.setdefault('file_sessions', {})
    fields = ('topic', 'truth', 'truths_by_topic', 'reference_frames', 'candidates', 'candidates_by_topic', 'agent_runs', 'agent_runs_by_topic')
    if state['video']:
        sessions[str(state['video'].path.resolve())] = {key: state.get(key) for key in fields}
    previous = sessions.get(item['path'], {})
    state.update(video=video, video_path=str(video.path), topic=next(iter(video.topics), None),
                 truth={}, truths_by_topic={}, reference_frames={}, candidates={}, candidates_by_topic={}, agent_runs={}, agent_runs_by_topic={})
    state.update(previous)
    if state['topic'] not in video.topics:
        state['topic'] = next(iter(video.topics), None)
    save()
    return dict(path=str(video.path), topics=video.topics, catalog=video.catalog, duration=video.duration,
                topic=state['topic'], truth=state['truth'], candidates=state.get('candidates', {}),
                agent_runs=state.get('agent_runs', {}), reference_frames=state.get('reference_frames', {}),
                candidate_review_status=state.get('candidate_review_status', {}))


@app.get('/api/files')
def get_files():
    with lock:
        if state['video']:
            register_file(state['video'].path)
        return jsonify(files=file_list())


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


@app.delete('/api/files/<file_id>')
def remove_file(file_id):
    with lock:
        files = read_library()
        item = next((item for item in files if item['id'] == file_id), None)
        if item is None:
            raise ValueError('文件不在列表中')
        write_library([entry for entry in files if entry['id'] != file_id])
        unloaded = state['video'] is not None and str(state['video'].path.resolve()) == item['path']
        if unloaded:
            sessions = state.setdefault('file_sessions', {})
            sessions[item['path']] = {key: state.get(key) for key in ('topic', 'truth', 'truths_by_topic', 'reference_frames', 'candidates', 'candidates_by_topic', 'agent_runs', 'agent_runs_by_topic')}
            state.update(video=None, video_path=None, topic=None, truth={}, truths_by_topic={}, reference_frames={}, candidates={}, candidates_by_topic={}, agent_runs={}, agent_runs_by_topic={})
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
            state["topic"] = body["topic"]
            state["truth"] = state["truths_by_topic"].get(state["topic"], {})
            state["candidates"] = state["candidates_by_topic"].get(state["topic"], {})
            state["agent_runs"] = state["agent_runs_by_topic"].get(state["topic"], {})
            save()
        return jsonify(topic=state["topic"], truth=state["truth"], candidates=state["candidates"], agent_runs=state["agent_runs"], reference_frames=state.get("reference_frames", {}), candidate_review_status=state.get("candidate_review_status", {}))


@app.post("/api/labels")
def set_labels():
    body = request.get_json()
    with lock:
        labels = [x.strip() for x in body["labels"] if x.strip()]
        if len(set(labels)) != len(labels) or not labels:
            raise ValueError("类别不能为空或重复")
        state["labels"] = labels
        save()
        return jsonify(labels=labels)


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
        existing = _review_by_idempotency_key(review["decision_idempotency_key"])
        if existing is not None:
            expected_frame, expected_candidate, review_scope = _assert_review_binding(body,
                frame_sha256=frame_sha,
                candidate_sha256=existing.get("candidate_sha256"))
            _assert_idempotent_review(existing, review=review, frame=index,
                frame_sha256=frame_sha,
                allowed_decisions={"manual_annotation", "accept_candidate",
                                   "correct_candidate", "confirmed_empty"},
                review_scope=review_scope, truth=shapes)
            saved_truth = state.setdefault("truth", {}).get(str(index))
            if saved_truth is None or _sha256_json(saved_truth) != existing.get("truth_sha256"):
                raise RuntimeError("幂等复放找到审计记录，但持久化真值不匹配")
            readback = _review_commit_readback(existing)
            if not readback["verified"]:
                raise RuntimeError("幂等复放的服务端读回验证失败")
            return jsonify(saved=len(shapes), review=existing, readback=readback,
                           idempotent_replay=True)
        candidate = state.setdefault("candidates", {}).get(str(index))
        candidate_sha = _sha256_json(candidate) if candidate is not None else None
        expected_frame, expected_candidate, review_scope = _assert_review_binding(body,
            frame_sha256=frame_sha, candidate_sha256=candidate_sha)
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
            confirmed_empty=confirmed_empty)
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
        candidate = state.setdefault("candidates", {}).get(str(index))
        candidate_sha = _sha256_json(candidate) if candidate is not None else None
        scope = _review_scope(frame_sha)
        return jsonify(frame=index, topic=state["topic"],
                       frame_sha256=frame_sha, candidate_sha256=candidate_sha,
                       dataset_id=scope["dataset_id"],
                       dataset_version=scope["dataset_version"], camera=scope["camera"])


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
        existing = _review_by_idempotency_key(review["decision_idempotency_key"])
        if existing is not None:
            expected_frame, expected_candidate, review_scope = _assert_review_binding(body,
                frame_sha256=frame_sha,
                candidate_sha256=existing.get("candidate_sha256"))
            _assert_idempotent_review(existing, review=review, frame=index,
                frame_sha256=frame_sha, allowed_decisions={"reject_candidate"},
                review_scope=review_scope)
            readback = _review_commit_readback(existing)
            if not readback["verified"]:
                raise RuntimeError("驳回幂等复放的服务端读回验证失败")
            return jsonify(rejected=True, review=existing, idempotent_replay=True,
                           remediation_required=True, readback=readback)
        candidate = state.setdefault("candidates", {}).get(str(index))
        if candidate is None:
            raise ValueError("当前帧没有可驳回的自动候选")
        status_key = _review_status_key(index)
        current_status = state.setdefault("candidate_review_status", {}).get(status_key)
        candidate_sha = _sha256_json(candidate)
        expected_frame, expected_candidate, review_scope = _assert_review_binding(body,
            frame_sha256=frame_sha, candidate_sha256=candidate_sha)
        if (current_status and current_status.get("status") == "rejected"
                and current_status.get("candidate_sha256") == candidate_sha):
            raise ValueError("该候选已被驳回；请复用原幂等键，或先完成修正后再复核")
        audit = prepare_human_review(frame=index, decision="reject_candidate", review=review,
            frame_sha256=frame_sha, expected_frame_sha256=expected_frame,
            expected_candidate_sha256=expected_candidate,
            review_scope=review_scope, candidate=candidate)
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


@app.get("/api/downstream_projection/<int:index>")
def get_downstream_projection(index):
    with lock:
        if state["video"] is None or state["topic"] is None:
            raise ValueError("请先加载图像消息")
        if not 0 <= index < state["video"].topics[state["topic"]]["count"]:
            raise ValueError("帧号无效")
        key = str(index)
        if key in state.setdefault("truth", {}):
            shapes, source_layer = state["truth"][key], "human_truth"
        elif key in state.setdefault("candidates", {}):
            shapes, source_layer = state["candidates"][key], "machine_candidate"
        else:
            raise ValueError("当前帧没有可投影的真值或机器候选")
        return jsonify(frame=index, topic=state["topic"],
                       projection=project_shapes(shapes, source_layer=source_layer))


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
                        run_status.update(status="idle", frame=None, phase=None, round=0,
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
        for key in ("dataset_id", "dataset_version", "camera"):
            if key in body and (not isinstance(body[key], str) or not body[key].strip()
                                or len(body[key].strip()) > 300):
                raise ValueError(f"{key} must be a non-empty string no longer than 300 characters")
        for key in ("llm_url", "llm_model", "llm_key", "llm_prompt_profile",
                    "dataset_id", "dataset_version", "camera"):
            if key in body and body[key]:
                state["config"][key] = body[key].strip() if isinstance(body[key], str) else body[key]
        save()
        return jsonify(saved=True)


@app.post("/api/describe")
def describe_truth():
    body = request.get_json()
    with lock:
        index = int(body["frame"])
        if index not in state.get("reference_frames", {}).get(state["topic"], []):
            raise ValueError("请先将当前帧设为参考帧")
        if not state["truth"].get(str(index)):
            raise ValueError("请先保存该参考帧的真值")
        allowed_labels = reference_labels(state["truth"][str(index)], state["labels"])
        trace = []
        save_debug(trace)
        def on_trace(event):
            event = redact_debug(event | {"frame": index, "purpose": "生成真值描述",
                "topic": state["topic"], "file": str(state["video"].path)})
            if trace:
                trace[0] = event
            else:
                trace.append(event)
            save_debug(trace)
        with trace_calls(on_trace, save_inspection_image):
            result = describe(state["config"], frame_image(index),
                state["truth"][str(index)], allowed_labels, overlay(index))
        result["reference_signature"] = reference_signature(index)
        save_description_trace(trace, result)
        (WORK / "description.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return jsonify(result)


@app.post("/api/run")
def run_workflow():
    body = request.get_json() or {}
    with lock:
        with run_status_lock:
            if run_status.get("status") in ("running", "stopping"):
                raise RuntimeError("已有自动标注任务正在运行")
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
        max_rounds = MAX_REVIEW_ROUNDS
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
        allowed_labels = reference_labels(state["truth"][str(reference)], state["labels"])
        description = body.get("description")
        if description is None:
            description_path = WORK / "description.json"
            description = json.loads(description_path.read_text(encoding="utf-8")) if description_path.exists() else None
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
        update_run_status(status="running", frame=None, completed=0, total=len(targets),
                          phase="starting", round=0, error=None, topic=state["topic"],
                          file=str(state["video"].path),
                          started_at=time.time(), events=[], shape_count=0)
        run_stop_event.clear()
        save_run(output)
        save_debug([])
        trace_frame = reference
        def on_trace(event):
            with run_status_lock:
                phase, round_number = run_status.get("phase"), run_status.get("round", 0)
            event = redact_debug(event | {"frame": trace_frame, "round": round_number,
                "phase": phase, "topic": state["topic"], "file": str(state["video"].path),
                "purpose": "自动标注"})
            trace = output["trace"]
            existing = next((i for i, item in enumerate(trace) if item["id"] == event["id"]), None)
            if existing is None:
                trace.append(event)
            else:
                trace[existing] = event
            save_debug(trace)
            save_run(output)
        try:
            with cancellation(run_stop_event), trace_calls(on_trace, save_inspection_image):
                reference_context = {"image": frame_image(reference),
                    "marked_image": overlay(reference),
                    "annotations": state["truth"][str(reference)]}
                for target in targets:
                    trace_frame = target
                    update_run_status(frame=target, phase="loading", round=0, shape_count=0)
                    target_image = frame_image(target)
                    def on_progress(event, target=target):
                        if run_stop_event.is_set():
                            raise RunCancelled("用户已停止自动标注")
                        record = {"time": time.time(), "frame": target,
                            "phase": event["phase"], "round": event["round"],
                            "shape_count": len(event["shapes"])}
                        if event.get("label"):
                            record["label"] = event["label"]
                        if event.get("operation"):
                            record["operation"] = event["operation"]
                        if event.get("outcome"):
                            record["outcome"] = event["outcome"]
                        if event["phase"] == "finished":
                            record["stop_reason"] = event["result"]["agent"]["stop_reason"]
                        with run_status_lock:
                            run_status.update(phase=event["phase"], round=event["round"],
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
                    if run_stop_event.is_set():
                        raise RunCancelled("用户已停止自动标注")
                    result = run(state["config"], target_image, description, allowed_labels,
                        max_rounds=max_rounds, on_progress=on_progress, reference=reference_context)
                    if run_stop_event.is_set():
                        raise RunCancelled("用户已停止自动标注")
                    state.setdefault("candidates", {})[str(target)] = result["shapes"]
                    state.setdefault("agent_runs", {})[str(target)] = result["agent"] | {
                        "review_queue": result["review_queue"], "applied": result["applied"],
                        "candidates_added": result["candidates_added"]}
                    save()
                    results.append({"frame": target, "candidates_added": result["candidates_added"],
                        "remaining": len(result["shapes"]), "applied": result["applied"],
                        "review_queue": result["review_queue"], "agent": result["agent"]})
                    output["current"] = None
                    save_run(output)
                    update_run_status(completed=len(results), phase="saved")
        except RunCancelled:
            output["current"] = None
            output["message"] = "用户已停止自动标注；已完成帧的候选已保存"
            save_run(output)
            update_run_status(status="stopped", phase="stopped", frame=None,
                              error=None, completed=len(results))
            return jsonify(output | {"stopped": True})
        except Exception as error:
            output["error"] = str(error)[:500]
            save_run(output)
            if run_stop_event.is_set():
                update_run_status(status="stopped", phase="stopped", error="用户已停止自动标注")
            else:
                update_run_status(status="failed", phase="failed", error=str(error)[:500])
            raise
        update_run_status(status="completed", phase="completed", frame=None)
        return jsonify(output)


@app.post("/api/run/stop")
def stop_workflow():
    with run_status_lock:
        if run_status.get("status") not in ("running", "stopping"):
            return jsonify(status=run_status.get("status", "idle"), stopped=False)
        if run_status.get("status") == "stopping":
            return jsonify(status="stopping", stopped=True)
        run_stop_event.set()
        run_status.update(status="stopping", phase="stopping")
    return jsonify(status="stopping", stopped=True)


@app.get("/api/run/status")
def get_run_status():
    with run_status_lock:
        return jsonify(run_status.copy())


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
