"""Flask workbench for MCAP review and local annotation."""
import io
import json
import math
import os
import threading
import time
import uuid
import urllib.request
import base64
import hashlib
from pathlib import Path
from werkzeug.utils import secure_filename
from flask import Flask, jsonify, request, send_file
from PIL import Image, ImageDraw
import numpy as np

from .mcap_video import Video
from .sensors import cloud_snapshot
from .workflow import (DESCRIPTION_INPUT_VERSION, MAX_REVIEW_ROUNDS, RunCancelled,
                       cancellation, describe, reference_labels, restrict_description,
                       run, trace_calls, candidate_overlay)

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "work"
WEB = ROOT / "source" / "web"
WORK.mkdir(exist_ok=True)
app = Flask(__name__, static_folder=str(WEB), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024 * 1024
state = {"video": None, "topic": None, "labels": [], "truth": {}, "reference_frames": {}, "config": {}, "candidates": {}, "candidates_by_topic": {}, "agent_runs": {}, "agent_runs_by_topic": {}}
lock = threading.RLock()
run_status_lock = threading.Lock()
run_status = {"status": "idle"}
run_stop_event = threading.Event()


def state_path():
    return WORK / "state.json"


def save():
    temporary = state_path().with_suffix(".tmp")
    temporary.write_text(json.dumps({k: v for k, v in state.items() if k != "video"}, ensure_ascii=False, indent=2))
    temporary.replace(state_path())


def update_run_status(**changes):
    with run_status_lock:
        run_status.update(changes)


def save_run(output):
    path = WORK / "last_run.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    temporary.replace(path)


def save_debug(trace):
    path = WORK / "model_debug.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"trace": trace}, ensure_ascii=False, indent=2))
    temporary.replace(path)


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
        temporary.replace(path)
    return "/api/inspection/image/" + name


@app.get("/api/inspection/image/<name>")
def inspection_image(name):
    if len(name) != 68 or not name.endswith(".jpg") or any(c not in "0123456789abcdef" for c in name[:-4]):
        raise ValueError("无效的过程图像名称")
    return send_file(WORK / "inspection" / name, mimetype="image/jpeg", max_age=31536000)


def save_description_trace(trace, description):
    path = WORK / "description_debug.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"trace": trace, "description": description}, ensure_ascii=False))
    temporary.replace(path)


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
        stored = json.loads(state_path().read_text())
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


def frame_image(index):
    return state["video"].frame(state["topic"], index)


def reference_signature(index):
    reference = {"file": str(state["video"].path), "topic": state["topic"], "frame": index,
                 "description_input_version": DESCRIPTION_INPUT_VERSION,
                 "truth": state["truth"][str(index)]}
    payload = json.dumps(reference, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def overlay(index):
    image = Image.open(io.BytesIO(frame_image(index))).convert("RGB")
    draw = ImageDraw.Draw(image)
    for number, shape in enumerate(state["truth"].get(str(index), []), 1):
        for part in shape.get("parts") or [shape]:
            for ring in [part.get("points", []), *(part.get("holes") or [])]:
                points = [tuple(p) for p in ring]
                if len(points) >= 3:
                    draw.line(points + points[:1], fill=(255, 0, 255), width=3)
        points = shape.get("points", [])
        if points:
            draw.text(tuple(points[0]), f"{number} {shape['label']}", fill=(255, 0, 255),
                      stroke_width=2, stroke_fill=(0, 0, 0))
    output = io.BytesIO()
    image.save(output, "JPEG", quality=88)
    return output.getvalue()


@app.errorhandler(ValueError)
@app.errorhandler(KeyError)
def bad_request(error):
    return jsonify(error=str(error)), 400


@app.errorhandler(RuntimeError)
def runtime_error(error):
    return jsonify(error=str(error)), 503


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/api/state")
def get_state():
    description = None
    reference = state.get("reference_frames", {}).get(state["topic"], []) if state["topic"] else []
    path = WORK / "description.json"
    if state["video"] and reference and path.exists():
        saved = json.loads(path.read_text())
        if saved.get("reference_signature") == reference_signature(reference[0]):
            description = saved
    return jsonify(loaded=state["video"] is not None,
        path=str(state["video"].path) if state["video"] else "",
        catalog=state["video"].catalog if state["video"] else {}, duration=state["video"].duration if state["video"] else 0,
        topics=state["video"].topics if state["video"] else {}, topic=state["topic"],
        labels=state["labels"], truth=state["truth"], reference_frames=state.get("reference_frames", {}),
        config={k: v for k, v in state["config"].items() if k != "llm_key"},
        candidates=state.get("candidates", {}), agent_runs=state.get("agent_runs", {}),
        description=description)


@app.get("/api/frame/<int:index>")
def get_frame(index):
    return send_file(io.BytesIO(state["video"].frame(request.args.get("topic", state["topic"]), index)), mimetype="image/jpeg", max_age=0)


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
    return json.loads(library_path().read_text()) if library_path().exists() else []


def write_library(files):
    temporary = library_path().with_suffix('.tmp')
    temporary.write_text(json.dumps(files, ensure_ascii=False, indent=2))
    temporary.replace(library_path())


def register_file(path, name=None):
    path = Path(path).expanduser().resolve()
    if not path.is_file() or path.suffix.lower() != '.mcap':
        raise ValueError('请选择存在的 MCAP 文件')
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
                agent_runs=state.get('agent_runs', {}), reference_frames=state.get('reference_frames', {}))


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
    if uploaded is None or not uploaded.filename or not uploaded.filename.lower().endswith(".mcap"):
        raise ValueError("请选择 MCAP 文件")
    destination = WORK / "uploads"
    destination.mkdir(exist_ok=True)
    path = destination / f"{uuid.uuid4().hex}-{secure_filename(uploaded.filename) or 'capture.mcap'}"
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
        state.setdefault("truths_by_topic", {})[state["topic"]] = state["truth"]
        state.setdefault("candidates_by_topic", {})[state["topic"]] = state.get("candidates", {})
        state.setdefault("agent_runs_by_topic", {})[state["topic"]] = state.get("agent_runs", {})
        state["topic"] = body["topic"]
        state["truth"] = state["truths_by_topic"].get(state["topic"], {})
        state["candidates"] = state["candidates_by_topic"].get(state["topic"], {})
        state["agent_runs"] = state["agent_runs_by_topic"].get(state["topic"], {})
        save()
        return jsonify(topic=state["topic"], truth=state["truth"], candidates=state["candidates"], agent_runs=state["agent_runs"], reference_frames=state.get("reference_frames", {}))


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
    body = request.get_json()
    with lock:
        index = int(body["frame"])
        width, height = Image.open(io.BytesIO(frame_image(index))).size
        shapes = body["shapes"]
        for shape in shapes:
            if shape["label"] not in state["labels"] or len(shape["points"]) < 3:
                raise ValueError("真值类别或多边形无效")
            if any(not (0 <= x < width and 0 <= y < height) for x, y in shape["points"]):
                raise ValueError("坐标超出图像范围")
        state["truth"][str(index)] = shapes
        state.setdefault("candidates", {}).pop(str(index), None)
        save()
        return jsonify(saved=len(shapes))


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
            record = json.loads(path.read_text())
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
            temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2))
            temporary.replace(path)
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
                    still_used.update(inspection_image_names(json.loads(path.read_text())))
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
    with lock:
        for key in ("llm_url", "llm_model", "llm_key"):
            if key in body and body[key]:
                state["config"][key] = body[key]
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
        (WORK / "description.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
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
            description = json.loads(description_path.read_text()) if description_path.exists() else None
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
    output = json.loads(path.read_text())
    return jsonify(trace=output.get("trace", []))


@app.get("/api/run/inspection")
def get_inspection():
    def read(name):
        path = WORK / name
        return json.loads(path.read_text()) if path.exists() else {}
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
