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
MAX_REVIEW_ROUNDS = 100
MAX_PROTOCOL_RETRIES = 2


class ModelOutputError(ValueError):
    """A model response that can be corrected without repeating tool execution."""

    def __init__(self, message, raw_text):
        super().__init__(message)
        self.raw_text = raw_text


class RunCancelled(Exception):
    """The user terminated this annotation run."""


def check_cancelled():
    event = _cancel_event.get()
    if event is not None and event.is_set():
        raise RunCancelled("用户已停止自动标注")


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


def sam_request(config, endpoint, image, **prompt):
    trace_prompt = dict(prompt)
    if "mask" in trace_prompt:
        trace_prompt["mask"] = _image_summary(base64.b64decode(trace_prompt["mask"]))
    record = {"id": uuid.uuid4().hex, "service": "SAM 3", "status": "running",
              "started_at": time.time(), "input": {"endpoint": endpoint,
              "image": _image_summary(image), **trace_prompt}}
    _emit_trace(record)
    try:
        response = _cancellable_call(_sam_request, config, endpoint, image, **prompt)
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


def _sam_request(config, endpoint, image, **prompt):
    url = config.get("sam3_url") or os.environ.get("SAM3_URL", "http://127.0.0.1:8766")
    payload = json.dumps({"image": base64.b64encode(image).decode("ascii"), **prompt}).encode()
    request = urllib.request.Request(url.rstrip("/") + endpoint, data=payload,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.load(response)
    except (TimeoutError, socket.timeout) as error:
        raise ValueError(f"SAM 3 请求超时（{endpoint}）") from error
    except urllib.error.HTTPError as error:
        try:
            message = json.load(error).get("error")
        except (ValueError, AttributeError):
            message = None
        raise ValueError(f"SAM 3 请求失败：{message or error.reason}") from error
    except urllib.error.URLError as error:
        raise ValueError(f"SAM 3 请求失败：{error.reason}") from error


def sam_text(image, prompt, config):
    return sam_request(config, "/segment_text", image, prompt=prompt).get("results", [])


def chat(config, instruction, image, extra_images=()):
    record = {"id": uuid.uuid4().hex, "service": "GPT", "status": "running",
              "started_at": time.time(), "input": {"model": config.get("llm_model"),
              "instruction": instruction, "images": [_image_summary(data)
              for data in (image, *extra_images)]}}
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
        response = _cancellable_call(_chat, config, instruction, image, extra_images, raw)
        record.update(status="succeeded", output={"raw_text": raw.get("text"),
                                                   "parsed": response})
        return response
    except Exception as error:
        record.update(status="failed", error=str(error)[:500])
        if raw:
            record["output"] = {"raw_text": raw["text"]}
        raise
    finally:
        record["elapsed_ms"] = round((time.time() - record["started_at"]) * 1000)
        _emit_trace(record)


def _chat(config, instruction, image, extra_images=(), raw=None):
    if not config.get("llm_url") or not config.get("llm_model"):
        raise ValueError("请先配置大模型地址和模型名")
    content = [{"type": "text", "text": instruction}]
    for data in (image, *extra_images):
        content.append({"type": "image_url", "image_url": {"url":
            "data:image/jpeg;base64," + base64.b64encode(data).decode()}})
    payload = {"model": config["llm_model"], "temperature": 0,
               "messages": [{"role": "user", "content": content}]}
    request = urllib.request.Request(config["llm_url"].rstrip("/") + "/chat/completions",
        json.dumps(payload).encode(), headers={"Content-Type": "application/json",
        "Authorization": "Bearer " + config.get("llm_key", "")})
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.load(response)
    except (TimeoutError, socket.timeout) as error:
        raise ValueError("llm_timeout: 大模型请求超过 300 秒") from error
    except urllib.error.HTTPError as error:
        try:
            detail = error.read().decode("utf-8", "replace")[:300]
        except Exception:
            detail = error.reason
        raise ValueError(f"大模型请求失败（HTTP {error.code}）：{detail}") from error
    except urllib.error.URLError as error:
        raise ValueError(f"大模型请求失败：{error.reason}") from error
    content = result["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content)
    if raw is not None:
        raw["text"] = content[:20000]
    content = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        result = json.loads(content)
    except json.JSONDecodeError as error:
        raise ModelOutputError(f"大模型未返回有效 JSON：{error}", content[:20000]) from error
    if not isinstance(result, dict):
        raise ModelOutputError("大模型必须返回 JSON 对象", content[:20000])
    return result


def reference_labels(annotations, labels):
    """Only categories actually annotated on the reference frame may propagate."""
    present = {shape.get("label") for shape in annotations}
    selected = [label for label in labels if label in present]
    if not selected or len(selected) != len(present):
        raise ValueError("参考帧真值类别与类别列表不一致，请先核对参考帧标注")
    return selected


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
        if label not in proposals:
            raise ValueError(f"类别 {label} 缺少 SAM 3 关键词，请重新生成描述")
        text = proposals.get(label, "").strip()
        # SAM text is an object query, not the detailed annotation rule.
        # If GPT returns a sentence, fall back to the known reference class.
        bad_words = {"exclude", "excluding", "include", "including", "without",
                     "around", "surrounding", "bounded", "boundary", "only", "not",
                     "visible", "the", "and", "or"}
        valid = (bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9 -]*", text))
                 and 1 <= len(text.split()) <= 4
                 and not any(word.lower() in bad_words for word in text.split()))
        if label == "地面" and valid:
            valid = any(word in text.lower().split() for word in ("floor", "ground"))
        if not valid:
            text = ({"地面": "floor"}.get(label) or
                    (label.replace("_", " ") if label.isascii() else ""))
        if not text:
            raise ValueError(f"类别 {label} 缺少简短英文 SAM 3 提示，请重新生成描述")
        prompts.append({"label": label, "text": text})
    return {"description": result["description"].strip(), "prompts": prompts}


DESCRIPTION_INPUT_VERSION = 5

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


def describe(config, image, annotations, labels, marked_image=None):
    labels = reference_labels(annotations, labels)
    prompt = (REFERENCE_PRINCIPLES
              + "\n图像 1 是未着色的原始参考帧；图像 2（若提供）只用洋红色轮廓和编号标出参考帧真值。"
              "标注轮廓颜色只是界面标记，绝不是目标物体本身的颜色。"
              "描述地面或其他目标的真实颜色、材质时，仅依据图像 1；无法确认则省略颜色。"
              "真值几何数据中的 points 是外轮廓，holes 是排除的内部孔洞，parts 是同一标注的组成区域。"
              "结合参考帧图像及真值几何数据，输出严格 JSON："
              '{"description":"参考帧真值标注规范","prompts":[{"label":"类别名","text":"SAM 3 英文关键词"}]}。'
              "\n8. 分别生成完整描述和 SAM 3 关键词："
              "description 给复核 Agent 使用，描述真值中的目标、实例归属、可见边界、遮挡及排除范围；"
              "只描述已标注类别，不为图中未标注的物体新增任务。"
              "prompts 给 SAM 3 分割使用，每个类别一个 text，仅 1～4 个英文关键词（名词短语），"
              "例如 floor、wooden pallet；不得写完整句子、排除项或边界规则。"
              "每个允许类别必须有且只有一个 prompt。"
              f"\n允许类别：{json.dumps(labels, ensure_ascii=False)}"
              f"\n真值：{json.dumps(annotations, ensure_ascii=False)}")
    result = chat(config, prompt, image, extra_images=(marked_image,) if marked_image else ())
    return restrict_description(result, labels)


REPAIR_PRINCIPLES = """
复查修复原则：
1. 复查阶段允许使用 SAM 3，但只能对现有候选的局部问题使用 sam_points，逐步提供当前帧坐标的正点和负点；历史提示点会累计，现有 mask 会作为输入，polygon 指定允许修改的局部区域。禁止使用 sam_text 对整幅图像重新分割，也不得用空 shape_ids 创建全新整图候选。
2. SAM 3 点提示结果必须由下一轮 GPT 与修改前候选比较后确认是否改进；有待评估结果时，必须输出 sam_assessment:{"improved":true或false,"reason":"比较依据"}。只有缺陷已改善且正确区域未受损才接受；无改进、退化或不确定时 improved:false，Agent 自动恢复上一版本，并使用本轮的 OpenCV 操作进行精细调整，不再为该候选重复 SAM 3。此时操作坐标针对恢复后的版本；不能仅因 SAM 3 返回成功或分数较高就接受结果。
3. 保留已正确标注的区域，只针对有证据的局部缺陷修复；不得为修复局部边界删除整个候选再重建。remove 仅用于整个实例均为误检；merge、split 仅用于确有参考规范依据的实例归属错误。
4. 每项操作的 reason 须说明待处理帧的具体缺陷、对应参考真值证据和预期变化；不得仅凭参考帧与当前帧坐标或面积不同认定错误，不得将标线、阴影或颜色变化自动视为排除区域。
5. 对地面框线逐段检查：如果原图中能确认存在未被物体遮挡的地面框线或标线，必须检查当前候选 mask 是否覆盖其可见部分；发现漏标时使用局部 sam_points 或 add_region 修复，不能返回 done。线条颜色不能作为扣除依据。
6. polygon 和点提示必须使用当前帧原始像素坐标，修改范围限于需要修复的区域；不得用大范围粗略区域破坏正确边界、孔洞或其他区域。
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


def _validate_review_result(result, holes, shapes, labels, width, height, sam_comparison=None):
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
    shape_ids = {item["id"] for item in shapes}
    hole_ids = {item["id"]: item for item in holes}
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
                              f"需要 3～2000 个 [x,y] 顶点，0≤x<{width}，0≤y<{height}")
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
    if holes:
        assessment = result.get("hole_assessment")
        expected = {item["id"] for item in holes}
        if not isinstance(assessment, list):
            errors.append("GPT 未逐一判断当前候选的内部空洞；必须为清单中的每个完整 hole_id "
                          "输出含 semantic 和 reason 的判断")
            assessment = []
        observed = {item["hole_id"] for item in assessment
                    if isinstance(item, dict) and isinstance(item.get("hole_id"), str)}
        if (len(assessment) != len(expected) or any(not isinstance(item, dict) for item in assessment)
                or observed != expected or any(item.get("semantic") not in ("ground", "non_ground")
                                       or not isinstance(item.get("reason"), str)
                                       or not item["reason"].strip()
                                       for item in assessment if isinstance(item, dict))):
            errors.append("GPT 空洞判断必须覆盖全部 hole_id，并包含语义和依据；"
                          f"缺失 ID：{sorted(expected - observed)}；"
                          f"无效 ID：{sorted(observed - expected)}。"
                          "每个完整 ID 恰好一次，禁止用 #5 或图片编号替代完整 hole_id；"
                          "semantic 必须为 ground/non_ground，reason 必须为非空字符串")
        filled = {item.get("hole_id") for item in result["operations"]
                  if isinstance(item, dict) and item.get("operation") == "fill_gap"
                  and isinstance(item.get("hole_id"), str)}
        ground = {item["hole_id"] for item in assessment if isinstance(item, dict)
                  and isinstance(item.get("hole_id"), str) and item.get("semantic") == "ground"}
        if not ground.issubset(filled) and result["status"] == "done":
            errors.append(f"GPT 不能在仍有属于地面的未修补空洞时返回 done：{sorted(ground - filled)}")
        result["hole_assessment"] = assessment
    if sam_comparison is not None:
        assessment = result.get("sam_assessment")
        if (not isinstance(assessment, dict) or type(assessment.get("improved")) is not bool
                or not isinstance(assessment.get("reason"), str) or not assessment["reason"].strip()):
            errors.append('GPT 未评估 SAM 3 点提示结果；必须提供 sam_assessment:'
                          '{"improved":true或false,"reason":"比较依据"}')
    if errors:
        raise ValueError("\n".join(errors))
    return result


def review(config, image, overlay, description, shapes, labels, history, reference=None,
           sam_comparison=None, hole_image=None, holes=None, gap_images=None):
    width, height = Image.open(io.BytesIO(image)).size
    prompt = (TOOLS + REFERENCE_PRINCIPLES + REPAIR_PRINCIPLES
        + "\n只能标注参考帧真值中已有的类别；即使图中可见其他物体，也不得为其创建标注。"
        "\n你正在复核当前候选（首轮为 SAM 3 初步分割，后续轮次可能经过 SAM 3 点提示或 OpenCV 修改）。比较原始帧、带候选轮廓/ID 的图像及下列真值标注规范，"
        "找出漏标、误检、边界偏差和类别错误，返回具体的工具操作；必要时多轮重新观察。"
        + f"\n图像尺寸：{width} x {height}"
        f"\n参考帧真值标注规范：{description.get('description', '')}"
        f"\nSAM 3 类别关键词：{json.dumps(description['prompts'], ensure_ascii=False)}"
        f"\n类别：{json.dumps(labels, ensure_ascii=False)}"
        f"\n当前候选（ID、类别、框、顶点数）：{json.dumps(shapes, ensure_ascii=False)}"
        f"\n历史决策及工具结果（最近 8 轮）：{json.dumps(history[-8:], ensure_ascii=False)}")
    holes = holes or []
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
    last_result = None
    last_error = None
    for attempt in range(MAX_PROTOCOL_RETRIES + 1):
        if attempt:
            correction = ("\n上一份输出未通过 Agent 协议校验。请保留其中正确的视觉判断，只修正协议错误，"
                "重新输出完整 JSON，不要解释。\n校验错误：" + str(last_error)
                + "\n上一份输出：" + json.dumps(last_result, ensure_ascii=False)[:12000])
            request_prompt = prompt + correction
        else:
            request_prompt = prompt
        call_state = {"attempt": attempt + 1, "feedback": str(last_error) if attempt else None}
        token = _review_call.set(call_state)
        try:
            check_cancelled()
            try:
                last_result = chat(config, request_prompt, image, extra_images=extra_images)
            except ModelOutputError as error:
                last_result = {"raw_text": error.raw_text}
                last_error = error
            else:
                try:
                    return _validate_review_result(last_result, holes, shapes, labels, width, height,
                                                   sam_comparison)
                except ValueError as error:
                    last_error = error
            # Update the original call rather than creating a fictitious model
            # request. Transport/configuration failures bypass this retry path.
            record = call_state.get("record")
            if record is not None:
                record.update(status="rejected", error=str(last_error),
                              retrying=attempt < MAX_PROTOCOL_RETRIES)
                _emit_trace(record)
        finally:
            _review_call.reset(token)
    raise ValueError(f"GPT 协议校验重试 {MAX_PROTOCOL_RETRIES} 次后仍失败：{last_error}")


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
    # The full-frame evidence already contains every ID.  Send only the four
    # largest actionable regions as high-resolution crops so a debugging
    # record cannot make the model request exceed its image budget.
    ranked = sorted(enumerate(holes, 1), key=lambda item: item[1]["area"], reverse=True)[:4]
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


def _fingerprint(shapes):
    geometry = [{key: shape.get(key) for key in ("id", "label", "points", "holes", "parts")}
                for shape in shapes]
    return hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()


class AnnotationAgent:
    """Observe, choose tools, execute, and inspect again within a fixed budget."""

    def __init__(self, config, image, description, labels, max_rounds=100, on_progress=None,
                 reference=None):
        if type(max_rounds) is not int or not 1 <= max_rounds <= MAX_REVIEW_ROUNDS:
            raise ValueError("Agent 最大轮数须为 1～100")
        self.config, self.image, self.description, self.labels = config, image, description, labels
        self.reference = reference
        self.width, self.height = Image.open(io.BytesIO(image)).size
        self.max_rounds = max_rounds
        self.on_progress = on_progress or (lambda event: None)
        self.shapes, self.audit, self.queue, self.applied = [], [], [], []
        self.next_id = 1
        self.generated = 0
        self.failed_operations = {}
        self.sam_pending = None
        self.sam_points_by_shape = {}
        self.sam_fallback = set()

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
            shapes.append(shape)
            self.next_id += 1
        return shapes

    def generate(self):
        for item in self.description.get("prompts", []):
            if (not isinstance(item, dict) or item.get("label") not in self.labels or
                    not isinstance(item.get("text"), str) or not item["text"].strip()):
                raise ValueError("SAM 3 提示缺少有效类别或文本")
            self.progress("generating", prompt=item["text"], label=item["label"])
            shapes = self.candidates(sam_text(self.image, item["text"], self.config), item["label"])
            self.shapes.extend(shapes)
            self.generated += len(shapes)
            self.progress("generated")

    def execute(self):
        stop_reason = "max_rounds"
        converged = False
        try:
            self.generate()
            seen = {_fingerprint(self.shapes)}
            for number in range(1, self.max_rounds + 1):
                self.progress("observing", round=number)
                pending = self.sam_pending
                comparison = ({"shape_id": pending["shape_id"],
                    "before_image": candidate_overlay(self.image, pending["before"])} if pending else None)
                holes = candidate_holes(self.shapes, self.labels, self.width, self.height)
                gap_images = gap_detail_images(self.image, self.shapes, holes)
                decision = review(self.config, self.image, candidate_overlay(self.image, self.shapes),
                    self.description, [_shape_summary(shape) for shape in self.shapes],
                    self.labels, self.audit, reference=self.reference, sam_comparison=comparison,
                    hole_image=hole_overlay(self.image, holes, self.shapes), holes=holes,
                    gap_images=gap_images)
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
                            previous_points = self.sam_points_by_shape.get(ids[0], [])
                            combined = {(point["x"], point["y"]): point for point in previous_points + points}
                            all_points = list(combined.values())
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
                                next_id=self.next_id, gap_regions=holes)
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
                    stop_reason = "done" if converged else "error"
                    break
                # A proposed SAM edit must be assessed even if it revisits an
                # earlier geometry. Failed/no-change prompts get one fresh
                # observation so GPT can choose an OpenCV fallback.
                if self.sam_pending is not None or sam_failed:
                    continue
                if after == before:
                    stop_reason = "no_progress"
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
        if not converged and stop_reason != "error":
            self.queue.append({"reason": {"max_rounds": "达到最大轮数，最新候选尚未确认",
                "no_progress": "本轮未产生有效修改", "cycle": "检测到循环修改"}[stop_reason]})
        if self.sam_pending is not None:
            self.shapes = copy.deepcopy(self.sam_pending["before"])
            self.queue.append({"reason": "SAM 3 点提示结果尚未经过下一轮 GPT 复核，已恢复修改前候选"})
            converged = False
            self.sam_pending = None
        result = {"shapes": self.shapes, "candidates_added": self.generated, "applied": self.applied,
            "review_queue": self.queue, "agent": {"converged": converged,
            "rounds": len(self.audit), "max_rounds": self.max_rounds,
            "stop_reason": stop_reason, "audit": self.audit}}
        self.progress("finished", result=result)
        return result


def run(config, image, description, labels, max_rounds=100, on_progress=None, reference=None):
    return AnnotationAgent(config, image, description, labels, max_rounds, on_progress,
                           reference=reference).execute()
