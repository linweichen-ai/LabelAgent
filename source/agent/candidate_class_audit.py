"""仅产生候选类别盲审证据，不修改候选或生成人工真值。"""
from __future__ import annotations

import hashlib
import io
import json
import math
from typing import Callable

from PIL import Image, ImageDraw


class CandidateClassProtocolError(ValueError):
    """模型输出不符合盲审协议。"""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _ring(points):
    if not isinstance(points, list) or len(points) < 3:
        raise ValueError("候选轮廓至少需要三个点")
    result = []
    for point in points:
        if (not isinstance(point, (list, tuple)) or len(point) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) for v in point)):
            raise ValueError("候选轮廓坐标必须是有限数值对")
        result.append([float(v) for v in point])
    return result


def _parts(candidate):
    parts = candidate.get("parts") or [candidate]
    return [{"points": _ring(part.get("points")),
             "holes": [_ring(hole) for hole in part.get("holes", [])]} for part in parts]


def candidate_fingerprint(candidate: dict) -> str:
    if type(candidate.get("id")) is not int or not candidate.get("label"):
        raise ValueError("候选需要整数 ID 和类别")
    return _sha256(_canonical_bytes({
        "id": candidate["id"], "label": candidate["label"],
        "score": candidate.get("score"), "parts": _parts(candidate)}))


HIGH_RISK_LABELS = frozenset({"ego", "person", "other_robot"})


def audit_high_risk_candidates(config, image_bytes, candidates, labels, class_definitions,
                               chat, *, check_cancelled=lambda: None,
                               tolerate_errors=False, candidate_ids=None):
    """共享盲审流程；仅在明确允许时将失败转为人工复核证据。"""
    artifact_sha = _sha256(_canonical_bytes(candidates))
    audits, queue = [], []
    for candidate in candidates:
        if (candidate.get("label") not in HIGH_RISK_LABELS
                or candidate_ids is not None and candidate.get("id") not in candidate_ids):
            continue
        check_cancelled()
        try:
            result = audit_candidate_class_blind(
                config, image_bytes, candidate, labels, class_definitions, chat)
            check_cancelled()
        except Exception as error:
            check_cancelled()
            if not tolerate_errors:
                raise
            result = {
                "schema_version": "labelagent.blind_candidate_class_audit.v1",
                "candidate_id": candidate.get("id"),
                "candidate_sha256": candidate_fingerprint(candidate),
                "route": "human_review_required", "candidate_mutated": False,
                "human_truth": False, "error": str(error)[:300]}
        result["candidate_artifact_sha256"] = artifact_sha
        result["audit_sha256"] = _sha256(_canonical_bytes({
            key: value for key, value in result.items() if key != "audit_sha256"}))
        audits.append(result)
        if result["route"] == "human_review_required":
            queue.append({
                "reason_code": ("candidate_class_blind_audit_failed" if result.get("error")
                                else "candidate_class_blind_review_required"),
                "route": "human_review_required", "source": "candidate_class_audit",
                "candidate_id": candidate["id"], "shape_ids": [candidate["id"]],
                "candidate_sha256": result["candidate_sha256"],
                "candidate_artifact_sha256": artifact_sha,
                "audit_sha256": result["audit_sha256"],
                "classification": result.get("decision", {}).get("classification"),
                "reason": result.get("decision", {}).get("reason", "候选类别盲审失败，请人工检查"),
                **({"error": result["error"]} if result.get("error") else {})})
    return audits, queue


def prepare_candidate_evidence(image_bytes: bytes, candidate: dict) -> dict:
    with Image.open(io.BytesIO(image_bytes)) as opened:
        source = opened.convert("RGB")
    parts = _parts(candidate)
    points = [point for part in parts for point in part["points"]]
    left, top = (min(p[i] for p in points) for i in (0, 1))
    right, bottom = (max(p[i] for p in points) for i in (0, 1))
    padding = max(24, round(max(1, right - left, bottom - top) * 0.45))
    box = (max(0, int(left) - padding), max(0, int(top) - padding),
           min(source.width, int(right) + padding + 1),
           min(source.height, int(bottom) + padding + 1))
    if right < 0 or bottom < 0 or left >= source.width or top >= source.height or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("候选轮廓不在源图像内")
    original = source.crop(box)
    mask = Image.new("L", source.size, 0)
    for part in parts:
        component = Image.new("L", source.size, 0)
        draw = ImageDraw.Draw(component)
        draw.polygon([tuple(p) for p in part["points"]], fill=255)
        for hole in part["holes"]:
            draw.polygon([tuple(p) for p in hole], fill=0)
        mask.paste(255, mask=component)
    local_mask = mask.crop(box).point(lambda value: 72 if value else 0)
    marked = Image.composite(Image.new("RGB", original.size, (0, 224, 190)), original, local_mask)

    def encode(image):
        output = io.BytesIO()
        image.save(output, "JPEG", quality=92, optimize=False, progressive=False)
        return output.getvalue()

    original_bytes, marked_bytes = encode(original), encode(marked)
    return {"original": original_bytes, "marked": marked_bytes,
            "crop_box_xyxy": list(box), "source_size": list(source.size),
            "current_label_shown": False,
            "original_sha256": _sha256(original_bytes), "marked_sha256": _sha256(marked_bytes)}


def blind_candidate_class_schema(labels: list[str]) -> dict:
    if not labels or len(set(labels)) != len(labels) or any(not isinstance(x, str) or not x.strip() for x in labels):
        raise ValueError("类别表必须包含不重复的非空字符串")
    properties = {
        "classification": {"type": "string", "enum": ["known_label", "uncertain", "conflict"]},
        "observed_label": {"anyOf": [{"type": "string", "enum": labels}, {"type": "null"}]},
        "candidate_supported": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason_code": {"type": "string", "enum": ["visible_known_label", "insufficient_visual_evidence", "mixed_or_multiple_objects", "ontology_conflict"]},
        "visible_cues": {"type": "array", "items": {"type": "string", "minLength": 1, "maxLength": 100}, "maxItems": 5},
        "reason": {"type": "string", "minLength": 1, "maxLength": 300},
    }
    return {"type": "object", "properties": properties, "required": sorted(properties), "additionalProperties": False}


def build_blind_candidate_class_prompt(candidate_id, labels, class_definitions):
    identity_rules = (
        " For person, require visible human anatomy inside the cyan mask; a driver/operator is still person."
        " For ego, require visible host robot body/chassis/forks inside the mask; viewpoint alone never proves ego."
        " For other_robot, require a distinct robot or mobile platform inside the mask, separate from ego and people."
        " A mask mixing a person with vehicle/robot structure is uncertain."
    )
    return (
        "Independently classify exactly one highlighted machine candidate. Image 1 is an uncolored "
        "local crop. Image 2 is the same crop with only the candidate mask highlighted in cyan; "
        "holes are excluded. Cyan is not an object color. No current class is provided: do not "
        "guess a previous label. Use known_label only for one visually supported allowed class. "
        "For tiny, blurred, occluded, mixed or weak evidence use uncertain and observed_label=null. "
        "If no allowed class fits use conflict and observed_label=null. This is candidate-only "
        "evidence, never human truth. Return only the strict JSON object."
        + identity_rules + "\nInput: "
        + json.dumps({"candidate_id": candidate_id, "allowed_labels": labels,
                      "class_definitions": {label: class_definitions[label] for label in labels
                                            if isinstance(class_definitions.get(label), str)}}, ensure_ascii=False))


def validate_blind_candidate_class_decision(value, labels):
    schema = blind_candidate_class_schema(labels)
    if not isinstance(value, dict) or set(value) != set(schema["required"]):
        raise CandidateClassProtocolError("盲审输出字段不合规")
    for key in ("classification", "reason_code"):
        if value[key] not in schema["properties"][key]["enum"]:
            raise CandidateClassProtocolError("盲审分类或原因码不合规")
    confidence = value["confidence"]
    if (type(value["candidate_supported"]) is not bool or isinstance(confidence, bool)
            or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1):
        raise CandidateClassProtocolError("盲审支持标记或置信度不合规")
    cues = value["visible_cues"]
    if (not isinstance(cues, list) or len(cues) > 5
            or any(not isinstance(cue, str) or not cue.strip() or len(cue) > 100 for cue in cues)
            or not isinstance(value["reason"], str) or not value["reason"].strip() or len(value["reason"]) > 300):
        raise CandidateClassProtocolError("盲审可见证据或原因不合规")
    normalized_cues = [" ".join(cue.casefold().split()) for cue in cues]
    if len(set(normalized_cues)) != len(normalized_cues):
        raise CandidateClassProtocolError("盲审可见证据重复")
    if value["classification"] == "known_label":
        if (value["observed_label"] not in labels or not value["candidate_supported"]
                or value["reason_code"] != "visible_known_label" or not cues):
            raise CandidateClassProtocolError("已知类别缺少有效支持")
    elif (value["candidate_supported"] or value["observed_label"] is not None
          or value["reason_code"] == "visible_known_label"):
        raise CandidateClassProtocolError("不确定或冲突不能声明已知类别")
    return value


def audit_candidate_class_blind(config: dict, image_bytes: bytes, candidate: dict,
                                labels: list[str], class_definitions: dict[str, str],
                                chat: Callable, minimum_confidence: float = 0.90,
                                candidate_artifact_sha256: str | None = None) -> dict:
    fingerprint = candidate_fingerprint(candidate)
    if candidate["label"] not in labels:
        raise ValueError("候选类别不在注册表中")
    evidence = prepare_candidate_evidence(image_bytes, candidate)
    prompt = build_blind_candidate_class_prompt(candidate["id"], labels, class_definitions)
    decision = validate_blind_candidate_class_decision(chat(
        config, prompt, evidence["original"], extra_images=(evidence["marked"],),
        response_schema=blind_candidate_class_schema(labels)), labels)
    matches = (decision["classification"] == "known_label"
               and decision["observed_label"] == candidate["label"]
               and decision["confidence"] >= minimum_confidence)
    result = {
        "schema_version": "labelagent.blind_candidate_class_audit.v1",
        "candidate_id": candidate["id"], "candidate_sha256": fingerprint,
        "candidate_artifact_sha256": candidate_artifact_sha256,
        "bound_current_label": candidate["label"], "source_image_sha256": _sha256(image_bytes),
        "crop": {"box_xyxy": evidence["crop_box_xyxy"], "source_size": evidence["source_size"],
                 "original_sha256": evidence["original_sha256"], "marked_sha256": evidence["marked_sha256"],
                 "current_label_shown_to_model": False},
        "prompt_sha256": _sha256(prompt.encode("utf-8")), "decision": decision,
        "route": "candidate_preserved" if matches else "human_review_required",
        "minimum_confidence": minimum_confidence,
        "operation_proposal": None, "candidate_mutated": False, "human_truth": False,
        "approved_for_training": False, "approved_for_deployment": False,
    }
    result["audit_sha256"] = _sha256(_canonical_bytes(result))
    return result
