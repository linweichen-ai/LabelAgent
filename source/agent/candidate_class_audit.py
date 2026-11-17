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
                               tolerate_errors=False, candidate_ids=None,
                               batch_candidates=False):
    """共享盲审流程；仅在明确允许时将失败转为人工复核证据。"""
    artifact_sha = _sha256(_canonical_bytes(candidates))
    audits, queue = [], []
    selected = [candidate for candidate in candidates
                if candidate.get("label") in HIGH_RISK_LABELS
                and (candidate_ids is None or candidate.get("id") in candidate_ids)]
    pending = []
    if batch_candidates and len(selected) > 1:
        check_cancelled()
        try:
            pending = audit_candidate_classes_blind_batch(
                config, image_bytes, selected, labels, class_definitions, chat)
            check_cancelled()
        except Exception as error:
            check_cancelled()
            if not tolerate_errors:
                raise
            batch_error = str(error)[:300]
            fallback_allowed = (isinstance(error, CandidateClassProtocolError)
                                or "大模型未返回有效 JSON" in str(error))
            if fallback_allowed:
                for candidate in selected:
                    check_cancelled()
                    try:
                        result = audit_candidate_class_blind(
                            config, image_bytes, candidate, labels, class_definitions, chat)
                        result["batch_fallback_reason"] = batch_error
                        result["batch_fallback_used"] = True
                    except Exception as fallback_error:
                        result = {
                            "schema_version": "labelagent.blind_candidate_class_audit.v1",
                            "candidate_id": candidate.get("id"),
                            "candidate_sha256": candidate_fingerprint(candidate),
                            "route": "human_review_required", "candidate_mutated": False,
                            "human_truth": False, "error": str(fallback_error)[:300],
                            "batch_fallback_reason": batch_error,
                            "batch_fallback_used": True}
                    pending.append(result)
            else:
                pending = [{
                    "schema_version": "labelagent.blind_candidate_class_audit.v1",
                    "candidate_id": candidate.get("id"),
                    "candidate_sha256": candidate_fingerprint(candidate),
                    "route": "human_review_required", "candidate_mutated": False,
                    "human_truth": False, "error": batch_error}
                    for candidate in selected]
    else:
        for candidate in selected:
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
            pending.append(result)
    by_id = {candidate["id"]: candidate for candidate in selected}
    for result in pending:
        candidate = by_id[result["candidate_id"]]
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
        "visible_cues": {"type": "array", "items": {"type": "string", "minLength": 1, "maxLength": 100}, "maxItems": 3},
        "reason": {"type": "string", "minLength": 1, "maxLength": 180},
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


def blind_candidate_class_batch_schema(labels: list[str], candidate_ids: list[int]) -> dict:
    if (not candidate_ids or len(set(candidate_ids)) != len(candidate_ids)
            or any(type(value) is not int for value in candidate_ids)):
        raise ValueError("批量盲审候选 ID 必须是不重复整数")
    decision = blind_candidate_class_schema(labels)
    properties = {"candidate_id": {"type": "integer", "enum": candidate_ids},
                  **decision["properties"]}
    item = {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}
    return {"type": "object", "properties": {
        "decisions": {"type": "array", "items": item,
                      "minItems": len(candidate_ids), "maxItems": len(candidate_ids)}},
        "required": ["decisions"], "additionalProperties": False}


def build_blind_candidate_class_batch_prompt(candidate_ids, labels, class_definitions):
    return (
        "Independently classify each highlighted machine candidate. Images are paired in the "
        "candidate_id order below: first an uncolored crop, then the same crop with only that "
        "candidate mask highlighted in cyan. Cyan is not an object color. No current class is "
        "provided. Return exactly one decision per candidate_id and never combine candidates. "
        "For known_label, observed_label must be an allowed label, candidate_supported must be true, "
        "and reason_code must be visible_known_label. For uncertain or conflict, observed_label must "
        "be null, candidate_supported must be false, and reason_code must not be visible_known_label. "
        "For person require visible human anatomy; for ego require visible host robot body, chassis, "
        "or forks; for other_robot require a distinct robot or mobile platform. Tiny, blurred, "
        "occluded, mixed, or weak evidence is uncertain. This is candidate-only evidence, never "
        "human truth. Return only the strict JSON object.\nInput: "
        + json.dumps({"candidate_ids": candidate_ids, "allowed_labels": labels,
                      "class_definitions": {label: class_definitions[label] for label in labels
                                            if isinstance(class_definitions.get(label), str)}},
                     ensure_ascii=False))


def validate_blind_candidate_class_batch(value, labels, candidate_ids, *,
                                         return_repairs=False):
    if not isinstance(value, dict) or set(value) != {"decisions"}:
        raise CandidateClassProtocolError("批量盲审输出字段不合规")
    decisions = value["decisions"]
    if not isinstance(decisions, list) or len(decisions) != len(candidate_ids):
        raise CandidateClassProtocolError("批量盲审必须逐候选返回结果")
    by_id, repairs = {}, {}
    for item in decisions:
        if not isinstance(item, dict) or type(item.get("candidate_id")) is not int:
            raise CandidateClassProtocolError("批量盲审候选 ID 不合规")
        candidate_id = item["candidate_id"]
        if candidate_id not in candidate_ids or candidate_id in by_id:
            raise CandidateClassProtocolError("批量盲审候选 ID 缺失、重复或越界")
        decision = {key: value for key, value in item.items() if key != "candidate_id"}
        try:
            decision = validate_blind_candidate_class_decision(decision, labels)
        except CandidateClassProtocolError:
            # A local model may attach a best-guess label to an explicitly
            # uncertain decision.  Only remove contradictory claims; never
            # upgrade uncertainty or invent positive evidence.
            if decision.get("classification") not in {"uncertain", "conflict"}:
                raise
            changed = []
            if decision.get("observed_label") is not None:
                decision["observed_label"] = None
                changed.append("observed_label_cleared")
            if decision.get("candidate_supported") is not False:
                decision["candidate_supported"] = False
                changed.append("candidate_supported_downgraded")
            if decision.get("reason_code") == "visible_known_label":
                decision["reason_code"] = "insufficient_visual_evidence"
                changed.append("reason_code_downgraded")
            decision = validate_blind_candidate_class_decision(decision, labels)
            if changed:
                repairs[candidate_id] = changed
        by_id[candidate_id] = decision
    if set(by_id) != set(candidate_ids):
        raise CandidateClassProtocolError("批量盲审未覆盖全部候选")
    ordered = [by_id[candidate_id] for candidate_id in candidate_ids]
    return (ordered, repairs) if return_repairs else ordered


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
    if (not isinstance(cues, list) or len(cues) > 3
            or any(not isinstance(cue, str) or not cue.strip() or len(cue) > 100 for cue in cues)
            or not isinstance(value["reason"], str) or not value["reason"].strip() or len(value["reason"]) > 180):
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


def _candidate_audit_result(image_bytes, candidate, evidence, prompt, decision,
                            minimum_confidence, candidate_artifact_sha256=None,
                            batch_candidate_ids=None):
    matches = (decision["classification"] == "known_label"
               and decision["observed_label"] == candidate["label"]
               and decision["confidence"] >= minimum_confidence)
    result = {
        "schema_version": "labelagent.blind_candidate_class_audit.v1",
        "candidate_id": candidate["id"],
        "candidate_sha256": candidate_fingerprint(candidate),
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
    if batch_candidate_ids is not None:
        result["batch_candidate_ids"] = list(batch_candidate_ids)
        result["batch_size"] = len(batch_candidate_ids)
    result["audit_sha256"] = _sha256(_canonical_bytes(result))
    return result


def audit_candidate_classes_blind_batch(config: dict, image_bytes: bytes,
                                        candidates: list[dict], labels: list[str],
                                        class_definitions: dict[str, str], chat: Callable,
                                        minimum_confidence: float = 0.90) -> list[dict]:
    """Audit multiple candidates in one request while retaining per-candidate binding."""
    if len(candidates) < 2:
        raise ValueError("批量盲审至少需要两个候选")
    candidate_ids = [candidate.get("id") for candidate in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("批量盲审候选 ID 重复")
    if any(candidate.get("label") not in labels for candidate in candidates):
        raise ValueError("候选类别不在注册表中")
    evidence = [prepare_candidate_evidence(image_bytes, candidate)
                for candidate in candidates]
    prompt = build_blind_candidate_class_batch_prompt(
        candidate_ids, labels, class_definitions)
    images = []
    for item in evidence:
        images.extend((item["original"], item["marked"]))
    raw = chat(config, prompt, images[0], extra_images=tuple(images[1:]),
               response_schema=blind_candidate_class_batch_schema(labels, candidate_ids))
    decisions, repairs = validate_blind_candidate_class_batch(
        raw, labels, candidate_ids, return_repairs=True)
    results = [_candidate_audit_result(
        image_bytes, candidate, item, prompt, decision, minimum_confidence,
        batch_candidate_ids=candidate_ids)
        for candidate, item, decision in zip(candidates, evidence, decisions)]
    for result in results:
        if result["candidate_id"] in repairs:
            result["protocol_repairs"] = repairs[result["candidate_id"]]
            result["protocol_repair_policy"] = "conservative_uncertainty_downgrade_only"
            result["audit_sha256"] = _sha256(_canonical_bytes({
                key: value for key, value in result.items() if key != "audit_sha256"}))
    return results


def audit_candidate_class_blind(config: dict, image_bytes: bytes, candidate: dict,
                                labels: list[str], class_definitions: dict[str, str],
                                chat: Callable, minimum_confidence: float = 0.90,
                                candidate_artifact_sha256: str | None = None) -> dict:
    if candidate["label"] not in labels:
        raise ValueError("候选类别不在注册表中")
    evidence = prepare_candidate_evidence(image_bytes, candidate)
    prompt = build_blind_candidate_class_prompt(candidate["id"], labels, class_definitions)
    decision = validate_blind_candidate_class_decision(chat(
        config, prompt, evidence["original"], extra_images=(evidence["marked"],),
        response_schema=blind_candidate_class_schema(labels)), labels)
    return _candidate_audit_result(
        image_bytes, candidate, evidence, prompt, decision, minimum_confidence,
        candidate_artifact_sha256=candidate_artifact_sha256)
