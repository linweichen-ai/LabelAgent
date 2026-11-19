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


def _polygon_area(points):
    return abs(sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))) / 2.0)


def candidate_requires_class_audit(candidate: dict) -> bool:
    """Return whether a candidate needs a focused, label-blind semantic check.

    The three historically high-risk labels are always checked.  Pallets are
    checked when a sizeable, nearly solid, box-like mask could be cargo or a
    container rather than the load-bearing pallet underneath it.  Extremely
    thin, long masks are also checked because the observed failure mode can be
    a forklift tine, floor seam, tape or shadow carrying a pallet label.  The
    focused check remains candidate-only and never mutates the mask.
    """
    label = str(candidate.get("label", ""))
    if label in HIGH_RISK_LABELS:
        return True
    if label.casefold() != "pallet":
        return False
    try:
        parts = _parts(candidate)
    except (TypeError, ValueError):
        return True
    points = [point for part in parts for point in part["points"]]
    if not points:
        return True
    left, top = (min(point[index] for point in points) for index in (0, 1))
    right, bottom = (max(point[index] for point in points) for index in (0, 1))
    width, height = right - left + 1, bottom - top + 1
    bbox_area = width * height
    foreground_area = sum(
        _polygon_area(part["points"])
        - sum(_polygon_area(hole) for hole in part["holes"])
        for part in parts)
    long_side, short_side = max(width, height), min(width, height)
    extreme_thin_artifact = (
        long_side >= 120 and short_side >= 4
        and short_side / max(long_side, 1) <= .08
        and foreground_area >= 500)
    if extreme_thin_artifact:
        return True
    if width < 120 or height < 80 or bbox_area < 5000:
        return False
    fill_ratio = foreground_area / max(bbox_area, 1)
    width_height_ratio = width / max(height, 1)
    # Cargo/container masks are often not perfect rectangles: handles, straps
    # and foreground occlusion brought the observed false pallet down to 0.848.
    # Keep the gate below that case while the size/aspect guards leave ordinary
    # side-view slatted pallets on the fast path. Only the much thinner <=.08
    # artifact band above is diverted to semantic review.
    return fill_ratio >= 0.82 and width_height_ratio <= 2.10


def candidate_audit_images(candidate: dict, evidence: dict) -> tuple[bytes, tuple[bytes, ...]]:
    """Keep box-like pallet checks free of misleading nearby pallet pixels."""
    if str(candidate.get("label", "")).casefold() == "pallet":
        return evidence["isolated"], ()
    return evidence["original"], (evidence["marked"], evidence["isolated"])


def audit_high_risk_candidates(config, image_bytes, candidates, labels, class_definitions,
                               chat, *, check_cancelled=lambda: None,
                               tolerate_errors=False, candidate_ids=None,
                               on_progress=None, batch_chat=None):
    """共享盲审流程；仅在明确允许时将失败转为人工复核证据。"""
    artifact_sha = _sha256(_canonical_bytes(candidates))
    audits, queue = [], []
    targets = [candidate for candidate in candidates
               if candidate_requires_class_audit(candidate)
               and (candidate_ids is None or candidate.get("id") in candidate_ids)]
    def run_sequential(candidate, position):
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
        if on_progress is not None:
            on_progress(position, len(targets), candidate, result)
        check_cancelled()

    # True batching stays restricted to independently encoded groups of three:
    # that is the only batch size with measured latency gain and accepted
    # semantic parity.  Longer target lists use as many full triples as they
    # contain; one or two remainders keep the established sequential path.
    # Each triple fails closed independently, so one uncertain or malformed
    # group is discarded and rerun sequentially without invalidating earlier
    # protocol-valid groups.
    batched_target_count = ((len(targets) // 3) * 3
                            if batch_chat is not None else 0)
    for start in range(0, batched_target_count, 3):
        group = targets[start:start + 3]
        check_cancelled()
        try:
            prepared = []
            for candidate in group:
                evidence = prepare_candidate_evidence(image_bytes, candidate)
                prompt = build_blind_candidate_class_prompt(
                    candidate["id"], labels, class_definitions,
                    isolated_only=str(candidate.get("label", "")).casefold() == "pallet")
                prepared.append((candidate, evidence, prompt))
            decisions = batch_chat(
                config,
                [{"instruction": prompt,
                  "image": candidate_audit_images(candidate, evidence)[0],
                  "extra_images": candidate_audit_images(candidate, evidence)[1]}
                 for _, evidence, prompt in prepared],
                response_schema=blind_candidate_class_schema(labels),
                max_tokens=384)
            if not isinstance(decisions, list) or len(decisions) != 3:
                raise CandidateClassProtocolError("批量盲审结果数量不匹配")
            decisions = [validate_blind_candidate_class_decision(value, labels)
                         for value in decisions]
            if not all(
                    decision["classification"] == "known_label"
                    and decision["observed_label"] == candidate["label"]
                    and decision["candidate_supported"]
                    and decision["confidence"] >= 0.90
                    for (candidate, _, _), decision in zip(prepared, decisions)):
                raise CandidateClassProtocolError("批量盲审未全部高置信确认原候选类别")
            for offset, ((candidate, evidence, prompt), decision) in enumerate(
                    zip(prepared, decisions)):
                result = _candidate_class_audit_result(
                    image_bytes, candidate, evidence, prompt, decision,
                    candidate_artifact_sha256=artifact_sha,
                    execution_mode="independent_batch_3")
                audits.append(result)
                if on_progress is not None:
                    on_progress(start + offset + 1, len(targets), candidate, result)
            check_cancelled()
        except Exception:
            check_cancelled()
            for offset, candidate in enumerate(group):
                run_sequential(candidate, start + offset + 1)
    for position, candidate in enumerate(
            targets[batched_target_count:], batched_target_count + 1):
        run_sequential(candidate, position)
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
    local_binary_mask = mask.crop(box)
    local_mask = local_binary_mask.point(lambda value: 72 if value else 0)
    marked = Image.composite(Image.new("RGB", original.size, (0, 224, 190)), original, local_mask)
    isolated = Image.composite(
        original, Image.new("RGB", original.size, (24, 24, 24)), local_binary_mask)

    def encode(image):
        output = io.BytesIO()
        image.save(output, "JPEG", quality=92, optimize=False, progressive=False)
        return output.getvalue()

    original_bytes, marked_bytes, isolated_bytes = (
        encode(original), encode(marked), encode(isolated))
    return {"original": original_bytes, "marked": marked_bytes,
            "isolated": isolated_bytes,
            "crop_box_xyxy": list(box), "source_size": list(source.size),
            "current_label_shown": False,
            "original_sha256": _sha256(original_bytes), "marked_sha256": _sha256(marked_bytes),
            "isolated_sha256": _sha256(isolated_bytes)}


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


def build_blind_candidate_class_prompt(
        candidate_id, labels, class_definitions, isolated_only=False):
    if isolated_only:
        return (
            "Classify only the non-dark pixels in this isolated crop. Everything outside the "
            "candidate mask is darkened and must be ignored. Do not infer an object below, behind, "
            "or near the visible pixels. A pallet requires visible slats, deck, feet, or fork-entry "
            "structure inside the non-dark region. A solid shipping box or container is not a "
            "pallet. If the non-dark pixels mainly show a solid box/container and no allowed class "
            "fits, return conflict with candidate_supported=false and observed_label=null. "
            "For uncertain use the same false/null field consistency. "
            "Field consistency is mandatory: known_label requires an allowed "
            "observed_label, candidate_supported=true, and reason_code=visible_known_label; "
            "uncertain or conflict requires observed_label=null, candidate_supported=false, "
            "and a reason_code other than visible_known_label. Never combine visible support "
            "for an allowed label with classification=conflict. Return only the strict JSON "
            "object. Input: "
            + json.dumps({"candidate_id": candidate_id, "allowed_labels": labels},
                         ensure_ascii=False))
    identity_rules = (
        " For person, require visible human anatomy inside the cyan mask; a driver/operator is still person."
        " For ego, require visible host robot body/chassis/forks inside the mask; viewpoint alone never proves ego."
        " For other_robot, require a distinct robot or mobile platform inside the mask, separate from ego and people."
        " For pallet, require visible load-bearing pallet slats, deck, feet, or fork-entry structure inside the cyan mask;"
        " a box, container, or stacked cargo is not pallet merely because a pallet is nearby or underneath it."
        " If the cyan mask mainly covers cargo instead of the pallet structure, use uncertain or conflict."
        " A mask mixing a person with vehicle/robot structure is uncertain."
    )
    evidence_rules = (
        "Image 1 is an uncolored local crop. Image 2 highlights only the candidate mask in cyan; "
        "holes are excluded. Image 3 keeps only pixels inside that mask and darkens all surrounding "
        "objects. Cyan is not an object color. Classify the masked pixels, not a nearby object. "
    )
    return (
        "Independently classify exactly one highlighted machine candidate. "
        + evidence_rules
        + "No current class is provided: do not "
        "guess a previous label. Use known_label only for one visually supported allowed class. "
        "For tiny, blurred, occluded, mixed or weak evidence use uncertain, observed_label=null, "
        "candidate_supported=false, and a non-visible_known_label reason_code. "
        "If no allowed class fits use conflict with the same null/false field consistency. "
        "Field consistency is mandatory: known_label requires an allowed observed_label, "
        "candidate_supported=true, and reason_code=visible_known_label; uncertain or conflict "
        "requires observed_label=null, candidate_supported=false, and a reason_code other than "
        "visible_known_label. Never combine visible support for an allowed label with "
        "classification=conflict. "
        "This is candidate-only "
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


def _repair_non_known_reason_code(value):
    """Normalize only a redundant reason code, never a semantic decision.

    When classification, observed label, and support already agree that the
    candidate is not a known label, ``visible_known_label`` is structurally
    impossible.  Replacing only that reason code avoids a second model call
    without changing the class, label, confidence, cues, or human-review route.
    Any broader contradiction remains invalid and uses the bounded retry.
    """
    if (not isinstance(value, dict)
            or value.get("classification") not in {"uncertain", "conflict"}
            or value.get("observed_label") is not None
            or value.get("candidate_supported") is not False
            or value.get("reason_code") != "visible_known_label"):
        return None
    repaired = dict(value)
    repaired["reason_code"] = (
        "ontology_conflict" if value["classification"] == "conflict"
        else "insufficient_visual_evidence")
    return repaired


def _candidate_class_audit_result(
        image_bytes, candidate, evidence, prompt, decision,
        minimum_confidence=0.90, candidate_artifact_sha256=None,
        execution_mode="sequential", protocol_retry_count=0,
        protocol_repair=None):
    matches = (decision["classification"] == "known_label"
               and decision["observed_label"] == candidate["label"]
               and decision["confidence"] >= minimum_confidence)
    result = {
        "schema_version": "labelagent.blind_candidate_class_audit.v1",
        "candidate_id": candidate["id"],
        "candidate_sha256": candidate_fingerprint(candidate),
        "candidate_artifact_sha256": candidate_artifact_sha256,
        "bound_current_label": candidate["label"],
        "source_image_sha256": _sha256(image_bytes),
        "crop": {"box_xyxy": evidence["crop_box_xyxy"],
                 "source_size": evidence["source_size"],
                 "original_sha256": evidence["original_sha256"],
                 "marked_sha256": evidence["marked_sha256"],
                 "isolated_sha256": evidence["isolated_sha256"],
                 "current_label_shown_to_model": False},
        "prompt_sha256": _sha256(prompt.encode("utf-8")),
        "decision": decision,
        "route": "candidate_preserved" if matches else "human_review_required",
        "minimum_confidence": minimum_confidence,
        "execution_mode": execution_mode,
        "protocol_retry_count": protocol_retry_count,
        "protocol_repair": protocol_repair,
        "operation_proposal": None, "candidate_mutated": False,
        "human_truth": False, "approved_for_training": False,
        "approved_for_deployment": False,
    }
    result["audit_sha256"] = _sha256(_canonical_bytes(result))
    return result


def audit_candidate_class_blind(config: dict, image_bytes: bytes, candidate: dict,
                                labels: list[str], class_definitions: dict[str, str],
                                chat: Callable, minimum_confidence: float = 0.90,
                                candidate_artifact_sha256: str | None = None) -> dict:
    if candidate["label"] not in labels:
        raise ValueError("候选类别不在注册表中")
    evidence = prepare_candidate_evidence(image_bytes, candidate)
    isolated_only = str(candidate.get("label", "")).casefold() == "pallet"
    prompt = build_blind_candidate_class_prompt(
        candidate["id"], labels, class_definitions, isolated_only=isolated_only)
    primary_image, extra_images = candidate_audit_images(candidate, evidence)
    response_schema = blind_candidate_class_schema(labels)
    protocol_retry_count = 0
    protocol_repair = None
    try:
        raw_decision = chat(
            config, prompt, primary_image, extra_images=extra_images,
            response_schema=response_schema)
        try:
            decision = validate_blind_candidate_class_decision(
                raw_decision, labels)
        except CandidateClassProtocolError:
            repaired = _repair_non_known_reason_code(raw_decision)
            if repaired is None:
                raise
            decision = validate_blind_candidate_class_decision(repaired, labels)
            protocol_repair = "non_known_reason_code_normalized"
    except CandidateClassProtocolError as error:
        # One bounded retry repairs contradictory JSON fields without changing
        # the visual task or inventing a class.  The retry deliberately does
        # not echo the invalid answer, which could anchor a new decision to an
        # unsupported label.  A second invalid response still fails closed and
        # the caller routes it to human review.
        protocol_retry_count = 1
        retry_prompt = prompt + (
            "\nThe previous JSON violated the response-field consistency contract: "
            + str(error)
            + ". Discard that JSON and independently return one fresh strict JSON object. "
              "For classification=known_label, observed_label must be exactly one allowed "
              "label, candidate_supported must be true, and reason_code must be "
              "visible_known_label. For classification=uncertain or conflict, "
              "observed_label must be null, candidate_supported must be false, and "
              "reason_code must not be visible_known_label. Keep the same visual "
              "evidence standard; do not change the judgment merely to satisfy fields.")
        decision = validate_blind_candidate_class_decision(chat(
            config, retry_prompt, primary_image, extra_images=extra_images,
            response_schema=response_schema), labels)
    return _candidate_class_audit_result(
        image_bytes, candidate, evidence, prompt, decision,
        minimum_confidence=minimum_confidence,
        candidate_artifact_sha256=candidate_artifact_sha256,
        protocol_retry_count=protocol_retry_count,
        protocol_repair=protocol_repair)
