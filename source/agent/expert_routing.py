"""Deterministic, upload-free routing advice for optional expert VLM review.

The router consumes bounded metadata only.  It never receives image bytes and
never dispatches a provider request.  Its output is therefore safe to attach to
machine-candidate provenance before a separate, explicitly authorized expert
review path exists.
"""
import copy
import re


SCHEMA_VERSION = "labelagent.expert-shadow-routing.v2"
EXECUTION_RECEIPT_SCHEMA_VERSION = "labelagent.expert-route-receipt.v1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
HARD_STOP_REASONS = frozenset({
    "error", "max_rounds", "no_progress", "review_deferred",
})
PROTOCOL_REASON_MARKERS = (
    "protocol", "invalid_json", "model_output", "decoder_constraint",
)
PRIORITY_LABELS = frozenset({
    "pallet", "slope", "车辆", "vehicle", "other_robot", "cargo_floor",
})
HARD_SIGNAL_CODES = frozenset({
    "edge_slope_candidate",
    "priority_semantic_uncertainty",
    "class_audit_conflict",
    "agent_nonconvergence",
    "open_set_candidate",
    "registry_conflict",
    "protocol_failure",
    "manual_priority",
})
GEOMETRY_LABELS = PRIORITY_LABELS | {"person"}
GEOMETRY_LABEL_CASEFOLDS = frozenset(value.casefold() for value in GEOMETRY_LABELS)


def _bounded_int(config, key, default, minimum, maximum):
    value = config.get(key, default)
    return value if type(value) is int and minimum <= value <= maximum else default


def _bounded_round_threshold(config):
    return _bounded_int(
        config, "llm_expert_shadow_round_threshold", 2, 1, 100)


def _signal_weight(config, name, default):
    """Return a bounded, explicitly configurable shadow-routing weight."""
    return _bounded_int(
        config, f"llm_expert_shadow_weight_{name}", default, 0, 20)


def _assessment_items(discovery):
    assessments = discovery.get("assessments") or []
    if isinstance(assessments, dict):
        return [dict(value, label=label) for label, value in assessments.items()
                if isinstance(value, dict)]
    return [item for item in assessments if isinstance(item, dict)]


def _shape_points(shape):
    points = []
    if not isinstance(shape, dict):
        return points
    for part in shape.get("parts") or [shape]:
        if not isinstance(part, dict):
            continue
        for point in part.get("points") or []:
            if (isinstance(point, (list, tuple)) and len(point) == 2
                    and all(type(value) in (int, float)
                            and not isinstance(value, bool) for value in point)):
                points.append((float(point[0]), float(point[1])))
    return points


def _candidate_geometry(candidates, image_size, edge_margin_percent,
                        small_candidate_percent):
    if (not isinstance(candidates, list)
            or not isinstance(image_size, (list, tuple)) or len(image_size) != 2
            or any(type(value) is not int or value <= 0 for value in image_size)):
        return {"eligible_candidates": 0, "edge_count": 0, "small_count": 0,
                "edge_slope_count": 0, "edge_labels": [], "small_labels": []}
    width, height = image_size
    margin_x = width * edge_margin_percent / 100
    margin_y = height * edge_margin_percent / 100
    image_area = width * height
    edge_labels, small_labels, eligible, edge_slope_count = [], [], 0, 0
    for shape in candidates:
        if not isinstance(shape, dict):
            continue
        label = str(shape.get("label") or "")
        if label.casefold() not in GEOMETRY_LABEL_CASEFOLDS:
            continue
        points = _shape_points(shape)
        if not points:
            continue
        eligible += 1
        xs, ys = [point[0] for point in points], [point[1] for point in points]
        left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
        if (left <= margin_x or right >= width - 1 - margin_x
                or top <= margin_y or bottom >= height - 1 - margin_y):
            edge_labels.append(label)
            if label.casefold() == "slope":
                edge_slope_count += 1
        bbox_percent = ((right - left + 1) * (bottom - top + 1)
                        / image_area * 100)
        if bbox_percent <= small_candidate_percent:
            small_labels.append(label)
    return {
        "eligible_candidates": eligible,
        "edge_count": len(edge_labels),
        "small_count": len(small_labels),
        "edge_slope_count": edge_slope_count,
        "edge_labels": sorted(set(edge_labels))[:20],
        "small_labels": sorted(set(small_labels))[:20],
    }


def _sam_instability(agent, filter_threshold, mask_iou_percent,
                     area_change_percent):
    generation = agent.get("sam_generation")
    generation = generation if isinstance(generation, dict) else {}
    explicit = agent.get("sam_stability")
    explicit = explicit if isinstance(explicit, dict) else {}
    filters = agent.get("candidate_filters")
    filter_count = len(filters) if isinstance(filters, list) else 0
    reasons = []
    mode = str(generation.get("mode") or "")
    if mode == "sequential_fallback" or generation.get("error"):
        reasons.append("generation_fallback")
    semantic = generation.get("semantic_fallbacks")
    if isinstance(semantic, list) and semantic:
        reasons.append("semantic_fallback")
    if filter_count >= filter_threshold:
        reasons.append("many_filtered_candidates")
    stable = explicit.get("stable")
    if stable is False:
        reasons.append("explicit_unstable")
    mask_iou = explicit.get("mask_iou")
    if (type(mask_iou) in (int, float) and not isinstance(mask_iou, bool)
            and mask_iou * 100 < mask_iou_percent):
        reasons.append("low_mask_iou")
    area_change = explicit.get("area_change_ratio")
    if (type(area_change) in (int, float) and not isinstance(area_change, bool)
            and abs(area_change) * 100 > area_change_percent):
        reasons.append("large_area_change")
    instance_delta = explicit.get("instance_count_delta")
    if type(instance_delta) is int and instance_delta != 0:
        reasons.append("instance_count_changed")
    return {
        "unstable": bool(reasons),
        "reasons": sorted(set(reasons)),
        "generation_mode": mode or None,
        "candidate_filter_count": filter_count,
    }


def evaluate_expert_shadow_route(config, *, discovery=None, agent=None,
                                 review_queue=None, class_audits=None,
                                 candidate_labels=None, candidates=None,
                                 image_size=None, manual_priority=False,
                                 local_model_latency_ms=None,
                                 sam_result_reused=True):
    """Return a candidate-only escalation recommendation without dispatching.

    The decision contains only bounded metadata. It never receives image bytes,
    never calls a provider, and cannot approve candidates for training.
    """
    if config.get("llm_expert_shadow_routing_enabled") is not True:
        return None
    discovery = discovery if isinstance(discovery, dict) else {}
    agent = agent if isinstance(agent, dict) else {}
    review_queue = review_queue if isinstance(review_queue, list) else []
    class_audits = class_audits if isinstance(class_audits, list) else []
    candidates = candidates if isinstance(candidates, list) else []
    candidate_labels = [str(label) for label in (candidate_labels or [])
                        if isinstance(label, str) and label]
    confidence_percent = _bounded_int(
        config, "llm_expert_shadow_low_confidence_percent", 65, 0, 100)
    score_threshold = _bounded_int(
        config, "llm_expert_shadow_score_threshold", 3, 1, 100)
    edge_margin_percent = _bounded_int(
        config, "llm_expert_shadow_edge_margin_percent", 3, 0, 20)
    small_candidate_percent = _bounded_int(
        config, "llm_expert_shadow_small_candidate_percent", 1, 0, 20)
    sam_filter_threshold = _bounded_int(
        config, "llm_expert_shadow_sam_filter_threshold", 3, 1, 200)
    sam_mask_iou_percent = _bounded_int(
        config, "llm_expert_shadow_sam_mask_iou_percent", 70, 0, 100)
    sam_area_change_percent = _bounded_int(
        config, "llm_expert_shadow_sam_area_change_percent", 25, 0, 500)
    weights = {
        "open_set": _signal_weight(config, "open_set", 3),
        "registry_conflict": _signal_weight(config, "registry_conflict", 3),
        "visibility_uncertain": _signal_weight(config, "visibility_uncertain", 1),
        "low_confidence": _signal_weight(config, "low_confidence", 2),
        "agent_nonconvergence": _signal_weight(config, "agent_nonconvergence", 3),
        "high_review_rounds": _signal_weight(config, "high_review_rounds", 2),
        "protocol_retry": _signal_weight(config, "protocol_retry", 1),
        "sam_instability": _signal_weight(config, "sam_instability", 3),
        # Edge and small objects are common in factory imagery.  They remain
        # visible routing evidence, but no longer escalate on geometry alone.
        "edge_candidate": _signal_weight(config, "edge_candidate", 1),
        "edge_slope": _signal_weight(config, "edge_slope", 2),
        "small_candidate": _signal_weight(config, "small_candidate", 1),
        "manual_priority": _signal_weight(config, "manual_priority", 4),
        "protocol_failure": _signal_weight(config, "protocol_failure", 3),
        "human_review_queue": _signal_weight(config, "human_review_queue", 1),
        "class_audit_conflict": _signal_weight(config, "class_audit_conflict", 3),
    }
    reasons = []
    seen = set()

    def add(code, severity, points, evidence=None):
        if code in seen:
            return
        seen.add(code)
        item = {"code": code, "severity": severity, "points": points}
        if evidence is not None:
            item["evidence"] = copy.deepcopy(evidence)
        reasons.append(item)

    unknown = discovery.get("unknown_candidates") or []
    if unknown:
        add("open_set_candidate", "high", weights["open_set"],
            {"count": len(unknown)})
    conflicts = discovery.get("registry_conflicts") or []
    if conflicts:
        add("registry_conflict", "high", weights["registry_conflict"],
            {"count": len(conflicts)})
    uncertain = discovery.get("uncertain_labels") or []
    if uncertain:
        uncertain_labels = sorted({str(label) for label in uncertain})[:20]
        add("visibility_uncertain", "medium", weights["visibility_uncertain"],
            {"labels": uncertain_labels})
        priority_uncertain = sorted(set(uncertain_labels) & PRIORITY_LABELS)
        if priority_uncertain:
            # Do not double-count the same semantic uncertainty in the score.
            # Priority-class uncertainty is nevertheless an explicit expert
            # escalation condition, independent of a potentially overconfident
            # scalar confidence emitted by the local model.
            add("priority_semantic_uncertainty", "high", 0,
                {"labels": priority_uncertain})
    low_confidence = []
    for item in _assessment_items(discovery):
        confidence = item.get("confidence")
        if (type(confidence) in (int, float) and not isinstance(confidence, bool)
                and confidence * 100 < confidence_percent
                and isinstance(item.get("label"), str)):
            low_confidence.append(item["label"])
    if low_confidence:
        add("low_visibility_confidence", "medium", weights["low_confidence"],
            {"labels": sorted(set(low_confidence))[:20],
             "threshold_percent": confidence_percent})

    stop_reason = str(agent.get("stop_reason") or "")
    if stop_reason in HARD_STOP_REASONS:
        add("agent_nonconvergence", "high", weights["agent_nonconvergence"],
            {"stop_reason": stop_reason})
    rounds = agent.get("rounds")
    threshold = _bounded_round_threshold(config)
    if type(rounds) is int and rounds >= threshold:
        add("high_review_round_count", "medium", weights["high_review_rounds"],
            {"rounds": rounds, "threshold": threshold})
    retry_count = agent.get("protocol_retry_count")
    if type(retry_count) is int and retry_count > 0:
        add("protocol_retry_used", "medium", weights["protocol_retry"],
            {"count": retry_count})

    sam = _sam_instability(
        agent, sam_filter_threshold, sam_mask_iou_percent,
        sam_area_change_percent)
    if sam["unstable"]:
        add("sam_candidate_instability", "high", weights["sam_instability"], sam)

    geometry = _candidate_geometry(
        candidates, image_size, edge_margin_percent, small_candidate_percent)
    if geometry["edge_count"]:
        add("edge_partial_candidate", "medium", weights["edge_candidate"], {
            "count": geometry["edge_count"],
            "labels": geometry["edge_labels"],
            "margin_percent": edge_margin_percent,
        })
    if geometry["edge_slope_count"]:
        add("edge_slope_candidate", "high", weights["edge_slope"], {
            "count": geometry["edge_slope_count"],
            "margin_percent": edge_margin_percent,
        })
    if geometry["small_count"]:
        add("small_candidate", "medium", weights["small_candidate"], {
            "count": geometry["small_count"],
            "labels": geometry["small_labels"],
            "bbox_area_threshold_percent": small_candidate_percent,
        })
    if manual_priority is True:
        add("manual_priority", "high", weights["manual_priority"])

    queue_codes = []
    for item in review_queue:
        if not isinstance(item, dict):
            continue
        code = str(item.get("reason_code") or item.get("code") or "")
        if code:
            queue_codes.append(code)
        if any(marker in code.casefold() for marker in PROTOCOL_REASON_MARKERS):
            add("protocol_failure", "high", weights["protocol_failure"],
                {"reason_code": code[:120]})
    if queue_codes:
        add("human_review_queue", "medium", weights["human_review_queue"], {
            "count": len(queue_codes),
            "reason_codes": sorted(set(queue_codes))[:20],
        })

    conflicting_audits = 0
    for audit in class_audits:
        if (isinstance(audit, dict)
                and audit.get("route") not in {None, "candidate_only", "preserve"}):
            conflicting_audits += 1
    if conflicting_audits:
        add("class_audit_conflict", "high", weights["class_audit_conflict"],
            {"count": conflicting_audits})

    difficult = sorted(set(candidate_labels) & PRIORITY_LABELS)
    if difficult and reasons:
        add("priority_class_involved", "context", 0, {"labels": difficult})

    score = sum(item["points"] for item in reasons)
    hard_signal_codes = sorted(
        item["code"] for item in reasons if item["code"] in HARD_SIGNAL_CODES)
    selected_by_score = score >= score_threshold
    # A high score is useful for combining several weak signals, but it must
    # never suppress one of the explicit escalation conditions from the
    # project contract.  This remains upload-free shadow advice only.
    selected_by_hard_signal = bool(hard_signal_codes)
    recommended = selected_by_score or selected_by_hard_signal
    latency = (round(float(local_model_latency_ms), 3)
               if type(local_model_latency_ms) in (int, float)
               and not isinstance(local_model_latency_ms, bool)
               and local_model_latency_ms >= 0 else None)
    return {
        "schema_version": SCHEMA_VERSION,
        "enabled": True,
        "dispatch_performed": False,
        "external_upload_performed": False,
        "recommended": recommended,
        "route": ("external_expert_shadow_recommended"
                  if recommended else "local_result_sufficient"),
        "priority": ("high" if any(item["severity"] == "high" and item["points"] > 0
                                   for item in reasons)
                     else "normal"),
        "difficulty_score": score,
        "score_threshold": score_threshold,
        "selection": {
            "selected_by_score": selected_by_score,
            "selected_by_hard_signal": selected_by_hard_signal,
            "hard_signal_codes": hard_signal_codes,
        },
        "thresholds": {
            "low_confidence_percent": confidence_percent,
            "agent_rounds": threshold,
            "edge_margin_percent": edge_margin_percent,
            "small_candidate_bbox_percent": small_candidate_percent,
            "sam_filter_count": sam_filter_threshold,
            "sam_mask_iou_percent": sam_mask_iou_percent,
            "sam_area_change_percent": sam_area_change_percent,
        },
        "weights": weights,
        "signals": {
            "low_confidence_labels": sorted(set(low_confidence))[:20],
            "registry_conflict_count": len(conflicts),
            "sam_stability": sam,
            "agent_rounds": rounds if type(rounds) is int else None,
            "edge_candidate_count": geometry["edge_count"],
            "edge_slope_candidate_count": geometry["edge_slope_count"],
            "small_candidate_count": geometry["small_count"],
            "unknown_candidate_count": len(unknown),
            "priority_labels": difficult,
            "manual_priority": manual_priority is True,
        },
        "reasons": reasons,
        "timing": {
            "local_model_latency_ms": latency,
            "external_model_latency_ms": None,
        },
        "fallback_performed": False,
        "sam_result_reused": sam_result_reused is True,
        "cache": {
            "external_result_cache_hit": False,
            "external_request_key": None,
        },
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
    }


def finalize_expert_route_receipt(shadow_decision, *, execution=None,
                                  external_model_latency_ms=None,
                                  external_metadata=None):
    """Merge bounded expert execution facts into one candidate-only receipt.

    This function performs no model request and accepts no image, prompt or raw
    provider response.  It is intentionally strict so an external review can
    be audited without turning a machine result into truth or training data.
    """
    if not isinstance(shadow_decision, dict):
        raise ValueError("shadow decision must be an object")
    if shadow_decision.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("shadow decision schema differs")
    if (shadow_decision.get("candidate_only") is not True
            or shadow_decision.get("human_truth") is not False
            or shadow_decision.get("approved_for_training") is not False):
        raise ValueError("shadow decision safety boundary differs")
    if (shadow_decision.get("dispatch_performed") is not False
            or shadow_decision.get("external_upload_performed") is not False):
        raise ValueError("shadow decision must precede expert execution")

    execution = execution if isinstance(execution, dict) else {}
    metadata = external_metadata if isinstance(external_metadata, dict) else {}
    attempted = execution.get("external_attempted") is True
    dispatched = execution.get("dispatch_performed") is True
    uploaded = execution.get("external_upload_performed") is True
    fallback = execution.get("fallback_performed") is True
    sam_reused = execution.get(
        "sam_result_reused", shadow_decision.get("sam_result_reused")) is True
    cache_hit = metadata.get("expert_cache_hit") is True

    if dispatched and not attempted:
        raise ValueError("expert dispatch requires an attempted route")
    if uploaded and not dispatched:
        raise ValueError("external upload requires a dispatched request")
    if fallback and not attempted:
        raise ValueError("expert fallback requires an attempted route")
    if attempted and shadow_decision.get("recommended") is not True:
        raise ValueError("expert execution requires a recommended shadow route")
    if cache_hit and (dispatched or uploaded):
        raise ValueError("cached expert result cannot be a new dispatch")
    if cache_hit and not attempted:
        raise ValueError("cached expert result requires an attempted route")
    if attempted and not (dispatched or cache_hit or fallback):
        raise ValueError("attempted expert route has no terminal outcome")

    timing = shadow_decision.get("timing")
    if (not isinstance(timing, dict)
            or "local_model_latency_ms" not in timing
            or not isinstance(shadow_decision.get("route"), str)
            or type(shadow_decision.get("recommended")) is not bool):
        raise ValueError("shadow decision audit fields are incomplete")

    latency = None
    if external_model_latency_ms is not None:
        if (type(external_model_latency_ms) not in (int, float)
                or isinstance(external_model_latency_ms, bool)
                or external_model_latency_ms < 0):
            raise ValueError("external model latency must be non-negative")
        latency = round(float(external_model_latency_ms), 3)
    if attempted and latency is None:
        raise ValueError("attempted expert route requires measured latency")
    if not attempted and latency is not None:
        raise ValueError("unattempted expert route cannot have external latency")

    fallback_reason = execution.get("fallback_reason")
    if fallback:
        if not isinstance(fallback_reason, str) or not fallback_reason.strip():
            raise ValueError("expert fallback requires a bounded reason")
        fallback_reason = fallback_reason.strip()[:160]
    elif fallback_reason not in (None, ""):
        raise ValueError("fallback reason requires fallback_performed")
    else:
        fallback_reason = None

    request_key = metadata.get("expert_request_key")
    if request_key is not None:
        request_key = str(request_key).lower()
        if SHA256_PATTERN.fullmatch(request_key) is None:
            raise ValueError("expert request key must be a SHA-256 digest")
    retry_count = metadata.get("expert_retry_count", 0)
    if type(retry_count) is not int or retry_count < 0 or retry_count > 8:
        raise ValueError("expert retry count is invalid")

    receipt = copy.deepcopy(shadow_decision)
    if not attempted:
        final_route = shadow_decision["route"]
    elif fallback:
        final_route = "local_fallback_after_expert"
    elif cache_hit:
        final_route = "external_expert_cache"
    else:
        final_route = "external_expert_completed"
    receipt.update(
        schema_version=EXECUTION_RECEIPT_SCHEMA_VERSION,
        shadow_schema_version=SCHEMA_VERSION,
        route=final_route,
        dispatch_performed=dispatched,
        external_upload_performed=uploaded,
        fallback_performed=fallback,
        fallback_reason=fallback_reason,
        sam_result_reused=sam_reused,
        execution={
            "external_attempted": attempted,
            "external_calls": 1 if dispatched else 0,
            "retry_count": retry_count,
        },
        cache={
            "external_result_cache_hit": cache_hit,
            "external_request_key": request_key,
        },
    )
    receipt["timing"]["external_model_latency_ms"] = latency
    receipt["candidate_only"] = True
    receipt["human_truth"] = False
    receipt["approved_for_training"] = False
    return receipt
