"""Deterministic, upload-free routing advice for optional expert VLM review."""
import copy


SCHEMA_VERSION = "labelagent.expert-shadow-routing.v1"
HARD_STOP_REASONS = frozenset({
    "error", "max_rounds", "no_progress", "review_deferred",
})
PROTOCOL_REASON_MARKERS = (
    "protocol", "invalid_json", "model_output", "decoder_constraint",
)


def _bounded_round_threshold(config):
    value = config.get("llm_expert_shadow_round_threshold", 4)
    return value if type(value) is int and 1 <= value <= 100 else 4


def evaluate_expert_shadow_route(config, *, discovery=None, agent=None,
                                 review_queue=None, class_audits=None,
                                 candidate_labels=None):
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
    candidate_labels = [str(label) for label in (candidate_labels or [])
                        if isinstance(label, str) and label]
    reasons = []
    seen = set()

    def add(code, severity, evidence=None):
        if code in seen:
            return
        seen.add(code)
        item = {"code": code, "severity": severity}
        if evidence is not None:
            item["evidence"] = copy.deepcopy(evidence)
        reasons.append(item)

    unknown = discovery.get("unknown_candidates") or []
    if unknown:
        add("open_set_candidate", "high", {"count": len(unknown)})
    conflicts = discovery.get("registry_conflicts") or []
    if conflicts:
        add("registry_conflict", "high", {"count": len(conflicts)})
    uncertain = discovery.get("uncertain_labels") or []
    if uncertain:
        add("visibility_uncertain", "medium",
            {"labels": sorted({str(label) for label in uncertain})[:20]})
    assessments = discovery.get("assessments") or []
    low_confidence = []
    for item in assessments:
        if not isinstance(item, dict):
            continue
        confidence = item.get("confidence")
        if (type(confidence) in (int, float) and not isinstance(confidence, bool)
                and confidence < 0.65 and isinstance(item.get("label"), str)):
            low_confidence.append(item["label"])
    if low_confidence:
        add("low_visibility_confidence", "medium",
            {"labels": sorted(set(low_confidence))[:20], "threshold": 0.65})

    stop_reason = str(agent.get("stop_reason") or "")
    if stop_reason in HARD_STOP_REASONS:
        add("agent_nonconvergence", "high", {"stop_reason": stop_reason})
    rounds = agent.get("rounds")
    threshold = _bounded_round_threshold(config)
    if type(rounds) is int and rounds >= threshold:
        add("high_review_round_count", "medium",
            {"rounds": rounds, "threshold": threshold})
    retry_count = agent.get("protocol_retry_count")
    if type(retry_count) is int and retry_count > 0:
        add("protocol_retry_used", "medium", {"count": retry_count})

    queue_codes = []
    for item in review_queue:
        if not isinstance(item, dict):
            continue
        code = str(item.get("reason_code") or item.get("code") or "")
        if code:
            queue_codes.append(code)
        if any(marker in code.casefold() for marker in PROTOCOL_REASON_MARKERS):
            add("protocol_failure", "high", {"reason_code": code[:120]})
    if queue_codes:
        add("human_review_queue", "medium", {
            "count": len(queue_codes),
            "reason_codes": sorted(set(queue_codes))[:20],
        })

    conflicting_audits = 0
    for audit in class_audits:
        if (isinstance(audit, dict)
                and audit.get("route") not in {None, "candidate_only", "preserve"}):
            conflicting_audits += 1
    if conflicting_audits:
        add("class_audit_conflict", "high", {"count": conflicting_audits})

    difficult = sorted(set(candidate_labels) & {"pallet", "slope"})
    if difficult and reasons:
        add("priority_class_involved", "context", {"labels": difficult})

    severity = {"context": 0, "medium": 1, "high": 2}
    score = max((severity[item["severity"]] for item in reasons), default=0)
    recommended = score > 0
    return {
        "schema_version": SCHEMA_VERSION,
        "enabled": True,
        "dispatch_performed": False,
        "external_upload_performed": False,
        "recommended": recommended,
        "route": ("external_expert_shadow_recommended"
                  if recommended else "local_result_sufficient"),
        "priority": "high" if score == 2 else "normal",
        "reasons": reasons,
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
    }
