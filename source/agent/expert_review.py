"""Bounded candidate-only contract for an explicitly invoked expert VLM."""
import copy
import json


SCHEMA_VERSION = "labelagent.external-expert-review.v1"
VERDICTS = frozenset({
    "supported", "wrong_class", "boundary_issue", "instance_merge",
    "instance_split", "missing_instance", "uncertain",
})
OVERALL_DECISIONS = frozenset({
    "candidate_acceptable", "review_required", "uncertain",
})


def _unique_strings(values, field, maximum):
    if (not isinstance(values, list) or not values or len(values) > maximum
            or any(not isinstance(value, str) or not value.strip()
                   for value in values)):
        raise ValueError(f"{field} must contain 1-{maximum} non-empty strings")
    normalized = [value.strip() for value in values]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field} must be unique")
    return normalized


def expert_review_schema(candidate_ids, labels):
    candidate_ids = _unique_strings(candidate_ids, "candidate_ids", 64)
    labels = _unique_strings(labels, "labels", 64)
    return {
        "type": "object",
        "properties": {
            "overall_decision": {"type": "string",
                                 "enum": sorted(OVERALL_DECISIONS)},
            "candidate_reviews": {
                "type": "array", "minItems": len(candidate_ids),
                "maxItems": len(candidate_ids),
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate_id": {"type": "string", "enum": candidate_ids},
                        "label": {"type": "string", "enum": labels},
                        "verdict": {"type": "string", "enum": sorted(VERDICTS)},
                        "confidence": {"type": "number", "minimum": 0,
                                       "maximum": 1},
                        "reason": {"type": "string", "minLength": 1,
                                   "maxLength": 200},
                        "suggested_label": {
                            "type": ["string", "null"], "enum": [*labels, None]},
                    },
                    "required": ["candidate_id", "label", "verdict",
                                 "confidence", "reason", "suggested_label"],
                    "additionalProperties": False,
                },
            },
            "missing_labels": {
                "type": "array", "uniqueItems": True,
                "items": {"type": "string", "enum": labels},
            },
            "new_label_candidates": {
                "type": "array", "maxItems": 5,
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "minLength": 1,
                                 "maxLength": 80},
                        "confidence": {"type": "number", "minimum": 0,
                                       "maximum": 1},
                        "evidence": {"type": "string", "minLength": 1,
                                     "maxLength": 200},
                    },
                    "required": ["name", "confidence", "evidence"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["overall_decision", "candidate_reviews", "missing_labels",
                     "new_label_candidates"],
        "additionalProperties": False,
    }


def build_expert_review_instruction(candidates, labels, routing_reasons=None):
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 64:
        raise ValueError("expert review requires 1-64 candidates")
    labels = _unique_strings(labels, "labels", 64)
    summary = []
    seen = set()
    for item in candidates:
        if not isinstance(item, dict):
            raise ValueError("candidate summary must be an object")
        candidate_id, label = item.get("id"), item.get("label")
        if (not isinstance(candidate_id, str) or not candidate_id.strip()
                or candidate_id in seen or label not in labels):
            raise ValueError("candidate id/label binding is invalid")
        seen.add(candidate_id)
        summary.append({"candidate_id": candidate_id, "label": label})
    reasons = [str(value)[:80] for value in (routing_reasons or [])
               if isinstance(value, str) and value]
    return (
        "You are reviewing machine-generated factory segmentation candidates. "
        "Image 1 is the original frame. Image 2 is a visualization whose visible "
        "candidate IDs bind to the list below. Review every candidate exactly once. "
        "Judge only visible pixels: class correctness, boundary ownership, merged or "
        "split instances, and clearly missing known-label instances. Do not invent "
        "geometry, do not output editing operations, and do not approve training truth. "
        "A new label is only a proposal for human ontology review. Return only the "
        "required compact JSON object.\nKnown labels: "
        + json.dumps(labels, ensure_ascii=False)
        + "\nCandidates: " + json.dumps(summary, ensure_ascii=False)
        + ("\nEscalation reason codes: " + json.dumps(reasons, ensure_ascii=False)
           if reasons else "")
    )


def validate_expert_review(result, candidates, labels):
    if not isinstance(result, dict) or set(result) != {
            "overall_decision", "candidate_reviews", "missing_labels",
            "new_label_candidates"}:
        raise ValueError("expert review response fields are invalid")
    labels = _unique_strings(labels, "labels", 64)
    expected = {}
    for item in candidates:
        if not isinstance(item, dict):
            raise ValueError("candidate summary must be an object")
        candidate_id, label = item.get("id"), item.get("label")
        if (not isinstance(candidate_id, str) or not candidate_id.strip()
                or candidate_id in expected or label not in labels):
            raise ValueError("candidate id/label binding is invalid")
        expected[candidate_id] = label
    if result["overall_decision"] not in OVERALL_DECISIONS:
        raise ValueError("expert overall decision is invalid")
    reviews = result["candidate_reviews"]
    if not isinstance(reviews, list) or len(reviews) != len(expected):
        raise ValueError("expert must review every candidate exactly once")
    seen = set()
    normalized_reviews = []
    for item in reviews:
        if not isinstance(item, dict) or set(item) != {
                "candidate_id", "label", "verdict", "confidence", "reason",
                "suggested_label"}:
            raise ValueError("expert candidate review fields are invalid")
        candidate_id = item["candidate_id"]
        confidence = item["confidence"]
        reason = item["reason"]
        if (candidate_id not in expected or candidate_id in seen
                or item["label"] != expected[candidate_id]
                or item["verdict"] not in VERDICTS
                or type(confidence) not in (int, float)
                or isinstance(confidence, bool) or not 0 <= confidence <= 1
                or not isinstance(reason, str) or not 1 <= len(reason.strip()) <= 200
                or item["suggested_label"] not in {*labels, None}):
            raise ValueError("expert candidate review is invalid")
        if item["verdict"] != "wrong_class" and item["suggested_label"] is not None:
            raise ValueError("suggested_label is only valid for wrong_class")
        if (item["verdict"] == "wrong_class"
                and item["suggested_label"] == item["label"]):
            raise ValueError("wrong_class suggested_label must change the class")
        seen.add(candidate_id)
        normalized_reviews.append(copy.deepcopy(item))
    missing = result["missing_labels"]
    if (not isinstance(missing, list) or len(missing) != len(set(missing))
            or any(label not in labels for label in missing)):
        raise ValueError("expert missing_labels are invalid")
    proposals = result["new_label_candidates"]
    if not isinstance(proposals, list) or len(proposals) > 5:
        raise ValueError("expert new_label_candidates are invalid")
    known = {label.casefold() for label in labels}
    normalized_proposals = []
    proposal_names = set()
    for item in proposals:
        if not isinstance(item, dict) or set(item) != {"name", "confidence", "evidence"}:
            raise ValueError("expert new-label proposal fields are invalid")
        name, confidence, evidence = item["name"], item["confidence"], item["evidence"]
        if (not isinstance(name, str) or not 1 <= len(name.strip()) <= 80
                or name.strip().casefold() in known
                or name.strip().casefold() in proposal_names
                or type(confidence) not in (int, float) or isinstance(confidence, bool)
                or not 0 <= confidence <= 1 or not isinstance(evidence, str)
                or not 1 <= len(evidence.strip()) <= 200):
            raise ValueError("expert new-label proposal is invalid")
        proposal_names.add(name.strip().casefold())
        normalized_proposals.append(copy.deepcopy(item))
    if (result["overall_decision"] == "candidate_acceptable"
            and (any(item["verdict"] != "supported" for item in normalized_reviews)
                 or missing or normalized_proposals)):
        raise ValueError("candidate_acceptable conflicts with reported issues")
    return {
        "schema_version": SCHEMA_VERSION,
        "overall_decision": result["overall_decision"],
        "candidate_reviews": normalized_reviews,
        "missing_labels": list(missing),
        "new_label_candidates": normalized_proposals,
        "route": "assistant_review_only",
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "candidate_mutated": False,
        "ontology_mutated": False,
    }


def review_external_candidate_shadow(config, original_image, overlay_image,
                                     candidates, labels, routing_reasons=None,
                                     chat_fn=None):
    """Explicitly call the expert; never apply its result to candidates."""
    if chat_fn is None:
        from .workflow import chat_external_expert
        chat_fn = chat_external_expert
    candidate_ids = [item.get("id") for item in candidates]
    instruction = build_expert_review_instruction(
        candidates, labels, routing_reasons=routing_reasons)
    result = chat_fn(
        config, instruction, original_image, extra_images=(overlay_image,),
        response_schema=expert_review_schema(candidate_ids, labels))
    return validate_expert_review(result, candidates, labels)
