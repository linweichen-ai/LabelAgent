"""Strict human-referenced A/B summary for external expert VLM reviews."""
import math
import statistics

from .expert_review import VERDICTS


INPUT_SCHEMA_VERSION = "labelagent.external-expert-ab-input.v1"
OUTPUT_SCHEMA_VERSION = "labelagent.external-expert-ab-report.v1"
RUN_STATUSES = frozenset({"passed", "failed", "fallback"})


def _ratio(numerator, denominator):
    return round(numerator / denominator, 6) if denominator else None


def _percentile(values, percentile):
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return round(ordered[index], 3)


def _string_set(values, field):
    if not isinstance(values, list) or any(
            not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{field} must be a string list")
    normalized = {value.strip().casefold() for value in values}
    if len(normalized) != len(values):
        raise ValueError(f"{field} must be unique")
    return normalized


def _validate_reference(payload):
    if payload.get("schema_version") != INPUT_SCHEMA_VERSION:
        raise ValueError("A/B input schema version is invalid")
    partition = payload.get("partition")
    if (not isinstance(partition, str) or not partition.strip()
            or "test24" in partition.casefold() or payload.get("test24") is not False):
        raise ValueError("test24 is evaluation-only and cannot enter expert A/B")
    synthetic_fixture = payload.get("synthetic_fixture")
    if type(synthetic_fixture) is not bool:
        raise ValueError("synthetic_fixture must be an explicit boolean")
    if synthetic_fixture:
        if (payload.get("human_reference") is not False
                or payload.get("reviewer_confirmed") is not False):
            raise ValueError("synthetic fixture cannot claim human review")
    elif (payload.get("human_reference") is not True
          or payload.get("reviewer_confirmed") is not True):
        raise ValueError("quality metrics require reviewer-confirmed human reference")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not 1 <= len(samples) <= 500:
        raise ValueError("A/B requires 1-500 human-referenced samples")
    references = {}
    for item in samples:
        if not isinstance(item, dict) or set(item) != {
                "sample_id", "candidate_verdicts", "missing_labels",
                "new_label_names"}:
            raise ValueError("human reference sample fields are invalid")
        sample_id = item["sample_id"]
        verdicts = item["candidate_verdicts"]
        if (not isinstance(sample_id, str) or not sample_id.strip()
                or sample_id in references or not isinstance(verdicts, dict)
                or not verdicts):
            raise ValueError("human reference sample identity is invalid")
        normalized_verdicts = {}
        for candidate_id, verdict in verdicts.items():
            if (not isinstance(candidate_id, str) or not candidate_id.strip()
                    or verdict not in VERDICTS):
                raise ValueError("human candidate verdict is invalid")
            normalized_verdicts[candidate_id] = verdict
        references[sample_id] = {
            "candidate_verdicts": normalized_verdicts,
            "missing_labels": _string_set(item["missing_labels"], "missing_labels"),
            "new_label_names": _string_set(item["new_label_names"], "new_label_names"),
        }
    return partition.strip(), references, synthetic_fixture


def _validate_run(run, references):
    if not isinstance(run, dict):
        raise ValueError("expert A/B run must be an object")
    required = {"model", "sample_id", "status", "latency_ms",
                "estimated_cost_usd", "candidate_only", "human_truth",
                "approved_for_training"}
    if not required.issubset(run):
        raise ValueError("expert A/B run fields are incomplete")
    model, sample_id, status = run["model"], run["sample_id"], run["status"]
    if (not isinstance(model, str) or not model.strip()
            or sample_id not in references or status not in RUN_STATUSES):
        raise ValueError("expert A/B run identity or status is invalid")
    if (run["candidate_only"] is not True or run["human_truth"] is not False
            or run["approved_for_training"] is not False):
        raise ValueError("expert A/B run violates candidate-only boundaries")
    latency, cost = run["latency_ms"], run["estimated_cost_usd"]
    if (type(latency) not in (int, float) or isinstance(latency, bool)
            or latency < 0 or type(cost) not in (int, float)
            or isinstance(cost, bool) or cost < 0):
        raise ValueError("expert A/B latency or cost is invalid")
    if status != "passed":
        if "review" in run:
            raise ValueError("failed/fallback run cannot contain a review")
        return model.strip(), sample_id, None
    review = run.get("review")
    if not isinstance(review, dict) or not {
            "candidate_reviews", "missing_labels", "new_label_candidates"}.issubset(review):
        raise ValueError("passed expert A/B run requires a review")
    candidate_reviews = review["candidate_reviews"]
    expected = references[sample_id]["candidate_verdicts"]
    if not isinstance(candidate_reviews, list) or len(candidate_reviews) != len(expected):
        raise ValueError("expert A/B candidate review count is invalid")
    predicted = {}
    for item in candidate_reviews:
        if not isinstance(item, dict):
            raise ValueError("expert A/B candidate review is invalid")
        candidate_id, verdict = item.get("candidate_id"), item.get("verdict")
        if candidate_id not in expected or candidate_id in predicted or verdict not in VERDICTS:
            raise ValueError("expert A/B candidate review binding is invalid")
        predicted[candidate_id] = verdict
    missing = _string_set(review["missing_labels"], "predicted missing_labels")
    proposals = review["new_label_candidates"]
    if not isinstance(proposals, list):
        raise ValueError("predicted new_label_candidates must be a list")
    names = []
    for proposal in proposals:
        if not isinstance(proposal, dict) or not isinstance(proposal.get("name"), str):
            raise ValueError("predicted new-label proposal is invalid")
        names.append(proposal["name"])
    new_names = _string_set(names, "predicted new-label names")
    return model.strip(), sample_id, {
        "candidate_verdicts": predicted,
        "missing_labels": missing,
        "new_label_names": new_names,
    }


def build_expert_ab_report(payload):
    if not isinstance(payload, dict):
        raise ValueError("expert A/B input must be an object")
    partition, references, synthetic_fixture = _validate_reference(payload)
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("expert A/B requires model runs")
    by_model, seen = {}, set()
    for run in runs:
        model, sample_id, prediction = _validate_run(run, references)
        identity = (model, sample_id)
        if identity in seen:
            raise ValueError("duplicate model/sample A/B run")
        seen.add(identity)
        metrics = by_model.setdefault(model, {
            "attempts": 0, "passed": 0, "failed": 0, "fallback": 0,
            "latencies": [], "cost": 0.0, "verdict_correct": 0,
            "verdict_total": 0, "missing_tp": 0, "missing_fp": 0,
            "missing_fn": 0, "new_tp": 0, "new_fp": 0, "new_fn": 0,
        })
        metrics["attempts"] += 1
        metrics[run["status"]] += 1
        metrics["latencies"].append(float(run["latency_ms"]))
        metrics["cost"] += float(run["estimated_cost_usd"])
        if prediction is None:
            continue
        reference = references[sample_id]
        metrics["verdict_total"] += len(reference["candidate_verdicts"])
        metrics["verdict_correct"] += sum(
            prediction["candidate_verdicts"][key] == verdict
            for key, verdict in reference["candidate_verdicts"].items())
        for prefix, predicted_key, reference_key in (
                ("missing", "missing_labels", "missing_labels"),
                ("new", "new_label_names", "new_label_names")):
            predicted_set = prediction[predicted_key]
            reference_set = reference[reference_key]
            metrics[f"{prefix}_tp"] += len(predicted_set & reference_set)
            metrics[f"{prefix}_fp"] += len(predicted_set - reference_set)
            metrics[f"{prefix}_fn"] += len(reference_set - predicted_set)
    reports = []
    for model in sorted(by_model):
        item = by_model[model]
        reports.append({
            "model": model,
            "attempted_samples": item["attempts"],
            "structured_review_success_rate": _ratio(item["passed"], item["attempts"]),
            "failed_runs": item["failed"],
            "fallback_runs": item["fallback"],
            "candidate_verdict_exact_match": _ratio(
                item["verdict_correct"], item["verdict_total"]),
            "candidate_verdict_correct": item["verdict_correct"],
            "candidate_verdict_total": item["verdict_total"],
            "missing_label_precision": _ratio(
                item["missing_tp"], item["missing_tp"] + item["missing_fp"]),
            "missing_label_recall": _ratio(
                item["missing_tp"], item["missing_tp"] + item["missing_fn"]),
            "new_label_name_exact_precision": _ratio(
                item["new_tp"], item["new_tp"] + item["new_fp"]),
            "new_label_name_exact_recall": _ratio(
                item["new_tp"], item["new_tp"] + item["new_fn"]),
            "latency_ms_median": round(statistics.median(item["latencies"]), 3),
            "latency_ms_p95": _percentile(item["latencies"], 0.95),
            "estimated_cost_usd_total": round(item["cost"], 6),
        })
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "partition": partition,
        "reference_mode": ("synthetic_fixture" if synthetic_fixture
                           else "reviewer_confirmed_human"),
        "synthetic_fixture": synthetic_fixture,
        "human_reference": not synthetic_fixture,
        "reviewer_confirmed": not synthetic_fixture,
        "test24_used": False,
        "sample_count": len(references),
        "quality_metrics_available": True,
        "real_world_quality_evidence": not synthetic_fixture,
        "models": reports,
        "automatic_winner_selected": False,
        "deployment_decision": "not_evaluated",
        "candidate_only": True,
        "human_truth_written": False,
        "approved_for_training": False,
    }
