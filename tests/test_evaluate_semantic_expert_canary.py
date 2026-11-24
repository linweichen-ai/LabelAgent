import hashlib
import json

import pytest

from tools.evaluate_semantic_expert_canary import evaluate_canary


def _fixture():
    plan = {
        "schema_version": "labelagent.semantic-expert-canary-plan.v1",
        "canary_count": 8, "remainder_count": 11,
        "canary": [], "remainder": [],
        "technical_gate": {
            "all_requests_passed": True,
            "usage_complete": True,
            "maximum_median_latency_ms": 60000,
            "maximum_average_tokens_per_call": 30000,
            "automatic_retry_enabled": False,
            "maximum_attempts_per_image": 1,
        },
        "candidate_only": True, "human_truth": False,
        "approved_for_training": False, "test24_used": False,
    }
    samples = []
    for index in range(19):
        row = {
            "sample_id": f"sample-{index}",
            "cohort": "hard",
            "source_image_sha256": f"{index + 1:064x}",
            "execution_stage": "canary" if index < 8 else "remainder",
        }
        samples.append(row)
        (plan["canary"] if index < 8 else plan["remainder"]).append({
            "sample_id": row["sample_id"], "cohort": row["cohort"],
            "source_image_sha256": row["source_image_sha256"],
        })
    plan_bytes = json.dumps(plan, sort_keys=True).encode()
    queue = {
        "schema_version": "labelagent.mixed-hard-expert-shadow-queue.v1",
        "semantic_canary_plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "sample_count": 19, "canary_count": 8, "remainder_count": 11,
        "candidate_only": True, "human_truth": False,
        "approved_for_training": False, "test24_used": False,
        "samples": samples,
    }
    queue_bytes = json.dumps(queue, sort_keys=True).encode()
    rows = []
    for row in samples[:8]:
        rows.append({
            "sample_id": row["sample_id"], "cohort": row["cohort"],
            "source_image_sha256": row["source_image_sha256"],
            "status": "passed", "latency_ms": 20000,
            "external_attempts": 1,
            "expert_metadata": {"external_vlm": {"usage": {
                "prompt_tokens": 300, "completion_tokens": 20,
                "total_tokens": 320,
            }}},
            "candidate_only": True, "human_truth": False,
            "approved_for_training": False, "test24_used": False,
        })
    result = {
        "schema_version": "labelagent.mixed-hard-external-shadow.v1",
        "queue_sha256": hashlib.sha256(queue_bytes).hexdigest(),
        "status": "in_progress", "samples_total": 19,
        "samples_attempted": 8, "samples_passed": 8, "samples_failed": 0,
        "external_attempts": 8, "automatic_retry_enabled": False,
        "latency_median_ms": 20000,
        "external_usage": {
            "samples_with_usage": 8, "samples_missing_usage": 0,
            "complete": True, "prompt_tokens": 2400,
            "completion_tokens": 160, "total_tokens": 2560,
        },
        "rows": rows, "candidate_only": True, "human_truth": False,
        "approved_for_training": False, "automatic_truth_write": False,
        "test24_used": False,
    }
    return queue_bytes, plan_bytes, result


def _run(queue_bytes, plan_bytes, result):
    return evaluate_canary(
        queue_bytes, plan_bytes, json.dumps(result, sort_keys=True).encode())


def test_passes_exact_eight_sample_canary_without_dispatching_remainder():
    queue, plan, result = _fixture()
    report = _run(queue, plan, result)
    assert report["technical_gate_passed"] is True
    assert report["remainder_technically_eligible"] is True
    assert report["remainder_dispatched"] is False
    assert report["additional_external_calls_performed"] == 0
    assert report["automatic_continuation_enabled"] is False
    assert report["candidate_only"] is True
    assert report["human_truth"] is False
    assert report["test24_used"] is False


@pytest.mark.parametrize("mutation, message", [
    (lambda r: r.update(queue_sha256="0" * 64), "result identity"),
    (lambda r: r["rows"].reverse(), "order or membership"),
    (lambda r: r["rows"].append(dict(r["rows"][0], sample_id="sample-8")),
     "result identity"),
    (lambda r: r["rows"][0].update(human_truth=True), "row identity"),
])
def test_rejects_identity_membership_or_boundary_drift(mutation, message):
    queue, plan, result = _fixture()
    mutation(result)
    with pytest.raises(ValueError, match=message):
        _run(queue, plan, result)


def test_reports_latency_or_usage_failure_without_raising():
    queue, plan, result = _fixture()
    result["latency_median_ms"] = 60001
    result["external_usage"]["total_tokens"] = 240001
    report = _run(queue, plan, result)
    assert report["technical_gate_passed"] is False
    assert report["remainder_technically_eligible"] is False
    assert report["technical_gate"]["reasons"] == [
        "canary_latency_exceeds_guard", "canary_tokens_exceed_guard"]


def test_rejects_plan_or_queue_drift():
    queue_bytes, plan_bytes, result = _fixture()
    plan = json.loads(plan_bytes)
    plan["technical_gate"]["maximum_median_latency_ms"] = 61000
    with pytest.raises(ValueError, match="queue identity"):
        _run(queue_bytes, json.dumps(plan, sort_keys=True).encode(), result)

    queue = json.loads(queue_bytes)
    queue["samples"][8]["execution_stage"] = "canary"
    changed_queue = json.dumps(queue, sort_keys=True).encode()
    result["queue_sha256"] = hashlib.sha256(changed_queue).hexdigest()
    with pytest.raises(ValueError, match="canary order"):
        _run(changed_queue, plan_bytes, result)
