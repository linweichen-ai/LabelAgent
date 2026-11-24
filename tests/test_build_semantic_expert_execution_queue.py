import hashlib
import json

import pytest

from tools.build_semantic_expert_execution_queue import build_execution_queue


def _plan():
    def row(index, cohort):
        return {
            "sample_id": f"sample-{index}", "cohort": cohort,
            "source_image_sha256": f"{index + 1:064x}",
            "copied_image_sha256": f"{index + 101:064x}",
            "route_source": "baseline_hard_route",
            "external_upload_authorized": False, "candidate_only": True,
        }
    return {
        "schema_version": "labelagent.semantic-expert-canary-plan.v1",
        "status": "authorization_required",
        "external_candidate_count": 3, "canary_count": 2,
        "remainder_count": 1,
        "canary": [row(0, "positive"), row(1, "negative")],
        "remainder": [row(2, "conflict")],
        "external_authorization_granted": False,
        "dispatch_performed": False, "external_upload_performed": False,
        "factory_data_uploaded": False, "candidate_only": True,
        "human_truth": False, "approved_for_training": False,
        "test24_used": False, "production_deployed": False,
    }


def test_builds_runner_compatible_queue_bound_to_copied_jpeg():
    encoded = json.dumps(_plan()).encode()
    result = build_execution_queue(encoded)
    assert result["schema_version"] == "labelagent.mixed-hard-expert-shadow-queue.v1"
    assert result["semantic_canary_plan_sha256"] == hashlib.sha256(encoded).hexdigest()
    assert result["sample_count"] == 3
    assert result["canary_count"] == 2
    assert [row["execution_stage"] for row in result["samples"]] == [
        "canary", "canary", "remainder"]
    assert result["samples"][0]["source_image_sha256"] == f"{101:064x}"
    assert result["samples"][0]["original_source_image_sha256"] == f"{1:064x}"
    assert result["external_upload_authorized"] is False
    assert result["ready_for_shadow_execution"] is False
    assert result["image_paths_included"] is False


def test_rejects_authorization_path_or_count_drift():
    plan = _plan()
    plan["canary"][0]["external_upload_authorized"] = True
    with pytest.raises(ValueError, match="row identity or boundary"):
        build_execution_queue(json.dumps(plan).encode())

    plan = _plan()
    plan["canary"][0]["image_path"] = "private.jpg"
    with pytest.raises(ValueError, match="row identity or boundary"):
        build_execution_queue(json.dumps(plan).encode())

    plan = _plan()
    plan["external_candidate_count"] = 4
    with pytest.raises(ValueError, match="plan identity or boundary"):
        build_execution_queue(json.dumps(plan).encode())
