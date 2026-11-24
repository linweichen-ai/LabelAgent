import hashlib
import json

import pytest

from tools.build_semantic_expert_authorization_template import (
    build_authorization_template,
)


def _queue():
    samples = [{
        "sample_id": f"sample-{index}", "cohort": "hard",
        "source_image_sha256": f"{index + 1:064x}",
        "external_upload_authorized": False,
    } for index in range(19)]
    return {
        "schema_version": "labelagent.mixed-hard-expert-shadow-queue.v1",
        "sample_count": 19, "canary_count": 8, "remainder_count": 11,
        "maximum_external_calls_if_authorized": 19,
        "maximum_canary_attempts_if_authorized": 8,
        "external_upload_authorized": False,
        "ready_for_shadow_execution": False,
        "images_copied": False, "image_paths_included": False,
        "candidate_only": True, "human_truth": False,
        "approved_for_training": False, "test24_used": False,
        "samples": samples,
    }


def test_builds_queue_bound_unauthorized_template_without_secrets():
    encoded = json.dumps(_queue()).encode()
    result = build_authorization_template(encoded)
    assert result["schema_version"] == "labelagent.external-upload-authorization.v1"
    assert result["queue_sha256"] == hashlib.sha256(encoded).hexdigest()
    assert result["authorized"] is False
    assert result["authorized_by"] is None
    assert result["authorized_upload_count"] == 0
    assert result["authorized_sample_count"] == 19
    assert result["first_stage_max_new_attempts"] == 8
    assert result["automatic_retry_enabled"] is False
    assert not any("key" in key.casefold() or "secret" in key.casefold()
                   for key in result)


def test_rejects_queue_that_is_authorized_or_path_bearing():
    queue = _queue()
    queue["external_upload_authorized"] = True
    with pytest.raises(ValueError, match="queue identity or boundary"):
        build_authorization_template(json.dumps(queue).encode())

    queue = _queue()
    queue["samples"][0]["image_path"] = "private.jpg"
    with pytest.raises(ValueError, match="sample boundary"):
        build_authorization_template(json.dumps(queue).encode())


def test_rejects_canary_or_count_drift():
    queue = _queue()
    queue["canary_count"] = 7
    with pytest.raises(ValueError, match="queue identity or boundary"):
        build_authorization_template(json.dumps(queue).encode())

    queue = _queue()
    queue["sample_count"] = 18
    with pytest.raises(ValueError, match="queue identity or boundary"):
        build_authorization_template(json.dumps(queue).encode())
