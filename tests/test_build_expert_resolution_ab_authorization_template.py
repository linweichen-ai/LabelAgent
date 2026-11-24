import json

import pytest

from tools.build_expert_resolution_ab_authorization_template import build_template


def _queue(tmp_path, **updates):
    payload = {
        "schema_version": "labelagent.mixed-hard-expert-shadow-queue.v1",
        "sample_count": 2, "candidate_only": True, "human_truth": False,
        "approved_for_training": False, "test24_used": False,
        "images_copied": False, "image_paths_included": False,
        "samples": [{"sample_id": "a"}, {"sample_id": "b"}],
    }
    payload.update(updates)
    path = tmp_path / "queue.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_builds_non_authorizing_exact_ab_scope(tmp_path):
    result = build_template(_queue(tmp_path))
    assert result["authorized"] is False
    assert result["authorized_by"] == ""
    assert result["authorized_sample_count"] == 2
    assert result["authorized_upload_count"] == 4
    assert result["max_external_calls"] == 4
    assert [item["name"] for item in result["authorized_image_variants"]] == [
        "source_640x512", "resize_512x410"]
    assert result["candidate_only"] is True
    assert result["human_truth"] is False
    assert result["test24_used"] is False


def test_rejects_unsafe_or_empty_queue(tmp_path):
    with pytest.raises(ValueError, match="safety boundary"):
        build_template(_queue(tmp_path, test24_used=True))
    with pytest.raises(ValueError, match="safety boundary"):
        build_template(_queue(tmp_path, sample_count=0, samples=[]))
