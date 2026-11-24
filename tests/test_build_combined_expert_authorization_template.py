import json

import pytest

from tools.build_combined_expert_authorization_template import build_template


def _queue(tmp_path, name, ids):
    path = tmp_path / name
    path.write_text(json.dumps({
        "schema_version": "labelagent.mixed-hard-expert-shadow-queue.v1",
        "sample_count": len(ids), "samples": [{"sample_id": item} for item in ids],
        "candidate_only": True, "human_truth": False,
        "approved_for_training": False, "test24_used": False,
        "images_copied": False, "image_paths_included": False,
    }), encoding="utf-8")
    return path


def test_combines_all34_and_hybrid17_without_duplicate_source_calls(tmp_path):
    ids = [f"sample-{index:02d}" for index in range(34)]
    result = build_template(
        _queue(tmp_path, "all.json", ids),
        _queue(tmp_path, "hybrid.json", ids[:17]))
    assert result["authorized"] is False
    assert result["authorized_unique_source_samples"] == 34
    assert result["authorized_upload_count"] == 51
    assert result["max_external_calls"] == 51
    assert [run["sample_count"] for run in result["authorized_runs"]] == [34, 17]
    assert result["reuse_contract"]["duplicate_source_calls_avoided"] == 17
    assert result["candidate_only"] is True
    assert result["test24_used"] is False


def test_rejects_non_subset_hybrid_queue(tmp_path):
    ids = [f"sample-{index:02d}" for index in range(34)]
    hybrid = ids[:16] + ["foreign"]
    with pytest.raises(ValueError, match="not a subset"):
        build_template(
            _queue(tmp_path, "all.json", ids),
            _queue(tmp_path, "hybrid.json", hybrid))

