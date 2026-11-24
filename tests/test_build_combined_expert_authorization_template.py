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


def _binding_files(tmp_path):
    manifest = tmp_path / "manifest.json"
    receipt = tmp_path / "receipt.json"
    manifest.write_text('{"rows": []}', encoding="utf-8")
    receipt.write_text('{"files": {}}', encoding="utf-8")
    return manifest, receipt


def _build(tmp_path, all_queue, hybrid_queue):
    manifest, receipt = _binding_files(tmp_path)
    return build_template(
        all_queue, hybrid_queue, manifest, receipt,
        provider="openai_chat",
        url="https://expert.example/v1/chat/completions/",
        model="openai/gpt-6-astra")


def test_combines_all34_and_hybrid17_without_duplicate_source_calls(tmp_path):
    ids = [f"sample-{index:02d}" for index in range(34)]
    result = _build(
        tmp_path,
        _queue(tmp_path, "all.json", ids),
        _queue(tmp_path, "hybrid.json", ids[:17]))
    assert result["schema_version"] == "labelagent.external-upload-authorization.v4"
    assert result["authorized"] is False
    assert result["authorized_unique_source_samples"] == 34
    assert result["authorized_upload_count"] == 51
    assert result["max_external_calls"] == 51
    assert len(result["review_manifest_sha256"]) == 64
    assert len(result["review_receipt_sha256"]) == 64
    assert result["expert_target"] == {
        "provider": "openai_chat",
        "url": "https://expert.example/v1/chat/completions",
        "model": "openai/gpt-6-astra",
    }
    assert [run["sample_count"] for run in result["authorized_runs"]] == [34, 17]
    assert result["reuse_contract"]["duplicate_source_calls_avoided"] == 17
    assert result["candidate_only"] is True
    assert result["test24_used"] is False


def test_rejects_non_subset_hybrid_queue(tmp_path):
    ids = [f"sample-{index:02d}" for index in range(34)]
    hybrid = ids[:16] + ["foreign"]
    with pytest.raises(ValueError, match="not a subset"):
        _build(
            tmp_path,
            _queue(tmp_path, "all.json", ids),
            _queue(tmp_path, "hybrid.json", hybrid))


def test_rejects_secret_bearing_or_non_https_target(tmp_path):
    ids = [f"sample-{index:02d}" for index in range(34)]
    all_queue = _queue(tmp_path, "all.json", ids)
    hybrid_queue = _queue(tmp_path, "hybrid.json", ids[:17])
    manifest, receipt = _binding_files(tmp_path)
    with pytest.raises(ValueError, match="secret-free HTTPS"):
        build_template(
            all_queue, hybrid_queue, manifest, receipt,
            provider="openai_chat",
            url="https://expert.example/v1?api_key=secret", model="model")
    with pytest.raises(ValueError, match="secret-free HTTPS"):
        build_template(
            all_queue, hybrid_queue, manifest, receipt,
            provider="openai_chat", url="http://expert.example/v1",
            model="model")

