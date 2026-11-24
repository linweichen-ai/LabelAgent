import json

import pytest

from tools.plan_semantic_expert_canary import build_canary_plan


def _fixture(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(10):
        rows.append({
            "sample_id": f"sample-{index}",
            "cohort": ("positive" if index < 3 else
                       "negative" if index < 7 else "conflict"),
            "source_image_sha256": f"{index + 1:064x}",
            "copied_image_sha256": f"{index + 101:064x}",
            "route_source": "baseline_hard_route",
            "external_upload_authorized": False,
            "candidate_only": True,
        })
    payload = {
        "schema_version": "labelagent.local-qwen-prompt-cascade-evaluation.v1",
        "semantic_context_cascade": {
            "external_queue": rows,
            "cascade_external_count": 10,
            "max_external_calls_if_authorized": 10,
            "external_authorization_granted": False,
        },
        "external_upload_performed": False,
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "test24_used": False,
        "production_deployed": False,
    }
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_builds_stratified_path_free_unauthorized_plan(tmp_path):
    result = build_canary_plan(_fixture(tmp_path), canary_count=5)
    assert result["status"] == "authorization_required"
    assert result["canary_count"] == 5
    assert result["remainder_count"] == 5
    assert [row["cohort"] for row in result["canary"][:3]] == [
        "positive", "negative", "conflict"]
    assert result["external_authorization_granted"] is False
    assert result["external_upload_performed"] is False
    assert result["technical_gate"]["maximum_attempts_per_image"] == 1
    assert result["technical_gate"]["automatic_retry_enabled"] is False
    assert all("path" not in row for row in result["canary"])


def test_rejects_authorized_or_path_bearing_queue(tmp_path):
    path = _fixture(tmp_path)
    payload = json.loads(path.read_text())
    payload["semantic_context_cascade"]["external_queue"][0][
        "external_upload_authorized"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="authorized or unsafe"):
        build_canary_plan(path)

    path = _fixture(tmp_path / "path")
    payload = json.loads(path.read_text())
    payload["semantic_context_cascade"]["external_queue"][0][
        "image_path"] = "private/image.jpg"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="path-free"):
        build_canary_plan(path)


def test_rejects_count_drift(tmp_path):
    path = _fixture(tmp_path)
    payload = json.loads(path.read_text())
    payload["semantic_context_cascade"]["cascade_external_count"] = 9
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="count differs"):
        build_canary_plan(path)
