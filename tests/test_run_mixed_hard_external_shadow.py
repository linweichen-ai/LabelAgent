import hashlib
import json

import pytest

from tools.run_local_qwen_mixed_hard_suggestions import LABELS
from tools.run_mixed_hard_external_shadow import (
    load_authorized_inputs, run_shadow, sha256_file,
)


def _expert_result():
    return {
        "assessments": {label: "absent" for label in LABELS},
        "overall_confidence": 0.9, "pallet_instance_count": 0,
        "pallet_quality": "no_pallet", "slope_geometry": "no_visible_slope",
        "class_conflicts": ["none"], "difficulty_flags": ["none"],
        "unknown_candidates": [], "review_note": "fixture",
    }


def _fixture(tmp_path, authorize=True):
    root = tmp_path / "workspace"; (root / "raw").mkdir(parents=True)
    image = root / "raw" / "sample.jpg"; image.write_bytes(b"image")
    image_sha = hashlib.sha256(b"image").hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "summary": {"test24_intersection_count": 0},
        "rows": [{"sample_id": "sample", "raw_image": "raw/sample.jpg"}],
    }), encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({
        "files": {"raw/sample.jpg": image_sha}, "candidate_only": True,
        "human_truth": False, "approved_for_training": False,
    }), encoding="utf-8")
    queue = tmp_path / "queue.json"
    queue.write_text(json.dumps({
        "schema_version": "labelagent.mixed-hard-expert-shadow-queue.v1",
        "sample_count": 1, "images_copied": False,
        "image_paths_included": False, "candidate_only": True,
        "human_truth": False, "approved_for_training": False,
        "test24_used": False,
        "samples": [{"sample_id": "sample", "cohort": "hard",
                     "source_image_sha256": image_sha}],
    }), encoding="utf-8")
    auth = tmp_path / "authorization.json"
    auth.write_text(json.dumps({
        "schema_version": "labelagent.external-upload-authorization.v1",
        "queue_sha256": sha256_file(queue), "authorized": authorize,
        "authorized_sample_count": 1, "scope": "mixed_hard_external_shadow",
        "authorized_by": "declared-user", "candidate_only": True,
        "human_truth": False, "approved_for_training": False,
        "test24_used": False,
    }), encoding="utf-8")
    return queue, manifest, receipt, root, auth


def test_refuses_execution_without_matching_explicit_authorization(tmp_path):
    queue, manifest, receipt, root, auth = _fixture(tmp_path, authorize=False)
    unauthorized = auth.read_text(encoding="utf-8")
    auth.unlink()
    with pytest.raises(ValueError, match="explicit external-upload authorization"):
        load_authorized_inputs(queue, manifest, receipt, root, auth)
    auth.write_text(unauthorized, encoding="utf-8")
    with pytest.raises(ValueError, match="explicit external-upload authorization"):
        load_authorized_inputs(queue, manifest, receipt, root, auth)
    payload = json.loads(auth.read_text(encoding="utf-8"))
    payload.update(authorized=True, queue_sha256="0" * 64)
    auth.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="explicit external-upload authorization"):
        load_authorized_inputs(queue, manifest, receipt, root, auth)


def test_authorized_fixture_runs_candidate_only_without_truth_write(tmp_path):
    queue, manifest, receipt, root, auth = _fixture(tmp_path)
    samples, bindings = load_authorized_inputs(queue, manifest, receipt, root, auth)
    calls = []
    def respond(*args, **kwargs):
        kwargs["metadata_out"].update({"external_vlm": {
            "usage": {"prompt_tokens": 20, "completion_tokens": 5,
                      "total_tokens": 25}, "usage_available": True}})
        calls.append(args)
        return _expert_result()
    result = run_shadow(
        samples=samples, bindings=bindings, output_path=tmp_path / "result.json",
        config={"llm_expert_model": "fixture"},
        chat_fn=respond)
    assert result["status"] == "completed"
    assert result["samples_passed"] == 1
    assert result["automatic_truth_write"] is False
    assert result["candidate_only"] is True
    assert result["human_truth"] is False
    assert result["approved_for_training"] is False
    assert result["test24_used"] is False
    assert result["rows"][0]["expert_metadata"]["external_vlm"]["usage"][
        "total_tokens"] == 25
    assert len(calls) == 1


def test_rejects_image_hash_drift_before_model_call(tmp_path):
    queue, manifest, receipt, root, auth = _fixture(tmp_path)
    (root / "raw" / "sample.jpg").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_authorized_inputs(queue, manifest, receipt, root, auth)
