import copy
import hashlib
import json

import pytest

from tools.audit_mixed_hard_route_coverage import audit_route_coverage


def _write_inputs(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rows, qwen_rows = [], []
    for index in range(34):
        sample_id = f"sample-{index:02d}"
        image = tmp_path / f"image-{index:02d}.jpg"
        image.write_bytes(f"image-{index}".encode())
        image_sha = hashlib.sha256(image.read_bytes()).hexdigest()
        rows.append({
            "sample_id": sample_id,
            "source_image_sha256": image_sha,
            "raw_image": image.name,
            "review_image": image.name,
            "cohort": "hard",
            "review_note": "fixture",
            "candidate_only": True,
            "human_truth": False,
            "approved_for_training": False,
        })
        qwen_rows.append({
            "sample_id": sample_id,
            "copied_image_sha256": f"{index + 100:064x}",
            "suggestion": {"assessments": {
                "slope": "absent" if index < 3 else "present"}},
        })
    manifest = {
        "schema_version": "labelagent.mixed-hard-review-pack.v1",
        "rows": rows, "sample_count": 34,
        "summary": {"test24_intersection_count": 0},
        "candidate_only": True, "human_truth": False,
        "approved_for_training": False, "training_authorized": False,
        "test24_used": False, "production_deployed": False,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    receipt = {
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "files": [{"path": f"image-{index:02d}.jpg",
                   "sha256": rows[index]["source_image_sha256"]}
                  for index in range(34)],
    }
    (tmp_path / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    qwen = {
        "schema_version": "labelagent.local-qwen-mixed-hard-suggestions.v1",
        "status": "completed", "samples_total": 34,
        "samples_passed": 34, "samples_failed": 0,
        "candidate_only": True, "human_truth": False,
        "approved_for_training": False, "test24_used": False,
        "rows": qwen_rows,
    }
    qwen_path = tmp_path / "qwen.json"
    qwen_path.write_text(json.dumps(qwen), encoding="utf-8")
    samples = [{
        "sample_id": f"sample-{index:02d}", "cohort": "hard",
        "source_image_sha256": f"{index + 100:064x}",
        "difficulty_score": 1, "hard_signal_codes": ["uncertain"],
        "local_model_latency_ms": 1000,
        "external_upload_authorized": False,
        "external_dispatch_performed": False,
        "human_reference_available": False,
    } for index in (0, 2)]
    queue = {
        "schema_version": "labelagent.mixed-hard-expert-shadow-queue.v1",
        "source_suggestions_sha256": hashlib.sha256(qwen_path.read_bytes()).hexdigest(),
        "sample_count": 2, "source_sample_count": 34,
        "external_call_ratio": 2 / 34,
        "external_upload_authorized": False,
        "ready_for_shadow_execution": False, "ready_for_quality_ab": False,
        "images_copied": False, "image_paths_included": False,
        "candidate_only": True, "human_truth": False,
        "approved_for_training": False, "test24_used": False,
        "samples": samples,
    }
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(json.dumps(queue), encoding="utf-8")
    triage = {
        "schema_version": "labelagent.local-second-assistant-slope-triage.v1",
        "status": "assistant_review_complete_human_confirmation_pending",
        "review_manifest_sha256": receipt["manifest_sha256"],
        "local_qwen_suggestions_sha256": hashlib.sha256(qwen_path.read_bytes()).hexdigest(),
        "reviewed_sample_count": 34, "priority_disagreement_count": 3,
        "rows": [{
            "sample_id": f"sample-{index:02d}",
            "source_image_sha256": rows[index]["source_image_sha256"],
            "local_qwen_slope": "absent", "assistant_slope_candidate": "present",
            "assistant_confidence": confidence,
            "difficulty_flags": flags, "rationale": "candidate",
        } for index, confidence, flags in (
            (0, "high", ["edge"]),
            (1, "medium", ["occluded"]),
            (2, "low", ["edge"]),
        )],
        "candidate_only": True, "human_truth": False,
        "approved_for_training": False, "training_authorized": False,
        "automatic_approval": False, "test24_used": False,
        "external_api_calls": 0, "factory_data_uploaded": False,
        "production_deployed": False, "human_review_required": True,
    }
    triage_path = tmp_path / "triage.json"
    triage_path.write_text(json.dumps(triage), encoding="utf-8")
    return tmp_path, qwen_path, queue_path, triage_path


def test_reports_priority_gap_without_authorizing_dispatch(tmp_path):
    paths = _write_inputs(tmp_path)
    result = audit_route_coverage(*paths)
    assert result["status"] == "coverage_gap_detected"
    assert result["baseline_queue_count"] == 2
    assert result["priority_routed_count"] == 2
    assert result["priority_missed_count"] == 1
    assert result["high_medium_missed_count"] == 1
    assert result["coverage_by_confidence"]["medium"] == {
        "routed": 0, "missed": 1}
    assert result["supplemental_review_proposal_count"] == 1
    assert result["augmented_shadow_candidate_count"] == 3
    assert result["coverage_gate_passed"] is False
    proposal = result["supplemental_review_proposal"][0]
    assert proposal["sample_id"] == "sample-01"
    assert proposal["external_upload_authorized"] is False
    assert result["dispatch_performed"] is False
    assert result["human_truth"] is False


def test_rejects_queue_hash_drift_and_unsafe_boundaries(tmp_path):
    packet, qwen_path, queue_path, triage_path = _write_inputs(tmp_path)
    queue = json.loads(queue_path.read_text())
    queue["source_suggestions_sha256"] = "0" * 64
    queue_path.write_text(json.dumps(queue), encoding="utf-8")
    with pytest.raises(ValueError, match="another Qwen"):
        audit_route_coverage(packet, qwen_path, queue_path, triage_path)

    packet, qwen_path, queue_path, triage_path = _write_inputs(tmp_path / "unsafe")
    queue = json.loads(queue_path.read_text())
    queue["external_upload_authorized"] = True
    queue_path.write_text(json.dumps(queue), encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe expert queue boundary"):
        audit_route_coverage(packet, qwen_path, queue_path, triage_path)


def test_rejects_per_sample_identity_drift(tmp_path):
    packet, qwen_path, queue_path, triage_path = _write_inputs(tmp_path)
    queue = json.loads(queue_path.read_text())
    queue["samples"][0]["source_image_sha256"] = "f" * 64
    queue_path.write_text(json.dumps(queue), encoding="utf-8")
    with pytest.raises(ValueError, match="image binding differs"):
        audit_route_coverage(packet, qwen_path, queue_path, triage_path)
