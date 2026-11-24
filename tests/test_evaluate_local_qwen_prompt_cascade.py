import hashlib
import json

import pytest

from tools.evaluate_local_qwen_prompt_cascade import evaluate_prompt_cascade
from tools.run_local_qwen_mixed_hard_suggestions import (
    CONTAINER_TRANSITION_V3_INSTRUCTION,
    LABELS,
    sha256_bytes,
)


def _fixture(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest_rows, receipt_files = [], []
    for index in range(34):
        image = tmp_path / f"image-{index:02d}.jpg"
        image.write_bytes(f"image-{index}".encode())
        image_sha = hashlib.sha256(image.read_bytes()).hexdigest()
        manifest_rows.append({
            "sample_id": f"sample-{index:02d}", "cohort": "hard",
            "source_image_sha256": image_sha,
            "raw_image": image.name, "review_image": image.name,
            "candidate_only": True, "human_truth": False,
            "approved_for_training": False,
        })
        receipt_files.append({"path": image.name, "sha256": image_sha})
    manifest = {
        "schema_version": "labelagent.mixed-hard-review-pack.v1",
        "rows": manifest_rows, "summary": {"test24_intersection_count": 0},
        "candidate_only": True, "human_truth": False,
        "approved_for_training": False, "training_authorized": False,
        "test24_used": False, "production_deployed": False,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (tmp_path / "receipt.json").write_text(json.dumps({
        "manifest_sha256": manifest_sha, "files": receipt_files,
    }), encoding="utf-8")

    def packet(*, focused):
        rows = []
        for index in range(34):
            route = index < 2
            slope = ("present" if focused and index == 2 else
                     "uncertain" if focused and index in {3, 4} else "absent")
            rows.append({
                "sample_id": f"sample-{index:02d}",
                "cohort": ("slope_negative_candidate" if index == 0 else "hard"),
                "status": "passed", "latency_ms": 1000,
                "copied_image_sha256": f"{index + 100:064x}",
                "suggestion": {
                    "assessments": {label: (slope if label == "slope" else "absent")
                                    for label in LABELS},
                    "slope_geometry": "no_visible_slope",
                    "review_note": ("inside container" if index in {2, 3}
                                    else "warehouse scene"),
                },
                "expert_shadow_route": {
                    "recommended": route, "dispatch_performed": False,
                    "external_upload_performed": False, "reasons": [],
                },
                "candidate_only": True, "human_truth": False,
                "approved_for_training": False, "test24_used": False,
            })
        payload = {
            "schema_version": "labelagent.local-qwen-mixed-hard-suggestions.v1",
            "status": "completed", "samples_total": 34,
            "samples_passed": 34, "samples_failed": 0,
            "manifest_sha256": manifest_sha, "receipt_sha256": "1" * 64,
            "rows": rows, "candidate_only": True, "human_truth": False,
            "approved_for_training": False, "test24_used": False,
        }
        if focused:
            payload.update({
                "prompt_profile": "mixed-hard-container-transition-v3",
                "instruction_sha256": sha256_bytes(
                    CONTAINER_TRANSITION_V3_INSTRUCTION.encode("utf-8")),
            })
        return payload

    baseline_path, focused_path = tmp_path / "baseline.json", tmp_path / "focused.json"
    baseline_path.write_text(json.dumps(packet(focused=False)), encoding="utf-8")
    focused_path.write_text(json.dumps(packet(focused=True)), encoding="utf-8")
    triage = {
        "schema_version": "labelagent.local-second-assistant-slope-triage.v1",
        "status": "assistant_review_complete_human_confirmation_pending",
        "review_manifest_sha256": manifest_sha,
        "local_qwen_suggestions_sha256": hashlib.sha256(
            baseline_path.read_bytes()).hexdigest(),
        "reviewed_sample_count": 34, "priority_disagreement_count": 3,
        "rows": [{
            "sample_id": f"sample-{index:02d}",
            "source_image_sha256": manifest_rows[index]["source_image_sha256"],
            "local_qwen_slope": "absent", "assistant_slope_candidate": "present",
            "assistant_confidence": "high", "difficulty_flags": ["edge"],
            "rationale": "candidate",
        } for index in (0, 2, 3)],
        "candidate_only": True, "human_truth": False,
        "approved_for_training": False, "training_authorized": False,
        "automatic_approval": False, "test24_used": False,
        "external_api_calls": 0, "factory_data_uploaded": False,
        "production_deployed": False, "human_review_required": True,
    }
    triage_path = tmp_path / "triage.json"
    triage_path.write_text(json.dumps(triage), encoding="utf-8")
    return tmp_path, baseline_path, focused_path, triage_path


def test_evaluates_two_stage_candidate_without_dispatch(tmp_path):
    result = evaluate_prompt_cascade(*_fixture(tmp_path), expert_latency_ms=20000)
    assert result["baseline_external_count"] == 2
    assert result["focused_second_pass_count"] == 32
    assert result["focused_local_slope_resolved_count"] == 1
    assert result["focused_external_added_count"] == 2
    assert result["cascade_external_count"] == 4
    assert result["priority_covered_count"] == 3
    assert result["priority_uncovered_count"] == 0
    assert result["negative_external_count"] == 1
    assert [row["sample_id"] for row in result["external_queue"]] == [
        "sample-00", "sample-01", "sample-03", "sample-04"]
    assert all(set(row) == {
        "sample_id", "cohort", "source_image_sha256", "copied_image_sha256",
        "route_source", "external_upload_authorized", "candidate_only"
    } for row in result["external_queue"])
    assert result["max_external_calls_if_authorized"] == 4
    assert result["external_authorization_granted"] is False
    context = result["semantic_context_cascade"]
    assert context["focused_second_pass_count"] == 2
    assert context["focused_local_slope_resolved_count"] == 1
    assert context["focused_external_added_count"] == 1
    assert context["cascade_external_count"] == 3
    assert context["priority_covered_count"] == 3
    assert context["negative_external_count"] == 1
    assert [row["sample_id"] for row in context["focused_second_pass_queue"]] == [
        "sample-02", "sample-03"]
    assert [row["sample_id"] for row in context["external_queue"]] == [
        "sample-00", "sample-01", "sample-03"]
    assert all("path" not in row for row in context["external_queue"])
    assert all(row["external_upload_authorized"] is False
               for row in context["external_queue"])
    assert context["max_external_calls_if_authorized"] == 3
    assert context["external_authorization_granted"] is False
    assert result["dispatch_performed"] is False
    assert result["external_upload_performed"] is False
    assert result["human_truth"] is False


def test_partial_edge_positive_is_escalated_not_double_counted_as_resolved(tmp_path):
    packet, baseline, focused, triage = _fixture(tmp_path)
    payload = json.loads(focused.read_text())
    payload["rows"][2]["suggestion"]["slope_geometry"] = "partial_edge_slope"
    focused.write_text(json.dumps(payload), encoding="utf-8")

    result = evaluate_prompt_cascade(
        packet, baseline, focused, triage, expert_latency_ms=20000)

    assert result["focused_local_slope_resolved_count"] == 0
    assert result["focused_external_added_count"] == 3
    context = result["semantic_context_cascade"]
    assert context["focused_local_slope_resolved_count"] == 0
    assert context["focused_external_added_count"] == 2
    assert context["priority_covered_count"] == 3


def test_rejects_focused_prompt_or_image_drift(tmp_path):
    packet, baseline, focused, triage = _fixture(tmp_path)
    payload = json.loads(focused.read_text())
    payload["instruction_sha256"] = "0" * 64
    focused.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="another prompt"):
        evaluate_prompt_cascade(packet, baseline, focused, triage)

    packet, baseline, focused, triage = _fixture(tmp_path / "image")
    payload = json.loads(focused.read_text())
    payload["rows"][0]["copied_image_sha256"] = "f" * 64
    focused.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="image binding differs"):
        evaluate_prompt_cascade(packet, baseline, focused, triage)


def test_rejects_unsafe_boundary(tmp_path):
    packet, baseline, focused, triage = _fixture(tmp_path)
    payload = json.loads(focused.read_text())
    payload["approved_for_training"] = True
    focused.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe Qwen packet boundary"):
        evaluate_prompt_cascade(packet, baseline, focused, triage)
