import copy

import pytest

from tools.build_mixed_hard_adjudication_workspace import LABELS
from tools.evaluate_expert_resolution_ab import evaluate_resolution_ab


def _prediction(*, slope=False):
    return {
        "assessments": {label: ("present" if label == "slope" and slope
                                else "absent") for label in LABELS},
        "overall_confidence": 0.9, "pallet_instance_count": 0,
        "pallet_quality": "no_pallet",
        "slope_geometry": "full_slope" if slope else "no_visible_slope",
        "class_conflicts": ["none"], "difficulty_flags": ["none"],
        "unknown_candidates": [], "review_note": "fixture",
    }


def _fixture():
    ids = [f"sample-{index:02d}" for index in range(17)]
    hashes = {sample_id: f"{index + 1:064x}"
              for index, sample_id in enumerate(ids)}
    manifest = {"rows": [{"sample_id": sample_id,
                           "source_image_sha256": hashes[sample_id]}
                          for sample_id in ids]}
    adjudication = {
        "schema_version": "labelagent.mixed-hard-adjudication.v2",
        "review_manifest_sha256": "f" * 64,
        "reviewer_declared_id": "declared-reviewer",
        "candidate_only": True, "human_truth": False,
        "approved_for_training": False, "training_authorized": False,
        "automatic_approval": False,
        "rows": [{
            "sample_id": sample_id,
            "source_image_sha256": hashes[sample_id],
            "candidate_decision": "accept_candidate",
            "slope_presence": "present" if index == 0 else "absent",
            "class_conflict": "none", "pallet_quality": "not_applicable",
            "class_presence": {label: (
                "present" if label == "slope" and index == 0 else "absent")
                for label in LABELS},
            "pallet_instance_count": 0,
        } for index, sample_id in enumerate(ids)],
    }
    queue = {
        "schema_version": "labelagent.mixed-hard-expert-shadow-queue.v1",
        "sample_count": 17, "candidate_only": True, "human_truth": False,
        "approved_for_training": False, "test24_used": False,
        "samples": [{"sample_id": sample_id} for sample_id in ids],
    }
    def packet(name, mode, *, resized=False):
        rows = []
        for index, sample_id in enumerate(ids):
            upload = {"name": name, "mode": mode,
                      "upload_image_sha256": ("b" * 64 if resized
                                               else hashes[sample_id])}
            if resized:
                upload.update(width=512, height=410, jpeg_quality=90,
                              source_width=640, source_height=512)
            rows.append({
                "sample_id": sample_id,
                "source_image_sha256": hashes[sample_id],
                "status": "passed", "latency_ms": 5 if resized else 20,
                "expert_result": _prediction(slope=(index == 0 and not resized)),
                "expert_metadata": {"external_vlm": {"usage": {
                    "prompt_tokens": 100 if resized else 1000,
                    "completion_tokens": 10, "total_tokens": 110 if resized else 1010}}},
                "upload_image": upload, "candidate_only": True,
                "human_truth": False, "approved_for_training": False,
                "test24_used": False,
            })
        return {
            "schema_version": "labelagent.mixed-hard-external-shadow.v1",
            "status": "completed", "samples_passed": 17,
            "samples_failed": 0, "manifest_sha256": "f" * 64,
            "queue_sha256": "q" * 64,
            "authorization_sha256": "a" * 64,
            "authorization_schema": "labelagent.external-upload-authorization.v2",
            "authorized_upload_count": 34,
            "image_variant": {"name": name, "mode": mode},
            "candidate_only": True, "human_truth": False,
            "approved_for_training": False, "test24_used": False,
            "rows": rows,
        }
    return (manifest, adjudication, queue,
            packet("source_640x512", "source"),
            packet("resize_512x410", "resize_lanczos_jpeg", resized=True))


def test_evaluates_bound_resolution_ab_without_deployment_claim():
    manifest, adjudication, queue, source, resized = _fixture()
    result = evaluate_resolution_ab(
        manifest, "f" * 64, adjudication, queue, "q" * 64,
        source, resized, input_cost_per_million_tokens=2,
        output_cost_per_million_tokens=8)
    assert result["sample_count"] == 17
    assert result["source_640x512"]["slope_positive_recall"] == 1
    assert result["resize_512x410"]["slope_positive_recall"] == 0
    assert result["delta_resize_minus_source"]["slope_positive_recall"] == -1
    assert result["resize_512x410"]["external_usage"]["total_tokens"] == 1870
    assert result["source_640x512"]["external_usage"]["total_tokens"] == 17170
    assert result["deployment_gate_passed"] is False
    assert result["human_truth"] is False


def test_rejects_variant_authorization_and_upload_drift():
    manifest, adjudication, queue, source, resized = _fixture()
    wrong_auth = copy.deepcopy(resized)
    wrong_auth["authorization_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="different authorizations"):
        evaluate_resolution_ab(
            manifest, "f" * 64, adjudication, queue, "q" * 64,
            source, wrong_auth)
    wrong_upload = copy.deepcopy(resized)
    wrong_upload["rows"][0]["upload_image"]["width"] = 640
    with pytest.raises(ValueError, match="upload image attestations"):
        evaluate_resolution_ab(
            manifest, "f" * 64, adjudication, queue, "q" * 64,
            source, wrong_upload)


def test_rejects_invalid_cost_rates():
    manifest, adjudication, queue, source, resized = _fixture()
    with pytest.raises(ValueError, match="provided together"):
        evaluate_resolution_ab(
            manifest, "f" * 64, adjudication, queue, "q" * 64,
            source, resized, input_cost_per_million_tokens=2)
    with pytest.raises(ValueError, match="finite non-negative"):
        evaluate_resolution_ab(
            manifest, "f" * 64, adjudication, queue, "q" * 64,
            source, resized, estimated_cost_per_call=-1)

