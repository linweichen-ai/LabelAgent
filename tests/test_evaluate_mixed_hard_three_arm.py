import copy
import subprocess
import sys
from pathlib import Path

import pytest

from tools.build_mixed_hard_adjudication_workspace import LABELS
from tools.evaluate_mixed_hard_three_arm import evaluate_three_arm


ROOT = Path(__file__).resolve().parents[1]


def _prediction(present=()):
    return {
        "assessments": {label: ("present" if label in present else "absent")
                        for label in LABELS},
        "overall_confidence": 0.9, "pallet_instance_count": 1 if "pallet" in present else 0,
        "pallet_quality": "clean" if "pallet" in present else "no_pallet",
        "slope_geometry": "full_slope" if "slope" in present else "no_visible_slope",
        "class_conflicts": ["none"], "difficulty_flags": ["none"],
        "unknown_candidates": [], "review_note": "fixture",
    }


def _fixture():
    sample_ids = ("a", "b")
    manifest = {"rows": [{"sample_id": sample, "source_image_sha256": sample * 64}
                         for sample in sample_ids]}
    adjudication = {
        "schema_version": "labelagent.mixed-hard-adjudication.v2",
        "review_manifest_sha256": "f" * 64,
        "reviewer_declared_id": "human-reviewer", "candidate_only": True,
        "human_truth": False, "approved_for_training": False,
        "training_authorized": False, "automatic_approval": False,
        "rows": [
            {"sample_id": "a", "source_image_sha256": "a" * 64,
             "candidate_decision": "accept_candidate", "slope_presence": "present",
             "class_conflict": "none", "pallet_quality": "complete",
             "class_presence": {label: ("present" if label in {"pallet", "slope"} else "absent")
                                for label in LABELS},
             "pallet_instance_count": 1},
            {"sample_id": "b", "source_image_sha256": "b" * 64,
             "candidate_decision": "reject_candidate", "slope_presence": "absent",
             "class_conflict": "none", "pallet_quality": "not_applicable",
             "class_presence": {label: "absent" for label in LABELS},
             "pallet_instance_count": 0},
        ],
    }
    local = {
        "schema_version": "labelagent.local-qwen-mixed-hard-suggestions.v1",
        "status": "completed", "samples_passed": 2, "samples_failed": 0,
        "manifest_sha256": "f" * 64, "candidate_only": True,
        "human_truth": False, "approved_for_training": False,
        "test24_used": False,
        "rows": [
            {"sample_id": "a", "status": "passed", "latency_ms": 5,
             "source_image_sha256": "a" * 64,
             "suggestion": _prediction({"pallet"}), "candidate_only": True,
             "human_truth": False, "approved_for_training": False, "test24_used": False},
            {"sample_id": "b", "status": "passed", "latency_ms": 5,
             "source_image_sha256": "b" * 64,
             "suggestion": _prediction(), "candidate_only": True,
             "human_truth": False, "approved_for_training": False, "test24_used": False},
        ],
    }
    external = {
        "schema_version": "labelagent.mixed-hard-external-shadow.v1",
        "manifest_sha256": "f" * 64,
        "status": "completed", "samples_passed": 2, "samples_failed": 0,
        "candidate_only": True, "human_truth": False,
        "approved_for_training": False, "test24_used": False,
        "rows": [
            {"sample_id": "a", "status": "passed", "latency_ms": 20,
             "source_image_sha256": "a" * 64,
             "expert_metadata": {"external_vlm": {"usage": {
                 "prompt_tokens": 100, "completion_tokens": 10,
                 "total_tokens": 110}}},
             "expert_result": _prediction({"pallet", "slope"}), "candidate_only": True,
             "human_truth": False, "approved_for_training": False, "test24_used": False},
            {"sample_id": "b", "status": "passed", "latency_ms": 20,
             "source_image_sha256": "b" * 64,
             "expert_metadata": {"external_vlm": {"usage": {
                 "prompt_tokens": 200, "completion_tokens": 20,
                 "total_tokens": 220}}},
             "expert_result": _prediction(), "candidate_only": True,
             "human_truth": False, "approved_for_training": False, "test24_used": False},
        ],
    }
    queue = {
        "schema_version": "labelagent.mixed-hard-expert-shadow-queue.v1",
        "candidate_only": True, "human_truth": False,
        "approved_for_training": False, "test24_used": False,
        "sample_count": 1, "source_sample_count": 2,
        "source_suggestions_sha256": "e" * 64,
        "samples": [{"sample_id": "a"}],
    }
    return manifest, adjudication, local, external, queue


def test_scores_three_arms_and_counts_latency_and_calls():
    manifest, adjudication, local, external, queue = _fixture()
    result = evaluate_three_arm(
        manifest, "f" * 64, adjudication, local, external, queue,
        "e" * 64,
        estimated_cost_per_call=0.1,
        input_cost_per_million_tokens=2.0,
        output_cost_per_million_tokens=8.0)
    assert result["arms"]["local_only"]["slope_positive_recall"] == 0
    assert result["arms"]["all_external"]["slope_positive_recall"] == 1
    assert result["arms"]["hybrid"]["slope_positive_recall"] == 1
    assert result["arms"]["local_only"]["external_calls"] == 0
    assert result["arms"]["hybrid"]["external_calls"] == 1
    assert result["arms"]["all_external"]["external_calls"] == 2
    assert result["arms"]["hybrid"]["latency_mean_ms"] == 15
    assert result["arms"]["local_only"]["estimated_cost"] == 0
    assert result["arms"]["hybrid"]["external_usage"]["total_tokens"] == 110
    assert result["arms"]["hybrid"]["estimated_cost"] == 0.00028
    assert result["arms"]["hybrid"]["cost_basis"] == "token_usage"
    assert result["arms"]["all_external"]["external_usage"]["total_tokens"] == 330
    assert result["arms"]["all_external"]["estimated_cost"] == 0.00084
    assert result["human_truth"] is False
    assert result["deployment_gate_passed"] is False


def test_rejects_uncertain_reference_and_prediction_boundary_drift():
    manifest, adjudication, local, external, queue = _fixture()
    uncertain = copy.deepcopy(adjudication)
    uncertain["rows"][0]["class_presence"]["slope"] = "uncertain"
    with pytest.raises(ValueError, match="incomplete or uncertain"):
        evaluate_three_arm(manifest, "f" * 64, uncertain, local, external, queue, "e" * 64)
    unsafe = copy.deepcopy(local); unsafe["human_truth"] = True
    with pytest.raises(ValueError, match="safety boundary"):
        evaluate_three_arm(manifest, "f" * 64, adjudication, unsafe, external, queue, "e" * 64)


def test_rejects_missing_predictions_and_foreign_hybrid_samples():
    manifest, adjudication, local, external, queue = _fixture()
    missing = copy.deepcopy(external)
    missing["rows"].pop(); missing["samples_passed"] = 1
    with pytest.raises(ValueError, match="samples differ"):
        evaluate_three_arm(manifest, "f" * 64, adjudication, local, missing, queue, "e" * 64)
    foreign = copy.deepcopy(queue); foreign["samples"] = [{"sample_id": "foreign"}]
    with pytest.raises(ValueError, match="hybrid queue samples differ"):
        evaluate_three_arm(manifest, "f" * 64, adjudication, local, external, foreign, "e" * 64)


def test_rejects_manifest_image_and_queue_hash_drift():
    manifest, adjudication, local, external, queue = _fixture()
    wrong_manifest = copy.deepcopy(external)
    wrong_manifest["manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="another review manifest"):
        evaluate_three_arm(
            manifest, "f" * 64, adjudication, local, wrong_manifest, queue,
            "e" * 64)
    wrong_image = copy.deepcopy(local)
    wrong_image["rows"][0]["source_image_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="image hashes differ"):
        evaluate_three_arm(
            manifest, "f" * 64, adjudication, wrong_image, external, queue,
            "e" * 64)
    with pytest.raises(ValueError, match="hybrid queue safety boundary"):
        evaluate_three_arm(
            manifest, "f" * 64, adjudication, local, external, queue,
            "0" * 64)


def test_rejects_invalid_assessments_and_latency():
    manifest, adjudication, local, external, queue = _fixture()
    invalid_assessment = copy.deepcopy(local)
    invalid_assessment["rows"][0]["suggestion"]["assessments"]["slope"] = "maybe"
    with pytest.raises(ValueError, match="assessments differ"):
        evaluate_three_arm(
            manifest, "f" * 64, adjudication, invalid_assessment, external,
            queue, "e" * 64)
    invalid_latency = copy.deepcopy(external)
    invalid_latency["rows"][0]["latency_ms"] = float("nan")
    with pytest.raises(ValueError, match="latency is invalid"):
        evaluate_three_arm(
            manifest, "f" * 64, adjudication, local, invalid_latency, queue,
            "e" * 64)


def test_direct_cli_entrypoint_loads_project_modules():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "evaluate_mixed_hard_three_arm.py"),
         "--help"], cwd=ROOT, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert "--hybrid-queue" in completed.stdout


def test_rejects_partial_or_negative_cost_rates():
    manifest, adjudication, local, external, queue = _fixture()
    with pytest.raises(ValueError, match="provided together"):
        evaluate_three_arm(
            manifest, "f" * 64, adjudication, local, external, queue,
            "e" * 64, input_cost_per_million_tokens=2.0)
    with pytest.raises(ValueError, match="finite non-negative"):
        evaluate_three_arm(
            manifest, "f" * 64, adjudication, local, external, queue,
            "e" * 64, estimated_cost_per_call=-1)
