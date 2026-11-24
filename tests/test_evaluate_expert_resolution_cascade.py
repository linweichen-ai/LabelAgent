import pytest

from tools.evaluate_expert_resolution_cascade import (
    _usage_for_calls, escalation_reasons,
)
from tools.run_local_qwen_mixed_hard_suggestions import LABELS


def result(**overrides):
    payload = {
        "assessments": {label: "absent" for label in LABELS},
        "overall_confidence": 0.93,
        "pallet_instance_count": 0,
        "pallet_quality": "no_pallet",
        "slope_geometry": "no_visible_slope",
        "class_conflicts": [],
        "difficulty_flags": [],
        "unknown_candidates": [],
        "review_note": "fixture",
    }
    payload.update(overrides)
    return payload


def test_accepts_confident_resized_result_that_agrees_with_local():
    assert escalation_reasons(result(), result()) == []


def test_escalates_low_confidence_uncertainty_and_critical_disagreement():
    local = result(assessments={**result()["assessments"], "slope": "absent"})
    resized = result(
        overall_confidence=0.72,
        assessments={**result()["assessments"], "slope": "uncertain"},
        slope_geometry="partial_edge_slope",
        class_conflicts=["floor_slope"],
        unknown_candidates=["inclined_platform"],
    )
    reasons = escalation_reasons(local, resized)
    assert "resized_low_confidence" in reasons
    assert "resized_uncertain:slope" in reasons
    assert "resized_class_conflict:floor_slope" in reasons
    assert "resized_unknown_candidate" in reasons
    assert "resized_difficult_slope_geometry" in reasons
    assert "local_resized_disagreement:slope" in reasons


def test_escalates_pallet_quality_or_count_disagreement():
    reasons = escalation_reasons(
        result(pallet_instance_count=2),
        result(pallet_instance_count=3, pallet_quality="uncertain"))
    assert reasons == [
        "resized_uncertain_pallet_quality",
        "local_resized_pallet_count_disagreement",
    ]


def test_rejects_invalid_threshold():
    with pytest.raises(ValueError, match="between zero and one"):
        escalation_reasons(result(), result(), min_confidence=1.1)


def test_usage_sums_both_passes_for_escalated_sample():
    def row(prompt, completion):
        return {"expert_metadata": {"external_vlm": {"usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }}}}
    usage = _usage_for_calls([
        ("resize", "a", row(10, 2)),
        ("source", "a", row(20, 3)),
        ("resize", "b", row(11, 2)),
    ])
    assert usage == {
        "samples_with_usage": 3,
        "samples_missing_usage": 0,
        "complete": True,
        "prompt_tokens": 41,
        "completion_tokens": 7,
        "total_tokens": 48,
    }


def test_usage_marks_missing_telemetry_incomplete():
    usage = _usage_for_calls([("resize", "a", {})])
    assert usage["complete"] is False
    assert usage["samples_missing_usage"] == 1
