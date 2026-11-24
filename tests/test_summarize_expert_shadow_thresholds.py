import copy

import pytest

from tools.summarize_expert_shadow_thresholds import summarize


def audit():
    return {
        "schema_version": "labelagent.expert-shadow-routing-audit.v2",
        "dispatch_performed": False,
        "external_upload_performed": False,
        "candidate_only": True,
        "frames": [
            {"frame": 1, "difficulty_score": 2,
             "reason_codes": ["edge_partial_candidate"]},
            {"frame": 2, "difficulty_score": 6,
             "reason_codes": ["class_audit_conflict"]},
            {"frame": 3, "difficulty_score": 8,
             "reason_codes": ["sam_candidate_instability"]},
            {"frame": 4, "difficulty_score": 4,
             "reason_codes": ["edge_slope_candidate"]},
        ],
    }


def test_threshold_keeps_hard_signals_without_dispatching():
    result = summarize(audit(), thresholds=[7])
    row = result["thresholds"][0]
    assert row["selected_count"] == 3
    assert row["external_call_ratio"] == 0.75
    assert [item["frame"] for item in row["selected_frames"]] == [2, 3, 4]
    assert row["reason_coverage"]["class_audit_conflict"] == {
        "available": 1, "selected": 1}
    assert result["dispatch_performed"] is False
    assert result["external_upload_performed"] is False
    assert result["human_truth"] is False


def test_rejects_empty_or_non_upload_free_audit():
    empty = audit()
    empty["frames"] = []
    with pytest.raises(ValueError, match="no frames"):
        summarize(empty, thresholds=[7])
    dispatched = copy.deepcopy(audit())
    dispatched["dispatch_performed"] = True
    with pytest.raises(ValueError, match="upload-free"):
        summarize(dispatched, thresholds=[7])


def test_thresholds_are_sorted_unique_and_bounded():
    result = summarize(audit(), thresholds=[9, 3, 9])
    assert [row["threshold"] for row in result["thresholds"]] == [3, 9]
    with pytest.raises(ValueError, match="1 to 100"):
        summarize(audit(), thresholds=[0])
