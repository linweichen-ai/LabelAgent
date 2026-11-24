import copy
import json

import pytest

from tools.build_mixed_hard_expert_queue import build_queue


def _source():
    rows = []
    for index in range(34):
        selected = index < 17
        rows.append({
            "sample_id": f"sample-{index:02d}", "cohort": "hard" if selected else "ordinary",
            "status": "passed", "latency_ms": 4500 + index,
            "copied_image_sha256": f"{index:064x}",
            "candidate_only": True, "human_truth": False,
            "approved_for_training": False, "test24_used": False,
            "expert_shadow_route": {
                "recommended": selected, "difficulty_score": 1 if selected else 0,
                "dispatch_performed": False, "external_upload_performed": False,
                "selection": {
                    "selected_by_score": False,
                    "selected_by_hard_signal": selected,
                    "hard_signal_codes": (["priority_semantic_uncertainty"]
                                          if selected else []),
                },
            },
        })
    return {
        "schema_version": "labelagent.local-qwen-mixed-hard-suggestions.v1",
        "status": "completed", "samples_total": 34, "samples_passed": 34,
        "samples_failed": 0, "candidate_only": True, "human_truth": False,
        "approved_for_training": False, "test24_used": False, "rows": rows,
    }


def test_builds_path_free_non_authorized_queue_with_budget():
    source = _source()
    encoded = json.dumps(source, sort_keys=True).encode()
    result = build_queue(encoded, expert_latency_ms=20000,
                         estimated_cost_per_call=0.01)
    assert result["sample_count"] == 17
    assert result["external_call_ratio"] == 0.5
    assert result["estimated_sequential_expert_latency_ms"] == 340000
    assert result["estimated_total_cost"] == 0.17
    assert result["external_upload_authorized"] is False
    assert result["ready_for_shadow_execution"] is False
    assert result["ready_for_quality_ab"] is False
    assert result["image_paths_included"] is False
    assert result["hard_signal_counts"] == {"priority_semantic_uncertainty": 17}
    assert "raw/" not in json.dumps(result)


def test_rejects_boundary_drift_and_dispatched_source():
    for key, value in (("candidate_only", False), ("human_truth", True),
                       ("approved_for_training", True), ("test24_used", True)):
        source = _source(); source[key] = value
        with pytest.raises(ValueError, match="safety boundary"):
            build_queue(json.dumps(source).encode())
    source = _source()
    source["rows"][0]["expert_shadow_route"]["dispatch_performed"] = True
    with pytest.raises(ValueError, match="no-dispatch"):
        build_queue(json.dumps(source).encode())


def test_rejects_incomplete_or_duplicate_source_without_mutation():
    source = _source(); before = copy.deepcopy(source)
    build_queue(json.dumps(source).encode())
    assert source == before
    incomplete = _source(); incomplete["samples_passed"] = 33
    with pytest.raises(ValueError, match="incomplete"):
        build_queue(json.dumps(incomplete).encode())
    duplicate = _source(); duplicate["rows"][1]["sample_id"] = "sample-00"
    with pytest.raises(ValueError, match="incomplete"):
        build_queue(json.dumps(duplicate).encode())
