from tools.audit_expert_shadow_routes import audit_state


def _shape(label, points):
    return {"id": 1, "label": label, "parts": [{"points": points}]}


def test_audit_is_upload_free_and_skips_test_split():
    state = {
        "config": {"llm_expert_shadow_score_threshold": 3},
        "agent_runs": {
            "1": {
                "dataset_split": "dev",
                "rounds": 1,
                "stop_reason": "done",
                "sam_generation": {"mode": "multi_prompt"},
                "visible_label_discovery": {"unknown_candidates": []},
            },
            "2": {"dataset_split": "test24", "rounds": 9,
                  "stop_reason": "no_progress"},
        },
        "candidates": {
            "1": [_shape("slope", [[0, 450], [50, 450], [50, 500], [0, 500]])],
            "2": [_shape("slope", [[0, 0], [10, 0], [10, 10]])],
        },
    }
    result = audit_state(state, image_size=(640, 512))
    assert result["frames_audited"] == 1
    assert result["test_frames_skipped"] == 1
    assert result["frames_recommended_for_external_expert"] == 1
    assert result["external_upload_performed"] is False
    assert result["dispatch_performed"] is False
    assert result["candidate_only"] is True
    assert result["human_truth"] is False
    assert result["approved_for_training"] is False


def test_audit_keeps_easy_frame_local():
    state = {
        "config": {},
        "agent_runs": {"3": {
            "dataset_split": "dev", "rounds": 1, "stop_reason": "done",
            "candidate_filters": [], "sam_generation": {"mode": "multi_prompt"},
        }},
        "candidates": {"3": [_shape(
            "floor", [[0, 100], [639, 100], [639, 511], [0, 511]])]},
    }
    result = audit_state(state, image_size=(640, 512))
    assert result["frames_local_result_sufficient"] == 1
    assert result["external_call_ratio"] == 0.0
