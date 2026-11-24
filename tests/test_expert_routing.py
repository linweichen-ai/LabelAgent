import copy

from source.agent.expert_routing import evaluate_expert_shadow_route


def _shape(label, points):
    return {"id": 1, "label": label, "parts": [{"points": points, "holes": []}]}


def test_router_records_all_requested_signals_without_dispatching():
    decision = evaluate_expert_shadow_route(
        {
            "llm_expert_shadow_routing_enabled": True,
            "llm_expert_shadow_score_threshold": 3,
            "llm_expert_shadow_low_confidence_percent": 70,
            "llm_expert_shadow_round_threshold": 2,
            "llm_expert_shadow_edge_margin_percent": 5,
            "llm_expert_shadow_small_candidate_percent": 2,
            "llm_expert_shadow_sam_filter_threshold": 2,
        },
        discovery={
            "assessments": {
                "slope": {"evidence": "uncertain", "confidence": 0.62},
            },
            "uncertain_labels": ["slope"],
            "unknown_candidates": [{"candidate_name": "rack"}],
            "registry_conflicts": [{"matching_known_label": "other_robot"}],
        },
        agent={
            "rounds": 2,
            "stop_reason": "no_progress",
            "candidate_filters": [{}, {}],
            "sam_generation": {"mode": "sequential_fallback", "error": "busy"},
        },
        candidates=[_shape("slope", [[0, 470], [40, 470], [40, 500], [0, 500]])],
        image_size=(640, 512),
        candidate_labels=["slope"],
        manual_priority=True,
        local_model_latency_ms=1234.5678,
        sam_result_reused=True,
    )
    assert decision["recommended"] is True
    assert decision["dispatch_performed"] is False
    assert decision["external_upload_performed"] is False
    assert decision["difficulty_score"] >= decision["score_threshold"]
    assert decision["signals"]["edge_candidate_count"] == 1
    assert decision["signals"]["small_candidate_count"] == 1
    assert decision["signals"]["sam_stability"]["unstable"] is True
    assert decision["timing"]["local_model_latency_ms"] == 1234.568
    assert decision["timing"]["external_model_latency_ms"] is None
    assert decision["fallback_performed"] is False
    assert decision["sam_result_reused"] is True
    assert decision["candidate_only"] is True
    assert decision["human_truth"] is False
    assert decision["approved_for_training"] is False


def test_ordinary_floor_and_ego_edges_do_not_trigger_expert():
    decision = evaluate_expert_shadow_route(
        {"llm_expert_shadow_routing_enabled": True},
        discovery={
            "assessments": [
                {"label": "floor", "evidence": "present", "confidence": 0.99},
                {"label": "ego", "evidence": "present", "confidence": 0.99},
            ],
        },
        agent={"rounds": 1, "stop_reason": "done",
               "candidate_filters": [], "sam_generation": {"mode": "multi_prompt"}},
        candidates=[
            _shape("floor", [[0, 100], [639, 100], [639, 511], [0, 511]]),
            _shape("ego", [[0, 450], [639, 450], [639, 511], [0, 511]]),
        ],
        image_size=(640, 512),
        candidate_labels=["floor", "ego"],
        local_model_latency_ms=500,
    )
    assert decision["recommended"] is False
    assert decision["difficulty_score"] == 0
    assert decision["signals"]["edge_candidate_count"] == 0
    assert decision["route"] == "local_result_sufficient"


def test_thresholds_are_configurable_and_invalid_values_fall_back():
    base = {
        "llm_expert_shadow_routing_enabled": True,
        "llm_expert_shadow_score_threshold": 5,
        "llm_expert_shadow_low_confidence_percent": 50,
    }
    discovery = {"assessments": [
        {"label": "pallet", "evidence": "uncertain", "confidence": 0.49},
    ]}
    first = evaluate_expert_shadow_route(base, discovery=discovery)
    assert first["difficulty_score"] == 2
    assert first["recommended"] is False
    lower = evaluate_expert_shadow_route(
        dict(base, llm_expert_shadow_score_threshold=2), discovery=discovery)
    assert lower["recommended"] is True
    invalid = evaluate_expert_shadow_route(
        dict(base, llm_expert_shadow_score_threshold="2"), discovery=discovery)
    assert invalid["score_threshold"] == 3


def test_priority_semantic_uncertainty_is_hard_signal_even_with_high_scalar_confidence():
    decision = evaluate_expert_shadow_route(
        {"llm_expert_shadow_routing_enabled": True,
         "llm_expert_shadow_score_threshold": 99},
        discovery={
            "assessments": {
                "slope": {"evidence": "uncertain", "confidence": 0.95},
            },
            "uncertain_labels": ["slope"],
        })
    assert decision["difficulty_score"] == 1
    assert decision["selection"]["selected_by_score"] is False
    assert decision["selection"]["selected_by_hard_signal"] is True
    assert decision["selection"]["hard_signal_codes"] == [
        "priority_semantic_uncertainty"]
    assert decision["recommended"] is True


def test_router_is_deterministic_and_does_not_copy_sensitive_unknown_evidence():
    config = {"llm_expert_shadow_routing_enabled": True}
    discovery = {"unknown_candidates": [{
        "candidate_name": "secret-name", "evidence": "sensitive pixels"}]}
    first = evaluate_expert_shadow_route(config, discovery=discovery)
    second = evaluate_expert_shadow_route(config, discovery=copy.deepcopy(discovery))
    assert first == second
    serialized = repr(first)
    assert "secret-name" not in serialized
    assert "sensitive pixels" not in serialized


def test_explicit_sam_stability_metrics_are_honored():
    decision = evaluate_expert_shadow_route(
        {
            "llm_expert_shadow_routing_enabled": True,
            "llm_expert_shadow_sam_mask_iou_percent": 80,
            "llm_expert_shadow_sam_area_change_percent": 10,
        },
        agent={
            "rounds": 1,
            "stop_reason": "done",
            "sam_stability": {
                "stable": False,
                "mask_iou": 0.75,
                "area_change_ratio": 0.2,
                "instance_count_delta": 1,
            },
        },
    )
    assert decision["recommended"] is True
    reasons = decision["signals"]["sam_stability"]["reasons"]
    assert {"explicit_unstable", "low_mask_iou", "large_area_change",
            "instance_count_changed"}.issubset(reasons)
