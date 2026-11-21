import io
import json
from unittest.mock import patch

from PIL import Image

from source.agent import server, workflow
from source.agent.workflow import (ego_bottom_geometry_profile, ego_bottom_point_candidates,
                                   ego_bottom_points)


def record(bbox, area):
    return {"bbox": bbox, "area": area}


def test_guard_accepts_wide_bottom_chassis_geometry():
    item = record([0, 399, 523, 512], 41_214)
    assert ego_bottom_geometry_profile(
        item["bbox"], item["area"], 0.875, 640, 512
    ) == "wide_bottom_chassis"


def test_guard_accepts_centered_fork_carriage_geometry():
    item = record([583, 947, 1435, 1536], 365_700)
    assert ego_bottom_geometry_profile(
        item["bbox"], item["area"], 0.92, 1920, 1536
    ) == "centered_fork_carriage"


def test_guard_rejects_floor_like_large_mask():
    item = record([0, 200, 640, 512], 180_000)
    assert ego_bottom_geometry_profile(
        item["bbox"], item["area"], 0.99, 640, 512
    ) is None


def test_guard_rejects_tall_side_occlusion():
    item = record([1350, 700, 1920, 1536], 250_000)
    assert ego_bottom_geometry_profile(
        item["bbox"], item["area"], 0.95, 1920, 1536
    ) is None


def test_guard_rejects_small_bottom_fragment():
    item = record([250, 470, 330, 512], 2_000)
    assert ego_bottom_geometry_profile(
        item["bbox"], item["area"], 0.99, 640, 512
    ) is None


def test_fixed_bottom_points_are_deterministic_unique_and_in_bounds():
    first = ego_bottom_points(640, 512)
    second = ego_bottom_points(640, 512)

    assert first == second
    assert len(first) == 18
    assert len({(point["x"], point["y"]) for point in first}) == 18
    assert all(0 <= point["x"] < 640 and 0 <= point["y"] < 512
               for point in first)


def image_bytes():
    output = io.BytesIO()
    Image.new("RGB", (640, 512), "gray").save(output, "PNG")
    return output.getvalue()


def wide_bottom_result(score=0.9):
    return {"score": score, "parts": [{
        "points": [[0, 399], [522, 399], [522, 511], [0, 511]],
        "holes": [],
    }]}


def test_point_candidate_requires_complete_single_encoding_batch():
    points = ego_bottom_points(640, 512)
    response = {
        "items": [{"point": point, "result": wide_bottom_result(0.8 + index / 1000)}
                  for index, point in enumerate(points)],
        "diagnostics": {"image_encoded_once": True, "point_count": len(points)},
    }
    with patch.object(workflow, "sam_request", return_value=response) as request:
        candidates, evidence = ego_bottom_point_candidates(
            {"sam3_point_url": "http://point-sam"}, image_bytes(), next_id=7)

    assert request.call_args.args[1] == "/segment_points_batch"
    assert request.call_args.kwargs["points"] == points
    assert len(candidates) == 1
    assert candidates[0]["id"] == 7
    assert candidates[0]["label"] == "ego"
    assert candidates[0]["ego_fallback_profile"] == "wide_bottom_chassis"
    assert evidence["point_count"] == 18
    assert evidence["guarded_candidate_count"] == 18


def test_point_candidate_rejects_incomplete_batch_response():
    points = ego_bottom_points(640, 512)
    response = {
        "items": [{"point": point, "result": wide_bottom_result()} for point in points[:-1]],
        "diagnostics": {"image_encoded_once": True, "point_count": len(points)},
    }
    with patch.object(workflow, "sam_request", return_value=response):
        try:
            ego_bottom_point_candidates({}, image_bytes())
        except ValueError as error:
            assert "完整点覆盖" in str(error)
        else:
            raise AssertionError("incomplete batch response was accepted")


class JsonResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def test_point_batch_uses_separate_point_service_route():
    with patch.object(workflow.urllib.request, "urlopen",
                      return_value=JsonResponse(json.dumps({}).encode())) as opened:
        workflow._sam_request({
            "sam3_url": "http://text-sam",
            "sam3_point_url": "http://point-sam",
        }, "/segment_points_batch", b"image", points=[])
    assert opened.call_args.args[0].full_url == "http://point-sam/segment_points_batch"


def test_sam_request_supports_a_bounded_shadow_timeout():
    with patch.object(workflow.urllib.request, "urlopen",
                      return_value=JsonResponse(json.dumps({}).encode())) as opened:
        workflow._sam_request(
            {"sam3_url": "http://sam"}, "/segment_points", b"image",
            timeout_seconds=5.0, points=[])
    assert opened.call_args.kwargs["timeout"] == 5.0
    for invalid in (True, 0, float("nan"), 1201):
        try:
            workflow._sam_request(
                {"sam3_url": "http://sam"}, "/segment_points", b"image",
                timeout_seconds=invalid, points=[])
        except ValueError as error:
            assert "超时设置无效" in str(error)
        else:
            raise AssertionError(f"invalid timeout was accepted: {invalid!r}")


def discovery(confidence=0.95):
    return {"visible_labels": ["ego"], "assessments": [{
        "label": "ego", "evidence": "present", "confidence": confidence,
        "reason": "Visible host chassis pixels",
    }]}


def test_reference_recovery_requires_high_confidence_missing_ego():
    result = {"shapes": [{"id": 3, "label": "pallet"}],
              "candidates_added": 1, "review_queue": [], "agent": {}}
    recovered = {"id": 4, "label": "ego"}
    with patch.object(server, "ego_bottom_point_candidates",
                      return_value=([recovered], {"selected_profile": "wide_bottom_chassis"})) as call:
        status = server.recover_reference_ego_candidate(
            {"ego_bottom_point_fallback": True}, b"image", discovery(), result)

    call.assert_called_once_with(
        {"ego_bottom_point_fallback": True}, b"image", next_id=4)
    assert status["status"] == "candidate_generated"
    assert result["shapes"][-1] == recovered
    assert result["candidates_added"] == 2
    assert result["review_queue"][-1]["reason_code"] == "ego_point_fallback_candidate"


def test_reference_recovery_does_not_run_for_weak_or_existing_ego():
    weak = {"shapes": [], "review_queue": [], "agent": {}}
    existing = {"shapes": [{"id": 1, "label": "ego"}],
                "review_queue": [], "agent": {}}
    with patch.object(server, "ego_bottom_point_candidates") as call:
        weak_status = server.recover_reference_ego_candidate(
            {"ego_bottom_point_fallback": True}, b"image", discovery(0.89), weak)
        existing_status = server.recover_reference_ego_candidate(
            {"ego_bottom_point_fallback": True}, b"image", discovery(), existing)
    call.assert_not_called()
    assert weak_status["status"] == "discovery_not_eligible"
    assert existing_status["status"] == "existing_candidate"


def test_reference_recovery_fails_closed_to_review_queue():
    result = {"shapes": [], "review_queue": [], "agent": {}}
    with patch.object(server, "ego_bottom_point_candidates",
                      side_effect=ValueError("batch unavailable")):
        status = server.recover_reference_ego_candidate(
            {"ego_bottom_point_fallback": True}, b"image", discovery(), result)
    assert status["status"] == "failed"
    assert result["shapes"] == []
    assert result["review_queue"][-1]["reason_code"] == "ego_point_fallback_failed"


def test_post_recovery_guard_routes_slope_inside_strong_ego_to_review():
    result = {
        "shapes": [
            {"id": 1, "label": "slope", "score": 0.66,
             "points": [[250, 400], [390, 400], [390, 470], [250, 470]],
             "holes": []},
            {"id": 2, "label": "ego", "score": 0.96,
             "points": [[120, 330], [520, 330], [520, 511], [120, 511]],
             "holes": []},
            {"id": 3, "label": "floor", "score": 0.98,
             "points": [[0, 0], [639, 0], [639, 300], [0, 300]],
             "holes": []},
        ],
        "candidates_added": 3,
        "review_queue": [],
        "agent": {"candidate_filters": []},
    }

    status = server.filter_post_recovery_slope_ego_conflicts(
        image_bytes(), result)

    assert status == {
        "status": "human_review_required",
        "suppressed_count": 1,
        "reason_code": "slope_candidate_current_ego_conflict",
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
    }
    assert [shape["label"] for shape in result["shapes"]] == ["ego", "floor"]
    assert result["candidates_added"] == 2
    assert result["agent"]["candidate_filters"][0]["id"] == 1
    assert result["review_queue"][0]["reason_code"] == (
        "slope_candidate_current_ego_conflict")
    assert result["review_queue"][0]["human_truth"] is False
    assert result["review_queue"][0]["approved_for_training"] is False


def test_reference_recovery_candidate_id_follows_final_merge_mapping():
    result = {
        "shapes": [{"id": 4, "label": "ego"}],
        "review_queue": [{
            "source": "ego_bottom_point_fallback",
            "candidate_id": 4,
        }],
        "applied": [],
        "agent": {
            "ego_bottom_point_fallback": {
                "status": "candidate_generated",
                "candidate_id": 4,
            },
        },
    }

    server.bind_merged_candidate_references(result, {4: [9]})

    assert result["review_queue"][0]["candidate_id"] == 9
    assert result["review_queue"][0]["original_candidate_id"] == 4
    fallback = result["agent"]["ego_bottom_point_fallback"]
    assert fallback["candidate_id"] == 9
    assert fallback["original_candidate_id"] == 4
