import copy
from io import BytesIO
import unittest
from unittest import mock

from PIL import Image

from source.agent.ego_shadow import (
    camera_template_key, evaluate_camera_ego_shadow, materialize_camera_template)


def image_bytes(width=40, height=40):
    output = BytesIO()
    Image.new("RGB", (width, height), "white").save(output, "PNG")
    return output.getvalue()


def polygon_shape(label="person", left=20, top=25, right=35, bottom=39):
    return {"id": 7, "label": label, "parts": [{"points": [
        [left, top], [right, top], [right, bottom], [left, bottom],
    ], "holes": []}]}


def discovery(confidence=0.95):
    return {"assessments": [{
        "label": "ego", "evidence": "present", "confidence": confidence,
    }]}


def point_response(left=6, top=32, right=25, bottom=39, score=0.8):
    return {"parts": [{"points": [
        [left, top], [right, top], [right, bottom], [left, bottom],
    ], "holes": []}], "score": score}


class EgoCameraShadowTests(unittest.TestCase):
    def test_materializes_camera_specific_points_inside_image(self):
        name, points = materialize_camera_template("camera0", 11, 21)
        self.assertEqual(name, "camera0-ego-broad")
        self.assertEqual(points[0], {"x": 2, "y": 18, "label": 1})
        self.assertTrue(all(0 <= point["x"] < 11 and 0 <= point["y"] < 21
                            for point in points))
        with self.assertRaisesRegex(ValueError, "dimensions"):
            materialize_camera_template("camera0", 0, 21)
        with self.assertRaisesRegex(ValueError, "no template"):
            materialize_camera_template("camera9", 11, 21)

    def test_maps_formal_capture_topics_without_fuzzy_cross_camera_matching(self):
        self.assertEqual(camera_template_key("/capture/camera_0/image"), "camera0")
        self.assertEqual(camera_template_key("/capture/camera_1/image"), "camera1")
        self.assertEqual(camera_template_key("camera-2"), "camera2")
        self.assertEqual(camera_template_key("camera3"), "camera3")
        self.assertIsNone(camera_template_key("/capture/camera_10/image"))
        self.assertIsNone(camera_template_key("front_camera"))
        self.assertIsNone(camera_template_key("/capture/front_camera_1/image"))
        self.assertIsNone(camera_template_key("plant-camera-2"))

    def test_default_off_never_calls_model_or_mutates_candidates(self):
        shapes = [polygon_shape()]
        before = copy.deepcopy(shapes)
        request = mock.Mock(side_effect=AssertionError("must stay disabled"))
        result = evaluate_camera_ego_shadow(
            {}, image_bytes(), "camera0", discovery(), shapes, request)
        self.assertEqual(result["status"], "disabled")
        self.assertFalse(result["enabled"])
        self.assertFalse(result["state_written"])
        self.assertEqual(shapes, before)
        request.assert_not_called()

    def test_existing_ego_and_low_confidence_fail_closed_without_model(self):
        request = mock.Mock()
        existing = [polygon_shape("ego")]
        self.assertEqual(evaluate_camera_ego_shadow(
            {"ego_camera_shadow_evaluator": True}, image_bytes(), "camera0",
            discovery(), existing, request)["status"], "existing_candidate")
        self.assertEqual(evaluate_camera_ego_shadow(
            {"ego_camera_shadow_evaluator": True}, image_bytes(), "camera0",
            discovery(0.89), [], request)["status"], "discovery_not_eligible")
        request.assert_not_called()

    def test_success_returns_summary_but_never_mergeable_geometry(self):
        shapes = [polygon_shape("person", 1, 1, 6, 6)]
        before = copy.deepcopy(shapes)
        response = point_response()
        result = evaluate_camera_ego_shadow(
            {"ego_camera_shadow_evaluator": True}, image_bytes(), "camera0",
            discovery(), shapes, lambda _points: response)
        self.assertEqual(result["status"], "shadow_observed")
        self.assertTrue(result["candidate_available"])
        self.assertEqual(result["guard_status"], "human_review_required")
        self.assertFalse(result["eligible_for_candidate_merge"])
        self.assertNotIn("parts", result)
        self.assertNotIn("points", result)
        self.assertEqual(len(result["response_sha256"]), 64)
        self.assertEqual(shapes, before)

    def test_camera1_low_coverage_and_person_overlap_are_review_only(self):
        response = point_response(22, 28, 28, 38)
        person = polygon_shape("person", 22, 28, 28, 38)
        result = evaluate_camera_ego_shadow(
            {"ego_camera_shadow_evaluator": True}, image_bytes(),
            "/capture/camera_1/image",
            discovery(), [person], lambda _points: response)
        self.assertEqual(result["coverage_band"], "camera1_low_coverage")
        self.assertEqual(result["camera"], "camera1")
        self.assertEqual(result["camera_topic"], "/capture/camera_1/image")
        self.assertEqual(result["guard_status"], "blocked_person_floor_overlap")
        self.assertGreater(result["person_overlap_ratio"], 0.9)
        self.assertFalse(result["candidate_available"])
        self.assertFalse(result["eligible_for_candidate_merge"])

    def test_overlap_guard_uses_union_of_fragmented_shapes(self):
        response = point_response()
        floor_fragments = [
            polygon_shape("floor", 6, 32, 10, 39),
            polygon_shape("floor", 21, 32, 25, 39),
        ]
        result = evaluate_camera_ego_shadow(
            {"ego_camera_shadow_evaluator": True}, image_bytes(), "camera0",
            discovery(), floor_fragments, lambda _points: response)
        self.assertGreater(result["floor_overlap_ratio"], 0.20)
        self.assertEqual(result["guard_status"], "blocked_person_floor_overlap")
        self.assertFalse(result["candidate_available"])

    def test_sam_warning_keeps_bounded_diagnostics_but_blocks_candidate(self):
        response = point_response(22, 28, 28, 38) | {
            "warnings": ["positive point was not covered"],
        }
        result = evaluate_camera_ego_shadow(
            {"ego_camera_shadow_evaluator": True}, image_bytes(), "camera1",
            discovery(), [], lambda _points: response)
        self.assertEqual(result["status"], "shadow_observed")
        self.assertTrue(result["geometry_observed"])
        self.assertFalse(result["candidate_available"])
        self.assertEqual(result["guard_status"], "blocked_sam_warning")
        self.assertEqual(result["sam_warning_count"], 1)
        self.assertEqual(len(result["sam_warning_sha256"]), 64)
        self.assertNotIn("warnings", result)
        self.assertFalse(result["eligible_for_candidate_merge"])

    def test_prompt_constraint_is_verified_even_without_service_warning(self):
        result = evaluate_camera_ego_shadow(
            {"ego_camera_shadow_evaluator": True}, image_bytes(), "camera0",
            discovery(), [], lambda _points: point_response(20, 25, 35, 39))
        self.assertEqual(result["status"], "shadow_observed")
        self.assertEqual(result["guard_status"], "blocked_prompt_constraint")
        self.assertLess(result["positive_points_covered"],
                        result["positive_point_count"])
        self.assertFalse(result["candidate_available"])

    def test_bad_responses_and_request_failures_become_non_candidates(self):
        cases = (
            None,
            {},
            {"parts": {}, "score": 0.8},
            {"parts": [], "score": 0.8},
            point_response(score=True),
            point_response(score=float("nan")),
            point_response(score=1.1),
            point_response() | {"warnings": "negative point missed"},
            point_response() | {"warnings": [1]},
            {"parts": [{"points": [[1, 1]], "holes": []}], "score": 0.8},
        )
        for payload in cases:
            with self.subTest(payload=payload):
                result = evaluate_camera_ego_shadow(
                    {"ego_camera_shadow_evaluator": True}, image_bytes(),
                    "camera0", discovery(), [], lambda _points: payload)
                self.assertEqual(result["status"], "failed")
                self.assertFalse(result["candidate_available"])
                self.assertFalse(result["eligible_for_candidate_merge"])
        failed = evaluate_camera_ego_shadow(
            {"ego_camera_shadow_evaluator": True}, image_bytes(), "camera0",
            discovery(), [], mock.Mock(side_effect=TimeoutError("slow")))
        self.assertEqual(failed["status"], "failed")
        self.assertIn("slow", failed["error"])

    def test_cancellation_propagates_and_unsupported_camera_does_not_call_model(self):
        class Cancelled(Exception):
            pass

        request = mock.Mock(side_effect=Cancelled("stop"))
        with self.assertRaises(Cancelled):
            evaluate_camera_ego_shadow(
                {"ego_camera_shadow_evaluator": True}, image_bytes(), "camera0",
                discovery(), [], request, cancelled_error_type=Cancelled)
        request.reset_mock()
        result = evaluate_camera_ego_shadow(
            {"ego_camera_shadow_evaluator": True}, image_bytes(), "camera9",
            discovery(), [], request)
        self.assertEqual(result["status"], "unsupported_camera")
        request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
