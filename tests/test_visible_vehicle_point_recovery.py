import copy
import io
import unittest
from unittest.mock import patch

from PIL import Image

from source.agent import server, workflow


def image_bytes(width=200, height=120):
    output = io.BytesIO()
    Image.new("RGB", (width, height), "gray").save(output, "PNG")
    return output.getvalue()


def sam_result(point, *, score=0.9):
    return {
        "items": [{
            "point": copy.deepcopy(point),
            "result": {
                "score": score,
                "points": [[40, 20], [80, 20], [80, 50], [40, 50]],
                "holes": [],
                "parts": [{
                    "points": [[40, 20], [80, 20], [80, 50], [40, 50]],
                    "holes": [],
                }],
            },
        }],
        "diagnostics": {"image_encoded_once": True, "point_count": 1},
    }


class VisibleVehiclePointRecoveryTests(unittest.TestCase):
    def test_candidate_uses_vlm_anchor_and_stays_review_only(self):
        located = {
            "decision": "located",
            "entity_type": "conventional_vehicle_or_forklift",
            "x": 500, "y": 300, "evidence": "forklift body visible",
        }
        point = {"x": 100, "y": 36}
        with patch.object(workflow, "chat", return_value=located), \
                patch.object(workflow, "sam_request",
                             return_value=sam_result(point)) as sam:
            candidates, evidence = workflow.visible_vehicle_point_candidate(
                {}, image_bytes(), next_id=4)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["id"], 4)
        self.assertEqual(candidates[0]["label"], "车辆")
        self.assertEqual(candidates[0]["source"],
                         "sam_vlm_visible_vehicle_point_recovery")
        self.assertTrue(candidates[0]["candidate_only"])
        self.assertFalse(candidates[0]["human_truth"])
        self.assertFalse(candidates[0]["approved_for_training"])
        self.assertEqual(evidence["status"], "candidate_generated")
        self.assertEqual(sam.call_args.args[1], "/segment_points_batch")
        self.assertEqual(sam.call_args.kwargs["points"], [point])

    def test_candidate_rejects_foreground_ego_anchor_before_sam(self):
        located = {
            "decision": "located",
            "entity_type": "conventional_vehicle_or_forklift",
            "x": 500, "y": 700, "evidence": "foreground vehicle",
        }
        with patch.object(workflow, "chat", return_value=located), \
                patch.object(workflow, "sam_request") as sam:
            candidates, evidence = workflow.visible_vehicle_point_candidate(
                {}, image_bytes())
        self.assertEqual(candidates, [])
        self.assertEqual(evidence["status"], "foreground_ego_point_rejected")
        sam.assert_not_called()

    def test_compact_guarded_low_score_candidate_still_routes_to_blind_audit(self):
        located = {
            "decision": "located",
            "entity_type": "conventional_vehicle_or_forklift",
            "x": 500, "y": 300, "evidence": "distant forklift body visible",
        }
        point = {"x": 100, "y": 36}
        with patch.object(workflow, "chat", return_value=located), \
                patch.object(workflow, "sam_request",
                             return_value=sam_result(point, score=0.60)):
            candidates, evidence = workflow.visible_vehicle_point_candidate(
                {}, image_bytes(), next_id=7)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(evidence["confidence_band"], "guarded_low")
        self.assertEqual(candidates[0]["recovery_confidence_band"],
                         "guarded_low")
        self.assertTrue(server.candidate_requires_class_audit(candidates[0]))

    def test_server_recovery_requires_high_confidence_visible_miss(self):
        discovery = {"assessments": [{
            "label": "车辆", "evidence": "present", "confidence": 0.95,
            "reason": "forklift visible",
        }]}
        result = {"shapes": [], "review_queue": [], "candidates_added": 0}
        candidate = {"id": 1, "label": "车辆", "candidate_only": True,
                     "human_truth": False, "approved_for_training": False}
        evidence = {"status": "candidate_generated"}
        with patch.object(server, "visible_vehicle_point_candidate",
                          return_value=([candidate], evidence)) as recover:
            status = server.recover_reference_vehicle_candidate(
                {"visible_vehicle_point_recovery": True}, b"image",
                discovery, result)
        self.assertEqual(status["status"], "candidate_generated")
        self.assertEqual(status["trigger_reason"], "visible_high_confidence")
        self.assertEqual(result["shapes"], [candidate])
        self.assertEqual(result["candidates_added"], 1)
        self.assertEqual(result["review_queue"][0]["route"],
                         "human_review_required")
        recover.assert_called_once_with(
            {"visible_vehicle_point_recovery": True}, b"image", next_id=1)

    def test_server_recovery_is_disabled_by_default(self):
        result = {"shapes": [], "review_queue": []}
        with patch.object(server, "visible_vehicle_point_candidate") as recover:
            status = server.recover_reference_vehicle_candidate(
                {}, b"image", {"assessments": []}, result)
        self.assertEqual(status, {"enabled": False, "triggered": False,
                                  "status": "disabled"})
        recover.assert_not_called()

    def test_server_recovery_uses_explicit_coverage_probe_after_discovery_miss(self):
        discovery = {
            "assessments": [{
                "label": "车辆", "evidence": "absent", "confidence": 0.99,
                "reason": "no visible vehicle",
            }],
            "recall_probe_labels": ["车辆"],
        }
        result = {"shapes": [], "review_queue": [], "candidates_added": 0}
        candidate = {"id": 1, "label": "车辆", "candidate_only": True,
                     "human_truth": False, "approved_for_training": False}
        with patch.object(server, "visible_vehicle_point_candidate",
                          return_value=([candidate], {
                              "status": "candidate_generated"})) as recover:
            status = server.recover_reference_vehicle_candidate(
                {"visible_vehicle_point_recovery": True}, b"image",
                discovery, result)
        self.assertEqual(status["status"], "candidate_generated")
        self.assertEqual(status["trigger_reason"], "coverage_probe")
        self.assertEqual(status["discovery_confidence"], 0.99)
        recover.assert_called_once()


if __name__ == "__main__":
    unittest.main()
