import copy
import json
from pathlib import Path
import unittest

from source.agent.closed_loop_contract import validate_event
from source.agent.downstream_mapping import mapping_contract


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "docs/test-results/closed-loop-slope-known-label-candidate-generated-20261111.json"
VLM = ROOT / "docs/test-results/closed-loop-slope-known-label-vlm-reviewed-20261111.json"
SCHEMA = ROOT / "schemas" / "labelagent_closed_loop_event.schema.json"


class ClosedLoopSlopeEventTests(unittest.TestCase):
    def setUp(self):
        self.candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
        self.vlm = json.loads(VLM.read_text(encoding="utf-8"))

    def current_policy_event(self):
        event = copy.deepcopy(self.vlm)
        slope = event["slope_context"]
        slope.pop("geometry_evidence_sufficient")
        slope.pop("geometry_evidence_artifact_ids")
        slope.update(business_mapping="traversable_area",
            downstream_mapping=mapping_contract(),
            traversability_status="passable", policy_default="passable",
            policy_basis="user_defined_project_semantics",
            policy_override={"status": "none", "rule_id": None, "reason": None,
                "decided_by": None, "evidence_artifact_ids": []})
        event["decision"]["reason_codes"] = ["KNOWN_LABEL_SLOPE_SUPPORTED",
            "HUMAN_PIXEL_TRUTH_MISSING", "PROJECT_POLICY_DEFAULT_PASSABLE"]
        event["decision"]["summary"] = ("VLM supports slope_visible; project policy maps "
            "slope to traversable_area while human pixel truth remains pending.")
        event["review"]["reason"] = "Human pixel review remains required for segmentation truth."
        return event

    def test_real_known_label_slope_events_pass_contract(self):
        self.assertEqual(validate_event(self.candidate), [])
        self.assertEqual(validate_event(self.vlm), [])
        self.assertNotIn("unknown_candidate", self.candidate)
        self.assertNotIn("unknown_candidate", self.vlm)

    def test_real_known_label_slope_events_still_validate_against_protocol_schema(self):
        try:
            from jsonschema import Draft202012Validator, FormatChecker
        except ImportError:
            self.skipTest("jsonschema development dependency is absent")
        validator = Draft202012Validator(json.loads(SCHEMA.read_text(encoding="utf-8")),
                                         format_checker=FormatChecker())
        for event in (self.candidate, self.vlm, self.current_policy_event()):
            errors = list(validator.iter_errors(event))
            self.assertEqual(errors, [], [error.message for error in errors])

    def test_legacy_evidence_is_preserved_but_not_used_as_current_policy(self):
        for event in (self.candidate, self.vlm):
            slope = event["slope_context"]
            self.assertFalse(slope["human_pixel_truth_available"])
            self.assertFalse(slope["approved_for_training"])
            self.assertEqual(slope["traversability_status"], "unknown")
            self.assertFalse(slope["geometry_evidence_sufficient"])
            self.assertTrue(event["review"]["required"])
            self.assertEqual(event["review"]["status"], "pending")

    def test_current_policy_maps_slope_to_passable_without_promoting_truth(self):
        event = self.current_policy_event()
        self.assertEqual(validate_event(event), [])
        slope = event["slope_context"]
        self.assertEqual(slope["business_mapping"], "traversable_area")
        self.assertEqual(slope["traversability_status"], "passable")
        self.assertEqual(slope["policy_default"], "passable")
        self.assertEqual(slope["policy_basis"], "user_defined_project_semantics")
        self.assertEqual(slope["policy_override"]["status"], "none")
        self.assertFalse(slope["human_pixel_truth_available"])
        self.assertFalse(slope["approved_for_training"])
        self.assertEqual(event["review"]["status"], "pending")

    def test_vlm_event_extends_candidate_event_without_ontology_mutation(self):
        self.assertEqual(self.vlm["parent_event_id"], self.candidate["event_id"])
        self.assertEqual(self.candidate["run_id"], self.vlm["run_id"])
        self.assertEqual(self.candidate["correlation_id"], self.vlm["correlation_id"])
        self.assertEqual(self.candidate["source"], self.vlm["source"])
        self.assertEqual(self.candidate["producer"]["ontology_version"],
                         self.vlm["producer"]["ontology_version"])

    def test_legacy_real_vlm_evidence_is_visible_support_only(self):
        self.assertIn("KNOWN_LABEL_SLOPE_SUPPORTED",
                      self.vlm["decision"]["reason_codes"])
        self.assertIn("TRAVERSABILITY_UNKNOWN",
                      self.vlm["decision"]["reason_codes"])
        self.assertNotEqual(self.vlm["decision"]["status"], "accepted")
        self.assertEqual(self.vlm["review"]["status"], "pending")
        self.assertNotIn("slope_iou", self.vlm["metrics"])


if __name__ == "__main__":
    unittest.main()
