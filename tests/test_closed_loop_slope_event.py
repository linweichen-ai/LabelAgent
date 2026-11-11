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
        for event in (self.candidate, self.vlm):
            errors = list(validator.iter_errors(event))
            self.assertEqual(errors, [], [error.message for error in errors])

    def test_all_current_evidence_uses_passable_policy_without_geometry_gate(self):
        for event in (self.candidate, self.vlm):
            slope = event["slope_context"]
            self.assertEqual(slope["label_type"], "slope")
            self.assertFalse(slope["human_pixel_truth_available"])
            self.assertFalse(slope["approved_for_training"])
            self.assertEqual(slope["business_mapping"], "traversable_area")
            self.assertEqual(slope["downstream_mapping"], mapping_contract())
            self.assertEqual(slope["traversability_status"], "passable")
            self.assertEqual(slope["policy_default"], "passable")
            self.assertEqual(slope["policy_basis"], "user_defined_project_semantics")
            self.assertNotIn("geometry_evidence_sufficient", slope)
            self.assertNotIn("geometry_evidence_artifact_ids", slope)
            self.assertTrue(event["review"]["required"])
            self.assertEqual(event["review"]["status"], "pending")

    def test_label_type_is_required_and_preserved_for_quality_evaluation(self):
        event = json.loads(json.dumps(self.candidate))
        event["slope_context"].pop("label_type")
        errors = validate_event(event)
        self.assertIn("slope_context.label_type must be slope", errors)

    def test_assistant_selected_point_is_never_recorded_as_human_seed(self):
        producer = self.candidate["producer"]
        self.assertEqual(producer["prompt_provenance"], "assistant_selected_prompt")
        self.assertEqual(producer["prompt_role"], "operator_seed")
        self.assertNotEqual(producer["prompt_provenance"], "human_selected_prompt")
        self.assertNotEqual(producer["prompt_role"], "human_seed")

    def test_point_prompt_without_explicit_provenance_is_rejected(self):
        event = json.loads(json.dumps(self.candidate))
        event["producer"].pop("prompt_provenance")
        event["producer"].pop("prompt_role")
        errors = validate_event(event)
        self.assertIn("point-prompt SAM events require explicit prompt_provenance", errors)
        self.assertIn("point-prompt SAM events require explicit prompt_role", errors)

    def test_old_geometry_unknown_policy_is_rejected(self):
        event = json.loads(json.dumps(self.candidate))
        slope = event["slope_context"]
        slope["traversability_status"] = "unknown"
        slope["geometry_evidence_sufficient"] = False
        slope["geometry_evidence_artifact_ids"] = []
        errors = validate_event(event)
        self.assertIn("slope geometry evidence cannot gate the default passable mapping", errors)
        self.assertIn("default slope policy must map to passable", errors)

    def test_vlm_event_extends_candidate_event_without_ontology_mutation(self):
        self.assertEqual(self.vlm["parent_event_id"], self.candidate["event_id"])
        self.assertEqual(self.candidate["run_id"], self.vlm["run_id"])
        self.assertEqual(self.candidate["correlation_id"], self.vlm["correlation_id"])
        self.assertEqual(self.candidate["source"], self.vlm["source"])
        self.assertEqual(self.candidate["producer"]["ontology_version"],
                         self.vlm["producer"]["ontology_version"])

    def test_real_vlm_evidence_is_visible_support_and_passable_policy_only(self):
        self.assertIn("KNOWN_LABEL_SLOPE_SUPPORTED",
                      self.vlm["decision"]["reason_codes"])
        self.assertIn("PROJECT_POLICY_DEFAULT_PASSABLE",
                      self.vlm["decision"]["reason_codes"])
        self.assertNotEqual(self.vlm["decision"]["status"], "accepted")
        self.assertEqual(self.vlm["review"]["status"], "pending")
        self.assertNotIn("slope_iou", self.vlm["metrics"])


if __name__ == "__main__":
    unittest.main()
