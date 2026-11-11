import json
from pathlib import Path
import unittest

from source.agent.closed_loop_contract import validate_event


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "docs/test-results/closed-loop-slope-known-label-candidate-generated-20261111.json"
VLM = ROOT / "docs/test-results/closed-loop-slope-known-label-vlm-reviewed-20261111.json"


class ClosedLoopSlopeEventTests(unittest.TestCase):
    def setUp(self):
        self.candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
        self.vlm = json.loads(VLM.read_text(encoding="utf-8"))

    def test_real_known_label_slope_events_pass_contract(self):
        self.assertEqual(validate_event(self.candidate), [])
        self.assertEqual(validate_event(self.vlm), [])
        self.assertNotIn("unknown_candidate", self.candidate)
        self.assertNotIn("unknown_candidate", self.vlm)

    def test_candidate_truth_and_traversability_remain_separated(self):
        for event in (self.candidate, self.vlm):
            slope = event["slope_context"]
            self.assertFalse(slope["human_pixel_truth_available"])
            self.assertFalse(slope["approved_for_training"])
            self.assertEqual(slope["traversability_status"], "unknown")
            self.assertFalse(slope["geometry_evidence_sufficient"])
            self.assertTrue(event["review"]["required"])
            self.assertEqual(event["review"]["status"], "pending")

    def test_vlm_event_extends_candidate_event_without_ontology_mutation(self):
        self.assertEqual(self.vlm["parent_event_id"], self.candidate["event_id"])
        self.assertEqual(self.candidate["run_id"], self.vlm["run_id"])
        self.assertEqual(self.candidate["correlation_id"], self.vlm["correlation_id"])
        self.assertEqual(self.candidate["source"], self.vlm["source"])
        self.assertEqual(self.candidate["producer"]["ontology_version"],
                         self.vlm["producer"]["ontology_version"])


if __name__ == "__main__":
    unittest.main()
