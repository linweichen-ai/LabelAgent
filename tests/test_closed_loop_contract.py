import copy
import json
from pathlib import Path
import unittest

from source.agent.closed_loop_contract import validate_event


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "closed_loop_unknown_candidate_event.json"


class ClosedLoopContractTests(unittest.TestCase):
    def setUp(self):
        self.event = json.loads(EXAMPLE.read_text(encoding="utf-8"))

    def test_placeholder_unknown_event_passes_but_is_visibly_non_real(self):
        self.assertEqual(validate_event(self.event), [])
        self.assertIn("PLACEHOLDER", self.event["decision"]["summary"])
        self.assertTrue(self.event["source"]["dataset_id"].startswith("EXAMPLE_"))

    def test_unknown_cannot_mutate_ontology_or_skip_review(self):
        event = copy.deepcopy(self.event)
        event["unknown_candidate"]["ontology_mutation_allowed"] = True
        event["review"].update(required=False, status="approved")
        errors = validate_event(event)
        self.assertIn("unknown candidates cannot mutate the production ontology", errors)
        self.assertIn("unknown candidates require review.required=true", errors)
        self.assertIn("new unknown candidates must start with pending review", errors)

    def test_unknown_evidence_must_exist(self):
        event = copy.deepcopy(self.event)
        event["unknown_candidate"]["evidence_artifact_ids"] = ["missing-artifact"]
        self.assertIn("unknown evidence references missing artifacts: missing-artifact",
                      validate_event(event))

    def test_simulation_source_requires_complete_metadata(self):
        event = copy.deepcopy(self.event)
        event["source"].update(origin="simulation", simulation={"simulator": "sim"})
        errors = validate_event(event)
        self.assertIn("source.simulation.scene_config is required", errors)
        self.assertIn("source.simulation.random_seed is required", errors)
        self.assertIn("source.simulation.sensor_version is required", errors)

    def test_unknown_cannot_bypass_ontology_review_state(self):
        event = copy.deepcopy(self.event)
        event["workflow"]["state"] = "auto_accepted"
        errors = validate_event(event)
        self.assertIn("unknown.proposed is incompatible with workflow.state=auto_accepted", errors)

    def test_slope_hard_case_requires_explicit_context(self):
        event = copy.deepcopy(self.event)
        event["task_context"] = {"hard_case_tags": ["slope"]}
        self.assertIn("slope hard cases require slope_context", validate_event(event))

    def test_slope_traversability_requires_geometry_evidence(self):
        event = copy.deepcopy(self.event)
        event["task_context"] = {"hard_case_tags": ["slope"]}
        event["slope_context"] = {
            "semantic_label": "slope_visible", "traversability_status": "traversable",
            "geometry_evidence_sufficient": False,
            "geometry_evidence_artifact_ids": [], "human_pixel_truth_available": False,
            "approved_for_training": False}
        errors = validate_event(event)
        self.assertIn("slope traversability decisions require sufficient geometry evidence", errors)
        self.assertIn("slope traversability decisions require geometry evidence artifacts", errors)

    def test_slope_candidate_can_remain_review_required_without_geometry(self):
        event = copy.deepcopy(self.event)
        event["task_context"] = {"hard_case_tags": ["slope"]}
        event["slope_context"] = {
            "semantic_label": "slope_visible", "traversability_status": "review_required",
            "geometry_evidence_sufficient": False,
            "geometry_evidence_artifact_ids": [], "human_pixel_truth_available": False,
            "approved_for_training": False}
        self.assertEqual(validate_event(event), [])

    def test_machine_slope_candidate_cannot_be_training_approved(self):
        event = copy.deepcopy(self.event)
        event["event_type"] = "candidate.generated"
        event["workflow"].update(previous_state="qc_passed", state="candidates_generated")
        event.pop("unknown_candidate")
        event["task_context"] = {"hard_case_tags": ["slope"]}
        event["slope_context"] = {
            "semantic_label": "slope_visible", "traversability_status": "unknown",
            "geometry_evidence_sufficient": False,
            "geometry_evidence_artifact_ids": [], "human_pixel_truth_available": False,
            "approved_for_training": True}
        errors = validate_event(event)
        self.assertIn("slope cannot be approved for training without human pixel truth", errors)
        self.assertIn("machine slope candidate events cannot be approved for training", errors)

    def test_timestamps_must_include_timezone(self):
        event = copy.deepcopy(self.event)
        event["occurred_at"] = "2026-11-11T10:00:00"
        event["source"]["capture_time"] = "2026-11-11T10:00:00"
        errors = validate_event(event)
        self.assertIn("occurred_at must be an ISO-8601 datetime with timezone", errors)
        self.assertIn("source.capture_time must be an ISO-8601 datetime with timezone", errors)

    def test_current_runtime_truth_commit_requires_timestamped_human_approval(self):
        event = copy.deepcopy(self.event)
        event["event_type"] = "truth.committed"
        event["workflow"].update(previous_state="auto_accepted", state="truth_committed")
        event["decision"]["status"] = "committed"
        event.pop("unknown_candidate")
        event["review"].update(required=False, status="not_required",
                               reviewer_type="policy", reviewed_at=None)
        errors = validate_event(event)
        self.assertIn("current runtime truth commits require previous_state=human_approved", errors)
        self.assertIn("current runtime truth commits require an approved, timestamped human review", errors)

    def test_human_approved_truth_commit_passes(self):
        event = copy.deepcopy(self.event)
        event["event_type"] = "truth.committed"
        event["workflow"].update(previous_state="human_approved", state="truth_committed")
        event["decision"].update(status="committed", confidence=None,
                                 reason_codes=["HUMAN_APPROVED"])
        event.pop("unknown_candidate")
        event["review"].update(required=True, status="approved",
                               reviewer_type="human", reviewed_at="2026-11-11T10:01:00+08:00")
        self.assertEqual(validate_event(event), [])

    def test_boolean_iteration_is_not_an_integer(self):
        event = copy.deepcopy(self.event)
        event["workflow"]["iteration"] = True
        self.assertIn("workflow.iteration must be a non-negative integer", validate_event(event))


if __name__ == "__main__":
    unittest.main()
