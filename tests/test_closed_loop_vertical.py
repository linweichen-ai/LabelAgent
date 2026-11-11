from argparse import Namespace
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

try:
    from jsonschema import Draft202012Validator, FormatChecker, ValidationError
except ImportError:  # Runtime tests do not require the development schema dependency.
    Draft202012Validator = FormatChecker = ValidationError = None

from source.agent.closed_loop import digest
from source.agent.downstream_mapping import mapping_contract
from tools.closed_loop_vertical import process_vertical
from tools.validate_closed_loop_bundle import validate_bundle_file


class ClosedLoopVerticalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.image = b"factory-frame-0"
        self.topic = "/capture/camera_0/image"
        self.shape = {"id": 1, "label": "slope", "points": [[1, 1], [8, 1], [8, 7]],
                      "holes": [], "score": 0.98046875}
        self.state = {"loaded": True, "path": "/private/factory.mcap",
                      "topic": self.topic, "topics": {self.topic: {"count": 1}},
                      "truth": {}, "truths_by_topic": {},
                      "candidates": {"0": [self.shape]},
                      "candidates_by_topic": {}, "reference_frames": {}}
        self.args = Namespace(dataset_id="sfl-cpd15y", source_batch="batch-a-000",
            frame=0, topic=self.topic, source_kind="real",
            ledger_root=str(self.root / "ledger"), output=str(self.root / "bundle.json"),
            sam_model_name="sam3", sam_model_version="sam3-verified-sha",
            sam_prompt_version="slope-visible-point-v1",
            version_id="factory-truth-v1", parent_version=None,
            dataset_split="val", split_group_id="ramp-sequence-000",
            ontology_version="seven-class-v1")

    def run_vertical(self, state, reviews=None):
        return process_vertical(self.args, state=state, image=self.image,
            media_type="image/png", timestamp="2026-09-20T09:45:06Z",
            timestamp_source="capture", reviews=reviews or [])

    def accepted_state_and_review(self):
        accepted = copy.deepcopy(self.state)
        candidate = accepted["candidates"].pop("0")
        accepted["truth"]["0"] = candidate
        accepted["reference_frames"][self.topic] = [0]
        body = {"schema_version": "labelagent.human-review.v2", "sequence": 1,
            "previous_review_id": None, "file": accepted["path"], "topic": self.topic,
            "dataset_id": "sfl-cpd15y", "dataset_version": "factory-truth-v1",
            "camera": self.topic,
            "frame": 0, "frame_sha256": hashlib.sha256(self.image).hexdigest(),
            "expected_frame_sha256": hashlib.sha256(self.image).hexdigest(),
            "decision": "accept_candidate", "reviewer_declared_id": "reviewer-a",
            "identity_assurance": "self_declared",
            "decision_idempotency_key": "vertical-acceptance-1",
            "remediation_of_review_id": None,
            "started_at": "2026-11-11T00:00:00Z",
            "reviewed_at": "2026-11-11T00:00:05Z", "elapsed_seconds": 5.0,
            "note": "slope boundary checked", "candidate_sha256": digest(candidate),
            "expected_candidate_sha256": digest(candidate),
            "candidate_event_id": "b" * 64, "candidate_shape_count": 1,
            "truth_sha256": digest(candidate), "truth_shape_count": 1,
            "confirmed_empty": False,
            "slope_context": {"label_type": "slope", "semantic_label": "slope_visible",
                "visible_status": "human_confirmed",
                "human_pixel_truth_available": True,
                "approved_for_training": False,
                "business_mapping": "traversable_area",
                "downstream_mapping": mapping_contract(),
                "traversability_status": "passable",
                "policy_default": "passable",
                "policy_basis": "user_defined_project_semantics",
                "policy_override": {"status": "none", "rule_id": None,
                    "reason": None, "decided_by": None,
                    "evidence_artifact_ids": []},
                "candidate_sha256": digest(candidate),
                "evidence_provenance": [{"kind": "candidate_hash",
                    "source": "human_review_receipt", "sha256": digest(candidate)}],
                "hard_set_strata": ["unassigned"]},
            "closed_loop_linkage": {"status": "pending_sync", "candidate_event_id": "b" * 64,
                "human_accepted_event_id": None, "truth_committed_event_id": None}}
        return accepted, {**body, "review_id": digest(body)}

    def test_candidate_rerun_and_human_acceptance_are_append_only(self):
        initial, initial_path = self.run_vertical(self.state)
        self.assertEqual([item["state"] for item in initial["events"]],
            ["received", "quality_checked", "sam_candidate_ready", "candidate_final",
             "human_review_required"])
        route = initial["events"][-1]["payload"]["routing"]
        self.assertFalse(route["quality_failure"])
        self.assertTrue(route["missing_human_gate"])
        slope = initial["events"][2]["payload"]["slope_context"]
        self.assertEqual(slope["label_type"], "slope")
        self.assertEqual(slope["semantic_label"], "slope_visible")
        self.assertFalse(slope["human_pixel_truth_available"])
        self.assertEqual(slope["traversability_status"], "passable")
        self.assertEqual(slope["business_mapping"], "traversable_area")
        self.assertEqual(slope["policy_override"]["status"], "none")
        initial_bytes = initial_path.read_bytes()
        event_files = sorted((Path(self.args.ledger_root) / "runs").rglob("events/*.json"))
        event_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in event_files}

        replay, replay_path = self.run_vertical(self.state)
        self.assertEqual(replay, initial)
        self.assertEqual(replay_path, initial_path)
        self.assertEqual(replay_path.read_bytes(), initial_bytes)

        accepted, review = self.accepted_state_and_review()
        final, final_path = self.run_vertical(accepted, [review])
        self.assertEqual([item["state"] for item in final["events"]][-3:],
            ["human_accepted", "truth_committed", "version_committed"])
        acceptance = final["events"][-3]
        frozen = initial["events"][-2]
        self.assertEqual(acceptance["payload"]["accepted_event_id"], frozen["event_id"])
        self.assertEqual(acceptance["payload"]["accepted_artifact_sha256"],
                         frozen["artifacts"][0]["sha256"])
        self.assertTrue(final["receipt"]["training_ready"])
        self.assertNotEqual(final_path, initial_path)
        self.assertEqual(initial_path.read_bytes(), initial_bytes)
        self.assertTrue(final_path.is_file())
        for path, before in event_hashes.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
        manifests = list((Path(self.args.ledger_root) / "runs").rglob(
            "artifacts/manifest-*.json"))
        self.assertEqual(len(manifests), 1)
        manifest_event = final["events"][-1]["artifacts"][0]
        self.assertEqual(hashlib.sha256(manifests[0].read_bytes()).hexdigest(),
                         manifest_event["sha256"])
        replay_final, replay_final_path = self.run_vertical(accepted, [review])
        self.assertEqual(replay_final, final)
        self.assertEqual(replay_final_path, final_path)

    @unittest.skipUnless(Draft202012Validator, "jsonschema development dependency is absent")
    def test_actual_initial_and_final_bundles_validate_against_schema(self):
        schema = json.loads((Path(__file__).resolve().parents[1] /
                             "schemas/labelagent-closed-loop.schema.json").read_text(
                                 encoding="utf-8"))
        self.assertTrue({"registration", "workbench_readback", "bundle_sha256"} <=
                        set(schema["properties"]))
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema, format_checker=FormatChecker())
        initial, initial_path = self.run_vertical(self.state)
        initial_bytes = initial_path.read_bytes()
        initial_from_file = json.loads(initial_bytes)
        self.assertEqual(initial_from_file, initial)
        validator.validate(initial_from_file)
        initial_validation = validate_bundle_file(initial_path,
            Path(__file__).resolve().parents[1] / "schemas/labelagent-closed-loop.schema.json")
        self.assertTrue(initial_validation["valid"], initial_validation["errors"])
        self.assertEqual(initial_path.read_bytes(), initial_bytes)
        accepted, review = self.accepted_state_and_review()
        final, final_path = self.run_vertical(accepted, [review])
        final_bytes = final_path.read_bytes()
        final_from_file = json.loads(final_bytes)
        self.assertEqual(final_from_file, final)
        validator.validate(final_from_file)
        final_validation = validate_bundle_file(final_path,
            Path(__file__).resolve().parents[1] / "schemas/labelagent-closed-loop.schema.json")
        self.assertTrue(final_validation["valid"], final_validation["errors"])
        self.assertEqual(final_path.read_bytes(), final_bytes)
        missing_split = copy.deepcopy(final)
        missing_split["events"][-1]["payload"]["dataset_version"].pop("split")
        with self.assertRaises(ValidationError):
            validator.validate(missing_split)
        missing_manifest = copy.deepcopy(final)
        missing_manifest["events"][-1]["artifacts"][0]["kind"] = "report"
        with self.assertRaises(ValidationError):
            validator.validate(missing_manifest)


if __name__ == "__main__":
    unittest.main()
