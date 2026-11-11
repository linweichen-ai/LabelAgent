import json
from pathlib import Path
import tempfile
import unittest

from source.agent.closed_loop import ClosedLoopLedger, digest
from source.agent.downstream_mapping import mapping_contract
from tools.closed_loop_human_gate import sync_human_gate
from tools.validate_closed_loop_bundle import validate_bundle_file


class HumanGateSyncTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.work = self.root / "work"
        self.work.mkdir()
        self.ledger_root = self.root / "ledger"
        self.envelope = {"dataset_id": "dataset", "source_batch": "batch",
            "frame_id": "0", "sensor": "/camera", "captured_at": "2026-11-11T00:00:00Z",
            "source_kind": "real", "content_sha256": "a" * 64}
        self.ledger = ClosedLoopLedger(self.ledger_root, self.envelope)
        self.ledger.register()
        self.truth = [{"label": "slope", "points": [[1, 1], [2, 1], [2, 2]]}]
        self.truth_sha = digest(self.truth)
        self.ledger.append_event("quality_checked",
            actor={"type": "quality", "id": "qc"}, action="check",
            label_layer="preannotation",
            payload={"quality_checked": {"passed": True, "checks": ["content_hash"]}})
        self.ledger.append_event("sam_candidate_ready",
            actor={"type": "sam", "id": "sam3"}, action="segment",
            label_layer="preannotation", model={"name": "sam3", "version": "test"},
            prompt={"version": "test"})
        self.candidate = self.ledger.append_event("candidate_final",
            actor={"type": "system", "id": "freezer"}, action="freeze",
            label_layer="agent_candidate",
            artifacts=[{"kind": "candidate_final", "sha256": self.truth_sha}])
        self.ledger.append_event("human_review_required",
            actor={"type": "system", "id": "gate"}, action="route",
            label_layer="preannotation")
        self.bundle = self.root / "bundle.json"
        self.bundle.write_text(json.dumps({"schema_version": "labelagent.closed-loop.v1",
            "input": self.envelope, "events": self.ledger.events(),
            "receipt": self.ledger.receipt()}), encoding="utf-8")

    def slope_context(self, visible_status, human_truth):
        return {"semantic_label": "slope_visible", "visible_status": visible_status,
            "human_pixel_truth_available": human_truth, "approved_for_training": False,
            "business_mapping": "traversable_area", "traversability_status": "passable",
            "downstream_mapping": mapping_contract(),
            "policy_default": "passable",
            "policy_basis": "user_defined_project_semantics",
            "policy_override": {"status": "none", "rule_id": None,
                "reason": None, "decided_by": None, "evidence_artifact_ids": []},
            "candidate_sha256": self.truth_sha,
            "evidence_provenance": [{"kind": "candidate_hash",
                "source": "human_review_receipt", "sha256": self.truth_sha}],
            "hard_set_strata": ["unassigned"]}

    def write_acceptance(self):
        (self.work / "state.json").write_text(json.dumps({"topic": "/camera",
            "truth": {"0": self.truth}}, ensure_ascii=False), encoding="utf-8")
        body = {"schema_version": "labelagent.human-review.v2", "sequence": 1,
            "previous_review_id": None, "file": "example.mcap", "topic": "/camera",
            "dataset_id": "dataset", "dataset_version": "dataset-v1", "camera": "/camera",
            "frame": 0, "frame_sha256": "a" * 64,
            "expected_frame_sha256": "a" * 64,
            "decision": "accept_candidate",
            "reviewer_declared_id": "reviewer", "identity_assurance": "self_declared",
            "decision_idempotency_key": "accept-review-1",
            "remediation_of_review_id": None,
            "started_at": "2026-11-11T00:00:00Z",
            "reviewed_at": "2026-11-11T00:00:12Z", "elapsed_seconds": 12.0,
            "note": "checked", "candidate_sha256": self.truth_sha,
            "expected_candidate_sha256": self.truth_sha,
            "candidate_event_id": "b" * 64, "candidate_shape_count": 1,
            "truth_sha256": self.truth_sha, "truth_shape_count": 1,
            "confirmed_empty": False,
            "slope_context": self.slope_context("human_confirmed", True),
            "closed_loop_linkage": {"status": "pending_sync", "candidate_event_id": "b" * 64,
                "human_accepted_event_id": None, "truth_committed_event_id": None}}
        record = {**body, "review_id": digest(body)}
        folder = self.work / "human_reviews"
        folder.mkdir()
        (folder / f"000001-{record['review_id']}.json").write_text(
            json.dumps(record), encoding="utf-8")

    def write_rejection(self):
        body = {"schema_version": "labelagent.human-review.v2", "sequence": 1,
            "previous_review_id": None, "file": "example.mcap", "topic": "/camera",
            "dataset_id": "dataset", "dataset_version": "dataset-v1", "camera": "/camera",
            "frame": 0, "frame_sha256": "a" * 64,
            "expected_frame_sha256": "a" * 64,
            "decision": "reject_candidate",
            "reviewer_declared_id": "reviewer", "identity_assurance": "self_declared",
            "decision_idempotency_key": "reject-review-1",
            "remediation_of_review_id": None,
            "started_at": "2026-11-11T00:00:00Z",
            "reviewed_at": "2026-11-11T00:00:05Z", "elapsed_seconds": 5.0,
            "note": "boundary leakage", "candidate_sha256": self.truth_sha,
            "expected_candidate_sha256": self.truth_sha,
            "candidate_event_id": "b" * 64, "candidate_shape_count": 1,
            "truth_sha256": None, "truth_shape_count": None,
            "confirmed_empty": False,
            "slope_context": self.slope_context("human_rejected", False),
            "closed_loop_linkage": {"status": "not_applicable", "candidate_event_id": "b" * 64,
                "human_accepted_event_id": None, "truth_committed_event_id": None}}
        record = {**body, "review_id": digest(body)}
        folder = self.work / "human_reviews"
        folder.mkdir()
        (folder / f"000001-{record['review_id']}.json").write_text(
            json.dumps(record), encoding="utf-8")

    def test_acceptance_is_verified_and_idempotently_committed(self):
        self.write_acceptance()
        output = self.root / "accepted.json"
        first = sync_human_gate(bundle_path=self.bundle, workbench_root=self.work,
            ledger_root=self.ledger_root, output_path=output)
        second = sync_human_gate(bundle_path=self.bundle, workbench_root=self.work,
            ledger_root=self.ledger_root, output_path=output)
        self.assertEqual(first["receipt"]["last_state"], "truth_committed")
        self.assertTrue(first["receipt"]["human_gate_passed"])
        self.assertFalse(first["receipt"]["minimum_closed_loop_complete"])
        self.assertEqual(len(first["events"]), len(second["events"]))
        self.assertEqual(first["human_review"]["reviewer_declared_id"], "reviewer")
        self.assertEqual(first["human_review"]["slope_context"]["semantic_label"],
                         "slope_visible")
        self.assertEqual(first["human_review"]["slope_context"]["traversability_status"],
                         "passable")
        self.assertEqual(first["human_review"]["slope_context"]["policy_basis"],
                         "user_defined_project_semantics")
        human_event = next(item for item in first["events"]
                           if item["state"] == "human_accepted")
        self.assertEqual(human_event["payload"]["slope_context"]["candidate_sha256"],
                         self.truth_sha)
        self.assertEqual(first["closed_loop_linkage"]["candidate_final_event_id"],
                         self.candidate["event_id"])
        self.assertIsNotNone(first["closed_loop_linkage"]["human_accepted_event_id"])
        self.assertIsNotNone(first["closed_loop_linkage"]["truth_committed_event_id"])
        truth_event = next(item for item in first["events"]
                           if item["state"] == "truth_committed")
        self.assertEqual(truth_event["payload"]["accepted_event_id"],
                         first["closed_loop_linkage"]["human_accepted_event_id"])
        self.assertEqual(truth_event["payload"]["accepted_candidate_event_id"],
                         self.candidate["event_id"])
        self.assertEqual(truth_event["payload"]["accepted_artifact_sha256"],
                         self.truth_sha)
        link = self.work / "human_review_links" / f"{first['human_review']['review_id']}.json"
        self.assertTrue(link.is_file())

    def test_changed_truth_is_rejected(self):
        self.write_acceptance()
        state = json.loads((self.work / "state.json").read_text(encoding="utf-8"))
        state["truth"]["0"][0]["label"] = "changed"
        (self.work / "state.json").write_text(json.dumps(state), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "does not match"):
            sync_human_gate(bundle_path=self.bundle, workbench_root=self.work,
                ledger_root=self.ledger_root, output_path=self.root / "bad.json")

    def test_rejected_candidate_uses_explicit_remediation_state(self):
        self.write_rejection()
        result = sync_human_gate(bundle_path=self.bundle, workbench_root=self.work,
            ledger_root=self.ledger_root, output_path=self.root / "rejected.json")
        self.assertEqual(result["receipt"]["last_state"], "candidate_rejected")
        ledger = ClosedLoopLedger(self.ledger_root, self.envelope)
        ledger.append_event("remediation_required", actor={"type": "system", "id": "gate"},
            action="correct_rejected_candidate", label_layer="agent_candidate")
        self.assertEqual(ledger.events()[-1]["state"], "remediation_required")

    def test_actual_human_gate_bundle_validates_unchanged(self):
        try:
            import jsonschema  # noqa: F401
        except ImportError:
            self.skipTest("jsonschema development dependency is absent")
        self.write_acceptance()
        output = self.root / "schema-checked.json"
        sync_human_gate(bundle_path=self.bundle, workbench_root=self.work,
            ledger_root=self.ledger_root, output_path=output)
        result = validate_bundle_file(output,
            Path(__file__).parents[1] / "schemas" / "labelagent-closed-loop.schema.json")
        self.assertTrue(result["valid"], result["errors"])


if __name__ == "__main__":
    unittest.main()
