import json
from pathlib import Path
import tempfile
import unittest

from source.agent.closed_loop import ClosedLoopLedger, validate_input


def envelope(content="a" * 64):
    return {"dataset_id": "sfl-cpd15y", "source_batch": "batch-a-000",
            "frame_id": "0", "sensor": "/capture/camera_0/image",
            "captured_at": "2026-09-20T10:00:00.000000Z", "source_kind": "real",
            "content_sha256": content, "source_uri": "/private/example.mcap",
            "timestamp_source": "capture"}


def actor(kind, identifier):
    return {"type": kind, "id": identifier}


class ClosedLoopTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.ledger = ClosedLoopLedger(self.temporary.name, envelope())

    def test_input_requires_timezone_source_and_hash(self):
        self.assertEqual(validate_input(envelope())["source_kind"], "real")
        for change in ({"captured_at": "2026-09-20T10:00:00"},
                       {"source_kind": "mixed"}, {"content_sha256": "bad"}):
            with self.assertRaises(ValueError):
                validate_input(envelope() | change)

    def test_duplicate_input_and_event_are_idempotent(self):
        first = self.ledger.register()
        self.assertTrue(first["created"])
        self.assertEqual(len(self.ledger.events()), 1)
        second = ClosedLoopLedger(self.temporary.name, envelope())
        self.assertTrue(second.register()["idempotent_replay"])
        self.assertEqual(len(second.events()), 1)
        kwargs = dict(actor=actor("system", "gate"), action="route",
                      label_layer="preannotation", payload={"reason": "missing truth"})
        original = second.append_event("human_review_required", **kwargs)
        replay = second.append_event("human_review_required", **kwargs)
        self.assertEqual(original["event_id"], replay["event_id"])
        self.assertTrue(replay["idempotent_replay"])

    def test_same_identity_with_changed_content_is_rejected(self):
        self.ledger.register()
        with self.assertRaisesRegex(ValueError, "different content"):
            ClosedLoopLedger(self.temporary.name, envelope("b" * 64)).register()

    def test_open_set_cannot_write_ontology_or_truth(self):
        self.ledger.register()
        self.ledger.append_event("sam_candidate_ready", actor=actor("sam", "sam3"),
            action="segment", label_layer="preannotation")
        self.ledger.append_event("vlm_reviewed", actor=actor("vlm", "qwen3-vl-8b"),
            action="classify_and_check", label_layer="agent_candidate")
        candidate = {"open_set": {"status": "new_label_candidate",
            "visual_evidence": ["distinct fork geometry"], "candidate_name": "fork-extension",
            "nearest_existing_class": "forklift", "difference": "separate attachment",
            "confidence": 0.72, "ontology_write": False}}
        self.ledger.append_event("open_set_candidate", actor=actor("vlm", "qwen3-vl-8b"),
            action="propose_unknown", label_layer="agent_candidate", payload=candidate)
        with self.assertRaisesRegex(ValueError, "cannot write"):
            bad = json.loads(json.dumps(candidate))
            bad["open_set"]["ontology_write"] = True
            ClosedLoopLedger(tempfile.mkdtemp(), envelope()).append_event(
                "open_set_candidate", actor=actor("vlm", "qwen"), action="bad",
                label_layer="agent_candidate", payload=bad)

    def test_truth_requires_human_gate_and_full_path_for_completion(self):
        self.ledger.register()
        self.ledger.append_event("human_review_required", actor=actor("system", "gate"),
            action="request_reference", label_layer="preannotation")
        receipt = self.ledger.receipt()
        self.assertFalse(receipt["human_gate_passed"])
        self.assertFalse(receipt["minimum_closed_loop_complete"])
        self.ledger.append_event("human_accepted", actor=actor("human", "reviewer-a"),
            action="approve_reference", label_layer="human_truth")
        self.ledger.append_event("sam_candidate_ready", actor=actor("sam", "sam3"),
            action="segment", label_layer="preannotation")
        self.ledger.append_event("vlm_reviewed", actor=actor("vlm", "qwen"),
            action="classify_and_check", label_layer="agent_candidate")
        self.ledger.append_event("agent_actioned", actor=actor("agent", "labelagent"),
            action="audit_or_edit", label_layer="agent_candidate")
        self.ledger.append_event("vlm_reverified", actor=actor("vlm", "qwen"),
            action="recheck", label_layer="agent_candidate")
        self.ledger.append_event("candidate_final", actor=actor("agent", "labelagent"),
            action="freeze_candidate", label_layer="agent_candidate")
        self.ledger.append_event("human_review_required", actor=actor("system", "gate"),
            action="route_final_candidate", label_layer="agent_candidate")
        self.ledger.append_event("human_accepted", actor=actor("human", "reviewer-a"),
            action="accept_candidate", label_layer="human_truth")
        self.ledger.append_event("truth_committed", actor=actor("system", "truth-writer"),
            action="commit_immutable_truth", label_layer="human_truth")
        receipt = self.ledger.receipt()
        self.assertTrue(receipt["human_gate_passed"])
        self.assertTrue(receipt["full_agent_path_observed"])
        self.assertTrue(receipt["minimum_closed_loop_complete"])
        self.assertFalse(receipt["training_ready"])

    def test_nonhuman_cannot_approve_and_hash_chain_detects_tampering(self):
        self.ledger.register()
        self.ledger.append_event("human_review_required", actor=actor("system", "gate"),
            action="route", label_layer="agent_candidate")
        with self.assertRaisesRegex(ValueError, "human actor"):
            self.ledger.append_event("human_accepted", actor=actor("agent", "bad"),
                action="approve", label_layer="human_truth")
        event_path = sorted((self.ledger.run / "events").glob("*.json"))[-1]
        event = json.loads(event_path.read_text(encoding="utf-8"))
        event["payload"] = {"tampered": True}
        event_path.write_text(json.dumps(event), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "hash chain"):
            self.ledger.events()

    def test_machine_schema_is_valid_json(self):
        schema = Path(__file__).resolve().parents[1] / "schemas/labelagent-closed-loop.schema.json"
        parsed = json.loads(schema.read_text(encoding="utf-8"))
        self.assertEqual(parsed["$schema"], "https://json-schema.org/draft/2020-12/schema")


if __name__ == "__main__":
    unittest.main()
