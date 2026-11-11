import json
from pathlib import Path
import tempfile
import unittest

from source.agent.closed_loop import ClosedLoopLedger, digest
from tools.closed_loop_human_gate import sync_human_gate


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
        self.ledger.append_event("human_review_required",
            actor={"type": "system", "id": "gate"}, action="route",
            label_layer="preannotation")
        self.bundle = self.root / "bundle.json"
        self.bundle.write_text(json.dumps({"schema_version": "labelagent.closed-loop.v1",
            "input": self.envelope, "events": self.ledger.events(),
            "receipt": self.ledger.receipt()}), encoding="utf-8")

    def write_acceptance(self):
        truth = [{"label": "pallet", "points": [[1, 1], [2, 1], [2, 2]]}]
        (self.work / "state.json").write_text(json.dumps({"topic": "/camera",
            "truth": {"0": truth}}, ensure_ascii=False), encoding="utf-8")
        body = {"schema_version": "labelagent.human-review.v1", "sequence": 1,
            "previous_review_id": None, "file": "example.mcap", "topic": "/camera",
            "frame": 0, "frame_sha256": "a" * 64, "decision": "accept_candidate",
            "reviewer_id": "reviewer", "started_at": "2026-11-11T00:00:00Z",
            "reviewed_at": "2026-11-11T00:00:12Z", "elapsed_seconds": 12.0,
            "note": "checked", "candidate_sha256": digest(truth),
            "candidate_shape_count": 1, "truth_sha256": digest(truth),
            "truth_shape_count": 1}
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
        self.assertEqual(first["human_review"]["reviewer_id"], "reviewer")

    def test_changed_truth_is_rejected(self):
        self.write_acceptance()
        state = json.loads((self.work / "state.json").read_text(encoding="utf-8"))
        state["truth"]["0"][0]["label"] = "changed"
        (self.work / "state.json").write_text(json.dumps(state), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "does not match"):
            sync_human_gate(bundle_path=self.bundle, workbench_root=self.work,
                ledger_root=self.ledger_root, output_path=self.root / "bad.json")


if __name__ == "__main__":
    unittest.main()
