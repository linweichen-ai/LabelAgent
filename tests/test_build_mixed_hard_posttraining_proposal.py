import json
import tempfile
import unittest
from pathlib import Path

from tests.test_build_mixed_hard_adjudication_workspace import (
    MixedHardAdjudicationWorkspaceTest,
)
from tools.build_mixed_hard_adjudication_workspace import validate_packet
from tools.build_mixed_hard_posttraining_proposal import (
    build_posttraining_proposal,
)


class MixedHardPosttrainingProposalTest(unittest.TestCase):
    def fixture(self, root: Path):
        helper = MixedHardAdjudicationWorkspaceTest()
        packet = helper.packet(root / "packet")
        manifest, digest = validate_packet(packet)
        submission = helper.submission(manifest, digest)
        submission.update({
            "status": "declared_human_review_complete",
            "reviewed_at": "2026-11-24T12:00:00+08:00",
        })
        for row in submission["rows"]:
            row.update({
                "candidate_decision": "accept_candidate",
                "slope_presence": "absent",
                "class_conflict": "none",
                "pallet_quality": "not_applicable",
                "difficulty_flags": [],
            })
        path = root / "adjudication.json"
        path.write_text(json.dumps(submission), encoding="utf-8")
        return packet, path

    def test_builds_bound_proposal_without_training_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            packet, adjudication = self.fixture(Path(directory))
            proposal = build_posttraining_proposal(packet, adjudication)
            self.assertEqual(proposal["sample_count"], 24)
            self.assertEqual(len(proposal["rows"]), 24)
            self.assertFalse(proposal["approved_for_training"])
            self.assertFalse(proposal["training_authorized"])
            self.assertFalse(proposal["training_dataset_materialized"])
            self.assertFalse(proposal["model_training_started"])
            self.assertFalse(proposal["test24_used"])
            self.assertTrue(all(
                row["approved_for_training"] is False
                for row in proposal["rows"]))

    def test_rejects_any_uncertain_review(self):
        with tempfile.TemporaryDirectory() as directory:
            packet, adjudication = self.fixture(Path(directory))
            data = json.loads(adjudication.read_text(encoding="utf-8"))
            data["rows"][3]["slope_presence"] = "uncertain"
            adjudication.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "uncertain review"):
                build_posttraining_proposal(packet, adjudication)

    def test_rejects_template_or_incomplete_review(self):
        with tempfile.TemporaryDirectory() as directory:
            packet, adjudication = self.fixture(Path(directory))
            data = json.loads(adjudication.read_text(encoding="utf-8"))
            data["status"] = "pending_declared_human_review"
            adjudication.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not declared complete"):
                build_posttraining_proposal(packet, adjudication)

    def test_requires_timezone_bound_review_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            packet, adjudication = self.fixture(Path(directory))
            data = json.loads(adjudication.read_text(encoding="utf-8"))
            data["reviewed_at"] = "2026-11-24T12:00:00"
            adjudication.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "timezone"):
                build_posttraining_proposal(packet, adjudication)


if __name__ == "__main__":
    unittest.main()
