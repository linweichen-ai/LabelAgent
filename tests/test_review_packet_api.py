import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from source.agent import server


def manifest(mcap, *, frame_sha="a" * 64):
    row = {
        "frame": 7,
        "cohort": "positive_proxy",
        "source_frame_sha256": frame_sha,
        "route": "full_frame",
        "candidate_count": 1,
        "target_provenance": "legacy_auto_preannotation",
        "review_status": "pending_human_review",
        "allowed_human_decisions": [
            "slope_present_and_mask_acceptable",
            "slope_present_mask_needs_correction",
            "no_slope_present",
            "uncertain",
        ],
        "human_decision": None,
        "reviewer_declared_id": None,
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "training_return_allowed": False,
    }
    return {
        "schema_version": "labelagent.slope-review-packet.v1",
        "mcap": str(mcap),
        "mcap_sha256": hashlib.sha256(mcap.read_bytes()).hexdigest(),
        "topic": "/camera",
        "summary": {
            "frames": 1,
            "positive_proxy_frames": 1,
            "negative_cohort_frames": 0,
            "pending_human_review": 1,
            "human_truth_frames": 0,
            "training_approved_frames": 0,
        },
        "rows": [row],
    }


class ReviewPacketApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.mcap = self.work / "source.mcap"
        self.mcap.write_bytes(b"bound-review-data")
        self.packet = self.work / "review_packets" / "slope-round"
        self.packet.mkdir(parents=True)
        self.value = manifest(self.mcap)
        (self.packet / "manifest.json").write_text(
            json.dumps(self.value, ensure_ascii=False), encoding="utf-8")
        self.work_patch = patch.object(server, "WORK", self.work)
        self.work_patch.start()
        self.addCleanup(self.work_patch.stop)

    def test_verified_packet_is_sanitized_and_never_promotes_truth_or_training(self):
        packet = server._read_review_packet("slope-round")
        self.assertEqual(packet["summary"]["frames"], 1)
        self.assertTrue(packet["read_only_packet"])
        self.assertFalse(packet["submits_human_truth"])
        row = packet["rows"][0]
        self.assertTrue(row["candidate_only"])
        self.assertFalse(row["human_truth"])
        self.assertFalse(row["approved_for_training"])
        self.assertFalse(row["training_return_allowed"])
        self.assertNotIn("raw_image", row)
        self.assertNotIn("review_image", row)

    def test_any_attempt_to_promote_a_packet_row_is_rejected(self):
        for field, value in (
            ("candidate_only", False), ("human_truth", True),
            ("approved_for_training", True), ("training_return_allowed", True),
            ("review_status", "approved"), ("human_decision", "accepted"),
            ("reviewer_declared_id", "assistant"),
        ):
            with self.subTest(field=field):
                changed = json.loads(json.dumps(self.value))
                changed["rows"][0][field] = value
                (self.packet / "manifest.json").write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "隔离门禁"):
                    server._read_review_packet("slope-round")
        (self.packet / "manifest.json").write_text(json.dumps(self.value), encoding="utf-8")

    def test_index_rejects_bad_manifest_without_breaking_valid_packets(self):
        bad = self.work / "review_packets" / "bad"
        bad.mkdir()
        (bad / "manifest.json").write_text("{}", encoding="utf-8")
        response = server.app.test_client().get("/api/review_packets")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json["packets"]), 1)
        self.assertEqual(response.json["rejected_packets"], 1)
        self.assertFalse(response.json["raw_images_served"])
        self.assertFalse(response.json["auto_truth_submission"])

    def test_activation_requires_mcap_hash_and_reports_zero_truth_writes(self):
        old_state, old_status = server.state, server.run_status
        server.state = {"video": None, "topic": None}
        server.run_status = {"status": "idle"}
        try:
            bootstrap = {"loaded": True, "path": str(self.mcap), "topic": "/camera",
                         "topics": {"/camera": {"count": 8}}, "compact_geometry": True}
            with patch.object(server, "activate_file", return_value=bootstrap), \
                 patch.object(server, "_activate_review_packet_topic"), \
                 patch.object(server, "_get_bootstrap_locked", return_value=bootstrap):
                response = server.app.test_client().post("/api/review_packets/slope-round/activate", json={})
            self.assertEqual(response.status_code, 200)
            receipt = response.json["review_packet_activation"]
            self.assertTrue(receipt["mcap_sha256_verified"])
            self.assertTrue(receipt["topic_verified"])
            self.assertFalse(receipt["human_truth_written"])
            self.assertFalse(receipt["training_data_written"])
            self.mcap.write_bytes(b"changed")
            response = server.app.test_client().post("/api/review_packets/slope-round/activate", json={})
            self.assertEqual(response.status_code, 400)
            self.assertIn("内容哈希不一致", response.json["error"])
        finally:
            server.state, server.run_status = old_state, old_status


if __name__ == "__main__":
    unittest.main(verbosity=2)
