import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
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

    def test_assistant_proposal_is_hash_bound_and_never_becomes_a_decision(self):
        packet = server._read_review_packet("slope-round")
        proposal = {
            "schema_version": "labelagent.slope-assistant-review.v1",
            "packet_id": "slope-round",
            "manifest_sha256": packet["manifest_sha256"],
            "rows": [{
                "frame": 7,
                "proposed_decision": "slope_present_and_mask_acceptable",
                "confidence_band": "high",
                "rationale": "Visible transition ramp; candidate follows the visible boundary.",
                "assistant_only": True,
                "human_truth": False,
                "approved_for_training": False,
                "auto_submit_allowed": False,
            }],
        }
        (self.packet / "assistant-review.json").write_text(
            json.dumps(proposal), encoding="utf-8")
        loaded = server._read_review_packet("slope-round")
        suggestion = loaded["rows"][0]["assistant_proposal"]
        self.assertEqual(suggestion["proposed_decision"],
                         "slope_present_and_mask_acceptable")
        self.assertTrue(suggestion["assistant_only"])
        self.assertFalse(suggestion["human_truth"])
        self.assertFalse(suggestion["approved_for_training"])
        self.assertFalse(suggestion["auto_submit_allowed"])
        self.assertEqual(len(suggestion["proposal_sha256"]), 64)
        self.assertEqual(loaded["assistant_proposals_loaded"], 1)
        proposal["rows"][0]["auto_submit_allowed"] = True
        (self.packet / "assistant-review.json").write_text(
            json.dumps(proposal), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "隔离门禁"):
            server._read_review_packet("slope-round")

    def test_priority_sidecar_is_hash_bound_and_cannot_promote_truth(self):
        packet = server._read_review_packet("slope-round")
        priority = {
            "schema_version": "labelagent.slope-review-priority.v1",
            "packet_id": "slope-round",
            "manifest_sha256": packet["manifest_sha256"],
            "source_review_schema": "labelagent.slope-blind-cross-review.v1",
            "source_review_sha256": "b" * 64,
            "assistant_only": True,
            "human_truth": False,
            "approved_for_training": False,
            "auto_submit_allowed": False,
            "rows": [{
                "frame": 7, "priority_rank": 1,
                "priority_reason": "partially visible boundary",
                "assistant_only": True, "human_truth": False,
                "approved_for_training": False, "auto_submit_allowed": False,
            }],
        }
        priority_path = self.packet / "review-priority.json"
        priority_path.write_text(json.dumps(priority), encoding="utf-8")
        loaded = server._read_review_packet("slope-round")
        self.assertEqual(loaded["priority_reviews_loaded"], 1)
        self.assertEqual(loaded["rows"][0]["review_priority"]["priority_rank"], 1)
        self.assertFalse(loaded["rows"][0]["review_priority"]["human_truth"])
        priority["rows"][0]["approved_for_training"] = True
        priority_path.write_text(json.dumps(priority), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "隔离门禁"):
            server._read_review_packet("slope-round")

    def test_scoped_decision_is_append_only_idempotent_and_writes_no_truth(self):
        frame_bytes = b"exact-review-frame"
        self.value = manifest(
            self.mcap, frame_sha=hashlib.sha256(frame_bytes).hexdigest())
        (self.packet / "manifest.json").write_text(
            json.dumps(self.value, ensure_ascii=False), encoding="utf-8")
        packet = server._read_review_packet("slope-round")
        old_state, old_status = server.state, server.run_status
        slope_shape = {"label": "slope", "points": [[1, 1], [4, 1], [4, 4]],
                       "holes": []}
        server.state = {"video": SimpleNamespace(path=self.mcap), "topic": "/camera",
                        "truth": {"7": [slope_shape]}, "candidates": {}, "config": {}}
        server.run_status = {"status": "idle"}
        review = {"reviewer_declared_id": "human-reviewer",
                  "started_at": "2026-11-18T05:00:00Z", "note": "slope only",
                  "decision_idempotency_key": "slope-review-frame-7-v1"}
        request_body = {
            "frame": 7,
            "decision": "slope_present_and_mask_acceptable",
            "expected_manifest_sha256": packet["manifest_sha256"],
            "expected_frame_sha256": hashlib.sha256(frame_bytes).hexdigest(),
            "expected_assistant_proposal_sha256": None,
            "expected_slope_geometry_sha256": server._sha256_json([slope_shape]),
            "review": review,
        }
        try:
            with patch.object(server, "frame_image", return_value=frame_bytes), \
                 patch.object(server, "save") as save_mock:
                client = server.app.test_client()
                response = client.post(
                    "/api/review_packets/slope-round/decisions", json=request_body)
                replay = client.post(
                    "/api/review_packets/slope-round/decisions", json=request_body)
                listing = client.get("/api/review_packets/slope-round/decisions")
            self.assertEqual(response.status_code, 200, response.json)
            self.assertTrue(response.json["readback"]["verified"])
            self.assertFalse(response.json["readback"]["idempotent_replay"])
            receipt = response.json["decision_receipt"]
            self.assertTrue(receipt["human_decision"])
            self.assertFalse(receipt["human_truth"])
            self.assertFalse(receipt["approved_for_training"])
            self.assertFalse(receipt["training_return_allowed"])
            self.assertFalse(response.json["human_truth_written"])
            self.assertFalse(response.json["training_data_written"])
            self.assertEqual(replay.status_code, 200, replay.json)
            self.assertTrue(replay.json["readback"]["idempotent_replay"])
            self.assertEqual(replay.json["decision_receipt"]["decision_id"],
                             receipt["decision_id"])
            self.assertEqual(listing.status_code, 200, listing.json)
            self.assertEqual(listing.json["history_count"], 1)
            self.assertEqual(len(listing.json["decisions"]), 1)
            self.assertFalse(listing.json["human_truth_written"])
            self.assertFalse(listing.json["training_data_written"])
            self.assertEqual(server.state["truth"], {"7": [slope_shape]})
            self.assertEqual(server.state["candidates"], {})
            save_mock.assert_not_called()
            files = list((self.packet / "decisions").glob("*.json"))
            self.assertEqual(len(files), 1)
        finally:
            server.state, server.run_status = old_state, old_status

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
