import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from source.agent import server


def png_bytes():
    target = io.BytesIO()
    Image.new("RGB", (32, 32), (20, 30, 40)).save(target, "PNG")
    return target.getvalue()


class ComponentReviewPacketApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.packet = self.work / "component_review_packs" / "ego-tail"
        self.packet.mkdir(parents=True)
        image = png_bytes()
        (self.packet / "c1-focused.png").write_bytes(image)
        self.value = {
            "schema_version": "labelagent.ego_tail_focused_review_pack.v1",
            "candidate_only": True,
            "assistant_review_only": True,
            "human_truth": False,
            "approved_for_training": False,
            "production_deployed": False,
            "default_enabled": False,
            "automatic_decisions_enabled": False,
            "review_count": 1,
            "priority_review": 1,
            "deferred_manual_review": 0,
            "rows": [{
                "component_id": "004-camera1-c001",
                "camera_rank": 1,
                "policy_queue": "priority_review",
                "policy_reason": "uncertain boundary",
                "focused_image": "c1-focused.png",
                "focused_image_sha256": hashlib.sha256(image).hexdigest(),
            }],
        }
        (self.packet / "focused_review_manifest.json").write_text(
            json.dumps(self.value), encoding="utf-8")
        self.work_patch = patch.object(server, "WORK", self.work)
        self.work_patch.start()
        self.addCleanup(self.work_patch.stop)

    def test_verified_packet_is_sanitized_and_read_only(self):
        packet = server._read_component_review_packet("ego-tail")
        self.assertTrue(packet["read_only_packet"])
        self.assertTrue(packet["session_drafts_only"])
        self.assertFalse(packet["submits_human_truth"])
        self.assertFalse(packet["training_data_writes"])
        row = packet["rows"][0]
        self.assertTrue(row["candidate_only"])
        self.assertFalse(row["human_truth"])
        self.assertFalse(row["approved_for_training"])
        self.assertIsNone(row["automatic_decision"])
        self.assertNotIn("source_image", row)

    def test_index_and_detail_expose_no_disk_paths_or_write_capability(self):
        client = server.app.test_client()
        listing = client.get("/api/component_review_packs")
        detail = client.get("/api/component_review_packs/ego-tail")
        self.assertEqual(listing.status_code, 200, listing.json)
        self.assertEqual(listing.json["schema_version"],
                         "labelagent.component-review-pack-index.v1")
        self.assertFalse(listing.json["human_truth_writes"])
        self.assertFalse(listing.json["training_data_writes"])
        self.assertEqual(detail.status_code, 200, detail.json)
        self.assertNotIn(str(self.work), json.dumps(detail.json))
        self.assertEqual(detail.json["rows"][0]["image_url"],
                         "/api/component_review_packs/ego-tail/images/c1-focused.png")
        self.assertEqual(client.post("/api/component_review_packs/ego-tail").status_code, 405)

    def test_serves_only_hash_bound_manifest_images(self):
        client = server.app.test_client()
        response = client.get(
            "/api/component_review_packs/ego-tail/images/c1-focused.png")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.data.startswith(b"\x89PNG"))
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
        self.assertIn("no-store", response.headers["Cache-Control"])
        missing = client.get(
            "/api/component_review_packs/ego-tail/images/not-bound.png")
        self.assertEqual(missing.status_code, 400)

    def test_rejects_promoted_flags_queue_drift_and_image_tamper(self):
        manifest = self.packet / "focused_review_manifest.json"
        for field, value in (
            ("candidate_only", False), ("human_truth", True),
            ("approved_for_training", True),
            ("automatic_decisions_enabled", True),
        ):
            with self.subTest(field=field):
                changed = json.loads(json.dumps(self.value))
                changed[field] = value
                manifest.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "隔离门禁"):
                    server._read_component_review_packet("ego-tail")
        changed = json.loads(json.dumps(self.value))
        changed["rows"][0]["policy_queue"] = "auto_accept"
        manifest.write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "未知复核队列"):
            server._read_component_review_packet("ego-tail")
        manifest.write_text(json.dumps(self.value), encoding="utf-8")
        (self.packet / "c1-focused.png").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "哈希不一致"):
            server._read_component_review_packet("ego-tail")

    def test_rejects_rank_path_and_fake_image(self):
        manifest = self.packet / "focused_review_manifest.json"
        changed = json.loads(json.dumps(self.value))
        changed["rows"][0]["camera_rank"] = "1"
        manifest.write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "相机排名"):
            server._read_component_review_packet("ego-tail")
        second = json.loads(json.dumps(self.value["rows"][0]))
        second["component_id"] = "004-camera1-c002"
        second["focused_image"] = "c2-focused.png"
        second["camera_rank"] = 3
        (self.packet / "c2-focused.png").write_bytes(png_bytes())
        second["focused_image_sha256"] = hashlib.sha256(png_bytes()).hexdigest()
        changed = json.loads(json.dumps(self.value))
        changed["review_count"] = 2
        changed["priority_review"] = 2
        changed["rows"].append(second)
        manifest.write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "排名不连续"):
            server._read_component_review_packet("ego-tail")
        changed = json.loads(json.dumps(self.value))
        changed["rows"][0]["focused_image"] = "../escape.png"
        manifest.write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "图片绑定"):
            server._read_component_review_packet("ego-tail")
        changed = json.loads(json.dumps(self.value))
        changed["rows"][0]["focused_image"] = "unsafe%23.png"
        manifest.write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "图片绑定"):
            server._read_component_review_packet("ego-tail")
        fake = b"not-a-real-png"
        (self.packet / "c1-focused.png").write_bytes(fake)
        changed = json.loads(json.dumps(self.value))
        changed["rows"][0]["focused_image_sha256"] = hashlib.sha256(fake).hexdigest()
        manifest.write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "安全解码"):
            server._read_component_review_packet("ego-tail")

    def test_bad_packet_is_counted_without_breaking_valid_index(self):
        bad = self.work / "component_review_packs" / "bad"
        bad.mkdir()
        (bad / "focused_review_manifest.json").write_text("{}", encoding="utf-8")
        response = server.app.test_client().get("/api/component_review_packs")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json["packets"]), 1)
        self.assertEqual(response.json["rejected_packets"], 1)


if __name__ == "__main__":
    unittest.main()
