import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import av
from PIL import Image
from mcap.writer import Writer
from source.agent import server
from source.agent.media import StillImage, open_media
from source.agent.mcap_video import Video


def photograph(fmt="PNG", orientation=None):
    image = Image.new("RGB", (41, 33), (25, 130, 220))
    output = io.BytesIO()
    options = {}
    if orientation:
        exif = Image.Exif()
        exif[274] = orientation
        options["exif"] = exif
    image.save(output, fmt, **options)
    return output.getvalue()


class ImageImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.patcher = patch.multiple(server, WORK=self.root, state={
            "video": None, "topic": None, "labels": ["车辆"], "truth": {},
            "reference_frames": {}, "config": {}, "candidates": {}, "agent_runs": {}})
        self.patcher.start()
        self.client = server.app.test_client()

    def tearDown(self):
        self.patcher.stop()
        self.temp.cleanup()

    def upload(self, data, name):
        return self.client.post("/api/upload", data={"file": (io.BytesIO(data), name)})

    @staticmethod
    def review(reviewer="test-reviewer", key=None, remediation=None):
        result = {"reviewer_declared_id": reviewer,
                  "started_at": "2026-11-11T00:00:00Z", "note": "test",
                  "decision_idempotency_key": key or f"decision-{reviewer}"}
        if remediation is not None:
            result["remediation_of_review_id"] = remediation
        return result

    def binding(self, frame=0):
        response = self.client.post("/api/human_review_binding", json={
            "frame": frame, "topic": server.state["topic"]})
        self.assertEqual(response.status_code, 200, response.get_json())
        return {"topic": response.json["topic"],
                "expected_frame_sha256": response.json["frame_sha256"],
                "expected_candidate_sha256": response.json["candidate_sha256"],
                "expected_dataset_id": response.json["dataset_id"],
                "expected_dataset_version": response.json["dataset_version"],
                "expected_camera": response.json["camera"]}

    def test_chinese_upload_original_pixels_odd_size_preview_and_persistence(self):
        result = self.upload(photograph(), "自己的图片.PNG")
        self.assertEqual(result.status_code, 200)
        item = result.json["file"]
        self.assertEqual(item["name"], "自己的图片.PNG")
        self.assertEqual(Path(item["path"]).suffix, ".png")
        self.assertEqual(self.client.post(f"/api/files/{item['id']}/load").status_code, 200)
        frame = self.client.get("/api/frame/0")
        self.assertEqual(frame.content_type, "image/png")
        self.assertEqual(frame.data, photograph())
        playback = self.client.get("/api/playback").json
        self.assertEqual(playback["status"], "ready")
        with av.open(str(self.root / "playback" / Path(playback["url"]).name)) as container:
            decoded = next(container.decode(video=0))
            self.assertEqual((decoded.width, decoded.height), (42, 34))
        shape = {"label": "车辆", "points": [[1, 1], [30, 1], [30, 25]], "holes": []}
        binding = self.binding()
        response = self.client.post("/api/truth", json={"frame": 0,
            "shapes": [shape], "review": self.review(), **binding})
        self.assertEqual(response.json["saved"], 1)
        self.assertTrue(response.json["readback"]["verified"])
        self.assertEqual(response.json["review"]["decision"], "manual_annotation")
        audit = self.client.get("/api/human_reviews?frame=0").json["reviews"]
        self.assertEqual(len(audit), 1)
        self.assertEqual(audit[0]["reviewer_declared_id"], "test-reviewer")
        self.assertEqual(audit[0]["identity_assurance"], "self_declared")
        self.assertEqual(audit[0]["truth_shape_count"], 1)
        other = self.upload(photograph("JPEG"), "另一个.jpg").json["file"]
        self.client.post(f"/api/files/{other['id']}/load")
        again = self.client.post(f"/api/files/{item['id']}/load").json
        self.assertEqual(again["truth"]["0"], [shape])
        persisted = json.loads((self.root / "state.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["truth"]["0"], [shape])

    def test_candidate_requires_explicit_human_gate_and_rejection_never_writes_truth(self):
        item = self.upload(photograph(), "candidate.png").json["file"]
        self.client.post(f"/api/files/{item['id']}/load")
        shape = {"label": "车辆", "points": [[1, 1], [30, 1], [30, 25]], "holes": []}
        server.state["candidates"]["0"] = [shape]
        binding = self.binding()
        missing = self.client.post("/api/truth", json={"frame": 0, "shapes": [shape],
            **binding})
        self.assertEqual(missing.status_code, 400)
        rejected = self.client.post("/api/human_review", json={"frame": 0,
            "review": self.review("rejector"), **binding})
        self.assertEqual(rejected.status_code, 200)
        self.assertEqual(rejected.json["review"]["decision"], "reject_candidate")
        self.assertTrue(rejected.json["remediation_required"])
        self.assertTrue(rejected.json["readback"]["verified"])
        self.assertNotIn("0", server.state["truth"])
        self.assertNotIn("0", server.state["candidates"])
        self.assertIn(rejected.json["review"]["review_id"],
                      server.state["rejected_candidates"])
        status = next(iter(server.state["candidate_review_status"].values()))
        self.assertEqual(status["status"], "rejected")
        self.assertEqual(status["route"], "agent_remediation_required")
        replay = self.client.post("/api/human_review", json={"frame": 0,
            "review": self.review("rejector"), **binding})
        self.assertTrue(replay.json["idempotent_replay"])
        self.assertTrue(replay.json["readback"]["verified"])
        duplicate = self.client.post("/api/human_review", json={"frame": 0,
            "review": self.review("rejector", key="different-rejection"), **binding})
        self.assertEqual(duplicate.status_code, 400)
        inactive_binding = self.binding()
        unchanged = self.client.post("/api/truth", json={"frame": 0,
            "shapes": [shape], "review": self.review("acceptor-unchanged",
                remediation=rejected.json["review"]["review_id"]), **inactive_binding})
        self.assertEqual(unchanged.status_code, 400)
        corrected_shape = {**shape, "points": [[2, 2], [30, 2], [30, 25]]}
        server.state["candidates"]["0"] = [corrected_shape]
        corrected_binding = self.binding()
        accepted = self.client.post("/api/truth", json={"frame": 0,
            "shapes": [corrected_shape], "review": self.review("acceptor",
                remediation=rejected.json["review"]["review_id"]), **corrected_binding})
        self.assertEqual(accepted.json["review"]["decision"], "accept_candidate")
        self.assertEqual(accepted.json["review"]["candidate_sha256"],
                         server._sha256_json([corrected_shape]))
        self.assertTrue(accepted.json["readback"]["verified"])
        self.assertEqual(accepted.json["review"]["closed_loop_linkage"]["status"],
                         "pending_sync")
        self.assertIn("0", server.state["truth"])
        self.assertNotIn("0", server.state["candidates"])
        audits = self.client.get("/api/human_reviews?frame=0").json["reviews"]
        self.assertEqual([item["decision"] for item in audits],
                         ["reject_candidate", "accept_candidate"])

    def test_stale_frame_or_candidate_binding_returns_409_without_audit(self):
        item = self.upload(photograph(), "stale.png").json["file"]
        self.client.post(f"/api/files/{item['id']}/load")
        original = {"label": "车辆", "points": [[1, 1], [30, 1], [30, 25]], "holes": []}
        changed = {**original, "points": [[2, 2], [30, 2], [30, 25]]}
        server.state["candidates"]["0"] = [original]
        stale_candidate = self.binding()
        server.state["candidates"]["0"] = [changed]
        response = self.client.post("/api/human_review", json={"frame": 0,
            "review": self.review("stale-candidate"), **stale_candidate})
        self.assertEqual(response.status_code, 409)
        self.assertTrue(response.json["stale_review"])
        current = self.binding()
        stale_frame = {**current, "expected_frame_sha256": "0" * 64}
        response = self.client.post("/api/human_review", json={"frame": 0,
            "review": self.review("stale-frame"), **stale_frame})
        self.assertEqual(response.status_code, 409)
        self.assertTrue(response.json["stale_review"])
        stale_scope = self.binding()
        server.state["config"]["dataset_version"] = "new-dataset-version"
        response = self.client.post("/api/human_review", json={"frame": 0,
            "review": self.review("stale-scope"), **stale_scope})
        self.assertEqual(response.status_code, 409)
        self.assertTrue(response.json["stale_review"])
        self.assertEqual(server.state["candidates"]["0"], [changed])
        self.assertNotIn("0", server.state["truth"])
        self.assertEqual(self.client.get("/api/human_reviews?frame=0").json["reviews"], [])

    def test_forced_state_save_failure_leaves_no_false_audit_or_untracked_truth(self):
        item = self.upload(photograph(), "failure.png").json["file"]
        self.client.post(f"/api/files/{item['id']}/load")
        shape = {"label": "车辆", "points": [[1, 1], [30, 1], [30, 25]], "holes": []}
        binding = self.binding()
        with patch.object(server, "save", side_effect=OSError("forced save failure")):
            response = self.client.post("/api/truth", json={"frame": 0,
                "shapes": [shape], "review": self.review("failure-reviewer"), **binding})
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("0", server.state["truth"])
        self.assertEqual(server.state.get("human_review_transactions", {}), {})
        self.assertEqual(list((self.root / "human_reviews").glob("*.json"))
                         if (self.root / "human_reviews").exists() else [], [])
        self.assertEqual(list((self.root / "human_review_pending").glob("*.json"))
                         if (self.root / "human_review_pending").exists() else [], [])
        persisted = json.loads((self.root / "state.json").read_text(encoding="utf-8"))
        self.assertNotIn("0", persisted.get("truth", {}))

    def test_final_transaction_save_failure_is_linked_and_recoverable(self):
        item = self.upload(photograph(), "recover.png").json["file"]
        self.client.post(f"/api/files/{item['id']}/load")
        shape = {"label": "车辆", "points": [[1, 1], [30, 1], [30, 25]], "holes": []}
        binding = self.binding()
        real_save = server.save
        calls = 0
        def fail_second_save():
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("forced final transaction save failure")
            real_save()
        with patch.object(server, "save", side_effect=fail_second_save):
            response = self.client.post("/api/truth", json={"frame": 0,
                "shapes": [shape], "review": self.review("recovery-reviewer"), **binding})
        self.assertEqual(response.status_code, 500)
        persisted = json.loads((self.root / "state.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["truth"]["0"], [shape])
        transaction = next(iter(persisted["human_review_transactions"].values()))
        self.assertEqual(transaction["status"], "state_saved_pending_audit")
        self.assertEqual(len(list((self.root / "human_reviews").glob("*.json"))), 1)
        recovered = self.client.get("/api/human_reviews?frame=0")
        self.assertEqual(recovered.status_code, 200)
        self.assertEqual(len(recovered.json["reviews"]), 1)
        final_state = json.loads((self.root / "state.json").read_text(encoding="utf-8"))
        final_transaction = next(iter(final_state["human_review_transactions"].values()))
        self.assertEqual(final_transaction["status"], "committed")

    def test_empty_truth_requires_explicit_confirmation_and_is_not_rejection(self):
        item = self.upload(photograph(), "empty.png").json["file"]
        self.client.post(f"/api/files/{item['id']}/load")
        shape = {"label": "车辆", "points": [[1, 1], [30, 1], [30, 25]], "holes": []}
        server.state["candidates"]["0"] = [shape]
        binding = self.binding()
        silent = self.client.post("/api/truth", json={"frame": 0, "shapes": [],
            "review": self.review("empty-reviewer", key="empty-silent"), **binding})
        self.assertEqual(silent.status_code, 400)
        self.assertIn("0", server.state["candidates"])
        implicit = self.client.post("/api/truth", json={"frame": 0, "shapes": [],
            "confirmed_empty": True,
            "review": self.review("empty-reviewer", key="empty-implicit"), **binding})
        self.assertEqual(implicit.status_code, 400)
        confirmed = self.client.post("/api/truth", json={"frame": 0, "shapes": [],
            "confirmed_empty": True, "decision": "confirmed_empty",
            "review": self.review("empty-reviewer", key="empty-confirmed"), **binding})
        self.assertEqual(confirmed.status_code, 200)
        self.assertEqual(confirmed.json["review"]["decision"], "confirmed_empty")
        self.assertTrue(confirmed.json["review"]["confirmed_empty"])
        self.assertEqual(server.state["truth"]["0"], [])

    def test_slope_human_review_receipt_keeps_visible_and_traversability_separate(self):
        item = self.upload(photograph(), "slope.png").json["file"]
        self.client.post(f"/api/files/{item['id']}/load")
        server.state["labels"] = ["车辆", "slope"]
        shape = {"label": "slope", "points": [[1, 1], [30, 1], [30, 25]],
                 "holes": [], "score": 0.98}
        server.state["candidates"]["0"] = [shape]
        binding = self.binding()
        response = self.client.post("/api/truth", json={"frame": 0,
            "shapes": [shape], "review": self.review("slope-reviewer"), **binding})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["readback"]["verified"])
        receipt = response.json["review"]
        slope = receipt["slope_context"]
        self.assertEqual(slope["semantic_label"], "slope_visible")
        self.assertEqual(slope["visible_status"], "human_confirmed")
        self.assertTrue(slope["human_pixel_truth_available"])
        self.assertFalse(slope["approved_for_training"])
        self.assertEqual(slope["business_mapping"], "traversable_area")
        self.assertEqual(slope["traversability_status"], "passable")
        self.assertEqual(slope["policy_default"], "passable")
        self.assertEqual(slope["policy_basis"], "user_defined_project_semantics")
        self.assertEqual(slope["policy_override"]["status"], "none")
        self.assertEqual(slope["candidate_sha256"], receipt["candidate_sha256"])
        self.assertEqual(slope["hard_set_strata"], ["unassigned"])
        self.assertEqual(slope["evidence_provenance"][0]["sha256"],
                         receipt["candidate_sha256"])
        self.assertEqual(receipt["expected_frame_sha256"], receipt["frame_sha256"])
        self.assertEqual(receipt["expected_candidate_sha256"], receipt["candidate_sha256"])

    def test_downstream_projection_unions_floor_and_slope_without_mutating_semantics(self):
        item = self.upload(photograph(), "projection.png").json["file"]
        self.client.post(f"/api/files/{item['id']}/load")
        source = [
            {"label": "floor", "points": [[1, 1], [10, 1], [10, 10]], "holes": []},
            {"label": "slope", "points": [[11, 1], [20, 1], [20, 10]], "holes": []},
            {"label": "person", "points": [[21, 1], [30, 1], [30, 10]], "holes": []},
        ]
        server.state["candidates"]["0"] = json.loads(json.dumps(source))
        response = self.client.get("/api/downstream_projection/0")
        self.assertEqual(response.status_code, 200)
        projection = response.json["projection"]
        self.assertEqual(projection["source_layer"], "machine_candidate")
        self.assertTrue(projection["machine_candidate_only"])
        self.assertTrue(projection["training_labels_preserved"])
        self.assertEqual(projection["mapping"]["operation"], "union")
        self.assertEqual(projection["mapping"]["source_labels"], ["floor", "slope"])
        self.assertEqual(projection["mapping"]["target_label"], "traversable_area")
        self.assertEqual([item["source_semantic_label"] for item in
                          projection["traversable_area_parts"]], ["floor", "slope"])
        self.assertTrue(all(item["label"] == "traversable_area"
                            for item in projection["traversable_area_parts"]))
        self.assertEqual(projection["passthrough_shapes"][0]["label"], "person")
        self.assertEqual([item["label"] for item in server.state["candidates"]["0"]],
                         ["floor", "slope", "person"])

        server.state["truth"]["0"] = [source[0]]
        projection = self.client.get("/api/downstream_projection/0").json["projection"]
        self.assertEqual(projection["source_layer"], "human_truth")
        self.assertFalse(projection["machine_candidate_only"])

    def test_exif_orientation_and_path_import(self):
        path = self.root / "手机照片.jpeg"
        path.write_bytes(photograph("JPEG", orientation=6))
        response = self.client.post("/api/open", json={"path": str(path)})
        self.assertEqual(response.status_code, 200)
        with Image.open(io.BytesIO(self.client.get("/api/frame/0").data)) as image:
            self.assertEqual(image.size, (33, 41))
        self.assertIsInstance(server.state["video"], StillImage)

    def test_corrupt_image_rejected_without_library_or_upload_residue(self):
        response = self.upload(b"not an image", "bad.png")
        self.assertEqual(response.status_code, 400)
        self.assertFalse((self.root / "files.json").exists())
        self.assertEqual(list((self.root / "uploads").iterdir()), [])

    def test_mcap_remains_on_existing_decoder(self):
        path = self.root / "existing.mcap"
        with path.open("wb") as output:
            writer = Writer(output)
            writer.start()
            schema = writer.register_schema("test.CompressedImage", "", b"")
            channel = writer.register_channel("camera", "", schema)
            writer.add_message(channel, log_time=1000000000, publish_time=1000000000, data=photograph())
            writer.finish()
        record = open_media(path)
        self.assertIsInstance(record, Video)
        try:
            with Image.open(io.BytesIO(record.frame("camera", 0))) as image:
                self.assertEqual(image.size, (41, 33))
        finally:
            for decoder in record._decoder_state.values():
                decoder["stream"].close()


if __name__ == "__main__":
    unittest.main()
