import io
import copy
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

    def upload_batch(self, items):
        return self.client.post("/api/uploads", data={
            "files": [(io.BytesIO(data), name) for data, name in items]})

    def test_batch_upload_registers_new_image_dataset_with_one_library_commit(self):
        items = [(photograph("PNG" if index % 2 == 0 else "JPEG"),
                  f"新数据集_{index:02d}.{'png' if index % 2 == 0 else 'jpg'}")
                 for index in range(24)]
        with patch.object(server, "write_library", wraps=server.write_library) as write:
            response = self.upload_batch(items)
        self.assertEqual(response.status_code, 200, response.json)
        self.assertEqual(len(response.json["added"]), 24)
        self.assertEqual(len(response.json["files"]), 24)
        self.assertEqual(write.call_count, 1)
        self.assertEqual(len(json.loads((self.root / "files.json").read_text())), 24)
        first, last = response.json["added"][0], response.json["added"][-1]
        self.assertEqual(self.client.post(f"/api/files/{first['id']}/load").status_code, 200)
        loaded = self.client.post(f"/api/files/{last['id']}/load")
        self.assertEqual(loaded.status_code, 200, loaded.json)
        self.assertEqual(loaded.json["path"], last["path"])

    def test_batch_upload_is_atomic_when_one_image_is_corrupt(self):
        response = self.upload_batch([
            (photograph(), "good.png"), (b"not an image", "bad.png")])
        self.assertEqual(response.status_code, 400, response.json)
        self.assertFalse((self.root / "files.json").exists())
        self.assertEqual(list((self.root / "uploads").iterdir()), [])

    def test_registered_current_image_file_list_does_not_redecode_pixels(self):
        item = self.upload(photograph(), "large-current.png").json["file"]
        loaded = self.client.post(f"/api/files/{item['id']}/load")
        self.assertEqual(loaded.status_code, 200, loaded.json)
        with patch.object(server, "read_image",
                          side_effect=AssertionError("registered image was decoded again")):
            for _ in range(10):
                response = self.client.get("/api/files")
                self.assertEqual(response.status_code, 200, response.json)
                self.assertEqual(len(response.json["files"]), 1)

    def test_chinese_upload_original_pixels_odd_size_preview_and_persistence(self):
        result = self.upload(photograph(), "自己的图片.PNG")
        self.assertEqual(result.status_code, 200)
        item = result.json["file"]
        self.assertEqual(item["name"], "自己的图片.PNG")
        self.assertEqual(Path(item["path"]).suffix, ".png")
        loaded = self.client.post(f"/api/files/{item['id']}/load")
        self.assertEqual(loaded.status_code, 200)
        frame = self.client.get("/api/frame/0")
        self.assertEqual(frame.content_type, "image/png")
        self.assertEqual(frame.data, photograph())
        playback = self.client.get("/api/playback").json
        self.assertEqual(playback["status"], "ready")
        with av.open(str(self.root / "playback" / Path(playback["url"]).name)) as container:
            decoded = next(container.decode(video=0))
            self.assertEqual((decoded.width, decoded.height), (42, 34))
        shape = {"label": "车辆", "points": [[1, 1], [30, 1], [30, 25]], "holes": []}
        with patch.object(server, "frame_image", wraps=server.frame_image) as frame_reader:
            stale_binding = self.client.post("/api/human_review_binding", json={
                "frame": 0, "topic": loaded.json["topic"],
                "file_activation_generation": loaded.json["file_activation_generation"] - 1})
        self.assertEqual(stale_binding.status_code, 409, stale_binding.json)
        self.assertEqual(frame_reader.call_count, 0)
        binding_response = self.client.post("/api/human_review_binding", json={
            "frame": 0, "topic": loaded.json["topic"],
            "file_activation_generation": loaded.json["file_activation_generation"]})
        self.assertEqual(binding_response.status_code, 200, binding_response.json)
        binding = binding_response.json
        review = {"reviewer_declared_id": "image-import-test",
                  "started_at": "2026-11-17T00:00:00Z",
                  "decision_idempotency_key": "image-import-persistence-0"}
        truth_payload = {"frame": 0,
            "shapes": [shape], "review": review, "topic": binding["topic"],
            "expected_file_activation_generation": binding["file_activation_generation"],
            "expected_frame_sha256": binding["frame_sha256"],
            "expected_candidate_sha256": binding["candidate_sha256"],
            "expected_dataset_id": binding["dataset_id"],
            "expected_dataset_version": binding["dataset_version"],
            "expected_camera": binding["camera"],
            "expected_dataset_split": binding["dataset_split"],
            "expected_split_policy": binding["split_policy"],
            "expected_training_return_allowed": binding["training_return_allowed"]}
        stale_payload = copy.deepcopy(truth_payload)
        stale_payload["expected_file_activation_generation"] -= 1
        stale_payload["review"]["decision_idempotency_key"] = "image-import-stale-generation"
        with patch.object(server, "frame_image", wraps=server.frame_image) as frame_reader:
            stale = self.client.post("/api/truth", json=stale_payload)
        self.assertEqual(stale.status_code, 409, stale.json)
        self.assertEqual(frame_reader.call_count, 0)
        self.assertNotIn("0", server.state["truth"])
        saved = self.client.post("/api/truth", json=truth_payload)
        self.assertEqual(saved.status_code, 200, saved.json)
        self.assertEqual(saved.json["saved"], 1)
        other = self.upload(photograph("JPEG"), "另一个.jpg").json["file"]
        self.client.post(f"/api/files/{other['id']}/load")
        again = self.client.post(f"/api/files/{item['id']}/load").json
        self.assertTrue(again["compact_geometry"])
        restored = self.client.get("/api/frame_state/0", query_string={
            "topic": again["topic"], "path": again["path"],
            "file_activation_generation": again["file_activation_generation"]})
        self.assertEqual(restored.status_code, 200, restored.json)
        self.assertEqual(restored.json["truth"]["0"], [shape])
        persisted = json.loads((self.root / "state.json").read_text())
        self.assertEqual(persisted["truth"]["0"], [shape])

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
