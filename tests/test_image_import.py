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
        binding_response = self.client.post("/api/human_review_binding", json={
            "frame": 0, "topic": loaded.json["topic"]})
        self.assertEqual(binding_response.status_code, 200, binding_response.json)
        binding = binding_response.json
        review = {"reviewer_declared_id": "image-import-test",
                  "started_at": "2026-11-17T00:00:00Z",
                  "decision_idempotency_key": "image-import-persistence-0"}
        saved = self.client.post("/api/truth", json={"frame": 0,
            "shapes": [shape], "review": review, "topic": binding["topic"],
            "expected_frame_sha256": binding["frame_sha256"],
            "expected_candidate_sha256": binding["candidate_sha256"],
            "expected_dataset_id": binding["dataset_id"],
            "expected_dataset_version": binding["dataset_version"],
            "expected_camera": binding["camera"],
            "expected_dataset_split": binding["dataset_split"],
            "expected_split_policy": binding["split_policy"],
            "expected_training_return_allowed": binding["training_return_allowed"]})
        self.assertEqual(saved.status_code, 200, saved.json)
        self.assertEqual(saved.json["saved"], 1)
        other = self.upload(photograph("JPEG"), "另一个.jpg").json["file"]
        self.client.post(f"/api/files/{other['id']}/load")
        again = self.client.post(f"/api/files/{item['id']}/load").json
        self.assertTrue(again["compact_geometry"])
        restored = self.client.get("/api/frame_state/0", query_string={
            "topic": again["topic"], "path": again["path"]})
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
