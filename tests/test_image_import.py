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
    def review(reviewer="test-reviewer"):
        return {"reviewer_id": reviewer,
                "started_at": "2026-11-11T00:00:00Z", "note": "test"}

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
        response = self.client.post("/api/truth", json={"frame": 0,
            "shapes": [shape], "review": self.review()})
        self.assertEqual(response.json["saved"], 1)
        self.assertEqual(response.json["review"]["decision"], "manual_annotation")
        audit = self.client.get("/api/human_reviews?frame=0").json["reviews"]
        self.assertEqual(len(audit), 1)
        self.assertEqual(audit[0]["reviewer_id"], "test-reviewer")
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
        missing = self.client.post("/api/truth", json={"frame": 0, "shapes": [shape]})
        self.assertEqual(missing.status_code, 400)
        rejected = self.client.post("/api/human_review", json={"frame": 0,
            "review": self.review("rejector")})
        self.assertEqual(rejected.status_code, 200)
        self.assertEqual(rejected.json["review"]["decision"], "reject_candidate")
        self.assertNotIn("0", server.state["truth"])
        self.assertIn("0", server.state["candidates"])
        accepted = self.client.post("/api/truth", json={"frame": 0,
            "shapes": [shape], "review": self.review("acceptor")})
        self.assertEqual(accepted.json["review"]["decision"], "accept_candidate")
        self.assertIn("0", server.state["truth"])
        self.assertNotIn("0", server.state["candidates"])
        audits = self.client.get("/api/human_reviews?frame=0").json["reviews"]
        self.assertEqual([item["decision"] for item in audits],
                         ["reject_candidate", "accept_candidate"])

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
