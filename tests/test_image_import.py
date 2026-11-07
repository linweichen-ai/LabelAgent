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
        self.assertEqual(self.client.post("/api/truth", json={"frame": 0, "shapes": [shape]}).json["saved"], 1)
        other = self.upload(photograph("JPEG"), "另一个.jpg").json["file"]
        self.client.post(f"/api/files/{other['id']}/load")
        again = self.client.post(f"/api/files/{item['id']}/load").json
        self.assertEqual(again["truth"]["0"], [shape])
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
