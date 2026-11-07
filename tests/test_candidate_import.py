import copy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from PIL import Image
from source.agent import server


class CandidateImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        raw = io.BytesIO()
        Image.new("RGB", (64, 48)).save(raw, "PNG")
        self.video = SimpleNamespace(path=self.root / "capture.mcap", topics={"camera": {"count": 3}}, frame=lambda topic, index: raw.getvalue())
        self.state = {"video": self.video, "video_path": str(self.video.path), "topic": "camera", "labels": ["pallet"],
                      "truth": {}, "truths_by_topic": {}, "candidates": {}, "candidates_by_topic": {}}
        self.patcher = patch.multiple(server, WORK=self.root, state=self.state, run_status={"status": "idle"})
        self.patcher.start()
        self.client = server.app.test_client()
        self.shape = {"id": "external-id", "label": "pallet", "points": [[1, 1], [20, 1], [20, 20]]}
        self.body = {"path": str(self.video.path), "candidates_by_topic": {"camera": {"1": [self.shape]}}}

    def tearDown(self):
        self.patcher.stop()
        self.temp.cleanup()

    def test_import_is_unconfirmed_and_normalizes_ids(self):
        response = self.client.post("/api/candidates/import", json=self.body)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json["saved_as_truth"])
        self.assertEqual(self.state["truth"], {})
        self.assertEqual(self.state["candidates"]["1"][0]["id"], 1)
        self.assertEqual(self.client.post("/api/candidates/import", json=self.body).status_code, 400)

    def test_confirmed_empty_truth_and_other_file_are_protected(self):
        self.state["truth"] = {"1": []}
        self.assertEqual(self.client.post("/api/candidates/import", json=self.body).status_code, 400)
        self.body["path"] = str(self.root / "other.mcap")
        self.assertEqual(self.client.post("/api/candidates/import", json=self.body).status_code, 400)
        self.assertEqual(self.state["candidates"], {})

    def test_invalid_later_frame_cannot_partially_import(self):
        invalid = copy.deepcopy(self.shape)
        invalid["points"][0][0] = 1000
        self.body["candidates_by_topic"]["camera"]["2"] = [invalid]
        self.assertEqual(self.client.post("/api/candidates/import", json=self.body).status_code, 400)
        self.assertEqual(self.state["candidates"], {})
        self.assertFalse((self.root / "state.json").exists())

    def test_disk_failure_rolls_back_memory_and_busy_task_blocks_import(self):
        with patch.object(server, "save", side_effect=RuntimeError("storage unavailable")):
            self.assertEqual(self.client.post("/api/candidates/import", json=self.body).status_code, 503)
        self.assertEqual(self.state["candidates"], {})
        server.run_status["status"] = "running"
        self.assertEqual(self.client.post("/api/candidates/import", json=self.body).status_code, 400)


if __name__ == "__main__":
    unittest.main()
