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
        self.body = self.bound_body()

    def bound_body(self, candidates=None, labels=None):
        binding = server.candidate_import_binding()
        body = {
            "path": binding["path"], "topic": binding["topic"],
            "expected_file_activation_generation": binding["file_activation_generation"],
            "expected_label_state_sha256": binding["label_state_sha256"],
            "expected_truth_state_sha256": binding["truth_state_sha256"],
            "expected_candidate_state_sha256": binding["candidate_state_sha256"],
            "candidates_by_topic": candidates or {"camera": {"1": [self.shape]}},
        }
        if labels is not None:
            body["labels"] = labels
        return body

    def tearDown(self):
        self.patcher.stop()
        self.temp.cleanup()

    def test_import_is_unconfirmed_and_normalizes_ids(self):
        response = self.client.post("/api/candidates/import", json=self.body)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json["saved_as_truth"])
        self.assertEqual(self.state["truth"], {})
        self.assertEqual(self.state["candidates"]["1"][0]["id"], 1)
        self.assertEqual(response.json["applied_binding"]["candidate_state_sha256"],
                         self.body["expected_candidate_state_sha256"])
        self.assertNotEqual(response.json["candidate_import_binding"]["candidate_state_sha256"],
                            self.body["expected_candidate_state_sha256"])
        self.assertEqual(self.client.post("/api/candidates/import", json=self.body).status_code, 409)

    def test_confirmed_empty_truth_and_other_file_are_protected(self):
        self.state["truth"] = {"1": []}
        self.assertEqual(self.client.post(
            "/api/candidates/import", json=self.bound_body()).status_code, 400)
        other = self.bound_body()
        other["path"] = str(self.root / "other.mcap")
        self.assertEqual(self.client.post("/api/candidates/import", json=other).status_code, 409)
        self.assertEqual(self.state["candidates"], {})

    def test_invalid_later_frame_cannot_partially_import(self):
        invalid = copy.deepcopy(self.shape)
        invalid["points"][0][0] = 1000
        body = self.bound_body(candidates={
            "camera": {"1": [self.shape], "2": [invalid]}})
        self.assertEqual(self.client.post("/api/candidates/import", json=body).status_code, 400)
        self.assertEqual(self.state["candidates"], {})
        self.assertFalse((self.root / "state.json").exists())

    def test_disk_failure_rolls_back_memory_and_busy_task_blocks_import(self):
        with patch.object(server, "save", side_effect=RuntimeError("storage unavailable")):
            self.assertEqual(self.client.post("/api/candidates/import", json=self.body).status_code, 503)
        self.assertEqual(self.state["candidates"], {})
        server.run_status["status"] = "running"
        self.assertEqual(self.client.post("/api/candidates/import", json=self.body).status_code, 400)

    def test_old_generation_labels_truth_and_candidate_bases_are_rejected(self):
        original = copy.deepcopy(self.state)
        stale_generation = self.bound_body()
        stale_generation["expected_file_activation_generation"] -= 1
        self.assertEqual(self.client.post(
            "/api/candidates/import", json=stale_generation).status_code, 409)

        stale_labels = self.bound_body()
        self.state["labels"].append("floor")
        self.assertEqual(self.client.post(
            "/api/candidates/import", json=stale_labels).status_code, 409)
        self.state["labels"] = original["labels"]

        stale_truth = self.bound_body()
        self.state["truth"]["2"] = []
        self.assertEqual(self.client.post(
            "/api/candidates/import", json=stale_truth).status_code, 409)
        self.state["truth"] = original["truth"]

        stale_candidates = self.bound_body()
        self.state["candidates"]["2"] = [copy.deepcopy(self.shape)]
        self.assertEqual(self.client.post(
            "/api/candidates/import", json=stale_candidates).status_code, 409)
        self.assertNotIn("1", self.state["candidates"])

    def test_merge_requires_current_binding_and_preserves_unselected_candidates(self):
        self.state["labels"] = ["pallet", "floor"]
        self.state["candidates"] = {"1": [{
            "id": 1, "label": "floor", "points": [[2, 2], [30, 2], [30, 30]]}]}
        body = self.bound_body(labels=["pallet"])
        response = self.client.post("/api/candidates/merge", json=body)
        self.assertEqual(response.status_code, 200, response.json)
        self.assertTrue(response.json["preserved_other_labels"])
        self.assertFalse(response.json["saved_as_truth"])
        self.assertEqual([item["label"] for item in self.state["candidates"]["1"]],
                         ["floor", "pallet"])
        self.assertEqual(response.json["applied_binding"]["candidate_state_sha256"],
                         body["expected_candidate_state_sha256"])
        self.assertNotEqual(response.json["candidate_import_binding"]["candidate_state_sha256"],
                            body["expected_candidate_state_sha256"])
        self.assertEqual(self.client.post(
            "/api/candidates/merge", json=body).status_code, 409)


if __name__ == "__main__":
    unittest.main()
