import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import benchmark_point_group_batch as benchmark


class PointGroupRealAbToolTests(unittest.TestCase):
    def test_compares_only_selected_frames_and_normalizes_batch_items(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            images.mkdir()
            frames = []
            for index in range(3):
                payload = f"image-{index}".encode()
                name = f"frame{index}.jpg"
                (images / name).write_bytes(payload)
                frames.append({"frame": index, "file": name,
                               "sha256": hashlib.sha256(payload).hexdigest()})
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"frames": frames}), encoding="utf-8")
            evidence = root / "evidence.json"
            evidence.write_text(json.dumps({
                "reference_frame": 0,
                "selected_frames": [1, 2],
                "contract": {
                    "positive_points": [[1, 2], [3, 4]],
                    "negative_points": [[5, 6]],
                },
            }), encoding="utf-8")

            calls = []

            def fake_post(_url, endpoint, image, **payload):
                calls.append((endpoint, image, payload))
                if endpoint == "/segment_points":
                    point = payload["points"][0]
                    return {"points": [[point["x"], point["y"]]],
                            "parts": [], "score": 0.9, "warnings": []}
                items = []
                for index, group in enumerate(payload["prompt_groups"]):
                    point = group[0]
                    items.append({"index": index, "points": group, "result": {
                        "points": [[point["x"], point["y"]]],
                        "parts": [], "score": 0.9, "warnings": []}})
                return {"items": items, "diagnostics": {
                    "image_encoded_once": True}}

            with mock.patch.object(benchmark, "_post", side_effect=fake_post):
                report = benchmark.benchmark(
                    images, manifest, evidence,
                    "http://127.0.0.1:1", "http://127.0.0.1:2")

        self.assertEqual(report["sample_count"], 2)
        self.assertEqual(report["results_exactly_equal_count"], 2)
        self.assertEqual(report["success_counts_equal_count"], 2)
        self.assertEqual(report["group_count_per_frame"], 2)
        self.assertEqual(len(calls), 7)  # one warmup + 2 * (2 sequential + 1 batch)
        self.assertEqual(calls[0][1], b"image-0")
        self.assertTrue(all(row["candidate_diagnostics"]["image_encoded_once"]
                            for row in report["rows"]))

    def test_percentile_is_bounded_to_an_observed_value(self):
        self.assertEqual(benchmark._percentile([4, 1, 3, 2], .95), 3)


if __name__ == "__main__":
    unittest.main()
