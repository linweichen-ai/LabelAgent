import io
from PIL import Image
import unittest

from tools.export_candidate_overlay import loopback_url
from source.agent.workflow import candidate_overlay


class CandidateOverlayExportTests(unittest.TestCase):
    def test_loopback_only(self):
        self.assertEqual(loopback_url("http://127.0.0.1:8767/"), "http://127.0.0.1:8767")
        for value in ("https://example.com", "http://user:pass@localhost:8767"):
            with self.assertRaises(ValueError):
                loopback_url(value)

    def test_overlay_preserves_size_and_changes_pixels(self):
        source = io.BytesIO()
        Image.new("RGB", (20, 16), "gray").save(source, "JPEG")
        shape = {"id": 1, "label": "slope", "points": [[2, 2], [12, 2], [12, 10], [2, 10]],
                 "holes": [], "score": 0.9, "source": "fixture"}
        rendered = candidate_overlay(source.getvalue(), [shape])
        image = Image.open(io.BytesIO(rendered))
        self.assertEqual(image.size, (20, 16))
        self.assertNotEqual(rendered, source.getvalue())


if __name__ == "__main__":
    unittest.main()
