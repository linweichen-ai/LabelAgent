import tempfile
import unittest
from pathlib import Path

from tools.verify_synthetic_signoff_browser import EXPECTED_COUNT, ONE_PIXEL_PNG, synthetic_rows


class SyntheticSignoffBrowserTests(unittest.TestCase):
    def test_public_fixture_is_deterministic_and_contains_no_real_identity(self):
        first = synthetic_rows()
        second = synthetic_rows()
        self.assertEqual(first, second)
        self.assertEqual(len(first), EXPECTED_COUNT)
        self.assertEqual(len({row["candidate_id"] for row in first}), EXPECTED_COUNT)
        self.assertTrue(all(row["candidate_id"].startswith("synthetic-") for row in first))
        self.assertTrue(all(row["frame_id"] == index for index, row in enumerate(first)))
        self.assertTrue(all(row["decision"] is None for row in first))
        self.assertTrue(ONE_PIXEL_PNG.startswith("data:image/png;base64,"))

    def test_all_bindings_are_sha256_and_human_fields_are_empty(self):
        for row in synthetic_rows():
            for field in ("source_image_sha256", "canonical_geometry_sha256", "focused_image_sha256"):
                self.assertEqual(len(row[field]), 64)
                int(row[field], 16)
            for field in ("decision", "reviewer", "reviewed_at", "note"):
                self.assertIsNone(row[field])

    def test_missing_edge_error_is_actionable(self):
        source = (Path(__file__).resolve().parents[1] / "tools" / "verify_synthetic_signoff_browser.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("Microsoft Edge could not be started for the browser check", source)
        self.assertIn("install or repair Edge", source)


if __name__ == "__main__":
    unittest.main()
